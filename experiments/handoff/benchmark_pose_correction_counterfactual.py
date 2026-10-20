"""Pose-correction counterfactual benchmark v1.

Paired exact-snapshot comparison: short-prefix HANDOFF vs verified
contact-preserving pivoting (and related branches). Diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import AIRBORNE_V3, map_action
from envs.ballistic_impact import z_tgt_of
from envs.contacts.frames import extract_contacts
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor
from envs.outcomes.horizon_outcome import recovery_failed, terminal_reason
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, object_vz, observe
from envs.simulation.capture import restore_capture
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.grasp_configuration import measure
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.recovery_episode import make_training_env
from training.recovery_execution import _escaped
from training.unified_k48 import HANDOFF

OUT = ROOT / "results" / "benchmarks" / "pose_correction_counterfactual_v1"
WRIST = ROOT / "results" / "diagnostics" / "wrist_tilt_gravity_pivoting"
PILOT = ROOT / "results" / "diagnostics" / "nontrivial_recovery_ic_mining"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
SEED = 20261018

U_TGT = np.array([-0.013840181460089382, -0.0030378179260123783, -0.9998996054801703], float)
P_TGT = np.array([-0.001115724743759791, -6.904231962095219e-06, 0.10474686676582946], float)

# Verified pivoting sequence from wrist_tilt pilot
PIVOT_SEQ = [6] * 20 + [12] * 15 + [0] * 8
PIVOT_IC = "hrv1_920031"

THETA_THRESH_DEG = [5.0, 10.0, 15.0, 20.5]
POS_THRESH_MM = [5.0, 10.0, 17.0]
FINAL_WINDOW_S = 2.0  # last 2s of 12s nominal — post-lift stable hold
HOLD_S_FOR_NEW = 0.5  # continuous satisfaction required inside window for SUCCESS_NEW

EXPAND_ICS = [
    "hrv1_920031",
    "hard_910101",
    "hrv1_920131",
    "hrv1_920099",
    "hard_910172",
    "hrv1_920151",
]

SHORT_KNOWN = {
    "hrv1_920031": [30],
    "hard_910101": [30],
    "hrv1_920131": [41],
    "hrv1_920099": [41],
    "hard_910172": [44],
    "hrv1_920151": [30],
}


def jsafe(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def load_snaps():
    m = {}
    for r in csv.DictReader((PILOT / "IC_MANIFEST.csv").open(encoding="utf-8")):
        m[r["id"]] = r["snapshot"]
    return m


def contact_summary(sim):
    rows, _ = extract_contacts(sim)
    nL = sum(1 for r in rows if r["side"] == "L")
    nR = sum(1 for r in rows if r["side"] == "R")
    return {"nL": nL, "nR": nR, "bilateral": bool(nL > 0 and nR > 0),
            "any": bool(nL + nR > 0)}


def step_legal_sublogged(sim, action, log_rows, token):
    action = np.clip(np.asarray(action, np.float32).reshape(7), -1.0, 1.0)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(action, rd, rh, AIRBORNE_V3)
    vel, ang, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.v_cmd = vel.copy()
    sim.fsm.w_cmd = ang.copy()
    sim.fsm.v_des = vel.copy()
    sim.fsm.w_des = ang.copy()
    sim.fsm.fg_cmd = float(-tau)
    dt = float(sim.model.opt.timestep)
    for s in range(N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, dt)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
        )
        vz_before = object_vz(sim)
        mujoco.mj_step(sim.model, sim.data)
        observe(sim, vz_before)
        m = measure(sim, U_TGT, P_TGT)
        cs = contact_summary(sim)
        log_rows.append({
            "t": float(sim.data.time), "token": int(token), "sub": s,
            "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
            "nL": cs["nL"], "nR": cs["nR"], "bilateral": cs["bilateral"],
            "w_rel_h": m["w_rel_h"], "obj_z": m["obj_z"],
        })
    return action


def topology(sub_log):
    if not sub_log:
        return {"substep": False, "frac_bilateral": None, "max_zero_ms": None,
                "sustained_release": None}
    dt = float(sub_log[1]["t"] - sub_log[0]["t"]) if len(sub_log) > 1 else 0.002
    max_zero = z = 0.0
    n_bi = n_z = 0
    for row in sub_log:
        n = int(row["nL"] > 0) + int(row["nR"] > 0)
        if n == 2:
            n_bi += 1
            z = 0.0
        elif n == 0:
            n_z += 1
            z += dt
            max_zero = max(max_zero, z)
        else:
            z = 0.0
    n = len(sub_log)
    return {
        "substep": True,
        "frac_bilateral": n_bi / n,
        "frac_zero": n_z / n,
        "max_zero_ms": max_zero * 1e3,
        "sustained_release": bool(max_zero * 1e3 > 40.0),
        "n_sub": n,
    }


def classify_mechanism(topo, d_theta_mid, tokens, book):
    if not topo.get("substep"):
        return "AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE"
    if topo.get("sustained_release") and d_theta_mid >= 3:
        return "RELEASE_RECATCH"
    if d_theta_mid >= 5 and topo.get("frac_bilateral", 0) >= 0.85 and not topo.get("sustained_release"):
        return "BILATERAL_CONTACT_PRESERVING_PIVOT"
    if d_theta_mid >= 5 and topo.get("frac_bilateral", 0) < 0.85 and not topo.get("sustained_release"):
        return "SINGLE_OR_PARTIAL_CONTACT_PIVOT"
    if abs(d_theta_mid) < 2:
        return "NO_MEANINGFUL_RELATIVE_ROTATION"
    return "AMBIGUOUS_CONTACT_TRANSITION"


def final_window_stats(samples, window_start_offset, hold_s=HOLD_S_FOR_NEW):
    """samples: list of dicts with t_rel (from handoff), e_theta_deg, e_pos_mm, bilateral."""
    win = [s for s in samples if s["t_rel"] + 1e-9 >= window_start_offset]
    if not win:
        return {
            "n": 0, "mean_theta": None, "max_theta": None,
            "mean_pos": None, "max_pos": None,
            "frac_bilateral": None, "matrix": {},
        }
    thetas = np.array([s["e_theta_deg"] for s in win], float)
    poss = np.array([s["e_pos_mm"] for s in win], float)
    bil = np.array([1.0 if s["bilateral"] else 0.0 for s in win], float)
    # estimate dt
    if len(win) > 1:
        dts = np.diff([s["t_rel"] for s in win])
        dt = float(np.median(dts)) if len(dts) else 0.002
    else:
        dt = 0.002
    matrix = {}
    for th, pm in product(THETA_THRESH_DEG, POS_THRESH_MM):
        ok = (thetas <= th) & (poss <= pm)
        frac = float(ok.mean())
        # longest continuous ok interval
        longest = 0.0
        cur = 0.0
        for v in ok:
            if v:
                cur += dt
                longest = max(longest, cur)
            else:
                cur = 0.0
        key = f"th{th}_pos{pm}"
        matrix[key] = {
            "frac_frames": frac,
            "longest_ok_s": longest,
            "SUCCESS_NEW": bool(longest + 1e-9 >= hold_s and frac > 0),
        }
    return {
        "n": len(win),
        "mean_theta": float(thetas.mean()),
        "max_theta": float(thetas.max()),
        "mean_pos": float(poss.mean()),
        "max_pos": float(poss.max()),
        "frac_bilateral": float(bil.mean()),
        "matrix": matrix,
        "window_s": float(win[-1]["t_rel"] - win[0]["t_rel"]) if len(win) > 1 else 0.0,
    }


def run_branch(env, book, blob, tokens, log_sub=True, sample_nominal_dt=0.02):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    m0 = measure(env.sim, U_TGT, P_TGT)
    sub = []
    lost_low = 0
    for tok in tokens:
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        if log_sub:
            step_legal_sublogged(env.sim, action, sub, int(tok))
        else:
            step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            topo = topology(sub)
            return {
                "tokens": list(tokens),
                "success_12s": False,
                "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                "m0": m0,
                "m_handoff": measure(env.sim, U_TGT, P_TGT),
                "m_final": measure(env.sim, U_TGT, P_TGT),
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "dropped_before_handoff": True,
                "sub": sub,
                "topo": topo,
                "window": final_window_stats([], NOMINAL_S - FINAL_WINDOW_S),
                "nominal_samples": [],
                "mechanism": classify_mechanism(topo, 0.0, tokens, book),
            }

    m_hand = measure(env.sim, U_TGT, P_TGT)
    recovery_elapsed = float(env.sim.data.time) - t0
    d_mid = float(m0["e_theta_deg"] - m_hand["e_theta_deg"])
    topo = topology(sub)
    mech = classify_mechanism(topo, d_mid, tokens, book)

    # Nominal with dense sampling
    apply_handoff(env.sim, blob["ref"])
    t_h = float(env.sim.data.time)
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    samples = []
    next_sample = 0.0
    if latched(env.sim):
        reason = "TABLE_DROP"
    while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        t_prev = float(env.sim.data.time)
        nominal_step(env.sim)
        t_rel = float(env.sim.data.time) - t_h
        if t_rel + 1e-12 >= next_sample:
            m = measure(env.sim, U_TGT, P_TGT)
            cs = contact_summary(env.sim)
            samples.append({
                "t_rel": t_rel,
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "bilateral": cs["bilateral"],
                "obj_z": m["obj_z"],
            })
            next_sample += sample_nominal_dt
        row = _monitor(env.sim)
        end_row = row
        target = float(z_tgt_of(env.sim))
        bilateral = _bilateral(row)
        if bilateral and float(env.sim.fsm.p_des[2]) >= target - 1e-9:
            lift_done = True
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        if _dropped(row) or _escaped(env.sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not bilateral else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
    if reason is None:
        if end_row is None or not _bilateral(end_row) or _dropped(end_row) or _escaped(env.sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    m_final = measure(env.sim, U_TGT, P_TGT)
    # Always sample end frame
    if not samples or samples[-1]["t_rel"] < float(env.sim.data.time) - t_h - 1e-6:
        samples.append({
            "t_rel": float(env.sim.data.time) - t_h,
            "e_theta_deg": m_final["e_theta_deg"],
            "e_pos_mm": m_final["e_pos_mm"],
            "bilateral": contact_summary(env.sim)["bilateral"],
            "obj_z": m_final["obj_z"],
        })
    win = final_window_stats(samples, float(NOMINAL_S) - FINAL_WINDOW_S)
    return {
        "tokens": list(tokens),
        "success_12s": reason == "SUCCESS_12S",
        "reason": reason,
        "m0": m0,
        "m_handoff": m_hand,
        "m_final": m_final,
        "recovery_elapsed": recovery_elapsed,
        "dropped_before_handoff": False,
        "sub": sub,
        "topo": topo,
        "window": win,
        "nominal_samples": samples,
        "mechanism": mech,
        "d_theta_recovery": d_mid,
        "d_theta_final": float(m0["e_theta_deg"] - m_final["e_theta_deg"]),
        "d_pos_final_mm": float(m0["e_pos_mm"] - m_final["e_pos_mm"]),
    }


def bounded_short_search(env, book, blob, rng, max_n=40):
    """Bounded short-prefix search (not only open tokens)."""
    openish = [i for i in range(47) if float(book[i, 6]) < -0.5]
    closeish = [i for i in range(47) if float(book[i, 6]) > 0.5]
    mid = [i for i in range(47) if abs(float(book[i, 6]) + 0.48) < 0.05]
    cands = []
    cands.append([])
    for t in [30, 44, 41, 3, 6, 0, 12, 36]:
        cands.append([t])
        cands.append([t, t])
    for t in openish[::4][:6]:
        cands.append([t])
    for t in closeish[::5][:4]:
        cands.append([t])
    for t in mid:
        cands.append([t])
        cands.append([t, t])
    for _ in range(15):
        a, b = int(rng.integers(0, 47)), int(rng.integers(0, 47))
        cands.append([a, b])
    # dedupe
    seen, uniq = set(), []
    for c in cands:
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    rng.shuffle(uniq)
    best = None
    results = []
    for seq in uniq[:max_n]:
        res = run_branch(env, book, blob, seq, log_sub=False)
        results.append(res)
        # rank by SUCCESS_12S then lowest mean window theta
        score = None
        if res["success_12s"] and res["window"]["mean_theta"] is not None:
            score = -res["window"]["mean_theta"]
        if score is not None and (best is None or score > best[0]):
            best = (score, seq, res)
    return best, results


def render_pair(env, book, blob, tokens_a, tokens_b, dest, labels):
    from visualization.camera import mjv_camera_from_preset
    from visualization.overlays import caption
    from visualization.video_recorder import write_frames_mp4

    def shoot(tokens, label):
        restore_capture(env, blob["capture"])
        cam = mjv_camera_from_preset({
            "lookat": np.array(env.sim.data.xpos[env.sim.ids.object_body], float),
            "distance": 0.55, "azimuth": 132.0, "elevation": -22.0,
        })
        renderer = mujoco.Renderer(env.sim.model, 360, 480)
        frames = []
        t0 = float(env.sim.data.time)
        next_frame = t0
        lost_low = 0

        def grab(text):
            cam.lookat[:] = np.array(env.sim.data.xpos[env.sim.ids.object_body], float)
            renderer.update_scene(env.sim.data, cam)
            frames.append(caption(np.asarray(renderer.render(), np.uint8).copy(), text))

        for tok in tokens:
            now = float(env.sim.data.time)
            if now + 1e-9 >= next_frame:
                m = measure(env.sim, U_TGT, P_TGT)
                cs = contact_summary(env.sim)
                grab(f"{label} t={now-t0:.2f} tok={tok} n={cs['nL']}/{cs['nR']} eθ={m['e_theta_deg']:.1f}")
                next_frame += 0.04
            _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
            step_legal(env.sim, action)
            env._prev = np.asarray(action, np.float32)
            failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
            if failed:
                grab(f"FAIL")
                break
        else:
            apply_handoff(env.sim, blob["ref"])
            t_h = float(env.sim.data.time)
            next_frame = t_h
            while float(env.sim.data.time) - t_h < 4.0 - 1e-9:
                nominal_step(env.sim)
                now = float(env.sim.data.time)
                if now + 1e-9 >= next_frame:
                    m = measure(env.sim, U_TGT, P_TGT)
                    grab(f"{label} nom {now-t_h:.2f} eθ={m['e_theta_deg']:.1f}")
                    next_frame += 0.04
                if latched(env.sim):
                    break
        renderer.close()
        return frames

    fa = shoot(tokens_a, labels[0])
    fb = shoot(tokens_b, labels[1])
    # side-by-side pad to equal length
    n = max(len(fa), len(fb))
    if not fa or not fb:
        write_frames_mp4(fa or fb, dest, fps=25)
        return
    while len(fa) < n:
        fa.append(fa[-1])
    while len(fb) < n:
        fb.append(fb[-1])
    paired = []
    for a, b in zip(fa, fb):
        paired.append(np.concatenate([a, b], axis=1))
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_frames_mp4(paired, dest, fps=25)


def flatten_result(ic, branch, res):
    win = res["window"]
    row = {
        "ic_id": ic,
        "branch": branch,
        "tokens": json.dumps(res["tokens"]),
        "n_tokens": len(res["tokens"]),
        "recovery_elapsed_s": res["recovery_elapsed"],
        "success_12s": int(res["success_12s"]),
        "reason": res["reason"],
        "mechanism": res["mechanism"],
        "e_theta0_deg": res["m0"]["e_theta_deg"],
        "e_pos0_mm": res["m0"]["e_pos_mm"],
        "e_theta_handoff_deg": res["m_handoff"]["e_theta_deg"],
        "e_pos_handoff_mm": res["m_handoff"]["e_pos_mm"],
        "e_theta_final_deg": res["m_final"]["e_theta_deg"],
        "e_pos_final_mm": res["m_final"]["e_pos_mm"],
        "d_theta_recovery_deg": res.get("d_theta_recovery"),
        "d_theta_final_deg": res.get("d_theta_final"),
        "d_pos_final_mm": res.get("d_pos_final_mm"),
        "win_mean_theta": win.get("mean_theta"),
        "win_max_theta": win.get("max_theta"),
        "win_mean_pos": win.get("mean_pos"),
        "win_max_pos": win.get("max_pos"),
        "win_frac_bilateral": win.get("frac_bilateral"),
        "frac_bilateral_recovery": res["topo"].get("frac_bilateral"),
        "max_zero_ms": res["topo"].get("max_zero_ms"),
        "sustained_release": res["topo"].get("sustained_release"),
    }
    for th, pm in product(THETA_THRESH_DEG, POS_THRESH_MM):
        key = f"th{th}_pos{pm}"
        cell = win.get("matrix", {}).get(key, {})
        row[f"NEW_{key}"] = int(bool(cell.get("SUCCESS_NEW")))
        row[f"frac_{key}"] = cell.get("frac_frames")
        row[f"longest_{key}"] = cell.get("longest_ok_s")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--short-search", type=int, default=35)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--primary-only", action="store_true",
                        help="Only evaluate hrv1_920031")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RAW").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    snaps = load_snaps()

    write_json(OUT / "CONFIGURATION_METRICS_SPEC.json", {
        "u_target_h": U_TGT.tolist(),
        "p_target_h": P_TGT.tolist(),
        "theta_thresh_deg": THETA_THRESH_DEG,
        "pos_thresh_mm": POS_THRESH_MM,
        "nominal_horizon_s": NOMINAL_S,
        "final_window_s": FINAL_WINDOW_S,
        "final_window_def": f"last {FINAL_WINDOW_S}s of unchanged {NOMINAL_S}s nominal continuation",
        "SUCCESS_NEW_rule": f"longest continuous interval in final window satisfying both thresholds >= {HOLD_S_FOR_NEW}s",
        "provisional_loose_ref": "20.5deg / 17mm from migration SUCCESS_12S distribution — not task freeze",
        "pivot_seq": PIVOT_SEQ,
        "axis_reversal_symmetric": True,
    })

    ics = [PIVOT_IC] if args.primary_only else EXPAND_ICS
    paired_rows = []
    threshold_rows = []
    primary_advantage = None

    for ic in ics:
        snap = snaps.get(ic)
        if not snap:
            print("SKIP", ic, flush=True)
            continue
        blob = pickle.loads((ROOT / snap).read_bytes())
        if "capture" not in blob:
            continue
        print("IC", ic, flush=True)

        branches = {}

        # A immediate
        branches["A_immediate"] = run_branch(env, book, blob, [], log_sub=True)
        # B known shorts
        known = SHORT_KNOWN.get(ic, [30])
        branches["B_short_known"] = run_branch(env, book, blob, known, log_sub=True)
        for name, seq in [("B_tok30", [30]), ("B_tok44", [44]), ("B_tok41", [41]),
                          ("B_tok30_30", [30, 30]), ("B_tok44_44", [44, 44])]:
            branches[name] = run_branch(env, book, blob, seq, log_sub=False)

        # C bounded short search — light log
        best_short, short_all = bounded_short_search(
            env, book, blob, np.random.default_rng(args.seed + hash(ic) % 10000),
            max_n=args.short_search,
        )
        if best_short is not None:
            # re-run best with substeps
            branches["C_best_short_search"] = run_branch(
                env, book, blob, best_short[1], log_sub=True
            )
        else:
            branches["C_best_short_search"] = branches["B_short_known"]

        # D verified / template pivot sequence
        branches["D_pivot_seq"] = run_branch(env, book, blob, PIVOT_SEQ, log_sub=True)

        # E prefixes of pivot
        for k in [0, 5, 10, 20, 25, 30, 35, len(PIVOT_SEQ)]:
            if k > len(PIVOT_SEQ):
                continue
            pref = PIVOT_SEQ[:k]
            # skip duplicate of A / D
            if k == 0:
                continue
            if k == len(PIVOT_SEQ):
                continue
            branches[f"E_prefix_{k}"] = run_branch(env, book, blob, pref, log_sub=(k in (20, 35)))

        # F release-recatch style alternative
        branches["F_release_recatch"] = run_branch(
            env, book, blob, [39, 39, 12, 12, 0, 0], log_sub=True
        )

        # Persist
        write_json(OUT / "RAW" / f"{ic}_branches.json", {
            name: {
                "tokens": r["tokens"],
                "success_12s": r["success_12s"],
                "reason": r["reason"],
                "m0": r["m0"],
                "m_handoff": r["m_handoff"],
                "m_final": r["m_final"],
                "recovery_elapsed": r["recovery_elapsed"],
                "window": r["window"],
                "topo": r["topo"],
                "mechanism": r["mechanism"],
                "d_theta_recovery": r.get("d_theta_recovery"),
                "d_theta_final": r.get("d_theta_final"),
            }
            for name, r in branches.items()
        })
        # full sublog for D
        write_json(OUT / "RAW" / f"{ic}_D_pivot_sublog.json", {
            "tokens": PIVOT_SEQ,
            "sub": branches["D_pivot_seq"]["sub"],
            "topo": branches["D_pivot_seq"]["topo"],
            "nominal_samples": branches["D_pivot_seq"]["nominal_samples"],
        })
        write_json(OUT / "RAW" / f"{ic}_B_short_sublog.json", {
            "tokens": branches["B_short_known"]["tokens"],
            "sub": branches["B_short_known"]["sub"],
            "topo": branches["B_short_known"]["topo"],
            "nominal_samples": branches["B_short_known"]["nominal_samples"],
        })

        for name, res in branches.items():
            paired_rows.append(flatten_result(ic, name, res))
            win = res["window"]
            for th, pm in product(THETA_THRESH_DEG, POS_THRESH_MM):
                key = f"th{th}_pos{pm}"
                cell = win.get("matrix", {}).get(key, {})
                threshold_rows.append({
                    "ic_id": ic,
                    "branch": name,
                    "theta_thresh_deg": th,
                    "pos_thresh_mm": pm,
                    "success_12s": int(res["success_12s"]),
                    "SUCCESS_NEW": int(bool(cell.get("SUCCESS_NEW"))),
                    "frac_frames_ok": cell.get("frac_frames"),
                    "longest_ok_s": cell.get("longest_ok_s"),
                    "win_mean_theta": win.get("mean_theta"),
                    "win_mean_pos": win.get("mean_pos"),
                    "mechanism": res["mechanism"],
                })

        # Primary paired comparison
        short_key = "C_best_short_search"
        # Prefer best among B_short_known and C by window mean theta among S12
        candidates = []
        for k in ("B_short_known", "C_best_short_search", "B_tok30", "B_tok44", "B_tok41", "B_tok30_30"):
            r = branches.get(k)
            if r and r["success_12s"] and r["window"]["mean_theta"] is not None:
                candidates.append((r["window"]["mean_theta"], k, r))
        if candidates:
            candidates.sort()
            best_short_name, best_short_res = candidates[0][1], candidates[0][2]
        else:
            best_short_name, best_short_res = "B_short_known", branches["B_short_known"]

        piv = branches["D_pivot_seq"]
        # Advantage: both S12, pivot better mean theta by >=2deg OR SUCCESS_NEW at stricter thresh
        adv = {
            "ic_id": ic,
            "short_branch": best_short_name,
            "short_s12": best_short_res["success_12s"],
            "short_win_mean_theta": best_short_res["window"]["mean_theta"],
            "short_win_mean_pos": best_short_res["window"]["mean_pos"],
            "short_e_handoff": best_short_res["m_handoff"]["e_theta_deg"],
            "short_e_final": best_short_res["m_final"]["e_theta_deg"],
            "short_recovery_s": best_short_res["recovery_elapsed"],
            "pivot_s12": piv["success_12s"],
            "pivot_win_mean_theta": piv["window"]["mean_theta"],
            "pivot_win_mean_pos": piv["window"]["mean_pos"],
            "pivot_e_handoff": piv["m_handoff"]["e_theta_deg"],
            "pivot_e_final": piv["m_final"]["e_theta_deg"],
            "pivot_recovery_s": piv["recovery_elapsed"],
            "pivot_mechanism": piv["mechanism"],
            "theta_improve_vs_short": None if (
                piv["window"]["mean_theta"] is None or best_short_res["window"]["mean_theta"] is None
            ) else float(best_short_res["window"]["mean_theta"] - piv["window"]["mean_theta"]),
            "persist": None if piv["m_handoff"] is None else abs(
                piv["m_handoff"]["e_theta_deg"] - piv["m_final"]["e_theta_deg"]
            ) < 2.0,
        }
        # SUCCESS_NEW matrix comparison at 10deg/10mm
        for th, pm in ((5, 5), (10, 10), (15, 10), (20.5, 17)):
            key = f"th{th}_pos{pm}"
            adv[f"short_NEW_{key}"] = int(bool(
                best_short_res["window"].get("matrix", {}).get(key, {}).get("SUCCESS_NEW")))
            adv[f"pivot_NEW_{key}"] = int(bool(
                piv["window"].get("matrix", {}).get(key, {}).get("SUCCESS_NEW")))
        write_json(OUT / "RAW" / f"{ic}_paired_summary.json", adv)
        if ic == PIVOT_IC:
            primary_advantage = adv
            print("PRIMARY", json.dumps(jsafe(adv), indent=2), flush=True)

        print(
            f"DONE {ic} short={best_short_name} θw={best_short_res['window']['mean_theta']} "
            f"pivot_θw={piv['window']['mean_theta']} mech={piv['mechanism']}",
            flush=True,
        )

        # Videos for primary
        if ic == PIVOT_IC and not args.skip_videos:
            try:
                render_pair(
                    env, book, blob,
                    best_short_res["tokens"], PIVOT_SEQ,
                    OUT / "videos" / f"{ic}_short_vs_pivot.mp4",
                    ["SHORT", "PIVOT"],
                )
            except Exception as exc:
                (OUT / "videos" / "VIDEO_ERROR.txt").write_text(str(exc), encoding="utf-8")

        # Stop expand early if primary shows no advantage? Task says diagnose before expand.
        # We still run expand ICs as planned unless primary_only.
        if args.primary_only:
            break

    write_csv(OUT / "PAIRED_RESULTS.csv", paired_rows)
    write_csv(OUT / "THRESHOLD_SENSITIVITY.csv", threshold_rows)
    write_json(OUT / "primary_advantage.json", primary_advantage)
    write_json(OUT / "summary.json", {
        "seed": args.seed,
        "n_ics": len({r["ic_id"] for r in paired_rows}),
        "n_rows": len(paired_rows),
        "final_window_s": FINAL_WINDOW_S,
        "pivot_seq": PIVOT_SEQ,
        "primary": primary_advantage,
    })
    print("BENCHMARK_DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
