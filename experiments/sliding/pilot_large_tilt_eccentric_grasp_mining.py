"""Bounded large-tilt eccentric grasp IC mining pilot.

Sources: (1) existing ballistic/hard snapshots, (2) legal mid-unload+wrist
captures from recoverable bases. No teleport. No training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

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
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.grasp_configuration import measure
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.recovery_episode import make_training_env
from training.recovery_execution import _escaped
from training.unified_k48 import HANDOFF

OUT = ROOT / "results" / "diagnostics" / "large_tilt_eccentric_grasp_mining"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
SEED = 20261018

U_TGT = np.array([-0.013840181460089382, -0.0030378179260123783, -0.9998996054801703], float)
P_TGT = np.array([-0.001115724743759791, -6.904231962095219e-06, 0.10474686676582946], float)

THETA_BINS = [(15, 25), (25, 40), (40, 180)]
FINAL_WINDOW_S = 2.0
SNAP_DIRS = [
    ROOT / "results/benchmarks/hard_recovery_v1/snapshots",
    ROOT / "results/diagnostics/hard_ic_mining/snapshots",
]


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


def contact_summary(sim):
    rows, _ = extract_contacts(sim)
    nL = sum(1 for r in rows if r["side"] == "L")
    nR = sum(1 for r in rows if r["side"] == "R")
    return {"nL": nL, "nR": nR, "bilateral": bool(nL > 0 and nR > 0), "any": bool(nL + nR > 0)}


def gravity_align(sim):
    mass = float(sim.model.body_mass[sim.ids.object_body])
    g_w = np.array(sim.model.opt.gravity, float)
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    rows, _ = extract_contacts(sim)
    pivot = np.mean([np.asarray(r["pos_w"], float) for r in rows], axis=0) if rows else ph + Rh @ np.array([0, 0, 0.105])
    tau_g_h = Rh.T @ np.cross(po - pivot, mass * g_w)
    from evaluation.grasp_configuration import cylinder_axis_hand
    u_c = cylinder_axis_hand(sim)
    u_t = U_TGT / max(float(np.linalg.norm(U_TGT)), 1e-12)
    u_c = u_c / max(float(np.linalg.norm(u_c)), 1e-12)
    if float(np.dot(u_c, u_t)) < 0:
        u_t = -u_t
    axis = np.cross(u_c, u_t)
    n = float(np.linalg.norm(axis))
    align = float(np.dot(tau_g_h, axis / n)) if n > 1e-9 else 0.0
    p_c_h = Rh.T @ (po - ph)
    return {
        "align": align,
        "helps": bool(align > 1e-6),
        "tau_g_norm": float(np.linalg.norm(tau_g_h)),
        "lever_m": float(np.linalg.norm(po - pivot)),
        "ecc_xy_m": float(np.linalg.norm(p_c_h[:2])),
    }


def fingerprint(m, g):
    vec = np.array([
        round(m["e_theta_deg"], 1),
        round(m["e_pos_mm"], 1),
        round(g["ecc_xy_m"] * 1e3, 1),
        round(m["p_c_h"][0] * 1e3, 1),
        round(m["p_c_h"][2] * 1e3, 1),
        round(m["u_c_h"][0], 2),
        round(m["u_c_h"][2], 2),
    ], float)
    return hashlib.md5(vec.tobytes()).hexdigest()[:12]


def bin_of(e):
    if 15 <= e < 25:
        return "15-25"
    if 25 <= e < 40:
        return "25-40"
    if e >= 40:
        return ">40"
    return "<15"


def run_nominal_window(env, blob):
    apply_handoff(env.sim, blob["ref"])
    t_h = float(env.sim.data.time)
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    samples = []
    next_s = max(0.0, float(NOMINAL_S) - FINAL_WINDOW_S)
    if latched(env.sim):
        reason = "TABLE_DROP"
    while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(env.sim)
        t_rel = float(env.sim.data.time) - t_h
        if t_rel + 1e-9 >= next_s:
            m = measure(env.sim, U_TGT, P_TGT)
            cs = contact_summary(env.sim)
            samples.append({"e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"], "bilateral": cs["bilateral"]})
            next_s += 0.02
        row = _monitor(env.sim)
        end_row = row
        if _bilateral(row) and float(env.sim.fsm.p_des[2]) >= float(z_tgt_of(env.sim)) - 1e-9:
            lift_done = True
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        if _dropped(row) or _escaped(env.sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not _bilateral(row) else 0
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
    if samples:
        mean_th = float(np.mean([s["e_theta_deg"] for s in samples]))
        mean_pos = float(np.mean([s["e_pos_mm"] for s in samples]))
    else:
        mean_th = m_final["e_theta_deg"]
        mean_pos = m_final["e_pos_mm"]
    return {
        "success_12s": reason == "SUCCESS_12S",
        "reason": reason,
        "m_final": m_final,
        "win_mean_theta": mean_th,
        "win_mean_pos": mean_pos,
    }


def run_prefix(env, book, blob, tokens, log_sub=False):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    m0 = measure(env.sim, U_TGT, P_TGT)
    lost_low = 0
    sub = []
    for tok in tokens:
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        if log_sub:
            # lightweight: one contact sample per token after step
            step_legal(env.sim, action)
            cs = contact_summary(env.sim)
            m = measure(env.sim, U_TGT, P_TGT)
            sub.append({"nL": cs["nL"], "nR": cs["nR"], "e_theta_deg": m["e_theta_deg"]})
        else:
            step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "tokens": list(tokens),
                "success_12s": False,
                "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                "m0": m0,
                "m_handoff": measure(env.sim, U_TGT, P_TGT),
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "win_mean_theta": None,
                "win_mean_pos": None,
                "sub": sub,
            }
    m_h = measure(env.sim, U_TGT, P_TGT)
    rec = float(env.sim.data.time) - t0
    nom = run_nominal_window(env, blob)
    # topology approx
    if sub:
        zero = sum(1 for s in sub if s["nL"] + s["nR"] == 0)
        bi = sum(1 for s in sub if s["nL"] > 0 and s["nR"] > 0) / len(sub)
    else:
        zero, bi = 0, None
    return {
        "tokens": list(tokens),
        "success_12s": nom["success_12s"],
        "reason": nom["reason"],
        "m0": m0,
        "m_handoff": m_h,
        "m_final": nom["m_final"],
        "recovery_elapsed": rec,
        "win_mean_theta": nom["win_mean_theta"],
        "win_mean_pos": nom["win_mean_pos"],
        "sub": sub,
        "frac_bilateral_tokens": bi,
        "zero_token_steps": zero,
    }


def classify_mech(res, d_mid):
    zero = res.get("zero_token_steps") or 0
    bi = res.get("frac_bilateral_tokens")
    if zero >= 2 and d_mid >= 3 and res["success_12s"]:
        return "RELEASE_RECATCH"
    if bi is not None and bi >= 0.85 and d_mid >= 5 and res["success_12s"] and zero == 0:
        return "CONTACT_PRESERVING_PIVOT"
    if d_mid >= 5 and res["success_12s"] and zero <= 1:
        return "LIKELY_PIVOT_OR_PARTIAL"
    if abs(d_mid) < 2:
        return "NO_MEANINGFUL_RELATIVE_ROTATION"
    if not res["success_12s"] and d_mid >= 3:
        return "ORIENTATION_CHANGE_UNSTABLE"
    return "AMBIGUOUS"


def short_search(env, book, blob, rng, max_n=30):
    cands = [[], [30], [44], [41], [30, 30], [44, 44], [3], [6], [12], [36], [0]]
    openish = [i for i in range(47) if float(book[i, 6]) < -0.5]
    closeish = [i for i in range(47) if float(book[i, 6]) > 0.5]
    for t in openish[::5][:5]:
        cands.append([t])
        cands.append([t, t])
    for t in closeish[::6][:4]:
        cands.append([t])
    for _ in range(12):
        cands.append([int(rng.integers(0, 47)), int(rng.integers(0, 47))])
    seen, uniq = set(), []
    for c in cands:
        k = tuple(c)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    rng.shuffle(uniq)
    best = None
    rows = []
    for seq in uniq[:max_n]:
        res = run_prefix(env, book, blob, seq, log_sub=False)
        rows.append(res)
        if res["success_12s"] and res["win_mean_theta"] is not None:
            # prefer lower window theta among successes
            score = -res["win_mean_theta"]
            if best is None or score > best[0]:
                best = (score, seq, res)
    return best, rows


def active_search(env, book, blob, rng, max_n=40):
    """Bounded structured tilt→mid/open→brake + release variants."""
    seqs = []
    for tilt in (6, 4, 10):
        for n_t in (5, 12, 20):
            for unload in (12, 36):
                for n_u in (5, 10, 15, 25):
                    for brake in (0, 14):
                        for n_b in (3, 8):
                            seqs.append([tilt] * n_t + [unload] * n_u + [brake] * n_b)
    # release-recatch proposals
    for o in (39, 36, 30):
        for n_o in (2, 3, 5):
            for c in (0, 14):
                seqs.append([o] * n_o + [12, 12] + [c] * 3)
                seqs.append([o] * n_o + [c] * 5)
    rng.shuffle(seqs)
    best_pivot = None
    best_recatch = None
    best_any = None
    out_rows = []
    for seq in seqs[:max_n]:
        res = run_prefix(env, book, blob, seq, log_sub=True)
        d_mid = float(res["m0"]["e_theta_deg"] - res["m_handoff"]["e_theta_deg"])
        mech = classify_mech(res, d_mid)
        row = {
            "tokens": seq,
            "success_12s": res["success_12s"],
            "reason": res["reason"],
            "e0": res["m0"]["e_theta_deg"],
            "e_hand": res["m_handoff"]["e_theta_deg"],
            "win_theta": res["win_mean_theta"],
            "win_pos": res["win_mean_pos"],
            "d_mid": d_mid,
            "mech": mech,
            "rec_s": res["recovery_elapsed"],
            "frac_bi": res.get("frac_bilateral_tokens"),
            "zero_steps": res.get("zero_token_steps"),
        }
        out_rows.append(row)
        if not res["success_12s"] or res["win_mean_theta"] is None:
            continue
        score = -res["win_mean_theta"]
        if best_any is None or score > best_any[0]:
            best_any = (score, seq, res, mech)
        if mech in ("CONTACT_PRESERVING_PIVOT", "LIKELY_PIVOT_OR_PARTIAL") and (best_pivot is None or score > best_pivot[0]):
            best_pivot = (score, seq, res, mech)
        if mech == "RELEASE_RECATCH" and (best_recatch is None or score > best_recatch[0]):
            best_recatch = (score, seq, res, mech)
    return {"best_any": best_any, "best_pivot": best_pivot, "best_recatch": best_recatch, "rows": out_rows}


def capture_legal_generated(env, book, base_blob, rng, n_target=12):
    """From a base recoverable IC, run legal unload/wrist and snapshot high-tilt states."""
    snaps = []
    recipes = []
    for tilt in (6, 10):
        for unload in (12, 36, 30):
            for n_t in (0, 8, 15):
                for n_u in (5, 10, 20, 30):
                    recipes.append([tilt] * n_t + [unload] * n_u)
    rng.shuffle(recipes)
    for seq in recipes:
        if len(snaps) >= n_target:
            break
        restore_capture(env, base_blob["capture"])
        lost_low = 0
        ok = True
        for tok in seq:
            _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
            step_legal(env.sim, action)
            env._prev = np.asarray(action, np.float32)
            failed, lost_low = recovery_failed(env.sim, base_blob["ref"], lost_low)
            if failed or latched(env.sim):
                ok = False
                break
        if not ok:
            continue
        m = measure(env.sim, U_TGT, P_TGT)
        if not m["bilateral"] or m["e_theta_deg"] < 15:
            continue
        # take capture at this legal state; keep same ref for nominal handoff
        cap = take_capture(env)
        blob = {"capture": cap, "ref": base_blob["ref"], "meta": {
            "origin": "legal_mid_unload_wrist",
            "parent": "base",
            "tokens": seq,
            "e_theta_deg": m["e_theta_deg"],
        }}
        snaps.append(blob)
    return snaps


def scan_existing(env, max_scan=300):
    paths = []
    for d in SNAP_DIRS:
        if d.is_dir():
            paths.extend(sorted(d.glob("*.pkl")))
    rows = []
    seen_fp = set()
    for p in paths[:max_scan]:
        try:
            blob = pickle.loads(p.read_bytes())
        except Exception:
            continue
        if "capture" not in blob or "ref" not in blob:
            continue
        try:
            restore_capture(env, blob["capture"])
        except Exception:
            continue
        if latched(env.sim):
            continue
        m = measure(env.sim, U_TGT, P_TGT)
        if not m["bilateral"] or m["obj_z"] < 0.40:
            continue
        g = gravity_align(env.sim)
        fp = fingerprint(m, g)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        rows.append({
            "id": p.stem,
            "path": str(p.relative_to(ROOT)),
            "source": "existing_snapshot",
            "e_theta_deg": m["e_theta_deg"],
            "e_pos_mm": m["e_pos_mm"],
            "bin": bin_of(m["e_theta_deg"]),
            "ecc_xy_mm": g["ecc_xy_m"] * 1e3,
            "lever_mm": g["lever_m"] * 1e3,
            "gravity_helps": int(g["helps"]),
            "align": g["align"],
            "tau_g": g["tau_g_norm"],
            "nL": m["nL"], "nR": m["nR"],
            "fp": fp,
            "blob": blob,
        })
    return rows, seen_fp


def select_pool(scanned, max_n=45, rng=None):
    rng = rng or np.random.default_rng(0)
    # prioritize larger tilt; keep some mid
    by_bin = {"15-25": [], "25-40": [], ">40": [], "<15": []}
    for r in scanned:
        by_bin[r["bin"]].append(r)
    for k in by_bin:
        by_bin[k].sort(key=lambda x: -x["e_theta_deg"])
    pool = []
    # take all >40 and 25-40, then fill 15-25
    for k in (">40", "25-40", "15-25"):
        for r in by_bin[k]:
            if len(pool) >= max_n:
                break
            pool.append(r)
        if len(pool) >= max_n:
            break
    # optional few <15 excluded unless pool small
    return pool


def render_video(env, book, blob, tokens, dest, label):
    from visualization.camera import mjv_camera_from_preset
    from visualization.overlays import caption
    from visualization.video_recorder import write_frames_mp4

    restore_capture(env, blob["capture"])
    cam = mjv_camera_from_preset({
        "lookat": np.array(env.sim.data.xpos[env.sim.ids.object_body], float),
        "distance": 0.55, "azimuth": 132.0, "elevation": -22.0,
    })
    renderer = __import__("mujoco").Renderer(env.sim.model, 360, 480)
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
            grab(f"{label} t={now-t0:.2f} tok={tok} eθ={m['e_theta_deg']:.1f} n={cs['nL']}/{cs['nR']}")
            next_frame += 0.04
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            grab("FAIL")
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
                grab(f"nom {now-t_h:.2f} eθ={m['e_theta_deg']:.1f}")
                next_frame += 0.04
            if latched(env.sim):
                break
    renderer.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_frames_mp4(frames, dest, fps=25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--max-scan", type=int, default=300)
    parser.add_argument("--short-budget", type=int, default=28)
    parser.add_argument("--active-budget", type=int, default=36)
    parser.add_argument("--active-top", type=int, default=8)
    parser.add_argument("--gen-extra", type=int, default=10)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RAW").mkdir(exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    print("SCAN_EXISTING", flush=True)
    scanned, seen_fp = scan_existing(env, max_scan=args.max_scan)
    write_json(OUT / "RAW" / "scan_summary.json", {
        "n_scanned_bilateral_unique": len(scanned),
        "bins": {b: sum(1 for r in scanned if r["bin"] == b) for b in ("<15", "15-25", "25-40", ">40")},
        "max_e_theta": max((r["e_theta_deg"] for r in scanned), default=None),
    })
    print("SCAN bins", {b: sum(1 for r in scanned if r["bin"] == b) for b in ("<15", "15-25", "25-40", ">40")}, flush=True)

    pool = select_pool(scanned, max_n=args.max_candidates, rng=rng)

    # Legal generation from a few high-tilt bases + pivot IC
    print("LEGAL_GENERATE", flush=True)
    gen_bases = [r for r in scanned if r["id"] in ("hrv1_920031", "hrv1_920117", "hrv1_920131", "hard_910101")]
    if not gen_bases and scanned:
        gen_bases = scanned[:3]
    gen_rows = []
    for base in gen_bases[:3]:
        extras = capture_legal_generated(env, book, base["blob"], rng, n_target=max(3, args.gen_extra // 3))
        for i, blob in enumerate(extras):
            restore_capture(env, blob["capture"])
            m = measure(env.sim, U_TGT, P_TGT)
            g = gravity_align(env.sim)
            fp = fingerprint(m, g)
            if fp in seen_fp or m["e_theta_deg"] < 15:
                continue
            seen_fp.add(fp)
            sid = f"gen_{base['id']}_{i}_{fp}"
            spath = OUT / "snapshots" / f"{sid}.pkl"
            spath.write_bytes(pickle.dumps(blob))
            row = {
                "id": sid,
                "path": str(spath.relative_to(ROOT)),
                "source": "legal_generated",
                "parent": base["id"],
                "gen_tokens": json.dumps(blob["meta"]["tokens"]),
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "bin": bin_of(m["e_theta_deg"]),
                "ecc_xy_mm": g["ecc_xy_m"] * 1e3,
                "lever_mm": g["lever_m"] * 1e3,
                "gravity_helps": int(g["helps"]),
                "align": g["align"],
                "tau_g": g["tau_g_norm"],
                "nL": m["nL"], "nR": m["nR"],
                "fp": fp,
                "blob": blob,
            }
            gen_rows.append(row)
            if len(pool) < args.max_candidates + 10:
                pool.append(row)
    print("GEN", len(gen_rows), "pool", len(pool), flush=True)

    # IC generation audit (without blobs)
    gen_audit = []
    for r in pool:
        gen_audit.append({k: r[k] for k in r if k != "blob"})
    write_csv(OUT / "IC_GENERATION_AUDIT.csv", gen_audit)

    # Short HANDOFF filter
    print("SHORT_FILTER", flush=True)
    short_rows = []
    promising = []
    for r in pool:
        ic = r["id"]
        blob = r["blob"]
        print("SHORT", ic, f"e0={r['e_theta_deg']:.1f}", flush=True)
        imm = run_prefix(env, book, blob, [])
        known = run_prefix(env, book, blob, [30])
        known44 = run_prefix(env, book, blob, [44])
        best, _ = short_search(env, book, blob, np.random.default_rng(args.seed + abs(hash(ic)) % 10000), max_n=args.short_budget)
        best_res = best[2] if best else None
        best_tok = best[1] if best else None
        row = {
            "ic_id": ic,
            "source": r["source"],
            "bin": r["bin"],
            "e_theta0": r["e_theta_deg"],
            "e_pos0": r["e_pos_mm"],
            "gravity_helps": r["gravity_helps"],
            "ecc_xy_mm": r["ecc_xy_mm"],
            "imm_s12": int(imm["success_12s"]),
            "imm_win_theta": imm["win_mean_theta"],
            "short30_s12": int(known["success_12s"]),
            "short30_win_theta": known["win_mean_theta"],
            "short44_s12": int(known44["success_12s"]),
            "short44_win_theta": known44["win_mean_theta"],
            "best_short_tokens": json.dumps(best_tok) if best_tok is not None else "",
            "best_short_s12": int(best_res["success_12s"]) if best_res else 0,
            "best_short_win_theta": None if not best_res else best_res["win_mean_theta"],
            "best_short_win_pos": None if not best_res else best_res["win_mean_pos"],
            "best_short_rec_s": None if not best_res else best_res["recovery_elapsed"],
            "stable_but_misaligned": 0,
        }
        # stable but significantly misaligned: S12 and win theta >= 10
        if best_res and best_res["success_12s"] and best_res["win_mean_theta"] is not None:
            if best_res["win_mean_theta"] >= 10.0:
                row["stable_but_misaligned"] = 1
        short_rows.append(row)
        write_json(OUT / "RAW" / f"{ic}_short.json", {
            "imm": {k: imm[k] for k in imm if k not in ("m0", "m_handoff", "m_final", "sub")},
            "best_tokens": best_tok,
            "best_win_theta": None if not best_res else best_res["win_mean_theta"],
            "meta": {k: r[k] for k in r if k != "blob"},
        })
        # promising: large e0 and (stable_but_misaligned OR failed all shorts with e0>=20)
        score = r["e_theta_deg"]
        if row["stable_but_misaligned"]:
            score += 50
        if not row["best_short_s12"] and r["e_theta_deg"] >= 20:
            score += 20
        if row["best_short_s12"] and row["best_short_win_theta"] is not None:
            score += row["best_short_win_theta"]  # higher residual = more interesting
        promising.append((score, r, row, best_res, best_tok))

    write_csv(OUT / "SHORT_HANDOFF_FILTER.csv", short_rows)
    promising.sort(key=lambda x: -x[0])
    active_set = promising[: args.active_top]
    print("ACTIVE_SET", [x[1]["id"] for x in active_set], flush=True)

    # Active correction
    active_rows = []
    for score, r, srow, best_short_res, best_short_tok in active_set:
        ic = r["id"]
        blob = r["blob"]
        print("ACTIVE", ic, flush=True)
        act = active_search(env, book, blob, np.random.default_rng(args.seed + 7), max_n=args.active_budget)
        def pack(tag, pack_):
            if pack_ is None:
                return {
                    f"{tag}_tokens": "", f"{tag}_s12": 0, f"{tag}_win_theta": None,
                    f"{tag}_win_pos": None, f"{tag}_mech": "", f"{tag}_d_mid": None, f"{tag}_rec_s": None,
                }
            _sc, seq, res, mech = pack_
            d_mid = float(res["m0"]["e_theta_deg"] - res["m_handoff"]["e_theta_deg"])
            return {
                f"{tag}_tokens": json.dumps(seq),
                f"{tag}_s12": int(res["success_12s"]),
                f"{tag}_win_theta": res["win_mean_theta"],
                f"{tag}_win_pos": res["win_mean_pos"],
                f"{tag}_mech": mech,
                f"{tag}_d_mid": d_mid,
                f"{tag}_rec_s": res["recovery_elapsed"],
                f"{tag}_e_handoff": res["m_handoff"]["e_theta_deg"],
            }
        row = {
            "ic_id": ic,
            "bin": r["bin"],
            "e_theta0": r["e_theta_deg"],
            "e_pos0": r["e_pos_mm"],
            "gravity_helps0": r["gravity_helps"],
            "best_short_tokens": json.dumps(best_short_tok) if best_short_tok else "",
            "best_short_s12": int(best_short_res["success_12s"]) if best_short_res else 0,
            "best_short_win_theta": None if not best_short_res else best_short_res["win_mean_theta"],
            "best_short_win_pos": None if not best_short_res else best_short_res["win_mean_pos"],
            "best_short_rec_s": None if not best_short_res else best_short_res["recovery_elapsed"],
            **pack("pivot", act["best_pivot"]),
            **pack("recatch", act["best_recatch"]),
            **pack("any", act["best_any"]),
        }
        # advantage vs short
        if best_short_res and best_short_res["success_12s"] and act["best_any"] and act["best_any"][2]["success_12s"]:
            row["theta_improve_vs_short"] = float(
                best_short_res["win_mean_theta"] - act["best_any"][2]["win_mean_theta"]
            )
        else:
            row["theta_improve_vs_short"] = None
        active_rows.append(row)
        write_json(OUT / "RAW" / f"{ic}_active.json", {
            "best_pivot": None if not act["best_pivot"] else {
                "tokens": act["best_pivot"][1], "mech": act["best_pivot"][3],
                "win_theta": act["best_pivot"][2]["win_mean_theta"],
                "s12": act["best_pivot"][2]["success_12s"],
            },
            "best_recatch": None if not act["best_recatch"] else {
                "tokens": act["best_recatch"][1], "mech": act["best_recatch"][3],
                "win_theta": act["best_recatch"][2]["win_mean_theta"],
            },
            "best_any": None if not act["best_any"] else {
                "tokens": act["best_any"][1], "mech": act["best_any"][3],
                "win_theta": act["best_any"][2]["win_mean_theta"],
            },
            "n_eval": len(act["rows"]),
            "label_counts": {},
        })
        # label counts
        from collections import Counter
        write_json(OUT / "RAW" / f"{ic}_active_labels.json", dict(Counter(x["mech"] for x in act["rows"])))

    write_csv(OUT / "ACTIVE_CORRECTION_RESULTS.csv", active_rows)

    # Videos: largest tilt + best improve
    if not args.skip_videos and active_rows:
        try:
            # largest e0 with any S12 active
            cand = sorted(active_set, key=lambda x: -x[1]["e_theta_deg"])
            for score, r, srow, bs, bt in cand[:2]:
                blob = r["blob"]
                # find matching active row
                ar = next(a for a in active_rows if a["ic_id"] == r["id"])
                short_tok = json.loads(ar["best_short_tokens"]) if ar["best_short_tokens"] else [30]
                render_video(env, book, blob, short_tok, OUT / "videos" / f"{r['id']}_short.mp4", "short")
                if ar.get("any_tokens"):
                    render_video(env, book, blob, json.loads(ar["any_tokens"]),
                                 OUT / "videos" / f"{r['id']}_active.mp4", "active")
        except Exception as exc:
            (OUT / "videos" / "VIDEO_ERROR.txt").write_text(str(exc), encoding="utf-8")
            print("VIDEO_ERROR", exc, flush=True)

    n_stable_mis = sum(1 for r in short_rows if r["stable_but_misaligned"])
    n_gt25 = sum(1 for r in gen_audit if r["e_theta_deg"] >= 25)
    n_gt40 = sum(1 for r in gen_audit if r["e_theta_deg"] >= 40)
    piv_wins = sum(1 for r in active_rows if (r.get("theta_improve_vs_short") or 0) >= 2.0)
    write_json(OUT / "summary.json", {
        "seed": args.seed,
        "n_pool": len(pool),
        "n_legal_generated": len(gen_rows),
        "n_existing_ge_25": n_gt25,
        "n_existing_ge_40": n_gt40,
        "n_stable_but_misaligned": n_stable_mis,
        "n_active": len(active_rows),
        "n_active_improve_ge2deg": piv_wins,
        "bins_pool": {b: sum(1 for r in gen_audit if r["bin"] == b) for b in ("15-25", "25-40", ">40", "<15")},
        "verified_pivot_ic_origin": "ballistic_impact hard_recovery_v1 hrv1_920031",
    })
    print("PILOT_DONE", OUT, "stable_mis", n_stable_mis, "improve", piv_wins, flush=True)


if __name__ == "__main__":
    main()
