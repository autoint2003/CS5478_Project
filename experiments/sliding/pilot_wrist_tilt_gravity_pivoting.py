"""Wrist-tilt-assisted gravity pivoting — structured bounded pilot.

Hypothesis: tilt wrist while closed → unload grip while retaining contact →
gravity rotates cylinder relative to hand → re-tighten → HANDOFF.

Does NOT repeat undirected open-first search from gravity_assisted_pivoting_feasibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import AIRBORNE_V3, grip_tau, map_action
from envs.ballistic_impact import z_tgt_of
from envs.contacts.frames import extract_contacts
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor
from envs.outcomes.horizon_outcome import recovery_failed, terminal_reason
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, object_vz, observe
from envs.simulation.capture import restore_capture
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.grasp_geometry import (
    AXIS_REVERSAL_SYMMETRIC,
    cylinder_axis_hand,
    measure,
    orientation_error_rad,
    p_com_hand,
)
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.recovery_episode import make_training_env
from training.recovery_execution import _escaped
from training.pipeline import HANDOFF

OUT = ROOT / "results" / "diagnostics" / "wrist_tilt_gravity_pivoting"
PRIOR = ROOT / "results" / "diagnostics" / "gravity_assisted_pivoting_feasibility"
PILOT = ROOT / "results" / "diagnostics" / "nontrivial_recovery_ic_mining"
MIG = ROOT / "results" / "diagnostics" / "grasp_configuration_migration"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
SEED = 20261018

U_TGT = np.array([-0.013840181460089382, -0.0030378179260123783, -0.9998996054801703], float)
P_TGT = np.array([-0.001115724743759791, -6.904231962095219e-06, 0.10474686676582946], float)

# Structured token roles (verified from codebook rows)
TILT_CLOSE = [6, 4, 10]       # a6=+1 secure + nonzero wrist
UNLOAD_MID = [12]             # only mid grip level a6≈-0.48, τ≈-3.2
UNLOAD_OPEN = [36]            # open with small trans (not pure)
BRAKE = [0, 14]               # secure close
SHORT_BASE = {  # from prior manifests
    "hard_910101": [30],
    "hrv1_920131": [41],
    "hrv1_920099": [41],
    "hrv1_920031": [30],
    "hard_910172": [44],
    "hrv1_920151": [30],
}
CANDIDATES = list(SHORT_BASE.keys())


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
    fnL = sum(abs(r["Fn"]) for r in rows if r["side"] == "L")
    fnR = sum(abs(r["Fn"]) for r in rows if r["side"] == "R")
    return {"nL": nL, "nR": nR, "fnL": fnL, "fnR": fnR,
            "bilateral": bool(nL > 0 and nR > 0), "any": bool(nL + nR > 0),
            "n": len(rows)}


def gravity_analysis(sim, u_tgt):
    mass = float(sim.model.body_mass[sim.ids.object_body])
    g_w = np.array(sim.model.opt.gravity, float)
    Fg = mass * g_w
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    rows, _ = extract_contacts(sim)
    if rows:
        pivot = np.mean([np.asarray(r["pos_w"], float) for r in rows], axis=0)
    else:
        pivot = ph + Rh @ np.array([0.0, 0.0, 0.105])
    r = po - pivot
    tau_g_h = Rh.T @ np.cross(r, Fg)
    u_c = cylinder_axis_hand(sim)
    u_t = np.asarray(u_tgt, float)
    u_t = u_t / max(float(np.linalg.norm(u_t)), 1e-12)
    u_c = u_c / max(float(np.linalg.norm(u_c)), 1e-12)
    if float(np.dot(u_c, u_t)) < 0:
        u_t = -u_t
    axis = np.cross(u_c, u_t)
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-9 else np.zeros(3)
    align = float(np.dot(tau_g_h, axis)) if n > 1e-9 else 0.0
    # wrist attitude proxy: hand z vs world -g
    hand_z_w = Rh[:, 2]
    tilt_from_up = float(np.degrees(np.arccos(np.clip(-float(np.dot(hand_z_w, g_w / np.linalg.norm(g_w))), -1, 1))))
    p_c_h = Rh.T @ (po - ph)
    ecc = float(np.linalg.norm(p_c_h[:2]))  # radial offset in hand xy
    return {
        "tau_g_h": tau_g_h.tolist(),
        "tau_g_h_norm": float(np.linalg.norm(tau_g_h)),
        "align": align,
        "helps": bool(align > 1e-6),
        "lever_m": float(np.linalg.norm(r)),
        "ecc_xy_m": ecc,
        "p_com_h": p_c_h.tolist(),
        "hand_tilt_from_anti_g_deg": tilt_from_up,
        "contact": contact_summary(sim),
        "e_theta_deg": float(np.degrees(orientation_error_rad(u_c, u_tgt, AXIS_REVERSAL_SYMMETRIC))),
    }


def step_legal_sublogged(sim, action, log_rows, token, phase):
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
            "t": float(sim.data.time), "token": int(token), "phase": phase, "sub": s,
            "tau": tau, "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
            "u_c_h": m["u_c_h"], "w_rel_h": m["w_rel_h"],
            "nL": cs["nL"], "nR": cs["nR"], "fnL": cs["fnL"], "fnR": cs["fnR"],
            "bilateral": cs["bilateral"], "obj_z": m["obj_z"],
        })
    return action


def topology(sub_log):
    if not sub_log:
        return {"substep": False}
    dt = float(sub_log[1]["t"] - sub_log[0]["t"]) if len(sub_log) > 1 else 0.002
    max_zero = max_single = z = s = 0.0
    n_bi = n_z = n_s = 0
    for row in sub_log:
        n = int(row["nL"] > 0) + int(row["nR"] > 0)
        if n == 2:
            n_bi += 1
            z = s = 0.0
        elif n == 0:
            n_z += 1
            z += dt
            s = 0.0
            max_zero = max(max_zero, z)
        else:
            n_s += 1
            s += dt
            z = 0.0
            max_single = max(max_single, s)
    n = len(sub_log)
    return {
        "substep": True,
        "frac_bilateral": n_bi / n,
        "frac_zero": n_z / n,
        "frac_single": n_s / n,
        "max_zero_ms": max_zero * 1e3,
        "max_single_ms": max_single * 1e3,
        "sustained_release": bool(max_zero * 1e3 > 40.0),
        "sustained_single": bool(max_single * 1e3 > 40.0),
        "min_fingers": int(min((int(r["nL"] > 0) + int(r["nR"] > 0) for r in sub_log), default=0)),
    }


def run_nominal(env, blob):
    apply_handoff(env.sim, blob["ref"])
    t_h = float(env.sim.data.time)
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    if latched(env.sim):
        reason = "TABLE_DROP"
    while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(env.sim)
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
    return reason == "SUCCESS_12S", reason, measure(env.sim, U_TGT, P_TGT)


def run_phased(env, book, blob, phases, log_sub=True):
    """phases: list of (phase_name, token_list)."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    m0 = measure(env.sim, U_TGT, P_TGT)
    g0 = gravity_analysis(env.sim, U_TGT)
    sub = []
    phase_marks = {"t0": g0}
    lost_low = 0
    tokens_flat = []
    for phase, toks in phases:
        for tok in toks:
            tokens_flat.append(int(tok))
            _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
            if log_sub:
                step_legal_sublogged(env.sim, action, sub, int(tok), phase)
            else:
                step_legal(env.sim, action)
            env._prev = np.asarray(action, np.float32)
            failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
            if failed:
                return {
                    "ok": False, "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                    "tokens": tokens_flat, "m0": m0, "g0": g0, "phase_marks": phase_marks,
                    "m_mid": measure(env.sim, U_TGT, P_TGT),
                    "m_final": measure(env.sim, U_TGT, P_TGT),
                    "success_12s": False, "sub": sub, "topo": topology(sub),
                    "recovery_elapsed": float(env.sim.data.time) - t0,
                }
        phase_marks[f"after_{phase}"] = {
            "grav": gravity_analysis(env.sim, U_TGT),
            "measure": measure(env.sim, U_TGT, P_TGT),
            "t": float(env.sim.data.time) - t0,
        }
    m_mid = measure(env.sim, U_TGT, P_TGT)
    s12, reason, m_final = run_nominal(env, blob)
    return {
        "ok": True, "reason": reason, "tokens": tokens_flat,
        "m0": m0, "g0": g0, "phase_marks": phase_marks,
        "m_mid": m_mid, "m_final": m_final, "success_12s": s12,
        "sub": sub, "topo": topology(sub),
        "recovery_elapsed": phase_marks.get("after_brake", phase_marks.get("after_unload", {})).get("t")
        if isinstance(phase_marks.get("after_brake"), dict) else float(env.sim.data.time) - t0,
    }


def classify(res):
    m0, mm, mf = res["m0"], res["m_mid"], res["m_final"]
    d_mid = float(m0["e_theta_deg"] - mm["e_theta_deg"])
    d_fin = float(m0["e_theta_deg"] - mf["e_theta_deg"])
    topo = res["topo"]
    s12 = res["success_12s"]
    # Did tilt produce favorable gravity?
    after_tilt = res["phase_marks"].get("after_tilt", {})
    grav_tilt = after_tilt.get("grav") if isinstance(after_tilt, dict) else None
    helps_after_tilt = bool(grav_tilt and grav_tilt.get("helps"))
    align0 = float(res["g0"]["align"])
    align_tilt = float(grav_tilt["align"]) if grav_tilt else None

    if topo.get("sustained_release"):
        label = "VERIFIED_RELEASE_REGRASP" if (s12 and d_fin >= 3) else (
            "RELATIVE_ORIENTATION_CHANGE_BUT_UNSTABLE" if d_mid >= 3 else "AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE"
        )
    elif d_mid >= 5 and topo.get("frac_bilateral", 0) >= 0.85 and not topo.get("sustained_release") and s12 and d_fin >= 3:
        label = "VERIFIED_CONTACT_PRESERVING_PIVOT"
    elif d_mid >= 5 and topo.get("sustained_single") and s12 and d_fin >= 3:
        label = "VERIFIED_SINGLE_CONTACT_PIVOT"
    elif abs(d_mid) < 2 and abs(d_fin) < 2:
        label = "NO_MEANINGFUL_RELATIVE_ROTATION"
    elif s12 and d_fin < 3 and mf["e_theta_deg"] > 12:
        label = "STABLE_BUT_CONFIGURATION_INCORRECT"
    elif d_mid >= 3 and not s12:
        label = "RELATIVE_ORIENTATION_CHANGE_BUT_UNSTABLE"
    else:
        label = "AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE"

    return {
        "label": label,
        "delta_theta_mid_deg": d_mid,
        "delta_theta_final_deg": d_fin,
        "helps_at_t0": bool(res["g0"]["helps"]),
        "helps_after_tilt": helps_after_tilt,
        "align_t0": align0,
        "align_after_tilt": align_tilt,
        "tau_g_after_tilt": None if not grav_tilt else grav_tilt["tau_g_h_norm"],
        "tilt_angle_deg": None if not grav_tilt else grav_tilt["hand_tilt_from_anti_g_deg"],
    }


def build_structured_sequences():
    """Small bounded set: tilt(closed) → unload → brake. No open-first undirected search."""
    seqs = []
    for tilt_tok in TILT_CLOSE:
        for n_tilt in (3, 5, 8, 12, 20):
            for unload_tok in UNLOAD_MID + UNLOAD_OPEN:
                for n_un in (2, 5, 10, 15, 25):
                    for brake_tok in BRAKE:
                        for n_br in (2, 5, 8):
                            seqs.append({
                                "name": f"tilt{tilt_tok}x{n_tilt}_u{unload_tok}x{n_un}_b{brake_tok}x{n_br}",
                                "phases": [
                                    ("tilt", [tilt_tok] * n_tilt),
                                    ("unload", [unload_tok] * n_un),
                                    ("brake", [brake_tok] * n_br),
                                ],
                            })
    # tilt then mid-hold then brake (no full open)
    for tilt_tok in TILT_CLOSE:
        for n_tilt in (5, 12):
            for n_un in (10, 25):
                seqs.append({
                    "name": f"tilt{tilt_tok}x{n_tilt}_mid12x{n_un}_b0x5",
                    "phases": [
                        ("tilt", [tilt_tok] * n_tilt),
                        ("unload", [12] * n_un),
                        ("brake", [0] * 5),
                    ],
                })
    # tilt-only then immediate handoff (is tilt alone enough?)
    for tilt_tok in TILT_CLOSE:
        for n_tilt in (5, 12, 20):
            seqs.append({
                "name": f"tilt_only_{tilt_tok}x{n_tilt}",
                "phases": [("tilt", [tilt_tok] * n_tilt)],
            })
    # controls: unload-first (prior failure mode) for contrast — few only
    for n_un in (5, 10):
        seqs.append({
            "name": f"CONTROL_open_first_36x{n_un}_b0x5",
            "phases": [("unload", [36] * n_un), ("brake", [0] * 5)],
        })
    return seqs


def phase1_analysis(book):
    """Document prior pilot failure modes from saved CSVs."""
    rows = list(csv.DictReader((PRIOR / "LEGAL_SEARCH_SUMMARY.csv").open(encoding="utf-8")))
    search = [r for r in rows if r["branch"].startswith("search")]
    n_start_open = 0
    n_start_close = 0
    for r in search:
        toks = json.loads(r["tokens"])
        if not toks:
            continue
        if float(book[toks[0], 6]) < -0.5:
            n_start_open += 1
        elif float(book[toks[0], 6]) > 0.5:
            n_start_close += 1
    grips = sorted({round(float(book[i, 6]), 5) for i in range(47)})
    cand = list(csv.DictReader((PRIOR / "CANDIDATE_IC_AUDIT.csv").open(encoding="utf-8")))
    return {
        "verified": {
            "prior_search_rows": len(search),
            "prior_start_open": n_start_open,
            "prior_start_close_or_mid": len(search) - n_start_open,
            "fraction_start_open": n_start_open / max(len(search), 1),
            "distinct_a_grip_values": grips,
            "n_distinct_grip_levels": len(grips),
            "grip_note": "K48 a[6] takes only 3 values: -1 (open τ=+2), ≈-0.48 (mid τ≈-3.2), +1 (secure τ=-18)",
            "prior_cpp_count": 0,
            "prior_release_regrasp": 2,
            "release_first_token": 39,
            "release_first_token_is_open": True,
            "levers_mm": [round(1000 * float(r["lever_arm_m"]), 2) for r in cand],
            "gravity_helps_count": sum(int(r["gravity_helps"]) for r in cand),
        },
        "hypotheses_for_this_pilot": [
            "Prior search did not systematically pre-tilt while closed before unload",
            "Wrist pre-tilt may flip gravity alignment onto the error-reduction axis",
            "Mid grip (tok12) may allow rotation without full bilateral loss",
        ],
    }


def render_video(env, book, blob, tokens, dest, label):
    from visualization.camera import mjv_camera_from_preset
    from visualization.overlays import caption
    from visualization.video_recorder import write_frames_mp4

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
            grab(f"FAIL {terminal_reason(env.sim)}")
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
    parser.add_argument("--max-per-ic", type=int, default=40,
                        help="Max structured sequences per IC (bounded)")
    parser.add_argument("--skip-videos", action="store_true")
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

    p1 = phase1_analysis(book)
    write_json(OUT / "PHASE1_PRIOR_PILOT_ANALYSIS.json", p1)
    print("PHASE1", p1["verified"], flush=True)

    all_seqs = build_structured_sequences()
    rng.shuffle(all_seqs)
    write_json(OUT / "RAW" / "structured_sequence_catalog.json", {
        "n_total_catalog": len(all_seqs),
        "max_per_ic": args.max_per_ic,
        "tilt_tokens": TILT_CLOSE,
        "unload_tokens": UNLOAD_MID + UNLOAD_OPEN,
        "brake_tokens": BRAKE,
        "token_semantics": {
            str(t): {"row": book[t].tolist(), "tau": float(grip_tau(book[t, 6]))}
            for t in sorted(set(TILT_CLOSE + UNLOAD_MID + UNLOAD_OPEN + BRAKE + [30, 39]))
        },
    })

    ic_audit = []
    trial_rows = []
    best_by_ic = {}

    for ic in CANDIDATES:
        snap = snaps.get(ic)
        if not snap:
            print("SKIP", ic, flush=True)
            continue
        blob = pickle.loads((ROOT / snap).read_bytes())
        if "capture" not in blob:
            continue
        print("IC", ic, flush=True)
        restore_capture(env, blob["capture"])
        g0 = gravity_analysis(env.sim, U_TGT)
        m0 = measure(env.sim, U_TGT, P_TGT)

        # Probe: apply each tilt token ×12 while closed; measure gravity after
        tilt_probes = []
        for tt in TILT_CLOSE:
            restore_capture(env, blob["capture"])
            for _ in range(12):
                _r, action = decode_action(book, tt, np.zeros(7, np.float32))
                step_legal(env.sim, action)
            gp = gravity_analysis(env.sim, U_TGT)
            tilt_probes.append({"tilt_tok": tt, "n": 12, **{k: gp[k] for k in (
                "align", "helps", "tau_g_h_norm", "lever_m", "ecc_xy_m",
                "hand_tilt_from_anti_g_deg", "e_theta_deg"
            )}, "contact": gp["contact"]})
        write_json(OUT / "RAW" / f"{ic}_tilt_probes.json", tilt_probes)

        # short baseline
        short = SHORT_BASE[ic]
        res_short = run_phased(env, book, blob, [("short", short)], log_sub=True)
        cls_short = classify(res_short)

        ic_audit.append({
            "ic_id": ic,
            "snapshot": snap,
            "e_theta0_deg": m0["e_theta_deg"],
            "e_pos0_mm": m0["e_pos_mm"],
            "ecc_xy_mm": g0["ecc_xy_m"] * 1e3,
            "lever_mm": g0["lever_m"] * 1e3,
            "align_t0": g0["align"],
            "helps_t0": int(g0["helps"]),
            "bilateral0": int(g0["contact"]["bilateral"]),
            "tilt_probe_any_helps": int(any(p["helps"] for p in tilt_probes)),
            "tilt_probe_best_align": max(p["align"] for p in tilt_probes),
            "short_s12": int(res_short["success_12s"]),
            "short_e_theta_final": res_short["m_final"]["e_theta_deg"],
        })

        trial_rows.append({
            "ic_id": ic, "trial": "short_baseline", "name": "short",
            "tokens": json.dumps(short),
            "success_12s": int(res_short["success_12s"]),
            "reason": res_short["reason"],
            "e0": res_short["m0"]["e_theta_deg"],
            "emid": res_short["m_mid"]["e_theta_deg"],
            "ef": res_short["m_final"]["e_theta_deg"],
            "label": cls_short["label"],
            "helps_after_tilt": cls_short["helps_after_tilt"],
            "align_after_tilt": cls_short["align_after_tilt"],
            "frac_bilateral": res_short["topo"].get("frac_bilateral"),
            "max_zero_ms": res_short["topo"].get("max_zero_ms"),
            "sustained_release": int(bool(res_short["topo"].get("sustained_release"))),
        })

        # Prefer sequences whose tilt tok helped in probe
        helping_tilts = {p["tilt_tok"] for p in tilt_probes if p["helps"]}
        ordered = sorted(
            all_seqs,
            key=lambda s: (
                0 if (s["phases"][0][0] == "tilt" and s["phases"][0][1][0] in helping_tilts) else 1,
                0 if s["phases"][0][0] == "tilt" else 2,  # tilt-first before open-first controls
            ),
        )
        n_run = 0
        for spec in ordered:
            if n_run >= args.max_per_ic:
                break
            # skip redundant controls after a few
            if spec["name"].startswith("CONTROL_") and n_run > 8:
                continue
            res = run_phased(env, book, blob, spec["phases"], log_sub=True)
            n_run += 1
            cls = classify(res)
            row = {
                "ic_id": ic, "trial": "structured", "name": spec["name"],
                "tokens": json.dumps(res["tokens"]),
                "success_12s": int(res["success_12s"]),
                "reason": res["reason"],
                "e0": res["m0"]["e_theta_deg"],
                "emid": res["m_mid"]["e_theta_deg"],
                "ef": res["m_final"]["e_theta_deg"],
                "d_mid": cls["delta_theta_mid_deg"],
                "d_fin": cls["delta_theta_final_deg"],
                "label": cls["label"],
                "helps_t0": int(cls["helps_at_t0"]),
                "helps_after_tilt": int(bool(cls["helps_after_tilt"])),
                "align_t0": cls["align_t0"],
                "align_after_tilt": cls["align_after_tilt"],
                "tau_g_after_tilt": cls["tau_g_after_tilt"],
                "tilt_angle_deg": cls["tilt_angle_deg"],
                "frac_bilateral": res["topo"].get("frac_bilateral"),
                "max_zero_ms": res["topo"].get("max_zero_ms"),
                "sustained_release": int(bool(res["topo"].get("sustained_release"))),
                "sustained_single": int(bool(res["topo"].get("sustained_single"))),
            }
            trial_rows.append(row)

            interesting = (
                cls["label"] in (
                    "VERIFIED_CONTACT_PRESERVING_PIVOT",
                    "VERIFIED_SINGLE_CONTACT_PIVOT",
                    "VERIFIED_RELEASE_REGRASP",
                )
                or (res["success_12s"] and cls["delta_theta_final_deg"] >= 3)
                or (cls["helps_after_tilt"] and cls["delta_theta_mid_deg"] >= 3)
            )
            if interesting:
                write_json(OUT / "RAW" / f"{ic}_{spec['name']}.json", {
                    "spec": spec, "classification": cls,
                    "m0": res["m0"], "m_mid": res["m_mid"], "m_final": res["m_final"],
                    "g0": res["g0"], "phase_marks": {
                        k: v for k, v in res["phase_marks"].items()
                        if k == "t0" or (isinstance(v, dict) and "grav" in v)
                    },
                    "topo": res["topo"],
                    "sub_log": res["sub"],
                    "success_12s": res["success_12s"], "reason": res["reason"],
                })
                score = (
                    (100 if "VERIFIED_CONTACT" in cls["label"] else 0)
                    + (50 if "SINGLE_CONTACT" in cls["label"] else 0)
                    + cls["delta_theta_final_deg"]
                    + (10 if res["success_12s"] else 0)
                )
                if ic not in best_by_ic or score > best_by_ic[ic][0]:
                    best_by_ic[ic] = (score, spec, res, cls)

            if cls["label"] == "VERIFIED_CONTACT_PRESERVING_PIVOT":
                print("VERIFIED_CPP", ic, spec["name"], flush=True)
                break

        print(f"DONE_IC {ic} trials={n_run} tilt_helps={any(p['helps'] for p in tilt_probes)}", flush=True)

    write_csv(OUT / "IC_AUDIT.csv", ic_audit)
    write_csv(OUT / "EXPERIMENT_TABLE.csv", trial_rows)

    # videos
    if not args.skip_videos:
        try:
            # best overall
            if best_by_ic:
                ic, (sc, spec, res, cls) = max(best_by_ic.items(), key=lambda kv: kv[1][0])
                blob = pickle.loads((ROOT / snaps[ic]).read_bytes())
                render_video(env, book, blob, res["tokens"],
                             OUT / "videos" / f"{ic}_best_structured.mp4", cls["label"][:20])
            # hard_910101 short + one tilt-first
            if "hard_910101" in snaps:
                blob = pickle.loads((ROOT / snaps["hard_910101"]).read_bytes())
                render_video(env, book, blob, SHORT_BASE["hard_910101"],
                             OUT / "videos" / "hard_910101_short.mp4", "short")
                # representative tilt sequence
                render_video(env, book, blob, [6] * 12 + [12] * 10 + [0] * 5,
                             OUT / "videos" / "hard_910101_tilt6_mid_brake.mp4", "tilt_mid_brake")
        except Exception as exc:
            (OUT / "videos" / "VIDEO_ERROR.txt").write_text(str(exc), encoding="utf-8")
            print("VIDEO_ERROR", exc, flush=True)

    labels = {}
    for r in trial_rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    summary = {
        "seed": args.seed,
        "max_per_ic": args.max_per_ic,
        "n_ics": len(ic_audit),
        "n_trials": len(trial_rows),
        "labels": labels,
        "tilt_probe_helps_ics": [r["ic_id"] for r in ic_audit if r["tilt_probe_any_helps"]],
        "verified_cpp": labels.get("VERIFIED_CONTACT_PRESERVING_PIVOT", 0),
        "verified_single": labels.get("VERIFIED_SINGLE_CONTACT_PIVOT", 0),
        "phase1": p1["verified"],
        "bottleneck_candidates": [
            "insufficient_eccentricity_lever",
            "gravity_alignment_after_tilt",
            "grip_unload_causes_release",
            "only_3_grip_levels_in_K48",
            "search_coverage_bounded",
        ],
    }
    write_json(OUT / "summary.json", summary)
    print("PILOT_DONE", OUT, "labels", labels, flush=True)


if __name__ == "__main__":
    main()
