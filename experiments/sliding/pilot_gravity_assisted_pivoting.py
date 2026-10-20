"""Bounded gravity-assisted contact-preserving pivoting feasibility pilot.

Read-only on protected components. Writes:
  results/diagnostics/gravity_assisted_pivoting_feasibility/
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
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
from envs.contacts.frames import extract_contacts
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor
from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed, terminal_reason
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, object_vz, observe
from envs.simulation.capture import restore_capture
from envs.step_legal import N_SUB, TRACK, step_legal
from envs.ballistic_impact import z_tgt_of
from evaluation.grasp_configuration import (
    AXIS_REVERSAL_SYMMETRIC,
    cylinder_axis_hand,
    measure,
    orientation_error_rad,
    p_com_hand,
    success_new_components,
)
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from training.recovery_execution import _escaped
from training.unified_k48 import HANDOFF

OUT = ROOT / "results" / "diagnostics" / "gravity_assisted_pivoting_feasibility"
MIG = ROOT / "results" / "diagnostics" / "grasp_configuration_migration"
PILOT = ROOT / "results" / "diagnostics" / "nontrivial_recovery_ic_mining"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
LIMIT_S = float(SELECTED_LIMIT_S)
DT = 0.02
SEED = 20261018

# Provisional targets from migration audit (not final task freeze)
U_TGT = np.array([-0.013840181460089382, -0.0030378179260123783, -0.9998996054801703], float)
P_TGT = np.array([-0.001115724743759791, -6.904231962095219e-06, 0.10474686676582946], float)

# Candidate thresholds (explicit sensitivity — not a single silent tol)
THRESHOLDS = [
    {"name": "tight", "tol_theta_deg": 10.0, "tol_pos_mm": 10.0},
    {"name": "medium", "tol_theta_deg": 15.0, "tol_pos_mm": 15.0},
    {"name": "loose_migration", "tol_theta_deg": 20.531549744867547, "tol_pos_mm": 17.318534604178453},
]

CANDIDATE_IDS = [
    "hard_910101",
    "hrv1_920131",
    "hrv1_920099",
    "hrv1_920031",
    "hard_910172",
    "hrv1_920151",
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


def load_snapshot_map():
    m = {}
    for path in (PILOT / "IC_MANIFEST.csv", MIG / "EXISTING_22IC_REEVALUATION.csv"):
        if not path.is_file():
            continue
    man = list(csv.DictReader((PILOT / "IC_MANIFEST.csv").open(encoding="utf-8")))
    for r in man:
        m[r["id"]] = {"snapshot": r["snapshot"], "shortest_short": r.get("shortest_short")}
    ree = list(csv.DictReader((MIG / "EXISTING_22IC_REEVALUATION.csv").open(encoding="utf-8")))
    for r in ree:
        if r["ic_id"] in m:
            m[r["ic_id"]].update({
                "e_theta0_deg": float(r["e_theta0_deg"]),
                "e_pos0_mm": float(r["e_pos0_mm"]),
                "shortest_short": r["shortest_short"],
                "stability_only": int(r["stability_only"]),
            })
    return m


def parse_tokens(s):
    if not s or not str(s).strip():
        return []
    return list(json.loads(s))


def contact_summary(sim):
    rows, _agg = extract_contacts(sim)
    nL = sum(1 for r in rows if r["side"] == "L")
    nR = sum(1 for r in rows if r["side"] == "R")
    fnL = sum(abs(r["Fn"]) for r in rows if r["side"] == "L")
    fnR = sum(abs(r["Fn"]) for r in rows if r["side"] == "R")
    posL = [r["pos_h"] for r in rows if r["side"] == "L"]
    posR = [r["pos_h"] for r in rows if r["side"] == "R"]
    return {
        "nL": nL, "nR": nR, "fnL": fnL, "fnR": fnR,
        "bilateral": bool(nL > 0 and nR > 0),
        "any_contact": bool(nL + nR > 0),
        "posL_h": posL[:3], "posR_h": posR[:3],
        "n_contacts": len(rows),
    }


def gravity_torque_analysis(sim, u_tgt):
    """Gravitational torque about candidate pivot (bilateral contact centroid)."""
    mass = float(sim.model.body_mass[sim.ids.object_body])
    g_w = np.array(sim.model.opt.gravity, float)
    Fg = mass * g_w
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    cs = contact_summary(sim)
    rows, _ = extract_contacts(sim)
    if rows:
        pivot_w = np.mean([np.asarray(r["pos_w"], float) for r in rows], axis=0)
    else:
        # fallback: pinch along hand z at RH_Z_NOM
        pivot_w = ph + Rh @ np.array([0.0, 0.0, 0.105])
    r = po - pivot_w
    tau_g_w = np.cross(r, Fg)
    tau_g_h = Rh.T @ tau_g_w
    # Desired rotation that reduces e_theta: axis ~ u_c × u_tgt (with sign for abs symmetry)
    u_c = cylinder_axis_hand(sim)
    u_t = np.asarray(u_tgt, float)
    u_t = u_t / max(float(np.linalg.norm(u_t)), 1e-12)
    u_c = u_c / max(float(np.linalg.norm(u_c)), 1e-12)
    if float(np.dot(u_c, u_t)) < 0:
        u_t = -u_t
    desired_axis_h = np.cross(u_c, u_t)
    dn = float(np.linalg.norm(desired_axis_h))
    if dn < 1e-9:
        align = 0.0
        desired_axis_h = np.zeros(3)
    else:
        desired_axis_h = desired_axis_h / dn
        align = float(np.dot(tau_g_h, desired_axis_h))  # >0 means gravity torque aligns with error-reducing rotation
    e0 = orientation_error_rad(u_c, u_tgt, AXIS_REVERSAL_SYMMETRIC)
    # friction params from object geom
    og = int(sim.ids.object_geom)
    fric = np.asarray(sim.model.geom_friction[og], float)
    return {
        "mass_kg": mass,
        "g_w": g_w.tolist(),
        "g_h": (Rh.T @ g_w).tolist(),
        "p_com_h": (Rh.T @ (po - ph)).tolist(),
        "pivot_w": pivot_w.tolist(),
        "r_com_from_pivot_w": r.tolist(),
        "tau_g_w": tau_g_w.tolist(),
        "tau_g_h": tau_g_h.tolist(),
        "tau_g_h_norm": float(np.linalg.norm(tau_g_h)),
        "desired_error_reduce_axis_h": desired_axis_h.tolist(),
        "gravity_aligns_error_reduction": align,
        "gravity_helps": bool(align > 1e-6),
        "e_theta0_rad": e0,
        "e_theta0_deg": float(np.degrees(e0)),
        "contact": cs,
        "object_friction": fric.tolist(),
        "lever_arm_m": float(np.linalg.norm(r)),
    }


def step_legal_sublogged(sim, action, log_rows, token, u_tgt, p_tgt):
    """Replica of step_legal with per-MuJoCo-substep logging (does not edit step_legal.py)."""
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
        m = measure(sim, u_tgt, p_tgt)
        cs = contact_summary(sim)
        log_rows.append({
            "t": float(sim.data.time),
            "token": int(token),
            "sub": s,
            "action": action.tolist(),
            "tau_cmd": tau,
            "e_theta_deg": m["e_theta_deg"],
            "e_pos_mm": m["e_pos_mm"],
            "u_c_h": m["u_c_h"],
            "p_c_h": m["p_c_h"],
            "w_rel_h": m["w_rel_h"],
            "nL": cs["nL"], "nR": cs["nR"],
            "fnL": cs["fnL"], "fnR": cs["fnR"],
            "bilateral": cs["bilateral"],
            "any_contact": cs["any_contact"],
            "obj_z": m["obj_z"],
        })
    return action


def analyze_contact_topology(sub_log, flicker_max_ms=20.0):
    """Classify sustained release vs flicker from substep contact flags."""
    if not sub_log:
        return {
            "min_fingers": 0, "max_zero_streak_ms": 0.0, "max_single_streak_ms": 0.0,
            "frac_bilateral": 0.0, "frac_zero": 0.0, "frac_single": 0.0,
            "sustained_release": False, "sustained_single": False,
        }
    dt = float(sub_log[1]["t"] - sub_log[0]["t"]) if len(sub_log) > 1 else 0.002
    zero_streak = single_streak = 0.0
    max_zero = max_single = 0.0
    n_bi = n_z = n_s = 0
    for row in sub_log:
        n = int(row["nL"] > 0) + int(row["nR"] > 0)
        if n == 2:
            n_bi += 1
            zero_streak = single_streak = 0.0
        elif n == 0:
            n_z += 1
            zero_streak += dt
            single_streak = 0.0
            max_zero = max(max_zero, zero_streak)
        else:
            n_s += 1
            single_streak += dt
            zero_streak = 0.0
            max_single = max(max_single, single_streak)
    n = len(sub_log)
    return {
        "min_fingers": int(min((int(r["nL"] > 0) + int(r["nR"] > 0) for r in sub_log), default=0)),
        "max_zero_streak_ms": max_zero * 1e3,
        "max_single_streak_ms": max_single * 1e3,
        "frac_bilateral": n_bi / n,
        "frac_zero": n_z / n,
        "frac_single": n_s / n,
        "sustained_release": bool(max_zero * 1e3 > flicker_max_ms * 2),  # >40ms default
        "sustained_single": bool(max_single * 1e3 > 40.0),
        "n_substeps": n,
    }


def run_nominal_tracked(env, blob, u_tgt, p_tgt, hold_s=0.5):
    apply_handoff(env.sim, blob["ref"])
    t_h = float(env.sim.data.time)
    window_start = float(NOMINAL_S) - float(hold_s)
    pose_ok = {th["name"]: 0.0 for th in THRESHOLDS}
    pos_ok = {th["name"]: 0.0 for th in THRESHOLDS}
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    if latched(env.sim):
        reason = "TABLE_DROP"
    while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        t_prev = float(env.sim.data.time)
        nominal_step(env.sim)
        dt = float(env.sim.data.time) - t_prev
        elapsed = float(env.sim.data.time) - t_h
        row = _monitor(env.sim)
        end_row = row
        m = measure(env.sim, u_tgt, p_tgt)
        if elapsed >= window_start:
            for th in THRESHOLDS:
                if m["e_theta_deg"] <= th["tol_theta_deg"]:
                    pose_ok[th["name"]] += dt
                if m["e_pos_mm"] <= th["tol_pos_mm"]:
                    pos_ok[th["name"]] += dt
        target = float(z_tgt_of(env.sim))
        bilateral = _bilateral(row)
        command_reached = float(env.sim.fsm.p_des[2]) >= target - 1e-9
        dropped = _dropped(row) or _escaped(env.sim)
        if bilateral and command_reached:
            lift_done = True
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        if dropped:
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
    m_final = measure(env.sim, u_tgt, p_tgt)
    comps = {}
    for th in THRESHOLDS:
        comps[th["name"]] = success_new_components(
            reason == "SUCCESS_12S",
            np.deg2rad(m_final["e_theta_deg"]),
            m_final["e_pos_m"],
            np.deg2rad(th["tol_theta_deg"]),
            th["tol_pos_mm"] * 1e-3,
            pose_ok_duration_s=pose_ok[th["name"]],
            pos_ok_duration_s=pos_ok[th["name"]],
            hold_required_s=hold_s,
        )
    return {
        "reason": reason,
        "success_12s": reason == "SUCCESS_12S",
        "m_final": m_final,
        "pose_ok_s": pose_ok,
        "pos_ok_s": pos_ok,
        "components": comps,
        "t_handoff": t_h,
    }


def run_sequence(env, book, blob, tokens, u_tgt, p_tgt, log_substeps=False):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    m0 = measure(env.sim, u_tgt, p_tgt)
    lost_low = 0
    sub_log = []
    token_log = []
    for tok in tokens:
        pre = np.zeros(7, np.float32)
        _r, action = decode_action(book, int(tok), pre)
        mapped = map_action(action, np.asarray(env.sim.fsm.r_des, float), 
                            np.asarray(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float),
                            AIRBORNE_V3)
        if log_substeps:
            executed = step_legal_sublogged(env.sim, action, sub_log, int(tok), u_tgt, p_tgt)
        else:
            executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        m = measure(env.sim, u_tgt, p_tgt)
        cs = contact_summary(env.sim)
        token_log.append({
            "token": int(tok),
            "action": np.asarray(executed, float).tolist(),
            "codebook_row": np.asarray(book[int(tok)], float).tolist(),
            "tau": float(mapped["tau"]),
            "v_hand": mapped["v_hand"].tolist(),
            "w_body": mapped["w_body"].tolist(),
            "e_theta_deg": m["e_theta_deg"],
            "e_pos_mm": m["e_pos_mm"],
            "nL": cs["nL"], "nR": cs["nR"],
            "bilateral": cs["bilateral"],
        })
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            topo = analyze_contact_topology(sub_log) if sub_log else analyze_contact_topology([])
            return {
                "tokens": list(tokens),
                "success_12s": False,
                "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                "m0": m0,
                "m_mid": m,
                "m_final": m,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "dropped_before_handoff": True,
                "token_log": token_log,
                "sub_log": sub_log,
                "topology": topo,
                "nominal": None,
            }
    m_mid = measure(env.sim, u_tgt, p_tgt)
    recovery_elapsed = float(env.sim.data.time) - t0
    nominal = run_nominal_tracked(env, blob, u_tgt, p_tgt)
    topo = analyze_contact_topology(sub_log) if sub_log else None
    # If not sublogged, approximate topology from token_log contacts
    if topo is None:
        approx = []
        for i, tl in enumerate(token_log):
            # fabricate 1 sample per token (not substep — mark insufficient)
            approx.append({
                "t": t0 + (i + 1) * DT, "nL": tl["nL"], "nR": tl["nR"],
            })
        topo = analyze_contact_topology(approx)
        topo["substep_evidence"] = False
    else:
        topo["substep_evidence"] = True
    return {
        "tokens": list(tokens),
        "success_12s": bool(nominal["success_12s"]),
        "reason": nominal["reason"],
        "m0": m0,
        "m_mid": m_mid,
        "m_final": nominal["m_final"],
        "recovery_elapsed": recovery_elapsed,
        "dropped_before_handoff": False,
        "token_log": token_log,
        "sub_log": sub_log,
        "topology": topo,
        "nominal": nominal,
    }


def classify(result, short_baseline_final_theta, delta_min_deg=5.0):
    m0 = result["m0"]
    mm = result["m_mid"]
    mf = result["m_final"]
    d_mid = float(m0["e_theta_deg"] - mm["e_theta_deg"])
    d_fin = float(m0["e_theta_deg"] - mf["e_theta_deg"])
    topo = result["topology"] or {}
    s12 = result["success_12s"]
    sub_ok = bool(topo.get("substep_evidence"))

    if abs(d_mid) < 2.0 and abs(d_fin) < 2.0:
        label = "NO_MEANINGFUL_RELATIVE_ROTATION"
    elif d_mid >= delta_min_deg and not s12:
        label = "RELATIVE_ORIENTATION_CHANGE_BUT_UNSTABLE"
    elif s12 and mf["e_theta_deg"] > 15.0 and d_fin < 3.0:
        label = "STABLE_BUT_CONFIGURATION_INCORRECT"
    elif d_mid >= delta_min_deg and topo.get("sustained_release"):
        label = "VERIFIED_RELEASE_REGRASP" if (s12 and sub_ok) else (
            "AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE" if not sub_ok else "VERIFIED_RELEASE_REGRASP"
        )
    elif d_mid >= delta_min_deg and topo.get("sustained_single") and not topo.get("sustained_release"):
        if s12 and sub_ok and d_fin >= 3.0:
            label = "VERIFIED_SINGLE_CONTACT_PIVOT"
        elif not sub_ok:
            label = "AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE"
        else:
            label = "RELATIVE_ORIENTATION_CHANGE_BUT_UNSTABLE"
    elif d_mid >= delta_min_deg and topo.get("frac_bilateral", 0) >= 0.85 and not topo.get("sustained_release"):
        if s12 and sub_ok and d_fin >= 3.0:
            label = "VERIFIED_CONTACT_PRESERVING_PIVOT"
        elif not sub_ok:
            label = "AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE"
        else:
            label = "RELATIVE_ORIENTATION_CHANGE_BUT_UNSTABLE"
    else:
        label = "AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE"

    beats_short = bool(
        s12 and short_baseline_final_theta is not None
        and mf["e_theta_deg"] < float(short_baseline_final_theta) - 2.0
    )
    return {
        "label": label,
        "delta_theta_mid_deg": d_mid,
        "delta_theta_final_deg": d_fin,
        "beats_short_baseline_theta": beats_short,
        "substep_evidence": sub_ok,
    }


def build_proposals(book, rng, max_n=280):
    """Bounded legal token sequences (zero residual). Inspect full 7D rows."""
    rows = []
    for i in range(47):
        a = np.asarray(book[i], float)
        rows.append({
            "tok": i,
            "a": a,
            "a_grip": float(a[6]),
            "tau": float(grip_tau(a[6])),
            "v_norm": float(np.linalg.norm(a[:3])),
            "w_norm": float(np.linalg.norm(a[3:6])),
        })
    openish = [r for r in rows if r["a_grip"] < -0.5]
    closeish = [r for r in rows if r["a_grip"] > 0.5]
    # "mild" = least |grip| among all, or small wrist+open
    mild = sorted(rows, key=lambda r: abs(r["a_grip"]))[:8]
    soft_open = sorted(openish, key=lambda r: (r["v_norm"] + r["w_norm"], -r["a_grip"]))[:10]
    soft_close = sorted(closeish, key=lambda r: (r["v_norm"] + r["w_norm"], r["a_grip"]))[:10]
    # prefer pure-ish tokens
    almost_pure_open = [r for r in openish if r["v_norm"] < 0.15 and r["w_norm"] < 0.15] or soft_open
    almost_pure_close = [r for r in closeish if r["v_norm"] < 0.15 and r["w_norm"] < 0.15] or soft_close

    seqs = []
    # empty / known shorts added by caller
    for o in almost_pure_open[:6]:
        for n_o in (1, 2, 3, 5, 8, 12, 20):
            for c in almost_pure_close[:4]:
                for n_c in (1, 2, 3, 5, 8):
                    seqs.append([o["tok"]] * n_o + [c["tok"]] * n_c)
    for o in soft_open[:5]:
        for m in mild[:3]:
            for c in soft_close[:3]:
                seqs.append([o["tok"], o["tok"], m["tok"], m["tok"], c["tok"], c["tok"]])
                seqs.append([o["tok"]] * 3 + [m["tok"]] * 5 + [c["tok"]] * 3)
    # wrist-assisted while gripping (close tokens with nonzero w)
    wrist_close = sorted(closeish, key=lambda r: -r["w_norm"])[:6]
    for w in wrist_close:
        for n in (3, 5, 8, 12):
            seqs.append([w["tok"]] * n)
    for o in soft_open[:4]:
        for w in wrist_close[:3]:
            seqs.append([o["tok"]] * 2 + [w["tok"]] * 5 + [almost_pure_close[0]["tok"]] * 3)

    # random chains up to 2s (100 steps) but short mean length
    for _ in range(80):
        n = int(rng.integers(3, 25))
        phase = []
        # random open then close
        no = int(rng.integers(1, max(2, n // 2)))
        ot = int(rng.choice([r["tok"] for r in soft_open]))
        ct = int(rng.choice([r["tok"] for r in soft_close]))
        phase = [ot] * no + [ct] * (n - no)
        seqs.append(phase)

    # dedupe
    uniq, seen = [], set()
    for s in seqs:
        # cap horizon 2s
        s = s[:100]
        key = tuple(s)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
        if len(uniq) >= max_n:
            break
    rng.shuffle(uniq)
    return uniq[:max_n], {
        "n_openish": len(openish),
        "n_closeish": len(closeish),
        "almost_pure_open": [r["tok"] for r in almost_pure_open[:8]],
        "almost_pure_close": [r["tok"] for r in almost_pure_close[:8]],
        "note": "No token treated as pure grip without checking v/w residuals",
    }


def render_tokens_video(env, book, blob, tokens, dest, label, max_nominal_s=4.0):
    from visualization.video_recorder import write_frames_mp4
    from visualization.camera import mjv_camera_from_preset
    from visualization.overlays import caption

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
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        now = float(env.sim.data.time)
        if now + 1e-9 >= next_frame:
            cs = contact_summary(env.sim)
            m = measure(env.sim, U_TGT, P_TGT)
            grab(f"{label} t={now-t0:.2f} tok={tok} n={cs['nL']}/{cs['nR']} eθ={m['e_theta_deg']:.1f}")
            next_frame += 0.04
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
        while float(env.sim.data.time) - t_h < max_nominal_s - 1e-9:
            nominal_step(env.sim)
            now = float(env.sim.data.time)
            if now + 1e-9 >= next_frame:
                m = measure(env.sim, U_TGT, P_TGT)
                grab(f"nom {now-t_h:.2f}s eθ={m['e_theta_deg']:.1f}")
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
    parser.add_argument("--max-traj-per-ic", type=int, default=200)
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
    snap_map = load_snapshot_map()
    proposals, prop_meta = build_proposals(book, rng, max_n=args.max_traj_per_ic)
    write_json(OUT / "RAW" / "proposal_meta.json", prop_meta)

    cand_rows = []
    search_rows = []
    promising = []
    verified_any = False

    for ic in CANDIDATE_IDS:
        meta = snap_map.get(ic)
        if not meta:
            print("SKIP_MISSING", ic, flush=True)
            continue
        blob = pickle.loads((ROOT / meta["snapshot"]).read_bytes())
        if "capture" not in blob:
            print("SKIP_NO_CAPTURE", ic, flush=True)
            continue
        print("IC", ic, flush=True)
        restore_capture(env, blob["capture"])
        grav = gravity_torque_analysis(env.sim, U_TGT)
        m0 = measure(env.sim, U_TGT, P_TGT)
        short = parse_tokens(meta.get("shortest_short"))

        # Counterfactual baselines (always)
        baselines = {
            "immediate": [],
            "short_known": short,
            "tok_44": [44],
            "tok_30": [30],
            "tok_41": [41],
            "tok_30_30": [30, 30],
            "tok_44_0": [44, 0],
            "tok_30_0_0": [30, 0, 0],
        }
        base_results = {}
        short_final_theta = None
        for name, toks in baselines.items():
            # log substeps only for short_known and a couple opens
            log_sub = name in ("short_known", "tok_30_0_0", "tok_44_0")
            res = run_sequence(env, book, blob, toks, U_TGT, P_TGT, log_substeps=log_sub)
            cls = classify(res, None)
            base_results[name] = {**{k: res[k] for k in (
                "success_12s", "reason", "recovery_elapsed", "tokens", "topology"
            )}, "m0": res["m0"], "m_mid": res["m_mid"], "m_final": res["m_final"],
                "classification": cls,
                "components_loose": None if not res["nominal"] else res["nominal"]["components"]["loose_migration"],
            }
            if name == "short_known":
                short_final_theta = res["m_final"]["e_theta_deg"]
            search_rows.append({
                "ic_id": ic, "branch": f"baseline_{name}",
                "tokens": json.dumps(toks), "n_tokens": len(toks),
                "success_12s": int(res["success_12s"]), "reason": res["reason"],
                "e_theta0": res["m0"]["e_theta_deg"], "e_theta_mid": res["m_mid"]["e_theta_deg"],
                "e_theta_final": res["m_final"]["e_theta_deg"],
                "e_pos0_mm": res["m0"]["e_pos_mm"], "e_pos_final_mm": res["m_final"]["e_pos_mm"],
                "delta_theta_mid": cls["delta_theta_mid_deg"],
                "delta_theta_final": cls["delta_theta_final_deg"],
                "label": cls["label"],
                "beats_short": int(cls["beats_short_baseline_theta"]),
                "substep": int(bool(res["topology"].get("substep_evidence"))),
                "frac_bilateral": res["topology"].get("frac_bilateral"),
                "max_zero_ms": res["topology"].get("max_zero_streak_ms"),
                "gravity_helps": int(grav["gravity_helps"]),
            })
            if log_sub and res["sub_log"]:
                write_json(OUT / "RAW" / f"{ic}_{name}_sublog.json", {
                    "tokens": toks, "sub_log": res["sub_log"], "topology": res["topology"],
                })

        cand_rows.append({
            "ic_id": ic,
            "snapshot": meta["snapshot"],
            "e_theta0_deg": m0["e_theta_deg"],
            "e_pos0_mm": m0["e_pos_mm"],
            "p_com_h": json.dumps(m0["p_c_h"]),
            "u_c_h": json.dumps(m0["u_c_h"]),
            "nL0": grav["contact"]["nL"], "nR0": grav["contact"]["nR"],
            "bilateral0": int(grav["contact"]["bilateral"]),
            "tau_g_h_norm": grav["tau_g_h_norm"],
            "gravity_aligns": grav["gravity_aligns_error_reduction"],
            "gravity_helps": int(grav["gravity_helps"]),
            "lever_arm_m": grav["lever_arm_m"],
            "imm_success_12s": int(base_results["immediate"]["success_12s"]),
            "short_tokens": json.dumps(short),
            "short_success_12s": int(base_results["short_known"]["success_12s"]),
            "short_e_theta_final": base_results["short_known"]["m_final"]["e_theta_deg"],
            "stability_only_prior": meta.get("stability_only", ""),
            "object_friction": json.dumps(grav["object_friction"]),
        })
        write_json(OUT / "RAW" / f"{ic}_gravity.json", grav)
        write_json(OUT / "RAW" / f"{ic}_baselines.json", base_results)

        # Bounded search
        n_eval = 0
        best = None
        for seq in proposals:
            if n_eval >= args.max_traj_per_ic:
                break
            # cheap pass first without substeps
            res = run_sequence(env, book, blob, seq, U_TGT, P_TGT, log_substeps=False)
            n_eval += 1
            cls = classify(res, short_final_theta)
            d_mid = cls["delta_theta_mid_deg"]
            interesting = (
                d_mid >= 4.0
                or (res["success_12s"] and res["m_final"]["e_theta_deg"] < float(short_final_theta) - 2.0)
            )
            if interesting:
                # re-run with substep evidence
                res2 = run_sequence(env, book, blob, seq, U_TGT, P_TGT, log_substeps=True)
                n_eval += 1
                cls2 = classify(res2, short_final_theta)
                res, cls = res2, cls2
                write_json(OUT / "RAW" / f"{ic}_promising_{n_eval}.json", {
                    "tokens": seq, "classification": cls, "topology": res["topology"],
                    "m0": res["m0"], "m_mid": res["m_mid"], "m_final": res["m_final"],
                    "success_12s": res["success_12s"], "reason": res["reason"],
                    "token_log": res["token_log"],
                    "sub_log_len": len(res["sub_log"]),
                    "components": None if not res["nominal"] else res["nominal"]["components"],
                })
                # prefix handoff counterfactuals for promising
                prefix_rows = []
                for k in sorted(set([0, max(1, len(seq) // 4), max(1, len(seq) // 2), len(seq) - 1, len(seq)])):
                    if k < 0:
                        continue
                    pref = seq[:k]
                    pr = run_sequence(env, book, blob, pref, U_TGT, P_TGT, log_substeps=False)
                    n_eval += 1
                    prefix_rows.append({
                        "prefix_len": k,
                        "success_12s": pr["success_12s"],
                        "e_theta_final": pr["m_final"]["e_theta_deg"],
                        "reason": pr["reason"],
                    })
                promising.append({
                    "ic_id": ic, "tokens": seq, "cls": cls, "res_summary": {
                        "success_12s": res["success_12s"],
                        "e0": res["m0"]["e_theta_deg"],
                        "emid": res["m_mid"]["e_theta_deg"],
                        "ef": res["m_final"]["e_theta_deg"],
                        "label": cls["label"],
                        "beats_short": cls["beats_short_baseline_theta"],
                    },
                    "prefix_cf": prefix_rows,
                })
                if cls["label"] == "VERIFIED_CONTACT_PRESERVING_PIVOT":
                    verified_any = True
                    best = (seq, res, cls)
                    # early stop this IC
                    search_rows.append({
                        "ic_id": ic, "branch": "search_verified_stop",
                        "tokens": json.dumps(seq), "n_tokens": len(seq),
                        "success_12s": int(res["success_12s"]), "reason": res["reason"],
                        "e_theta0": res["m0"]["e_theta_deg"], "e_theta_mid": res["m_mid"]["e_theta_deg"],
                        "e_theta_final": res["m_final"]["e_theta_deg"],
                        "e_pos0_mm": res["m0"]["e_pos_mm"], "e_pos_final_mm": res["m_final"]["e_pos_mm"],
                        "delta_theta_mid": cls["delta_theta_mid_deg"],
                        "delta_theta_final": cls["delta_theta_final_deg"],
                        "label": cls["label"],
                        "beats_short": int(cls["beats_short_baseline_theta"]),
                        "substep": 1,
                        "frac_bilateral": res["topology"].get("frac_bilateral"),
                        "max_zero_ms": res["topology"].get("max_zero_streak_ms"),
                        "gravity_helps": int(grav["gravity_helps"]),
                        "n_eval_at_stop": n_eval,
                    })
                    break

            search_rows.append({
                "ic_id": ic, "branch": "search",
                "tokens": json.dumps(seq), "n_tokens": len(seq),
                "success_12s": int(res["success_12s"]), "reason": res["reason"],
                "e_theta0": res["m0"]["e_theta_deg"], "e_theta_mid": res["m_mid"]["e_theta_deg"],
                "e_theta_final": res["m_final"]["e_theta_deg"],
                "e_pos0_mm": res["m0"]["e_pos_mm"], "e_pos_final_mm": res["m_final"]["e_pos_mm"],
                "delta_theta_mid": cls["delta_theta_mid_deg"],
                "delta_theta_final": cls["delta_theta_final_deg"],
                "label": cls["label"],
                "beats_short": int(cls["beats_short_baseline_theta"]),
                "substep": int(bool(res["topology"].get("substep_evidence"))),
                "frac_bilateral": res["topology"].get("frac_bilateral"),
                "max_zero_ms": res["topology"].get("max_zero_streak_ms"),
                "gravity_helps": int(grav["gravity_helps"]),
            })

            if best is None and interesting and res["success_12s"] and cls["delta_theta_final_deg"] > 3:
                best = (seq, res, cls)

        write_json(OUT / "RAW" / f"{ic}_search_budget.json", {
            "n_eval": n_eval, "max": args.max_traj_per_ic,
            "verified_cpp": verified_any and best is not None,
        })
        print(f"DONE_IC {ic} eval={n_eval} promising={sum(1 for p in promising if p['ic_id']==ic)}", flush=True)
        if verified_any:
            # continue other ICs briefly? task says stop early if sufficient evidence —
            # still finish remaining candidates with lighter budget
            pass

    write_csv(OUT / "CANDIDATE_IC_AUDIT.csv", cand_rows)
    write_csv(OUT / "LEGAL_SEARCH_SUMMARY.csv", search_rows)
    write_json(OUT / "RAW" / "promising.json", promising)

    # Videos: best improving SUCCESS_12S and a clear failure/short baseline
    if not args.skip_videos:
        try:
            # pick best beats_short or largest delta among promising with S12
            scored = []
            for p in promising:
                if p["res_summary"]["success_12s"]:
                    scored.append(p)
            scored.sort(key=lambda p: (-float(p["cls"]["delta_theta_final_deg"]),))
            if scored:
                p = scored[0]
                blob = pickle.loads((ROOT / snap_map[p["ic_id"]]["snapshot"]).read_bytes())
                render_tokens_video(
                    env, book, blob, p["tokens"],
                    OUT / "videos" / f"{p['ic_id']}_best_attempt.mp4",
                    label=p["cls"]["label"][:18],
                )
            # short baseline on hard_910101
            if "hard_910101" in snap_map:
                blob = pickle.loads((ROOT / snap_map["hard_910101"]["snapshot"]).read_bytes())
                short = parse_tokens(snap_map["hard_910101"].get("shortest_short"))
                render_tokens_video(
                    env, book, blob, short,
                    OUT / "videos" / "hard_910101_short_baseline.mp4",
                    label="short_base",
                )
                render_tokens_video(
                    env, book, blob, [],
                    OUT / "videos" / "hard_910101_immediate_handoff.mp4",
                    label="immediate",
                )
        except Exception as exc:
            (OUT / "videos" / "VIDEO_ERROR.txt").write_text(str(exc), encoding="utf-8")
            print("VIDEO_ERROR", exc, flush=True)

    write_json(OUT / "summary.json", {
        "seed": args.seed,
        "n_candidates": len(cand_rows),
        "n_search_rows": len(search_rows),
        "n_promising": len(promising),
        "labels": {lab: sum(1 for r in search_rows if r["label"] == lab)
                   for lab in sorted(set(r["label"] for r in search_rows))},
        "n_beats_short": sum(1 for r in search_rows if int(r.get("beats_short") or 0)),
        "n_gravity_helps": sum(1 for r in cand_rows if int(r["gravity_helps"])),
        "verified_contact_preserving_pivot": any(
            r["label"] == "VERIFIED_CONTACT_PRESERVING_PIVOT" for r in search_rows
        ),
        "u_target_h": U_TGT.tolist(),
        "p_target_h": P_TGT.tolist(),
        "thresholds": THRESHOLDS,
        "max_traj_per_ic": args.max_traj_per_ic,
        "sources": {
            "metrics": "evaluation/grasp_configuration.py",
            "contacts": "envs/contacts/frames.py::extract_contacts",
            "step": "envs/step_legal.py (+ diagnostic substep replica)",
            "action_map": "controllers/recovery/action_map.py",
            "handoff": "envs/outcomes/horizon_outcome.py / recovery_success.py",
        },
    })
    print("PILOT_DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
