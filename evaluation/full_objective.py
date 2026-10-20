"""Standalone full-objective evaluator for grasp-recovery expert search.

Does NOT modify SUCCESS_12S, reward_v2, physics, or the nominal controller.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from envs.ballistic_impact import z_tgt_of
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.grasp_configuration import measure, success_new_components
from training.k47_codebook import decode_action
from training.recovery_execution import _escaped

# Calibrated targets (migration audit)
U_TGT = np.array([-0.013840181460089382, -0.0030378179260123783, -0.9998996054801703], float)
P_TGT = np.array([-0.001115724743759791, -6.904231962095219e-06, 0.10474686676582946], float)

# Fixed working acceptance (see EXPERT_GENERATION_PROTOCOL.md)
WORKING_THETA_DEG = 10.0
WORKING_POS_MM = 12.0
FINAL_WINDOW_S = 2.0
HOLD_S = 0.5

THETA_GRID = [5.0, 10.0, 15.0, 20.5]
POS_GRID = [5.0, 10.0, 17.0]


def contact_summary(sim) -> dict:
    from envs.contacts.frames import extract_contacts

    rows, _ = extract_contacts(sim)
    nL = sum(1 for r in rows if r["side"] == "L")
    nR = sum(1 for r in rows if r["side"] == "R")
    return {"nL": nL, "nR": nR, "bilateral": bool(nL > 0 and nR > 0), "any": bool(nL + nR > 0)}


def window_stats(samples: list[dict], window_start: float, hold_s: float = HOLD_S) -> dict:
    win = [s for s in samples if s["t_rel"] + 1e-9 >= window_start]
    empty = {
        "n": 0,
        "mean_theta": None,
        "max_theta": None,
        "mean_pos": None,
        "max_pos": None,
        "frac_bilateral": None,
        "longest_both_ok_s": 0.0,
        "matrix": {},
        "working_SUCCESS_NEW_hold": False,
    }
    if not win:
        return empty
    thetas = np.array([s["e_theta_deg"] for s in win], float)
    poss = np.array([s["e_pos_mm"] for s in win], float)
    bil = np.array([1.0 if s["bilateral"] else 0.0 for s in win], float)
    dt = 0.02
    if len(win) > 1:
        dts = np.diff([s["t_rel"] for s in win])
        if len(dts):
            dt = float(np.median(dts))

    def longest_ok(mask) -> float:
        longest = cur = 0.0
        for v in mask:
            if v:
                cur += dt
                longest = max(longest, cur)
            else:
                cur = 0.0
        return float(longest)

    matrix = {}
    for th, pm in product(THETA_GRID, POS_GRID):
        ok = (thetas <= th) & (poss <= pm)
        lo = longest_ok(ok)
        matrix[f"th{th}_pos{pm}"] = {
            "frac_frames": float(ok.mean()),
            "longest_ok_s": lo,
            "SUCCESS_NEW": bool(lo + 1e-9 >= hold_s),
        }
    work_ok = (thetas <= WORKING_THETA_DEG) & (poss <= WORKING_POS_MM)
    work_lo = longest_ok(work_ok)
    return {
        "n": len(win),
        "mean_theta": float(thetas.mean()),
        "max_theta": float(thetas.max()),
        "mean_pos": float(poss.mean()),
        "max_pos": float(poss.max()),
        "frac_bilateral": float(bil.mean()),
        "longest_both_ok_s": work_lo,
        "matrix": matrix,
        "working_SUCCESS_NEW_hold": bool(work_lo + 1e-9 >= hold_s),
    }


def classify_outcome(s12: bool, win: dict) -> str:
    if not s12:
        return "FAILED_OR_DROPPED"
    if not win or win.get("mean_theta") is None:
        return "UNVERIFIED"
    mat = win.get("matrix") or {}
    key = f"th{WORKING_THETA_DEG}_pos{WORKING_POS_MM}"
    # Prefer explicit working matrix entry; fall back to working hold flag
    full = bool(win.get("working_SUCCESS_NEW_hold"))
    if key in mat:
        full = bool(mat[key]["SUCCESS_NEW"])
    if full:
        return "VERIFIED_FULL_EXPERT"
    pose_ok = float(win["mean_theta"]) <= WORKING_THETA_DEG and float(win["max_theta"]) <= WORKING_THETA_DEG * 1.25
    pos_ok = float(win["mean_pos"]) <= WORKING_POS_MM and float(win["max_pos"]) <= WORKING_POS_MM * 1.25
    # Use hold-aware pose/pos from matrix edges
    pose_hold = bool(mat.get(f"th{WORKING_THETA_DEG}_pos{17.0}", {}).get("SUCCESS_NEW"))
    pos_hold = bool(mat.get(f"th{20.5}_pos{WORKING_POS_MM}", {}).get("SUCCESS_NEW"))
    if pose_hold and not pos_hold:
        return "POSE_ONLY"
    if pos_hold and not pose_hold:
        return "POSITION_ONLY"
    if pose_ok and not pos_ok:
        return "POSE_ONLY"
    if pos_ok and not pose_ok:
        return "POSITION_ONLY"
    return "STABILITY_ONLY"


def evaluate_tokens(
    env,
    book,
    blob: dict,
    tokens: list[int],
    *,
    record_obs: bool = False,
    sample_nominal_dt: float = 0.02,
    from_horizon_outcome=None,
) -> dict[str, Any]:
    """Restore IC, run legal tokens, HANDOFF, full nominal, return full-objective report."""
    from envs.outcomes.horizon_outcome import recovery_failed, terminal_reason

    restore_capture(env, blob["capture"])
    # Sync policy window from capture if present
    if blob["capture"].get("prev") is not None:
        env._prev = np.asarray(blob["capture"]["prev"], np.float32)
    t0 = float(env.sim.data.time)
    m0 = measure(env.sim, U_TGT, P_TGT)
    lost_low = 0
    observations = []
    actions = []
    contacts = []
    residuals = []

    for tok in tokens:
        if record_obs:
            obs = np.asarray(env.observation(), np.float32).reshape(39)
            observations.append(obs.copy())
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        action = np.asarray(step_legal(env.sim, action), np.float32)
        env._prev = action.copy()
        if record_obs:
            actions.append(action.copy())
            residuals.append(np.zeros(7, np.float32))
            cs = contact_summary(env.sim)
            contacts.append(
                {
                    "t": float(env.sim.data.time),
                    "nL": cs["nL"],
                    "nR": cs["nR"],
                    "z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                }
            )
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            m = measure(env.sim, U_TGT, P_TGT)
            return {
                "tokens": list(tokens),
                "success_12s": False,
                "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                "m0": m0,
                "m_handoff": m,
                "m_final": m,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "window": window_stats([], float(NOMINAL_S) - FINAL_WINDOW_S),
                "label": "FAILED_OR_DROPPED",
                "SUCCESS_NEW": False,
                "observations": observations,
                "actions": actions,
                "contacts": contacts,
                "residuals": residuals,
                "handoff_obs": None,
            }

    m_hand = measure(env.sim, U_TGT, P_TGT)
    recovery_elapsed = float(env.sim.data.time) - t0
    handoff_obs = None
    if record_obs:
        handoff_obs = np.asarray(env.observation(), np.float32).reshape(39).copy()

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
        nominal_step(env.sim)
        t_rel = float(env.sim.data.time) - t_h
        if t_rel + 1e-12 >= next_sample:
            m = measure(env.sim, U_TGT, P_TGT)
            cs = contact_summary(env.sim)
            samples.append(
                {
                    "t_rel": t_rel,
                    "e_theta_deg": m["e_theta_deg"],
                    "e_pos_mm": m["e_pos_mm"],
                    "bilateral": cs["bilateral"],
                    "obj_z": m["obj_z"],
                }
            )
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
    if not samples or samples[-1]["t_rel"] < float(env.sim.data.time) - t_h - 1e-6:
        samples.append(
            {
                "t_rel": float(env.sim.data.time) - t_h,
                "e_theta_deg": m_final["e_theta_deg"],
                "e_pos_mm": m_final["e_pos_mm"],
                "bilateral": contact_summary(env.sim)["bilateral"],
                "obj_z": m_final["obj_z"],
            }
        )
    win = window_stats(samples, float(NOMINAL_S) - FINAL_WINDOW_S)
    s12 = reason == "SUCCESS_12S"
    label = classify_outcome(s12, win)
    components = success_new_components(
        s12,
        float(np.radians(win["mean_theta"] or 180.0)),
        float((win["mean_pos"] or 1e3) * 1e-3),
        float(np.radians(WORKING_THETA_DEG)),
        float(WORKING_POS_MM * 1e-3),
        pose_ok_duration_s=float(win.get("longest_both_ok_s") or 0.0),
        pos_ok_duration_s=float(win.get("longest_both_ok_s") or 0.0),
        hold_required_s=HOLD_S,
    )
    # Override SUCCESS_NEW with hold-aware working matrix (mean alone is insufficient)
    success_new = bool(s12 and win.get("working_SUCCESS_NEW_hold"))
    components["SUCCESS_NEW"] = success_new
    if success_new:
        label = "VERIFIED_FULL_EXPERT"

    return {
        "tokens": list(tokens),
        "success_12s": s12,
        "reason": reason,
        "m0": m0,
        "m_handoff": m_hand,
        "m_final": m_final,
        "recovery_elapsed": recovery_elapsed,
        "window": win,
        "label": label,
        "SUCCESS_NEW": success_new,
        "components": components,
        "observations": observations,
        "actions": actions,
        "contacts": contacts,
        "residuals": residuals,
        "handoff_obs": handoff_obs,
        "d_theta_mid": float(m0["e_theta_deg"] - m_hand["e_theta_deg"]),
        "d_pos_mid_mm": float(m0["e_pos_mm"] - m_hand["e_pos_mm"]),
    }


def guide_score_after_prefix(env, book, blob, tokens) -> dict:
    """Cheap mid-recovery score (no HANDOFF). Heuristic only."""
    from envs.outcomes.horizon_outcome import recovery_failed, terminal_reason

    restore_capture(env, blob["capture"])
    lost_low = 0
    for tok in tokens:
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "ok": False,
                "reason": terminal_reason(env.sim) or "DROP",
                "e_theta": 180.0,
                "e_pos": 1e3,
                "score": -1e6,
                "bilateral": False,
            }
    m = measure(env.sim, U_TGT, P_TGT)
    cs = contact_summary(env.sim)
    # Lower errors better; bilateral bonus; mild penalty for extreme duration later
    score = -float(m["e_theta_deg"]) - 0.4 * float(m["e_pos_mm"]) + (5.0 if cs["bilateral"] else -20.0)
    return {
        "ok": True,
        "reason": None,
        "e_theta": float(m["e_theta_deg"]),
        "e_pos": float(m["e_pos_mm"]),
        "score": score,
        "bilateral": cs["bilateral"],
        "m": m,
    }
