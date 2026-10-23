"""Plan A sliding/pivot mechanism gates and traced rollouts.

Authoritative shared helpers for paired-scene regression. Historical audit
scripts re-export from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from envs.contacts.frames import extract_contacts
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.support_drop import arm_recovery_drop, latched, terminal_reason
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.hand_frame_kinematics import sample_kinematics
from evaluation.success_criteria import (
    FINAL_WINDOW_S,
    HOLD_S,
    NOMINAL_S,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    window_stats,
)
from training.action_vocab_k49 import HANDOFF, build_k49_table, decode_token
from training.k47_codebook import load_codebook
from training.recovery_execution import _escaped

ROOT = Path(__file__).resolve().parents[1]

SLIDE_AXIAL_MM_MIN = 3.0
PIVOT_AXIS_DEG_MIN = 5.0
CO_MOTION_RATIO_MAX = 0.35  # |Δp_rel| / max(|Δp_hand|,|Δp_obj|) for co-motion flag
BILATERAL_GAP_S_REPORT = 0.02


@dataclass
class ExpertSpec:
    ic_id: str
    root_id: str
    source: str
    snap: Path
    tokens_npz: Path
    meta: Path | None
    claimed_mechanisms: str
    config_id: str = "K49_tok48_wy0p25"
    priority: int = 0


def load_vocab(config_id: str = "K49_tok48_wy0p25"):
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=config_id)
    return table, book48


def analyze_mechanism(trace: list[dict]) -> dict[str, Any]:
    """Classify genuine sliding / pivoting / co-motion from a kinematics trace."""
    if len(trace) < 2:
        return {
            "genuine_axial_sliding": "UNVERIFIED",
            "genuine_pivoting": "UNVERIFIED",
            "contact_preservation": "UNVERIFIED",
            "bilateral_gap_s": 0.0,
            "dp_axial_mm": 0.0,
            "d_axis_deg": 0.0,
            "co_motion": False,
            "notes": "trace_too_short",
        }

    bil = np.array([bool(r["bilateral"]) for r in trace], bool)
    # gaps in bilateral contact
    gap_s = 0.0
    if len(trace) >= 2:
        for i in range(1, len(trace)):
            if not bil[i]:
                gap_s += max(0.0, float(trace[i]["t"] - trace[i - 1]["t"]))

    # Prefer bilateral windows for mechanism measurement
    idx = np.where(bil)[0]
    if len(idx) < 2:
        return {
            "genuine_axial_sliding": "FAIL",
            "genuine_pivoting": "FAIL",
            "contact_preservation": "FAIL",
            "bilateral_gap_s": gap_s,
            "dp_axial_mm": 0.0,
            "d_axis_deg": 0.0,
            "co_motion": False,
            "notes": "insufficient_bilateral_contact",
        }

    i0, i1 = int(idx[0]), int(idx[-1])
    r0, r1 = trace[i0], trace[i1]
    # Axial relative COM displacement along initial cylinder axis in hand frame
    u0 = np.asarray(r0["u_obj_h"], float)
    dp_rel = np.asarray(r1["p_rel"], float) - np.asarray(r0["p_rel"], float)
    dp_axial = float(np.dot(dp_rel, u0))
    dp_trans = float(np.linalg.norm(dp_rel - dp_axial * u0))
    # Contact axial migration
    s0, s1 = r0.get("contact_s_mean"), r1.get("contact_s_mean")
    ds_contact = float(s1 - s0) if (np.isfinite(s0) and np.isfinite(s1)) else float("nan")

    d_hand = np.asarray(r1["p_hand"], float) - np.asarray(r0["p_hand"], float)
    d_obj = np.asarray(r1["p_obj"], float) - np.asarray(r0["p_obj"], float)
    hand_move = float(np.linalg.norm(d_hand))
    obj_move = float(np.linalg.norm(d_obj))
    rel_world = float(np.linalg.norm(d_obj - d_hand))
    co_motion = bool(
        max(hand_move, obj_move) > 1e-4
        and rel_world / max(max(hand_move, obj_move), 1e-6) < CO_MOTION_RATIO_MAX
        and abs(dp_axial) * 1e3 < SLIDE_AXIAL_MM_MIN
    )

    d_axis = abs(float(r1["axis_err_deg"] - r0["axis_err_deg"]))
    # Also track absolute change of u_obj_h direction
    u1 = np.asarray(r1["u_obj_h"], float)
    d_u = float(np.degrees(np.arccos(np.clip(abs(np.dot(u0, u1)), 0.0, 1.0))))

    contact_pres = "PASS" if gap_s < BILATERAL_GAP_S_REPORT else "FAIL"
    if gap_s >= BILATERAL_GAP_S_REPORT:
        slide_v = "FAIL"
        pivot_v = "FAIL"
        notes = f"bilateral_gap_s={gap_s:.3f}"
    else:
        notes = ""
        # Sliding: meaningful axial relative COM (or contact s) while bilateral
        axial_mm = abs(dp_axial) * 1e3
        contact_mm = abs(ds_contact) * 1e3 if np.isfinite(ds_contact) else 0.0
        if co_motion:
            slide_v = "FAIL"
            notes += ";rigid_co_motion"
        elif max(axial_mm, contact_mm) >= SLIDE_AXIAL_MM_MIN and axial_mm >= dp_trans * 1e3 * 0.5:
            slide_v = "PASS"
        else:
            slide_v = "FAIL"
            notes += f";axial_mm={axial_mm:.2f}"

        if max(d_axis, d_u) >= PIVOT_AXIS_DEG_MIN:
            pivot_v = "PASS"
        else:
            pivot_v = "FAIL"
            notes += f";d_axis={d_axis:.2f}"

    return {
        "genuine_axial_sliding": slide_v,
        "genuine_pivoting": pivot_v,
        "contact_preservation": contact_pres,
        "bilateral_gap_s": gap_s,
        "dp_axial_mm": float(dp_axial * 1e3),
        "dp_trans_mm": float(dp_trans * 1e3),
        "ds_contact_mm": float(ds_contact * 1e3) if np.isfinite(ds_contact) else float("nan"),
        "d_axis_err_deg": d_axis,
        "d_u_obj_h_deg": d_u,
        "hand_move_mm": hand_move * 1e3,
        "obj_move_mm": obj_move * 1e3,
        "rel_world_mm": rel_world * 1e3,
        "co_motion": co_motion,
        "frac_bilateral": float(bil.mean()),
        "notes": notes.lstrip(";") or "ok",
    }


def audit_action_legality(
    tokens: np.ndarray,
    residuals: np.ndarray | None,
    table: np.ndarray,
    book48: np.ndarray,
) -> dict[str, Any]:
    issues = []
    actions = []
    for i, tok in enumerate(tokens):
        tok = int(tok)
        if tok == HANDOFF:
            issues.append(f"step{i}:HANDOFF_in_motion_tape")
            continue
        if tok < 0 or tok > 48:
            issues.append(f"step{i}:token_out_of_range_{tok}")
            continue
        res = None if residuals is None else residuals[i]
        try:
            a = decode_token(table, book48, tok, res)
        except Exception as exc:
            issues.append(f"step{i}:decode_fail:{exc}")
            continue
        if np.any(np.abs(a) > 1.0 + 1e-6):
            issues.append(f"step{i}:action_out_of_bounds")
        actions.append(a)
    return {
        "n_tokens": int(len(tokens)),
        "n_decoded": len(actions),
        "all_in_bounds": int(len(issues) == 0 and len(actions) == len(tokens)),
        "issues": ";".join(issues) if issues else "",
        "actions": actions,
        "verdict": "PASS" if not issues and len(actions) == len(tokens) else "FAIL",
    }


def rollout_traced(
    env,
    blob: dict,
    tokens: np.ndarray,
    table: np.ndarray,
    book48: np.ndarray,
    residuals: np.ndarray | None = None,
    *,
    arm_drop: bool = False,
    action_override: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Restore snapshot, execute legal Recovery tape, HANDOFF, nominal; collect traces."""
    restore_capture(env, blob["capture"])
    if arm_drop:
        arm_recovery_drop(env.sim)
    if blob["capture"].get("prev") is not None:
        env._prev = np.asarray(blob["capture"]["prev"], np.float32)

    # Snapshot integrity: record p_des vs hand at t0
    sim = env.sim
    p_hand0 = np.array(sim.data.xpos[sim.ids.hand_body], float)
    p_des0 = np.asarray(sim.fsm.p_des, float)
    target_mismatch = float(np.linalg.norm(p_hand0 - p_des0))

    # Illegal-edit detectors: watch qpos object freejoint address uniqueness via hash each step mid-recovery only via step_legal
    qadr = int(sim.model.jnt_qposadr[sim.ids.object_jnt])

    legality = audit_action_legality(tokens, residuals, table, book48)
    actions = legality["actions"] if action_override is None else action_override
    if action_override is not None:
        legality = {
            **legality,
            "verdict": "PASS" if all(np.all(np.abs(a) <= 1.0 + 1e-6) for a in actions) else "FAIL",
            "note": "negative_control_override",
        }

    trace = [sample_kinematics(sim)]
    contact_rows = []
    t0 = float(sim.data.time)
    lost_low = 0
    failed_reason = None
    qpos_edits = 0

    for i, action in enumerate(actions):
        q_before = np.array(sim.data.qpos[qadr : qadr + 7], float).copy()
        step_legal(sim, action)
        # step_legal integrates physics; object freejoint may change — that is expected.
        # Flag only NaNs / explosions.
        if not np.all(np.isfinite(sim.data.qpos)):
            qpos_edits += 1
            failed_reason = "NONFINITE_QPOS"
            break
        env._prev = np.asarray(action, np.float32)
        kin = sample_kinematics(sim)
        kin["tok"] = int(tokens[i]) if i < len(tokens) else -1
        kin["action"] = np.asarray(action, float).copy()
        trace.append(kin)
        rows, _ = extract_contacts(sim)
        for r in rows:
            contact_rows.append({
                "t": kin["t"],
                "step": i,
                "side": r.get("side"),
                "Fn": r.get("Fn", r.get("fn")),
                "pos_w": r.get("pos_w", r.get("pos")),
            })
        failed, lost_low = recovery_failed(sim, blob["ref"], lost_low)
        if failed:
            failed_reason = terminal_reason(sim) or "DROP_DURING_RECOVERY"
            break

    mech = analyze_mechanism(trace)
    m_hand = measure(sim, U_TGT, P_TGT)
    recovery_elapsed = float(sim.data.time) - t0

    # HANDOFF + nominal (unchanged path)
    samples = []
    reason = failed_reason
    success_12s = False
    success_new = False
    if reason is None:
        apply_handoff(sim, blob["ref"])
        # Detect handoff qpos edit: apply_handoff should not edit physics
        t_h = float(sim.data.time)
        lost = 0
        lift_done = False
        end_row = None
        next_sample = 0.0
        if latched(sim):
            reason = terminal_reason(sim, default="TABLE_DROP")
        from envs.ballistic_impact import z_tgt_of
        from envs.outcomes.horizon_outcome import LOST_STEPS, _bilateral, _dropped, _monitor

        while reason is None and float(sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
            nominal_step(sim)
            t_rel = float(sim.data.time) - t_h
            if t_rel + 1e-12 >= next_sample:
                m = measure(sim, U_TGT, P_TGT)
                cs = contact_summary(sim)
                samples.append({
                    "t_rel": t_rel,
                    "e_theta_deg": m["e_theta_deg"],
                    "e_pos_mm": m["e_pos_mm"],
                    "bilateral": cs["bilateral"],
                    "obj_z": m["obj_z"],
                })
                next_sample += 0.02
            row = _monitor(sim)
            end_row = row
            target = float(z_tgt_of(sim))
            bilateral = _bilateral(row)
            if bilateral and float(sim.fsm.p_des[2]) >= target - 1e-9:
                lift_done = True
            if latched(sim):
                reason = terminal_reason(sim, default="TABLE_DROP")
                break
            if _dropped(row, sim) or _escaped(sim):
                reason = "DROP_AFTER_HANDOFF"
                break
            lost = lost + 1 if not bilateral else 0
            if lost >= LOST_STEPS:
                reason = "LOST_GRASP"
                break
        if reason is None:
            if end_row is None or not _bilateral(end_row) or _dropped(end_row, sim) or _escaped(sim):
                reason = "LOST_GRASP"
            elif not lift_done:
                reason = "NOMINAL_LIFT_NOT_COMPLETED"
            else:
                reason = "SUCCESS_12S"
                success_12s = True
        win = window_stats(samples, float(NOMINAL_S) - FINAL_WINDOW_S)
        success_new = bool(success_12s and win.get("working_SUCCESS_NEW_hold"))
    else:
        win = window_stats([], float(NOMINAL_S) - FINAL_WINDOW_S)

    m_final = measure(sim, U_TGT, P_TGT)
    return {
        "legality": legality,
        "controller_target_mismatch_m": target_mismatch,
        "qpos_nonfinite_events": qpos_edits,
        "trace": trace,
        "contact_rows": contact_rows,
        "mechanism": mech,
        "reason": reason,
        "success_12s": success_12s,
        "SUCCESS_NEW": success_new,
        "window": win,
        "m_handoff": m_hand,
        "m_final": m_final,
        "recovery_elapsed": recovery_elapsed,
        "working_theta_deg": WORKING_THETA_DEG,
        "working_pos_mm": WORKING_POS_MM,
        "hold_s": HOLD_S,
    }


def transfer_state_to_ground(elev_env, ground_env) -> dict[str, Any]:
    """Copy live physical + controller state from elevated ballistic sim onto GROUND_SCENE."""
    from envs.simulation.cross_scene_restore import transfer_live_state

    diffs = transfer_live_state(elev_env.sim, ground_env.sim)
    cs_s = contact_summary(elev_env.sim)
    cs_d = contact_summary(ground_env.sim)
    diffs["bilateral_src"] = int(cs_s["bilateral"])
    diffs["bilateral_dst"] = int(cs_d["bilateral"])
    return diffs


def make_negative_actions(
    tokens: np.ndarray,
    table: np.ndarray,
    book48: np.ndarray,
    kind: str,
) -> list[np.ndarray]:
    """Legal negative-control action sequences from the same IC."""
    actions = [decode_token(table, book48, int(t)) for t in tokens]
    if kind == "zero":
        return [np.zeros(7, np.float32) for _ in actions]
    if kind == "no_translation":
        out = []
        for a in actions:
            b = a.copy()
            b[:3] = 0.0
            out.append(b)
        return out
    if kind == "no_wrist":
        out = []
        for a in actions:
            b = a.copy()
            b[3:6] = 0.0
            out.append(b)
        return out
    if kind == "hold_grip_only":
        out = []
        for a in actions:
            b = np.zeros(7, np.float32)
            b[6] = a[6]
            out.append(b)
        return out
    raise ValueError(kind)
