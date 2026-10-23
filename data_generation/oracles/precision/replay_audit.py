"""Action/capture replay audits and failure snapshots (v2.3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.oracles.precision.corrections.residual_directions import (
    decide_precision,
    lateral_desire_direction,
)
from data_generation.oracles.precision.mechanics.open_axis import (
    DOWN,
    UP,
    choose_roll_sign_for_support,
    gates,
    reference_pad_vector,
    run_roll_phase,
    run_tilt_phase,
)
from data_generation.oracles.precision.mechanics.finger_surface import support_force_decomposition
from data_generation.oracles.precision.mechanics.finger_vertical import U_OPEN_HAND
from data_generation.oracles.precision.mechanics.dual_loop import (
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    contact_forces,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.corrections.orientation import (
    _height_hold_v_hand,
    execute_orientation_plan,
    plan_orientation,
    score_gravity_alignment,
)
from data_generation.oracles.precision.corrections.contact_primitives import (
    _action_vw,
    _wrist,
    audit_grasp,
    pivot_segment_metrics,
    primitive_contact_pivot,
)
from data_generation.oracles.precision.mechanics.reverse_rotation import HoldModeGrip
from data_generation.oracles.precision.corrections.axial_sign import (
    fingertip_overlap_margin_mm,
    select_axial_mode,
)
from data_generation.oracles.precision.mechanics.grasp_stabilization import (
    STABLE_THRESHOLDS,
    hold_and_confirm_stable,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import (
    AXIS_V1_EX_MM,
    AXIS_V1_EY_MM,
    AXIS_V1_EZ_MM,
    AXIS_V1_THETA_DEG,
    evaluate_handoff_with_axis,
)
from evaluation.grasp_geometry import hand_R, measure
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG

from data_generation.oracles.precision.residuals import measure_errors

def _failure_snapshot(sim, *, phase: str, code: str, last_action=None) -> dict[str, Any]:
    s = sample_state(sim)
    err = measure_errors(sim)
    return {
        "failure_code": code,
        "phase": phase,
        "sim_time": float(sim.data.time),
        "last_action": None if last_action is None else np.asarray(last_action, float).tolist(),
        "fn_L": s["fn_L"],
        "fn_R": s["fn_R"],
        "fn_tot": s["fn_tot"],
        "e_signed_mm": err["e_signed_mm"],
        "v_rel_norm": s["v_rel_norm"],
        "alpha_up_deg": err["alpha_up_deg"],
        "obj_z": err["obj_z"],
        "bilateral": s["bilateral"],
    }


def audit_action_capture_replay(
    env,
    start_capture,
    actions: list | None,
    terminal_capture,
    *,
    atol_qpos: float = 1e-3,
    atol_qvel: float = 2e-2,
    atol_pose: float = 1e-3,
    atol_time: float = 1e-6,
    atol_err_mm: float = 0.25,
    atol_theta_deg: float = 0.5,
) -> dict[str, Any]:
    """Replay accepted action prefix from start_capture; compare to terminal_capture.

    Tolerances match deterministic MuJoCo replay with minor contact drift — not
    SUCCESS_* label equality.
    """
    if terminal_capture is None:
        return {"ok": False, "reason": "no_terminal_capture", "n_actions": len(actions or [])}
    acts = list(actions or [])
    restore_capture(env, start_capture)
    t0 = float(env.sim.data.time)
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
    ph_exp = terminal_capture["physics"]
    qpos_err = float(np.max(np.abs(np.asarray(env.sim.data.qpos, float) - np.asarray(ph_exp["qpos"], float))))
    qvel_err = float(np.max(np.abs(np.asarray(env.sim.data.qvel, float) - np.asarray(ph_exp["qvel"], float))))
    # Object free-joint pose: last 7 qpos of object body — use xpos/xquat for robustness
    oid = env.sim.ids.object_body
    pos_got = np.asarray(env.sim.data.xpos[oid], float)
    # Terminal pose from restored capture comparison
    restore_capture(env, terminal_capture)
    pos_exp = np.asarray(env.sim.data.xpos[oid], float)
    err_exp = measure_errors(env.sim)
    # Re-replay to get terminal errors without clobbering comparison state
    restore_capture(env, start_capture)
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
    err_got = measure_errors(env.sim)
    pos_err = float(np.linalg.norm(pos_got - pos_exp))
    time_err = abs(float(env.sim.data.time) - float(ph_exp["time"]))
    e_got = np.asarray(err_got["e_signed_mm"], float)
    e_exp = np.asarray(err_exp["e_signed_mm"], float)
    e_err = float(np.max(np.abs(e_got - e_exp)))
    th_err = abs(float(err_got["dtheta_deg"]) - float(err_exp["dtheta_deg"]))
    mismatches = []
    if qpos_err > atol_qpos:
        mismatches.append({"field": "qpos", "max_abs": qpos_err, "atol": atol_qpos})
    if qvel_err > atol_qvel:
        mismatches.append({"field": "qvel", "max_abs": qvel_err, "atol": atol_qvel})
    if pos_err > atol_pose:
        mismatches.append({"field": "object_xpos", "err": pos_err, "atol": atol_pose})
    if time_err > atol_time + DT * 0.5:
        # Allow half-step slack only if action count implies time; else flag
        expected_dt = len(acts) * DT
        if abs((float(env.sim.data.time) - t0) - expected_dt) > atol_time + 1e-9:
            mismatches.append({"field": "sim_time", "err": time_err, "atol": atol_time})
    if e_err > atol_err_mm:
        mismatches.append({"field": "e_signed_mm", "max_abs": e_err, "atol": atol_err_mm})
    if th_err > atol_theta_deg:
        mismatches.append({"field": "dtheta_deg", "err": th_err, "atol": atol_theta_deg})
    if bool(err_got["bilateral"]) != bool(err_exp["bilateral"]):
        mismatches.append({"field": "bilateral", "got": err_got["bilateral"], "exp": err_exp["bilateral"]})
    return {
        "ok": len(mismatches) == 0,
        "n_actions": len(acts),
        "qpos_max_abs": qpos_err,
        "qvel_max_abs": qvel_err,
        "object_xpos_err": pos_err,
        "sim_time_err": time_err,
        "e_signed_replay": e_got.tolist(),
        "e_signed_terminal": e_exp.tolist(),
        "dtheta_replay": float(err_got["dtheta_deg"]),
        "dtheta_terminal": float(err_exp["dtheta_deg"]),
        "bilateral_replay": bool(err_got["bilateral"]),
        "mismatches": mismatches,
    }

