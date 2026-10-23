"""Error measurement, need assessment, and goal-region stop tests (v2.3)."""

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

from data_generation.oracles.precision.state import (
    PrecisionConfig,
    STOP_GOAL_REACHED,
    STOP_HANDOFF_READY,
    SlipGoal,
)

def measure_errors(sim) -> dict[str, Any]:
    s = sample_state(sim)
    m = measure(sim, U_TGT, P_TGT)
    e = np.asarray(s["e_signed_mm"], float)
    return {
        "e_signed_mm": e.tolist(),
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "dtheta_deg": float(s["e_theta_deg"]),
        "e_pos_mm": float(s["e_pos_mm"]),
        "bilateral": bool(s["bilateral"]),
        "v_rel_norm": float(s["v_rel_norm"]),
        "w_rel_norm": float(s.get("w_rel_norm") or 0.0),
        "fn_L": float(s["fn_L"]),
        "fn_R": float(s["fn_R"]),
        "fn_tot": float(s["fn_tot"]),
        "obj_z": float(m["obj_z"]),
        "alpha_up_deg": float(gates(sim)["alpha_up_deg"]),
        "fingertips_down": bool(gates(sim)["fingertips_down"]),
        "fingertips_up": bool(gates(sim)["fingertips_up"]),
    }


def handoff_geometry_ready(errors: dict, cfg: PrecisionConfig) -> bool:
    """Level-2 geometry: Euclidean + theta compatible with SUCCESS_NEW working region."""
    e = np.asarray(errors["e_signed_mm"], float)
    return bool(
        errors.get("bilateral")
        and float(errors["dtheta_deg"]) <= float(cfg.working_theta_deg)
        and float(errors["e_pos_mm"]) <= float(cfg.working_pos_mm)
        and abs(float(e[1])) <= float(cfg.ey_tol_mm)
        and float(errors.get("v_rel_norm") or 0.0) < 0.08
    )


def evaluate_slip_goal(errors: dict, goal: SlipGoal) -> tuple[bool, str | None]:
    """High-frequency local stop test (not a 12 s HANDOFF probe).

    SUCCESS_NEW uses Euclidean e_pos ≤ working_pos_mm. AXIS_V1 axis tolerances
    (especially ez≤20) must not stop sliding while e_pos remains excessive.
    When ``use_euclidean_stop`` is False (tipped-up inward), only the signed axis
    band is used — live e_pos is not predictive of the post-restore grasp.
    """
    e = np.asarray(errors["e_signed_mm"], float)
    theta = float(errors["dtheta_deg"])
    e_pos = float(errors["e_pos_mm"])
    wp = float(goal.working_pos_mm)
    idx = 0 if goal.axis == "dx" else 2
    other = 2 if idx == 0 else 0
    tgt = 0.0 if goal.target_signed_mm is None else float(goal.target_signed_mm)
    axis_err = abs(float(e[idx]) - tgt)
    margin = max(0.0, float(goal.brake_margin_mm))

    if not bool(goal.use_euclidean_stop):
        if axis_err <= float(goal.axis_tol_mm) + margin:
            if axis_err <= float(goal.axis_tol_mm):
                return True, STOP_GOAL_REACHED
        return False, None

    if (
        theta <= float(goal.working_theta_deg)
        and e_pos <= wp
        and abs(float(e[1])) <= float(goal.ey_tol_mm)
    ):
        return True, STOP_HANDOFF_READY

    # Euclidean budget: stop this axis when the controlled component alone cannot
    # be the reason e_pos exceeds the working threshold.
    budget_sq = wp * wp - float(e[0]) ** 2 - float(e[1]) ** 2 - float(e[2]) ** 2 + float(e[idx]) ** 2
    axis_budget = float(np.sqrt(budget_sq)) if budget_sq > 0.0 else 0.0
    if e_pos <= wp:
        return True, STOP_GOAL_REACHED
    if axis_err <= axis_budget + margin:
        if abs(float(e[other])) > abs(float(e[idx])) + 1.0 and abs(float(e[other])) > axis_budget:
            return True, STOP_GOAL_REACHED
        if axis_err <= axis_budget:
            return True, STOP_GOAL_REACHED
    return False, None

