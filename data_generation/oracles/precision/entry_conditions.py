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

from data_generation.oracles.precision.residuals import (
    evaluate_slip_goal,
    handoff_geometry_ready,
    measure_errors,
)

def entry_gate(env, *, thresholds: dict | None = None) -> dict[str, Any]:
    """Short bilateral contact safety check (~150–200 ms).

    This is **not** Stage-1 success. Stage-1 requires ``SUCCESS_12S`` via
    ``stage1_eligibility.check_stage1_eligible`` / ``stage1_12s_gate``.
    """
    stab = hold_and_confirm_stable(env, hold_s=0.20, thresholds=thresholds)
    err = measure_errors(env.sim)
    return {
        "ok": bool(stab["stable"]),
        "stability": {k: stab[k] for k in ("stable", "reasons", "consecutive_bil_s_at_end", "e_theta_deg", "e_pos_mm", "e_signed_mm")},
        "errors": err,
        "thresholds": {**(STABLE_THRESHOLDS), **(thresholds or {})},
        "is_stage1_success": False,
        "note": "Contact safety only; does not satisfy Stage-1 SUCCESS_12S.",
    }


def stage1_12s_gate(env, blob: dict, **kwargs) -> dict[str, Any]:
    """Authoritative Stage-1 eligibility wrapper (SUCCESS_12S on exact snapshot)."""
    from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible

    res = check_stage1_eligible(env, blob, **kwargs)
    return {
        "ok": bool(res.eligible),
        "status": res.status,
        "reason": res.reason,
        "eligibility": res.to_dict(),
        "is_stage1_success": bool(res.eligible),
    }


def assess_need(errors: dict, cfg: PrecisionConfig) -> dict[str, Any]:
    """Signed-error decision: dtheta / dx / dz needed?"""
    e = np.asarray(errors["e_signed_mm"], float)
    theta = float(errors["dtheta_deg"])
    axis = decide_precision(
        e,
        theta_deg=theta,
        bilateral=bool(errors["bilateral"]),
        ex_tol_mm=cfg.ex_tol_mm,
        ey_tol_mm=cfg.ey_tol_mm,
        ez_tol_mm=cfg.ez_tol_mm,
        theta_tol_deg=cfg.theta_tol_deg,
    )
    dtheta_needed = theta > cfg.theta_tol_deg
    # Prefer SUCCESS_NEW Euclidean for "already ready" when configured
    pose_ready = True
    if cfg.require_success_new_pose:
        pose_ready = (
            float(errors["e_pos_mm"]) <= cfg.working_pos_mm
            and theta <= cfg.working_theta_deg
            and abs(float(e[1])) <= cfg.ey_tol_mm
        )
    axial = select_axial_mode(float(e[2]), axial_tol_mm=cfg.ez_tol_mm)
    dx_needed = abs(float(e[0])) > cfg.ex_tol_mm
    dz_needed = abs(float(e[2])) > cfg.ez_tol_mm
    # P15 forced-Precision generation: tighter experimental need bands (off by default).
    force_prec = bool(getattr(cfg, "force_precision", False))
    force_skill = str(getattr(cfg, "force_skill", None) or "").strip().lower() or None
    if force_prec:
        fex = getattr(cfg, "force_ex_tol_mm", None)
        fez = getattr(cfg, "force_ez_tol_mm", None)
        ex_f = float(fex) if fex is not None else float(cfg.ex_tol_mm)
        ez_f = float(fez) if fez is not None else float(cfg.ez_tol_mm)
        if force_skill in (None, "dx", "coupled"):
            if abs(float(e[0])) > ex_f:
                dx_needed = True
        if force_skill in (None, "dz", "coupled"):
            if abs(float(e[2])) > ez_f:
                dz_needed = True
        if force_skill == "dx":
            # DX-only demos: do not schedule DZ from e_pos residual heuristics below.
            dz_needed = False
            if abs(float(e[0])) > ex_f:
                dx_needed = True
        if force_skill == "dz":
            # DZ-only demos: do not schedule DX from e_pos residual heuristics below.
            dx_needed = False
            if abs(float(e[2])) > ez_f:
                dz_needed = True
                axial = select_axial_mode(
                    float(e[2]), axial_tol_mm=min(ez_f, max(abs(float(e[2])) - 0.1, 0.5))
                )
    # SUCCESS_NEW uses Euclidean e_pos≤12 mm. AXIS_V1 may leave |ez|≤20 with e_pos>12;
    # request correction along the dominant residual so Direct/HANDOFF readiness matches SUCCESS_NEW.
    e_pos = float(errors.get("e_pos_mm") or 0.0)
    residual_note = None
    if force_prec and force_skill == "dx":
        residual_note = "force_precision_dx"
    elif force_prec and force_skill == "dz":
        residual_note = "force_precision_dz"
    elif force_prec and force_skill == "coupled":
        residual_note = "force_precision_coupled"
    if (
        (not force_prec or force_skill not in ("dx", "dz", "coupled"))
        and cfg.require_success_new_pose
        and e_pos > cfg.working_pos_mm
        and not dtheta_needed
    ):
        if not dx_needed and not dz_needed:
            if abs(float(e[2])) >= abs(float(e[0])):
                dz_needed = True
                residual_note = "e_pos_over_working_dominant_ez"
            else:
                dx_needed = True
                residual_note = "e_pos_over_working_dominant_ex"
        elif not dz_needed and abs(float(e[2])) >= abs(float(e[0])) and abs(float(e[2])) > 0.5 * cfg.ez_tol_mm:
            dz_needed = True
            residual_note = "e_pos_over_working_add_ez"
    # Inward dz entry requires |ex|≤dz_safe_ex_mm (tighter than AXIS_V1 ex_tol).
    # If Euclidean residual asks for dz while lateral is still outside the flip-safe
    # band, schedule dx first — otherwise the scheduler aborts with DZ_UNSAFE_ENTRY
    # and zero actions (dominant PRECISION_PRIMITIVE_FAILED mode on hard B exits).
    dz_safe_ex = float(getattr(cfg, "dz_safe_ex_mm", 6.0) or 6.0)
    if (
        (not force_prec or force_skill not in ("dx", "dz"))
        and dz_needed
        and abs(float(e[0])) > dz_safe_ex
        and not dx_needed
    ):
        dx_needed = True
        residual_note = residual_note or "dz_requires_prior_dx_safe_band"
    # Recompute axial mode if we newly requested dz
    if dz_needed and axial.get("mode") == "none":
        axial = select_axial_mode(float(e[2]), axial_tol_mm=min(cfg.ez_tol_mm, max(abs(float(e[2])) - 0.1, 1.0)))
    ready = bool(pose_ready and not dtheta_needed and not dx_needed and not dz_needed and axis["mode"] != "abnormal")
    return {
        "ready_for_direct_handoff_assessment": ready,
        "dtheta_needed": dtheta_needed,
        "dx_needed": dx_needed,
        "dz_needed": dz_needed,
        "dz_mode": axial.get("mode"),
        "axis_decision": axis,
        "pose_ready_success_new": pose_ready,
        "e_signed_mm": e.tolist(),
        "dtheta_deg": theta,
        "e_pos_mm": e_pos,
        "residual_note": residual_note,
    }

