"""Goal-aware DualLoop sliding, staged slip, brake/regrip helpers (v2.3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.oracles.precision.axis_aware import (
    decide_precision,
    lateral_desire_direction,
)
from data_generation.oracles.precision.endpoint_only_axial_roll import (
    DOWN,
    UP,
    choose_roll_sign_for_support,
    gates,
    reference_pad_vector,
    run_roll_phase,
    run_tilt_phase,
)
from data_generation.oracles.precision.finger_surface_support import support_force_decomposition
from data_generation.oracles.precision.finger_supported_vertical import U_OPEN_HAND
from data_generation.oracles.precision.force_slip_control import (
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    contact_forces,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import (
    _height_hold_v_hand,
    execute_orientation_plan,
    plan_orientation,
    score_gravity_alignment,
)
from data_generation.oracles.precision.primitives import (
    _action_vw,
    _wrist,
    audit_grasp,
    pivot_segment_metrics,
    primitive_contact_pivot,
)
from data_generation.oracles.precision.reverse_supported_rotation import HoldModeGrip
from data_generation.oracles.precision.sign_aware_axial import (
    fingertip_overlap_margin_mm,
    select_axial_mode,
)
from data_generation.oracles.precision.stable_grasp import (
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

from data_generation.oracles.precision.precision_types import (
    A0_SLIP,
    FAIL_BRAKE,
    FAIL_REGRIP,
    FAIL_SLIP_CONTACT,
    FAIL_SLIP_TARGET,
    FINGERTIP_MARGIN_MIN_MM,
    FN_BRAKE,
    FN_HOLD,
    FN_INIT,
    FN_MAX,
    FN_MIN,
    FORCE_DA,
    FORCE_KP,
    HOLD_STABLE_S,
    LATERAL_ABS_ABORT_MM,
    LATERAL_DRIFT_ABORT_MM,
    PrecisionConfig,
    SLIP_KN,
    STOP_BRAKE_AND_REASSESS,
    STOP_CONTACT_SAFETY,
    STOP_GOAL_REACHED,
    STOP_HANDOFF_READY,
    STOP_MAX_SAFE_TRAVEL,
    STOP_NO_PROGRESS,
    STOP_TIME_BUDGET,
    STOP_WRONG_DIRECTION,
    SlipGoal,
    V_SLIP,
    Z_ABORT,
)
from data_generation.oracles.precision.precision_goals import evaluate_slip_goal, measure_errors
from data_generation.oracles.precision.precision_audit import _failure_snapshot

def run_slip_dual_loop(
    env,
    blob,
    d_axis,
    *,
    milestone_mm: float,
    max_s: float = 12.0,
    z_ref: float | None = None,
    guard_lateral: bool = True,
    e_x0: float | None = None,
    vx_target: float = V_SLIP,
) -> dict[str, Any]:
    """Verified DualLoop slip along desire direction d (cylinder COM relative)."""
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    e0 = np.asarray(s0["e_signed_mm"], float)
    if e_x0 is None:
        e_x0 = float(e0[0])
    d = np.asarray(d_axis, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return {
            "ok": False,
            "failure_code": FAIL_SLIP_TARGET,
            "reason": "zero_desire",
            "actions": [],
            "series": [],
            "capture": take_capture(env),
            "reached": False,
            "brake_ok": False,
            "regrip_ok": False,
        }
    d = d / n
    # SlipOuterLoop: axis = −d with vx_target < 0 ⇒ unload when not sliding along d
    axis = -d
    if z_ref is None:
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=vx_target, fn_min=FN_MIN, fn_max=FN_MAX, kn=SLIP_KN),
        use_slip=True,
    )
    ctl.reset(a0=A0_SLIP, fn0=s0["fn_tot"], fn_target=FN_INIT, slip_axis=tuple(axis.tolist()))
    actions, series = [], []
    lost = 0
    reached = False
    abort_reason = None
    failure_code = None
    for i in range(max(1, int(round(max_s / DT)))):
        s = sample_state(env.sim)
        _, meta = ctl.step(s)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        e2 = np.asarray(s2["e_signed_mm"], float)
        disp = float(np.dot(dp, d)) * 1e3
        series.append(
            {
                "t": (i + 1) * DT,
                "fn": s2["fn_tot"],
                "disp_along_mm": disp,
                "e_signed": s2["e_signed_mm"],
                "theta": s2["e_theta_deg"],
                "bilateral": s2["bilateral"],
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "phase": "slip",
            }
        )
        if failed or latched(env.sim) or not s2["bilateral"] or series[-1]["obj_z"] < Z_ABORT:
            code = FAIL_SLIP_CONTACT if not s2["bilateral"] else "WORLD_HEIGHT"
            return {
                "ok": False,
                "failure_code": code,
                "reason": "contact_loss" if not s2["bilateral"] else "world_height",
                "actions": actions,
                "series": series,
                "reached": False,
                "brake_ok": False,
                "regrip_ok": False,
                "capture": take_capture(env),
                "failure": _failure_snapshot(env.sim, phase="slip", code=code, last_action=a),
            }
        if guard_lateral:
            if abs(float(e2[0]) - e_x0) > LATERAL_DRIFT_ABORT_MM or abs(float(e2[0])) > LATERAL_ABS_ABORT_MM:
                abort_reason = "lateral_drift"
                failure_code = FAIL_SLIP_WRONG_DIR
                break
        if disp < -0.5 and i > 10:
            abort_reason = "wrong_direction"
            failure_code = FAIL_SLIP_WRONG_DIR
            break
        if disp >= milestone_mm:
            reached = True
            break

    ctl_b = ForcePController(kp=0.15, da_max=0.2)
    ctl_b.reset(0.5, series[-1]["fn"] if series else 10.0)
    n_brake = max(1, int(round(0.40 / DT)))
    n_hold = max(1, int(round(HOLD_STABLE_S / DT)))
    for j in range(n_brake + n_hold):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], FN_BRAKE)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        series.append(
            {
                "t": (series[-1]["t"] if series else 0.0) + DT,
                "fn": s2["fn_tot"],
                "disp_along_mm": float(np.dot(dp, d) * 1e3),
                "e_signed": s2["e_signed_mm"],
                "theta": s2["e_theta_deg"],
                "bilateral": s2["bilateral"],
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "phase": "brake" if j < n_brake else "hold",
            }
        )
        if not s2["bilateral"]:
            code = FAIL_BRAKE if j < n_brake else FAIL_REGRIP
            return {
                "ok": False,
                "failure_code": code,
                "reason": "loss_on_brake" if j < n_brake else "loss_on_regrip",
                "actions": actions,
                "series": series,
                "reached": reached,
                "brake_ok": False,
                "regrip_ok": False,
                "capture": take_capture(env),
                "failure": _failure_snapshot(env.sim, phase="brake" if j < n_brake else "hold", code=code, last_action=a),
            }

    disp = float(series[-1]["disp_along_mm"])
    hold_ok = all(r["bilateral"] for r in series if r["phase"] == "hold")
    ok = bool(hold_ok and (reached or disp >= 0.5 * milestone_mm) and abort_reason is None)
    if not ok and failure_code is None:
        failure_code = FAIL_SLIP_TARGET if not reached else FAIL_REGRIP
    return {
        "ok": ok,
        "reached": reached,
        "abort_reason": abort_reason,
        "failure_code": failure_code,
        "actions": actions,
        "series": series,
        "disp_along_mm": disp,
        "e_signed": series[-1]["e_signed"],
        "theta": series[-1]["theta"],
        "brake_ok": True,
        "regrip_ok": hold_ok,
        "capture": take_capture(env),
    }


def settle_hold(env, actions: list, *, fn_hold: float = FN_HOLD, hold_s: float = 0.4) -> None:
    hold = HoldModeGrip(fn_hold=fn_hold)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for _ in range(max(1, int(round(hold_s / DT)))):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())


def run_goal_aware_axial_slip(
    env,
    blob,
    goal: SlipGoal,
    *,
    z_ref: float | None = None,
    guard_lateral: bool = False,
    guard_fingertip_margin: bool = False,
    min_margin_mm: float = FINGERTIP_MARGIN_MIN_MM,
    fn_min: float = FN_MIN,
    fn_max: float = FN_MAX,
    kn: float = SLIP_KN,
    vx_target: float = V_SLIP,
    fn_init: float = FN_INIT,
    a0_slip: float = A0_SLIP,
    allow_post_brake_nudge: bool = True,
) -> dict[str, Any]:
    """DualLoop slip that stops on geometric goal / safety — not a fixed displacement.

    Displacement is logged diagnostically. ``max_safe_travel_mm`` is a hard safety cap.
    """
    d = np.asarray(goal.desired_direction, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return {
            "ok": False,
            "failure_code": FAIL_SLIP_TARGET,
            "stop_reason": STOP_WRONG_DIRECTION,
            "reason": "zero_direction",
            "actions": [],
            "capture": take_capture(env),
        }
    d = d / n
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    if z_ref is None:
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    e_x0 = float(err0["e_signed_mm"][0])
    idx = 0 if goal.axis == "dx" else 2
    e_axis0 = float(err0["e_signed_mm"][idx])

    already, why = evaluate_slip_goal(err0, goal)
    if already:
        return {
            "ok": True,
            "skipped": True,
            "stop_reason": why or STOP_GOAL_REACHED,
            "disp_along_mm": 0.0,
            "disp_total_mm": 0.0,
            "e_signed": err0["e_signed_mm"],
            "errors_entry": err0,
            "errors": err0,
            "actions": [],
            "capture": take_capture(env),
            "reached": True,
            "brake_ok": True,
            "regrip_ok": True,
            "contract": "goal_driven",
            "goal": {"axis": goal.axis, "max_safe_travel_mm": goal.max_safe_travel_mm},
        }

    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=vx_target, fn_min=fn_min, fn_max=fn_max, kn=kn),
        use_slip=True,
    )
    ctl.reset(a0=a0_slip, fn0=s0["fn_tot"], fn_target=fn_init, slip_axis=tuple((-d).tolist()))
    actions: list = []
    series: list[dict] = []
    lost = 0
    last_disp = 0.0
    stop_reason = STOP_TIME_BUDGET
    best_axis_err = abs(e_axis0)
    stall_steps = 0
    goal_hit_pre_brake = False

    def _disp() -> float:
        return float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p0, d)) * 1e3

    max_steps = int(float(goal.max_duration_s) / DT)
    for i in range(max_steps):
        err = measure_errors(env.sim)
        hit, why = evaluate_slip_goal(err, goal)
        if hit:
            stop_reason = why or STOP_GOAL_REACHED
            goal_hit_pre_brake = True
            break
        last_disp = _disp()
        if last_disp >= float(goal.max_safe_travel_mm):
            stop_reason = STOP_MAX_SAFE_TRAVEL
            break
        if last_disp < -0.5:
            stop_reason = STOP_WRONG_DIRECTION
            break
        axis_err = abs(float(err["e_signed_mm"][idx]) - (0.0 if goal.target_signed_mm is None else float(goal.target_signed_mm)))
        if axis_err + 1e-6 < best_axis_err - float(goal.min_progress_mm) * 0.25:
            best_axis_err = axis_err
            stall_steps = 0
        else:
            stall_steps += 1
        if stall_steps > int(2.5 / DT) and last_disp > 0.8:
            stop_reason = STOP_NO_PROGRESS
            break

        s = sample_state(env.sim)
        _, meta = ctl.step(s)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        last_disp = _disp()
        if failed or latched(env.sim) or not s2["bilateral"]:
            return {
                "ok": False,
                "failure_code": FAIL_SLIP_CONTACT,
                "stop_reason": STOP_CONTACT_SAFETY,
                "reason": "contact_loss",
                "actions": actions,
                "capture": take_capture(env),
                "failure": _failure_snapshot(env.sim, phase="slip", code=FAIL_SLIP_CONTACT, last_action=a),
                "disp_along_mm": last_disp,
                "disp_total_mm": last_disp,
                "errors_entry": err0,
                "contract": "goal_driven",
            }
        if guard_fingertip_margin:
            marg = fingertip_overlap_margin_mm(env.sim, U_OPEN_HAND)
            if marg < min_margin_mm:
                stop_reason = STOP_CONTACT_SAFETY
                break
        if guard_lateral and abs(float(s2["e_signed_mm"][0]) - e_x0) > LATERAL_DRIFT_ABORT_MM:
            stop_reason = STOP_WRONG_DIRECTION
            break
        if (i + 1) % 5 == 0:
            series.append(
                {
                    "t": (i + 1) * DT,
                    "disp_mm": last_disp,
                    "e_signed_mm": s2["e_signed_mm"],
                    "e_pos_mm": float(measure_errors(env.sim)["e_pos_mm"]),
                    "fn_tot": s2["fn_tot"],
                    "v_rel_norm": s2["v_rel_norm"],
                }
            )
    else:
        stop_reason = STOP_TIME_BUDGET

    disp_at_stop = _disp()
    # Brake + regrip
    ctl_b = ForcePController(kp=0.22, da_max=0.25, a_min=-0.5, a_max=1.0)
    ctl_b.reset(0.95, sample_state(env.sim)["fn_tot"])
    for j in range(int(0.45 / DT) + int(0.25 / DT)):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], max(FN_BRAKE, 14.0))
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        if not sample_state(env.sim)["bilateral"]:
            for _ in range(8):
                a_em = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], 1.0)
                step_legal(env.sim, a_em)
                actions.append(a_em.copy())
                if sample_state(env.sim)["bilateral"]:
                    break
            if not sample_state(env.sim)["bilateral"]:
                code = FAIL_BRAKE if j < int(0.45 / DT) else FAIL_REGRIP
                return {
                    "ok": False,
                    "failure_code": code,
                    "stop_reason": STOP_CONTACT_SAFETY,
                    "reason": "loss_on_brake",
                    "actions": actions,
                    "capture": take_capture(env),
                    "disp_along_mm": disp_at_stop,
                    "disp_total_mm": _disp(),
                    "disp_before_brake_mm": disp_at_stop,
                    "errors_entry": err0,
                    "contract": "goal_driven",
                }

    disp_after = _disp()
    err_after = measure_errors(env.sim)
    hit_after, why_after = evaluate_slip_goal(err_after, goal)

    # Bounded post-brake nudge if braking pushed us out of the target region
    nudged = False
    if (
        allow_post_brake_nudge
        and goal_hit_pre_brake
        and not hit_after
        and err_after["bilateral"]
        and stop_reason in (STOP_GOAL_REACHED, STOP_HANDOFF_READY)
        and disp_after < float(goal.max_safe_travel_mm) - 0.5
    ):
        nudged = True
        ctl2 = DualLoopController(
            force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
            slip=SlipOuterLoop(vx_target=vx_target * 0.7, fn_min=fn_min, fn_max=fn_max, kn=kn),
            use_slip=True,
        )
        ctl2.reset(a0=0.65, fn0=sample_state(env.sim)["fn_tot"], fn_target=fn_init, slip_axis=tuple((-d).tolist()))
        for _ in range(int(1.2 / DT)):
            err = measure_errors(env.sim)
            hit, why = evaluate_slip_goal(err, goal)
            if hit:
                stop_reason = why or STOP_BRAKE_AND_REASSESS
                break
            if _disp() >= float(goal.max_safe_travel_mm):
                break
            s = sample_state(env.sim)
            _, meta = ctl2.step(s)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
            step_legal(env.sim, a)
            actions.append(a.copy())
            if not sample_state(env.sim)["bilateral"]:
                break
        # Short re-brake
        for _ in range(int(0.25 / DT)):
            st = sample_state(env.sim)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], 0.95)
            step_legal(env.sim, a)
            actions.append(a.copy())
        disp_after = _disp()
        err_after = measure_errors(env.sim)
        hit_after, why_after = evaluate_slip_goal(err_after, goal)
        if hit_after:
            stop_reason = why_after or STOP_BRAKE_AND_REASSESS

    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    for _ in range(int(0.25 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
    err_final = measure_errors(env.sim)
    hit_final, why_final = evaluate_slip_goal(err_final, goal)
    disp_final = _disp()

    # Success = contact retained and (goal after brake OR meaningful axis improvement without safety abort)
    axis_final = float(err_final["e_signed_mm"][idx])
    improved = abs(axis_final) < abs(e_axis0) - 0.3
    safe_stop = stop_reason not in (STOP_CONTACT_SAFETY, STOP_WRONG_DIRECTION)
    ok = bool(err_final["bilateral"] and safe_stop and (hit_final or improved or stop_reason == STOP_MAX_SAFE_TRAVEL and improved))
    if hit_final:
        stop_reason = why_final or stop_reason

    return {
        "ok": ok,
        "stop_reason": stop_reason,
        "goal_reached": bool(hit_final),
        "goal_reached_pre_brake": bool(goal_hit_pre_brake),
        "post_brake_nudge": nudged,
        "failure_code": None if ok else (FAIL_SLIP_TARGET if safe_stop else FAIL_SLIP_CONTACT),
        "disp_along_mm": disp_final,
        "disp_total_mm": disp_final,
        "disp_before_brake_mm": disp_at_stop,
        "disp_net_after_brake_mm": disp_after,
        "e_signed": err_final["e_signed_mm"],
        "errors_entry": err0,
        "errors": err_final,
        "actions": actions,
        "capture": take_capture(env),
        "reached": bool(hit_final or goal_hit_pre_brake),
        "brake_ok": True,
        "regrip_ok": True,
        "direction": d.tolist(),
        "series": series,
        "contract": "goal_driven",
        "goal": {
            "axis": goal.axis,
            "axis_tol_mm": goal.axis_tol_mm,
            "max_safe_travel_mm": goal.max_safe_travel_mm,
            "working_pos_mm": goal.working_pos_mm,
        },
    }


def run_staged_axial_slip(
    env,
    blob,
    direction: np.ndarray | list[float],
    *,
    milestone_mm: float = 5.0,
    z_ref: float | None = None,
    e_x0: float | None = None,
    guard_lateral: bool = True,
    guard_fingertip_margin: bool = False,
    min_margin_mm: float = FINGERTIP_MARGIN_MIN_MM,
    fn_min: float = FN_MIN,
    fn_max: float = FN_MAX,
    kn: float = SLIP_KN,
    vx_target: float = V_SLIP,
    fn_init: float = FN_INIT,
    a0_slip: float = A0_SLIP,
) -> dict[str, Any]:
    """Shared staged DualLoop axial slip with explicit total-displacement contract.

    `milestone_mm` is interpreted as ``target_total_mm``: net slip along +d from the
    original stage-entry relative pose (not an additional stage-local request).
    Stage 1 calibrates at ``min(2, target)``; stage 2 commands only the remaining
    distance after measured net displacement (including brake drift).

    `direction` is desired cylinder COM motion in the hand frame (unitized).
    Inward: −u_open; outward: +u_open. Same controller architecture; optional
    firmer bands for outward contact safety.
    """
    d = np.asarray(direction, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return {
            "ok": False,
            "failure_code": FAIL_SLIP_TARGET,
            "reason": "zero_direction",
            "actions": [],
            "capture": take_capture(env),
        }
    d = d / n
    restore_capture(env, blob["capture"])
    if z_ref is None:
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    if e_x0 is None:
        e_x0 = float(sample_state(env.sim)["e_signed_mm"][0])

    def _one(blob_i, milestone, e_x):
        restore_capture(env, blob_i["capture"])
        s0 = sample_state(env.sim)
        p0 = np.asarray(s0["p_c_h"], float)
        ctl = DualLoopController(
            force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
            slip=SlipOuterLoop(vx_target=vx_target, fn_min=fn_min, fn_max=fn_max, kn=kn),
            use_slip=True,
        )
        ctl.reset(a0=a0_slip, fn0=s0["fn_tot"], fn_target=fn_init, slip_axis=tuple((-d).tolist()))
        actions = []
        lost = 0
        reached = False
        abort = None
        fail_code = None
        last_disp = 0.0
        for _ in range(int(12.0 / DT)):
            s = sample_state(env.sim)
            _, meta = ctl.step(s)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
            step_legal(env.sim, a)
            actions.append(a.copy())
            failed, lost = recovery_failed(env.sim, blob_i["ref"], lost)
            s2 = sample_state(env.sim)
            disp = float(np.dot(np.asarray(s2["p_c_h"], float) - p0, d)) * 1e3
            last_disp = disp
            if failed or latched(env.sim) or not s2["bilateral"]:
                return {
                    "ok": False,
                    "failure_code": FAIL_SLIP_CONTACT,
                    "reason": "contact_loss",
                    "actions": actions,
                    "capture": take_capture(env),
                    "failure": _failure_snapshot(env.sim, phase="slip", code=FAIL_SLIP_CONTACT, last_action=a),
                    "reached": False,
                    "brake_ok": False,
                    "regrip_ok": False,
                }
            if guard_fingertip_margin:
                marg = fingertip_overlap_margin_mm(env.sim, U_OPEN_HAND)
                if marg < min_margin_mm:
                    abort = "fingertip_margin"
                    fail_code = FAIL_SLIP_TARGET
                    break
            if guard_lateral and abs(float(s2["e_signed_mm"][0]) - e_x) > LATERAL_DRIFT_ABORT_MM:
                abort = "lateral_drift"
                fail_code = FAIL_SLIP_WRONG_DIR
                break
            if disp < -0.5:
                abort = "wrong_direction"
                fail_code = FAIL_SLIP_WRONG_DIR
                break
            if disp >= milestone:
                reached = True
                break
        # Firm brake: raise grip quickly (da_max higher) to avoid unloading into loss
        ctl_b = ForcePController(kp=0.22, da_max=0.25, a_min=-0.5, a_max=1.0)
        ctl_b.reset(0.95, sample_state(env.sim)["fn_tot"])
        for j in range(int(0.45 / DT) + int(0.25 / DT)):
            s = sample_state(env.sim)
            a_g, _ = ctl_b.step(s["fn_tot"], max(FN_BRAKE, 14.0))
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
            step_legal(env.sim, a)
            actions.append(a.copy())
            if not sample_state(env.sim)["bilateral"]:
                # One emergency firm-up burst
                for _ in range(8):
                    a_em = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], 1.0)
                    step_legal(env.sim, a_em)
                    actions.append(a_em.copy())
                    if sample_state(env.sim)["bilateral"]:
                        break
                if not sample_state(env.sim)["bilateral"]:
                    code = FAIL_BRAKE if j < int(0.45 / DT) else FAIL_REGRIP
                    return {
                        "ok": False,
                        "failure_code": code,
                        "reason": "loss_on_brake",
                        "actions": actions,
                        "capture": take_capture(env),
                        "reached": reached,
                        "brake_ok": False,
                        "regrip_ok": False,
                        "disp": last_disp,
                        "failure": _failure_snapshot(env.sim, phase="brake", code=code, last_action=a),
                    }
        # Stage-local displacement from this stage's p0.
        # Caller interprets cumulative totals via target_total_mm contract.
        ok = bool(abort is None and (reached or last_disp >= 0.5 * milestone))
        net_after = float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p0, d)) * 1e3
        return {
            "ok": ok,
            "failure_code": None if ok else (fail_code or FAIL_SLIP_TARGET),
            "disp": last_disp,
            "disp_before_brake_mm": last_disp,
            "disp_net_after_brake_mm": net_after,
            "e_signed": sample_state(env.sim)["e_signed_mm"],
            "actions": actions,
            "capture": take_capture(env),
            "abort_reason": abort,
            "reached": reached,
            "brake_ok": True,
            "regrip_ok": True,
        }

    # Explicit total-displacement contract:
    #   target_total_mm = intended net slip along +d from the original stage-entry pose.
    # Stage 1 calibrates at min(2 mm, target). Stage 2 requests only the remaining
    # distance after accounting for net displacement achieved (incl. brake drift).
    target_total = float(milestone_mm)
    p_entry = np.asarray(sample_state(env.sim)["p_c_h"], float).copy()
    stage_log: list[dict] = []

    def _cum_from_entry() -> float:
        return float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p_entry, d)) * 1e3

    cal_mm = min(2.0, target_total)
    s_cal = _one(blob, cal_mm, e_x0)
    stage_log.append({
        "stage": 1,
        "commanded_mm": cal_mm,
        "disp_before_brake_mm": s_cal.get("disp_before_brake_mm", s_cal.get("disp")),
        "disp_net_after_brake_mm": s_cal.get("disp_net_after_brake_mm", s_cal.get("disp")),
        "ok": bool(s_cal.get("ok")),
        "abort_reason": s_cal.get("abort_reason"),
    })
    if not s_cal.get("ok"):
        return {
            **s_cal,
            "target_total_mm": target_total,
            "disp_total_mm": s_cal.get("disp"),
            "remaining_mm": target_total - float(s_cal.get("disp") or 0.0),
            "staged_log": stage_log,
            "contract": "target_total_mm",
        }

    restore_capture(env, s_cal["capture"])
    cum_after_cal = _cum_from_entry()
    remaining = max(0.0, target_total - cum_after_cal)
    all_actions = list(s_cal["actions"])
    last = s_cal

    if remaining > 0.15:
        s_rem = _one(
            {"capture": s_cal["capture"], "ref": blob["ref"]},
            remaining,
            float(s_cal["e_signed"][0]),
        )
        stage_log.append({
            "stage": 2,
            "commanded_mm": remaining,
            "disp_before_brake_mm": s_rem.get("disp_before_brake_mm", s_rem.get("disp")),
            "disp_net_after_brake_mm": s_rem.get("disp_net_after_brake_mm", s_rem.get("disp")),
            "ok": bool(s_rem.get("ok")),
            "abort_reason": s_rem.get("abort_reason"),
            "cum_before_stage_mm": cum_after_cal,
        })
        all_actions = all_actions + list(s_rem.get("actions") or [])
        if not s_rem.get("ok"):
            restore_capture(env, s_rem.get("capture") or s_cal["capture"])
            cum_fail = _cum_from_entry()
            return {
                **s_rem,
                "actions": all_actions,
                "disp_2": s_cal.get("disp"),
                "disp_total_mm": cum_fail,
                "target_total_mm": target_total,
                "remaining_mm": target_total - cum_fail,
                "staged_log": stage_log,
                "contract": "target_total_mm",
                "overshoot_mm": max(0.0, cum_fail - target_total),
            }
        last = s_rem

    restore_capture(env, last["capture"])
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    extra = []
    for _ in range(int(0.25 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        extra.append(a.copy())
    cum_final = _cum_from_entry()
    return {
        "ok": True,
        "disp_along_mm": cum_final,
        "disp_total_mm": cum_final,
        "disp_2": s_cal.get("disp"),
        "target_total_mm": target_total,
        "remaining_mm": target_total - cum_final,
        "overshoot_mm": max(0.0, cum_final - target_total),
        "backward_drift_mm": max(0.0, -cum_final) if cum_final < 0 else 0.0,
        "e_signed": sample_state(env.sim)["e_signed_mm"],
        "actions": all_actions + extra,
        "capture": take_capture(env),
        "reached": True,
        "brake_ok": True,
        "regrip_ok": True,
        "direction": d.tolist(),
        "staged_log": stage_log,
        "contract": "target_total_mm",
    }


def _inward_slip_staged(
    env, blob, z_ref: float, e_x0: float, *, milestone_mm: float = 12.0, cfg: PrecisionConfig | None = None
) -> dict[str, Any]:
    """Inward (−u_open) DualLoop — goal-driven when cfg.use_goal_driven_slip (default)."""
    cfg = cfg or PrecisionConfig()
    if bool(getattr(cfg, "use_goal_driven_slip", True)):
        # Tipped-up e_pos is not predictive of post-restore SUCCESS_NEW; drive |dz|
        # into a Euclidean-compatible band estimated from the pre-slip lateral errors.
        restore_capture(env, blob["capture"])
        e0 = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
        wp = float(cfg.working_pos_mm)
        budget_sq = wp * wp - float(e0[0]) ** 2 - float(e0[1]) ** 2
        axis_tol = float(np.sqrt(max(budget_sq, 0.0))) if budget_sq > 0 else 3.0
        axis_tol = float(np.clip(axis_tol, 3.0, 8.0))
        goal = SlipGoal(
            axis="dz",
            desired_direction=-np.asarray(U_OPEN_HAND, float),
            max_safe_travel_mm=float(milestone_mm),
            max_duration_s=12.0,
            axis_tol_mm=axis_tol,
            working_pos_mm=wp,
            working_theta_deg=float(cfg.working_theta_deg),
            ey_tol_mm=float(cfg.ey_tol_mm),
            use_euclidean_stop=False,
            brake_margin_mm=1.0,
        )
        return run_goal_aware_axial_slip(
            env, blob, goal, z_ref=z_ref, guard_lateral=True, guard_fingertip_margin=False,
        )
    return run_staged_axial_slip(
        env, blob, -np.asarray(U_OPEN_HAND, float),
        milestone_mm=milestone_mm, z_ref=z_ref, e_x0=e_x0, guard_lateral=True, guard_fingertip_margin=False,
    )


def _outward_slip_staged(
    env, blob, *, z_ref: float, e_x0: float, milestone_mm: float = 12.0, cfg: PrecisionConfig | None = None
) -> dict[str, Any]:
    """Outward (+u_open) DualLoop — goal-driven when cfg.use_goal_driven_slip (default)."""
    cfg = cfg or PrecisionConfig()
    if bool(getattr(cfg, "use_goal_driven_slip", True)):
        goal = SlipGoal(
            axis="dz",
            desired_direction=np.asarray(U_OPEN_HAND, float),
            max_safe_travel_mm=float(milestone_mm),
            max_duration_s=12.0,
            axis_tol_mm=float(cfg.ez_tol_mm),
            working_pos_mm=float(cfg.working_pos_mm),
            working_theta_deg=float(cfg.working_theta_deg),
            ey_tol_mm=float(cfg.ey_tol_mm),
        )
        return run_goal_aware_axial_slip(
            env,
            blob,
            goal,
            z_ref=z_ref,
            guard_lateral=True,
            guard_fingertip_margin=True,
            min_margin_mm=FINGERTIP_MARGIN_MIN_MM,
            fn_min=5.0,
            fn_max=12.0,
            kn=500.0,
            vx_target=-0.0007,
            fn_init=7.0,
            a0_slip=0.70,
        )
    return run_staged_axial_slip(
        env,
        blob,
        np.asarray(U_OPEN_HAND, float),
        milestone_mm=milestone_mm,
        z_ref=z_ref,
        e_x0=e_x0,
        guard_lateral=True,
        guard_fingertip_margin=True,
        min_margin_mm=FINGERTIP_MARGIN_MIN_MM,
        fn_min=5.0,
        fn_max=12.0,
        kn=500.0,
        vx_target=-0.0007,
        fn_init=7.0,
        a0_slip=0.70,
    )

