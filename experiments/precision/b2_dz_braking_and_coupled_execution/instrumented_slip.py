"""Instrumented goal-aware inward slip + continuation variants (experimental).

Does not modify production ``axial_sliding`` / ``correct_dz_inward``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.mechanics.dual_loop import (  # noqa: E402
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    DT,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.mechanics.finger_vertical import U_OPEN_HAND  # noqa: E402
from data_generation.oracles.precision.corrections.orientation import _height_hold_v_hand  # noqa: E402
from data_generation.oracles.precision.mechanics.reverse_rotation import HoldModeGrip  # noqa: E402
from data_generation.oracles.precision.residuals import evaluate_slip_goal, measure_errors  # noqa: E402
from data_generation.oracles.precision.state import (  # noqa: E402
    A0_SLIP,
    FAIL_BRAKE,
    FAIL_REGRIP,
    FAIL_SLIP_CONTACT,
    FAIL_SLIP_TARGET,
    FN_BRAKE,
    FN_HOLD,
    FN_INIT,
    FN_MAX,
    FN_MIN,
    FORCE_DA,
    FORCE_KP,
    LATERAL_DRIFT_ABORT_MM,
    PrecisionConfig,
    SLIP_KN,
    SlipGoal,
    STOP_CONTACT_SAFETY,
    STOP_GOAL_REACHED,
    STOP_MAX_SAFE_TRAVEL,
    STOP_NO_PROGRESS,
    STOP_TIME_BUDGET,
    STOP_WRONG_DIRECTION,
    V_SLIP,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal


def compute_inward_axis_tol(e_signed_mm: np.ndarray, working_pos_mm: float) -> dict[str, float]:
    e = np.asarray(e_signed_mm, float).reshape(3)
    wp = float(working_pos_mm)
    budget_sq = wp * wp - float(e[0]) ** 2 - float(e[1]) ** 2
    raw = float(np.sqrt(max(budget_sq, 0.0))) if budget_sq > 0 else 3.0
    clipped = float(np.clip(raw, 3.0, 8.0))
    return {
        "ex_mm": float(e[0]),
        "ey_mm": float(e[1]),
        "ez_mm": float(e[2]),
        "working_pos_mm": wp,
        "budget_sq": float(budget_sq),
        "axis_tol_raw_mm": raw,
        "axis_tol_mm": clipped,
        "clip_lo": 3.0,
        "clip_hi": 8.0,
        "brake_margin_mm": 1.0,
        "expression": "clip(sqrt(wp^2 - ex^2 - ey^2), 3, 8)",
    }


def make_inward_goal(
    e0: np.ndarray,
    cfg: PrecisionConfig,
    *,
    axis_tol_override: float | None = None,
    max_safe_travel_mm: float | None = None,
    use_euclidean_stop: bool = False,
) -> tuple[SlipGoal, dict]:
    info = compute_inward_axis_tol(e0, float(cfg.working_pos_mm))
    tol = float(axis_tol_override) if axis_tol_override is not None else float(info["axis_tol_mm"])
    info["axis_tol_effective_mm"] = tol
    goal = SlipGoal(
        axis="dz",
        desired_direction=-np.asarray(U_OPEN_HAND, float),
        max_safe_travel_mm=float(max_safe_travel_mm if max_safe_travel_mm is not None else cfg.inward_slip_mm),
        max_duration_s=12.0,
        axis_tol_mm=tol,
        working_pos_mm=float(cfg.working_pos_mm),
        working_theta_deg=float(cfg.working_theta_deg),
        ey_tol_mm=float(cfg.ey_tol_mm),
        use_euclidean_stop=bool(use_euclidean_stop),
        brake_margin_mm=1.0,
    )
    return goal, info


def _row(env, *, t, phase, goal, stop_reason=None, brake_trigger=False, budget_left_s=None) -> dict[str, Any]:
    err = measure_errors(env.sim)
    e = np.asarray(err["e_signed_mm"], float)
    st = sample_state(env.sim)
    axis_err = abs(float(e[2]))
    hit, why = evaluate_slip_goal(err, goal) if goal is not None else (False, None)
    return {
        "t": float(t),
        "phase": phase,
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "e_pos_mm": float(err["e_pos_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "v_rel_norm": float(err.get("v_rel_norm") or 0.0),
        "v_axial_approx": float(st.get("v_rel_norm") or 0.0),  # magnitude proxy; signed axial in DualLoop meta
        "bilateral": bool(err.get("bilateral")),
        "fn_L": float(err.get("fn_L") or 0.0),
        "fn_R": float(err.get("fn_R") or 0.0),
        "fn_tot": float(err.get("fn_tot") or 0.0),
        "grip_cmd": None,
        "axis_tol_mm": float(goal.axis_tol_mm) if goal else None,
        "brake_margin_mm": float(goal.brake_margin_mm) if goal else None,
        "dz_target_signed_mm": 0.0,
        "goal_hit": bool(hit),
        "goal_why": why,
        "brake_trigger": bool(brake_trigger),
        "stop_reason": stop_reason,
        "budget_left_s": budget_left_s,
        "obj_z": float(err.get("obj_z") or 0.0),
    }


def run_instrumented_inward_slip(
    env,
    blob: dict,
    cfg: PrecisionConfig,
    *,
    axis_tol_override: float | None = None,
    max_safe_travel_mm: float | None = None,
    continue_past_production_tol: bool = False,
    production_tol_for_mark: float | None = None,
) -> dict[str, Any]:
    """Goal-aware inward slip with per-step CSV timeline (mirrors production DualLoop)."""
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    e0 = np.asarray(err0["e_signed_mm"], float)
    goal, tol_info = make_inward_goal(
        e0,
        cfg,
        axis_tol_override=axis_tol_override,
        max_safe_travel_mm=max_safe_travel_mm,
    )
    # D1: continue past production tol — use tighter override already set; mark when crossing prod tol
    prod_tol = float(production_tol_for_mark) if production_tol_for_mark is not None else float(tol_info["axis_tol_mm"])
    if continue_past_production_tol and axis_tol_override is None:
        # Explicit D1: keep slipping until |dz|<=2mm or safety, while production would stop at clipped tol
        goal, tol_info = make_inward_goal(e0, cfg, axis_tol_override=2.0, max_safe_travel_mm=max_safe_travel_mm or 18.0)
        prod_tol = float(compute_inward_axis_tol(e0, float(cfg.working_pos_mm))["axis_tol_mm"])

    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    e_x0 = float(e0[0])
    d = -np.asarray(U_OPEN_HAND, float)
    d = d / float(np.linalg.norm(d))

    timeline: list[dict] = []
    markers: dict[str, Any] = {
        "dz_at_slip_start": float(e0[2]),
        "dz_at_brake_start": None,
        "dz_at_slip_stop": None,
        "production_tol_crossed": False,
        "dz_when_production_tol_hit": None,
    }

    already, why = evaluate_slip_goal(err0, goal)
    timeline.append(_row(env, t=0.0, phase="slip_entry", goal=goal, stop_reason=why if already else None, budget_left_s=goal.max_duration_s))
    if already:
        return {
            "ok": True,
            "skipped": True,
            "stop_reason": why or STOP_GOAL_REACHED,
            "timeline": timeline,
            "markers": markers,
            "tol_info": tol_info,
            "actions": [],
            "capture": take_capture(env),
            "errors0": err0,
            "errors1": err0,
            "goal": goal,
        }

    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=V_SLIP, fn_min=FN_MIN, fn_max=FN_MAX, kn=SLIP_KN),
        use_slip=True,
    )
    ctl.reset(a0=A0_SLIP, fn0=s0["fn_tot"], fn_target=FN_INIT, slip_axis=tuple((-d).tolist()))
    actions: list = []
    lost = 0
    stop_reason = STOP_TIME_BUDGET
    best_axis_err = abs(float(e0[2]))
    stall_steps = 0
    goal_hit_pre_brake = False
    t = 0.0
    max_steps = int(float(goal.max_duration_s) / DT)

    def _disp() -> float:
        return float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p0, d)) * 1e3

    for i in range(max_steps):
        err = measure_errors(env.sim)
        hit, why = evaluate_slip_goal(err, goal)
        # Mark when production tol would have stopped (D1 comparison)
        if not markers["production_tol_crossed"] and abs(float(err["dz_mm"])) <= prod_tol:
            markers["production_tol_crossed"] = True
            markers["dz_when_production_tol_hit"] = float(err["dz_mm"])
            markers["t_production_tol_hit"] = t
            timeline.append(
                _row(env, t=t, phase="MARK_production_tol_would_stop", goal=goal, stop_reason="PRODUCTION_TOL", budget_left_s=goal.max_duration_s - t)
            )
            if not continue_past_production_tol and axis_tol_override is None:
                # Standard production path uses goal.axis_tol == prod_tol; hit handles it
                pass

        if hit:
            stop_reason = why or STOP_GOAL_REACHED
            goal_hit_pre_brake = True
            markers["dz_at_brake_start"] = float(err["dz_mm"])
            markers["t_brake_start"] = t
            timeline.append(_row(env, t=t, phase="brake_trigger", goal=goal, stop_reason=stop_reason, brake_trigger=True, budget_left_s=goal.max_duration_s - t))
            break
        last_disp = _disp()
        if last_disp >= float(goal.max_safe_travel_mm):
            stop_reason = STOP_MAX_SAFE_TRAVEL
            markers["dz_at_brake_start"] = float(err["dz_mm"])
            break
        if last_disp < -0.5:
            stop_reason = STOP_WRONG_DIRECTION
            break
        axis_err = abs(float(err["dz_mm"]))
        if axis_err + 1e-6 < best_axis_err - float(goal.min_progress_mm) * 0.25:
            best_axis_err = axis_err
            stall_steps = 0
        else:
            stall_steps += 1
        if stall_steps > int(2.5 / DT) and last_disp > 0.8:
            stop_reason = STOP_NO_PROGRESS
            markers["dz_at_brake_start"] = float(err["dz_mm"])
            break

        s = sample_state(env.sim)
        _, meta = ctl.step(s)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        t += DT
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        if (i + 1) % 2 == 0 or i < 3:
            r = _row(env, t=t, phase="slip", goal=goal, budget_left_s=goal.max_duration_s - t)
            r["grip_cmd"] = float(a[6])
            r["disp_along_mm"] = _disp()
            timeline.append(r)
        if failed or latched(env.sim) or not s2["bilateral"]:
            return {
                "ok": False,
                "stop_reason": STOP_CONTACT_SAFETY,
                "reason": "contact_loss",
                "timeline": timeline,
                "markers": markers,
                "tol_info": tol_info,
                "actions": actions,
                "capture": take_capture(env),
                "errors0": err0,
                "errors1": measure_errors(env.sim),
                "goal": goal,
            }
        if abs(float(s2["e_signed_mm"][0]) - e_x0) > LATERAL_DRIFT_ABORT_MM:
            stop_reason = STOP_WRONG_DIRECTION
            break
    else:
        stop_reason = STOP_TIME_BUDGET
        markers["dz_at_brake_start"] = float(measure_errors(env.sim)["dz_mm"])

    markers["dz_at_slip_stop"] = float(measure_errors(env.sim)["dz_mm"])
    # Brake + regrip (same as production)
    ctl_b = ForcePController(kp=0.22, da_max=0.25, a_min=-0.5, a_max=1.0)
    ctl_b.reset(0.95, sample_state(env.sim)["fn_tot"])
    for j in range(int(0.45 / DT) + int(0.25 / DT)):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], max(FN_BRAKE, 14.0))
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        t += DT
        if j % 3 == 0:
            timeline.append(_row(env, t=t, phase="brake" if j < int(0.45 / DT) else "regrip", goal=goal, stop_reason=stop_reason, brake_trigger=True))
        if not sample_state(env.sim)["bilateral"]:
            for _ in range(8):
                a_em = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], 1.0)
                step_legal(env.sim, a_em)
                actions.append(a_em.copy())
                t += DT
                if sample_state(env.sim)["bilateral"]:
                    break
            if not sample_state(env.sim)["bilateral"]:
                return {
                    "ok": False,
                    "stop_reason": STOP_CONTACT_SAFETY,
                    "reason": "loss_on_brake",
                    "failure_code": FAIL_BRAKE if j < int(0.45 / DT) else FAIL_REGRIP,
                    "timeline": timeline,
                    "markers": markers,
                    "tol_info": tol_info,
                    "actions": actions,
                    "capture": take_capture(env),
                    "errors0": err0,
                    "errors1": measure_errors(env.sim),
                    "goal": goal,
                }

    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    for _ in range(int(0.25 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        t += DT
    err_final = measure_errors(env.sim)
    timeline.append(_row(env, t=t, phase="slip_done", goal=goal, stop_reason=stop_reason))
    markers["dz_after_brake_regrip"] = float(err_final["dz_mm"])
    hit_final, why_final = evaluate_slip_goal(err_final, goal)
    improved = abs(float(err_final["dz_mm"])) < abs(float(e0[2])) - 0.3
    safe_stop = stop_reason not in (STOP_CONTACT_SAFETY, STOP_WRONG_DIRECTION)
    ok = bool(err_final["bilateral"] and safe_stop and (hit_final or improved))
    return {
        "ok": ok,
        "stop_reason": why_final if hit_final else stop_reason,
        "goal_reached": bool(hit_final),
        "goal_reached_pre_brake": bool(goal_hit_pre_brake),
        "timeline": timeline,
        "markers": markers,
        "tol_info": tol_info,
        "actions": actions,
        "capture": take_capture(env),
        "errors0": err0,
        "errors1": err_final,
        "goal": goal,
        "prod_tol_mm": prod_tol,
        "failure_code": None if ok else FAIL_SLIP_TARGET,
    }
