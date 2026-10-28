"""Experimental Coupled DX/DZ sliding (not production).

Under verified support, orient for gravity-driven slip along a 2D desire in the
hand xz plane, then DualLoop along that direction. No silent fallback to
single-axis DX/DZ inside this primitive.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.corrections.orientation import (
    execute_orientation_plan,
    plan_orientation,
    score_gravity_alignment,
)
from data_generation.oracles.precision.mechanics.axial_sliding import run_goal_aware_axial_slip
from data_generation.oracles.precision.mechanics.dual_loop import sample_state
from data_generation.oracles.precision.mechanics.open_axis import choose_roll_sign_for_support, run_roll_phase
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import FN_HOLD, PrecisionConfig, SlipGoal
from envs.simulation.capture import restore_capture, take_capture
from evaluation.grasp_geometry import hand_R

G_WORLD = np.array([0.0, 0.0, -9.81], float)


def coupled_desire_hand(e_signed_mm: np.ndarray) -> dict[str, Any]:
    """2D desire in hand xz from signed residuals. atan2 only as initial estimate."""
    e = np.asarray(e_signed_mm, float).reshape(3)
    dx, dz = float(e[0]), float(e[2])
    # Desired COM motion reduces residuals: d ∝ (-dx, 0, -dz)
    d = np.array([-dx, 0.0, -dz], float)
    n = float(np.linalg.norm(d))
    geom_angle_deg = float(np.degrees(np.arctan2(-dx, -dz))) if (abs(dx) + abs(dz)) > 1e-9 else 0.0
    if n < 1e-6:
        return {"ok": False, "reason": "residuals_near_zero", "d": [0, 0, 0], "geom_angle_deg": geom_angle_deg}
    d = d / n
    # Feasibility: both components must be meaningfully represented in d
    w_dx = abs(float(d[0]))
    w_dz = abs(float(d[2]))
    if w_dx < 0.15 and abs(dx) > 3.0:
        return {
            "ok": False,
            "reason": "dx_not_representable_in_single_slip_direction",
            "d": d.tolist(),
            "dx_mm": dx,
            "dz_mm": dz,
            "geom_angle_deg": geom_angle_deg,
        }
    if w_dz < 0.15 and abs(dz) > 3.0:
        return {
            "ok": False,
            "reason": "dz_not_representable_in_single_slip_direction",
            "d": d.tolist(),
            "dx_mm": dx,
            "dz_mm": dz,
            "geom_angle_deg": geom_angle_deg,
        }
    return {
        "ok": True,
        "d": d.tolist(),
        "dx_mm": dx,
        "dz_mm": dz,
        "geom_angle_deg": geom_angle_deg,
        "weight_dx": w_dx,
        "weight_dz": w_dz,
        "norm_mm": n,
    }


def coupled_dx_dz(env, blob: dict, *, cfg: PrecisionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PrecisionConfig(use_verified_lateral_fallback=False)
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    if not err0.get("bilateral"):
        return {"ok": False, "reason": "no_bilateral_contact", "actions": [], "capture": blob["capture"]}

    desire = coupled_desire_hand(err0["e_signed_mm"])
    if not desire.get("ok"):
        return {
            "ok": False,
            "reason": desire.get("reason") or "desire_infeasible",
            "desire": desire,
            "errors0": err0,
            "actions": [],
            "capture": blob["capture"],
            "infeasible": True,
        }

    d = np.asarray(desire["d"], float)
    actions: list = []
    logs: dict[str, Any] = {"desire": desire}

    # --- Orientation for gravity alignment ---
    plan = plan_orientation(env, blob, d=d, g_useful_min=6.0, max_candidates=36)
    logs["orientation_plan"] = {k: v for k, v in plan.items() if k not in ("actions",)}
    if not plan.get("ok") and not plan.get("already_sufficient"):
        return {
            "ok": False,
            "reason": "no_feasible_gravity_orientation",
            "desire": desire,
            "orientation_plan": logs["orientation_plan"],
            "errors0": err0,
            "actions": [],
            "capture": blob["capture"],
            "infeasible": True,
        }

    cur = {"capture": blob["capture"], "ref": blob.get("ref")}
    if not plan.get("already_sufficient"):
        eo = execute_orientation_plan(env, cur, plan)
        logs["orientation_exec"] = {k: v for k, v in eo.items() if k not in ("actions", "capture")}
        actions += eo.get("actions") or []
        if not eo.get("ok"):
            return {
                "ok": False,
                "reason": "orientation_exec_failed",
                "desire": desire,
                "orientation_exec": logs["orientation_exec"],
                "actions": actions,
                "capture": take_capture(env),
                "errors0": err0,
                "errors1": measure_errors(env.sim),
            }
        cur = {"capture": eo.get("capture") or take_capture(env), "ref": blob.get("ref")}
    else:
        restore_capture(env, blob["capture"])

    # --- Support bias (modest roll) + force check ---
    restore_capture(env, cur["capture"])
    Rh = hand_R(env.sim)
    g_h = Rh.T @ G_WORLD
    sc = score_gravity_alignment(g_h, d)
    logs["gravity_score_pre_slip"] = sc
    if float(sc["useful"]) < 4.0:
        return {
            "ok": False,
            "reason": "gravity_alignment_insufficient_after_orient",
            "desire": desire,
            "gravity_score": sc,
            "actions": actions,
            "capture": take_capture(env),
            "infeasible": True,
        }

    roll_sign = float(choose_roll_sign_for_support(env.sim))
    y0 = None
    try:
        from data_generation.oracles.precision.mechanics.open_axis import reference_pad_vector

        y0 = reference_pad_vector(env.sim)
    except Exception:
        y0 = np.array([0.0, 1.0, 0.0])
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    pr = run_roll_phase(
        env,
        cur,
        sign=roll_sign,
        angle_deg=10.0,
        phase="coupled_S1_bias",
        rate_scale=0.12,
        fn_hold=FN_HOLD,
        require_down=False,
        y0_w=y0,
        e_ref=e_ref,
    )
    actions += pr.get("actions") or []
    logs["s1_roll"] = {k: v for k, v in pr.items() if k not in ("actions", "capture")}
    mode = classify_contact_mode(env.sim)
    logs["contact_mode_pre_slip"] = mode
    if not sample_state(env.sim).get("bilateral"):
        return {
            "ok": False,
            "reason": "contact_loss_during_support_bias",
            "actions": actions,
            "capture": take_capture(env),
            "desire": desire,
        }
    # Require designated support or one-sided mechanical under gravity when tilted
    if not (mode.get("support_side") or mode.get("one_sided_mechanical") or mode.get("bilateral")):
        return {
            "ok": False,
            "reason": "mechanical_support_not_verified",
            "contact_mode": mode,
            "actions": actions,
            "capture": take_capture(env),
            "infeasible": True,
        }

    cur = {"capture": take_capture(env), "ref": blob.get("ref")}
    theta0 = float(measure_errors(env.sim)["dtheta_deg"])
    e0 = np.asarray(measure_errors(env.sim)["e_signed_mm"], float)

    # Coupled objective: stop when Euclidean e_pos is in working band OR both |dx|,|dz| small
    wp = float(cfg.working_pos_mm)
    axis_tol = float(np.clip(np.sqrt(max(wp * wp - float(e0[1]) ** 2, 0.0)) * 0.5, 2.0, 6.0))
    # Use dx axis label but desire is 2D; DualLoop projects along d
    goal = SlipGoal(
        axis="dx",
        desired_direction=d,
        max_safe_travel_mm=18.0,
        max_duration_s=10.0,
        axis_tol_mm=axis_tol,
        working_pos_mm=wp,
        working_theta_deg=float(cfg.working_theta_deg),
        ey_tol_mm=float(cfg.ey_tol_mm),
        use_euclidean_stop=True,
        brake_margin_mm=1.5,
    )
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    slip = run_goal_aware_axial_slip(
        env,
        cur,
        goal,
        z_ref=z_ref,
        guard_lateral=False,
        guard_fingertip_margin=False,
    )
    actions += slip.get("actions") or []
    logs["slip"] = {k: v for k, v in slip.items() if k not in ("actions", "capture")}

    err1 = measure_errors(env.sim)
    e1 = np.asarray(err1["e_signed_mm"], float)
    theta1 = float(err1["dtheta_deg"])
    # Reject large induced rotation
    if theta1 > theta0 + 5.0:
        return {
            "ok": False,
            "reason": "induced_dtheta_increase",
            "desire": desire,
            "theta0": theta0,
            "theta1": theta1,
            "errors0": err0,
            "errors1": err1,
            "actions": actions,
            "capture": take_capture(env),
            "logs": logs,
        }

    # Success of primitive: reduced coupled residual and grasp retained
    r0 = float(np.linalg.norm([e0[0], e0[2]]))
    r1 = float(np.linalg.norm([e1[0], e1[2]]))
    improved = r1 < r0 - 1.0 or float(err1["e_pos_mm"]) <= wp
    ok = bool(slip.get("ok") and err1.get("bilateral") and improved and not slip.get("infeasible"))
    return {
        "ok": bool(ok),
        "mechanism": "coupled_dx_dz_experimental",
        "reason": None if ok else (slip.get("stop_reason") or slip.get("reason") or "coupled_slip_failed"),
        "desire": desire,
        "errors0": err0,
        "errors1": err1,
        "residual_xz0_mm": r0,
        "residual_xz1_mm": r1,
        "d_theta_deg": theta1 - theta0,
        "stop_reason": slip.get("stop_reason"),
        "goal_reached": slip.get("goal_reached"),
        "disp_along_mm": slip.get("disp_along_mm"),
        "actions": actions,
        "capture": take_capture(env),
        "logs": logs,
        "infeasible": bool(logs.get("desire", {}).get("ok") is False),
    }
