"""Lateral (verified) and horizontal P0–P9 dx correction (v2.3)."""

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
    FAIL_FINAL_UNROLL,
    FAIL_HORIZONTAL_TILT,
    FAIL_PREPARATION,
    FAIL_REVERSE_ROLL,
    FAIL_SLIDING_ROLL,
    FAIL_SLIP_CONTACT,
    FAIL_SLIP_TARGET,
    FAIL_SLIP_WRONG_DIR,
    FAIL_SUPPORT_ROLL,
    FN_HOLD,
    G_LATERAL_MIN,
    HORIZONTAL_SLIDE_ROLL_CANDIDATES,
    PrecisionConfig,
    STOP_GOAL_REACHED,
    STOP_HANDOFF_READY,
    SlipGoal,
    Z_ABORT,
)
from data_generation.oracles.precision.precision_goals import (
    handoff_geometry_ready,
    measure_errors,
)
from data_generation.oracles.precision.precision_audit import _failure_snapshot
from data_generation.oracles.precision.precision_slip import (
    run_goal_aware_axial_slip,
    run_slip_dual_loop,
    run_staged_axial_slip,
    settle_hold,
)

def correct_dx_verified(env, blob, cfg: PrecisionConfig) -> dict[str, Any]:
    """Verified gravity-assisted lateral slip (orientation plan → DualLoop).

    Goal-driven by default: stop when |dx| enters ``lateral_axis_tol_mm`` (safety
    travel cap only). Fixed-displacement DualLoop is retained when
    ``use_goal_driven_slip`` is False.
    """
    actions: list = []
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    d_lat = lateral_desire_direction(s0["e_signed_mm"])
    if float(np.linalg.norm(d_lat)) < 0.5:
        return {"ok": True, "skipped": True, "reason": "dx_near_zero", "actions": [], "capture": blob["capture"]}

    plan = plan_orientation(env, blob, d=d_lat, simulate=True, g_useful_min=8.5)
    if not (plan.get("ok") or plan.get("already_sufficient")):
        return {
            "ok": False,
            "mechanism": "verified_lateral",
            "phase": "plan",
            "reason": "lateral_orientation_unreachable",
            "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
            "actions": [],
            "capture": blob["capture"],
        }
    exe = execute_orientation_plan(env, blob, plan, adaptive=True)
    actions.extend(exe.get("actions") or [])
    if not exe.get("ok") and not plan.get("already_sufficient"):
        return {
            "ok": False,
            "mechanism": "verified_lateral",
            "phase": "reori",
            "exe": {k: v for k, v in exe.items() if k not in ("actions", "series", "capture")},
            "actions": actions,
            "capture": exe.get("capture") or blob["capture"],
        }
    tilt = {"capture": exe.get("capture") or take_capture(env), "ref": blob["ref"]}
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    max_safe = float(cfg.lateral_slip_mm)

    if bool(getattr(cfg, "use_goal_driven_slip", True)):
        # Axis-only stop: tipped gravity-assist e_pos is not a reliable HANDOFF proxy,
        # and overshooting past ~3 mm often breaks subsequent inward axial contact.
        axis_tol = float(getattr(cfg, "lateral_axis_tol_mm", 3.0) or 3.0)
        goal = SlipGoal(
            axis="dx",
            desired_direction=d_lat,
            max_safe_travel_mm=max_safe,
            max_duration_s=20.0,
            axis_tol_mm=axis_tol,
            working_pos_mm=float(cfg.working_pos_mm),
            working_theta_deg=float(cfg.working_theta_deg),
            ey_tol_mm=float(cfg.ey_tol_mm),
            use_euclidean_stop=False,
            brake_margin_mm=0.8,
        )
        slip = run_goal_aware_axial_slip(
            env, tilt, goal, z_ref=z_ref, guard_lateral=False,
        )
        actions.extend(slip.get("actions") or [])
        reached = bool(slip.get("ok") and (
            slip.get("goal_reached")
            or slip.get("stop_reason") in (STOP_GOAL_REACHED, STOP_HANDOFF_READY, STOP_MAX_SAFE_TRAVEL)
            or float(slip.get("disp_total_mm") or 0.0) >= 0.5 * min(max_safe, abs(float(s0["e_signed_mm"][0])))
        ))
        if not reached:
            return {
                "ok": False,
                "mechanism": "verified_lateral",
                "phase": "slip",
                "stop_reason": slip.get("stop_reason"),
                "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
                "actions": actions,
                "capture": slip.get("capture") or tilt["capture"],
            }
        restore_capture(env, slip["capture"])
        settle_hold(env, actions, hold_s=0.35)
        return {
            "ok": True,
            "mechanism": "verified_lateral",
            "contract": "goal_driven",
            "stop_reason": slip.get("stop_reason"),
            "goal_reached": slip.get("goal_reached"),
            "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
            "disp_along_mm": slip.get("disp_along_mm"),
            "disp_total_mm": slip.get("disp_total_mm"),
            "actions": actions,
            "capture": take_capture(env),
            "errors": measure_errors(env.sim),
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        }

    milestone = min(max_safe, abs(float(s0["e_signed_mm"][0])) + 1.0)
    slip = run_slip_dual_loop(
        env, tilt, d_lat, milestone_mm=milestone, max_s=20.0, z_ref=z_ref, guard_lateral=False
    )
    actions.extend(slip.get("actions") or [])
    if not slip.get("ok") and not (slip.get("disp_along_mm") or 0) >= 0.5 * milestone:
        return {
            "ok": False,
            "mechanism": "verified_lateral",
            "phase": "slip",
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
            "actions": actions,
            "capture": slip.get("capture") or tilt["capture"],
        }
    restore_capture(env, slip["capture"])
    # Restore toward fingertips-down if needed (plan_restore optional — settle is enough for many states)
    settle_hold(env, actions, hold_s=0.35)
    return {
        "ok": True,
        "mechanism": "verified_lateral",
        "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
        "disp_along_mm": slip.get("disp_along_mm"),
        "actions": actions,
        "capture": take_capture(env),
        "errors": measure_errors(env.sim),
    }


def _horizontal_open_target(sim) -> np.ndarray:
    """World direction for u_open at α≈90° (fingertips horizontal)."""
    u = hand_R(sim) @ np.asarray(U_OPEN_HAND, float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    axis = np.cross(u, UP)
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        n = float(np.linalg.norm(axis))
    axis = axis / max(n, 1e-12)
    a0 = float(gates(sim)["alpha_up_deg"])
    delta = np.radians(a0 - 90.0)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], float)
    R = np.eye(3) + np.sin(delta) * K + (1 - np.cos(delta)) * (K @ K)
    u_tgt = R @ u
    return u_tgt / max(float(np.linalg.norm(u_tgt)), 1e-12)


def _score_horizontal_slide_config(sim, d_lat: np.ndarray) -> dict[str, Any]:
    """Score contact/gravity readiness for lateral DualLoop at current pose."""
    s = sample_state(sim)
    g_sc = score_gravity_alignment(np.asarray(s["g_h"], float), d_lat)
    cf = contact_forces(sim)
    sup = support_force_decomposition(sim)
    bil = bool(s["bilateral"] and cf["fn_L"] > 1.0 and cf["fn_R"] > 1.0)
    useful = float(g_sc["useful"])
    score = useful + (5.0 if bil else -20.0) + 0.01 * float(cf["fn_tot"])
    return {
        "ok": bool(bil and useful >= G_LATERAL_MIN),
        "score": score,
        "g_score": g_sc,
        "bilateral": bil,
        "fn_L": cf["fn_L"],
        "fn_R": cf["fn_R"],
        "support": {k: sup.get(k) for k in ("proximal_fraction", "palm_or_base_support_detected") if k in sup},
        "alpha_up_deg": gates(sim)["alpha_up_deg"],
        "e_signed_mm": s["e_signed_mm"],
    }


def select_horizontal_sliding_roll(
    env, blob, d_lat: np.ndarray, y0, e_ref
) -> dict[str, Any]:
    """Bounded counterfactual search for P3 axial-roll angle/sign at horizontal."""
    trials = []
    best = None
    for sign, ang in HORIZONTAL_SLIDE_ROLL_CANDIDATES:
        restore_capture(env, blob["capture"])
        r = run_roll_phase(
            env, blob, sign=sign, angle_deg=ang, phase=f"dxH_P3_probe_s{int(sign)}_{int(ang)}",
            rate_scale=0.12, fn_hold=FN_HOLD, require_down=False, y0_w=y0, e_ref=e_ref,
        )
        # Rejected branch — do not keep actions; restore probe state only for scoring
        if not r.get("ok"):
            trials.append({"sign": sign, "angle_deg": ang, "ok": False, "reason": r.get("reason")})
            continue
        sc = _score_horizontal_slide_config(env.sim, d_lat)
        row = {"sign": sign, "angle_deg": ang, "ok": sc["ok"], "score": sc["score"], "diag": sc}
        trials.append(row)
        if sc["ok"] and (best is None or sc["score"] > best["score"]):
            best = {**row, "capture": r["capture"], "actions": r["actions"]}
    restore_capture(env, blob["capture"])
    if best is None:
        return {"ok": False, "failure_code": FAIL_SLIDING_ROLL, "trials": trials, "reason": "no_viable_P3_roll"}
    return {
        "ok": True,
        "sign": best["sign"],
        "angle_deg": best["angle_deg"],
        "score": best["score"],
        "diag": best["diag"],
        "trials": [{k: v for k, v in t.items() if k not in ("capture", "actions", "diag")} for t in trials],
    }


def _physical_finger_support(sim) -> tuple[bool, dict[str, Any]]:
    """Bilateral contact + force decomposition (not orientation-inferred support)."""
    s = sample_state(sim)
    cf = contact_forces(sim)
    sup = support_force_decomposition(sim)
    sides = sup.get("sides") or {}
    bil = bool(s["bilateral"] and cf["fn_L"] > 1.0 and cf["fn_R"] > 1.0)
    slim_sides = {}
    for k, v in sides.items():
        slim_sides[k] = {
            "n_contacts": v.get("n_contacts"),
            "Fn_z_normal": v.get("Fn_z_normal"),
            "Ft_z_friction": v.get("Ft_z_friction"),
            "normal_support_meaningful": v.get("normal_support_meaningful"),
        }
    return bil, {
        "fn_L": float(cf["fn_L"]),
        "fn_R": float(cf["fn_R"]),
        "fn_tot": float(cf["fn_tot"]),
        "sides": slim_sides,
        "alpha_up_deg": float(gates(sim)["alpha_up_deg"]),
        "u_open_world": (hand_R(sim) @ np.asarray(U_OPEN_HAND, float)).tolist(),
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "v_rel_norm": float(s["v_rel_norm"]),
        "w_rel_norm": float(s.get("w_rel_norm") or 0.0),
        "e_signed_mm": s["e_signed_mm"],
    }


def classify_tilt_physical_reason(tilt_result: dict | None, sim=None) -> str:
    """Map tilt failure to a physical event (not a blanket HORIZONTAL_TILT_FAILED)."""
    if not tilt_result:
        return "unknown"
    r = str(tilt_result.get("reason") or "")
    if r == "contact_loss":
        return "true_contact_loss"
    if r == "world_height":
        return "world_height_abort"
    if r == "timeout":
        return "controller_timeout"
    if r == "roll_leak_during_tilt":
        return "roll_leak"
    if r == "ok" and sim is not None:
        g = gates(sim)
        if abs(float(g["alpha_up_deg"]) - 90.0) > 12.0:
            return "orientation_target_not_reached"
    return r or "unknown"


def search_p1_p2_horizontal(env, blob, *, candidates: list[dict] | None = None) -> dict[str, Any]:
    """Bounded joint search over P1 support-roll and P2 tilt-to-horizontal.

    Each trial restores the same pre-P1 capture, executes legal actions, and scores
    contact retention via force decomposition — not hand orientation alone.
    Rejected trials are counterfactual; only the winner's parameters are returned
    for live re-execution on the accepted trajectory.
    """
    restore_capture(env, blob["capture"])
    th0 = float(sample_state(env.sim)["e_theta_deg"])
    e0 = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    auto_sign = float(choose_roll_sign_for_support(env.sim))
    y0 = reference_pad_vector(env.sim)
    cur0 = {"capture": blob["capture"], "ref": blob["ref"]}

    if not gates(env.sim)["fingertips_down"]:
        p0 = run_tilt_phase(
            env, cur0, target_dir_w=DOWN, phase="p12_search_P0", rate_scale=0.20, fn_hold=FN_HOLD,
            ang_tol_deg=5.0, y0_w=y0, e_ref=e0, allow_roll=False, contact_aware=False, max_time_s=8.0,
        )
        if not p0.get("ok"):
            return {
                "ok": False,
                "limiting": "P0_preparation_failed",
                "physical_reason": classify_tilt_physical_reason(p0),
                "n_trials": 0,
                "trials": [],
                "best": None,
                "counterfactual": True,
            }
        cur0 = {"capture": p0["capture"], "ref": blob["ref"]}

    if candidates is None:
        candidates = []
        for ang in (0.0, 25.0, 40.0, 55.0, 70.0):
            signs = (auto_sign,) if ang < 0.5 else (auto_sign, -auto_sign)
            for sgn in signs:
                for rate in (0.06, 0.10):
                    for zb in (0.05, 0.08):
                        candidates.append(
                            {
                                "support_roll_sign": float(sgn),
                                "support_roll_deg": float(ang),
                                "p2_rate": float(rate),
                                "z_boost": float(zb),
                                "ang_tol": 12.0,
                            }
                        )

    trials: list[dict] = []
    best: dict | None = None
    # Prefer low support-roll first (Plan B +dx succeeds with P1=0° in offline search).
    candidates = sorted(
        candidates,
        key=lambda c: (float(c.get("support_roll_deg") or 0.0), float(c.get("p2_rate") or 0.0)),
    )

    for c in candidates:
        restore_capture(env, cur0["capture"])
        e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
        y0_i = reference_pad_vector(env.sim)
        p1_actions: list = []
        if float(c["support_roll_deg"]) > 0.5:
            p1 = run_roll_phase(
                env, cur0, sign=float(c["support_roll_sign"]), angle_deg=float(c["support_roll_deg"]),
                phase="p12_search_P1", rate_scale=0.12, fn_hold=FN_HOLD, require_down=True,
                y0_w=y0_i, e_ref=e_ref,
            )
            if not p1.get("ok"):
                trials.append({
                    **c, "ok": False, "fail_phase": "P1", "reason": p1.get("reason"),
                    "limiting": "support_roll_failed", "counterfactual": True,
                })
                continue
            cur = {"capture": p1["capture"], "ref": blob["ref"]}
            p1_actions = list(p1.get("actions") or [])
        else:
            cur = cur0

        restore_capture(env, cur["capture"])
        bil1, sup1 = _physical_finger_support(env.sim)
        if not bil1:
            trials.append({
                **c, "ok": False, "fail_phase": "P1_support", "support": sup1,
                "limiting": "loss_of_bilateral_after_P1", "counterfactual": True,
            })
            continue

        u_tgt = _horizontal_open_target(env.sim)
        p2 = run_tilt_phase(
            env, cur, target_dir_w=u_tgt, phase="p12_search_P2",
            rate_scale=float(c["p2_rate"]), fn_hold=FN_HOLD,
            ang_tol_deg=float(c["ang_tol"]), y0_w=y0_i, e_ref=e_ref,
            allow_roll=False, contact_aware=True, max_time_s=16.0,
            z_ref_boost=float(c["z_boost"]), height_kp=4.5,
        )
        restore_capture(env, p2.get("capture") or cur["capture"])
        err = measure_errors(env.sim)
        bil2, sup2 = _physical_finger_support(env.sim)
        alpha = float(err["alpha_up_deg"])
        horiz = abs(alpha - 90.0) <= float(c["ang_tol"]) + 2.0
        theta_ok = float(err["dtheta_deg"]) <= max(th0 + 5.0, 12.0)
        z_ok = float(err["obj_z"]) >= Z_ABORT
        try:
            margin = float(fingertip_overlap_margin_mm(env.sim, U_OPEN_HAND))
        except Exception:
            margin = 0.0
        phys = classify_tilt_physical_reason(p2, env.sim)
        # P3 viability (counterfactual): margin alone is unreliable at horizontal;
        # require at least one sliding-roll candidate that keeps bilateral contact.
        p3_ok = False
        p3_best = None
        if bool(p2.get("ok") and horiz and bil2 and z_ok and theta_ok):
            d_lat = lateral_desire_direction(np.asarray(err["e_signed_mm"], float))
            if float(np.linalg.norm(d_lat)) >= 0.5:
                p3_probe = select_horizontal_sliding_roll(
                    env,
                    {"capture": p2["capture"], "ref": blob["ref"]},
                    d_lat,
                    y0_i,
                    np.asarray(err["e_signed_mm"], float),
                )
                p3_ok = bool(p3_probe.get("ok"))
                if p3_ok:
                    p3_best = {
                        "sign": p3_probe.get("sign"),
                        "angle_deg": p3_probe.get("angle_deg"),
                        "score": p3_probe.get("score"),
                    }
            restore_capture(env, p2["capture"])
        ok = bool(p2.get("ok") and horiz and bil2 and z_ok and theta_ok and p3_ok)
        score = (
            -abs(alpha - 90.0)
            + 0.05 * margin
            + (8.0 if bil2 else -30.0)
            + (3.0 if theta_ok else -10.0)
            + (12.0 if p3_ok else -20.0)
            + 0.02 * float(sup2.get("fn_tot") or 0.0)
            + float((p3_best or {}).get("score") or 0.0) * 0.01
        )
        limiting = None
        if not ok:
            if not p2.get("ok"):
                limiting = phys
            elif not bil2:
                limiting = "true_contact_loss"
            elif not z_ok:
                limiting = "world_height_abort"
            elif not horiz:
                limiting = "orientation_target_not_reached"
            elif not theta_ok:
                limiting = "dtheta_not_preserved"
            elif not p3_ok:
                limiting = "no_viable_P3_after_horizontal"
            else:
                limiting = "unknown"
        row = {
            **c,
            "ok": ok,
            "score": score,
            "p1_ok": True,
            "p2_ok": bool(p2.get("ok")),
            "p2_reason": p2.get("reason"),
            "p3_viable": p3_ok,
            "p3_best": p3_best,
            "physical_reason": phys if not ok else "ok",
            "limiting": limiting,
            "alpha_up_deg": alpha,
            "bilateral": bil2,
            "support_after_P1": {k: sup1[k] for k in ("fn_L", "fn_R", "fn_tot", "sides") if k in sup1},
            "support_after_P2": {k: sup2[k] for k in ("fn_L", "fn_R", "fn_tot", "sides") if k in sup2},
            "margin_mm": margin,
            "dtheta0": th0,
            "dtheta": float(err["dtheta_deg"]),
            "obj_z": float(err["obj_z"]),
            "e_signed_mm": err["e_signed_mm"],
            "counterfactual": True,
        }
        trials.append(row)
        if ok and (best is None or score > float(best["score"])):
            best = {
                **row,
                "capture": p2["capture"],
                "actions_p1": p1_actions,
                "actions_p2": list(p2.get("actions") or []),
                "pre_p1_capture": cur0["capture"],
            }
            # Early exit: first viable candidate is enough for live re-execution.
            break

    restore_capture(env, blob["capture"])
    if best is None:
        lim_counts: dict[str, int] = {}
        for t in trials:
            key = str(t.get("limiting") or t.get("fail_phase") or "unknown")
            lim_counts[key] = lim_counts.get(key, 0) + 1
        limiting = max(lim_counts, key=lim_counts.get) if lim_counts else "no_candidates"
        return {
            "ok": False,
            "best": None,
            "trials": trials,
            "n_trials": len(trials),
            "limiting": limiting,
            "limiting_counts": lim_counts,
            "counterfactual": True,
            "physical_reason": limiting,
        }
    return {
        "ok": True,
        "best": {k: v for k, v in best.items() if k not in ("capture", "actions_p1", "actions_p2", "pre_p1_capture")},
        "trials": trials,
        "n_trials": len(trials),
        "limiting": None,
        "counterfactual": True,
        "pre_p1_capture": best["pre_p1_capture"],
        "p1_p2_capture": best["capture"],
        "actions_p1": best["actions_p1"],
        "actions_p2": best["actions_p2"],
        "winner_params": {
            "support_roll_sign": best["support_roll_sign"],
            "support_roll_deg": best["support_roll_deg"],
            "p2_rate": best["p2_rate"],
            "z_boost": best["z_boost"],
            "ang_tol": best["ang_tol"],
        },
    }


def correct_dx_horizontal_attempt(env, blob, cfg: PrecisionConfig) -> dict[str, Any]:
    """Full horizontal-gripper dx: P0–P9 including P3/P6 sliding rolls."""
    phase_log: list[dict] = []
    actions: list = []
    restore_capture(env, blob["capture"])
    y0 = reference_pad_vector(env.sim)
    err0 = measure_errors(env.sim)
    e0 = np.asarray(err0["e_signed_mm"], float)
    th0 = float(err0["dtheta_deg"])
    # Skip unnecessary manipulation when geometry is already SUCCESS_NEW-compatible.
    if handoff_geometry_ready(err0, cfg):
        return {
            "ok": True,
            "skipped": True,
            "reason": STOP_HANDOFF_READY,
            "stop_reason": STOP_HANDOFF_READY,
            "goal_reached": True,
            "partial_progress": False,
            "dx_solved": abs(float(e0[0])) <= float(cfg.ex_tol_mm),
            "actions": [],
            "capture": blob["capture"],
            "errors": err0,
            "errors_entry": {"e_signed_mm": e0.tolist(), "dtheta_deg": th0, "e_pos_mm": float(err0["e_pos_mm"])},
            "disp_total_mm": 0.0,
            "contract": "goal_driven",
        }
    d_lat = lateral_desire_direction(e0)
    if float(np.linalg.norm(d_lat)) < 0.5:
        return {"ok": True, "skipped": True, "reason": "dx_near_zero", "actions": [], "capture": blob["capture"]}

    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    e_ref = e0.copy()

    def _fail(code, phase, cur_cap, extra=None):
        restore_capture(env, cur_cap)
        extra = dict(extra or {})
        phys = extra.get("physical_reason")
        out = {
            "ok": False,
            "mechanism": "horizontal_dx",
            "failure_code": code,
            "phase": phase,
            "reason": phys or code,
            "physical_reason": phys,
            "actions": list(actions),
            "capture": take_capture(env),
            "phase_log": phase_log,
            "failure": _failure_snapshot(env.sim, phase=phase, code=code, last_action=actions[-1] if actions else None),
            "n_accepted_actions": len(actions),
            "rejected_branch": bool(extra.get("rejected_branch", False)),
            "in_selected_trajectory": bool(extra.get("in_selected_trajectory", True)),
        }
        out.update(extra)
        return out

    # P0: fingertips-down
    if not gates(env.sim)["fingertips_down"]:
        p0 = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="dxH_P0", rate_scale=0.20, fn_hold=FN_HOLD,
            ang_tol_deg=5.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=False, max_time_s=8.0,
        )
        actions += p0.get("actions") or []
        phase_log.append({"phase": "P0", "ok": p0.get("ok"), "errors": measure_errors(env.sim)})
        if not p0.get("ok"):
            return _fail(
                FAIL_PREPARATION, "P0", p0.get("capture") or cur["capture"],
                {"physical_reason": classify_tilt_physical_reason(p0)},
            )
        cur = {"capture": p0["capture"], "ref": blob["ref"]}
    else:
        phase_log.append({"phase": "P0", "ok": True, "skipped": True, "errors": measure_errors(env.sim)})

    # P1 + P2: support roll then tilt to horizontal
    roll_sign = float(choose_roll_sign_for_support(env.sim))
    support_roll_deg = 55.0
    p2_rate, p2_boost, p2_tol = 0.12, 0.05, 10.0

    if bool(getattr(cfg, "use_p1_p2_search", False)):
        wp = getattr(cfg, "p1_p2_winner", None)
        search = None
        if not isinstance(wp, dict) or not wp:
            search = search_p1_p2_horizontal(env, cur)
            phase_log.append(
                {
                    "phase": "P1_P2_search",
                    "ok": search.get("ok"),
                    "best": search.get("best"),
                    "n_trials": search.get("n_trials"),
                    "limiting": search.get("limiting"),
                    "counterfactual": True,
                    "in_selected_trajectory": False,
                }
            )
            if not search.get("ok"):
                return _fail(
                    FAIL_HORIZONTAL_TILT,
                    "P2",
                    cur["capture"],
                    {
                        "physical_reason": search.get("physical_reason") or search.get("limiting"),
                        "P1_P2_search": {
                            k: search[k]
                            for k in ("ok", "best", "n_trials", "limiting", "limiting_counts", "trials")
                            if k in search
                        },
                    },
                )
            wp = search["winner_params"]
        else:
            phase_log.append(
                {
                    "phase": "P1_P2_search",
                    "ok": True,
                    "best": wp,
                    "reused_winner": True,
                    "counterfactual": True,
                    "in_selected_trajectory": False,
                }
            )
        roll_sign = float(wp["support_roll_sign"])
        support_roll_deg = float(wp["support_roll_deg"])
        p2_rate = float(wp["p2_rate"])
        p2_boost = float(wp["z_boost"])
        p2_tol = float(wp.get("ang_tol", 12.0))
        restore_capture(env, cur["capture"])
        e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
        y0 = reference_pad_vector(env.sim)

    if support_roll_deg > 0.5:
        p1 = run_roll_phase(
            env, cur, sign=roll_sign, angle_deg=support_roll_deg, phase="dxH_P1",
            rate_scale=0.12, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
        )
        actions += p1.get("actions") or []
        phase_log.append({
            "phase": "P1", "ok": p1.get("ok"), "angle_deg": support_roll_deg,
            "sign": roll_sign, "errors": measure_errors(env.sim),
            "in_selected_trajectory": True,
        })
        if not p1.get("ok"):
            return _fail(
                FAIL_SUPPORT_ROLL, "P1", p1.get("capture") or cur["capture"],
                {"physical_reason": p1.get("reason")},
            )
        cur = {"capture": p1["capture"], "ref": blob["ref"]}
        bil1, sup1 = _physical_finger_support(env.sim)
        phase_log.append({"phase": "P1_support", "ok": bil1, "support": sup1, "in_selected_trajectory": True})
        if not bil1:
            return _fail(
                FAIL_SUPPORT_ROLL, "P1_support", cur["capture"],
                {"physical_reason": "loss_of_bilateral_after_P1", "support": sup1},
            )
    else:
        phase_log.append({"phase": "P1", "ok": True, "skipped": True, "angle_deg": 0.0, "in_selected_trajectory": True})

    restore_capture(env, cur["capture"])
    u_tgt = _horizontal_open_target(env.sim)
    p2 = run_tilt_phase(
        env, cur, target_dir_w=u_tgt, phase="dxH_P2", rate_scale=p2_rate, fn_hold=FN_HOLD,
        ang_tol_deg=p2_tol, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=16.0,
        z_ref_boost=p2_boost, height_kp=4.5,
    )
    if not p2.get("ok") and not bool(getattr(cfg, "use_p1_p2_search", False)):
        rejected_p2 = {
            "reason": p2.get("reason"),
            "physical_reason": classify_tilt_physical_reason(p2),
            "counterfactual": True,
            "in_selected_trajectory": False,
            "n_actions_rejected": len(p2.get("actions") or []),
        }
        phase_log.append({"phase": "P2_attempt", "ok": False, **rejected_p2})
        restore_capture(env, cur["capture"])
        p2b = run_tilt_phase(
            env, cur, target_dir_w=u_tgt, phase="dxH_P2_retry", rate_scale=0.08, fn_hold=FN_HOLD,
            ang_tol_deg=12.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=18.0,
            z_ref_boost=0.06, height_kp=5.0,
        )
        if p2b.get("ok"):
            p2 = p2b
        else:
            phase_log.append({
                "phase": "P2", "ok": False, "alpha": gates(env.sim)["alpha_up_deg"],
                "retry_failed": True, "physical_reason": classify_tilt_physical_reason(p2b),
                "in_selected_trajectory": True,
            })
            return _fail(
                FAIL_HORIZONTAL_TILT, "P2", p2b.get("capture") or p2.get("capture") or cur["capture"],
                {"physical_reason": classify_tilt_physical_reason(p2b), "first_attempt": rejected_p2},
            )
    elif not p2.get("ok"):
        actions += p2.get("actions") or []
        phase_log.append({
            "phase": "P2", "ok": False, "alpha": gates(env.sim)["alpha_up_deg"],
            "physical_reason": classify_tilt_physical_reason(p2),
            "in_selected_trajectory": True,
        })
        return _fail(
            FAIL_HORIZONTAL_TILT, "P2", p2.get("capture") or cur["capture"],
            {"physical_reason": classify_tilt_physical_reason(p2)},
        )
    actions += p2.get("actions") or []
    bil2, sup2 = _physical_finger_support(env.sim)
    phase_log.append({
        "phase": "P2", "ok": p2.get("ok"), "alpha": gates(env.sim)["alpha_up_deg"],
        "errors": measure_errors(env.sim), "support": sup2, "in_selected_trajectory": True,
    })
    if not p2.get("ok") or not bil2:
        return _fail(
            FAIL_HORIZONTAL_TILT, "P2", p2.get("capture") or cur["capture"],
            {
                "physical_reason": "true_contact_loss" if not bil2 else classify_tilt_physical_reason(p2),
                "support": sup2,
            },
        )
    cur = {"capture": p2["capture"], "ref": blob["ref"]}

    # P3: select + execute sliding roll (counterfactual search; only winner enters tape)
    sel = select_horizontal_sliding_roll(env, cur, d_lat, y0, e_ref)
    if not sel.get("ok"):
        return _fail(FAIL_SLIDING_ROLL, "P3_select", cur["capture"], {"P3_search": sel})
    p3 = run_roll_phase(
        env, cur, sign=float(sel["sign"]), angle_deg=float(sel["angle_deg"]), phase="dxH_P3",
        rate_scale=0.12, fn_hold=FN_HOLD, require_down=False, y0_w=y0, e_ref=e_ref,
    )
    actions += p3.get("actions") or []
    phase_log.append({"phase": "P3", "ok": p3.get("ok"), "sign": sel["sign"], "angle_deg": sel["angle_deg"], "errors": measure_errors(env.sim)})
    if not p3.get("ok"):
        return _fail(FAIL_SLIDING_ROLL, "P3", p3.get("capture") or cur["capture"], {"P3_search": sel})
    cur = {"capture": p3["capture"], "ref": blob["ref"]}
    restore_capture(env, cur["capture"])
    sc = _score_horizontal_slide_config(env.sim, d_lat)
    if not sc["ok"]:
        return _fail(FAIL_SLIDING_ROLL, "P3_geometry", cur["capture"], {"geometry": sc, "P3_search": sel})

    # P4–P5: goal-driven DualLoop lateral (max_safe_travel only; stop on dx / e_pos region)
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    max_travel = float(getattr(cfg, "horizontal_slip_mm", 14.0) or 14.0)
    if bool(getattr(cfg, "use_goal_driven_slip", True)):
        slip_goal = SlipGoal(
            axis="dx",
            desired_direction=d_lat,
            max_safe_travel_mm=max_travel,
            max_duration_s=12.0,
            axis_tol_mm=float(cfg.ex_tol_mm),
            working_pos_mm=float(cfg.working_pos_mm),
            working_theta_deg=float(cfg.working_theta_deg),
            ey_tol_mm=float(cfg.ey_tol_mm),
        )
        slip = run_goal_aware_axial_slip(
            env, cur, slip_goal, z_ref=z_ref, guard_lateral=False, guard_fingertip_margin=False,
        )
    else:
        milestone = max_travel
        if bool(getattr(cfg, "horizontal_legacy_total", False)):
            milestone = 2.0 + max_travel
        slip = run_staged_axial_slip(
            env, cur, d_lat, milestone_mm=milestone, z_ref=z_ref,
            e_x0=None, guard_lateral=False, guard_fingertip_margin=False,
        )
    actions += slip.get("actions") or []
    phase_log.append(
        {
            "phase": "P4_P5",
            "ok": slip.get("ok"),
            "disp_along_mm": slip.get("disp_along_mm"),
            "disp_total_mm": slip.get("disp_total_mm"),
            "stop_reason": slip.get("stop_reason"),
            "goal_reached": slip.get("goal_reached"),
            "contract": slip.get("contract"),
            "errors": measure_errors(env.sim) if slip.get("capture") else None,
            "failure_code": slip.get("failure_code"),
            "in_selected_trajectory": True,
        }
    )
    if not slip.get("ok"):
        return _fail(
            slip.get("failure_code") or FAIL_SLIP_CONTACT,
            "P4_P5",
            slip.get("capture") or cur["capture"],
            {
                "physical_reason": slip.get("reason") or slip.get("abort_reason") or slip.get("failure_code"),
                "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
            },
        )
    # Coupling check — warn, but only abort on severe loss of grasp geometry
    e_post = np.asarray(slip["e_signed"], float)
    th_post = float(measure_errors(env.sim)["dtheta_deg"])
    if abs(float(e_post[2]) - float(e0[2])) > 8.0 and abs(float(e_post[2])) > abs(float(e0[2])) + 4.0:
        return _fail(
            FAIL_SLIP_WRONG_DIR, "P4_P5_dz_regression", slip["capture"],
            {"e0": e0.tolist(), "e_post": e_post.tolist()},
        )
    if th_post > 15.0 and (th_post - th0) > 10.0:
        return _fail(FAIL_SLIP_WRONG_DIR, "P4_P5_theta_regression", slip["capture"], {"th0": th0, "th_post": th_post})
    if th_post > 10.0:
        phase_log.append({"phase": "P4_P5_theta_warn", "th0": th0, "th_post": th_post})
    cur = {"capture": slip["capture"], "ref": blob["ref"]}
    e_ref = e_post.copy()

    # P6: reverse P3 sliding roll (still near-horizontal)
    p6 = run_roll_phase(
        env, cur, sign=-float(sel["sign"]), angle_deg=float(sel["angle_deg"]), phase="dxH_P6",
        rate_scale=0.12, fn_hold=FN_HOLD, require_down=False, y0_w=y0, e_ref=e_ref,
    )
    actions += p6.get("actions") or []
    phase_log.append({"phase": "P6", "ok": p6.get("ok"), "errors": measure_errors(env.sim)})
    if not p6.get("ok"):
        return _fail(FAIL_REVERSE_ROLL, "P6", p6.get("capture") or cur["capture"])
    cur = {"capture": p6["capture"], "ref": blob["ref"]}

    # P7: tilt back to fingertips-down (counterfactual retry; rejected attempt not concatenated)
    p7 = run_tilt_phase(
        env, cur, target_dir_w=DOWN, phase="dxH_P7", rate_scale=0.16, fn_hold=FN_HOLD,
        ang_tol_deg=5.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=14.0,
        z_ref_boost=0.03, height_kp=4.0,
    )
    if not p7.get("ok"):
        phase_log.append({
            "phase": "P7_attempt",
            "ok": False,
            "physical_reason": classify_tilt_physical_reason(p7),
            "counterfactual": True,
            "in_selected_trajectory": False,
            "n_actions_rejected": len(p7.get("actions") or []),
        })
        restore_capture(env, cur["capture"])
        p7b = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="dxH_P7_retry", rate_scale=0.10, fn_hold=FN_HOLD,
            ang_tol_deg=7.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=16.0,
            z_ref_boost=0.06, height_kp=5.0,
        )
        if p7b.get("ok"):
            p7 = p7b
        else:
            # Live failure after retry — keep retry actions as the executed failure prefix
            actions += p7b.get("actions") or []
            phase_log.append({
                "phase": "P7", "ok": False, "errors": measure_errors(env.sim),
                "physical_reason": classify_tilt_physical_reason(p7b),
                "in_selected_trajectory": True,
            })
            return _fail(
                FAIL_RETURN_TILT, "P7", p7b.get("capture") or cur["capture"],
                {"physical_reason": classify_tilt_physical_reason(p7b)},
            )
    actions += p7.get("actions") or []
    phase_log.append({"phase": "P7", "ok": p7.get("ok"), "errors": measure_errors(env.sim), "in_selected_trajectory": True})
    if not p7.get("ok"):
        return _fail(
            FAIL_RETURN_TILT, "P7", p7.get("capture") or cur["capture"],
            {"physical_reason": classify_tilt_physical_reason(p7)},
        )
    cur = {"capture": p7["capture"], "ref": blob["ref"]}

    # P8: reverse P1 support roll (require down)
    p8 = run_roll_phase(
        env, cur, sign=-roll_sign, angle_deg=support_roll_deg, phase="dxH_P8",
        rate_scale=0.12, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    actions += p8.get("actions") or []
    phase_log.append({"phase": "P8", "ok": p8.get("ok"), "errors": measure_errors(env.sim)})
    if not p8.get("ok"):
        return _fail(FAIL_FINAL_UNROLL, "P8", p8.get("capture") or cur["capture"])
    restore_capture(env, p8["capture"])
    settle_hold(env, actions, hold_s=0.4)
    err_f = measure_errors(env.sim)
    phase_log.append({"phase": "P9", "ok": True, "errors": err_f})
    # Safe full P0–P9 with meaningful progress (need not solve entire SUCCESS_NEW alone).
    dx_improved = abs(float(err_f["dx_mm"])) < abs(float(e0[0])) - 0.3
    e_pos0 = float(np.linalg.norm(e0))
    e_pos_f = float(err_f["e_pos_mm"])
    pos_improved = e_pos_f < e_pos0 - 0.3
    theta_ok = float(err_f["dtheta_deg"]) <= max(th0 + 8.0, 15.0)
    dz_ok = abs(float(err_f["dz_mm"])) <= abs(float(e0[2])) + 8.0
    partial_ok = bool(err_f["bilateral"] and theta_ok and dz_ok and (dx_improved or pos_improved))
    ok_h = partial_ok
    return {
        "ok": ok_h,
        "mechanism": "horizontal_dx",
        "failure_code": None if ok_h else FAIL_SLIP_TARGET,
        "partial_progress": bool(dx_improved or pos_improved),
        "dx_solved": abs(float(err_f["dx_mm"])) <= float(cfg.ex_tol_mm),
        "stop_reason": slip.get("stop_reason"),
        "goal_reached": slip.get("goal_reached"),
        "P3_search": {k: sel[k] for k in sel if k != "trials"},
        "P3_trials": sel.get("trials"),
        "geometry": sc,
        "disp_along_mm": slip.get("disp_along_mm"),
        "disp_total_mm": slip.get("disp_total_mm"),
        "contract": slip.get("contract"),
        "support_roll_sign": roll_sign,
        "support_roll_deg": support_roll_deg,
        "phase_log": phase_log,
        "actions": actions,
        "capture": take_capture(env),
        "errors": err_f,
        "errors_entry": {"e_signed_mm": e0.tolist(), "dtheta_deg": th0, "e_pos_mm": e_pos0},
        "rejected_branch": False,
        "n_accepted_actions": len(actions),
    }

