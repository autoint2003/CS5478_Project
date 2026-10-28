"""Experimental Lever-Arm Assisted Pivot (phases A–D). Not production.

Legal Panda actions only. Requires measured relative cylinder rotation (not
world-θ alone / not rigid co-rotation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.corrections.contact_primitives import (
    MAX_DELTA_P_REL_MM,
    _action_vw,
    _wrist,
    classify_pivot_motion,
    relative_pose,
)
from data_generation.oracles.precision.corrections.lateral import _horizontal_open_target
from data_generation.oracles.precision.mechanics.dual_loop import DT, make_action, sample_state
from data_generation.oracles.precision.mechanics.open_axis import (
    DOWN,
    choose_roll_sign_for_support,
    gates,
    reference_pad_vector,
    run_roll_phase,
    run_tilt_phase,
)
from data_generation.oracles.precision.mechanics.reverse_rotation import HoldModeGrip
from data_generation.oracles.precision.mechanics.rotation_braking import braking_distance_deg
from data_generation.oracles.precision.mechanics.support_verification import (
    classify_contact_mode,
    make_s1_state,
)
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import FN_HOLD
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import cylinder_axis_hand, cylinder_axis_world
from evaluation.success_criteria import U_TGT
from experiments.precision.lever_arm_assisted_pivot.geometry import M_PROXY, G_WORLD, measure_lever_geometry

# Diagnostic reference only — does not change SUCCESS_NEW / SUCCESS_12S.
DIAG_MAX_DELTA_P_MM = MAX_DELTA_P_REL_MM  # 3 mm

FAILURE = {
    "LEVER_ARM_UNREACHABLE",
    "WRONG_TORQUE_DIRECTION",
    "SUPPORT_TRANSFER_FAILED",
    "SUPPORT_LOST_DURING_TILT",
    "PIVOT_SUPPORT_NOT_ESTABLISHED",
    "RIGID_CO_ROTATION",
    "WRONG_DIRECTION_ROTATION",
    "UNCONTROLLED_TRANSLATION",
    "UNCONTROLLED_SLIP",
    "BRAKE_FAILED",
    "REGRIP_FAILED",
    "NO_MEANINGFUL_RELATIVE_ROTATION",
}


def _unit(v) -> np.ndarray:
    a = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.zeros(3)


def _desired_axis(sim) -> np.ndarray:
    u = _unit(cylinder_axis_world(sim))
    tgt = _unit(U_TGT)
    ax = np.cross(u, tgt)
    n = float(np.linalg.norm(ax))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return ax / n


def _predict_tau_g_for_axial_shift(sim, axial_shift_m: float, support_side: str | None) -> dict[str, Any]:
    """Predict gravity torque if COM−contact axial lever changes by shift (meters)."""
    geo = measure_lever_geometry(sim)
    side = support_side or geo.get("support_side")
    prim = (geo.get("per_side") or {}).get(side) if side else geo.get("primary")
    if prim is None:
        # fall back to either side
        for s in ("L", "R"):
            if (geo.get("per_side") or {}).get(s):
                prim = geo["per_side"][s]
                side = s
                break
    if prim is None:
        return {"ok": False, "reason": "no_contact"}
    u = _unit(cylinder_axis_world(sim))
    r0 = np.asarray(prim["r_com_minus_contact_w"], float)
    # Shift contact along −u by axial_shift (hand moves +u ⇒ contact on cylinder moves −u in COM frame)
    r1 = r0 + float(axial_shift_m) * u
    tau = np.cross(r1, M_PROXY * G_WORLD)
    ax = _desired_axis(sim)
    return {
        "ok": True,
        "side": side,
        "axial_lever_m": float(np.dot(r1, u)),
        "tau_g_along_des": float(np.dot(tau, ax)),
        "tau_g_w": tau.tolist(),
    }


def phase_a_lever_arm(
    env,
    blob: dict,
    *,
    target_axial_mm: float | None = None,
    candidates_mm: tuple[float, ...] = (-12.0, -8.0, -5.0, 5.0, 8.0, 12.0),
    v_axial: float = 0.025,
    grip: float = 0.95,
    max_s: float = 4.0,
    force_sign: float | None = None,
) -> dict[str, Any]:
    """Create a temporary axial offset with controlled hand motion along cylinder axis."""
    restore_capture(env, blob["capture"])
    if not sample_state(env.sim).get("bilateral"):
        return {"ok": False, "failure": "LEVER_ARM_UNREACHABLE", "reason": "no_bilateral", "actions": [], "capture": blob["capture"]}

    geo0 = measure_lever_geometry(env.sim)
    axis_des = _desired_axis(env.sim)
    side = geo0.get("support_side")
    # Prefer candidate maximizing |τ_g · axis_des| with matching corrective sign (reduce dθ)
    # Corrective: positive tau along axis_des rotates u toward U_TGT.
    scored = []
    for c_mm in candidates_mm:
        if force_sign is not None and np.sign(c_mm) != np.sign(force_sign) and abs(c_mm) > 1e-6:
            continue
        pred = _predict_tau_g_for_axial_shift(env.sim, c_mm * 1e-3, side)
        if not pred.get("ok"):
            continue
        # Reject if fingertip margin would be unsafe: |dz + c| roughly; use |primary lever + c|
        lever0 = float((geo0.get("primary") or {}).get("axial_lever_mm") or geo0.get("dz_mm") or 0.0)
        if abs(lever0 + c_mm) > 28.0:
            continue
        scored.append({"candidate_mm": c_mm, **pred, "score": abs(float(pred["tau_g_along_des"]))})

    if target_axial_mm is not None:
        # force specific offset magnitude/sign
        scored = [
            {
                "candidate_mm": float(target_axial_mm),
                **_predict_tau_g_for_axial_shift(env.sim, float(target_axial_mm) * 1e-3, side),
                "score": 1.0,
            }
        ]
        if not scored[0].get("ok"):
            return {
                "ok": False,
                "failure": "LEVER_ARM_UNREACHABLE",
                "reason": "predict_failed",
                "geo0": geo0,
                "actions": [],
                "capture": blob["capture"],
            }

    if not scored:
        return {
            "ok": False,
            "failure": "LEVER_ARM_UNREACHABLE",
            "reason": "no_safe_candidate",
            "geo0": geo0,
            "actions": [],
            "capture": blob["capture"],
        }

    # Choose max |τ|; if force_sign set, already filtered
    best = max(scored, key=lambda r: r["score"])
    # Prefer torque that helps reduce dθ (positive along des). If all negative, still take max |τ|
    # but flag WRONG_TORQUE_DIRECTION if best is strongly wrong-signed and alternatives exist.
    helpful = [s for s in scored if float(s["tau_g_along_des"]) > 0.0]
    if helpful:
        best = max(helpful, key=lambda r: float(r["tau_g_along_des"]))
    elif force_sign is None and float(best["tau_g_along_des"]) < -1e-5:
        # Try opposite-preferring: keep best magnitude but note warning
        pass

    target_mm = float(best["candidate_mm"])
    u_h = _unit(cylinder_axis_hand(env.sim))
    # Hand +u moves grasp toward +axis of cylinder → contact slides −axis on object
    v_h = np.sign(target_mm) * abs(v_axial) * u_h
    actions: list = []
    series: list = []
    lever0 = float((geo0.get("primary") or {}).get("axial_lever_mm") or 0.0)
    n_steps = max(1, int(round(max_s / DT)))
    lost = False
    for i in range(n_steps):
        a = _action_vw(v_h.tolist(), [0.0, 0.0, 0.0], grip)
        step_legal(env.sim, a)
        actions.append(a.copy())
        st = sample_state(env.sim)
        if not st.get("bilateral") or latched(env.sim):
            lost = True
            break
        geo = measure_lever_geometry(env.sim)
        lever = float((geo.get("primary") or {}).get("axial_lever_mm") or 0.0)
        traveled = lever - lever0
        # Stop when achieved travel matches target_mm in sign and ~70% magnitude
        if abs(target_mm) > 1e-3 and np.sign(traveled) == np.sign(target_mm) and abs(traveled) >= 0.7 * abs(target_mm):
            series.append({"i": i, "lever_mm": lever, "traveled_mm": traveled, "stop": "target"})
            break
        if i % 4 == 0:
            series.append({"i": i, "lever_mm": lever, "traveled_mm": traveled, "dz_mm": geo.get("dz_mm")})

    # Settle
    for _ in range(int(0.25 / DT)):
        a = _wrist(0.0, 0.0, grip)
        step_legal(env.sim, a)
        actions.append(a.copy())

    geo1 = measure_lever_geometry(env.sim)
    lever1 = float((geo1.get("primary") or {}).get("axial_lever_mm") or 0.0)
    achieved = lever1 - lever0
    pred1 = _predict_tau_g_for_axial_shift(env.sim, 0.0, side)  # current
    bil = bool(sample_state(env.sim).get("bilateral"))
    # Accept modest travel, or retention of an already-large lever without contact loss.
    ok = (not lost) and bil and (
        abs(achieved) >= 1.5
        or (abs(lever1) >= 6.0 and abs(achieved) >= 0.5)
        or (abs(lever0) >= 8.0 and abs(achieved) < 1.5 and abs(lever1) >= 6.0)
    )
    failure = None
    if lost or not bil:
        failure = "LEVER_ARM_UNREACHABLE"
        # Restore IC so downstream paired metrics are not dominated by a freefall wreck.
        restore_capture(env, blob["capture"])
        geo1 = measure_lever_geometry(env.sim)
        lever1 = float((geo1.get("primary") or {}).get("axial_lever_mm") or 0.0)
        achieved = 0.0
    elif not ok:
        failure = "LEVER_ARM_UNREACHABLE"

    return {
        "ok": bool(ok),
        "failure": failure,
        "target_mm": target_mm,
        "lever0_mm": lever0,
        "lever1_mm": lever1,
        "achieved_travel_mm": achieved,
        "predicted_tau_g": best.get("tau_g_along_des"),
        "tau_g_after": (geo1.get("primary") or {}).get("tau_g_along_des"),
        "candidates_scored": [{k: v for k, v in s.items() if k != "tau_g_w"} for s in scored],
        "geo0": geo0,
        "geo1": geo1,
        "series": series,
        "actions": actions,
        "capture": take_capture(env),
        "errors": measure_errors(env.sim),
    }


def phase_b_protected_rotation(env, blob: dict, *, rate_scale: float = 0.08) -> dict[str, Any]:
    """Establish S1 support bias and rotate toward horizontal under support_guard."""
    restore_capture(env, blob["capture"])
    actions: list = []
    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    roll_sign = float(choose_roll_sign_for_support(env.sim))

    if not gates(env.sim)["fingertips_down"]:
        p0 = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="LAP_B_prep_down", rate_scale=0.14, fn_hold=FN_HOLD,
            ang_tol_deg=5.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=8.0,
        )
        actions += p0.get("actions") or []
        if not (p0.get("ok") and gates(env.sim)["fingertips_down"]):
            return {
                "ok": False,
                "failure": "SUPPORT_TRANSFER_FAILED",
                "reason": p0.get("reason") or "prep_down_failed",
                "actions": actions,
                "capture": p0.get("capture") or blob["capture"],
            }
        cur = {"capture": p0["capture"], "ref": blob["ref"]}

    p1 = run_roll_phase(
        env, cur, sign=roll_sign, angle_deg=12.0, phase="LAP_B_S1",
        rate_scale=0.14, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    actions += p1.get("actions") or []
    mode1 = classify_contact_mode(env.sim)
    support_side = mode1.get("support_side") or ("R" if roll_sign < 0 else "L")
    if not p1.get("ok"):
        return {
            "ok": False,
            "failure": "SUPPORT_TRANSFER_FAILED",
            "reason": "S1_roll_failed",
            "actions": actions,
            "capture": p1.get("capture") or cur["capture"],
            "support_side": support_side,
        }
    cur = {"capture": p1["capture"], "ref": blob["ref"]}

    u_tgt = _horizontal_open_target(env.sim)
    p2 = run_tilt_phase(
        env, cur, target_dir_w=u_tgt, phase="LAP_B_horizontal",
        rate_scale=rate_scale, fn_hold=FN_HOLD, ang_tol_deg=12.0, y0_w=y0, e_ref=e_ref,
        allow_roll=False, contact_aware=True, max_time_s=22.0, z_ref_boost=0.05, height_kp=5.0,
        support_guard=True, support_side=support_side,
    )
    actions += p2.get("actions") or []
    mode2 = classify_contact_mode(env.sim)
    alpha = float(gates(env.sim)["alpha_up_deg"])
    near_h = 70.0 <= alpha <= 120.0
    support_side = mode2.get("support_side") or support_side
    fnz_L = float(mode2.get("Fn_z_L") or 0.0)
    fnz_R = float(mode2.get("Fn_z_R") or 0.0)
    fnz_sup = fnz_L if support_side == "L" else fnz_R
    fnz_oth = fnz_R if support_side == "L" else fnz_L
    s1 = bool(mode2.get("one_sided_mechanical")) or (fnz_sup > 2.0 and fnz_sup > 1.5 * max(fnz_oth, 0.5))
    support_held = bool(
        sample_state(env.sim).get("bilateral")
        or mode2.get("one_sided_mechanical")
        or (support_side == "L" and float(mode2.get("fn_L") or 0) >= 0.8)
        or (support_side == "R" and float(mode2.get("fn_R") or 0) >= 0.8)
    )

    failure = None
    if p2.get("reason") in ("support_lost", "uncontrolled_slip", "contact_loss") or not support_held:
        failure = "SUPPORT_LOST_DURING_TILT"
    elif not near_h or not s1:
        failure = "PIVOT_SUPPORT_NOT_ESTABLISHED" if near_h else "SUPPORT_LOST_DURING_TILT"

    return {
        "ok": failure is None and bool(p2.get("ok") or near_h) and s1 and support_held,
        "failure": failure,
        "reason": p2.get("reason"),
        "support_side": support_side,
        "roll_sign": roll_sign,
        "alpha_up_deg": alpha,
        "s1_verified": s1,
        "mode": mode2,
        "geo": measure_lever_geometry(env.sim),
        "actions": actions,
        "capture": take_capture(env),
        "errors": measure_errors(env.sim),
        "s1_state": make_s1_state(env.sim, roll_sign=roll_sign, force_valid=False),
    }


def phase_c_controlled_release(
    env,
    blob: dict,
    *,
    grip_start: float = 0.55,
    grip_end: float = -0.15,
    release_s: float = 2.5,
    hold_height: bool = True,
) -> dict[str, Any]:
    """Reduce rotational constraint; allow gravity/contact torque relative rotation."""
    restore_capture(env, blob["capture"])
    p0 = relative_pose(env.sim)
    err0 = measure_errors(env.sim)
    geo0 = measure_lever_geometry(env.sim)
    u0_h = np.asarray(p0["u_hand"], float)
    th0 = float(err0["dtheta_deg"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    actions: list = []
    series: list = []
    n = max(1, int(round(release_s / DT)))
    lost = False
    wrong_dir = False
    max_w_rel = 0.0
    peak_rel = 0.0

    for i in range(n):
        frac = i / max(n - 1, 1)
        grip = float(grip_start + (grip_end - grip_start) * frac)
        if hold_height:
            # small upward world hold via hand-frame: use zero ω, tiny vertical via height hold helper
            from data_generation.oracles.precision.corrections.orientation import _height_hold_v_hand

            v = _height_hold_v_hand(env.sim, z_ref, kp=3.0)
            a = make_action(v, np.zeros(3), grip)
        else:
            a = _wrist(0.0, 0.0, grip)
        step_legal(env.sim, a)
        actions.append(np.asarray(a, np.float32).copy())
        st = sample_state(env.sim)
        pr = relative_pose(env.sim)
        failed, _ = recovery_failed(env.sim, blob["ref"], 0)
        if failed or latched(env.sim) or (not st.get("bilateral") and float(st.get("fn_tot") or 0) < 0.5):
            # Allow one-sided support; only fail if total force collapses
            if float(st.get("fn_tot") or 0) < 0.4 or latched(env.sim):
                lost = True
                break
        w_rel = float(st.get("w_rel_norm") or 0.0)
        max_w_rel = max(max_w_rel, w_rel)
        u_h = np.asarray(pr["u_hand"], float)
        rel = float(np.degrees(np.arccos(np.clip(np.dot(_unit(u0_h), _unit(u_h)), -1, 1))))
        peak_rel = max(peak_rel, rel)
        th = float(st["e_theta_deg"])
        if i % 2 == 0 or i == n - 1:
            geo = measure_lever_geometry(env.sim)
            series.append({
                "i": i,
                "t": (i + 1) * DT,
                "grip": grip,
                "dtheta_deg": th,
                "rel_rot_deg": rel,
                "w_rel_norm": w_rel,
                "v_rel_norm": float(st.get("v_rel_norm") or 0),
                "delta_p_mm": float(1e3 * np.linalg.norm(np.asarray(pr["p_rel"]) - np.asarray(p0["p_rel"]))),
                "fn_tot": float(st.get("fn_tot") or 0),
                "support_side": classify_contact_mode(env.sim).get("support_side"),
                "tau_g": (geo.get("primary") or {}).get("tau_g_along_des"),
            })
        # Wrong direction: dθ increasing while relative rotation growing
        if rel >= 2.0 and (th - th0) > 2.0 and th > th0 + 0.5 * rel:
            wrong_dir = True
            break
        # Early success band
        if th <= 10.0 and rel >= 2.0:
            break

    err1 = measure_errors(env.sim)
    p1 = relative_pose(env.sim)
    rel_final = float(np.degrees(np.arccos(np.clip(np.dot(_unit(u0_h), _unit(p1["u_hand"])), -1, 1))))
    delta_p = float(1e3 * np.linalg.norm(np.asarray(p1["p_rel"]) - np.asarray(p0["p_rel"])))
    d_theta_improve = th0 - float(err1["dtheta_deg"])
    # Rigid co-rotation: world hand α changes but relative axis ~0
    alpha0 = float(gates(env.sim)["alpha_up_deg"])  # end state; use series if needed
    motion = classify_pivot_motion(
        delta_p_rel_mm=delta_p,
        d_axis_deg=rel_final,
        d_theta=d_theta_improve,
        bilateral_end=bool(err1.get("bilateral") or float(err1.get("fn_tot") or 0) > 0.8),
    )

    failure = None
    if lost:
        failure = "UNCONTROLLED_SLIP"
    elif wrong_dir:
        failure = "WRONG_DIRECTION_ROTATION"
    elif rel_final < 2.0 and max_w_rel < 0.08:
        failure = "NO_MEANINGFUL_RELATIVE_ROTATION" if delta_p <= DIAG_MAX_DELTA_P_MM else "RIGID_CO_ROTATION"
    elif motion == "rigid_hand_object_rotation" and rel_final < 3.0:
        failure = "RIGID_CO_ROTATION"
    elif delta_p > 12.0:
        failure = "UNCONTROLLED_TRANSLATION"

    ok = failure is None and rel_final >= 2.0 and (d_theta_improve >= 1.0 or float(err1["dtheta_deg"]) <= 12.0)
    return {
        "ok": bool(ok),
        "failure": failure,
        "motion_class": motion,
        "relative_rotation_deg": rel_final,
        "peak_relative_rotation_deg": peak_rel,
        "max_w_rel": max_w_rel,
        "d_theta_improve_deg": d_theta_improve,
        "delta_p_rel_mm": delta_p,
        "errors0": err0,
        "errors1": err1,
        "geo0": geo0,
        "geo1": measure_lever_geometry(env.sim),
        "series": series,
        "actions": actions,
        "capture": take_capture(env),
        "alpha_end": float(gates(env.sim)["alpha_up_deg"]),
        "diag_3mm_ok": delta_p <= DIAG_MAX_DELTA_P_MM,
    }


def phase_d_brake_regrip(
    env,
    blob: dict,
    *,
    w_rel_hint: float = 0.0,
    target_dtheta: float = 10.0,
) -> dict[str, Any]:
    """Brake using relative ω estimate; regrip; do NOT force fingertips DOWN."""
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    st0 = sample_state(env.sim)
    w_rel = float(st0.get("w_rel_norm") or w_rel_hint or 0.0)
    coast = braking_distance_deg(w_rel, decel_rad_s2=2.5, margin_deg=3.0)
    actions: list = []
    # If still far and spinning, wait coast then clamp grip
    th = float(err0["dtheta_deg"])
    if th - coast > target_dtheta and w_rel > 0.05:
        wait_s = min(0.8, coast / max(np.degrees(w_rel), 1.0))
        for _ in range(int(wait_s / DT)):
            a = _wrist(0.0, 0.0, 0.2)
            step_legal(env.sim, a)
            actions.append(a.copy())

    # Hard brake + regrip
    for _ in range(int(0.45 / DT)):
        a = _wrist(0.0, 0.0, 1.0)
        step_legal(env.sim, a)
        actions.append(a.copy())

    # Extra settle
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    from data_generation.oracles.precision.corrections.orientation import _height_hold_v_hand

    for _ in range(int(0.5 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref, kp=2.5), np.zeros(3), a_g)
        step_legal(env.sim, a)
        actions.append(np.asarray(a, np.float32).copy())

    err1 = measure_errors(env.sim)
    bil = bool(err1.get("bilateral"))
    dropped = latched(env.sim) or (not bil and float(err1.get("fn_tot") or 0) < 0.5)
    overshoot = float(err1["dtheta_deg"]) > float(err0["dtheta_deg"]) + 3.0
    failure = None
    if dropped:
        failure = "REGRIP_FAILED"
    elif overshoot and w_rel > 0.1:
        failure = "BRAKE_FAILED"

    return {
        "ok": failure is None and bil,
        "failure": failure,
        "coast_deg": coast,
        "w_rel_used": w_rel,
        "errors0": err0,
        "errors1": err1,
        "geo": measure_lever_geometry(env.sim),
        "gates": gates(env.sim),
        "s1_state": make_s1_state(env.sim, force_valid=False),
        "final_posture": {
            "alpha_up_deg": float(gates(env.sim)["alpha_up_deg"]),
            "fingertips_down": bool(gates(env.sim)["fingertips_down"]),
            "fingertips_up": bool(gates(env.sim)["fingertips_up"]),
            "note": "DOWN restoration intentionally omitted for isolated LAP feasibility",
        },
        "actions": actions,
        "capture": take_capture(env),
    }


def run_lap(
    env,
    blob: dict,
    *,
    lever_mm: float | None = None,
    lever_sign: float | None = None,
    skip_lever: bool = False,
    grip_end: float = -0.15,
    release_s: float = 2.5,
    rate_tilt: float = 0.08,
) -> dict[str, Any]:
    """Full LAP A→B→C→D from identical IC capture."""
    restore_capture(env, blob["capture"])
    err_ic = measure_errors(env.sim)
    geo_ic = measure_lever_geometry(env.sim)
    all_actions: list = []
    phases: dict[str, Any] = {}
    cur = {"capture": blob["capture"], "ref": blob["ref"]}

    # Phase A — if a usable axial lever already exists, keep it (avoid destructive chase).
    lever_now = float((geo_ic.get("primary") or {}).get("axial_lever_mm") or 0.0)
    auto_skip = bool(abs(lever_now) >= 8.0 and lever_mm is None and lever_sign is None)
    if skip_lever or auto_skip:
        phases["A"] = {
            "ok": True,
            "skipped": True,
            "auto_skip_existing_lever": auto_skip,
            "lever0_mm": lever_now,
            "actions": [],
        }
    else:
        # Prefer score-based candidate search unless an explicit mm/sign is requested.
        a = phase_a_lever_arm(
            env,
            cur,
            target_axial_mm=lever_mm,
            force_sign=lever_sign,
            v_axial=0.018,
            max_s=3.0,
        )
        phases["A"] = {k: v for k, v in a.items() if k not in ("actions",)}
        all_actions += a.get("actions") or []
        if not a.get("ok"):
            # Fallback: proceed with existing geometry if still bilaterally stable at IC.
            restore_capture(env, blob["capture"])
            if sample_state(env.sim).get("bilateral") and abs(lever_now) >= 4.0:
                phases["A"]["fallback_keep_existing_lever"] = True
                phases["A"]["ok"] = True
                phases["A"]["failure"] = None
            else:
                return _pack_fail(
                    env, blob, err_ic, geo_ic, phases, all_actions,
                    a.get("failure") or "LEVER_ARM_UNREACHABLE", "A",
                )
        else:
            cur = {"capture": a["capture"], "ref": blob["ref"]}

    # Phase B
    b = phase_b_protected_rotation(env, cur, rate_scale=rate_tilt)
    phases["B"] = {k: v for k, v in b.items() if k not in ("actions",)}
    all_actions += b.get("actions") or []
    if not b.get("ok"):
        return _pack_fail(env, blob, err_ic, geo_ic, phases, all_actions, b.get("failure") or "SUPPORT_LOST_DURING_TILT", "B")
    cur = {"capture": b["capture"], "ref": blob["ref"]}

    # Phase C
    c = phase_c_controlled_release(env, cur, grip_end=grip_end, release_s=release_s)
    phases["C"] = {k: v for k, v in c.items() if k not in ("actions",)}
    all_actions += c.get("actions") or []
    if c.get("failure") in ("WRONG_DIRECTION_ROTATION", "UNCONTROLLED_SLIP", "UNCONTROLLED_TRANSLATION"):
        # Still attempt brake for safety logging
        d = phase_d_brake_regrip(env, {"capture": c.get("capture") or cur["capture"], "ref": blob["ref"]}, w_rel_hint=c.get("max_w_rel") or 0.0)
        phases["D"] = {k: v for k, v in d.items() if k not in ("actions",)}
        all_actions += d.get("actions") or []
        return _pack_fail(env, blob, err_ic, geo_ic, phases, all_actions, c["failure"], "C", partial_ok=False)
    cur = {"capture": c["capture"], "ref": blob["ref"]}

    # Phase D
    d = phase_d_brake_regrip(env, cur, w_rel_hint=c.get("max_w_rel") or 0.0)
    phases["D"] = {k: v for k, v in d.items() if k not in ("actions",)}
    all_actions += d.get("actions") or []
    if not d.get("ok"):
        return _pack_fail(env, blob, err_ic, geo_ic, phases, all_actions, d.get("failure") or "BRAKE_FAILED", "D")

    err_f = measure_errors(env.sim)
    restore_capture(env, blob["capture"])
    p0 = relative_pose(env.sim)
    restore_capture(env, d["capture"])
    p1 = relative_pose(env.sim)
    rel = float(np.degrees(np.arccos(np.clip(
        np.dot(_unit(p0["u_hand"]), _unit(p1["u_hand"])), -1, 1
    ))))
    # Prefer Phase C measured relative rotation (from release window)
    rel_use = float(c.get("relative_rotation_deg") or rel)
    delta_p = float(c.get("delta_p_rel_mm") or 0.0)
    d_imp = float(err_ic["dtheta_deg"]) - float(err_f["dtheta_deg"])
    motion = classify_pivot_motion(
        delta_p_rel_mm=delta_p,
        d_axis_deg=rel_use,
        d_theta=d_imp,
        bilateral_end=bool(err_f.get("bilateral")),
    )
    failure = None
    if rel_use < 2.0:
        failure = c.get("failure") or "NO_MEANINGFUL_RELATIVE_ROTATION"
    elif motion == "rigid_hand_object_rotation" and d_imp < 2.0:
        failure = "RIGID_CO_ROTATION"

    ok = failure is None and bool(c.get("ok") or rel_use >= 2.0) and bool(d.get("ok"))
    return {
        "ok": bool(ok),
        "mechanism": "lever_arm_assisted_pivot_experimental",
        "failure": failure,
        "phase_failed": None if ok else (failure and "C"),
        "phases": phases,
        "relative_rotation_deg": rel_use,
        "d_theta_improve_deg": d_imp,
        "delta_p_rel_mm": delta_p,
        "motion_class": motion,
        "errors0": err_ic,
        "errors1": err_f,
        "geo0": geo_ic,
        "geo1": measure_lever_geometry(env.sim),
        "final_posture": (phases.get("D") or {}).get("final_posture"),
        "s1_reentry_candidate": bool((phases.get("D") or {}).get("s1_state", {}).get("valid"))
        or bool(classify_contact_mode(env.sim).get("support_side")),
        "diag_3mm_ok": delta_p <= DIAG_MAX_DELTA_P_MM,
        "actions": all_actions,
        "capture": d.get("capture") or take_capture(env),
        "verified_lateral_fallback": False,
    }


def _pack_fail(env, blob, err_ic, geo_ic, phases, actions, failure, phase, partial_ok=False):
    err_f = measure_errors(env.sim)
    return {
        "ok": False,
        "mechanism": "lever_arm_assisted_pivot_experimental",
        "failure": failure,
        "phase_failed": phase,
        "phases": phases,
        "errors0": err_ic,
        "errors1": err_f,
        "geo0": geo_ic,
        "geo1": measure_lever_geometry(env.sim),
        "relative_rotation_deg": (phases.get("C") or {}).get("relative_rotation_deg"),
        "d_theta_improve_deg": float(err_ic["dtheta_deg"]) - float(err_f["dtheta_deg"]),
        "delta_p_rel_mm": (phases.get("C") or {}).get("delta_p_rel_mm"),
        "final_posture": (phases.get("D") or {}).get("final_posture") or {
            "alpha_up_deg": float(gates(env.sim)["alpha_up_deg"]),
            "fingertips_down": bool(gates(env.sim)["fingertips_down"]),
            "note": "failed before/during LAP; DOWN not restored",
        },
        "actions": actions,
        "capture": take_capture(env),
        "verified_lateral_fallback": False,
    }


def run_lever_only(env, blob: dict, *, lever_mm: float = 8.0, grip: float = 0.95, hold_s: float = 2.0) -> dict[str, Any]:
    """Condition B: increase lever arm but keep original grip (no protected release)."""
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    a = phase_a_lever_arm(env, blob, target_axial_mm=lever_mm)
    actions = list(a.get("actions") or [])
    if not a.get("ok"):
        return {
            "ok": False,
            "mechanism": "lever_only",
            "failure": a.get("failure") or "LEVER_ARM_UNREACHABLE",
            "phase_a": {k: v for k, v in a.items() if k not in ("actions",)},
            "errors0": err0,
            "errors1": measure_errors(env.sim),
            "actions": actions,
            "capture": a.get("capture") or blob["capture"],
        }
    cur = {"capture": a["capture"], "ref": blob["ref"]}
    # Hold firm grip; optional small wrist hold — no release
    restore_capture(env, cur["capture"])
    p0 = relative_pose(env.sim)
    for _ in range(int(hold_s / DT)):
        act = _wrist(0.0, 0.0, grip)
        step_legal(env.sim, act)
        actions.append(act.copy())
    err1 = measure_errors(env.sim)
    p1 = relative_pose(env.sim)
    rel = float(np.degrees(np.arccos(np.clip(np.dot(_unit(p0["u_hand"]), _unit(p1["u_hand"])), -1, 1))))
    return {
        "ok": rel >= 2.0 and (float(err0["dtheta_deg"]) - float(err1["dtheta_deg"])) >= 1.0,
        "mechanism": "lever_only",
        "failure": None if rel >= 2.0 else "NO_MEANINGFUL_RELATIVE_ROTATION",
        "phase_a": {k: v for k, v in a.items() if k not in ("actions",)},
        "relative_rotation_deg": rel,
        "d_theta_improve_deg": float(err0["dtheta_deg"]) - float(err1["dtheta_deg"]),
        "delta_p_rel_mm": float(1e3 * np.linalg.norm(np.asarray(p1["p_rel"]) - np.asarray(p0["p_rel"]))),
        "errors0": err0,
        "errors1": err1,
        "actions": actions,
        "capture": take_capture(env),
    }
