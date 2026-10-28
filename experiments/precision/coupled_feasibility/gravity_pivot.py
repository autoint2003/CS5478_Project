"""Experimental Gravity-Aligned Pivot (not production).

Corrects world-frame cylinder orientation via contact-supported relative rotation
assisted by gravity. Rejects rigid hand–object co-rotation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.corrections.contact_primitives import (
    _action_vw,
    _wrist,
    classify_pivot_motion,
    relative_pose,
)
from data_generation.oracles.precision.mechanics.dual_loop import DT, sample_state
from data_generation.oracles.precision.mechanics.open_axis import gates
from data_generation.oracles.precision.residuals import measure_errors
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import cylinder_axis_world, hand_R
from evaluation.success_criteria import U_TGT

G_WORLD = np.array([0.0, 0.0, -9.81], float)


def _signed_tilt(sim) -> dict[str, Any]:
    u = np.asarray(cylinder_axis_world(sim), float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    tgt = np.asarray(U_TGT, float)
    tgt = tgt / max(float(np.linalg.norm(tgt)), 1e-12)
    # Smallest rotation taking u → tgt: axis = u × tgt
    axis_w = np.cross(u, tgt)
    n = float(np.linalg.norm(axis_w))
    theta = float(np.degrees(np.arctan2(n, float(np.clip(np.dot(u, tgt), -1.0, 1.0)))))
    if n < 1e-9:
        axis_w = np.array([1.0, 0.0, 0.0])
    else:
        axis_w = axis_w / n
    # Gravity torque sense about axis through approximate pad mid (hand origin proxy):
    # τ ∝ r × F_g with r ≈ COM offset from hand; use u × g as tipping tendency of axis.
    tip = np.cross(u, G_WORLD)
    grav_help = float(np.dot(tip, axis_w))  # >0 ⇒ gravity assists rotation toward tgt
    Rh = hand_R(sim)
    axis_h = Rh.T @ axis_w
    return {
        "u_world": u.tolist(),
        "theta_to_tgt_deg": theta,
        "axis_world": axis_w.tolist(),
        "axis_hand": axis_h.tolist(),
        "gravity_helps": bool(grav_help > 0.0),
        "gravity_help_score": grav_help,
        "dtheta_deg": float(sample_state(sim)["e_theta_deg"]),
    }


def gravity_aligned_pivot(env, blob: dict, *, max_s: float = 4.0, grip_slip: float = 0.45) -> dict[str, Any]:
    """Experimental primitive. Does not call production correct_dtheta."""
    restore_capture(env, blob["capture"])
    if not sample_state(env.sim).get("bilateral"):
        return {"ok": False, "reason": "no_bilateral_contact", "actions": [], "capture": blob["capture"]}

    tilt0 = _signed_tilt(env.sim)
    err0 = measure_errors(env.sim)
    p0 = relative_pose(env.sim)
    actions: list = []

    if float(tilt0["dtheta_deg"]) <= 10.0 and float(tilt0["theta_to_tgt_deg"]) <= 10.0:
        return {
            "ok": True,
            "skipped": True,
            "reason": "already_within_theta_tol",
            "tilt0": tilt0,
            "errors0": err0,
            "actions": [],
            "capture": blob["capture"],
        }

    axis_h = np.asarray(tilt0["axis_hand"], float)
    # Wrist body rate about the relative-error axis in hand frame.
    # Sign: rotate gripper so contact friction lets gravity/object lag → relative pivot.
    # If gravity helps, soften and rotate slowly along +axis; else flip to load contact first.
    sign = 1.0 if tilt0["gravity_helps"] else -1.0
    w_body = sign * 0.35 * axis_h
    w_body = w_body / max(float(np.linalg.norm(w_body)), 1e-9) * 0.35

    n_steps = max(1, int(round(max_s / DT)))
    theta_best = float(err0["dtheta_deg"])
    lost = False
    rigid_streak = 0
    series = []

    for i in range(n_steps):
        a = _action_vw([0.0, 0.0, 0.0], w_body.tolist(), grip_slip)
        step_legal(env.sim, a)
        actions.append(a.copy())
        st = sample_state(env.sim)
        if not st.get("bilateral"):
            lost = True
            break
        th = float(st["e_theta_deg"])
        series.append({"i": i, "dtheta_deg": th, "fn_tot": float(st["fn_tot"])})
        if th < theta_best - 0.15:
            theta_best = th
            rigid_streak = 0
        else:
            rigid_streak += 1
        # Early stop if within working band
        if th <= 10.0:
            break
        # Abort if long stretch without theta improvement (likely rigid co-rotation)
        if rigid_streak >= int(1.2 / DT) and th > float(err0["dtheta_deg"]) - 1.0:
            break

    # Brake + regrip
    for _ in range(int(0.35 / DT)):
        a = _wrist(0.0, 0.0, 1.0)
        step_legal(env.sim, a)
        actions.append(a.copy())

    err1 = measure_errors(env.sim)
    p1 = relative_pose(env.sim)
    tilt1 = _signed_tilt(env.sim)
    delta_p = float(np.linalg.norm(np.asarray(p1["p_rel"]) - np.asarray(p0["p_rel"]))) * 1e3
    d_axis = abs(float(p1["e_theta_deg"]) - float(p0["e_theta_deg"]))
    d_theta_improve = float(err0["dtheta_deg"]) - float(err1["dtheta_deg"])
    motion = classify_pivot_motion(
        delta_p_rel_mm=delta_p,
        d_axis_deg=abs(d_theta_improve),
        d_theta=d_theta_improve,
        bilateral_end=bool(err1.get("bilateral")),
    )

    # Must improve world/hand theta and not be pure rigid co-rotation
    improved = d_theta_improve >= 2.0 or float(err1["dtheta_deg"]) <= 10.0
    reject_rigid = motion == "rigid_hand_object_rotation" and d_theta_improve < 2.0
    ok = bool(
        err1.get("bilateral")
        and not lost
        and improved
        and not reject_rigid
        and motion in ("position_preserving_relative_pivot", "coupled_rotation_translation")
    )
    reason = None
    if lost:
        reason = "contact_loss"
    elif reject_rigid:
        reason = "rigid_hand_object_rotation_no_relative_correction"
    elif not improved:
        reason = "pivot_not_improved"
    elif not err1.get("bilateral"):
        reason = "bilateral_lost_after_settle"
    elif not ok:
        reason = f"motion_class_{motion}"

    return {
        "ok": bool(ok),
        "mechanism": "gravity_aligned_pivot_experimental",
        "reason": reason,
        "motion_class": motion,
        "tilt0": tilt0,
        "tilt1": tilt1,
        "errors0": err0,
        "errors1": err1,
        "d_theta_improve_deg": d_theta_improve,
        "delta_p_rel_mm": delta_p,
        "gravity_helps": tilt0["gravity_helps"],
        "wrist_sign": sign,
        "series_tail": series[-8:],
        "actions": actions,
        "capture": take_capture(env),
        "fingertips_down": bool(gates(env.sim)["fingertips_down"]),
    }
