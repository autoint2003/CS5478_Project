"""Signed relative-rotation and per-step OAR diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.mechanics.dual_loop import sample_state
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode
from data_generation.oracles.precision.residuals import measure_errors
from envs.contacts.frames import extract_contacts
from envs.simulation.grasp_quality import body_twist
from evaluation.grasp_geometry import cylinder_axis_world, hand_R, measure
from evaluation.success_criteria import P_TGT, U_TGT


def _unit(v) -> np.ndarray:
    a = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.zeros(3)


def omega_rel_hand(sim) -> np.ndarray:
    """ω_rel_hand = R_hand.T @ (ω_object_world − ω_hand_world)."""
    Rh = hand_R(sim)
    _, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    _, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    return Rh.T @ (np.asarray(wo, float) - np.asarray(wh, float))


def hand_omega_world(sim) -> np.ndarray:
    _, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    return np.asarray(wh, float)


def object_omega_world(sim) -> np.ndarray:
    _, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    return np.asarray(wo, float)


def signed_angle_about_axis(u: np.ndarray, u_ref: np.ndarray, axis: np.ndarray) -> float:
    """Continuous signed angle (deg) taking u_ref → u about ``axis`` (right-hand)."""
    ax = _unit(axis)
    a = np.asarray(u, float) - float(np.dot(u, ax)) * ax
    b = np.asarray(u_ref, float) - float(np.dot(u_ref, ax)) * ax
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    a, b = a / na, b / nb
    s = float(np.dot(ax, np.cross(b, a)))
    c = float(np.clip(np.dot(b, a), -1.0, 1.0))
    return float(np.degrees(np.arctan2(s, c)))


def hand_tilt_about(sim, axis_body: np.ndarray, R0: np.ndarray) -> float:
    """Signed hand orientation change about body axis since R0 (deg)."""
    Rh = hand_R(sim)
    # Relative rotation R_rel = R0.T @ Rh; rotvec in world, then to R0 frame
    R_rel = np.asarray(R0, float).T @ Rh
    # Rodriguez angle about axis expressed in R0 frame
    ax_w = np.asarray(R0, float) @ _unit(axis_body)
    # Extract signed rotation of R0's x/y via open-axis proxy: use column 2 difference
    z0 = np.asarray(R0, float)[:, 2]
    z1 = Rh[:, 2]
    return signed_angle_about_axis(z1, z0, ax_w)


def relative_cyl_angle(sim, u0_h: np.ndarray, axis_h: np.ndarray) -> float:
    """Signed cylinder-axis angle in hand frame about axis_h vs reference u0_h (deg)."""
    Rh = hand_R(sim)
    u_h = Rh.T @ _unit(cylinder_axis_world(sim))
    return signed_angle_about_axis(u_h, _unit(u0_h), _unit(axis_h))


def desired_correction_axis_hand(sim) -> np.ndarray:
    """Hand-frame axis for rotating cylinder axis toward U_TGT."""
    m = measure(sim, U_TGT, P_TGT)
    u = _unit(m["u_c_h"])
    tgt = _unit(m["u_target_h"])
    ax = np.cross(u, tgt)
    n = float(np.linalg.norm(ax))
    if n < 1e-8:
        # fall back to hand y
        return np.array([0.0, 1.0, 0.0])
    return ax / n


def snapshot_state(sim, *, u0_h: np.ndarray, axis_h: np.ndarray, R0: np.ndarray | None = None) -> dict[str, Any]:
    err = measure_errors(sim)
    st = sample_state(sim)
    mode = classify_contact_mode(sim)
    Rh = hand_R(sim)
    w_rel = omega_rel_hand(sim)
    wh = hand_omega_world(sim)
    wo = object_omega_world(sim)
    e = np.asarray(err.get("e_signed_mm") or [0, 0, 0], float)
    rows, _ = extract_contacts(sim)
    slip = 0.0
    for r in rows:
        # tangential relative velocity magnitude if present
        vt = r.get("v_tang_norm") or r.get("slip") or 0.0
        try:
            slip = max(slip, float(vt))
        except Exception:
            pass
    out = {
        "dtheta_deg": float(err.get("dtheta_deg") or 0.0),
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "e_pos_mm": float(err.get("e_pos_mm") or 0.0),
        "bilateral": bool(st.get("bilateral")),
        "fn_L": float(st.get("fn_L") or 0.0),
        "fn_R": float(st.get("fn_R") or 0.0),
        "fn_tot": float(st.get("fn_tot") or 0.0),
        "Fn_z_L": mode.get("Fn_z_L"),
        "Fn_z_R": mode.get("Fn_z_R"),
        "support_side": mode.get("support_side"),
        "contact_mode": mode.get("mode"),
        "p_c_h": st.get("p_c_h"),
        "w_rel_h": w_rel.tolist(),
        "w_rel_along_axis": float(np.dot(w_rel, _unit(axis_h))),
        "w_hand_norm": float(np.linalg.norm(wh)),
        "w_obj_norm": float(np.linalg.norm(wo)),
        "signed_rel_deg": relative_cyl_angle(sim, u0_h, axis_h),
        "gripper_q": None,
        "n_contacts": len(rows),
        "slip_proxy": slip,
    }
    if R0 is not None:
        # body y oscillation default axis for hand tilt reporting
        out["hand_tilt_deg"] = hand_tilt_about(sim, np.array([0.0, 1.0, 0.0]), R0)
    return out


def ic_log(sim) -> dict[str, Any]:
    err = measure_errors(sim)
    st = sample_state(sim)
    mode = classify_contact_mode(sim)
    m = measure(sim, U_TGT, P_TGT)
    axis_h = desired_correction_axis_hand(sim)
    e = np.asarray(err.get("e_signed_mm") or [0, 0, 0], float)
    wh = hand_omega_world(sim)
    wo = object_omega_world(sim)
    return {
        "dtheta_deg": float(err.get("dtheta_deg") or 0.0),
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "e_pos_mm": float(err.get("e_pos_mm") or 0.0),
        "u_c_h": m["u_c_h"],
        "u_c_w": m["u_c_w"],
        "p_c_h": m["p_c_h"],
        "bilateral": bool(st.get("bilateral")),
        "fn_L": float(st.get("fn_L") or 0.0),
        "fn_R": float(st.get("fn_R") or 0.0),
        "fn_tot": float(st.get("fn_tot") or 0.0),
        "Fn_z_L": mode.get("Fn_z_L"),
        "Fn_z_R": mode.get("Fn_z_R"),
        "support_side": mode.get("support_side"),
        "contact_mode": mode.get("mode"),
        "correction_axis_h": axis_h.tolist(),
        "w_hand": wh.tolist(),
        "w_obj": wo.tolist(),
        "w_rel_h": omega_rel_hand(sim).tolist(),
        "grip_opening_proxy": float(st.get("fn_tot") or 0.0),
    }
