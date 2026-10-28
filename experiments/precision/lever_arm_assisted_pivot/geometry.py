"""Lever-arm / contact geometry helpers for LAP (experimental)."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.mechanics.dual_loop import sample_state
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode
from envs.contacts.frames import extract_contacts
from evaluation.grasp_geometry import cylinder_axis_world, hand_R
from evaluation.success_criteria import U_TGT

G_WORLD = np.array([0.0, 0.0, -9.81], float)
# Approximate cylinder mass used only for torque *proxy* diagnostics (not applied to sim).
M_PROXY = 0.25


def measure_lever_geometry(sim) -> dict[str, Any]:
    """Actual COM–contact lever metrics — not numerical dz_mm alone."""
    rows, _ = extract_contacts(sim)
    com = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    u = np.asarray(cylinder_axis_world(sim), float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    Rh = hand_R(sim)
    st = sample_state(sim)
    e = np.asarray(st["e_signed_mm"], float)
    mode = classify_contact_mode(sim)

    per_side: dict[str, Any] = {}
    for side in ("L", "R"):
        rs = [r for r in rows if r.get("side") == side]
        if not rs:
            per_side[side] = None
            continue
        pts = np.asarray([r["pos_w"] for r in rs], float)
        fs = np.asarray([r["f_obj_w"] for r in rs], float)
        p = pts.mean(axis=0)
        f = fs.sum(axis=0)
        r = com - p
        # Axial lever: projection of COM−contact along cylinder axis (meters)
        axial_m = float(np.dot(r, u))
        # Gravity torque about contact: τ = r × (m g)
        tau_g = np.cross(r, M_PROXY * G_WORLD)
        # Contact-force torque about COM: τ_c = (p−com) × f = −r × f
        tau_c = np.cross(-r, f)
        # Desired corrective axis: u × U_TGT
        tgt = np.asarray(U_TGT, float)
        tgt = tgt / max(float(np.linalg.norm(tgt)), 1e-12)
        axis_des = np.cross(u, tgt)
        n_ax = float(np.linalg.norm(axis_des))
        if n_ax > 1e-9:
            axis_des = axis_des / n_ax
        else:
            axis_des = np.array([1.0, 0.0, 0.0])
        per_side[side] = {
            "n_contacts": len(rs),
            "contact_centroid_w": p.tolist(),
            "f_obj_w": f.tolist(),
            "r_com_minus_contact_w": r.tolist(),
            "axial_lever_m": axial_m,
            "axial_lever_mm": axial_m * 1e3,
            "tau_gravity_w": tau_g.tolist(),
            "tau_contact_w": tau_c.tolist(),
            "tau_g_along_des": float(np.dot(tau_g, axis_des)),
            "tau_c_along_des": float(np.dot(tau_c, axis_des)),
            "tau_total_along_des": float(np.dot(tau_g + tau_c, axis_des)),
        }

    # Prefer designated / force-backed support side
    side = mode.get("support_side")
    if side not in ("L", "R") or per_side.get(side) is None:
        # choose larger |Fn_z|
        scores = {}
        for s in ("L", "R"):
            if per_side.get(s) is None:
                continue
            scores[s] = abs(float(mode.get(f"Fn_z_{s}") or 0.0)) + abs(float(mode.get(f"fn_{s}") or 0.0))
        side = max(scores, key=scores.get) if scores else None

    primary = per_side.get(side) if side else None
    return {
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "dtheta_deg": float(st["e_theta_deg"]),
        "e_pos_mm": float(st["e_pos_mm"]),
        "bilateral": bool(st["bilateral"]),
        "support_side": side,
        "contact_mode": mode.get("mode"),
        "Fn_z_L": mode.get("Fn_z_L"),
        "Fn_z_R": mode.get("Fn_z_R"),
        "fn_L": mode.get("fn_L"),
        "fn_R": mode.get("fn_R"),
        "u_world": u.tolist(),
        "primary": primary,
        "per_side": per_side,
        "dz_mm_vs_axial_lever_mm": {
            "dz_mm": float(e[2]),
            "axial_lever_mm": None if primary is None else primary["axial_lever_mm"],
            "note": "dz_mm is hand-frame residual; axial_lever_mm is COM−contact along cylinder axis",
        },
        "w_rel_norm": float(st.get("w_rel_norm") or 0.0),
        "v_rel_norm": float(st.get("v_rel_norm") or 0.0),
        "p_c_h": st.get("p_c_h"),
    }
