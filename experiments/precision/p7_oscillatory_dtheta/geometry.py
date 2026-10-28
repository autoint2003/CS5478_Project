"""Signed orientation diagnostics for P7 (experimental)."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.residuals import measure_errors
from evaluation.grasp_geometry import hand_R, measure, object_R
from evaluation.success_criteria import P_TGT, U_TGT
from experiments.precision.oscillation_assisted_reorientation.metrics import (
    desired_correction_axis_hand,
    hand_omega_world,
    object_omega_world,
    omega_rel_hand,
    signed_angle_about_axis,
    snapshot_state,
)


def _unit(v) -> np.ndarray:
    a = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.zeros(3)


def signed_theta_to_target(sim) -> dict[str, Any]:
    """Signed θ (deg) taking current cylinder axis → target about correction axis."""
    m = measure(sim, U_TGT, P_TGT)
    u = _unit(m["u_c_h"])
    tgt = _unit(m["u_target_h"])
    ax = desired_correction_axis_hand(sim)
    # Positive ⇒ need +rotation of cylinder about ax (right-hand) to reach target.
    signed = signed_angle_about_axis(tgt, u, ax)
    err = measure_errors(sim)
    return {
        "abs_theta_deg": float(err["dtheta_deg"]),
        "signed_theta_deg": float(signed),
        "axis_h": ax.tolist(),
        "u_c_h": u.tolist(),
        "u_target_h": tgt.tolist(),
        "e_signed_mm": err.get("e_signed_mm"),
        "e_pos_mm": float(err.get("e_pos_mm") or 0),
        "bilateral": bool(err.get("bilateral")),
        "dx_mm": float(np.asarray(err.get("e_signed_mm") or [0, 0, 0])[0]),
        "dy_mm": float(np.asarray(err.get("e_signed_mm") or [0, 0, 0])[1]),
        "dz_mm": float(np.asarray(err.get("e_signed_mm") or [0, 0, 0])[2]),
    }


def body_axis_for_correction(sim) -> np.ndarray:
    """Map hand-frame correction axis onto ±body {x,y,z} for wrist ω commands."""
    ax_h = desired_correction_axis_hand(sim)
    cands = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    name, base = max(cands.items(), key=lambda kv: abs(float(np.dot(ax_h, kv[1]))))
    sgn = float(np.sign(np.dot(ax_h, base)) or 1.0)
    return base * sgn, name


def full_orientation_snapshot(sim, *, u0_h=None, axis_h=None, R0=None) -> dict[str, Any]:
    th = signed_theta_to_target(sim)
    ax = np.asarray(axis_h if axis_h is not None else th["axis_h"], float)
    u0 = np.asarray(u0_h if u0_h is not None else th["u_c_h"], float)
    Rref = R0 if R0 is not None else hand_R(sim).copy()
    snap = snapshot_state(sim, u0_h=u0, axis_h=ax, R0=Rref)
    Rh = hand_R(sim)
    Ro = object_R(sim)
    return {
        **th,
        **snap,
        "R_hand": Rh.tolist(),
        "R_obj": Ro.tolist(),
        "w_hand": hand_omega_world(sim).tolist(),
        "w_obj": object_omega_world(sim).tolist(),
        "w_rel_h": omega_rel_hand(sim).tolist(),
    }
