"""Axis-aware precision decision logic (offline Oracle only).

Privileged: uses signed hand-frame errors and contact forces.
Does not introduce a runtime FSM or plan selector.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.axis_aware_success import (
    AXIS_V1_EX_MM,
    AXIS_V1_EY_MM,
    AXIS_V1_EZ_MM,
    AXIS_V1_THETA_DEG,
)


def axial_desire_direction(e_signed_mm: np.ndarray | list[float]) -> np.ndarray:
    """d_z = -sign(e_z) * hand_z  — desired relative COM motion along hand-z."""
    e = np.asarray(e_signed_mm, float).reshape(3)
    ez = float(e[2])
    if abs(ez) < 1e-9:
        return np.zeros(3)
    return np.array([0.0, 0.0, -float(np.sign(ez))], float)


def lateral_desire_direction(e_signed_mm: np.ndarray | list[float]) -> np.ndarray:
    """Desire in hand xy-plane (drop axial component)."""
    e = np.asarray(e_signed_mm, float).reshape(3) * 1e-3
    e_xy = np.array([e[0], e[1], 0.0])
    n = float(np.linalg.norm(e_xy))
    if n < 1e-9:
        return np.zeros(3)
    return -e_xy / n


def decide_precision(
    e_signed_mm: np.ndarray | list[float],
    *,
    theta_deg: float,
    bilateral: bool,
    ex_tol_mm: float = AXIS_V1_EX_MM,
    ey_tol_mm: float = AXIS_V1_EY_MM,
    ez_tol_mm: float = AXIS_V1_EZ_MM,
    theta_tol_deg: float = AXIS_V1_THETA_DEG,
) -> dict[str, Any]:
    """Privileged offline decision at a stable grasp.

    Returns mode in {handoff_ready, lateral_correction, axial_correction,
    lateral_then_axial, abnormal}.
    """
    e = np.asarray(e_signed_mm, float).reshape(3)
    ex, ey, ez = float(e[0]), float(e[1]), float(e[2])
    abs_x, abs_y, abs_z = abs(ex), abs(ey), abs(ez)

    if not bilateral:
        return {
            "mode": "abnormal",
            "reason": "no_bilateral_contact",
            "lateral_needed": False,
            "axial_needed": False,
            "e_signed_mm": e.tolist(),
        }
    if abs_y > ey_tol_mm:
        return {
            "mode": "abnormal",
            "reason": "significant_ey",
            "lateral_needed": abs_x > ex_tol_mm,
            "axial_needed": abs_z > ez_tol_mm,
            "e_signed_mm": e.tolist(),
            "ey_mm": ey,
        }
    if float(theta_deg) > theta_tol_deg * 1.5:
        return {
            "mode": "abnormal",
            "reason": "theta_unsuitable_for_controlled_slip",
            "lateral_needed": abs_x > ex_tol_mm,
            "axial_needed": abs_z > ez_tol_mm,
            "e_signed_mm": e.tolist(),
            "theta_deg": float(theta_deg),
        }

    lat = abs_x > ex_tol_mm
    ax = abs_z > ez_tol_mm
    if not lat and not ax and float(theta_deg) <= theta_tol_deg:
        return {
            "mode": "handoff_ready",
            "reason": "within_axis_v1_limits",
            "lateral_needed": False,
            "axial_needed": False,
            "d_lateral": [0.0, 0.0, 0.0],
            "d_axial": [0.0, 0.0, 0.0],
            "e_signed_mm": e.tolist(),
            "limits_mm": {"ex": ex_tol_mm, "ey": ey_tol_mm, "ez": ez_tol_mm},
        }
    if lat and ax:
        mode = "lateral_then_axial"
    elif lat:
        mode = "lateral_correction"
    elif ax:
        mode = "axial_correction"
    else:
        # theta only / edge — still handoff-ready under axis limits if pose ok
        mode = "handoff_ready" if float(theta_deg) <= theta_tol_deg else "abnormal"

    d_lat = lateral_desire_direction(e)
    d_ax = axial_desire_direction(e)
    return {
        "mode": mode,
        "reason": f"lat={lat}_ax={ax}",
        "lateral_needed": lat,
        "axial_needed": ax,
        "d_lateral": d_lat.tolist(),
        "d_axial": d_ax.tolist(),
        "e_signed_mm": e.tolist(),
        "limits_mm": {"ex": ex_tol_mm, "ey": ey_tol_mm, "ez": ez_tol_mm},
        "note": "Axial correction only when |e_z| exceeds validated/provisional axial tolerance.",
    }
