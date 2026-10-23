"""Contact-aware accel–cruise–decel rate profiles for S1-protected tilts."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.contact_modes import classify_contact_mode
from data_generation.oracles.precision.force_slip_control import DT


def braking_distance_deg(
    omega_closing_rad_s: float,
    *,
    decel_rad_s2: float = 2.0,
    margin_deg: float = 2.5,
) -> float:
    """Coast angle under constant deceleration, plus safety margin (degrees).

    Conservative (larger) estimates brake earlier so residual inertia can finish
    the approach without crossing the target.
    """
    w = abs(float(omega_closing_rad_s))
    a = float(max(decel_rad_s2, 0.4))
    coast_rad = (w * w) / (2.0 * a)
    return float(np.degrees(coast_rad) + max(0.0, margin_deg))


def scheduled_rate(
    *,
    t_s: float,
    ang_err_deg: float,
    ang0_deg: float,
    cruise: float,
    peak: float,
    floor: float,
    accel_s: float,
    brake_deg: float | None = None,
    braking_latched: bool = False,
) -> float:
    """Trapezoidal rate vs time and remaining angle (fraction of W_MAX).

    Once ``braking_latched`` is set (velocity-based early brake), rate stays at
    zero — never reintroduce a same-direction minimum / floor rate.
    """
    if braking_latched:
        return 0.0
    cruise = float(np.clip(cruise, floor, peak))
    peak = float(max(peak, cruise))
    floor = float(max(0.03, floor))
    accel_s = float(max(0.1, accel_s))
    # Soft zero band (short of vertical); hard latch usually fires earlier via coast estimate.
    brake = float(brake_deg if brake_deg is not None else 8.0)
    brake = float(max(brake, 2.0))
    # Accel ramp
    if t_s < accel_s:
        r = floor + (cruise - floor) * (t_s / accel_s)
    else:
        r = cruise
    # Decel when approaching target (use remaining angle)
    decel_span = max(25.0, 0.35 * max(ang0_deg, 1.0))
    if ang_err_deg < decel_span:
        blend = float(np.clip(ang_err_deg / decel_span, 0.0, 1.0))
        r = floor + (r - floor) * blend
    # Approach band: soft caps without re-inflating to cruise floor.
    if ang_err_deg < 35.0:
        r = min(r, 0.05)
    if ang_err_deg < 28.0:
        r = min(r, 0.038)
    if ang_err_deg < 22.0:
        r = min(r, 0.028)
    if ang_err_deg < 16.0:
        r = min(r, 0.018)
    if ang_err_deg <= brake:
        return 0.0
    if ang_err_deg < 35.0:
        return float(np.clip(r, 0.0, min(peak, 0.05)))
    return float(np.clip(r, floor, peak))


def contact_rate_guard(
    sim,
    rate: float,
    *,
    support_side: str | None,
    floor: float,
    vrel_soft: float = 0.012,
    vrel_hard: float = 0.030,
    fnz_soft: float = 1.5,
) -> tuple[float, dict[str, Any]]:
    """Reduce commanded rate when contact margins deteriorate.

    Never re-inflate above the incoming command — soft approach / brake caps
    must stick even under vrel/fnz guards (guards only slow further).
    """
    from data_generation.oracles.precision.force_slip_control import sample_state

    st = sample_state(sim)
    m = classify_contact_mode(sim)
    vrel = float(st.get("v_rel_norm") or 0.0)
    if support_side == "L":
        fnz = float(m.get("Fn_z_L") or 0.0)
    elif support_side == "R":
        fnz = float(m.get("Fn_z_R") or 0.0)
    else:
        fnz = max(float(m.get("Fn_z_L") or 0.0), float(m.get("Fn_z_R") or 0.0))
    r_in = float(rate)
    r = r_in
    # Do not re-inflate an intentional brake (rate≈0) up to contact floor.
    if r <= 1e-9:
        return 0.0, {
            "v_rel_norm": vrel,
            "fnz_support": fnz,
            "rate_after_guard": 0.0,
            "guard": "brake_hold",
            "bilateral": bool(st.get("bilateral")),
        }
    reason = None
    if vrel > vrel_hard:
        r = min(r, float(floor))
        reason = "vrel_hard"
    elif vrel > vrel_soft:
        scale = 1.0 - 0.7 * (vrel - vrel_soft) / max(vrel_hard - vrel_soft, 1e-6)
        r = min(r, r * float(np.clip(scale, 0.25, 1.0)))
        # Mid-cruise: keep a small floor; approach soft-caps may already be below it.
        if r_in >= float(floor):
            r = max(float(floor), r)
        reason = "vrel_soft"
    if fnz < fnz_soft and fnz > -1.0:
        r = min(r, r * 0.55)
        if r_in >= float(floor):
            r = max(float(floor), r)
        reason = (reason or "") + "+fnz_soft"
    r = min(r, r_in)  # hard: never above commanded
    return float(r), {
        "v_rel_norm": vrel,
        "fnz_support": fnz,
        "rate_after_guard": r,
        "guard": reason,
        "bilateral": bool(st.get("bilateral")),
    }
