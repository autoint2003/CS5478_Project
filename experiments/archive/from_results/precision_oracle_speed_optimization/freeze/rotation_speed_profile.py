"""Contact-aware accel–cruise–decel rate profiles for S1-protected tilts."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.contact_modes import classify_contact_mode
from data_generation.oracles.precision.force_slip_control import DT


def scheduled_rate(
    *,
    t_s: float,
    ang_err_deg: float,
    ang0_deg: float,
    cruise: float,
    peak: float,
    floor: float,
    accel_s: float,
) -> float:
    """Trapezoidal rate vs time and remaining angle (fraction of W_MAX)."""
    cruise = float(np.clip(cruise, floor, peak))
    peak = float(max(peak, cruise))
    floor = float(max(0.03, floor))
    accel_s = float(max(0.1, accel_s))
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
    # Near-target fine approach
    if ang_err_deg < 15.0:
        r = min(r, max(floor, 0.08))
    if ang_err_deg < 8.0:
        r = min(r, max(floor, 0.06))
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
    """Reduce commanded rate when contact margins deteriorate."""
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
    r = float(rate)
    reason = None
    if vrel > vrel_hard:
        r = floor
        reason = "vrel_hard"
    elif vrel > vrel_soft:
        scale = 1.0 - 0.7 * (vrel - vrel_soft) / max(vrel_hard - vrel_soft, 1e-6)
        r = max(floor, r * float(np.clip(scale, 0.25, 1.0)))
        reason = "vrel_soft"
    if fnz < fnz_soft and fnz > -1.0:
        r = max(floor, r * 0.55)
        reason = (reason or "") + "+fnz_soft"
    return float(r), {
        "v_rel_norm": vrel,
        "fnz_support": fnz,
        "rate_after_guard": r,
        "guard": reason,
        "bilateral": bool(st.get("bilateral")),
    }
