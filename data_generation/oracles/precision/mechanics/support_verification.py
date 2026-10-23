"""Force-backed contact mode classification for the unified Precision flowchart."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.mechanics.open_axis import gates
from data_generation.oracles.precision.mechanics.finger_surface import support_force_decomposition
from data_generation.oracles.precision.mechanics.dual_loop import contact_forces, sample_state

MODE_STABLE_BILATERAL = "STABLE_BILATERAL_GRASP"
MODE_ONE_SIDED_SUPPORT = "ONE_SIDED_MECHANICAL_SUPPORT"
MODE_BILATERAL_FRICTION = "BILATERAL_FRICTION_SUPPORT"
MODE_CONTROLLED_PIVOT = "CONTROLLED_PIVOT"
MODE_CONTROLLED_DX = "CONTROLLED_DX_SLIDING"
MODE_CONTROLLED_DZ = "CONTROLLED_DZ_SLIDING"
MODE_BRAKED = "BRAKED_GRASP"
MODE_UNSAFE = "UNSAFE_OR_UNKNOWN"


def classify_contact_mode(sim, *, sliding: str | None = None) -> dict[str, Any]:
    """Classify load-bearing mode from measured contacts/forces (not orientation alone)."""
    s = sample_state(sim)
    cf = contact_forces(sim)
    sup = support_force_decomposition(sim)
    g = gates(sim)
    sides = sup.get("sides") or {}
    fn_L = float(cf.get("fn_L") or 0.0)
    fn_R = float(cf.get("fn_R") or 0.0)
    bil = bool(s.get("bilateral") and fn_L >= 1.0 and fn_R >= 1.0)
    # Mechanical support: one side carries dominant world-Z normal load
    fnz_L = float((sides.get("L") or {}).get("Fn_z_normal") or 0.0)
    fnz_R = float((sides.get("R") or {}).get("Fn_z_normal") or 0.0)
    ftz_L = float((sides.get("L") or {}).get("Ft_z_friction") or 0.0)
    ftz_R = float((sides.get("R") or {}).get("Ft_z_friction") or 0.0)
    one_sided = False
    support_side = None
    if bil:
        if fnz_L > 2.0 and fnz_L > 1.5 * max(fnz_R, 0.5):
            one_sided = True
            support_side = "L"
        elif fnz_R > 2.0 and fnz_R > 1.5 * max(fnz_L, 0.5):
            one_sided = True
            support_side = "R"
    friction_hold = bool(
        bil and (abs(ftz_L) + abs(ftz_R)) > 1.5 and abs(fnz_L - fnz_R) < 1.5 * max(abs(fnz_L), abs(fnz_R), 1.0)
    )
    v_rel = float(s.get("v_rel_norm") or 0.0)
    w_rel = float(s.get("w_rel_norm") or 0.0)
    if sliding == "dx" and bil and v_rel > 0.01:
        mode = MODE_CONTROLLED_DX
    elif sliding == "dz" and bil and v_rel > 0.01:
        mode = MODE_CONTROLLED_DZ
    elif sliding == "pivot" and bil and w_rel > 0.05:
        mode = MODE_CONTROLLED_PIVOT
    elif one_sided:
        mode = MODE_ONE_SIDED_SUPPORT
    elif friction_hold and 60.0 <= float(g["alpha_up_deg"]) <= 120.0:
        mode = MODE_BILATERAL_FRICTION
    elif bil and v_rel < 0.05 and w_rel < 0.5:
        mode = MODE_STABLE_BILATERAL if fn_L + fn_R < 22.0 else MODE_BRAKED
    elif bil:
        mode = MODE_STABLE_BILATERAL
    else:
        mode = MODE_UNSAFE

    return {
        "mode": mode,
        "bilateral": bil,
        "fn_L": fn_L,
        "fn_R": fn_R,
        "Fn_z_L": fnz_L,
        "Fn_z_R": fnz_R,
        "Ft_z_L": ftz_L,
        "Ft_z_R": ftz_R,
        "support_side": support_side,
        "one_sided_mechanical": one_sided,
        "friction_balanced": friction_hold,
        "alpha_up_deg": float(g["alpha_up_deg"]),
        "fingertips_down": bool(g["fingertips_down"]),
        "fingertips_up": bool(g["fingertips_up"]),
        "v_rel_norm": v_rel,
        "w_rel_norm": w_rel,
    }


def make_s1_state(sim, *, roll_sign: float | None = None, force_valid: bool = False) -> dict[str, Any]:
    """Snapshot of verified one-sided support for scheduler handoff dx→dz.

    ``force_valid`` is accepted only when live ``one_sided_mechanical`` is true.
    Bilateral contact alone must not be treated as mechanically protected S1.
    """
    m = classify_contact_mode(sim)
    force_ok = bool(m["one_sided_mechanical"])
    # force_valid may only confirm an already one-sided state; never upgrade bilateral-only.
    valid = bool(force_ok)
    return {
        "valid": valid,
        "mode": m["mode"],
        "support_side": m["support_side"],
        "roll_sign": roll_sign,
        "alpha_up_deg": m["alpha_up_deg"],
        "fn_L": m["fn_L"],
        "fn_R": m["fn_R"],
        "Fn_z_L": m["Fn_z_L"],
        "Fn_z_R": m["Fn_z_R"],
        "bilateral": m["bilateral"],
        "force_ok": force_ok,
        "one_sided_mechanical": bool(m["one_sided_mechanical"]),
    }
