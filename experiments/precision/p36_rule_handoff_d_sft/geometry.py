"""Frozen original Precision geometric HANDOFF targets (not tuned on eval)."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import PrecisionConfig
from evaluation.axis_aware_success import (
    AXIS_V1_EX_MM,
    AXIS_V1_EZ_MM,
    AXIS_V1_THETA_DEG,
)

# Documented frozen thresholds (PrecisionConfig / AXIS_V1 defaults).
THETA_TOL_DEG = float(AXIS_V1_THETA_DEG)  # 10.0
DX_TOL_MM = float(AXIS_V1_EX_MM)  # 12.0
DZ_TOL_MM = float(AXIS_V1_EZ_MM)  # 20.0


def geometric_errors(sim) -> dict[str, float]:
    err = measure_errors(sim)
    e = np.asarray(err["e_signed_mm"], float)
    return {
        "dtheta_deg": float(err["dtheta_deg"]),
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "e_pos_mm": float(err["e_pos_mm"]),
        "bilateral": float(bool(err.get("bilateral"))),
    }


def geometric_target_reached(sim, *, cfg: PrecisionConfig | None = None) -> tuple[bool, dict[str, Any]]:
    """True iff |dtheta|≤θ_tol AND |dx|≤ex_tol AND |dz|≤ez_tol (original Precision targets).

    Does NOT imply STRICT success — caller must run 12s HANDOFF evaluation.
    """
    pcfg = cfg or PrecisionConfig()
    th = float(pcfg.theta_tol_deg)
    ex = float(pcfg.ex_tol_mm)
    ez = float(pcfg.ez_tol_mm)
    g = geometric_errors(sim)
    ok = bool(
        abs(g["dtheta_deg"]) <= th
        and abs(g["dx_mm"]) <= ex
        and abs(g["dz_mm"]) <= ez
    )
    return ok, {
        **g,
        "theta_tol_deg": th,
        "dx_tol_mm": ex,
        "dz_tol_mm": ez,
        "theta_ok": abs(g["dtheta_deg"]) <= th,
        "dx_ok": abs(g["dx_mm"]) <= ex,
        "dz_ok": abs(g["dz_mm"]) <= ez,
        "geometric_ready": ok,
    }


def initially_target_satisfied(sim) -> bool:
    ok, _ = geometric_target_reached(sim)
    return ok
