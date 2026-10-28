"""HANDOFF trigger rules for P37. Thresholds frozen; not tuned on eval.

AXIS_V1: original P36 three-axis box.
WORKING: single-step trigger matching SUCCESS_NEW instantaneous region
         (theta + Euclidean e_pos from evaluation.grasp_geometry.measure /
          evaluation.success_criteria WORKING_*). No 0.5s dwell here —
         STRICT evaluator owns sustained hold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.residuals import measure_errors
from evaluation.axis_aware_success import AXIS_V1_EX_MM, AXIS_V1_EZ_MM, AXIS_V1_THETA_DEG
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG

# --- AXIS_V1 (P36 baseline) ---
AXIS_THETA_DEG = float(AXIS_V1_THETA_DEG)  # 10
AXIS_DX_MM = float(AXIS_V1_EX_MM)  # 12
AXIS_DZ_MM = float(AXIS_V1_EZ_MM)  # 20

# --- WORKING (SUCCESS_NEW instantaneous region) ---
# Re-exported from canonical SUCCESS_NEW definition — do not approximate.
WORK_THETA_DEG = float(WORKING_THETA_DEG)  # 10
WORK_POS_MM = float(WORKING_POS_MM)  # 12


def axis_geometry(sim) -> dict[str, Any]:
    """Hand-frame signed axis errors via Precision measure_errors (same as P36)."""
    err = measure_errors(sim)
    e = np.asarray(err["e_signed_mm"], float)
    th = float(err["dtheta_deg"])
    dx, dy, dz = float(e[0]), float(e[1]), float(e[2])
    ok = bool(abs(th) <= AXIS_THETA_DEG and abs(dx) <= AXIS_DX_MM and abs(dz) <= AXIS_DZ_MM)
    return {
        "dtheta_deg": th,
        "dx_mm": dx,
        "dy_mm": dy,
        "dz_mm": dz,
        "e_pos_mm": float(err["e_pos_mm"]),
        "bilateral": bool(err.get("bilateral")),
        "axis_ok": ok,
        "theta_ok": abs(th) <= AXIS_THETA_DEG,
        "dx_ok": abs(dx) <= AXIS_DX_MM,
        "dz_ok": abs(dz) <= AXIS_DZ_MM,
        "theta_tol_deg": AXIS_THETA_DEG,
        "dx_tol_mm": AXIS_DX_MM,
        "dz_tol_mm": AXIS_DZ_MM,
    }


def working_geometry(sim) -> dict[str, Any]:
    """WORKING instantaneous region via the same measure() STRICT uses.

    Trigger: e_theta_deg <= WORKING_THETA_DEG AND e_pos_mm <= WORKING_POS_MM.
    Single timestep — no dwell.
    """
    m = measure(sim, U_TGT, P_TGT)
    th = float(m["e_theta_deg"])
    e_pos = float(m["e_pos_mm"])
    # Also record axis components for reporting (Precision signed frame)
    err = measure_errors(sim)
    e = np.asarray(err["e_signed_mm"], float)
    ok = bool(th <= WORK_THETA_DEG and e_pos <= WORK_POS_MM)
    return {
        "dtheta_deg": th,
        "e_pos_mm": e_pos,
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "bilateral": bool(err.get("bilateral")),
        "working_ok": ok,
        "theta_ok": th <= WORK_THETA_DEG,
        "pos_ok": e_pos <= WORK_POS_MM,
        "theta_tol_deg": WORK_THETA_DEG,
        "pos_tol_mm": WORK_POS_MM,
        "p_c_h": m.get("p_c_h"),
        "p_target_h": m.get("p_target_h"),
        # Cross-check: Precision e_pos should match measure e_pos
        "e_pos_precision_mm": float(err["e_pos_mm"]),
        "e_pos_match": abs(float(err["e_pos_mm"]) - e_pos) < 1e-3,
    }


def handoff_ready(sim, rule: str) -> tuple[bool, dict[str, Any]]:
    """rule ∈ {'axis', 'working'}."""
    ax = axis_geometry(sim)
    wk = working_geometry(sim)
    if rule == "axis":
        ok = bool(ax["axis_ok"])
    elif rule == "working":
        ok = bool(wk["working_ok"])
    else:
        raise ValueError(rule)
    return ok, {"axis": ax, "working": wk, "rule": rule, "ready": ok}


def init_labels(sim) -> dict[str, Any]:
    ax = axis_geometry(sim)
    wk = working_geometry(sim)
    axis_sat = bool(ax["axis_ok"])
    work_sat = bool(wk["working_ok"])
    return {
        "init_axis_satisfied": int(axis_sat),
        "init_working_satisfied": int(work_sat),
        "init_axis_only": int(axis_sat and not work_sat),
        "init_nontrivial_axis": int(not axis_sat),
        "init_dtheta_deg": ax["dtheta_deg"],
        "init_dx_mm": ax["dx_mm"],
        "init_dy_mm": ax["dy_mm"],
        "init_dz_mm": ax["dz_mm"],
        "init_e_pos_mm": wk["e_pos_mm"],
        "init_e_pos_match": int(wk["e_pos_match"]),
    }
