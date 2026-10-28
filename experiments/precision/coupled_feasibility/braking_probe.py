"""Diagnose ~10 mm dz stopping in production inward DualLoop (read-only probe)."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.corrections.axial_inward import correct_dz_inward
from data_generation.oracles.precision.mechanics.axial_sliding import _inward_slip_staged
from data_generation.oracles.precision.mechanics.dual_loop import sample_state
from data_generation.oracles.precision.residuals import evaluate_slip_goal, measure_errors
from data_generation.oracles.precision.state import PrecisionConfig, SlipGoal
from envs.simulation.capture import restore_capture, take_capture
from evaluation.success_criteria import WORKING_POS_MM


def probe_inward_braking(env, blob: dict, cfg: PrecisionConfig | None = None) -> dict[str, Any]:
    """Run production-style inward DualLoop and log stop geometry."""
    cfg = cfg or PrecisionConfig(use_verified_lateral_fallback=False)
    restore_capture(env, blob["capture"])
    e0 = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    err0 = measure_errors(env.sim)
    wp = float(cfg.working_pos_mm)
    budget_sq = wp * wp - float(e0[0]) ** 2 - float(e0[1]) ** 2
    axis_tol = float(np.sqrt(max(budget_sq, 0.0))) if budget_sq > 0 else 3.0
    axis_tol_clipped = float(np.clip(axis_tol, 3.0, 8.0))

    # Theoretical stop band used by _inward_slip_staged
    theory = {
        "working_pos_mm": wp,
        "ex0_mm": float(e0[0]),
        "ey0_mm": float(e0[1]),
        "ez0_mm": float(e0[2]),
        "axis_tol_raw_mm": axis_tol,
        "axis_tol_clipped_mm": axis_tol_clipped,
        "brake_margin_mm": 1.0,
        "use_euclidean_stop": False,
        "expected_stop_when_abs_ez_le_mm": axis_tol_clipped + 1.0,
        "note": (
            "Inward DualLoop stops on |ez|<=clip(sqrt(wp^2-ex^2-ey^2),[3,8]) with "
            "use_euclidean_stop=False; ~8–9 mm residual is by design, not timeout."
        ),
    }

    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    # Prefer full correct_dz_inward if entry allows; else raw DualLoop from current pose
    try:
        out = correct_dz_inward(env, blob, cfg)
        mechanism = "correct_dz_inward"
    except Exception as exc:
        restore_capture(env, blob["capture"])
        out = _inward_slip_staged(env, blob, z_ref, float(e0[0]), milestone_mm=12.0, cfg=cfg)
        out = {**out, "correct_dz_exception": str(exc)}
        mechanism = "_inward_slip_staged_direct"

    err1 = measure_errors(env.sim)
    e1 = np.asarray(err1["e_signed_mm"], float)
    goal = SlipGoal(
        axis="dz",
        desired_direction=np.array([0.0, 0.0, -1.0]),
        axis_tol_mm=axis_tol_clipped,
        working_pos_mm=wp,
        use_euclidean_stop=False,
        brake_margin_mm=1.0,
    )
    hit, why = evaluate_slip_goal(err1, goal)

    return {
        "mechanism": mechanism,
        "theory": theory,
        "errors0": err0,
        "errors1": err1,
        "dz0_mm": float(e0[2]),
        "dz1_mm": float(e1[2]),
        "requested_correction_mm": abs(float(e0[2])),
        "actual_displacement_mm": abs(float(e0[2])) - abs(float(e1[2])),
        "residual_at_end_mm": abs(float(e1[2])),
        "e_pos0_mm": float(err0["e_pos_mm"]),
        "e_pos1_mm": float(err1["e_pos_mm"]),
        "stop_reason": out.get("stop_reason") or out.get("reason"),
        "goal_reached": out.get("goal_reached"),
        "goal_test_at_end": {"hit": hit, "why": why},
        "ok": bool(out.get("ok")),
        "bilateral_end": bool(err1.get("bilateral")),
        "primitive_log": {k: v for k, v in out.items() if k not in ("actions", "capture", "series")},
    }
