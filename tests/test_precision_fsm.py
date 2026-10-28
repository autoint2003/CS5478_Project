"""Unit tests for P9 Precision FSM planning and dual success metrics."""

from __future__ import annotations

import numpy as np

from data_generation.oracles.precision.fsm import (
    PrecisionState,
    classify_case,
    next_skill_from_plan,
)
from data_generation.oracles.precision.state import PrecisionConfig
from data_generation.oracles.precision.success_versions import annotate_dual_theta_success


def _err(dx, dz, th=5.0, e_pos=None):
    e_pos = e_pos if e_pos is not None else float(np.hypot(dx, dz))
    return {
        "e_signed_mm": [float(dx), 0.0, float(dz)],
        "dtheta_deg": float(th),
        "e_pos_mm": float(e_pos),
        "bilateral": True,
        "dx_mm": float(dx),
        "dz_mm": float(dz),
    }


def test_case1_dx_large_dz_small_orders_dx_first():
    cfg = PrecisionConfig()
    plan = classify_case(_err(15.0, 3.0, th=12.0), cfg)
    assert plan.case == "CASE1_DX_LARGE_DZ_SMALL"
    assert plan.planned_skills[0] == PrecisionState.CORRECT_DX.value
    assert PrecisionState.CORRECT_DZ.value in plan.planned_skills
    assert plan.planned_skills.index(PrecisionState.CORRECT_DX.value) < plan.planned_skills.index(
        PrecisionState.CORRECT_DZ.value
    )


def test_case2_dz_large_dx_small_orders_dz_first():
    cfg = PrecisionConfig()
    plan = classify_case(_err(2.0, 25.0, th=5.0), cfg)
    assert plan.case == "CASE2_DZ_LARGE_DX_SMALL"
    assert plan.planned_skills[0] == PrecisionState.CORRECT_DZ.value
    nxt = next_skill_from_plan(
        plan,
        done_dx=False,
        done_dz=False,
        done_dtheta=False,
        reconfigure_done=True,
        need={"dx_needed": False, "dz_needed": True, "dtheta_needed": False},
        max_dtheta_attempts=1,
        dtheta_attempts=0,
    )
    assert nxt == PrecisionState.CORRECT_DZ


def test_case3_both_large_plans_reconfigure():
    cfg = PrecisionConfig()
    plan = classify_case(_err(18.0, 25.0), cfg)
    assert plan.case == "CASE3_BOTH_LARGE"
    assert any(s.startswith("RECONFIGURE") for s in plan.planned_skills)


def test_case4_both_small_optional_dtheta():
    cfg = PrecisionConfig()
    plan = classify_case(_err(1.0, 2.0, th=14.0), cfg)
    assert plan.case == "CASE4_BOTH_SMALL"
    assert plan.planned_skills[0] == PrecisionState.CORRECT_DTHETA.value


def test_case4_skip_dtheta_when_within_tol():
    cfg = PrecisionConfig()
    plan = classify_case(_err(1.0, 2.0, th=5.0), cfg)
    assert plan.case == "CASE4_BOTH_SMALL"
    assert PrecisionState.CORRECT_DTHETA.value not in plan.planned_skills


def test_no_silent_dx_when_dz_first_selected():
    cfg = PrecisionConfig()
    plan = classify_case(_err(2.0, 25.0), cfg)
    nxt = next_skill_from_plan(
        plan,
        done_dx=False,
        done_dz=False,
        done_dtheta=False,
        reconfigure_done=True,
        need={"dx_needed": True, "dz_needed": True, "dtheta_needed": False},
        max_dtheta_attempts=1,
        dtheta_attempts=0,
    )
    assert nxt == PrecisionState.CORRECT_DZ


def test_dual_success_annotation():
    hand = {
        "SUCCESS_12S": True,
        "SUCCESS_NEW": True,
        "window": {"mean_theta": 8.0, "max_theta": 9.0, "mean_pos": 4.0, "matrix": {}},
    }
    out = annotate_dual_theta_success(hand)
    assert out["SUCCESS_STRICT_10DEG"] is True
    assert out["SUCCESS_RELAXED_15DEG"] is True

    hand2 = {
        "SUCCESS_12S": True,
        "SUCCESS_NEW": False,
        "window": {"mean_theta": 12.0, "max_theta": 13.0, "mean_pos": 4.0, "matrix": {}},
    }
    out2 = annotate_dual_theta_success(hand2)
    assert out2["SUCCESS_STRICT_10DEG"] is False
    assert out2["SUCCESS_RELAXED_15DEG"] is True
