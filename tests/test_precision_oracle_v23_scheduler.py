"""Focused closed-loop scheduler tests for Precision Oracle v2.3 (no new recovery)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.residuals import (
    evaluate_slip_goal,
    handoff_geometry_ready,
)
from data_generation.oracles.precision.state import PrecisionConfig, SlipGoal, STOP_GOAL_REACHED
from data_generation.oracles.precision.state_machine import (
    measure_errors,
    probe_direct_handoff,
    run_precision,
)
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.ground_sim import make_ground_sim
from models.transformer_policy import DT_POLICY

STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
POST_P5 = ROOT / "results/diagnostics/handoff_state_optimization/snapshots/post_P5.pkl"


def _env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def _load(plan: str):
    p = {
        "A": STATES / "plan_a__ball_train_v2_0070__stable.pkl",
        "B": STATES / "plan_b__hard_910220__stable.pkl",
        "C": STATES / "plan_c__gbi_010__stable.pkl",
    }[plan]
    return pickle.loads(p.read_bytes())


@pytest.fixture(scope="module")
def env():
    return _env()


def test_skip_when_already_successful(env):
    st = _load("C")
    r = run_precision(env, st, "C", PrecisionConfig(allow_horizontal_dx=False))
    assert r.precision_skipped
    assert (r.handoff or {}).get("SUCCESS_NEW")


def test_post_p5_skip_no_unnecessary_dx(env):
    if not POST_P5.exists():
        pytest.skip("post_P5 snapshot missing")
    p5 = pickle.loads(POST_P5.read_bytes())
    r = run_precision(env, {"capture": p5["capture"], "ref": p5["ref"]}, "B", PrecisionConfig())
    assert r.precision_skipped
    assert (r.handoff or {}).get("SUCCESS_NEW")
    assert len(r.actions) == 0


def test_plan_b_dx_then_dz_reassess(env):
    st = _load("B")
    r = run_precision(env, st, "B", PrecisionConfig(allow_horizontal_dx=False))
    prims = [c.get("primitive") for c in ((r.primitive_logs or {}).get("cycles") or [])]
    assert any(p and "dx" in str(p) for p in prims)
    assert any(p and str(p).startswith("dz") for p in prims)
    assert (r.handoff or {}).get("SUCCESS_NEW")


def test_goal_reached_label_not_success_new():
    goal = SlipGoal(axis="dz", desired_direction=[0, 0, -1], axis_tol_mm=8.0, use_euclidean_stop=False)
    err = {"e_signed_mm": [0.0, 0.0, 5.0], "dtheta_deg": 1.0, "e_pos_mm": 5.0, "bilateral": True, "v_rel_norm": 0.01}
    hit, why = evaluate_slip_goal(err, goal)
    assert hit and why == STOP_GOAL_REACHED
    # Geometry-ready is Level-2, not SUCCESS_NEW
    assert handoff_geometry_ready(err, PrecisionConfig())


def test_handoff_probe_restores_caller(env):
    st = _load("C")
    restore_capture(env, st["capture"])
    before = take_capture(env)
    probe_direct_handoff(env, st["capture"], st["ref"], stabilize_n=8)
    after = take_capture(env)
    q0 = np.asarray(before["physics"]["qpos"], float)
    q1 = np.asarray(after["physics"]["qpos"], float)
    assert float(np.max(np.abs(q0 - q1))) < 1e-5


def test_cycle_budget_zero_no_infinite_loop(env):
    st = _load("A")
    r = run_precision(
        env,
        st,
        "A",
        PrecisionConfig(allow_horizontal_dx=False, max_correction_cycles=0, try_direct_handoff=False),
    )
    assert r.status.startswith("PRECISION")
    cycles = (r.primitive_logs or {}).get("cycles") or []
    assert cycles == [] or all(c.get("primitive") in (None, "none") for c in cycles)


def test_entry_grip_settle_records_load_bearing_start(env):
    """Unilateral Plan-C IC: gate hold proves stability; settle must enter the tape."""
    ic = (
        ROOT
        / "results/data_generation/stage2_abc_mining/v2/ics"
        / "C__gbi_032__v2__5c0760e331323457.pkl"
    )
    if not ic.exists():
        pytest.skip("Plan-C unilateral IC missing")
    pack = pickle.loads(ic.read_bytes())
    restore_capture(env, pack["capture"])
    pre = measure_errors(env.sim)
    assert not (
        pre.get("bilateral")
        and float(pre.get("fn_L") or 0) >= 1.0
        and float(pre.get("fn_R") or 0) >= 1.0
    )
    r = run_precision(env, {"capture": pack["capture"], "ref": pack["ref"]}, "C", PrecisionConfig())
    assert "entry_grip_settle" in (r.phases or [])
    settle = (r.primitive_logs or {}).get("entry_grip_settle") or {}
    assert settle.get("ok") is True
    assert len(r.actions or []) >= 10
    # Must not abort solely because pre-hold was unilateral when settle restores grasp.
    assert r.status != "STABLE_GATE_FAILED" or "entry_grip_settle_failed" in (r.failures or [])


def test_entry_grip_settle_records_high_motion_start(env):
    """Bilateral contact alone must not bypass stabilization at high relative speed."""
    ic = (
        ROOT
        / "results/data_generation/stage2_abc_mining/v2/ics"
        / "C__gbi_092__v2__16eef59d75927479.pkl"
    )
    if not ic.exists():
        pytest.skip("Plan-C high-motion IC missing")
    pack = pickle.loads(ic.read_bytes())
    restore_capture(env, pack["capture"])
    pre = measure_errors(env.sim)
    assert pre.get("bilateral")
    assert float(pre.get("v_rel_norm") or 0) > 0.08 or float(pre.get("w_rel_norm") or 0) > 1.5
    r = run_precision(env, {"capture": pack["capture"], "ref": pack["ref"]}, "C", PrecisionConfig())
    assert "entry_grip_settle" in (r.phases or [])
    settle = (r.primitive_logs or {}).get("entry_grip_settle") or {}
    assert settle.get("ok") is True
    assert "V_REL_HIGH" in (settle.get("pre_reasons") or []) or "W_REL_HIGH" in (
        settle.get("pre_reasons") or []
    )
