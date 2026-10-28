"""Regression: Stage-1 SUCCESS_12S eligibility gate."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.entry_conditions import entry_gate
from data_generation.oracles.precision.stage1_eligibility import (
    STATUS_INELIGIBLE,
    capture_fingerprint,
    check_stage1_eligible,
    lookup_evidence,
)
from data_generation.oracles.precision.state import PrecisionConfig
from data_generation.oracles.precision.state_machine import run_precision
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.capture import restore_capture
from envs.simulation.ground_sim import make_ground_sim
from models.transformer_policy import DT_POLICY

INVALID_IC = (
    ROOT
    / "results/data_generation/stage2_abc_final_v2/ics"
    / "PILOT_A__ball_train_v2_0012__93a263b2b1a49030.pkl"
)
# Prefer an A Stage-1 terminal known to pass 12s in the audit
VALID_IC = ROOT / "results/data_generation/plan_a_stage1_terminals_v1/ics/A_STAGE1__ball_train_0001.pkl"


def _env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


@pytest.fixture(scope="module")
def env():
    return _env()


def test_case_a_invalid_ic_rejected(env):
    assert INVALID_IC.is_file(), "counterexample IC missing"
    blob = pickle.loads(INVALID_IC.read_bytes())
    # Short gate may still pass — must not imply Stage-1 success
    restore_capture(env, blob["capture"])
    gate = entry_gate(env)
    restore_capture(env, blob["capture"])
    elig = check_stage1_eligible(
        env,
        {"capture": blob["capture"], "ref": blob["ref"], "sample_id": "PILOT_A__ball_train_v2_0012__93a263b2b1a49030"},
        allow_live_verify=True,
    )
    assert elig.eligible is False
    assert elig.status == STATUS_INELIGIBLE
    assert gate.get("is_stage1_success") is False

    restore_capture(env, blob["capture"])
    res = run_precision(
        env,
        {"capture": blob["capture"], "ref": blob["ref"]},
        "A",
        PrecisionConfig(use_verified_lateral_fallback=False, evaluate_handoff=False),
    )
    assert res.status == "STAGE1_INELIGIBLE"
    assert res.precision_executed is False
    assert len(res.actions or []) == 0
    assert any("STAGE1_INELIGIBLE" in f for f in (res.failures or []))


def test_case_b_valid_ic_admitted(env):
    if not VALID_IC.is_file():
        pytest.skip("valid IC missing")
    blob = pickle.loads(VALID_IC.read_bytes())
    fp_before = capture_fingerprint(blob["capture"])
    elig = check_stage1_eligible(
        env,
        {"capture": blob["capture"], "ref": blob["ref"], "sample_id": "A_STAGE1__ball_train_0001"},
        allow_live_verify=True,
    )
    assert elig.eligible is True
    assert elig.SUCCESS_12S is True
    # Eligibility must not mutate stored capture bytes
    assert capture_fingerprint(blob["capture"]) == fp_before
    restore_capture(env, blob["capture"])
    # Admission path reaches beyond STAGE1_INELIGIBLE (may skip or execute Precision)
    res = run_precision(
        env,
        {"capture": blob["capture"], "ref": blob["ref"]},
        "A_STAGE1",
        PrecisionConfig(use_verified_lateral_fallback=False, evaluate_handoff=False),
    )
    assert res.status != "STAGE1_INELIGIBLE"


def test_case_c_missing_evidence_no_silent_short_hold(env):
    if not VALID_IC.is_file():
        pytest.skip("valid IC missing")
    blob = pickle.loads(VALID_IC.read_bytes())
    # Force missing evidence path: disallow live verify and use empty registry path
    empty = ROOT / "results/diagnostics/precision_baseline_rebuild/empty_evidence.json"
    empty.write_text('{"eval_config": {}, "by_capture_fp": {}}', encoding="utf-8")
    elig = check_stage1_eligible(
        env,
        {"capture": blob["capture"], "ref": blob["ref"]},
        allow_live_verify=False,
        evidence_path=empty,
    )
    assert elig.eligible is False
    assert elig.reason == "MISSING_OR_MISMATCHED_EVIDENCE"
    # Short gate alone must not flip eligibility
    restore_capture(env, blob["capture"])
    assert entry_gate(env).get("is_stage1_success") is False


def test_case_d_official_expert_replay_unchanged_actions(env):
    """Replay a valid official expert's action tape from its stored capture."""
    from envs.step_legal import step_legal

    expert = (
        ROOT
        / "data/experts_v2_official/A_STABLE_DIRECT_HANDOFF"
        / "A_STABLE_DIRECT_HANDOFF__A_STAGE1__ball_train_0001.pkl"
    )
    if not expert.is_file():
        pytest.skip("official expert missing")
    rec = pickle.loads(expert.read_bytes())
    assert rec.get("SUCCESS_NEW") is True
    blob = {"capture": rec["capture"], "ref": rec["ref"]}
    elig = check_stage1_eligible(env, blob, allow_live_verify=True)
    assert elig.eligible is True
    restore_capture(env, blob["capture"])
    fp0 = capture_fingerprint(blob["capture"])
    acts = rec.get("actions_7d") or []
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
    # Stored capture bytes unchanged
    assert capture_fingerprint(rec["capture"]) == fp0


def test_registry_lookup_matches_fingerprint():
    if not INVALID_IC.is_file():
        pytest.skip("IC missing")
    blob = pickle.loads(INVALID_IC.read_bytes())
    fp = capture_fingerprint(blob["capture"])
    ev = lookup_evidence(fp)
    # After evidence registry build, invalid IC must be recorded as FAILURE
    if ev is not None:
        assert bool(ev.get("SUCCESS_12S")) is False
