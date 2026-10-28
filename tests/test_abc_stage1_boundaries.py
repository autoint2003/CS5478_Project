"""A/B/C Stage-1 boundary + common exit contract + Precision snapshot identity."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data_generation.oracles.plan_a.stage1_runner import plan_a_verify_prefix_is_success_12s
from data_generation.oracles.plan_c.expert_search import lift_and_validate, run_stage1_catch_and_lift
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint
from data_generation.oracles.stage1_exit import (
    classify_stage1_continuation,
    finalize_stage1_exit,
    stage1_exit_from_existing_ic,
)

ROOT = Path(__file__).resolve().parents[1]


def test_plan_a_verify_prefix_maps_to_success_12s_only():
    assert plan_a_verify_prefix_is_success_12s({"success": True, "reason": "SUCCESS_12S"})
    assert not plan_a_verify_prefix_is_success_12s({"success": True, "reason": "LOST_GRASP"})
    assert not plan_a_verify_prefix_is_success_12s({"success": False, "reason": "SUCCESS_12S"})


def test_classification_a1_a2_stage1_failed():
    assert classify_stage1_continuation(stage1_success_12s=False, precision_required=True, plan_id="A") == "STAGE1_FAILED"
    assert classify_stage1_continuation(stage1_success_12s=True, precision_required=False, plan_id="A") == "A1"
    assert classify_stage1_continuation(stage1_success_12s=True, precision_required=True, plan_id="B") == "B2"
    assert classify_stage1_continuation(stage1_success_12s=True, precision_required=True, plan_id="C") == "C2"


def test_plan_c_stage1_only_never_calls_pose_correct():
    sim = MagicMock()
    sim.data.time = 0.0
    catch_out = {"capture_actions": [np.zeros(7, np.float32)], "mechanism": "x", "params": {}, "cand": {}}

    with patch("data_generation.oracles.plan_c.expert_search.restore_ballistic"), \
         patch("data_generation.oracles.plan_c.expert_search.restore_drop"), \
         patch("data_generation.oracles.plan_c.expert_search.arm"), \
         patch("data_generation.oracles.plan_c.expert_search.step_legal"), \
         patch("data_generation.oracles.plan_c.expert_search.latched", return_value=False), \
         patch("data_generation.oracles.plan_c.expert_search.pad_bilateral", return_value={"bilateral": True}), \
         patch("data_generation.oracles.plan_c.expert_search.classify_contacts", return_value={
             "grasp_kind": "sidewall", "opposing_sidewall": 1, "n_sidewall": 2, "n_end_face": 0, "contact_s_mean": 0.1
         }), \
         patch("data_generation.oracles.plan_c.expert_search.measure", return_value={
             "e_theta_deg": 20.0, "e_pos_mm": 25.0, "obj_z": 0.4
         }), \
         patch("data_generation.oracles.plan_c.expert_search.make_snap", return_value={"snap": 1}), \
         patch("data_generation.oracles.plan_c.expert_search.fast_world_lift_sequence", return_value={
             "actions": [np.ones(7, np.float32)], "reached_target_z": True, "obj_z_end": 0.55
         }), \
         patch("data_generation.oracles.plan_c.expert_search.pose_correct_and_handoff") as pose:
        out = run_stage1_catch_and_lift(sim, {"root": 1}, {"ref": 1}, catch_out)
        assert out["ok"] is True
        assert out["pose_correct_called"] is False
        assert out["stage"] == "stage1_exit"
        pose.assert_not_called()


def test_plan_c_historical_lift_and_validate_still_calls_pose():
    sim = MagicMock()
    sim.data.time = 0.0
    catch_out = {"capture_actions": [np.zeros(7, np.float32)], "mechanism": "x", "params": {}, "cand": {},
                 "score": 1.0, "grasp_kind": "sidewall", "vrel_at_first_contact": 0.1, "max_contact_free_s": 0.02}

    with patch("data_generation.oracles.plan_c.expert_search.run_stage1_catch_and_lift", return_value={
        "ok": True,
        "stage1_actions": [np.zeros(7, np.float32)],
        "elev_snap": {"e": 1},
        "surf_brake": {"grasp_kind": "sidewall", "opposing_sidewall": 1, "n_sidewall": 2, "n_end_face": 0, "contact_s_mean": 0.1},
        "surf_elev": {"grasp_kind": "sidewall", "opposing_sidewall": 1, "n_sidewall": 2, "n_end_face": 0, "contact_s_mean": 0.1},
        "m_brake": {"e_theta_deg": 20, "e_pos_mm": 25, "obj_z": 0.3},
        "m_elev": {"e_theta_deg": 18, "e_pos_mm": 22, "obj_z": 0.55},
        "lift": {"reached": True, "obj_z_end": 0.55, "vz_up": 0.12},
    }), \
         patch("data_generation.oracles.plan_c.expert_search.pose_correct_and_handoff", return_value=(
             [np.zeros(7, np.float32)],
             {"SUCCESS_NEW": True, "reason": "SUCCESS_12S", "success_12s": True, "m_handoff": {}, "n_pose_actions": 1},
         )) as pose, \
         patch("data_generation.oracles.plan_c.expert_search.pad_bilateral", return_value={"bilateral": True}), \
         patch("data_generation.oracles.plan_c.expert_search.latched", return_value=False):
        out = lift_and_validate(sim, {"root": 1}, {"ref": 1}, catch_out)
        pose.assert_called()
        assert out["pose_correct_called"] is True
        assert out.get("stage1_actions") is not None


def test_catch_stabilization_stage1_flag_in_signature():
    from data_generation.oracles.plan_c.catch_stabilization import run_sequence

    sig = inspect.signature(run_sequence)
    assert "include_pose_handoff" in sig.parameters
    assert sig.parameters["include_pose_handoff"].default is True


@pytest.mark.parametrize(
    "branch,rel",
    [
        ("A", "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF"),
        ("B", "data/experts_v3_stage1_verified/B_STABLE_PRECISION"),
        ("C", "data/experts_v3_stage1_verified/C_RECATCH_PRECISION"),
    ],
)
def test_existing_ic_stage1_exit_contract_and_snapshot_identity(branch, rel):
    import pickle

    from controllers.recovery.geometry_switch import GeometrySwitch
    from data_generation.oracles.precision.stage1_eligibility import live_verify_stage1_12s
    from data_generation.oracles.precision.state_machine import run_precision
    from data_generation.oracles.precision.state import PrecisionConfig
    from envs.observations.observable_obs import ObservableObsState
    from envs.simulation.capture import restore_capture, take_capture
    from envs.simulation.ground_sim import make_ground_sim
    from models.transformer_policy import DT_POLICY

    folder = ROOT / rel
    if not folder.is_dir():
        pytest.skip(f"missing {rel}")
    pkls = sorted(folder.glob("*.pkl"))
    if not pkls:
        pytest.skip(f"no pickles in {rel}")
    # Prefer a known Precision expert for B/C; any for A
    path = pkls[0]
    blob = pickle.loads(path.read_bytes())
    assert "capture" in blob and "ref" in blob

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)

    fp0 = capture_fingerprint(blob["capture"])
    exit_obj = stage1_exit_from_existing_ic(
        env, {"capture": blob["capture"], "ref": blob["ref"], "sample_id": path.stem},
        plan_id=branch,
        source_ic_id=path.stem,
        stage1_actions=[],
        allow_live_verify=True,
    )
    # Stored exit capture object is the IC capture (not a re-taken asymmetric snap)
    assert exit_obj.capture_fingerprint == fp0
    assert capture_fingerprint(exit_obj.stage1_exit_capture) == fp0

    if not exit_obj.stage1_success_12s:
        res = run_precision(env, exit_obj.as_precision_blob(), branch, PrecisionConfig())
        assert res.status == "STAGE1_INELIGIBLE"
        assert not res.precision_executed
        return

    assert exit_obj.classification in {f"{branch}1", f"{branch}2", f"{branch}_STAGE1_OK_UNCLASSIFIED"}
    # Live verify mutates sim during the 12s window then restores the *same* capture object
    live = live_verify_stage1_12s(env, exit_obj.as_precision_blob())
    assert live.SUCCESS_12S is True
    assert live.capture_fp == fp0
    # Precision entry restores the verified capture object (byte-identical store)
    restore_capture(env, exit_obj.stage1_exit_capture)
    phys = blob["capture"].get("physics") if isinstance(blob["capture"], dict) else None
    if isinstance(phys, dict) and "qpos" in phys:
        assert np.allclose(
            np.asarray(env.sim.data.qpos, float),
            np.asarray(phys["qpos"], float),
            atol=1e-9,
            rtol=0.0,
        )


def _capture_qpos(capture):
    phys = capture.get("physics") if isinstance(capture, dict) else None
    if isinstance(phys, dict) and "qpos" in phys:
        return np.asarray(phys["qpos"], float)
    return None


def test_finalize_does_not_advance_exit_for_precision():
    """finalize_stage1_exit keeps the stored exit capture and restores it after live 12S."""
    import pickle

    from controllers.recovery.geometry_switch import GeometrySwitch
    from data_generation.oracles.precision.stage1_eligibility import live_verify_stage1_12s
    from envs.observations.observable_obs import ObservableObsState
    from envs.simulation.capture import restore_capture
    from envs.simulation.ground_sim import make_ground_sim
    from models.transformer_policy import DT_POLICY

    folder = ROOT / "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF"
    pkls = sorted(folder.glob("*.pkl")) if folder.is_dir() else []
    if not pkls:
        pytest.skip("no A experts")
    blob = pickle.loads(pkls[0].read_bytes())

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)

    fp0 = capture_fingerprint(blob["capture"])
    out = finalize_stage1_exit(
        env,
        plan_id="A",
        source_ic_id="t",
        raw_ic=blob,
        stage1_actions=[],
        stage1_exit_capture=blob["capture"],
        stage1_exit_ref=blob["ref"],
        allow_live_verify=True,
    )
    assert out.capture_fingerprint == fp0
    assert capture_fingerprint(out.stage1_exit_capture) == fp0
    live = live_verify_stage1_12s(env, out.as_precision_blob())
    assert live.capture_fp == fp0
    restore_capture(env, out.stage1_exit_capture)
    q0 = _capture_qpos(blob["capture"])
    if q0 is not None:
        assert np.allclose(np.asarray(env.sim.data.qpos, float), q0, atol=1e-9, rtol=0.0)
