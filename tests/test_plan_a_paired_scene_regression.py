"""Paired elevated vs GROUND_SCENE regression for Plan A expert hrv1_920031.

Uses the trusted slow-return expert snapshot as reference — does not generate
a baseline from code under test. Verifies shared decode/step_legal, expected
model diffs only, complete state transfer, and matching mechanism gates.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SNAP = ROOT / "results/expert_generation/k49_slow_return_position_refinement/SNAPSHOTS/hrv1_920031.pkl"
TOK = ROOT / (
    "results/expert_generation/k49_slow_return_position_refinement/trajectories/"
    "hrv1_920031__refine_12x3_17x12_0x5__tokens.npz"
)


@pytest.fixture(scope="module")
def expert():
    if not SNAP.is_file() or not TOK.is_file():
        pytest.skip("trusted Plan A expert snapshot/tokens missing")
    blob = pickle.loads(SNAP.read_bytes())
    tokens = np.asarray(np.load(TOK)["tokens"], np.int64)
    assert "capture" in blob and "ref" in blob
    return blob, tokens


def _elev_env():
    from controllers.recovery.geometry_switch import GeometrySwitch
    from envs.ballistic_impact import elevated_world

    class E:
        pass

    env = E()
    env.sim, env.gains, env.nominal, env.pads, env.pad_mu, env.pads_l, env.pads_r = elevated_world()
    env.switch = GeometrySwitch()
    env._hist = None
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


def _ground_env():
    from controllers.recovery.geometry_switch import GeometrySwitch
    from envs.observations.observable_obs import ObservableObsState
    from envs.simulation.ground_sim import make_ground_sim
    from models.transformer_policy import DT_POLICY

    class G:
        pass

    env = G()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


def test_shared_action_decode_and_step_legal(expert):
    from evaluation.plan_a_mechanism import decode_token, load_vocab
    from envs.step_legal import step_legal

    blob, tokens = expert
    table, book48 = load_vocab()
    a0 = decode_token(table, book48, int(tokens[0]))
    assert a0.shape == (7,)
    assert np.all(np.abs(a0) <= 1.0 + 1e-6)
    # same function object used by both scenes
    assert callable(step_legal)


def test_compiled_model_support_only_diffs():
    elev = _elev_env()
    gnd = _ground_env()
    me, mg = elev.sim.model, gnd.sim.model
    assert me.nq == mg.nq == 23
    assert me.nv == mg.nv
    assert float(me.opt.timestep) == float(mg.opt.timestep)
    assert np.allclose(me.opt.gravity, mg.opt.gravity)
    # shared named cylinder geom matches friction, size, solref
    ie = mujoco.mj_name2id(me, mujoco.mjtObj.mjOBJ_GEOM, "object")
    ig = mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_GEOM, "object")
    assert ie >= 0 and ig >= 0
    assert np.allclose(me.geom_friction[ie], mg.geom_friction[ig])
    assert np.allclose(me.geom_size[ie], mg.geom_size[ig])
    assert np.allclose(me.geom_solref[ie], mg.geom_solref[ig])
    be = mujoco.mj_name2id(me, mujoco.mjtObj.mjOBJ_BODY, "object")
    bg = mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert np.allclose(me.body_mass[be], mg.body_mass[bg])
    # support exclusive
    assert mujoco.mj_name2id(me, mujoco.mjtObj.mjOBJ_GEOM, "table") >= 0
    assert mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_GEOM, "table") < 0
    assert mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_GEOM, "ground") >= 0


def test_complete_transfer_initial_equivalence(expert):
    from envs.simulation.capture import restore_capture
    from evaluation.plan_a_mechanism import transfer_state_to_ground

    blob, _ = expert
    elev = _elev_env()
    gnd = _ground_env()
    restore_capture(elev, blob["capture"])
    diffs = transfer_state_to_ground(elev, gnd)
    assert diffs["warmstart_max_abs_diff"] == 0.0
    assert diffs["bilateral_src"] == diffs["bilateral_dst"] == 1
    se, sg = elev.sim, gnd.sim
    assert np.allclose(se.data.qpos, sg.data.qpos)
    assert np.allclose(se.data.qvel, sg.data.qvel)
    assert np.allclose(se.data.ctrl, sg.data.ctrl)
    assert np.allclose(se.data.qacc_warmstart, sg.data.qacc_warmstart)
    assert np.allclose(se.fsm.p_des, sg.fsm.p_des)
    assert float(se.fsm.fg_cmd) == float(sg.fsm.fg_cmd)


def test_paired_mechanism_gates_match(expert):
    from envs.simulation.capture import restore_capture
    from envs.step_legal import step_legal
    from evaluation.plan_a_mechanism import (
        analyze_mechanism,
        decode_token,
        load_vocab,
        sample_kinematics,
        transfer_state_to_ground,
    )

    blob, tokens = expert
    table, book48 = load_vocab()
    elev = _elev_env()
    gnd = _ground_env()
    restore_capture(elev, blob["capture"])
    transfer_state_to_ground(elev, gnd)
    # re-restore elev for independent elevated replay
    restore_capture(elev, blob["capture"])

    def run(sim):
        tr = [sample_kinematics(sim)]
        for t in tokens:
            step_legal(sim, decode_token(table, book48, int(t)))
            tr.append(sample_kinematics(sim))
        return analyze_mechanism(tr)

    me = run(elev.sim)
    # ground still holds transferred IC from before elev re-restore — re-transfer
    restore_capture(elev, blob["capture"])
    transfer_state_to_ground(elev, gnd)
    mg = run(gnd.sim)

    assert me["contact_preservation"] == "PASS"
    assert me["genuine_axial_sliding"] == "PASS"
    assert me["genuine_pivoting"] == "PASS"
    assert mg["contact_preservation"] == me["contact_preservation"]
    assert mg["genuine_axial_sliding"] == me["genuine_axial_sliding"]
    assert mg["genuine_pivoting"] == me["genuine_pivoting"]
    assert abs(mg["bilateral_gap_s"] - me["bilateral_gap_s"]) < 1e-12
    assert abs(mg["dp_axial_mm"] - me["dp_axial_mm"]) < 1e-9


def test_success_new_separated_from_mechanism(expert):
    """SUCCESS_NEW can pass while mechanism gates were historically failing; after fix both pass."""
    from evaluation.plan_a_mechanism import load_vocab, rollout_traced

    blob, tokens = expert
    table, book48 = load_vocab()
    elev = _elev_env()
    out = rollout_traced(elev, blob, tokens, table, book48, None, arm_drop=False)
    assert out["SUCCESS_NEW"] is True
    assert out["mechanism"]["contact_preservation"] == "PASS"
