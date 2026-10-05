"""Observable policy observation must not consume object GT."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_observe_signature():
    from envs.observable_obs import observe_observable

    names = list(inspect.signature(observe_observable).parameters)
    assert names == ["model", "data", "ids", "fsm", "tactile", "hist"]


def test_source_no_object_gt():
    src = (ROOT / "envs" / "observable_obs.py").read_text(encoding="utf-8")
    assert "object_body" not in src
    assert "object_geom" not in src
    assert "physical_pack" not in src
    assert "obs_from_sim" not in src
    assert "e_x_GT" not in src
    assert '["e_x"]' not in src
    assert "v_rel" not in src
    assert "obj_z" not in src


def test_dim_and_placeholder():
    from envs.observable_obs import OBS_DIM_OBSERVABLE, OBS_NAMES, ObservableObsState, observe_observable
    from envs.config_util import load_yaml, merge_sim_config
    from envs.grasp_sim import GraspSim
    from envs.observable_reward import read_tactile
    from training.replay_core import MASS, MU

    assert len(OBS_NAMES) == OBS_DIM_OBSERVABLE == 27
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    _, tac = read_tactile(sim.model, sim.data, sim.ids)
    tac_inv = dict(tac)
    tac_inv["estimate_valid"] = False
    tac_inv["e_hat_x"] = float("nan")
    tac_inv["valid_L"] = False
    tac_inv["valid_R"] = False
    hist = ObservableObsState(sim.dt_policy)
    obs = observe_observable(sim.model, sim.data, sim.ids, sim.fsm, tac_inv, hist)
    assert obs.shape == (27,)
    assert np.all(np.isfinite(obs))
    assert float(obs[0]) == 0.0
    assert float(obs[1]) == 0.0


def test_poison_object_qpos_cached_tactile():
    from envs.config_util import load_yaml, merge_sim_config
    from envs.grasp_sim import GraspSim
    from envs.observable_obs import ObservableObsState, observe_observable
    from envs.observable_reward import read_tactile
    from training.replay_core import MASS, MU, object_free_adr
    import mujoco

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    _, tac = read_tactile(sim.model, sim.data, sim.ids)
    h1 = ObservableObsState(sim.dt_policy)
    o1 = observe_observable(sim.model, sim.data, sim.ids, sim.fsm, tac, h1)
    qadr, _ = object_free_adr(sim)
    sim.data.qpos[qadr : qadr + 3] += 0.08
    mujoco.mj_forward(sim.model, sim.data)
    h2 = ObservableObsState(sim.dt_policy)
    o2 = observe_observable(sim.model, sim.data, sim.ids, sim.fsm, tac, h2)
    assert np.allclose(o1, o2, atol=1e-9), (o1 - o2)


def test_oracle_obs_dim_unchanged():
    from envs.config_util import load_yaml, merge_sim_config
    from envs.physical_recovery import OBS_DIM
    from envs.recovery_env import RecoveryEnv
    from training.offset_construct import make_one_training_ic

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    snap = make_one_training_ic(cfg, np.random.default_rng(1))
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d", reward_mode="oracle")
    obs, _ = env.reset(options={"snapshot": snap})
    assert obs.shape[0] == OBS_DIM
    env2 = RecoveryEnv(
        cfg=cfg, snapshots=[snap], action_kind="recovery4d", reward_mode="observable_tactile"
    )
    obs2, _ = env2.reset(options={"snapshot": snap})
    assert obs2.shape[0] == 27


def test_reward_formula_untouched():
    src = (ROOT / "envs" / "observable_reward.py").read_text(encoding="utf-8")
    assert "T_PROGRESS_S = 0.100" in src
    assert "a[0] ** 2 + a[1] ** 2 + a[2] ** 2" in src
    assert "lambda_D" not in src
