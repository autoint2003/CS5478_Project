"""Lightweight mainline regressions. No SAC training. No large grids."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg  # noqa: E402
from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.grasp_sim import GraspSim  # noqa: E402
from envs.recovery_env import RecoveryEnv  # noqa: E402
from training.replay_core import (  # noqa: E402
    dump_ball_contacts,
    load_impact_ic,
    load_offset_ic,
    make_impact_sim,
    make_offset_sim,
    obs_from_sim,
    pack_replay,
    params_from_frozen,
    ball_is_parked,
    prepare_replay,
    restore_replay,
    rollout_from_replay,
    verify_frozen_hashes,
)


def test_imports():
    import controllers.nominal  # noqa: F401
    import controllers.residual  # noqa: F401
    import controllers.rule_based_recovery  # noqa: F401
    import envs.deterioration  # noqa: F401
    import envs.recovery_env  # noqa: F401
    import training.train_recovery  # noqa: F401
    import training.collect_recovery_states  # noqa: F401


def test_frozen_hashes():
    verify_frozen_hashes()


def test_npz_load():
    ic = load_offset_ic(0)
    assert "qpos" in ic and "p_des" in ic
    imp = load_impact_ic(8.5)
    assert imp["qpos"].shape[0] == 23


def test_gripper_sign():
    # Positive squeeze effort maps to negative tendon tau (close).
    # Frozen RULE: tau_secure=-18 (close), tau_open=-1 (unload/open).
    params, _ = params_from_frozen()
    assert params.tau_secure < params.tau_open
    assert params.tau_secure < 0
    from controllers.gripper_controller import fg_to_tau as _fg
    from envs.ids import resolve_ids
    from envs.xml_build import write_panda_torque
    import mujoco
    write_panda_torque()
    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "scene.xml"))
    ids = resolve_ids(m)
    assert _fg(18.0, ids) < 0


def test_snapshot_roundtrip():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_offset_sim(cfg)
    ic = load_offset_ic(0)
    restore_replay(sim, ic)
    q0 = np.array(sim.data.qpos, float).copy()
    p0 = np.asarray(sim.fsm.p_des, float).copy()
    packed = pack_replay(sim)
    sim.reset(0.20, 1.0, np.zeros(3))
    restore_replay(sim, packed)
    assert float(np.max(np.abs(np.array(sim.data.qpos, float) - q0))) < 1e-9
    assert float(np.max(np.abs(np.asarray(sim.fsm.p_des, float) - p0))) < 1e-9


def test_offset_zero_rule_smoke():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    params, _ = params_from_frozen()
    sim = make_offset_sim(cfg)
    ic = load_offset_ic(0)
    prepare_replay(sim, ic)
    o = obs_from_sim(sim)
    assert o["nL"] > 0 and o["nR"] > 0
    _, z = rollout_from_replay(sim, ic, gains, params, True, 0.4)
    _, r = rollout_from_replay(sim, ic, gains, params, False, 4.0)
    assert z["grasp_retention"] == 1
    assert r["modes"][0] == "STABILIZE"
    print("offset smoke ZERO keep", z["grasp_retention"], "RULE rec", r["recovery_success"], r["path"])


def test_impact_replay_smoke():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    params, _ = params_from_frozen()
    sim = make_impact_sim(cfg)
    ic = load_impact_ic(8.5)
    restore_replay(sim, ic)
    assert not ball_is_parked(sim), "impact IC itself should contain the unparked ball"
    prepare_replay(sim, ic)
    assert ball_is_parked(sim)
    o = obs_from_sim(sim)
    assert o["nL"] > 0 and o["nR"] > 0
    bcs = dump_ball_contacts(sim)
    assert all(c["kind"] != "ball-finger" for c in bcs)
    q_ball = np.array(sim.data.qpos[sim.ball_qadr : sim.ball_qadr + 3], float).copy()
    _, z = rollout_from_replay(sim, ic, gains, params, True, 0.4, restore=False)
    assert ball_is_parked(sim)
    assert float(np.max(np.abs(np.array(sim.data.qpos[sim.ball_qadr : sim.ball_qadr + 3], float) - q_ball))) < 1e-9
    _, r = rollout_from_replay(sim, ic, gains, params, False, 4.0)
    assert ball_is_parked(sim)
    print("impact smoke ZERO keep", z["grasp_retention"], "RULE rec", r["recovery_success"])


def test_recovery_env_import_step():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    env = RecoveryEnv(cfg=cfg, snapshots=[], action_kind="2d")
    obs, _ = env.reset()
    env.step(env.action_space.sample() * 0)


def test_nominal_grasp_smoke():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = GraspSim(cfg)
    sim.reset()
    for _ in range(8000):
        sim.physics_step(None, False)
        sim.maybe_capture_reference()
        if sim.fsm.phase == "lift" and sim.captured:
            print("nominal reached lift")
            return
    raise AssertionError("did not reach lift in 200 steps")


if __name__ == "__main__":
    test_imports()
    test_frozen_hashes()
    test_npz_load()
    test_gripper_sign()
    test_snapshot_roundtrip()
    test_nominal_grasp_smoke()
    test_offset_zero_rule_smoke()
    test_impact_replay_smoke()
    test_recovery_env_import_step()
    print("MAINLINE REGRESSION: PASS")
