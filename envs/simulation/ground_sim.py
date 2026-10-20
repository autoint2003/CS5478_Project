"""MuJoCo simulator for GROUND_SCENE (no elevated table)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from envs.ballistic_impact import PAIR_MU, assert_noslip
from envs.simulation.ground_scene import (
    CYL_M,
    SCENE_ID,
    write_ground_scene,
)
from envs.simulation.grasp_sim import apply_solver_from_cfg, reset_home
from envs.simulation.impact_ball import apply_ball_free_state
from envs.simulation.mujoco_ids import resolve_ids
from envs.simulation.object_dynamics import set_finger_object_sliding_mu
from envs.simulation.panda_xml import write_panda_torque
from envs.simulation.yaml_config import load_yaml, merge_sim_config

ROOT = Path(__file__).resolve().parents[2]


class GroundBallisticSim:
    """Ballistic-capable sim on GROUND_SCENE. Mirrors BallisticSim API."""

    def __init__(self, cfg: dict):
        scene = write_ground_scene()
        write_panda_torque()
        self.cfg = cfg
        self.scene_id = SCENE_ID
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        apply_solver_from_cfg(self.model, cfg)
        self.data = mujoco.MjData(self.model)
        self.ids = resolve_ids(self.model)
        from controllers.nominal import GraspFSM
        from controllers.recovery.residual_limits import ResidualLimiter
        from envs.simulation.grasp_quality import DeteriorationMeter

        self.fsm = GraspFSM(cfg)
        self.meter = DeteriorationMeter(cfg)
        self.limiter = ResidualLimiter(cfg)
        self.n_sub = int(cfg.get("n_substeps", 10))
        self.dt_policy = float(self.model.opt.timestep) * self.n_sub
        self.mass = CYL_M
        self.friction = PAIR_MU
        self.captured = False
        self.ball_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "impact_ball")
        self.ball_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "impact_ball")
        self.ball_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qadr = int(self.model.jnt_qposadr[self.ball_jnt])
        self.ball_dadr = int(self.model.jnt_dofadr[self.ball_jnt])
        self.guide_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball_guide")
        self.guide_eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_guide_weld")
        self.guide_mocap = int(self.model.body_mocapid[self.guide_body])

    def set_guide_weld(self, on: bool) -> None:
        flag = int(bool(on))
        if hasattr(self.data, "eq_active"):
            self.data.eq_active[self.guide_eq] = flag
        if hasattr(self.model, "eq_active0"):
            self.model.eq_active0[self.guide_eq] = flag

    def set_guide_pos(self, p) -> None:
        self.data.mocap_pos[self.guide_mocap] = np.asarray(p, float).reshape(3)
        self.data.mocap_quat[self.guide_mocap] = np.array([1.0, 0.0, 0.0, 0.0])

    def eq_active(self) -> int:
        if hasattr(self.data, "eq_active"):
            return int(self.data.eq_active[self.guide_eq])
        return int(self.model.eq_active0[self.guide_eq])

    def park_ball(self) -> None:
        apply_ball_free_state(self, np.array([1.00, 0.80, 1.20]), np.zeros(3))

    def snapshot(self) -> dict:
        from envs.simulation.grasp_sim import GraspSim

        return GraspSim.snapshot(self)

    def load_snapshot(self, snap) -> None:
        from envs.simulation.grasp_sim import GraspSim

        GraspSim.load_snapshot(self, snap)

    def physics_step(self, residual=None, in_recovery: bool = False) -> None:
        from envs.simulation.grasp_sim import GraspSim

        GraspSim.physics_step(self, residual, in_recovery=in_recovery)

    def maybe_capture_reference(self) -> None:
        from envs.simulation.grasp_sim import GraspSim

        GraspSim.maybe_capture_reference(self)

    def reset(self, mass=None, friction=None, grasp_offset=None) -> None:
        from envs.outcomes.ground_drop import disarm, restore_state

        self.mass = CYL_M
        self.friction = PAIR_MU
        reset_home(self.model, self.data)
        set_finger_object_sliding_mu(self.model, self.ids, PAIR_MU, data=None)
        self.set_guide_weld(True)
        hold = np.array([0.50, -0.40, 0.55])
        apply_ball_free_state(self, hold, np.zeros(3))
        self.set_guide_pos(hold)
        mujoco.mj_forward(self.model, self.data)
        off = np.zeros(3) if grasp_offset is None else np.asarray(grasp_offset, dtype=float)
        self.fsm.reset(off)
        self.meter.reset()
        self.limiter.reset()
        self.captured = False
        restore_state(self, None)
        disarm(self)
        assert_noslip(self)


def make_ground_sim() -> tuple[GroundBallisticSim, dict]:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "simulation" / "nominal.yaml"))
    sim = GroundBallisticSim(cfg)
    sim.reset()
    assert_noslip(sim)
    return sim, cfg


def ground_world():
    """GROUND_SCENE plus nominal grasp and finger-pad records."""
    from controllers.cartesian.jacobian import gains_from_cfg
    from envs.simulation.object_dynamics import finger_pad_geom_ids
    from training.recovery_execution import capture_nominal, capture_pads

    sim, cfg = make_ground_sim()
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pad_ids, pad_mu = capture_pads(sim)
    pads_l, pads_r = finger_pad_geom_ids(sim.model, sim.ids)
    return sim, gains, nominal, pad_ids, pad_mu, pads_l, pads_r
