"""Nominal ballistic scene. noslip=1. Cylinder 0.20 kg, diagnostic ball 0.05 kg."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from envs.config_util import load_yaml, merge_sim_config
from envs.dynamics import set_finger_object_sliding_mu
from envs.grasp_sim import GraspSim, apply_solver_from_cfg, reset_home
from envs.ids import resolve_ids
from envs.xml_build import write_panda_torque
from training.impact_ball import apply_ball_free_state
from training.write_ballistic_impact_scene import CYL_M, write_ballistic_scene

ROOT = Path(__file__).resolve().parents[1]
PAIR_MU = 1.0
HAND_RISE_M = 0.05
TAU_SEC = -18.0


def assert_noslip(sim) -> None:
    n = int(sim.model.opt.noslip_iterations)
    if n != 1:
        raise RuntimeError(f"noslip_iterations={n}, expected 1")


def geom_name(model, gid: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return str(n) if n else f"geom_{int(gid)}"


class BallisticSim(GraspSim):
    def __init__(self, cfg: dict):
        scene = write_ballistic_scene()
        write_panda_torque()
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        apply_solver_from_cfg(self.model, cfg)
        self.data = mujoco.MjData(self.model)
        self.ids = resolve_ids(self.model)
        from controllers.nominal import GraspFSM
        from controllers.residual import ResidualLimiter
        from envs.deterioration import DeteriorationMeter

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

    def reset(self, mass=None, friction=None, grasp_offset=None) -> None:
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
        assert_noslip(self)


def make_sim() -> tuple[BallisticSim, dict]:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = BallisticSim(cfg)
    sim.reset()
    assert_noslip(sim)
    return sim, cfg


def z_tgt_of(sim) -> float:
    return float(sim.fsm.grasp_z) + float(sim.cfg["fsm"]["lift_offset_z"])


def ball_force_on_ball(sim) -> tuple[np.ndarray, list]:
    model, data = sim.model, sim.data
    wr = np.zeros(6)
    F = np.zeros(3)
    rows = []
    bg = int(sim.ball_geom)
    og = int(sim.ids.object_geom)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if bg not in (g1, g2):
            continue
        mujoco.mj_contactForce(model, data, i, wr)
        n = np.array(c.frame[0:3], float)
        t1 = np.array(c.frame[3:6], float)
        t2 = np.array(c.frame[6:9], float)
        fn, ft1, ft2 = float(wr[0]), float(wr[1]), float(wr[2])
        f_on_geom2 = n * fn + t1 * ft1 + t2 * ft2
        if g2 == bg:
            f_ball = f_on_geom2
        else:
            f_ball = -f_on_geom2
        F += f_ball
        other = g2 if g1 == bg else g1
        kind = "ball-cylinder" if other == og else "ball-other"
        rows.append({"kind": kind, "pos": np.array(c.pos, float).copy(), "fn": fn})
    return F, rows
