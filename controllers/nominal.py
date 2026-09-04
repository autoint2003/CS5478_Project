"""Nominal grasp controller: approach → descend → close → lift."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from controllers.gripper_controller import (
    finger_opening,
    gripper_torque,
    read_touch,
)
from controllers.jacobian_controller import (
    apply_cartesian_ctrl,
    gains_from_cfg,
    ori_error_deg,
)
from envs.ids import PandaIds

PHASES = ("approach", "descend", "close", "lift")


def grasp_orientation() -> np.ndarray:
    """Top-down pinch: hand Z world-down, fingers opening along world X."""
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )


def cube_pos(data: mujoco.MjData, ids: PandaIds) -> np.ndarray:
    return data.xpos[ids.cube_body].copy()


@dataclass
class GraspFSM:
    cfg: dict
    phase: str = "approach"
    t_phase: float = 0.0
    t_stable: float = 0.0
    t_held: float = 0.0
    grasp_xy: np.ndarray | None = None
    grasp_z: float = 0.0
    r_des: np.ndarray = field(default_factory=grasp_orientation)
    success: bool = False

    def reset(self) -> None:
        self.phase = "approach"
        self.t_phase = 0.0
        self.t_stable = 0.0
        self.t_held = 0.0
        self.grasp_xy = None
        self.grasp_z = 0.0
        self.r_des = grasp_orientation()
        self.success = False

    def phase_onehot(self) -> np.ndarray:
        z = np.zeros(len(PHASES))
        z[PHASES.index(self.phase)] = 1.0
        return z

    def _advance(self, nxt: str) -> None:
        self.phase = nxt
        self.t_phase = 0.0
        self.t_stable = 0.0

    def _target(self, cube: np.ndarray) -> np.ndarray:
        fsm = self.cfg["fsm"]
        if self.phase == "approach":
            return np.array(
                [cube[0], cube[1], cube[2] + float(fsm["approach_offset_z"])]
            )
        if self.phase == "descend":
            return np.array(
                [cube[0], cube[1], cube[2] + float(fsm["grasp_offset_z"])]
            )
        xy = self.grasp_xy if self.grasp_xy is not None else cube[:2]
        if self.phase == "close":
            return np.array([xy[0], xy[1], self.grasp_z])
        return np.array(
            [xy[0], xy[1], self.grasp_z + float(fsm["lift_offset_z"])]
        )

    def step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ids: PandaIds,
    ) -> np.ndarray:
        dt = float(model.opt.timestep)
        fsm = self.cfg["fsm"]
        cube = cube_pos(data, ids)
        p_des = self._target(cube)

        touch = read_touch(data, ids)
        g_tau = gripper_torque(self.phase, touch, self.cfg)
        ctrl = apply_cartesian_ctrl(
            model,
            data,
            ids,
            p_des,
            self.r_des,
            gripper_tau=g_tau,
            **gains_from_cfg(self.cfg),
        )

        p = data.xpos[ids.hand_body]
        r = data.xmat[ids.hand_body].reshape(3, 3)
        pos_err = float(np.linalg.norm(p_des - p))
        ori_err = ori_error_deg(self.r_des, r)
        pos_tol = float(fsm["pos_tol"])
        ori_tol = float(fsm["ori_tol_deg"])
        self.t_phase += dt

        if self.phase == "approach":
            ok = pos_err < pos_tol and ori_err < ori_tol
            self.t_stable = self.t_stable + dt if ok else 0.0
            if self.t_stable >= float(fsm["dwell_approach"]):
                self._advance("descend")
        elif self.phase == "descend":
            ok = pos_err < pos_tol and ori_err < ori_tol
            self.t_stable = self.t_stable + dt if ok else 0.0
            if self.t_stable >= float(fsm["dwell_descend"]):
                self.grasp_xy = cube[:2].copy()
                self.grasp_z = float(p_des[2])
                self._advance("close")
        elif self.phase == "close":
            opened = finger_opening(data, ids)
            contacted = float(np.min(touch)) > float(
                self.cfg["gripper"]["touch_threshold"]
            )
            pinched = opened < float(fsm["pinch_opening"])
            ok = contacted or pinched
            self.t_stable = self.t_stable + dt if ok else 0.0
            if (
                self.t_stable >= float(fsm["dwell_close"])
                or self.t_phase >= float(fsm["close_timeout"])
            ):
                self._advance("lift")
        else:
            table_top = float(self.cfg.get("table_top", 0.40))
            half = float(self.cfg.get("cube_half_height", 0.03))
            z_ok = cube[2] > table_top + half + float(fsm["lift_clearance"])
            near = float(np.linalg.norm(cube[:2] - p[:2])) < float(fsm["hold_xy_tol"])
            if z_ok and near:
                self.t_held += dt
            else:
                self.t_held = 0.0
            if self.t_held >= float(fsm["hold_success"]):
                self.success = True

        return ctrl
