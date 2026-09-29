"""Nominal grasp controller: approach → descend → close → lift (+ recovery overlay)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from controllers.gripper_controller import (
    clip_fg,
    fg_to_tau,
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


def object_pos(data: mujoco.MjData, ids: PandaIds) -> np.ndarray:
    return data.xpos[ids.object_body].copy()


def cube_pos(data: mujoco.MjData, ids: PandaIds) -> np.ndarray:
    return object_pos(data, ids)


def _integrate_rot(r: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    th = float(np.linalg.norm(w) * dt)
    if th < 1e-12:
        return r
    k = w / (np.linalg.norm(w) + 1e-12)
    kx, ky, kz = k
    k_hat = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    d_r = np.eye(3) + np.sin(th) * k_hat + (1.0 - np.cos(th)) * (k_hat @ k_hat)
    out = d_r @ r
    u, _, vh = np.linalg.svd(out)
    return u @ vh


@dataclass
class GraspFSM:
    cfg: dict
    phase: str = "approach"
    t_phase: float = 0.0
    t_stable: float = 0.0
    t_held: float = 0.0
    grasp_xy: np.ndarray | None = None
    grasp_z: float = 0.0
    grasp_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    r_des: np.ndarray = field(default_factory=grasp_orientation)
    p_des: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_des: np.ndarray = field(default_factory=lambda: np.zeros(3))
    w_des: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fg_cmd: float = 0.0
    v_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    w_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    success: bool = False
    lift_started: bool = False

    def reset(self, grasp_offset: np.ndarray | None = None) -> None:
        self.phase = "approach"
        self.t_phase = 0.0
        self.t_stable = 0.0
        self.t_held = 0.0
        self.grasp_xy = None
        self.grasp_z = 0.0
        self.grasp_offset = (
            np.zeros(3)
            if grasp_offset is None
            else np.asarray(grasp_offset, dtype=float).reshape(3)
        )
        self.r_des = grasp_orientation()
        self.p_des = np.zeros(3)
        self.v_des = np.zeros(3)
        self.w_des = np.zeros(3)
        self.fg_cmd = 0.0
        self.v_cmd = np.zeros(3)
        self.w_cmd = np.zeros(3)
        self.success = False
        self.lift_started = False

    def phase_onehot(self) -> np.ndarray:
        z = np.zeros(len(PHASES))
        z[PHASES.index(self.phase if self.phase in PHASES else "lift")] = 1.0
        return z

    def _advance(self, nxt: str) -> None:
        self.phase = nxt
        self.t_phase = 0.0
        self.t_stable = 0.0

    def _target(self, obj: np.ndarray) -> np.ndarray:
        fsm = self.cfg["fsm"]
        off = self.grasp_offset
        if self.phase == "approach":
            return np.array(
                [obj[0], obj[1], obj[2] + float(fsm["approach_offset_z"])]
            )
        if self.phase == "descend":
            return np.array(
                [
                    obj[0] + off[0],
                    obj[1] + off[1],
                    obj[2] + float(fsm["grasp_offset_z"]) + off[2],
                ]
            )
        xy = self.grasp_xy if self.grasp_xy is not None else obj[:2] + off[:2]
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
        residual_dv: np.ndarray | None = None,
        residual_dw: np.ndarray | None = None,
        residual_dfg: float = 0.0,
        in_recovery: bool = False,
    ) -> np.ndarray:
        dt = float(model.opt.timestep)
        fsm = self.cfg["fsm"]
        obj = object_pos(data, ids)
        p_hand = data.xpos[ids.hand_body].copy()
        touch = read_touch(data, ids)
        dv = np.zeros(3) if residual_dv is None else np.asarray(residual_dv, dtype=float)
        dw = np.zeros(3) if residual_dw is None else np.asarray(residual_dw, dtype=float)

        if self.phase in ("approach", "descend", "close"):
            self.p_des = self._target(obj)
            self.v_des = np.zeros(3)
            self.w_des = np.zeros(3)
            self.v_cmd = np.zeros(3)
            self.w_cmd = np.zeros(3)
            g_tau = gripper_torque(self.phase, touch, self.cfg)
            self.fg_cmd = abs(float(g_tau))
        else:
            if not self.lift_started:
                self.p_des = p_hand.copy()
                self.lift_started = True
            v_nom = np.array([0.0, 0.0, float(fsm.get("lift_speed", 0.08))])
            w_nom = np.zeros(3)
            if in_recovery:
                self.v_cmd = v_nom + dv
                self.w_cmd = w_nom + dw
            else:
                self.v_cmd = v_nom
                self.w_cmd = w_nom
            self.v_des = self.v_cmd.copy()
            self.w_des = self.w_cmd.copy()
            self.p_des = self.p_des + self.v_cmd * dt
            self.r_des = _integrate_rot(self.r_des, self.w_cmd, dt)
            fg_nom = float(self.cfg["gripper"].get("fg_nom", 18.0))
            if in_recovery:
                self.fg_cmd = clip_fg(fg_nom + float(residual_dfg), self.cfg)
            else:
                self.fg_cmd = clip_fg(fg_nom, self.cfg)
            g_tau = fg_to_tau(self.fg_cmd, ids)

        ctrl = apply_cartesian_ctrl(
            model,
            data,
            ids,
            self.p_des,
            self.r_des,
            gripper_tau=g_tau,
            v_des=self.v_des,
            w_des=self.w_des,
            **gains_from_cfg(self.cfg),
        )

        p = data.xpos[ids.hand_body]
        r = data.xmat[ids.hand_body].reshape(3, 3)
        pos_err = float(np.linalg.norm(self.p_des - p))
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
                self.grasp_xy = self.p_des[:2].copy()
                self.grasp_z = float(self.p_des[2])
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
            half = float(
                self.cfg.get("object_half_height", self.cfg.get("cube_half_height", 0.03))
            )
            z_ok = obj[2] > table_top + half + float(fsm["lift_clearance"])
            near = float(np.linalg.norm(obj[:2] - p[:2])) < float(fsm["hold_xy_tol"])
            if z_ok and near:
                self.t_held += dt
            else:
                self.t_held = 0.0
            if self.t_held >= float(fsm["hold_success"]):
                self.success = True

        return ctrl
