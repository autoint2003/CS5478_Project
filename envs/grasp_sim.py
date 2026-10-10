"""Shared Panda grasp simulation used by collection, env, and eval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from controllers.nominal import GraspFSM, object_pos
from controllers.residual import ResidualCommand, ResidualLimiter
from envs.config_util import ROOT
from envs.contact import touch_values
from envs.deterioration import DeteriorationMeter, object_dropped, unrecoverable
from envs.dynamics import set_object_dynamics
from envs.ids import resolve_ids
from envs.xml_build import write_panda_torque

SCENE = ROOT / "assets" / "scene.xml"


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def apply_solver_from_cfg(model: mujoco.MjModel, cfg: dict) -> None:
    """Apply frozen solver/contact options from merged sim config.

    Contact-model freeze: noslip_iterations / noslip_tolerance only.
    Does not write solref, solimp, mu, condim, integrator, or Newton iterations.
    """
    model.opt.timestep = float(cfg["physics_dt"])
    if "noslip_iterations" in cfg:
        model.opt.noslip_iterations = int(cfg["noslip_iterations"])
    if "noslip_tolerance" in cfg:
        model.opt.noslip_tolerance = float(cfg["noslip_tolerance"])


def load_model(cfg: dict) -> tuple[mujoco.MjModel, mujoco.MjData]:
    write_panda_torque()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    apply_solver_from_cfg(model, cfg)
    data = mujoco.MjData(model)
    return model, data


def mat6(r: np.ndarray) -> np.ndarray:
    return r.reshape(9)[:6].astype(np.float32)


class GraspSim:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model, self.data = load_model(cfg)
        self.ids = resolve_ids(self.model)
        self.fsm = GraspFSM(cfg)
        self.meter = DeteriorationMeter(cfg)
        self.limiter = ResidualLimiter(cfg)
        self.n_sub = int(cfg.get("n_substeps", 10))
        self.dt_policy = float(self.model.opt.timestep) * self.n_sub
        self.mass = float(cfg.get("mass", cfg.get("nominal_mass", 0.08)))
        self.friction = float(cfg.get("friction", cfg.get("nominal_friction", 1.0)))
        self.captured = False

    def reset(
        self,
        mass: float | None = None,
        friction: float | None = None,
        grasp_offset: np.ndarray | None = None,
    ) -> None:
        self.mass = float(self.cfg.get("mass", 0.08) if mass is None else mass)
        self.friction = float(
            self.cfg.get("friction", 1.0) if friction is None else friction
        )
        reset_home(self.model, self.data)
        set_object_dynamics(self.model, self.ids, self.mass, self.friction, self.data)
        mujoco.mj_forward(self.model, self.data)
        off = np.zeros(3) if grasp_offset is None else np.asarray(grasp_offset, dtype=float)
        self.fsm.reset(off)
        self.meter.reset()
        self.limiter.reset()
        self.captured = False

    def load_snapshot(self, snap: dict) -> None:
        self.reset(snap["mass"], snap["friction"], snap["grasp_offset"])
        self.data.qpos[:] = snap["qpos"]
        self.data.qvel[:] = snap["qvel"]
        self.data.ctrl[:] = snap["ctrl"]
        self.data.time = float(snap.get("time", 0.0))
        if "qacc_warmstart" in snap:
            self.data.qacc_warmstart[:] = np.asarray(snap["qacc_warmstart"], dtype=float)
        if len(snap.get("act", [])):
            self.data.act[:] = np.asarray(snap["act"], dtype=float)
        mujoco.mj_forward(self.model, self.data)
        # Old recovery buffers omit FSM clocks; default keeps prior lift+t_phase=0 behaviour.
        self.fsm.phase = str(snap.get("phase", "lift"))
        self.fsm.t_phase = float(snap.get("t_phase", 0.0))
        self.fsm.t_stable = float(snap.get("t_stable", 0.0))
        self.fsm.t_held = float(snap.get("t_held", 0.0))
        self.fsm.lift_started = bool(snap.get("lift_started", self.fsm.phase == "lift"))
        self.fsm.success = bool(snap.get("success", False))
        self.fsm.p_des = np.asarray(snap["p_des"], dtype=float).copy()
        self.fsm.r_des = np.asarray(snap["r_des"], dtype=float).reshape(3, 3).copy()
        self.fsm.grasp_xy = np.asarray(snap["grasp_xy"], dtype=float).copy()
        self.fsm.grasp_z = float(snap["grasp_z"])
        if "fg_cmd" in snap:
            self.fsm.fg_cmd = float(snap["fg_cmd"])
        if "v_cmd" in snap:
            self.fsm.v_cmd = np.asarray(snap["v_cmd"], dtype=float).reshape(3).copy()
        if "w_cmd" in snap:
            self.fsm.w_cmd = np.asarray(snap["w_cmd"], dtype=float).reshape(3).copy()
        self.meter.prel_ref = np.asarray(snap["prel_ref"], dtype=float).copy()
        self.meter.r_rel_ref = np.asarray(snap["r_rel_ref"], dtype=float).reshape(3, 3).copy()
        self.meter.prev_D = float(snap["D"])
        self.meter.in_recovery = bool(snap.get("in_recovery", True))
        self.meter.exit_count = int(snap.get("exit_count", 0))
        self.captured = bool(snap.get("captured", True))

    def snapshot(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
            "mass": self.mass,
            "friction": self.friction,
            "grasp_offset": self.fsm.grasp_offset.copy(),
            "p_des": self.fsm.p_des.copy(),
            "r_des": self.fsm.r_des.copy(),
            "grasp_xy": (
                self.fsm.grasp_xy.copy()
                if self.fsm.grasp_xy is not None
                else object_pos(self.data, self.ids)[:2]
            ),
            "grasp_z": float(self.fsm.grasp_z),
            "prel_ref": np.zeros(3) if self.meter.prel_ref is None else self.meter.prel_ref.copy(),
            "r_rel_ref": (
                np.eye(3) if self.meter.r_rel_ref is None else self.meter.r_rel_ref.copy()
            ),
            "D": float(self.meter.last.D),
            "object_z": float(self.data.xpos[self.ids.object_body][2]),
            "hand_z": float(self.data.xpos[self.ids.hand_body][2]),
            "phase": self.fsm.phase,
            "t_phase": float(self.fsm.t_phase),
            "t_stable": float(self.fsm.t_stable),
            "t_held": float(self.fsm.t_held),
            "lift_started": bool(self.fsm.lift_started),
            "success": bool(self.fsm.success),
            "fg_cmd": float(self.fsm.fg_cmd),
            "v_cmd": self.fsm.v_cmd.copy(),
            "w_cmd": self.fsm.w_cmd.copy(),
            "captured": bool(self.captured),
            "in_recovery": bool(self.meter.in_recovery),
            "exit_count": int(self.meter.exit_count),
            "qacc_warmstart": self.data.qacc_warmstart.copy(),
            "act": (
                np.array(self.data.act, float).copy()
                if int(getattr(self.model, "na", 0) or 0)
                else np.zeros(0)
            ),
            "tau_grip": float(self.data.ctrl[7]) if self.data.ctrl.size > 7 else 0.0,
        }

    def physics_step(self, residual: ResidualCommand | None = None, in_recovery: bool = False) -> None:
        mujoco.mj_forward(self.model, self.data)
        dv = None if residual is None else residual.dv
        dw = None if residual is None else residual.dw
        dfg = 0.0 if residual is None else residual.dfg
        self.fsm.step(
            self.model,
            self.data,
            self.ids,
            residual_dv=dv,
            residual_dw=dw,
            residual_dfg=dfg,
            in_recovery=in_recovery,
        )
        mujoco.mj_step(self.model, self.data)

    def maybe_capture_reference(self) -> None:
        if self.fsm.phase == "lift" and not self.captured:
            self.meter.capture_reference(self.model, self.data, self.ids)
            self.captured = True

    def policy_tick(self, residual: ResidualCommand | None, in_recovery: bool) -> None:
        self.maybe_capture_reference()
        for _ in range(self.n_sub):
            self.physics_step(residual, in_recovery=in_recovery)

    def recovery4d_tick(self, v_world: np.ndarray, w_world: np.ndarray, tau: float) -> None:
        """Absolute Cartesian command. No nominal lift, no clip_fg."""
        from controllers.jacobian_controller import apply_cartesian_ctrl, gains_from_cfg
        from controllers.nominal import _integrate_rot

        dt = float(self.model.opt.timestep)
        v = np.asarray(v_world, float).reshape(3)
        w = np.asarray(w_world, float).reshape(3)
        tau = float(np.clip(tau, self.ids.ctrl_low[7], self.ids.ctrl_high[7]))
        gains = gains_from_cfg(self.cfg)
        for _ in range(self.n_sub):
            self.fsm.p_des = self.fsm.p_des + v * dt
            self.fsm.r_des = _integrate_rot(self.fsm.r_des, w, dt)
            self.fsm.v_cmd = v.copy()
            self.fsm.w_cmd = w.copy()
            self.fsm.v_des = v.copy()
            self.fsm.w_des = w.copy()
            self.fsm.fg_cmd = float(-tau)
            apply_cartesian_ctrl(
                self.model,
                self.data,
                self.ids,
                self.fsm.p_des,
                self.fsm.r_des,
                gripper_tau=tau,
                v_des=v,
                w_des=w,
                **gains,
            )
            mujoco.mj_step(self.model, self.data)

    def observe(self, use_contact: bool = True) -> np.ndarray:
        d = self.data
        ids = self.ids
        q = d.qpos[ids.arm_jnt]
        qd = d.qvel[ids.arm_dof]
        r_ee = d.xmat[ids.hand_body].reshape(3, 3)
        r_obj = d.xmat[ids.object_body].reshape(3, 3)
        from envs.deterioration import body_twist

        v_ee, w_ee = body_twist(self.model, d, ids.hand_body)
        v_obj, w_obj = body_twist(self.model, d, ids.object_body)
        snap = self.meter.last
        touch = touch_values(d, ids)
        contact = np.array(
            [snap.f_left, snap.f_right, touch[0], touch[1]], dtype=np.float32
        )
        if not use_contact:
            contact = np.zeros_like(contact)
        unom = np.concatenate(
            [
                self.fsm.v_cmd,
                self.fsm.w_cmd,
                [self.fsm.fg_cmd],
            ]
        )
        obs = np.concatenate(
            [
                q,
                qd,
                d.xpos[ids.hand_body],
                mat6(r_ee),
                v_ee,
                w_ee,
                d.xpos[ids.object_body],
                mat6(r_obj),
                v_obj,
                w_obj,
                snap.p_rel,
                snap.v_rel,
                [snap.theta_rel],
                snap.omega_rel,
                contact,
                unom,
                [snap.D, snap.Ddot],
            ]
        ).astype(np.float32)
        return obs

    def dropped(self) -> bool:
        return object_dropped(
            self.data,
            self.ids,
            self.cfg,
            phase=self.fsm.phase,
            t_phase=self.fsm.t_phase,
        )

    def bad_state(self) -> bool:
        return unrecoverable(self.data, self.ids, self.cfg)

