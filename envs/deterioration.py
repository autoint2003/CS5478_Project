"""Ground-truth grasp deterioration score D_t (policy-rate)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from envs.contact import finger_object_normals, touch_values
from envs.ids import PandaIds


def body_twist(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
    omega = vel[:3].copy()
    lin = vel[3:].copy()
    return lin, omega


def relative_angle(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r_err = r_a @ r_b.T
    c = 0.5 * (np.trace(r_err) - 1.0)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


@dataclass
class DeteriorationSnapshot:
    D: float
    Ddot: float
    p_rel: np.ndarray
    v_rel: np.ndarray
    theta_rel: float
    omega_rel: np.ndarray
    C: float
    f_left: float
    f_right: float
    lost_contact: bool


class DeteriorationMeter:
    def __init__(self, cfg: dict):
        d = cfg.get("deterioration", {})
        self.cfg = d
        self.p_scale = float(d.get("p_scale", 0.02))
        self.v_scale = float(d.get("v_scale", 0.08))
        self.theta_scale = float(d.get("theta_scale", 0.25))
        self.omega_scale = float(d.get("omega_scale", 1.0))
        self.w_p = float(d.get("w_p", 1.0))
        self.w_v = float(d.get("w_v", 1.0))
        self.w_theta = float(d.get("w_theta", 1.0))
        self.w_omega = float(d.get("w_omega", 0.5))
        self.w_c = float(d.get("w_c", 1.0))
        self.eps = float(d.get("contact_eps", 0.05))
        self.lost_force = float(d.get("lost_contact_force", 0.5))
        self.D_enter = float(d.get("D_enter", 0.85))
        self.D_exit = float(d.get("D_exit", 0.40))
        self.N_stable = int(d.get("N_stable", 10))
        self.prel_ref: np.ndarray | None = None
        self.r_rel_ref: np.ndarray | None = None
        self.prev_D: float | None = None
        self.exit_count = 0
        self.in_recovery = False
        self.last = DeteriorationSnapshot(
            D=0.0,
            Ddot=0.0,
            p_rel=np.zeros(3),
            v_rel=np.zeros(3),
            theta_rel=0.0,
            omega_rel=np.zeros(3),
            C=0.0,
            f_left=0.0,
            f_right=0.0,
            lost_contact=False,
        )

    def reset(self) -> None:
        self.prel_ref = None
        self.r_rel_ref = None
        self.prev_D = None
        self.exit_count = 0
        self.in_recovery = False

    def capture_reference(self, model, data, ids: PandaIds) -> None:
        p_obj = data.xpos[ids.object_body]
        p_g = data.xpos[ids.hand_body]
        r_obj = data.xmat[ids.object_body].reshape(3, 3)
        r_g = data.xmat[ids.hand_body].reshape(3, 3)
        self.prel_ref = (p_obj - p_g).copy()
        self.r_rel_ref = r_g.T @ r_obj

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ids: PandaIds,
        dt_policy: float,
    ) -> DeteriorationSnapshot:
        p_obj = data.xpos[ids.object_body].copy()
        p_g = data.xpos[ids.hand_body].copy()
        r_obj = data.xmat[ids.object_body].reshape(3, 3).copy()
        r_g = data.xmat[ids.hand_body].reshape(3, 3).copy()
        v_obj, w_obj = body_twist(model, data, ids.object_body)
        v_g, w_g = body_twist(model, data, ids.hand_body)
        p_rel = p_obj - p_g
        v_rel = v_obj - v_g
        if self.prel_ref is None:
            dp = np.zeros(3)
            theta = 0.0
        else:
            dp = p_rel - self.prel_ref
            r_rel = r_g.T @ r_obj
            theta = relative_angle(r_rel, self.r_rel_ref)
        omega_rel = w_obj - w_g
        forces = finger_object_normals(model, data, ids)
        f_l, f_r = float(forces[0]), float(forces[1])
        lost = (f_l < self.lost_force) or (f_r < self.lost_force)
        c_t = abs((f_l - f_r) / (f_l + f_r + self.eps)) + float(lost)
        D = (
            self.w_p * float(np.linalg.norm(dp)) / self.p_scale
            + self.w_v * float(np.linalg.norm(v_rel)) / self.v_scale
            + self.w_theta * abs(theta) / self.theta_scale
            + self.w_omega * float(np.linalg.norm(omega_rel)) / self.omega_scale
            + self.w_c * c_t
        )
        if self.prev_D is None or dt_policy <= 0:
            ddot = 0.0
        else:
            ddot = (D - self.prev_D) / dt_policy
        self.prev_D = D
        snap = DeteriorationSnapshot(
            D=float(D),
            Ddot=float(ddot),
            p_rel=p_rel,
            v_rel=v_rel,
            theta_rel=float(theta),
            omega_rel=omega_rel,
            C=float(c_t),
            f_left=f_l,
            f_right=f_r,
            lost_contact=bool(lost),
        )
        self.last = snap
        return snap

    def update_mode(self, D: float) -> bool:
        if not self.in_recovery:
            if D > self.D_enter:
                self.in_recovery = True
                self.exit_count = 0
        else:
            if D < self.D_exit:
                self.exit_count += 1
                if self.exit_count >= self.N_stable:
                    self.in_recovery = False
                    self.exit_count = 0
            else:
                self.exit_count = 0
        return self.in_recovery


def object_dropped(
    data: mujoco.MjData,
    ids: PandaIds,
    cfg: dict,
    phase: str = "lift",
    t_phase: float = 1.0,
) -> bool:
    if phase != "lift":
        return False
    d = cfg.get("deterioration", {})
    table = float(cfg.get("table_top", 0.40))
    half = float(cfg.get("object_half_height", cfg.get("cube_half_height", 0.03)))
    z = float(data.xpos[ids.object_body][2])
    p_obj = data.xpos[ids.object_body][:2]
    p_g = data.xpos[ids.hand_body][:2]
    xy_lim = float(d.get("workspace_xy", 0.18))
    if float(np.linalg.norm(p_obj - p_g)) > xy_lim:
        return True
    if t_phase > 0.4 and z < table + half + 0.01:
        return True
    return False


def unrecoverable(data: mujoco.MjData, ids: PandaIds, cfg: dict) -> bool:
    xy = data.xpos[ids.object_body][:2]
    home_xy = np.array(cfg.get("object_xy", [0.5, 0.0]), dtype=float)
    return float(np.linalg.norm(xy - home_xy)) > 0.25
