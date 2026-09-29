"""6D Cartesian PD in torque space: tau = J^T (Kp e - Kd xdot) + b(q, qdot)."""

from __future__ import annotations

import numpy as np
import mujoco

from controllers.bias import arm_bias
from envs.ids import PandaIds

_DEFAULTS = dict(
    kp_pos=180.0,
    kd_pos=28.0,
    kp_ori=16.0,
    kd_ori=2.4,
    kp_null=4.0,
    kd_null=0.8,
)


def rotation_error(r_des: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
    """World-frame orientation error 0.5 vee(R* R^T - R R*^T)."""
    r_err = r_des @ r_cur.T
    return 0.5 * np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ]
    )


def ori_error_deg(r_des: np.ndarray, r_cur: np.ndarray) -> float:
    r_err = r_des @ r_cur.T
    c = 0.5 * (np.trace(r_err) - 1.0)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def jacobian_6d(
    model: mujoco.MjModel, data: mujoco.MjData, ids: PandaIds
) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ids.hand_body)
    return np.vstack([jacp[:, ids.arm_dof], jacr[:, ids.arm_dof]])


def cartesian_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PandaIds,
    p_des: np.ndarray,
    r_des: np.ndarray,
    kp_pos: float = _DEFAULTS["kp_pos"],
    kd_pos: float = _DEFAULTS["kd_pos"],
    kp_ori: float = _DEFAULTS["kp_ori"],
    kd_ori: float = _DEFAULTS["kd_ori"],
    kp_null: float = _DEFAULTS["kp_null"],
    kd_null: float = _DEFAULTS["kd_null"],
    q_home: np.ndarray | None = None,
    v_des: np.ndarray | None = None,
    w_des: np.ndarray | None = None,
) -> np.ndarray:
    p = data.xpos[ids.hand_body].copy()
    r = data.xmat[ids.hand_body].reshape(3, 3).copy()
    e_p = p_des - p
    e_o = rotation_error(r_des, r)
    e = np.concatenate([e_p, e_o])
    j = jacobian_6d(model, data, ids)
    qdot = data.qvel[ids.arm_dof]
    xdot = j @ qdot
    vd = np.zeros(3) if v_des is None else np.asarray(v_des, dtype=float).reshape(3)
    wd = np.zeros(3) if w_des is None else np.asarray(w_des, dtype=float).reshape(3)
    xdot_des = np.concatenate([vd, wd])
    kp = np.array([kp_pos, kp_pos, kp_pos, kp_ori, kp_ori, kp_ori])
    kd = np.array([kd_pos, kd_pos, kd_pos, kd_ori, kd_ori, kd_ori])
    wrench = kp * e + kd * (xdot_des - xdot)
    tau = j.T @ wrench + arm_bias(model, data, ids)
    q_ref = ids.home_qpos[ids.arm_jnt] if q_home is None else q_home
    lam = 1e-3
    j_pinv = j.T @ np.linalg.inv(j @ j.T + lam * np.eye(6))
    nproj = np.eye(7) - j_pinv @ j
    tau = tau + nproj @ (kp_null * (q_ref - data.qpos[ids.arm_jnt]) - kd_null * qdot)
    return np.clip(tau, ids.ctrl_low[:7], ids.ctrl_high[:7])


def apply_cartesian_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PandaIds,
    p_des: np.ndarray,
    r_des: np.ndarray,
    gripper_tau: float = 0.0,
    v_des: np.ndarray | None = None,
    w_des: np.ndarray | None = None,
    **gains,
) -> np.ndarray:
    """Write 8-D ctrl: Cartesian arm torque + gripper tendon force."""
    ctrl = np.zeros(ids.n_act)
    ctrl[:7] = cartesian_torque(
        model, data, ids, p_des, r_des, v_des=v_des, w_des=w_des, **gains
    )
    ctrl[7] = float(np.clip(gripper_tau, ids.ctrl_low[7], ids.ctrl_high[7]))
    data.ctrl[:] = ctrl
    return ctrl


def gains_from_cfg(cfg: dict) -> dict:
    block = cfg.get("cartesian", {})
    keys = ("kp_pos", "kd_pos", "kp_ori", "kd_ori", "kp_null", "kd_null")
    return {k: float(block[k]) for k in keys if k in block}
