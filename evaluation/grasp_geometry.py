"""Object-to-gripper grasp configuration metrics (migration diagnostics).

Does NOT alter SUCCESS_12S, reward_v2, or the nominal evaluator.
Cylinder body-local +Z is the geometric/inertial symmetry axis
(see ``envs/simulation/ballistic_scene.py``: cylinder geom + Ixx=Iyy≠Izz).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# MuJoCo cylinder geom axis = body local +Z.
CYL_AXIS_BODY = np.array([0.0, 0.0, 1.0], dtype=float)

# Provisional defaults; overwrite via calibrate_from_samples().
DEFAULT_U_TARGET_H = np.array([0.0, 0.0, 1.0], dtype=float)  # filled by calibration
DEFAULT_P_TARGET_H = np.array([0.0, 0.0, 0.099], dtype=float)  # RH_Z_NOM in physical_recovery
AXIS_REVERSAL_SYMMETRIC = True  # verified: uniform cylinder + Ixx=Iyy


def hand_R(sim) -> np.ndarray:
    return np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)


def object_R(sim) -> np.ndarray:
    return np.asarray(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)


def cylinder_axis_world(sim) -> np.ndarray:
    Ro = object_R(sim)
    u = Ro @ CYL_AXIS_BODY
    n = float(np.linalg.norm(u))
    return u / max(n, 1e-12)


def cylinder_axis_hand(sim) -> np.ndarray:
    Rh = hand_R(sim)
    return Rh.T @ cylinder_axis_world(sim)


def p_com_hand(sim) -> np.ndarray:
    Rh = hand_R(sim)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    return Rh.T @ (po - ph)


def orientation_error_rad(u_c_h: np.ndarray, u_target_h: np.ndarray, axis_symmetric: bool = True) -> float:
    a = np.asarray(u_c_h, float).reshape(3)
    b = np.asarray(u_target_h, float).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    d = float(np.dot(a, b))
    if axis_symmetric:
        d = abs(d)
    return float(math.acos(float(np.clip(d, 0.0 if axis_symmetric else -1.0, 1.0))))


def position_error_m(p_c_h: np.ndarray, p_target_h: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(p_c_h, float) - np.asarray(p_target_h, float)))


def measure(sim, u_target_h: np.ndarray, p_target_h: np.ndarray) -> dict[str, Any]:
    """Diagnostic snapshot of object–gripper configuration (hand frame)."""
    from envs.simulation.grasp_quality import body_twist

    Rh = hand_R(sim)
    Ro = object_R(sim)
    u_c_w = cylinder_axis_world(sim)
    u_c_h = Rh.T @ u_c_w
    p_c_h = p_com_hand(sim)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    e_theta = orientation_error_rad(u_c_h, u_target_h, AXIS_REVERSAL_SYMMETRIC)
    e_pos = position_error_m(p_c_h, p_target_h)
    # Contact counts (finger–object)
    obj = int(sim.ids.object_body)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    n_l = n_r = 0
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        if other == left:
            n_l += 1
        elif other == right:
            n_r += 1
    return {
        "u_c_w": u_c_w.tolist(),
        "u_c_h": u_c_h.tolist(),
        "u_target_h": np.asarray(u_target_h, float).tolist(),
        "p_c_h": p_c_h.tolist(),
        "p_target_h": np.asarray(p_target_h, float).tolist(),
        "e_theta_rad": e_theta,
        "e_theta_deg": float(np.degrees(e_theta)),
        "e_pos_m": e_pos,
        "e_pos_mm": e_pos * 1e3,
        "w_rel_h": (Rh.T @ (wo - wh)).tolist(),
        "v_rel_h": (Rh.T @ (vo - vh)).tolist(),
        "nL": n_l,
        "nR": n_r,
        "bilateral": bool(n_l > 0 and n_r > 0),
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "R_rel_rotvec_norm": float(
            np.linalg.norm(_rotvec(Rh.T @ Ro))
        ),
    }


def _rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, float).reshape(3, 3)
    c = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float)
    s = 2.0 * np.sin(ang)
    if abs(s) < 1e-8:
        return np.zeros(3)
    return axis * (ang / s)


def calibrate_from_samples(samples_u: list[np.ndarray], samples_p: list[np.ndarray]) -> dict:
    """Mean target + empirical percentiles for tolerance proposal."""
    U = np.stack([np.asarray(u, float) / max(float(np.linalg.norm(u)), 1e-12) for u in samples_u], axis=0)
    # Axis-symmetric: flip samples so they agree with first
    ref = U[0]
    for i in range(len(U)):
        if float(np.dot(U[i], ref)) < 0:
            U[i] = -U[i]
    u_mean = U.mean(axis=0)
    u_mean = u_mean / max(float(np.linalg.norm(u_mean)), 1e-12)
    P = np.stack([np.asarray(p, float) for p in samples_p], axis=0)
    p_mean = P.mean(axis=0)
    # Angular deviations from mean target
    e_th = np.array([orientation_error_rad(u, u_mean, True) for u in U])
    e_p = np.array([position_error_m(p, p_mean) for p in P])
    return {
        "u_target_h": u_mean,
        "p_target_h": p_mean,
        "n_samples": len(samples_u),
        "e_theta_deg_p50": float(np.degrees(np.percentile(e_th, 50))),
        "e_theta_deg_p90": float(np.degrees(np.percentile(e_th, 90))),
        "e_theta_deg_p95": float(np.degrees(np.percentile(e_th, 95))),
        "e_theta_deg_max": float(np.degrees(e_th.max())),
        "e_pos_mm_p50": float(1e3 * np.percentile(e_p, 50)),
        "e_pos_mm_p90": float(1e3 * np.percentile(e_p, 90)),
        "e_pos_mm_p95": float(1e3 * np.percentile(e_p, 95)),
        "e_pos_mm_max": float(1e3 * e_p.max()),
        # Proposed tolerances: cover ~95% of stable nominal samples with margin
        "tol_theta_rad_proposed": float(np.percentile(e_th, 95) * 1.25 + 1e-4),
        "tol_pos_m_proposed": float(np.percentile(e_p, 95) * 1.25 + 1e-5),
    }


def pose_correct(e_theta_rad: float, tol_rad: float) -> bool:
    return float(e_theta_rad) <= float(tol_rad)


def position_correct(e_pos_m: float, tol_m: float) -> bool:
    return float(e_pos_m) <= float(tol_m)


def success_new_components(
    success_12s: bool,
    e_theta_rad: float,
    e_pos_m: float,
    tol_theta: float,
    tol_pos: float,
    pose_ok_duration_s: float = 0.0,
    pos_ok_duration_s: float = 0.0,
    hold_required_s: float = 0.5,
) -> dict:
    """Candidate SUCCESS_NEW without mutating the old evaluator."""
    pose_ok = pose_correct(e_theta_rad, tol_theta)
    pos_ok = position_correct(e_pos_m, tol_pos)
    pose_held = pose_ok_duration_s >= hold_required_s
    pos_held = pos_ok_duration_s >= hold_required_s
    return {
        "SUCCESS_12S": bool(success_12s),
        "POSE_CORRECT": bool(pose_ok and pose_held),
        "POSITION_CORRECT": bool(pos_ok and pos_held),
        "SUCCESS_NEW": bool(success_12s and pose_ok and pos_ok and pose_held and pos_held),
        "e_theta_rad": float(e_theta_rad),
        "e_pos_m": float(e_pos_m),
        "pose_ok_duration_s": float(pose_ok_duration_s),
        "pos_ok_duration_s": float(pos_ok_duration_s),
    }
