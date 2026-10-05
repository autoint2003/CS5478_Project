"""Finger-local spatial tactile abstraction (SIMULATED, hardware-unvalidated).

Emulates what a finger-pad CoP / contact-location sensor could output.
Raw MuJoCo contact.pos is used ONLY as an emulation source.

This module must not read object pose, object velocity, or e_x.
It must not expose world-frame contact coordinates as the public reading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import mujoco

from envs.ids import PandaIds

# Fingertip pad 1 from assets/panda_torque.xml class fingertip_pad_collision_1:
#   geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445"
# VERIFIED FROM CODE (XML), not a hardware spec.
PAD1_POS_FINGER = np.array([0.0, 0.0055, 0.0445], dtype=float)
PAD1_HALF = np.array([0.0085, 0.004, 0.0085], dtype=float)
# Inner face (toward grasp midline in default finger +Y): center_y - half_y
PAD_INNER_Y = float(PAD1_POS_FINGER[1] - PAD1_HALF[1])  # 0.0015 m

# Numerical dust; not a hardware force rating.
MIN_COP_FORCE_N = 1e-2

SENSOR_KEYS = (
    "contact_present_L",
    "contact_present_R",
    "valid_L",
    "valid_R",
    "u_L",
    "v_L",
    "u_R",
    "v_R",
    "fn_L",
    "fn_R",
    "n_L",
    "n_R",
)


@dataclass(frozen=True)
class FingerTactile:
    """Pad-frame reading. u,v are NaN when valid is False (zero is a legal CoP)."""

    contact_present: bool
    valid: bool
    u: float
    v: float
    fn: float
    n_contacts: int


@dataclass(frozen=True)
class SpatialTactileReading:
    left: FingerTactile
    right: FingerTactile

    def to_array(self) -> dict:
        d = {
            "contact_present_L": float(self.left.contact_present),
            "contact_present_R": float(self.right.contact_present),
            "valid_L": float(self.left.valid),
            "valid_R": float(self.right.valid),
            "u_L": float(self.left.u),
            "v_L": float(self.left.v),
            "u_R": float(self.right.u),
            "v_R": float(self.right.v),
            "fn_L": float(self.left.fn),
            "fn_R": float(self.right.fn),
            "n_L": float(self.left.n_contacts),
            "n_R": float(self.right.n_contacts),
        }
        leak = set(d) - set(SENSOR_KEYS)
        if leak:
            raise RuntimeError(f"sensor key leak: {leak}")
        return d


def _body_R_p(data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    R = np.array(data.xmat[int(body_id)].reshape(3, 3), dtype=float)
    p = np.array(data.xpos[int(body_id)], dtype=float)
    return R, p


def world_to_finger_local(
    p_world: np.ndarray, R_finger: np.ndarray, p_finger: np.ndarray
) -> np.ndarray:
    return R_finger.T @ (np.asarray(p_world, float) - p_finger)


def finger_local_to_world(
    p_local: np.ndarray, R_finger: np.ndarray, p_finger: np.ndarray
) -> np.ndarray:
    return p_finger + R_finger @ np.asarray(p_local, float)


def pad_uv_from_finger_local(p_finger: np.ndarray) -> tuple[float, float]:
    """Surface coords on pad 1, origin at pad box center.

    u: finger-local X (pad width). Left finger X aligns with hand X (identity
    child of hand). Right finger XML quat wxyz=(0,0,0,1) is 180 deg about Z,
    so right-finger X = -hand X. Do NOT assume u == e_x without that sign.
    v: finger-local Z minus pad center Z (pad height).
    """
    u = float(p_finger[0] - PAD1_POS_FINGER[0])
    v = float(p_finger[2] - PAD1_POS_FINGER[2])
    return u, v


def _empty_finger() -> FingerTactile:
    return FingerTactile(
        contact_present=False,
        valid=False,
        u=float("nan"),
        v=float("nan"),
        fn=0.0,
        n_contacts=0,
    )


def _reduce_finger(points: list[np.ndarray], forces: list[float]) -> FingerTactile:
    if not points:
        return _empty_finger()
    w = np.array(forces, dtype=float)
    wsum = float(np.sum(w))
    n = len(points)
    present = n > 0
    if wsum < MIN_COP_FORCE_N:
        # Contact geom exists but force too small to define CoP.
        return FingerTactile(
            contact_present=present,
            valid=False,
            u=float("nan"),
            v=float("nan"),
            fn=wsum,
            n_contacts=n,
        )
    P = np.stack(points, axis=0)
    cop = (w[:, None] * P).sum(axis=0) / wsum
    u, v = pad_uv_from_finger_local(cop)
    return FingerTactile(
        contact_present=True,
        valid=True,
        u=u,
        v=v,
        fn=wsum,
        n_contacts=n,
    )


def measure_spatial_tactile(
    model: mujoco.MjModel, data: mujoco.MjData, ids: PandaIds
) -> SpatialTactileReading:
    """Public sensor API. Arguments: MuJoCo model/data + robot/object *IDs only*.

    Object body/geom IDs are used solely to filter finger–object contacts.
    Object qpos/qvel/xpos are not read.
    """
    left_pts: list[np.ndarray] = []
    left_fn: list[float] = []
    right_pts: list[np.ndarray] = []
    right_fn: list[float] = []
    RL, pL = _body_R_p(data, ids.left_body)
    RR, pR = _body_R_p(data, ids.right_body)
    wr = np.zeros(6)
    obj_b = int(ids.object_body)
    fingers = {int(ids.left_body), int(ids.right_body)}
    for i in range(int(data.ncon)):
        con = data.contact[i]
        b1 = int(model.geom_bodyid[int(con.geom1)])
        b2 = int(model.geom_bodyid[int(con.geom2)])
        bodies = {b1, b2}
        if obj_b not in bodies:
            continue
        fb = b1 if b1 in fingers else (b2 if b2 in fingers else None)
        if fb is None:
            continue
        mujoco.mj_contactForce(model, data, i, wr)
        fn = abs(float(wr[0]))
        p_w = np.array(con.pos, dtype=float)
        if fb == int(ids.left_body):
            left_pts.append(world_to_finger_local(p_w, RL, pL))
            left_fn.append(fn)
        else:
            right_pts.append(world_to_finger_local(p_w, RR, pR))
            right_fn.append(fn)
    return SpatialTactileReading(
        left=_reduce_finger(left_pts, left_fn),
        right=_reduce_finger(right_pts, right_fn),
    )


def reconstruct_world_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, ids: PandaIds
) -> dict:
    """DIAGNOSTIC: round-trip contact.pos through finger-local. Not a sensor output."""
    RL, pL = _body_R_p(data, ids.left_body)
    RR, pR = _body_R_p(data, ids.right_body)
    err = []
    n = 0
    wr = np.zeros(6)
    obj_b = int(ids.object_body)
    fingers = {int(ids.left_body), int(ids.right_body)}
    for i in range(int(data.ncon)):
        con = data.contact[i]
        b1 = int(model.geom_bodyid[int(con.geom1)])
        b2 = int(model.geom_bodyid[int(con.geom2)])
        bodies = {b1, b2}
        if obj_b not in bodies:
            continue
        fb = b1 if b1 in fingers else (b2 if b2 in fingers else None)
        if fb is None:
            continue
        mujoco.mj_contactForce(model, data, i, wr)
        p_w = np.array(con.pos, dtype=float)
        R, p = (RL, pL) if fb == int(ids.left_body) else (RR, pR)
        local = world_to_finger_local(p_w, R, p)
        recon = finger_local_to_world(local, R, p)
        err.append(float(np.linalg.norm(recon - p_w)))
        n += 1
    return {
        "n_finger_object_contacts": n,
        "max_recon_err_m": float(max(err) if err else 0.0),
        "mean_recon_err_m": float(np.mean(err) if err else 0.0),
        "pad_inner_y_m": PAD_INNER_Y,
        "pad1_pos_finger": PAD1_POS_FINGER.tolist(),
        "min_cop_force_N": MIN_COP_FORCE_N,
    }


def finger_vs_hand_axes(data: mujoco.MjData, ids: PandaIds) -> dict:
    """DIAGNOSTIC kinematics (robot FK only)."""
    Rh = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    RL = np.array(data.xmat[ids.left_body].reshape(3, 3), float)
    RR = np.array(data.xmat[ids.right_body].reshape(3, 3), float)
    rel_L = Rh.T @ RL
    rel_R = Rh.T @ RR
    return {
        "R_hand_T_R_left_diag": np.diag(rel_L).tolist(),
        "R_hand_T_R_right_diag": np.diag(rel_R).tolist(),
        "left_x_dot_hand_x": float(rel_L[0, 0]),
        "right_x_dot_hand_x": float(rel_R[0, 0]),
        "note": "left ~ I; right ~ diag(-1,-1,1) from XML quat 0 0 0 1 (180 deg about Z)",
    }


def as_plain(reading: SpatialTactileReading) -> dict:
    return {
        "left": asdict(reading.left),
        "right": asdict(reading.right),
    }
