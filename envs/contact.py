"""Fingertip contact from touch sites and MuJoCo contacts. No slip estimator."""

from __future__ import annotations

import numpy as np
import mujoco

from envs.ids import PandaIds


def touch_values(data: mujoco.MjData, ids: PandaIds) -> np.ndarray:
    if ids.touch_adr is None:
        return np.zeros(2)
    a0, a1 = ids.touch_adr
    return np.array([float(data.sensordata[a0]), float(data.sensordata[a1])])


def _contact_normal(model: mujoco.MjModel, data: mujoco.MjData, i: int) -> float:
    wrench = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, wrench)
    return abs(float(wrench[0]))


def finger_object_normals(
    model: mujoco.MjModel, data: mujoco.MjData, ids: PandaIds
) -> np.ndarray:
    """Sum contact normal forces between each finger body and the object geom."""
    left = 0.0
    right = 0.0
    obj_b = ids.object_body
    if obj_b < 0:
        return np.zeros(2)
    for i in range(int(data.ncon)):
        con = data.contact[i]
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        bodies = {b1, b2}
        if obj_b not in bodies:
            continue
        n = _contact_normal(model, data, i)
        if ids.left_body in bodies:
            left += n
        if ids.right_body in bodies:
            right += n
    return np.array([left, right])


def finger_cube_normals(
    model: mujoco.MjModel, data: mujoco.MjData, ids: PandaIds
) -> np.ndarray:
    return finger_object_normals(model, data, ids)
