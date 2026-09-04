"""Bias-force compensation b(q, qdot) from MuJoCo qfrc_bias."""

from __future__ import annotations

import numpy as np
import mujoco

from envs.ids import PandaIds


def arm_bias(model: mujoco.MjModel, data: mujoco.MjData, ids: PandaIds) -> np.ndarray:
    """Return 7-vector b(q, qdot) for arm actuators (gravity + Coriolis/centrifugal)."""
    del model
    return data.qfrc_bias[ids.arm_dof].copy()


def finger_bias(data: mujoco.MjData, ids: PandaIds) -> np.ndarray:
    return data.qfrc_bias[ids.finger_dof].copy()


def hold_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PandaIds,
    gripper: str = "zero",
) -> np.ndarray:
    """Static-hold torques: arm uses b(q,qdot)~g(q); gripper is hold-open or zero.

    The split tendon uses coef 0.5 on each finger, so tendon force
    ``f`` contributes ``0.5 f`` to each finger DoF. Mapping
    ``ctrl_grip = bias_left + bias_right`` therefore matches mean finger bias.
    """
    ctrl = np.zeros(ids.n_act)
    ctrl[:7] = arm_bias(model, data, ids)
    if gripper == "bias":
        fb = finger_bias(data, ids)
        ctrl[7] = float(np.sum(fb))
    elif gripper == "open":
        ctrl[7] = 5.0
    elif gripper != "zero":
        raise ValueError(f"unknown gripper hold mode: {gripper}")
    return np.clip(ctrl, ids.ctrl_low, ids.ctrl_high)
