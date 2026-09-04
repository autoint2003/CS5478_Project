"""Open/close gripper torques with optional contact-aware squeeze. No slip estimator."""

from __future__ import annotations

import numpy as np
import mujoco

from envs.ids import PandaIds


def read_touch(data: mujoco.MjData, ids: PandaIds) -> np.ndarray:
    if ids.touch_adr is None:
        return np.zeros(2)
    a0, a1 = ids.touch_adr
    return np.array([float(data.sensordata[a0]), float(data.sensordata[a1])])


def finger_opening(data: mujoco.MjData, ids: PandaIds) -> float:
    """Mean finger joint position in metres (0 closed, 0.04 fully open)."""
    return float(np.mean(data.qpos[ids.finger_jnt]))


def gripper_torque(phase: str, touch: np.ndarray, cfg: dict) -> float:
    g = cfg["gripper"]
    if phase in ("approach", "descend"):
        return float(g["open_tau"])
    tau = float(g["close_tau"])
    if g.get("contact_aware", True) and float(np.max(touch)) > float(g["touch_threshold"]):
        tau = float(g["close_boost"])
    return tau
