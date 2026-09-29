from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco


def _name2id(model: mujoco.MjModel, kind, names: tuple[str, ...]) -> int:
    for name in names:
        i = mujoco.mj_name2id(model, kind, name)
        if i >= 0:
            return int(i)
    return -1


@dataclass
class PandaIds:
    arm_jnt: np.ndarray
    arm_dof: np.ndarray
    finger_jnt: np.ndarray
    finger_dof: np.ndarray
    hand_body: int
    object_body: int
    object_jnt: int
    object_dof: int
    object_geom: int
    left_body: int
    right_body: int
    n_act: int
    ctrl_low: np.ndarray
    ctrl_high: np.ndarray
    home_qpos: np.ndarray
    touch_adr: tuple[int, int] | None

    @property
    def cube_body(self) -> int:
        return self.object_body

    @property
    def cube_jnt(self) -> int:
        return self.object_jnt

    @property
    def cube_dof(self) -> int:
        return self.object_dof

    @property
    def cube_geom(self) -> int:
        return self.object_geom


def _jid(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0:
        raise RuntimeError(f"joint not found: {name}")
    return i


def _bid(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise RuntimeError(f"body not found: {name}")
    return i


def resolve_ids(model: mujoco.MjModel) -> PandaIds:
    arm_jnt = np.array([_jid(model, f"joint{i}") for i in range(1, 8)], dtype=int)
    arm_dof = np.array([model.jnt_dofadr[j] for j in arm_jnt], dtype=int)
    finger_jnt = np.array(
        [_jid(model, "finger_joint1"), _jid(model, "finger_joint2")], dtype=int
    )
    finger_dof = np.array([model.jnt_dofadr[j] for j in finger_jnt], dtype=int)
    object_jnt_id = _name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, ("object_joint", "cube_joint")
    )
    object_body_id = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, ("object", "cube"))
    object_geom_id = _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ("object", "cube"))
    touch = None
    if model.nsensor >= 2:
        s0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_left")
        s1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_right")
        if s0 >= 0 and s1 >= 0:
            touch = (int(model.sensor_adr[s0]), int(model.sensor_adr[s1]))
    home = np.zeros(model.nq)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        home[:] = model.key_qpos[key]
    else:
        home[:9] = np.array(
            [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04]
        )
        if model.nq >= 16:
            home[9:16] = np.array([0.5, 0.0, 0.43, 1.0, 0.0, 0.0, 0.0])
    return PandaIds(
        arm_jnt=arm_jnt,
        arm_dof=arm_dof,
        finger_jnt=finger_jnt,
        finger_dof=finger_dof,
        hand_body=_bid(model, "hand"),
        object_body=int(object_body_id),
        object_jnt=int(object_jnt_id),
        object_dof=int(model.jnt_dofadr[object_jnt_id]) if object_jnt_id >= 0 else -1,
        object_geom=int(object_geom_id),
        left_body=_bid(model, "left_finger"),
        right_body=_bid(model, "right_finger"),
        n_act=int(model.nu),
        ctrl_low=model.actuator_ctrlrange[:, 0].copy(),
        ctrl_high=model.actuator_ctrlrange[:, 1].copy(),
        home_qpos=home,
        touch_adr=touch,
    )
