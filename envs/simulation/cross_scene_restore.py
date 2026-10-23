"""Identity-based physical restore across table vs GROUND_SCENE models.

Joint/body qpos layouts match (nq=23). Geom indexing does not — never replay a
foreign ``mj_setState`` blob or floor/table support masks into GROUND_SCENE.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from envs.outcomes.support_drop import arm_recovery_drop, is_ground_scene
from envs.simulation.ground_scene import SCENE_ID as GROUND_SCENE_ID
from envs.simulation.scene_state import ball_is_parked, park_impact_ball


def models_layout_compatible(src_model, dst_model) -> bool:
    return (
        int(src_model.nq) == int(dst_model.nq)
        and int(src_model.nv) == int(dst_model.nv)
        and int(src_model.nu) == int(dst_model.nu)
    )


def capture_scene_id(cap: dict) -> str:
    return str(cap.get("scene_id") or "")


def needs_identity_restore(sim, cap: dict) -> bool:
    """True when capture physics must not be applied via foreign mj_setState."""
    if not is_ground_scene(sim):
        return False
    sid = capture_scene_id(cap)
    if sid == GROUND_SCENE_ID:
        return False
    # Legacy elevated captures omit scene_id or mark table scenes.
    return True


def apply_physics_identity(sim, physics: dict) -> None:
    """Write qpos/qvel/ctrl/time/warmstart/act/mocap by value (same nq layout)."""
    qpos = np.asarray(physics["qpos"], float)
    qvel = np.asarray(physics["qvel"], float)
    ctrl = np.asarray(physics["ctrl"], float)
    if qpos.shape[0] != int(sim.model.nq) or qvel.shape[0] != int(sim.model.nv):
        raise RuntimeError(
            f"physics layout mismatch nq={sim.model.nq}/{qpos.shape[0]} "
            f"nv={sim.model.nv}/{qvel.shape[0]}"
        )
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.data.ctrl[:] = ctrl
    sim.data.time = float(physics["time"])
    if "qacc_warmstart" in physics:
        sim.data.qacc_warmstart[:] = np.asarray(physics["qacc_warmstart"], float)
    if "act" in physics and int(sim.model.na):
        act = np.asarray(physics["act"], float)
        if act.shape[0] == int(sim.model.na):
            sim.data.act[:] = act
    if "eq_model" in physics and hasattr(sim, "guide_eq"):
        sim.model.eq_active0[sim.guide_eq] = int(physics["eq_model"])
        if physics.get("eq_data") is not None and hasattr(sim.data, "eq_active"):
            sim.data.eq_active[sim.guide_eq] = int(physics["eq_data"])
        if "mocap_pos" in physics:
            sim.data.mocap_pos[sim.guide_mocap] = np.asarray(physics["mocap_pos"], float)
        if "mocap_quat" in physics:
            sim.data.mocap_quat[sim.guide_mocap] = np.asarray(physics["mocap_quat"], float)
    sim.captured = bool(physics.get("captured", False))


def copy_fsm(src_fsm, dst_fsm) -> None:
    for name in (
        "phase",
        "t_phase",
        "t_stable",
        "t_held",
        "lift_started",
        "success",
        "grasp_z",
        "fg_cmd",
    ):
        setattr(dst_fsm, name, getattr(src_fsm, name))
    for name in ("p_des", "r_des", "v_des", "w_des", "v_cmd", "w_cmd", "grasp_xy", "grasp_offset"):
        val = getattr(src_fsm, name, None)
        if val is None:
            setattr(dst_fsm, name, None)
        else:
            setattr(dst_fsm, name, np.asarray(val, float).copy())


def transfer_live_state(src_sim, dst_sim) -> dict[str, Any]:
    """Copy live elevated (or any layout-compatible) sim onto GROUND_SCENE."""
    if not models_layout_compatible(src_sim.model, dst_sim.model):
        raise RuntimeError("nq/nv/nu mismatch; cannot transfer")
    dst_sim.data.qpos[:] = src_sim.data.qpos
    dst_sim.data.qvel[:] = src_sim.data.qvel
    dst_sim.data.ctrl[:] = src_sim.data.ctrl
    dst_sim.data.time = float(src_sim.data.time)
    dst_sim.data.qacc_warmstart[:] = src_sim.data.qacc_warmstart
    if int(getattr(src_sim.model, "na", 0) or 0) and int(getattr(dst_sim.model, "na", 0) or 0):
        dst_sim.data.act[:] = src_sim.data.act
    if hasattr(src_sim, "guide_mocap") and hasattr(dst_sim, "guide_mocap"):
        dst_sim.data.mocap_pos[dst_sim.guide_mocap] = np.asarray(
            src_sim.data.mocap_pos[src_sim.guide_mocap], float
        )
        dst_sim.data.mocap_quat[dst_sim.guide_mocap] = np.asarray(
            src_sim.data.mocap_quat[src_sim.guide_mocap], float
        )
    if hasattr(src_sim, "guide_eq") and hasattr(dst_sim, "guide_eq") and hasattr(src_sim, "eq_active"):
        dst_sim.set_guide_weld(bool(src_sim.eq_active()))
    if ball_is_parked(src_sim):
        park_impact_ball(dst_sim)
    copy_fsm(src_sim.fsm, dst_sim.fsm)
    dst_sim.captured = bool(getattr(src_sim, "captured", False))
    mujoco.mj_forward(dst_sim.model, dst_sim.data)
    if is_ground_scene(dst_sim):
        arm_recovery_drop(dst_sim)
    return {
        "obj_z_src": float(src_sim.data.xpos[src_sim.ids.object_body][2]),
        "obj_z_dst": float(dst_sim.data.xpos[dst_sim.ids.object_body][2]),
        "hand_z_src": float(src_sim.data.xpos[src_sim.ids.hand_body][2]),
        "hand_z_dst": float(dst_sim.data.xpos[dst_sim.ids.hand_body][2]),
        "warmstart_max_abs_diff": float(
            np.max(np.abs(np.asarray(src_sim.data.qacc_warmstart) - np.asarray(dst_sim.data.qacc_warmstart)))
        ),
        "table_absent": int(mujoco.mj_name2id(dst_sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table") < 0),
        "ground_present": int(mujoco.mj_name2id(dst_sim.model, mujoco.mjtObj.mjOBJ_GEOM, "ground") >= 0),
    }


def object_ground_contact(sim) -> bool:
    ground = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    floor = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    obj = int(sim.ids.object_geom)
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if obj not in pair:
            continue
        other = (pair - {obj}).pop()
        if other in (ground, floor, table) and other >= 0:
            return True
    return False


def airborne_root_ok(sim, *, min_bottom_clearance: float = 0.05) -> dict[str, Any]:
    """Verify recovery IC is not support-borne after table removal."""
    gid = int(sim.ids.object_geom)
    hh = float(sim.model.geom_size[gid][1])
    z = float(sim.data.xpos[sim.ids.object_body][2])
    bottom = z - hh
    support = object_ground_contact(sim)
    return {
        "obj_z": z,
        "bottom_z": bottom,
        "support_contact": bool(support),
        "airborne": bool((not support) and bottom >= float(min_bottom_clearance)),
    }
