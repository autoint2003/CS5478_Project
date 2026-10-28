"""Synthetic D2 IC construction for Precision skill coverage.

Construction happens before freeze. No object teleport during Precision.
Tags: synthetic_initial_condition=True.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from data_generation.expert_pipeline.live_stage1 import settle_grip
from envs.simulation.capture import restore_capture, take_capture
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary


def _object_qpos_adr(sim) -> int | None:
    jnt = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if jnt < 0:
        # try free joint by body
        bid = int(sim.ids.object_body)
        for j in range(sim.model.njnt):
            if int(sim.model.jnt_bodyid[j]) == bid and int(sim.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(sim.model.jnt_qposadr[j])
        return None
    return int(sim.model.jnt_qposadr[jnt])


def construct_d2_ic(
    env,
    base_blob: dict,
    *,
    target_skill: str,
    dtheta_deg: float = 0.0,
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    dz_mm: float = 0.0,
    settle_n: int = 12,
) -> dict[str, Any]:
    """Perturb object pose in hand frame from a stable base IC, then settle.

    Returns dict with capture/ref/construction meta, or ok=False.
    """
    restore_capture(env, base_blob["capture"])
    sim = env.sim
    mujoco.mj_forward(sim.model, sim.data)
    cs0 = contact_summary(sim)
    if not cs0.get("bilateral"):
        return {"ok": False, "reason": "BASE_NOT_BILATERAL"}

    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float).copy()
    # Lateral / axial offsets in hand frame → world
    offset_h = np.array([dx_mm, dy_mm, dz_mm], float) * 1e-3
    po_new = po + Rh @ offset_h

    adr = _object_qpos_adr(sim)
    if adr is None:
        return {"ok": False, "reason": "NO_OBJECT_FREE_JOINT"}
    # free joint: quat is qpos[adr+3:adr+7] (wxyz)
    qpos = np.array(sim.data.qpos, float).copy()
    qpos[adr : adr + 3] = po_new
    if abs(dtheta_deg) > 1e-9:
        # rotate about hand open axis (approx hand y) for orientation residual
        axis_w = Rh[:, 1]
        axis_w = axis_w / max(float(np.linalg.norm(axis_w)), 1e-12)
        ang = np.deg2rad(float(dtheta_deg))
        # quaternion multiply: q_new = q_delta * q_old
        qw, qx, qy, qz = qpos[adr + 3 : adr + 7]
        half = 0.5 * ang
        dw = np.cos(half)
        dx_, dy_, dz_ = axis_w * np.sin(half)
        # q_delta * q
        nw = dw * qw - dx_ * qx - dy_ * qy - dz_ * qz
        nx = dw * qx + dx_ * qw + dy_ * qz - dz_ * qy
        ny = dw * qy - dx_ * qz + dy_ * qw + dz_ * qx
        nz = dw * qz + dx_ * qy - dy_ * qx + dz_ * qw
        qpos[adr + 3 : adr + 7] = np.array([nw, nx, ny, nz], float)
        qpos[adr + 3 : adr + 7] /= max(float(np.linalg.norm(qpos[adr + 3 : adr + 7])), 1e-12)

    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    settle_grip(env, n=settle_n)
    cs1 = contact_summary(sim)
    if not cs1.get("bilateral"):
        return {"ok": False, "reason": "LOST_BILATERAL_AFTER_CONSTRUCT", "target_skill": target_skill}
    if float(measure(sim, U_TGT, P_TGT)["obj_z"]) < 0.20:
        return {"ok": False, "reason": "OBJECT_TOO_LOW", "target_skill": target_skill}

    cap = take_capture(env)
    meta = {
        "target_skill": target_skill,
        "requested_dtheta_deg": float(dtheta_deg),
        "requested_dx_mm": float(dx_mm),
        "requested_dy_mm": float(dy_mm),
        "requested_dz_mm": float(dz_mm),
        "settle_n": int(settle_n),
        "base_sample_id": base_blob.get("sample_id"),
        "construction": "hand_frame_object_pose_perturb_then_settle",
    }
    return {
        "ok": True,
        "capture": cap,
        "ref": base_blob["ref"],
        "construction": meta,
        "contact": {
            "nL": int(cs1.get("nL") or 0),
            "nR": int(cs1.get("nR") or 0),
            "bilateral": bool(cs1.get("bilateral")),
        },
    }


# Skill-targeted construction recipes (physically modest offsets).
D2_RECIPES: list[dict[str, Any]] = [
    {"target_skill": "dtheta", "dtheta_deg": 18.0, "dx_mm": 0.0, "dz_mm": 0.0},
    {"target_skill": "dtheta", "dtheta_deg": -22.0, "dx_mm": 2.0, "dz_mm": 0.0},
    {"target_skill": "dx", "dtheta_deg": 0.0, "dx_mm": 14.0, "dz_mm": 0.0},
    {"target_skill": "dx", "dtheta_deg": 0.0, "dx_mm": -14.0, "dz_mm": 2.0},
    {"target_skill": "dz_inward", "dtheta_deg": 0.0, "dx_mm": 2.0, "dz_mm": -18.0},
    {"target_skill": "dz_outward", "dtheta_deg": 0.0, "dx_mm": 2.0, "dz_mm": 18.0},
    {"target_skill": "support_transfer", "dtheta_deg": 5.0, "dx_mm": 16.0, "dz_mm": 4.0},
    {"target_skill": "dtheta+dx", "dtheta_deg": 20.0, "dx_mm": 12.0, "dz_mm": 0.0},
    {"target_skill": "dx+dz", "dtheta_deg": 0.0, "dx_mm": 12.0, "dz_mm": -14.0},
    {"target_skill": "dtheta+dz", "dtheta_deg": 16.0, "dx_mm": 3.0, "dz_mm": 16.0},
]
