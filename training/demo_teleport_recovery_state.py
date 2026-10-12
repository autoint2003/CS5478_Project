"""Coupled translation+rotation teleport IC (privileged pose feasibility).

Not a physical impact. Direction/coupling taken from the canonical ball trajectory.

    python training/demo_teleport_recovery_state.py --measure-impact
    python training/demo_teleport_recovery_state.py --sweep
    python training/demo_teleport_recovery_state.py --mode zero --case coupled_boundary
    python training/demo_teleport_recovery_state.py --ablate --s 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import deque

from controllers.residual import (
    RECOVERY4D_TAU_OPEN,
    RECOVERY4D_TAU_SECURE,
    RECOVERY4D_W_HY_MAX,
)
from controllers.jacobian_controller import (
    gains_from_cfg,
    jacobian_6d,
    ori_error_deg,
    rotation_error,
)
from envs.deterioration import body_twist
from controllers.nominal import grasp_orientation
from envs.config_util import load_yaml, merge_sim_config
from envs.dynamics import set_finger_object_sliding_mu
from envs.grasp_sim import GraspSim
from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from training.impact_ball import apply_ball_free_state
from training.impact_demo_core import (
    RENDER_HZ,
    BallFlight,
    GuidedApproach,
    RealtimePacer,
    _tilt_deg,
    load_demo_config,
    make_demo_sim,
    plan_guided_release,
    save_rgb,
)
from training.impact_visualization_utils import apply_viewer_camera, enable_viewer_flags, tick_vw_park
from training.replay_core import (
    G_HOLD,
    disable_object_table,
    dump_ball_contacts,
    freeze,
    isolate_ball_object_only,
    object_free_adr,
    pack_replay,
    restore_replay,
    tick_vw,
)

LOG_DIR = ROOT / "results" / "logs" / "coupled_teleport"
AUTH_DIR = ROOT / "results" / "logs" / "coupled_ic_recovery"
IC_NPZ = AUTH_DIR / "coupled_s2_recovery_ic.npz"
SEQ_JSON = AUTH_DIR / "successful_sequence.json"
S_COUPLED = 2.0
PARENT_HOLD = 0.20
PAIR_MU = 1.0
CYL_MASS = 0.20
PEN_REJECT = 0.003  # m; reject only gross overlap
EXPLODE_V = 4.0
IMMEDIATE_S = 0.08
DELAYED_LO = 0.10
DELAYED_HI = 1.50
LOG_DT = 0.002  # physics-rate around construction; downsample later in hold
POST_MS = 0.080  # representative post-impact pose


def skew(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, float).reshape(3)
    return np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])


def rotmat_from_rotvec(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float).reshape(3)
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def rotvec_from_rotmat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, float).reshape(3, 3)
    c = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-12:
        return np.zeros(3)
    if abs(np.pi - th) < 1e-6:
        ev = np.array([R[0, 0] + 1.0, R[1, 1] + 1.0, R[2, 2] + 1.0])
        k = np.sqrt(np.maximum(ev, 0.0))
        if k[0] >= k[1] and k[0] >= k[2]:
            k = np.array([k[0], R[0, 1] / max(k[0], 1e-12), R[0, 2] / max(k[0], 1e-12)])
        elif k[1] >= k[2]:
            k = np.array([R[0, 1] / max(k[1], 1e-12), k[1], R[1, 2] / max(k[1], 1e-12)])
        else:
            k = np.array([R[0, 2] / max(k[2], 1e-12), R[1, 2] / max(k[2], 1e-12), k[2]])
        n = float(np.linalg.norm(k))
        return (k / max(n, 1e-12)) * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    k = k / (2.0 * np.sin(th))
    return k * th


def rel_pack(sim) -> dict:
    o = physical_pack(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    qadr, dadr = object_free_adr(sim)
    return {
        "p_rel_h": np.asarray(o["rh"], float).copy(),
        "R_rel": np.asarray(o["R_rel"], float).copy(),
        "Rh": Rh,
        "Ro": Ro,
        "ph": ph,
        "po": po,
        "e_x": float(o["e_x"]),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "Fn_L": float(o["Fn_L"]),
        "Fn_R": float(o["Fn_R"]),
        "v_rel_h": np.asarray(o["v_rel_h"], float).copy(),
        "w_rel_h": np.asarray(o["w_rel_h"], float).copy(),
        "v_rel": float(o["v_rel"]),
        "w_rel": float(o["w_rel"]),
        "obj_z": float(o["obj_z"]),
        "clear": float(o["clear"]),
        "aperture": float(o["aperture"]),
        "tau": float(o["tau"]),
        "tilt_deg": _tilt_deg(sim),
        "obj_linvel": np.array(sim.data.qvel[dadr : dadr + 3], float).copy(),
        "obj_angvel": np.array(sim.data.qvel[dadr + 3 : dadr + 6], float).copy(),
        "cyl_axis_w": Ro[:, 2].copy(),
    }


def contact_debug(sim) -> dict:
    model, data, ids = sim.model, sim.data, sim.ids
    rows = []
    min_dist = None
    wr = np.zeros(6)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        bodies = {int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])}
        if int(ids.object_geom) not in (g1, g2):
            continue
        if ids.left_body not in bodies and ids.right_body not in bodies:
            continue
        mujoco.mj_contactForce(model, data, i, wr)
        d = float(c.dist)
        min_dist = d if min_dist is None else min(min_dist, d)
        rows.append(
            {
                "side": "L" if ids.left_body in bodies else "R",
                "pos": np.array(c.pos, float).tolist(),
                "normal": np.array(c.frame[:3], float).tolist(),
                "dist": d,
                "Fn": float(abs(wr[0])),
            }
        )
    return {"contacts": rows, "min_dist": min_dist}


def frames_at(sim) -> dict:
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    u_ball_w = Rh[:, 0].copy()
    return {
        "hand_x_world": Rh[:, 0].tolist(),
        "hand_y_world": Rh[:, 1].tolist(),
        "hand_z_world": Rh[:, 2].tolist(),
        "cylinder_axis_world": Ro[:, 2].tolist(),
        "finger_opposition_world": Rh[:, 1].tolist(),
        "canonical_ball_incoming_world": u_ball_w.tolist(),
        "canonical_ball_incoming_hand": (Rh.T @ u_ball_w).tolist(),
        "yaml_impact_direction_world": [0.0, 1.0, 0.0],
    }


def apply_coupled_pose(sim, dp_h: np.ndarray, R_delta: np.ndarray) -> dict:
    """Teleport object pose only. Keep object twist. Hand/ctrl/p_des untouched."""
    before = rel_pack(sim)
    qadr, dadr = object_free_adr(sim)
    v_keep = np.array(sim.data.qvel[dadr : dadr + 6], float).copy()
    Rh, ph = before["Rh"], before["ph"]
    p_rel = before["p_rel_h"]
    R_rel = before["R_rel"]
    p_new = ph + Rh @ (p_rel + np.asarray(dp_h, float).reshape(3))
    R_rel_new = np.asarray(R_delta, float).reshape(3, 3) @ R_rel
    Ro_new = Rh @ R_rel_new
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Ro_new.reshape(9))
    sim.data.qpos[qadr : qadr + 3] = p_new
    sim.data.qpos[qadr + 3 : qadr + 7] = quat
    sim.data.qvel[dadr : dadr + 6] = v_keep
    mujoco.mj_forward(sim.model, sim.data)
    after = rel_pack(sim)
    return {"before": before, "after": after, "v_keep": v_keep.tolist()}


def coupled_pose_from_ref(ref: dict, s: float, kind: str) -> tuple[np.ndarray, np.ndarray]:
    dp_ref = np.asarray(ref["delta_p_h"], float)
    w_ref = np.asarray(ref["rotvec_h"], float)
    if kind == "translation":
        return s * dp_ref, np.eye(3)
    if kind == "rotation":
        return np.zeros(3), rotmat_from_rotvec(s * w_ref)
    return s * dp_ref, rotmat_from_rotvec(s * w_ref)


def pack_canonical_ic(sim) -> dict:
    """Complete restore payload at the recovery decision boundary."""
    snap = sim.snapshot()
    snap["v_des"] = np.asarray(sim.fsm.v_des, float).copy()
    snap["w_des"] = np.asarray(sim.fsm.w_des, float).copy()
    snap["qacc"] = np.array(sim.data.qacc, float).copy()
    snap["pair_mu"] = float(PAIR_MU)
    snap["mass"] = float(sim.mass)
    snap["friction"] = float(sim.friction)
    return snap


def restore_canonical_ic(sim, ic: dict) -> None:
    """Restore pack_canonical_ic. Do not call mj_setConst after writing qpos."""
    sim.load_snapshot(ic)
    # Friction write only. Passing data would mj_setConst and wipe qpos (home).
    set_finger_object_sliding_mu(sim.model, sim.ids, PAIR_MU, data=None)
    sim.data.qpos[:] = np.asarray(ic["qpos"], float)
    sim.data.qvel[:] = np.asarray(ic["qvel"], float)
    sim.data.ctrl[:] = np.asarray(ic["ctrl"], float)
    if len(ic.get("act", [])):
        sim.data.act[:] = np.asarray(ic["act"], float)
    if "qacc_warmstart" in ic:
        sim.data.qacc_warmstart[:] = np.asarray(ic["qacc_warmstart"], float)
    sim.data.time = float(np.asarray(ic["time"]))
    if "v_des" in ic:
        sim.fsm.v_des = np.asarray(ic["v_des"], float).copy()
    if "w_des" in ic:
        sim.fsm.w_des = np.asarray(ic["w_des"], float).copy()
    mujoco.mj_forward(sim.model, sim.data)


def save_canonical_ic(path: Path, ic: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in ic.items():
        if v is None:
            payload[k] = np.array(None, dtype=object)
        elif isinstance(v, str):
            payload[k] = np.array(v)
        elif isinstance(v, (bool, np.bool_)):
            payload[k] = np.array(bool(v))
        else:
            payload[k] = np.asarray(v)
    np.savez_compressed(path, **payload)


def load_canonical_ic(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in z.files:
        v = z[k]
        if v.shape == () and v.dtype.kind in ("U", "S", "O"):
            item = v.item()
            out[k] = item
        elif v.shape == () and v.dtype == bool:
            out[k] = bool(v)
        else:
            out[k] = np.array(v)
    return out


def dump_branch_state(sim) -> dict:
    qadr, dadr = object_free_adr(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    objg = int(sim.ids.object_geom)
    return {
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "act": np.array(sim.data.act, float).copy() if int(getattr(sim.model, "na", 0) or 0) else np.zeros(0),
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "r_des": np.asarray(sim.fsm.r_des, float).copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).copy(),
        "time": float(sim.data.time),
        "object_qpos": np.array(sim.data.qpos[qadr : qadr + 7], float).copy(),
        "object_qvel": np.array(sim.data.qvel[dadr : dadr + 6], float).copy(),
        "arm_qpos": np.array(sim.data.qpos[sim.ids.arm_jnt], float).copy(),
        "arm_qvel": np.array(sim.data.qvel[sim.ids.arm_dof], float).copy(),
        "p_hand": np.array(sim.data.xpos[sim.ids.hand_body], float).copy(),
        "R_hand": Rh.copy(),
        "ori_err_deg": ori_error_deg(sim.fsm.r_des, Rh),
        "obj_contype": int(sim.model.geom_contype[objg]),
        "obj_conaffinity": int(sim.model.geom_conaffinity[objg]),
        "table_contype": int(sim.model.geom_contype[table]) if table >= 0 else None,
        "table_conaffinity": int(sim.model.geom_conaffinity[table]) if table >= 0 else None,
        "mass": float(sim.mass),
        "friction": float(sim.friction),
        "pair_mu_object": float(sim.model.geom_friction[objg][0]),
    }


def freeze_targets(sim) -> None:
    """Explicit hold: p_des := p_hand, r_des := R_hand, v_cmd=w_cmd=0."""
    freeze(sim)
    sim.fsm.v_des[:] = 0.0
    sim.fsm.w_des[:] = 0.0


def visual_pause(viewer, sim, s, phase, duration: float, mode: str) -> None:
    """Wall-clock pause. Does not step physics or change targets."""
    if viewer is None or duration <= 0:
        return
    import time as _time

    t0 = _time.perf_counter()
    t = float(sim.data.time)
    while _time.perf_counter() - t0 < float(duration):
        overlay(viewer, s, phase, t, mode=mode)
        viewer.sync()
        _time.sleep(1.0 / 60.0)


def _row(sim, phase: str) -> dict:
    o = physical_pack(sim)
    rh = np.asarray(o["rh"], float)
    R_rel = np.asarray(o["R_rel"], float)
    rv = rotvec_from_rotmat(R_rel)
    dbg = contact_debug(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    p_des = np.asarray(sim.fsm.p_des, float)
    r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    v_cmd = np.asarray(sim.fsm.v_cmd, float)
    w_cmd = np.asarray(sim.fsm.w_cmd, float)
    j = jacobian_6d(sim.model, sim.data, sim.ids)
    xdot = j @ np.array(sim.data.qvel[sim.ids.arm_dof], float)
    e_p = p_des - ph
    e_o = rotation_error(r_des, Rh)
    kp = np.array([180.0, 180.0, 180.0, 16.0, 16.0, 16.0])
    kd = np.array([28.0, 28.0, 28.0, 2.4, 2.4, 2.4])
    wrench = kp * np.concatenate([e_p, e_o]) + kd * (np.concatenate([v_cmd, w_cmd]) - xdot)
    return {
        "t": float(sim.data.time),
        "phase": phase,
        "story": story_name(phase),
        "e_x": float(rh[0]),
        "e_y": float(rh[1]),
        "e_z": float(rh[2]),
        "rotvec_h": rv.tolist(),
        "rot_angle_deg": float(np.degrees(np.linalg.norm(rv))),
        "v_rel_hand": np.asarray(o["v_rel_h"], float).tolist(),
        "omega_rel_hand": np.asarray(o["w_rel_h"], float).tolist(),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "Fn_L": float(o["Fn_L"]),
        "Fn_R": float(o["Fn_R"]),
        "contacts": dbg["contacts"],
        "min_dist": dbg["min_dist"],
        "aperture": float(o["aperture"]),
        "tau": float(o["tau"]),
        "object": np.array(sim.data.xpos[sim.ids.object_body], float).tolist(),
        "hand": ph.tolist(),
        "hand_R": Rh.reshape(9).tolist(),
        "hand_linvel": vh.tolist(),
        "hand_angvel": wh.tolist(),
        "obj_linvel": vo.tolist(),
        "obj_angvel": wo.tolist(),
        "wrench": wrench.tolist(),
        "p_des": p_des.tolist(),
        "r_des": r_des.reshape(9).tolist(),
        "v_cmd": v_cmd.tolist(),
        "w_cmd": w_cmd.tolist(),
        "pos_err": (p_des - ph).tolist(),
        "ori_err_deg": ori_error_deg(r_des, Rh),
        "rot_err": rotation_error(r_des, Rh).tolist(),
        "arm_qpos": np.array(sim.data.qpos[sim.ids.arm_jnt], float).tolist(),
        "arm_qvel": np.array(sim.data.qvel[sim.ids.arm_dof], float).tolist(),
        "ctrl": np.array(sim.data.ctrl, float).tolist(),
        "grip_ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float("nan"),
        "tilt_deg": _tilt_deg(sim),
        "cyl_axis_h": cyl_axis_hand(sim).tolist(),
        "obj_z": float(o["obj_z"]),
        "clear": float(o["clear"]),
        "v_rel": float(o["v_rel"]),
        "w_rel": float(o["w_rel"]),
        "g_h": np.asarray(o["g_h"], float).tolist(),
        "contacts_h": [
            {
                "side": c["side"],
                "pos_h": (Rh.T @ (np.asarray(c["pos"], float) - ph)).tolist(),
                "normal_h": (Rh.T @ np.asarray(c["normal"], float)).tolist(),
                "Fn": c["Fn"],
                "dist": c["dist"],
            }
            for c in dbg["contacts"]
        ],
        "wrist_cmd_deg": float(getattr(sim, "_wrist_cmd_deg", 0.0)),
        "wrist_act_deg": float(getattr(sim, "_wrist_act_deg", 0.0)),
        "wrist_des_deg": float(getattr(sim, "_wrist_des_deg", 0.0)),
        "fail_latch": getattr(sim, "_fail_latch", None),
        **{k: v for k, v in d_in_pack(sim, rh).items()},
    }


def story_name(phase: str) -> str:
    p = str(phase).lower()
    if p in ("approach", "descend", "close", "lift", "parent_hold", "normal"):
        return "NORMAL"
    if p in ("teleport", "disturb", "disturbed"):
        return "DISTURB"
    if p in ("observe", "observe_common"):
        return "OBSERVE"
    if p in ("wrist_sweep", "sweep"):
        return "WRIST SWEEP"
    if p in ("contact",):
        return "CONTACT"
    if p in ("capture", "capturable"):
        return "CAPTURE"
    if p in ("wrist", "hx", "grip_secure", "grip_stronger", "unload", "open", "reclose", "recover", "pulse"):
        return "RECOVER"
    if p in ("rotate_to_capture", "rotate"):
        return "ROTATE"
    if p in ("reposition",):
        return "REPOSITION"
    if p in ("regrasp",):
        return "REGRASP"
    if p in ("capture_verify",):
        return "CAPTURE VERIFY"
    if p in ("return_vertical", "return"):
        return "RETURN"
    if p in ("vertical_verify", "vertical_stabilize", "vertical_hold"):
        return "VERTICAL HOLD"
    if p in ("inward_slide", "gravity_reposition"):
        return "INWARD SLIDE"
    if p in ("secure", "secure_hold"):
        return "SECURE HOLD" if p == "secure_hold" else "SECURE"
    if p in ("failure_watch", "failure"):
        return "FAILURE"
    if p in ("continue_lift", "resume", "nominal_resume", "stop", "lift_resume"):
        return "LIFT" if p in ("continue_lift", "lift_resume") else "RESUME"
    if p in ("task_hold",):
        return "TASK HOLD"
    if p in ("hold", "diagnostic_hold"):
        return "DIAGNOSTIC HOLD"
    return str(phase).upper()


def cyl_axis_hand(sim) -> np.ndarray:
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    return Rh.T @ Ro[:, 2]


def pad_span_hand(sim) -> dict:
    """Finger-pad AABB in the hand frame from live box geoms (not e_x/e_z by fiat).

    Finger bodies sit at hand +z ≈ 0.0584 m. Pad boxes are in the finger frame:
    main pad size (8.5, 4, 8.5) mm at finger (0, 5.5, 44.5) mm. Finger +z is the
    distal/fingertip direction, which is +hand-z. Pads face each other along ±y.
    Pad width is ±hand-x about 0. The useful capture box is that AABB.
    """
    model, data, ids = sim.model, sim.data, sim.ids
    Rh = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(data.xpos[ids.hand_body], float)
    xs, ys, zs = [], [], []
    for bid in (int(ids.left_body), int(ids.right_body)):
        for g in range(int(model.ngeom)):
            if int(model.geom_bodyid[g]) != bid:
                continue
            if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            sz = np.array(model.geom_size[g], float)
            xpos = np.array(data.geom_xpos[g], float)
            xmat = np.array(data.geom_xmat[g].reshape(3, 3), float)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for szz in (-1.0, 1.0):
                        pw = xpos + xmat @ (sz * np.array([sx, sy, szz]))
                        h = Rh.T @ (pw - ph)
                        xs.append(float(h[0]))
                        ys.append(float(h[1]))
                        zs.append(float(h[2]))
    if not xs:
        return {
            "x_lo": -0.0085,
            "x_hi": 0.0085,
            "y_lo": -0.02,
            "y_hi": 0.02,
            "z_lo": 0.094,
            "z_hi": 0.111,
            "center": [0.0, 0.0, 0.103],
        }
    return {
        "x_lo": min(xs),
        "x_hi": max(xs),
        "y_lo": min(ys),
        "y_hi": max(ys),
        "z_lo": min(zs),
        "z_hi": max(zs),
        "center": [0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)), 0.5 * (min(zs) + max(zs))],
    }


def d_in_pack(sim, rh=None) -> dict:
    """Positive d_in = deeper into the pad capture box from the outer-distal corner.

    û_in points from the object's current-side distal pad corner toward the pad
    center in the (hand-x, hand-z) pad plane. This is NOT identically -e_x or -e_z:
    at 60 deg gravity is mostly -hand-x, so inward seating is primarily across the
    pad width (toward x=0), while +e_z is still distal toward the fingertips.
    """
    if rh is None:
        o = physical_pack(sim)
        rh = np.asarray(o["rh"], float)
    else:
        rh = np.asarray(rh, float).reshape(3)
    pad = pad_span_hand(sim)
    c = np.array(pad["center"], float)
    side = 1.0 if float(rh[0]) >= float(c[0]) else -1.0
    corner = np.array(
        [float(pad["x_hi"]) if side > 0 else float(pad["x_lo"]), 0.0, float(pad["z_hi"])],
        float,
    )
    u = np.array([c[0] - corner[0], 0.0, c[2] - corner[2]], float)
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-12 else np.array([-1.0, 0.0, 0.0])
    r_xz = np.array([rh[0], 0.0, rh[2]], float)
    d_in = float(np.dot(r_xz - corner, u))
    return {
        "d_in": d_in,
        "d_in_mm": 1e3 * d_in,
        "u_in_h": u.tolist(),
        "pad": pad,
        "from_plus_x_rim_mm": 1e3 * (float(pad["x_hi"]) - float(rh[0])),
        "from_distal_tip_mm": 1e3 * (float(pad["z_hi"]) - float(rh[2])),
        "d_in_definition": (
            "d_in = (r_h_xz - pad_outer_distal_corner) · û_in; "
            "û_in = pad_center_xz - that corner; +d_in = farther from the "
            "distal/outer loss faces into the pad interior / grasp midline"
        ),
    }


def unsigned_axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).reshape(3)
    b = np.asarray(b, float).reshape(3)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    c = float(np.clip(abs(np.dot(a / na, b / nb)), 0.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def overlay(viewer, s, phase, t, mode: str = "ZERO", extra: dict | None = None) -> None:
    if viewer is None or not hasattr(viewer, "add_overlay"):
        return
    try:
        pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
        extra = extra or {}
        viewer.add_overlay(pos, "PHASE", story_name(phase) if phase else "-")
        if extra.get("minimal"):
            viewer.add_overlay(
                pos,
                "WRIST",
                f"{float(extra.get('wrist_act_deg', 0)):.1f} deg",
            )
            viewer.add_overlay(pos, "TAU", f"{float(extra.get('tau', extra.get('grip_ctrl', 0))):.1f}")
            viewer.add_overlay(pos, "APERTURE", f"{float(extra.get('aperture', 0)):.4f} m")
            if extra.get("v_rel") is not None:
                viewer.add_overlay(pos, "|v_rel|", f"{float(extra['v_rel']):.4f} m/s")
            if extra.get("d_in_mm") is not None:
                viewer.add_overlay(pos, "d_in", f"{float(extra['d_in_mm']):.2f} mm")
            if extra.get("fail_phase"):
                viewer.add_overlay(pos, "FAILURE", str(extra["fail_phase"]))
            return
        viewer.add_overlay(pos, "MODE", str(mode).upper())
        viewer.add_overlay(pos, "t", f"{t:.3f} s")
        viewer.add_overlay(pos, "CASE", f"coupled s={float(s if s is not None else S_COUPLED):.1f}")
        if "wrist_cmd_deg" in extra or "wrist_act_deg" in extra:
            viewer.add_overlay(
                pos,
                "WRIST",
                f"cmd {float(extra.get('wrist_cmd_deg', 0)):.1f} deg  "
                f"act {float(extra.get('wrist_act_deg', 0)):.1f} deg",
            )
        if "nL" in extra:
            viewer.add_overlay(
                pos,
                "CONTACT",
                f"nL={int(extra['nL'])} nR={int(extra.get('nR', 0))}  "
                f"FnL={float(extra.get('Fn_L', 0)):.1f} FnR={float(extra.get('Fn_R', 0)):.1f}",
            )
    except Exception:
        return


def _cam(sim) -> dict:
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    return {"lookat": po + np.array([0.0, 0.0, 0.01]), "distance": 0.42, "azimuth": 125.0, "elevation": -18.0}


def make_parent_sim() -> GraspSim:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = GraspSim(cfg)
    sim.reset(CYL_MASS, PAIR_MU, np.zeros(3))
    set_finger_object_sliding_mu(sim.model, sim.ids, PAIR_MU, sim.data)
    return sim


def advance_to_parent(sim, cfg, *, viewer=None, pacer=None, mode: str = "ZERO") -> dict:
    """Nominal FSM to airborne bilateral hold PARENT_HOLD. Returns snap + geometry."""
    dt = float(sim.model.opt.timestep)
    air_ok = 0.0
    n_max = int(round(12.0 / dt))
    pre = deque(maxlen=int(round(0.050 / dt)) + 2)
    for _ in range(n_max):
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
        t = float(sim.data.time)
        o = physical_pack(sim)
        pre.append(_row(sim, str(sim.fsm.phase)))
        if viewer is not None and pacer is not None and pacer.should_render(t):
            overlay(viewer, None, str(sim.fsm.phase), t, mode=mode)
            viewer.sync()
            pacer.note_sync(t)
            pacer.wait_if_ahead(t)
        if sim.fsm.phase == "lift" and sim.captured and float(o["obj_z"]) >= Z_AIR:
            if int(o["nL"]) > 0 and int(o["nR"]) > 0 and float(o["v_rel"]) < 0.05:
                air_ok += dt
            else:
                air_ok = 0.0
            if air_ok >= PARENT_HOLD and abs(float(o["e_x"])) < 0.003:
                freeze(sim)
                geom = frames_at(sim)
                rel = rel_pack(sim)
                snap = pack_replay(sim)
                snap["mass"] = CYL_MASS
                snap["friction"] = PAIR_MU
                return {
                    "ok": True,
                    "snap": snap,
                    "geom": geom,
                    "rel": rel,
                    "t": t,
                    "pre_log": list(pre),
                }
        if t >= 12.0:
            break
    return {"ok": False, "reason": "failed to reach stable airborne parent"}


def _ball_cyl_contact_pos(sim) -> np.ndarray | None:
    if not hasattr(sim, "ball_geom"):
        return None
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if sim.ball_geom in (g1, g2) and int(sim.ids.object_geom) in (g1, g2):
            return np.array(c.pos, float).copy()
    return None


def measure_canonical_impact() -> dict:
    """Raw pre/post relative pose from the existing canonical ZERO impact episode."""
    demo = load_demo_config()
    phys = demo["physical"]
    timing = demo["timing"]
    sim, cfg = make_demo_sim(float(phys["ball_mass"]))
    set_finger_object_sliding_mu(sim.model, sim.ids, PAIR_MU, sim.data)
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    g = np.array(sim.model.opt.gravity, float)
    T_impact = float(phys.get("desired_impact_time", timing["t_stable_airborne"] + timing["impact_margin"]))
    p_hit = np.asarray(phys["intended_contact_region"], float)
    t_free = float(phys.get("free_flight", 0.20))
    t_release = T_impact - t_free
    plan = plan_guided_release(
        p_hit,
        float(phys.get("v_hit_lateral", 3.0)),
        t_release,
        t_free,
        float(phys.get("t_accel", 0.18)),
        float(phys.get("v_cruise", 0.28)),
        g,
    )
    guide = GuidedApproach(plan)
    apply_ball_free_state(sim, plan["p_start"], plan["v_cruise"])
    sim.set_guide_pos(plan["p_start"])
    sim.set_guide_weld(True)
    mujoco.mj_forward(sim.model, sim.data)
    ball = BallFlight()
    guided_on = True
    air_extra = 0.0
    phase = "grasp"
    pre = None
    post = None
    first_hit_rel = None
    r_imp = None
    contact_pos = None
    t_post = None
    n_max = int(round((T_impact + 1.2) / dt))
    for _ in range(n_max):
        t = float(sim.data.time)
        if guided_on:
            sim.set_guide_pos(guide.pos(t))
            if t + 0.5 * dt >= t_release:
                sim.set_guide_weld(False)
                guided_on = False
        if phase == "grasp":
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
            o = physical_pack(sim)
            if sim.fsm.phase == "lift" and sim.captured and float(o["obj_z"]) >= Z_AIR:
                air_extra += dt
                if air_extra >= float(timing["stable_airborne_hold"]):
                    freeze(sim)
                    disable_object_table(sim)
                    isolate_ball_object_only(sim)
                    phase = "hold"
        else:
            # tick_vw parks the impact ball; keep it free-flying.
            tick_vw_park(sim, np.zeros(3), np.zeros(3), G_HOLD, gains, park=False)
        hits = [r for r in dump_ball_contacts(sim) if r.get("kind") == "ball-cylinder"]
        ball.note(sim, guided=guided_on)
        if hits and first_hit_rel is None:
            first_hit_rel = rel_pack(sim)
            contact_pos = _ball_cyl_contact_pos(sim)
            po = first_hit_rel["po"]
            if contact_pos is not None:
                r_imp = contact_pos - po
            else:
                q = int(sim.ball_qadr)
                pb = np.array(sim.data.qpos[q : q + 3], float)
                r_imp = pb - po
        rp = rel_pack(sim)
        if ball.t_first_contact is None:
            pre = rp
        if ball.t_first_contact is not None and post is None:
            departed = ball.state in ("DEPARTED", "PARKED_AFTER_IMPACT") or (
                ball.t_last_contact is not None
                and t >= float(ball.t_last_contact) + POST_MS
                and not hits
            )
            fallback = t >= float(ball.t_first_contact) + 0.12
            if departed or fallback:
                post = rp
                t_post = t
                break
        if t >= T_impact + 1.0:
            break
    if pre is None or post is None:
        o = physical_pack(sim)
        return {
            "ok": False,
            "reason": "could not sample pre/post impact relative pose",
            "t": float(sim.data.time),
            "fsm": str(sim.fsm.phase),
            "measure_phase": phase,
            "captured": bool(sim.captured),
            "obj_z": float(o["obj_z"]),
            "t_first": ball.t_first_contact,
            "t_last": ball.t_last_contact,
            "ball_state": ball.state,
            "pre_is_none": pre is None,
            "post_is_none": post is None,
            "n_max": n_max,
            "T_impact": T_impact,
            "t_release": t_release,
            "air_extra": air_extra,
        }
    dp = post["p_rel_h"] - pre["p_rel_h"]
    dR = post["R_rel"] @ pre["R_rel"].T
    w = rotvec_from_rotmat(dR)
    ang = float(np.linalg.norm(w))
    axis = w / ang if ang > 1e-12 else np.array([1.0, 0.0, 0.0])
    geom_post = frames_at(sim)
    J_h = np.array(geom_post["canonical_ball_incoming_hand"], float)
    r_h = None
    L_h = None
    if r_imp is not None and first_hit_rel is not None:
        r_h = first_hit_rel["Rh"].T @ r_imp
        L_h = np.cross(r_h, J_h)
    return {
        "ok": True,
        "t_first_contact": ball.t_first_contact,
        "t_last_contact": ball.t_last_contact,
        "t_post": t_post,
        "p_rel_pre": pre["p_rel_h"].tolist(),
        "p_rel_post": post["p_rel_h"].tolist(),
        "R_rel_pre": pre["R_rel"].tolist(),
        "R_rel_post": post["R_rel"].tolist(),
        "delta_p_h": dp.tolist(),
        "delta_p_h_mm": (1e3 * dp).tolist(),
        "delta_p_norm_mm": float(1e3 * np.linalg.norm(dp)),
        "delta_R_h": dR.tolist(),
        "rotvec_h": w.tolist(),
        "rot_axis_h": axis.tolist(),
        "rot_angle_deg": float(np.degrees(ang)),
        "pre_nL": pre["nL"],
        "pre_nR": pre["nR"],
        "post_nL": post["nL"],
        "post_nR": post["nR"],
        "post_e_x_mm": 1e3 * float(post["e_x"]),
        "post_tilt_deg": float(post["tilt_deg"]),
        "contact_pos_world": None if contact_pos is None else contact_pos.tolist(),
        "r_com_to_contact_world": None if r_imp is None else r_imp.tolist(),
        "r_hand": None if r_h is None else r_h.tolist(),
        "J_hand": J_h.tolist(),
        "r_cross_J_hand": None if L_h is None else L_h.tolist(),
        "geom_at_post": geom_post,
        "pre_vel": pre["obj_linvel"].tolist(),
        "post_vel": post["obj_linvel"].tolist(),
        "note": "delta_R = R_rel_post @ R_rel_pre.T so R_rel_post = delta_R @ R_rel_pre",
    }


def classify_run(log, t_tele, t0_ok, dropped, lift_done) -> str:
    if not t0_ok:
        return "INVALID_CONSTRUCTION"
    after = [r for r in log if r["t"] >= t_tele - 1e-9]
    t_uni = t_both = t_drop = None
    explode = False
    for r in after:
        if t_uni is None and ((r["nL"] == 0) ^ (r["nR"] == 0)):
            t_uni = r["t"]
        if t_both is None and r["nL"] == 0 and r["nR"] == 0:
            t_both = r["t"]
        if t_drop is None and (r["obj_z"] < TABLE_DROP or r["clear"] < -0.005):
            t_drop = r["t"]
        if r["v_rel"] > EXPLODE_V:
            explode = True
    t_fail = None
    for cand in (t_both, t_drop):
        if cand is not None:
            t_fail = cand if t_fail is None else min(t_fail, cand)
    delay = None if t_fail is None else float(t_fail - t_tele)
    if explode and (delay is None or delay < IMMEDIATE_S):
        return "IMMEDIATE_FAILURE"
    if t_fail is not None and delay is not None:
        if delay < IMMEDIATE_S:
            return "IMMEDIATE_FAILURE"
        if DELAYED_LO <= delay:
            return "DELAYED_TASK_FAILURE"
        return "IMMEDIATE_FAILURE"
    if lift_done and not dropped:
        # surviving but messy?
        max_tilt = max(r["tilt_deg"] for r in after) if after else 0.0
        max_e = max(abs(r["e_x"]) for r in after) if after else 0.0
        if t_uni is not None or max_tilt > 15.0 or max_e > 0.008:
            return "MARGINAL_SURVIVES"
        return "STABLE_DISTURBED"
    if dropped:
        delay = float((t_drop or log[-1]["t"]) - t_tele)
        return "DELAYED_TASK_FAILURE" if delay >= DELAYED_LO else "IMMEDIATE_FAILURE"
    return "MARGINAL_SURVIVES"


def t0_valid(after: dict, dbg: dict) -> tuple[bool, str]:
    nL, nR = after["nL"], after["nR"]
    if nL == 0 and nR == 0:
        return False, "both contacts absent at t=0+"
    md = dbg.get("min_dist")
    if md is not None and md < -PEN_REJECT:
        return False, f"gross penetration min_dist={md:.4f}"
    # object still near fingers: |e_x|,|e_y| not tens of cm
    rh = after["p_rel_h"]
    if abs(rh[0]) > 0.04 or abs(rh[1]) > 0.04:
        return False, "object teleported far from pinch"
    return True, "ok"


def zero_continue(
    sim,
    cfg,
    *,
    viewer,
    pacer,
    s,
    log,
    t_tele,
    mode: str = "ZERO",
    diagnostic_hold: float = 2.0,
    p0z: float | None = None,
) -> dict:
    """Nominal lift then diagnostic hold. p0z is world-z of p_des at teleport resume."""
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    v_lift = float(cfg["fsm"]["lift_speed"])
    lift_dz = float(cfg["fsm"]["lift_offset_z"])
    if p0z is None:
        ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
        sim.fsm.p_des = ph.copy()
        p0z = float(sim.fsm.p_des[2])
    lift_v = np.array([0.0, 0.0, v_lift])
    t_lift_done = None
    t_hold0 = None
    t_drop = None
    passed_2s = False
    hold_s = float(diagnostic_hold)
    n_max = int(round((lift_dz / max(v_lift, 1e-9) + hold_s + 2.0) / dt))
    dense_until = t_tele + 0.50
    last = -1e9
    for _ in range(n_max):
        if t_lift_done is None:
            tick_vw(sim, lift_v, np.zeros(3), G_HOLD, gains)
        else:
            tick_vw(sim, np.zeros(3), np.zeros(3), G_HOLD, gains)
        o = physical_pack(sim)
        t = float(sim.data.time)
        if t_lift_done is None:
            phase = "continue_lift"
        elif t < float(t_hold0) + 2.0 - 1e-9:
            phase = "task_hold"
        else:
            phase = "diagnostic_hold"
        step_dt = LOG_DT if t <= dense_until else 0.0095
        if t - last >= step_dt - 1e-9:
            log.append(_row(sim, phase))
            last = t
        if t_drop is None and (sim.dropped() or float(o["obj_z"]) < TABLE_DROP):
            t_drop = t
        if t_lift_done is None and float(sim.fsm.p_des[2]) - p0z >= lift_dz - 1e-9:
            t_lift_done = t
            t_hold0 = t
        if (
            t_hold0 is not None
            and t_drop is None
            and t >= float(t_hold0) + 2.0 - 1e-9
        ):
            passed_2s = True
        if viewer is not None and pacer is not None and pacer.should_render(t):
            extra = None
            if str(mode).upper() in ("REPOSITION_REGRASP", "WRIST_RECOVERY_CYCLE"):
                extra = {
                    "minimal": True,
                    "wrist_act_deg": float(getattr(sim, "_wrist_act_deg", 0.0)),
                    "tau": float(o["tau"]),
                    "aperture": float(o["aperture"]),
                    "v_rel": float(o["v_rel"]),
                    "fail_phase": getattr(sim, "_fail_latch", None),
                }
            overlay(viewer, s, phase, t, mode=mode, extra=extra)
            viewer.sync()
            pacer.note_sync(t)
            pacer.wait_if_ahead(t)
            if hasattr(viewer, "is_running") and not viewer.is_running():
                break
        if t_drop is not None and t >= t_drop + 3.0:
            break
        if t_hold0 is not None and t >= t_hold0 + hold_s:
            break
    end = physical_pack(sim)
    hold_completed_2s = bool(
        t_hold0 is not None
        and t_drop is None
        and float(sim.data.time) >= float(t_hold0) + 2.0 - 1e-9
    )
    hold_completed_diag = bool(
        t_hold0 is not None
        and t_drop is None
        and float(sim.data.time) >= float(t_hold0) + hold_s - 1e-9
    )
    captured_end = int(end["nL"]) > 0 or int(end["nR"]) > 0
    return {
        "t_lift_done": t_lift_done,
        "t_drop": t_drop,
        "t_hold0": t_hold0,
        "p0z": p0z,
        "diagnostic_hold": hold_s,
        "lift_completed": bool(t_lift_done is not None),
        "hold_completed_2s": passed_2s,
        "hold_completed": hold_completed_diag,
        "drop": t_drop is not None,
        "end_nL": int(end["nL"]),
        "end_nR": int(end["nR"]),
        "end_e_x_mm": 1e3 * float(end["e_x"]),
        "end_e_z_mm": 1e3 * float(end["rh"][2]) if "rh" in end else 1e3 * float(physical_pack(sim)["rh"][2]),
        "task_success_2s": bool(
            t_lift_done is not None and passed_2s
        ),
        "task_success": bool(
            t_lift_done is not None and hold_completed_diag and t_drop is None and captured_end
        ),
    }


def observe_common(sim, cfg, delay, *, viewer, pacer, s, log, mode, p0z, t_tele, diagnostic_hold):
    """Identical ZERO lift ticks for `delay` seconds. No extra freeze."""
    if delay <= 0:
        return {"t_observe_end": float(sim.data.time), "n_steps": 0, "p0z": p0z}
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    v_lift = float(cfg["fsm"]["lift_speed"])
    lift_dz = float(cfg["fsm"]["lift_offset_z"])
    lift_v = np.array([0.0, 0.0, v_lift])
    n = int(round(float(delay) / dt))
    t_lift_done = None
    for _ in range(n):
        tick_vw(sim, lift_v, np.zeros(3), G_HOLD, gains)
        t = float(sim.data.time)
        log.append(_row(sim, "observe"))
        if t_lift_done is None and float(sim.fsm.p_des[2]) - p0z >= lift_dz - 1e-9:
            t_lift_done = t
        if viewer is not None and pacer is not None and pacer.should_render(t):
            overlay(viewer, s, "observe", t, mode=mode)
            viewer.sync()
            pacer.note_sync(t)
            pacer.wait_if_ahead(t)
    return {"t_observe_end": float(sim.data.time), "n_steps": n, "p0z": p0z, "t_lift_done": t_lift_done}


def signed_rot_about_y_deg(R0: np.ndarray, R1: np.ndarray) -> float:
    """Signed rotation of R1 relative to R0 about the first-frame y axis (deg)."""
    d = np.asarray(R0, float).reshape(3, 3).T @ np.asarray(R1, float).reshape(3, 3)
    return float(np.degrees(np.arctan2(d[0, 2], d[0, 0])))


def run_continuous_wrist(
    sim,
    cfg,
    *,
    w_hy: float,
    max_deg: float,
    tau: float,
    viewer,
    pacer,
    s,
    log,
    mode: str,
) -> dict:
    """Slow legal w_hy sweep. v=0. No lift. Does not teleport orientation."""
    legal = float(RECOVERY4D_W_HY_MAX)
    w_hy = float(np.clip(w_hy, -legal, legal))
    if abs(w_hy) < 1e-9:
        raise ValueError("w_hy must be nonzero")
    freeze_targets(sim)
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    Rd0 = np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy()
    start = physical_pack(sim)
    t0 = float(sim.data.time)
    missing = []
    if int(start["nL"]) == 0:
        missing.append("L")
    if int(start["nR"]) == 0:
        missing.append("R")
    max_rad = abs(float(np.radians(max_deg)))
    n = int(round(max_rad / abs(w_hy) / dt))
    cmd = 0.0
    first_missing_touch = None
    first_bilat = None
    persist_s = 0.0
    persist_start = None
    first_persist = None
    t_drop = None
    for _ in range(max(n, 0)):
        Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        w_w = Rd @ np.array([0.0, w_hy, 0.0])
        tick_vw(sim, np.zeros(3), w_w, tau, gains)
        cmd += w_hy * dt
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        sim._wrist_cmd_deg = float(np.degrees(cmd))
        sim._wrist_des_deg = signed_rot_about_y_deg(Rd0, Rd)
        sim._wrist_act_deg = signed_rot_about_y_deg(Rh0, Rh)
        r = _row(sim, "wrist_sweep")
        log.append(r)
        nL, nR = int(r["nL"]), int(r["nR"])
        if t_drop is None and (r["obj_z"] < TABLE_DROP or r["clear"] < -0.005):
            t_drop = r["t"]
        if first_missing_touch is None:
            hit = False
            if "R" in missing and nR > 0:
                hit = True
            if "L" in missing and nL > 0:
                hit = True
            if (not missing) and nL > 0 and nR > 0:
                hit = False
            if missing and hit:
                first_missing_touch = {
                    "t": r["t"],
                    "wrist_cmd_deg": r["wrist_cmd_deg"],
                    "wrist_act_deg": r["wrist_act_deg"],
                    "nL": nL,
                    "nR": nR,
                    "Fn_L": r["Fn_L"],
                    "Fn_R": r["Fn_R"],
                    "e_x_mm": 1e3 * r["e_x"],
                    "e_y_mm": 1e3 * r["e_y"],
                    "e_z_mm": 1e3 * r["e_z"],
                }
        if nL > 0 and nR > 0:
            if persist_start is None:
                persist_start = r["t"]
            persist_s = r["t"] - persist_start
            if first_bilat is None:
                first_bilat = {
                    "t": r["t"],
                    "wrist_cmd_deg": r["wrist_cmd_deg"],
                    "wrist_act_deg": r["wrist_act_deg"],
                    "nL": nL,
                    "nR": nR,
                }
            if first_persist is None and persist_s >= 0.050 - 1e-9:
                first_persist = {
                    "t": r["t"],
                    "dt": persist_s,
                    "wrist_cmd_deg": r["wrist_cmd_deg"],
                    "wrist_act_deg": r["wrist_act_deg"],
                    "nL": nL,
                    "nR": nR,
                    "Fn_L": r["Fn_L"],
                    "Fn_R": r["Fn_R"],
                    "e_x_mm": 1e3 * r["e_x"],
                    "e_y_mm": 1e3 * r["e_y"],
                    "e_z_mm": 1e3 * r["e_z"],
                }
        else:
            persist_start = None
            persist_s = 0.0
        phase = "wrist_sweep"
        if nL > 0 and nR > 0:
            phase = "capture" if persist_s >= 0.050 else "contact"
        if viewer is not None and pacer is not None and pacer.should_render(r["t"]):
            overlay(
                viewer,
                s,
                phase,
                r["t"],
                mode=mode,
                extra={
                    "wrist_cmd_deg": r["wrist_cmd_deg"],
                    "wrist_act_deg": r["wrist_act_deg"],
                    "nL": nL,
                    "nR": nR,
                    "Fn_L": r["Fn_L"],
                    "Fn_R": r["Fn_R"],
                },
            )
            viewer.sync()
            pacer.note_sync(r["t"])
            pacer.wait_if_ahead(r["t"])
        if t_drop is not None:
            break
    freeze_targets(sim)
    return {
        "w_hy": w_hy,
        "w_hy_legal_max": legal,
        "max_deg": float(max_deg),
        "tau": float(tau),
        "t_start": t0,
        "t_end": float(sim.data.time),
        "start_nL": int(start["nL"]),
        "start_nR": int(start["nR"]),
        "missing_at_start": missing,
        "wrist_cmd_end_deg": float(getattr(sim, "_wrist_cmd_deg", 0.0)),
        "wrist_act_end_deg": float(getattr(sim, "_wrist_act_deg", 0.0)),
        "wrist_des_end_deg": float(getattr(sim, "_wrist_des_deg", 0.0)),
        "first_missing_finger_touch": first_missing_touch,
        "first_bilateral": first_bilat,
        "first_persistent_bilateral_50ms": first_persist,
        "t_drop": t_drop,
        "tracking_err_end_deg": abs(
            float(getattr(sim, "_wrist_des_deg", 0.0)) - float(getattr(sim, "_wrist_act_deg", 0.0))
        ),
    }


def _dropped_row(r) -> bool:
    return float(r["obj_z"]) < TABLE_DROP or float(r["clear"]) < -0.005


def _sync_wrist(sim, Rh0, Rd0, cmd_rad: float) -> None:
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    sim._wrist_cmd_deg = float(np.degrees(cmd_rad))
    sim._wrist_des_deg = signed_rot_about_y_deg(Rd0, Rd)
    sim._wrist_act_deg = signed_rot_about_y_deg(Rh0, Rh)


def _overlay_row(viewer, pacer, s, phase, row, mode):
    if viewer is None or pacer is None:
        return
    t = float(row["t"])
    if not pacer.should_render(t):
        return
    overlay(
        viewer,
        s,
        phase,
        t,
        mode=mode,
        extra={
            "minimal": True,
            "wrist_act_deg": row.get("wrist_act_deg", 0.0),
            "tau": row.get("tau", row.get("grip_ctrl", 0.0)),
            "aperture": row.get("aperture", 0.0),
            "v_rel": row.get("v_rel"),
            "d_in_mm": row.get("d_in_mm"),
            "fail_phase": row.get("fail_latch"),
        },
    )
    viewer.sync()
    pacer.note_sync(t)
    pacer.wait_if_ahead(t)


def _boundary(sim, name: str) -> dict:
    r = _row(sim, name)
    return {
        "name": name,
        "t": r["t"],
        "w_cmd": r["w_cmd"],
        "v_cmd": r["v_cmd"],
        "ori_err_deg": r["ori_err_deg"],
        "arm_qvel_norm": float(np.linalg.norm(r["arm_qvel"])),
        "hand_angvel_norm": float(np.linalg.norm(r["hand_angvel"])),
        "p_des": r["p_des"],
        "r_des": r["r_des"][:3],
    }


def snap_object(sim) -> dict:
    r = _row(sim, "snap")
    return {
        "t": r["t"],
        "e_mm": [1e3 * r["e_x"], 1e3 * r["e_y"], 1e3 * r["e_z"]],
        "cyl_axis_h": r["cyl_axis_h"],
        "g_h": r.get("g_h"),
        "nL": r["nL"],
        "nR": r["nR"],
        "Fn_L": r["Fn_L"],
        "Fn_R": r["Fn_R"],
        "v_rel": r["v_rel"],
        "w_rel": r["w_rel"],
        "v_rel_hand": r["v_rel_hand"],
        "omega_rel_hand": r["omega_rel_hand"],
        "aperture": r["aperture"],
        "contacts_h": r.get("contacts_h"),
        "wrist_act_deg": r.get("wrist_act_deg"),
        "wrist_des_deg": r.get("wrist_des_deg"),
        "d_in": r.get("d_in"),
        "d_in_mm": r.get("d_in_mm"),
        "u_in_h": r.get("u_in_h"),
        "from_plus_x_rim_mm": r.get("from_plus_x_rim_mm"),
        "from_distal_tip_mm": r.get("from_distal_tip_mm"),
        "obj_z": r["obj_z"],
        "tau": r["tau"],
    }


def run_wrist_recovery_cycle(
    sim,
    cfg,
    *,
    viewer,
    pacer,
    s,
    log,
    mode: str,
    omega: float,
    theta_capture_deg: float,
    tau: float,
    secure_s: float,
    vertical_s: float,
    stop_before_lift: bool,
    diagnostic_hold: float,
    p0z: float,
    t_tele: float,
) -> dict:
    """Outbound +hy to theta_capture, secure, return vertical, stabilize, optional lift.

    Stop rotation by zeroing w and leaving r_des as last integrated target (no snap).
    """
    legal = float(RECOVERY4D_W_HY_MAX)
    omega = float(np.clip(abs(omega), 1e-6, legal))
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.p_des = ph.copy()
    sim.fsm.v_cmd[:] = 0.0
    sim.fsm.w_cmd[:] = 0.0
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    Rd0 = np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy()
    pre = snap_object(sim)
    cmd = 0.0
    t_drop = None
    fail = None
    bounds = {"rotate_start": _boundary(sim, "rotate_start")}

    def tick_hy(w_hy, phase):
        nonlocal cmd, t_drop
        Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        w_w = Rd @ np.array([0.0, w_hy, 0.0])
        tick_vw(sim, np.zeros(3), w_w, tau, gains)
        cmd += w_hy * dt
        _sync_wrist(sim, Rh0, Rd0, cmd)
        r = _row(sim, phase)
        log.append(r)
        if t_drop is None and _dropped_row(r):
            t_drop = r["t"]
        _overlay_row(viewer, pacer, s, phase, r, mode)
        return r

    def hold_zero(n, phase):
        nonlocal t_drop
        last = None
        for _ in range(max(n, 0)):
            tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)
            _sync_wrist(sim, Rh0, Rd0, cmd)
            last = _row(sim, phase)
            log.append(last)
            if t_drop is None and _dropped_row(last):
                t_drop = last["t"]
            _overlay_row(viewer, pacer, s, phase, last, mode)
            if t_drop is not None:
                break
        return last

    # A. rotate to capture (actual angle)
    n_out = int(round(np.radians(abs(theta_capture_deg) + 15.0) / omega / dt))
    reached = False
    for _ in range(n_out):
        r = tick_hy(omega, "rotate_to_capture")
        if t_drop is not None:
            fail = "OUTBOUND_FAILURE"
            break
        if abs(float(r["wrist_act_deg"])) >= abs(float(theta_capture_deg)) - 0.15:
            reached = True
            break
    bounds["capture_stop_pre_w0"] = _boundary(sim, "capture_stop_pre_w0")
    hold_zero(1, "rotate_to_capture")  # first zero-w tick; r_des unchanged
    bounds["capture_stop_post_w0"] = _boundary(sim, "capture_stop_post_w0")
    if fail is None and not reached:
        fail = "OUTBOUND_FAILURE"
    at_cap = snap_object(sim)

    # B. secure 0.5 s
    if fail is None:
        n_sec = int(round(float(secure_s) / dt))
        hold_zero(n_sec, "secure")
        if t_drop is not None:
            fail = "CAPTURE_FAILURE"
    after_secure = snap_object(sim)

    # C. return to vertical
    bounds["return_start"] = _boundary(sim, "return_start")
    if fail is None:
        n_ret = int(round(np.radians(abs(theta_capture_deg) + 20.0) / omega / dt))
        for _ in range(n_ret):
            r = tick_hy(-omega, "return_vertical")
            if t_drop is not None:
                fail = "RETURN_FAILURE"
                break
            if float(r["wrist_act_deg"]) <= 1.0:
                break
        bounds["return_stop_pre_w0"] = _boundary(sim, "return_stop_pre_w0")
        hold_zero(1, "return_vertical")
        bounds["return_stop_post_w0"] = _boundary(sim, "return_stop_post_w0")
        if fail is None and t_drop is not None:
            fail = "RETURN_FAILURE"
    after_return = snap_object(sim)

    # D. vertical stabilize
    if fail is None:
        n_v = int(round(float(vertical_s) / dt))
        hold_zero(n_v, "vertical_stabilize")
        if t_drop is not None:
            fail = "VERTICAL_FAILURE"
    after_vert = snap_object(sim)

    lift = {
        "t_lift_done": None,
        "t_drop": t_drop,
        "lift_completed": False,
        "drop": t_drop is not None,
        "hold_completed": False,
        "hold_completed_2s": False,
        "task_success": False,
        "task_success_2s": False,
        "diagnostic_hold": float(diagnostic_hold),
    }
    if fail is None and not stop_before_lift:
        bounds["lift_resume"] = _boundary(sim, "lift_resume")
        ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
        sim.fsm.p_des = ph.copy()
        p0z_lift = float(sim.fsm.p_des[2])
        lift = zero_continue(
            sim,
            cfg,
            viewer=viewer,
            pacer=pacer,
            s=s,
            log=log,
            t_tele=t_tele,
            mode=mode,
            diagnostic_hold=float(diagnostic_hold),
            p0z=p0z_lift,
        )
        if lift.get("drop"):
            t_drop = lift.get("t_drop")
            if lift.get("task_success_2s"):
                fail = "DELAYED_FAILURE"
            elif lift.get("lift_completed"):
                fail = "LIFT_FAILURE"
            else:
                fail = "LIFT_FAILURE"
        t_drop = lift.get("t_drop", t_drop)

    if fail is None and t_drop is None and not stop_before_lift:
        if lift.get("task_success"):
            fail = None
        elif lift.get("task_success_2s"):
            fail = "DELAYED_FAILURE"

    return {
        "theta_capture_deg": float(theta_capture_deg),
        "omega": omega,
        "tau": tau,
        "reached_capture": reached,
        "fail_phase": fail,
        "t_drop": t_drop,
        "pre_recovery": pre,
        "at_capture": at_cap,
        "after_secure": after_secure,
        "after_return": after_return,
        "after_vertical": after_vert,
        "boundaries": bounds,
        "stop_before_lift": bool(stop_before_lift),
        "lift": lift,
        "wrist_act_end_deg": float(getattr(sim, "_wrist_act_deg", 0.0)),
        "wrist_des_end_deg": float(getattr(sim, "_wrist_des_deg", 0.0)),
        "wrist_cmd_end_deg": float(getattr(sim, "_wrist_cmd_deg", 0.0)),
    }


def run_reposition_regrasp_cycle(
    sim,
    cfg,
    *,
    viewer,
    pacer,
    s,
    log,
    mode: str,
    omega_out: float,
    omega_return: float,
    theta_capture_deg: float,
    tau_secure: float,
    tau_reposition: float,
    reposition_s: float,
    capture_verify_s: float,
    vertical_s: float,
    diagnostic_hold: float,
    t_tele: float,
    fail_continue_s: float = 3.0,
    do_return: bool = True,
    do_lift: bool = True,
    skip_unload: bool = False,
    secure_hold_s: float = 0.0,
    inward_slide: bool = False,
) -> dict:
    """Rotate, partial-unload reposition, regrasp, return, lift. Always includes lift."""
    legal = float(RECOVERY4D_W_HY_MAX)
    omega_out = float(np.clip(abs(omega_out), 1e-6, legal))
    omega_return = float(np.clip(abs(omega_return), 1e-6, legal))
    lo, hi = float(sim.ids.ctrl_low[7]), float(sim.ids.ctrl_high[7])
    tau_secure = float(np.clip(tau_secure, lo, hi))
    tau_reposition = float(np.clip(tau_reposition, lo, hi))
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.p_des = ph.copy()
    sim.fsm.v_cmd[:] = 0.0
    sim.fsm.w_cmd[:] = 0.0
    sim._fail_latch = None
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    Rd0 = np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy()
    pre = snap_object(sim)
    cmd = 0.0
    t_drop = None
    fail = None

    def tick_hy(w_hy, phase, tau):
        nonlocal cmd, t_drop
        Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        w_w = Rd @ np.array([0.0, w_hy, 0.0])
        tick_vw(sim, np.zeros(3), w_w, tau, gains)
        cmd += w_hy * dt
        _sync_wrist(sim, Rh0, Rd0, cmd)
        r = _row(sim, phase)
        log.append(r)
        if t_drop is None and _dropped_row(r):
            t_drop = r["t"]
        _overlay_row(viewer, pacer, s, phase, r, mode)
        return r

    def hold_tau(n, phase, tau):
        nonlocal t_drop
        last = None
        for _ in range(max(n, 0)):
            tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)
            _sync_wrist(sim, Rh0, Rd0, cmd)
            last = _row(sim, phase)
            log.append(last)
            if t_drop is None and _dropped_row(last):
                t_drop = last["t"]
            _overlay_row(viewer, pacer, s, phase, last, mode)
            if t_drop is not None:
                break
            if phase == "reposition" and int(last["nL"]) == 0 and int(last["nR"]) == 0:
                t_drop = last["t"]
                break
        return last

    def fail_watch(reason: str, tau_hold: float) -> dict:
        """Keep physics running after a latched failure. Does not change fail_phase."""
        sim._fail_latch = reason
        n = int(round(max(float(fail_continue_s), 0.0) / dt))
        extra = int(round(12.0 / dt)) if viewer is not None else 0
        mj_err = None
        for i in range(n + extra):
            try:
                tick_vw(sim, np.zeros(3), np.zeros(3), tau_hold, gains)
            except Exception as e:
                mj_err = f"{type(e).__name__}: {e}"
                break
            _sync_wrist(sim, Rh0, Rd0, cmd)
            r = _row(sim, "failure_watch")
            log.append(r)
            _overlay_row(viewer, pacer, s, "failure_watch", r, mode)
            if viewer is None and i + 1 >= n:
                break
            if viewer is not None and i + 1 >= n:
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    break
        return {"mujoco_error": mj_err}

    n_out = int(round(np.radians(abs(theta_capture_deg) + 20.0) / omega_out / dt))
    reached = False
    for _ in range(n_out):
        r = tick_hy(omega_out, "rotate_to_capture", tau_secure)
        if t_drop is not None:
            fail = "OUTBOUND_FAILURE"
            break
        if abs(float(r["wrist_act_deg"])) >= abs(float(theta_capture_deg)) - 0.15:
            reached = True
            break
    hold_tau(1, "rotate_to_capture", tau_secure)
    if fail is None and not reached:
        fail = "OUTBOUND_FAILURE"
    at_cap = snap_object(sim)

    after_repos = at_cap
    after_hold = at_cap
    last_tau = tau_secure
    if fail is None and skip_unload:
        freeze_targets(sim)
        n_sh = int(round(float(secure_hold_s) / dt))
        slide_phase = "inward_slide" if inward_slide else "secure_hold"
        hold_tau(n_sh, slide_phase, tau_secure)
        if t_drop is not None:
            fail = "CAPTURE_FAILURE"
        after_hold = snap_object(sim)
        after_repos = after_hold
    elif fail is None:
        n_rp = int(round(float(reposition_s) / dt))
        last_tau = tau_reposition
        hold_tau(n_rp, "reposition", tau_reposition)
        if t_drop is not None:
            fail = "REPOSITION_FAILURE"
        after_repos = snap_object(sim)

    after_regrasp = after_repos
    after_verify = after_repos
    if fail is None and not skip_unload:
        n_rg = int(round(0.10 / dt))
        last_tau = tau_secure
        hold_tau(n_rg, "regrasp", tau_secure)
        n_cv = int(round(float(capture_verify_s) / dt))
        hold_tau(n_cv, "capture_verify", tau_secure)
        if t_drop is not None:
            fail = "CAPTURE_FAILURE"
        after_regrasp = snap_object(sim)
        after_verify = after_regrasp

    after_return = after_verify
    if fail is None and do_return:
        n_ret = int(round(np.radians(abs(theta_capture_deg) + 25.0) / omega_return / dt))
        for _ in range(n_ret):
            r = tick_hy(-omega_return, "return_vertical", tau_secure)
            if t_drop is not None:
                fail = "RETURN_FAILURE"
                break
            if float(r["wrist_act_deg"]) <= 1.0:
                break
        hold_tau(1, "return_vertical", tau_secure)
        after_return = snap_object(sim)

    after_vert = after_return
    if fail is None and do_return:
        freeze_targets(sim)
        n_v = int(round(float(vertical_s) / dt))
        hold_tau(n_v, "vertical_hold", tau_secure)
        if t_drop is not None:
            fail = "VERTICAL_FAILURE"
        after_vert = snap_object(sim)

    lift = {
        "t_lift_done": None,
        "t_drop": t_drop,
        "lift_completed": False,
        "drop": t_drop is not None,
        "hold_completed": False,
        "hold_completed_2s": False,
        "task_success": False,
        "task_success_2s": False,
        "diagnostic_hold": float(diagnostic_hold),
    }
    if fail is None and do_lift:
        last_tau = tau_secure
        ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
        sim.fsm.p_des = ph.copy()
        lift = zero_continue(
            sim,
            cfg,
            viewer=viewer,
            pacer=pacer,
            s=s,
            log=log,
            t_tele=t_tele,
            mode=mode,
            diagnostic_hold=float(diagnostic_hold),
            p0z=float(sim.fsm.p_des[2]),
        )
        if lift.get("drop"):
            t_drop = lift.get("t_drop")
            fail = "DELAYED_FAILURE" if lift.get("task_success_2s") else "LIFT_FAILURE"

    viewer_end_t = float(sim.data.time)
    fail_event_t = t_drop
    continuation = None
    if fail is not None and float(fail_continue_s) > 0:
        t0 = float(sim.data.time)
        watch = fail_watch(fail, last_tau)
        continuation = {
            "enabled": True,
            "fail_event_t": fail_event_t,
            "continue_start_t": t0,
            "continue_end_t": float(sim.data.time),
            "requested_s": float(fail_continue_s),
            "exit_reason": (
                "mujoco_exception_in_fail_watch"
                if watch.get("mujoco_error")
                else "fail_watch_completed_then_return"
            ),
            "mujoco_error": watch.get("mujoco_error"),
            "viewer_close_after_cycle": True,
            "main_loop_exited_because_of_failure": True,
            "note": (
                "Recovery phases stop at failure; physics continues fail_continue_s "
                "(plus extra interactive watch). Classification remains fail_phase."
            ),
        }
        viewer_end_t = float(sim.data.time)

    if viewer is not None and float(fail_continue_s) >= 0:
        extra = int(round(8.0 / dt))
        for i in range(extra):
            try:
                tick_vw(sim, np.zeros(3), np.zeros(3), last_tau, gains)
            except Exception:
                break
            _sync_wrist(sim, Rh0, Rd0, cmd)
            r = _row(sim, "failure_watch" if fail is not None else log[-1]["phase"] if log else "hold")
            log.append(r)
            _overlay_row(viewer, pacer, s, r["phase"], r, mode)
            if hasattr(viewer, "is_running") and not viewer.is_running():
                break
        viewer_end_t = float(sim.data.time)

    return {
        "theta_capture_deg": float(theta_capture_deg),
        "omega_out": omega_out,
        "omega_return": omega_return,
        "tau_secure": tau_secure,
        "tau_reposition": tau_reposition,
        "tau_range": [lo, hi],
        "tendon": {
            "secure": tau_secure,
            "reposition_partial_unload": tau_reposition,
            "open_recovery4d": float(RECOVERY4D_TAU_OPEN),
            "yaml_close": -18.0,
            "yaml_open": 8.0,
            "actuator_lo": lo,
            "actuator_hi": hi,
        },
        "reached_capture": reached,
        "fail_phase": fail,
        "t_drop": t_drop,
        "pre_recovery": pre,
        "at_capture": at_cap,
        "after_reposition": after_repos,
        "after_capture_verify": after_verify,
        "after_return": after_return,
        "after_vertical": after_vert,
        "after_secure_hold": after_hold,
        "viewer_end_t": viewer_end_t,
        "continuation": continuation,
        "lift": lift,
        "e_z_convention": (
            "rh = Rh.T@(po-ph); grasp_orientation hand-z is world-down, "
            "so +e_z is further along +hand-z = toward the fingertips. "
            "d_in is pad-interior depth from the outer-distal corner; "
            "it is not identically -e_x or -e_z."
        ),
        "inward_slide": bool(inward_slide),
        "slide_s": float(secure_hold_s) if skip_unload else 0.0,
    }


def build_parent() -> dict:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    parent = advance_to_parent(sim, cfg)
    parent["cfg"] = cfg
    return parent


def run_teleport_case(
    s: float,
    ref: dict,
    kind: str = "coupled",
    *,
    interactive=False,
    playback=0.5,
    parent=None,
    mode: str = "ZERO",
    schedule=None,
    recovery_delay: float = 0.0,
    diagnostic_hold: float = 2.0,
    wrist_sweep: dict | None = None,
    wrist_cycle: dict | None = None,
    wrist_reposition: dict | None = None,
) -> dict:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    dt = float(sim.model.opt.timestep)
    viewer = None
    viewer_cm = None
    pacer = None
    if interactive:
        import mujoco.viewer as mjviewer

        viewer_cm = mjviewer.launch_passive(sim.model, sim.data)
        viewer = viewer_cm.__enter__()
        apply_viewer_camera(viewer, _cam(sim))
        enable_viewer_flags(viewer, show_contact_points=False, show_contact_forces=False)
        pacer = RealtimePacer(playback, RENDER_HZ, dt)
        pacer.start(0.0)
        pacer.note_sync(0.0)
        viewer.sync()
        parent = advance_to_parent(sim, cfg, viewer=viewer, pacer=pacer, mode=mode)
    else:
        if parent is None:
            parent = advance_to_parent(sim, cfg, mode=mode)
        else:
            restore_replay(sim, parent["snap"])
            freeze(sim)
    if not parent.get("ok"):
        if viewer_cm:
            viewer_cm.__exit__(None, None, None)
        return {"ok": False, **parent, "s": s, "kind": kind}
    # Viewer-only wall-clock pause. No physics ticks (snapshot semantics).
    visual_pause(viewer, sim, s, "parent_hold", 0.40 if interactive else 0.0, mode=mode)

    dp, Rd = coupled_pose_from_ref(ref, s, kind)

    vel_before = rel_pack(sim)
    applied = apply_coupled_pose(sim, dp, Rd)
    after = applied["after"]
    dbg = contact_debug(sim)
    ok0, why = t0_valid(after, dbg)
    t_tele = float(sim.data.time)
    log = list(parent.get("pre_log") or [])
    log.append(_row(sim, "teleport"))
    events_meta = {
        "t_teleport": t_tele,
        "t_first_post_teleport_mj_step": None,
        "t_recovery_command_start": None,
        "t_first_r_des_change": None,
        "t_first_p_des_change": None,
        "t_first_nonzero_recovery_w": None,
        "t_first_right_contact": None,
        "t_first_bilateral": None,
    }
    r_des0 = np.asarray(sim.fsm.r_des, float).copy()
    p_des0 = np.asarray(sim.fsm.p_des, float).copy()
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.p_des = ph.copy()
    p0z = float(sim.fsm.p_des[2])
    cont = {"t_lift_done": None, "t_drop": None, "lift_completed": False, "drop": False}
    if ok0:
        if viewer is not None:
            overlay(viewer, s, "teleport", t_tele, mode=mode)
            viewer.sync()
        visual_pause(viewer, sim, s, "teleport", 0.35 if interactive else 0.0, mode=mode)
        obs = observe_common(
            sim,
            cfg,
            float(recovery_delay),
            viewer=viewer,
            pacer=pacer,
            s=s,
            log=log,
            mode=mode,
            p0z=p0z,
            t_tele=t_tele,
            diagnostic_hold=diagnostic_hold,
        )
        if log:
            for r in log:
                if r["t"] > t_tele + 1e-12:
                    events_meta["t_first_post_teleport_mj_step"] = r["t"]
                    break
        if wrist_reposition:
            events_meta["t_recovery_command_start"] = float(sim.data.time)
            cy = run_reposition_regrasp_cycle(
                sim,
                cfg,
                viewer=viewer,
                pacer=pacer,
                s=s,
                log=log,
                mode=mode,
                omega_out=float(wrist_reposition.get("omega_out", 1.2)),
                omega_return=float(wrist_reposition.get("omega_return", 0.8)),
                theta_capture_deg=float(wrist_reposition.get("theta_capture_deg", 60.0)),
                tau_secure=float(wrist_reposition.get("tau_secure", RECOVERY4D_TAU_SECURE)),
                tau_reposition=float(wrist_reposition.get("tau_reposition", -2.0)),
                reposition_s=float(wrist_reposition.get("reposition_s", 0.5)),
                capture_verify_s=float(wrist_reposition.get("capture_verify_s", 1.0)),
                vertical_s=float(wrist_reposition.get("vertical_s", 2.0)),
                diagnostic_hold=float(diagnostic_hold),
                t_tele=t_tele,
                fail_continue_s=float(wrist_reposition.get("fail_continue_s", 3.0)),
                do_return=bool(wrist_reposition.get("do_return", True)),
                do_lift=bool(wrist_reposition.get("do_lift", True)),
                skip_unload=bool(wrist_reposition.get("skip_unload", False)),
                secure_hold_s=float(wrist_reposition.get("secure_hold_s", 0.0)),
                inward_slide=bool(wrist_reposition.get("inward_slide", False)),
            )
            lift = cy.get("lift") or {}
            cont = {
                "t_lift_done": lift.get("t_lift_done"),
                "t_drop": cy.get("t_drop") or lift.get("t_drop"),
                "lift_completed": lift.get("lift_completed", False),
                "drop": bool(cy.get("t_drop") or lift.get("drop")),
                "hold_completed": lift.get("hold_completed", False),
                "hold_completed_2s": lift.get("hold_completed_2s", False),
                "task_success": lift.get("task_success", False),
                "task_success_2s": lift.get("task_success_2s", False),
                "observe": obs,
                "wrist_reposition": cy,
                "diagnostic_hold": lift.get("diagnostic_hold"),
            }
        elif wrist_cycle:
            events_meta["t_recovery_command_start"] = float(sim.data.time)
            cy = run_wrist_recovery_cycle(
                sim,
                cfg,
                viewer=viewer,
                pacer=pacer,
                s=s,
                log=log,
                mode=mode,
                omega=float(wrist_cycle.get("omega", 0.40)),
                theta_capture_deg=float(wrist_cycle.get("theta_capture_deg", 60.0)),
                tau=float(wrist_cycle.get("tau", G_HOLD)),
                secure_s=float(wrist_cycle.get("secure_s", 0.5)),
                vertical_s=float(wrist_cycle.get("vertical_s", 0.5)),
                stop_before_lift=bool(wrist_cycle.get("stop_before_lift", False)),
                diagnostic_hold=float(diagnostic_hold),
                p0z=p0z,
                t_tele=t_tele,
            )
            lift = cy.get("lift") or {}
            cont = {
                "t_lift_done": lift.get("t_lift_done"),
                "t_drop": cy.get("t_drop") or lift.get("t_drop"),
                "lift_completed": lift.get("lift_completed", False),
                "drop": bool(cy.get("t_drop") or lift.get("drop")),
                "hold_completed": lift.get("hold_completed", False),
                "hold_completed_2s": lift.get("hold_completed_2s", False),
                "task_success": lift.get("task_success", False),
                "task_success_2s": lift.get("task_success_2s", False),
                "observe": obs,
                "wrist_cycle": cy,
                "diagnostic_hold": lift.get("diagnostic_hold"),
            }
        elif wrist_sweep:
            events_meta["t_recovery_command_start"] = float(sim.data.time)
            sw = run_continuous_wrist(
                sim,
                cfg,
                w_hy=float(wrist_sweep["w_hy"]),
                max_deg=float(wrist_sweep.get("max_deg", 120.0)),
                tau=float(wrist_sweep.get("tau", G_HOLD)),
                viewer=viewer,
                pacer=pacer,
                s=s,
                log=log,
                mode=mode,
            )
            cont = {
                "t_lift_done": None,
                "t_drop": sw.get("t_drop"),
                "lift_completed": False,
                "drop": sw.get("t_drop") is not None,
                "hold_completed": False,
                "hold_completed_2s": False,
                "task_success": False,
                "task_success_2s": False,
                "observe": obs,
                "wrist_sweep": sw,
            }
            for r in log:
                if events_meta["t_first_r_des_change"] is None and float(np.linalg.norm(
                    np.asarray(r["r_des"]) - r_des0.reshape(-1)
                )) > 1e-8 and r["t"] > t_tele + 1e-12:
                    events_meta["t_first_r_des_change"] = r["t"]
                if events_meta["t_first_nonzero_recovery_w"] is None and float(np.linalg.norm(r["w_cmd"])) > 1e-8 and r["t"] > t_tele + 1e-12:
                    events_meta["t_first_nonzero_recovery_w"] = r["t"]
        elif schedule:
            events_meta["t_recovery_command_start"] = float(sim.data.time)
            ev = apply_schedule(
                sim, cfg, schedule, viewer=viewer, pacer=pacer, s=s, log=log, mode=mode
            )
            for r in log:
                if events_meta["t_first_r_des_change"] is None and float(np.linalg.norm(
                    np.asarray(r["r_des"]) - r_des0.reshape(-1)
                )) > 1e-8 and r["t"] > t_tele + 1e-12:
                    events_meta["t_first_r_des_change"] = r["t"]
                if events_meta["t_first_p_des_change"] is None and float(np.linalg.norm(
                    np.asarray(r["p_des"]) - p_des0
                )) > 1e-8 and r["t"] > t_tele + 1e-12:
                    events_meta["t_first_p_des_change"] = r["t"]
                if events_meta["t_first_nonzero_recovery_w"] is None and float(np.linalg.norm(r["w_cmd"])) > 1e-8 and r["t"] > t_tele + 1e-12:
                    events_meta["t_first_nonzero_recovery_w"] = r["t"]
            log.append(_row(sim, "nominal_resume"))
            cont = zero_continue(
                sim,
                cfg,
                viewer=viewer,
                pacer=pacer,
                s=s,
                log=log,
                t_tele=t_tele,
                mode=mode,
                diagnostic_hold=float(diagnostic_hold),
                p0z=p0z,
            )
            cont["recovery_events"] = ev
            cont["observe"] = obs
        else:
            cont = zero_continue(
                sim,
                cfg,
                viewer=viewer,
                pacer=pacer,
                s=s,
                log=log,
                t_tele=t_tele,
                mode=mode,
                diagnostic_hold=float(diagnostic_hold),
                p0z=p0z,
            )
            cont["observe"] = obs
            if events_meta["t_first_post_teleport_mj_step"] is None:
                for r in log:
                    if r["t"] > t_tele + 1e-12:
                        events_meta["t_first_post_teleport_mj_step"] = r["t"]
                        break
            for r in log:
                if events_meta["t_first_p_des_change"] is None and float(np.linalg.norm(
                    np.asarray(r["p_des"]) - p_des0
                )) > 1e-8 and r["t"] >= t_tele - 1e-12:
                    events_meta["t_first_p_des_change"] = r["t"]
                if events_meta["t_first_r_des_change"] is None and float(np.linalg.norm(
                    np.asarray(r["r_des"]) - r_des0.reshape(-1)
                )) > 1e-8 and r["t"] > t_tele + 1e-12:
                    events_meta["t_first_r_des_change"] = r["t"]
        for r in log:
            if r["t"] <= t_tele + 1e-12:
                continue
            if events_meta["t_first_post_teleport_mj_step"] is None:
                events_meta["t_first_post_teleport_mj_step"] = r["t"]
            if events_meta["t_first_right_contact"] is None and int(r["nR"]) > 0:
                events_meta["t_first_right_contact"] = r["t"]
            if events_meta["t_first_bilateral"] is None and int(r["nL"]) > 0 and int(r["nR"]) > 0:
                events_meta["t_first_bilateral"] = r["t"]
        cont["timeline"] = events_meta
        post_early = [r for r in log if r["t"] > t_tele + 1e-12][:40]
        if any(r["v_rel"] > EXPLODE_V for r in post_early):
            ok0, why = False, "explosive relative velocity after teleport"
            cls_force_invalid = True
        else:
            cls_force_invalid = False
    else:
        cls_force_invalid = True

    cls = classify_run(log, t_tele, ok0, cont["drop"], cont["lift_completed"])
    if wrist_sweep and ok0:
        cls = "WRIST_SWEEP_DIAGNOSTIC"
    if wrist_reposition and ok0:
        cls = str((cont.get("wrist_reposition") or {}).get("fail_phase") or "REPOSITION_REGRASP")
    if wrist_cycle and ok0:
        cls = str((cont.get("wrist_cycle") or {}).get("fail_phase") or "WRIST_CYCLE")
    if not ok0 or cls_force_invalid:
        cls = "INVALID_CONSTRUCTION"
    # events
    t_uni = t_both = t_drop = None
    for r in log:
        if r["t"] < t_tele - 1e-12:
            continue
        if t_uni is None and ((r["nL"] == 0) ^ (r["nR"] == 0)):
            t_uni = r["t"]
        if t_both is None and r["nL"] == 0 and r["nR"] == 0:
            t_both = r["t"]
        if t_drop is None and (r["obj_z"] < TABLE_DROP or r["clear"] < -0.005):
            t_drop = r["t"]
    w_s = rotvec_from_rotmat(Rd)
    ang = float(np.linalg.norm(w_s))
    out = {
        "ok": True,
        "s": s,
        "kind": kind,
        "classification": cls,
        "t0_valid": ok0,
        "t0_reason": why,
        "t_teleport": t_tele,
        "delta_p_h_mm": (1e3 * dp).tolist(),
        "rot_axis_h": (w_s / ang).tolist() if ang > 1e-12 else [0.0, 0.0, 0.0],
        "rot_angle_deg": float(np.degrees(ang)),
        "t0_nL": after["nL"],
        "t0_nR": after["nR"],
        "t0_Fn_L": after["Fn_L"],
        "t0_Fn_R": after["Fn_R"],
        "t0_p_rel_h_mm": (1e3 * after["p_rel_h"]).tolist(),
        "t0_tilt_deg": after["tilt_deg"],
        "t0_min_dist": dbg.get("min_dist"),
        "t0_contacts": dbg.get("contacts"),
        "t0_aperture": after["aperture"],
        "vel_before": vel_before["obj_linvel"].tolist(),
        "vel_after": after["obj_linvel"].tolist(),
        "w_before": vel_before["obj_angvel"].tolist(),
        "w_after": after["obj_angvel"].tolist(),
        "t_first_unilateral": t_uni,
        "t_both_lost": t_both,
        "t_drop": t_drop,
        "dt_uni": None if t_uni is None else float(t_uni - t_tele),
        "dt_both": None if t_both is None else float(t_both - t_tele),
        "dt_drop": None if t_drop is None else float(t_drop - t_tele),
        "lift_completed": cont.get("lift_completed"),
        "hold_completed": cont.get("hold_completed"),
        "task_success": cont.get("task_success"),
        "drop": cont.get("drop"),
        "timeline": cont.get("timeline"),
        "observe": cont.get("observe"),
        "task_success_2s": cont.get("task_success_2s"),
        "hold_completed_2s": cont.get("hold_completed_2s"),
        "diagnostic_hold": cont.get("diagnostic_hold"),
        "t_hold0": cont.get("t_hold0"),
        "t_lift_done": cont.get("t_lift_done"),
        "recovery_events": cont.get("recovery_events"),
        "wrist_sweep": cont.get("wrist_sweep"),
        "wrist_cycle": cont.get("wrist_cycle"),
        "wrist_reposition": cont.get("wrist_reposition"),
        "parent_geom": parent["geom"],
        "parent_p_rel_mm": (1e3 * parent["rel"]["p_rel_h"]).tolist(),
        "log": log,
    }
    if viewer_cm is not None:
        viewer_cm.__exit__(None, None, None)
    return out


def apply_schedule(sim, cfg, schedule, *, viewer, pacer, s, log, mode="RECOVERY") -> dict:
    """Legal recovery4d-style pulses. Live Rh for v_hx, live r_des for w_hy."""
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    t_start = float(sim.data.time)
    events = {"t_intervention_start": t_start, "stages": []}
    for st in schedule:
        name = str(st.get("name", "pulse"))
        dur = float(st.get("duration", 0.0))
        if st.get("freeze_targets"):
            freeze_targets(sim)
        n = int(round(dur / dt)) if dur > 0 else 0
        v_hx = float(st.get("v_hx", 0.0))
        w_hy = float(st.get("w_hy", 0.0))
        v_z = float(st.get("v_z", 0.0))
        tau = float(st.get("tau", G_HOLD))
        t_st = float(sim.data.time)
        for _ in range(max(n, 0)):
            Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
            Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
            v_w = Rh @ np.array([v_hx, 0.0, 0.0]) + np.array([0.0, 0.0, v_z])
            w_w = Rd @ np.array([0.0, w_hy, 0.0])
            tick_vw(sim, v_w, w_w, tau, gains)
            t = float(sim.data.time)
            log.append(_row(sim, name))
            if viewer is not None and pacer is not None and pacer.should_render(t):
                overlay(viewer, s, name, t, mode=mode)
                viewer.sync()
                pacer.note_sync(t)
                pacer.wait_if_ahead(t)
        events["stages"].append(
            {
                "name": name,
                "t_start": t_st,
                "t_end": float(sim.data.time),
                "v_hx": v_hx,
                "w_hy": w_hy,
                "v_z": v_z,
                "tau": tau,
                "duration": dur,
                "p_des_semantics": "p_des += v_world*dt; r_des integrates w_world"
                + ("; then freeze_targets p_des:=p_hand r_des:=R_hand v=0" if st.get("freeze_targets") else ""),
            }
        )
    freeze_targets(sim)
    events["t_intervention_end"] = float(sim.data.time)
    events["t_nominal_resume"] = float(sim.data.time)
    return events


def contact_events(log, t0: float) -> dict:
    t_uni = t_right = t_bilat = t_both = t_drop = None
    bilat_s = 0.0
    prev_t = t0
    in_bilat = False
    for r in log:
        t = float(r["t"])
        nL, nR = int(r["nL"]), int(r["nR"])
        if t_uni is None and ((nL == 0) ^ (nR == 0)):
            t_uni = t
        if t_right is None and nR > 0:
            t_right = t
        if t_bilat is None and nL > 0 and nR > 0:
            t_bilat = t
        if t_both is None and nL == 0 and nR == 0:
            t_both = t
        if t_drop is None and (r["obj_z"] < TABLE_DROP or r["clear"] < -0.005):
            t_drop = t
        now_b = nL > 0 and nR > 0
        if in_bilat and now_b:
            bilat_s += t - prev_t
        in_bilat = now_b
        prev_t = t
    return {
        "t_first_unilateral": t_uni,
        "t_first_right_contact": t_right,
        "t_first_bilateral": t_bilat,
        "t_both_lost": t_both,
        "t_drop": t_drop,
        "bilateral_duration_s": bilat_s,
        "dt_uni": None if t_uni is None else t_uni - t0,
        "dt_right": None if t_right is None else t_right - t0,
        "dt_bilat": None if t_bilat is None else t_bilat - t0,
        "dt_both": None if t_both is None else t_both - t0,
        "dt_drop": None if t_drop is None else t_drop - t0,
    }


def slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("log", "t0_contacts", "parent_geom")}


def _save_case(r: dict) -> None:
    s = float(r.get("s") or 0)
    kind = r.get("kind", "coupled")
    (LOG_DIR / f"{kind}_s{s:.2f}.json").write_text(
        json.dumps({"summary": slim(r), "log": r.get("log", [])}, indent=2, default=str),
        encoding="utf-8",
    )


def run_sweep(ref: dict) -> list[dict]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("building parent airborne snapshot ...", flush=True)
    parent = build_parent()
    if not parent.get("ok"):
        print("parent failed", parent)
        return [parent]
    (LOG_DIR / "parent_geom.json").write_text(
        json.dumps(
            {
                "geom": parent["geom"],
                "rel_p_mm": (1e3 * parent["rel"]["p_rel_h"]).tolist(),
                "nL": parent["rel"]["nL"],
                "nR": parent["rel"]["nR"],
                "t": parent["t"],
                "vel": parent["rel"]["obj_linvel"].tolist(),
                "w": parent["rel"]["obj_angvel"].tolist(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    ss = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    rows = []
    for s in ss:
        print(f"s={s:.3f} ...", flush=True)
        r = run_teleport_case(s, ref, "coupled", interactive=False, parent=parent)
        rows.append(r)
        print("  ", r.get("classification"), "t0", r.get("t0_valid"), "dt_drop", r.get("dt_drop"), "dt_both", r.get("dt_both"))
        _save_case(r)
    classes = [r.get("classification") for r in rows]
    stabilize = {"STABLE_DISTURBED", "MARGINAL_SURVIVES"}
    fail = {"DELAYED_TASK_FAILURE", "IMMEDIATE_FAILURE"}
    if all(c in stabilize for c in classes):
        for s in (5.0, 6.0, 8.0):
            print(f"extend s={s:.3f} ...", flush=True)
            r = run_teleport_case(s, ref, "coupled", interactive=False, parent=parent)
            rows.append(r)
            ss.append(s)
            _save_case(r)
            print("  ", r.get("classification"), "dt_drop", r.get("dt_drop"))
            classes = [x.get("classification") for x in rows]
            if r.get("classification") in fail:
                break
    trans = None
    for i in range(len(rows) - 1):
        if classes[i] in stabilize and classes[i + 1] in fail:
            trans = (float(rows[i]["s"]), float(rows[i + 1]["s"]))
            break
    extra = []
    if trans is not None:
        a, b = trans
        for s in np.linspace(a, b, 5)[1:-1]:
            s = float(np.round(s, 3))
            print(f"refine s={s:.3f} ...", flush=True)
            r = run_teleport_case(s, ref, "coupled", interactive=False, parent=parent)
            extra.append(r)
            _save_case(r)
            print("  ", r.get("classification"), "dt_drop", r.get("dt_drop"), "dt_both", r.get("dt_both"))
    allr = rows + extra
    allr.sort(key=lambda x: float(x.get("s") or 0))
    (LOG_DIR / "sweep.json").write_text(
        json.dumps([slim(r) for r in allr], indent=2, default=str), encoding="utf-8"
    )
    return allr


def pick_delayed(rows: list[dict]) -> dict | None:
    delayed = [r for r in rows if r.get("classification") == "DELAYED_TASK_FAILURE"]
    if not delayed:
        return None
    delayed.sort(key=lambda r: abs(float(r.get("dt_drop") or r.get("dt_both") or 9) - 0.5))
    return delayed[0]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--measure-impact", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--mode", choices=("zero", "recovery", "wrist_sweep", "wrist_recovery_cycle", "reposition_regrasp"), default="zero")
    p.add_argument("--case", default=None)
    p.add_argument("--s", type=float, default=None)
    p.add_argument("--kind", choices=("coupled", "translation", "rotation"), default="coupled")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--playback", type=float, default=None)
    p.add_argument("--playback-speed", dest="playback_speed", type=float, default=None)
    p.add_argument("--schedule-json", default=None)
    p.add_argument("--recovery-delay", type=float, default=0.0)
    p.add_argument("--diagnostic-hold", type=float, default=10.0)
    p.add_argument("--wrist-sign", type=float, default=1.0)
    p.add_argument("--wrist-omega", type=float, default=0.40)
    p.add_argument("--wrist-max-deg", type=float, default=120.0)
    p.add_argument("--wrist-tau", type=float, default=-18.0)
    p.add_argument("--theta-capture", type=float, default=60.0)
    p.add_argument("--stop-before-lift", action="store_true")
    p.add_argument("--with-lift", action="store_true")
    p.add_argument("--omega-return", type=float, default=0.8)
    p.add_argument("--reposition-tau", type=float, default=-2.0)
    p.add_argument("--reposition-s", type=float, default=0.5)
    p.add_argument("--fail-continue", type=float, default=3.0)
    p.add_argument("--secure-hold-only", action="store_true")
    p.add_argument("--secure-hold-s", type=float, default=2.0)
    p.add_argument("--inward-slide", action="store_true")
    p.add_argument("--slide-s", type=float, default=0.5)
    return p.parse_args(argv)


def load_or_measure_ref() -> dict:
    path = LOG_DIR / "impact_reference.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ref = measure_canonical_impact()
    path.write_text(json.dumps(ref, indent=2, default=str), encoding="utf-8")
    return ref


def main(argv=None) -> int:
    args = parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.measure_impact:
        ref = measure_canonical_impact()
        (LOG_DIR / "impact_reference.json").write_text(json.dumps(ref, indent=2, default=str), encoding="utf-8")
        print(json.dumps(ref, indent=2, default=str))
        return 0 if ref.get("ok") else 1
    ref = load_or_measure_ref()
    if not ref.get("ok"):
        print("impact reference failed", ref)
        return 1
    if args.sweep:
        rows = run_sweep(ref)
        print(json.dumps([slim(r) for r in rows], indent=2, default=str))
        pick = pick_delayed(rows)
        if pick:
            (LOG_DIR / "representative_delayed.json").write_text(
                json.dumps(slim(pick), indent=2, default=str), encoding="utf-8"
            )
            print("representative s", pick["s"], pick["classification"])
        return 0
    if args.ablate:
        s = args.s
        if s is None:
            p = LOG_DIR / "representative_delayed.json"
            if p.exists():
                s = float(json.loads(p.read_text(encoding="utf-8"))["s"])
            else:
                print("provide --s")
                return 2
        out = {}
        parent = build_parent()
        for kind in ("translation", "rotation", "coupled"):
            print("ablate", kind, "s", s, flush=True)
            r = run_teleport_case(float(s), ref, kind, interactive=False, parent=parent)
            out[kind] = slim(r)
            (LOG_DIR / f"ablate_{kind}_s{s:.2f}.json").write_text(
                json.dumps({"summary": slim(r), "log": r.get("log", [])}, indent=2, default=str),
                encoding="utf-8",
            )
        (LOG_DIR / "ablations.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(json.dumps(out, indent=2, default=str))
        return 0
    s = args.s
    if args.case == "coupled_boundary":
        s = S_COUPLED
        args.kind = "coupled"
    if s is None:
        print("provide --s or --case coupled_boundary")
        return 2
    playback = 0.5
    if args.playback_speed is not None:
        playback = float(args.playback_speed)
    elif args.playback is not None:
        playback = float(args.playback)
    schedule = None
    wrist_sweep = None
    wrist_cycle = None
    wrist_reposition = None
    mode = str(args.mode).upper()
    delay = float(args.recovery_delay)
    if mode == "REPOSITION_REGRASP":
        if delay <= 0:
            delay = 0.10
        wrist_reposition = {
            "omega_out": min(abs(float(args.wrist_omega if args.wrist_omega != 0.40 else 1.2)), float(RECOVERY4D_W_HY_MAX)),
            "omega_return": min(abs(float(args.omega_return)), float(RECOVERY4D_W_HY_MAX)),
            "theta_capture_deg": float(args.theta_capture),
            "tau_secure": float(RECOVERY4D_TAU_SECURE),
            "tau_reposition": float(args.reposition_tau),
            "reposition_s": float(args.reposition_s),
            "capture_verify_s": 1.0,
            "vertical_s": 2.0,
            "fail_continue_s": float(args.fail_continue),
            "do_return": (not bool(args.secure_hold_only)) or bool(args.inward_slide),
            "do_lift": bool(args.with_lift) if (args.secure_hold_only or args.inward_slide) else True,
            "skip_unload": bool(args.secure_hold_only) or bool(args.inward_slide),
            "secure_hold_s": float(args.slide_s if args.inward_slide else args.secure_hold_s),
            "inward_slide": bool(args.inward_slide),
        }
        if args.inward_slide:
            wrist_reposition["do_return"] = True
            wrist_reposition["do_lift"] = bool(args.with_lift)
            wrist_reposition["skip_unload"] = True
        if abs(float(args.wrist_omega) - 0.40) > 1e-9:
            wrist_reposition["omega_out"] = min(abs(float(args.wrist_omega)), float(RECOVERY4D_W_HY_MAX))
        else:
            wrist_reposition["omega_out"] = 1.2
        mode = "REPOSITION_REGRASP"
    elif mode == "WRIST_RECOVERY_CYCLE":
        if delay <= 0:
            delay = 0.10
        omega = min(abs(float(args.wrist_omega)), float(RECOVERY4D_W_HY_MAX))
        stop = bool(args.stop_before_lift)
        if (not args.headless) and (not args.with_lift):
            stop = True
        if args.with_lift:
            stop = False
        wrist_cycle = {
            "omega": omega,
            "theta_capture_deg": float(args.theta_capture),
            "tau": float(args.wrist_tau),
            "secure_s": 0.5,
            "vertical_s": 0.5,
            "stop_before_lift": stop,
        }
        mode = "WRIST_RECOVERY_CYCLE"
    elif mode == "WRIST_SWEEP":
        if delay <= 0:
            delay = 0.10
        sign = 1.0 if float(args.wrist_sign) >= 0 else -1.0
        omega = min(abs(float(args.wrist_omega)), float(RECOVERY4D_W_HY_MAX))
        wrist_sweep = {
            "w_hy": sign * omega,
            "max_deg": float(args.wrist_max_deg),
            "tau": float(args.wrist_tau),
        }
        mode = "WRIST_SWEEP"
    elif mode == "RECOVERY":
        sp = Path(args.schedule_json) if args.schedule_json else SEQ_JSON
        if sp.exists():
            schedule = json.loads(sp.read_text(encoding="utf-8"))["schedule"]
        else:
            schedule = [
                {"name": "wrist", "w_hy": 1.5, "tau": -18.0, "duration": 0.10},
                {"name": "stop", "duration": 0.0, "freeze_targets": True},
            ]
    r = run_teleport_case(
        float(s),
        ref,
        args.kind,
        interactive=not args.headless,
        playback=playback,
        mode=mode,
        schedule=schedule,
        recovery_delay=delay,
        diagnostic_hold=float(args.diagnostic_hold),
        wrist_sweep=wrist_sweep,
        wrist_cycle=wrist_cycle,
        wrist_reposition=wrist_reposition,
    )
    print(json.dumps(slim(r), indent=2, default=str))
    if str(args.mode).lower() == "reposition_regrasp":
        out_dir = ROOT / "results" / "diagnostics" / "reposition_regrasp"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "secure_hold_tau18" if args.secure_hold_only else f"unload_tau{int(args.reposition_tau)}"
        if args.inward_slide:
            tag = f"inward_slide_{float(args.slide_s):.2f}s"
            if args.with_lift:
                tag += "_lift"
        (out_dir / f"{tag}.json").write_text(
            json.dumps({"summary": slim(r), "log": r.get("log", [])}, indent=2, default=str),
            encoding="utf-8",
        )
        print("wrote", out_dir / f"{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
