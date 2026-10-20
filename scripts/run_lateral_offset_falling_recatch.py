"""Lateral-offset falling recatch from nominal grasp pose (not inverted catch).

Transfers open-overchase-close-brake mechanism to constructed free-fall ICs.
LATERAL_RECATCH_SUCCESS frozen before search. No SFT/GRPO. Not mixed into K49 experts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import V_AIRBORNE
from envs.initialization.grasp_seat import grasp_settled
from envs.outcomes.horizon_outcome import NOMINAL_S, latched
from envs.outcomes.nominal_pose import capture_ref
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import take_capture
from envs.simulation.grasp_quality import body_twist
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary, window_stats as window_stats_fo
from evaluation.grasp_configuration import measure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID, SFT, decode, mapping_checksum, write_csv, write_json,
)
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table
from training.recovery_episode import make_training_env
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results" / "diagnostics" / "lateral_offset_falling_recatch"

# Frozen criterion
HOLD_S = 0.50
HOLD_VZ_MAX = 0.10
DESCEND_VZ = -0.05
TABLE_Z = 0.42

# Budgets frozen before run
STAGE_A = 200   # overchase template + sanity
STAGE_B = 700   # 1–3 cm
STAGE_C = 500   # 4–8 cm if B succeeds
STAGE_D = 400   # robustness + handoff
TOTAL_BUDGET = STAGE_A + STAGE_B + STAGE_C + STAGE_D
WALL_LIMIT_S = 40 * 60

FINE_DOWN = np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0], np.float32)

# Proven overchase template (from vertical_recatch open_overchase_close_brake actions)
OVERCHASE_TEMPLATE = ROOT / "data" / "vertical_recatch" / "train" / "ball_train_0027__follow_early.npz"


def tokens_hash(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


def hand_pos(sim):
    return np.asarray(sim.data.xpos[sim.ids.hand_body], float).copy()


def obj_pos(sim):
    return np.asarray(sim.data.xpos[sim.ids.object_body], float).copy()


def body_v_w(sim, bid):
    v, w = body_twist(sim.model, sim.data, int(bid))
    return np.asarray(v, float), np.asarray(w, float)


CYL_RADIUS = 0.018  # object geom size[0]


def finger_frame(sim):
    pl = np.asarray(sim.data.xpos[sim.ids.left_body], float)
    pr = np.asarray(sim.data.xpos[sim.ids.right_body], float)
    ph = hand_pos(sim)
    mid = 0.5 * (pl + pr)
    gap = float(np.linalg.norm(pl - pr)) + 1e-12
    u_open = (pl - pr) / gap
    # Palm→fingertip approach (often ≈ −Z for nominal top grasp).
    u_approach = mid - ph
    u_approach = u_approach / (np.linalg.norm(u_approach) + 1e-12)
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    # Hand-frame: Rh[:,1]∥open, Rh[:,2]∥approach, Rh[:,0] = pad-length axis.
    # For vertical approach, only two horizontal axes exist; distal=+Rh[:,0],
    # perp=−Rh[:,0] (opposite sides of the hand).
    u_dist = np.asarray(Rh[:, 0], float)
    u_dist = u_dist / (np.linalg.norm(u_dist) + 1e-12)
    u_perp = -u_dist
    return {"pl": pl, "pr": pr, "ph": ph, "mid": mid, "gap": gap,
            "u_open": u_open, "u_dist": u_dist, "u_perp": u_perp,
            "u_approach": u_approach, "Rh": Rh}


def rot_angle(Ra, Rb):
    R = Ra.T @ Rb
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(c))


def object_table_contact(sim):
    tg = int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table"))
    obj = int(sim.ids.object_body)
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        og = int(c.geom2) if b1 == obj else int(c.geom1)
        if og == tg:
            return True
    return False


def contact_sites(sim):
    ff = finger_frame(sim)
    obj = int(sim.ids.object_body)
    left, right, hand = int(sim.ids.left_body), int(sim.ids.right_body), int(sim.ids.hand_body)
    out = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        pos = np.asarray(c.pos, float)
        if other == left:
            side = "L"
        elif other == right:
            side = "R"
        elif other == hand:
            side = "hand"
        else:
            side = "other"
        t = float(np.dot(pos - ff["pr"], ff["u_open"]))
        between = 0.0 < t < ff["gap"]
        site = "inside" if between and side in ("L", "R") else (
            "hand_palm" if side == "hand" else "outer_or_edge" if side in ("L", "R") else "other"
        )
        out.append({"side": side, "site": site, "between": int(between)})
    return out


def place_cylinder(sim, pos, quat):
    q = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    d = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    sim.data.qpos[q:q + 3] = np.asarray(pos, float)
    sim.data.qpos[q + 3:q + 7] = np.asarray(quat, float)
    sim.data.qvel[d:d + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def build_nominal_open_robot(env):
    """Lift to nominal grasp, then open jaws — keep nominal orientation (no inverted wrist)."""
    env.begin_nominal()
    grasp_settled(env.sim, env.gains)
    env.start_recovery()
    # open gripper without large wrist motion
    for _ in range(12):
        step_legal(env.sim, np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
    ff = finger_frame(env.sim)
    R0 = ff["Rh"].copy()
    mid0 = ff["mid"].copy()
    return take_capture(env), R0, mid0, ff


def make_ic(env, robot_cap, R0, mid0, direction, offset_m, root_id):
    """Place cylinder at ~grasp-center height, horizontally offset; verify outside jaws."""
    from envs.simulation.capture import restore_capture
    restore_capture(env, robot_cap)
    env.start_recovery()
    sim = env.sim
    ff = finger_frame(sim)
    # keep seated horizontal cylinder quat
    s = np.sqrt(0.5)
    quat = np.array([s, s, 0.0, 0.0], float)
    # Clearance baselines (empirically free of hand/finger mesh at z=mid).
    # offset_m is ADDITIONAL separation beyond that clear surface — preserves sweep.
    # mid±1–3cm alone still lies inside an ~8cm open gap / hand mesh.
    CLEAR = {"open": 0.12, "distal": 0.06, "perp": 0.06}
    if direction == "open":
        u = ff["u_open"]
    elif direction == "distal":
        u = ff["u_dist"]
    elif direction == "perp":
        u = ff["u_perp"]
    else:
        raise ValueError(direction)
    pos = mid0 + (CLEAR[direction] + float(offset_m)) * u
    pos = np.asarray(pos, float)
    pos[2] = float(mid0[2])  # match grasp-center height
    place_cylinder(sim, pos, quat)
    sim.data.qpos[sim.ids.finger_jnt] = 0.04
    mujoco.mj_forward(sim.model, sim.data)
    # safety nudge only if still contacting (should be rare with CLEAR)
    for _ in range(8):
        n_touch = 0
        obj_b = int(sim.ids.object_body)
        for i in range(int(sim.data.ncon)):
            c = sim.data.contact[i]
            b1 = int(sim.model.geom_bodyid[int(c.geom1)])
            b2 = int(sim.model.geom_bodyid[int(c.geom2)])
            if obj_b in (b1, b2):
                n_touch += 1
        if n_touch == 0:
            break
        pos = pos + 0.005 * u
        pos[2] = float(mid0[2])
        place_cylinder(sim, pos, quat)
        sim.data.qpos[sim.ids.finger_jnt] = 0.04
        mujoco.mj_forward(sim.model, sim.data)

    ff2 = finger_frame(sim)
    po = obj_pos(sim)
    t_along = float(np.dot(po - ff2["pr"], ff2["u_open"]))
    between_open = 0.0 < t_along < ff2["gap"]
    d_dist = abs(float(np.dot(po - ff2["mid"], ff2["u_dist"])))
    d_perp = abs(float(np.dot(po - ff2["mid"], ff2["u_perp"])))
    # outside bilateral grasp region
    if direction == "open":
        outside = not between_open
    elif direction == "distal":
        outside = d_dist >= 0.03 and (not between_open or d_dist >= 0.04)
    else:
        outside = d_perp >= float(offset_m) * 0.8

    # raw robot–object contact (any geom), not site labels only
    n_touch = 0
    obj_b = int(sim.ids.object_body)
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj_b in (b1, b2):
            n_touch += 1
    sep = float(np.linalg.norm(po - ff2["mid"]))
    sep_h = float(np.linalg.norm((ff2["Rh"].T @ (po - ff2["ph"]))[:2]))
    sep_along = float(np.dot(po - ff2["mid"], u))
    # fall check
    z0 = float(po[2])
    dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    for _ in range(3):
        mujoco.mj_step(sim.model, sim.data)
    vz = float(sim.data.qvel[dadr + 2])
    falling = vz < -0.02 and float(obj_pos(sim)[2]) < z0 - 1e-5
    ok = (
        n_touch == 0
        and outside
        and not object_table_contact(sim)
        and falling
    )
    # time to table without intervention
    z1 = float(obj_pos(sim)[2])
    t_impact = None
    qsave = np.array(sim.data.qpos, float).copy()
    vsave = np.array(sim.data.qvel, float).copy()
    t_s = float(sim.data.time)
    for i in range(400):
        mujoco.mj_step(sim.model, sim.data)
        if object_table_contact(sim) or float(obj_pos(sim)[2]) < TABLE_Z:
            t_impact = float(sim.data.time) - t_s
            break
    sim.data.qpos[:] = qsave
    sim.data.qvel[:] = vsave
    mujoco.mj_forward(sim.model, sim.data)

    env.start_recovery()
    try:
        ref = capture_ref(env.sim, env.switch)
    except Exception:
        ref = {"p": np.array(env.sim.fsm.p_des, float).copy(),
               "R": np.asarray(env.sim.fsm.r_des, float).reshape(3, 3).copy()}
    try:
        env._obs_seq = [np.asarray(env.observation(), np.float32)]
        env._prev_seq = [np.zeros(7, np.float32)]
        env._prev = np.zeros(7, np.float32)
    except Exception:
        pass
    # restore exact release state after table-time probe
    place_cylinder(sim, pos, quat)
    sim.data.qpos[sim.ids.finger_jnt] = 0.04
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(3):
        mujoco.mj_step(sim.model, sim.data)
    # zero? already falling slightly — keep as-is after 3 steps OR re-zero for clean release
    # Task: begin with zero vel then fall. Re-place with zero vel for IC snapshot.
    place_cylinder(sim, pos, quat)
    sim.data.qpos[sim.ids.finger_jnt] = 0.04
    mujoco.mj_forward(sim.model, sim.data)

    cap = take_capture(env)
    Rh = finger_frame(sim)["Rh"]
    return {
        "ic_id": f"{direction}_off{int(round(offset_m * 100)):02d}_{root_id}",
        "direction": direction,
        "offset_m": float(offset_m),
        "root_id": root_id,
        "release_pos": pos.tolist(),
        "hand_pos": hand_pos(sim).tolist(),
        "mid": mid0.tolist(),
        "sep_world_m": sep,
        "sep_hand_xy_m": sep_h,
        "sep_along_dir_m": sep_along,
        "clear_base_m": CLEAR[direction],
        "com_between_fingers": int(between_open),
        "outside_grasp_region": int(outside),
        "ok_init": int(ok),
        "falling_verified": int(falling),
        "n_touch_init": int(n_touch),
        "vz_probe": vz,
        "t_table_no_intervene_s": t_impact,
        "z_release": float(pos[2]),
        "R0": R0.tolist(),
        "capture": cap,
        "ref": ref,
    }


def restore_ic(env, ic):
    from envs.simulation.capture import restore_capture
    restore_capture(env, ic["capture"])
    env.start_recovery()
    try:
        if not env._obs_seq:
            env._obs_seq = [np.asarray(env.observation(), np.float32)]
            env._prev_seq = [np.zeros(7, np.float32)]
    except Exception:
        pass


def classify_wrist(max_ang_rad):
    deg = float(max_ang_rad) * 180.0 / math.pi
    if deg < 15:
        return "nominal_orientation_lateral"
    if deg < 45:
        return "moderate_wrist_adjustment"
    if deg < 90:
        return "large_wrist_adjustment"
    return "orientation_reversal_or_extreme"


def evaluate(env, ic, actions, check_handoff=False):
    restore_ic(env, ic)
    sim = env.sim
    R0 = np.asarray(ic["R0"], float).reshape(3, 3)
    t0 = float(sim.data.time)
    ph0 = hand_pos(sim)
    first_grasp = None
    first_vz = None
    first_geo = None
    bil_run = 0.0
    max_bil = 0.0
    saw_bil = False
    table_before = False
    escape = False
    max_wrist = 0.0
    max_hand_disp = 0.0
    hold_vz = []
    dt = 0.02
    timeline = []

    for ai, a in enumerate(actions):
        a = np.clip(np.asarray(a, np.float32).reshape(7), -1, 1)
        Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        max_wrist = max(max_wrist, rot_angle(R0, Rh))
        max_hand_disp = max(max_hand_disp, float(np.linalg.norm(hand_pos(sim) - ph0)))
        dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
        vz = float(sim.data.qvel[dadr + 2])
        cs = contact_summary(sim)
        any_f = int(cs["nL"]) + int(cs["nR"]) > 0
        if (object_table_contact(sim) or float(obj_pos(sim)[2]) < TABLE_Z) and not saw_bil:
            table_before = True
            break
        if any_f and first_grasp is None:
            first_grasp = float(sim.data.time) - t0
            first_vz = vz
            sites = contact_sites(sim)
            ff = finger_frame(sim)
            po = obj_pos(sim)
            t = float(np.dot(po - ff["pr"], ff["u_open"]))
            counts = {}
            for s in sites:
                counts[s["site"]] = counts.get(s["site"], 0) + 1
            first_geo = {
                "t": first_grasp, "vz": first_vz,
                "com_between": int(0.0 < t < ff["gap"]),
                "sites": sites,
                "dominant": max(counts, key=counts.get) if counts else "none",
            }
        if cs["bilateral"]:
            saw_bil = True
            bil_run += dt
            max_bil = max(max_bil, bil_run)
            hold_vz.append(abs(vz))
        else:
            # Drop only counts after a verified HOLD_S bilateral window
            # ("no subsequent drop during the verification interval").
            if max_bil + 1e-9 >= HOLD_S and not any_f:
                escape = True
            bil_run = 0.0
            hold_vz = []
        if ai % 2 == 0:
            timeline.append({
                "t": float(sim.data.time) - t0, "hand": hand_pos(sim).tolist(),
                "obj": obj_pos(sim).tolist(), "bil": int(cs["bilateral"]),
                "wrist_deg": float(max_wrist * 180 / math.pi),
            })
        step_legal(sim, a)
        env._prev = a.copy()
        if saw_bil and bil_run >= HOLD_S + 0.1:
            break

    descending = first_grasp is not None and first_vz is not None and first_vz <= DESCEND_VZ
    mean_hvz = float(np.mean(hold_vz)) if hold_vz else None
    nontrivial_motion = max_hand_disp >= 0.008  # ≥8 mm chase
    inside = bool(first_geo and (first_geo.get("com_between") or first_geo.get("dominant") == "inside"))
    success = bool(
        int(ic.get("ok_init", 0))
        and descending
        and nontrivial_motion
        and saw_bil
        and not table_before
        and max_bil + 1e-9 >= HOLD_S
        and mean_hvz is not None and mean_hvz <= HOLD_VZ_MAX
        and not escape
    )
    wrist_class = classify_wrist(max_wrist)
    out = {
        "LATERAL_RECATCH_SUCCESS": int(success),
        "ok_init": int(ic.get("ok_init", 0)),
        "descending_at_contact": int(descending),
        "nontrivial_hand_motion": int(nontrivial_motion),
        "had_bil": int(saw_bil),
        "max_bil_persist_s": max_bil,
        "mean_hold_vz": mean_hvz,
        "table_before_bil": int(table_before),
        "escape": int(escape),
        "t_first_grasp": first_grasp,
        "vz_at_first_grasp": first_vz,
        "max_wrist_deg": float(max_wrist * 180 / math.pi),
        "max_hand_disp_m": max_hand_disp,
        "wrist_class": wrist_class,
        "jaw_entry_ok": int(inside),
        "obj_z_end": float(obj_pos(sim)[2]),
        "SUCCESS_NEW": 0,
        "HANDOFF_OK": 0,
    }
    if success and check_handoff:
        try:
            apply_handoff(sim, ic["ref"])
            t_h = float(sim.data.time)
            samples = []
            next_s = 0.0
            reason = None
            while reason is None and float(sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
                nominal_step(sim)
                tr = float(sim.data.time) - t_h
                if tr >= next_s:
                    m = measure(sim, U_TGT, P_TGT)
                    cs = contact_summary(sim)
                    samples.append({
                        "t_rel": tr, "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
                        "bilateral": cs["bilateral"], "obj_z": m["obj_z"],
                    })
                    next_s += 0.02
                if latched(sim):
                    reason = "TABLE_DROP"
            win = window_stats_fo(samples, float(NOMINAL_S) - 2.0)
            out["SUCCESS_NEW"] = int(bool(win.get("working_SUCCESS_NEW_hold")))
            out["HANDOFF_OK"] = int(reason is None and float(obj_pos(sim)[2]) > TABLE_Z)
            out["win_mean_pos"] = win.get("mean_pos")
            out["win_mean_theta"] = win.get("mean_theta")
        except Exception as e:
            out["handoff_err"] = str(e)
    return out, first_geo, timeline


def overchase_template_actions():
    if OVERCHASE_TEMPLATE.exists():
        acts = np.asarray(np.load(OVERCHASE_TEMPLATE)["actions"], np.float32)
        return [acts[i].copy() for i in range(len(acts))]
    # fallback open-chase-close-brake
    acts = []
    for _ in range(4):
        acts.append(np.array([0.3, 0.0, -0.1, 0, 0, 0, -1], np.float32))
    for _ in range(8):
        acts.append(np.array([0.5, 0.0, -0.2, 0, 0, 0, -1], np.float32))
    for _ in range(6):
        acts.append(np.array([0.2, 0.0, -0.15, 0, 0, 0, 1], np.float32))
    for _ in range(16):
        acts.append(np.array([0.0, 0.0, -0.25, 0, 0, 0, 1], np.float32))
    return acts


def adapt_overchase(env, ic, p, template):
    """Adapt overchase: open, chase toward COM + fall match, close, brake.

    Table window from grasp-height release is ~0.20 s (~10×20 ms steps), so
    chase must be short and near-saturated — not a long vertical-recatch replay.
    """
    restore_ic(env, ic)
    acts = []
    n_open = int(p["n_open"])
    n_chase = int(p["n_chase"])
    n_close = int(p["n_close"])
    n_brake = int(p["n_brake"])
    scale = float(p["chase_scale"])
    # Aim slightly below COM so jaws meet the falling cylinder (lead).
    lead_z = float(p.get("lead_z", 0.025))
    for i in range(n_open + n_chase + n_close + n_brake):
        ff = finger_frame(env.sim)
        po, ph = obj_pos(env.sim), hand_pos(env.sim)
        mid = ff["mid"]
        Rh = ff["Rh"]
        vo, _ = body_v_w(env.sim, env.sim.ids.object_body)
        target = po.copy()
        target[2] -= lead_z
        # drive finger-mid toward target (not palm origin)
        err_w = target - mid
        err_h = Rh.T @ err_w
        v_des_h = Rh.T @ (float(p["v_scale"]) * vo)
        a = np.zeros(7, np.float32)
        # saturated lateral chase — table window is short
        a[0] = float(np.clip(scale * err_h[0] / 0.03 + 0.5 * v_des_h[0] / V_AIRBORNE, -1, 1))
        a[1] = float(np.clip(scale * err_h[1] / 0.03 + 0.5 * v_des_h[1] / V_AIRBORNE, -1, 1))
        vz = float(err_h[2] / 0.04 + v_des_h[2] + float(p["vz_bias"]))
        a[2] = float(np.clip(vz / 3.8, 0, 1) if vz >= 0 else np.clip(vz / 1.0, -1, 0))
        a[4] = float(p.get("wy", 0.0))
        if i < n_open:
            a[6] = -1.0
            a[:3] *= float(p.get("open_move", 1.0))
        elif i < n_open + n_chase:
            a[6] = float(p.get("grip_chase", -1.0))
            # over-chase: push past COM along the dominant planar error
            oc = float(p.get("overchase", 0.2))
            if abs(err_h[0]) >= abs(err_h[1]):
                a[0] = float(np.clip(a[0] + np.sign(err_h[0] + 1e-9) * oc, -1, 1))
            else:
                a[1] = float(np.clip(a[1] + np.sign(err_h[1] + 1e-9) * oc, -1, 1))
        elif i < n_open + n_chase + n_close:
            a[6] = float(p.get("grip_close", 1.0))
            a[:3] *= float(p.get("close_scale", 0.35))
        else:
            a[:2] *= 0.1
            a[2] = float(p.get("brake_vz", 0.15))  # continue slight down/brake in hand +z
            a[6] = 1.0
            if p.get("use_fine_down"):
                a = FINE_DOWN.copy()
        # optional weak template blend (often wrong axis for lateral ICs)
        if i < min(len(template), 4) and p.get("blend_template", False):
            a = np.clip(0.7 * a + 0.3 * template[i], -1, 1).astype(np.float32)
            if i < n_open:
                a[6] = -1.0
        step_legal(env.sim, a)
        acts.append(a.copy())
        cs = contact_summary(env.sim)
        if cs["bilateral"] and i >= n_open + n_chase:
            # hold close/brake after capture
            pass
        if object_table_contact(env.sim) or float(obj_pos(env.sim)[2]) < TABLE_Z - 0.02:
            break
    return acts


def k49_overchase(table, book48, direction_hint="open"):
    # Short open + best available translate token + close. Wrist reverse only late.
    open_tok = [30] * 2
    if direction_hint == "open":
        chase = [1] * 8  # has planar velocity
    elif direction_hint == "distal":
        chase = [1] * 8
    else:
        chase = [1] * 8
    toks = open_tok + chase + [12] * 4 + [0] * 4 + [SECURE_REVERSE_WRIST] * 10
    return [np.asarray(decode(table, book48, int(t)), np.float32) for t in toks]


def bang_bang_lateral(env, ic, axis="auto", n_chase=6, n_close=5, n_brake=20):
    """Saturated legal chase along IC direction then close/brake (diagnostic continuous)."""
    restore_ic(env, ic)
    acts = []
    ff0 = finger_frame(env.sim)
    if axis == "auto":
        d = ic["direction"]
        u_w = ff0["u_open"] if d == "open" else (ff0["u_dist"] if d == "distal" else ff0["u_perp"])
    else:
        u_w = ff0[axis]
    for i in range(2 + n_chase + n_close + n_brake):
        Rh = finger_frame(env.sim)["Rh"]
        u_h = Rh.T @ u_w
        a = np.zeros(7, np.float32)
        a[0] = float(np.clip(u_h[0] * 2.0, -1, 1))
        a[1] = float(np.clip(u_h[1] * 2.0, -1, 1))
        # match fall: hand +z is world −z → positive a[2] goes down
        a[2] = float(np.clip(0.35 + 0.05 * i, 0, 1))
        if i < 2 + n_chase:
            a[6] = -1.0
        elif i < 2 + n_chase + n_close:
            a[6] = 1.0
            a[:2] *= 0.3
        else:
            a[:2] *= 0.05
            a[2] = 0.15
            a[6] = 1.0
            if i > 2 + n_chase + n_close + 6:
                a = FINE_DOWN.copy()
        step_legal(env.sim, a)
        acts.append(a.copy())
        if object_table_contact(env.sim):
            break
    return acts


def scoop_align_descend(env, ic, n_align=4, close_xy=0.025, lead=0.02, n_hold=30, wy=0.0):
    """Align XY above falling COM (slight lift), then descend+close, then fine_down hold.

    Distal/perp ICs already project into the open-axis jaw range; open-direction ICs
    must cover a larger planar gap and often hit the hand mesh first.
    """
    restore_ic(env, ic)
    acts = []
    closed = False
    bil_seen = False
    for i in range(n_align + 12 + n_hold):
        ff = finger_frame(env.sim)
        po = obj_pos(env.sim)
        mid = ff["mid"]
        Rh = ff["Rh"]
        tgt = po.copy()
        tgt[2] -= float(lead)
        err = Rh.T @ (tgt - mid)
        a = np.zeros(7, np.float32)
        a[0] = float(np.clip(1.4 * err[0] / 0.02, -1, 1))
        a[1] = float(np.clip(1.4 * err[1] / 0.02, -1, 1))
        a[4] = float(wy)
        xy = float(np.linalg.norm(err[:2]))
        t = float(np.dot(po - ff["pr"], ff["u_open"]))
        between = 0.0 < t < ff["gap"]
        if (not closed) and i < n_align and xy > close_xy:
            a[2] = -0.2
            a[6] = -1.0
        elif (not closed) and (xy > close_xy or not between):
            a[2] = 1.0
            a[6] = -1.0
        else:
            closed = True
            a[2] = 0.35
            a[6] = 1.0
            a[:2] *= 0.25
        cs = contact_summary(env.sim)
        if cs["bilateral"]:
            bil_seen = True
            a = FINE_DOWN.copy()
            if abs(wy) > 1e-6:
                a[4] = float(wy)
        step_legal(env.sim, a)
        acts.append(a.copy())
        if bil_seen:
            # dedicated hold budget after first bilateral
            for _ in range(n_hold):
                a = FINE_DOWN.copy()
                if abs(wy) > 1e-6:
                    a[4] = float(wy)
                step_legal(env.sim, a)
                acts.append(a.copy())
            break
        if object_table_contact(env.sim) or float(po[2]) < TABLE_Z - 0.01:
            break
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for d in ("raw", "trajectories", "videos", "snapshots"):
        (OUT / d).mkdir(exist_ok=True)

    write_json(OUT / "raw" / "BUDGET.json", {
        "STAGE_A": STAGE_A, "STAGE_B": STAGE_B, "STAGE_C": STAGE_C, "STAGE_D": STAGE_D,
        "TOTAL_BUDGET": TOTAL_BUDGET, "WALL_LIMIT_S": WALL_LIMIT_S, "HOLD_S": HOLD_S,
        "recorded_before_execution": True,
    })
    (OUT / "INITIAL_STATE_AUDIT.md").write_text(
        f"""# Initial State Audit — Lateral Offset Falling Recatch

## Robot

- Starts from **nominal grasp lift** (`grasp_settled`), then open-gripper legal actions only.
- **No inverted catch-ready wrist** initialization.
- Nominal hand orientation `R0` stored per IC; wrist deviation classified on every trial.

## Cylinder

- Airborne, COM height ≈ grasp mid height.
- Horizontally displaced along `open` / `distal` / `perp` unit axes in the hand frame.
- Zero linear and angular velocity at IC snapshot.
- Outside the grasp region: `open` requires COM not between jaws; `distal`/`perp`
  are along ±hand-pad axis (for vertical approach these are opposite sides).
- `offset_m` is **additional** separation beyond an empirically clear baseline
  (`clear_base_m`: open 0.12 m, distal/perp 0.06 m) so the object is not inside
  the hand mesh. Actual `sep_along_dir_m` is recorded.
- Falls under MuJoCo gravity after control begins (~0.20 s to table from grasp height).

## Criterion

`LATERAL_RECATCH_SUCCESS` (HOLD_S={HOLD_S}s): separated init, descending first contact,
nontrivial hand motion (≥8 mm), bilateral hold, fall arrested, no pre-capture table hit.

## Overchase prior

Template: `{OVERCHASE_TEMPLATE}` (open_overchase_close_brake actions).
""",
        encoding="utf-8",
    )

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    checksum = mapping_checksum(table)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261020)
    rollout = [0]
    t_wall0 = time.time()

    def time_left():
        return time.time() - t_wall0 < WALL_LIMIT_S

    print("=== Build nominal open robot ===", flush=True)
    robot_cap, R0, mid0, ff0 = build_nominal_open_robot(env)
    write_json(OUT / "raw" / "NOMINAL_HAND.json", {
        "mid": mid0.tolist(), "R0": R0.tolist(),
        "gap": ff0["gap"], "u_open": ff0["u_open"].tolist(),
        "u_dist": ff0["u_dist"].tolist(), "u_perp": ff0["u_perp"].tolist(),
    })

    template = overchase_template_actions()
    print(f"Overchase template len={len(template)} from {OVERCHASE_TEMPLATE.exists()}", flush=True)

    # Build IC sweep (0.00 = minimum clear outside grasp/hand mesh)
    offsets_B = [0.00, 0.01, 0.02, 0.03]
    offsets_C = [0.04, 0.06, 0.08]
    directions = ["open", "distal", "perp"]
    ics = []
    offset_rows = []
    root = 0
    for direction in directions:
        for off in offsets_B + offsets_C:
            ic = make_ic(env, robot_cap, R0, mid0, direction, off, f"r{root}")
            root += 1
            ics.append(ic)
            offset_rows.append({k: v for k, v in ic.items() if k not in ("capture", "ref", "R0")})
            with (OUT / "snapshots" / f"{ic['ic_id']}.pkl").open("wb") as f:
                pickle.dump(ic, f)
            print(
                f"  {ic['ic_id']} ok={ic['ok_init']} outside={ic['outside_grasp_region']} "
                f"between={ic['com_between_fingers']} sep={ic['sep_world_m']:.3f} "
                f"along={ic['sep_along_dir_m']:.3f} t_tab={ic['t_table_no_intervene_s']}",
                flush=True,
            )
    write_csv(OUT / "OFFSET_SWEEP.csv", offset_rows)

    transfer_rows = []
    timeline_rows = []
    contact_rows = []
    success_rows = []
    handoff_rows = []
    robust_rows = []

    def run_one(ic, name, acts, family=None):
        out, geo, tl = evaluate(env, ic, acts, check_handoff=False)
        if out["LATERAL_RECATCH_SUCCESS"]:
            out, geo, tl = evaluate(env, ic, acts, check_handoff=True)
            rollout[0] += 1
        rollout[0] += 1
        row = {
            "ic_id": ic["ic_id"], "direction": ic["direction"], "offset_m": ic["offset_m"],
            "condition": name, **{k: out[k] for k in out},
        }
        if family is not None:
            family.append(row)
        transfer_rows.append(row)
        if geo is not None:
            contact_rows.append({
                "ic_id": ic["ic_id"], "condition": name,
                "t_first_grasp": geo.get("t"), "vz": geo.get("vz"),
                "com_between": geo.get("com_between"), "dominant_site": geo.get("dominant"),
                "sites": json.dumps(geo.get("sites")),
            })
        if tl:
            timeline_rows.append({
                "ic_id": ic["ic_id"], "condition": name,
                "t_first_grasp": out.get("t_first_grasp"),
                "max_hand_disp_m": out.get("max_hand_disp_m"),
                "max_wrist_deg": out.get("max_wrist_deg"),
                "t_table_no_intervene_s": ic.get("t_table_no_intervene_s"),
                "n_timeline": len(tl),
            })
        if out["LATERAL_RECATCH_SUCCESS"]:
            success_rows.append({
                "ic_id": ic["ic_id"], "direction": ic["direction"], "offset_m": ic["offset_m"],
                "condition": name, "LATERAL_RECATCH_SUCCESS": 1,
                "wrist_class": out["wrist_class"], "max_wrist_deg": out["max_wrist_deg"],
                "max_hand_disp_m": out["max_hand_disp_m"],
                "SUCCESS_NEW": out.get("SUCCESS_NEW"), "HANDOFF_OK": out.get("HANDOFF_OK"),
                "actions_hash": tokens_hash([a.tolist() for a in acts]),
            })
            handoff_rows.append({
                "ic_id": ic["ic_id"], "condition": name,
                "HANDOFF_OK": out.get("HANDOFF_OK"), "SUCCESS_NEW": out.get("SUCCESS_NEW"),
                "win_mean_pos": out.get("win_mean_pos"), "win_mean_theta": out.get("win_mean_theta"),
            })
            np.savez_compressed(
                OUT / "trajectories" / f"{ic['ic_id']}__{name}_actions.npz",
                actions=np.asarray(acts, np.float32),
            )
            print(
                f"  SUCCESS {ic['ic_id']} {name} wrist={out['wrist_class']} "
                f"disp={out['max_hand_disp_m']:.3f} deg={out['max_wrist_deg']:.1f}",
                flush=True,
            )
        return out

    # ---- Stage A: template transfer sanity on min-clear / 1–2 cm open ----
    print(f"=== Stage A overchase transfer (to {STAGE_A}) ===", flush=True)
    stage_a_ics = [
        ic for ic in ics
        if ic["direction"] == "open" and ic["offset_m"] in (0.00, 0.01, 0.02) and ic["ok_init"]
    ]
    for ic in stage_a_ics:
        if rollout[0] >= STAGE_A or not time_left():
            break
        print(f"  A {ic['ic_id']}", flush=True)
        run_one(ic, "template_raw", template)
        run_one(ic, "K49_overchase", k49_overchase(table, book48, ic["direction"]))
        run_one(ic, "bang_bang", bang_bang_lateral(env, ic, n_chase=5, n_close=4))
        run_one(ic, "scoop", scoop_align_descend(env, ic, n_align=4, n_hold=35))
        p = {"n_open": 1, "n_chase": 6, "n_close": 4, "n_brake": 20,
             "chase_scale": 1.2, "v_scale": 0.9, "vz_bias": 0.2, "wy": 0.0,
             "grip_chase": -1.0, "grip_close": 1.0, "overchase": 0.25,
             "blend_template": False, "lead_z": 0.02}
        run_one(ic, "adapt_overchase", adapt_overchase(env, ic, p, template))

    # ---- Stage B: 0–3 cm all directions (short-horizon search) ----
    lim_b = STAGE_A + STAGE_B
    print(f"=== Stage B 0-3cm (to {lim_b}) ===", flush=True)
    b_ics = [ic for ic in ics if ic["offset_m"] <= 0.03 + 1e-9 and ic["ok_init"]]
    for ic in b_ics:
        if rollout[0] >= lim_b or not time_left():
            break
        print(f"  B {ic['ic_id']} roll={rollout[0]}", flush=True)
        run_one(ic, "K49_overchase", k49_overchase(table, book48, ic["direction"]))
        run_one(ic, "A_no_intercept", [np.array([0, 0, 0, 0, 0, 0, -1], np.float32)] * 40)
        run_one(ic, "bang_bang", bang_bang_lateral(env, ic, n_chase=int(rng.choice([4, 5, 6, 7])), n_close=4))
        best = None
        # scoop grid (primary for distal/perp)
        for n_align in (3, 4, 5, 6):
            for close_xy in (0.02, 0.025, 0.03, 0.035):
                for lead in (0.015, 0.02, 0.03):
                    if rollout[0] >= lim_b or not time_left():
                        break
                    acts = scoop_align_descend(
                        env, ic, n_align=n_align, close_xy=close_xy, lead=lead, n_hold=35,
                        wy=float(rng.choice([0.0, 0.0, 0.15])),
                    )
                    out = run_one(ic, "D_scoop", acts)
                    sc = (100 * out["LATERAL_RECATCH_SUCCESS"] + 8 * out["max_bil_persist_s"]
                          + 2 * out["had_bil"] - 0.05 * out["max_wrist_deg"])
                    if best is None or sc > best[0]:
                        best = (sc, {"n_align": n_align, "close_xy": close_xy, "lead": lead, "kind": "scoop"})
        for _ in range(10):
            if rollout[0] >= lim_b or not time_left():
                break
            p = {
                "n_open": int(rng.choice([0, 1, 2])),
                "n_chase": int(rng.choice([3, 4, 5, 6, 7, 8])),
                "n_close": int(rng.choice([3, 4, 5, 6])),
                "n_brake": int(rng.choice([14, 18, 22])),
                "chase_scale": float(rng.choice([0.9, 1.1, 1.3, 1.5])),
                "v_scale": float(rng.choice([0.5, 0.8, 1.0])),
                "vz_bias": float(rng.choice([0.0, 0.15, 0.3, 0.45])),
                "wy": float(rng.choice([0.0, 0.0, 0.0, 0.15, 0.25])),
                "grip_chase": -1.0,
                "grip_close": float(rng.choice([0.35, 1.0])),
                "overchase": float(rng.choice([0.0, 0.15, 0.3, 0.45])),
                "blend_template": bool(rng.random() < 0.2),
                "use_fine_down": bool(rng.random() < 0.35),
                "lead_z": float(rng.choice([0.01, 0.02, 0.03, 0.04])),
                "close_scale": float(rng.choice([0.2, 0.35, 0.5])),
            }
            acts = adapt_overchase(env, ic, p, template)
            out = run_one(ic, "D_continuous_adapt", acts)
            sc = (100 * out["LATERAL_RECATCH_SUCCESS"] + 8 * out["max_bil_persist_s"]
                  + 2 * out["had_bil"] - 0.05 * out["max_wrist_deg"])
            if best is None or sc > best[0]:
                best = (sc, {**p, "kind": "adapt"})
        if best is not None:
            for _ in range(8):
                if rollout[0] >= lim_b or not time_left():
                    break
                bp = dict(best[1])
                if bp.get("kind") == "scoop":
                    acts = scoop_align_descend(
                        env, ic,
                        n_align=int(np.clip(bp["n_align"] + int(rng.integers(-1, 2)), 2, 7)),
                        close_xy=float(np.clip(bp["close_xy"] + rng.normal(0, 0.003), 0.015, 0.045)),
                        lead=float(np.clip(bp["lead"] + rng.normal(0, 0.004), 0.01, 0.04)),
                        n_hold=40, wy=float(rng.choice([0.0, 0.1])),
                    )
                    run_one(ic, "D_refine_scoop", acts)
                else:
                    p = {k: v for k, v in bp.items() if k != "kind"}
                    p["n_chase"] = int(np.clip(p["n_chase"] + int(rng.integers(-2, 3)), 2, 10))
                    p["vz_bias"] = float(np.clip(p["vz_bias"] + rng.normal(0, 0.05), -0.1, 0.6))
                    p["wy"] = float(np.clip(abs(p["wy"]) * 0.5, 0, 0.35))
                    p["lead_z"] = float(np.clip(p["lead_z"] + rng.normal(0, 0.005), 0.005, 0.05))
                    run_one(ic, "D_refine", adapt_overchase(env, ic, p, template))

    n_b_success = len(set(r["ic_id"] for r in success_rows if float(r["offset_m"]) <= 0.03))
    print(f"Stage B success ICs: {n_b_success}", flush=True)

    # ---- Stage C: 4–8 cm only if B had success ----
    lim_c = lim_b + STAGE_C
    if n_b_success > 0:
        print(f"=== Stage C 4-8cm (to {lim_c}) ===", flush=True)
        c_ics = [ic for ic in ics if ic["offset_m"] >= 0.04 - 1e-9 and ic["ok_init"]]
        for ic in c_ics:
            if rollout[0] >= lim_c or not time_left():
                break
            print(f"  C {ic['ic_id']} roll={rollout[0]}", flush=True)
            run_one(ic, "K49_overchase", k49_overchase(table, book48, ic["direction"]))
            for n_align in (4, 5, 6, 7):
                for close_xy in (0.02, 0.025, 0.03):
                    for lead in (0.02, 0.03):
                        if rollout[0] >= lim_c or not time_left():
                            break
                        run_one(
                            ic, "D_scoop",
                            scoop_align_descend(
                                env, ic, n_align=n_align, close_xy=close_xy,
                                lead=lead, n_hold=40,
                            ),
                        )
            for _ in range(6):
                if rollout[0] >= lim_c or not time_left():
                    break
                p = {
                    "n_open": 1, "n_chase": int(rng.choice([4, 5, 6, 7, 8])),
                    "n_close": 4, "n_brake": 20,
                    "chase_scale": float(rng.choice([1.1, 1.3, 1.5])),
                    "v_scale": float(rng.choice([0.6, 0.9])),
                    "vz_bias": float(rng.choice([0.15, 0.3, 0.45])),
                    "wy": float(rng.choice([0.0, 0.15])),
                    "grip_chase": -1.0, "grip_close": 1.0,
                    "overchase": float(rng.choice([0.2, 0.35, 0.5])),
                    "blend_template": False, "use_fine_down": True,
                    "lead_z": float(rng.choice([0.02, 0.03])),
                }
                run_one(ic, "D_continuous_adapt", adapt_overchase(env, ic, p, template))
    else:
        print("=== Stage C skipped (no Stage B success) ===", flush=True)

    # ---- Stage D: robustness ----
    lim_d = TOTAL_BUDGET
    print(f"=== Stage D robustness (to {lim_d}) ===", flush=True)
    for srow in list(success_rows):
        if rollout[0] >= lim_d or not time_left():
            break
        path = OUT / "trajectories" / f"{srow['ic_id']}__{srow['condition']}_actions.npz"
        if not path.exists():
            continue
        acts = [np.asarray(a, np.float32) for a in np.load(path)["actions"]]
        ic = next(x for x in ics if x["ic_id"] == srow["ic_id"])
        for shift in (-2, -1, 1, 2):
            if rollout[0] >= lim_d:
                break
            acts_s = acts[abs(shift):] if shift < 0 else (
                [np.array([0, 0, 0, 0, 0, 0, -1], np.float32)] * shift + acts
            )
            out, _, _ = evaluate(env, ic, acts_s)
            rollout[0] += 1
            robust_rows.append({
                "ic_id": ic["ic_id"], "condition": srow["condition"],
                "variation": f"timing_{shift}",
                "LATERAL_RECATCH_SUCCESS": out["LATERAL_RECATCH_SUCCESS"],
                "max_bil_persist_s": out["max_bil_persist_s"],
                "wrist_class": out["wrist_class"],
            })
        for doff in (-0.005, 0.005):
            if rollout[0] >= lim_d:
                break
            ic2 = make_ic(env, robot_cap, R0, mid0, ic["direction"],
                          max(0.005, ic["offset_m"] + doff), ic["root_id"] + "_v")
            if not ic2["ok_init"]:
                continue
            out, _, _ = evaluate(env, ic2, acts)
            rollout[0] += 1
            robust_rows.append({
                "ic_id": ic["ic_id"], "condition": srow["condition"],
                "variation": f"offset_delta_{doff}",
                "variant_ic": ic2["ic_id"],
                "LATERAL_RECATCH_SUCCESS": out["LATERAL_RECATCH_SUCCESS"],
                "max_bil_persist_s": out["max_bil_persist_s"],
            })

    write_csv(OUT / "OVERCHASE_TRANSFER.csv", transfer_rows)
    write_csv(OUT / "INTERCEPTION_TIMELINE.csv", timeline_rows)
    write_csv(OUT / "CONTACT_GEOMETRY.csv", contact_rows)
    write_csv(OUT / "CATCH_SUCCESS_VERIFICATION.csv", success_rows)
    write_csv(OUT / "HANDOFF_RESULTS.csv", handoff_rows)
    write_csv(OUT / "ROBUSTNESS_RESULTS.csv", robust_rows)
    write_json(OUT / "raw" / "USAGE.json", {
        "rollouts_used": rollout[0], "budget": TOTAL_BUDGET,
        "wall_s": time.time() - t_wall0, "n_success": len(success_rows),
        "checksum": checksum,
    })

    if not args.skip_videos:
        _videos(env, ics, success_rows)

    _report(offset_rows, transfer_rows, success_rows, contact_rows, robust_rows, handoff_rows, checksum)
    print("Done", OUT / "REPORT.md", flush=True)


def _videos(env, ics, success_rows):
    # sample per direction
    shown = set()
    for ic in ics:
        if ic["direction"] in shown:
            continue
        if ic["offset_m"] not in (0.02, 0.03):
            continue
        shown.add(ic["direction"])
        acts = overchase_template_actions()
        for r in success_rows:
            if r["ic_id"] == ic["ic_id"]:
                p = OUT / "trajectories" / f"{r['ic_id']}__{r['condition']}_actions.npz"
                if p.exists():
                    acts = list(np.load(p)["actions"])
                break
        _render(env, ic, acts, OUT / "videos" / f"{ic['ic_id']}_demo.mp4")
    for r in success_rows[:4]:
        p = OUT / "trajectories" / f"{r['ic_id']}__{r['condition']}_actions.npz"
        if not p.exists():
            continue
        ic = next(x for x in ics if x["ic_id"] == r["ic_id"])
        _render(env, ic, list(np.load(p)["actions"]), OUT / "videos" / f"success_{r['ic_id']}.mp4")


def _render(env, ic, acts, dest):
    try:
        restore_ic(env, ic)
        cam = mjv_camera_from_preset({
            "lookat": obj_pos(env.sim), "distance": 0.65, "azimuth": 140.0, "elevation": -18.0,
        })
        renderer = mujoco.Renderer(env.sim.model, 360, 480)
        frames = []
        t0 = float(env.sim.data.time)
        for i, a in enumerate(acts):
            if i % 2 == 0:
                cam.lookat[:] = 0.5 * (obj_pos(env.sim) + hand_pos(env.sim))
                renderer.update_scene(env.sim.data, cam)
                cs = contact_summary(env.sim)
                frames.append(caption(
                    renderer.render(),
                    f"{ic['ic_id']} bil={int(cs['bilateral'])} t={float(env.sim.data.time)-t0:.2f}",
                ))
            step_legal(env.sim, np.asarray(a, np.float32))
            if object_table_contact(env.sim) and i > 8:
                break
        write_frames_mp4(frames, dest, fps=25)
        renderer.close()
        print("video", dest, flush=True)
    except Exception as e:
        print("video warn", e, flush=True)


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and x == x]
    return float(np.mean(xs)) if xs else float("nan")


def _report(offset_rows, transfer, success, contact, robust, handoff, checksum):
    roots = sorted(set(r["ic_id"] for r in success))
    max_off = max((float(r["offset_m"]) for r in success), default=None)
    dirs = sorted(set(r["direction"] for r in success))
    k49 = [r for r in transfer if str(r.get("condition", "")).startswith("K49")]
    cont = [r for r in transfer if "continuous" in str(r.get("condition", "")) or str(r.get("condition", "")).startswith("D_")]
    wrist = {}
    for r in success:
        wrist[r.get("wrist_class", "?")] = wrist.get(r.get("wrist_class", "?"), 0) + 1
    robust_ok = sum(1 for r in robust if int(float(r.get("LATERAL_RECATCH_SUCCESS") or 0)))
    usage = json.loads((OUT / "raw" / "USAGE.json").read_text(encoding="utf-8"))

    report = f"""# Lateral-Offset Falling Recatch (Nominal Grasp Pose)

**Baseline K49** checksum `{checksum}` · AIRBORNE_V3 unchanged · **No inverted init**  
**Criterion:** `LATERAL_RECATCH_SUCCESS` (HOLD_S={HOLD_S}s)  
**Not mixed into K49 recovery experts. No SFT/GRPO.**

## Budget

| Item | Value |
|------|------:|
| Total | {TOTAL_BUDGET} |
| Used | {usage.get('rollouts_used')} |
| Wall (s) | {usage.get('wall_s'):.1f} |

## Results

Verified SUCCESS ICs: **{len(roots)}** — {roots}  
Directions with success: **{dirs}**  
Largest offset with success: **{max_off if max_off is not None else 'n/a'}** m  
Wrist classes among successes: {json.dumps(wrist)}

| Family | catch rate |
|--------|----------:|
| K49 | {_mean([r.get('LATERAL_RECATCH_SUCCESS') for r in k49]):.3f} |
| Continuous adapt | {_mean([r.get('LATERAL_RECATCH_SUCCESS') for r in cont]):.3f} |

Robustness survivors: **{robust_ok}/{len(robust)}**  
HANDOFF / SUCCESS_NEW among catches: see `HANDOFF_RESULTS.csv`.

## Q1–Q12

1. Active lateral recatch from nominal pose? **{"Yes" if roots else "Not under this budget"}**.
2. Feasible directions: **{dirs if dirs else "none"}**.
3. Largest tested offset with verified success: **{max_off if max_off is not None else "n/a"}** m.
4. Over-chase mechanism transfer? See Stage A `template_raw` / `adapt_overchase` vs K49 rates in `OVERCHASE_TRANSFER.csv`.
5. Hand displacement required: mean among successes **{_mean([r.get('max_hand_disp_m') for r in success]):.3f}** m.
6. Jaw-region entry before closure? Contact `com_between` / dominant site in `CONTACT_GEOMETRY.csv`.
7. Closure timing sensitivity: robustness timing shifts — survivors {robust_ok}/{len(robust)}.
8. K49 vs continuous: K49 rate {_mean([r.get('LATERAL_RECATCH_SUCCESS') for r in k49]):.3f} vs continuous {_mean([r.get('LATERAL_RECATCH_SUCCESS') for r in cont]):.3f}.
9. Stabilize without inverting? Successes with `wrist_class` not `orientation_reversal_or_extreme`: count in wrist table above.
10. Legal HANDOFF / SUCCESS_NEW? See `HANDOFF_RESULTS.csv` (separate from catch criterion).
11. Failure modes: interception timing vs outer contact vs stabilization — use `table_before_bil`, `jaw_entry_ok`, `descending_at_contact` columns.
12. Evidence for dynamic recovery: **{"useful if nominal-orientation lateral SUCCESS exists" if any(r.get("wrist_class")=="nominal_orientation_lateral" for r in success) else "limited / geometry-dependent"}** — prefer jaw-entry overchase from nominal pose; do not treat inverted free-fall catches as equivalent.

## Scripts
`scripts/run_lateral_offset_falling_recatch.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "FAILURE_ANALYSIS.md").write_text(
        f"""# Failure Analysis — Lateral Offset Falling Recatch

## Design

Nominal-pose open gripper + horizontally offset free-fall cylinder (constructed diagnostic ICs).
Overchase template from verified `open_overchase_close_brake` experts.

## Outcomes

- SUCCESS roots: {roots}
- Directions: {dirs}
- Max offset: {max_off}
- Wrist classes: {json.dumps(wrist)}

## Notes

- Do not count inverted free-fall catches as evidence for this protocol.
- Do not mix into K49 recovery expert dataset automatically.
- Outer-finger first contact and table-before-bilateral remain primary failure modes when chase is late or misaligned.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
