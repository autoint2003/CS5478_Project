"""Full-3D airborne recatch through the versioned action map.

Privileged object state chooses the 7-D command. The command is executed only
by recovery7d_airborne_v2 or recovery7d_local_v1, then the existing torque
controller. No training. No images. The cylinder is not teleported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery_actions import (
    AIRBORNE_V2,
    LOCAL_V1,
    V_AIRBORNE,
    V_LOCAL,
    W_MAX,
    map_action,
    normalize_action,
    v1_normalized_to_v2,
)
from envs.airborne_obs import OBS_DIM_AIR, OBS_NAMES_AIR, OBS_VERSION, observe_airborne, relative_state
from envs.deterioration import body_twist
from envs.observable_obs import ObservableObsState, OBS_NAMES
from envs.physical_recovery import physical_pack
from training.demo_ballistic_impact import TAU_SEC, assert_noslip, make_sim
from training.horizontal_grasp_offset_freefall_recatch import DT, begin, grasp_settled, hold_step
from training.long_context_recovery_pilot import jsonable
from training.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import capture_nominal, capture_pads, gains_from_cfg, stamp
from training.single_step_action_token_policy import kmeans

OUT = ROOT / "results" / "diagnostics" / "raw" / "full_3d_airborne_recatch"
# Tracking gains used by the verified catch. Torques are still clipped to the actuators.
TRACK = dict(kp_pos=1400.0, kd_pos=80.0, kp_ori=80.0, kd_ori=6.0, kp_null=2.0, kd_null=0.4)


def drop_supports(sim):
    names = []
    for name in ("floor", "table"):
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        sim.model.geom_contype[gid] = 0
        sim.model.geom_conaffinity[gid] = 0
        names.append(name)
    return names


def limits_of(sim):
    q = np.array(sim.data.qpos[sim.ids.arm_jnt], float)
    lo = np.array([sim.model.jnt_range[int(j), 0] for j in sim.ids.arm_jnt])
    hi = np.array([sim.model.jnt_range[int(j), 1] for j in sim.ids.arm_jnt])
    return q, np.minimum(q - lo, hi - q)


def tau_lim(sim):
    return np.array(sim.model.actuator_ctrlrange[:7, 1], float)


def self_check():
    eye = np.eye(3)
    a = np.array([0.5, -0.25, 0.0, 0.0, 0.25, 0.0, 1.0])
    m1 = map_action(a, eye, eye, LOCAL_V1)
    m2 = map_action(v1_normalized_to_v2(a), eye, eye, AIRBORNE_V2)
    if abs(m1["v_hand"][0] - 0.04) > 1e-9 or abs(m1["w_body"][1] - 1.0) > 1e-9:
        raise RuntimeError("local map mismatch")
    if np.max(np.abs(m1["v_world"] - m2["v_world"])) > 1e-6:
        raise RuntimeError("v1 physical command is not preserved in v2")
    if abs(m1["tau"] - m2["tau"]) > 1e-6 or abs(m1["tau"] + 18.0) > 1e-6:
        raise RuntimeError("grip map mismatch")
    wide = map_action(np.array([1, -1, 0.8, 0, 0, 0, -1.0]), eye, eye, AIRBORNE_V2)
    if abs(wide["v_hand"][0] - 1.0) > 1e-9 or abs(wide["tau"] - 2.0) > 1e-6:
        raise RuntimeError("airborne range mismatch")


def desired_velocity(sim, r_des, rh_target, mode, lead, k):
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
    R = np.asarray(r_des, float).reshape(3, 3)
    ph_des = po - R @ np.asarray(rh_target, float)
    if mode == "below":
        ph_des = ph_des + np.array([0.0, 0.0, -float(lead)])
    err = ph_des - ph
    v_world = vo + float(k) * err
    return v_world, np.zeros(3), err, po, ph, vo


def oracle_action(sim, r_des, rh_target, spec, phase, version):
    v_world, w_world, err, po, ph, vo = desired_velocity(
        sim, r_des, rh_target, spec["mode"], spec.get("lead", 0.0), spec.get("k", 4.0),
    )
    if phase in ("open", "chase"):
        # Command the translational limit downward and match lateral object
        # velocity. With these gains the hand meets the fall near 90 ms.
        v_world = np.array([
            float(np.clip(vo[0], -1.0, 1.0)),
            float(np.clip(vo[1], -1.0, 1.0)),
            -1.0,
        ])
        tau = 2.0
    else:
        track = np.clip(vo + 1.5 * err, -1.0, 1.0)
        # Once both pads have held, stop chasing hard. A full velocity
        # command here yanks the new pinch and drops a pad.
        v_world = 0.25 * track if phase == "hold" else track
        tau = -18.0
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Rd = np.asarray(r_des, float).reshape(3, 3)
    action = normalize_action(Rh.T @ v_world, Rd.T @ w_world, tau, version)
    return action, err, po, ph, vo


REBASE_SETPOINT = True


def step_mapped(sim, action, version, gains, q_home):
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(action, Rd, Rh, version)
    v, w, tau = cmd["v_world"], cmd["w_world"], cmd["tau"]
    dt = float(sim.model.opt.timestep)
    sim.fsm.p_des = np.asarray(sim.fsm.p_des, float) + v * dt
    sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, w, dt)
    sim.fsm.v_cmd = v.copy()
    sim.fsm.w_cmd = w.copy()
    sim.fsm.v_des = v.copy()
    sim.fsm.w_des = w.copy()
    sim.fsm.fg_cmd = float(-tau)
    apply_cartesian_ctrl(
        sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
        gripper_tau=tau, v_des=v, w_des=w, q_home=q_home, **gains,
    )
    mujoco.mj_step(sim.model, sim.data)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
    return cmd


def restore_to(sim, snap, nominal, pads, pad_mu):
    from training.multi_disturbance_airborne_feasibility import restore

    info = restore(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = DT
    if hasattr(sim, "park_ball"):
        sim.park_ball()
    return info


def run_chase(sim, snap, nominal, pads, pad_mu, spec, q_ref, rh0, version, gains, action_hold=1):
    info = restore_to(sim, snap, nominal, pads, pad_mu)
    if not info.get("pose_ok", False):
        return {"pose_ok": False, "ok": False}
    if spec.get("dx") or spec.get("dy") or spec.get("dz"):
        qadr = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
        sim.data.qpos[qadr] += float(spec.get("dx", 0.0))
        sim.data.qpos[qadr + 1] += float(spec.get("dy", 0.0))
        sim.data.qpos[qadr + 2] += float(spec.get("dz", 0.0))
        mujoco.mj_forward(sim.model, sim.data)
    if spec.get("dq") is not None:
        sim.data.qpos[sim.ids.arm_jnt] = np.asarray(q_ref, float) + np.asarray(spec["dq"], float)
        sim.data.qvel[sim.ids.arm_dof] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        for _ in range(int(0.08 / DT)):
            hold_step(sim, gains_from_cfg_cached(sim))
    delay = float(spec.get("open_delay", 0.0))
    for _ in range(int(delay / DT)):
        hold_step(sim, gains_from_cfg_cached(sim))
    r_des = np.array(sim.fsm.r_des, float).reshape(3, 3).copy()
    rh_target = np.array(rh0, float) + np.array([0.0, 0.0, float(spec.get("rz", 0.0))])
    return _roll(sim, r_des, rh_target, spec, q_ref, version, gains, action_hold=action_hold)


def gains_from_cfg_cached(sim):
    return getattr(sim, "_nom_gains")


def _roll(sim, r_des, rh_target, spec, q_ref, version, gains, action_hold=1):
    hist = ObservableObsState(DT)
    t0 = float(sim.data.time)
    o0 = physical_pack(sim)
    aper0 = float(o0["aperture"])
    phase = "chase" if int(o0["nL"]) == 0 and int(o0["nR"]) == 0 and aper0 > 0.028 else "open"
    free_run = 0.020 if phase == "chase" else 0.0
    free_t = None
    re_t = None
    hold_t = None
    both_hold = 0.0
    vrel_re = None
    vrel_close = None
    rows = []
    demos = []
    peak_cmd = np.zeros(3)
    peak_ach = np.zeros(3)
    peak_tau = np.zeros(7)
    peak_qd = np.zeros(7)
    min_margin = np.full(7, 1e9)
    lim = tau_lim(sim)
    sat_steps = 0
    held_action = None
    for i in range(int(2.6 / DT)):
        o = physical_pack(sim)
        rel = relative_state(sim.model, sim.data, sim.ids)
        both = int(o["nL"]) > 0 and int(o["nR"]) > 0
        none = int(o["nL"]) == 0 and int(o["nR"]) == 0
        aper = float(o["aperture"])
        t = float(sim.data.time)
        rh = rel["p_rel_h"]
        vrel = rel["v_obj_w"] - rel["v_hand_w"]
        if none:
            free_run += DT
            if free_t is None and free_run >= 0.020:
                free_t = t - free_run
        elif free_t is None:
            free_run = 0.0
        aligned = (
            abs(float(rh[0])) < 0.012
            and abs(float(rh[1])) < 0.012
            and abs(float(rh[2]) - float(rh_target[2])) < 0.012
        )
        slow = abs(float(vrel[2])) < 0.12 and float(np.linalg.norm(vrel[:2])) < 0.15
        if phase == "open" and aper >= 0.034:
            phase = "chase"
        if phase == "chase" and free_t is not None and aligned and slow and aper > 0.038:
            phase = "close"
            vrel_close = vrel.copy()
            if REBASE_SETPOINT:
                sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
        if phase in ("close", "hold") and free_t is not None and re_t is None and both:
            re_t = t
            vrel_re = vrel.copy()
        if phase == "close" and both:
            both_hold += DT
            if both_hold >= 0.04 and hold_t is None:
                hold_t = t
                phase = "hold"
                if REBASE_SETPOINT:
                    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
        elif phase == "close":
            both_hold = 0.0
        action, err, po, ph, vo = oracle_action(sim, r_des, rh_target, spec, phase, version)
        if held_action is None or i % int(action_hold) == 0:
            held_action = action
        action = held_action
        if getattr(sim, "_policy_trace", None) is not None and i % int(action_hold) == 0:
            sim._policy_trace.append({
                "t": float(t - t0),
                "phase": phase,
                "stamp": stamp(sim),
                "prev": np.zeros(7, np.float32) if len(sim._policy_trace) == 0 else np.asarray(sim._policy_trace[-1]["act"], np.float32),
                "act": np.asarray(action, np.float32).copy(),
                "nL": int(o["nL"]),
                "nR": int(o["nR"]),
                "both": bool(both),
                "none": bool(none),
            })
        obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
        cmd = step_mapped(sim, action, version, gains, q_ref)
        vh = np.array(body_twist(sim.model, sim.data, sim.ids.hand_body)[0], float)
        q, margin = limits_of(sim)
        qd = np.array(sim.data.qvel[sim.ids.arm_dof], float)
        tau = np.array(sim.data.ctrl[:7], float)
        peak_cmd = np.maximum(peak_cmd, np.abs(cmd["v_world"]))
        peak_ach = np.maximum(peak_ach, np.abs(vh))
        peak_tau = np.maximum(peak_tau, np.abs(tau))
        peak_qd = np.maximum(peak_qd, np.abs(qd))
        min_margin = np.minimum(min_margin, margin)
        if np.any(np.abs(tau) > 0.95 * lim):
            sat_steps += 1
        if i % 5 == 0:
            rows.append({
                "t": t - t0,
                "phase": phase,
                "nL": int(o["nL"]),
                "nR": int(o["nR"]),
                "aper": aper,
                "cmd_v": cmd["v_world"].copy(),
                "cmd_w": cmd["w_world"].copy(),
                "ach_v": vh.copy(),
                "obj_v": vo.copy(),
                "vrel": vrel.copy(),
                "prel": (po - ph).copy(),
                "action": np.asarray(action, float).copy(),
            })
        demos.append((obs, np.asarray(action, np.float32), t - t0, phase))
        if hold_t is not None and t >= hold_t + 1.0:
            break
        if po[2] < -1.2 or (free_t is not None and re_t is None and t > free_t + 1.2):
            break
    held = False
    if hold_t is not None and rows:
        tail = [r for r in rows if r["t"] >= (hold_t - t0) - 1e-3]
        held = bool(tail) and all(r["nL"] > 0 and r["nR"] > 0 for r in tail)
    hold_rows = [r for r in rows if r["phase"] == "hold"]
    vrel_hold = None
    if hold_rows:
        stack = np.stack([r["vrel"] for r in hold_rows])
        vrel_hold = [round(float(x), 4) for x in np.median(stack, axis=0)]
    end = physical_pack(sim)
    ok = bool(free_t is not None and re_t is not None and held and float(sim.data.time) >= (hold_t or 0) + 0.95)
    return {
        "pose_ok": True,
        "ok": ok,
        "version": version,
        "free_t": None if free_t is None else round(float(free_t - t0), 4),
        "re_t": None if re_t is None else round(float(re_t - t0), 4),
        "air_s": None if free_t is None or re_t is None else round(float(re_t - free_t), 4),
        "vrel_re": None if vrel_re is None else [round(float(x), 4) for x in vrel_re],
        "vrel_close": None if vrel_close is None else [round(float(x), 4) for x in vrel_close],
        "vrel_hold": vrel_hold,
        "end_n": [int(end["nL"]), int(end["nR"])],
        "end_z": round(float(end["obj_z"]), 4),
        "peak_cmd_v": [round(float(x), 4) for x in peak_cmd],
        "peak_ach_v": [round(float(x), 4) for x in peak_ach],
        "peak_tau": [round(float(x), 2) for x in peak_tau],
        "peak_qd": [round(float(x), 3) for x in peak_qd],
        "min_margin": [round(float(x), 3) for x in min_margin],
        "sat_steps": int(sat_steps),
        "tau_lim": [round(float(x), 1) for x in lim],
        "spec": {k: v for k, v in spec.items() if not isinstance(v, np.ndarray)},
        "demos": demos if ok else [],
        "rows": rows,
    }


def carry_then_release(sim, snap, nominal, pads, pad_mu, q_ref, speed, sign, dur):
    """Move the closed hand sideways through the v2 map, then open. Object velocity comes from contact."""
    info = restore_to(sim, snap, nominal, pads, pad_mu)
    if not info.get("pose_ok", False):
        return None
    r_des = np.array(sim.fsm.r_des, float).reshape(3, 3).copy()
    t0 = float(sim.data.time)
    phase_open_at = t0 + float(dur)
    while float(sim.data.time) < phase_open_at + 0.08:
        t = float(sim.data.time)
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        v_h = np.zeros(3)
        if t < phase_open_at:
            v_h[1] = float(sign) * float(speed)
            tau = -18.0
        else:
            v_h[2] = 0.3
            tau = 2.0
        action = normalize_action(v_h, np.zeros(3), tau, AIRBORNE_V2)
        step_mapped(sim, action, AIRBORNE_V2, TRACK, q_ref)
        o = physical_pack(sim)
        if t > phase_open_at and int(o["nL"]) == 0 and int(o["nR"]) == 0:
            # one extra 20 ms so the free-flight velocity is real
            for _ in range(10):
                step_mapped(sim, action, AIRBORNE_V2, TRACK, q_ref)
            break
    vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
    o = physical_pack(sim)
    return {
        "snap": stamp(sim),
        "r_des": r_des,
        "vo": [round(float(x), 4) for x in vo],
        "n": [int(o["nL"]), int(o["nR"])],
        "z": round(float(o["obj_z"]), 4),
        "aper": round(float(o["aperture"]), 4),
    }


def pack_demo(row, tag):
    demos = row.get("demos") or []
    if not demos:
        return None
    obs = np.stack([d[0] for d in demos]).astype(np.float32)
    act = np.stack([d[1] for d in demos]).astype(np.float32)
    t = np.array([d[2] for d in demos], np.float32)
    phase = np.array([d[3] for d in demos])
    return {"tag": tag, "obs": obs, "action": act, "t": t, "phase": phase, "rows": row.get("rows") or [], "summary": {k: v for k, v in row.items() if k not in ("demos", "rows")}}


def skill_audit():
    eye = np.eye(3)
    sch = schedules()
    keep = ("wrist_reseat", "recatch_open20", "gravity_inward")
    out = {}
    actions = []
    labels = []
    for name in keep:
        seq = sch[name]
        errs = []
        v2s = []
        for a in seq:
            m1 = map_action(a, eye, eye, LOCAL_V1)
            a2 = v1_normalized_to_v2(a)
            m2 = map_action(a2, eye, eye, AIRBORNE_V2)
            errs.append(float(np.max(np.abs(m1["v_world"] - m2["v_world"])) + abs(m1["tau"] - m2["tau"]) + np.max(np.abs(m1["w_world"] - m2["w_world"]))))
            v2s.append(a2)
            actions.append(a2)
            labels.append(name)
        arr = np.stack(v2s)
        out[name] = {
            "n": len(seq),
            "max_reconstruct_err": float(np.max(errs)),
            "v2_trans_max_abs": [round(float(x), 4) for x in np.max(np.abs(arr[:, 0:3]), axis=0)],
            "v2_rot_max_abs": [round(float(x), 4) for x in np.max(np.abs(arr[:, 3:6]), axis=0)],
            "grip_min": round(float(arr[:, 6].min()), 3),
            "grip_max": round(float(arr[:, 6].max()), 3),
        }
    return out, actions, labels


def token_audit(skill_actions, skill_labels, demos):
    extra_a = []
    extra_y = []
    for demo in demos:
        act = demo["action"]
        phase = demo["phase"]
        # Decimate to every 10th physics step so chase actions are not 10x overweighted.
        for i in range(0, len(act), 10):
            a = act[i]
            speed_xy = float(np.linalg.norm(a[0:2]))
            if abs(float(a[6]) + 1.0) < 0.05 and speed_xy < 0.15 and abs(float(a[2])) > 0.25:
                lab = "down_chase_open"
            elif abs(float(a[6]) + 1.0) < 0.05 and speed_xy >= 0.15:
                lab = "lateral_chase_open"
            elif abs(float(a[6]) - 1.0) < 0.05 and (speed_xy > 0.15 or abs(float(a[2])) > 0.25):
                lab = "velocity_match_close"
            elif phase[i] == "hold":
                lab = "post_catch_hold"
            else:
                lab = "airborne_other"
            extra_a.append(a)
            extra_y.append(lab)
    all_a = np.stack(skill_actions + extra_a).astype(np.float32)
    all_y = skill_labels + extra_y
    # Physical feature: translation in m/s, rotation in rad/s, grip torque roughly scaled to [-1,1].
    phys = all_a.copy()
    phys[:, 0:3] *= V_AIRBORNE
    phys[:, 3:6] *= W_MAX
    std = phys.std(axis=0)
    std[std < 1e-6] = 1.0
    whitened = ((phys - phys.mean(0)) / std).astype(np.float32)
    raw_fit = kmeans(all_a, 8, 0)
    white_fit = kmeans(whitened, 8, 0)

    def purity(labels_km, names):
        rows = []
        for k in range(8):
            mask = labels_km == k
            if not np.any(mask):
                continue
            labs, cnt = np.unique(np.array(names)[mask], return_counts=True)
            order = np.argsort(-cnt)
            top = [(str(labs[i]), int(cnt[i])) for i in order[:4]]
            rows.append({"k": int(k), "n": int(mask.sum()), "top": top, "majority": round(float(cnt[order[0]] / mask.sum()), 3)})
        return rows

    def separated(labels_km, names, group_a, group_b):
        names = np.array(names)
        la = set(labels_km[np.isin(names, group_a)].tolist())
        lb = set(labels_km[np.isin(names, group_b)].tolist())
        return {
            "a_clusters": sorted(int(x) for x in la),
            "b_clusters": sorted(int(x) for x in lb),
            "overlap": sorted(int(x) for x in (la & lb)),
        }

    local = ["wrist_reseat", "recatch_open20", "gravity_inward"]
    fast = ["down_chase_open", "lateral_chase_open", "velocity_match_close"]
    return {
        "n_actions": int(len(all_a)),
        "label_counts": {k: int(v) for k, v in zip(*np.unique(all_y, return_counts=True))},
        "raw_v2_normalized": {
            "inertia": raw_fit[0],
            "clusters": purity(raw_fit[2], all_y),
            "local_vs_fast": separated(raw_fit[2], all_y, local, fast),
        },
        "whitened_physical": {
            "inertia": white_fit[0],
            "clusters": purity(white_fit[2], all_y),
            "local_vs_fast": separated(white_fit[2], all_y, local, fast),
            "feature_std": [round(float(x), 4) for x in std],
        },
    }


def main():
    self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    sim, cfg = make_sim()
    assert_noslip(sim)
    nom_gains = gains_from_cfg(cfg)
    sim._nom_gains = nom_gains
    nominal = capture_nominal(sim)
    pads, pad_mu = capture_pads(sim)
    base = grasp_settled(sim, nom_gains)
    begin(sim, base, nominal, pads, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0})
    for _ in range(40):
        hold_step(sim, nom_gains)
    supports = drop_supports(sim)
    q_ref, margin0 = limits_of(sim)
    o = physical_pack(sim)
    rel = relative_state(sim.model, sim.data, sim.ids)
    rh0 = rel["p_rel_h"].copy()
    snap = stamp(sim)
    pose = {
        "q": [round(float(x), 4) for x in q_ref],
        "margin": [round(float(x), 3) for x in margin0],
        "rh": [round(float(x), 4) for x in rh0],
        "n": [int(o["nL"]), int(o["nR"])],
        "supports_disabled": supports,
        "joint_velocity_limit": "none in the torque model; armature 0.1, damping 1, torque motors",
        "actuator_torque_nm": [round(float(x), 1) for x in tau_lim(sim)],
        "obs_version": OBS_VERSION,
        "obs_dim": OBS_DIM_AIR,
        "obs_names": OBS_NAMES_AIR,
        "legacy_obs_names": list(OBS_NAMES),
    }
    print("POSE", pose["q"], "n", pose["n"], "obs", OBS_DIM_AIR, flush=True)
    vertical_specs = [
        {"tag": "above", "mode": "above", "lead": 0.0, "k": 4.0, "rz": -0.012, "vrel": 0.45},
        {"tag": "below", "mode": "below", "lead": 0.04, "k": 4.0, "rz": 0.0, "vrel": 0.45},
        {"tag": "match", "mode": "match", "lead": 0.0, "k": 6.0, "rz": -0.012, "vrel": 0.35},
    ]
    results = []
    demos = []
    best = None
    for spec in vertical_specs:
        row = run_chase(sim, snap, nominal, pads, pad_mu, spec, q_ref, rh0, AIRBORNE_V2, TRACK)
        print("VERT", spec["tag"], "ok", row["ok"], "air", row.get("air_s"), "vrel", row.get("vrel_re"), "cmd", row.get("peak_cmd_v"), "ach", row.get("peak_ach_v"), "end", row.get("end_n"), flush=True)
        results.append({"family": "vertical", **{k: v for k, v in row.items() if k not in ("demos", "rows")}})
        if row["ok"]:
            demos.append(pack_demo(row, "vertical_" + spec["tag"]))
            if best is None:
                best = spec
    compare = {}
    if best is not None:
        for version, gains, name in (
            (AIRBORNE_V2, TRACK, "v2_track"),
            (LOCAL_V1, TRACK, "v1_cap_track"),
            (AIRBORNE_V2, nom_gains, "v2_nominal_gains"),
        ):
            row = run_chase(sim, snap, nominal, pads, pad_mu, best, q_ref, rh0, version, gains)
            compare[name] = {k: v for k, v in row.items() if k not in ("demos", "rows")}
            print("CMP", name, row["ok"], row.get("air_s"), row.get("vrel_re"), row.get("peak_cmd_v"), row.get("peak_ach_v"), row.get("end_n"), flush=True)
        perts = [
            {"tag": "dx5", "dx": 0.005},
            {"tag": "dx-5", "dx": -0.005},
            {"tag": "delay30", "open_delay": 0.03},
            {"tag": "q2", "dq": [0, 0.01, 0, 0, 0, 0, 0]},
            {"tag": "higher5", "dz": 0.005},
            {"tag": "deeper5", "dz": -0.005},
        ]
        for pert in perts:
            spec = dict(best)
            spec.update(pert)
            row = run_chase(sim, snap, nominal, pads, pad_mu, spec, q_ref, rh0, AIRBORNE_V2, TRACK)
            results.append({"family": "vertical_perturb", **{k: v for k, v in row.items() if k not in ("demos", "rows")}})
            print("PERT", pert["tag"], row["ok"], row.get("end_n"), row.get("vrel_re"), flush=True)
            if row["ok"]:
                demos.append(pack_demo(row, "vertical_" + pert["tag"]))
    lateral = []
    for speed, sign in ((0.30, 1.0), (0.30, -1.0), (0.45, 1.0)):
        made = carry_then_release(sim, snap, nominal, pads, pad_mu, q_ref, speed, sign, 0.08)
        if made is None:
            print("CARRY_FAIL", speed, sign, flush=True)
            continue
        print("AIR", speed, sign, "vo", made["vo"], "n", made["n"], "z", made["z"], flush=True)
        spec = {"tag": f"lat_{speed}_{sign}", "mode": "below", "lead": 0.03, "k": 5.0, "rz": 0.0, "vrel": 0.55}
        rh_now = relative_state(sim.model, sim.data, sim.ids)["p_rel_h"]
        # The carry function already left sim at the airborne state, but run via snap.
        for version, name in ((AIRBORNE_V2, "v2"), (LOCAL_V1, "v1")):
            row = run_chase(sim, made["snap"], nominal, pads, pad_mu, spec, q_ref, rh_now, version, TRACK)
            rec = {"family": "lateral", "setup_vo": made["vo"], "setup_n": made["n"], "which": name, **{k: v for k, v in row.items() if k not in ("demos", "rows")}}
            lateral.append(rec)
            print("LAT", name, speed, sign, row["ok"], row.get("air_s"), row.get("vrel_re"), "cmd", row.get("peak_cmd_v"), "ach", row.get("peak_ach_v"), row.get("end_n"), flush=True)
            if row["ok"] and version == AIRBORNE_V2:
                demos.append(pack_demo(row, rec["spec"]["tag"] + "_v2"))
    skill, skill_a, skill_y = skill_audit()
    tokens = token_audit(skill_a, skill_y, [d for d in demos if d is not None]) if demos else {}
    saved = []
    for demo in demos:
        if demo is None:
            continue
        path = OUT / f"{demo['tag']}.npz"
        np.savez_compressed(
            path,
            obs=demo["obs"],
            action=demo["action"],
            t=demo["t"],
            phase=demo["phase"],
            obs_names=np.array(OBS_NAMES_AIR),
            trace_t=np.array([r["t"] for r in demo["rows"]], np.float32),
            trace_nL=np.array([r["nL"] for r in demo["rows"]], np.int16),
            trace_nR=np.array([r["nR"] for r in demo["rows"]], np.int16),
            trace_aper=np.array([r["aper"] for r in demo["rows"]], np.float32),
            trace_cmd_v=np.stack([r["cmd_v"] for r in demo["rows"]]).astype(np.float32),
            trace_cmd_w=np.stack([r["cmd_w"] for r in demo["rows"]]).astype(np.float32),
            trace_ach_v=np.stack([r["ach_v"] for r in demo["rows"]]).astype(np.float32),
            trace_obj_v=np.stack([r["obj_v"] for r in demo["rows"]]).astype(np.float32),
            trace_vrel=np.stack([r["vrel"] for r in demo["rows"]]).astype(np.float32),
            trace_prel=np.stack([r["prel"] for r in demo["rows"]]).astype(np.float32),
        )
        saved.append({"tag": demo["tag"], "steps": int(len(demo["t"])), "file": path.name, "summary": demo["summary"]})
    blob = {
        "pose": pose,
        "action": {
            "local_v1": {"name": LOCAL_V1, "v_mps": V_LOCAL, "w_rad_s": W_MAX, "grip": "a=-1 -> +2 Nm, a=+1 -> -18 Nm"},
            "airborne_v2": {"name": AIRBORNE_V2, "v_mps": V_AIRBORNE, "w_rad_s": W_MAX, "grip": "unchanged"},
            "frame": "v = R_hand @ a[0:3]*V; w = R_des @ a[3:6]*4",
        },
        "vertical": results,
        "compare": compare,
        "lateral": lateral,
        "skills": skill,
        "tokens": tokens,
        "demos": saved,
    }
    (OUT / "summary.json").write_text(json.dumps(jsonable(blob), indent=2), encoding="utf-8")
    print("DEMOS", len(saved), "DONE", flush=True)


if __name__ == "__main__":
    main()
