"""Geometry hysteresis for recovery entry and exit.

Nominal control runs until the object pose in the hand leaves the lift
reference for 40 ms. The frozen 39-D policy then runs until a wide stable
grasp has held for 100 ms. Thresholds are one pair for every family.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_data_scaling as scale
import training.natural_loss_recatch_transformer_v3 as v3
from controllers.recovery_actions import AIRBORNE_V2, normalize_action
from envs.airborne_obs import observe_airborne
from envs.observable_obs import ObservableObsState
from envs.physical_recovery import Z_AIR, physical_pack
from controllers.geometry_switch import (
    ENTRY_PERSISTENCE_STEPS as ENTER_STEPS,
    EXIT_STABILITY_STEPS as EXIT_STEPS,
    ENTRY_POSITION_THRESHOLD_M,
    EXIT_ORIENTATION_THRESHOLD_RAD,
    EXIT_POSITION_THRESHOLD_M,
    EXIT_RELATIVE_SPEED_THRESHOLD_MPS,
)
from training.autonomous_episode import nominal_substep, seat_base
from training.balanced_unified_recovery import EVAL_NATURAL
from training.demo_ballistic_impact import HAND_RISE_M, TAU_SEC, assert_noslip, ball_force_on_ball, z_tgt_of
from training.full_3d_airborne_recatch_dataset import TRACK, drop_supports, restore_to, step_mapped
from training.horizontal_grasp_offset_freefall_recatch import grasp_settled
from training.impact_ball import apply_ball_free_state
from training.observation_study_common import load_balanced
from training.phase_decoupled_recovery_pilot import world
from training.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import (
    T_TASK,
    Z_LIFT,
    apply_dyn,
    continue_long,
    launch_offcenter,
    physical_of,
    restore_dyn,
    stamp,
    write_dyn_quiet,
)
from training.replay_core import park_impact_ball, tick_vw

RAW = ROOT / "results" / "diagnostics" / "raw" / "geometry_recovery_switch"
CLEAN = ROOT / "results" / "diagnostics" / "raw" / "balanced_39d" / "summary.json"
Z_ABORT = 0.44
D_ENTER_OLD = 0.85


def hand_pose(sim):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    return rh.T @ (po - ph), rh.T @ ro


def rot_angle(r0, r):
    err = r0.T @ r
    c = float(np.clip(0.5 * (np.trace(err) - 1.0), -1.0, 1.0))
    return float(np.arccos(c))


def vrel_of(sim):
    from envs.deterioration import body_twist
    vo, _wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, _wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    return float(np.linalg.norm(np.asarray(vo, float) - np.asarray(vh, float)))


class Geom:
    def __init__(self):
        self.p0 = None
        self.r0 = None
        self.pos_run = 0
        self.rot_run = 0
        self.exit_run = 0
        self.t_old = None
        self.old_pose = None

    def maybe_capture(self, sim):
        if self.p0 is None and sim.captured and str(sim.fsm.phase) == "lift":
            self.p0, self.r0 = hand_pose(sim)

    def sample(self, sim, d_enter, theta_enter, use_rot):
        self.maybe_capture(sim)
        o = physical_pack(sim)
        t = float(sim.data.time)
        if self.p0 is None:
            e_pos, e_rot = 0.0, 0.0
        else:
            p, r = hand_pose(sim)
            e_pos = float(np.linalg.norm(p - self.p0))
            e_rot = rot_angle(self.r0, r)
        snap = sim.meter.compute(sim.model, sim.data, sim.ids, v3.DT_POLICY)
        if self.p0 is not None and self.t_old is None and float(snap.D) > D_ENTER_OLD:
            self.t_old = t
            self.old_pose = {
                "t": round(t, 4),
                "D": round(float(snap.D), 4),
                "e_pos": round(e_pos, 5),
                "e_rot": round(e_rot, 5),
                "n": [int(o["nL"]), int(o["nR"])],
                "z": round(float(o["obj_z"]), 4),
            }
        pos_hit = self.p0 is not None and e_pos > d_enter
        rot_hit = self.p0 is not None and use_rot and e_rot > theta_enter
        self.pos_run = self.pos_run + 1 if pos_hit else 0
        self.rot_run = self.rot_run + 1 if rot_hit else 0
        reason = None
        if self.pos_run >= ENTER_STEPS and self.rot_run >= ENTER_STEPS:
            reason = "both"
        elif self.pos_run >= ENTER_STEPS:
            reason = "position"
        elif self.rot_run >= ENTER_STEPS:
            reason = "orientation"
        return {
            "t": round(t, 4),
            "e_pos": round(e_pos, 5),
            "e_rot": round(e_rot, 5),
            "vrel": round(vrel_of(sim), 4),
            "nL": int(o["nL"]),
            "nR": int(o["nR"]),
            "z": round(float(o["obj_z"]), 4),
            "D": round(float(snap.D), 4),
            "reason": reason,
        }


def nominal_block(sim, gains, st, n=v3.N_SUB):
    for _ in range(n):
        nominal_substep(sim, gains, st)
        if float(sim.data.time) >= T_TASK - 1e-9:
            return


def empty_ball():
    return {
        "holding": False,
        "hand_z0": None,
        "z_tgt": None,
        "released": False,
        "parked": False,
        "shifted": False,
        "t_impact": None,
        "t_last": None,
        "t_park": None,
        "t_release": None,
    }


def ball_substep(sim, gains, nominal, ball, v_hit, z_off, phi, delta):
    nominal_substep(sim, gains, ball)
    t = float(sim.data.time)
    o = physical_pack(sim)
    hz = float(sim.data.xpos[sim.ids.hand_body][2])
    if (
        (not ball["released"])
        and sim.fsm.phase == "lift"
        and sim.captured
        and ball["hand_z0"] is not None
        and float(o["obj_z"]) >= Z_AIR
        and (hz - ball["hand_z0"]) >= HAND_RISE_M
    ):
        write_dyn_quiet(sim, nominal, None, None)
        launch = launch_offcenter(sim, v_hit, z_off, phi)
        sim.set_guide_weld(False)
        apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
        ball["released"] = True
        ball["t_release"] = float(sim.data.time)
    if ball["released"] and not ball["parked"]:
        _fb, bc = ball_force_on_ball(sim)
        has = any(r["kind"] == "ball-cylinder" for r in bc)
        if has and ball["t_impact"] is None:
            ball["t_impact"] = t
        if has:
            ball["t_last"] = t
        departed = (
            ball["t_impact"] is not None
            and ball["t_last"] is not None
            and (not has)
            and t > ball["t_last"] + 0.025
        )
        stuck = ball["t_impact"] is not None and t > ball["t_impact"] + 0.35
        if departed or stuck:
            park_impact_ball(sim)
            ball["parked"] = True
            ball["t_park"] = float(sim.data.time)
    if delta is not None and ball["parked"] and (not ball["shifted"]) and ball["t_park"] is not None and t >= ball["t_park"] + 0.040:
        adr = int(sim.model.jnt_qposadr[int(sim.ids.object_jnt)])
        sim.data.qpos[adr:adr + 3] = sim.data.qpos[adr:adr + 3] + np.asarray(delta, float)
        mujoco.mj_forward(sim.model, sim.data)
        ball["shifted"] = True
    return t


def trace_ball(sim, gains, nominal, v_hit, z_off, phi, delta=None):
    """Nominal through the ball. No policy. Snap at park+40 ms, with p0 from lift."""
    assert_noslip(sim)
    sim.reset()
    apply_dyn(sim, nominal, None, None)
    mujoco.mj_forward(sim.model, sim.data)
    ball = empty_ball()
    geom = Geom()
    rows = []
    snap = None
    while float(sim.data.time) < T_TASK - 1e-9:
        for _ in range(v3.N_SUB):
            ball_substep(sim, gains, nominal, ball, v_hit, z_off, phi, delta)
            if snap is None and ball["parked"] and ball["t_park"] is not None and float(sim.data.time) >= ball["t_park"] + 0.040:
                if delta is None or ball["shifted"]:
                    snap = stamp(sim)
        row = geom.sample(sim, 1e9, 1e9, False)
        rows.append(row)
        if snap is not None and float(sim.data.time) >= ball["t_park"] + 0.24:
            break
        if ball["released"] and row["z"] < Z_ABORT:
            break
        if (not ball["released"]) and float(sim.data.time) > 8.0:
            break
    return {
        "rows": rows,
        "snap": snap,
        "p0": None if geom.p0 is None else geom.p0.copy(),
        "r0": None if geom.r0 is None else geom.r0.copy(),
        "ball": {k: ball[k] for k in ("t_release", "t_impact", "t_park", "released", "shifted")},
        "old": geom.old_pose,
    }


def trace_healthy_lift(sim, gains, nominal):
    assert_noslip(sim)
    sim.reset()
    apply_dyn(sim, nominal, None, None)
    mujoco.mj_forward(sim.model, sim.data)
    st = {"holding": False, "hand_z0": None, "z_tgt": None}
    geom = Geom()
    rows = []
    while float(sim.data.time) < T_TASK - 1e-9:
        nominal_block(sim, gains, st)
        rows.append(geom.sample(sim, 1e9, 1e9, False))
    return rows, geom


def trace_hold(sim, seconds=4.0):
    geom = Geom()
    rows = []
    t1 = float(sim.data.time) + seconds
    st = {"holding": False, "hand_z0": None, "z_tgt": None}
    while float(sim.data.time) < t1:
        nominal_block(sim, TRACK, st)
        rows.append(geom.sample(sim, 1e9, 1e9, False))
    return rows


def peak_after(rows, t0):
    sub = [r for r in rows if r["t"] + 1e-9 >= t0]
    if not sub:
        return None
    return {
        "max_pos": max(r["e_pos"] for r in sub),
        "max_rot": max(r["e_rot"] for r in sub),
        "n": len(sub),
    }


def first_persist(rows, key, limit, steps, t_min=None):
    run = 0
    for r in rows:
        if t_min is not None and r["t"] < t_min:
            run = 0
            continue
        if r[key] > limit:
            run += 1
            if run >= steps:
                return r["t"]
        else:
            run = 0
    return None


def play_suffix(sim, actions, index):
    for i, act in enumerate(actions):
        if i == index:
            return True
        v3.step_v3(sim, v3.v1_normalized_to_v3(np.asarray(act, np.float32)))
        if float(physical_pack(sim)["obj_z"]) < Z_ABORT:
            return False
    return False


def rollout_pose(sim, model, p0, r0, n_steps):
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    device = next(model.parameters()).device
    rows = []
    model.eval()
    with torch.no_grad():
        for _k in range(int(n_steps)):
            if float(sim.data.time) >= T_TASK - 1e-9:
                break
            obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            obs_seq.append(np.asarray(obs, np.float32))
            prev_seq.append(prev.copy())
            pred = model(
                torch.tensor(np.stack(obs_seq), device=device),
                torch.tensor(np.stack(prev_seq), device=device),
            )[2][-1].float().cpu().numpy()
            act = np.clip(pred, -1.0, 1.0).astype(np.float32)
            v3.step_v3(sim, act)
            prev = act
            p, r = hand_pose(sim)
            o = physical_pack(sim)
            rows.append({
                "t": round(float(sim.data.time), 4),
                "e_pos": round(float(np.linalg.norm(p - p0)), 5),
                "e_rot": round(rot_angle(r0, r), 5),
                "vrel": round(vrel_of(sim), 4),
                "nL": int(o["nL"]),
                "nR": int(o["nR"]),
                "z": round(float(o["obj_z"]), 4),
            })
            if float(o["obj_z"]) < Z_ABORT:
                break
    return rows


def handoff_pose(rows):
    if not rows:
        return None
    end = rows[-1]
    if end["z"] < Z_ABORT:
        return None
    return end


def policy_until(sim, model, geom, limits, mode_steps):
    """Run the frozen policy. mode_steps is the fixed budget, or None for geometry exit."""
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    device = next(model.parameters()).device
    steps = 0
    exit_kind = "none"
    last = None
    model.eval()
    with torch.no_grad():
        while float(sim.data.time) < T_TASK - 1e-9:
            if mode_steps is not None and steps >= int(mode_steps):
                exit_kind = "fixed_horizon"
                break
            obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            obs_seq.append(np.asarray(obs, np.float32))
            prev_seq.append(prev.copy())
            pred = model(
                torch.tensor(np.stack(obs_seq), device=device),
                torch.tensor(np.stack(prev_seq), device=device),
            )[2][-1].float().cpu().numpy()
            act = np.clip(pred, -1.0, 1.0).astype(np.float32)
            v3.step_v3(sim, act)
            prev = act
            steps += 1
            row = geom.sample(sim, limits["d_enter"], limits["theta_enter"], limits["use_rot"])
            last = row
            stable = (
                row["nL"] > 0
                and row["nR"] > 0
                and row["e_pos"] <= limits["d_exit"]
                and row["e_rot"] <= limits["theta_exit"]
                and row["vrel"] <= limits["v_exit"]
            )
            geom.exit_run = geom.exit_run + 1 if stable else 0
            if row["z"] < Z_ABORT:
                exit_kind = "fall_abort"
                break
            if mode_steps is None and geom.exit_run >= EXIT_STEPS:
                exit_kind = "geometry"
                break
        else:
            if exit_kind == "none":
                exit_kind = "time_cap"
    if exit_kind == "fall_abort":
        t12 = "DROP"
    else:
        rec = continue_long(sim, TRACK, T_TASK, "geom")
        y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
        t12 = "RETAINED_TO_12" if y == 1 else kind
    return {
        "t12": t12,
        "exit_kind": exit_kind,
        "steps": steps,
        "t_exit": None if last is None else last["t"],
        "e_pos_exit": None if last is None else last["e_pos"],
        "e_rot_exit": None if last is None else last["e_rot"],
        "vrel_exit": None if last is None else last["vrel"],
        "n_exit": None if last is None else [last["nL"], last["nR"]],
    }


def online_ball(sim, gains, nominal, pads, pad_mu, model, limits, v_hit, z_off, phi, delta=None):
    assert_noslip(sim)
    sim.reset()
    apply_dyn(sim, nominal, None, None)
    mujoco.mj_forward(sim.model, sim.data)
    ball = empty_ball()
    geom = Geom()
    entered = None
    contact_pose = None
    while float(sim.data.time) < T_TASK - 1e-9 and entered is None:
        for _ in range(v3.N_SUB):
            ball_substep(sim, gains, nominal, ball, v_hit, z_off, phi, delta)
            if float(sim.data.time) >= T_TASK - 1e-9:
                break
        row = geom.sample(sim, limits["d_enter"], limits["theta_enter"], limits["use_rot"])
        if ball["t_impact"] is not None and contact_pose is None and row["t"] + 1e-9 >= ball["t_impact"]:
            contact_pose = {
                "t": row["t"], "e_pos": row["e_pos"], "e_rot": row["e_rot"],
                "n": [row["nL"], row["nR"]], "z": row["z"], "D": row["D"],
            }
        if row["reason"] is not None and ball["released"]:
            entered = row
            break
        if ball["released"] and row["z"] < Z_ABORT:
            break
        if (not ball["released"]) and float(sim.data.time) > 8.0:
            break
    out = {
        "entered": entered is not None,
        "t_release": ball["t_release"],
        "t_impact": ball["t_impact"],
        "t_park": ball["t_park"],
        "t_old": None if geom.old_pose is None else geom.old_pose["t"],
        "old_pose": geom.old_pose,
        "contact_pose": contact_pose,
        "t_geom": None if entered is None else entered["t"],
        "reason": None if entered is None else entered["reason"],
        "geom_pose": None if entered is None else {
            "e_pos": entered["e_pos"], "e_rot": entered["e_rot"],
            "n": [entered["nL"], entered["nR"]], "z": entered["z"], "vrel": entered["vrel"],
        },
    }
    if entered is None:
        rec = continue_long(sim, gains, T_TASK, "nominal")
        y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
        out["policy"] = {"t12": "RETAINED_TO_12" if y == 1 else kind, "exit_kind": "no_entry", "steps": 0,
                         "t_exit": None, "e_pos_exit": None, "e_rot_exit": None, "vrel_exit": None, "n_exit": None}
        return out
    # Keep the live state. Do not restore a snapshot.
    out["policy"] = policy_until(sim, model, geom, limits, None)
    return out


def online_carry(sim, nominal, pads, pad_mu, base, q_ref, model, limits, speed, sign, dur):
    info = restore_to(sim, base, nominal, pads, pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        raise RuntimeError("carry restore failed")
    geom = Geom()
    geom.maybe_capture(sim)
    if geom.p0 is None:
        geom.p0, geom.r0 = hand_pose(sim)
    t0 = float(sim.data.time)
    open_at = t0 + float(dur)
    opened = False
    lost = False
    open_hold = 0
    entered = None
    st = {"holding": False, "hand_z0": None, "z_tgt": None}
    while float(sim.data.time) < T_TASK - 1e-9 and entered is None:
        for _ in range(v3.N_SUB):
            t = float(sim.data.time)
            if open_hold < 10 and t < open_at + 0.20:
                v_h = np.zeros(3)
                if t < open_at:
                    v_h[1] = float(sign) * float(speed)
                    tau = -18.0
                else:
                    opened = True
                    v_h[2] = 0.3
                    tau = 2.0
                action = normalize_action(v_h, np.zeros(3), tau, AIRBORNE_V2)
                step_mapped(sim, action, AIRBORNE_V2, TRACK, q_ref)
                o = physical_pack(sim)
                if opened and int(o["nL"]) == 0 and int(o["nR"]) == 0:
                    lost = True
                    open_hold += 1
            else:
                nominal_substep(sim, TRACK, st)
        row = geom.sample(sim, limits["d_enter"], limits["theta_enter"], limits["use_rot"])
        if row["reason"] is not None:
            entered = row
            break
        if lost and open_hold >= 10 and row["z"] < Z_ABORT:
            break
    out = {
        "entered": entered is not None,
        "t_release": t0,
        "t_impact": open_at,
        "t_old": None if geom.old_pose is None else geom.old_pose["t"],
        "old_pose": geom.old_pose,
        "t_geom": None if entered is None else entered["t"],
        "reason": None if entered is None else entered["reason"],
        "geom_pose": None if entered is None else {
            "e_pos": entered["e_pos"], "e_rot": entered["e_rot"],
            "n": [entered["nL"], entered["nR"]], "z": entered["z"], "vrel": entered["vrel"],
        },
        "opened": opened,
        "lost": lost,
    }
    if entered is None:
        rec = continue_long(sim, TRACK, T_TASK, "nominal")
        y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
        out["policy"] = {"t12": "RETAINED_TO_12" if y == 1 else kind, "exit_kind": "no_entry", "steps": 0,
                         "t_exit": None, "e_pos_exit": None, "e_rot_exit": None, "vrel_exit": None, "n_exit": None}
        return out
    out["policy"] = policy_until(sim, model, geom, limits, None)
    return out


def online_natural(sim, nominal, pads, pad_mu, base, model, limits, spec):
    info = restore_to(sim, base, nominal, pads, pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        raise RuntimeError("natural restore failed")
    geom = Geom()
    geom.maybe_capture(sim)
    if geom.p0 is None:
        geom.p0, geom.r0 = hand_pose(sim)
    # One sample of the seated grasp, then the offset. The reference stays the seated pose.
    geom.sample(sim, 1e9, 1e9, False)
    scale.apply_spec(sim, spec)
    t_off = float(sim.data.time)
    entered = None
    st = {"holding": False, "hand_z0": None, "z_tgt": None}
    while float(sim.data.time) < T_TASK - 1e-9 and entered is None:
        nominal_block(sim, TRACK, st)
        row = geom.sample(sim, limits["d_enter"], limits["theta_enter"], limits["use_rot"])
        if row["reason"] is not None:
            entered = row
            break
        if row["z"] < Z_ABORT:
            break
    out = {
        "entered": entered is not None,
        "t_release": t_off,
        "t_impact": t_off,
        "t_old": None if geom.old_pose is None else geom.old_pose["t"],
        "old_pose": geom.old_pose,
        "t_geom": None if entered is None else entered["t"],
        "reason": None if entered is None else entered["reason"],
        "geom_pose": None if entered is None else {
            "e_pos": entered["e_pos"], "e_rot": entered["e_rot"],
            "n": [entered["nL"], entered["nR"]], "z": entered["z"], "vrel": entered["vrel"],
        },
    }
    if entered is None:
        rec = continue_long(sim, TRACK, T_TASK, "nominal")
        y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
        out["policy"] = {"t12": "RETAINED_TO_12" if y == 1 else kind, "exit_kind": "no_entry", "steps": 0,
                         "t_exit": None, "e_pos_exit": None, "e_rot_exit": None, "vrel_exit": None, "n_exit": None}
        return out
    out["policy"] = policy_until(sim, model, geom, limits, None)
    return out


def max_of(rows):
    if not rows:
        return {"max_pos": None, "max_rot": None}
    return {
        "max_pos": max(r["e_pos"] for r in rows),
        "max_rot": max(r["e_rot"] for r in rows),
    }


def false_count(rows, limits):
    """How many times a fresh run would enter. One episode enters at most once."""
    pos_run = rot_run = 0
    for r in rows:
        pos_run = pos_run + 1 if r["e_pos"] > limits["d_enter"] else 0
        rot_run = rot_run + 1 if limits["use_rot"] and r["e_rot"] > limits["theta_enter"] else 0
        if pos_run >= ENTER_STEPS or rot_run >= ENTER_STEPS:
            return 1
    return 0


def window_after(rows, t0, n=8):
    sub = [r for r in rows if t0 is None or r["t"] + 1e-9 >= t0]
    return [{k: r[k] for k in ("t", "e_pos", "e_rot", "nL", "nR", "z", "D")} for r in sub[:n]]


def choose_limits(healthy_rows, family_stats, handoffs):
    """Entry sits above the undisturbed grasp. Exit covers the retained handoffs.

    Those handoffs are about 12 mm and 0.30 rad from the lift pose, which is
    the same offset the ball leaves behind. An exit ball smaller than the
    entry line would reject every retained impact, slide, wrist, and recatch.
    The handoff is one-way, so nominal control is not asked to re-enter.
    """
    h_pos = max(r["e_pos"] for r in healthy_rows)
    h_rot = max(r["e_rot"] for r in healthy_rows)
    d_enter = ENTRY_POSITION_THRESHOLD_M
    # Orientation is recorded for the family table. It is not an entry signal.
    theta_enter = 0.10
    d_exit = EXIT_POSITION_THRESHOLD_M
    theta_exit = EXIT_ORIENTATION_THRESHOLD_RAD
    v_exit = EXIT_RELATIVE_SPEED_THRESHOLD_MPS
    family_need = {}
    use_rot = False
    for fam, st in family_stats.items():
        pos_t = first_persist(st["rows"], "e_pos", d_enter, ENTER_STEPS, st.get("t_min"))
        rot_t = first_persist(st["rows"], "e_rot", theta_enter, ENTER_STEPS, st.get("t_min"))
        family_need[fam] = {
            "max_pos": st["max_pos"],
            "max_rot": st["max_rot"],
            "t_pos": pos_t,
            "t_rot": rot_t,
            "position_detects": pos_t is not None,
            "orientation_only": pos_t is None and rot_t is not None,
        }
    return {
        "d_enter": float(d_enter),
        "theta_enter": float(theta_enter),
        "d_exit": float(d_exit),
        "theta_exit": float(theta_exit),
        "v_exit": float(v_exit),
        "use_rot": bool(use_rot),
        "healthy_max_pos": float(h_pos),
        "healthy_max_rot": float(h_rot),
        "families": family_need,
        "enter_window_s": (ENTER_STEPS - 1) * v3.DT_POLICY,
        "exit_window_s": EXIT_STEPS * v3.DT_POLICY,
        "hysteresis_note": "exit ball is wider than the entry line because retained handoffs do not return inside it",
    }


def slim_trace(tr):
    ball = tr["ball"]
    t_hit = ball.get("t_impact")
    t_des = None if ball.get("t_park") is None else ball["t_park"] + 0.040
    before = [r for r in tr["rows"] if t_des is None or r["t"] <= t_des + 1e-9]
    return {
        "t_release": ball.get("t_release"),
        "t_impact": t_hit,
        "t_park": ball.get("t_park"),
        "max_pos_before_park40": None if not before else max(r["e_pos"] for r in before),
        "max_rot_before_park40": None if not before else max(r["e_rot"] for r in before),
        "after_contact": window_after(tr["rows"], t_hit),
        "old": tr["old"],
    }


def main():
    raise SystemExit(
        "Geometry thresholds are frozen in config/geometry_switch.yaml. "
        "The online path is training/autonomous_episode.py. "
        "Running this study again would rewrite the pre-training geometry log."
    )
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _proto, _saved, _device = load_balanced(device)
    sim, gains, nominal, pads, pad_mu, _pl, _pr = world()
    print("HEALTHY LIFT", flush=True)
    lift_rows, _g = trace_healthy_lift(sim, gains, nominal)
    print("  max", max_of(lift_rows), flush=True)
    print("SEAT HOLD", flush=True)
    seat = grasp_settled(sim, TRACK)
    if seat is None:
        raise RuntimeError("seat failed")
    hold_rows = trace_hold(sim, 4.0)
    print("  max", max_of(hold_rows), flush=True)
    healthy_rows = lift_rows + hold_rows

    specs = {
        "zm6": (6.5, -0.006, 0.0, None),
        "zp6": (6.5, 0.006, 0.0, None),
        "lat": (6.5, 0.0, 90.0, None),
        "z3": (6.5, -0.003, 0.0, None),
        "dy1": (6.5, -0.006, 0.0, (0.0, 0.001, 0.0)),
        "dz1": (6.5, -0.006, 0.0, (0.0, 0.0, 0.001)),
    }
    traces = {}
    for key, (v_hit, z_off, phi, delta) in specs.items():
        print("TRACE", key, flush=True)
        traces[key] = trace_ball(sim, gains, nominal, v_hit, z_off, phi, delta)
        tr = traces[key]
        print("  hit", tr["ball"].get("t_impact"), "park", tr["ball"].get("t_park"), "old", tr["old"], flush=True)

    sch = schedules()
    handoff_plan = [
        ("impact_zm6", "impact", "zm6", None, None, 56),
        ("impact_phi90", "impact", "lat", None, None, 56),
        ("impact_z3", "impact", "z3", None, None, 56),
        ("slide_suffix_20", "sliding", "zm6", "gravity_inward", 20, 36),
        ("slide_suffix_36", "sliding", "zm6", "gravity_inward", 36, 20),
        ("slide_dy1", "sliding", "dy1", None, None, 56),
        ("slide_dz1", "sliding", "dz1", None, None, 56),
        ("wrist_suffix_3", "wrist", "zm6", "wrist_reseat", 3, 53),
        ("wrist_suffix_6", "wrist", "zm6", "wrist_reseat", 6, 50),
        ("wrist_suffix_9", "wrist", "zm6", "wrist_reseat", 9, 47),
        ("wrist_suffix_12", "wrist", "zm6", "wrist_reseat", 12, 44),
        ("recatch_suffix_8", "recatch", "zm6", "recatch_open20", 8, 48),
        ("recatch_suffix_16", "recatch", "zm6", "recatch_open20", 16, 40),
        ("recatch_suffix_24", "recatch", "zm6", "recatch_open20", 24, 32),
        ("recatch_suffix_28", "recatch", "zm6", "recatch_open20", 28, 28),
    ]
    handoffs = []
    for cid, fam, key, sched_name, index, n_steps in handoff_plan:
        tr = traces[key]
        if tr["snap"] is None or tr["p0"] is None:
            print("REPLAY MISSING", cid, flush=True)
            handoffs.append({"id": cid, "family": fam, "pose": None})
            continue
        info = restore_dyn(sim, tr["snap"], nominal, pads, pad_mu, None, None)
        park_impact_ball(sim)
        ok = bool(info.get("pose_ok", False))
        if ok and sched_name is not None:
            ok = play_suffix(sim, sch[sched_name], index)
        if not ok:
            print("REPLAY FAIL", cid, flush=True)
            handoffs.append({"id": cid, "family": fam, "pose": None})
            continue
        p, r = hand_pose(sim)
        start = {
            "e_pos": round(float(np.linalg.norm(p - tr["p0"])), 5),
            "e_rot": round(rot_angle(tr["r0"], r), 5),
        }
        rows = rollout_pose(sim, model, tr["p0"], tr["r0"], n_steps)
        pose = handoff_pose(rows)
        handoffs.append({"id": cid, "family": fam, "start": start, "pose": pose, "steps": len(rows)})
        print("REPLAY", cid, "start", start, "handoff", pose, flush=True)

    family_stats = {}
    for key, tr in traces.items():
        rows = tr["rows"]
        t_min = tr["ball"].get("t_release")
        family_stats[key] = {
            "rows": rows,
            "max_pos": max((r["e_pos"] for r in rows), default=0.0),
            "max_rot": max((r["e_rot"] for r in rows), default=0.0),
            "t_min": t_min,
        }
    # Suffix starts are not a second disturbance. They show whether a wrist or
    # open-finger state that the benchmark recovers is invisible to position.
    for h in handoffs:
        if not h.get("start"):
            continue
        family_stats[h["id"]] = {
            "rows": [{
                "t": 1.0, "e_pos": h["start"]["e_pos"], "e_rot": h["start"]["e_rot"],
            }],
            "max_pos": h["start"]["e_pos"],
            "max_rot": h["start"]["e_rot"],
            "t_min": None,
        }
    limits = choose_limits(healthy_rows, family_stats, handoffs)
    print("LIMITS", {k: limits[k] for k in ("d_enter", "theta_enter", "d_exit", "theta_exit", "v_exit", "use_rot")}, flush=True)
    print("FAMILIES", limits["families"], flush=True)

    # Online suite. Impacts while the table is still active.
    online = {}
    for key, (v_hit, z_off, phi, delta) in specs.items():
        print("ONLINE", key, flush=True)
        online[key] = online_ball(sim, gains, nominal, pads, pad_mu, model, limits, v_hit, z_off, phi, delta)
        ep = online[key]
        print("  geom", ep["t_geom"], ep["reason"], "old", ep["t_old"], "t12", ep["policy"]["t12"], flush=True)

    print("SEAT BASE", flush=True)
    base, q_ref = seat_base(sim, nominal, pads, pad_mu)
    # Carry and natural traces for the family table, then the online runs.
    # The online run is also the detection measurement for these families.
    carries = (
        ("air_lat_pos", 0.30, 1.0),
        ("air_lat_neg", 0.30, -1.0),
        ("air_fast_pos", 0.45, 1.0),
        ("air_slow_neg", 0.20, -1.0),
    )
    for key, speed, sign in carries:
        print("ONLINE", key, flush=True)
        online[key] = online_carry(sim, nominal, pads, pad_mu, base, q_ref, model, limits, speed, sign, 0.08)
        ep = online[key]
        print("  geom", ep["t_geom"], ep["reason"], "old", ep["t_old"], "t12", ep["policy"]["t12"], flush=True)
    for spec in EVAL_NATURAL:
        key = "nl_" + spec["tag"]
        print("ONLINE", key, flush=True)
        online[key] = online_natural(sim, nominal, pads, pad_mu, base, model, limits, spec)
        ep = online[key]
        print("  geom", ep["t_geom"], ep["reason"], "old", ep["t_old"], "t12", ep["policy"]["t12"], flush=True)

    published = {r["id"]: r for r in json.loads(CLEAN.read_text(encoding="utf-8"))["clean"]["rows"]}
    # Previous autonomous C result, fixed from the geometry-meter study.
    old_online = {
        "impact_zm6": "DROP", "impact_zp6": "DROP", "impact_phi90": "DROP", "impact_z3": "DROP",
        "slide_suffix_20": "DROP", "slide_suffix_36": "DROP", "slide_dy1": "RETAINED_TO_12", "slide_dz1": "RETAINED_TO_12",
        "wrist_suffix_3": "DROP", "wrist_suffix_6": "DROP", "wrist_suffix_9": "DROP", "wrist_suffix_12": "DROP",
        "recatch_suffix_8": "DROP", "recatch_suffix_16": "DROP", "recatch_suffix_24": "DROP", "recatch_suffix_28": "DROP",
        "air_lat_pos": "RETAINED_TO_12", "air_lat_neg": "DROP", "air_fast_pos": "RETAINED_TO_12", "air_slow_neg": "RETAINED_TO_12",
        "nl_h_dz14.00": "DROP", "nl_h_dz14.25": "DROP", "nl_h_dy": "DROP", "nl_h_dx": "DROP",
    }
    case_map = [
        ("impact_zm6", "impact", "zm6"),
        ("impact_zp6", "impact", "zp6"),
        ("impact_phi90", "impact", "lat"),
        ("impact_z3", "impact", "z3"),
        ("slide_suffix_20", "sliding", "zm6"),
        ("slide_suffix_36", "sliding", "zm6"),
        ("slide_dy1", "sliding", "dy1"),
        ("slide_dz1", "sliding", "dz1"),
        ("wrist_suffix_3", "wrist", "zm6"),
        ("wrist_suffix_6", "wrist", "zm6"),
        ("wrist_suffix_9", "wrist", "zm6"),
        ("wrist_suffix_12", "wrist", "zm6"),
        ("recatch_suffix_8", "recatch", "zm6"),
        ("recatch_suffix_16", "recatch", "zm6"),
        ("recatch_suffix_24", "recatch", "zm6"),
        ("recatch_suffix_28", "recatch", "zm6"),
        ("air_lat_pos", "airborne", "air_lat_pos"),
        ("air_lat_neg", "airborne", "air_lat_neg"),
        ("air_fast_pos", "airborne", "air_fast_pos"),
        ("air_slow_neg", "airborne", "air_slow_neg"),
        ("nl_h_dz14.00", "natural", "nl_h_dz14.00"),
        ("nl_h_dz14.25", "natural", "nl_h_dz14.25"),
        ("nl_h_dy", "natural", "nl_h_dy"),
        ("nl_h_dx", "natural", "nl_h_dx"),
    ]
    rows = []
    for cid, fam, key in case_map:
        ep = online[key]
        pol = ep["policy"]
        rows.append({
            "id": cid,
            "family": fam,
            "disturbance": key,
            "A": published[cid]["t12"],
            "B": old_online[cid],
            "C": pol["t12"],
            "t_old": ep.get("t_old"),
            "t_geom": ep.get("t_geom"),
            "reason": ep.get("reason"),
            "t_exit": pol.get("t_exit"),
            "e_pos_exit": pol.get("e_pos_exit"),
            "e_rot_exit": pol.get("e_rot_exit"),
            "vrel_exit": pol.get("vrel_exit"),
            "exit_kind": pol.get("exit_kind"),
            "steps": pol.get("steps"),
            "t_impact": ep.get("t_impact"),
            "old_pose": ep.get("old_pose"),
            "geom_pose": ep.get("geom_pose"),
            "contact_pose": ep.get("contact_pose"),
        })
        print("CASE", cid, rows[-1]["A"], rows[-1]["B"], rows[-1]["C"], rows[-1]["reason"], flush=True)

    limits_out = {k: v for k, v in limits.items() if k != "families"}
    limits_out["families"] = limits["families"]
    # Drop the raw row lists from the saved family stats. choose_limits already
    # replaced them with the summary dict.
    payload = {
        "limits": limits_out,
        "healthy_lift": max_of(lift_rows),
        "healthy_hold": max_of(hold_rows),
        "false_entry_lift": false_count(lift_rows, limits),
        "false_entry_hold": false_count(hold_rows, limits),
        "traces": {k: slim_trace(v) for k, v in traces.items()},
        "handoffs": handoffs,
        "rows": rows,
        "A_ok": sum(r["A"] == "RETAINED_TO_12" for r in rows),
        "B_ok": sum(r["B"] == "RETAINED_TO_12" for r in rows),
        "C_ok": sum(r["C"] == "RETAINED_TO_12" for r in rows),
    }
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", RAW / "summary.json", "A", payload["A_ok"], "B", payload["B_ok"], "C", payload["C_ok"], flush=True)


if __name__ == "__main__":
    main()
