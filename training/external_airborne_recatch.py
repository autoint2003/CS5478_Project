"""Externally induced airborne recatch. No training, no pre-impact recovery.

The gripper stays on the nominal secure torque through the ball collision.
A recovery command is applied only after the impact has already held both
pads off for 20 ms. The cylinder is never teleported.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.gripper_controller import finger_opening
from envs.deterioration import body_twist
from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from training.demo_ballistic_impact import HAND_RISE_M, assert_noslip, ball_force_on_ball, z_tgt_of
from training.impact_ball import apply_ball_free_state, incoming_hand_x
from training.long_context_recovery_pilot import jsonable
from training.recovery_policy_pretraining_preparation import embed4, schedules
from training.recovery_runtime import (
    _dropped,
    _escaped,
    stamp,
    step_cmd,
)
from training.write_ballistic_impact_scene import BALL_R, CYL_R

OUT = ROOT / "results" / "diagnostics" / "raw" / "external_airborne_recatch"
VID = OUT / "videos"
PERSIST_STEPS = 10
SECURE_TAU = -17.0


def launch_aim(sim, v_hit: float, z_off: float, phi_deg: float, elev_deg: float = 0.0) -> dict:
    """Ballistic aim. phi is the contact azimuth; elev tilts the incoming velocity."""
    u = incoming_hand_x(sim).astype(float)
    elev = np.radians(float(elev_deg))
    up = np.array([0.0, 0.0, 1.0])
    u = u * np.cos(elev) + up * np.sin(elev)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    vo, _wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    axis = Ro[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    u_perp = u - axis * float(np.dot(axis, u))
    nrm = float(np.linalg.norm(u_perp))
    if nrm < 1e-6:
        u_perp = u
        nrm = float(np.linalg.norm(u_perp))
    e_face = -u_perp / nrm
    e_tan = np.cross(axis, e_face)
    e_tan = e_tan / max(float(np.linalg.norm(e_tan)), 1e-12)
    phi = np.radians(float(phi_deg))
    flight = 0.10
    g = np.array(sim.model.opt.gravity, float)
    po_pred = po + vo * flight
    p_contact = po_pred + axis * float(z_off) + e_face * (CYL_R * np.cos(phi)) + e_tan * (CYL_R * np.sin(phi))
    n_out = e_face * np.cos(phi) + e_tan * np.sin(phi)
    p_ball = p_contact + n_out * BALL_R
    p_launch = p_ball - u * (float(v_hit) * flight)
    v0 = (p_ball - p_launch) / flight - 0.5 * g * flight
    miss = float(np.linalg.norm(np.cross(p_ball - po_pred, u)))
    return {"p_launch": p_launch, "v0": v0, "miss_mm": miss * 1e3}


def momentum(sim) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bid = sim.ids.object_body
    R = np.array(sim.data.xmat[bid].reshape(3, 3), float)
    inertia = np.diag(np.array(sim.model.body_inertia[bid], float))
    world = R @ inertia @ R.T
    vo, wo = body_twist(sim.model, sim.data, bid)
    mass = float(sim.model.body_mass[bid])
    linear = mass * np.asarray(vo, float)
    angular = world @ np.asarray(wo, float)
    return linear, angular, np.asarray(vo, float), np.asarray(wo, float)


def tilt_deg(sim) -> float:
    axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
    up = abs(float(np.dot(axis, np.array([0.0, 0.0, 1.0]))))
    return float(np.degrees(np.arccos(np.clip(up, 0.0, 1.0))))


def pack_now(sim) -> dict:
    o = physical_pack(sim)
    _lin, _ang, vo, wo = momentum(sim)
    rh = np.asarray(o["rh"], float)
    rel = np.asarray(o["v_rel_h"], float)
    return {
        "t": round(float(sim.data.time), 4),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "FnL": round(float(o["Fn_L"]), 3),
        "FnR": round(float(o["Fn_R"]), 3),
        "rh": [round(float(x), 5) for x in rh],
        "rh_mm": round(float(np.linalg.norm(rh)) * 1e3, 2),
        "vrel": [round(float(x), 4) for x in rel],
        "vrel_mps": round(float(np.linalg.norm(rel)), 4),
        "vcom": round(float(np.linalg.norm(vo)), 4),
        "w": [round(float(x), 3) for x in wo],
        "wmag": round(float(np.linalg.norm(wo)), 3),
        "ghx": round(float(o["g_h"][0]), 4),
        "obj_z": round(float(o["obj_z"]), 4),
        "ctrl": round(float(sim.data.ctrl[7]), 3),
        "aper": round(float(finger_opening(sim.data, sim.ids)), 5),
        "tilt": round(tilt_deg(sim), 2),
        "gh": [round(float(x), 4) for x in np.asarray(o["g_h"], float)],
        "ball": int(any(r["kind"] == "ball-cylinder" for r in ball_force_on_ball(sim)[1])),
    }


def closest_predicted(rh, vel, gravity) -> dict:
    rh = np.asarray(rh, float)
    vel = np.asarray(vel, float)
    gravity = np.asarray(gravity, float)
    times = np.linspace(0.0, 0.30, 61)
    dist = []
    for t in times:
        point = rh + vel * t + 0.5 * gravity * t * t
        dist.append(float(np.linalg.norm(point)))
    index = int(np.argmin(dist))
    return {"m": round(dist[index], 4), "t": round(float(times[index]), 3)}


def nominal_step(sim):
    sim.physics_step(None, in_recovery=False)
    sim.maybe_capture_reference()


def simulate(sim, gains, v_hit, z_off, phi, elev, nominal, watch=0.30, keep_trace=False):
    """Lift, launch a real ball, hold the nominal secure grip, and detect pad loss.

    Returns the scan summary and, when both pads stay off for 20 ms, the snapshot
    at that confirmation instant. The ball is not parked.
    """
    del gains, nominal
    import mujoco

    assert_noslip(sim)
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    dt = float(sim.model.opt.timestep)
    released = False
    t_release = None
    hand_z0 = None
    t_impact = None
    t_last = None
    pre_n = None
    pre_fn = None
    pre_mom = None
    post_mom = None
    w_axial = None
    w_tumble = None
    last_mom = None
    last_n = None
    last_fn = None
    grip_max = -1e9
    aper_pre = None
    zero_run = 0
    max_off = 0
    t_first = None
    release_pack = None
    branch_pack = None
    snap = None
    min_rh = 1e9
    t_re = None
    t_left = None
    t_right = None
    confirmed = False
    t_confirm = None
    trace = []
    launch = None
    missed_fall = False
    for _ in range(int(round(8.0 / dt))):
        nominal_step(sim)
        t = float(sim.data.time)
        o = physical_pack(sim)
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if sim.fsm.phase == "lift" and hand_z0 is None:
            hand_z0 = hz
        if (
            (not released)
            and sim.fsm.phase == "lift"
            and sim.captured
            and hand_z0 is not None
            and float(o["obj_z"]) >= Z_AIR
            and (hz - hand_z0) >= HAND_RISE_M
        ):
            launch = launch_aim(sim, v_hit, z_off, phi, elev)
            sim.set_guide_weld(False)
            apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
            released = True
            t_release = t
            aper_pre = float(finger_opening(sim.data, sim.ids))
        _fb, contacts = ball_force_on_ball(sim)
        hitting = any(r["kind"] == "ball-cylinder" for r in contacts)
        if released and hitting and t_impact is None:
            t_impact = t
            pre_n = last_n
            pre_fn = last_fn
            pre_mom = last_mom
        if hitting:
            t_last = t
        if t_impact is not None and post_mom is None and t_last is not None and (not hitting) and t > t_last:
            post_mom = momentum(sim)
            axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
            wo = post_mom[3]
            along = float(np.dot(wo, axis))
            w_axial = abs(along)
            w_tumble = float(np.linalg.norm(wo - axis * along))
        if released:
            grip_max = max(grip_max, float(sim.data.ctrl[7]))
        if t_impact is not None and int(o["nL"]) > 0:
            t_left = t
        if t_impact is not None and int(o["nR"]) > 0:
            t_right = t
        both_off = int(o["nL"]) == 0 and int(o["nR"]) == 0
        if t_impact is not None and both_off and snap is None:
            if zero_run == 0:
                t_first = t
                release_pack = pack_now(sim)
            zero_run += 1
            min_rh = min(min_rh, float(np.linalg.norm(o["rh"])))
            if zero_run == PERSIST_STEPS:
                snap = stamp(sim)
                branch_pack = pack_now(sim)
                confirmed = True
                t_confirm = t
        elif snap is None:
            if zero_run > 0:
                max_off = max(max_off, zero_run)
            zero_run = 0
            t_first = None
            release_pack = None
        elif t_re is None and (int(o["nL"]) + int(o["nR"])) > 0:
            t_re = t
        elif t_re is None:
            min_rh = min(min_rh, float(np.linalg.norm(o["rh"])))
        if keep_trace and t_impact is not None and t <= t_impact + 0.20:
            row = pack_now(sim)
            row["zero_run"] = int(zero_run)
            trace.append(row)
        last_mom = momentum(sim)
        last_n = (int(o["nL"]), int(o["nR"]))
        last_fn = (round(float(o["Fn_L"]), 3), round(float(o["Fn_R"]), 3))
        if confirmed and t >= t_confirm + watch:
            break
        if released and float(o["obj_z"]) < 0.42:
            missed_fall = True
            break
        if t_impact is not None and snap is None and t > t_impact + 0.45:
            break
        if t_release is not None and t_impact is None and t > t_release + 0.35:
            break
    if zero_run > 0:
        max_off = max(max_off, zero_run)
    tail = pack_now(sim) if t_impact is not None else None
    air = None
    if snap is not None and t_first is not None:
        end_t = float(sim.data.time) if t_re is None else t_re
        air = end_t - t_first
    dw = dv = reff = None
    if pre_mom is not None and post_mom is not None:
        d_lin = post_mom[0] - pre_mom[0]
        d_ang = post_mom[1] - pre_mom[1]
        dv = float(np.linalg.norm(post_mom[2] - pre_mom[2]))
        dw = float(np.linalg.norm(post_mom[3] - pre_mom[3]))
        reff = float(np.linalg.norm(d_ang) / max(float(np.linalg.norm(d_lin)), 1e-9))
    external = bool(
        snap is not None
        and pre_n is not None
        and pre_n[0] > 0
        and pre_n[1] > 0
        and grip_max <= SECURE_TAU
        and t_first is not None
        and t_impact is not None
        and t_first >= t_impact
    )
    pred = None
    if release_pack is not None:
        pred = closest_predicted(release_pack["rh"], release_pack["vrel"], release_pack["gh"])
    lost_first = None
    if t_left is not None or t_right is not None:
        if t_left is None:
            lost_first = "right_only"
        elif t_right is None:
            lost_first = "left_only"
        elif abs(t_left - t_right) <= dt:
            lost_first = "same_step"
        elif t_left < t_right:
            lost_first = "left"
        else:
            lost_first = "right"
    summary = {
        "z_mm": round(float(z_off) * 1e3, 2),
        "phi": float(phi),
        "speed": float(v_hit),
        "elev": float(elev),
        "miss_mm": None if launch is None else round(float(launch["miss_mm"]), 2),
        "hit": t_impact is not None,
        "t_impact": None if t_impact is None else round(float(t_impact), 4),
        "pre_n": None if pre_n is None else [int(pre_n[0]), int(pre_n[1])],
        "pre_fn": None if pre_fn is None else [float(pre_fn[0]), float(pre_fn[1])],
        "grip_max": None if grip_max < -1e8 else round(float(grip_max), 3),
        "aper_pre": None if aper_pre is None else round(float(aper_pre), 5),
        "external": external,
        "air_s": None if air is None else round(float(air), 4),
        "min_rh_mm": None if min_rh > 10 else round(float(min_rh) * 1e3, 2),
        "t_re": None if t_re is None else round(float(t_re), 4),
        "fell": bool(missed_fall),
        "dw": None if dw is None else round(float(dw), 3),
        "dv": None if dv is None else round(float(dv), 4),
        "reff_mm": None if reff is None else round(float(reff) * 1e3, 2),
        "w_axial": None if w_axial is None else round(float(w_axial), 3),
        "w_tumble": None if w_tumble is None else round(float(w_tumble), 3),
        "max_off_s": round(float(max_off) * dt, 4),
        "tail": None if tail is None else {
            "nL": tail["nL"], "nR": tail["nR"],
            "FnL": tail["FnL"], "FnR": tail["FnR"],
            "rhx_mm": round(tail["rh"][0] * 1e3, 2),
            "rhy_mm": round(tail["rh"][1] * 1e3, 2),
            "rhz_mm": round(tail["rh"][2] * 1e3, 2),
            "vrel": tail["vrel_mps"], "wmag": tail["wmag"],
            "obj_z": tail["obj_z"], "tilt": tail["tilt"],
            "ghx": tail["ghx"], "aper": tail["aper"], "ball": tail["ball"],
        },
        "release": release_pack,
        "branch": branch_pack,
        "lost_first": lost_first,
        "pred_mm": None if pred is None or release_pack is None else round(float(pred["m"]) * 1e3, 2),
        "pred_t": None if pred is None or release_pack is None else pred["t"],
        "ball_departed": bool(t_impact is not None and t_last is not None and (snap is None or t_confirm > t_last)),
    }
    return summary, snap, trace


def catch_stats(sim, t0, state):
    o = physical_pack(sim)
    t = float(sim.data.time) - t0
    both = int(o["nL"]) > 0 and int(o["nR"]) > 0
    any_pad = int(o["nL"]) + int(o["nR"]) > 0
    hitting = any(r["kind"] == "ball-cylinder" for r in ball_force_on_ball(sim)[1])
    if hitting:
        if state["saw_clear"]:
            state["rehit"] = True
            if state["t_rehit"] is None:
                state["t_rehit"] = round(t, 4)
    else:
        state["saw_clear"] = True
    if any_pad and state["t_any"] is None and t > 1e-4:
        state["t_any"] = round(t, 4)
    if both and state["t_both"] is None and t >= 0.02:
        state["t_both"] = round(t, 4)
    if _dropped(o) and state["t_drop"] is None:
        state["t_drop"] = round(t, 4)
    if _escaped(sim) and state["t_esc"] is None:
        state["t_esc"] = round(t, 4)
    state["min_rh"] = min(state["min_rh"], float(np.linalg.norm(o["rh"])))
    state["end_n"] = [int(o["nL"]), int(o["nR"])]
    state["end_z"] = round(float(o["obj_z"]), 4)
    state["gmin"] = min(state["gmin"], state.get("g_now", 1.0))
    return both


def fresh_state():
    return {
        "saw_clear": False,
        "rehit": False,
        "t_rehit": None,
        "t_any": None,
        "t_both": None,
        "t_drop": None,
        "t_esc": None,
        "min_rh": 1e9,
        "end_n": None,
        "end_z": None,
        "gmin": 1.0,
        "tokens": [],
    }


def finish_roll(sim, gains, nominal, pad_ids, pad_mu, state, kind):
    from training.phase_decoupled_recovery_pilot import oracle_now

    table_first = state["t_drop"] is not None and (state["t_both"] is None or state["t_drop"] < state["t_both"])
    rehit_first = bool(state["rehit"] and (state["t_both"] is None or (state["t_rehit"] or 0) <= (state["t_both"] or 9)))
    short_ok = bool(
        state["t_both"] is not None
        and not table_first
        and not rehit_first
        and state["end_n"] is not None
        and state["end_n"][0] > 0
        and state["end_n"][1] > 0
        and state["end_z"] is not None
        and state["end_z"] > 0.48
    )
    outcome = None
    if short_ok or kind in ("nominal", "token", "known"):
        lab = oracle_now(sim, gains, nominal, pad_ids, pad_mu, None, None)
        outcome = lab.get("outcome")
    else:
        if table_first:
            outcome = "TABLE_BEFORE_REGRASP"
        elif rehit_first:
            outcome = "BALL_REHIT"
        elif state["t_esc"] is not None and state["t_both"] is None:
            outcome = "ESCAPE"
        else:
            outcome = "NO_REGRASP"
    success = bool(
        outcome == "RETAINED_TO_12"
        and state["t_both"] is not None
        and not table_first
        and not rehit_first
    )
    return {
        "kind": kind,
        "outcome": outcome,
        "success": success,
        "t_any": state["t_any"],
        "t_both": state["t_both"],
        "t_drop": state["t_drop"],
        "t_esc": state["t_esc"],
        "rehit": bool(state["rehit"]),
        "table_first": bool(table_first),
        "min_rh_mm": None if state["min_rh"] > 10 else round(state["min_rh"] * 1e3, 2),
        "end_n": state["end_n"],
        "end_z": state["end_z"],
        "gmin": None if state["gmin"] > 0.99 else round(float(state["gmin"]), 3),
        "tokens": state["tokens"][:16],
    }


def rollout(sim, gains, v_hit, z_off, phi, elev, nominal, pad_ids, pad_mu, kind, actions, encoder, head, centers, horizon=0.45):
    """Re-simulate the same impact and apply `kind` only after external loss."""
    summary, snap, _trace = simulate(sim, gains, v_hit, z_off, phi, elev, nominal, watch=0.0)
    if not summary["external"] or snap is None:
        return summary, {"kind": kind, "outcome": "NO_EXTERNAL_LOSS", "success": False}
    from envs.observable_obs import ObservableObsState
    from training.recovery_policy_pretraining_preparation import legal_obs
    from training.single_step_token_behavior_audit import decode, predict
    from training.value_guided_recovery_bootstrap import DT_POLICY
    import torch

    state = fresh_state()
    t0 = float(sim.data.time)
    # The confirmation step is already contact-free. Mark the ball clear if it is gone.
    if pack_now(sim)["ball"] == 0:
        state["saw_clear"] = True
    if kind == "nominal":
        dt = float(sim.model.opt.timestep)
        n = int(round(horizon / dt))
        for _ in range(n):
            nominal_step(sim)
            catch_stats(sim, t0, state)
            if state["t_drop"] is not None or state["t_esc"] is not None:
                break
    elif kind == "script":
        for act in np.asarray(actions, np.float32):
            state["g_now"] = float(act[6])
            step_cmd(sim, "7", act)
            catch_stats(sim, t0, state)
            if state["t_drop"] is not None or state["t_esc"] is not None:
                break
            if float(sim.data.time) - t0 >= horizon and state["t_both"] is None:
                break
    elif kind == "token":
        hist = ObservableObsState(DT_POLICY)
        prev = np.zeros(7, np.float32)
        obs_l, prev_l = [], []
        centers_t = torch.tensor(centers)
        for _k in range(int(horizon / 0.02)):
            obs_l.append(np.asarray(legal_obs(sim, hist), np.float32))
            prev_l.append(prev.copy())
            h, prob = predict(encoder, head, centers_t, obs_l, prev_l)
            z = int(np.argmax(prob))
            act, _delta = decode(head, h, centers_t, z)
            state["g_now"] = float(act[6])
            state["tokens"].append(z)
            step_cmd(sim, "7", act)
            prev = act.copy()
            catch_stats(sim, t0, state)
            if state["t_drop"] is not None or state["t_esc"] is not None:
                break
    else:
        raise ValueError(kind)
    rolled = finish_roll(sim, gains, nominal, pad_ids, pad_mu, state, kind)
    return summary, rolled


def script_bank():
    known = np.asarray(schedules()["recatch_open20"], np.float32)
    bank = {
        "known": known,
        "pitch_closed": np.asarray([embed4([1, 0, 0, 1])] * 24, np.float32),
        "pitch_inward": np.asarray([embed4([1, -1, 0, 1])] * 12 + [embed4([0, 0, 0, 1])] * 12, np.float32),
        "inward": np.asarray([embed4([0, -1, 0, 1])] * 12 + [embed4([0, 0, 0, 1])] * 8, np.float32),
        "up_close": np.asarray([[0, 0, 1, 0, 0, 0, 1]] * 12 + [[0, 0, 0, 0, 0, 0, 1]] * 8, np.float32),
        "counter_pitch": np.asarray([embed4([-1, 0, 0, 1])] * 16, np.float32),
    }
    return bank


def tag_of(row) -> str:
    z = int(round(row["z_mm"]))
    v = int(round(row["speed"] * 10))
    e = int(round(row["elev"]))
    return f"z{z}_phi{int(row['phi'])}_v{v}_e{e}"


def interesting(row) -> bool:
    if not row.get("external"):
        return False
    rel = row.get("release") or {}
    if row.get("air_s") is None or row["air_s"] < 0.02:
        return False
    if rel.get("obj_z", 0) < 0.46:
        return False
    return True


def catchable(row) -> bool:
    if not interesting(row):
        return False
    rel = row["release"]
    return bool(rel["rh_mm"] <= 45.0 and rel["vrel_mps"] <= 0.60 and rel["wmag"] >= 2.0)


def film(sim, gains, v_hit, z_off, phi, elev, nominal, kind, actions, encoder, head, centers, path):
    import mujoco
    from training.real_impact_recatch_over_slide import grab, write_video

    assert_noslip(sim)
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    renderer = mujoco.Renderer(sim.model, 240, 320)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(sim.model, cam)
    cam.distance = 0.55
    cam.azimuth = 140
    cam.elevation = -16
    frames = []
    released = False
    t_release = None
    hand_z0 = None
    t_impact = None
    t_last = None
    zero_run = 0
    confirmed = False
    dt = float(sim.model.opt.timestep)
    step_i = 0
    t_confirm = None
    for _ in range(int(round(8.0 / dt))):
        nominal_step(sim)
        step_i += 1
        t = float(sim.data.time)
        o = physical_pack(sim)
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if sim.fsm.phase == "lift" and hand_z0 is None:
            hand_z0 = hz
        if (
            (not released)
            and sim.fsm.phase == "lift"
            and sim.captured
            and hand_z0 is not None
            and float(o["obj_z"]) >= Z_AIR
            and (hz - hand_z0) >= HAND_RISE_M
        ):
            launch = launch_aim(sim, v_hit, z_off, phi, elev)
            sim.set_guide_weld(False)
            apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
            released = True
            t_release = t
        hitting = any(r["kind"] == "ball-cylinder" for r in ball_force_on_ball(sim)[1])
        if released and hitting and t_impact is None:
            t_impact = t
        if hitting:
            t_last = t
        both_off = int(o["nL"]) == 0 and int(o["nR"]) == 0
        if t_impact is not None and both_off and not confirmed:
            zero_run += 1
            if zero_run == PERSIST_STEPS:
                confirmed = True
                t_confirm = t
        elif not confirmed:
            zero_run = 0
        if released and (step_i % 2 == 0 or hitting or zero_run > 0):
            frames.append(grab(renderer, sim, cam))
        if confirmed:
            break
        if released and float(o["obj_z"]) < 0.42:
            break
        if t_release is not None and t_impact is None and t > t_release + 0.35:
            break
        if t_impact is not None and t > t_impact + 0.45:
            break
    if not confirmed:
        if frames:
            write_video(path, frames)
        renderer.close()
        return False
    t0 = float(sim.data.time)
    if kind == "nominal":
        n = int(round(0.80 / dt))
        for i in range(n):
            nominal_step(sim)
            if i % 4 == 0:
                frames.append(grab(renderer, sim, cam))
            if _dropped(physical_pack(sim)) or _escaped(sim):
                break
    elif kind == "token":
        from envs.observable_obs import ObservableObsState
        from training.recovery_policy_pretraining_preparation import legal_obs
        from training.single_step_token_behavior_audit import decode, predict
        from training.value_guided_recovery_bootstrap import DT_POLICY
        import torch
        hist = ObservableObsState(DT_POLICY)
        prev = np.zeros(7, np.float32)
        obs_l, prev_l = [], []
        centers_t = torch.tensor(centers)
        for _k in range(30):
            obs_l.append(np.asarray(legal_obs(sim, hist), np.float32))
            prev_l.append(prev.copy())
            h, prob = predict(encoder, head, centers_t, obs_l, prev_l)
            act, _delta = decode(head, h, centers_t, int(np.argmax(prob)))
            step_cmd(sim, "7", act)
            prev = act.copy()
            frames.append(grab(renderer, sim, cam))
            if _dropped(physical_pack(sim)) or _escaped(sim):
                break
    else:
        for act in np.asarray(actions, np.float32):
            step_cmd(sim, "7", act)
            frames.append(grab(renderer, sim, cam))
            if _dropped(physical_pack(sim)) or _escaped(sim):
                break
            if float(sim.data.time) - t0 > 0.80:
                break
    # Ball stays in the scene through the collision and the airborne interval.
    # The long tail only subsamples; it does not rewind or hide the ball.
    t_drop = None
    n = 0
    while float(sim.data.time) < t_confirm + 1.6:
        nominal_step(sim)
        n += 1
        if n % 25 == 0:
            frames.append(grab(renderer, sim, cam))
        if _dropped(physical_pack(sim)):
            t_drop = float(sim.data.time)
            break
        if _escaped(sim):
            break
    if t_drop is not None:
        extra = 0
        while extra < int(0.15 / dt):
            nominal_step(sim)
            extra += 1
        frames.append(grab(renderer, sim, cam))
    write_video(path, frames)
    renderer.close()
    return True


def run_grid(sim, gains, nominal, jobs, rows, seen):
    for z_off, phi, speed, elev in jobs:
        key = (round(z_off, 4), round(phi, 1), round(speed, 2), round(elev, 1))
        if key in seen:
            continue
        seen.add(key)
        t0 = time.time()
        summary, snap, _trace = simulate(sim, gains, speed, z_off, phi, elev, nominal, watch=0.30)
        summary["tag"] = tag_of(summary)
        rows.append(summary)
        rel = summary.get("release") or {}
        print(
            "SCAN", summary["tag"],
            "hit", int(summary["hit"]),
            "ext", int(summary["external"]),
            "air", summary["air_s"],
            "rh", rel.get("rh_mm"),
            "v", rel.get("vrel_mps"),
            "w", rel.get("wmag"),
            "dv", summary["dv"],
            "dw", summary["dw"],
            "reff", summary["reff_mm"],
            "grip", summary["grip_max"],
            "z", rel.get("obj_z"),
            "tilt", rel.get("tilt"),
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
        del snap


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    from training.phase_decoupled_recovery_pilot import world
    from training.recatch_over_slide_counterexample import load_policy

    sim, gains, nominal, pad_ids, pad_mu, _pads_l, _pads_r = world()
    encoder, head, centers = load_policy()
    bank = script_bank()
    rows = []
    seen = set()
    jobs = [
        (z / 1000.0, phi, speed, 0.0)
        for z in (-28, -20, -12, 0, 12, 20, 28)
        for phi in (0.0, 35.0)
        for speed in (3.5, 5.0)
    ]
    run_grid(sim, gains, nominal, jobs, rows, seen)
    losses = [r for r in rows if r["external"] and (r.get("dw") or 0) >= 2.0]
    losses.sort(key=lambda r: (r.get("release") or {}).get("rh_mm", 999))
    extra = []
    for base in losses[:2]:
        z = base["z_mm"] / 1000.0
        phi = base["phi"]
        speed = base["speed"]
        for dz in (-0.004, 0.004):
            extra.append((z + dz, phi, speed, 0.0))
        for dv in (-0.5, 0.5):
            extra.append((z, phi, max(1.5, speed + dv), 0.0))
        for dphi in (-15.0, 15.0):
            extra.append((z, phi + dphi, speed, 0.0))
    if not losses:
        for z in (-25, 25):
            for speed in (2.5, 4.5, 6.0):
                extra.append((z / 1000.0, 0.0, speed, 0.0))
            for elev in (-25.0, 25.0):
                extra.append((z / 1000.0, 0.0, 4.0, elev))
    run_grid(sim, gains, nominal, extra, rows, seen)
    (OUT / "grid.json").write_text(json.dumps(jsonable(rows), indent=2), encoding="utf-8")

    near = [r for r in rows if catchable(r)]
    if not near:
        near = [r for r in rows if interesting(r)]
        near.sort(key=lambda r: (r.get("release") or {}).get("rh_mm", 999))
        near = near[:4]
    else:
        near.sort(key=lambda r: ((r.get("release") or {}).get("rh_mm", 999), -(r.get("release") or {}).get("wmag", 0)))
        near = near[:4]
    print("CANDIDATES", [r["tag"] for r in near], flush=True)

    compared = []
    successes = []
    for row in near:
        z = row["z_mm"] / 1000.0
        bundle = {
            "tag": row["tag"],
            "impact": {k: row[k] for k in ("z_mm", "phi", "speed", "elev", "miss_mm")},
            "scan_air_s": row["air_s"],
            "scan_release": row["release"],
            "scan_dw": row["dw"],
            "scan_dv": row["dv"],
            "scan_reff_mm": row["reff_mm"],
            "scan_grip_max": row["grip_max"],
            "scan_min_rh_mm": row["min_rh_mm"],
            "lost_first": row["lost_first"],
            "pred_mm": row["pred_mm"],
            "rolls": [],
        }
        for kind, actions in (
            ("nominal", None),
            ("token", None),
            ("known", bank["known"]),
        ):
            summary, rolled = rollout(
                sim, gains, row["speed"], z, row["phi"], row["elev"], nominal, pad_ids, pad_mu,
                "script" if kind == "known" else kind,
                actions, encoder, head, centers, horizon=0.80 if kind == "nominal" else 0.60,
            )
            rolled["kind"] = kind
            bundle["rolls"].append(rolled)
            bundle["live"] = {
                "external": summary["external"],
                "air_s": summary["air_s"],
                "release": summary["release"],
                "dw": summary["dw"],
                "dv": summary["dv"],
                "reff_mm": summary["reff_mm"],
                "grip_max": summary["grip_max"],
            }
            print("ROLL", row["tag"], kind, rolled["outcome"], "both", rolled.get("t_both"), "g", rolled.get("gmin"), flush=True)
        for name, actions in bank.items():
            if name == "known":
                continue
            summary, rolled = rollout(
                sim, gains, row["speed"], z, row["phi"], row["elev"], nominal, pad_ids, pad_mu,
                "script", actions, encoder, head, centers, horizon=0.50,
            )
            rolled["kind"] = name
            bundle["rolls"].append(rolled)
            print("ROLL", row["tag"], name, rolled["outcome"], "both", rolled.get("t_both"), flush=True)
            if rolled.get("success"):
                successes.append({"tag": row["tag"], "kind": name, "impact": bundle["impact"]})
        for rolled in bundle["rolls"]:
            if rolled.get("success") and rolled["kind"] in ("nominal", "token", "known"):
                successes.append({"tag": row["tag"], "kind": rolled["kind"], "impact": bundle["impact"]})
        compared.append(bundle)
    (OUT / "compared.json").write_text(json.dumps(jsonable(compared), indent=2), encoding="utf-8")

    robust = []
    if successes:
        # Require the same controller on nearby real impacts, not a new search.
        seed = successes[0]
        kind = seed["kind"]
        actions = None if kind in ("nominal", "token") else bank[kind]
        mode = "script" if actions is not None else kind
        base = next(r for r in rows if r["tag"] == seed["tag"])
        z0 = base["z_mm"] / 1000.0
        neighbors = [
            (z0 + 0.002, base["phi"], base["speed"], base["elev"], "z+2"),
            (z0 - 0.002, base["phi"], base["speed"], base["elev"], "z-2"),
            (z0, base["phi"], base["speed"] + 0.3, base["elev"], "v+0.3"),
            (z0, base["phi"], max(1.0, base["speed"] - 0.3), base["elev"], "v-0.3"),
            (z0, base["phi"] + 10.0, base["speed"], base["elev"], "phi+10"),
            (z0, base["phi"] - 10.0, base["speed"], base["elev"], "phi-10"),
        ]
        for z, phi, speed, elev, label in neighbors:
            summary, rolled = rollout(
                sim, gains, speed, z, phi, elev, nominal, pad_ids, pad_mu,
                mode, actions, encoder, head, centers, horizon=0.60,
            )
            item = {
                "label": label,
                "external": summary["external"],
                "air_s": summary["air_s"],
                "release": summary["release"],
                "dw": summary["dw"],
                "dv": summary["dv"],
                "reff_mm": summary["reff_mm"],
                "grip_max": summary["grip_max"],
                "roll": rolled,
            }
            robust.append(item)
            print("ROBUST", label, "ext", int(summary["external"]), rolled["outcome"], flush=True)
        (OUT / "robust.json").write_text(json.dumps(jsonable(robust), indent=2), encoding="utf-8")

    # Videos and high-res traces for the boundary cases. Ball stays visible.
    film_rows = list(near[:2])
    centered = next((r for r in rows if r["z_mm"] == 0 and r["phi"] == 0 and r["speed"] == 5.0), None)
    if centered is not None:
        film_rows.append(centered)
    filmed = []
    for row in film_rows:
        z = row["z_mm"] / 1000.0
        summary, snap, trace = simulate(
            sim, gains, row["speed"], z, row["phi"], row["elev"], nominal, watch=0.20, keep_trace=True,
        )
        if snap is not None:
            import torch
            torch.save(snap, OUT / f"{row['tag']}.pt")
        (OUT / f"{row['tag']}_trace.json").write_text(json.dumps(jsonable(trace), indent=2), encoding="utf-8")
        kinds = [("nominal", None), ("token", None), ("known", bank["known"])]
        if successes and successes[0]["tag"] == row["tag"] and successes[0]["kind"] in bank:
            kinds.append((successes[0]["kind"], bank[successes[0]["kind"]]))
        if row is centered:
            kinds = [("nominal", None)]
        for kind, actions in kinds:
            path = VID / f"{row['tag']}_{kind}.mp4"
            ok = film(
                sim, gains, row["speed"], z, row["phi"], row["elev"], nominal,
                "script" if actions is not None else kind,
                actions, encoder, head, centers, path,
            )
            filmed.append({"tag": row["tag"], "kind": kind, "ok": bool(ok), "bytes": path.stat().st_size if path.exists() else 0})
            print("VIDEO", row["tag"], kind, ok, filmed[-1]["bytes"], flush=True)
    (OUT / "videos.json").write_text(json.dumps(filmed, indent=2), encoding="utf-8")
    print("SUCCESS", [(s["tag"], s["kind"]) for s in successes], flush=True)
    print("DONE", flush=True)
    return 0


def ladder_main() -> int:
    """Harder face-on and elevated impacts. Stops once the release boundary is visible."""
    from training.phase_decoupled_recovery_pilot import world
    from training.recatch_over_slide_counterexample import load_policy
    import torch

    sim, gains, nominal, pad_ids, pad_mu, _l, _r = world()
    encoder, head, centers = load_policy()
    bank = script_bank()
    jobs = []
    for z in (-28, -20, 28):
        for speed in (6.0, 8.0, 10.0):
            jobs.append((z / 1000.0, 0.0, speed, 0.0))
    for speed in (5.0, 7.0, 9.0):
        jobs.append((-0.025, 0.0, speed, -25.0))
    for speed in (6.0, 8.0):
        jobs.append((-0.028, 35.0, speed, 0.0))
    for speed in (7.0, 8.0, 10.0):
        jobs.append((0.0, 0.0, speed, 0.0))
    rows = []
    snaps = {}
    for z, phi, speed, elev in jobs:
        summary, snap, trace = simulate(
            sim, gains, speed, z, phi, elev, nominal, watch=0.30, keep_trace=True,
        )
        summary["tag"] = tag_of(summary)
        rows.append(summary)
        tail = summary.get("tail") or {}
        rel = summary.get("release") or {}
        print(
            "LAD", summary["tag"],
            "ext", int(summary["external"]),
            "off", summary["max_off_s"],
            "air", summary["air_s"],
            "tum", summary["w_tumble"],
            "ax", summary["w_axial"],
            "dw", summary["dw"],
            "dv", summary["dv"],
            "reff", summary["reff_mm"],
            "rhx", None if not rel else round(rel["rh"][0] * 1e3, 2),
            "tail_n", tail.get("nL"), tail.get("nR"),
            "tail_rhx", tail.get("rhx_mm"),
            "tail_w", tail.get("wmag"),
            "tail_z", tail.get("obj_z"),
            "grip", summary["grip_max"],
            flush=True,
        )
        if summary["external"] and snap is not None:
            snaps[summary["tag"]] = snap
            (OUT / f"{summary['tag']}_trace.json").write_text(json.dumps(jsonable(trace), indent=2), encoding="utf-8")
            torch.save(snap, OUT / f"{summary['tag']}.pt")
    (OUT / "ladder.json").write_text(json.dumps(jsonable(rows), indent=2), encoding="utf-8")
    lost = [r for r in rows if r["external"]]
    print("LOST", [r["tag"] for r in lost], flush=True)
    compared = []
    successes = []
    for row in lost:
        rel = row["release"]
        near = rel["rh"][0] * 1e3 <= 40.0 and rel["obj_z"] >= 0.46 and rel["vrel_mps"] <= 0.8
        if not near and len(lost) > 3:
            continue
        bundle = {"tag": row["tag"], "near": bool(near), "rolls": []}
        z = row["z_mm"] / 1000.0
        for kind, actions in (("nominal", None), ("token", None), ("known", bank["known"])):
            _summary, rolled = rollout(
                sim, gains, row["speed"], z, row["phi"], row["elev"], nominal, pad_ids, pad_mu,
                "script" if kind == "known" else kind, actions, encoder, head, centers,
                horizon=0.70,
            )
            rolled["kind"] = kind
            bundle["rolls"].append(rolled)
            print("ROLL", row["tag"], kind, rolled["outcome"], "both", rolled.get("t_both"), flush=True)
            if rolled.get("success"):
                successes.append((row["tag"], kind))
        if near:
            for name, actions in bank.items():
                if name == "known":
                    continue
                _summary, rolled = rollout(
                    sim, gains, row["speed"], z, row["phi"], row["elev"], nominal, pad_ids, pad_mu,
                    "script", actions, encoder, head, centers, horizon=0.50,
                )
                rolled["kind"] = name
                bundle["rolls"].append(rolled)
                print("ROLL", row["tag"], name, rolled["outcome"], "both", rolled.get("t_both"), flush=True)
                if rolled.get("success"):
                    successes.append((row["tag"], name))
        compared.append(bundle)
    (OUT / "ladder_compared.json").write_text(json.dumps(jsonable(compared), indent=2), encoding="utf-8")
    # Visuals: one held tumble, one centered hit, and any external loss.
    film_jobs = []
    held = [r for r in rows if r["hit"] and not r["external"] and r.get("w_tumble")]
    held.sort(key=lambda r: -float(r["w_tumble"]))
    if held:
        film_jobs.append(held[0])
    centered = next((r for r in rows if r["z_mm"] == 0 and r["speed"] == 8.0), None)
    if centered is not None:
        film_jobs.append(centered)
    for row in lost[:2]:
        film_jobs.append(row)
    filmed = []
    for row in film_jobs:
        z = row["z_mm"] / 1000.0
        kinds = [("nominal", None)]
        if row["external"]:
            kinds = [("nominal", None), ("token", None), ("known", bank["known"])]
        for kind, actions in kinds:
            path = VID / f"{row['tag']}_{kind}.mp4"
            ok = film(
                sim, gains, row["speed"], z, row["phi"], row["elev"], nominal,
                "script" if actions is not None else kind, actions, encoder, head, centers, path,
            )
            filmed.append({"tag": row["tag"], "kind": kind, "ok": bool(ok), "bytes": path.stat().st_size if path.exists() else 0})
            print("VIDEO", row["tag"], kind, ok, filmed[-1]["bytes"], flush=True)
    (OUT / "ladder_videos.json").write_text(json.dumps(filmed, indent=2), encoding="utf-8")
    print("SUCCESS", successes, flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ladder":
        raise SystemExit(ladder_main())
    raise SystemExit(main())
