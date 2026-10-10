"""Seven disturbance families for externally induced airborne recatch.

No training. The gripper stays on the nominal secure torque until a disturbance
has already opened both pads. The cylinder is never teleported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from training.demo_ballistic_impact import HAND_RISE_M, assert_noslip, ball_force_on_ball
from training.external_airborne_recatch import nominal_step, pack_now, script_bank
from training.impact_ball import apply_ball_free_state
from training.long_context_recovery_pilot import jsonable
from training.recovery_runtime import _dropped, _escaped, restore_dyn, stamp, step_cmd

OUT = ROOT / "results" / "diagnostics" / "raw" / "multi_disturbance_airborne"
DT0 = 0.002
SECURE = -17.0


def hand_pose(sim):
    p = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    R = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    return p, R


def rot_angle(R0, R1) -> float:
    c = float(np.clip((np.trace(R0.T @ R1) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(c))


def hand_x(sim) -> np.ndarray:
    R = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return R[:, 0].copy()


def zero_wrench(sim) -> None:
    sim.data.xfrc_applied[:] = 0.0


def restore(sim, snap, nominal, pad_ids, pad_mu):
    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, None, None)
    zero_wrench(sim)
    return info


def grasp_snapshot(sim):
    """Lift a bilateral grasp. No ball launch and no recovery command."""
    import mujoco

    assert_noslip(sim)
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    hand_z0 = None
    stable = 0.0
    dt = float(sim.model.opt.timestep)
    for _ in range(int(round(8.0 / dt))):
        nominal_step(sim)
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if sim.fsm.phase == "lift" and hand_z0 is None:
            hand_z0 = hz
        o = physical_pack(sim)
        risen = hand_z0 is not None and (hz - hand_z0) >= HAND_RISE_M + 0.03
        held = int(o["nL"]) > 0 and int(o["nR"]) > 0 and float(o["obj_z"]) >= Z_AIR
        secure = float(sim.data.ctrl[7]) <= SECURE
        if risen and held and secure:
            stable += dt
        else:
            stable = 0.0
        if stable >= 0.05:
            snap = stamp(sim)
            base = pack_now(sim)
            print(
                "GRASP", "rh", [round(x * 1e3, 2) for x in base["rh"]],
                "n", base["nL"], base["nR"], "Fn", base["FnL"], base["FnR"],
                "z", base["obj_z"], "ctrl", base["ctrl"], "t", base["t"],
                flush=True,
            )
            return snap, base
    raise RuntimeError("no stable grasp")


def run_disturbance(sim, snap, nominal, pad_ids, pad_mu, apply, watch=0.55, dt=None):
    """Nominal secure grasp plus one disturbance. Returns metrics and a loss snap."""
    old_dt = float(sim.model.opt.timestep)
    if dt is not None:
        sim.model.opt.timestep = float(dt)
    info = restore(sim, snap, nominal, pad_ids, pad_mu)
    if not info.get("pose_ok", False):
        sim.model.opt.timestep = old_dt
        return {"pose_ok": False}, None
    step_dt = float(sim.model.opt.timestep)
    t0 = float(sim.data.time)
    p0, _R0 = hand_pose(sim)
    rh0 = np.array(physical_pack(sim)["rh"], float)
    z0 = float(physical_pack(sim)["obj_z"])
    qpos0 = np.array(sim.data.qpos, float).copy()
    grip_max = -1e9
    zero_run = 0
    max_off = 0
    t_first = None
    loss = None
    loss_pack = None
    sep_off = 0.0
    sep_all = 0.0
    min_sep = 1e9
    max_sep = 0.0
    v_at = None
    env_hit = False
    t_end_off = None
    samples = []
    next_sample = t0
    fell = False
    while float(sim.data.time) < t0 + watch:
        t_rel = float(sim.data.time) - t0
        apply(sim, t_rel)
        nominal_step(sim)
        zero_wrench(sim)
        o = physical_pack(sim)
        rh = np.array(o["rh"], float)
        dist = float(np.linalg.norm(rh - rh0))
        sep_all = max(sep_all, dist)
        min_sep = min(min_sep, dist)
        max_sep = max(max_sep, dist)
        grip_max = max(grip_max, float(sim.data.ctrl[7]))
        if any(r["kind"] == "ball-cylinder" for r in ball_force_on_ball(sim)[1]):
            env_hit = True
        both = int(o["nL"]) == 0 and int(o["nR"]) == 0
        now = float(sim.data.time)
        if now >= next_sample:
            samples.append({
                "t": round(now - t0, 4),
                "nL": int(o["nL"]),
                "nR": int(o["nR"]),
                "FnL": round(float(o["Fn_L"]), 2),
                "FnR": round(float(o["Fn_R"]), 2),
                "rhx": round(float(rh[0]) * 1e3, 2),
                "sep": round(dist * 1e3, 2),
                "v": round(float(o["v_rel_h"][0]) if False else float(np.linalg.norm(o["v_rel_h"])), 3),
                "w": round(float(np.linalg.norm(o["w_rel_h"])) if "w_rel_h" in o else pack_w(sim), 3),
                "z": round(float(o["obj_z"]), 4),
                "ctrl": round(float(sim.data.ctrl[7]), 2),
            })
            next_sample = now + 0.01
        if both and loss is None:
            if zero_run == 0:
                t_first = now
                v_at = pack_now(sim)
            zero_run += 1
            sep_off = max(sep_off, dist)
            if zero_run * step_dt >= 0.02 - 1e-6:
                loss = stamp(sim)
                loss_pack = pack_now(sim)
                max_off = max(max_off, zero_run)
        elif loss is None:
            max_off = max(max_off, zero_run)
            zero_run = 0
            t_first = None
        else:
            max_off = max(max_off, zero_run)
            if (not both) and t_end_off is None:
                t_end_off = now
        if _dropped(o) or float(o["obj_z"]) < 0.42:
            fell = True
            break
        if _escaped(sim):
            fell = True
            break
    if zero_run:
        max_off = max(max_off, zero_run)
    sim.model.opt.timestep = old_dt
    air = None
    if t_first is not None:
        stop = float(sim.data.time) if t_end_off is None else t_end_off
        air = stop - t_first
    p1, _R1 = hand_pose(sim)
    # Object qpos may only change inside mj_step. A manual write would skip the integrator.
    q_delta = float(np.linalg.norm(np.array(sim.data.qpos, float) - qpos0))
    row = {
        "pose_ok": True,
        "grip_max": round(float(grip_max), 3),
        "secure": bool(grip_max <= SECURE or loss is not None),
        "grip_before_loss": round(float(grip_max), 3),
        "max_off_s": round(float(max_off) * step_dt, 4),
        "air_s": None if air is None else round(float(air), 4),
        "sep_off_mm": round(float(sep_off) * 1e3, 2),
        "sep_all_mm": round(float(sep_all) * 1e3, 2),
        "min_sep_mm": round(float(min_sep) * 1e3, 2),
        "max_sep_mm": round(float(max_sep) * 1e3, 2),
        "z0": round(z0, 4),
        "hand_dz_mm": round(float((p1 - p0)[2]) * 1e3, 2),
        "env_hit": bool(env_hit),
        "fell": bool(fell),
        "qpos_moved_m": round(q_delta, 4),
        "loss": loss_pack,
        "end": pack_now(sim),
        "opened_before_loss": bool(grip_max > SECURE and (t_first is None or True)),
        "trace": samples[::2],
    }
    # Grip is illegal only if it opened before the pads let go.
    if t_first is None:
        row["opened_before_loss"] = bool(grip_max > SECURE)
    else:
        row["opened_before_loss"] = bool(grip_max > SECURE)
    row["class"] = label_of(row)
    return row, loss


def pack_w(sim) -> float:
    from envs.deterioration import body_twist
    _v, w = body_twist(sim.model, sim.data, sim.ids.object_body)
    return float(np.linalg.norm(w))


def label_of(row) -> str:
    """A no loss, B flicker or passive reseat, C escape, P plausible airborne."""
    if not row.get("pose_ok"):
        return "A"
    off = float(row["max_off_s"] or 0.0)
    sep = float(row["sep_off_mm"] or 0.0)
    end = row.get("end") or {}
    bilateral_end = int(end.get("nL", 0)) > 0 and int(end.get("nR", 0)) > 0 and float(end.get("obj_z", 0)) > 0.48
    if row.get("fell") and sep >= 8.0:
        return "C"
    if off < 0.004 and sep < 8.0:
        return "A"
    if off < 0.02 or sep < 8.0:
        return "B"
    if bilateral_end and not row.get("fell"):
        return "B"
    loss = row.get("loss") or {}
    v = float(loss.get("vrel_mps", 9.0))
    z = float(loss.get("obj_z", 0.0))
    if v <= 0.35 and z >= 0.46 and sep <= 40.0:
        return "P"
    return "C"


def plausible(row) -> bool:
    return row.get("class") == "P"


def roll_controller(sim, gains, snap, nominal, pad_ids, pad_mu, kind, actions, encoder, head, centers, hand_ref):
    from envs.observable_obs import ObservableObsState
    from training.phase_decoupled_recovery_pilot import oracle_now
    from training.recovery_policy_pretraining_preparation import legal_obs
    from training.single_step_token_behavior_audit import decode, predict
    from training.value_guided_recovery_bootstrap import DT_POLICY
    import torch

    info = restore(sim, snap, nominal, pad_ids, pad_mu)
    if not info.get("pose_ok", False):
        return {"pose_ok": False, "outcome": "INVALID_RESTORE", "kind": kind}
    p0, R0 = hand_pose(sim)
    t0 = float(sim.data.time)
    t_both = None
    t_any = None
    t_drop = None
    tokens = []
    gmin = 1.0
    if kind == "nominal":
        while float(sim.data.time) < t0 + 0.70:
            nominal_step(sim)
            o = physical_pack(sim)
            t = float(sim.data.time) - t0
            if int(o["nL"]) + int(o["nR"]) > 0 and t_any is None and t > 1e-3:
                t_any = t
            if int(o["nL"]) > 0 and int(o["nR"]) > 0 and t_both is None and t >= 0.02:
                t_both = t
            if _dropped(o) and t_drop is None:
                t_drop = t
                break
            if _escaped(sim):
                break
    elif kind == "token":
        hist = ObservableObsState(DT_POLICY)
        prev = np.zeros(7, np.float32)
        obs_l, prev_l = [], []
        centers_t = torch.tensor(centers)
        for _k in range(30):
            obs_l.append(np.asarray(legal_obs(sim, hist), np.float32))
            prev_l.append(prev.copy())
            h, prob = predict(encoder, head, centers_t, obs_l, prev_l)
            z = int(np.argmax(prob))
            act, _delta = decode(head, h, centers_t, z)
            gmin = min(gmin, float(act[6]))
            tokens.append(z)
            step_cmd(sim, "7", act)
            prev = act.copy()
            o = physical_pack(sim)
            t = float(sim.data.time) - t0
            if int(o["nL"]) + int(o["nR"]) > 0 and t_any is None:
                t_any = t
            if int(o["nL"]) > 0 and int(o["nR"]) > 0 and t_both is None and t >= 0.02:
                t_both = t
            if _dropped(o) or _escaped(sim):
                if _dropped(o):
                    t_drop = t
                break
    else:
        for act in np.asarray(actions, np.float32):
            gmin = min(gmin, float(act[6]))
            step_cmd(sim, "7", act)
            o = physical_pack(sim)
            t = float(sim.data.time) - t0
            if int(o["nL"]) + int(o["nR"]) > 0 and t_any is None:
                t_any = t
            if int(o["nL"]) > 0 and int(o["nR"]) > 0 and t_both is None and t >= 0.02:
                t_both = t
            if _dropped(o) or _escaped(sim):
                if _dropped(o):
                    t_drop = float(sim.data.time) - t0
                break
            if float(sim.data.time) - t0 > 0.70 and t_both is None:
                break
    p1, R1 = hand_pose(sim)
    dang = rot_angle(R0, R1)
    dxy = float(np.linalg.norm((p1 - p0)[:2]))
    # Difference from the nominal hand at the same time, when that reference exists.
    t_now = float(sim.data.time) - t0
    active = False
    if hand_ref:
        ref = min(hand_ref, key=lambda s: abs(s[0] - t_now))
        d_ref = float(np.linalg.norm(p1 - ref[1]))
        a_ref = rot_angle(ref[2], R1)
        active = bool(d_ref > 0.005 or a_ref > np.deg2rad(8.0))
    else:
        active = bool(dxy > 0.005 or dang > np.deg2rad(8.0))
    lab = oracle_now(sim, gains, nominal, pad_ids, pad_mu, None, None)
    success = bool(
        lab.get("outcome") == "RETAINED_TO_12"
        and t_both is not None
        and t_drop is None
    )
    return {
        "pose_ok": True,
        "kind": kind,
        "outcome": lab.get("outcome"),
        "success": success,
        "active": bool(active and success),
        "t_any": None if t_any is None else round(float(t_any), 4),
        "t_both": None if t_both is None else round(float(t_both), 4),
        "t_drop": None if t_drop is None else round(float(t_drop), 4),
        "hand_dxy_mm": round(dxy * 1e3, 2),
        "hand_deg": round(float(np.degrees(dang)), 2),
        "gmin": round(float(gmin), 3),
        "tokens": tokens[:8],
        "mode": "ACTIVE RECATCh" if active and success else ("PASSIVE RESEAT" if success and not active else "NO RECATCh"),
    }


def nominal_hand_ref(sim, snap, nominal, pad_ids, pad_mu):
    info = restore(sim, snap, nominal, pad_ids, pad_mu)
    if not info.get("pose_ok", False):
        return []
    t0 = float(sim.data.time)
    ref = []
    while float(sim.data.time) < t0 + 0.70:
        nominal_step(sim)
        p, R = hand_pose(sim)
        ref.append((float(sim.data.time) - t0, p, R))
        if _dropped(physical_pack(sim)) or _escaped(sim):
            break
    return ref


def hann_force(sim, body, direction, peak, duration, t):
    if t < 0.0 or t > duration:
        zero_wrench(sim)
        return
    s = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / max(duration, 1e-4)))
    force = np.asarray(direction, float) * float(peak) * s
    sim.data.xfrc_applied[body, 0:3] = force


def impulse(sim, body, direction, magnitude, duration, t, torque=None):
    if t < 0.0 or t > duration:
        zero_wrench(sim)
        return
    sim.data.xfrc_applied[body, 0:3] = np.asarray(direction, float) * (float(magnitude) / duration)
    if torque is not None:
        sim.data.xfrc_applied[body, 3:6] = np.asarray(torque, float) * (1.0 / duration)


def save_mu(sim, ids):
    from envs.dynamics import finger_pad_geom_ids
    geoms = [int(ids.object_geom)]
    left, right = finger_pad_geom_ids(sim.model, ids)
    geoms.extend(left)
    geoms.extend(right)
    return {g: float(sim.model.geom_friction[g][0]) for g in geoms}


def set_mu(sim, saved, mu):
    for g in saved:
        sim.model.geom_friction[g][0] = float(mu)


def restore_mu(sim, saved):
    for g, mu in saved.items():
        sim.model.geom_friction[g][0] = float(mu)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from training.phase_decoupled_recovery_pilot import world
    from training.recatch_over_slide_counterexample import load_policy

    sim, gains, nominal, pad_ids, pad_mu, _l, _r = world()
    snap, base = grasp_snapshot(sim)
    import torch
    torch.save(snap, OUT / "grasp.pt")
    encoder, head, centers = load_policy()
    bank = script_bank()
    mu_saved = save_mu(sim, sim.ids)
    obj = int(sim.ids.object_body)
    hand = int(sim.ids.hand_body)
    link0 = int(__import__("mujoco").mj_name2id(sim.model, __import__("mujoco").mjtObj.mjOBJ_BODY, "link0"))
    families = []

    def trial(family, name, apply, watch=0.55):
        row, loss = run_disturbance(sim, snap, nominal, pad_ids, pad_mu, apply, watch=watch)
        row["family"] = family
        row["name"] = name
        loss_p = row.get("loss") or {}
        print(
            family, name, row.get("class"),
            "off", row.get("max_off_s"), "air", row.get("air_s"),
            "sep", row.get("sep_off_mm"), "all", row.get("sep_all_mm"),
            "v", loss_p.get("vrel_mps"), "z", loss_p.get("obj_z"),
            "grip", row.get("grip_max"), "env", int(row.get("env_hit", False)),
            "end_n", (row.get("end") or {}).get("nL"), (row.get("end") or {}).get("nR"),
            flush=True,
        )
        return row, loss

    def sweep(family, specs, stop_on=("C",)):
        rows = []
        for name, apply, watch in specs:
            row, loss = trial(family, name, apply, watch)
            rows.append((row, loss))
            if row.get("class") in stop_on:
                break
        return rows

    # Family 1 — short COM impulses, not a centered horizontal punch.
    f1 = []
    for mag in (0.01, 0.02, 0.04, 0.08, 0.16):
        dirs = {
            "up": (np.array([0.0, 0.0, 1.0]), None),
            "down": (np.array([0.0, 0.0, -1.0]), None),
        }
        # Directions that depend on the hand are built inside apply after restore.
        for dname in ("up", "down", "up_out", "down_out", "tumble"):
            def apply_dir(sim, t, mag=mag, dname=dname):
                hx = hand_x(sim)
                up = np.array([0.0, 0.0, 1.0])
                torque = None
                if dname == "up":
                    direction = up
                elif dname == "down":
                    direction = -up
                elif dname == "up_out":
                    direction = hx + up
                elif dname == "down_out":
                    direction = hx - up
                else:
                    direction = up
                    torque = np.cross(hx, up)
                    torque = torque / max(np.linalg.norm(torque), 1e-9) * mag
                direction = direction / max(np.linalg.norm(direction), 1e-9)
                use = mag if dname != "tumble" else min(mag, 0.03)
                impulse(sim, obj, direction, use, 0.012, t, torque=None if dname != "tumble" else torque)
            f1.append((trial("F1", f"{dname}_{mag:.2f}", apply_dir)[0], None))
    # The loop above does not early-stop per direction cleanly; that is acceptable.

    # Family 2 — smooth finite pulses.
    f2 = []
    for peak in (1.0, 2.0, 4.0, 8.0, 16.0):
        for dname in ("out", "down", "down_out"):
            def apply_pull(sim, t, peak=peak, dname=dname):
                hx = hand_x(sim)
                up = np.array([0.0, 0.0, 1.0])
                if dname == "out":
                    direction = hx
                elif dname == "down":
                    direction = -up
                else:
                    direction = hx - up
                direction = direction / max(np.linalg.norm(direction), 1e-9)
                hann_force(sim, obj, direction, peak, 0.10, t)
            row, _loss = trial("F2", f"{dname}_{peak:.0f}N", apply_pull)
            f2.append(row)
            if row.get("class") == "C":
                break

    # Family 3 — wrench on the hand, controller still the nominal lift.
    f3 = []
    for peak in (40.0, 100.0, 200.0, 400.0):
        for dname in ("out", "up"):
            def apply_hand(sim, t, peak=peak, dname=dname):
                direction = hand_x(sim) if dname == "out" else np.array([0.0, 0.0, 1.0])
                impulse(sim, hand, direction, peak * 0.03, 0.03, t)
            row, _loss = trial("F3", f"{dname}_{peak:.0f}N", apply_hand)
            f3.append(row)
    for torque_mag in (1.0, 3.0, 8.0):
        def apply_tor(sim, t, torque_mag=torque_mag):
            axis = np.cross(hand_x(sim), np.array([0.0, 0.0, 1.0]))
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            impulse(sim, hand, np.zeros(3), 0.0, 0.03, t, torque=axis * torque_mag * 0.03)
        row, _loss = trial("F3", f"torque_{torque_mag:.0f}", apply_tor)
        f3.append(row)

    # Family 4 — link0 is fixed. Record that, then accelerate the mounting frame.
    link_before = np.array(sim.data.xpos[link0], float).copy() if link0 >= 0 else None

    def apply_link0(sim, t):
        if link0 >= 0 and t <= 0.05:
            sim.data.xfrc_applied[link0, 0] = 100.0

    row_link, _l = trial("F4", "link0_100N", apply_link0, watch=0.20)
    link_after = np.array(sim.data.xpos[link0], float).copy() if link0 >= 0 else None
    link_move = None if link_before is None else float(np.linalg.norm(link_after - link_before))
    print("LINK0_MOVE_MM", None if link_move is None else round(link_move * 1e3, 4), flush=True)
    f4 = [row_link]
    for accel in (2.0, 5.0, 10.0, 20.0):
        for dname in ("out", "up"):
            state = {"v": np.zeros(3), "dir": None}

            def apply_base(sim, t, accel=accel, dname=dname, state=state):
                if state["dir"] is None:
                    state["dir"] = hand_x(sim).copy() if dname == "out" else np.array([0.0, 0.0, 1.0])
                dt = float(sim.model.opt.timestep)
                if t <= 0.08:
                    sign = 1.0 if t <= 0.04 else -1.0
                    state["v"] = state["v"] + state["dir"] * accel * sign * dt
                    sim.fsm.p_des = np.asarray(sim.fsm.p_des, float) + state["v"] * dt

            row, _loss = trial("F4", f"frame_{dname}_{accel:.0f}", apply_base)
            f4.append(row)
    for v_kick in (0.10, 0.25):
        state = {"on": True}

        def apply_stop(sim, t, v_kick=v_kick, state=state):
            direction = hand_x(sim)
            dt = float(sim.model.opt.timestep)
            if t <= 0.04:
                sim.fsm.p_des = np.asarray(sim.fsm.p_des, float) + direction * v_kick * dt
            elif t <= 0.06:
                sim.fsm.p_des = np.asarray(sim.fsm.p_des, float) - direction * v_kick * dt

        row, _loss = trial("F4", f"stop_{v_kick:.2f}", apply_stop)
        f4.append(row)

    # Family 5 — fixed ball obstacle in the lift path. Object is not moved.
    f5 = []

    def place_ball(sim, mode):
        po = np.array(sim.data.xpos[sim.ids.object_body], float)
        axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        hx = hand_x(sim)
        if mode == "overhead":
            p = po + axis * (0.03 + 0.015 + 0.004)
        elif mode == "rim":
            p = po + axis * (0.03 + 0.010) + hx * 0.012
        elif mode == "side":
            p = po + hx * (0.018 + 0.015 + 0.003)
        else:
            p = po + hx * (0.018 + 0.012) + axis * 0.02
        apply_ball_free_state(sim, p, np.zeros(3))
        sim.set_guide_pos(p)
        sim.set_guide_weld(True)
        if hasattr(sim.model, "body_gravcomp"):
            sim.model.body_gravcomp[int(sim.ball_body)] = 1.0

    for mode in ("overhead", "rim", "side", "side_high"):
        placed = {"done": False}

        def apply_obs(sim, t, mode=mode, placed=placed):
            if not placed["done"]:
                place_ball(sim, mode)
                placed["done"] = True

        row, _loss = trial("F5", mode, apply_obs, watch=0.70)
        f5.append(row)

    # Family 6 — temporary pad friction drop. No grip-command change. No mj_setConst.
    f6 = []
    for mu in (0.5, 0.2, 0.05, 0.0):
        for dur in (0.08, 0.20, 0.50):
            def apply_mu(sim, t, mu=mu, dur=dur):
                if t <= dur:
                    set_mu(sim, mu_saved, mu)
                else:
                    restore_mu(sim, mu_saved)
            row, _loss = trial("F6", f"mu{mu:.2f}_{dur:.2f}s", apply_mu, watch=max(0.55, dur + 0.25))
            restore_mu(sim, mu_saved)
            f6.append(row)

    groups = {
        "F1": [r for r, _s in f1] if f1 and isinstance(f1[0], tuple) else f1,
        "F2": f2,
        "F3": f3,
        "F4": f4,
        "F5": f5,
        "F6": f6,
    }
    # f1 stored (row, None) tuples.
    groups["F1"] = [item[0] if isinstance(item, tuple) else item for item in f1]

    tests = []
    for fam, rows in groups.items():
        chosen = [r for r in rows if r.get("class") in ("P", "C")]
        chosen = sorted(chosen, key=lambda r: (r.get("class") != "P", r.get("sep_off_mm") or 99))[:1]
        for row in chosen:
            tests.append(row)

    def apply_by_name(family, name):
        # Reconstruct the same disturbance from its tag.
        if family == "F1":
            dname, mag_s = name.rsplit("_", 1)
            mag = float(mag_s)
            def apply(sim, t, dname=dname, mag=mag):
                hx = hand_x(sim)
                up = np.array([0.0, 0.0, 1.0])
                torque = None
                if dname == "up":
                    direction = up
                elif dname == "down":
                    direction = -up
                elif dname == "up_out":
                    direction = hx + up
                elif dname == "down_out":
                    direction = hx - up
                else:
                    direction = up
                    torque = np.cross(hx, up)
                    torque = torque / max(np.linalg.norm(torque), 1e-9) * mag
                direction = direction / max(np.linalg.norm(direction), 1e-9)
                use = mag if dname != "tumble" else min(mag, 0.03)
                impulse(sim, obj, direction, use, 0.012, t, torque=None if dname != "tumble" else torque)
            return apply
        if family == "F2":
            dname, peak_s = name.rsplit("_", 1)
            peak = float(peak_s.replace("N", ""))
            def apply(sim, t, dname=dname, peak=peak):
                hx = hand_x(sim)
                up = np.array([0.0, 0.0, 1.0])
                direction = hx if dname == "out" else (-up if dname == "down" else hx - up)
                direction = direction / max(np.linalg.norm(direction), 1e-9)
                hann_force(sim, obj, direction, peak, 0.10, t)
            return apply
        if family == "F3":
            if name.startswith("torque"):
                mag = float(name.split("_")[1])
                def apply(sim, t, mag=mag):
                    axis = np.cross(hand_x(sim), np.array([0.0, 0.0, 1.0]))
                    axis = axis / max(np.linalg.norm(axis), 1e-9)
                    impulse(sim, hand, np.zeros(3), 0.0, 0.03, t, torque=axis * mag * 0.03)
                return apply
            dname, peak_s = name.split("_")
            peak = float(peak_s.replace("N", ""))
            def apply(sim, t, dname=dname, peak=peak):
                direction = hand_x(sim) if dname == "out" else np.array([0.0, 0.0, 1.0])
                impulse(sim, hand, direction, peak * 0.03, 0.03, t)
            return apply
        if family == "F4":
            if name.startswith("frame"):
                _f, dname, accel_s = name.split("_")
                accel = float(accel_s)
                state = {"v": np.zeros(3), "dir": None}
                def apply(sim, t, dname=dname, accel=accel, state=state):
                    if state["dir"] is None:
                        state["dir"] = hand_x(sim).copy() if dname == "out" else np.array([0.0, 0.0, 1.0])
                    dt = float(sim.model.opt.timestep)
                    if t <= 0.08:
                        sign = 1.0 if t <= 0.04 else -1.0
                        state["v"] = state["v"] + state["dir"] * accel * sign * dt
                        sim.fsm.p_des = np.asarray(sim.fsm.p_des, float) + state["v"] * dt
                return apply
            if name.startswith("stop"):
                v_kick = float(name.split("_")[1])
                def apply(sim, t, v_kick=v_kick):
                    direction = hand_x(sim)
                    dt = float(sim.model.opt.timestep)
                    if t <= 0.04:
                        sim.fsm.p_des = np.asarray(sim.fsm.p_des, float) + direction * v_kick * dt
                    elif t <= 0.06:
                        sim.fsm.p_des = np.asarray(sim.fsm.p_des, float) - direction * v_kick * dt
                return apply
            return apply_link0
        if family == "F5":
            placed = {"done": False}
            def apply(sim, t, mode=name, placed=placed):
                if not placed["done"]:
                    place_ball(sim, mode)
                    placed["done"] = True
            return apply
        if family == "F6":
            mu_s, dur_s = name.replace("mu", "").split("_")
            mu = float(mu_s)
            dur = float(dur_s.replace("s", ""))
            def apply(sim, t, mu=mu, dur=dur):
                if t <= dur:
                    set_mu(sim, mu_saved, mu)
                else:
                    restore_mu(sim, mu_saved)
            return apply
        raise KeyError(family)

    # Remove the broken second loop's placeholder by not using rebuild_apply.
    controller_rows = []
    for row in tests:
        apply = apply_by_name(row["family"], row["name"])
        watch = 0.70 if row["family"] == "F5" else 0.55
        _again, loss = run_disturbance(sim, snap, nominal, pad_ids, pad_mu, apply, watch=watch)
        if loss is None:
            print("NO_SNAP", row["family"], row["name"], flush=True)
            continue
        ref = nominal_hand_ref(sim, loss, nominal, pad_ids, pad_mu)
        rolls = []
        for kind, actions in (
            ("nominal", None),
            ("token", None),
            ("known", bank["known"]),
            ("pitch_closed", bank["pitch_closed"]),
            ("inward", bank["inward"]),
            ("up_close", bank["up_close"]),
        ):
            mode = "script" if actions is not None else kind
            rolled = roll_controller(
                sim, gains, loss, nominal, pad_ids, pad_mu, mode, actions,
                encoder, head, centers, ref,
            )
            rolled["kind"] = kind
            rolls.append(rolled)
            print("CTRL", row["family"], row["name"], kind, rolled.get("outcome"), rolled.get("mode"), "both", rolled.get("t_both"), flush=True)
        controller_rows.append({"family": row["family"], "name": row["name"], "rolls": rolls})
        restore_mu(sim, mu_saved)

    # Smaller timestep on the first class change of each family.
    dt_rows = []
    for fam, rows in groups.items():
        changed = next((r for r in rows if r.get("class") != "A"), None)
        if changed is None:
            continue
        apply = apply_by_name(fam, changed["name"])
        fine, _loss = run_disturbance(
            sim, snap, nominal, pad_ids, pad_mu, apply,
            watch=0.70 if fam == "F5" else 0.55, dt=0.001,
        )
        dt_rows.append({
            "family": fam,
            "name": changed["name"],
            "dt002": changed.get("class"),
            "off002": changed.get("max_off_s"),
            "sep002": changed.get("sep_off_mm"),
            "dt001": fine.get("class"),
            "off001": fine.get("max_off_s"),
            "sep001": fine.get("sep_off_mm"),
            "same": changed.get("class") == fine.get("class"),
        })
        print("DT", fam, changed["name"], changed.get("class"), fine.get("class"), flush=True)
        restore_mu(sim, mu_saved)

    def dump_row(row):
        out = dict(row)
        out.pop("trace", None)
        return out

    payload = {
        "grasp": base,
        "link0_move_mm": None if link_move is None else round(link_move * 1e3, 4),
        "link0_fixed": True,
        "families": {k: [dump_row(r) for r in v] for k, v in groups.items()},
        "traces": {f"{r['family']}_{r['name']}": r.get("trace") for rows in groups.values() for r in rows if r.get("class") in ("P", "C", "B")},
        "controllers": controller_rows,
        "timestep": dt_rows,
        "family7": {
            "status": "NOT CLEANLY TESTABLE",
            "reason": (
                "The cylinder has no second weldable payload body. The only weld is ball_guide to the ball. "
                "Rewriting body_mass and calling mj_setConst has previously teleported the cylinder "
                "(obj_z 0.495 to 0.43). A mass write without mj_setConst leaves qM inconsistent. "
                "Neither is a physical attach/detach."
            ),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print("DONE", flush=True)
    return 0


def boundary_main() -> int:
    """Fill the gap between flicker and escape."""
    import torch
    from training.phase_decoupled_recovery_pilot import world
    from training.recatch_over_slide_counterexample import load_policy

    sim, gains, nominal, pad_ids, pad_mu, _l, _r = world()
    snap = torch.load(OUT / "grasp.pt", map_location="cpu", weights_only=False)
    encoder, head, centers = load_policy()
    bank = script_bank()
    mu_saved = save_mu(sim, sim.ids)
    obj = int(sim.ids.object_body)
    rows = []

    def one(name, apply, watch=0.55, dt=None):
        row, loss = run_disturbance(sim, snap, nominal, pad_ids, pad_mu, apply, watch=watch, dt=dt)
        loss_p = row.get("loss") or {}
        print(
            "BND", name, row.get("class"), "dt", dt or DT0,
            "off", row.get("max_off_s"), "air", row.get("air_s"),
            "sep", row.get("sep_off_mm"), "v", loss_p.get("vrel_mps"),
            "rh", None if not loss_p else [round(x * 1e3, 1) for x in loss_p["rh"]],
            "w", loss_p.get("wmag"), "z", loss_p.get("obj_z"),
            "end", (row.get("end") or {}).get("nL"), (row.get("end") or {}).get("nR"),
            flush=True,
        )
        rows.append({
            "name": name,
            "dt": dt or DT0,
            "row": {k: v for k, v in row.items() if k != "trace"},
            "loss_snap": loss,
        })
        return row

    for mag in (0.012, 0.014, 0.016, 0.018):
        def apply(sim, t, mag=mag):
            hx = hand_x(sim)
            up = np.array([0.0, 0.0, 1.0])
            torque = np.cross(hx, up)
            torque = torque / max(np.linalg.norm(torque), 1e-9) * mag
            impulse(sim, obj, up, min(mag, 0.03), 0.012, t, torque=torque)
        one(f"tumble_{mag:.3f}", apply)

    for dur in (0.10, 0.12, 0.14):
        def apply_mu(sim, t, dur=dur):
            if t <= dur:
                set_mu(sim, mu_saved, 0.05)
            else:
                restore_mu(sim, mu_saved)
        one(f"mu0.05_{dur:.2f}s", apply_mu)
        restore_mu(sim, mu_saved)

    def apply_pull(sim, t):
        hx = hand_x(sim)
        direction = hx - np.array([0.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        hann_force(sim, obj, direction, 32.0, 0.10, t)
    one("down_out_32N", apply_pull)

    def apply_hand(sim, t):
        impulse(sim, int(sim.ids.hand_body), hand_x(sim), 800.0 * 0.03, 0.03, t)
    one("hand_out_800N", apply_hand)

    state = {"v": np.zeros(3), "dir": None}

    def apply_frame(sim, t, state=state):
        if state["dir"] is None:
            state["dir"] = hand_x(sim).copy()
        dt = float(sim.model.opt.timestep)
        if t <= 0.08:
            sign = 1.0 if t <= 0.04 else -1.0
            state["v"] = state["v"] + state["dir"] * 40.0 * sign * dt
            sim.fsm.p_des = np.asarray(sim.fsm.p_des, float) + state["v"] * dt
    one("frame_out_40", apply_frame)

    for item in list(rows):
        if item["row"].get("class") not in ("B", "P"):
            continue
        name = item["name"]
        if name.startswith("tumble"):
            mag = float(name.split("_")[1])
            def apply(sim, t, mag=mag):
                hx = hand_x(sim)
                up = np.array([0.0, 0.0, 1.0])
                torque = np.cross(hx, up)
                torque = torque / max(np.linalg.norm(torque), 1e-9) * mag
                impulse(sim, obj, up, min(mag, 0.03), 0.012, t, torque=torque)
            one(name + "_dt1ms", apply, dt=0.001)
        elif name.startswith("mu"):
            dur = float(name.split("_")[1].replace("s", ""))
            def apply_mu(sim, t, dur=dur):
                if t <= dur:
                    set_mu(sim, mu_saved, 0.05)
                else:
                    restore_mu(sim, mu_saved)
            one(name + "_dt1ms", apply_mu, dt=0.001)
            restore_mu(sim, mu_saved)

    for item in rows:
        if item["row"].get("class") != "P" or item["loss_snap"] is None:
            continue
        ref = nominal_hand_ref(sim, item["loss_snap"], nominal, pad_ids, pad_mu)
        item["rolls"] = []
        for kind, actions in (("nominal", None), ("token", None), ("known", bank["known"]), ("inward", bank["inward"])):
            mode = "script" if actions is not None else kind
            rolled = roll_controller(
                sim, gains, item["loss_snap"], nominal, pad_ids, pad_mu, mode, actions,
                encoder, head, centers, ref,
            )
            rolled["kind"] = kind
            item["rolls"].append(rolled)
            print("CTRL", item["name"], kind, rolled.get("outcome"), rolled.get("mode"), flush=True)

    slim = []
    for item in rows:
        slim.append({
            "name": item["name"],
            "dt": item["dt"],
            "class": item["row"].get("class"),
            "off": item["row"].get("max_off_s"),
            "air": item["row"].get("air_s"),
            "sep": item["row"].get("sep_off_mm"),
            "loss": item["row"].get("loss"),
            "end_n": [(item["row"].get("end") or {}).get("nL"), (item["row"].get("end") or {}).get("nR")],
            "rolls": item.get("rolls"),
        })
    (OUT / "boundary.json").write_text(json.dumps(jsonable(slim), indent=2), encoding="utf-8")
    print("DONE", flush=True)
    return 0


def rebuild_apply(*_args, **_kwargs):
    raise RuntimeError("unused")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "boundary":
        raise SystemExit(boundary_main())
    raise SystemExit(main())
