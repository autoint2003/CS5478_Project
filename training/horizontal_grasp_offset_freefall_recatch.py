"""Horizontal grasp, offset, gravity slip, free fall, legal recatch.

No ball, no impulse, no training, no opening before a verified bilateral loss,
and no pose change after a trial has started.
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

from controllers.gripper_controller import finger_opening
from envs.deterioration import body_twist
from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from training.demo_ballistic_impact import HAND_RISE_M, TAU_SEC, assert_noslip, make_sim, z_tgt_of
from training.multi_disturbance_airborne_feasibility import nominal_step, restore
from controllers.jacobian_controller import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from training.recovery_runtime import (
    T_TASK,
    Z_LIFT,
    _dropped,
    _escaped,
    capture_nominal,
    capture_pads,
    gains_from_cfg,
    map7,
    physical_of,
    stamp,
)
from training.replay_core import tick_vw
from training.write_ballistic_impact_scene import CYL_R

OUT = ROOT / "results" / "diagnostics" / "raw" / "horizontal_grasp_offset_freefall"
VID = OUT / "videos"
REPORT = ROOT / "results" / "diagnostics" / "HORIZONTAL_GRASP_OFFSET_FREEFALL_RECATCh.md"
TABLE = 0.40
LOSS_S = 0.020
DT = 0.002


def seat_horizontal(sim):
    q = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    d = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    sim.data.qpos[q:q + 3] = np.array([0.50, 0.0, TABLE + CYL_R + 0.001])
    s = np.sqrt(0.5)
    sim.data.qpos[q + 3:q + 7] = np.array([s, s, 0.0, 0.0])
    sim.data.qvel[d:d + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def axis_of(sim):
    axis = np.array(sim.data.xmat[sim.ids.object_body], float).reshape(3, 3)[:, 2]
    n = float(np.linalg.norm(axis))
    return axis / max(n, 1e-12)


def hold_step(sim, gains):
    z_now = float(z_tgt_of(sim))
    lift_hold = str(sim.fsm.phase) == "lift" and float(sim.fsm.p_des[2]) >= z_now - 1e-9
    if lift_hold:
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
    else:
        nominal_step(sim)


def grasp_settled(sim, gains):
    sim.model.opt.timestep = DT
    sim.reset()
    seat_horizontal(sim)
    hand_z0 = None
    risen_at = None
    for _ in range(int(8.0 / DT)):
        if risen_at is not None and float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-4:
            hold_step(sim, gains)
        else:
            nominal_step(sim)
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if sim.fsm.phase == "lift" and hand_z0 is None:
            hand_z0 = hz
        o = physical_pack(sim)
        risen = (
            hand_z0 is not None
            and (hz - hand_z0) >= HAND_RISE_M + 0.03
            and int(o["nL"]) > 0
            and int(o["nR"]) > 0
            and float(o["obj_z"]) >= Z_AIR
            and float(sim.data.ctrl[7]) <= -17.0
        )
        if risen and risen_at is None:
            risen_at = float(sim.data.time)
        vz = float(body_twist(sim.model, sim.data, sim.ids.object_body)[0][2])
        if risen_at is not None and float(sim.data.time) >= risen_at + 0.30 and abs(vz) < 0.02:
            return stamp(sim)
    return None


def pad_rows(sim, pads_l, pads_r):
    left, right = set(int(g) for g in pads_l), set(int(g) for g in pads_r)
    rows = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        side = "L" if g1 in left or g2 in left else "R" if g1 in right or g2 in right else None
        if side is None:
            continue
        rows.append(
            {
                "side": side,
                "pos_mm": [round(float(x) * 1e3, 2) for x in c.pos],
                "dist_mm": round(float(c.dist) * 1e3, 3),
            }
        )
    return rows


def down_audit(sim):
    q = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    saved_q = np.array(sim.data.qpos, float).copy()
    saved_v = np.array(sim.data.qvel, float).copy()
    flags = []
    for step in range(0, 61):
        sim.data.qpos[:] = saved_q
        sim.data.qpos[q + 2] = saved_q[q + 2] - 0.001 * step
        sim.data.qvel[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        o = physical_pack(sim)
        flags.append(0 if int(o["nL"]) + int(o["nR"]) == 0 else 1)
    sim.data.qpos[:] = saved_q
    sim.data.qvel[:] = saved_v
    mujoco.mj_forward(sim.model, sim.data)
    release = next((i for i, f in enumerate(flags) if f == 0), None)
    reseat = None
    if release is not None:
        reseat = next((i for i in range(release + 1, len(flags)) if flags[i] == 1), None)
    return {
        "release_mm": release,
        "reseat_mm": reseat,
        "open_after_release_mm": None if release is None else (60 if reseat is None else reseat - release),
        "flags": "".join(str(f) for f in flags),
    }


def apply_offset(sim, spec):
    q = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    sim.data.qpos[q] += float(spec["dx"]) * 1e-3
    sim.data.qpos[q + 1] += float(spec["dy"]) * 1e-3
    sim.data.qpos[q + 2] += float(spec["dz"]) * 1e-3
    pitch = float(spec.get("pitch", 0.0))
    if abs(pitch) > 1e-6:
        ang = np.radians(pitch) * 0.5
        extra = np.array([np.cos(ang), np.sin(ang), 0.0, 0.0])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, extra, np.array(sim.data.qpos[q + 3:q + 7], float))
        sim.data.qpos[q + 3:q + 7] = out
    mujoco.mj_forward(sim.model, sim.data)


def sample(sim):
    o = physical_pack(sim)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    bottom = float(o["obj_z"]) - CYL_R
    return {
        "t": float(sim.data.time),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "FnL": float(o["Fn_L"]),
        "FnR": float(o["Fn_R"]),
        "rh": np.array(o["rh"], float),
        "vrel": np.array(o["v_rel_h"], float),
        "vo": np.array(vo, float),
        "wo": np.array(wo, float),
        "z": float(o["obj_z"]),
        "bottom": bottom,
        "aper": float(o["aperture"]),
        "ctrl": float(sim.data.ctrl[7]),
        "hand": np.array(sim.data.xpos[sim.ids.hand_body], float),
    }


def table_hit(row):
    return float(row["bottom"]) <= TABLE + 0.001 or float(row["z"]) < TABLE_DROP


def physics_action(sim, gains, action):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    v, w, tau, _vh, _wb = map7(action, rd, rh)
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    dt = float(sim.model.opt.timestep)
    sim.fsm.p_des = sim.fsm.p_des + v * dt
    sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, w, dt)
    sim.fsm.v_cmd = v.copy()
    sim.fsm.w_cmd = w.copy()
    sim.fsm.v_des = v.copy()
    sim.fsm.w_des = w.copy()
    sim.fsm.fg_cmd = float(-tau)
    apply_cartesian_ctrl(
        sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
        gripper_tau=tau, v_des=v, w_des=w, **gains,
    )
    mujoco.mj_step(sim.model, sim.data)


def track_action(sim, close):
    rh = np.array(physical_pack(sim)["rh"], float)
    err = rh - np.array([0.0, 0.0, 0.097])
    action = np.zeros(7)
    action[:3] = np.clip(err / 0.08, -1.0, 1.0)
    action[6] = 1.0 if close else -1.0
    return action


def scripted_action(mode, step_i):
    down_open = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    down_close = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    still_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    still_close = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if mode == "down4":
        return down_open if step_i < 4 else down_close if step_i < 12 else still_close
    if mode == "down8":
        return down_open if step_i < 8 else down_close if step_i < 16 else still_close
    if mode == "still6":
        return still_open if step_i < 6 else still_close
    return still_close


def roll(sim, gains, mode, policy, film_path, t_end=T_TASK):
    """One controller from the current state through absolute t=12, logged every physics step."""
    from training.real_impact_recatch_over_slide import grab, write_video

    dt = float(sim.model.opt.timestep)
    init = sample(sim)
    t0 = float(init["t"])
    rh0 = init["rh"].copy()
    hand0 = init["hand"].copy()
    loss_t = None
    off_run = 0.0
    free_s = 0.0
    reseat_t = None
    table_t = None
    slip_mm = 0.0
    opened_before = False
    max_ctrl_before = float(init["ctrl"])
    loss_row = None
    first_off = None
    re_t = None
    re_row = None
    recovery = False
    pol_i = 0
    sub = 0
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    max_aper = float(init["aper"])
    hand_at_loss = None
    aper_at_loss = None
    max_hand_from_loss = 0.0
    max_aper_after = float(init["aper"])
    token = None
    if mode == "token":
        import torch
        from envs.observable_obs import ObservableObsState
        from training.recovery_policy_pretraining_preparation import legal_obs
        from training.single_step_token_behavior_audit import decode, predict
        from training.value_guided_recovery_bootstrap import DT_POLICY

        token = {
            "state": ObservableObsState(DT_POLICY),
            "hist": [],
            "prev_l": [],
            "prev": np.zeros(7, np.float32),
            "centers": torch.tensor(policy[2]),
            "predict": predict,
            "decode": decode,
            "legal_obs": legal_obs,
            "encoder": policy[0],
            "head": policy[1],
        }
    frames = []
    renderer = None
    video_out = film_path
    next_frame = t0
    if film_path is not None:
        renderer = mujoco.Renderer(sim.model, 240, 320)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(sim.model, cam)
        cam.distance = 0.42
        cam.azimuth = 90
        cam.elevation = -10
        frames.append(grab(renderer, sim, cam))
    drop_t = None
    trace = []
    next_store = t0
    while float(sim.data.time) < float(t_end) - 1e-9:
        use_nominal = mode == "nominal" or (mode not in ("nominal", "token") and not recovery)
        if use_nominal:
            hold_step(sim, gains)
        elif mode == "token":
            if sub == 0:
                token["hist"].append(np.asarray(token["legal_obs"](sim, token["state"]), np.float32))
                token["prev_l"].append(token["prev"].copy())
                h, prob = token["predict"](
                    token["encoder"], token["head"], token["centers"], token["hist"], token["prev_l"]
                )
                act, _delta = token["decode"](token["head"], h, token["centers"], int(prob.argmax()))
                action = np.asarray(act, float).copy()
                token["prev"] = np.asarray(act, np.float32).copy()
            physics_action(sim, gains, action)
            sub = (sub + 1) % 10
        else:
            if sub == 0:
                if mode == "track":
                    action = track_action(sim, close=(pol_i >= 6))
                else:
                    action = scripted_action(mode, pol_i)
                pol_i += 1
            physics_action(sim, gains, action)
            sub = (sub + 1) % 10
        row = sample(sim)
        both = row["nL"] > 0 and row["nR"] > 0
        none = row["nL"] == 0 and row["nR"] == 0
        max_aper = max(max_aper, float(row["aper"]))
        if loss_t is None:
            max_ctrl_before = max(max_ctrl_before, float(row["ctrl"]))
            if float(row["ctrl"]) > -17.0:
                opened_before = True
            if both:
                slip_mm = max(slip_mm, float((row["rh"][2] - rh0[2]) * 1e3))
            if none:
                off_run += dt
                if first_off is None:
                    first_off = row
                if off_run >= LOSS_S:
                    loss_t = float(row["t"]) - off_run
                    loss_row = row
                    free_s = off_run
                    hand_at_loss = row["hand"].copy()
                    aper_at_loss = float(row["aper"])
            else:
                off_run = 0.0
        else:
            if none and reseat_t is None:
                free_s += dt
            elif reseat_t is None:
                reseat_t = float(row["t"])
            if both and re_t is None and float(row["t"]) > loss_t + LOSS_S:
                re_t = float(row["t"])
                re_row = row
            if hand_at_loss is not None:
                max_hand_from_loss = max(max_hand_from_loss, float(np.linalg.norm(row["hand"] - hand_at_loss)))
                max_aper_after = max(max_aper_after, float(row["aper"]))
            if (
                mode not in ("nominal", "token")
                and not recovery
                and none
                and float(row["t"]) >= loss_t + LOSS_S
            ):
                recovery = True
                sub = 0
                pol_i = 0
        if table_hit(row) and table_t is None:
            table_t = float(row["t"])
        if float(row["t"]) >= next_store:
            trace.append(pack_row(row, rh0))
            next_store += 0.010
        if video_out is not None and renderer is not None and float(row["t"]) >= next_frame and len(frames) < 160:
            frames.append(grab(renderer, sim, cam))
            next_frame += 0.05
        if _dropped({"obj_z": row["z"]}):
            if drop_t is None:
                drop_t = float(row["t"])
            if float(row["t"]) >= drop_t + 0.15:
                break
        else:
            drop_t = None
        if _escaped(sim):
            break
    if renderer is not None:
        if frames and video_out is not None:
            write_video(video_out, frames)
        renderer.close()
    end = sample(sim)
    kind = classify(init, loss_t, free_s, reseat_t, table_t, slip_mm, opened_before, loss_row, end)
    retained = finish_label(sim, gains, end, drop_t)
    return {
        "mode": mode,
        "t0": round(t0, 4),
        "init_bilateral": bool(init["nL"] > 0 and init["nR"] > 0),
        "init_n": [int(init["nL"]), int(init["nR"])],
        "init_ctrl": round(float(init["ctrl"]), 2),
        "init_z": round(float(init["z"]), 4),
        "init_rh_mm": [round(float(x) * 1e3, 2) for x in init["rh"]],
        "init_vo": [round(float(x), 3) for x in init["vo"]],
        "slip_mm": round(float(slip_mm), 2),
        "loss_t": None if loss_t is None else round(float(loss_t - t0), 4),
        "loss_abs": None if loss_t is None else round(float(loss_t), 4),
        "free_s": round(float(free_s if reseat_t is None else max(0.0, reseat_t - (loss_t or reseat_t))), 4),
        "reseat_t": None if reseat_t is None else round(float(reseat_t - t0), 4),
        "table_t": None if table_t is None else round(float(table_t - t0), 4),
        "opened_before_loss": bool(opened_before),
        "max_ctrl_before_loss": round(float(max_ctrl_before), 2),
        "loss_vo": None if loss_row is None else [round(float(x), 3) for x in loss_row["vo"]],
        "loss_rh_mm": None if loss_row is None else [round(float(x) * 1e3, 2) for x in loss_row["rh"]],
        "first_off_vo": None if first_off is None else [round(float(x), 3) for x in first_off["vo"]],
        "first_off_rh_mm": None if first_off is None else [round(float(x) * 1e3, 2) for x in first_off["rh"]],
        "loss_z": None if loss_row is None else round(float(loss_row["z"]), 4),
        "re_t": None if re_t is None else round(float(re_t - t0), 4),
        "re_n": None if re_row is None else [int(re_row["nL"]), int(re_row["nR"])],
        "hand_move_mm": round(float(max_hand_from_loss) * 1e3, 2),
        "aper_at_loss": None if aper_at_loss is None else round(float(aper_at_loss), 4),
        "aper_max_after": round(float(max_aper_after), 4),
        "end_z": round(float(end["z"]), 4),
        "end_n": [int(end["nL"]), int(end["nR"])],
        "end_t": round(float(end["t"]), 3),
        "regime": kind,
        "retained_to_12": retained,
        "trace_10ms": trace[::5],
    }


def pack_row(row, rh0):
    return {
        "t": round(float(row["t"]), 4),
        "n": [int(row["nL"]), int(row["nR"])],
        "Fn": [round(float(row["FnL"]), 2), round(float(row["FnR"]), 2)],
        "rh_mm": [round(float(x) * 1e3, 2) for x in row["rh"]],
        "vrel_mm_s": [round(float(x) * 1e3, 1) for x in row["vrel"]],
        "vo": [round(float(x), 3) for x in row["vo"]],
        "wo": [round(float(x), 3) for x in row["wo"]],
        "z": round(float(row["z"]), 4),
        "aper": round(float(row["aper"]), 4),
        "ctrl": round(float(row["ctrl"]), 2),
        "slip_mm": round(float((row["rh"][2] - rh0[2]) * 1e3), 2),
    }


def classify(init, loss_t, free_s, reseat_t, table_t, slip_mm, opened_before, loss_row, end):
    bilateral0 = int(init["nL"]) > 0 and int(init["nR"]) > 0
    if not bilateral0:
        return "STARTS_FREE"
    if loss_t is None:
        return "B_SLIP_RETAINED" if slip_mm >= 2.0 else "A_STABLE"
    if opened_before:
        return "OPENED_BEFORE_LOSS"
    gap = float(free_s)
    held_up = int(end["nL"]) > 0 and int(end["nR"]) > 0 and float(end["z"]) > Z_LIFT
    reseated_before_table = reseat_t is not None and (table_t is None or float(reseat_t) < float(table_t))
    if gap < LOSS_S or (reseated_before_table and held_up):
        return "PASSIVE_RESEAT"
    if loss_row is None:
        return "D_UNRECOVERABLE"
    speed = float(np.linalg.norm(loss_row["vo"]))
    rh = loss_row["rh"]
    near = abs(float(rh[0])) < 0.040 and abs(float(rh[1])) < 0.040
    above = float(loss_row["bottom"]) > TABLE + 0.005
    window = None if table_t is None else float(table_t) - float(loss_t)
    catchable = above and near and speed < 1.2 and (window is None or window >= 0.06) and not held_up
    if catchable and gap >= LOSS_S:
        return "C_CATCHABLE"
    return "D_UNRECOVERABLE"


def finish_label(sim, gains, end, drop_t):
    del gains
    if drop_t is not None or _escaped(sim) or float(end["t"]) < T_TASK - 0.05:
        return False
    both = int(end["nL"]) > 0 and int(end["nR"]) > 0
    return both and float(end["z"]) > Z_LIFT


def end_state(sim):
    row = sample(sim)
    return {"obj_z": row["z"], "nL": row["nL"], "nR": row["nR"]}


def begin(sim, snap, nominal, pads, pad_mu, spec):
    info = restore(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = DT
    if not info.get("pose_ok", False):
        return False
    apply_offset(sim, spec)
    return True


def spec_name(spec):
    name = "dz{dz}_dx{dx}_dy{dy}_p{pitch}".format(**spec)
    if spec.get("mass") is not None:
        name += f"_m{float(spec['mass']):.3f}"
    if spec.get("mu") is not None:
        name += f"_mu{float(spec['mu']):.2f}"
    return name


def grid():
    rows = []
    for dz in (0, -4, -8, -12, -16, -18, -20, -22):
        rows.append({"dx": 0, "dy": 0, "dz": dz, "pitch": 0})
    for dz in (-12, -16, -18):
        for dx in (-2, 2):
            rows.append({"dx": dx, "dy": 0, "dz": dz, "pitch": 0})
    for dy in (-10, 10):
        rows.append({"dx": 0, "dy": dy, "dz": -16, "pitch": 0})
    for pitch in (8, 15):
        rows.append({"dx": 0, "dy": 0, "dz": 0, "pitch": pitch})
        rows.append({"dx": 0, "dy": 0, "dz": -12, "pitch": pitch})
    return rows


def active_success(nom, best):
    if best is None or nom is None:
        return False
    if nom.get("retained_to_12"):
        return False
    if not best.get("retained_to_12"):
        return False
    if not nom.get("init_bilateral") or nom.get("opened_before_loss"):
        return False
    if nom.get("regime") not in ("C_CATCHABLE", "D_UNRECOVERABLE"):
        return False
    if nom.get("free_s", 0) < LOSS_S:
        return False
    moved = float(best.get("hand_move_mm") or 0) >= 4.0 or (
        best.get("aper_at_loss") is not None
        and float(best.get("aper_max_after") or 0) >= float(best["aper_at_loss"]) + 0.004
    )
    return bool(best.get("re_t") is not None and moved)


def write_report(ref, audit, rows, verdict):
    lines = [
        "# Horizontal grasp offset to free fall",
        "",
        "A cylinder is seated on its side and pinched by the Panda fingers before any offset.",
        "Offsets change only that initial relative pose. The rollout itself is ordinary physics:",
        "closed nominal hold, gravity, and, after a verified bilateral loss, a legal 7-D hand command.",
        "No ball, no applied wrench, no training, and no pose edit after the trial starts.",
        "",
        "## Reference grasp",
        "",
        f"- Bilateral contacts {ref['n'][0]}/{ref['n'][1]}, pad forces {ref['Fn'][0]:.2f} N and {ref['Fn'][1]:.2f} N.",
        f"- Aperture {ref['aper']:.4f} m, grip command {ref['ctrl']:.1f} N·m, COM height {ref['z']:.4f} m.",
        f"- Cylinder axis in world {ref['axis']}. Hand-frame COM {ref['rh_mm']} mm.",
        f"- Nominal continuation: {ref['t12']}.",
        f"- Downward rigid translation stays in contact for {audit['release_mm']} mm, then stays free.",
        f"- First later contact on that path: {audit['reseat_mm']}. Open distance after release: {audit['open_after_release_mm']} mm.",
        "",
        "Pad contact positions at the settled grasp (mm):",
        "",
    ]
    for row in ref["pads"][:8]:
        lines.append(f"- {row['side']} at {row['pos_mm']}, gap {row['dist_mm']} mm")
    lines += [
        "",
        "## Offset trials",
        "",
        "Nominal closed hold from each offset. Token and recatch columns are filled where a loss occurs,",
        "from that same initial pose. `active recatch` requires a real free-fall gap, a hand or aperture",
        "change after that gap, recontact, and retention to absolute t=12, while the nominal hold does not retain.",
        "",
        "| offset | bilateral | slip mm | loss s | free s | table s | nominal | token | best recatch | t=12 | active |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        off = spec_name(row["spec"])
        nom = row["nominal"]
        tok = row.get("token")
        best = row.get("best")
        lines.append(
            "| {off} | {bi} | {slip} | {loss} | {free} | {table} | {nom} | {tok} | {best} | {t12} | {act} |".format(
                off=off,
                bi="yes" if nom["init_bilateral"] else "no",
                slip=nom["slip_mm"],
                loss=nom["loss_t"] if nom["loss_t"] is not None else "",
                free=nom["free_s"],
                table=nom["table_t"] if nom["table_t"] is not None else "",
                nom=nom["regime"] + (" retained" if nom["retained_to_12"] else " lost"),
                tok="" if tok is None else tok["regime"] + (" retained" if tok["retained_to_12"] else " lost"),
                best="" if best is None else f"{best['mode']} {best['regime']}" + (" retained" if best["retained_to_12"] else " lost"),
                t12="yes" if (best or nom)["retained_to_12"] else "no",
                act="yes" if row.get("active") else "no",
            )
        )
    lines += [
        "",
        "## Regimes",
        "",
        "A is a closed hold that does not slip. B slips and stays in the hand. C is a bilateral loss with a",
        "contact-free interval, the cylinder still above the table and still near the pinch. D is a loss that",
        "is already too fast, too far, or already on the table. A gap shorter than 20 ms that returns to both",
        "pads is a passive reseat.",
        "",
        f"Videos: `{VID.relative_to(ROOT).as_posix()}`.",
        "",
        verdict,
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    sim, cfg = make_sim()
    assert_noslip(sim)
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pads, pad_mu = capture_pads(sim)
    from envs.dynamics import finger_pad_geom_ids

    pads_l, pads_r = finger_pad_geom_ids(sim.model, sim.ids)
    snap = grasp_settled(sim, gains)
    if snap is None:
        print("NO_GRASP", flush=True)
        write_report(
            {"n": [0, 0], "Fn": [0, 0], "aper": 0, "ctrl": 0, "z": 0, "axis": [], "rh_mm": [], "t12": "NO_GRASP", "pads": []},
            {"release_mm": None, "reseat_mm": None, "open_after_release_mm": None},
            [],
            "HORIZONTAL GRASP HAS NO CLEAN DOWNWARD FREE-FALL PATH",
        )
        return
    o = physical_pack(sim)
    audit = down_audit(sim)
    print("AUDIT", audit["release_mm"], audit["reseat_mm"], audit["open_after_release_mm"], flush=True)
    ref = {
        "n": [int(o["nL"]), int(o["nR"])],
        "Fn": [float(o["Fn_L"]), float(o["Fn_R"])],
        "aper": float(o["aperture"]),
        "ctrl": float(o["tau"]),
        "z": float(o["obj_z"]),
        "axis": [round(float(x), 3) for x in axis_of(sim)],
        "rh_mm": [round(float(x) * 1e3, 2) for x in np.array(o["rh"], float)],
        "t12": "PENDING",
        "pads": pad_rows(sim, pads_l, pads_r),
    }
    info = restore(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = DT
    print("RESTORE", info.get("pose_ok"), flush=True)
    ref_roll = roll(sim, gains, "nominal", None, VID / "stable_grasp.mp4")
    ref["t12"] = "RETAINED_TO_12" if ref_roll["retained_to_12"] else ref_roll["regime"]
    print("REF", ref["n"], ref["z"], ref["t12"], ref["rh_mm"], flush=True)
    if audit["release_mm"] is None or (audit["reseat_mm"] is not None and audit["open_after_release_mm"] < 15):
        write_report(ref, audit, [], "HORIZONTAL GRASP HAS NO CLEAN DOWNWARD FREE-FALL PATH")
        print("CAGED", flush=True)
        return
    if not ref_roll["retained_to_12"]:
        write_report(ref, audit, [], "INCONCLUSIVE")
        print("REF_NOT_RETAINED", flush=True)
        return
    policy = None
    results = []
    for spec in grid():
        ok = begin(sim, snap, nominal, pads, pad_mu, spec)
        if not ok:
            print("BAD_RESTORE", spec, flush=True)
            continue
        film = None
        nom = roll(sim, gains, "nominal", None, film)
        print(
            "NOM", spec_name(spec), nom["regime"], "bi", nom["init_bilateral"],
            "slip", nom["slip_mm"], "loss", nom["loss_t"], "free", nom["free_s"],
            "table", nom["table_t"], "z0", nom["init_z"], "vo", nom["init_vo"],
            flush=True,
        )
        results.append({"spec": spec, "nominal": nom})
    # Controllers on the reference, the last retain, and every loss.
    from training.recatch_over_slide_counterexample import load_policy

    policy = load_policy()
    interesting = []
    last_retain = None
    for i, row in enumerate(results):
        if row["nominal"]["retained_to_12"] and row["nominal"]["init_bilateral"]:
            last_retain = i
        if row["nominal"]["loss_t"] is not None or row["nominal"]["regime"] in ("C_CATCHABLE", "D_UNRECOVERABLE", "STARTS_FREE", "PASSIVE_RESEAT"):
            interesting.append(i)
    chosen = []
    if last_retain is not None:
        chosen.append(last_retain)
    for i in interesting:
        if i not in chosen:
            chosen.append(i)
    modes = ("down4", "down8", "still6", "track")
    for i in chosen:
        row = results[i]
        spec = row["spec"]
        begin(sim, snap, nominal, pads, pad_mu, spec)
        tok = roll(sim, gains, "token", policy, None)
        row["token"] = {k: v for k, v in tok.items() if k != "trace_10ms"}
        print("TOK", spec_name(spec), tok["regime"], "ret", tok["retained_to_12"], "loss", tok["loss_t"], flush=True)
        best = None
        for mode in modes:
            begin(sim, snap, nominal, pads, pad_mu, spec)
            got = roll(sim, gains, mode, None, None)
            slim = {k: v for k, v in got.items() if k != "trace_10ms"}
            print("REC", spec_name(spec), mode, got["regime"], "ret", got["retained_to_12"], "re", got["re_t"], "hand", got["hand_move_mm"], flush=True)
            if best is None or (got["retained_to_12"] and not best["retained_to_12"]) or (
                got["retained_to_12"] == best["retained_to_12"] and (got["re_t"] or 99) < (best["re_t"] or 99)
            ):
                best = slim
        row["best"] = best
        row["active"] = active_success(row["nominal"], best)
    # Videos of the four requested regimes.
    films = {}
    for row in results:
        reg = row["nominal"]["regime"]
        if reg == "A_STABLE" and "stable" not in films:
            films["stable"] = row
        if reg == "B_SLIP_RETAINED" and "slip" not in films:
            films["slip"] = row
        if reg == "C_CATCHABLE" and "catch" not in films:
            films["catch"] = row
        if reg == "D_UNRECOVERABLE" and "drop" not in films:
            films["drop"] = row
    if "slip" not in films and last_retain is not None:
        films["slip"] = results[last_retain]
    for key, row in films.items():
        path = VID / f"{key}_{spec_name(row['spec'])}.mp4"
        begin(sim, snap, nominal, pads, pad_mu, row["spec"])
        roll(sim, gains, "nominal", None, path)
        print("VID", path.name, flush=True)
    success_rows = [r for r in results if r.get("active")]
    if success_rows:
        base = success_rows[0]
        basin = robustness(sim, gains, snap, nominal, pads, pad_mu, base)
        for item in basin:
            results.append(item)
        n_ok = sum(1 for item in basin if item.get("active"))
        verdict = (
            "HORIZONTAL OFFSET CREATES A ROBUST ACTIVE-RECATCh BASIN"
            if n_ok >= 3
            else "INCONCLUSIVE"
        )
    else:
        any_c = any(r["nominal"]["regime"] == "C_CATCHABLE" for r in results)
        any_loss = any(r["nominal"]["loss_t"] is not None for r in results)
        if any_c:
            verdict = "CATCHABLE FREE-FALL STATES EXIST BUT CURRENT CONTROL CANNOT RECATCh THEM"
        elif any_loss or any(not r["nominal"]["retained_to_12"] for r in results):
            verdict = "HORIZONTAL OFFSETS PRODUCE ONLY RETENTION OR UNRECOVERABLE DROP"
        else:
            verdict = "HORIZONTAL OFFSETS PRODUCE ONLY RETENTION OR UNRECOVERABLE DROP"
    slim_rows = []
    for row in results:
        slim_rows.append(
            {
                "spec": row["spec"],
                "nominal": {k: v for k, v in row["nominal"].items() if k != "trace_10ms"},
                "token": row.get("token"),
                "best": row.get("best"),
                "active": bool(row.get("active")),
            }
        )
    (OUT / "summary.json").write_text(json.dumps({"ref": ref, "audit": {k: v for k, v in audit.items() if k != "flags"}, "rows": slim_rows, "verdict": verdict}, indent=2), encoding="utf-8")
    (OUT / "down_flags.txt").write_text(audit["flags"], encoding="utf-8")
    write_report(ref, audit, results, verdict)
    print("VERDICT", verdict, flush=True)


def robustness(sim, gains, snap, nominal, pads, pad_mu, base):
    spec0 = dict(base["spec"])
    trials = []
    variants = [
        {**spec0, "dz": spec0["dz"] - 1},
        {**spec0, "dz": spec0["dz"] + 1},
        {**spec0, "dx": spec0["dx"] + 1},
        {**spec0, "dy": spec0["dy"] + 2},
        {**spec0, "pitch": spec0["pitch"] + 3},
    ]
    saved_mass = float(sim.model.body_mass[sim.ids.object_body])
    for spec in variants:
        begin(sim, snap, nominal, pads, pad_mu, spec)
        nom = roll(sim, gains, "nominal", None, None)
        best = None
        if nom["loss_t"] is not None:
            for mode in ("down4", "down8", "still6", "track"):
                begin(sim, snap, nominal, pads, pad_mu, spec)
                got = roll(sim, gains, mode, None, None)
                slim = {k: v for k, v in got.items() if k != "trace_10ms"}
                if best is None or got["retained_to_12"]:
                    best = slim
                    if got["retained_to_12"]:
                        break
        item = {"spec": spec, "nominal": nom, "best": best, "active": active_success(nom, best), "robustness": True}
        trials.append(item)
        print("ROB", spec_name(spec), nom["regime"], item["active"], flush=True)
    # Mass and friction on the successful pose.
    for mass, mu in ((saved_mass * 1.15, None), (None, 0.7)):
        begin(sim, snap, nominal, pads, pad_mu, spec0)
        if mass is not None:
            sim.model.body_mass[sim.ids.object_body] = float(mass)
        if mu is not None:
            from envs.dynamics import set_finger_object_sliding_mu
            set_finger_object_sliding_mu(sim.model, sim.ids, float(mu), data=None)
        nom = roll(sim, gains, "nominal", None, None)
        best = None
        if nom["loss_t"] is not None:
            begin(sim, snap, nominal, pads, pad_mu, spec0)
            if mass is not None:
                sim.model.body_mass[sim.ids.object_body] = float(mass)
            if mu is not None:
                set_finger_object_sliding_mu(sim.model, sim.ids, float(mu), data=None)
            got = roll(sim, gains, "down8", None, None)
            best = {k: v for k, v in got.items() if k != "trace_10ms"}
        spec = dict(spec0)
        spec["mass"] = mass
        spec["mu"] = mu
        trials.append({"spec": spec, "nominal": nom, "best": best, "active": active_success(nom, best), "robustness": True})
        print("ROB_DYN", mass, mu, nom["regime"], trials[-1]["active"], flush=True)
    sim.model.body_mass[sim.ids.object_body] = saved_mass
    return trials


def fine():
    """1 mm vertical sweep from a hand that has stopped rising."""
    sim, cfg = make_sim()
    assert_noslip(sim)
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pads, pad_mu = capture_pads(sim)
    snap = grasp_settled(sim, gains)
    if snap is None:
        print("NO_GRASP", flush=True)
        return
    row0 = sample(sim)
    print(
        "QUIET", "n", row0["nL"], row0["nR"], "z", round(row0["z"], 4),
        "vo", [round(float(x), 3) for x in row0["vo"]],
        "rh", [round(float(x) * 1e3, 2) for x in row0["rh"]],
        flush=True,
    )
    for dz in (0, -2, -4, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18):
        spec = {"dx": 0, "dy": 0, "dz": dz, "pitch": 0}
        begin(sim, snap, nominal, pads, pad_mu, spec)
        raw = sample(sim)
        for _ in range(40):
            hold_step(sim, gains)
        got = roll(sim, gains, "nominal", None, None, t_end=float(sim.data.time) + 2.5)
        air = None
        if got["loss_t"] is not None and got["table_t"] is not None:
            air = round(float(got["table_t"]) - float(got["loss_t"]), 3)
        print(
            "FINE", dz,
            "raw", raw["nL"], raw["nR"],
            "seat", got["init_n"], got["init_bilateral"],
            "slip", got["slip_mm"],
            "loss", got["loss_t"],
            "air", air,
            "table", got["table_t"],
            "vo", got["loss_vo"],
            "rh", got["loss_rh_mm"],
            "reg", got["regime"],
            "z0", got["init_z"],
            "v0", got["init_vo"],
            flush=True,
        )


def edge():
    """t=12 and legal recatch on the quiet-hold cliff between 15 and 16 mm."""
    from training.recatch_over_slide_counterexample import load_policy

    sim, cfg = make_sim()
    assert_noslip(sim)
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pads, pad_mu = capture_pads(sim)
    snap = grasp_settled(sim, gains)
    policy = load_policy()
    VID.mkdir(parents=True, exist_ok=True)
    jobs = [
        (-15.0, "nominal", VID / "slow_slip_dz-15.mp4"),
        (-15.5, "nominal", None),
        (-16.0, "nominal", VID / "drop_boundary_dz-16.mp4"),
        (-16.0, "token", None),
        (-16.0, "down4", None),
        (-16.0, "down8", VID / "recatch_down8_dz-16.mp4"),
        (-16.0, "still6", None),
        (-16.0, "track", None),
        (0.0, "nominal", VID / "stable_quiet.mp4"),
    ]
    for dz, mode, film in jobs:
        spec = {"dx": 0.0, "dy": 0.0, "dz": dz, "pitch": 0.0}
        begin(sim, snap, nominal, pads, pad_mu, spec)
        for _ in range(40):
            hold_step(sim, gains)
        got = roll(sim, gains, mode, policy if mode == "token" else None, film)
        air = None
        if got["loss_t"] is not None and got["table_t"] is not None:
            air = round(float(got["table_t"]) - float(got["loss_t"]), 3)
        print(
            "EDGE", dz, mode,
            "seat", got["init_n"], got["init_bilateral"],
            "slip", got["slip_mm"],
            "loss", got["loss_t"],
            "air", air,
            "table", got["table_t"],
            "first", got["first_off_vo"], got["first_off_rh_mm"],
            "lossvo", got["loss_vo"],
            "lossrh", got["loss_rh_mm"],
            "re", got["re_t"],
            "hand", got["hand_move_mm"],
            "aper", got["aper_at_loss"], got["aper_max_after"],
            "ret", got["retained_to_12"],
            "reg", got["regime"],
            "open_before", got["opened_before_loss"],
            flush=True,
        )


def dbg():
    sim, cfg = make_sim()
    assert_noslip(sim)
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pads, pad_mu = capture_pads(sim)
    snap = grasp_settled(sim, gains)
    begin(sim, snap, nominal, pads, pad_mu, {"dx": 0, "dy": 0, "dz": -16, "pitch": 0})
    for _ in range(40):
        hold_step(sim, gains)
    t0 = float(sim.data.time)
    off = 0.0
    recovery = False
    sub = 0
    pol_i = 0
    action = np.zeros(7)
    while float(sim.data.time) < t0 + 3.0:
        if not recovery:
            hold_step(sim, gains)
        else:
            if sub == 0:
                action = scripted_action("down8", pol_i)
                pol_i += 1
            physics_action(sim, gains, action)
            sub = (sub + 1) % 10
        row = sample(sim)
        none = row["nL"] == 0 and row["nR"] == 0
        if none:
            off += DT
            if off <= 0.08:
                print(
                    "OFF", round(off * 1e3, 1),
                    "ctrl", round(row["ctrl"], 1),
                    "aper", round(row["aper"], 4),
                    "n", row["nL"], row["nR"],
                    "z", round(row["z"], 4),
                    "vz", round(float(row["vo"][2]), 3),
                    "rhz", round(float(row["rh"][2]) * 1e3, 1),
                    flush=True,
                )
        else:
            off = 0.0
        if (not recovery) and none and float(row["aper"]) < 0.010:
            recovery = True
            sub = 0
            pol_i = 0
            print(
                "LOSS", round(row["t"] - t0, 3),
                "z", round(row["z"], 3),
                "vz", round(float(row["vo"][2]), 3),
                "aper", round(row["aper"], 4),
                "rhz", round(float(row["rh"][2]) * 1e3, 1),
                flush=True,
            )
        if recovery and int((row["t"] - t0) * 500) % 10 == 0:
            print(
                "OPEN", round(row["t"] - t0, 3),
                "ctrl", round(row["ctrl"], 2),
                "aper", round(row["aper"], 4),
                "n", row["nL"], row["nR"],
                "z", round(row["z"], 3),
                "pol", pol_i,
                flush=True,
            )
        if row["bottom"] <= TABLE + 0.001:
            print("TABLE", round(row["t"] - t0, 3), "aper", round(row["aper"], 4), "ctrl", round(row["ctrl"], 2), flush=True)
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fine":
        fine()
    elif len(sys.argv) > 1 and sys.argv[1] == "edge":
        edge()
    elif len(sys.argv) > 1 and sys.argv[1] == "dbg":
        dbg()
    else:
        main()



