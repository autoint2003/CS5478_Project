"""CENTER-6 large-angle wrist gravity transfer (contact-preserving authority).

Uses frozen snap_early.pkl. No impact retune, no SAC, no recatch, no MP4.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import mujoco
import numpy as np

from controllers.jacobian_controller import gains_from_cfg, ori_error_deg
from controllers.residual import RECOVERY4D_W_HY_MAX
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import TABLE_DROP
from envs.deterioration import body_twist
from training.demo_ballistic_impact import dump_json, make_sim, z_tgt_of
from training.demo_ballistic_recovery import (
    HOLD_S,
    OUT,
    RAW,
    SITE,
    V_HIT,
    continue_zero,
    make_snap,
    measure,
    restore_audit,
    restore_ballistic,
    run_center6_until,
    run_episode,
    slim_m,
    tick_4d,
)
from training.demo_teleport_recovery_state import signed_rot_about_y_deg
from training.grav_reposition_v2_viz import apply_camera_preset
from training.map_ballistic_disturbance import CAM, _viewer_overlay
from training.replay_core import freeze


def dump(path, obj):
    def fix(x):
        if isinstance(x, dict):
            return {str(k): fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [fix(v) for v in x]
        if isinstance(x, np.ndarray):
            return fix(x.tolist())
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return x

    dump_json(path, fix(obj))

LA_RAW = RAW / "large_angle"
ANGLES_DEG = (0.0, 30.0, 60.0, 90.0, 105.0, 120.0)
GRIP_TAUS = (-18.0, -8.0, -5.0, -4.0, -3.0, -2.0)
OMEGA = float(RECOVERY4D_W_HY_MAX)  # legal bound, not an optimized rate
HOLD_AFTER_ROT = 0.15
GRIP_S = 0.70
SLIP_CAP_S = 0.80
BRAKE_S = 0.40
G_HX_SUBSTANTIAL = 4.0  # m/s^2
CORRECT_MM = 0.80  # min |Δr_h.x| to call a directed correction
LOG_DT = 0.008
RETURN_KP = 4.0  # 1/s on body-y SO(3) error; saturates at legal ω_y
RETURN_DES_STOP_DEG = 0.50
RETURN_HAND_STOP_DEG = 2.5
RETURN_CATCH_S = 0.30
RETURN_TIMEOUT_S = 4.0


def so3_log(R) -> np.ndarray:
    R = np.asarray(R, float).reshape(3, 3)
    c = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-10:
        return np.zeros(3)
    s = np.sin(ang)
    if s < 1e-10:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float) / (2.0 * s)
    return axis * ang


def so3_angle_deg(Ra, Rb) -> float:
    return float(ori_error_deg(np.asarray(Ra, float).reshape(3, 3), np.asarray(Rb, float).reshape(3, 3)))


def omega_y_to_nominal(R_nom, R_des) -> float:
    """Legal body-y rate that reduces SO(3) error of R_des toward R_nom."""
    phi = so3_log(np.asarray(R_des, float).reshape(3, 3).T @ np.asarray(R_nom, float).reshape(3, 3))
    e_y = float(phi[1])
    if abs(e_y) < np.deg2rad(RETURN_DES_STOP_DEG):
        return 0.0
    return float(np.clip(RETURN_KP * e_y, -OMEGA, OMEGA))


def ori_pack(sim, R_nom):
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    wh = body_twist(sim.model, sim.data, sim.ids.hand_body)[1]
    return {
        "err_des_deg": so3_angle_deg(R_nom, Rd),
        "err_hand_deg": so3_angle_deg(R_nom, Rh),
        "y_err_des_deg": float(np.degrees(so3_log(Rd.T @ np.asarray(R_nom, float).reshape(3, 3))[1])),
        "w_hand": np.asarray(wh, float).copy(),
        "R_des": Rd.copy(),
        "R_hand": Rh.copy(),
    }


def load_early():
    p = RAW / "snap_early.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"missing frozen EARLY snapshot {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def load_zero_rows():
    z = np.load(RAW / "zero_center6.npz", allow_pickle=True)
    rows = []
    t = z["t"]
    for i in range(len(t)):
        rows.append(
            {
                "t": float(t[i]),
                "rh": np.asarray(z["rh"][i], float),
                "nL": int(z["nL"][i]),
                "nR": int(z["nR"][i]),
            }
        )
    return rows


def pred_g_h(g0, theta_deg):
    th = np.deg2rad(float(theta_deg))
    c, s = np.cos(th), np.sin(th)
    gx, gy, gz = (float(g0[0]), float(g0[1]), float(g0[2]))
    # R1 = R0 @ Ry(θ) with atan2(d02, d00)=θ  => g_h' = Ry(-θ) g_h0
    return np.array([c * gx - s * gz, gy, s * gx + c * gz], float)


def both_off_persist(log, dt=0.002) -> float:
    best = cur = 0.0
    for r in log:
        if int(r["nL"]) == 0 and int(r["nR"]) == 0:
            cur += dt
            best = max(best, cur)
        else:
            cur = 0.0
    return best


def subsample(log, dt_keep=0.020):
    out = []
    for m in log:
        sm = slim_m(m)
        if not out or float(sm["t"]) - float(out[-1]["t"]) >= dt_keep - 1e-9:
            out.append(sm)
    if log:
        last = slim_m(log[-1])
        if not out or abs(float(last["t"]) - float(out[-1]["t"])) > 1e-6:
            out.append(last)
    return out


def tick_log(sim, gains, omega_y, v_x, v_z, tau, Rh0, log, n, ctl=None, label=""):
    t_table = None
    lost_t = None
    for _ in range(max(int(n), 0)):
        if ctl and ctl.get("reset"):
            break
        tick_4d(sim, gains, omega_y, v_x, v_z, tau)
        m = measure(sim, tau, Rh0)
        m["omega_y"] = float(omega_y)
        m["v_x"] = float(v_x)
        m["v_z"] = float(v_z)
        if not log or (m["t"] - log[-1]["t"] >= LOG_DT):
            log.append(m)
        if t_table is None and m["obj_z"] < TABLE_DROP:
            t_table = m["t"]
            break
        if lost_t is None and m["nL"] == 0 and m["nR"] == 0:
            lost_t = m["t"]
        if ctl and ctl.get("sync"):
            ctl["overlay"] = {
                "case": label or ctl.get("label", ""),
                "t": m["t"],
                "phase": label,
                "nL": m["nL"],
                "nR": m["nR"],
            }
            ctl["sync"]()
    if log:
        log[-1] = measure(sim, tau, Rh0)
        log[-1]["omega_y"] = float(omega_y)
        log[-1]["v_x"] = float(v_x)
        log[-1]["v_z"] = float(v_z)
    return t_table, lost_t


def rotate_to(sim, gains, Rh0, target_deg, tau, log, ctl=None, label="ROTATE"):
    freeze(sim)
    dt = float(sim.model.opt.timestep)
    goal = abs(float(target_deg))
    sign = 1.0 if float(target_deg) >= 0 else -1.0
    omega = OMEGA if goal > 0.5 else 0.0
    t0 = float(sim.data.time)
    timeout = t0 + (np.deg2rad(goal) / max(OMEGA, 1e-6)) + 1.2
    stall_t = t0
    last_abs = 0.0
    t_table = lost_t = None
    reached = goal <= 0.5
    while float(sim.data.time) < timeout:
        if ctl and ctl.get("reset"):
            break
        ang = signed_rot_about_y_deg(
            Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        )
        if abs(ang) >= goal - 0.6:
            freeze(sim)
            reached = True
            break
        if abs(ang) > last_abs + 0.15:
            last_abs = abs(ang)
            stall_t = float(sim.data.time)
        elif float(sim.data.time) - stall_t > 0.35 and goal > 1.0:
            break
        tt, lt = tick_log(
            sim, gains, sign * omega, 0.0, 0.0, tau, Rh0, log, 1, ctl=ctl, label=label
        )
        if tt is not None:
            t_table = tt
            break
        if lt is not None:
            lost_t = lt
    freeze(sim)
    n_hold = int(round(HOLD_AFTER_ROT / dt))
    if t_table is None:
        tt, lt = tick_log(
            sim, gains, 0.0, 0.0, 0.0, tau, Rh0, log, n_hold, ctl=ctl, label="HOLD_SECURE"
        )
        t_table = tt
        if lt is not None:
            lost_t = lost_t or lt
    ang = signed_rot_about_y_deg(
        Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    )
    m = measure(sim, tau, Rh0)
    return {
        "target_deg": float(target_deg),
        "actual_deg": float(ang),
        "reached": bool(reached and abs(ang) >= goal - 1.5),
        "t_table": t_table,
        "lost_t": lost_t,
        "hold": slim_m(m),
        "dt_rot": float(sim.data.time) - t0,
    }


def start_from_early(sim, snap):
    restore_ballistic(sim, snap)
    freeze(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    m0 = measure(sim, -18.0, Rh0)
    return Rh0, m0


def contacts_ok(m) -> bool:
    return int(m["nL"]) > 0 and int(m["nR"]) > 0


def classify_grip(m0, m1, log, desired_drhx_sign, g_hx):
    dr = float(m1["rh"][0]) - float(m0["rh"][0])
    table = m1["obj_z"] < TABLE_DROP
    off = both_off_persist(log) >= 0.020
    kept = contacts_ok(m1) and not table and not off
    toward = (desired_drhx_sign < 0 and dr < -CORRECT_MM * 1e-3) or (
        desired_drhx_sign > 0 and dr > CORRECT_MM * 1e-3
    )
    agrees = (
        bool(np.sign(dr) == np.sign(g_hx))
        if abs(g_hx) > 1.0 and abs(dr) > 4e-4
        else None
    )
    vrel_max = max(float(np.linalg.norm(r["v_rel_h"])) for r in log) if log else 0.0
    ballistic = (not kept) or (vrel_max > 0.12 and not kept)
    if table or off:
        kind = "DROP"
    elif kept and toward and agrees and vrel_max < 0.12:
        kind = "CONTROLLED_CORRECTING_SLIP"
    elif kept and abs(dr) < CORRECT_MM * 1e-3:
        kind = "SECURE_LIKE"
    elif kept and toward:
        kind = "CORRECTING_MOTION"
    elif kept:
        kind = "HELD_WRONG_WAY"
    else:
        kind = "LOSS"
    return {
        "kind": kind,
        "drhx_mm": 1e3 * dr,
        "toward_zero": bool(toward),
        "agrees_g_hx": agrees,
        "contacts_end": contacts_ok(m1),
        "vrel_max": vrel_max,
        "ballistic_escape": bool(ballistic),
        "table": bool(table),
        "both_off_persist_s": both_off_persist(log),
    }


def run_grip_at_angle(sim, gains, snap, theta, tau, desired_sign, g0=None):
    Rh0, m_early = start_from_early(sim, snap)
    log = [m_early]
    rot = rotate_to(sim, gains, Rh0, theta, -18.0, log)
    m_ang = measure(sim, -18.0, Rh0)
    if not rot["reached"] or rot["t_table"] is not None or not contacts_ok(m_ang):
        return {
            "theta": theta,
            "tau": tau,
            "rot": rot,
            "ok_to_slip": False,
            "cls": {"kind": "NO_ANGLE"},
            "log": subsample(log),
        }
    n = int(round(GRIP_S / float(sim.model.opt.timestep)))
    t0 = float(sim.data.time)
    i0 = len(log)
    tick_log(sim, gains, 0.0, 0.0, 0.0, tau, Rh0, log, n, label=f"GRIP_{tau}")
    m1 = log[-1]
    g_hx = float(m_ang["g_h"][0])
    cls = classify_grip(m_ang, m1, log[i0:], desired_sign, g_hx)
    return {
        "theta": theta,
        "tau": tau,
        "rot": {k: rot[k] for k in rot if k != "hold"},
        "ok_to_slip": True,
        "start_angle": slim_m(m_ang),
        "end": slim_m(m1),
        "dt": float(sim.data.time) - t0,
        "g_hx": g_hx,
        "cls": cls,
        "log": subsample(log),
    }


def return_to_nominal(sim, gains, R_nom, Rh0, log, ctl=None):
    """Drive r_des to saved R_nominal with legal omega_y; tau=-18. No object GT."""
    freeze(sim)  # park translation; r_des := current hand, then servo r_des to R_nom
    dt = float(sim.model.opt.timestep)
    t0 = float(sim.data.time)
    timeout = t0 + RETURN_TIMEOUT_S
    t_table = None
    arrived_des = False
    t_des_ok = None
    ori_log = []
    while float(sim.data.time) < timeout:
        if ctl and ctl.get("reset"):
            break
        Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        y_err_deg = float(np.degrees(so3_log(Rd.T @ np.asarray(R_nom, float).reshape(3, 3))[1]))
        omega_y = omega_y_to_nominal(R_nom, Rd)
        err_des = so3_angle_deg(R_nom, Rd)
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        err_hand = so3_angle_deg(R_nom, Rh)
        # 1-DOF ω_y cannot cancel off-axis SO(3) leftovers from tracking; once the
        # legal y-error is small, latch the *desired* matrix to R_nominal (not qpos).
        if abs(y_err_deg) <= RETURN_DES_STOP_DEG or err_des <= RETURN_DES_STOP_DEG:
            omega_y = 0.0
            sim.fsm.r_des = np.asarray(R_nom, float).reshape(3, 3).copy()
            if not arrived_des:
                arrived_des = True
                t_des_ok = float(sim.data.time)
            if err_hand <= RETURN_HAND_STOP_DEG and (float(sim.data.time) - t_des_ok) >= RETURN_CATCH_S:
                break
        tt, _ = tick_log(
            sim, gains, omega_y, 0.0, 0.0, -18.0, Rh0, log, 1, ctl=ctl, label="RETURN"
        )
        if log:
            op = ori_pack(sim, R_nom)
            log[-1]["err_des_deg"] = op["err_des_deg"]
            log[-1]["err_hand_deg"] = op["err_hand_deg"]
            log[-1]["y_err_des_deg"] = op["y_err_des_deg"]
            log[-1]["w_hand"] = op["w_hand"].tolist()
            if (not ori_log) or (log[-1]["t"] - ori_log[-1]["t"] >= 0.020):
                ori_log.append(
                    {
                        "t": log[-1]["t"],
                        "omega_y": float(omega_y),
                        "err_des_deg": op["err_des_deg"],
                        "err_hand_deg": op["err_hand_deg"],
                        "y_err_des_deg": op["y_err_des_deg"],
                        "rhx_mm": 1e3 * float(log[-1]["rh"][0]),
                        "tilt_deg": log[-1]["tilt_deg"],
                        "nL": log[-1]["nL"],
                        "nR": log[-1]["nR"],
                        "Fn_L": log[-1]["Fn_L"],
                        "Fn_R": log[-1]["Fn_R"],
                        "aperture": log[-1]["aperture"],
                    }
                )
        if tt is not None:
            t_table = tt
            break
    n_tail = int(round(RETURN_CATCH_S / dt))
    if t_table is None and arrived_des:
        sim.fsm.r_des = np.asarray(R_nom, float).reshape(3, 3).copy()
        tick_log(sim, gains, 0.0, 0.0, 0.0, -18.0, Rh0, log, n_tail, ctl=ctl, label="RETURN_SETTLE")
    op = ori_pack(sim, R_nom)
    if log:
        log[-1]["err_des_deg"] = op["err_des_deg"]
        log[-1]["err_hand_deg"] = op["err_hand_deg"]
    return {
        "t_table": t_table,
        "dt": float(sim.data.time) - t0,
        "arrived_des": bool(arrived_des),
        "t_des_ok": t_des_ok,
        "end": op,
        "end_measure": slim_m(measure(sim, -18.0, Rh0)),
        "ori_log": ori_log,
    }


def apply_gravity_sequence(sim, gains, snap, plan, ctl=None, from_live=False):
    """Rotate / slip / brake / return. If from_live, current state is already EARLY."""
    if not from_live:
        restore_ballistic(sim, snap)
    R_nominal = np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy()
    R_hand_start = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    freeze(sim)
    Rh0 = R_hand_start.copy()
    z_tgt = float(snap.get("z_tgt", z_tgt_of(sim)))
    log = [measure(sim, -18.0, Rh0)]
    theta = float(plan["theta_deg"])
    tau_s = float(plan["tau_slip"])
    rot = rotate_to(sim, gains, Rh0, theta, -18.0, log, ctl=ctl, label="ROTATE_SECURE")
    events = {
        "rotate": rot,
        "R_nominal": R_nominal,
        "R_hand_start": R_hand_start,
        "ori0": {
            "err_des_vs_nom_deg": so3_angle_deg(R_nominal, R_nominal),
            "err_hand_vs_nom_deg": so3_angle_deg(R_nominal, R_hand_start),
        },
    }
    if rot["t_table"] is not None or not contacts_ok(log[-1]):
        return log, rot["t_table"], events, Rh0
    n_slip = int(round(float(plan.get("slip_s", SLIP_CAP_S)) / float(sim.model.opt.timestep)))
    rh0 = float(log[-1]["rh"][0])
    want = float(plan["desired_drhx_sign"])
    i_slip = len(log)
    t_table, _ = tick_log(
        sim, gains, 0.0, 0.0, 0.0, tau_s, Rh0, log, n_slip, ctl=ctl, label="WEAK_SLIP"
    )
    useful = False
    for m in log[i_slip:]:
        d = float(m["rh"][0]) - rh0
        if (want < 0 and d < -0.002) or (want > 0 and d > 0.002):
            useful = True
            break
        if m["obj_z"] < TABLE_DROP or (m["nL"] == 0 and m["nR"] == 0):
            break
    events["slip_start_rhx_mm"] = 1e3 * rh0
    events["slip_end"] = slim_m(log[-1])
    events["slip_drhx_mm"] = 1e3 * (float(log[-1]["rh"][0]) - rh0)
    events["useful_before_cap"] = useful
    if log[-1]["obj_z"] < TABLE_DROP:
        return log, log[-1]["t"], events, Rh0
    n_br = int(round(float(plan.get("brake_s", BRAKE_S)) / float(sim.model.opt.timestep)))
    tick_log(sim, gains, 0.0, 0.0, 0.0, -18.0, Rh0, log, n_br, ctl=ctl, label="BRAKE")
    events["brake_end"] = slim_m(log[-1])
    ret = return_to_nominal(sim, gains, R_nominal, Rh0, log, ctl=ctl)
    events["return"] = {
        "dt": ret["dt"],
        "arrived_des": ret["arrived_des"],
        "err_des_deg": ret["end"]["err_des_deg"],
        "err_hand_deg": ret["end"]["err_hand_deg"],
        "y_err_des_deg": ret["end"]["y_err_des_deg"],
        "end_measure": ret["end_measure"],
        "ori_log": ret["ori_log"],
        "t_table": ret["t_table"],
    }
    events["return_end"] = ret["end_measure"]
    events["return_deg"] = signed_rot_about_y_deg(Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float))
    if ret["t_table"] is not None:
        return log, ret["t_table"], events, Rh0
    t_table = continue_zero(sim, gains, HOLD_S + 4.0, log, Rh0, z_tgt, ctl)
    events["end"] = slim_m(log[-1])
    events["end_ori"] = ori_pack(sim, R_nominal)
    return log, t_table, events, Rh0


def write_report(doc: dict) -> None:
    p = OUT / "LARGE_ANGLE_GRAVITY_TRANSFER.md"
    a = []
    ap = a.append
    g = doc["geometry"]
    ap("# CENTER-6 large-angle gravity transfer")
    ap("")
    ap("Contact-preserving **authority** test. Not a learned policy, not an optimized heuristic.")
    ap("")
    ap("Prior ±13° wrist pulses are **small-angle probes that did not produce correction**.")
    ap("They do **not** show that the wrist/gravity mechanism fails to transfer.")
    ap("Grip `τ=-3` dropping at the original orientation does **not** imply the same at a large-angle state.")
    ap("")
    ap("## 1. EARLY state geometry")
    ap("")
    ap(f"- snapshot `raw/snap_early.pkl` at t={g['t']:.3f} s")
    ap(f"- r_h = [{g['rh_mm'][0]:.2f}, {g['rh_mm'][1]:.2f}, {g['rh_mm'][2]:.2f}] mm")
    ap(f"- g_h = [{g['g_h'][0]:.3f}, {g['g_h'][1]:.3f}, {g['g_h'][2]:.3f}] m/s²")
    ap(f"- nL/nR = {g['nL']}/{g['nR']}; tilt {g['tilt_deg']:.2f}°")
    ap(f"- ZERO restore table {doc['restore']['table']}; max rh.x err {doc['restore'].get('max_err_mm')} mm")
    ap("")
    ap("## 2. Required correction sign")
    ap("")
    ap(f"r_h.x = **{g['rh_mm'][0]:+.2f} mm** so desired Δr_h.x has sign **{g['desired_drhx_sign']:+.0f}**.")
    ap("Gravity-driven sliding in hand-x should match sign(g_h.x).")
    ap("Therefore we want **sign(g_h.x) = desired Δr_h.x sign** after reorientation.")
    ap("")
    ap("## 3. g_h versus wrist angle")
    ap("")
    ap("| θ cmd | θ act | g_h.x pred | g_h.x meas | g_h.z meas | nL/nR | r_h.x mm | Fn L/R | rho | hold |")
    ap("|---:|---:|---:|---:|---:|---|---:|---|---:|---|")
    for r in doc["secure_sweep"]:
        h = r["hold"]
        ap(
            f"| {r['target_deg']:+.0f} | {r['actual_deg']:+.1f} | {r['g_hx_pred']:.2f} | "
            f"{h['g_h'][0]:.2f} | {h['g_h'][2]:.2f} | {h['nL']}/{h['nR']} | "
            f"{1e3*h['rh'][0]:.2f} | {h['Fn_L']:.1f}/{h['Fn_R']:.1f} | "
            f"{h['rho_max']} | {'OK' if r['secure_ok'] else 'FAIL'} |"
        )
    ap("")
    ap(f"Correcting-sign geometry: g_h.x {'<' if g['desired_drhx_sign']<0 else '>'} 0.")
    ap("")
    ap("## 4. Maximum secure rotation")
    ap("")
    ap(f"- max |θ| held bilateral at τ=-18: **{doc['max_secure_deg']} deg**")
    ap(f"- first failure: {doc.get('first_fail')}")
    ap("- During secure rotation the cylinder is expected to **rigidly follow**. That is not recovery.")
    ap("")
    ap("## 5. Grip regimes at large angle")
    ap("")
    ap("| θ | τ | kind | Δr_h.x mm | keep | agrees g_h.x | |v_rel|max |")
    ap("|---:|---:|---|---:|---|---|---:|")
    for r in doc["grip"]:
        c = r["cls"]
        ap(
            f"| {r['theta']:+.0f} | {r['tau']:.0f} | {c.get('kind')} | {c.get('drhx_mm', float('nan')):.2f} | "
            f"{c.get('contacts_end')} | {c.get('agrees_g_hx')} | {c.get('vrel_max', float('nan')):.3f} |"
        )
    ap("")
    ap("## 6. Controlled-slip evidence")
    ap("")
    ap(doc["slip_text"])
    ap("")
    ap("## 7. Gravity causality")
    ap("")
    ap(doc["causality_text"])
    ap("")
    ap("## 8. Secure-brake evidence")
    ap("")
    ap(doc["brake_text"])
    ap("")
    ap("## 9. Return-to-nominal evidence")
    ap("")
    ap(doc["return_text"])
    ap("")
    ap("## 10. Task-level existence proof")
    ap("")
    ap(doc["task_text"])
    ap("")
    ap("## 11. Interactive viewer commands")
    ap("")
    ap("No MP4. Camera once; mouse; SPACE pause; `[` `]` speed; R restart.")
    ap("")
    ap("```text")
    ap("python training/demo_ballistic_recovery.py --mode zero")
    ap("python training/demo_ballistic_recovery.py --mode gravity_large_angle")
    ap("```")
    ap("")
    ap("ZERO is the frozen CENTER-6 trajectory. Recovery mode runs only if a plan exists;")
    ap("otherwise it rotates securely from EARLY so the large-angle attempt is visible.")
    ap("")
    ap("## 12. Unresolved limitations")
    ap("")
    for line in doc["limits"]:
        ap(f"- {line}")
    ap("")
    ap("**No SAC, recatch, reward/obs change, impact retune, or MP4.**")
    p.write_text("\n".join(a), encoding="utf-8")
    print("wrote", p, flush=True)


def headless_large_angle():
    LA_RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(Path(__file__).resolve().parents[1] / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    snap = load_early()
    orig = load_zero_rows()
    sim, _ = make_sim()
    restore = restore_audit(sim, gains, snap, "EARLY", orig, float(snap["time"]))
    print("ZERO restore table", restore.get("table"), "err_mm", restore.get("max_err_mm"), flush=True)

    sim, _ = make_sim()
    Rh0, m0 = start_from_early(sim, snap)
    desired = -float(np.sign(m0["rh"][0]))
    if desired == 0:
        desired = -1.0
    geometry = {
        "t": m0["t"],
        "rh_mm": (1e3 * np.asarray(m0["rh"])).tolist(),
        "g_h": np.asarray(m0["g_h"], float).tolist(),
        "nL": m0["nL"],
        "nR": m0["nR"],
        "tilt_deg": m0["tilt_deg"],
        "desired_drhx_sign": desired,
        "wrist0_deg": 0.0,
    }
    print(
        f"EARLY rhx={geometry['rh_mm'][0]:.2f} mm g_h={geometry['g_h']} desired_sign={desired}",
        flush=True,
    )

    sweep = []
    max_secure = 0.0
    first_fail = None
    targets = [0.0]
    for a in ANGLES_DEG[1:]:
        targets.extend([a, -a])
    for th in targets:
        sim, _ = make_sim()
        Rh0, m_early = start_from_early(sim, snap)
        log = [m_early]
        rot = rotate_to(sim, gains, Rh0, th, -18.0, log)
        h = rot["hold"]
        secure_ok = bool(
            rot["reached"]
            and rot["t_table"] is None
            and contacts_ok(h)
            and both_off_persist(log) < 0.020
        )
        row = {
            "target_deg": th,
            "actual_deg": rot["actual_deg"],
            "reached": rot["reached"],
            "secure_ok": secure_ok,
            "g_hx_pred": float(pred_g_h(m0["g_h"], th)[0]),
            "hold": slim_m(h),
            "t_table": rot["t_table"],
            "lost_t": rot["lost_t"],
            "dt_rot": rot["dt_rot"],
            "log": subsample(log),
        }
        sweep.append(row)
        print(
            f"secure θ={th:+.0f} act={rot['actual_deg']:+.1f} ok={secure_ok} "
            f"ghx={h['g_h'][0]:+.2f} rhx={1e3*h['rh'][0]:.2f} n={h['nL']}/{h['nR']}",
            flush=True,
        )
        if secure_ok:
            max_secure = max(max_secure, abs(float(rot["actual_deg"])))
        elif first_fail is None and abs(th) > 0.5:
            first_fail = {
                "target": th,
                "actual": rot["actual_deg"],
                "reached": rot["reached"],
                "nL": h["nL"],
                "nR": h["nR"],
                "t_table": rot["t_table"],
            }

    dump(LA_RAW / "secure_sweep.json", sweep)

    selected = []
    for r in sweep:
        if not r["secure_ok"]:
            continue
        ghx = float(r["hold"]["g_h"][0])
        if abs(ghx) < G_HX_SUBSTANTIAL:
            continue
        if np.sign(ghx) != np.sign(desired):
            continue
        selected.append(r)
    # compact 2–3: prefer ~90 and beyond, skip 0
    selected.sort(key=lambda r: -abs(r["actual_deg"]))
    pick = []
    for r in selected:
        if len(pick) >= 3:
            break
        if pick and abs(abs(r["target_deg"]) - abs(pick[-1]["target_deg"])) < 8:
            continue
        pick.append(r)
    print("selected angles", [r["target_deg"] for r in pick], flush=True)

    grip_rows = []
    slip_hits = []
    if pick:
        for r in pick:
            th = r["target_deg"]
            for tau in GRIP_TAUS:
                sim, _ = make_sim()
                gr = run_grip_at_angle(sim, gains, snap, th, tau, desired)
                grip_rows.append(gr)
                c = gr["cls"]
                print(
                    f"grip θ={th:+.0f} τ={tau:.0f} {c.get('kind')} drhx={c.get('drhx_mm')} "
                    f"keep={c.get('contacts_end')}",
                    flush=True,
                )
                if c.get("kind") in ("CONTROLLED_CORRECTING_SLIP", "CORRECTING_MOTION"):
                    slip_hits.append(gr)
        dump(LA_RAW / "grip.json", grip_rows)
    else:
        print("no large-angle secure state with substantial correcting g_h.x", flush=True)

    causality = []
    g0_run = None
    brake_run = None
    return_run = None
    task = None
    plan = {"none": True}

    controlled = [g for g in slip_hits if g["cls"]["kind"] == "CONTROLLED_CORRECTING_SLIP"]
    use = controlled[0] if controlled else None

    if use is not None:
        th = float(use["theta"])
        tau = float(use["tau"])
        sim, _ = make_sim()
        opp = run_grip_at_angle(sim, gains, snap, -th, tau, desired)
        causality.append({"name": "plus", **{k: use[k] for k in use if k != "log"}})
        causality.append({"name": "minus", **{k: opp[k] for k in opp if k != "log"}})
        if not opp.get("ok_to_slip"):
            negs = [r["target_deg"] for r in sweep if r["secure_ok"] and r["target_deg"] < -1]
            alt = min(negs) if negs else -90.0
            sim, _ = make_sim()
            opp2 = run_grip_at_angle(sim, gains, snap, alt, tau, desired)
            causality.append({"name": "minus_secure_alt", **{k: opp2[k] for k in opp2 if k != "log"}})
        # 0g at same +theta
        sim, _ = make_sim()
        g_save = np.array(sim.model.opt.gravity, float).copy()
        Rh0, m_early = start_from_early(sim, snap)
        log = [m_early]
        rotate_to(sim, gains, Rh0, th, -18.0, log)
        sim.model.opt.gravity[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        m_ang = measure(sim, -18.0, Rh0)
        n = int(round(GRIP_S / float(sim.model.opt.timestep)))
        i0 = len(log)
        tick_log(sim, gains, 0.0, 0.0, 0.0, tau, Rh0, log, n, label="SLIP_0G")
        m1 = log[-1]
        cls0 = classify_grip(m_ang, m1, log[i0:], desired, float(m_ang["g_h"][0]))
        g0_run = {"theta": th, "tau": tau, "cls": cls0, "drhx_mm": cls0["drhx_mm"], "end": slim_m(m1)}
        sim.model.opt.gravity[:] = g_save
        dump(LA_RAW / "causality.json", {"plus_minus": causality, "zero_g": g0_run})
        print("0g drhx_mm", cls0["drhx_mm"], "kind", cls0["kind"], flush=True)

        plan = {
            "none": False,
            "theta_deg": th,
            "tau_slip": tau,
            "slip_s": GRIP_S,
            "brake_s": BRAKE_S,
            "desired_drhx_sign": desired,
            "omega_y": OMEGA,
            "stage": "EARLY",
        }
        sim, _ = make_sim()
        log, t_table, events, _ = apply_gravity_sequence(sim, gains, snap, plan)
        brake_run = {
            "table": t_table,
            "events": {k: events[k] for k in events},
            "rhx_after_slip_mm": 1e3 * float(events["slip_end"]["rh"][0]),
            "rhx_after_brake_mm": 1e3 * float(events["brake_end"]["rh"][0]),
            "n_brake": [events["brake_end"]["nL"], events["brake_end"]["nR"]],
        }
        return_run = {
            "return_deg": events.get("return_deg"),
            "rhx_after_return_mm": 1e3 * float(events["return_end"]["rh"][0]),
            "n_return": [events["return_end"]["nL"], events["return_end"]["nR"]],
            "table": t_table,
        }
        dump(LA_RAW / "brake_return.json", {"brake": brake_run, "ret": return_run})
        print("brake/return table", t_table, "return_deg", events.get("return_deg"), flush=True)

        if (
            t_table is None
            and contacts_ok(events["brake_end"])
            and abs(float(events.get("return_deg", 99))) < 8.0
        ):
            sim, _ = make_sim()
            t_go = float(snap["time"])
            ep_pre, _ = run_center6_until(sim, gains, t_go, label="DRIFT")
            log, t_table, events, _ = apply_gravity_sequence(
                sim, gains, snap, plan, from_live=True
            )
            mend = log[-1]
            task = {
                "intervene_t": t_go,
                "table": t_table,
                "t_end": mend["t"],
                "end_rh_mm": (1e3 * np.asarray(mend["rh"])).tolist(),
                "end_nL": mend["nL"],
                "end_nR": mend["nR"],
                "end_obj_z": mend["obj_z"],
                "held_like": bool(
                    t_table is None
                    and mend["nL"] > 0
                    and mend["nR"] > 0
                    and mend["obj_z"] > 0.55
                    and mend["t"] >= t_go + HOLD_S
                ),
                "plan": plan,
                "t_lift_done_zero": ep_pre.get("t_lift_done"),
            }
            dump(LA_RAW / "task_level.json", task)
            print("task-level held_like", task["held_like"], "table", t_table, flush=True)
        else:
            print("skip task-level: brake/return did not retain grasp", flush=True)

    dump(RAW / "large_angle_plan.json", plan)

    if not pick:
        slip_text = (
            "Secure rotation never produced a held orientation with substantial "
            "correcting g_h.x. Grip-weakening for controlled slip was **not** run. STOP."
        )
    elif use is None:
        slip_text = (
            "Large-angle secure rotation succeeded at the selected angles, but no "
            "tested τ produced controlled correcting slip (bilateral contact + r_h.x "
            "toward zero + motion consistent with g_h.x without ballistic escape). STOP."
        )
    else:
        slip_text = (
            f"Candidate at θ={use['theta']:+.0f}°, τ={use['tau']:.0f}: "
            f"{use['cls']['kind']}, Δr_h.x={use['cls']['drhx_mm']:.2f} mm, "
            f"|v_rel|_max={use['cls']['vrel_max']:.3f} m/s, "
            f"agrees_g_hx={use['cls']['agrees_g_hx']}."
        )
        if use["cls"]["kind"] != "CONTROLLED_CORRECTING_SLIP":
            slip_text += " Not all controlled-slip criteria were met."

    if use is None:
        causality_text = "Not run (no correcting-slip candidate)."
        brake_text = "Not run."
        return_text = "Not run."
        task_text = "Not run."
    else:
        pm = ""
        if len(causality) >= 2:
            c0, c1 = causality[0]["cls"], causality[1]["cls"]
            dp = c0.get("drhx_mm")
            dm = c1.get("drhx_mm")
            pm = (
                f" +θ={causality[0].get('theta')} {c0.get('kind')} Δr_h.x={dp}; "
                f"−θ={causality[1].get('theta')} {c1.get('kind')} Δr_h.x={dm}."
            )
            if not causality[1].get("ok_to_slip"):
                pm += (
                    " Opposite large angle is **not securely reachable**: gravity of the "
                    "wrong sign dumps the cylinder during/after secure rotation."
                )
            elif dp is not None and dm is not None:
                if np.sign(dp) == np.sign(dm) and abs(dp) > 0.5 and abs(dm) > 0.5:
                    pm += " Signs did **not** reverse with θ."
                elif np.sign(dp) != np.sign(dm):
                    pm += " Relative x motion **reversed** with wrist sign."
            if len(causality) >= 3:
                c2 = causality[2]["cls"]
                pm += (
                    f" Secure opposite alt θ={causality[2].get('theta')} "
                    f"{c2.get('kind')} Δr_h.x={c2.get('drhx_mm')} mm."
                )
        zg = ""
        if g0_run is not None:
            zg = f" 0g Δr_h.x={g0_run['drhx_mm']:.2f} mm ({g0_run['cls']['kind']})."
            if abs(g0_run["drhx_mm"]) < 0.5 * abs(float(use["cls"]["drhx_mm"])) + 0.3:
                zg += " Correction reduced under 0g (gravity-consistent)."
            else:
                zg += " Correction did **not** clearly vanish in 0g."
        causality_text = (pm + zg).strip() or "See raw/large_angle/causality.json."
        if brake_run is None:
            brake_text = "Not run."
            return_text = "Not run."
        else:
            brake_text = (
                f"After slip, τ→−18 at fixed wrist: r_h.x {brake_run['rhx_after_slip_mm']:.2f} → "
                f"{brake_run['rhx_after_brake_mm']:.2f} mm; n={brake_run['n_brake']}; "
                f"table={brake_run['table']}."
            )
            return_text = (
                f"Return wrist at τ=-18: residual angle {return_run['return_deg']:.1f}°; "
                f"r_h.x={return_run['rhx_after_return_mm']:.2f} mm; n={return_run['n_return']}."
            )
        if task is None:
            task_text = "Not run (brake/return did not establish a retainable correction)."
        else:
            task_text = (
                "One construction: CENTER-6 → EARLY visible drift → secure large-angle "
                f"rotation (θ={plan['theta_deg']:.0f}°) → τ={plan['tau_slip']:.0f} slip → "
                "τ=-18 brake → return wrist → resume lift/hold. "
                f"held_like={task['held_like']} table={task['table']} "
                f"end t={task['t_end']:.2f}s end r_h.x={task['end_rh_mm'][0]:.2f} mm "
                f"n={task['end_nL']}/{task['end_nR']} obj_z={task['end_obj_z']:.3f} m. "
                "ZERO on this frozen case hits the table at **6.132 s**. "
                "This is an **authority proof**, not a recenter-to-zero or an optimized heuristic. "
                "Residual lateral offset may remain; the claim is arrest of the progressive escape."
            )

    limits = [
        "Angle set is compact (0, ±30, ±60, ±90, ±105, ±120). No fine search.",
        "Grip set is compact (−18, −8, −5, −4, −3, −2). No τ optimizer.",
        "Wrist rate is the legal recovery4d bound ω_y=3 rad/s, not tuned.",
        "Secure follow during rotation is not counted as recovery.",
        "0g is diagnostic only.",
        "No UPPER/LOWER, SAC, recatch, or MP4.",
    ]
    if max_secure < 80:
        limits.append("Panda/workspace may limit >90° wrist about hand-y from this lift pose.")

    doc = {
        "geometry": geometry,
        "restore": restore,
        "secure_sweep": [{k: v for k, v in r.items() if k != "log"} for r in sweep],
        "max_secure_deg": max_secure,
        "first_fail": first_fail,
        "selected": [r["target_deg"] for r in pick],
        "grip": [{k: v for k, v in r.items() if k != "log"} for r in grip_rows],
        "slip_text": slip_text,
        "causality_text": causality_text,
        "brake_text": brake_text,
        "return_text": return_text,
        "task_text": task_text,
        "limits": limits,
        "plan": plan,
    }
    dump(LA_RAW / "summary.json", doc)
    write_report(doc)
    return plan, doc


def append_return_audit(audit: dict) -> None:
    p = OUT / "LARGE_ANGLE_GRAVITY_TRANSFER.md"
    text = p.read_text(encoding="utf-8") if p.is_file() else ""
    if "## Return-to-Nominal Orientation Audit" in text:
        pre = text.split("## Return-to-Nominal Orientation Audit")[0].rstrip() + "\n\n"
    else:
        pre = text.rstrip() + "\n\n"
    a = [pre, "## Return-to-Nominal Orientation Audit", ""]
    ap = a.append
    ap("This section is appended after the validated large-angle mechanism. Earlier sections are not rewritten.")
    ap("")
    ap("### Previous residual orientation error")
    ap("")
    ap("From the construction that the user visually validated (`raw/large_angle/brake_return.json`):")
    ap("")
    ap("- RETURN stop metric was `signed_rot_about_y_deg(R_hand_start, R_hand)`; it reported **+0.65°** at return-end.")
    ap("- That metric is a **single-axis atan2(R[0,2], R[0,0])**, not the SO(3) geodesic.")
    ap("- `r_des` vs `R_nominal` was **not** logged.")
    ap("- After the subsequent lift/hold, `wrist_deg` was **−9.64°** and `g_h.x = +1.65 m/s²` (atan2(1.65, 9.67) ≈ **9.7°**). That matches the visible leftover wrist tilt.")
    ap("")
    ap("### Cause in implementation")
    ap("")
    ap("RETURN did **not** use a fixed duration, but it was equivalent in effect to an open-loop unwind:")
    ap("")
    ap("- direction: `sign = −1` if `signed_rot_about_y(R_hand_start, R_hand) > 0` else `+1`")
    ap("- rate: legal bound `ω_y = ±3 rad/s` (constant, not reduced near the target)")
    ap("- timeout: `|θ_cmd| / 3 + 1.5 s` with `θ_cmd = +120°`")
    ap("- stop: `|signed_rot_about_y(R_hand_start, R_hand)| < 0.8°`")
    ap("")
    ap("`tick_4d` / `map_recovery4d` integrates **`r_des`** at 3 rad/s about current `r_des` y. The hand lags. Stopping on the **hand** angle therefore leaves **`r_des` past the nominal pose**. `omega` then goes to 0, so `r_des` stays overshot. `continue_zero` / lift holds that `r_des` (`w_cmd = 0`), and the hand tracks the leftover tilt.")
    ap("")
    ap("`freeze()` at RETURN start parked `r_des := R_hand` (good for aligning the target with the arm) but never drove `r_des` back to the **saved EARLY `r_des`**.")
    ap("")
    ap("### Corrected return logic")
    ap("")
    ap("Frozen mechanism unchanged: CENTER-6, EARLY, +120° secure, `τ=-5` for 0.70 s, `τ=-18` brake 0.40 s.")
    ap("")
    ap("Before recovery: `R_nominal = r_des` at EARLY (FSM lift / grasp orientation); also log `R_hand_start`.")
    ap("RETURN: `τ=-18`; legal `ω_y` P-control on the body-y component of `log(R_des^T R_nominal)`, saturated at 3 rad/s; `ω_y → 0` when `angle(R_nominal, R_des) ≤ 0.35°`; latch `r_des := R_nominal` (controller target only, not `qpos`); short catch-up with `ω_y=0` so the hand can track. No object GT. No `v_x`.")
    ap("")
    e = audit["return"]
    ap("### After the fix (exact same construction, new RETURN only)")
    ap("")
    ap(f"- weak-slip Δr_h.x: **{audit['slip_drhx_mm']:.2f} mm** (previous construction **−1.62 mm**)")
    ap(f"- contacts at slip end nL/nR: {audit['slip_n']}")
    ap(f"- contacts at return end nL/nR: {audit['return_n']}")
    ap(f"- `angle(R_nominal, R_des)` at RETURN end: **{e['err_des_deg']:.3f}°**")
    ap(f"- `angle(R_nominal, R_hand)` at RETURN end: **{e['err_hand_deg']:.3f}°**")
    ap(f"- signed-y(R_hand_start, R_hand) leftover (old metric): {audit.get('return_signed_y_deg'):.2f}°")
    ap(f"- final r_h.x after hold: **{audit['end_rhx_mm']:.2f} mm**")
    ap(f"- table: {audit['table']}; held_like={audit['held_like']}")
    ap(f"- end nL/nR: {audit['end_n']}; obj_z={audit['end_obj_z']:.3f} m; t_end={audit['t_end']:.2f} s")
    if audit.get("end_ori"):
        ap(f"- after long hold: `angle(R_nominal, R_des)`={audit['end_ori']['err_des_deg']:.3f}°, `angle(R_nominal, R_hand)`={audit['end_ori']['err_hand_deg']:.3f}°")
    ap("")
    ap("Weak-slip before RETURN is the same recipe (duration/τ/angle). Millimetre Δr_h.x is compared above; it is not re-optimized.")
    ap("")
    ap("Viewer: `python training/demo_ballistic_recovery.py --mode gravity_large_angle`")
    ap("")
    p.write_text("".join(line if line.endswith("\n") else line + "\n" for line in a), encoding="utf-8")
    print("appended return audit to", p, flush=True)


def headless_return_audit():
    LA_RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(Path(__file__).resolve().parents[1] / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    snap = load_early()
    plan = json.loads((RAW / "large_angle_plan.json").read_text(encoding="utf-8"))
    sim, _ = make_sim()
    log, t_table, events, _ = apply_gravity_sequence(sim, gains, snap, plan, from_live=False)
    mend = log[-1]
    held = bool(
        t_table is None
        and mend["nL"] > 0
        and mend["nR"] > 0
        and mend["obj_z"] > 0.55
        and mend["t"] >= float(snap["time"]) + HOLD_S
    )
    ret = events["return"]
    audit = {
        "slip_drhx_mm": events["slip_drhx_mm"],
        "slip_n": [events["slip_end"]["nL"], events["slip_end"]["nR"]],
        "return_n": [events["return_end"]["nL"], events["return_end"]["nR"]],
        "return": ret,
        "return_signed_y_deg": events.get("return_deg"),
        "end_rhx_mm": 1e3 * float(mend["rh"][0]),
        "table": t_table,
        "held_like": held,
        "end_n": [mend["nL"], mend["nR"]],
        "end_obj_z": mend["obj_z"],
        "t_end": mend["t"],
        "end_ori": {
            "err_des_deg": events["end_ori"]["err_des_deg"],
            "err_hand_deg": events["end_ori"]["err_hand_deg"],
        }
        if events.get("end_ori")
        else None,
        "ori0": events["ori0"],
    }
    dump(LA_RAW / "return_audit.json", audit)
    print(
        f"RETURN audit slip_drhx={audit['slip_drhx_mm']:.2f} mm "
        f"des_err={ret['err_des_deg']:.3f} deg hand_err={ret['err_hand_deg']:.3f} deg "
        f"held={held} table={t_table}",
        flush=True,
    )
    append_return_audit(audit)
    return audit


def interactive_large_angle(plan_path=None):
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(Path(__file__).resolve().parents[1] / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    plan_p = Path(plan_path) if plan_path else RAW / "large_angle_plan.json"
    plan = json.loads(plan_p.read_text(encoding="utf-8")) if plan_p.is_file() else {"none": True}
    snap = load_early()
    t_go = float(snap["time"])
    ctl = {"pause": False, "reset": False, "abort": False, "speed": 0.40, "overlay": {}}

    def sync():
        vwr = ctl.get("viewer")
        if vwr is None:
            return
        o = ctl.get("overlay") or {}
        _viewer_overlay(
            vwr,
            [
                ("CASE", str(o.get("case", "GRAVITY_LARGE_ANGLE"))),
                ("t", f"{float(o.get('t', sim.data.time)):.3f} s"),
                ("PHASE", str(o.get("phase", ""))),
                ("nL / nR", f"{o.get('nL', '-')}/{o.get('nR', '-')}"),
                ("keys", "SPACE pause  R restart  [ ] speed"),
            ],
        )
        vwr.sync()
        dt = 0.002
        spd = max(float(ctl.get("speed", 0.4)), 0.05)
        now = time.perf_counter()
        tgt = ctl.get("_wall")
        if tgt is None:
            ctl["_wall"] = now + dt / spd
        else:
            sl = tgt - now
            if sl > 0:
                time.sleep(min(sl, 0.05))
            ctl["_wall"] = max(tgt, time.perf_counter()) + dt / spd
        while ctl.get("pause") and vwr.is_running() and not ctl.get("reset"):
            vwr.sync()
            time.sleep(0.02)

    ctl["sync"] = sync

    def on_key(kc):
        k = int(kc)
        if k == 32:
            ctl["pause"] = not ctl["pause"]
        elif k in (ord("R"), ord("r")):
            ctl["reset"] = True
        elif k == ord("["):
            ctl["speed"] = max(0.08, float(ctl["speed"]) * 0.7)
        elif k == ord("]"):
            ctl["speed"] = min(2.5, float(ctl["speed"]) / 0.7)

    sim, _ = make_sim()
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        ctl["viewer"] = vwr
        apply_camera_preset(vwr, CAM)
        while vwr.is_running():
            ctl["reset"] = False
            ctl["abort"] = False
            ctl["_wall"] = None
            ctl["sync"] = sync
            sim.reset()

            def wrap_stop():
                sync()
                if "GO" in ctl.get("snaps", {}):
                    ctl["abort"] = True

            ctl["sync"] = wrap_stop
            ctl["capture_at"] = {"GO": t_go}
            ctl["snaps"] = {}
            ctl["make_snap"] = make_snap
            run_episode(sim, gains, V_HIT, SITE, ctl=ctl, label="DRIFT")
            if ctl.get("reset") or not vwr.is_running():
                if ctl.get("reset"):
                    continue
                break
            ctl["sync"] = sync
            if plan.get("none"):
                freeze(sim)
                Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
                log = []
                rotate_to(sim, gains, Rh0, 90.0, -18.0, log, ctl=ctl, label="ROTATE_NO_PLAN")
            else:
                apply_gravity_sequence(sim, gains, snap, plan, ctl=ctl, from_live=True)
            if ctl.get("reset"):
                continue
            while vwr.is_running() and not ctl.get("reset"):
                sync()
                time.sleep(0.03)
            if not ctl.get("reset"):
                break
