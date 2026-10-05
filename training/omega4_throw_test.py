"""Diagnostic omega_y=4 throw/recatch. Does not change RECOVERY4D_W_HY_MAX."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.bias import arm_bias
from controllers.jacobian_controller import (
    gains_from_cfg,
    jacobian_6d,
    ori_error_deg,
    rotation_error,
)
from controllers.residual import (
    RECOVERY4D_TAU_OPEN,
    RECOVERY4D_V_HX_MAX,
    RECOVERY4D_V_Z_MAX,
    RECOVERY4D_W_HY_MAX,
    map_recovery4d,
)
from envs.config_util import load_yaml, merge_sim_config
from envs.deterioration import body_twist
from training.demo_airborne_recapture import a3_of_tau
from training.demo_airborne_recovery import copy_masks, disable_object_table_only, prepare_high_parent
from training.demo_airborne_recapture import make_parent_sim
from training.demo_dynamic_recatch import (
    BOTH_OFF_MIN,
    HOLD_S,
    OUT,
    RAW,
    TAU_OPEN,
    TAU_SEC,
    append_log,
    dump,
    measure,
    overlay,
    restore,
    save_npz,
    slim_row,
)
from training.demo_teleport_recovery_state import CYL_MASS, PAIR_MU, signed_rot_about_y_deg
from training.grav_reposition_v2_viz import apply_camera_preset
from training.map_ballistic_disturbance import CAM, _viewer_overlay
from training.replay_core import freeze, tick_vw

OMEGA3 = float(RECOVERY4D_W_HY_MAX)
OMEGA4 = 4.0
MAX_DEG = 125.0
HOLD_S_CATCH = 2.0
PLAN_PATH = RAW / "omega4_plan.json"
REPORT = OUT / "OMEGA4_THROW_TEST.md"


def tick_omega_override(sim, gains, omega_y, v_x, v_z, tau):
    """Same 4D v/tau map; omega_y is NOT clipped to RECOVERY4D_W_HY_MAX."""
    v_x = float(np.clip(v_x, -RECOVERY4D_V_HX_MAX, RECOVERY4D_V_HX_MAX))
    v_z = float(np.clip(v_z, -RECOVERY4D_V_Z_MAX, RECOVERY4D_V_Z_MAX))
    a1 = v_x / RECOVERY4D_V_HX_MAX
    a2 = v_z / RECOVERY4D_V_Z_MAX
    a3 = float(np.clip(a3_of_tau(tau, float(RECOVERY4D_TAU_OPEN)), -1.0, 1.0))
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    cmd = map_recovery4d(np.array([0.0, a1, a2, a3]), sim.fsm.r_des, Rh)
    r_d = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    w_world = r_d @ np.array([0.0, float(omega_y), 0.0])
    tick_vw(sim, cmd["v_world"], w_world, cmd["tau"], gains)
    cmd["w_hy"] = float(omega_y)
    cmd["w_world"] = w_world
    cmd["omega_override"] = True
    cmd["legal_w_hy_max_unchanged"] = OMEGA3
    return cmd


def enrich(sim, m: dict, gains: dict) -> dict:
    ids = sim.ids
    r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    r = np.array(sim.data.xmat[ids.hand_body].reshape(3, 3), float)
    p = np.array(sim.data.xpos[ids.hand_body], float)
    p_des = np.asarray(sim.fsm.p_des, float)
    wh = np.asarray(m["w_hand"], float)
    y_des = r_des[:, 1]
    qdot = np.array(sim.data.qvel[ids.arm_dof], float)
    q = np.array(sim.data.qpos[ids.arm_jnt], float)
    tau_a = np.array(sim.data.ctrl[:7], float)
    lim = np.array(ids.ctrl_high[:7], float)
    j = jacobian_6d(sim.model, sim.data, ids)
    e_p = p_des - p
    e_o = rotation_error(r_des, r)
    e = np.concatenate([e_p, e_o])
    xdot = j @ qdot
    vd = np.asarray(sim.fsm.v_cmd, float).reshape(3)
    wd = np.asarray(sim.fsm.w_cmd, float).reshape(3)
    kp = np.array([gains["kp_pos"]] * 3 + [gains["kp_ori"]] * 3)
    kd = np.array([gains["kd_pos"]] * 3 + [gains["kd_ori"]] * 3)
    wrench = kp * e + kd * (np.concatenate([vd, wd]) - xdot)
    tau_u = j.T @ wrench + arm_bias(sim.model, sim.data, ids)
    q_ref = ids.home_qpos[ids.arm_jnt]
    lam = 1e-3
    j_pinv = j.T @ np.linalg.inv(j @ j.T + lam * np.eye(6))
    nproj = np.eye(7) - j_pinv @ j
    tau_u = tau_u + nproj @ (gains.get("kp_null", 4.0) * (q_ref - q) - gains.get("kd_null", 0.8) * qdot)
    sat = np.abs(tau_u) >= (np.abs(lim) - 1e-6)
    jnt_range = np.array(sim.model.jnt_range[ids.arm_jnt], float)
    mid = 0.5 * (jnt_range[:, 0] + jnt_range[:, 1])
    half = 0.5 * (jnt_range[:, 1] - jnt_range[:, 0])
    prox = np.abs(q - mid) / np.maximum(half, 1e-9)
    m["omega_axis"] = float(np.dot(wh, y_des))
    m["w_hand_norm"] = float(np.linalg.norm(wh))
    m["ori_err_deg"] = float(ori_error_deg(r_des, r))
    m["p_err_norm"] = float(np.linalg.norm(e_p))
    m["q"] = q.tolist()
    m["qdot"] = qdot.tolist()
    m["tau_unclipped"] = tau_u.tolist()
    m["tau_applied"] = tau_a.tolist()
    m["tau_util"] = (np.abs(tau_u) / np.maximum(np.abs(lim), 1e-9)).tolist()
    m["any_sat"] = bool(np.any(sat))
    m["joint_limit_prox_max"] = float(np.max(prox))
    m["J_cond"] = float(np.linalg.cond(j))
    m["mass"] = float(sim.model.body_mass[ids.object_body])
    return m


def step_ovr(sim, gains, omega_y, v_x, v_z, tau, log, Rh0, label, ev=None, ctl=None):
    cmd = tick_omega_override(sim, gains, omega_y, v_x, v_z, tau)
    m = measure(sim, omega_y=omega_y, v_x=v_x, v_z=v_z, tau=tau, label=label, Rh0=Rh0)
    m["w_cmd_world"] = np.asarray(cmd["w_world"], float).tolist()
    enrich(sim, m, gains)
    append_log(log, m)
    if ev is not None:
        from training.demo_airborne_recovery import object_floor_hit

        if object_floor_hit(sim) and "FLOOR" not in ev:
            ev["FLOOR"] = m["t"]
            overlay(ctl, m, label)
            return m, True
    overlay(ctl, m, label)
    if ctl and ctl.get("reset"):
        return m, True
    return m, False


def cruise_omega(log, cmd):
    rows = [r for r in log if abs(float(r.get("omega_y_cmd", 0.0))) > 0.5 * abs(cmd)]
    rows = [r for r in rows if abs(float(r["ang_deg"])) >= 25.0]
    if not rows:
        rows = [r for r in log if abs(float(r.get("omega_y_cmd", 0.0))) > 0.5]
    if not rows:
        return {"median": 0.0, "max": 0.0, "min": 0.0}
    om = np.array([r["omega_axis"] for r in rows], float)
    return {"median": float(np.median(om)), "max": float(np.max(om)), "min": float(np.min(om))}


def supported_peak(log):
    best = None
    for r in log:
        if int(r["nL"]) <= 0 or int(r["nR"]) <= 0:
            continue
        if r.get("label") not in ("SWING", "ROTATE", "PARENT"):
            continue
        if best is None or float(r["v_obj"][2]) > float(best["v_obj"][2]):
            best = r
    return best


def swing(sim, gains, parent, omega_cmd, ctl=None):
    restore(sim, parent)
    disable_object_table_only(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    log = []
    m0 = measure(sim, omega_y=0.0, tau=TAU_SEC, label="PARENT", Rh0=Rh0)
    enrich(sim, m0, gains)
    append_log(log, m0)
    ev = {"ROTATE_START": float(sim.data.time), "omega_cmd": float(omega_cmd)}
    timeout = float(sim.data.time) + np.deg2rad(MAX_DEG) / max(abs(omega_cmd), 1e-6) + 1.2
    while float(sim.data.time) < timeout:
        m, stop = step_ovr(sim, gains, omega_cmd, 0.0, 0.0, TAU_SEC, log, Rh0, "SWING", ev, ctl)
        if stop:
            break
        if m["nL"] == 0 and m["nR"] == 0:
            ev["CONTACT_LOST"] = m["t"]
            break
        if abs(float(m["ang_deg"])) >= MAX_DEG - 0.8:
            break
    pk = supported_peak(log)
    cr = cruise_omega(log, omega_cmd)
    sat_n = int(sum(1 for r in log if r.get("any_sat")))
    qdot = np.array([r["qdot"] for r in log], float)
    util = np.array([r["tau_util"] for r in log], float)
    return {
        "omega_cmd": float(omega_cmd),
        "mass": float(sim.model.body_mass[sim.ids.object_body]),
        "cruise": cr,
        "peak_supported": None if pk is None else slim_row(pk),
        "peak_vz": None if pk is None else float(pk["v_obj"][2]),
        "peak_ang": None if pk is None else float(pk["ang_deg"]),
        "ori_err_max": float(max(r["ori_err_deg"] for r in log)),
        "p_err_max": float(max(r["p_err_norm"] for r in log)),
        "qdot_peak": np.max(np.abs(qdot), axis=0).tolist(),
        "tau_util_peak": np.max(util, axis=0).tolist(),
        "n_sat": sat_n,
        "ang_end": float(log[-1]["ang_deg"]),
        "nL_end": int(log[-1]["nL"]),
        "nR_end": int(log[-1]["nR"]),
        "events": ev,
        "log": log,
        "Rh0": Rh0,
    }


def open_at_angle(sim, gains, parent, omega_cmd, open_deg, ctl=None, recatch=False, t_close=None):
    restore(sim, parent)
    disable_object_table_only(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    log = []
    m0 = measure(sim, omega_y=0.0, tau=TAU_SEC, label="PARENT", Rh0=Rh0)
    enrich(sim, m0, gains)
    append_log(log, m0)
    ev = {"ROTATE_START": float(sim.data.time), "open_cmd_deg": float(open_deg), "omega_cmd": float(omega_cmd)}
    timeout = float(sim.data.time) + np.deg2rad(MAX_DEG) / max(abs(omega_cmd), 1e-6) + 1.2
    while float(sim.data.time) < timeout:
        ang = signed_rot_about_y_deg(
            Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        )
        if abs(ang) >= abs(open_deg) - 0.8:
            break
        m, stop = step_ovr(sim, gains, omega_cmd, 0.0, 0.0, TAU_SEC, log, Rh0, "SWING", ev, ctl)
        if stop:
            break
        if m["nL"] == 0 and m["nR"] == 0:
            ev["CONTACT_LOST_BEFORE_OPEN"] = m["t"]
            break
    if "CONTACT_LOST_BEFORE_OPEN" in ev:
        return {"lost_before_open": True, "events": ev, "log": log, "Rh0": Rh0}

    ev["OPEN_COMMAND"] = float(sim.data.time)
    ev["theta_open_command"] = float(log[-1]["ang_deg"])
    dt = float(sim.model.opt.timestep)
    both = 0.0
    t_open0 = float(sim.data.time)
    while float(sim.data.time) < t_open0 + 0.45:
        m, stop = step_ovr(sim, gains, omega_cmd, 0.0, 0.0, TAU_OPEN, log, Rh0, "OPEN", ev, ctl)
        if stop:
            break
        if m["nL"] == 0 and m["nR"] == 0:
            if "FIRST_BOTH_OFF" not in ev:
                ev["FIRST_BOTH_OFF"] = m["t"]
                ev["theta_first_both_off"] = float(m["ang_deg"])
                ev["both_off_state"] = slim_row(m)
                ev["open_latency_s"] = float(m["t"] - ev["OPEN_COMMAND"])
            both += dt
            if both >= BOTH_OFF_MIN:
                break
        else:
            both = 0.0

    z_off = None
    t_off = ev.get("FIRST_BOTH_OFF")
    if t_off is not None:
        z_off = float(ev["both_off_state"]["obj_z"])

    captured = False
    hold_ok = False
    fail_why = None
    if "FIRST_BOTH_OFF" in ev:
        z_max_flight = z_off
        t_apex = None
        vz_prev = None
        t_lim = float(sim.data.time) + (1.20 if recatch else 0.80)
        closed = False
        while float(sim.data.time) < t_lim:
            tau = TAU_OPEN
            wy = 0.0
            lab = "BALLISTIC"
            if recatch and closed:
                tau = TAU_SEC
                lab = "CLOSE" if "HOLD_START" not in ev else "HOLD"
            elif recatch and t_close is not None and float(sim.data.time) >= float(t_close) and not closed:
                closed = True
                ev["CLOSE_START"] = float(sim.data.time)
                ev["CLOSE_TIMING"] = "PRIVILEGED_GT"
                tau = TAU_SEC
                lab = "CLOSE"
            m, stop = step_ovr(sim, gains, wy, 0.0, 0.0, tau, log, Rh0, lab, ev, ctl)
            if stop:
                fail_why = "floor"
                break
            z = float(m["obj_z"])
            vz = float(m["v_obj"][2])
            if z_off is not None and z > z_max_flight:
                z_max_flight = z
            if vz_prev is not None and vz_prev > 0.0 and vz <= 0.0 and "BALLISTIC_APEX" not in ev:
                ev["BALLISTIC_APEX"] = m["t"]
                t_apex = m["t"]
            vz_prev = vz
            if recatch and closed:
                if (m["nL"] > 0 or m["nR"] > 0) and "FIRST_RECONTACT" not in ev:
                    ev["FIRST_RECONTACT"] = m["t"]
                if m["nL"] > 0 and m["nR"] > 0 and "FIRST_BILATERAL" not in ev:
                    ev["FIRST_BILATERAL"] = m["t"]
                if "FIRST_BILATERAL" in ev and "HOLD_START" not in ev:
                    if m["t"] - ev["FIRST_BILATERAL"] >= 0.050:
                        ev["HOLD_START"] = m["t"]
                        captured = True
                        t_lim = m["t"] + HOLD_S_CATCH
                if "HOLD_START" in ev:
                    if m["nL"] == 0 or m["nR"] == 0:
                        ev["HOLD_LOST"] = m["t"]
                        captured = False
                        fail_why = "hold_lost_contact"
                        break
            if (not recatch) and (m["nL"] > 0 or m["nR"] > 0) and "RECONTACT_DURING_BALLISTIC" not in ev:
                ev["RECONTACT_DURING_BALLISTIC"] = m["t"]
        if recatch and captured and "HOLD_LOST" not in ev and "FLOOR" not in ev:
            end = log[-1]
            hold_ok = (
                int(end["nL"]) > 0
                and int(end["nR"]) > 0
                and float(end.get("v_rel", 99)) < 0.08
                and abs(float(end["rh"][0])) < 0.020
            )
            if not hold_ok:
                fail_why = "hold_unbounded_rel"
                captured = False
        if recatch and not captured and fail_why is None:
            fail_why = "no_bilateral_hold"
        ball = {
            "z_at_both_off": z_off,
            "z_max_after_both_off": z_max_flight,
            "rise_after_both_off": None if z_off is None else float(z_max_flight - z_off),
            "t_apex": ev.get("BALLISTIC_APEX", t_apex),
            "t_both_off": t_off,
            "dt_apex": None
            if t_off is None or ev.get("BALLISTIC_APEX") is None
            else float(ev["BALLISTIC_APEX"] - t_off),
        }
    else:
        ball = None
        fail_why = "no_first_both_off"

    both = ev.get("both_off_state")
    vz = None if both is None else float(both["v_obj"][2])
    return {
        "lost_before_open": False,
        "open_cmd_deg": float(open_deg),
        "omega_cmd": float(omega_cmd),
        "mass": float(CYL_MASS),
        "events": {k: v for k, v in ev.items() if k != "Rh0"},
        "both_off": both,
        "v_obj_z_both_off": vz,
        "theta_both_off": ev.get("theta_first_both_off"),
        "open_latency_s": ev.get("open_latency_s"),
        "ballistic": ball,
        "captured": captured,
        "ok_hold": hold_ok,
        "fail_why": fail_why,
        "end": slim_row(log[-1]),
        "log": log,
        "Rh0": Rh0,
    }


def pick_gt_close(log, ev) -> float | None:
    t_off = ev.get("FIRST_BOTH_OFF")
    if t_off is None:
        return None
    flight = [r for r in log if r["t"] >= t_off - 1e-12]
    if not flight:
        return None
    t_apex = ev.get("BALLISTIC_APEX")
    if t_apex is None:
        for a, b in zip(flight, flight[1:]):
            if float(a["v_obj"][2]) > 0.0 and float(b["v_obj"][2]) <= 0.0:
                t_apex = float(b["t"])
                break
    after = [r for r in flight if t_apex is None or r["t"] >= t_apex - 1e-12]
    best = None
    best_score = None
    for r in after:
        ph = np.asarray(r["p_hand"], float)
        po = np.asarray(r["p_obj"], float)
        rh = np.asarray(r["rh"], float)
        d = float(np.linalg.norm(po - ph))
        vz = float(r["v_obj"][2])
        if po[2] < ph[2] - 0.015:
            continue
        if vz > 0.05:
            continue
        score = d + 2.0 * abs(rh[0]) + 1.5 * abs(rh[1])
        if best_score is None or score < best_score:
            best_score = score
            best = r
    if best is None:
        cand = [r for r in after if float(r["v_obj"][2]) <= 0.02]
        if cand:
            best = min(cand, key=lambda r: float(np.linalg.norm(np.asarray(r["p_obj"]) - np.asarray(r["p_hand"]))))
    return None if best is None else float(best["t"])


def corridor_note(log, t_off):
    flight = [r for r in log if r["t"] >= t_off - 1e-12 and r.get("label") in ("OPEN", "BALLISTIC", "CLOSE")]
    out = []
    for r in flight[:: max(1, len(flight) // 12)]:
        ph = np.asarray(r["p_hand"], float)
        po = np.asarray(r["p_obj"], float)
        out.append(
            {
                "t": r["t"],
                "label": r["label"],
                "obj_z": r["obj_z"],
                "hand_z": float(ph[2]),
                "dz": float(po[2] - ph[2]),
                "d": float(np.linalg.norm(po - ph)),
                "rh_mm": (1e3 * np.asarray(r["rh"], float)).tolist(),
                "vz": r["v_obj"][2],
                "nL": r["nL"],
                "nR": r["nR"],
                "ap": r["aperture"],
            }
        )
    return out


def classify(sw3, sw4, throw4, throw3_matched):
    om3 = sw3["cruise"]["median"]
    om4 = sw4["cruise"]["median"]
    if om4 < om3 + 0.25:
        return "C", "CONTROLLER-LIMITED"
    vz4s = sw4["peak_vz"]
    vz3s = sw3["peak_vz"]
    vz4r = throw4.get("v_obj_z_both_off")
    vz3r = None if throw3_matched is None else throw3_matched.get("v_obj_z_both_off")
    rise4 = (throw4.get("ballistic") or {}).get("rise_after_both_off")
    rise3 = None if throw3_matched is None else (throw3_matched.get("ballistic") or {}).get("rise_after_both_off")
    supported_up = vz4s is not None and vz3s is not None and vz4s >= vz3s + 0.04
    release_up = vz4r is not None and (vz3r is None or vz4r >= vz3r + 0.04)
    ball_up = rise4 is not None and rise4 >= 0.003 and (rise3 is None or rise4 >= max(0.002, 2.0 * float(rise3 or 0.0)))
    if om4 >= om3 + 0.25 and (not supported_up) and (vz4r is None or (vz3r is not None and vz4r < vz3r + 0.02)):
        return "B", "CONTACT-LIMITED"
    if supported_up and release_up and (ball_up or (vz4r is not None and vz4r >= 0.32)):
        return "A", "EFFECTIVE AUTHORITY INCREASE"
    if om4 >= om3 + 0.25 and (supported_up or release_up):
        if vz4r is not None and vz4s is not None and vz4r < 0.70 * vz4s:
            return "B", "CONTACT-LIMITED"
        return "A", "EFFECTIVE AUTHORITY INCREASE"
    return "B", "CONTACT-LIMITED"


def recatch_fail_from_log(trial):
    ev = trial.get("events") or {}
    end = trial.get("end") or {}
    both = trial.get("both_off") or {}
    reasons = []
    if trial.get("fail_why"):
        reasons.append(trial["fail_why"])
    if ev.get("FLOOR") and "FIRST_BILATERAL" not in ev:
        reasons.append("object reached floor before bilateral recapture")
    if "FIRST_RECONTACT" not in ev:
        reasons.append("misses capture corridor (no recontact)")
    elif "FIRST_BILATERAL" not in ev:
        reasons.append("one-finger recontact only; aperture/geometry mismatch")
    if ev.get("HOLD_LOST"):
        reasons.append("unstable recontact / hold lost")
    ap = end.get("aperture")
    if ap is not None and ap > 0.035 and "FIRST_BILATERAL" not in ev:
        reasons.append("aperture still large at end")
    vz = trial.get("v_obj_z_both_off")
    if vz is not None and abs(vz) > 0.6:
        reasons.append("object fast at release")
    rh = both.get("rh")
    return {
        "fail_why": trial.get("fail_why"),
        "notes": reasons,
        "end_nL_nR": [end.get("nL"), end.get("nR")],
        "end_v_rel": end.get("v_rel"),
        "end_rh": end.get("rh"),
        "events": {k: ev.get(k) for k in (
            "OPEN_COMMAND", "FIRST_BOTH_OFF", "BALLISTIC_APEX", "CLOSE_START",
            "FIRST_RECONTACT", "FIRST_BILATERAL", "HOLD_START", "HOLD_LOST", "FLOOR",
        ) if k in ev},
    }


def write_report(prep, sw3, sw4, throw3, throw4s, recatch_row, klass, klass_name, gate_ok):
    a = []
    A = a.append
    A("# Omega=4 diagnostic throw / recapture test")
    A("")
    A("Cylinder mass was **intentionally held fixed** at the current construction value.")
    A("Mass robustness is out of scope. No mass sweep, friction change, or disturbance-distribution change.")
    A("")
    A(f"Mass = **{float(prep['mass']):.4f} kg** (`CYL_MASS={CYL_MASS}`). Friction pair μ = {PAIR_MU}.")
    A("")
    A("`RECOVERY4D_W_HY_MAX` was **not** changed. `omega_y_cmd = +4.0` is a diagnostic override via `tick_omega_override` (`w_world = r_des @ [0, 4, 0]`). Legal map still clips `v_x`, `v_z`, and gripper tau.")
    A("")
    A("## 1. Rotation gate (secure swing, tau=-18)")
    A("")
    A("| | omega_cmd | cruise ω_axis median | max | supported peak v_obj,z | angle of peak | ori_err max (°) | pos drift max (mm) | peak |q̇_5| | peak τ util j5 | sat steps |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for sw, name in ((sw3, "omega=3"), (sw4, "omega=4")):
        pk = sw.get("peak_supported") or {}
        A(
            f"| {name} | {sw['omega_cmd']} | {sw['cruise']['median']:.3f} | {sw['cruise']['max']:.3f} | "
            f"{sw.get('peak_vz')} | {None if sw.get('peak_ang') is None else round(sw['peak_ang'], 1)} | "
            f"{sw['ori_err_max']:.2f} | {1e3*sw['p_err_max']:.1f} | {sw['qdot_peak'][4]:.3f} | "
            f"{sw['tau_util_peak'][4]:.3f} | {sw['n_sat']} |"
        )
    A("")
    A(f"omega=3 q̇ peak (j1–j7) = {np.round(sw3['qdot_peak'], 3).tolist()}")
    A(f"omega=4 q̇ peak (j1–j7) = {np.round(sw4['qdot_peak'], 3).tolist()}")
    A(f"omega=3 τ util peak = {np.round(sw3['tau_util_peak'], 3).tolist()}")
    A(f"omega=4 τ util peak = {np.round(sw4['tau_util_peak'], 3).tolist()}")
    A("")
    if not gate_ok:
        A("**STOP at rotation gate.** Actual hand rotation did not materially exceed the omega=3 case. Subsequent throw is not interpreted as an authority increase.")
        A("")
        A(f"**Class {klass}: {klass_name}.**")
        A("")
        A("## Viewer")
        A("")
        A("```text")
        A("python training/demo_dynamic_recatch.py --mode throw_omega4")
        A("```")
        REPORT.write_text("\n".join(a), encoding="utf-8")
        return

    pk4 = sw4.get("peak_supported") or {}
    A("omega=4 peak-supported snapshot:")
    A(f"- r_h = {pk4.get('rh')}")
    A(f"- topology nL/nR = {pk4.get('nL')}/{pk4.get('nR')}  Fn={pk4.get('Fn_L')}/{pk4.get('Fn_R')}  ρ={pk4.get('rho_max')}")
    A(f"- v_rel = {pk4.get('v_rel')}  carry_err = {pk4.get('v_carry_err_norm')}")
    A("")
    A("## 2. Matched comparison (same parent IC)")
    A("")
    A("Kinematic predictions (not required matches): ω=3 → ~0.30 m/s; ω=4 → ~0.40 m/s with r_eff≈0.099 m.")
    A("")
    A("| | actual ω cruise | supported peak v_obj,z | peak angle | FIRST_BOTH_OFF v_obj,z | both-off angle | rise after both-off (m) |")
    A("|---|---|---|---|---|---|---|")

    def row(name, sw, th):
        b = (th or {}).get("ballistic") or {}
        A(
            f"| {name} | {sw['cruise']['median']:.3f} | {sw.get('peak_vz')} | {sw.get('peak_ang')} | "
            f"{None if th is None else th.get('v_obj_z_both_off')} | "
            f"{None if th is None else th.get('theta_both_off')} | {b.get('rise_after_both_off')} |"
        )

    row("omega=3", sw3, throw3)
    row("omega=4 chosen", sw4, throw4s[-1] if throw4s else None)
    A("")
    A("## 3. OPEN phases (rotation continues; at most two)")
    A("")
    A("OPEN while `omega_y_cmd` stays +4. Physical release = FIRST_BOTH_OFF. Latency compensation uses the measured OPEN→BOTH_OFF delay, not a 90° ritual.")
    A("")
    for i, th in enumerate(throw4s):
        bo = th.get("both_off") or {}
        A(f"### phase {i+1}: OPEN_COMMAND at {th.get('open_cmd_deg')} deg")
        A(f"- OPEN_COMMAND angle = {th.get('events', {}).get('theta_open_command')}")
        A(f"- FIRST_BOTH_OFF t, angle = {th.get('events', {}).get('FIRST_BOTH_OFF')}, {th.get('theta_both_off')} deg")
        A(f"- open latency = {th.get('open_latency_s')} s")
        A(f"- actual ω_axis at both-off = {None if not bo else bo.get('omega_axis')}")
        A(f"- v_obj = {None if not bo else bo.get('v_obj')}  especially v_obj,z = {th.get('v_obj_z_both_off')}")
        A(f"- w_obj = {None if not bo else bo.get('w_obj')}")
        A(f"- r_h = {None if not bo else bo.get('rh')}")
        A(f"- v_rel = {None if not bo else bo.get('v_rel')}  v_rel_h = {None if not bo else bo.get('v_rel_h')}")
        A(f"- aperture = {None if not bo else bo.get('aperture')}")
        A(f"- last topology = {None if not bo else bo.get('contact_topology')}  nL/nR={None if not bo else bo.get('nL')}/{None if not bo else bo.get('nR')}")
        b = th.get("ballistic") or {}
        A(f"- height at both-off z={b.get('z_at_both_off')}; **after** FIRST_BOTH_OFF z_max={b.get('z_max_after_both_off')} rise={b.get('rise_after_both_off')} dt_apex={b.get('dt_apex')}")
        A("")
    A("Supported-phase height (before both-off) is the swing `obj_z` at the supported v_z peak, distinct from post-release rise.")
    if sw4.get("peak_supported"):
        A(f"Supported peak obj_z = {sw4['peak_supported'].get('obj_z')} at {sw4['peak_ang']} deg.")
    A("")
    A("## 4. Classification")
    A("")
    A(f"**Class {klass}: {klass_name}.**")
    A("")
    A("## 5. Recapture")
    A("")
    if recatch_row is None:
        A("Not attempted (class B/C, or throw not materially better).")
    else:
        A("**PRIVILEGED** GT close timing: after FIRST_BOTH_OFF the wrist parks (`ω=0`) at the catch orientation; CLOSE when the GT ballistic sample is returning into the capture corridor (vz≤0, not below the hand). Not an observable policy.")
        A(f"CLOSE_START = {recatch_row.get('events', {}).get('CLOSE_START')}  (PRIVILEGED)")
        A(f"captured={recatch_row.get('captured')} ok_hold={recatch_row.get('ok_hold')} fail={recatch_row.get('fail_why')}")
        A(f"events: {recatch_row.get('events')}")
        if recatch_row.get("ok_hold"):
            end = recatch_row.get("end") or {}
            A("")
            A("Hold ≥2 s at **catch orientation** (no return-to-nominal).")
            A(f"end nL/nR={end.get('nL')}/{end.get('nR')} v_rel={end.get('v_rel')} rh={end.get('rh')} obj_z={end.get('obj_z')}")
            A("")
            A("Mechanism established for this stage: diagnostic ω=4 rotation → dynamic release → ballistic flight → recapture → stable hold. STOP (no ω=5, no release optimization, no CENTER-6).")
        else:
            why = recatch_fail_from_log(recatch_row)
            A("")
            A("Recapture failed. Why from raw trajectory:")
            A(f"- {why}")
            A("Did not increase omega again.")
    A("")
    A("## Viewer")
    A("")
    A("```text")
    A("python training/demo_dynamic_recatch.py --mode throw_omega4")
    if recatch_row is not None:
        A("python training/demo_dynamic_recatch.py --mode recatch_omega4")
    A("```")
    A("")
    A("SPACE pause, R restart, `[` `]` speed. No MP4.")
    REPORT.write_text("\n".join(a), encoding="utf-8")
    print("wrote", REPORT, flush=True)


def headless_omega4():
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    if "kp_null" not in gains:
        gains["kp_null"] = 4.0
        gains["kd_null"] = 0.8
    sim = make_parent_sim()
    orig = copy_masks(sim)
    prep = prepare_high_parent(sim, cfg, gains, orig)
    if not prep.get("ok"):
        dump(RAW / "omega4_prep_fail.json", prep)
        raise RuntimeError(prep)
    parent = prep["parent"]
    mass = float(sim.model.body_mass[sim.ids.object_body])
    assert abs(mass - float(CYL_MASS)) < 1e-9, (mass, CYL_MASS)
    print("MASS FIXED", mass, "RECOVERY4D_W_HY_MAX unchanged", OMEGA3, "override", OMEGA4, flush=True)

    sw3 = swing(sim, gains, parent, OMEGA3)
    sw4 = swing(sim, gains, parent, OMEGA4)
    save_npz(RAW / "omega4_swing3.npz", sw3["log"])
    save_npz(RAW / "omega4_swing4.npz", sw4["log"])
    dump(
        RAW / "omega4_swings.json",
        {
            "mass": mass,
            "omega3": {k: v for k, v in sw3.items() if k not in ("log", "Rh0")},
            "omega4": {k: v for k, v in sw4.items() if k not in ("log", "Rh0")},
        },
    )
    print(
        "SWING3 om", sw3["cruise"], "vz", sw3["peak_vz"], "ang", sw3["peak_ang"],
        "SWING4 om", sw4["cruise"], "vz", sw4["peak_vz"], "ang", sw4["peak_ang"],
        "sat", sw4["n_sat"],
        flush=True,
    )

    gate_ok = sw4["cruise"]["median"] >= sw3["cruise"]["median"] + 0.25
    throw3 = None
    throw4s = []
    recatch_row = None
    klass, klass_name = ("C", "CONTROLLER-LIMITED") if not gate_ok else ("A", "EFFECTIVE AUTHORITY INCREASE")

    if gate_ok:
        peak_ang = float(sw4["peak_ang"])
        latency_deg = np.degrees(OMEGA4 * 0.011)
        open_a = peak_ang - latency_deg
        open_b = peak_ang - 12.0
        print("OPEN phases", open_a, open_b, "peak_ang", peak_ang, flush=True)
        t4a = open_at_angle(sim, gains, parent, OMEGA4, open_a, recatch=False)
        throw4s.append(t4a)
        dump(RAW / "omega4_throw_phaseA.json", {k: v for k, v in t4a.items() if k not in ("log", "Rh0")})
        save_npz(RAW / "omega4_throw_phaseA.npz", t4a["log"])
        print("phaseA both_off vz", t4a.get("v_obj_z_both_off"), "theta", t4a.get("theta_both_off"), "lat", t4a.get("open_latency_s"), flush=True)

        need_b = True
        vz_a = t4a.get("v_obj_z_both_off")
        if vz_a is not None and sw4["peak_vz"] is not None and vz_a >= 0.85 * float(sw4["peak_vz"]):
            need_b = False
        if abs(open_b - open_a) < 4.0:
            need_b = False
        if need_b:
            t4b = open_at_angle(sim, gains, parent, OMEGA4, open_b, recatch=False)
            throw4s.append(t4b)
            dump(RAW / "omega4_throw_phaseB.json", {k: v for k, v in t4b.items() if k not in ("log", "Rh0")})
            save_npz(RAW / "omega4_throw_phaseB.npz", t4b["log"])
            print("phaseB both_off vz", t4b.get("v_obj_z_both_off"), "theta", t4b.get("theta_both_off"), flush=True)

        chosen = max(throw4s, key=lambda t: -1e9 if t.get("v_obj_z_both_off") is None else float(t["v_obj_z_both_off"]))
        throw3 = open_at_angle(sim, gains, parent, OMEGA3, float(chosen["open_cmd_deg"]), recatch=False)
        dump(RAW / "omega4_throw3_matched.json", {k: v for k, v in throw3.items() if k not in ("log", "Rh0")})
        save_npz(RAW / "omega4_throw3_matched.npz", throw3["log"])
        print("matched omega3 both_off vz", throw3.get("v_obj_z_both_off"), flush=True)

        klass, klass_name = classify(sw3, sw4, chosen, throw3)
        print("CLASS", klass, klass_name, flush=True)

        plan = {
            "omega_cmd": OMEGA4,
            "open_cmd_deg": chosen["open_cmd_deg"],
            "class": klass,
            "recatch": False,
            "mass": mass,
            "RECOVERY4D_W_HY_MAX_unchanged": OMEGA3,
        }
        if klass == "A":
            scout = open_at_angle(sim, gains, parent, OMEGA4, float(chosen["open_cmd_deg"]), recatch=False)
            t_close = pick_gt_close(scout["log"], scout["events"])
            if t_close is None:
                t_close = (scout.get("events") or {}).get("BALLISTIC_APEX")
            print("PRIVILEGED t_close", t_close, "apex", (scout.get("events") or {}).get("BALLISTIC_APEX"), flush=True)
            rec = open_at_angle(
                sim, gains, parent, OMEGA4, float(chosen["open_cmd_deg"]), recatch=True, t_close=t_close
            )
            recatch_row = {k: v for k, v in rec.items() if k not in ("log", "Rh0")}
            recatch_row["corridor"] = corridor_note(rec["log"], rec["events"].get("FIRST_BOTH_OFF") or 0.0)
            recatch_row["t_close_privileged"] = t_close
            save_npz(RAW / "omega4_recatch.npz", rec["log"])
            dump(RAW / "omega4_recatch.json", recatch_row)
            plan["recatch"] = True
            plan["t_close"] = t_close
            plan["ok_hold"] = rec.get("ok_hold")
            print("RECATCH captured", rec.get("captured"), "hold", rec.get("ok_hold"), rec.get("fail_why"), flush=True)
        dump(PLAN_PATH, plan)
        dump(RAW / "omega4_throw_chosen.json", {k: v for k, v in chosen.items() if k not in ("log", "Rh0")})
    else:
        dump(
            PLAN_PATH,
            {
                "omega_cmd": OMEGA4,
                "open_cmd_deg": 90.0,
                "class": "C",
                "recatch": False,
                "mass": mass,
                "gate_ok": False,
            },
        )
        klass, klass_name = "C", "CONTROLLER-LIMITED"

    write_report(
        prep,
        {k: v for k, v in sw3.items() if k not in ("log", "Rh0")},
        {k: v for k, v in sw4.items() if k not in ("log", "Rh0")},
        None if throw3 is None else {k: v for k, v in throw3.items() if k not in ("log", "Rh0")},
        [{k: v for k, v in t.items() if k not in ("log", "Rh0")} for t in throw4s],
        recatch_row,
        klass,
        klass_name,
        gate_ok,
    )
    return klass


def interactive_omega4(mode: str) -> None:
    import mujoco.viewer

    from training.demo_airborne_recovery import apply_masks

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    if "kp_null" not in gains:
        gains["kp_null"] = 4.0
        gains["kd_null"] = 0.8
    plan = {"omega_cmd": OMEGA4, "open_cmd_deg": 87.0, "recatch": False, "t_close": None}
    if PLAN_PATH.is_file():
        plan.update(json.loads(PLAN_PATH.read_text(encoding="utf-8")))
    recatch = bool(mode == "recatch_omega4" and plan.get("recatch"))
    ctl = {"pause": False, "reset": False, "speed": 0.35, "overlay": {}, "label": mode}

    def sync():
        vwr = ctl.get("viewer")
        if vwr is None:
            return
        o = ctl.get("overlay") or {}
        _viewer_overlay(
            vwr,
            [
                ("CASE", str(o.get("case", mode))),
                ("t", f"{float(o.get('t', 0)):.3f} s"),
                ("PHASE", str(o.get("phase", ""))),
                ("nL/nR", f"{o.get('nL','-')}/{o.get('nR','-')}"),
                ("ang", f"{float(o.get('ang', 0)):.0f} deg"),
                ("v_obj,z", f"{float(o.get('vz', 0)):+.3f} m/s"),
                ("keys", "SPACE pause  R restart  [ ] speed"),
            ],
        )
        vwr.sync()
        dt = 0.002
        spd = max(float(ctl.get("speed", 0.35)), 0.05)
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

    sim = make_parent_sim()
    orig = copy_masks(sim)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        ctl["viewer"] = vwr
        apply_camera_preset(vwr, CAM)
        while vwr.is_running():
            ctl["reset"] = False
            ctl["_wall"] = None
            apply_masks(sim, orig)
            sim.reset(CYL_MASS, PAIR_MU, np.zeros(3))
            prep = prepare_high_parent(sim, cfg, gains, orig, ctl=ctl)
            if not prep.get("ok"):
                time.sleep(0.2)
                continue
            open_at_angle(
                sim,
                gains,
                prep["parent"],
                float(plan.get("omega_cmd", OMEGA4)),
                float(plan.get("open_cmd_deg", 87.0)),
                ctl=ctl,
                recatch=recatch,
                t_close=plan.get("t_close"),
            )
            if ctl.get("reset"):
                continue
            while vwr.is_running() and not ctl.get("reset"):
                sync()
                time.sleep(0.03)
            if not ctl.get("reset"):
                break


if __name__ == "__main__":
    headless_omega4()
