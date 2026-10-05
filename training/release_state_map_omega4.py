"""Release-state map at diagnostic omega_y=4. No recatch until map completes."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg, ori_error_deg
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import physical_pack
from training.demo_airborne_recovery import copy_masks, disable_object_table_only, prepare_high_parent
from training.demo_airborne_recapture import make_parent_sim
from training.demo_ballistic_impact import make_sim as make_ballistic_sim
from training.demo_ballistic_recovery import restore_ballistic
from training.demo_dynamic_recatch import (
    BOTH_OFF_MIN,
    EARLY_DOC_RHX_MM,
    EARLY_PKL,
    OUT,
    RAW,
    TAU_OPEN,
    TAU_SEC,
    append_log,
    dump,
    measure,
    overlay,
    slim_row,
)
from training.demo_teleport_recovery_state import CYL_MASS, PAIR_MU, signed_rot_about_y_deg
from training.omega4_throw_test import (
    MAX_DEG,
    OMEGA3,
    OMEGA4,
    enrich,
    step_ovr,
    tick_omega_override,
)
from training.replay_core import freeze

REPORT = OUT / "RELEASE_STATE_MAP_OMEGA4.md"
FLIGHT_S = 0.120
PARK_S = 0.100
# Geometric capture pocket of an open gripper (not a score). Cylinder R=0.018.
RHX_IN = 0.022
RHY_IN = 0.020
RHZ_LO = 0.050
RHZ_HI = 0.130

# From omega=4 throw log: peak v_z ~79.5°, current OPEN ~76.9°.
PHASES = (
    ("current", 76.9),
    ("modestly_earlier", 70.0),
    ("clearly_earlier", 60.0),
)


def mm(x):
    return 1e3 * float(x)


def in_corridor(rh) -> bool:
    r = np.asarray(rh, float)
    return bool(abs(r[0]) < RHX_IN and abs(r[1]) < RHY_IN and RHZ_LO < r[2] < RHZ_HI)


def rel_ori_deg(R_rel) -> float:
    r = np.asarray(R_rel, float).reshape(3, 3)
    c = 0.5 * (np.trace(r) - 1.0)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def snapshot_ic(sim, name: str, extra=None) -> dict:
    o = physical_pack(sim)
    m = measure(sim, tau=TAU_SEC, label="IC")
    d = {
        "name": name,
        "t": float(sim.data.time),
        "mass": float(sim.model.body_mass[sim.ids.object_body]),
        "rh": np.asarray(o["rh"], float).tolist(),
        "rh_mm": (1e3 * np.asarray(o["rh"], float)).tolist(),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "v_rel": float(o["v_rel"]),
        "aperture": float(m["aperture"]),
        "obj_z": float(o["obj_z"]),
        "bilateral": int(o["nL"]) > 0 and int(o["nR"]) > 0,
    }
    if extra:
        d.update(extra)
    return d


def supported_samples(log, degs):
    out = []
    for deg in degs:
        cand = [r for r in log if int(r["nL"]) > 0 and int(r["nR"]) > 0]
        if not cand:
            continue
        r = min(cand, key=lambda x: abs(float(x["ang_deg"]) - deg))
        vo = np.asarray(r["v_obj"], float)
        rh = np.asarray(r["rh"], float)
        out.append(
            {
                "ang_deg": float(r["ang_deg"]),
                "v_obj": vo.tolist(),
                "v_obj_z": float(vo[2]),
                "rh_mm": (1e3 * rh).tolist(),
                "v_rel_h": r["v_rel_h"],
                "w_obj": r["w_obj"],
                "nL": r["nL"],
                "nR": r["nR"],
                "aperture": r["aperture"],
                "Fn_L": r["Fn_L"],
                "Fn_R": r["Fn_R"],
                "rho": r.get("rho_max"),
                "omega_axis": r.get("omega_axis"),
            }
        )
    return out


def swing_supported(sim, gains, restore_fn):
    restore_fn()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    log = []
    m0 = measure(sim, omega_y=0.0, tau=TAU_SEC, label="PARENT", Rh0=Rh0)
    enrich(sim, m0, gains)
    append_log(log, m0)
    ev = {"ROTATE_START": float(sim.data.time)}
    timeout = float(sim.data.time) + np.deg2rad(MAX_DEG) / OMEGA4 + 1.2
    while float(sim.data.time) < timeout:
        m, stop = step_ovr(sim, gains, OMEGA4, 0.0, 0.0, TAU_SEC, log, Rh0, "SWING", ev)
        if stop:
            break
        if m["nL"] == 0 and m["nR"] == 0:
            break
        if abs(float(m["ang_deg"])) >= MAX_DEG - 0.8:
            break
    peak = None
    for r in log:
        if int(r["nL"]) > 0 and int(r["nR"]) > 0:
            if peak is None or float(r["v_obj"][2]) > float(peak["v_obj"][2]):
                peak = r
    return log, peak, Rh0


def park_transient(sim, gains, restore_fn, park_deg: float):
    restore_fn()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    log = []
    m0 = measure(sim, omega_y=0.0, tau=TAU_SEC, label="PARENT", Rh0=Rh0)
    enrich(sim, m0, gains)
    append_log(log, m0)
    ev = {}
    timeout = float(sim.data.time) + np.deg2rad(MAX_DEG) / OMEGA4 + 1.0
    while float(sim.data.time) < timeout:
        ang = signed_rot_about_y_deg(
            Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        )
        if abs(ang) >= abs(park_deg) - 0.8:
            break
        m, stop = step_ovr(sim, gains, OMEGA4, 0.0, 0.0, TAU_SEC, log, Rh0, "SWING", ev)
        if stop:
            break
    ev["PARK"] = float(sim.data.time)
    t0 = float(sim.data.time)
    park_log = []
    while float(sim.data.time) < t0 + PARK_S:
        m, stop = step_ovr(sim, gains, 0.0, 0.0, 0.0, TAU_SEC, log, Rh0, "PARK", ev)
        if stop:
            break
        r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        r = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        park_log.append(
            {
                "dt": float(m["t"] - t0),
                "ang_deg": float(m["ang_deg"]),
                "omega_axis": float(m.get("omega_axis", 0.0)),
                "w_hand_norm": float(m.get("w_hand_norm", 0.0)),
                "ori_err_deg": float(m.get("ori_err_deg", 0.0)),
                "r_des_y": r_des[:, 1].tolist(),
                "R_hand_y": r[:, 1].tolist(),
            }
        )
    dang = 0.0 if not park_log else float(park_log[-1]["ang_deg"] - park_log[0]["ang_deg"])
    return {
        "park_cmd_deg": float(park_deg),
        "t_park": t0,
        "ang_at_park": None if not park_log else park_log[0]["ang_deg"],
        "ang_after_100ms": None if not park_log else park_log[-1]["ang_deg"],
        "delta_ang_deg": dang,
        "omega_at_park": None if not park_log else park_log[0]["omega_axis"],
        "omega_after_100ms": None if not park_log else park_log[-1]["omega_axis"],
        "ori_err_at_park": None if not park_log else park_log[0]["ori_err_deg"],
        "ori_err_after_100ms": None if not park_log else park_log[-1]["ori_err_deg"],
        "samples": park_log[::5],
    }


def flight_series(log, t_off, p_obj0, p_hand0, ang0):
    rows = []
    apex = None
    vz_prev = None
    for r in log:
        if r["t"] < t_off - 1e-12:
            continue
        if r.get("label") not in ("OPEN", "BALLISTIC", "PARK"):
            if r["t"] < t_off + 1e-9:
                pass
            elif r.get("label") not in ("OPEN", "BALLISTIC"):
                continue
        po = np.asarray(r["p_obj"], float)
        ph = np.asarray(r["p_hand"], float)
        rh = np.asarray(r["rh"], float)
        vo = np.asarray(r["v_obj"], float)
        item = {
            "t": float(r["t"]),
            "dt": float(r["t"] - t_off),
            "ang_deg": float(r["ang_deg"]),
            "d_ang": float(r["ang_deg"] - ang0),
            "p_obj": po.tolist(),
            "p_hand": ph.tolist(),
            "dp_obj": (po - p_obj0).tolist(),
            "dp_hand": (ph - p_hand0).tolist(),
            "rh_mm": (1e3 * rh).tolist(),
            "v_obj": vo.tolist(),
            "v_rel": float(r["v_rel"]),
            "v_rel_h": r["v_rel_h"],
            "aperture": float(r["aperture"]),
            "nL": int(r["nL"]),
            "nR": int(r["nR"]),
            "omega_hand": float(r.get("omega_axis", 0.0)),
            "ori_err_deg": float(r.get("ori_err_deg", 0.0)),
            "in_corridor": in_corridor(rh),
            "rel_ori_deg": rel_ori_deg(r.get("R_rel", np.eye(3))),
        }
        rows.append(item)
        if vz_prev is not None and vz_prev > 0.0 and vo[2] <= 0.0 and apex is None:
            apex = item
        vz_prev = vo[2]
        if r["t"] > t_off + FLIGHT_S:
            break
    if apex is None and rows:
        apex = max(rows, key=lambda x: x["p_obj"][2])
    return rows, apex


def release_trial(sim, gains, restore_fn, ic_name, phase_name, open_deg):
    restore_fn()
    ic_rh = np.asarray(physical_pack(sim)["rh"], float).copy()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    log = []
    m0 = measure(sim, omega_y=0.0, tau=TAU_SEC, label="PARENT", Rh0=Rh0)
    enrich(sim, m0, gains)
    append_log(log, m0)
    ev = {
        "ic": ic_name,
        "phase": phase_name,
        "open_cmd_deg": float(open_deg),
        "ic_rh": ic_rh.tolist(),
        "ic_rh_mm": (1e3 * ic_rh).tolist(),
        "ROTATE_START": float(sim.data.time),
    }
    timeout = float(sim.data.time) + np.deg2rad(MAX_DEG) / OMEGA4 + 1.2
    while float(sim.data.time) < timeout:
        ang = signed_rot_about_y_deg(
            Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        )
        if abs(ang) >= abs(open_deg) - 0.8:
            break
        m, stop = step_ovr(sim, gains, OMEGA4, 0.0, 0.0, TAU_SEC, log, Rh0, "SWING", ev)
        if stop or (m["nL"] == 0 and m["nR"] == 0):
            ev["LOST_BEFORE_OPEN"] = True
            break
    if ev.get("LOST_BEFORE_OPEN"):
        return {"ic": ic_name, "phase": phase_name, "lost": True, "events": ev, "log": log}

    ev["OPEN_COMMAND"] = float(sim.data.time)
    ev["theta_OPEN_COMMAND"] = float(log[-1]["ang_deg"])
    ev["rh_OPEN_COMMAND"] = log[-1]["rh"]
    ev["rh_OPEN_COMMAND_mm"] = (1e3 * np.asarray(log[-1]["rh"], float)).tolist()
    dt = float(sim.model.opt.timestep)
    both = 0.0
    t_open = float(sim.data.time)
    prev = log[-1]
    last_finger = None
    while float(sim.data.time) < t_open + 0.45:
        m, stop = step_ovr(sim, gains, OMEGA4, 0.0, 0.0, TAU_OPEN, log, Rh0, "OPEN", ev)
        if stop:
            break
        if "FIRST_SINGLE_OFF" not in ev:
            if (m["nL"] == 0) != (m["nR"] == 0):
                ev["FIRST_SINGLE_OFF"] = m["t"]
                ev["theta_FIRST_SINGLE_OFF"] = float(m["ang_deg"])
                if m["nL"] == 0 and prev["nL"] > 0:
                    last_finger = "left"
                if m["nR"] == 0 and prev["nR"] > 0:
                    last_finger = "right"
        if m["nL"] == 0 and m["nR"] == 0:
            if "FIRST_BOTH_OFF" not in ev:
                ev["FIRST_BOTH_OFF"] = m["t"]
                ev["theta_FIRST_BOTH_OFF"] = float(m["ang_deg"])
                ev["open_latency_s"] = float(m["t"] - ev["OPEN_COMMAND"])
                if last_finger is None:
                    if prev["nL"] > 0 and prev["nR"] == 0:
                        last_finger = "left"
                    elif prev["nR"] > 0 and prev["nL"] == 0:
                        last_finger = "right"
                    elif prev["nL"] > 0:
                        last_finger = "left"
                    elif prev["nR"] > 0:
                        last_finger = "right"
                ev["last_contact_finger"] = last_finger
                ev["both_off"] = slim_row(m)
            both += dt
            if both >= BOTH_OFF_MIN:
                break
        else:
            both = 0.0
        prev = m

    if "FIRST_BOTH_OFF" not in ev:
        return {"ic": ic_name, "phase": phase_name, "lost": False, "no_both_off": True, "events": ev, "log": log}

    bo = None
    for r in log:
        if abs(r["t"] - ev["FIRST_BOTH_OFF"]) < 1e-9 or (
            r["t"] >= ev["FIRST_BOTH_OFF"] - 1e-12 and r["nL"] == 0 and r["nR"] == 0
        ):
            bo = r
            break
    if bo is None:
        bo = log[-1]
    p_obj0 = np.asarray(bo["p_obj"], float)
    p_hand0 = np.asarray(bo["p_hand"], float)
    ang0 = float(bo["ang_deg"])
    z0 = float(bo["obj_z"])
    t_off = float(ev["FIRST_BOTH_OFF"])

    t_lim = t_off + FLIGHT_S
    vz_prev = float(bo["v_obj"][2])
    while float(sim.data.time) < t_lim:
        m, stop = step_ovr(sim, gains, 0.0, 0.0, 0.0, TAU_OPEN, log, Rh0, "BALLISTIC", ev)
        if stop:
            break
        vz = float(m["v_obj"][2])
        if vz_prev > 0.0 and vz <= 0.0 and "BALLISTIC_APEX" not in ev:
            ev["BALLISTIC_APEX"] = m["t"]
        vz_prev = vz

    series, apex = flight_series(log, t_off, p_obj0, p_hand0, ang0)
    z_max = max((s["p_obj"][2] for s in series), default=z0)
    rh_off = np.asarray(bo["rh"], float)
    rh_apex = None if apex is None else np.asarray(apex["rh_mm"], float) / 1e3
    dp_obj_apex = None if apex is None else apex["dp_obj"]
    dp_hand_apex = None if apex is None else apex["dp_hand"]
    hand_rot = None if apex is None else apex["d_ang"]

    rel = {
        "ic": ic_name,
        "phase": phase_name,
        "open_cmd_deg": float(open_deg),
        "mass": float(CYL_MASS),
        "omega_cmd": OMEGA4,
        "ic_rh_mm": (1e3 * ic_rh).tolist(),
        "theta_OPEN_COMMAND": ev.get("theta_OPEN_COMMAND"),
        "theta_FIRST_SINGLE_OFF": ev.get("theta_FIRST_SINGLE_OFF"),
        "theta_FIRST_BOTH_OFF": ev.get("theta_FIRST_BOTH_OFF"),
        "open_latency_s": ev.get("open_latency_s"),
        "rh_OPEN_COMMAND_mm": ev.get("rh_OPEN_COMMAND_mm"),
        "rh_BOTH_OFF_mm": (1e3 * rh_off).tolist(),
        "last_contact_finger": ev.get("last_contact_finger"),
        "release": {
            "t": bo["t"],
            "ang": float(bo["ang_deg"]),
            "rh": rh_off.tolist(),
            "v_obj_world": bo["v_obj"],
            "v_obj_hand": bo["v_obj_h"],
            "v_rel_hand": bo["v_rel_h"],
            "v_rel": bo["v_rel"],
            "w_obj": bo["w_obj"],
            "w_hand": bo["w_hand"],
            "omega_hand": bo.get("omega_axis"),
            "R_rel": bo.get("R_rel"),
            "rel_ori_deg": rel_ori_deg(bo.get("R_rel", np.eye(3))),
            "aperture": bo["aperture"],
            "nL": bo["nL"],
            "nR": bo["nR"],
            "p_obj": bo["p_obj"],
            "p_hand": bo["p_hand"],
            "ori_err_deg": bo.get("ori_err_deg"),
        },
        "ballistic": {
            "z_both_off": z0,
            "z_max": z_max,
            "rise_after_both_off": float(z_max - z0),
            "t_apex": ev.get("BALLISTIC_APEX"),
            "dt_apex": None if ev.get("BALLISTIC_APEX") is None else float(ev["BALLISTIC_APEX"] - t_off),
            "apex": apex,
            "in_corridor_at_both_off": in_corridor(rh_off),
            "in_corridor_at_apex": None if rh_apex is None else in_corridor(rh_apex),
            "rh_apex_mm": None if apex is None else apex["rh_mm"],
            "dp_obj_apex_mm": None if dp_obj_apex is None else (1e3 * np.asarray(dp_obj_apex)).tolist(),
            "dp_hand_apex_mm": None if dp_hand_apex is None else (1e3 * np.asarray(dp_hand_apex)).tolist(),
            "hand_ang_change_to_apex": hand_rot,
            "omega_hand_at_apex": None if apex is None else apex["omega_hand"],
        },
        "events": {k: v for k, v in ev.items() if k != "both_off"},
        "flight": series[::2],
        "log": log,
    }
    return rel


def catchable(trial) -> dict:
    rel = trial["release"]
    b = trial["ballistic"]
    vz = float(rel["v_obj_world"][2])
    apex = b.get("apex") or {}
    rh_a = None if not apex else np.asarray(apex["rh_mm"], float) / 1e3
    geo = bool(b.get("in_corridor_at_apex"))
    vz_ok = vz > 0.20
    dt = b.get("dt_apex")
    time_ok = dt is not None and dt >= 0.020
    rhx_ok = rh_a is not None and abs(rh_a[0]) <= 0.012
    notes = {
        "useful_vz": vz_ok,
        "vz": vz,
        "contact_free_to_apex_s": dt,
        "time_ok": time_ok,
        "in_corridor_apex": geo,
        "rhx_apex_mm": None if rh_a is None else 1e3 * float(rh_a[0]),
        "rhx_apex_tight": rhx_ok,
        "hand_ang_change_to_apex": b.get("hand_ang_change_to_apex"),
        "dp_obj_apex_mm": b.get("dp_obj_apex_mm"),
        "dp_hand_apex_mm": b.get("dp_hand_apex_mm"),
    }
    notes["clearly_catchable"] = bool(vz_ok and time_ok and geo and rhx_ok)
    return notes


def mark_event(ctl, name, m=None, extra=""):
    """Print/overlay a transition. Does not pause or change dynamics."""
    if ctl is None:
        return
    seen = ctl.setdefault("_announced", set())
    if name in seen:
        return
    seen.add(name)
    ctl["last_event"] = name
    t = "" if m is None else f"t={float(m['t']):.4f}"
    nlr = "" if m is None else f"nL={m['nL']} nR={m['nR']}"
    ang = "" if m is None else f"ang={float(m['ang_deg']):.1f}"
    rhx = "" if m is None else f"rhx_mm={1e3 * float(m['rh'][0]):+.2f}"
    print(f"EVENT {name} {t} {nlr} {ang} {rhx} {extra}".strip(), flush=True)


def recatch_one(sim, gains, restore_fn, open_deg, t_close, ctl=None, hold_s=2.0):
    restore_fn()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    log = []
    m0 = measure(sim, omega_y=0.0, tau=TAU_SEC, label="PARENT", Rh0=Rh0)
    enrich(sim, m0, gains)
    append_log(log, m0)
    overlay(ctl, m0, "ROTATE")
    ev = {"ROTATE": float(sim.data.time)}
    mark_event(ctl, "ROTATE", m0)
    timeout = float(sim.data.time) + np.deg2rad(MAX_DEG) / OMEGA4 + 1.2
    while float(sim.data.time) < timeout:
        ang = signed_rot_about_y_deg(
            Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        )
        if abs(ang) >= abs(open_deg) - 0.8:
            break
        m, stop = step_ovr(sim, gains, OMEGA4, 0.0, 0.0, TAU_SEC, log, Rh0, "ROTATE", ev, ctl)
        if stop:
            break
    ev["OPEN_COMMAND"] = float(sim.data.time)
    mark_event(ctl, "OPEN_COMMAND", log[-1])
    t_open = float(sim.data.time)
    both = 0.0
    dt = float(sim.model.opt.timestep)
    vz_prev = float(log[-1]["v_obj"][2])
    while float(sim.data.time) < t_open + 0.45:
        m, stop = step_ovr(sim, gains, OMEGA4, 0.0, 0.0, TAU_OPEN, log, Rh0, "OPEN", ev, ctl)
        if stop:
            break
        if m["nL"] == 0 and m["nR"] == 0:
            if "FIRST_BOTH_OFF" not in ev:
                ev["FIRST_BOTH_OFF"] = m["t"]
                mark_event(ctl, "FIRST_BOTH_OFF", m)
            both += dt
            if both >= BOTH_OFF_MIN:
                break
        else:
            both = 0.0
        vz_prev = float(m["v_obj"][2])
    closed = False
    captured = False
    hold_ok = False
    t_end = float(sim.data.time) + 1.4
    lost = 0.0
    while float(sim.data.time) < t_end:
        tau = TAU_OPEN
        lab = "BALLISTIC"
        if (not closed) and t_close is not None and float(sim.data.time) >= float(t_close):
            closed = True
            ev["CLOSE_COMMAND"] = float(sim.data.time)
        if closed:
            tau = TAU_SEC
            lab = "CLOSE" if "HOLD_START" not in ev else "HOLD"
        m, stop = step_ovr(sim, gains, 0.0, 0.0, 0.0, tau, log, Rh0, lab, ev, ctl)
        if stop:
            break
        if closed and "CLOSE_COMMAND" in ev:
            mark_event(ctl, "CLOSE_COMMAND", m)
        vz = float(m["v_obj"][2])
        if vz_prev > 0.0 and vz <= 0.0 and "APEX" not in ev:
            ev["APEX"] = m["t"]
            mark_event(ctl, "APEX", m)
        vz_prev = vz
        if closed:
            if (m["nL"] > 0 or m["nR"] > 0) and "FIRST_RECONTACT" not in ev:
                ev["FIRST_RECONTACT"] = m["t"]
                mark_event(ctl, "FIRST_RECONTACT", m)
            if m["nL"] > 0 and m["nR"] > 0:
                if "FIRST_BILATERAL" not in ev:
                    ev["FIRST_BILATERAL"] = m["t"]
                    mark_event(ctl, "FIRST_BILATERAL", m)
                    lost = 0.0
                elif "HOLD_START" not in ev and m["t"] - ev["FIRST_BILATERAL"] >= 0.050:
                    if m["nL"] > 0 and m["nR"] > 0:
                        ev["HOLD_START"] = m["t"]
                        mark_event(ctl, "HOLD_START", m)
                        captured = True
                        t_end = m["t"] + float(hold_s)
            if "HOLD_START" in ev:
                if m["nL"] == 0 or m["nR"] == 0:
                    lost += dt
                    if lost >= 0.08:
                        ev["HOLD_LOST"] = m["t"]
                        captured = False
                        mark_event(ctl, "HOLD_LOST", m)
                        break
                else:
                    lost = 0.0
                if m["t"] + 1e-12 >= ev["HOLD_START"] + float(hold_s):
                    ev["HOLD_COMPLETE"] = m["t"]
                    mark_event(ctl, "HOLD_COMPLETE", m)
    if captured and "HOLD_LOST" not in ev:
        end = log[-1]
        hold_ok = (
            int(end["nL"]) > 0
            and int(end["nR"]) > 0
            and float(end.get("v_rel", 99)) < 0.08
            and abs(float(end["rh"][0])) < 0.020
        )
    return {
        "captured": captured,
        "ok_hold": hold_ok,
        "events": ev,
        "end": slim_row(log[-1]),
        "log": log,
        "t_close": t_close,
    }


CLAIMED_OPEN_DEG = 70.0
CLAIMED_JSON = RAW / "release_map_recatch.json"
CLAIMED_TIMES = {
    "FIRST_BOTH_OFF": 3.702,
    "CLOSE_COMMAND": 3.722,
    "FIRST_BILATERAL": 3.738,
    "HOLD_START": 3.790,
}


def claimed_t_close() -> float:
    if not CLAIMED_JSON.is_file():
        raise FileNotFoundError(CLAIMED_JSON)
    rec = json.loads(CLAIMED_JSON.read_text(encoding="utf-8"))
    return float(rec["t_close"])


def throw_open_no_close(sim, gains, restore_fn, open_deg, ctl=None, flight_s=0.25):
    """Matched ω=4 OPEN, then PARK (ω=0, still open). No CLOSE."""
    restore_fn()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    log = []
    m0 = measure(sim, omega_y=0.0, tau=TAU_SEC, label="PARENT", Rh0=Rh0)
    enrich(sim, m0, gains)
    append_log(log, m0)
    overlay(ctl, m0, "ROTATE")
    ev = {"ROTATE": float(sim.data.time)}
    mark_event(ctl, "ROTATE", m0)
    timeout = float(sim.data.time) + np.deg2rad(MAX_DEG) / OMEGA4 + 1.2
    while float(sim.data.time) < timeout:
        ang = signed_rot_about_y_deg(
            Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        )
        if abs(ang) >= abs(open_deg) - 0.8:
            break
        m, stop = step_ovr(sim, gains, OMEGA4, 0.0, 0.0, TAU_SEC, log, Rh0, "ROTATE", ev, ctl)
        if stop:
            break
    ev["OPEN_COMMAND"] = float(sim.data.time)
    mark_event(ctl, "OPEN_COMMAND", log[-1])
    t_open = float(sim.data.time)
    both = 0.0
    dt = float(sim.model.opt.timestep)
    vz_prev = float(log[-1]["v_obj"][2])
    while float(sim.data.time) < t_open + 0.45:
        m, stop = step_ovr(sim, gains, OMEGA4, 0.0, 0.0, TAU_OPEN, log, Rh0, "OPEN", ev, ctl)
        if stop:
            break
        if m["nL"] == 0 and m["nR"] == 0:
            if "FIRST_BOTH_OFF" not in ev:
                ev["FIRST_BOTH_OFF"] = m["t"]
                mark_event(ctl, "FIRST_BOTH_OFF", m)
            both += dt
            if both >= BOTH_OFF_MIN:
                break
        else:
            both = 0.0
        vz_prev = float(m["v_obj"][2])
    t_lim = float(sim.data.time) + float(flight_s)
    while float(sim.data.time) < t_lim:
        m, stop = step_ovr(sim, gains, 0.0, 0.0, 0.0, TAU_OPEN, log, Rh0, "BALLISTIC", ev, ctl)
        if stop:
            break
        vz = float(m["v_obj"][2])
        if vz_prev > 0.0 and vz <= 0.0 and "APEX" not in ev:
            ev["APEX"] = m["t"]
            mark_event(ctl, "APEX", m)
        vz_prev = vz
    return {"events": ev, "end": slim_row(log[-1]), "log": log}


def _early_restore_pair():
    if not EARLY_PKL.is_file():
        raise FileNotFoundError(EARLY_PKL)
    with EARLY_PKL.open("rb") as f:
        early = pickle.load(f)
    sim_e, _ = make_ballistic_sim()

    def restore_off():
        restore_ballistic(sim_e, early)
        if hasattr(sim_e, "park_ball"):
            sim_e.park_ball()
        mujoco.mj_forward(sim_e.model, sim_e.data)
        freeze(sim_e)
        disable_object_table_only(sim_e)

    return sim_e, restore_off


def verify_claimed_replay(gains) -> dict:
    """Replay the saved EARLY/70°/close construction. Not a new experiment."""
    rec = json.loads(CLAIMED_JSON.read_text(encoding="utf-8"))
    t_close = float(rec["t_close"])
    sim_e, restore_off = _early_restore_pair()
    out = recatch_one(sim_e, gains, restore_off, CLAIMED_OPEN_DEG, t_close, ctl=None)
    ev = out["events"]
    exp_ev = rec["events"]
    tol = 0.0021
    mismatches = []
    for k, exp in (
        ("FIRST_BOTH_OFF", exp_ev["FIRST_BOTH_OFF"]),
        ("CLOSE_COMMAND", exp_ev["CLOSE_COMMAND"]),
        ("FIRST_BILATERAL", exp_ev["FIRST_BILATERAL"]),
        ("HOLD_START", exp_ev["HOLD_START"]),
    ):
        got = ev.get(k)
        if got is None or abs(float(got) - float(exp)) > tol:
            mismatches.append({"key": k, "expected": exp, "got": got})
    end = out["end"]
    hold_s = None
    if ev.get("HOLD_START") is not None:
        hold_s = float(end["t"]) - float(ev["HOLD_START"])
    if abs(hold_s or 0) < 1.99:
        mismatches.append({"key": "hold_duration", "expected": 2.002, "got": hold_s})
    if abs(1e3 * float(end["rh"][0]) - (-0.2449)) > 0.15:
        mismatches.append({"key": "rhx_mm", "expected": -0.24, "got": 1e3 * float(end["rh"][0])})
    if abs(float(end["v_rel"]) - 4e-4) > 5e-4:
        mismatches.append({"key": "v_rel", "expected": 4e-4, "got": end["v_rel"]})
    if abs(float(end["tau_cmd"]) - (-18.0)) > 1e-6:
        mismatches.append({"key": "tau", "expected": -18.0, "got": end["tau_cmd"]})
    if not out.get("ok_hold"):
        mismatches.append({"key": "ok_hold", "expected": True, "got": out.get("ok_hold")})
    return {
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
        "events": {k: ev.get(k) for k in (
            "ROTATE", "OPEN_COMMAND", "FIRST_BOTH_OFF", "APEX", "CLOSE_COMMAND",
            "FIRST_RECONTACT", "FIRST_BILATERAL", "HOLD_START", "HOLD_COMPLETE",
        )},
        "end_t": end["t"],
        "hold_s": hold_s,
        "rhx_mm": 1e3 * float(end["rh"][0]),
        "v_rel": end["v_rel"],
        "tau": end["tau_cmd"],
        "ok_hold": out.get("ok_hold"),
        "t_close_used": t_close,
        "open_deg": CLAIMED_OPEN_DEG,
    }


def interactive_claimed(mode: str) -> None:
    import mujoco.viewer
    import time

    from training.demo_airborne_recovery import apply_masks, copy_masks, prepare_high_parent
    from training.grav_reposition_v2_viz import apply_camera_preset
    from training.map_ballistic_disturbance import CAM, _viewer_overlay

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    gains.setdefault("kp_null", 4.0)
    gains.setdefault("kd_null", 0.8)

    chk = verify_claimed_replay(gains)
    dump(RAW / "claimed_replay_verify.json", chk)
    print("CLAIMED_REPLAY_VERIFY", chk["ok"], chk, flush=True)
    if not chk["ok"]:
        print("STOP: viewer replay does not match the saved claimed run. Not launching a different trajectory.", flush=True)
        return

    recatch = mode == "offset_throw_recatch_omega4"
    t_close = claimed_t_close() if recatch else None
    ctl = {"pause": False, "reset": False, "speed": 0.12, "overlay": {}, "label": mode, "_announced": set()}

    def sync():
        vwr = ctl.get("viewer")
        if vwr is None:
            return
        o = ctl.get("overlay") or {}
        _viewer_overlay(
            vwr,
            [
                ("CASE", str(o.get("case", mode))),
                ("EVENT", str(ctl.get("last_event", o.get("event", "")))),
                ("t", f"{float(o.get('t', 0)):.3f} s"),
                ("PHASE", str(o.get("phase", ""))),
                ("nL/nR", f"{o.get('nL','-')}/{o.get('nR','-')}"),
                ("ang", f"{float(o.get('ang', 0)):.0f} deg"),
                ("rhx_mm", f"{float(o.get('rhx_mm', 0)):+.2f}"),
                ("v_obj,z", f"{float(o.get('vz', 0)):+.3f} m/s"),
                ("keys", "SPACE pause  R restart  [ ] speed"),
            ],
        )
        vwr.sync()
        dt = 0.002
        spd = max(float(ctl.get("speed", 0.12)), 0.04)
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
            ctl["speed"] = max(0.04, float(ctl["speed"]) * 0.7)
        elif k == ord("]"):
            ctl["speed"] = min(2.5, float(ctl["speed"]) / 0.7)

    if recatch:
        sim, restore_fn = _early_restore_pair()
        parent = None
        orig = None
    else:
        sim = make_parent_sim()
        orig = copy_masks(sim)
        parent = {"snap": None}

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        ctl["viewer"] = vwr
        apply_camera_preset(vwr, CAM)
        while vwr.is_running():
            ctl["reset"] = False
            ctl["_wall"] = None
            ctl["_announced"] = set()
            ctl["last_event"] = ""
            if recatch:
                recatch_one(sim, gains, restore_fn, CLAIMED_OPEN_DEG, t_close, ctl=ctl)
            else:
                apply_masks(sim, orig)
                sim.reset(CYL_MASS, PAIR_MU, np.zeros(3))
                prep = prepare_high_parent(sim, cfg, gains, orig, ctl=ctl)
                if not prep.get("ok"):
                    time.sleep(0.2)
                    continue

                def restore_cen():
                    sim.load_snapshot(prep["parent"])
                    mujoco.mj_forward(sim.model, sim.data)
                    freeze(sim)
                    disable_object_table_only(sim)

                throw_open_no_close(sim, gains, restore_cen, CLAIMED_OPEN_DEG, ctl=ctl)
            if ctl.get("reset"):
                continue
            while vwr.is_running() and not ctl.get("reset"):
                sync()
                time.sleep(0.03)
            if not ctl.get("reset"):
                break


def fmt_rel(trial):
    r = trial["release"]
    b = trial["ballistic"]
    return (
        f"| {trial['ic']} | {trial['phase']} | {trial['open_cmd_deg']:.1f} | "
        f"{trial.get('theta_OPEN_COMMAND')} | {trial.get('theta_FIRST_SINGLE_OFF')} | "
        f"{trial.get('theta_FIRST_BOTH_OFF')} | {trial.get('open_latency_s')} | "
        f"{trial.get('last_contact_finger')} | {np.round(trial['ic_rh_mm'], 2).tolist()} | "
        f"{np.round(trial['rh_OPEN_COMMAND_mm'], 2).tolist()} | "
        f"{np.round(trial['rh_BOTH_OFF_mm'], 2).tolist()} | "
        f"{float(r['v_obj_world'][2]):+.3f} | {np.round(r['v_obj_world'], 3).tolist()} | "
        f"{np.round(r['v_rel_hand'], 3).tolist()} | {float(np.linalg.norm(r['w_obj'])):.2f} | "
        f"{r['rel_ori_deg']:.1f} | {r['aperture']:.4f} | {r['omega_hand']:.3f} | "
        f"{None if b['dt_apex'] is None else round(b['dt_apex'], 3)} | "
        f"{1e3*b['rise_after_both_off']:.1f} | {b['rh_apex_mm']} | "
        f"{b['in_corridor_at_apex']} | {b['hand_ang_change_to_apex']} |"
    )


def write_report(ics, samples, park, trials, recatch_row, candidate):
    a = []
    A = a.append
    A("# Release-state map at diagnostic ω_y = 4 rad/s")
    A("")
    A("Mass **held fixed at 0.20 kg**. Friction, controller, grip map, noslip, and `v_x`/`v_z` bounds unchanged.")
    A("`RECOVERY4D_W_HY_MAX` **not** changed. `omega_y_cmd = +4.0` is a diagnostic override.")
    A("No recatch during mapping. Offset is evaluated in the **full release state**, not by `v_obj,z` alone.")
    A("")
    A("Geometric corridor (open-gripper pocket, not a score): `|r_h.x|<22 mm`, `|r_h.y|<20 mm`, `50 < r_h.z < 130 mm`.")
    A("")
    A("## 1. Initial-state provenance")
    A("")
    for ic in ics:
        A(f"### {ic['name']}")
        A(f"- {ic.get('provenance', '')}")
        A(f"- t={ic['t']}, mass={ic['mass']}, nL/nR={ic['nL']}/{ic['nR']} bilateral={ic['bilateral']}")
        A(f"- initial r_h mm = {ic['rh_mm']}")
        A(f"- aperture={ic['aperture']}, v_rel={ic['v_rel']}, obj_z={ic['obj_z']}")
        A("")
    A("`snap_mid.pkl` exists on the same impact trajectory but was **not** added: the map is centered × EARLY only (six trials).")
    A("")
    A("## 2. Supported trajectory near the upward-velocity peak (centered, tau=-18, ω=4)")
    A("")
    A("| ang | v_obj,z | v_obj | r_h mm | v_rel_h | |w_obj| | nL/nR | ap | Fn L/R |")
    A("|---|---|---|---|---|---|---|---|---|")
    for s in samples:
        A(
            f"| {s['ang_deg']:.1f} | {s['v_obj_z']:+.3f} | {np.round(s['v_obj'], 3).tolist()} | "
            f"{np.round(s['rh_mm'], 2).tolist()} | {np.round(s['v_rel_h'], 3).tolist()} | "
            f"{float(np.linalg.norm(s['w_obj'])):.2f} | {s['nL']}/{s['nR']} | {s['aperture']:.4f} | "
            f"{s['Fn_L']:.2f}/{s['Fn_R']:.2f} |"
        )
    A("")
    A("OPEN phases (from this log, not a dense sweep): current 76.9° (previous command, BOTH_OFF ~79° near peak), modestly earlier 70°, clearly earlier 60°.")
    A("12 ms latency ≈ 2.7° at 4 rad/s; contact dynamics may need a different lead, which is why earlier commands are included.")
    A("")
    A("## 3–4. FIRST_BOTH_OFF complete states")
    A("")
    A("| IC | phase | cmd | θ_OPEN | θ_SINGLE | θ_BOTH | lat s | last finger | r_h IC mm | r_h OPEN mm | r_h BOTH mm | v_z | v_obj | v_rel_h | |ω_obj| | rel° | ap | ω_hand | dt_apex | rise mm | r_h apex mm | corridor apex | Δhand° to apex |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in trials:
        A(fmt_rel(t))
    A("")
    A("Do not rank by `v_obj,z` alone. Initial offset is **not** assumed equal to release offset.")
    A("")
    A("## 5–8. World object vs hand vs relative (`r_h`) at apex")
    A("")
    A("If `dp_obj` is small in a world axis but `r_h` changes, the corridor moved. Do not call that “the cylinder flew sideways” without world-frame evidence.")
    A("")
    for t in trials:
        b = t["ballistic"]
        A(
            f"- **{t['ic']}/{t['phase']}**: BOTH_OFF r_h mm={np.round(t['rh_BOTH_OFF_mm'], 2).tolist()}; "
            f"apex r_h mm={b['rh_apex_mm']}; dp_obj mm={None if b['dp_obj_apex_mm'] is None else np.round(b['dp_obj_apex_mm'], 2).tolist()}; "
            f"dp_hand mm={None if b['dp_hand_apex_mm'] is None else np.round(b['dp_hand_apex_mm'], 2).tolist()}; "
            f"hand Δθ={b['hand_ang_change_to_apex']}; ω_hand apex={b['omega_hand_at_apex']}; "
            f"corridor apex={b['in_corridor_at_apex']}; rise={1e3*b['rise_after_both_off']:.1f} mm; dt_apex={b['dt_apex']}."
        )
    A("")
    A("## 9. PARK transient (secure, tau=-18, ω_cmd→0 at current OPEN phase 76.9°)")
    A("")
    A(f"- ang at PARK = {park.get('ang_at_park')}")
    A(f"- ang after 100 ms = {park.get('ang_after_100ms')}")
    A(f"- **Δθ in 100 ms = {park.get('delta_ang_deg')} deg**")
    A(f"- ω_axis at PARK = {park.get('omega_at_park')}, after 100 ms = {park.get('omega_after_100ms')}")
    A(f"- SO(3) err at PARK = {park.get('ori_err_at_park')}°, after 100 ms = {park.get('ori_err_after_100ms')}°")
    A("Controller not modified. This lag is part of the capture-corridor motion during the ~38 ms ballistic window.")
    A("")
    A("## 10–11. Catchability (separate quantities, no weighted score)")
    A("")
    A("| IC | phase | vz>0.20 | vz | dt_apex≥20ms | corridor apex | |r_h.x| apex ≤12 mm | clearly catchable |")
    A("|---|---|---|---|---|---|---|---|")
    flags = []
    for t in trials:
        n = catchable(t)
        flags.append((t, n))
        A(
            f"| {t['ic']} | {t['phase']} | {n['useful_vz']} | {n['vz']:+.3f} | {n['time_ok']} | "
            f"{n['in_corridor_apex']} | {n['rhx_apex_tight']} ({n['rhx_apex_mm']}) | {n['clearly_catchable']} |"
        )
    A("")
    A("## 12. Selected candidate")
    A("")
    if candidate is None:
        A("**None.** No (IC × phase) was both ballistically useful and geometrically catchable at apex under the stated corridor. Privileged recatch **not** run.")
        A("Causal structure is in the table: phase and physical offset both move `r_h` at release and during flight; `v_obj,z` is only one coordinate.")
    else:
        A(f"Selected: **{candidate['ic']} / {candidate['phase']}** (open cmd {candidate['open_cmd_deg']}°).")
        A("Lexicographic: require clearly_catchable, then prefer larger `v_obj,z`. Not a blended score.")
    A("")
    A("## 13. Privileged recatch")
    A("")
    if recatch_row is None:
        A("Not run.")
    else:
        A("**PRIVILEGED.** CLOSE_COMMAND from GT of the mapped free-flight (not assumed equal to apex). Hold at catch orientation ≥2 s. No return-to-nominal.")
        A(f"CLOSE_COMMAND={recatch_row.get('t_close')} FIRST_RECONTACT={recatch_row.get('events', {}).get('FIRST_RECONTACT')} FIRST_BILATERAL={recatch_row.get('events', {}).get('FIRST_BILATERAL')}")
        A(f"captured={recatch_row.get('captured')} ok_hold={recatch_row.get('ok_hold')}")
        A(f"events={ {k: recatch_row.get('events', {}).get(k) for k in ('OPEN_COMMAND','FIRST_BOTH_OFF','CLOSE_COMMAND','FIRST_RECONTACT','FIRST_BILATERAL','HOLD_START','HOLD_LOST','FLOOR') if k in (recatch_row.get('events') or {})} }")
        end = recatch_row.get("end") or {}
        A(f"end nL/nR={end.get('nL')}/{end.get('nR')} v_rel={end.get('v_rel')} rh={end.get('rh')}")
    A("")
    A("No ω>4, no mass/friction change, no synthetic offsets, no SAC, no permanent bound change, no MP4.")
    REPORT.write_text("\n".join(a), encoding="utf-8")
    print("wrote", REPORT, flush=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    gains.setdefault("kp_null", 4.0)
    gains.setdefault("kd_null", 0.8)

    sim_c = make_parent_sim()
    orig = copy_masks(sim_c)
    prep = prepare_high_parent(sim_c, cfg, gains, orig)
    if not prep.get("ok"):
        raise RuntimeError(prep)
    parent = prep["parent"]
    mass = float(sim_c.model.body_mass[sim_c.ids.object_body])
    assert abs(mass - float(CYL_MASS)) < 1e-9

    def restore_cen():
        sim_c.load_snapshot(parent)
        mujoco.mj_forward(sim_c.model, sim_c.data)
        freeze(sim_c)
        disable_object_table_only(sim_c)

    restore_cen()
    ic_c = snapshot_ic(
        sim_c,
        "centered",
        extra={
            "provenance": "High-clearance physically lifted parent (same construction as omega=4 throw). Not teleported.",
        },
    )

    if not EARLY_PKL.is_file():
        raise FileNotFoundError(EARLY_PKL)
    with EARLY_PKL.open("rb") as f:
        early = pickle.load(f)
    sim_e, _ = make_ballistic_sim()

    def restore_off():
        restore_ballistic(sim_e, early)
        if hasattr(sim_e, "park_ball"):
            sim_e.park_ball()
        mujoco.mj_forward(sim_e.model, sim_e.data)
        freeze(sim_e)
        disable_object_table_only(sim_e)

    restore_off()
    ic_e = snapshot_ic(
        sim_e,
        "center6_early",
        extra={
            "provenance": (
                f"Physically generated CENTER-6 EARLY `{EARLY_PKL}` from the impact/ballistic trajectory "
                f"(documented r_h.x={EARLY_DOC_RHX_MM} mm). Not a synthetic offset grid."
            ),
            "doc_rhx_mm": EARLY_DOC_RHX_MM,
        },
    )
    dump(RAW / "release_map_ics.json", {"centered": ic_c, "early": ic_e, "mass": mass, "omega_cmd": OMEGA4, "bound_unchanged": OMEGA3})
    print("IC centered rh_mm", ic_c["rh_mm"], "early rh_mm", ic_e["rh_mm"], "mass", mass, flush=True)

    slog, peak, _ = swing_supported(sim_c, gains, restore_cen)
    degs = (50.0, 60.0, 70.0, 75.0, 79.5, 85.0, 90.0)
    samples = supported_samples(slog, degs)
    dump(
        RAW / "release_map_supported.json",
        {
            "peak_ang": None if peak is None else float(peak["ang_deg"]),
            "peak_vz": None if peak is None else float(peak["v_obj"][2]),
            "samples": samples,
        },
    )
    print("supported peak", None if peak is None else (peak["ang_deg"], peak["v_obj"][2]), flush=True)

    park = park_transient(sim_c, gains, restore_cen, 76.9)
    dump(RAW / "release_map_park.json", park)
    print("PARK dtheta_100ms", park.get("delta_ang_deg"), "ori", park.get("ori_err_at_park"), "->", park.get("ori_err_after_100ms"), flush=True)

    ics_run = (("centered", restore_cen, sim_c), ("center6_early", restore_off, sim_e))
    trials = []
    for ic_name, rst, sim in ics_run:
        for pname, odegs in PHASES:
            print("trial", ic_name, pname, odegs, flush=True)
            tr = release_trial(sim, gains, rst, ic_name, pname, odegs)
            slim = {k: v for k, v in tr.items() if k != "log"}
            dump(RAW / f"release_map_{ic_name}_{pname}.json", slim)
            from training.demo_dynamic_recatch import save_npz

            save_npz(RAW / f"release_map_{ic_name}_{pname}.npz", tr["log"])
            trials.append(slim)
            rel = slim.get("release") or {}
            print(
                "  both", slim.get("theta_FIRST_BOTH_OFF"),
                "vz", None if not rel else rel.get("v_obj_world", [None, None, None])[2],
                "rh_off", slim.get("rh_BOTH_OFF_mm"),
                "rh_apex", (slim.get("ballistic") or {}).get("rh_apex_mm"),
                "corr", (slim.get("ballistic") or {}).get("in_corridor_at_apex"),
                "dhand", (slim.get("ballistic") or {}).get("hand_ang_change_to_apex"),
                flush=True,
            )

    cand = None
    best_vz = -1e9
    for t in trials:
        n = catchable(t)
        if n["clearly_catchable"] and n["vz"] > best_vz:
            best_vz = n["vz"]
            cand = t

    recatch_row = None
    if cand is not None:
        rst = restore_cen if cand["ic"] == "centered" else restore_off
        sim = sim_c if cand["ic"] == "centered" else sim_e
        apex_t = (cand.get("ballistic") or {}).get("t_apex")
        # CLOSE before apex by ~20 ms (aperture delay), not assumed equal to apex.
        t_close = None if apex_t is None else float(apex_t) - 0.020
        both_t = cand.get("events", {}).get("FIRST_BOTH_OFF")
        if t_close is not None and both_t is not None:
            t_close = max(float(both_t), t_close)
        print("RECATCH candidate", cand["ic"], cand["phase"], "t_close", t_close, flush=True)
        rec = recatch_one(sim, gains, rst, float(cand["open_cmd_deg"]), t_close)
        recatch_row = {k: v for k, v in rec.items() if k != "log"}
        dump(RAW / "release_map_recatch.json", recatch_row)
        from training.demo_dynamic_recatch import save_npz

        save_npz(RAW / "release_map_recatch.npz", rec["log"])
        print("RECATCH captured", rec.get("captured"), "hold", rec.get("ok_hold"), flush=True)

    write_report([ic_c, ic_e], samples, park, trials, recatch_row, cand)


if __name__ == "__main__":
    main()
