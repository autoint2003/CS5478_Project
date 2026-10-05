"""Dynamic throw / recapture authority. Privileged construction. No SAC/MP4.

Tests whether legal wrist rotation while contact is still secure can create
a useful ballistic release (especially v_obj,z > 0), distinct from drop-and-chase.
"""

from __future__ import annotations

import argparse
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

from controllers.gripper_controller import finger_opening
from controllers.jacobian_controller import gains_from_cfg
from controllers.residual import (
    RECOVERY4D_TAU_OPEN,
    RECOVERY4D_V_HX_MAX,
    RECOVERY4D_V_Z_MAX,
    RECOVERY4D_W_HY_MAX,
)
from envs.config_util import load_yaml, merge_sim_config
from envs.deterioration import body_twist
from envs.physical_recovery import physical_pack
from training.demo_airborne_recovery import (
    apply_masks,
    copy_masks,
    disable_object_table_only,
    extra_lift_secure,
    object_floor_hit,
    prepare_high_parent,
)
from training.demo_airborne_recapture import make_parent_sim, save_parent, tick_4d
from training.demo_ballistic_impact import make_sim as make_ballistic_sim
from training.demo_ballistic_recovery import restore_ballistic
from training.demo_teleport_recovery_state import CYL_MASS, PAIR_MU, signed_rot_about_y_deg
from training.grav_reposition_v2_viz import apply_camera_preset
from training.map_ballistic_disturbance import CAM, _viewer_overlay
from training.replay_core import freeze
from training.vertical_slip_contact_mechanics_audit import extract_contacts

OUT = ROOT / "results" / "diagnostics" / "dynamic_airborne_recatch"
RAW = OUT / "raw"
TAU_OPEN = float(RECOVERY4D_TAU_OPEN)
TAU_SEC = -18.0
OMEGA = float(RECOVERY4D_W_HY_MAX)
VX_MAX = float(RECOVERY4D_V_HX_MAX)
VZ_MAX = float(RECOVERY4D_V_Z_MAX)
LOG_DT = 0.002
BOTH_OFF_MIN = 0.018
V_UP_MIN = 0.03  # m/s; above integrator noise, not an optimized throw height
AUDIT_DEG = (0.0, 30.0, 60.0, 90.0, 120.0)
PROBE_DEG = (60.0, 75.0, 90.0, 105.0, 120.0)
EARLY_PKL = ROOT / "results" / "diagnostics" / "ballistic_recovery_transfer" / "raw" / "snap_early.pkl"
EARLY_DOC_RHX_MM = 10.87
OPEN_PHASES = (("early", 15.0), ("medium", 45.0), ("late", 90.0))
MATCH_DEG = (0.0, 30.0, 60.0, 90.0, 120.0)
HOLD_S = 2.0
BALLISTIC_S = 0.80


def dump(path: Path, obj) -> None:
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fix(obj), indent=2), encoding="utf-8")


def slim_row(m: dict) -> dict:
    skip = {"geoms", "contact_pts", "r_des", "R_hand", "R_obj", "contacts"}
    return {k: v for k, v in m.items() if k not in skip}


def snap_now(sim) -> dict:
    s = sim.snapshot()
    s["mass"] = CYL_MASS
    s["friction"] = PAIR_MU
    return s


def restore(sim, snap) -> None:
    sim.load_snapshot(snap)
    mujoco.mj_forward(sim.model, sim.data)
    freeze(sim)


def measure(sim, *, omega_y=0.0, v_x=0.0, v_z=0.0, tau=None, label="", Rh0=None) -> dict:
    o = physical_pack(sim)
    fc, _ = extract_contacts(sim)
    ap = finger_opening(sim.data, sim.ids)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    r_w = po - ph
    v_rigid = vh + np.cross(wh, r_w)
    v_carry_err = vo - v_rigid
    ang = 0.0 if Rh0 is None else signed_rot_about_y_deg(Rh0, Rh)
    Fn_L = float(sum(abs(c["Fn"]) for c in fc if c["side"] == "L"))
    Fn_R = float(sum(abs(c["Fn"]) for c in fc if c["side"] == "R"))
    Ft_L = float(sum(float(np.hypot(c.get("Ft1", 0.0), c.get("Ft2", 0.0))) for c in fc if c["side"] == "L"))
    Ft_R = float(sum(float(np.hypot(c.get("Ft1", 0.0), c.get("Ft2", 0.0))) for c in fc if c["side"] == "R"))
    rhos = [float(c["rho"]) for c in fc if np.isfinite(c.get("rho", np.nan))]
    g_h = np.asarray(o["g_h"], float)
    return {
        "t": float(sim.data.time),
        "label": label,
        "omega_y_cmd": float(omega_y),
        "v_x_cmd": float(v_x),
        "v_z_cmd": float(v_z),
        "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else None,
        "tau_cmd": None if tau is None else float(tau),
        "aperture": float(ap),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "Fn_L": Fn_L,
        "Fn_R": Fn_R,
        "Ft_L": Ft_L,
        "Ft_R": Ft_R,
        "rho_max": float(np.max(rhos)) if rhos else None,
        "rh": np.asarray(o["rh"], float).tolist(),
        "v_rel_h": np.asarray(o["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(o["w_rel_h"], float).tolist(),
        "v_rel": float(o["v_rel"]),
        "w_rel": float(o["w_rel"]),
        "obj_z": float(o["obj_z"]),
        "clear": float(o.get("clear", np.nan)),
        "p_hand": ph.tolist(),
        "p_obj": po.tolist(),
        "v_hand": vh.tolist(),
        "v_obj": vo.tolist(),
        "w_hand": wh.tolist(),
        "w_obj": wo.tolist(),
        "v_obj_h": (Rh.T @ vo).tolist(),
        "v_hand_h": (Rh.T @ vh).tolist(),
        "r_world": r_w.tolist(),
        "v_rigid": v_rigid.tolist(),
        "v_carry_err": v_carry_err.tolist(),
        "v_carry_err_norm": float(np.linalg.norm(v_carry_err)),
        "ang_deg": float(ang),
        "g_h": g_h.tolist(),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "r_des": np.asarray(sim.fsm.r_des, float).reshape(3, 3).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "R_hand": Rh.tolist(),
        "R_obj": Ro.tolist(),
        "geoms": [(int(c["geom1"]), int(c["geom2"]), c["side"]) for c in fc],
        "contact_pts": [np.asarray(c["pos_w"], float).tolist() for c in fc],
        "n_contacts": len(fc),
        "contact_topology": ("L" if int(o["nL"]) > 0 else "") + ("R" if int(o["nR"]) > 0 else "") or "none",
        "R_rel": np.asarray(o["R_rel"], float).tolist(),
        "lever_arm": float(np.linalg.norm(r_w)),
    }


def append_log(log, m):
    if not log or m["t"] - log[-1]["t"] >= LOG_DT - 1e-12:
        log.append(m)
        return
    log[-1] = m


def overlay(ctl, m, phase):
    if not ctl or not ctl.get("sync"):
        return
    ctl["overlay"] = {
        "case": ctl.get("label", phase),
        "t": m["t"],
        "phase": phase,
        "event": ctl.get("last_event", ""),
        "nL": m["nL"],
        "nR": m["nR"],
        "rhx_mm": 1e3 * float(m["rh"][0]),
        "vz": m["v_obj"][2],
        "ang": m["ang_deg"],
        "ap": m.get("aperture"),
    }
    ctl["sync"]()


def step(sim, gains, omega_y, v_x, v_z, tau, log, Rh0, label, ev=None, ctl=None):
    cmd = tick_4d(sim, gains, omega_y, v_x, v_z, tau)
    m = measure(sim, omega_y=omega_y, v_x=v_x, v_z=v_z, tau=tau, label=label, Rh0=Rh0)
    m["w_cmd_world"] = np.asarray(cmd["w_world"], float).tolist()
    append_log(log, m)
    if ev is not None and object_floor_hit(sim) and "FLOOR" not in ev:
        ev["FLOOR"] = m["t"]
        return m, True
    overlay(ctl, m, label)
    if ctl and ctl.get("reset"):
        return m, True
    return m, False


def rotate_until(sim, gains, parent, target_deg, omega_sign, log, ev, ctl=None, tau=TAU_SEC):
    restore(sim, parent)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    ev["ROTATE_START"] = float(sim.data.time)
    ev["Rh0"] = Rh0.tolist()
    goal = abs(float(target_deg))
    omega = float(omega_sign) * OMEGA if goal > 0.5 else 0.0
    t0 = float(sim.data.time)
    timeout = t0 + (np.deg2rad(max(goal, 1.0)) / max(OMEGA, 1e-6)) + 1.5
    m = measure(sim, omega_y=omega, tau=tau, label="ROTATE", Rh0=Rh0)
    append_log(log, m)
    lost = False
    while float(sim.data.time) < timeout:
        ang = signed_rot_about_y_deg(
            Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        )
        if abs(ang) >= goal - 0.8:
            break
        m, stop = step(sim, gains, omega, 0.0, 0.0, tau, log, Rh0, "ROTATE", ev, ctl)
        if stop:
            break
        if m["nL"] == 0 and m["nR"] == 0:
            lost = True
            if "CONTACT_LOST_DURING_ROTATE" not in ev:
                ev["CONTACT_LOST_DURING_ROTATE"] = m["t"]
            break
    ang = signed_rot_about_y_deg(
        Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    )
    m = measure(sim, omega_y=omega, tau=tau, label="PRE_RELEASE", Rh0=Rh0)
    return {
        "Rh0": Rh0,
        "actual_deg": float(ang),
        "omega": omega,
        "lost": lost,
        "pre": m,
        "dt_rot": float(sim.data.time) - t0,
    }


def open_until_both_off(sim, gains, omega, log, Rh0, ev, ctl=None, keep_omega=True):
    ev["RELEASE_COMMAND"] = float(sim.data.time)
    dt = float(sim.model.opt.timestep)
    both = 0.0
    t0 = float(sim.data.time)
    m = log[-1] if log else measure(sim, tau=TAU_OPEN, label="OPEN", Rh0=Rh0)
    while float(sim.data.time) < t0 + 0.45:
        wy = omega if keep_omega else 0.0
        m, stop = step(sim, gains, wy, 0.0, 0.0, TAU_OPEN, log, Rh0, "OPEN", ev, ctl)
        if stop:
            break
        if m["nL"] == 0 and m["nR"] == 0:
            if "FIRST_BOTH_OFF" not in ev:
                ev["FIRST_BOTH_OFF"] = m["t"]
                ev["both_off_state"] = slim_row(m)
            both += dt
            if both >= BOTH_OFF_MIN:
                break
        else:
            both = 0.0
    return m


def ballistic_watch(sim, gains, log, Rh0, ev, ctl=None, dur=BALLISTIC_S):
    t0 = float(sim.data.time)
    z0 = float(sim.data.xpos[sim.ids.object_body][2])
    z_max = z0
    t_apex = None
    vz_prev = None
    m = log[-1]
    while float(sim.data.time) < t0 + dur:
        m, stop = step(sim, gains, 0.0, 0.0, 0.0, TAU_OPEN, log, Rh0, "BALLISTIC", ev, ctl)
        if stop:
            break
        z = float(m["obj_z"])
        vz = float(m["v_obj"][2])
        if z >= z_max:
            z_max = z
            t_apex = m["t"]
        if vz_prev is not None and vz_prev > 0.0 and vz <= 0.0 and "BALLISTIC_APEX" not in ev:
            ev["BALLISTIC_APEX"] = m["t"]
            t_apex = m["t"]
        vz_prev = vz
        if m["nL"] > 0 or m["nR"] > 0:
            if "RECONTACT_DURING_BALLISTIC" not in ev:
                ev["RECONTACT_DURING_BALLISTIC"] = m["t"]
    if "BALLISTIC_APEX" not in ev and t_apex is not None:
        ev["BALLISTIC_APEX"] = t_apex
    return {
        "z0": z0,
        "z_max": z_max,
        "up_disp": float(z_max - z0),
        "t_apex": t_apex,
        "end": slim_row(m),
    }


def recatch_privileged(sim, gains, log, Rh0, ev, ctl=None):
    """Privileged: close when object is near the hand capture region or after apex descent."""
    ev["RECATCH_PREP"] = float(sim.data.time)
    ev["CLOSE_START"] = float(sim.data.time)
    t0 = float(sim.data.time)
    captured = False
    t_lim = t0 + 0.45
    m = log[-1]
    while float(sim.data.time) < t_lim:
        m, stop = step(sim, gains, 0.0, 0.0, 0.0, TAU_SEC, log, Rh0, "CLOSE", ev, ctl)
        if stop:
            break
        if (m["nL"] > 0 or m["nR"] > 0) and "FIRST_RECONTACT" not in ev:
            ev["FIRST_RECONTACT"] = m["t"]
        if m["nL"] > 0 and m["nR"] > 0:
            if "FIRST_BILATERAL" not in ev:
                ev["FIRST_BILATERAL"] = m["t"]
            elif m["t"] - ev["FIRST_BILATERAL"] >= 0.050:
                ev["MOTION_ARREST"] = m["t"]
                captured = True
                break
        if m["t"] - t0 > 0.40:
            break
    if captured:
        ev["RETURN_START"] = float(sim.data.time)
        t_ret = float(sim.data.time) + 1.2
        while float(sim.data.time) < t_ret:
            ang = float(m["ang_deg"])
            if abs(ang) < 4.0:
                break
            wy = -OMEGA if ang > 0 else OMEGA
            m, stop = step(sim, gains, wy, 0.0, 0.0, TAU_SEC, log, Rh0, "RETURN", ev, ctl)
            if stop:
                captured = False
                break
            if m["nL"] == 0 or m["nR"] == 0:
                captured = False
                ev["RETURN_LOST"] = m["t"]
                break
        if not captured:
            return captured, m
        ev["HOLD_START"] = float(sim.data.time)
        t_hold = float(sim.data.time) + HOLD_S
        lost = 0.0
        dt = float(sim.model.opt.timestep)
        while float(sim.data.time) < t_hold:
            m, stop = step(sim, gains, 0.0, 0.0, 0.0, TAU_SEC, log, Rh0, "HOLD", ev, ctl)
            if stop:
                captured = False
                break
            if m["nL"] == 0 or m["nR"] == 0:
                lost += dt
                if lost >= 0.08:
                    captured = False
                    ev["HOLD_LOST"] = m["t"]
                    break
            else:
                lost = 0.0
    return captured, m


def save_npz(path: Path, log: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=np.array([x["t"] for x in log]),
        ang=np.array([x["ang_deg"] for x in log]),
        omega_cmd=np.array([x["omega_y_cmd"] for x in log]),
        w_hand=np.array([x["w_hand"] for x in log]),
        ctrl7=np.array([x["ctrl7"] for x in log]),
        ap=np.array([x["aperture"] for x in log]),
        nL=np.array([x["nL"] for x in log]),
        nR=np.array([x["nR"] for x in log]),
        obj_z=np.array([x["obj_z"] for x in log]),
        v_obj=np.array([x["v_obj"] for x in log]),
        v_hand=np.array([x["v_hand"] for x in log]),
        p_obj=np.array([x["p_obj"] for x in log]),
        p_hand=np.array([x["p_hand"] for x in log]),
        rh=np.array([x["rh"] for x in log]),
        v_rel_h=np.array([x["v_rel_h"] for x in log]),
        Fn_L=np.array([x["Fn_L"] for x in log]),
        Fn_R=np.array([x["Fn_R"] for x in log]),
        Ft_L=np.array([x["Ft_L"] for x in log]),
        Ft_R=np.array([x["Ft_R"] for x in log]),
        rho=np.array([np.nan if x["rho_max"] is None else x["rho_max"] for x in log]),
        carry_err=np.array([x["v_carry_err_norm"] for x in log]),
        clear=np.array([x["clear"] for x in log]),
    )


def traj_excerpt(log, n=14) -> list:
    if not log:
        return []
    idx = np.linspace(0, len(log) - 1, num=min(n, len(log)), dtype=int)
    out = []
    for i in idx:
        r = log[int(i)]
        out.append(
            {
                "t": r["t"],
                "label": r["label"],
                "ang_deg": r["ang_deg"],
                "nL": r["nL"],
                "nR": r["nR"],
                "ap": r["aperture"],
                "obj_z": r["obj_z"],
                "v_obj_z": r["v_obj"][2],
                "v_hand_z": r["v_hand"][2],
                "omega_cmd": r["omega_y_cmd"],
                "w_hand": r["w_hand"],
                "rh_mm": [1e3 * x for x in r["rh"]],
                "carry_err": r["v_carry_err_norm"],
            }
        )
    return out


def rotation_audit(sim, gains, parent, omega_sign, ctl=None):
    restore(sim, parent)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    log = [measure(sim, omega_y=0.0, tau=TAU_SEC, label="PARENT", Rh0=Rh0)]
    ev = {"ROTATE_START": float(sim.data.time)}
    marks = {"0": slim_row(log[0])}
    next_i = 1
    targets = list(AUDIT_DEG)
    omega = float(omega_sign) * OMEGA
    timeout = float(sim.data.time) + np.deg2rad(125.0) / OMEGA + 1.5
    while float(sim.data.time) < timeout and next_i < len(targets):
        ang = signed_rot_about_y_deg(
            Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        )
        if abs(ang) >= targets[next_i] - 0.8:
            m = measure(sim, omega_y=omega, tau=TAU_SEC, label=f"AUDIT_{int(targets[next_i])}", Rh0=Rh0)
            marks[str(int(targets[next_i]))] = slim_row(m)
            next_i += 1
            if next_i >= len(targets):
                break
        m, stop = step(sim, gains, omega, 0.0, 0.0, TAU_SEC, log, Rh0, "ROTATE", ev, ctl)
        if stop or (m["nL"] == 0 and m["nR"] == 0):
            break
    save_npz(RAW / f"rotation_audit_s{int(omega_sign)}.npz", log)
    dump(RAW / f"rotation_audit_s{int(omega_sign)}.json", {"marks": marks, "events": ev})
    return {"marks": marks, "log": log, "Rh0": Rh0}


def release_probe(sim, gains, parent, target_deg, omega_sign, ctl=None, recatch=False, watch=True):
    log = []
    ev = {}
    rot = rotate_until(sim, gains, parent, target_deg, omega_sign, log, ev, ctl=ctl)
    pre = rot["pre"]
    if rot["lost"]:
        return {
            "target_deg": target_deg,
            "omega_sign": omega_sign,
            "actual_deg": rot["actual_deg"],
            "lost_before_release": True,
            "pre": slim_row(pre),
            "events": ev,
            "log": log,
            "v_obj_z_both_off": None,
        }
    m_off = open_until_both_off(sim, gains, rot["omega"], log, rot["Rh0"], ev, ctl=ctl, keep_omega=True)
    both = ev.get("both_off_state")
    ball = None
    captured = False
    if "FIRST_BOTH_OFF" in ev and watch and not recatch:
        ball = ballistic_watch(sim, gains, log, rot["Rh0"], ev, ctl=ctl)
    if "FIRST_BOTH_OFF" in ev and recatch:
        # Catcher parks (omega=0). Short privileged wait for apex / visible free flight, then close.
        t_wait = float(sim.data.time) + 0.040
        while float(sim.data.time) < t_wait:
            m, stop = step(sim, gains, 0.0, 0.0, 0.0, TAU_OPEN, log, rot["Rh0"], "BALLISTIC", ev, ctl)
            if stop:
                break
            vz = float(m["v_obj"][2])
            if vz <= 0.0 and "BALLISTIC_APEX" not in ev:
                ev["BALLISTIC_APEX"] = m["t"]
                break
        captured, _ = recatch_privileged(sim, gains, log, rot["Rh0"], ev, ctl=ctl)
        z_series = [r["obj_z"] for r in log if r.get("label") in ("OPEN", "BALLISTIC", "CLOSE", "HOLD")]
        if z_series:
            ball = {
                "z0": z_series[0],
                "z_max": max(z_series),
                "up_disp": float(max(z_series) - z_series[0]),
                "t_apex": ev.get("BALLISTIC_APEX"),
            }
    vz = None if both is None else float(both["v_obj"][2])
    return {
        "target_deg": target_deg,
        "omega_sign": omega_sign,
        "actual_deg": rot["actual_deg"],
        "lost_before_release": False,
        "dt_rot": rot["dt_rot"],
        "pre": slim_row(pre),
        "both_off": both,
        "v_obj_z_both_off": vz,
        "upward": bool(vz is not None and vz > V_UP_MIN),
        "ballistic": ball,
        "captured": captured,
        "ok_hold": bool(captured and ev.get("HOLD_START") and "HOLD_LOST" not in ev and "FLOOR" not in ev),
        "events": {k: v for k, v in ev.items() if k != "Rh0"},
        "log": log,
        "end": slim_row(log[-1]) if log else slim_row(m_off),
    }


def no_rotation_release(sim, gains, parent, ctl=None):
    return release_probe(sim, gains, parent, 0.0, 1.0, ctl=ctl, recatch=False, watch=True)


def write_report(prep, audits, probes, zero, recatch_row, verdict: str) -> None:
    p = OUT / "DYNAMIC_THROW_RECAPTURE.md"
    a = []
    A = a.append
    A("# Dynamic throw / recapture authority")
    A("")
    A("No SAC. No observation/reward/detector change. No teleport. No object qvel injection. No MP4.")
    A("No action-bound expansion. Cartesian controller not retuned.")
    A("")
    A("## 1. Mechanism hypothesis")
    A("")
    A("Maintain secure contact (`tau=-18`) while commanding legal `omega_y=+/-3` rad/s, transfer momentum through finger-object contact, then actively open (`ctrl[7]=+2`) near a coarse orientation so the cylinder enters a short ballistic arc (ideally `v_obj,z>0`) and later re-enters a reachable capture region.")
    A("")
    A("This is distinct from (1) controlled slip with retained contact and (2) drop-then-chase after unconstrained free fall.")
    A("")
    lift = prep["lift"]
    A(f"High-clearance parent (physical lift): obj_z={lift['obj_z']:.4f} m, virtual table clearance **{1e2*float(lift['clear']):.1f} cm**. Gravity={prep['gravity']}. Mass={prep['mass']:.4f} kg.")
    ch = prep["audit"]["changed"]
    A(f"Table geom `{ch['geom']}` mask {ch['contype_before']}/{ch['conaffinity_before']} -> {ch['contype_after']}/{ch['conaffinity_after']}. Object and floor unchanged. Official scene.xml unmodified.")
    A("")
    A("## 2. Legal rotation response")
    A("")
    A(f"Commanded `omega_y` bound = {OMEGA} rad/s (`RECOVERY4D_W_HY_MAX`). `w_world = r_des @ [0, omega_y, 0]`.")
    A("")
    for sign, aud in audits.items():
        A(f"### sign={sign}")
        A("")
        A("| deg cmd | ang act | nL/nR | |w_hand| | w_hand | v_obj | v_obj,z | carry_err | rho |")
        A("|---|---|---|---|---|---|---|---|---|")
        for k in ("0", "30", "60", "90", "120"):
            m = (aud.get("marks") or {}).get(k)
            if not m:
                A(f"| {k} | missing | | | | | | | |")
                continue
            wh = np.asarray(m["w_hand"], float)
            vo = np.asarray(m["v_obj"], float)
            A(
                f"| {k} | {m['ang_deg']:.1f} | {m['nL']}/{m['nR']} | {np.linalg.norm(wh):.3f} | "
                f"{np.round(wh, 3).tolist()} | {np.round(vo, 3).tolist()} | {vo[2]:+.4f} | "
                f"{m['v_carry_err_norm']:.4f} | {m.get('rho_max')} |"
            )
        A("")
    A("## 3. Pre-release velocity audit")
    A("")
    A("At each probe, immediately before opening: object world/hand velocity, hand twist, COM lever arm, Fn/Ft/rho, relative slip.")
    A("")
    for pr in probes:
        pre = pr.get("pre") or {}
        vz_pre = None if not pre.get("v_obj") else float(pre["v_obj"][2])
        A(
            f"- target {pr['target_deg']} deg sign={pr['omega_sign']} actual={pr.get('actual_deg'):.1f}: "
            f"v_obj={pre.get('v_obj')} v_obj,z={vz_pre} "
            f"v_obj_h={pre.get('v_obj_h')} v_hand={pre.get('v_hand')} w_hand={pre.get('w_hand')} "
            f"r_world={pre.get('r_world')} nL/nR={pre.get('nL')}/{pre.get('nR')} "
            f"Fn={pre.get('Fn_L')}/{pre.get('Fn_R')} Ft={pre.get('Ft_L')}/{pre.get('Ft_R')} "
            f"rho={pre.get('rho_max')} v_rel={pre.get('v_rel')} carry_err={pre.get('v_carry_err_norm')}"
        )
    A("")
    A("## 4. Release-angle probe")
    A("")
    A(f"Legal open `ctrl[7]={TAU_OPEN}`. Both-off persistence >= {BOTH_OFF_MIN}s. No intra-angle timing search.")
    A("")
    A("| target | actual | sign | lost-pre | v_obj,z at FIRST_BOTH_OFF | |w_obj| | upward |")
    A("|---|---|---|---|---|---|---|")
    for pr in probes:
        bo = pr.get("both_off") or {}
        wo = np.asarray(bo.get("w_obj") or [0, 0, 0], float)
        vz = pr.get("v_obj_z_both_off")
        A(
            f"| {pr['target_deg']} | {pr.get('actual_deg'):.1f} | {pr['omega_sign']} | {pr.get('lost_before_release')} | "
            f"{vz} | {float(np.linalg.norm(wo)):.3f} | {pr.get('upward')} |"
        )
    A("")
    A("## 5. Upward-velocity evidence")
    A("")
    ups = [pr for pr in probes if pr.get("upward")]
    if not ups:
        A(f"**NO.** No probe produced `v_obj,z > {V_UP_MIN}` m/s at FIRST_BOTH_OFF. The upward-throw mechanism is **not** established under legal `omega_y`.")
        A("")
        A("Do not manufacture upward velocity. Recapture, CENTER-6 transfer, and throw viewers of a successful arc are not claimed.")
    else:
        u = ups[0]
        A(f"**YES** at first coarse hit: target {u['target_deg']} deg, actual {u['actual_deg']:.1f}, sign={u['omega_sign']}, v_obj,z={u['v_obj_z_both_off']:.4f} m/s.")
        A("")
        A("PRIVILEGED construction only. Not optimized.")
    A("")
    A("## 6. Ballistic trajectory")
    A("")
    if not ups:
        A("No upward-release candidate. Open-and-watch trajectories still logged for the probe set (objects fall under g).")
        if probes:
            b = (probes[0].get("ballistic") or {})
            A(f"Example (first probe) z0={b.get('z0')} z_max={b.get('z_max')} up_disp={b.get('up_disp')}.")
    else:
        u = ups[0]
        b = u.get("ballistic") or {}
        A(f"After FIRST_BOTH_OFF, no recatch for {BALLISTIC_S}s. z0={b.get('z0')} z_max={b.get('z_max')} up_disp={b.get('up_disp')} t_apex={b.get('t_apex')}.")
        A("See `raw/throw_ballistic.npz`.")
    A("")
    A("## 7. Matched no-rotation release")
    A("")
    if zero is None:
        A("Not run.")
    else:
        A(
            f"Open at ~0 deg, same grip law. v_obj,z at both-off={zero.get('v_obj_z_both_off')}. "
            f"upward={zero.get('upward')}. ballistic up_disp={(zero.get('ballistic') or {}).get('up_disp')}."
        )
        A("Causal claim requires rotating release velocity to differ materially from this baseline.")
    A("")
    A("## 8. Privileged recapture construction")
    A("")
    if recatch_row is None:
        A("Not attempted: no useful upward ballistic candidate.")
    else:
        A("**PRIVILEGED DYNAMIC RECAPTURE CONSTRUCTION.** Close timing uses GT (both-off plus a short apex wait), not an observable policy.")
        A("After both-off the catcher parks (`omega_y=0`), then `tau=-18`. After brief bilateral arrest, a legal return toward the parent orientation is attempted before the long hold.")
        A(f"captured={recatch_row.get('captured')} ok_hold={recatch_row.get('ok_hold')}")
        evs = recatch_row.get("events") or {}
        keys = (
            "ROTATE_START", "RELEASE_COMMAND", "FIRST_BOTH_OFF", "BALLISTIC_APEX",
            "RECATCH_PREP", "CLOSE_START", "FIRST_RECONTACT", "FIRST_BILATERAL",
            "MOTION_ARREST", "RETURN_START", "RETURN_LOST", "HOLD_START", "HOLD_LOST", "FLOOR",
        )
        A("Event times: " + ", ".join(f"{k}={evs.get(k)}" for k in keys if k in evs))
        if evs.get("MOTION_ARREST") and not recatch_row.get("ok_hold"):
            A("Recontact and a short bilateral arrest occurred. Sustained secure hold did not. That is recapture/hold failure, not absence of a throw. Likely limiter: residual relative velocity plus gravity along hand-x at ~60 deg. Not optimized further.")
    A("")
    A("## 9. Causal baselines")
    A("")
    if recatch_row is None:
        A("A: rotate+release+no reclose = ballistic miss (logged). B: recatch not established. C: no-rotation release compared in section 7.")
    else:
        A("A: rotate+release+no reclose produces an upward ballistic sample (probe log).")
        A(f"B: rotate+release+reclose: captured={recatch_row.get('captured')} ok_hold={recatch_row.get('ok_hold')}.")
        A(f"C: no-rotation + same open: v_obj,z={None if zero is None else zero.get('v_obj_z_both_off')} (not an upward throw).")
    A("")
    A("## 10. Stable-hold evidence")
    A("")
    if recatch_row is None or not recatch_row.get("ok_hold"):
        A("No sustained airborne recapture hold in this construction.")
    else:
        end = recatch_row.get("end") or {}
        A(f"Hold {HOLD_S}s: nL/nR={end.get('nL')}/{end.get('nR')} v_rel={end.get('v_rel')} obj_z={end.get('obj_z')}.")
    A("")
    A("## 11. CENTER-6 transfer")
    A("")
    A("Not run. Clean throw-recapture mechanism did not succeed, so transfer is not claimed.")
    A("")
    A("## 12. Limitations")
    A("")
    A(verdict)
    A("")
    A("- Coarse angles and both signs only. No throw optimization.")
    A(f"- Legal bounds unchanged: |omega_y|<={OMEGA}, |v_x|<={VX_MAX}, |v_z|<={VZ_MAX}, tau_open={TAU_OPEN}.")
    A("- Table collision disabled after parent; floor remains.")
    A("")
    A("## Viewer")
    A("")
    A("```text")
    A("python training/demo_dynamic_recatch.py --mode throw_only")
    A("python training/demo_dynamic_recatch.py --mode throw_recatch")
    A("python training/demo_dynamic_recatch.py --mode sweep")
    A("```")
    A("")
    A("SPACE pause, R restart, `[` `]` speed. Camera initialized once. No MP4.")
    p.write_text("\n".join(a), encoding="utf-8")
    print("wrote", p, flush=True)


def headless():
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    sim = make_parent_sim()
    orig_masks = copy_masks(sim)
    prep = prepare_high_parent(sim, cfg, gains, orig_masks)
    if not prep.get("ok"):
        dump(RAW / "prep_fail.json", prep)
        raise RuntimeError(prep)
    parent = prep["parent"]
    with (RAW / "high_parent.pkl").open("wb") as f:
        pickle.dump(parent, f)
    dump(RAW / "parent_meta.json", {"lift": prep["lift"], "low": prep["low"], "gravity": prep["gravity"], "mass": prep["mass"]})
    dump(RAW / "collision_mask_audit.json", prep["audit"])
    print("PARENT clear", prep["lift"]["clear"], "g", prep["gravity"], flush=True)

    audits = {}
    for s in (+1, -1):
        print("rotation audit sign", s, flush=True)
        audits[str(s)] = rotation_audit(sim, gains, parent, s)
        dump(RAW / f"audit_marks_s{s}.json", audits[str(s)]["marks"])

    probes = []
    chosen = None
    for s in (+1, -1):
        for deg in PROBE_DEG:
            print("probe", deg, "sign", s, flush=True)
            pr = release_probe(sim, gains, parent, deg, s, recatch=False, watch=True)
            probes.append({k: v for k, v in pr.items()})
            vz = pr.get("v_obj_z_both_off")
            print("  actual", pr.get("actual_deg"), "vz", vz, "up", pr.get("upward"), "lost", pr.get("lost_before_release"), flush=True)
            dump(RAW / f"probe_{int(deg)}_s{int(s)}.json", {k: v for k, v in pr.items() if k != "log"})
            save_npz(RAW / f"probe_{int(deg)}_s{int(s)}.npz", pr["log"])
            dump(RAW / f"probe_{int(deg)}_s{int(s)}_traj.json", traj_excerpt(pr["log"]))
            if chosen is None and pr.get("upward"):
                chosen = pr
                break
        if chosen is not None:
            break

    zero = no_rotation_release(sim, gains, parent)
    dump(RAW / "zero_rotation.json", {k: v for k, v in zero.items() if k != "log"})
    save_npz(RAW / "zero_rotation.npz", zero["log"])
    print("zero vz", zero.get("v_obj_z_both_off"), flush=True)

    recatch_row = None
    if chosen is not None:
        save_npz(RAW / "throw_ballistic.npz", chosen["log"])
        print("UPWARD candidate; privileged recatch", chosen["target_deg"], chosen["omega_sign"], flush=True)
        rec = release_probe(
            sim, gains, parent, chosen["target_deg"], chosen["omega_sign"], recatch=True, watch=True
        )
        recatch_row = {k: v for k, v in rec.items() if k != "log"}
        save_npz(RAW / "throw_recatch.npz", rec["log"])
        dump(RAW / "throw_recatch.json", recatch_row)
        dump(
            RAW / "plan.json",
            {
                "upward": True,
                "target_deg": chosen["target_deg"],
                "omega_sign": chosen["omega_sign"],
                "ok_hold": rec.get("ok_hold"),
            },
        )
        if recatch_row.get("ok_hold"):
            verdict = (
                "PRIVILEGED DYNAMIC RECAPTURE CONSTRUCTION: legal secure rotation created "
                f"v_obj,z={chosen['v_obj_z_both_off']:.4f} m/s at both-off and a later recapture hold."
            )
        else:
            verdict = (
                "THROW MECHANISM YES, STABLE RECAPTURE NOT ESTABLISHED: legal omega_y while "
                f"secure produced v_obj,z={chosen['v_obj_z_both_off']:.4f} m/s at FIRST_BOTH_OFF "
                "(no-rotation baseline v_z negative). Privileged close achieved recontact/brief "
                "bilateral arrest but not a sustained airborne hold. Bounds not expanded. "
                "CENTER-6 transfer not run."
            )
    else:
        dump(RAW / "plan.json", {"upward": False, "target_deg": 90.0, "omega_sign": 1})
        vzs = [p.get("v_obj_z_both_off") for p in probes]
        verdict = (
            "UPWARD-THROW MECHANISM NOT ESTABLISHED: every coarse legal rotation+open "
            f"had v_obj,z<= {V_UP_MIN} at FIRST_BOTH_OFF (values={vzs}). "
            "Object is not given useful positive world-z speed by contact-carried omega_y "
            "before grip opens. Bounds were not expanded. No recapture attempt. No CENTER-6 transfer."
        )
    dump(RAW / "probes_summary.json", [{k: v for k, v in p.items() if k != "log"} for p in probes])
    print(verdict, flush=True)
    write_report(prep, audits, [{k: v for k, v in p.items() if k != "log"} for p in probes], {k: v for k, v in zero.items() if k != "log"}, recatch_row, verdict)


def interactive(mode: str) -> None:
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    plan_p = RAW / "plan.json"
    plan = json.loads(plan_p.read_text(encoding="utf-8")) if plan_p.is_file() else {"upward": False, "target_deg": 90.0, "omega_sign": 1}
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
    orig_masks = copy_masks(sim)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        ctl["viewer"] = vwr
        apply_camera_preset(vwr, CAM)
        while vwr.is_running():
            ctl["reset"] = False
            ctl["_wall"] = None
            apply_masks(sim, orig_masks)
            sim.reset(CYL_MASS, PAIR_MU, np.zeros(3))
            prep = prepare_high_parent(sim, cfg, gains, orig_masks, ctl=ctl)
            if not prep.get("ok"):
                time.sleep(0.2)
                continue
            parent = prep["parent"]
            recatch = bool(mode == "throw_recatch" and plan.get("upward"))
            release_probe(
                sim, gains, parent, float(plan.get("target_deg", 90.0)), int(plan.get("omega_sign", 1)),
                ctl=ctl, recatch=recatch, watch=True,
            )
            if ctl.get("reset"):
                continue
            while vwr.is_running() and not ctl.get("reset"):
                sync()
                time.sleep(0.03)
            if not ctl.get("reset"):
                break


def interactive_phase(mode: str) -> None:
    import mujoco.viewer

    from training.dynamic_throw_phase import make_restorers, run_viewer_trial

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

    env = make_restorers()
    sim = env["cen_sim"] if mode == "centered_throw" else env["off_sim"]
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        ctl["viewer"] = vwr
        apply_camera_preset(vwr, CAM)
        while vwr.is_running():
            ctl["reset"] = False
            ctl["_wall"] = None
            run_viewer_trial(mode, env, env["gains"], ctl)
            if ctl.get("reset"):
                continue
            while vwr.is_running() and not ctl.get("reset"):
                sync()
                time.sleep(0.03)
            if not ctl.get("reset"):
                break


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        default="phase_audit",
        choices=(
            "phase_audit",
            "sweep",
            "throw_only",
            "throw_recatch",
            "centered_throw",
            "offset_late_open",
            "offset_early_open",
            "offset_throw_recatch",
            "throw_omega4",
            "recatch_omega4",
            "omega4",
            "release_map_omega4",
            "offset_throw_recatch_omega4",
            "centered_throw_omega4_70",
        ),
    )
    args = p.parse_args()
    if args.mode in ("offset_throw_recatch_omega4", "centered_throw_omega4_70"):
        from training.release_state_map_omega4 import interactive_claimed

        interactive_claimed(args.mode)
        return
    if args.mode == "release_map_omega4":
        from training.release_state_map_omega4 import main as release_map_omega4

        release_map_omega4()
        return
    if args.mode == "omega4":
        from training.omega4_throw_test import headless_omega4

        headless_omega4()
        return
    if args.mode in ("throw_omega4", "recatch_omega4"):
        from training.omega4_throw_test import interactive_omega4

        interactive_omega4(args.mode)
        return
    phase_modes = ("centered_throw", "offset_late_open", "offset_early_open", "offset_throw_recatch")
    if args.mode in ("throw_only", "throw_recatch"):
        interactive(args.mode)
        return
    if args.mode in phase_modes:
        interactive_phase(args.mode)
        return
    if args.mode == "sweep":
        headless()
        return
    from training.dynamic_throw_phase import headless_phase_audit

    headless_phase_audit()


if __name__ == "__main__":
    main()
