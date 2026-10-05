"""Gravitational reposition recovery construction on S_FAIL (noslip=1).

Manual physics sequence. Does not call RULE or SAC. Privileged relative
geometry is allowed for brake targeting.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.dynamics import set_finger_object_sliding_mu
from envs.physical_recovery import RH_Z_NOM, TABLE_DROP, physical_pack
from training.demo_teleport_recovery_state import (
    PAIR_MU,
    advance_to_parent,
    apply_coupled_pose,
    coupled_pose_from_ref,
    load_or_measure_ref,
    make_parent_sim,
    signed_rot_about_y_deg,
    t0_valid,
    contact_debug,
)
from training.grav_reposition_v2_viz import (
    PRE_VIEW_PAUSE_S,
    capture_hold_refs,
    init_camera_once,
    make_reset_cam_callback,
    vis_overlay,
    viz_frame,
    wall_pause,
)
from training.nominal_grasp_creep_root_cause import dump_json
from training.replay_core import freeze, tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts

OUT = ROOT / "results" / "diagnostics" / "grav_reposition_recovery_construction"
RAW = OUT / "raw"
FIG = OUT / "figures"

V_LIFT = 0.08
DZ_LIFT = 0.18
HOLD_S = 10.0
TAU_SEC = -18.0
TAU_SLIP0 = -3.0
OMEGA = 1.2
OMEGA_RET = 1.0
THETA = 30.0
LOG_DT = 0.010
OBS_S = 0.100
VERT_HOLD = 2.0
BRAKE_HOLD = 0.50
PLAYBACK = 0.28


def restore(sim, snap) -> None:
    sim.load_snapshot(snap)
    set_finger_object_sliding_mu(sim.model, sim.ids, float(snap.get("friction", PAIR_MU)), data=None)
    mujoco.mj_forward(sim.model, sim.data)


def snap_now(sim) -> dict:
    s = sim.snapshot()
    s["friction"] = float(PAIR_MU)
    return s


def live_opt(sim) -> dict:
    m = sim.model
    return {
        "mujoco": mujoco.__version__,
        "noslip_iterations": int(m.opt.noslip_iterations),
        "noslip_tolerance": float(m.opt.noslip_tolerance),
        "timestep": float(m.opt.timestep),
        "gravity": np.array(m.opt.gravity, float).tolist(),
    }


def measure(sim, tau: float) -> dict:
    rows, sm = extract_contacts(sim)
    o = physical_pack(sim)
    rhos = [r["rho"] for r in rows if np.isfinite(r.get("rho", np.nan))]
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    rh = np.asarray(o["rh"], float)
    return {
        "t": float(sim.data.time),
        "rh": rh.copy(),
        "e_x": float(rh[0]),
        "e_z": float(rh[2] - RH_Z_NOM),
        "v_rel_h": np.asarray(o["v_rel_h"], float).copy(),
        "w_rel_h": np.asarray(o["w_rel_h"], float).copy(),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "obj_z": float(o["obj_z"]),
        "aperture": float(o["aperture"]),
        "tau": float(tau),
        "ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float(tau),
        "g_h": np.asarray(o["g_h"], float).copy(),
        "rho_max": float(np.max(rhos)) if rhos else float("nan"),
        "rho_med": float(np.median(rhos)) if rhos else float("nan"),
        "Fn_L": float(sum(abs(r["Fn"]) for r in rows if r["side"] == "L")),
        "Fn_R": float(sum(abs(r["Fn"]) for r in rows if r["side"] == "R")),
        "cyl_h": (Rh.T @ Ro[:, 2]).copy(),
        "wrist_deg": None,
    }


def emit(viz, sim, phase, tau, Rh0, m, rh0=None) -> None:
    if viz is None:
        return
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    wdeg = float(signed_rot_about_y_deg(Rh0, Rh)) if Rh0 is not None else 0.0
    rh = np.asarray(m["rh"], float)
    drh = (rh - np.asarray(rh0, float)) * 1e3 if rh0 is not None else rh * 1e3
    viz_frame(
        viz, sim, phase=phase, wrist_deg=wdeg, tau=tau, rho=float(m["rho_max"]),
        drh_mm=drh, progress_mm=float(np.linalg.norm(drh[:3:2])),
        v_rel_h=m["v_rel_h"], nL=m["nL"], nR=m["nR"],
    )


def dropped(sim, m) -> bool:
    return bool(sim.dropped() or float(m["obj_z"]) < TABLE_DROP or (m["nL"] == 0 and m["nR"] == 0))


def apply_sfail(sim, parent, ref, s: float) -> dict:
    restore(sim, parent)
    freeze(sim)
    dp, Rd = coupled_pose_from_ref(ref, float(s), "coupled")
    applied = apply_coupled_pose(sim, dp, Rd)
    freeze(sim)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.p_des = ph.copy()
    sim.fsm.v_cmd[:] = 0
    sim.fsm.w_cmd[:] = 0
    dbg = contact_debug(sim)
    ok, why = t0_valid(applied["after"], dbg)
    m = measure(sim, TAU_SEC)
    return {
        "ok": bool(ok),
        "why": why,
        "s": float(s),
        "snap": snap_now(sim),
        "after": {
            "rh": m["rh"].tolist(),
            "e_x_mm": 1e3 * m["e_x"],
            "nL": m["nL"],
            "nR": m["nR"],
            "obj_z": m["obj_z"],
            "g_h": m["g_h"].tolist(),
            "cyl_h": m["cyl_h"].tolist(),
            "v_rel_h": m["v_rel_h"].tolist(),
            "w_rel_h": m["w_rel_h"].tolist(),
        },
        "dbg": dbg,
        "live": live_opt(sim),
        "dp_h": np.asarray(dp, float).tolist(),
    }


def continue_nominal(sim, cfg, gains, *, hold_s=HOLD_S, viz=None, Rh0=None, rh0=None, phase_prefix="LIFT"):
    dt = float(sim.model.opt.timestep)
    freeze(sim)
    p0z = float(sim.fsm.p_des[2])
    lift_v = np.array([0.0, 0.0, V_LIFT])
    t0 = float(sim.data.time)
    t_lift = None
    t_drop = None
    t_ul = None
    t_both = None
    log = []
    log_n = max(int(round(LOG_DT / dt)), 1)
    n_max = int(round((DZ_LIFT / V_LIFT + hold_s + 2.0) / dt))
    i = 0
    while i < n_max:
        if t_lift is None:
            tick_vw(sim, lift_v, np.zeros(3), TAU_SEC, gains)
            phs = phase_prefix
        else:
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
            phs = "HOLD10"
        m = measure(sim, TAU_SEC)
        t = float(sim.data.time)
        if t_lift is None and float(sim.fsm.p_des[2]) - p0z >= DZ_LIFT - 1e-9:
            t_lift = t
        if t_drop is None and dropped(sim, m):
            t_drop = t
        if t_ul is None and (m["nL"] == 0 or m["nR"] == 0):
            t_ul = t
        if t_both is None and m["nL"] == 0 and m["nR"] == 0:
            t_both = t
        if i % log_n == 0:
            m["phase"] = phs
            log.append(m)
            emit(viz, sim, phs, TAU_SEC, Rh0, m, rh0)
        i += 1
        if t_drop is not None and t >= t_drop + 2.0:
            break
        if t_lift is not None and t >= t_lift + hold_s:
            break
    end = measure(sim, TAU_SEC)
    return {
        "t0": t0,
        "t_lift_done": t_lift,
        "t_drop": None if t_drop is None else t_drop - t0,
        "t_unilateral": None if t_ul is None else t_ul - t0,
        "t_both_loss": None if t_both is None else t_both - t0,
        "lift_completed": t_lift is not None,
        "drop": t_drop is not None,
        "hold_s": hold_s,
        "hold_survived": bool(t_lift is not None and t_drop is None and (float(sim.data.time) - t_lift) >= hold_s - 1e-6),
        "end": _slim_m(end),
        "log": log,
        "p0z": p0z,
        "noslip": live_opt(sim),
    }


def _slim_m(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def classify_zero(z: dict, t0_ok: bool) -> str:
    if not t0_ok:
        return "INVALID"
    td = z.get("t_drop")
    if td is not None and td < 0.10:
        return "IMMEDIATE"
    if z.get("hold_survived") and not z.get("drop"):
        return "SURVIVE"
    if td is not None and td >= 0.10:
        return "DELAYED_FAIL"
    if z.get("t_both_loss") is not None and z["t_both_loss"] >= 0.10:
        return "DELAYED_FAIL"
    if not z.get("lift_completed") and (z.get("t_unilateral") or 0) >= 0.10:
        return "DELAYED_FAIL"
    return "UNCLEAR"


def save_log_npz(path: Path, log: list) -> None:
    if not log:
        return
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in log]),
        rh=np.stack([r["rh"] for r in log]),
        v_rel_h=np.stack([r["v_rel_h"] for r in log]),
        nL=np.array([r["nL"] for r in log]),
        nR=np.array([r["nR"] for r in log]),
        rho=np.array([r["rho_max"] for r in log]),
        obj_z=np.array([r["obj_z"] for r in log]),
        g_h=np.stack([r["g_h"] for r in log]),
        cyl_h=np.stack([r["cyl_h"] for r in log]),
    )


def hold_steps(sim, gains, tau, seconds, viz, Rh0, rh0, phase, log):
    dt = float(sim.model.opt.timestep)
    n = int(round(seconds / dt))
    log_n = max(int(round(LOG_DT / dt)), 1)
    t_drop = None
    t_ul = None
    for i in range(n):
        tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)
        m = measure(sim, tau)
        if t_drop is None and dropped(sim, m):
            t_drop = float(sim.data.time)
        if t_ul is None and (m["nL"] == 0 or m["nR"] == 0):
            t_ul = float(sim.data.time)
        if i % log_n == 0 or i == n - 1:
            m["phase"] = phase
            m["wrist_deg"] = float(signed_rot_about_y_deg(Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)))
            log.append(m)
            emit(viz, sim, phase, tau, Rh0, m, rh0)
        if t_drop is not None:
            break
    return {"t_drop": t_drop, "t_ul": t_ul, "end": measure(sim, tau)}


def rotate_hy(sim, gains, target_deg, tau, viz, Rh0, rh0, phase, log, omega=OMEGA):
    dt = float(sim.model.opt.timestep)
    sign = 1.0 if target_deg >= 0 else -1.0
    goal = abs(float(target_deg))
    t0 = float(sim.data.time)
    timeout = t0 + goal / max(omega, 1e-6) + 1.5
    log_n = max(int(round(LOG_DT / dt)), 1)
    i = 0
    while float(sim.data.time) < timeout:
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        ang = signed_rot_about_y_deg(Rh0, Rh)
        if abs(ang) >= goal - 0.35:
            break
        w_w = Rh @ np.array([0.0, sign * omega, 0.0])
        tick_vw(sim, np.zeros(3), w_w, tau, gains)
        m = measure(sim, tau)
        if i % log_n == 0:
            m["phase"] = phase
            m["wrist_deg"] = float(signed_rot_about_y_deg(Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)))
            log.append(m)
            emit(viz, sim, phase, tau, Rh0, m, rh0)
        i += 1
        if dropped(sim, m):
            break
    freeze(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return float(signed_rot_about_y_deg(Rh0, Rh))


def return_vertical(sim, gains, tau, viz, Rh0, rh0, log, omega=OMEGA_RET):
    dt = float(sim.model.opt.timestep)
    t0 = float(sim.data.time)
    timeout = t0 + (abs(THETA) * np.pi / 180.0) / max(omega, 1e-6) + 2.0
    log_n = max(int(round(LOG_DT / dt)), 1)
    i = 0
    while float(sim.data.time) < timeout:
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        ang = signed_rot_about_y_deg(Rh0, Rh)
        if abs(ang) < 0.5:
            break
        sign = -1.0 if ang > 0 else 1.0
        w_w = Rh @ np.array([0.0, sign * omega, 0.0])
        tick_vw(sim, np.zeros(3), w_w, tau, gains)
        m = measure(sim, tau)
        if i % log_n == 0:
            m["phase"] = "RETURN"
            m["wrist_deg"] = float(ang)
            log.append(m)
            emit(viz, sim, "RETURN", tau, Rh0, m, rh0)
        i += 1
        if dropped(sim, m):
            break
    freeze(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return float(signed_rot_about_y_deg(Rh0, Rh))


def useful_mm(rh, rh0, u) -> float:
    return float((np.asarray(rh) - np.asarray(rh0)) @ np.asarray(u) * 1e3)


def score_brake(end: dict, rh0, u) -> float:
    rh = np.asarray(end["rh"], float)
    if end["nL"] == 0 or end["nR"] == 0:
        return -1e6
    if dropped(type("S", (), {"dropped": lambda self: False})(), end) and False:
        pass
    if float(end["obj_z"]) < TABLE_DROP:
        return -1e6
    ex = abs(float(rh[0]))
    ez = abs(float(rh[2] - RH_Z_NOM))
    v = float(np.linalg.norm(end["v_rel_h"]))
    w = float(np.linalg.norm(end["w_rel_h"]))
    cyl = np.asarray(end["cyl_h"], float)
    align = abs(float(cyl[2]))  # want |cyl_h.z| ~ 1
    prog = useful_mm(rh, rh0, u)
    return (-800.0 * ex) + (-150.0 * ez) + (-40.0 * v) + (-8.0 * w) + (20.0 * align) + (2.0 * prog)


def run_zero(sim, cfg, gains, sfail, viz=None):
    restore(sim, sfail["snap"])
    freeze(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rh0 = measure(sim, TAU_SEC)["rh"]
    z = continue_nominal(sim, cfg, gains, viz=viz, Rh0=Rh0, rh0=rh0)
    z["class"] = classify_zero(z, bool(sfail["ok"]))
    z["s"] = sfail["s"]
    return z


def probe_slip(sim, cfg, gains, sfail, theta, viz=None):
    restore(sim, sfail["snap"])
    freeze(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    log = []
    hold_steps(sim, gains, TAU_SEC, OBS_S, viz, Rh0, measure(sim, TAU_SEC)["rh"], "OBSERVE", log)
    m0 = measure(sim, TAU_SEC)
    rh0 = m0["rh"].copy()
    rotate_hy(sim, gains, theta, TAU_SEC, viz, Rh0, rh0, "ROTATE", log)
    freeze(sim)
    if viz is not None:
        viz["refs"] = capture_hold_refs(sim, rh0, np.array([1.0, 0.0, 0.0]))
    cands = [TAU_SLIP0, -4.0, -5.0, -2.5]
    seen = set()
    results = []
    post_rot = snap_now(sim)
    for tau in cands:
        if tau in seen or tau < -5.0 or tau > -2.5:
            continue
        seen.add(tau)
        restore(sim, post_rot)
        freeze(sim)
        rh_s = measure(sim, tau)["rh"].copy()
        e_x = float(rh_s[0])
        u = np.array([-np.sign(e_x) if abs(e_x) > 1e-6 else 1.0, 0.0, 0.0])
        dt = float(sim.model.opt.timestep)
        n = int(round(0.40 / dt))
        log_n = max(int(round(LOG_DT / dt)), 1)
        lost = False
        disp = 0.0
        last = None
        for i in range(n):
            tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)
            m = measure(sim, tau)
            last = m
            disp = useful_mm(m["rh"], rh_s, u)
            if m["nL"] == 0 or m["nR"] == 0 or dropped(sim, m):
                lost = True
                break
            if i % log_n == 0:
                emit(viz, sim, "SLIP_PROBE", tau, Rh0, m, rh_s)
        rho = float(last["rho_max"]) if last else float("nan")
        if lost:
            kind = "LOSS"
        elif abs(disp) < 0.35:
            kind = "STICK"
        else:
            kind = "CONTROLLED_SLIP"
        results.append({"tau": tau, "kind": kind, "useful_mm": disp, "rho": rho, "nL": last["nL"], "nR": last["nR"]})
        if kind == "CONTROLLED_SLIP":
            return {"tau": tau, "u": u, "rh_s": rh_s, "post_rot": post_rot, "Rh0": Rh0, "results": results, "m0": _slim_m(m0)}
        if kind == "STICK" and tau == TAU_SLIP0:
            continue
        if kind == "LOSS" and tau == TAU_SLIP0:
            continue
    pick = None
    for r in results:
        if r["kind"] == "CONTROLLED_SLIP":
            pick = r
            break
    return {
        "tau": None if pick is None else pick["tau"],
        "u": np.array([-np.sign(float(m0["e_x"])) if abs(float(m0["e_x"])) > 1e-6 else 1.0, 0.0, 0.0]),
        "rh_s": rh0,
        "post_rot": post_rot,
        "Rh0": Rh0,
        "results": results,
        "m0": _slim_m(m0),
    }


def brake_grid(sim, cfg, gains, sfail, theta, tau_slip, u, viz=None):
    rows = []
    best = None
    best_sc = -1e99
    for mm in (1.0, 2.0, 3.0, 4.0):
        restore(sim, sfail["snap"])
        freeze(sim)
        Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        log = []
        hold_steps(sim, gains, TAU_SEC, OBS_S, viz, Rh0, measure(sim, TAU_SEC)["rh"], "OBSERVE", log)
        rh0 = measure(sim, TAU_SEC)["rh"].copy()
        rotate_hy(sim, gains, theta, TAU_SEC, viz, Rh0, rh0, "ROTATE", log)
        freeze(sim)
        rh_s = measure(sim, tau_slip)["rh"].copy()
        if viz is not None:
            viz["refs"] = capture_hold_refs(sim, rh_s, u)
        dt = float(sim.model.opt.timestep)
        n = int(round(1.20 / dt))
        hit = False
        t_hit = None
        lost = False
        last = None
        for i in range(n):
            tick_vw(sim, np.zeros(3), np.zeros(3), tau_slip, gains)
            m = measure(sim, tau_slip)
            last = m
            prog = useful_mm(m["rh"], rh_s, u)
            if m["nL"] == 0 or m["nR"] == 0 or dropped(sim, m):
                lost = True
                break
            if prog >= mm:
                hit = True
                t_hit = float(sim.data.time)
                if viz is not None and viz.get("refs") is not None:
                    viz["refs"]["rh_brake"] = np.asarray(m["rh"], float).copy()
                break
        if lost or not hit:
            rows.append({"target_mm": mm, "hit": hit, "lost": lost, "prog": None if last is None else useful_mm(last["rh"], rh_s, u), "score": -1e6})
            continue
        freeze(sim)
        br = hold_steps(sim, gains, TAU_SEC, BRAKE_HOLD, viz, Rh0, rh_s, "SECURE", log)
        end = br["end"]
        sc = score_brake(end, rh_s, u)
        rec = {
            "target_mm": mm,
            "hit": True,
            "lost": False,
            "t_hit": t_hit,
            "prog_at_brake_mm": useful_mm(last["rh"], rh_s, u),
            "rh_at_brake": last["rh"].tolist(),
            "end": _slim_m(end),
            "score": sc,
            "snap": snap_now(sim),
            "Rh0": Rh0,
            "rh_s": rh_s,
        }
        rows.append({k: v for k, v in rec.items() if k not in ("snap", "Rh0", "rh_s")})
        if sc > best_sc:
            best_sc = sc
            best = rec
    return rows, best


def run_full(sim, cfg, gains, sfail, theta, tau_slip, u, s_brake, viz=None, do_slip=True, do_brake=True, slip_tau=None):
    """FULL / ROTATE_ONLY / NO_BRAKE from the same S_FAIL."""
    restore(sim, sfail["snap"])
    freeze(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    log = []
    m0 = measure(sim, TAU_SEC)
    rh0 = m0["rh"].copy()
    hold_steps(sim, gains, TAU_SEC, OBS_S, viz, Rh0, rh0, "OBSERVE", log)
    rotate_hy(sim, gains, theta, TAU_SEC, viz, Rh0, rh0, "ROTATE", log)
    freeze(sim)
    phase_break = None
    if do_slip:
        tau = float(slip_tau if slip_tau is not None else tau_slip)
        rh_s = measure(sim, tau)["rh"].copy()
        if viz is not None:
            viz["refs"] = capture_hold_refs(sim, rh_s, u)
        target = float(s_brake["prog_at_brake_mm"]) if do_brake and s_brake else 3.0
        dt = float(sim.model.opt.timestep)
        n = int(round(1.5 / dt))
        hit = not do_brake
        last = None
        for i in range(n):
            tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)
            m = measure(sim, tau)
            last = m
            m["phase"] = "SLIP"
            if i % max(int(round(LOG_DT / dt)), 1) == 0:
                log.append(m)
                emit(viz, sim, "SLIP", tau, Rh0, m, rh_s)
            if dropped(sim, m) or m["nL"] == 0 or m["nR"] == 0:
                phase_break = "SLIP"
                break
            if do_brake and useful_mm(m["rh"], rh_s, u) >= target:
                hit = True
                if viz is not None and viz.get("refs") is not None:
                    viz["refs"]["rh_brake"] = np.asarray(m["rh"], float).copy()
                break
        if do_brake:
            if phase_break:
                return {"broke": phase_break, "log": log, "cont": None}
            freeze(sim)
            hold_steps(sim, gains, TAU_SEC, BRAKE_HOLD, viz, Rh0, rh_s, "BRAKE", log)
            tau_ret = TAU_SEC
        else:
            tau_ret = tau
            if phase_break:
                return {"broke": phase_break, "log": log, "cont": None}
    else:
        tau_ret = TAU_SEC
        rh_s = rh0
    ang = return_vertical(sim, gains, tau_ret, viz, Rh0, rh0, log)
    if dropped(sim, measure(sim, tau_ret)):
        return {"broke": "RETURN", "log": log, "cont": None, "wrist_end": ang}
    freeze(sim)
    vh = hold_steps(sim, gains, TAU_SEC, VERT_HOLD, viz, Rh0, rh0, "VERT_HOLD", log)
    if vh["t_drop"] is not None:
        return {"broke": "VERT_HOLD", "log": log, "cont": None, "vert": _slim_m(vh["end"])}
    freeze(sim)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.p_des = ph.copy()
    cont = continue_nominal(sim, cfg, gains, viz=viz, Rh0=Rh0, rh0=rh0)
    log.extend(cont["log"])
    cont["log"] = []
    return {
        "broke": None,
        "log": log,
        "cont": cont,
        "wrist_end": ang,
        "vert": _slim_m(vh["end"]),
        "success": bool(cont.get("hold_survived") and not cont.get("drop") and cont["end"]["nL"] > 0 and cont["end"]["nR"] > 0),
    }


def plot_cmp(zlog, flog, path: Path, title: str) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    def _t(log):
        t0 = log[0]["t"]
        return np.array([r["t"] - t0 for r in log])
    if zlog:
        tz = _t(zlog)
        ax[0].plot(tz, [1e3 * r["rh"][0] for r in zlog], label="ZERO e_x")
        ax[1].plot(tz, [r["obj_z"] for r in zlog], label="ZERO z")
        ax[2].plot(tz, [r["nL"] for r in zlog], label="ZERO nL")
        ax[2].plot(tz, [r["nR"] for r in zlog], label="ZERO nR", ls="--")
    if flog:
        tf = _t(flog)
        ax[0].plot(tf, [1e3 * r["rh"][0] for r in flog], label="FULL e_x")
        ax[1].plot(tf, [r["obj_z"] for r in flog], label="FULL z")
        ax[2].plot(tf, [r["nL"] for r in flog], label="FULL nL")
        ax[2].plot(tf, [r["nR"] for r in flog], label="FULL nR", ls=":")
    ax[0].set_ylabel("e_x mm")
    ax[1].set_ylabel("obj z")
    ax[2].set_ylabel("contacts")
    ax[2].set_xlabel("t - t0 (s)")
    for a in ax:
        a.grid(True, alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(meta: dict) -> None:
    p = OUT / "GRAV_REPOSITION_RECOVERY_CONSTRUCTION.md"
    a = []
    ap = a.append
    ap("# Gravitational reposition recovery construction (noslip=1)")
    ap("")
    ap("Privileged/manual construction. RULE was not used. SAC was not trained.")
    ap("")
    ap("## USER VISUAL OBSERVATION")
    ap("")
    ap("- Simple CONTROLLED SLIP: **CONFIRMED** (prior primitive viewer).")
    ap("- S_FAIL recovery FULL: **PENDING** until the user watches `--mode full`.")
    ap("")
    ap("## 1. noslip=1 S_FAIL revalidation")
    ap("")
    ap("```json")
    ap(json.dumps(meta["sfail_search"], indent=2, default=str))
    ap("```")
    ap("")
    ap(f"S_FAIL s = **{meta['s']}**. ZERO class = **{meta['zero']['class']}**.")
    ap("")
    ap("## 2. Exact disturbed initial state")
    ap("")
    ap("```json")
    ap(json.dumps(meta["sfail_state"], indent=2, default=str))
    ap("```")
    ap("")
    ap("## 3. Wrist-direction derivation")
    ap("")
    ap("```json")
    ap(json.dumps(meta["wrist"], indent=2, default=str))
    ap("```")
    ap("")
    ap("## 4. Local controlled-slip regime")
    ap("")
    ap("```json")
    ap(json.dumps(meta["slip"], indent=2, default=str))
    ap("```")
    ap("")
    ap("## 5. Brake-target comparison")
    ap("")
    ap("```json")
    ap(json.dumps(meta["brake_rows"], indent=2, default=str))
    ap("```")
    ap("")
    ap("## 6. Selected S_BRAKE")
    ap("")
    ap("```json")
    ap(json.dumps(meta["s_brake_why"], indent=2, default=str))
    ap("```")
    ap("")
    ap("## 7–9. Return / vertical hold / continuation (FULL)")
    ap("")
    ap("```json")
    ap(json.dumps(meta["full_summary"], indent=2, default=str))
    ap("```")
    ap("")
    ap("## 10. ZERO vs ROTATE_ONLY vs NO_BRAKE vs FULL")
    ap("")
    ap("```json")
    ap(json.dumps(meta["ablation"], indent=2, default=str))
    ap("```")
    ap("")
    ap("## 11. Raw trajectory evidence")
    ap("")
    ap("See `raw/*.npz` and `figures/`.")
    ap("")
    ap("## 12. Viewer commands")
    ap("")
    ap("```text")
    ap("python training/demo_grav_reposition_recovery.py --mode zero")
    ap("python training/demo_grav_reposition_recovery.py --mode full")
    ap("python training/demo_grav_reposition_recovery.py --mode rotate_only")
    ap("python training/demo_grav_reposition_recovery.py --mode no_brake")
    ap("```")
    ap("")
    ap("## 13. USER VISUAL OBSERVATION: pending")
    ap("")
    ap("## 14. Unresolved")
    ap("")
    ap(meta.get("unresolved", "-"))
    p.write_text("\n".join(a), encoding="utf-8")


def construction(sim=None, viz=None, viewer_mode: str | None = None) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    if sim is None:
        sim = make_parent_sim()
    opt = live_opt(sim)
    if int(opt["noslip_iterations"]) != 1:
        raise RuntimeError(opt)
    parent_pack = advance_to_parent(sim, cfg)
    if not parent_pack.get("ok"):
        raise RuntimeError(parent_pack)
    freeze(sim)
    parent = snap_now(sim)
    with (RAW / "parent.pkl").open("wb") as f:
        pickle.dump(parent, f)
    if viz is not None:
        init_camera_once(viz, sim)
        s0 = measure(sim, TAU_SEC)
        wall_pause(
            viz, PRE_VIEW_PAUSE_S, sim,
            phase="PAUSED — adjust camera", wrist_deg=0.0, tau=TAU_SEC,
            rho=float(s0["rho_max"]), drh_mm=np.zeros(3), progress_mm=0.0,
            v_rel_h=s0["v_rel_h"], nL=s0["nL"], nR=s0["nR"],
        )
    ref = load_or_measure_ref()
    gains = gains_from_cfg(cfg)
    search_s = [2.0, 1.8, 1.9, 2.1, 2.2, 1.6, 2.4]
    search = []
    sfail = None
    for s in search_s:
        ic = apply_sfail(sim, parent, ref, s)
        z = run_zero(sim, cfg, gains, ic, viz=None)
        rec = {"s": s, "t0_ok": ic["ok"], "why": ic["why"], "class": z["class"],
               "lift": z["lift_completed"], "drop": z["drop"], "t_drop": z["t_drop"],
               "hold_survived": z["hold_survived"], "end_nL": z["end"]["nL"], "end_nR": z["end"]["nR"]}
        search.append(rec)
        if ic["ok"] and z["class"] == "DELAYED_FAIL":
            sfail = ic
            zero = z
            break
        if s == 2.0:
            zero20 = z
            ic20 = ic
    if sfail is None:
        meta = {
            "sfail_search": search,
            "s": None,
            "zero": zero20 if "zero20" in dir() else search[0],
            "sfail_state": ic20["after"] if "ic20" in dir() else {},
            "wrist": {},
            "slip": {},
            "brake_rows": [],
            "s_brake_why": "STOP: no delayed-failure S_FAIL in local s search.",
            "full_summary": {},
            "ablation": {},
            "unresolved": "s=2.0 (and neighbors) did not produce delayed task failure under noslip=1. Not used as recovery benchmark.",
        }
        dump_json(RAW / "summary.json", meta)
        write_report(meta)
        return meta

    with (RAW / "sfail.pkl").open("wb") as f:
        pickle.dump(sfail["snap"], f)
    dump_json(RAW / "sfail_t0.json", sfail["after"])
    save_log_npz(RAW / "zero.npz", zero["log"])
    dump_json(RAW / "zero.json", {k: v for k, v in zero.items() if k != "log"})

    # geometry audit 100 ms
    restore(sim, sfail["snap"])
    freeze(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    log_obs = []
    hold_steps(sim, gains, TAU_SEC, OBS_S, None, Rh0, measure(sim, TAU_SEC)["rh"], "OBSERVE", log_obs)
    geo = measure(sim, TAU_SEC)
    e_x = float(geo["e_x"])
    theta = THETA if e_x > 0 else -THETA
    wrist = {
        "e_x_mm": 1e3 * e_x,
        "rh": geo["rh"].tolist(),
        "g_h": geo["g_h"].tolist(),
        "cyl_h": geo["cyl_h"].tolist(),
        "nL": geo["nL"],
        "nR": geo["nR"],
        "rho": geo["rho_max"],
        "Fn_L": geo["Fn_L"],
        "Fn_R": geo["Fn_R"],
        "v_rel_h": geo["v_rel_h"].tolist(),
        "w_rel_h": geo["w_rel_h"].tolist(),
        "rule": "primitive +30 deg => g_h.x < 0 => Delta r_h.x < 0. If e_x>0, command +30 to drive object toward r_h.x=0.",
        "theta_cmd_deg": theta,
    }
    dump_json(RAW / "geometry_100ms.json", wrist)

    if viewer_mode == "zero":
        z = run_zero(sim, cfg, gains, sfail, viz=viz)
        save_log_npz(RAW / "viewer_zero.npz", z["log"])
        return {"viewer": "zero", "zero": {k: v for k, v in z.items() if k != "log"}}

    pr = probe_slip(sim, cfg, gains, sfail, theta, viz=None)
    dump_json(RAW / "slip_probe.json", {"tau": pr["tau"], "results": pr["results"], "m0": pr["m0"]})
    if pr["tau"] is None:
        meta = {
            "sfail_search": search,
            "s": sfail["s"],
            "zero": {k: v for k, v in zero.items() if k != "log"},
            "sfail_state": sfail["after"],
            "wrist": wrist,
            "slip": pr["results"],
            "brake_rows": [],
            "s_brake_why": "STOP: no local CONTROLLED_SLIP in tau [-5,-2.5] on S_FAIL.",
            "full_summary": {},
            "ablation": {},
            "unresolved": "S_FAIL exists as delayed ZERO failure, but no controlled-slip tau in the allowed band.",
        }
        dump_json(RAW / "summary.json", meta)
        write_report(meta)
        return meta

    u = np.asarray(pr["u"], float)
    rows, best = brake_grid(sim, cfg, gains, sfail, theta, pr["tau"], u, viz=None)
    dump_json(RAW / "brake_grid.json", rows)
    if best is None:
        meta = {
            "sfail_search": search,
            "s": sfail["s"],
            "zero": {k: v for k, v in zero.items() if k != "log"},
            "sfail_state": sfail["after"],
            "wrist": wrist,
            "slip": {"tau": pr["tau"], "results": pr["results"]},
            "brake_rows": rows,
            "s_brake_why": "STOP: no bilateral brake target 1–4 mm.",
            "full_summary": {},
            "ablation": {},
            "unresolved": "Controlled slip exists but no 1–4 mm brake endpoint stayed bilateral.",
        }
        dump_json(RAW / "summary.json", meta)
        write_report(meta)
        return meta

    why = {
        "target_mm": best["target_mm"],
        "score": best["score"],
        "prog_at_brake_mm": best["prog_at_brake_mm"],
        "end_e_x_mm": 1e3 * best["end"]["e_x"],
        "end_rh": best["end"]["rh"],
        "end_nL": best["end"]["nL"],
        "end_nR": best["end"]["nR"],
        "end_rho": best["end"]["rho_max"],
        "end_vrel": float(np.linalg.norm(best["end"]["v_rel_h"])),
        "reason": "Highest score among bilateral 0.5 s secure holds: smaller |e_x|, |e_z|, |v_rel|, |w_rel|, better cyl axis z, without using lift success.",
    }
    with (RAW / "s_brake.pkl").open("wb") as f:
        pickle.dump(best["snap"], f)

    if viewer_mode in ("full", "rotate_only", "no_brake"):
        if viewer_mode == "full":
            out = run_full(sim, cfg, gains, sfail, theta, pr["tau"], u, best, viz=viz, do_slip=True, do_brake=True)
        elif viewer_mode == "rotate_only":
            out = run_full(sim, cfg, gains, sfail, theta, pr["tau"], u, best, viz=viz, do_slip=False, do_brake=False)
        else:
            out = run_full(sim, cfg, gains, sfail, theta, pr["tau"], u, best, viz=viz, do_slip=True, do_brake=False, slip_tau=pr["tau"])
        save_log_npz(RAW / f"viewer_{viewer_mode}.npz", out.get("log") or [])
        return {"viewer": viewer_mode, "success": out.get("success"), "broke": out.get("broke")}

    full = run_full(sim, cfg, gains, sfail, theta, pr["tau"], u, best, viz=None, do_slip=True, do_brake=True)
    save_log_npz(RAW / "full.npz", full.get("log") or [])
    rot = run_full(sim, cfg, gains, sfail, theta, pr["tau"], u, best, viz=None, do_slip=False, do_brake=False)
    nob = run_full(sim, cfg, gains, sfail, theta, pr["tau"], u, best, viz=None, do_slip=True, do_brake=False, slip_tau=pr["tau"])
    save_log_npz(RAW / "rotate_only.npz", rot.get("log") or [])
    save_log_npz(RAW / "no_brake.npz", nob.get("log") or [])

    def _sum(name, r, z=None):
        if z is not None:
            c = z
            return {
                "mode": name,
                "class": z.get("class"),
                "lift": z.get("lift_completed"),
                "drop": z.get("drop"),
                "t_drop": z.get("t_drop"),
                "hold_survived": z.get("hold_survived"),
                "end_nL": z["end"]["nL"],
                "end_nR": z["end"]["nR"],
                "end_e_x_mm": 1e3 * z["end"]["e_x"],
                "end_obj_z": z["end"]["obj_z"],
            }
        c = r.get("cont") or {}
        end = c.get("end") or r.get("vert") or {}
        return {
            "mode": name,
            "broke": r.get("broke"),
            "success": r.get("success"),
            "lift": c.get("lift_completed"),
            "drop": c.get("drop"),
            "t_drop": c.get("t_drop"),
            "hold_survived": c.get("hold_survived"),
            "end_nL": end.get("nL"),
            "end_nR": end.get("nR"),
            "end_e_x_mm": None if end.get("e_x") is None else 1e3 * end["e_x"],
            "end_obj_z": end.get("obj_z"),
        }

    ab = {
        "ZERO": _sum("ZERO", None, z=zero),
        "ROTATE_ONLY": _sum("ROTATE_ONLY", rot),
        "NO_BRAKE": _sum("NO_BRAKE", nob),
        "FULL": _sum("FULL", full),
    }
    plot_cmp(zero["log"], full.get("log"), FIG / "zero_vs_full.png", f"ZERO vs FULL s={sfail['s']}")
    full_summary = {
        "broke": full.get("broke"),
        "success": full.get("success"),
        "cont": {k: v for k, v in (full.get("cont") or {}).items() if k != "log"},
        "vert": full.get("vert"),
        "wrist_end": full.get("wrist_end"),
    }
    unresolved = []
    if not full.get("success"):
        unresolved.append(f"FULL broke at {full.get('broke')} or failed continuation.")
    if ab["ROTATE_ONLY"].get("success"):
        unresolved.append("ROTATE_ONLY also survived — wrist rotation alone may explain success.")
    if not ab["ZERO"]["drop"] and ab["ZERO"].get("hold_survived"):
        unresolved.append("ZERO survived; S_FAIL is not a failure IC.")
    meta = {
        "sfail_search": search,
        "s": sfail["s"],
        "live": opt,
        "zero": {k: v for k, v in zero.items() if k != "log"},
        "sfail_state": sfail["after"],
        "wrist": wrist,
        "slip": {"tau": pr["tau"], "results": pr["results"]},
        "brake_rows": rows,
        "s_brake_why": why,
        "full_summary": full_summary,
        "ablation": ab,
        "unresolved": " ".join(unresolved) if unresolved else "None identified in this construction pass.",
    }
    dump_json(RAW / "summary.json", meta)
    write_report(meta)
    return meta


def run_self_arrest_viewer(sim, cfg, gains, sfail, theta, tau_slip, u, viz) -> None:
    """Rotate +30 at tau=-18, then tau=-4 at frozen tilt for >=5 s. Never restore -18."""
    tau_w = float(tau_slip if tau_slip is not None else -4.0)
    restore(sim, sfail["snap"])
    freeze(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    log = []
    m0 = measure(sim, TAU_SEC)
    rh0 = m0["rh"].copy()
    hold_steps(sim, gains, TAU_SEC, OBS_S, viz, Rh0, rh0, "OBSERVE", log)
    rotate_hy(sim, gains, theta, TAU_SEC, viz, Rh0, rh0, "ROTATE", log)
    freeze(sim)
    rh_s = measure(sim, tau_w)["rh"].copy()
    if viz is not None:
        viz["refs"] = capture_hold_refs(sim, rh_s, u)
    hold_steps(sim, gains, tau_w, 6.0, viz, Rh0, rh_s, "SELF_ARREST", log)


def _load_saved_sfail():
    pkl = RAW / "sfail.pkl"
    summary_p = RAW / "summary.json"
    if not pkl.is_file() or not summary_p.is_file():
        return None
    with pkl.open("rb") as f:
        snap = pickle.load(f)
    meta = json.loads(summary_p.read_text(encoding="utf-8"))
    sfail = {
        "snap": snap,
        "ok": True,
        "s": float(meta.get("s", 2.0)),
        "after": meta.get("sfail_state") or {},
    }
    theta = float((meta.get("wrist") or {}).get("theta_cmd_deg", THETA))
    tau = (meta.get("slip") or {}).get("tau")
    why = meta.get("s_brake_why") or {}
    e_x = float((meta.get("sfail_state") or {}).get("rh", [1.0, 0, 0])[0])
    u = np.array([-np.sign(e_x) if abs(e_x) > 1e-6 else 1.0, 0.0, 0.0])
    best = {"prog_at_brake_mm": float(why.get("prog_at_brake_mm", 1.0))}
    return sfail, theta, tau, u, best, meta


def viewer(mode: str) -> None:
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    key_holder = {}

    def _key(keycode):
        fn = key_holder.get("fn")
        if fn:
            fn(keycode)

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=_key) as vwr:
        viz = {"viewer": vwr, "camera": "closeup", "playback_speed": PLAYBACK, "pace": True, "refs": None}
        key_holder["fn"] = make_reset_cam_callback(viz)
        saved = _load_saved_sfail()
        if saved is None:
            construction(sim=sim, viz=viz, viewer_mode=mode)
            return
        sfail, theta, tau, u, best, _meta = saved
        parent_p = RAW / "parent.pkl"
        if parent_p.is_file():
            with parent_p.open("rb") as f:
                restore(sim, pickle.load(f))
            freeze(sim)
        else:
            restore(sim, sfail["snap"])
            freeze(sim)
        init_camera_once(viz, sim)
        s0 = measure(sim, TAU_SEC)
        wall_pause(
            viz, PRE_VIEW_PAUSE_S, sim,
            phase="PAUSED — adjust camera", wrist_deg=0.0, tau=TAU_SEC,
            rho=float(s0["rho_max"]), drh_mm=np.zeros(3), progress_mm=0.0,
            v_rel_h=s0["v_rel_h"], nL=s0["nL"], nR=s0["nR"],
        )
        gains = gains_from_cfg(cfg)
        if mode == "zero":
            run_zero(sim, cfg, gains, sfail, viz=viz)
        elif mode == "rotate_only":
            run_full(sim, cfg, gains, sfail, theta, tau, u, best, viz=viz, do_slip=False, do_brake=False)
        elif mode == "no_brake":
            run_full(sim, cfg, gains, sfail, theta, tau, u, best, viz=viz, do_slip=True, do_brake=False, slip_tau=tau)
        elif mode == "self_arrest":
            run_self_arrest_viewer(sim, cfg, gains, sfail, theta, tau, u, viz)
        else:
            run_full(sim, cfg, gains, sfail, theta, tau, u, best, viz=viz, do_slip=True, do_brake=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="", help="zero|full|rotate_only|no_brake|self_arrest (viewer)")
    args = p.parse_args()
    if args.mode:
        viewer(args.mode)
        return
    construction()


if __name__ == "__main__":
    main()
