"""CENTER-6 ballistic contact-preserving recovery authority. ZERO vs 4D probes.

No MP4. No SAC. No recatch. Frozen impact: CENTER 6 m/s, m_ball=0.05 kg.
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
    RECOVERY4D_TAU_SECURE,
    RECOVERY4D_V_HX_MAX,
    RECOVERY4D_V_Z_MAX,
    RECOVERY4D_W_HY_MAX,
    map_recovery4d,
)
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import TABLE_DROP, physical_pack
from training.demo_ballistic_impact import TAU_SEC, dump_json, live_opt, make_sim, z_tgt_of
from training.demo_teleport_recovery_state import signed_rot_about_y_deg
from training.grav_reposition_v2_viz import apply_camera_preset
from training.map_ballistic_disturbance import (
    CAM,
    _viewer_overlay,
    run_episode,
)
from training.replay_core import tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts
from training.write_ballistic_impact_scene import BALL_M

OUT = ROOT / "results" / "diagnostics" / "ballistic_recovery_transfer"
RAW = OUT / "raw"
V_HIT = 6.0
SITE = "CENTER"
HOLD_S = 12.0
AFTER_DROP_S = 2.0


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def axis_tilt_deg(R_rel) -> float:
    c = float(np.clip(np.asarray(R_rel, float).reshape(3, 3)[2, 2], -1.0, 1.0))
    ang = float(np.degrees(np.arccos(abs(c))))
    return ang


def make_snap(sim) -> dict:
    s = sim.snapshot()
    s["friction"] = 1.0
    s["eq_active"] = int(sim.eq_active())
    s["mocap_pos"] = np.array(sim.data.mocap_pos, float).copy()
    s["mocap_quat"] = np.array(sim.data.mocap_quat, float).copy()
    s["z_tgt"] = z_tgt_of(sim)
    return s


def restore_ballistic(sim, snap) -> None:
    sim.load_snapshot(snap)
    on = bool(int(snap.get("eq_active", 0)))
    sim.set_guide_weld(on)
    if "mocap_pos" in snap:
        sim.data.mocap_pos[:] = np.asarray(snap["mocap_pos"], float)
        sim.data.mocap_quat[:] = np.asarray(snap["mocap_quat"], float)
    mujoco.mj_forward(sim.model, sim.data)


def measure(sim, tau: float, Rh0=None) -> dict:
    o = physical_pack(sim)
    fc, _ = extract_contacts(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    return {
        "t": float(sim.data.time),
        "rh": np.asarray(o["rh"], float).copy(),
        "tilt_deg": axis_tilt_deg(o["R_rel"]),
        "v_rel_h": np.asarray(o["v_rel_h"], float).copy(),
        "w_rel_h": np.asarray(o["w_rel_h"], float).copy(),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "Fn_L": float(sum(abs(r["Fn"]) for r in fc if r["side"] == "L")),
        "Fn_R": float(sum(abs(r["Fn"]) for r in fc if r["side"] == "R")),
        "rho_max": float(np.nanmax([r["rho"] for r in fc])) if fc else float("nan"),
        "aperture": finger_opening(sim.data, sim.ids),
        "obj_z": float(o["obj_z"]),
        "clear": float(o.get("clear", np.nan)),
        "g_h": np.asarray(o["g_h"], float).copy(),
        "p_hand": ph.copy(),
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "fsm": str(sim.fsm.phase),
        "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float(tau),
        "tau_cmd": float(tau),
        "wrist_deg": 0.0 if Rh0 is None else float(signed_rot_about_y_deg(Rh0, Rh)),
        "n_contacts": len(fc),
    }


def slim_m(m: dict) -> dict:
    return _jsonable(m)


def define_stages(ep: dict) -> dict:
    rows = ep["rows"]
    ti = float(ep["t_impact"])
    t_table = ep.get("t_table_contact")
    t_restore = ep.get("t_bilateral_restored")
    post = [r for r in rows if r["t"] >= ti + 0.020]
    t_trans = None
    persist = 0.0
    dt = 0.002
    for r in post:
        vr = float(np.linalg.norm(r["v_rel_h"]))
        ok = vr < 0.015 and r["nL"] > 0 and r["nR"] > 0
        persist = persist + dt if ok else 0.0
        if persist >= 0.020:
            t_trans = float(r["t"])
            break
    if t_trans is None:
        t_trans = ti + 0.050
    rh_t = abs(next(r["rh"][0] for r in rows if r["t"] >= t_trans))
    t_early = None
    for r in rows:
        if r["t"] < t_trans + 0.15:
            continue
        if r["nL"] > 0 and r["nR"] > 0 and abs(r["rh"][0]) >= rh_t + 0.0004:
            if float(np.linalg.norm(r["v_rel_h"])) < 0.012:
                t_early = float(r["t"])
                break
    if t_early is None:
        t_early = ti + 0.40
    t_mid = None
    for r in rows:
        if r["t"] < t_early + 0.40:
            continue
        if r["nL"] > 0 and r["nR"] > 0 and abs(r["rh"][0]) >= 0.0125:
            t_mid = float(r["t"])
            break
    if t_mid is None:
        t_mid = min((t_table or (ti + 2.4)) - 0.40, ti + 2.4)
    t_late = None
    for r in rows:
        if r["t"] < t_mid + 0.20:
            continue
        if r["nL"] > 0 and r["nR"] > 0 and abs(r["rh"][0]) >= 0.0135:
            t_late = float(r["t"])
            break
    t_run = None
    for r in rows:
        if r["t"] < t_mid:
            continue
        if abs(r["rh"][0]) >= 0.016 and float(np.linalg.norm(r["v_rel_h"])) > 0.015:
            t_run = float(r["t"])
            break
    def at(t):
        cand = [r for r in rows if r["t"] >= t]
        return cand[0] if cand else rows[-1]
    out = {
        "T_IMPACT": ti,
        "T_TRANSIENT_END": t_trans,
        "T_EARLY_DRIFT": t_early,
        "T_MID_DRIFT": t_mid,
        "T_LATE": t_late,
        "T_RUNAWAY": t_run,
        "T_DROP": t_table,
        "t_bilateral_restored": t_restore,
        "t_lift_done": ep.get("t_lift_done"),
        "samples": {
            "impact": slim_m({"t": at(ti)["t"], "rh": at(ti)["rh"], "nL" : at(ti)["nL"], "nR": at(ti)["nR"], "v_rel": float(np.linalg.norm(at(ti)["v_rel_h"]))}),
            "transient_end": slim_m({"t": at(t_trans)["t"], "rh": at(t_trans)["rh"], "nL": at(t_trans)["nL"], "nR": at(t_trans)["nR"], "v_rel": float(np.linalg.norm(at(t_trans)["v_rel_h"])), "tilt": axis_tilt_deg(at(t_trans)["R_rel"])}),
            "early": slim_m({"t": at(t_early)["t"], "rh": at(t_early)["rh"], "nL": at(t_early)["nL"], "nR": at(t_early)["nR"], "v_rel": float(np.linalg.norm(at(t_early)["v_rel_h"]))}),
            "mid": slim_m({"t": at(t_mid)["t"], "rh": at(t_mid)["rh"], "nL": at(t_mid)["nL"], "nR": at(t_mid)["nR"], "v_rel": float(np.linalg.norm(at(t_mid)["v_rel_h"]))}),
        },
    }
    return out


def tick_4d(sim, gains, omega_y, v_x, v_z, tau):
    omega_y = float(np.clip(omega_y, -RECOVERY4D_W_HY_MAX, RECOVERY4D_W_HY_MAX))
    v_x = float(np.clip(v_x, -RECOVERY4D_V_HX_MAX, RECOVERY4D_V_HX_MAX))
    v_z = float(np.clip(v_z, -RECOVERY4D_V_Z_MAX, RECOVERY4D_V_Z_MAX))
    a0 = omega_y / RECOVERY4D_W_HY_MAX if RECOVERY4D_W_HY_MAX else 0.0
    a1 = v_x / RECOVERY4D_V_HX_MAX
    a2 = v_z / RECOVERY4D_V_Z_MAX
    a3 = 2.0 * (float(tau) - RECOVERY4D_TAU_OPEN) / (RECOVERY4D_TAU_SECURE - RECOVERY4D_TAU_OPEN) - 1.0
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    cmd = map_recovery4d(np.array([a0, a1, a2, a3]), sim.fsm.r_des, Rh)
    tick_vw(sim, cmd["v_world"], cmd["w_world"], cmd["tau"], gains)
    return cmd


def continue_zero(sim, gains, duration: float, log, Rh0, z_tgt, ctl=None):
    dt = float(sim.model.opt.timestep)
    n = int(round(duration / dt))
    t_table = None
    for _ in range(max(n, 0)):
        if ctl and ctl.get("reset"):
            break
        if float(sim.fsm.p_des[2]) >= z_tgt - 1e-9:
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        else:
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
        m = measure(sim, TAU_SEC, Rh0)
        if log is not None and (len(log) == 0 or m["t"] - log[-1]["t"] >= 0.008):
            log.append(m)
        if t_table is None and m["obj_z"] < TABLE_DROP:
            t_table = m["t"]
            break
        if ctl and ctl.get("sync"):
            ctl["overlay"] = {
                "case": ctl.get("label", "ZERO"),
                "t": m["t"],
                "phase": m["fsm"],
                "nL": m["nL"],
                "nR": m["nR"],
            }
            ctl["sync"]()
    return t_table


def run_zero_full(sim, gains, capture_at=None, ctl=None, label="ZERO CENTER-6"):
    if ctl is None:
        ctl = {}
    ctl["make_snap"] = make_snap
    if capture_at:
        ctl["capture_at"] = capture_at
        ctl.setdefault("snaps", {})
    ep = run_episode(sim, gains, V_HIT, SITE, ctl=ctl, label=label)
    ep["snaps"] = ctl.get("snaps") or {}
    return ep


def save_snap_file(snap: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(snap, f)


def restore_audit(sim, gains, snap, name: str, orig_rows, t0: float) -> dict:
    restore_ballistic(sim, snap)
    z_tgt = float(snap.get("z_tgt", z_tgt_of(sim)))
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    q0 = np.array(sim.data.qpos, float).copy()
    pdes0 = np.array(sim.fsm.p_des, float).copy()
    chk = {
        "name": name,
        "t0": float(sim.data.time),
        "phase": str(sim.fsm.phase),
        "p_des": pdes0.tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "ctrl7": float(sim.data.ctrl[7]),
        "eq_active": sim.eq_active(),
        "qpos_restored": bool(np.allclose(q0, snap["qpos"], atol=1e-12)),
        "p_des_restored": bool(np.allclose(pdes0, snap["p_des"], atol=1e-12)),
        "time_restored": abs(float(sim.data.time) - float(snap["time"])) < 1e-12,
    }
    log = []
    t_table = continue_zero(sim, gains, 8.0, log, Rh0, z_tgt)
    # compare rh.x vs original at +0.2, +0.5, +1.0 if available
    cmp = []
    for tau in (0.20, 0.50, 1.00, 2.00):
        tt = t0 + tau
        o = [r for r in orig_rows if r["t"] >= tt]
        p = [r for r in log if r["t"] >= tt]
        if o and p:
            cmp.append(
                {
                    "tau": tau,
                    "orig_rhx_mm": 1e3 * float(o[0]["rh"][0]),
                    "replay_rhx_mm": 1e3 * float(p[0]["rh"][0]),
                    "err_mm": 1e3 * abs(float(o[0]["rh"][0]) - float(p[0]["rh"][0])),
                    "orig_n": [o[0]["nL"], o[0]["nR"]],
                    "replay_n": [p[0]["nL"], p[0]["nR"]],
                }
            )
    chk["table"] = t_table
    chk["compare"] = cmp
    chk["max_err_mm"] = max((c["err_mm"] for c in cmp), default=None)
    chk["replay_dropped"] = t_table is not None
    return chk


def probe(sim, gains, snap, spec: dict) -> dict:
    restore_ballistic(sim, snap)
    z_tgt = float(snap.get("z_tgt", z_tgt_of(sim)))
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    m0 = measure(sim, spec["tau"], Rh0)
    dt = float(sim.model.opt.timestep)
    n = int(round(float(spec["dur"]) / dt))
    log = [m0]
    t_table = None
    for _ in range(n):
        tick_4d(sim, gains, spec["omega_y"], spec["v_x"], spec["v_z"], spec["tau"])
        m = measure(sim, spec["tau"], Rh0)
        if m["t"] - log[-1]["t"] >= 0.008:
            log.append(m)
        if m["obj_z"] < TABLE_DROP:
            t_table = m["t"]
            break
    m1 = measure(sim, spec["tau"], Rh0)
    t_after = None
    if t_table is None:
        t_after = continue_zero(sim, gains, 0.50, log, Rh0, z_tgt)
        t_table = t_after
    m2 = log[-1]
    ap0, ap1 = float(m0["aperture"]), float(m1["aperture"])
    dap = (ap1 - ap0) / max(spec["dur"], 1e-6)
    return {
        "name": spec["name"],
        "stage": spec["stage"],
        "cmd": {k: spec[k] for k in ("omega_y", "v_x", "v_z", "tau", "dur")},
        "t0": m0["t"],
        "start": slim_m(m0),
        "end_pulse": slim_m(m1),
        "end_after": slim_m(m2),
        "drhx_pulse_mm": 1e3 * (float(m1["rh"][0]) - float(m0["rh"][0])),
        "drhx_after_mm": 1e3 * (float(m2["rh"][0]) - float(m0["rh"][0])),
        "dvrel": float(np.linalg.norm(m1["v_rel_h"])) - float(np.linalg.norm(m0["v_rel_h"])),
        "d_wrist_deg": float(m1["wrist_deg"]) - float(m0["wrist_deg"]),
        "d_g_h": (np.asarray(m1["g_h"]) - np.asarray(m0["g_h"])).tolist(),
        "d_p_hand_mm": (1e3 * (np.asarray(m1["p_hand"]) - np.asarray(m0["p_hand"]))).tolist(),
        "contacts_kept": int(m1["nL"] > 0 and m1["nR"] > 0),
        "daperture": dap,
        "table": t_table,
        "log": [slim_m(x) for x in log[:: max(1, len(log)//40)]],
    }


def pick_existence(probes: list[dict]) -> dict | None:
    early = [p for p in probes if p["stage"] == "EARLY" and p["table"] is None and p["contacts_kept"]]
    if not early:
        early = [p for p in probes if p["stage"] == "MID" and p["table"] is None and p["contacts_kept"]]
    scored = []
    for p in early:
        # useful: reduce |rh.x| without losing contact
        rh0 = abs(float(p["start"]["rh"][0]))
        rh1 = abs(float(p["end_pulse"]["rh"][0]))
        if rh1 < rh0 - 0.0015:
            scored.append((rh0 - rh1, p))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best = scored[0][1]
    return {
        "stage": best["stage"],
        "reason": "largest |rh.x| reduction with contacts kept, no table in pulse+0.5s",
        "base_probe": best["name"],
        "steps": [
            {"omega_y": 0.0, "v_x": best["cmd"]["v_x"], "v_z": 0.0, "tau": -18.0, "dur": 0.25, "label": "FOLLOW_X"},
            {"omega_y": 0.0, "v_x": 0.0, "v_z": 0.0, "tau": -18.0, "dur": 0.40, "label": "SECURE"},
        ],
    }


def apply_plan(sim, gains, plan, z_tgt, ctl=None, Rh0=None):
    if Rh0 is None:
        Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    log = []
    for step in plan["steps"]:
        dt = float(sim.model.opt.timestep)
        n = int(round(float(step["dur"]) / dt))
        for _ in range(n):
            tick_4d(sim, gains, step["omega_y"], step["v_x"], step["v_z"], step["tau"])
            m = measure(sim, step["tau"], Rh0)
            if not log or m["t"] - log[-1]["t"] >= 0.008:
                log.append(m)
            if m["obj_z"] < TABLE_DROP:
                return log, m["t"]
            if ctl and ctl.get("sync"):
                ctl["overlay"] = {
                    "case": step.get("label", "RECOVERY"),
                    "t": m["t"],
                    "phase": step.get("label", "RECOVERY"),
                    "nL": m["nL"],
                    "nR": m["nR"],
                }
                ctl["sync"]()
    t_table = continue_zero(sim, gains, HOLD_S + 4.0, log, Rh0, z_tgt, ctl)
    return log, t_table


def run_center6_until(sim, gains, t_stop, ctl=None, label="PRE"):
    ctl = dict(ctl or {})
    ctl["make_snap"] = make_snap
    ctl["capture_at"] = {"GO": t_stop}
    ctl["snaps"] = {}
    dt = float(sim.model.opt.timestep)
    inner = ctl.get("sync")

    def sync_and_stop():
        if inner:
            inner()
        if "GO" in ctl.get("snaps", {}):
            ctl["abort"] = True

    ctl["sync"] = sync_and_stop
    ep = run_episode(sim, gains, V_HIT, SITE, ctl=ctl, label=label)
    return ep, ctl.get("snaps", {})


def headless():
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    sim, _ = make_sim()
    print("ZERO CENTER-6", flush=True)
    ep = run_zero_full(sim, gains)
    dump_json(RAW / "zero_center6.json", {k: _jsonable(v) for k, v in ep.items() if k not in ("rows", "snaps", "snap")})
    np.savez_compressed(
        RAW / "zero_center6.npz",
        t=np.array([r["t"] for r in ep["rows"]]),
        rh=np.array([r["rh"] for r in ep["rows"]]),
        nL=np.array([r["nL"] for r in ep["rows"]]),
        nR=np.array([r["nR"] for r in ep["rows"]]),
        vrel=np.array([np.linalg.norm(r["v_rel_h"]) for r in ep["rows"]]),
        wrel=np.array([np.linalg.norm(r["w_rel_h"]) for r in ep["rows"]]),
        tilt=np.array([axis_tilt_deg(r["R_rel"]) for r in ep["rows"]]),
        obj_z=np.array([r["obj_z"] for r in ep["rows"]]),
        aperture=np.array([r.get("aperture", np.nan) for r in ep["rows"]]),
        Fn_L=np.array([r["Fn_L"] for r in ep["rows"]]),
        Fn_R=np.array([r["Fn_R"] for r in ep["rows"]]),
    )
    stages = define_stages(ep)
    dump_json(RAW / "stages.json", stages)
    print("stages", stages["T_IMPACT"], stages["T_TRANSIENT_END"], stages["T_EARLY_DRIFT"], stages["T_MID_DRIFT"], stages["T_RUNAWAY"], stages["T_DROP"], flush=True)

    cap = {
        "EARLY": stages["T_EARLY_DRIFT"],
        "MID": stages["T_MID_DRIFT"],
    }
    if stages["T_LATE"] is not None:
        cap["LATE"] = stages["T_LATE"]
    sim, _ = make_sim()
    epc = run_zero_full(sim, gains, capture_at=cap)
    snaps = epc["snaps"]
    for k, s in snaps.items():
        save_snap_file(s, RAW / f"snap_{k.lower()}.pkl")
        dump_json(RAW / f"snap_{k.lower()}_meta.json", {kk: _jsonable(s[kk]) for kk in s if kk not in ("qpos", "qvel", "qacc_warmstart", "act", "ctrl", "mocap_pos", "mocap_quat")})
        print("snap", k, "t", s["time"], flush=True)

    audits = []
    for name in ("EARLY", "MID"):
        if name not in snaps:
            continue
        sim, _ = make_sim()
        au = restore_audit(sim, gains, snaps[name], name, ep["rows"], float(snaps[name]["time"]))
        audits.append(au)
        print("restore", name, "max_err_mm", au.get("max_err_mm"), "table", au.get("table"), flush=True)
    dump_json(RAW / "restore_audit.json", audits)

    specs = []
    for stage in ("EARLY", "MID"):
        if stage not in snaps:
            continue
        specs += [
            {"name": f"{stage}_secure", "stage": stage, "omega_y": 0.0, "v_x": 0.0, "v_z": 0.0, "tau": -18.0, "dur": 0.40},
            {"name": f"{stage}_wy+", "stage": stage, "omega_y": 1.2, "v_x": 0.0, "v_z": 0.0, "tau": -18.0, "dur": 0.25},
            {"name": f"{stage}_wy-", "stage": stage, "omega_y": -1.2, "v_x": 0.0, "v_z": 0.0, "tau": -18.0, "dur": 0.25},
            {"name": f"{stage}_tau-8", "stage": stage, "omega_y": 0.0, "v_x": 0.0, "v_z": 0.0, "tau": -8.0, "dur": 0.40},
            {"name": f"{stage}_tau-3", "stage": stage, "omega_y": 0.0, "v_x": 0.0, "v_z": 0.0, "tau": -3.0, "dur": 0.40},
            {"name": f"{stage}_tau-1", "stage": stage, "omega_y": 0.0, "v_x": 0.0, "v_z": 0.0, "tau": -1.0, "dur": 0.40},
            {"name": f"{stage}_vx+", "stage": stage, "omega_y": 0.0, "v_x": 0.06, "v_z": 0.0, "tau": -18.0, "dur": 0.30},
            {"name": f"{stage}_vx-", "stage": stage, "omega_y": 0.0, "v_x": -0.06, "v_z": 0.0, "tau": -18.0, "dur": 0.30},
            {"name": f"{stage}_vz+", "stage": stage, "omega_y": 0.0, "v_x": 0.0, "v_z": 0.04, "tau": -18.0, "dur": 0.30},
            {"name": f"{stage}_vz-", "stage": stage, "omega_y": 0.0, "v_x": 0.0, "v_z": -0.04, "tau": -18.0, "dur": 0.30},
        ]
    probes = []
    for sp in specs:
        sim, _ = make_sim()
        pr = probe(sim, gains, snaps[sp["stage"]], sp)
        probes.append(pr)
        print(
            f"probe {pr['name']} drhx={pr['drhx_pulse_mm']:.2f}mm keep={pr['contacts_kept']} "
            f"wrist={pr['d_wrist_deg']:.1f} table={pr['table']}",
            flush=True,
        )
    dump_json(RAW / "probes.json", probes)

    # one motivated combo if vx helped
    combos = []
    for stage in ("EARLY", "MID"):
        vx_p = next((p for p in probes if p["name"] == f"{stage}_vx+"), None)
        vx_m = next((p for p in probes if p["name"] == f"{stage}_vx-"), None)
        tau3 = next((p for p in probes if p["name"] == f"{stage}_tau-3"), None)
        if vx_p and vx_m:
            better = vx_p if abs(vx_p["end_pulse"]["rh"][0]) < abs(vx_m["end_pulse"]["rh"][0]) else vx_m
            if better["contacts_kept"] and better["table"] is None:
                sp = {
                    "name": f"{stage}_followx_secure",
                    "stage": stage,
                    "omega_y": 0.0,
                    "v_x": better["cmd"]["v_x"],
                    "v_z": 0.0,
                    "tau": -18.0,
                    "dur": 0.30,
                }
                sim, _ = make_sim()
                combos.append(probe(sim, gains, snaps[stage], sp))
        if tau3 and tau3["contacts_kept"] and tau3["table"] is None and vx_p:
            sign_v = vx_p["cmd"]["v_x"] if abs(vx_p["end_pulse"]["rh"][0]) <= abs(vx_m["end_pulse"]["rh"][0]) else vx_m["cmd"]["v_x"]
            sp = {
                "name": f"{stage}_followx_tau-3",
                "stage": stage,
                "omega_y": 0.0,
                "v_x": sign_v,
                "v_z": 0.0,
                "tau": -3.0,
                "dur": 0.30,
            }
            sim, _ = make_sim()
            combos.append(probe(sim, gains, snaps[stage], sp))
    dump_json(RAW / "combos.json", combos)
    for c in combos:
        print(f"combo {c['name']} drhx={c['drhx_pulse_mm']:.2f} keep={c['contacts_kept']} table={c['table']}", flush=True)

    plan = pick_existence(probes + combos)
    task = None
    if plan is not None:
        t_go = stages["T_EARLY_DRIFT"] if plan["stage"] == "EARLY" else stages["T_MID_DRIFT"]
        sim, _ = make_sim()
        ep_pre, snaps_go = run_center6_until(sim, gains, t_go, label="RECOVERY")
        z_tgt = z_tgt_of(sim)
        log, t_table = apply_plan(sim, gains, plan, z_tgt)
        held = t_table is None and float(sim.data.time) >= (ep_pre.get("t_lift_done") or 0) + HOLD_S - 2.0
        # more honest: check obj_z and contacts at end
        mend = log[-1]
        task = {
            "plan": plan,
            "intervene_t": t_go,
            "t_end": mend["t"],
            "end_rh": mend["rh"],
            "end_nL": mend["nL"],
            "end_nR": mend["nR"],
            "end_obj_z": mend["obj_z"],
            "table": t_table,
            "held_like": bool(t_table is None and mend["nL"] > 0 and mend["nR"] > 0 and mend["obj_z"] > 0.55),
        }
        dump_json(RAW / "task_level.json", task)
        dump_json(RAW / "recovery_plan.json", plan)
        print("task-level", task["held_like"], "table", t_table, "end_rhx_mm", 1e3 * float(mend["rh"][0]), flush=True)
    else:
        dump_json(RAW / "recovery_plan.json", {"none": True})
        print("no existence plan from probes", flush=True)

    write_report(stages, audits, probes, combos, plan, task, ep)
    return plan, stages


def write_report(stages, audits, probes, combos, plan, task, ep):
    p = OUT / "BALLISTIC_RECOVERY_TRANSFER.md"
    a = []
    ap = a.append
    ap("# Ballistic recovery transfer (CENTER-6)")
    ap("")
    ap("**BALLISTIC CONTACT-PRESERVING RECOVERY AUTHORITY PROOF** (if a sequence works).")
    ap("Not a learned policy, not an optimized heuristic.")
    ap("")
    ap("## USER VISUAL OBSERVATION: CENTER-6 confirmed as translational progressive failure.")
    ap("")
    ap("## 1. Frozen CENTER-6 benchmark")
    ap("")
    ap("Matched +hand-x central impact, `v=6.0 m/s`, `m_ball=0.05 kg`, noslip=1.")
    ap("Impact/mass/timing/geometry **frozen**. No UPPER/LOWER.")
    ap("Label: **CENTER-6: ballistic translational progressive failure.**")
    ap("Do not describe as translation+tilt unless orientation data shows otherwise.")
    ap("")
    ap("## 2. User visual validation")
    ap("")
    ap("- ball hits cylinder normally")
    ap("- cylinder remains between fingers after impact")
    ap("- slow lateral drift through the grasp")
    ap("- eventual escape and fall")
    ap("- **no meaningful impact-induced rotation** (visual)")
    ap("")
    ap("## 3. ZERO stage timeline")
    ap("")
    ap("```json")
    ap(json.dumps(_jsonable(stages), indent=2))
    ap("```")
    ap("")
    ap(f"ZERO table time {ep.get('t_table_contact')}. Lift done {ep.get('t_lift_done')}.")
    ap("")
    ap("## 4. EARLY / MID / LATE snapshots")
    ap("")
    ap(f"`{RAW.as_posix()}/snap_early.pkl` at T_EARLY_DRIFT={stages['T_EARLY_DRIFT']}")
    ap(f"`{RAW.as_posix()}/snap_mid.pkl` at T_MID_DRIFT={stages['T_MID_DRIFT']}")
    if stages.get("T_LATE"):
        ap(f"`{RAW.as_posix()}/snap_late.pkl` at T_LATE={stages['T_LATE']}")
    ap("Snapshots include qpos/qvel/ctrl/time, p_des, r_des, v_cmd, w_cmd, FSM clocks,")
    ap("eq_active, mocap, grasp_z, qacc_warmstart.")
    ap("")
    ap("## 5. Restore audit")
    ap("")
    ap("```json")
    ap(json.dumps(_jsonable(audits), indent=2))
    ap("```")
    ap("")
    ap("## 6–10. Single-action and combination probes")
    ap("")
    ap("| name | drhx pulse mm | contacts kept | wrist deg | table |")
    ap("|---|---:|---:|---:|---|")
    for pr in probes + combos:
        ap(
            f"| {pr['name']} | {pr['drhx_pulse_mm']:.2f} | {pr['contacts_kept']} | "
            f"{pr['d_wrist_deg']:.1f} | {pr['table']} |"
        )
    ap("")
    ap("Raw: `raw/probes.json`, `raw/combos.json`.")
    ap("")
    ap("## 11. Comparison with teleport mechanism")
    ap("")
    if plan is None:
        ap("No contact-preserving sequence was selected from these probes.")
        ap("Outcome tentatively **D or incomplete** until the table is read: either")
        ap("authority is weak, or useful signs exist but did not reduce |rh.x| cleanly.")
    else:
        ap("Selected existence plan uses the probe with largest |rh.x| reduction while")
        ap("keeping bilateral contact. Compare to teleport A (wrist→weak grip→secure):")
        ap("CENTER-6 is translation-dominated, so a **B (hand-x follow)** outcome is")
        ap("the expected alternative if vx probes dominate wrist probes.")
        ap("")
        ap("```json")
        ap(json.dumps(_jsonable(plan), indent=2))
        ap("```")
    ap("")
    ap("## 12. Task-level existence proof")
    ap("")
    if task is None:
        ap("Not run (no selected plan).")
    else:
        ap("Intervention after visible drift (EARLY or MID), not at ball contact.")
        ap("")
        ap("```json")
        ap(json.dumps(_jsonable(task), indent=2))
        ap("```")
        if task.get("held_like"):
            ap("Numerical hold-like end state. **Not optimized. Viewer confirmation required.**")
        else:
            ap("Sequence did not produce a 12 s elevated hold in this one-shot construction.")
    ap("")
    ap("## 13. Interactive viewer commands")
    ap("")
    ap("```text")
    ap("python training/demo_ballistic_recovery.py --mode zero")
    ap("python training/demo_ballistic_recovery.py --mode recovery")
    ap("```")
    ap("")
    ap("SPACE pause, `[` `]` speed, R restart. Camera once, then mouse.")
    ap("")
    ap("## 14. Unresolved limitations")
    ap("")
    ap("- Probes are short pulses, not a policy.")
    ap("- Grip tau meaning is state-dependent.")
    ap("- Wrist pulses do not search an angle.")
    ap("- Restore uses `load_snapshot` plus weld/mocap; remaining solver-state mismatch may appear in mm-level rh error.")
    ap("- No UPPER/LOWER, no SAC, no recatch.")
    ap("")
    ap("## USER VISUAL OBSERVATION: PENDING for recovery sequence")
    p.write_text("\n".join(a), encoding="utf-8")
    print("wrote", p, flush=True)


def interactive(mode: str) -> None:
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    plan_p = RAW / "recovery_plan.json"
    stages_p = RAW / "stages.json"
    plan = json.loads(plan_p.read_text(encoding="utf-8")) if plan_p.is_file() else {"none": True}
    stages = json.loads(stages_p.read_text(encoding="utf-8")) if stages_p.is_file() else {"T_EARLY_DRIFT": 3.25}
    ctl = {"pause": False, "reset": False, "abort": False, "speed": 0.40, "overlay": {}}

    def sync():
        vwr = ctl.get("viewer")
        if vwr is None:
            return
        o = ctl.get("overlay") or {}
        _viewer_overlay(
            vwr,
            [
                ("CASE", str(o.get("case", mode.upper()))),
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
            if mode == "zero" or plan.get("none"):
                run_zero_full(
                    sim,
                    gains,
                    ctl=ctl,
                    label="ZERO CENTER-6" if mode == "zero" else "NO PLAN — ZERO",
                )
            else:
                t_go = float(stages["T_EARLY_DRIFT"] if plan.get("stage") == "EARLY" else stages["T_MID_DRIFT"])
                def wrap_pause_sync():
                    while ctl.get("pause") and vwr.is_running() and not ctl.get("reset"):
                        sync()
                        time.sleep(0.02)
                    sync()
                    if "GO" in ctl.get("snaps", {}):
                        ctl["abort"] = True

                ctl["sync"] = wrap_pause_sync
                ctl["capture_at"] = {"GO": t_go}
                ctl["snaps"] = {}
                ctl["make_snap"] = make_snap
                run_episode(sim, gains, V_HIT, SITE, ctl=ctl, label="DRIFT")
                if ctl.get("reset") or not vwr.is_running():
                    if ctl.get("reset"):
                        continue
                    break
                z_tgt = z_tgt_of(sim)

                def pause_idle():
                    while ctl.get("pause") and vwr.is_running() and not ctl.get("reset"):
                        sync()
                        time.sleep(0.02)
                    sync()

                ctl["sync"] = pause_idle
                apply_plan(sim, gains, plan, z_tgt, ctl=ctl)
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
        default="sweep",
        choices=(
            "sweep",
            "zero",
            "recovery",
            "large_angle",
            "gravity_large_angle",
            "gravity_more_slip",
            "return_audit",
            "slip_sufficiency",
        ),
    )
    args = p.parse_args()
    if args.mode in ("zero", "recovery"):
        interactive(args.mode)
        return
    if args.mode == "gravity_large_angle":
        from training.ballistic_large_angle import interactive_large_angle

        interactive_large_angle()
        return
    if args.mode == "gravity_more_slip":
        from training.ballistic_large_angle import RAW, interactive_large_angle

        pth = RAW / "more_slip_plan.json"
        if not pth.is_file():
            raise SystemExit("no more_slip_plan.json (longer-slip case not selected)")
        interactive_large_angle(pth)
        return
    if args.mode == "return_audit":
        from training.ballistic_large_angle import headless_return_audit

        headless_return_audit()
        return
    if args.mode == "slip_sufficiency":
        from training.ballistic_slip_sufficiency import headless_slip_sufficiency

        headless_slip_sufficiency()
        return
    if args.mode == "large_angle":
        from training.ballistic_large_angle import headless_large_angle

        headless_large_angle()
        return
    headless()


if __name__ == "__main__":
    main()
