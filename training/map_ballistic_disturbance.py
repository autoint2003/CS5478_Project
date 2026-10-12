"""Map ballistic disturbance regimes (matched geometry). ZERO only. No MP4.

Physical-loss events are explicit; a 2 ms both-off is not a drop.
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
from envs.config_util import load_yaml, merge_sim_config
from envs.deterioration import body_twist
from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from training.demo_ballistic_impact import (
    AFTER_DROP_S,
    HAND_RISE_M,
    HOLD_S,
    LOG_DT,
    TAU_SEC,
    assert_noslip,
    ball_force_on_ball,
    dump_json,
    live_opt,
    make_sim,
    pack_row,
    z_tgt_of,
)
from training.grav_reposition_v2_viz import apply_camera_preset
from training.impact_ball import apply_ball_free_state, incoming_hand_x
from training.replay_core import tick_vw
from training.write_ballistic_impact_scene import BALL_M, CYL_H, CYL_R

OUT = ROOT / "results" / "diagnostics" / "ballistic_impact_noslip1"
MAP = OUT / "disturbance_map"
T_FLIGHT = 0.10
# Cylinder half-height 30 mm. Offset 12 mm keeps first contact on the wall, not the rim.
Z_OFF = {"CENTER": 0.0, "UPPER": 0.012, "LOWER": -0.012}
COARSE_V = (3.0, 4.0, 5.0, 6.0, 7.0)
CAM = {
    "lookat": np.array([0.50, -0.02, 0.51]),
    "distance": 0.88,
    "azimuth": 118.0,
    "elevation": -16.0,
}


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


def launch_matched(sim, v_hit: float, z_off: float) -> dict:
    """Aim at predicted cylinder, +hand-x, with axial offset from COM (object z)."""
    u = incoming_hand_x(sim)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    axis = Ro[:, 2]
    r_ball = float(sim.model.geom_size[int(sim.ball_geom)][0])
    r_obj = float(sim.model.geom_size[int(sim.ids.object_geom)][0])
    r_surf = r_obj + r_ball
    T = float(T_FLIGHT)
    g = np.array(sim.model.opt.gravity, float)
    po_pred = po + vo * T
    p_hit = po_pred - u * r_surf + axis * float(z_off)
    along = float(v_hit) * T
    p_launch = p_hit - u * along
    v0 = (p_hit - p_launch) / T - 0.5 * g * T
    return {
        "mode": "matched_predicted_cylinder_axial",
        "t_flight": T,
        "z_off_m": float(z_off),
        "z_off_mm": float(z_off) * 1e3,
        "p_obj_release": po.copy(),
        "v_obj_release": vo.copy(),
        "w_obj_release": wo.copy(),
        "axis_w": axis.copy(),
        "p_obj_pred": po_pred.copy(),
        "p_hit": p_hit,
        "p_launch": p_launch,
        "v0": v0,
        "direction": u,
        "r_surf": r_surf,
        "v_impact_requested": float(v_hit),
        "cyl_half_h_m": 0.5 * CYL_H,
        "cyl_r_m": CYL_R,
    }


def in_finger_span(sim) -> tuple[bool, np.ndarray]:
    ids = sim.ids
    Rh = np.array(sim.data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[ids.hand_body], float)
    po = np.array(sim.data.xpos[ids.object_body], float)
    r = Rh.T @ (po - ph)
    pL = Rh.T @ (np.array(sim.data.xpos[ids.left_body], float) - ph)
    pR = Rh.T @ (np.array(sim.data.xpos[ids.right_body], float) - ph)
    y0, y1 = sorted((float(pL[1]), float(pR[1])))
    r_cyl = float(sim.model.geom_size[int(ids.object_geom)][0])
    inside_y = (y0 - r_cyl - 0.006) <= float(r[1]) <= (y1 + r_cyl + 0.006)
    inside_x = abs(float(r[0])) < 0.040
    inside_z = abs(float(r[2]) - 0.099) < 0.055
    return bool(inside_y and inside_x and inside_z), r


def tilt_deg(R_rel) -> float:
    axis_h = np.asarray(R_rel, float).reshape(3, 3)[:, 2]
    c = float(np.clip(axis_h[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def sample_at(rows, t0, tau):
    cand = [r for r in rows if r["t"] >= t0 + tau - 1e-12]
    return cand[0] if cand else None


def pub_row(r: dict) -> dict:
    return {
        "t": r["t"],
        "rh": np.asarray(r["rh"], float).tolist(),
        "tilt_deg": tilt_deg(r["R_rel"]),
        "v_rel": float(np.linalg.norm(r["v_rel_h"])),
        "w_rel": float(np.linalg.norm(r["w_rel_h"])),
        "nL": r["nL"],
        "nR": r["nR"],
        "Fn_L": r["Fn_L"],
        "Fn_R": r["Fn_R"],
        "rho_max": r.get("rho_max"),
        "aperture": r.get("aperture"),
        "obj_z": r["obj_z"],
        "clear": r["clear"],
        "p_obj": np.asarray(r["p_obj"], float).tolist(),
        "p_hand": np.asarray(r["p_hand"], float).tolist(),
        "fsm": r["fsm"],
    }


def run_episode(sim, gains, v_hit: float, site: str, *, ctl=None, label="MAP") -> dict:
    """Nominal task + matched free-flight. Does not stop on 2 ms both-off."""
    assert_noslip(sim)
    sim.reset()
    dt = float(sim.model.opt.timestep)
    log_n = max(int(round(LOG_DT / dt)), 1)
    z_off = float(Z_OFF[site])
    rows = []
    events = []
    hand_z0 = None
    t_lift0 = None
    released = False
    t_release = None
    t_impact = None
    launch = None
    first_c = None
    Rh_imp = None
    J_ball = np.zeros(3)
    t_last_c = None
    vo_pre = wo_pre = None
    po_pre = None
    t_left_off = t_right_off = None
    t_both = None
    t_restore = None
    both_acc = 0.0
    both_max = 0.0
    t_both20 = t_both50 = t_both100 = None
    t_exit = None
    exit_acc = 0.0
    t_table = None
    t_phys = None
    t_lift_done = None
    z_tgt = None
    holding = False
    n_max = int(round(30.0 / dt))
    ctl = ctl or {}

    for i in range(n_max):
        if ctl.get("abort"):
            break
        while ctl.get("pause") and not ctl.get("abort") and not ctl.get("reset"):
            if ctl.get("sync"):
                ctl["sync"]()
            time.sleep(0.02)
        if ctl.get("reset"):
            break

        if (not holding) and sim.fsm.phase == "lift" and z_tgt is not None:
            if float(sim.fsm.p_des[2]) >= z_tgt - 1e-9:
                holding = True
                t_lift_done = float(sim.data.time)
        if holding:
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        else:
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()

        t = float(sim.data.time)
        if sim.fsm.phase == "lift" and hand_z0 is None:
            hand_z0 = float(sim.data.xpos[sim.ids.hand_body][2])
            t_lift0 = t
            z_tgt = z_tgt_of(sim)

        if released:
            Fb, bc = ball_force_on_ball(sim)
            has_cyl = any(r["kind"] == "ball-cylinder" for r in bc)
            if t_impact is None and has_cyl:
                t_impact = t
                first_c = next(r for r in bc if r["kind"] == "ball-cylinder")
                Rh_imp = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
                events.append({"event": "impact", "t": t})
            if has_cyl:
                J_ball += Fb * dt
                t_last_c = t

        o = physical_pack(sim)
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if (
            (not released)
            and sim.fsm.phase == "lift"
            and sim.captured
            and hand_z0 is not None
            and float(o["obj_z"]) >= Z_AIR
            and (hz - hand_z0) >= HAND_RISE_M
        ):
            launch = launch_matched(sim, v_hit, z_off)
            sim.set_guide_weld(False)
            apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
            released = True
            t_release = float(sim.data.time)
            events.append({"event": "release", "t": t_release, "eq_active": sim.eq_active()})

        if released and t_impact is None:
            vo_pre, wo_pre = body_twist(sim.model, sim.data, sim.ids.object_body)
            po_pre = np.array(sim.data.xpos[sim.ids.object_body], float).copy()

        nL, nR = int(o["nL"]), int(o["nR"])
        if t_impact is not None:
            if nL == 0 and t_left_off is None:
                t_left_off = t
            if nR == 0 and t_right_off is None:
                t_right_off = t
            if nL == 0 and nR == 0:
                if t_both is None:
                    t_both = t
                both_acc += dt
                both_max = max(both_max, both_acc)
                if t_both20 is None and both_acc >= 0.020 - 1e-9:
                    t_both20 = t
                if t_both50 is None and both_acc >= 0.050 - 1e-9:
                    t_both50 = t
                if t_both100 is None and both_acc >= 0.100 - 1e-9:
                    t_both100 = t
            else:
                if t_both is not None and t_restore is None and nL > 0 and nR > 0:
                    t_restore = t
                both_acc = 0.0
            inside, _rh = in_finger_span(sim)
            if not inside:
                exit_acc += dt
                if t_exit is None and exit_acc >= 0.050 - 1e-9:
                    t_exit = t
                    events.append({"event": "object_exits_finger_span", "t": t})
            else:
                exit_acc = 0.0
            if t_table is None and float(o["obj_z"]) < TABLE_DROP:
                t_table = t
                events.append({"event": "table_contact", "t": t})
            ph = np.array(sim.data.xpos[sim.ids.hand_body][:2], float)
            po = np.array(sim.data.xpos[sim.ids.object_body][:2], float)
            xy = float(np.linalg.norm(po - ph))
            if t_phys is None and (t_table is not None or (t_exit is not None and xy > 0.12)):
                t_phys = t_table if t_table is not None else t_exit
                events.append({"event": "physical_drop", "t": t_phys})

        fine = t_impact is not None and abs(t - t_impact) <= 0.25
        if (i % log_n == 0) or fine or (released and t_impact is None):
            r = pack_row(sim, TAU_SEC)
            r["aperture"] = finger_opening(sim.data, sim.ids)
            rows.append(r)

        cap = ctl.get("capture_at") or {}
        bag = ctl.setdefault("snaps", {})
        for name, tt in cap.items():
            if name not in bag and t + 0.51 * dt >= float(tt):
                maker = ctl.get("make_snap")
                bag[name] = maker(sim) if callable(maker) else sim.snapshot()

        if ctl.get("sync"):
            phase = str(sim.fsm.phase).upper()
            if released and t_impact is None:
                phase = "FREE FLIGHT"
            elif t_table is not None:
                phase = "TABLE"
            elif holding:
                phase = "HIGH HOLD"
            elif t_impact is not None:
                phase = "LIFT"
            ctl["overlay"] = {
                "case": label,
                "t": t,
                "phase": phase,
                "nL": nL,
                "nR": nR,
            }
            ctl["sync"]()

        table_end = t_table is not None and t >= t_table + AFTER_DROP_S
        held = t_lift_done is not None and t_phys is None and t_table is None and t >= t_lift_done + HOLD_S
        if table_end or held or t > 28.0:
            break

    dt = float(sim.model.opt.timestep)
    impact_geom = None
    if first_c is not None and t_impact is not None:
        # reconstruct from nearest row
        near = min(rows, key=lambda r: abs(r["t"] - t_impact))
        pos = np.asarray(first_c["pos"], float)
        po = np.asarray(near["p_obj"], float)
        Ro = np.asarray(near["R_obj"], float)
        Rh = np.asarray(near["R_hand"], float)
        ph = np.asarray(near["p_hand"], float)
        r_cyl = Ro.T @ (pos - po)
        impact_geom = {
            "pos_w": pos.tolist(),
            "pos_cyl_m": r_cyl.tolist(),
            "pos_cyl_mm": (r_cyl * 1e3).tolist(),
            "axial_z_mm": float(r_cyl[2]) * 1e3,
            "radial_mm": float(np.linalg.norm(r_cyl[:2])) * 1e3,
            "near_end": bool(abs(r_cyl[2]) > 0.022),
            "p_obj_imp": po.tolist(),
            "p_hand_imp": ph.tolist(),
        }

    J_obj = -J_ball
    r_c = None
    L_proxy = None
    if first_c is not None and po_pre is not None:
        r_c = np.asarray(first_c["pos"], float) - po_pre
        L_proxy = np.cross(r_c, J_obj)

    vo_post = wo_post = None
    s20 = sample_at(rows, t_impact, 0.020) if t_impact is not None else None
    if s20 is not None:
        vo_post = np.asarray(s20["v_obj"], float)
        wo_post = np.asarray(s20["w_obj"], float)

    times = (0.0, 0.020, 0.050, 0.100, 0.200, 0.500, 1.000)
    samples = {}
    if t_impact is not None:
        for tau in times:
            r = sample_at(rows, t_impact, tau)
            if r is not None:
                samples[f"plus_{int(tau*1000)}ms"] = pub_row(r)

    if t_phys is None and t_table is None and t_lift_done is not None and (float(sim.data.time) - t_lift_done) >= HOLD_S - 1e-3:
        end_state = "held_12s"
    elif t_table is not None:
        end_state = "table_contact"
    elif t_exit is not None:
        end_state = "exited_finger_span"
    else:
        end_state = "truncated"

    mech = classify_mechanism(
        t_impact=t_impact,
        t_phys=t_phys,
        t_table=t_table,
        t_exit=t_exit,
        t_restore=t_restore,
        end_state=end_state,
        rows=rows,
        samples=samples,
    )

    snap = sim.snapshot() if t_impact is not None else None
    ep = {
        "site": site,
        "v_hit": float(v_hit),
        "ball_mass": BALL_M,
        "z_off_mm": z_off * 1e3,
        "t_lift0": t_lift0,
        "t_release": t_release,
        "t_impact": t_impact,
        "free_flight_s": None if t_release is None or t_impact is None else t_impact - t_release,
        "t_first_left_off": t_left_off,
        "t_first_right_off": t_right_off,
        "t_first_both_off": t_both,
        "both_off_duration_max_s": both_max,
        "t_bilateral_restored": t_restore,
        "t_sustained_both_off_20ms": t_both20,
        "t_sustained_both_off_50ms": t_both50,
        "t_sustained_both_off_100ms": t_both100,
        "t_object_exits_finger_span": t_exit,
        "t_physical_drop": t_phys,
        "t_table_contact": t_table,
        "t_lift_done": t_lift_done,
        "t_end": float(sim.data.time),
        "end_state": end_state,
        "impact_geom": impact_geom,
        "first_contact": None
        if first_c is None
        else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in first_c.items() if k != "f_ball_w"},
        "J_world_on_ball": J_ball.tolist(),
        "J_world_on_obj": J_obj.tolist(),
        "J_hand_on_obj": None if Rh_imp is None else (Rh_imp.T @ J_obj).tolist(),
        "J_mag": float(np.linalg.norm(J_obj)),
        "r_contact_from_com": None if r_c is None else r_c.tolist(),
        "angular_impulse_proxy": None if L_proxy is None else L_proxy.tolist(),
        "angular_impulse_proxy_mag": None if L_proxy is None else float(np.linalg.norm(L_proxy)),
        "v_obj_pre": None if vo_pre is None else np.asarray(vo_pre).tolist(),
        "w_obj_pre": None if wo_pre is None else np.asarray(wo_pre).tolist(),
        "v_obj_plus20": None if vo_post is None else vo_post.tolist(),
        "w_obj_plus20": None if wo_post is None else wo_post.tolist(),
        "delta_v_obj": None if vo_pre is None or vo_post is None else (vo_post - vo_pre).tolist(),
        "delta_omega_obj": None if wo_pre is None or wo_post is None else (wo_post - wo_pre).tolist(),
        "delta_v_mag": None if vo_pre is None or vo_post is None else float(np.linalg.norm(vo_post - vo_pre)),
        "delta_w_mag": None if wo_pre is None or wo_post is None else float(np.linalg.norm(wo_post - wo_pre)),
        "post_samples": samples,
        "mechanism": mech,
        "end_nL": rows[-1]["nL"] if rows else None,
        "end_nR": rows[-1]["nR"] if rows else None,
        "end_obj_z": rows[-1]["obj_z"] if rows else None,
        "launch": None if launch is None else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in launch.items()},
        "live": live_opt(sim),
        "rows": rows,
        "events": events,
        "snap": snap,
        "t_last_ball_contact": t_last_c,
    }
    return ep


def classify_mechanism(*, t_impact, t_phys, t_table, t_exit, t_restore, end_state, rows, samples) -> str:
    if t_impact is None:
        return "NO_IMPACT"
    t_loss = t_table if t_table is not None else t_phys
    early = t_loss is not None and (t_loss - t_impact) < 0.080
    s200 = samples.get("plus_200ms")
    captured_200 = s200 is not None and (s200["nL"] + s200["nR"]) > 0 and abs(s200["rh"][0]) < 0.04
    if early and not captured_200:
        return "DIRECT_EJECTION"
    post = [r for r in rows if r["t"] >= t_impact + 0.20]
    if not post:
        return "SHORT_RECORD"
    rhx = np.array([abs(r["rh"][0]) for r in post])
    tilt = np.array([tilt_deg(r["R_rel"]) for r in post])
    vrel = np.array([float(np.linalg.norm(r["v_rel_h"])) for r in post])
    late_loss = t_loss is not None and (t_loss - t_impact) >= 0.20
    grew = (rhx[-1] > rhx[min(8, len(rhx) - 1)] + 0.004) or (tilt[-1] > tilt[min(8, len(tilt) - 1)] + 4.0)
    if late_loss and grew:
        return "IMPACT_INITIATED_SLIP"
    if end_state == "held_12s":
        if vrel[-1] < 0.01 and not grew:
            if rhx[0] > 0.003:
                return "IMPACT_REPOSITIONED_STABLE"
            return "IMPACT_DAMPED"
        if grew and not late_loss:
            return "IMPACT_INITIATED_SLIP"
        return "IMPACT_REPOSITIONED_STABLE"
    if t_loss is not None and (t_loss - t_impact) < 0.20:
        return "DIRECT_EJECTION"
    return "UNCLEAR"


def slim(ep: dict) -> dict:
    skip = {"rows", "snap"}
    return {k: _jsonable(v) for k, v in ep.items() if k not in skip}


def save_traj(ep: dict, path: Path) -> None:
    rows = ep.get("rows") or []
    if not rows:
        return
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in rows]),
        rh=np.array([r["rh"] for r in rows]),
        nL=np.array([r["nL"] for r in rows]),
        nR=np.array([r["nR"] for r in rows]),
        obj_z=np.array([r["obj_z"] for r in rows]),
        vrel=np.array([np.linalg.norm(r["v_rel_h"]) for r in rows]),
        wrel=np.array([np.linalg.norm(r["w_rel_h"]) for r in rows]),
        tilt=np.array([tilt_deg(r["R_rel"]) for r in rows]),
        Fn_L=np.array([r["Fn_L"] for r in rows]),
        Fn_R=np.array([r["Fn_R"] for r in rows]),
        aperture=np.array([r.get("aperture", np.nan) for r in rows]),
    )


def tag(site: str, v: float) -> str:
    return f"{site.lower()}_v{v:.1f}".replace(".", "p")


def run_headless() -> dict:
    MAP.mkdir(parents=True, exist_ok=True)
    (MAP / "figures").mkdir(exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    results = []
    found_c = None

    def one(site, v):
        nonlocal found_c
        sim, _ = make_sim()
        ep = run_episode(sim, gains, v, site, label=tag(site, v))
        dump_json(MAP / f"{tag(site, v)}.json", slim(ep))
        save_traj(ep, MAP / f"{tag(site, v)}.npz")
        if ep["mechanism"] == "IMPACT_INITIATED_SLIP" and found_c is None:
            found_c = ep
            with (MAP / "progressive_candidate.pkl").open("wb") as f:
                pickle.dump(ep.get("snap"), f)
        print(
            f"{site} v={v:.1f} tof={ep['free_flight_s']} J={ep['J_mag']:.4f} "
            f"L={ep['angular_impulse_proxy_mag']} both={ep['t_first_both_off']} "
            f"restore={ep['t_bilateral_restored']} both_max={ep['both_off_duration_max_s']} "
            f"table={ep['t_table_contact']} phys={ep['t_physical_drop']} "
            f"end={ep['end_state']} mech={ep['mechanism']}",
            flush=True,
        )
        results.append(ep)
        return ep

    print("PHASE A CENTER coarse speed", flush=True)
    phase_a = []
    for v in COARSE_V:
        ep = one("CENTER", v)
        phase_a.append(ep)
        if ep["mechanism"] == "IMPACT_INITIATED_SLIP":
            break
        if ep["mechanism"] == "DIRECT_EJECTION":
            break

    if found_c is None:
        print("PHASE B off-center at informative speeds", flush=True)
        # last non-ejection and first ejection if any
        last_ok = None
        first_ej = None
        for e in phase_a:
            if e["mechanism"] == "DIRECT_EJECTION":
                first_ej = e["v_hit"]
                break
            last_ok = e["v_hit"]
        pick = []
        if last_ok is not None:
            pick.append(last_ok)
        if first_ej is not None and first_ej not in pick:
            pick.append(first_ej)
        if not pick:
            pick = [e["v_hit"] for e in phase_a][-2:] or [e["v_hit"] for e in phase_a]
        for v in pick:
            for site in ("UPPER", "LOWER"):
                if found_c is not None:
                    break
                one(site, v)
            if found_c is not None:
                break

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
        for e in results:
            rows = e["rows"]
            ti = e["t_impact"]
            if not rows or ti is None:
                continue
            t = np.array([r["t"] for r in rows]) - ti
            m = t >= -0.05
            ax[0].plot(t[m], 1e3 * np.array([r["rh"][0] for r in rows])[m], lw=0.9, label=tag(e["site"], e["v_hit"]))
            ax[1].plot(t[m], np.array([r["nL"] + r["nR"] for r in rows])[m], lw=0.9)
        ax[0].set_ylabel("rh.x mm")
        ax[1].set_ylabel("nL+nR")
        ax[1].set_xlabel("t - t_impact (s)")
        ax[0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(MAP / "figures" / "post_impact_rh.png", dpi=110)
        plt.close(fig)
    except Exception:
        pass

    summary = {
        "runs": [slim(e) for e in results],
        "progressive_found": found_c is not None,
        "phase_a_mechs": [(e["v_hit"], e["mechanism"], e["end_state"]) for e in results if e["site"] == "CENTER"],
    }
    dump_json(MAP / "summary.json", summary)
    return {"results": results, "found_c": found_c, "summary": summary}


def write_map_md(payload: dict) -> None:
    results = payload["results"]
    found_c = payload["found_c"]
    lines = []
    ap = lines.append
    ap("# Ballistic disturbance map (noslip=1, m_ball=0.05 kg)")
    ap("")
    ap("**Diagnostic only. No recovery / recatch / SAC. No MP4.**")
    ap("")
    ap("## 11. USER VISUAL OBSERVATION: PENDING")
    ap("")
    ap("Interactive viewer cases are listed in section 9. Watch before treating mechanism labels as confirmed.")
    ap("")
    ap("## 1. Corrected loss semantics")
    ap("")
    ap("Legacy `dropped()` treats a single 2 ms `nL=nR=0` sample as drop. That is **not** used here.")
    ap("Diagnostic events are kept separate:")
    ap("")
    ap("- `first_left_off` / `first_right_off` / `first_both_off`")
    ap("- `both_off_duration` (max consecutive)")
    ap("- `bilateral_restored`")
    ap("- `sustained_both_off_20/50/100ms`")
    ap("- `object_exits_finger_span` (COM outside finger pads for ≥50 ms)")
    ap("- `table_contact` (`obj_z < TABLE_DROP`)")
    ap("- `physical_drop` (table, or span-exit with xy > 0.12 m)")
    ap("- `end_state`: `held_12s` | `table_contact` | `exited_finger_span` | `truncated`")
    ap("")
    ap("A 2 ms both-off is logged, never labelled DROP / IMMEDIATE LOSS.")
    ap("")
    ap("## 2. Matched impact construction")
    ap("")
    ap("Nominal FSM APPROACH→DESCEND→CLOSE→active LIFT. Release after captured + `Z_AIR` + 5 cm hand rise.")
    ap("Weld off; free flight `T=0.10 s`; aim at `p_obj + v_obj T` along `+hand-x`;")
    ap("optional axial offset along cylinder axis (object z). No teleport, no snapshot at impact.")
    ap(f"Cylinder r={CYL_R*1e3:.1f} mm, half-height {0.5*CYL_H*1e3:.0f} mm. Offsets: CENTER 0, UPPER +12 mm, LOWER −12 mm.")
    ap("Ball mass 0.05 kg. noslip_iterations=1.")
    ap("")
    ap("## 3. Central-impact coarse severity")
    ap("")
    ap("| v | TOF | |J| | |r×J| | Δv | Δω | both-off max | restore | table | end | mechanism |")
    ap("|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for e in results:
        if e["site"] != "CENTER":
            continue
        ap(
            f"| {e['v_hit']:.1f} | {e['free_flight_s']} | {e['J_mag']:.4f} | "
            f"{e['angular_impulse_proxy_mag']} | {e['delta_v_mag']} | {e['delta_w_mag']} | "
            f"{e['both_off_duration_max_s']:.3f} | {e['t_bilateral_restored']} | {e['t_table_contact']} | "
            f"{e['end_state']} | {e['mechanism']} |"
        )
    ap("")
    ap("## 4. Off-center impact construction")
    ap("")
    ap("UPPER/LOWER add `±12 mm` along the cylinder axis at the predicted COM, keeping the hit on the")
    ap("cylindrical wall (18 mm remaining to the rim). Intended to change `r_contact × J`, not TOF.")
    ap("")
    ap("| site | v | axial z mm (actual) | radial mm | near_end | |J| | |r×J| | end | mechanism |")
    ap("|---|---:|---:|---:|---|---:|---:|---|---|")
    for e in results:
        g = e.get("impact_geom") or {}
        ap(
            f"| {e['site']} | {e['v_hit']:.1f} | {g.get('axial_z_mm')} | {g.get('radial_mm')} | "
            f"{g.get('near_end')} | {e['J_mag']:.4f} | {e['angular_impulse_proxy_mag']} | "
            f"{e['end_state']} | {e['mechanism']} |"
        )
    ap("")
    ap("## 5. Impulse / angular impulse")
    ap("")
    ap("`J` is integrated ball–cylinder contact force on the ball (`mj_contactForce`), object impulse `−J`.")
    ap("Angular proxy is `r_contact×J_obj` with `r` from pre-impact COM to first contact point.")
    ap("`Δv`, `Δω` are object twist change from last pre-impact sample to +20 ms.")
    ap("")
    for e in results:
        ap(
            f"- {e['site']} {e['v_hit']:.1f} m/s: J={_jsonable(e['J_world_on_obj'])} |J|={e['J_mag']:.4f} "
            f"L={_jsonable(e['angular_impulse_proxy'])} Δv={_jsonable(e['delta_v_obj'])} Δω={_jsonable(e['delta_omega_obj'])}"
        )
    ap("")
    ap("## 6. Post-impact state trajectories")
    ap("")
    ap("Samples at 0/20/50/100/200/500/1000 ms in each `disturbance_map/*.json` (`post_samples`).")
    ap("Raw: `disturbance_map/*.npz`. Optional plot: `disturbance_map/figures/post_impact_rh.png`.")
    ap("")
    ap("## 7. Mechanism classification")
    ap("")
    ap("Analysis labels only (not controller logic): IMPACT_DAMPED, IMPACT_REPOSITIONED_STABLE,")
    ap("IMPACT_INITIATED_SLIP (candidate C), DIRECT_EJECTION.")
    ap("")
    ap("| run | mechanism | end | t_phys | t_table | notes |")
    ap("|---|---|---|---|---|---|")
    for e in results:
        ap(
            f"| {tag(e['site'], e['v_hit'])} | {e['mechanism']} | {e['end_state']} | "
            f"{e['t_physical_drop']} | {e['t_table_contact']} | both_max={e['both_off_duration_max_s']:.3f}s |"
        )
    ap("")
    ap("## 8. Progressive-instability candidate")
    ap("")
    if found_c is None:
        ap("**None found** in this structured speed + impact-point search.")
        ap("Do not start a random parameter search. Mass/timescale is a later experiment.")
    else:
        e = found_c
        ap("**Candidate saved. STOP. Inspect in the viewer before any recovery.**")
        ap("")
        ap(f"- mass {e['ball_mass']} kg, v={e['v_hit']} m/s, site={e['site']}, z_off={e['z_off_mm']} mm")
        ap(f"- contact {e.get('impact_geom')}")
        ap(f"- |J|={e['J_mag']}, L={e['angular_impulse_proxy']}")
        ap(f"- t_impact={e['t_impact']}, t_physical_drop={e['t_physical_drop']}, t_table={e['t_table_contact']}")
        ap(f"- post_samples: see `{tag(e['site'], e['v_hit'])}.json`")
        ap(f"- snapshot: `disturbance_map/progressive_candidate.pkl`")
    ap("")
    ap("## 9. Interactive viewer commands")
    ap("")
    ap("No MP4. Camera is set once; mouse rotate/pan/zoom. SPACE pause, `[`/`]` speed, R restart case.")
    ap("")
    ap("```text")
    for e in results:
        ap(f"python training/demo_ballistic_impact.py --case {tag(e['site'], e['v_hit'])}")
    # aliases
    damped = [e for e in results if e["mechanism"] in ("IMPACT_DAMPED", "IMPACT_REPOSITIONED_STABLE") and e["site"] == "CENTER"]
    captured = [e for e in results if e["end_state"] == "held_12s" and e["site"] == "CENTER"]
    strong = captured[-1:] if captured else []
    if damped:
        ap(f"python training/demo_ballistic_impact.py --case central_stable   # alias {tag(damped[0]['site'], damped[0]['v_hit'])}")
    if strong:
        ap(f"python training/demo_ballistic_impact.py --case central_strong   # alias {tag(strong[-1]['site'], strong[-1]['v_hit'])}")
    up = next((e for e in results if e["site"] == "UPPER"), None)
    lo = next((e for e in results if e["site"] == "LOWER"), None)
    if up:
        ap(f"python training/demo_ballistic_impact.py --case upper_offset   # alias {tag(up['site'], up['v_hit'])}")
    if lo:
        ap(f"python training/demo_ballistic_impact.py --case lower_offset   # alias {tag(lo['site'], lo['v_hit'])}")
    if found_c is not None:
        ap(f"python training/demo_ballistic_impact.py --case progressive   # alias {tag(found_c['site'], found_c['v_hit'])}")
    ap("```")
    ap("")
    ap("Or: `python training/map_ballistic_disturbance.py --case CENTER --v 4`")
    ap("")
    ap("## 10. Unresolved issues")
    ap("")
    ap("- Finger-span exit is a geometric heuristic, not a contact-manifold proof.")
    ap("- `J` omits finger forces during the impact window.")
    ap("- Axial offset uses object-frame z at release; residual geometry error should be checked in the viewer.")
    ap("- Mechanism labels are not visually confirmed.")
    ap("")
    ap("## 11. USER VISUAL OBSERVATION: PENDING")
    (OUT / "BALLISTIC_DISTURBANCE_MAP.md").write_text("\n".join(lines), encoding="utf-8")
    # aliases file
    aliases = {}
    if damped:
        aliases["central_stable"] = tag(damped[0]["site"], damped[0]["v_hit"])
    if strong:
        aliases["central_strong"] = tag(strong[-1]["site"], strong[-1]["v_hit"])
    if up:
        aliases["upper_offset"] = tag(up["site"], up["v_hit"])
    if lo:
        aliases["lower_offset"] = tag(lo["site"], lo["v_hit"])
    if found_c is not None:
        aliases["progressive"] = tag(found_c["site"], found_c["v_hit"])
    for e in results:
        aliases[tag(e["site"], e["v_hit"])] = tag(e["site"], e["v_hit"])
    dump_json(MAP / "case_aliases.json", aliases)


def parse_case(name: str) -> tuple[str, float]:
    aliases_p = MAP / "case_aliases.json"
    aliases = json.loads(aliases_p.read_text(encoding="utf-8")) if aliases_p.is_file() else {}
    key = aliases.get(name, name)
    # center_v3p0 / upper_v4p0
    parts = key.split("_v")
    if len(parts) == 2:
        site = parts[0].upper()
        v = float(parts[1].replace("p", "."))
        if site in Z_OFF:
            return site, v
    raise SystemExit(f"unknown --case {name!r}. Known: {sorted(set(aliases) | set(Z_OFF))}")


def _viewer_overlay(vwr, rows: list[tuple[str, str]]) -> None:
    """MuJoCo 3.12 uses set_texts; older Handle had add_overlay."""
    left = "\n".join(k for k, _ in rows)
    right = "\n".join(str(v) for _, v in rows)
    if hasattr(vwr, "set_texts"):
        font = mujoco.mjtFontScale.mjFONTSCALE_150
        pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
        vwr.set_texts([(font, pos, left, right)])
        return
    if hasattr(vwr, "add_overlay"):
        pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
        for k, v in rows:
            vwr.add_overlay(pos, k, str(v))


def interactive(site: str, v: float, case_name: str) -> None:
    import mujoco.viewer

    sim, cfg = make_sim()
    gains = gains_from_cfg(cfg)
    ctl = {"pause": False, "reset": False, "abort": False, "speed": 0.45, "overlay": {}}

    def sync():
        vwr = ctl.get("viewer")
        if vwr is None:
            return
        o = ctl.get("overlay") or {}
        _viewer_overlay(
            vwr,
            [
                ("CASE", str(o.get("case", case_name))),
                ("t", f"{float(o.get('t', sim.data.time)):.3f} s"),
                ("PHASE", str(o.get("phase", sim.fsm.phase)).upper()),
                ("nL / nR", f"{o.get('nL', '-')}/{o.get('nR', '-')}"),
                ("keys", "SPACE pause   R restart   [ ] speed"),
            ],
        )
        vwr.sync()
        dt = float(sim.model.opt.timestep)
        spd = max(float(ctl.get("speed", 0.45)), 0.05)
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

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        ctl["viewer"] = vwr
        apply_camera_preset(vwr, CAM)
        while vwr.is_running():
            ctl["reset"] = False
            ctl["_wall"] = None
            run_episode(sim, gains, v, site, ctl=ctl, label=case_name)
            if ctl.get("abort") or not vwr.is_running():
                break
            if ctl.get("reset"):
                continue
            while vwr.is_running() and not ctl.get("reset"):
                sync()
                time.sleep(0.03)
            if not ctl.get("reset"):
                break


def main():
    p = argparse.ArgumentParser(description="Ballistic disturbance map (no MP4).")
    p.add_argument("--case", default="", help="viewer case, e.g. center_v3p0 or central_stable")
    p.add_argument("--site", default="", choices=("", *Z_OFF))
    p.add_argument("--v", type=float, default=None)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    if args.case:
        site, v = parse_case(args.case)
        interactive(site, v, args.case)
        return
    if args.site and args.v is not None:
        interactive(args.site, float(args.v), f"{args.site}_{args.v}")
        return
    payload = run_headless()
    write_map_md(payload)
    print("wrote", OUT / "BALLISTIC_DISTURBANCE_MAP.md")


if __name__ == "__main__":
    main()
