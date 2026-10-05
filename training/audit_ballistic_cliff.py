"""Audit of the apparent ballistic capture-loss cliff. ZERO only.

Does not change noslip, friction, ball mass, recovery, recatch, or heuristic.
Does not treat I0–I3 labels as physical conclusions.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    RENDER_H,
    RENDER_W,
    TAU_SEC,
    assert_noslip,
    ball_force_on_ball,
    dump_json,
    launch_state,
    live_opt,
    make_sim,
    pack_row,
    pres_show,
    write_frames,
    z_tgt_of,
)
from training.impact_ball import apply_ball_free_state, incoming_hand_x
from training.impact_visualization_utils import mjv_camera_from_preset
from training.replay_core import tick_vw

OUT = ROOT / "results" / "diagnostics" / "ballistic_impact_noslip1"
RAW = OUT / "raw"
AUD = OUT / "cliff_audit"
FIG = AUD / "figures"
VIDB = OUT / "videos" / "boundary"
VIDM = OUT / "videos" / "matched"

BOUNDARY_V = (2.70, 2.85, 2.95, 2.967, 2.983, 3.000, 3.017, 3.033, 3.05)
MATCHED_V = (2.6, 2.7, 2.8, 2.9, 3.0, 3.1)
T_FLIGHT_MATCHED = 0.10
HIRES_PRE = 0.020
HIRES_POST = 0.200
SLOW_PRE = 0.100
SLOW_POST = 0.500
SPEED_TASK = 0.70
SPEED_SLOW = 0.08
SPEED_AFTER = 1.10
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


def launch_state_matched(sim, v_hit: float, t_flight: float = T_FLIGHT_MATCHED) -> dict:
    """Aim at predicted cylinder pose after t_flight. Isolates speed from TOF geometry."""
    u = incoming_hand_x(sim)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    vo, _ = body_twist(sim.model, sim.data, sim.ids.object_body)
    r_ball = float(sim.model.geom_size[int(sim.ball_geom)][0])
    r_obj = float(sim.model.geom_size[int(sim.ids.object_geom)][0])
    r_surf = r_obj + r_ball
    T = float(t_flight)
    g = np.array(sim.model.opt.gravity, float)
    po_pred = po + vo * T
    p_hit = po_pred - u * r_surf
    along = float(v_hit) * T
    p_launch = p_hit - u * along
    v0 = (p_hit - p_launch) / T - 0.5 * g * T
    return {
        "mode": "matched_predicted_cylinder",
        "t_flight": T,
        "p_obj_release": po.copy(),
        "v_obj_release": vo.copy(),
        "p_obj_pred": po_pred.copy(),
        "p_hit": p_hit,
        "p_launch": p_launch,
        "v0": v0,
        "direction": u,
        "r_surf": r_surf,
        "v_impact_requested": float(v_hit),
        "hand_x_world": u.tolist(),
    }


def mat2quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def enrich_row(sim, r: dict) -> dict:
    r["aperture"] = finger_opening(sim.data, sim.ids)
    r["p_des_err"] = float(np.linalg.norm(r["p_des"] - r["p_hand"]))
    r["r_des"] = np.asarray(sim.fsm.r_des, float).copy()
    r["q_obj"] = mat2quat(r["R_obj"])
    r["q_hand"] = mat2quat(r["R_hand"])
    return r


def cylinder_rel_point(sim, pos_w: np.ndarray) -> dict:
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    r_cyl = Ro.T @ (np.asarray(pos_w, float) - po)
    r_h = Rh.T @ (np.asarray(pos_w, float) - ph)
    return {
        "pos_w": np.asarray(pos_w, float).tolist(),
        "pos_cyl": r_cyl.tolist(),
        "pos_h": r_h.tolist(),
        "axial_z_cyl": float(r_cyl[2]),
        "radial_cyl": float(np.linalg.norm(r_cyl[:2])),
        "below_center": bool(r_cyl[2] < -0.005),
        "above_center": bool(r_cyl[2] > 0.005),
        "near_end": bool(abs(r_cyl[2]) > 0.022),
    }


def playback_speed(t: float, t_impact: float | None) -> float:
    if t_impact is None:
        return SPEED_TASK
    if t_impact - SLOW_PRE <= t <= t_impact + SLOW_POST:
        return SPEED_SLOW
    if t < t_impact:
        return SPEED_TASK
    return SPEED_AFTER


def run_audit_episode(sim, gains, v_hit: float, *, matched: bool, record=False, hires=False, label="AUDIT"):
    """Same physics/controller as the cliff sweep. Does not stop on a 1-step both-off."""
    assert_noslip(sim)
    sim.reset()
    dt = float(sim.model.opt.timestep)
    log_n = max(int(round(LOG_DT / dt)), 1)
    rows = []
    hires_rows = []
    events = []
    hand_z0 = None
    t_lift0 = None
    released = False
    t_release = None
    t_impact = None
    launch = None
    first_c = None
    Rh_imp = None
    p_des_imp = None
    v_cmd_imp = None
    t_ul = None
    t_both = None
    t_both_restore = None
    both_off_s = 0.0
    both_off_acc = 0.0
    t_table = None
    t_xy_escape = None
    t_lift_done = None
    z_tgt = None
    holding = False
    p_des_before = None
    r_des_before = None
    impact_geom = None
    pre_buf = []
    ctrl_across = []
    n_max = int(round(30.0 / dt))
    pres = None
    if record:
        renderer = mujoco.Renderer(sim.model, RENDER_H, RENDER_W)
        cam = mjv_camera_from_preset(CAM)
        pres = {
            "record": True,
            "renderer": renderer,
            "cam": cam,
            "speed": SPEED_TASK,
            "phase": "APPROACH",
            "frames": [],
            "frame_t": [],
            "_rt": None,
        }

    for i in range(n_max):
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
                p_des_imp = np.asarray(sim.fsm.p_des, float).copy()
                v_cmd_imp = np.asarray(sim.fsm.v_cmd, float).copy()
                impact_geom = cylinder_rel_point(sim, first_c["pos"])
                po = np.array(sim.data.xpos[sim.ids.object_body], float)
                ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
                impact_geom["p_obj_imp"] = po.tolist()
                impact_geom["p_hand_imp"] = ph.tolist()
                impact_geom["obj_z_imp"] = float(po[2])
                impact_geom["hand_z_imp"] = float(ph[2])
                impact_geom["fsm"] = str(sim.fsm.phase)
                events.append({"event": "impact", "t": t})
                for br in pre_buf:
                    if br["t"] >= t_impact - HIRES_PRE - 1e-12:
                        hires_rows.append(br)

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
            p_des_before = np.asarray(sim.fsm.p_des, float).copy()
            r_des_before = np.asarray(sim.fsm.r_des, float).copy()
            launch = launch_state_matched(sim, v_hit) if matched else launch_state(sim, v_hit)
            sim.set_guide_weld(False)
            apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
            released = True
            t_release = float(sim.data.time)
            events.append({"event": "release", "t": t_release, "eq_active": sim.eq_active(), "matched": matched})

        nL, nR = int(o["nL"]), int(o["nR"])
        if t_impact is not None:
            if (nL == 0 or nR == 0) and t_ul is None:
                t_ul = t
            if nL == 0 and nR == 0:
                if t_both is None:
                    t_both = t
                both_off_acc += dt
                both_off_s = max(both_off_s, both_off_acc)
            else:
                if t_both is not None and t_both_restore is None and nL > 0 and nR > 0:
                    t_both_restore = t
                both_off_acc = 0.0
            ph = np.array(sim.data.xpos[sim.ids.hand_body][:2], float)
            po = np.array(sim.data.xpos[sim.ids.object_body][:2], float)
            if t_xy_escape is None and float(np.linalg.norm(po - ph)) > 0.18:
                t_xy_escape = t
            if t_table is None and float(o["obj_z"]) < TABLE_DROP:
                t_table = t
            if abs(t - t_impact) <= 0.30:
                ctrl_across.append(
                    {
                        "t": t,
                        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
                        "r_des": np.asarray(sim.fsm.r_des, float).copy(),
                        "p_hand": np.array(sim.data.xpos[sim.ids.hand_body], float).copy(),
                        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
                        "fsm": str(sim.fsm.phase),
                    }
                )

        fine = t_impact is not None and t <= t_impact + HIRES_POST + 1e-12
        pre_release = released and t_impact is None
        want = (i % log_n == 0) or fine or pre_release
        if want:
            r = enrich_row(sim, pack_row(sim, TAU_SEC))
            rows.append(r)
            if pre_release:
                pre_buf.append(r)
                max_n = int(round(HIRES_PRE / dt)) + 4
                if len(pre_buf) > max_n:
                    pre_buf.pop(0)
            if fine:
                hires_rows.append(r)

        if pres is not None:
            nfr0 = len(pres.get("frames") or [])
            pres["speed"] = playback_speed(t, t_impact)
            if t_impact is None:
                pres["phase"] = "FREE FLIGHT" if released else str(sim.fsm.phase).upper()
            elif t_table is not None:
                pres["phase"] = "TABLE"
            elif nL == 0 and nR == 0:
                pres["phase"] = "BOTH-OFF"
            elif holding:
                pres["phase"] = "HIGH HOLD"
            else:
                pres["phase"] = "LIFT"
            pres_show(pres, sim, label)
            if len(pres.get("frames") or []) > nfr0:
                pres["frame_t"].append(t)

        table_end = t_table is not None and t >= t_table + AFTER_DROP_S
        held = t_lift_done is not None and t_table is None and t >= t_lift_done + HOLD_S
        if table_end or held or t > 28.0:
            break

    frames = []
    frame_t = []
    if pres is not None:
        frames = list(pres.get("frames") or [])
        frame_t = list(pres.get("frame_t") or [])
        pres["renderer"].close()

    samples = {}
    if t_impact is not None:
        for tau in (0.020, 0.050, 0.100, 0.200):
            cand = [r for r in rows if r["t"] >= t_impact + tau]
            if cand:
                r = cand[0]
                samples[f"plus_{int(tau*1000)}ms"] = {
                    "t": r["t"],
                    "rh": r["rh"].tolist(),
                    "cyl_h": r["cyl_h"].tolist(),
                    "v_rel": float(np.linalg.norm(r["v_rel_h"])),
                    "w_rel": float(np.linalg.norm(r["w_rel_h"])),
                    "nL": r["nL"],
                    "nR": r["nR"],
                    "Fn_L": r["Fn_L"],
                    "Fn_R": r["Fn_R"],
                    "aperture": r.get("aperture"),
                    "obj_z": r["obj_z"],
                    "clear": r["clear"],
                }

    ctrl_jump = None
    if p_des_imp is not None and p_des_before is not None:
        r_des_imp = None
        if ctrl_across:
            r_des_imp = ctrl_across[0].get("r_des")
        ctrl_jump = {
            "p_des_delta_release_to_impact": float(np.linalg.norm(p_des_imp - p_des_before)),
            "r_des_reset": False
            if r_des_before is None or r_des_imp is None
            else float(np.linalg.norm(np.asarray(r_des_imp) - r_des_before)) > 1e-6,
            "v_cmd_at_impact": None if v_cmd_imp is None else v_cmd_imp.tolist(),
            "fsm_at_impact": "lift",
            "freeze_used": False,
            "snapshot_reload": False,
            "p_des_err_samples": [
                {
                    "t": c["t"],
                    "p_des_minus_p_hand": (np.asarray(c["p_des"]) - np.asarray(c["p_hand"])).tolist(),
                    "v_cmd": np.asarray(c["v_cmd"]).tolist(),
                    "fsm": c["fsm"],
                }
                for c in ctrl_across
                if abs(c["t"] - t_impact) <= 0.05
            ][:12],
        }

    # post-impact instability class (descriptive, not policy)
    mode = "unknown"
    if t_impact is None:
        mode = "no_impact"
    elif t_table is not None and (t_table - t_impact) < 0.05 and (t_both_restore is None):
        mode = "D_immediate_ejection"
    elif t_table is not None:
        mode = "C_or_D_later_table"
    else:
        post = [r for r in rows if t_impact is not None and r["t"] > t_impact + 0.20]
        if post:
            ex = [abs(r["rh"][0]) for r in post]
            wr = [float(np.linalg.norm(r["w_rel_h"])) for r in post]
            if max(ex) < 0.004 and max(wr) < 0.8:
                mode = "B_stable_disturbed"
            elif ex[-1] > ex[min(5, len(ex) - 1)] + 0.003 or wr[-1] > wr[min(5, len(wr) - 1)] + 0.4:
                mode = "C_progressive_instability"
            else:
                mode = "A_or_B_damped_or_held"
        else:
            mode = "short_record"

    vb_h = None
    if t_impact is not None and Rh_imp is not None:
        pre = [r for r in rows if r["t"] < t_impact]
        if pre:
            vb_h = (Rh_imp.T @ np.asarray(pre[-1]["v_ball"], float)).tolist()

    ep = {
        "v_hit": float(v_hit),
        "matched": bool(matched),
        "t_lift0": t_lift0,
        "t_release": t_release,
        "t_impact": t_impact,
        "free_flight_s": None if (t_release is None or t_impact is None) else t_impact - t_release,
        "t_first_unilateral": t_ul,
        "t_first_both_off": t_both,
        "t_bilateral_restored": t_both_restore,
        "duration_both_off_max_s": both_off_s,
        "t_table": t_table,
        "t_xy_escape": t_xy_escape,
        "t_lift_done": t_lift_done,
        "t_end": float(sim.data.time),
        "legacy_dropped_first_fire": None,
        "impact_geom": impact_geom,
        "first_contact": None
        if first_c is None
        else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in first_c.items() if k != "f_ball_w"},
        "v_ball_h_pre": vb_h,
        "post_samples": samples,
        "ctrl_jump": ctrl_jump,
        "descriptive_mode": mode,
        "end_nL": rows[-1]["nL"] if rows else None,
        "end_nR": rows[-1]["nR"] if rows else None,
        "end_obj_z": rows[-1]["obj_z"] if rows else None,
        "launch": None if launch is None else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in launch.items()},
        "live": live_opt(sim),
        "rows": rows,
        "hires_rows": hires_rows,
        "frames": frames,
        "frame_t": frame_t,
        "t_impact_dt": dt,
    }
    # reconstruct when legacy classifier would have fired
    if t_impact is not None:
        for r in rows:
            if r["t"] + 1e-12 < t_impact:
                continue
            fake = {"obj_z": r["obj_z"], "nL": r["nL"], "nR": r["nR"]}
            if (r["nL"] == 0 and r["nR"] == 0) or float(r["obj_z"]) < TABLE_DROP:
                ep["legacy_dropped_first_fire"] = r["t"]
                ep["legacy_delay_s"] = r["t"] - t_impact
                break
    return ep


def analyze_existing() -> dict:
    table = []
    for p in sorted(RAW.glob("ep_v*.json")):
        ep = json.loads(p.read_text(encoding="utf-8"))
        v = float(ep["v_hit"])
        traj = RAW / f"traj_v{v:.2f}.npz"
        if not traj.is_file():
            traj = RAW / f"traj_v{v:.3f}.npz"
        n_return = None
        both_dur = None
        if traj.is_file() and ep.get("t_impact") is not None:
            d = np.load(traj, allow_pickle=True)
            t = d["t"]
            nL, nR = d["nL"], d["nR"]
            ti = float(ep["t_impact"])
            both = (nL == 0) & (nR == 0) & (t >= ti - 1e-9)
            if np.any(both):
                i0 = int(np.argmax(both))
                i1 = i0
                while i1 < len(both) and both[i1]:
                    i1 += 1
                both_dur = float(t[min(i1, len(t) - 1)] - t[i0]) if i1 > i0 else float(d.get("t", [0])[0] * 0)
                both_dur = float(t[i1 - 1] - t[i0] + 0.002)
                after = (t > t[i1 - 1] + 1e-9) & (nL > 0) & (nR > 0)
                n_return = bool(np.any(after))
        fc = ep.get("first_contact") or {}
        pos = fc.get("pos")
        table.append(
            {
                "file": p.name,
                "v_hit": v,
                "legacy_class": ep.get("class"),
                "t_release": ep.get("t_release"),
                "t_impact": ep.get("t_impact"),
                "free_flight_s": ep.get("free_flight_s"),
                "t_unilateral": ep.get("t_unilateral"),
                "t_both": ep.get("t_both"),
                "t_drop_legacy": ep.get("t_drop"),
                "legacy_delay_s": None
                if ep.get("t_drop") is None or ep.get("t_impact") is None
                else ep["t_drop"] - ep["t_impact"],
                "end_nL": ep.get("end_nL"),
                "end_nR": ep.get("end_nR"),
                "end_obj_z": ep.get("end_obj_z"),
                "hold_survived_flag": ep.get("hold_survived"),
                "contact_pos_w": pos,
                "v_ball_h_pre": ep.get("v_ball_h_pre"),
                "npz_both_off_dur_s": both_dur,
                "npz_contacts_returned": n_return,
            }
        )
    dump_json(AUD / "existing_sweep_events.json", table)
    return {"n": len(table), "rows": table}


def dump_hires_csv(ep: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ep.get("hires_rows") or []
    if not rows:
        return
    ap_prev = None
    t_prev = None
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t",
                "obj_x", "obj_y", "obj_z",
                "qobj_w", "qobj_x", "qobj_y", "qobj_z",
                "hand_x", "hand_y", "hand_z",
                "qhand_w", "qhand_x", "qhand_y", "qhand_z",
                "rh_x", "rh_y", "rh_z",
                "vobj_x", "vobj_y", "vobj_z",
                "wobj_x", "wobj_y", "wobj_z",
                "vhand_x", "vhand_y", "vhand_z",
                "whand_x", "whand_y", "whand_z",
                "vrel_hx", "vrel_hy", "vrel_hz",
                "wrel_hx", "wrel_hy", "wrel_hz",
                "nL", "nR", "Fn_L", "Fn_R", "rho_max",
                "aperture", "daperture", "ctrl7",
                "pdes_x", "pdes_y", "pdes_z", "vcmd_z", "fsm",
                "n_finger_contacts", "contact_geoms", "contact_pos", "contact_n",
                "Fn_list", "Ft1_list", "Ft2_list", "rho_list",
            ]
        )
        for r in rows:
            ap = r.get("aperture")
            da = None
            if ap_prev is not None and t_prev is not None:
                da = (ap - ap_prev) / max(r["t"] - t_prev, 1e-9)
            fc = r.get("fc") or []
            geoms = ";".join(f"{c.get('geom1')}:{c.get('geom1_name')}-{c.get('geom2')}:{c.get('geom2_name')}" for c in fc)
            poss = ";".join(str(c.get("pos_w")) for c in fc)
            ns = ";".join(str(c.get("n_w")) for c in fc)
            fns = ";".join(str(c.get("Fn")) for c in fc)
            ft1s = ";".join(str(c.get("Ft1")) for c in fc)
            ft2s = ";".join(str(c.get("Ft2")) for c in fc)
            rhos = ";".join(str(c.get("rho")) for c in fc)
            qo = r.get("q_obj", [None] * 4)
            qh = r.get("q_hand", [None] * 4)
            w.writerow(
                [
                    r["t"], *r["p_obj"], *qo, *r["p_hand"], *qh, *r["rh"],
                    *r["v_obj"], *r["w_obj"], *r["v_hand"], *r["w_hand"],
                    *r["v_rel_h"], *r["w_rel_h"],
                    r["nL"], r["nR"], r["Fn_L"], r["Fn_R"], r.get("rho_max"),
                    ap, da, r["ctrl7"], *r["p_des"], r["v_cmd"][2], r["fsm"],
                    len(fc), geoms, poss, ns, fns, ft1s, ft2s, rhos,
                ]
            )
            ap_prev, t_prev = ap, r["t"]


def pub(ep: dict) -> dict:
    skip = {"rows", "hires_rows", "frames", "frame_t", "ctrl_across"}
    return {k: _jsonable(v) for k, v in ep.items() if k not in skip}


def figures_geometry(eps: list, tag: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    vs = [e["v_hit"] for e in eps]
    tof = [e["free_flight_s"] for e in eps]
    zobj = [None if not e.get("impact_geom") else e["impact_geom"].get("obj_z_imp") for e in eps]
    axz = [None if not e.get("impact_geom") else e["impact_geom"].get("axial_z_cyl") for e in eps]
    fig, ax = plt.subplots(3, 1, figsize=(7.4, 6.4), sharex=True)
    ax[0].plot(vs, tof, "o-")
    ax[0].set_ylabel("free-flight s")
    ax[1].plot(vs, zobj, "o-")
    ax[1].set_ylabel("obj_z at impact")
    ax[2].plot(vs, axz, "o-")
    ax[2].set_ylabel("contact z in cyl frame")
    ax[2].set_xlabel("v_hit")
    fig.tight_layout()
    fig.savefig(FIG / f"geom_vs_speed_{tag}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(4, 1, figsize=(8.0, 8.0), sharex=True)
    for e in eps:
        rows = e.get("rows") or []
        ti = e.get("t_impact")
        if not rows or ti is None:
            continue
        t = np.array([r["t"] for r in rows]) - ti
        mask = t >= -0.05
        t = t[mask]
        rhx = np.array([r["rh"][0] for r in rows])[mask]
        nsum = np.array([r["nL"] + r["nR"] for r in rows])[mask]
        vz = np.array([np.linalg.norm(r["v_rel_h"]) for r in rows])[mask]
        oz = np.array([r["obj_z"] for r in rows])[mask]
        lab = f"{e['v_hit']:.3f}"
        ax[0].plot(t, 1e3 * rhx, label=lab, lw=0.9)
        ax[1].plot(t, nsum, lw=0.9)
        ax[2].plot(t, vz, lw=0.9)
        ax[3].plot(t, oz, lw=0.9)
    ax[0].set_ylabel("rh.x mm")
    ax[1].set_ylabel("nL+nR")
    ax[2].set_ylabel("|v_rel_h|")
    ax[3].set_ylabel("obj z")
    ax[3].set_xlabel("t - t_impact (s)")
    ax[0].legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"post_impact_{tag}.png", dpi=120)
    plt.close(fig)


def side_by_side(left: dict, right: dict, dest: Path) -> str | None:
    def aligned(ep, n_pre=50, n_post=280):
        ts = ep.get("frame_t") or []
        fr = ep.get("frames") or []
        ti = ep.get("t_impact")
        if not fr or ti is None or not ts:
            return fr
        i0 = 0
        for i, tt in enumerate(ts):
            if tt >= ti:
                i0 = i
                break
        a = max(0, i0 - n_pre)
        b = min(len(fr), i0 + n_post)
        return fr[a:b]

    L, R = aligned(left), aligned(right)
    n = max(len(L), len(R))
    if n == 0:
        return None
    out = []
    for i in range(n):
        a = L[i] if i < len(L) else L[-1]
        b = R[i] if i < len(R) else R[-1]
        out.append(np.concatenate([a, b], axis=1))
    return write_frames(out, dest)


def write_report(meta: dict) -> None:
    p = AUD / "AUTO_SUMMARY.md"
    p.write_text(
        "Auto dump from audit_ballistic_cliff.py. Canonical write-up is "
        "../BALLISTIC_CLIFF_AUDIT.md (do not treat this dump as the audit).\n\n"
        + json.dumps(
            {
                "timing_original": meta.get("timing_original"),
                "points_original": meta.get("points_original"),
                "matched_events": [
                    {k: e.get(k) for k in (
                        "v_hit", "t_impact", "free_flight_s", "t_first_both_off",
                        "t_bilateral_restored", "duration_both_off_max_s", "t_table",
                        "mode", "legacy_dropped_first_fire",
                    )}
                    for e in (meta.get("matched_events") or [])
                ],
                "modes": meta.get("modes"),
                "aiming": meta.get("aiming"),
                "classifier": meta.get("classifier"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def vname(v: float) -> str:
    s = f"{v:.3f}".replace(".", "p")
    return "v" + s


def aliases(v: float) -> list[str]:
    out = [vname(v)]
    two = f"v{v:.2f}".replace(".", "p")
    if two not in out:
        out.append(two)
    return out


def save_traj(ep: dict, path: Path) -> None:
    rows = ep.get("rows") or []
    if not rows:
        return
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in rows]),
        nL=np.array([r["nL"] for r in rows]),
        nR=np.array([r["nR"] for r in rows]),
        rh=np.array([r["rh"] for r in rows]),
        obj_z=np.array([r["obj_z"] for r in rows]),
        Fn_L=np.array([r["Fn_L"] for r in rows]),
        Fn_R=np.array([r["Fn_R"] for r in rows]),
        rho=np.array([r.get("rho_max", np.nan) for r in rows]),
        vrel=np.array([np.linalg.norm(r["v_rel_h"]) for r in rows]),
        wrel=np.array([np.linalg.norm(r["w_rel_h"]) for r in rows]),
        aperture=np.array([r.get("aperture", np.nan) for r in rows]),
        p_des_err=np.array([r.get("p_des_err", np.nan) for r in rows]),
        fsm=np.array([r["fsm"] for r in rows]),
    )


def save_hires_npz(ep: dict, path: Path) -> None:
    rows = ep.get("hires_rows") or []
    if not rows:
        return
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in rows]),
        nL=np.array([r["nL"] for r in rows]),
        nR=np.array([r["nR"] for r in rows]),
        p_obj=np.array([r["p_obj"] for r in rows]),
        p_hand=np.array([r["p_hand"] for r in rows]),
        rh=np.array([r["rh"] for r in rows]),
        v_obj=np.array([r["v_obj"] for r in rows]),
        w_obj=np.array([r["w_obj"] for r in rows]),
        v_hand=np.array([r["v_hand"] for r in rows]),
        w_hand=np.array([r["w_hand"] for r in rows]),
        v_rel_h=np.array([r["v_rel_h"] for r in rows]),
        w_rel_h=np.array([r["w_rel_h"] for r in rows]),
        Fn_L=np.array([r["Fn_L"] for r in rows]),
        Fn_R=np.array([r["Fn_R"] for r in rows]),
        ctrl7=np.array([r["ctrl7"] for r in rows]),
        p_des=np.array([r["p_des"] for r in rows]),
        v_cmd=np.array([r["v_cmd"] for r in rows]),
    )


def main():
    AUD.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    VIDB.mkdir(parents=True, exist_ok=True)
    VIDM.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    existing = analyze_existing()

    orig_eps = []
    for v in BOUNDARY_V:
        sim, _ = make_sim()
        gains = gains_from_cfg(cfg)
        ep = run_audit_episode(
            sim, gains, v, matched=False, record=True, hires=abs(v - 2.967) < 1e-4
        )
        dump_json(AUD / f"orig_{vname(v)}.json", pub(ep))
        orig_eps.append(ep)
        print(
            f"orig v={v:.3f} tof={ep['free_flight_s']} both={ep['t_first_both_off']} "
            f"restore={ep['t_bilateral_restored']} table={ep['t_table']} mode={ep['descriptive_mode']} "
            f"legacy_fire={ep.get('legacy_dropped_first_fire')}",
            flush=True,
        )
        write_frames(ep["frames"], VIDB / f"{vname(v)}.mp4")
        for alt in aliases(v):
            src = VIDB / f"{vname(v)}.mp4"
            dst = VIDB / f"{alt}.mp4"
            if src.is_file() and dst != src:
                dst.write_bytes(src.read_bytes())
        print("wrote", VIDB / f"{vname(v)}.mp4", flush=True)
        save_traj(ep, AUD / f"traj_orig_{vname(v)}.npz")
        if abs(v - 2.967) < 1e-4:
            dump_hires_csv(ep, AUD / "hires_v2p967.csv")
            save_hires_npz(ep, AUD / "hires_v2p967.npz")

    captured = None
    reported_loss = None
    for e in orig_eps:
        if abs(e["v_hit"] - 2.95) < 1e-6:
            captured = e
        if abs(e["v_hit"] - 2.967) < 1e-4:
            reported_loss = e
    if captured and reported_loss and captured["frames"] and reported_loss["frames"]:
        side_by_side(captured, reported_loss, VIDB / "captured_vs_reported_loss.mp4")
        print("wrote side-by-side", flush=True)

    figures_geometry(orig_eps, "original")

    matched_eps = []
    for v in MATCHED_V:
        sim, _ = make_sim()
        gains = gains_from_cfg(cfg)
        rec = v in (2.7, 3.0, 3.1) or True
        ep = run_audit_episode(sim, gains, v, matched=True, record=True)
        dump_json(AUD / f"matched_{vname(v)}.json", pub(ep))
        matched_eps.append(ep)
        write_frames(ep["frames"], VIDM / f"{vname(v)}.mp4")
        save_traj(ep, AUD / f"traj_matched_{vname(v)}.npz")
        print(
            f"matched v={v:.3f} tof={ep['free_flight_s']} both={ep['t_first_both_off']} "
            f"restore={ep['t_bilateral_restored']} table={ep['t_table']} mode={ep['descriptive_mode']}",
            flush=True,
        )
    figures_geometry(matched_eps, "matched")

    def slim_evt(e):
        return {
            "v_hit": e["v_hit"],
            "matched": e["matched"],
            "t_release": e["t_release"],
            "t_impact": e["t_impact"],
            "free_flight_s": e["free_flight_s"],
            "t_first_unilateral": e["t_first_unilateral"],
            "t_first_both_off": e["t_first_both_off"],
            "t_bilateral_restored": e["t_bilateral_restored"],
            "duration_both_off_max_s": e["duration_both_off_max_s"],
            "t_table": e["t_table"],
            "t_xy_escape": e["t_xy_escape"],
            "t_lift_done": e["t_lift_done"],
            "t_end": e["t_end"],
            "legacy_dropped_first_fire": e.get("legacy_dropped_first_fire"),
            "legacy_delay_s": e.get("legacy_delay_s"),
            "end_nL": e["end_nL"],
            "end_nR": e["end_nR"],
            "end_obj_z": e["end_obj_z"],
            "mode": e["descriptive_mode"],
            "impact_geom": e.get("impact_geom"),
            "v_ball_h_pre": e.get("v_ball_h_pre"),
            "ctrl_jump": e.get("ctrl_jump"),
            "post_samples": e.get("post_samples"),
        }

    orig_s = [slim_evt(e) for e in orig_eps]
    mat_s = [slim_evt(e) for e in matched_eps]
    tr = next((e for e in orig_s if abs(e["v_hit"] - 2.967) < 1e-3), None)
    unresolved = [
        "Original speed sweep aims at stale COM at release; TOF shrinks with v_hit.",
        "Legacy IMMEDIATE_LOSS can fire on a 1-step both-off; inspect restore times before calling it a drop.",
        "Matched-geometry sweep is a diagnostic isolation of impulse vs location, not a new disturbance family claim.",
        "Visual inspection is still required; no physical cliff is confirmed here.",
    ]
    meta = {
        "existing": existing,
        "timing_original": [{k: e[k] for k in ("v_hit", "t_release", "t_impact", "free_flight_s")} for e in orig_s],
        "points_original": [{k: e[k] for k in ("v_hit", "impact_geom", "v_ball_h_pre")} for e in orig_s],
        "prestate_original": orig_s,
        "trace_2967": tr,
        "continuum_original": [{k: e[k] for k in ("v_hit", "post_samples", "mode", "t_bilateral_restored", "t_table")} for e in orig_s],
        "matched_events": mat_s,
        "modes": {
            "original": [{k: e[k] for k in ("v_hit", "mode")} for e in orig_s],
            "matched": [{k: e[k] for k in ("v_hit", "mode")} for e in mat_s],
        },
        "unresolved": " ".join(unresolved),
        "classifier": {
            "dropped": "sim.dropped() OR obj_z<TABLE_DROP OR (nL==0 AND nR==0)",
            "IMMEDIATE_LOSS": "legacy t_drop - t_impact < 0.10 s",
            "note": "nL==nR==0 is instantaneous, not a duration.",
        },
        "aiming": "B: cylinder COM at release; T=dist/v_hit; not predicted future pose.",
    }
    dump_json(AUD / "summary.json", meta)
    write_report(meta)
    print("wrote", OUT / "BALLISTIC_CLIFF_AUDIT.md", "and", AUD / "AUTO_SUMMARY.md")


if __name__ == "__main__":
    main()
