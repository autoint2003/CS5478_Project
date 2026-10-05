"""Matched ballistic-impact comparison: ZERO | CONTROLLED_SLIP | AIRBORNE_RECAPTURE.

Same CENTER-6 family (scene, cylinder, ball mass, v=6 m/s, CENTER geometry,
noslip=1, nominal grasp). Video starts before the disturbance.

ZERO uses continue_zero (nominal lift-to-hold, tau=-18). Not stale-p_des STOP.

CONTROLLED_SLIP and AIRBORNE_RECAPTURE are privileged authority constructions,
not learned policies.

Diagnostic omega_y=4 is used only on the airborne branch via tick_omega_override.
RECOVERY4D_W_HY_MAX remains 3 rad/s.
"""

from __future__ import annotations

import argparse
import hashlib
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

from controllers.jacobian_controller import gains_from_cfg
from controllers.residual import RECOVERY4D_W_HY_MAX
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import TABLE_DROP, physical_pack
from training.ballistic_large_angle import (
    BRAKE_S,
    RETURN_HAND_STOP_DEG,
    return_to_nominal,
    rotate_to,
    so3_angle_deg,
    tick_log,
)
from training.ballistic_slip_sufficiency import DURS
from training.demo_ballistic_impact import TAU_SEC, dump_json, make_sim, write_frames, z_tgt_of
from training.demo_ballistic_recovery import (
    SITE,
    V_HIT,
    continue_zero,
    make_snap,
    measure,
    restore_ballistic,
    run_episode,
)
from training.demo_dynamic_recatch import EARLY_PKL
from training.grav_reposition_v2_viz import apply_camera_preset, make_reset_cam_callback
from training.impact_visualization_utils import mjv_camera_from_preset
from training.map_ballistic_disturbance import _viewer_overlay
from training.release_state_map_omega4 import CLAIMED_OPEN_DEG, claimed_t_close, recatch_one
from training.replay_core import freeze, tick_vw
from training.write_ballistic_impact_scene import BALL_M

OUT = ROOT / "results" / "comparison" / "ballistic_recovery"
RAW = OUT / "raw"
VID = ROOT / "results" / "videos"
STAGES = ROOT / "results" / "diagnostics" / "ballistic_recovery_transfer" / "raw" / "stages.json"
REPORT = OUT / "BALLISTIC_RECOVERY_COMPARISON.md"

SLIP_S = 0.70
SLIP_TAU = -5.0
SLIP_THETA = 120.0
HOLD_AFTER_S = 2.0
CATCH_CONFIRM_S = 0.25
ZERO_AFTER_TABLE_S = 1.50
VIDEO_FPS = 25
RENDER_W, RENDER_H = 640, 480
MODES = ("zero", "controlled_slip", "airborne_recatch")

# Presentation camera: look at the grasp (not the table). Side/oblique so
# the cylinder stays visible in the finger gap after ~70–90° wrist-y rotation.
CAM = {
    "lookat": np.array([0.55, 0.00, 0.58]),
    "distance": 0.48,
    "azimuth": 250.0,
    "elevation": 12.0,
}
CAM_PROBE = (
    ("az250", CAM),
    ("az270", {**CAM, "azimuth": 270.0}),
    ("az220", {**CAM, "azimuth": 220.0}),
    ("az80", {**CAM, "azimuth": 80.0}),
    ("az10", {**CAM, "azimuth": 10.0, "elevation": 18.0}),
    ("az170", {**CAM, "azimuth": 170.0, "elevation": 15.0}),
)


def t_early_default() -> float:
    if STAGES.is_file():
        return float(json.loads(STAGES.read_text(encoding="utf-8"))["T_EARLY_DRIFT"])
    return 3.322


def branch_arrays(sim) -> dict:
    ncon = int(sim.data.ncon)
    geoms = []
    pos = []
    for i in range(ncon):
        c = sim.data.contact[i]
        geoms.append((int(c.geom1), int(c.geom2)))
        pos.append(np.array(c.pos, float))
    qadr = getattr(sim, "ball_qadr", None)
    dadr = getattr(sim, "ball_dadr", None)
    ball_p = np.array(sim.data.qpos[qadr : qadr + 3], float) if qadr is not None else np.zeros(3)
    ball_v = np.array(sim.data.qvel[dadr : dadr + 6], float) if dadr is not None else np.zeros(6)
    return {
        "t": np.array([float(sim.data.time)]),
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "r_des": np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).copy(),
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "cyl_p": np.array(sim.data.xpos[sim.ids.object_body], float).copy(),
        "cyl_R": np.array(sim.data.xmat[sim.ids.object_body], float).copy(),
        "hand_p": np.array(sim.data.xpos[sim.ids.hand_body], float).copy(),
        "ball_p": ball_p,
        "ball_v": ball_v,
        "ncon": np.array([ncon], float),
        "contact_geoms": np.array(geoms, float).reshape(-1) if geoms else np.zeros(0),
        "contact_pos": np.concatenate(pos) if pos else np.zeros(0),
        "eq": np.array([float(sim.eq_active())]),
    }


def hash_arrays(arrs: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        h.update(k.encode())
        h.update(np.asarray(arrs[k], float).tobytes())
    return h.hexdigest()


def max_abs_diff(a: dict, b: dict) -> dict:
    out = {}
    keys = sorted(set(a) | set(b))
    worst = 0.0
    for k in keys:
        if k not in a or k not in b:
            out[k] = None
            continue
        xa, xb = np.asarray(a[k], float).ravel(), np.asarray(b[k], float).ravel()
        n = min(xa.size, xb.size)
        d = float(np.max(np.abs(xa[:n] - xb[:n]))) if n else 0.0
        if xa.size != xb.size:
            d = max(d, abs(float(xa.size - xb.size)))
        out[k] = d
        worst = max(worst, d)
    out["max"] = worst
    return out


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    img = np.asarray(frame, np.uint8).copy()
    bar = 28
    img[:bar] = (18, 18, 18)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((8, 6), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:
        return img


class FrameRec:
    def __init__(self, sim, renderer, cam, fps=VIDEO_FPS):
        self.sim = sim
        self.renderer = renderer
        self.cam = cam
        self.dt = 1.0 / float(fps)
        self.last = -1e9
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []
        self.phases: list[str] = []

    def maybe(self, phase: str, label: str):
        t = float(self.sim.data.time)
        if t - self.last + 1e-12 < self.dt:
            return
        self.renderer.update_scene(self.sim.data, self.cam)
        fr = caption(self.renderer.render(), f"{label}  {phase}  t={t:.2f}s")
        self.frames.append(fr)
        self.times.append(t)
        self.phases.append(phase)
        self.last = t


def phase_from_overlay(ov: dict, t_impact) -> str:
    ph = str(ov.get("phase", "")).upper()
    if "FAIL" in ph or "TABLE" in ph:
        return "FAILURE"
    if ph in ("FREE FLIGHT",):
        return "IMPACT"
    if t_impact is not None and float(ov.get("t", 0)) >= float(t_impact) - 1e-6:
        if ph in ("HIGH HOLD", "LIFT", "HOLD"):
            return "IMPACT"
    if ph in ("APPROACH", "DESCEND", "GRASP", "LIFT", "NOMINAL"):
        return "NOMINAL"
    return ph or "NOMINAL"


def overlay_sync(sim, ctl, rec, vwr, label: str, t_impact=None, force_phase=None):
    ov = ctl.get("overlay") or {}
    t = float(ov.get("t", sim.data.time))
    phase = force_phase or phase_from_overlay(ov, t_impact)
    ctl["last_phase"] = phase
    if rec is not None:
        rec.maybe(phase, label)
    if vwr is None:
        return
    rows = [
        ("MODE", label),
        ("PHASE", phase),
        ("t", f"{t:.3f}"),
        ("nL/nR", f"{ov.get('nL', '-')} / {ov.get('nR', '-')}"),
    ]
    if ov.get("event"):
        rows.append(("event", str(ov["event"])))
    _viewer_overlay(vwr, rows)
    vwr.sync()
    sp = float(ctl.get("speed", 0.45))
    if sp > 0:
        time.sleep(max(0.0, float(sim.model.opt.timestep) / sp))


def run_shared_prefix(sim, gains, t_early: float, rec=None, vwr=None, speed=0.45):
    ctl = {
        "make_snap": make_snap,
        "capture_at": {"GO": t_early},
        "snaps": {},
        "abort": False,
        "speed": speed,
        "overlay": {},
    }
    t_impact_box = [None]

    def sync():
        ov = ctl.get("overlay") or {}
        if t_impact_box[0] is None and str(ov.get("phase", "")).upper() in (
            "FREE FLIGHT",
            "LIFT",
            "HIGH HOLD",
        ):
            pass
        tnow = float(sim.data.time)
        force = "IMPACT" if tnow >= 2.80 else None
        overlay_sync(sim, ctl, rec, vwr, "PREFIX", t_impact_box[0], force_phase=force)
        if "GO" in ctl.get("snaps", {}):
            ctl["abort"] = True

    ctl["sync"] = sync
    if vwr is not None:
        ctl["viewer"] = vwr
    ep = run_episode(sim, gains, V_HIT, SITE, ctl=ctl, label="PREFIX")
    t_impact_box[0] = ep.get("t_impact")
    snap = ctl["snaps"].get("GO")
    if snap is None:
        raise RuntimeError(f"failed to capture branch snap at t={t_early}")
    return ep, snap, make_snap(sim)


def log_row(sim, tau, Rh0, extra=None):
    m = measure(sim, tau, Rh0)
    o = physical_pack(sim)
    m["p_obj"] = np.array(sim.data.xpos[sim.ids.object_body], float).copy()
    m["p_hand"] = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    m["R_hand"] = np.array(sim.data.xmat[sim.ids.hand_body], float).copy()
    m["aperture"] = float(o.get("aperture", m.get("aperture", np.nan)))
    m["R_rel"] = np.asarray(o["R_rel"], float).copy()
    if extra:
        m.update(extra)
    return m


def continue_nominal(sim, gains, duration, log, Rh0, z_tgt, ctl=None, stop_on_table=True):
    """ZERO controller: lift until z_tgt, then v_cmd=0, tau=-18. Optional post-drop run."""
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
            if stop_on_table:
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


def run_zero(sim, gains, snap, rec=None, vwr=None, speed=0.45):
    restore_ballistic(sim, snap)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    z_tgt = float(snap.get("z_tgt", z_tgt_of(sim)))
    log = [log_row(sim, -18.0, Rh0, {"phase": "RECOVERY"})]
    start = branch_arrays(sim)
    ctl = {"speed": speed, "overlay": {}, "label": "ZERO"}

    def sync():
        overlay_sync(sim, ctl, rec, vwr, "ZERO", force_phase=ctl.get("force_phase", "RECOVERY"))

    ctl["sync"] = sync
    ctl["force_phase"] = "RECOVERY"
    t_table = continue_nominal(sim, gains, 8.0, log, Rh0, z_tgt, ctl, stop_on_table=False)
    if t_table is not None:
        ctl["force_phase"] = "FAILURE"
    else:
        ctl["force_phase"] = "HOLD"
        continue_nominal(sim, gains, HOLD_AFTER_S, log, Rh0, z_tgt, ctl, stop_on_table=True)
    return {"log": log, "t_table": t_table, "start": start, "end": log[-1]}


def run_slip(sim, gains, snap, rec=None, vwr=None, speed=0.45):
    restore_ballistic(sim, snap)
    R_nom = np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    freeze(sim)
    z_tgt = float(snap.get("z_tgt", z_tgt_of(sim)))
    log = [log_row(sim, -18.0, Rh0, {"phase": "RECOVERY"})]
    start = branch_arrays(sim)
    ctl = {"speed": speed, "overlay": {}, "label": "CONTROLLED_SLIP"}

    def sync():
        overlay_sync(
            sim, ctl, rec, vwr, "CONTROLLED SLIP", force_phase=ctl.get("force_phase", "RECOVERY")
        )

    ctl["sync"] = sync
    ctl["force_phase"] = "RECOVERY"
    rot = rotate_to(sim, gains, Rh0, SLIP_THETA, -18.0, log, ctl=ctl, label="ROTATE")
    dt = float(sim.model.opt.timestep)
    ret = None
    hold0 = None
    ctl["force_phase"] = "SLIP"
    t_table, _ = tick_log(
        sim, gains, 0.0, 0.0, 0.0, SLIP_TAU, Rh0, log, int(round(SLIP_S / dt)), ctl=ctl, label="SLIP"
    )
    ctl["force_phase"] = "HOLD"
    if t_table is None:
        t_table, _ = tick_log(
            sim,
            gains,
            0.0,
            0.0,
            0.0,
            -18.0,
            Rh0,
            log,
            int(round(BRAKE_S / dt)),
            ctl=ctl,
            label="BRAKE",
        )
    if t_table is None:
        ret = return_to_nominal(sim, gains, R_nom, Rh0, log, ctl=ctl)
        t_table = ret.get("t_table")
    if t_table is None:
        t_table = continue_zero(sim, gains, HOLD_AFTER_S, log, Rh0, z_tgt, ctl)
        hold0 = float(log[-1]["t"]) - HOLD_AFTER_S
    else:
        ctl["force_phase"] = "FAILURE"
        continue_zero(sim, gains, ZERO_AFTER_TABLE_S, log, Rh0, z_tgt, ctl)
    return {
        "log": log,
        "t_table": t_table,
        "start": start,
        "end": log[-1],
        "rotate": rot,
        "return": ret,
        "hold0": hold0,
    }


def run_airborne(sim, gains, snap, rec=None, vwr=None, speed=0.45, t_close=None, stills=None):
    restore_ballistic(sim, snap)
    R_nom = np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    z_tgt = float(snap.get("z_tgt", z_tgt_of(sim)))
    start = branch_arrays(sim)
    ctl = {"speed": speed, "overlay": {}, "label": "AIRBORNE"}
    phase_map = {
        "ROTATE": "ROTATE",
        "OPEN": "THROW",
        "BALLISTIC": "AIRBORNE",
        "CLOSE": "RECAPTURE",
        "HOLD": "SECURE",
        "HOLD_START": "SECURE",
        "HOLD_COMPLETE": "SECURE",
        "FIRST_BOTH_OFF": "AIRBORNE",
        "FIRST_RECONTACT": "RECAPTURE",
        "FIRST_BILATERAL": "RECAPTURE",
        "CLOSE_COMMAND": "RECAPTURE",
        "OPEN_COMMAND": "THROW",
        "RETURN": "RETURN UPRIGHT",
        "RETURN_SETTLE": "RETURN UPRIGHT",
    }
    saved_stills = set()

    def maybe_still(tag: str):
        if stills is None or rec is None or tag in saved_stills:
            return
        saved_stills.add(tag)
        rec.renderer.update_scene(rec.sim.data, rec.cam)
        stills[tag] = rec.renderer.render().copy()
        if tag in ("OPEN_COMMAND", "FIRST_BOTH_OFF", "CLOSE_COMMAND", "FIRST_BILATERAL"):
            for name, params in CAM_PROBE:
                cam = mjv_camera_from_preset(params)
                rec.renderer.update_scene(rec.sim.data, cam)
                stills[f"{tag}__{name}"] = rec.renderer.render().copy()
            rec.renderer.update_scene(rec.sim.data, rec.cam)

    def sync():
        ov = ctl.get("overlay") or {}
        raw = str(ov.get("phase") or ov.get("event") or "RECOVERY")
        force = phase_map.get(raw, ctl.get("force_phase", "RECOVERY"))
        overlay_sync(sim, ctl, rec, vwr, "AIRBORNE RECAPTURE", force_phase=force)
        ev = str(ov.get("event") or "")
        if ev in ("OPEN_COMMAND", "FIRST_BOTH_OFF", "CLOSE_COMMAND", "FIRST_BILATERAL", "HOLD_START"):
            maybe_still(ev)

    ctl["sync"] = sync

    def restore_fn():
        restore_ballistic(sim, snap)

    out = recatch_one(
        sim, gains, restore_fn, CLAIMED_OPEN_DEG, t_close, ctl=ctl, hold_s=CATCH_CONFIRM_S
    )
    out["start"] = start
    out["R_nominal"] = R_nom
    out["return"] = None
    out["resume"] = None
    out["complete_recovery"] = False
    out["catch_ok"] = bool(out.get("captured") and out.get("ok_hold") and "HOLD_LOST" not in (out.get("events") or {}))
    if not out["catch_ok"]:
        out["fail_stage"] = "catch"
        return out
    maybe_still("POST_CATCH")
    ctl["force_phase"] = "RETURN UPRIGHT"
    ret_log = out["log"]
    ret = return_to_nominal(sim, gains, R_nom, Rh0, ret_log, ctl=ctl)
    out["return"] = {
        "t_table": ret.get("t_table"),
        "arrived_des": ret.get("arrived_des"),
        "dt": ret.get("dt"),
        "end": ret.get("end"),
        "ori_log": ret.get("ori_log"),
    }
    maybe_still("RETURN")
    lost_return = ret.get("t_table") is not None
    end_m = ret.get("end_measure") or {}
    nL = int(end_m.get("nL", 0))
    nR = int(end_m.get("nR", 0))
    if lost_return or nL == 0 or nR == 0:
        out["fail_stage"] = "return"
        out["events"]["RETURN_LOST"] = float(sim.data.time)
        return out
    err_h = float((ret.get("end") or {}).get("err_hand_deg", 99.0))
    out["return_err_hand_deg"] = err_h
    out["return_ok"] = bool(ret.get("arrived_des") and err_h <= max(RETURN_HAND_STOP_DEG, 5.0))
    if not out["return_ok"]:
        out["fail_stage"] = "return_orientation"
        return out
    maybe_still("UPRIGHT")
    ctl["force_phase"] = "LIFT / HOLD"
    t_table = continue_zero(sim, gains, HOLD_AFTER_S, ret_log, Rh0, z_tgt, ctl)
    out["resume"] = {"t_table": t_table, "t_end": float(sim.data.time)}
    maybe_still("HOLD")
    end = measure(sim, -18.0, Rh0)
    persist = (
        t_table is None
        and int(end["nL"]) > 0
        and int(end["nR"]) > 0
        and float(np.linalg.norm(end["v_rel_h"])) < 0.08
        and float(end["obj_z"]) >= TABLE_DROP
    )
    out["resume_ok"] = bool(persist)
    out["end_err_hand_deg"] = float(so3_angle_deg(R_nom, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)))
    if not persist:
        out["fail_stage"] = "resume"
        return out
    out["complete_recovery"] = True
    out["fail_stage"] = None
    return out


def summarize(name: str, log, t_table, events=None, t0=None):
    events = events or {}
    rows = log
    post = [r for r in rows if t0 is None or float(r["t"]) >= float(t0) - 1e-9]
    rhx = [abs(float(r["rh"][0])) for r in post]
    both_off = []
    both_on = []
    for r in post:
        nl, nr = int(r["nL"]), int(r["nR"])
        if nl == 0 and nr == 0:
            both_off.append(float(r["t"]))
        if nl > 0 and nr > 0:
            both_on.append(float(r["t"]))
    first_both_off = events.get("FIRST_BOTH_OFF")
    if first_both_off is None and both_off:
        first_both_off = both_off[0]
    recapture = events.get("FIRST_BILATERAL") or events.get("FIRST_RECONTACT")
    hold_t = 0.0
    if events.get("HOLD_START") and events.get("HOLD_COMPLETE"):
        hold_t = float(events["HOLD_COMPLETE"]) - float(events["HOLD_START"])
    elif events.get("HOLD_START") and not events.get("HOLD_LOST"):
        hold_t = float(post[-1]["t"]) - float(events["HOLD_START"])
    elif name == "controlled_slip" and t_table is None and post:
        hold_t = HOLD_AFTER_S
        if int(post[-1]["nL"]) > 0 and int(post[-1]["nR"]) > 0:
            t_end = float(post[-1]["t"])
            t_s = t_end
            for r in reversed(post):
                if int(r["nL"]) > 0 and int(r["nR"]) > 0 and float(np.linalg.norm(r.get("v_rel_h", np.zeros(3)))) < 0.02:
                    t_s = float(r["t"])
                else:
                    break
            hold_t = min(float(t_end - t_s), float(HOLD_AFTER_S + BRAKE_S + 4.0))
    contact_loss = first_both_off is not None
    if t_table is not None:
        outcome = "DROP"
        mode = "loss_to_table"
    elif name == "airborne_recatch" and (events.get("ok_hold") or events.get("HOLD_COMPLETE")):
        outcome = "HOLD"
        mode = "airborne_recapture"
    elif name == "airborne_recatch" and recapture:
        outcome = "HOLD" if hold_t >= 1.5 else "UNSTABLE"
        mode = "airborne_recapture"
    elif t_table is None and int(post[-1]["nL"]) > 0 and int(post[-1]["nR"]) > 0:
        outcome = "HOLD"
        mode = "contact_preserving_or_slip" if name != "zero" else "still_holding"
    else:
        outcome = "UNSTABLE"
        mode = "other"
    if name == "zero":
        mode = "no_recovery"
    if name == "controlled_slip":
        mode = "controlled_gravitational_slip"
    return {
        "branch": name,
        "outcome": outcome,
        "contact_mode": mode,
        "max_abs_rhx_mm": None if not rhx else 1e3 * float(max(rhx)),
        "end_rhx_mm": None if not post else 1e3 * float(post[-1]["rh"][0]),
        "complete_contact_loss": bool(contact_loss),
        "first_both_off": first_both_off,
        "recapture": recapture,
        "stable_hold_s": float(hold_t),
        "t_table": t_table,
        "t_end": None if not post else float(post[-1]["t"]),
        "end_nL": None if not post else int(post[-1]["nL"]),
        "end_nR": None if not post else int(post[-1]["nR"]),
        "end_obj_z": None if not post else float(post[-1]["obj_z"]),
        "end_vrel": None
        if not post
        else float(np.linalg.norm(post[-1].get("v_rel_h", post[-1].get("v_rel", [0, 0, 0])))),
        "end_ctrl7": None if not post else post[-1].get("ctrl7"),
    }


def dump_traj(path: Path, log):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([r["t"] for r in log], float)
    rh = np.array([r["rh"] for r in log], float)
    nL = np.array([r["nL"] for r in log], int)
    nR = np.array([r["nR"] for r in log], int)
    obj_z = np.array([r["obj_z"] for r in log], float)

    def _vrel(r):
        if "v_rel_h" in r:
            return float(np.linalg.norm(r["v_rel_h"]))
        v = r.get("v_rel", np.nan)
        if np.isscalar(v):
            return float(v)
        return float(np.linalg.norm(v))

    vrel = np.array([_vrel(r) for r in log], float)
    ap = np.array([r.get("aperture", np.nan) for r in log], float)
    ctrl7 = np.array([r.get("ctrl7", np.nan) for r in log], float)
    np.savez_compressed(path, t=t, rh=rh, nL=nL, nR=nR, obj_z=obj_z, vrel=vrel, aperture=ap, ctrl7=ctrl7)


def pad_frames(seqs: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    n = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        if not s:
            out.append(s)
            continue
        last = s[-1]
        out.append(s + [last] * (n - len(s)))
    return out


def hstack_frames(a, b, c):
    return [np.concatenate([x, y, z], axis=1) for x, y, z in zip(a, b, c)]


def write_report(meta: dict) -> None:
    a = []
    ap = a.append
    ap("# Ballistic-impact recovery comparison")
    ap("")
    ap("Privileged **authority constructions**, not learned-policy results.")
    ap("No SAC was trained. Main training bound `RECOVERY4D_W_HY_MAX = 3 rad/s` was not changed.")
    ap("")
    ap("## Benchmark definition")
    ap("")
    ap("- Scene: `assets/scene_ballistic_impact.xml` (noslip=1)")
    ap(f"- Cylinder + ball mass `{BALL_M}` kg, CENTER site, `v_hit = {V_HIT}` m/s")
    ap("- Nominal grasp/lift, then genuine ballistic collision")
    ap(f"- Branch point: EARLY drift `t = {meta['t_early']:.6f}` s (after impact, before drop)")
    ap("- Shared prefix recorded from `t = 0` (before disturbance)")
    ap("")
    ap("## Branch-point equivalence")
    ap("")
    ap("All three recovery branches restore the **same live prefix snapshot**.")
    d = meta["branch_diffs"]
    ap("")
    ap(f"- SHA256 of live branch state: `{meta['hash_live']}`")
    ap(f"- SHA256 after ZERO restore: `{meta['hash_zero']}`")
    ap(f"- SHA256 after SLIP restore: `{meta['hash_slip']}`")
    ap(f"- SHA256 after AIRBORNE restore: `{meta['hash_air']}`")
    ap(f"- max |Δ| ZERO vs SLIP restore: `{d['zero_slip']['max']:.3e}`")
    ap(f"- max |Δ| ZERO vs AIRBORNE restore: `{d['zero_air']['max']:.3e}`")
    ap(f"- max |Δ| live vs saved `snap_early.pkl`: `{meta['diff_pkl']['max']:.3e}`")
    ap("")
    if meta["matched"]:
        ap("**PASS:** branches are numerically identical at the branch point before recovery diverges.")
    else:
        ap("**FAIL:** branch states differ. Videos must not be used as a matched comparison.")
    ap("")
    ap("## ZERO semantics")
    ap("")
    ap("After the matched impact, **no recovery primitive** is applied.")
    ap("")
    ap("- `p_des`: continues the nominal lift until `z_tgt`, then holds that height")
    ap("- `r_des`: unchanged (nominal orientation from the lift controller)")
    ap("- `v_cmd` / `w_cmd`: nominal lift velocity until `z_tgt`, then **zero** (`tick_vw` hold)")
    ap("- grip: `tau = -18` (`ctrl[7]` secure) throughout")
    ap("- **Not** the old stale-`p_des` STOP freeze")
    ap("")
    ap("Implementation: `continue_nominal` in `training/demo_ballistic_recovery_comparison.py`")
    ap("(same lift-then-hold semantics as `continue_zero`; does not abort the video at first table contact).")
    ap("")
    ap("## CONTROLLED_SLIP construction")
    ap("")
    ap("**PRIVILEGED AUTHORITY CONSTRUCTION** (timing/angle from the validated CENTER-6 proof).")
    ap("")
    ap(f"1. `freeze()` (copy current `p_des`/`r_des`, zero `v_cmd`/`w_cmd`)")
    ap(f"2. Secure rotate `+{SLIP_THETA:.0f}°` at legal `omega_y = {RECOVERY4D_W_HY_MAX}` rad/s, `tau=-18`")
    ap(f"3. Weak-grip slip `tau={SLIP_TAU:.0f}` for `{SLIP_S:.2f}` s (`v_x=v_z=omega=0`)")
    ap(f"4. Brake `tau=-18` for `{BRAKE_S:.2f}` s")
    ap("5. `return_to_nominal` (legal `omega_y` ≤ 3 rad/s)")
    ap("6. Hold via `continue_zero`")
    ap("")
    ap(f"Baseline duration `{SLIP_S}` s is the retained sufficiency result (also in `{DURS}`).")
    ap("Not an observable policy.")
    ap("")
    ap("## AIRBORNE_RECAPTURE construction")
    ap("")
    ap("**PRIVILEGED AUTHORITY CONSTRUCTION** + **DIAGNOSTIC SIMULATION AUTHORITY** (`omega_y=+4`).")
    ap("")
    ap("This is **not** the training action bound and **not** a hardware-safe Panda velocity.")
    ap("`tick_omega_override` bypasses `RECOVERY4D_W_HY_MAX=3` for this diagnostic only.")
    ap("")
    ap("Retained construction (user-visually-verified candidate):")
    ap("")
    ap("- physically generated EARLY impact-offset state")
    ap("- diagnostic `omega_y = +4` rad/s")
    ap(f"- OPEN command at `{CLAIMED_OPEN_DEG:.0f}°` (`tau = +2`)")
    ap("- genuine `FIRST_BOTH_OFF` and nonzero contact-free interval")
    ap(f"- privileged CLOSE at `t_close = {meta.get('t_close')}` (`tau = -18`)")
    ap("- bilateral recapture + short secure confirmation (`tau=-18`)")
    ap("- **then** SO(3) `return_to_nominal` (legal `|omega_y|<=3`, latch `r_des=R_nominal`)")
    ap("- **then** resume nominal lift/hold (`continue_zero`)")
    ap("")
    ap("Post-catch RETURN is new work; catch was previously stopped at the rotated pose.")
    ap("")
    ev = meta.get("air_events") or {}
    ap("This replay events:")
    for k in (
        "OPEN_COMMAND",
        "FIRST_BOTH_OFF",
        "CLOSE_COMMAND",
        "FIRST_RECONTACT",
        "FIRST_BILATERAL",
        "HOLD_START",
        "HOLD_COMPLETE",
        "APEX",
        "RETURN_LOST",
    ):
        ap(f"- `{k}`: {ev.get(k)}")
    ap("")
    cr = meta.get("air_complete") or {}
    ap("### Completeness (logged separately; not one SUCCESS bit)")
    ap("")
    ap(f"- genuine FIRST_BOTH_OFF: `{cr.get('both_off')}`")
    ap(f"- new bilateral contact: `{cr.get('bilateral')}`")
    ap(f"- recaptured grasp stable: `{cr.get('catch_ok')}`")
    ap(f"- SO(3) return to R_nominal: `{cr.get('return_ok')}`  (hand error `{cr.get('return_err_hand_deg')}` deg)")
    ap(f"- secure resumed lift/hold: `{cr.get('resume_ok')}`")
    ap(f"- **complete airborne recovery**: `{cr.get('complete_recovery')}`")
    ap(f"- fail_stage: `{cr.get('fail_stage')}`")
    ap(f"- end orientation error: `{cr.get('end_err_hand_deg')}` deg")
    ap("")
    ap("## Raw outcome table")
    ap("")
    ap("| branch | outcome | contact mode | max_abs_rhx_mm | contact_loss | recapture | hold_s | t_table |")
    ap("|---|---|---|---|---|---|---|---|")
    for s in meta["summaries"]:
        mx = s["max_abs_rhx_mm"]
        mx_s = "—" if mx is None else f"{mx:.2f}"
        ap(
            "| {branch} | {outcome} | {contact_mode} | {mx} | {loss} | {rec} | {hold:.3f} | {table} |".format(
                branch=s["branch"],
                outcome=s["outcome"],
                contact_mode=s["contact_mode"],
                mx=mx_s,
                loss=s["complete_contact_loss"],
                rec=s["recapture"],
                hold=s["stable_hold_s"],
                table=s["t_table"],
            )
        )
    ap("")
    ap("See `results/comparison/ballistic_recovery/raw/` for npz trajectories.")
    ap("")
    ap("## Videos")
    ap("")
    for p in meta.get("videos") or []:
        ap(f"- `{p}`")
    ap("")
    ap("## Limitations")
    ap("")
    ap("- Controlled slip and airborne recatch are **authority constructions**, not learned policies.")
    ap("- Airborne recatch uses diagnostic `omega_y=4` and privileged CLOSE timing.")
    ap("- Not hardware validated.")
    ap("- Unified observable SAC has **not** been trained on this benchmark.")
    ap("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(a) + "\n", encoding="utf-8")


def interactive_one(mode: str, speed: float):
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    sim, _ = make_sim()
    t_early = t_early_default()
    t_close = claimed_t_close()
    kh = {}

    def on_key(kc):
        fn = kh.get("fn")
        if fn:
            fn(kc)

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        apply_camera_preset(vwr, CAM)
        kh["fn"] = make_reset_cam_callback({"viewer": vwr, "_cam_preset": CAM, "_cam_inited": True})
        t_end = time.perf_counter() + 1.5
        while time.perf_counter() < t_end:
            vwr.sync()
            time.sleep(0.03)
        ep, snap, _ = run_shared_prefix(sim, gains, t_early, rec=None, vwr=vwr, speed=speed)
        _ = ep
        if mode == "zero":
            run_zero(sim, gains, snap, vwr=vwr, speed=speed)
        elif mode == "controlled_slip":
            run_slip(sim, gains, snap, vwr=vwr, speed=speed)
        else:
            run_airborne(sim, gains, snap, vwr=vwr, speed=speed, t_close=t_close)
        t_end = time.perf_counter() + 2.0
        while time.perf_counter() < t_end:
            vwr.sync()
            time.sleep(0.03)


def run_headless(write_videos: bool, stills_only: bool = False) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    t_early = t_early_default()
    t_close = claimed_t_close()
    sim, _ = make_sim()
    renderer = None
    rec = None
    if write_videos or stills_only:
        VID.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(sim.model, RENDER_H, RENDER_W)
        rec = FrameRec(sim, renderer, mjv_camera_from_preset(CAM))
    print("prefix CENTER-6 until EARLY", t_early, flush=True)
    ep, snap, live = run_shared_prefix(sim, gains, t_early, rec=rec, vwr=None)
    prefix_frames = list(rec.frames) if rec else []
    prefix_t_last = rec.last if rec else 0.0
    live_arr = branch_arrays(sim)
    hash_live = hash_arrays(live_arr)
    pkl_diff = {"max": float("nan")}
    if EARLY_PKL.is_file():
        with EARLY_PKL.open("rb") as f:
            pkl = pickle.load(f)
        tmp, _ = make_sim()
        restore_ballistic(tmp, pkl)
        pkl_diff = max_abs_diff(live_arr, branch_arrays(tmp))

    def restore_hash(tag):
        s, _ = make_sim()
        restore_ballistic(s, snap)
        a = branch_arrays(s)
        return a, hash_arrays(a)

    a0, h0 = restore_hash("zero")
    a1, h1 = restore_hash("slip")
    a2, h2 = restore_hash("air")
    diffs = {
        "zero_slip": max_abs_diff(a0, a1),
        "zero_air": max_abs_diff(a0, a2),
    }
    matched = diffs["zero_slip"]["max"] < 1e-12 and diffs["zero_air"]["max"] < 1e-12
    print("matched", matched, "max", diffs["zero_slip"]["max"], diffs["zero_air"]["max"], flush=True)
    if not matched:
        raise RuntimeError("branch-point states differ; refusing to write a comparison video")

    recz = recs = reca = None
    if write_videos or stills_only:
        recz = FrameRec(sim, renderer, mjv_camera_from_preset(CAM))
        recz.last = prefix_t_last
    if not stills_only:
        print("ZERO", flush=True)
        z = run_zero(sim, gains, snap, rec=recz)
        dump_traj(RAW / "zero.npz", z["log"])
    else:
        z = {"log": [{"t": t_early, "rh": [0, 0, 0], "nL": 1, "nR": 1, "obj_z": 0.5, "v_rel_h": [0, 0, 0]}], "t_table": None}

    if write_videos or stills_only:
        recs = FrameRec(sim, renderer, mjv_camera_from_preset(CAM))
        recs.last = prefix_t_last
    if not stills_only:
        print("CONTROLLED_SLIP", flush=True)
        s = run_slip(sim, gains, snap, rec=recs)
        dump_traj(RAW / "controlled_slip.npz", s["log"])
    else:
        s = {"log": z["log"], "t_table": None}

    if write_videos or stills_only:
        reca = FrameRec(sim, renderer, mjv_camera_from_preset(CAM))
        reca.last = prefix_t_last
    print("AIRBORNE t_close", t_close, flush=True)
    stills = {}
    air = run_airborne(sim, gains, snap, rec=reca, t_close=t_close, stills=stills)
    if stills:
        sp = OUT / "cam_stills"
        sp.mkdir(parents=True, exist_ok=True)
        import matplotlib.image as mpimg

        for k, fr in stills.items():
            mpimg.imsave(str(sp / f"{k}.png"), fr)
            print("still", k, flush=True)
    dump_traj(RAW / "airborne_recatch.npz", air["log"])
    dump_json(RAW / "airborne_events.json", json.loads(json.dumps(air["events"], default=str)))
    dump_json(
        RAW / "airborne_return.json",
        {
            "catch_ok": air.get("catch_ok"),
            "return_ok": air.get("return_ok"),
            "resume_ok": air.get("resume_ok"),
            "complete_recovery": air.get("complete_recovery"),
            "fail_stage": air.get("fail_stage"),
            "return_err_hand_deg": air.get("return_err_hand_deg"),
            "end_err_hand_deg": air.get("end_err_hand_deg"),
            "return": air.get("return"),
            "resume": air.get("resume"),
        },
    )

    t0 = float(live_arr["t"][0])
    sm_z = summarize("zero", z["log"], z["t_table"], t0=t0)
    sm_s = summarize("controlled_slip", s["log"], s["t_table"], t0=t0)
    air_ev = dict(air["events"])
    air_ev["ok_hold"] = air.get("ok_hold")
    sm_a = summarize("airborne_recatch", air["log"], (air.get("resume") or {}).get("t_table"), events=air_ev, t0=t0)
    if air.get("complete_recovery"):
        sm_a["outcome"] = "COMPLETE"
        sm_a["contact_mode"] = "airborne_recapture_return_hold"
    elif air.get("catch_ok") and not air.get("return_ok"):
        sm_a["outcome"] = "CATCH_RETURN_FAIL"
    dump_json(RAW / "summaries.json", [sm_z, sm_s, sm_a])

    videos = []
    if write_videos:
        fz = prefix_frames + recz.frames
        fs = prefix_frames + recs.frames
        fa = prefix_frames + reca.frames
        p_z = VID / "ballistic_zero.mp4"
        p_s = VID / "ballistic_controlled_slip.mp4"
        p_a = VID / "ballistic_airborne_recatch.mp4"
        p_c = VID / "ballistic_recovery_comparison.mp4"
        for frames, dest in ((fz, p_z), (fs, p_s), (fa, p_a)):
            print("encode", dest, "n", len(frames), flush=True)
            write_frames(frames, dest)
            videos.append(str(dest.relative_to(ROOT)).replace("\\", "/"))
        pz, ps, pa = pad_frames([fz, fs, fa])
        combo = hstack_frames(pz, ps, pa)
        print("encode comparison", len(combo), flush=True)
        write_frames(combo, p_c)
        videos.append(str(p_c.relative_to(ROOT)).replace("\\", "/"))
        # drop staging frames
        for tmp in VID.glob("_frames_*"):
            if tmp.is_dir():
                for p in tmp.glob("*"):
                    p.unlink()
                tmp.rmdir()

    meta = {
        "t_early": t_early,
        "t_impact": ep.get("t_impact"),
        "t_close": t_close,
        "open_deg": CLAIMED_OPEN_DEG,
        "ball_m": BALL_M,
        "v_hit": V_HIT,
        "site": SITE,
        "omega_train": RECOVERY4D_W_HY_MAX,
        "hash_live": hash_live,
        "hash_zero": h0,
        "hash_slip": h1,
        "hash_air": h2,
        "branch_diffs": diffs,
        "diff_pkl": pkl_diff,
        "matched": matched,
        "summaries": [sm_z, sm_s, sm_a],
        "air_events": air["events"],
        "air_ok_hold": air.get("ok_hold"),
        "air_captured": air.get("captured"),
        "air_complete": {
            "both_off": (air.get("events") or {}).get("FIRST_BOTH_OFF"),
            "bilateral": (air.get("events") or {}).get("FIRST_BILATERAL"),
            "catch_ok": air.get("catch_ok"),
            "return_ok": air.get("return_ok"),
            "resume_ok": air.get("resume_ok"),
            "complete_recovery": air.get("complete_recovery"),
            "fail_stage": air.get("fail_stage"),
            "return_err_hand_deg": air.get("return_err_hand_deg"),
            "end_err_hand_deg": air.get("end_err_hand_deg"),
        },
        "camera": CAM,
        "videos": videos,
        "zero_semantics": {
            "p_des": "nominal lift until z_tgt, then hold",
            "r_des": "unchanged nominal",
            "v_cmd": "nominal then 0",
            "grip": "tau=-18",
        },
    }
    dump_json(RAW / "comparison_meta.json", meta)
    write_report(meta)
    if renderer is not None:
        renderer.close()
    print("report", REPORT, flush=True)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="all", choices=["zero", "controlled_slip", "airborne_recatch", "all"])
    p.add_argument("--write-videos", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--speed", type=float, default=0.45)
    p.add_argument("--stills-only", action="store_true")
    args = p.parse_args()
    if args.write_videos or args.headless or args.stills_only:
        run_headless(write_videos=bool(args.write_videos), stills_only=bool(args.stills_only))
        return
    if args.mode == "all":
        for m in MODES:
            print("interactive", m, flush=True)
            interactive_one(m, args.speed)
        return
    interactive_one(args.mode, args.speed)


if __name__ == "__main__":
    main()
