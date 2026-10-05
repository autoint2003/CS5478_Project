"""Complete-task ZERO vs HEURISTIC comparison.

Nominal GraspFSM: APPROACH → DESCEND → CLOSE → LIFT, then coupled
s=2.0 disturbance during active lift (no controller reset).

HEURISTIC is the *manually constructed* gravitational-reposition
sequence (not RULE, not SAC). Does not retune recovery, physics,
disturbance, reward, or observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from training.demo_grav_reposition_recovery import (
    BRAKE_HOLD,
    LOG_DT,
    OMEGA,
    OMEGA_RET,
    PAIR_MU,
    TAU_SEC,
    THETA,
    V_LIFT,
    apply_sfail,
    dropped,
    freeze,
    live_opt,
    make_parent_sim,
    restore,
    useful_mm,
)
from training.demo_teleport_recovery_state import (
    apply_coupled_pose,
    coupled_pose_from_ref,
    load_or_measure_ref,
    signed_rot_about_y_deg,
)
from training.impact_visualization_utils import add_capsule, mjv_camera_from_preset
from training.grav_reposition_v2_viz import (
    PRE_VIEW_PAUSE_S,
    apply_camera_preset,
    make_reset_cam_callback,
    object_pose,
)
from training.replay_core import tick_vw

SFAIL_PKL = ROOT / "results" / "diagnostics" / "grav_reposition_recovery_construction" / "raw" / "sfail.pkl"
PARENT_PKL = ROOT / "results" / "diagnostics" / "grav_reposition_recovery_construction" / "raw" / "parent.pkl"
OUT = ROOT / "results" / "videos" / "gravity_recovery_comparison"
RAW = OUT / "raw"
ARCH = OUT / "superseded_airborne_opening"
SFAIL_LIFT_PKL = RAW / "sfail_lift.pkl"
PRE_DISTURB_PKL = RAW / "pre_disturbance_lift.pkl"

HOLD_S = 12.0
AFTER_DROP_S = 2.0
S_COUPLED = 2.0
TAU_WEAK = -4.0
SLIP_TARGET_MM = 1.0
SLIP_MAX_S = 1.5
OBS_S = 0.100
# Task-state disturbance trigger (not tuned for heuristic success).
HAND_RISE_M = 0.05
DISTURB_FREEZE_S = 0.40
VIDEO_FPS = 25
SPEED_TASK = 0.70
SPEED_LIFT = 0.60
SPEED_DISTURB = 0.35
SPEED_RECOVERY = 0.40
SPEED_HOLD = 1.50
SPEED_DROP = 0.50
RENDER_W, RENDER_H = 640, 480

# Wide fixed view of the full pick-and-lift workspace (not a grasp close-up).
CAM_DISTANCE = 1.65
CAM_AZIMUTH = 142.0
CAM_ELEVATION = -22.0


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


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")


def sha_bytes(blob: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(blob).tobytes()).hexdigest()


def ctrl_pack(sim) -> dict:
    return {
        "phase": str(sim.fsm.phase),
        "t_phase": float(sim.fsm.t_phase),
        "t_stable": float(sim.fsm.t_stable),
        "t_held": float(sim.fsm.t_held),
        "lift_started": bool(sim.fsm.lift_started),
        "success": bool(sim.fsm.success),
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "r_des": np.asarray(sim.fsm.r_des, float).copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).copy(),
        "v_des": np.asarray(sim.fsm.v_des, float).copy(),
        "w_des": np.asarray(sim.fsm.w_des, float).copy(),
        "grasp_z": float(sim.fsm.grasp_z),
        "fg_cmd": float(sim.fsm.fg_cmd),
        "captured": bool(sim.captured),
    }


def ic_pack(sim) -> dict:
    qpos = np.asarray(sim.data.qpos, float)
    qvel = np.asarray(sim.data.qvel, float)
    p_des = np.asarray(sim.fsm.p_des, float)
    r_des = np.asarray(sim.fsm.r_des, float).reshape(-1)
    v_cmd = np.asarray(sim.fsm.v_cmd, float)
    ctrl = np.asarray(sim.data.ctrl, float)
    blob = np.concatenate([qpos, qvel, p_des, r_des, v_cmd, ctrl])
    c = ctrl_pack(sim)
    return {
        "sha256": sha_bytes(blob),
        "time": float(sim.data.time),
        "qpos": qpos.tolist(),
        "qvel": qvel.tolist(),
        "p_des": p_des.tolist(),
        "r_des": np.asarray(sim.fsm.r_des, float).reshape(3, 3).tolist(),
        "v_cmd": v_cmd.tolist(),
        "ctrl": ctrl.tolist(),
        "phase": c["phase"],
        "t_phase": c["t_phase"],
        "lift_started": c["lift_started"],
        "noslip_iterations": int(sim.model.opt.noslip_iterations),
        "live": live_opt(sim),
    }


def md(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def cam_params(sim) -> dict:
    po, _ = object_pose(sim)
    lookat = np.array([float(po[0]), float(po[1]), 0.52], float)
    return {
        "lookat": lookat,
        "distance": CAM_DISTANCE,
        "azimuth": CAM_AZIMUTH,
        "elevation": CAM_ELEVATION,
    }


def overlay_lines(label: str, phase: str, sim, tau: float) -> list[tuple[str, str]]:
    live = float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float(tau)
    return [
        (label, ""),
        ("phase", str(phase)),
        ("t", f"{float(sim.data.time):.2f} s"),
        ("tau", f"{live:.0f}"),
    ]


def viewer_overlay(viewer, rows) -> None:
    if viewer is None or not hasattr(viewer, "add_overlay"):
        return
    pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
    for k, v in rows:
        viewer.add_overlay(pos, k, v)


def burn_overlay(img: np.ndarray, rows) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle((6, 6, 280, 18 + 16 * len(rows)), fill=(0, 0, 0, 140))
    y = 10
    for k, v in rows:
        txt = k if not v else f"{k}: {v}"
        dr.text((12, y), txt, fill=(255, 255, 255, 255))
        y += 16
    return np.asarray(im)


def traj_row(sim, overlay_phase: str, tau: float) -> dict:
    o = physical_pack(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    return {
        "t": float(sim.data.time),
        "overlay": str(overlay_phase),
        "fsm_phase": str(sim.fsm.phase),
        "p_hand": np.array(sim.data.xpos[sim.ids.hand_body], float).copy(),
        "p_obj": np.array(sim.data.xpos[sim.ids.object_body], float).copy(),
        "rh": np.asarray(o["rh"], float).copy(),
        "obj_R": Ro.copy(),
        "cyl_h": (Rh.T @ Ro[:, 2]).copy(),
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "tau": float(tau),
        "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float(tau),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "obj_z": float(o["obj_z"]),
        "noslip": int(sim.model.opt.noslip_iterations),
        "lift_started": bool(sim.fsm.lift_started),
        "t_phase": float(sim.fsm.t_phase),
    }


def save_traj_npz(path: Path, rows: list, extra: dict | None = None) -> None:
    if not rows:
        return
    kw = dict(
        t=np.array([r["t"] for r in rows]),
        overlay=np.array([str(r["overlay"]) for r in rows]),
        fsm_phase=np.array([str(r["fsm_phase"]) for r in rows]),
        p_hand=np.stack([r["p_hand"] for r in rows]),
        p_obj=np.stack([r["p_obj"] for r in rows]),
        rh=np.stack([r["rh"] for r in rows]),
        cyl_h=np.stack([r["cyl_h"] for r in rows]),
        p_des=np.stack([r["p_des"] for r in rows]),
        v_cmd=np.stack([r["v_cmd"] for r in rows]),
        tau=np.array([r["tau"] for r in rows]),
        nL=np.array([r["nL"] for r in rows]),
        nR=np.array([r["nR"] for r in rows]),
        obj_z=np.array([r["obj_z"] for r in rows]),
        noslip=np.array([r["noslip"] for r in rows]),
        lift_started=np.array([int(r["lift_started"]) for r in rows]),
    )
    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            kw[k] = np.asarray(v)
    np.savez_compressed(path, **kw)


class Presenter:
    def __init__(self, sim, *, label: str, interactive: bool, record: bool, viz=None):
        self.sim = sim
        self.label = label
        self.interactive = interactive
        self.record = record
        self.viz = viz or {}
        self.speed = SPEED_TASK
        self.phase = "APPROACH"
        self.tau = TAU_SEC
        self.ghost_cyl = None
        self.frames: list[np.ndarray] = []
        self.events = []
        self.renderer = None
        self.mjv_cam = None
        self._next_wall = None
        self._next_render_t = None
        self.gate_visible = True
        if record:
            self.renderer = mujoco.Renderer(sim.model, RENDER_H, RENDER_W)

    def set_cam(self) -> None:
        params = cam_params(self.sim)
        if self.interactive and self.viz.get("viewer") is not None:
            if not self.viz.get("_cam_inited"):
                apply_camera_preset(self.viz["viewer"], params)
                self.viz["_cam_preset"] = params
                self.viz["_cam_inited"] = True
        if self.record:
            self.mjv_cam = mjv_camera_from_preset(params)

    def _rows(self) -> list:
        return overlay_lines(self.label, self.phase, self.sim, self.tau)

    def _ghost(self, scn) -> None:
        po, ax = self.ghost_cyl
        ax = np.asarray(ax, float)
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        add_capsule(scn, po - 0.032 * ax, po + 0.032 * ax, [0.95, 0.95, 0.92, 0.40], 0.00075)

    def _draw_extras(self, scn, *, overlay_scene: bool) -> None:
        if scn is None:
            return
        if overlay_scene:
            scn.ngeom = 0
            if self.ghost_cyl is not None:
                self._ghost(scn)
            return
        if self.ghost_cyl is not None:
            self._ghost(scn)

    def show(self) -> None:
        if not self.gate_visible:
            return
        rows = self._rows()
        viewer = self.viz.get("viewer") if self.interactive else None
        if viewer is not None:
            scn = getattr(viewer, "user_scn", None)
            self._draw_extras(scn, overlay_scene=True)
            viewer_overlay(viewer, rows)
            if hasattr(viewer, "sync"):
                viewer.sync()
            if self.speed > 1e-6:
                dt = float(self.sim.model.opt.timestep)
                now = time.perf_counter()
                target = self._next_wall
                if target is None:
                    self._next_wall = now + dt / self.speed
                else:
                    sl = target - now
                    if sl > 0:
                        time.sleep(min(sl, 0.05))
                    self._next_wall = max(target, time.perf_counter()) + dt / self.speed
        if self.record:
            t = float(self.sim.data.time)
            interval = (self.speed / VIDEO_FPS) if self.speed > 1e-9 else 1e9
            if self._next_render_t is None or t + 1e-12 >= self._next_render_t:
                self._grab(rows)
                self._next_render_t = (t if self._next_render_t is None else self._next_render_t) + interval

    def _grab(self, rows) -> None:
        self.renderer.update_scene(self.sim.data, camera=self.mjv_cam)
        self._draw_extras(self.renderer.scene, overlay_scene=False)
        img = self.renderer.render()
        self.frames.append(burn_overlay(img, rows))

    def pause_wall(self, seconds: float) -> None:
        rows = self._rows()
        if self.interactive and self.viz.get("viewer") is not None:
            t_end = time.perf_counter() + float(seconds)
            self._next_wall = None
            while time.perf_counter() < t_end:
                self.speed = 0.0
                self.show()
                time.sleep(0.03)
            self._next_wall = None
        if self.record:
            n = max(int(round(float(seconds) * VIDEO_FPS)), 1)
            self.renderer.update_scene(self.sim.data, camera=self.mjv_cam)
            self._draw_extras(self.renderer.scene, overlay_scene=False)
            img = burn_overlay(self.renderer.render(), rows)
            for _ in range(n):
                self.frames.append(img.copy())

    def note(self, name: str, **kw) -> None:
        o = physical_pack(self.sim)
        rec = {
            "event": name,
            "t": float(self.sim.data.time),
            "phase": self.phase,
            "tau": self.tau,
            "nL": int(o["nL"]),
            "nR": int(o["nR"]),
            "obj_z": float(o["obj_z"]),
        }
        rec.update(kw)
        self.events.append(rec)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def hand_above_cylinder(sim) -> bool:
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    return float(np.linalg.norm(ph[:2] - po[:2])) < 0.08 and float(ph[2]) > float(po[2]) + 0.10


def maybe_show(sim, pres: Presenter, i: int, dt: float) -> None:
    if pres is None:
        return
    if not pres.gate_visible:
        return
    if pres.record or pres.interactive:
        step = max(int(round((pres.speed / VIDEO_FPS) / dt)), 1)
        if i % step == 0:
            pres.show()
    elif i % max(int(round(LOG_DT / dt)), 1) == 0:
        pass


def apply_lift_disturbance(sim, ref, s: float) -> dict:
    """Object coupled teleport. Does not freeze or rewrite FSM / p_des / v_cmd."""
    before = ctrl_pack(sim)
    ic_before = ic_pack(sim)
    dp, Rd = coupled_pose_from_ref(ref, float(s), "coupled")
    applied = apply_coupled_pose(sim, dp, Rd)
    after = ctrl_pack(sim)
    ctrl_diffs = {
        "p_des": md(before["p_des"], after["p_des"]),
        "r_des": md(before["r_des"], after["r_des"]),
        "v_cmd": md(before["v_cmd"], after["v_cmd"]),
        "w_cmd": md(before["w_cmd"], after["w_cmd"]),
        "phase_same": before["phase"] == after["phase"],
        "t_phase": abs(before["t_phase"] - after["t_phase"]),
        "lift_started_same": before["lift_started"] == after["lift_started"],
        "t_held": abs(before["t_held"] - after["t_held"]),
    }
    untouched = (
        ctrl_diffs["p_des"] == 0.0
        and ctrl_diffs["r_des"] == 0.0
        and ctrl_diffs["v_cmd"] == 0.0
        and ctrl_diffs["w_cmd"] == 0.0
        and ctrl_diffs["phase_same"]
        and ctrl_diffs["t_phase"] == 0.0
        and ctrl_diffs["lift_started_same"]
    )
    return {
        "applied": applied,
        "dp_h": np.asarray(dp, float).tolist(),
        "ctrl_before": {k: _jsonable(v) for k, v in before.items()},
        "ctrl_after": {k: _jsonable(v) for k, v in after.items()},
        "ctrl_diffs": ctrl_diffs,
        "controller_untouched": bool(untouched),
        "ic_before": ic_before,
        "ic_after": ic_pack(sim),
        "s": float(s),
    }


def compare_to_snap(sim, snap) -> dict:
    diffs = {
        "qpos": md(sim.data.qpos, snap["qpos"]),
        "qvel": md(sim.data.qvel, snap["qvel"]),
        "p_des": md(sim.fsm.p_des, snap["p_des"]),
        "r_des": md(sim.fsm.r_des, snap["r_des"]),
        "v_cmd": md(sim.fsm.v_cmd, snap.get("v_cmd", np.zeros(3))),
        "ctrl": md(sim.data.ctrl, snap["ctrl"]),
        "time": abs(float(sim.data.time) - float(snap.get("time", 0.0))),
        "phase_same": str(sim.fsm.phase) == str(snap.get("phase", "")),
    }
    return {"diffs": diffs, "match": all(
        (diffs[k] < 1e-8 if k != "phase_same" and k != "time" else True)
        for k in ("qpos", "qvel", "p_des", "r_des", "v_cmd", "ctrl")
    ) and diffs["phase_same"]}


def audit_old_parent_embed(sim) -> dict:
    """Can canonical parent.pkl be reached as a live nominal-LIFT state?"""
    with PARENT_PKL.open("rb") as f:
        parent = pickle.load(f)
    with SFAIL_PKL.open("rb") as f:
        sfail = pickle.load(f)
    sim.reset(sim.mass, sim.friction, np.zeros(3))
    dt = float(sim.model.opt.timestep)
    n_max = int(round(12.0 / dt))
    best = {"qpos": 1e9, "t": None, "phase": None, "v_cmd": None, "obj_z": None, "lift_started": None}
    first_air = None
    for _ in range(n_max):
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
        if str(sim.fsm.phase) != "lift":
            continue
        d = md(sim.data.qpos, parent["qpos"])
        if d < best["qpos"]:
            best = {
                "qpos": d,
                "t": float(sim.data.time),
                "phase": str(sim.fsm.phase),
                "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
                "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
                "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
                "lift_started": bool(sim.fsm.lift_started),
                "hand": np.array(sim.data.xpos[sim.ids.hand_body], float).tolist(),
            }
        o = physical_pack(sim)
        if first_air is None and float(o["obj_z"]) >= Z_AIR:
            first_air = {
                "t": float(sim.data.time),
                "obj_z": float(o["obj_z"]),
                "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
                "qpos_vs_parent": md(sim.data.qpos, parent["qpos"]),
                "p_des_vs_parent": md(sim.fsm.p_des, parent["p_des"]),
                "v_cmd_vs_parent": md(sim.fsm.v_cmd, parent.get("v_cmd", np.zeros(3))),
            }
        if float(sim.data.time) >= 12.0:
            break
    parent_v = np.asarray(parent.get("v_cmd", np.zeros(3)), float)
    live_lift = first_air is not None and float(np.linalg.norm(first_air["v_cmd"])) > 0.02
    frozen_parent = float(np.linalg.norm(parent_v)) < 1e-6
    embeddable = bool(
        best["qpos"] < 1e-6
        and live_lift
        and not frozen_parent
        and md(best.get("v_cmd", [0, 0, 0]), parent_v) < 1e-6
    )
    # Old construction explicitly froze then applied apply_sfail (which snaps p_des).
    restore(sim, parent)
    freeze(sim)
    ref = load_or_measure_ref()
    apply_sfail(sim, parent, ref, S_COUPLED)
    vs_sfail = compare_to_snap(sim, sfail)
    out = {
        "verdict": "A" if embeddable else "B",
        "reason": (
            "canonical parent is a live nominal-LIFT state on the current FSM trajectory"
            if embeddable
            else "canonical parent is a frozen airborne hold (v_cmd≈0 after freeze), not a live in-lift state; "
            "do not splice it into a continuing LIFT"
        ),
        "parent_v_cmd": parent_v.tolist(),
        "parent_phase": parent.get("phase"),
        "parent_time": parent.get("time"),
        "parent_obj_z": parent.get("object_z"),
        "frozen_parent": frozen_parent,
        "best_replay_qpos_diff": best["qpos"],
        "best_replay": best,
        "first_airborne_live": first_air,
        "old_apply_sfail_vs_sfail_pkl": vs_sfail,
        "sfail_pkl_untouched": True,
    }
    dump_json(RAW / "parent_embed_audit.json", out)
    return out


def z_lift_target(sim) -> float:
    return float(sim.fsm.grasp_z) + float(sim.cfg["fsm"]["lift_offset_z"])


def run_nominal_to_disturbance(sim, pres: Presenter, ref, log: list) -> dict:
    dt = float(sim.model.opt.timestep)
    log_n = max(int(round(LOG_DT / dt)), 1)
    n_max = int(round(20.0 / dt))
    hand_z_lift0 = None
    t_lift0 = None
    t_close = None
    overlay_map = {"approach": "APPROACH", "descend": "DESCEND", "close": "CLOSE", "lift": "LIFT"}
    pres.tau = TAU_SEC
    pres.speed = SPEED_TASK
    pres.phase = overlay_map.get(str(sim.fsm.phase), str(sim.fsm.phase).upper())
    if pres.record:
        pres.gate_visible = False
    pres.note("task_start", fsm=str(sim.fsm.phase))
    for i in range(n_max):
        prev = str(sim.fsm.phase)
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
        if (not pres.gate_visible) and hand_above_cylinder(sim):
            pres.gate_visible = True
            pres._next_render_t = None
            pres.note("visible_start", fsm=str(sim.fsm.phase))
        ph = str(sim.fsm.phase)
        if ph == "close" and t_close is None:
            t_close = float(sim.data.time)
        if ph == "lift":
            pres.speed = SPEED_LIFT
            if hand_z_lift0 is None:
                hand_z_lift0 = float(sim.data.xpos[sim.ids.hand_body][2])
                t_lift0 = float(sim.data.time)
                pres.note("lift_start", hand_z=hand_z_lift0, p_des=np.asarray(sim.fsm.p_des, float).tolist())
        pres.phase = overlay_map.get(ph, ph.upper())
        o = physical_pack(sim)
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if i % log_n == 0:
            log.append(traj_row(sim, pres.phase, TAU_SEC))
        maybe_show(sim, pres, i, dt)
        if (
            ph == "lift"
            and bool(sim.captured)
            and hand_z_lift0 is not None
            and float(o["obj_z"]) >= float(Z_AIR)
            and (hz - hand_z_lift0) >= float(HAND_RISE_M)
        ):
            po, Ro = object_pose(sim)
            pre_ctrl = ctrl_pack(sim)
            pre_ic = ic_pack(sim)
            pre_snap = sim.snapshot()
            pre_snap["friction"] = float(PAIR_MU)
            rise = hz - hand_z_lift0
            pres.ghost_cyl = (np.asarray(po, float).copy(), np.asarray(Ro[:, 2], float).copy())
            pres.phase = "DISTURBANCE"
            pres.speed = SPEED_DISTURB
            dist = apply_lift_disturbance(sim, ref, S_COUPLED)
            post_snap = sim.snapshot()
            post_snap["friction"] = float(PAIR_MU)
            o2 = physical_pack(sim)
            log.append(traj_row(sim, "DISTURBANCE", TAU_SEC))
            log[-1]["disturbed"] = True
            pres.note(
                "disturbance",
                hand_rise=rise,
                obj_z=float(o2["obj_z"]),
                controller_untouched=dist["controller_untouched"],
                v_cmd=np.asarray(sim.fsm.v_cmd, float).tolist(),
            )
            pres.show()
            pres.pause_wall(DISTURB_FREEZE_S)
            pres.ghost_cyl = None
            return {
                "pre_snap": pre_snap,
                "post_snap": post_snap,
                "pre_ic": pre_ic,
                "post_ic": ic_pack(sim),
                "pre_ctrl": {k: _jsonable(v) for k, v in pre_ctrl.items()},
                "disturbance": dist,
                "t_disturb": float(sim.data.time),
                "t_lift0": t_lift0,
                "t_close": t_close,
                "hand_z_lift0": hand_z_lift0,
                "hand_rise": rise,
                "obj_z_pre": float(o["obj_z"]),
                "obj_z_post": float(o2["obj_z"]),
                "z_tgt": z_lift_target(sim),
                "trigger": {
                    "definition": (
                        "fsm.phase==lift AND captured AND obj_z>=Z_AIR "
                        f"({Z_AIR} m) AND (hand_z - hand_z_at_lift_start) >= {HAND_RISE_M} m"
                    ),
                    "Z_AIR": Z_AIR,
                    "HAND_RISE_M": HAND_RISE_M,
                    "obj_z_pre": float(o["obj_z"]),
                    "hand_rise": rise,
                },
            }
        if prev != ph:
            pres.note("phase", fsm=ph)
    raise RuntimeError("disturbance trigger never reached during nominal LIFT")


def continue_lift_and_hold(sim, gains, pres: Presenter, log: list, *, z_tgt: float, phase_lift: str) -> dict:
    """Keep current p_des/r_des. Finish remaining lift to z_tgt, then hold. No p_des snap."""
    dt = float(sim.model.opt.timestep)
    log_n = max(int(round(LOG_DT / dt)), 1)
    t0 = float(sim.data.time)
    t_lift_done = None
    t_drop = None
    t_both = None
    i = 0
    n_max = int(round((8.0 + HOLD_S + AFTER_DROP_S) / dt))
    while i < n_max:
        lifting = float(sim.fsm.p_des[2]) < float(z_tgt) - 1e-9 and t_lift_done is None
        if lifting:
            pres.phase = phase_lift
            pres.speed = SPEED_LIFT
            pres.tau = TAU_SEC
            tick_vw(sim, np.array([0.0, 0.0, V_LIFT]), np.zeros(3), TAU_SEC, gains)
        else:
            if t_lift_done is None:
                t_lift_done = float(sim.data.time)
                pres.note("lift_done", p_des_z=float(sim.fsm.p_des[2]), z_tgt=float(z_tgt))
            mchk = physical_pack(sim)
            if dropped(sim, {"obj_z": mchk["obj_z"], "nL": mchk["nL"], "nR": mchk["nR"]}) or (
                mchk["nL"] == 0 and mchk["nR"] == 0
            ):
                pres.phase = "DROP"
                pres.speed = SPEED_DROP
                if t_drop is None:
                    t_drop = float(sim.data.time)
                    pres.note("drop", obj_z=float(mchk["obj_z"]))
            else:
                pres.phase = "HIGH HOLD"
                pres.speed = SPEED_HOLD
            pres.tau = TAU_SEC
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        o = physical_pack(sim)
        t = float(sim.data.time)
        if t_both is None and o["nL"] == 0 and o["nR"] == 0:
            t_both = t
        if t_lift_done is None and float(sim.fsm.p_des[2]) >= float(z_tgt) - 1e-9:
            t_lift_done = t
        if i % log_n == 0:
            log.append(traj_row(sim, pres.phase, TAU_SEC))
        maybe_show(sim, pres, i, dt)
        i += 1
        if t_drop is not None and t >= t_drop + AFTER_DROP_S:
            break
        if t_lift_done is not None and t_drop is None and t >= t_lift_done + HOLD_S:
            break
    end = physical_pack(sim)
    out = {
        "t0": t0,
        "t_lift_done": t_lift_done,
        "t_drop": None if t_drop is None else t_drop - t0,
        "t_both_loss": None if t_both is None else t_both - t0,
        "lift_completed": t_lift_done is not None,
        "drop": t_drop is not None,
        "hold_s": HOLD_S,
        "hold_survived": bool(
            t_lift_done is not None
            and t_drop is None
            and (float(sim.data.time) - t_lift_done) >= HOLD_S - 1e-6
        ),
        "end_nL": int(end["nL"]),
        "end_nR": int(end["nR"]),
        "end_obj_z": float(end["obj_z"]),
        "end_p_des_z": float(sim.fsm.p_des[2]),
        "z_tgt": float(z_tgt),
        "dt": dt,
        "v_lift": V_LIFT,
        "tau": TAU_SEC,
    }
    pres.note("continuation_end", **out)
    return out


def run_zero_branch(sim, gains, packed, pres: Presenter, log: list) -> dict:
    restore(sim, packed["post_snap"])
    pres.tau = TAU_SEC
    pres.phase = "LIFT"
    pres.note("zero_resume", sha=packed["post_ic"]["sha256"], v_cmd=np.asarray(sim.fsm.v_cmd, float).tolist())
    return continue_lift_and_hold(sim, gains, pres, log, z_tgt=packed["z_tgt"], phase_lift="LIFT")


def _rotate(sim, gains, pres, Rh0, target_deg, tau, omega, log) -> None:
    dt = float(sim.model.opt.timestep)
    sign = 1.0 if target_deg >= 0 else -1.0
    goal = abs(float(target_deg))
    timeout = float(sim.data.time) + goal / max(omega, 1e-6) + 1.5
    log_n = max(int(round(LOG_DT / dt)), 1)
    i = 0
    while float(sim.data.time) < timeout:
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        ang = signed_rot_about_y_deg(Rh0, Rh)
        if abs(ang) >= goal - 0.35:
            break
        w_w = Rh @ np.array([0.0, sign * omega, 0.0])
        tick_vw(sim, np.zeros(3), w_w, tau, gains)
        if i % log_n == 0:
            log.append(traj_row(sim, pres.phase, tau))
            pres.show()
        i += 1


def _return(sim, gains, pres, Rh0, tau, omega, log) -> None:
    dt = float(sim.model.opt.timestep)
    timeout = float(sim.data.time) + (abs(THETA) * np.pi / 180.0) / max(omega, 1e-6) + 2.0
    log_n = max(int(round(LOG_DT / dt)), 1)
    i = 0
    while float(sim.data.time) < timeout:
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        ang = signed_rot_about_y_deg(Rh0, Rh)
        if abs(ang) < 0.5:
            break
        sgn = -1.0 if ang > 0 else 1.0
        w_w = Rh @ np.array([0.0, sgn * omega, 0.0])
        tick_vw(sim, np.zeros(3), w_w, tau, gains)
        if i % log_n == 0:
            log.append(traj_row(sim, pres.phase, tau))
            pres.show()
        i += 1


def run_heuristic_branch(sim, gains, packed, pres: Presenter, log: list) -> dict:
    restore(sim, packed["post_snap"])
    dt = float(sim.model.opt.timestep)
    log_n = max(int(round(LOG_DT / dt)), 1)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    o0 = physical_pack(sim)
    u = np.array([-np.sign(float(o0["rh"][0])) if abs(float(o0["rh"][0])) > 1e-6 else 1.0, 0.0, 0.0])
    pres.speed = SPEED_RECOVERY
    pres.tau = TAU_SEC
    pres.phase = "RECOVERY — ROTATE"
    pres.note("heuristic_start", sha=packed["post_ic"]["sha256"])
    n_obs = int(round(OBS_S / dt))
    for i in range(n_obs):
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        if i % log_n == 0:
            log.append(traj_row(sim, pres.phase, TAU_SEC))
            pres.show()
    _rotate(sim, gains, pres, Rh0, THETA, TAU_SEC, OMEGA, log)
    freeze(sim)
    pres.note("rotate_end")

    pres.phase = "RECOVERY — WEAK REPOSITION"
    pres.tau = TAU_WEAK
    rh_s = physical_pack(sim)["rh"].copy()
    n = int(round(SLIP_MAX_S / dt))
    for i in range(n):
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_WEAK, gains)
        m = physical_pack(sim)
        if i % log_n == 0:
            log.append(traj_row(sim, pres.phase, TAU_WEAK))
            pres.show()
        if dropped(sim, m) or m["nL"] == 0 or m["nR"] == 0:
            pres.note("lost_during_slip")
            break
        if useful_mm(m["rh"], rh_s, u) >= SLIP_TARGET_MM:
            pres.note("slip_target", useful_mm=useful_mm(m["rh"], rh_s, u))
            break

    pres.phase = "RECOVERY — SECURE"
    pres.tau = TAU_SEC
    freeze(sim)
    n_br = int(round(BRAKE_HOLD / dt))
    for i in range(n_br):
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        if i % log_n == 0:
            log.append(traj_row(sim, pres.phase, TAU_SEC))
            pres.show()
    pres.note("secure")

    pres.phase = "RECOVERY — RETURN"
    pres.tau = TAU_SEC
    _return(sim, gains, pres, Rh0, TAU_SEC, OMEGA_RET, log)
    freeze(sim)
    pres.note("return_end", p_des_z=float(sim.fsm.p_des[2]), z_tgt=float(packed["z_tgt"]))
    return continue_lift_and_hold(
        sim, gains, pres, log, z_tgt=packed["z_tgt"], phase_lift="LIFT RESUME"
    )


def archive_old_videos() -> None:
    ARCH.mkdir(parents=True, exist_ok=True)
    for name in ("zero.mp4", "heuristic.mp4", "zero_vs_heuristic.mp4"):
        src = OUT / name
        dest = ARCH / name
        if src.is_file() and src.stat().st_size > 1000 and not dest.is_file():
            shutil.copy2(src, dest)


def write_readme(meta: dict) -> None:
    p = OUT / "README.md"
    a = []
    ap = a.append
    ap("# ZERO vs HEURISTIC — complete pick-and-lift task")
    ap("")
    ap("Presentation of a continuous nominal grasping task with an in-lift disturbance.")
    ap("Physics, disturbance family, RULE, SAC, reward, and observation were not changed.")
    ap("Heuristic parameters were not retuned.")
    ap("")
    ap("**HEURISTIC** means the *manually constructed* gravitational-reposition sequence.")
    ap("It is **not** `rule_based_recovery.yaml`, and not SAC.")
    ap("")
    ap("## USER VISUAL VALIDATION: PENDING")
    ap("")
    ap("Until the user watches the videos, do not treat them as visually confirmed.")
    ap("")
    ap("Desired visual sequence to inspect:")
    ap("")
    ap("    hand starts above object → descends → grasps → starts lifting")
    ap("        → disturbance suddenly tilts/displaces cylinder DURING lift")
    ap("        → ZERO vs HEURISTIC diverge → final lift/hold outcome.")
    ap("")
    verdict = meta.get("case")
    ap("## Experiment identity")
    ap("")
    if verdict == "A":
        ap("**A. COMPLETE TASK REPLAY REPRODUCES THE EXISTING CANONICAL PARENT → S_FAIL.**")
    else:
        ap("**B. IN-LIFT DISTURBANCE CREATES A NEW POST-DISTURBANCE STATE `S_FAIL_LIFT`;")
        ap("old S_FAIL recovery evidence is not being reused.**")
    ap("")
    ap(meta.get("case_reason", ""))
    ap("")
    ap("The previous videos started from an airborne parent / `sfail.pkl` (or a")
    ap("presentation of that construction). They hid APPROACH/DESCEND/CLOSE and")
    ap("applied the disturbance before a separate lift continuation.")
    ap("")
    ap("This revision runs the actual nominal Cartesian GraspFSM from reset.")
    ap("Disturbance is applied while LIFT is live. Controller targets are not reset.")
    ap("`sfail.pkl` is left untouched.")
    ap("")
    ap("## Disturbance trigger")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("trigger"), indent=2))
    ap("```")
    ap("")
    ap("## Parent embed audit")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("embed"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## Controller at disturbance")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("controller_audit"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## S_FAIL_LIFT vs old sfail.pkl")
    ap("")
    ap(f"- old parent: `{PARENT_PKL.as_posix()}`")
    ap(f"- old S_FAIL: `{SFAIL_PKL.as_posix()}` (untouched)")
    ap(f"- new pre-disturbance: `{PRE_DISTURB_PKL.as_posix()}`")
    ap(f"- new S_FAIL_LIFT: `{SFAIL_LIFT_PKL.as_posix()}`")
    ap(f"- S_FAIL_LIFT sha256: `{meta.get('sfail_lift_sha')}`")
    ap(f"- ZERO / HEURISTIC post-disturbance hashes equal: **{meta.get('hash_equal')}**")
    ap(f"- disturbance: `coupled_pose_from_ref(ref, s=2.0, kind='coupled') + apply_coupled_pose`")
    ap(f"- vs old sfail.pkl match: **{meta.get('vs_old_sfail_match')}**")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("vs_old_sfail"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## Physics")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("live"), indent=2))
    ap("```")
    ap("")
    ap("m=0.20 kg, pair μ=1.0, noslip_iterations=1, coupled teleport s=2.0, pose-only.")
    ap("")
    ap("## ZERO")
    ap("")
    ap("Nominal FSM through disturbance, then remaining LIFT (same p_des integration,")
    ap("no snap) → HIGH HOLD 12 s or drop + 2 s.")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("zero"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## HEURISTIC")
    ap("")
    ap("Same prefix. After disturbance: ROTATE +30° τ=−18 → WEAK REPOSITION τ=−4")
    ap("until ~1 mm useful x → SECURE 0.5 s τ=−18 → RETURN. Then **resume** remaining")
    ap("lift to the original `grasp_z + lift_offset_z` (no second lift from a snapped p_des).")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("heuristic"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## Actual outcomes (not forced)")
    ap("")
    ap(f"- ZERO dropped: **{meta.get('zero_drop')}**")
    ap(f"- HEURISTIC hold survived 12 s: **{meta.get('heu_survived')}**")
    if meta.get("outcome_note"):
        ap("")
        ap(meta["outcome_note"])
    ap("")
    ap("## Playback / render")
    ap("")
    ap(f"- task {SPEED_TASK}×, lift {SPEED_LIFT}×, disturbance {SPEED_DISTURB}×,")
    ap(f"  recovery {SPEED_RECOVERY}×, high hold {SPEED_HOLD}×, near drop {SPEED_DROP}×")
    ap("- playback is sleep/frame skip only; dt=0.002 and step count are physical")
    ap(f"- ~{DISTURB_FREEZE_S:.2f} s presentation freeze after the teleport (no mj_step)")
    ap(f"- video {VIDEO_FPS} fps, {RENDER_W}×{RENDER_H}")
    ap(f"- camera distance {CAM_DISTANCE} m, azimuth {CAM_AZIMUTH}, elevation {CAM_ELEVATION}, one-shot")
    ap("- superseded airborne-opening mp4s: `superseded_airborne_opening/`")
    ap("")
    ap("## Commands")
    ap("")
    ap("```text")
    ap("python training/demo_grav_reposition_comparison.py")
    ap("python training/demo_grav_reposition_comparison.py --mode zero")
    ap("python training/demo_grav_reposition_comparison.py --mode heuristic")
    ap("python training/demo_grav_reposition_comparison.py --render-video side_by_side")
    ap("```")
    ap("")
    ap("## Paths")
    ap("")
    ap(f"- videos: `{OUT.as_posix()}`")
    ap(f"- raw: `{RAW.as_posix()}`")
    p.write_text("\n".join(a), encoding="utf-8")


def run_headless_pair() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    gains = gains_from_cfg(cfg)
    live = live_opt(sim)
    if int(live["noslip_iterations"]) != 1:
        raise RuntimeError(live)
    embed = audit_old_parent_embed(sim)
    ref = load_or_measure_ref()

    sim = make_parent_sim()
    pres_p = Presenter(sim, label="PREFIX", interactive=False, record=False)
    log_p: list = []
    packed = run_nominal_to_disturbance(sim, pres_p, ref, log_p)
    with PRE_DISTURB_PKL.open("wb") as f:
        pickle.dump(packed["pre_snap"], f)
    with SFAIL_LIFT_PKL.open("wb") as f:
        pickle.dump(packed["post_snap"], f)

    vs_old = None
    vs_match = False
    if SFAIL_PKL.is_file():
        with SFAIL_PKL.open("rb") as f:
            old = pickle.load(f)
        vs_old = compare_to_snap(sim, old)
        vs_match = bool(vs_old.get("match"))

    log_z = list(log_p)
    pres_z = Presenter(sim, label="ZERO", interactive=False, record=False)
    z = run_zero_branch(sim, gains, packed, pres_z, log_z)
    z_sha = packed["post_ic"]["sha256"]
    save_traj_npz(RAW / "zero.npz", log_z, extra={"t_disturb": packed["t_disturb"]})
    dump_json(RAW / "zero_events.json", pres_p.events + pres_z.events)

    log_h = list(log_p)
    pres_h = Presenter(sim, label="HEURISTIC", interactive=False, record=False)
    h = run_heuristic_branch(sim, gains, packed, pres_h, log_h)
    heu_sha = packed["post_ic"]["sha256"]
    save_traj_npz(RAW / "heuristic.npz", log_h, extra={"t_disturb": packed["t_disturb"]})
    dump_json(RAW / "heuristic_events.json", pres_p.events + pres_h.events)

    dump_json(RAW / "ic_sfail_lift.json", packed["post_ic"])
    dump_json(RAW / "ic_pre_disturbance.json", packed["pre_ic"])
    dump_json(RAW / "disturbance.json", packed["disturbance"])

    case = "A" if (embed.get("verdict") == "A" and vs_match) else "B"
    z_pub = {k: v for k, v in z.items() if k != "log"}
    h_pub = {k: v for k, v in h.items() if k != "log"}
    outcome_note = None
    if not z_pub.get("drop"):
        outcome_note = (
            "ZERO did not drop during remaining lift + 12 s hold on this in-lift IC. "
            "Parameters were not retuned."
        )
    if not h_pub.get("hold_survived"):
        extra = "HEURISTIC did not survive the 12 s high hold on this in-lift IC. Parameters were not retuned."
        outcome_note = extra if outcome_note is None else outcome_note + " " + extra

    meta = {
        "case": case,
        "case_reason": embed.get("reason"),
        "embed": {k: v for k, v in embed.items() if k != "best_replay"},
        "best_replay": embed.get("best_replay"),
        "trigger": packed["trigger"],
        "t_disturb": packed["t_disturb"],
        "t_lift0": packed["t_lift0"],
        "controller_audit": {
            "untouched": packed["disturbance"]["controller_untouched"],
            "diffs": packed["disturbance"]["ctrl_diffs"],
            "v_cmd_pre": packed["pre_ctrl"].get("v_cmd"),
        },
        "sfail_lift_sha": z_sha,
        "hash_equal": z_sha == heu_sha,
        "vs_old_sfail": vs_old,
        "vs_old_sfail_match": vs_match,
        "live": live,
        "zero": z_pub,
        "heuristic": h_pub,
        "zero_drop": z_pub.get("drop"),
        "heu_survived": h_pub.get("hold_survived"),
        "outcome_note": outcome_note,
        "zero_sha": z_sha,
        "heu_sha": heu_sha,
        "packed_meta": {
            "hand_rise": packed["hand_rise"],
            "obj_z_pre": packed["obj_z_pre"],
            "obj_z_post": packed["obj_z_post"],
            "z_tgt": packed["z_tgt"],
        },
    }
    dump_json(RAW / "validation.json", meta)
    write_readme(meta)
    return meta, packed, ref


def _run_recorded(sim, gains, ref, label: str, branch: str) -> tuple[dict, list, list]:
    pres_p = Presenter(sim, label=label, interactive=False, record=True)
    pres_p.set_cam()
    log_p: list = []
    packed = run_nominal_to_disturbance(sim, pres_p, ref, log_p)
    prefix_frames = list(pres_p.frames)
    prefix_events = list(pres_p.events)
    pres_p.close()
    pres = Presenter(sim, label=label, interactive=False, record=True)
    pres.set_cam()
    log = list(log_p)
    if branch == "zero":
        out = run_zero_branch(sim, gains, packed, pres, log)
    else:
        out = run_heuristic_branch(sim, gains, packed, pres, log)
    frames = prefix_frames + list(pres.frames)
    events = prefix_events + list(pres.events)
    pres.close()
    return out, frames, events


def encode_mp4(frame_dir: Path, dest: Path, fps: int = VIDEO_FPS) -> str | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%06d.png")
    cmd_tail = ["-y", "-framerate", str(fps), "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)]
    candidates = []
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    candidates.append("ffmpeg")
    import subprocess
    for exe in candidates:
        try:
            subprocess.run([exe, *cmd_tail], check=True, capture_output=True)
            if dest.is_file() and dest.stat().st_size > 1000:
                return str(dest)
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            continue
    return None


def write_frames_mp4(frames, dest: Path) -> str | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f"_frames_{dest.stem}"
    if tmp.exists():
        for p in tmp.glob("*.png"):
            p.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        for i, fr in enumerate(frames):
            imageio.imwrite(str(tmp / f"frame_{i:06d}.png"), fr)
    except Exception:
        import matplotlib.image as mpimg
        for i, fr in enumerate(frames):
            mpimg.imsave(str(tmp / f"frame_{i:06d}.png"), fr)
    return encode_mp4(tmp, dest, fps=VIDEO_FPS)


def hstack_videos(left: list[np.ndarray], right: list[np.ndarray], dest: Path) -> str | None:
    n = max(len(left), len(right))
    if n == 0:
        return None
    L = left[-1] if left else np.zeros((RENDER_H, RENDER_W, 3), np.uint8)
    R = right[-1] if right else np.zeros((RENDER_H, RENDER_W, 3), np.uint8)
    frames = []
    for i in range(n):
        a = left[i] if i < len(left) else L
        b = right[i] if i < len(right) else R
        if a.shape != b.shape:
            b = np.array(b)
        frames.append(np.concatenate([a, b], axis=1))
    return write_frames_mp4(frames, dest)


def interactive(mode: str) -> None:
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    gains = gains_from_cfg(cfg)
    ref = load_or_measure_ref()
    key_holder = {}

    def _key(keycode):
        fn = key_holder.get("fn")
        if fn:
            fn(keycode)

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=_key) as vwr:
        viz = {"viewer": vwr, "camera": "wide"}
        key_holder["fn"] = make_reset_cam_callback(viz)
        pres = Presenter(
            sim,
            label="ZERO" if mode == "zero" else "HEURISTIC",
            interactive=True,
            record=False,
            viz=viz,
        )
        pres.set_cam()
        pres.pause_wall(PRE_VIEW_PAUSE_S)
        log: list = []
        packed = run_nominal_to_disturbance(sim, pres, ref, log)
        if mode == "zero":
            run_zero_branch(sim, gains, packed, pres, log)
        else:
            run_heuristic_branch(sim, gains, packed, pres, log)
        pres.pause_wall(2.0)


def render_video(kind: str, ref) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    archive_old_videos()
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)

    def _one(label, branch, dest):
        sim = make_parent_sim()
        out, frames, events = _run_recorded(sim, gains, ref, label, branch)
        mp4 = write_frames_mp4(frames, dest)
        return out, frames, mp4, events

    if kind == "zero":
        z, _, mp4, _ = _one("ZERO", "zero", OUT / "zero.mp4")
        dump_json(RAW / "video_zero.json", {k: v for k, v in z.items() if k != "log"} | {"mp4": mp4})
        print("wrote", mp4)
        return
    if kind == "heuristic":
        h, _, mp4, _ = _one("HEURISTIC", "heuristic", OUT / "heuristic.mp4")
        dump_json(RAW / "video_heuristic.json", {k: v for k, v in h.items() if k != "log"} | {"mp4": mp4})
        print("wrote", mp4)
        return
    z, fz, mp4z, _ = _one("ZERO", "zero", OUT / "zero.mp4")
    h, fh, mp4h, _ = _one("HEURISTIC", "heuristic", OUT / "heuristic.mp4")
    side = hstack_videos(fz, fh, OUT / "zero_vs_heuristic.mp4")
    dump_json(
        RAW / "video_side.json",
        {
            "zero": {k: v for k, v in z.items() if k != "log"},
            "heuristic": {k: v for k, v in h.items() if k != "log"},
            "zero_mp4": mp4z,
            "heuristic_mp4": mp4h,
            "side_mp4": side,
        },
    )
    print("wrote", mp4z, mp4h, side)


def main():
    p = argparse.ArgumentParser(
        description="Complete-task ZERO vs HEURISTIC (manual grav-reposition, not RULE/SAC)."
    )
    p.add_argument("--mode", default="", help="zero | heuristic (interactive viewer)")
    p.add_argument("--render-video", default="", dest="render_video", help="zero | heuristic | side_by_side")
    args = p.parse_args()
    if args.mode:
        interactive(args.mode)
        return
    meta, packed, ref = run_headless_pair()
    pub = {
        "case": meta.get("case"),
        "controller_untouched": (meta.get("controller_audit") or {}).get("untouched"),
        "hash_equal": meta.get("hash_equal"),
        "vs_old_sfail_match": meta.get("vs_old_sfail_match"),
        "zero_drop": meta.get("zero_drop"),
        "heu_survived": meta.get("heu_survived"),
        "outcome_note": meta.get("outcome_note"),
        "trigger": meta.get("trigger"),
        "zero": meta.get("zero"),
        "heuristic": meta.get("heuristic"),
    }
    print(json.dumps(pub, indent=2, default=str))
    if args.render_video:
        kind = args.render_video
        render_video("side_by_side" if kind == "side_by_side" else kind, ref)
        write_readme(meta)
        return


if __name__ == "__main__":
    main()
