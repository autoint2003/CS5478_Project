"""Canonical impact-demo episode: guided approach + free impact, shared by viewer and headless.

Does not train SAC, does not modify frozen RULE YAML/FSM, does not read held-out eval npz.

Clocks (visualization only; physics dt is unchanged):
    PHYSICS  500 Hz  (dt = 0.002 s, one mj_step per loop)
    CONTROL  existing per-step rates (nominal FSM and recovery tick once per dt)
    RENDER   ~60 Hz wall-clock viewer.sync, paced by sim-time / playback_speed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time as _time

import mujoco
import numpy as np
import yaml

from controllers.jacobian_controller import gains_from_cfg
from controllers.nominal import grasp_orientation
from controllers.rule_based_recovery import RuleBasedRecovery
from envs.config_util import load_yaml, merge_sim_config
from envs.observable_reward import read_tactile
from envs.physical_recovery import (
    SUCCESS_HOLD,
    Z_AIR,
    fail_kind,
    physical_pack,
    recovered,
)
from training.impact_ball import apply_ball_free_state, ball_hits_cylinder, ball_pose_twist
from training.impact_visualization_utils import (
    HoldClocks,
    apply_viewer_camera,
    enable_viewer_flags,
    tick_vw_park,
)
from training.replay_core import (
    G_HOLD,
    MASS,
    MU,
    ImpactSim,
    disable_object_table,
    dump_ball_contacts,
    freeze,
    isolate_ball_object_only,
    obs_from_sim,
    park_impact_ball,
    params_from_frozen,
    raw_object_contacts,
    sha256_file,
    verify_frozen_hashes,
    CTRL_PY,
    FROZEN_YAML,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO_YAML = ROOT / "config" / "impact_demo.yaml"
LOG_DIR = ROOT / "results" / "logs" / "impact_demo"
SHOT_DIR = LOG_DIR / "screenshots"
RENDER_HZ = 60.0
PHYSICS_DT = 0.002


def load_demo_config() -> dict:
    return yaml.safe_load(DEMO_YAML.read_text(encoding="utf-8"))


def interval_stats(ts) -> dict:
    ts = np.asarray(ts, float)
    if ts.size < 2:
        return {
            "n_sync": int(ts.size),
            "mean_s": None,
            "median_s": None,
            "p95_s": None,
            "max_s": None,
            "fps": None,
            "n_gt_25ms": 0,
            "n_gt_50ms": 0,
            "n_gt_100ms": 0,
        }
    dtv = np.diff(ts)
    mean = float(np.mean(dtv))
    return {
        "n_sync": int(ts.size),
        "mean_s": mean,
        "median_s": float(np.median(dtv)),
        "p95_s": float(np.percentile(dtv, 95)),
        "max_s": float(np.max(dtv)),
        "fps": float(1.0 / max(mean, 1e-12)),
        "n_gt_25ms": int(np.sum(dtv > 0.025)),
        "n_gt_50ms": int(np.sum(dtv > 0.050)),
        "n_gt_100ms": int(np.sum(dtv > 0.100)),
    }


def ball_step_stats(t, p_ball, p_guide, v_ball) -> dict:
    t = np.asarray(t, float)
    p_ball = np.asarray(p_ball, float).reshape(-1, 3)
    p_guide = np.asarray(p_guide, float).reshape(-1, 3)
    v_ball = np.asarray(v_ball, float).reshape(-1, 3)
    n = int(p_ball.shape[0])
    if n < 3:
        return {"n": n}
    dp = np.diff(p_ball, axis=0)
    dv = np.diff(v_ball, axis=0)
    dg = np.diff(p_guide, axis=0)
    dpn = np.linalg.norm(dp, axis=1)
    dvn = np.linalg.norm(dv, axis=1)
    dgn = np.linalg.norm(dg, axis=1)
    # Piecewise-constant guide at 50 Hz would spike every 10 physics steps.
    lag10 = dpn[9::10] if dpn.size >= 10 else np.array([])
    other = np.delete(dpn, np.arange(9, dpn.size, 10)) if dpn.size >= 10 else dpn
    return {
        "n": n,
        "dt_median": float(np.median(np.diff(t))) if t.size > 1 else None,
        "dp_mean_m": float(np.mean(dpn)),
        "dp_max_m": float(np.max(dpn)),
        "dp_p95_m": float(np.percentile(dpn, 95)),
        "dv_mean_mps": float(np.mean(dvn)),
        "dv_max_mps": float(np.max(dvn)),
        "dguide_mean_m": float(np.mean(dgn)),
        "dguide_max_m": float(np.max(dgn)),
        "dp_mean_every_10_steps_m": float(np.mean(lag10)) if lag10.size else None,
        "dp_mean_other_steps_m": float(np.mean(other)) if other.size else None,
        "lag10_vs_other_ratio": (
            float(np.mean(lag10) / max(float(np.mean(other)), 1e-12))
            if lag10.size and other.size
            else None
        ),
    }


class RealtimePacer:
    """Map simulation time to wall time. Sleep only when sim is ahead of playback.

    Render cadence is in *simulation* time so a 16.0 ms vs 16.7 ms wall comparison
    cannot skip a frame (that skip produced ~36 FPS at 0.25x playback).
    """

    def __init__(self, playback_rate: float, render_hz: float = RENDER_HZ, dt: float = PHYSICS_DT):
        self.rate = max(float(playback_rate), 1e-9)
        self.dt = float(dt)
        self.render_hz = max(float(render_hz), 1.0)
        # Sim time between displays so wall FPS ≈ render_hz when pacing holds.
        self.sim_per_frame = (1.0 / self.render_hz) * self.rate
        self.wall0 = None
        self.sim0 = None
        self.last_render_sim = -1.0e9
        self.sync_times: list[float] = []

    def start(self, sim_t: float) -> None:
        self.wall0 = _time.perf_counter()
        self.sim0 = float(sim_t)
        self.last_render_sim = float(sim_t) - self.sim_per_frame

    def should_render(self, sim_t: float) -> bool:
        return (float(sim_t) - self.last_render_sim) + 0.5 * self.dt >= self.sim_per_frame

    def note_sync(self, sim_t: float) -> None:
        self.sync_times.append(_time.perf_counter())
        self.last_render_sim = float(sim_t)

    def wait_if_ahead(self, sim_t: float) -> None:
        desired = self.wall0 + (float(sim_t) - self.sim0) / self.rate
        while True:
            remaining = desired - _time.perf_counter()
            if remaining <= 0.0:
                return
            if remaining > 0.002:
                _time.sleep(remaining - 0.001)
            else:
                while _time.perf_counter() < desired:
                    pass
                return


def _sim_snapshot(sim) -> dict:
    snap = {
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "time": float(sim.data.time),
        "mocap_pos": np.array(sim.data.mocap_pos, float).copy(),
        "mocap_quat": np.array(sim.data.mocap_quat, float).copy(),
    }
    if hasattr(sim.data, "eq_active"):
        snap["eq_active"] = np.array(sim.data.eq_active).copy()
    nact = int(getattr(sim.model, "na", 0) or 0)
    if nact:
        snap["act"] = np.array(sim.data.act, float).copy()
    return snap


def _apply_snapshot(sim, snap: dict) -> None:
    sim.data.qpos[:] = snap["qpos"]
    sim.data.qvel[:] = snap["qvel"]
    sim.data.ctrl[:] = snap["ctrl"]
    sim.data.time = snap["time"]
    sim.data.mocap_pos[:] = snap["mocap_pos"]
    sim.data.mocap_quat[:] = snap["mocap_quat"]
    if "eq_active" in snap and hasattr(sim.data, "eq_active"):
        sim.data.eq_active[:] = snap["eq_active"]
    if "act" in snap and int(getattr(sim.model, "na", 0) or 0):
        sim.data.act[:] = snap["act"]
    mujoco.mj_forward(sim.model, sim.data)


def _write_event_screenshots(sim, cam: dict, prefix: str, events: dict) -> None:
    if not events:
        return
    final = _sim_snapshot(sim)
    for name, snap in events.items():
        _apply_snapshot(sim, snap)
        save_rgb(SHOT_DIR / f"{prefix}{name}.png", sim, cam)
    _apply_snapshot(sim, final)


def frozen_hashes() -> dict:
    return {
        "yaml": sha256_file(FROZEN_YAML),
        "controller_py": sha256_file(CTRL_PY),
        "verify": verify_frozen_hashes(),
    }


def ballistic_boundary(p_hit, v_hit, T, g) -> tuple[np.ndarray, np.ndarray]:
    """p0, v0 for constant gravity so (p,v)=(p_hit,v_hit) at t=T."""
    p_hit = np.asarray(p_hit, float).reshape(3)
    v_hit = np.asarray(v_hit, float).reshape(3)
    g = np.asarray(g, float).reshape(3)
    T = float(T)
    v0 = v_hit - g * T
    p0 = p_hit - v_hit * T + 0.5 * g * T * T
    return p0, v0


def design_v_hit(p_hit, T, g, v_hit_y: float, p0_z: float) -> np.ndarray:
    """Horizontal +Y impact with v_hit_z chosen so p0_z matches a visible start height."""
    p_hit = np.asarray(p_hit, float)
    g = np.asarray(g, float)
    T = float(T)
    v_hit_z = (float(p_hit[2]) - float(p0_z) + 0.5 * g[2] * T * T) / T
    return np.array([0.0, float(v_hit_y), float(v_hit_z)])


def apex_of(p0, v0, g) -> tuple[float, np.ndarray]:
    p0 = np.asarray(p0, float)
    v0 = np.asarray(v0, float)
    g = np.asarray(g, float)
    t_a = float(-v0[2] / g[2]) if (g[2] < -1e-9 and v0[2] > 0) else 0.0
    t_a = max(0.0, t_a)
    return t_a, p0 + v0 * t_a + 0.5 * g * t_a * t_a


def path_length_analytic(p0, v0, T, g, n: int = 80) -> float:
    ts = np.linspace(0.0, float(T), n)
    pts = np.stack([p0 + v0 * t + 0.5 * g * t * t for t in ts], axis=0)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def make_demo_sim(ball_mass: float) -> tuple[ImpactSim, dict]:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = ImpactSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    isolate_ball_object_only(sim)
    sim.set_ball_mass(float(ball_mass))
    if hasattr(sim.model, "body_gravcomp"):
        sim.model.body_gravcomp[int(sim.ball_body)] = 0.0
    return sim, cfg


def compute_demo_camera(p0, p_hit, p_obj) -> dict:
    p0 = np.asarray(p0, float)
    p_hit = np.asarray(p_hit, float)
    p_obj = np.asarray(p_obj, float)
    lookat = 0.62 * p_obj + 0.38 * p0
    lookat[2] = float(np.clip(0.5 * (p_obj[2] + p0[2]), 0.42, 0.70))
    span = float(np.linalg.norm(p0 - p_obj))
    dist = float(np.clip(1.15 * span + 0.85, 1.45, 2.55))
    return {
        "lookat": lookat,
        "distance": dist,
        "azimuth": 125.0,
        "elevation": -22.0,
        "span": span,
    }


def plan_guided_release(
    p_hit,
    v_lat: float,
    t_release: float,
    t_free: float,
    t_accel: float,
    v_cruise: float,
    g,
    p_start_z: float = 0.55,
) -> dict:
    """Lateral +Y impact. Guided cruise then explicit accel; then free flight.

    v_hit_z is the gravity compensation over t_free so p_release_z ~ p_hit_z.
    Cruise height is a visualization choice (p_start_z); v_c_z is solved so
    position is C1-consistent with the release state.
    """
    p_hit = np.asarray(p_hit, float).reshape(3)
    g = np.asarray(g, float).reshape(3)
    t_free = float(t_free)
    t_rel = float(t_release)
    t_acc = float(t_accel)
    v_lat = float(v_lat)
    v_hit = np.array([0.0, v_lat, 0.5 * float(g[2]) * t_free])
    v_release = v_hit - g * t_free
    p_release = p_hit - v_hit * t_free + 0.5 * g * t_free * t_free
    t1 = t_rel - t_acc
    if t1 <= 0.05:
        raise ValueError("t_accel too large vs t_release")
    integ_y = float(v_cruise) * t1 + 0.5 * (float(v_cruise) + float(v_release[1])) * t_acc
    p_start_y = float(p_release[1]) - integ_y
    den = t1 + 0.5 * t_acc
    v_c_z = (float(p_release[2]) - float(p_start_z) - 0.5 * float(v_release[2]) * t_acc) / den
    v_c = np.array([0.0, float(v_cruise), float(v_c_z)])
    p_start = np.array([float(p_release[0]), p_start_y, float(p_start_z)])
    return {
        "p_start": p_start,
        "p_release": p_release,
        "v_cruise": v_c,
        "v_release": v_release,
        "v_hit": v_hit,
        "p_hit": p_hit,
        "t_release": t_rel,
        "t_free": t_free,
        "t_accel": t_acc,
        "t1": t1,
    }


class GuidedApproach:
    """Explicit launcher: constant-velocity cruise, then linear speed ramp.

    Not free ballistic. Drives a mocap weld target; does not write ball qpos.
    """

    def __init__(self, plan: dict):
        self.plan = plan
        self.p_start = np.asarray(plan["p_start"], float)
        self.v_c = np.asarray(plan["v_cruise"], float)
        self.v_rel = np.asarray(plan["v_release"], float)
        self.t_rel = float(plan["t_release"])
        self.t_acc = float(plan["t_accel"])
        self.t1 = float(plan["t1"])

    def vel(self, t: float) -> np.ndarray:
        t = float(t)
        if t <= self.t1:
            return self.v_c.copy()
        if t >= self.t_rel:
            return self.v_rel.copy()
        s = (t - self.t1) / self.t_acc
        return (1.0 - s) * self.v_c + s * self.v_rel

    def pos(self, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, self.t_rel))
        if t <= self.t1:
            return self.p_start + self.v_c * t
        tau = t - self.t1
        return self.p_start + self.v_c * self.t1 + self.v_c * tau + 0.5 * (
            (self.v_rel - self.v_c) / self.t_acc
        ) * tau * tau

    def phase_name(self, t: float) -> str:
        if t < self.t1:
            return "guided_cruise"
        if t < self.t_rel:
            return "guided_accel"
        return "free_flight"


def mjv_from_params(params: dict) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat[:] = np.asarray(params["lookat"], float)
    cam.distance = float(params["distance"])
    cam.azimuth = float(params["azimuth"])
    cam.elevation = float(params["elevation"])
    return cam


def save_rgb(path: Path, sim, cam_params: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_forward(sim.model, sim.data)
    r = mujoco.Renderer(sim.model, 480, 640)
    cam = None if cam_params is None else mjv_from_params(cam_params)
    r.update_scene(sim.data, camera=cam) if cam is not None else r.update_scene(sim.data)
    _ = r.render()
    r.update_scene(sim.data, camera=cam) if cam is not None else r.update_scene(sim.data)
    img = r.render()
    r.close()
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, img)
    except Exception:
        import matplotlib.image as mpimg

        mpimg.imsave(str(path), img)


@dataclass
class BallFlight:
    state: str = "GUIDED_APPROACH"
    n_qpos_writes: int = 0
    t_first_contact: float | None = None
    t_last_contact: float | None = None
    t_park: float | None = None
    t_release: float | None = None
    pre_pos: np.ndarray | None = None
    pre_vel: np.ndarray | None = None
    pre_Rh: np.ndarray | None = None
    contact_pos: np.ndarray | None = None
    samples: list = field(default_factory=list)
    clear_steps: int = 0

    def note(self, sim, guided: bool) -> None:
        if self.state == "PARKED_AFTER_IMPACT":
            return
        hit = ball_hits_cylinder(sim)
        st = ball_pose_twist(sim)
        if guided:
            self.state = "GUIDED_APPROACH"
        elif self.state == "GUIDED_APPROACH":
            self.state = "FREE_FLIGHT"
        if hit:
            if self.t_first_contact is None:
                self.t_first_contact = float(sim.data.time)
                self.contact_pos = np.array(sim.data.xpos[int(sim.ids.object_body)], float).copy()
            self.t_last_contact = float(sim.data.time)
            self.state = "CONTACT"
            self.clear_steps = 0
        else:
            if self.t_first_contact is None:
                self.pre_vel = st["vel"].copy()
                self.pre_pos = st["pos"].copy()
                self.pre_Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
            if self.state == "CONTACT":
                self.clear_steps += 1
                if self.clear_steps >= 10:
                    self.state = "DEPARTED"

    def maybe_park(self, sim) -> bool:
        if self.state != "DEPARTED":
            return False
        po = np.array(sim.data.xpos[sim.ids.object_body], float)
        pb = ball_pose_twist(sim)["pos"]
        if float(np.linalg.norm(po - pb)) < 0.04:
            return False
        park_impact_ball(sim)
        self.n_qpos_writes += 1
        self.state = "PARKED_AFTER_IMPACT"
        self.t_park = float(sim.data.time)
        return True


def _overlay(viewer, mode: str, phase: str, t: float, ehat, state: str | None = None) -> None:
    if viewer is None or not hasattr(viewer, "add_overlay"):
        return
    try:
        import mujoco as mj

        pos = mj.mjtGridPos.mjGRID_TOPLEFT
        viewer.add_overlay(pos, "MODE", str(mode).upper())
        viewer.add_overlay(pos, "PHASE", str(phase).upper())
        viewer.add_overlay(pos, "t", f"{t:.3f} s")
        if state:
            viewer.add_overlay(pos, "STATE", str(state).upper())
        if ehat is not None and np.isfinite(ehat):
            viewer.add_overlay(pos, "e_hat_x", f" {1e3 * float(ehat):+.1f} mm")
    except Exception:
        return


def _tilt_deg(sim) -> float:
    z = float(np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(z, -1.0, 1.0))))


def _overlay_phase(mode: str, phase: str) -> str:
    if phase == "grasp":
        return "GRASP"
    if phase in ("hold", "impact"):
        return "IMPACT"
    if phase == "continue_lift":
        return "CONTINUE LIFT"
    if mode == "heuristic":
        return "RECOVERING" if phase == "recovery" else "RECOVERED"
    if phase in ("recovery", "observe"):
        return "DISTURBED"
    return str(phase).upper()


def _eval_fields(o: dict, clock: dict, t: float, mode: str, ctrl) -> dict:
    return {
        "t": float(t),
        "e_x_GT": float(o["e_x"]),
        "v_rel_GT": float(o["v_rel"]),
        "omega_rel_GT": float(o["w_rel"]),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "obj_z": float(o["obj_z"]),
        "scene": int(o["scene"]),
        "recovered_instant": bool(recovered(o)),
        "success_GT": bool(clock.get("success_GT")),
        "failure_GT": bool(clock.get("failure_GT")),
        "fail_kind_GT": clock.get("fail_kind_GT") or "",
        "heuristic_modes": list(ctrl.s.modes) if mode == "heuristic" else ["ZERO"],
    }


def run_episode(
    mode: str,
    demo_cfg: dict | None = None,
    *,
    viewer=None,
    interactive: bool = False,
    screenshots: bool = False,
    shot_prefix: str = "",
    recovery_timeout: float | None = None,
    seed: int = 0,
    diag_probe_offsets: list[float] | None = None,
) -> dict:
    """One canonical physical episode. mode in {zero, heuristic}."""
    np.random.seed(int(seed))
    demo = demo_cfg if demo_cfg is not None else load_demo_config()
    phys = demo["physical"]
    timing = demo["timing"]
    vis = demo.get("visualization", {})
    mode = str(mode).lower()
    if mode == "rule":
        mode = "heuristic"
    hashes = frozen_hashes()
    params, _yaml = params_from_frozen()
    sim, cfg = make_demo_sim(float(phys["ball_mass"]))
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    g = np.array(sim.model.opt.gravity, float)
    T_impact = float(phys.get("desired_impact_time", timing["t_stable_airborne"] + timing["impact_margin"]))
    p_hit = np.asarray(phys["intended_contact_region"], float)
    t_free = float(phys.get("free_flight", 0.20))
    t_release = T_impact - t_free
    plan = plan_guided_release(
        p_hit,
        float(phys.get("v_hit_lateral", 1.60)),
        t_release,
        t_free,
        float(phys.get("t_accel", 0.18)),
        float(phys.get("v_cruise", 0.28)),
        g,
    )
    guide = GuidedApproach(plan)
    p0 = plan["p_start"]
    v0 = plan["v_cruise"]
    apply_ball_free_state(sim, p0, v0)
    sim.set_guide_pos(p0)
    sim.set_guide_weld(True)
    mujoco.mj_forward(sim.model, sim.data)
    ball = BallFlight()
    ball.n_qpos_writes = 1
    guided_on = True
    release_log = None
    speed_log = []
    p_obj0 = np.array(sim.data.xpos[sim.ids.object_body], float)
    cam = compute_demo_camera(p0, p_hit, p_obj0)
    holder = {"cam": cam, "v": None}

    def _on_key(key: int) -> None:
        v = holder["v"]
        if v is not None and int(key) in (82, ord("R"), ord("r")):
            apply_viewer_camera(v, holder["cam"])

    viewer_cm = None
    if interactive and viewer is None:
        import mujoco.viewer as mjviewer

        viewer_cm = mjviewer.launch_passive(sim.model, sim.data, key_callback=_on_key)
        viewer = viewer_cm.__enter__()
        holder["v"] = viewer
    if viewer is not None:
        holder["v"] = viewer
        apply_viewer_camera(viewer, cam)
        enable_viewer_flags(
            viewer,
            show_contact_points=False,
            show_contact_forces=False,
        )
    shot_events: dict = {}
    if screenshots:
        shot_events["t0"] = _sim_snapshot(sim)

    ctrl = RuleBasedRecovery(params)
    clocks = HoldClocks(dt=dt)
    phase = "grasp"
    t_stable = None
    air_extra = 0.0
    t_recovery = None
    frozen = False
    table_off = False
    last_mode = "NOMINAL"
    mid_shot = False
    pre_shot = False
    post_shot = False
    post_pack = None
    pct_log = {}
    qadr = int(sim.ball_qadr)
    prev_ball = np.array(sim.data.qpos[qadr : qadr + 3], float).copy()
    ball_moved = False
    n_manual_flight_writes = 0
    playback = float(vis.get("playback_speed", 0.25))
    pacer = RealtimePacer(playback, RENDER_HZ, dt) if viewer is not None else None
    guide_t: list[float] = []
    guide_pb: list = []
    guide_pg: list = []
    guide_vb: list = []
    cost = {
        "mj_step_s": 0.0,
        "pack_s": 0.0,
        "sync_s": 0.0,
        "n_physics": 0,
        "n_render": 0,
    }

    rec_timeout = float(
        timing["post_impact_duration"] if recovery_timeout is None else recovery_timeout
    )
    post_impact_s = float(vis.get("post_impact_seconds", 5.0))
    post_drop_s = float(vis.get("post_drop_seconds", 1.5))
    post_rec_s = float(vis.get("post_recovery_seconds", 2.0))
    v_lift = float(cfg["fsm"].get("lift_speed", 0.08))
    lift_dz = float(cfg["fsm"].get("lift_offset_z", 0.18))
    zero_settle_s = 0.20
    lift_tail_s = 0.25
    n_max = int(round((T_impact + rec_timeout + lift_dz / max(v_lift, 1e-6) + post_drop_s + 3.0) / dt))
    eval_frozen = False
    eval_report = None
    t_eval_end = None
    t_drop = None
    t_uni = None
    t_both_lost = None
    t_table = None
    vis_log = []
    t_viewer_end = None
    probe_offsets = [] if diag_probe_offsets is None else [float(x) for x in diag_probe_offsets]
    probe_done: set[int] = set()
    probes: list[dict] = []
    t_continue = None
    continue_start = None
    continue_log: list[dict] = []
    slip_hx = 0.0
    t_uni_cont = None
    t_both_cont = None
    t_lift_cmd_done = None
    lift_v = np.array([0.0, 0.0, v_lift])

    def _update_guide_for_this_physics_step() -> None:
        nonlocal guided_on, release_log
        t_now = float(sim.data.time)
        if not guided_on:
            return
        sim.set_guide_pos(guide.pos(t_now))
        if t_now + 0.5 * dt >= t_release:
            st_b = ball_pose_twist(sim)
            v_g = guide.vel(t_now)
            sim.set_guide_weld(False)
            guided_on = False
            ball.t_release = t_now
            release_log = {
                "t": t_now,
                "p_ball": st_b["pos"].tolist(),
                "v_ball": st_b["vel"].tolist(),
                "p_guide": guide.pos(t_now).tolist(),
                "v_guide": v_g.tolist(),
                "dp": float(np.linalg.norm(st_b["pos"] - guide.pos(t_now))),
                "dv": float(np.linalg.norm(st_b["vel"] - v_g)),
            }

    if viewer is not None and hasattr(viewer, "sync"):
        viewer.sync()
        if pacer is not None:
            pacer.start(float(sim.data.time))
            pacer.note_sync(float(sim.data.time))

    for _ in range(n_max):
        t = float(sim.data.time)
        if pacer is not None and pacer.wall0 is None:
            pacer.start(t)

        # Guide target is set at every physics step (dt=0.002), then exactly one mj_step.
        _update_guide_for_this_physics_step()

        t_phys0 = _time.perf_counter()
        if phase == "grasp":
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
            o = physical_pack(sim)
            if sim.fsm.phase == "lift" and sim.captured and float(o["obj_z"]) >= Z_AIR:
                air_extra += dt
                if t_stable is None and air_extra >= float(timing["stable_airborne_hold"]):
                    t_stable = t
                    freeze(sim)
                    frozen = True
                    disable_object_table(sim)
                    isolate_ball_object_only(sim)
                    table_off = True
                    phase = "hold"
            last_mode = f"NOMINAL_{sim.fsm.phase}"
        elif phase in ("hold", "impact"):
            tick_vw_park(sim, np.zeros(3), np.zeros(3), G_HOLD, gains, park=False)
            last_mode = "HOLD"
        elif phase == "continue_lift":
            tick_vw_park(
                sim,
                lift_v,
                np.zeros(3),
                params.tau_secure,
                gains,
                park=ball.state == "PARKED_AFTER_IMPACT",
            )
            last_mode = "CONTINUE_LIFT"
        else:
            o_rule = obs_from_sim(sim)
            if mode == "zero":
                cmd = {
                    "v_world": np.zeros(3),
                    "tau": params.tau_secure,
                    "r_des": np.asarray(sim.fsm.r_des, float).copy(),
                    "mode": "ZERO",
                    "freeze_p": False,
                }
                ctrl.s.mode = "ZERO"
            elif eval_frozen and eval_report is not None and eval_report["success_GT"]:
                cmd = {
                    "v_world": np.zeros(3),
                    "tau": params.tau_secure,
                    "r_des": np.asarray(sim.fsm.r_des, float).copy(),
                    "mode": str(ctrl.s.mode),
                    "freeze_p": False,
                }
            else:
                cmd = ctrl.step(o_rule, dt)
            if cmd.get("freeze_p"):
                freeze(sim)
            if cmd.get("r_des") is not None:
                sim.fsm.r_des = np.asarray(cmd["r_des"], float).copy()
            tick_vw_park(
                sim,
                cmd["v_world"],
                np.zeros(3),
                cmd["tau"],
                gains,
                park=ball.state == "PARKED_AFTER_IMPACT",
            )
            last_mode = str(cmd["mode"])
        cost["mj_step_s"] += _time.perf_counter() - t_phys0
        cost["n_physics"] += 1

        pb = np.array(sim.data.qpos[qadr : qadr + 3], float)
        if float(np.linalg.norm(pb - prev_ball)) > 1e-10:
            ball_moved = True
        prev_ball = pb.copy()
        if ball.state != "PARKED_AFTER_IMPACT":
            ball.note(sim, guided=guided_on)
            if ball.state == "DEPARTED":
                ball.maybe_park(sim)

        t = float(sim.data.time)
        st_now = ball_pose_twist(sim)
        if t <= t_release + dt:
            guide_t.append(t)
            guide_pb.append(st_now["pos"].copy())
            guide_pg.append(guide.pos(min(t, t_release)))
            guide_vb.append(st_now["vel"].copy())
        if (not speed_log) or t - speed_log[-1]["t"] >= 0.10:
            speed_log.append(
                {
                    "t": t,
                    "speed": float(np.linalg.norm(st_now["vel"])),
                    "vel": st_now["vel"].tolist(),
                    "pos": st_now["pos"].tolist(),
                    "guide_phase": guide.phase_name(t) if guided_on or t < t_release + 1e-9 else "free_flight",
                    "ball_state": ball.state,
                }
            )
        for frac, key in ((0.0, "0"), (0.25, "25"), (0.50, "50"), (0.75, "75")):
            if key not in pct_log and t >= frac * T_impact:
                pct_log[key] = {
                    "t": t,
                    "pos": st_now["pos"].tolist(),
                    "vel": st_now["vel"].tolist(),
                    "state": ball.state,
                }

        if ball.t_first_contact is not None and phase in ("grasp", "hold"):
            phase = "impact"
        if ball.t_park is not None and phase in ("grasp", "hold", "impact"):
            phase = "recovery"
            t_recovery = float(sim.data.time)
            ctrl.reset(grasp_orientation())
            freeze(sim)
            clocks = HoldClocks(dt=dt)

        t_pack0 = _time.perf_counter()
        o = physical_pack(sim)
        clock = (
            clocks.update(o, read_tactile(sim.model, sim.data, sim.ids)[1])
            if phase in ("recovery", "observe", "continue_lift")
            else {
                "success_GT": False,
                "failure_GT": False,
                "fail_kind_GT": "",
            }
        )
        cost["pack_s"] += _time.perf_counter() - t_pack0
        live_fk = (
            fail_kind(o, clocks.lost_gt)
            if phase in ("recovery", "observe", "continue_lift")
            else (fail_kind(o, 0.0) if ball.t_first_contact is not None else None)
        )
        if ball.t_first_contact is not None:
            nL, nR = int(o["nL"]), int(o["nR"])
            if t_uni is None and ((nL == 0) ^ (nR == 0)):
                t_uni = t
            if t_both_lost is None and nL == 0 and nR == 0:
                t_last_b = ball.t_last_contact
                if t_last_b is not None and t > float(t_last_b) + 0.020:
                    t_both_lost = t
            if t_drop is None and live_fk == "drop":
                t_drop = t
            kinds = [c["kind"] for c in raw_object_contacts(sim)]
            if t_table is None and any(k == "object-table" for k in kinds):
                t_table = t
            if (not vis_log) or t - vis_log[-1]["t"] >= 0.05:
                vis_log.append(
                    {
                        "t": t,
                        "e_x": float(o["e_x"]),
                        "obj_z": float(o["obj_z"]),
                        "nL": nL,
                        "nR": nR,
                        "Fn_L": float(o["Fn_L"]),
                        "Fn_R": float(o["Fn_R"]),
                        "aperture": float(o["aperture"]),
                        "v_rel": float(o["v_rel"]),
                        "w_rel": float(o["w_rel"]),
                        "v_rel_h": np.asarray(o["v_rel_h"], float).tolist(),
                        "w_rel_h": np.asarray(o["w_rel_h"], float).tolist(),
                        "tilt_z": float(np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3)[2, 2])),
                        "fail_kind": live_fk or "",
                    }
                )
            if phase == "continue_lift" and continue_start is not None:
                vr_h = np.asarray(o["v_rel_h"], float)
                slip_hx += abs(float(vr_h[0])) * dt
                nL, nR = int(o["nL"]), int(o["nR"])
                if t_uni_cont is None and ((nL == 0) ^ (nR == 0)):
                    t_uni_cont = t
                if t_both_cont is None and nL == 0 and nR == 0:
                    t_both_cont = t
                if t_lift_cmd_done is None:
                    if float(sim.fsm.p_des[2]) - float(continue_start["p_des_z"]) >= lift_dz - 1e-9:
                        t_lift_cmd_done = t
                rh = np.asarray(o["rh"], float)
                if (not continue_log) or t - continue_log[-1]["t"] >= 0.0095:
                    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
                    po = np.array(sim.data.xpos[sim.ids.object_body], float)
                    continue_log.append(
                        {
                            "t": t,
                            "hand": ph.tolist(),
                            "object": po.tolist(),
                            "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
                            "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
                            "e_x": float(rh[0]),
                            "e_y": float(rh[1]),
                            "e_z": float(rh[2]),
                            "v_rel_hand": vr_h.tolist(),
                            "omega_rel_hand": np.asarray(o["w_rel_h"], float).tolist(),
                            "tilt_deg": _tilt_deg(sim),
                            "nL": nL,
                            "nR": nR,
                            "Fn_L": float(o["Fn_L"]),
                            "Fn_R": float(o["Fn_R"]),
                            "aperture": float(o["aperture"]),
                            "tau": float(o["tau"]),
                            "obj_z": float(po[2]),
                            "hand_z": float(ph[2]),
                            "slip_hx_cum": float(slip_hx),
                            "recovered": bool(recovered(o)),
                            "fail_kind": live_fk or "",
                        }
                    )
            if probe_offsets:
                t_imp = float(ball.t_first_contact)
                for i, off in enumerate(probe_offsets):
                    if i in probe_done:
                        continue
                    if t + 0.5 * dt >= t_imp + off:
                        st_p = ball_pose_twist(sim)
                        probes.append(
                            {
                                "offset_s": off,
                                "t": t,
                                "e_x": float(o["e_x"]),
                                "v_rel": float(o["v_rel"]),
                                "w_rel": float(o["w_rel"]),
                                "v_rel_h": np.asarray(o["v_rel_h"], float).tolist(),
                                "w_rel_h": np.asarray(o["w_rel_h"], float).tolist(),
                                "Fn_L": float(o["Fn_L"]),
                                "Fn_R": float(o["Fn_R"]),
                                "nL": nL,
                                "nR": nR,
                                "aperture": float(o["aperture"]),
                                "obj_z": float(o["obj_z"]),
                                "obj_pos": np.array(sim.data.xpos[sim.ids.object_body], float).tolist(),
                                "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
                                "r_des": np.asarray(sim.fsm.r_des, float).tolist(),
                                "tau": float(o["tau"]),
                                "qpos": np.array(sim.data.qpos, float).tolist(),
                                "qvel": np.array(sim.data.qvel, float).tolist(),
                                "ball_pos": st_p["pos"].tolist(),
                                "ball_vel": st_p["vel"].tolist(),
                                "mode": last_mode,
                            }
                        )
                        probe_done.add(i)
            if not eval_frozen:
                terminal = (
                    bool(clock.get("success_GT"))
                    or bool(clock.get("failure_GT"))
                    or (t_recovery is not None and t - t_recovery >= rec_timeout)
                    or (mode == "heuristic" and ctrl.s.done)
                    or (
                        mode == "zero"
                        and t_recovery is not None
                        and t >= t_recovery + zero_settle_s
                    )
                )
                if terminal:
                    eval_frozen = True
                    t_eval_end = t
                    eval_report = _eval_fields(o, clock, t, mode, ctrl)
                    phase = "observe"
            ready_zero = (
                mode == "zero"
                and t_recovery is not None
                and t >= t_recovery + zero_settle_s
                and ball.state == "PARKED_AFTER_IMPACT"
            )
            ready_h = (
                mode == "heuristic"
                and eval_frozen
                and eval_report is not None
                and bool(eval_report["success_GT"])
            )
            if phase != "continue_lift" and t_continue is None and (ready_zero or ready_h):
                ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
                rhand = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
                sim.fsm.p_des = ph.copy()
                sim.fsm.v_cmd = lift_v.copy()
                sim.fsm.w_cmd = np.zeros(3)
                t_continue = t
                continue_start = {
                    "t": t,
                    "p_hand": ph.tolist(),
                    "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
                    "r_hand": rhand.tolist(),
                    "r_des": np.asarray(sim.fsm.r_des, float).tolist(),
                    "v_cmd": lift_v.tolist(),
                    "w_cmd": [0.0, 0.0, 0.0],
                    "tau": float(params.tau_secure),
                    "z_hand": float(ph[2]),
                    "z_obj": float(o["obj_z"]),
                    "p_des_z": float(sim.fsm.p_des[2]),
                    "lift_speed": v_lift,
                    "lift_dz": lift_dz,
                    "expected_duration_s": lift_dz / max(v_lift, 1e-9),
                    "semantics": (
                        "common nominal LIFT: p_des := p_hand (same as GraspFSM.lift_started), "
                        "then p_des += [0,0,lift_speed]*dt each physics step; w_cmd=0; "
                        "r_des held; tau=tau_secure=-fg_nom. Not RULE RESUME."
                    ),
                }
                phase = "continue_lift"
        if post_pack is None and ball.t_first_contact is not None:
            if t - ball.t_first_contact >= 0.020:
                post_pack = {
                    "t": t,
                    "e_x_GT": float(o["e_x"]),
                    "v_rel_GT": float(o["v_rel"]),
                    "omega_rel_GT": float(o["w_rel"]),
                    "v_rel_h": np.asarray(o["v_rel_h"], float).tolist(),
                    "w_rel_h": np.asarray(o["w_rel_h"], float).tolist(),
                    "Fn_L": float(o["Fn_L"]),
                    "Fn_R": float(o["Fn_R"]),
                    "aperture": float(o["aperture"]),
                    "nL": int(o["nL"]),
                    "nR": int(o["nR"]),
                    "obj_z": float(o["obj_z"]),
                    "captured_bilateral": int(o["nL"]) > 0 and int(o["nR"]) > 0,
                    "ball_contacts": dump_ball_contacts(sim),
                    "object_contacts": [c["kind"] for c in raw_object_contacts(sim)],
                    "tilt_z": float(np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3)[2, 2])),
                }

        if screenshots:
            if (not mid_shot) and t >= 0.45 * T_impact:
                shot_events["mid_flight"] = _sim_snapshot(sim)
                mid_shot = True
            if (
                (not pre_shot)
                and ball.t_first_contact is None
                and t >= T_impact - 0.04
            ):
                shot_events["pre_impact"] = _sim_snapshot(sim)
                pre_shot = True
            if (not post_shot) and ball.t_first_contact is not None and t - ball.t_first_contact >= 0.03:
                shot_events["post_impact"] = _sim_snapshot(sim)
                post_shot = True

        if viewer is not None:
            if pacer is not None and pacer.should_render(t):
                ehat = None
                if phase in ("recovery", "observe", "continue_lift"):
                    tac = read_tactile(sim.model, sim.data, sim.ids)
                    ehat = tac[1]["e_hat_x"] if tac[1].get("estimate_valid") else None
                vis_state = None
                if t_drop is not None:
                    vis_state = "DROP"
                elif phase == "continue_lift":
                    vis_state = "CONTINUE LIFT"
                elif mode == "heuristic":
                    vis_state = (
                        "RECOVERED"
                        if eval_frozen and eval_report and eval_report["success_GT"]
                        else "RECOVERING"
                    )
                elif mode == "zero":
                    vis_state = "DISTURBED"
                _overlay(viewer, mode, _overlay_phase(mode, phase), t, ehat, vis_state)
                if hasattr(viewer, "sync"):
                    t_s0 = _time.perf_counter()
                    viewer.sync()
                    cost["sync_s"] += _time.perf_counter() - t_s0
                    cost["n_render"] += 1
                    pacer.note_sync(t)
            if pacer is not None:
                pacer.wait_if_ahead(t)
            if hasattr(viewer, "is_running") and not viewer.is_running():
                break

        if phase == "continue_lift":
            t_stop = None
            if t_lift_cmd_done is not None:
                t_stop = float(t_lift_cmd_done) + lift_tail_s
            if t_drop is not None:
                extra = float(t_drop) + post_drop_s
                t_stop = extra if t_stop is None else max(t_stop, extra)
            if t_stop is not None and t >= t_stop:
                phase = "done"
                t_viewer_end = t
                break
        elif phase in ("recovery", "observe") and t_recovery is not None and eval_frozen:
            t_stop = None
            if mode == "heuristic" and eval_report and not eval_report["success_GT"]:
                t_stop = float(t_eval_end) + 0.25
            if t_drop is not None:
                extra = float(t_drop) + post_drop_s
                t_stop = extra if t_stop is None else max(t_stop, extra)
            if t_stop is not None and t >= t_stop:
                phase = "done"
                t_viewer_end = t
                break

        if sim.dropped() and phase == "grasp":
            phase = "done"
            t_viewer_end = t
            break

    t_viewer_end = float(sim.data.time) if t_viewer_end is None else t_viewer_end
    oF = physical_pack(sim)
    clock_live = {
        "success_GT": bool(clocks.ok_gt >= SUCCESS_HOLD),
        "failure_GT": fail_kind(oF, clocks.lost_gt) is not None,
        "fail_kind_GT": fail_kind(oF, clocks.lost_gt) or "",
    }
    if eval_report is None:
        eval_report = _eval_fields(oF, clock_live, t_viewer_end, mode, ctrl)
        t_eval_end = t_viewer_end
        eval_frozen = True
    pre_v = np.zeros(3) if ball.pre_vel is None else np.asarray(ball.pre_vel, float)
    pre_spd = float(np.linalg.norm(pre_v))
    mass = float(phys["ball_mass"])
    Rh = (
        np.eye(3)
        if ball.pre_Rh is None
        else np.asarray(ball.pre_Rh, float).reshape(3, 3)
    )
    v_hit_hand = Rh.T @ pre_v
    lat_mom = mass * float(v_hit_hand[0])
    if screenshots:
        shot_events["final"] = _sim_snapshot(sim)
        _write_event_screenshots(sim, cam, shot_prefix, shot_events)
        # Canonical names used by the previous demo: final_{mode}.png
        final_src = SHOT_DIR / f"{shot_prefix}final.png"
        final_named = SHOT_DIR / f"{shot_prefix}final_{mode}.png"
        if final_src.exists():
            final_named.write_bytes(final_src.read_bytes())

    guide_stats = ball_step_stats(guide_t, guide_pb, guide_pg, guide_vb)
    sync_stats = interval_stats(pacer.sync_times) if pacer is not None else None
    if cost["n_physics"]:
        inv = 1.0 / cost["n_physics"]
        cost_mean = {
            "mj_step_mean_ms": 1e3 * cost["mj_step_s"] * inv,
            "pack_mean_ms": 1e3 * cost["pack_s"] * inv,
            "sync_mean_ms": (1e3 * cost["sync_s"] / cost["n_render"]) if cost["n_render"] else 0.0,
            "n_physics": cost["n_physics"],
            "n_render": cost["n_render"],
        }
    else:
        cost_mean = cost

    _result = {
        "mode": mode,
        "seed": int(seed),
        "hashes": hashes,
        "filter": {"ball_contype": 4, "ball_conaffinity": 4, "pairs": "ball<->cylinder only"},
        "guide": "mocap ball_guide + equality weld ball_guide_weld; eq_active=0 at t_release",
        "plan": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in plan.items()},
        "release": release_log,
        "speed_log": speed_log,
        "t_stable_airborne": t_stable,
        "t_release": ball.t_release,
        "free_flight_s": None if ball.t_release is None or ball.t_first_contact is None else float(ball.t_first_contact - ball.t_release),
        "phase_final": phase,
        "ball_state": ball.state,
        "n_qpos_writes": ball.n_qpos_writes,
        "n_manual_flight_writes": n_manual_flight_writes,
        "ball_moved_via_mj_step": ball_moved,
        "p0": np.asarray(p0, float).tolist(),
        "v0": np.asarray(v0, float).tolist(),
        "p_hit_design": p_hit.tolist(),
        "T_impact_design": T_impact,
        "g": g.tolist(),
        "samples": pct_log,
        "t_first_contact": ball.t_first_contact,
        "t_last_contact": ball.t_last_contact,
        "t_park": ball.t_park,
        "t_recovery": t_recovery,
        "t_eval_end": t_eval_end,
        "t_continue": t_continue,
        "t_viewer_end": t_viewer_end,
        "eval_horizon_s": rec_timeout,
        "visualization": {
            "post_impact_seconds": post_impact_s,
            "post_drop_seconds": post_drop_s,
            "post_recovery_seconds": post_rec_s,
            "t_drop": t_drop,
            "t_first_unilateral_contact": t_uni,
            "t_both_contacts_lost": t_both_lost,
            "t_table_contact": t_table,
            "no_drop_observed": t_drop is None,
            "tail_physical": {
                "e_x_GT": float(oF["e_x"]),
                "v_rel_GT": float(oF["v_rel"]),
                "omega_rel_GT": float(oF["w_rel"]),
                "nL": int(oF["nL"]),
                "nR": int(oF["nR"]),
                "obj_z": float(oF["obj_z"]),
                "scene": int(oF["scene"]),
                "fail_kind_live": fail_kind(oF, clocks.lost_gt) or "",
            },
            "post_impact_log": vis_log,
        },
        "continue_lift": {
            "used_rule_resume": False,
            "rule_resume_audit": {
                "matches_nominal_lift": False,
                "reasons": [
                    "RESUME lasts t_resume=0.50 s only; nominal remaining lift is lift_offset_z/lift_speed",
                    "RESUME sets r_des = r_des_task (wrist snap); nominal LIFT holds r_des with w_cmd=0",
                    "RESUME hardcodes v=[0,0,0.08] then done; not a full task displacement",
                ],
            },
            "v_lift": v_lift,
            "lift_dz": lift_dz,
            "zero_settle_s": zero_settle_s,
            "start": continue_start,
            "t_lift_cmd_done": t_lift_cmd_done,
            "t_unilateral_continuation": t_uni_cont,
            "t_both_lost_continuation": t_both_cont,
            "slip_hx_cum": slip_hx,
            "log": continue_log,
        },
        "diag_probes": probes,
        "pre_contact_pos": None if ball.pre_pos is None else ball.pre_pos.tolist(),
        "pre_contact_vel_world": pre_v.tolist(),
        "pre_contact_vel": pre_v.tolist(),
        "pre_contact_speed": pre_spd,
        "v_hit_hand": v_hit_hand.tolist(),
        "v_hit_hx": float(v_hit_hand[0]),
        "v_hit_hy": float(v_hit_hand[1]),
        "v_hit_hz": float(v_hit_hand[2]),
        "lateral_momentum_hx": lat_mom,
        "momentum": mass * pre_spd,
        "kinetic_energy": 0.5 * mass * pre_spd * pre_spd,
        "ball_mass": mass,
        "post_impact": post_pack,
        "final": eval_report,
        "camera": {
            "lookat": np.asarray(cam["lookat"], float).tolist(),
            "distance": float(cam["distance"]),
            "azimuth": float(cam["azimuth"]),
            "elevation": float(cam["elevation"]),
        },
        "table_disabled": table_off,
        "frozen_hold": frozen,
        "used_final_eval_data": False,
        "playback": {
            "physics_dt": dt,
            "physics_hz": 1.0 / dt,
            "render_hz_target": RENDER_HZ,
            "playback_speed": playback,
            "guide_updated_every_physics_step": True,
            "viewer_sync": sync_stats,
            "guided_ball_motion": guide_stats,
            "loop_cost_mean": cost_mean,
        },
    }
    if viewer_cm is not None:
        viewer_cm.__exit__(None, None, None)
    return _result
