"""Display/debug helpers for impact+recovery visualization.

Does not change physics, RULE, reward, or estimator. Overlay/GT fields are
display-only. The only simulation writes are the existing freeze / tick /
park_impact_ball / disable_object_table / isolate_ball_object_only calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from controllers.jacobian_controller import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from envs.observable_reward import read_tactile
from envs.physical_recovery import (
    CONTACT_LOSS_HOLD,
    E_TOL,
    SUCCESS_HOLD,
    fail_kind,
    physical_pack,
    recovered,
)
from sensors.spatial_tactile import PAD_INNER_Y, PAD1_POS_FINGER, finger_local_to_world
from training.replay_core import BALL_R, dump_ball_contacts, park_impact_ball

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs" / "impact_visualization"
VIDEO_DIR = ROOT / "results" / "videos" / "impact_recovery"

# After last ball-cylinder contact, wait this many physics steps before park
# (existing park_impact_ball). Enough for the contact pair to clear.
PARK_AFTER_CLEAR_STEPS = 10


def tick_vw_park(sim, v_w, w_w, tau, gains, park: bool) -> None:
    """Same Cartesian tick as replay_core.tick_vw; park is optional.

    Impact flight must not park every step (tick_vw would freeze the ball
    at BALL_PARK_POS). After the experiment parks the ball, park=True
    matches recovery-replay semantics.
    """
    dt = float(sim.model.opt.timestep)
    v = np.asarray(v_w, float).reshape(3)
    w = np.asarray(w_w, float).reshape(3)
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = sim.fsm.p_des + v * dt
    sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, w, dt)
    sim.fsm.v_cmd = v.copy()
    sim.fsm.w_cmd = w.copy()
    apply_cartesian_ctrl(
        sim.model,
        sim.data,
        sim.ids,
        sim.fsm.p_des,
        sim.fsm.r_des,
        gripper_tau=tau,
        v_des=v,
        w_des=w,
        **gains,
    )
    mujoco.mj_step(sim.model, sim.data)
    if park and hasattr(sim, "park_ball"):
        sim.park_ball()


def cop_world_markers(sim, reading) -> list[tuple[np.ndarray, str]]:
    """Finger-local CoP mapped back to world (sensor frame, not raw contact.pos)."""
    out = []
    for side, ft, bid in (
        ("L", reading.left, sim.ids.left_body),
        ("R", reading.right, sim.ids.right_body),
    ):
        if not ft.valid:
            continue
        R = np.array(sim.data.xmat[int(bid)].reshape(3, 3), float)
        p = np.array(sim.data.xpos[int(bid)], float)
        local = np.array(
            [
                PAD1_POS_FINGER[0] + float(ft.u),
                PAD_INNER_Y,
                PAD1_POS_FINGER[2] + float(ft.v),
            ]
        )
        out.append((finger_local_to_world(local, R, p), side))
    return out


def raw_contact_world(sim) -> list[np.ndarray]:
    pts = []
    obj_b = int(sim.ids.object_body)
    fingers = {int(sim.ids.left_body), int(sim.ids.right_body)}
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        bodies = {b1, b2}
        if obj_b in bodies and (bodies & fingers):
            pts.append(np.array(c.pos, float).copy())
    return pts


def add_capsule(scn, p0, p1, rgba, width=0.004) -> None:
    if scn is None or scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, float),
    )
    mujoco.mjv_connector(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(width),
        np.asarray(p0, float),
        np.asarray(p1, float),
    )
    scn.ngeom += 1


def add_sphere(scn, p, rgba, size=0.004) -> None:
    if scn is None or scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0.0, 0.0]),
        np.asarray(p, float),
        np.eye(3).reshape(9),
        np.asarray(rgba, float),
    )
    scn.ngeom += 1


def draw_debug_geoms(
    scn,
    sim,
    trail_h,
    trail_o,
    reading,
    *,
    show_raw_contacts: bool = False,
    show_cop: bool = True,
    show_trails: bool = True,
    show_hand_x: bool = True,
    reset: bool = True,
) -> None:
    """Display-only markers. Does not write model/data or change contacts."""
    if scn is None:
        return
    if reset:
        scn.ngeom = 0
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    hx = Rh[:, 0]
    if show_hand_x:
        add_capsule(scn, ph, ph + 0.08 * hx, [1.0, 0.15, 0.15, 1.0], 0.0035)
        add_capsule(scn, ph, ph - 0.08 * hx, [0.2, 0.45, 1.0, 1.0], 0.0035)
        Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
        add_capsule(scn, po, po + 0.04 * Ro[:, 2], [0.2, 0.85, 0.3, 1.0], 0.0025)
    if show_raw_contacts:
        for p in raw_contact_world(sim):
            add_sphere(scn, p, [1.0, 0.92, 0.1, 1.0], 0.0035)
    if show_cop:
        for p, side in cop_world_markers(sim, reading):
            rgba = [1.0, 0.4, 0.05, 1.0] if side == "L" else [0.85, 0.1, 0.75, 1.0]
            add_sphere(scn, p, rgba, 0.0045)
    if show_trails:
        # Sparse / small so the cylinder stays visible.
        for p in trail_h[-60:][::3]:
            add_sphere(scn, p, [0.35, 0.75, 1.0, 0.22], 0.0014)
        for p in trail_o[-60:][::3]:
            add_sphere(scn, p, [0.20, 0.90, 0.35, 0.18], 0.0012)


def enable_viewer_flags(
    viewer,
    *,
    show_contact_points: bool = False,
    show_contact_forces: bool = False,
) -> None:
    opt = getattr(viewer, "opt", None)
    if opt is None:
        return
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(bool(show_contact_points))
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(bool(show_contact_forces))
    opt.frame = mujoco.mjtFrame.mjFRAME_NONE


def planned_impact_points(sim, d_launch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visualization-only target/launch points. Does not write qpos."""
    from controllers.nominal import grasp_orientation

    p_obj = np.array(sim.data.xpos[sim.ids.object_body], float).copy()
    p_obj[2] = max(float(p_obj[2]), 0.50)
    u = np.asarray(grasp_orientation(), float).reshape(3, 3)[:, 0]
    u = u / (np.linalg.norm(u) + 1e-12)
    p_launch = p_obj - u * float(d_launch)
    return p_obj, p_launch, u


def compute_camera_preset(sim, preset: str, d_launch: float = 0.30) -> dict | None:
    """Initial free-camera numbers. None means leave MuJoCo default (preset 'free')."""
    preset = str(preset or "impact").lower()
    if preset == "free":
        return None
    p_obj, p_launch, _u = planned_impact_points(sim, d_launch)
    if preset == "grasp":
        lookat = p_obj.copy()
        distance = 0.62
        azimuth = 128.0
        elevation = -28.0
    else:
        # Side/oblique: scene default azimuth is 120. Incoming ball is +Y
        # (~hand-x); azimuth 90 would look along the trajectory.
        lookat = 0.65 * p_obj + 0.35 * p_launch
        span = float(np.linalg.norm(p_launch - p_obj))
        distance = float(np.clip(3.15 * span + 0.62, 0.95, 2.25))
        azimuth = 125.0
        elevation = -24.0
    return {
        "preset": "grasp" if preset == "grasp" else "impact",
        "lookat": lookat,
        "distance": distance,
        "azimuth": azimuth,
        "elevation": elevation,
        "p_obj": p_obj,
        "p_launch": p_launch,
    }


def apply_viewer_camera(viewer, params: dict | None) -> None:
    """One-shot free-camera pose. Do not call from the physics/render loop."""
    if params is None:
        return
    cam = getattr(viewer, "cam", None)
    if cam is None:
        return
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat[:] = np.asarray(params["lookat"], float)
    cam.distance = float(params["distance"])
    cam.azimuth = float(params["azimuth"])
    cam.elevation = float(params["elevation"])


def mjv_camera_from_preset(params: dict) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat[:] = np.asarray(params["lookat"], float)
    cam.distance = float(params["distance"])
    cam.azimuth = float(params["azimuth"])
    cam.elevation = float(params["elevation"])
    return cam


@dataclass
class HoldClocks:
    ok_gt: float = 0.0
    lost_gt: float = 0.0
    ok_obs: float = 0.0
    lost_obs: float = 0.0
    prev_e_hat: float = field(default_factory=lambda: float("nan"))
    prev_e_hat_valid: bool = False
    dt: float = 0.002

    def update(self, o: dict, tac: dict) -> dict:
        dt = self.dt
        if recovered(o):
            self.ok_gt += dt
        else:
            self.ok_gt = 0.0
        fk = fail_kind(o, self.lost_gt + dt)
        if int(o["nL"]) == 0 and int(o["nR"]) == 0:
            self.lost_gt += dt
        else:
            self.lost_gt = 0.0
        success_gt = self.ok_gt >= SUCCESS_HOLD
        failure_gt = fail_kind(o, self.lost_gt) is not None

        valid = bool(tac["estimate_valid"] and tac["bilateral_valid"])
        ehat = float(tac["e_hat_x"]) if tac["estimate_valid"] else float("nan")
        edot_ok = True
        if valid and self.prev_e_hat_valid:
            edot = abs(ehat - self.prev_e_hat) / max(dt, 1e-9)
            edot_ok = edot <= 0.02
        inst_obs = valid and abs(ehat) <= E_TOL and edot_ok
        if inst_obs:
            self.ok_obs += dt
        else:
            self.ok_obs = 0.0
        both_lost = (not tac["contact_present_L"]) and (not tac["contact_present_R"])
        if both_lost:
            self.lost_obs += dt
        else:
            self.lost_obs = 0.0
        if tac["estimate_valid"]:
            self.prev_e_hat = ehat
            self.prev_e_hat_valid = True
        else:
            self.prev_e_hat_valid = False
        return {
            "success_GT": bool(success_gt),
            "failure_GT": bool(failure_gt),
            "fail_kind_GT": fk if failure_gt else "",
            "success_obs": bool(self.ok_obs >= SUCCESS_HOLD),
            "failure_obs": bool(self.lost_obs >= CONTACT_LOSS_HOLD),
        }


class EventTracker:
    def __init__(self):
        self.prev_sign = None
        self.prev_n = None
        self.prev_mode = None
        self.seen = set()
        self.lines: list[str] = []

    def emit(self, t: float, name: str, extra: str) -> None:
        line = f"[t={t:.3f}] {name}\n{extra}"
        self.lines.append(line)
        print(line, flush=True)

    def once(self, key: str, t: float, name: str, extra: str) -> None:
        if key in self.seen:
            return
        self.seen.add(key)
        self.emit(t, name, extra)

    def update(self, t, phase, mode, o, tac, clocks, first_contact, parked) -> None:
        ex = float(o["e_x"])
        eh = tac["e_hat_x"]
        extra = f"    e_x_GT = {ex * 1e3:+.1f} mm\n    e_hat_x = {float(eh) * 1e3:+.1f} mm" if tac[
            "estimate_valid"
        ] else f"    e_x_GT = {ex * 1e3:+.1f} mm\n    e_hat_x = nan"

        if first_contact:
            self.once("IMPACT", t, "IMPACT", extra)
        if parked:
            self.once("POST-IMPACT SETTLED", t, "POST-IMPACT SETTLED", extra)
        if phase == "recovery":
            self.once("RECOVERY START", t, "RECOVERY START", extra)
        if mode == "CONTROLLED_SLIP":
            self.once("FIRST SLIP", t, "FIRST SLIP", extra)

        n = int(o["nL"] > 0) + int(o["nR"] > 0)
        post = phase in ("ball_flight", "recovery", "done")
        if post and self.prev_n is not None:
            if self.prev_n > 0 and n == 0:
                self.emit(t, "CONTACT LOSS", extra)
            if self.prev_n == 0 and n > 0:
                self.emit(t, "CONTACT REGAIN", extra)
        self.prev_n = n

        sgn = int(np.sign(ex)) if abs(ex) > 1e-5 else 0
        if (
            phase == "recovery"
            and self.prev_sign is not None
            and sgn != 0
            and self.prev_sign != 0
            and sgn != self.prev_sign
        ):
            self.emit(t, "CENTER CROSSING", extra)
        if phase == "recovery" and sgn != 0:
            self.prev_sign = sgn
        elif phase != "recovery":
            self.prev_sign = None

        if phase == "recovery":
            if clocks["success_obs"]:
                self.once("OBS SUCCESS", t, "OBS SUCCESS", extra)
            if clocks["success_GT"]:
                self.once("GT SUCCESS", t, "GT SUCCESS", extra)
            if clocks["failure_obs"]:
                self.once("OBS FAILURE", t, "OBS FAILURE", extra)
            if clocks["failure_GT"]:
                self.once("GT FAILURE", t, "GT FAILURE", extra)


def overlay_dict(sim, phase, mode, o, tac, clocks, cmd, t) -> dict:
    eh = tac["e_hat_x"]
    Rh = o["Rh"]
    vw = np.asarray(cmd.get("v_world", np.zeros(3)), float)
    vh = Rh.T @ vw
    return {
        "t": float(t),
        "phase": str(phase),
        "mode": str(mode),
        "e_x_GT_mm": float(o["e_x"]) * 1e3,
        "e_hat_x_mm": float(eh) * 1e3 if tac["estimate_valid"] else float("nan"),
        "tactile_valid": bool(tac["estimate_valid"]),
        "Fn_L": float(tac["fn_L"]),
        "Fn_R": float(tac["fn_R"]),
        "u_L": float(tac["u_L"]),
        "v_L": float(tac["v_L"]),
        "u_R": float(tac["u_R"]),
        "v_R": float(tac["v_R"]),
        "contact_L": bool(tac["contact_present_L"]),
        "contact_R": bool(tac["contact_present_R"]),
        "bilateral": bool(tac["contact_present_L"] and tac["contact_present_R"]),
        "aperture_mm": float(o["aperture"]) * 1e3,
        "v_rel_GT": float(o["v_rel"]),
        "w_rel_GT": float(o["w_rel"]),
        "success_obs": clocks["success_obs"],
        "success_GT": clocks["success_GT"],
        "failure_obs": clocks["failure_obs"],
        "failure_GT": clocks["failure_GT"],
        "wrist_y_cmd": float(cmd.get("w_hy", 0.0)),
        "hand_x_vel": float(vh[0]),
        "world_z_vel": float(vw[2]),
        "grip_ctrl": float(cmd.get("tau", sim.data.ctrl[7] if sim.data.ctrl.size > 7 else 0.0)),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
    }


def format_overlay(d: dict) -> str:
    eh = d["e_hat_x_mm"]
    eh_s = f"{eh:+.1f} mm" if np.isfinite(eh) else "nan"
    return (
        f"t={d['t']:.3f}s  phase={d['phase']}  FSM={d['mode']}\n"
        f"  e_x GT [GT]:     {d['e_x_GT_mm']:+.1f} mm\n"
        f"  e_x tactile:     {eh_s}   valid={str(d['tactile_valid']).upper()}\n"
        f"  Fn_L/R:          {d['Fn_L']:.2f} / {d['Fn_R']:.2f} N"
        f"  contact L/R={int(d['contact_L'])}/{int(d['contact_R'])}\n"
        f"  CoP L (u,v):     {d['u_L']:+.4f}, {d['v_L']:+.4f}\n"
        f"  CoP R (u,v):     {d['u_R']:+.4f}, {d['v_R']:+.4f}\n"
        f"  aperture:        {d['aperture_mm']:.1f} mm\n"
        f"  |v_rel| [GT]:    {d['v_rel_GT']:.4f} m/s\n"
        f"  |omega_rel| [GT]:{d['w_rel_GT']:.4f} rad/s\n"
        f"  wrist_y cmd:     {d['wrist_y_cmd']:+.3f} rad/s\n"
        f"  hand_x vel:      {d['hand_x_vel']:+.4f} m/s\n"
        f"  world_z vel:     {d['world_z_vel']:+.4f} m/s\n"
        f"  grip ctrl:       {d['grip_ctrl']:+.2f}\n"
        f"  success_obs={d['success_obs']}  success_GT[GT]={d['success_GT']}\n"
        f"  failure_obs={d['failure_obs']}  failure_GT[GT]={d['failure_GT']}"
    )


def snapshot_integrity(sim, modes: list[str], clocks: dict) -> dict:
    o = physical_pack(sim)
    return {
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "e_x_GT": float(o["e_x"]),
        "success_GT": bool(clocks["success_GT"]),
        "failure_GT": bool(clocks["failure_GT"]),
        "success_obs": bool(clocks["success_obs"]),
        "failure_obs": bool(clocks["failure_obs"]),
        "modes": list(modes),
        "time": float(sim.data.time),
    }


def compare_integrity(a: dict, b: dict, atol_q=1e-9, atol_e=1e-12) -> dict:
    dq = float(np.max(np.abs(a["qpos"] - b["qpos"])))
    dv = float(np.max(np.abs(a["qvel"] - b["qvel"])))
    de = abs(float(a["e_x_GT"]) - float(b["e_x_GT"]))
    ok = (
        dq <= atol_q
        and dv <= atol_q
        and de <= atol_e
        and a["success_GT"] == b["success_GT"]
        and a["failure_GT"] == b["failure_GT"]
        and a["success_obs"] == b["success_obs"]
        and a["failure_obs"] == b["failure_obs"]
        and a["modes"] == b["modes"]
    )
    return {
        "match": bool(ok),
        "max_abs_dqpos": dq,
        "max_abs_dqvel": dv,
        "abs_de_x": de,
        "success_GT": [a["success_GT"], b["success_GT"]],
        "failure_GT": [a["failure_GT"], b["failure_GT"]],
        "success_obs": [a["success_obs"], b["success_obs"]],
        "failure_obs": [a["failure_obs"], b["failure_obs"]],
        "modes_equal": a["modes"] == b["modes"],
        "n_modes": [len(a["modes"]), len(b["modes"])],
    }


def save_ex_plot(hist: dict, events: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(hist["t"], float)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, np.asarray(hist["e_x_GT_mm"], float), label="e_x GT [GT]", color="k")
    axes[0].plot(t, np.asarray(hist["e_hat_x_mm"], float), label="e_hat_x tactile", color="C1")
    axes[0].axhline(3.0, color="0.6", ls="--", lw=0.8)
    axes[0].axhline(-3.0, color="0.6", ls="--", lw=0.8)
    axes[0].set_ylabel("e_x [mm]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(t, np.asarray(hist["Fn_L"], float), label="Fn_L")
    axes[1].plot(t, np.asarray(hist["Fn_R"], float), label="Fn_R")
    axes[1].set_ylabel("Fn [N]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(t, np.asarray(hist["grip_ctrl"], float), label="grip ctrl", color="C2")
    axes[2].set_ylabel("tau")
    axes[2].set_xlabel("t [s]")
    colors = {
        "impact": "r",
        "recovery_start": "b",
        "center_crossing": "m",
        "success": "g",
        "failure": "orange",
    }
    for ax in axes:
        for k, c in colors.items():
            ts = events.get(k, [])
            for i, ti in enumerate(ts):
                ax.axvline(float(ti), color=c, ls=":", lw=0.9, label=k if i == 0 else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def try_write_mp4(frame_dir: Path, dest: Path, fps: int = 50) -> str | None:
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(dest)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
