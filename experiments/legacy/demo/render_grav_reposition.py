"""Display-only helpers for gravitational reposition primitive v2.

Does not write qpos/qvel, contacts, or options. Camera is initialized once;
markers, overlay, and wall-clock pacing only after that.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from training.impact_visualization_utils import add_capsule, add_sphere


PLAYBACK_SPEED = 0.28
PRE_SLIP_PAUSE_S = 0.80
PRE_VIEW_PAUSE_S = 2.50
END_HOLD_S = 1.00
CYL_HALF = 0.03
RULER_MM = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
# Hand-frame offset so the ruler sits beside the cylinder, not on its surface.
RULER_OFFSET_H = np.array([0.0, 0.012, 0.0])

# One-time free-camera presets (world spherical). Not updated during playback.
DIAG_DISTANCE = 0.55
DIAG_AZIMUTH = 148.0
DIAG_ELEVATION = -25.0
ROBOT_DISTANCE = 0.85
ROBOT_AZIMUTH = 125.0
ROBOT_ELEVATION = -22.0


def hand_pose(sim):
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return ph, Rh


def object_pose(sim):
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    return po, Ro


def h2w(ph, Rh, p_h):
    return ph + Rh @ np.asarray(p_h, float)


def camera_preset(sim, mode: str) -> dict:
    po, _ = object_pose(sim)
    lookat = np.asarray(po, float).copy()
    mode = str(mode or "closeup").lower()
    if mode in ("robot", "wide"):
        return {
            "lookat": lookat,
            "distance": ROBOT_DISTANCE,
            "azimuth": ROBOT_AZIMUTH,
            "elevation": ROBOT_ELEVATION,
        }
    return {
        "lookat": lookat,
        "distance": DIAG_DISTANCE,
        "azimuth": DIAG_AZIMUTH,
        "elevation": DIAG_ELEVATION,
    }


def apply_camera_preset(viewer, params: dict) -> None:
    if viewer is None or not hasattr(viewer, "cam"):
        return
    cam = viewer.cam
    cam.lookat[:] = np.asarray(params["lookat"], float)
    cam.distance = float(params["distance"])
    cam.azimuth = float(params["azimuth"])
    cam.elevation = float(params["elevation"])


def init_camera_once(viz, sim) -> None:
    """Write viewer.cam only on the first call. After that the user owns the camera."""
    if viz is None or viz.get("viewer") is None:
        return
    if viz.get("_cam_inited"):
        return
    params = camera_preset(sim, viz.get("camera", "closeup"))
    apply_camera_preset(viz["viewer"], params)
    viz["_cam_preset"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in params.items()}
    viz["_cam_inited"] = True


def vis_overlay(viewer, *, phase, wrist_deg, tau, rho, drh_mm, progress_mm, v_rel_mm_s, nL, nR, extra=None) -> None:
    if viewer is None or not hasattr(viewer, "add_overlay"):
        return
    extra = extra or {}
    pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
    drh = np.asarray(drh_mm, float).reshape(3)
    viewer.add_overlay(pos, "PHASE", str(phase))
    if extra.get("visual_diag"):
        viewer.add_overlay(pos, "NOTE", "VISUAL DIAGNOSTIC — NOT CANONICAL")
    viewer.add_overlay(pos, "WRIST", f"{float(wrist_deg):.1f} deg")
    viewer.add_overlay(pos, "tau", f"{float(tau):.1f}")
    if str(phase) == "BRAKE":
        viewer.add_overlay(
            pos,
            "BRAKE",
            f"tau: {float(extra.get('tau_from', -3)):.0f} -> {float(extra.get('tau_to', -18)):.0f}",
        )
        viewer.add_overlay(pos, "trigger", f"{float(extra.get('trigger_mm', 0)):.2f} mm")
    viewer.add_overlay(pos, "rho", f"{float(rho):.2f}" if np.isfinite(rho) else "nan")
    viewer.add_overlay(pos, "d r_h.x", f"{drh[0]:.2f} mm")
    viewer.add_overlay(pos, "d r_h.z", f"{drh[2]:.2f} mm")
    viewer.add_overlay(pos, "progress s", f"{float(progress_mm):.2f} mm")
    viewer.add_overlay(pos, "|v_rel|", f"{float(v_rel_mm_s):.2f} mm/s")
    viewer.add_overlay(pos, "nL / nR", f"{int(nL)} / {int(nR)}")
    viewer.add_overlay(pos, "CAM", "mouse rotate/pan/zoom   R reset")


def draw_refs(scn, sim, refs: dict) -> None:
    """Hand-fixed ruler, axes, start ghost, relative trail. No physics writes."""
    if scn is None or refs is None:
        return
    scn.ngeom = 0
    ph, Rh = hand_pose(sim)
    rh0 = np.asarray(refs["rh0"], float)
    u = np.asarray(refs["u"], float)
    R_rel0 = np.asarray(refs["R_rel0"], float).reshape(3, 3)

    hx, hz = Rh[:, 0], Rh[:, 2]
    add_capsule(scn, ph, ph + 0.028 * hx, [1.0, 0.15, 0.15, 1.0], 0.0009)
    add_capsule(scn, ph, ph + 0.028 * hz, [0.2, 0.45, 1.0, 1.0], 0.0009)

    r0 = rh0 + RULER_OFFSET_H
    add_capsule(scn, h2w(ph, Rh, r0), h2w(ph, Rh, r0 + u * 0.006), [1.0, 0.85, 0.1, 0.90], 0.00055)
    for mm in RULER_MM:
        add_sphere(scn, h2w(ph, Rh, r0 + u * (mm * 1e-3)), [1.0, 0.75, 0.05, 1.0], 0.00055)

    axis_h = R_rel0[:, 2]
    ax = axis_h / (np.linalg.norm(axis_h) + 1e-12)
    top = h2w(ph, Rh, rh0 + ax * CYL_HALF)
    bot = h2w(ph, Rh, rh0 - ax * CYL_HALF)
    add_capsule(scn, bot, top, [0.95, 0.95, 0.95, 0.40], 0.00055)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ax, tmp)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(ax, e1)
    # Wire slightly outside the cylinder radius so it does not hide the geom.
    rad = 0.022
    for e in (e1, e2):
        add_capsule(
            scn,
            h2w(ph, Rh, rh0 - e * rad),
            h2w(ph, Rh, rh0 + e * rad),
            [0.85, 0.85, 0.9, 0.35],
            0.00045,
        )

    trail = refs.get("trail") or []
    for rh in trail[::5]:
        add_sphere(scn, h2w(ph, Rh, rh), [0.15, 0.95, 0.35, 0.45], 0.00055)

    if refs.get("rh_brake") is not None:
        pb = h2w(ph, Rh, np.asarray(refs["rh_brake"], float))
        add_sphere(scn, pb, [1.0, 0.15, 0.85, 1.0], 0.0018)
        hx = Rh[:, 0]
        hz = Rh[:, 2]
        add_capsule(scn, pb - 0.006 * hx, pb + 0.006 * hx, [1.0, 0.2, 0.85, 1.0], 0.0005)
        add_capsule(scn, pb - 0.006 * hz, pb + 0.006 * hz, [1.0, 0.2, 0.85, 1.0], 0.0005)


def viz_frame(viz, sim, *, phase, wrist_deg, tau, rho, drh_mm, progress_mm, v_rel_h, nL, nR) -> None:
    if viz is None or viz.get("viewer") is None:
        return
    viewer = viz["viewer"]
    if viz.get("refs") is not None and viz.get("pace", True):
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).T @ (
            np.array(sim.data.xpos[sim.ids.object_body], float)
            - np.array(sim.data.xpos[sim.ids.hand_body], float)
        )
        trail = viz["refs"].setdefault("trail", [])
        trail.append(rh.copy())
        nkeep = int(round(1.0 / float(sim.model.opt.timestep)))
        if len(trail) > nkeep:
            del trail[: len(trail) - nkeep]
    scn = getattr(viewer, "user_scn", None)
    draw_refs(scn, sim, viz.get("refs"))
    vis_overlay(
        viewer,
        phase=phase,
        wrist_deg=wrist_deg,
        tau=tau,
        rho=rho,
        drh_mm=drh_mm,
        progress_mm=progress_mm,
        v_rel_mm_s=float(np.linalg.norm(v_rel_h)) * 1e3,
        nL=nL,
        nR=nR,
        extra=viz.get("overlay_extra"),
    )
    if hasattr(viewer, "sync"):
        viewer.sync()
    speed = float(viz.get("playback_speed", PLAYBACK_SPEED))
    if speed > 1e-6 and viz.get("pace", True):
        dt = float(sim.model.opt.timestep)
        now = time.perf_counter()
        target = viz.get("_next_wall")
        if target is None:
            viz["_next_wall"] = now + dt / speed
        else:
            sleep = target - now
            if sleep > 0:
                time.sleep(min(sleep, 0.05))
            viz["_next_wall"] = max(target, time.perf_counter()) + dt / speed


def wall_pause(viz, seconds: float, sim, **overlay_kw) -> None:
    if viz is None or viz.get("viewer") is None:
        return
    t_end = time.perf_counter() + float(seconds)
    viz["pace"] = False
    while time.perf_counter() < t_end:
        viz_frame(viz, sim, **overlay_kw)
        time.sleep(0.03)
    viz["pace"] = True
    viz["_next_wall"] = None


def capture_hold_refs(sim, rh0, u) -> dict:
    ph, Rh = hand_pose(sim)
    po, Ro = object_pose(sim)
    R_rel0 = Rh.T @ Ro
    return {"rh0": np.asarray(rh0, float).copy(), "u": np.asarray(u, float).copy(), "R_rel0": R_rel0, "trail": []}


def make_reset_cam_callback(viz):
    def _cb(keycode):
        if int(keycode) not in (ord("R"), ord("r")):
            return
        params = viz.get("_cam_preset")
        if not params or viz.get("viewer") is None:
            return
        lookat = np.asarray(params["lookat"], float)
        apply_camera_preset(
            viz["viewer"],
            {
                "lookat": lookat,
                "distance": params["distance"],
                "azimuth": params["azimuth"],
                "elevation": params["elevation"],
            },
        )

    return _cb
