#!/usr/bin/env python
"""Render Mode B slow wrist reorientation with full Panda arm + cylinder visible."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.force_slip_control import sample_state  # noqa: E402
from data_generation.oracles.precision.sign_aware_axial import (  # noqa: E402
    desire_direction,
    waypoint_reorient,
)
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world, hand_R  # noqa: E402
from evaluation.success_criteria import P_TGT  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

B_POST = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl"
OUT = ROOT / "results/diagnostics/sign_aware_axial_sliding/videos/mode_b_reori_full_arm.mp4"
FPS = 25
DT = 0.02


def overlay(img: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    y = 6
    for line in lines:
        dr.text((8, y), line, fill=(255, 255, 255), font=font)
        y += 18
    return np.asarray(im)


def main():
    import pickle

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    blob = pickle.loads(B_POST.read_bytes())
    if "ref" not in blob:
        st = pickle.loads(
            (ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_stable.pkl").read_bytes()
        )
        blob = {"capture": blob["capture"], "ref": st["ref"]}

    d = desire_direction("B_inward", np.array([0.0, 0.0, 1.0]))
    print("Running Mode B reori +wy@85 rate=0.20 ...", flush=True)
    tr = waypoint_reorient(
        env,
        blob,
        d=d,
        axis_name="+wy",
        angle_deg=85.0,
        rate_scale=0.20,
        fn_target=12.5,
        dx_abort_mm=3.5,
        dtheta_abort_deg=3.0,
    )
    actions = tr.get("actions") or []
    print(
        f"  ok={tr.get('ok')} n={len(actions)} useful={tr.get('useful')} "
        f"dex={tr.get('delta_ex_mm')} dth={tr.get('delta_theta_deg')}",
        flush=True,
    )
    if not actions:
        print("No actions to render", flush=True)
        return 1

    restore_capture(env, blob["capture"])
    # Wide full-arm view + medium side view (both show whole arm)
    # Renderer(model, height, width); scene FB is typically 480×640
    r_wide = mujoco.Renderer(env.sim.model, 480, 640)
    r_side = mujoco.Renderer(env.sim.model, 480, 640)
    cam_w = mjv_camera_from_preset(
        {"lookat": [0.30, 0.0, 0.40], "distance": 2.35, "azimuth": 145.0, "elevation": -20.0}
    )
    cam_s = mjv_camera_from_preset(
        {"lookat": [0.35, 0.0, 0.38], "distance": 2.10, "azimuth": 85.0, "elevation": -14.0}
    )
    frames = []

    def base_pos():
        # Approximate base / link0
        try:
            bid = env.sim.model.body("link0").id
            return np.asarray(env.sim.data.xpos[bid], float)
        except Exception:
            return np.array([0.0, 0.0, 0.0])

    def update_cams():
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        pb = base_pos()
        # Look toward mid-arm: between base and hand/object
        mid = 0.35 * pb + 0.35 * ph + 0.30 * po
        mid[2] = max(0.35, min(0.55, mid[2]))
        span = float(np.linalg.norm(ph - pb)) + 0.55
        cam_w.lookat[:] = mid
        cam_w.distance = float(np.clip(1.45 * span, 1.9, 2.8))
        cam_s.lookat[:] = 0.35 * ph + 0.35 * po + 0.30 * pb
        cam_s.lookat[2] = max(0.28, min(0.50, float(cam_s.lookat[2])))
        cam_s.distance = float(np.clip(1.25 * span, 1.7, 2.6))

    def draw(scn):
        # Do NOT reset scn.ngeom — that would wipe the Panda meshes from update_scene.
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.008)
        add_sphere(scn, ph + Rh @ np.asarray(P_TGT, float), [1.0, 0.9, 0.15, 1.0], 0.006)
        add_capsule(scn, po - 0.06 * u, po + 0.06 * u, [0.2, 0.95, 0.25, 1.0], 0.003)
        # opening axis (hand +z) in world
        u_open_w = Rh @ np.array([0.0, 0.0, 1.0])
        add_capsule(scn, ph, ph + 0.12 * u_open_w, [1.0, 0.35, 0.1, 1.0], 0.0035)
        g_h = sample_state(env.sim)["g_h"]
        add_capsule(scn, po, po + 0.01 * (Rh @ np.asarray(g_h, float)), [0.85, 0.4, 1.0, 1.0], 0.003)

    def frame(tag: str):
        update_cams()
        s = sample_state(env.sim)
        lines = [
            f"{tag}  Mode B inward reori  +wy@85°  rate=0.20",
            f"e={np.round(s['e_signed_mm'], 1)}  θ={s['e_theta_deg']:.2f}°  e_pos={s['e_pos_mm']:.1f}mm",
            f"g_h={np.round(s['g_h'], 1)}  g·d={float(np.dot(s['g_h'], d)):.1f}  Fn={s['fn_tot']:.1f}N",
            "cyan=COM  yellow=target  green=cyl axis  orange=u_open  purple=g",
        ]
        r_wide.update_scene(env.sim.data, cam_w)
        draw(r_wide.scene)
        iw = overlay(np.asarray(r_wide.render(), np.uint8), ["FULL ARM (wide)"] + lines)
        r_side.update_scene(env.sim.data, cam_s)
        draw(r_side.scene)
        is_ = overlay(np.asarray(r_side.render(), np.uint8), ["FULL ARM (side)"] + lines)
        return np.concatenate([iw, is_], axis=1)

    print("Rendering frames...", flush=True)
    img = frame("BEFORE")
    for _ in range(int(2.0 * FPS)):
        frames.append(img)

    # Slow playback: every control step, ~3 frames → ~15–20 s total
    hold = 3
    for i, a in enumerate(actions):
        step_legal(env.sim, a)
        img = frame(f"{i + 1}/{len(actions)}")
        for _ in range(hold):
            frames.append(img)

    img = frame("AFTER")
    for _ in range(int(3.0 * FPS)):
        frames.append(img)

    r_wide.close()
    r_side.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    path = write_frames_mp4(frames, OUT, fps=FPS)
    print(f"Wrote {path}  frames={len(frames)}  duration={len(frames)/FPS:.1f}s", flush=True)
    return 0 if path else 1


if __name__ == "__main__":
    raise SystemExit(main())
