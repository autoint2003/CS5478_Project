#!/usr/bin/env python
"""Render hard_910026 Stage-1 + Precision + short HANDOFF video (arm + cylinder visible)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.recovery_success import apply_handoff, nominal_step  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

DS = ROOT / "results/data_generation/stage2_b_precision_dominant_v1"
OUT = ROOT / "results/diagnostics/precision_expert_failure_analysis/videos"
SID = "B__diagnostic__hard_910026__5e9806de8bc40b97"
FPS = 25
DT = float(DT_POLICY)


def overlay(img: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    y = 6
    for line in lines:
        dr.text((6, y), str(line)[:100], fill=(255, 255, 255))
        y += 16
    return np.asarray(im)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim, _ = make_ground_sim()

    class E:
        pass

    env = E()
    env.sim = sim
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)

    ic = pickle.loads((DS / "ics" / f"{SID}__recovery.pkl").read_bytes())
    cont = np.load(DS / "continuous" / f"{SID}__fullchain_continuous.npz")
    acts = np.asarray(cont["actions"], np.float32)
    n_s1 = int(np.asarray(cont["stage1_end_idx"]).reshape(-1)[0]) if "stage1_end_idx" in cont.files else 37
    n_s1 = max(1, min(n_s1, len(acts)))

    r_w = mujoco.Renderer(env.sim.model, 400, 560)
    r_c = mujoco.Renderer(env.sim.model, 400, 560)
    cam_w = mjv_camera_from_preset(
        {"lookat": [0.40, 0.0, 0.42], "distance": 1.85, "azimuth": 145, "elevation": -20}
    )
    cam_c = mjv_camera_from_preset(
        {"lookat": [0.52, 0.02, 0.52], "distance": 0.58, "azimuth": 125, "elevation": -22}
    )
    frames: list[np.ndarray] = []

    def draw(scn):
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.008)
        add_capsule(scn, po - 0.06 * u, po + 0.06 * u, [0.15, 0.95, 0.25, 1.0], 0.003)

    def frame(tag: str):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.40 * ph + 0.45 * po + np.array([0.0, 0.0, 0.04])
        cam_c.lookat[:] = 0.45 * ph + 0.55 * po
        err = measure_errors(env.sim)
        lines = [
            f"hard_910026  {tag}",
            f"θ={err['dtheta_deg']:.1f}°  e_pos={err['e_pos_mm']:.1f}mm  "
            f"e=[{err['dx_mm']:.1f},{err['dy_mm']:.1f},{err['dz_mm']:.1f}]",
            f"bilateral={int(err['bilateral'])}  down={int(err['fingertips_down'])}",
            "LEFT=wide (arm+cylinder)   RIGHT=close-up grasp",
        ]
        r_w.update_scene(env.sim.data, cam_w)
        draw(r_w.scene)
        iw = overlay(np.asarray(r_w.render(), np.uint8), ["WIDE"] + lines)
        r_c.update_scene(env.sim.data, cam_c)
        draw(r_c.scene)
        icam = overlay(np.asarray(r_c.render(), np.uint8), ["CLOSE"] + lines)
        return np.concatenate([iw, icam], axis=1)

    restore_capture(env, ic["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)

    # Hold start
    img = frame("START recovery-entry")
    for _ in range(int(1.2 * FPS)):
        frames.append(img)

    # Stage-1 (Tree80 tape)
    for i, a in enumerate(acts[:n_s1]):
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
        if i % 2 == 0 or i == n_s1 - 1:
            img = frame(f"STAGE1 {i+1}/{n_s1}")
            frames.append(img)
            frames.append(img)  # slight slow-mo

    img = frame("STAGE1 exit / Precision entry")
    for _ in range(int(1.0 * FPS)):
        frames.append(img)

    # Precision tape
    for i, a in enumerate(acts[n_s1:]):
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
        # subsample: precision is long (~557 steps)
        if i % 4 == 0 or i == len(acts) - n_s1 - 1:
            img = frame(f"PRECISION {i+1}/{len(acts)-n_s1}")
            frames.append(img)

    img = frame("PRE_HANDOFF")
    for _ in range(int(1.2 * FPS)):
        frames.append(img)

    # Short nominal after handoff (show drift / 12S path)
    apply_handoff(env.sim, ic["ref"])
    for k in range(int(6.0 / DT)):
        nominal_step(env.sim)
        if k % 3 == 0:
            img = frame(f"HANDOFF+NOM t={k*DT:.1f}s")
            frames.append(img)

    img = frame("END (SUCCESS_12S, not SUCCESS_NEW)")
    for _ in range(int(1.5 * FPS)):
        frames.append(img)

    r_w.close()
    r_c.close()

    dest = OUT / "hard_910026__stage1_precision_handoff.mp4"
    path = write_frames_mp4(frames, dest, fps=FPS)
    print(f"Wrote {path}  frames={len(frames)}  duration_s={len(frames)/FPS:.1f}", flush=True)


if __name__ == "__main__":
    main()
