"""Render one verified vertical-cylinder recatch. Display only."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.demo_ballistic_recovery_comparison import caption
from demo.demo_grav_reposition_comparison import write_frames_mp4
from envs.physical_recovery import physical_pack
from training.capture_geometry import capture_geometry
from training.history_aware_critic_ablation import restore_capture
from training.impact_visualization_utils import mjv_camera_from_preset
from training.horizon_outcome import NOMINAL_S
from training.privileged_ball_impact_search import step_legal
from training.recovery_success import apply_handoff, nominal_step
from training.vertical_recatch_search import load_blob, make_env

OUT = ROOT / "results" / "diagnostics" / "vertical_recatch" / "VIDEOS"
EPISODE = "ball_train_v2_0070"


def _phase(t, n_l, n_r):
    if t < 0.04:
        return "OPEN"
    if t < 0.08:
        return "CLOSE"
    if t < 0.10:
        return "BRAKE"
    if t < 0.48:
        return "LIFT"
    if n_l and n_r:
        return "SETTLE"
    return "RECOVERY"


def main():
    manifest = ROOT / "data" / "vertical_recatch" / "manifest.jsonl"
    chosen = None
    for line in manifest.open(encoding="utf-8"):
        item = json.loads(line)
        if item["id"] == EPISODE and item["strategy"] == "follow_early":
            chosen = item
            break
    if chosen is None:
        raise RuntimeError(EPISODE)
    blob = load_blob(ROOT / chosen["snapshot"])
    with np.load(ROOT / chosen["continuous_npz"]) as archive:
        actions = np.asarray(archive["actions"], np.float32)
    env = make_env()
    restore_capture(env, blob["capture"])
    sim = env.sim
    look = np.array(sim.data.xpos[sim.ids.object_body], float)
    cam = mjv_camera_from_preset({"lookat": look, "distance": 0.48, "azimuth": 128.0, "elevation": -18.0})
    renderer = mujoco.Renderer(sim.model, 480, 640)
    recovery = []
    nominal_frames = []
    t0 = float(sim.data.time)

    def grab(bucket, label, clock):
        cam.lookat[:] = np.array(sim.data.xpos[sim.ids.object_body], float)
        renderer.update_scene(sim.data, cam)
        pack = physical_pack(sim)
        geo = capture_geometry(sim)
        text = (
            f"{EPISODE} {label} t={clock:.2f}s "
            f"n={int(pack['nL'])}/{int(pack['nR'])} gap={geo['inner_gap_m']*1000:.0f}mm "
            f"z={float(pack['obj_z']):.3f}"
        )
        bucket.append(caption(renderer.render(), text))

    grab(recovery, "OPEN", 0.0)
    for _ in range(8):
        recovery.append(recovery[-1])
    for action in actions:
        step_legal(sim, action)
        elapsed = float(sim.data.time) - t0
        grab(recovery, _phase(elapsed, 1, 1), elapsed)
    handoff = apply_handoff(sim, blob["ref"])
    if abs(float(handoff["qpos_max_abs"])) > 1e-12:
        raise RuntimeError("handoff edited the physical state")
    t_hand = float(sim.data.time)
    dt = float(sim.model.opt.timestep)
    stride = max(1, int(round(0.04 / dt)))
    steps = 0
    while float(sim.data.time) - t_hand < float(NOMINAL_S) - 1e-9:
        nominal_step(sim)
        steps += 1
        if steps % stride == 0:
            grab(nominal_frames, "NOMINAL", float(sim.data.time) - t_hand)
    grab(nominal_frames, "NOMINAL", float(sim.data.time) - t_hand)
    frames = []
    for frame in recovery:
        frames.extend([frame] * 6)
    frames.extend(nominal_frames)
    renderer.close()
    print(
        "nominal_elapsed", round(float(sim.data.time) - t_hand, 3),
        "recovery_frames", len(recovery),
        "nominal_frames", len(nominal_frames),
        flush=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{EPISODE}__follow_early.mp4"
    write_frames_mp4(frames, dest)
    shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
    print(dest, "frames", len(frames), flush=True)


if __name__ == "__main__":
    main()
