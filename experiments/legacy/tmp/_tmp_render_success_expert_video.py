"""Render one SUCCESS_NEW hard-dynamic expert to MP4."""
from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_b.vertical_recatch import make_env
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

OUT_DIR = ROOT / "results" / "expert_generation" / "k49_hard_dynamic_expansion" / "videos"
IC = "hard_910024"
TAG = "a2_raw"
SNAP = ROOT / "results" / "expert_generation" / "k49_hard_dynamic_expansion" / "snapshots" / f"{IC}.pkl"
TRAJ = ROOT / "results" / "expert_generation" / "k49_hard_dynamic_expansion" / "trajectories" / f"{IC}__{TAG}.npz"
HOLD_S = 2.5  # SUCCESS_NEW needs >=0.5s; keep video watchable


def main():
    env = make_env()
    blob = pickle.loads(SNAP.read_bytes())
    with np.load(TRAJ) as archive:
        actions = np.asarray(archive["actions"], np.float32)

    restore_capture(env, blob["capture"])
    sim = env.sim
    look = np.array(sim.data.xpos[sim.ids.object_body], float)
    cam = mjv_camera_from_preset(
        {"lookat": look, "distance": 0.52, "azimuth": 128.0, "elevation": -18.0}
    )
    renderer = mujoco.Renderer(sim.model, 480, 640)
    frames = []
    t0 = float(sim.data.time)
    dt = float(sim.model.opt.timestep)
    stride = max(1, int(round(0.04 / dt)))

    def grab(label: str, clock: float):
        cam.lookat[:] = np.array(sim.data.xpos[sim.ids.object_body], float)
        renderer.update_scene(sim.data, cam)
        pack = physical_pack(sim)
        text = (
            f"{IC} {label} t={clock:.2f}s "
            f"n={int(pack['nL'])}/{int(pack['nR'])} z={float(pack['obj_z']):.3f}"
        )
        frames.append(caption(renderer.render(), text))

    grab("START", 0.0)
    for i, action in enumerate(actions):
        step_legal(sim, action)
        if i % stride == 0 or i == len(actions) - 1:
            grab("RECOVERY", float(sim.data.time) - t0)

    apply_handoff(sim, blob["ref"])
    t_hand = float(sim.data.time)
    steps = 0
    while float(sim.data.time) - t_hand < HOLD_S - 1e-9:
        nominal_step(sim)
        steps += 1
        if steps % stride == 0:
            grab("HOLD", float(sim.data.time) - t_hand)
    grab("END", float(sim.data.time) - t_hand)
    renderer.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"{IC}__{TAG}_success.mp4"
    path = write_frames_mp4(frames, dest, fps=25)
    shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
    print("ok", path, "frames", len(frames), "actions", len(actions), flush=True)


if __name__ == "__main__":
    main()
