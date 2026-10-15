"""Side-by-side video: nominal continuation versus the verified data-generation recovery.

Left is the nominal controller with no recovery actions. Right replays the
continuous actions that passed the original 12 s hold. Physics, the action
file, and the handoff rule are not edited.
"""

from __future__ import annotations


import argparse
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from envs.outcomes.nominal_pose import capture_ref, closed_grip, errors, hand_R, rotvec
import envs.outcomes.horizon_outcome as horizon
import data_generation.rebuild_k48_recovery as rebuild
import envs.outcomes.recovery_success as success
import experiments.sliding.sliding_vs_regrasp as replay
from experiments.legacy.demo.demonstrations.demo_ballistic_recovery_comparison import caption
from experiments.legacy.demo.demonstrations.demo_grav_reposition_comparison import hstack_videos
from training.recovery_episode import Z_ABORT, make_training_env, nominal_substep
from data_generation.ball_impact_dataset import _hold_state

from visualization.impact_view import mjv_camera_from_preset

from training.unified_k48 import load_trained_k48

FPS = 25
RENDER_W, RENDER_H = 640, 480


class Recorder:
    def __init__(self, sim, renderer, cam, label):
        self.sim = sim
        self.renderer = renderer
        self.cam = cam
        self.label = label
        self.dt = 1.0 / float(FPS)
        self.next = float(sim.data.time)
        self.frames = []
        self.phase = "START"

    def maybe(self):
        t = float(self.sim.data.time)
        if t + 1e-9 < self.next:
            return
        self.renderer.update_scene(self.sim.data, self.cam)
        self.frames.append(caption(self.renderer.render(), f"{self.label}  {self.phase}  t={t:.2f}s"))
        self.next += self.dt


def camera_for(sim):
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    return mjv_camera_from_preset({
        "lookat": hand,
        "distance": 0.85,
        "azimuth": 270.0,
        "elevation": -20.0,
    })


def run_zero(env, blob, renderer, cam, seconds):
    restore_capture(env, blob["capture"])
    rec = Recorder(env.sim, renderer, cam, "ZERO nominal")
    st = _hold_state(env.sim)
    t0 = float(env.sim.data.time)
    dropped = False
    rec.maybe()
    while float(env.sim.data.time) < t0 + seconds - 1e-9:
        nominal_substep(env.sim, env.gains, st)
        row = errors(env.sim, blob["ref"])
        if (not dropped) and float(row["z"]) < Z_ABORT:
            dropped = True
        rec.phase = "DROP" if dropped else "NOMINAL"
        rec.maybe()
    return {
        "frames": rec.frames,
        "elapsed_s": float(env.sim.data.time) - t0,
        "dropped": dropped,
        "end_z": float(errors(env.sim, blob["ref"])["z"]),
    }


def run_verified(env, blob, actions, renderer, cam):
    restore_capture(env, blob["capture"])
    rec = Recorder(env.sim, renderer, cam, "chase_lost")
    original_step = replay.step_legal
    original_nominal = success.nominal_step

    def stepped(sim, action):
        out = original_step(sim, action)
        rec.phase = "RECOVERY"
        rec.maybe()
        return out

    def nominal(sim):
        original_nominal(sim)
        rec.phase = "HOLD"
        rec.maybe()

    replay.step_legal = stepped
    success.nominal_step = nominal
    try:
        result = replay._replay_actions(env, blob, actions, "regrasp", None)
    finally:
        replay.step_legal = original_step
        success.nominal_step = original_nominal
    hold = result.get("hold") or {}
    return {
        "frames": rec.frames,
        "label": result.get("existing_label"),
        "primary_10s": bool(result.get("primary_10s")),
        "held_12": hold.get("held_12"),
        "steps": int(result["metrics"]["steps"]),
    }


def run_inward_verified(env, blob, actions, renderer, cam, label="inward_reseat"):
    rebuild._restore_error(env, blob)
    rec = Recorder(env.sim, renderer, cam, label)
    original_step = rebuild.step_legal
    original_nominal = horizon.nominal_step

    def stepped(sim, action):
        out = original_step(sim, action)
        rec.phase = "RECOVERY"
        rec.maybe()
        return out

    def nominal(sim):
        original_nominal(sim)
        row = errors(env.sim, blob["ref"])
        rec.phase = "DROP" if float(row["z"]) < Z_ABORT else "HOLD"
        rec.maybe()

    rebuild.step_legal = stepped
    horizon.nominal_step = nominal
    try:
        out, _observations, executed, _contacts, qpos_error, phase_ok = rebuild.legal_verify(env, blob, actions)
    finally:
        rebuild.step_legal = original_step
        horizon.nominal_step = original_nominal
    return {
        "frames": rec.frames,
        "reason": out.get("reason"),
        "success": bool(out.get("success")),
        "steps": len(executed),
        "qpos_error": qpos_error,
        "phase_ok": phase_ok,
    }


def run_inward_zero(env, blob, renderer, cam, seconds):
    qpos_error, phase_ok = rebuild._restore_error(env, blob)
    if qpos_error > 1e-8 or not phase_ok:
        raise RuntimeError(f"snapshot restore failed: qpos {qpos_error} phase {phase_ok}")
    rec = Recorder(env.sim, renderer, cam, "ZERO nominal")
    rec.maybe()
    t0 = float(env.sim.data.time)
    original_nominal = horizon.nominal_step

    def nominal(sim):
        original_nominal(sim)
        row = errors(env.sim, blob["ref"])
        rec.phase = "DROP" if float(row["z"]) < Z_ABORT else "NOMINAL"
        rec.maybe()

    horizon.nominal_step = nominal
    try:
        out = horizon.finish_recovery(env, blob["ref"], t0, "early")
        while float(env.sim.data.time) < t0 + seconds - 1e-9:
            nominal(env.sim)
    finally:
        horizon.nominal_step = original_nominal
    return {
        "frames": rec.frames,
        "reason": out.get("reason"),
        "success": bool(out.get("success")),
        "dropped": rec.phase == "DROP",
        "end_z": float(errors(env.sim, blob["ref"])["z"]),
    }


def render_legal(episode_id, snapshot, actions_path, label, dest_name):
    with snapshot.open("rb") as handle:
        blob = pickle.load(handle)
    actions = np.load(actions_path)["actions"].astype(np.float32)
    device = torch.device("cpu")
    model, _saved = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", device)
    model.eval()
    env = make_training_env(device, model=model)
    restore_capture(env, blob["capture"])
    cam = camera_for(env.sim)
    renderer = mujoco.Renderer(env.sim.model, RENDER_H, RENDER_W)
    verified = run_inward_verified(env, blob, actions, renderer, cam, label=label)
    elapsed = max(len(verified["frames"]) / FPS, 1.0)
    zero = run_inward_zero(env, blob, renderer, cam, elapsed)
    out = ROOT / "results" / "videos" / dest_name
    path = hstack_videos(zero["frames"], verified["frames"], out)
    print(
        "ZERO", zero["reason"], "success", zero["success"],
        "end_z", round(zero["end_z"], 4), "frames", len(zero["frames"]),
        flush=True,
    )
    print(
        "RECOVERY", verified["reason"], "success", verified["success"],
        "steps", verified["steps"], "qpos", verified["qpos_error"],
        "phase", verified["phase_ok"], "frames", len(verified["frames"]),
        flush=True,
    )
    print("VIDEO", path, flush=True)


def render_inward(episode_id):
    render_legal(
        episode_id,
        ROOT / "data" / "inward_sliding" / "snapshots" / "train" / f"{episode_id}.pkl",
        ROOT / "data" / "inward_sliding" / "train" / f"{episode_id}__inward_reseat.npz",
        "inward_reseat",
        f"{episode_id}_zero_vs_inward_reseat.mp4",
    )


def render_slide(episode_id):
    render_legal(
        episode_id,
        ROOT / "data" / "snapshots" / "train" / f"{episode_id}.pkl",
        ROOT / "data" / "controlled_sliding" / "train" / f"{episode_id}__controlled_slide__verified.npz",
        "controlled_slide",
        f"{episode_id}_zero_vs_controlled_slide.mp4",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="ball_train_v2_0051")
    args = parser.parse_args()
    episode_id = args.id
    inward = ROOT / "data" / "inward_sliding" / "snapshots" / "train" / f"{episode_id}.pkl"
    slide = ROOT / "data" / "controlled_sliding" / "train" / f"{episode_id}__controlled_slide__verified.npz"
    if inward.is_file():
        render_inward(episode_id)
        return
    if slide.is_file():
        render_slide(episode_id)
        return
    snapshot = ROOT / "data" / "snapshots" / "train" / f"{episode_id}.pkl"
    actions_path = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/trajectories" / f"{episode_id}__chase_lost.npz"
    with snapshot.open("rb") as handle:
        blob = pickle.load(handle)
    actions = np.load(actions_path)["actions"].astype(np.float32)
    device = torch.device("cpu")
    model, _saved = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", device)
    model.eval()
    env = make_training_env(device, model=model)
    restore_capture(env, blob["capture"])
    cam = camera_for(env.sim)
    renderer = mujoco.Renderer(env.sim.model, RENDER_H, RENDER_W)
    verified = run_verified(env, blob, actions, renderer, cam)
    elapsed = max(len(verified["frames"]) / FPS, 1.0)
    zero = run_zero(env, blob, renderer, cam, elapsed)
    out = ROOT / "results" / "videos" / f"{episode_id}_zero_vs_chase_lost.mp4"
    path = hstack_videos(zero["frames"], verified["frames"], out)
    print(
        "ZERO", "dropped" if zero["dropped"] else "held",
        "end_z", round(zero["end_z"], 4),
        "frames", len(zero["frames"]),
        flush=True,
    )
    print(
        "RECOVERY", verified["label"],
        "primary_10s", verified["primary_10s"],
        "held_12", verified["held_12"],
        "steps", verified["steps"],
        "frames", len(verified["frames"]),
        flush=True,
    )
    print("VIDEO", path, flush=True)


if __name__ == "__main__":
    main()
