#!/usr/bin/env python
"""Render one Plan A / B / C expert vs immediate-nominal side-by-side videos.

Outputs under results/videos/plan_abc_vs_nominal/.
"""

from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from training.recovery_execution import restore_ballistic
from visualization.camera import mjv_camera_from_preset
from visualization.compare_strategy_nominal import (
    CASES,
    _open_env,
    load_blob,
    load_row,
    recatch_case,
    sliding_case,
    stitch,
)
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results" / "videos" / "plan_abc_vs_nominal"
SAMPLE_S = 0.04

# One representative SUCCESS expert per plan family.
PLAN_A = ("controlled_sliding", "ball_train_v2_0070", "controlled_slide", "data/controlled_sliding/manifest.jsonl")
PLAN_B = ("vertical_recatch", "ball_train_v2_0168", "follow_early", "data/vertical_recatch/manifest.jsonl")
PLAN_C_PKL = (
    ROOT
    / "results/diagnostics/ground_plan_c_scale50/raw/accepted/gbi_010__f0b96d3bd0__SUCCESS_NEW_expert.pkl"
)


def shoot(sim, cam, renderer):
    cam.lookat[:] = np.array(sim.data.xpos[sim.ids.object_body], float)
    renderer.update_scene(sim.data, cam)
    return np.asarray(renderer.render(), np.uint8).copy()


def pair(left, right, left_text, right_text):
    return np.hstack([caption(left, left_text), caption(right, right_text)])


def pad(frames, count):
    if len(frames) >= count:
        return frames[:count]
    return frames + [frames[-1]] * (count - len(frames))


def render_ab(family: str, episode: str, strategy: str, manifest: str, label: str) -> Path:
    row = load_row(ROOT / manifest, episode, strategy)
    blob = load_blob(ROOT / row["snapshot"])
    print(f"CASE {label} {family} {episode}", flush=True)
    if family == "recatch":
        prefix, left, right, err = recatch_case(row, blob)
    else:
        env = _open_env()
        prefix, left, right, err = sliding_case(env, row, blob)
    frames = stitch(prefix, left, right, strategy)
    dest = OUT / f"{label}__{episode}__vs_nominal.mp4"
    write_frames_mp4(frames, dest, fps=25)
    shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
    print("WROTE", dest, "frames", len(frames), "qpos_err", err, flush=True)
    return dest


def render_plan_c() -> Path:
    expert = pickle.loads(PLAN_C_PKL.read_bytes())
    snap = expert.get("catch_snap") or expert["snap"]
    ref = expert["ref"]
    acts = expert["actions"] if "actions" in expert else expert["prefix_actions"]
    actions = np.asarray(acts, np.float32)
    rid = expert.get("root_id", "gbi_010")
    print(f"CASE PlanC {rid} n_act={len(actions)}", flush=True)

    sim, _ = make_ground_sim()
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    look = 0.5 * (
        np.array(sim.data.xpos[sim.ids.object_body], float)
        + np.array(sim.data.xpos[sim.ids.hand_body], float)
    )
    cam = mjv_camera_from_preset(
        {"lookat": look, "distance": 0.55, "azimuth": 135.0, "elevation": -20.0}
    )
    renderer = mujoco.Renderer(sim.model, 360, 480)

    def grab(active):
        cam.lookat[:] = 0.5 * (
            np.array(active.data.xpos[active.ids.object_body], float)
            + np.array(active.data.xpos[active.ids.hand_body], float)
        )
        renderer.update_scene(active.data, cam)
        return np.asarray(renderer.render(), np.uint8).copy()

    # Left: expert recovery + handoff + 12s nominal
    left = []
    t0 = float(sim.data.time)
    next_frame = t0
    for a in actions:
        step_legal(sim, np.asarray(a, np.float32))
        now = float(sim.data.time)
        if now + 1e-9 >= next_frame:
            left.append((grab(sim), f"PlanC recover {now - t0:.2f}s"))
            next_frame += SAMPLE_S
        if latched(sim):
            break
    apply_handoff(sim, ref)
    t_h = float(sim.data.time)
    next_frame = t_h
    while float(sim.data.time) - t_h < NOMINAL_S - 1e-9:
        nominal_step(sim)
        now = float(sim.data.time)
        if now + 1e-9 >= next_frame:
            left.append((grab(sim), f"PlanC hold {now - t_h:.2f}/12s"))
            next_frame += SAMPLE_S

    # Right: immediate nominal from same catch snap
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    right = []
    apply_handoff(sim, ref)
    t_h = float(sim.data.time)
    next_frame = t_h
    while float(sim.data.time) - t_h < NOMINAL_S - 1e-9:
        nominal_step(sim)
        now = float(sim.data.time)
        if now + 1e-9 >= next_frame:
            right.append((grab(sim), f"nominal hold {now - t_h:.2f}/12s"))
            next_frame += SAMPLE_S

    renderer.close()
    n = max(len(left), len(right))
    left = pad(left, n)
    right = pad(right, n)
    frames = [
        pair(img_l, img_r, text_l, text_r)
        for (img_l, text_l), (img_r, text_r) in zip(left, right)
    ]
    dest = OUT / f"plan_c__{rid}__vs_nominal.mp4"
    write_frames_mp4(frames, dest, fps=25)
    shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
    print("WROTE", dest, "frames", len(frames), flush=True)
    return dest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    # Plan A — controlled sliding (in-hand)
    paths.append(render_ab(*PLAN_A, label="plan_a"))
    # Plan B — vertical / airborne recatch
    paths.append(render_ab(*PLAN_B, label="plan_b"))
    # Plan C — GROUND_SCENE sidewall dynamic recatch
    paths.append(render_plan_c())
    print("DONE", flush=True)
    for p in paths:
        print(p, flush=True)


if __name__ == "__main__":
    main()
