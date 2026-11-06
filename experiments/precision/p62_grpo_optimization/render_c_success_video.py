"""Render a successful final-test rollout of C__gbi_022 (22/64 under P62_epoch3)."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.policy_sample import sample_step
from experiments.precision.p61_reproducible_joint_eval.eval_core import (
    load_actor,
    make_eval_env,
    rollout_seeded,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p62_grpo_optimization.eval_dev import _fresh_final_starts, _flat_starts
from experiments.precision.p62_grpo_optimization.paths import OUT as P62_OUT
from experiments.precision.p62_grpo_optimization.paths import P57A_CKPT
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

TID = "C__gbi_022_gbi_022__dbc9e653e0e4a340"
CKPT = P62_OUT / "checkpoints/P62_epoch3.pt"
VDIR = P62_OUT / "videos"
SPLIT = "final_test"


def _find_start():
    flat, _ = _flat_starts(4)
    dev_ids = {s.traj_id for s in flat}
    final_flat, _ = _fresh_final_starts(dev_ids, 4)
    for s in final_flat:
        if s.traj_id == TID:
            return s
    raise RuntimeError(f"IC not found in final_test pool: {TID}")


def _find_success_seed(env, actor, start, device) -> tuple[int, dict]:
    for i in range(64):
        seed = rollout_seed(split=SPLIT, traj_id=TID, rollout_i=i)
        out = rollout_seeded(env, actor, start, device, seed=seed)
        print(
            f"  probe i={i} seed={seed} succ={out['SUCCESS_RELAXED_12S']} "
            f"hand={out['handoff']} drop={out['drop']} steps={out['n_steps']}",
            flush=True,
        )
        if out["SUCCESS_RELAXED_12S"]:
            return seed, out
    raise RuntimeError("no successful rollout in 64 final_test seeds")


def _hand_obj(sim):
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    return ph, po


def render_success(*, device: str = "cuda") -> dict:
    VDIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)

    actor, book, rho, _ = load_actor(CKPT, device_t)
    env = make_eval_env(book, rho)
    start = _find_start()
    print(f"IC={start.traj_id} ckpt={CKPT.name}", flush=True)

    seed, meta = _find_success_seed(env, actor, start, device_t)
    print(f"=== rendering success seed={seed} ===", flush=True)

    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    sim = env.env.sim
    cam = mjv_camera_from_preset(
        {"lookat": [0.55, 0.0, 0.55], "distance": 0.65, "azimuth": 135, "elevation": -22}
    )
    renderer = mujoco.Renderer(sim.model, 480, 640)
    frames: list[np.ndarray] = []

    def grab(tag: str):
        ph, po = _hand_obj(sim)
        cam.lookat[:] = 0.5 * (ph + po)
        renderer.update_scene(sim.data, cam)
        img = np.asarray(renderer.render(), np.uint8).copy()
        frames.append(
            caption(
                img,
                f"P62_ep3 {TID[-24:]} {tag} t={float(sim.data.time):.2f}s",
            )
        )

    grab("START")
    done = False
    info = {}
    step_i = 0
    while not done:
        grab(f"step{step_i}")
        tok, resid, _lp, _m = sample_step(actor, env.obs_seq, env.prev_seq, device_t)
        r, done, info = env.step(int(tok), resid)
        step_i += 1

    # Final pose after HANDOFF+12s evaluation inside last step
    ok = int(info.get("SUCCESS_RELAXED_12S") or 0)
    for k in range(40):
        grab(f"END_ok{ok}_{k}")

    renderer.close()
    out_path = VDIR / f"P62_epoch3_final_{TID}_success_seed{seed}.mp4"
    wrote = write_frames_mp4(frames, out_path, fps=20)
    summary = {
        "traj_id": TID,
        "ckpt": str(CKPT),
        "split": SPLIT,
        "seed": seed,
        "SUCCESS_RELAXED_12S": ok,
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "n_policy_steps": step_i,
        "n_frames": len(frames),
        "video": wrote or str(out_path),
        "probe_meta": {k: meta[k] for k in ("SUCCESS_RELAXED_12S", "n_steps", "return", "fail_reason")},
    }
    (VDIR / f"P62_epoch3_final_{TID}_success_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    render_success(device=args.device)
