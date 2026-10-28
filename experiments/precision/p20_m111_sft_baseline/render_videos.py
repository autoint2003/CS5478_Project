"""Render a few representative P20 closed-loop videos."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p20_m111_sft_baseline.run_p20 import load_record, policy_step  # noqa: E402
from models.m111_checkpoint import load_m111  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/p20_m111_sft_baseline/videos"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, saved = load_m111(
        ROOT / "results/diagnostics/p20_m111_sft_baseline/checkpoints/A_natural/checkpoint_best.pt",
        device,
    )
    rho = saved.get("residual_rho", 0.35)
    man = list(
        csv.DictReader(
            (ROOT / "results/diagnostics/p19_sft_packaging/SFT_DATASET_MANIFEST.csv").open(encoding="utf-8")
        )
    )
    val = [r for r in man if r["split"] == "val"]
    direct = next(r for r in val if str(r.get("meta_is_direct_handoff_only")).lower() == "true")
    actionful = next(r for r in val if int(r["n_actions_continuous"]) > 100)
    env = make_env()
    cam = mjv_camera_from_preset(
        {"lookat": [0.0, 0.0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
    )

    def render_policy(row, tag, max_steps=120):
        rec = load_record(row["record_path"])
        restore_capture(env, rec["capture"])
        hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        renderer = mujoco.Renderer(env.sim.model, 480, 640)
        frames = []
        obs_seq: list = []
        prev_seq: list = []
        for _t in range(max_steps):
            renderer.update_scene(env.sim.data, cam)
            frames.append(renderer.render().copy())
            obs = np.asarray(
                observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
                np.float32,
            )
            obs_seq.append(obs)
            prev_seq.append(env._prev.copy())
            tok, _r, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
            if action is None:
                for _ in range(40):
                    step_legal(env.sim, np.zeros(7, np.float32))
                    renderer.update_scene(env.sim.data, cam)
                    frames.append(renderer.render().copy())
                break
            step_legal(env.sim, action)
            env._prev = action
        dest = OUT / f"{tag}.mp4"
        write_frames_mp4(frames, dest, fps=20)
        print(tag, len(frames), dest)

    render_policy(direct, "A_natural__direct_handoff")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in actionful["traj_id"][:48])
    render_policy(actionful, f"A_natural__actionful_{safe}")

    # Continuous expert contrast on same actionful IC
    rec = load_record(actionful["record_path"])
    restore_capture(env, rec["capture"])
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    frames = []
    acts = rec["actions"]
    step = max(1, len(acts) // 120) if len(acts) else 1
    for a in acts[::step]:
        step_legal(env.sim, np.asarray(a, np.float32))
        renderer.update_scene(env.sim.data, cam)
        frames.append(renderer.render().copy())
    write_frames_mp4(frames, OUT / "continuous_expert__actionful_contrast.mp4", fps=20)
    print("expert", len(frames), list(OUT.glob("*.mp4")))


if __name__ == "__main__":
    main()
