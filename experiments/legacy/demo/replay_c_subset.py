"""Replay three saved C trajectories. Does not search or train."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.ball_impact_dataset import load_episode
from experiments.sliding.sliding_vs_regrasp import _replay_actions

OUT = ROOT / "results" / "diagnostics" / "raw" / "abc_original_revalidation"
CASES = (
    ("ball_train_0003", "regrasp_lift_soft"),
    ("ball_train_0014", "regrasp_chase_lost"),
    ("ball_train_0018", "regrasp_lift_soft"),
)


def main():
    import torch
    from training.autonomous_episode import make_training_env
    from training.k36_checkpoint import load_models
    from experiments.ball_impact.privileged_ball_impact_search import PROTO_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(PROTO_PATH, PROTO_PATH.with_name("policy_A_reference.pt"), device)
    env = make_training_env(device, model=model)
    rows = []
    folder = ROOT / "results" / "diagnostics" / "raw" / "sliding_vs_regrasp" / "trajectories"
    for episode_id, tag in CASES:
        path = folder / f"{episode_id}__{tag}.npz"
        if not path.is_file():
            rows.append({"id": episode_id, "tag": tag, "missing": True})
            print("MISSING", path, flush=True)
            continue
        actions = np.load(path)["actions"]
        result = _replay_actions(env, load_episode(episode_id), actions, "regrasp", None)
        hold = result.get("hold") or {}
        row = {
            "id": episode_id,
            "tag": tag,
            "primary_10s": bool(result["primary_10s"]),
            "held_12": None if not hold else bool(hold.get("held_12")),
            "label": result.get("existing_label"),
            "zero_steps": int(result["metrics"]["zero_steps"]),
            "min_ep_m": result["metrics"].get("min_ep_m"),
        }
        rows.append(row)
        print("C", episode_id, row, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c_subset_replay.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
