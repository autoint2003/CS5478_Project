"""Sequence coverage of revised-codebook prefixes. Does not replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/search.jsonl"
TRAJ = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/trajectories"


def main():
    proto = np.asarray(torch.load(ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"], np.float32)
    extra = np.load(ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/revised_extra.npy")
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("primary_10s")]
    loaded = []
    for row in rows:
        actions = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")["actions"].astype(np.float32)
        number = int(row["id"].rsplit("_", 1)[-1])
        loaded.append((number % 5 == 0, row["id"], row["strategy"], actions))
    print("prefix fit hold all")
    for count in (0, 1, 4, 8, 11, 15, len(extra)):
        book = np.concatenate([proto, extra[:count]], axis=0)
        scores = {"fit": [0, 0], "hold": [0, 0]}
        for held, episode_id, strategy, actions in loaded:
            key = "hold" if held else "fit"
            scores[key][1] += 1
            linf = np.max(np.abs(book[None] - actions[:, None]), axis=2).min(1)
            if float(linf.max()) <= 0.25 + 1e-6:
                scores[key][0] += 1
            if count == len(extra) and episode_id in {"ball_train_v2_0024", "ball_train_v2_0025"} and strategy == "chase_lost":
                print(episode_id, "linf", round(float(linf.max()), 4))
        print(36 + count, scores["fit"], scores["hold"], [scores["fit"][0] + scores["hold"][0], scores["fit"][1] + scores["hold"][1]])


if __name__ == "__main__":
    main()
