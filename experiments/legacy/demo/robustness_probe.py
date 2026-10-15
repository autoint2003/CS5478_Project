"""Determinism repeats and small legal action perturbations. Does not train."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAJ = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/trajectories"
V2 = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
OUT = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/robustness.json"
CASES = [
    ("ball_train_v2_0016", "chase_lost"),
    ("ball_train_v2_0020", "chase_lost"),
    ("ball_train_v2_0035", "chase_lost"),
    ("ball_train_v2_0024", "chase_lost"),
    ("ball_train_v2_0025", "chase_lost"),
    ("ball_train_v2_0021", "release_long"),
    ("ball_train_v2_0002", "release_long"),
]


def main():
    import torch
    from training.autonomous_episode import make_training_env
    from training.k36_checkpoint import load_models
    from experiments.sliding.sliding_vs_regrasp import _replay_actions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(
        ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt",
        ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt",
        device,
    )
    env = make_training_env(device, model=model)
    blobs = {}
    for path in V2.glob("ball_train_v2_0002.pkl"):
        pass
    for path in V2.glob("*.pkl"):
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        if blob["spec"]["id"] in {item[0] for item in CASES}:
            blobs[blob["spec"]["id"]] = blob
    report = []
    for episode_id, name in CASES:
        actions = np.load(TRAJ / f"{episode_id}__{name}.npz")["actions"].astype(np.float32)
        exact = []
        for _repeat in range(3):
            result = _replay_actions(env, blobs[episode_id], actions.copy(), "regrasp", None)
            exact.append(result.get("existing_label"))
        perturbed = []
        for sigma in (1e-6, 1e-4, 1e-3):
            labels = []
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                noisy = np.clip(actions + rng.normal(0.0, sigma, size=actions.shape).astype(np.float32), -1.0, 1.0).astype(np.float32)
                result = _replay_actions(env, blobs[episode_id], noisy, "regrasp", None)
                labels.append(result.get("existing_label"))
            perturbed.append({"sigma": sigma, "labels": labels, "success": sum(1 for label in labels if label == "SUCCESS_12S")})
        item = {"id": episode_id, "strategy": name, "exact": exact, "perturbed": perturbed}
        report.append(item)
        print(episode_id, exact, [(row["sigma"], row["success"]) for row in perturbed], flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
