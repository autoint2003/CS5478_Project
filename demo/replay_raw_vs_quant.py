"""Compare raw saved actions with quantized replay on the three regressions."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAJ = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/trajectories"
V2 = ROOT / "demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
CASES = [
    ("ball_train_v2_0016", "chase_lost"),
    ("ball_train_v2_0020", "chase_lost"),
    ("ball_train_v2_0035", "chase_lost"),
]


def main():
    import torch
    from training.autonomous_episode import make_training_env
    from training.formal_grpo import load_models
    from training.sliding_vs_regrasp import _replay_actions

    proto = np.asarray(
        torch.load(ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"],
        np.float32,
    )
    extra = np.load(ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy")
    candidate = np.concatenate([proto, extra], axis=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(
        ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt",
        ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt",
        device,
    )
    env = make_training_env(device, model=model)
    blobs = {}
    for path in V2.glob("*.pkl"):
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        blobs[blob["spec"]["id"]] = blob
    episode_id, name = CASES[0]
    actions = np.load(TRAJ / f"{episode_id}__{name}.npz")["actions"].astype(np.float32)
    gap = np.max(np.abs(candidate[None] - actions[:, None]), axis=2)
    nearest = gap.argmin(1)
    resid = np.clip(actions - candidate[nearest], -0.25, 0.25)
    recon = np.clip(candidate[nearest] + resid, -1.0, 1.0).astype(np.float32)
    print("equal", bool(np.array_equal(actions, recon)), "max", float(np.max(np.abs(actions - recon))), flush=True)
    first = _replay_actions(env, blobs[episode_id], actions.copy(), "regrasp", None)
    second = _replay_actions(env, blobs[episode_id], actions.copy(), "regrasp", None)
    third = _replay_actions(env, blobs[episode_id], recon.copy(), "regrasp", None)
    print(episode_id, first.get("existing_label"), second.get("existing_label"), third.get("existing_label"), flush=True)


if __name__ == "__main__":
    main()
