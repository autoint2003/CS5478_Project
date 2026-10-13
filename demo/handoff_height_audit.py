"""Replay saved C actions and record object height at basin readiness.

Does not change the evaluator. Does not train.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.recovery_success as success
from envs.physical_recovery import TABLE_DROP, Z_AIR
from training.autonomous_episode import Z_ABORT, make_training_env
from training.ball_impact_dataset import load_episode
from training.history_aware_critic_ablation import restore_capture
from training.nominal_pose_audit import errors
from training.privileged_ball_impact_search import step_legal
from training.sliding_vs_regrasp import HANDOFF_Z

OUT = ROOT / "results" / "diagnostics" / "raw" / "handoff_height_audit" / "replay.json"
CASES = (
    ("ball_train_0003", "regrasp_lift_soft"),
    ("ball_train_0014", "regrasp_chase_lost"),
    ("ball_train_0018", "regrasp_lift_soft"),
)


def scan(env, episode_id, tag):
    path = ROOT / "results" / "diagnostics" / "raw" / "sliding_vs_regrasp" / "trajectories" / f"{episode_id}__{tag}.npz"
    blob = load_episode(episode_id)
    actions = np.load(path)["actions"]
    restore_capture(env, blob["capture"])
    basin_run = 0
    ready = None
    best = None
    for action in actions:
        step_legal(env.sim, action)
        row = errors(env.sim, blob["ref"])
        if best is None or float(row["ep"]) < float(best["ep"]):
            best = {"ep": float(row["ep"]), "z": float(row["z"]), "eR": float(row["eR"]), "vrel": float(row["vrel"]), "t": float(row["t"])}
        inside = success.basin_ok(row, float(action[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        if basin_run >= success.PERSIST_STEPS and ready is None:
            ready = {
                "t": float(row["t"]),
                "z": float(row["z"]),
                "ep": float(row["ep"]),
                "eR": float(row["eR"]),
                "vrel": float(row["vrel"]),
                "nL": int(row["nL"]),
                "nR": int(row["nR"]),
                "passes_0.455": bool(float(row["z"]) >= HANDOFF_Z),
                "passes_lift_0.48": bool(float(row["z"]) >= Z_AIR),
            }
    return {
        "id": episode_id,
        "tag": tag,
        "ready": ready,
        "best_pinch": best,
    }


def main():
    from training.formal_grpo import load_models
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(
        ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_B_init.pt",
        ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_A_reference.pt",
        device,
    )
    env = make_training_env(device, model=model)
    rows = [scan(env, episode_id, tag) for episode_id, tag in CASES]
    payload = {
        "object_z_is": "cylinder body origin, world frame, metres",
        "handoff_z": HANDOFF_Z,
        "Z_ABORT": Z_ABORT,
        "Z_AIR": Z_AIR,
        "TABLE_DROP": TABLE_DROP,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
