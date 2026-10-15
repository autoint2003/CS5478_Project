"""Describe the release steps that remain outside the candidate residual cap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
TRAJ = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/trajectories"
NAMES = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]


def phase_of(action, contact):
    grip = float(action[6])
    trans = np.abs(action[:3])
    horizontal = float(max(trans[0], trans[1]))
    open_grip = grip < -0.5
    if contact == "none" and open_grip and horizontal >= 0.35:
        return "airborne_or_open_lateral_chase"
    if contact == "none" and open_grip:
        return "open_while_no_contact"
    if open_grip and float(np.max(trans)) < 0.15:
        return "open_nearly_stationary"
    if open_grip and float(np.max(trans)) < 0.40:
        return "open_small_translation"
    if open_grip and horizontal >= 0.40:
        return "lateral_chase_open"
    if open_grip:
        return "open_other_translation"
    return "closed_or_partial"


def main():
    proto = np.asarray(torch.load(ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"], np.float32)
    extra = np.load(ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy")
    candidate = np.concatenate([proto, extra], axis=0)
    rows = []
    for episode_id, name in (("ball_train_v2_0024", "chase_lost"), ("ball_train_v2_0025", "chase_lost")):
        blob = np.load(TRAJ / f"{episode_id}__{name}.npz")
        actions = blob["actions"].astype(np.float32)
        obs = blob["observations"].astype(np.float32)
        gap = np.max(np.abs(candidate[None] - actions[:, None]), axis=2)
        nearest = gap.argmin(1)
        linf = gap.min(1)
        k36 = np.max(np.abs(proto[None] - actions[:, None]), axis=2).min(1)
        for index in np.flatnonzero(linf > 0.25):
            action = actions[index]
            contact = "bilateral" if obs[index, 2] > 0.5 and obs[index, 3] > 0.5 else "none" if obs[index, 2] <= 0.5 and obs[index, 3] <= 0.5 else "unilateral"
            resid = action - candidate[nearest[index]]
            rows.append({
                "id": episode_id,
                "step": int(index),
                "phase": phase_of(action, contact),
                "contact": contact,
                "action": [round(float(x), 3) for x in action],
                "nearest": int(nearest[index]),
                "residual": [round(float(x), 3) for x in resid],
                "over": [NAMES[d] for d in range(7) if abs(float(resid[d])) > 0.25],
                "k36_linf": round(float(k36[index]), 3),
                "cand_linf": round(float(linf[index]), 3),
            })
    out = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/uncovered_steps.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("steps", len(rows), flush=True)
    for row in rows:
        print(row["id"], row["step"], row["phase"], row["contact"], row["over"], row["action"], row["cand_linf"], flush=True)


if __name__ == "__main__":
    main()
