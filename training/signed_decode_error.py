"""Signed float error between saved actions and K=47 prototype-plus-residual."""

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(".")
proto = np.asarray(torch.load("results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"], np.float32)
extra = np.load("results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy")
book = np.concatenate([proto, extra], axis=0)
names = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]
for episode_id in ("ball_train_v2_0016", "ball_train_v2_0020", "ball_train_v2_0035"):
    actions = np.load(f"results/diagnostics/raw/legal_recovery_v2/trajectories/{episode_id}__chase_lost.npz")["actions"].astype(np.float32)
    gap = np.max(np.abs(book[None] - actions[:, None]), axis=2)
    nearest = gap.argmin(1)
    residual = np.clip(actions - book[nearest], -0.25, 0.25)
    recon = np.clip(book[nearest] + residual, -1.0, 1.0).astype(np.float32)
    delta = recon - actions
    print(episode_id, "steps_differ", int(np.sum(np.any(delta != 0, axis=1))), "max", float(np.max(np.abs(delta))))
    for dim, name in enumerate(names):
        column = delta[:, dim]
        if np.any(column != 0):
            print(" ", name, "n", int(np.sum(column != 0)), "sum", float(column.sum()), "max", float(np.max(np.abs(column))))
