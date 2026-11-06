"""Stable evaluation seeding — never use Python's salted built-in hash()."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch

from experiments.precision.p61_reproducible_joint_eval.paths import GLOBAL_SEED, PROTOCOL_VERSION


def stable_u32(*parts: Any) -> int:
    """Deterministic 32-bit seed from protocol + parts (SHA-256)."""
    payload = PROTOCOL_VERSION.encode("utf-8") + b"|"
    for p in parts:
        payload += str(p).encode("utf-8") + b"|"
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def rollout_seed(
    *,
    global_seed: int = GLOBAL_SEED,
    protocol: str = PROTOCOL_VERSION,
    split: str,
    traj_id: str,
    rollout_i: int,
) -> int:
    """IC-specific rollout seed shared across all checkpoints for the same (split, traj, i)."""
    # Include protocol string explicitly so version bumps invalidate old seeds.
    return stable_u32(protocol, global_seed, split, traj_id, int(rollout_i))


def seed_all(seed: int, *, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed Python / NumPy / PyTorch. Does not rely on leftover global RNG across rollouts."""
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    meta = {
        "seed": seed,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "torch_deterministic_attempted": bool(deterministic_torch),
    }
    if deterministic_torch:
        # Best-effort; some CUDA kernels remain nondeterministic.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            meta["deterministic_algorithms_error"] = str(e)[:200]
    return meta


def build_seed_manifest(
    *,
    heldout_ids: list[tuple[str, str]],
    val_c_ids: list[str],
    n_heldout: int,
    n_val: int,
) -> dict[str, Any]:
    heldout_seeds = []
    for skill, tid in heldout_ids:
        for i in range(n_heldout):
            heldout_seeds.append(
                {
                    "split": "heldout",
                    "skill": skill,
                    "traj_id": tid,
                    "rollout_i": i,
                    "seed": rollout_seed(
                        split="heldout", traj_id=tid, rollout_i=i
                    ),
                }
            )
    val_seeds = []
    for tid in val_c_ids:
        for i in range(n_val):
            val_seeds.append(
                {
                    "split": "val_c",
                    "skill": "C",
                    "traj_id": tid,
                    "rollout_i": i,
                    "seed": rollout_seed(split="val_c", traj_id=tid, rollout_i=i),
                }
            )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "global_seed": GLOBAL_SEED,
        "n_heldout_rollouts": n_heldout,
        "n_val_c_rollouts": n_val,
        "n_heldout_seed_entries": len(heldout_seeds),
        "n_val_c_seed_entries": len(val_seeds),
        "note": (
            "Seeds are SHA-256 derived and identical across checkpoints for the same "
            "(split, traj_id, rollout_i). Built-in hash() is never used."
        ),
        "heldout_seeds": heldout_seeds,
        "val_c_seeds": val_seeds,
    }
