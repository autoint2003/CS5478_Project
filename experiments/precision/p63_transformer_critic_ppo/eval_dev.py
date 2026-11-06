"""Deterministic development eval (P61 protocol) + paired delta vs P62_epoch3."""

from __future__ import annotations

from pathlib import Path

from experiments.precision.p62_grpo_optimization.eval_dev import (
    eval_checkpoint_dev,
    paired_delta,
)
from experiments.precision.p63_transformer_critic_ppo.paths import OUT, P62_EPOCH3


def eval_p62_baseline(*, device: str = "cuda") -> dict:
    return eval_checkpoint_dev("P62_epoch3", P62_EPOCH3, device=device)


def eval_actor(tag: str, ckpt: Path, *, device: str = "cuda") -> dict:
    return eval_checkpoint_dev(tag, ckpt, device=device)
