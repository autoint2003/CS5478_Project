"""P55 paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p55_expert_takeover"
CKPT_DIR = OUT / "checkpoints"
CAPTURE_DIR = OUT / "captures"
REPLAY_DIR = OUT / "replay"

P54_OUT = ROOT / "results/experiments/p54_rebalanced_grpo64"
P54_CKPT = P54_OUT / "checkpoints/P54_grpo64_epoch1.pt"
P54_PROGRESS = P54_OUT / "P54_FULL_EPOCH_PROGRESS.csv"
P54_MANIFEST = P54_OUT / "P54_TRAIN_IC_MANIFEST.json"
P47T_CKPT = ROOT / "results/experiments/p47_b_ordinary_expansion/checkpoints/P47T_best.pt"

SEED = 5478
GROUP = 64
T_MAX = 3100
MAX_EP_STEPS = 80
TEMPERATURE = 1.0
# Candidate collection
N_PROBE_ROLLOUTS = 8  # policy rollouts per 0/64 IC to harvest states
MAX_CANDIDATES_PER_IC = 4
CAPTURE_STRIDE = 5
SKILLS = ("A1", "B1", "C", "D")
