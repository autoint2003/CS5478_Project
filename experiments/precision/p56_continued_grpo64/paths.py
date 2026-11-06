"""P56 paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p56_continued_grpo64"
CKPT_DIR = OUT / "checkpoints"

P54_OUT = ROOT / "results/experiments/p54_rebalanced_grpo64"
P54_CKPT = P54_OUT / "checkpoints/P54_grpo64_epoch1.pt"
P55_OUT = ROOT / "results/experiments/p55_expert_takeover"
P55A_CKPT = P55_OUT / "checkpoints/P55_A_control_best.pt"

SEED = 5478
GROUP = 64
T_MAX = 3100
MAX_EP_STEPS = 80
TEMPERATURE = 1.0
SKILLS = ("A1", "B1", "C", "D")
