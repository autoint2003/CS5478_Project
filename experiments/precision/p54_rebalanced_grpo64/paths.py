"""P54 paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p54_rebalanced_grpo64"
CKPT_DIR = OUT / "checkpoints"
P53_OUT = ROOT / "results/experiments/p53_independent_grpo64"
P53_MANIFEST = P53_OUT / "FULL_TRAIN_IC_MANIFEST.json"
P53_CKPT = P53_OUT / "checkpoints/P53_grpo64_epoch1.pt"
P47T_CKPT = ROOT / "results/experiments/p47_b_ordinary_expansion/checkpoints/P47T_best.pt"
P50_CKPT = ROOT / "results/experiments/p50_ppo_reward_redesign/checkpoints/P50B_best.pt"
P51_CKPT = ROOT / "results/experiments/p51_four_class_mixed_ppo/checkpoints/P51_best.pt"

SEED = 5478
GROUP = 64
T_MAX = 3100
MAX_EP_STEPS = 80
TARGET_A = 80
TARGET_B = 100
SKILLS = ("A1", "B1", "C", "D")
