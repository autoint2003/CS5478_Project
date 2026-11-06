"""P58 paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p58_fast_c_grpo"
CKPT_DIR = OUT / "checkpoints"

P57_OUT = ROOT / "results/experiments/p57_post_recatch_stabilization"
P57A_CKPT = P57_OUT / "checkpoints/P57_A_epoch1.pt"
P57A_PROGRESS = P57_OUT / "P57_A_FULL_EPOCH_PROGRESS.csv"
P56_OUT = ROOT / "results/experiments/p56_continued_grpo64"
P56_C_AUDIT = P56_OUT / "C_FAILURE_AUDIT.json"

SEED = 5478
GROUP = 64
N_PASSES = 2
N_TRAIN_C = 16
VAL_C_N = 8
VAL_ABD_PER = 2
VAL_ROLLOUTS = 32
T_MAX = 3100
MAX_EP_STEPS = 80
TEMPERATURE = 1.0
