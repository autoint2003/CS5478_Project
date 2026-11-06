"""P59 paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p59_expanded_c_grpo"
CKPT_DIR = OUT / "checkpoints"

P58_OUT = ROOT / "results/experiments/p58_fast_c_grpo"
P58_PASS2 = P58_OUT / "checkpoints/P58_pass2.pt"
P58_SELECTED = P58_OUT / "SELECTED_C_ICS.json"
P57_OUT = ROOT / "results/experiments/p57_post_recatch_stabilization"
P57A_CKPT = P57_OUT / "checkpoints/P57_A_epoch1.pt"
P57A_PROGRESS = P57_OUT / "P57_A_FULL_EPOCH_PROGRESS.csv"

SEED = 5478
GROUP = 64
N_PASSES = 2
N_TRAIN_C = 48
VAL_ROLLOUTS = 32
T_MAX = 3100
MAX_EP_STEPS = 80
TEMPERATURE = 1.0
