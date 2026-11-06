"""P62 paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p62_grpo_optimization"
CKPT_DIR = OUT / "checkpoints"

P57_OUT = ROOT / "results/experiments/p57_post_recatch_stabilization"
P57A_CKPT = P57_OUT / "checkpoints/P57_A_epoch1.pt"
P57A_OPT = P57_OUT / "checkpoints/P57_A_grpo64_opt.pt"
P57A_PROGRESS = P57_OUT / "P57_A_FULL_EPOCH_PROGRESS.csv"
P56_PROGRESS = ROOT / "results/experiments/p56_continued_grpo64/P56_FULL_EPOCH_PROGRESS.csv"
P59_PASS1 = ROOT / "results/experiments/p59_expanded_c_grpo/checkpoints/P59_pass1.pt"
P58_SELECTED = ROOT / "results/experiments/p58_fast_c_grpo/SELECTED_C_ICS.json"

SEED = 5478
GROUP = 64
MAX_EXTRA_EPOCHS = 3  # epoch2..epoch4
LR = 1e-5
KL_COEF = 0.05
CLIP = 0.2
TEMPERATURE = 1.0
GAMMA_RETURN = 1.0
T_MAX = 3100
MAX_EP_STEPS = 80
DEV_ROLLOUTS = 64
FINAL_ROLLOUTS = 64
PER_CLASS_DEV = 4
PER_CLASS_FINAL = 4
