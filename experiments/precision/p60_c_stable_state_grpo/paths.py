"""P60 paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p60_c_stable_state_grpo"
CKPT_DIR = OUT / "checkpoints"
STATE_DIR = OUT / "states"

P59_OUT = ROOT / "results/experiments/p59_expanded_c_grpo"
P59_PASS1 = P59_OUT / "checkpoints/P59_pass1.pt"
P59_PASS2 = P59_OUT / "checkpoints/P59_pass2.pt"
P59_MANIFEST = P59_OUT / "TRAIN_IC_MANIFEST.json"
P59_VAL_PASS1 = P59_OUT / "VALIDATION_after_pass1.json"
P59_VAL_PASS2 = P59_OUT / "VALIDATION_after_pass2.json"
P58_OUT = ROOT / "results/experiments/p58_fast_c_grpo"
P58_SELECTED = P58_OUT / "SELECTED_C_ICS.json"
P57_OUT = ROOT / "results/experiments/p57_post_recatch_stabilization"
P57A_CKPT = P57_OUT / "checkpoints/P57_A_epoch1.pt"

SEED = 5478
GROUP = 64
N_PASSES = 2
N_STATES_TARGET = 20  # within 16–24
N_STATES_MIN = 16
N_STATES_MAX = 24
PROBE_CONTINUATIONS = 16
MAX_TRAIN_CONTINUATIONS = 3072  # hard cap: 24 × 64 × 2

T_MAX = 3100
MAX_EP_STEPS = 80
TEMPERATURE = 1.0
VAL_ROLLOUTS = 32
MIN_REMAINING_STEPS = 10
COLLECT_ROLLOUTS_PER_IC = 4
MAX_STATES_PER_IC = 2
