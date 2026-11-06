"""P61 paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p61_reproducible_joint_eval"

P57_OUT = ROOT / "results/experiments/p57_post_recatch_stabilization"
P57A_CKPT = P57_OUT / "checkpoints/P57_A_epoch1.pt"
P59_OUT = ROOT / "results/experiments/p59_expanded_c_grpo"
P59_PASS1 = P59_OUT / "checkpoints/P59_pass1.pt"
P59_PASS2 = P59_OUT / "checkpoints/P59_pass2.pt"
P58_OUT = ROOT / "results/experiments/p58_fast_c_grpo"
P58_SELECTED = P58_OUT / "SELECTED_C_ICS.json"

PROTOCOL_VERSION = "p61.v1"
GLOBAL_SEED = 5478
N_HELDOUT_ROLLOUTS = 64
N_VAL_ROLLOUTS = 32
PER_CLASS_HELDOUT = 4
TEMPERATURE = 1.0
T_MAX = 3100
MAX_EP_STEPS = 80
