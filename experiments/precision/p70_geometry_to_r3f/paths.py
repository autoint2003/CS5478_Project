"""P70 paths and budgets — diagnosis + no-train counterfactuals only."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p70_geometry_to_r3f"
STATE_DIR = OUT / "states"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"

P69_OUT = ROOT / "results/experiments/p69_skill_sequence"
P69_A_ACTOR = P69_OUT / "checkpoints/P69_A_control_actor.pt"
P69_B_ACTOR = P69_OUT / "checkpoints/P69_B_seq_pref_actor.pt"
P69_SPLIT = P69_OUT / "DISCOVERY_VALIDATION_SPLIT.json"
P69_HYP = P69_OUT / "FIXED_SEQUENCE_HYPOTHESIS.json"

SEED = 5478
T_MAX = 3100
MAX_EP_STEPS = 80

P69_SEQ = [56, 82, 50, 71]
APPROACH_TOKS = [56, 50, 71]  # APPROACH family members in frozen seq
GRIP_TOK = 82

# Phase 1 funnel
N_FUNNEL_ICS = 8
N_FUNNEL_ROLLOUTS = 8

# Phase 3 / 5 budgets
N_TRANSITION_STATES = 20
N_TUNE_ICS = 6
N_TUNE_ROLLOUTS = 8
N_VAL_ICS = 8
N_VAL_ROLLOUTS = 12

E_POS_THRESHOLDS_MM = (100.0, 50.0, 25.0, 12.0, 9.0)
