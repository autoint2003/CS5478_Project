"""P72 paths and budgets — post-contact reliability, no training."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p72_postcontact_reliability"
STATE_DIR = OUT / "states"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"
P69_SPLIT = ROOT / "results/experiments/p69_skill_sequence/DISCOVERY_VALIDATION_SPLIT.json"

SEED = 5478
T_MAX = 3100
MAX_EP_STEPS = 80

# Phase 1–2 funnel
N_FUNNEL_ICS = 10
N_FUNNEL_ROLLOUTS = 10

# Phase 3 CF
N_SNAP_PER_KIND = 12
N_CF_SEEDS = 2
CF_HORIZONS = (3, 5)  # short bounded interventions

# Phase 6
N_VAL_ICS = 8
N_VAL_ROLLOUTS = 12
PANEL_C_ROLLOUTS = 12
