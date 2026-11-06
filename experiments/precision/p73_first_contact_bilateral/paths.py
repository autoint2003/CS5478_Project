"""P73 paths and budgets — first-contact → opposing bilateral; no training."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p73_first_contact_bilateral"
STATE_DIR = OUT / "states"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"
P69_SPLIT = ROOT / "results/experiments/p69_skill_sequence/DISCOVERY_VALIDATION_SPLIT.json"
P72_FUNNEL = ROOT / "results/experiments/p72_postcontact_reliability/FUNNEL_RAW.json"

SEED = 5478
T_MAX = 3100
MAX_EP_STEPS = 80

# Phase 1 attribution
N_ATTR_ICS = 10
N_ATTR_ROLLOUTS = 10
FC_WINDOW = 12  # steps after first contact for attribution window

# Phase 2–3 CF
N_SNAP = 16
N_CF_SEEDS = 2
CF_MAX_STEPS = (3, 5)

# Phase 5
N_VAL_ICS = 8
N_VAL_ROLLOUTS = 12
PANEL_C_ROLLOUTS = 12
