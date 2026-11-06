"""P68 paths and budgets."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p68_precontact_geometry"
STATE_DIR = OUT / "states"
TRACE_DIR = OUT / "traces"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"

SEED = 5478
T_MAX = 3100
MAX_EP_STEPS = 80

N_ICS = 12
N_PER_IC = 12

# Counterfactual
N_SNAPSHOTS = 24
TOP_K_TOKENS = 8
N_PHYS_TOKENS = 6  # book-based approach/alignment candidates
N_CF_SEEDS = 2
MULTI_STEP_STATES = 4
MULTI_STEP_DEPTH = 3
MULTI_STEP_WIDTH = 4
