"""P69 paths and budgets."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p69_skill_sequence"
STATE_DIR = OUT / "states"
CKPT_DIR = OUT / "checkpoints"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"
P64_STEP4_OPTIM = P64_OUT / "checkpoints/P64_step4_optim.pt"

P68_OUT = ROOT / "results/experiments/p68_precontact_geometry"

SEED = 5478
T_MAX = 3100
MAX_EP_STEPS = 80

# Phase 0
N_ICS_AUDIT = 16
N_PER_IC_AUDIT = 10

# Phase 1 (kept tractable: beam search is closed-loop expensive)
N_DISCOVERY_ICS = 10
N_VALIDATION_ICS = 10
STATES_TARGET = 64
SEQ_LENS = (2, 4)  # length-6 only on top discovery subset
BEAM_WIDTH = 3
TOP_K = 4
N_CF_SEEDS = 2
N_DISC_STATES = 28
N_VAL_STATES = 24

# Phase 3
N_HELD_OUT_ICS = 8
N_ROLLOUTS_PER_IC = 16

# Phase 4 (conditional)
TRAJ_PER_BRANCH = 2048
ICS_PER_UPDATE = 64
GROUP = 8
UPDATES = 4
