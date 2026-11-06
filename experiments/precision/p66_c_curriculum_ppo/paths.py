"""P66 paths and pilot budgets."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p66_c_curriculum_ppo"
CKPT_DIR = OUT / "checkpoints"
STATE_DIR = OUT / "states"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"
P64_STEP4_OPTIM = P64_OUT / "checkpoints/P64_step4_optim.pt"

P65_OUT = ROOT / "results/experiments/p65_ppo_continuation"
P65_STEP6_ACTOR = P65_OUT / "checkpoints/P65_step6_actor.pt"

SEED = 5478
GROUP = 8
ICS_PER_UPDATE = 64
UPDATES_PER_STEP = 4
TRAJ_PER_STEP = ICS_PER_UPDATE * GROUP * UPDATES_PER_STEP  # 2048

T_MAX = 3100
MAX_EP_STEPS = 80
MIN_REMAINING_STEPS = 12
TEMPERATURE = 1.0

# Phase-2 offsets relative to first correct sidewall step (negative = before)
STAGE_OFFSETS = {
    "early_intercept": -4,
    "approach_geometry": -2,
    "before_closing": -1,
    "before_bilateral_recatch": 0,  # at first sidewall frame (pre next-action snapshot)
}

CURRICULUM_C_FRAC_INITIAL = 0.50
N_ACTOR_PROBE = 4
N_EXPERT_FEAS = 1
CRITIC_CALIB_CONTINUATIONS = 8

# Geometry validation rollouts
GEOM_POLICY_ROLLOUTS = 8
GEOM_STAGE_ROLLOUTS = 16
