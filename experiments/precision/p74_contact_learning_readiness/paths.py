"""P74 paths — contact-learning readiness audit; no training."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p74_contact_learning_readiness"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"
P69_SPLIT = ROOT / "results/experiments/p69_skill_sequence/DISCOVERY_VALIDATION_SPLIT.json"
P73_CF = ROOT / "results/experiments/p73_first_contact_bilateral/M111_COUNTERFACTUALS.json"

SEED = 5478
T_MAX = 3100
MAX_EP_STEPS = 80

N_ICS = 10
N_ROLLOUTS = 10
# Critic MC subsample per stage
N_CRITIC_MAX = 400
