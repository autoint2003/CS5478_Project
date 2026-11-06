"""P75 paths — Critic-only contact-stage calibration; Actor/V2 frozen."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p75_critic_calibration"
CKPT_DIR = OUT / "checkpoints"
DATA_DIR = OUT / "data"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"

SEED = 5478
T_MAX = 3100
MAX_EP_STEPS = 80
WINDOW = 160
GAMMA = 1.0
GAE_LAMBDA = 0.95

# Collection budget (~2048–4096 episodes)
N_TARGET_EP = 2048
N_EVAL_FRAC_ICS = 0.25  # IC-disjoint holdout for Critic eval
C_FRACTION = 0.60  # enrich C contact-formation coverage

# Critic training
CRITIC_LR = 1e-4
CRITIC_EPOCHS = 12
MINIBATCH = 64
MAX_GRAD_NORM = 1.0
WEIGHTS_B = (2.0, 4.0)  # contact-stage loss weights (plus 1.0 = Branch A)
FC_STAGE_NAMES = ("fc_to_opposing", "fc_to_contact_loss")
