"""P76 — post-hoc value calibration + observability audit (no NN / Actor training)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p76_value_observability_audit"
CACHE = OUT / "cache"

P75_OUT = ROOT / "results/experiments/p75_critic_calibration"
P75_DATA = P75_OUT / "data"
P75_CKPT = P75_OUT / "checkpoints"
P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"
P74_AUDIT = ROOT / "results/experiments/p74_contact_learning_readiness/AUDIT_RAW.json"
# P73 package path (was a dead/wrong counterfactual name)
P73_CF = ROOT / "results/experiments/p73_first_contact_bilateral"

SEED = 5476
GAMMA = 1.0
GAE_LAMBDA = 0.95
WINDOW = 160
FC_STAGE_NAMES = ("fc_to_opposing", "fc_to_contact_loss")
CALIB_VAL_FRAC = 0.25  # of P75 *train* ICs held out for calibration selection
SHORT_HIST = 3
NN_RADIUS_QUANTILE = 0.05  # local neighborhood radius as quantile of pairwise dists
