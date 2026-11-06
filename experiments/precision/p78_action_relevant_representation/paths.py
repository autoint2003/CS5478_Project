"""P78 — action-relevant observation representation (diagnostic only)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p78_action_relevant_representation"
CACHE = OUT / "cache"

P74_AUDIT = ROOT / "results/experiments/p74_contact_learning_readiness/AUDIT_RAW.json"
P75_DATA = ROOT / "results/experiments/p75_critic_calibration/data"
P77_SENSE = ROOT / "results/experiments/p77_contact_observability/cache/sensing_diag_episodes.pkl"
P73_OUT = ROOT / "results/experiments/p73_first_contact_bilateral"
P73_CF = P73_OUT / "M111_COUNTERFACTUALS.json"
P73_SNAP = P73_OUT / "SNAP_BANK.json"
P64_ACTOR = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt"
P64_CRITIC = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_critic.pt"

SEED = 5478
VAL_FRAC = 0.30
SHORT_HIST = 3
ENG_HIST = 5
FC_WINDOWS = ("fc_to_opposing", "fc_to_contact_loss")
# Bounded CF replay
CF_KIND = "first_contact"
CF_RULES = ("baseline", "soft_approach", "grip_then_approach", "A_continue_h5")
CF_MAX_SNAPS = 24  # diagnostic budget
CF_SEEDS_PER = 1
