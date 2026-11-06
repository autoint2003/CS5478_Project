"""P77 — contact geometry observability & sensing feasibility (no NN/policy training)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p77_contact_observability"
CACHE = OUT / "cache"

P74_AUDIT = ROOT / "results/experiments/p74_contact_learning_readiness/AUDIT_RAW.json"
P75_DATA = ROOT / "results/experiments/p75_critic_calibration/data"
P76_ALIAS = ROOT / "results/experiments/p76_value_observability_audit/OBSERVATION_ALIASING.json"
P73_CF = ROOT / "results/experiments/p73_first_contact_bilateral/M111_COUNTERFACTUALS.json"
P64_ACTOR = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt"

SEED = 5477
SHORT_HIST = 3
FC_WINDOWS = ("fc_to_opposing", "fc_to_contact_loss")
N_SENSE_EP = 96  # small diagnostic collection for Ft/shear upper-bound
VAL_FRAC = 0.30
