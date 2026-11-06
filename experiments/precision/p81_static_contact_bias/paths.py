"""P81 — static contact-stage action bias audit (diagnostic only)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p81_static_contact_bias"
CACHE = OUT / "cache"
STATE_DIR = OUT / "states"

P64_ACTOR = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt"
P64_CRITIC = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_critic.pt"
P79_OUT = ROOT / "results/experiments/p79_matched_cf_coverage"
P79_CACHE = P79_OUT / "cache"
P79_BANK = P79_OUT / "SNAP_BANK.json"
P80_OUT = ROOT / "results/experiments/p80_policy_learning_feasibility"

SEED = 5481
T_MAX = 3100
MAX_EP_STEPS = 80
TOKEN_56 = 56
TOP_K = 5

# Bounded simulation budgets (set before execution)
N_DIST_ICS = 28  # Actor distribution audit
N_CF_VALIDATE_ICS = 40  # matched Actor vs tok56 (prefer unused + P79)
N_CF_SEEDS = 2
N_TIMING_SNAPS = 16
N_FULL_EP_PER_CLASS = 8  # A/B/C/D × 2 conditions
N_FULL_EP_SEEDS = 1

PRIMARY_CANDIDATES = ("baseline", "tok_56_soft", "tok_50_approach", "tok_82_grip")
