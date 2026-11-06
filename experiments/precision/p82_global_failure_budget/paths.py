"""P82 — global failure budget and research reprioritization."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p82_global_failure_budget"
CACHE = OUT / "cache"

P64_ACTOR = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt"
P64_CRITIC = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_critic.pt"

SEED = 5482
T_MAX = 3100
MAX_EP_STEPS = 80

# Bounded baseline collection (no final-test)
N_PER_CLASS = 16
N_SEEDS = 1
SKILLS = ("A1", "B1", "C", "D")
