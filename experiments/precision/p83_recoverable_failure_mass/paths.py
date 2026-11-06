"""P83 — cross-class recoverable failure mass."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p83_recoverable_failure_mass"
CACHE = OUT / "cache"

P64_ACTOR = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt"
P64_CRITIC = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_critic.pt"
P82_OUT = ROOT / "results/experiments/p82_global_failure_budget"
P82_CACHE = P82_OUT / "cache" / "baseline_episodes.pkl"
P82_FREEZE = P82_OUT / "BASELINE_FREEZE.json"

SEED = 5483
T_MAX = 3100
MAX_EP_STEPS = 80
SKILLS = ("A1", "B1", "C", "D")

N_STAGE1 = 32
N_STAGE2 = 48
N_SEEDS = 1

# Expected P64 hashes from P82 freeze
ACTOR_MD5 = "9f4c50413535e4c7db3d3a868bc1b6e2"
CRITIC_MD5 = "95de94e5a5d31ba8acddbfb49a6981cb"
