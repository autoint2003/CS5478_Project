"""P84 — Non-C (B1) mechanism discovery diagnostic."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p84_nonC_mechanism_discovery"
CACHE = OUT / "cache"

P64_ACTOR = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt"
P64_CRITIC = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_critic.pt"
P83_OUT = ROOT / "results/experiments/p83_recoverable_failure_mass"
P83_PANEL = P83_OUT / "cache" / "panel_32.pkl"
P82_FREEZE = ROOT / "results/experiments/p82_global_failure_budget" / "BASELINE_FREEZE.json"

ACTOR_MD5 = "9f4c50413535e4c7db3d3a868bc1b6e2"
CRITIC_MD5 = "95de94e5a5d31ba8acddbfb49a6981cb"

GATE_FRAC = 0.40  # >=40% of B1 failures
N_B1_FAIL_REF = 26
GATE_MIN_COUNT = 11  # ceil(0.4*26)=11
