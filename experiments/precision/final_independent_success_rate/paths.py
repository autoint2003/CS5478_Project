"""Small-scale independent final evaluation — paths and freeze constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/final_independent_success_rate"
CACHE = OUT / "cache"

P64_ACTOR = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt"
P64_CRITIC = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_critic.pt"
P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
IC_EXPOSURE = P64_OUT / "IC_EXPOSURE.jsonl"
P83_PANEL = ROOT / "results/experiments/p83_recoverable_failure_mass/cache/panel_32.pkl"
P82_CACHE = ROOT / "results/experiments/p82_global_failure_budget/cache/baseline_episodes.pkl"
P82_FREEZE = ROOT / "results/experiments/p82_global_failure_budget/BASELINE_FREEZE.json"

SKILLS = ("A1", "B1", "C", "D")
TARGET_PER_CLASS = 10  # small-scale independent evaluation
N_ROLLOUTS_PER_IC = 1
MANIFEST_SEED = 5478
EVAL_SPLIT = "final_independent_small"
PROTOCOL_VERSION = "final_independent_sr.small.v1"

T_MAX = 3100
MAX_EP_STEPS = 80
