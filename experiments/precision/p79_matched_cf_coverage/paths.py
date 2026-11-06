"""P79 — matched CF coverage expansion (diagnostic only)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p79_matched_cf_coverage"
CACHE = OUT / "cache"
STATE_DIR = OUT / "states"

P64_ACTOR = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt"
P64_CRITIC = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_critic.pt"
P75_DATA = ROOT / "results/experiments/p75_critic_calibration/data"
P73_SNAP = ROOT / "results/experiments/p73_first_contact_bilateral/SNAP_BANK.json"
P78_FEAT = ROOT / "experiments/precision/p78_action_relevant_representation/features.py"

SEED = 5479
T_MAX = 3100
MAX_EP_STEPS = 80
ENG_HIST = 5

# Coverage targets (planning; do not fabricate independence)
N_IC_TARGET = 80
N_EVAL_IC_MIN = 30
N_PILOT_IC = 20
N_ACTIONS = 4
N_SEEDS_PILOT = 2
N_SEEDS_EXPAND = 1  # after determinism check

# Single-token candidates (primary benchmark) — from P73 families, not new M111
# baseline = frozen Actor sample (no force)
CANDIDATE_TOKENS = {
    "baseline": None,  # Actor sample
    "tok_56_soft": 56,
    "tok_50_approach": 50,
    "tok_82_grip": 82,
    "tok_40_orient": 40,
}
# Use first 4 named including baseline
PRIMARY_CANDIDATES = ("baseline", "tok_56_soft", "tok_50_approach", "tok_82_grip")

EVAL_FRAC = 0.35
