"""P80 — history-conditioned policy learning feasibility (diagnostic only)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p80_policy_learning_feasibility"
CACHE = OUT / "cache"

P79_OUT = ROOT / "results/experiments/p79_matched_cf_coverage"
P79_CACHE = P79_OUT / "cache"
P79_BANK = P79_OUT / "SNAP_BANK.json"

PRIMARY_CANDIDATES = ("baseline", "tok_56_soft", "tok_50_approach", "tok_82_grip")
SEED = 5480
