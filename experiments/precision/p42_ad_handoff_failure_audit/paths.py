from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p42_ad_handoff_failure_audit"
P41_ROLLOUTS = (
    ROOT
    / "results/diagnostics/p41_drop_detection_fix/p39_reeval/expert_validation/EXPERT_ROLLOUTS.csv"
)
