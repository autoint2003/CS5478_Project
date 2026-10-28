"""Canonical paths for P39."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p39_working_hold_handoff"

P38 = ROOT / "results/diagnostics/p38_unified_motion_only_sft"
P38_CKPTS = {
    "E0": P38 / "checkpoints/E0/checkpoint_best.pt",
    "E1": P38 / "checkpoints/E1/checkpoint_best.pt",
    "E2": P38 / "checkpoints/E2/checkpoint_best.pt",
}

P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P23 = ROOT / "results/diagnostics/p23_c_d2_sft"
P35 = ROOT / "results/diagnostics/p35_precision_sft"
P21_MAN = P21 / "SFT_FULL_HORIZON_MANIFEST.json"
P23_MAN = P23 / "COMBINED_DATASET_MANIFEST.json"
P35_MAN = P35 / "PACKAGED_MANIFEST.csv"

M111_META = ROOT / (
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/"
    "SELECTED_skill_aware_hybrid_meta.json"
)
