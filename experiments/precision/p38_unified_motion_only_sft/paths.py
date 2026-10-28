"""Canonical paths for P38 (read-only refs to P21/P23/P31/P35/P36)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p38_unified_motion_only_sft"

P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P23 = ROOT / "results/diagnostics/p23_c_d2_sft"
P25 = ROOT / "results/diagnostics/p25_c_dagger"
P31 = ROOT / "results/diagnostics/p31_skill_interference"
P35 = ROOT / "results/diagnostics/p35_precision_sft"
P36 = ROOT / "results/diagnostics/p36_rule_handoff_d_sft"

P31_CKPT = P31 / "checkpoints/M2_pcgrad/checkpoint_best.pt"
P36_CKPT = P36 / "checkpoints/D_motion_rule_handoff/checkpoint_epoch_0010.pt"
M111_META = ROOT / (
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/"
    "SELECTED_skill_aware_hybrid_meta.json"
)

P21_MAN = P21 / "SFT_FULL_HORIZON_MANIFEST.json"
P23_MAN = P23 / "COMBINED_DATASET_MANIFEST.json"
P35_MAN = P35 / "PACKAGED_MANIFEST.csv"

SEED = 38
