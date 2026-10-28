from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p44_r3f_unified_sft"
EXP_OUT = OUT  # deliverables live here

# Reuse existing expert corpora (compatibility checked via re-verify under current nominal)
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P21_MAN = P21 / "SFT_FULL_HORIZON_MANIFEST.json"
P35 = ROOT / "results/diagnostics/p35_precision_sft"
P35_MAN = P35 / "PACKAGED_MANIFEST.csv"
M111_META = (
    ROOT
    / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
)

# From P21/P35 length distribution (D max 3042); see PIPELINE_AUDIT.md
T_MAX_DEFAULT = 3100
SEED = 5478
