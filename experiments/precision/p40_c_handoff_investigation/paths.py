from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p40_c_handoff_investigation"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P21_MAN = P21 / "SFT_FULL_HORIZON_MANIFEST.json"
P39 = ROOT / "results/diagnostics/p39_working_hold_handoff"
M111_META = ROOT / (
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/"
    "SELECTED_skill_aware_hybrid_meta.json"
)
