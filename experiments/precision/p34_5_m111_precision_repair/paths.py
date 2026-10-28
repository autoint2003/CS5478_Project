from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p34_5_m111_precision_repair"
P34 = ROOT / "results/diagnostics/p34_precision_data_expansion"
M111_META = ROOT / (
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/"
    "SELECTED_skill_aware_hybrid_meta.json"
)
M111_REPLAY = P34 / "M111_REPLAY_RESULTS.csv"
P34_MANIFEST = P34 / "DATASET_MANIFEST.csv"
HANDOFF_M111 = 111
MOTION_K = 111
