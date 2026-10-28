from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p34_6_grip_aware_encoding"
P34 = ROOT / "results/diagnostics/p34_precision_data_expansion"
P34_5 = ROOT / "results/diagnostics/p34_5_m111_precision_repair"
P34_MANIFEST = P34 / "DATASET_MANIFEST.csv"
M111_META = ROOT / (
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/"
    "SELECTED_skill_aware_hybrid_meta.json"
)
ENCODER_VERSION = "m111_r1b_grip_w2.0_v1"
DATASET_DIR = OUT / "dataset_precision_r1b_v1"
HANDOFF = 111
MOTION_K = 111
# R1b grip weight (frozen; not retuned in P34.6)
R1B_GRIP_WEIGHT = 2.0
