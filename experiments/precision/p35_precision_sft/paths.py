from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p35_precision_sft"
R1B = ROOT / "results/diagnostics/p34_6_grip_aware_encoding"
R1B_MANIFEST = R1B / "DATASET_MANIFEST.csv"
R1B_PAIRED = R1B / "PAIRED_REPLAY.csv"
P0_MANIFEST = ROOT / "results/diagnostics/p34_precision_data_expansion/P0_MANIFEST.csv"
M111_META = ROOT / (
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/"
    "SELECTED_skill_aware_hybrid_meta.json"
)
ENCODER_VERSION = "m111_r1b_grip_w2.0_v1"
R1B_GRIP_WEIGHT = 2.0
SEED = 35
# Expert Precision tapes are often ~1.7–2.1k steps; 800 caused P33 timeouts.
MAX_ROLLOUT_STEPS = 2500
P22 = ROOT / "results/diagnostics/p22_full_horizon_sft/checkpoints/A_natural/checkpoint_best.pt"
P28 = ROOT / "results/diagnostics/p28_multiskill_distillation/checkpoints/C_distill_corr_K32/checkpoint_best.pt"
P31 = ROOT / "results/diagnostics/p31_skill_interference/checkpoints/M2_pcgrad/checkpoint_best.pt"
P33_CKPT = ROOT / "results/diagnostics/p33_precision_learnability/checkpoints"
