from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p49_b_targeted_dagger"
P47 = ROOT / "results/experiments/p47_b_ordinary_expansion"
P47T_CKPT = P47 / "checkpoints/P47T_best.pt"
P45 = ROOT / "results/experiments/p45_sft_drop_reduction"
P44 = ROOT / "results/experiments/p44_r3f_unified_sft"
P44_EVAL = P44 / "SFT_EVAL_ROLLOUTS.csv"
P45B_DATA = P45 / "dataset"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
BALL_V2 = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
DATA_SNAP = ROOT / "data/snapshots/train"
SEED = 5478
T_MAX = 3100
