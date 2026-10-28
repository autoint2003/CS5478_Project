from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p47_b_ordinary_expansion"
P45 = ROOT / "results/experiments/p45_sft_drop_reduction"
P45B_CKPT = P45 / "checkpoints/P45B_best.pt"
P44 = ROOT / "results/experiments/p44_r3f_unified_sft"
P44_EVAL = P44 / "SFT_EVAL_ROLLOUTS.csv"
TREE80 = ROOT / "results/grpo/tree80/checkpoint_epoch_0003.pt"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
TABLE_FREE = ROOT / "results/diagnostics/plan_b_table_free_recovery/raw/accepted"
BALL_V2 = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
SEED = 5478
T_MAX = 3100
