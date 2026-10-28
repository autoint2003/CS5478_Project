from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p51_four_class_mixed_ppo"
P50 = ROOT / "results/experiments/p50_ppo_reward_redesign"
P50_CKPT = P50 / "checkpoints/P50B_best.pt"
P47 = ROOT / "results/experiments/p47_b_ordinary_expansion"
P47T_CKPT = P47 / "checkpoints/P47T_best.pt"
P44 = ROOT / "results/experiments/p44_r3f_unified_sft"
P44_EVAL = P44 / "SFT_EVAL_ROLLOUTS.csv"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
BALL_V2 = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
DATA_SNAP = ROOT / "data/snapshots/train"
SEED = 5478
T_MAX = 3100
SKILLS = ("A1", "B1", "C", "D")
CLASS_LABEL = {"A1": "A", "B1": "B", "C": "C", "D": "D"}
