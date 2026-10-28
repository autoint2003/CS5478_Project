from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p46_b_tree80_expansion"
P45 = ROOT / "results/experiments/p45_sft_drop_reduction"
P45B_CKPT = P45 / "checkpoints/P45B_best.pt"
P44 = ROOT / "results/experiments/p44_r3f_unified_sft"
P44_CKPT = P44 / "checkpoints/p44_best.pt"
P44_EVAL = P44 / "SFT_EVAL_ROLLOUTS.csv"
TREE80_DS = ROOT / "results/data_generation/stage2_b_grpo_multistage_v1"
M111_REP = ROOT / "results/diagnostics/b_grpo_multistage_validation/M111_REPRESENTABILITY.csv"
SEED = 5478
T_MAX = 3100
