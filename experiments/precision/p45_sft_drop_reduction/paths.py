from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p45_sft_drop_reduction"
P44 = ROOT / "results/experiments/p44_r3f_unified_sft"
P44_CKPT = P44 / "checkpoints/p44_best.pt"
P44_EVAL = P44 / "SFT_EVAL_ROLLOUTS.csv"
P44_DATA = P44 / "dataset"
SEED = 5478
T_MAX = 3100
