from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p48_b_failure_root_cause"
P44 = ROOT / "results/experiments/p44_r3f_unified_sft"
P44_CKPT = P44 / "checkpoints/p44_best.pt"
P44_EVAL = P44 / "SFT_EVAL_ROLLOUTS.csv"
P45 = ROOT / "results/experiments/p45_sft_drop_reduction"
P45B_CKPT = P45 / "checkpoints/P45B_best.pt"
P45_DIAG = P45 / "DIAGNOSIS_ROWS.csv"
P47 = ROOT / "results/experiments/p47_b_ordinary_expansion"
P47_CKPT = P47 / "checkpoints/P47_best.pt"
P47T_CKPT = P47 / "checkpoints/P47T_best.pt"
TABLE_FREE = ROOT / "results/diagnostics/plan_b_table_free_recovery/raw/accepted"
P47_RAW = P47 / "raw"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
SEED = 5478
T_MAX = 3100
