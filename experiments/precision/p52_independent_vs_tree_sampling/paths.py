from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p52_independent_vs_tree_sampling"
P47T_CKPT = ROOT / "results/experiments/p47_b_ordinary_expansion/checkpoints/P47T_best.pt"
P44_EVAL = ROOT / "results/experiments/p44_r3f_unified_sft/SFT_EVAL_ROLLOUTS.csv"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
SEED = 52
T_MAX = 3100
N_CANDIDATES = 64
N_TRUNKS = 16  # Tree80-style independent trunks
# 16 trunks + 48 branches = 64 terminal candidates
N_UNIFORM_PER_TRUNK = 2
N_ENTROPY_PER_TRUNK = 1
SKILLS = ("A1", "B1", "C", "D")
CLASS_LABEL = {"A1": "A", "B1": "B", "C": "C", "D": "D"}
BUDGET_CHECKPOINTS = (8, 16, 32, 64)
# Tree80 used ~200-step trunks; R3F T_max=3100 would OOM if every step stores MjData.
CAPTURE_STRIDE = 5
MAX_CAPTURES_PER_TRUNK = 80
