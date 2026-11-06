"""P53 paths and immutable baseline references."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p53_independent_grpo64"
CKPT_DIR = OUT / "checkpoints"

P47 = ROOT / "results/experiments/p47_b_ordinary_expansion"
P47T_CKPT = P47 / "checkpoints/P47T_best.pt"
P50 = ROOT / "results/experiments/p50_ppo_reward_redesign"
P50_CKPT = P50 / "checkpoints/P50B_best.pt"
P51 = ROOT / "results/experiments/p51_four_class_mixed_ppo"
P51_CKPT = P51 / "checkpoints/P51_best.pt"
P44 = ROOT / "results/experiments/p44_r3f_unified_sft"
P44_EVAL = P44 / "SFT_EVAL_ROLLOUTS.csv"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"

SEED = 5478
T_MAX = 3100
SKILLS = ("A1", "B1", "C", "D")
CLASS_LABEL = {"A1": "A", "B1": "B", "C": "C", "D": "D"}

# Independent GRPO-64 (no Tree Branching)
GROUP_SIZE = 64
MAX_EP_STEPS = 80
TEMPERATURE = 1.0  # categorical T; must match all checkpoint evals
KL_COEF = 0.05
CLIP = 0.2
LR = 1e-5
GAMMA_RETURN = 1.0  # undiscounted trajectory return of dense+terminal rewards
