"""P63 Transformer PPO + C grasp — paths and pilot budgets."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p63_transformer_ppo_c_grasp"
CKPT_DIR = OUT / "checkpoints"

# Reuse prior P63 Critic PPO implementation
P63_CORE = ROOT / "experiments/precision/p63_transformer_critic_ppo"
P63_CORE_OUT = ROOT / "results/experiments/p63_transformer_critic_ppo"
P62_EPOCH3 = ROOT / "results/experiments/p62_grpo_optimization/checkpoints/P62_epoch3.pt"
CRITIC_WARM = P63_CORE_OUT / "checkpoints/P63_critic_warm.pt"
C_EPISODES = P63_CORE_OUT / "c_experts/episodes"
C_MANIFEST = P63_CORE_OUT / "C_DATA_MANIFEST.jsonl"

SEED = 5478
WINDOW = 160
GROUP = 8
MAX_EP_STEPS = 80
T_MAX = 3100
DT = 0.02

# Pilot budget: matched traj across branches
PILOT_ICS = 64
PILOT_UPDATES = 4  # 64×8×4 = 2048 traj/branch
PILOT_TRAJ = GROUP

# C grasp event rewards (C episodes only)
R_CORRECT_RECATCH = 2.0
R_SUSTAINED = 2.0
R_R3F_READY = 1.0
R_BAD_CONTACT = -1.0
R_LOST_AFTER = -1.0
SUSTAIN_S = 0.5
MAX_POS_EVENT = 5.0
