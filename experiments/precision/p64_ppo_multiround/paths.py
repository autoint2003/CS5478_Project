"""P64 multi-round PPO continuation paths and budgets."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p64_ppo_multiround"
CKPT_DIR = OUT / "checkpoints"

P62_EPOCH3 = ROOT / "results/experiments/p62_grpo_optimization/checkpoints/P62_epoch3.pt"
P63_A_ACTOR = (
    ROOT
    / "results/experiments/p63_transformer_ppo_c_grasp/checkpoints/P63b_pilot_A_actor.pt"
)
P63_A_CRITIC = (
    ROOT
    / "results/experiments/p63_transformer_ppo_c_grasp/checkpoints/P63b_pilot_A_critic.pt"
)
P63_OUT = ROOT / "results/experiments/p63_transformer_ppo_c_grasp"
P63_CORE_OUT = ROOT / "results/experiments/p63_transformer_critic_ppo"
CRITIC_WARM = P63_CORE_OUT / "checkpoints/P63_critic_warm.pt"

SEED = 5478
GROUP = 8
ICS_PER_UPDATE = 64  # stratified 16/class
UPDATES_PER_STEP = 4  # 64×8×4 = 2048 traj/step
TRAJ_PER_STEP = ICS_PER_UPDATE * GROUP * UPDATES_PER_STEP  # 2048
MAX_STEPS = 4  # +8192 traj after P63-A
P63_A_CUMULATIVE = 2048

# Early-stop soft thresholds (do not stop on 0.002–0.005 MSP noise)
KL_ABORT = 0.5
KL_WARN = 0.05
EV_WARN = 0.05
ENTROPY_COLLAPSE = 0.05
CLASS_REGRESS_ABS = 0.08  # vs previous best on a class
MSP_DECLINE_STREAK = 2  # consecutive credible declines vs best
