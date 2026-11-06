"""P65 PPO continuation + A/C diagnosis paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p65_ppo_continuation"
CKPT_DIR = OUT / "checkpoints"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"
P64_STEP4_OPTIM = P64_OUT / "checkpoints/P64_step4_optim.pt"
P64_STEP1_ACTOR = P64_OUT / "checkpoints/P64_step1_actor.pt"
P63_A_ACTOR = (
    ROOT
    / "results/experiments/p63_transformer_ppo_c_grasp/checkpoints/P63b_pilot_A_actor.pt"
)
P63_A_CRITIC = (
    ROOT
    / "results/experiments/p63_transformer_ppo_c_grasp/checkpoints/P63b_pilot_A_critic.pt"
)
P62_EPOCH3 = ROOT / "results/experiments/p62_grpo_optimization/checkpoints/P62_epoch3.pt"

SEED = 5478
GROUP = 8
ICS_PER_UPDATE = 64
UPDATES_PER_STEP = 4
TRAJ_PER_STEP = ICS_PER_UPDATE * GROUP * UPDATES_PER_STEP  # 2048
P64_CUMULATIVE = 10240
P65_STEP5_CUM = P64_CUMULATIVE + TRAJ_PER_STEP  # 12288
P65_STEP6_CUM = P65_STEP5_CUM + TRAJ_PER_STEP  # 14336

# Additional development panel
PANEL_PER_CLASS = 8
PANEL_ROLLOUTS = 32
PANEL_SEED = 65065
