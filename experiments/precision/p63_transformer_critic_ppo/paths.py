"""P63 paths and pilot hyperparameters."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p63_transformer_critic_ppo"
CKPT_DIR = OUT / "checkpoints"

P62_EPOCH3 = (
    ROOT / "results/experiments/p62_grpo_optimization/checkpoints/P62_epoch3.pt"
)
P54_MANIFEST = ROOT / "results/experiments/p54_rebalanced_grpo64/P54_TRAIN_IC_MANIFEST.json"
P62_FINAL = ROOT / "results/experiments/p62_grpo_optimization/FINAL_TEST_RESULTS.json"
P61_JOINT = ROOT / "results/experiments/p61_reproducible_joint_eval/JOINT_EVAL.json"

SEED = 5478
WINDOW = 160
GROUP_PILOT = 8  # trajectories per IC per collection (pilot)
MAX_EP_STEPS = 80
T_MAX = 3100

# Pilot PPO hyperparams (initial, not claimed optimal)
CLIP = 0.2
GAE_LAMBDA = 0.95
GAMMA = 1.0
PPO_EPOCHS = 4
ACTOR_LR = 1e-5
CRITIC_LR = 1e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
TARGET_KL = 0.02
MINIBATCH = 32
CRITIC_WARM_TRAJ = 128
CRITIC_WARM_EPOCHS = 8
PILOT_UPDATES = 4  # short pilot collections
EXPAND_C_N = 24
