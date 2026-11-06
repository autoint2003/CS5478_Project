"""P67 paths and diagnosis budgets."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p67_r3f_gate_diagnosis"
STATE_DIR = OUT / "states"
TRACE_DIR = OUT / "traces"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"

P66_OUT = ROOT / "results/experiments/p66_c_curriculum_ppo"
P66_A_ACTOR = P66_OUT / "checkpoints/P66_A_control_actor.pt"

SEED = 5478
T_MAX = 3100
MAX_EP_STEPS = 80

# Phase 1 budgets
N_ICS = 16
N_PER_IC = 16

# Phase 3
N_SNAPSHOTS_TARGET = 32
N_ACTOR_CF = 4
N_EXPERT_CF = 1
