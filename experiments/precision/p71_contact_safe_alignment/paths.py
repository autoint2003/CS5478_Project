"""P71 paths and budgets — no training."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p71_contact_safe_alignment"
STATE_DIR = OUT / "states"

P64_OUT = ROOT / "results/experiments/p64_ppo_multiround"
P64_STEP4_ACTOR = P64_OUT / "checkpoints/P64_step4_actor.pt"
P64_STEP4_CRITIC = P64_OUT / "checkpoints/P64_step4_critic.pt"

P69_OUT = ROOT / "results/experiments/p69_skill_sequence"
P69_SPLIT = P69_OUT / "DISCOVERY_VALIDATION_SPLIT.json"
P70_OUT = ROOT / "results/experiments/p70_geometry_to_r3f"
P70_VARIANTS = P70_OUT / "CLOSED_LOOP_VARIANTS.json"

SEED = 5478
T_MAX = 3100
MAX_EP_STEPS = 80

P69_SEQ = [56, 82, 50, 71]
TOKEN_56 = 56
GRIP_TOK = 82
# Alignment-capable tokens: APPROACH/ORIENT with non-trivial linear/angular; exclude pure GRIP
# Populated at runtime from codebook; fallback list from P69 families
ALIGN_FALLBACK = [50, 71, 40, 37, 67, 39]  # non-56 APPROACH-ish from P69 search

# P70 V2 frozen hyperparams
V2_MAX_APPROACH = 6
V2_MIN_APPROACH = 2
V2_READY = -0.05
V2_GRIP_STEPS = 1
V2_POST = 2
V2_MAX_ACTIVATE = 7

# Phase 1
N_P1_ICS = 8
N_P1_ROLLOUTS = 12  # match P70 val density for uncertainty

# Phase 2
N_TOK56_STATES = 24

# Phase 3–4
N_DEV_ICS = 8
N_DEV_ROLLOUTS = 10
N_HELD_ICS = 8
N_HELD_ROLLOUTS = 12

# Phase 5 panel
PANEL_C_ROLLOUTS = 16
E_THRESH = (25.0, 12.0, 9.0)
