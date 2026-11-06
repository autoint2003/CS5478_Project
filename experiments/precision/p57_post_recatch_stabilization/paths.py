"""P57 paths and fixed physical criteria (frozen before training)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p57_post_recatch_stabilization"
CKPT_DIR = OUT / "checkpoints"

P56_OUT = ROOT / "results/experiments/p56_continued_grpo64"
P56_CKPT = P56_OUT / "checkpoints/P56_epoch1.pt"
P56_C_AUDIT = P56_OUT / "C_FAILURE_AUDIT.json"
P56_C_DETAIL = P56_OUT / "C_FAILURE_AUDIT_DETAIL.json"

SEED = 5478
GROUP = 64
T_MAX = 3100
MAX_EP_STEPS = 80
TEMPERATURE = 1.0
SKILLS = ("A1", "B1", "C", "D")

# --- Fixed recatch / stable-hold criteria (do not change mid-experiment) ---
# Valid recatch: first run of RECATCH_MIN_STEPS consecutive bilateral steps
# with min_pad >= RECATCH_MIN_PAD (after episode start or after a no-contact gap).
RECATCH_MIN_STEPS = 3
RECATCH_MIN_PAD = 1
# Stable hold after recatch: sustained bilateral with min_pad >= STABLE_MIN_PAD
# for STABLE_HOLD_STEPS consecutive steps AND mean vrel over that window <= STABLE_VREL_MAX.
STABLE_HOLD_STEPS = 12
STABLE_MIN_PAD = 2
STABLE_VREL_MAX = 0.35
