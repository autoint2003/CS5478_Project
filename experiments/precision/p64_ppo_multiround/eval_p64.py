"""P61 SHA-256 development eval with extra process rates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.precision.p62_grpo_optimization.eval_dev import (
    eval_checkpoint_dev,
    paired_delta,
)
from experiments.precision.p64_ppo_multiround.paths import OUT, P62_EPOCH3, P63_A_ACTOR


def _enrich(rep: dict) -> dict:
    per = rep.get("per_ic") or []
    if not per:
        return rep
    out = dict(rep)
    out["p_recatch"] = float(np.mean([r.get("p_recatch") or 0.0 for r in per]))
    out["p_stable_recatch"] = float(np.mean([r.get("p_stable") or 0.0 for r in per]))
    out["p_handoff"] = float(np.mean([r.get("p_handoff") or 0.0 for r in per]))
    # R3F-ready ≈ handoff rate under R3FOnline (includes early + timeout fallback)
    out["p_r3f_ready"] = out["p_handoff"]
    by = {}
    for lab, sk in (("A", "A1"), ("B", "B1"), ("C", "C"), ("D", "D")):
        rows = [r for r in per if r.get("skill") == sk]
        if not rows:
            continue
        by[lab] = {
            "MSP_RELAXED": float(np.mean([r["p_relaxed"] for r in rows])),
            "p_recatch": float(np.mean([r.get("p_recatch") or 0.0 for r in rows])),
            "p_stable_recatch": float(np.mean([r.get("p_stable") or 0.0 for r in rows])),
            "p_handoff": float(np.mean([r.get("p_handoff") or 0.0 for r in rows])),
            "zero_success_ics": int(sum(int(r.get("zero_success") or 0) for r in rows)),
            "n_ic": len(rows),
        }
    out["process_by_class"] = by
    return out


def eval_tag(tag: str, ckpt: Path, *, device: str = "cuda") -> dict:
    return _enrich(eval_checkpoint_dev(tag, ckpt, device=device))


def eval_baselines(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    p62 = eval_tag("P62_epoch3", P62_EPOCH3, device=device)
    p63a = eval_tag("P63_PPO_A", P63_A_ACTOR, device=device)
    return {"P62_epoch3": p62, "P63_PPO_A": p63a, "paired_p63a_vs_p62": paired_delta(p62, p63a)}
