"""Deterministic development / final evaluation (P61 SHA-256 seeds)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p53_independent_grpo64.probabilistic_eval import load_heldout_starts
from experiments.precision.p61_reproducible_joint_eval.eval_core import eval_starts
from experiments.precision.p61_reproducible_joint_eval.paths import PROTOCOL_VERSION
from experiments.precision.p61_reproducible_joint_eval.seeding import GLOBAL_SEED
from experiments.precision.p62_grpo_optimization.paths import (
    DEV_ROLLOUTS,
    FINAL_ROLLOUTS,
    OUT,
    P57A_CKPT,
    PER_CLASS_DEV,
    PER_CLASS_FINAL,
)


def _flat_starts(per_class: int):
    held = load_heldout_starts(per_class=per_class)
    flat = []
    for sk in ("A1", "B1", "C", "D"):
        flat.extend(held[sk])
    return flat, held


def _fresh_final_starts(dev_ids: set[str], per_class: int):
    """Next holdout ICs not used in the P61/P62 development set.

    ``load_heldout_starts`` shuffles with a fixed seed then truncates; request a
    larger pool and take the slice after the development prefix.
    """
    pool = load_heldout_starts(per_class=per_class * 3)
    out = {sk: [] for sk in ("A1", "B1", "C", "D")}
    for sk in out:
        # Prefer IDs not in development set, preserving shuffle order
        cand = [st for st in pool[sk] if st.traj_id not in dev_ids]
        if len(cand) < per_class:
            cand = list(pool[sk][per_class : per_class * 2])
        out[sk] = cand[:per_class]
    flat = []
    for sk in ("A1", "B1", "C", "D"):
        flat.extend(out[sk])
    return flat, out


def _summarize(rep: dict) -> dict:
    per = rep["per_ic"]
    by_class = {}
    for sk, lab in (("A1", "A"), ("B1", "B"), ("C", "C"), ("D", "D")):
        rows = [r for r in per if r["skill"] == sk]
        if not rows:
            continue
        by_class[lab] = {
            "MSP_RELAXED": float(np.mean([r["p_relaxed"] for r in rows])),
            "MSP_12S": float(np.mean([r["p_12s"] for r in rows])),
            "MSP_NEW": float(np.mean([r["p_new"] for r in rows])),
            "zero_success_ics": int(sum(r["zero_success"] for r in rows)),
            "n_ic": len(rows),
        }
    return {
        "tag": rep["tag"],
        "ckpt": rep["ckpt"],
        "MSP_RELAXED": rep["MSP_RELAXED"],
        "MSP_12S": rep["MSP_12S"],
        "MSP_NEW": rep["MSP_NEW"],
        "zero_success_ics": rep["zero_success_ics"],
        "by_class": by_class,
        "per_ic": per,
    }


def eval_checkpoint_dev(tag: str, ckpt: Path, *, device: str = "cuda") -> dict:
    flat, _ = _flat_starts(PER_CLASS_DEV)
    # Use split name matching P61 heldout seeds so comparisons align with P61 protocol
    rep = eval_starts(
        tag, ckpt, flat, split="heldout", n_rollouts=DEV_ROLLOUTS, device=device
    )
    return _summarize(rep)


def eval_checkpoint_final(tag: str, ckpt: Path, *, device: str = "cuda") -> dict:
    dev_flat, _ = _flat_starts(PER_CLASS_DEV)
    dev_ids = {s.traj_id for s in dev_flat}
    flat, _ = _fresh_final_starts(dev_ids, PER_CLASS_FINAL)
    rep = eval_starts(
        tag, ckpt, flat, split="final_test", n_rollouts=FINAL_ROLLOUTS, device=device
    )
    out = _summarize(rep)
    out["split"] = "final_test"
    out["protocol_note"] = (
        f"Fresh ICs excluding P61/P62 development set; seeds via {PROTOCOL_VERSION} "
        f"split=final_test, global_seed={GLOBAL_SEED}"
    )
    out["ic_ids"] = [{"skill": s.skill, "traj_id": s.traj_id} for s in flat]
    return out


def paired_delta(base: dict, cur: dict) -> dict:
    bmap = {(r["skill"], r["traj_id"]): r for r in base["per_ic"]}
    deltas = []
    for r in cur["per_ic"]:
        b = bmap.get((r["skill"], r["traj_id"]))
        if b:
            deltas.append(r["p_relaxed"] - b["p_relaxed"])
    arr = np.asarray(deltas, float)
    n = len(arr)
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean_delta": float(arr.mean()) if n else None,
        "se": se,
        "ci95": [float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)] if n else None,
        "n_improved": int(sum(d > 1e-12 for d in deltas)),
        "n_regressed": int(sum(d < -1e-12 for d in deltas)),
        "n": n,
    }
