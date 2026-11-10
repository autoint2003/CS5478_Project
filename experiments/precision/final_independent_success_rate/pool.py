"""Build leakage-free heldout candidate pools (no train / diagnostic recycle)."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id
from experiments.precision.p53_independent_grpo64.probabilistic_eval import load_heldout_starts
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p62_grpo_optimization.eval_dev import _flat_starts, _fresh_final_starts
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids, is_blocked_eval
from experiments.precision.final_independent_success_rate.paths import (
    IC_EXPOSURE,
    P82_CACHE,
    P83_PANEL,
    SKILLS,
    TARGET_PER_CLASS,
)


def _load_jsonl_traj_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for t in row.get("traj_ids") or []:
            out.add(str(t))
            out.add(root_id(str(t)))
    return out


def _panel_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.is_file():
        return out
    rows = pickle.loads(path.read_bytes())
    for r in rows:
        tid = str(r.get("traj_id") or "")
        if tid:
            out.add(tid)
            out.add(root_id(tid))
    return out


def exclusion_sets() -> dict[str, Any]:
    train = build_p54_dataset(seed=5478)["schedule"]
    train_ids = {s.traj_id for s in train} | {root_id(s.traj_id) for s in train}
    blocked = blocked_eval_ids()
    flat, _ = _flat_starts(4)
    final, _ = _fresh_final_starts({s.traj_id for s in flat}, 4)
    used_dev = {s.traj_id for s in flat} | {root_id(s.traj_id) for s in flat}
    used_final = {s.traj_id for s in final} | {root_id(s.traj_id) for s in final}
    exposure = _load_jsonl_traj_ids(IC_EXPOSURE)
    panel = _panel_ids(P83_PANEL) | _panel_ids(P82_CACHE)
    return {
        "train": train_ids,
        "blocked_eval": blocked,
        "p61_p62_dev": used_dev,
        "p61_p62_final_slice": used_final,
        "p64_ic_exposure": exposure,
        "p82_p83_diagnostic_panel": panel,
    }


def classify_exclusion(tid: str, sets: dict[str, set[str]]) -> list[str]:
    rid = root_id(tid)
    tags = []
    for name, s in sets.items():
        if name == "blocked_eval":
            if is_blocked_eval(tid, s):
                tags.append(name)
            continue
        if tid in s or rid in s:
            tags.append(name)
    return tags


def eligible_heldout_pools(*, per_class_cap: int = 200) -> dict[str, Any]:
    """Heldout MixedStarts with full exclusion; de-duped by root_id."""
    held = load_heldout_starts(per_class=per_class_cap)
    sets = exclusion_sets()
    by_ok: dict[str, list] = {sk: [] for sk in SKILLS}
    by_excluded: dict[str, list] = {sk: [] for sk in SKILLS}
    seen_root: dict[str, set[str]] = {sk: set() for sk in SKILLS}

    for sk in SKILLS:
        for s in held[sk]:
            rid = root_id(s.traj_id)
            if rid in seen_root[sk]:
                continue
            seen_root[sk].add(rid)
            tags = classify_exclusion(s.traj_id, sets)
            if tags:
                by_excluded[sk].append({"traj_id": s.traj_id, "root_id": rid, "reasons": tags})
            else:
                by_ok[sk].append(s)

    summary = {
        sk: {
            "heldout_loaded": len(held[sk]),
            "unique_roots_considered": len(seen_root[sk]),
            "eligible": len(by_ok[sk]),
            "excluded": len(by_excluded[sk]),
            "target": TARGET_PER_CLASS,
            "shortfall": max(0, TARGET_PER_CLASS - len(by_ok[sk])),
        }
        for sk in SKILLS
    }
    return {
        "eligible": by_ok,
        "excluded_rows": by_excluded,
        "summary": summary,
        "exclusion_set_sizes": {k: len(v) for k, v in sets.items()},
    }


def sample_manifest_starts(
    eligible: dict[str, list],
    *,
    n_per_class: int = TARGET_PER_CLASS,
    seed: int,
) -> tuple[dict[str, list], dict[str, Any]]:
    """Sample up to n_per_class per skill; never pad with ineligible ICs."""
    rng = np.random.default_rng(seed)
    selected: dict[str, list] = {}
    gaps: dict[str, Any] = {}
    for sk in SKILLS:
        pool = list(eligible.get(sk) or [])
        rng.shuffle(pool)
        take = pool[:n_per_class]
        selected[sk] = take
        gaps[sk] = {
            "requested": n_per_class,
            "available": len(pool),
            "selected": len(take),
            "shortfall": max(0, n_per_class - len(take)),
            "padded_with_train": False,
        }
    return selected, gaps
