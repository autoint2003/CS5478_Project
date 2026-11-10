"""Reproducible randomized stratified IC sampling from original 321 pool."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p62_grpo_optimization.eval_dev import _flat_starts, _fresh_final_starts
from experiments.precision.p64_ppo_multiround.paths import ICS_PER_UPDATE, OUT, SEED


def full_train_pool(seed: int = SEED) -> list:
    return list(build_p54_dataset(seed=seed)["schedule"])


def blocked_eval_ids() -> set[str]:
    flat, _ = _flat_starts(4)
    blocked = {s.traj_id for s in flat} | {root_id(s.traj_id) for s in flat}
    final, _ = _fresh_final_starts({s.traj_id for s in flat}, 4)
    blocked |= {s.traj_id for s in final} | {root_id(s.traj_id) for s in final}
    return blocked


def is_blocked_eval(tid: str, blocked: set[str] | None = None) -> bool:
    """True if traj_id or its physical twin root is in the eval block set."""
    blocked = blocked if blocked is not None else blocked_eval_ids()
    if tid in blocked or root_id(tid) in blocked:
        return True
    return False


# Back-compat alias used by sample_stratified
_is_blocked = is_blocked_eval


def sample_stratified(
    pool: list,
    *,
    n_ics: int = ICS_PER_UPDATE,
    seed: int,
    blocked: set[str] | None = None,
) -> list:
    """Sample n_ics with equal per-class quota from eligible train pool."""
    blocked = blocked or set()
    by = defaultdict(list)
    for s in pool:
        if is_blocked_eval(s.traj_id, blocked):
            continue
        by[s.skill].append(s)
    rng = np.random.default_rng(seed)
    per = max(1, n_ics // 4)
    out = []
    for sk in ("A1", "B1", "C", "D"):
        cand = list(by[sk])
        rng.shuffle(cand)
        out.extend(cand[:per])
    # top-up if short
    if len(out) < n_ics:
        rest = [s for s in pool if s not in out and not is_blocked_eval(s.traj_id, blocked)]
        rng.shuffle(rest)
        out.extend(rest[: n_ics - len(out)])
    return out[:n_ics]


def exposure_row(starts: list, *, step: int, update: int, n_traj: int) -> dict:
    counts = Counter(s.skill for s in starts)
    return {
        "step": step,
        "update": update,
        "n_ics": len(starts),
        "n_traj_per_ic": n_traj,
        "by_skill": dict(counts),
        "traj_ids": [s.traj_id for s in starts],
    }


def append_exposure(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "IC_EXPOSURE.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
