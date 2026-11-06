"""Resolve P54 train MixedStarts for 0/64 IC ids."""

from __future__ import annotations

import json
from typing import Any

from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p55_expert_takeover.paths import P54_MANIFEST, SEED


_CACHE: dict[str, Any] | None = None


def _dataset() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = build_p54_dataset(seed=SEED)
    return _CACHE


def resolve_starts(ic_ids: list[str] | None = None) -> dict[str, MixedStart]:
    ds = _dataset()
    lookup = ds["lookup"]  # (skill, traj_id) -> MixedStart
    by_tid = {s.traj_id: s for s in ds["schedule"]}
    if ic_ids is None:
        return by_tid
    out = {}
    for tid in ic_ids:
        if tid in by_tid:
            out[tid] = by_tid[tid]
        else:
            # try skill-prefixed lookup from manifest
            man = json.loads(P54_MANIFEST.read_text(encoding="utf-8"))
            for row in man["train_ics"]:
                if row["traj_id"] == tid:
                    key = (row["skill"], tid)
                    if key in lookup:
                        out[tid] = lookup[key]
                    break
    return out
