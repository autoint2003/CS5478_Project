"""Select fixed 48 C train ICs: P58's 16 + 32 diverse additions (exclude P58 val C)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p59_expanded_c_grpo.paths import (
    N_TRAIN_C,
    OUT,
    P57A_PROGRESS,
    P58_SELECTED,
    SEED,
)


def _bucket(r: dict) -> str:
    s = int(r["success_relaxed_count"])
    rc = int(r["recatch_count"])
    st = int(r["stable_hold_count"])
    hd = int(r["handoff_count"])
    if s >= 12:
        return "successful_recovery"
    if s == 0 and rc == 0:
        return "failed_recatch"
    if s == 0 and rc > 0 and st == 0:
        return "partial_bilateral"
    if s == 0 and st > 0:
        return "unstable_or_partial_stable"
    if s > 0 and hd == 0:
        return "occasional_success_no_hand_agg"  # rare aggregate
    if st > 0 and s < 8:
        return "unstable_recatch_or_weak_complete"
    if st > 0 and s >= 8 and s < 12:
        return "stable_incomplete"
    return "intermediate"


def build_manifest() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    p58 = json.loads(P58_SELECTED.read_text(encoding="utf-8"))
    core16 = list(p58["train_c"]["all"])
    val_c = set(p58["val_c"])
    val_abd = {tid for ids in p58["val_abd"].values() for tid in ids}

    rows = [r for r in csv.DictReader(P57A_PROGRESS.open(encoding="utf-8")) if r["class"] == "C"]
    by_id = {r["ic_id"]: r for r in rows}
    all_c = [r["ic_id"] for r in rows]
    assert len(all_c) == 68

    remaining = [tid for tid in all_c if tid not in core16 and tid not in val_c]
    # Stratify remaining into buckets
    buckets: dict[str, list[str]] = defaultdict(list)
    for tid in remaining:
        buckets[_bucket(by_id[tid])].append(tid)

    rng = np.random.default_rng(SEED + 59)
    for b in buckets:
        rng.shuffle(buckets[b])

    # Target mix among 32 adds
    targets = {
        "failed_recatch": 8,
        "partial_bilateral": 4,
        "unstable_or_partial_stable": 6,
        "unstable_recatch_or_weak_complete": 6,
        "stable_incomplete": 4,
        "successful_recovery": 4,
    }
    added = []
    for b, n in targets.items():
        take = buckets[b][:n]
        added.extend(take)
        buckets[b] = buckets[b][n:]
    # Fill to 32 from leftover remaining (prefer underrepresented)
    leftover = [tid for tid in remaining if tid not in added]
    rng.shuffle(leftover)
    while len(added) < 32 and leftover:
        added.append(leftover.pop())
    added = added[:32]

    train48 = core16 + added
    assert len(train48) == N_TRAIN_C
    assert len(set(train48)) == N_TRAIN_C
    assert not (set(train48) & val_c)

    # Tag each
    tags = {}
    for tid in core16:
        tags[tid] = "p58_core"
    for tid in added:
        tags[tid] = _bucket(by_id[tid])

    ds = build_p54_dataset(seed=SEED)
    lookup = {s.traj_id: s.skill for s in ds["schedule"]}
    for tid in train48:
        assert lookup.get(tid) == "C", tid

    man = {
        "seed": SEED,
        "n_train": len(train48),
        "p58_core_16": core16,
        "added_32": added,
        "all": train48,
        "tags": tags,
        "bucket_counts_added": {
            k: sum(1 for t in added if tags[t] == k) for k in sorted(set(tags[t] for t in added))
        },
        "excluded_val_c": sorted(val_c),
        "val_c": p58["val_c"],
        "val_abd": p58["val_abd"],
        "init_ckpt": "P58_pass2.pt",
        "notes": "Fixed manifest; no synthetic/duplicate ICs; P58 val C held out of train.",
    }
    (OUT / "TRAIN_IC_MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


if __name__ == "__main__":
    m = build_manifest()
    print(json.dumps({k: m[k] for k in ("n_train", "bucket_counts_added", "excluded_val_c")}, indent=2))
