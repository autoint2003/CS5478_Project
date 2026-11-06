"""Enumerate all eligible A/B/C/D training ICs (no arbitrary train cap)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import build_pools
from experiments.precision.p53_independent_grpo64.paths import OUT, SEED, SKILLS


def build_uncapped_pools(*, per_class_val: int = 4, seed: int = SEED) -> dict:
    """All non-holdout ICs; carve a small val set; remainder = full train (uncapped)."""
    return build_pools(
        per_class_train=10_000_000,
        per_class_val=per_class_val,
        b_per_family=10_000_000,
    )


def start_key(s: MixedStart) -> tuple[str, str]:
    return (s.skill, s.traj_id)


def dedupe_starts(starts: list[MixedStart]) -> list[MixedStart]:
    seen: set[tuple[str, str]] = set()
    out: list[MixedStart] = []
    for s in starts:
        k = start_key(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def build_epoch_schedule(
    *,
    per_class_val: int = 4,
    seed: int = SEED,
) -> dict[str, Any]:
    pools = build_uncapped_pools(per_class_val=per_class_val, seed=seed)
    train_by = {sk: dedupe_starts(list(pools["train"].get(sk) or [])) for sk in SKILLS}
    val_by = {sk: dedupe_starts(list(pools["val"].get(sk) or [])) for sk in SKILLS}

    flat: list[MixedStart] = []
    for sk in SKILLS:
        flat.extend(train_by[sk])
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(flat))
    rng.shuffle(order)
    schedule = [flat[int(i)] for i in order]

    # Index for resume lookup
    lookup: dict[tuple[str, str], MixedStart] = {}
    for s in schedule:
        lookup[start_key(s)] = s
    for sk in SKILLS:
        for s in val_by[sk]:
            lookup[start_key(s)] = s

    per_class = {sk: len(train_by[sk]) for sk in SKILLS}
    n_train = sum(per_class.values())
    group = 64
    manifest = {
        "seed": int(seed),
        "per_class_val": int(per_class_val),
        "group_size": group,
        "n_train_ics": n_train,
        "n_train_per_class": per_class,
        "n_val_per_class": {sk: len(val_by[sk]) for sk in SKILLS},
        "n_expected_groups": n_train,
        "n_expected_rollouts": n_train * group,
        "expected_sim_budget": {
            "policy_rollouts": n_train * group,
            "note": "Each rollout may add 12s nominal if HANDOFF fires; DROP/TIMEOUT shorter.",
        },
        "train_ics": [
            {
                "index": i,
                "skill": s.skill,
                "class": {"A1": "A", "B1": "B", "C": "C", "D": "D"}[s.skill],
                "traj_id": s.traj_id,
                "family": s.family,
                "kind": s.kind,
            }
            for i, s in enumerate(schedule)
        ],
        "val_ics": [
            {
                "skill": s.skill,
                "traj_id": s.traj_id,
                "family": s.family,
                "kind": s.kind,
            }
            for sk in SKILLS
            for s in val_by[sk]
        ],
        "excluded": {
            "heldout": pools.get("heldout"),
            "validation_carved_from_eligible": True,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "FULL_TRAIN_IC_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "schedule": schedule,
        "val_by": val_by,
        "lookup": lookup,
        "path": path,
    }


def load_manifest_schedule(seed: int = SEED) -> dict[str, Any]:
    """Rebuild MixedStart objects matching FULL_TRAIN_IC_MANIFEST.json order."""
    built = build_epoch_schedule(seed=seed)
    man_path = OUT / "FULL_TRAIN_IC_MANIFEST.json"
    if man_path.is_file():
        saved = json.loads(man_path.read_text(encoding="utf-8"))
        # Prefer saved order if traj_ids match rebuilt set
        by_id = {(s.skill, s.traj_id): s for s in built["schedule"]}
        rebuilt: list[MixedStart] = []
        missing = []
        for row in saved.get("train_ics") or []:
            k = (row["skill"], row["traj_id"])
            if k in by_id:
                rebuilt.append(by_id[k])
            else:
                missing.append(k)
        if not missing and len(rebuilt) == len(saved["train_ics"]):
            built["schedule"] = rebuilt
            built["manifest"] = saved
        elif missing:
            # Keep freshly built; rewrite manifest
            built["path"].write_text(json.dumps(built["manifest"], indent=2), encoding="utf-8")
    return built
