"""Non-holdout A/B/C/D training + validation IC pools."""

from __future__ import annotations

import csv
import pickle
import re
from collections import defaultdict

import numpy as np

from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs
from experiments.precision.p49_b_targeted_dagger.collect_probe import load_family_ics
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p51_four_class_mixed_ppo.paths import P44_EVAL, SEED, SKILLS


def root_id(tid: str) -> str:
    t = re.sub(r"^(plan_b__)?", "", tid)
    t = re.sub(r"__([0-9a-f]{12,})$", "", t)
    return t


def holdout_ids() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {s: set() for s in SKILLS}
    for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")):
        if r.get("tag") != "holdout":
            continue
        sk = r["skill"]
        if sk in out:
            out[sk].add(r["traj_id"])
            out[sk].add(root_id(r["traj_id"]))
    return out


def _is_held(traj_id: str, held: set[str]) -> bool:
    if traj_id in held:
        return True
    rid = root_id(traj_id)
    if rid in held:
        return True
    for h in held:
        if traj_id in h or h in traj_id or rid in h or h in rid:
            return True
    return False


def build_pools(*, per_class_train: int = 24, per_class_val: int = 4, b_per_family: int = 8) -> dict:
    held = holdout_ids()
    jobs = collect_jobs()
    by_skill: dict[str, list[MixedStart]] = {s: [] for s in SKILLS}

    # A1 / C from P21 train episodes (non-holdout)
    for j in jobs:
        sk = j["skill"]
        if sk not in ("A1", "C"):
            continue
        if j.get("split") != "train":
            continue
        if _is_held(j["traj_id"], held[sk]):
            continue
        by_skill[sk].append(
            MixedStart(
                skill=sk,
                traj_id=j["traj_id"],
                family=j.get("family") or sk,
                kind="ep",
                ep=j["ep"],
            )
        )

    # D from P35 train captures
    for j in jobs:
        if j["skill"] != "D" or j.get("split") != "train":
            continue
        if _is_held(j["traj_id"], held["D"]):
            continue
        by_skill["D"].append(
            MixedStart(
                skill="D",
                traj_id=j["traj_id"],
                family="D",
                kind="capture",
                capture=j["capture"],
                ref=j["ref"],
            )
        )

    # B1: P21 train overlaps holdout entirely — use ordinary ball/slide/k48r/inward like P50
    held_b = held["B1"]
    ics = load_family_ics(held_b, per_family=b_per_family, seed=SEED)
    for ic in ics:
        by_skill["B1"].append(
            MixedStart(
                skill="B1",
                traj_id=ic["ic_id"],
                family=ic["family"],
                kind="capture",
                capture=ic["capture"],
                ref=ic["ref"],
            )
        )

    rng = np.random.default_rng(SEED)
    train: dict[str, list[MixedStart]] = {}
    val: dict[str, list[MixedStart]] = {}
    for sk in SKILLS:
        pool = list(by_skill[sk])
        rng.shuffle(pool)
        n_v = min(per_class_val, max(1, len(pool) // 5)) if pool else 0
        val[sk] = pool[:n_v]
        rest = pool[n_v:]
        train[sk] = rest[:per_class_train] if rest else []
        # if train empty but pool exists, borrow
        if not train[sk] and pool:
            train[sk] = pool[: min(per_class_train, len(pool))]

    return {
        "train": train,
        "val": val,
        "heldout": {k: sorted(v) for k, v in held.items()},
        "counts": {
            "raw": {sk: len(by_skill[sk]) for sk in SKILLS},
            "train": {sk: len(train[sk]) for sk in SKILLS},
            "val": {sk: len(val[sk]) for sk in SKILLS},
        },
    }


def sample_balanced(
    pools: dict[str, list[MixedStart]],
    rng: np.random.Generator,
) -> MixedStart:
    """Uniform over classes, then uniform within class."""
    nonempty = [sk for sk in SKILLS if pools.get(sk)]
    sk = nonempty[int(rng.integers(0, len(nonempty)))]
    cand = pools[sk]
    return cand[int(rng.integers(0, len(cand)))]
