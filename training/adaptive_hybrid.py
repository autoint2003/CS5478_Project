"""Adaptive hybrid exploration for K48 GRPO.

Independent rollouts are the only samples that enter the GRPO loss. Tree
branches reuse the verified prefix-branching rules and are logged as
exploration. A tree success never becomes an independent success.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

STAGE_A = 8
STAGE_B = 8
TREE_EACH = 8
SLIDING = "gravity_assisted_in_hand_sliding"
HORIZONTAL = "horizontal_recatch"
VERTICAL = "vertical_recatch"
CATEGORIES = (HORIZONTAL, VERTICAL, SLIDING)


def primary_category(categories: set[str] | list[str]) -> str:
    """One snapshot occupies one sampling slot. Vertical wins over sliding."""
    present = set(categories)
    if HORIZONTAL in present:
        return HORIZONTAL
    if VERTICAL in present:
        return VERTICAL
    if SLIDING in present:
        return SLIDING
    raise RuntimeError(f"snapshot has no recovery category: {sorted(present)}")


def balanced_schedule(rows: list[dict]) -> list[dict]:
    """Round-robin primary categories. Each snapshot appears once."""
    buckets = {name: [] for name in CATEGORIES}
    seen = set()
    for row in rows:
        if row["snapshot"] in seen:
            raise RuntimeError(f"duplicate snapshot {row['snapshot']}")
        seen.add(row["snapshot"])
        buckets[primary_category(row["categories"])].append(row)
    order = []
    while any(buckets.values()):
        for name in CATEGORIES:
            if buckets[name]:
                order.append(buckets[name].pop(0))
    return order


def sampling_frequency(rows: list[dict]) -> dict:
    """Probability a uniform draw from the schedule hits each category label.

    A multi-category snapshot is one schedule entry. Its labels are all counted
    in the coverage frequencies below, and the primary-only frequencies show
    the draw probability.
    """
    schedule = balanced_schedule(rows)
    total = max(len(schedule), 1)
    primary = {name: 0 for name in CATEGORIES}
    coverage = {name: 0 for name in CATEGORIES}
    for row in schedule:
        primary[primary_category(row["categories"])] += 1
        for name in row["categories"]:
            coverage[name] += 1
    return {
        "snapshots": len(schedule),
        "primary_probability": {name: primary[name] / total for name in CATEGORIES},
        "coverage_count": coverage,
    }


def select_branch_points(candidates: list[dict], rng, count: int, strategy: str) -> list[dict]:
    """Pick branch decisions from realized independent steps.

    Uniform draws distinct candidates. Entropy ranks by the 48-way token
    entropy in nats, then candidate index. Neither rule reads the outcome.
    """
    if count <= 0 or not candidates:
        return []
    take = min(int(count), len(candidates))
    if strategy == "uniform":
        picked = rng.choice(np.arange(len(candidates)), size=take, replace=False)
        return [candidates[int(index)] for index in sorted(int(value) for value in picked)]
    if strategy == "entropy":
        order = sorted(range(len(candidates)), key=lambda index: (-float(candidates[index]["entropy"]), index))
        return [candidates[index] for index in order[:take]]
    raise RuntimeError(f"unknown branch strategy {strategy}")


def exploration_decision(state: dict) -> dict:
    """Return the next action for one IC. Tree successes do not authorize an update."""
    if int(state.get("stage_a_success") or 0) > 0 and not state.get("stage_b_done"):
        return {"action": "update", "source": "independent_stage_a", "group": "stage_a"}
    if not state.get("stage_b_done"):
        return {"action": "sample", "stage": "B"}
    independent = int(state.get("stage_a_success") or 0) + int(state.get("stage_b_success") or 0)
    if independent > 0 and not state.get("tree_done"):
        return {"action": "update", "source": "independent_stage_ab", "group": "stage_ab"}
    if not state.get("tree_done"):
        return {"action": "sample", "stage": "C"}
    tree = int(state.get("tree_uniform_success") or 0) + int(state.get("tree_entropy_success") or 0)
    if tree > 0 and not state.get("resample_done"):
        return {"action": "sample", "stage": "resample"}
    if state.get("resample_done") and int(state.get("resample_success") or 0) > 0:
        return {"action": "update", "source": "independent_resample", "group": "resample"}
    return {"action": "skip", "source": None, "tree_success_excluded": tree > 0}


def loss_members(members: list[dict]) -> list[dict]:
    """Keep every independent rollout in the group, including failures."""
    if not members:
        raise RuntimeError("GRPO group is empty")
    if any(member.get("origin") != "independent" for member in members):
        raise RuntimeError("tree or off-policy rollout entered the GRPO group")
    if not any(member.get("label") == "SUCCESS_12S" for member in members):
        raise RuntimeError("GRPO group has no SUCCESS_12S rollout")
    return members


def load_train_snapshots() -> list[dict]:
    """Unique train snapshots counted by the quota rules. Valid rows are excluded."""

    def read(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    grouped: dict[str, dict] = {}

    def add(snapshot: str, episode_id: str, split: str, category: str) -> None:
        if split != "train":
            return
        row = grouped.setdefault(snapshot, {"id": episode_id, "snapshot": snapshot, "split": "train", "categories": []})
        if category not in row["categories"]:
            row["categories"].append(category)

    for row in read(ROOT / "data/gravity_assisted_in_hand_sliding/manifest.jsonl"):
        if row.get("verified_success") is False or row.get("formal") not in (None, "SUCCESS_12S"):
            continue
        add(row["snapshot"], row["id"], row.get("split") or "train", SLIDING)
    for row in read(ROOT / "data/recatch/manifest.jsonl"):
        if row.get("task_success") is False or row.get("counts_as_recatch") is False:
            continue
        add(row["snapshot"], row["id"], row.get("split") or "train", HORIZONTAL)
    for row in read(ROOT / "data/vertical_recatch/manifest.jsonl"):
        if row.get("task_success") is False or row.get("counts_as_recatch") is False:
            continue
        add(row["snapshot"], row["id"], row.get("split") or "train", VERTICAL)
    rows = []
    for snapshot in sorted(grouped):
        row = grouped[snapshot]
        row["categories"] = sorted(row["categories"])
        row["nominal_class"] = "RECOVERY_CANDIDATE"
        row["pool"] = "recovery"
        row["primary_category"] = primary_category(row["categories"])
        rows.append(row)
    return rows


def assert_train_valid_disjoint(train_rows: list[dict], valid_path: Path) -> None:
    valid_ids = set()
    valid_snaps = set()
    for line in valid_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        valid_ids.add(row["id"])
        valid_snaps.add(row["snapshot"])
    leaked = [row["id"] for row in train_rows if row["id"] in valid_ids or row["snapshot"] in valid_snaps]
    if leaked:
        raise RuntimeError(f"train/valid leak: {leaked[:5]}")
