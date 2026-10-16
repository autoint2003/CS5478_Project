"""Adaptive hybrid exploration for K48 GRPO.

Stage A and Stage B update from independent rollouts only. When all 16
independent rollouts fail, each trunk contributes two uniform and two entropy
branches. Those 64 tree trajectories join the 16 trunks in one G=80 update.
Tree success does not trigger another independent resample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

STAGE_A = 8
STAGE_B = 8
BRANCHES_PER_METHOD = 2
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


def select_trunk_branches(candidates: list[dict], rng, count: int, strategy: str) -> list[dict]:
    """Pick ``count`` positions on one trunk.

    Distinct positions are preferred. A short trunk repeats positions instead
    of dropping a required branch. Uniform and entropy are separate calls.
    """
    needed = int(count)
    if needed <= 0:
        return []
    if not candidates:
        raise RuntimeError("trunk has no branchable position")
    if strategy == "uniform":
        if len(candidates) >= needed:
            picked = rng.choice(np.arange(len(candidates)), size=needed, replace=False)
            return [candidates[int(index)] for index in picked]
        extra = rng.choice(np.arange(len(candidates)), size=needed - len(candidates), replace=True)
        return list(candidates) + [candidates[int(index)] for index in extra]
    if strategy == "entropy":
        order = sorted(range(len(candidates)), key=lambda index: (-float(candidates[index]["entropy"]), index))
        ranked = [candidates[index] for index in order]
        if len(ranked) >= needed:
            return ranked[:needed]
        return ranked + [ranked[0]] * (needed - len(ranked))
    raise RuntimeError(f"unknown branch strategy {strategy}")


def trunk_candidates(trunk: dict, trunk_index: int) -> list[dict]:
    """Pre-action states that were captured on this independent rollout."""
    captured = trunk.get("captures") or {}
    candidates = []
    for step in trunk.get("steps") or []:
        index = int(step["step"])
        if index not in captured:
            continue
        candidates.append({
            "traj": int(trunk_index),
            "step": index,
            "entropy": float(step["entropy"]),
            "lost_low": int(step["lost_low"]),
            "t": float(step["t"]),
            "token": int(step["token"]),
        })
    return candidates


def allocate_tree_branches(trunks: list[dict], rng_uniform, rng_entropy, per_method: int = BRANCHES_PER_METHOD) -> tuple[list[dict], list[dict]]:
    """Two uniform and two entropy branches on every trunk. No global pool."""
    uniform = []
    entropy = []
    for index, trunk in enumerate(trunks):
        candidates = trunk_candidates(trunk, index)
        if not candidates:
            raise RuntimeError(f"trunk {index} has no branchable pre-action state")
        uniform.extend(select_trunk_branches(candidates, rng_uniform, per_method, "uniform"))
        entropy.extend(select_trunk_branches(candidates, rng_entropy, per_method, "entropy"))
    expected = len(trunks) * int(per_method)
    if len(uniform) != expected or len(entropy) != expected:
        raise RuntimeError(f"tree allocation produced {len(uniform)} uniform and {len(entropy)} entropy branches, expected {expected}")
    return uniform, entropy


def execute_allocated_branches(points: list[dict], rollout_one) -> list[dict]:
    """Run every allocated branch. A success does not cancel the rest."""
    leaves = []
    for point in points:
        leaves.append(rollout_one(point))
    if len(leaves) != len(points):
        raise RuntimeError(f"executed {len(leaves)} of {len(points)} allocated branches")
    return leaves


def exploration_decision(state: dict) -> dict:
    """Stage A or B updates only after an independent SUCCESS_12S.

    After the fixed tree pass, the G=80 group is updated directly. There is
    no independent resample.
    """
    if int(state.get("stage_a_success") or 0) > 0 and not state.get("stage_b_done"):
        return {"action": "update", "source": "independent_stage_a", "group": "stage_a", "stage": "A"}
    if not state.get("stage_b_done"):
        return {"action": "sample", "stage": "B"}
    independent = int(state.get("stage_a_success") or 0) + int(state.get("stage_b_success") or 0)
    if independent > 0 and not state.get("tree_done"):
        return {"action": "update", "source": "independent_stage_ab", "group": "stage_ab", "stage": "B"}
    if not state.get("tree_done"):
        return {"action": "sample", "stage": "C"}
    return {"action": "update", "source": "tree_augmented", "group": "stage_c", "stage": "C"}


def loss_members(members: list[dict]) -> list[dict]:
    """Keep every trajectory in the group, including failures and tree suffixes."""
    if not members:
        raise RuntimeError("GRPO group is empty")
    allowed = {"independent", "tree"}
    bad = [member.get("origin") for member in members if member.get("origin") not in allowed]
    if bad:
        raise RuntimeError(f"off-policy rollout entered the GRPO group: {bad[:3]}")
    return members


def prefix_multiplicity(members: list[dict]) -> dict:
    """How often each trunk prefix action is reused by stitched tree trajectories.

    Equal trajectory weights still apply. This count is diagnostic only.
    """
    counts: dict[tuple[int, int], int] = {}
    for member in members:
        if member.get("origin") != "tree":
            continue
        trunk = int(member["prefix_source"])
        for step in range(int(member.get("prefix_len") or 0)):
            key = (trunk, step)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return {"shared_prefix_actions": 0, "max": 0, "mean": 0.0}
    values = list(counts.values())
    return {
        "shared_prefix_actions": len(values),
        "max": int(max(values)),
        "mean": float(np.mean(values)),
    }


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
