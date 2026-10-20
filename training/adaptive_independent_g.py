"""Adaptive Independent-G exploration for K48 GRPO.

Cumulative independent rollouts only: G = 8 → 16 → 32 → 64.
Stop expanding once SUCCESS_12S appears in the cumulative group.
No tree trunks, shared prefixes, or privileged recovery actions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.adaptive_hybrid import (
    CATEGORIES,
    HORIZONTAL,
    SLIDING,
    VERTICAL,
    balanced_schedule,
    primary_category,
)
from training.k48_config import ROOT

STAGE_G = (8, 16, 32, 64)
DEFAULT_DATASETS = ROOT / "results" / "grpo" / "adaptive_g" / "datasets"


def independent_g_decision(state: dict) -> dict:
    """Decide whether to expand the independent group or update.

    ``state`` tracks cumulative SUCCESS_12S counts and how many rollouts are done.
    Stages are labeled A/B/C/D for G8/G16/G32/G64.
    """
    done = int(state.get("rollouts_done") or 0)
    success = int(state.get("success_count") or 0)
    if done < 8:
        raise RuntimeError("Stage A (G8) must already be sampled before deciding")
    if success > 0:
        stage = {8: "A", 16: "B", 32: "C", 64: "D"}[done] if done in (8, 16, 32, 64) else "X"
        return {
            "action": "update",
            "source": f"independent_g{done}",
            "group": f"g{done}",
            "stage": stage,
            "final_g": done,
            "stop_reason": f"G{done}_stop",
        }
    if done == 8:
        return {"action": "sample", "stage": "B", "add": 8, "target_g": 16}
    if done == 16:
        return {"action": "sample", "stage": "C", "add": 16, "target_g": 32}
    if done == 32:
        return {"action": "sample", "stage": "D", "add": 32, "target_g": 64}
    if done == 64:
        return {
            "action": "update",
            "source": "independent_g64",
            "group": "g64",
            "stage": "D",
            "final_g": 64,
            "stop_reason": "G64_no_success" if success == 0 else "G64_success",
        }
    raise RuntimeError(f"unexpected independent-G state done={done} success={success}")


def stop_bucket(final_g: int, found_success: bool) -> str:
    if found_success and final_g == 8:
        return "G8_stop"
    if found_success and final_g == 16:
        return "G16_stop"
    if found_success and final_g == 32:
        return "G32_stop"
    if found_success and final_g == 64:
        return "G64_success"
    if not found_success and final_g == 64:
        return "G64_no_success"
    return f"G{final_g}_other"


def load_merged_rows(manifest: Path | str) -> list[dict]:
    path = Path(manifest)
    if not path.is_absolute():
        path = ROOT / path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty merged manifest {path}")
    seen = set()
    for row in rows:
        snap = row["snapshot"]
        if snap in seen:
            raise RuntimeError(f"duplicate snapshot in {path}: {snap}")
        seen.add(snap)
        cats = list(row.get("categories") or [])
        if not cats:
            raise RuntimeError(f"{row.get('id')} has empty categories")
        row["categories"] = sorted(set(cats))
        row["primary_category"] = row.get("primary_category") or primary_category(row["categories"])
    return rows


def unique_ic_shuffle_schedule(rows: list[dict], rng) -> list[dict]:
    """One epoch: every unique IC exactly once, random order.

    No source-aware or skill-aware reweighting. ``rng`` must be a NumPy Generator
    (or compatible object with ``permutation``). Validation must not use this.
    """
    if not rows:
        raise RuntimeError("unique_ic_shuffle_schedule received an empty train pool")
    seen = set()
    unique = []
    for row in rows:
        snap = row["snapshot"]
        if snap in seen:
            raise RuntimeError(f"duplicate snapshot before shuffle: {snap}")
        seen.add(snap)
        unique.append(row)
    order = np.asarray(rng.permutation(len(unique)), dtype=np.int64)
    return [unique[int(index)] for index in order]


def source_aware_schedule(rows: list[dict], hard_fraction: float = 0.5) -> list[dict]:
    """Historical Heuristic:Hard rebalancing schedule (kept for ablation only).

    Formal Adaptive-G training uses ``unique_ic_shuffle_schedule`` instead.
    Each unique Heuristic IC appears once. Hard ICs are cycled (with replacement
    across the schedule) so that roughly ``hard_fraction`` of updates are Hard.
    Within each source, primary-category round-robin is preserved.
    Validation must not use this function.
    """
    if not 0.0 < float(hard_fraction) < 1.0:
        raise RuntimeError(f"hard_fraction must be in (0,1), got {hard_fraction}")
    heuristic = [row for row in rows if str(row.get("source_dataset") or "") == "heuristic"]
    hard = [row for row in rows if str(row.get("source_dataset") or "") == "hard"]
    other = [row for row in rows if str(row.get("source_dataset") or "") not in ("heuristic", "hard")]
    if other:
        raise RuntimeError(f"unknown source_dataset values: {sorted({r.get('source_dataset') for r in other})}")
    if not heuristic and not hard:
        raise RuntimeError("merged train has neither heuristic nor hard rows")
    if not hard:
        return balanced_schedule(heuristic)
    if not heuristic:
        return balanced_schedule(hard)

    h_sched = balanced_schedule(heuristic)
    hard_sched = balanced_schedule(hard)
    # Target: hard_fraction of total draws are hard; each heuristic once.
    # Interleave so *prefixes* also stay near the target (critical when limit_ics truncates).
    n_h = len(h_sched)
    n_hard_draws = max(len(hard_sched), int(round(n_h * hard_fraction / (1.0 - hard_fraction))))
    schedule = []
    hard_i = 0
    hard_draws = 0
    for index, row in enumerate(h_sched):
        schedule.append(row)
        # After seeing (index+1) heuristic draws, hard_draws should be ≈ f/(1-f) * (index+1).
        target_hard = int(round((index + 1) * hard_fraction / (1.0 - hard_fraction)))
        target_hard = min(target_hard, n_hard_draws)
        while hard_draws < target_hard:
            schedule.append(hard_sched[hard_i % len(hard_sched)])
            hard_i += 1
            hard_draws += 1
    while hard_draws < n_hard_draws:
        schedule.append(hard_sched[hard_i % len(hard_sched)])
        hard_i += 1
        hard_draws += 1
    return schedule


def schedule_source_counts(schedule: list[dict]) -> dict:
    counts = {"heuristic": 0, "hard": 0, "unique_heuristic": 0, "unique_hard": 0}
    seen_h, seen_hard = set(), set()
    for row in schedule:
        src = str(row.get("source_dataset") or "")
        if src == "heuristic":
            counts["heuristic"] += 1
            seen_h.add(row["snapshot"])
        elif src == "hard":
            counts["hard"] += 1
            seen_hard.add(row["snapshot"])
    counts["unique_heuristic"] = len(seen_h)
    counts["unique_hard"] = len(seen_hard)
    total = max(len(schedule), 1)
    counts["hard_fraction"] = counts["hard"] / total
    counts["heuristic_fraction"] = counts["heuristic"] / total
    by_primary = {name: 0 for name in CATEGORIES}
    for row in schedule:
        by_primary[primary_category(row["categories"])] += 1
    counts["primary"] = by_primary
    return counts


def assert_no_train_valid_overlap(train_rows: list[dict], valid_rows: list[dict]) -> list[str]:
    """Return conflict descriptions; raise if snapshot or fingerprint overlaps."""
    conflicts = []
    valid_snaps = {row["snapshot"] for row in valid_rows}
    valid_ids = {row["id"] for row in valid_rows}
    valid_fp = {row["fingerprint"] for row in valid_rows if row.get("fingerprint")}
    for row in train_rows:
        if row["snapshot"] in valid_snaps:
            conflicts.append(f"snapshot overlap {row['id']} ↔ validation")
        if row["id"] in valid_ids:
            conflicts.append(f"id overlap {row['id']}")
        fp = row.get("fingerprint")
        if fp and fp in valid_fp:
            conflicts.append(f"fingerprint overlap {row['id']} ({fp[:12]}…)")
    if conflicts:
        raise RuntimeError("train/validation leak: " + "; ".join(conflicts[:8]))
    return conflicts
