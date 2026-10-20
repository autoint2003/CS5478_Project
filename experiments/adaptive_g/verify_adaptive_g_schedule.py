"""Verify fixed source_aware_schedule for full Adaptive-G run."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.adaptive_independent_g import (
    load_merged_rows,
    schedule_source_counts,
    source_aware_schedule,
)

OUT = ROOT / "results/grpo/adaptive_g/full_run_v1/schedule_audit.json"
TRAIN = ROOT / "results/grpo/adaptive_g/datasets/merged_train.jsonl"


def prefix_ratio(schedule, n):
    part = schedule[:n]
    hard = sum(1 for r in part if r["source_dataset"] == "hard")
    return {
        "n": n,
        "heuristic": n - hard,
        "hard": hard,
        "hard_fraction": hard / n if n else None,
    }


def main() -> None:
    rows = load_merged_rows(TRAIN)
    unique = len(rows)
    by_source = Counter(r["source_dataset"] for r in rows)
    s1 = source_aware_schedule(rows, hard_fraction=0.5)
    s2 = source_aware_schedule(rows, hard_fraction=0.5)
    reproducible = [a["id"] for a in s1] == [b["id"] for b in s2]
    exposures = Counter(r["id"] for r in s1)
    report = {
        "unique_ic_count": unique,
        "unique_by_source": dict(by_source),
        "training_exposure_count_total_slots": len(s1),
        "source_exposure_ratio": schedule_source_counts(s1),
        "prefix_ratios": {str(n): prefix_ratio(s1, n) for n in (10, 20, 40, 80, 198)},
        "exposure_per_ic": {
            "min": min(exposures.values()),
            "max": max(exposures.values()),
            "heuristic_unique_seen": len({r["id"] for r in s1 if r["source_dataset"] == "heuristic"}),
            "hard_unique_seen": len({r["id"] for r in s1 if r["source_dataset"] == "hard"}),
            "hard_repeat_mean": (
                sum(exposures[i] for i in exposures if any(
                    r["id"] == i and r["source_dataset"] == "hard" for r in s1
                )) / max(by_source["hard"], 1)
            ),
            "histogram_hard_exposures": dict(Counter(
                exposures[i] for i in {r["id"] for r in s1 if r["source_dataset"] == "hard"}
            )),
        },
        "reproducible_with_same_call": reproducible,
        "first_40_sources": [r["source_dataset"] for r in s1[:40]],
        "assert_first_40_not_hard_dominated": prefix_ratio(s1, 40)["hard_fraction"] <= 0.6,
        "assert_first_40_near_half": abs(prefix_ratio(s1, 40)["hard_fraction"] - 0.5) <= 0.05,
    }
    if not report["assert_first_40_near_half"]:
        raise RuntimeError(f"first-40 hard fraction not ~0.5: {prefix_ratio(s1, 40)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "unique_ic_count": unique,
        "slots": len(s1),
        "first_40": prefix_ratio(s1, 40),
        "full_ratio": schedule_source_counts(s1)["hard_fraction"],
        "ok": True,
        "out": str(OUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
