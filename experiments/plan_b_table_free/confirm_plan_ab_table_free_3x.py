#!/usr/bin/env python
"""3/3 SUCCESS_NEW confirmation for triage-PASS Plan A/B roots."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.plan_b_table_free.migrate_plan_ab_table_free import OUT, jsafe, load_ab_rows, validate_row  # noqa: E402


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    pass_ids = set(summary["pass_ids"])
    rows = [r for r in load_ab_rows() if r["expert_id"] in pass_ids]
    print(f"3/3 confirming {len(rows)} triage PASS roots", flush=True)
    results = []
    t0 = time.time()
    for i, row in enumerate(rows):
        t = time.time()
        out = validate_row(row, n_replay=3)
        out["wall_s"] = round(time.time() - t, 3)
        results.append(out)
        er = out["exact_replay"]
        print(
            f"[{i+1}/{len(rows)}] {out['expert_id']} {out['status']} "
            f"{er['n_pass']}/3 {out['wall_s']}s",
            flush=True,
        )
    by = Counter(r["status"] for r in results)
    by_plan = {
        p: Counter(r["status"] for r in results if r["plan_family"] == p) for p in ("A", "B")
    }
    final = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(results),
        "by_status": dict(by),
        "by_plan": {k: dict(v) for k, v in by_plan.items()},
        "wall_s": round(time.time() - t0, 2),
        "n_replay": 3,
        "pass_ids": [r["expert_id"] for r in results if r["status"] == "PASS"],
        "fail_ids": [r["expert_id"] for r in results if r["status"] != "PASS"],
    }
    (OUT / "confirm_3x_results.jsonl").write_text(
        "\n".join(json.dumps(jsafe(r), ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )
    (OUT / "confirm_3x_summary.json").write_text(
        json.dumps(jsafe(final), indent=2), encoding="utf-8"
    )
    print(json.dumps(final["by_status"], indent=2), flush=True)
    print(
        "PASS",
        len(final["pass_ids"]),
        "A",
        by_plan["A"].get("PASS", 0),
        "B",
        by_plan["B"].get("PASS", 0),
        flush=True,
    )
    return 0 if by.get("PASS", 0) == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
