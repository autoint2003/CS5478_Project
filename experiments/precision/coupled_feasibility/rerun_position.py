#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.coupled_feasibility.run_paired import (  # noqa: E402
    KNOWN,
    OUT,
    cfg,
    jsafe,
    load_rows,
    make_env,
    run_position_pair,
)


def main() -> int:
    rows = load_rows()
    by_id = {r["sample_id"]: r for r in rows}
    st_fail = [
        r
        for r in rows
        if r.get("class") == "PRECISION_REQUIRED_BUT_FAILED"
        and ((r.get("precision") or {}).get("first_failed_primitive") in ("support_transfer", "DX1", "dz"))
    ]
    successes = [r for r in rows if r.get("class") == "PRECISION_REQUIRED_AND_SUCCESSFUL"]
    pos_cohort = st_fail[:20] + [by_id[k] for k in KNOWN if k in by_id] + successes[:5]
    seen: set[str] = set()
    pos_unique = []
    for r in pos_cohort:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        pos_unique.append(r)
    env = make_env()
    pcfg = cfg()
    pos_rows = []
    print("position pairs", len(pos_unique), flush=True)
    for i, row in enumerate(pos_unique):
        sid = row["sample_id"]
        print(f"[pos {i+1}/{len(pos_unique)}] {sid}", flush=True)
        try:
            pos_rows.append(run_position_pair(env, row, pcfg))
        except Exception as exc:
            pos_rows.append({"sample_id": sid, "error": str(exc), "traceback": traceback.format_exc(limit=4)})
    (OUT / "POSITION_PAIRS.json").write_text(json.dumps(jsafe(pos_rows), indent=2), encoding="utf-8")
    print("cand_ok", sum(1 for r in pos_rows if (r.get("candidate") or {}).get("ok")), flush=True)
    print("base_ok", sum(1 for r in pos_rows if r.get("baseline_ok")), flush=True)
    print("errors", sum(1 for r in pos_rows if r.get("error")), flush=True)
    print(Counter((r.get("candidate") or {}).get("reason") for r in pos_rows).most_common(15), flush=True)
    print(
        "cand_NEW",
        sum(1 for r in pos_rows if (r.get("candidate_handoff") or {}).get("SUCCESS_NEW")),
        "base_NEW",
        sum(1 for r in pos_rows if (r.get("baseline_handoff") or {}).get("SUCCESS_NEW")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
