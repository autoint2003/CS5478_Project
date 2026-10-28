#!/usr/bin/env python
"""Audit data/experts_v2_official for Stage-1 eligibility vs final SUCCESS_*."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/diagnostics/stage1_precision_integrity_audit"
OFFICIAL = ROOT / "data/experts_v2_official"
ELIG = OUT / "IC_ELIGIBILITY_AUDIT.json"


def main() -> int:
    elig_by = {}
    if ELIG.is_file():
        for r in json.loads(ELIG.read_text(encoding="utf-8")):
            elig_by[r["sample_id"]] = r

    rows = []
    for pkl in sorted(OFFICIAL.rglob("*.pkl")):
        # skip non-trajectory dumps if any
        try:
            blob = pickle.loads(pkl.read_bytes())
        except Exception as exc:
            rows.append({"path": str(pkl.relative_to(ROOT)), "error": str(exc)})
            continue
        if not isinstance(blob, dict):
            continue
        # Heuristic: expert records usually have SUCCESS_NEW or actions
        if not any(k in blob for k in ("SUCCESS_NEW", "actions", "branch_id", "source_ic_id", "handoff")):
            continue
        sid = str(
            blob.get("source_ic_id")
            or blob.get("source_sample_id")
            or blob.get("ic_id")
            or blob.get("sample_id")
            or ""
        )
        el = elig_by.get(sid) or {}
        handoff = blob.get("handoff") or {}
        rows.append(
            {
                "expert_path": str(pkl.relative_to(ROOT)).replace("\\", "/"),
                "branch_id": blob.get("branch_id") or blob.get("branch"),
                "source_ic_id": sid or None,
                "source_ic_path": blob.get("source_ic_path") or blob.get("ic_path"),
                "SUCCESS_NEW": blob.get("SUCCESS_NEW") if "SUCCESS_NEW" in blob else handoff.get("SUCCESS_NEW"),
                "SUCCESS_12S": blob.get("SUCCESS_12S") if "SUCCESS_12S" in blob else handoff.get("SUCCESS_12S"),
                "use_verified_lateral_fallback": (blob.get("precision_config") or blob.get("cfg") or {}).get(
                    "use_verified_lateral_fallback"
                ),
                "stage1_eligibility": el.get("eligibility"),
                "stage1_revalidate_SUCCESS_12S": el.get("revalidate_SUCCESS_12S"),
                "stage1_original_SUCCESS_12S": el.get("original_direct_SUCCESS_12S"),
                "pipeline_valid": bool(
                    el.get("eligibility") in ("ORIGINALLY_VERIFIED", "REVALIDATED_ONLY")
                    and (blob.get("SUCCESS_NEW") or handoff.get("SUCCESS_NEW"))
                ),
                "final_success_without_valid_stage1": bool(
                    (blob.get("SUCCESS_NEW") or handoff.get("SUCCESS_NEW"))
                    and el.get("eligibility") == "FAILED_12S"
                ),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "OFFICIAL_EXPERT_AUDIT.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        keys = list(rows[0].keys())
        with (OUT / "OFFICIAL_EXPERT_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    n = len(rows)
    n_new = sum(1 for r in rows if r.get("SUCCESS_NEW"))
    n_pipe = sum(1 for r in rows if r.get("pipeline_valid"))
    n_bad = sum(1 for r in rows if r.get("final_success_without_valid_stage1"))
    n_failed_s1 = sum(1 for r in rows if r.get("stage1_eligibility") == "FAILED_12S")
    n_missing_ic = sum(1 for r in rows if not r.get("source_ic_id") or r.get("stage1_eligibility") is None)

    md = f"""# OFFICIAL_EXPERT_AUDIT

Generated: {datetime.now(timezone.utc).isoformat()}

Source: `data/experts_v2_official/` joined to `IC_ELIGIBILITY_AUDIT.json`.

## Counts

| Metric | n |
|---|---:|
| Expert trajectory files parsed | {n} |
| Final `SUCCESS_NEW=True` | {n_new} |
| Full pipeline valid (Stage-1 12S eligible ∧ SUCCESS_NEW) | {n_pipe} |
| SUCCESS_NEW but Stage-1 revalidate FAILED_12S | {n_bad} |
| Trajectories whose source IC is FAILED_12S | {n_failed_s1} |
| Missing / unmatched source IC eligibility | {n_missing_ic} |

## Eligibility of source ICs (among matched)

```
{json.dumps(dict(Counter(r.get('stage1_eligibility') for r in rows)), indent=2)}
```

## Interpretation

A final successful expert trajectory does **not** prove Stage-1 12 s provenance.
`final_success_without_valid_stage1` flags accepted experts whose restored source IC
fails independent `SUCCESS_12S` revalidation (or whose IC eligibility is missing).

Artifacts: `OFFICIAL_EXPERT_AUDIT.csv`, `OFFICIAL_EXPERT_AUDIT.json`.
"""
    (OUT / "OFFICIAL_EXPERT_AUDIT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
