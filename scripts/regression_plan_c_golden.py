#!/usr/bin/env python
"""Golden Plan C SUCCESS_NEW 3/3 regression (gbi_010, gbi_022, gbi_014 sidewall)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.consolidation.freeze_experts import discover_plan_c_roots, replay_plan_c  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "plan_ab_table_free_migration" / "plan_c_golden_regression.json"
GOLDEN = ("gbi_010", "gbi_022", "gbi_014")


def main():
    pairs = discover_plan_c_roots()
    by_fam = {}
    for rid, path in pairs:
        for fam in GOLDEN:
            if fam in rid and fam not in by_fam:
                by_fam[fam] = (rid, path)
    rows = []
    ok_all = True
    for fam in GOLDEN:
        if fam not in by_fam:
            rows.append({"family": fam, "ok": False, "error": "missing expert"})
            ok_all = False
            continue
        rid, path = by_fam[fam]
        print(f"Replaying {fam} via {rid} ...", flush=True)
        payload, _raw, replay_rows = replay_plan_c(rid, n_replay=3, src=path)
        exact = payload["exact_replay"]
        n_ok = int(exact["n_pass"])
        row = {
            "family": fam,
            "expert_id": payload.get("expert_id") or rid,
            "source": str(path).replace("\\", "/"),
            "n_pass": n_ok,
            "n": 3,
            "ok": n_ok == 3,
            "exact_replay": exact,
            "replay_rows": replay_rows,
        }
        rows.append(row)
        ok_all = ok_all and row["ok"]
        print(f"  {fam}: {n_ok}/3 SUCCESS_NEW", flush=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok_all,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": ok_all, "summary": [(r["family"], r.get("n_pass")) for r in rows]}, indent=2))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
