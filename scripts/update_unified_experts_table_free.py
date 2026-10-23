#!/usr/bin/env python
"""Rewrite unified_experts to keep only table-free verified A/B + unchanged Plan C.

Requires confirm_3x_summary.json from the migration pipeline. Does not regenerate
Plan C. Preserves schema, splits by physical root, and 39D/7D interface.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.migrate_plan_ab_table_free import OUT  # noqa: E402

DATA = ROOT / "data" / "unified_experts"
CONFIRM = OUT / "confirm_3x_summary.json"


def main():
    if not CONFIRM.is_file():
        raise SystemExit(f"missing {CONFIRM}")
    confirm = json.loads(CONFIRM.read_text(encoding="utf-8"))
    pass_ids = set(confirm["pass_ids"])
    if not pass_ids:
        raise SystemExit("no PASS ids in confirm_3x_summary")

    manifest_src = DATA / "manifest.jsonl"
    rows = []
    with manifest_src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("plan_family") == "C":
                rows.append(r)
            elif r.get("expert_id") in pass_ids:
                r = dict(r)
                r["scene"] = "GROUND_SCENE"
                r["environment_version"] = "ground_ballistic_v1"
                r["exact_replay"] = {
                    "n": 3,
                    "n_pass": 3,
                    "note": "table-free migration; identity restore + legal tape + SUCCESS_NEW",
                }
                r["provenance"] = dict(r.get("provenance") or {})
                r["provenance"]["table_free_migration_utc"] = datetime.now(timezone.utc).isoformat()
                r["provenance"]["migration_report"] = str(
                    (OUT / "confirm_3x_summary.json").relative_to(ROOT)
                ).replace("\\", "/")
                rows.append(r)

    by_plan = {"A": 0, "B": 0, "C": 0}
    for r in rows:
        by_plan[r["plan_family"]] = by_plan.get(r["plan_family"], 0) + 1

    # Preserve prior split labels for kept rows; rebuild index.
    index = {
        "schema_version": "unified_experts_v1",
        "n_experts": len(rows),
        "by_plan": by_plan,
        "table_free": True,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "experts": [
            {
                "expert_id": r["expert_id"],
                "plan_family": r["plan_family"],
                "physical_root_id": r["physical_root_id"],
                "split": r.get("split"),
                "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                "checksum": r.get("data_integrity_checksum"),
            }
            for r in rows
        ],
    }
    splits = {"train": [], "val": [], "test": [], "regression_fixture": []}
    for r in rows:
        sp = r.get("split") or "train"
        splits.setdefault(sp, []).append(r["expert_id"])

    # Backup tracked manifest once beside migration results (not under results overwrite of old).
    bak = OUT / "pre_update_unified_manifest.jsonl"
    if not bak.exists():
        shutil.copy2(manifest_src, bak)

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    (DATA / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (DATA / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    report = {
        "by_plan": by_plan,
        "n_experts": len(rows),
        "n_ab_pass_3x": len(pass_ids),
        "confirm": str(CONFIRM.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "dataset_update.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
