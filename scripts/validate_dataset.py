"""Validate experts_v2 admission, provenance, and record loadability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.expert_pipeline.loader import (  # noqa: E402
    audit_manifest,
    load_expert_record,
    load_manifest,
)
from data_generation.expert_pipeline.provenance import (  # noqa: E402
    ACTIVE_DATASET,
    LEGACY_MANIFEST,
    row_allowed_for_training,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate expert-manifest admission.")
    parser.add_argument("--manifest", type=Path, default=ACTIVE_DATASET / "manifest.jsonl")
    parser.add_argument("--load-sample", type=int, default=3, help="Load N accepted records")
    args = parser.parse_args(argv)

    rows = load_manifest(args.manifest)
    audit = audit_manifest(args.manifest)
    rejected = [r.get("expert_id") or r.get("branch_id") for r in rows if not row_allowed_for_training(r)]

    loaded = []
    errors = []
    for row in rows[: max(0, args.load_sample)]:
        try:
            blob = load_expert_record(row)
            loaded.append(
                {
                    "attempt_id": blob.get("attempt_id"),
                    "branch_id": blob.get("branch_id"),
                    "n_actions": blob.get("n_actions"),
                    "SUCCESS_NEW": blob.get("SUCCESS_NEW"),
                }
            )
        except Exception as exc:
            errors.append({"row": row.get("attempt_id"), "error": str(exc)})

    report = {
        "manifest": str(args.manifest),
        "n_rows": len(rows),
        "n_rejected_admission": len(rejected),
        "legacy_manifest_present": LEGACY_MANIFEST.is_file(),
        "legacy_namespace": "data/unified_experts",
        "active_namespace": "data/experts_v2",
        "audit": audit,
        "sample_loaded": loaded,
        "load_errors": errors,
    }
    print(json.dumps(report, indent=2))
    ok = audit.get("ok", False) and not errors and len(rejected) == 0
    # Empty dataset is valid before generation.
    if len(rows) == 0:
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
