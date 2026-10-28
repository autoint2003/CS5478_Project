#!/usr/bin/env python
"""Populate data/stage1_eligibility_evidence from the integrity audit CSV/JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.precision.stage1_eligibility import (  # noqa: E402
    EVAL_CONFIG,
    DEFAULT_EVIDENCE_PATH,
    save_evidence_record,
)

AUDIT = ROOT / "results/diagnostics/stage1_precision_integrity_audit/IC_ELIGIBILITY_AUDIT.json"


def main() -> int:
    rows = json.loads(AUDIT.read_text(encoding="utf-8"))
    n_ok = n_fail = n_skip = 0
    # Fresh registry
    DEFAULT_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_EVIDENCE_PATH.write_text(
        json.dumps({"eval_config": EVAL_CONFIG, "by_capture_fp": {}}, indent=2),
        encoding="utf-8",
    )
    for r in rows:
        fp = r.get("capture_fp")
        if not fp:
            n_skip += 1
            continue
        ok = bool(r.get("revalidate_SUCCESS_12S"))
        save_evidence_record(
            capture_fp=str(fp),
            sample_id=r.get("sample_id"),
            SUCCESS_12S=ok,
            SUCCESS_NEW=bool(r.get("revalidate_SUCCESS_NEW")) if r.get("revalidate_SUCCESS_NEW") is not None else None,
            reason=r.get("revalidate_reason"),
            extra={
                "eligibility_audit": r.get("eligibility"),
                "ic_path": r.get("ic_path"),
                "source": "stage1_precision_integrity_audit",
            },
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1
    print(f"wrote {DEFAULT_EVIDENCE_PATH} ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
