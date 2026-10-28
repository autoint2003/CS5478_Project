#!/usr/bin/env python
"""Create data/experts_v3_stage1_verified/ by copying only Stage-1–eligible SUCCESS_NEW experts.

Does not modify data/experts_v2_official/.
"""

from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.precision.stage1_eligibility import (  # noqa: E402
    capture_fingerprint,
    lookup_evidence,
)

SRC = ROOT / "data/experts_v2_official"
DST = ROOT / "data/experts_v3_stage1_verified"
OUT = ROOT / "results/diagnostics/precision_baseline_rebuild"
AUDIT_EX = ROOT / "results/diagnostics/stage1_precision_integrity_audit/OFFICIAL_EXPERT_AUDIT.json"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if DST.exists():
        # recreate cleanly but never touch SRC
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    accepted = []
    excluded = []
    for pkl in sorted(SRC.rglob("*.pkl")):
        try:
            rec = pickle.loads(pkl.read_bytes())
        except Exception as exc:
            excluded.append({"expert_path": str(pkl.relative_to(ROOT)), "reason": f"load_error:{exc}"})
            continue
        if not isinstance(rec, dict) or not rec.get("SUCCESS_NEW"):
            continue
        # Stage-1 evidence: prefer capture_fp on stored expert capture
        try:
            fp = capture_fingerprint(rec["capture"])
        except Exception as exc:
            excluded.append(
                {
                    "expert_path": str(pkl.relative_to(ROOT)).replace("\\", "/"),
                    "source_ic_id": rec.get("source_ic_id"),
                    "reason": f"capture_fp_error:{exc}",
                    "SUCCESS_NEW": True,
                }
            )
            continue
        ev = lookup_evidence(fp)
        # Also allow matching via audit official join if registry miss
        ok = bool(ev and ev.get("SUCCESS_12S"))
        if not ok and AUDIT_EX.is_file():
            # fallback: source_ic eligibility from prior audit
            for row in json.loads(AUDIT_EX.read_text(encoding="utf-8")):
                if row.get("expert_path") == str(pkl.relative_to(ROOT)).replace("\\", "/"):
                    ok = row.get("stage1_eligibility") in ("ORIGINALLY_VERIFIED", "REVALIDATED_ONLY")
                    break
        if not ok:
            excluded.append(
                {
                    "expert_path": str(pkl.relative_to(ROOT)).replace("\\", "/"),
                    "source_ic_id": rec.get("source_ic_id"),
                    "branch_id": rec.get("branch_id"),
                    "capture_fp": fp,
                    "reason": "STAGE1_INELIGIBLE_OR_MISSING_EVIDENCE",
                    "SUCCESS_NEW": True,
                    "SUCCESS_12S": rec.get("SUCCESS_12S"),
                }
            )
            continue
        # Copy preserving relative branch layout
        rel = pkl.relative_to(SRC)
        dest = DST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pkl, dest)
        accepted.append(
            {
                "expert_path_v3": str(dest.relative_to(ROOT)).replace("\\", "/"),
                "expert_path_v2": str(pkl.relative_to(ROOT)).replace("\\", "/"),
                "expert_id": rec.get("expert_id"),
                "branch_id": rec.get("branch_id"),
                "source_ic_id": rec.get("source_ic_id"),
                "source_ic_path": rec.get("source_ic_path"),
                "capture_fp": fp,
                "SUCCESS_NEW": True,
                "SUCCESS_12S": rec.get("SUCCESS_12S"),
                "stage1_evidence_SUCCESS_12S": True if ev is None else bool(ev.get("SUCCESS_12S")),
                "n_actions": rec.get("n_actions"),
            }
        )

    with (OUT / "EXPERT_DATASET_V3_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        if accepted:
            w = csv.DictWriter(f, fieldnames=list(accepted[0].keys()))
            w.writeheader()
            w.writerows(accepted)
    with (OUT / "EXCLUDED_EXPERTS.csv").open("w", newline="", encoding="utf-8") as f:
        if excluded:
            w = csv.DictWriter(f, fieldnames=list(excluded[0].keys()))
            w.writeheader()
            w.writerows(excluded)
        else:
            f.write("expert_path,reason\n")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SRC.relative_to(ROOT)).replace("\\", "/"),
        "dest": str(DST.relative_to(ROOT)).replace("\\", "/"),
        "n_accepted": len(accepted),
        "n_excluded_success_new": len(excluded),
        "note": "Original data/experts_v2_official untouched. Copies only.",
    }
    (DST / "DATASET_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "EXPERT_V3_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
