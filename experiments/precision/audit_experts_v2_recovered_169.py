#!/usr/bin/env python
"""Audit the 169-record experts_v2_recovered set for rejected fallback provenance.

Does not modify or overwrite the recovered dataset.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "data/experts_v2_recovered/manifest.jsonl"
OUT = ROOT / "results/diagnostics/official_oracle_restore/RECOVERED_169_AUDIT.json"


def load_rows() -> list[dict]:
    return [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]


def inspect_pkl(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.is_file():
        return {"missing_pkl": True}
    blob = pickle.loads(path.read_bytes())
    result = blob.get("result") or {}
    logs = result.get("primitive_logs") or blob.get("primitive_logs") or {}
    phases = result.get("phases") or blob.get("phases") or []
    dx_mech = logs.get("dx_mechanism") or result.get("dx_mechanism")
    fallback = bool(logs.get("legacy_fallback_explicit")) or str(dx_mech or "").startswith("verified_lateral")
    if any("verified_lateral" in str(p) for p in phases):
        fallback = True
    return {
        "dx_mechanism": dx_mech,
        "legacy_fallback_explicit": bool(logs.get("legacy_fallback_explicit")),
        "fallback_entered": fallback,
        "skill_sequence": blob.get("skill_sequence"),
        "SUCCESS_NEW": blob.get("SUCCESS_NEW"),
        "SUCCESS_12S": blob.get("SUCCESS_12S"),
        "status": blob.get("status") or result.get("status"),
        "oracle_implementation": blob.get("oracle_implementation"),
        "config_id": blob.get("config_id"),
    }


def main() -> int:
    rows = load_rows()
    audited = []
    for row in rows:
        rel = row.get("record_path") or ""
        detail = inspect_pkl(rel) if rel else {"missing_pkl": True}
        audited.append(
            {
                "branch_id": row.get("branch_id"),
                "source_ic_id": row.get("source_ic_id"),
                "skill_sequence": row.get("skill_sequence") or detail.get("skill_sequence"),
                "SUCCESS_NEW": row.get("SUCCESS_NEW"),
                "SUCCESS_12S": row.get("SUCCESS_12S"),
                "record_path": rel,
                **detail,
            }
        )
    rejected = [r for r in audited if r.get("fallback_entered")]
    retained_candidates = [r for r in audited if not r.get("fallback_entered")]
    report = {
        "n_total": len(audited),
        "n_rejected_fallback": len(rejected),
        "n_retained_candidate": len(retained_candidates),
        "by_branch_total": dict(Counter(r["branch_id"] for r in audited)),
        "by_branch_rejected_fallback": dict(Counter(r["branch_id"] for r in rejected)),
        "by_branch_retained_candidate": dict(Counter(r["branch_id"] for r in retained_candidates)),
        "by_skill_total": dict(Counter(str(r.get("skill_sequence")) for r in audited)),
        "by_skill_retained": dict(Counter(str(r.get("skill_sequence")) for r in retained_candidates)),
        "note": (
            "Rejected = entered verified_lateral / verified_lateral_explicit. "
            "Retained candidates still require re-validation under official ST-only Oracle "
            "before production admission."
        ),
        "rejected": rejected,
        "retained_candidates": [
            {
                "branch_id": r["branch_id"],
                "source_ic_id": r["source_ic_id"],
                "skill_sequence": r.get("skill_sequence"),
                "dx_mechanism": r.get("dx_mechanism"),
            }
            for r in retained_candidates
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"audited={len(audited)} rejected_fallback={len(rejected)} "
        f"retained_candidate={len(retained_candidates)} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
