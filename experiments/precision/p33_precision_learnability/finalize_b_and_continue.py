"""Finalize Phase B from saved ICs; continue generation to target_total if needed."""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p33_precision_learnability.phase_b_generate import (  # noqa: E402
    OUT,
    ICS,
    SEED,
    run_phase_b,
    _write_csv,
    _write_md,
    load_existing_p1_roots,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def finalize_from_disk() -> dict:
    rows = []
    for p in sorted(ICS.glob("*.pkl")):
        blob = pickle.loads(p.read_bytes())
        rows.append(
            {
                "tag": blob.get("tag") or p.stem,
                "lineage_id": blob.get("lineage_id"),
                "physical_root": blob.get("physical_root"),
                "n_actions": int(len(np.asarray(blob.get("actions")).reshape(-1, 7))),
                "class": blob.get("class", "P1"),
                "oracle_status": blob.get("oracle_status"),
                "ic_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "family": (blob.get("construction") or {}).get("family"),
                "target_dx_mm": (blob.get("construction") or {}).get("requested_dx_mm"),
                "target_dz_mm": (blob.get("construction") or {}).get("requested_dz_mm"),
            }
        )
    by_root = defaultdict(list)
    for r in rows:
        by_root[r["physical_root"]].append(r)
    roots = list(by_root.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(roots)
    n = len(roots)
    n_test = max(1, n // 5) if n >= 3 else 0
    n_val = max(1, n // 5) if n >= 3 else 0
    test_roots = set(roots[:n_test])
    val_roots = set(roots[n_test : n_test + n_val])
    for r in rows:
        if r["physical_root"] in test_roots:
            r["split"] = "test"
        elif r["physical_root"] in val_roots:
            r["split"] = "val"
        else:
            r["split"] = "train"
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    man = {
        "generated_at": _utc(),
        "seed": SEED,
        "p1_total": len(rows),
        "p1_train": len(train),
        "p1_val": len(val),
        "p1_test": len(test),
        "unique_train_roots": len({r["physical_root"] for r in train}),
        "unique_val_roots": len({r["physical_root"] for r in val}),
        "unique_test_roots": len({r["physical_root"] for r in test}),
        "existing_audit_p1_roots": sorted(load_existing_p1_roots()),
        "note": "Partial Phase B yield; Oracle tapes long (~1.5–2k acts). Targets 100/200 not reached in wall-clock.",
        "train": train,
        "val": val,
        "test": test,
    }
    (OUT / "GENERATED_DATASET.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    _write_csv(OUT / "GENERATED_P1.csv", rows)
    report = f"""# GENERATION_REPORT — P33 Phase B (partial finalize)

Generated: {_utc()}

## Yield (actual)

| Metric | Value |
|---|---:|
| P1 ICs on disk | {len(rows)} |
| Unique physical roots | {len(by_root)} |
| Train / Val / Test | {len(train)} / {len(val)} / {len(test)} |
| Pre-expansion audited P1 train roots | {len(load_existing_p1_roots())} |

## Bottleneck

Production `run_precision` on Precision-eligible offsets produces **long** trajectories (≈1.7–2.1k actions).
Wall-clock limited collection: ~4 P1 / 15 min. Targets 100/200 not reached without multi-hour run.
Acceptance criteria and Oracle **unchanged**.

## Regime

{Counter((r.get('family'), r.get('target_dx_mm'), r.get('target_dz_mm')) for r in rows)}

## Decision for Phase C

Proceed with available nested subsets (all / half) for learnability probe, documenting undersized N.
Combine with audited pre-expansion P1 where packaging is possible.
"""
    _write_md(OUT / "GENERATION_REPORT.md", report)
    print(f"finalized P1={len(rows)} train={len(train)} val={len(val)} test={len(test)}", flush=True)
    return man


if __name__ == "__main__":
    # Continue a bit toward 25 total if below
    n = len(list(ICS.glob("*.pkl"))) if ICS.is_dir() else 0
    if n < 25:
        print(f"continue generation from {n} toward 25...", flush=True)
        # need 25 total in pool before split; ask for 25 + small holdout
        run_phase_b(target_p1=max(1, 25 - n), val_n=5, test_n=5)
    finalize_from_disk()
