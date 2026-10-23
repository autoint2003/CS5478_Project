"""Load and validate experts_v2 records for training readiness (no training)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from data_generation.expert_pipeline.branches import BRANCHES
from data_generation.expert_pipeline.provenance import (
    ACTIVE_DATASET,
    DATASET_VERSION,
    ORACLE_IMPLEMENTATION,
    row_allowed_for_training,
)

ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    manifest = path or (ACTIVE_DATASET / "manifest.jsonl")
    if not manifest.is_file():
        return []
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_expert_record(row_or_path: dict[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(row_or_path, dict):
        rel = row_or_path.get("record_path")
        if not rel:
            raise KeyError("manifest row missing record_path")
        path = Path(rel)
        if not path.is_file():
            path = ROOT / rel
    else:
        path = Path(row_or_path)
    blob = pickle.loads(path.read_bytes())
    if blob.get("dataset_version") != DATASET_VERSION:
        raise ValueError(f"unexpected dataset_version {blob.get('dataset_version')}")
    return blob


def audit_manifest(path: Path | None = None) -> dict[str, Any]:
    rows = load_manifest(path)
    by_branch = {b: 0 for b in BRANCHES}
    rejected = []
    missing_pkl = []
    bad_oracle = []
    for row in rows:
        bid = str(row.get("branch_id"))
        if bid in by_branch:
            by_branch[bid] += 1
        if not row_allowed_for_training(row):
            rejected.append(row.get("attempt_id") or row.get("expert_id"))
            continue
        if bid.endswith("_PRECISION") or bid == "D_PRECISION":
            if row.get("oracle_implementation") != ORACLE_IMPLEMENTATION:
                bad_oracle.append(row.get("attempt_id"))
        rel = row.get("record_path")
        if rel:
            p = Path(rel)
            if not p.is_file() and not (ROOT / rel).is_file():
                missing_pkl.append(rel)
    return {
        "n_rows": len(rows),
        "by_branch": by_branch,
        "n_rejected_by_admission": len(rejected),
        "n_missing_records": len(missing_pkl),
        "n_bad_oracle": len(bad_oracle),
        "ok": len(rejected) == 0 and len(missing_pkl) == 0 and len(bad_oracle) == 0,
    }


def write_splits(manifest_path: Path | None = None, *, seed: int = 0, val_frac: float = 0.1, test_frac: float = 0.1) -> dict[str, Any]:
    """Split by source_ic_id so the same physical root never crosses partitions."""
    import random

    rows = load_manifest(manifest_path)
    roots = sorted({str(r.get("source_ic_id")) for r in rows})
    rng = random.Random(int(seed))
    rng.shuffle(roots)
    n = len(roots)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = set(roots[:n_test])
    val = set(roots[n_test : n_test + n_val])
    train = set(roots[n_test + n_val :])
    # Ensure no leakage
    assert not (train & val or train & test or val & test)
    split_of = {}
    for r in roots:
        if r in test:
            split_of[r] = "test"
        elif r in val:
            split_of[r] = "val"
        else:
            split_of[r] = "train"
    out = {
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "by_source_ic_id": split_of,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
    }
    dest = (manifest_path.parent if manifest_path else ACTIVE_DATASET) / "splits.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
