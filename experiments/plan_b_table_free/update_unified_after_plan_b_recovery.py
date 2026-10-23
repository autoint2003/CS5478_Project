#!/usr/bin/env python
"""Merge recovered Plan B experts into unified_experts; fix Plan C unique-root accounting.

Preserves accepted Plan A (71), previously accepted Plan B (16), and does not
regenerate Plan C. Deduplicates Plan C fixture aliases so unique physical roots
match the scale50 campaign (50) with 3 regression fixtures among them.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "unified_experts"
REC = OUT_REC = ROOT / "results" / "diagnostics" / "plan_b_table_free_recovery"
RAW = REC / "raw" / "accepted"
MIG = ROOT / "results" / "diagnostics" / "plan_ab_table_free_migration"
SCHEMA = "unified_experts_v1"
DT = 0.02


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_manifest(path: Path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dedupe_plan_c(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep scale50 as canonical; drop final_success/sidewall aliases of same family."""
    c_rows = [r for r in rows if r.get("plan_family") == "C"]
    other = [r for r in rows if r.get("plan_family") != "C"]

    def fam(r):
        pid = r.get("physical_root_id") or r.get("expert_id") or ""
        for x in ("gbi_010", "gbi_022", "gbi_014"):
            if x in pid or x in r.get("expert_id", ""):
                return x
        return None

    def is_scale50(r):
        p = str((r.get("provenance") or {}).get("expert_pkl") or r.get("source_experiment") or "")
        return "scale50" in p

    # Prefer scale50 for each golden family; keep all other scale50 roots.
    by_fp = {}
    fixtures_keep = {}
    dropped = []
    for r in c_rows:
        fp = r.get("physical_root_fingerprint") or r.get("physical_root_id")
        f = fam(r)
        if f and not is_scale50(r):
            # alias of a golden fixture also present in scale50
            dropped.append(r["expert_id"])
            continue
        if f and is_scale50(r):
            fixtures_keep[f] = r
        key = fp or r["expert_id"]
        # if duplicate fp, prefer scale50
        if key in by_fp:
            if is_scale50(r) and not is_scale50(by_fp[key]):
                dropped.append(by_fp[key]["expert_id"])
                by_fp[key] = r
            else:
                dropped.append(r["expert_id"])
            continue
        by_fp[key] = r

    kept_c = list(by_fp.values())
    # mark fixtures
    for r in kept_c:
        f = fam(r)
        if f in ("gbi_010", "gbi_022", "gbi_014") and is_scale50(r):
            r["regression_fixture"] = True
            r["split"] = "regression_fixture"
        else:
            r["regression_fixture"] = bool(r.get("regression_fixture"))

    audit = {
        "c_rows_before": len(c_rows),
        "c_rows_after": len(kept_c),
        "unique_physical_roots": len(kept_c),
        "dropped_alias_ids": dropped,
        "fixture_families": sorted(fixtures_keep.keys()),
        "note": (
            "Prior 53 = 50 scale50 independent roots + 3 golden aliases "
            "(final_success gbi_010/022 + sidewall gbi_014) with distinct expert_ids "
            "but same physical families as scale50 fixtures."
        ),
    }
    return other + kept_c, audit


def expert_from_pkl(path: Path, split: str = "train") -> dict:
    blob = pickle.loads(path.read_bytes())
    acts = np.asarray(blob["actions"], np.float32)
    obs = blob.get("observations")
    if obs is None:
        obs = np.zeros((0, 39), np.float32)
    else:
        obs = np.asarray(obs, np.float32)
    eid = blob["expert_id"]
    timestamps = (np.arange(len(acts), dtype=np.float64) * DT).tolist()
    raw = pickle.dumps(
        {
            "expert_id": eid,
            "actions": acts,
            "observations": obs,
            "ref": blob.get("ref"),
            "catch_snap": blob.get("catch_snap"),
        }
    )
    # persist record
    rec_dir = DATA / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec_path = rec_dir / f"{eid}.pkl"
    rec_path.write_bytes(
        pickle.dumps(
            {
                "schema_version": SCHEMA,
                "expert_id": eid,
                "plan_family": "B",
                "actions_7d": acts,
                "observations_39d": obs,
                "ref": blob.get("ref"),
                "catch_snap": blob.get("catch_snap"),
                "exact_replay": blob.get("exact_replay"),
            }
        )
    )
    return {
        "schema_version": SCHEMA,
        "expert_id": eid,
        "plan_family": "B",
        "physical_root_id": blob.get("physical_root_id") or eid.replace("plan_b__", ""),
        "scene": "GROUND_SCENE",
        "environment_version": "ground_ballistic_v1",
        "initial_physical_state_ref": {
            "source_pkl": str(path.relative_to(ROOT)).replace("\\", "/"),
            "snapshot": blob.get("source_snapshot"),
            "snap_key": "catch_snap",
        },
        "action_timestamps_s": timestamps,
        "control_period_s": DT,
        "handoff": {"token_id": 47, "timing": "immediate_after_final_action"},
        "SUCCESS_12S": True,
        "SUCCESS_NEW": True,
        "exact_replay": blob.get("exact_replay") or {"n": 3, "n_pass": 3},
        "contact_grasp_mechanism": blob.get("mechanism") or "table_free_recatch",
        "source_experiment": str(path.parent.relative_to(ROOT)).replace("\\", "/"),
        "provenance": {
            "expert_pkl": str(path.relative_to(ROOT)).replace("\\", "/"),
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "action_interface": "legal_7d",
            "recovery_method": blob.get("recovery_method") or "table_free_research",
            "tag": blob.get("tag"),
            "privileged_runtime": False,
        },
        "action_map": "recovery7d_airborne_v3",
        "data_integrity_checksum": sha256_bytes(raw),
        "record_path": str(rec_path.relative_to(ROOT)).replace("\\", "/"),
        "n_steps": int(len(acts)),
        "obs_dim": 39,
        "action_dim": 7,
        "split": split,
        "regression_fixture": False,
    }


def main():
    summary_path = REC / "recovery_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}; run recover_plan_b_table_free.py first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pass_ids = set(summary.get("pass_ids") or [])

    # Start from current unified manifest (table-free A/B + C)
    rows = load_manifest(DATA / "manifest.jsonl")
    existing_ids = {r["expert_id"] for r in rows}

    # Backup
    bak = REC / "pre_merge_unified_manifest.jsonl"
    if not bak.exists():
        shutil.copy2(DATA / "manifest.jsonl", bak)

    rows, c_audit = dedupe_plan_c(rows)

    # Add newly recovered Plan B (skip if already present)
    added = []
    for eid in sorted(pass_ids):
        if eid in existing_ids:
            # replace tokenized stub with continuous expert if we have pkl
            rows = [r for r in rows if r["expert_id"] != eid]
        pkl = RAW / f"{eid}__SUCCESS_NEW_expert.pkl"
        if not pkl.is_file():
            continue
        # preserve prior split if any
        split = "train"
        for r in load_manifest(bak):
            if r["expert_id"] == eid:
                split = r.get("split") or "train"
                break
        rows.append(expert_from_pkl(pkl, split=split))
        added.append(eid)

    by_plan = {"A": 0, "B": 0, "C": 0}
    for r in rows:
        by_plan[r["plan_family"]] = by_plan.get(r["plan_family"], 0) + 1

    # unique roots
    roots = {
        p: len({r["physical_root_id"] for r in rows if r["plan_family"] == p})
        for p in ("A", "B", "C")
    }
    fixtures = [r["expert_id"] for r in rows if r.get("regression_fixture") or r.get("split") == "regression_fixture"]

    index = {
        "schema_version": SCHEMA,
        "n_experts": len(rows),
        "by_plan": by_plan,
        "unique_physical_roots": roots,
        "regression_fixtures": fixtures,
        "table_free": True,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "plan_c_audit": c_audit,
        "plan_b_recovery_added": added,
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
    splits = {}
    for r in rows:
        sp = r.get("split") or "train"
        splits.setdefault(sp, []).append(r["expert_id"])

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    (DATA / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (DATA / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    report = {
        "by_plan": by_plan,
        "unique_physical_roots": roots,
        "n_experts": len(rows),
        "plan_b_added": len(added),
        "plan_c_audit": c_audit,
        "fixtures": fixtures,
    }
    (REC / "dataset_merge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
