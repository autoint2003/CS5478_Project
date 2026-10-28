#!/usr/bin/env python
"""Unified census of existing expert artifacts (P9 Phase 3). Does not modify sources."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.expert_pipeline.branches import BRANCH_TO_CLASS, CLASS_ALIASES  # noqa: E402

OUT = ROOT / "results/diagnostics/p9_expert_fsm_dataset"

SOURCES = [
    ("experts_v2_official", ROOT / "data/experts_v2_official"),
    ("experts_v2", ROOT / "data/experts_v2"),
    ("experts_v2_recovered", ROOT / "data/experts_v2_recovered"),
    ("experts_v2_recovery", ROOT / "data/experts_v2_recovery"),
    ("experts_v3_stage1_verified", ROOT / "data/experts_v3_stage1_verified"),
    ("unified_experts", ROOT / "data/unified_experts"),
]
# Live P2 precision-required pilot (short A1–D2 labels); not merged into official.
P2_PILOT_ACCEPTED = ROOT / "results/data_generation/precision_required_pilot_v2/ACCEPTED_TRAJECTORIES.csv"
P2_PILOT_SUMMARY = ROOT / "results/data_generation/precision_required_pilot_v2/PILOT_SUMMARY.json"


def jsafe(o: Any) -> Any:
    if o is None or isinstance(o, (bool, str, int, float)):
        return o
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def class_of(branch_id: str | None) -> str:
    if not branch_id:
        return "?"
    return BRANCH_TO_CLASS.get(str(branch_id), str(branch_id))


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def infer_skills(row: dict) -> dict:
    phases = row.get("phases") or row.get("precision_phases") or []
    if isinstance(phases, str):
        phases = phases.split("|")
    plogs = row.get("primitive_logs") or row.get("precision_logs") or {}
    if isinstance(plogs, str):
        plogs = {}
    executed = []
    for p in phases:
        ps = str(p)
        if "correct_dtheta" in ps or ps == "CORRECT_DTHETA":
            executed.append("dtheta")
        elif "correct_dx" in ps or "dx_support" in ps or ps == "CORRECT_DX":
            executed.append("dx")
        elif "correct_dz" in ps or ps.startswith("dz") or ps == "CORRECT_DZ":
            executed.append("dz")
        elif "RECONFIGURE" in ps:
            executed.append(ps)
        elif "PRECISION_SKIPPED" in ps or "direct_handoff" in ps.lower():
            executed.append("direct_handoff")
    # Deduplicate consecutive
    dedup = []
    for s in executed:
        if not dedup or dedup[-1] != s:
            dedup.append(s)
    cont = str(row.get("continuation") or "")
    if cont == "direct_handoff" and not dedup:
        dedup = ["direct_handoff"]
    return {
        "skills_executed": dedup,
        "skill_transition": "→".join(dedup) if dedup else "unknown",
        "fsm_planned": (plogs.get("fsm_planned_skills") if isinstance(plogs, dict) else None),
        "fsm_executed": (plogs.get("fsm_executed_skills") if isinstance(plogs, dict) else None),
    }


def row_to_census(corpus: str, row: dict, *, eligible_default: bool | None = None) -> dict:
    bid = row.get("branch_id")
    th = row.get("final_dtheta_deg") or row.get("dtheta_deg") or (row.get("errors_exit") or {}).get("dtheta_deg")
    s12 = row.get("SUCCESS_12S")
    snew = row.get("SUCCESS_NEW") or row.get("physical_success")
    if snew is None and row.get("admit_success") == "SUCCESS_NEW":
        snew = row.get("accepted")
    skills = infer_skills(row)
    strict = row.get("SUCCESS_STRICT_10DEG")
    relaxed = row.get("SUCCESS_RELAXED_15DEG")
    if strict is None:
        strict = bool(snew)
    if relaxed is None and th is not None and s12:
        try:
            relaxed = bool(s12) and float(th) <= 15.0 and bool(snew or float(th) <= 15.0)
            # Conservative: if SUCCESS_NEW true → relaxed true; if false but θ≤15 and 12S, mark candidate
            if not snew and float(th) <= 15.0 and bool(s12):
                relaxed = True
            elif snew:
                relaxed = True
        except Exception:
            relaxed = None
    lineage = row.get("source_ic_id") or row.get("sample_id") or row.get("ic_id")
    phys = bool(snew) if snew is not None else False
    # Recovered corpus is diagnostic-only
    train_ok = phys and corpus == "experts_v2_official"
    if corpus == "experts_v2_recovered":
        train_ok = False
    if eligible_default is False:
        train_ok = False
    return {
        "corpus": corpus,
        "class_label": class_of(bid),
        "branch_id": bid,
        "source_ic_id": lineage,
        "parent_lineage": row.get("parent_ic_id") or row.get("lineage") or lineage,
        "physics_config": row.get("config_id") or row.get("physics_id") or row.get("dataset_version"),
        "stage1_skill": row.get("stage1") or row.get("plan_provenance"),
        "stage1_qualified": row.get("stage1_eligible", row.get("stage1_ok", True if phys else None)),
        "continuation": row.get("continuation"),
        "skills_executed": "|".join(skills["skills_executed"]),
        "skill_transition": skills["skill_transition"],
        "fsm_planned": skills.get("fsm_planned"),
        "fsm_executed": skills.get("fsm_executed"),
        "physical_success": phys,
        "SUCCESS_12S": s12,
        "SUCCESS_NEW": snew,
        "SUCCESS_STRICT_10DEG": strict,
        "SUCCESS_RELAXED_15DEG": relaxed,
        "final_dtheta_deg": th,
        "complete_verified": bool(phys and s12 is not False),
        "duplicate_lineage": False,  # filled later
        "training_eligible": train_ok,
        "oracle_implementation": row.get("oracle_implementation"),
        "success_criteria_version": row.get("success_criteria_version"),
        "n_actions": row.get("n_actions") or (len(row.get("actions") or []) if isinstance(row.get("actions"), list) else None),
    }


def mark_duplicates(rows: list[dict]) -> None:
    seen = {}
    for r in rows:
        key = (r.get("class_label"), r.get("source_ic_id"), r.get("corpus"))
        if key in seen and r.get("source_ic_id"):
            r["duplicate_lineage"] = True
        else:
            seen[key] = True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    census: list[dict] = []
    corpus_notes = []

    for name, root in SOURCES:
        man = root / "manifest.jsonl"
        rows = read_manifest(man)
        note = {"corpus": name, "root": str(root), "manifest_rows": len(rows), "exists": root.is_dir()}
        if not rows and root.is_dir():
            # Count pkls per branch
            branch_counts = {}
            for d in root.iterdir() if root.is_dir() else []:
                if d.is_dir() and not d.name.startswith("."):
                    n = len(list(d.glob("*.pkl")))
                    if n:
                        branch_counts[d.name] = n
                        for pkl in d.glob("*.pkl"):
                            sid = pkl.stem.split("__", 1)[-1] if "__" in pkl.stem else pkl.stem
                            census.append(
                                row_to_census(
                                    name,
                                    {
                                        "branch_id": d.name if d.name in BRANCH_TO_CLASS or d.name in CLASS_ALIASES.values() else d.name,
                                        "source_ic_id": sid,
                                        "continuation": "unknown",
                                        "physical_success": name.startswith("experts_v"),
                                        "SUCCESS_NEW": name == "experts_v2_official",
                                        "SUCCESS_12S": None,
                                    },
                                    eligible_default=(name == "experts_v2_official"),
                                )
                            )
            note["pkl_branch_counts"] = branch_counts
        else:
            for row in rows:
                census.append(row_to_census(name, row))
        corpus_notes.append(note)

    # P2 pilot accepted rows (short class labels A2/B2/…)
    p2_note = {"corpus": "precision_required_pilot_v2", "accepted": 0, "path": str(P2_PILOT_ACCEPTED)}
    if P2_PILOT_ACCEPTED.is_file():
        import csv as _csv

        with P2_PILOT_ACCEPTED.open(encoding="utf-8", newline="") as f:
            for prow in _csv.DictReader(f):
                if str(prow.get("accepted")).lower() not in ("true", "1"):
                    continue
                p2_note["accepted"] = int(p2_note["accepted"]) + 1
                cls = str(prow.get("classification") or prow.get("branch_target") or "?")
                census.append(
                    row_to_census(
                        "precision_required_pilot_v2",
                        {
                            "branch_id": CLASS_ALIASES.get(cls, cls),
                            "source_ic_id": prow.get("raw_scenario_id") or prow.get("capture_fp"),
                            "continuation": "precision",
                            "physical_success": True,
                            "SUCCESS_NEW": prow.get("SUCCESS_NEW"),
                            "SUCCESS_12S": prow.get("SUCCESS_12S_final"),
                            "phases": str(prow.get("precision_phases") or "").split("|"),
                        },
                        eligible_default=False,
                    )
                )
                census[-1]["class_label"] = cls if cls in CLASS_ALIASES else class_of(census[-1]["branch_id"])
        corpus_notes.append(p2_note)

    mark_duplicates(census)

    # Coverage for official only
    official = [r for r in census if r["corpus"] == "experts_v2_official" and r["training_eligible"]]
    by_class = Counter(r["class_label"] for r in official)
    # Also count all official physical regardless of eligible flag
    official_all = [r for r in census if r["corpus"] == "experts_v2_official" and r["physical_success"]]
    by_class_all = Counter(r["class_label"] for r in official_all)

    keys: list[str] = []
    for r in census:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (OUT / "EXISTING_DATASET_CENSUS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in census:
            w.writerow({k: jsafe(r.get(k)) for k in keys})

    cov_rows = []
    for cls in ("A1", "A2", "B1", "B2", "C1", "C2", "D2"):
        n = by_class_all.get(cls, 0)
        cov_rows.append(
            {
                "class_label": cls,
                "branch_id": CLASS_ALIASES[cls],
                "n_verified_official": n,
                "milestone_20": n >= 20,
                "target_50": n >= 50,
                "deficit_to_20": max(0, 20 - n),
                "deficit_to_50": max(0, 50 - n),
                "priority": "HIGH" if cls in ("A2", "C2", "B2", "D2") and n < 20 else ("MED" if n < 20 else "OK"),
            }
        )
    with (OUT / "CLASS_COVERAGE.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov_rows[0].keys()))
        w.writeheader()
        for r in cov_rows:
            w.writerow(r)

    # Skill transitions from official
    trans = Counter()
    trans_success = Counter()
    for r in official_all:
        t = r.get("skill_transition") or "unknown"
        trans[t] += 1
        if r.get("physical_success"):
            trans_success[t] += 1
    sk_rows = [
        {
            "transition": k,
            "count": v,
            "success_count": trans_success.get(k, 0),
            "note": "requested_vs_executed_not_separated_in_legacy_manifests",
        }
        for k, v in trans.most_common()
    ]
    if sk_rows:
        with (OUT / "SKILL_TRANSITION_COVERAGE.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sk_rows[0].keys()))
            w.writeheader()
            for r in sk_rows:
                w.writerow(r)
    else:
        (OUT / "SKILL_TRANSITION_COVERAGE.csv").write_text("transition,count,success_count,note\n", encoding="utf-8")

    # θ impact estimate on official
    with_th = [r for r in official_all if r.get("final_dtheta_deg") is not None]
    n_strict = sum(1 for r in official_all if r.get("SUCCESS_STRICT_10DEG") or r.get("SUCCESS_NEW"))
    n_relaxed_est = sum(1 for r in official_all if r.get("SUCCESS_RELAXED_15DEG"))
    theta_summary = {
        "n_official_physical": len(official_all),
        "n_with_theta": len(with_th),
        "n_SUCCESS_STRICT_10DEG_est": n_strict,
        "n_SUCCESS_RELAXED_15DEG_est": n_relaxed_est,
        "by_class": dict(by_class_all),
        "corpus_notes": corpus_notes,
    }
    (OUT / "raw_census_summary.json").write_text(json.dumps(jsafe(theta_summary), indent=2), encoding="utf-8")
    print(json.dumps(theta_summary["by_class"], indent=2))
    print("wrote census", len(census), "rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
