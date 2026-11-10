"""Recompute P103 NEW_SUCCESS_ANALYSIS + gates from existing PAIRED_CLOSED_LOOP.csv.

Does not retrain or re-roll closed-loop. Applies fixed root-disjoint definition.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p103_retention_constrained_residual"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def phys_root(tid: str) -> str:
    t = re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", str(tid), flags=re.I)
    t = re.sub(r"__([0-9a-f]{8,})$", "", t)
    return t


def main() -> None:
    paired = list(csv.DictReader((OUT / "PAIRED_CLOSED_LOOP.csv").open(encoding="utf-8")))
    corr = list(csv.DictReader((P98 / "B1_C_CORRECTION_TARGETS.csv").open(encoding="utf-8")))
    b1_corr = {phys_root(r.get("phys_root") or r["traj_id"]) for r in corr if r["skill"] == "B1"}
    c_corr = {phys_root(r.get("phys_root") or r["traj_id"]) for r in corr if r["skill"] == "C"}

    new_rows = []
    for r in paired:
        if r["arm"] != "C_P103":
            continue
        if int(r["gain"]) != 1:
            continue
        pr = phys_root(r["phys_root"])
        if r["skill"] == "B1" and int(r["P64_SUCCESS"]) == 0:
            train_root = pr in b1_corr
            genuine = int(not train_root)
        elif r["skill"] == "C_Feasible" and int(r["P64_SUCCESS"]) == 0:
            train_root = pr in c_corr
            genuine = int(not train_root)
        else:
            train_root = pr in (b1_corr | c_corr)
            genuine = 0
        new_rows.append(
            {
                "train_seed": r["train_seed"],
                "skill": r["skill"],
                "traj_id": r["traj_id"],
                "phys_root": pr,
                "P64_SUCCESS": r["P64_SUCCESS"],
                "CAND_SUCCESS": r["CAND_SUCCESS"],
                "genuine_root_disjoint": genuine,
                "is_train_corr_root": int(train_root),
                "is_c_train_diag": int(r["skill"] == "C_train"),
                "note": "recomputed: genuine only if B1/C_Feasible P64-fail→succ AND not that skill's corr train root",
            }
        )

    # keep protected_loss diagnostics from prior file if present
    old = list(csv.DictReader((OUT / "NEW_SUCCESS_ANALYSIS.csv").open(encoding="utf-8"))) if (OUT / "NEW_SUCCESS_ANALYSIS.csv").is_file() else []
    for r in old:
        if r.get("diag") == "protected_loss":
            new_rows.append(r)

    keys: list[str] = []
    seen: set[str] = set()
    for r in new_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with (OUT / "NEW_SUCCESS_ANALYSIS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(new_rows)

    summary = json.loads((OUT / "SUMMARY.json").read_text(encoding="utf-8"))
    n_genuine = sum(int(r.get("genuine_root_disjoint") or 0) for r in new_rows)
    summary["gates"]["n_genuine_root_disjoint_gains"] = n_genuine
    summary["gates"]["genuine_gains"] = [r for r in new_rows if int(r.get("genuine_root_disjoint") or 0) == 1]
    summary["questions"]["3_new_root_disjoint_successes"] = n_genuine > 0
    summary["recomputed"] = {
        "created": utc(),
        "fix": "genuine_root_disjoint now excludes skill-matched correction train roots",
        "n_genuine": n_genuine,
        "note": "Closed-loop numbers unchanged; only gain labeling/gates recomputed.",
    }
    # outcome class unchanged if still no genuine and retention_zero false
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    note = (
        f"\n\n## Recompute note ({utc()})\n\n"
        f"Fixed `genuine_root_disjoint` to require phys_root ∉ skill-matched correction train set. "
        f"n_genuine={n_genuine}. Closed-loop rollouts not re-run; outcome class remains "
        f"`{summary.get('outcome_class')}`.\n"
    )
    report = (OUT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    if "## Recompute note" in report:
        report = report.split("## Recompute note")[0].rstrip() + note
    else:
        report = report.rstrip() + note
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(f"[P103 recompute] n_genuine={n_genuine} outcome={summary.get('outcome_class')}", flush=True)


if __name__ == "__main__":
    main()
