"""Reclassify ROOT_PROVENANCE_SPLITS with P98 / new_ics / panel exposure tags."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p104_joint_token_residual_pilot.run_p104 import (  # noqa: E402
    P97_NEW,
    phys_root,
)

OUT = ROOT / "results/experiments/p107_action_aware_multiseed_validation"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation/SPLITS.json"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    splits = json.loads(P98.read_text(encoding="utf-8"))
    train_c = {phys_root(x) for x in splits["c_train_phys"]}
    exposed_c = {phys_root(x) for x in splits["c_dev_phys"]}
    new_ics = {phys_root(p.stem) for p in P97_NEW.glob("*.pkl")} if P97_NEW.is_dir() else set()
    man = json.loads((OUT / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
    corr_pr = {phys_root(t) for t in man["panels"].get("C_train", [])}

    rows = list(csv.DictReader((OUT / "ROOT_PROVENANCE_SPLITS.csv").open(encoding="utf-8")))
    fixed = []
    for r in rows:
        pr = r["phys_root_truncated"]
        exp = r["exposure_recorded"]
        tags = []
        if pr in train_c:
            tags.append("p98_c_train")
        if pr in exposed_c:
            tags.append("p98_c_dev")
        if pr in corr_pr:
            tags.append("correction_demo_C")
        if pr in new_ics:
            tags.append("p97_new_ics")
        if exp == "newly_selected":
            tags.append("panel_newly_selected")
        if exp == "previously_exposed":
            tags.append("panel_previously_exposed")
        if exp == "train":
            tags.append("panel_labeled_train")
        if r["result_validity"] != "VALID_EXACT_IC":
            tags.append("INVALID_IC")

        if r["result_validity"] != "VALID_EXACT_IC":
            primary = f"{exp}|INVALID_IC"
        elif pr in train_c or pr in corr_pr:
            primary = "training_roots"
        elif pr in exposed_c or exp == "previously_exposed":
            primary = "previously_exposed_dev"
        elif exp == "newly_selected":
            primary = "newly_selected_dev"
        elif pr in new_ics:
            primary = "p97_new_ics_non_p98_train"
        else:
            primary = "other"

        r2 = dict(r)
        r2["provenance_category"] = primary
        r2["provenance_tags"] = "|".join(tags)
        r2["in_p98_c_train"] = int(pr in train_c)
        r2["in_p98_c_dev"] = int(pr in exposed_c)
        r2["in_p97_new_ics"] = int(pr in new_ics)
        r2["in_correction_demo_C"] = int(pr in corr_pr)
        fixed.append(r2)

    write_csv(OUT / "ROOT_PROVENANCE_SPLITS.csv", fixed)

    summary = []
    for panel in ("C_Feasible", "C_Unfiltered"):
        sub = [r for r in fixed if r["panel"] == panel]
        cats: dict[str, list] = defaultdict(list)
        for r in sub:
            cats[r["provenance_category"]].append(r)
        for cat, lst in sorted(cats.items()):
            rec = {
                "panel": panel,
                "provenance_category": cat,
                "n_roots": len(lst),
                "canonical_ids": "|".join(sorted(r["canonical_physical_id"] for r in lst)),
            }
            for arm in ("B", "C", "D"):
                for seed in (1, 2, 3):
                    key = f"succ_{arm}_s{seed}"
                    rec[key] = sum(int(r[key]) for r in lst if r.get(key) != "")
                rec[f"succ_sum_{arm}"] = sum(int(r[f"succ_sum_{arm}"]) for r in lst)
            summary.append(rec)
    write_csv(OUT / "ROOT_PROVENANCE_CATEGORY_SUMMARY.csv", summary)
    for r in summary:
        print(
            r["panel"],
            r["provenance_category"],
            "n=",
            r["n_roots"],
            "succB/C/D",
            r["succ_sum_B"],
            r["succ_sum_C"],
            r["succ_sum_D"],
        )
        print(" ", r["canonical_ids"])


if __name__ == "__main__":
    main()
