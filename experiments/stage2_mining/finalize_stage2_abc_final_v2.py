#!/usr/bin/env python
"""Build versioned manifest for stage2_abc_final_v2 (A/B from prior + C post-lift)."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/data_generation/stage2_abc_final_v2"
FINAL_V1 = ROOT / "results/data_generation/stage2_abc_final"
V1 = ROOT / "results/data_generation/stage2_abc_mining/v1"
V2 = ROOT / "results/data_generation/stage2_abc_mining/v2"


def classify(o: dict) -> str:
    # Prefer already-finalized labels from stage2_abc_final
    if o.get("final_category") in {
        "DIRECT_HANDOFF",
        "NONTRIVIAL_K49_SUCCESS",
        "CONTINUOUS_ONLY_SUCCESS",
        "ORACLE_FAILURE",
        "INTERMEDIATE_CURRICULUM",
    }:
        return o["final_category"]
    cat = o.get("sft_category") or o.get("sft_category_raw")
    k49 = o.get("k49_demo") or {}
    cont = o.get("continuous_demo") or {}
    if cat == "DIRECT_HANDOFF":
        return "DIRECT_HANDOFF"
    if cat == "INTERMEDIATE_CURRICULUM":
        return "INTERMEDIATE_CURRICULUM"
    if cat == "NONTRIVIAL_PRECISION":
        if k49.get("nontrivial_representability") or k49.get("verified"):
            return "NONTRIVIAL_K49_SUCCESS"
        if cont.get("verified") or cont.get("replay_SUCCESS_NEW"):
            return "CONTINUOUS_ONLY_SUCCESS"
        return "CONTINUOUS_ONLY_SUCCESS"
    if cat in ("C_POST_LIFT_UNSUITABLE", "FAILED_ORACLE", "GATE_FAIL"):
        return "ORACLE_FAILURE"
    return "ORACLE_FAILURE"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.jsonl").unlink(missing_ok=True)
    rows = []

    # Import A/B (+ HARD_IC) from final v1 / mining — do not mix into C lift tree overwrites
    sources = []
    for base in (FINAL_V1, V2, V1):
        od = base / "outcomes"
        if od.exists():
            sources.append(base)

    seen = set()
    for base in sources:
        for p in sorted((base / "outcomes").glob("*.json")):
            o = json.loads(p.read_text(encoding="utf-8"))
            plan = o.get("plan_provenance")
            sid = o.get("sample_id")
            if not sid or sid in seen:
                continue
            # Prefer post-lift C from OUT over old C
            if plan == "C" and not sid.startswith("C__postlift"):
                continue
            if plan not in ("A", "B", "C", "HARD_IC"):
                continue
            seen.add(sid)
            final_cat = classify(o)
            # copy ic if missing in OUT
            for folder in ("ics", "continuous", "tokens"):
                src_dir = base / folder
                if not src_dir.exists():
                    continue
                for f in src_dir.glob(f"{sid}*"):
                    dst = OUT / folder / f.name
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dst)
            row = {
                "sample_id": sid,
                "plan_provenance": plan,
                "final_category": final_cat,
                "sft_category_raw": o.get("sft_category"),
                "source_kind": o.get("source_kind"),
                "source_pool": "final_v2_postlift" if sid.startswith("C__postlift") else str(base.name),
                "genuine_stage1_plan_b": o.get("genuine_stage1_plan_b"),
                "stage2_starts_at_post_lift_ic": o.get("stage2_starts_at_post_lift_ic"),
                "catch_sample_id": o.get("catch_sample_id"),
                "continuous_demo": o.get("continuous_demo"),
                "k49_demo": o.get("k49_demo"),
                "direct": o.get("direct"),
                "precision": o.get("precision"),
            }
            rows.append(row)
            with (OUT / "manifest.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    # Add post-lift C from OUT outcomes (already written)
    for p in sorted((OUT / "outcomes").glob("C__postlift__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        sid = o["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        row = {
            "sample_id": sid,
            "plan_provenance": "C",
            "final_category": classify(o),
            "sft_category_raw": o.get("sft_category"),
            "source_kind": o.get("source_kind"),
            "source_pool": "final_v2_postlift",
            "stage2_starts_at_post_lift_ic": True,
            "catch_sample_id": o.get("catch_sample_id"),
            "lift_npz": o.get("lift_npz"),
            "continuous_demo": o.get("continuous_demo"),
            "k49_demo": o.get("k49_demo"),
            "direct": o.get("direct"),
            "precision": o.get("precision"),
        }
        rows.append(row)
        with (OUT / "manifest.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    by_plan = defaultdict(Counter)
    for r in rows:
        by_plan[r["plan_provenance"]][r["final_category"]] += 1

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "by_plan_category": {p: dict(c) for p, c in by_plan.items()},
        "plan_c_postlift": {
            "direct": by_plan["C"].get("DIRECT_HANDOFF", 0),
            "nontrivial_k49": by_plan["C"].get("NONTRIVIAL_K49_SUCCESS", 0),
            "continuous_only": by_plan["C"].get("CONTINUOUS_ONLY_SUCCESS", 0),
            "failures": by_plan["C"].get("ORACLE_FAILURE", 0),
        },
        "note": "Lift tapes are Stage-1; Stage-2 SFT begins at post-lift IC. HARD_IC kept separate.",
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
