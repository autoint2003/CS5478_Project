"""Inventory unused C roots for P108 panel construction."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p104_joint_token_residual_pilot.run_p104 import (  # noqa: E402
    eval_banned_phys,
    phys_root,
)


def gbi_key(s: str) -> str | None:
    m = re.search(r"gbi_\d+", str(s), re.I)
    return m.group(0).lower() if m else None


def main() -> None:
    splits = json.loads(
        (ROOT / "results/experiments/p98_unified_b1_c_consolidation/SPLITS.json").read_text(
            encoding="utf-8"
        )
    )
    train_g = {gbi_key(x) for x in splits["c_train_phys"]} - {None}
    dev_g = {gbi_key(x) for x in splits.get("c_dev_phys", [])} - {None}
    ind_g = {gbi_key(t) for t in (splits.get("ind") or {}).get("C") or []} - {None}
    ban = set(eval_banned_phys())
    ban_g = {gbi_key(t) for t in ban} - {None}

    p107 = json.loads(
        (
            ROOT / "results/experiments/p107_action_aware_multiseed_validation/TRAIN_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    p107_g = set()
    for skill in ("C_Feasible", "C_Unfiltered", "C_train"):
        for t in p107["panels"].get(skill, []):
            g = gbi_key(t)
            if g:
                p107_g.add(g)

    v2_dir = ROOT / "results/data_generation/stage2_abc_final_v2/ics"
    v1_dir = ROOT / "results/data_generation/stage2_abc_final/ics"
    rows = []
    for p in sorted(v2_dir.glob("C__*.pkl")):
        g = gbi_key(p.stem)
        if not g:
            continue
        rows.append(
            {
                "gbi": g,
                "path": str(p),
                "stem": p.stem,
                "in_train": g in train_g,
                "in_dev": g in dev_g,
                "in_ind": g in ind_g,
                "in_ban": g in ban_g,
                "in_p107_c": g in p107_g,
            }
        )

    # unique by gbi prefer postlift
    by_g: dict[str, list] = {}
    for r in rows:
        by_g.setdefault(r["gbi"], []).append(r)

    eligible = []
    for g, lst in sorted(by_g.items()):
        # eligibility: not train, not sealed ind
        if any(x["in_train"] for x in lst) or any(x["in_ind"] for x in lst):
            continue
        # prefer postlift file
        pick = next((x for x in lst if "postlift" in x["stem"]), lst[0])
        eligible.append(pick)

    print("v2 distinct gbi", len(by_g))
    print("eligible (not train, not ind)", len(eligible))
    clean = [e for e in eligible if not e["in_ban"]]
    never_p107 = [e for e in clean if not e["in_p107_c"]]
    print("eligible + not freeze-ban", len(clean))
    print("never in P107 C panels", len(never_p107))
    for e in eligible:
        flags = []
        if e["in_ban"]:
            flags.append("BAN")
        if e["in_dev"]:
            flags.append("DEV")
        if e["in_p107_c"]:
            flags.append("P107")
        print(e["gbi"], e["stem"][:60], ",".join(flags) or "NEW")

    # also count v1-only
    v1_g = {gbi_key(p.stem) for p in v1_dir.glob("C__*.pkl")} - {None}
    print("v1 gbi", len(v1_g), "v2-only", len(set(by_g) - v1_g), "v1-only", len(v1_g - set(by_g)))


if __name__ == "__main__":
    main()
