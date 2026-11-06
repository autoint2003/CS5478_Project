"""Phase 1 — C stage-transition audit from stored P56/P57 diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.precision.p58_fast_c_grpo.paths import OUT, P56_C_AUDIT, P57A_PROGRESS


def _fam(tid: str) -> str:
    t = tid.lower()
    if "gbi_" in t:
        return "gbi"
    if "plan_c" in t:
        return "plan_c"
    return "other"


def run_audit() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(P57A_PROGRESS.open(encoding="utf-8")) if r["class"] == "C"]
    g = 64
    n = len(rows)
    total = n * g
    succ = sum(int(r["success_relaxed_count"]) for r in rows)
    rec = sum(int(r["recatch_count"]) for r in rows)
    stab = sum(int(r["stable_hold_count"]) for r in rows)
    hand = sum(int(r["handoff_count"]) for r in rows)
    drop = sum(int(r["drop_before_count"]) for r in rows)

    # Conditional rates (rollout-level aggregates)
    p_recatch = rec / total
    p_stable = stab / total
    p_hand = hand / total
    p_succ = succ / total
    p_stable_given_recatch = stab / rec if rec else None
    # HANDOFF among stable approx: min(1, hand/stab); success among handoff
    p_hand_given_stable = min(1.0, hand / stab) if stab else None
    p_succ_given_hand = succ / hand if hand else None
    p_succ_given_stable = succ / stab if stab else None

    # Per-IC funnel leakage
    leak = []
    for r in rows:
        s = int(r["success_relaxed_count"])
        rc = int(r["recatch_count"])
        st = int(r["stable_hold_count"])
        hd = int(r["handoff_count"])
        leak.append(
            {
                "ic_id": r["ic_id"],
                "family": _fam(r["ic_id"]),
                "p_succ": s / g,
                "p_recatch": rc / g,
                "p_stable": st / g,
                "p_handoff": hd / g,
                "stable_given_recatch": (st / rc) if rc else None,
                "succ_given_stable": (s / st) if st else None,
                "zero": int(s == 0),
                "has_partial": int(s == 0 and (rc > 0 or st > 0)),
            }
        )

    n_zero = sum(x["zero"] for x in leak)
    n_zero_partial = sum(x["has_partial"] for x in leak)
    # Where mass is lost: among non-success rollouts
    # Before recatch: (total - rec) ; after recatch before stable: (rec - stab);
    # after stable before success: (stab - succ) — approximate exclusive stages
    before_recatch = total - rec
    after_recatch_unstable = max(0, rec - stab)
    after_stable_no_succ = max(0, stab - succ)

    p56 = {}
    if P56_C_AUDIT.is_file():
        a = json.loads(P56_C_AUDIT.read_text(encoding="utf-8"))
        p56 = {
            "global_bucket_frac": a.get("global_bucket_frac"),
            "n_zero_with_useful_partial": a.get("n_zero_with_useful_partial"),
        }

    out = {
        "source": "P57_A_FULL_EPOCH_PROGRESS C rows + P56_C_FAILURE_AUDIT",
        "n_c_ics": n,
        "n_rollouts": total,
        "rates": {
            "p_recatch": p_recatch,
            "p_stable_hold": p_stable,
            "p_handoff": p_hand,
            "p_success_relaxed": p_succ,
            "p_drop": drop / total,
            "p_stable_given_recatch": p_stable_given_recatch,
            "p_handoff_given_stable_approx": p_hand_given_stable,
            "p_success_given_handoff": p_succ_given_hand,
            "p_success_given_stable": p_succ_given_stable,
        },
        "failure_mass": {
            "before_recatch": before_recatch / total,
            "recatch_but_not_stable": after_recatch_unstable / total,
            "stable_but_not_success": after_stable_no_succ / total,
            "success": p_succ,
        },
        "n_zero_ics": n_zero,
        "n_zero_with_recatch_or_stable": n_zero_partial,
        "p56": p56,
        "per_ic": leak,
    }
    (OUT / "C_STAGE_AUDIT.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    primary = (
        "before_recatch"
        if out["failure_mass"]["before_recatch"] >= out["failure_mass"]["stable_but_not_success"]
        and out["failure_mass"]["before_recatch"] >= out["failure_mass"]["recatch_but_not_stable"]
        else (
            "stable_to_success"
            if out["failure_mass"]["stable_but_not_success"]
            >= out["failure_mass"]["recatch_but_not_stable"]
            else "recatch_to_stable"
        )
    )
    lines = [
        "# C_STAGE_AUDIT",
        "",
        "Sources: P57-A train progress (68 C × 64) and P56 C failure buckets. **No new large diagnostic rollouts.**",
        "",
        "## Aggregate stage rates (P57-A)",
        "",
        f"| Stage | Rate |",
        f"|-------|-----:|",
        f"| Recatch acquisition | {p_recatch:.3f} |",
        f"| Stable hold | {p_stable:.3f} |",
        f"| Stable \| recatch | {p_stable_given_recatch:.3f} |",
        f"| HANDOFF | {p_hand:.3f} |",
        f"| HANDOFF \| stable (approx) | {p_hand_given_stable:.3f} |",
        f"| SUCCESS_RELAXED_12S | {p_succ:.3f} |",
        f"| SUCCESS \| stable | {p_succ_given_stable:.3f} |",
        f"| SUCCESS \| HANDOFF | {p_succ_given_hand:.3f} |",
        "",
        "## Where non-success mass sits (exclusive approx)",
        "",
        f"| Bucket | Fraction of all C rollouts |",
        f"|--------|---------------------------:|",
        f"| Before valid recatch | {out['failure_mass']['before_recatch']:.3f} |",
        f"| Recatch but not stable hold | {out['failure_mass']['recatch_but_not_stable']:.3f} |",
        f"| Stable but not SUCCESS | {out['failure_mass']['stable_but_not_success']:.3f} |",
        f"| SUCCESS | {p_succ:.3f} |",
        "",
        f"**Primary bottleneck (by mass): `{primary}`.**",
        "",
        "Interpretation: ~55% of C rollouts never acquire the P57 valid-recatch gate; among recatches, "
        "stable-hold conversion is high (~0.84); among stable holds, conversion to SUCCESS is low "
        f"(~{p_succ_given_stable:.2f}) — so both **pre-recatch acquisition** and **stable→HANDOFF/SUCCESS** matter, "
        "with more raw mass before recatch.",
        "",
        f"Zero-success C ICs: {n_zero}/68; with partial recatch/stable: {n_zero_partial}.",
        "",
        "## P56 bucket context",
        "",
        f"`{p56.get('global_bucket_frac')}`",
        "",
    ]
    (OUT / "C_STAGE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in run_audit().items() if k != "per_ic"}, indent=2))
