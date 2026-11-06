"""Phase 5 — coverage and generalization audit."""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict

from experiments.precision.p80_policy_learning_feasibility.paths import OUT, P79_BANK, P79_CACHE, PRIMARY_CANDIDATES


def run_coverage(selection: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    bank = json.loads(P79_BANK.read_text(encoding="utf-8"))
    rows = pickle.loads((P79_CACHE / "cf_rows_expand.pkl").read_bytes())
    snap = selection["snap"]
    reversals = selection["reversals"]
    eval_ics = selection["eval_ics"]

    strata = Counter(cands[PRIMARY_CANDIDATES[0]].get("stratum") or "unknown" for cands in snap.values())
    sw = Counter(int(cands[PRIMARY_CANDIDATES[0]].get("sw0") or 0) for cands in snap.values())

    # Bank kinds coverage
    kinds = Counter()
    for m in bank["populations"].get("A_natural_all") or []:
        kinds[m.get("kind")] += 1

    # Outcomes coverage
    r3_pos_ics = set()
    r3_neg_ics = set()
    for r in rows:
        if r.get("seed_k", 0) != 0:
            continue
        if r["r3_ready"]:
            r3_pos_ics.add(r["traj_id"])
        else:
            r3_neg_ics.add(r["traj_id"])

    # Skill families of forced tokens
    families = {
        "baseline": "actor_sample",
        "tok_56_soft": "soft_approach_family",
        "tok_50_approach": "approach_family",
        "tok_82_grip": "grip_family",
    }

    # Disagreement-enriched snaps (diagnostic)
    comps = pickle.loads((P79_CACHE / "comparisons.pkl").read_bytes())
    disagree_snaps = {c["snap_id"] for c in comps if c["pref"] != "tie"}

    correct_sw_ics = {cands[PRIMARY_CANDIDATES[0]]["traj_id"] for cands in snap.values() if cands[PRIMARY_CANDIDATES[0]].get("sw0") == 1}

    gaps = []
    if len(correct_sw_ics) < 5:
        gaps.append(
            {
                "gap": "correct_sidewall_fc",
                "current": len(correct_sw_ics),
                "needed": "≥15–20 independent correct-sidewall FC ICs with matched CF",
                "why": "Cannot infer correct-SW policy value from incorrect-SW states alone",
            }
        )
    if reversals["n_reliable_reversals_vs_56"] < 10:
        gaps.append(
            {
                "gap": "reliable_preference_reversals",
                "current": reversals["n_reliable_reversals_vs_56"],
                "needed": "≥20 reliable reversals across ≥15 ICs",
                "why": "Conditional learning requires state-dependent preference diversity",
            }
        )
    if len(eval_ics) < 30:
        gaps.append(
            {
                "gap": "heldout_eval_ics",
                "current": len(eval_ics),
                "needed": "≥30 eval ICs (P79 planning target)",
                "why": "Wide IC-bootstrap CIs; generalization claims fragile",
            }
        )

    # Smallest targeted collection
    targeted = {
        "priority_1": "Mine additional correct-sidewall first-contact snapshots (privileged label for stratification only) until ≥15–20 independent ICs; run same 4-candidate matched CF.",
        "priority_2": "Enrich FC states where tok_56 is NOT uniquely best on fit diagnostics (geometry/obs filters only — not final-test outcomes).",
        "priority_3": "Keep natural-distribution primary population separate; report enriched separately.",
        "estimate": "Approximately 20–40 additional independent IC rollouts with CF (4 actions × 1–2 seeds) if correct-SW contacts appear; otherwise expand IC pool beyond current 68 C-skill set.",
        "do_not": "Do not start Actor training to collect these; diagnostic CF only.",
    }

    natural_eval = [p for p in selection["per_snap"]]  # all eval are natural primary FC
    enriched_eval = [p for p in selection["per_snap"] if p["snap_id"] in disagree_snaps]

    report = {
        "n_independent_ics": len({cands[PRIMARY_CANDIDATES[0]]["traj_id"] for cands in snap.values()}),
        "n_cf_snaps": len(snap),
        "n_bank_snaps_all_kinds": bank.get("n_unique_snapshots"),
        "strata_fc_primary": dict(strata),
        "sidewall": {"correct_snaps": sw.get(1, 0), "incorrect_snaps": sw.get(0, 0), "correct_ics": len(correct_sw_ics)},
        "bank_kinds": dict(kinds),
        "r3_coverage_ics": {"success_ics": len(r3_pos_ics), "failure_ics": len(r3_neg_ics - r3_pos_ics), "any_success_ics": len(r3_pos_ics)},
        "m111_families_in_candidate_set": families,
        "preference_reversals": {
            "n_reliable": reversals["n_reliable_reversals_vs_56"],
            "n_ics": len(reversals.get("reversal_ics") or []),
        },
        "populations": {
            "natural_eval_snaps": len(natural_eval),
            "disagreement_enriched_eval_snaps": len(enriched_eval),
            "note": "Enriched performance is diagnostic-only, not natural-policy performance.",
        },
        "sufficient_for_actor_learning_pilot": len(gaps) == 0,
        "gaps": gaps,
        "smallest_targeted_collection": targeted,
        "warning": (
            "Only one correct-sidewall IC in CF primary set. "
            "Do not infer correct-sidewall action value from incorrect-sidewall results."
        ),
    }
    (OUT / "COVERAGE_AND_GENERALIZATION.md").write_text(
        "# Coverage and Generalization (P80 Phase 5)\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    from experiments.precision.p80_policy_learning_feasibility.selection import run_selection_analyses

    run_coverage(run_selection_analyses())
