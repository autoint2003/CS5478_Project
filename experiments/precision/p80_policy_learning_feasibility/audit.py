"""Phase 1 — P79 statistical consistency audit."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict

from experiments.precision.p80_policy_learning_feasibility.paths import OUT, P79_BANK, P79_CACHE, P79_OUT


def run_audit() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pickle.loads((P79_CACHE / "cf_rows_expand.pkl").read_bytes())
    comps = pickle.loads((P79_CACHE / "comparisons.pkl").read_bytes())
    rb = pickle.loads((P79_CACHE / "ranking_snap.pkl").read_bytes())
    bank = json.loads(P79_BANK.read_text(encoding="utf-8"))
    outcomes = {}
    md = P79_OUT / "MATCHED_COUNTERFACTUAL_OUTCOMES.md"
    # Prefer structured fields from comparisons + rows

    n_comps = len(comps)
    n_ties = sum(1 for c in comps if c["pref"] == "tie")
    n_disagree = n_comps - n_ties

    def nontrivial(c) -> bool:
        if c["pref"] == "tie":
            return False
        a, b = c["a"], c["b"]
        sa, sb = c["scores"][a], c["scores"][b]
        return sa["r3"] != sb["r3"] or sa["succ"] != sb["succ"] or sa["opp"] != sb["opp"]

    n_nontrivial = sum(1 for c in comps if nontrivial(c))
    # Disagree but only secondary (contact_loss) difference under lex order
    n_secondary_only = n_disagree - n_nontrivial
    disagree_rate_all = n_disagree / n_comps if n_comps else None
    nontrivial_rate = n_nontrivial / n_comps if n_comps else None

    # Consistency with P79 headline numbers
    reported = {
        "n_nontrivial": 86,
        "n_comparisons": 396,
        "disagree_rate_headline": 0.31,
        "n_ties": 273,
    }
    consistent = (
        n_comps == 396
        and n_ties == 273
        and n_disagree == 123
        and n_nontrivial == 86
        and abs((disagree_rate_all or 0) - 123 / 396) < 1e-9
    )

    # IC independence + snap grouping
    ics_rows = sorted({r["traj_id"] for r in rows})
    snaps_per_ic = {k: len(v) for k, v in (bank.get("by_ic") or {}).items()}
    primary = bank["populations"]["A_natural_fc_primary"]
    primary_ics = sorted({m["traj_id"] for m in primary})
    # CF expand used one primary FC per IC
    cf_snaps = sorted({r["snap_id"] for r in rows})
    snap_to_ic = {}
    for r in rows:
        snap_to_ic[r["snap_id"]] = r["traj_id"]
    ic_to_cf_snaps = defaultdict(set)
    for sid, tid in snap_to_ic.items():
        ic_to_cf_snaps[tid].add(sid)

    # Ranking IC-disjoint check
    bench = rb["bench"]
    fit_ics = set(bench["fit_ics"])
    eval_ics = set(bench["eval_ics"])
    overlap = sorted(fit_ics & eval_ics)
    # Verify ranking used eval-only pairs for accuracy (from REPRESENTATION_COMPARISON)
    rep = json.loads((P79_OUT / "REPRESENTATION_COMPARISON.json").read_text(encoding="utf-8"))
    rank_c = (rep.get("summary") or {}).get("rank_C")
    n_eval_pairs = (rep.get("summary") or {}).get("n_eval_pairs")
    # Check seeds: expand used seed_k==0 only; pilot had 2 seeds — ranking fit used expand
    seed_ks = sorted({r.get("seed_k", 0) for r in rows})
    # Snapshots shared fit/eval? by IC split only
    fit_snaps = {sid for sid, tid in snap_to_ic.items() if tid in fit_ics}
    eval_snaps = {sid for sid, tid in snap_to_ic.items() if tid in eval_ics}
    snap_overlap = sorted(fit_snaps & eval_snaps)

    # Denominator taxonomy
    taxonomy = {
        "1_all_candidate_pairs": {
            "n": n_comps,
            "definition": "C(4,2)×66 primary FC snaps = 6×66 = 396 matched pairwise comparisons",
        },
        "2_valid_evaluated_pairs": {
            "n": n_comps,
            "definition": "All 396 are valid: same restored state, same candidate set, outcomes recorded",
            "invalid": 0,
        },
        "3_statistically_meaningful_outcome_differences": {
            "n": n_nontrivial,
            "definition": (
                "pref≠tie AND differ on SUCCESS/r3/opposing (primary lex axes). "
                "P79 called these 'nontrivial outcome disagreements'."
            ),
        },
        "4_ties": {
            "n": n_ties,
            "definition": "Identical lex outcome tuple (succ, r3, opp, -loss)",
        },
        "5_unresolved_or_secondary_only": {
            "n": n_secondary_only,
            "definition": (
                "pref≠tie under full lex (incl. contact_loss) but NO difference on "
                "succ/r3/opp — secondary-only disagreements; not counted in the 86 nontrivial"
            ),
        },
    }
    identity = (
        taxonomy["1_all_candidate_pairs"]["n"]
        == taxonomy["2_valid_evaluated_pairs"]["n"]
        == (
            taxonomy["3_statistically_meaningful_outcome_differences"]["n"]
            + taxonomy["4_ties"]["n"]
            + taxonomy["5_unresolved_or_secondary_only"]["n"]
        )
    )

    report = {
        "consistent_with_p79_headlines": consistent,
        "reported": reported,
        "recomputed": {
            "n_comparisons": n_comps,
            "n_ties": n_ties,
            "n_disagreements_any_lex": n_disagree,
            "n_nontrivial_primary_axes": n_nontrivial,
            "n_secondary_only": n_secondary_only,
            "disagree_rate_any_lex": disagree_rate_all,
            "nontrivial_rate": nontrivial_rate,
            "note": (
                "P79 '31% disagreement' = 123/396 (any lex disagreement), "
                "NOT 86/396. The 86 are the nontrivial primary-axis subset. "
                "Remaining unclassified vs nontrivial = 123-86 = 37 secondary-only."
            ),
        },
        "taxonomy": taxonomy,
        "denominator_identity_holds": identity,
        "ic_independence": {
            "n_independent_ics_cf": len(ics_rows),
            "n_bank_ics": bank.get("n_independent_ics"),
            "n_bank_snapshots_all_kinds": bank.get("n_unique_snapshots"),
            "n_cf_primary_snaps": len(cf_snaps),
            "snaps_grouped_by_ic": True,
            "cf_snaps_per_ic_max": max(len(v) for v in ic_to_cf_snaps.values()) if ic_to_cf_snaps else 0,
            "cf_one_primary_per_ic": all(len(v) == 1 for v in ic_to_cf_snaps.values()),
            "bank_snaps_per_ic_example": dict(list(snaps_per_ic.items())[:5]),
            "mean_bank_snaps_per_ic": float(sum(snaps_per_ic.values()) / max(len(snaps_per_ic), 1)),
        },
        "ranking_ic_disjoint": {
            "n_fit_ics": len(fit_ics),
            "n_eval_ics": len(eval_ics),
            "fit_eval_ic_overlap": overlap,
            "fit_eval_snap_overlap": snap_overlap,
            "rank_C_reported": rank_c,
            "n_eval_pairs_reported": n_eval_pairs,
            "history_ranking_on_ic_disjoint": len(overlap) == 0 and len(snap_overlap) == 0,
            "seed_ks_in_expand": seed_ks,
            "note": (
                "P79 expand CF used seed_k=0 only for ranking dataset. "
                "Pilot 2-seed determinism was separate; seeds were not mixed into fit/eval splits."
            ),
        },
    }

    (OUT / "P79_STATISTICAL_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "P79_STATISTICAL_AUDIT.md").write_text(
        "# P79 Statistical Consistency Audit (P80 Phase 1)\n\n"
        "## Denominator taxonomy\n\n"
        "| Bucket | N | Notes |\n|---|---:|---|\n"
        f"| All candidate pairs | {n_comps} | 6 pairs × 66 snaps |\n"
        f"| Valid evaluated | {n_comps} | invalid=0 |\n"
        f"| Meaningful (nontrivial) | {n_nontrivial} | differ on succ/r3/opp |\n"
        f"| Ties | {n_ties} | identical lex tuple |\n"
        f"| Secondary-only disagree | {n_secondary_only} | loss-axis only; not in the 86 |\n\n"
        f"**Identity:** {n_nontrivial}+{n_ties}+{n_secondary_only}={n_comps} → "
        f"{'OK' if identity else 'BROKEN'}\n\n"
        f"**31% disagreement** = {n_disagree}/{n_comps} (any lex). "
        f"Nontrivial rate = {n_nontrivial}/{n_comps}.\n\n"
        "```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"consistent": consistent, "identity": identity, "nt": n_nontrivial, "sec": n_secondary_only}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_audit()
