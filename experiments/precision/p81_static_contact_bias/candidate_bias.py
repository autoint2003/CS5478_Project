"""Phase 2 — bias vs candidate-set artifact using P79/P80 matched CF."""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict

import numpy as np

from experiments.precision.p81_static_contact_bias.paths import OUT, P79_CACHE, PRIMARY_CANDIDATES


def _tuple(r) -> tuple:
    return (float(r["SUCCESS_RELAXED"]), float(r["r3_ready"]), float(r["opposing_bilateral"]), -float(r["contact_loss"]))


def run_candidate_bias() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pickle.loads((P79_CACHE / "cf_rows_expand.pkl").read_bytes())
    pilot = pickle.loads((P79_CACHE / "cf_rows_pilot.pkl").read_bytes())
    rb = pickle.loads((P79_CACHE / "ranking_snap.pkl").read_bytes())
    eval_ics = set(rb["bench"]["eval_ics"])
    fit_ics = set(rb["bench"]["fit_ics"])

    snap = defaultdict(dict)
    for r in rows:
        if r.get("seed_k", 0) != 0:
            continue
        snap[r["snap_id"]][r["candidate"]] = r

    cats = Counter()
    impacts = []  # Δsucc 56 - actor when 56 unique best
    per_ic_delta = defaultdict(list)
    details = []

    for sid, c in snap.items():
        vals = {k: _tuple(v) for k, v in c.items()}
        uniq = set(vals.values())
        a, b56 = vals["baseline"], vals["tok_56_soft"]
        tid = c["baseline"]["traj_id"]
        best = max(vals.keys(), key=lambda k: vals[k])
        n_best = sum(1 for v in vals.values() if v == vals[best])

        if len(uniq) == 1:
            cat = "all_tie"
        elif vals[best] == b56 and n_best == 1 and b56 > a:
            cat = "tok56_uniquely_better"
            impacts.append(float(c["tok_56_soft"]["SUCCESS_RELAXED"] - c["baseline"]["SUCCESS_RELAXED"]))
            per_ic_delta[tid].append(impacts[-1])
        elif vals[best] == a and n_best == 1 and a > b56:
            cat = "actor_uniquely_better"
        elif best not in ("baseline", "tok_56_soft") and n_best == 1:
            cat = "other_uniquely_better"
        else:
            cat = "unresolved_or_tied_best"
        cats[cat] += 1
        details.append(
            {
                "snap_id": sid,
                "traj_id": tid,
                "cat": cat,
                "best": best if n_best == 1 else "tie",
                "d_succ_56_minus_actor": float(c["tok_56_soft"]["SUCCESS_RELAXED"] - c["baseline"]["SUCCESS_RELAXED"]),
                "d_r3_56_minus_actor": float(c["tok_56_soft"]["r3_ready"] - c["baseline"]["r3_ready"]),
                "split": "eval" if tid in eval_ics else "fit",
                "sw0": c["baseline"].get("sw0"),
            }
        )

    # Concentration: what fraction of total +Δsucc comes from top-k snaps
    pos = sorted([d for d in details if d["d_succ_56_minus_actor"] > 0], key=lambda x: -x["d_succ_56_minus_actor"])
    total_pos = sum(d["d_succ_56_minus_actor"] for d in pos) or 1.0
    top3_share = sum(d["d_succ_56_minus_actor"] for d in pos[:3]) / total_pos

    # Seed sensitivity on pilot for 56 vs actor
    pilot_snap = defaultdict(lambda: defaultdict(list))
    for r in pilot:
        pilot_snap[r["snap_id"]][r["candidate"]].append(r)
    seed_agree = seed_n = 0
    for sid, cands in pilot_snap.items():
        if "baseline" not in cands or "tok_56_soft" not in cands:
            continue
        # per seed: is 56 better on succ?
        by_k = defaultdict(dict)
        for name, rs in cands.items():
            for r in rs:
                by_k[r["seed_k"]][name] = r["SUCCESS_RELAXED"]
        if len(by_k) < 2:
            continue
        prefs = []
        for k, m in by_k.items():
            if m.get("tok_56_soft", 0) > m.get("baseline", 0):
                prefs.append("56")
            elif m.get("tok_56_soft", 0) < m.get("baseline", 0):
                prefs.append("actor")
            else:
                prefs.append("tie")
        seed_n += 1
        if len(set(prefs)) == 1:
            seed_agree += 1

    # Held-out means (P80 replicated)
    def mean_on(split_ics, cand):
        xs = [c[cand]["SUCCESS_RELAXED"] for sid, c in snap.items() if c["baseline"]["traj_id"] in split_ics]
        return float(np.mean(xs)) if xs else None

    report = {
        "n_snaps": len(snap),
        "categories": dict(cats),
        "category_rates": {k: v / len(snap) for k, v in cats.items()},
        "tok56_unique_best": {
            "n": cats["tok56_uniquely_better"],
            "mean_delta_succ_vs_actor": float(np.mean(impacts)) if impacts else None,
            "n_ics": len(per_ic_delta),
            "note": "Tied states are NOT counted as evidence favoring tok_56",
        },
        "impact_concentration": {
            "n_positive_delta_snaps": len(pos),
            "top3_share_of_positive_delta_succ": float(top3_share),
            "top_snaps": pos[:5],
            "dominated_by_few_states": bool(top3_share >= 0.5 and len(pos) >= 3),
        },
        "seed_sensitivity_pilot_56_vs_actor": {
            "n": seed_n,
            "frac_same_pref_across_seeds": float(seed_agree / seed_n) if seed_n else None,
        },
        "heldout_success": {
            "actor": mean_on(eval_ics, "baseline"),
            "tok56": mean_on(eval_ics, "tok_56_soft"),
            "tok50": mean_on(eval_ics, "tok_50_approach"),
            "tok82": mean_on(eval_ics, "tok_82_grip"),
            "n_eval_ics": len(eval_ics),
        },
        "artifact_assessment": {
            "oracle_equals_tok56_on_p80_eval": True,  # from P80
            "candidate_set_small": True,
            "many_ties": cats["all_tie"] / len(snap) >= 0.4,
            "likely_candidate_set_limited_ceiling": True,
            "note": (
                "Oracle SUCCESS = tok_56 on P80 eval means no better candidate in {56,50,82,actor} "
                "— not global optimality over M111."
            ),
        },
        "details_sample": details[:15],
    }
    (OUT / "CANDIDATE_SET_BIAS.md").write_text(
        "# Candidate-Set Bias Analysis (P81 Phase 2)\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"cats": dict(cats), "top3_share": top3_share, "n_56_unique": cats["tok56_uniquely_better"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_candidate_bias()
