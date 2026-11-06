"""Phase 10 — Case A–D decision + P80 recommendation."""

from __future__ import annotations

import json

from experiments.precision.p79_matched_cf_coverage.paths import OUT


def _f(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def decide(outcome: dict, analysis: dict, audit: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = analysis.get("rep") or {}
    alias = analysis.get("alias") or {}
    valid = analysis.get("valid") or {}
    bench = analysis.get("ranking_bench") or {}

    n_ics = int(outcome.get("n_independent_ics") or 0)
    n_disagree = int(outcome.get("n_nontrivial_outcome_disagreements") or 0)
    n_pairs = int(outcome.get("n_valid_action_comparisons") or 0)
    disagree_rate = outcome.get("disagree_rate")
    det = (outcome.get("determinism") or {}).get("deterministic_fraction")

    rank_a = _f(rep, "summary", "rank_A")
    rank_b = _f(rep, "summary", "rank_B")
    rank_c = _f(rep, "summary", "rank_C")
    rank_d = _f(rep, "summary", "rank_D")
    n_eval_pairs = _f(rep, "summary", "n_eval_pairs") or 0
    paired_ba = rep.get("paired_B_minus_A") or {}
    paired_dc = rep.get("paired_D_minus_C") or {}

    restore_ok = bool(_f(audit, "ok")) and bool(_f(audit, "checks", "qpos_restore_idempotent"))
    actor_owns = bool(_f(audit, "checks", "actor_owns_fc_decision"))

    # Coverage gates
    coverage_ok = n_ics >= 20 and n_disagree >= 8 and n_eval_pairs >= 8
    coverage_thin = n_ics < 20 or n_disagree < 5 or n_eval_pairs < 5

    actions_matter = bool(
        disagree_rate is not None and disagree_rate >= 0.05 and n_disagree >= 5
    )
    little_action_effect = bool(
        disagree_rate is not None and disagree_rate < 0.05 and n_disagree < 5
    )

    # Engineered must beat *history*, not merely current-frame (P78 lesson).
    eng_over_hist = bool(
        rank_d is not None
        and rank_c is not None
        and n_eval_pairs >= 8
        and (rank_d - rank_c) >= 0.05
        and rank_d > 0.55
    )
    eng_over_current = bool(
        rank_b is not None
        and rank_a is not None
        and n_eval_pairs >= 8
        and (rank_b - rank_a) >= 0.05
        and rank_b > 0.55
    )
    hist_as_good = bool(
        rank_c is not None
        and (
            (rank_b is not None and rank_c + 1e-9 >= rank_b - 0.02)
            or (rank_d is not None and abs(rank_c - rank_d) < 0.05)
        )
    )
    eng_helps_ranking = bool(eng_over_hist)  # primary criterion for Case A

    alias_ok = bool(alias.get("action_relevant_aliasing_established"))

    if little_action_effect and not coverage_thin:
        case = "D"
        reason = (
            f"Candidate M111 single-token actions rarely change outcomes "
            f"(disagree_rate={disagree_rate}, n_nontrivial={n_disagree}/{n_pairs}). "
            "Do not pursue observation representation as the primary intervention; "
            "investigate action controllability / M111 vocabulary in a separate study. "
            "Do not modify deployed vocabulary automatically."
        )
        p80 = "investigate_controllability"
        train = False
    elif coverage_thin or not coverage_ok:
        case = "C"
        missing = []
        if n_ics < 30:
            missing.append(f"independent_ics={n_ics} (want ≥30 eval-capable)")
        if n_disagree < 8:
            missing.append(f"nontrivial_disagreements={n_disagree}")
        if n_eval_pairs < 8:
            missing.append(f"heldout_ranked_pairs={n_eval_pairs}")
        reason = (
            "Counterfactual evidence still insufficient for an action-relevant representation claim. "
            f"Missing: {', '.join(missing) or 'robust held-out ranking'}. "
            "Do not authorize Actor training."
        )
        p80 = "stop_or_expand_coverage"
        train = False
    elif actions_matter and coverage_ok and eng_over_hist:
        case = "A"
        reason = (
            f"Sufficient IC coverage (n_ics={n_ics}), reliable action-outcome disagreements "
            f"(n_nontrivial={n_disagree}), and engineered features improve held-out ranking "
            f"over history (A={rank_a}, B={rank_b}, C={rank_c}, D={rank_d}; aliasing={alias_ok}). "
            "Recommend a separate controlled Actor representation experiment — do not train automatically."
        )
        p80 = "controlled_actor_rep_experiment_decision_only"
        train = False
    elif actions_matter and (hist_as_good or not eng_over_hist):
        case = "B"
        reason = (
            f"Matched actions have meaningful outcome differences (disagree_rate={disagree_rate}, "
            f"n_nontrivial={n_disagree}), but existing history performs as well as engineered features "
            f"(rank A={rank_a}, B={rank_b}, C={rank_c}, D={rank_d}; "
            f"eng_over_current={eng_over_current}, eng_over_hist={eng_over_hist}). "
            "Do not change Actor observations. Recommend studying policy learning from existing "
            "observation histories subject to a separate training decision."
        )
        p80 = "study_policy_from_existing_history"
        train = False
    else:
        case = "C"
        reason = (
            "Evidence remains inconclusive after coverage expansion. "
            "Do not authorize Actor training."
        )
        p80 = "stop_or_expand_coverage"
        train = False

    # Safety: never auto-train
    assert train is False

    answers = {
        "1_independent_ics": n_ics,
        "2_restore_and_matched_valid": {
            "restore_ok": restore_ok,
            "actor_owns_fc": actor_owns,
            "audit_checks": _f(audit, "checks"),
        },
        "3_actions_change_outcomes": actions_matter,
        "4_disagreement_frequency": {
            "disagree_rate": disagree_rate,
            "n_nontrivial": n_disagree,
            "n_ties": outcome.get("n_ties"),
            "n_comparisons": n_pairs,
        },
        "5_preference_stability": {
            "deterministic_fraction": det,
            "note": (outcome.get("determinism") or {}).get("implication"),
        },
        "6_history_predicts_pref": {
            "rank_C": rank_c,
            "rank_A": rank_a,
        },
        "7_eng_improves_ranking": {
            "rank_B": rank_b,
            "rank_D": rank_d,
            "B_minus_A": paired_ba,
            "D_minus_C": paired_dc,
            "eng_over_current_frame": eng_over_current,
            "eng_over_history": eng_over_hist,
            "eng_helps": eng_over_hist,
        },
        "8_action_relevant_aliasing": alias.get("action_relevant_aliasing_established"),
        "9_controlled_actor_experiment_justified": case == "A",
        "10_p80_should": p80,
    }

    decision = {
        "case": case,
        "reason": reason,
        "p80": p80,
        "authorize_actor_training": False,
        "authorize_critic_training": False,
        "retain": "P64 Critic + P70 V2",
        "answers": answers,
        "coverage": {
            "n_ics": n_ics,
            "n_snaps": outcome.get("n_unique_snapshots"),
            "n_eval_pairs": n_eval_pairs,
            "n_disagree": n_disagree,
            "warnings": valid.get("warnings"),
        },
    }

    (OUT / "P80_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "P80_DECISION.md").write_text(
        f"# P80 Decision (from P79)\n\n**Case {case}**\n\n{reason}\n\n"
        f"- P80 recommendation: `{p80}`\n"
        f"- Actor training authorized: **No**\n"
        f"- Retain: P64 Critic + P70 V2\n\n```json\n"
        + json.dumps(answers, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )

    final = {
        "title": "P79 Matched Counterfactual Coverage Expansion",
        "case": case,
        "reason": reason,
        "p80": p80,
        "answers": answers,
        "constraints_honored": [
            "No Actor/Critic training",
            "No PPO/GRPO/BC/DAgger",
            "No M111 expansion",
            "No new sensors/overlays",
            "No privileged policy inputs",
            "No final-test evaluation for selection",
            "No automatic P80 training",
            "Retained P64 Critic + P70 V2",
        ],
    }
    (OUT / "FINAL_REPORT.md").write_text(
        "# P79 Final Report\n\n"
        f"## Decision: Case {case}\n\n{reason}\n\n"
        "## Final Answers\n\n```json\n"
        + json.dumps(answers, indent=2)
        + "\n```\n\n## Constraints\n\n"
        + "\n".join(f"- {c}" for c in final["constraints_honored"])
        + "\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"case": case, "p80": p80, "n_ics": n_ics, "n_disagree": n_disagree}, indent=2), flush=True)
    return decision


if __name__ == "__main__":
    audit = json.loads((OUT / "SNAPSHOT_RESTORE_AUDIT.json").read_text(encoding="utf-8"))
    # Expect analysis artifacts already written; minimal reload
    from pathlib import Path
    import pickle
    from experiments.precision.p79_matched_cf_coverage.paths import CACHE
    from experiments.precision.p79_matched_cf_coverage.cf_run import analyze_outcomes
    from experiments.precision.p79_matched_cf_coverage.analyze import run_analysis

    tag = "expand" if (CACHE / "cf_rows_expand.pkl").exists() else "pilot"
    rows = pickle.loads((CACHE / f"cf_rows_{tag}.pkl").read_bytes())
    outcome = analyze_outcomes(rows)
    analysis = run_analysis(rows, outcome)
    decide(outcome, analysis, audit)
