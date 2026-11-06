"""Phase 8 — Case A–D decision."""

from __future__ import annotations

import json

from experiments.precision.p80_policy_learning_feasibility.paths import OUT


def decide(audit: dict, selection: dict, coverage: dict, learn: dict, retention: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rev = selection["reversals"]
    offline = selection["offline"]
    paired = selection["paired"]
    methods = selection["static_vs"]["methods"]

    d_succ = methods["D_history_ranker"]["SUCCESS_RELAXED"]
    c_succ = methods["C_best_fixed_fit"]["SUCCESS_RELAXED"]
    a_succ = methods["A_actor"]["SUCCESS_RELAXED"]
    b_succ = methods["B_always_56"]["SUCCESS_RELAXED"]
    e_succ = methods["E_oracle"]["SUCCESS_RELAXED"]

    d_minus_c = paired["D_minus_C_succ"]
    d_minus_b = paired["D_minus_B_succ"]
    ci = d_minus_c.get("ci95")

    beats_fixed = bool(
        d_minus_c.get("mean") is not None
        and d_minus_c["mean"] > 0.02
        and ci is not None
        and ci[0] > 0
    )
    # softer point estimate
    beats_fixed_point = bool(d_minus_c.get("mean") is not None and d_minus_c["mean"] > 0.02)

    frac_match_fixed = offline.get("frac_ranker_matches_best_fixed") or 0
    n_rev = rev["n_reliable_reversals_vs_56"]
    pref_mode = rev["preference_mode"]
    gaps = coverage.get("gaps") or []
    coverage_ok = len(gaps) == 0

    # Robustness: check IC concentration of D wins
    per_snap = selection.get("per_snap") or []
    d_wins = []
    for p in per_snap:
        if p["D_history_ranker"]["SUCCESS_RELAXED"] > p["C_best_fixed_fit"]["SUCCESS_RELAXED"]:
            d_wins.append(p["traj_id"])
    unique_win_ics = sorted(set(d_wins))

    ranking_acc_note = "P79 history pairwise accuracy 0.75 ≠ selection utility"

    if beats_fixed and n_rev >= 8 and coverage_ok and len(unique_win_ics) >= 5:
        case = "A"
        reason = (
            f"History-conditioned selection outperforms best fixed action on held-out ICs "
            f"(D_succ={d_succ:.3f} vs C_succ={c_succ:.3f}; paired Δ CI excludes 0), "
            f"with {n_rev} reliable preference reversals and adequate coverage. "
            "Recommend a separate P81 preference-learning pilot — do not train automatically."
        )
        p81 = "controlled_preference_learning_pilot_decision_only"
    elif (
        pref_mode == "global_bias_tok56"
        or (frac_match_fixed >= 0.8 and not beats_fixed)
        or (selection["fixed_name"] == "tok_56_soft" and abs(d_succ - c_succ) <= 0.02 and abs(d_succ - b_succ) <= 0.02)
    ):
        case = "B"
        reason = (
            f"Static action explains most gains: best_fixed={selection['fixed_name']}, "
            f"ranker matches fixed on {frac_match_fixed:.0%} of eval snaps; "
            f"D_succ={d_succ:.3f}, C_succ={c_succ:.3f}, always56={b_succ:.3f}, actor={a_succ:.3f}. "
            f"Preference mode={pref_mode}, reliable reversals vs 56={n_rev}. "
            "Do not train Actor yet. Investigate global contact-stage action bias. "
            "Do not deploy a fixed-token overlay automatically."
        )
        p81 = "investigate_static_bias_no_train"
    elif gaps or n_rev < 5 or (beats_fixed_point and not beats_fixed and not coverage_ok):
        case = "C"
        reason = (
            "Insufficient coverage / preference-reversal evidence for a controlled Actor learning pilot. "
            f"Gaps={json.dumps([g['gap'] for g in gaps])}; reliable_reversals={n_rev}; "
            f"correct_sidewall_ics={coverage.get('sidewall', {}).get('correct_ics')}; "
            f"D_minus_C={d_minus_c}. Recommend targeted matched CF expansion before any training."
        )
        p81 = "targeted_cf_expansion"
    else:
        # No reliable selection benefit
        case = "D"
        reason = (
            f"Action-ranking accuracy does not translate into meaningful selection gains over best fixed "
            f"(D_succ={d_succ:.3f}, C_succ={c_succ:.3f}, paired={d_minus_c}). "
            "Retain P64 Critic + P70 V2. Stop current policy-learning direction until new evidence."
        )
        p81 = "stop"

    # Refine: if Case B-like numbers but gaps dominate → still C if correct-SW missing and reversals unclear
    # Prefer C over B when coverage critically missing AND we cannot claim static bias confidently
    if case == "B" and coverage.get("sidewall", {}).get("correct_ics", 0) < 3 and n_rev >= 3:
        # static bias claim itself may not generalize to correct-SW
        case = "C"
        reason = (
            "Apparent static-token gains are measured almost entirely on incorrect-sidewall FC states "
            f"(correct_sidewall_ics={coverage.get('sidewall', {}).get('correct_ics')}). "
            f"Ranker vs fixed Δ not CI-robust (paired={d_minus_c}); reliable reversals={n_rev}. "
            "Do not authorize Actor training; expand correct-sidewall matched CF coverage first."
        )
        p81 = "targeted_cf_expansion"

    answers = {
        "1_p79_stats_consistent": audit.get("consistent_with_p79_headlines") and audit.get("denominator_identity_holds"),
        "2_history_beats_best_fixed": {
            "point": beats_fixed_point,
            "ci_excludes_0": beats_fixed,
            "D_succ": d_succ,
            "C_succ": c_succ,
            "paired": d_minus_c,
        },
        "3_pref_change_frequency": {
            "frac_tok56": rev.get("frac_tok56_preferred"),
            "frac_other": rev.get("frac_other_preferred"),
            "frac_tie": rev.get("frac_all_tie"),
            "counts": rev.get("preference_counts"),
        },
        "4_reversals_reliable": {
            "n_reliable_vs_56": n_rev,
            "seed_flip_rate": (rev.get("seed_sensitivity_pilot") or {}).get("flip_rate"),
            "mode": pref_mode,
        },
        "5_ranker_reduces_regret": {
            "D_regret_r3": offline["regret"]["D_mean_regret_r3"],
            "C_regret_r3": offline["regret"]["C_mean_regret_r3"],
            "A_regret_r3": offline["regret"]["A_mean_regret_r3"],
        },
        "6_robust_across_ics": {
            "D_minus_C_ci": ci,
            "win_ics": unique_win_ics,
            "n_eval_ics": methods["D_history_ranker"]["n_eval_ics"],
        },
        "7_dataset_sufficient_for_actor_pilot": coverage_ok and beats_fixed and n_rev >= 8,
        "8_most_defensible_objective": learn.get("most_defensible_if_training_justified"),
        "9_abd_protection": retention.get("constraints_before_any_future_training"),
        "10_p81_should": p81,
        "oracle_upper_bound_succ": e_succ,
        "actor_succ": a_succ,
        "always56_succ": b_succ,
        "ranking_vs_utility": ranking_acc_note,
    }

    decision = {
        "case": case,
        "reason": reason,
        "p81": p81,
        "authorize_actor_training": False,
        "authorize_critic_training": False,
        "retain": "P64 Critic + P70 V2",
        "answers": answers,
    }

    (OUT / "P81_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "P81_DECISION.md").write_text(
        f"# P81 Decision (from P80)\n\n**Case {case}**\n\n{reason}\n\n"
        f"- P81 recommendation: `{p81}`\n"
        f"- Actor training authorized: **No**\n"
        f"- Retain: P64 Critic + P70 V2\n\n```json\n"
        + json.dumps(answers, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    final = {
        "title": "P80 History-Conditioned Policy Learning Feasibility",
        "case": case,
        "reason": reason,
        "p81": p81,
        "answers": answers,
        "constraints_honored": [
            "No Actor/Critic training",
            "No PPO/GRPO/BC/DAgger updates",
            "No new observation features / sensors",
            "No M111 expansion / overlays",
            "No reward/physics/R3F changes",
            "No final-test evaluation",
            "No automatic P81 training",
            "Retained P64 Critic + P70 V2",
        ],
    }
    (OUT / "FINAL_REPORT.md").write_text(
        "# P80 Final Report\n\n"
        f"## Decision: Case {case}\n\n{reason}\n\n"
        "## Final Answers\n\n```json\n"
        + json.dumps(answers, indent=2)
        + "\n```\n\n## Constraints\n\n"
        + "\n".join(f"- {c}" for c in final["constraints_honored"])
        + "\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"case": case, "p81": p81, "D": d_succ, "C": c_succ, "B": b_succ, "A": a_succ}, indent=2), flush=True)
    return decision
