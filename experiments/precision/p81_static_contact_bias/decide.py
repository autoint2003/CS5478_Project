"""Phase 8 — Case A–D decision."""

from __future__ import annotations

import json

from experiments.precision.p81_static_contact_bias.paths import OUT


def decide(semantics, dist, bias, val, timing, full, mech, cross) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fc = dist.get("fc_decision") or {}
    delta = val.get("paired_delta_succ_ic") or {}
    ci = delta.get("ci95")
    ci_pos = bool(ci and ci[0] > 0)
    mean_pos = bool(delta.get("mean") is not None and delta["mean"] > 0.02)
    seed_stable = ((val.get("seed_sign_stability") or {}).get("frac_stable") or 0) >= 0.7
    underselect = bool(fc.get("systematic_underselect"))

    msp_d = full.get("msp_delta")
    full_skip = bool(full.get("skipped"))
    full_helps = bool(msp_d is not None and msp_d > 0.02 and not full.get("abd_any_regression"))
    artifact = bool((mech.get("primary_explanation") == "6_candidate_set_or_eval_artifact") or (bias.get("categories") or {}).get("all_tie", 0) >= 30)

    n_sw = val.get("correct_sw_rows") or 0
    timing_narrow = bool((timing.get("narrow_vs_broad") or {}).get("helps_only_fc"))

    if ci_pos and seed_stable and full_helps and underselect and not artifact:
        case = "A"
        reason = (
            f"tok_56 shows reproducible matched benefit (ΔSUCCESS mean={delta.get('mean')}, CI={ci}), "
            f"Actor systematically underselects it at FC, full-episode MSP improves ({msp_d}) without A/B/D regression. "
            "Recommend a separate conservative policy-correction study — do not train or deploy automatically."
        )
        p82 = "conservative_policy_correction_study_decision_only"
    elif mean_pos and not full_skip and msp_d is not None and msp_d <= 0.01:
        case = "B"
        reason = (
            f"Matched snapshots show directional tok_56 benefit (mean Δ={delta.get('mean')}) "
            f"but full-episode transfer fails (MSP Δ={msp_d}). "
            "Retain P64 Critic + P70 V2. Do not pursue fixed-token intervention. "
            "Document closed-loop transfer failure (consistent with P73 overlay non-generalization)."
        )
        p82 = "stop_fixed_token_direction"
    elif (not ci_pos) and (n_sw < 8 or timing_narrow or artifact or not mean_pos):
        case = "C"
        reason = (
            "Stage-dependent / insufficient evidence for a robust static tok_56 bias. "
            f"Matched CI excludes 0? {ci_pos}; mean Δ={delta.get('mean')}; "
            f"correct_sw_rows={n_sw}; seed_stable={seed_stable}; "
            f"impact_top3_share={(bias.get('impact_concentration') or {}).get('top3_share_of_positive_delta_succ')}; "
            f"primary_mechanism={mech.get('primary_explanation')}. "
            "Minimum additional evidence: ≥15–20 correct-sidewall FC ICs with matched Actor-vs-56, "
            "plus IC-clustered CI excluding 0, before any correction study."
        )
        p82 = "stop_or_bounded_correct_sw_cf_only"
    else:
        case = "D"
        reason = (
            f"Independent validation does not reproduce a robust static tok_56 bias "
            f"(Δ={delta}, underselect={underselect}, full={full.get('msp_delta') or full.get('skipped')}). "
            "Stop the static-bias direction. Retain P64 Critic + P70 V2."
        )
        p82 = "stop"

    # If matched benefit absent → prefer D over C when artifact/ties dominate without timing ambiguity
    if case == "C" and not mean_pos and artifact and (delta.get("mean") or 0) <= 0.01:
        case = "D"
        reason = (
            "No robust static tok_56 bias after matched re-validation: "
            f"mean IC ΔSUCCESS={delta.get('mean')}, CI={ci}, "
            f"all_tie-heavy candidate outcomes, Oracle≡tok_56 reflects candidate-set ceiling not proven bias. "
            "Stop static-bias direction. Retain P64 Critic + P70 V2."
        )
        p82 = "stop"

    answers = {
        "1_actor_underselects_56": underselect,
        "2_genuine_vs_artifact": {
            "genuine_matched_ci": ci_pos,
            "candidate_set_artifact_likely": artifact,
            "primary_mechanism": mech.get("primary_explanation"),
        },
        "3_generalizes_ics_seeds": {
            "ci_excludes_0": ci_pos,
            "seed_sign_stability": val.get("seed_sign_stability"),
            "n_ics": val.get("n_independent_ics"),
        },
        "4_helpful_stage": timing.get("by_stage"),
        "5_harms_correct_contact": (timing.get("already_opposing_adverse") or {}).get("mean_d_succ"),
        "6_full_episode_improves": full_helps if not full_skip else False,
        "7_abd_preserved": (not full.get("abd_any_regression")) if not full_skip else None,
        "8_mechanism_is_actor_bias": bool(underselect and ci_pos and mech.get("primary_explanation") != "6_candidate_set_or_eval_artifact"),
        "9_future_correction_justified": case == "A",
        "10_p82_should": p82,
        "tok56_semantics_family": (semantics.get("codebook_decode") or {}).get("family"),
        "p80_delta_not_full_episode_proof": True,
    }

    decision = {
        "case": case,
        "reason": reason,
        "p82": p82,
        "authorize_actor_training": False,
        "authorize_critic_training": False,
        "deploy_fixed_token_overlay": False,
        "retain": "P64 Critic + P70 V2",
        "answers": answers,
    }
    (OUT / "P82_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "P82_DECISION.md").write_text(
        f"# P82 Decision (from P81)\n\n**Case {case}**\n\n{reason}\n\n"
        f"- P82: `{p82}`\n- Actor training: **No**\n- Deploy overlay: **No**\n"
        f"- Retain: P64 Critic + P70 V2\n\n```json\n"
        + json.dumps(answers, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    final = {
        "title": "P81 Static Contact-Stage Action Bias Audit",
        "case": case,
        "reason": reason,
        "p82": p82,
        "answers": answers,
        "constraints_honored": [
            "No Actor/Critic training",
            "No PPO/GRPO/BC/DAgger",
            "No new sensing/obs features",
            "No M111 expansion / reward changes",
            "No deployed fixed-token overlay",
            "No production baseline modifications",
            "No final-test evaluation",
            "No automatic P82 training",
            "Retained P64 Critic + P70 V2",
        ],
    }
    (OUT / "FINAL_REPORT.md").write_text(
        f"# P81 Final Report\n\n## Decision: Case {case}\n\n{reason}\n\n"
        "## Final Answers\n\n```json\n"
        + json.dumps(answers, indent=2)
        + "\n```\n\n## Constraints\n\n"
        + "\n".join(f"- {c}" for c in final["constraints_honored"])
        + "\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"case": case, "p82": p82}, indent=2), flush=True)
    return decision
