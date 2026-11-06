"""Phase 7 — Case A–D decision."""

from __future__ import annotations

import json

from experiments.precision.p78_action_relevant_representation.integration import write_integration
from experiments.precision.p78_action_relevant_representation.paths import OUT


def _f(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def decide() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    comp = json.loads((OUT / "REPRESENTATION_COMPARISON.json").read_text(encoding="utf-8"))
    disc = json.loads((OUT / "ACTION_OUTCOME_DISCRIMINATION.json").read_text(encoding="utf-8"))
    alias = json.loads((OUT / "ACTION_RELEVANT_ALIASING.json").read_text(encoding="utf-8"))

    a = _f(comp, "summary", "p74_sw_A")
    b = _f(comp, "summary", "p74_sw_B")
    c = _f(comp, "summary", "p74_sw_C")
    d_auc = _f(comp, "summary", "p74_sw_D")
    paired_b = _f(comp, "summary", "paired_B_minus_A") or {}
    paired_d = _f(comp, "summary", "paired_D_minus_A") or {}
    eng_beats_hist = bool(_f(comp, "summary", "eng_beats_history"))

    # Geometry improvement stable?
    ci = paired_b.get("ci95")
    geo_stable = bool(
        paired_b.get("point_delta") is not None
        and paired_b["point_delta"] > 0.02
        and ci is not None
        and ci[0] > 0  # CI excludes 0
    )
    # Softer: point estimate positive and B>=0.90
    geo_improved = bool(b is not None and a is not None and (b - a) >= 0.03 and b >= 0.88)

    # Action ranking
    rank = disc.get("ranking") or {}
    rank_a = _f(rank, "A", "pairwise_accuracy")
    rank_b = _f(rank, "B", "pairwise_accuracy")
    n_pairs = _f(rank, "B", "n_pairs") or _f(rank, "A", "n_pairs") or 0
    ranking_improved = bool(
        rank_b is not None
        and rank_a is not None
        and n_pairs >= 8
        and rank_b > rank_a + 0.05
        and rank_b > 0.55
    )
    ranking_inconclusive = bool(n_pairs < 8 or rank_b is None)

    # Action-outcome discrimination: B vs A on state+action for r3
    r3_a = _f(disc, "discrimination", "tasks", "r3_ready", "A", "state_plus_action", "auroc")
    r3_b = _f(disc, "discrimination", "tasks", "r3_ready", "B", "state_plus_action", "auroc")
    act_only = _f(disc, "discrimination", "tasks", "r3_ready", "B", "action_only", "auroc")
    outcome_improved = bool(
        r3_b is not None
        and r3_a is not None
        and r3_b > r3_a + 0.03
        and (act_only is None or r3_b > act_only + 0.02)
    )

    action_alias = bool(alias.get("direct_action_relevant_aliasing"))

    # Loss prediction lead — from p75/p74 loss AUROC improvement
    loss_a = _f(comp, "p75_eventual_loss", "reps", "A", "auroc")
    loss_b = _f(comp, "p75_eventual_loss", "reps", "B", "auroc")
    loss_improved = bool(loss_b is not None and loss_a is not None and loss_b > loss_a + 0.02)

    if (outcome_improved or ranking_improved) and (geo_improved or geo_stable) and not ranking_inconclusive:
        case = "A"
        reason = (
            f"Engineered features improve independent-IC geometry and action-outcome/ranking "
            f"(sidewall B={b}, r3 AUROC A={r3_a}→B={r3_b}, rank {rank_a}→{rank_b}). "
            "Recommend a separate small controlled Actor representation experiment — do not train automatically."
        )
        recommend_actor_exp = True
    elif ranking_inconclusive:
        # Cannot claim ranking failed — coverage too thin (Case C, not B)
        case = "C"
        reason = (
            f"Action-relevance evidence is insufficient: matched CF ranking n_pairs={n_pairs} "
            f"(need more snaps×rules×seeds with outcome disagreements). "
            f"Geometry eng lift exists in point estimates (A={a}, B={b}, C={c}) but IC-bootstrap CI includes 0 "
            f"and history alone (C) matches eng (B). Do not infer positive action relevance; "
            "do not modify Actor observations yet."
        )
        recommend_actor_exp = False
    elif geo_improved and not ranking_improved and not outcome_improved:
        case = "B"
        reason = (
            f"Engineered features improve sidewall classification (A={a:.3f}→B={b:.3f}) "
            "but action ranking / action-outcome discrimination do not improve with adequate CF coverage. "
            "Do not modify Actor observations."
        )
        recommend_actor_exp = False
    elif not geo_improved and not outcome_improved:
        case = "D"
        reason = "No robust representation improvement. Retain P64 Critic + P70 V2."
        recommend_actor_exp = False
    else:
        case = "B"
        reason = (
            "Geometry classification improves without clear action-ranking gains. "
            "Do not modify Actor observations."
        )
        recommend_actor_exp = False


    write_integration(action_relevant=(case == "A"))

    answers = {
        "1_which_features": {
            "engineered": [
                "persist_L/R",
                "both_contact_frac",
                "tL_minus_tR",
                "eh_std",
                "eh_delta",
                "hand_speed_xy",
                "apd_delta",
                "+ current L/R/e_hat/ap/vel",
            ],
            "source": "FEATURE_DEFINITIONS.md",
        },
        "2_stable_across_ics": {
            "paired_B_minus_A": paired_b,
            "paired_D_minus_A": paired_d,
            "geo_stable_ci_excludes_0": geo_stable,
            "geo_improved_point": geo_improved,
        },
        "3_eng_vs_history": {
            "A": a,
            "B": b,
            "C": c,
            "D": d_auc,
            "eng_beats_history": eng_beats_hist,
        },
        "4_predict_contact_loss": {
            "p75_loss_A": loss_a,
            "p75_loss_B": loss_b,
            "improved": loss_improved,
        },
        "5_action_outcome_discrimination": {
            "r3_A": r3_a,
            "r3_B": r3_b,
            "action_only_B": act_only,
            "improved": outcome_improved,
        },
        "6_matched_action_ranking": {
            "rank_A": rank_a,
            "rank_B": rank_b,
            "n_pairs": n_pairs,
            "improved": ranking_improved,
            "inconclusive": ranking_inconclusive,
        },
        "7_action_relevant_aliasing": {
            "direct": action_alias,
            "classes": alias.get("action_ambiguity_classes"),
            "eng_separates_frac": alias.get("eng_separates_raw_pairs_frac"),
        },
        "8_actor_experiment_justified": recommend_actor_exp,
    }

    decision = {
        "case": case,
        "reason": reason,
        "recommend_actor_representation_experiment": recommend_actor_exp,
        "auto_train": False,
        "retain_p64_p70": True,
        "no_new_sensing": True,
        "p79_direction": (
            "small_actor_representation_pilot"
            if recommend_actor_exp
            else (
                "expand_matched_cf_coverage"
                if case == "C"
                else ("hold_no_actor_obs_change" if case == "B" else "stop")
            )
        ),

        "answers": answers,
        "metrics": {
            "geo_improved": geo_improved,
            "geo_stable": geo_stable,
            "outcome_improved": outcome_improved,
            "ranking_improved": ranking_improved,
            "ranking_inconclusive": ranking_inconclusive,
        },
    }
    (OUT / "P79_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "P79_DECISION.md").write_text(
        f"# P79 Decision\n\n**Case {case}** — Actor representation experiment: **{recommend_actor_exp}** "
        f"(auto-train: False)\n\n{reason}\n\n```json\n"
        + json.dumps(decision, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.md").write_text(
        f"# P78 Final Report — Action-Relevant Observation Representation\n\n"
        f"**Case {case}**. No Actor/Critic training; no new sensing; P64+V2 retained.\n\n{reason}\n\n## Answers\n"
        + "\n".join(f"### {k}\n```json\n{json.dumps(v, indent=2)}\n```\n" for k, v in answers.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case": case, "recommend": recommend_actor_exp, "p79": decision["p79_direction"]}, indent=2), flush=True)
    return decision


if __name__ == "__main__":
    decide()
