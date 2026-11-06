"""Phases 6–7 — learning objective feasibility + cross-class retention plan."""

from __future__ import annotations

import json

from experiments.precision.p80_policy_learning_feasibility.paths import OUT


def write_learning_feasibility(selection: dict, coverage: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rev = selection["reversals"]
    offline = selection["offline"]
    paired = selection["paired"]
    n_rev = rev["n_reliable_reversals_vs_56"]
    pref_mode = rev["preference_mode"]
    d_vs_c = paired["D_minus_C_succ"]
    ci = d_vs_c.get("ci95")
    conditional_beats_static = bool(
        d_vs_c.get("mean") is not None
        and d_vs_c["mean"] > 0.02
        and ci is not None
        and ci[0] > 0
    )

    approaches = {
        "1_preference_supervised": {
            "description": "Supervised preference / ranking loss on matched CF pairs with reliable outcome differences",
            "label_availability": f"{n_rev} reliable reversals; 86 nontrivial pairs overall (many ties)",
            "label_noise": "Pilot seed flip_rate=" + str((rev.get("seed_sensitivity_pilot") or {}).get("flip_rate")),
            "coverage": coverage.get("gaps"),
            "preserves_m111_multiskill": "Medium — can constrain updates to FC window / C-skill contexts only",
            "overfit_C_risk": "High if trained only on C FC snaps",
            "critic_dependence": "None",
            "abd_regression_risk": "Low–medium if FC-gated and A/B/D frozen elsewhere",
            "defensibility_now": "Best among options IF conditional>static AND coverage gaps closed",
            "score": 3 if (n_rev >= 10 and not coverage["gaps"]) else 2 if n_rev >= 5 else 1,
        },
        "2_advantage_weighted_bc": {
            "description": "AW-BC on actions with positive matched Δoutcome vs Actor baseline",
            "label_availability": "Uses same CF deltas; filters ties",
            "label_noise": "Same seed sensitivity; need minimum effect-size threshold",
            "coverage": "Same sidewall gap",
            "preserves_m111_multiskill": "Medium–high with small learning rate / trust region on token logits",
            "overfit_C_risk": "High",
            "critic_dependence": "None (uses CF outcomes, not Critic)",
            "abd_regression_risk": "Medium",
            "defensibility_now": "Reasonable second choice after preference pairs; still needs coverage",
            "score": 2,
        },
        "3_conservative_around_p64": {
            "description": "Conservative policy improvement / KL-ball around frozen P64 at FC only",
            "label_availability": "Requires reliable advantages or preferences",
            "label_noise": "Ties dominate (273/396)",
            "coverage": "Insufficient correct-SW",
            "preserves_m111_multiskill": "Highest among learning options",
            "overfit_C_risk": "Medium",
            "critic_dependence": "Optional; can use CF returns instead",
            "abd_regression_risk": "Lowest if KL tight and non-C actions frozen",
            "defensibility_now": "Good safety wrapper IF a learning signal exists; not a substitute for coverage",
            "score": 2,
        },
        "4_actor_ppo_existing_critic": {
            "description": "Actor PPO with frozen/current P64 Critic",
            "label_availability": "N/A — on-policy advantages",
            "label_noise": "P75/P76: Critic FC-window advantages unreliable",
            "coverage": "Would not fix representation/coverage issues",
            "preserves_m111_multiskill": "Poor without strong constraints",
            "overfit_C_risk": "High",
            "critic_dependence": "Critical — and currently unsupported by P75/P76",
            "abd_regression_risk": "High",
            "defensibility_now": "Not defensible as first intervention",
            "score": 0,
        },
    }

    # Pick most defensible IF training later justified
    ranked = sorted(approaches.items(), key=lambda kv: -kv[1]["score"])
    recommended_if_train = ranked[0][0]
    if approaches["4_actor_ppo_existing_critic"]["score"] >= approaches[recommended_if_train]["score"]:
        recommended_if_train = "1_preference_supervised"

    # Given current evidence
    if conditional_beats_static and not coverage["gaps"]:
        stance = "small_preference_pilot_justified_later"
    elif pref_mode in ("global_bias_tok56",) or (
        offline.get("frac_ranker_matches_best_fixed") is not None
        and offline["frac_ranker_matches_best_fixed"] >= 0.85
        and (d_vs_c.get("mean") or 0) <= 0.02
    ):
        stance = "static_bias_explains_gains_no_train"
    elif coverage["gaps"]:
        stance = "collect_cf_before_any_train"
    else:
        stance = "no_reliable_selection_benefit"

    report = {
        "approaches": approaches,
        "most_defensible_if_training_justified": "1_preference_supervised",
        "do_not_auto_choose_ppo": True,
        "p75_p76_critic_warning": (
            "Existing Critic does not provide reliable FC-window PPO advantages; "
            "do not assume PPO+Critic is viable."
        ),
        "smallest_intervention_principle": (
            "If later justified: preference-based supervised improvement on reliable CF pairs, "
            "FC/C-gated, with conservative KL to P64 — not full PPO."
        ),
        "current_stance": stance,
        "conditional_beats_static_ci": conditional_beats_static,
        "D_minus_C_succ": d_vs_c,
    }
    (OUT / "LEARNING_OBJECTIVE_FEASIBILITY.md").write_text(
        "# Learning Objective Feasibility (P80 Phase 6)\n\n"
        "No models trained in P80.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return report


def write_retention_plan() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "primary_promotion_metric": "Overall A/B/C/D SUCCESS_RELAXED MSP (matched full-episode)",
        "local_fc_improvement_insufficient": True,
        "frozen_references": {
            "actor": "P64_step4_actor.pt",
            "critic": "P64_step4_critic.pt (no Critic training assumed)",
            "overlay": "P70 V2 unchanged",
            "vocab": "M111 unchanged",
            "obs": "39D unchanged",
        },
        "constraints_before_any_future_training": [
            "IC-disjoint fit / validation / final-test (final-test never for selection)",
            "Matched full-episode eval vs frozen P64+V2",
            "A/B/D behavior-preservation: max allowed MSP drop per class (pre-register, e.g. ≤1–2 pp)",
            "C first-contact + terminal SUCCESS_RELAXED reported separately",
            "No new overlays in production path during pilot",
            "Abort if A/B/D regression exceeds budget even if C improves",
        ],
        "evaluation_protocol_sketch": {
            "1": "Freeze candidate checkpoint",
            "2": "Roll matched seeds on held-out ICs for A/B/C/D",
            "3": "Compare MSP and per-class SUCCESS_RELAXED vs P64+V2",
            "4": "Promote only if overall MSP improves and A/B/D within budget",
        },
        "p80_note": "No new policy checkpoints evaluated in P80 (no Actor training permitted).",
    }
    (OUT / "CROSS_CLASS_RETENTION_PLAN.md").write_text(
        "# Cross-Class Retention Plan (P80 Phase 7)\n\n```json\n"
        + json.dumps(plan, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return plan
