"""Phase 7 — Case A–D decision + FINAL_REPORT."""

from __future__ import annotations

import json

from experiments.precision.p77_contact_observability.paths import OUT


def _f(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def decide() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    existing = json.loads((OUT / "EXISTING_OBSERVATION_INFORMATION.json").read_text(encoding="utf-8"))
    alias = json.loads((OUT / "RARE_POSITIVE_ALIASING.json").read_text(encoding="utf-8"))
    gain = json.loads((OUT / "INCREMENTAL_INFORMATION_GAIN.json").read_text(encoding="utf-8"))
    action = json.loads((OUT / "ACTION_RELEVANCE.json").read_text(encoding="utf-8"))
    catalog = json.loads((OUT / "SENSING_FEASIBILITY.json").read_text(encoding="utf-8"))

    # Existing feature engineering lift on sidewall
    sw = _f(existing, "p74_sidewall_detection", "tasks", "correct_sidewall_now") or {}
    a_auc = _f(sw, "A", "auroc")
    d_auc = _f(sw, "D", "auroc")
    c_auc = _f(sw, "C", "auroc")
    eng_lift = None if a_auc is None or d_auc is None else d_auc - a_auc
    hist_lift = None if a_auc is None or c_auc is None else c_auc - a_auc
    # Material improvement: absolute discrimination strong + nonzero lift, or large lift
    existing_sufficient = bool(
        (d_auc is not None and d_auc >= 0.90 and eng_lift is not None and eng_lift >= 0.03)
        or (eng_lift is not None and eng_lift > 0.08 and (d_auc or 0) > 0.75)
        or (hist_lift is not None and hist_lift > 0.08 and (c_auc or 0) > 0.75)
    )

    # Sensing gain: Ft vs base on sidewall
    gsw = _f(gain, "tasks", "correct_sidewall_now") or {}
    base_g = _f(gsw, "base", "auroc")
    ft_g = _f(gsw, "plus_ft", "auroc")
    cop_g = _f(gsw, "plus_cop_hist", "auroc")
    ft_cop_g = _f(gsw, "plus_ft_cop", "auroc")
    priv_g = _f(gsw, "plus_priv_D", "auroc")
    best_phys = max(
        [
            ("plus_ft", ft_g),
            ("plus_cop_hist", cop_g),
            ("plus_ft_cop", ft_cop_g),
        ],
        key=lambda x: (x[1] if x[1] is not None else -1),
    )
    sense_lift = None if base_g is None or best_phys[1] is None else best_phys[1] - base_g
    substantial_sensing = bool(sense_lift is not None and sense_lift > 0.05 and (best_phys[1] or 0) > 0.7)

    lead = gain.get("detection_lead_before_loss_steps") or {}
    before_loss = bool(lead.get("frac_positive_lead") is not None and lead["frac_positive_lead"] > 0.5)

    action_rel = bool(action.get("action_relevant_sensing_signal"))
    pair_out = _f(action, "discrimination", "3_action_choice_discrimination", "outcome_differs_frac")
    alias_robust = bool(_f(alias, "verdict", "aliasing_robust"))

    if existing_sufficient and not substantial_sensing:
        case = "A"
        reason = (
            f"Causal engineered / history features materially improve sidewall discrimination "
            f"(A AUROC={a_auc}, D={d_auc}, C={c_auc}). Recommend a controlled observation-representation "
            "experiment — do not train automatically."
        )
        p78 = "reuse_existing_observations_representation"
        recommend_obs_ext = True
        recommend_sensing_hw = False
    elif substantial_sensing and action_rel:
        case = "B"
        reason = (
            f"Additional physical sensing ({best_phys[0]}) provides substantial incremental sidewall info "
            f"(base={base_g} → {best_phys[1]}, Δ={sense_lift}) with action-relevant discrimination support. "
            "Recommend a minimal sensing extension with realistic noise/latency; keep privileged geometry diagnostic-only."
        )
        p78 = "sensing_extension"
        recommend_obs_ext = True
        recommend_sensing_hw = True
    elif substantial_sensing and not action_rel:
        case = "C"
        reason = (
            f"Sensing improves geometry classification ({best_phys[0]} AUROC {base_g}→{best_phys[1]}) "
            "but does not establish action-choice discrimination sufficient for Actor expansion. "
            "Do not expand Actor observations yet; investigate contact-control separately."
        )
        p78 = "investigate_contact_control_not_obs_expand"
        recommend_obs_ext = False
        recommend_sensing_hw = False
    else:
        case = "D"
        reason = (
            f"No reliable improvement: existing eng lift={eng_lift}, sensing lift={sense_lift} "
            f"(best={best_phys}). Retain P64 Critic + P70 V2; stop contact-observability improvement direction."
        )
        p78 = "stop"
        recommend_obs_ext = False
        recommend_sensing_hw = False

    # nuance: if sensing helps class but Case C, still note B-class Ft is sim-derivable
    answers = {
        "1_already_observable": {
            "in_39d": [
                "e_hat_x",
                "contact_L/R",
                "CoP u,v",
                "fn_L/R",
                "aperture/ap_dot",
                "hand vel",
            ],
            "sidewall_auroc_current": a_auc,
            "sidewall_auroc_engineered": d_auc,
            "sidewall_auroc_full_hist": c_auc,
            "note": "Pad CoP/fn exist but do not equal cylinder sidewall class.",
        },
        "2_history_vs_current": {
            "current_A": a_auc,
            "short_B": _f(sw, "B", "auroc"),
            "full_C": c_auc,
            "history_helps": bool(hist_lift is not None and hist_lift > 0.03),
        },
        "3_aliasing_robust": {
            "robust": alias_robust,
            "prevalence": alias.get("prevalence"),
            "q05": _f(alias, "multi_radius", "by_radius", "0.05"),
            "matched_pairs": alias.get("matched_pairs_diff_sidewall"),
        },
        "4_best_physical_signal": {
            "name": best_phys[0],
            "auroc": best_phys[1],
            "base": base_g,
            "lift": sense_lift,
            "catalog_ft": catalog.get("4_slip_or_shear_force"),
            "priv_D_ceiling": priv_g,
        },
        "5_before_contact_loss": {
            "lead": lead,
            "available_before_loss": before_loss,
        },
        "6_distinguishes_actions": {
            "action_relevant_flag": action_rel,
            "pair_outcome_differs": pair_out,
            "p73": _f(action, "p73_counterfactuals", "soft_approach"),
        },
        "7_observation_extension_justified": recommend_obs_ext,
        "8_p78": p78,
    }

    decision = {
        "case": case,
        "reason": reason,
        "retain_p64_p70": True,
        "no_actor_training": True,
        "no_critic_training": True,
        "recommend_observation_extension": recommend_obs_ext,
        "recommend_sensing_hardware_or_sim_B": recommend_sensing_hw,
        "p78_direction": p78,
        "auto_train": False,
        "answers": answers,
        "metrics": {
            "existing_sufficient": existing_sufficient,
            "substantial_sensing": substantial_sensing,
            "action_rel": action_rel,
            "eng_lift": eng_lift,
            "sense_lift": sense_lift,
            "best_phys": best_phys[0],
        },
    }
    (OUT / "P78_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "P78_DECISION.md").write_text(
        f"# P78 Decision\n\n**Case {case}** — P78 direction: `{p78}`\n\n{reason}\n\n```json\n"
        + json.dumps(decision, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.md").write_text(
        f"# P77 Final Report — Contact Geometry Observability & Sensing Feasibility\n\n"
        f"**Case {case}**. No Actor/Critic training; P64+V2 retained.\n\n{reason}\n\n## Answers\n"
        + "\n".join(f"### {k}\n```json\n{json.dumps(v, indent=2)}\n```\n" for k, v in answers.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case": case, "p78": p78, "obs_ext": recommend_obs_ext}, indent=2), flush=True)
    return decision


if __name__ == "__main__":
    decide()
