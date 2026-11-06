"""Phase 6 — Case A–D decision + FINAL_REPORT."""

from __future__ import annotations

import json

from experiments.precision.p76_value_observability_audit.paths import OUT


def _f(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def decide() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ind = json.loads((OUT / "INDEPENDENT_IC_VALIDATION.json").read_text(encoding="utf-8"))
    scale = json.loads((OUT / "VALUE_SCALE_ANALYSIS.json").read_text(encoding="utf-8"))
    adv = json.loads((OUT / "ADVANTAGE_DIAGNOSTICS.json").read_text(encoding="utf-8"))
    alias = json.loads((OUT / "OBSERVATION_ALIASING.json").read_text(encoding="utf-8"))
    bott = json.loads((OUT / "INFORMATION_BOTTLENECK.json").read_text(encoding="utf-8"))

    # Focus on A_w1 (P75 paradox) and P64
    aw = ind["results"]["A_w1"]
    p64 = ind["results"]["P64"]
    aw_raw_ev = _f(aw, "raw_fc", "ev")
    aw_cal_ev = _f(aw, "fc", "ev")
    p64_raw_ev = _f(p64, "raw_fc", "ev")
    p64_cal_ev = _f(p64, "fc", "ev")

    ev_restored = (
        aw_cal_ev is not None
        and aw_raw_ev is not None
        and aw_cal_ev > aw_raw_ev + 0.05
        and aw_cal_ev > 0.05
    )
    ev_beats_p64 = aw_cal_ev is not None and p64_raw_ev is not None and aw_cal_ev > p64_raw_ev + 0.05

    # Cross-class: check A_w1 calibrated vs its own raw (and vs P64 raw if available)
    stage = ind.get("stage", {})
    harm = False
    harm_detail = {}
    cross = _f(stage, "A_w1", "cross_class_delta_vs_raw") or {}
    for sk in ("A1", "B1", "D"):
        dlt = _f(cross, sk, "delta")
        if dlt is not None and dlt < -0.05:
            harm = True
            harm_detail[sk] = dlt

    adv_improved = bool(_f(adv, "comparison", "cal_improves_gap_vs_A_raw")) and bool(
        _f(adv, "comparison", "cal_beats_P64_gap")
    )
    # softer: cal gap > P64 gap
    adv_ok = bool(_f(adv, "comparison", "cal_beats_P64_gap"))

    sw_dis = _f(
        alias, "p74_privileged_sidewall", "aliasing_evidence", "sidewall_disagreement_among_obs_neighbors"
    )
    sw_cond = _f(
        alias, "p74_privileged_sidewall", "aliasing_evidence", "sidewall_pos_neighbor_without_sidewall"
    )
    strong_alias = bool(_f(alias, "p74_privileged_sidewall", "aliasing_evidence", "strong_aliasing"))
    if sw_dis is not None and sw_dis > 0.15:
        strong_alias = True
    if sw_cond is not None and sw_cond > 0.5:
        strong_alias = True

    mod_pred = bool(_f(bott, "verdict_bits", "obs_moderately_predictive"))
    strong_pred = bool(_f(bott, "verdict_bits", "obs_strongly_predictive"))
    ceiling_r2 = _f(bott, "verdict_bits", "linear_probe_ceiling_r2")
    aw_c_ev = _f(bott, "components", "4_calibration_errors", "A_w1_cal_fc_C", "ev")
    p64_c_ev = _f(bott, "components", "4_calibration_errors", "P64_fc_C", "ev")

    if ev_restored and (ev_beats_p64 or aw_cal_ev > 0.1) and adv_ok and not harm:
        case = "A"
        reason = (
            f"Affine calibration restores A_w1 FC EV ({aw_raw_ev:.3f}→{aw_cal_ev:.3f}) "
            "and improves advantage diagnostics without major cross-class regression. "
            "A tightly controlled Actor PPO pilot may be considered — not auto-started."
        )
        recommend_ppo = True
        retain_baseline = False
    elif ev_restored and not adv_ok:
        case = "B"
        reason = (
            f"Calibration improves FC EV ({aw_raw_ev:.3f}→{aw_cal_ev:.3f}) but advantage quality "
            "does not — do not start Actor PPO; investigate credit assignment."
        )
        recommend_ppo = False
        retain_baseline = True
    elif (not ev_restored) and strong_alias:
        case = "C"
        reason = (
            f"Post-hoc affine calibration fails to restore useful overall FC EV "
            f"(A_w1 raw={aw_raw_ev:.3f}→cal={aw_cal_ev:.3f}; P64 cal={p64_cal_ev:.3f}). "
            f"Rare-positive conditional aliasing is strong: "
            f"{sw_cond:.0%} of obs-neighbors of correct-sidewall=1 states lack sidewall "
            f"(unconditional disagreement={sw_dis} is base-rate masked). "
            f"Note: C-only A_w1 FC EV≈{aw_c_ev} shows some predictability remains, "
            "but hidden geometry still aliases deployable neighborhoods. "
            "Retain P64 Critic + P70 V2; stop Critic-calibration; "
            "recommend a sensing/observability feasibility study before new learning."
        )
        recommend_ppo = False
        retain_baseline = True
    elif not ev_restored:
        case = "D"
        reason = (
            f"Calibration fails to restore FC EV (A_w1 raw={aw_raw_ev}, cal={aw_cal_ev}) "
            f"without strong aliasing evidence (sidewall NN disagreement={sw_dis}, "
            f"strong_alias={strong_alias}). Do not claim partial observability as established; "
            f"investigate return stochasticity (mc_std, linear R²={ceiling_r2}) and target quality. "
            "Retain P64 Critic + P70 V2 as production baseline."
        )
        recommend_ppo = False
        retain_baseline = True
    else:
        case = "D"
        reason = "Ambiguous partial improvements; retain baseline; no Actor PPO."
        recommend_ppo = False
        retain_baseline = True

    # Why corr↑ EV↓
    paradox = scale.get("paradox_A_w1_vs_P64") or {}
    a_sel = aw.get("a")
    answers = {
        "1_corr_up_ev_down": {
            "mechanism": (
                "EV=1-Var(y-pred)/Var(y) is invariant to additive bias but sensitive to scale/slope. "
                "A_w1 increased Corr while pred_var/y_var and residual magnitude worsened → EV fell."
            ),
            "P64_fc": paradox.get("P64_fc"),
            "A_w1_fc": paradox.get("A_w1_fc"),
            "selected_affine_a_b": {"a": a_sel, "b": aw.get("b"), "method": aw.get("selected")},
        },
        "2_affine_restores_ev": {
            "A_w1_raw_ev": aw_raw_ev,
            "A_w1_cal_ev": aw_cal_ev,
            "P64_raw_ev": p64_raw_ev,
            "P64_cal_ev": p64_cal_ev,
            "restored": bool(ev_restored),
        },
        "3_advantage_improved": {
            "comparison": adv.get("comparison"),
            "improved": bool(adv_improved or adv_ok),
        },
        "4_fc_failures_predictable": {
            "pre_opposing_auroc": _f(bott, "components", "2_hidden_state_ambiguity", "pre_opposing_auroc_at_first_fc"),
            "moderately": mod_pred,
            "strongly": strong_pred,
            "linear_ceiling_r2": ceiling_r2,
        },
        "5_aliasing_evidence": {
            "strong": strong_alias,
            "sidewall_nn_disagreement": sw_dis,
            "sidewall_pos_neighbor_without_sidewall": sw_cond,
            "p75_outcome_disagreement": _f(alias, "synthesis", "p75_outcome_disagreement"),
            "note": (
                "Unconditional sidewall disagreement can be base-rate masked; "
                "also report rare-positive conditional neighborhood purity."
            ),
        },
        "6_primary_limitation": {
            "A_w1_overall_fc_ev_still_negative": aw_cal_ev,
            "A_w1_C_only_fc_ev": aw_c_ev,
            "P64_C_only_fc_ev": p64_c_ev,
            "calibration": bool(not ev_restored),
            "stochasticity": bool(ceiling_r2 is not None and ceiling_r2 < 0.25),
            "observability_aliasing": strong_alias,
            "scale_mismatch_A_w1": _f(scale, "A_w1", "raw_fc", "pred_var_over_y_var"),
            "ranking": (
                "observability_aliasing"
                if strong_alias and not ev_restored
                else (
                    "mixed_class_value_optimization_and_scale"
                    if (aw_c_ev is not None and aw_c_ev > 0.2 and (aw_cal_ev is None or aw_cal_ev < 0.05))
                    else (
                        "stochasticity"
                        if ceiling_r2 is not None and ceiling_r2 < 0.25
                        else "calibration_and_optimization"
                    )
                )
            ),
            "note": (
                "If C-only FC EV is healthy while overall FC EV is not, the failure is not pure "
                "unobservability — cross-class / scale / optimization dominate."
            ),
        },

        "7_actor_learning_justified": recommend_ppo,
        "8_retain_p64_v2": retain_baseline,
    }

    decision = {
        "case": case,
        "recommend_actor_ppo_pilot": recommend_ppo,
        "auto_start_ppo": False,
        "retain_p64_critic_p70_v2": retain_baseline,
        "stop_critic_calibration_direction": case in ("C", "D") or not ev_restored,
        "recommend_sensing_study": case == "C",
        "reason": reason,
        "metrics": {
            "aw_raw_ev": aw_raw_ev,
            "aw_cal_ev": aw_cal_ev,
            "p64_raw_ev": p64_raw_ev,
            "p64_cal_ev": p64_cal_ev,
            "harm": harm,
            "harm_detail": harm_detail,
            "strong_alias": strong_alias,
            "sw_disagreement": sw_dis,
        },
        "answers": answers,
    }

    (OUT / "P77_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "P77_DECISION.md").write_text(
        f"# P77 Decision\n\n**Case {case}** — Actor PPO pilot: **{recommend_ppo}** "
        f"(auto-start: False). Retain P64+V2: **{retain_baseline}**\n\n{reason}\n\n```json\n"
        + json.dumps(decision, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.md").write_text(
        f"# P76 Final Report — Post-hoc Value Calibration & Observability Audit\n\n"
        f"**Case {case}**. No NN training; no Actor PPO; production baseline unchanged.\n\n"
        f"{reason}\n\n## Final answers\n"
        + "\n".join(f"### {k}\n```json\n{json.dumps(v, indent=2)}\n```\n" for k, v in answers.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case": case, "recommend_ppo": recommend_ppo, "retain": retain_baseline}, indent=2), flush=True)
    return decision


if __name__ == "__main__":
    decide()
