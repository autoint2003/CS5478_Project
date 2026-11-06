"""Phases 8–9 — Case A–E decision and single P84 recommendation."""

from __future__ import annotations

import json

from experiments.precision.p83_recoverable_failure_mass.paths import OUT, N_STAGE1, N_STAGE2


def decide(panel, funnel, attr, mass, recov, scenarios, sens, *, stage: int = 1) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fm = mass["comparisons"]
    c_cf = fm["A_C_contact_formation_mass"]
    nonc = fm["B_combined_A1B1D_any_failure_mass"]
    per = fm["C_individual_nonC_any_failure_mass"]
    a1, b1, d = per["A1"], per["B1"], per["D"]
    total_nonc = a1 + b1 + d

    c_sc = scenarios["scenarios"]["c_contact_formation"]["delta_msp"]
    nonc_sc = scenarios["scenarios"]["combined_nonC_assumed"]["delta_msp"]
    pref = sens.get("assumption_preference_counts") or {}
    flips = (sens.get("leave_one_ic_out_mass_order_flips") or {}).get("flip_rate", 0)

    # Evidence-supported recoverable: C has bounded scenarios; non-C conversion unknown
    nonc_unknown = True
    c_cons = c_sc.get("conservative")
    c_opt = c_sc.get("optimistic")

    # Dominance checks on *raw* failure mass
    nonc_dominates_raw = nonc > c_cf * 1.25
    c_dominates_raw = c_cf > nonc * 1.25
    single_nonc_share = {k: (v / total_nonc if total_nonc else 0) for k, v in per.items()}
    single_dom = max(single_nonc_share, key=single_nonc_share.get) if single_nonc_share else None
    single_dom_frac = single_nonc_share.get(single_dom, 0) if single_dom else 0

    # Case selection per Phase 8
    if c_dominates_raw and c_cons is not None and c_cons > 0 and pref.get("C", 0) >= pref.get("nonC", 0):
        case = "A"
        reason = (
            "C contact-formation failure mass robustly exceeds non-C alternatives under sampling stability, "
            "with a plausible recoverable MSP scenario after accounting for regression. "
            "Next experiment must explicitly avoid P73/P81 transfer failures (no fixed-token/sequence overlays)."
        )
        p84 = "c_contact_formation_non_overlay_diagnostic"
    elif single_dom_frac >= 0.50 and nonc_dominates_raw:
        case = "B"
        reason = (
            f"One non-C class ({single_dom}) contributes ≥50% of non-C failure mass "
            f"(share={single_dom_frac:.2f}) and combined non-C mass exceeds C contact-formation mass. "
            "Recommend one class-specific diagnostic — no training."
        )
        p84 = f"{single_dom}_specific_failure_diagnostic"
    elif nonc_dominates_raw and single_dom_frac < 0.50:
        case = "C"
        reason = (
            "Combined A1/B1/D failure mass dominates C contact-formation mass across bootstrap/LOO, "
            "but no single non-C class reaches 50% of that mass and recoverability remains unknown. "
            "Recommend a narrowly scoped non-C diagnostic that does not assume a shared mechanism."
        )
        p84 = "nonC_mechanism_discovery_diagnostic"
    elif stage == 1 and abs(nonc - c_cf) / max(nonc, c_cf, 1e-9) < 0.25:
        # Still close — may need stage 2 OR declare D
        case = "D"
        reason = (
            "Expanded panel still cannot reliably distinguish recoverable opportunities: "
            f"C_cf_mass={c_cf:.3f}, nonC_any_mass={nonc:.3f}; assumption sensitivity prefers "
            f"{dict(pref)}; LOO flip_rate={flips:.3f}. "
            "Do not expand broad baseline diagnostics indefinitely."
        )
        p84 = "lowest_cost_causal_probe_or_stop"
    else:
        # Default: nonC larger raw but unknown recoverability → Case C or D
        if nonc > c_cf:
            case = "C"
            reason = (
                "Non-C failure mass is larger, but recoverability is unknown; C has higher causal evidence quality "
                "yet negative transfer risk. Narrow non-C mechanism discovery is the priority diagnostic."
            )
            p84 = "nonC_mechanism_discovery_diagnostic"
        else:
            case = "E"
            reason = (
                "No credible recoverable overall-MSP opportunity remains after accounting for P73/P81 regressions "
                "and unknown non-C recoverability. Retain baseline and consolidate final project results."
            )
            p84 = "stop_consolidate"

    # Refine: if nonC unknown recoverability and C scenarios not robustly > 0 after regression → prefer C case over A
    if case == "A" and (c_cons is None or c_cons <= 0):
        case = "D"
        p84 = "lowest_cost_causal_probe_or_stop"
        reason = "C scenario conservative ΔMSP ≤ 0 after regression penalty; do not claim C dominance."

    # If Case D at stage1 and target < stage2, recommend stage2 only if within budget - but user said
    # stage2 only if unresolved. We'll set need_stage2 flag.
    need_stage2 = bool(case == "D" and stage == 1)

    # Build P84 recommendation block
    if p84 == "nonC_mechanism_discovery_diagnostic":
        proposal = {
            "hypothesis": (
                "A1/B1/D failures are not a single shared mechanism; the largest non-C gap (by failure mass) "
                "has a dominant earliest-decisive mode that can be identified observationally and then tested "
                "with one matched full-episode contrast against frozen P64+V2 — without overlays."
            ),
            "supporting_evidence": [
                f"P83 failure mass: nonC_any={nonc:.3f} > C_cf={c_cf:.3f}",
                f"Class masses A1={a1:.3f}, B1={b1:.3f}, D={d:.3f}",
                "P81: C-local interventions can destroy overall MSP",
            ],
            "contradictory_evidence": [
                "Non-C recoverability unknown (evidence class C only)",
                "P70 V2 already addresses some geometry/contact issues globally",
            ],
            "expected_overall_msp_opportunity": "Unknown point estimate; raw non-C failure mass "
            f"{nonc:.3f}; assumed-conversion scenarios only (see MSP_OPPORTUNITY_SCENARIOS)",
            "cross_class_risk": "Must gate any later fix on A/B/C/D MSP; no class-local promotion",
            "smallest_useful_experiment": (
                "P84 diagnostic: for the top non-C class by FailureMass, chart exclusive earliest-decisive "
                "modes with ≥32 ICs; propose one falsifiable non-overlay intervention hypothesis; "
                "optional matched baseline vs. no-intervention variance check only."
            ),
            "success_criterion": (
                "Identify one exclusive failure mode contributing ≥40% of that class's failures "
                "with a deployable observational signature — else abort to stop/consolidate."
            ),
            "abort_criterion": (
                "If no mode ≥40% or signature requires privileged info / overlay / training → stop optimization."
            ),
        }
    elif p84.endswith("_specific_failure_diagnostic"):
        sk = single_dom
        proposal = {
            "hypothesis": f"{sk} has a dominant exclusive failure mode whose reduction would raise equal-weight MSP more than residual C-formation work, without A/B/D regression.",
            "supporting_evidence": [f"{sk} failure mass share of non-C = {single_dom_frac:.2f}", f"mass={per[sk]:.3f}"],
            "contradictory_evidence": ["Recoverability unknown", "No prior causal intervention for this class"],
            "expected_overall_msp_opportunity": "Unknown; upper-bounded by failure mass " f"{per[sk]:.3f}",
            "cross_class_risk": "High if intervention is not class-gated; require MSP primary",
            "smallest_useful_experiment": f"P84 {sk}-only failure-mode diagnostic (no training/overlays)",
            "success_criterion": f"One mode ≥50% of {sk} failures with deployable signature",
            "abort_criterion": "No dominant mode / requires overlay or training → Case E stop",
        }
    elif p84 == "c_contact_formation_non_overlay_diagnostic":
        proposal = {
            "hypothesis": "Residual C FC→opposing/e≤9 failures remain after V2 and can be reduced by a non-overlay diagnostic measurement plan that does not change Actor actions online.",
            "supporting_evidence": ["P72 funnel", f"C_cf mass={c_cf:.3f}", "P79 controllability (B)"],
            "contradictory_evidence": ["P73/P81 transfer failures", "Conservative ΔMSP may be ≤0 after regression"],
            "expected_overall_msp_opportunity": {
                "conservative": c_cons,
                "optimistic": c_opt,
                "note": "scenario only",
            },
            "cross_class_risk": "Explicitly avoid tok56/sequence overlays; abort if any A/B/D probe regresses",
            "smallest_useful_experiment": "Measurement-only C funnel expansion + privileged-free signature search",
            "success_criterion": "Deployable signature predicts opposing/e9 with held-out AUROC≥0.7 without privileged labels as inputs",
            "abort_criterion": "No signature / would require overlay or training → stop",
        }
    elif p84 == "stop_consolidate":
        proposal = {
            "hypothesis": None,
            "supporting_evidence": ["P73/P81 negative A-level transfer", "Unknown non-C recoverability"],
            "contradictory_evidence": [],
            "expected_overall_msp_opportunity": 0.0,
            "cross_class_risk": "N/A",
            "smallest_useful_experiment": "None — consolidate P64+V2 final results",
            "success_criterion": "N/A",
            "abort_criterion": "N/A",
        }
    else:
        proposal = {
            "hypothesis": (
                "A single lowest-cost causal contrast can resolve whether non-C failures are actionable: "
                "compare variance of SUCCESS under repeated seeds on top non-C class vs C, without policy change."
            ),
            "supporting_evidence": ["P83 indistinguishability under assumptions"],
            "contradictory_evidence": ["May not identify mechanism"],
            "expected_overall_msp_opportunity": "0 unless probe reveals actionable mode",
            "cross_class_risk": "None (no policy change)",
            "smallest_useful_experiment": (
                "Optional: 2 seeds × 16 ICs on top non-C class to estimate aleatoric SUCCESS variance; "
                "if seed-flip rate high and modes diffuse → stop (Case E). "
                "Do NOT expand to 48/class baseline by default."
            ),
            "success_criterion": "Clear dominant mode OR decision to stop",
            "abort_criterion": "If probe inconclusive → stop further optimization (Case E)",
        }

    decision = {
        "case": case,
        "p84": p84,
        "need_stage2_to_48": need_stage2,
        "reason": reason,
        "authorize_training": False,
        "authorize_overlays": False,
        "retain": "P64 Actor + P64 Critic + P70 V2",
        "primary_promotion_metric": "Equal-weight A1/B1/C/D SUCCESS_RELAXED MSP",
        "mass_snapshot": {
            "c_contact_formation": c_cf,
            "nonC_any": nonc,
            "A1": a1,
            "B1": b1,
            "D": d,
            "single_nonc_shares": single_nonc_share,
        },
        "proposal": proposal,
        "panel": {
            "n_per_class": panel.get("n_per_class"),
            "msp": panel.get("msp_equal_weight"),
            "stage": stage,
            "stage1_target": N_STAGE1,
            "stage2_cap": N_STAGE2,
        },
    }

    (OUT / "P84_DECISION.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    (OUT / "P84_DECISION.md").write_text(
        f"# P84 Decision (from P83)\n\n**Case {case}**\n\n{reason}\n\n"
        f"- P84: `{p84}`\n"
        f"- Training: **No**\n- Overlays: **No**\n"
        f"- Retain: P64 Actor + P64 Critic + P70 V2\n"
        f"- Need stage-2 (48/class): **{need_stage2}**\n\n"
        "## Recommendation details\n\n```json\n"
        + json.dumps(proposal, indent=2, default=str)
        + "\n```\n\n## Mass snapshot\n\n```json\n"
        + json.dumps(decision["mass_snapshot"], indent=2)
        + "\n```\n",
        encoding="utf-8",
    )

    final = {
        "title": "P83 Cross-Class Recoverable Failure Mass",
        "case": case,
        "p84": p84,
        "reason": reason,
        "one_recommendation": p84,
        "constraints_honored": [
            "No Actor/Critic training",
            "No PPO/GRPO/BC/DAgger",
            "No new overlays/sensing/reward/obs/M111 changes",
            "No final-test evaluation",
            "No automatic P84 execution",
            "Retained frozen P64 + P70 V2",
        ],
    }
    (OUT / "FINAL_REPORT.md").write_text(
        f"# P83 Final Report\n\n## Decision: Case {case}\n\n{reason}\n\n"
        f"**P84 recommendation:** `{p84}`\n\n"
        "## Key numbers\n\n```json\n"
        + json.dumps(
            {
                "panel_msp": panel.get("msp_equal_weight"),
                "success_by_class": scenarios.get("success_by_class"),
                "failure_mass": decision["mass_snapshot"],
                "c_scenario_delta_msp": c_sc,
                "assumption_preferences": pref,
            },
            indent=2,
            default=str,
        )
        + "\n```\n\n## Constraints\n\n"
        + "\n".join(f"- {c}" for c in final["constraints_honored"])
        + "\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"case": case, "p84": p84, "need_stage2": need_stage2, "c_cf": c_cf, "nonc": nonc}, indent=2), flush=True)
    return decision
