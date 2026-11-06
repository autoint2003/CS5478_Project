"""Phases 4–6 — evidence matrix, MSP opportunity, P83 direction."""

from __future__ import annotations

import json

from experiments.precision.p82_global_failure_budget.paths import OUT


def evidence_matrix() -> dict:
    """Reconcile P70–P81 without promoting B/C to A."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "P70_V2",
            "claim": "Deployable adaptive pre-contact V2 improves full-episode SUCCESS/R3 vs P64 alone",
            "class": "A",
            "note": "Closed-loop full-episode held-out gains; retained as baseline overlay",
        },
        {
            "id": "P71_postcontact",
            "claim": "Additional post-contact alignment overlays improve overall MSP",
            "class": "D",
            "note": "Variants often regress SUCCESS; no promotion",
        },
        {
            "id": "P72_funnel",
            "claim": "Largest C drop is FC→opposing bilateral; geometry/e_pos bottleneck",
            "class": "C",
            "note": "Observational funnel + success/fail geometry contrast; overlays failed transfer (not A)",
        },
        {
            "id": "P72_overlay",
            "claim": "Fixed post-contact sequence overlays improve full episodes",
            "class": "D",
            "note": "CF local gains; held-out full-episode regression",
        },
        {
            "id": "P73_overlay_transfer",
            "claim": "Event-conditioned FC overlays transfer to full episodes",
            "class": "D",
            "note": "Matched CF ≠ full-episode generalization",
        },
        {
            "id": "P74_critic_readiness",
            "claim": "Actor owns post-contact decisions; Critic training justified now",
            "class": "D",
            "note": "Ownership observationally true; readiness Case led to calibration attempts that failed",
        },
        {
            "id": "P75_critic_calib",
            "claim": "Critic recalibration restores reliable FC advantages for PPO",
            "class": "D",
            "note": "Calibration failed; keep P64 Critic",
        },
        {
            "id": "P76_observability",
            "claim": "Missing sensing / aliasing blocks Critic; need new sensors",
            "class": "C",
            "note": "Observational/alias analysis; P77 revised sensing feasibility",
        },
        {
            "id": "P77_sensing",
            "claim": "Engineered features from existing 39D improve deployable sidewall observability",
            "class": "B",
            "note": "Diagnostic probes; not full-episode Actor intervention; Ft invalid",
        },
        {
            "id": "P78_representation",
            "claim": "Engineered features improve action-relevant ranking",
            "class": "D",
            "note": "Insufficient matched CF; Case C",
        },
        {
            "id": "P79_controllability",
            "claim": "M111 actions change matched FC outcomes; history ranks actions; eng ≯ history",
            "class": "B",
            "note": "Matched CF action ranking; Case B — no Actor obs change",
        },
        {
            "id": "P80_static_vs_cond",
            "claim": "History-conditioned selection beats best fixed token",
            "class": "D",
            "note": "tok_56 matches ranker; conditional adds no held-out utility",
        },
        {
            "id": "P81_fixed_token",
            "claim": "Static tok_56 FC substitution improves overall MSP",
            "class": "D",
            "note": "Matched C Δ>0 (B) but full-episode MSP collapses A/B/D (A-level contradiction)",
        },
        {
            "id": "P81_underselect_56",
            "claim": "Low p(tok_56) proves defective Actor requiring correction",
            "class": "D",
            "note": "Underselection exists observationally but intervention harms; do not reinterpret as defect proof",
        },
    ]
    report = {
        "legend": {
            "A": "Supported by full-episode intervention",
            "B": "Supported by matched-state counterfactuals only",
            "C": "Observational association only",
            "D": "Contradicted or inconclusive",
        },
        "entries": rows,
        "rule": "Do not promote B/C evidence to A. P81 closed the fixed-token path at evidence class A (negative).",
    }
    (OUT / "P70_P81_EVIDENCE_MATRIX.md").write_text(
        "# P70–P81 Evidence Matrix (P82 Phase 4)\n\n"
        "| ID | Class | Claim |\n|---|:---:|---|\n"
        + "\n".join(f"| {e['id']} | **{e['class']}** | {e['claim']} |" for e in rows)
        + "\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return report


def msp_opportunity(budget: dict, skills: dict) -> dict:
    """Estimate contribution of failure categories to overall MSP loss."""
    OUT.mkdir(parents=True, exist_ok=True)
    succ = budget.get("success_by_class") or {}
    msp = budget.get("msp_equal_class_weight")
    # Per-class loss to 1.0
    class_gaps = {sk: 1.0 - float(v) for sk, v in succ.items() if v is not None}
    # Equal weight: each class contributes 0.25 * gap to MSP gap
    msp_gap = None if msp is None else 1.0 - msp

    decisive = budget.get("earliest_decisive_share_of_failures") or {}
    # Map decisive failures to research buckets
    bucket_map = {
        "no_first_contact": "precontact_approach",
        "no_opposing_after_fc": "c_contact_formation",
        "no_min_pad4": "c_contact_formation",
        "no_e_pos_le9": "c_contact_formation",
        "no_frame_ok": "c_postcontact_stabilization",
        "no_r3_ready": "handoff_robustness",
        "post_handoff_or_terminal_fail": "handoff_robustness",
        "timeout": "timeout_other",
        "other_terminal": "timeout_other",
    }
    # Frequency among failures (pooled) — convert to rough MSP share via class gaps
    # Scenario bounds on recoverability
    scenarios = {
        "c_contact_formation": {
            "failure_frequency_proxy": float(
                decisive.get("no_opposing_after_fc", 0)
                + decisive.get("no_min_pad4", 0)
                + decisive.get("no_e_pos_le9", 0)
            ),
            "plausibly_recoverable_low": 0.10,
            "plausibly_recoverable_high": 0.35,
            "cross_class_regression_risk": "high (P73/P81)",
            "evidence_quality": "C funnel + B matched CF; A interventions failed",
            "experimental_cost": "high if training; medium if diagnostic CF only",
            "overall_msp_priority_note": "Large on C, but equal-weight MSP also limited by A/B/D gaps",
        },
        "c_postcontact_stabilization": {
            "failure_frequency_proxy": float(decisive.get("no_frame_ok", 0)),
            "plausibly_recoverable_low": 0.05,
            "plausibly_recoverable_high": 0.20,
            "cross_class_regression_risk": "medium–high",
            "evidence_quality": "C (P72 near-ready rare)",
            "experimental_cost": "medium",
        },
        "skill_transition_control": {
            "failure_frequency_proxy": None,
            "plausibly_recoverable_low": 0.00,
            "plausibly_recoverable_high": 0.15,
            "cross_class_regression_risk": "high (P72/P73/P81)",
            "evidence_quality": "C association; A interventions negative",
            "experimental_cost": "medium",
            "deprioritized": True,
            "reason": "Fixed skill / token interventions closed by P81",
        },
        "handoff_robustness": {
            "failure_frequency_proxy": float(
                decisive.get("no_r3_ready", 0) + decisive.get("post_handoff_or_terminal_fail", 0)
            ),
            "plausibly_recoverable_low": 0.05,
            "plausibly_recoverable_high": 0.25,
            "cross_class_regression_risk": "medium",
            "evidence_quality": "C; P72 says near-ready uncommon on C fails",
            "experimental_cost": "medium",
        },
        "other_class_msp": {
            "failure_frequency_proxy": None,
            "class_gaps": class_gaps,
            "plausibly_recoverable_low": 0.05,
            "plausibly_recoverable_high": 0.30,
            "cross_class_regression_risk": "depends on intervention locality",
            "evidence_quality": "A-level baseline rates from P82 panel; mechanism less mapped than C funnel",
            "experimental_cost": "medium diagnostic first",
            "note": "If A/B/D gaps ≥ C gap, overall MSP may favor non-C work",
        },
    }

    # Expected MSP impact bounds: 0.25 * class_gap_C * recoverability for C-formation
    # plus share of non-C gaps for other_class
    c_gap = class_gaps.get("C")
    for name, sc in scenarios.items():
        if name == "c_contact_formation" and c_gap is not None:
            sc["expected_msp_delta_low"] = 0.25 * c_gap * sc["plausibly_recoverable_low"]
            sc["expected_msp_delta_high"] = 0.25 * c_gap * sc["plausibly_recoverable_high"]
        elif name == "other_class_msp":
            gaps = [class_gaps[k] for k in ("A1", "B1", "D") if k in class_gaps]
            if gaps:
                # equal weight each class 0.25
                sc["expected_msp_delta_low"] = sum(0.25 * g * sc["plausibly_recoverable_low"] for g in gaps)
                sc["expected_msp_delta_high"] = sum(0.25 * g * sc["plausibly_recoverable_high"] for g in gaps)
        elif c_gap is not None and sc.get("failure_frequency_proxy") is not None:
            # scale by how often this is earliest fail among C-ish path — conservative use pooled share * C gap weight
            sc["expected_msp_delta_low"] = 0.25 * c_gap * sc["failure_frequency_proxy"] * sc["plausibly_recoverable_low"]
            sc["expected_msp_delta_high"] = 0.25 * c_gap * sc["failure_frequency_proxy"] * sc["plausibly_recoverable_high"]

    # Rank by midpoint expected MSP / risk penalty
    ranked = []
    for name, sc in scenarios.items():
        if sc.get("deprioritized"):
            score = -1.0
        else:
            lo = sc.get("expected_msp_delta_low") or 0.0
            hi = sc.get("expected_msp_delta_high") or 0.0
            mid = 0.5 * (lo + hi)
            risk = {"high (P73/P81)": 0.5, "medium–high": 0.65, "medium": 0.8, "depends on intervention locality": 0.75}.get(
                sc.get("cross_class_regression_risk"), 0.7
            )
            score = mid * risk
        ranked.append((score, name, sc))
    ranked.sort(key=lambda x: -x[0])

    report = {
        "msp": msp,
        "msp_gap_to_1": msp_gap,
        "class_gaps": class_gaps,
        "scenarios": scenarios,
        "ranking": [{"direction": n, "score": s, "expected_low": sc.get("expected_msp_delta_low"), "expected_high": sc.get("expected_msp_delta_high")} for s, n, sc in ranked],
        "warnings": [
            "Not every failed episode is recoverable.",
            "Do not rank solely by C-class improvement potential.",
            "Scenario recoverability bounds are explicit assumptions, not measured yields.",
        ],
    }
    (OUT / "OVERALL_MSP_OPPORTUNITY.md").write_text(
        "# Overall MSP Opportunity (P82 Phase 5)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report


def propose_p83(budget: dict, opportunity: dict, skills: dict) -> dict:
    ranking = opportunity.get("ranking") or []
    top = ranking[0]["direction"] if ranking else "none"
    second = ranking[1]["direction"] if len(ranking) > 1 else None
    class_gaps = opportunity.get("class_gaps") or {}
    c_gap = class_gaps.get("C")
    nonc_gap = sum(class_gaps.get(k, 0) for k in ("A1", "B1", "D")) / 3 if class_gaps else None

    # Decide case based on separation of top scores
    scores = [r["score"] for r in ranking if r["score"] >= 0]
    clear = bool(len(scores) >= 2 and scores[0] >= scores[1] * 1.35 and scores[0] > 0.01)
    comparable = bool(len(scores) >= 2 and scores[0] < scores[1] * 1.35 and scores[0] > 0.005)
    insufficient = bool(not scores or scores[0] <= 0.005)

    # Prefer distinguishing experiment when C formation vs other-class MSP are close
    if top == "other_class_msp" or (
        second == "other_class_msp"
        and ranking[0]["score"] > 0
        and ranking[1]["score"] > 0
        and ranking[0]["score"] < ranking[1]["score"] * 1.4
    ):
        direction = "distinguish_C_formation_vs_nonC_msp"
        case_hint = "B"
    elif top == "c_contact_formation" and clear:
        direction = "c_contact_formation_diagnostic"
        case_hint = "A"
    elif insufficient:
        direction = "stop"
        case_hint = "C"
    else:
        direction = top
        case_hint = "B" if comparable else "A"

    # Falsifiable hypothesis for the recommended next step
    if direction == "distinguish_C_formation_vs_nonC_msp":
        proposal = {
            "direction": direction,
            "falsifiable_hypothesis": (
                "On an IC-disjoint stratified A/B/C/D baseline panel, the equal-weight MSP gap "
                "attributable to non-C classes is at least as large as the C gap attributable to "
                "FC→opposing/e≤9 failures; therefore a C-only contact-formation intervention cannot "
                "be the unique highest-EV next step without first quantifying non-C recoverable mass."
            ),
            "supporting_evidence": [
                "P82 cross-class SUCCESS gaps (this panel)",
                "P72 C funnel: FC→opposing is largest C drop (observational)",
                "P81: C-local matched gains can destroy overall MSP",
            ],
            "contradictory_evidence": [
                "P70 V2 already lifted C/full SUCCESS via pre-contact geometry (A-level)",
                "C may still be largest single-class gap on some panels",
            ],
            "primary_metric": "Equal-weight A/B/C/D SUCCESS_RELAXED MSP gap decomposition (class gaps + earliest-decisive shares)",
            "bounded_experiment": (
                "P83 diagnostic-only: expand baseline failure budget to ≥30 ICs/class (no training, no overlays); "
                "report MSP opportunity with IC bootstrap; stop if non-C mid EV < 50% of C mid EV and C gap dominates."
            ),
            "stop_criteria": [
                "If one class gap explains ≥50% of MSP gap with recoverable earliest-decisive mass → Case A specialize",
                "If gaps remain within 20% relative and mechanisms unclear → consolidate/stop (Case C)",
                "No Actor/Critic training; no fixed-token overlays",
            ],
            "abd_retention": "Any later intervention must pre-register ≤2pp A/B/D SUCCESS drop and MSP primary",
        }
    elif direction == "c_contact_formation_diagnostic":
        proposal = {
            "direction": direction,
            "falsifiable_hypothesis": (
                "A non-overlay, non-training diagnostic that increases correct opposing / e≤9 rates "
                "on held-out C without changing Actor weights will raise equal-weight MSP only if "
                "A/B/D SUCCESS remain within retention budgets — else C formation is not overall-MSP-positive."
            ),
            "supporting_evidence": ["P72 funnel", "P79 action controllability (B)"],
            "contradictory_evidence": ["P73/P81 transfer failures", "P81 A/B/D regressions"],
            "primary_metric": "Equal-weight MSP (not C-only R3)",
            "bounded_experiment": "Diagnostic CF / measurement only; no training",
            "stop_criteria": ["No MSP lift with retention → stop C-formation push"],
            "abd_retention": "Hard gate on A/B/D",
        }
    else:
        proposal = {
            "direction": "stop_or_consolidate",
            "falsifiable_hypothesis": None,
            "bounded_experiment": "No further optimization experiment",
            "stop_criteria": ["Consolidate P64+V2 baseline for final project reporting"],
            "abd_retention": "N/A",
        }

    proposal["case_hint"] = case_hint
    proposal["top_ranked"] = ranking[:5]
    proposal["class_gaps"] = class_gaps
    return proposal
