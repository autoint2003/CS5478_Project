"""Phases 7–8 — intervention feasibility + MSP opportunity."""

from __future__ import annotations

import json

from experiments.precision.p84_nonC_mechanism_discovery.paths import OUT


def intervention_feasibility(gate: dict, sig: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    gate_ok = bool(gate.get("gate_passed"))
    sig_ok = bool((sig or {}).get("signature", {}).get("actionable")) if isinstance(sig, dict) and "signature" in (sig or {}) else bool((sig or {}).get("actionable"))

    # Normalize sig structure from analyze_signatures return
    if sig and sig.get("skipped"):
        sig_ok = False
        detectable = False
    else:
        detectable = bool((sig.get("signature") or sig).get("actionable")) if sig else False

    claims = {
        "A_frequent": {
            "holds": gate_ok,
            "note": "Phase 3 concentration gate",
        },
        "B_detectable_before_failure": {
            "holds": detectable,
            "note": "Requires gate pass + causal signature with lead time",
        },
        "C_recoverable": {
            "holds": False,
            "note": (
                "No B1-specific full-episode intervention evidence in P70–P83. "
                "P73/P81 show overlay/token interventions regress MSP. "
                "P79 controllability is C-FC matched CF — scope mismatch for B1."
            ),
            "evidence_reviewed": [
                "P70 V2: pre-contact; not B1-mode-specific",
                "P73/P81: negative closed-loop transfer for fixed interventions",
                "P79: matched CF action effects on C FC — not B1 taxonomy",
                "P82/P83: observational failure mass only for B1",
            ],
        },
    }

    report = {
        "claims": claims,
        "passing_A_and_B_does_not_establish_C": True,
        "revive_fixed_token_overlays": False,
        "intervention_hypothesis": None,
        "reason_no_hypothesis": (
            None
            if (gate_ok and detectable)
            else (
                "No dominant B1 mode (≥40%) — cannot define a targeted intervention hypothesis."
                if not gate_ok
                else "Dominant mode lacks actionable online signature."
            )
        ),
    }
    if gate_ok and detectable:
        report["intervention_hypothesis"] = {
            "statement": "TBD only if Case A — not auto-executed",
            "requires_separate_p85": True,
        }

    (OUT / "INTERVENTION_FEASIBILITY.md").write_text(
        "# Intervention Feasibility (P84 Phase 7)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report


def msp_opportunity(gate: dict, tax: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    n_b1 = tax["n_b1"]
    top = (gate.get("largest_mode") or {})
    n_aff = top.get("n_failed_ics") or 0
    raw_mass = 0.25 * n_aff / n_b1 if n_b1 else 0.0
    gate_ok = bool(gate.get("gate_passed"))

    report = {
        "largest_mode": top.get("mode"),
        "affected_b1_ics": n_aff,
        "raw_msp_failure_mass_upper_bound": raw_mass,
        "definition": "0.25 × affected_B1_ICs / 32 — upper bound if perfectly fixed with zero regression",
        "not_expected_improvement": True,
        "detectable_failure_mass": None if not gate_ok else "see signature coverage × raw_mass (gate failed in this run → N/A)",
        "potentially_recoverable_failure_mass": {
            "value": None,
            "reason": "No conversion probability manufactured; recoverability unsupported for B1 modes",
        },
        "cross_class_regression_risk": "high if any overlay-like fix (P73/P81)",
        "ultimate_metric": "Equal-weight A1/B1/C/D SUCCESS_RELAXED MSP",
        "gate_passed": gate_ok,
    }
    (OUT / "MSP_OPPORTUNITY.md").write_text(
        "# MSP Opportunity (P84 Phase 8)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report
