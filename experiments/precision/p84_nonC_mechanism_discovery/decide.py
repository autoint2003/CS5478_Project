"""Phase 9 — Case A–E decision."""

from __future__ import annotations

import json

from experiments.precision.p84_nonC_mechanism_discovery.paths import GATE_MIN_COUNT, OUT


def decide(audit, tax, gate, sig, feas, msp) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    gate_ok = bool(gate.get("gate_passed"))
    top = gate.get("largest_mode") or {}
    n_top = top.get("n_failed_ics") or 0
    amb = gate.get("ambiguity_sensitivity") or {}

    sig_skipped = bool(sig.get("skipped")) if sig else True
    detectable = False
    if not sig_skipped:
        detectable = bool((sig.get("signature") or sig).get("actionable"))

    recoverable = bool((feas.get("claims") or {}).get("C_recoverable", {}).get("holds"))
    data_ok = bool(audit.get("report", audit).get("checkpoint_hashes", {}).get("match_expected", True))
    if isinstance(audit, dict) and "report" in audit:
        data_ok = audit["report"]["checkpoint_hashes"]["match_expected"] and tax.get("exclusive_sum_equals_26")

    if not data_ok or not tax.get("exclusive_sum_equals_26"):
        case = "E"
        reason = (
            "Evidence invalid or incomplete: checkpoint/panel integrity or exclusive label sum failed. "
            "Do not force a positive decision. Minimum repair only — not a broad new campaign."
        )
        p85 = "minimum_data_repair_only"
    elif not gate_ok:
        case = "D"
        reason = (
            f"No exclusive B1 failure mode reaches the 40% gate "
            f"(largest={top.get('mode')} with {n_top}/26 < {GATE_MIN_COUNT}). "
            "Stop the optimization direction. Consolidate the existing baseline and research results."
        )
        p85 = "stop_consolidate_baseline"
    elif gate_ok and not detectable:
        case = "B"
        reason = (
            f"Dominant mode {top.get('mode')} passes concentration but no useful online signature "
            "exists from causal observations with detection lead time. Do not modify the controller."
        )
        p85 = "stop_no_detectable_signature"
    elif gate_ok and detectable and not recoverable:
        case = "C"
        reason = (
            "Dominant mechanism has a candidate online signature, but recoverability is unsupported "
            "(no credible B1 intervention evidence; P73/P81 warn against overlay revival). Stop unless "
            "a tiny existing-state CF could resolve one named uncertainty — not authorized automatically."
        )
        p85 = "stop_or_tiny_cf_only_if_named"
    elif gate_ok and detectable and recoverable:
        case = "A"
        reason = (
            "Dominant, detectable, and recoverability-supported B1 failure — recommend one bounded "
            "P85 causal validation. Do not authorize Actor training or deployment automatically."
        )
        p85 = "bounded_causal_validation"
    else:
        case = "E"
        reason = "Unclassified evidence state."
        p85 = "minimum_data_repair_only"

    answers = {
        "1_largest_exclusive_mode": top.get("mode"),
        "2_at_least_11_of_26": n_top >= GATE_MIN_COUNT,
        "3_robust_to_ambiguous_labeling": amb.get("still_passes_if_ambiguous_removed") if gate_ok else (
            n_top < GATE_MIN_COUNT  # concentration fails regardless of ambiguity
        ),
        "4_detectable_before_irreversible_failure": detectable,
        "5_signature_from_causal_obs": detectable and not sig_skipped,
        "6_false_alarms_on_success": None if sig_skipped else (sig.get("signature") or sig),
        "7_generalizes_beyond_b1_examples": None if sig_skipped else sig.get("cross"),
        "8_recoverability_credible": recoverable,
        "9_max_msp_mass": msp.get("raw_msp_failure_mass_upper_bound"),
        "10_p85_should": p85,
    }

    decision = {
        "case": case,
        "reason": reason,
        "p85": p85,
        "authorize_actor_training": False,
        "authorize_overlays": False,
        "controller_modification": False,
        "retain": "P64 Actor + P64 Critic + P70 V2",
        "one_defensible_intervention_hypothesis_or_stop": (
            feas.get("intervention_hypothesis") if case == "A" else "STOP optimization"
        ),
        "answers": answers,
        "concentration": {
            "largest": top,
            "gate_passed": gate_ok,
            "ambiguity": amb,
        },
    }

    (OUT / "P85_DECISION.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    (OUT / "P85_DECISION.md").write_text(
        f"# P85 Decision (from P84)\n\n**Case {case}**\n\n{reason}\n\n"
        f"- P85: `{p85}`\n"
        f"- Actor training: **No**\n"
        f"- Overlays / controller mods: **No**\n"
        f"- Retain: P64 Actor + P64 Critic + P70 V2\n\n"
        "## Final answers\n\n```json\n"
        + json.dumps(answers, indent=2, default=str)
        + "\n```\n",
        encoding="utf-8",
    )

    final = {
        "title": "P84 Non-C Mechanism Discovery Diagnostic",
        "case": case,
        "reason": reason,
        "p85": p85,
        "recommendation": "stop" if case in ("B", "C", "D", "E") and case != "A" else p85,
        "constraints_honored": [
            "No Actor/Critic training",
            "No PPO/GRPO/BC/DAgger",
            "No new sensing/obs/M111/reward changes",
            "No deployed overlays / controller modifications",
            "No final-test evaluation",
            "No automatic P85 execution",
            "Retained P64 + P70 V2",
            "No additional broad baseline collection",
        ],
    }
    (OUT / "FINAL_REPORT.md").write_text(
        f"# P84 Final Report\n\n## Decision: Case {case}\n\n{reason}\n\n"
        f"**Recommendation:** `{final['recommendation']}`\n\n"
        "## Answers\n\n```json\n"
        + json.dumps(answers, indent=2, default=str)
        + "\n```\n\n## Concentration\n\n```json\n"
        + json.dumps(decision["concentration"], indent=2)
        + "\n```\n\n## Constraints\n\n"
        + "\n".join(f"- {c}" for c in final["constraints_honored"])
        + "\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"case": case, "p85": p85, "top": top.get("mode"), "n_top": n_top}, indent=2), flush=True)
    return decision
