"""Phase 7 — Case A–D decision."""

from __future__ import annotations

import json

from experiments.precision.p82_global_failure_budget.paths import OUT


def decide(freeze, budget, skills, matrix, opportunity, proposal) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    hint = proposal.get("case_hint") or "C"
    ranking = opportunity.get("ranking") or []
    scores = [r["score"] for r in ranking if r.get("score", -1) >= 0]

    # Consistency check vs evidence: if all high-EV directions rely on B/C only and A interventions failed
    a_negative = any(e["id"] in ("P81_fixed_token", "P73_overlay_transfer", "P72_overlay") and e["class"] == "D" for e in matrix["entries"])

    if hint == "A" and scores and scores[0] > 0.015:
        case = "A"
        reason = (
            f"One clear high-value bottleneck emerges: {proposal.get('direction')}. "
            "Recommend a narrowly scoped P83 diagnostic with measurable expected overall MSP benefit. "
            "No automatic training."
        )
        p83 = proposal.get("direction")
    elif hint == "B" or (len(scores) >= 2 and scores[0] < scores[1] * 1.35):
        case = "B"
        reason = (
            "Multiple bottlenecks have comparable overall-MSP value (notably C contact-formation vs non-C class gaps), "
            "while P73/P81 show C-local interventions can destroy MSP. "
            "Recommend the lowest-cost experiment that distinguishes their relative importance: "
            f"{proposal.get('direction')}. No training; no overlays."
        )
        p83 = proposal.get("direction")
    elif hint == "C" or not scores or scores[0] <= 0.005:
        case = "C"
        reason = (
            "No remaining direction has sufficient A-level evidence of overall-MSP upside after P70–P81. "
            "Stop further optimization experiments. Consolidate P64 Actor + P64 Critic + P70 V2 for final project results."
        )
        p83 = "stop_consolidate_baseline"
    else:
        case = "D"
        reason = (
            "Existing evidence is inconsistent across matched CF vs full-episode outcomes. "
            "Identify minimum validation to resolve inconsistency; do not launch a broad new search."
        )
        p83 = "minimum_validation_only"

    # Prefer B over A when proposal says distinguish — already handled
    # If Case A would revive closed interventions, downgrade
    if case == "A" and "token" in (p83 or ""):
        case = "C"
        p83 = "stop_consolidate_baseline"
        reason = "Blocked: would revive closed fixed-token path."

    answers = {
        "baseline": freeze.get("baseline"),
        "msp": opportunity.get("msp"),
        "class_gaps": opportunity.get("class_gaps"),
        "top_directions": ranking[:5],
        "recommended_p83": p83,
        "hypothesis": proposal.get("falsifiable_hypothesis"),
        "primary_metric": proposal.get("primary_metric"),
        "stop_criteria": proposal.get("stop_criteria"),
        "abd_retention": proposal.get("abd_retention"),
        "a_level_negatives_closed": a_negative,
    }

    decision = {
        "case": case,
        "reason": reason,
        "p83": p83,
        "authorize_training": False,
        "authorize_overlays": False,
        "retain": "P64 Actor + P64 Critic + P70 V2",
        "proposal": proposal,
        "answers": answers,
    }

    (OUT / "P83_DECISION.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    (OUT / "P83_DECISION.md").write_text(
        f"# P83 Decision (from P82)\n\n**Case {case}**\n\n{reason}\n\n"
        f"- P83: `{p83}`\n"
        f"- Training authorized: **No**\n"
        f"- Overlays authorized: **No**\n"
        f"- Retain: P64 Actor + P64 Critic + P70 V2\n\n"
        "## Proposed bounded experiment (not executed)\n\n```json\n"
        + json.dumps(proposal, indent=2, default=str)
        + "\n```\n",
        encoding="utf-8",
    )

    final = {
        "title": "P82 Global Failure Budget and Research Reprioritization",
        "case": case,
        "reason": reason,
        "p83": p83,
        "one_defensible_next_direction_or_stop": p83,
        "constraints_honored": [
            "No Actor/Critic training",
            "No PPO/GRPO/BC/DAgger",
            "No fixed-token or sequence overlays",
            "No reward/sensing/obs/M111 changes",
            "No final-test evaluation",
            "No automatic P83 execution",
            "Retained P64 + P70 V2",
        ],
    }
    (OUT / "FINAL_REPORT.md").write_text(
        f"# P82 Final Report\n\n## Decision: Case {case}\n\n{reason}\n\n"
        f"**Next:** `{p83}`\n\n"
        "## Snapshot\n\n```json\n"
        + json.dumps(
            {
                "msp": opportunity.get("msp"),
                "class_gaps": opportunity.get("class_gaps"),
                "ranking_top3": ranking[:3],
                "earliest_decisive": budget.get("earliest_decisive_pooled"),
                "transitions_C": budget.get("transitions_C"),
            },
            indent=2,
            default=str,
        )
        + "\n```\n\n## Constraints\n\n"
        + "\n".join(f"- {c}" for c in final["constraints_honored"])
        + "\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"case": case, "p83": p83, "msp": opportunity.get("msp")}, indent=2), flush=True)
    return decision
