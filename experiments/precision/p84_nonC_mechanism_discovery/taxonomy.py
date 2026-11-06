"""Phases 2–3 — B1 exclusive failure taxonomy + concentration gate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from experiments.precision.p84_nonC_mechanism_discovery.paths import GATE_FRAC, GATE_MIN_COUNT, OUT


# Human-readable mapping (not C-forced); exclusive labels already on rows
LABEL_MEANING = {
    "no_first_contact": "Failure to establish initial pad contact",
    "no_opposing_after_fc": "Failure to form opposing bilateral after first contact",
    "no_min_pad4": "Insufficient pad engagement (min_pad never ≥4) after opposing path",
    "no_e_pos_le9": "Failure to reach e_pos≤9 mm after pad engagement milestones",
    "no_frame_ok": "Frame-ok never achieved after geometry milestones",
    "no_r3_ready": "R3-ready never achieved after frame_ok path",
    "post_handoff_or_terminal_fail": "Handoff/R3 progressed but SUCCESS_RELAXED not achieved (post-handoff/terminal)",
    "timeout": "Timeout after otherwise meaningful progress",
    "other_terminal": "Other verified terminal failure",
}


def _refine_descriptor(r: dict) -> dict:
    """Trajectory-informed descriptors — do NOT merge exclusive labels to pass gate."""
    excl = r["earliest_decisive_failure"]
    desc = {
        "traj_id": r["traj_id"],
        "exclusive": excl,
        "meaning": LABEL_MEANING.get(excl, excl),
        "t_fc": r.get("t_fc"),
        "t_opp": r.get("t_opp"),
        "t_loss": r.get("t_loss"),
        "contact_loss_ever": r.get("contact_loss_ever"),
        "recatch_unstable": r.get("recatch_unstable"),
        "timeout": r.get("timeout"),
        "n_family_switches": r.get("n_family_switches"),
        "post_fc_families": r.get("post_fc_family_counts"),
        "grip_open_near_contact": r.get("grip_open_near_contact_steps"),
        "strong_approach_near_contact": r.get("strong_approach_near_contact_steps"),
        "ep_len": r.get("ep_len"),
        "milestones": {
            "fc": r["first_contact"],
            "opp": r["opposing_ever"],
            "pad4": r["min_pad_ge4"],
            "e9": r["e_pos_le9"],
            "frame": r["frame_ok"],
            "r3": r["r3_ready"],
            "handoff": r["handoff"],
        },
    }
    # Ambiguity flags (documentation only)
    ambiguous = []
    if excl == "no_e_pos_le9" and r.get("contact_loss_ever"):
        ambiguous.append("also_had_contact_loss_but_exclusive_is_e9_by_priority_rule")
    if excl == "no_min_pad4" and r.get("contact_loss_ever"):
        ambiguous.append("also_had_contact_loss_but_exclusive_is_pad4_by_priority_rule")
    if excl == "no_frame_ok" and not r.get("e_pos_le9"):
        ambiguous.append("inconsistent_milestone_if_priority_broken")  # shouldn't happen
    if r.get("timeout") and excl not in ("timeout", "post_handoff_or_terminal_fail"):
        ambiguous.append("timeout_co_occurs_with_earlier_milestone_miss")
    desc["ambiguity_flags"] = ambiguous
    desc["label_uncertainty"] = "medium" if ambiguous else "low"
    return desc


def build_taxonomy(rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    b1 = [r for r in rows if r["skill"] == "B1"]
    fails = [r for r in b1 if not r["SUCCESS_RELAXED"]]
    succs = [r for r in b1 if r["SUCCESS_RELAXED"]]

    exclusive = Counter(r["earliest_decisive_failure"] for r in fails)
    refined = [_refine_descriptor(r) for r in fails]
    succ_desc = [
        {
            "traj_id": r["traj_id"],
            "ep_len": r["ep_len"],
            "t_fc": r.get("t_fc"),
            "t_opp": r.get("t_opp"),
            "n_family_switches": r.get("n_family_switches"),
            "contact_loss_ever": r.get("contact_loss_ever"),
        }
        for r in succs
    ]

    # Secondary non-exclusive among fails
    secondary = {
        "contact_loss_ever": sum(r["contact_loss_ever"] for r in fails),
        "recatch_unstable": sum(r["recatch_unstable"] for r in fails),
        "timeout": sum(r["timeout"] for r in fails),
        "missed_e9_any": sum(1 for r in fails if not r["e_pos_le9"]),
        "missed_pad4_any": sum(1 for r in fails if not r["min_pad_ge4"]),
        "missed_opp_any": sum(1 for r in fails if not r["opposing_ever"]),
    }

    report = {
        "n_b1": len(b1),
        "n_success": len(succs),
        "n_fail": len(fails),
        "exclusive_counts": dict(exclusive),
        "exclusive_sum": sum(exclusive.values()),
        "exclusive_sum_equals_26": sum(exclusive.values()) == 26 and len(fails) == 26,
        "label_meanings": LABEL_MEANING,
        "priority_rule": (
            "success > no_fc > no_opposing > no_pad4 > no_e9 > no_frame > no_r3 > "
            "post_handoff_fail > timeout > other — same as P82/P83; B1 not forced into C-only categories"
        ),
        "secondary_nonexclusive": secondary,
        "failed_episode_details": refined,
        "success_episode_details": succ_desc,
        "note": (
            "Symptoms (e.g. timeout co-occurrence) are not reassigned as exclusive causes. "
            "Do not merge categories solely to pass the 40% gate."
        ),
    }
    (OUT / "B1_EXCLUSIVE_FAILURE_TAXONOMY.md").write_text(
        "# B1 Exclusive Failure Taxonomy (P84 Phase 2)\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return report


def concentration_gate(tax: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    n_fail = tax["n_fail"]
    counts = tax["exclusive_counts"]
    rows = []
    for mode, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        rows.append(
            {
                "mode": mode,
                "meaning": LABEL_MEANING.get(mode, mode),
                "n_failed_ics": n,
                "frac_of_26_failures": n / n_fail if n_fail else None,
                "frac_of_32_episodes": n / tax["n_b1"] if tax["n_b1"] else None,
                "msp_failure_mass_upper": 0.25 * n / tax["n_b1"],
                "passes_40pct_gate": n >= GATE_MIN_COUNT and (n / n_fail) >= GATE_FRAC - 1e-12,
            }
        )
    top = rows[0] if rows else None

    # Ambiguity sensitivity: if medium-uncertainty episodes in top mode were relabeled away
    top_mode = top["mode"] if top else None
    amb_in_top = [
        d for d in tax["failed_episode_details"] if d["exclusive"] == top_mode and d["label_uncertainty"] == "medium"
    ]
    n_top = top["n_failed_ics"] if top else 0
    # Worst case: remove all ambiguous from top
    n_top_robust = n_top - len(amb_in_top)
    still_passes_if_ambiguous_removed = n_top_robust >= GATE_MIN_COUNT

    gate_pass = bool(top and top["passes_40pct_gate"])
    report = {
        "gate_definition": f">={GATE_FRAC:.0%} of B1 failures AND >= {GATE_MIN_COUNT} failed ICs",
        "n_fail": n_fail,
        "modes": rows,
        "largest_mode": top,
        "gate_passed": gate_pass,
        "ambiguity_sensitivity": {
            "n_medium_uncertainty_in_top": len(amb_in_top),
            "top_count_if_ambiguous_removed": n_top_robust,
            "still_passes_if_ambiguous_removed": still_passes_if_ambiguous_removed,
            "depends_on_one_or_two_ambiguous": bool(
                gate_pass and n_top - GATE_MIN_COUNT <= 2 and len(amb_in_top) >= n_top - GATE_MIN_COUNT + 1
            ),
        },
        "forbidden": "No combined mega-category invented to pass threshold",
        "decision_implication": (
            "Proceed to signature analysis"
            if gate_pass
            else "STOP optimization direction — Case D (no dominant mechanism)"
        ),
    }
    (OUT / "FAILURE_CONCENTRATION.md").write_text(
        "# Failure Concentration Gate (P84 Phase 3)\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate_passed": gate_pass, "top": top}, indent=2), flush=True)
    return report
