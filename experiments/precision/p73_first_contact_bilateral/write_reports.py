"""Assemble P74 decision + FINAL_REPORT."""

from __future__ import annotations

import json

from experiments.precision.p73_first_contact_bilateral.paths import OUT


def _load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_all() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    attr = _load("FIRST_CONTACT_FAILURES.json") or {}
    win = _load("INTERVENTION_WINDOW.json") or {}
    val = _load("FULL_EPISODE_VALIDATION.json") or {}

    case = val.get("case") or "D"
    decision = "retain_v2"
    if case == "A" and val.get("promote_candidate"):
        decision = "candidate_further_validation"
    reason = val.get("reason") or "Keep P70 V2 unchanged."

    answers = {
        "1_why_fail_opp_bilateral": {
            "label_rates": attr.get("failure_label_rates"),
            "geom_fail": attr.get("geom_fail_opp"),
            "geom_succ": attr.get("geom_succ_opp"),
            "opp_rate": attr.get("opp_rate"),
        },
        "2_geometry_vs_disp_vs_switch": {
            "frac_ever_sw_window_fail": (attr.get("geom_fail_opp") or {}).get("frac_ever_sw_window"),
            "frac_lost_fail": (attr.get("geom_fail_opp") or {}).get("frac_lost_contact"),
            "switch_rate_fail": (attr.get("geom_fail_opp") or {}).get("mean_switch_rate"),
            "switch_rate_succ": (attr.get("geom_succ_opp") or {}).get("mean_switch_rate"),
            "mean_disp_from_fc_fail": (attr.get("geom_fail_opp") or {}).get("mean_max_disp_from_fc"),
            "note": "Switch rates are normalized by decisions before bil; do not infer causation from frequency alone.",
        },
        "3_why_A_continue_helps_then_hurts": {
            "window": win.get("hurts_after_bilateral"),
            "chosen": win.get("chosen_for_phase4"),
            "generalizes": win.get("generalizes"),
        },
        "4_fixed_obs_rule_exploits_window": {
            "chosen": win.get("chosen_for_phase4"),
            "obs_termination": win.get("obs_termination"),
            "deployed": bool(val.get("chosen_rule")),
        },
        "5_bilateral_improves_frame_ok": {
            "held_delta": ((val.get("held_deltas") or {}).get("P73_minus_V2") or {}),
        },
        "6_terminal_success": {
            "held_delta_success": ((val.get("held_deltas") or {}).get("P73_minus_V2") or {}).get(
                "SUCCESS_RELAXED"
            ),
            "held_agg": val.get("held_agg"),
        },
        "7_generalizes": {
            "held_per_ic": val.get("held_per_ic"),
            "p61": val.get("p61_agg"),
            "p65": val.get("p65_agg"),
        },
        "8_promote_or_retain": decision,
        "case": case,
    }

    p74 = {
        "decision": decision,
        "case": case,
        "recommend_train": False,
        "retain_v2": True,
        "replace_v2": False,
        "promote_candidate": bool(val.get("promote_candidate")),
        "chosen_rule": val.get("chosen_rule"),
        "reason": reason,
        "answers": answers,
    }
    (OUT / "P74_DECISION.json").write_text(json.dumps(p74, indent=2), encoding="utf-8")
    (OUT / "P74_DECISION.md").write_text(
        f"# P74 Decision\n\n**{decision}** — Case {case} — train=False — retain V2\n\n{reason}\n\n"
        f"```json\n{json.dumps({k: p74[k] for k in p74 if k != 'answers'}, indent=2)}\n```\n",
        encoding="utf-8",
    )

    md = [
        "# P73 Final Report — First-Contact → Opposing Bilateral\n",
        f"**Decision: {decision}** (Case {case}; no training; V2 pre-contact unchanged)\n",
        f"{reason}\n",
        "## Answers\n",
    ]
    for k, v in answers.items():
        md.append(f"### {k}\n```json\n{json.dumps(v, indent=2, default=str)}\n```\n")
    md.append("Do not promote on snapshot-only gains. Final-test untouched.\n")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(write_all())
