"""Assemble P70 markdown deliverables."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p70_geometry_to_r3f.paths import OUT


def _load(name: str):
    p = OUT / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_funnel_md(funnel: dict) -> None:
    md = ["# Geometry → R3F Funnel (P70 Phase 1)\n", "## Aggregate by policy/tag\n"]
    for tag, s in (funnel.get("summary") or {}).items():
        md.append(f"### {tag}\n```json\n{json.dumps(s, indent=2)}\n```\n")
    md.append("## Cause flags (overlapping)\n")
    for tag, c in (funnel.get("causes") or {}).items():
        md.append(f"- **{tag}**: `{c}`")
    md.append("\n## Per-IC P64 vs V1\n")
    for row in funnel.get("per_ic_P64_vs_V1") or []:
        md.append(
            f"- `{row['traj_id'][-40:]}`: Δr3={row.get('delta_r3')} Δsw={row.get('delta_sw')} "
            f"Δe_fc={row.get('delta_e_fc')}"
        )
    md.append(
        "\nKeys: A=rare activation, B=late, C=e_pos still above margin, D=lost at contact, "
        "E=sidewall but min_pad<4, F=geometry then lost, G=large-error insufficient.\n"
    )
    (OUT / "GEOMETRY_TO_R3F_FUNNEL.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_abs_md(abs_by: dict) -> None:
    md = [
        "# Absolute Alignment Analysis (P70 Phase 2)\n",
        "Fractions are absolute e_pos, not deltas. 25 mm is diagnostic only; R3 margin remains 9 mm.\n",
    ]
    for tag, stages in (abs_by or {}).items():
        md.append(f"## {tag}\n```json\n{json.dumps(stages, indent=2)}\n```\n")
    (OUT / "ABSOLUTE_ALIGNMENT_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_transition_md(tr: dict) -> None:
    md = [
        "# Sequence Transition Analysis (P70 Phase 3)\n",
        f"Frozen sequence: `{tr.get('sequence')}`\n",
        "## Token metadata\n```json\n" + json.dumps(tr.get("token_meta"), indent=2) + "\n```\n",
        "## Per-index contribution\n```json\n" + json.dumps(tr.get("per_index"), indent=2) + "\n```\n",
        "## Failure-mode summary\n```json\n" + json.dumps(tr.get("fail_modes"), indent=2) + "\n```\n",
    ]
    (OUT / "SEQUENCE_TRANSITION_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_variants_md(var: dict | None, matched: dict | None) -> None:
    md = [
        "# Closed-Loop Variants (P70 Phase 4)\n",
        "## Deployability\n",
        "- Actor obs has **no object e_pos**. Controllers never read privileged working_geometry for decisions.\n",
        "- V2 ready_score = proprioceptive proxy (hand_speed, aperture, track_z).\n",
        "- V3 uses contact_present_L/R from obs.\n",
        "## Frozen rules\n```json\n"
        + json.dumps((matched or {}).get("frozen_variants") or (var or {}), indent=2)
        + "\n```\n",
    ]
    (OUT / "CLOSED_LOOP_VARIANTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_full_ep_md(matched: dict) -> None:
    md = ["# Full-Episode Validation (P70 Phase 5)\n", "Held-out train ICs; rules frozen after discovery tuning.\n"]
    val = matched.get("validation_heldout") or {}
    for k, v in val.items():
        md.append(f"## {k}\n```json\n{json.dumps(v.get('agg'), indent=2)}\n```\n")
    md.append("## Deltas best vs V0\n```json\n" + json.dumps(matched.get("deltas_best_vs_V0"), indent=2) + "\n```\n")
    md.append("## Promotion gate\n```json\n" + json.dumps(matched.get("promotion_gate"), indent=2) + "\n```\n")
    md.append("## Per-IC\n")
    for row in matched.get("per_ic") or []:
        md.append(
            f"- `{row['traj_id'][-40:]}`: V0_r3={row.get('V0',{}).get('r3_ready')} "
            f"ΔV1={row.get('delta_r3_V1')} ΔV2={row.get('delta_r3_V2')} ΔV3={row.get('delta_r3_V3')}"
        )
    (OUT / "FULL_EPISODE_VALIDATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_p71(matched: dict | None, funnel: dict | None, tr: dict | None) -> dict:
    gate = (matched or {}).get("promotion_gate") or {}
    recommend = bool(gate.get("recommend_train"))
    deltas = (matched or {}).get("deltas_best_vs_V0") or {}
    causes = (funnel or {}).get("causes") or {}
    fail = (tr or {}).get("fail_modes") or {}

    if recommend:
        decision = "recommend_small_ppo_or_seq_supervision_pilot"
        limiting = None
        next_step = (
            "One small matched PPO / sequence-supervision pilot from P64_step4, "
            "original reward/M111, strong P64 KL anchor, correct likelihoods. Do not auto-launch."
        )
    else:
        decision = "do_not_train"
        # Infer limiting factor
        hyps = fail.get("hypotheses") or []
        v1causes = causes.get("P64_plus_V1") or {}
        if (deltas.get("r3_ready") or 0) <= 0.01 and (deltas.get("e_first_contact") or 0) < -5:
            limiting = "insufficient_correction_range_or_threshold_cross"
        elif "2_GRIP_before_alignment_sufficient" in hyps or "3_alignment_lost_during_APPROACH_to_GRIP" in hyps:
            limiting = "incorrect_closing_dynamics_APPROACH_to_GRIP"
        elif v1causes.get("E_sidewall_but_minpad_lt4", 0) > v1causes.get("C_epos_above_r3_margin_9", 0):
            limiting = "insufficient_pad_contacts"
        elif not gate.get("improve_abs_alignment"):
            limiting = "lack_of_suitable_observable_trigger_or_weak_proxy"
        else:
            limiting = "inadequate_M111_temporal_expressivity_or_post_contact_disruption"
        next_step = {
            "insufficient_correction_range_or_threshold_cross": (
                "Prioritize longer / stronger APPROACH before GRIP; measure absolute ≤9/12 mm rates, not mean Δe."
            ),
            "incorrect_closing_dynamics_APPROACH_to_GRIP": (
                "Redesign transition: delay GRIP until deployable contact/proprio proxy; test contact-aware abort."
            ),
            "insufficient_pad_contacts": (
                "Focus post-contact pad-building tokens / dwell geometry; sidewall alone is not enough (min_pad≥4)."
            ),
            "lack_of_suitable_observable_trigger_or_weak_proxy": (
                "Do not train on privileged e_pos triggers; improve tactile/proprio proxy or accept open-loop limits."
            ),
            "inadequate_M111_temporal_expressivity_or_post_contact_disruption": (
                "Diagnose post-contact Actor disruption with state restore; consider short post-contact hold, not more pre-contact PPO."
            ),
        }.get(limiting, "Re-run focused diagnostics on best variant failure stage.")

    out = {
        "recommend_train": recommend,
        "decision": decision,
        "promotion_gate": gate,
        "deltas_best_vs_V0": deltas,
        "limiting_factor": limiting if not recommend else None,
        "next_intervention": next_step,
        "note": "No PPO auto-launched.",
    }
    (OUT / "P71_DECISION.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [
        "# P71 Decision (P70 Phase 6–7)\n",
        f"**recommend_train**: `{recommend}`\n",
        f"**decision**: `{decision}`\n",
        f"**limiting_factor**: `{limiting}`\n",
        f"## Next intervention\n{next_step}\n",
        "## Gate\n```json\n" + json.dumps(gate, indent=2) + "\n```\n",
        "Do not automatically launch training.\n",
    ]
    (OUT / "P71_DECISION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def write_final(p71: dict, funnel: dict | None, matched: dict | None, tr: dict | None) -> str:
    summ = (funnel or {}).get("summary") or {}
    v1 = summ.get("P64_plus_V1") or {}
    p64 = summ.get("P64") or {}
    gate = (matched or {}).get("promotion_gate") or {}
    deltas = (matched or {}).get("deltas_best_vs_V0") or {}
    fail = (tr or {}).get("fail_modes") or {}

    answers = {
        "1_why_neg78_no_full_r3f": (
            f"V1 activation/r3 on held-out funnel: act_rate={v1.get('activation_rate')} "
            f"e_after={v1.get('e_after_sequence')} e_fc={v1.get('e_first_contact')} "
            f"r3={v1.get('r3_ready')} vs P64 r3={p64.get('r3_ready')}; causes={((funnel or {}).get('causes') or {}).get('P64_plus_V1')}"
        ),
        "2_triggered_correct_time": v1.get("activation_rate"),
        "3_remaining_e_pos_too_large": fail.get("final_mean_e"),
        "4_geometry_lost_APPROACH_to_GRIP": fail.get("hypotheses"),
        "5_adaptive_timing_helps": {
            "best": gate.get("best_variant"),
            "deltas": deltas,
        },
        "6_deployable_rule_improves_full_r3f": gate.get("improve_r3_full_episode"),
        "7_ppo_justified": p71.get("recommend_train"),
        "decision": p71.get("decision"),
        "limiting_factor": p71.get("limiting_factor"),
        "next": p71.get("next_intervention"),
    }

    md = [
        "# P70 Final Report — Geometry-to-R3F Bridge\n",
        f"**P71 recommendation**: `{p71.get('decision')}` "
        f"(train={p71.get('recommend_train')})\n",
        "## Answers\n",
    ]
    for k, v in answers.items():
        md.append(f"### {k}\n```\n{json.dumps(v, indent=2, default=str)}\n```\n")
    md.append("No PPO training launched. Final-test untouched.\n")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "FINAL_ANSWERS.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
    return str(p71.get("decision"))


def write_all() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    funnel = _load("GEOMETRY_TO_R3F_FUNNEL.json")
    abs_by = _load("ABSOLUTE_ALIGNMENT_ANALYSIS.json")
    tr = _load("SEQUENCE_TRANSITION_ANALYSIS.json")
    matched = _load("MATCHED_COUNTERFACTUALS.json")
    variants = _load("CLOSED_LOOP_VARIANTS.json")
    if funnel:
        write_funnel_md(funnel)
    if abs_by:
        write_abs_md(abs_by)
    if tr:
        write_transition_md(tr)
    write_variants_md(variants, matched)
    if matched:
        write_full_ep_md(matched)
    p71 = write_p71(matched, funnel, tr)
    return write_final(p71, funnel, matched, tr)


if __name__ == "__main__":
    print(write_all())
