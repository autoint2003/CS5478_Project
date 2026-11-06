"""Assemble R3F_GATE_AUDIT, INTERVENTION_DECISION, FINAL_REPORT."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import gate_spec
from experiments.precision.p67_r3f_gate_diagnosis.paths import OUT


def _load(name: str):
    p = OUT / name
    if not p.is_file():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return p.read_text(encoding="utf-8")


def write_gate_audit(timelines: dict | None = None) -> str:
    spec = gate_spec()
    timelines = timelines or _load("GATE_PREDICATE_TIMELINES.json") or {}
    rates = timelines.get("stage_rates") or {}
    cond = timelines.get("conditionals") or {}
    disc = timelines.get("discrepancy_resolution") or {}

    md = ["# R3F Gate Audit (P67 Phase 1)\n"]
    md.append("## Source of truth\n")
    md.append(f"- `{spec['source']}`\n")
    md.append("## Implemented predicates (no inventions)\n")
    md.append(
        "| Predicate | Definition |\n|---|---|\n"
        "| not_latched | `latched(sim)` is false |\n"
        f"| working_theta_ok | θ ≤ {spec['WORKING_THETA_DEG']}° |\n"
        f"| working_pos_ok | e_pos ≤ {spec['WORKING_POS_MM']} mm |\n"
        "| working_ok | working_theta ∧ working_pos (pose-only; **not** FSM phase) |\n"
        "| bilateral | pad_bilateral both sides |\n"
        f"| min_pad_ok | min(nL,nR) ≥ {spec['MIN_PAD_CONTACTS']} |\n"
        f"| r3_theta_ok | θ ≤ {spec['R3_MARGIN_THETA_DEG']}° |\n"
        f"| r3_pos_ok | e_pos ≤ {spec['R3_MARGIN_POS_MM']} mm |\n"
        "| margin_ok | r3_theta ∧ r3_pos |\n"
        "| frame_ok | working_ok ∧ bilateral ∧ min_pad_ok ∧ margin_ok |\n"
        f"| dwell_complete | continuous ok_s ≥ {spec['HOLD_S']} s "
        f"({spec['dwell_steps']} steps @ dt={spec['DT_POLICY']}) |\n"
    )
    md.append(f"- Force balance enforced? **{spec['force_balance_enforced']}**")
    md.append(f"- FSM phase enforced? **{spec['fsm_phase_enforced']}**")
    md.append(f"- Fallbacks: {spec['fallbacks']}")
    md.append(f"\n_{spec['note_WORKING']}_\n")

    md.append("## Stage rates (P64_step4, train C)\n")
    for k, v in rates.items():
        md.append(f"- {k}: **{100*v:.1f}%** (n={timelines.get('n_episodes')})")
    md.append("\n## Conditional probabilities (with denominators)\n")
    for name, c in cond.items():
        p = c.get("p")
        ps = "n/a" if p is None else f"{p:.3f}"
        md.append(f"- {name} = {ps}  (num={c.get('num')}, den={c.get('den')})")
    md.append("\n## Resolving the ~0.21 vs aggregate discrepancy\n")
    md.append(str(disc.get("resolution") or ""))
    md.append(f"\n```json\n{json.dumps(disc.get('p67_measured'), indent=2)}\n```\n")

    text = "\n".join(md) + "\n"
    (OUT / "R3F_GATE_AUDIT.md").write_text(text, encoding="utf-8")
    return text


def write_intervention_decision() -> str:
    attr = _load("C_FAILURE_ATTRIBUTION.json") or {}
    cf = _load("EXPERT_COUNTERFACTUAL.json") or {}
    critic = _load("CRITIC_STAGE_ANALYSIS.json") or {}
    timelines = _load("GATE_PREDICATE_TIMELINES.json") or {}
    rates = timelines.get("stage_rates") or {}
    cond = timelines.get("conditionals") or {}
    buckets = cf.get("buckets") or {}
    cats = attr.get("category_counts") or {}
    pred_fail = attr.get("predicate_failure_totals") or {}

    n_cf = max(1, cf.get("n") or 1)
    frac_exp_actor = buckets.get("expert_ok_actor_fail", 0) / n_cf
    frac_neither = buckets.get("neither", 0) / n_cf
    frac_both = buckets.get("both_ok", 0) / n_cf

    top_pred = next(iter(pred_fail.keys()), None) if pred_fail else None
    p_near_to_r3 = (cond.get("P(r3_ready|near_ready)") or {}).get("p")
    p_stable_to_r3 = (cond.get("P(r3_ready|stable_hold)") or {}).get("p")
    v0_bias = critic.get("v0_minus_mc_mean")

    # Decision logic
    case = "C"
    rationale = []
    if frac_neither >= 0.55 and frac_exp_actor < 0.25:
        case = "C"
        rationale.append(
            f"Expert fails on most Actor post-stable states (neither={frac_neither:.2f}); "
            "states often physically unfavorable / Expert-incompatible."
        )
    elif frac_exp_actor >= 0.35:
        case = "A"
        rationale.append(
            f"Expert recovers where Actor fails on {frac_exp_actor:.0%} of snapshots — "
            "control/skill gap at post-stable stage."
        )
    elif v0_bias is not None and abs(float(v0_bias)) >= 10 and frac_exp_actor < 0.35:
        case = "B"
        rationale.append(
            f"Large Critic V0−MC bias ({v0_bias:.1f}) with limited Expert-only recoveries."
        )
    elif (p_near_to_r3 is not None and p_near_to_r3 < 0.35) and cats.get("D_near_ready_disrupted", 0) >= cats.get(
        "A_incorrect_grasp_geometry", 0
    ):
        case = "D"
        rationale.append(
            f"Near-ready→R3 P={p_near_to_r3}; disruptions dominate — dwell interruption."
        )
    else:
        # Prefer geometry if min_pad / sidewall dominate
        if str(top_pred).endswith("min_pad_ok") or cats.get("A_incorrect_grasp_geometry", 0) >= max(
            1, cats.get("D_near_ready_disrupted", 0)
        ):
            case = "C"
            rationale.append(
                "Dominant failures are min_pad / incorrect geometry entering sustained contact; "
                "post-stable imitation unlikely to help."
            )
        elif frac_exp_actor >= 0.2:
            case = "A"
            rationale.append("Moderate Expert-only recoveries → small skill-aware pilot candidate.")
        else:
            case = "C"
            rationale.append("Default: investigate earlier contact geometry (Case C).")

    # Micro-pilot justified?
    pilot_ok = case in ("A", "B", "D") and frac_neither < 0.7
    if case == "C":
        pilot_ok = False

    answers = {
        "1_blocking_predicate": top_pred,
        "2_dwell_main_bottleneck": bool(
            p_near_to_r3 is not None and p_near_to_r3 < 0.5 and rates.get("near_ready", 0) > rates.get("r3_ready", 0)
        ),
        "3_bilateral_geometrically_correct": float(rates.get("ever_correct_sidewall") or 0)
        / max(1e-9, float(rates.get("ever_bilateral") or 1)),
        "4_expert_recovers_actor_poststable": {
            "expert_ok_actor_fail": buckets.get("expert_ok_actor_fail"),
            "both_ok": buckets.get("both_ok"),
            "neither": buckets.get("neither"),
            "incompatible": buckets.get("incompatible"),
        },
        "5_skill_vs_timing": "See SKILL_TRANSITION_ANALYSIS — tokens unlabeled; switches counted.",
        "6_critic_miscalibration_material": v0_bias,
        "7_single_intervention": case,
        "8_training_pilot_justified": pilot_ok,
    }

    case_text = {
        "A": "Case A — Expert can recover, Actor cannot → small post-stable skill-aware imitation/DAgger (P64-anchored; account for P55 regression).",
        "B": "Case B — Critic-only calibration on fresh on-policy Actor returns (no Expert returns as targets).",
        "C": "Case C — States physically unrecoverable / geometry-dominated → do NOT train post-stable imitation; investigate earlier contact geometry.",
        "D": "Case D — R3F dwell interruptions → minimal stage-specific intervention without changing R3F thresholds.",
    }

    md = ["# Intervention Decision (P67 Phase 5)\n"]
    md.append(f"## Selected: **{case}**\n")
    md.append(case_text[case] + "\n")
    md.append("## Evidence\n")
    for r in rationale:
        md.append(f"- {r}")
    md.append(f"- Stable→R3 conditional: {p_stable_to_r3}")
    md.append(f"- Near→R3 conditional: {p_near_to_r3}")
    md.append(f"- Top predicate failure key: `{top_pred}`")
    md.append(f"- Category counts: `{cats}`")
    md.append(f"- Counterfactual buckets: `{buckets}`")
    md.append(f"- Critic V0−MC: `{v0_bias}`")
    md.append(f"\n## Micro-pilot justified? **{pilot_ok}**\n")
    if not pilot_ok:
        md.append("Do **not** launch training. Diagnosis-only stop.\n")
    else:
        md.append(
            "Optional micro-pilot only; init P64_step4; matched sim steps vs unchanged PPO; "
            "evaluate full original C ICs; stop if gate progression does not improve.\n"
        )
    md.append("## Final answers\n")
    for k, v in answers.items():
        md.append(f"- **{k}**: `{v}`")

    (OUT / "INTERVENTION_DECISION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "INTERVENTION_DECISION.json").write_text(
        json.dumps({"case": case, "pilot_justified": pilot_ok, "answers": answers, "rationale": rationale}, indent=2, default=str),
        encoding="utf-8",
    )
    return "\n".join(md)


def write_final_report() -> str:
    write_gate_audit()
    decision = write_intervention_decision()
    parts = [
        "# P67 Final Report — C Post-Stable → R3F Gate Diagnosis\n",
        "## Executive summary\n",
        "Diagnosis only. No long PPO/GRPO. P66 curriculum not promoted. P66_A comparator only.\n",
        _load("R3F_GATE_AUDIT.md") or "",
        _load("C_FAILURE_ATTRIBUTION.md") or "",
        _load("EXPERT_COUNTERFACTUAL.md") or "",
        _load("SKILL_TRANSITION_ANALYSIS.md") or "",
        _load("CRITIC_STAGE_ANALYSIS.md") or "",
        decision,
        "\n## Immutable constraints respected\n",
        "- No reward / physics / M111 / R3F / success / Actor architecture changes.\n",
        "- No automatic training pilot.\n",
    ]
    text = "\n".join(parts)
    (OUT / "FINAL_REPORT.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(write_final_report()[:2000])
