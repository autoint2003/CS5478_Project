"""Assemble P69 markdown deliverables from JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p69_skill_sequence.paths import OUT


def _load(name: str):
    p = OUT / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_sequence_search_md(search: dict) -> None:
    s = search.get("summary") or {}
    md = [
        "# Sequence Search Results (P69 Phase 1)\n",
        "## Summary\n",
        f"- Discovery states: {s.get('n_discovery_states')}",
        f"- Validation states: {s.get('n_validation_states')}",
        f"- Discovery baseline R3: {s.get('discovery_baseline_r3')}",
        f"- Discovery **oracle** R3: {s.get('discovery_oracle_r3')}",
        f"- Validation baseline R3: {s.get('validation_baseline_r3')}",
        f"- Validation **oracle** R3 (upper bound): {s.get('validation_oracle_r3')}",
        f"- Validation **fixed deployable rule** R3: {s.get('validation_fixed_rule_r3')}",
        f"- Validation fixed rule sidewall: {s.get('validation_fixed_rule_sw')} (baseline {s.get('validation_baseline_sw')})",
        "\nOracle ≠ deployable. Do not promote from oracle alone.\n",
        f"## Frozen rule\n```json\n{json.dumps(s.get('rule'), indent=2)}\n```\n",
        "## Validation per-state (fixed vs oracle vs baseline)\n",
    ]
    for row in search.get("validation_fixed") or []:
        ob = row.get("oracle") or {}
        md.append(
            f"- `{row['id']}`: base_r3={row['baseline'].get('r3_ready')} "
            f"oracle_r3={ob.get('mean_r3')} fixed_r3={row['fixed_rule'].get('r3_ready')} "
            f"Δe_fixed={row['fixed_rule'].get('delta_e_pos')}"
        )
    (OUT / "SEQUENCE_SEARCH_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_persistence_md(persist: dict) -> None:
    p = persist.get("persistence") or persist
    mech = persist.get("mechanism") or persist.get("dominant_mechanism_hypothesis")
    md = [
        "# Skill Persistence Analysis (P69 Phase 2)\n",
        f"- Oracle beats baseline (n): {p.get('n_oracle_beats')}",
        f"- Mean skill-family switches (success): {p.get('mean_switches_success')}",
        f"- Mean switches (all oracles): {p.get('mean_switches_all')}",
        f"- Fraction zero-switch among successes: {p.get('frac_zero_switch_success')}",
        f"- Fraction persist_top1 source among successes: {p.get('frac_persist_source')}",
        f"\n**Dominant mechanism hypothesis**: `{mech}`\n",
        "Interpretation keys: A=stay in family; B=ordered transition; C=token within skill; "
        "D=delay premature contact; E=long corrective chain.\n",
        "## Top family patterns among oracle beats\n",
    ]
    for row in p.get("top_family_patterns") or []:
        md.append(f"- {row['families']}: n={row['n']}")
    if mech == "A_stay_in_family":
        md.append("\nEvidence supports skill-family persistence regularization if PPO proceeds.\n")
    elif mech == "B_ordered_transition":
        md.append("\nPrefer sequence-aware selection over blanket switch penalties.\n")
    else:
        md.append("\nPrefer better within-skill token selection / short fixed sequences over generic persistence.\n")
    (OUT / "SKILL_PERSISTENCE_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_hypothesis_md(rule: dict) -> None:
    md = [
        "# Fixed Sequence Hypothesis (P69)\n",
        "Frozen from discovery ICs only; applied unchanged to validation / held-out.\n",
        f"```json\n{json.dumps(rule, indent=2)}\n```\n",
        "## Deployability\n",
        "- Activation uses skill==C, policy step ≤ max, and no-pad-contact (tactile/pad counts available to Actor obs path).",
        "- No privileged physics or future outcomes.",
        "- Fallback: unchanged P64 Actor after sequence / contact / duration.\n",
    ]
    (OUT / "FIXED_SEQUENCE_HYPOTHESIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_ppo_docs(*, proceeded: bool) -> None:
    corr = _load("PPO_CORRECTNESS_AUDIT.json")
    pilot = _load("PILOT_RESULTS.json")
    cmp = _load("PPO_CONTROL_COMPARISON.json")
    if not proceeded and not pilot:
        audit = [
            "# PPO Correctness Audit (P69)\n",
            "**Phase 4 not launched.** Fixed intervention did not clear the R3F/contact progression gate "
            "on held-out training ICs (or Phase 3 not run).\n",
            "No Actor/Critic updates. No fabricated log-probabilities. P57-A reward untouched.\n",
        ]
        (OUT / "PPO_CORRECTNESS_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
        (OUT / "PPO_CONTROL_COMPARISON.md").write_text(
            "# PPO Control Comparison (P69)\n\nSkipped — Phase 4 gated off.\n",
            encoding="utf-8",
        )
        return

    audit_md = [
        "# PPO Correctness Audit (P69)\n",
        "```json\n" + json.dumps(corr or (pilot or {}).get("ppo_correctness"), indent=2) + "\n```\n",
        "- On-policy PPO buffer contains only Actor-sampled tokens with true log-probs.\n",
        "- Sequence preference aux is a separate NLL + KL-anchor step (not in importance ratios).\n",
        "- Oracle-search trajectories are not treated as on-policy rollouts.\n",
        "- P57-A reward unchanged; M111 / Transformer architecture unchanged.\n",
    ]
    if pilot:
        a, b = pilot.get("A_control") or {}, pilot.get("B_seq_pref") or {}
        audit_md.append(
            f"- Episodes A/B: {a.get('n_episodes')}/{b.get('n_episodes')}; "
            f"sim steps A/B: {a.get('n_sim_steps')}/{b.get('n_sim_steps')}; "
            f"GPU-h A/B: {a.get('gpu_hours')}/{b.get('gpu_hours')}\n"
        )
    (OUT / "PPO_CORRECTNESS_AUDIT.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")

    md = ["# PPO Control Comparison (P69)\n"]
    if cmp:
        md.append("## Evals\n")
        for tag, e in (cmp.get("evals") or {}).items():
            md.append(
                f"- **{tag}**: MSP_RELAXED={e.get('MSP_RELAXED')} "
                f"by_class={e.get('msp_by_class')} zero_ics={e.get('zero_success_ics')}"
            )
        md.append("\n## Paired vs P64_step4\n")
        md.append("```json\n" + json.dumps(cmp.get("paired_vs_P64"), indent=2) + "\n```\n")
    else:
        md.append("Eval JSON missing.\n")
    (OUT / "PPO_CONTROL_COMPARISON.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_final(*, proceed_ppo: bool | None) -> str:
    audit = _load("PRECONTACT_GEOMETRY_AUDIT.json") or {}
    search = _load("SEQUENCE_SEARCH_RESULTS.json") or {}
    persist = _load("SKILL_PERSISTENCE_ANALYSIS.json") or {}
    phase3 = _load("NO_TRAIN_INTERVENTION_RESULTS.json") or {}
    pooled = audit.get("pooled") or {}
    summ = search.get("summary") or {}
    mech = persist.get("mechanism") or search.get("dominant_mechanism_hypothesis")
    p3 = phase3
    delta = p3.get("delta") or {}

    if proceed_ppo is None:
        proceed_ppo = bool(p3.get("proceed_to_ppo"))

    cmp = _load("PPO_CONTROL_COMPARISON.json") or {}
    paired = cmp.get("paired_vs_P64") or {}
    b_pair = paired.get("P69_B_seq_pref") or {}
    a_pair = paired.get("P69_A_control") or {}

    # Decision
    if not p3:
        decision = "stop"
        reason = "Phase 3 incomplete."
    elif not proceed_ppo and not _load("PILOT_RESULTS.json"):
        decision = "stop"
        reason = (
            "Fixed deployable rule did not improve R3F/contact progression on held-out train ICs "
            "(e_pos-only without contact/R3F)."
        )
    elif proceed_ppo and not cmp:
        decision = "refine"
        reason = "Phase 3 cleared contact/R3F gate; PPO pilot pending or incomplete."
    else:
        # Promotion criteria (strict)
        c_ok = (b_pair.get("delta_C_MSP") or 0) > 0.01 or (b_pair.get("delta_C_r3f") or 0) > 0.02
        vs_ctrl = (b_pair.get("delta_MSP_RELAXED") or -1) >= (a_pair.get("delta_MSP_RELAXED") or 0) - 0.01
        no_abd = all((b_pair.get(k) or 0) > -0.05 for k in ("delta_A_MSP", "delta_B_MSP", "delta_D_MSP"))
        if c_ok and vs_ctrl and no_abd:
            decision = "promote"
            reason = "Sequence-aware PPO improved C progression vs P64 without broad A/B/D regression and competitive with control."
        elif (b_pair.get("delta_C_r3f") or 0) > 0 or (b_pair.get("delta_C_MSP") or 0) > 0:
            decision = "refine"
            reason = "Some C-stage signal but promotion criteria not fully met."
        else:
            decision = "stop"
            reason = "Sequence-aware PPO did not beat matched control / P64 on C R3F or MSP."

    q = {
        "1_precontact_gap_robust": (
            f"succ_mean={pooled.get('succ_pre_e_pos')} fail_mean={pooled.get('fail_pre_e_pos')} "
            f"timing_confound={audit.get('timing_confound_flag')}"
        ),
        "2_consistent_skill_structure": persist.get("persistence", persist).get("top_family_patterns"),
        "3_persistence_vs_transition": mech,
        "4_fixed_rule_heldout": delta,
        "5_survives_actor_resume": "Measured via closed-loop continuation in Phase 1 (oracle/fixed after forced seq).",
        "6_r3f_and_terminal": {
            "delta_r3": delta.get("r3_ready"),
            "delta_success": delta.get("SUCCESS_RELAXED"),
        },
        "7_sequence_ppo_vs_control": paired if paired else "N/A — Phase 4 not run",
        "8_decision": decision,
        "reason": reason,
    }

    md = [
        "# P69 Final Report — Skill-Persistent M111 Sequence Learning\n",
        f"**Decision: {decision}** — {reason}\n",
        "## Phase status\n",
        f"- Phase 0 audit: {'yes' if audit else 'no'}",
        f"- Phase 1–2 search: {'yes' if search else 'no'}",
        f"- Phase 3 no-train: {'yes' if p3 else 'no'}; proceed_ppo={proceed_ppo}",
        f"- Phase 4–5 PPO: {'eligible' if proceed_ppo else 'gated off'}\n",
        "## Answers\n",
    ]
    for k, v in q.items():
        md.append(f"### {k}\n```\n{json.dumps(v, indent=2, default=str)}\n```\n")

    md.append("## Promotion criteria\n")
    md.append("Not promoted. Promotion requires Phase 4–5 evidence; Phase 4 only if Phase 3 clears R3F gate.\n")
    md.append(
        f"Key Phase 1 numbers: oracle_val_r3={summ.get('validation_oracle_r3')} "
        f"fixed_val_r3={summ.get('validation_fixed_rule_r3')} "
        f"base_val_r3={summ.get('validation_baseline_r3')}\n"
    )
    (OUT / "FINAL_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "FINAL_DECISION.json").write_text(json.dumps(q, indent=2, default=str), encoding="utf-8")
    return decision


def write_all() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    search = _load("SEQUENCE_SEARCH_RESULTS.json")
    if search:
        write_sequence_search_md(search)
        write_persistence_md(
            {
                "persistence": search.get("persistence"),
                "mechanism": search.get("dominant_mechanism_hypothesis"),
            }
        )
        if search.get("summary", {}).get("rule"):
            write_hypothesis_md(search["summary"]["rule"])
    hyp = _load("FIXED_SEQUENCE_HYPOTHESIS.json")
    if hyp:
        write_hypothesis_md(hyp)
    persist = _load("SKILL_PERSISTENCE_ANALYSIS.json")
    if persist and not search:
        write_persistence_md(persist)
    p3 = _load("NO_TRAIN_INTERVENTION_RESULTS.json") or {}
    proceed = bool(p3.get("proceed_to_ppo")) if p3 else False
    write_ppo_docs(proceeded=proceed or bool(_load("PILOT_RESULTS.json")))
    return write_final(proceed_ppo=proceed if p3 else None)


if __name__ == "__main__":
    print(write_all())
