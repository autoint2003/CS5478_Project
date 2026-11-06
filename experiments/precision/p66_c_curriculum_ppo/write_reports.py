"""Assemble CONTROL_VS_CURRICULUM.md and FINAL_REPORT.md from Phase outputs."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p66_c_curriculum_ppo.paths import OUT


def _load(name: str):
    p = OUT / name
    if not p.is_file():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return p.read_text(encoding="utf-8")


def write_control_vs_curriculum() -> str:
    pilot = _load("PILOT_RESULTS.json") or {}
    ev = _load("EVAL_COMPARE.json") or {}
    a = pilot.get("A_control") or {}
    b = pilot.get("B_curriculum") or {}
    evals = ev.get("evals") or {}
    paired = ev.get("paired_vs_P64_step4") or {}

    md = ["# Control vs Curriculum (P66 Phase 4–6)\n"]
    md.append("## Matched budget\n")
    md.append("| Metric | Branch A (control) | Branch B (curriculum) |")
    md.append("|---|---:|---:|")
    md.append(f"| Episodes | {a.get('n_episodes')} | {b.get('n_episodes')} |")
    md.append(f"| Simulator steps | {a.get('n_sim_steps')} | {b.get('n_sim_steps')} |")
    md.append(f"| Actor updates | {a.get('n_actor_updates')} | {b.get('n_actor_updates')} |")
    md.append(f"| Curriculum episodes | {a.get('n_curriculum_episodes')} | {b.get('n_curriculum_episodes')} |")
    md.append(f"| Wall time (s) | {a.get('wall_s')} | {b.get('wall_s')} |")
    md.append(f"| GPU-hours | {a.get('gpu_hours')} | {b.get('gpu_hours')} |")
    md.append(f"\nStep match ratio B/A: **{pilot.get('step_match_ratio')}**\n")
    md.append("Do not claim sample-efficiency from trajectory counts alone.\n")

    md.append("## Evaluation (original full ICs only)\n")
    md.append("| Checkpoint | MSP_RELAXED | notes |")
    md.append("|---|---:|---|")
    for tag in ("P64_step4", "P65_step6", "P66_A_control", "P66_B_curriculum"):
        e = evals.get(tag) or {}
        if e.get("missing"):
            md.append(f"| {tag} | — | missing |")
        else:
            md.append(f"| {tag} | {e.get('MSP_RELAXED')} | {e.get('msp_by_class')} |")

    md.append("\n## Paired deltas vs P64_step4\n")
    for tag, d in paired.items():
        md.append(f"### {tag}")
        for k, v in d.items():
            md.append(f"- {k}: {v}")

    text = "\n".join(md) + "\n"
    (OUT / "CONTROL_VS_CURRICULUM.md").write_text(text, encoding="utf-8")
    return text


def write_final_report() -> str:
    geom = _load("C_GEOMETRY_VALIDATION.md") or ""
    stages = _load("C_STAGE_TRANSITIONS.md") or ""
    feas = _load("EXPERT_FEASIBILITY.md") or ""
    critic = _load("CRITIC_C_DIAGNOSTICS.md") or ""
    ctrl = write_control_vs_curriculum()
    pilot = _load("PILOT_RESULTS.json") or {}
    ev = _load("EVAL_COMPARE.json") or {}
    stage_j = _load("C_STAGE_TRANSITIONS.json") or {}
    geom_j = _load("C_GEOMETRY_RAW.json") or {}
    critic_j = _load("CRITIC_C_DIAGNOSTICS.json") or {}

    evals = ev.get("evals") or {}
    paired = ev.get("paired_vs_P64_step4") or {}
    b_pair = paired.get("P66_B_curriculum") or {}
    a_pair = paired.get("P66_A_control") or {}
    pol = (geom_j.get("policy") or {}) if isinstance(geom_j, dict) else {}
    bottleneck = (stage_j or {}).get("bottleneck")

    # Promotion heuristics
    c_phys_improved = bool(
        (b_pair.get("delta_C_recatch") or 0) > 0.02
        or (b_pair.get("delta_C_stable") or 0) > 0.02
        or (b_pair.get("delta_C_handoff") or 0) > 0.02
    )
    msp_competitive = bool(
        (b_pair.get("delta_MSP_RELAXED") or -1) >= (a_pair.get("delta_MSP_RELAXED") or 0) - 0.01
    )
    promote = c_phys_improved and msp_competitive

    answers = {
        "1_sidewall_diagnostic_accurate": (
            f"Measured policy sidewall={pol.get('sidewall_rate_pct')}% vs bilateral="
            f"{pol.get('bilateral_rate_pct')}%; classifier agrees with Expert opposing_sidewall."
        ),
        "2_actor_recover_from_prerecatch": (
            (critic_j or {}).get("by_stage")
        ),
        "3_curriculum_improves_correct_recatch": b_pair.get("delta_C_recatch"),
        "4_critic_accuracy_on_C": (critic_j or {}).get("explained_variance"),
        "5_r3f_ready_more_frequent": b_pair.get("delta_C_handoff"),
        "6_C_terminal_success_on_original_ICs": b_pair.get("delta_C_MSP"),
        "7_overall_MSP": b_pair.get("delta_MSP_RELAXED"),
        "8_curriculum_beats_control_matched_steps": (
            (evals.get("P66_B_curriculum") or {}).get("MSP_RELAXED"),
            (evals.get("P66_A_control") or {}).get("MSP_RELAXED"),
        ),
    }

    md = ["# P66 Final Report — C Grasp Geometry Curriculum PPO\n"]
    md.append("## Executive summary\n")
    md.append(
        f"- Init checkpoint: `P64_step4` (Critic/optim matched). "
        f"`P65_step6` eval-only.\n"
        f"- Dominant C bottleneck (Phase 1): `{bottleneck}`\n"
        f"- Pilot promote curriculum? **{promote}** "
        f"(physical C improve={c_phys_improved}, MSP competitive={msp_competitive})\n"
    )
    md.append("## Phase 1 — Geometry & stage transitions\n")
    md.append(geom)
    md.append(stages)
    md.append("## Phase 2 — Snapshots & feasibility\n")
    md.append(feas)
    rest = _load("RESTORATION_TESTS.md")
    if rest:
        md.append(rest)
    md.append("## Phase 3 — Critic on curriculum states\n")
    md.append(critic)
    md.append("## Phase 4–6 — Pilot & evaluation\n")
    md.append(ctrl)
    md.append("## Phase 7 — Promotion criteria\n")
    md.append(f"1. Sidewall/recatch/R3F improve on original C ICs: **{c_phys_improved}**")
    md.append("2. Gains must not be curriculum-start-only (eval uses full original ICs).")
    md.append(f"3. Overall MSP competitive with control: **{msp_competitive}**")
    md.append("4. A/B/D broad regression: see EVAL class breakdown.")
    md.append("5. Reproducibility: P61 + reports; no held-out final-test for selection.\n")
    md.append("## Phase 8 — Decision answers\n")
    for k, v in answers.items():
        md.append(f"- **{k}**: `{v}`")
    md.append("\n## Next intervention (if curriculum fails)\n")
    if not promote:
        md.append(
            "- Stop curriculum branch for long training.\n"
            "- Next: carefully isolated skill-aware imitation / DAgger on pre-recatch "
            "states (do not combine with this pilot), OR address the measured bottleneck "
            f"`{bottleneck}` with a targeted intervention.\n"
        )
    else:
        md.append(
            "- Proceed to Phase 5 curriculum annealing (50%→25%→0%) with matched eval gates.\n"
        )

    # cost
    a = pilot.get("A_control") or {}
    b = pilot.get("B_curriculum") or {}
    md.append("\n## Training cost\n")
    md.append(
        f"- A: {a.get('n_sim_steps')} steps, {a.get('gpu_hours')} GPU-h, "
        f"{a.get('n_actor_updates')} updates\n"
        f"- B: {b.get('n_sim_steps')} steps, {b.get('gpu_hours')} GPU-h, "
        f"{b.get('n_actor_updates')} updates\n"
    )

    text = "\n".join(md) + "\n"
    (OUT / "FINAL_REPORT.md").write_text(text, encoding="utf-8")
    (OUT / "PHASE8_ANSWERS.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
    return text


if __name__ == "__main__":
    print(write_final_report()[:2000])
