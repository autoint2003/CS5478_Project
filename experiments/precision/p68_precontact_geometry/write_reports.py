"""Write all P68 markdown deliverables + P69 recommendation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p68_precontact_geometry.paths import OUT


def _load(name: str):
    p = OUT / name
    if not p.is_file():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return p.read_text(encoding="utf-8")


def write_all() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    trace = _load("TRACE_EPISODES.json") or {}
    cf = _load("COUNTERFACTUAL_RESULTS.json") or {}
    design = _load("COUNTERFACTUAL_DESIGN.json") or {}
    svf = trace.get("success_vs_fail") or {}
    classes = trace.get("class_summary") or {}
    timing = cf.get("timing") or {}
    overall = cf.get("overall") or {}

    # PRECONTACT_GEOMETRY.md
    md = ["# Pre-Contact Geometry (P68 Phase 1)\n"]
    md.append(f"Episodes: **{trace.get('n_episodes')}** ({trace.get('n_ics')} train C ICs)\n")
    md.append("## Success vs fail (pre-contact)\n")
    for lab in ("success", "fail"):
        s = svf.get(lab) or {}
        md.append(f"### {lab} (n={s.get('n')})")
        md.append(f"- pre e_pos: `{s.get('pre_e_pos')}`")
        md.append(f"- pre theta: `{s.get('pre_theta')}`")
        md.append(f"- first-contact object disp: `{s.get('first_contact_obj_disp_m')}`")
        md.append(f"- Δe_pos at contact: `{s.get('delta_e_pos_at_contact')}`")
        md.append(f"- correct-sidewall rate: {s.get('correct_sidewall_rate')}")
    md.append("\n## Stage rates\n")
    for k, v in (trace.get("stage_rates") or {}).items():
        md.append(f"- {k}: **{100*v:.1f}%**")
    md.append(f"\n## Pre-e_pos → R3 predictor scan\n`{trace.get('pre_e_pos_predictor')}`\n")
    (OUT / "PRECONTACT_GEOMETRY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # FIRST_CONTACT_ANALYSIS.md
    md = ["# First Contact Analysis (P68)\n"]
    md.append("Windows inspected: before first contact; at first contact; bilateral formation; pre stable/R3.\n")
    fail = svf.get("fail") or {}
    succ = svf.get("success") or {}
    md.append("## Displacement at first contact\n")
    md.append(f"- Fail object disp: `{fail.get('first_contact_obj_disp_m')}`")
    md.append(f"- Success object disp: `{succ.get('first_contact_obj_disp_m')}`")
    md.append(f"- Fail Δe_pos: `{fail.get('delta_e_pos_at_contact')}`")
    md.append(f"- Success Δe_pos: `{succ.get('delta_e_pos_at_contact')}`")
    md.append(
        "\nIf fail pre-e_pos already exceeds success and Δe_pos is modest, "
        "R3 failures often **pre-exist** before stable_hold (alignment), "
        "with contact adding further displacement when Δe_pos/disp are large.\n"
    )
    (OUT / "FIRST_CONTACT_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # FAILURE_CLASSIFICATION.md
    md = ["# Failure Classification (P68 Phase 2)\n"]
    md.append("Multi-label (not exclusive).\n")
    md.append("| Class | n | mean pre e_pos | sidewall | near | R3 | success |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for c, s in sorted(classes.items(), key=lambda kv: -kv[1]["n"]):
        md.append(
            f"| {c} | {s['n']} | {s['mean_pre_e_pos']:.1f} | {s['correct_sidewall_rate']:.2f} | "
            f"{s['near_ready_rate']:.2f} | {s['r3_ready_rate']:.2f} | {s['success_rate']:.2f} |"
        )
    (OUT / "FAILURE_CLASSIFICATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # COUNTERFACTUAL_DESIGN.md
    md = ["# Counterfactual Design (P68 Phase 3)\n"]
    md.append(design.get("protocol") or "")
    md.append(f"\n- Snapshots: **{design.get('n_snapshots')}**")
    md.append(f"- Top-k Actor tokens: {design.get('top_k')}")
    md.append(f"- Physical-align book tokens: {design.get('n_phys')}")
    md.append(f"- Matched seeds per condition: {design.get('n_cf_seeds')}")
    md.append("\nCandidates = baseline + Actor top-k + book approach-alignment prototypes.\n")
    md.append("Residual = Actor residual at the restored state (deterministic decoder).\n")
    md.append("Discovery on **train** ICs only; no per-dev-IC outcome search.\n")
    (OUT / "COUNTERFACTUAL_DESIGN.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # M111_EXPRESSIVITY.md
    md = ["# M111 Expressivity Audit (P68 Phase 4)\n"]
    md.append(f"- Fraction of states with any token improving R3: **{overall.get('frac_any_r3_improve')}**")
    md.append(f"- Fraction improving correct sidewall: **{overall.get('frac_any_sw_improve')}**")
    md.append(f"- Fraction improving e_pos (Δ more negative by ≥1 mm): **{overall.get('frac_any_epos_improve')}**")
    md.append(f"- Baseline R3 mean: {overall.get('baseline_r3_mean')}")
    md.append(f"- Best-token R3 mean: {overall.get('best_r3_mean')}")
    md.append(f"- Multi-step search helps (n states): **{cf.get('multi_step_helps_n')}** / {len(cf.get('multi_step') or [])}")
    md.append("\n## Interpretation guide\n")
    md.append("- High R3-improve + low baseline prob on best token → **selection** problem.")
    md.append("- Single-token weak, multi-step helps → **temporal coordination**.")
    md.append("- Neither helps → expressivity gap or state already hard (need Expert-comparable evidence).")
    (OUT / "M111_EXPRESSIVITY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # INTERVENTION_TIMING.md
    md = ["# Intervention Timing (P68 Phase 5)\n"]
    md.append("| Window | n | base R3 | best R3 | frac R3↑ | frac SW↑ | frac e_pos↑ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for kind, t in timing.items():
        md.append(
            f"| {kind} | {t['n']} | {t['baseline_r3']:.2f} | {t['best_r3']:.2f} | "
            f"{t['frac_any_r3_improve']:.2f} | {t['frac_any_sw_improve']:.2f} | {t['frac_any_epos_improve']:.2f} |"
        )
    if timing:
        best_kind = max(timing.items(), key=lambda kv: kv[1]["frac_any_r3_improve"])
        md.append(f"\nMost effective window (R3↑ fraction): **{best_kind[0]}**\n")
    (OUT / "INTERVENTION_TIMING.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Decision case
    frac_r3 = float(overall.get("frac_any_r3_improve") or 0)
    frac_sw = float(overall.get("frac_any_sw_improve") or 0)
    multi_helps = int(cf.get("multi_step_helps_n") or 0)
    base_r3 = float(overall.get("baseline_r3_mean") or 0)
    best_r3 = float(overall.get("best_r3_mean") or 0)
    pre_pred = trace.get("pre_e_pos_predictor") or {}

    fail_pre = ((svf.get("fail") or {}).get("pre_e_pos") or {}).get("mean")
    succ_pre = ((svf.get("success") or {}).get("pre_e_pos") or {}).get("mean")
    alignment_bad = (
        fail_pre is not None and succ_pre is not None and float(fail_pre) > float(succ_pre) + 3.0
    )

    if frac_r3 >= 0.25 and (best_r3 - base_r3) >= 0.05:
        case = "A"
    elif multi_helps >= 2 and frac_r3 < 0.25:
        case = "B"
    elif frac_r3 < 0.15 and frac_sw < 0.2:
        case = "C"
    elif alignment_bad is False and frac_sw < 0.2:
        case = "D"
    else:
        case = "C" if frac_r3 < 0.2 else "A"

    answers = {
        "1_incorrect_alignment_before_contact": bool(alignment_bad),
        "2_first_contact_extra_displacement": (
            ((svf.get("fail") or {}).get("delta_e_pos_at_contact") or {}).get("mean")
        ),
        "3_error_predicting_r3_fail": pre_pred,
        "4_m111_can_correct": frac_r3 >= 0.2 or frac_sw >= 0.25,
        "5_when_correction": max(timing.items(), key=lambda kv: kv[1]["frac_any_r3_improve"])[0] if timing else None,
        "6_geometry_to_r3_and_success": {
            "best_r3_mean": best_r3,
            "baseline_r3_mean": base_r3,
            "best_success_proxy": float(np.mean([r["best"]["SUCCESS_RELAXED"] for r in (cf.get("rows") or []) or [{"best":{"SUCCESS_RELAXED":0}}]])),
        },
        "7_selection_vs_temporal_vs_expressivity": case,
        "8_p69_intervention": case,
    }

    case_text = {
        "A": (
            "Case A — Corrective M111 actions exist. Recommend P69: targeted skill-aware "
            "action supervision on Actor-visited pre-contact states (identified window), "
            "strong policy anchoring to P64_step4, matched PPO control. Do not implement yet."
        ),
        "B": (
            "Case B — Corrections need short M111 sequences. Investigate temporal credit / "
            "skill persistence before expanding vocabulary."
        ),
        "C": (
            "Case C — No tested actions reliably correct geometry. Compare Expert trajectories "
            "under equivalent pre-contact conditions; do not conclude unrecoverable yet."
        ),
        "D": (
            "Case D — Alignment OK but contact still fails. Shift to contact dynamics / closing timing; "
            "avoid more position-alignment reward."
        ),
    }

    md = ["# P69 Recommendation (P68 Phase 6)\n"]
    md.append(f"## Selected: **{case}**\n")
    md.append(case_text[case] + "\n")
    md.append("## Evidence\n")
    md.append(f"- Fail vs succ pre-e_pos means: fail={fail_pre}, succ={succ_pre}")
    md.append(f"- frac_any_r3_improve={frac_r3}, frac_sw_improve={frac_sw}")
    md.append(f"- baseline_r3={base_r3}, best_r3={best_r3}")
    md.append(f"- multi_step_helps_n={multi_helps}")
    md.append(f"- sim_steps (CF): {cf.get('sim_steps')}")
    md.append("\n## Final answers\n")
    for k, v in answers.items():
        md.append(f"- **{k}**: `{v}`")
    md.append("\n**Do not launch training until a plausible improvement mechanism is established.**\n")
    (OUT / "P69_RECOMMENDATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "P69_RECOMMENDATION.json").write_text(
        json.dumps({"case": case, "answers": answers}, indent=2, default=str), encoding="utf-8"
    )

    # FINAL_REPORT
    parts = [
        "# P68 Final Report — Pre-Contact Geometry & M111 Counterfactual\n",
        "## Executive summary\n",
        f"Diagnosis only (no PPO/GRPO/BC/DAgger). Decision **Case {case}**.\n",
        (OUT / "PRECONTACT_GEOMETRY.md").read_text(encoding="utf-8"),
        (OUT / "FIRST_CONTACT_ANALYSIS.md").read_text(encoding="utf-8"),
        (OUT / "FAILURE_CLASSIFICATION.md").read_text(encoding="utf-8"),
        (OUT / "COUNTERFACTUAL_DESIGN.md").read_text(encoding="utf-8"),
        (OUT / "M111_EXPRESSIVITY.md").read_text(encoding="utf-8"),
        (OUT / "INTERVENTION_TIMING.md").read_text(encoding="utf-8"),
        (OUT / "P69_RECOMMENDATION.md").read_text(encoding="utf-8"),
        "\n## Constraints\n",
        "- No physics/reward/M111/R3F/success/architecture changes.\n",
        "- No training launched.\n",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    return case


if __name__ == "__main__":
    print(write_all())
