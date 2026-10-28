#!/usr/bin/env python
"""Enrich 277 analysis + write FINAL_REPORT.md from P6 artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/c_reconfiguration_p6"


def enrich_277() -> None:
    p = json.loads((OUT / "raw" / "910277_p0_p1.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "raw" / "910277_frozen_state.json").read_text(encoding="utf-8"))
    p0, p1 = p["P0"], p["P1"]
    e0 = p0["errors0"]
    e1 = p0["errors1"]
    lines = [
        "# 910277_PRECISION_FAILURE_ANALYSIS",
        "",
        "## Frozen C output (continuation start)",
        "",
        "- Snapshot: `hard_910277__mid_body_early__capture.pkl` after complete C lift+12S.",
        f"- DX={e0['dx']:.3f} mm, DZ={e0['dz']:.3f} mm, dθ={e0['dtheta']:.3f}°, e_pos={e0['e_pos']:.3f} mm",
        f"- grasp_kind=`{e0.get('grasp_kind')}`, contact_s_mean={e0.get('contact_s_mean')}",
        f"- assess_need: dtheta_needed={frozen['assess_need'].get('dtheta_needed')}, "
        f"dx_needed={frozen['assess_need'].get('dx_needed')}, "
        f"dz_needed={frozen['assess_need'].get('dz_needed')}",
        "- Continuation starts from this capture (no manual pose reset).",
        "",
        "## Skills requested vs executed (P0)",
        "",
        f"- Requested: {p0['skills_requested']}",
        f"- Executed: {p0['skills_executed']}",
        f"- Unnecessary DX triggered: **{p0.get('unnecessary_dx')}**",
        f"- Support transfer executed: **{p0['skills_executed'].get('support_transfer')}**",
        f"- Phases: `{' | '.join(p0.get('phases') or [])}`",
        "",
        "## Exact failure mechanism",
        "",
        "1. Only **dθ** correction is requested and executed; DX/DZ primitives are not triggered.",
        "2. `correct_dtheta` runs but only reduces orientation from "
        f"**{e0['dtheta']:.2f}° → {e1['dtheta']:.2f}°** (Δ≈{e0['dtheta']-e1['dtheta']:.2f}°).",
        "3. Working SUCCESS_NEW requires **θ ≤ 10°** and **e_pos ≤ 12 mm** hold for 0.5 s "
        f"(WORKING_THETA_DEG / WORKING_POS_MM). Final dθ={e1['dtheta']:.2f}° **still exceeds 10°** "
        f"while e_pos={e1['e_pos']:.2f} mm is already well inside the position band.",
        "4. Handoff therefore yields **SUCCESS_12S=True** (looser 12 s nominal) but "
        "**SUCCESS_NEW=False** — verification/hold failure on orientation residual, "
        "not contact loss and not DX misfire.",
        "5. First geometric worsening: e_pos rises slightly "
        f"({e0['e_pos']:.2f}→{e1['e_pos']:.2f} mm) during θ attempt; bilateral contact retained.",
        "",
        f"- PRECISION_FAILURE_REASON: `{p0.get('PRECISION_FAILURE_REASON')}`",
        f"- FINAL_POSE_ERROR: dx={e1['dx']:.3f}, dz={e1['dz']:.3f}, dθ={e1['dtheta']:.3f}, e_pos={e1['e_pos']:.3f}",
        "",
        "## P1 paired correction",
        "",
        "Hypothesis: skip any latent DX/DZ path; run `correct_dtheta` + final DOWN/handoff only.",
        "P0 already executed only dθ, so P1 is mechanistically identical and confirms the bottleneck.",
        "",
        f"- P1 SUCCESS_12S={p1.get('SUCCESS_12S')} SUCCESS_NEW={p1.get('SUCCESS_NEW')}",
        f"- P1 reason: `{p1.get('PRECISION_FAILURE_REASON')}`",
        f"- P1 FINAL_POSE_ERROR: `{p1.get('FINAL_POSE_ERROR')}`",
        "",
        "## Ownership",
        "",
        "- **C reconfiguration**: already produced a mid-body opposing grasp with DX/DZ in-band "
        "and SUCCESS_12S. Not the remaining blocker.",
        "- **Precision**: `correct_dtheta` insufficient to bring θ under the 10° SUCCESS_NEW hold. "
        "Further work belongs to orientation skill / support-transfer strategy, not C placement.",
        "",
    ]
    (OUT / "910277_PRECISION_FAILURE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def write_final() -> None:
    place = list(csv.DictReader((OUT / "910263_RECATCH_PLACEMENT.csv").open(encoding="utf-8")))
    cmp = list(csv.DictReader((OUT / "P0_P1_COMPARISON.csv").open(encoding="utf-8")))
    best = next((r for r in place if r["label"] == "s_m015"), place[0])
    base = next((r for r in place if r["label"] == "baseline_s0"), place[0])
    p0 = next(r for r in cmp if r["variant"] == "P0")
    p1 = next(r for r in cmp if r["variant"] == "P1")

    def tf(x):
        return str(x) in ("True", "true", "1")

    lines = [
        "# FINAL_REPORT — P6 C Recatch Placement + Precision Continuation",
        "",
        "Production Oracle unchanged. No expert regeneration. No RL.",
        "",
        "## Answers",
        "",
        "### 1. Can C deliberately produce a more centrally positioned stable grasp for 910263?",
        "",
        "**Yes.** Commanding cylinder-axis `s_axial=−15 mm` (keeping mid_body_early timing, "
        "legal closure, stabilization, production lift, and 12S) moves realized contact from "
        f"`s_mean≈{float(base['realized_contact_s_mean']):.4f}` (near +end, "
        f"`sidewall_partial`) to `s_mean≈{float(best['realized_contact_s_mean']):.4f}` "
        f"(`opposing_sidewall`), with DX={float(best['final_dx']):.2f} mm, "
        f"DZ≈{float(best['final_dz']):.3f} mm, dθ={float(best['final_dtheta']):.2f}°, "
        "CONTACT_PLACEMENT_VALID=True, SUCCESS_12S=True, RECONFIG_IMPROVED=True.",
        "",
        "Baseline DZ deterioration originates mainly **at_release → at_reacquire** "
        "(interception), not during lift. Commanded vs realized: baseline bias "
        f"≈{float(base['s_command_minus_realized']):.4f} m; s_m015 residual "
        f"≈{float(best['s_command_minus_realized']):.4f} m (command nearly matches mid-body).",
        "",
        "### 2. Can the already good C grasp for 910277 be converted into SUCCESS_NEW?",
        "",
        "**Not with the current Precision θ skill in one paired correction.** "
        "Frozen C state already has DX≈1.3 mm, DZ≈2.2 mm, opposing mid-body grasp, SUCCESS_12S. "
        "Only dθ≈14.6° blocks SUCCESS_NEW. P0 and P1 both run `correct_dtheta` and finish with "
        f"dθ≈{float(p0['final_dtheta']):.2f}° (>10° working hold), SUCCESS_12S=True, SUCCESS_NEW=False. "
        "Failure class: **orientation residual / SUCCESS_NEW hold**, not contact loss and not unnecessary DX.",
        "",
        "### 3. Remaining failure ownership",
        "",
        "| IC | Remaining issue | Owner |",
        "|---|---|---|",
        "| 910263 baseline end-grasp | near-end contact + DZ after interception | **C placement** (mitigated by s_m015) |",
        "| 910263 s_m020 / s_p010 | DROP_AFTER_HANDOFF / NO_REACQUIRE | **C** physical limits of placement |",
        "| 910277 | θ≈12.4° after Precision; NEW hold fails | **Precision** (`correct_dtheta` underpowered) |",
        "",
        "## 910263 outcome table",
        "",
        "| label | REACQUIRE | LIFT | 12S | PLACE | IMPROVED | s_real | dx | dz | fail |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for r in place:
        lines.append(
            f"| {r['label']} | {r['REACQUIRE_EXECUTED']} | {r['LIFT_COMPLETED']} | "
            f"{r['SUCCESS_12S']} | {r['CONTACT_PLACEMENT_VALID']} | {r['RECONFIG_IMPROVED']} | "
            f"{r['realized_contact_s_mean']} | {r['final_dx']} | {r['final_dz']} | "
            f"{r.get('stage_fail') or '—'} |"
        )
    lines.extend(
        [
            "",
            "## 910277 outcome table",
            "",
            "| variant | PRECISION_SKILLS | FAILURE_REASON | FINAL_POSE (dx,dz,dθ) | 12S | NEW |",
            "|---|---|---|---|---|---|",
            f"| P0 | {p0['skills']} | `{p0['PRECISION_FAILURE_REASON']}` | "
            f"({p0['final_dx']}, {p0['final_dz']}, {p0['final_dtheta']}) | "
            f"{p0['SUCCESS_12S']} | {p0['SUCCESS_NEW']} |",
            f"| P1 | {p1['skills']} | `{p1['PRECISION_FAILURE_REASON']}` | "
            f"({p1['final_dx']}, {p1['final_dz']}, {p1['final_dtheta']}) | "
            f"{p1['SUCCESS_12S']} | {p1['SUCCESS_NEW']} |",
            "",
            "## Stop conditions",
            "",
            "- 910263: small axial pilot only; best candidate reaches mid-body (stop).",
            "- 910277: diagnosed first Precision failure (θ hold); one paired P1 (stop).",
            "- Production Oracle unmodified; no expert regen; no RL.",
            "",
            "## Artifacts",
            "",
            "- `910263_CONTACT_GEOMETRY_AUDIT.md`",
            "- `910263_RECATCH_PLACEMENT.csv`",
            "- `910277_PRECISION_TRACE.csv`",
            "- `910277_PRECISION_FAILURE_ANALYSIS.md`",
            "- `P0_P1_COMPARISON.csv`",
            "- `VIDEO_INDEX.md`",
            "- `FINAL_REPORT.md`",
            "",
        ]
    )
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    enrich_277()
    write_final()
    print("wrote reports", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
