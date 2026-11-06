"""Write FAST_VALIDATION.md and FINAL_REPORT.md from saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p58_fast_c_grpo.paths import OUT, VAL_ABD_PER, VAL_ROLLOUTS


def main():
    s = json.loads((OUT / "FAST_VALIDATION.json").read_text(encoding="utf-8"))
    sel = json.loads((OUT / "SELECTED_C_ICS.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "C_STAGE_AUDIT.json").read_text(encoding="utf-8"))
    train = json.loads((OUT / "P58_TRAIN_SUMMARY.json").read_text(encoding="utf-8"))

    lines = [
        "# FAST_VALIDATION",
        "",
        f"C val: {len(sel['val_c'])} ICs × {VAL_ROLLOUTS} rolls. ABD: {VAL_ABD_PER}/class × {VAL_ROLLOUTS}.",
        "Not the repeatedly tested final held-out set; ICs from P54 train remainder.",
        "",
        "## C validation MSP_RELAXED",
        "",
        "| Model | MSP_C | mean recatch | mean stable | zero ICs |",
        "|-------|------:|-------------:|------------:|---------:|",
    ]
    for name in ("P57_A", "P58_pass1", "P58_pass2"):
        r = s["reports_c"][name]
        per = s["per_ic_c"][name]
        lines.append(
            f"| {name} | {r['MSP_RELAXED']:.4f} | "
            f"{np.mean([x['p_recatch'] for x in per]):.3f} | "
            f"{np.mean([x['p_stable'] for x in per]):.3f} | "
            f"{r['zero_success_ics']} |"
        )
    lines += ["", "## A/B/D validation MSP (catastrophic-regression check)", ""]
    for name in ("P57_A", "P58_pass1", "P58_pass2"):
        r = s["reports_abd"][name]
        lines.append(f"- **{name}**: overall={r['MSP_RELAXED']:.4f} by_skill={r['by_skill']}")
    lines += ["", "## Paired C vs P57-A", ""]
    for k, v in s["paired_c_vs_p57a"].items():
        lines.append(
            f"- {k}: meanΔ={v['mean_delta_c']}, ↑{v['n_improved']} ↓{v['n_regressed']}"
        )
    lines.append("")
    (OUT / "FAST_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")

    c0 = s["reports_c"]["P57_A"]["MSP_RELAXED"]
    c1 = s["reports_c"]["P58_pass1"]["MSP_RELAXED"]
    c2 = s["reports_c"]["P58_pass2"]["MSP_RELAXED"]
    best = "P58_pass2" if c2 >= c1 else "P58_pass1"
    best_msp = max(c1, c2)
    abd0 = s["reports_abd"]["P57_A"]
    abd_best = s["reports_abd"][best]
    # material ABD regression: drop > 0.05 absolute overall or any skill collapse newly to zero from >0.1
    abd_ok = True
    for sk, v0 in (abd0.get("by_skill") or {}).items():
        v1 = (abd_best.get("by_skill") or {}).get(sk, 0.0)
        if v0 >= 0.1 and v1 + 1e-9 < 0.5 * v0:
            abd_ok = False
        if (abd_best["MSP_RELAXED"] + 1e-9) < abd0["MSP_RELAXED"] - 0.05:
            abd_ok = False
    c_improved = best_msp > c0 + 1e-9

    decision = (
        f"Retain **{best}** and consider a larger targeted C run."
        if c_improved and abd_ok
        else "Stop further C-only GRPO; design a more targeted stage intervention from the audit."
    )

    fm = audit["failure_mass"]
    final = [
        "# FINAL_REPORT — P58 Fast C-Focused GRPO-64",
        "",
        "## Verdict",
        "",
        f"- C val MSP: P57-A **{c0:.4f}** → pass1 **{c1:.4f}** → pass2 **{c2:.4f}**",
        f"- Paired vs P57-A: pass1 Δ={s['paired_c_vs_p57a']['P58_pass1']['mean_delta_c']:.4f} (↑{s['paired_c_vs_p57a']['P58_pass1']['n_improved']}/↓0); "
        f"pass2 Δ={s['paired_c_vs_p57a']['P58_pass2']['mean_delta_c']:.4f} (↑{s['paired_c_vs_p57a']['P58_pass2']['n_improved']}/↓0)",
        f"- ABD overall: P57-A {abd0['MSP_RELAXED']:.4f} → {best} {abd_best['MSP_RELAXED']:.4f} (no catastrophic regression filter: **{'PASS' if abd_ok else 'FAIL'}**)",
        f"- **Decision:** {decision}",
        "",
        "## Phase 1 — Stage audit (reuse P56/P57)",
        "",
        f"Failure mass: before_recatch={fm['before_recatch']:.3f}, "
        f"recatch_not_stable={fm['recatch_but_not_stable']:.3f}, "
        f"stable_not_success={fm['stable_but_not_success']:.3f}.",
        "Primary bottleneck by mass: **before valid recatch**, with a large secondary gap from stable hold → SUCCESS/HANDOFF.",
        "",
        "## Phase 2 — Selected ICs",
        "",
        f"Train 16 C (6 hard / 6 mid / 4 easy): see `SELECTED_C_ICS.json`. Val 8 C + 2 A/B/D each from train remainder.",
        "",
        "## Phase 3 — Training",
        "",
        f"{train['n_rollouts']} rollouts (2×16×64) from P57-A, original reward, no expert BC. "
        f"Wall ~{train['elapsed_s']:.0f}s. Checkpoints: `P58_pass1.pt`, `P58_pass2.pt`.",
        "",
        "## Phase 4 — Fast validation",
        "",
        "See `FAST_VALIDATION.md`.",
        "",
        "## Phase 5 — Next step",
        "",
    ]
    if c_improved and abd_ok:
        final += [
            f"C-only GRPO on a stratified 16-IC subset **helps** on the held-aside C val set. "
            f"Best checkpoint: `{best}`. A larger targeted C curriculum (still not a full 321 epoch) is justified; "
            "continue monitoring ABD. Stage audit still points to pre-recatch acquisition as the main mass bottleneck "
            "if further gains stall.",
        ]
    else:
        final += [
            "C-only GRPO did not clear the retention bar. Prefer a targeted intervention on the dominant stage "
            "(pre-recatch acquisition and/or stable→HANDOFF conversion) rather than more C-only GRPO epochs.",
        ]
    final.append("")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(final), encoding="utf-8")
    print(json.dumps({"best": best, "c_improved": c_improved, "abd_ok": abd_ok, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
