"""Enrich P39 FINAL_REPORT conclusions from AGGREGATE.csv."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("results/diagnostics/p39_working_hold_handoff")


def main():
    agg = list(csv.DictReader((OUT / "expert_validation" / "AGGREGATE.csv").open(encoding="utf-8")))
    verdict = json.loads((OUT / "expert_validation" / "R2_RELIABILITY.json").read_text(encoding="utf-8"))
    meta = json.loads((OUT / "expert_validation" / "RULE_METRIC.json").read_text(encoding="utf-8"))

    def get(rule, skill):
        for a in agg:
            if a["rule"] == rule and a["skill"] == skill:
                return a
        raise KeyError((rule, skill))

    skills = sorted({a["skill"] for a in agg})

    lines = [
        "# FINAL_REPORT — P39 Unified Rule-Based HANDOFF Validation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "- Compare shared HANDOFF rules R0 / R1 / R2 on A/B/C/D expert discrete replays.",
        "- No learned HANDOFF, no skill IDs, no physics/STRICT changes, **no new SFT/RL**.",
        "- Nominal contract: P36 `evaluate_handoff_with_axis` (stabilize_n=8).",
        "",
        "## Rules",
        "",
        f"- **R0:** {meta['rules']['R0']['desc']}",
        f"- **R1:** {meta['rules']['R1']['desc']}",
        f"- **R2:** {meta['rules']['R2']['desc']}",
        f"- WORKING: θ≤{meta['WORKING_THETA_DEG']}°, e_pos≤{meta['WORKING_POS_MM']} mm; "
        f"HOLD_S={meta['HOLD_S']}s ({meta['dwell_steps_R1_R2']} policy steps @ dt={meta['DT_POLICY']}).",
        f"- Bilateral: `{meta['bilateral']}`.",
        "",
        "## Expert validation (SUCCESS over **all** ICs)",
        "",
        "### SUCCESS_NEW / n",
        "",
        "| Skill | R0 | R1 | R2 |",
        "|---|---|---|---|",
    ]
    for sk in skills:
        cells = [f"{get(r, sk)['SUCCESS_NEW']}/{get(r, sk)['n_traj']}" for r in ("R0", "R1", "R2")]
        lines.append(f"| {sk} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += [
        "",
        "### SUCCESS_12S / n",
        "",
        "| Skill | R0 | R1 | R2 |",
        "|---|---|---|---|",
    ]
    for sk in skills:
        cells = [f"{get(r, sk)['SUCCESS_12S']}/{get(r, sk)['n_traj']}" for r in ("R0", "R1", "R2")]
        lines.append(f"| {sk} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += [
        "",
        "### HANDOFF rate · timing · conversion · never-satisfied",
        "",
        "| Skill | Rule | HANDOFF | mean step | NEW|hand | drop_before | never |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for a in agg:
        ms = a.get("mean_handoff_step") or ""
        try:
            ms_s = f"{float(ms):.1f}"
        except Exception:
            ms_s = "—"
        lines.append(
            f"| {a['skill']} | {a['rule']} | {a['handoff']}/{a['n_traj']} | {ms_s} | "
            f"{a['NEW_given_handoff']} | {a['drop_before_handoff']} | {a['never_satisfied']} |"
        )

    lines += ["", "### Failure categories", ""]
    for a in agg:
        lines.append(f"- **{a['rule']} {a['skill']}**: `{a['fail_cats']}`")

    lines += [
        "",
        "## Failure analysis",
        "",
        "### R0 (single-frame WORKING) — P38 baseline confirmed",
        "- **C:** 92/92 HANDOFF at ~step 6 → all `DROP_AFTER_HANDOFF`. Instant WORKING mid-reconfiguration.",
        "- **A1/B1:** early HANDOFF → mostly `LOST_GRASP` / hold-fail (6/63, 2/16 NEW).",
        "- **D:** 53/139 NEW; many late HANDOFFs then hold-fail.",
        "",
        "### R1 (0.5 s continuous WORKING)",
        "- **Delays HANDOFF** (~+24 steps ≈ 0.5 s) vs R0 for all skills.",
        "- **A1:** 6/63 → **31/63** NEW. Dwell filters many premature geometric crossings.",
        "- **B1:** 2/16 → **16/16** NEW. Rule becomes reliable on this expert set.",
        "- **C:** still **0/92**. WORKING remains continuously true for ≥0.5 s during Plan-C "
        "recovery (mean HANDOFF ~step 31) but nominal still drops — dwell alone does not mark a safe takeover.",
        "- **D:** 53/139 → **70/139** NEW (fewer premature borderline cuts).",
        "",
        "### R2 (WORKING + bilateral + no latch, 0.5 s)",
        "- Nearly **identical to R1** on A1/B1/C (A1 31/63, B1 16/16, C 0/92).",
        "- D: 71/139 (+1 vs R1). Bilateral rarely changes the decision once WORKING has held 0.5 s.",
        "- Implication: for these experts, **contact was already bilateral** whenever WORKING held; "
        "C’s failure is not ‘missing bilateral’ but **WORKING+bilateral still ≠ nominal basin**.",
        "",
        "## R2 reliability gate",
        "",
        f"- **Reliable for frozen-policy eval?** **{'YES' if verdict['r2_reliable'] else 'NO'}**",
    ]
    for f in verdict.get("flags") or []:
        lines.append(f"- {f}")

    lines += [
        "",
        "## Frozen P38 checkpoints (no retrain)",
        "",
        "Skipped — R2 failed the shared reliability gate (primarily **C = 0/92**). "
        "Per P39 mandate, frozen E0/E1/E2 re-eval waits until a shared rule clears expert validation.",
        "",
        "Note: R1/R2 already recover expert A/B strongly; the **shared-rule** blocker is C, not A/B dwell.",
        "",
        "## Conclusions",
        "",
        "1. **R1 vs R0:** Yes — 0.5 s WORKING dwell substantially improves A1/B1/D all-IC NEW and "
        "eliminates many premature HANDOFFs. It does **not** help C.",
        "2. **R2 vs R1:** Marginal — bilateral+no-latch adds almost nothing beyond R1 on this corpus.",
        "3. **Shared unified rule?** **Not yet.** R2 is not safe for C (0/92 NEW, all DROP_AFTER). "
        "A skill-agnostic HANDOFF rule that works for A/B/D still fails Plan-C mid-recovery states "
        "that satisfy WORKING+bilateral for ≥0.5 s.",
        "4. **Frozen P38:** not run (gate failed). Do not interpret P38 A/B/C zeros as solely premature R0 — "
        "for C, even R1/R2 expert HANDOFF fails.",
        "",
        "**Next (not started):** diagnose C states at R1/R2 HANDOFF (velocity, grip, phase); "
        "consider longer dwell, stability proxies, or C-safe nominal contract — still without new SFT/RL "
        "until a shared rule clears Gate.",
        "",
        "**Do not start new SFT/RL until the HANDOFF rule has been validated.**",
        "",
        "## Artifacts",
        "",
        "- `expert_validation/EXPERT_ROLLOUTS.csv`",
        "- `expert_validation/AGGREGATE.csv`",
        "- `expert_validation/PER_ROOT_COMPARE.csv`",
        "- `expert_validation/FAILURES.csv`",
        "- `expert_validation/R2_RELIABILITY.json`",
        "- `expert_validation/RULE_METRIC.json`",
        "- `FINAL_REPORT.md`",
        "",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote FINAL_REPORT")


if __name__ == "__main__":
    main()
