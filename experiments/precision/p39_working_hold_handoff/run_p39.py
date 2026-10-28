"""P39 orchestrator — expert R0/R1/R2 validation → optional frozen P38 eval → FINAL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p39_working_hold_handoff.expert_validate import (  # noqa: E402
    run_expert_validation,
)
from experiments.precision.p39_working_hold_handoff.frozen_eval import eval_frozen  # noqa: E402
from experiments.precision.p39_working_hold_handoff.paths import OUT  # noqa: E402
from experiments.precision.p39_working_hold_handoff.rules import RULE_IDS, metric_meta  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_final(expert: dict, frozen: dict | None) -> None:
    agg = expert["agg"]
    verdict = expert["verdict"]
    meta = metric_meta()

    def cell(rule, skill, key):
        for a in agg:
            if a["rule"] == rule and a["skill"] == skill:
                return a.get(key)
        return None

    skills = sorted({a["skill"] for a in agg})
    lines = [
        "# FINAL_REPORT — P39 Unified Rule-Based HANDOFF Validation",
        "",
        f"Generated: {utc()}",
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
        cells = []
        for rid in RULE_IDS:
            a = next(x for x in agg if x["rule"] == rid and x["skill"] == sk)
            cells.append(f"{a['SUCCESS_NEW']}/{a['n_traj']}")
        lines.append(f"| {sk} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += [
        "",
        "### HANDOFF rate · conversion NEW|handoff · never-satisfied",
        "",
        "| Skill | Rule | HANDOFF | mean step | NEW|hand | drop_before | never |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for a in agg:
        ms = a["mean_handoff_step"]
        ms_s = f"{ms:.1f}" if ms is not None else "—"
        lines.append(
            f"| {a['skill']} | {a['rule']} | {a['handoff']}/{a['n_traj']} | {ms_s} | "
            f"{a['NEW_given_handoff']} | {a['drop_before_handoff']} | {a['never_satisfied']} |"
        )

    lines += [
        "",
        "### Failure categories (per rule×skill)",
        "",
    ]
    for a in agg:
        lines.append(f"- **{a['rule']} {a['skill']}**: `{a['fail_cats']}`")

    lines += [
        "",
        "## R2 reliability gate",
        "",
        f"- **Reliable for frozen-policy eval?** **{'YES' if verdict['r2_reliable'] else 'NO'}**",
    ]
    if verdict.get("flags"):
        lines.append("- Flags:")
        for f in verdict["flags"]:
            lines.append(f"  - {f}")
    else:
        lines.append("- No reliability flags.")

    lines += ["", "## Frozen P38 checkpoints (no retrain)", ""]
    if frozen is None:
        lines.append(
            "Skipped — R2 did not pass the reliability gate. "
            "Premature-HANDOFF hypothesis for A/B/C remains untested under R2 on learned policies."
        )
    else:
        lines.append(f"Rule used: **{frozen['rule']}**.")
        lines.append("")
        lines.append("| Kind | Suite | NEW (all) | NEW (nontrivial) | HANDOFF | timeout | drop |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        by = {(s["kind"], s["suite"], s["nontrivial_only"]): s for s in frozen["summary"]}
        for kind in ("E0", "E1", "E2"):
            for suite in ("A1", "B1", "C", "D2", "D_nt_test"):
                a0 = by.get((kind, suite, 0))
                a1 = by.get((kind, suite, 1))
                if not a0:
                    continue
                n0 = f"{a0['SUCCESS_NEW']}/{a0['n']}"
                n1 = f"{a1['SUCCESS_NEW']}/{a1['n']}" if a1 else "—"
                lines.append(
                    f"| {kind} | {suite} | {n0} | {n1} | {a0['HANDOFF']} | "
                    f"{a0['timeout']} | {a0['drop']} |"
                )
        lines.append("")
        lines.append(
            "Compare to P38 (R0 WORKING): A/B/C were 0 NEW; D_nt E0=7/11. "
            "If A/B/C NEW rises under R2, premature HANDOFF was a primary limiter."
        )

    lines += [
        "",
        "## Conclusions",
        "",
        "1. Does R1 (0.5s WORKING dwell) reduce mid-traj false HANDOFF vs R0?",
        "2. Does R2 (WORKING+bilateral+no-latch dwell) further improve conversion / all-IC NEW?",
        "3. Is R2 reliable enough to reuse as the shared unified HANDOFF rule?",
        f"4. Frozen P38 recovery under R2: {'ran' if frozen else 'not run (gate failed)'}.",
        "",
        "**Do not start new SFT/RL until a HANDOFF rule clears expert validation.**",
        "",
        "## Artifacts",
        "",
        "- `expert_validation/EXPERT_ROLLOUTS.csv`",
        "- `expert_validation/AGGREGATE.csv`",
        "- `expert_validation/PER_ROOT_COMPARE.csv`",
        "- `expert_validation/FAILURES.csv`",
        "- `expert_validation/R2_RELIABILITY.json`",
        "- `frozen_p38/` (if gate passed)",
        "- `FINAL_REPORT.md`",
        "",
    ]
    text = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FINAL_REPORT.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-expert", action="store_true")
    ap.add_argument("--skip-frozen", action="store_true")
    ap.add_argument("--force-frozen", action="store_true", help="Run frozen eval even if R2 gate fails")
    ap.add_argument("--frozen-rule", default="R2", choices=list(RULE_IDS))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.skip_expert:
        agg = list(csv.DictReader((OUT / "expert_validation" / "AGGREGATE.csv").open(encoding="utf-8")))
        for a in agg:
            for k in (
                "n_traj",
                "handoff",
                "SUCCESS_12S",
                "SUCCESS_NEW",
                "drop_before_handoff",
                "never_satisfied",
            ):
                if k in a and a[k] != "":
                    a[k] = int(float(a[k]))
            if a.get("conversion_rate") not in (None, ""):
                try:
                    a["conversion_rate"] = float(a["conversion_rate"])
                except Exception:
                    a["conversion_rate"] = None
            else:
                a["conversion_rate"] = None
            if a.get("NEW_over_all") not in (None, ""):
                try:
                    a["NEW_over_all"] = float(a["NEW_over_all"])
                except Exception:
                    a["NEW_over_all"] = None
            if a.get("mean_handoff_step") not in (None, ""):
                try:
                    a["mean_handoff_step"] = float(a["mean_handoff_step"])
                except Exception:
                    a["mean_handoff_step"] = None
            if isinstance(a.get("fail_cats"), str):
                try:
                    a["fail_cats"] = json.loads(a["fail_cats"])
                except Exception:
                    pass
        verdict = json.loads((OUT / "expert_validation" / "R2_RELIABILITY.json").read_text(encoding="utf-8"))
        expert = {"agg": agg, "verdict": verdict}
    else:
        print("=== P39 expert R0/R1/R2 ===", flush=True)
        expert = run_expert_validation(limit=args.limit)

    frozen = None
    run_fr = args.force_frozen or (expert["verdict"]["r2_reliable"] and not args.skip_frozen)
    if run_fr and not args.skip_frozen:
        print(f"=== P39 frozen P38 under {args.frozen_rule} ===", flush=True)
        frozen = eval_frozen(rule_id=args.frozen_rule)
    elif not args.skip_frozen:
        print("=== Skipping frozen eval (R2 not reliable; use --force-frozen to override) ===", flush=True)

    write_final(expert, frozen)


if __name__ == "__main__":
    main()
