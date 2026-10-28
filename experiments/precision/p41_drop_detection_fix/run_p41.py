"""P41 orchestrator — drop fix audit → regression → P39 revalidation → FINAL."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p41_drop_detection_fix.paths import OUT, P38, P39  # noqa: E402
from experiments.precision.p41_drop_detection_fix.regression import run_regression  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_audit() -> dict:
    """Document drop-detection call sites and the minimal fix."""
    audit = {
        "utc": utc(),
        "problem": (
            "P40: evaluate_handoff_with_axis used _dropped(mon,sim) which was "
            "`below_geometric_drop(z) or z < Z_ABORT(0.44)`. Grasped C objects at "
            "z≈0.16–0.31 failed immediately as DROP_AFTER_HANDOFF despite bilateral contact."
        ),
        "call_sites": [
            "envs/outcomes/horizon_outcome.py::_dropped (definition)",
            "envs/outcomes/horizon_outcome.py::nominal_verify",
            "evaluation/axis_aware_success.py::evaluate_handoff_with_axis",
            "data_generation/oracles/plan_c/future_trajectory.py (handoff verify)",
            "experiments/.../p36 audit_handoff_contract.py (diagnostic copy)",
        ],
        "preserved": [
            "latched(sim) / GROUND_DROP — genuine ground collision",
            "_escaped — XY escape from hand",
            "LOST_GRASP — sustained loss of bilateral (LOST_STEPS)",
            "recovery_failed contact-aware lost_low (unchanged)",
            "WORKING thresholds, nominal_step, physics, experts, M111",
        ],
        "fix": (
            "_dropped: if finger-object contacts nL+nR > 0 → False; else keep "
            "below_geometric_drop or z < Z_ABORT. Low height alone while grasped is not a drop."
        ),
        "file": "envs/outcomes/horizon_outcome.py",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DROP_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def run_p39_reeval(*, limit: int = 0, skip_frozen: bool = False, force_frozen: bool = False) -> dict:
    """Re-run P39 expert validation writing into p41/p39_reeval/ (copy results)."""
    # Run P39 expert validation in-process but redirect OUT via env is hard;
    # instead call expert_validate and copy artifacts into p41.
    from experiments.precision.p39_working_hold_handoff import expert_validate as ev
    from experiments.precision.p39_working_hold_handoff.expert_validate import (
        run_expert_validation,
        r2_reliable,
    )
    from experiments.precision.p39_working_hold_handoff import paths as p39_paths

    # Temporarily redirect P39 OUT to p41 subfolder by monkeypatching
    dest = OUT / "p39_reeval"
    dest.mkdir(parents=True, exist_ok=True)
    old_out = p39_paths.OUT
    old_expert_out = ev.EXPERT_OUT
    p39_paths.OUT = dest
    ev.EXPERT_OUT = dest / "expert_validation"
    try:
        print("=== P41 / P39 expert R0/R1/R2 revalidation ===", flush=True)
        expert = run_expert_validation(limit=limit)
    finally:
        p39_paths.OUT = old_out
        ev.EXPERT_OUT = old_expert_out

    frozen = None
    reliable = bool(expert["verdict"]["r2_reliable"])
    if (reliable or force_frozen) and not skip_frozen:
        from experiments.precision.p39_working_hold_handoff.frozen_eval import eval_frozen
        from experiments.precision.p39_working_hold_handoff import frozen_eval as fe

        old_fout = fe.FROZEN_OUT
        # Also patch P38 ckpt paths remain; write frozen under dest
        fe.FROZEN_OUT = dest / "frozen_p38"
        try:
            print("=== P41 frozen P38 under R2 ===", flush=True)
            frozen = eval_frozen(rule_id="R2")
        finally:
            fe.FROZEN_OUT = old_fout
    else:
        print(
            f"=== Skipping frozen (r2_reliable={reliable}; force={force_frozen}) ===",
            flush=True,
        )

    return {"expert": expert, "frozen": frozen, "dest": str(dest)}


def write_final(audit: dict, regress: dict, p39: dict | None) -> None:
    summ = regress["summary"]
    lines = [
        "# FINAL_REPORT — P41 Drop Detection Fix & Unified HANDOFF Revalidation",
        "",
        f"Generated: {utc()}",
        "",
        "## Objective",
        "",
        "Eliminate false `DROP_AFTER_HANDOFF` when a bilaterally grasped object is below "
        "`Z_ABORT=0.44 m` (P40), then revalidate shared R0/R1/R2 HANDOFF on A/B/C/D.",
        "",
        "## 1. Drop detection audit",
        "",
        f"- Problem: {audit['problem']}",
        f"- Fix: {audit['fix']}",
        f"- File: `{audit['file']}`",
        "",
        "Preserved genuine failures:",
    ]
    for p in audit["preserved"]:
        lines.append(f"- {p}")

    lines += [
        "",
        "## 2. Regression",
        "",
        f"- Unit `_dropped` tests pass: **{summ['unit_pass']}**",
        f"- C branches (8 P40 reps) SUCCESS_NEW — "
        f"first_WORKING={summ['c_first_working_NEW']}/8, "
        f"R2={summ['c_r2_NEW']}/8, mid={summ['c_mid_NEW']}/8, "
        f"terminal={summ['c_terminal_NEW']}/8",
        f"- D terminal smoke NEW={summ['d_terminal_NEW']}/{summ['d_n']} "
        f"(prior P39-R2 successes retained {summ['d_p39_success_retained']}/{summ['d_p39_success_n']})",
        "",
        "P40 before: first/R2/mid = 0/8 NEW (false DROP); terminal = 8/8.",
        "",
    ]

    if p39 is None:
        lines += ["## 3. P39 revalidation", "", "Not run.", ""]
    else:
        expert = p39["expert"]
        agg = expert["agg"]
        lines += [
            "## 3. P39 revalidation (after fix)",
            "",
            "### SUCCESS_NEW / all ICs",
            "",
            "| Skill | R0 | R1 | R2 |",
            "|---|---|---|---|",
        ]
        skills = sorted({a["skill"] for a in agg})
        for sk in skills:
            cells = []
            for rid in ("R0", "R1", "R2"):
                a = next(x for x in agg if x["rule"] == rid and x["skill"] == sk)
                cells.append(f"{a['SUCCESS_NEW']}/{a['n_traj']}")
            lines.append(f"| {sk} | {cells[0]} | {cells[1]} | {cells[2]} |")

        lines += [
            "",
            "### HANDOFF / conversion / never",
            "",
            "| Skill | Rule | HANDOFF | NEW|hand | drop_before | never | fail_cats |",
            "|---|---|---:|---|---:|---:|---|",
        ]
        for a in agg:
            lines.append(
                f"| {a['skill']} | {a['rule']} | {a['handoff']}/{a['n_traj']} | "
                f"{a['NEW_given_handoff']} | {a['drop_before_handoff']} | "
                f"{a['never_satisfied']} | `{a['fail_cats']}` |"
            )

        v = expert["verdict"]
        lines += [
            "",
            f"### R2 reliability gate: **{'PASS' if v['r2_reliable'] else 'FAIL'}**",
        ]
        for f in v.get("flags") or []:
            lines.append(f"- {f}")

        lines += ["", "## 4. Frozen P38", ""]
        if p39.get("frozen") is None:
            lines.append("Skipped (R2 gate failed or --skip-frozen).")
        else:
            lines.append("Ran under R2 — see `p39_reeval/frozen_p38/SUMMARY_R2.csv`.")
            by = {}
            for s in p39["frozen"]["summary"]:
                if int(s.get("nontrivial_only") or 0) == 0:
                    by[(s["kind"], s["suite"])] = s
            lines += [
                "",
                "| Kind | Suite | NEW | HANDOFF | timeout | drop |",
                "|---|---|---|---:|---:|---:|",
            ]
            for kind in ("E0", "E1", "E2"):
                for suite in ("A1", "B1", "C", "D2", "D_nt_test"):
                    s = by.get((kind, suite))
                    if not s:
                        continue
                    lines.append(
                        f"| {kind} | {suite} | {s['SUCCESS_NEW']}/{s['n']} | "
                        f"{s['HANDOFF']} | {s['timeout']} | {s['drop']} |"
                    )

    lines += [
        "",
        "## 5. Remaining failures",
        "",
        "Do not retune WORKING/HANDOFF readiness to chase metrics. Remaining "
        "LOST_GRASP / HOLD_FAIL / DROP (latched) after the height-gate fix are "
        "separate physical/takeover issues.",
        "",
        "## Artifacts",
        "",
        "- `DROP_AUDIT.json`",
        "- `REGRESSION_*.csv`, `REGRESSION_SUMMARY.json`",
        "- `p39_reeval/` (expert + optional frozen)",
        "- `FINAL_REPORT.md`",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "FINAL_REPORT.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-regress", action="store_true")
    ap.add_argument("--resume-regress", action="store_true", help="Reuse C branch CSV; finish D+summary")
    ap.add_argument("--skip-p39", action="store_true")
    ap.add_argument("--skip-frozen", action="store_true")
    ap.add_argument("--force-frozen", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Limit P39 expert jobs (0=all)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    audit = write_audit()

    regress = {"summary": {}}
    if args.resume_regress and (OUT / "REGRESSION_C_BRANCHES.csv").is_file():
        from experiments.precision.p41_drop_detection_fix.regression import (
            d_terminal_smoke,
            write_csv,
            unit_dropped_semantics,
            genuine_fall_smoke,
        )
        from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env

        print("=== P41 resume: unit + D (reuse C CSV) ===", flush=True)
        env = make_env()
        unit = unit_dropped_semantics(env)
        unit.append(genuine_fall_smoke(env))
        write_csv(OUT / "REGRESSION_UNIT_DROPPED.csv", unit)
        c_rows = list(csv.DictReader((OUT / "REGRESSION_C_BRANCHES.csv").open(encoding="utf-8")))
        d_rows = d_terminal_smoke(limit=12)
        write_csv(OUT / "REGRESSION_D_TERMINAL.csv", d_rows)
        summary = {
            "unit_pass": all(int(r["pass"]) for r in unit if "pass" in r),
            "c_first_working_NEW": sum(int(r["SUCCESS_NEW"]) for r in c_rows if r["branch"] == "first_working"),
            "c_r2_NEW": sum(int(r["SUCCESS_NEW"]) for r in c_rows if r["branch"] == "r2"),
            "c_mid_NEW": sum(int(r["SUCCESS_NEW"]) for r in c_rows if r["branch"] == "mid"),
            "c_terminal_NEW": sum(int(r["SUCCESS_NEW"]) for r in c_rows if r["branch"] == "terminal"),
            "c_n_per_branch": 8,
            "d_terminal_NEW": sum(int(r["SUCCESS_NEW"]) for r in d_rows),
            "d_n": len(d_rows),
            "d_p39_success_retained": sum(
                1 for r in d_rows if int(r["was_p39_r2_success"]) and int(r["SUCCESS_NEW"])
            ),
            "d_p39_success_n": sum(int(r["was_p39_r2_success"]) for r in d_rows),
        }
        (OUT / "REGRESSION_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        regress = {"unit": unit, "c": c_rows, "d": d_rows, "summary": summary}
        print("REGRESSION", summary, flush=True)
    elif not args.skip_regress:
        regress = run_regression()
    else:
        summ_path = OUT / "REGRESSION_SUMMARY.json"
        if summ_path.is_file():
            regress = {"summary": json.loads(summ_path.read_text(encoding="utf-8"))}

    p39 = None
    if not args.skip_p39:
        p39 = run_p39_reeval(
            limit=args.limit, skip_frozen=args.skip_frozen, force_frozen=args.force_frozen
        )

    write_final(audit, regress, p39)


if __name__ == "__main__":
    main()
