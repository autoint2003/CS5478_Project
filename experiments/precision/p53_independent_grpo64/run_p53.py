"""P53 orchestrator: audit docs → reward sanity → GRPO smoke → pilot → probabilistic eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.precision.p53_independent_grpo64.grpo_independent import GrpoConfig, train_independent_grpo
from experiments.precision.p53_independent_grpo64.paths import CKPT_DIR, OUT, P47T_CKPT
from experiments.precision.p53_independent_grpo64.probabilistic_eval import compare_baselines, load_heldout_starts
from experiments.precision.p53_independent_grpo64.reward_sanity import run_sanity


def write_docs_if_needed():
    """Ensure deliverable markdown stubs exist (full text written by companion writers)."""
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


def smoke_grpo(device: str = "cpu") -> dict:
    """Group=8 (not 64) × 2 updates — verifies independent sampling + finite grads."""
    cfg = GrpoConfig(
        group=8,
        updates=2,
        device=device,
        per_class_train=4,
        per_class_val=1,
        max_ep_steps=40,
    )
    return train_independent_grpo(cfg, tag="smoke")


def pilot_grpo(device: str = "cpu", group: int = 16, updates: int = 8) -> dict:
    cfg = GrpoConfig(
        group=group,
        updates=updates,
        device=device,
        per_class_train=8,
        per_class_val=2,
        max_ep_steps=80,
    )
    return train_independent_grpo(cfg, tag="pilot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["sanity", "smoke", "pilot", "eval", "all"],
        default="all",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--group", type=int, default=16, help="pilot group size (64 for full)")
    ap.add_argument("--updates", type=int, default=8)
    ap.add_argument("--eval-rollouts", type=int, default=8, help="MSP rollouts/IC (64 for protocol)")
    ap.add_argument("--eval-per-class", type=int, default=2)
    args = ap.parse_args()
    write_docs_if_needed()
    results = {}

    if args.stage in ("sanity", "all"):
        results["sanity"] = run_sanity()
        print("SANITY all_pass=", results["sanity"]["all_pass"], flush=True)

    if args.stage in ("smoke", "all"):
        results["smoke"] = smoke_grpo(args.device)
        (OUT / "GRPO_SMOKE_TEST.md").write_text(
            _smoke_md(results["smoke"]), encoding="utf-8"
        )

    if args.stage in ("pilot", "all"):
        results["pilot"] = pilot_grpo(args.device, group=args.group, updates=args.updates)
        (OUT / "TRAINING_REPORT.md").write_text(
            _train_md(results["pilot"]), encoding="utf-8"
        )

    if args.stage in ("eval", "all"):
        p53 = CKPT_DIR / "P53_pilot_best.pt"
        if not p53.is_file():
            p53 = CKPT_DIR / "P53_smoke_best.pt"
        starts = load_heldout_starts(per_class=args.eval_per_class)
        results["eval"] = compare_baselines(
            starts,
            n_rollouts=args.eval_rollouts,
            p53_ckpt=p53 if p53.is_file() else None,
            device=args.device,
            per_class=args.eval_per_class,
        )
        (OUT / "PROBABILISTIC_EVAL.md").write_text(
            _eval_md(results["eval"]), encoding="utf-8"
        )
        (OUT / "FINAL_REPORT.md").write_text(
            _final_md(results), encoding="utf-8"
        )

    (OUT / "run_p53_summary.json").write_text(
        json.dumps({k: (v if k != "eval" else {"MSP": {t: s.get("MSP_RELAXED_12S") for t, s in v.get("summaries", {}).items()}}) for k, v in results.items()}, indent=2, default=str),
        encoding="utf-8",
    )
    print("DONE", OUT, flush=True)


def _smoke_md(rep: dict) -> str:
    log = rep.get("log") or []
    lines = [
        "# GRPO_SMOKE_TEST",
        "",
        f"- Init: `{P47T_CKPT}`",
        f"- Updates: {len(log)}",
        f"- Group size (smoke): {(log[0]['group'] if log else 'n/a')}",
        "",
        "## Checks",
        "",
    ]
    ok_group = all(r.get("n_members") == r.get("group") for r in log) if log else False
    finite = all(
        r.get("loss") is None or (isinstance(r.get("loss"), float) and abs(r["loss"]) < 1e6)
        for r in log
    )
    lines += [
        f"- Independent groups exact size: **{ok_group}**",
        f"- Finite losses: **{finite}**",
        f"- Best val (1-roll proxy): {rep.get('best_val_relaxed_1roll')}",
        "",
        "## Per-update",
        "",
        "| u | skill | n_succ | ret_std | nz_adv | kl | skipped |",
        "|---|-------|--------|---------|--------|----|---------|",
    ]
    for r in log:
        lines.append(
            f"| {r['update']} | {r['skill']} | {r['n_success_relaxed']} | "
            f"{r['return_std']:.3f} | {r['nonzero_adv']} | {r.get('kl')} | {r['skipped']} |"
        )
    return "\n".join(lines) + "\n"


def _train_md(rep: dict) -> str:
    log = rep.get("log") or []
    by = {}
    for r in log:
        by.setdefault(r["skill"], []).append(r)
    lines = [
        "# TRAINING_REPORT",
        "",
        "Independent GRPO (no Tree Branching). Init = frozen P47T.",
        "",
        f"- Updates: {len(log)}",
        f"- Group: {(log[0]['group'] if log else 'n/a')}",
        f"- Best ckpt: `{rep.get('best_ckpt')}`",
        f"- Val proxy (1-roll SUCCESS_RELAXED): {rep.get('best_val_relaxed_1roll')}",
        "",
        "## Per-class update coverage",
        "",
    ]
    for sk, rows in by.items():
        lines.append(
            f"- {sk}: {len(rows)} updates; mean succ/group={sum(r['n_success_relaxed'] for r in rows)/max(len(rows),1):.2f}; "
            f"mean KL={sum((r.get('kl') or 0) for r in rows)/max(len(rows),1):.4f}"
        )
    lines += ["", "## Note", "", "Do not promote on training reward alone; use MSP on val/held-out.", ""]
    return "\n".join(lines)


def _eval_md(rep: dict) -> str:
    lines = ["# PROBABILISTIC_EVAL", "", f"Rollouts/IC: {rep.get('n_rollouts')}", ""]
    lines += ["| Tag | MSP_RELAXED | MSP_12S | MSP_NEW | zero-IC |", "|-----|-------------|---------|---------|---------|"]
    for tag, s in (rep.get("summaries") or {}).items():
        if "error" in s:
            lines.append(f"| {tag} | ERR | | | |")
            continue
        lines.append(
            f"| {tag} | {s.get('MSP_RELAXED_12S', 0):.3f} | {s.get('MSP_12S', 0):.3f} | "
            f"{s.get('MSP_NEW', 0):.3f} | {s.get('zero_success_ics', 0)} |"
        )
    lines += ["", "## Paired Δp vs P47T", ""]
    for tag, p in (rep.get("paired_vs_P47T") or {}).items():
        lines.append(f"- {tag}: mean Δp={p.get('mean_delta_p_relaxed')} improved={p.get('n_improved')} regressed={p.get('n_regressed')}")
    lines += ["", "Do not compare MSP to historical single-rollout success counts.", ""]
    return "\n".join(lines)


def _final_md(results: dict) -> str:
    ev = results.get("eval") or {}
    summ = ev.get("summaries") or {}
    lines = [
        "# FINAL_REPORT — P53 Independent GRPO-64",
        "",
        "## Verdict (MSP primary)",
        "",
    ]
    for tag in ("P47T", "P50", "P51", "P53"):
        s = summ.get(tag) or {}
        if "MSP_RELAXED_12S" in s:
            lines.append(f"- **{tag}** MSP_RELAXED={s['MSP_RELAXED_12S']:.3f} MSP_12S={s['MSP_12S']:.3f} MSP_NEW={s['MSP_NEW']:.3f}")
    lines += [
        "",
        "## Answers",
        "",
        "1. **Does GRPO increase overall MSP?** See table above vs P47T/P50/P51 under identical protocol.",
        "2. **Improve previously failing ICs?** See paired_vs_P47T n_improved in PROBABILISTIC_EVAL.",
        "3. **Increase successful-IC coverage?** Compare zero_success_ics across tags.",
        "4. **Preserve A/B/C/D?** See by_class MSP in probabilistic_eval.json.",
        "5. **Dense pose shaping?** Terminal favors SUCCESS_RELAXED_12S; sanity tests in REWARD_SANITY_TEST.md.",
        "6. **Strict HANDOFF?** Unchanged R3F (8°/9mm/min_pad≥4/0.5s); see HANDOFF_SPEC.md.",
        "7. **GRPO vs PPO budget?** Compare simulation count: group×updates vs PPO rollout_steps; MSP under same eval N.",
        "8. **Bottleneck class?** Lowest by_class MSP_RELAXED.",
        "",
        "## Constraints honored",
        "",
        "- No MuJoCo/M111/oracle/nominal/SUCCESS_NEW/SUCCESS_12S changes.",
        "- No Tree Branching; Independent GRPO only.",
        "- No training on held-out ICs; temperature=1.0 shared.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
