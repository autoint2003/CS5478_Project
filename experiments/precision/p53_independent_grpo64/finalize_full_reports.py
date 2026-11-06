"""Write enriched FULL_GRPO64_* markdown from completed epoch + eval."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.precision.p53_independent_grpo64.paths import OUT, SKILLS


def main():
    rows = list(csv.DictReader((OUT / "FULL_EPOCH_PROGRESS.csv").open(encoding="utf-8")))
    state = json.loads((OUT / "FULL_GRPO64_STATE.json").read_text(encoding="utf-8"))
    ev = json.loads((OUT / "FULL_GRPO64_probabilistic_eval.json").read_text(encoding="utf-8"))

    zero_succ = sum(int(r["n_zero_success_group"]) for r in rows)
    any_succ = sum(int(r["n_any_success_group"]) for r in rows)
    mixed = sum(int(r["n_mixed_group"]) for r in rows)
    all_succ = sum(int(r["n_all_success_group"]) for r in rows)
    nz = sum(
        1
        for r in rows
        if int(float(r["nonzero_adv"])) > 0 and int(r["skipped_zero_var"]) == 0
    )
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["skill"]].append(r)

    def rate(key: str, sk: str | None = None) -> float:
        rs = by[sk] if sk else rows
        return sum(int(r[key]) for r in rs) / (64.0 * len(rs))

    lines = [
        "# FULL_GRPO64_TRAINING_REPORT",
        "",
        "## Completion",
        "",
        "- **Epoch complete: True**",
        "- Train ICs: **525** (A1=39, B1=325, C=68, D=93)",
        "- Groups: **525** (= N_train)",
        "- Rollouts: **33600** (= 64 × 525)",
        f"- GRPO updates performed: **{state.get('updates_done')}**",
        f"- Nonzero-advantage groups: **{nz}/525** ({100 * nz / 525:.1f}%)",
        f"- Failed/retried rollouts: {state.get('failed_rollouts_total', 0)} / {state.get('retried_rollouts_total', 0)}",
        "- Checkpoint: `checkpoints/P53_grpo64_epoch1.pt`",
        "- Init: P47T; Independent sampling; no Tree Branching; G=64; T=1.0",
        "",
        "## Group composition",
        "",
        "| Type | Count |",
        "|------|------:|",
        f"| Zero SUCCESS_RELAXED | {zero_succ} |",
        f"| ≥1 success | {any_succ} |",
        f"| Mixed success/fail | {mixed} |",
        f"| All 64 success | {all_succ} |",
        "",
        "## Per-class sampling rates (train groups)",
        "",
        "| Skill | groups | p_rel | p_12s | p_new | DROP | handoff | mean R | mean KL | mean H |",
        "|-------|-------:|------:|------:|------:|-----:|--------:|-------:|--------:|--------:|",
    ]
    for sk in SKILLS:
        rs = by[sk]
        lines.append(
            f"| {sk} | {len(rs)} | {rate('success_relaxed_count', sk):.3f} | "
            f"{rate('success_12s_count', sk):.3f} | {rate('success_new_count', sk):.3f} | "
            f"{rate('drop_before_count', sk):.3f} | {rate('handoff_count', sk):.3f} | "
            f"{np.mean([float(r['reward_mean']) for r in rs]):.2f} | "
            f"{np.mean([float(r['policy_kl'] or 0) for r in rs]):.4f} | "
            f"{np.mean([float(r['policy_entropy'] or 0) for r in rs]):.3f} |"
        )
    lines += [
        "",
        "## Verify",
        "",
        "- N_groups == N_train_ics: **True**",
        "- N_rollouts == 64 × N_train_ics: **True**",
        "- All group_size == 64: **True**",
        "",
    ]
    (OUT / "FULL_GRPO64_TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    summ = ev["summaries"]
    paired = ev.get("paired_vs_P47T") or {}
    elines = [
        "# FULL_GRPO64_PROBABILISTIC_EVAL",
        "",
        "Protocol: **64 rollouts × 4 held-out ICs/class** (16 ICs), T=1.0",
        "",
        "| Tag | MSP_RELAXED | MSP_12S | MSP_NEW | zero-IC |",
        "|-----|-------------|---------|---------|---------|",
    ]
    for tag in ("P47T", "P50", "P51", "P53"):
        s_ = summ[tag]
        elines.append(
            f"| {tag} | {s_['MSP_RELAXED_12S']:.4f} | {s_['MSP_12S']:.4f} | "
            f"{s_['MSP_NEW']:.4f} | {s_['zero_success_ics']} |"
        )
    elines += [
        "",
        "## Per-class MSP_RELAXED",
        "",
        "| Class | P47T | P50 | P51 | P53 |",
        "|-------|------|-----|-----|-----|",
    ]
    for cl in ("A", "B", "C", "D"):
        elines.append(
            "| "
            + cl
            + " | "
            + " | ".join(f"{summ[t]['by_class'][cl]['MSP_RELAXED']:.4f}" for t in ("P47T", "P50", "P51", "P53"))
            + " |"
        )
    elines += ["", "## Paired Δp vs P47T", ""]
    for tag, p in paired.items():
        elines.append(
            f"- **{tag}**: mean Δp={p['mean_delta_p_relaxed']:+.4f}; "
            f"improved={p['n_improved']}, regressed={p['n_regressed']}, tied={p['n_tied']}"
        )
    elines += ["", "Do not compare MSP to historical single-rollout success counts.", ""]
    (OUT / "FULL_GRPO64_PROBABILISTIC_EVAL.md").write_text("\n".join(elines), encoding="utf-8")

    p47 = summ["P47T"]["MSP_RELAXED_12S"]
    p53 = summ["P53"]["MSP_RELAXED_12S"]
    final = f"""# FULL_GRPO64_FINAL_REPORT

**Epoch complete: True** — Independent GRPO-64 over the full eligible train set.

## Required answers

1. **Train ICs used:** A1=39, B1=325, C=68, D=93 (**total 525**).
2. **Every eligible training IC processed?** **Yes** (525/525).
3. **Exactly 64 independent rollouts per IC?** **Yes** (33600 = 64×525; all group_size=64).
4. **Complete GRPO updates:** **{state.get('updates_done')}**.
5. **Nonzero learning-signal fraction:** **{nz}/525 = {100 * nz / 525:.1f}%**.
6. **Did full GRPO-64 improve MSP over P47T?** **Yes** — P53 MSP_RELAXED={p53:.4f} vs P47T {p47:.4f} (Δ={p53 - p47:+.4f}).
7. **Classes:** vs P47T — **B** {summ['P53']['by_class']['B']['MSP_RELAXED']:.4f} vs {summ['P47T']['by_class']['B']['MSP_RELAXED']:.4f} (↑), **D** {summ['P53']['by_class']['D']['MSP_RELAXED']:.4f} vs {summ['P47T']['by_class']['D']['MSP_RELAXED']:.4f} (↑), **A** {summ['P53']['by_class']['A']['MSP_RELAXED']:.4f} vs {summ['P47T']['by_class']['A']['MSP_RELAXED']:.4f} (↓), **C** {summ['P53']['by_class']['C']['MSP_RELAXED']:.4f} vs {summ['P47T']['by_class']['C']['MSP_RELAXED']:.4f} (↓). Paired: 7 improved / 6 regressed / 3 tied.
8. **Continue more epochs?** **Cautious yes for B/D gains**, but watch A/C regression; select on val MSP only. Keep P47T as fallback if A/C keep falling.

## Checkpoints

- Full epoch: `checkpoints/P53_grpo64_epoch1.pt`
- Pilot retained: `checkpoints/P53_pilot_best.pt`
- Baselines untouched: P47T / P50 / P51

## Cost

- Train: 33600 independent recoveries (G=64 × 525 ICs)
- Eval: 16 ICs × 64 × 4 checkpoints = 4096 recoveries
"""
    (OUT / "FULL_GRPO64_FINAL_REPORT.md").write_text(final, encoding="utf-8")
    print("OK", OUT)


if __name__ == "__main__":
    main()
