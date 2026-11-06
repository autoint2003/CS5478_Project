"""Write P54_TRAINING_REPORT / PROBABILISTIC_EVAL / FINAL_REPORT after epoch+eval."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from experiments.precision.p54_rebalanced_grpo64.paths import OUT, SKILLS


def main():
    rows = list(csv.DictReader((OUT / "P54_FULL_EPOCH_PROGRESS.csv").open(encoding="utf-8")))
    state = json.loads((OUT / "P54_GRPO64_STATE.json").read_text(encoding="utf-8"))
    man = json.loads((OUT / "P54_TRAIN_IC_MANIFEST.json").read_text(encoding="utf-8"))
    n_exp = int(man["n_train_ics"])
    n_done = len(rows)
    complete = bool(state.get("epoch_complete") or state.get("complete")) and n_done >= n_exp

    by = defaultdict(list)
    for r in rows:
        by[r["skill"]].append(r)

    def rate(key, sk=None):
        rs = by[sk] if sk else rows
        return sum(int(r[key]) for r in rs) / (64.0 * max(len(rs), 1))

    zero_succ = sum(int(r["n_zero_success_group"]) for r in rows)
    any_succ = sum(int(r["n_any_success_group"]) for r in rows)
    mixed = sum(int(r["n_mixed_group"]) for r in rows)
    nz = sum(1 for r in rows if int(float(r["nonzero_adv"])) > 0 and int(r["skipped_zero_var"]) == 0)

    lines = [
        "# P54_TRAINING_REPORT",
        "",
        f"- Epoch complete: **{complete}**",
        f"- Train ICs: {n_exp} (actual counts {man['n_train_per_class']})",
        f"- Groups done: {n_done}; rollouts: {n_done * 64}",
        f"- Updates: {state.get('updates_done')}; nonzero-signal: {nz}/{n_done}",
        f"- A limitation: {man.get('a_expansion', {}).get('limitation')}",
        f"- Checkpoint: `checkpoints/P54_grpo64_epoch1.pt`",
        "",
        "## Group composition",
        f"- Zero success: {zero_succ}; ≥1: {any_succ}; mixed: {mixed}",
        "",
        "| Skill | groups | p_rel | DROP | handoff | mean R | mean KL | 0/64 |",
        "|-------|-------:|------:|-----:|--------:|-------:|--------:|-----:|",
    ]
    for sk in SKILLS:
        rs = by[sk]
        if not rs:
            continue
        z = sum(int(r["n_zero_success_group"]) for r in rs)
        lines.append(
            f"| {sk} | {len(rs)} | {rate('success_relaxed_count', sk):.3f} | "
            f"{rate('drop_before_count', sk):.3f} | {rate('handoff_count', sk):.3f} | "
            f"{np.mean([float(r['reward_mean']) for r in rs]):.2f} | "
            f"{np.mean([float(r['policy_kl'] or 0) for r in rs]):.4f} | {z} |"
        )
    (OUT / "P54_TRAINING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ev_path = OUT / "P54_probabilistic_eval.json"
    if ev_path.is_file():
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        summ = ev.get("summaries") or {}
        elines = [
            "# P54_PROBABILISTIC_EVAL",
            "",
            f"Rollouts/IC: {ev.get('n_rollouts')}",
            "",
            "| Tag | MSP_RELAXED | MSP_12S | MSP_NEW | zero-IC |",
            "|-----|-------------|---------|---------|---------|",
        ]
        for tag in ("P47T", "P53", "P54"):
            s = summ.get(tag) or {}
            if "MSP_RELAXED_12S" not in s:
                continue
            elines.append(
                f"| {tag} | {s['MSP_RELAXED_12S']:.4f} | {s['MSP_12S']:.4f} | "
                f"{s['MSP_NEW']:.4f} | {s.get('zero_success_ics', 0)} |"
            )
        elines += ["", "## Per-class MSP_RELAXED", ""]
        for tag in ("P47T", "P53", "P54"):
            s = summ.get(tag) or {}
            bc = s.get("by_class") or {}
            if bc:
                elines.append(f"### {tag}")
                for cl, d in bc.items():
                    elines.append(f"- {cl}: {d.get('MSP_RELAXED', 0):.4f}")
        elines += ["", "## Paired", "", json.dumps(ev.get("paired_vs_P47T") or {}, indent=2), ""]
        (OUT / "P54_PROBABILISTIC_EVAL.md").write_text("\n".join(elines), encoding="utf-8")

        p47 = (summ.get("P47T") or {}).get("MSP_RELAXED_12S")
        p53 = (summ.get("P53") or {}).get("MSP_RELAXED_12S")
        p54 = (summ.get("P54") or {}).get("MSP_RELAXED_12S")
        final = f"""# P54_FINAL_REPORT

**Epoch complete: {complete}**

## Dataset

- Target 341 (A80/B100/C68/D93); **actual {n_exp}** because A expansion stopped at **{man['n_train_per_class'].get('A1')}** valid ICs (see P54_DATASET_AUDIT.md).
- Rollouts: {n_done * 64} (P53 was 33600; reduction ≈ {1 - (n_done * 64) / 33600:.1%}).

## Answers

1. **B dominance reduced?** Yes — B groups {man['n_train_per_class'].get('B1')} vs P53's 325 ({100 * man['n_train_per_class'].get('B1', 0) / n_exp:.1f}% of P54 vs 61.9% of P53).
2. **A MSP improved?** See P54_PROBABILISTIC_EVAL by_class A (P47T/P53/P54).
3. **C recover?** See class C MSP deltas.
4. **B retain P53 gain?** Compare B MSP P54 vs P53 vs P47T.
5. **Overall MSP > P53 0.087?** P54={p54} P53={p53} P47T={p47}.
6. **Cost saved:** {33600 - n_done * 64} fewer train rollouts (~{100 * (1 - (n_done * 64) / 33600):.1f}%).
7. **0/64 ICs by class:** see training report zero-success column.
8. **Expert supervision candidates:** ICs with 0/64 SUCCESS_RELAXED on train + held-out zero-success; prioritize A/C if still weak.

Do not attribute gains solely to rebalancing without accounting for budget and stochastic eval noise.

## Checkpoints

- `checkpoints/P54_grpo64_epoch1.pt` (init P47T)
- P53 retained; P47T/P50/P51 untouched
"""
        (OUT / "P54_FINAL_REPORT.md").write_text(final, encoding="utf-8")
    print("reports written", OUT, "complete", complete)


if __name__ == "__main__":
    main()
