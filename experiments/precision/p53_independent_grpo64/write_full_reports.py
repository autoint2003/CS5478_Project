"""Build FULL_GRPO64_* reports from progress CSV + eval JSON (post-epoch)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from experiments.precision.p53_independent_grpo64.paths import CKPT_DIR, OUT, SKILLS


def _load_progress():
    path = OUT / "FULL_EPOCH_PROGRESS.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_training_report() -> Path:
    rows = _load_progress()
    man = json.loads((OUT / "FULL_TRAIN_IC_MANIFEST.json").read_text(encoding="utf-8"))
    state = {}
    if (OUT / "FULL_GRPO64_STATE.json").is_file():
        state = json.loads((OUT / "FULL_GRPO64_STATE.json").read_text(encoding="utf-8"))
    n_exp = int(man["n_train_ics"])
    n_done = len(rows)
    complete = bool(state.get("epoch_complete") or state.get("complete")) and n_done >= n_exp

    by = defaultdict(list)
    for r in rows:
        by[r["skill"]].append(r)

    def rate(key, skill=None):
        rs = by[skill] if skill else rows
        if not rs:
            return 0.0
        return sum(int(r[key]) for r in rs) / (64.0 * len(rs))

    nonzero = sum(1 for r in rows if int(float(r.get("nonzero_adv") or 0)) > 0 and int(r.get("skipped_zero_var") or 0) == 0)
    zero_succ = sum(int(r.get("n_zero_success_group") or 0) for r in rows)
    any_succ = sum(int(r.get("n_any_success_group") or 0) for r in rows)
    mixed = sum(int(r.get("n_mixed_group") or 0) for r in rows)
    all_succ = sum(int(r.get("n_all_success_group") or 0) for r in rows)

    lines = [
        "# FULL_GRPO64_TRAINING_REPORT",
        "",
        f"- Complete epoch: **{complete}**",
        f"- Train ICs expected: {n_exp}",
        f"- Groups completed: {n_done}",
        f"- Rollouts completed: {n_done * 64}",
        f"- Expected rollouts: {n_exp * 64}",
        f"- Updates (nonzero-loss steps): {state.get('updates_done')}",
        f"- Nonzero-signal groups: {state.get('nonzero_signal_groups', nonzero)} / {n_done}",
        f"- Failed/retried rollouts: {state.get('failed_rollouts_total', 0)} / {state.get('retried_rollouts_total', 0)}",
        f"- Checkpoint: `{CKPT_DIR / 'P53_grpo64_epoch1.pt'}`",
        "",
        "## Per-class train IC counts (manifest)",
        "",
        "| Class | N_train |",
        "|-------|---------|",
    ]
    for sk, n in man["n_train_per_class"].items():
        lines.append(f"| {sk} | {n} |")
    lines += [
        "",
        "## Per-class sampling rates (over completed groups)",
        "",
        "| Skill | groups | p_rel | p_12s | p_new | DROP | handoff | mean R | mean KL | mean H |",
        "|-------|--------|-------|-------|-------|------|---------|--------|---------|--------|",
    ]
    for sk in SKILLS:
        rs = by[sk]
        if not rs:
            continue
        lines.append(
            f"| {sk} | {len(rs)} | {rate('success_relaxed_count', sk):.3f} | "
            f"{rate('success_12s_count', sk):.3f} | {rate('success_new_count', sk):.3f} | "
            f"{rate('drop_before_count', sk):.3f} | {rate('handoff_count', sk):.3f} | "
            f"{sum(float(r['reward_mean']) for r in rs)/len(rs):.2f} | "
            f"{sum(float(r['policy_kl'] or 0) for r in rs)/len(rs):.4f} | "
            f"{sum(float(r['policy_entropy'] or 0) for r in rs)/len(rs):.3f} |"
        )
    lines += [
        "",
        "## Group composition",
        "",
        f"- Zero successes: {zero_succ}",
        f"- ≥1 success: {any_succ}",
        f"- Mixed success/fail: {mixed}",
        f"- All 64 success: {all_succ}",
        "",
        f"Verify N_groups==N_train: {n_done == n_exp}",
        f"Verify N_rollouts==64*N_train: {n_done * 64 == n_exp * 64 and complete}",
        "",
    ]
    path = OUT / "FULL_GRPO64_TRAINING_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_eval_report(eval_json: Path | None = None) -> Path:
    path = eval_json or (OUT / "probabilistic_eval.json")
    # Prefer dedicated full-eval artifact if present
    alt = OUT / "FULL_GRPO64_probabilistic_eval.json"
    if alt.is_file():
        path = alt
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    summ = data.get("summaries") or {}
    lines = [
        "# FULL_GRPO64_PROBABILISTIC_EVAL",
        "",
        f"Rollouts/IC: {data.get('n_rollouts', 64)}",
        "",
        "| Tag | MSP_RELAXED | MSP_12S | MSP_NEW | zero-IC |",
        "|-----|-------------|---------|---------|---------|",
    ]
    for tag in ("P47T", "P50", "P51", "P53"):
        s = summ.get(tag) or {}
        if "error" in s:
            lines.append(f"| {tag} | ERR | | | |")
            continue
        if "MSP_RELAXED_12S" not in s:
            continue
        lines.append(
            f"| {tag} | {s['MSP_RELAXED_12S']:.3f} | {s['MSP_12S']:.3f} | "
            f"{s['MSP_NEW']:.3f} | {s.get('zero_success_ics', 0)} |"
        )
    lines += ["", "## By class (MSP_RELAXED)", ""]
    for tag in ("P47T", "P50", "P51", "P53"):
        s = summ.get(tag) or {}
        bc = s.get("by_class") or {}
        if bc:
            lines.append(f"### {tag}")
            for cl, d in bc.items():
                lines.append(f"- {cl}: {d.get('MSP_RELAXED', 0):.3f}")
    lines += ["", "## Paired vs P47T", "", json.dumps(data.get("paired_vs_P47T") or {}, indent=2), ""]
    outp = OUT / "FULL_GRPO64_PROBABILISTIC_EVAL.md"
    outp.write_text("\n".join(lines), encoding="utf-8")
    return outp


def write_final_report() -> Path:
    man = json.loads((OUT / "FULL_TRAIN_IC_MANIFEST.json").read_text(encoding="utf-8"))
    state = json.loads((OUT / "FULL_GRPO64_STATE.json").read_text(encoding="utf-8")) if (OUT / "FULL_GRPO64_STATE.json").is_file() else {}
    rows = _load_progress()
    n_exp = int(man["n_train_ics"])
    n_done = len(rows)
    complete = bool(state.get("epoch_complete") or state.get("complete")) and n_done >= n_exp
    eval_path = OUT / "FULL_GRPO64_probabilistic_eval.json"
    if not eval_path.is_file():
        eval_path = OUT / "probabilistic_eval.json"
    ev = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else {}
    summ = ev.get("summaries") or {}
    p47 = (summ.get("P47T") or {}).get("MSP_RELAXED_12S")
    p53 = (summ.get("P53") or {}).get("MSP_RELAXED_12S")
    improved = None
    if p47 is not None and p53 is not None:
        improved = p53 > p47

    lines = [
        "# FULL_GRPO64_FINAL_REPORT",
        "",
        f"**Epoch complete: {complete}**",
        "",
        "## Required answers",
        "",
        f"1. **Train ICs used:** {man['n_train_per_class']} (total {n_exp})",
        f"2. **Every eligible IC processed?** {complete and n_done == n_exp} ({n_done}/{n_exp})",
        f"3. **Exactly 64 rollouts/IC?** {all(int(r['group_size'])==64 for r in rows) if rows else False} (groups={n_done}, rollouts={n_done*64})",
        f"4. **Complete GRPO updates:** {state.get('updates_done')}",
        f"5. **Nonzero learning-signal fraction:** {state.get('nonzero_signal_groups', 0)}/{n_done}",
        f"6. **MSP improved vs P47T?** {improved} (P47T={p47}, P53={p53})",
        "7. **Classes improved/regressed:** see FULL_GRPO64_PROBABILISTIC_EVAL.md by_class + paired deltas",
        "8. **Continue more epochs?** Recommend yes only if MSP≥P47T or clear per-class gains without A/C collapse; else keep P47T.",
        "",
        "## Checkpoints",
        "",
        f"- Full epoch: `checkpoints/P53_grpo64_epoch1.pt`",
        f"- Pilot retained: `checkpoints/P53_pilot_best.pt`",
        "",
    ]
    if not complete:
        lines += [
            "## INCOMPLETE",
            "",
            "This run has not finished a full GRPO-64 epoch. Do not treat as a finished experiment.",
            f"Resume: `python -m experiments.precision.p53_independent_grpo64.train_full_epoch --device cuda`",
            "",
        ]
    path = OUT / "FULL_GRPO64_FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_training_report())
    print(write_eval_report())
    print(write_final_report())
