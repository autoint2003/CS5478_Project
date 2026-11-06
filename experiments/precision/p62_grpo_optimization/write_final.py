"""Write EPOCH_COMPARISON.md and FINAL_REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p62_grpo_optimization.paths import OUT


def write_reports(*, final_eval: dict | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    val = {}
    if (OUT / "VALIDATION_RESULTS.json").is_file():
        val = json.loads((OUT / "VALIDATION_RESULTS.json").read_text(encoding="utf-8"))
    summary = {}
    if (OUT / "P62_TRAIN_SUMMARY.json").is_file():
        summary = json.loads((OUT / "P62_TRAIN_SUMMARY.json").read_text(encoding="utf-8"))
    decision = {}
    if (OUT / "PHASE2_DECISION.json").is_file():
        decision = json.loads((OUT / "PHASE2_DECISION.json").read_text(encoding="utf-8"))

    base = val.get("P57_A") or {}
    epochs = val.get("epochs") or {}

    lines = [
        "# EPOCH_COMPARISON",
        "",
        "Development eval = P61 SHA-256 held-out protocol (16 ICs × 64). Treated as **dev**, not final test.",
        "",
        "| Checkpoint | MSP | A | B | C | D | Zero | Δ vs P57_A |",
        "|------------|----:|--:|--:|--:|--:|-----:|-----------:|",
    ]

    def row(name, r, delta=None):
        bc = r.get("by_class") or {}
        d = "" if delta is None else f"{delta:.4f}"
        return (
            f"| {name} | {r.get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('A') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('B') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('C') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('D') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{r.get('zero_success_ics', '')} | {d} |"
        )

    if base:
        lines.append(row("P57_A (epoch1)", base, 0.0))
    for key in sorted(epochs.keys()):
        ep = epochs[key]
        r = ep.get("dev") or {}
        d = (ep.get("paired_vs_P57_A") or {}).get("mean_delta")
        lines.append(row(f"P62_{key}", r, d))
    lines.append("")
    (OUT / "EPOCH_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")

    # Select best on overall MSP; tie-break zero ICs
    cands = [("P57_A", base)]
    for key, ep in epochs.items():
        cands.append((f"P62_{key}", ep.get("dev") or {}))
    cands = [(n, r) for n, r in cands if r]
    cands.sort(
        key=lambda nr: (
            -float(nr[1].get("MSP_RELAXED") or -1),
            int(nr[1].get("zero_success_ics") or 99),
        )
    )
    best_name, best = cands[0]
    improved = False
    if base and best_name != "P57_A":
        improved = float(best.get("MSP_RELAXED") or 0) > float(base.get("MSP_RELAXED") or 0) + 1e-9

    lines = [
        "# FINAL_REPORT — P62 GRPO Optimization & Multi-Epoch Continuation",
        "",
        "## Answers",
        "",
        f"- **Was P57-A undertrained?** Diagnostics: learning signals remained strong after epoch1 "
        f"(0% zero-var groups, 321/321 updates). Whether more epochs help is answered by the table below.",
        f"- **Did additional epochs improve overall MSP?** "
        f"**{'YES' if improved else 'NO / pending / tied'}** (best={best_name}).",
        f"- **Saturate or regress?** See EPOCH_COMPARISON.md; stop_reason="
        f"`{summary.get('stop_reason')}`.",
        f"- **Zero-success IC coverage?** Compare Zero column in EPOCH_COMPARISON.md.",
        "- **Balanced class sampling justified?** Not for the baseline run "
        "(uncertain hypothesis deferred; see CLASS_SAMPLING_ANALYSIS.md).",
        f"- **Best checkpoint (dev protocol):** **`{best_name}`**",
        "",
        f"Phase-2 decision: {decision.get('decision', 'n/a')}",
        "",
        "## Development ranking",
        "",
    ]
    # Keep comparison table header + rows (skip title / prose above the table)
    ep_lines = open(OUT / "EPOCH_COMPARISON.md", encoding="utf-8").read().splitlines()
    table_start = next(i for i, ln in enumerate(ep_lines) if ln.startswith("| Checkpoint"))
    lines.extend(ep_lines[table_start:])
    if final_eval:
        lines += [
            "",
            "## Fresh final test (untouched ICs)",
            "",
            f"Selected `{best_name}` → final MSP_RELAXED="
            f"**{final_eval.get('MSP_RELAXED')}**, zeros={final_eval.get('zero_success_ics')}, "
            f"by_class={final_eval.get('by_class')}",
            "",
            "This final test was not used for tuning.",
            "",
        ]
    path = OUT / "FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "SELECTED_CHECKPOINT.json").write_text(
        json.dumps({"selected": best_name, "dev": {k: v for k, v in best.items() if k != "per_ic"}}, indent=2),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    print(write_reports())
