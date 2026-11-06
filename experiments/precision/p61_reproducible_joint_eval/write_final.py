"""Compose FINAL_REPORT.md from P61 joint evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p61_reproducible_joint_eval.paths import OUT


def _ci_covers_zero(ci) -> bool:
    if not ci or len(ci) != 2:
        return True
    return ci[0] <= 0.0 <= ci[1]


def select_checkpoint(joint: dict) -> dict:
    """Primary: overall held-out MSP_RELAXED. Secondary: C, ABD, zeros, paired uncertainty."""
    held = joint["heldout"]
    ranking = sorted(
        (("P57_A", held["P57_A"]), ("P59_pass1", held["P59_pass1"]), ("P59_pass2", held["P59_pass2"])),
        key=lambda kv: kv[1]["MSP_RELAXED"],
        reverse=True,
    )
    best_name, best = ranking[0]
    second_name, second = ranking[1]
    gap = float(best["MSP_RELAXED"] - second["MSP_RELAXED"])

    # Paired uncertainty vs runner-up
    key = None
    if best_name == "P57_A":
        # use paired of second vs P57 (negate)
        pk = f"{second_name}_vs_P57_A"
        paired = joint["paired_heldout"].get(pk) or {}
        # Δ(second - P57); we want Δ(best - second) ≈ -paired if best is P57
        mean_d = None if paired.get("mean_delta") is None else -float(paired["mean_delta"])
        ci = None
        if paired.get("ci95"):
            ci = [-paired["ci95"][1], -paired["ci95"][0]]
    elif second_name == "P57_A":
        pk = f"{best_name}_vs_P57_A"
        paired = joint["paired_heldout"].get(pk) or {}
        mean_d = paired.get("mean_delta")
        ci = paired.get("ci95")
        key = pk
    else:
        pk = "P59_pass2_vs_P59_pass1"
        paired = joint["paired_heldout"].get(pk) or {}
        if best_name == "P59_pass2":
            mean_d = paired.get("mean_delta")
            ci = paired.get("ci95")
        else:
            mean_d = None if paired.get("mean_delta") is None else -float(paired["mean_delta"])
            ci = [-paired["ci95"][1], -paired["ci95"][0]] if paired.get("ci95") else None
        key = pk

    tie = gap < 0.005 or (ci is not None and _ci_covers_zero(ci))
    # ABD retention: mean of A,B,D
    def abd(r):
        bc = r.get("by_class") or {}
        vals = [(bc.get(c) or {}).get("MSP_RELAXED") for c in ("A", "B", "D")]
        vals = [v for v in vals if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {
        "selected": best_name if not tie else f"STATISTICAL_TIE:{best_name}~{second_name}",
        "primary_leader": best_name,
        "msp_relaxed": best["MSP_RELAXED"],
        "gap_vs_second": gap,
        "second": second_name,
        "paired_mean_delta_vs_second": mean_d,
        "paired_ci95_vs_second": ci,
        "statistical_tie": bool(tie),
        "ranking": [(n, r["MSP_RELAXED"]) for n, r in ranking],
        "c_msp": {n: (held[n].get("by_class") or {}).get("C", {}).get("MSP_RELAXED") for n, _ in ranking},
        "abd_msp": {n: abd(held[n]) for n, _ in ranking},
        "zero_ics": {n: held[n]["zero_success_ics"] for n, _ in ranking},
        "further_c_only_justified": None,  # filled by write_final
    }


def write_final() -> Path:
    joint = json.loads((OUT / "JOINT_EVAL.json").read_text(encoding="utf-8"))
    repro = {}
    if (OUT / "REPRODUCIBILITY_TEST.json").is_file():
        repro = json.loads((OUT / "REPRODUCIBILITY_TEST.json").read_text(encoding="utf-8"))
    sel = select_checkpoint(joint)

    # Further C-only justified? Only if C clearly improves overall without ABD collapse
    p57 = joint["heldout"]["P57_A"]
    p59 = joint["heldout"][sel["primary_leader"] if sel["primary_leader"].startswith("P59") else "P59_pass1"]
    c57 = (p57.get("by_class") or {}).get("C", {}).get("MSP_RELAXED")
    c59 = (p59.get("by_class") or {}).get("C", {}).get("MSP_RELAXED")
    overall_gain = float(p59["MSP_RELAXED"] - p57["MSP_RELAXED"])
    paired = joint["paired_heldout"].get("P59_pass1_vs_P57_A") or joint["paired_heldout"].get(
        "P59_pass2_vs_P57_A"
    )
    c_only_ok = False
    if c59 is not None and c57 is not None and c59 > c57 + 0.01 and overall_gain > 0 and paired:
        c_only_ok = (not _ci_covers_zero(paired.get("ci95"))) and overall_gain > 0.005
    # If P57 leads overall, further C-only not justified by overall MSP
    if sel["primary_leader"] == "P57_A":
        c_only_ok = False
    sel["further_c_only_justified"] = bool(c_only_ok)

    (OUT / "CHECKPOINT_SELECTION.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")

    lines = [
        "# FINAL_REPORT — P61 Reproducible Joint Evaluation",
        "",
        "## Verdict",
        "",
    ]
    if sel["statistical_tie"]:
        lines.append(
            f"**Statistical tie** between `{sel['primary_leader']}` and `{sel['second']}` "
            f"on held-out MSP_RELAXED (gap={sel['gap_vs_second']:.4f}; paired CI covers 0 or gap<0.005)."
        )
        lines.append(
            f"Primary leader by point estimate: **`{sel['primary_leader']}`** "
            f"(MSP={sel['msp_relaxed']:.4f})."
        )
    else:
        lines.append(
            f"**Selected checkpoint: `{sel['primary_leader']}`** "
            f"(held-out MSP_RELAXED={sel['msp_relaxed']:.4f})."
        )

    lines += [
        "",
        f"Further C-only GRPO justified by this joint eval: **{sel['further_c_only_justified']}**",
        "",
        "## Held-out (this harness only)",
        "",
        "| Model | MSP | A | B | C | D | Zero |",
        "|-------|----:|--:|--:|--:|--:|-----:|",
    ]
    for name in ("P57_A", "P59_pass1", "P59_pass2"):
        r = joint["heldout"][name]
        bc = r.get("by_class") or {}
        lines.append(
            f"| {name} | {r['MSP_RELAXED']:.4f} | "
            f"{(bc.get('A') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('B') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('C') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('D') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{r['zero_success_ics']} |"
        )
    lines += ["", "## Paired Δ (held-out)", ""]
    for k, v in joint["paired_heldout"].items():
        lines.append(
            f"- {k}: meanΔ={v['mean_delta']:.4f} CI95=[{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}] "
            f"↑{v['n_improved']} ↓{v['n_regressed']}"
        )
    lines += [
        "",
        "## Reproducibility",
        "",
        f"Exact match (see REPRODUCIBILITY_TEST.md): "
        f"**{(repro.get('comparison') or {}).get('exact_match')}** "
        f"on device={(repro.get('device'))}.",
        "",
        "## Residual / KL (summary)",
        "",
        "Token stochastic; residual deterministic; GRPO optimizes token logp only; "
        "KL is a single-state probe to frozen init — weak A/B/D coverage under C-only training. "
        "See RESIDUAL_KL_AUDIT.md.",
        "",
        "## Constraints honored",
        "",
        "- No GRPO training, no reward changes, no checkpoint modification.",
        "- Selection uses **this** joint held-out eval only (not old report numbers).",
        "- Held-out not used for hyperparameter tuning beyond selection.",
        "",
    ]
    path = OUT / "FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_final())
