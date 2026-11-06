"""Assemble FINAL_REPORT.md from P59 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p59_expanded_c_grpo.paths import OUT


def main():
    man = json.loads((OUT / "TRAIN_IC_MANIFEST.json").read_text(encoding="utf-8"))
    train = json.loads((OUT / "P59_TRAIN_SUMMARY.json").read_text(encoding="utf-8"))
    val_path = OUT / "VALIDATION_after_pass2.json"
    if not val_path.is_file():
        val_path = OUT / "VALIDATION.json"
    val = json.loads(val_path.read_text(encoding="utf-8")) if val_path.is_file() else {}
    hold = {}
    if (OUT / "HELDOUT_EVAL.json").is_file():
        hold = json.loads((OUT / "HELDOUT_EVAL.json").read_text(encoding="utf-8"))
    stage = {}
    if (OUT / "C_STAGE_DIAGNOSTICS.json").is_file():
        stage = json.loads((OUT / "C_STAGE_DIAGNOSTICS.json").read_text(encoding="utf-8"))

    rc = val.get("reports_c") or {}
    lines = [
        "# FINAL_REPORT — P59 Expanded C-Focused GRPO-64",
        "",
        "## Verdict",
        "",
    ]
    def msp(name):
        r = rc.get(name) or {}
        return r.get("MSP_RELAXED")

    lines.append("| Model | C val MSP |")
    lines.append("|-------|----------:|")
    for n in ("P57_A", "P58_pass2", "P59_pass1", "P59_pass2"):
        v = msp(n)
        if v is not None:
            lines.append(f"| {n} | {v:.4f} |")
    best = hold.get("best_ckpt")
    lines += [
        "",
        f"Validation-selected checkpoint: **{best}**.",
        "",
        "## Setup",
        "",
        f"- Init: P58_pass2; {man['n_train']} C ICs (16 P58 core + 32 added); 2 passes × 64 = {train.get('n_rollouts')} rollouts",
        f"- Unchanged: M111, P57-A reward, HANDOFF, SUCCESS_RELAXED_12S, frozen KL ref, no expert BC / stab reward",
        "",
        "## Stage diagnostics (train)",
        "",
        f"`{stage}`",
        "",
        "## Held-out",
        "",
    ]
    if hold.get("reports"):
        lines.append("| Model | MSP | A | B | C | D | Zero |")
        lines.append("|-------|----:|--:|--:|--:|--:|-----:|")
        for name, rep in hold["reports"].items():
            if "error" in rep:
                continue
            bc = rep.get("by_class") or {}
            lines.append(
                f"| {name} | {rep.get('MSP_RELAXED_12S'):.4f} | "
                f"{(bc.get('A') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
                f"{(bc.get('B') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
                f"{(bc.get('C') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
                f"{(bc.get('D') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
                f"{rep.get('zero_success_ics')} |"
            )
        lines.append("")
        lines.append(f"Paired: `{hold.get('paired')}`")
    # Success criteria checklist
    p58 = msp("P58_pass2")
    p59b = msp(best) if best else None
    lines += [
        "",
        "## Success criteria",
        "",
        f"1. C val beyond P58_pass2: **{'YES' if (p59b is not None and p58 is not None and p59b > p58) else 'NO/SEE TABLE'}**",
        "2. Gains on newly added ICs: see TRAINING_REPORT core vs added.",
        "3. ABD regression: see VALIDATION_REPORT.",
        "4. Held-out overall MSP preserved/improved: see HELDOUT_EVAL.",
        "",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote FINAL_REPORT.md")


if __name__ == "__main__":
    main()
