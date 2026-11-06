"""Assemble comparison / P66 recommendations / FINAL_REPORT."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p65_ppo_continuation.paths import OUT, P64_OUT


def _msp_table_row(name, e, note=""):
    bc = e.get("by_class") or {}

    def g(c):
        return float((bc.get(c) or {}).get("MSP_RELAXED") or float("nan"))

    return (
        f"| {name} | {float(e.get('MSP_RELAXED', float('nan'))):.4f} | "
        f"{g('A'):.4f} | {g('B'):.4f} | {g('C'):.4f} | {g('D'):.4f} | "
        f"{e.get('zero_success_ics', '')} | {note} |"
    )


def write_all(*, best: dict | None = None, p66: list | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Collect evals
    rows = []
    # P64 baselines from files
    for tag, path in (
        ("P64_step1", P64_OUT / "P64_step1.json"),
        ("P64_step4", P64_OUT / "P64_step4.json"),
    ):
        if path.is_file():
            d = json.loads(path.read_text(encoding="utf-8"))
            rows.append((tag, d.get("eval") or {}, str(d.get("actor") or "")))
    for tag in ("P65_step5", "P65_step6"):
        p = OUT / f"{tag}.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            rows.append((tag, d.get("eval") or {}, str(d.get("actor") or "")))

    if best is None:
        if (OUT / "BEST_CHECKPOINT.json").is_file():
            best = json.loads((OUT / "BEST_CHECKPOINT.json").read_text(encoding="utf-8"))
        elif rows:
            best_t = max(rows, key=lambda x: float(x[1].get("MSP_RELAXED") or -1))
            best = {"tag": best_t[0], "msp": best_t[1].get("MSP_RELAXED"), "ckpt": best_t[2]}

    lines = [
        "# CHECKPOINT_COMPARISON",
        "",
        "Primary panel = frozen P61 SHA-256 development (not final-test).",
        "",
        "| Checkpoint | MSP | A | B | C | D | Zero | Notes |",
        "|------------|----:|--:|--:|--:|--:|-----:|-------|",
    ]
    # include P63-A / P62 from P64 baselines if present
    b64 = P64_OUT / "BASELINES.json"
    if b64.is_file():
        bb = json.loads(b64.read_text(encoding="utf-8"))
        if "P62_epoch3" in bb:
            lines.append(_msp_table_row("P62_epoch3", bb["P62_epoch3"], "ref"))
        if "P63_PPO_A" in bb:
            lines.append(_msp_table_row("P63 PPO-A", bb["P63_PPO_A"], "cum=2048"))
    for tag, e, _ in rows:
        lines.append(_msp_table_row(tag, e, ""))
    lines += ["", f"**Selected:** `{best.get('tag') if best else None}` MSP={best.get('msp') if best else None}", ""]

    # panel evals
    panel_files = sorted(OUT.glob("PANEL_EVAL_*.json"))
    if panel_files:
        lines += ["## Extra development panel", ""]
        lines.append("| Checkpoint | Panel MSP | A | B | C | D | Zero |")
        lines.append("|------------|----------:|--:|--:|--:|--:|-----:|")
        for pf in panel_files:
            d = json.loads(pf.read_text(encoding="utf-8"))
            lines.append(_msp_table_row(d["tag"], d, "").replace(" |  |", " |"))
        lines.append("")

    (OUT / "CHECKPOINT_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")

    # P66 recommendations
    if p66 is None:
        p66 = _default_p66()
    (OUT / "P66_RECOMMENDATIONS.md").write_text("\n".join(p66), encoding="utf-8")

    # FINAL_REPORT skeleton filled by run_p65 preferably
    if not (OUT / "FINAL_REPORT.md").is_file() or True:
        final = [
            "# FINAL_REPORT — P65 PPO Continuation & A/C Diagnosis",
            "",
            f"Best checkpoint: **{best.get('tag') if best else 'TBD'}**",
            "",
            "See `A_REGRESSION_DIAGNOSIS.md`, `C_BOTTLENECK_DIAGNOSIS.md`,",
            "`TRAINING_EXPOSURE.md`, `CHECKPOINT_COMPARISON.md`, `P66_RECOMMENDATIONS.md`.",
            "",
        ]
        (OUT / "FINAL_REPORT.md").write_text("\n".join(final), encoding="utf-8")


def _default_p66() -> list[str]:
    return [
        "# P66_RECOMMENDATIONS",
        "",
        "At most two targeted proposals (not implemented in P65).",
        "",
        "## 1. C exploration / hard-IC curriculum (preferred)",
        "",
        "- **Evidence:** C MSP stuck ~0.02; diagnostic buckets likely dominated by FAILED_RECATCH / early drop.",
        "- **Expected MSP impact:** Medium–high if C moves off floor; overall MSP currently B/D-driven.",
        "- **A/B/D risk:** Medium — need anchoring or class-capped updates.",
        "- **Cost:** Medium (extra C rollouts + eval).",
        "- **Ablation:** curriculum on vs off at matched traj budget.",
        "",
        "## 2. Class-balanced sampling + mild multi-class KL anchor to P64_step1",
        "",
        "- **Evidence:** A peaked at step1 then declined while overall rose; exposure is stratified slots but may under-repeat hard A ICs.",
        "- **Expected impact:** Stabilize A while preserving B/D gains.",
        "- **Risk:** Low–medium if KL cap small.",
        "- **Cost:** Low.",
        "- **Ablation:** balanced sampling ± anchor.",
        "",
        "**Single most promising for P66:** proposal 1 (C exploration/curriculum), with A-anchor as safety.",
        "",
    ]


if __name__ == "__main__":
    write_all()
