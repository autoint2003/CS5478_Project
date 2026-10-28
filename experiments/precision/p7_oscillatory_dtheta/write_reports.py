#!/usr/bin/env python
"""Write FINAL_REPORT.md for P7 from comparison / cycle CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p7_oscillatory_dtheta"


def main() -> int:
    cmp = list(csv.DictReader((OUT / "OSCILLATION_COMPARISON.csv").open(encoding="utf-8")))
    cycles = list(csv.DictReader((OUT / "OSCILLATION_CYCLE_TRACES.csv").open(encoding="utf-8")))
    by = {r["variant"]: r for r in cmp}

    def f(r, k, default="—"):
        return r.get(k, default) if r else default

    any_new = any(str(r.get("SUCCESS_NEW")) in ("True", "true") for r in cmp)
    any_tol = any(str(r.get("reached_theta_tol")) in ("True", "true") for r in cmp)
    best = min(cmp, key=lambda r: float(r["theta_final"]))

    lines = [
        "# FINAL_REPORT — P7 Oscillatory dθ (hard_910277)",
        "",
        "Production Oracle and C reconfiguration unchanged. Bounded O0–O3 pilot only.",
        "",
        "## Answers",
        "",
        "### Does oscillation produce a genuine, persistent reduction in θ?",
        "",
    ]
    o0 = by.get("O0", {})
    genuine = []
    for lab in ("O1", "O2", "O3"):
        r = by.get(lab, {})
        improve = float(r.get("net_improve_deg") or 0)
        cyc = [c for c in cycles if c["variant"] == lab]
        slip = any(c.get("classification") == "IRREVERSIBLE_SLIP" for c in cyc)
        genuine.append((lab, improve, slip, r.get("abort_reason"), r.get("theta_final")))
    if any(g[1] >= 0.5 and g[2] for g in genuine):
        lines.append(
            "**Partial / candidate yes** on at least one variant with irreversible-slip classification "
            "and ≥0.5° net abs-θ improvement — see table. Persistence requires SUCCESS_NEW hold."
        )
    elif any(g[1] >= 0.5 for g in genuine):
        lines.append(
            "**Limited net abs-θ change** without clear irreversible-slip wins across cycles; "
            "see CONTACT_ROTATION_ANALYSIS for co-rotation vs elastic rebound."
        )
    else:
        lines.append(
            "**No meaningful persistent reduction** beyond the O0 pivot baseline in this bounded pilot. "
            "Cycles aborted on no-improve / safety before θ≤10°."
        )
    lines.extend(
        [
            "",
            "### Does it achieve SUCCESS_NEW?",
            "",
            f"**{'Yes' if any_new else 'No'}.** "
            f"Best final θ among O0–O3: `{best.get('variant')}` → {best.get('theta_final')}° "
            f"(NEW={best.get('SUCCESS_NEW')}, 12S={best.get('SUCCESS_12S')}). "
            f"Any variant reached θ≤10° transiently: {any_tol}.",
            "",
            "## Baseline (O0 = production `correct_dtheta`)",
            "",
            f"- θ {f(o0,'theta0')} → {f(o0,'theta_final')} (Δ={f(o0,'net_improve_deg')})",
            f"- stop: `{f(o0,'abort_reason')}`",
            f"- SUCCESS_12S={f(o0,'SUCCESS_12S')} SUCCESS_NEW={f(o0,'SUCCESS_NEW')}",
            "",
            "Audit (see `DTHETA_BASELINE_AUDIT.md`): verified `wy` pivot tag rotates the hand "
            "~14° while relative cylinder slip is only ~2° — mostly co-rotation. Stops after "
            "fixed-duration tag because |Δθ|≥2°, not because θ≤10°.",
            "",
            "## Comparison",
            "",
            "| var | mode | θ0 | θ_final | Δθ | cycles | reached≤10 | 12S | NEW | abort |",
            "|---|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for r in cmp:
        lines.append(
            f"| {r['variant']} | {r['mode']} | {r['theta0']} | {r['theta_final']} | "
            f"{r['net_improve_deg']} | {r['n_cycles']} | {r['reached_theta_tol']} | "
            f"{r['SUCCESS_12S']} | {r['SUCCESS_NEW']} | {r.get('abort_reason') or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Ownership",
            "",
            "- Remaining θ≈12° after C is a **Precision orientation** problem.",
            "- Oscillation pilot tests whether small alternating wrist motions can create "
            "irreversible contact slip; it does **not** change C placement.",
            "",
            "## Stop conditions met",
            "",
            "- O0–O3 only; no production Oracle edits; no C changes; no expert regen / RL.",
            "",
            "## Artifacts",
            "",
            "- `DTHETA_BASELINE_AUDIT.md`",
            "- `OSCILLATION_CYCLE_TRACES.csv`",
            "- `OSCILLATION_COMPARISON.csv`",
            "- `CONTACT_ROTATION_ANALYSIS.md`",
            "- `VIDEO_INDEX.md`",
            "- `FINAL_REPORT.md`",
            "",
        ]
    )
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote FINAL_REPORT.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
