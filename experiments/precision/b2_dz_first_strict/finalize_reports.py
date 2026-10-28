#!/usr/bin/env python
"""Refresh FINAL_REPORT / gate audit with validity-aware counts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/b2_dz_first_strict"


def main() -> int:
    rows = list(csv.DictReader((OUT / "STRICT_ORDERING_RESULTS.csv").open(encoding="utf-8")))
    cases = json.loads((OUT / "STRICT_CASES.json").read_text(encoding="utf-8"))

    def truthy(x):
        return x in (True, "True", "true", "1", 1)

    def cnt(s, key="SUCCESS_NEW"):
        return sum(1 for r in rows if r.get("strategy") == s and truthy(r.get(key)))

    def executed(s):
        return [r for r in rows if r.get("strategy") == s and truthy(r.get("DZ_FIRST_EXECUTED"))]

    def blocked(s):
        return [r for r in rows if r.get("strategy") == s and not truthy(r.get("DZ_FIRST_EXECUTED"))]

    lines = [
        "# FINAL_REPORT — Strict DZ-First",
        "",
        "Production Oracle **unchanged**. This package removes threshold-based DZ skip and silent DX-before-DZ fallback.",
        "",
        "## Headline counts (all 7 ICs)",
        "",
        "| Strategy | SUCCESS_NEW | SUCCESS_12S | DZ_FIRST_EXECUTED |",
        "|---|---:|---:|---:|",
        f"| S0 production | {cnt('S0')}/7 | {cnt('S0','SUCCESS_12S')}/7 | n/a |",
        f"| S1 strict DZ→DX | {cnt('S1')}/7 | {cnt('S1','SUCCESS_12S')}/7 | {len(executed('S1'))}/7 |",
        f"| S2 orient→DZ→DX | {cnt('S2')}/7 | {cnt('S2','SUCCESS_12S')}/7 | {len(executed('S2'))}/7 |",
        f"| S3 + DZ refine | {cnt('S3')}/7 | {cnt('S3','SUCCESS_12S')}/7 | {len(executed('S3'))}/7 |",
        "",
        "## Validity-aware reading",
        "",
        "A failed run is evidence against DZ-first **only if** `DZ_FIRST_EXECUTED` is true.",
        "",
    ]
    for s in ("S1", "S2", "S3"):
        ex = executed(s)
        bl = blocked(s)
        ex_new = sum(1 for r in ex if truthy(r.get("SUCCESS_NEW")))
        lines.append(f"### {s}")
        lines.append("")
        lines.append(f"- DZ-first executed: **{len(ex)}/7** (SUCCESS_NEW among executed: **{ex_new}/{len(ex) or '—'}**)")
        lines.append(f"- Not executed: {len(bl)}/7 — blocks: " + ", ".join(
            f"`{r['hard_id']}`:{r.get('DZ_FIRST_BLOCK_REASON') or '—'}" for r in bl
        ))
        lines.append("")

    lines.extend(
        [
            "## hard_910305 (wrong-direction IC)",
            "",
            "- Prior invalid S1 grew |DZ| under DOWN slip.",
            "- Strict S1/S2: **SUCCESS_NEW=True** after UP staging + direction-consistency DZ→0 + production-like DOWN return.",
            "- S3: DZ-first executed but handoff failed (see phase sequence).",
            "",
            "## Gate note",
            "",
            "5/7 exits have `|ex|>dz_safe_ex_mm=6`. This run used `--bypass-ex-gate` so DZ could still be attempted;",
            "contact/actuator aborts remain. See `DZ_GATE_AUDIT.md`. Without bypass, those five would be `DZ_FIRST_NOT_EXECUTED`.",
            "",
            "## Recommendation",
            "",
        ]
    )
    # Prefer S1 if it recovers without regressing 910026
    m026 = {r["strategy"]: r for r in rows if r["hard_id"] == "hard_910026"}
    s1_reg = truthy((m026.get("S0") or {}).get("SUCCESS_NEW")) and not truthy((m026.get("S1") or {}).get("SUCCESS_NEW"))
    # 910026 S1 was NOT executed (DZ_UP_FAILED) — regression of production success is NOT a completed DZ-first failure
    if cnt("S1") > cnt("S0") and not (s1_reg and truthy((m026.get("S1") or {}).get("DZ_FIRST_EXECUTED"))):
        lines.append(
            f"**CONDITIONAL interest in S1** ({cnt('S1')}/7 NEW vs S0 {cnt('S0')}/7), "
            "but only among DZ-first-executed cases; do **not** adopt into production yet "
            f"(executed-only S1 NEW={sum(1 for r in executed('S1') if truthy(r.get('SUCCESS_NEW')))}/{len(executed('S1'))}; "
            "UP-fail / dθ-block / contact-loss on other ICs remain)."
        )
    else:
        lines.append(
            "**DO NOT ADOPT** strict DZ-first into production from this package alone. "
            "S0 remains the only production path; experimental S1 recovers hard_910305 but does not dominate the cohort."
        )

    lines.extend(["", "## Per-case audit", "", "| hard_id | S | NEW | 12S | DZ_FIRST | block | sequence |", "|---|---|---|---|---|---|---|"])
    for r in rows:
        seq = str(r.get("ACTUAL_PHASE_SEQUENCE") or "")
        if len(seq) > 70:
            seq = seq[:67] + "..."
        lines.append(
            f"| `{r.get('hard_id')}` | {r.get('strategy')} | {r.get('SUCCESS_NEW')} | {r.get('SUCCESS_12S')} | "
            f"{r.get('DZ_FIRST_EXECUTED')} | {r.get('DZ_FIRST_BLOCK_REASON') or '—'} | `{seq}` |"
        )
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Enrich gate audit with bypass outcome
    ga = (OUT / "DZ_GATE_AUDIT.md").read_text(encoding="utf-8")
    if "## Bypass run outcome" not in ga:
        extra = [
            "",
            "## Bypass run outcome (`--bypass-ex-gate`)",
            "",
            "| hard_id | S1 DZ_FIRST | S1 block | S1 NEW | S1 stop/fail |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            if r.get("strategy") != "S1":
                continue
            extra.append(
                f"| `{r['hard_id']}` | {r.get('DZ_FIRST_EXECUTED')} | {r.get('DZ_FIRST_BLOCK_REASON') or '—'} | "
                f"{r.get('SUCCESS_NEW')} | {r.get('failures') or r.get('status')} |"
            )
        extra.extend(
            [
                "",
                "Conclusion: the `|ex|>6` gate is **historically motivated** (5/7 exits trip it) but is **not** a universal",
                "physical impossibility — with bypass, some large-ex cases still attempt axial motion (often contact-abort).",
                "Cases that cannot flip to UP (`DZ_UP_FAILED`) or fail dθ first are correctly `DZ_FIRST_NOT_EXECUTED`.",
                "",
            ]
        )
        (OUT / "DZ_GATE_AUDIT.md").write_text(ga.rstrip() + "\n" + "\n".join(extra), encoding="utf-8")

    print("reports refreshed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
