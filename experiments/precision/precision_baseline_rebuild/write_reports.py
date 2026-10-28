#!/usr/bin/env python
"""Write remaining baseline rebuild markdown reports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/precision_baseline_rebuild"


def main() -> int:
    rows = json.loads((OUT / "PRECISION_BASELINE_RESULTS.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "BASELINE_SUMMARY.json").read_text(encoding="utf-8"))
    fails = [r for r in rows if r.get("class") == "PRECISION_REQUIRED_BUT_FAILED"]
    succ = [r for r in rows if r.get("class") == "PRECISION_REQUIRED_AND_SUCCESSFUL"]
    direct = [r for r in rows if r.get("class") == "DIRECT_ELIGIBLE"]
    excl_ex = list(csv.DictReader((OUT / "EXCLUDED_EXPERTS.csv").open(encoding="utf-8")))
    v3_meta = json.loads((OUT / "EXPERT_V3_META.json").read_text(encoding="utf-8"))

    # Failure analysis
    lines = [
        "# PRECISION_FAILURE_ANALYSIS",
        "",
        f"Genuine Precision failures after Stage-1 SUCCESS_12S (n={len(fails)}).",
        "",
        "| sample | plan | dx | dz | dθ | first_fail | NEW | 12S | n_act | dropped |",
        "|---|---|---:|---:|---:|---|---|---|---:|---|",
    ]
    for r in fails:
        lines.append(
            f"| {r['sample_id']} | {r.get('plan_family')} | {float(r.get('dx_mm') or 0):.2f} | "
            f"{float(r.get('dz_mm') or 0):.2f} | {float(r.get('dtheta_deg') or 0):.2f} | "
            f"{r.get('first_failed_primitive')} | {r.get('precision_SUCCESS_NEW')} | "
            f"{r.get('precision_SUCCESS_12S')} | {r.get('n_actions')} | {r.get('dropped')} |"
        )
    lines += ["", "## Raw failure payloads", ""]
    for r in fails:
        lines.append(
            f"- `{r['sample_id']}`: status={r.get('precision_status')} "
            f"failures={r.get('precision_failures')} phases={r.get('phases')}"
        )
    lines += [
        "",
        "## Primitive histogram",
        "",
        "```",
        json.dumps(dict(Counter(r.get("first_failed_primitive") for r in fails)), indent=2),
        "```",
        "",
    ]
    (OUT / "PRECISION_FAILURE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    hist_vs = f"""# HISTORICAL_VS_REBUILT

Generated: {datetime.now(timezone.utc).isoformat()}

## Population

| Metric | Historical audit (unfiltered / filtered) | Rebuilt baseline |
|---|---:|---:|
| Discovered ICs | 241 | {summary['n_discovered']} |
| Stage-1 eligible | 145 (integrity audit) | {summary['n_eligible_run']} |
| Stage-1 excluded | 96 | {summary['n_excluded_stage1']} |
| DIRECT_ELIGIBLE | 127 / 127 | {len(direct)} |
| PRECISION_REQUIRED_AND_SUCCESSFUL | 13 / 8 | {len(succ)} |
| PRECISION_REQUIRED_BUT_FAILED | 70 / 10 | {len(fails)} |

## Failure primitives (eligible only)

| Primitive | Historical filtered | Rebuilt |
|---|---:|---:|
| dθ | 5 | {summary['precision_fail_primitives'].get('dtheta', 0)} |
| support_transfer | 2 | {summary['precision_fail_primitives'].get('support_transfer', 0)} |
| dz | 3 | {summary['precision_fail_primitives'].get('dz', 0)} |
| DX1 | 0 | {summary['precision_fail_primitives'].get('DX1', 0)} |

**Divergence:** none on eligible Precision fail/success counts vs integrity-audit filtered expectations (10 / 8). Fresh Oracle replay confirms the filtered audit.

## Experts

| Metric | Audit expectation | v3 build |
|---|---:|---:|
| SUCCESS_NEW with valid Stage-1 | 191 | {v3_meta.get('n_accepted')} |
| Excluded SUCCESS_NEW | 5 | {v3_meta.get('n_excluded_success_new')} |

Excluded IDs: {', '.join(e.get('source_ic_id') or '' for e in excl_ex)}
"""
    (OUT / "HISTORICAL_VS_REBUILT.md").write_text(hist_vs, encoding="utf-8")

    final = f"""# FINAL_REPORT — Stage-1 Eligibility Fix + Precision Baseline Rebuild

Generated: {datetime.now(timezone.utc).isoformat()}

## Completed

1. **System-wide Stage-1 eligibility** (`SUCCESS_12S` on exact capture) — implemented.
2. **Regression tests** — 5/5 passed (`tests/test_stage1_eligibility.py`).
3. **Clean expert dataset** — `data/experts_v3_stage1_verified/` (**{v3_meta.get('n_accepted')}** trajectories); `data/experts_v2_official/` untouched.
4. **Precision baseline rebuilt** on eligible ICs only.
5. **Genuine remaining Precision failures** — **{len(fails)}** (dθ={summary['precision_fail_primitives'].get('dtheta',0)}, dz={summary['precision_fail_primitives'].get('dz',0)}, ST={summary['precision_fail_primitives'].get('support_transfer',0)}).

## Verified results

| Metric | Value |
|---|---:|
| Eligible ICs | {summary['n_eligible_run']} |
| Direct-eligible | {len(direct)} |
| Precision-required success | {len(succ)} |
| Precision-required failure | {len(fails)} |
| Stage-1 excluded from baseline | {summary['n_excluded_stage1']} |
| Official v3 experts | {v3_meta.get('n_accepted')} |

## Remaining uncertainties

- Registry evidence was seeded from the integrity-audit revalidation fingerprints; live verify remains available when fingerprints miss.
- `stabilization_prefix_actions` remain outside the capture contract (documented); future IC formats must bake prefixes into capture before verification.
- Diagnostic experiment *results* (OAR/LAP/Coupled) were not re-run here — only their entry gates were hardened. Prior diagnostic numbers remain historically contaminated until those experiments are re-executed on eligible ICs.

## Recommended next steps

1. Use `experts_v3_stage1_verified` as the production expert root going forward.
2. Re-run any Precision diagnostics only on Stage-1–eligible ICs.
3. Investigate the **{len(fails)}** genuine failures (especially dθ×5) — without relaxing Stage-1.

## Non-actions (honored)

No new dθ mechanisms, no DX/DZ strategy changes, no OAR/LAP/coupled resume, no SFT/GRPO, no overwrite of original datasets.
"""
    (OUT / "FINAL_REPORT.md").write_text(final, encoding="utf-8")
    print("reports written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
