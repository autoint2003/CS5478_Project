#!/usr/bin/env python
"""Write P3 markdown deliverables from paired/ablation JSON+CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/diagnostics/b2_coupled_dx_dz"
PRIOR = ROOT / "results/diagnostics/precision_coupled_feasibility"


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def truthy(x) -> bool:
    return str(x).lower() in ("1", "true", "yes")


def main() -> int:
    cases = json.loads((OUT / "PAIRED_CASES.json").read_text(encoding="utf-8"))
    paired = load_csv(OUT / "PAIRED_RESULTS.csv")
    abl = load_csv(OUT / "ABLATION_RESULTS.csv")
    summary = json.loads((OUT / "RUN_SUMMARY.json").read_text(encoding="utf-8"))

    # --- BASELINE_REPRODUCTION.md ---
    lines = [
        "# BASELINE_REPRODUCTION — Seven Verified B2 ICs",
        "",
        "Stage-1 exits rebuilt via Plan-B known `b1_tokens` + hard IC snapshots "
        "(same method as `render_b_precision_required_videos.py`).",
        "",
        "Production Precision Oracle: `use_verified_lateral_fallback=False`, `evaluate_handoff=True`.",
        "",
        "| hard_id | Stage-1 eligible | Expected status | V0 status | match | SUCCESS_NEW | SUCCESS_12S | skills |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        hid = c["hard_id"]
        v0 = (c.get("variants") or {}).get("V0") or {}
        elig = (c.get("stage1_eligibility") or {}).get("eligible")
        exp = c.get("expected_oracle_status")
        match = v0.get("status") == exp
        lines.append(
            f"| `{hid}` | {elig} | `{exp}` | `{v0.get('status')}` | {match} | "
            f"{v0.get('SUCCESS_NEW')} | {v0.get('SUCCESS_12S')} | `{c.get('pilot_skills')}` |"
        )
    lines += [
        "",
        f"V0 status matches historical expected on "
        f"**{summary.get('v0_match_expected')}/{summary.get('n_cases')}** cases.",
        "",
        "Raw exits: `stage1_exits/`. V0 action tapes: `raw/*__V0__actions.npz`.",
        "",
    ]
    (OUT / "BASELINE_REPRODUCTION.md").write_text("\n".join(lines), encoding="utf-8")

    # --- DX_DZ_THETA_CAUSAL_ANALYSIS.md ---
    lines = [
        "# DX_DZ_THETA_CAUSAL_ANALYSIS",
        "",
        "Traces from replaying each variant's action tape (`raw/*__trace.json`).",
        "θ mode uses object–hand relative orientation vs `dtheta_deg` rise.",
        "",
        "| hard_id | var | max Δθ↑ | θ mode | DZ@DX start→end | DZ return drift | contact losses |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for r in paired:
        lines.append(
            f"| `{r.get('hard_id')}` | {r.get('variant')} | "
            f"{r.get('max_dtheta_deterioration')} | {r.get('theta_mode')} | "
            f"{r.get('dz_at_dx_start')}→{r.get('dz_at_dx_end')} | "
            f"{r.get('dz_return_drift')} | {r.get('contact_loss_events')} |"
        )
    lines += [
        "",
        "## Per-case notes (V0)",
        "",
    ]
    for c in cases:
        hid = c["hard_id"]
        tr = (c.get("traces") or {}).get("V0") or {}
        v0 = (c.get("variants") or {}).get("V0") or {}
        ee0 = (v0.get("errors_entry") or {})
        lines.append(f"### `{hid}`")
        lines.append(
            f"- Entry: dx={ee0.get('dx_mm')}, dz={ee0.get('dz_mm')}, dθ={ee0.get('dtheta_deg')}, "
            f"skills=`{c.get('pilot_skills')}`."
        )
        lines.append(
            f"- θ mode={tr.get('theta_mode')}; max dθ deterioration={tr.get('max_dtheta_deterioration')}; "
            f"rel-orient rise={tr.get('rel_orient_rise_to_peak')}."
        )
        lines.append(
            f"- DZ during DX window: {tr.get('dz_at_dx_start')} → {tr.get('dz_at_dx_end')}; "
            f"return/regrip DZ drift={tr.get('dz_return_drift')}."
        )
        lines.append(
            f"- Failure class: status=`{v0.get('status')}`; failures={v0.get('failures')}; "
            f"dropped={v0.get('dropped')}."
        )
        lines.append(f"- Coupled relevant at entry: {c.get('coupled_relevant_at_entry')}.")
        lines.append("")
    (OUT / "DX_DZ_THETA_CAUSAL_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    # --- DZ_STOP_AND_RETURN_ANALYSIS.md ---
    prior_brake = ""
    bp = PRIOR / "BRAKING_DIAGNOSIS.md"
    if bp.is_file():
        prior_brake = bp.read_text(encoding="utf-8")[:1200]
    lines = [
        "# DZ_STOP_AND_RETURN_ANALYSIS",
        "",
        "## Production stop law (traced, not assumed)",
        "",
        "Inward DualLoop (`mechanics/axial_sliding.py` `_inward_slip_staged`):",
        "",
        "- `use_euclidean_stop=False` while fingertips-up.",
        "- `axis_tol = clip(sqrt(working_pos² − ex² − ey²), 3, 8)` with `working_pos≈12`.",
        "- `brake_margin_mm≈1.0`.",
        "- Therefore residual |DZ| often lands near **3–8 mm**, commonly ~8 mm when lateral budget is large — "
        "**not** a hard-coded 15 mm constant. Prior report used ~10 mm colloquially for the same band.",
        "",
        "Prior braking note (excerpt):",
        "",
        "```",
        prior_brake,
        "```",
        "",
        "## Return / regrip DZ drift on this B2 cohort",
        "",
        "| hard_id | V0 DZ return drift | V1 | V2 | V2 compensation |",
        "|---|---:|---:|---:|---|",
    ]
    for c in cases:
        hid = c["hard_id"]
        def drift(v):
            return ((c.get("traces") or {}).get(v) or {}).get("dz_return_drift")
        cr = (((c.get("variants") or {}).get("V2") or {}).get("coupled_result") or {})
        lines.append(
            f"| `{hid}` | {drift('V0')} | {drift('V1')} | {drift('V2')} | "
            f"{cr.get('compensation_applied')} wp_eff={cr.get('working_pos_eff_mm')} |"
        )
    lines += [
        "",
        "V2 only tightens Euclidean `working_pos` when a probe shows return **worsens** |DZ|, "
        "and rejects adjustments that cause grasp loss.",
        "",
    ]
    (OUT / "DZ_STOP_AND_RETURN_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    # --- CONTACT_FORCE_ANALYSIS.md ---
    lines = [
        "# CONTACT_FORCE_ANALYSIS",
        "",
        "Contact / force asymmetry from replay traces (MuJoCo residuals + support mode).",
        "",
        "| hard_id | var | contact_loss_events | final bil | θ mode | note |",
        "|---|---|---:|---|---|---|",
    ]
    for r in paired:
        hid = r.get("hard_id")
        var = r.get("variant")
        c = next(x for x in cases if x["hard_id"] == hid)
        v = (c.get("variants") or {}).get(var) or {}
        lines.append(
            f"| `{hid}` | {var} | {r.get('contact_loss_events')} | "
            f"{(v.get('errors_exit') or {}).get('bilateral')} | {r.get('theta_mode')} | "
            f"coupled_attempted={r.get('coupled_attempted')} |"
        )
    lines += [
        "",
        "Interpretation guide:",
        "",
        "- Rising dθ with flat object–hand relative angle → **rigid co-rotation** (not contact torque slip).",
        "- Rising relative angle with contact force asymmetry → **relative slip / lever torque** candidate.",
        "- Contact loss events mid-tape → **grasp loss / support transfer abort**.",
        "",
        "See per-step series in `raw/*__trace.json` (`fn_L`, `fn_R`, `fn_asym`, `Fn_z_*`, `support_side`).",
        "",
    ]
    (OUT / "CONTACT_FORCE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    # --- FINAL_RECOMMENDATION.md ---
    def count_new(var):
        return sum(1 for r in paired if r.get("variant") == var and truthy(r.get("SUCCESS_NEW")))

    v0n, v1n, v2n = count_new("V0"), count_new("V1"), count_new("V2")
    # recoveries among the six historical fails
    recovers = []
    regress = []
    for hid in [c["hard_id"] for c in cases]:
        def g(var, _hid=hid):
            rows = [r for r in paired if r.get("hard_id") == _hid and r.get("variant") == var]
            return rows[0] if rows else {}

        v0, v1, v2 = g("V0"), g("V1"), g("V2")
        if hid == "hard_910026":
            if not truthy(v1.get("SUCCESS_NEW")):
                regress.append(("V1", hid))
            if not truthy(v2.get("SUCCESS_NEW")):
                regress.append(("V2", hid))
        else:
            if not truthy(v0.get("SUCCESS_NEW")) and truthy(v1.get("SUCCESS_NEW")):
                recovers.append(("V1", hid))
            if not truthy(v0.get("SUCCESS_NEW")) and truthy(v2.get("SUCCESS_NEW")):
                recovers.append(("V2", hid))

    adopt = (
        len(recovers) > 0
        and not regress
        and (v1n > v0n or v2n > v0n)
    )
    lines = [
        "# FINAL_RECOMMENDATION — B2 Coupled DX+DZ (P3)",
        "",
        "**Stop condition:** seven-case paired experiment + analysis only. "
        "Production Oracle, Stage-1 data, expert merge, and training were **not** modified.",
        "",
        "## Counts (full trajectory)",
        "",
        f"| Variant | SUCCESS_NEW | SUCCESS_12S (among NEW) |",
        f"|---|---:|---:|",
        f"| V0 production | {v0n}/7 | "
        f"{sum(1 for r in paired if r.get('variant')=='V0' and truthy(r.get('SUCCESS_12S')))}/7 |",
        f"| V1 prior Coupled | {v1n}/7 | "
        f"{sum(1 for r in paired if r.get('variant')=='V1' and truthy(r.get('SUCCESS_12S')))}/7 |",
        f"| V2 coupled + return-aware stop | {v2n}/7 | "
        f"{sum(1 for r in paired if r.get('variant')=='V2' and truthy(r.get('SUCCESS_12S')))}/7 |",
        "",
        f"- Recoveries of historical fails: {recovers or 'none'}",
        f"- Regressions of `hard_910026`: {regress or 'none'}",
        "",
        "## Decision criteria",
        "",
        f"1. Improves full-trajectory SUCCESS_NEW on these B2 ICs? "
        f"**{'Yes' if (v1n > v0n or v2n > v0n) else 'No'}** (V0={v0n}, V1={v1n}, V2={v2n}).",
        f"2. No regression on `hard_910026`? **{'Yes' if not regress else 'No — ' + str(regress)}**.",
        "3. Legal physically valid actions? Experimental path uses existing DualLoop / orientation / restore_DOWN only (no teleport).",
        "4. Final 12s HANDOFF: see SUCCESS_12S column above.",
        "5. Reproducible: fixed Stage-1 exit cache + deterministic Oracle config.",
        "6. No privileged snapshot manipulation beyond restoring the verified Stage-1 exit.",
        "",
        f"## Recommendation: **{'CONDITIONAL ADOPT (see caveats)' if adopt else 'DO NOT ADOPT Coupled DX+DZ into production'}**",
        "",
        "Do **not** claim generalization beyond these seven verified B2 cases.",
        "",
        "If primitive-only metrics improved without SUCCESS_NEW, that is **not** counted as success "
        "(see `primitive_only` in `PAIRED_CASES.json`).",
        "",
        "If failures are dominated by dθ-only cases (`hard_910263`, `hard_910277`) or DZ-stop/return "
        "rather than DX–DZ coupling, treat those as separate problems.",
        "",
        "## Ablations",
        "",
    ]
    if abl:
        lines.append("| hard_id | ablation | SUCCESS_NEW | status | dropped |")
        lines.append("|---|---|---|---|---|")
        for r in abl:
            lines.append(
                f"| `{r.get('hard_id')}` | {r.get('ablation')} | {r.get('SUCCESS_NEW')} | "
                f"`{r.get('status')}` | {r.get('dropped')} |"
            )
    else:
        lines.append("_No ablation rows._")
    lines.append("")
    (OUT / "FINAL_RECOMMENDATION.md").write_text("\n".join(lines), encoding="utf-8")

    # Ensure VIDEO_INDEX exists even before render
    if not (OUT / "VIDEO_INDEX.md").is_file():
        (OUT / "VIDEO_INDEX.md").write_text(
            "# VIDEO_INDEX\n\n_Run `render_videos.py` after paired runs._\n", encoding="utf-8"
        )

    print("reports written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
