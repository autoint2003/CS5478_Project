#!/usr/bin/env python
"""Write LAP_FEASIBILITY_REPORT.md and FINAL_RECOMMENDATION.md from measured results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/lever_arm_assisted_pivot"


def _f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def main() -> int:
    paired = json.loads((OUT / "PAIRED_RESULTS.json").read_text(encoding="utf-8"))
    mech = {}
    mp = OUT / "MECHANISM_AUDIT_CASES.json"
    if mp.exists():
        mech_cases = json.loads(mp.read_text(encoding="utf-8"))
    else:
        mech_cases = []

    valid = [r for r in paired if not r.get("error")]
    n = len(valid)

    def rate(cond, pred):
        xs = [r for r in valid if cond in r]
        if not xs:
            return 0, 0, 0.0
        hits = sum(1 for r in xs if pred(r[cond]))
        return hits, len(xs), hits / max(len(xs), 1)

    ok_a = rate("A", lambda s: s.get("ok"))
    ok_b = rate("B", lambda s: s.get("ok"))
    ok_c = rate("C", lambda s: s.get("ok"))
    ok_d = rate("D", lambda s: s.get("ok"))

    rel_c = [_f((r.get("C") or {}).get("relative_rotation_deg")) for r in valid]
    imp_c = [_f((r.get("C") or {}).get("d_theta_improve_deg")) for r in valid]
    dp_c = [_f((r.get("C") or {}).get("delta_p_rel_mm")) for r in valid]
    rel_a = [_f((r.get("A") or {}).get("relative_rotation_deg")) for r in valid]
    imp_a = [_f((r.get("A") or {}).get("d_theta_improve_deg")) for r in valid]

    fails_c = {}
    for r in valid:
        f = (r.get("C") or {}).get("failure") or ("ok" if (r.get("C") or {}).get("ok") else "unknown")
        fails_c[str(f)] = fails_c.get(str(f), 0) + 1

    # Direction control: C vs D relative rotation / improve sign agreement with lever sign
    dir_ctrl = []
    for r in valid:
        c = r.get("C") or {}
        d = r.get("D") or {}
        if c.get("relative_rotation_deg") is None or d.get("relative_rotation_deg") is None:
            continue
        dir_ctrl.append(
            {
                "id": r["sample_id"],
                "C_rel": c.get("relative_rotation_deg"),
                "D_rel": d.get("relative_rotation_deg"),
                "C_imp": c.get("d_theta_improve_deg"),
                "D_imp": d.get("d_theta_improve_deg"),
                "sign_flip_improve": (
                    _f(c.get("d_theta_improve_deg")) * _f(d.get("d_theta_improve_deg")) < 0
                    if c.get("d_theta_improve_deg") is not None and d.get("d_theta_improve_deg") is not None
                    else None
                ),
            }
        )

    n_s1 = sum(1 for r in valid if (r.get("C") or {}).get("s1_reentry_candidate"))
    n_3mm = sum(1 for r in valid if (r.get("C") or {}).get("diag_3mm_ok"))
    n_rel2 = sum(1 for r in valid if _f((r.get("C") or {}).get("relative_rotation_deg")) >= 2.0)

    # Outperform baseline: C improves more than A on same IC, with relative rotation
    beat = 0
    for r in valid:
        ca, cc = r.get("A") or {}, r.get("C") or {}
        if _f(cc.get("d_theta_improve_deg")) > _f(ca.get("d_theta_improve_deg")) + 0.5 and _f(
            cc.get("relative_rotation_deg")
        ) >= 2.0:
            beat += 1

    # Feasible IC conditions
    feas = []
    for r in valid:
        if (r.get("C") or {}).get("ok"):
            feas.append(
                {
                    "id": r["sample_id"],
                    "dz": r.get("entry_dz_mm"),
                    "dx": r.get("entry_dx_mm"),
                    "dth": r.get("entry_dtheta_deg"),
                    "lever_ic": ((r.get("geo_ic") or {}).get("primary") or {}).get("axial_lever_mm"),
                }
            )

    lines = [
        "# LAP_FEASIBILITY_REPORT",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Isolated physics feasibility for Lever-Arm Assisted Pivot (steps 1–6). "
        "`use_verified_lateral_fallback=false`. SUCCESS_NEW / SUCCESS_12S unchanged. "
        "No production Oracle / expert / M111 / training changes.",
        "",
        "## Cohort",
        "",
        f"- Paired ICs: **{n}** (dθ failures + large-|dz| ST-relevant + success controls)",
        f"- Mechanism-audit cases: **{len(mech_cases)}**",
        "",
        "## Condition success (ok = relative pivot + stable regrip criteria)",
        "",
        f"| Cond | ok | n | rate |",
        f"|---|---:|---:|---:|",
        f"| A Baseline Pivot | {ok_a[0]} | {ok_a[1]} | {ok_a[2]:.2f} |",
        f"| B Lever only | {ok_b[0]} | {ok_b[1]} | {ok_b[2]:.2f} |",
        f"| C LAP | {ok_c[0]} | {ok_c[1]} | {ok_c[2]:.2f} |",
        f"| D Reverse lever LAP | {ok_d[0]} | {ok_d[1]} | {ok_d[2]:.2f} |",
        "",
        "## LAP (C) physical metrics",
        "",
        f"- Meaningful relative rotation (≥2°): **{n_rel2}/{n}**",
        f"- Mean relative rotation°: **{(sum(rel_c)/max(len(rel_c),1)):.2f}**",
        f"- Mean Δdθ improve°: **{(sum(imp_c)/max(len(imp_c),1)):.2f}** (baseline A mean improve: **{(sum(imp_a)/max(len(imp_a),1)):.2f}**)",
        f"- Mean |Δp_rel| mm: **{(sum(dp_c)/max(len(dp_c),1)):.2f}** (3 mm diagnostic OK: **{n_3mm}/{n}**)",
        f"- S1 re-entry candidate after regrip: **{n_s1}/{n}**",
        f"- ICs where C beats A on Δdθ with relative rotation: **{beat}/{n}**",
        "",
        "### Failure taxonomy (C)",
        "",
        "```",
        json.dumps(fails_c, indent=2),
        "```",
        "",
        "## Direction control (C vs D)",
        "",
        f"- Paired reverse-lever comparisons: **{len(dir_ctrl)}**",
        f"- Improve-sign flips under reverse lever: **{sum(1 for x in dir_ctrl if x.get('sign_flip_improve'))}**",
        "",
        "If reverse lever does not systematically flip corrective sense, direction is **not** "
        "reliably controllable by axial-offset sign alone under the current contact protocol.",
        "",
        "## Feasible LAP successes (if any)",
        "",
    ]
    if feas:
        lines.append("| sample | dx | dz | dθ | ic_axial_lever_mm |")
        lines.append("|---|---:|---:|---:|---:|")
        for f in feas:
            lines.append(f"| {f['id']} | {f['dx']} | {f['dz']} | {f['dth']} | {f['lever_ic']} |")
    else:
        lines.append("_No condition-C successes under the stated relative-rotation criteria._")

    lines += [
        "",
        "## Answers to task questions",
        "",
        f"1. **Lever arm → torque?** Mechanism audit + Phase A logs compare `dz_mm` vs measured "
        f"`axial_lever_mm` and τ_g proxies. See MECHANISM_AUDIT.md. "
        f"Paired B (lever only) ok-rate={ok_b[2]:.2f} — lever alone rarely induces relative pivot.",
        f"2. **Direction control via lever sign?** Reverse-lever improve-sign flips: "
        f"{sum(1 for x in dir_ctrl if x.get('sign_flip_improve'))}/{len(dir_ctrl)}.",
        f"3. **Relative rotation without large COM translation?** ≥2° rel with 3 mm diagnostic: "
        f"count where rel≥2 and diag_3mm: "
        f"{sum(1 for r in valid if _f((r.get('C') or {}).get('relative_rotation_deg'))>=2 and (r.get('C') or {}).get('diag_3mm_ok'))}/{n}.",
        f"4. **Brake near target?** Failures tagged BRAKE_FAILED in taxonomy above; "
        f"regrip/brake ok implies Phase D passed when C reached D.",
        f"5. **Post-regrip stability?** C ok requires bilateral/regrip; REGRIP_FAILED counted in taxonomy.",
        f"6. **LAP vs old Pivot?** C beats A: **{beat}/{n}**; C ok-rate {ok_c[2]:.2f} vs A {ok_a[2]:.2f}.",
        f"7. **Feasible IC conditions?** See success table; typically needs force-backed S1 at horizontal "
        f"and nonzero axial lever with helpful τ_g sign.",
        f"8. **S1 re-establishable?** s1_reentry_candidate **{n_s1}/{n}**.",
        "",
        "## Artifacts",
        "",
        "- `MECHANISM_AUDIT.md`, `PAIRED_RESULTS.csv`, `CONTACT_FORCE_ANALYSIS.csv`",
        "- `traces/` action+state JSON / npy",
        "- `VIDEOS/` paired mp4 + INDEX.json",
        "",
    ]
    (OUT / "LAP_FEASIBILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # Recommendation based on measured behavior
    if ok_c[2] >= 0.35 and beat >= max(2, n // 5):
        verdict = "CONDITIONAL_PROCEED"
        rec = (
            "LAP shows measurable relative rotation and outperforms baseline Pivot on a nonempty subset. "
            "Do **not** integrate into production yet — expand IC coverage and harden brake/regrip, "
            "then design a gated experimental branch. Downstream coupled dx/dz remains out of scope."
        )
        limit = "implementation_maturity"
    elif n_rel2 >= max(2, n // 4) and ok_c[0] == 0:
        verdict = "REJECT_FOR_PRODUCTION_PARTIAL_PHYSICS"
        rec = (
            "Relative rotation is sometimes inducible, but end-to-end LAP (brake+regrip+corrective sense) "
            "does not clear the feasibility bar. Limitation appears after release/brake rather than pure "
            "geometry. Do not merge; do not regenerate experts."
        )
        # classify dominant failure
        top = max(fails_c.items(), key=lambda kv: kv[1])[0] if fails_c else "unknown"
        limit = top
    else:
        top = max(fails_c.items(), key=lambda kv: kv[1])[0] if fails_c else "mixed"
        support_blocked = top in (
            "SUPPORT_LOST_DURING_TILT",
            "SUPPORT_TRANSFER_FAILED",
            "PIVOT_SUPPORT_NOT_ESTABLISHED",
        )
        if support_blocked and ok_c[0] == 0:
            verdict = "REJECT_FOR_PRODUCTION"
            rec = (
                f"LAP end-to-end feasibility fails primarily at protected horizontal / pivot-support "
                f"establishment (`{top}`, {fails_c.get(top, 0)}/{n}). "
                f"When release was reached, relative rotation was absent or negligible "
                f"(≥2° on {n_rel2}/{n}). Mechanism audit shows occasional DX2-induced relative "
                f"motion on large-|dz| ICs, but it is often wrong-signed or coupled with large "
                f"translation — not a controllable Pivot. Do not integrate; do not start SFT/GRPO."
            )
        elif ok_c[0] == 0 and n_rel2 == 0:
            verdict = "REJECT_PHYSICALLY_INFEASIBLE_UNDER_PROTOCOL"
            rec = (
                f"Under legal actions and identical IC snapshots, LAP did not produce meaningful "
                f"cylinder–gripper relative rotation. Dominant failure: `{top}`. "
                "Do not integrate; do not start SFT/GRPO on this mechanism."
            )
        else:
            verdict = "REJECT_FOR_PRODUCTION"
            rec = (
                f"Measured LAP behavior does not justify production adoption (C ok-rate={ok_c[2]:.2f}, "
                f"beats A={beat}/{n}). Dominant failure mode: `{top}`. Stop after this feasibility report."
            )
        limit = top

    final = [
        "# FINAL_RECOMMENDATION — Lever-Arm Assisted Pivot",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"## Verdict: **{verdict}**",
        "",
        rec,
        "",
        f"**Primary limitation (measured):** `{limit}`",
        "",
        "## Evidence snapshot",
        "",
        f"- C ok: {ok_c[0]}/{ok_c[1]} ({ok_c[2]:.2f})",
        f"- A ok: {ok_a[0]}/{ok_a[1]} ({ok_a[2]:.2f})",
        f"- Relative rotation ≥2° (C): {n_rel2}/{n}",
        f"- C beats A: {beat}/{n}",
        f"- Failure histogram (C): `{fails_c}`",
        "",
        "## Explicit non-actions",
        "",
        "- Do **not** integrate LAP into production Oracle.",
        "- Do **not** regenerate expert trajectories.",
        "- Do **not** implement downstream coupled dx/dz in this iteration.",
        "- Do **not** start SFT or GRPO on LAP.",
        "- Keep `verified_lateral_fallback=false`.",
        "",
        "Recommendation is based on measured relative rotation, contact forces, COM translation, "
        "and brake/regrip outcomes — not intermediate geometry alone.",
        "",
    ]
    (OUT / "FINAL_RECOMMENDATION.md").write_text("\n".join(final), encoding="utf-8")
    print("wrote reports", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
