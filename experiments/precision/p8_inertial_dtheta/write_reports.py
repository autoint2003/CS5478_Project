#!/usr/bin/env python
"""FINAL_REPORT.md for P8 from CSVs + authority audit."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p8_inertial_dtheta"


def main() -> int:
    val = list(csv.DictReader((OUT / "FINAL_VALIDATION.csv").open(encoding="utf-8")))
    accel = list(csv.DictReader((OUT / "ACCELERATION_PROBES.csv").open(encoding="utf-8")))
    grip = list(csv.DictReader((OUT / "GRIP_MODULATION_PROBES.csv").open(encoding="utf-8")))
    cycles = list(csv.DictReader((OUT / "STICK_SLIP_CYCLE_TRACES.csv").open(encoding="utf-8")))
    by = {r["variant"]: r for r in val}
    best = min(val, key=lambda r: float(r["theta_final"]))
    any_new = any(str(r.get("SUCCESS_NEW")) in ("True", "true") for r in val)
    any_slip = any(c.get("classification") == "USEFUL_ROTATIONAL_SLIP" for c in cycles)
    useful_grip = [g for g in grip if g.get("behavior") == "USEFUL_ROTATIONAL_SLIP"]

    lines = [
        "# FINAL_REPORT — P8 Inertial Stick-Slip dθ (hard_910277)",
        "",
        "Production Oracle / C / physics unchanged. Bounded I0–I3 pilot.",
        "",
        "## Answers",
        "",
        "### Does higher angular acceleration + grip relaxation produce useful relative rotation?",
        "",
    ]
    if any_slip or useful_grip:
        lines.append(
            "**Partial.** At least one probe/cycle classified as useful rotational slip — "
            "see tables. Check whether improvement persisted through settle + SUCCESS_NEW."
        )
    else:
        lines.append(
            "**No useful rotational slip within legal control limits.** "
            "Faster ω commands increase achieved hand α somewhat, but relative ω stays near zero "
            "under secure grip; grip relaxation either stays co-rotation or risks translation/"
            "contact loss without producing persistent θ reduction to ≤10°."
        )
    lines.extend(
        [
            "",
            "### Does it achieve SUCCESS_NEW?",
            "",
            f"**{'Yes' if any_new else 'No'}.** Best final θ: `{best.get('variant')}` → "
            f"{best.get('theta_final')}° (NEW={best.get('SUCCESS_NEW')}, 12S={best.get('SUCCESS_12S')}).",
            "",
            "## Rotational authority (summary)",
            "",
            "Actions command `r_des` rate up to W_MAX=4 rad/s; achieved hand ω is tracking-limited "
            "(kp_ori/kd_ori), not a direct actuator ω. Object I_xx is ~7.6e-5 kg·m² while "
            "μNR friction torque at Fn≈18 N is ~0.3 N·m → required α to break static friction "
            "by object inertia alone is enormous vs what the wrist tracker delivers. "
            "Hence P7-style and I1 secure-grip motions remain co-rotation.",
            "",
            "### Acceleration probes",
            "",
            "| label | cmd ω | peak ω | peak α | peak w_rel | Δθ | ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in accel:
        lines.append(
            f"| {r['label']} | {r['w_cmd_rad_s']} | {r['peak_w_hand']} | {r['peak_alpha_hand']} | "
            f"{r['peak_w_rel']} | {r['d_theta_abs']} | {r['cmd_vs_achieved_w_ratio']} |"
        )
    lines.extend(
        [
            "",
            "### Grip modulation (fast step ω=4)",
            "",
            "| grip | fn0→fn1 | peak w_rel | Δθ | behavior | bil |",
            "|---:|---|---:|---:|---|---|",
        ]
    )
    for r in grip:
        lines.append(
            f"| {r['grip_cmd']} | {r['fn0']}→{r['fn1']} | {r['peak_w_rel']} | {r['d_theta_abs']} | "
            f"{r['behavior']} | {r['bilateral_end']} |"
        )
    lines.extend(
        [
            "",
            "## I0–I3 validation",
            "",
            "| var | mode | θ0 | θ_final | Δθ | cycles | α_peak | w_rel | 12S | NEW | abort |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for r in val:
        lines.append(
            f"| {r['variant']} | {r['mode']} | {r['theta0']} | {r['theta_final']} | "
            f"{r['net_improve_deg']} | {r['n_cycles']} | {r['peak_alpha']} | {r['peak_w_rel']} | "
            f"{r['SUCCESS_12S']} | {r['SUCCESS_NEW']} | {r.get('abort_reason') or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Physical limitation (if no success)",
            "",
            "Within legal action bounds, the gripper cannot generate sufficient "
            "**relative** angular impulse against current pad friction without either "
            "(a) remaining in stick (co-rotation) or (b) relaxing grip enough to invite "
            "translational slip / contact loss. Inertial stick-slip therefore does not "
            "convert the C-stable 14.6° residual into SUCCESS_NEW in this pilot.",
            "",
            "## Stop conditions",
            "",
            "- Bounded probes + I0–I3 only; no Oracle/C/physics edits; no RL / expert regen.",
            "",
            "## Artifacts",
            "",
            "- `ROTATIONAL_AUTHORITY_AUDIT.md`",
            "- `ACCELERATION_PROBES.csv`",
            "- `GRIP_MODULATION_PROBES.csv`",
            "- `STICK_SLIP_CYCLE_TRACES.csv`",
            "- `FINAL_VALIDATION.csv`",
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
