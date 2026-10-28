#!/usr/bin/env python
"""OAR_MECHANISM_REPORT.md + FINAL_RECOMMENDATION.md + optional plots."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/oscillation_assisted_reorientation"


def _f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def make_plots(paired: list) -> None:
    """Time-series plots for a few representative trials."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (OUT / "plots" / "PLOT_SKIP.txt").write_text(f"matplotlib unavailable: {exc}", encoding="utf-8")
        return

    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    # Pick up to 4 trials with longest series / interesting failures
    traces = sorted((OUT / "traces").glob("*__*.json"))
    picked = []
    for p in traces:
        if "actuation" in p.name or "multi" in p.name:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        series = d.get("series") or []
        if len(series) < 10:
            continue
        picked.append((p, d))
        if len(picked) >= 4:
            break

    for p, d in picked:
        series = d["series"]
        t = [s.get("t") or i * 0.02 for i, s in enumerate(series)]
        rel = [_f(s.get("signed_rel_deg")) for s in series]
        dth = [_f(s.get("dtheta_deg")) for s in series]
        grip = [_f(s.get("grip_cmd")) for s in series]
        fn = [_f(s.get("fn_tot")) for s in series]
        wh = [_f(s.get("w_hand_norm")) for s in series]
        dp = []
        p0 = None
        for s in series:
            pc = np.asarray(s.get("p_c_h") or [0, 0, 0], float)
            if p0 is None:
                p0 = pc
            dp.append(float(1e3 * np.linalg.norm(pc - p0)))

        fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
        axs[0].plot(t, rel, label="signed rel°")
        axs[0].plot(t, dth, label="world dθ°")
        axs[0].legend(loc="best")
        axs[0].set_ylabel("deg")
        axs[0].set_title(p.stem)
        axs[1].plot(t, wh, label="‖ω_hand‖")
        axs[1].legend(loc="best")
        axs[1].set_ylabel("rad/s")
        axs[2].plot(t, grip, label="grip cmd")
        axs[2].plot(t, np.asarray(fn) / max(max(fn), 1.0), label="fn_tot (norm)")
        axs[2].legend(loc="best")
        axs[3].plot(t, dp, label="ΔCOM mm")
        axs[3].legend(loc="best")
        axs[3].set_xlabel("t (s)")
        fig.tight_layout()
        fig.savefig(plots / f"{p.stem}.png", dpi=120)
        plt.close(fig)


def main() -> int:
    paired_path = OUT / "PAIRED_SINGLE_CYCLE.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8")) if paired_path.exists() else []
    flat = []
    csv_path = OUT / "SINGLE_CYCLE_RESULTS.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            flat = list(csv.DictReader(f))

    n_ic = len([r for r in paired if not r.get("error")])
    trials = [r for r in flat if r.get("condition")]
    n_trial = len(trials)
    n_ok = sum(1 for r in trials if str(r.get("ok")).lower() in ("true", "1"))
    n_persist = sum(
        1
        for r in trials
        if abs(_f(r.get("net_relative_rotation_deg"))) >= 1.5
        and _f(r.get("d_theta_improve_deg")) >= 1.0
        and str(r.get("bilateral_end")).lower() in ("true", "1")
        and str(r.get("ok")).lower() in ("true", "1")
        and str(r.get("failure") or "") not in (
            "CONTACT_LOSS",
            "DROP",
            "EXCESSIVE_COM_TRANSLATION",
            "UNCONTROLLED_SLIP",
        )
        and _f(r.get("delta_p_rel_mm")) <= 12.0
    )

    fails = {}
    for r in trials:
        k = str(r.get("failure") or ("ok" if str(r.get("ok")).lower() in ("true", "1") else "unknown"))
        fails[k] = fails.get(k, 0) + 1

    by_cond = {}
    for r in trials:
        c = str(r.get("condition"))
        by_cond.setdefault(c, {"n": 0, "ok": 0, "persist": 0, "mean_net": []})
        by_cond[c]["n"] += 1
        if str(r.get("ok")).lower() in ("true", "1"):
            by_cond[c]["ok"] += 1
        if (
            str(r.get("ok")).lower() in ("true", "1")
            and abs(_f(r.get("net_relative_rotation_deg"))) >= 1.5
            and _f(r.get("d_theta_improve_deg")) >= 1.0
            and _f(r.get("delta_p_rel_mm")) <= 12.0
        ):
            by_cond[c]["persist"] += 1
        by_cond[c]["mean_net"].append(_f(r.get("net_relative_rotation_deg")))

    # Direction control: B vs C opposite net signs on same IC
    dir_ctrl = 0
    dir_n = 0
    for r in paired:
        if r.get("error"):
            continue
        res = r.get("results") or {}
        for d in (1, -1):
            b = res.get(f"B_dir{d}") or {}
            c = res.get(f"C_dir{d}") or {}
            if b.get("net_relative_rotation_deg") is None or c.get("net_relative_rotation_deg") is None:
                continue
            dir_n += 1
            if _f(b.get("net_relative_rotation_deg")) * _f(c.get("net_relative_rotation_deg")) < 0:
                dir_ctrl += 1

    multi_path = OUT / "MULTI_CYCLE_RESULTS.csv"
    multi_note = multi_path.read_text(encoding="utf-8")[:200] if multi_path.exists() else "missing"

    make_plots(paired)

    lines = [
        "# OAR_MECHANISM_REPORT — Single-cycle Oscillation-Assisted Reorientation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Isolated diagnostic. Production Oracle / experts / M111 / SUCCESS_* unchanged. "
        "`verified_lateral_fallback` untouched. Legal `step_legal` actions only.",
        "",
        "## Scale",
        "",
        f"- ICs tested: **{n_ic}**",
        f"- Paired trials (A/B/C/D × directions): **{n_trial}**",
        f"- Trials with `ok=True`: **{n_ok}**",
        f"- Persistent corrective (net |rel|≥1.5° AND Δdθ≥1° AND bilateral): **{n_persist}**",
        "",
        "## Per-condition",
        "",
        "| Cond | n | ok | persist | mean net rel° |",
        "|---|---:|---:|---:|---:|",
    ]
    for c, v in sorted(by_cond.items()):
        mean_net = float(np.mean(v["mean_net"])) if v["mean_net"] else 0.0
        lines.append(f"| {c} | {v['n']} | {v['ok']} | {v['persist']} | {mean_net:.3f} |")

    lines += [
        "",
        "### Failure taxonomy",
        "",
        "```",
        json.dumps(fails, indent=2),
        "```",
        "",
        "## Decision criteria checklist",
        "",
        f"1. Repeatable nonzero net relative rotation after cycle+settle: **{'YES' if n_persist > 0 else 'NO'}** ({n_persist}/{n_trial})",
        f"2. Direction control via actuation/grip phase (B vs C opposite net): **{dir_ctrl}/{dir_n}** pairs",
        f"3. World-frame dθ reduction with persistent rel: counted in persist={n_persist}",
        "4. COM translation / grasp: see `diag_3mm_ok` and CONTACT/DROP failures in taxonomy",
        "5. No privileged forces/edits: satisfied by construction (`step_legal` only)",
        "",
        "## Multi-cycle",
        "",
        f"```\n{multi_note}\n```",
        "",
        "## Interpretation",
        "",
    ]
    if n_persist == 0:
        lines.append(
            "Single-cycle OAR did **not** produce persistent, directional cylinder–gripper relative "
            "rotation that survives the return stroke and settling. Transient peaks (if any) that "
            "vanish on reverse are classified `REVERSE_STROKE_UNDOES_CORRECTION` / "
            "`NO_NET_RELATIVE_ROTATION` — not successes."
        )
    else:
        lines.append(
            "A nonempty set of trials shows persistent net relative rotation. See FINAL_RECOMMENDATION "
            "for whether this clears the bar for a feedback OAR prototype."
        )

    lines += [
        "",
        "## Artifacts",
        "",
        "- `ACTUATION_AUDIT.md`",
        "- `SINGLE_CYCLE_RESULTS.csv`, `PAIRED_SINGLE_CYCLE.json`",
        "- `MULTI_CYCLE_RESULTS.csv`",
        "- `traces/`, `plots/`, `VIDEOS/`",
        "",
    ]
    (OUT / "OAR_MECHANISM_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # Final recommendation
    if n_persist >= max(2, n_ic // 3) and n_ok >= 1:
        verdict = "CONDITIONAL_PROCEED"
        rec = (
            "OAR shows repeatable persistent relative rotation on a nonempty IC subset. "
            "Next iteration: feedback-timed grip around measured ω_rel, still isolated — "
            "do not merge into production Oracle yet."
        )
        limit = "needs_feedback_controller"
    elif n_persist > 0:
        verdict = "WEAK_SIGNAL_REJECT_FOR_PRODUCTION"
        rec = (
            "Sparse persistent corrections exist but are not reliable enough for production. "
            "Do not integrate; optional follow-up only as isolated research."
        )
        limit = max(fails.items(), key=lambda kv: kv[1])[0] if fails else "sparse"
    else:
        top = max(fails.items(), key=lambda kv: kv[1])[0] if fails else "NO_NET_RELATIVE_ROTATION"
        verdict = "REJECT_FOR_PRODUCTION"
        rec = (
            f"OAR single-cycle feasibility failed. Dominant mode: `{top}`. "
            "Symmetric oscillation and grip-phased ratcheting did not yield controllable "
            "net relative rotation after return+settle. Recommend investigating a different "
            "contact-manipulation mechanism (or revisiting force-backed support-transfer physics), "
            "not amplifying oscillation amplitude/frequency."
        )
        limit = top

    final = f"""# FINAL_RECOMMENDATION — Oscillation-Assisted Reorientation

Generated: {datetime.now(timezone.utc).isoformat()}

## Verdict: **{verdict}**

{rec}

**Primary limitation (measured):** `{limit}`

## Evidence snapshot

- ICs: {n_ic}
- Trials: {n_trial}
- ok: {n_ok}
- Persistent corrective: {n_persist}
- Failure histogram: `{fails}`

## Explicit non-actions

- Do **not** integrate OAR into the production Precision Oracle.
- Do **not** regenerate expert data.
- Do **not** modify official success thresholds.
- Do **not** begin SFT/GRPO training on OAR.

A positive conclusion would require physical evidence of persistent, directional, controllable
relative rotation — not a lower dθ at an intermediate oscillation phase.
"""
    (OUT / "FINAL_RECOMMENDATION.md").write_text(final, encoding="utf-8")
    print("wrote OAR reports", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
