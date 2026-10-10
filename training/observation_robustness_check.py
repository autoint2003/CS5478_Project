"""Observation robustness of the frozen balanced policy. No retraining.

Object channels 27-38 are perturbed as a stand-in for RGB-D tracking error.
Channels 0-26 and 39-44 stay current except in the tactile conditions.
The simulator state is not modified.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from training.observation_study_common import (
    build_cases,
    family_counts,
    load_balanced,
    run_suite,
)

OUT = ROOT / "results" / "diagnostics" / "raw" / "observation_robustness"
REPORT = ROOT / "results" / "diagnostics" / "OBSERVATION_ROBUSTNESS_CHECK.md"
FAMS = ("impact", "sliding", "wrist", "recatch", "airborne", "natural")


def conditions():
    rows = []

    def add(name, group, level, spec, seeds):
        rows.append({"name": name, "group": group, "level": level, "spec": spec, "seeds": list(seeds)})

    stochastic = [0, 1, 2]
    for mm in (1, 2, 5):
        add(f"pos_white_{mm}mm", "position noise", f"sigma {mm} mm per axis", {"pos_sigma": mm / 1000.0}, stochastic)
    for mm in (2, 5):
        for axis, label in ((0, "x"), (1, "y"), (2, "z")):
            bias = [0.0, 0.0, 0.0]
            bias[axis] = mm / 1000.0
            add(
                f"pos_bias_{mm}mm_{label}",
                "position bias",
                f"+{mm} mm on hand-{label}",
                {"pos_bias": bias},
                [0],
            )
    for deg in (0.5, 1.0, 2.0, 5.0):
        add(f"ori_{deg:g}deg", "orientation noise", f"sigma {deg:g} deg", {"ori_sigma_deg": deg}, stochastic)
    for sig in (0.01, 0.03, 0.05, 0.10):
        add(f"vwhite_{sig:.2f}", "linear velocity white", f"sigma {sig:.2f} m/s", {"v_sigma": sig}, stochastic)
        add(
            f"vslow_{sig:.2f}",
            "linear velocity slow",
            f"AR(1) sigma {sig:.2f} m/s, correlation 0.2 s",
            {"v_slow": sig},
            stochastic,
        )
    for sig in (0.1, 0.3, 0.5, 1.0):
        add(f"wwhite_{sig:.1f}", "angular velocity white", f"sigma {sig:.1f} rad/s", {"w_sigma": sig}, stochastic)
        add(
            f"wslow_{sig:.1f}",
            "angular velocity slow",
            f"AR(1) sigma {sig:.1f} rad/s, correlation 0.2 s",
            {"w_slow": sig},
            stochastic,
        )
    for ms in (20, 40, 60, 100):
        add(f"delay_{ms}ms", "tracker delay", f"{ms} ms", {"delay_s": ms / 1000.0}, [0])
    for ms in (20, 50, 100):
        add(f"lowpass_{ms}ms", "low-pass", f"tau {ms} ms", {"tau": ms / 1000.0}, [0])
    for n in (1, 2, 5):
        for mode in ("hold", "propagate"):
            add(
                f"drop_{n}_{mode}",
                "track loss",
                f"{n} frames, {mode}, starting at policy step 1",
                {"dropout": {"start": 1, "n": n, "mode": mode}},
                [0],
            )
    for mm in (0.5, 1.0):
        add(f"cop_{mm:.1f}mm", "tactile CoP", f"sigma {mm:.1f} mm", {"cop_sigma": mm / 1000.0}, stochastic)
        add(f"ehat_{mm:.1f}mm", "tactile lateral", f"sigma {mm:.1f} mm", {"ehat_sigma": mm / 1000.0}, stochastic)
    for newton in (1.0, 2.0):
        add(f"fn_{newton:.0f}N", "tactile force", f"sigma {newton:.0f} N", {"fn_sigma": newton}, stochastic)
    add(
        "combined_moderate",
        "combined",
        "2 mm position, 1 deg orientation, 0.03 m/s, 0.3 rad/s, 40 ms delay, tau 50 ms, CoP 0.5 mm, force 1 N",
        {
            "pos_sigma": 0.002,
            "ori_sigma_deg": 1.0,
            "v_sigma": 0.03,
            "w_sigma": 0.3,
            "delay_s": 0.04,
            "tau": 0.05,
            "cop_sigma": 0.0005,
            "fn_sigma": 1.0,
        },
        stochastic,
    )
    return rows


def jobs_of(specs):
    jobs = []
    for spec in specs:
        for seed in spec["seeds"]:
            jobs.append({
                "name": spec["name"],
                "group": spec["group"],
                "level": spec["level"],
                "spec": spec["spec"],
                "seed": int(seed),
            })
    return jobs


CTX = None
MODEL = None


def _init():
    global CTX, MODEL
    torch.set_num_threads(1)
    CTX = build_cases()
    MODEL, _proto, _saved, device = load_balanced()
    print("worker ready", device, flush=True)


def _run(job):
    got = run_suite(CTX, MODEL, job["spec"], job["seed"])
    fam = family_counts(got["rows"])
    print("DONE", job["name"], "seed", job["seed"], got["ok"], fam, flush=True)
    return {
        "name": job["name"],
        "group": job["group"],
        "level": job["level"],
        "seed": job["seed"],
        "ok": got["ok"],
        "n": got["n"],
        "families": fam,
        "rows": [{k: r[k] for k in ("id", "family", "t12", "kind", "mechanism")} for r in got["rows"]],
    }


def _aggregate(clean_rows, results):
    clean_ok = {r["id"]: r["t12"] == "RETAINED_TO_12" for r in clean_rows}
    by = {}
    for row in results:
        by.setdefault(row["name"], []).append(row)
    summary = []
    for name, items in by.items():
        total_ok = sum(it["ok"] for it in items)
        total_n = sum(it["n"] for it in items)
        fam = {f: sum(it["families"][f] for it in items) for f in FAMS}
        fam_n = {f: sum(1 for r in items[0]["rows"] if r["family"] == f) * len(items) for f in FAMS}
        lost, gained = [], []
        fail_kinds = []
        for it in items:
            for r in it["rows"]:
                was = clean_ok[r["id"]]
                now = r["t12"] == "RETAINED_TO_12"
                if was and not now:
                    lost.append(f"{r['id']}@{it['seed']}:{r['t12']}")
                    fail_kinds.append(r["t12"])
                elif (not was) and now:
                    gained.append(f"{r['id']}@{it['seed']}")
                elif not now:
                    fail_kinds.append(r["t12"])
        mode = Counter(fail_kinds).most_common(1)
        summary.append({
            "name": name,
            "group": items[0]["group"],
            "level": items[0]["level"],
            "seeds": [it["seed"] for it in items],
            "ok": total_ok,
            "n": total_n,
            "per_seed": [it["ok"] for it in items],
            "families": fam,
            "family_n": fam_n,
            "lost": lost,
            "gained": gained,
            "failure": mode[0][0] if mode else "none",
        })
    return summary


def write_report(clean_ok, summary):
    lines = []
    lines.append("# Observation robustness check")
    lines.append("")
    lines.append(
        "The frozen balanced checkpoint is evaluated on the same 24-case t = 12 benchmark. "
        "Noise is added only to the observation that enters the Transformer. "
        "The MuJoCo state, the previous action, the v3 torque command, and the nominal continuation are unchanged. "
        "Stochastic conditions use seeds 0, 1, and 2, with an independent stream on each case. "
        f"The unmodified policy scores {clean_ok}/24, matching the published balanced result."
    )
    lines.append("")
    lines.append("## How each sensor error is applied")
    lines.append("")
    lines.append(
        "Relative position, orientation, and twist are decoded with the airborne scales "
        "(0.25 m, π rad, 2 m/s, 8 rad/s), perturbed, then scaled and clipped again. "
        "Orientation noise is a random-axis rotation whose angle is drawn from a zero-mean normal, "
        "composed on the left of the hand-frame relative rotation and converted back through the rotation vector. "
        "White noise is independent across axes and policy steps. "
        "Slow bias is an AR(1) process with correlation time 0.20 s, matched to the stated standard deviation. "
        "Delay holds channels 27–38 from the earlier policy step. Until that step exists, the first object sample is repeated. "
        "Proprioception, tactile readings, the wide-scale hand twist (39–44), and the previous action stay at the current step. "
        "Low-pass filtering is causal. Position, linear velocity, and angular velocity use y ← y + α(x − y) with α = dt / (τ + dt) and dt = 20 ms. "
        "Orientation takes the same step along the relative rotation and is reprojected to SO(3). The filter starts at the first measurement. "
        "Track loss begins at policy step 1 and lasts 1, 2, or 5 steps. Hold repeats the last object sample. "
        "Propagate advances that sample's position by its linear velocity and its rotation by its angular velocity, in the stored hand frame, then reprojects the rotation. "
        "The combined profile low-passes the object channels at τ = 50 ms, adds the white pose and twist noise, then delays the result by 40 ms. "
        "Tactile center-of-pressure noise is applied in metres on the pad coordinates and the lateral estimate is recomputed from those pads. "
        "Force noise is applied in newtons. Neither changes the contact flags or the simulator contact."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| condition | level | t=12 success | per seed | impact | sliding | wrist | recatch | airborne | natural | failure |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for row in summary:
        fam = row["families"]
        fn = row["family_n"]
        seeds = ",".join(str(x) for x in row["per_seed"])
        lines.append(
            f"| {row['group']} | {row['level']} | {row['ok']}/{row['n']} | {seeds} | "
            f"{fam['impact']}/{fn['impact']} | {fam['sliding']}/{fn['sliding']} | {fam['wrist']}/{fn['wrist']} | "
            f"{fam['recatch']}/{fn['recatch']} | {fam['airborne']}/{fn['airborne']} | {fam['natural']}/{fn['natural']} | "
            f"{row['failure']} |"
        )
    lines.append("")
    lines.append("## Case changes against the clean 15/24")
    lines.append("")
    for row in summary:
        if not row["lost"] and not row["gained"]:
            continue
        lines.append(f"**{row['name']}** ({row['level']}). Lost: {', '.join(row['lost']) or 'none'}. Gained: {', '.join(row['gained']) or 'none'}.")
        lines.append("")
    combined = next(r for r in summary if r["name"] == "combined_moderate")
    small = [
        r for r in summary
        if r["name"] in ("pos_white_1mm", "pos_white_2mm", "ori_0.5deg", "ori_1deg", "vwhite_0.01", "wwhite_0.1", "delay_20ms", "delay_40ms", "cop_0.5mm", "fn_1N")
    ]
    small_rate = min(r["ok"] / r["n"] for r in small)
    comb_rate = combined["ok"] / combined["n"]
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        f"Clean success is {clean_ok}/24. "
        f"The lowest rate among the modest single-factor conditions listed in the script "
        f"(1–2 mm position, 0.5–1 deg, 0.01 m/s, 0.1 rad/s, 20–40 ms delay, 0.5 mm CoP, 1 N force) "
        f"is {small_rate:.3f}. The combined moderate profile scores {combined['ok']}/{combined['n']} "
        f"({comb_rate:.3f}), seeds {combined['per_seed']}."
    )
    lines.append("")
    if small_rate >= 14 / 24 and comb_rate >= 13 / 24:
        closing = (
            "Performance is stable under modest realistic sensing error. "
            "Larger noise and longer delay are reported in the table and are the conditions that move the score, when any do."
        )
    elif small_rate >= 13 / 24 and comb_rate < 12 / 24:
        closing = (
            "Only the combined profile, or the higher single-factor levels in the table, moves t = 12 success. "
            "The modest levels stay close to the clean 15/24."
        )
    elif small_rate < 12 / 24:
        closing = (
            "The policy is highly dependent on exact simulator object state: modest tracker or tactile error removes retained grasps."
        )
    else:
        closing = (
            "Modest single-factor error leaves most retained grasps in place. "
            "The combined profile is the deployment-relevant number in the table."
        )
    lines.append(closing)
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("REPORT", REPORT, flush=True)


def main():
    raise SystemExit(
        "OBSERVATION_ROBUSTNESS_CHECK.md records the sweep of the retired 45-D policy. "
        "The 39-D combined profile is run by training/final_39d_balanced_policy.py. "
        "This entry point is closed so it cannot rewrite that report."
    )
    path = OUT / "results.json"
    clean_path = OUT / "clean.json"
    specs = conditions()
    if not clean_path.exists():
        print("clean pass", flush=True)
        _init()
        clean_run = run_suite(CTX, MODEL, {}, 0)
        clean_path.write_text(json.dumps({
            "ok": clean_run["ok"],
            "rows": [{k: r[k] for k in ("id", "family", "t12", "kind", "mechanism")} for r in clean_run["rows"]],
        }, indent=2), encoding="utf-8")
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    print("clean", clean["ok"], flush=True)
    if clean["ok"] != 15:
        raise SystemExit(f"clean balanced pass scored {clean['ok']}/24, expected 15")
    done = {}
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        done = {f"{r['name']}:{r['seed']}": r for r in prev.get("runs", [])}
    pending = []
    for job in jobs_of(specs):
        key = f"{job['name']}:{job['seed']}"
        if key not in done:
            pending.append(job)
    print("pending", len(pending), "cached", len(done), flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    if pending:
        with ctx.Pool(2, initializer=_init) as pool:
            for row in pool.imap_unordered(_run, pending):
                done[f"{row['name']}:{row['seed']}"] = row
                path.write_text(json.dumps({"runs": list(done.values())}, indent=2), encoding="utf-8")
    order = {s["name"]: i for i, s in enumerate(specs)}
    summary = _aggregate(clean["rows"], list(done.values()))
    summary.sort(key=lambda r: order[r["name"]])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(clean["ok"], summary)


if __name__ == "__main__":
    main()
