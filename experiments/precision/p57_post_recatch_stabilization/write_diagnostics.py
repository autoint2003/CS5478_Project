"""Build C_STABILIZATION_DIAGNOSTICS.md from P57-A/B training progress CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.precision.p57_post_recatch_stabilization.paths import OUT


def _load(tag: str):
    p = OUT / f"P57_{tag}_FULL_EPOCH_PROGRESS.csv"
    if not p.is_file():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8")))


def summarize(tag: str) -> dict:
    rows = _load(tag)
    c = [r for r in rows if r["class"] == "C"]
    if not c:
        return {"tag": tag, "n_c": 0}
    n = len(c)
    g = 64
    rec = sum(int(r["recatch_count"]) for r in c)
    stab = sum(int(r["stable_hold_count"]) for r in c)
    lost = sum(int(r["lost_after_recatch_count"]) for r in c)
    succ = sum(int(r["success_relaxed_count"]) for r in c)
    drop = sum(int(r["drop_before_count"]) for r in c)
    zero = sum(int(r["n_zero_success_group"]) for r in c)
    return {
        "tag": tag,
        "n_c_ics": n,
        "recatch_rate": rec / (n * g),
        "stable_hold_rate": stab / (n * g),
        "stable_given_recatch": (stab / rec) if rec else None,
        "lost_after_recatch_rate": lost / (n * g),
        "p_relaxed": succ / (n * g),
        "drop_rate": drop / (n * g),
        "zero_success_ics": zero,
        "per_ic": [
            {
                "ic_id": r["ic_id"],
                "p_relaxed": int(r["success_relaxed_count"]) / g,
                "p_recatch": int(r["recatch_count"]) / g,
                "p_stable": int(r["stable_hold_count"]) / g,
                "p_lost": int(r["lost_after_recatch_count"]) / g,
                "p_drop": int(r["drop_before_count"]) / g,
                "stable_given_recatch": (
                    int(r["stable_hold_count"]) / int(r["recatch_count"])
                    if int(r["recatch_count"])
                    else None
                ),
            }
            for r in c
        ],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = summarize("A")
    b = summarize("B")
    out = {"A": a, "B": b}
    (OUT / "C_STABILIZATION_DIAGNOSTICS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# C_STABILIZATION_DIAGNOSTICS",
        "",
        "Fixed stable-hold criterion: ≥12 bilateral steps, min_pad≥2, mean vrel≤0.35 after valid recatch.",
        "",
        "| Branch | C ICs | recatch | stable | stable|recatch | lost_after | p_relaxed | DROP | zero ICs |",
        "|--------|------:|--------:|-------:|-----------------:|-----------:|----------:|-----:|---------:|",
    ]
    for tag, s in (("A", a), ("B", b)):
        if not s.get("n_c_ics"):
            lines.append(f"| P57-{tag} | — | | | | | | | |")
            continue
        lines.append(
            f"| P57-{tag} | {s['n_c_ics']} | {s['recatch_rate']:.3f} | {s['stable_hold_rate']:.3f} | "
            f"{s['stable_given_recatch'] if s['stable_given_recatch'] is not None else 'n/a'} | "
            f"{s['lost_after_recatch_rate']:.3f} | {s['p_relaxed']:.4f} | {s['drop_rate']:.3f} | "
            f"{s['zero_success_ics']} |"
        )
    lines += [
        "",
        "A/B/D rates and GRPO stats: see `P57_*_TRAINING_REPORT.md`.",
        "",
    ]
    (OUT / "C_STABILIZATION_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_ic"} for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
