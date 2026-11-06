"""Phase 1 — Audit P54 0/64 SUCCESS_RELAXED groups."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.precision.p55_expert_takeover.paths import OUT, P54_MANIFEST, P54_PROGRESS, SKILLS


def load_zero_ics() -> list[dict]:
    rows = list(csv.DictReader(P54_PROGRESS.open(encoding="utf-8")))
    zeros = [r for r in rows if int(r["success_relaxed_count"]) == 0]
    return zeros


def write_audit() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(P54_PROGRESS.open(encoding="utf-8")))
    man = json.loads(P54_MANIFEST.read_text(encoding="utf-8"))
    zeros = load_zero_ics()
    by_total = Counter(r["skill"] for r in rows)
    by_zero = Counter(r["skill"] for r in zeros)

    lines = [
        "# ZERO_SUCCESS_AUDIT",
        "",
        "Source: P54 Independent GRPO-64 full epoch (`P54_FULL_EPOCH_PROGRESS.csv`).",
        "Metric: `SUCCESS_RELAXED_12S` count within each IC's 64 independent rollouts.",
        "",
        "## Summary",
        "",
        f"- Train ICs: **{len(rows)}**",
        f"- 0/64 ICs: **{len(zeros)}** ({100 * len(zeros) / len(rows):.1f}%)",
        f"- Expected sim cost of original groups: {len(zeros)} × 64 = **{len(zeros) * 64}** failed rollouts",
        "",
        "| Class | N_train | 0/64 | Rate | Mean DROP/64 | Mean HANDOFF/64 | Mean e_pos (mm) |",
        "|-------|--------:|-----:|-----:|-------------:|----------------:|----------------:|",
    ]
    detail = {}
    for sk in SKILLS:
        zs = [r for r in zeros if r["skill"] == sk]
        n = by_total[sk]
        if not zs:
            lines.append(f"| {sk} | {n} | 0 | 0% | — | — | — |")
            detail[sk] = {"n_train": n, "n_zero": 0, "ics": []}
            continue
        md = sum(int(r["drop_before_count"]) for r in zs) / len(zs)
        mh = sum(int(r["handoff_count"]) for r in zs) / len(zs)
        me = sum(float(r["e_pos_mean"] or 0) for r in zs) / len(zs)
        lines.append(
            f"| {sk} | {n} | {len(zs)} | {100 * len(zs) / n:.1f}% | {md:.1f} | {mh:.2f} | {me:.1f} |"
        )
        # Failure typology from aggregate counters
        mostly_drop = sum(1 for r in zs if int(r["drop_before_count"]) >= 32)
        mostly_timeout = sum(
            1
            for r in zs
            if int(r["drop_before_count"]) < 32 and int(r["handoff_count"]) == 0
        )
        rare_hand = sum(1 for r in zs if int(r["handoff_count"]) > 0)
        detail[sk] = {
            "n_train": n,
            "n_zero": len(zs),
            "mean_drop_per_group": md,
            "mean_handoff_per_group": mh,
            "mean_e_pos_mm": me,
            "failure_types": {
                "majority_DROP_BEFORE": mostly_drop,
                "timeout_no_handoff": mostly_timeout,
                "any_handoff_then_fail": rare_hand,
            },
            "ics": [
                {
                    "ic_id": r["ic_id"],
                    "drop_before_count": int(r["drop_before_count"]),
                    "handoff_count": int(r["handoff_count"]),
                    "e_pos_mean": float(r["e_pos_mean"] or 0),
                    "e_theta_mean": float(r["e_theta_mean"] or 0),
                    "reward_mean": float(r["reward_mean"]),
                }
                for r in zs
            ],
        }

    lines += [
        "",
        "## Failure typology (from group aggregates)",
        "",
        "P54 progress does not store per-step traces; types inferred from DROP/HANDOFF counts within the 64.",
        "",
    ]
    for sk in SKILLS:
        d = detail[sk]
        if d["n_zero"] == 0:
            continue
        ft = d["failure_types"]
        lines.append(
            f"- **{sk}**: majority DROP {ft['majority_DROP_BEFORE']}, "
            f"timeout/no-handoff {ft['timeout_no_handoff']}, "
            f"any handoff-then-fail {ft['any_handoff_then_fail']}"
        )

    lines += [
        "",
        "## Intermediate snapshots",
        "",
        "Not stored in P54 progress CSV. P55 re-rolls policy on 0/64 ICs and captures "
        "`take_branch_state` (full MjData + R3F + policy history) for takeover candidates.",
        "",
        "## C zero ICs (primary target)",
        "",
    ]
    for ic in detail.get("C", {}).get("ics") or []:
        lines.append(
            f"- `{ic['ic_id']}` DROP={ic['drop_before_count']}/64 e_pos≈{ic['e_pos_mean']:.0f}mm"
        )

    lines += [
        "",
        "## Held-out",
        "",
        "Expert-data generation uses **training** 0/64 ICs only (not held-out).",
        "",
    ]
    (OUT / "ZERO_SUCCESS_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "n_train": len(rows),
        "n_zero": len(zeros),
        "by_class": detail,
        "manifest_counts": man.get("n_train_per_class"),
    }
    (OUT / "ZERO_SUCCESS_AUDIT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    p = write_audit()
    print("zeros", p["n_zero"])
    for sk, d in p["by_class"].items():
        print(sk, d["n_zero"], "/", d["n_train"])
