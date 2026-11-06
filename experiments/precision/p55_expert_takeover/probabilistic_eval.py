"""P55 probabilistic eval: P54 vs P55-A vs P55-B on same held-out ICs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p53_independent_grpo64.probabilistic_eval import (
    eval_checkpoint,
    load_heldout_starts,
)
from experiments.precision.p55_expert_takeover.paths import OUT, P54_CKPT, SEED


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-rollouts", type=int, default=64)
    ap.add_argument("--per-class", type=int, default=4)
    args = ap.parse_args()

    starts = load_heldout_starts(per_class=args.per_class)
    ckpts = {
        "P54": P54_CKPT,
        "P55_A": OUT / "checkpoints/P55_A_control_best.pt",
        "P55_B": OUT / "checkpoints/P55_B_expert_best.pt",
    }
    reports = {}
    for name, ckpt in ckpts.items():
        if not Path(ckpt).is_file():
            reports[name] = {"error": f"missing {ckpt}"}
            continue
        print(f"=== eval {name} {ckpt} ===", flush=True)
        reports[name] = eval_checkpoint(
            name, Path(ckpt), starts, n_rollouts=args.n_rollouts, device=args.device
        )

    paired = {}
    base = reports.get("P54") or {}
    base_ics = {(r["skill"], r["traj_id"]): r for r in (base.get("per_ic") or [])}
    for name in ("P55_A", "P55_B"):
        cur = reports.get(name) or {}
        deltas = []
        for r in cur.get("per_ic") or []:
            b = base_ics.get((r["skill"], r["traj_id"]))
            if not b:
                continue
            d = float(r["p_relaxed"] - b["p_relaxed"])
            deltas.append(
                {
                    "ic_id": r["traj_id"],
                    "skill": r["skill"],
                    "msp_relaxed_delta": d,
                    "msp_relaxed": r["p_relaxed"],
                    "msp_relaxed_p54": b["p_relaxed"],
                    "zero_success": r["zero_success"],
                }
            )
        paired[name] = {
            "mean_delta": float(np.mean([x["msp_relaxed_delta"] for x in deltas])) if deltas else None,
            "n_improved": int(sum(x["msp_relaxed_delta"] > 1e-9 for x in deltas)),
            "n_regressed": int(sum(x["msp_relaxed_delta"] < -1e-9 for x in deltas)),
            "per_ic": deltas,
        }

    summary = {
        "seed": SEED,
        "ckpts": {k: str(v) for k, v in ckpts.items()},
        "n_rollouts": args.n_rollouts,
        "reports": {
            k: {kk: vv for kk, vv in (v or {}).items() if kk != "per_ic"}
            for k, v in reports.items()
        },
        "paired_vs_p54": paired,
        "per_ic": {k: (v or {}).get("per_ic") for k, v in reports.items()},
    }
    (OUT / "PROBABILISTIC_EVAL.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# PROBABILISTIC_EVAL",
        "",
        f"Held-out ICs ({args.per_class}/class), {args.n_rollouts} independent rollouts / IC, T=1.",
        "",
    ]
    for name, rep in reports.items():
        if "error" in (rep or {}):
            lines.append(f"## {name}: ERROR {rep['error']}")
            continue
        lines.append(f"## {name}")
        lines.append(f"- MSP_RELAXED: {rep.get('MSP_RELAXED_12S')}")
        lines.append(f"- By class: {rep.get('by_class')}")
        lines.append(f"- Zero-success IC count: {rep.get('zero_success_ics')}")
        lines.append(f"- SUCCESS_NEW MSP: {rep.get('MSP_NEW')}")
        lines.append(f"- SUCCESS_12S MSP: {rep.get('MSP_12S')}")
        lines.append("")
    for name, p in paired.items():
        lines.append(f"### Paired vs P54 ({name})")
        lines.append(f"- mean ΔMSP_RELAXED: {p.get('mean_delta')}")
        lines.append(f"- improved/regressed: {p.get('n_improved')}/{p.get('n_regressed')}")
        lines.append("")
    (OUT / "PROBABILISTIC_EVAL.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["reports"], indent=2))


if __name__ == "__main__":
    main()
