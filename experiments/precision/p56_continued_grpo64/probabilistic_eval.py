"""P56 probabilistic eval: P54 vs P55-A vs P56 on same held-out ICs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p53_independent_grpo64.probabilistic_eval import (
    eval_checkpoint,
    load_heldout_starts,
)
from experiments.precision.p56_continued_grpo64.paths import (
    OUT,
    P54_CKPT,
    P55A_CKPT,
    SEED,
)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-rollouts", type=int, default=64)
    ap.add_argument("--per-class", type=int, default=4)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    starts = load_heldout_starts(per_class=args.per_class)
    ckpts = {
        "P54": P54_CKPT,
        "P55_A": P55A_CKPT,
        "P56": OUT / "checkpoints/P56_epoch1.pt",
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
    for base_name in ("P54", "P55_A"):
        base = reports.get(base_name) or {}
        base_ics = {(r["skill"], r["traj_id"]): r for r in (base.get("per_ic") or [])}
        for name in ("P55_A", "P56"):
            if name == base_name:
                continue
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
                        f"msp_relaxed_{base_name.lower()}": b["p_relaxed"],
                        "zero_success": r["zero_success"],
                    }
                )
            key = f"{name}_vs_{base_name}"
            paired[key] = {
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
        "paired": paired,
        "per_ic": {k: (v or {}).get("per_ic") for k, v in reports.items()},
    }
    (OUT / "PROBABILISTIC_EVAL.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# PROBABILISTIC_EVAL",
        "",
        f"Held-out ICs ({args.per_class}/class), {args.n_rollouts} independent rollouts / IC, T=1.",
        "",
        "| Model | MSP_RELAXED | A | B | C | D | Zero-succ |",
        "|-------|------------:|--:|--:|--:|--:|----------:|",
    ]
    for name in ("P54", "P55_A", "P56"):
        rep = reports.get(name) or {}
        if "error" in rep:
            lines.append(f"| {name} | ERROR | | | | | |")
            continue
        bc = rep.get("by_class") or {}
        lines.append(
            f"| {name} | {rep.get('MSP_RELAXED_12S'):.4f} | "
            f"{(bc.get('A') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('B') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('C') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('D') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{rep.get('zero_success_ics')} |"
        )
    lines += ["", "## Paired deltas", ""]
    for key, p in paired.items():
        lines.append(
            f"- **{key}**: mean Δ={p.get('mean_delta')}, "
            f"improved={p.get('n_improved')}, regressed={p.get('n_regressed')}"
        )
    lines.append("")
    (OUT / "PROBABILISTIC_EVAL.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["reports"], indent=2))


if __name__ == "__main__":
    main()
