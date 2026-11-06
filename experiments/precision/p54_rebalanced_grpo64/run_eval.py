"""P54 probabilistic eval: P47T vs P53 vs P54 on identical held-out ICs."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p53_independent_grpo64.probabilistic_eval import (
    compare_baselines,
    load_heldout_starts,
)
from experiments.precision.p54_rebalanced_grpo64.paths import CKPT_DIR, OUT, P47T_CKPT, P53_CKPT


def main(device: str = "cuda", n_rollouts: int = 64, per_class: int = 4):
    OUT.mkdir(parents=True, exist_ok=True)
    starts = load_heldout_starts(per_class=per_class)
    # compare_baselines hardcodes P50/P51; customize for P47T/P53/P54
    from experiments.precision.p53_independent_grpo64.probabilistic_eval import eval_checkpoint
    import numpy as np

    ckpts = [("P47T", P47T_CKPT), ("P53", P53_CKPT)]
    p54 = CKPT_DIR / "P54_grpo64_epoch1.pt"
    if p54.is_file():
        ckpts.append(("P54", p54))
    results = {}
    for tag, ck in ckpts:
        if not Path(ck).is_file():
            results[tag] = {"error": f"missing {ck}"}
            continue
        results[tag] = eval_checkpoint(
            tag, Path(ck), starts, n_rollouts=n_rollouts, device=device
        )

    paired = {}
    if "P47T" in results and "per_ic" in results["P47T"]:
        base = {(r["skill"], r["traj_id"]): r for r in results["P47T"]["per_ic"]}
        for tag, summ in results.items():
            if tag == "P47T" or "per_ic" not in summ:
                continue
            deltas = []
            for r in summ["per_ic"]:
                b = base.get((r["skill"], r["traj_id"]))
                if b:
                    deltas.append(r["p_relaxed"] - b["p_relaxed"])
            paired[tag] = {
                "mean_delta_p_relaxed": float(np.mean(deltas)) if deltas else None,
                "n_improved": int(sum(d > 1e-9 for d in deltas)),
                "n_regressed": int(sum(d < -1e-9 for d in deltas)),
                "n_tied": int(sum(abs(d) <= 1e-9 for d in deltas)),
            }
    # also pair P54 vs P53
    if "P53" in results and "P54" in results and "per_ic" in results["P53"]:
        base = {(r["skill"], r["traj_id"]): r for r in results["P53"]["per_ic"]}
        deltas = []
        for r in results["P54"]["per_ic"]:
            b = base.get((r["skill"], r["traj_id"]))
            if b:
                deltas.append(r["p_relaxed"] - b["p_relaxed"])
        paired["P54_vs_P53"] = {
            "mean_delta_p_relaxed": float(np.mean(deltas)) if deltas else None,
            "n_improved": int(sum(d > 1e-9 for d in deltas)),
            "n_regressed": int(sum(d < -1e-9 for d in deltas)),
            "n_tied": int(sum(abs(d) <= 1e-9 for d in deltas)),
        }

    out = {"summaries": results, "paired_vs_P47T": paired, "n_rollouts": n_rollouts}
    (OUT / "P54_probabilistic_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for tag, s in results.items():
        if "MSP_RELAXED_12S" in s:
            print(
                tag,
                "MSP",
                round(s["MSP_RELAXED_12S"], 4),
                "by",
                {c: round(d["MSP_RELAXED"], 4) for c, d in (s.get("by_class") or {}).items()},
            )
    print("paired", paired)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-rollouts", type=int, default=64)
    ap.add_argument("--eval-per-class", type=int, default=4)
    args = ap.parse_args()
    main(device=args.device, n_rollouts=args.eval_rollouts, per_class=args.eval_per_class)
