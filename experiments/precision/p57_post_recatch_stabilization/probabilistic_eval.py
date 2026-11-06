"""P57 held-out eval: P56 vs P57-A vs P57-B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.probabilistic_eval import (
    eval_checkpoint,
    load_heldout_starts,
)
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor, sample_step
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p57_post_recatch_stabilization.paths import (
    MAX_EP_STEPS,
    OUT,
    P56_CKPT,
    SEED,
    T_MAX,
    TEMPERATURE,
)


def _c_holdout_diag(ckpt: Path, starts_c, *, n_rollouts: int, device: str) -> dict:
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(ckpt, device_t)
    actor.eval()
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    rows = []
    for st in starts_c:
        rec = stab = lost = succ = drop = 0
        for _ in range(n_rollouts):
            env.reset(st)
            done = False
            info = {}
            while not done:
                tok, resid, _, _ = sample_step(
                    actor, env.obs_seq, env.prev_seq, device_t, temperature=TEMPERATURE
                )
                _r, done, info = env.step(tok, resid)
            cd = info.get("c_diag") or {}
            rec += int(cd.get("recatch") or 0)
            stab += int(cd.get("stable_hold") or 0)
            lost += int(cd.get("lost_after_recatch") or 0)
            succ += int(info.get("SUCCESS_RELAXED_12S") or 0)
            drop += int(info.get("drop") or 0)
        n = n_rollouts
        rows.append(
            {
                "ic_id": st.traj_id,
                "p_recatch": rec / n,
                "p_stable_hold": stab / n,
                "p_stable_given_recatch": (stab / rec) if rec else None,
                "p_lost_after_recatch": lost / n,
                "p_relaxed": succ / n,
                "p_drop": drop / n,
            }
        )
    return {
        "per_ic": rows,
        "mean_p_recatch": float(np.mean([r["p_recatch"] for r in rows])) if rows else None,
        "mean_p_stable": float(np.mean([r["p_stable_hold"] for r in rows])) if rows else None,
        "mean_p_relaxed": float(np.mean([r["p_relaxed"] for r in rows])) if rows else None,
    }


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
        "P56": P56_CKPT,
        "P57_A": OUT / "checkpoints/P57_A_epoch1.pt",
        "P57_B": OUT / "checkpoints/P57_B_epoch1.pt",
    }
    reports = {}
    c_diag = {}
    for name, ckpt in ckpts.items():
        if not Path(ckpt).is_file():
            reports[name] = {"error": f"missing {ckpt}"}
            continue
        print(f"=== eval {name} ===", flush=True)
        reports[name] = eval_checkpoint(
            name, Path(ckpt), starts, n_rollouts=args.n_rollouts, device=args.device
        )
        print(f"=== C diag {name} ===", flush=True)
        c_diag[name] = _c_holdout_diag(
            Path(ckpt), starts.get("C") or [], n_rollouts=args.n_rollouts, device=args.device
        )

    paired = {}
    for base_name in ("P56", "P57_A"):
        base = reports.get(base_name) or {}
        base_ics = {(r["skill"], r["traj_id"]): r for r in (base.get("per_ic") or [])}
        for name in ("P57_A", "P57_B"):
            if name == base_name:
                continue
            cur = reports.get(name) or {}
            deltas = []
            for r in cur.get("per_ic") or []:
                b = base_ics.get((r["skill"], r["traj_id"]))
                if not b:
                    continue
                deltas.append(float(r["p_relaxed"] - b["p_relaxed"]))
            paired[f"{name}_vs_{base_name}"] = {
                "mean_delta": float(np.mean(deltas)) if deltas else None,
                "n_improved": int(sum(d > 1e-9 for d in deltas)),
                "n_regressed": int(sum(d < -1e-9 for d in deltas)),
            }

    summary = {
        "seed": SEED,
        "reports": {
            k: {kk: vv for kk, vv in (v or {}).items() if kk != "per_ic"} for k, v in reports.items()
        },
        "paired": paired,
        "c_holdout_diag": c_diag,
        "per_ic": {k: (v or {}).get("per_ic") for k, v in reports.items()},
    }
    (OUT / "PROBABILISTIC_EVAL.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# PROBABILISTIC_EVAL",
        "",
        f"Held-out 4/class, {args.n_rollouts} rolls, T=1.",
        "",
        "| Model | MSP_RELAXED | A | B | C | D | Zero |",
        "|-------|------------:|--:|--:|--:|--:|-----:|",
    ]
    for name in ("P56", "P57_A", "P57_B"):
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
    lines += ["", "## Paired", ""]
    for k, v in paired.items():
        lines.append(f"- {k}: meanΔ={v['mean_delta']}, ↑{v['n_improved']} ↓{v['n_regressed']}")
    lines += ["", "## Held-out C recatch/stable", ""]
    for name, d in c_diag.items():
        lines.append(
            f"- {name}: p_recatch={d.get('mean_p_recatch')}, p_stable={d.get('mean_p_stable')}, "
            f"p_relaxed={d.get('mean_p_relaxed')}"
        )
    lines.append("")
    (OUT / "PROBABILISTIC_EVAL.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["reports"], indent=2))


if __name__ == "__main__":
    main()
