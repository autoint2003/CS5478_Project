"""Phase 7 — Held-out probabilistic eval after validation-based checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p53_independent_grpo64.probabilistic_eval import (
    eval_checkpoint,
    load_heldout_starts,
)
from experiments.precision.p60_c_stable_state_grpo.paths import (
    CKPT_DIR,
    OUT,
    P57A_CKPT,
    P59_PASS1,
    SEED,
)


def select_best_from_validation() -> tuple[str, Path]:
    """Select best P60 pass by C val MSP; if pass2 regresses vs pass1, keep pass1.

    Compare also against P59_pass1; do not use held-out.
    """
    candidates = [
        OUT / "VALIDATION_after_pass2.json",
        OUT / "VALIDATION.json",
        OUT / "VALIDATION_after_pass1.json",
    ]
    data = None
    for p in candidates:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            break
    if data is None:
        if (CKPT_DIR / "P60_pass2.pt").is_file():
            return "P60_pass2", CKPT_DIR / "P60_pass2.pt"
        return "P60_pass1", CKPT_DIR / "P60_pass1.pt"

    rc = data.get("reports_c") or {}
    m1 = (rc.get("P60_pass1") or {}).get("MSP_RELAXED")
    m2 = (rc.get("P60_pass2") or {}).get("MSP_RELAXED")
    # Prefer higher C MSP; if pass2 regresses vs pass1, retain pass1
    if m1 is None and m2 is None:
        return "P60_pass1", CKPT_DIR / "P60_pass1.pt"
    if m2 is None:
        return "P60_pass1", CKPT_DIR / "P60_pass1.pt"
    if m1 is None:
        return "P60_pass2", CKPT_DIR / "P60_pass2.pt"
    # If pass2 regresses vs pass1 on C, retain pass1
    if float(m2) + 1e-12 < float(m1):
        return "P60_pass1", CKPT_DIR / "P60_pass1.pt"
    # Tied C: prefer pass1 if ABD retention is better (or equal)
    if abs(float(m2) - float(m1)) <= 1e-12:
        ra = data.get("reports_abd") or {}
        a1 = (ra.get("P60_pass1") or {}).get("MSP_RELAXED")
        a2 = (ra.get("P60_pass2") or {}).get("MSP_RELAXED")
        if a1 is not None and a2 is not None and float(a1) + 1e-12 >= float(a2):
            return "P60_pass1", CKPT_DIR / "P60_pass1.pt"
        return "P60_pass1", CKPT_DIR / "P60_pass1.pt"
    return "P60_pass2", CKPT_DIR / "P60_pass2.pt"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-rollouts", type=int, default=64)
    ap.add_argument("--per-class", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    best_name, best_ckpt = select_best_from_validation()
    starts = load_heldout_starts(per_class=args.per_class)
    ckpts = {
        "P57_A": P57A_CKPT,
        "P59_pass1": P59_PASS1,
        best_name: best_ckpt,
    }
    reports = {}
    for name, ckpt in ckpts.items():
        if not Path(ckpt).is_file():
            reports[name] = {"error": f"missing {ckpt}"}
            continue
        print(f"=== heldout {name} ===", flush=True)
        reports[name] = eval_checkpoint(
            name, Path(ckpt), starts, n_rollouts=args.n_rollouts, device=args.device
        )

    paired = {}
    for base_name in ("P57_A", "P59_pass1"):
        base = reports.get(base_name) or {}
        base_ics = {(r["skill"], r["traj_id"]): r for r in (base.get("per_ic") or [])}
        cur = reports.get(best_name) or {}
        deltas = []
        for r in cur.get("per_ic") or []:
            b = base_ics.get((r["skill"], r["traj_id"]))
            if b:
                deltas.append(float(r["p_relaxed"] - b["p_relaxed"]))
        paired[f"{best_name}_vs_{base_name}"] = {
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "n_improved": int(sum(d > 1e-9 for d in deltas)),
            "n_regressed": int(sum(d < -1e-9 for d in deltas)),
        }

    summary = {
        "best_ckpt": best_name,
        "best_path": str(best_ckpt),
        "selection_note": "Validation C MSP only; held-out not used for selection.",
        "reports": {
            k: {kk: vv for kk, vv in (v or {}).items() if kk != "per_ic"} for k, v in reports.items()
        },
        "paired": paired,
        "per_ic": {k: (v or {}).get("per_ic") for k, v in reports.items()},
    }
    (OUT / "HELDOUT_EVAL.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# HELDOUT_EVAL",
        "",
        f"Best by validation rule: **{best_name}** (`{best_ckpt.name}`).",
        f"Protocol: {args.per_class}/class held-out, {args.n_rollouts} rolls, T=1.",
        "",
        "| Model | MSP_RELAXED | A | B | C | D | Zero |",
        "|-------|------------:|--:|--:|--:|--:|-----:|",
    ]
    for name in ckpts:
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
    lines.append("")
    (OUT / "HELDOUT_EVAL.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["reports"], indent=2))


if __name__ == "__main__":
    main()
