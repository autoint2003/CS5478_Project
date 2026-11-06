"""Joint held-out + C-val evaluation of P57_A / P59_pass1 / P59_pass2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.precision.p53_independent_grpo64.probabilistic_eval import load_heldout_starts
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p61_reproducible_joint_eval.eval_core import eval_starts
from experiments.precision.p61_reproducible_joint_eval.paths import (
    GLOBAL_SEED,
    N_HELDOUT_ROLLOUTS,
    N_VAL_ROLLOUTS,
    OUT,
    P57A_CKPT,
    P58_SELECTED,
    P59_PASS1,
    P59_PASS2,
    PER_CLASS_HELDOUT,
    PROTOCOL_VERSION,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import build_seed_manifest


CLASS_OF = {"A1": "A", "B1": "B", "C": "C", "D": "D"}


def _paired_stats(base_per: list[dict], cur_per: list[dict]) -> dict[str, Any]:
    bmap = {(r["skill"], r["traj_id"]): r for r in base_per}
    deltas = []
    pairs = []
    for r in cur_per:
        b = bmap.get((r["skill"], r["traj_id"]))
        if not b:
            continue
        d = float(r["p_relaxed"] - b["p_relaxed"])
        deltas.append(d)
        pairs.append(
            {
                "skill": r["skill"],
                "traj_id": r["traj_id"],
                "p_base": b["p_relaxed"],
                "p_cur": r["p_relaxed"],
                "delta": d,
                "n": r["n_rollouts"],
            }
        )
    if not deltas:
        return {"mean_delta": None, "n": 0}
    arr = np.asarray(deltas, float)
    n = len(arr)
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    # Normal approx 95% CI for mean paired delta
    ci_lo = float(arr.mean() - 1.96 * se)
    ci_hi = float(arr.mean() + 1.96 * se)
    return {
        "mean_delta": float(arr.mean()),
        "std_delta": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "se_mean": se,
        "ci95": [ci_lo, ci_hi],
        "n_improved": int(sum(d > 1e-12 for d in deltas)),
        "n_regressed": int(sum(d < -1e-12 for d in deltas)),
        "n_tied": int(sum(abs(d) <= 1e-12 for d in deltas)),
        "n": n,
        "per_ic": pairs,
    }


def _by_class_msp(per_ic: list[dict]) -> dict[str, dict]:
    out = {}
    for sk, lab in CLASS_OF.items():
        rows = [r for r in per_ic if r["skill"] == sk]
        if not rows:
            continue
        out[lab] = {
            "MSP_RELAXED": float(np.mean([r["p_relaxed"] for r in rows])),
            "MSP_12S": float(np.mean([r["p_12s"] for r in rows])),
            "MSP_NEW": float(np.mean([r["p_new"] for r in rows])),
            "zero_success_ics": int(sum(r["zero_success"] for r in rows)),
            "n_ic": len(rows),
        }
    return out


def run_joint_eval(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    held = load_heldout_starts(per_class=PER_CLASS_HELDOUT)
    held_flat = []
    held_ids = []
    for sk in ("A1", "B1", "C", "D"):
        for st in held[sk]:
            held_flat.append(st)
            held_ids.append((sk, st.traj_id))

    sel = json.loads(P58_SELECTED.read_text(encoding="utf-8"))
    ds = build_p54_dataset(seed=GLOBAL_SEED)
    lookup = {(s.skill, s.traj_id): s for s in ds["schedule"]}
    val_c = [lookup[("C", tid)] for tid in sel["val_c"]]

    man = build_seed_manifest(
        heldout_ids=held_ids,
        val_c_ids=list(sel["val_c"]),
        n_heldout=N_HELDOUT_ROLLOUTS,
        n_val=N_VAL_ROLLOUTS,
    )
    (OUT / "SEED_MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

    ckpts = {
        "P57_A": P57A_CKPT,
        "P59_pass1": P59_PASS1,
        "P59_pass2": P59_PASS2,
    }
    for name, p in ckpts.items():
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    heldout_reports = {}
    val_reports = {}
    for name, ckpt in ckpts.items():
        print(f"=== heldout {name} ===", flush=True)
        heldout_reports[name] = eval_starts(
            name, ckpt, held_flat, split="heldout", n_rollouts=N_HELDOUT_ROLLOUTS, device=device
        )
        heldout_reports[name]["by_class"] = _by_class_msp(heldout_reports[name]["per_ic"])
        print(f"=== val_c {name} ===", flush=True)
        val_reports[name] = eval_starts(
            name, ckpt, val_c, split="val_c", n_rollouts=N_VAL_ROLLOUTS, device=device
        )

    # Paired vs P57_A and vs each other on held-out
    paired = {}
    base = heldout_reports["P57_A"]["per_ic"]
    for name in ("P59_pass1", "P59_pass2"):
        paired[f"{name}_vs_P57_A"] = _paired_stats(base, heldout_reports[name]["per_ic"])
    paired["P59_pass2_vs_P59_pass1"] = _paired_stats(
        heldout_reports["P59_pass1"]["per_ic"], heldout_reports["P59_pass2"]["per_ic"]
    )
    paired_val = {
        f"{name}_vs_P57_A": _paired_stats(
            val_reports["P57_A"]["per_ic"], val_reports[name]["per_ic"]
        )
        for name in ("P59_pass1", "P59_pass2")
    }
    paired_val["P59_pass2_vs_P59_pass1"] = _paired_stats(
        val_reports["P59_pass1"]["per_ic"], val_reports["P59_pass2"]["per_ic"]
    )

    joint = {
        "protocol_version": PROTOCOL_VERSION,
        "global_seed": GLOBAL_SEED,
        "device": device,
        "n_heldout_ics": len(held_flat),
        "n_heldout_rollouts": N_HELDOUT_ROLLOUTS,
        "n_val_c": len(val_c),
        "n_val_rollouts": N_VAL_ROLLOUTS,
        "total_heldout_rollouts": len(held_flat) * N_HELDOUT_ROLLOUTS * len(ckpts),
        "heldout_ic_ids": [{"skill": sk, "traj_id": tid} for sk, tid in held_ids],
        "val_c_ids": list(sel["val_c"]),
        "heldout": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_ic"} for k, v in heldout_reports.items()
        },
        "heldout_per_ic": {k: v["per_ic"] for k, v in heldout_reports.items()},
        "val_c": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_ic"} for k, v in val_reports.items()
        },
        "val_c_per_ic": {k: v["per_ic"] for k, v in val_reports.items()},
        "paired_heldout": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_ic"} for k, v in paired.items()
        },
        "paired_heldout_per_ic": {k: v.get("per_ic") for k, v in paired.items()},
        "paired_val_c": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_ic"} for k, v in paired_val.items()
        },
    }
    (OUT / "JOINT_EVAL.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    _write_comparison_md(joint)
    return joint


def _write_comparison_md(joint: dict):
    lines = [
        "# CHECKPOINT_COMPARISON",
        "",
        f"Protocol `{joint['protocol_version']}` | seed={joint['global_seed']} | "
        f"held-out {joint['n_heldout_ics']}×{joint['n_heldout_rollouts']} | "
        f"val_c {joint['n_val_c']}×{joint['n_val_rollouts']}",
        "",
        "All checkpoints evaluated in one harness with identical SHA-256 seeds.",
        "Do not mix these numbers with older cross-folder reports.",
        "",
        "## Held-out MSP",
        "",
        "| Model | MSP_RELAXED | MSP_12S | MSP_NEW | A | B | C | D | Zero ICs |",
        "|-------|------------:|--------:|--------:|--:|--:|--:|--:|---------:|",
    ]
    for name in ("P57_A", "P59_pass1", "P59_pass2"):
        r = joint["heldout"][name]
        bc = r.get("by_class") or {}
        lines.append(
            f"| {name} | {r['MSP_RELAXED']:.4f} | {r['MSP_12S']:.4f} | {r['MSP_NEW']:.4f} | "
            f"{(bc.get('A') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('B') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('C') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('D') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{r['zero_success_ics']} |"
        )
    lines += ["", "## Paired held-out ΔMSP (mean, 95% CI)", ""]
    for k, v in joint["paired_heldout"].items():
        lines.append(
            f"- **{k}**: meanΔ={v['mean_delta']:.4f} "
            f"CI95=[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] "
            f"↑{v['n_improved']} ↓{v['n_regressed']} ↔{v['n_tied']}"
        )
    lines += [
        "",
        "## Independent C validation (same seeds)",
        "",
        "| Model | MSP_C | zero |",
        "|-------|------:|-----:|",
    ]
    for name in ("P57_A", "P59_pass1", "P59_pass2"):
        r = joint["val_c"][name]
        lines.append(f"| {name} | {r['MSP_RELAXED']:.4f} | {r['zero_success_ics']} |")
    lines += ["", "## Paired C-val ΔMSP", ""]
    for k, v in joint["paired_val_c"].items():
        lines.append(
            f"- **{k}**: meanΔ={v['mean_delta']:.4f} "
            f"CI95=[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] "
            f"↑{v['n_improved']} ↓{v['n_regressed']}"
        )
    lines.append("")
    (OUT / "CHECKPOINT_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_joint_eval(device=args.device)
