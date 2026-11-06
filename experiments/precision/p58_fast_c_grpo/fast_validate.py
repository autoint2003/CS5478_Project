"""Phase 4 — Fast validation: 8 C × 32 + small A/B/D set; paired seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.grpo_independent import rollout_one
from experiments.precision.p53_independent_grpo64.paths import GAMMA_RETURN, TEMPERATURE
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p58_fast_c_grpo.paths import (
    CKPT_DIR,
    MAX_EP_STEPS,
    OUT,
    P57A_CKPT,
    SEED,
    T_MAX,
    VAL_ABD_PER,
    VAL_ROLLOUTS,
)


def _eval_ckpt(tag: str, ckpt: Path, starts: list, *, n: int, device_t, seed: int) -> dict:
    actor, book, rho, _ = load_policy_actor(ckpt, device_t)
    actor.eval()
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    # Seed for paired comparisons: same order
    rng = np.random.default_rng(seed)
    _ = rng  # reserved if we need draws; rollouts are policy-stochastic under torch
    torch.manual_seed(seed)
    per = []
    for st in starts:
        torch.manual_seed(seed + hash(st.traj_id) % 100000)
        succ = rec = stab = hand = drop = 0
        for i in range(n):
            torch.manual_seed(seed + hash(st.traj_id) % 100000 + i * 9973)
            m = rollout_one(
                env, actor, st, device_t, temperature=TEMPERATURE, gamma_return=GAMMA_RETURN
            )
            info = m.get("info") or {}
            cd = info.get("c_diag") or {}
            succ += int(m["SUCCESS_RELAXED_12S"])
            rec += int(cd.get("recatch") or 0)
            stab += int(cd.get("stable_hold") or 0)
            hand += int(m["handoff"])
            drop += int(m["drop"])
        per.append(
            {
                "traj_id": st.traj_id,
                "skill": st.skill,
                "p_relaxed": succ / n,
                "p_recatch": rec / n,
                "p_stable": stab / n,
                "p_handoff": hand / n,
                "p_drop": drop / n,
                "zero": int(succ == 0),
            }
        )
        print(f"[{tag}] {st.skill} {st.traj_id}: p_rel={succ/n:.3f}", flush=True)
    return {
        "tag": tag,
        "ckpt": str(ckpt),
        "n_rollouts": n,
        "MSP_RELAXED": float(np.mean([r["p_relaxed"] for r in per])),
        "per_ic": per,
        "by_skill": {
            sk: float(np.mean([r["p_relaxed"] for r in per if r["skill"] == sk]))
            for sk in sorted({r["skill"] for r in per})
        },
        "zero_success_ics": int(sum(r["zero"] for r in per)),
    }


def run_val(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((OUT / "SELECTED_C_ICS.json").read_text(encoding="utf-8"))
    ds = build_p54_dataset(seed=SEED)
    lookup = {(s.skill, s.traj_id): s for s in ds["schedule"]}
    starts_c = []
    for tid in sel["val_c"]:
        s = lookup.get(("C", tid))
        if s is None:
            raise RuntimeError(f"missing val C {tid}")
        starts_c.append(s)
    starts_abd = []
    for lab, sk in (("A", "A1"), ("B", "B1"), ("D", "D")):
        for tid in sel["val_abd"][lab]:
            s = lookup.get((sk, tid))
            if s is None:
                raise RuntimeError(f"missing val {sk} {tid}")
            starts_abd.append(s)

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    ckpts = {
        "P57_A": P57A_CKPT,
        "P58_pass1": CKPT_DIR / "P58_pass1.pt",
        "P58_pass2": CKPT_DIR / "P58_pass2.pt",
    }
    reports_c = {}
    reports_abd = {}
    for name, ckpt in ckpts.items():
        if not Path(ckpt).is_file():
            reports_c[name] = {"error": f"missing {ckpt}"}
            continue
        print(f"=== val C {name} ===", flush=True)
        reports_c[name] = _eval_ckpt(
            name, Path(ckpt), starts_c, n=VAL_ROLLOUTS, device_t=device_t, seed=SEED + 580
        )
        print(f"=== val ABD {name} ===", flush=True)
        reports_abd[name] = _eval_ckpt(
            name, Path(ckpt), starts_abd, n=VAL_ROLLOUTS, device_t=device_t, seed=SEED + 581
        )

    # Paired vs P57-A on C
    paired = {}
    base = {(r["skill"], r["traj_id"]): r for r in (reports_c.get("P57_A") or {}).get("per_ic") or []}
    for name in ("P58_pass1", "P58_pass2"):
        deltas = []
        for r in (reports_c.get(name) or {}).get("per_ic") or []:
            b = base.get((r["skill"], r["traj_id"]))
            if b:
                deltas.append(r["p_relaxed"] - b["p_relaxed"])
        paired[name] = {
            "mean_delta_c": float(np.mean(deltas)) if deltas else None,
            "n_improved": int(sum(d > 1e-9 for d in deltas)),
            "n_regressed": int(sum(d < -1e-9 for d in deltas)),
        }

    out = {
        "val_c_ids": sel["val_c"],
        "val_abd": sel["val_abd"],
        "n_rollouts": VAL_ROLLOUTS,
        "reports_c": {
            k: {kk: vv for kk, vv in (v or {}).items() if kk != "per_ic"} for k, v in reports_c.items()
        },
        "reports_abd": {
            k: {kk: vv for kk, vv in (v or {}).items() if kk != "per_ic"} for k, v in reports_abd.items()
        },
        "paired_c_vs_p57a": paired,
        "per_ic_c": {k: (v or {}).get("per_ic") for k, v in reports_c.items()},
        "per_ic_abd": {k: (v or {}).get("per_ic") for k, v in reports_abd.items()},
    }
    (OUT / "FAST_VALIDATION.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# FAST_VALIDATION",
        "",
        f"C val: {len(sel['val_c'])} ICs × {VAL_ROLLOUTS} rolls. ABD: {VAL_ABD_PER}/class × {VAL_ROLLOUTS}.",
        "Not the repeatedly tested final held-out set; ICs from P54 train remainder.",
        "",
        "## C validation MSP_RELAXED",
        "",
        "| Model | MSP_C | mean recatch | mean stable | zero ICs |",
        "|-------|------:|-------------:|------------:|---------:|",
    ]
    for name in ("P57_A", "P58_pass1", "P58_pass2"):
        r = reports_c.get(name) or {}
        if "error" in r:
            lines.append(f"| {name} | ERROR | | | |")
            continue
        per = r.get("per_ic") or []
        lines.append(
            f"| {name} | {r.get('MSP_RELAXED'):.4f} | "
            f"{np.mean([x['p_recatch'] for x in per]):.3f} | "
            f"{np.mean([x['p_stable'] for x in per]):.3f} | "
            f"{r.get('zero_success_ics')} |"
        )
    lines += ["", "## A/B/D validation MSP (catastrophic-regression check)", ""]
    for name in ("P57_A", "P58_pass1", "P58_pass2"):
        r = reports_abd.get(name) or {}
        lines.append(f"- **{name}**: overall={r.get('MSP_RELAXED')} by_skill={r.get('by_skill')}")
    lines += ["", "## Paired C vs P57-A", ""]
    for k, v in paired.items():
        lines.append(f"- {k}: meanΔ={v['mean_delta_c']}, ↑{v['n_improved']} ↓{v['n_regressed']}")
    lines.append("")
    (OUT / "FAST_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_val(device=args.device)


if __name__ == "__main__":
    main()
