"""Phase 6 — Full-IC validation (not intermediate-state eval)."""

from __future__ import annotations

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
from experiments.precision.p60_c_stable_state_grpo.paths import (
    CKPT_DIR,
    MAX_EP_STEPS,
    OUT,
    P57A_CKPT,
    P58_SELECTED,
    P59_PASS1,
    P59_PASS2,
    SEED,
    T_MAX,
    VAL_ROLLOUTS,
)


def _eval_ckpt(tag: str, ckpt: Path, starts: list, *, n: int, device_t, seed: int) -> dict:
    actor, book, rho, _ = load_policy_actor(ckpt, device_t)
    actor.eval()
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    per = []
    stage_acc = {
        "recatch": 0,
        "stable": 0,
        "handoff": 0,
        "success": 0,
        "success_and_stable": 0,
        "stable_n": 0,
        "success_and_handoff": 0,
        "handoff_n": 0,
        "n": 0,
    }
    for st in starts:
        succ = rec = stab = hand = drop = 0
        for i in range(n):
            from experiments.precision.p61_reproducible_joint_eval.seeding import stable_u32

            torch.manual_seed(seed + stable_u32("p60", st.traj_id, i) % 100000 + i * 9973)
            m = rollout_one(
                env, actor, st, device_t, temperature=TEMPERATURE, gamma_return=GAMMA_RETURN
            )
            info = m.get("info") or {}
            cd = info.get("c_diag") or {}
            s = int(m["SUCCESS_RELAXED_12S"])
            r = int(cd.get("recatch") or 0)
            sb = int(cd.get("stable_hold") or 0)
            h = int(m["handoff"])
            succ += s
            rec += r
            stab += sb
            hand += h
            drop += int(m["drop"])
            stage_acc["n"] += 1
            stage_acc["recatch"] += r
            stage_acc["stable"] += sb
            stage_acc["handoff"] += h
            stage_acc["success"] += s
            if sb:
                stage_acc["stable_n"] += 1
                stage_acc["success_and_stable"] += s
            if h:
                stage_acc["handoff_n"] += 1
                stage_acc["success_and_handoff"] += s
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

    nn = max(stage_acc["n"], 1)
    transitions = {
        "p_recatch": stage_acc["recatch"] / nn,
        "p_stable": stage_acc["stable"] / nn,
        "p_handoff": stage_acc["handoff"] / nn,
        "p_success": stage_acc["success"] / nn,
        "SUCCESS_given_stable": (
            stage_acc["success_and_stable"] / max(stage_acc["stable_n"], 1)
        ),
        "SUCCESS_given_handoff": (
            stage_acc["success_and_handoff"] / max(stage_acc["handoff_n"], 1)
        ),
        "n_rollouts": stage_acc["n"],
        "n_stable": stage_acc["stable_n"],
        "n_handoff": stage_acc["handoff_n"],
    }
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
        "mean_recatch": float(np.mean([r["p_recatch"] for r in per])),
        "mean_stable": float(np.mean([r["p_stable"] for r in per])),
        "mean_handoff": float(np.mean([r["p_handoff"] for r in per])),
        "stage_transitions": transitions,
    }


def run_validation(
    *,
    device: str = "cuda",
    ckpts_extra: dict | None = None,
    tag_suffix: str = "",
    include_baselines: bool = True,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads(P58_SELECTED.read_text(encoding="utf-8"))
    ds = build_p54_dataset(seed=SEED)
    lookup = {(s.skill, s.traj_id): s for s in ds["schedule"]}
    starts_c = [lookup[("C", tid)] for tid in sel["val_c"]]
    starts_abd = []
    for lab, sk in (("A", "A1"), ("B", "B1"), ("D", "D")):
        for tid in sel["val_abd"][lab]:
            starts_abd.append(lookup[(sk, tid)])

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    ckpts = {}
    if include_baselines:
        ckpts["P59_pass1"] = P59_PASS1
        ckpts["P59_pass2"] = P59_PASS2
    if ckpts_extra:
        ckpts.update(ckpts_extra)
    for name in ("P60_pass1", "P60_pass2"):
        p = CKPT_DIR / f"{name}.pt"
        if p.is_file() and name not in ckpts:
            ckpts[name] = p

    reports_c, reports_abd = {}, {}
    for name, ckpt in ckpts.items():
        if not Path(ckpt).is_file():
            reports_c[name] = {"error": f"missing {ckpt}"}
            continue
        print(f"=== val C {name}{tag_suffix} ===", flush=True)
        reports_c[name] = _eval_ckpt(
            name, Path(ckpt), starts_c, n=VAL_ROLLOUTS, device_t=device_t, seed=SEED + 580
        )
        print(f"=== val ABD {name}{tag_suffix} ===", flush=True)
        reports_abd[name] = _eval_ckpt(
            name, Path(ckpt), starts_abd, n=VAL_ROLLOUTS, device_t=device_t, seed=SEED + 581
        )

    base = {(r["skill"], r["traj_id"]): r for r in (reports_c.get("P59_pass1") or {}).get("per_ic") or []}
    paired = {}
    for name in reports_c:
        if name == "P59_pass1" or "error" in (reports_c[name] or {}):
            continue
        deltas = []
        for r in reports_c[name].get("per_ic") or []:
            b = base.get((r["skill"], r["traj_id"]))
            if b:
                deltas.append(r["p_relaxed"] - b["p_relaxed"])
        if deltas:
            paired[f"{name}_vs_P59_pass1"] = {
                "mean_delta_c": float(np.mean(deltas)),
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
        "paired_c_vs_p59_pass1": paired,
        "per_ic_c": {k: (v or {}).get("per_ic") for k, v in reports_c.items()},
        "per_ic_abd": {k: (v or {}).get("per_ic") for k, v in reports_abd.items()},
    }
    (OUT / f"VALIDATION{tag_suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_val_md(out)
    _write_stage_md(out)
    return out


def _write_val_md(out: dict):
    lines = [
        "# VALIDATION_REPORT",
        "",
        "Full-episode rollouts from original ICs (not restored intermediate states).",
        f"Same P58 val set: {len(out['val_c_ids'])} C × {out['n_rollouts']} + ABD regression.",
        "",
        "| Model | MSP_C | recatch | stable | handoff | zero |",
        "|-------|------:|--------:|-------:|--------:|-----:|",
    ]
    order = ("P59_pass1", "P59_pass2", "P60_pass1", "P60_pass2")
    for name in order:
        r = out["reports_c"].get(name)
        if not r or "error" in r:
            continue
        lines.append(
            f"| {name} | {r['MSP_RELAXED']:.4f} | {r.get('mean_recatch', float('nan')):.3f} | "
            f"{r.get('mean_stable', float('nan')):.3f} | {r.get('mean_handoff', float('nan')):.3f} | "
            f"{r['zero_success_ics']} |"
        )
    lines += ["", "## ABD", ""]
    for name in order:
        r = out["reports_abd"].get(name)
        if not r or "error" in (r or {}):
            continue
        lines.append(f"- **{name}**: overall={r['MSP_RELAXED']:.4f} by_skill={r.get('by_skill')}")
    lines += ["", "## Paired C vs P59_pass1", ""]
    for k, v in (out.get("paired_c_vs_p59_pass1") or {}).items():
        lines.append(f"- {k}: meanΔ={v['mean_delta_c']:.4f} ↑{v['n_improved']} ↓{v['n_regressed']}")
    lines.append("")
    (OUT / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_stage_md(out: dict):
    lines = [
        "# C_STAGE_TRANSITIONS",
        "",
        "From full-IC C validation rollouts.",
        "",
        "| Model | p_recatch | p_stable | p_handoff | p_success | SUCCESS\\|stable | SUCCESS\\|handoff |",
        "|-------|----------:|---------:|----------:|----------:|-----------------:|------------------:|",
    ]
    for name in ("P59_pass1", "P59_pass2", "P60_pass1", "P60_pass2"):
        r = out["reports_c"].get(name) or {}
        tr = r.get("stage_transitions") or {}
        if not tr:
            continue
        lines.append(
            f"| {name} | {tr['p_recatch']:.3f} | {tr['p_stable']:.3f} | {tr['p_handoff']:.3f} | "
            f"{tr['p_success']:.3f} | {tr['SUCCESS_given_stable']:.3f} | {tr['SUCCESS_given_handoff']:.3f} |"
        )
    lines.append("")
    (OUT / "C_STAGE_TRANSITIONS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_validation(device=args.device, include_baselines=True)


if __name__ == "__main__":
    main()
