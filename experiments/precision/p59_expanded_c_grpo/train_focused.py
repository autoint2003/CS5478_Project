"""Two passes × 48 C ICs × Independent GRPO-64 from P58_pass2; validate after each pass."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.grpo_independent import (
    _group_advantage,
    grpo_loss_from_group,
    independent_group,
)
from experiments.precision.p53_independent_grpo64.paths import (
    CLIP,
    GAMMA_RETURN,
    KL_COEF,
    LR,
    TEMPERATURE,
)
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p59_expanded_c_grpo.paths import (
    CKPT_DIR,
    GROUP,
    MAX_EP_STEPS,
    N_PASSES,
    OUT,
    P58_PASS2,
    SEED,
    T_MAX,
)
from experiments.precision.p59_expanded_c_grpo.select_manifest import build_manifest
from experiments.precision.p59_expanded_c_grpo.validate import run_validation
from models.m111_checkpoint import save_m111

PROGRESS_FIELDS = [
    "pass",
    "group_index",
    "ic_id",
    "tag",
    "success_relaxed_count",
    "success_12s_count",
    "recatch_count",
    "stable_hold_count",
    "lost_after_recatch_count",
    "handoff_count",
    "drop_before_count",
    "reward_mean",
    "skipped_zero_var",
    "policy_kl",
    "policy_entropy",
    "loss",
    "elapsed_s",
]


def run_train(*, device: str = "cuda", skip_val: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if not P58_PASS2.is_file():
        raise FileNotFoundError(P58_PASS2)
    man_path = OUT / "TRAIN_IC_MANIFEST.json"
    if not man_path.is_file():
        build_manifest()
    man = json.loads(man_path.read_text(encoding="utf-8"))
    train_ids = man["all"]
    tags = man["tags"]
    ds = build_p54_dataset(seed=SEED)
    lookup = {s.traj_id: s for s in ds["schedule"] if s.skill == "C"}
    schedule = [lookup[tid] for tid in train_ids]

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(P58_PASS2, device_t)
    ref_actor, _, _, _ = load_policy_actor(P58_PASS2, device_t)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    progress = OUT / "P59_TRAIN_PROGRESS.csv"
    if progress.is_file():
        progress.unlink()
    t0 = time.perf_counter()
    n_updates = 0
    ckpts = []
    val_after = {}
    gi = 0
    for p_i in range(1, N_PASSES + 1):
        for st in schedule:
            gi += 1
            t1 = time.perf_counter()
            members = independent_group(
                env, actor, st, device_t, group=GROUP, temperature=TEMPERATURE, gamma_return=GAMMA_RETURN
            )
            returns = [float(m["return"]) for m in members]
            adv, std, skipped, n_nz = _group_advantage(returns)
            n_rel = int(sum(m["SUCCESS_RELAXED_12S"] for m in members))
            n_12 = int(sum(m["SUCCESS_12S"] for m in members))
            n_hand = int(sum(m["handoff"] for m in members))
            n_drop = int(sum(m["drop"] for m in members))
            n_rec = n_st = n_lost = 0
            for m in members:
                cd = (m.get("info") or {}).get("c_diag") or {}
                n_rec += int(cd.get("recatch") or 0)
                n_st += int(cd.get("stable_hold") or 0)
                n_lost += int(cd.get("lost_after_recatch") or 0)

            loss_v = kl_v = ent_v = None
            opt.zero_grad(set_to_none=True)
            if not skipped:
                loss, meta = grpo_loss_from_group(
                    actor, ref_actor, members, adv, device_t, clip=CLIP, kl_coef=KL_COEF
                )
                if loss is not None:
                    loss.backward()
                    gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
                    if np.isfinite(gn):
                        opt.step()
                        n_updates += 1
                    loss_v = float(loss.detach())
                    kl_v = float(meta["kl"])
                    ent_v = float(meta["entropy_mean"])

            row = {
                "pass": p_i,
                "group_index": gi,
                "ic_id": st.traj_id,
                "tag": tags.get(st.traj_id, ""),
                "success_relaxed_count": n_rel,
                "success_12s_count": n_12,
                "recatch_count": n_rec,
                "stable_hold_count": n_st,
                "lost_after_recatch_count": n_lost,
                "handoff_count": n_hand,
                "drop_before_count": n_drop,
                "reward_mean": float(np.mean(returns)),
                "skipped_zero_var": int(skipped),
                "policy_kl": kl_v,
                "policy_entropy": ent_v,
                "loss": loss_v,
                "elapsed_s": time.perf_counter() - t1,
            }
            new = not progress.is_file()
            with progress.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(
                json.dumps(
                    {
                        "pass": p_i,
                        "idx": gi,
                        "ic": st.traj_id,
                        "succ": n_rel,
                        "recatch": n_rec,
                        "stable": n_st,
                        "hand": n_hand,
                    }
                ),
                flush=True,
            )

        ckpt = CKPT_DIR / f"P59_pass{p_i}.pt"
        save_m111(
            ckpt,
            actor,
            book,
            epoch=p_i,
            rho=rho,
            extra={
                "p59": True,
                "pass": p_i,
                "init": str(P58_PASS2),
                "n_train_c": len(schedule),
                "group": GROUP,
            },
        )
        ckpts.append(str(ckpt))
        print(f"SAVED {ckpt}", flush=True)
        if not skip_val:
            # After pass1: eval new ckpt only; after pass2: full baseline compare
            extra = {f"P59_pass{p_i}": ckpt}
            if p_i == N_PASSES and (CKPT_DIR / "P59_pass1.pt").is_file():
                extra["P59_pass1"] = CKPT_DIR / "P59_pass1.pt"
            val_after[f"pass{p_i}"] = run_validation(
                device=device,
                ckpts_extra=extra,
                tag_suffix=f"_after_pass{p_i}",
                include_baselines=(p_i == N_PASSES),
            )

    summary = {
        "n_passes": N_PASSES,
        "n_ics": len(schedule),
        "n_groups": gi,
        "n_rollouts": gi * GROUP,
        "n_updates": n_updates,
        "elapsed_s": time.perf_counter() - t0,
        "ckpts": ckpts,
        "init": str(P58_PASS2),
    }
    (OUT / "P59_TRAIN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_training_report(progress, summary, tags)
    _write_stage_diagnostics(progress, tags)
    return {**summary, "val_after": {k: True for k in val_after}}


def _write_training_report(progress: Path, summary: dict, tags: dict):
    by_pass = {1: [], 2: []}
    for r in csv.DictReader(progress.open(encoding="utf-8")):
        by_pass[int(r["pass"])].append(r)
    lines = [
        "# TRAINING_REPORT",
        "",
        f"- Init: `P58_pass2.pt`; reward = P57-A (no stab / no expert BC)",
        f"- {summary['n_ics']} C ICs × {summary['n_passes']} passes × G={GROUP} = **{summary['n_rollouts']}** rollouts",
        f"- Updates: {summary['n_updates']}; wall {summary['elapsed_s']:.0f}s",
        "",
        "| Pass | mean p_relaxed | mean recatch | mean stable | mean handoff | mean KL |",
        "|-----:|---------------:|-------------:|------------:|-------------:|--------:|",
    ]
    for p_i in (1, 2):
        sub = by_pass[p_i]
        kls = [float(r["policy_kl"]) for r in sub if r["policy_kl"] not in (None, "")]
        lines.append(
            f"| {p_i} | {np.mean([int(r['success_relaxed_count']) for r in sub])/GROUP:.4f} | "
            f"{np.mean([int(r['recatch_count']) for r in sub])/GROUP:.3f} | "
            f"{np.mean([int(r['stable_hold_count']) for r in sub])/GROUP:.3f} | "
            f"{np.mean([int(r['handoff_count']) for r in sub])/GROUP:.3f} | "
            f"{(np.mean(kls) if kls else float('nan')):.5f} |"
        )
    # Core vs added
    lines += ["", "## Core-16 vs added-32 (pass 2 train groups)", ""]
    p2 = by_pass[2]
    for label, pred in (
        ("p58_core", lambda t: t == "p58_core"),
        ("added", lambda t: t != "p58_core"),
    ):
        sub = [r for r in p2 if pred(r.get("tag") or tags.get(r["ic_id"], ""))]
        if not sub:
            continue
        lines.append(
            f"- **{label}** (n={len(sub)}): p_rel={np.mean([int(r['success_relaxed_count']) for r in sub])/GROUP:.4f}, "
            f"recatch={np.mean([int(r['recatch_count']) for r in sub])/GROUP:.3f}, "
            f"stable={np.mean([int(r['stable_hold_count']) for r in sub])/GROUP:.3f}"
        )
    lines.append("")
    (OUT / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_stage_diagnostics(progress: Path, tags: dict):
    rows = list(csv.DictReader(progress.open(encoding="utf-8")))
    lines = [
        "# C_STAGE_DIAGNOSTICS",
        "",
        "Train-time stage rates from StabilizeGrpoEnv c_diag (recatch / stable_hold gates).",
        "",
    ]
    for p_i in (1, 2):
        sub = [r for r in rows if int(r["pass"]) == p_i]
        rec = sum(int(r["recatch_count"]) for r in sub)
        stab = sum(int(r["stable_hold_count"]) for r in sub)
        hand = sum(int(r["handoff_count"]) for r in sub)
        succ = sum(int(r["success_relaxed_count"]) for r in sub)
        tot = len(sub) * GROUP
        zero = sum(1 for r in sub if int(r["success_relaxed_count"]) == 0)
        lines += [
            f"## Pass {p_i}",
            "",
            f"| Metric | Rate |",
            f"|--------|-----:|",
            f"| Recatch | {rec/tot:.4f} |",
            f"| Stable hold | {stab/tot:.4f} |",
            f"| Stable \| recatch | {(stab/rec if rec else 0):.4f} |",
            f"| HANDOFF | {hand/tot:.4f} |",
            f"| SUCCESS \| recatch | {(succ/rec if rec else 0):.4f} |",
            f"| SUCCESS \| stable | {(succ/stab if stab else 0):.4f} |",
            f"| SUCCESS_RELAXED | {succ/tot:.4f} |",
            f"| Zero-success groups | {zero}/{len(sub)} |",
            "",
        ]
    # Attribution: pass1→pass2 delta
    def rates(p):
        sub = [r for r in rows if int(r["pass"]) == p]
        tot = len(sub) * GROUP
        return {
            "recatch": sum(int(r["recatch_count"]) for r in sub) / tot,
            "stable": sum(int(r["stable_hold_count"]) for r in sub) / tot,
            "succ": sum(int(r["success_relaxed_count"]) for r in sub) / tot,
            "hand": sum(int(r["handoff_count"]) for r in sub) / tot,
        }

    r1, r2 = rates(1), rates(2)
    lines += [
        "## Pass1 → Pass2 deltas (train)",
        "",
        f"- Δrecatch={r2['recatch']-r1['recatch']:+.4f}",
        f"- Δstable={r2['stable']-r1['stable']:+.4f}",
        f"- Δhandoff={r2['hand']-r1['hand']:+.4f}",
        f"- Δsuccess={r2['succ']-r1['succ']:+.4f}",
        "",
        "If Δsuccess tracks Δrecatch more than Δ(success|stable), gains are acquisition-driven; "
        "if success|stable rises with flat recatch, gains are completion-driven.",
        "",
    ]
    (OUT / "C_STAGE_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "C_STAGE_DIAGNOSTICS.json").write_text(
        json.dumps({"pass1": r1, "pass2": r2, "delta": {k: r2[k] - r1[k] for k in r1}}, indent=2),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-val", action="store_true")
    args = ap.parse_args()
    print(json.dumps({k: v for k, v in run_train(device=args.device, skip_val=args.skip_val).items() if k != "val_after"}, indent=2))


if __name__ == "__main__":
    main()
