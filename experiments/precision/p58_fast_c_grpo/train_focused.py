"""Phase 3 — 2 passes × 16 C ICs × Independent GRPO-64 from P57-A (original reward)."""

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
from experiments.precision.p58_fast_c_grpo.paths import (
    CKPT_DIR,
    GROUP,
    MAX_EP_STEPS,
    N_PASSES,
    OUT,
    P57A_CKPT,
    SEED,
    T_MAX,
)
from experiments.precision.p58_fast_c_grpo.select_ics import select
from models.m111_checkpoint import save_m111

PROGRESS_FIELDS = [
    "pass",
    "group_index",
    "ic_id",
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
    "loss",
    "elapsed_s",
]


def run_train(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if not P57A_CKPT.is_file():
        raise FileNotFoundError(P57A_CKPT)
    sel_path = OUT / "SELECTED_C_ICS.json"
    if not sel_path.is_file():
        select()
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    train_ids = sel["train_c"]["all"]
    ds = build_p54_dataset(seed=SEED)
    lookup = {s.traj_id: s for s in ds["schedule"] if s.skill == "C"}
    schedule = []
    for tid in train_ids:
        if tid not in lookup:
            raise RuntimeError(f"missing start {tid}")
        schedule.append(lookup[tid])

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(P57A_CKPT, device_t)
    ref_actor, _, _, _ = load_policy_actor(P57A_CKPT, device_t)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    # Original P57-A reward: StabilizeGrpoEnv with use_stabilization=False
    env = StabilizeGrpoEnv(
        book,
        rho,
        max_steps=MAX_EP_STEPS,
        t_max=T_MAX,
        weights=PoseRewardWeights(),
        use_stabilization=False,
    )

    progress = OUT / "P58_TRAIN_PROGRESS.csv"
    if progress.is_file():
        progress.unlink()
    t0 = time.perf_counter()
    n_updates = 0
    ckpts = []
    gi = 0
    for p_i in range(1, N_PASSES + 1):
        for st in schedule:
            gi += 1
            t1 = time.perf_counter()
            members = independent_group(
                env,
                actor,
                st,
                device_t,
                group=GROUP,
                temperature=TEMPERATURE,
                gamma_return=GAMMA_RETURN,
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

            loss_v = kl_v = None
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

            row = {
                "pass": p_i,
                "group_index": gi,
                "ic_id": st.traj_id,
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
                "loss": loss_v,
                "elapsed_s": time.perf_counter() - t1,
            }
            new = not progress.is_file()
            with progress.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(json.dumps({"pass": p_i, "ic": st.traj_id, "succ": n_rel, "recatch": n_rec, "stable": n_st, "hand": n_hand}), flush=True)

        ckpt = CKPT_DIR / f"P58_pass{p_i}.pt"
        save_m111(
            ckpt,
            actor,
            book,
            epoch=p_i,
            rho=rho,
            extra={
                "p58": True,
                "pass": p_i,
                "init": str(P57A_CKPT),
                "n_train_c": len(schedule),
                "group": GROUP,
                "use_stabilization": False,
            },
        )
        ckpts.append(str(ckpt))
        print(f"SAVED {ckpt}", flush=True)

    summary = {
        "n_passes": N_PASSES,
        "n_ics": len(schedule),
        "n_groups": gi,
        "n_rollouts": gi * GROUP,
        "n_updates": n_updates,
        "elapsed_s": time.perf_counter() - t0,
        "ckpts": ckpts,
        "train_ids": train_ids,
        "init": str(P57A_CKPT),
    }
    (OUT / "P58_TRAIN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # TRAINING_REPORT
    by_pass = {1: [], 2: []}
    for r in csv.DictReader(progress.open(encoding="utf-8")):
        by_pass[int(r["pass"])].append(r)
    lines = [
        "# TRAINING_REPORT",
        "",
        f"- Init: P57-A (`{P57A_CKPT.name}`), original reward (no stab, no expert BC)",
        f"- Schedule: {len(schedule)} C ICs × {N_PASSES} passes × G={GROUP} = **{summary['n_rollouts']}** rollouts",
        f"- Updates: {n_updates}; wall {summary['elapsed_s']:.0f}s",
        f"- Checkpoints: {', '.join(Path(c).name for c in ckpts)}",
        "",
        "| Pass | mean p_relaxed | mean recatch | mean stable | mean handoff |",
        "|-----:|---------------:|-------------:|------------:|-------------:|",
    ]
    for p_i in (1, 2):
        sub = by_pass[p_i]
        lines.append(
            f"| {p_i} | {np.mean([int(r['success_relaxed_count']) for r in sub])/GROUP:.4f} | "
            f"{np.mean([int(r['recatch_count']) for r in sub])/GROUP:.3f} | "
            f"{np.mean([int(r['stable_hold_count']) for r in sub])/GROUP:.3f} | "
            f"{np.mean([int(r['handoff_count']) for r in sub])/GROUP:.3f} |"
        )
    lines.append("")
    (OUT / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(json.dumps(run_train(device=args.device), indent=2))


if __name__ == "__main__":
    main()
