"""Phase 5 — Independent GRPO-64 from restored stable intermediate states (2 passes)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.grpo_independent import (
    _group_advantage,
    grpo_loss_from_group,
)
from experiments.precision.p53_independent_grpo64.paths import (
    CLIP,
    GAMMA_RETURN,
    KL_COEF,
    LR,
)
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.continuation import (
    continuation_group,
    load_state_pkl,
)
from experiments.precision.p60_c_stable_state_grpo.paths import (
    CKPT_DIR,
    GROUP,
    MAX_EP_STEPS,
    MAX_TRAIN_CONTINUATIONS,
    N_PASSES,
    OUT,
    SEED,
    T_MAX,
    TEMPERATURE,
)
from models.m111_checkpoint import save_m111

PROGRESS_FIELDS = [
    "pass",
    "group_index",
    "state_id",
    "ic_id",
    "pattern",
    "success_relaxed_count",
    "handoff_count",
    "drop_count",
    "lost_contact_count",
    "truncated_count",
    "reward_mean",
    "adv_std",
    "skipped_zero_var",
    "policy_kl",
    "policy_entropy",
    "loss",
    "remaining_steps",
    "elapsed_s",
]


def _load_train_states() -> tuple[dict, list[dict], dict]:
    man = json.loads((OUT / "STATE_SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    probe = {}
    pp = OUT / "FEASIBILITY_PROBE.json"
    if pp.is_file():
        probe = json.loads(pp.read_text(encoding="utf-8"))
    retain = {
        r["state_id"]: r
        for r in (probe.get("per_state") or [])
        if r.get("retain_for_train")
    }
    states = []
    for row in man["states"]:
        if retain and row["state_id"] not in retain:
            continue
        states.append(row)
    if not states:
        states = list(man["states"])
    return man, states, probe


def run_train(*, device: str = "cuda", skip_val: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    man, states, probe = _load_train_states()
    n_cont = len(states) * GROUP * N_PASSES
    if n_cont > MAX_TRAIN_CONTINUATIONS:
        # Trim states to fit budget
        max_states = MAX_TRAIN_CONTINUATIONS // (GROUP * N_PASSES)
        states = states[:max_states]
        n_cont = len(states) * GROUP * N_PASSES
    assert n_cont <= MAX_TRAIN_CONTINUATIONS

    ckpt_init = Path(man["init_checkpoint"]["path"])
    ds = build_p54_dataset(seed=SEED)
    lookup = {s.traj_id: s for s in ds["schedule"] if s.skill == "C"}

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(ckpt_init, device_t)
    ref_actor, _, _, _ = load_policy_actor(ckpt_init, device_t)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    # Preload pickled states
    packed = []
    for row in states:
        packed.append((row, load_state_pkl(Path(row["path"])), lookup[row["ic_id"]]))

    progress = OUT / "P60_TRAIN_PROGRESS.csv"
    if progress.is_file():
        progress.unlink()
    t0 = time.perf_counter()
    n_updates = 0
    ckpts = []
    gi = 0
    pass_stats = []

    for p_i in range(1, N_PASSES + 1):
        pass_succ = pass_hand = pass_lost = 0
        pass_groups = 0
        for row, state, st in packed:
            gi += 1
            pass_groups += 1
            t1 = time.perf_counter()
            members = continuation_group(
                env,
                actor,
                st,
                state,
                device_t,
                group=GROUP,
                temperature=TEMPERATURE,
                gamma_return=GAMMA_RETURN,
            )
            returns = [float(m["return"]) for m in members]
            adv, std, skipped, n_nz = _group_advantage(returns)
            n_rel = int(sum(m["SUCCESS_RELAXED_12S"] for m in members))
            n_hand = int(sum(m["handoff"] for m in members))
            n_drop = int(sum(m["drop"] for m in members))
            n_lost = int(sum(m.get("lost_contact") or 0 for m in members))
            n_trunc = int(sum(m.get("truncated") or 0 for m in members))
            pass_succ += n_rel
            pass_hand += n_hand
            pass_lost += n_lost

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

            row_csv = {
                "pass": p_i,
                "group_index": gi,
                "state_id": row["state_id"],
                "ic_id": row["ic_id"],
                "pattern": row["failure_pattern"],
                "success_relaxed_count": n_rel,
                "handoff_count": n_hand,
                "drop_count": n_drop,
                "lost_contact_count": n_lost,
                "truncated_count": n_trunc,
                "reward_mean": float(np.mean(returns)),
                "adv_std": float(std),
                "skipped_zero_var": int(skipped),
                "policy_kl": "" if kl_v is None else kl_v,
                "policy_entropy": "" if ent_v is None else ent_v,
                "loss": "" if loss_v is None else loss_v,
                "remaining_steps": row["remaining_steps"],
                "elapsed_s": time.perf_counter() - t1,
            }
            write_header = not progress.is_file()
            with progress.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
                if write_header:
                    w.writeheader()
                w.writerow(row_csv)
            print(
                f"[P60 p{p_i} {gi}] {row['state_id']} succ={n_rel}/{GROUP} hand={n_hand} "
                f"lost={n_lost} skip={int(skipped)} kl={kl_v} ({row_csv['elapsed_s']:.1f}s)",
                flush=True,
            )

        ckpt_path = CKPT_DIR / f"P60_pass{p_i}.pt"
        save_m111(
            ckpt_path, actor, book, epoch=p_i, rho=rho, extra={"experiment": "P60", "pass": p_i}
        )
        ckpts.append(str(ckpt_path))
        denom = pass_groups * GROUP
        pass_stats.append(
            {
                "pass": p_i,
                "n_groups": pass_groups,
                "p_success": pass_succ / max(denom, 1),
                "p_handoff": pass_hand / max(denom, 1),
                "p_lost": pass_lost / max(denom, 1),
                "ckpt": str(ckpt_path),
            }
        )
        print(f"=== saved {ckpt_path} p_succ={pass_stats[-1]['p_success']:.4f} ===", flush=True)

        if not skip_val:
            from experiments.precision.p60_c_stable_state_grpo.validate import run_validation

            run_validation(
                device=device,
                ckpts_extra={f"P60_pass{p_i}": ckpt_path},
                tag_suffix=f"_after_pass{p_i}",
                include_baselines=(p_i == N_PASSES),
            )

    summary = {
        "init_checkpoint": man["init_checkpoint"],
        "n_states": len(states),
        "group": GROUP,
        "n_passes": N_PASSES,
        "n_continuations": n_cont,
        "n_updates": n_updates,
        "wall_s": time.perf_counter() - t0,
        "pass_stats": pass_stats,
        "ckpts": ckpts,
        "probe_viable": probe.get("viable_for_training"),
        "state_ids": [r["state_id"] for r in states],
    }
    (OUT / "P60_TRAIN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# TRAINING_REPORT",
        "",
        f"Init: `{man['init_checkpoint']['selected']}`",
        f"States × group × passes = {len(states)} × {GROUP} × {N_PASSES} = **{n_cont}** continuations "
        f"(cap {MAX_TRAIN_CONTINUATIONS}).",
        f"Updates: {n_updates}. Wall: {summary['wall_s']/3600:.2f} h.",
        "",
        "Unchanged: M111, P57-A reward, frozen KL ref, R3F HANDOFF, SUCCESS_RELAXED_12S.",
        "No expert BC / new reward shaping. Remaining episode time budget preserved.",
        "",
        "| Pass | p_success (cont.) | p_handoff | p_lost | ckpt |",
        "|-----:|------------------:|----------:|-------:|------|",
    ]
    for ps in pass_stats:
        lines.append(
            f"| {ps['pass']} | {ps['p_success']:.4f} | {ps['p_handoff']:.4f} | "
            f"{ps['p_lost']:.4f} | `{Path(ps['ckpt']).name}` |"
        )
    lines.append("")
    (OUT / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-val", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_train(device=args.device, skip_val=args.skip_val), indent=2))
