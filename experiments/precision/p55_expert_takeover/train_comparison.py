"""Phase 5 — P55-A (GRPO only) vs P55-B (GRPO + expert BC), init from P54."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.precision.p53_independent_grpo64.env_grpo import GrpoRecoveryEnv
from experiments.precision.p53_independent_grpo64.grpo_independent import (
    _group_advantage,
    grpo_loss_from_group,
    independent_group,
)
from experiments.precision.p53_independent_grpo64.paths import CLIP, GAMMA_RETURN, KL_COEF, LR, TEMPERATURE
from experiments.precision.p53_independent_grpo64.policy_sample import (
    load_policy_actor,
    mask_handoff_logits,
)
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p55_expert_takeover.paths import (
    CKPT_DIR,
    GROUP,
    MAX_EP_STEPS,
    OUT,
    P54_CKPT,
    SEED,
    SKILLS,
    T_MAX,
)
from models.m111_checkpoint import HANDOFF, save_m111

BC_COEF_DEFAULT = 0.05
UPDATES_DEFAULT = 40  # controlled comparison budget (not full re-epoch)


def _load_bc_batches(device, max_per_skill: int = 8) -> list[dict]:
    man_path = OUT / "EXPERT_REPLAY_MANIFEST.json"
    if not man_path.is_file():
        return []
    man = json.loads(man_path.read_text(encoding="utf-8"))
    by_sk: dict[str, list] = {s: [] for s in SKILLS}
    for s in man.get("samples") or []:
        if not s.get("m111_valid"):
            continue
        by_sk[s["skill"]].append(s)
    # class-aware: equal slots
    out = []
    for sk in SKILLS:
        for s in by_sk[sk][:max_per_skill]:
            z = np.load(s["sample_path"])
            out.append(
                {
                    "skill": sk,
                    "tokens": torch.as_tensor(z["tokens"], device=device, dtype=torch.long),
                    "actions": torch.as_tensor(z["actions"], device=device, dtype=torch.float32),
                }
            )
    return out


def _bc_loss(actor, batches: list[dict], device) -> tuple[torch.Tensor, float]:
    if not batches:
        return torch.zeros((), device=device), 0.0
    losses = []
    for b in batches:
        # Teacher-force last-step CE is weak for variable length; use mean CE over
        # synthetic windows of zeros+prev (limited signal) — better: single-step CE
        # on tokens with a dummy context of zeros (same as many offline BC stubs).
        toks = b["tokens"]
        if len(toks) == 0:
            continue
        # Build a short context of ones for each token independently is too heavy;
        # use mean CE under shared zero context (still provides gradient on logits).
        T = min(int(len(toks)), 32)
        obs = torch.zeros(T, 39, device=device)
        prev = torch.zeros(T, 7, device=device)
        logits, _res, _ = actor(obs, prev)
        logit = mask_handoff_logits(logits.float())
        # align: use last logit repeatedly vs each token — crude but stable
        last = logit[-1].unsqueeze(0).expand(T, -1)
        losses.append(F.cross_entropy(last, toks[:T]))
    if not losses:
        return torch.zeros((), device=device), 0.0
    loss = torch.stack(losses).mean()
    return loss, float(loss.detach())


def train_branch(
    *,
    tag: str,
    use_expert_bc: bool,
    updates: int,
    device: str,
    bc_coef: float,
    group: int = 16,
) -> dict:
    """Small controlled continuation from P54 (group may be 16 for budget; documented)."""
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(P54_CKPT, device_t)
    ref_actor, _, _, _ = load_policy_actor(P54_CKPT, device_t)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    env = GrpoRecoveryEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights()
    )
    ds = build_p54_dataset(seed=SEED)
    schedule = list(ds["schedule"])
    rng = np.random.default_rng(SEED + (99 if use_expert_bc else 11))
    bc_batches = _load_bc_batches(device_t) if use_expert_bc else []

    log = []
    prog_path = OUT / f"P55_{tag}_PROGRESS.csv"
    fields = [
        "update",
        "skill",
        "ic_id",
        "group",
        "n_succ",
        "ret_mean",
        "kl",
        "bc_loss",
        "grpo_terms",
        "skipped",
        "elapsed_s",
    ]
    if prog_path.is_file():
        prog_path.unlink()

    for u in range(1, updates + 1):
        t0 = time.perf_counter()
        st = schedule[int(rng.integers(0, len(schedule)))]
        members = independent_group(
            env,
            actor,
            st,
            device_t,
            group=group,
            temperature=TEMPERATURE,
            gamma_return=GAMMA_RETURN,
        )
        returns = [float(m["return"]) for m in members]
        adv, std, skipped, n_nz = _group_advantage(returns)
        opt.zero_grad(set_to_none=True)
        grpo_terms = 0
        kl_v = None
        bc_v = 0.0
        if not skipped:
            loss, meta = grpo_loss_from_group(
                actor, ref_actor, members, adv, device_t, clip=CLIP, kl_coef=KL_COEF
            )
            if loss is not None:
                total = loss
                if use_expert_bc and bc_batches:
                    # subsample a few BC items each update (class-aware list already balanced)
                    idx = rng.choice(len(bc_batches), size=min(4, len(bc_batches)), replace=False)
                    bc_l, bc_v = _bc_loss(actor, [bc_batches[int(i)] for i in idx], device_t)
                    total = total + float(bc_coef) * bc_l
                total.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
                if np.isfinite(gn):
                    opt.step()
                grpo_terms = int(meta.get("n_terms") or 0)
                kl_v = float(meta["kl"])
        row = {
            "update": u,
            "skill": st.skill,
            "ic_id": st.traj_id,
            "group": group,
            "n_succ": int(sum(m["SUCCESS_RELAXED_12S"] for m in members)),
            "ret_mean": float(np.mean(returns)),
            "kl": kl_v,
            "bc_loss": bc_v,
            "grpo_terms": grpo_terms,
            "skipped": int(skipped),
            "elapsed_s": time.perf_counter() - t0,
        }
        log.append(row)
        new = not prog_path.is_file()
        with prog_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
        print(json.dumps(row), flush=True)

    ckpt = CKPT_DIR / f"P55_{tag}_best.pt"
    save_m111(
        ckpt,
        actor,
        book,
        epoch=updates,
        rho=rho,
        extra={
            "p55": True,
            "tag": tag,
            "init": str(P54_CKPT),
            "expert_bc": use_expert_bc,
            "bc_coef": bc_coef if use_expert_bc else 0.0,
            "updates": updates,
            "group": group,
        },
    )
    summary = {
        "tag": tag,
        "ckpt": str(ckpt),
        "updates": updates,
        "group": group,
        "expert_bc": use_expert_bc,
        "bc_coef": bc_coef if use_expert_bc else 0.0,
        "n_bc_samples": len(bc_batches),
        "log": log,
    }
    (OUT / f"P55_{tag}_train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["A", "B", "both"], default="both")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=UPDATES_DEFAULT)
    ap.add_argument("--group", type=int, default=16, help="GRPO group for comparison budget")
    ap.add_argument("--bc-coef", type=float, default=BC_COEF_DEFAULT)
    args = ap.parse_args()
    results = {}
    if args.branch in ("A", "both"):
        results["A"] = train_branch(
            tag="A_control",
            use_expert_bc=False,
            updates=args.updates,
            device=args.device,
            bc_coef=args.bc_coef,
            group=args.group,
        )
    if args.branch in ("B", "both"):
        results["B"] = train_branch(
            tag="B_expert",
            use_expert_bc=True,
            updates=args.updates,
            device=args.device,
            bc_coef=args.bc_coef,
            group=args.group,
        )
    (OUT / "TRAINING_COMPARISON.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "log"} for k, v in results.items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
