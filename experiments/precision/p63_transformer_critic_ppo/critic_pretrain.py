"""Phase 3 — Critic warm-start on on-policy P62_epoch3 Monte Carlo returns."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (
    init_critic_from_actor,
    load_actor,
    save_pair,
)
from experiments.precision.p63_transformer_critic_ppo.gae import explained_variance
from experiments.precision.p63_transformer_critic_ppo.paths import (
    CKPT_DIR,
    CRITIC_LR,
    CRITIC_WARM_EPOCHS,
    CRITIC_WARM_TRAJ,
    GAMMA,
    GROUP_PILOT,
    OUT,
    P62_EPOCH3,
    SEED,
)
from experiments.precision.p63_transformer_critic_ppo.rollout import collect_schedule, make_env
from experiments.precision.p63_transformer_critic_ppo.sampling import pad_batch


def _mc_returns_from_batch(batch: dict, gamma: float = GAMMA) -> np.ndarray:
    """Recompute Monte Carlo returns from rewards stored indirectly via ret≈adv+V.

    Warm-start uses complete on-policy trajectories; ``batch['ret']`` from GAE with
    λ=1, γ=1 equals MC return when bootstrap=0. We force that here by regenerating
    from episode structure when available — for warm-start we collect with GAE λ=1.
    """
    return np.asarray(batch["ret"], dtype=np.float32)


def train_critic_mc(
    critic,
    batch: dict,
    *,
    epochs: int = CRITIC_WARM_EPOCHS,
    lr: float = CRITIC_LR,
    minibatch: int = 64,
) -> dict:
    device = next(critic.parameters()).device
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    targets = torch.as_tensor(_mc_returns_from_batch(batch), device=device, dtype=torch.float32)
    n = len(batch["obs"])
    losses = []
    for _ in range(epochs):
        idxs = np.random.permutation(n)
        for start in range(0, n, minibatch):
            mb = idxs[start : start + minibatch]
            o, p, lengths, pad_mask = pad_batch(
                [batch["obs"][i] for i in mb],
                [batch["prev"][i] for i in mb],
                device,
            )
            vals = critic(o, p, pad_mask=pad_mask)
            last = torch.stack([vals[j, lengths[j] - 1] for j in range(len(mb))])
            loss = F.mse_loss(last, targets[mb])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
    with torch.no_grad():
        # full pass predictions
        preds = []
        for start in range(0, n, minibatch):
            mb = list(range(start, min(start + minibatch, n)))
            o, p, lengths, pad_mask = pad_batch(
                [batch["obs"][i] for i in mb],
                [batch["prev"][i] for i in mb],
                device,
            )
            vals = critic(o, p, pad_mask=pad_mask)
            last = torch.stack([vals[j, lengths[j] - 1] for j in range(len(mb))])
            preds.append(last.cpu().numpy())
        pred = np.concatenate(preds)
    y = targets.cpu().numpy()
    return {
        "mean_loss": float(np.mean(losses)) if losses else None,
        "explained_var": explained_variance(pred, y),
        "pred_mean": float(pred.mean()),
        "target_mean": float(y.mean()),
        "pred_std": float(pred.std()),
        "target_std": float(y.std()),
        "mae": float(np.mean(np.abs(pred - y))),
    }


def run_critic_pretrain(*, device: str = "cuda", n_ics: int | None = None) -> dict:
    """Collect on-policy traj from frozen P62 actor; fit critic to MC returns."""
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P62_EPOCH3, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()
    critic = init_critic_from_actor(actor).to(device_t)

    ds = build_p54_dataset(seed=SEED)
    # Sample a diverse subset covering A/B/C/D for warm-start
    schedule = list(ds["schedule"])
    rng = np.random.default_rng(SEED + 63)
    # Aim for ~CRITIC_WARM_TRAJ episodes with GROUP_PILOT traj/IC
    n_needed = max(16, int(np.ceil(CRITIC_WARM_TRAJ / GROUP_PILOT)))
    if n_ics is not None:
        n_needed = n_ics
    # stratified sample
    by = {"A1": [], "B1": [], "C": [], "D": []}
    for st in schedule:
        by[st.skill].append(st)
    picked = []
    per = max(1, n_needed // 4)
    for sk in ("A1", "B1", "C", "D"):
        pool = by[sk]
        idx = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        picked.extend(pool[int(i)] for i in idx)
    while len(picked) < n_needed:
        picked.append(schedule[int(rng.integers(0, len(schedule)))])
    picked = picked[:n_needed]

    env = make_env(book, rho)
    print(f"=== Critic warm-start: {len(picked)} ICs × {GROUP_PILOT} traj ===", flush=True)
    # Use λ=1 so GAE returns == MC returns under γ=1 and terminal bootstrap 0
    from experiments.precision.p63_transformer_critic_ppo import rollout as roll_mod

    batches = []
    stats_rows = []
    for i, st in enumerate(picked):
        b, s = roll_mod.collect_ic_group(
            env, actor, critic, st, device_t, n_traj=GROUP_PILOT, gamma=GAMMA, gae_lambda=1.0
        )
        batches.append(b)
        stats_rows.append(s)
        if (i + 1) % 4 == 0 or i == 0:
            print(f"  warm {i+1}/{len(picked)} {s['skill']} ret={s['mean_return']:.2f}", flush=True)
    batch = roll_mod.merge_batches(batches)

    before = {
        "explained_var": explained_variance(batch["values"], batch["ret"]),
        "mae": float(np.mean(np.abs(batch["values"] - batch["ret"]))),
    }
    fit = train_critic_mc(critic, batch)
    # Per-class / success diagnostics on post-fit predictions
    # Re-evaluate values
    device = device_t
    preds = []
    mb = 64
    with torch.no_grad():
        for start in range(0, len(batch["obs"]), mb):
            idx = list(range(start, min(start + mb, len(batch["obs"]))))
            o, p, lengths, pad_mask = pad_batch(
                [batch["obs"][i] for i in idx],
                [batch["prev"][i] for i in idx],
                device,
            )
            vals = critic(o, p, pad_mask=pad_mask)
            last = torch.stack([vals[j, lengths[j] - 1] for j in range(len(idx))])
            preds.append(last.cpu().numpy())
    pred = np.concatenate(preds)
    y = batch["ret"]

    per_class = defaultdict(list)
    for s in stats_rows:
        cl = {"A1": "A", "B1": "B", "C": "C", "D": "D"}[s["skill"]]
        per_class[cl].append(s["mean_return"])

    # success vs fail using episode returns vs typical success terminal (+25ish)
    # Use per-episode mean return from stats
    succ_rets = [s["mean_return"] for s in stats_rows if s["p_relaxed"] > 0]
    fail_rets = [s["mean_return"] for s in stats_rows if s["p_relaxed"] == 0]

    sanity = {
        "before": before,
        "after": fit,
        "per_class_mean_return": {c: float(np.mean(v)) for c, v in per_class.items()},
        "succ_traj_mean_return": float(np.mean(succ_rets)) if succ_rets else None,
        "fail_traj_mean_return": float(np.mean(fail_rets)) if fail_rets else None,
        "n_ics": len(picked),
        "n_steps": int(sum(s["n_steps"] for s in stats_rows)),
        "pred_vs_return_corr": float(np.corrcoef(pred, y)[0, 1]) if len(pred) > 2 else None,
    }
    # Sanity gates
    gates = {
        "explained_var_positive": bool(fit["explained_var"] > 0.05),
        "finite_loss": bool(np.isfinite(fit["mean_loss"] or np.nan)),
        "corr_positive": bool((sanity["pred_vs_return_corr"] or 0) > 0.2),
        "improved_vs_init": bool(fit["explained_var"] >= before["explained_var"] - 1e-6),
    }
    gates["pass"] = bool(all(gates.values()))

    actor_path = CKPT_DIR / "P63_actor_init.pt"
    critic_path = CKPT_DIR / "P63_critic_warm.pt"
    # Reload trainable actor copy for saving init
    actor_train, book, rho, _ = load_actor(P62_EPOCH3, device_t)
    save_pair(
        actor_path,
        critic_path,
        actor_train,
        critic,
        book,
        rho,
        epoch=0,
        extra={"critic_warm": True, "gates": gates},
    )

    report = {"sanity": sanity, "gates": gates, "actor": str(actor_path), "critic": str(critic_path)}
    (OUT / "CRITIC_PRETRAIN_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CRITIC_PRETRAIN_REPORT",
        "",
        f"On-policy warm-start from frozen P62_epoch3 ({len(picked)} ICs × {GROUP_PILOT}).",
        "",
        f"- Explained variance before → after: **{before['explained_var']:.3f} → {fit['explained_var']:.3f}**",
        f"- MAE: {before['mae']:.3f} → {fit['mae']:.3f}",
        f"- corr(pred, return): {sanity['pred_vs_return_corr']}",
        f"- succ mean return: {sanity['succ_traj_mean_return']}; fail: {sanity['fail_traj_mean_return']}",
        f"- per-class mean return: {sanity['per_class_mean_return']}",
        "",
        f"**Gates pass = {gates['pass']}** ({gates})",
        "",
        f"Saved `{critic_path.name}` (actor init copy `{actor_path.name}`).",
        "",
        "Expert trajectories were **not** used.",
        "",
    ]
    (OUT / "CRITIC_PRETRAIN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-ics", type=int, default=None)
    args = ap.parse_args()
    print(json.dumps(run_critic_pretrain(device=args.device, n_ics=args.n_ics), indent=2))
