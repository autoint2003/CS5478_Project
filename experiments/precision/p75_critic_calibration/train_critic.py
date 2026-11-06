"""Phase 3 — Critic-only training (Actor frozen)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import pad_batch
from experiments.precision.p75_critic_calibration.collect import actor_param_digest
from experiments.precision.p75_critic_calibration.dataset import (
    build_step_index,
    load_episodes,
    stage_mask,
    window_at,
)
from experiments.precision.p75_critic_calibration.paths import (
    CKPT_DIR,
    CRITIC_EPOCHS,
    CRITIC_LR,
    FC_STAGE_NAMES,
    MAX_GRAD_NORM,
    MINIBATCH,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    WEIGHTS_B,
)


def _train_one(
    critic,
    episodes,
    index,
    weights: np.ndarray,
    *,
    epochs: int,
    lr: float,
    device,
    name: str,
) -> dict:
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    n = len(index)
    # Normalize weights to mean 1 so effective LR is comparable across branches
    w = np.asarray(weights, np.float64)
    w = w * (n / max(w.sum(), 1e-9))
    losses = []
    rng = np.random.default_rng(SEED + hash(name) % 10000)
    critic.train()
    for ep in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, MINIBATCH):
            mb = order[start : start + MINIBATCH]
            obs_list, prev_list, tgt, wt = [], [], [], []
            for j in mb:
                ei, t = index[j]
                o, p = window_at(episodes[ei], t)
                obs_list.append(o)
                prev_list.append(p)
                tgt.append(float(episodes[ei]["mc_returns"][t]))
                wt.append(float(w[j]))
            o, p, lengths, pad_mask = pad_batch(obs_list, prev_list, device)
            vals = critic(o, p, pad_mask=pad_mask)
            last = torch.stack([vals[j, lengths[j] - 1] for j in range(len(mb))])
            target = torch.as_tensor(tgt, device=device, dtype=torch.float32)
            ww = torch.as_tensor(wt, device=device, dtype=torch.float32)
            per = (last - target) ** 2
            loss = (per * ww).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            opt.step()
            losses.append(float(loss.detach()))
        print(f"[train:{name}] epoch {ep+1}/{epochs} loss={np.mean(losses[-max(1,n//MINIBATCH):]):.4f}", flush=True)
    return {"mean_loss": float(np.mean(losses)) if losses else None, "n_updates": len(losses)}


def run_train(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()
    digest0 = actor_param_digest(actor)

    train_eps = load_episodes("train")
    index = build_step_index(train_eps)
    fc = stage_mask(train_eps, index, FC_STAGE_NAMES)

    # Branch A: uniform weight 1
    # Branch B: weights 2 and 4 on FC stages
    variants = [("A_w1", 1.0)] + [(f"B_w{int(w)}", float(w)) for w in WEIGHTS_B]
    results = {}
    for name, w_fc in variants:
        critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
        weights = np.ones(len(index), np.float64)
        weights[fc] = w_fc
        print(f"=== Train {name} fc_weight={w_fc} n_steps={len(index)} fc_frac={fc.mean():.3f} ===", flush=True)
        fit = _train_one(
            critic, train_eps, index, weights, epochs=CRITIC_EPOCHS, lr=CRITIC_LR, device=device_t, name=name
        )
        path = CKPT_DIR / f"critic_{name}.pt"
        torch.save({"critic": critic.state_dict(), "name": name, "fc_weight": w_fc, "fit": fit}, path)
        results[name] = {"path": str(path), "fc_weight": w_fc, "fit": fit, "n_train_steps": len(index), "fc_frac": float(fc.mean())}

    digest1 = actor_param_digest(actor)
    assert digest0 == digest1, "Actor weights mutated during Critic training"
    results["actor_digest_unchanged"] = True
    results["actor_digest"] = digest0
    (OUT / "CRITIC_TRAINING_COMPARISON.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "CRITIC_TRAINING_COMPARISON.md").write_text(
        "# Critic Training Comparison (P75 Phase 3)\n\nActor frozen (digest verified).\n\n```json\n"
        + json.dumps(results, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == "__main__":
    run_train()
