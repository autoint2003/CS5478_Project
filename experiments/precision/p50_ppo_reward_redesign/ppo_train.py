"""P50 B-focused PPO: Categorical M111 + KL(P47T) + SFT A/B/C/D anchor."""

from __future__ import annotations

import copy
import pickle
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from experiments.precision.p26_c_curriculum_rl.ppo_m111 import (
    WINDOW,
    ActorCriticM111,
    PPOConfig,
    compute_gae,
    load_actor_critic,
    save_actor_critic,
)
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import BRecoveryEnv, BStart
from experiments.precision.p50_ppo_reward_redesign.paths import OUT, P47, SEED
from experiments.precision.p50_ppo_reward_redesign.reward import RewardWeights
from models.m111_checkpoint import HANDOFF, MOTION_K, load_m111
from training.m111_dataset import collate_m111


@dataclass
class P50PPOConfig(PPOConfig):
    kl_coef: float = 0.05
    sft_coef: float = 0.10
    sft_batch: int = 8
    max_ep_steps: int = 80
    total_steps: int = 4096
    rollout_steps: int = 256
    lr: float = 1e-5
    ent_coef: float = 0.02
    epochs: int = 3
    minibatch: int = 32


def mask_handoff_logits(logits: torch.Tensor) -> torch.Tensor:
    out = logits.clone()
    out[..., HANDOFF] = -1e9
    return out


@torch.no_grad()
def act_masked(ac: ActorCriticM111, obs_seq, prev_seq, device, *, deterministic: bool = False):
    obs = torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device, dtype=torch.float32)
    logits, res, value = ac.forward(obs, prev)
    logits = mask_handoff_logits(logits[-1])
    dist = Categorical(logits=logits)
    tok = torch.argmax(logits) if deterministic else dist.sample()
    logp = dist.log_prob(tok)
    resid = res[-1].detach().cpu().numpy().astype(np.float32)
    return int(tok.item()), resid, float(logp.item()), float(value.item())


def load_sft_pools(pool: str = "P47T") -> dict[str, list]:
    skills = ("A1", "B1", "C", "D")
    pools = {}
    for sk in skills:
        path = P47 / "dataset" / f"chunks_{sk}_{pool}.pkl"
        if path.is_file():
            pools[sk] = pickle.loads(path.read_bytes())
        else:
            pools[sk] = []
    return pools


def sft_loss(ac: ActorCriticM111, pools: dict[str, list], n_per: int, device, rng: random.Random) -> torch.Tensor:
    samples = []
    for sk, chs in pools.items():
        if not chs:
            continue
        for _ in range(n_per):
            samples.append(chs[rng.randrange(len(chs))])
    if not samples:
        return torch.zeros((), device=device)
    obs, prev, tokens, _res, mask, _w = collate_m111(samples, device)
    mask = mask & (tokens != HANDOFF)
    logits, _, _ = ac.forward(obs, prev)
    losses = []
    for i in range(logits.shape[0]):
        active = mask[i]
        if not bool(active.any()):
            continue
        losses.append(F.cross_entropy(logits[i][active], tokens[i][active]))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def kl_to_ref(ac: ActorCriticM111, ref: ActorCriticM111, obs_t, prev_t, lengths) -> torch.Tensor:
    """Mean KL(pi || pi_ref) on last-step categorical (HANDOFF masked)."""
    logits, _, _ = ac.forward(obs_t, prev_t)
    with torch.no_grad():
        ref_logits, _, _ = ref.forward(obs_t, prev_t)
    last = torch.stack([mask_handoff_logits(logits[j, lengths[j] - 1]) for j in range(len(lengths))])
    last_ref = torch.stack([mask_handoff_logits(ref_logits[j, lengths[j] - 1]) for j in range(len(lengths))])
    p = F.softmax(last, dim=-1)
    log_p = F.log_softmax(last, dim=-1)
    log_q = F.log_softmax(last_ref, dim=-1)
    return (p * (log_p - log_q)).sum(-1).mean()


def ppo_update_p50(
    ac: ActorCriticM111,
    ref: ActorCriticM111,
    batch: dict,
    cfg: P50PPOConfig,
    opt,
    pools: dict[str, list],
    rng: random.Random,
) -> dict:
    device = next(ac.parameters()).device
    obs = batch["obs"]
    prev = batch["prev"]
    tokens = torch.as_tensor(batch["tokens"], device=device, dtype=torch.long)
    old_logp = torch.as_tensor(batch["logp"], device=device, dtype=torch.float32)
    advantages = torch.as_tensor(batch["adv"], device=device, dtype=torch.float32)
    returns = torch.as_tensor(batch["ret"], device=device, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(obs)
    idxs = np.arange(n)
    metrics = []
    for _ in range(cfg.epochs):
        np.random.shuffle(idxs)
        for start in range(0, n, cfg.minibatch):
            mb = idxs[start : start + cfg.minibatch]
            lengths = [obs[i].shape[0] for i in mb]
            tmax = max(lengths)
            o = torch.zeros(len(mb), tmax, 39, device=device)
            p = torch.zeros(len(mb), tmax, 7, device=device)
            for j, i in enumerate(mb):
                ti = lengths[j]
                o[j, :ti] = torch.as_tensor(obs[i], device=device)
                p[j, :ti] = torch.as_tensor(prev[i], device=device)
            logits, _res, _values_pad = ac.forward(o, p)
            last_logits = torch.stack(
                [mask_handoff_logits(logits[j, lengths[j] - 1]) for j in range(len(mb))]
            )
            # value at true last timestep (not padding tail)
            x = torch.cat([o, p], dim=-1)
            t_dim = x.shape[1]
            h = ac.actor.in_proj(x) + ac.actor.pos(torch.arange(t_dim, device=device))
            causal = torch.triu(torch.ones(t_dim, t_dim, device=device) * float("-inf"), diagonal=1)
            ctx = ac.actor.enc(h, mask=causal)
            last_values = torch.stack(
                [ac.value(ctx[j, lengths[j] - 1]).squeeze(-1) for j in range(len(mb))]
            )
            dist = Categorical(logits=last_logits)
            tok_mb = tokens[mb]
            logp = dist.log_prob(tok_mb)
            entropy = dist.entropy().mean()
            ratio = torch.exp(logp - old_logp[mb])
            adv = advantages[mb]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv
            pg = -torch.min(surr1, surr2).mean()
            vf = F.mse_loss(last_values, returns[mb])
            kl = kl_to_ref(ac, ref, o, p, lengths)
            sft = sft_loss(ac, pools, max(1, cfg.sft_batch // 4), device, rng)
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * entropy + cfg.kl_coef * kl + cfg.sft_coef * sft
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
            opt.step()
            approx_kl = float((old_logp[mb] - logp).mean().detach())
            clipfrac = float(((ratio - 1.0).abs() > cfg.clip).float().mean().detach())
            metrics.append(
                {
                    "loss": float(loss.detach()),
                    "pg": float(pg.detach()),
                    "vf": float(vf.detach()),
                    "ent": float(entropy.detach()),
                    "kl_ref": float(kl.detach()),
                    "sft": float(sft.detach()),
                    "approx_kl": approx_kl,
                    "clipfrac": clipfrac,
                }
            )
    return {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}


def collect_rollout(
    ac: ActorCriticM111,
    env: BRecoveryEnv,
    starts: list[BStart],
    cfg: P50PPOConfig,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    device = next(ac.parameters()).device
    obs_buf, prev_buf, tok_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], [], []
    ep_returns, ep_infos = [], []
    steps = 0
    bootstrap = 0.0
    while steps < cfg.rollout_steps:
        st = starts[int(rng.integers(0, len(starts)))]
        env.reset(st)
        ep_r = 0.0
        done = False
        info: dict = {}
        while not done and steps < cfg.rollout_steps:
            tok, resid, logp, val = act_masked(ac, env.obs_seq, env.prev_seq, device)
            assert 0 <= tok < MOTION_K, f"sampled non-motion token {tok}"
            obs_buf.append(np.stack(env.obs_seq[-WINDOW:]).astype(np.float32))
            prev_buf.append(np.stack(env.prev_seq[-WINDOW:]).astype(np.float32))
            tok_buf.append(tok)
            logp_buf.append(logp)
            val_buf.append(val)
            r, done, info = env.step(tok, resid)
            rew_buf.append(r)
            done_buf.append(float(done))
            ep_r += r
            steps += 1
        # bootstrap if rollout cut mid-episode
        if not done:
            _, _, _, boot_v = act_masked(ac, env.obs_seq, env.prev_seq, device)
            bootstrap = boot_v
        else:
            bootstrap = 0.0
        ep_returns.append(ep_r)
        ep_infos.append(info)
        # GAE uses values with one extra bootstrap at end of this episode segment —
        # we flatten across episodes; mark episode ends as done so GAE resets.
        # For mid-buffer cut: last done_buf entry is 0; append bootstrap into values list below.

    values_ext = list(val_buf) + [bootstrap if (done_buf and done_buf[-1] < 0.5) else 0.0]
    # For multi-episode buffer, compute_gae with done flags is correct when each episode
    # ends with done=1. Mid-buffer incomplete episode uses bootstrap above.
    adv, ret = compute_gae(
        np.asarray(rew_buf, np.float32),
        np.asarray(values_ext, np.float32),
        np.asarray(done_buf, np.float32),
        cfg.gamma,
        cfg.gae_lambda,
    )
    y_pred = np.asarray(val_buf, np.float32)
    y_true = ret
    var_y = float(np.var(y_true))
    explained = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 1e-8 else 0.0
    drops = sum(int(i.get("drop") or 0) for i in ep_infos)
    s12 = sum(int(i.get("SUCCESS_12S") or 0) for i in ep_infos)
    batch = {
        "obs": obs_buf,
        "prev": prev_buf,
        "tokens": np.asarray(tok_buf, np.int64),
        "logp": np.asarray(logp_buf, np.float32),
        "adv": adv,
        "ret": ret,
    }
    stats = {
        "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "n_episodes": len(ep_returns),
        "drop_rate": drops / max(1, len(ep_infos)),
        "success_12s_rate": s12 / max(1, len(ep_infos)),
        "explained_var": explained,
        "mean_entropy_proxy": float(np.mean([-lp for lp in logp_buf])) if logp_buf else 0.0,
    }
    return batch, stats


def train_b_ppo(
    *,
    init_ckpt: Path,
    starts_train: list[BStart],
    starts_val: list[BStart],
    cfg: P50PPOConfig | None = None,
    tag: str = "P50B",
) -> dict[str, Any]:
    cfg = cfg or P50PPOConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)

    _actor, book, saved = load_m111(init_ckpt, device)
    rho = saved.get("residual_rho", saved.get("res_cap"))
    ac = load_actor_critic(init_ckpt, book, device, rho)
    ref = ActorCriticM111(copy.deepcopy(ac.actor)).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    pools = load_sft_pools("P47T")
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    env = BRecoveryEnv(book, rho, max_steps=cfg.max_ep_steps, weights=RewardWeights())
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hist = []
    best = {"val_s12": -1.0, "path": None, "steps": 0}
    steps = 0
    while steps < cfg.total_steps:
        batch, stats = collect_rollout(ac, env, starts_train, cfg, rng)
        metrics = ppo_update_p50(ac, ref, batch, cfg, opt, pools, py_rng)
        steps += cfg.rollout_steps
        row = {"tag": tag, "steps": steps, **stats, **metrics}
        # cheap val every 2 rollouts
        if steps % (cfg.rollout_steps * 2) == 0 or steps >= cfg.total_steps:
            val = eval_starts(ac, book, rho, starts_val, device, max_steps=cfg.max_ep_steps)
            row["val_s12"] = val["success_12s_rate"]
            row["val_drop"] = val["drop_rate"]
            row["val_return"] = val["mean_return"]
            if val["success_12s_rate"] >= best["val_s12"]:
                path = ckpt_dir / f"{tag}_best.pt"
                save_actor_critic(path, ac, book, rho, extra={"tag": tag, "steps": steps, "p50": True})
                best = {"val_s12": val["success_12s_rate"], "val_drop": val["drop_rate"], "path": str(path), "steps": steps}
        hist.append(row)
        print(
            f"  {tag} steps={steps}/{cfg.total_steps} ret={stats['mean_return']:.2f} "
            f"drop={stats['drop_rate']:.2f} s12={stats['success_12s_rate']:.2f} "
            f"kl={metrics.get('kl_ref', float('nan')):.4f} "
            f"val_s12={row.get('val_s12', float('nan'))}",
            flush=True,
        )
        path_u = ckpt_dir / f"{tag}_s{steps:05d}.pt"
        save_actor_critic(path_u, ac, book, rho, extra={"tag": tag, "steps": steps})

    return {"history": hist, "best": best, "cfg": asdict(cfg), "book_ok": True, "init": str(init_ckpt)}


def eval_starts(ac, book, rho, starts: list[BStart], device, *, max_steps: int = 80) -> dict:
    env = BRecoveryEnv(book, rho, max_steps=max_steps)
    returns, drops, s12s = [], [], []
    for st in starts:
        env.reset(st)
        ep_r = 0.0
        done = False
        info = {}
        while not done:
            tok, resid, _, _ = act_masked(ac, env.obs_seq, env.prev_seq, device, deterministic=True)
            r, done, info = env.step(tok, resid)
            ep_r += r
        returns.append(ep_r)
        drops.append(int(info.get("drop") or 0))
        s12s.append(int(info.get("SUCCESS_12S") or 0))
    return {
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "drop_rate": float(np.mean(drops)) if drops else 0.0,
        "success_12s_rate": float(np.mean(s12s)) if s12s else 0.0,
        "n": len(starts),
    }
