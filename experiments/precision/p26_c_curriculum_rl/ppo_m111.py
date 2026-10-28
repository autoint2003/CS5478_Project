"""Categorical M111 PPO with a lightweight value head on the Transformer context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.m111_checkpoint import HANDOFF, LOGIT_K, MOTION_K, load_m111, save_m111
from models.transformer_policy import DT_POLICY

WINDOW = 160


class ActorCriticM111(nn.Module):
    """Wrap frozen-proto M111 policy; add value head on last context features."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor
        d = actor.in_proj.out_features
        self.value = nn.Linear(d, 1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def _context(self, obs, prev, pad_mask=None):
        x = torch.cat([obs, prev], dim=-1)
        t = x.shape[-2]
        h = self.actor.in_proj(x) + self.actor.pos(torch.arange(t, device=x.device))
        causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
        if x.dim() == 2:
            ctx = self.actor.enc(h.unsqueeze(0), mask=causal)[0]
        else:
            ctx = self.actor.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        return ctx

    def forward(self, obs, prev):
        ctx = self._context(obs, prev)
        logits = self.actor.logits(ctx)
        res = self.actor.res_cap * torch.tanh(self.actor.res(ctx))
        value = self.value(ctx[..., -1, :] if ctx.dim() == 3 else ctx[-1]).squeeze(-1)
        return logits, res, value

    def act(self, obs_seq: list, prev_seq: list, device, *, deterministic: bool = False):
        obs = torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device, dtype=torch.float32)
        logits, res, value = self.forward(obs, prev)
        dist = Categorical(logits=logits[-1])
        tok = torch.argmax(logits[-1]) if deterministic else dist.sample()
        logp = dist.log_prob(tok)
        tok_i = int(tok.item())
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        return tok_i, resid, float(logp.item()), float(value.item())


@dataclass
class PPOConfig:
    lr: float = 3e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4
    minibatch: int = 64
    max_grad_norm: float = 1.0
    total_steps: int = 20000
    rollout_steps: int = 512
    device: str = "cuda"


def decode_action(book, tok: int, resid: np.ndarray, rho) -> np.ndarray | None:
    from models.m111_checkpoint import decode_m111_action

    if tok == HANDOFF:
        return None
    return decode_m111_action(book, tok, resid, rho=rho)


def compute_gae(rewards, values, dones, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float64)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - float(dones[t])
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last = delta + gamma * lam * next_nonterminal * last
        adv[t] = last
    ret = adv + values[: len(rewards)]
    return adv.astype(np.float32), ret.astype(np.float32)


def ppo_update(ac: ActorCriticM111, batch: dict, cfg: PPOConfig, opt) -> dict:
    device = next(ac.parameters()).device
    obs = batch["obs"]  # list of (T,39)
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
            # pad sequences in minibatch
            lengths = [obs[i].shape[0] for i in mb]
            tmax = max(lengths)
            o = torch.zeros(len(mb), tmax, 39, device=device)
            p = torch.zeros(len(mb), tmax, 7, device=device)
            for j, i in enumerate(mb):
                ti = lengths[j]
                o[j, :ti] = torch.as_tensor(obs[i], device=device)
                p[j, :ti] = torch.as_tensor(prev[i], device=device)
            logits, _res, values = ac.forward(o, p)
            # last-step logits; values is (B,) from forward on batched input
            last_logits = torch.stack([logits[j, lengths[j] - 1] for j in range(len(mb))])
            last_values = values.reshape(-1)
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
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
            opt.step()
            metrics.append(
                {
                    "loss": float(loss.detach()),
                    "pg": float(pg.detach()),
                    "vf": float(vf.detach()),
                    "ent": float(entropy.detach()),
                }
            )
    return {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}


def load_actor_critic(ckpt: Path, book, device, rho) -> ActorCriticM111:
    actor, _, _ = load_m111(ckpt, device)
    ac = ActorCriticM111(actor).to(device)
    return ac


def save_actor_critic(path: Path, ac: ActorCriticM111, book, rho, *, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # save actor via m111 + value head
    save_m111(path, ac.actor, book, epoch=0, rho=rho, extra={"p26_ppo": True, **(extra or {})})
    torch.save({"value": ac.value.state_dict()}, path.with_suffix(".value.pt"))


def load_actor_critic_full(ckpt: Path, book, device, rho) -> ActorCriticM111:
    ac = load_actor_critic(ckpt, book, device, rho)
    vpath = Path(ckpt).with_suffix(".value.pt")
    if vpath.is_file():
        blob = torch.load(vpath, map_location="cpu", weights_only=False)
        ac.value.load_state_dict(blob["value"])
    return ac
