"""Clipped PPO with separate Actor / Critic optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits
from experiments.precision.p63_transformer_critic_ppo.critic import TransformerCritic
from experiments.precision.p63_transformer_critic_ppo.paths import (
    ACTOR_LR,
    CLIP,
    CRITIC_LR,
    ENT_COEF,
    MAX_GRAD_NORM,
    MINIBATCH,
    PPO_EPOCHS,
    TARGET_KL,
    VF_COEF,
)
from experiments.precision.p63_transformer_critic_ppo.sampling import pad_batch


@dataclass
class PPOHyper:
    clip: float = CLIP
    epochs: int = PPO_EPOCHS
    minibatch: int = MINIBATCH
    ent_coef: float = ENT_COEF
    vf_coef: float = VF_COEF
    max_grad_norm: float = MAX_GRAD_NORM
    target_kl: float = TARGET_KL
    actor_lr: float = ACTOR_LR
    critic_lr: float = CRITIC_LR


def make_optimizers(actor, critic, hyp: PPOHyper | None = None):
    hyp = hyp or PPOHyper()
    opt_a = torch.optim.Adam(actor.parameters(), lr=hyp.actor_lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=hyp.critic_lr)
    return opt_a, opt_c


def ppo_update(
    actor: nn.Module,
    critic: TransformerCritic,
    batch: dict,
    opt_actor,
    opt_critic,
    hyp: PPOHyper | None = None,
) -> dict:
    """One PPO update cycle.

    Each buffer entry is a *complete causal history prefix* ending at the
    decision timestep. Minibatches shuffle prefixes, never raw timesteps
    stripped of history.
    ``batch['logp']`` are frozen old action log-probs.
    """
    hyp = hyp or PPOHyper()
    device = next(actor.parameters()).device
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
    early_stop = False
    for _ep in range(hyp.epochs):
        if early_stop:
            break
        np.random.shuffle(idxs)
        for start in range(0, n, hyp.minibatch):
            mb = idxs[start : start + hyp.minibatch]
            o, p, lengths, pad_mask = pad_batch(
                [obs[i] for i in mb], [prev[i] for i in mb], device
            )
            # Actor
            logits, _res, _act = actor(o, p, pad_mask=pad_mask)
            last_logits = torch.stack(
                [mask_handoff_logits(logits[j, lengths[j] - 1].float()) for j in range(len(mb))]
            )
            dist = Categorical(logits=last_logits)
            tok_mb = tokens[mb]
            logp = dist.log_prob(tok_mb)
            entropy = dist.entropy().mean()
            ratio = torch.exp(logp - old_logp[mb])
            adv = advantages[mb]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - hyp.clip, 1.0 + hyp.clip) * adv
            pg = -torch.min(surr1, surr2).mean()

            # Critic (independent graph)
            values_all = critic(o, p, pad_mask=pad_mask)
            last_values = torch.stack(
                [values_all[j, lengths[j] - 1] for j in range(len(mb))]
            )
            vf = F.mse_loss(last_values, returns[mb])

            loss_a = pg - hyp.ent_coef * entropy
            loss_c = hyp.vf_coef * vf

            opt_actor.zero_grad(set_to_none=True)
            loss_a.backward()
            gn_a = float(nn.utils.clip_grad_norm_(actor.parameters(), hyp.max_grad_norm))
            opt_actor.step()

            opt_critic.zero_grad(set_to_none=True)
            loss_c.backward()
            gn_c = float(nn.utils.clip_grad_norm_(critic.parameters(), hyp.max_grad_norm))
            opt_critic.step()

            approx_kl = float((old_logp[mb] - logp.detach()).mean().item())
            clipfrac = float(((ratio.detach() - 1.0).abs() > hyp.clip).float().mean().item())
            metrics.append(
                {
                    "pg": float(pg.detach()),
                    "vf": float(vf.detach()),
                    "ent": float(entropy.detach()),
                    "approx_kl": approx_kl,
                    "clipfrac": clipfrac,
                    "grad_actor": gn_a,
                    "grad_critic": gn_c,
                    "ratio_mean": float(ratio.detach().mean().item()),
                }
            )
            if approx_kl > hyp.target_kl * 1.5:
                early_stop = True
                break

    out = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}
    out["n_minibatches"] = len(metrics)
    out["early_kl_stop"] = int(early_stop)
    return out


@torch.no_grad()
def mean_ratio_vs_old(actor, batch, device) -> float:
    """Sanity: before any update, ratio should be ~1."""
    obs, prev = batch["obs"], batch["prev"]
    tokens = torch.as_tensor(batch["tokens"], device=device, dtype=torch.long)
    old_logp = torch.as_tensor(batch["logp"], device=device, dtype=torch.float32)
    o, p, lengths, pad_mask = pad_batch(obs, prev, device)
    logits, _, _ = actor(o, p, pad_mask=pad_mask)
    last = torch.stack(
        [mask_handoff_logits(logits[j, lengths[j] - 1].float()) for j in range(len(obs))]
    )
    dist = Categorical(logits=last)
    logp = dist.log_prob(tokens)
    return float(torch.exp(logp - old_logp).mean().item())
