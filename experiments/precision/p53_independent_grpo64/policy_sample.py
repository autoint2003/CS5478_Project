"""Stochastic M111 sampling shared by Independent GRPO and probabilistic eval.

HANDOFF logit is masked — R3F decides transfer; no policy action during nominal.
Temperature fixed at 1.0 for all checkpoints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from experiments.precision.p26_c_curriculum_rl.ppo_m111 import WINDOW, ActorCriticM111
from models.m111_checkpoint import HANDOFF, load_m111


def mask_handoff_logits(logits: torch.Tensor) -> torch.Tensor:
    out = logits.clone()
    out[..., HANDOFF] = -1e9
    return out


def load_policy_actor(ckpt, device):
    """Load M111 actor from P47T-style or PPO ActorCritic checkpoint."""
    actor, book, saved = load_m111(ckpt, device)
    rho = saved.get("residual_rho", saved.get("res_cap", saved.get("rho")))
    return actor, book, rho, saved


def as_actor(module: torch.nn.Module) -> torch.nn.Module:
    if isinstance(module, ActorCriticM111):
        return module.actor
    return module


@torch.no_grad()
def sample_step(
    actor: torch.nn.Module,
    obs_seq: list,
    prev_seq: list,
    device,
    *,
    temperature: float = 1.0,
    deterministic: bool = False,
) -> tuple[int, np.ndarray, float, dict[str, Any]]:
    obs = torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device, dtype=torch.float32)
    logits, res, _act = actor(obs, prev)
    logit = mask_handoff_logits(logits[-1].float())
    if temperature != 1.0:
        logit = logit / float(temperature)
    dist = Categorical(logits=logit)
    tok = torch.argmax(logit) if deterministic else dist.sample()
    logp = dist.log_prob(tok)
    resid = res[-1].detach().cpu().numpy().astype(np.float32)
    ent = float(dist.entropy().item())
    return int(tok.item()), resid, float(logp.item()), {"entropy": ent}


def token_logprob(
    actor: torch.nn.Module,
    obs: np.ndarray | torch.Tensor,
    prev: np.ndarray | torch.Tensor,
    token: int,
    device,
) -> torch.Tensor:
    """Per-step categorical log-prob under current policy (HANDOFF masked)."""
    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
    else:
        obs = obs.to(device=device, dtype=torch.float32)
    if not torch.is_tensor(prev):
        prev = torch.as_tensor(prev, device=device, dtype=torch.float32)
    else:
        prev = prev.to(device=device, dtype=torch.float32)
    logits, _res, _act = actor(obs, prev)
    logit = mask_handoff_logits(logits[-1].float())
    return torch.log_softmax(logit, dim=-1)[int(token)]


def categorical_kl_masked(actor, ref_actor, obs, prev, device) -> torch.Tensor:
    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
    if not torch.is_tensor(prev):
        prev = torch.as_tensor(prev, device=device, dtype=torch.float32)
    logits, _, _ = actor(obs, prev)
    with torch.no_grad():
        old, _, _ = ref_actor(obs, prev)
    log_q = torch.log_softmax(mask_handoff_logits(logits[-1].float()), dim=-1)
    log_p = torch.log_softmax(mask_handoff_logits(old[-1].float()), dim=-1)
    q = log_q.exp()
    return (q * (log_q - log_p)).sum()
