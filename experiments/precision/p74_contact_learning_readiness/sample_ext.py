"""Extended Actor/Critic sampling for ownership + Critic audits."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits
from experiments.precision.p63_transformer_critic_ppo.paths import WINDOW
from models.m111_checkpoint import MOTION_K


@torch.no_grad()
def act_diagnose(actor, critic, obs_seq, prev_seq, device, *, temperature: float = 1.0):
    """Sample token; return diagnostics including full Categorical stats."""
    obs = torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device, dtype=torch.float32)
    logits, res, _ = actor(obs, prev)
    logit = mask_handoff_logits(logits[-1].float())
    if temperature != 1.0:
        logit = logit / float(temperature)
    dist = Categorical(logits=logit)
    tok = dist.sample()
    tok_i = int(tok.item())
    assert 0 <= tok_i < MOTION_K
    probs = dist.probs.detach().cpu().numpy()
    top5 = np.argsort(-probs)[:5]
    return {
        "tok": tok_i,
        "resid": res[-1].detach().cpu().numpy().astype(np.float32),
        "logp": float(dist.log_prob(tok).item()),
        "prob": float(probs[tok_i]),
        "entropy": float(dist.entropy().item()),
        "value": float(critic.value_at_last(obs, prev).item()),
        "top5_tok": [int(i) for i in top5],
        "top5_prob": [float(probs[i]) for i in top5],
        "probs_snapshot": probs,  # full K — large; drop before JSON if needed
    }


@torch.no_grad()
def logp_of_token(actor, obs_seq, prev_seq, device, tok: int) -> float:
    """Behavior log-prob of an executed token under current Actor (for audit)."""
    obs = torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device, dtype=torch.float32)
    logits, _, _ = actor(obs, prev)
    logit = mask_handoff_logits(logits[-1].float())
    dist = Categorical(logits=logit)
    t = torch.as_tensor(int(tok), device=device, dtype=torch.long)
    return float(dist.log_prob(t).item())
