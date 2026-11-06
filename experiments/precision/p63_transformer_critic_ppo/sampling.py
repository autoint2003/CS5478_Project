"""Actor sampling with frozen old-logp semantics (HANDOFF masked)."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits
from experiments.precision.p63_transformer_critic_ppo.critic import TransformerCritic
from experiments.precision.p63_transformer_critic_ppo.paths import WINDOW
from models.m111_checkpoint import MOTION_K


@torch.no_grad()
def act_with_value(
    actor: torch.nn.Module,
    critic: TransformerCritic,
    obs_seq: list,
    prev_seq: list,
    device,
    *,
    temperature: float = 1.0,
    deterministic: bool = False,
) -> tuple[int, np.ndarray, float, float]:
    """Sample motion token; return (tok, resid, logp, value)."""
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
    value = float(critic.value_at_last(obs, prev).item())
    tok_i = int(tok.item())
    assert 0 <= tok_i < MOTION_K, f"non-motion token {tok_i}"
    return tok_i, resid, float(logp.item()), value


def pad_batch(obs_list, prev_list, device):
    """Pad variable-length history prefixes; return tensors + lengths + pad_mask.

    pad_mask True = PADDED (ignored by Transformer src_key_padding_mask).
    """
    lengths = [int(o.shape[0]) for o in obs_list]
    tmax = max(lengths)
    b = len(obs_list)
    o = torch.zeros(b, tmax, 39, device=device)
    p = torch.zeros(b, tmax, 7, device=device)
    pad_mask = torch.ones(b, tmax, device=device, dtype=torch.bool)
    for j, (obs, prev) in enumerate(zip(obs_list, prev_list)):
        ti = lengths[j]
        o[j, :ti] = torch.as_tensor(obs, device=device, dtype=torch.float32)
        p[j, :ti] = torch.as_tensor(prev, device=device, dtype=torch.float32)
        pad_mask[j, :ti] = False
    return o, p, lengths, pad_mask
