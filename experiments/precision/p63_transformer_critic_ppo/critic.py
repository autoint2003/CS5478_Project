"""Separate full-history Transformer Critic mirroring the M111 Actor backbone."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


class TransformerCritic(nn.Module):
    """V(h_t) from causal full-history Transformer; no privileged info.

    Backbone (in_proj, pos, enc) matches ``models.k48_policy.Policy``.
    Value head replaces the action heads.
    """

    def __init__(self, *, d: int = 64, nhead: int = 4, ff: int = 128, n_layers: int = 2, window: int = 160):
        super().__init__()
        self.window = int(window)
        self.in_proj = nn.Linear(39 + 7, d)
        self.pos = nn.Embedding(window, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=ff, dropout=0.0,
            batch_first=True, activation="relu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.value = nn.Linear(d, 1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    @classmethod
    def from_actor(cls, actor: nn.Module) -> "TransformerCritic":
        """Clone Actor backbone weights; fresh scalar value head."""
        d = int(actor.in_proj.out_features)
        crit = cls(d=d, window=int(actor.pos.num_embeddings))
        crit.in_proj.load_state_dict(actor.in_proj.state_dict())
        crit.pos.load_state_dict(actor.pos.state_dict())
        crit.enc.load_state_dict(actor.enc.state_dict())
        return crit

    def encode(self, obs: torch.Tensor, prev: torch.Tensor, pad_mask=None) -> torch.Tensor:
        """Return causal context for every timestep. obs/prev: (T,D) or (B,T,D)."""
        x = torch.cat([obs, prev], dim=-1)
        t = x.shape[-2]
        if t > self.window:
            raise ValueError(f"history length {t} exceeds window {self.window}")
        h = self.in_proj(x) + self.pos(torch.arange(t, device=x.device))
        # Use bool causal mask (same dtype family as padding mask) to avoid
        # PyTorch deprecation on mismatched mask types.
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        if x.dim() == 2:
            return self.enc(h.unsqueeze(0), mask=causal)[0]
        return self.enc(h, mask=causal, src_key_padding_mask=pad_mask)

    def forward(self, obs: torch.Tensor, prev: torch.Tensor, pad_mask=None) -> torch.Tensor:
        """Values at every timestep. Returns (T,) or (B,T)."""
        ctx = self.encode(obs, prev, pad_mask=pad_mask)
        return self.value(ctx).squeeze(-1)

    def value_at_last(self, obs: torch.Tensor, prev: torch.Tensor, lengths=None) -> torch.Tensor:
        """Scalar value(s) at the last valid timestep."""
        vals = self.forward(obs, prev)
        if vals.dim() == 1:
            return vals[-1]
        assert lengths is not None
        return torch.stack([vals[j, int(lengths[j]) - 1] for j in range(vals.shape[0])])

    @torch.no_grad()
    def value_seq(self, obs_seq: list, prev_seq: list, device) -> float:
        obs = torch.as_tensor(np.stack(obs_seq[-self.window:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(prev_seq[-self.window:]), device=device, dtype=torch.float32)
        return float(self.value_at_last(obs, prev).item())


def critic_parameters_disjoint(actor: nn.Module, critic: TransformerCritic) -> bool:
    """True iff no shared Parameter objects between actor and critic."""
    a_ids = {id(p) for p in actor.parameters()}
    c_ids = {id(p) for p in critic.parameters()}
    return a_ids.isdisjoint(c_ids)


def clone_actor(actor: nn.Module) -> nn.Module:
    return copy.deepcopy(actor)
