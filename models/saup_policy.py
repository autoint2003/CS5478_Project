"""SAUP residual-delta Actor: frozen P64 backbone/token/res-preact + trainable Δ.

Arm B: Δ(h) unconditional MLP.
Arm C: Δ(h, z_hat) token-conditioned MLP.

r = res_cap * tanh(W_P64 h + b_P64 + Δ)
Final Δ layer zero-initialized for exact P64 parity at init.
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn

from models.m111_checkpoint import HANDOFF


ArmKind = Literal["B", "C"]


class ResidualDeltaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 7)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class SAUPPolicy(nn.Module):
    """Wraps a loaded P64 Policy; only delta_* parameters train."""

    def __init__(
        self,
        base: nn.Module,
        arm: ArmKind,
        *,
        hidden_b: int = 96,
        hidden_c: int = 72,
        d_z: int = 8,
    ):
        super().__init__()
        if arm not in ("B", "C"):
            raise ValueError(f"arm must be B or C, got {arm}")
        self.arm = arm
        self.base = base
        # Expose attributes used by env stacks / losses
        self.proto = base.proto
        self.n_motion = int(base.n_motion)
        self.handoff_index = base.handoff_index
        self.res_cap = float(base.res_cap)

        # Freeze entire base (backbone, token head, original residual preact)
        for p in self.base.parameters():
            p.requires_grad_(False)

        if arm == "B":
            self.token_emb = None
            self.delta = ResidualDeltaMLP(64, hidden_b)
        else:
            self.token_emb = nn.Embedding(112, d_z)
            self.delta = ResidualDeltaMLP(64 + d_z + 7, hidden_c)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _encode(self, obs, prev, pad_mask=None):
        x = torch.cat([obs, prev], dim=-1)
        t = x.shape[-2]
        h = self.base.in_proj(x) + self.base.pos(torch.arange(t, device=x.device))
        causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
        if x.dim() == 2:
            ctx = self.base.enc(h.unsqueeze(0), mask=causal)[0]
        else:
            ctx = self.base.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        logits = self.base.logits(ctx)
        return ctx, logits

    def _z_hat(self, logits: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        if self.handoff_index is not None:
            masked[..., int(self.handoff_index)] = -1e9
        return torch.argmax(masked, dim=-1)

    def _delta(self, ctx: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        if self.arm == "B":
            return self.delta(ctx)
        motion = torch.clamp(z_hat, 0, self.n_motion - 1)
        emb = self.token_emb(z_hat)
        proto_z = self.proto[motion]
        return self.delta(torch.cat([ctx, emb, proto_z], dim=-1))

    def forward(self, obs, prev, pad_mask=None, z_cond=None):
        """Returns logits, residual (bounded), executed action.

        z_cond: optional token ids for Δ conditioning (Arm C). If None, uses
        HANDOFF-masked argmax of frozen logits (deploy path).
        """
        ctx, logits = self._encode(obs, prev, pad_mask=pad_mask)
        z_hat = self._z_hat(logits) if z_cond is None else z_cond
        preact = self.base.res(ctx)
        delta = self._delta(ctx, z_hat)
        res = self.res_cap * torch.tanh(preact + delta)
        tok = self._z_hat(logits)
        if self.handoff_index is None:
            act = torch.clamp(self.proto[tok] + res, -1.0, 1.0)
        else:
            motion = torch.clamp(tok, 0, self.n_motion - 1)
            act = torch.clamp(self.proto[motion] + res, -1.0, 1.0)
            act = torch.where((tok == self.handoff_index).unsqueeze(-1), torch.zeros_like(act), act)
        return logits, res, act


def clone_p64_base(actor: nn.Module) -> nn.Module:
    return copy.deepcopy(actor)


def build_saup_from_p64(p64: nn.Module, arm: ArmKind, device) -> SAUPPolicy:
    base = clone_p64_base(p64).to(device)
    model = SAUPPolicy(base, arm).to(device)
    model.eval()
    return model


def count_delta_params(arm: ArmKind, *, hidden_b: int = 96, hidden_c: int = 72, d_z: int = 8) -> int:
    if arm == "B":
        return 64 * hidden_b + hidden_b + hidden_b * 7 + 7
    return 112 * d_z + (64 + d_z + 7) * hidden_c + hidden_c + hidden_c * 7 + 7
