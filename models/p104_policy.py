"""P104 pilot Actor: frozen Transformer backbone; trainable token head; optional Δ(h).

Arm B (token_only): train logits only; original residual frozen.
Arm C (joint): train logits + unconditional ResidualDeltaMLP (P101 Arm-B style);
               original residual preactivation frozen; Δ zero-init for P64 parity.
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn

from models.saup_policy import ResidualDeltaMLP

Mode = Literal["token_only", "joint"]


class P104Policy(nn.Module):
    def __init__(self, base: nn.Module, mode: Mode, *, hidden_delta: int = 96):
        super().__init__()
        if mode not in ("token_only", "joint"):
            raise ValueError(mode)
        self.mode = mode
        self.base = base
        self.proto = base.proto
        self.n_motion = int(base.n_motion)
        self.handoff_index = base.handoff_index
        self.res_cap = float(base.res_cap)

        for name, p in self.base.named_parameters():
            if name.startswith("logits."):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        if mode == "joint":
            self.delta = ResidualDeltaMLP(64, hidden_delta)
        else:
            self.delta = None

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

    def forward(self, obs, prev, pad_mask=None):
        ctx, logits = self._encode(obs, prev, pad_mask=pad_mask)
        preact = self.base.res(ctx)
        if self.delta is not None:
            res = self.res_cap * torch.tanh(preact + self.delta(ctx))
        else:
            res = self.res_cap * torch.tanh(preact)
        tok = self._z_hat(logits)
        if self.handoff_index is None:
            act = torch.clamp(self.proto[tok] + res, -1.0, 1.0)
        else:
            motion = torch.clamp(tok, 0, self.n_motion - 1)
            act = torch.clamp(self.proto[motion] + res, -1.0, 1.0)
            act = torch.where((tok == self.handoff_index).unsqueeze(-1), torch.zeros_like(act), act)
        return logits, res, act


def build_p104_from_p64(p64: nn.Module, mode: Mode, device) -> P104Policy:
    base = copy.deepcopy(p64).to(device)
    model = P104Policy(base, mode).to(device)
    model.eval()
    return model
