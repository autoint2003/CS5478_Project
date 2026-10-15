"""K=36 continual policy and the future-state auxiliary head.

Token 35 has its own logit and residual. Tokens 0-34 share the copied head.
The auxiliary head is not used at execution.
"""

from __future__ import annotations

import numpy as np
import torch

HORIZONS = (5, 10, 20, 40)
G_NAMES = (
    "p_rel_hx", "p_rel_hy", "p_rel_hz",
    "rot_rel_x", "rot_rel_y", "rot_rel_z",
    "v_rel_hx", "v_rel_hy", "v_rel_hz",
    "w_rel_hx", "w_rel_hy", "w_rel_hz",
    "aperture",
    "contact_present_L", "contact_present_R",
)


class ContinualPolicy(torch.nn.Module):
    """K=36 policy. Token 35 has its own logit and residual. Tokens 0-34 share the copied head."""

    def __init__(self, proto):
        super().__init__()
        self.register_buffer("proto", torch.tensor(np.asarray(proto, np.float32)))
        d = 64
        self.in_proj = torch.nn.Linear(39 + 7, d)
        self.pos = torch.nn.Embedding(160, d)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=128, dropout=0.0,
            batch_first=True, activation="relu",
        )
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.logits_old = torch.nn.Linear(d, 35)
        self.logit35_w = torch.nn.Parameter(torch.zeros(d))
        self.logit35_b = torch.nn.Parameter(torch.tensor(-100.0))
        self.res = torch.nn.Linear(d, 7)
        self.res35 = torch.nn.Linear(d, 7)
        torch.nn.init.zeros_(self.res.weight)
        torch.nn.init.zeros_(self.res.bias)
        torch.nn.init.zeros_(self.res35.weight)
        torch.nn.init.zeros_(self.res35.bias)
        self.res_cap = 0.25

    def encode(self, obs, prev, pad_mask=None):
        x = torch.cat([obs, prev], dim=-1)
        t = x.shape[-2]
        h = self.in_proj(x) + self.pos(torch.arange(t, device=x.device))
        causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
        if x.dim() == 2:
            ctx = self.enc(h.unsqueeze(0), mask=causal)[0]
        else:
            ctx = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        return ctx

    def heads(self, ctx):
        old = self.logits_old(ctx)
        extra = (ctx * self.logit35_w).sum(-1) + self.logit35_b
        logits = torch.cat([old, extra.unsqueeze(-1)], dim=-1)
        res_old = self.res_cap * torch.tanh(self.res(ctx))
        res35 = self.res_cap * torch.tanh(self.res35(ctx))
        return logits, res_old, res35

    def forward(self, obs, prev, pad_mask=None):
        ctx = self.encode(obs, prev, pad_mask)
        logits, res_old, res35 = self.heads(ctx)
        tok = torch.argmax(logits, dim=-1)
        res = torch.where(tok.unsqueeze(-1) == 35, res35, res_old)
        act = torch.clamp(self.proto[tok] + res, -1.0, 1.0)
        self.last_res_old = res_old
        self.last_res35 = res35
        return logits, res, act


class FuturePolicy(ContinualPolicy):
    def __init__(self, proto):
        super().__init__(proto)
        self.fut = torch.nn.Linear(64, len(HORIZONS) * len(G_NAMES))
        torch.nn.init.zeros_(self.fut.weight)
        torch.nn.init.zeros_(self.fut.bias)

    def future_of(self, obs, prev, pad_mask=None):
        raw = self.fut(self.encode(obs, prev, pad_mask))
        return raw.view(*obs.shape[:-1], len(HORIZONS), len(G_NAMES))
