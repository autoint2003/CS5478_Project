"""K48 causal Transformer: 47 motion tokens, optional HANDOFF logit, residual head.

Parameter names match the historical nested Policy so existing checkpoints
load with the same state_dict keys: proto, in_proj, pos, enc, logits, res.
"""

from __future__ import annotations

import numpy as np
import torch


class Policy(torch.nn.Module):
    def __init__(self, proto, res_cap, handoff=False):
        super().__init__()
        motion = np.asarray(proto, np.float32)
        self.register_buffer("proto", torch.tensor(motion))
        self.n_motion = int(motion.shape[0])
        self.handoff_index = self.n_motion if handoff else None
        d = 64
        self.in_proj = torch.nn.Linear(39 + 7, d)
        self.pos = torch.nn.Embedding(160, d)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=128, dropout=0.0,
            batch_first=True, activation="relu",
        )
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.logits = torch.nn.Linear(d, self.n_motion + (1 if handoff else 0))
        self.res = torch.nn.Linear(d, 7)
        torch.nn.init.zeros_(self.res.weight)
        torch.nn.init.zeros_(self.res.bias)
        self.res_cap = float(res_cap)

    def forward(self, obs, prev, pad_mask=None):
        x = torch.cat([obs, prev], dim=-1)
        t = x.shape[-2]
        h = self.in_proj(x) + self.pos(torch.arange(t, device=x.device))
        causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
        if x.dim() == 2:
            ctx = self.enc(h.unsqueeze(0), mask=causal)[0]
        else:
            ctx = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        logits = self.logits(ctx)
        res = self.res_cap * torch.tanh(self.res(ctx))
        tok = torch.argmax(logits, dim=-1)
        if self.handoff_index is None:
            act = torch.clamp(self.proto[tok] + res, -1.0, 1.0)
        else:
            motion = torch.clamp(tok, 0, self.n_motion - 1)
            act = torch.clamp(self.proto[motion] + res, -1.0, 1.0)
            act = torch.where((tok == self.handoff_index).unsqueeze(-1), torch.zeros_like(act), act)
        return logits, res, act


def build_policy(proto, device, res_cap, handoff=False):
    return Policy(proto, res_cap, handoff=handoff).to(device)
