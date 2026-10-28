"""Full-history causal M111 Transformer (P29).

Uses deterministic sinusoidal positional encoding so sequence length is not
capped by a learned Embedding(160). Compatible warm-start: load non-pos
weights from legacy M111 checkpoints; skip incompatible `pos.*` keys.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.m111_checkpoint import DEFAULT_RHO, HANDOFF, LOGIT_K, MOTION_K  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget  # noqa: E402


def sinusoidal_position_encoding(length: int, d_model: int, device, dtype) -> torch.Tensor:
    """(length, d_model) deterministic PE; supports arbitrary length."""
    pe = torch.zeros(length, d_model, device=device, dtype=dtype)
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe


class FullHistoryPolicy(nn.Module):
    """Causal Transformer over the complete available episode history.

    Input per step: concat(obs_39, prev_action_7).
    Output: 112-way logits + residual (7). No skill head.
    """

    def __init__(self, proto: np.ndarray, res_cap: float, *, d_model: int = 64, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        motion = np.asarray(proto, np.float32)
        if motion.shape != (MOTION_K, 7):
            raise RuntimeError(f"FullHistoryPolicy expects proto {(MOTION_K, 7)}, got {motion.shape}")
        self.register_buffer("proto", torch.tensor(motion))
        self.n_motion = MOTION_K
        self.handoff_index = HANDOFF
        self.d_model = int(d_model)
        self.pos_encoding = "sinusoidal"
        self.max_supported_len = None  # unlimited (sinusoidal)
        self.in_proj = nn.Linear(39 + 7, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.logits = nn.Linear(d_model, LOGIT_K)
        self.res = nn.Linear(d_model, 7)
        nn.init.zeros_(self.res.weight)
        nn.init.zeros_(self.res.bias)
        self.res_cap = float(res_cap)

    def forward(self, obs, prev, pad_mask=None, position_ids=None):
        """
        obs, prev: (T, D) or (B, T, D)
        pad_mask: (B, T) True where PAD (ignored by encoder) — PyTorch key_padding_mask convention
        position_ids: optional (T,) or (B, T) chronological indices; default arange(T)
        """
        x = torch.cat([obs, prev], dim=-1)
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)
            if pad_mask is not None and pad_mask.dim() == 1:
                pad_mask = pad_mask.unsqueeze(0)
        b, t, _ = x.shape
        h = self.in_proj(x)
        if position_ids is None:
            pe = sinusoidal_position_encoding(t, self.d_model, x.device, h.dtype)
            h = h + pe.unsqueeze(0)
        else:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0).expand(b, -1)
            # gather sinusoidal at arbitrary indices (for future packing); here usually 0..T-1
            max_pos = int(position_ids.max().item()) + 1
            table = sinusoidal_position_encoding(max_pos, self.d_model, x.device, h.dtype)
            pe = table[position_ids.clamp(min=0)]
            h = h + pe
        # Use bool masks for both attn mask and key padding (PyTorch preference).
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        if pad_mask is not None and pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.bool()
        ctx = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        logits = self.logits(ctx)
        res = self.res_cap * torch.tanh(self.res(ctx))
        tok = torch.argmax(logits, dim=-1)
        motion = torch.clamp(tok, 0, self.n_motion - 1)
        act = torch.clamp(self.proto[motion] + res, -1.0, 1.0)
        act = torch.where((tok == self.handoff_index).unsqueeze(-1), torch.zeros_like(act), act)
        if not batched:
            return logits[0], res[0], act[0]
        return logits, res, act


def build_full_history_m111(book: np.ndarray, device, *, rho=DEFAULT_RHO) -> FullHistoryPolicy:
    r = ResidualBudget.from_spec(rho).as_array()
    res_cap = float(np.max(r))
    model = FullHistoryPolicy(book, res_cap).to(device)
    return model


def load_legacy_weights_into_full_history(
    model: FullHistoryPolicy,
    legacy_state: dict[str, Any],
    *,
    strict_non_pos: bool = False,
) -> dict[str, Any]:
    """Warm-start from P27/P28 M111 checkpoint; skip learned `pos` embeddings."""
    src = legacy_state.get("model") if "model" in legacy_state else legacy_state
    dst = model.state_dict()
    transferred, skipped = [], []
    new_state = {}
    for k, v in src.items():
        if k.startswith("pos."):
            skipped.append(k)
            continue
        if k in dst and tuple(dst[k].shape) == tuple(v.shape):
            new_state[k] = v
            transferred.append(k)
        else:
            skipped.append(k)
    missing = model.load_state_dict(new_state, strict=False)
    return {
        "transferred": transferred,
        "skipped": skipped,
        "missing_keys": list(missing.missing_keys),
        "unexpected_keys": list(missing.unexpected_keys),
        "pos_encoding": "sinusoidal (legacy Embedding(160) not loaded)",
    }
