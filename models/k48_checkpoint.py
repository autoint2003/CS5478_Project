"""Build a K48 policy and load K47 or K48 checkpoints without changing weights."""

from __future__ import annotations

import numpy as np
import torch

from models.k48_policy import build_policy
from training.unified_k47 import CAP

MOTION_K = 47
HANDOFF = 47
LOGIT_K = 48


def build_k48(book, device):
    if tuple(book.shape) != (MOTION_K, 7):
        raise RuntimeError(f"motion codebook is {book.shape}, expected {(MOTION_K, 7)}")
    model = build_policy(book, device, CAP, handoff=True)
    if model.logits.out_features != LOGIT_K or model.handoff_index != HANDOFF:
        raise RuntimeError("HANDOFF head was not added past the 47 motion tokens")
    return model


def expand_k47_weights(model, state):
    """Copy the 47 motion-token rows. Initialize only the new HANDOFF row at zero."""
    own = model.state_dict()
    expanded = {}
    for key, value in state.items():
        if key not in own:
            raise RuntimeError(f"unexpected checkpoint tensor {key}")
        if tuple(own[key].shape) == tuple(value.shape):
            expanded[key] = value
            continue
        if key in ("logits.weight", "logits.bias") and own[key].shape[0] == value.shape[0] + 1 and value.shape[0] == MOTION_K:
            fresh = torch.zeros_like(own[key])
            fresh[:MOTION_K] = value
            expanded[key] = fresh
            continue
        raise RuntimeError(f"incompatible token head: {key} checkpoint {tuple(value.shape)} model {tuple(own[key].shape)}")
    missing = [key for key in own if key not in expanded and key != "proto"]
    # proto is a buffer already filled from the codebook, not a trained tensor in every checkpoint.
    missing = [key for key in missing if not key.startswith("proto")]
    if missing:
        raise RuntimeError("missing " + ", ".join(missing))
    proto = own["proto"]
    payload = {key: value for key, value in own.items()}
    for key, value in expanded.items():
        payload[key] = value if torch.is_tensor(value) else value
    payload["proto"] = proto
    model.load_state_dict(payload)
    return model


def load_expanded(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    book = np.asarray(saved["proto"], np.float32)
    model = build_k48(book, device)
    expand_k47_weights(model, saved["model"])
    copied = model.logits.weight.detach()[:MOTION_K]
    original = saved["model"]["logits.weight"]
    if not torch.equal(copied.cpu(), original):
        raise RuntimeError("motion-token rows were not copied exactly")
    if float(model.logits.weight.detach()[HANDOFF].abs().sum()) != 0.0 or float(model.logits.bias.detach()[HANDOFF]) != 0.0:
        raise RuntimeError("HANDOFF row was not zero-initialized")
    return model, book


def load_trained_k48(path, device):
    """Load a finished 48-token checkpoint. The file is not rewritten or expanded."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["model"]
    logits = state.get("logits.weight")
    if logits is None or tuple(logits.shape)[:1] != (LOGIT_K,):
        raise RuntimeError("checkpoint is not a trained 48-token K=48 model")
    book = np.asarray(saved["proto"], np.float32)
    if tuple(book.shape) != (MOTION_K, 7):
        raise RuntimeError(f"checkpoint motion codebook is {book.shape}")
    model = build_k48(book, device)
    model.load_state_dict(state)
    model.eval()
    return model, book
