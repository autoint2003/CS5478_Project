"""K=48 head: 47 unchanged motion tokens plus one HANDOFF token.

Token 47 is not a motion prototype. Selecting it ends recovery and does not
decode a residual. Loading a K=47 checkpoint into this head is an explicit
expansion: the 47 motion rows are copied and only the new row is initialized.
"""

from __future__ import annotations

import numpy as np
import torch

from models.k48_policy import build_policy
from training.grpo_math import gaussian_logprob

MOTION_K = 47
HANDOFF_TOKEN = 47
N_TOKENS = 48
CAP = 0.25


def pause_command(last_action):
    """Zero twist with the grip channel left unchanged.

    A full zero vector is not a pause. Grip action 0 is the midpoint torque,
    which unloads a closed gripper. Token 0 of the K=47 book is exactly
    ``[0, 0, 0, 0, 0, 0, 1]``.
    """
    command = np.zeros(7, np.float32)
    command[6] = float(np.asarray(last_action, np.float32).reshape(7)[6])
    return command


def expand_k47_state(state):
    """Copy a K=47 state dict and append a zero HANDOFF logit row."""
    weight = state["logits.weight"]
    bias = state["logits.bias"]
    if tuple(weight.shape) != (MOTION_K, 64) or tuple(bias.shape) != (MOTION_K,):
        raise RuntimeError(f"expected a 47-token head, got {tuple(weight.shape)} {tuple(bias.shape)}")
    expanded = {key: value.detach().cpu().clone() for key, value in state.items()}
    new_weight = torch.zeros(N_TOKENS, 64)
    new_bias = torch.zeros(N_TOKENS)
    new_weight[:MOTION_K] = weight.detach().cpu()
    new_bias[:MOTION_K] = bias.detach().cpu()
    expanded["logits.weight"] = new_weight
    expanded["logits.bias"] = new_bias
    return expanded


def build_k48(book, device):
    book = np.asarray(book, np.float32)
    if book.shape != (MOTION_K, 7):
        raise RuntimeError(f"motion codebook must stay {(MOTION_K, 7)}, got {book.shape}")
    return build_policy(book, device, CAP, n_tokens=N_TOKENS)


def motion_action(book, token, pre):
    """Decode a motion token. Refuses HANDOFF."""
    token = int(token)
    if token == HANDOFF_TOKEN:
        raise RuntimeError("HANDOFF has no 7-D decode")
    if token < 0 or token >= MOTION_K:
        raise RuntimeError(f"motion token {token} outside 0..{MOTION_K - 1}")
    residual = np.clip(np.asarray(pre, np.float32), -CAP, CAP)
    action = np.clip(np.asarray(book[token], np.float32) + residual, -1.0, 1.0)
    return residual, action


def step_logprob(model, rec):
    """Categorical token log-probability. Residual density only for motion tokens."""
    device = next(model.parameters()).device
    obs = rec["obs"] if torch.is_tensor(rec["obs"]) else torch.tensor(rec["obs"], device=device)
    prev = rec["prev"] if torch.is_tensor(rec["prev"]) else torch.tensor(rec["prev"], device=device)
    logits, mean, _act = model(obs.to(device), prev.to(device))
    if logits.shape[-1] != N_TOKENS:
        raise RuntimeError(f"expected {N_TOKENS} logits, got {logits.shape[-1]}")
    token = int(rec["token"])
    token_lp = torch.log_softmax(logits[-1].float(), dim=-1)[token]
    if token == HANDOFF_TOKEN:
        return token_lp
    resid_lp = gaussian_logprob(torch.tensor(rec["pre"], device=device), mean[-1].float())
    return token_lp + resid_lp


def supervised_loss(model, obs, prev, tokens, residuals, weight, mask):
    """Token CE on every step. Residual MSE only where the label is a motion token."""
    logits, pred, _act = model(obs, prev)
    if logits.shape[-1] != N_TOKENS:
        raise RuntimeError(f"forward width {logits.shape[-1]}")
    per_token = []
    per_resid = []
    handoff_hits = []
    for index in range(logits.shape[0]):
        active = mask[index]
        if not bool(active.any()):
            raise RuntimeError("a batch row has no unmasked steps")
        per_token.append(torch.nn.functional.cross_entropy(logits[index][active], tokens[index][active]))
        motion = active & (tokens[index] != HANDOFF_TOKEN)
        if bool(motion.any()):
            per_resid.append(((pred[index][motion] - residuals[index][motion]) ** 2).mean())
        else:
            per_resid.append(pred[index].sum() * 0.0)
        handoff_hits.append(bool((active & (tokens[index] == HANDOFF_TOKEN)).any()))
    token_vec = torch.stack(per_token)
    resid_vec = torch.stack(per_resid)
    w = weight.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    loss = (w * (token_vec + resid_vec)).mean()
    return loss, token_vec.mean().detach(), resid_vec.mean().detach(), handoff_hits
