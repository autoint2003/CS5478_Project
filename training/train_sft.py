"""Split from training/unified_k48.py. The formulas are unchanged."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from envs.outcomes.horizon_outcome import LAMBDA_TIME, NOMINAL_S, physical_reward
from envs.step_legal import step_legal
from models.k48_checkpoint import (
    HANDOFF,
    LOGIT_K,
    MOTION_K,
    build_k48,
    expand_k47_weights,
    load_expanded,
    load_trained_k48,
)
from training.grpo_math import SIGMA, advantages, gaussian_logprob
from training.k48_config import CKPT, DATA, ROOT, SELECTED_LIMIT_S
from training.reward_v2 import grpo_return
from training.k47_codebook import CAP, categorical_kl, decode_action, episode_arrays, group_advantage, load_codebook, read_jsonl

from training.dataset import assert_recovery_v3_paths, collate, episode_tensors
from training.grpo_progress import reward_order

def supervised_loss(model, obs, prev, tokens, residuals, weight, mask):
    logits, pred, _act = model(obs, prev)
    if logits.shape[-1] != LOGIT_K:
        raise RuntimeError("token head is not 48-wide")
    per_token = []
    per_resid = []
    for index in range(logits.shape[0]):
        active = mask[index]
        per_token.append(F.cross_entropy(logits[index][active], tokens[index][active]))
        motion = active & (tokens[index] != HANDOFF)
        if bool(motion.any()):
            per_resid.append(((pred[index][motion] - residuals[index][motion]) ** 2).mean())
        else:
            per_resid.append(pred.new_zeros(()))
    token_vec = torch.stack(per_token)
    resid_vec = torch.stack(per_resid)
    w = weight.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    loss = (w * (token_vec + resid_vec)).mean()
    return loss, token_vec.mean().detach(), resid_vec.mean().detach()

def check():
    device = torch.device("cpu")
    book = load_codebook()
    model, loaded = load_expanded(CKPT, device)
    if not np.array_equal(loaded, book):
        raise RuntimeError("expanded checkpoint codebook differs from the K=47 table")
    try:
        decode_action(book, HANDOFF, np.zeros(7, np.float32))
    except RuntimeError as exc:
        if "outside the motion codebook" not in str(exc):
            raise
    else:
        raise RuntimeError("HANDOFF was decoded as a 7-D motion")
    # decode_action indexes the book. Token 47 must not be a legal motion index.
    if len(book) != MOTION_K:
        raise RuntimeError("codebook width changed")
    saved_bias = model.logits.bias.detach().clone()
    with torch.no_grad():
        model.logits.bias.zero_()
        model.logits.bias[HANDOFF] = 1.0e6
    _logits, _res, forced = model(torch.zeros(1, 2, 39, device=device), torch.zeros(1, 2, 7, device=device))
    if int(torch.argmax(_logits, dim=-1).reshape(-1)[0]) != HANDOFF or float(forced.detach().abs().max()) > 1e-6:
        raise RuntimeError("HANDOFF still decodes a motion command")
    with torch.no_grad():
        model.logits.bias.copy_(saved_bias)
    obs = torch.zeros(2, 5, 39, device=device)
    prev = torch.zeros(2, 5, 7, device=device)
    tokens = torch.tensor([
        [0, 3, 5, HANDOFF, 0],
        [1, 2, 4, 6, HANDOFF],
    ], device=device)
    residuals = torch.zeros(2, 5, 7, device=device)
    mask = torch.tensor([
        [True, True, True, True, False],
        [True, True, True, True, True],
    ], device=device)
    weight = torch.tensor([1 / 61, 1 / 81], device=device)
    model.train()
    loss, token_loss, resid_loss = supervised_loss(model, obs, prev, tokens, residuals, weight, mask)
    loss.backward()
    model.zero_grad(set_to_none=True)
    model.eval()
    hand = step_logprob(model, obs[0, :4], prev[0, :4], HANDOFF, np.zeros(7, np.float32))
    motion = step_logprob(model, obs[0, :3], prev[0, :3], 0, np.zeros(7, np.float32))
    if not torch.isfinite(hand) or not torch.isfinite(motion):
        raise RuntimeError("log-probability is not finite")
    # A HANDOFF log-probability has no residual term, so changing the residual pre-value does not change it.
    hand_other = step_logprob(model, obs[0, :4], prev[0, :4], HANDOFF, np.ones(7, np.float32))
    if float(hand.detach()) != float(hand_other.detach()):
        raise RuntimeError("HANDOFF log-probability depends on a residual")
    rewards = reward_order()
    print("K48_CHECK", {
        "logits": int(model.logits.out_features),
        "token_loss": float(token_loss),
        "resid_loss": float(resid_loss),
        "sigma": SIGMA,
        **rewards,
    }, flush=True)

def train_supervised(cfg, checkpoint):
    from envs.simulation.yaml_config import load_yaml
    from training.k47_codebook import read_jsonl

    del load_yaml
    book = load_codebook(ROOT / cfg["codebook"])
    rows = read_jsonl(ROOT / cfg["supervised_targets"])
    rebuilt = assert_recovery_v3_paths(cfg)
    if rebuilt:
        if not rows:
            raise RuntimeError("rebuilt supervised manifest is empty")
        for row in rows:
            label = row.get("nominal_class")
            if label == "RECOVERY_CANDIDATE" and row.get("verified_success") is not True:
                raise RuntimeError(f"{row.get('id')} recovery row is not a verified trajectory")
            if label == "NOMINAL_SOLVABLE" and row.get("handoff_label") != "immediate_handoff":
                raise RuntimeError(f"{row.get('id')} control row is not an immediate HANDOFF label")
            if label not in ("RECOVERY_CANDIDATE", "NOMINAL_SOLVABLE"):
                raise RuntimeError(f"{row.get('id')} has nominal class {label}")
    elif len(rows) < 296:
        raise RuntimeError(f"K=48 supervised file has {len(rows)} rows")
    device = torch.device(cfg.get("device") or "cpu")
    torch.manual_seed(int(cfg["seed"]))
    model, _book = load_expanded(Path(checkpoint), device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    out = ROOT / cfg["supervised_output"]
    historical = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k47/supervised"
    collapsed = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/supervised"
    if out.resolve() == historical.resolve() or (rebuilt and out.resolve() == collapsed.resolve()):
        raise RuntimeError("refusing to write into a historical supervised directory")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["seed"]))
    batch = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        order = rng.permutation(len(rows))
        total = 0.0
        seen = 0
        for start in range(0, len(rows), batch):
            batch_rows = [rows[int(index)] for index in order[start:start + batch]]
            obs, prev, tokens, residuals, mask, weight = collate(batch_rows, book, device)
            opt.zero_grad(set_to_none=True)
            loss, _token_loss, _resid_loss = supervised_loss(model, obs, prev, tokens, residuals, weight, mask)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            seen += 1
        torch.save(
            {"proto": book, "res_cap": CAP, "handoff": HANDOFF, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epoch": epoch},
            out / f"checkpoint_epoch_{epoch:04d}.pt",
        )
        print("SUPERVISED", epoch, round(total / max(seen, 1), 6), flush=True)
