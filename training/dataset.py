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

def recorded_end_includes_handoff(n_actions):
    """A recorded_end tape is supervised with token 47 only if another decision fits.

    Each stored action is one 20 ms step. After 200 steps the recovery clock is
    exactly 4 s, and the rollout forces handoff before it reads another
    observation. That forced transition is not a model target. A shorter tape
    leaves the clock inside the limit, so the next observation is the state
    after the last action and the target there is HANDOFF.
    """
    steps = int(n_actions)
    if steps < 1:
        return False
    return steps * 0.02 < SELECTED_LIMIT_S - 1e-9

def _append_handoff_target(row, obs, prev, actions, tokens, residuals):
    packed = np.load(ROOT / row["continuous_npz"])
    if "handoff_obs" not in packed.files:
        raise RuntimeError(f"{row['id']} HANDOFF target has no handoff observation")
    extra = np.asarray(packed["handoff_obs"], np.float32).reshape(1, 39)
    obs = np.concatenate([obs, extra], axis=0)
    tokens = np.concatenate([tokens, np.array([HANDOFF], np.int64)])
    residuals = np.concatenate([residuals, np.zeros((1, 7), np.float32)], axis=0)
    prev_h = np.zeros((len(prev) + 1, 7), np.float32)
    prev_h[:-1] = prev
    prev_h[-1] = actions[-1]
    return obs, prev_h, tokens, residuals

def episode_tensors(row, book):
    from training.k47_codebook import episode_arrays

    if row.get("handoff_label") == "immediate_handoff":
        packed = np.load(ROOT / row["continuous_npz"])
        if "handoff_obs" not in packed.files:
            raise RuntimeError(f"{row['id']} immediate HANDOFF has no observation")
        obs = np.asarray(packed["handoff_obs"], np.float32).reshape(1, 39)
        prev = np.zeros((1, 7), np.float32)
        tokens = np.array([HANDOFF], np.int64)
        residuals = np.zeros((1, 7), np.float32)
        return obs, prev, tokens, residuals
    obs, prev, actions, tokens, residuals = episode_arrays(row, book)
    label = row.get("handoff_label")
    if label == "early_handoff":
        return _append_handoff_target(row, obs, prev, actions, tokens, residuals)
    if label == "recorded_end" and recorded_end_includes_handoff(len(actions)):
        return _append_handoff_target(row, obs, prev, actions, tokens, residuals)
    return obs, prev, tokens, residuals

def collate(rows, book, device):
    packed = [episode_tensors(row, book) for row in rows]
    length = max(item[0].shape[0] for item in packed)
    obs = torch.zeros(len(packed), length, 39, device=device)
    prev = torch.zeros(len(packed), length, 7, device=device)
    tokens = torch.zeros(len(packed), length, dtype=torch.long, device=device)
    residuals = torch.zeros(len(packed), length, 7, device=device)
    mask = torch.zeros(len(packed), length, dtype=torch.bool, device=device)
    weights = []
    handoff_steps = 0
    motion_steps = 0
    for index, (obs_i, prev_i, tok_i, res_i) in enumerate(packed):
        n = obs_i.shape[0]
        obs[index, :n] = torch.tensor(obs_i, device=device)
        prev[index, :n] = torch.tensor(prev_i, device=device)
        tokens[index, :n] = torch.tensor(tok_i, device=device)
        residuals[index, :n] = torch.tensor(res_i, device=device)
        mask[index, :n] = True
        weights.append(float(rows[index]["sample_weight"]))
        handoff_steps += int(np.sum(tok_i == HANDOFF))
        motion_steps += int(np.sum(tok_i != HANDOFF))
    immediate = any(row.get("handoff_label") == "immediate_handoff" for row in rows)
    if handoff_steps > motion_steps and not immediate:
        raise RuntimeError("HANDOFF labels outnumber motion labels in this batch")
    return obs, prev, tokens, residuals, mask, torch.tensor(weights, device=device, dtype=torch.float32)

def assert_recovery_v3_paths(cfg):
    """Training reads data/. The diagnostic copy under k48_recovery_v3 remains readable."""
    if str(cfg.get("dataset_version") or "") != RECOVERY_V3:
        return False
    supervised = ROOT / cfg["supervised_targets"]
    grpo = ROOT / cfg["grpo_initial_conditions"]
    allowed = (DATA_ROOT.resolve(), RECOVERY_V3_ROOT.resolve())
    for path in (supervised, grpo):
        resolved = path.resolve()
        if not any(resolved == root or root in resolved.parents for root in allowed):
            raise RuntimeError(f"{path} is not inside the rebuilt recovery dataset")
        text = str(path).replace("\\", "/")
        if "k48_handoff/supervised_targets" in text or "k48_handoff/grpo_initial_conditions" in text:
            raise RuntimeError(f"refusing the collapsed dataset manifest {path}")
        if "final" in text:
            raise RuntimeError(f"refusing a final path {path}")
    return True
