"""K47 motion codebook and token decode.

47 motion codes. decode_action rejects an index outside that table, including
token 47. The K48 HANDOFF logit is a different head and is not a motion code.
The historical K47 training loop is experiments/legacy/k47_training.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.k48_policy import build_policy
from training.grpo_math import advantages

RAW = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2"
CODEBOOK = RAW / "codebook_selected_k47.npy"
K36_PATH = ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt"
EXTRA_PATH = RAW / "candidate_extra.npy"
SUPERVISED = RAW / "supervised_targets.jsonl"
GRPO_ICS = RAW / "grpo_initial_conditions.jsonl"
FRAGILE = RAW / "fragile_trajectories.jsonl"
CAP = 0.25
REQUIRED_K = 47


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_codebook(path=CODEBOOK):
    book = np.load(path).astype(np.float32)
    if book.shape != (REQUIRED_K, 7):
        raise RuntimeError(f"selected codebook must be ({REQUIRED_K}, 7), got {book.shape}")
    k36 = np.asarray(torch.load(K36_PATH, map_location="cpu", weights_only=False)["proto"], np.float32)
    extra = np.load(EXTRA_PATH).astype(np.float32)
    if not np.array_equal(book[:36], k36):
        raise RuntimeError("tokens 0-35 do not match the original K=36 table")
    if extra.shape != (11, 7) or not np.array_equal(book[36:], extra):
        raise RuntimeError("tokens 36-46 do not match the selected appended centers")
    return book


def build_model(book, device):
    model = build_policy(book, device, CAP)
    if model.logits.out_features != book.shape[0] or model.res.out_features != 7:
        raise RuntimeError("policy head does not match the codebook")
    return model


def decode_action(proto, token, pre):
    """Same map used by the supervised target, the GRPO sample, and evaluation."""
    if int(token) < 0 or int(token) >= len(proto):
        raise RuntimeError(f"token {int(token)} is outside the motion codebook")
    residual = np.clip(np.asarray(pre, np.float32), -CAP, CAP)
    action = np.clip(np.asarray(proto[int(token)], np.float32) + residual, -1.0, 1.0)
    return residual, action


def episode_arrays(row, book):
    blob = np.load(ROOT / row["continuous_npz"])
    actions = blob["actions"].astype(np.float32)
    obs = blob["observations"].astype(np.float32)
    if obs.ndim != 2 or obs.shape[1] != 39 or len(obs) != len(actions):
        raise RuntimeError(f"{row['id']} {row['strategy']} observation alignment")
    token_path = ROOT / row["token_residual_npz"]
    packed = np.load(token_path)
    tokens = packed["tokens"].astype(np.int64)
    residuals = packed["residuals"].astype(np.float32)
    if len(tokens) != len(actions):
        raise RuntimeError(f"{row['id']} {row['strategy']} was split across token files")
    if int(tokens.max()) >= book.shape[0] or int(tokens.min()) < 0:
        raise RuntimeError(f"{row['id']} token outside K={book.shape[0]}")
    if float(np.max(np.abs(residuals))) > CAP + 1e-5:
        raise RuntimeError(f"{row['id']} residual outside ±{CAP}")
    prev = np.zeros_like(actions)
    if len(actions) > 1:
        prev[1:] = actions[:-1]
    return obs, prev, actions, tokens, residuals


def categorical_kl(model, ref_model, obs, prev):
    """KL(pi || pi_ref) on equal-width logit vectors. No token is padded."""
    logits, _mean, _act = model(obs, prev)
    with torch.no_grad():
        old, _old_mean, _old_act = ref_model(obs, prev)
    if logits.shape[-1] != old.shape[-1]:
        raise RuntimeError(f"KL requires equal token counts, got {logits.shape[-1]} and {old.shape[-1]}")
    log_q = torch.log_softmax(logits[-1].float(), dim=-1)
    log_p = torch.log_softmax(old[-1].float(), dim=-1)
    q = log_q.exp()
    return (q * (log_q - log_p)).sum()


def group_advantage(returns, truncated):
    """Group-relative advantages. Identical returns get advantage 0.

    A truncated rollout is omitted. It is not scored as a drop.
    """
    keep = [value for value, flag in zip(returns, truncated) if not flag]
    if len(keep) < 2:
        return np.zeros(len(returns), np.float64), True
    adv_keep, _std, skipped = advantages(keep)
    out = np.zeros(len(returns), np.float64)
    cursor = 0
    for index, flag in enumerate(truncated):
        if flag:
            continue
        out[index] = adv_keep[cursor]
        cursor += 1
    return out, skipped


