"""GRPO progress log, reward-order check, and update timing.

Split from training/unified_k48.py. The formulas are unchanged.
"""

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

def reward_order():
    fast = physical_reward("SUCCESS_12S", 0.4)
    slow = physical_reward("SUCCESS_12S", SELECTED_LIMIT_S)
    fail = physical_reward("DROP_DURING_RECOVERY", 0.2)
    if not (fast > slow > 0.0 > fail):
        raise RuntimeError(f"reward order {fast} {slow} {fail}")
    if abs((1.0 - slow) - LAMBDA_TIME * SELECTED_LIMIT_S) > 1e-9:
        raise RuntimeError("time penalty does not use recovery seconds")
    adv, _std, skipped = advantages([fast, slow, fail, fail])
    _same, _std2, skipped_same = advantages([slow, slow, slow])
    grouped, skipped_trunc = group_advantage([slow, fail, None], [False, False, True])
    if skipped or skipped_trunc or not skipped_same or abs(float(grouped[2])) > 0.0:
        raise RuntimeError("advantage smoke failed")
    return {"fast": fast, "slow": slow, "fail": fail, "adv0": float(adv[0]), "nominal_s": NOMINAL_S}

def format_seconds(seconds):
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"

def epoch_progress(update, n_ics, epoch_total):
    epoch = (int(update) - 1) // int(n_ics) + 1
    total = int(epoch_total)
    return epoch, total, f"{epoch}/{total}"

def note_update_duration(recent, update_s):
    """Rolling window of compute time. Checkpoint writes are not passed in."""
    recent.append(float(update_s))
    return float(sum(recent) / len(recent))

def progress_timing(update, total, n_ics, epoch_total, update_s, recent, elapsed_s, now):
    average = note_update_duration(recent, update_s)
    remaining = max(0, int(total) - int(update))
    eta_s = 0.0 if int(update) >= int(total) else average * remaining
    finish = now + timedelta(seconds=eta_s)
    epoch, epochs, label = epoch_progress(update, n_ics, epoch_total)
    return {
        "percent": 100.0 * int(update) / int(total),
        "update_s": float(update_s),
        "avg_update_s": average,
        "elapsed_s": float(elapsed_s),
        "eta_s": eta_s,
        "eta_local": finish.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": epoch,
        "epochs": epochs,
        "epoch_label": label,
    }

def progress_postfix(timing, success, group):
    return (
        f"last={timing['update_s']:.2f}s "
        f"avg20={timing['avg_update_s']:.2f}s "
        f"elapsed={format_seconds(timing['elapsed_s'])} "
        f"eta={format_seconds(timing['eta_s'])} "
        f"done={timing['eta_local']} "
        f"epoch={timing['epoch_label']} "
        f"success={int(success)}/{int(group)}"
    )

def open_grpo_progress(total, file=None, initial=0):
    from tqdm import tqdm

    stream = sys.stdout if file is None else file
    return tqdm(
        total=int(total),
        initial=int(initial),
        file=stream,
        ascii=True,
        dynamic_ncols=bool(getattr(stream, "isatty", lambda: False)()),
        mininterval=0,
        miniters=1,
        leave=True,
        bar_format="{percentage:5.1f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
    )

def summarize_group(members, advantages, skipped, kl_value, loss_value):
    labels = [member["label"] for member in members]
    returns = [float(member["return"]) for member in members if member["return"] is not None]
    advantage = np.asarray(advantages, np.float64)
    handoff = Counter()
    for member in members:
        kind = member["reward_log"].get("handoff")
        handoff[kind if kind is not None else "none"] += 1
    return {
        "outcomes": dict(Counter(labels)),
        "success": sum(1 for label in labels if label == "SUCCESS_12S"),
        "group": len(members),
        "reward_mean": None if not returns else float(np.mean(returns)),
        "reward_std": None if not returns else float(np.std(returns)),
        "advantage_mean": None if advantage.size == 0 else float(advantage.mean()),
        "advantage_std": None if advantage.size == 0 else float(advantage.std()),
        "advantage_skipped": bool(skipped),
        "kl": None if kl_value is None else float(kl_value),
        "policy_loss": None if loss_value is None else float(loss_value),
        "handoff": dict(handoff),
    }

def write_progress_row(handle, record):
    handle.write(json.dumps(record) + "\n")
    handle.flush()

def assert_selected_reward(version, limit_s):
    """Fail before training if the configured version is wired to the other formula."""
    from training.reward_v2 import reward_components

    elapsed = 1.0
    historical = physical_reward("SUCCESS_12S", elapsed)
    outcome = {
        "reason": "SUCCESS_12S",
        "recovery_elapsed": elapsed,
        "lift_completed": True,
        "q_nominal": 0.0,
        "handoff": "early",
    }
    selected = grpo_return(outcome, limit_s, version, historical)["return"]
    outcome_aware = reward_components("SUCCESS_12S", elapsed, limit_s, q_nominal=0.0, lift_completed=True)["return"]
    original = 1.0 - LAMBDA_TIME * elapsed
    if abs(historical - original) > 1e-12:
        raise RuntimeError("original success reward is not 1 - 0.05 * recovery_seconds")
    if version == "v1" and abs(selected - historical) > 1e-12:
        raise RuntimeError("v1 selected a reward other than the original physical return")
    if version == "v2" and abs(selected - outcome_aware) > 1e-12:
        raise RuntimeError("v2 selected a reward other than the outcome-aware return")
    if abs(historical - outcome_aware) < 1e-9:
        raise RuntimeError("v1 and v2 success returns are the same")
    drop_elapsed = 0.2
    drop_historical = physical_reward("DROP_DURING_RECOVERY", drop_elapsed)
    drop = {
        "reason": "DROP_DURING_RECOVERY",
        "recovery_elapsed": drop_elapsed,
        "lift_completed": False,
        "q_nominal": 0.0,
        "handoff": None,
    }
    drop_selected = grpo_return(drop, limit_s, version, drop_historical)["return"]
    drop_v2 = reward_components("DROP_DURING_RECOVERY", drop_elapsed, limit_s)["return"]
    if version == "v1" and abs(drop_selected - drop_historical) > 1e-12:
        raise RuntimeError("v1 drop return is not the original physical return")
    if version == "v2" and abs(drop_selected - drop_v2) > 1e-12:
        raise RuntimeError("v2 drop return is not the outcome-aware return")

def _completed_grpo_update(out, checkpoint):
    """Return the finished update when ``checkpoint`` is an existing file in ``out``."""
    path = Path(checkpoint).resolve()
    if path.parent != out.resolve():
        return 0
    match = re.fullmatch(r"checkpoint_u(\d+)\.pt", path.name)
    if match is None:
        return 0
    done = int(match.group(1))
    log = out / "training_log.jsonl"
    if not log.exists():
        raise RuntimeError(f"{path.name} is in {out} but training_log.jsonl is missing")
    last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    if int(last["update"]) != done:
        raise RuntimeError(f"training log ends at update {last['update']}, not {done}")
    return done
