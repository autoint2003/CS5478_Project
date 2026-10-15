"""K=48 recovery policy: 47 motion tokens plus one HANDOFF token.

`--check` loads the historical K=47 checkpoint, expands the token head, and
runs loss and log-probability tests. It does not take an optimizer step and
does not write a checkpoint.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.horizon_outcome import LAMBDA_TIME, NOMINAL_S, physical_reward
from training.reward_v2 import grpo_return
from training.grpo_math import SIGMA, advantages, gaussian_logprob
from training.unified_k47 import CAP, decode_action, group_advantage, load_codebook
from models.k48_checkpoint import (
    HANDOFF,
    LOGIT_K,
    MOTION_K,
    build_k48,
    expand_k47_weights,
    load_expanded,
    load_trained_k48,
)

from training.k48_config import CKPT, DATA, SELECTED_LIMIT_S
from training.dataset import assert_recovery_v3_paths, collate, episode_tensors, recorded_end_includes_handoff
from training.train_sft import check, supervised_loss, train_supervised
from training.train_grpo import grpo_ic_schedule, step_logprob, train_grpo
from training.trainer import (
    _completed_grpo_update,
    assert_selected_reward,
    epoch_progress,
    format_seconds,
    note_update_duration,
    open_grpo_progress,
    progress_postfix,
    progress_timing,
    reward_order,
    summarize_group,
    write_progress_row,
)


















RECOVERY_V3 = "k48_recovery_v3"
RECOVERY_V3_ROOT = ROOT / "results/archive/demo_results/diagnostics/raw/k48_recovery_v3"
DATA_ROOT = ROOT / "data"
































def evaluate(cfg, checkpoint, eval_data=""):
    from evaluation.k48_eval import evaluate as run

    return run(cfg, checkpoint, eval_data)



def main():
    parser = argparse.ArgumentParser(description="K=48 HANDOFF policy. --check does not train.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-output", default="")
    parser.add_argument("--eval-data", default="data/valid_recovery.jsonl")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--config", default="config/unified_k48.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    if args.check:
        check()
        return
    from envs.config_util import load_yaml
    cfg = load_yaml(args.config)
    if abs(float(cfg.get("recovery_limit_s", SELECTED_LIMIT_S)) - SELECTED_LIMIT_S) > 1e-9:
        raise RuntimeError("recovery limit does not match the selected 4 second maximum")
    if abs(float(cfg.get("nominal_seconds", NOMINAL_S)) - NOMINAL_S) > 1e-9:
        raise RuntimeError("nominal window is not 12 seconds")
    if args.eval:
        if not args.checkpoint:
            raise SystemExit("--eval requires a trained K=48 checkpoint")
        if args.eval_output:
            cfg["eval_output"] = args.eval_output
        evaluate(cfg, args.checkpoint, args.eval_data)
        return
    if not args.checkpoint:
        raise SystemExit("pass the epoch-40 K=47 checkpoint; it is expanded and not overwritten")
    if args.supervised:
        train_supervised(cfg, args.checkpoint)
        return
    if args.grpo:
        train_grpo(cfg, args.checkpoint)
        return
    raise SystemExit("choose --check, --supervised, --eval, or --grpo")


if __name__ == "__main__":
    main()
