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
from training.grpo_progress import (
    _completed_grpo_update,
    assert_selected_reward,
    format_seconds,
    note_update_duration,
    open_grpo_progress,
    progress_postfix,
    progress_timing,
    reward_order,
    summarize_group,
    write_progress_row,
)

def step_logprob(model, obs, prev, token, pre):
    logits, mean, _act = model(obs, prev)
    logit = logits[-1].float()
    if int(token) < 0 or int(token) >= logit.shape[-1]:
        raise RuntimeError("token outside the 48-way head")
    token_lp = torch.log_softmax(logit, dim=-1)[int(token)]
    if int(token) == HANDOFF:
        return token_lp
    return token_lp + gaussian_logprob(torch.as_tensor(pre, device=token_lp.device, dtype=token_lp.dtype), mean[-1].float())

def grpo_ic_schedule(ics, updates, ic_epochs=None):
    """Return the update order. The index is a function of the update number only."""
    ids = [row["id"] for row in ics]
    if len(ids) != len(set(ids)):
        raise RuntimeError("GRPO manifest repeats an initial condition")
    sequence = [ids[(update - 1) % len(ids)] for update in range(1, int(updates) + 1)]
    if ic_epochs is None:
        return sequence
    epochs = int(ic_epochs)
    width = len(ids)
    if int(updates) != epochs * width:
        raise RuntimeError(f"{updates} updates are not {epochs} passes over {width} initial conditions")
    counts = Counter(sequence)
    if any(counts[episode_id] != epochs for episode_id in ids):
        raise RuntimeError("IC scheduler did not visit every initial condition once per epoch")
    for epoch in range(epochs):
        if sequence[epoch * width:(epoch + 1) * width] != ids:
            raise RuntimeError("an epoch did not repeat the manifest order")
    return sequence

def grpo_checkpoint_updates(updates, every):
    step = int(every)
    return [update for update in range(1, int(updates) + 1) if update % step == 0 or update == int(updates)]

def train_grpo(cfg, checkpoint):
    from training.recovery_episode import make_training_env
    from envs.simulation.capture import restore_capture
    from envs.outcomes.horizon_outcome import finish_recovery, physical_reward, recovery_failed
    from envs.step_legal import step_legal
    from training.k47_codebook import categorical_kl, read_jsonl

    ics = read_jsonl(ROOT / cfg["grpo_initial_conditions"])
    rebuilt = assert_recovery_v3_paths(cfg)
    if any("final_test" in row["snapshot"] or "validation" in row["snapshot"] or "final" in row["snapshot"] for row in ics):
        raise RuntimeError("GRPO manifest contains a sealed split")
    if rebuilt:
        pools = {row.get("pool") for row in ics}
        if pools != {"recovery"}:
            raise RuntimeError(f"GRPO pool must be recovery only, found {pools}")
        nominal = [row["id"] for row in ics if row.get("nominal_class") != "RECOVERY_CANDIDATE"]
        if nominal:
            raise RuntimeError(f"nominal-solvable controls are not in the GRPO recovery pool: {nominal}")
        if any(row.get("split") == "valid" for row in ics):
            raise RuntimeError("GRPO recovery pool contains a validation initial condition")
    updates = int(cfg["grpo_updates"])
    limit_s = float(cfg.get("recovery_limit_s") or SELECTED_LIMIT_S)
    reward_version = str(cfg.get("reward_version") or "v1")
    if reward_version not in ("v1", "v2"):
        raise RuntimeError(f"unknown reward version {reward_version}")
    schedule = grpo_ic_schedule(ics, updates, cfg.get("ic_epochs"))
    if cfg.get("cover_all_ics") and len(set(schedule)) != len(ics):
        raise RuntimeError(f"GRPO updates {updates} cover {len(set(schedule))} of {len(ics)} initial conditions")
    by_id = {row["id"]: row for row in ics}
    assert_selected_reward(reward_version, limit_s)
    save_updates = grpo_checkpoint_updates(updates, int(cfg.get("grpo_save_every") or 5))
    if cfg.get("ic_epochs") is not None:
        expected_saves = [epoch * len(ics) for epoch in range(1, int(cfg["ic_epochs"]) + 1)]
        if save_updates != expected_saves:
            raise RuntimeError(f"checkpoints at {save_updates} are not once per epoch {expected_saves}")
    out = ROOT / cfg["grpo_output"]
    completed = _completed_grpo_update(out, checkpoint)
    if completed >= updates:
        raise RuntimeError(f"{checkpoint} already reached update {completed}; configured total is {updates}")
    protected = {
        (ROOT / "results/archive/demo_results/diagnostics/raw/unified_k47/supervised").resolve(),
        (ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/supervised").resolve(),
    }
    if completed == 0:
        protected.add(Path(checkpoint).resolve().parent)
    if out.resolve() in protected:
        raise RuntimeError(f"refusing to write GRPO checkpoints into {out}")
    if completed == 0 and (any(out.glob("checkpoint_u*.pt")) or (out / "rollouts.jsonl").exists() or (out / "training_log.jsonl").exists()):
        raise RuntimeError(f"refusing to overwrite existing GRPO outputs in {out}")
    out.mkdir(parents=True, exist_ok=True)
    schedule = schedule[completed:]

    book = load_codebook(ROOT / cfg["codebook"])
    device = torch.device(cfg.get("device") or "cpu")
    model, saved_book = load_trained_k48(Path(checkpoint), device)
    model.train()
    reference = cfg.get("grpo_reference") if completed else checkpoint
    if completed and not reference:
        raise RuntimeError("appending GRPO requires grpo_reference, the frozen policy from the first update")
    ref_model, _ref_book = load_trained_k48(Path(reference), device)
    if not np.array_equal(saved_book, book):
        raise RuntimeError("checkpoint motion codebook differs from the configured codebook")
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    env = make_training_env(device, model=model)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["grpo_lr"]))
    cap = int(round(limit_s / 0.02))
    n_ics = len(ics)
    epoch_total = int(cfg["ic_epochs"]) if cfg.get("ic_epochs") is not None else max(1, (updates + n_ics - 1) // n_ics)
    recent_updates = deque(maxlen=20)
    wall_start = time.perf_counter()
    last_checkpoint = None
    progress = open_grpo_progress(updates, initial=completed)
    progress_log = (out / "training_log.jsonl").open("a", encoding="utf-8")
    from tqdm import tqdm
    try:
        for update, episode_id in enumerate(schedule, start=completed + 1):
            compute_started = time.perf_counter()
            row = by_id[episode_id]
            with (ROOT / row["snapshot"]).open("rb") as handle:
                blob = pickle.load(handle)
            rng = np.random.default_rng(int(cfg["seed"]) + 19 * update)
            members = []
            for _member in range(int(cfg["group"])):
                restore_capture(env, blob["capture"])
                records = []
                lost_low = 0
                how = None
                t0 = float(env.sim.data.time)
                for _step in range(cap):
                    if float(env.sim.data.time) - t0 >= limit_s - 1e-9:
                        how = "forced"
                        break
                    obs = np.asarray(env.observation(), np.float32)
                    env._obs_seq.append(obs)
                    env._prev_seq.append(env._prev.copy())
                    window_obs = np.stack(env._obs_seq[-160:]).astype(np.float32)
                    window_prev = np.stack(env._prev_seq[-160:]).astype(np.float32)
                    with torch.no_grad():
                        logits, mean, _act = model(
                            torch.tensor(window_obs, device=device),
                            torch.tensor(window_prev, device=device),
                        )
                    logit = logits[-1].float()
                    probs = torch.softmax(logit, dim=-1).cpu().numpy()
                    probs = np.clip(probs, 0.0, None)
                    probs = probs / max(float(probs.sum()), 1e-12)
                    token = int(rng.choice(len(probs), p=probs))
                    pre = mean[-1].detach().float().cpu().numpy() + rng.normal(0.0, SIGMA, size=7).astype(np.float32)
                    logp = step_logprob(model, torch.tensor(window_obs, device=device), torch.tensor(window_prev, device=device), token, pre)
                    records.append({
                        "obs": window_obs,
                        "prev": window_prev,
                        "token": token,
                        "pre": pre.astype(np.float32),
                        "logp": float(logp.detach()),
                    })
                    if token == HANDOFF:
                        how = "early"
                        break
                    _residual, action = decode_action(book, token, pre)
                    executed = step_legal(env.sim, action)
                    env._prev = np.asarray(executed, np.float32)
                    failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
                    if failed:
                        how = "drop"
                        break
                if how == "drop":
                    elapsed = float(env.sim.data.time) - t0
                    from envs.outcomes.table_drop import terminal_reason
                    reason = terminal_reason(env.sim)
                    outcome = {
                        "reason": reason,
                        "handoff": None,
                        "recovery_elapsed": elapsed,
                        "lift_completed": False,
                        "q_nominal": 0.0,
                        "reward": physical_reward(reason, elapsed),
                    }
                    invalid = False
                else:
                    outcome = finish_recovery(env, blob["ref"], t0, "forced" if how is None else how)
                    invalid = outcome["reason"] == "INVALID"
                scored = grpo_return(outcome, limit_s, reward_version, outcome.get("reward"))
                members.append({
                    "records": records,
                    "return": None if invalid else float(scored["return"]),
                    "truncated": invalid,
                    "label": outcome["reason"],
                    "reward_log": scored,
                })
            from training.k47_codebook import group_advantage
            adv, skipped = group_advantage([member["return"] for member in members], [member["truncated"] for member in members])
            opt.zero_grad(set_to_none=True)
            kl_value = None
            loss_value = None
            if not skipped:
                terms = []
                for member, advantage in zip(members, adv):
                    if member["truncated"] or abs(float(advantage)) < 1e-8 or not member["records"]:
                        continue
                    logp = torch.stack([
                        step_logprob(model, torch.tensor(rec["obs"], device=device), torch.tensor(rec["prev"], device=device), rec["token"], rec["pre"])
                        for rec in member["records"]
                    ]).mean()
                    behavior = float(np.mean([rec["logp"] for rec in member["records"]]))
                    ratio = torch.exp(torch.clamp(logp - behavior, -20.0, 20.0))
                    unclipped = ratio * float(advantage)
                    clipped = torch.clamp(ratio, 1.0 - float(cfg["clip"]), 1.0 + float(cfg["clip"])) * float(advantage)
                    terms.append(-torch.minimum(unclipped, clipped))
                if terms:
                    surrogate = torch.stack(terms).mean()
                    rec = next(member["records"][0] for member in members if member["records"])
                    kl = categorical_kl(
                        model, ref_model,
                        torch.tensor(rec["obs"], device=device),
                        torch.tensor(rec["prev"], device=device),
                    )
                    loss = surrogate + float(cfg["kl_coef"]) * kl
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    loss_value = float(loss.detach())
                    kl_value = float(kl.detach())
            update_s = time.perf_counter() - compute_started
            checkpoint_s = 0.0
            if update in save_updates:
                save_started = time.perf_counter()
                last_checkpoint = out / f"checkpoint_u{update:04d}.pt"
                if last_checkpoint.exists():
                    raise RuntimeError(f"refusing to overwrite {last_checkpoint}")
                torch.save(
                    {
                        "proto": book,
                        "res_cap": CAP,
                        "handoff": HANDOFF,
                        "reward_version": reward_version,
                        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "update": update,
                    },
                    last_checkpoint,
                )
                checkpoint_s = time.perf_counter() - save_started
            elapsed_s = time.perf_counter() - wall_start
            summary = summarize_group(members, adv, skipped, kl_value, loss_value)
            timing = progress_timing(
                update, updates, n_ics, epoch_total, update_s, recent_updates, elapsed_s, datetime.now(),
            )
            record = {
                "update": update,
                "total": updates,
                "episode_id": row["id"],
                "reward_version": reward_version,
                "checkpoint_s": checkpoint_s,
                "checkpoint": None if update not in save_updates else str(last_checkpoint),
                **summary,
                **timing,
            }
            write_progress_row(progress_log, record)
            with (out / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
                for index, member in enumerate(members):
                    logged = dict(member["reward_log"])
                    logged.update({"update": update, "member": index, "episode_id": row["id"]})
                    handle.write(json.dumps(logged) + "\n")
                    handle.flush()
            tqdm.write(
                "GRPO "
                f"{update}/{updates} {reward_version} {row['id']} "
                f"outcomes={summary['outcomes']} "
                f"reward_mean={summary['reward_mean']} reward_std={summary['reward_std']} "
                f"adv_mean={summary['advantage_mean']} adv_std={summary['advantage_std']} "
                f"adv_skipped={summary['advantage_skipped']} kl={summary['kl']} loss={summary['policy_loss']} "
                f"handoff={summary['handoff']} success={summary['success']}/{summary['group']}"
            )
            progress.set_postfix_str(progress_postfix(timing, summary["success"], summary["group"]), refresh=False)
            progress.update(1)
        elapsed_s = time.perf_counter() - wall_start
        tqdm.write(
            f"GRPO_DONE {updates}/{updates} elapsed={format_seconds(elapsed_s)} checkpoint={last_checkpoint}"
        )
    finally:
        progress.close()
        progress_log.close()
