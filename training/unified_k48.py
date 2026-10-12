"""K=48 recovery policy: 47 motion tokens plus one HANDOFF token.

`--check` loads the historical K=47 checkpoint, expands the token head, and
runs loss and log-probability tests. It does not take an optimizer step and
does not write a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import pickle
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

from training.action_representation_expansion import build_policy
from training.horizon_outcome import LAMBDA_TIME, NOMINAL_S, physical_reward
from training.reward_v2 import grpo_return
from training.pre_grpo_sanity import SIGMA, advantages, gaussian_logprob
from training.unified_k47 import CAP, decode_action, group_advantage, load_codebook

MOTION_K = 47
HANDOFF = 47
LOGIT_K = 48
SELECTED_LIMIT_S = 4.0
CKPT = ROOT / "results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt"
DATA = ROOT / "results/diagnostics/raw/k48_handoff"


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


def step_logprob(model, obs, prev, token, pre):
    logits, mean, _act = model(obs, prev)
    logit = logits[-1].float()
    if int(token) < 0 or int(token) >= logit.shape[-1]:
        raise RuntimeError("token outside the 48-way head")
    token_lp = torch.log_softmax(logit, dim=-1)[int(token)]
    if int(token) == HANDOFF:
        return token_lp
    return token_lp + gaussian_logprob(torch.as_tensor(pre, device=token_lp.device, dtype=token_lp.dtype), mean[-1].float())


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


def episode_tensors(row, book):
    from training.unified_k47 import episode_arrays

    obs, prev, actions, tokens, residuals = episode_arrays(row, book)
    if row.get("handoff_label") != "early_handoff":
        return obs, prev, tokens, residuals
    packed = np.load(ROOT / row["continuous_npz"])
    if "handoff_obs" not in packed.files:
        raise RuntimeError(f"{row['id']} early HANDOFF has no handoff observation")
    extra = np.asarray(packed["handoff_obs"], np.float32).reshape(1, 39)
    obs = np.concatenate([obs, extra], axis=0)
    tokens = np.concatenate([tokens, np.array([HANDOFF], np.int64)])
    residuals = np.concatenate([residuals, np.zeros((1, 7), np.float32)], axis=0)
    prev_h = np.zeros((len(prev) + 1, 7), np.float32)
    prev_h[:-1] = prev
    prev_h[-1] = actions[-1]
    return obs, prev_h, tokens, residuals


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
    if handoff_steps > motion_steps:
        raise RuntimeError("HANDOFF labels outnumber motion labels in this batch")
    return obs, prev, tokens, residuals, mask, torch.tensor(weights, device=device, dtype=torch.float32)


def train_supervised(cfg, checkpoint):
    from envs.config_util import load_yaml
    from training.unified_k47 import read_jsonl

    del load_yaml
    book = load_codebook(ROOT / cfg["codebook"])
    rows = read_jsonl(ROOT / cfg["supervised_targets"])
    if len(rows) < 296:
        raise RuntimeError(f"K=48 supervised file has {len(rows)} rows")
    device = torch.device(cfg.get("device") or "cpu")
    torch.manual_seed(int(cfg["seed"]))
    model, _book = load_expanded(Path(checkpoint), device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    out = ROOT / cfg["supervised_output"]
    historical = ROOT / "results/diagnostics/raw/unified_k47/supervised"
    if out.resolve() == historical.resolve():
        raise RuntimeError("refusing to write into the historical epoch-40 directory")
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


def open_grpo_progress(total, file=None):
    from tqdm import tqdm

    stream = sys.stdout if file is None else file
    return tqdm(
        total=int(total),
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


def train_grpo(cfg, checkpoint):
    from training.autonomous_episode import make_training_env
    from training.history_aware_critic_ablation import restore_capture
    from training.horizon_outcome import finish_recovery, physical_reward, recovery_failed
    from training.privileged_ball_impact_search import step_legal
    from training.unified_k47 import categorical_kl, read_jsonl

    ics = read_jsonl(ROOT / cfg["grpo_initial_conditions"])
    if any("final_test" in row["snapshot"] or "validation" in row["snapshot"] for row in ics):
        raise RuntimeError("GRPO manifest contains a sealed split")
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
    protected = {
        (ROOT / "results/diagnostics/raw/unified_k47/supervised").resolve(),
        (ROOT / "results/diagnostics/raw/unified_k48/supervised").resolve(),
        Path(checkpoint).resolve().parent,
    }
    if out.resolve() in protected:
        raise RuntimeError(f"refusing to write GRPO checkpoints into {out}")
    if any(out.glob("checkpoint_u*.pt")) or (out / "rollouts.jsonl").exists() or (out / "training_log.jsonl").exists():
        raise RuntimeError(f"refusing to overwrite existing GRPO outputs in {out}")
    out.mkdir(parents=True, exist_ok=True)

    book = load_codebook(ROOT / cfg["codebook"])
    device = torch.device(cfg.get("device") or "cpu")
    model, saved_book = load_trained_k48(Path(checkpoint), device)
    model.train()
    ref_model, _ref_book = load_trained_k48(Path(checkpoint), device)
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
    progress = open_grpo_progress(updates)
    progress_log = (out / "training_log.jsonl").open("a", encoding="utf-8")
    from tqdm import tqdm
    try:
        for update, episode_id in enumerate(schedule, start=1):
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
                    outcome = {
                        "reason": "DROP_DURING_RECOVERY",
                        "handoff": None,
                        "recovery_elapsed": elapsed,
                        "lift_completed": False,
                        "q_nominal": 0.0,
                        "reward": physical_reward("DROP_DURING_RECOVERY", elapsed),
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
            from training.unified_k47 import group_advantage
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


def evaluate(cfg, checkpoint):
    from training.autonomous_episode import make_training_env
    from training.horizon_outcome import roll_policy
    from training.unified_k47 import read_jsonl

    checkpoint_path = Path(checkpoint)
    before = checkpoint_path.stat().st_mtime
    size_before = checkpoint_path.stat().st_size
    device_name = str(cfg.get("eval_device") or "cpu")
    device = torch.device(device_name)
    model, book = load_trained_k48(checkpoint_path, device)
    if model.handoff_index != HANDOFF or model.logits.out_features != LOGIT_K:
        raise RuntimeError("loaded policy is not the 48-token handoff head")
    limit_s = float(cfg.get("recovery_limit_s") or SELECTED_LIMIT_S)
    if abs(limit_s - SELECTED_LIMIT_S) > 1e-9 or abs(float(cfg.get("nominal_seconds") or NOMINAL_S) - NOMINAL_S) > 1e-9:
        raise RuntimeError("evaluation must use a 4 s recovery limit and a 12 s nominal window")
    env = make_training_env(device, model=model)
    baseline_path = ROOT / cfg["baseline_eval"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_ids = [row["id"] for row in baseline]
    if len(baseline_ids) != 56 or len(set(baseline_ids)) != 56:
        raise RuntimeError("K=47 baseline is not the 56-IC evaluation")
    by_id = {
        row["id"]: row
        for row in read_jsonl(ROOT / cfg["grpo_initial_conditions"])
        if row.get("source") == "ball_impact_v2" and int(row["id"].rsplit("_", 1)[-1]) % 5 == 0
    }
    if set(by_id) != set(baseline_ids):
        raise RuntimeError("K=48 evaluation ICs differ from the 56 K=47 baseline ICs")
    rows = []
    for episode_id in baseline_ids:
        row = by_id[episode_id]
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if blob["spec"].get("split") not in (None, "train"):
            raise RuntimeError(f"evaluation refused non-train snapshot {episode_id}")
        out = roll_policy(
            env, model, book, blob, device,
            cap_steps=int(cfg.get("eval_cap_steps") or 200),
            limit_s=limit_s,
        )
        rows.append({
            "id": episode_id,
            "label": out["reason"],
            "success": bool(out.get("success")),
            "handoff": out.get("handoff"),
            "recovery_elapsed": out.get("recovery_elapsed"),
            "recovery_steps": out.get("recovery_steps"),
            "nominal_elapsed": out.get("nominal_elapsed"),
            "lift_completed": out.get("lift_completed"),
            "recovery_limit_s": limit_s,
            "nominal_seconds": NOMINAL_S,
            "device": device_name,
            "truncated": out["reason"] == "INVALID",
        })
        print("EVAL", episode_id, out["reason"], out.get("handoff"), out.get("recovery_elapsed"), flush=True)
    elapsed = [float(row["recovery_elapsed"]) for row in rows if row.get("recovery_elapsed") is not None]
    base_success = sum(1 for row in baseline if row.get("success") or row.get("reason") == "SUCCESS_12S")
    summary = {
        "episodes": len(rows),
        "success": sum(1 for row in rows if row["success"]),
        "baseline_success": base_success,
        "baseline_episodes": len(baseline),
        "baseline_file": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
        "device": device_name,
        "recovery_limit_s": limit_s,
        "nominal_seconds": NOMINAL_S,
        "reasons": dict(Counter(row["label"] for row in rows)),
        "handoff": dict(Counter(row["handoff"] for row in rows)),
        "recovery_elapsed_s": {
            "min": min(elapsed) if elapsed else None,
            "median": float(np.median(elapsed)) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
        },
    }
    out_dir = ROOT / cfg["eval_output"]
    historical = {
        (ROOT / "results/diagnostics/raw/unified_k47/eval").resolve(),
        (ROOT / "results/diagnostics/raw/horizon_migration").resolve(),
        (ROOT / "results/diagnostics/raw/unified_k47/supervised").resolve(),
    }
    if out_dir.resolve() in historical:
        raise RuntimeError(f"refusing to write evaluation into {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    (out_dir / "heldout_train.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if checkpoint_path.stat().st_mtime != before or checkpoint_path.stat().st_size != size_before:
        raise RuntimeError("evaluation changed the checkpoint")
    print("EVAL_DONE", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description="K=48 HANDOFF policy. --check does not train.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--eval", action="store_true")
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
        evaluate(cfg, args.checkpoint)
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
