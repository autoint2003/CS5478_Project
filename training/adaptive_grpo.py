"""Short and full GRPO runs that explore with adaptive hybrid sampling.

The loss is the existing clipped group ratio plus KL. Only an independent
group that contains SUCCESS_12S is updated. Tree leaves are logged and then
discarded from the loss.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from experiments.handoff.benchmark_k48_prefix_branching import (
    rollout as tree_rollout,
    take_snapshot,
)
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.adaptive_hybrid import (
    STAGE_A,
    STAGE_B,
    TREE_EACH,
    balanced_schedule,
    exploration_decision,
    load_train_snapshots,
    loss_members,
    sampling_frequency,
    select_branch_points,
)
from training.grpo_math import SIGMA
from training.k47_codebook import CAP, categorical_kl, decode_action, group_advantage, load_codebook
from training.k48_config import ROOT, SELECTED_LIMIT_S
from training.reward_v2 import grpo_return
from training.train_grpo import step_logprob

SFT_CHECKPOINT = ROOT / "results/sft/current/checkpoint_epoch_0040.pt"


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key not in {"records", "captures", "steps"}}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _prepare_blob(blob: dict, seated_ref: dict | None) -> dict:
    if "ref" in blob and "capture" in blob:
        return blob
    if "ref" not in blob:
        if seated_ref is None:
            raise RuntimeError("horizontal snapshot needs the seated grasp reference")
        blob = dict(blob)
        blob["ref"] = seated_ref
    return blob


def independent_rollout(env, model, book, blob, rng, device, limit_s, reward_version, keep_captures):
    """One on-policy rollout from the original snapshot. Matches train_grpo sampling."""
    from envs.outcomes.horizon_outcome import finish_recovery, physical_reward, recovery_failed
    from envs.outcomes.table_drop import terminal_reason
    from envs.simulation.capture import restore_capture
    from envs.step_legal import step_legal

    if "capture" in blob:
        restore_capture(env, blob["capture"])
    else:
        from models.airborne_v3 import DT_POLICY
        from envs.observations.observable_obs import ObservableObsState
        from training.recovery_execution import restore_ballistic

        restore_ballistic(env.sim, blob)
        env._hist = ObservableObsState(DT_POLICY)
        env._obs_seq = []
        env._prev_seq = []
        env._prev = np.zeros(7, np.float32)
    t0 = float(env.sim.data.time)
    lost_low = 0
    how = None
    records = []
    steps = []
    captures = {}
    physics = 0
    for _step in range(int(round(limit_s / 0.02))):
        if float(env.sim.data.time) - t0 >= limit_s - 1e-9:
            how = "forced"
            break
        if keep_captures:
            captures[len(steps)] = take_snapshot(env)
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
        logp = step_logprob(
            model,
            torch.tensor(window_obs, device=device),
            torch.tensor(window_prev, device=device),
            token,
            pre,
        )
        entropy = float(-(probs[probs > 0.0] * np.log(probs[probs > 0.0])).sum())
        step = {
            "step": len(steps),
            "t": float(env.sim.data.time) - t0,
            "token": int(token),
            "entropy": entropy,
            "lost_low": int(lost_low),
        }
        steps.append(step)
        records.append({
            "obs": window_obs,
            "prev": window_prev,
            "token": int(token),
            "pre": pre.astype(np.float32),
            "logp": float(logp.detach()),
        })
        if token == HANDOFF:
            how = "early"
            break
        if token < 0 or token >= HANDOFF:
            raise RuntimeError(f"illegal motion token {token}")
        _residual, action = decode_action(book, token, pre)
        if float(np.max(np.abs(_residual))) > CAP + 1e-6:
            raise RuntimeError("residual left the K48 cap")
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        physics += 10
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how = "drop"
            break
    if how == "drop":
        elapsed = float(env.sim.data.time) - t0
        reason = terminal_reason(env.sim)
        outcome = {
            "reason": reason,
            "success": False,
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
    handoff_t = None
    if records and records[-1]["token"] == HANDOFF:
        handoff_t = float(steps[-1]["t"])
    return {
        "origin": "independent",
        "records": records,
        "steps": steps,
        "captures": captures,
        "return": None if invalid else float(scored["return"]),
        "truncated": invalid,
        "label": outcome["reason"],
        "success": bool(outcome.get("success")),
        "handoff_t": handoff_t,
        "recovery_physics": physics,
        "nominal_elapsed": float(outcome.get("nominal_elapsed") or 0.0),
        "reward_log": scored,
        "t0": t0,
    }


def _branch(env, model, book, blob, device, trunks, count, strategy, rng):
    candidates = []
    for trunk_index, trunk in enumerate(trunks):
        for step in trunk["steps"]:
            if step["step"] not in trunk["captures"]:
                continue
            candidates.append({
                "traj": trunk_index,
                "step": int(step["step"]),
                "entropy": float(step["entropy"]),
                "lost_low": int(step["lost_low"]),
                "t": float(step["t"]),
                "token": int(step["token"]),
            })
    points = select_branch_points(candidates, rng, count, strategy)
    leaves = []
    for index, point in enumerate(points):
        trunk = trunks[point["traj"]]
        suffix_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        started = time.perf_counter()
        scored, rows, _caps, _t0 = tree_rollout(
            env, model, book, blob, suffix_rng, device,
            capture=trunk["captures"][point["step"]],
            lost_low=point["lost_low"],
            t0=trunk["t0"],
            keep_captures=False,
        )
        leaves.append({
            "origin": "tree",
            "strategy": strategy,
            "success": bool(scored["success"]),
            "label": scored["reason"],
            "branch_step": point["step"],
            "branch_t": point["t"],
            "branch_entropy": point["entropy"],
            "prefix_source": trunk_index_of(trunks, point["traj"]),
            "trunk_token": point["token"],
            "suffix_tokens": [int(row["token"]) for row in rows],
            "handoff_t": None if not rows or rows[-1]["token"] != HANDOFF else float(rows[-1]["t"]),
            "wall_s": time.perf_counter() - started,
        })
    return leaves


def trunk_index_of(trunks, index):
    return index


def _sample_group(env, model, book, blob, device, limit_s, reward_version, seed, offset, count, keep_captures):
    group = []
    for index in range(count):
        rng = np.random.default_rng(int(seed) + 17 * (offset + index + 1))
        group.append(independent_rollout(
            env, model, book, blob, rng, device, limit_s, reward_version, keep_captures,
        ))
    return group


def explore_ic(env, model, book, blob, device, limit_s, reward_version, seed, episode_id):
    stage_a = _sample_group(env, model, book, blob, device, limit_s, reward_version, seed, 0, STAGE_A, False)
    state = {
        "stage_a_success": sum(int(item["success"]) for item in stage_a),
        "stage_b_success": 0,
        "stage_b_done": False,
        "tree_uniform_success": 0,
        "tree_entropy_success": 0,
        "tree_done": False,
        "resample_success": 0,
        "resample_done": False,
    }
    # Inject the already sampled stage into the shared runner by closing over it.
    return _continue(env, model, book, blob, device, limit_s, reward_version, seed, episode_id, stage_a, state)


def _continue(env, model, book, blob, device, limit_s, reward_version, seed, episode_id, stage_a, state):
    stage_b = []
    uniform = []
    entropy = []
    resample = []
    physics = sum(item["recovery_physics"] for item in stage_a)
    nominal = sum(item["nominal_elapsed"] for item in stage_a)
    while True:
        decision = exploration_decision(state)
        if decision["action"] == "update":
            chosen = {"stage_a": stage_a, "stage_ab": stage_a + stage_b, "resample": resample}[decision["group"]]
            log = _ic_log(episode_id, True, decision["source"], state, stage_a, stage_b, uniform, entropy, resample, physics, nominal)
            return log, loss_members(chosen)
        if decision["action"] == "skip":
            log = _ic_log(episode_id, False, None, state, stage_a, stage_b, uniform, entropy, resample, physics, nominal)
            log["tree_success_excluded"] = bool(decision.get("tree_success_excluded"))
            return log, None
        stage = decision["stage"]
        if stage == "B":
            stage_b = _sample_group(env, model, book, blob, device, limit_s, reward_version, seed, 100, STAGE_B, False)
            state["stage_b_done"] = True
            state["stage_b_success"] = sum(int(item["success"]) for item in stage_b)
            physics += sum(item["recovery_physics"] for item in stage_b)
            nominal += sum(item["nominal_elapsed"] for item in stage_b)
        elif stage == "C":
            recorded_a = _sample_group(env, model, book, blob, device, limit_s, reward_version, seed, 0, STAGE_A, True)
            recorded_b = _sample_group(env, model, book, blob, device, limit_s, reward_version, seed, 100, STAGE_B, True)
            if [item["label"] for item in recorded_a] != [item["label"] for item in stage_a]:
                raise RuntimeError("stage A replay labels differ from the original independent sample")
            if [item["label"] for item in recorded_b] != [item["label"] for item in stage_b]:
                raise RuntimeError("stage B replay labels differ from the original independent sample")
            physics += sum(item["recovery_physics"] for item in recorded_a + recorded_b)
            uniform = _branch(env, model, book, blob, device, recorded_a + recorded_b, TREE_EACH, "uniform", np.random.default_rng(seed + 300))
            entropy = _branch(env, model, book, blob, device, recorded_a + recorded_b, TREE_EACH, "entropy", np.random.default_rng(seed + 400))
            state["tree_done"] = True
            state["tree_uniform_success"] = sum(int(item["success"]) for item in uniform)
            state["tree_entropy_success"] = sum(int(item["success"]) for item in entropy)
        elif stage == "resample":
            resample = _sample_group(env, model, book, blob, device, limit_s, reward_version, seed, 500, STAGE_A, False)
            state["resample_done"] = True
            state["resample_success"] = sum(int(item["success"]) for item in resample)
            physics += sum(item["recovery_physics"] for item in resample)
            nominal += sum(item["nominal_elapsed"] for item in resample)
        else:
            raise RuntimeError(stage)


def _ic_log(episode_id, updated, source, state, stage_a, stage_b, uniform, entropy, resample, physics, nominal):
    def handoffs(items):
        return [item.get("handoff_t") for item in items]

    return {
        "id": episode_id,
        "updated": updated,
        "source": source,
        "stage_a_success": state["stage_a_success"],
        "stage_a_rollouts": len(stage_a),
        "stage_b_success": state["stage_b_success"],
        "stage_b_rollouts": len(stage_b),
        "tree_uniform_success": state["tree_uniform_success"],
        "tree_uniform_rollouts": len(uniform),
        "tree_entropy_success": state["tree_entropy_success"],
        "tree_entropy_rollouts": len(entropy),
        "resample_success": state["resample_success"],
        "found_success": bool(
            state["stage_a_success"] or state["stage_b_success"] or state["tree_uniform_success"] or state["tree_entropy_success"] or state["resample_success"]
        ),
        "independent_success": bool(state["stage_a_success"] or state["stage_b_success"] or state["resample_success"]),
        "recovery_physics": physics,
        "nominal_elapsed": nominal,
        "rewards": [item["return"] for item in stage_a + stage_b + resample],
        "handoff_t": handoffs(stage_a + stage_b + resample),
        "outcomes": [item["label"] for item in stage_a + stage_b],
        "tree_uniform": _jsonable(uniform),
        "tree_entropy": _jsonable(entropy),
        "immediate_handoff": sum(1 for item in stage_a + stage_b + resample if item.get("handoff_t") is not None and item["handoff_t"] <= 0.04),
    }


def apply_group_update(model, ref_model, members, cfg, opt, device):
    """Same clipped surrogate as train_grpo. Returns loss, kl, advantages, grad norm."""
    kept = loss_members(members)
    advantages, skipped = group_advantage(
        [member["return"] for member in kept],
        [member["truncated"] for member in kept],
    )
    opt.zero_grad(set_to_none=True)
    terms = []
    if not skipped:
        for member, advantage in zip(kept, advantages):
            if member["truncated"] or abs(float(advantage)) < 1e-8 or not member["records"]:
                continue
            logp = torch.stack([
                step_logprob(
                    model,
                    torch.tensor(rec["obs"], device=device),
                    torch.tensor(rec["prev"], device=device),
                    rec["token"],
                    rec["pre"],
                )
                for rec in member["records"]
            ]).mean()
            behavior = float(np.mean([rec["logp"] for rec in member["records"]]))
            ratio = torch.exp(torch.clamp(logp - behavior, -20.0, 20.0))
            unclipped = ratio * float(advantage)
            clipped = torch.clamp(ratio, 1.0 - float(cfg["clip"]), 1.0 + float(cfg["clip"])) * float(advantage)
            terms.append(-torch.minimum(unclipped, clipped))
    loss_value = None
    kl_value = None
    grad_norm = None
    if terms:
        surrogate = torch.stack(terms).mean()
        rec = next(member["records"][0] for member in kept if member["records"])
        kl = categorical_kl(
            model, ref_model,
            torch.tensor(rec["obs"], device=device),
            torch.tensor(rec["prev"], device=device),
        )
        loss = surrogate + float(cfg["kl_coef"]) * kl
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        loss_value = float(loss.detach())
        kl_value = float(kl.detach())
    finite = loss_value is None or np.isfinite(loss_value)
    if kl_value is not None and not np.isfinite(kl_value):
        finite = False
    return {
        "skipped": bool(skipped or not terms),
        "loss": loss_value,
        "kl": kl_value,
        "grad_norm": grad_norm,
        "advantages": [float(value) for value in advantages],
        "finite": finite,
        "group": len(kept),
        "success": sum(int(member["label"] == "SUCCESS_12S") for member in kept),
    }


def train_adaptive(cfg, checkpoint, limit_ics=None):
    from training.recovery_episode import make_training_env

    checkpoint = str(checkpoint)
    if Path(checkpoint).resolve() != SFT_CHECKPOINT.resolve():
        raise RuntimeError(f"adaptive GRPO must start from {SFT_CHECKPOINT}")
    rows = balanced_schedule(load_train_snapshots())
    if limit_ics is not None:
        rows = rows[: int(limit_ics)]
    out = ROOT / cfg["grpo_output"]
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    book = load_codebook(ROOT / cfg["codebook"])
    device = torch.device(cfg.get("device") or "cpu")
    before = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, saved_book = load_trained_k48(Path(checkpoint), device)
    if not np.array_equal(saved_book, book):
        raise RuntimeError("SFT checkpoint codebook differs from the configured codebook")
    loaded = model.logits.weight.detach().cpu()
    if not torch.equal(loaded, before["model"]["logits.weight"]):
        raise RuntimeError("adaptive GRPO did not load the SFT weights")
    model.train()
    ref_model, _ref_book = load_trained_k48(Path(checkpoint), device)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    env = make_training_env(device, model=model)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["grpo_lr"]))
    limit_s = float(cfg.get("recovery_limit_s") or SELECTED_LIMIT_S)
    reward_version = str(cfg.get("reward_version") or "v2")
    seated_ref = None
    logs = []
    updates = 0
    for index, row in enumerate(rows):
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if "ref" not in blob and seated_ref is None:
            from data_generation.strategies.recatch_oracle import _setup

            _sim, _base, seated_ref, _target, _code = _setup()
        blob = _prepare_blob(blob, seated_ref)
        print("ADAPTIVE_IC", row["id"], row["primary_category"], flush=True)
        ic_log, group = explore_ic(
            env, model, book, blob, device, limit_s, reward_version,
            int(cfg["seed"]) + 1009 * (index + 1), row["id"],
        )
        ic_log["primary_category"] = row["primary_category"]
        ic_log["categories"] = row["categories"]
        stats = None
        if group is not None:
            stats = apply_group_update(model, ref_model, group, cfg, opt, device)
            if not stats["finite"]:
                raise RuntimeError(f"non-finite GRPO update on {row['id']}")
            updates += 0 if stats["skipped"] else 1
        ic_log["update_stats"] = stats
        logs.append(ic_log)
        with (out / "ic_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(ic_log)) + "\n")
        print("ADAPTIVE_DONE", row["id"], "update", ic_log["updated"], "source", ic_log["source"], "a", ic_log["stage_a_success"], "b", ic_log["stage_b_success"], flush=True)
    summary = {
        "ics": len(logs),
        "updates": updates,
        "independent_success_ics": sum(int(row["independent_success"]) for row in logs),
        "skipped": sum(int(not row["updated"]) for row in logs),
        "tree_excluded": sum(int(row.get("tree_success_excluded") or False) for row in logs),
        "sampling": sampling_frequency(rows),
        "checkpoint": checkpoint,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "proto": book,
            "res_cap": CAP,
            "handoff": HANDOFF,
            "reward_version": reward_version,
            "init_checkpoint": checkpoint,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "updates": updates,
        },
        out / "checkpoint_adaptive.pt",
    )
    print("ADAPTIVE_SUMMARY", json.dumps(summary), flush=True)
    return summary
