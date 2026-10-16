"""Adaptive GRPO with fixed 64-branch tree augmentation.

Stage A updates on 8 independent rollouts when one reaches SUCCESS_12S.
Stage B updates on all 16 when a later independent rollout succeeds. Otherwise
every trunk grows two uniform and two entropy suffixes, and the 80 stitched
trajectories take one group-relative update. Tree success does not resample
independent rollouts.

The surrogate is per sampled policy action. The trajectory advantage multiplies
every action in that trajectory, ratios are clipped per action, and trajectories
are weighted equally. Shared prefixes are counted in the log and are not
deduplicated. HANDOFF contributes only its token log-probability. Nominal
actions after handoff are not policy actions.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from envs.simulation.capture import take_snapshot
from training.prefix_rollout import rollout as tree_rollout
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.adaptive_hybrid import (
    BRANCHES_PER_METHOD,
    STAGE_A,
    STAGE_B,
    allocate_tree_branches,
    balanced_schedule,
    execute_allocated_branches,
    exploration_decision,
    load_train_snapshots,
    loss_members,
    prefix_multiplicity,
    sampling_frequency,
)
from training.grpo_math import SIGMA
from training.k47_codebook import CAP, categorical_kl, decode_action, group_advantage, load_codebook
from training.k48_config import ROOT, SELECTED_LIMIT_S
from training.reward_v2 import grpo_return
from training.train_grpo import step_logprob

SFT_CHECKPOINT = ROOT / "results/sft/checkpoint_epoch_0040.pt"


def adaptive_epoch_path(out, epoch):
    epoch = int(epoch)
    if epoch < 1:
        raise RuntimeError(f"adaptive epoch must be positive, got {epoch}")
    return Path(out) / f"checkpoint_epoch_{epoch:04d}.pt"


def adaptive_round_count(cfg, epochs):
    chosen = int(cfg.get("ic_epochs") or 1) if epochs is None else int(epochs)
    if chosen < 1:
        raise RuntimeError(f"adaptive rounds must be positive, got {chosen}")
    return chosen


def _stored_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def cpu_adam_state(opt):
    """Adam moments on CPU. param_groups keep the learning rate and betas."""
    saved = opt.state_dict()
    state = {}
    for index, slot in saved["state"].items():
        state[index] = {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else value
            for key, value in slot.items()
        }
        if "exp_avg" not in state[index] or "exp_avg_sq" not in state[index]:
            raise RuntimeError("Adam state is missing exp_avg or exp_avg_sq")
    return {"state": state, "param_groups": saved["param_groups"]}


def adaptive_resume_plan(checkpoint, out):
    """Fresh SFT start is epoch 0. A saved adaptive file continues after its epoch."""
    path = _stored_path(checkpoint)
    destination = Path(out).resolve()
    if path == SFT_CHECKPOINT.resolve():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"refusing to overwrite {destination}")
        return {"epoch": 0, "updates": 0, "init_checkpoint": SFT_CHECKPOINT, "weights": path, "optimizer": None}
    if not path.is_file():
        raise RuntimeError(f"missing adaptive checkpoint {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if "epoch" not in saved:
        raise RuntimeError(f"{path.name} has no epoch; expected checkpoint_epoch_XXXX.pt")
    epoch = int(saved["epoch"])
    if epoch < 1:
        raise RuntimeError(f"adaptive epoch must be positive, got {epoch}")
    if path.parent.resolve() != destination:
        raise RuntimeError(f"resume checkpoint must live in {destination}")
    optimizer = saved.get("optimizer")
    if not isinstance(optimizer, dict) or "state" not in optimizer or "param_groups" not in optimizer:
        raise RuntimeError(f"{path.name} has no Adam moments; refusing to resume")
    for slot in optimizer["state"].values():
        if "exp_avg" not in slot or "exp_avg_sq" not in slot:
            raise RuntimeError(f"{path.name} Adam state is missing exp_avg or exp_avg_sq")
    init = _stored_path(saved.get("init_checkpoint") or SFT_CHECKPOINT)
    if init != SFT_CHECKPOINT.resolve():
        raise RuntimeError(f"adaptive KL reference must stay {SFT_CHECKPOINT}")
    return {
        "epoch": epoch,
        "updates": int(saved.get("updates") or 0),
        "init_checkpoint": SFT_CHECKPOINT,
        "weights": path,
        "optimizer": optimizer,
    }


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
        from envs.outcomes.table_drop import restore_state
        from training.recovery_execution import restore_ballistic

        restore_ballistic(env.sim, blob)
        restore_state(env.sim, None)
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


def _suffix_records(rows):
    records = []
    for row in rows:
        if "logp" not in row or "obs" not in row or "prev" not in row:
            raise RuntimeError("tree suffix is missing a policy record")
        records.append({
            "obs": row["obs"],
            "prev": row["prev"],
            "token": int(row["token"]),
            "pre": np.asarray(row["pre"], np.float32),
            "logp": float(row["logp"]),
        })
    return records


def _tree_member(trunk, point, strategy, scored, rows, reward_version, limit_s, wall_s):
    """Stitch the unexecuted prefix to the newly sampled suffix.

    The branch capture is the pre-action state of ``point['step']``. Prefix
    actions are the trunk records before that step. Their physics is not run
    again. The suffix return is the full episode return from the original t0.
    """
    branch_step = int(point["step"])
    if branch_step > len(trunk["records"]):
        raise RuntimeError(f"branch step {branch_step} is outside the trunk record")
    prefix = trunk["records"][:branch_step]
    outcome = {
        "reason": scored["reason"],
        "success": bool(scored["success"]),
        "recovery_elapsed": scored.get("recovery_elapsed") or 0.0,
        "q_nominal": scored.get("q_nominal") or 0.0,
        "lift_completed": bool(scored.get("lift_completed")),
        "reward": scored.get("reward"),
        "handoff": scored.get("handoff"),
    }
    graded = grpo_return(outcome, limit_s, reward_version, scored.get("reward"))
    invalid = outcome["reason"] == "INVALID" or graded["return"] is None
    handoff_t = None
    if rows and int(rows[-1]["token"]) == HANDOFF:
        handoff_t = float(rows[-1]["t"])
    return {
        "origin": "tree",
        "strategy": strategy,
        "records": prefix + _suffix_records(rows),
        "return": None if invalid else float(graded["return"]),
        "truncated": invalid,
        "label": outcome["reason"],
        "success": bool(scored["success"]),
        "handoff_t": handoff_t,
        "recovery_physics": int(scored["recovery_physics"]),
        "nominal_elapsed": float(scored.get("nominal_elapsed") or 0.0),
        "reward_log": graded,
        "branch_step": branch_step,
        "branch_t": float(point["t"]),
        "branch_entropy": float(point["entropy"]),
        "prefix_source": int(point["traj"]),
        "prefix_len": branch_step,
        "trunk_token": int(point["token"]),
        "suffix_tokens": [int(row["token"]) for row in rows],
        "wall_s": float(wall_s),
    }


def _roll_one_branch(env, model, book, blob, device, trunks, point, strategy, rng, reward_version, limit_s):
    trunk = trunks[int(point["traj"])]
    capture = trunk["captures"][int(point["step"])]
    expected = None
    if int(point["step"]) < len(trunk["records"]):
        expected = np.asarray(trunk["records"][int(point["step"])]["obs"][-1], np.float32)
    suffix_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    started = time.perf_counter()
    scored, rows, _caps, _t0 = tree_rollout(
        env, model, book, blob, suffix_rng, device,
        capture=capture,
        lost_low=int(point["lost_low"]),
        t0=trunk["t0"],
        keep_captures=False,
        keep_records=True,
        expected_obs=expected,
    )
    return _tree_member(trunk, point, strategy, scored, rows, reward_version, limit_s, time.perf_counter() - started)


def _run_tree(env, model, book, blob, device, trunks, seed, reward_version, limit_s):
    uniform_points, entropy_points = allocate_tree_branches(
        trunks,
        np.random.default_rng(int(seed) + 300),
        np.random.default_rng(int(seed) + 400),
        BRANCHES_PER_METHOD,
    )
    uniform_rng = np.random.default_rng(int(seed) + 301)
    entropy_rng = np.random.default_rng(int(seed) + 401)

    def roll(strategy, rng):
        def rollout_one(point):
            return _roll_one_branch(
                env, model, book, blob, device, trunks, point, strategy, rng, reward_version, limit_s,
            )
        return rollout_one

    uniform = execute_allocated_branches(uniform_points, roll("uniform", uniform_rng))
    entropy = execute_allocated_branches(entropy_points, roll("entropy", entropy_rng))
    expected = len(trunks) * BRANCHES_PER_METHOD
    if len(uniform) != expected or len(entropy) != expected:
        raise RuntimeError(f"tree execution produced {len(uniform)} uniform and {len(entropy)} entropy trajectories")
    return uniform, entropy


def _sample_group(env, model, book, blob, device, limit_s, reward_version, seed, offset, count, keep_captures):
    group = []
    for index in range(count):
        rng = np.random.default_rng(int(seed) + 17 * (offset + index + 1))
        group.append(independent_rollout(
            env, model, book, blob, rng, device, limit_s, reward_version, keep_captures,
        ))
    return group


def _release_captures(items):
    for item in items:
        item["captures"] = {}


def explore_ic(env, model, book, blob, device, limit_s, reward_version, seed, episode_id):
    stage_a = _sample_group(env, model, book, blob, device, limit_s, reward_version, seed, 0, STAGE_A, True)
    state = {
        "stage_a_success": sum(int(item["success"]) for item in stage_a),
        "stage_b_success": 0,
        "stage_b_done": False,
        "tree_uniform_success": 0,
        "tree_entropy_success": 0,
        "tree_done": False,
    }
    return _continue(env, model, book, blob, device, limit_s, reward_version, seed, episode_id, stage_a, state)


def _continue(env, model, book, blob, device, limit_s, reward_version, seed, episode_id, stage_a, state):
    stage_b = []
    uniform = []
    entropy = []
    physics = sum(item["recovery_physics"] for item in stage_a)
    nominal = sum(item["nominal_elapsed"] for item in stage_a)
    tree_physics = 0
    while True:
        decision = exploration_decision(state)
        if decision["action"] == "update":
            if decision["group"] == "stage_a":
                chosen = stage_a
                _release_captures(stage_a)
            elif decision["group"] == "stage_ab":
                chosen = stage_a + stage_b
                _release_captures(chosen)
            elif decision["group"] == "stage_c":
                chosen = stage_a + stage_b + uniform + entropy
                expected = (STAGE_A + STAGE_B) * (1 + 2 * BRANCHES_PER_METHOD)
                if len(chosen) != expected:
                    raise RuntimeError(f"stage C group has {len(chosen)} trajectories, expected {expected}")
                _release_captures(stage_a + stage_b)
            else:
                raise RuntimeError(decision["group"])
            log = _ic_log(
                episode_id, True, decision["source"], decision["stage"], state,
                stage_a, stage_b, uniform, entropy, chosen, physics, tree_physics, nominal,
            )
            return log, loss_members(chosen)
        stage = decision["stage"]
        if stage == "B":
            stage_b = _sample_group(env, model, book, blob, device, limit_s, reward_version, seed, 100, STAGE_B, True)
            state["stage_b_done"] = True
            state["stage_b_success"] = sum(int(item["success"]) for item in stage_b)
            physics += sum(item["recovery_physics"] for item in stage_b)
            nominal += sum(item["nominal_elapsed"] for item in stage_b)
        elif stage == "C":
            trunks = stage_a + stage_b
            if len(trunks) != STAGE_A + STAGE_B:
                raise RuntimeError(f"tree stage expected {STAGE_A + STAGE_B} trunks, found {len(trunks)}")
            uniform, entropy = _run_tree(env, model, book, blob, device, trunks, seed, reward_version, limit_s)
            state["tree_done"] = True
            state["tree_uniform_success"] = sum(int(item["success"]) for item in uniform)
            state["tree_entropy_success"] = sum(int(item["success"]) for item in entropy)
            tree_physics = sum(item["recovery_physics"] for item in uniform + entropy)
            nominal += sum(item["nominal_elapsed"] for item in uniform + entropy)
        else:
            raise RuntimeError(stage)


def _return_stats(items):
    values = [float(item["return"]) for item in items if item.get("return") is not None and not item.get("truncated")]
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _ic_log(episode_id, updated, source, stage, state, stage_a, stage_b, uniform, entropy, group, physics, tree_physics, nominal):
    def handoffs(items):
        return [item.get("handoff_t") for item in items]

    multiplicity = prefix_multiplicity(group)
    returns = _return_stats(group)
    return {
        "id": episode_id,
        "updated": updated,
        "source": source,
        "stage": stage,
        "stage_a_success": state["stage_a_success"],
        "stage_a_rollouts": len(stage_a),
        "stage_b_success": state["stage_b_success"],
        "stage_b_rollouts": len(stage_b),
        "tree_uniform_success": state["tree_uniform_success"],
        "tree_uniform_rollouts": len(uniform),
        "tree_entropy_success": state["tree_entropy_success"],
        "tree_entropy_rollouts": len(entropy),
        "group_size": len(group),
        "success_count": sum(int(item.get("label") == "SUCCESS_12S") for item in group),
        "found_success": bool(
            state["stage_a_success"] or state["stage_b_success"] or state["tree_uniform_success"] or state["tree_entropy_success"]
        ),
        "independent_success": bool(state["stage_a_success"] or state["stage_b_success"]),
        "recovery_physics": physics,
        "tree_suffix_physics": tree_physics,
        "nominal_elapsed": nominal,
        "return_mean": returns["mean"],
        "return_std": returns["std"],
        "return_min": returns["min"],
        "return_max": returns["max"],
        "prefix_multiplicity": multiplicity,
        "rewards": [item["return"] for item in group],
        "labels": [item["label"] for item in group],
        "handoff_t": handoffs(group),
        "outcomes": [item["label"] for item in stage_a + stage_b],
        "tree_uniform": _jsonable(uniform),
        "tree_entropy": _jsonable(entropy),
        "immediate_handoff": sum(1 for item in group if item.get("handoff_t") is not None and item["handoff_t"] <= 0.04),
    }


def _action_surrogate(model, record, advantage, clip, device):
    """Clipped probability ratio for one sampled policy action.

    The trajectory advantage is shared by every action. HANDOFF has no residual
    term because ``step_logprob`` returns only the token log-probability.
    """
    logp = step_logprob(
        model,
        torch.tensor(record["obs"], device=device),
        torch.tensor(record["prev"], device=device),
        record["token"],
        record["pre"],
    )
    ratio = torch.exp(torch.clamp(logp - float(record["logp"]), -20.0, 20.0))
    unclipped = ratio * float(advantage)
    clipped = torch.clamp(ratio, 1.0 - float(clip), 1.0 + float(clip)) * float(advantage)
    return -torch.minimum(unclipped, clipped), ratio


def apply_group_update(model, ref_model, members, cfg, opt, device):
    """Equal-weight trajectory surrogate. Advantages use the whole group.

    Each trajectory's clipped action losses are averaged, then those trajectory
    losses are averaged. A shared prefix therefore contributes once per stitched
    trajectory. KL is the same equal-trajectory mean of per-action categorical KL.
    Zero return variance skips the optimizer without inventing an advantage.
    """
    kept = loss_members(members)
    advantages, skipped = group_advantage(
        [member["return"] for member in kept],
        [member["truncated"] for member in kept],
    )
    opt.zero_grad(set_to_none=True)
    trajectory_losses = []
    kl_terms = []
    clipped_actions = 0
    total_actions = 0
    if not skipped:
        clip = float(cfg["clip"])
        for member, advantage in zip(kept, advantages):
            if member["truncated"] or abs(float(advantage)) < 1e-8 or not member["records"]:
                continue
            action_losses = []
            for record in member["records"]:
                surrogate, ratio = _action_surrogate(model, record, advantage, clip, device)
                action_losses.append(surrogate)
                total_actions += 1
                outside = (ratio < 1.0 - clip) or (ratio > 1.0 + clip)
                clipped_actions += int(bool(outside.detach()))
                kl_terms.append(categorical_kl(
                    model, ref_model,
                    torch.tensor(record["obs"], device=device),
                    torch.tensor(record["prev"], device=device),
                ))
            trajectory_losses.append(torch.stack(action_losses).mean())
    loss_value = None
    kl_value = None
    grad_norm = None
    if trajectory_losses:
        surrogate = torch.stack(trajectory_losses).mean()
        kl = torch.stack(kl_terms).mean()
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
        "skipped": bool(skipped or not trajectory_losses),
        "loss": loss_value,
        "kl": kl_value,
        "clip_fraction": None if total_actions == 0 else float(clipped_actions) / float(total_actions),
        "grad_norm": grad_norm,
        "advantages": [float(value) for value in advantages],
        "advantage_max": None if not len(advantages) else float(np.max(advantages)),
        "advantage_min": None if not len(advantages) else float(np.min(advantages)),
        "finite": finite,
        "group": len(kept),
        "success": sum(int(member["label"] == "SUCCESS_12S") for member in kept),
    }


def _epoch_summary(logs, rows, epoch, updates, cumulative, checkpoint_path, init_checkpoint):
    by_category = {}
    for item in logs:
        slot = by_category.setdefault(item["primary_category"], {
            "ics": 0, "updates": 0, "stage_c": 0, "independent_success_ics": 0,
            "tree_success_ics": 0, "success_trajectories": 0,
        })
        slot["ics"] += 1
        slot["updates"] += int(item["optimizer_updated"])
        slot["stage_c"] += int(item["stage"] == "C")
        slot["independent_success_ics"] += int(item["independent_success"])
        slot["tree_success_ics"] += int(item["tree_uniform_success"] + item["tree_entropy_success"] > 0)
        slot["success_trajectories"] += int(item["success_count"])
    return {
        "epoch": int(epoch),
        "ics": len(logs),
        "updates": updates,
        "cumulative_updates": cumulative,
        "independent_success_ics": sum(int(row["independent_success"]) for row in logs),
        "stage_c_ics": sum(int(row["stage"] == "C") for row in logs),
        "skipped": sum(int(not row["optimizer_updated"]) for row in logs),
        "by_category": by_category,
        "sampling": sampling_frequency(rows),
        "checkpoint": str(checkpoint_path),
        "init_checkpoint": str(init_checkpoint),
    }


def train_adaptive(cfg, checkpoint, limit_ics=None, epochs=None):
    from training.recovery_episode import make_training_env

    rows = balanced_schedule(load_train_snapshots())
    if limit_ics is not None:
        rows = rows[: int(limit_ics)]
    rounds = adaptive_round_count(cfg, epochs)
    out = ROOT / cfg["grpo_output"]
    plan = adaptive_resume_plan(checkpoint, out)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(plan["epoch"] + 1, plan["epoch"] + rounds + 1):
        dest = adaptive_epoch_path(out, epoch)
        if dest.exists():
            raise RuntimeError(f"refusing to overwrite {dest}")
    book = load_codebook(ROOT / cfg["codebook"])
    device = torch.device(cfg.get("device") or "cpu")
    before = torch.load(plan["weights"], map_location="cpu", weights_only=False)
    model, saved_book = load_trained_k48(plan["weights"], device)
    if not np.array_equal(saved_book, book):
        raise RuntimeError("checkpoint codebook differs from the configured codebook")
    if plan["epoch"] == 0:
        loaded = model.logits.weight.detach().cpu()
        if not torch.equal(loaded, before["model"]["logits.weight"]):
            raise RuntimeError("adaptive GRPO did not load the SFT weights")
    model.train()
    ref_model, _ref_book = load_trained_k48(plan["init_checkpoint"], device)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    env = make_training_env(device, model=model)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["grpo_lr"]))
    if plan["optimizer"] is not None:
        opt.load_state_dict(plan["optimizer"])
        for slot in opt.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if not torch.isfinite(slot[key]).all():
                    raise RuntimeError(f"non-finite Adam moment {key}")
    limit_s = float(cfg.get("recovery_limit_s") or SELECTED_LIMIT_S)
    reward_version = str(cfg.get("reward_version") or "v2")
    seated_ref = None
    cumulative = int(plan["updates"])
    summary = None
    for epoch in range(plan["epoch"] + 1, plan["epoch"] + rounds + 1):
        logs = []
        updates = 0
        for index, row in enumerate(rows):
            with (ROOT / row["snapshot"]).open("rb") as handle:
                blob = pickle.load(handle)
            if "ref" not in blob and seated_ref is None:
                from data_generation.strategies.recatch_oracle import _setup

                _sim, _base, seated_ref, _target, _code = _setup()
            blob = _prepare_blob(blob, seated_ref)
            print("ADAPTIVE_IC", epoch, row["id"], row["primary_category"], flush=True)
            ic_log, group = explore_ic(
                env, model, book, blob, device, limit_s, reward_version,
                int(cfg["seed"]) + 1009 * (index + 1) + 10007 * (epoch - 1), row["id"],
            )
            ic_log["epoch"] = epoch
            ic_log["primary_category"] = row["primary_category"]
            ic_log["categories"] = row["categories"]
            stats = None
            if group is not None:
                stats = apply_group_update(model, ref_model, group, cfg, opt, device)
                if not stats["finite"]:
                    raise RuntimeError(f"non-finite GRPO update on {row['id']}")
                updates += 0 if stats["skipped"] else 1
            ic_log["update_stats"] = stats
            ic_log["optimizer_updated"] = bool(stats and not stats["skipped"])
            logs.append(ic_log)
            with (out / "ic_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(ic_log)) + "\n")
            print(
                "ADAPTIVE_DONE", epoch, row["id"], "stage", ic_log["stage"], "update", ic_log["optimizer_updated"],
                "group", ic_log["group_size"], "success", ic_log["success_count"],
                "a", ic_log["stage_a_success"], "b", ic_log["stage_b_success"],
                "tree", ic_log["tree_uniform_success"] + ic_log["tree_entropy_success"],
                flush=True,
            )
        cumulative += updates
        dest = adaptive_epoch_path(out, epoch)
        summary = _epoch_summary(logs, rows, epoch, updates, cumulative, dest, plan["init_checkpoint"])
        (out / f"summary_epoch_{epoch:04d}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if plan["epoch"] == 0:
            (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        torch.save(
            {
                "proto": book,
                "res_cap": CAP,
                "handoff": HANDOFF,
                "reward_version": reward_version,
                "init_checkpoint": str(plan["init_checkpoint"]),
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "updates": cumulative,
                "epoch": epoch,
                "optimizer": cpu_adam_state(opt),
            },
            dest,
        )
        print("ADAPTIVE_SUMMARY", json.dumps(summary), flush=True)
    return summary
