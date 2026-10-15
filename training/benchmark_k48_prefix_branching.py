"""Exploration benchmark: independent GRPO-style rollouts versus prefix branching.

Does not train and does not change the policy, reward, codebook, or SUCCESS_12S.
All conditions sample the u783 policy. Suffixes are stochastic, not argmax.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import tracemalloc
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.autonomous_episode import make_training_env
from training.history_aware_critic_ablation import restore_capture, take_capture
import training.natural_loss_recatch_transformer_v3 as v3
from training.horizon_outcome import _contacts, finish_recovery, recovery_failed
from training.pre_grpo_sanity import SIGMA
from training.privileged_ball_impact_search import step_legal
from training.recovery_success import CONTROL_PERIOD_S
from training.unified_k47 import decode_action, load_codebook
from training.unified_k48 import HANDOFF, load_trained_k48

OUT = ROOT / "results/diagnostics/k48_prefix_branching"
LIMIT = 4.0
CAP = 200
CODEBOOK = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
DT = float(CONTROL_PERIOD_S)


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def take_snapshot(env):
    """Python controller capture plus a full MjData copy.

    ``restore_capture`` restores qpos, qvel, ctrl, time, and the controller
    objects, then calls ``mj_forward``. That forward pass does not reproduce
    the pre-forward ``cvel`` or contact-force state, and the resulting
    observation differed by about 0.04. Copying ``MjData`` back after
    ``restore_capture`` returns the observation to equality.
    """
    data = mujoco.MjData(env.sim.model)
    mujoco.mj_copyData(data, env.sim.model, env.sim.data)
    return {"capture": take_capture(env), "data": data}


def restore_snapshot(env, snap):
    restore_capture(env, snap["capture"])
    mujoco.mj_copyData(env.sim.data, env.sim.model, snap["data"])


def token_entropy(probs):
    """Categorical entropy of the 48-way token distribution, in nats.

    The residual is a fixed diagonal Gaussian, so its differential entropy does
    not depend on the state and is not included.
    """
    clipped = np.clip(np.asarray(probs, np.float64), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return 0.0
    clipped = clipped / total
    positive = clipped[clipped > 0.0]
    return float(-(positive * np.log(positive)).sum())


def sample_action(model, window_obs, window_prev, rng):
    with torch.no_grad():
        logits, mean, _act = model(window_obs, window_prev)
    probs = torch.softmax(logits[-1].float(), dim=-1).cpu().numpy()
    probs = np.clip(probs, 0.0, None)
    probs = probs / max(float(probs.sum()), 1e-12)
    token = int(rng.choice(len(probs), p=probs))
    pre = mean[-1].detach().float().cpu().numpy() + rng.normal(0.0, SIGMA, size=7).astype(np.float32)
    return token, pre.astype(np.float32), probs, token_entropy(probs)


def _windows(env, device):
    obs = np.asarray(env.observation(), np.float32)
    env._obs_seq.append(obs)
    env._prev_seq.append(env._prev.copy())
    window_obs = torch.tensor(np.stack(env._obs_seq[-160:]), device=device)
    window_prev = torch.tensor(np.stack(env._prev_seq[-160:]), device=device)
    return obs, window_obs, window_prev


def _nominal_steps(outcome, physics_dt):
    """Count MuJoCo steps. Nominal control advances one ``opt.timestep`` per call."""
    if outcome.get("handoff") is None:
        return 0
    elapsed = float(outcome.get("nominal_elapsed") or 0.0)
    counted = int(round(elapsed / float(physics_dt)))
    if abs(elapsed - counted * float(physics_dt)) > 1e-6:
        raise RuntimeError(f"nominal elapsed {elapsed} is not an integer number of physics steps")
    return counted


def _score_view(outcome, recovery_physics, physics_dt):
    return {
        "reason": outcome.get("reason"),
        "success": bool(outcome.get("success")),
        "handoff": outcome.get("handoff"),
        "recovery_elapsed": None if outcome.get("recovery_elapsed") is None else float(outcome["recovery_elapsed"]),
        "recovery_physics": int(recovery_physics),
        "nominal_steps": _nominal_steps(outcome, physics_dt),
    }


def rollout(env, model, book, blob, rng, device, capture=None, lost_low=0, t0=None, keep_captures=False):
    """One stochastic K48 rollout. Matches train_grpo sampling, including noise before HANDOFF."""
    if capture is None:
        restore_capture(env, blob["capture"])
        lost_low = 0
        t0 = float(env.sim.data.time)
    else:
        restore_snapshot(env, capture)
        if t0 is None:
            raise RuntimeError("a restored rollout needs the original recovery start time")
    how = None
    physics = 0
    physics_dt = float(env.sim.model.opt.timestep)
    rows = []
    captures = {}
    for _ in range(CAP):
        if float(env.sim.data.time) - t0 >= LIMIT - 1e-9:
            how = "forced"
            break
        if keep_captures:
            captures[len(rows)] = take_snapshot(env)
        _obs, window_obs, window_prev = _windows(env, device)
        token, pre, probs, entropy = sample_action(model, window_obs, window_prev, rng)
        n_l, n_r, z = _contacts(env.sim)
        row = {
            "step": len(rows),
            "t": float(env.sim.data.time) - t0,
            "token": int(token),
            "pre": np.asarray(pre, np.float32),
            "entropy": float(entropy),
            "p47": float(probs[HANDOFF]),
            "lost_low": int(lost_low),
            "nL": int(n_l),
            "nR": int(n_r),
            "z": float(z),
            "obj": np.asarray(env.sim.data.xpos[env.sim.ids.object_body], np.float32).copy(),
            "hand": np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], np.float32).copy(),
            "hist_len": int(len(env._obs_seq)),
        }
        rows.append(row)
        if token == HANDOFF:
            how = "early"
            break
        if int(token) < 0 or int(token) >= HANDOFF:
            raise RuntimeError(f"illegal motion token {token}")
        _residual, action = decode_action(book, token, pre)
        before = float(env.sim.data.time)
        executed = np.asarray(step_legal(env.sim, action), np.float32)
        advanced = float(env.sim.data.time) - before
        expected = v3.N_SUB * physics_dt
        if abs(advanced - expected) > 1e-8:
            raise RuntimeError(f"recovery step advanced {advanced}, expected {expected}")
        env._prev = executed
        row["residual"] = np.asarray(_residual, np.float32)
        row["grip"] = float(executed[6])
        physics += int(v3.N_SUB)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            from training.table_drop import terminal_reason
            outcome = {
                "reason": terminal_reason(env.sim),
                "success": False,
                "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "nominal_elapsed": 0.0,
                "lift_completed": False,
                "q_nominal": 0.0,
            }
            return _score_view(outcome, physics, physics_dt), rows, captures, t0
    if how is None:
        how = "forced"
    outcome = finish_recovery(env, blob["ref"], t0, how)
    return _score_view(outcome, physics, physics_dt), rows, captures, t0


def select_points(rows, rng, count, strategy):
    """Choose branch decisions without reading the outcome label."""
    eligible = list(range(len(rows)))
    if not eligible or count <= 0:
        return []
    take = min(int(count), len(eligible))
    if strategy == "uniform":
        picked = rng.choice(np.asarray(eligible, np.int64), size=take, replace=False)
        return sorted(int(value) for value in picked)
    if strategy == "entropy":
        order = sorted(eligible, key=lambda index: (-float(rows[index]["entropy"]), index))
        return order[:take]
    raise RuntimeError(f"unknown branch strategy {strategy}")


def leaf_record(condition, episode_id, repeat, seed, leaf_id, scored, rows, extra):
    residuals = [row["residual"] for row in rows if "residual" in row]
    obj = np.stack([row["obj"] for row in rows]) if rows else np.zeros((0, 3), np.float32)
    stride = obj[::5]
    item = {
        "condition": condition,
        "id": episode_id,
        "repeat": int(repeat),
        "seed": int(seed),
        "leaf": leaf_id,
        "tokens": [int(row["token"]) for row in rows],
        "token_entropy_mean": None if not rows else float(np.mean([row["entropy"] for row in rows])),
        "residual_abs_mean": None if not residuals else float(np.mean(np.abs(np.stack(residuals)))),
        "obj_stride": [[round(float(value), 5) for value in point] for point in stride],
        "handoff_t": None if not rows or rows[-1]["token"] != HANDOFF else float(rows[-1]["t"]),
        **scored,
        **extra,
    }
    return item


def independent(env, model, book, blob, device, episode_id, repeat, seed, count, condition, ic_index):
    leaves = []
    for index in range(count):
        rng = np.random.default_rng(_stream(seed, condition, ic_index, repeat, index))
        started = time.perf_counter()
        scored, rows, _caps, _t0 = rollout(env, model, book, blob, rng, device, keep_captures=False)
        leaves.append(leaf_record(condition, episode_id, repeat, seed, f"g{index}", scored, rows, {
            "depth": 0,
            "branch_step": None,
            "prefix_physics": 0,
            "recovery_physics_new": scored["recovery_physics"],
            "nominal_steps_new": scored["nominal_steps"],
            "restores": 1,
            "strategy": None,
            "branch_entropy": None,
            "wall_s": time.perf_counter() - started,
            "hist_at_end": None if not rows else int(rows[-1]["hist_len"]),
        }))
    return leaves


def _stream(seed, condition, ic_index, repeat, index):
    code = {"A": 1, "B": 2, "C_uniform": 3, "C_entropy": 4}[condition]
    return int(seed) + 100003 * int(repeat) + 10007 * int(ic_index) + 97 * int(index) + 1000003 * code


def prefix_tree(env, model, book, blob, device, episode_id, repeat, seed, ic_index, trunks, suffixes, strategy):
    """Four trunks by default, each with extra stochastic suffixes. Depth is one.

    Shared prefix physics is stored on the trunk only. Each suffix adds only the
    physics steps simulated after the restored branch point, plus its own nominal run.
    """
    leaves = []
    condition = f"C_{strategy}"
    for trunk in range(trunks):
        rng = np.random.default_rng(_stream(seed, condition, ic_index, repeat, trunk))
        started = time.perf_counter()
        scored, rows, captures, t0 = rollout(env, model, book, blob, rng, device, keep_captures=True)
        leaves.append(leaf_record(condition, episode_id, repeat, seed, f"t{trunk}", scored, rows, {
            "depth": 0,
            "branch_step": None,
            "prefix_physics": 0,
            "recovery_physics_new": scored["recovery_physics"],
            "nominal_steps_new": scored["nominal_steps"],
            "restores": 1,
            "strategy": strategy,
            "branch_entropy": None,
            "wall_s": time.perf_counter() - started,
            "hist_at_end": None if not rows else int(rows[-1]["hist_len"]),
        }))
        selector = np.random.default_rng(_stream(seed, condition, ic_index, repeat, 1000 + trunk))
        points = select_points(rows, selector, suffixes, strategy)
        for nth, step in enumerate(points):
            suffix_rng = np.random.default_rng(_stream(seed, condition, ic_index, repeat, 5000 + 100 * trunk + nth))
            started = time.perf_counter()
            suffix_score, suffix_rows, _caps, _t0 = rollout(
                env, model, book, blob, suffix_rng, device,
                capture=captures[step], lost_low=int(rows[step]["lost_low"]), t0=t0, keep_captures=False,
            )
            if suffix_rows and int(suffix_rows[0]["hist_len"]) != int(rows[step]["hist_len"]):
                raise RuntimeError("suffix causal history does not match the trunk prefix")
            leaves.append(leaf_record(condition, episode_id, repeat, seed, f"t{trunk}-s{nth}", suffix_score, suffix_rows, {
                "depth": 1,
                "branch_step": int(step),
                "branch_t": float(rows[step]["t"]),
                "prefix_physics": int(step) * int(v3.N_SUB),
                "recovery_physics_new": suffix_score["recovery_physics"],
                "nominal_steps_new": suffix_score["nominal_steps"],
                "restores": 1,
                "strategy": strategy,
                "branch_entropy": float(rows[step]["entropy"]),
                "trunk_token": int(rows[step]["token"]),
                "suffix_first_token": None if not suffix_rows else int(suffix_rows[0]["token"]),
                "wall_s": time.perf_counter() - started,
                "hist_at_end": None if not suffix_rows else int(suffix_rows[-1]["hist_len"]),
            }))
        del captures
    return leaves


def restore_probe(env, model, book, blob, device, seed):
    """Check physics, controller, contacts, observation, and one-step continuation."""
    rng = np.random.default_rng(int(seed) + 17)
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    saved = None
    target = None
    for step in range(8):
        if float(env.sim.data.time) - t0 >= LIMIT - 1e-9:
            break
        capture = take_snapshot(env)
        obs = np.asarray(env.observation(), np.float32).copy()
        n_l, n_r, _z = _contacts(env.sim)
        phase = str(env.sim.fsm.phase)
        qvel = np.asarray(env.sim.data.qvel, np.float64).copy()
        ctrl = np.asarray(env.sim.data.ctrl, np.float64).copy()
        clock = float(env.sim.data.time)
        hist = len(env._obs_seq)
        if step == min(3, step) and saved is None and step == 3:
            saved = {
                "capture": capture,
                "obs": obs,
                "nL": int(n_l),
                "nR": int(n_r),
                "phase": phase,
                "qvel": qvel,
                "ctrl": ctrl,
                "time": clock,
                "hist": hist,
                "lost_low": int(lost_low),
                "rng": pickle.loads(pickle.dumps(rng.bit_generator.state)),
            }
            target = step
        _obs, window_obs, window_prev = _windows(env, device)
        token, pre, _probs, _entropy = sample_action(model, window_obs, window_prev, rng)
        if token == HANDOFF:
            break
        _residual, action = decode_action(book, token, pre)
        executed = np.asarray(step_legal(env.sim, action), np.float32)
        env._prev = executed
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            break
    if saved is None:
        raise RuntimeError("restore probe did not reach a branchable step")
    restore_snapshot(env, saved["capture"])
    obs2 = np.asarray(env.observation(), np.float32)
    n_l, n_r, _z = _contacts(env.sim)
    report = {
        "step": int(target),
        "obs_max_abs": float(np.max(np.abs(obs2 - saved["obs"]))),
        "qvel_max_abs": float(np.max(np.abs(np.asarray(env.sim.data.qvel, np.float64) - saved["qvel"]))),
        "ctrl_max_abs": float(np.max(np.abs(np.asarray(env.sim.data.ctrl, np.float64) - saved["ctrl"]))),
        "time_abs": abs(float(env.sim.data.time) - saved["time"]),
        "phase_match": str(env.sim.fsm.phase) == saved["phase"],
        "contact_match": (int(n_l), int(n_r)) == (saved["nL"], saved["nR"]),
        "hist_match": len(env._obs_seq) == saved["hist"],
        "lost_low_stored": saved["lost_low"],
    }
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    rng_a.bit_generator.state = pickle.loads(pickle.dumps(saved["rng"]))
    rng_b.bit_generator.state = pickle.loads(pickle.dumps(saved["rng"]))
    _obs, window_obs, window_prev = _windows(env, device)
    token_a, pre_a, _probs, _entropy = sample_action(model, window_obs, window_prev, rng_a)
    qpos_a = None
    if token_a != HANDOFF:
        _residual, action = decode_action(book, token_a, pre_a)
        step_legal(env.sim, action)
        qpos_a = np.asarray(env.sim.data.qpos, np.float64).copy()
    restore_snapshot(env, saved["capture"])
    _obs, window_obs, window_prev = _windows(env, device)
    token_b, pre_b, _probs, _entropy = sample_action(model, window_obs, window_prev, rng_b)
    qpos_b = None
    if token_b != HANDOFF:
        _residual, action = decode_action(book, token_b, pre_b)
        step_legal(env.sim, action)
        qpos_b = np.asarray(env.sim.data.qpos, np.float64).copy()
    report["continuation_token_match"] = int(token_a) == int(token_b)
    report["continuation_pre_max_abs"] = float(np.max(np.abs(pre_a - pre_b)))
    report["continuation_qpos_max_abs"] = None if qpos_a is None or qpos_b is None else float(np.max(np.abs(qpos_a - qpos_b)))
    report["passed"] = bool(
        report["obs_max_abs"] <= 1e-6
        and report["qvel_max_abs"] <= 1e-6
        and report["ctrl_max_abs"] <= 1e-6
        and report["time_abs"] <= 1e-9
        and report["phase_match"]
        and report["contact_match"]
        and report["hist_match"]
        and report["continuation_token_match"]
        and report["continuation_pre_max_abs"] <= 1e-6
        and (report["continuation_qpos_max_abs"] is None or report["continuation_qpos_max_abs"] <= 1e-6)
    )
    return report


def regression_ids():
    path = ROOT / "results/diagnostics/k48_counterfactual_credit/raw/baseline.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    ordered = []
    for row in rows:
        if row["policy"] == "u174" and row["id"] not in ordered:
            ordered.append(row["id"])
    if len(ordered) != 18:
        raise RuntimeError(f"expected 18 confirmed regression ICs, found {len(ordered)}")
    return ordered


def write_audit():
    text = """# Implementation audit

This benchmark samples `results/grpo/v2/checkpoint_u0783.pt` only. It does not update the policy.

## Stochastic sampling

`training/unified_k48.py` `train_grpo` draws a token with `rng.choice` on the renormalized 48-way softmax, then draws `pre = mean + N(0, SIGMA)` with `SIGMA = 0.05` before it looks at the token. Token 47 is recorded and ends the recovery without `decode_action` or `step_legal`. A motion token is clipped by `decode_action` (`training/unified_k47.py`) and passed to `step_legal`. Forced handoff happens when the recovery clock reaches 4 s, before another observation is read, and is not a sampled token.

`training/action_representation_expansion.py` is not on this path.

The benchmark uses that same order: noise is consumed even when the token is HANDOFF. Evaluation here is stochastic. It is not the deterministic argmax used in the earlier regression report.

## Restoration

`take_capture` / `restore_capture` restore MuJoCo `mj_state`, `qpos`, `qvel`, `ctrl`, time, warmstart, actuators, equality, mocap, supports, nominal FSM, switch, observation-filter state, and the Transformer `_obs_seq` / `_prev_seq` / `_prev`. `restore_capture` then calls `mj_forward`. On a live branch point that forward pass changed `cvel` and `qacc`, and the 39-D observation differed by up to about 0.04. Branch snapshots therefore keep a full `MjData` and copy it back with `mj_copyData` after `restore_capture`. The smoke test requires the restored observation to match.

`lost_low`, the consecutive contact-loss counter inside `roll_policy` / `train_grpo`, is not inside the capture. Branch suffixes store it beside the snapshot and pass it back into `recovery_failed`. The smoke test checks observation, `qvel`, `ctrl`, time, FSM phase, finger contact counts, history length, and a repeated one-step sample after restoring both the snapshot and the RNG state.

Suffix RNGs are independent streams. Replaying the trunk RNG from the branch point would reproduce the trunk's own future actions, so it is not used as an alternative suffix. A suffix therefore does not read a sibling's future tokens or that sibling's RNG.

## Budget

Recovery cost counts `step_legal` calls. Nominal cost counts `nominal_step` calls inferred from `nominal_elapsed / control_period`, and the runner rejects a non-integer count. HANDOFF itself is not a physics step.

Independent G=4 and G=16 count every trajectory in full. Prefix branching counts each trunk once. A suffix counts only physics steps simulated after the restored state, plus that leaf's own nominal continuation. Prefix steps are not multiplied by the number of suffixes. Nominal cost is not waived for a shared recovery prefix.

G=4 is a smaller budget than G=16. Raw discovery counts are not a matched-compute comparison. Matched numbers truncate every condition to the same cumulative physics-step budget.

## Branch points

Uniform selection draws distinct decision indices from the realized trunk. Entropy selection ranks those indices by categorical token entropy in nats. Residual entropy is constant for the fixed Gaussian and is not used. Neither rule reads `SUCCESS_12S` or the failure label. Depth is one: trunks plus suffixes, no suffix of a suffix. u174 trajectories and expert demonstrations are not prefixes.
"""
    _write(OUT / "IMPLEMENTATION_AUDIT.md", text)


def _budget(leaves):
    recovery = sum(int(row["recovery_physics_new"]) for row in leaves)
    nominal = sum(int(row["nominal_steps_new"]) for row in leaves)
    return recovery, nominal, recovery + nominal


def _first_success_cost(leaves):
    recovery = nominal = 0
    for row in leaves:
        recovery += int(row["recovery_physics_new"])
        nominal += int(row["nominal_steps_new"])
        if row["success"]:
            return recovery + nominal, True
    return recovery + nominal, False


def summarize(leaves, label):
    by_ic = {}
    for row in leaves:
        by_ic.setdefault(row["id"], []).append(row)
    ic_success = 0
    success_leaves = 0
    reasons = {}
    handoff_times = []
    sequences = set()
    for episode_id, rows in by_ic.items():
        if any(row["success"] for row in rows):
            ic_success += 1
        for row in rows:
            success_leaves += int(bool(row["success"]))
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
            if row["success"]:
                sequences.add((episode_id, tuple(row["tokens"])))
                if row["handoff_t"] is not None:
                    handoff_times.append(float(row["handoff_t"]))
    recovery, nominal, total = _budget(leaves)
    return {
        "label": label,
        "ics": len(by_ic),
        "leaves": len(leaves),
        "ic_success": ic_success,
        "ic_success_fraction": ic_success / max(len(by_ic), 1),
        "success_leaves": success_leaves,
        "unique_success_sequences": len(sequences),
        "reasons": reasons,
        "handoff_t_success": handoff_times,
        "recovery_steps": recovery,
        "nominal_steps": nominal,
        "total_steps": total,
        "success_leaves_per_100k_steps": 1e5 * success_leaves / max(total, 1),
        "ic_success_per_100k_steps": 1e5 * ic_success / max(total, 1),
    }


def matched_ic_success(groups, mode):
    """Paired per-IC budget. Leaves are consumed in sample order inside each IC.

    ``mode='a'`` uses that IC's G=4 cost. ``mode='bc'`` uses the minimum of
    G=16 and the two tree costs on that same IC. A global cap would spend the
    whole budget on the first ICs and is not used.
    """
    by_condition = {}
    order = []
    for row in groups:
        by_condition.setdefault(row["condition"], {}).setdefault(row["id"], [])
        if row["id"] not in order:
            order.append(row["id"])
        by_condition[row["condition"]][row["id"]].append(row)
    names = ("A", "B", "C_uniform", "C_entropy")
    found = {name: 0 for name in names}
    spent = {name: 0 for name in names}

    def cost(row):
        return int(row["recovery_physics_new"]) + int(row["nominal_steps_new"])

    for episode_id in order:
        totals = {
            name: sum(cost(row) for row in by_condition.get(name, {}).get(episode_id, []))
            for name in names
        }
        budget = totals["A"] if mode == "a" else min(totals["B"], totals["C_uniform"], totals["C_entropy"])
        for name in names:
            used = 0
            hit = False
            for row in by_condition.get(name, {}).get(episode_id, []):
                step_cost = cost(row)
                if used + step_cost > budget:
                    break
                used += step_cost
                hit = hit or bool(row["success"])
            spent[name] += used
            found[name] += int(hit)
    return found, spent


def write_smoke(leaves, probes, seconds, repeats, ids):
    summaries = []
    for name in ("A", "B", "C_uniform", "C_entropy"):
        summaries.append(summarize([row for row in leaves if row["condition"] == name], name))
    differ = 0
    pairs = 0
    reproduced = True
    by_key = {}
    for row in leaves:
        if row["condition"] != "A":
            continue
        by_key.setdefault(row["id"], []).append(tuple(row["tokens"]))
    for tokens in by_key.values():
        for index in range(len(tokens)):
            for other in range(index + 1, len(tokens)):
                pairs += 1
                differ += int(tokens[index] != tokens[other])
    lines = [
        "# Smoke test",
        "",
        f"ICs: {', '.join(ids)}",
        f"Repeats: {repeats}. Wall time: {seconds:.1f} s.",
        "",
        "## Restoration",
        "",
    ]
    for probe in probes:
        lines.append(
            f"- `{probe['id']}` step {probe['step']}: obs {probe['obs_max_abs']:.3g}, qvel {probe['qvel_max_abs']:.3g}, "
            f"ctrl {probe['ctrl_max_abs']:.3g}, time {probe['time_abs']:.3g}, phase {probe['phase_match']}, "
            f"contacts {probe['contact_match']}, history {probe['hist_match']}, "
            f"continuation token {probe['continuation_token_match']}, qpos {probe['continuation_qpos_max_abs']}, passed {probe['passed']}"
        )
    lines.extend([
        "",
        f"Independent G=4 token sequences that differ within an IC: {differ}/{pairs}.",
        "",
        "## Budgets on this smoke subset",
        "",
        "| condition | ICs with a success | success leaves | recovery steps | nominal steps | total steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['ic_success']}/{row['ics']} | {row['success_leaves']} | {row['recovery_steps']} | {row['nominal_steps']} | {row['total_steps']} |"
        )
    lines.extend([
        "",
        "These smoke counts are not a TRAIN-wide estimate.",
        "",
        f"Same-seed reproduction check recorded: {reproduced}.",
        "",
    ])
    _write(OUT / "SMOKE_TEST.md", "\n".join(lines) + "\n")
    return summaries


def write_comparison(leaves, stage, ids, seconds):
    names = ("A", "B", "C_uniform", "C_entropy")
    rows = [summarize([row for row in leaves if row["condition"] == name], name) for name in names]
    matched_a, spent_a = matched_ic_success(leaves, "a")
    matched_bc, spent_bc = matched_ic_success(leaves, "bc")
    lines = [
        f"# Exploration comparison ({stage})",
        "",
        f"ICs ({len(ids)}): {', '.join(ids)}",
        f"Wall time: {seconds:.1f} s. One repeat unless noted in the raw records.",
        "",
        "A is G=4 independent samples. B is G=16 independent samples. C is 4 stochastic trunks plus up to 3 stochastic suffixes per trunk. Shared prefix physics is counted once. Every leaf that reaches HANDOFF pays its own nominal steps.",
        "",
        "## Raw totals",
        "",
        "| condition | IC successes | success leaves | unique success token sequences | recovery | nominal | total | success leaves / 100k steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['ic_success']}/{row['ics']} | {row['success_leaves']} | {row['unique_success_sequences']} | "
            f"{row['recovery_steps']} | {row['nominal_steps']} | {row['total_steps']} | {row['success_leaves_per_100k_steps']:.2f} |"
        )
    lines.extend([
        "",
        "Raw IC counts for A versus B or C are not compute-matched, because the budgets differ.",
        "",
        "## Per IC, truncated to that IC's G=4 cost",
        "",
        "Inside each IC, leaves are kept in sample order until adding the next leaf would exceed that IC's A cost. IC successes are then summed. This does not spend the global budget on the first ICs only.",
        "",
    ])
    for name in names:
        lines.append(f"- {name}: {matched_a.get(name, 0)} ICs with a success, steps used {spent_a.get(name, 0)}")
    lines.extend([
        "",
        "## Per IC, truncated to min(G=16, uniform tree, entropy tree) on that IC",
        "",
    ])
    for name in names:
        lines.append(f"- {name}: {matched_bc.get(name, 0)} ICs with a success, steps used {spent_bc.get(name, 0)}")
    lines.extend(["", "## Failure reasons", ""])
    for row in rows:
        lines.append(f"- {row['label']}: {row['reasons']}")
    lines.append("")
    _write(OUT / "EXPLORATION_COMPARISON.md", "\n".join(lines) + "\n")
    return rows, matched_a, matched_bc


def write_branch_analysis(leaves):
    suffixes = [row for row in leaves if row["condition"].startswith("C_") and row["depth"] == 1]
    lines = [
        "# Branch analysis",
        "",
        "Token entropy is categorical entropy in nats. It does not measure residual uncertainty.",
        "",
        f"Suffix leaves: {len(suffixes)}.",
        "",
        "| condition | branch step | branch t | trunk token | suffix first token | entropy | suffix reason | suffix success | prefix physics | new recovery | nominal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    productive = []
    for row in suffixes:
        lines.append(
            f"| {row['condition']} | {row['branch_step']} | {None if row.get('branch_t') is None else round(float(row['branch_t']), 3)} | "
            f"{row.get('trunk_token')} | {row.get('suffix_first_token')} | {None if row.get('branch_entropy') is None else round(float(row['branch_entropy']), 3)} | "
            f"{row['reason']} | {row['success']} | {row['prefix_physics']} | {row['recovery_physics_new']} | {row['nominal_steps_new']} |"
        )
        if row["success"]:
            productive.append(row)
    lines.extend(["", f"Successful suffixes: {len(productive)}.", ""])
    # Original trunk failed but a suffix succeeded.
    rescued = []
    trunks = {(row["id"], row["repeat"], row["leaf"].split("-")[0], row["condition"]): row
              for row in leaves if row["condition"].startswith("C_") and row["depth"] == 0}
    for row in productive:
        key = (row["id"], row["repeat"], row["leaf"].split("-")[0], row["condition"])
        trunk = trunks.get(key)
        if trunk is not None and not trunk["success"]:
            rescued.append((row["id"], row["condition"], row["branch_step"], trunk["reason"], row["reason"]))
    lines.append(f"Suffix successes whose trunk leaf failed: {len(rescued)}.")
    for item in rescued:
        lines.append(f"- `{item[0]}` {item[1]} step {item[2]} trunk {item[3]} suffix {item[4]}")
    lines.append("")
    _write(OUT / "BRANCH_ANALYSIS.md", "\n".join(lines) + "\n")
    return rescued


def write_conclusions(stage, smoke_s, diag_s, full_note, comparison_note):
    text = f"""# Conclusions

Stage completed: {stage}. Smoke wall time {smoke_s:.1f} s. Diagnostic wall time {diag_s}.

{full_note}

{comparison_note}

## Answers

1. G=16 versus G=4 is reported as a raw count and again after truncating both to the same physics-step budget. A higher raw G=16 count is not evidence that independent sampling is more efficient.
2. Prefix branching versus independent sampling is judged at matched total physics steps, with shared prefixes counted once and nominal continuation counted per leaf.
3. Productive branch steps are the successful suffix rows in `BRANCH_ANALYSIS.md`. One seed and a depth-1 tree do not identify a general best depth.
4. A rescued prefix means the trunk leaf failed and a stochastic suffix from a restored state succeeded. That is not a claim that the original u783 argmax policy succeeds, and it is not a claim that the trunk action was the only cause.
5. Unique token sequences are counted. Object-position strides are stored in `raw/` for descriptive comparison. Different tokens are not labeled as different recovery mechanisms.
6. Whether exploration coverage limits GRPO is not identified by this benchmark. The training update uses G=4 on-policy samples and a sequence-level advantage. Finding extra stochastic successes at G=16 or in a tree shows that those samples exist under u783. It does not show that training would have kept them, or that the deterministic regression was caused by too little exploration.
7. There is not enough evidence here to implement tree-based policy optimization. This run does not train, does not change the advantage, and does not use a second seed large enough for a significance claim.

VALID initial conditions were not used to choose branch rules or budgets.
"""
    _write(OUT / "CONCLUSIONS.md", text)


def _leaf_cost(row):
    return int(row["recovery_physics_new"]) + int(row["nominal_steps_new"])


def _grouped(leaves, ids):
    grouped = {episode_id: {"B": [], "C_uniform": [], "C_entropy": []} for episode_id in ids}
    for row in leaves:
        if row["condition"] in grouped.get(row["id"], {}):
            grouped[row["id"]][row["condition"]].append(row)
    return grouped


def _first_success(rows):
    used = 0
    for row in rows:
        used += _leaf_cost(row)
        if row["success"]:
            return used, False
    return used, True


def _coverage_curve(rows, ids):
    by_id = {episode_id: [] for episode_id in ids}
    for row in rows:
        if row["id"] in by_id:
            by_id[row["id"]].append(row)
    steps = 0
    covered = set()
    successes = 0
    points = []
    for episode_id in ids:
        for row in by_id[episode_id]:
            steps += _leaf_cost(row)
            successes += int(bool(row["success"]))
            if row["success"]:
                covered.add(episode_id)
            points.append({"steps": steps, "ics": len(covered), "success_leaves": successes})
    return points


def _curve_at(points, budget):
    covered = 0
    successes = 0
    for point in points:
        if point["steps"] > budget:
            break
        covered = point["ics"]
        successes = point["success_leaves"]
    return covered, successes


def write_seed_comparison(leaves, ids, seconds, seed):
    names = ("B", "C_uniform", "C_entropy")
    labels = {"B": "Independent G=16", "C_uniform": "Tree uniform", "C_entropy": "Tree token-entropy"}
    lines = [
        "# 87-IC exploration comparison",
        "",
        f"Policy `checkpoint_u0783.pt`, seed {seed}, one repeat. G=4 is not in this run.",
        f"TRAIN ICs: {len(ids)}. Wall time: {seconds:.1f} s.",
        "Shared prefix physics is counted once. Nominal continuation is counted for every leaf.",
        "An IC without a successful leaf is censored at that method's total physics steps on the IC.",
        "",
        "## Aggregates",
        "",
        "| method | IC coverage | success leaves | leaves | leaf ratio | physics steps | successes / 100k steps | leaf shortfall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    curves = {}
    summaries = {}
    for name in names:
        subset = [row for row in leaves if row["condition"] == name]
        summary = summarize(subset, name)
        shortfall = len(ids) * 16 - len(subset)
        summary["shortfall"] = shortfall
        summary["leaf_ratio"] = summary["success_leaves"] / max(summary["leaves"], 1)
        summaries[name] = summary
        curves[name] = _coverage_curve(subset, ids)
        lines.append(
            f"| {labels[name]} | {summary['ic_success']}/{summary['ics']} | {summary['success_leaves']} | {summary['leaves']} | "
            f"{summary['leaf_ratio']:.3f} | {summary['total_steps']} | {summary['success_leaves_per_100k_steps']:.2f} | {shortfall} |"
        )
    budgets = [summaries[name]["total_steps"] for name in names]
    common = min(budgets) if budgets else 0
    lines.extend([
        "",
        f"## Coverage versus compute at the common budget ({common} physics steps)",
        "",
        "Each curve consumes ICs in manifest order and leaves in sample order. The value is the IC coverage reached without exceeding the common budget.",
        "",
        "| method | ICs covered | success leaves |",
        "| --- | ---: | ---: |",
    ])
    for name in names:
        covered, successes = _curve_at(curves[name], common)
        lines.append(f"| {labels[name]} | {covered}/{len(ids)} | {successes} |")
    milestones = [0.1, 0.25, 0.5, 0.75, 1.0]
    lines.extend(["", "## Coverage at fractions of each method's own budget", ""])
    lines.append("| method | " + " | ".join(f"{int(100 * fraction)}%" for fraction in milestones) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in milestones) + " |")
    for name in names:
        total = max(summaries[name]["total_steps"], 1)
        cells = []
        for fraction in milestones:
            covered, _successes = _curve_at(curves[name], int(total * fraction))
            cells.append(f"{covered}/{len(ids)}")
        lines.append(f"| {labels[name]} | " + " | ".join(cells) + " |")
    grouped = _grouped(leaves, ids)
    only_tree = []
    only_g16 = []
    lines.extend([
        "",
        "## Per IC",
        "",
        "First-success cost is censored when the method never succeeds. `C` means censored.",
        "",
        "| IC | G=16 | uniform | entropy | G=16 first cost | uniform first cost | entropy first cost |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ])

    def mark(rows):
        cost, censored = _first_success(rows)
        text = f"{cost}C" if censored else str(cost)
        return bool(rows) and any(row["success"] for row in rows), text, len(rows)

    for episode_id in ids:
        item = grouped[episode_id]
        hit = {}
        costs = {}
        counts = {}
        for name in names:
            hit[name], costs[name], counts[name] = mark(item[name])
        if (hit["C_uniform"] or hit["C_entropy"]) and not hit["B"]:
            only_tree.append(episode_id)
        if hit["B"] and not hit["C_uniform"] and not hit["C_entropy"]:
            only_g16.append(episode_id)
        lines.append(
            f"| {episode_id} | {hit['B']} | {hit['C_uniform']} | {hit['C_entropy']} | "
            f"{costs['B']} | {costs['C_uniform']} | {costs['C_entropy']} |"
        )
    best_cover = max(names, key=lambda name: (summaries[name]["ic_success"], -summaries[name]["total_steps"]))
    best_ratio = max(names, key=lambda name: summaries[name]["leaf_ratio"])
    best_eff = max(names, key=lambda name: summaries[name]["success_leaves_per_100k_steps"])
    lines.extend([
        "",
        "## Answers",
        "",
        f"1. Highest IC coverage: {labels[best_cover]} ({summaries[best_cover]['ic_success']}/{len(ids)}).",
        f"2. Highest success-leaf ratio: {labels[best_ratio]} ({summaries[best_ratio]['leaf_ratio']:.3f}).",
        f"3. Most success leaves per 100,000 physics steps: {labels[best_eff]} ({summaries[best_eff]['success_leaves_per_100k_steps']:.2f}).",
        f"4. ICs recovered by a tree and missed by G=16: {len(only_tree)}.",
        "5. The 18-IC diagnostic is a different seed and a regressed subset. Compare orderings only after reading both reports; do not treat one seed as a significant test.",
        "",
        "## Disagreements",
        "",
    ])
    lines.append(f"Recovered by at least one tree and missed by G=16 ({len(only_tree)}):")
    lines.append("")
    if only_tree:
        for episode_id in only_tree:
            lines.append(f"- `{episode_id}`")
    else:
        lines.append("- none")
    lines.extend(["", f"Recovered by G=16 and missed by both trees ({len(only_g16)}):", ""])
    if only_g16:
        for episode_id in only_g16:
            lines.append(f"- `{episode_id}`")
    else:
        lines.append("- none")
    lines.append("")
    _write(OUT / "COMPARISON.md", "\n".join(lines) + "\n")
    (OUT / "raw" / "coverage.json").write_text(json.dumps({name: curves[name] for name in names}), encoding="utf-8")
    (OUT / "raw" / "shortfall.json").write_text(json.dumps({
        name: {"leaves": summaries[name]["leaves"], "target": len(ids) * 16, "shortfall": summaries[name]["shortfall"]}
        for name in names
    }, indent=2), encoding="utf-8")
    return summaries, only_tree, only_g16


def run_stage(env, model, book, device, ids, manifest, blobs, seed, repeats, trunks, suffixes, skip_g4=False):
    leaves = []
    probes = []
    started = time.perf_counter()
    for repeat in range(repeats):
        for ic_index, episode_id in enumerate(ids):
            blob = blobs[episode_id]
            if repeat == 0 and ic_index == 0:
                probe = restore_probe(env, model, book, blob, device, seed)
                probe["id"] = episode_id
                probes.append(probe)
                print("RESTORE", episode_id, probe["passed"], flush=True)
                if not probe["passed"]:
                    raise RuntimeError(f"restore probe failed: {probe}")
            independent_plan = (("B", 16),) if skip_g4 else (("A", 4), ("B", 16))
            for condition, count in independent_plan:
                batch = independent(env, model, book, blob, device, episode_id, repeat, seed, count, condition, ic_index)
                leaves.extend(batch)
                print("IND", episode_id, condition, sum(int(row["success"]) for row in batch), "/", len(batch), flush=True)
                if condition == "B" and skip_g4 and repeat == 0 and ic_index == 0:
                    rng = np.random.default_rng(_stream(seed, "B", ic_index, repeat, 0))
                    scored, rows, _caps, _t0 = rollout(env, model, book, blob, rng, device)
                    if [int(row["token"]) for row in rows] != batch[0]["tokens"] or scored["reason"] != batch[0]["reason"]:
                        raise RuntimeError("repeating the same seed did not reproduce the first G=16 rollout")
                    print("REPRODUCE", episode_id, scored["reason"], flush=True)
                if condition == "A" and repeat == 0 and ic_index == 0:
                    rng = np.random.default_rng(_stream(seed, "A", ic_index, repeat, 0))
                    scored, rows, _caps, _t0 = rollout(env, model, book, blob, rng, device)
                    if [int(row["token"]) for row in rows] != batch[0]["tokens"] or scored["reason"] != batch[0]["reason"]:
                        raise RuntimeError("repeating the same seed did not reproduce the first G=4 rollout")
                    print("REPRODUCE", episode_id, scored["reason"], flush=True)
            for strategy in ("uniform", "entropy"):
                batch = prefix_tree(env, model, book, blob, device, episode_id, repeat, seed, ic_index, trunks, suffixes, strategy)
                leaves.extend(batch)
                print("TREE", episode_id, strategy, sum(int(row["success"]) for row in batch), "/", len(batch), flush=True)
    return leaves, probes, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description="K48 prefix-branching exploration benchmark")
    parser.add_argument("--stage", choices=("smoke", "diagnostic", "full"), default="smoke")
    parser.add_argument("--checkpoint", default="results/grpo/v2/checkpoint_u0783.pt")
    parser.add_argument("--config", default="config/unified_k48_recovery_v3_reward_v2.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--trunks", type=int, default=4)
    parser.add_argument("--suffixes", type=int, default=3)
    parser.add_argument("--strategy", choices=("uniform", "entropy", "both"), default="both")
    parser.add_argument("--max-ics", type=int, default=0)
    parser.add_argument("--reports-only", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--skip-g4", action="store_true")
    args = parser.parse_args()
    global OUT
    if args.output:
        OUT = Path(args.output)
        if not OUT.is_absolute():
            OUT = ROOT / OUT
    if args.reports_only:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "raw" / f"{args.stage}_leaves.jsonl"
        leaves = _rows(path)
        meta = json.loads((OUT / "raw" / f"{args.stage}_meta.json").read_text(encoding="utf-8"))
        write_comparison(leaves, args.stage, meta["ids"], float(meta["seconds"]))
        write_branch_analysis(leaves)
        print("REPORTS", args.stage, "leaves", len(leaves), flush=True)
        return
    if args.strategy != "both":
        raise RuntimeError("this entry compares both branch strategies; run the file as published")
    if args.skip_g4 and OUT.resolve() == (ROOT / "results/diagnostics/k48_prefix_branching").resolve():
        raise RuntimeError("refusing to write a G=16-only run over the existing diagnostic directory")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    write_audit()
    device = torch.device("cpu")
    book = load_codebook(CODEBOOK)
    model, saved = load_trained_k48(ROOT / args.checkpoint, device)
    model.eval()
    if not np.array_equal(saved, book):
        raise RuntimeError("checkpoint codebook differs from the K48 motion book")
    manifest = {row["id"]: row for row in _rows(ROOT / "data/grpo_initial_conditions.jsonl")}
    regressed = [episode_id for episode_id in regression_ids() if episode_id in manifest]
    if args.stage == "smoke":
        ids = regressed[:4]
    elif args.stage == "diagnostic":
        ids = regressed
    else:
        ids = list(manifest)
    if args.max_ics:
        ids = ids[: args.max_ics]
    blobs = {}
    for episode_id in ids:
        with (ROOT / manifest[episode_id]["snapshot"]).open("rb") as handle:
            blobs[episode_id] = pickle.load(handle)
    env = make_training_env(device, model=model)
    tracemalloc.start()
    leaves, probes, seconds = run_stage(
        env, model, book, device, ids, manifest, blobs, args.seed, args.repeats, args.trunks, args.suffixes, skip_g4=args.skip_g4,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _dump(OUT / "raw" / f"{args.stage}_leaves.jsonl", leaves)
    (OUT / "raw" / f"{args.stage}_restore.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")
    (OUT / "raw" / f"{args.stage}_meta.json").write_text(json.dumps({
        "stage": args.stage,
        "ids": ids,
        "seed": args.seed,
        "repeats": args.repeats,
        "trunks": args.trunks,
        "suffixes": args.suffixes,
        "sigma": SIGMA,
        "skip_g4": bool(args.skip_g4),
        "seconds": seconds,
        "tracemalloc_peak_bytes": int(peak),
        "checkpoint": args.checkpoint,
    }, indent=2), encoding="utf-8")
    if args.skip_g4:
        summaries, only_tree, only_g16 = write_seed_comparison(leaves, ids, seconds, args.seed)
        print(
            "SUMMARY",
            {name: (summaries[name]["ic_success"], summaries[name]["success_leaves"], summaries[name]["total_steps"]) for name in summaries},
            "tree_only", len(only_tree),
            "g16_only", len(only_g16),
            flush=True,
        )
    elif args.stage == "smoke":
        write_smoke(leaves, probes, seconds, args.repeats, ids)
        per_ic = seconds / max(len(ids), 1)
        print(f"ESTIMATE diagnostic_18_s {per_ic * 18:.0f} full_87_s {per_ic * 87:.0f} peak_bytes {peak}", flush=True)
        write_comparison(leaves, args.stage, ids, seconds)
        write_branch_analysis(leaves)
    else:
        write_comparison(leaves, args.stage, ids, seconds)
        write_branch_analysis(leaves)
    if args.skip_g4:
        print("DONE", args.stage, "leaves", len(leaves), "seconds", round(seconds, 1), flush=True)
        return
    smoke_meta = OUT / "raw" / "smoke_meta.json"
    smoke_s = seconds if args.stage == "smoke" else -1.0
    if smoke_meta.exists() and args.stage != "smoke":
        smoke_s = float(json.loads(smoke_meta.read_text(encoding="utf-8"))["seconds"])
    per_ic = seconds / max(len(ids), 1)
    full_note = (
        f"Peak traced memory {peak} bytes. Seed {args.seed}, repeats {args.repeats}. "
        f"This stage took {seconds:.1f} s for {len(ids)} ICs ({per_ic:.1f} s/IC). "
        "A full 87-IC run at this rate is about "
        f"{per_ic * 87 / 3600:.1f} h per seed and was not launched from the smoke estimate."
    )
    write_conclusions(
        args.stage,
        smoke_s,
        f"{seconds:.1f} s" if args.stage == "diagnostic" else "not this process",
        full_note,
        "See EXPLORATION_COMPARISON.md for the raw and budget-truncated counts from this stage. "
        "The 18-IC diagnostic set is the regressed subset, not a TRAIN-wide rate. "
        "One seed is not a significance claim.",
    )
    print("DONE", args.stage, "leaves", len(leaves), "seconds", round(seconds, 1), flush=True)


if __name__ == "__main__":
    main()
