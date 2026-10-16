"""Diagnose why Horizontal Recatch rollouts miss SUCCESS_12S.

Does not change physics, the nominal controller, the codebook, reward_v2,
SUCCESS_12S, or handoff. Writes traces under
results/diagnostics/horizontal_recatch_sampling/.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from data_generation.strategies.build_open_overchase_recatch_dataset import _specs
from data_generation.strategies.recatch_oracle import (
    _action_from,
    _bottom,
    _class_force,
    _pad_band,
    _pose,
    _setup,
    contacts_now,
    staged_choose,
)
from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed
from envs.outcomes.table_drop import export_state, latched, restore_state
from envs.simulation.capture import restore_capture
from envs.simulation.scene_state import TABLE_TOP
from envs.step_legal import step_legal
from models.airborne_v3 import DT_POLICY
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.adaptive_hybrid import HORIZONTAL, SLIDING, VERTICAL, load_train_snapshots
from training.grpo_math import SIGMA
from training.k47_codebook import CAP, decode_action, load_codebook
from training.k48_config import ROOT
from training.recovery_episode import make_training_env
from training.recovery_execution import restore_ballistic

OUT = ROOT / "results/diagnostics/horizontal_recatch_sampling"
CKPT = ROOT / "results/sft/current/checkpoint_epoch_0040.pt"
LIMIT_S = 4.0
KS = (0, 1, 2, 3, 5, 10)


def _json(value):
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.round(value.astype(np.float64), 6).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(_json(payload), indent=2), encoding="utf-8")
    print("WROTE", path, flush=True)


def manifest_by_id():
    rows = {}
    for line in (ROOT / "data/recatch/manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def load_expert(row, book):
    meta = manifest_by_id()[row["id"]]
    actions = np.load(ROOT / meta["continuous_npz"])["actions"].astype(np.float32)
    packed = np.load(ROOT / meta["token_residual_npz"])
    tokens = packed["tokens"].astype(np.int64)
    residuals = packed["residuals"].astype(np.float32)
    decoded = []
    for token, residual in zip(tokens, residuals):
        _residual, action = decode_action(book, int(token), residual)
        decoded.append(action)
    decoded = np.stack(decoded, axis=0)
    return meta, actions, tokens, residuals, decoded


def begin(env, blob):
    if "capture" in blob:
        restore_capture(env, blob["capture"])
    else:
        from envs.observations.observable_obs import ObservableObsState

        restore_ballistic(env.sim, blob)
        restore_state(env.sim, None)
        env._hist = ObservableObsState(DT_POLICY)
        env._obs_seq = []
        env._prev_seq = []
        env._prev = np.zeros(7, np.float32)
    return float(env.sim.data.time)


def snapshot_state(env):
    sim = env.sim
    pose = _pose(sim)
    fsm = sim.fsm
    latch = export_state(sim)
    obj = int(sim.ids.object_body)
    return {
        "time": float(sim.data.time),
        "qpos": np.array(sim.data.qpos, np.float64),
        "qvel": np.array(sim.data.qvel, np.float64),
        "ctrl": np.array(sim.data.ctrl, np.float64),
        "p_des": np.array(fsm.p_des, np.float64),
        "r_des": np.array(fsm.r_des, np.float64).reshape(-1),
        "v_cmd": np.array(fsm.v_cmd, np.float64),
        "w_cmd": np.array(fsm.w_cmd, np.float64),
        "fg_cmd": float(fsm.fg_cmd),
        "phase": str(fsm.phase),
        "obj_pos": np.array(sim.data.xpos[obj], np.float64),
        "obj_cvel": np.array(sim.data.cvel[obj], np.float64),
        "nL": int(pose["nL"]),
        "nR": int(pose["nR"]),
        "bottom": float(pose["bottom"]),
        "gap_to_table": float(pose["bottom"]) - TABLE_TOP,
        "obj_z": float(pose["p_obj"][2]),
        "v_obj": np.array(pose["v_obj"], np.float64),
        "tilt_deg": float(pose["tilt_deg"]),
        "finger_gap": float(pose["gap"]),
        "latched": bool(latch["latched"]),
        "latch_kind": latch["kind"],
        "latch_t": latch["t"],
    }


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def compare_states(left, right):
    keys = ("qpos", "qvel", "ctrl", "p_des", "r_des", "v_cmd", "w_cmd", "obj_pos", "obj_cvel", "v_obj")
    return {
        "time": abs(float(left["time"]) - float(right["time"])),
        **{key: _max_abs(left[key], right[key]) for key in keys},
        "nL": int(left["nL"]) - int(right["nL"]),
        "nR": int(left["nR"]) - int(right["nR"]),
        "bottom": abs(float(left["bottom"]) - float(right["bottom"])),
        "latched": bool(left["latched"]) != bool(right["latched"]),
        "phase": left["phase"] != right["phase"],
        "fg_cmd": abs(float(left["fg_cmd"]) - float(right["fg_cmd"])),
    }


def play(env, blob, actions, observe=True):
    """GRPO recovery loop with a prescribed action list. Stops on recovery_failed."""
    t0 = begin(env, blob)
    rows = [snapshot_state(env)]
    executed = []
    lost_low = 0
    how = None
    obs0 = None
    for action in np.asarray(actions, np.float32):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            how = "forced"
            break
        if observe:
            obs = np.asarray(env.observation(), np.float32)
            if obs0 is None:
                obs0 = obs.copy()
            env._obs_seq.append(obs)
            env._prev_seq.append(env._prev.copy())
        done = step_legal(env.sim, action)
        env._prev = np.asarray(done, np.float32)
        executed.append(np.asarray(done, np.float32))
        rows.append(snapshot_state(env))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how = "drop"
            break
    if how == "drop":
        from envs.outcomes.horizon_outcome import _fail_recovery

        outcome = _fail_recovery(env.sim, blob["ref"], t0)
    else:
        outcome = finish_recovery(env, blob["ref"], t0, "forced" if how is None else how)
    return {
        "reason": outcome["reason"],
        "success": bool(outcome.get("success")),
        "steps": len(executed),
        "recovery_elapsed": float(outcome.get("recovery_elapsed") or 0.0),
        "latch": export_state(env.sim),
        "rows": rows,
        "executed": np.stack(executed, axis=0) if executed else np.zeros((0, 7), np.float32),
        "obs0": obs0,
    }


def policy_distribution(model, env, device):
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
    probs = torch.softmax(logits[-1].float(), dim=-1).cpu().numpy()
    probs = np.clip(probs, 0.0, None)
    probs = probs / max(float(probs.sum()), 1e-12)
    residual = mean[-1].detach().float().cpu().numpy().astype(np.float32)
    entropy = float(-(probs[probs > 0.0] * np.log(probs[probs > 0.0])).sum())
    return obs, probs, residual, entropy


def sample_action(model, env, rng, device, book):
    _obs, probs, mean, entropy = policy_distribution(model, env, device)
    token = int(rng.choice(len(probs), p=probs))
    pre = mean + rng.normal(0.0, SIGMA, size=7).astype(np.float32)
    if int(token) == HANDOFF:
        return token, pre, None, probs, entropy
    _residual, action = decode_action(book, token, pre)
    return token, pre, action, probs, entropy


def rollout_policy(env, blob, model, book, rng, device, prefix_actions=None):
    t0 = begin(env, blob)
    lost_low = 0
    how = None
    steps = []
    for action in np.asarray(prefix_actions, np.float32) if prefix_actions is not None else []:
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            how = "forced"
            break
        env._obs_seq.append(np.asarray(env.observation(), np.float32))
        env._prev_seq.append(env._prev.copy())
        done = step_legal(env.sim, action)
        env._prev = np.asarray(done, np.float32)
        steps.append({"source": "prefix", "action": np.asarray(done, np.float32)})
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how = "drop"
            break
    while how is None and float(env.sim.data.time) - t0 < LIMIT_S - 1e-9:
        token, pre, action, probs, entropy = sample_action(model, env, rng, device, book)
        if token == HANDOFF:
            steps.append({"source": "policy", "token": token, "p": float(probs[token]), "entropy": entropy})
            how = "early"
            break
        done = step_legal(env.sim, action)
        env._prev = np.asarray(done, np.float32)
        steps.append({
            "source": "policy",
            "token": int(token),
            "residual_linf": float(np.max(np.abs(pre))),
            "p": float(probs[token]),
            "entropy": entropy,
            "p_handoff": float(probs[HANDOFF]),
        })
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how = "drop"
            break
    if how == "drop":
        from envs.outcomes.horizon_outcome import _fail_recovery

        outcome = _fail_recovery(env.sim, blob["ref"], t0)
    else:
        outcome = finish_recovery(env, blob["ref"], t0, "forced" if how is None else how)
    return {
        "reason": outcome["reason"],
        "success": bool(outcome.get("success")),
        "policy_steps": sum(1 for step in steps if step["source"] == "policy" and step.get("token") != HANDOFF),
        "recovery_elapsed": float(outcome.get("recovery_elapsed") or 0.0),
        "first_token": next((step.get("token") for step in steps if step["source"] == "policy"), None),
        "latch_kind": export_state(env.sim)["kind"],
    }


def feedback_oracle(env, blob, spec):
    """Closed-loop open-overchase law from the current state. Mode starts at open."""
    sim = env.sim
    choose = staged_choose(spec)
    state = {"mode": "open"}
    t0 = float(sim.data.time)
    lost_low = 0
    actions = 0
    stop = None
    while float(sim.data.time) - t0 < LIMIT_S - 1e-9:
        pose = _pose(sim)
        raw = contacts_now(sim)
        forces, sides = _class_force(raw)
        band = _pad_band(sim)
        pose["t"] = float(sim.data.time) - t0
        pose["cz"] = float(pose["cz"])
        pose["pad_lo"], pose["pad_hi"] = band
        pose["fn_pad"] = forces["LOWER_VERTICAL_PAD"]
        pose["fn_slope"] = forces["UPPER_SLOPED_PART"]
        pose["pad_sides"] = sides["LOWER_VERTICAL_PAD"]
        pose["table"] = bool(latched(sim) or pose["bottom"] < TABLE_TOP)
        decision = choose(pose, raw, state)
        if pose["table"] or decision.get("stop"):
            stop = "table" if pose["table"] else state["mode"]
            break
        action = _action_from(sim, decision["v"], np.zeros(3), decision["tau"])
        step_legal(sim, action)
        actions += 1
        failed, lost_low = recovery_failed(sim, blob["ref"], lost_low)
        if failed:
            stop = "recovery_failed"
            break
    if stop == "recovery_failed" or (stop == "table" and latched(sim)):
        from envs.outcomes.horizon_outcome import _fail_recovery

        outcome = _fail_recovery(sim, blob["ref"], t0)
    elif stop == "handoff":
        outcome = finish_recovery(env, blob["ref"], t0, "early")
    else:
        outcome = finish_recovery(env, blob["ref"], t0, "forced")
    return {
        "reason": outcome["reason"],
        "success": bool(outcome.get("success")),
        "oracle_steps": actions,
        "stop": stop,
        "recovery_elapsed": float(outcome.get("recovery_elapsed") or 0.0),
    }


def expert_probabilities(env, blob, model, book, tokens, residuals, device):
    t0 = begin(env, blob)
    rows = []
    lost_low = 0
    for index, (token, residual) in enumerate(zip(tokens, residuals)):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            break
        _obs, probs, mean, entropy = policy_distribution(model, env, device)
        order = np.argsort(-probs)
        rank = int(np.where(order == int(token))[0][0]) + 1
        _decoded_residual, action = decode_action(book, int(token), residual)
        rows.append({
            "step": index,
            "expert_token": int(token),
            "probability": float(probs[int(token)]),
            "log_prob": float(np.log(max(probs[int(token)], 1e-12))),
            "rank": rank,
            "entropy": entropy,
            "handoff_probability": float(probs[HANDOFF]),
            "top_token": int(order[0]),
            "top_probability": float(probs[int(order[0])]),
            "residual_l2": float(np.linalg.norm(mean - residual)),
            "residual_linf": float(np.max(np.abs(mean - residual))),
        })
        if int(token) == HANDOFF:
            break
        done = step_legal(env.sim, action)
        env._prev = np.asarray(done, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            break
    return rows


def first_action_scan(env, blob, model, book, device):
    begin(env, blob)
    before = snapshot_state(env)
    _obs, probs, mean, entropy = policy_distribution(model, env, device)
    # policy_distribution already appended this observation. Restore and rescan per token.
    survivors = []
    dropped = []
    for token in range(len(probs)):
        begin(env, blob)
        policy_distribution(model, env, device)
        if token == HANDOFF:
            continue
        _residual, action = decode_action(book, token, mean)
        step_legal(env.sim, action)
        after = snapshot_state(env)
        failed, _lost = recovery_failed(env.sim, blob["ref"], 0)
        item = {
            "token": int(token),
            "probability": float(probs[token]),
            "failed": bool(failed),
            "latched": bool(after["latched"]),
            "latch_kind": after["latch_kind"],
            "bottom": float(after["bottom"]),
            "gap_to_table": float(after["gap_to_table"]),
            "nL": int(after["nL"]),
            "nR": int(after["nR"]),
            "vz": float(after["v_obj"][2]),
        }
        (dropped if failed or after["latched"] else survivors).append(item)
    order = np.argsort(-probs)
    return {
        "before": before,
        "entropy": entropy,
        "handoff_probability": float(probs[HANDOFF]),
        "top_tokens": [int(token) for token in order[:5]],
        "survivor_tokens": [item["token"] for item in survivors],
        "dropped_tokens": len(dropped),
        "surviving_high_prob": [item for item in survivors if item["probability"] >= 0.02],
    }


def prepare_horizontal(seated_ref):
    tapes = manifest_by_id()
    rows = []
    for row in load_train_snapshots():
        if row["primary_category"] != HORIZONTAL:
            continue
        if row["id"] not in tapes:
            raise RuntimeError(f"horizontal schedule id has no recatch tape: {row['id']}")
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if "ref" not in blob:
            blob = dict(blob)
            blob["ref"] = seated_ref
        blob["_id"] = row["id"]
        blob["_had_ref"] = "ref" in pickle.loads((ROOT / row["snapshot"]).read_bytes())
        rows.append((row, blob))
    return rows


def main():
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    book = load_codebook()
    device = torch.device("cpu")
    model, saved_book = load_trained_k48(CKPT, device)
    model.eval()
    if not np.array_equal(saved_book, book):
        raise RuntimeError("checkpoint codebook does not match the configured codebook")
    adaptive = ROOT / "results/grpo/adaptive_hybrid/checkpoint_adaptive.pt"
    log_path = ROOT / "results/grpo/adaptive_hybrid/ic_log.jsonl"
    identity = {
        "checkpoint": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
        "epoch": int(saved.get("epoch")),
        "logits": list(saved["model"]["logits.weight"].shape),
        "adaptive_checkpoint_exists": adaptive.is_file(),
        "ic_log_bytes": log_path.stat().st_size if log_path.is_file() else 0,
        "sigma": SIGMA,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "residual_cap": CAP,
        "sampling": "full 48-way softmax, then residual mean + N(0, 0.05)",
    }
    _write("checkpoint.json", identity)
    print("IDENTITY", identity, flush=True)

    _sim, _base, seated_ref, _target, _code = _setup()
    specs = dict(_specs())
    horizontal = prepare_horizontal(seated_ref)
    env = make_training_env(device, model=model)
    print("HORIZONTAL", len(horizontal), flush=True)

    consistency = []
    for row, blob in horizontal[:4]:
        meta, actions, tokens, residuals, decoded = load_expert(row, book)
        direct = play(env, blob, decoded, observe=False)
        grpo = play(env, blob, decoded, observe=True)
        # Second call is the same function. Compare two independent restores.
        mismatches = []
        limit = min(len(direct["rows"]), len(grpo["rows"]))
        for index in range(limit):
            diff = compare_states(direct["rows"][index], grpo["rows"][index])
            if diff["qpos"] > 1e-6 or diff["qvel"] > 1e-6 or diff["latched"] or diff["nL"] or diff["nR"]:
                mismatches.append({"step": index, **diff})
        action_error = float(np.max(np.abs(decoded - actions)))
        consistency.append({
            "id": row["id"],
            "strategy": meta["strategy"],
            "snapshot_had_ref": blob["_had_ref"],
            "expert_steps": int(len(actions)),
            "decoded_vs_stored_linf": action_error,
            "direct_reason": direct["reason"],
            "replay_reason": grpo["reason"],
            "direct_success": direct["success"],
            "replay_success": grpo["success"],
            "direct_steps": direct["steps"],
            "replay_steps": grpo["steps"],
            "state_mismatches": mismatches[:8],
            "start": direct["rows"][0],
            "after_one": None if len(direct["rows"]) < 2 else direct["rows"][1],
            "latch_after": direct["latch"],
        })
        print("CONSISTENCY", row["id"], direct["reason"], grpo["reason"], "mismatch", len(mismatches), flush=True)
    _write("consistency.json", consistency)
    if any(item["state_mismatches"] or item["direct_reason"] != item["replay_reason"] for item in consistency):
        raise SystemExit("execution paths diverged; sampling experiments were not started")

    # Full expert replay on every horizontal train IC, GRPO execution path.
    expert_replay = []
    for row, blob in horizontal:
        meta, actions, tokens, residuals, decoded = load_expert(row, book)
        result = play(env, blob, decoded, observe=True)
        expert_replay.append({
            "id": row["id"],
            "strategy": meta["strategy"],
            "reason": result["reason"],
            "success": result["success"],
            "steps": result["steps"],
            "tape_steps": int(len(decoded)),
            "recovery_elapsed": result["recovery_elapsed"],
            "start_gap": result["rows"][0]["gap_to_table"],
            "start_bottom": result["rows"][0]["bottom"],
            "start_nL": result["rows"][0]["nL"],
            "start_nR": result["rows"][0]["nR"],
            "start_vz": float(result["rows"][0]["v_obj"][2]),
            "start_latched": result["rows"][0]["latched"],
            "latch_kind": result["latch"]["kind"],
            "decoded_vs_stored_linf": float(np.max(np.abs(decoded - actions))),
        })
        print("EXPERT", row["id"], result["reason"], result["steps"], flush=True)
    _write("expert_replay.json", expert_replay)

    # Probabilities along the expert tape. Also a few sliding and vertical controls.
    probability = []
    for row, blob in horizontal:
        meta, _actions, tokens, residuals, _decoded = load_expert(row, book)
        rows = expert_probabilities(env, blob, model, book, tokens, residuals, device)
        probability.append({"id": row["id"], "category": HORIZONTAL, "strategy": meta["strategy"], "steps": rows})
        print("PROB", row["id"], "step0", None if not rows else round(rows[0]["probability"], 4), flush=True)
    controls = []
    for row in load_train_snapshots():
        if row["primary_category"] == HORIZONTAL:
            continue
        controls.append(row)
        if sum(1 for item in controls if item["primary_category"] == SLIDING) >= 4 and sum(1 for item in controls if item["primary_category"] == VERTICAL) >= 4:
            break
    tapes = {}
    for name, path in (
        (SLIDING, ROOT / "data/gravity_assisted_in_hand_sliding/manifest.jsonl"),
        (VERTICAL, ROOT / "data/vertical_recatch/manifest.jsonl"),
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                tapes.setdefault(item["id"], item)
    for row in controls:
        if row["primary_category"] == HORIZONTAL or row["id"] not in tapes:
            continue
        meta = tapes[row["id"]]
        if "token_residual_npz" not in meta:
            continue
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if "ref" not in blob:
            blob = dict(blob)
            blob["ref"] = seated_ref
        packed = np.load(ROOT / meta["token_residual_npz"])
        rows = expert_probabilities(
            env, blob, model, book,
            packed["tokens"].astype(np.int64),
            packed["residuals"].astype(np.float32),
            device,
        )
        probability.append({"id": row["id"], "category": row["primary_category"], "strategy": meta.get("strategy"), "steps": rows})
        print("PROB", row["primary_category"], row["id"], "step0", None if not rows else round(rows[0]["probability"], 4), flush=True)
    _write("probabilities.json", probability)

    # Prefix experiments on the first three horizontal ICs.
    prefix_rows = []
    oracle_rows = []
    chosen = horizontal[:3]
    for row, blob in chosen:
        meta, _actions, _tokens, _residuals, decoded = load_expert(row, book)
        spec = specs.get(meta["strategy"])
        for k in KS:
            if k > len(decoded):
                continue
            for seed in (1, 2, 3, 4):
                result = rollout_policy(
                    env, blob, model, book, np.random.default_rng(1000 + 17 * seed + k), device, decoded[:k],
                )
                prefix_rows.append({"id": row["id"], "k": k, "seed": seed, "kind": "expert_prefix_policy_suffix", **result})
                print("PREFIX", row["id"], "k", k, "seed", seed, result["reason"], flush=True)
            t0 = begin(env, blob)
            lost_low = 0
            prefix_failed = False
            prefix_reason = None
            for action in decoded[:k]:
                step_legal(env.sim, action)
                failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
                if failed:
                    prefix_failed = True
                    from envs.outcomes.horizon_outcome import _fail_recovery

                    prefix_reason = _fail_recovery(env.sim, blob["ref"], t0)["reason"]
                    break
            if prefix_failed:
                oracle_rows.append({
                    "id": row["id"], "k": k, "kind": "fixed_tape_suffix",
                    "reason": prefix_reason, "success": False, "note": "expert prefix already terminal",
                })
                oracle_rows.append({
                    "id": row["id"], "k": k, "kind": "feedback_oracle",
                    "reason": prefix_reason, "success": False, "note": "expert prefix already terminal",
                })
                print("ORACLE", row["id"], "k", k, prefix_reason, flush=True)
                continue
            how = None
            for action in decoded[k:]:
                if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
                    how = "forced"
                    break
                step_legal(env.sim, action)
                failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
                if failed:
                    how = "drop"
                    break
            if how == "drop":
                from envs.outcomes.horizon_outcome import _fail_recovery

                outcome = _fail_recovery(env.sim, blob["ref"], t0)
            else:
                outcome = finish_recovery(env, blob["ref"], t0, "forced" if how is None else how)
            oracle_rows.append({
                "id": row["id"], "k": k, "kind": "fixed_tape_suffix",
                "reason": outcome["reason"], "success": bool(outcome.get("success")),
                "note": "remaining original tape, not a replan",
            })
            begin(env, blob)
            lost_low = 0
            for action in decoded[:k]:
                step_legal(env.sim, action)
                failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
                if failed:
                    break
            if spec is None:
                oracle_rows.append({"id": row["id"], "k": k, "kind": "feedback_oracle", "reason": "NO_SPEC", "success": False})
            else:
                oracle_rows.append({"id": row["id"], "k": k, "kind": "feedback_oracle", **feedback_oracle(env, blob, spec)})
            print("ORACLE", row["id"], "k", k, oracle_rows[-1]["reason"], flush=True)
        # Policy's own first step, then the feedback oracle replans.
        for seed in (1, 2, 3, 4):
            begin(env, blob)
            rng = np.random.default_rng(5000 + seed)
            token, pre, action, probs, entropy = sample_action(model, env, rng, device, book)
            if token == HANDOFF:
                oracle_rows.append({
                    "id": row["id"], "k": 1, "seed": seed, "kind": "policy_prefix_feedback_oracle",
                    "reason": "HANDOFF", "success": False, "token": token,
                })
                continue
            step_legal(env.sim, action)
            failed, _lost = recovery_failed(env.sim, blob["ref"], 0)
            if failed or spec is None:
                from envs.outcomes.horizon_outcome import _fail_recovery

                outcome = _fail_recovery(env.sim, blob["ref"], float(env.sim.data.time))
                oracle_rows.append({
                    "id": row["id"], "k": 1, "seed": seed, "kind": "policy_prefix_feedback_oracle",
                    "reason": outcome["reason"], "success": False, "token": int(token),
                    "note": "terminal before oracle" if failed else "no spec",
                })
            else:
                followed = feedback_oracle(env, blob, spec)
                oracle_rows.append({
                    "id": row["id"], "k": 1, "seed": seed, "kind": "policy_prefix_feedback_oracle",
                    "token": int(token), "token_probability": float(probs[token]), **followed,
                })
            print("POLICY_ORACLE", row["id"], seed, oracle_rows[-1]["reason"], flush=True)
    _write("prefix_policy.json", prefix_rows)
    _write("oracle_suffix.json", oracle_rows)

    scans = []
    terminations = []
    for row, blob in chosen:
        scan = first_action_scan(env, blob, model, book, device)
        meta, _actions, tokens, residuals, decoded = load_expert(row, book)
        _residual, expert_action = decode_action(book, int(tokens[0]), residuals[0])
        begin(env, blob)
        before = snapshot_state(env)
        step_legal(env.sim, expert_action)
        after = snapshot_state(env)
        failed, _lost = recovery_failed(env.sim, blob["ref"], 0)
        scan["id"] = row["id"]
        scan["expert_token"] = int(tokens[0])
        scan["expert_first_failed"] = bool(failed)
        scan["expert_after"] = after
        scan["expert_before"] = before
        scans.append(scan)
        for seed in range(16):
            result = rollout_policy(env, blob, model, book, np.random.default_rng(7000 + seed), device)
            terminations.append({"id": row["id"], "seed": seed, **result})
            print("INDEPENDENT", row["id"], seed, result["reason"], result["recovery_elapsed"], flush=True)
    _write("first_action_scan.json", scans)
    _write("independent16.json", terminations)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
