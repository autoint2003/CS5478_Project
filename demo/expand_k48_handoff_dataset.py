"""Expand the migrated recovery set with pauses and verified HANDOFF labels.

Does not train. Does not modify the K=47 codebook, the 296-row manifest, or
historical checkpoints.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.autonomous_episode import make_training_env
from training.horizon_outcome import (
    CONTROL_PERIOD_S,
    finish_recovery,
    recovery_failed,
    replay_commands,
)
from training.history_aware_critic_ablation import restore_capture
from training.migrate_horizon_dataset import mechanism
from training.privileged_ball_impact_search import representability, step_legal
from training.unified_k47 import CAP, load_codebook, read_jsonl
from training.unified_k48 import HANDOFF, SELECTED_LIMIT_S

RAW = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2"
MIGRATED = ROOT / "demo_results/diagnostics/raw/horizon_migration"
OUT = ROOT / "demo_results/diagnostics/raw/k48_handoff"
LIMITS = (2.4, 4.0, 6.0, 8.0)
PAUSE_S = (0.04, 0.08, 0.12, 0.20)
PAUSE_FRACTIONS = (0.25, 0.50, 0.75)
PARENTS_PER_MECHANISM = 4


def pause_action(previous):
    """Zero translation and wrist, with the grip channel held.

    The all-zero 7-vector is not this command. Grip action 0 maps to -8 N·m,
    halfway between open (+2 N·m) and secure (-18 N·m). Keeping the previous
    grip command avoids opening the fingers just because the hand is asked to
    hold still. Physics keeps integrating.
    """
    grip = float(np.asarray(previous, np.float32).reshape(7)[6])
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip], np.float32)


def _execute(env, blob, actions, limit_s):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    observations = []
    executed = []
    lost_low = 0
    forced = False
    for action in np.asarray(actions, np.float32):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            forced = True
            break
        observations.append(np.asarray(env.observation(), np.float32))
        done = step_legal(env.sim, action)
        executed.append(np.asarray(done, np.float32))
        env._prev = executed[-1]
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "ok": False,
                "reason": "DROP_DURING_RECOVERY",
                "observations": observations,
                "actions": executed,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "t0": t0,
                "forced": False,
            }
    return {
        "ok": True,
        "observations": observations,
        "actions": executed,
        "recovery_elapsed": float(env.sim.data.time) - t0,
        "t0": t0,
        "forced": forced,
        "handoff_obs": np.asarray(env.observation(), np.float32),
        "qpos": np.array(env.sim.data.qpos, np.float64).copy(),
    }


def _finish(env, blob, executed_prefix, t0, how):
    outcome = finish_recovery(env, blob["ref"], t0, how)
    outcome["executed"] = len(executed_prefix["actions"])
    return outcome


def limits(env):
    rows = read_jsonl(RAW / "replay_k47_all.jsonl")
    lengths = []
    for row in rows:
        path = RAW / "trajectories" / f"{row['id']}__{row['strategy']}.npz"
        lengths.append((row["id"], row["strategy"], int(np.load(path)["actions"].shape[0])))
    counts = {limit: sum(1 for _i, _s, n in lengths if n * CONTROL_PERIOD_S <= limit + 1e-9) for limit in LIMITS}
    long = [item for item in lengths if item[2] > int(round(2.4 / CONTROL_PERIOD_S))]
    probes = []
    for episode_id, strategy, n_steps in long:
        actions = np.load(RAW / "trajectories" / f"{episode_id}__{strategy}.npz")["actions"]
        snap = ROOT / "demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train" / f"{episode_id}.pkl"
        with snap.open("rb") as handle:
            blob = pickle.load(handle)
        for limit in LIMITS:
            out = replay_commands(env, blob, actions, limit_s=limit, early_basin=False)
            probes.append({
                "id": episode_id,
                "strategy": strategy,
                "saved_steps": n_steps,
                "limit_s": limit,
                "reason": out["reason"],
                "handoff": out.get("handoff"),
                "recovery_elapsed": out.get("recovery_elapsed"),
                "nominal_elapsed": out.get("nominal_elapsed"),
                "recovery_steps": out.get("recovery_steps"),
            })
            print("LIMIT", episode_id, limit, out["reason"], out.get("recovery_steps"), flush=True)
    search = read_jsonl(RAW / "search.jsonl")
    missing = [row for row in search if not (RAW / "trajectories" / f"{row['id']}__{row['strategy']}.npz").is_file()]
    payload = {
        "saved_action_files": len(lengths),
        "length_steps": {"min": min(n for _i, _s, n in lengths), "max": max(n for _i, _s, n in lengths)},
        "sequences_fitting_limit": counts,
        "longer_than_2_4s": [{"id": i, "strategy": s, "steps": n} for i, s, n in long],
        "replays": probes,
        "searches_without_actions": len(missing),
        "searches_without_actions_by_label": dict(Counter(row.get("label") for row in missing)),
        "selected_limit_s": SELECTED_LIMIT_S,
        "selection": (
            "No saved recovery is longer than 2.44 s, and the 1345 unsuccessful searches "
            "have no action array. 4 s is the shortest tested limit that contains the "
            "122-step demonstration and a 0.20 s pause. 6 s and 8 s add no saved trajectory."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "limit_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "unresolved_searches.json").write_text(json.dumps({
        "count": len(missing),
        "reason": "search.jsonl has a label and no saved action array, so the attempt cannot be replayed",
        "by_historical_label": dict(Counter(row.get("label") for row in missing)),
    }, indent=2), encoding="utf-8")
    print("LIMITS", json.dumps({key: payload[key] for key in ("sequences_fitting_limit", "searches_without_actions", "selected_limit_s")}), flush=True)


def _parents():
    rows = read_jsonl(MIGRATED / "supervised_targets.jsonl")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mechanism"]].append(row)
    rng = np.random.default_rng(1)
    chosen = []
    for name, items in grouped.items():
        by_ic = defaultdict(list)
        for item in items:
            by_ic[item["id"]].append(item)
        ics = sorted(by_ic)
        take = min(PARENTS_PER_MECHANISM, len(ics))
        pick = rng.choice(ics, size=take, replace=False)
        for episode_id in pick:
            chosen.append(by_ic[episode_id][0])
    return chosen


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _done(path):
    found = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                found.add(row["key"])
    return found


def pauses(env, book):
    path = OUT / "pause_trials.jsonl"
    done = _done(path)
    cache = {}
    for parent in _parents():
        actions = np.load(ROOT / parent["continuous_npz"])["actions"].astype(np.float32)
        n = len(actions)
        if parent["snapshot"] not in cache:
            with (ROOT / parent["snapshot"]).open("rb") as handle:
                cache[parent["snapshot"]] = pickle.load(handle)
        blob = cache[parent["snapshot"]]
        for fraction in PAUSE_FRACTIONS:
            start = int(np.clip(round(fraction * n), 1, n - 1))
            for duration in PAUSE_S:
                key = f"{parent['id']}|{parent['strategy']}|{fraction}|{duration}"
                if key in done:
                    continue
                n_pause = int(round(duration / CONTROL_PERIOD_S))
                hold = pause_action(actions[start - 1])
                sequence = np.concatenate([
                    actions[:start],
                    np.repeat(hold.reshape(1, 7), n_pause, axis=0),
                    actions[start:],
                ], axis=0)
                prefix = _execute(env, blob, sequence, SELECTED_LIMIT_S)
                record = {
                    "key": key,
                    "id": parent["id"],
                    "strategy": parent["strategy"],
                    "mechanism": parent["mechanism"],
                    "snapshot": parent["snapshot"],
                    "parent_npz": parent["continuous_npz"],
                    "pause_fraction": fraction,
                    "pause_start_step": start,
                    "pause_duration_s": duration,
                    "pause_command": [round(float(x), 5) for x in hold],
                    "reason": prefix["reason"] if not prefix["ok"] else None,
                }
                if not prefix["ok"]:
                    record["admitted"] = False
                    _append(path, record)
                    print("PAUSE", key, record["reason"], flush=True)
                    continue
                how = "forced" if prefix["forced"] else "recorded_end"
                outcome = _finish(env, blob, prefix, prefix["t0"], how)
                record["reason"] = outcome["reason"]
                record["recovery_elapsed"] = outcome.get("recovery_elapsed")
                record["nominal_elapsed"] = outcome.get("nominal_elapsed")
                record["lift_completed"] = outcome.get("lift_completed")
                record["executed_steps"] = len(prefix["actions"])
                if outcome["reason"] != "SUCCESS_12S":
                    record["admitted"] = False
                    _append(path, record)
                    print("PAUSE", key, outcome["reason"], flush=True)
                    continue
                info = representability(np.stack(prefix["actions"]), book)
                record["max_linf"] = info["max_linf"]
                if info["fraction_within_0.25"] < 1.0 - 1e-9:
                    record["admitted"] = False
                    record["error"] = "pause trajectory is outside the ±0.25 residual"
                    _append(path, record)
                    print("PAUSE", key, "UNRESOLVED", flush=True)
                    continue
                decoded = _execute(env, blob, info["quantized"], SELECTED_LIMIT_S)
                if not decoded["ok"]:
                    record["admitted"] = False
                    record["decoded_reason"] = decoded["reason"]
                    _append(path, record)
                    print("PAUSE", key, "DECODED", decoded["reason"], flush=True)
                    continue
                decoded_out = _finish(env, blob, decoded, decoded["t0"], "forced" if decoded["forced"] else "recorded_end")
                record["decoded_reason"] = decoded_out["reason"]
                record["admitted"] = decoded_out["reason"] == "SUCCESS_12S"
                if record["admitted"]:
                    folder = OUT / "trajectories"
                    folder.mkdir(parents=True, exist_ok=True)
                    stem = f"{parent['id']}__{parent['strategy']}__p{int(fraction * 100)}_{int(duration * 1000)}"
                    tokens = np.asarray(info["tokens"], np.int64)
                    residuals = np.clip(np.stack(prefix["actions"]) - book[tokens], -CAP, CAP)
                    np.savez_compressed(
                        folder / f"{stem}.npz",
                        actions=np.stack(prefix["actions"]),
                        observations=np.stack(prefix["observations"]),
                        handoff_obs=prefix["handoff_obs"],
                        tokens=tokens,
                        residuals=residuals,
                    )
                    record["npz"] = str((folder / f"{stem}.npz").relative_to(ROOT)).replace("\\", "/")
                _append(path, record)
                print("PAUSE", key, record["decoded_reason"], "admitted", record["admitted"], flush=True)


def _candidate_steps(n_actions):
    picks = []
    for fraction in PAUSE_FRACTIONS:
        step = int(np.clip(round(fraction * n_actions), 1, max(1, n_actions - 1)))
        if step not in picks and step < n_actions:
            picks.append(step)
    return picks


def _handoff_at(env, blob, actions, steps, limit_s):
    prefix = _execute(env, blob, actions[:steps], limit_s)
    if not prefix["ok"]:
        return {"steps": steps, "reason": prefix["reason"], "success": False}
    outcome = _finish(env, blob, prefix, prefix["t0"], "early")
    return {
        "steps": steps,
        "reason": outcome["reason"],
        "success": outcome["reason"] == "SUCCESS_12S",
        "recovery_elapsed": outcome.get("recovery_elapsed"),
        "nominal_elapsed": outcome.get("nominal_elapsed"),
        "lift_completed": outcome.get("lift_completed"),
        "observations": None if not prefix["ok"] else np.stack(prefix["observations"]),
        "handoff_obs": None if not prefix["ok"] else prefix["handoff_obs"],
        "actions": None if not prefix["ok"] else np.stack(prefix["actions"]),
    }


def handoffs(env):
    path = OUT / "handoff_candidates.jsonl"
    done = _done(path)
    core = read_jsonl(MIGRATED / "supervised_targets.jsonl")
    pause_rows = []
    pause_path = OUT / "pause_trials.jsonl"
    if pause_path.is_file():
        pause_rows = [json.loads(line) for line in pause_path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("admitted")]
    jobs = []
    for row in core:
        jobs.append({
            "key": f"core|{row['id']}|{row['strategy']}",
            "id": row["id"],
            "strategy": row["strategy"],
            "mechanism": row["mechanism"],
            "snapshot": row["snapshot"],
            "actions": np.load(ROOT / row["continuous_npz"])["actions"].astype(np.float32),
            "source": "historical",
        })
    for row in pause_rows:
        jobs.append({
            "key": f"pause|{row['key']}",
            "id": row["id"],
            "strategy": row["strategy"],
            "mechanism": row["mechanism"],
            "snapshot": row["snapshot"],
            "actions": np.load(ROOT / row["npz"])["actions"].astype(np.float32),
            "source": "pause",
        })
    cache = {}
    for job in jobs:
        if job["key"] in done:
            continue
        if job["snapshot"] not in cache:
            with (ROOT / job["snapshot"]).open("rb") as handle:
                cache[job["snapshot"]] = pickle.load(handle)
        blob = cache[job["snapshot"]]
        tested = []
        payloads = []
        for step in _candidate_steps(len(job["actions"])):
            item = _handoff_at(env, blob, job["actions"], step, SELECTED_LIMIT_S)
            payloads.append(item)
            tested.append({key: value for key, value in item.items() if key not in ("observations", "handoff_obs", "actions")})
        positives = [item for item in tested if item["success"]]
        robust = None
        for item in positives:
            steps = int(item["steps"])
            hold = pause_action(job["actions"][steps - 1])
            perturbed = np.concatenate([job["actions"][:steps], np.repeat(hold.reshape(1, 7), 2, axis=0)], axis=0)
            probe = _execute(env, blob, perturbed, SELECTED_LIMIT_S)
            if not probe["ok"]:
                item["pause_probe"] = probe["reason"]
                continue
            outcome = _finish(env, blob, probe, probe["t0"], "early")
            item["pause_probe"] = outcome["reason"]
            if outcome["reason"] == "SUCCESS_12S" and robust is None:
                robust = steps
        selected = robust if robust is not None else (int(positives[-1]["steps"]) if positives else None)
        saved = None
        if selected is not None:
            chosen = next(item for item in payloads if int(item["steps"]) == int(selected) and item.get("actions") is not None)
            book = load_codebook()
            info = representability(chosen["actions"], book)
            if info["fraction_within_0.25"] < 1.0 - 1e-9:
                selected = None
                robust = None
            else:
                folder = OUT / "handoff_prefixes"
                folder.mkdir(parents=True, exist_ok=True)
                stem = job["key"].replace("|", "__")
                target = folder / f"{stem}.npz"
                tokens = np.asarray(info["tokens"], np.int64)
                residuals = np.clip(chosen["actions"] - book[tokens], -float(CAP), float(CAP))
                np.savez_compressed(
                    target,
                    actions=chosen["actions"],
                    observations=chosen["observations"],
                    handoff_obs=chosen["handoff_obs"],
                    tokens=tokens,
                    residuals=residuals,
                )
                saved = str(target.relative_to(ROOT)).replace("\\", "/")
        record = {
            "key": job["key"],
            "id": job["id"],
            "strategy": job["strategy"],
            "mechanism": job["mechanism"],
            "source": job["source"],
            "snapshot": job["snapshot"],
            "tested": tested,
            "selected_steps": selected,
            "robust_to_pause": robust is not None,
            "label": "early_handoff" if selected is not None else "recorded_end",
            "prefix_npz": saved,
        }
        _append(path, record)
        print("HANDOFF", job["key"], record["label"], selected, flush=True)


def views():
    core = read_jsonl(MIGRATED / "supervised_targets.jsonl")
    labels = {}
    label_path = OUT / "handoff_candidates.jsonl"
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["key"]] = row
    rows = []
    for row in core:
        label = labels.get(f"core|{row['id']}|{row['strategy']}", {})
        item = {
            **row,
            "source_kind": "historical",
            "parent_id": row["id"],
            "handoff_label": label.get("label", "recorded_end"),
            "handoff_steps": label.get("selected_steps"),
            "robust_to_pause": label.get("robust_to_pause", False),
            "pause_perturbed": False,
        }
        if label.get("prefix_npz"):
            item["continuous_npz"] = label["prefix_npz"]
            item["token_residual_npz"] = label["prefix_npz"]
        rows.append(item)
    pause_path = OUT / "pause_trials.jsonl"
    if pause_path.is_file():
        for line in pause_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("admitted"):
                continue
            label = labels.get(f"pause|{row['key']}", {})
            rows.append({
                "id": row["id"],
                "snapshot": row["snapshot"],
                "strategy": row["strategy"],
                "mechanism": row["mechanism"],
                "continuous_npz": row["npz"],
                "token_residual_npz": row["npz"],
                "source_kind": "pause",
                "parent_id": row["id"],
                "parent_strategy": row["strategy"],
                "handoff_label": label.get("label", "recorded_end"),
                "handoff_steps": label.get("selected_steps"),
                "robust_to_pause": label.get("robust_to_pause", False),
                "pause_perturbed": True,
                "pause_duration_s": row["pause_duration_s"],
                "steps_are_one_episode": True,
                "supervisory_target": True,
            })
    # Each mechanism has total weight 1 on historical rows and 0.25 on pause rows.
    # Within a kind and mechanism, each source IC shares that budget equally, then
    # its trajectories split that IC share. This keeps pause copies from dominating.
    buckets = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        kind = "pause" if row["pause_perturbed"] else "historical"
        buckets[(kind, row["mechanism"])][row["id"]].append(index)
    for (kind, _name), by_ic in buckets.items():
        budget = 0.25 if kind == "pause" else 1.0
        share = budget / len(by_ic)
        for indices in by_ic.values():
            for index in indices:
                rows[index]["sample_weight"] = share / len(indices)
    with (OUT / "supervised_targets.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    grpo = read_jsonl(MIGRATED / "grpo_initial_conditions.jsonl")
    with (OUT / "grpo_initial_conditions.jsonl").open("w", encoding="utf-8") as handle:
        for row in grpo:
            item = dict(row)
            item["horizon"] = "recovery_limit_4s_then_nominal_12s"
            item["recovery_limit_s"] = SELECTED_LIMIT_S
            handle.write(json.dumps(item) + "\n")
    summary = {
        "supervised_rows": len(rows),
        "ics": len({row["id"] for row in rows}),
        "historical_rows": sum(1 for row in rows if not row["pause_perturbed"]),
        "pause_rows": sum(1 for row in rows if row["pause_perturbed"]),
        "early_handoff": sum(1 for row in rows if row["handoff_label"] == "early_handoff"),
        "recorded_end": sum(1 for row in rows if row["handoff_label"] == "recorded_end"),
        "by_mechanism": dict(Counter(row["mechanism"] for row in rows)),
        "weight_sums": {},
        "grpo_ics": len(grpo),
        "handoff_token": HANDOFF,
        "recovery_limit_s": SELECTED_LIMIT_S,
    }
    sums = defaultdict(float)
    for row in rows:
        sums[row["mechanism"] if not row["pause_perturbed"] else row["mechanism"] + "|pause"] += row["sample_weight"]
    summary["weight_sums"] = {key: round(value, 6) for key, value in sums.items()}
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("VIEWS", json.dumps(summary), flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--limits"
    if mode == "--views":
        views()
        return
    import torch
    device = torch.device("cpu")
    book = load_codebook()
    from training.unified_k47 import build_model
    env = make_training_env(device, model=build_model(book, device))
    if mode == "--limits":
        limits(env)
        return
    if mode == "--pauses":
        pauses(env, book)
        return
    if mode == "--handoff":
        handoffs(env)
        return
    raise SystemExit("use --limits, --pauses, --handoff, or --views")


if __name__ == "__main__":
    main()
