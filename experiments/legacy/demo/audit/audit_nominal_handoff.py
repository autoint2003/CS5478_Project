"""Audit immediate nominal handoff on the 312 TRAIN initial conditions.

Restores each saved snapshot and hands off with no recovery command.
Does not write snapshots, checkpoints, or FINAL files.
"""

from __future__ import annotations


import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture

from envs.outcomes.horizon_outcome import finish_recovery, replay_commands
from training.k47_codebook import read_jsonl
from training.unified_k48 import load_trained_k48

OUT = ROOT / "results/archive/demo_results/diagnostics/raw/k48_handoff/nominal_audit"
ICS = ROOT / "results/archive/demo_results/diagnostics/raw/k48_handoff/grpo_initial_conditions.jsonl"
SUPERVISED = ROOT / "results/archive/demo_results/diagnostics/raw/k48_handoff/supervised_targets.jsonl"
VALID = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset/accepted/validation"
CKPT = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt"
LOGS = {
    "v1": ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v1",
    "v2": ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v2",
}


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def _done(path):
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _spec_fields(blob):
    spec = blob.get("spec") or {}
    evidence = blob.get("evidence") or {}
    states = evidence.get("states") or []
    return {
        "source_split": spec.get("split"),
        "family": spec.get("family") or spec.get("direction"),
        "direction": spec.get("direction"),
        "azimuth_deg": spec.get("azimuth_deg"),
        "speed_mps": spec.get("speed_mps"),
        "z_off_m": spec.get("z_off_m"),
        "phi_deg": spec.get("phi_deg"),
        "delay_s": spec.get("delay_s"),
        "states": list(states),
    }


def immediate(env, blob):
    if "final" in str(blob.get("spec", {}).get("split", "")):
        raise RuntimeError("refusing a final split")
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    out = finish_recovery(env, blob["ref"], t0, "early")
    return {
        "reason": out["reason"],
        "success": bool(out.get("success")),
        "nominal_solvable": bool(out.get("success")),
        "recovery_elapsed": out.get("recovery_elapsed"),
        "nominal_elapsed": out.get("nominal_elapsed"),
        "lift_completed": out.get("lift_completed"),
        "handoff": out.get("handoff"),
    }


def scan(env, jobs, destination):
    finished = _done(destination)
    for job in jobs:
        if job["id"] in finished:
            continue
        path = ROOT / job["snapshot"]
        if "final" in path.name or "final" in str(path):
            raise RuntimeError(f"refusing {path}")
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        row = {"id": job["id"], "snapshot": job["snapshot"], **_spec_fields(blob), **immediate(env, blob)}
        row.update({key: job.get(key) for key in ("source", "n_success", "unsolved", "verified_k47") if key in job})
        _append(destination, row)
        print("HANDOFF", row["id"], row["reason"], flush=True)


def replay_failing(env, failing_ids):
    destination = OUT / "demo_replays.jsonl"
    finished = _done(destination)
    by_id = defaultdict(list)
    for row in read_jsonl(SUPERVISED):
        if row["id"] in failing_ids and row.get("horizon_reason") == "SUCCESS_12S":
            by_id[row["id"]].append(row)
    for episode_id in sorted(failing_ids):
        if episode_id in finished:
            continue
        rows = by_id.get(episode_id, [])
        ordered = sorted(rows, key=lambda item: (item.get("pause_perturbed", False), item.get("strategy") or ""))
        attempts = []
        confirmed = False
        seen = set()
        for row in ordered:
            key = (row.get("strategy"), row.get("source_kind"), row.get("continuous_npz"))
            if key in seen:
                continue
            seen.add(key)
            with (ROOT / row["snapshot"]).open("rb") as handle:
                blob = pickle.load(handle)
            actions = np.load(ROOT / row["continuous_npz"])["actions"]
            out = replay_commands(env, blob, actions, limit_s=4.0, early_basin=False)
            attempts.append({
                "strategy": row.get("strategy"),
                "source_kind": row.get("source_kind"),
                "pause_perturbed": bool(row.get("pause_perturbed")),
                "reason": out["reason"],
                "success": bool(out.get("success")),
                "recovery_elapsed": out.get("recovery_elapsed"),
            })
            if out.get("success"):
                confirmed = True
                break
        status = "confirmed_recoverable" if confirmed else ("demo_replay_failed" if attempts else "no_demonstration")
        _append(destination, {"id": episode_id, "status": status, "attempts": attempts})
        print("REPLAY", episode_id, status, len(attempts), flush=True)


def _bucket(rows, key):
    grouped = defaultdict(lambda: {"n": 0, "solvable": 0})
    for row in rows:
        label = row.get(key)
        if isinstance(label, list):
            label = ",".join(label) if label else "none"
        grouped[str(label)]["n"] += 1
        grouped[str(label)]["solvable"] += int(bool(row.get("nominal_solvable")))
    return {name: value for name, value in sorted(grouped.items(), key=lambda item: (-item[1]["n"], item[0]))}


def _epoch(update):
    if update <= 312:
        return 1
    if update <= 624:
        return 2
    return 3


def analyze_logs(solvable):
    summary = {}
    for version, folder in LOGS.items():
        rollouts = [json.loads(line) for line in (folder / "rollouts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        updates = [json.loads(line) for line in (folder / "training_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        by_epoch = {}
        for epoch in (1, 2, 3):
            members = [row for row in rollouts if _epoch(int(row["update"])) == epoch]
            group_rows = [row for row in updates if _epoch(int(row["update"])) == epoch]
            immediate_n = sum(1 for row in members if row.get("handoff") == "early" and float(row.get("recovery_elapsed") or 0.0) < 1e-6)
            multistep = [row for row in members if float(row.get("recovery_elapsed") or 0.0) >= 0.02 or row.get("handoff") in ("forced", None)]
            successes = [row for row in members if row.get("reason") == "SUCCESS_12S"]
            recovery_success = [row for row in successes if float(row.get("recovery_elapsed") or 0.0) >= 0.02]
            solvable_members = [row for row in members if row["episode_id"] in solvable]
            failing_members = [row for row in members if row["episode_id"] not in solvable]
            by_epoch[str(epoch)] = {
                "rollouts": len(members),
                "first_step_handoff": immediate_n,
                "first_step_handoff_rate": immediate_n / len(members) if members else None,
                "multistep_rollouts": len(multistep),
                "multistep_on_nominal_failing": sum(1 for row in multistep if row["episode_id"] not in solvable),
                "success": len(successes),
                "success_with_recovery_steps": len(recovery_success),
                "success_on_solvable": sum(1 for row in solvable_members if row.get("reason") == "SUCCESS_12S"),
                "solvable_rollouts": len(solvable_members),
                "success_on_failing": sum(1 for row in failing_members if row.get("reason") == "SUCCESS_12S"),
                "failing_rollouts": len(failing_members),
                "skipped_advantage_updates": sum(1 for row in group_rows if row.get("advantage_skipped")),
                "reward_std_median": float(np.median([row.get("reward_std") or 0.0 for row in group_rows])) if group_rows else None,
                "nonzero_advantage_updates": sum(1 for row in group_rows if not row.get("advantage_skipped")),
            }
        summary[version] = {"epochs": by_epoch, "updates": len(updates), "rollouts": len(rollouts)}
    return summary


def write_report(train_rows, valid_rows, replays, log_summary, demo_rows):
    solvable = {row["id"] for row in train_rows if row["nominal_solvable"]}
    failing = {row["id"] for row in train_rows if not row["nominal_solvable"]}
    demo_ids = {row["id"] for row in demo_rows}
    demo_on_solvable = sum(1 for row in demo_rows if row["id"] in solvable)
    demo_ics_solvable = len({row["id"] for row in demo_rows if row["id"] in solvable})
    demo_ics_failing = len({row["id"] for row in demo_rows if row["id"] in failing})
    replay_counts = Counter(row["status"] for row in replays)
    payload = {
        "train_n": len(train_rows),
        "train_solvable": len(solvable),
        "train_failing": len(failing),
        "by_source": _bucket(train_rows, "source"),
        "by_family": _bucket(train_rows, "family"),
        "by_delay_s": _bucket(train_rows, "delay_s"),
        "by_speed_mps": _bucket(train_rows, "speed_mps"),
        "by_z_off_m": _bucket(train_rows, "z_off_m"),
        "by_phi_deg": _bucket(train_rows, "phi_deg"),
        "by_azimuth_deg": _bucket(train_rows, "azimuth_deg"),
        "by_states": _bucket(train_rows, "states"),
        "supervised_rows": len(demo_rows),
        "supervised_ics": len(demo_ids),
        "supervised_rows_on_solvable_ics": demo_on_solvable,
        "supervised_ics_solvable": demo_ics_solvable,
        "supervised_ics_failing": demo_ics_failing,
        "failing_without_demo": len(failing - demo_ids),
        "replay_status": dict(replay_counts),
        "validation_n": len(valid_rows),
        "validation_solvable": sum(1 for row in valid_rows if row["nominal_solvable"]),
        "validation_by_azimuth": _bucket(valid_rows, "azimuth_deg"),
        "validation_by_delay": _bucket(valid_rows, "delay_s"),
        "logs": log_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Nominal handoff audit",
        "",
        f"Immediate handoff succeeds on {payload['train_solvable']}/{payload['train_n']} TRAIN initial conditions and fails on {payload['train_failing']}.",
        f"Validation immediate handoff succeeds on {payload['validation_solvable']}/{payload['validation_n']}.",
        f"Of {payload['supervised_rows']} supervised trajectories on {payload['supervised_ics']} ICs, {payload['supervised_rows_on_solvable_ics']} trajectories and {payload['supervised_ics_solvable']} ICs are nominal-solvable.",
        f"Nominal-failing ICs with a demonstration: {payload['supervised_ics_failing']}. Without a demonstration: {payload['failing_without_demo']}.",
        f"Replay status on nominal-failing ICs: {payload['replay_status']}.",
        "",
        "Generation accepted a post-impact geometry entry that was outside the pinch basin. It did not require the nominal controller to fail.",
        "",
    ]
    report = ROOT / "results/archive/demo_results/diagnostics/K48_NOMINAL_HANDOFF_AUDIT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("AUDIT_DONE", json.dumps({key: payload[key] for key in ("train_solvable", "train_failing", "validation_solvable", "replay_status", "supervised_ics_solvable")}), flush=True)


def main():
    import torch
    from training.recovery_episode import make_training_env

    before = (CKPT.stat().st_mtime, CKPT.stat().st_size)
    model, _book = load_trained_k48(CKPT, torch.device("cpu"))
    env = make_training_env(torch.device("cpu"), model=model)
    ics = read_jsonl(ICS)
    if len(ics) != 312:
        raise RuntimeError(f"expected 312 TRAIN ICs, found {len(ics)}")
    scan(env, ics, OUT / "train_immediate.jsonl")
    valid_jobs = [{"id": path.stem, "snapshot": str(path.relative_to(ROOT)).replace("\\", "/")} for path in sorted(VALID.glob("ball_validation_*.pkl"))]
    scan(env, valid_jobs, OUT / "valid_immediate.jsonl")
    train_rows = [json.loads(line) for line in (OUT / "train_immediate.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    failing = {row["id"] for row in train_rows if not row["nominal_solvable"]}
    replay_failing(env, failing)
    replays = [json.loads(line) for line in (OUT / "demo_replays.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (OUT / "demo_replays.jsonl").exists() else []
    valid_rows = [json.loads(line) for line in (OUT / "valid_immediate.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    solvable = {row["id"] for row in train_rows if row["nominal_solvable"]}
    demo_rows = read_jsonl(SUPERVISED)
    if CKPT.stat().st_mtime != before[0] or CKPT.stat().st_size != before[1]:
        raise RuntimeError("audit changed the supervised checkpoint")
    write_report(train_rows, valid_rows, replays, analyze_logs(solvable), demo_rows)


if __name__ == "__main__":
    main()
