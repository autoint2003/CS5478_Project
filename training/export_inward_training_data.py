"""Export confirmed inward-sliding successes into data/inward_sliding.

Replays each initial condition with the current legal controller and keeps the
row only when the unchanged nominal controller holds for 12 seconds. Does not
modify demo_results/diagnostics/raw/k48_recovery_v3.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.horizon_outcome import recovery_failed
from training.k48_inward_sliding_search import row_snapshot
from training.privileged_ball_impact_search import step_legal
from training.rebuild_k48_recovery import (
    CAP_STEPS,
    LIMIT_S,
    _load_blob,
    _load_jsonl,
    _open_env,
    inward_reseat_action,
    legal_verify,
    legalize,
)
from training.unified_k47 import load_codebook

OLD = ROOT / "demo_results/diagnostics/raw/k48_recovery_v3"
REPLAY = ROOT / "demo_results/diagnostics/raw/k48_inward_sliding_search_v1/old_gate_replay.jsonl"
DEST = ROOT / "data" / "inward_sliding"


def confirmed_ids():
    found = {}
    for row in _load_jsonl(REPLAY):
        if (row.get("ungated_verify") or {}).get("success"):
            found[row["id"]] = row["split"]
    for row in _load_jsonl(OLD / "inward_reseat" / "verified_trajectories.jsonl"):
        if row.get("success"):
            found[row["id"]] = row.get("split") or "train"
    return found


def propose(env, blob):
    from training.history_aware_critic_ablation import restore_capture

    restore_capture(env, blob["capture"])
    spec = {"name": "inward_reseat", "ref": blob["ref"]}
    memory = {}
    actions = []
    lost_low = 0
    t0 = float(env.sim.data.time)
    for _index in range(CAP_STEPS):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            break
        action, phase = inward_reseat_action(env.sim, spec, memory)
        if phase == "done":
            break
        step_legal(env.sim, action)
        actions.append(np.asarray(action, np.float32))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            break
    if not actions:
        return np.zeros((0, 7), np.float32)
    return np.stack(actions, axis=0)


def save_pair(folder, stem, executed, observations, contacts, tokens, residuals, handoff_obs):
    folder.mkdir(parents=True, exist_ok=True)
    continuous = folder / f"{stem}.npz"
    tokens_path = folder / f"{stem}__tokens.npz"
    payload = {
        "actions": np.asarray(executed, np.float32),
        "observations": np.asarray(observations, np.float32),
        "timestamps": np.asarray([row["t"] for row in contacts], np.float64),
        "nL": np.asarray([row["nL"] for row in contacts], np.int32),
        "nR": np.asarray([row["nR"] for row in contacts], np.int32),
        "obj_z": np.asarray([row["z"] for row in contacts], np.float32),
    }
    if handoff_obs is not None:
        payload["handoff_obs"] = np.asarray(handoff_obs, np.float32)
    np.savez_compressed(continuous, **payload)
    np.savez_compressed(
        tokens_path,
        tokens=np.asarray(tokens, np.int64),
        residuals=np.asarray(residuals, np.float32),
    )
    return continuous, tokens_path


def main():
    ids = confirmed_ids()
    candidates = {row["id"]: row for row in _load_jsonl(OLD / "candidates.jsonl")}
    env = _open_env()
    book = load_codebook()
    manifest = []
    failed = []
    for episode_id in sorted(ids):
        split = "valid" if ids[episode_id] == "valid" else "train"
        blob = _load_blob(row_snapshot(episode_id))
        proposed = propose(env, blob)
        tokens, residuals, decoded, packed = legalize(proposed, book)
        if float(packed["fraction_within_0.25"]) < 1.0 - 1e-8 or len(decoded) == 0:
            failed.append((episode_id, "illegal"))
            print("SKIP", episode_id, "illegal", flush=True)
            continue
        out, observations, executed, contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded)
        if not (out.get("success") and qpos_error <= 1e-8 and phase_ok):
            failed.append((episode_id, out.get("reason")))
            print("SKIP", episode_id, out.get("reason"), flush=True)
            continue
        folder = DEST / split
        stem = f"{episode_id}__inward_reseat"
        continuous, token_path = save_pair(
            folder, stem, executed, observations, contacts, tokens, residuals, out.get("handoff_obs"),
        )
        snap_src = ROOT / candidates[episode_id]["snapshot"]
        snap_dst = DEST / "snapshots" / split / f"{episode_id}.pkl"
        snap_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snap_src, snap_dst)
        row = {
            "id": episode_id,
            "split": split,
            "strategy": "inward_reseat",
            "reason": "SUCCESS_12S",
            "success": True,
            "qpos_error": qpos_error,
            "phase_restored": phase_ok,
            "fraction_within_0.25": packed["fraction_within_0.25"],
            "max_linf": packed["max_linf"],
            "recovery_elapsed": out.get("recovery_elapsed"),
            "nominal_elapsed": out.get("nominal_elapsed"),
            "lift_completed": out.get("lift_completed"),
            "steps": int(len(executed)),
            "trajectory": str(continuous.relative_to(ROOT)).replace("\\", "/"),
            "tokens": str(token_path.relative_to(ROOT)).replace("\\", "/"),
            "snapshot": str(snap_dst.relative_to(ROOT)).replace("\\", "/"),
            "source_snapshot": candidates[episode_id]["snapshot"],
        }
        manifest.append(row)
        print("WROTE", split, episode_id, "steps", row["steps"], flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "train": sum(1 for row in manifest if row["split"] == "train"),
        "valid": sum(1 for row in manifest if row["split"] == "valid"),
        "failed": failed,
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EXPORT", summary, flush=True)


if __name__ == "__main__":
    main()
