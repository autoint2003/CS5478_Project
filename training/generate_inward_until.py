"""Generate new ballistic initial conditions until inward sliding has 50 samples.

Writes only under data/inward_sliding. A row is kept when immediate handoff
fails and a legal inward trajectory then holds for 12 seconds.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.ball_impact_targeted import cylinder_first, near
from training.export_inward_training_data import DEST, propose, save_pair
from training.rebuild_k48_recovery import (
    _draw_spec,
    _grasp_now,
    _known_specs,
    _load_jsonl,
    _open_env,
    _reserved_keys,
    _scenario_key,
    _split_for_key,
    immediate_handoff,
    legal_verify,
    legalize,
)
from training.unified_k47 import load_codebook

TARGET = 50
MAX_LAUNCHES = 180


def _load_manifest():
    path = DEST / "manifest.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_manifest(rows):
    with (DEST / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    have = _load_manifest()
    if len(have) >= TARGET:
        print("ALREADY", len(have), flush=True)
        return
    env = _open_env()
    book = load_codebook()
    specs = _known_specs()
    reserved = _reserved_keys()
    draw = 250_000
    launches = 0
    added = 0
    need = TARGET - len(have)
    log_path = DEST / "generated.jsonl"
    while added < need and launches < MAX_LAUNCHES:
        spec = _draw_spec(draw)
        draw += 1
        key = _scenario_key(spec)
        if key in reserved or any(near(spec, item) for item in specs):
            continue
        split = _split_for_key(key)
        episode_id = f"inward_extra_{draw:05d}"
        spec.update({
            "split": split,
            "attempt_id": episode_id,
            "protocol": "inward_sliding_data",
        })
        from training.ball_impact_dataset import run_episode

        evidence, blob = run_episode(env, spec)
        evidence, blob = cylinder_first(evidence, blob)
        launches += 1
        specs.append(spec)
        reserved.add(key)
        if blob is None:
            print("LAUNCH", launches, episode_id, evidence.get("category"), flush=True)
            continue
        outcome = immediate_handoff(env, blob)
        if outcome.get("success"):
            print("LAUNCH", launches, episode_id, "nominal_solvable", flush=True)
            continue
        from training.history_aware_critic_ablation import restore_capture

        restore_capture(env, blob["capture"])
        _pack, bilateral = _grasp_now(env.sim)
        if not bilateral:
            print("LAUNCH", launches, episode_id, "not_bilateral", outcome.get("reason"), flush=True)
            continue
        proposed = propose(env, blob)
        tokens, residuals, decoded, packed = legalize(proposed, book)
        if len(decoded) == 0 or float(packed["fraction_within_0.25"]) < 1.0 - 1e-8:
            print("LAUNCH", launches, episode_id, "illegal_or_empty", flush=True)
            continue
        out, observations, executed, contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded)
        ok = bool(out.get("success") and qpos_error <= 1e-8 and phase_ok)
        print("LAUNCH", launches, episode_id, split, out.get("reason"), "steps", len(executed), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "id": episode_id, "split": split, "reason": out.get("reason"), "success": ok,
            }) + "\n")
        if not ok:
            continue
        folder = DEST / split
        stem = f"{episode_id}__inward_reseat"
        continuous, token_path = save_pair(
            folder, stem, executed, observations, contacts, tokens, residuals, out.get("handoff_obs"),
        )
        snap_dst = DEST / "snapshots" / split / f"{episode_id}.pkl"
        snap_dst.parent.mkdir(parents=True, exist_ok=True)
        with snap_dst.open("wb") as handle:
            pickle.dump(blob, handle)
        have.append({
            "id": episode_id,
            "split": split,
            "strategy": "inward_reseat",
            "schedule": "inward_reseat",
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
            "azimuth_deg": spec["azimuth_deg"],
            "speed_mps": spec["speed_mps"],
            "z_off_m": spec["z_off_m"],
            "phi_deg": spec["phi_deg"],
            "delay_s": spec["delay_s"],
        })
        _write_manifest(have)
        added += 1
        print("ADDED", len(have), split, episode_id, flush=True)
    summary = {
        "train": sum(1 for row in have if row["split"] == "train"),
        "valid": sum(1 for row in have if row["split"] == "valid"),
        "total": len(have),
        "added_generated": added,
        "launches": launches,
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("GENERATE", summary, flush=True)


if __name__ == "__main__":
    main()
