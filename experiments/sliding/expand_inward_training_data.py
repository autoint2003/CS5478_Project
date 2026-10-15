"""Add inward-sliding successes until data/inward_sliding has 50 rows.

Uses initial conditions already rejected by the single-sign controller. Each
accepted row is a legal K48 replay that the unchanged nominal controller holds
for 12 seconds. Does not modify k48_recovery_v3.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sliding.export_inward_training_data import DEST, save_pair
from data_generation.strategies.k48_inward_sliding_search import (
    OLD,
    SEEDS,
    _priority_cuts,
    _trace_summary,
    _verify_prefix,
    build_schedule,
    rollout,
    row_snapshot,
)
from data_generation.rebuild_k48_recovery import _load_blob, _load_jsonl, _open_env
from training.k47_codebook import load_codebook

TARGET = 50
REPLAY = ROOT / "results/archive/demo_results/diagnostics/raw/k48_inward_sliding_search_v1/old_gate_replay.jsonl"
SEED_BY_NAME = {item["name"]: item for item in SEEDS}


def _manifest():
    path = DEST / "manifest.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _attempts(ex: float, gates: list[str]):
    """Legal schedules. The sign that already finished a relaxation is not repeated."""
    if ex >= 0.0:
        seq = [
            ("hist_simultaneous", 1.0),
            ("slow_pitch", -1.0),
            ("slip_grip_m5", 1.0),
            ("slow_pitch", 1.0),
            ("brake_then_relax", 1.0),
        ]
    else:
        seq = [
            ("slow_pitch", -1.0),
            ("hist_simultaneous", 1.0),
            ("slow_pitch", 1.0),
            ("brake_then_relax", 1.0),
        ]
    finished = "relaxation_duration_cutoff" in gates or "offset_inside_3mm" in gates
    if finished:
        seq = seq[1:]
    return seq


def _candidates():
    replay = {row["id"]: row for row in _load_jsonl(REPLAY)}
    logged = {row["id"]: row for row in _load_jsonl(OLD / "inward_reseat" / "search.jsonl")}
    have = {row["id"] for row in _manifest()}
    rows = []
    for episode_id, row in replay.items():
        if episode_id in have:
            continue
        if (row.get("ungated_verify") or {}).get("success"):
            continue
        info = logged.get(episode_id, {})
        rows.append({
            "id": episode_id,
            "split": row.get("split") or "train",
            "ex": float(info.get("ex_start") or 0.0),
            "gates": list(row.get("gates") or []),
        })
    rows.sort(key=lambda item: abs(item["ex"]))
    return rows


def main():
    have = _manifest()
    if len(have) >= TARGET:
        print("ALREADY", len(have), flush=True)
        return
    need = TARGET - len(have)
    pool = _candidates()
    print("NEED", need, "POOL", len(pool), flush=True)
    candidates = {row["id"]: row for row in _load_jsonl(OLD / "candidates.jsonl")}
    env = _open_env()
    book = load_codebook()
    added = []
    verifies = 0
    log_path = DEST / "expansion.jsonl"
    for item in pool:
        if len(added) >= need:
            break
        blob = _load_blob(row_snapshot(item["id"]))
        for seed_name, wy_sign in _attempts(item["ex"], item["gates"]):
            if len(added) >= need:
                break
            actions, cuts, illegal = build_schedule(SEED_BY_NAME[seed_name], wy_sign, book)
            name = f"{seed_name}_{'pos' if wy_sign > 0 else 'neg'}"
            if illegal or actions is None:
                continue
            _executed, rows, drop_step = rollout(env, blob, actions)
            summary = _trace_summary(rows, drop_step, cuts)
            trial = {"actions": actions, "cuts": cuts, "summary": summary}
            cuts_now = _priority_cuts(trial)
            if not cuts_now:
                continue
            cut = cuts_now[0]
            verifies += 1
            outcome = _verify_prefix(env, book, blob, actions[: int(cut["end"])])
            print(
                "TRY", item["id"], name, cut["tag"], outcome.get("reason"),
                "inward_mm", round(summary["inward_mm"], 2), "s", outcome.get("seconds"),
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "id": item["id"], "schedule": name, "reason": outcome.get("reason"),
                    "success": bool(outcome.get("success")), "inward_mm": summary["inward_mm"],
                }) + "\n")
            if not outcome.get("success"):
                continue
            split = "valid" if item["split"] == "valid" else "train"
            folder = DEST / split
            stem = f"{item['id']}__inward_reseat"
            tokens_note = outcome
            from data_generation.rebuild_k48_recovery import legalize
            tokens, residuals, decoded, packed = legalize(actions[: int(cut["end"])], book)
            out, observations, executed, contacts, qpos_error, phase_ok = __import__(
                "data_generation.rebuild_k48_recovery", fromlist=["legal_verify"]
            ).legal_verify(env, blob, decoded)
            if not (out.get("success") and qpos_error <= 1e-8 and phase_ok):
                print("RECHECK_FAIL", item["id"], out.get("reason"), flush=True)
                continue
            continuous, token_path = save_pair(
                folder, stem, executed, observations, contacts, tokens, residuals, out.get("handoff_obs"),
            )
            snap_src = ROOT / candidates[item["id"]]["snapshot"]
            snap_dst = DEST / "snapshots" / split / f"{item['id']}.pkl"
            snap_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snap_src, snap_dst)
            record = {
                "id": item["id"],
                "split": split,
                "strategy": "inward_reseat",
                "schedule": name,
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
                "inward_mm": summary["inward_mm"],
                "trajectory": str(continuous.relative_to(ROOT)).replace("\\", "/"),
                "tokens": str(token_path.relative_to(ROOT)).replace("\\", "/"),
                "snapshot": str(snap_dst.relative_to(ROOT)).replace("\\", "/"),
                "source_snapshot": candidates[item["id"]]["snapshot"],
            }
            have.append(record)
            added.append(record)
            with (DEST / "manifest.jsonl").open("w", encoding="utf-8") as handle:
                for row in have:
                    handle.write(json.dumps(row) + "\n")
            print("ADDED", len(have), split, item["id"], name, flush=True)
            break
    summary = {
        "train": sum(1 for row in have if row["split"] == "train"),
        "valid": sum(1 for row in have if row["split"] == "valid"),
        "total": len(have),
        "added": len(added),
        "verifies": verifies,
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EXPAND", summary, flush=True)


if __name__ == "__main__":
    main()
