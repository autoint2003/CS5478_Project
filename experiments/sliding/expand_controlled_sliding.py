"""Expand data/controlled_sliding to 50 legal K48 successes.

Keeps the existing rows. Adds a sample only when immediate handoff fails and a
codebook-legal trajectory then holds for 12 seconds. Writes under data/ and
refreshes the training entry. Does not modify k48_recovery_v3 or train.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ball_impact.ball_impact_targeted import cylinder_first, near
from experiments.sliding.export_inward_training_data import save_pair
from data_generation.strategies.k48_controlled_slide_search import SEEDS, _measure, build_schedule, rollout
from data_generation.rebuild_k48_recovery import (
    _draw_spec,
    _known_specs,
    _load_blob,
    _load_jsonl,
    _open_env,
    _reserved_keys,
    _scenario_key,
    _split_for_key,
    _weights,
    immediate_handoff,
    legal_verify,
    legalize,
    propose_actions,
    resolve_stored_path,
)
from training.unified_k47 import load_codebook

TARGET = 50
MAX_LAUNCHES = 360
DRAW_START = 260_000
DEST = ROOT / "data" / "controlled_sliding"
DATA = ROOT / "data"
OLD = ROOT / "results/archive/demo_results/diagnostics/raw/k48_recovery_v3"
SEARCH = ROOT / "results/archive/demo_results/diagnostics/raw/k48_controlled_slide_search_v1/search.jsonl"
SEED_BY_NAME = {item["name"]: item for item in SEEDS}

# Cut that succeeded for that family in the bounded search. A fallen prefix is
# not handed to the 12-second check.
PLAN = {
    "pos": (
        ("center6_pos", "return"),
        ("short_pos", "relax"),
        ("no_return_pos", "retighten"),
    ),
    "neg": (
        ("short_neg", "retighten"),
        ("center6_neg", "brake"),
    ),
}


def _rows(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_manifest(rows):
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _log(row):
    with (DEST / "expansion.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _prefix(seed_name, tag, book):
    actions, cuts, illegal = build_schedule(SEED_BY_NAME[seed_name], book)
    if illegal or actions is None:
        return None
    for cut in cuts:
        if cut["tag"] == tag:
            return actions[: int(cut["end"])]
    return None


def _accept(env, book, blob, actions):
    tokens, residuals, decoded, packed = legalize(actions, book)
    if len(decoded) == 0 or float(packed["fraction_within_0.25"]) < 1.0 - 1e-8:
        return None
    out, observations, executed, contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded)
    if not (out.get("success") and qpos_error <= 1e-8 and phase_ok):
        return {"ok": False, "reason": out.get("reason")}
    return {
        "ok": True,
        "reason": "SUCCESS_12S",
        "tokens": tokens,
        "residuals": residuals,
        "executed": executed,
        "observations": observations,
        "contacts": contacts,
        "handoff_obs": out.get("handoff_obs"),
        "qpos_error": qpos_error,
        "phase_ok": phase_ok,
        "fraction": packed["fraction_within_0.25"],
        "max_linf": packed["max_linf"],
        "recovery_elapsed": out.get("recovery_elapsed"),
        "nominal_elapsed": out.get("nominal_elapsed"),
        "lift_completed": out.get("lift_completed"),
    }


def _try_seed(env, book, blob, seed_name, tag):
    actions, cuts, illegal = build_schedule(SEED_BY_NAME[seed_name], book)
    if illegal or actions is None:
        return None
    _executed, _rows_trace, drop_step = rollout(env, blob, actions)
    end = next(int(cut["end"]) for cut in cuts if cut["tag"] == tag)
    if drop_step is not None and end >= drop_step:
        return {"ok": False, "reason": "DROP_DURING_RECOVERY", "schedule": seed_name}
    packed = _accept(env, book, blob, actions[:end])
    if packed is None:
        return {"ok": False, "reason": "illegal", "schedule": seed_name}
    packed["schedule"] = seed_name
    packed["handoff_tag"] = tag
    return packed


def _try_baseline(env, book, blob):
    spec = {"name": "controlled_slide", "ref": blob["ref"]}
    actions, stop = propose_actions(env, blob, "controlled_slide", spec)
    if len(actions) == 0 or stop == "recovery_failed":
        return {"ok": False, "reason": stop or "no_actions", "schedule": "deterministic_baseline"}
    packed = _accept(env, book, blob, actions)
    if packed is None:
        return {"ok": False, "reason": "illegal", "schedule": "deterministic_baseline"}
    packed["schedule"] = "deterministic_baseline"
    packed["handoff_tag"] = "recorded_end"
    return packed


def _search_one(env, book, blob, rhx, bilateral):
    family = "pos" if rhx >= 0.0 else "neg"
    for seed_name, tag in PLAN[family]:
        outcome = _try_seed(env, book, blob, seed_name, tag)
        if outcome and outcome.get("ok"):
            return outcome
    if bilateral:
        outcome = _try_baseline(env, book, blob)
        if outcome and outcome.get("ok"):
            return outcome
    return None


def _store(have, episode_id, split, blob, outcome, source_snapshot, provenance, spec=None):
    folder = DEST / split
    stem = f"{episode_id}__controlled_slide__verified"
    continuous, token_path = save_pair(
        folder, stem, outcome["executed"], outcome["observations"], outcome["contacts"],
        outcome["tokens"], outcome["residuals"], outcome["handoff_obs"],
    )
    snap_dst = DATA / "snapshots" / split / f"{episode_id}.pkl"
    snap_dst.parent.mkdir(parents=True, exist_ok=True)
    if source_snapshot is not None and not snap_dst.exists():
        shutil.copy2(resolve_stored_path(source_snapshot), snap_dst)
    elif source_snapshot is None:
        with snap_dst.open("wb") as handle:
            pickle.dump(blob, handle)
    row = {
        "id": episode_id,
        "snapshot": str(snap_dst.relative_to(ROOT)).replace("\\", "/"),
        "strategy": "controlled_slide",
        "family": "sliding",
        "mechanism": "contact_preserving_slide",
        "continuous_npz": str(continuous.relative_to(ROOT)).replace("\\", "/"),
        "token_residual_npz": str(token_path.relative_to(ROOT)).replace("\\", "/"),
        "handoff_label": "early_handoff",
        "source_kind": "search",
        "nominal_class": "RECOVERY_CANDIDATE",
        "verified_success": True,
        "success": True,
        "reason": "SUCCESS_12S",
        "split": split,
        "provenance": provenance,
        "schedule": outcome.get("schedule"),
        "handoff_tag": outcome.get("handoff_tag"),
        "recovery_elapsed": outcome.get("recovery_elapsed"),
        "nominal_elapsed": outcome.get("nominal_elapsed"),
        "lift_completed": outcome.get("lift_completed"),
        "fraction_within_0.25": outcome.get("fraction"),
        "max_linf": outcome.get("max_linf"),
        "steps": int(len(outcome["executed"])),
        "qpos_error": outcome.get("qpos_error"),
        "phase_restored": outcome.get("phase_ok"),
    }
    if source_snapshot is not None:
        row["source_snapshot"] = source_snapshot
    if spec is not None:
        row.update({
            "azimuth_deg": spec.get("azimuth_deg"),
            "speed_mps": spec.get("speed_mps"),
            "z_off_m": spec.get("z_off_m"),
            "phi_deg": spec.get("phi_deg"),
            "delay_s": spec.get("delay_s"),
            "scenario_key": list(_scenario_key(spec)),
        })
    have.append(row)
    _write_manifest(have)
    return row


def _known_wins():
    chosen = {}
    for row in _rows(SEARCH):
        if not row.get("success"):
            continue
        current = chosen.get(row["id"])
        if current is None or (current.get("stage") != "A" and row.get("stage") == "A"):
            chosen[row["id"]] = row
    return chosen


def _replay_known(env, book, have):
    known = _known_wins()
    candidates = {row["id"]: row for row in _load_jsonl(OLD / "candidates.jsonl")}
    have_ids = {row["id"] for row in have}
    for episode_id, win in known.items():
        if len(have) >= TARGET or episode_id in have_ids:
            continue
        candidate = candidates.get(episode_id)
        if candidate is None:
            continue
        blob = _load_blob(candidate["snapshot"])
        schedule = str(win.get("schedule") or "")
        if schedule == "deterministic_baseline":
            outcome = _try_baseline(env, book, blob)
        else:
            seed_name = schedule.replace("_transfer", "")
            tag = win.get("handoff_tag") or "return"
            prefix = _prefix(seed_name, tag, book)
            outcome = None if prefix is None else _accept(env, book, blob, prefix)
            if outcome is not None:
                outcome["schedule"] = schedule
                outcome["handoff_tag"] = tag
        if not outcome or not outcome.get("ok"):
            print("REPLAY_FAIL", episode_id, None if outcome is None else outcome.get("reason"), flush=True)
            _log({"id": episode_id, "phase": "replay", "success": False, "reason": None if outcome is None else outcome.get("reason")})
            continue
        split = candidate.get("split") or "train"
        _store(have, episode_id, split, blob, outcome, candidate["snapshot"], candidate.get("provenance") or "controlled_slide_search")
        have_ids.add(episode_id)
        print("ADDED", len(have), split, episode_id, outcome.get("schedule"), flush=True)
        _log({"id": episode_id, "phase": "replay", "success": True, "schedule": outcome.get("schedule")})


def _candidate_pool(env, have_ids):
    from training.history_aware_critic_ablation import restore_capture

    pool = []
    for row in _load_jsonl(OLD / "candidates.jsonl"):
        if row.get("nominal_class") != "RECOVERY_CANDIDATE" or row["id"] in have_ids:
            continue
        blob = _load_blob(row["snapshot"])
        restore_capture(env, blob["capture"])
        state = _measure(env.sim, blob["ref"])
        bilateral = state["nL"] > 0 and state["nR"] > 0
        pool.append((0 if bilateral else 1, 0 if row.get("split") == "valid" else 1, -abs(state["rhx"]), row, state["rhx"], bilateral))
    pool.sort(key=lambda item: item[:3])
    return pool


def _search_candidates(env, book, have):
    tried = {row["id"] for row in _rows(DEST / "expansion.jsonl") if row.get("phase") == "candidate"}
    pool = _candidate_pool(env, {row["id"] for row in have})
    print("CANDIDATES", len(pool), flush=True)
    for _rank, _split_key, _offset, row, rhx, bilateral in pool:
        if len(have) >= TARGET:
            return
        if row["id"] in tried:
            continue
        blob = _load_blob(row["snapshot"])
        outcome = _search_one(env, book, blob, rhx, bilateral)
        _log({
            "id": row["id"], "phase": "candidate", "success": bool(outcome),
            "schedule": None if outcome is None else outcome.get("schedule"),
            "bilateral": bilateral, "rhx": rhx,
        })
        if outcome is None:
            print("MISS", row["id"], "bilateral" if bilateral else "open", round(1e3 * rhx, 1), flush=True)
            continue
        split = row.get("split") or "train"
        _store(have, row["id"], split, blob, outcome, row["snapshot"], row.get("provenance") or "controlled_slide_search")
        print("ADDED", len(have), split, row["id"], outcome.get("schedule"), flush=True)


def _generate(env, book, have):
    state_path = DEST / "expand_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"next_draw": DRAW_START}
    draw = int(state.get("next_draw", DRAW_START))
    specs = _known_specs()
    reserved = _reserved_keys()
    launches = 0
    from data_generation.ball_impact_dataset import run_episode
    from training.history_aware_critic_ablation import restore_capture

    while len(have) < TARGET and launches < MAX_LAUNCHES:
        spec = _draw_spec(draw)
        draw += 1
        state["next_draw"] = draw
        state_path.write_text(json.dumps(state), encoding="utf-8")
        key = _scenario_key(spec)
        if key in reserved or any(near(spec, item) for item in specs):
            continue
        split = _split_for_key(key)
        episode_id = f"slide_extra_{draw:05d}"
        spec.update({"split": split, "attempt_id": episode_id, "protocol": "controlled_sliding_data"})
        evidence, blob = run_episode(env, spec)
        evidence, blob = cylinder_first(evidence, blob)
        launches += 1
        specs.append(spec)
        reserved.add(key)
        if blob is None:
            print("LAUNCH", launches, episode_id, evidence.get("category"), flush=True)
            _log({"id": episode_id, "phase": "launch", "success": False, "reason": evidence.get("category")})
            continue
        immediate = immediate_handoff(env, blob)
        if immediate.get("success"):
            print("LAUNCH", launches, episode_id, "nominal_solvable", flush=True)
            _log({"id": episode_id, "phase": "launch", "success": False, "reason": "nominal_solvable"})
            continue
        restore_capture(env, blob["capture"])
        state_now = _measure(env.sim, blob["ref"])
        bilateral = state_now["nL"] > 0 and state_now["nR"] > 0
        outcome = _search_one(env, book, blob, state_now["rhx"], bilateral)
        if outcome is None:
            print("LAUNCH", launches, episode_id, split, "miss", flush=True)
            _log({"id": episode_id, "phase": "launch", "success": False, "reason": "miss"})
            continue
        _store(have, episode_id, split, blob, outcome, None, "controlled_sliding_generated", spec)
        print("ADDED", len(have), split, episode_id, outcome.get("schedule"), flush=True)
        _log({"id": episode_id, "phase": "launch", "success": True, "schedule": outcome.get("schedule")})
    return launches


def _sync_entry(manifest):
    train = _rows(DATA / "supervised_targets.jsonl")
    valid = _rows(DATA / "valid_demonstrations.jsonl")
    present = {(row["id"], row.get("continuous_npz")) for row in train + valid}
    for row in manifest:
        key = (row["id"], row["continuous_npz"])
        if key in present:
            continue
        item = {
            "id": row["id"],
            "snapshot": row["snapshot"],
            "strategy": "controlled_slide",
            "family": "sliding",
            "mechanism": "contact_preserving_slide",
            "continuous_npz": row["continuous_npz"],
            "token_residual_npz": row["token_residual_npz"],
            "handoff_label": row.get("handoff_label") or "early_handoff",
            "source_kind": row.get("source_kind") or "search",
            "nominal_class": "RECOVERY_CANDIDATE",
            "verified_success": True,
            "success": True,
            "reason": "SUCCESS_12S",
            "split": row["split"],
            "provenance": row.get("provenance"),
            "schedule": row.get("schedule"),
            "recovery_elapsed": row.get("recovery_elapsed"),
            "nominal_elapsed": row.get("nominal_elapsed"),
        }
        if row.get("source_snapshot"):
            item["source_snapshot"] = row["source_snapshot"]
        if row["split"] == "valid":
            valid.append(item)
        else:
            train.append(item)
        present.add(key)
    recovery = [row for row in train if row.get("nominal_class") == "RECOVERY_CANDIDATE"]
    weights = _weights(recovery)
    for row in train:
        if row.get("nominal_class") == "NOMINAL_SOLVABLE":
            row["sample_weight"] = 0.0
        elif row.get("nominal_class") == "RECOVERY_CANDIDATE":
            row["sample_weight"] = weights[(row["id"], row.get("continuous_npz"))]
    weight_for = {(row["id"], row.get("continuous_npz")): row.get("sample_weight") for row in train}
    for row in manifest:
        if row["split"] != "valid":
            row["sample_weight"] = weight_for.get((row["id"], row["continuous_npz"]), row.get("sample_weight"))
    _write_manifest(manifest)
    for path, rows in (
        (DATA / "supervised_targets.jsonl", train),
        (DATA / "valid_demonstrations.jsonl", valid),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    recovery_rows = _rows(DATA / "valid_recovery.jsonl")
    have_valid = {row["id"] for row in recovery_rows}
    for row in valid:
        if row.get("strategy") != "controlled_slide" or row["id"] in have_valid:
            continue
        have_valid.add(row["id"])
        recovery_rows.append({
            "id": row["id"],
            "snapshot": row["snapshot"],
            "pool": "recovery",
            "nominal_class": "RECOVERY_CANDIDATE",
            "split": "valid",
            "horizon": "recovery_limit_4s_then_nominal_12s",
            "recovery_limit_s": 4.0,
            "success": False,
            "provenance": row.get("provenance"),
            "strategy": "controlled_slide",
        })
    with (DATA / "valid_recovery.jsonl").open("w", encoding="utf-8") as handle:
        for row in recovery_rows:
            handle.write(json.dumps(row) + "\n")
    entry = json.loads((DATA / "entry.json").read_text(encoding="utf-8"))
    entry["counts"] = {
        "supervised_train": len(train),
        "valid_demonstrations": len(valid),
        "grpo": len(_rows(DATA / "grpo_initial_conditions.jsonl")),
        "valid_recovery": len(recovery_rows),
        "by_strategy_train": dict(Counter(row.get("strategy") for row in train)),
        "by_strategy_valid": dict(Counter(row.get("strategy") for row in valid)),
    }
    (DATA / "entry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")


def main():
    have = _rows(DEST / "manifest.jsonl")
    if len(have) >= TARGET:
        print("ALREADY", len(have), flush=True)
        return
    env = _open_env()
    book = load_codebook()
    before = len(have)
    _replay_known(env, book, have)
    if len(have) < TARGET:
        _search_candidates(env, book, have)
    launches = 0
    if len(have) < TARGET:
        launches = _generate(env, book, have)
    _sync_entry(have)
    summary = {
        "train": sum(1 for row in have if row["split"] == "train"),
        "valid": sum(1 for row in have if row["split"] == "valid"),
        "total": len(have),
        "added": len(have) - before,
        "launches": launches,
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EXPAND", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
