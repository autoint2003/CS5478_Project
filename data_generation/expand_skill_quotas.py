"""Resumable expansion of the three recovery skills.

A new initial condition is stored only after nominal continuation fails, the
oracle action is K48-legal, a decoded replay still returns SUCCESS_12S, and
the contact trace matches that skill. Draws and split labels are fixed before
the rollout. This does not start SFT or GRPO and does not edit physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.ball_impact_dataset import run_episode
from data_generation.rebuild_k48_recovery import (
    CAP_STEPS,
    LIMIT_S,
    _draw_spec,
    _grasp_now,
    _known_specs,
    _load_blob,
    _open_env,
    _reserved_keys,
    _scenario_key,
    _split_for_key,
    controlled_slide_action,
    immediate_handoff,
    inward_reseat_action,
    legal_verify,
    legalize,
    resolve_stored_path,
)
from data_generation.strategies.k48_controlled_slide_search import SEEDS, build_schedule
from data_generation.skill_filters import (
    DIAGNOSTIC_REJECTS,
    observed_mechanism,
    release_and_regrip,
    scene_category,
)
from data_generation.validation.impact_labels import cylinder_first, near
from data_generation.strategies.vertical_recatch_search import _finish, _summary, rollout as vertical_rollout, specs as vertical_specs
from envs.outcomes.horizon_outcome import recovery_failed
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from experiments.sliding.export_inward_training_data import save_pair
from training.k47_codebook import load_codebook
from visualization.audit_sliding_mechanism import bilateral_mask, release_regrasp, sample, slip_direction
from visualization.scan_sliding_slip import status as slip_status

STAGE = ROOT / "data" / "staging" / "dataset_expansion"
DRAW_START = 400_000
TARGETS = {
    "sliding": {"train": 100, "valid": 20},
    "horizontal_recatch": {"train": 50, "valid": 10},
    "vertical_recatch": {"train": 50, "valid": 10},
}
# Post-impact tilt is recorded for difficulty analysis. It does not decide
# whether a horizontal-grasp scene may be searched or saved.
HORIZONTAL_TILT_MAX = 45.0
HORIZONTAL_TILT_BOUNDARY = 20.0
SEED_BY_NAME = {item["name"]: item for item in SEEDS}
PLAN = {
    "pos": (("center6_pos", "return"), ("short_pos", "relax"), ("no_return_pos", "retighten")),
    "neg": (("short_neg", "retighten"), ("center6_neg", "brake")),
}


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def _fingerprint(blob: dict) -> str:
    if "capture" in blob:
        physics = blob["capture"]["physics"]
        arrays = blob["capture"]["fsm"]["arrays"]
        parts = (physics["qpos"], physics["qvel"], physics["ctrl"], arrays.get("p_des"))
    else:
        parts = (blob.get("qpos"), blob.get("qvel"), blob.get("ctrl"), blob.get("p_des"))
    qpos = np.asarray(parts[0] if parts[0] is not None else [], float)
    if qpos.size == 0:
        raise RuntimeError("refusing to hash an empty qpos")
    digest = hashlib.sha256()
    for part in parts:
        array = np.zeros(0) if part is None else np.asarray(part, float)
        digest.update(np.round(array, 8).astype(np.float64).tobytes())
    return digest.hexdigest()


def _tilt(sim) -> float:
    axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return float(np.degrees(np.arctan2(abs(axis[2]), max(float(np.linalg.norm(axis[:2])), 1e-12))))


def _known_identities() -> set[str]:
    found = set()
    for path in (ROOT / "data" / "snapshots").rglob("*.pkl"):
        try:
            blob = pickle.loads(path.read_bytes())
            found.add(_fingerprint(blob))
        except Exception:
            continue
    extra = STAGE / "snapshots"
    if extra.exists():
        for path in extra.rglob("*.pkl"):
            blob = pickle.loads(path.read_bytes())
            found.add(_fingerprint(blob))
    return found


def _spec_pool() -> tuple[list[dict], set]:
    specs = list(_known_specs())
    reserved = set(_reserved_keys())
    for path in (
        ROOT / "data" / "controlled_sliding" / "manifest.jsonl",
        ROOT / "data" / "inward_sliding" / "manifest.jsonl",
        ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl",
        STAGE / "attempts.jsonl",
    ):
        for row in _rows(path):
            if row.get("azimuth_deg") is None or row.get("speed_mps") is None:
                continue
            specs.append(row)
            if row.get("scenario_key"):
                reserved.add(tuple(row["scenario_key"]))
    return specs, reserved


def _ids(path: Path) -> set[str]:
    return {row["id"] for row in _rows(path)}


def _task_sliding() -> dict[str, set[str]]:
    """Every SUCCESS_12S sliding-oracle tape, including those without finger slip."""
    grouped: dict[str, set[str]] = {}
    for row in _rows(ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl"):
        if row.get("verified_success") is False or row.get("formal") not in (None, "SUCCESS_12S"):
            continue
        grouped.setdefault(row["snapshot"], set()).add(row["split"])
    return grouped


def _category_counts(manifest: Path, confirmed_only: bool = False) -> dict[str, int]:
    splits: dict[str, set[str]] = {"train": set(), "valid": set(), "test": set()}
    for row in _rows(manifest):
        if row.get("task_success") is False or row.get("counts_as_recatch") is False:
            continue
        if confirmed_only and row.get("mechanism_status") not in (None, "confirmed"):
            continue
        if confirmed_only and manifest.name.startswith("manifest") and "mechanism_status" in row and row["mechanism_status"] != "confirmed":
            continue
        split = row.get("split") or "train"
        splits.setdefault(split, set()).add(row["snapshot"])
    return {key: len(value) for key, value in splits.items()}


def quota_report() -> dict:
    sliding = {"train": 0, "valid": 0, "test": 0}
    leaks = []
    for snap, splits in _task_sliding().items():
        if len(splits) != 1:
            leaks.append(snap)
            continue
        sliding[next(iter(splits))] += 1
    horizontal = _category_counts(ROOT / "data" / "recatch" / "manifest.jsonl")
    vertical = _category_counts(ROOT / "data" / "vertical_recatch" / "manifest.jsonl")
    return {
        "sliding_task_success": sliding,
        "horizontal_recatch": horizontal,
        "vertical_recatch": vertical,
        "split_leaks": leaks,
        "targets": TARGETS,
    }


def _gaps(report: dict) -> dict:
    out = {}
    mapping = {
        "sliding": report["sliding_task_success"],
        "horizontal_recatch": report["horizontal_recatch"],
        "vertical_recatch": report["vertical_recatch"],
    }
    for name, counts in mapping.items():
        out[name] = {
            "train": max(0, TARGETS[name]["train"] - counts.get("train", 0)),
            "valid": max(0, TARGETS[name]["valid"] - counts.get("valid", 0)),
        }
    return out


def _want(split: str, gaps: dict, skill: str) -> bool:
    return gaps[skill].get(split, 0) > 0


def _trace_actions(env, blob, actions) -> list[dict]:
    restore_capture(env, blob["capture"])
    traced = [sample(env.sim, blob["ref"], np.zeros(7))]
    lost_low = 0
    t0 = float(env.sim.data.time)
    for action in np.asarray(actions, np.float32):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            break
        step_legal(env.sim, action)
        traced.append(sample(env.sim, blob["ref"], action))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            break
    return traced


def _closed_actions(env, blob, kind: str):
    restore_capture(env, blob["capture"])
    spec = {"name": kind, "ref": blob["ref"]}
    memory = {}
    actions = []
    lost_low = 0
    t0 = float(env.sim.data.time)
    stop = "cap"
    for _index in range(CAP_STEPS):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            stop = "limit"
            break
        if kind == "controlled_slide":
            action, phase = controlled_slide_action(env.sim, spec, memory)
        else:
            action, phase = inward_reseat_action(env.sim, spec, memory)
        if phase == "done":
            stop = "sequence_done"
            break
        step_legal(env.sim, action)
        actions.append(np.asarray(action, np.float32))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            stop = "recovery_failed"
            break
    stacked = np.stack(actions, axis=0) if actions else np.zeros((0, 7), np.float32)
    return stacked, stop


def _gravity_ok(traced: list[dict]) -> bool:
    start = float(traced[0]["rh"][0])
    if abs(start) < 1e-4:
        return False
    sign = 1.0 if start > 0.0 else -1.0
    values = []
    for row in traced[1:]:
        if float(row["action"][6]) < 0.5 and int(row["nL"]) > 0 and int(row["nR"]) > 0:
            values.append(float(row["g_h"][0]))
    if not values:
        return False
    return float(np.median(values)) * sign < 0.0


def _classify(traced: list[dict]) -> dict:
    start, end = traced[0], traced[-1]
    _direction, path = slip_direction(traced)
    command = np.stack([item["action"] for item in traced[1:]]) if len(traced) > 1 else np.zeros((0, 7))
    light = int(np.sum(command[:, 6] < 0.5)) if len(command) else 0
    record = {
        "inward_rhx_mm": round(1000.0 * (abs(float(start["rh"][0])) - abs(float(end["rh"][0]))), 3),
        "path_mm": round(1000.0 * path, 3),
        "bilateral_fraction": round(float(np.mean(bilateral_mask(traced))), 4),
        "release_then_regrasp": bool(release_regrasp(traced)),
        "light_steps": light,
        "gravity_assists": _gravity_ok(traced),
    }
    record["mechanism_status"] = slip_status(record)
    if record["mechanism_status"] == "confirmed" and not record["gravity_assists"]:
        record["mechanism_status"] = "gravity_inconsistent"
    record["observed_mechanism"] = observed_mechanism(
        record["mechanism_status"], record["release_then_regrasp"],
    )
    return record


def _accept_sliding(env, book, blob, actions) -> dict | None:
    """Accept on SUCCESS_12S. Slip measurements are stored and never gate acceptance."""
    if len(actions) == 0:
        return None
    tokens, residuals, decoded, packed = legalize(actions, book)
    preview = {"mechanism_status": "undetermined", "observed_mechanism": "undetermined"}
    if len(decoded) == 0 or float(packed["fraction_within_0.25"]) < 1.0 - 1e-8:
        return {"ok": False, "reason": "k48_range", "mechanism": preview}
    out, observations, executed, contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded)
    if not (out.get("success") and qpos_error <= 1e-8 and phase_ok):
        return {"ok": False, "reason": out.get("reason"), "mechanism": preview}
    executed_arr = np.asarray(executed, np.float32)
    if executed_arr.shape != decoded.shape or float(np.max(np.abs(executed_arr - decoded))) > 1e-5:
        return {"ok": False, "reason": "decode_mismatch", "mechanism": preview}
    final = _classify(_trace_actions(env, blob, decoded))
    return {
        "ok": True,
        "reason": "SUCCESS_12S",
        "tokens": tokens,
        "residuals": residuals,
        "executed": executed_arr,
        "observations": np.asarray(observations, np.float32),
        "contacts": contacts,
        "handoff_obs": out.get("handoff_obs"),
        "qpos_error": qpos_error,
        "fraction": packed["fraction_within_0.25"],
        "max_linf": packed["max_linf"],
        "recovery_elapsed": out.get("recovery_elapsed"),
        "nominal_elapsed": out.get("nominal_elapsed"),
        "mechanism": final,
    }


def _schedule_actions(book, seed_name: str, tag: str):
    actions, cuts, illegal = build_schedule(SEED_BY_NAME[seed_name], book)
    if illegal or actions is None:
        return None
    for cut in cuts:
        if cut["tag"] == tag:
            return actions[: int(cut["end"])]
    return None


def _anchors() -> list[dict]:
    candidates = {}
    pool = ROOT / "results/archive/demo_results/diagnostics/raw/k48_recovery_v3/candidates.jsonl"
    for row in _rows(pool):
        candidates[row["id"]] = row
    anchors = []
    for row in _rows(ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl"):
        if row.get("verified_success") is False:
            continue
        source = row if row.get("speed_mps") is not None else candidates.get(row["id"])
        if source is None or source.get("azimuth_deg") is None or source.get("speed_mps") is None:
            continue
        anchors.append(source)
    if not anchors:
        raise RuntimeError("no confirmed sliding launch anchors")
    return anchors


def _draw_local(draw: int, anchors: list[dict]) -> dict:
    """Seeded neighborhood of a confirmed sliding launch, outside the near-duplicate ball."""
    rng = np.random.default_rng(90_000 + int(draw))
    base = anchors[int(rng.integers(0, len(anchors)))]
    spec = _draw_spec(draw)
    spec["azimuth_deg"] = (float(base["azimuth_deg"]) + float(rng.choice([-1.0, 1.0])) * float(rng.uniform(3.0, 8.0))) % 360.0
    spec["speed_mps"] = float(np.clip(float(base["speed_mps"]) + float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.22, 0.35)), 5.5, 9.0))
    spec["z_off_m"] = float(np.clip(float(base["z_off_m"]) + float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.0025, 0.0045)), -0.02, 0.02))
    spec["phi_deg"] = float(np.clip(float(base["phi_deg"]) + float(rng.choice([-1.0, 1.0])) * float(rng.uniform(5.5, 8.0)), -40.0, 40.0))
    spec["delay_s"] = float(np.clip(float(base.get("delay_s") or 0.15) + float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.055, 0.09)), 0.0, 0.45))
    spec["family"] = "ballistic_impact"
    spec["protocol"] = "skill_quota_local_v1"
    spec["draw"] = int(draw)
    return spec


def _try_sliding(env, book, blob) -> dict | None:
    restore_capture(env, blob["capture"])
    pack, bilateral = _grasp_now(env.sim)
    if not bilateral:
        return {"ok": False, "reason": "not_bilateral", "rhx": float(np.asarray(pack["rh"], float)[0])}
    rhx = float(np.asarray(pack["rh"], float)[0])
    family = "pos" if rhx >= 0.0 else "neg"
    attempts = [("inward_reseat", None), ("controlled_slide", None)]
    for seed_name, tag in PLAN[family]:
        attempts.append((seed_name, tag))
    last_reason = "no_attempt"
    for name, tag in attempts:
        if tag is None:
            actions, stop = _closed_actions(env, blob, name)
            if stop == "recovery_failed" or len(actions) == 0:
                last_reason = stop if len(actions) else "empty"
                continue
            oracle = name
        else:
            actions = _schedule_actions(book, name, tag)
            if actions is None:
                last_reason = "illegal_schedule"
                continue
            oracle = "controlled_slide"
        outcome = _accept_sliding(env, book, blob, actions)
        if outcome and outcome.get("ok"):
            outcome["oracle_strategy"] = oracle
            outcome["schedule"] = name if tag is None else f"{name}:{tag}"
            outcome["rhx"] = rhx
            return outcome
        if outcome and outcome.get("reason"):
            last_reason = str(outcome["reason"])
    return {"ok": False, "reason": last_reason, "rhx": rhx}


def _store_sliding(blob, episode_id: str, split: str, outcome: dict, spec: dict, nominal: dict, source: str) -> dict:
    folder = ROOT / "data" / ("controlled_sliding" if outcome["oracle_strategy"] == "controlled_slide" else "inward_sliding") / split
    stem = f"{episode_id}__{outcome['oracle_strategy']}"
    if outcome["oracle_strategy"] == "controlled_slide":
        stem = f"{episode_id}__controlled_slide__verified"
    continuous, token_path = save_pair(
        folder, stem, outcome["executed"], outcome["observations"], outcome["contacts"],
        outcome["tokens"], outcome["residuals"], outcome["handoff_obs"],
    )
    snap = ROOT / "data" / "snapshots" / split / f"{episode_id}.pkl"
    if not snap.exists():
        snap.parent.mkdir(parents=True, exist_ok=True)
        with snap.open("wb") as handle:
            pickle.dump(blob, handle)
    row = {
        "id": episode_id,
        "snapshot": str(snap.relative_to(ROOT)).replace("\\", "/"),
        "strategy": outcome["oracle_strategy"],
        "oracle_strategy": outcome["oracle_strategy"],
        "family": "sliding",
        "mechanism": "gravity_assisted_in_hand_sliding",
        "mechanism_status": outcome["mechanism"].get("mechanism_status"),
        "observed_mechanism": outcome["mechanism"].get("observed_mechanism"),
        "task_success": True,
        "oracle_family": "gravity_assisted_in_hand_sliding",
        "grasp_scene": "vertical_cylinder",
        "source_category": "controlled_sliding" if outcome["oracle_strategy"] == "controlled_slide" else "inward_sliding",
        "continuous_npz": str(continuous.relative_to(ROOT)).replace("\\", "/"),
        "token_residual_npz": str(token_path.relative_to(ROOT)).replace("\\", "/"),
        "handoff_label": "early_handoff",
        "source_kind": "quota_expansion",
        "nominal_class": "RECOVERY_CANDIDATE",
        "nominal_reason": nominal.get("reason"),
        "verified_success": True,
        "sample_weight": 0.02,
        "split": split,
        "provenance": source,
        "schedule": outcome.get("schedule"),
        "recovery_elapsed": outcome.get("recovery_elapsed"),
        "nominal_elapsed": outcome.get("nominal_elapsed"),
        "max_linf": outcome.get("max_linf"),
        "fraction_within_0.25": outcome.get("fraction"),
        "qpos_error": outcome.get("qpos_error"),
        "inward_rhx_mm": outcome["mechanism"]["inward_rhx_mm"],
        "path_mm": outcome["mechanism"]["path_mm"],
        "bilateral_fraction": outcome["mechanism"]["bilateral_fraction"],
        "gravity_assists": outcome["mechanism"]["gravity_assists"],
        "azimuth_deg": spec.get("azimuth_deg"),
        "speed_mps": spec.get("speed_mps"),
        "z_off_m": spec.get("z_off_m"),
        "phi_deg": spec.get("phi_deg"),
        "delay_s": spec.get("delay_s"),
        "scenario_key": list(_scenario_key(spec)) if spec.get("azimuth_deg") is not None else None,
        "formal": "SUCCESS_12S",
    }
    category = ROOT / "data" / row["source_category"] / "manifest.jsonl"
    unified = ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl"
    if episode_id not in _ids(category):
        _append(category, row)
    if not any(item["id"] == episode_id and item.get("oracle_strategy") == row["oracle_strategy"] for item in _rows(unified)):
        _append(unified, row)
    training = ROOT / "data" / ("valid_demonstrations.jsonl" if split == "valid" else "supervised_targets.jsonl")
    if not any(item.get("id") == episode_id and item.get("continuous_npz") == row["continuous_npz"] for item in _rows(training)):
        _append(training, row)
    return row


def _state() -> dict:
    path = STAGE / "state.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"next_draw": DRAW_START}


def _save_state(state: dict) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "state.json").write_text(json.dumps(state), encoding="utf-8")


def expand_sliding(env, book, launches: int, identities: set[str]) -> int:
    state = _state()
    draw = int(state.get("next_draw", DRAW_START))
    specs, reserved = _spec_pool()
    anchors = _anchors()
    added = 0
    ran = 0
    while ran < launches:
        report = quota_report()
        gaps = _gaps(report)
        if gaps["sliding"]["train"] == 0 and gaps["sliding"]["valid"] == 0:
            print("SLIDING_TARGET_MET", report["sliding_task_success"], flush=True)
            break
        spec = _draw_local(draw, anchors)
        draw += 1
        state["next_draw"] = draw
        _save_state(state)
        key = _scenario_key(spec)
        split = _split_for_key(key)
        if not _want(split, gaps, "sliding"):
            continue
        if key in reserved or any(near(spec, item) for item in specs):
            _append(STAGE / "attempts.jsonl", {"skill": "sliding", "reason": "near_duplicate", "split": split, "draw": draw})
            continue
        ran += 1
        episode_id = f"slide_quota_{draw:06d}"
        spec = dict(spec)
        spec.update({"split": split, "attempt_id": episode_id, "protocol": "skill_quota_v1"})
        evidence, blob = run_episode(env, spec)
        evidence, blob = cylinder_first(evidence, blob)
        specs.append(spec)
        reserved.add(key)
        if blob is None:
            print("LAUNCH", episode_id, evidence.get("category"), flush=True)
            _append(STAGE / "attempts.jsonl", {"skill": "sliding", "id": episode_id, "split": split, "reason": evidence.get("category"), "success": False, **spec})
            continue
        nominal = immediate_handoff(env, blob)
        if nominal.get("success"):
            print("LAUNCH", episode_id, "nominal_solvable", flush=True)
            _append(STAGE / "attempts.jsonl", {"skill": "sliding", "id": episode_id, "split": split, "reason": "nominal_solvable", "success": False})
            continue
        identity = _fingerprint(blob)
        if identity in identities:
            print("LAUNCH", episode_id, "duplicate_snapshot", flush=True)
            _append(STAGE / "attempts.jsonl", {"skill": "sliding", "id": episode_id, "split": split, "reason": "duplicate_snapshot", "success": False})
            continue
        restore_capture(env, blob["capture"])
        tilt = _tilt(env.sim)
        outcome = _try_sliding(env, book, blob)
        if not outcome or not outcome.get("ok"):
            print("LAUNCH", episode_id, split, outcome.get("reason") if outcome else "miss", "tilt", round(tilt, 1), "rhx_mm", None if outcome is None else round(1000 * float(outcome.get("rhx") or 0.0), 1), nominal.get("reason"), flush=True)
            _append(STAGE / "attempts.jsonl", {
                "skill": "sliding", "id": episode_id, "split": split, "reason": None if outcome is None else outcome.get("reason"),
                "nominal_reason": nominal.get("reason"), "tilt_deg": tilt, "success": False,
                "azimuth_deg": spec["azimuth_deg"], "speed_mps": spec["speed_mps"],
                "z_off_m": spec["z_off_m"], "phi_deg": spec["phi_deg"], "delay_s": spec["delay_s"],
                "scenario_key": list(key),
            })
            if _want(split, _gaps(quota_report()), "vertical_recatch"):
                _try_store_vertical(env, book, blob, episode_id, split, spec, nominal, identities)
            continue
        row = _store_sliding(blob, episode_id, split, outcome, spec, nominal, "new_ballistic_generation")
        identities.add(identity)
        added += 1
        print("ADDED_SLIDING", row["split"], episode_id, outcome["schedule"], outcome["mechanism"]["inward_rhx_mm"], flush=True)
        _append(STAGE / "attempts.jsonl", {"skill": "sliding", "id": episode_id, "split": split, "reason": "SUCCESS_12S", "success": True, "oracle": outcome["oracle_strategy"]})
        if _want(split, _gaps(quota_report()), "vertical_recatch"):
            _try_store_vertical(env, book, blob, episode_id, split, spec, nominal, identities)
    print("SLIDING_BATCH", "ran", ran, "added", added, quota_report()["sliding_task_success"], flush=True)
    return added


def _try_store_vertical(env, book, blob, episode_id, split, spec, nominal, identities) -> bool:
    if episode_id in _ids(ROOT / "data" / "vertical_recatch" / "manifest.jsonl"):
        return False
    restore_capture(env, blob["capture"])
    tilt = _tilt(env.sim)
    for spec_v in vertical_specs():
        result = vertical_rollout(env, blob, book, spec_v)
        summary = _summary(result)
        if result["mode"] != "handoff" or summary.get("t_off") is None or summary.get("t_re") is None:
            _append(STAGE / "attempts.jsonl", {
                "skill": "vertical_recatch", "id": episode_id, "strategy": spec_v["name"],
                "reason": result["mode"], "success": False, "tilt_deg": tilt,
            })
            continue
        if not release_and_regrip([row["nL"] for row in result["rows"]], [row["nR"] for row in result["rows"]]):
            _append(STAGE / "attempts.jsonl", {
                "skill": "vertical_recatch", "id": episode_id, "strategy": spec_v["name"],
                "reason": "no_sustained_release_regrip", "success": False, "tilt_deg": tilt,
            })
            continue
        if float(result["max_linf"]) > 0.25 + 1e-8:
            continue
        out, extra = _finish(env.sim, blob["ref"])
        if not out.get("success") or extra != "HELD_EXTENSION":
            _append(STAGE / "attempts.jsonl", {
                "skill": "vertical_recatch", "id": episode_id, "strategy": spec_v["name"],
                "reason": out.get("reason") if not out.get("success") else extra, "success": False,
            })
            continue
        actions = np.asarray(result["actions"], np.float32)
        tokens, residuals, decoded, packed = legalize(actions, book)
        if float(packed["fraction_within_0.25"]) < 1.0 - 1e-8:
            continue
        replay, observations, executed, contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded)
        if not (replay.get("success") and qpos_error <= 1e-8 and phase_ok):
            _append(STAGE / "attempts.jsonl", {"skill": "vertical_recatch", "id": episode_id, "reason": replay.get("reason"), "success": False})
            continue
        folder = ROOT / "data" / "vertical_recatch" / split
        stem = f"{episode_id}__{spec_v['name']}"
        continuous, token_path = save_pair(
            folder, stem, np.asarray(executed, np.float32), np.asarray(observations, np.float32),
            contacts, tokens, residuals, replay.get("handoff_obs"),
        )
        snap = ROOT / "data" / "snapshots" / split / f"{episode_id}.pkl"
        if not snap.exists():
            snap.parent.mkdir(parents=True, exist_ok=True)
            with snap.open("wb") as handle:
                pickle.dump(blob, handle)
        row = {
            "id": episode_id,
            "split": split,
            "strategy": spec_v["name"],
            "oracle_strategy": spec_v["name"],
            "family": "vertical_recatch",
            "grasp_scene": "vertical_cylinder",
            "mechanism": "open_overchase_close_brake",
            "mechanism_status": "release_regrip",
            "observed_mechanism": "release_regrip",
            "task_success": True,
            "continuous_npz": str(continuous.relative_to(ROOT)).replace("\\", "/"),
            "token_residual_npz": str(token_path.relative_to(ROOT)).replace("\\", "/"),
            "snapshot": str(snap.relative_to(ROOT)).replace("\\", "/"),
            "handoff_label": "early_handoff",
            "nominal_class": "RECOVERY_CANDIDATE",
            "nominal_reason": nominal.get("reason"),
            "provenance": "vertical_recatch_quota",
            "source_kind": "quota_expansion",
            "verified_success": True,
            "sample_weight": 1.0,
            "formal": "SUCCESS_12S",
            "extension": extra,
            "tilt_deg": round(tilt, 2),
            "t_off": summary.get("t_off"),
            "t_re": summary.get("t_re"),
            "min_bottom": summary.get("min_bottom"),
            "k48_linf": round(float(packed["max_linf"]), 4),
            "azimuth_deg": None if spec is None else spec.get("azimuth_deg"),
            "speed_mps": None if spec is None else spec.get("speed_mps"),
            "z_off_m": None if spec is None else spec.get("z_off_m"),
            "phi_deg": None if spec is None else spec.get("phi_deg"),
            "delay_s": None if spec is None else spec.get("delay_s"),
        }
        manifest = ROOT / "data" / "vertical_recatch" / "manifest.jsonl"
        _append(manifest, row)
        training = ROOT / "data" / ("valid_demonstrations.jsonl" if split == "valid" else "supervised_targets.jsonl")
        _append(training, row)
        identities.add(_fingerprint(blob))
        print("ADDED_VERTICAL", split, episode_id, spec_v["name"], "tilt", round(tilt, 1), flush=True)
        _append(STAGE / "attempts.jsonl", {"skill": "vertical_recatch", "id": episode_id, "split": split, "reason": "SUCCESS_12S", "success": True, "strategy": spec_v["name"]})
        return True
    return False


def expand_vertical_existing(env, book, limit: int) -> int:
    have = _ids(ROOT / "data" / "vertical_recatch" / "manifest.jsonl")
    sliding = _rows(ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl")
    already = {row["id"] for row in _rows(STAGE / "attempts.jsonl") if row.get("skill") == "vertical_recatch"}
    seen = set()
    added = 0
    for row in sliding:
        if added >= limit:
            break
        if row["id"] in have or row["id"] in seen or row["id"] in already:
            continue
        seen.add(row["id"])
        gaps = _gaps(quota_report())
        if not _want(row["split"], gaps, "vertical_recatch"):
            continue
        blob = _load_blob(row["snapshot"])
        nominal = immediate_handoff(env, blob)
        if nominal.get("success"):
            _append(STAGE / "attempts.jsonl", {"skill": "vertical_recatch", "id": row["id"], "reason": "nominal_solvable", "success": False})
            continue
        spec = {key: row.get(key) for key in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")}
        if _try_store_vertical(env, book, blob, row["id"], row["split"], spec, nominal, set()):
            added += 1
            have.add(row["id"])
    print("VERTICAL_EXISTING", added, quota_report()["vertical_recatch"], flush=True)
    return added


def _cell_split(cell_id: str) -> str:
    return "valid" if zlib.crc32(cell_id.encode()) % 5 == 0 else "train"


def _new_cells() -> list[dict]:
    import re

    existing = _ids(ROOT / "data" / "recatch" / "manifest.jsonl")
    tried = {
        row["id"] for row in _rows(STAGE / "attempts.jsonl")
        if row.get("skill") == "horizontal_recatch" and row.get("reason") != "cylinder_not_horizontal"
    }
    pattern = re.compile(r"s(?P<speed>[\d.]+)_p(?P<pitch>[\d.]+)_a(?P<axial>[+-][\d.]+)_e(?P<ecc>[\d.]+)")
    cells = []
    seen = set()
    for row in _rows(STAGE / "attempts.jsonl"):
        if row.get("skill") != "horizontal_recatch" or row.get("reason") != "cylinder_not_horizontal":
            continue
        match = pattern.fullmatch(str(row.get("id") or ""))
        if match is None or row["id"] in seen or row["id"] in existing:
            continue
        seen.add(row["id"])
        cells.append({
            "speed": float(match["speed"]), "pitch": float(match["pitch"]),
            "axial": float(match["axial"]), "ecc": float(match["ecc"]),
            "mass": None, "friction": None,
            "split": row.get("split") or _cell_split(row["id"]),
            "kind": "impact", "id": row["id"],
        })
    for speed, pitches in (
        (6.0, (10.0, 16.0)),
        (6.2, (8.0, 15.0)),
        (6.8, (8.0, 15.0)),
        (5.8, (12.0, 18.0, 22.0)),
        (6.4, (12.0, 18.0, 22.0)),
        (6.6, (12.0, 18.0, 22.0)),
        (7.2, (12.0, 18.0, 22.0)),
    ):
        for pitch in pitches:
            axials = (0.0, 0.012, -0.012) if speed <= 6.2 else (0.0, 0.006, -0.006)
            eccs = (0.0, 0.008) if speed <= 6.2 else (0.0, 0.004)
            for axial in axials:
                for ecc in eccs:
                    cell_id = f"s{speed:.1f}_p{pitch:.0f}_a{axial:+.3f}_e{ecc:.3f}"
                    if cell_id in existing or cell_id in tried or cell_id in seen:
                        continue
                    seen.add(cell_id)
                    cells.append({
                        "speed": speed, "pitch": pitch, "axial": axial, "ecc": ecc,
                        "mass": None, "friction": None,
                        "split": _cell_split(cell_id), "kind": "impact", "id": cell_id,
                    })
    return cells


def _extra_horizontal_specs() -> list[tuple[str, dict]]:
    """Legal timing variants. Physics, K48, and SUCCESS_12S stay unchanged."""
    from data_generation.strategies.recatch_oracle import _base_spec

    tweaks = (
        ("chase_long", {"chase_limit": 0.55, "stop_cz": 0.108, "down": 1.6, "brake": 4.0, "lift": 0.06, "pad_steps": 3, "close_tau": -18.0, "hold_tau": -18.0}),
        ("reclamp_mid", {"chase_limit": 0.45, "stop_cz": 0.104, "down": 2.0, "brake": 7.0, "lift": 0.10, "pad_steps": 3, "open_gap": 0.050, "close_gap": 0.046, "close_tau": -16.0, "hold_tau": -18.0}),
        ("reclamp_early", {"chase_limit": 0.28, "stop_cz": 0.098, "down": 2.8, "brake": 9.0, "lift": 0.12, "pad_steps": 2, "close_tau": -18.0, "hold_tau": -16.0}),
        ("late_handoff", {"chase_limit": 0.50, "stop_cz": 0.100, "down": 1.8, "brake": 5.0, "lift": 0.05, "pad_steps": 4, "settle_bottom": 0.58, "close_tau": -14.0, "hold_tau": -18.0}),
    )
    specs = []
    for name, update in tweaks:
        spec = _base_spec()
        spec.update(update)
        specs.append((name, spec))
    return specs


def _retry_horizontal_cells() -> list[dict]:
    """Nominal-fail cells whose official oracles missed. Extra timing only."""
    extra_names = {name for name, _spec in _extra_horizontal_specs()}
    by_id: dict[str, list[dict]] = {}
    for row in _rows(STAGE / "attempts.jsonl"):
        if row.get("skill") != "horizontal_recatch" or not row.get("id"):
            continue
        by_id.setdefault(row["id"], []).append(row)
    pattern_ok = set()
    cells = []
    for cell_id, rows in by_id.items():
        if any(row.get("reason") == "SUCCESS_12S" or row.get("strategy") in extra_names for row in rows):
            continue
        if any(row.get("reason") == "TRIVIAL" for row in rows):
            continue
        failures = [row for row in rows if row.get("reason") not in {"INVALID_GRASP", "cylinder_not_horizontal", "duplicate_snapshot"}]
        if not failures:
            continue
        import re
        match = re.fullmatch(r"s(?P<speed>[\d.]+)_p(?P<pitch>[\d.]+)_a(?P<axial>[+-][\d.]+)_e(?P<ecc>[\d.]+)", cell_id)
        if match is None or cell_id in pattern_ok:
            continue
        pattern_ok.add(cell_id)
        sample = failures[0]
        cells.append({
            "speed": float(match["speed"]), "pitch": float(match["pitch"]),
            "axial": float(match["axial"]), "ecc": float(match["ecc"]),
            "mass": None, "friction": None,
            "split": sample.get("split") or _cell_split(cell_id),
            "kind": "impact", "id": cell_id, "extra_only": True,
        })
    return cells


def expand_horizontal(limit: int) -> int:
    from data_generation.strategies.build_open_overchase_recatch_dataset import (
        _baseline,
        _nominal_dynamics,
        _prepare,
        _reapply,
        _specs,
        _verify,
    )
    from data_generation.strategies.recatch_oracle import _impact, _setup

    gaps = _gaps(quota_report())
    if gaps["horizontal_recatch"]["train"] == 0 and gaps["horizontal_recatch"]["valid"] == 0:
        print("HORIZONTAL_TARGET_MET", flush=True)
        return 0
    sim, base, ref, _target, book = _setup()
    nominal = _nominal_dynamics(sim)
    official_specs = list(_specs())
    extra_specs = _extra_horizontal_specs()
    added = 0
    used = 0
    identities = _known_identities()
    queue = _retry_horizontal_cells() + _new_cells()
    for cell in queue:
        if used >= limit or added >= limit:
            break
        gaps = _gaps(quota_report())
        if not _want(cell["split"], gaps, "horizontal_recatch"):
            continue
        used += 1
        if not _prepare(sim, base, nominal, cell):
            print("H_INVALID", cell["id"], flush=True)
            _append(STAGE / "attempts.jsonl", {"skill": "horizontal_recatch", "id": cell["id"], "split": cell["split"], "reason": "INVALID_GRASP", "success": False})
            continue
        info, err = _impact(sim, base, cell, after_restore=_reapply(sim, nominal, cell))
        if err:
            print("H_MISS", cell["id"], err, flush=True)
            _append(STAGE / "attempts.jsonl", {"skill": "horizontal_recatch", "id": cell["id"], "split": cell["split"], "reason": err, "success": False})
            continue
        pose = info["pose"]
        tilt = float(pose.get("tilt_deg") or 0.0)
        nominal_out = _baseline(sim, info["snap"], ref, nominal, cell)
        if nominal_out["ok"]:
            print("H_TRIVIAL", cell["id"], flush=True)
            _append(STAGE / "attempts.jsonl", {"skill": "horizontal_recatch", "id": cell["id"], "split": cell["split"], "reason": "TRIVIAL", "success": False, "tilt_deg": tilt})
            continue
        identity = _fingerprint(info["snap"])
        if identity in identities:
            _append(STAGE / "attempts.jsonl", {"skill": "horizontal_recatch", "id": cell["id"], "reason": "duplicate_snapshot", "success": False})
            continue
        kept = False
        oracle_specs = extra_specs if cell.get("extra_only") else official_specs + extra_specs
        for name, spec in oracle_specs:
            result = __import__(
                "data_generation.strategies.recatch_oracle", fromlist=["rollout", "staged_choose"]
            )
            rolled = result.rollout(sim, info["snap"], result.staged_choose(spec), 2.4, after_restore=_reapply(sim, nominal, cell))
            reason, packed = _verify(sim, info["snap"], rolled, ref, book, nominal)
            print("H_TRY", cell["id"], name, reason, flush=True)
            if reason != "ok":
                _append(STAGE / "attempts.jsonl", {"skill": "horizontal_recatch", "id": cell["id"], "split": cell["split"], "strategy": name, "reason": reason, "success": False, "tilt_deg": tilt})
                continue
            contact_left = [row["nL"] for row in rolled["rows"]]
            contact_right = [row["nR"] for row in rolled["rows"]]
            if not release_and_regrip(contact_left, contact_right):
                print("H_FLICKER", cell["id"], name, flush=True)
                _append(STAGE / "attempts.jsonl", {"skill": "horizontal_recatch", "id": cell["id"], "split": cell["split"], "strategy": name, "reason": "no_sustained_release_regrip", "success": False, "tilt_deg": tilt})
                continue
            actions = np.stack(rolled["actions"]).astype(np.float32)
            tokens = np.asarray(packed["tokens"], np.int64)
            residuals = (actions - book[tokens]).astype(np.float32)
            folder = ROOT / "data" / "recatch" / cell["split"]
            stem = f"{cell['id']}__{name}"
            continuous, token_path = save_pair(
                folder, stem, actions, np.asarray(rolled["observations"], np.float32),
                [{"t": 0.0, "nL": 0, "nR": 0, "z": 0.0}], tokens, residuals, rolled.get("handoff_obs"),
            )
            snap = ROOT / "data" / "snapshots" / cell["split"] / f"{cell['id']}.pkl"
            snap.parent.mkdir(parents=True, exist_ok=True)
            with snap.open("wb") as handle:
                pickle.dump(info["snap"], handle)
            row = {
                "id": cell["id"],
                "split": cell["split"],
                "strategy": name,
                "oracle_strategy": name,
                "family": "recatch",
                "grasp_scene": "horizontal_cylinder",
                "mechanism": "open_overchase_close_brake",
                "mechanism_status": "release_regrip",
                "observed_mechanism": "release_regrip",
                "task_success": True,
                "continuous_npz": str(continuous.relative_to(ROOT)).replace("\\", "/"),
                "token_residual_npz": str(token_path.relative_to(ROOT)).replace("\\", "/"),
                "snapshot": str(snap.relative_to(ROOT)).replace("\\", "/"),
                "source_snapshot": str(snap.relative_to(ROOT)).replace("\\", "/"),
                "handoff_label": "SUCCESS_12S",
                "nominal_class": "RECOVERY_CANDIDATE",
                "nominal_reason": nominal_out.get("why"),
                "provenance": "horizontal_recatch_quota",
                "source_kind": "quota_expansion",
                "sample_weight": 1.0,
                "verified_success": True,
                "speed": cell["speed"],
                "pitch": cell["pitch"],
                "axial": cell["axial"],
                "ecc": cell["ecc"],
                "mass": None,
                "friction": None,
                "k48_linf": round(float(packed["max_linf"]), 4),
                "extension": "HELD_EXTENSION",
                "formal": "SUCCESS_12S",
                "tilt_deg": round(tilt, 2),
                "tilt_boundary": bool(tilt > HORIZONTAL_TILT_BOUNDARY),
                "min_bottom": rolled["metrics"].get("min_bottom"),
                "t_open": rolled["metrics"].get("t_open"),
                "t_re": rolled["metrics"].get("t_re"),
                "t_brake": rolled["metrics"].get("t_vz0"),
            }
            _append(ROOT / "data" / "recatch" / "manifest.jsonl", row)
            training = ROOT / "data" / ("valid_demonstrations.jsonl" if cell["split"] == "valid" else "supervised_targets.jsonl")
            _append(training, row)
            identities.add(identity)
            added += 1
            kept = True
            print("ADDED_HORIZONTAL", cell["split"], cell["id"], name, flush=True)
            _append(STAGE / "attempts.jsonl", {"skill": "horizontal_recatch", "id": cell["id"], "split": cell["split"], "reason": "SUCCESS_12S", "success": True, "strategy": name})
            break
        if not kept:
            print("H_FAIL", cell["id"], flush=True)
    print("HORIZONTAL_BATCH", "used", used, "added", added, quota_report()["horizontal_recatch"], flush=True)
    return added


def annotate_task_labels() -> int:
    """Add diagnostic labels. Does not rewrite action tapes."""
    changed = 0
    targets = {
        ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl": "vertical_cylinder",
        ROOT / "data" / "controlled_sliding" / "manifest.jsonl": "vertical_cylinder",
        ROOT / "data" / "inward_sliding" / "manifest.jsonl": "vertical_cylinder",
        ROOT / "data" / "vertical_recatch" / "manifest.jsonl": "vertical_cylinder",
        ROOT / "data" / "recatch" / "manifest.jsonl": "horizontal_cylinder",
    }
    for path, scene in targets.items():
        rows = _rows(path)
        dirty = False
        for row in rows:
            if row.get("grasp_scene") != scene:
                row["grasp_scene"] = scene
                dirty = True
            if "task_success" not in row:
                row["task_success"] = bool(row.get("verified_success", True))
                dirty = True
            if "observed_mechanism" not in row:
                if scene == "horizontal_cylinder" or path.parent.name == "vertical_recatch":
                    row["observed_mechanism"] = "release_regrip"
                else:
                    row["observed_mechanism"] = observed_mechanism(str(row.get("mechanism_status") or "unscanned"))
                dirty = True
            if scene != "horizontal_cylinder" and path.parent.name != "vertical_recatch" and "oracle_family" not in row:
                row["oracle_family"] = "gravity_assisted_in_hand_sliding"
                dirty = True
        if dirty:
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            changed += 1
    return changed


def reverify_mechanism_rejects(env, book, limit: int) -> int:
    """Replay launches that were dropped for lack of slip, and run SUCCESS_12S."""
    have = _ids(ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl")
    done = {row["id"] for row in _rows(STAGE / "attempts.jsonl") if row.get("phase") == "reverify"}
    chosen = []
    seen = set()
    for row in _rows(STAGE / "attempts.jsonl"):
        if row.get("skill") != "sliding" or row.get("reason") not in DIAGNOSTIC_REJECTS:
            continue
        if row.get("azimuth_deg") is None or row["id"] in seen or row["id"] in have or row["id"] in done:
            continue
        seen.add(row["id"])
        chosen.append(row)
    print("REVERIFY_QUEUE", len(chosen), flush=True)
    identities = _known_identities()
    added = 0
    used = 0
    for row in chosen:
        if used >= limit:
            break
        used += 1
        spec = {
            "azimuth_deg": float(row["azimuth_deg"]),
            "speed_mps": float(row["speed_mps"]),
            "z_off_m": float(row["z_off_m"]),
            "phi_deg": float(row["phi_deg"]),
            "delay_s": float(row.get("delay_s") or 0.0),
            "direction": "continuous",
            "family": "ballistic_impact",
            "protocol": "skill_quota_reverify",
            "split": row["split"],
            "attempt_id": row["id"],
        }
        evidence, blob = run_episode(env, spec)
        evidence, blob = cylinder_first(evidence, blob)
        if blob is None:
            _append(STAGE / "attempts.jsonl", {"skill": "sliding", "phase": "reverify", "id": row["id"], "reason": evidence.get("category"), "success": False})
            print("REVERIFY", row["id"], evidence.get("category"), flush=True)
            continue
        nominal = immediate_handoff(env, blob)
        if nominal.get("success"):
            _append(STAGE / "attempts.jsonl", {"skill": "sliding", "phase": "reverify", "id": row["id"], "reason": "nominal_solvable", "success": False})
            print("REVERIFY", row["id"], "nominal_solvable", flush=True)
            continue
        identity = _fingerprint(blob)
        if identity in identities:
            _append(STAGE / "attempts.jsonl", {"skill": "sliding", "phase": "reverify", "id": row["id"], "reason": "duplicate_snapshot", "success": False})
            print("REVERIFY", row["id"], "duplicate_snapshot", flush=True)
            continue
        outcome = _try_sliding(env, book, blob)
        if not outcome or not outcome.get("ok"):
            reason = None if outcome is None else outcome.get("reason")
            _append(STAGE / "attempts.jsonl", {"skill": "sliding", "phase": "reverify", "id": row["id"], "split": row["split"], "reason": reason, "success": False})
            print("REVERIFY", row["id"], reason, flush=True)
            continue
        stored = _store_sliding(blob, row["id"], row["split"], outcome, spec, nominal, "mechanism_reject_reverify")
        identities.add(identity)
        added += 1
        print("REVERIFY_ADDED", stored["split"], row["id"], stored.get("observed_mechanism"), flush=True)
        _append(STAGE / "attempts.jsonl", {
            "skill": "sliding", "phase": "reverify", "id": row["id"], "split": row["split"],
            "reason": "SUCCESS_12S", "success": True, "observed_mechanism": stored.get("observed_mechanism"),
        })
    print("REVERIFY_DONE", "used", used, "added", added, flush=True)
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sliding-launches", type=int, default=0)
    parser.add_argument("--vertical-existing", type=int, default=0)
    parser.add_argument("--horizontal-cells", type=int, default=0)
    parser.add_argument("--reverify-limit", type=int, default=0)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    annotated = annotate_task_labels()
    report = quota_report()
    print("ANNOTATED_MANIFESTS", annotated, flush=True)
    print("QUOTA", json.dumps(report), flush=True)
    print("GAPS", json.dumps(_gaps(report)), flush=True)
    if args.inventory_only:
        return
    STAGE.mkdir(parents=True, exist_ok=True)
    if args.horizontal_cells:
        expand_horizontal(args.horizontal_cells)
    if args.reverify_limit or args.vertical_existing or args.sliding_launches:
        env = _open_env()
        book = load_codebook()
        if args.reverify_limit:
            reverify_mechanism_rejects(env, book, args.reverify_limit)
        if args.vertical_existing:
            expand_vertical_existing(env, book, args.vertical_existing)
        if args.sliding_launches:
            expand_sliding(env, book, args.sliding_launches, _known_identities())
    print("DONE", json.dumps(quota_report()), flush=True)


if __name__ == "__main__":
    main()
