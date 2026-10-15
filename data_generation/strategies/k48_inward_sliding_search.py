"""Bounded K48 gravity-assisted inward-sliding search.

Success is only the unchanged nominal controller completing the lift and holding
the grasp for 12 seconds. Geometric thresholds rank trials. They do not accept
or reject a rollout.

Writes under results/archive/demo_results/diagnostics/raw/k48_inward_sliding_search_v1.
Does not train, does not open FINAL, and does not modify k48_recovery_v3.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, grip_action, grip_tau, normalize_action
from envs.outcomes.physical_recovery import physical_pack
from data_generation.ball_impact_dataset import run_episode
from envs.outcomes.horizon_outcome import recovery_failed
from data_generation.oracles.representability import representability
from envs.step_legal import step_legal
from data_generation.rebuild_k48_recovery import (
    CAP_STEPS,
    LIMIT_S,
    _finger_x,
    _load_blob,
    _load_jsonl,
    _open_env,
    inward_reseat_action,
    legal_verify,
    legalize,
)
from training.k47_codebook import load_codebook

OLD = ROOT / "results/archive/demo_results/diagnostics/raw/k48_recovery_v3"
OUT = ROOT / "results/archive/demo_results/diagnostics/raw/k48_inward_sliding_search_v1"
DT = 0.02
STAGE_A_VERIFIES = 40
STAGE_B_VERIFIES = 8
STAGE_C_VERIFIES = 8
N_TRAIN = 6
N_VALID = 4

# Body rates are rad/s. Grip values are recovery actions. `signed` multiplies wy
# by the searched sign. Historical simultaneous support-roll plus pitch is the
# first seed, not a fixed law for every initial condition.
SEEDS = (
    {
        "name": "hist_simultaneous",
        "parts": (
            {"tag": "reorient", "s": 0.40, "wx": 1.12, "wy": 2.0, "grip": 1.0, "signed": True},
            {"tag": "relax", "s": 0.48, "wx": 0.0, "wy": 0.0, "grip": -0.48, "signed": False},
            {"tag": "retighten", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0, "signed": False},
        ),
    },
    {
        "name": "roll_then_pitch",
        "parts": (
            {"tag": "reorient", "s": 0.20, "wx": 1.12, "wy": 0.0, "grip": 1.0, "signed": False},
            {"tag": "reorient", "s": 0.40, "wx": 0.0, "wy": 2.0, "grip": 1.0, "signed": True},
            {"tag": "relax", "s": 0.48, "wx": 0.0, "wy": 0.0, "grip": -0.48, "signed": False},
            {"tag": "retighten", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0, "signed": False},
        ),
    },
    {
        "name": "pitch_then_roll",
        "parts": (
            {"tag": "reorient", "s": 0.40, "wx": 0.0, "wy": 2.0, "grip": 1.0, "signed": True},
            {"tag": "reorient", "s": 0.20, "wx": 1.12, "wy": 0.0, "grip": 1.0, "signed": False},
            {"tag": "relax", "s": 0.48, "wx": 0.0, "wy": 0.0, "grip": -0.48, "signed": False},
            {"tag": "retighten", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0, "signed": False},
        ),
    },
    {
        "name": "relax_during_motion",
        "parts": (
            {"tag": "reorient", "s": 0.40, "wx": 1.12, "wy": 2.0, "grip": -0.48, "signed": True},
            {"tag": "retighten", "s": 0.24, "wx": 0.0, "wy": 0.0, "grip": 1.0, "signed": False},
        ),
    },
    {
        "name": "slip_grip_m5",
        "parts": (
            {"tag": "reorient", "s": 0.40, "wx": 1.12, "wy": 2.0, "grip": 1.0, "signed": True},
            {"tag": "relax", "s": 0.70, "wx": 0.0, "wy": 0.0, "grip": None, "signed": False},
            {"tag": "retighten", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0, "signed": False},
        ),
    },
    {
        "name": "brake_then_relax",
        "parts": (
            {"tag": "reorient", "s": 0.40, "wx": 1.12, "wy": 2.0, "grip": 1.0, "signed": True},
            {"tag": "brake", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0, "signed": False},
            {"tag": "relax", "s": 0.48, "wx": 0.0, "wy": 0.0, "grip": -0.48, "signed": False},
            {"tag": "retighten", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0, "signed": False},
        ),
    },
    {
        "name": "slow_pitch",
        "parts": (
            {"tag": "reorient", "s": 0.80, "wx": 0.0, "wy": 1.0, "grip": 1.0, "signed": True},
            {"tag": "relax", "s": 0.48, "wx": 0.0, "wy": 0.0, "grip": -0.48, "signed": False},
            {"tag": "retighten", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0, "signed": False},
        ),
    },
    {
        "name": "return_after_tight",
        "parts": (
            {"tag": "reorient", "s": 0.40, "wx": 1.12, "wy": 2.0, "grip": 1.0, "signed": True},
            {"tag": "relax", "s": 0.48, "wx": 0.0, "wy": 0.0, "grip": -0.48, "signed": False},
            {"tag": "retighten", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0, "signed": False},
            {"tag": "return", "s": 0.40, "wx": 0.0, "wy": -1.0, "grip": 1.0, "signed": True},
        ),
    },
)


def _refuse(path: Path):
    text = str(path).replace("\\", "/").lower()
    if "final" in text:
        raise RuntimeError(f"refusing final path {path}")


def _append(path: Path, row: dict):
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def _done_keys(path: Path) -> set[str]:
    return {row.get("key") for row in _load_jsonl(path) if row.get("key")}


def _wrist(wx: float, wy: float, grip: float) -> np.ndarray:
    action = normalize_action(np.zeros(3), np.array([wx, wy, 0.0]), grip_tau(grip), AIRBORNE_V3)
    return np.asarray(action, np.float32)


def _legal(book, action) -> bool:
    packed = representability(np.asarray(action, np.float32).reshape(1, 7), book)
    return float(packed["max_linf"]) <= 0.25 + 1e-8


def _steps(seconds: float) -> int:
    return max(1, int(round(float(seconds) / DT)))


def _grip_value(part: dict) -> float:
    if part["grip"] is None:
        return float(grip_action(-5.0))
    return float(part["grip"])


def build_schedule(seed: dict, wy_sign: float, book):
    """Return actions, cut indices, and an illegality note. wy_sign is +1 or -1."""
    actions = []
    cuts = []
    for part in seed["parts"]:
        wy = float(part["wy"]) * (float(wy_sign) if part["signed"] else 1.0)
        grip = _grip_value(part)
        action = _wrist(float(part["wx"]), wy, grip)
        if not _legal(book, action):
            return None, [], f"illegal wx={part['wx']} wy={wy} grip={grip}"
        count = _steps(part["s"])
        actions.extend([action] * count)
        cuts.append({"tag": part["tag"], "end": len(actions)})
    if len(actions) > CAP_STEPS:
        actions = actions[:CAP_STEPS]
        cuts = [item for item in cuts if item["end"] <= CAP_STEPS]
    return np.stack(actions, axis=0), cuts, None


def _snapshot(sim) -> dict:
    pack = physical_pack(sim)
    rh = np.asarray(pack["rh"], float)
    return {
        "rhx": float(rh[0]),
        "gx": float(np.asarray(pack["g_h"], float)[0]),
        "vx": float(np.asarray(pack["v_rel_h"], float)[0]),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "fnL": float(pack["Fn_L"]),
        "fnR": float(pack["Fn_R"]),
        "z": float(pack["obj_z"]),
    }


def rollout(env, blob, actions):
    """Execute legal actions from the saved capture. Stop if the object falls."""
    from envs.simulation.capture import restore_capture

    restore_capture(env, blob["capture"])
    start = _snapshot(env.sim)
    rows = [start]
    lost_low = 0
    drop_step = None
    t0 = float(env.sim.data.time)
    executed = []
    for index, action in enumerate(np.asarray(actions, np.float32)):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            break
        step_legal(env.sim, action)
        executed.append(np.asarray(action, np.float32))
        rows.append(_snapshot(env.sim))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            drop_step = index + 1
            break
    return np.stack(executed, axis=0) if executed else np.zeros((0, 7), np.float32), rows, drop_step


def _trace_summary(rows, drop_step, cuts) -> dict:
    rh0 = float(rows[0]["rhx"])
    end = rows[-1]
    inward = abs(rh0) - abs(float(end["rhx"]))
    bilateral = [row["nL"] > 0 and row["nR"] > 0 for row in rows]
    relax_cuts = [item["end"] for item in cuts if item["tag"] == "relax"]
    relax_note = None
    if relax_cuts:
        a = 0
        for item in cuts:
            if item["tag"] == "relax":
                b = min(item["end"], len(rows) - 1)
                window = rows[a:b + 1]
                if window:
                    relax_note = {
                        "rhx_start": float(window[0]["rhx"]),
                        "rhx_end": float(window[-1]["rhx"]),
                        "inward_mm": 1e3 * (abs(float(window[0]["rhx"])) - abs(float(window[-1]["rhx"]))),
                        "bilateral_fraction": float(np.mean([row["nL"] > 0 and row["nR"] > 0 for row in window])),
                        "gx_mean": float(np.mean([row["gx"] for row in window])),
                        "max_abs_vx": float(np.max(np.abs([row["vx"] for row in window]))),
                    }
                break
            a = item["end"]
    return {
        "rhx_start": rh0,
        "rhx_end": float(end["rhx"]),
        "inward_mm": 1e3 * inward,
        "gx_start": float(rows[0]["gx"]),
        "gx_end": float(end["gx"]),
        "gx_min": float(min(row["gx"] for row in rows)),
        "gx_max": float(max(row["gx"] for row in rows)),
        "bilateral_fraction": float(np.mean(bilateral)) if bilateral else 0.0,
        "min_nL": int(min(row["nL"] for row in rows)),
        "min_nR": int(min(row["nR"] for row in rows)),
        "max_abs_vx": float(max(abs(row["vx"]) for row in rows)),
        "fnL_end": float(end["fnL"]),
        "fnR_end": float(end["fnR"]),
        "drop_step": drop_step,
        "steps": int(len(rows) - 1),
        "relax": relax_note,
    }


def _verify_prefix(env, book, blob, actions):
    tokens, residuals, decoded, packed = legalize(actions, book)
    if float(packed["fraction_within_0.25"]) < 1.0 - 1e-8:
        return {"reason": "illegal_prefix", "success": False, "legal_replay": False}
    started = time.perf_counter()
    out, _obs, _exe, contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded)
    return {
        "reason": out.get("reason"),
        "success": bool(out.get("success")) and qpos_error <= 1e-8 and phase_ok,
        "physical_success": bool(out.get("success")),
        "recovery_elapsed": out.get("recovery_elapsed"),
        "nominal_elapsed": out.get("nominal_elapsed"),
        "lift_completed": out.get("lift_completed"),
        "qpos_error": qpos_error,
        "phase_restored": phase_ok,
        "legal_replay": True,
        "fraction_within_0.25": packed["fraction_within_0.25"],
        "max_linf": packed["max_linf"],
        "steps": int(len(decoded)),
        "seconds": round(time.perf_counter() - started, 3),
        "end_contacts": None if not contacts else {"nL": int(contacts[-1]["nL"]), "nR": int(contacts[-1]["nR"])},
        "during_recovery": out.get("reason") == "DROP_DURING_RECOVERY",
        "after_handoff": out.get("reason") in ("DROP_AFTER_HANDOFF", "LOST_GRASP", "NOMINAL_LIFT_NOT_COMPLETED"),
    }


def diagnose_old(env, book):
    """Replay the gated inward tapes. The 3 mm gate is not applied."""
    path = OUT / "old_gate_replay.jsonl"
    source = _load_jsonl(OLD / "inward_reseat" / "search.jsonl")
    done = _done_keys(path)
    counts = Counter()
    replayed = []
    for row in source:
        key = f"old|{row['id']}"
        if key in done:
            continue
        blob = _load_blob(row_snapshot(row["id"]))
        spec = {"name": "inward_reseat", "ref": blob["ref"]}
        from envs.simulation.capture import restore_capture

        restore_capture(env, blob["capture"])
        memory = {}
        actions = []
        lost_low = 0
        t0 = float(env.sim.data.time)
        phase_steps = Counter()
        stop = "cap"
        for _index in range(CAP_STEPS):
            if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
                stop = "limit"
                break
            _action, phase = inward_reseat_action(env.sim, spec, memory)
            phase_steps[phase] += 1
            if phase == "done":
                stop = "sequence_done"
                break
            step_legal(env.sim, _action)
            actions.append(np.asarray(_action, np.float32))
            failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
            if failed:
                stop = "recovery_failed"
                break
        stacked = np.stack(actions, axis=0) if actions else np.zeros((0, 7), np.float32)
        gates = list(memory.get("gates") or [])
        record = {
            "key": key,
            "id": row["id"],
            "split": row["split"],
            "logged_reason": row.get("reason"),
            "logged_corrected_m": row.get("corrected_m"),
            "logged_steps": row.get("proposal_steps"),
            "replay_steps": int(len(stacked)),
            "stop": stop,
            "gates": gates,
            "retighten_steps": int(phase_steps.get("retighten", 0)),
        }
        for gate in gates:
            counts[gate] += 1
        if row.get("reason") == "error_not_reduced" and len(stacked):
            verified = _verify_prefix(env, book, blob, stacked)
            record["ungated_verify"] = verified
            replayed.append(verified.get("reason"))
            print(
                "REPLAY", row["id"], "logged", row.get("reason"),
                "gates", ",".join(gates), "now", verified.get("reason"),
                "s", verified.get("seconds"), flush=True,
            )
        else:
            print("REPLAY", row["id"], "logged", row.get("reason"), "gates", ",".join(gates), "stop", stop, flush=True)
        _append(path, record)
        done.add(key)
    summary = {
        "logged": dict(Counter(row.get("reason") for row in source)),
        "gates_observed": dict(counts),
        "ungated_reasons": dict(Counter(replayed)),
    }
    (OUT / "old_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("OLD_GATE_SUMMARY", summary, flush=True)
    return summary


def row_snapshot(episode_id: str) -> str:
    for row in _load_jsonl(OLD / "candidates.jsonl"):
        if row.get("id") == episode_id:
            return row["snapshot"]
    raise KeyError(episode_id)


def _candidate_map():
    return {row["id"]: row for row in _load_jsonl(OLD / "candidates.jsonl")}


def select_ics():
    """Nominal-failing ballistic ICs that the old inward search actually attempted."""
    logged = _load_jsonl(OLD / "inward_reseat" / "search.jsonl")
    chosen = {"train": [], "valid": []}
    ranked = sorted(
        [row for row in logged if row.get("reason") == "error_not_reduced"],
        key=lambda row: abs(float(row.get("ex_start") or 0.0)),
        reverse=True,
    )
    for row in ranked:
        split = "valid" if row.get("split") == "valid" else "train"
        cap = N_VALID if split == "valid" else N_TRAIN
        if len(chosen[split]) >= cap:
            continue
        if any(item["id"] == row["id"] for item in chosen[split]):
            continue
        chosen[split].append(row)
    return chosen["valid"] + chosen["train"]


def _schedule_menu(book):
    menu = []
    for seed in SEEDS:
        for sign, label in ((1.0, "pos"), (-1.0, "neg")):
            actions, cuts, illegal = build_schedule(seed, sign, book)
            menu.append({
                "name": f"{seed['name']}_{label}",
                "seed": seed["name"],
                "wy_sign": sign,
                "actions": actions,
                "cuts": cuts,
                "illegal": illegal,
            })
    return menu


def _priority_cuts(trial):
    """Handoff indices strictly before a fall. Later phases are not required."""
    drop = trial["summary"]["drop_step"]
    usable = []
    for cut in trial["cuts"]:
        end = int(cut["end"])
        if drop is not None and end >= int(drop):
            continue
        if end <= 0:
            continue
        usable.append(cut)
    # Prefer the latest legal handoff, then the handoff at the end of relaxation.
    ordered = []
    for tag in ("retighten", "return", "relax", "brake", "reorient"):
        matches = [item for item in usable if item["tag"] == tag]
        if matches:
            ordered.append(matches[-1])
    seen = set()
    unique = []
    for item in ordered:
        if item["end"] in seen:
            continue
        seen.add(item["end"])
        unique.append(item)
    if not unique and drop is not None and int(drop) > 1:
        unique.append({"tag": "before_drop", "end": int(drop) - 1})
    return unique[:2]


def search(env, book):
    path = OUT / "search.jsonl"
    done = _done_keys(path)
    menu = _schedule_menu(book)
    ics = select_ics()
    candidates = _candidate_map()
    slide_rows = [row for row in _load_jsonl(OLD / "search.jsonl") if row.get("strategy") == "slide_50"]
    slide_by_id = {}
    for row in slide_rows:
        slide_by_id.setdefault(row["id"], row.get("reason"))
    verifies = 0
    successes = []
    stage_seconds = Counter()
    started = time.perf_counter()
    print("STAGE_A_ICS", [row["id"] for row in ics], flush=True)
    promising = []
    for row in ics:
        if verifies >= STAGE_A_VERIFIES:
            break
        blob = _load_blob(row_snapshot(row["id"]))
        candidate = candidates[row["id"]]
        baseline_key = f"baseline|{row['id']}"
        if baseline_key not in done:
            _append(path, {
                "key": baseline_key,
                "stage": "baseline",
                "id": row["id"],
                "split": row["split"],
                "immediate_handoff_reason": candidate.get("reason"),
                "immediate_handoff_class": candidate.get("nominal_class"),
                "slide_50_reason": slide_by_id.get(row["id"]),
                "speed_mps": candidate.get("speed_mps"),
                "z_off_m": candidate.get("z_off_m"),
            })
            done.add(baseline_key)
        ic_success = False
        survived = []
        for item in menu:
            if item["illegal"]:
                key = f"a|{row['id']}|{item['name']}|illegal"
                if key not in done:
                    _append(path, {
                        "key": key, "stage": "A", "id": row["id"], "split": row["split"],
                        "schedule": item["name"], "reason": item["illegal"], "success": False,
                    })
                    done.add(key)
                continue
            actions, rows, drop_step = rollout(env, blob, item["actions"])
            summary = _trace_summary(rows, drop_step, item["cuts"])
            trial = {**item, "actions": actions, "summary": summary}
            survived.append(trial)
            print(
                "ROLL", row["id"], item["name"],
                "inward_mm", round(summary["inward_mm"], 2),
                "bilat", round(summary["bilateral_fraction"], 2),
                "drop", drop_step, flush=True,
            )
        # Verify the historical seed both signs, plus the longest-lived other tape.
        must = [trial for trial in survived if trial["seed"] == "hist_simultaneous"]
        others = [trial for trial in survived if trial["seed"] != "hist_simultaneous"]
        others.sort(key=lambda trial: (
            trial["summary"]["drop_step"] is None,
            trial["summary"]["steps"],
            trial["summary"]["inward_mm"],
        ), reverse=True)
        picks = must + others[:1]
        for trial in picks:
            if ic_success or verifies >= STAGE_A_VERIFIES:
                break
            for cut in _priority_cuts(trial):
                key = f"a|{row['id']}|{trial['name']}|{cut['tag']}|{cut['end']}"
                if key in done:
                    continue
                if verifies >= STAGE_A_VERIFIES:
                    break
                t_verify = time.perf_counter()
                outcome = _verify_prefix(env, book, blob, trial["actions"][: int(cut["end"])])
                elapsed = time.perf_counter() - t_verify
                stage_seconds["A"] += elapsed
                verifies += 1
                record = {
                    "key": key,
                    "stage": "A",
                    "id": row["id"],
                    "split": row["split"],
                    "schedule": trial["name"],
                    "seed": trial["seed"],
                    "wy_sign": trial["wy_sign"],
                    "handoff_tag": cut["tag"],
                    "handoff_step": int(cut["end"]),
                    "trace": trial["summary"],
                    **outcome,
                }
                if outcome.get("success"):
                    _save_success(env, book, blob, row, trial["actions"][: int(cut["end"])], record)
                _append(path, record)
                done.add(key)
                print(
                    "VERIFY_A", row["id"], trial["name"], cut["tag"], outcome.get("reason"),
                    "inward_mm", round(trial["summary"]["inward_mm"], 2),
                    "s", outcome.get("seconds"), flush=True,
                )
                if outcome.get("success"):
                    ic_success = True
                    successes.append(record)
                    break
            if not ic_success:
                promising.append((row, picks[0] if picks else None))
        if not ic_success and others:
            promising.append((row, others[0]))
    stage_b(env, book, promising, path, done, stage_seconds)
    stage_c(env, book, path, done, stage_seconds)
    wall = time.perf_counter() - started
    summary = _search_summary(path, wall, stage_seconds)
    (OUT / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SEARCH_SUMMARY", json.dumps(summary), flush=True)
    return summary


def _save_success(env, book, blob, row, actions, record):
    folder = OUT / "verified"
    folder.mkdir(parents=True, exist_ok=True)
    tokens, residuals, decoded, _packed = legalize(actions, book)
    stem = f"{row['id']}__{record['schedule']}__{record['handoff_tag']}"
    path = folder / f"{stem}.npz"
    _refuse(path)
    np.savez(
        path,
        actions=np.asarray(decoded, np.float32),
        tokens=np.asarray(tokens, np.int64),
        residuals=np.asarray(residuals, np.float32),
    )
    record["npz"] = str(path.relative_to(ROOT)).replace("\\", "/")


def stage_b(env, book, promising, path, done, stage_seconds):
    """Lengthen or shorten pitch and relaxation on tapes that stayed in hand."""
    used = 0
    seen = set()
    for row, trial in promising:
        if trial is None or used >= STAGE_B_VERIFIES:
            break
        if row["id"] in seen:
            continue
        if trial["summary"]["drop_step"] is not None:
            continue
        seen.add(row["id"])
        blob = _load_blob(row_snapshot(row["id"]))
        seed = next(item for item in SEEDS if item["name"] == trial["seed"])
        for pitch_scale, relax_scale in ((0.5, 1.0), (1.5, 1.0), (1.0, 0.5), (1.0, 1.5)):
            if used >= STAGE_B_VERIFIES:
                break
            variant = {
                "name": trial["seed"],
                "parts": tuple(_scale_part(part, pitch_scale, relax_scale) for part in seed["parts"]),
            }
            actions, cuts, illegal = build_schedule(variant, trial["wy_sign"], book)
            name = f"{trial['name']}_p{pitch_scale}_r{relax_scale}"
            key = f"b|{row['id']}|{name}"
            if key in done or illegal or actions is None:
                continue
            _actions, rows, drop_step = rollout(env, blob, actions)
            summary = _trace_summary(rows, drop_step, cuts)
            usable = [item for item in cuts if drop_step is None or item["end"] < drop_step]
            finals = [item for item in usable if item["tag"] in ("retighten", "return", "relax")]
            if not finals:
                continue
            cut = finals[-1]
            t0 = time.perf_counter()
            outcome = _verify_prefix(env, book, blob, actions[: int(cut["end"])])
            stage_seconds["B"] += time.perf_counter() - t0
            used += 1
            record = {
                "key": key, "stage": "B", "id": row["id"], "split": row["split"],
                "schedule": name, "seed": trial["seed"], "wy_sign": trial["wy_sign"],
                "handoff_tag": cut["tag"], "handoff_step": int(cut["end"]),
                "trace": summary, **outcome,
            }
            _append(path, record)
            done.add(key)
            print("VERIFY_B", row["id"], name, outcome.get("reason"), flush=True)
            if outcome.get("success"):
                _save_success(env, book, blob, row, actions[: int(cut["end"])], record)
                break


def _scale_part(part, pitch_scale, relax_scale):
    scaled = dict(part)
    if part["tag"] == "reorient":
        scaled["s"] = float(part["s"]) * float(pitch_scale)
    elif part["tag"] == "relax":
        scaled["s"] = float(part["s"]) * float(relax_scale)
    return scaled


def stage_c(env, book, path, done, stage_seconds):
    """Transfer a successful phase structure. Signs are recomputed on the target."""
    rows = _load_jsonl(path)
    wins = [row for row in rows if row.get("success") and row.get("stage") in ("A", "B")]
    if not wins:
        print("STAGE_C_NO_SEED", flush=True)
        return
    seed_row = wins[0]
    source = _load_blob(row_snapshot(seed_row["id"]))
    source_rh = _finger_at(env, source)
    source_sign = float(seed_row["wy_sign"])
    relation = "same" if source_sign == (1.0 if source_rh > 0.0 else -1.0) else "opposite"
    logged = _load_jsonl(OLD / "inward_reseat" / "search.jsonl")
    targets = []
    for row in logged:
        if row["id"] == seed_row["id"] or row.get("reason") != "error_not_reduced":
            continue
        if (row.get("split") == "valid") != (seed_row.get("split") == "valid"):
            continue
        targets.append(row)
        if len(targets) >= 4:
            break
    seed = next(item for item in SEEDS if item["name"] == seed_row["seed"])
    used = 0
    for row in targets:
        if used >= STAGE_C_VERIFIES:
            break
        blob = _load_blob(row_snapshot(row["id"]))
        rh = _finger_at(env, blob)
        target_sign = 1.0 if rh > 0.0 else -1.0
        wy_sign = target_sign if relation == "same" else -target_sign
        actions, cuts, illegal = build_schedule(seed, wy_sign, book)
        key = f"c|{row['id']}|{seed_row['seed']}|{wy_sign}"
        if key in done or illegal or actions is None:
            continue
        _actions, trace_rows, drop_step = rollout(env, blob, actions)
        summary = _trace_summary(trace_rows, drop_step, cuts)
        usable = [item for item in cuts if drop_step is None or item["end"] < drop_step]
        finals = [item for item in usable if item["tag"] in ("retighten", "return", "relax")]
        if not finals:
            _append(path, {
                "key": key, "stage": "C", "id": row["id"], "split": row["split"],
                "schedule": seed_row["seed"], "wy_sign": wy_sign, "reason": "dropped_before_handoff",
                "success": False, "trace": summary,
            })
            done.add(key)
            continue
        cut = finals[-1]
        t0 = time.perf_counter()
        outcome = _verify_prefix(env, book, blob, actions[: int(cut["end"])])
        stage_seconds["C"] += time.perf_counter() - t0
        used += 1
        record = {
            "key": key, "stage": "C", "id": row["id"], "split": row["split"],
            "schedule": f"{seed_row['seed']}_transfer", "seed": seed_row["seed"],
            "wy_sign": wy_sign, "source_id": seed_row["id"], "sign_relation": relation,
            "handoff_tag": cut["tag"], "handoff_step": int(cut["end"]),
            "trace": summary, **outcome,
        }
        _append(path, record)
        done.add(key)
        print("VERIFY_C", row["id"], outcome.get("reason"), flush=True)
        if outcome.get("success"):
            _save_success(env, book, blob, row, actions[: int(cut["end"])], record)


def _finger_at(env, blob) -> float:
    from envs.simulation.capture import restore_capture

    restore_capture(env, blob["capture"])
    return float(_finger_x(env.sim))


def historical_reference(env, book):
    """6.5 m/s, -6 mm impact, then the historical phase structure under step_legal."""
    path = OUT / "historical_reference.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    spec = {
        "azimuth_deg": 42.0,
        "speed_mps": 6.5,
        "z_off_m": -0.006,
        "phi_deg": 0.0,
        "delay_s": 0.0,
        "direction": "continuous",
        "family": "ballistic_impact",
        "protocol": "k48_inward_sliding_search_v1_diagnostic",
        "split": "diagnostic",
        "attempt_id": "hist_6p5_m6mm",
    }
    evidence, blob = run_episode(env, spec)
    record = {
        "launch": {key: spec[key] for key in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")},
        "impact_category": evidence.get("category"),
        "impact_reason": evidence.get("reason"),
        "entry": evidence.get("entry"),
    }
    if blob is None:
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print("HISTORICAL_NO_CAPTURE", record["impact_category"], record["impact_reason"], flush=True)
        return record
    snap = OUT / "diagnostic"
    snap.mkdir(parents=True, exist_ok=True)
    with (snap / "hist_6p5_m6mm.pkl").open("wb") as handle:
        pickle.dump(blob, handle)
    immediate = legal_verify(env, blob, np.zeros((0, 7), np.float32))
    record["immediate_handoff"] = immediate[0].get("reason")
    actions, cuts, illegal = build_schedule(SEEDS[0], 1.0, book)
    record["schedule_illegal"] = illegal
    if actions is not None:
        _executed, rows, drop_step = rollout(env, blob, actions)
        record["trace"] = _trace_summary(rows, drop_step, cuts)
        end = None if drop_step is not None else int(cuts[-1]["end"])
        if end:
            outcome = _verify_prefix(env, book, blob, actions[:end])
            record["k48_replay"] = {key: outcome[key] for key in (
                "reason", "success", "nominal_elapsed", "lift_completed", "qpos_error", "steps", "seconds",
            )}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("HISTORICAL", json.dumps(record), flush=True)
    return record


def _search_summary(path: Path, wall: float, stage_seconds: Counter):
    rows = _load_jsonl(path)
    verifies = [row for row in rows if row.get("stage") in ("A", "B", "C") and "reason" in row and row.get("steps") is not None]
    wins = [row for row in verifies if row.get("success")]
    by_stage = {}
    for stage in ("A", "B", "C"):
        subset = [row for row in verifies if row.get("stage") == stage]
        ids = {row["id"] for row in subset if row.get("success")}
        by_stage[stage] = {
            "verifies": len(subset),
            "successes": len([row for row in subset if row.get("success")]),
            "seconds": round(float(stage_seconds.get(stage, 0.0)), 1),
            "reasons": dict(Counter(row.get("reason") for row in subset)),
            "train_ids": sorted(row["id"] for row in subset if row.get("success") and row.get("split") != "valid"),
            "valid_ids": sorted(row["id"] for row in subset if row.get("success") and row.get("split") == "valid"),
            "distinct_success_ids": sorted(ids),
        }
    small = []
    worsened = []
    sliding = []
    for row in wins:
        trace = row.get("trace") or {}
        inward = trace.get("inward_mm")
        if inward is not None and float(inward) < 3.0:
            small.append({"id": row["id"], "schedule": row.get("schedule"), "inward_mm": inward})
        if inward is not None and float(inward) < 0.0:
            worsened.append({"id": row["id"], "schedule": row.get("schedule"), "inward_mm": inward})
        relax = trace.get("relax") or {}
        if relax and float(relax.get("inward_mm") or 0.0) > 0.5 and float(relax.get("bilateral_fraction") or 0.0) >= 0.5:
            sliding.append({"id": row["id"], "schedule": row.get("schedule"), "relax": relax})
    during = sum(1 for row in verifies if row.get("during_recovery"))
    after = sum(1 for row in verifies if row.get("after_handoff"))
    train_ids = sorted({row["id"] for row in wins if row.get("split") != "valid"})
    valid_ids = sorted({row["id"] for row in wins if row.get("split") == "valid"})
    return {
        "wall_s": round(wall, 1),
        "stage_seconds": {key: round(float(value), 1) for key, value in stage_seconds.items()},
        "verifies": len(verifies),
        "during_recovery_failures": during,
        "after_handoff_failures": after,
        "train_success_ids": train_ids,
        "valid_success_ids": valid_ids,
        "success_with_under_3mm": small,
        "success_with_worsened_offset": worsened,
        "success_with_pad_slide": sliding,
        "stages": by_stage,
        "wins": [{
            "id": row["id"], "split": row.get("split"), "stage": row.get("stage"),
            "schedule": row.get("schedule"), "reason": row.get("reason"),
            "inward_mm": (row.get("trace") or {}).get("inward_mm"),
            "handoff_tag": row.get("handoff_tag"),
        } for row in wins],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    if not (args.diagnose or args.historical or args.search):
        args.diagnose = args.historical = args.search = True
    OUT.mkdir(parents=True, exist_ok=True)
    env = _open_env()
    book = load_codebook()
    if args.diagnose:
        diagnose_old(env, book)
    if args.historical:
        historical_reference(env, book)
    if args.search:
        search(env, book)


if __name__ == "__main__":
    main()
