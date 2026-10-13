"""Bounded legal K48 controlled-sliding search.

Success is only a codebook-legal recovery followed by the unchanged nominal
controller holding the grasp for 12 seconds. Geometric thresholds are logged
and used to order trials. They do not accept or reject a rollout.

Writes under demo_results/diagnostics/raw/k48_controlled_slide_search_v1.
Does not train and does not modify k48_recovery_v3 or data/.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.geometry_switch import hand_pose
from controllers.recovery_actions import AIRBORNE_V3, grip_action, grip_tau, normalize_action
from envs.physical_recovery import RH_Z_NOM, physical_pack
from training.ball_impact_dataset import run_episode
from training.horizon_outcome import recovery_failed
from training.k48_inward_sliding_search import _priority_cuts, _verify_prefix
from training.privileged_ball_impact_search import representability, step_legal
from training.rebuild_k48_recovery import (
    CAP_STEPS,
    LIMIT_S,
    _load_blob,
    _load_jsonl,
    _open_env,
    controlled_slide_action,
)
from training.unified_k47 import load_codebook

OLD = ROOT / "demo_results/diagnostics/raw/k48_recovery_v3"
OUT = ROOT / "demo_results/diagnostics/raw/k48_controlled_slide_search_v1"
EARLY_PKL = ROOT / "demo_results/diagnostics/ballistic_recovery_transfer/raw/snap_early.pkl"
DT = 0.02
STAGE_A_CAP = 36
STAGE_B_CAP = 8
STAGE_D_CAP = 4

# Historical CENTER-6 phases, transcribed into K48 rates. +3 rad/s for 0.70 s
# is about +120 degrees. Negative pitch at 3 rad/s is outside the codebook,
# so the opposite family uses the legal -1 rad/s rate. Grip -0.30 is tau=-5.
SEEDS = (
    {
        "name": "center6_pos",
        "parts": (
            {"tag": "reorient", "s": 0.70, "wx": 0.0, "wy": 3.0, "grip": 1.0},
            {"tag": "relax", "s": 0.70, "wx": 0.0, "wy": 0.0, "grip": None},
            {"tag": "brake", "s": 0.40, "wx": 0.0, "wy": 0.0, "grip": 1.0},
            {"tag": "return", "s": 0.70, "wx": 0.0, "wy": -1.0, "grip": 1.0},
        ),
    },
    {
        "name": "center6_neg",
        "parts": (
            {"tag": "reorient", "s": 1.20, "wx": 0.0, "wy": -1.0, "grip": 1.0},
            {"tag": "relax", "s": 0.70, "wx": 0.0, "wy": 0.0, "grip": None},
            {"tag": "brake", "s": 0.40, "wx": 0.0, "wy": 0.0, "grip": 1.0},
        ),
    },
    {
        "name": "short_pos",
        "parts": (
            {"tag": "reorient", "s": 0.40, "wx": 0.0, "wy": 3.0, "grip": 1.0},
            {"tag": "relax", "s": 0.48, "wx": 0.0, "wy": 0.0, "grip": -0.48},
            {"tag": "brake", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0},
        ),
    },
    {
        "name": "short_neg",
        "parts": (
            {"tag": "reorient", "s": 0.80, "wx": 0.0, "wy": -1.0, "grip": 1.0},
            {"tag": "relax", "s": 0.48, "wx": 0.0, "wy": 0.0, "grip": -0.48},
            {"tag": "retighten", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0},
        ),
    },
    {
        "name": "no_return_pos",
        "parts": (
            {"tag": "reorient", "s": 0.70, "wx": 0.0, "wy": 3.0, "grip": 1.0},
            {"tag": "relax", "s": 0.70, "wx": 0.0, "wy": 0.0, "grip": None},
            {"tag": "retighten", "s": 0.16, "wx": 0.0, "wy": 0.0, "grip": 1.0},
        ),
    },
)


def _append(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def _done(path: Path) -> set[str]:
    return {row.get("key") for row in _load_jsonl(path) if row.get("key")}


def _wrist(wx, wy, grip) -> np.ndarray:
    action = normalize_action(np.zeros(3), np.array([wx, wy, 0.0], float), grip_tau(float(grip)), AIRBORNE_V3)
    return np.asarray(action, np.float32)


def _grip(part) -> float:
    if part["grip"] is None:
        return float(grip_action(-5.0))
    return float(part["grip"])


def _legal(book, action) -> bool:
    packed = representability(np.asarray(action, np.float32).reshape(1, 7), book)
    return float(packed["max_linf"]) <= 0.25 + 1e-8


def _steps(seconds: float) -> int:
    return max(1, int(round(float(seconds) / DT)))


def build_schedule(seed: dict, book):
    actions = []
    cuts = []
    for part in seed["parts"]:
        action = _wrist(part["wx"], part["wy"], _grip(part))
        if not _legal(book, action):
            return None, [], f"illegal wx={part['wx']} wy={part['wy']} grip={_grip(part)}"
        actions.extend([action] * _steps(part["s"]))
        cuts.append({"tag": part["tag"], "end": len(actions)})
    if len(actions) > CAP_STEPS:
        actions = actions[:CAP_STEPS]
        cuts = [item for item in cuts if item["end"] <= CAP_STEPS]
    return np.stack(actions, axis=0), cuts, None


def _pad_contacts(sim):
    """Hand-frame contact positions on the two finger bodies. Logging only."""
    model, data, ids = sim.model, sim.data, sim.ids
    obj = int(ids.object_body)
    hand = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    origin = np.array(data.xpos[ids.hand_body], float)
    points = {"L": [], "R": []}
    forces = {"L": 0.0, "R": 0.0}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        body_1 = int(model.geom_bodyid[int(contact.geom1)])
        body_2 = int(model.geom_bodyid[int(contact.geom2)])
        if obj not in (body_1, body_2):
            continue
        other = body_2 if body_1 == obj else body_1
        side = "L" if other == int(ids.left_body) else "R" if other == int(ids.right_body) else None
        if side is None:
            continue
        pos = hand.T @ (np.array(contact.pos, float) - origin)
        if len(points[side]) < 2:
            points[side].append([round(float(x), 5) for x in pos])
        frame = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, frame)
        forces[side] += float(frame[0])
    return points, forces


def _measure(sim, ref):
    pack = physical_pack(sim)
    rh = np.asarray(pack["rh"], float)
    pinch, _relative = hand_pose(sim)
    nominal = np.asarray(ref["p_nom"], float).reshape(-1)
    points, forces = _pad_contacts(sim)
    return {
        "rh": [round(float(x), 6) for x in rh],
        "rhx": float(rh[0]),
        "rhy": float(rh[1]),
        "rhz_from_nom": float(rh[2] - RH_Z_NOM),
        "pinch_world": [round(float(x), 6) for x in np.asarray(pinch, float)[:3]],
        "world_ex": float(np.asarray(pinch, float)[0] - nominal[0]),
        "gx": float(np.asarray(pack["g_h"], float)[0]),
        "vx": float(np.asarray(pack["v_rel_h"], float)[0]),
        "vy": float(np.asarray(pack["v_rel_h"], float)[1]),
        "wx": float(np.asarray(pack["w_rel_h"], float)[0]),
        "wy": float(np.asarray(pack["w_rel_h"], float)[1]),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "fnL": float(pack["Fn_L"]),
        "fnR": float(pack["Fn_R"]),
        "ftL": round(forces["L"], 4),
        "ftR": round(forces["R"], 4),
        "z": float(pack["obj_z"]),
        "tau": float(pack["tau"]),
        "pads": points,
    }


def rollout(env, blob, actions):
    from training.history_aware_critic_ablation import restore_capture

    restore_capture(env, blob["capture"])
    rows = [_measure(env.sim, blob["ref"])]
    lost_low = 0
    drop_step = None
    t0 = float(env.sim.data.time)
    executed = []
    for index, action in enumerate(np.asarray(actions, np.float32)):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            break
        step_legal(env.sim, action)
        executed.append(np.asarray(action, np.float32))
        rows.append(_measure(env.sim, blob["ref"]))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            drop_step = index + 1
            break
    stacked = np.stack(executed, axis=0) if executed else np.zeros((0, 7), np.float32)
    return stacked, rows, drop_step


def _summary(rows, drop_step, cuts):
    start, end = rows[0], rows[-1]
    bilateral = [row["nL"] > 0 and row["nR"] > 0 for row in rows]
    unilateral = [(row["nL"] > 0) ^ (row["nR"] > 0) for row in rows]
    none = [row["nL"] == 0 and row["nR"] == 0 for row in rows]
    relax = None
    cursor = 0
    for cut in cuts:
        if cut["tag"] == "relax":
            window = rows[cursor:min(cut["end"], len(rows) - 1) + 1]
            if window:
                relax = {
                    "rhx_start": window[0]["rhx"],
                    "rhx_end": window[-1]["rhx"],
                    "inward_mm": 1e3 * (abs(window[0]["rhx"]) - abs(window[-1]["rhx"])),
                    "bilateral_fraction": float(np.mean([row["nL"] > 0 and row["nR"] > 0 for row in window])),
                    "max_abs_vx": float(max(abs(row["vx"]) for row in window)),
                }
            break
        cursor = cut["end"]
    rh_delta = abs(start["rhx"]) - abs(end["rhx"])
    world_delta = abs(start["world_ex"]) - abs(end["world_ex"])
    mechanism = "other"
    if relax and relax["bilateral_fraction"] >= 0.5 and abs(relax["inward_mm"]) >= 0.5:
        mechanism = "controlled_slide"
    elif any(none) and bilateral[-1]:
        mechanism = "recapture"
    elif float(np.mean(bilateral)) > 0.8 and abs(rh_delta) < 0.0005:
        mechanism = "held_without_slide"
    return {
        "rhx_start": start["rhx"],
        "rhx_end": end["rhx"],
        "rhz_from_nom_start": start["rhz_from_nom"],
        "world_ex_start": start["world_ex"],
        "world_ex_end": end["world_ex"],
        "hand_correction_mm": 1e3 * rh_delta,
        "world_correction_mm": 1e3 * world_delta,
        "gx_min": float(min(row["gx"] for row in rows)),
        "gx_max": float(max(row["gx"] for row in rows)),
        "bilateral_fraction": float(np.mean(bilateral)),
        "unilateral_fraction": float(np.mean(unilateral)),
        "contact_free_fraction": float(np.mean(none)),
        "max_abs_vx": float(max(abs(row["vx"]) for row in rows)),
        "fnL_end": end["fnL"],
        "fnR_end": end["fnR"],
        "pads_start": start["pads"],
        "pads_end": end["pads"],
        "drop_step": drop_step,
        "steps": int(len(rows) - 1),
        "relax": relax,
        "mechanism": mechanism,
    }


def _baseline_actions(env, blob):
    from training.history_aware_critic_ablation import restore_capture

    restore_capture(env, blob["capture"])
    spec = {"name": "controlled_slide", "ref": blob["ref"]}
    memory = {}
    actions = []
    lost_low = 0
    t0 = float(env.sim.data.time)
    stop = "cap"
    for _index in range(CAP_STEPS):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            stop = "limit"
            break
        action, phase = controlled_slide_action(env.sim, spec, memory)
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


def _candidate_map():
    return {row["id"]: row for row in _load_jsonl(OLD / "candidates.jsonl")}


def _select(env):
    from training.history_aware_critic_ablation import restore_capture

    rows = [row for row in _load_jsonl(OLD / "candidates.jsonl") if row.get("nominal_class") == "RECOVERY_CANDIDATE"]
    rows.sort(key=lambda row: (0 if row.get("split") == "valid" else 1, row["id"]))
    chosen = []
    for row in rows:
        if len(chosen) >= 16:
            break
        blob = _load_blob(row["snapshot"])
        restore_capture(env, blob["capture"])
        state = _measure(env.sim, blob["ref"])
        bilateral = state["nL"] > 0 and state["nR"] > 0
        chosen.append({**row, "bilateral": bilateral, "rhx": state["rhx"], "world_ex": state["world_ex"]})
    bilateral = [row for row in chosen if row["bilateral"]]
    other = [row for row in chosen if not row["bilateral"]]
    valid = [row for row in bilateral if row.get("split") == "valid"][:3]
    train = [row for row in bilateral if row.get("split") != "valid"][:5]
    extra = other[:2]
    return valid + train + extra


def _menu(book):
    menu = []
    for seed in SEEDS:
        actions, cuts, illegal = build_schedule(seed, book)
        menu.append({"name": seed["name"], "seed": seed, "actions": actions, "cuts": cuts, "illegal": illegal})
    return menu


def _save_success(book, row, actions, record):
    from training.rebuild_k48_recovery import legalize

    folder = OUT / "verified"
    folder.mkdir(parents=True, exist_ok=True)
    tokens, residuals, decoded, _packed = legalize(actions, book)
    stem = f"{row['id']}__{record['schedule']}__{record.get('handoff_tag') or 'end'}"
    path = folder / f"{stem}.npz"
    np.savez(
        path,
        actions=np.asarray(decoded, np.float32),
        tokens=np.asarray(tokens, np.int64),
        residuals=np.asarray(residuals, np.float32),
    )
    record["npz"] = str(path.relative_to(ROOT)).replace("\\", "/")


def search(env, book):
    path = OUT / "search.jsonl"
    done = _done(path)
    menu = _menu(book)
    ics = _select(env)
    candidates = _candidate_map()
    verifies = 0
    stage_seconds = Counter()
    started = time.perf_counter()
    print("ICS", [(row["id"], row["split"], row["bilateral"], round(1e3 * row["rhx"], 1)) for row in ics], flush=True)
    promising = []
    successes = []
    for row in ics:
        if verifies >= STAGE_A_CAP:
            break
        blob = _load_blob(row["snapshot"])
        candidate = candidates[row["id"]]
        base_key = f"baseline|{row['id']}|controlled_slide_deterministic"
        ic_success = False
        if base_key not in done and verifies < STAGE_A_CAP:
            actions, stop = _baseline_actions(env, blob)
            t0 = time.perf_counter()
            if len(actions) == 0:
                outcome = {"reason": "no_actions", "success": False, "seconds": 0.0}
            elif stop == "recovery_failed":
                outcome = {"reason": "DROP_DURING_RECOVERY", "success": False, "during_recovery": True, "seconds": 0.0, "steps": int(len(actions))}
            else:
                outcome = _verify_prefix(env, book, blob, actions)
                verifies += 1
            stage_seconds["A"] += time.perf_counter() - t0
            record = {
                "key": base_key, "stage": "A", "id": row["id"], "split": row["split"],
                "schedule": "deterministic_baseline", "handoff_tag": "recorded_end",
                "rhx_start": row["rhx"], "world_ex_start": row["world_ex"],
                "immediate_handoff_reason": candidate.get("reason"), **outcome,
            }
            _append(path, record)
            done.add(base_key)
            print("BASE", row["id"], outcome.get("reason"), flush=True)
            if outcome.get("success"):
                _save_success(book, row, actions, record)
                ic_success = True
                successes.append(record)
        if ic_success:
            continue
        survived = []
        for item in menu:
            if item["illegal"]:
                key = f"a|{row['id']}|{item['name']}|illegal"
                if key not in done:
                    _append(path, {"key": key, "stage": "A", "id": row["id"], "schedule": item["name"], "reason": item["illegal"], "success": False})
                    done.add(key)
                continue
            _executed, rows, drop_step = rollout(env, blob, item["actions"])
            summary = _summary(rows, drop_step, item["cuts"])
            trial = {**item, "summary": summary}
            survived.append(trial)
            print(
                "ROLL", row["id"], item["name"], summary["mechanism"],
                "hand_mm", round(summary["hand_correction_mm"], 2),
                "world_mm", round(summary["world_correction_mm"], 2),
                "drop", drop_step, flush=True,
            )
        survived.sort(key=lambda trial: (
            trial["summary"]["drop_step"] is None,
            trial["summary"]["bilateral_fraction"],
            trial["summary"]["hand_correction_mm"],
        ), reverse=True)
        for trial in survived[:3]:
            if ic_success or verifies >= STAGE_A_CAP:
                break
            for cut in _priority_cuts({"cuts": trial["cuts"], "summary": trial["summary"]}):
                key = f"a|{row['id']}|{trial['name']}|{cut['tag']}|{cut['end']}"
                if key in done or verifies >= STAGE_A_CAP:
                    continue
                t0 = time.perf_counter()
                outcome = _verify_prefix(env, book, blob, trial["actions"][: int(cut["end"])])
                stage_seconds["A"] += time.perf_counter() - t0
                verifies += 1
                record = {
                    "key": key, "stage": "A", "id": row["id"], "split": row["split"],
                    "schedule": trial["name"], "handoff_tag": cut["tag"], "handoff_step": int(cut["end"]),
                    "trace": trial["summary"], "rhx_start": row["rhx"], **outcome,
                }
                if outcome.get("success"):
                    _save_success(book, row, trial["actions"][: int(cut["end"])], record)
                _append(path, record)
                done.add(key)
                print("VERIFY", row["id"], trial["name"], cut["tag"], outcome.get("reason"), flush=True)
                if outcome.get("success"):
                    ic_success = True
                    successes.append(record)
                    break
        if not ic_success and survived:
            promising.append((row, survived[0]))
    _stage_b(env, book, promising, path, done, stage_seconds)
    _stage_d(env, book, path, done, stage_seconds)
    summary = _summarize(path, time.perf_counter() - started, stage_seconds)
    (OUT / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SEARCH_SUMMARY", json.dumps(summary), flush=True)
    return summary


def _stage_b(env, book, promising, path, done, stage_seconds):
    used = 0
    seen = set()
    for row, trial in promising:
        if used >= STAGE_B_CAP or row["id"] in seen:
            continue
        if trial["summary"]["drop_step"] is not None:
            continue
        seen.add(row["id"])
        blob = _load_blob(row["snapshot"])
        for scale in (0.5, 1.5):
            if used >= STAGE_B_CAP:
                break
            variant = {
                "name": trial["name"],
                "parts": tuple(
                    {**part, "s": part["s"] * scale} if part["tag"] in ("reorient", "relax") else part
                    for part in trial["seed"]["parts"]
                ),
            }
            actions, cuts, illegal = build_schedule(variant, book)
            name = f"{trial['name']}_x{scale}"
            key = f"b|{row['id']}|{name}"
            if illegal or actions is None or key in done:
                continue
            _executed, rows, drop_step = rollout(env, blob, actions)
            summary = _summary(rows, drop_step, cuts)
            usable = [item for item in cuts if drop_step is None or item["end"] < drop_step]
            if not usable:
                continue
            cut = usable[-1]
            t0 = time.perf_counter()
            outcome = _verify_prefix(env, book, blob, actions[: int(cut["end"])])
            stage_seconds["B"] += time.perf_counter() - t0
            used += 1
            record = {
                "key": key, "stage": "B", "id": row["id"], "split": row["split"],
                "schedule": name, "handoff_tag": cut["tag"], "trace": summary, **outcome,
            }
            if outcome.get("success"):
                _save_success(book, row, actions[: int(cut["end"])], record)
            _append(path, record)
            done.add(key)
            print("VERIFY_B", row["id"], name, outcome.get("reason"), flush=True)


def _stage_d(env, book, path, done, stage_seconds):
    wins = [row for row in _load_jsonl(path) if row.get("success") and row.get("stage") in ("A", "B")]
    if not wins:
        print("STAGE_D_NO_SEED", flush=True)
        return
    seed_row = wins[0]
    seed = next(item for item in SEEDS if item["name"] == seed_row.get("schedule") or item["name"] == str(seed_row.get("schedule", "")).split("_x")[0])
    source_sign = 1.0 if float(seed_row.get("rhx_start") or 0.0) >= 0.0 else -1.0
    used = 0
    for row in _load_jsonl(OLD / "candidates.jsonl"):
        if used >= STAGE_D_CAP or row["id"] == seed_row["id"] or row.get("nominal_class") != "RECOVERY_CANDIDATE":
            continue
        if row.get("split") != seed_row.get("split"):
            continue
        blob = _load_blob(row["snapshot"])
        from training.history_aware_critic_ablation import restore_capture
        restore_capture(env, blob["capture"])
        rhx = _measure(env.sim, blob["ref"])["rhx"]
        if (1.0 if rhx >= 0.0 else -1.0) != source_sign:
            continue
        actions, cuts, illegal = build_schedule(seed, book)
        key = f"d|{row['id']}|{seed['name']}"
        if illegal or actions is None or key in done:
            continue
        _executed, rows, drop_step = rollout(env, blob, actions)
        summary = _summary(rows, drop_step, cuts)
        usable = [item for item in cuts if drop_step is None or item["end"] < drop_step]
        if not usable:
            continue
        cut = usable[-1]
        t0 = time.perf_counter()
        outcome = _verify_prefix(env, book, blob, actions[: int(cut["end"])])
        stage_seconds["D"] += time.perf_counter() - t0
        used += 1
        record = {
            "key": key, "stage": "D", "id": row["id"], "split": row["split"],
            "schedule": seed["name"] + "_transfer", "source_id": seed_row["id"],
            "handoff_tag": cut["tag"], "trace": summary, **outcome,
        }
        if outcome.get("success"):
            _save_success(book, row, actions[: int(cut["end"])], record)
        _append(path, record)
        done.add(key)
        print("VERIFY_D", row["id"], outcome.get("reason"), flush=True)


def historical(env, book):
    record = {"snap_early_exists": EARLY_PKL.is_file(), "snap_early": str(EARLY_PKL.relative_to(ROOT)).replace("\\", "/")}
    if EARLY_PKL.is_file():
        with EARLY_PKL.open("rb") as handle:
            import pickle
            snap = pickle.load(handle)
        record["snap_early_type"] = type(snap).__name__
        record["snap_early_keys"] = sorted(snap)[:20] if isinstance(snap, dict) else None
        record["legal_k48_replay"] = "not_run"
        record["reason"] = "snap_early is the ballistic-demo snapshot, not a K48 training capture with capture/ref"
    spec = {
        "azimuth_deg": 0.0, "speed_mps": 6.0, "z_off_m": 0.0, "phi_deg": 0.0, "delay_s": 0.0,
        "direction": "continuous", "family": "ballistic_impact",
        "protocol": "controlled_slide_center6_diagnostic", "split": "diagnostic",
        "attempt_id": "center6_k48_6mps",
    }
    evidence, blob = run_episode(env, spec)
    record["k48_launch_category"] = evidence.get("category")
    record["k48_launch_reason"] = evidence.get("reason")
    record["k48_entry"] = evidence.get("entry")
    if blob is not None:
        immediate = _verify_prefix(env, book, blob, np.zeros((0, 7), np.float32))
        record["immediate_handoff"] = immediate.get("reason")
        actions, cuts, illegal = build_schedule(SEEDS[0], book)
        record["schedule_illegal"] = illegal
        if actions is not None:
            _executed, rows, drop_step = rollout(env, blob, actions)
            record["trace"] = _summary(rows, drop_step, cuts)
            usable = [item for item in cuts if drop_step is None or item["end"] < drop_step]
            if usable:
                outcome = _verify_prefix(env, book, blob, actions[: int(usable[-1]["end"])])
                record["k48_replay"] = {key: outcome.get(key) for key in ("reason", "success", "steps", "seconds", "nominal_elapsed")}
    path = OUT / "historical_reference.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("HISTORICAL", json.dumps({key: record[key] for key in record if key != "trace"}), flush=True)
    return record


def _summarize(path: Path, wall: float, stage_seconds: Counter):
    rows = _load_jsonl(path)
    verifies = [row for row in rows if row.get("steps") is not None or row.get("reason") in ("SUCCESS_12S", "DROP_AFTER_HANDOFF", "LOST_GRASP", "DROP_DURING_RECOVERY", "NOMINAL_LIFT_NOT_COMPLETED")]
    wins = [row for row in rows if row.get("success")]
    small = []
    worsened = []
    for row in wins:
        trace = row.get("trace") or {}
        hand = trace.get("hand_correction_mm")
        if hand is not None and abs(float(hand)) < 3.0:
            small.append({"id": row["id"], "schedule": row.get("schedule"), "hand_mm": hand})
        if hand is not None and float(hand) < 0.0:
            worsened.append({"id": row["id"], "schedule": row.get("schedule"), "hand_mm": hand})
    return {
        "wall_s": round(wall, 1),
        "stage_seconds": {key: round(float(value), 1) for key, value in stage_seconds.items()},
        "attempt_rows": len(rows),
        "verifies": len(verifies),
        "during_recovery": sum(1 for row in rows if row.get("during_recovery") or row.get("reason") == "DROP_DURING_RECOVERY"),
        "after_handoff": sum(1 for row in rows if row.get("after_handoff")),
        "success_ids": sorted({row["id"] for row in wins}),
        "train_ids": sorted({row["id"] for row in wins if row.get("split") != "valid"}),
        "valid_ids": sorted({row["id"] for row in wins if row.get("split") == "valid"}),
        "mechanisms": dict(Counter((row.get("trace") or {}).get("mechanism") for row in rows if row.get("trace"))),
        "under_3mm_success": small,
        "worsened_hand_offset_success": worsened,
        "reasons": dict(Counter(row.get("reason") for row in rows)),
        "wins": [{
            "id": row["id"], "split": row.get("split"), "stage": row.get("stage"),
            "schedule": row.get("schedule"), "handoff_tag": row.get("handoff_tag"),
            "hand_mm": (row.get("trace") or {}).get("hand_correction_mm"),
            "world_mm": (row.get("trace") or {}).get("world_correction_mm"),
            "mechanism": (row.get("trace") or {}).get("mechanism"),
        } for row in wins],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = _open_env()
    book = load_codebook()
    historical(env, book)
    search(env, book)


if __name__ == "__main__":
    main()
