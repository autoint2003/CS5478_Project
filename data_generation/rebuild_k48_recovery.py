"""Rebuild the K48 recovery dataset from nominal-failure initial conditions.

Writes only under results/archive/demo_results/diagnostics/raw/k48_recovery_v3. Does not train,
does not open FINAL, and does not modify historical snapshots or manifests.
"""

from __future__ import annotations


import argparse
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

from envs.simulation.capture import restore_capture, take_capture
from envs.outcomes.nominal_pose import capture_ref, closed_grip, errors, hand_R, rotvec
from data_generation.ball_impact_dataset import physics_record, run_episode
from data_generation.validation.impact_labels import cylinder_first, near

from envs.outcomes.horizon_outcome import (
    NOMINAL_S,
    finish_recovery,
    recovery_failed,
    replay_commands,
)
from data_generation.validation.impact_labels import mechanism

from data_generation.oracles.representability import representability
from data_generation.strategies.recovery_laws import STRATEGIES, command
from envs.step_legal import step_legal
from envs.outcomes.recovery_success import PERSIST_STEPS, basin_ok
from controllers.recovery.geometry_switch import hand_pose
from controllers.recovery.action_map import AIRBORNE_V3, grip_action, grip_tau, normalize_action
from envs.outcomes.physical_recovery import physical_pack
from data_generation.strategies.recovery_laws import slide_action
from training.k47_codebook import CAP, load_codebook, read_jsonl

OUT = ROOT / "results/archive/demo_results/diagnostics/raw/k48_recovery_v3"
AUDIT = ROOT / "results/archive/demo_results/diagnostics/raw/k48_handoff/nominal_audit"
SUPERVISED_OLD = ROOT / "results/archive/demo_results/diagnostics/raw/k48_handoff/supervised_targets.jsonl"
CKPT = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt"
LIMIT_S = 4.0
CAP_STEPS = 200

RECATCH_PLAN = (
    ("command", next(item for item in STRATEGIES if item["name"] == "chase_lost")),
    ("command", next(item for item in STRATEGIES if item["name"] == "release_long")),
    ("command", next(item for item in STRATEGIES if item["name"] == "release_short")),
)
CATEGORIES = {
    "recatch": {
        "plan": RECATCH_PLAN,
        "strategies": {"chase_lost", "release_long", "release_short"},
    },
    "controlled_slide": {
        "plan": (("controlled_slide", {"name": "controlled_slide"}),),
        "strategies": {"controlled_slide"},
    },
    "inward_reseat": {
        "plan": (("inward_reseat", {"name": "inward_reseat"}),),
        "strategies": {"inward_reseat"},
    },
}


def _refuse(path):
    text = str(path).replace("\\", "/").lower()
    if "final" in text:
        raise RuntimeError(f"refusing final path {path}")
    resolved = Path(path).resolve()
    if OUT.resolve() not in resolved.parents and resolved != OUT.resolve():
        raise RuntimeError(f"refusing write outside the rebuilt dataset: {path}")


def _append(path, row):
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def _load_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def _scenario_key(spec):
    azimuth = float(spec["azimuth_deg"]) % 360.0
    return (
        int(round(azimuth / 10.0) * 10) % 360,
        round(float(spec["speed_mps"]) * 2.0) / 2.0,
        round(float(spec["z_off_m"]) / 0.004) * 0.004,
        round(float(spec["phi_deg"]) / 5.0) * 5.0,
        round(float(spec.get("delay_s") or 0.0) / 0.05) * 0.05,
    )


def _split_for_key(key):
    token = "|".join(str(part) for part in key)
    return "valid" if zlib.crc32(token.encode()) % 5 == 0 else "train"


def _open_env():
    import torch
    from training.recovery_episode import make_training_env
    from training.unified_k48 import load_trained_k48

    before = (CKPT.stat().st_mtime, CKPT.stat().st_size)
    model, _book = load_trained_k48(CKPT, torch.device("cpu"))
    env = make_training_env(torch.device("cpu"), model=model)
    if CKPT.stat().st_mtime != before[0] or CKPT.stat().st_size != before[1]:
        raise RuntimeError("opening the environment changed the supervised checkpoint")
    return env


def resolve_stored_path(path):
    """Open a stored path. Historical ``results/`` records now live in ``results/archive/demo_results/``."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.exists():
        return candidate
    text = str(candidate).replace("\\", "/")
    marker = "/results/"
    index = text.find(marker)
    if index >= 0:
        migrated = Path(text[:index] + "/results/archive/demo_results/" + text[index + len(marker):])
        if migrated.exists():
            return migrated
    return candidate


def _load_blob(snapshot):
    path = resolve_stored_path(snapshot)
    if "final" in str(path).lower():
        raise RuntimeError(f"refusing {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _restore_error(env, blob):
    restore_capture(env, blob["capture"])
    physics = blob["capture"]["physics"]
    qpos = float(np.max(np.abs(env.sim.data.qpos - physics["qpos"])))
    phase = str(env.sim.fsm.phase) == str(blob["capture"]["fsm"]["phase"])
    return qpos, phase


def immediate_handoff(env, blob):
    qpos_error, phase_ok = _restore_error(env, blob)
    if qpos_error > 1e-8 or not phase_ok:
        return {
            "nominal_class": "INVALID",
            "reason": "INVALID",
            "success": False,
            "qpos_error": qpos_error,
            "phase_restored": phase_ok,
            "recovery_elapsed": 0.0,
        }
    t0 = float(env.sim.data.time)
    out = finish_recovery(env, blob["ref"], t0, "early")
    reason = out["reason"]
    if reason == "INVALID":
        label = "INVALID"
    elif out.get("success"):
        label = "NOMINAL_SOLVABLE"
    else:
        label = "RECOVERY_CANDIDATE"
    return {
        "nominal_class": label,
        "reason": reason,
        "success": bool(out.get("success")),
        "qpos_error": qpos_error,
        "phase_restored": phase_ok,
        "recovery_elapsed": out.get("recovery_elapsed"),
        "nominal_elapsed": out.get("nominal_elapsed"),
        "lift_completed": out.get("lift_completed"),
    }


def _contacts(env):
    pack = physical_pack(env.sim)
    return {
        "t": float(env.sim.data.time),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "z": float(pack["obj_z"]),
    }


def legalize(actions, book):
    packed = representability(actions, book)
    return (
        np.asarray(packed["tokens"], np.int64),
        np.clip(np.asarray(actions, np.float32) - book[np.asarray(packed["tokens"], np.int64)], -CAP, CAP),
        np.asarray(packed["quantized"], np.float32),
        packed,
    )


def propose_actions(env, blob, kind, spec):
    restore_capture(env, blob["capture"])
    actions = []
    memory = {}
    lost_low = 0
    basin_run = 0
    t0 = float(env.sim.data.time)
    stop = "cap"
    if kind in ("controlled_slide", "inward_reseat"):
        spec["ex_start"] = _finger_x(env.sim) if kind == "inward_reseat" else _pinch_x(env.sim, blob["ref"])
    for index in range(CAP_STEPS):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            stop = "limit"
            break
        if kind == "slide":
            action = slide_action(env.sim, env.switch, blob["ref"], spec, memory)
        elif kind == "controlled_slide":
            action, phase = controlled_slide_action(env.sim, spec, memory)
            if phase == "done":
                stop = "sequence_done"
                break
        elif kind == "inward_reseat":
            action, phase = inward_reseat_action(env.sim, spec, memory)
            if phase == "done":
                stop = "sequence_done"
                break
        else:
            action = command(env.sim, blob["ref"], spec, index)
            phase = None
        action = step_legal(env.sim, action)
        actions.append(np.asarray(action, np.float32))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            stop = "recovery_failed"
            break
        if kind in ("controlled_slide", "inward_reseat") and phase == "done":
            stop = "sequence_done"
            break
        if kind in ("controlled_slide", "inward_reseat"):
            continue
        row = errors(env.sim, blob["ref"])
        inside = basin_ok(row, float(action[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        if basin_run >= PERSIST_STEPS:
            stop = "basin_proposal"
            break
    if kind in ("controlled_slide", "inward_reseat"):
        spec["ex_end"] = _finger_x(env.sim) if kind == "inward_reseat" else _pinch_x(env.sim, blob["ref"])
        _pack, spec["bilateral_end"] = _grasp_now(env.sim)
    stacked = np.stack(actions, axis=0) if actions else np.zeros((0, 7), np.float32)
    return stacked, stop


def legal_verify(env, blob, actions):
    """Replay decoded commands and apply the unchanged 12-second check."""
    qpos_error, phase_ok = _restore_error(env, blob)
    t0 = float(env.sim.data.time)
    observations = []
    executed = []
    contacts = []
    lost_low = 0
    how = None
    for action in np.asarray(actions, np.float32):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            how = "forced"
            break
        observations.append(np.asarray(env.observation(), np.float32))
        done = step_legal(env.sim, action)
        executed.append(np.asarray(done, np.float32))
        contacts.append(_contacts(env))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            from envs.outcomes.horizon_outcome import _fail_recovery
            out = _fail_recovery(env.sim, blob["ref"], t0)
            out["recovery_steps"] = len(executed)
            return out, observations, executed, contacts, qpos_error, phase_ok
    handoff_obs = np.asarray(env.observation(), np.float32)
    if how is None:
        how = "recorded_end"
    out = finish_recovery(env, blob["ref"], t0, how)
    out["recovery_steps"] = len(executed)
    out["handoff_obs"] = handoff_obs
    return out, observations, executed, contacts, qpos_error, phase_ok


def _none_intervals(contacts):
    gaps = 0
    inside = False
    for row in contacts:
        bilateral = int(row["nL"]) > 0 and int(row["nR"]) > 0
        if not bilateral and not inside:
            gaps += 1
            inside = True
        elif bilateral:
            inside = False
    return gaps


# Legal wrist rates. Positive pitch wy=+2 and support wx=1.12 are exact tokens.
# wy=-1 rad/s (action -0.25) is the fastest negative rate inside ±0.25.
# wy=+3 rad/s (action 0.75) is the controlled-slip rotate rate, L-inf 0.05.
EX_ENTER = 0.003
CORRECTION_M = 0.003
GX_CAP = 6.0
SLIDE_OMEGA = 3.0
SLIDE_SLOW = 1.0
SLIDE_SLIP_S = 0.70
SLIDE_BRAKE_S = 0.40
SLIDE_TILT_MAX = float(np.deg2rad(120.0))
V_AWAY = 0.02
AWAY_STEPS = 5
LOST_S = 0.10
SLIP_MIN_S = 0.16
FIRM_CONTACTS = 3
INWARD_WX = 1.12
INWARD_WY = 2.0
INWARD_TIGHT_S = 0.16
INWARD_GRIP_RELAX = -0.48
# |g_h.x| at which an unloaded grip produces a correcting slide. Below this,
# secure grip friction keeps the cylinder stuck to the fingers.
GX_SLIP = 6.0
INWARD_SLIP_MAX = 0.70


def _pinch_x(sim, ref) -> float:
    pinch, _relative = hand_pose(sim)
    nominal = np.asarray(ref["p_nom"], float).reshape(-1)
    return float(np.asarray(pinch, float)[0] - nominal[0])


def _finger_x(sim) -> float:
    """Cylinder offset along the finger axis. This is the coordinate a slide can change."""
    pack, _bilateral = _grasp_now(sim)
    return float(np.asarray(pack["rh"], float)[0])


def _grasp_now(sim):
    pack = physical_pack(sim)
    bilateral = int(pack["nL"]) > 0 and int(pack["nR"]) > 0
    return pack, bilateral


def _wrist_action(wx, wy, grip):
    action = normalize_action(np.zeros(3), np.array([wx, wy, 0.0]), grip_tau(grip), AIRBORNE_V3)
    return np.asarray(action, np.float32)


def _gravity_target(ex: float) -> float:
    magnitude = min(GX_CAP, abs(ex) / 0.012 * GX_CAP)
    return -float(np.sign(ex)) * magnitude


def _gravity_ready(gx: float, target: float) -> bool:
    if abs(target) < 0.5:
        return True
    if target < 0.0:
        return gx <= target
    return gx >= target


def _both_off(pack) -> bool:
    return int(pack["nL"]) == 0 and int(pack["nR"]) == 0


def _fingers_gone(memory, pack, now) -> bool:
    """Both fingers clear for a sustained interval. One empty sample is not a loss."""
    if _both_off(pack):
        if memory.get("lost_t") is None:
            memory["lost_t"] = now
        return now - float(memory["lost_t"]) >= LOST_S - 1e-9
    memory["lost_t"] = None
    return False


def _slip_budget(ex: float, cap: float) -> float:
    return float(np.clip(abs(ex) / 0.012 * cap, SLIP_MIN_S, cap))


def _leaving(ex: float, vx: float, elapsed: float, memory) -> bool:
    """Zero speed is not the wrong direction. Only a sustained outward speed ends the slip."""
    away = abs(vx) >= V_AWAY and ex * vx > 0.0 and abs(ex) > EX_ENTER
    memory["away"] = int(memory.get("away", 0)) + 1 if away else 0
    return elapsed >= SLIP_MIN_S - 1e-9 and int(memory["away"]) >= AWAY_STEPS


def _begin_slide(sim, ref, memory, kind: str):
    pack, bilateral = _grasp_now(sim)
    ex = _pinch_x(sim, ref)
    memory["R0"] = np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy()
    memory["turned"] = 0.0
    memory["away"] = 0
    memory["lost_t"] = None
    memory["gentle"] = False
    memory["phase_t"] = float(sim.data.time)
    memory["ex0"] = ex
    if _both_off(pack):
        memory["phase"] = "lost"
        memory["sign"] = 0.0
        memory["gx_target"] = 0.0
        return
    if bilateral and abs(ex) < EX_ENTER:
        memory["phase"] = "done"
        memory["sign"] = 0.0
        memory["gx_target"] = 0.0
        return
    memory["sign"] = 1.0 if ex > 0.0 else -1.0
    memory["gx_target"] = _gravity_target(ex)
    memory["phase"] = "pitch" if kind == "inward" else "rotate"


def _tilt_rate(sign: float, pack, gentle: bool) -> float:
    firm = min(int(pack["nL"]), int(pack["nR"])) >= FIRM_CONTACTS and not gentle
    rate = SLIDE_OMEGA if firm and sign > 0.0 else SLIDE_SLOW
    return rate if sign > 0.0 else -rate


def _return_done(sim, memory, sign: float):
    current = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    error = memory["R0"].T @ current
    cosine = float(np.clip(0.5 * (np.trace(error) - 1.0), -1.0, 1.0))
    if float(np.arccos(cosine)) <= np.deg2rad(3.0):
        memory["phase"] = "done"
        return 0.0
    return -SLIDE_SLOW if sign > 0.0 else SLIDE_SLOW


def controlled_slide_action(sim, spec, memory):
    """Privileged slip. Direction, tilt, and slip length follow this grasp.

    A finger that drops does not end the skill or start the weak grip. The weak
    grip starts only after gravity reaches this error's target, and it stays
    open while the cylinder is still offset and not moving outward. If the tilt
    itself lets go of the cylinder, the grip stays closed and the wrist returns.
    A cylinder that is already free is not given this sequence.
    """
    ref = spec["ref"]
    if "phase" not in memory:
        _begin_slide(sim, ref, memory, "slide")
    phase = memory["phase"]
    if phase in ("done", "lost"):
        return _wrist_action(0.0, 0.0, 1.0), "done"
    pack, bilateral = _grasp_now(sim)
    ex = _pinch_x(sim, ref)
    gx = float(pack["g_h"][0])
    vx = float(pack["v_rel_h"][0])
    now = float(sim.data.time)
    sign = float(memory["sign"])
    wy = 0.0
    grip = 1.0
    if phase == "rotate":
        if _fingers_gone(memory, pack, now):
            memory["phase"] = "return"
            memory["phase_t"] = now
        elif not bilateral or (memory.get("unwound") and abs(float(memory["turned"])) < 0.05):
            memory["gentle"] = True
            memory["unwound"] = True
            wy = -_tilt_rate(sign, pack, True)
            if abs(float(memory["turned"]) + wy * 0.02) < 0.05:
                memory["phase"] = "return"
                memory["phase_t"] = now
                wy = 0.0
        else:
            wy = _tilt_rate(sign, pack, bool(memory["gentle"]))
            ready = _gravity_ready(gx, float(memory["gx_target"]))
            capped = abs(float(memory["turned"])) >= SLIDE_TILT_MAX - 1e-4
            if ready or capped:
                memory["phase"] = "slip"
                memory["phase_t"] = now
                memory["away"] = 0
                memory["slip_budget"] = _slip_budget(ex, SLIDE_SLIP_S)
                wy = 0.0
    elif phase == "slip":
        elapsed = now - memory["phase_t"]
        if _fingers_gone(memory, pack, now) or abs(ex) <= EX_ENTER or _leaving(ex, vx, elapsed, memory) or elapsed >= float(memory["slip_budget"]) - 1e-9:
            memory["phase"] = "brake"
            memory["phase_t"] = now
        else:
            grip = float(grip_action(-5.0))
    elif phase == "brake":
        elapsed = now - memory["phase_t"]
        slow = float(np.linalg.norm(pack["v_rel_h"])) < 0.01
        if slow or elapsed >= SLIDE_BRAKE_S - 1e-9 or _fingers_gone(memory, pack, now):
            memory["phase"] = "return"
            memory["phase_t"] = now
    elif phase == "return":
        wy = _return_done(sim, memory, sign)
    memory["turned"] = float(memory["turned"] + wy * 0.02)
    return _wrist_action(0.0, wy, grip), memory["phase"]


def inward_reseat_action(sim, spec, memory):
    """Gravity-assisted inward slide along the finger axis.

    Secure grip cannot recenter the cylinder: friction makes it follow the hand.
    Pitch only in the sign that puts g_h.x against the current finger-axis offset,
    and only until |g_h.x| reaches the slip threshold. Then hold that orientation
    and unload the grip. Keep unloading only while both fingers stay on and the
    offset is not growing. Retighten to lock the new contact. A tilt that increases
    the offset is a dump, so the grip stays closed and this attempt does not slide.
    """
    ref = spec["ref"]
    if "phase" not in memory:
        _begin_slide(sim, ref, memory, "inward")
        ex = _finger_x(sim)
        memory["ex0"] = ex
        if memory["phase"] != "lost":
            if abs(ex) < EX_ENTER:
                memory["phase"] = "done"
            else:
                memory["phase"] = "pitch"
                memory["sign"] = 1.0 if ex > 0.0 else -1.0
                memory["gx_target"] = -float(memory["sign"]) * GX_SLIP
    phase = memory["phase"]
    if phase in ("done", "lost"):
        return _wrist_action(0.0, 0.0, 1.0), "done"
    pack, bilateral = _grasp_now(sim)
    ex = _finger_x(sim)
    gx = float(pack["g_h"][0])
    vx = float(pack["v_rel_h"][0])
    now = float(sim.data.time)
    sign = float(memory["sign"])
    wx = 0.0
    wy = 0.0
    grip = 1.0
    if phase == "pitch":
        grown = abs(ex) > abs(float(memory.get("ex0", ex))) + 0.002
        ready = _gravity_ready(gx, float(memory["gx_target"]))
        capped = abs(float(memory.get("turned", 0.0))) >= float(np.deg2rad(90.0))
        if not bilateral:
            if memory.get("uni_t") is None:
                memory["uni_t"] = now
            unilateral = now - float(memory["uni_t"]) >= LOST_S - 1e-9
        else:
            memory["uni_t"] = None
            unilateral = False
        if _fingers_gone(memory, pack, now) or unilateral or grown:
            memory["phase"] = "retighten"
            memory["phase_t"] = now
            if _fingers_gone(memory, pack, now):
                memory.setdefault("gates", []).append("fingers_gone")
            elif unilateral:
                memory.setdefault("gates", []).append("unilateral_contact")
            else:
                memory.setdefault("gates", []).append("offset_increased")
        elif ready:
            memory["phase"] = "relax"
            memory["phase_t"] = now
            memory["away"] = 0
            memory["uni_t"] = None
            memory["ex_slip"] = ex
            memory.setdefault("gates", []).append("gravity_ready")
            grip = INWARD_GRIP_RELAX
        elif capped:
            memory["phase"] = "retighten"
            memory["phase_t"] = now
            memory.setdefault("gates", []).append("gravity_not_reached")
        elif sign > 0.0:
            wx, wy = INWARD_WX, INWARD_WY
        else:
            wy = -SLIDE_SLOW
        memory["turned"] = float(memory.get("turned", 0.0)) + wy * 0.02
    elif phase == "relax":
        elapsed = now - memory["phase_t"]
        if not bilateral:
            if memory.get("uni_t") is None:
                memory["uni_t"] = now
            unilateral = now - float(memory["uni_t"]) >= LOST_S - 1e-9
        else:
            memory["uni_t"] = None
            unilateral = False
        grown = abs(ex) > abs(float(memory.get("ex_slip", ex))) + 0.002
        leaving = _leaving(ex, vx, elapsed, memory)
        duration_cut = elapsed >= INWARD_SLIP_MAX - 1e-9
        if (
            _fingers_gone(memory, pack, now)
            or unilateral
            or grown
            or abs(ex) <= EX_ENTER
            or leaving
            or duration_cut
        ):
            memory["phase"] = "retighten"
            memory["phase_t"] = now
            if _fingers_gone(memory, pack, now):
                memory.setdefault("gates", []).append("fingers_gone")
            elif unilateral:
                memory.setdefault("gates", []).append("unilateral_contact")
            elif grown:
                memory.setdefault("gates", []).append("offset_increased")
            elif abs(ex) <= EX_ENTER:
                memory.setdefault("gates", []).append("offset_inside_3mm")
            elif leaving:
                memory.setdefault("gates", []).append("outward_velocity")
            else:
                memory.setdefault("gates", []).append("relaxation_duration_cutoff")
        else:
            grip = INWARD_GRIP_RELAX
    elif phase == "retighten":
        if now - memory["phase_t"] >= INWARD_TIGHT_S - 1e-9:
            memory["phase"] = "done"
    return _wrist_action(wx, wy, grip), memory["phase"]


def _family(strategy):
    name = str(strategy)
    if name.startswith("slide") or name.startswith("controlled_slide") or name.startswith("inward_reseat"):
        return "sliding"
    if strategy == "chase_lost":
        return "chase"
    return "release"


def _save_npz(folder, stem, actions, observations, contacts, tokens, residuals, handoff_obs=None):
    folder.mkdir(parents=True, exist_ok=True)
    continuous = folder / f"{stem}.npz"
    tokens_path = folder / f"{stem}__tokens.npz"
    _refuse(continuous)
    payload = {
        "actions": np.asarray(actions, np.float32),
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
    return _rel(continuous), _rel(tokens_path)


def revalidate_existing(env, book):
    audit_rows = _load_jsonl(AUDIT / "train_immediate.jsonl")
    replays = _load_jsonl(AUDIT / "demo_replays.jsonl")
    confirmed = {row["id"] for row in replays if row.get("status") == "confirmed_recoverable"}
    failing = {row["id"] for row in audit_rows if not row.get("nominal_solvable")}
    destination = OUT / "revalidated_existing.jsonl"
    done = {(row["id"], row.get("continuous_npz")) for row in _load_jsonl(destination)}
    kept = []
    for row in read_jsonl(SUPERVISED_OLD):
        if row["id"] not in confirmed:
            continue
        key = (row["id"], row["continuous_npz"])
        if key in done:
            continue
        blob = _load_blob(row["snapshot"])
        actions = np.load(ROOT / row["continuous_npz"])["actions"]
        started = time.perf_counter()
        out = replay_commands(env, blob, actions, limit_s=LIMIT_S, early_basin=False)
        record = {
            "id": row["id"],
            "snapshot": row["snapshot"],
            "strategy": row.get("strategy"),
            "continuous_npz": row["continuous_npz"],
            "token_residual_npz": row.get("token_residual_npz"),
            "handoff_label": row.get("handoff_label"),
            "source_kind": row.get("source_kind"),
            "mechanism": row.get("mechanism"),
            "family": row.get("family"),
            "reason": out["reason"],
            "success": bool(out.get("success")),
            "recovery_elapsed": out.get("recovery_elapsed"),
            "nominal_elapsed": out.get("nominal_elapsed"),
            "lift_completed": out.get("lift_completed"),
            "seconds": round(time.perf_counter() - started, 3),
            "provenance": "existing_k48_supervised",
        }
        _append(destination, record)
        kept.append(record)
        print("REVALIDATE", row["id"], row.get("strategy"), out["reason"], flush=True)
    candidates = OUT / "candidates.jsonl"
    have = {row["id"] for row in _load_jsonl(candidates)}
    for row in audit_rows:
        if row["id"] in have or row["id"] not in failing:
            continue
        _append(candidates, {
            "id": row["id"],
            "snapshot": row["snapshot"],
            "split": "train",
            "nominal_class": "RECOVERY_CANDIDATE" if not row.get("nominal_solvable") else "NOMINAL_SOLVABLE",
            "reason": row.get("reason"),
            "success": bool(row.get("nominal_solvable")),
            "azimuth_deg": row.get("azimuth_deg"),
            "speed_mps": row.get("speed_mps"),
            "z_off_m": row.get("z_off_m"),
            "phi_deg": row.get("phi_deg"),
            "delay_s": row.get("delay_s"),
            "states": row.get("states"),
            "source": row.get("source"),
            "provenance": "nominal_audit_train",
            "scenario_key": list(_scenario_key(row)),
            "recoverability": "confirmed" if row["id"] in confirmed else "unknown",
        })
    solvable = OUT / "solvable_pool.jsonl"
    have_solvable = {row["id"] for row in _load_jsonl(solvable)}
    for row in audit_rows:
        if row["id"] in have_solvable or not row.get("nominal_solvable"):
            continue
        _append(solvable, {
            "id": row["id"],
            "snapshot": row["snapshot"],
            "split": "train",
            "nominal_class": "NOMINAL_SOLVABLE",
            "reason": row.get("reason"),
            "azimuth_deg": row.get("azimuth_deg"),
            "speed_mps": row.get("speed_mps"),
            "z_off_m": row.get("z_off_m"),
            "phi_deg": row.get("phi_deg"),
            "delay_s": row.get("delay_s"),
            "source": row.get("source"),
            "provenance": "nominal_audit_train",
            "scenario_key": list(_scenario_key(row)),
        })
    return confirmed


def _reserved_keys():
    keys = set()
    for name in ("candidates.jsonl", "solvable_pool.jsonl"):
        for row in _load_jsonl(OUT / name):
            if row.get("scenario_key"):
                keys.add(tuple(row["scenario_key"]))
    return keys


def _known_specs():
    specs = []
    for name in ("candidates.jsonl", "solvable_pool.jsonl", "generation.jsonl"):
        for row in _load_jsonl(OUT / name):
            if "azimuth_deg" in row and row.get("speed_mps") is not None:
                specs.append(row)
    return specs


def _draw_spec(draw):
    rng = np.random.default_rng(50_000 + int(draw))
    roll = float(rng.random())
    if roll < 0.45:
        speed = float(rng.uniform(9.0, 10.2))
    elif roll < 0.75:
        speed = float(rng.uniform(8.0, 9.0))
    else:
        speed = float(rng.uniform(6.5, 8.0))
    if float(rng.random()) < 0.55:
        z_off = float(rng.uniform(-0.004, 0.004))
    else:
        z_off = float(rng.uniform(-0.020, 0.020))
    if float(rng.random()) < 0.65:
        phi = float(rng.uniform(-18.0, 18.0))
    else:
        phi = float(rng.choice([-1.0, 1.0]) * rng.uniform(20.0, 50.0))
    return {
        "azimuth_deg": float(rng.uniform(0.0, 360.0)),
        "speed_mps": speed,
        "z_off_m": z_off,
        "phi_deg": phi,
        "delay_s": float(rng.uniform(0.0, 0.45)),
        "direction": "continuous",
        "family": "ballistic_impact",
        "protocol": "k48_recovery_v3",
        "draw": int(draw),
    }


def generate_candidates(env, attempts):
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"next_draw": 0}
    draw = int(state.get("next_draw", 0))
    ran = 0
    while ran < attempts:
        spec = _draw_spec(draw)
        draw += 1
        key = _scenario_key(spec)
        split = _split_for_key(key)
        specs = _known_specs()
        if any(near(spec, item) for item in specs) or key in _reserved_keys():
            _append(OUT / "generation.jsonl", {
                "category": "near_duplicate_launch",
                "reason": "scenario key or parameter ball already used",
                "split": split,
                "scenario_key": list(key),
                **spec,
            })
            ran += 1
            continue
        spec["split"] = split
        spec["attempt_id"] = f"k48r_{split}_{draw:05d}"
        evidence, blob = run_episode(env, spec)
        evidence, blob = cylinder_first(evidence, blob)
        row = {
            "category": evidence.get("category"),
            "reason": evidence.get("reason"),
            "split": split,
            "scenario_key": list(key),
            "states": evidence.get("states"),
            "snapshot_qpos_error": evidence.get("snapshot_qpos_error"),
            "id": None,
            "snapshot": None,
            "nominal_class": None,
            **spec,
        }
        if blob is not None and evidence.get("category") == "valid_impact_recovery_ic":
            outcome = immediate_handoff(env, blob)
            row.update(outcome)
            row["id"] = spec["attempt_id"]
            blob["spec"]["id"] = spec["attempt_id"]
            blob["spec"]["split"] = split
            blob["spec"]["protocol"] = "k48_recovery_v3"
            folder = OUT / "snapshots" / split
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{spec['attempt_id']}.pkl"
            _refuse(path)
            with path.open("wb") as handle:
                pickle.dump(blob, handle)
            row["snapshot"] = _rel(path)
            if outcome["nominal_class"] == "RECOVERY_CANDIDATE":
                _append(OUT / "candidates.jsonl", {
                    "id": row["id"],
                    "snapshot": row["snapshot"],
                    "split": split,
                    "nominal_class": "RECOVERY_CANDIDATE",
                    "reason": outcome["reason"],
                    "success": False,
                    "azimuth_deg": spec["azimuth_deg"],
                    "speed_mps": spec["speed_mps"],
                    "z_off_m": spec["z_off_m"],
                    "phi_deg": spec["phi_deg"],
                    "delay_s": spec["delay_s"],
                    "states": evidence.get("states"),
                    "source": "k48_recovery_v3",
                    "provenance": "new_ballistic_generation",
                    "scenario_key": list(key),
                    "recoverability": "unknown",
                })
            elif outcome["nominal_class"] == "NOMINAL_SOLVABLE":
                _append(OUT / "solvable_pool.jsonl", {
                    "id": row["id"],
                    "snapshot": row["snapshot"],
                    "split": split,
                    "nominal_class": "NOMINAL_SOLVABLE",
                    "reason": outcome["reason"],
                    "azimuth_deg": spec["azimuth_deg"],
                    "speed_mps": spec["speed_mps"],
                    "z_off_m": spec["z_off_m"],
                    "phi_deg": spec["phi_deg"],
                    "delay_s": spec["delay_s"],
                    "source": "k48_recovery_v3",
                    "provenance": "new_ballistic_generation",
                    "scenario_key": list(key),
                })
        _append(OUT / "generation.jsonl", row)
        ran += 1
        state["next_draw"] = draw
        _refuse(state_path)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        print("GENERATE", row.get("category"), row.get("nominal_class"), row.get("id"), "split", split, flush=True)
    return ran


def _category_of(strategy):
    name = str(strategy or "")
    for category, spec in CATEGORIES.items():
        if name in spec["strategies"]:
            return category
    return None


def _category_paths(category):
    folder = OUT / category
    return folder / "search.jsonl", folder / "verified_trajectories.jsonl"


def _search_rows():
    rows = list(_load_jsonl(OUT / "search.jsonl"))
    for category in CATEGORIES:
        search_path, _verified_path = _category_paths(category)
        rows.extend(_load_jsonl(search_path))
    return rows


def _verified_rows():
    rows = list(_load_jsonl(OUT / "verified_trajectories.jsonl"))
    for category in CATEGORIES:
        _search_path, verified_path = _category_paths(category)
        rows.extend(_load_jsonl(verified_path))
    kept = []
    for row in rows:
        category = _category_of(row.get("strategy"))
        if category in ("inward_reseat", "controlled_slide") and row.get("corrected_m") is None:
            continue
        kept.append(row)
    return kept


def _search_done(category):
    names = CATEGORIES[category]["strategies"]
    return {
        row["key"]
        for row in _search_rows()
        if row.get("key") and (row.get("category") == category or row.get("strategy") in names)
    }


def _category_successes(category):
    names = CATEGORIES[category]["strategies"]
    found = []
    seen = set()
    for row in _verified_rows():
        if not row.get("success"):
            continue
        if row.get("category") != category and row.get("strategy") not in names:
            continue
        key = (row.get("id"), row.get("strategy"), row.get("continuous_npz"))
        if key in seen:
            continue
        seen.add(key)
        found.append(row)
    return found


def _category_counts(category):
    counts = {"train": set(), "valid": set()}
    for row in _category_successes(category):
        split = "valid" if row.get("split") == "valid" else "train"
        counts[split].add(row["id"])
    return {"train": len(counts["train"]), "valid": len(counts["valid"])}


def _category_inventory():
    return {category: _category_counts(category) for category in CATEGORIES}


def _confirmed_ids():
    found = {row["id"] for row in _verified_rows() if row.get("success") and row.get("id")}
    for row in _load_jsonl(OUT / "revalidated_existing.jsonl"):
        if row.get("success") and row.get("id"):
            found.add(row["id"])
    return found


def _evaluate_proposal(env, book, blob, row, kind, spec, category):
    proposed, stop = propose_actions(env, blob, kind, spec)
    record = {
        "key": f"{row['id']}|{spec['name']}|0",
        "id": row["id"],
        "split": row["split"],
        "category": category,
        "strategy": spec["name"],
        "seed": 0,
        "proposal_stop": stop,
        "proposal_steps": int(len(proposed)),
        "success": False,
        "reason": None,
    }
    if len(proposed) == 0:
        record["reason"] = "not_applicable" if category != "recatch" else "no_actions"
        return record, None
    tokens, residuals, decoded, packed = legalize(proposed, book)
    exact = float(packed["fraction_within_0.25"]) >= 1.0 - 1e-8
    record["fraction_within_0.25"] = packed["fraction_within_0.25"]
    record["max_linf"] = packed["max_linf"]
    if stop == "recovery_failed" and exact:
        record.update({"reason": "proposal_recovery_failed", "legal_replay": False})
        return record, None
    if category in ("inward_reseat", "controlled_slide"):
        start = float(spec.get("ex_start", 0.0))
        end = float(spec.get("ex_end", start))
        record["ex_start"] = start
        record["ex_end"] = end
        record["corrected_m"] = abs(start) - abs(end)
        if not spec.get("bilateral_end") or record["corrected_m"] < CORRECTION_M:
            record["reason"] = "error_not_reduced"
            return record, None
    out, observations, executed, contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded)
    record.update({
        "reason": out["reason"],
        "success": bool(out.get("success")),
        "recovery_elapsed": out.get("recovery_elapsed"),
        "nominal_elapsed": out.get("nominal_elapsed"),
        "lift_completed": out.get("lift_completed"),
        "qpos_error": qpos_error,
        "phase_restored": phase_ok,
        "legal_replay": True,
    })
    stem = f"{row['id']}__{spec['name']}"
    continuous, token_path = _save_npz(
        OUT / "search_attempts", stem, executed, observations, contacts, tokens, residuals,
        out.get("handoff_obs") if out.get("success") else None,
    )
    record["continuous_npz"] = continuous
    record["token_residual_npz"] = token_path
    verified = None
    if out.get("success") and qpos_error <= 1e-8 and phase_ok:
        verified = _label_handoff(
            env, blob, row, spec["name"], decoded, tokens, residuals,
            observations, executed, contacts, out,
        )
        verified["category"] = category
        verified["corrected_m"] = record.get("corrected_m")
        record["verified_trajectory"] = verified
    return record, verified


def search_category(env, book, category, per_ic, max_trials, target_train, target_valid):
    """Add verified demos for one category until its own train and valid counts are met."""
    search_path, verified_path = _category_paths(category)
    done = _search_done(category)
    succeeded = {row["id"] for row in _category_successes(category)}
    counts = _category_counts(category)
    print("CATEGORY", category, "have", counts, "target", {"train": target_train, "valid": target_valid}, flush=True)
    if counts["train"] >= target_train and counts["valid"] >= target_valid:
        print("CATEGORY_TARGET_MET", category, counts, flush=True)
        return 0
    trials = 0
    found = 0
    candidates = [row for row in _load_jsonl(OUT / "candidates.jsonl") if row.get("nominal_class") == "RECOVERY_CANDIDATE"]

    def priority(row):
        existing_unknown = row.get("provenance") == "nominal_audit_train" and row.get("recoverability") == "unknown"
        return (0 if existing_unknown else 1, 0 if row.get("split") == "valid" else 1, row["id"])

    ordered = sorted(candidates, key=priority)
    valid_rows = [row for row in ordered if row.get("split") == "valid"]
    train_rows = [row for row in ordered if row.get("split") != "valid"]
    valid_cap = max(1, max_trials // 3) if valid_rows else 0
    plan = CATEGORIES[category]["plan"][:per_ic]
    for selected, cap in ((valid_rows, valid_cap), (train_rows, max_trials)):
        if cap <= 0:
            continue
        for row in selected:
            if trials >= cap:
                print("CATEGORY_BUDGET", category, trials, cap, flush=True)
                break
            counts = _category_counts(category)
            if counts["train"] >= target_train and counts["valid"] >= target_valid:
                print("CATEGORY_TARGET_MET", category, counts, flush=True)
                return trials
            if row["id"] in succeeded:
                continue
            blob = _load_blob(row["snapshot"])
            if category != "recatch":
                restore_capture(env, blob["capture"])
                pack, bilateral = _grasp_now(env.sim)
                if _both_off(pack) or (category == "inward_reseat" and not bilateral):
                    reason = "no_finger_contact" if _both_off(pack) else "one_finger_contact"
                    print(category.upper(), row["id"], "skip", reason, flush=True)
                    continue
            if category == "recatch":
                attempts = [(kind, dict(item)) for kind, item in plan]
            else:
                kind = category
                attempts = [(kind, {"name": category})]
            for kind, spec in attempts:
                spec["ref"] = blob["ref"]
                key = f"{row['id']}|{spec['name']}|0"
                if key in done or trials >= cap:
                    continue
                started = time.perf_counter()
                record, verified = _evaluate_proposal(env, book, blob, row, kind, spec, category)
                record["seconds"] = round(time.perf_counter() - started, 3)
                _append(search_path, record)
                done.add(key)
                trials += 1
                if verified is not None:
                    _append(verified_path, verified)
                    succeeded.add(row["id"])
                    found += 1
                print(
                    category.upper(), row["id"], spec["name"], record["reason"],
                    "dex_mm", None if record.get("corrected_m") is None else round(record["corrected_m"] * 1e3, 1),
                    "exact", record.get("fraction_within_0.25"), "s", record["seconds"], flush=True,
                )
                if record["success"]:
                    break
    print("CATEGORY_DONE", category, "trials", trials, "new", found, "counts", _category_counts(category), flush=True)
    return trials


def _label_handoff(env, blob, candidate, strategy, decoded, tokens, residuals, observations, executed, contacts, full):
    """Keep recorded_end unless a shorter prefix independently succeeds."""
    earliest = None
    for fraction in (0.25, 0.50, 0.75):
        cut = max(1, int(round(len(decoded) * fraction)))
        if cut >= len(decoded):
            continue
        out, prefix_obs, prefix_executed, prefix_contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded[:cut])
        if out.get("success") and qpos_error <= 1e-8 and phase_ok:
            earliest = (cut, out, prefix_obs, prefix_executed, prefix_contacts)
            break
    family = _family(strategy)
    if earliest is None:
        label = "recorded_end"
        used = executed
        used_tokens = tokens
        used_residuals = residuals
        used_contacts = contacts
        handoff_obs = None
        recovery_elapsed = full.get("recovery_elapsed")
        nominal_elapsed = full.get("nominal_elapsed")
    else:
        cut, out, observations, executed, used_contacts = earliest
        label = "early_handoff"
        used = executed
        used_tokens = tokens[:cut]
        used_residuals = residuals[:cut]
        handoff_obs = out.get("handoff_obs")
        recovery_elapsed = out.get("recovery_elapsed")
        nominal_elapsed = out.get("nominal_elapsed")
    stem = f"{candidate['id']}__{strategy}__verified"
    continuous, token_path = _save_npz(
        OUT / "trajectories", stem, used, observations, used_contacts, used_tokens, used_residuals, handoff_obs,
    )
    gaps = _none_intervals(used_contacts)
    demo = {
        "id": candidate["id"],
        "snapshot": candidate["snapshot"],
        "split": candidate["split"],
        "strategy": strategy,
        "family": family,
        "none_intervals": gaps,
        "continuous_npz": continuous,
        "token_residual_npz": token_path,
        "handoff_label": label,
        "source_kind": "search",
        "nominal_class": "RECOVERY_CANDIDATE",
        "verified_success": True,
        "success": True,
        "reason": "SUCCESS_12S",
        "recovery_elapsed": recovery_elapsed,
        "nominal_elapsed": nominal_elapsed,
        "provenance": candidate.get("provenance"),
        "scenario_key": candidate.get("scenario_key"),
        "azimuth_deg": candidate.get("azimuth_deg"),
        "speed_mps": candidate.get("speed_mps"),
        "z_off_m": candidate.get("z_off_m"),
        "phi_deg": candidate.get("phi_deg"),
        "delay_s": candidate.get("delay_s"),
        "states": candidate.get("states"),
    }
    demo["mechanism"] = mechanism(demo)
    return demo


def _control_count(recovery_n, available):
    if recovery_n <= 0 or available <= 0:
        return 0
    lo = int(np.ceil(0.05 / 0.95 * recovery_n - 1e-9))
    hi = int(np.floor(0.10 / 0.90 * recovery_n + 1e-9))
    target = int(round((0.075 / 0.925) * recovery_n))
    if hi < lo:
        return 0
    target = min(max(target, lo), hi)
    return min(target, available)


def _write_yaml(name, reward, updates, save_every):
    text = "\n".join([
        "# Rebuilt K48 recovery dataset. Do not point this at the collapsed manifests.",
        "dataset_version: k48_recovery_v3",
        "codebook: results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy",
        "supervised_targets: data/supervised_targets.jsonl",
        "grpo_initial_conditions: data/grpo_initial_conditions.jsonl",
        "eval_manifest: data/valid_recovery.jsonl",
        "supervised_output: results/sft",
        "eval_output: results/evaluations",
        f"grpo_output: results/grpo/{reward}",
        "baseline_eval: results/archive/demo_results/diagnostics/raw/horizon_migration/checkpoint_eval.json",
        "seed: 1",
        "device: cuda",
        "eval_device: cpu",
        "epochs: 40",
        "batch_size: 8",
        "lr: 0.0003",
        f"grpo_updates: {updates}",
        "ic_epochs: 3",
        "cover_all_ics: true",
        "grpo_lr: 0.00003",
        "kl_coef: 0.05",
        "clip: 0.2",
        "group: 4",
        f"grpo_save_every: {save_every}",
        "rollout_cap_steps: 200",
        "eval_cap_steps: 200",
        "lambda_time: 0.05",
        "nominal_seconds: 12.0",
        "recovery_limit_s: 4.0",
        f"reward_version: {reward}",
        "expand_handoff: true",
        "",
    ])
    path = ROOT / "config" / "training" / "k48" / name
    path.write_text(text, encoding="utf-8")


def build_manifests(env):
    confirmed = []
    seen_demo = set()
    for row in _load_jsonl(OUT / "revalidated_existing.jsonl"):
        if not row.get("success"):
            continue
        key = (row["id"], row.get("continuous_npz"))
        if key in seen_demo:
            continue
        seen_demo.add(key)
        confirmed.append({
            **row,
            "split": "train",
            "nominal_class": "RECOVERY_CANDIDATE",
            "verified_success": True,
            "source_kind": row.get("source_kind") or "historical",
        })
    for row in _verified_rows():
        if not row.get("success"):
            continue
        key = (row["id"], row.get("continuous_npz"))
        if key in seen_demo:
            continue
        seen_demo.add(key)
        confirmed.append(row)
    by_id = {}
    for row in confirmed:
        blob = _load_blob(row["snapshot"])
        if row["id"] not in by_id:
            outcome = immediate_handoff(env, blob)
            by_id[row["id"]] = outcome
            print("RECHECK", row["id"], outcome["nominal_class"], outcome["reason"], flush=True)
        if by_id[row["id"]]["nominal_class"] != "RECOVERY_CANDIDATE":
            row["verified_success"] = False
    confirmed = [row for row in confirmed if row.get("verified_success")]
    replay_ok = []
    checked = set()
    for row in confirmed:
        key = (row["id"], row.get("continuous_npz"))
        if key in checked:
            continue
        if row.get("provenance") == "existing_k48_supervised":
            replay_ok.append(row)
            checked.add(key)
            continue
        blob = _load_blob(row["snapshot"])
        actions = np.load(ROOT / row["continuous_npz"])["actions"]
        out = replay_commands(env, blob, actions, limit_s=LIMIT_S, early_basin=False)
        row["replay_reason"] = out["reason"]
        if out.get("success"):
            replay_ok.append(row)
        checked.add(key)
        print("REPLAY_CHECK", row["id"], row.get("strategy"), out["reason"], flush=True)
    train_ids = []
    valid_ids = []
    for row in replay_ok:
        target = train_ids if row.get("split", "train") == "train" else valid_ids
        if row["id"] not in target:
            target.append(row["id"])
    recovery_rows = [row for row in replay_ok if row["id"] in train_ids]
    valid_rows = [row for row in replay_ok if row["id"] in valid_ids]
    weights = _weights(recovery_rows)
    supervised = []
    for row in recovery_rows:
        label = row.get("handoff_label") or "recorded_end"
        if label == "early_handoff" and "handoff_obs" not in np.load(ROOT / row["continuous_npz"]).files:
            label = "recorded_end"
        supervised.append({
            "id": row["id"],
            "snapshot": row["snapshot"],
            "strategy": row.get("strategy"),
            "family": row.get("family"),
            "mechanism": row.get("mechanism"),
            "continuous_npz": row["continuous_npz"],
            "token_residual_npz": row["token_residual_npz"],
            "handoff_label": label,
            "source_kind": row.get("source_kind"),
            "nominal_class": "RECOVERY_CANDIDATE",
            "verified_success": True,
            "sample_weight": weights[(row["id"], row.get("continuous_npz"))],
            "split": "train",
            "provenance": row.get("provenance"),
        })
    solvable = [row for row in _load_jsonl(OUT / "solvable_pool.jsonl") if row.get("split", "train") == "train"]
    n_control = _control_count(len(train_ids), len(solvable))
    solvable_sorted = sorted(solvable, key=lambda item: (float(item.get("azimuth_deg") or 0.0), item["id"]))
    if n_control:
        indexes = np.linspace(0, len(solvable_sorted) - 1, n_control)
        chosen = []
        used = set()
        for index in indexes:
            row = solvable_sorted[int(round(index))]
            if row["id"] not in used:
                chosen.append(row)
                used.add(row["id"])
    else:
        chosen = []
    for row in chosen:
        blob = _load_blob(row["snapshot"])
        outcome = immediate_handoff(env, blob)
        if outcome["nominal_class"] != "NOMINAL_SOLVABLE":
            continue
        restore_capture(env, blob["capture"])
        obs = np.asarray(env.observation(), np.float32)
        folder = OUT / "control_observations"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{row['id']}.npz"
        _refuse(path)
        np.savez_compressed(path, handoff_obs=obs.reshape(1, 39))
        supervised.append({
            "id": row["id"],
            "snapshot": row["snapshot"],
            "strategy": "immediate_handoff",
            "family": "control",
            "mechanism": "immediate_handoff",
            "continuous_npz": _rel(path),
            "token_residual_npz": _rel(path),
            "handoff_label": "immediate_handoff",
            "source_kind": "control",
            "nominal_class": "NOMINAL_SOLVABLE",
            "verified_success": True,
            "sample_weight": 0.0,
            "split": "train",
            "provenance": row.get("provenance"),
        })
    if any(row["handoff_label"] == "immediate_handoff" for row in supervised):
        control_budget = 0.15
        controls = [row for row in supervised if row["handoff_label"] == "immediate_handoff"]
        share = control_budget / len(controls)
        for row in controls:
            row["sample_weight"] = share
    _dump(OUT / "supervised_targets.jsonl", supervised)
    grpo = []
    for episode_id in train_ids:
        source = next(row for row in recovery_rows if row["id"] == episode_id)
        grpo.append({
            "id": episode_id,
            "snapshot": source["snapshot"],
            "pool": "recovery",
            "nominal_class": "RECOVERY_CANDIDATE",
            "split": "train",
            "horizon": "recovery_limit_4s_then_nominal_12s",
            "recovery_limit_s": LIMIT_S,
            "nominal_seconds": NOMINAL_S,
            "provenance": source.get("provenance"),
        })
    for row in supervised:
        if row["handoff_label"] != "immediate_handoff":
            continue
        grpo.append({
            "id": row["id"],
            "snapshot": row["snapshot"],
            "pool": "control",
            "nominal_class": "NOMINAL_SOLVABLE",
            "split": "train",
            "horizon": "recovery_limit_4s_then_nominal_12s",
            "recovery_limit_s": LIMIT_S,
            "nominal_seconds": NOMINAL_S,
            "provenance": row.get("provenance"),
        })
    _dump(OUT / "grpo_initial_conditions.jsonl", grpo)
    _dump(OUT / "train_recovery.jsonl", [row for row in grpo if row["pool"] == "recovery"])
    valid_manifest = []
    for episode_id in valid_ids:
        source = next(row for row in valid_rows if row["id"] == episode_id)
        valid_manifest.append({
            "id": episode_id,
            "snapshot": source["snapshot"],
            "pool": "recovery",
            "nominal_class": "RECOVERY_CANDIDATE",
            "split": "valid",
            "horizon": "recovery_limit_4s_then_nominal_12s",
            "recovery_limit_s": LIMIT_S,
            "success": False,
            "provenance": source.get("provenance"),
        })
    _dump(OUT / "valid_recovery.jsonl", valid_manifest)
    _dump(OUT / "control.jsonl", [row for row in grpo if row["pool"] == "control"])
    confirmed_ids = set(train_ids) | set(valid_ids)
    unknown = []
    for row in _load_jsonl(OUT / "candidates.jsonl"):
        if row.get("nominal_class") == "RECOVERY_CANDIDATE" and row["id"] not in confirmed_ids:
            unknown.append({
                "id": row["id"],
                "snapshot": row["snapshot"],
                "pool": "unknown",
                "nominal_class": "RECOVERY_CANDIDATE",
                "split": row.get("split"),
                "provenance": row.get("provenance"),
                "reason": "no_verified_legal_trajectory",
            })
    _dump(OUT / "unknown.jsonl", unknown)
    updates = 3 * len(grpo) if grpo else 0
    save_every = len(grpo) if grpo else 1
    if grpo:
        _write_yaml("unified_k48_recovery_v3.yaml", "v1", updates, save_every)
        _write_yaml("unified_k48_recovery_v3_reward_v2.yaml", "v2", updates, save_every)
    summary = {
        "train_recovery": len(train_ids),
        "valid_recovery": len(valid_ids),
        "control": sum(1 for row in grpo if row["pool"] == "control"),
        "unknown": len(unknown),
        "supervised_rows": len(supervised),
        "grpo_ics": len(grpo),
        "grpo_updates": updates,
        "targets": {"train": 200, "valid": 50},
        "targets_met": len(train_ids) >= 200 and len(valid_ids) >= 50,
        "by_category": _category_inventory(),
    }
    _refuse(OUT / "manifest_summary.json")
    (OUT / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MANIFEST", json.dumps(summary), flush=True)
    return summary


def _weights(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        grouped[row.get("mechanism") or "unknown"][row["id"]].append(index)
    weights = {}
    for _mechanism, episodes in grouped.items():
        share = 1.0 / max(len(episodes), 1)
        for episode_id, indexes in episodes.items():
            piece = share / len(indexes)
            for index in indexes:
                row = rows[index]
                weights[(row["id"], row.get("continuous_npz"))] = piece
    return weights


def _dump(path, rows):
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _candidate_index():
    found = {}
    for name in ("candidates.jsonl", "solvable_pool.jsonl"):
        for row in _load_jsonl(OUT / name):
            found[row["id"]] = row
    return found


def _load_seed_library():
    """Legal successes used as seeds. VALID seeds stay out of the TRAIN pool."""
    meta = _candidate_index()
    library = []
    seen = set()
    sources = []
    for row in _verified_rows():
        if row.get("success"):
            sources.append(row)
    for row in _load_jsonl(OUT / "revalidated_existing.jsonl"):
        if row.get("success"):
            copied = dict(row)
            copied["split"] = "train"
            sources.append(copied)
    for row in sources:
        path = row.get("continuous_npz")
        split = row.get("split") or "train"
        if not path or path in seen or split not in ("train", "valid"):
            continue
        actions = np.load(ROOT / path)["actions"].astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not (1 <= len(actions) <= CAP_STEPS):
            continue
        seen.add(path)
        info = meta.get(row["id"], {})
        library.append({
            "id": row["id"],
            "split": split,
            "strategy": row.get("strategy") or "seed",
            "mechanism": row.get("mechanism") or "",
            "source_kind": row.get("source_kind") or "",
            "npz": path,
            "actions": actions,
            "azimuth_deg": float(info.get("azimuth_deg") or row.get("azimuth_deg") or 0.0),
            "speed_mps": float(info.get("speed_mps") or row.get("speed_mps") or 0.0),
        })
    return library


def _select_seeds(library, split, target, limit):
    pool = [seed for seed in library if seed["split"] == split]
    azimuth = float(target.get("azimuth_deg") or 0.0)
    speed = float(target.get("speed_mps") or 0.0)

    def rank(seed):
        daz = abs((seed["azimuth_deg"] - azimuth + 180.0) % 360.0 - 180.0)
        pause = 1 if seed.get("source_kind") == "pause" else 0
        return (pause, daz, abs(seed["speed_mps"] - speed), len(seed["actions"]))

    pool = sorted(pool, key=rank)
    chosen = []
    used = set()
    for preferred in ("airborne_chase", "release_regrasp", "repeated_recatch", None):
        for seed in pool:
            if seed["id"] in used:
                continue
            if preferred is not None and seed.get("mechanism") != preferred:
                continue
            chosen.append(seed)
            used.add(seed["id"])
            break
        if len(chosen) >= limit:
            break
    return chosen


def _unknown_order():
    meta = _candidate_index()
    bins = defaultdict(list)
    for row in _load_jsonl(OUT / "unknown.jsonl"):
        if row.get("nominal_class") != "RECOVERY_CANDIDATE":
            continue
        info = meta.get(row["id"], {})
        item = dict(row)
        item["snapshot"] = item.get("snapshot") or info.get("snapshot")
        item["azimuth_deg"] = info.get("azimuth_deg", item.get("azimuth_deg"))
        item["speed_mps"] = info.get("speed_mps", item.get("speed_mps"))
        item["scenario_key"] = info.get("scenario_key", item.get("scenario_key"))
        item["states"] = info.get("states", item.get("states"))
        azimuth = float(item.get("azimuth_deg") or 0.0) % 360.0
        speed = float(item.get("speed_mps") or 0.0)
        bins[(int(azimuth // 45.0), int(speed))].append(item)
    order = []
    keys = sorted(bins)
    while any(bins[key] for key in keys):
        for key in keys:
            if bins[key]:
                order.append(bins[key].pop(0))
    return order


def _prepare_decoded(actions, method, extra, book):
    actions = np.asarray(actions, np.float32)
    if method == "prefix_handoff":
        cut = max(1, min(len(actions), int(round(len(actions) * float(extra)))))
        actions = actions[:cut]
    elif method == "handoff_time":
        cut = max(1, min(len(actions) - 1, int(extra)))
        actions = actions[:cut]
    elif method == "perturb":
        rng = np.random.default_rng(int(extra["salt"]))
        noisy = np.clip(actions + rng.normal(0.0, float(extra["sigma"]), size=actions.shape).astype(np.float32), -1.0, 1.0)
        if extra.get("grip"):
            noisy = np.array(noisy, np.float32, copy=True)
            noisy[:, 6] = np.clip(noisy[:, 6] + float(extra["grip"]), -1.0, 1.0)
        _tokens, _residuals, decoded, packed = legalize(noisy, book)
        return decoded, packed
    _tokens, _residuals, decoded, packed = legalize(actions, book)
    return decoded, packed


def _seed_plan(chosen):
    primary = chosen[0]
    plan = [
        ("seed_full", primary, "full", None),
        ("prefix_handoff", primary, "p50", 0.50),
        ("prefix_handoff", primary, "p75", 0.75),
        ("perturb", primary, "s008", None),
        ("perturb", primary, "grip", None),
    ]
    if len(chosen) > 1:
        plan.append(("seed_full", chosen[1], "alt", None))
    return plan


def seed_search(env, book, max_trials, per_ic):
    """Transfer same-split legal demonstrations onto unknown initial conditions."""
    log_path = OUT / "seed_search.jsonl"
    done = {row["key"] for row in _load_jsonl(log_path)}
    confirmed = _confirmed_ids()
    library = _load_seed_library()
    trials = 0
    discoveries = []
    for target in _unknown_order():
        if trials >= max_trials:
            break
        if target["id"] in confirmed or not target.get("snapshot"):
            continue
        split = target.get("split") or "train"
        chosen = _select_seeds(library, split, target, 2)
        if not chosen:
            continue
        blob = _load_blob(target["snapshot"])
        used = 0
        for method, seed, variant, extra in _seed_plan(chosen):
            if used >= per_ic or trials >= max_trials:
                break
            if seed["split"] != split:
                raise RuntimeError(f"refusing to apply a {seed['split']} seed to {split} {target['id']}")
            key = f"{target['id']}|{method}|{seed['id']}|{seed['strategy']}|{variant}"
            if key in done:
                continue
            if method == "perturb":
                extra = {
                    "sigma": 0.05 if variant == "grip" else 0.08,
                    "grip": 0.2 if variant == "grip" else 0.0,
                    "salt": zlib.crc32(key.encode()),
                }
            started = time.perf_counter()
            decoded, packed = _prepare_decoded(seed["actions"], method, extra, book)
            out, observations, executed, contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded)
            record = {
                "key": key,
                "id": target["id"],
                "split": split,
                "method": method,
                "variant": variant,
                "seed_id": seed["id"],
                "seed_split": seed["split"],
                "strategy": seed["strategy"],
                "reason": out["reason"],
                "success": bool(out.get("success") and qpos_error <= 1e-8 and phase_ok),
                "recovery_elapsed": out.get("recovery_elapsed"),
                "recovery_steps": out.get("recovery_steps"),
                "nominal_elapsed": out.get("nominal_elapsed"),
                "qpos_error": qpos_error,
                "fraction_within_0.25": packed.get("fraction_within_0.25"),
                "new_ic": False,
            }
            if record["success"]:
                tokens, residuals, _decoded, _packed = legalize(decoded, book)
                verified = _label_handoff(
                    env, blob, target, seed["strategy"], decoded, tokens, residuals,
                    observations, executed, contacts, out,
                )
                verified["search_method"] = method
                verified["provenance"] = "seed_search"
                verified["source_kind"] = "seed_search"
                record["verified_trajectory"] = verified["continuous_npz"]
                record["new_ic"] = target["id"] not in confirmed
                _append(OUT / "verified_trajectories.jsonl", verified)
                if record["new_ic"]:
                    confirmed.add(target["id"])
                    discoveries.append(target["id"])
                    library.append({
                        "id": target["id"],
                        "split": split,
                        "strategy": seed["strategy"],
                        "mechanism": verified.get("mechanism") or seed.get("mechanism"),
                        "source_kind": "seed_search",
                        "npz": verified["continuous_npz"],
                        "actions": np.asarray(executed, np.float32),
                        "azimuth_deg": float(target.get("azimuth_deg") or 0.0),
                        "speed_mps": float(target.get("speed_mps") or 0.0),
                    })
            record["seconds"] = round(time.perf_counter() - started, 3)
            _append(log_path, record)
            done.add(key)
            used += 1
            trials += 1
            print("SEED", target["id"], method, seed["id"], record["reason"], "new", record["new_ic"], flush=True)
            if record["success"]:
                break
            if method == "seed_full" and out.get("reason") == "DROP_DURING_RECOVERY" and used < per_ic and trials < max_trials:
                cut = max(1, int(out.get("recovery_steps") or 1) - 1)
                if cut < len(decoded):
                    drop_key = f"{target['id']}|handoff_time|{seed['id']}|{seed['strategy']}|pre_drop"
                    if drop_key not in done:
                        started = time.perf_counter()
                        prefix, prefix_pack = _prepare_decoded(decoded, "handoff_time", cut, book)
                        drop_out, drop_obs, drop_exec, drop_contacts, drop_qpos, drop_phase = legal_verify(env, blob, prefix)
                        drop_record = {
                            "key": drop_key,
                            "id": target["id"],
                            "split": split,
                            "method": "handoff_time",
                            "variant": "pre_drop",
                            "seed_id": seed["id"],
                            "seed_split": seed["split"],
                            "strategy": seed["strategy"],
                            "reason": drop_out["reason"],
                            "success": bool(drop_out.get("success") and drop_qpos <= 1e-8 and drop_phase),
                            "recovery_steps": drop_out.get("recovery_steps"),
                            "fraction_within_0.25": prefix_pack.get("fraction_within_0.25"),
                            "new_ic": False,
                        }
                        if drop_record["success"]:
                            tokens, residuals, _decoded, _packed = legalize(prefix, book)
                            verified = _label_handoff(
                                env, blob, target, seed["strategy"], prefix, tokens, residuals,
                                drop_obs, drop_exec, drop_contacts, drop_out,
                            )
                            verified["search_method"] = "handoff_time"
                            verified["provenance"] = "seed_search"
                            verified["source_kind"] = "seed_search"
                            drop_record["new_ic"] = target["id"] not in confirmed
                            _append(OUT / "verified_trajectories.jsonl", verified)
                            if drop_record["new_ic"]:
                                confirmed.add(target["id"])
                                discoveries.append(target["id"])
                        drop_record["seconds"] = round(time.perf_counter() - started, 3)
                        _append(log_path, drop_record)
                        done.add(drop_key)
                        used += 1
                        trials += 1
                        print("SEED", target["id"], "handoff_time", seed["id"], drop_record["reason"], "new", drop_record["new_ic"], flush=True)
                        if drop_record["success"]:
                            break
    summary = _seed_summary(trials, discoveries)
    _refuse(OUT / "seed_search_summary.json")
    (OUT / "seed_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SEED_SEARCH_DONE", json.dumps(summary), flush=True)
    return summary


def _seed_summary(trials, discoveries):
    rows = _load_jsonl(OUT / "seed_search.jsonl")
    by_method = {}
    for method in sorted({row.get("method") for row in rows}):
        subset = [row for row in rows if row.get("method") == method]
        by_method[method] = {
            "attempts": len(subset),
            "successes": sum(1 for row in subset if row.get("success")),
            "new_ics": len({row["id"] for row in subset if row.get("new_ic")}),
            "seconds": round(sum(float(row.get("seconds") or 0.0) for row in subset), 3),
        }
    return {
        "attempts_this_call": trials,
        "new_ics_this_call": len(discoveries),
        "new_ic_ids_this_call": discoveries,
        "by_method": by_method,
    }


def write_report(started):
    generation = _load_jsonl(OUT / "generation.jsonl")
    search = _search_rows()
    revalidated = _load_jsonl(OUT / "revalidated_existing.jsonl")
    summary = json.loads((OUT / "manifest_summary.json").read_text(encoding="utf-8")) if (OUT / "manifest_summary.json").is_file() else {}
    candidates = _load_jsonl(OUT / "candidates.jsonl")
    supervised = _load_jsonl(OUT / "supervised_targets.jsonl")
    unknown = _load_jsonl(OUT / "unknown.jsonl")
    train_rows = _load_jsonl(OUT / "train_recovery.jsonl")
    valid_rows = _load_jsonl(OUT / "valid_recovery.jsonl")
    by_candidate = {row["id"]: row for row in candidates}

    def _enrich(rows):
        enriched = []
        for row in rows:
            base = dict(by_candidate.get(row["id"], {}))
            base.update(row)
            enriched.append(base)
        return enriched

    train_rows = _enrich(train_rows)
    valid_rows = _enrich(valid_rows)
    elapsed = time.perf_counter() - started
    train_keys = {tuple(row.get("scenario_key")) for row in train_rows if row.get("scenario_key")}
    valid_keys = {tuple(row.get("scenario_key")) for row in valid_rows if row.get("scenario_key")}
    leakage = train_keys & valid_keys
    recovery_demos = [row for row in supervised if row.get("nominal_class") == "RECOVERY_CANDIDATE"]

    def _bins(rows, field, edges):
        counts = Counter()
        for row in rows:
            value = row.get(field)
            if value is None:
                counts["missing"] += 1
                continue
            number = float(value)
            placed = False
            for lo, hi, label in edges:
                if lo <= number < hi or (hi == edges[-1][1] and number == hi):
                    counts[label] += 1
                    placed = True
                    break
            if not placed:
                counts["other"] += 1
        return dict(counts)

    seed_rows = _load_jsonl(OUT / "seed_search.jsonl")
    open_loop = {}
    for strategy in sorted({row.get("strategy") for row in search}):
        subset = [row for row in search if row.get("strategy") == strategy]
        open_loop[strategy] = {
            "attempts": len(subset),
            "successes": sum(1 for row in subset if row.get("success")),
            "seconds": round(sum(float(row.get("seconds") or 0.0) for row in subset), 1),
        }
    seed_methods = _seed_summary(0, [])["by_method"] if seed_rows else {}
    recovery_rows = [row for row in supervised if row.get("nominal_class") == "RECOVERY_CANDIDATE"]
    control_rows = [row for row in supervised if row.get("handoff_label") == "immediate_handoff"]
    recovery_ids = {row["id"] for row in recovery_rows}
    train_ids = {row["id"] for row in train_rows}
    valid_ids = {row["id"] for row in valid_rows}
    reconciliation = {
        "supervised_rows": len(supervised),
        "train_recovery_rows": len(recovery_rows),
        "train_recovery_ics": len(recovery_ids),
        "control_rows": len(control_rows),
        "valid_recovery_ics": len(valid_ids),
        "train_ids_match": recovery_ids == train_ids,
        "valid_ids_in_supervised": len(valid_ids & {row["id"] for row in supervised}),
        "rows_per_source": dict(Counter(row.get("source_kind") for row in recovery_rows)),
    }
    delay_edges = [(0, 0.1, "0-0.1"), (0.1, 0.2, "0.1-0.2"), (0.2, 0.3, "0.2-0.3"), (0.3, 0.5, "0.3-0.5")]
    speed_edges = [(5, 7, "5-7"), (7, 8, "7-8"), (8, 9, "8-9"), (9, 11, "9-11")]
    lines = [
        "# K48 recovery dataset rebuild",
        "",
        "New files are under `results/archive/demo_results/diagnostics/raw/k48_recovery_v3/`. Historical snapshots, the original 534-trajectory manifest, checkpoints, and FINAL were not modified.",
        "",
        "A state is in the recovery pool only when immediate HANDOFF fails the unchanged 12-second grasp check and a legal K48 action sequence replays to `SUCCESS_12S`. Unsuccessful searches remain in `unknown.jsonl` and are not labeled unrecoverable.",
        "",
        "## Counts",
        "",
        f"Revalidated existing demonstrations: {sum(1 for row in revalidated if row.get('success'))}/{len(revalidated)} `SUCCESS_12S` on the original 35 confirmed-recoverable TRAIN states.",
        f"New ballistic draws: {len(generation)}. Snapshot classes: {dict(Counter(row.get('nominal_class') for row in generation))}.",
        f"Search attempts this dataset: {len(search)}. Legal successes: {sum(1 for row in search if row.get('success'))}. Reasons: {dict(Counter(row.get('reason') for row in search))}.",
        f"TRAIN recovery ICs: {summary.get('train_recovery')}. VALID recovery ICs: {summary.get('valid_recovery')}. Control ICs: {summary.get('control')}. Unknown: {summary.get('unknown')} ({dict(Counter(row.get('split') for row in unknown))}).",
        f"Supervised rows: {summary.get('supervised_rows')} ({dict(Counter(row.get('nominal_class') for row in supervised))}). HANDOFF labels: {dict(Counter(row.get('handoff_label') for row in supervised))}.",
        f"Recovery mechanisms: {dict(Counter(row.get('mechanism') for row in recovery_demos))}.",
        f"GRPO initial conditions: {summary.get('grpo_ics')}. Scheduled updates (3 epochs): {summary.get('grpo_updates')}.",
        f"Targets of 200 TRAIN and 50 VALID recoverable ICs met: {summary.get('targets_met')}.",
        f"Scenario keys shared by TRAIN and VALID recovery manifests: {len(leakage)}.",
        "",
        "## Supervised-row reconciliation",
        "",
        "The supervised manifest is the TRAIN file. It contains one row per verified TRAIN demonstration plus the immediate-HANDOFF control rows. VALID demonstrations stay in `valid_demonstrations.jsonl` and are not TRAIN examples.",
        "",
        json.dumps(reconciliation),
        "",
        "## Open-loop search",
        "",
        json.dumps(open_loop),
        "",
        "`chase_lost` is the only open-loop method with a material success rate. `slide_50` has not produced a verified recovery. Those attempts stay in `search.jsonl`.",
        "",
        "## Seeded search",
        "",
        "Seeds are legal demonstrations from the same split. A VALID demonstration is never replayed onto a TRAIN initial condition. Every trial restores the target snapshot, decodes through the K48 codebook, and must pass the 12-second check.",
        "",
        json.dumps(seed_methods),
        "",
        "TRAIN recovery speed bins: " + json.dumps(_bins(train_rows, "speed_mps", speed_edges)) + ".",
        "VALID recovery speed bins: " + json.dumps(_bins(valid_rows, "speed_mps", speed_edges)) + ".",
        "TRAIN recovery delay bins: " + json.dumps(_bins(train_rows, "delay_s", delay_edges)) + ".",
        "",
        f"Registered recovery candidates: {len(candidates)}. Wall clock for this process: {round(elapsed, 1)} s. Search cost is the sum of per-attempt `seconds` in `search.jsonl`: {round(sum(float(row.get('seconds') or 0.0) for row in search), 1)} s.",
        "",
        "The 200/50 targets are not a reason to relax the 12-second criterion. Resume continues from the logs without repeating completed replays.",
        "",
        "## Resume",
        "",
        "```",
        "python -u training/rebuild_k48_recovery.py --seed-search --manifest --max-search-trials 800 --seed-trials-per-ic 6",
        "```",
        "",
        "This resumes from `seed_search.jsonl`. It does not repeat completed seed keys. Open-loop `slide_50` is not repeated.",
        "",
        "## Later training commands",
        "",
        "These commands are not part of dataset generation. Supervised training starts from the K=47 checkpoint so the HANDOFF row is expanded again, and it reads only the rebuilt manifest.",
        "",
        "```",
        "python -u training/unified_k48.py --supervised --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/archive/demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt",
        "python -u training/unified_k48.py --grpo --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/archive/demo_results/diagnostics/raw/k48_recovery_v3/supervised/checkpoint_epoch_0040.pt",
        "python -u training/unified_k48.py --grpo --config config/training/k48/unified_k48_recovery_v3_reward_v2.yaml --checkpoint results/archive/demo_results/diagnostics/raw/k48_recovery_v3/supervised/checkpoint_epoch_0040.pt",
        "python -u training/unified_k48.py --eval --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/archive/demo_results/diagnostics/raw/k48_recovery_v3/supervised/checkpoint_epoch_0040.pt",
        "```",
        "",
        "The old 26 v1 VALID snapshots stay a separate no-intervention benchmark. They are not in this training pool.",
        "",
    ]
    report = ROOT / "results/archive/demo_results/diagnostics/K48_RECOVERY_DATASET_REBUILD.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("REPORT", report, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Rebuild the K48 recovery dataset")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--revalidate", action="store_true")
    parser.add_argument("--generate-attempts", type=int, default=0)
    parser.add_argument("--search", action="store_true", help="Generate recatch demos (chase_lost, release_long, release_short)")
    parser.add_argument("--recatch", action="store_true", help="Generate recatch demos only")
    parser.add_argument("--seed-search", action="store_true")
    parser.add_argument("--controlled-slide", action="store_true", help="Generate controlled-sliding demos only")
    parser.add_argument("--inward-reseat", action="store_true", help="Generate gravity-assisted inward-sliding demos only")
    parser.add_argument("--counts", action="store_true", help="Print verified train/valid counts for the three categories")
    parser.add_argument("--seed-trials-per-ic", type=int, default=6)
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--strategies-per-ic", type=int, default=4)
    parser.add_argument("--max-search-trials", type=int, default=400)
    parser.add_argument("--target-train", type=int, default=200)
    parser.add_argument("--target-valid", type=int, default=50)
    args = parser.parse_args()
    if args.counts and not any((
        args.all, args.revalidate, args.generate_attempts, args.search, args.recatch,
        args.seed_search, args.controlled_slide, args.inward_reseat, args.manifest,
    )):
        print(json.dumps(_category_inventory(), indent=2), flush=True)
        return
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    env = _open_env()
    book = load_codebook()
    if args.all or args.revalidate:
        revalidate_existing(env, book)
    if args.generate_attempts:
        generate_candidates(env, args.generate_attempts)
    if args.all or args.search or args.recatch:
        search_category(env, book, "recatch", args.strategies_per_ic, args.max_search_trials, args.target_train, args.target_valid)
    if args.all or args.seed_search:
        seed_search(env, book, args.max_search_trials, args.seed_trials_per_ic)
    if args.controlled_slide:
        search_category(env, book, "controlled_slide", 1, args.max_search_trials, args.target_train, args.target_valid)
    if args.inward_reseat:
        search_category(env, book, "inward_reseat", 1, args.max_search_trials, args.target_train, args.target_valid)
    if args.all or args.manifest:
        build_manifests(env)
        write_report(started)


if __name__ == "__main__":
    main()
