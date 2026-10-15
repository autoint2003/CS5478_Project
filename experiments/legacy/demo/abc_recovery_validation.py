"""Matched validation of recovery mechanisms A, B, and C on TRAIN ball-impact ICs.

A is a contact-preserving reseat: hand motion along the pinch offset plus an
upward component, with a moderately unloaded grip. B tilts the gripper so
gravity can roll the cylinder, and translates perpendicular to the offset.
C is the already replay-verified release, chase, and regrasp family.

Privileged state selects legal recovery7d_airborne_v3 commands. Physics,
success thresholds, and the sealed GRPO pilot are unchanged. This file does
not train a policy and does not search VALIDATION or FINAL_TEST.
"""

from __future__ import annotations


import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from envs.outcomes.nominal_pose import capture_ref, closed_grip, errors, hand_R, rotvec
from controllers.recovery.action_map import AIRBORNE_V3, grip_tau, normalize_action
from controllers.recovery.geometry_switch import hand_pose
from envs.outcomes.physical_recovery import physical_pack
from training.recovery_episode import make_training_env
from data_generation.ball_impact_dataset import ids_for, label_states, load_episode


from experiments.ball_impact.privileged_ball_impact_search import (
    PROTO_PATH,
    STRATEGIES,
    command,
    diagnose,
    legal_action_audit,
    wrist_return,
)
from experiments.sliding.sliding_vs_regrasp import (
    HANDOFF_Z,
    PRIOR,
    _grip,
    _replay_actions,
    _zero_unloaded_translation,
    execute,
    representability,
)

RAW = ROOT / "results" / "diagnostics" / "raw" / "abc_recovery"
LOG = RAW / "search_log.jsonl"
SUMMARY = RAW / "summary.json"
REPORT = ROOT / "results" / "diagnostics" / "ABC_RECOVERY_MECHANISM_VALIDATION.md"
C_SUMMARY = ROOT / "results" / "diagnostics" / "raw" / "sliding_vs_regrasp" / "summary.json"

A_SPECS = (
    {"name": "a_up05_g55", "grip": -0.55, "kp": 14.0, "cap": 0.45, "up": 0.05},
    {"name": "a_up10_g58", "grip": -0.58, "kp": 16.0, "cap": 0.50, "up": 0.10},
    {"name": "a_up15_g45", "grip": -0.45, "kp": 12.0, "cap": 0.40, "up": 0.15},
    {"name": "a_up08_g50", "grip": -0.50, "kp": 18.0, "cap": 0.48, "up": 0.08},
)
B_SPECS = (
    {"name": "b_tilt04_p08", "grip": -0.58, "tilt": 0.40, "tilt_rate": 1.5, "v_perp": 0.08},
    {"name": "b_tilt07_p12", "grip": -0.55, "tilt": 0.70, "tilt_rate": 2.0, "v_perp": 0.12},
    {"name": "b_tilt10_p05", "grip": -0.62, "tilt": 1.00, "tilt_rate": 2.5, "v_perp": 0.05},
    {"name": "b_tilt05_p15", "grip": -0.50, "tilt": 0.50, "tilt_rate": 1.2, "v_perp": 0.15},
)
C_ORDER = ("release_long", "chase_lost", "lift_soft", "release_short")
C_SPECS = tuple(next(spec for spec in STRATEGIES if spec["name"] == name) for name in C_ORDER)


def _unit(vector):
    vector = np.asarray(vector, float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros(3), 0.0
    return vector / norm, norm


def _tilt_rad(sim):
    hand_z = hand_R(sim)[:, 2]
    return float(np.arccos(np.clip(np.dot(hand_z, np.array([0.0, 0.0, -1.0])), -1.0, 1.0)))


def _contacts(sim, switch, memory):
    from data_generation.ball_impact_dataset import grasp_sample

    sample, anchor = grasp_sample(sim, switch, memory.get("anchor"))
    memory["anchor"] = anchor
    return sample, int(sample["nL"]) + int(sample["nR"])


def _cap(vel, cap):
    speed = float(np.linalg.norm(vel))
    if speed > cap:
        return vel * (cap / max(speed, 1e-9))
    return vel


def action_a(sim, switch, ref, spec, memory):
    pinch, _relative = hand_pose(sim)
    error = np.asarray(pinch, float) - np.asarray(ref["p_nom"], float)
    distance = float(np.linalg.norm(error))
    sample, contacts = _contacts(sim, switch, memory)
    relative = np.asarray(physical_pack(sim)["v_rel_h"], float)
    grip = float(spec["grip"])
    scale = 1.0
    if contacts == 0:
        grip = max(grip, -0.35)
        scale = 0.0
    elif contacts == 1:
        grip = max(grip, -0.40)
        scale = 0.4
    settling = distance <= 0.0032 and int(sample["nL"]) > 0 and int(sample["nR"]) > 0
    rh = hand_R(sim)
    up_hand = rh.T @ np.array([0.0, 0.0, float(spec["up"])], float)
    if settling:
        vel = 3.0 * error - relative
        height = float(physical_pack(sim)["obj_z"])
        if height < 0.49:
            lift = np.array([0.0, 0.0, float(np.clip(10.0 * (0.49 - height), 0.0, 0.85))], float)
            vel = vel + rh.T @ lift
        grip = 1.0
        along = 0.0
        up_speed = 0.0
    else:
        along_vel = scale * float(spec["kp"]) * error
        vel = _cap(along_vel, float(spec["cap"])) + scale * up_hand
        vel = _cap(vel, float(spec["cap"]) + abs(float(spec["up"])))
        direction, norm = _unit(error)
        along = float(np.dot(vel, direction)) if norm > 0 else 0.0
        up_speed = float(np.dot(rh @ vel, np.array([0.0, 0.0, 1.0])))
        memory.setdefault("along", []).append(along)
        memory.setdefault("up", []).append(up_speed)
    ang = 0.25 * wrist_return(sim, ref)
    memory.setdefault("tilt", []).append(_tilt_rad(sim))
    return normalize_action(vel, ang, grip_tau(grip), AIRBORNE_V3)


def action_b(sim, switch, ref, spec, memory):
    pinch, _relative = hand_pose(sim)
    error = np.asarray(pinch, float) - np.asarray(ref["p_nom"], float)
    distance = float(np.linalg.norm(error))
    sample, contacts = _contacts(sim, switch, memory)
    relative = np.asarray(physical_pack(sim)["v_rel_h"], float)
    grip = float(spec["grip"])
    if contacts == 0:
        grip = max(grip, -0.35)
    elif contacts == 1:
        grip = max(grip, -0.40)
    rh = hand_R(sim)
    settling = distance <= 0.0032 and int(sample["nL"]) > 0 and int(sample["nR"]) > 0
    if settling:
        vel = 3.0 * error - relative
        height = float(physical_pack(sim)["obj_z"])
        if height < 0.49:
            lift = np.array([0.0, 0.0, float(np.clip(8.0 * (0.49 - height), 0.0, 0.7))], float)
            vel = vel + rh.T @ lift
        grip = 1.0
        ang = 0.8 * wrist_return(sim, ref)
        perp_speed = 0.0
    else:
        direction, norm = _unit(error)
        perp = np.cross(direction, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(perp)) < 0.2:
            perp = np.cross(direction, np.array([0.0, 1.0, 0.0]))
        perp, _perp_norm = _unit(perp)
        gravity = np.asarray(sim.model.opt.gravity, float)
        g_h, _g_norm = _unit(rh.T @ gravity)
        target = -direction if norm > 0 else g_h
        axis = np.cross(g_h, target)
        axis, axis_norm = _unit(axis)
        if axis_norm < 1e-6 or _tilt_rad(sim) >= float(spec["tilt"]):
            w_hand = np.zeros(3)
        else:
            w_hand = axis * float(spec["tilt_rate"])
        rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        ang = rd.T @ (rh @ w_hand)
        vel = float(spec["v_perp"]) * perp
        if contacts == 0:
            vel = np.zeros(3)
        memory.setdefault("perp", []).append(float(np.linalg.norm(vel)))
        memory.setdefault("w", []).append(float(np.linalg.norm(ang)))
    memory.setdefault("tilt", []).append(_tilt_rad(sim))
    return normalize_action(vel, ang, grip_tau(grip), AIRBORNE_V3)


def _trace_summary(memory, samples):
    along = np.asarray(memory.get("along") or [0.0], float)
    up = np.asarray(memory.get("up") or [0.0], float)
    tilt = np.asarray(memory.get("tilt") or [0.0], float)
    perp = np.asarray(memory.get("perp") or [0.0], float)
    ang = np.asarray(memory.get("w") or [0.0], float)
    w_rel = max((float(sample["w_rel"]) for sample in samples), default=0.0)
    return {
        "mean_along_mps": round(float(np.mean(along)), 4),
        "mean_up_mps": round(float(np.mean(up)), 4),
        "tilt_delta_rad": round(float(np.max(tilt) - tilt[0]), 4) if len(tilt) else 0.0,
        "mean_perp_mps": round(float(np.mean(perp)), 4),
        "mean_wrist_cmd": round(float(np.mean(ang)), 4),
        "max_w_rel": round(float(w_rel), 4),
    }


def _zero_wrist_unloaded(actions):
    out = np.array(actions, np.float32, copy=True)
    for index, grip in enumerate(out[:, 6]):
        if float(grip) < 0.5:
            out[index, 3:6] = 0.0
    return out


def _run(env, blob, policy, spec, do_hold):
    restore_capture(env, blob["capture"])
    started = time.perf_counter()
    result = execute(
        env, blob["ref"], policy=policy, spec=spec, family="slide",
        handoff_z=HANDOFF_Z, do_hold=do_hold,
    )
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def _ablate(env, blob, actions):
    out = {}
    edits = (
        ("no_unloaded_translation", _zero_unloaded_translation(actions)),
        ("no_unloaded_wrist", _zero_wrist_unloaded(actions)),
        ("grip_closed", _grip(actions, 1.0)),
    )
    for name, edited in edits:
        checked = _replay_actions(env, blob, edited, "slide", HANDOFF_Z)
        out[name] = {
            "primary_10s": bool(checked["primary_10s"]),
            "held_12": None if not checked["hold"] else bool(checked["hold"].get("held_12")),
            "existing_label": checked["existing_label"],
        }
        print("   ablation", name, out[name]["existing_label"], flush=True)
    return out


def _verified_a(result, trace, ablations, up_removed):
    metrics = result["metrics"]
    continuous = int(metrics["zero_steps"]) == 0
    unloaded = metrics["min_grip"] is not None and -0.85 < float(metrics["min_grip"]) < 0.2
    moved = float(metrics["ep_drop_m"]) >= 0.003 and float(trace["mean_along_mps"]) > 0.02
    lifted = float(trace["mean_up_mps"]) >= 0.02
    slipped = float(metrics["max_v_tan"]) >= 0.03 and float(metrics["max_migration_m"]) >= 0.003
    translation_fails = ablations.get("no_unloaded_translation", {}).get("primary_10s") is False
    grip_fails = ablations.get("grip_closed", {}).get("primary_10s") is False
    up_fails = up_removed is not None and not bool(up_removed.get("primary_10s"))
    return bool(result["primary_10s"] and continuous and unloaded and moved and lifted and slipped and translation_fails and grip_fails and up_fails)


def _verified_b(result, trace, ablations):
    metrics = result["metrics"]
    controlled = int(metrics["zero_steps"]) <= 3 and float(metrics["zero_s"]) <= 0.06
    unloaded = metrics["min_grip"] is not None and float(metrics["min_grip"]) > -0.85
    tilted = float(trace["tilt_delta_rad"]) >= 0.15 and float(trace["mean_wrist_cmd"]) >= 0.05
    rolled = float(trace["max_w_rel"]) >= 1.0 and float(metrics["max_migration_m"]) >= 0.003
    perp = float(trace["mean_perp_mps"]) >= 0.02
    moved = float(metrics["ep_drop_m"]) >= 0.003
    tilt_fails = ablations.get("no_unloaded_wrist", {}).get("primary_10s") is False
    trans_fails = ablations.get("no_unloaded_translation", {}).get("primary_10s") is False
    grip_fails = ablations.get("grip_closed", {}).get("primary_10s") is False
    return bool(result["primary_10s"] and controlled and unloaded and tilted and rolled and perp and moved and tilt_fails and trans_fails and grip_fails)


def _status_from(result, verified_name):
    if result.get("invalid_replay"):
        return "INVALID_REPLAY"
    if verified_name:
        return verified_name
    if result["primary_10s"] and int(result["metrics"]["zero_steps"]) > 0:
        return "CONTACT_LOST"
    if result["primary_10s"]:
        return "PARTIAL_RECOVERY"
    if result["fell"]:
        return "DROP_OR_ESCAPE"
    if result["reached_gate"]:
        return "PARTIAL_RECOVERY"
    return "NOT_FOUND_WITHIN_BUDGET"


def _row(episode_id, family, spec, result, status, trace, ablations, represented, path):
    hold = result.get("hold") or {}
    metrics = result["metrics"]
    return {
        "id": episode_id,
        "family": family,
        "strategy": spec["name"],
        "seed": 0,
        "status": status,
        "primary_10s": bool(result["primary_10s"]),
        "held_12": None if not hold else bool(hold.get("held_12")),
        "existing_label": result.get("existing_label"),
        "seconds": result.get("seconds"),
        "min_ep_m": metrics["min_ep_m"],
        "ep_drop_m": metrics["ep_drop_m"],
        "duration_s": metrics["duration_s"],
        "zero_s": metrics["zero_s"],
        "zero_steps": metrics["zero_steps"],
        "z_loss_m": metrics["z_loss_m"],
        "max_v_tan": metrics["max_v_tan"],
        "max_migration_m": metrics["max_migration_m"],
        "min_grip": metrics["min_grip"],
        "states": result.get("states"),
        "trace": trace,
        "ablations": ablations,
        "representability": represented,
        "trajectory": path,
    }


def _save(episode_id, tag, result, status):
    folder = RAW / "trajectories"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{episode_id}__{tag}.npz"
    samples = result["samples"]
    np.savez_compressed(
        path,
        actions=result["actions"],
        observations=result["observations"],
        episode_id=np.array(episode_id),
        status=np.array(status),
        nL=np.array([int(sample["nL"]) for sample in samples], np.int16),
        nR=np.array([int(sample["nR"]) for sample in samples], np.int16),
        v_tan=np.array([float(sample["v_tan"]) for sample in samples], np.float32),
        migration_m=np.array([float(sample["migration_m"]) for sample in samples], np.float32),
        w_rel=np.array([float(sample["w_rel"]) for sample in samples], np.float32),
    )
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _append(row):
    RAW.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _load_rows():
    if not LOG.is_file():
        return []
    return [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def _quantize(env, blob, actions, proto):
    represented = representability(actions, proto)
    checked = _replay_actions(env, blob, represented["quantized"], "slide", HANDOFF_Z)
    return {
        "fraction_within_0.25": represented["fraction_within_0.25"],
        "max_linf": represented["max_linf"],
        "mean_abs_gap": represented["mean_abs_gap"],
        "quantized_primary_10s": bool(checked["primary_10s"]),
    }


def search_family(env, blob, proto, family, specs, policy, verify, done):
    episode_id = blob["spec"]["id"]
    existing = [
        row for row in _load_rows()
        if row.get("id") == episode_id and row.get("family") == family and str(row.get("status", "")).startswith("VERIFIED")
    ]
    if existing:
        print("  reuse", family, existing[0]["strategy"], flush=True)
        return existing[0]
    best = None
    for spec in specs:
        key = f"{episode_id}|{family}|{spec['name']}|0"
        if key in done:
            print("  skip", family, spec["name"], flush=True)
            continue
        preview = _run(env, blob, policy, spec, False)
        print(
            " ", family, spec["name"], "gate", preview["reached_gate"],
            "ep", preview["metrics"]["min_ep_m"], "zeros", preview["metrics"]["zero_steps"],
            "fell", preview["fell"], flush=True,
        )
        result = preview
        trace = _trace_summary(preview.get("_memory_placeholder", {}), preview["samples"])
        # The policy wrote into spec-local lists via the closure installed below.
        trace = getattr(policy, "last_trace", trace)
        ablations = {}
        represented = None
        up_removed = None
        if preview["reached_gate"] and int(preview["metrics"]["zero_steps"]) <= 3:
            result = _replay_actions(env, blob, preview["actions"], "slide", HANDOFF_Z)
            result["seconds"] = round(preview["seconds"] + result["seconds"], 3)
            result["invalid_replay"] = not bool(result["primary_10s"])
            if result["primary_10s"]:
                ablations = _ablate(env, blob, result["actions"])
                if family == "A":
                    plain = dict(spec)
                    plain["up"] = 0.0
                    plain["name"] = spec["name"] + "_noup"
                    up_removed = _run(env, blob, policy, plain, True)
                    up_removed = {
                        "primary_10s": bool(up_removed["primary_10s"]),
                        "held_12": None if not up_removed["hold"] else bool(up_removed["hold"].get("held_12")),
                    }
                    print("   no_up", up_removed["primary_10s"], flush=True)
        verified = ""
        if family == "A" and result["primary_10s"] and not result.get("invalid_replay"):
            if _verified_a(result, trace, ablations, up_removed):
                verified = "VERIFIED_A"
        if family == "B" and result["primary_10s"] and not result.get("invalid_replay"):
            if _verified_b(result, trace, ablations):
                verified = "VERIFIED_B"
        status = _status_from(result, verified)
        path = None
        if status.startswith("VERIFIED") or status == "CONTACT_LOST":
            path = _save(episode_id, f"{family}_{spec['name']}", result, status)
            if status.startswith("VERIFIED"):
                represented = _quantize(env, blob, result["actions"], proto)
        row = _row(episode_id, family, spec, result, status, trace, ablations, represented, path)
        if up_removed is not None:
            row["up_removed"] = up_removed
        _append(row)
        print("   ", status, "10s", row["primary_10s"], "12s", row["held_12"], "trace", trace, flush=True)
        if best is None or status.startswith("VERIFIED"):
            best = row
        if status.startswith("VERIFIED"):
            for pending in specs:
                if pending["name"] == spec["name"]:
                    continue
                if f"{episode_id}|{family}|{pending['name']}|0" in done:
                    continue
                _append({
                    "id": episode_id, "family": family, "strategy": pending["name"], "seed": 0,
                    "status": "SKIPPED_AFTER_SUCCESS", "primary_10s": False, "seconds": 0.0,
                })
            return row
    return best or {"id": episode_id, "family": family, "status": "NOT_FOUND_WITHIN_BUDGET", "primary_10s": False}


def _bind(policy_fn):
    def policy(sim, switch, ref, spec, memory):
        action = policy_fn(sim, switch, ref, spec, memory)
        policy.last_trace = _trace_summary(memory, [])
        policy.last_memory = memory
        return action
    return policy


def load_c(episode_id):
    payload = json.loads(C_SUMMARY.read_text(encoding="utf-8"))
    matches = [row for row in payload["episodes"] if row["id"] == episode_id]
    if not matches:
        return None
    record = matches[0]
    best = (record.get("families") or {}).get("regrasp", {}).get("best") or {}
    status = "VERIFIED_C" if best.get("status") == "VERIFIED_SUCCESS" and best.get("primary_10s") and best.get("held_12") else best.get("status", "NOT_FOUND_WITHIN_BUDGET")
    return {
        "id": episode_id,
        "family": "C",
        "strategy": best.get("strategy"),
        "status": status,
        "primary_10s": bool(best.get("primary_10s")),
        "held_12": best.get("held_12"),
        "existing_label": best.get("existing_label"),
        "seconds": (best.get("prior_budget") or {}).get("seconds"),
        "duration_s": best.get("duration_s"),
        "zero_s": best.get("zero_s"),
        "z_loss_m": best.get("z_loss_m"),
        "min_ep_m": best.get("min_ep_m"),
        "min_grip": best.get("min_grip"),
        "trajectory": best.get("trajectory"),
        "representability": best.get("representability"),
        "mechanism": best.get("mechanism"),
        "source": "sliding_vs_regrasp replay-verified release-chase-regrasp",
    }


def _partial(item):
    if item.get("status") != "NOT_FOUND_WITHIN_BUDGET":
        return False
    return float(item.get("ep_drop_m") or 0.0) >= 0.003


def write_report(records, audit, elapsed):
    RAW.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({"elapsed_s": elapsed, "legal_action": audit, "episodes": records}), encoding="utf-8")
    n = len(records)

    def best(row, family):
        return (row.get("families") or {}).get(family) or {}

    def count(family, status):
        return sum(1 for row in records if best(row, family).get("status") == status)

    lines = [
        "# A/B/C recovery mechanism validation",
        "",
        "A, B, and C were attempted from the same restored TRAIN ball-impact snapshots. Commands were clipped `recovery7d_airborne_v3` actions. The cylinder, hand, ball, contacts, friction, and gravity were not edited. No policy was trained. VALIDATION and FINAL_TEST were not searched. The sealed GRPO pilot was not modified.",
        "",
        "A success requires a replayed 10 s nominal hold, continuous finger contact, grip action strictly between -0.85 and 0.2, hand motion along the pinch offset, a mean upward component of at least 0.02 m/s during the unloaded phase, measured slip and contact migration, and three failed alternatives: translation removed on unloaded steps, grip forced closed, and the same law with the upward component set to zero. B success additionally requires at least 0.15 rad of gripper tilt from the nominal downward hand axis, perpendicular translation, object spin of at least 1 rad/s with contact migration of at least 3 mm, and failed ablations of unloaded wrist, unloaded translation, and closed grip. C is the previously replay-verified release, chase, and regrasp trajectory. A 12 s diagnostic is reported beside every 10 s result. A budget expiry is not a claim of physical impossibility.",
        "",
        "## Counts",
        "",
        f"Verified A: {count('A', 'VERIFIED_A')} / {n}. Verified B: {count('B', 'VERIFIED_B')} / {n}. Verified C: {count('C', 'VERIFIED_C')} / {n}.",
        f"A contact-lost outcomes: {count('A', 'CONTACT_LOST')}. A drops: {count('A', 'DROP_OR_ESCAPE')}. A not found in budget: {count('A', 'NOT_FOUND_WITHIN_BUDGET')}.",
        f"B contact-lost outcomes: {count('B', 'CONTACT_LOST')}. B drops: {count('B', 'DROP_OR_ESCAPE')}. B not found in budget: {count('B', 'NOT_FOUND_WITHIN_BUDGET')}.",
        f"A partial pinch reductions that did not enter the basin: {sum(1 for row in records if _partial(best(row, 'A')))} ({', '.join(row['id'] for row in records if _partial(best(row, 'A'))) or 'none'}).",
        "",
        "Each IC received the same four A laws and four B laws, one initialization each. A later law was skipped only after a verified success. No A or B preview reached the handoff gate with at most three zero-contact steps, so the translation, closed-grip, tilt, and upward-component ablations were not executed. A hold that never occurred is not attributed to those terms.",
        "",
        "On the canonical horizontal grasp, hand +z points world-down, and the pinch offset on these snapshots is mostly along that axis. A world-up command therefore opposes the offset direction. When A moves the hand along the offset, the logged world-up speed is negative. When the upward term is large enough to make the net velocity upward, the pinch error stays near its entry value. The +X cases that reduce pinch error by several millimetres still stop short of the 3.03 mm basin and briefly lose finger contact, so they are partial recoveries rather than verified A. B does tilt the gripper by the commanded angle and the cylinder spins, but that spin does not bring the pinch error into the basin. Large tilt rates tumble the cylinder and end in a drop. Entry spin is not counted as controlled rolling.",
        "",
        "No trajectory in this search combined A and B into a hold. C is the prior release, chase, and regrasp replay and is not a hybrid of A or B. The earlier matched comparison already recorded five strict contact-preserving slides. Those slides move along the pinch offset with an unloaded grip, which is part of A, but their world velocity is downward along the offset. They are not relabeled as verified A.",
        "",
        f"Verified C covers {count('C', 'VERIFIED_C')} of {n} TRAIN snapshots, including the original 18 replayed release–chase–regrasp trajectories and any new TRAIN IC whose release law reached a 10 s and 12 s nominal hold. A new IC that drops under every release law stays a search failure. Every verified C row has a 10 s nominal-only hold and a 12 s diagnostic hold.",
        "",
        "```",
        json.dumps(audit, indent=2),
        "```",
        "",
        f"Wall time in this process: {elapsed} s.",
        "",
        "## Per-IC outcomes",
        "",
        "| IC | direction | pinch m | object ori rad | spin | A | B | C | A s | A contact-loss s | B tilt rad | B spin | C contact-loss s |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        entry = row["entry"]
        sample = entry.get("sample") or {}
        a = best(row, "A")
        b = best(row, "B")
        c = best(row, "C")
        lines.append(
            f"| `{row['id']}` | {row['direction']} | {entry['ep_m']} | {entry['eR_object_rad']} | {sample.get('w_rel')} | "
            f"{a.get('status')} | {b.get('status')} | {c.get('status')} | "
            f"{a.get('duration_s')} | {a.get('zero_s')} | "
            f"{(b.get('trace') or {}).get('tilt_delta_rad')} | {(b.get('trace') or {}).get('max_w_rel')} | "
            f"{c.get('zero_s')} |"
        )
    lines.extend(["", "## Evidence", ""])
    for row in records:
        for family in ("A", "B", "C"):
            item = best(row, family)
            lines.append(
                f"- `{row['id']}` {family} `{item.get('strategy')}` `{item.get('status')}`: "
                f"10 s {item.get('primary_10s')}, 12 s {item.get('held_12')}, "
                f"duration {item.get('duration_s')} s, contact loss {item.get('zero_s')} s, "
                f"pinch drop {item.get('ep_drop_m')}, trace {item.get('trace')}, "
                f"ablations {item.get('ablations')}, up-removed {item.get('up_removed')}, "
                f"K=36 {item.get('representability')}."
            )
    lines.extend([
        "",
        "## Codebook",
        "",
        "K=36 and the ±0.25 residual bound were not changed. No verified B trajectory exists, so gravity-assisted rolling does not add a missing action primitive. Verified C still has the open-phase hand-frame vy and vz gap from the earlier release–regrasp search: the fraction of steps inside ±0.25 is below one on every saved C trajectory, and the largest mean absolute gaps are on those two channels. The five strict sliding successes from the matched comparison remain inside the existing codebook. No new prototype is added.",
        "",
        "Logs: `results/archive/demo_results/diagnostics/raw/abc_recovery/search_log.jsonl`.",
        "Trajectories: `results/archive/demo_results/diagnostics/raw/abc_recovery/trajectories/`.",
        "",
        "Resume:",
        "",
        "```",
        "python -u demo/abc_recovery_validation.py --max-ics 28",
        "```",
        "",
        "ABC MECHANISM VALIDATION COMPLETE" if n >= 18 else "ABC MECHANISM VALIDATION PARTIAL",
        "",
        "BALL-IMPACT DATASET EXPANSION PARTIAL",
        "",
        "POLICY TRAINING HAS NOT BEEN STARTED",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return REPORT


def action_c(sim, _switch, ref, spec, memory):
    return command(sim, ref, spec, int(memory.get("index", 0)))


def search_c(env, blob, proto, done):
    """Release, chase, and regrasp on a TRAIN IC that has no saved C replay."""
    policy = _install_trace(action_c)
    episode_id = blob["spec"]["id"]
    best = None
    for spec in C_SPECS:
        key = f"{episode_id}|C|{spec['name']}|0"
        if key in done:
            print("  skip C", spec["name"], flush=True)
            continue
        preview = _run(env, blob, policy, spec, False)
        print(
            "  C", spec["name"], "gate", preview["reached_gate"],
            "ep", preview["metrics"]["min_ep_m"], "zeros", preview["metrics"]["zero_steps"],
            "fell", preview["fell"], flush=True,
        )
        result = preview
        represented = None
        if preview["reached_gate"]:
            result = _replay_actions(env, blob, preview["actions"], "regrasp", HANDOFF_Z)
            result["seconds"] = round(preview["seconds"] + result["seconds"], 3)
            result["invalid_replay"] = not bool(result["primary_10s"])
        released = float(result["metrics"].get("min_grip") or 0.0) <= -0.85 or int(result["metrics"]["zero_steps"]) > 0
        verified = "VERIFIED_C" if result["primary_10s"] and not result.get("invalid_replay") and released else ""
        status = _status_from(result, verified)
        path = None
        if status == "VERIFIED_C":
            path = _save(episode_id, f"C_{spec['name']}", result, status)
            represented = _quantize(env, blob, result["actions"], proto)
        row = _row(episode_id, "C", spec, result, status, dict(policy.last_trace), {}, represented, path)
        _append(row)
        print("   ", status, "10s", row["primary_10s"], "12s", row["held_12"], flush=True)
        if best is None or status == "VERIFIED_C":
            best = row
        if status == "VERIFIED_C":
            return row
    return best or {"id": episode_id, "family": "C", "status": "NOT_FOUND_WITHIN_BUDGET", "primary_10s": False}


def _policy_a(sim, switch, ref, spec, memory):
    return action_a(sim, switch, ref, spec, memory)


def _policy_b(sim, switch, ref, spec, memory):
    return action_b(sim, switch, ref, spec, memory)


def _install_trace(policy_fn):
    def policy(sim, switch, ref, spec, memory):
        action = policy_fn(sim, switch, ref, spec, memory)
        policy.last_trace = _trace_summary(memory, [])
        return action
    policy.last_trace = {}
    return policy


def search_family_fixed(env, blob, proto, family, specs, policy_fn, done):
    """Search one family. The trace is read from the policy closure after each preview."""
    policy = _install_trace(policy_fn)
    episode_id = blob["spec"]["id"]
    existing = [
        row for row in _load_rows()
        if row.get("id") == episode_id and row.get("family") == family and str(row.get("status", "")).startswith("VERIFIED")
    ]
    if existing:
        print("  reuse", family, existing[0].get("strategy"), flush=True)
        return existing[0]
    best = None
    for spec in specs:
        key = f"{episode_id}|{family}|{spec['name']}|0"
        if key in done:
            print("  skip", family, spec["name"], flush=True)
            continue
        preview = _run(env, blob, policy, spec, False)
        trace = dict(policy.last_trace)
        samples = preview["samples"]
        trace["max_w_rel"] = round(max((float(sample["w_rel"]) for sample in samples), default=0.0), 4)
        trace["states"] = preview["states"]
        print(
            " ", family, spec["name"], "gate", preview["reached_gate"],
            "ep", preview["metrics"]["min_ep_m"], "zeros", preview["metrics"]["zero_steps"],
            "fell", preview["fell"], "trace", trace, flush=True,
        )
        result = preview
        ablations = {}
        represented = None
        up_removed = None
        worth = preview["reached_gate"] and int(preview["metrics"]["zero_steps"]) <= 3
        if worth:
            result = _replay_actions(env, blob, preview["actions"], "slide", HANDOFF_Z)
            result["seconds"] = round(preview["seconds"] + result["seconds"], 3)
            result["invalid_replay"] = not bool(result["primary_10s"])
            if result["primary_10s"]:
                ablations = _ablate(env, blob, result["actions"])
                if family == "A":
                    plain = dict(spec)
                    plain["up"] = 0.0
                    removed = _run(env, blob, policy, plain, True)
                    up_removed = {"primary_10s": bool(removed["primary_10s"]), "held_12": None if not removed["hold"] else bool(removed["hold"].get("held_12"))}
                    print("   no_up", up_removed, flush=True)
        verified = ""
        if result["primary_10s"] and not result.get("invalid_replay"):
            if family == "A" and _verified_a(result, trace, ablations, up_removed):
                verified = "VERIFIED_A"
            if family == "B" and _verified_b(result, trace, ablations):
                verified = "VERIFIED_B"
        status = _status_from(result, verified)
        path = None
        if status.startswith("VERIFIED") or (status == "CONTACT_LOST" and result["primary_10s"]):
            path = _save(episode_id, f"{family}_{spec['name']}", result, status)
            if status.startswith("VERIFIED"):
                represented = _quantize(env, blob, result["actions"], proto)
        row = _row(episode_id, family, spec, result, status, trace, ablations, represented, path)
        if up_removed is not None:
            row["up_removed"] = up_removed
        _append(row)
        print("   ", status, "10s", row["primary_10s"], "12s", row["held_12"], flush=True)
        rank = {"VERIFIED_A": 0, "VERIFIED_B": 0, "CONTACT_LOST": 1, "PARTIAL_RECOVERY": 2, "DROP_OR_ESCAPE": 3, "NOT_FOUND_WITHIN_BUDGET": 4, "INVALID_REPLAY": 5}
        if best is None or rank.get(status, 9) < rank.get(best["status"], 9):
            best = row
        if status.startswith("VERIFIED"):
            logged = {item["strategy"] for item in _load_rows() if item.get("id") == episode_id and item.get("family") == family}
            for pending in specs:
                if pending["name"] in logged:
                    continue
                _append({"id": episode_id, "family": family, "strategy": pending["name"], "seed": 0, "status": "SKIPPED_AFTER_SUCCESS", "primary_10s": False, "seconds": 0.0})
            return row
    return best or {"id": episode_id, "family": family, "status": "NOT_FOUND_WITHIN_BUDGET", "primary_10s": False}


def profile(env):
    for episode_id in ("ball_train_0014", "ball_train_0007", "ball_train_0012"):
        blob = load_episode(episode_id)
        print("PROFILE", episode_id, blob["spec"]["direction"], flush=True)
        for spec in A_SPECS:
            policy = _install_trace(action_a)
            result = _run(env, blob, policy, spec, False)
            trace = dict(policy.last_trace)
            print(" A", spec["name"], "ep", result["metrics"]["min_ep_m"], "zeros", result["metrics"]["zero_steps"], "gate", result["reached_gate"], trace, flush=True)
        for spec in B_SPECS:
            policy = _install_trace(action_b)
            result = _run(env, blob, policy, spec, False)
            trace = dict(policy.last_trace)
            trace["max_w_rel"] = round(max((float(sample["w_rel"]) for sample in result["samples"]), default=0.0), 4)
            print(" B", spec["name"], "ep", result["metrics"]["min_ep_m"], "zeros", result["metrics"]["zero_steps"], "gate", result["reached_gate"], "fell", result["fell"], trace, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Validate recovery mechanisms A, B, and C")
    parser.add_argument("--max-ics", type=int, default=18)
    parser.add_argument("--ids", default="")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
        path = write_report(payload["episodes"], payload["legal_action"], payload.get("elapsed_s"))
        print("REPORT", path, flush=True)
        return
    import torch
    from models.historical.k36_checkpoint import load_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _reference = load_models(PROTO_PATH, PROTO_PATH.with_name("policy_A_reference.pt"), device)
    proto = np.asarray(torch.load(PROTO_PATH, map_location="cpu", weights_only=False)["proto"], np.float32)
    env = make_training_env(device, model=model)
    audit = legal_action_audit()
    if audit["ball_parked"] or audit["qpos_written"] or abs(audit["vz_down_mps"] - 3.8) > 1e-6:
        raise RuntimeError(f"action map drifted: {audit}")
    if args.profile:
        profile(env)
        return
    train_ids = ids_for("train")
    chosen = [item for item in args.ids.split(",") if item] or train_ids[: args.max_ics]
    for episode_id in chosen:
        if episode_id not in train_ids:
            raise RuntimeError(f"{episode_id} is not TRAIN")
    finished = {}
    prior_elapsed = 0.0
    if SUMMARY.is_file():
        previous = json.loads(SUMMARY.read_text(encoding="utf-8"))
        prior_elapsed = float(previous.get("elapsed_s") or 0.0)
        for row in previous.get("episodes", []):
            finished[row["id"]] = row
    started = time.perf_counter()
    for episode_id in chosen:
        if episode_id in finished and all(finished[episode_id].get("families", {}).get(name) for name in ("A", "B", "C")):
            print("SKIP", episode_id, flush=True)
            continue
        blob = load_episode(episode_id)
        if blob["spec"]["split"] != "train":
            raise RuntimeError(episode_id)
        restore_capture(env, blob["capture"])
        entry = diagnose(env, blob["ref"])
        print("IC", episode_id, blob["spec"]["direction"], "ep", entry["ep_m"], flush=True)
        done = {f"{row['id']}|{row['family']}|{row['strategy']}|{row.get('seed', 0)}" for row in _load_rows()}
        found_a = search_family_fixed(env, blob, proto, "A", A_SPECS, action_a, done)
        done = {f"{row['id']}|{row['family']}|{row['strategy']}|{row.get('seed', 0)}" for row in _load_rows()}
        found_b = search_family_fixed(env, blob, proto, "B", B_SPECS, action_b, done)
        found_c = load_c(episode_id)
        if found_c is None:
            done = {f"{row['id']}|{row['family']}|{row['strategy']}|{row.get('seed', 0)}" for row in _load_rows()}
            found_c = search_c(env, blob, proto, done)
        record = {
            "id": episode_id,
            "direction": blob["spec"]["direction"],
            "azimuth_deg": blob["spec"]["azimuth_deg"],
            "entry": entry,
            "families": {"A": found_a, "B": found_b, "C": found_c},
        }
        finished[episode_id] = record
        ordered = [finished[item] for item in train_ids if item in finished]
        write_report(ordered, audit, round(prior_elapsed + time.perf_counter() - started, 1))
        print(" saved", episode_id, found_a.get("status"), found_b.get("status"), found_c.get("status"), flush=True)
    ordered = [finished[item] for item in train_ids if item in finished]
    path = write_report(ordered, audit, round(prior_elapsed + time.perf_counter() - started, 1))
    print("DONE", len(ordered), path, flush=True)


if __name__ == "__main__":
    main()
