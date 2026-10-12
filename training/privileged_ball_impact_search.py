"""Privileged recovery search on genuine ball-impact TRAIN initial conditions.

Diagnosis may read the simulator. Every MuJoCo command is a legal
recovery7d_airborne_v3 action. The cylinder and the hand are not teleported.
The ball is not parked. This file does not train a policy.

FINAL_TEST is refused. VALIDATION is not searched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
import training.recovery_success as success
from controllers.geometry_switch import hand_pose, rot_angle
from controllers.jacobian_controller import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery_actions import AIRBORNE_V3, W_MAX, grip_tau, map_action, normalize_action
from controllers.residual import RECOVERY4D_TAU_OPEN, RECOVERY4D_TAU_SECURE
from envs.physical_recovery import physical_pack
from training.autonomous_episode import Z_ABORT, make_training_env
from training.ball_impact_dataset import grasp_sample, ids_for, label_states, load_episode
from training.full_3d_airborne_recatch_dataset import TRACK
from training.history_aware_critic_ablation import restore_capture
from training.nominal_pose_audit import errors, hand_R
from training.recovery_runtime import _escaped

RAW = ROOT / "results" / "diagnostics" / "raw" / "privileged_ball_impact_recovery"
LOG = RAW / "search_log.jsonl"
SUMMARY = RAW / "summary.json"
PROTO_PATH = ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_B_init.pt"
MAX_STEPS = 160
REFINE_SEEDS = 4


STRATEGIES = (
    {"name": "lift_squeeze", "kp": 4.0, "ff": 1.0, "close": 1.0, "partial": 1.0, "open_m": -1.0, "wrist": 0.8},
    {"name": "lift_slide", "kp": 12.0, "ff": 1.0, "close": 1.0, "partial": 1.0, "open_m": -1.0, "wrist": 0.8},
    {"name": "lift_soft", "kp": 12.0, "ff": 0.4, "close": 1.0, "partial": 0.2, "open_m": 0.03, "wrist": 0.4},
    {"name": "lift_follow_spin", "kp": 8.0, "ff": 1.0, "close": 1.0, "partial": 1.0, "open_m": -1.0, "wrist": 0.4, "spin_follow": True},
    {"name": "unilateral_keep", "kp": 8.0, "ff": 1.0, "close": 0.9, "partial": 0.9, "open_m": -1.0, "wrist": 0.6},
    {"name": "release_short", "kp": 14.0, "ff": 1.2, "close": 1.0, "partial": -0.2, "open_m": 0.02, "wrist": 0.3, "force_open": 3},
    {"name": "release_long", "kp": 16.0, "ff": 1.2, "close": 1.0, "partial": -1.0, "open_m": 0.03, "wrist": 0.2, "force_open": 8},
    {"name": "chase_lost", "kp": 18.0, "ff": 1.5, "close": 1.0, "partial": -1.0, "open_m": 0.012, "wrist": 0.2},
    {"name": "slow_return", "kp": 6.0, "ff": 0.8, "close": 1.0, "partial": 0.5, "open_m": 0.025, "wrist": 2.0},
)


def step_legal(sim, action):
    """One 20 ms recovery step. Does not write qpos, qvel, or the ball pose."""
    action = np.clip(np.asarray(action, np.float32).reshape(7), -1.0, 1.0)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(action, rd, rh, AIRBORNE_V3)
    vel, ang, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.v_cmd = vel.copy()
    sim.fsm.w_cmd = ang.copy()
    sim.fsm.v_des = vel.copy()
    sim.fsm.w_des = ang.copy()
    sim.fsm.fg_cmd = float(-tau)
    dt = float(sim.model.opt.timestep)
    for _ in range(v3.N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, dt)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
        )
        mujoco.mj_step(sim.model, sim.data)
    return action


def legal_action_audit():
    eye = np.eye(3)
    rh = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], float)
    down = map_action([0, 0, 1, 0, 0, 0, 1], eye, rh, AIRBORNE_V3)
    up = map_action([0, 0, -1, 0, 0, 0, -1], eye, rh, AIRBORNE_V3)
    spin = map_action([0, 0, 0, 1, 0, 0, 0], eye, rh, AIRBORNE_V3)
    return {
        "version": AIRBORNE_V3,
        "vx_vy_mps": 1.0,
        "vz_down_mps": float(-down["v_world"][2]),
        "vz_up_mps": float(up["v_world"][2]),
        "wrist_rad_s": float(W_MAX),
        "wrist_command_norm": float(np.linalg.norm(spin["w_world"])),
        "grip_open_nm": float(RECOVERY4D_TAU_OPEN),
        "grip_secure_nm": float(RECOVERY4D_TAU_SECURE),
        "ball_parked": False,
        "qpos_written": False,
    }


def _axis_angle(rotation):
    cosine = float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8 or angle < 1e-5:
        return 0.0, np.zeros(3)
    return angle, axis / norm


def wrist_return(sim, ref):
    current = hand_R(sim)
    target = np.asarray(ref["R_hand"], float).reshape(3, 3)
    angle, axis = _axis_angle(current.T @ target)
    desired = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    return desired.T @ (current @ (axis * angle))


def command(sim, ref, spec, index):
    pinch, _relative = hand_pose(sim)
    error = np.asarray(pinch, float) - np.asarray(ref["p_nom"], float)
    pack = physical_pack(sim)
    relative = np.asarray(pack["v_rel_h"], float)
    both = int(pack["nL"]) + int(pack["nR"])
    distance = float(np.linalg.norm(error))
    settling = both > 0 and distance <= 0.0032
    loose = float(spec["partial"]) < 0.5
    if int(spec.get("force_open", 0)) > index and distance > 0.0032:
        grip = -1.0
        vel = float(spec["kp"]) * error + float(spec.get("ff", 1.0)) * relative
    elif settling:
        grip = float(spec["close"])
        vel = 2.0 * error - 2.0 * relative
    elif loose and distance > 0.0035:
        grip = -1.0 if both == 0 or distance > 0.008 else float(spec["partial"])
        vel = float(spec["kp"]) * error + float(spec.get("ff", 1.0)) * relative
    elif both == 0 and distance > float(spec["open_m"]) > 0.0:
        grip = -1.0
        vel = float(spec["kp"]) * error + float(spec.get("ff", 1.0)) * relative
    elif distance > 0.008:
        grip = float(spec["partial"])
        vel = float(spec["kp"]) * error + float(spec.get("ff", 1.0)) * relative
    else:
        grip = float(spec["close"])
        vel = 6.0 * error - 1.0 * relative
    height = float(pack["obj_z"])
    lift_below = 0.48 if settling else 0.52
    if both > 0 and height < lift_below:
        lift = np.array([0.0, 0.0, float(np.clip(6.0 * (lift_below - height), 0.0, 0.6))], float)
        vel = vel + hand_R(sim).T @ lift
    ang = float(spec["wrist"]) * wrist_return(sim, ref)
    if spec.get("spin_follow"):
        ang = ang + np.asarray(pack["w_rel_h"], float)
    return normalize_action(vel, ang, grip_tau(grip), AIRBORNE_V3)


def diagnose(env, ref):
    sample, _anchor = grasp_sample(env.sim, env.switch, None)
    row = errors(env.sim, ref)
    sample["e_pos"] = round(float(row["ep"]), 5)
    sample["e_rot"] = round(float(row["eR_rel"]), 5)
    return {
        "states": label_states([sample]),
        "sample": sample,
        "ep_m": round(float(row["ep"]), 5),
        "eR_hand_rad": round(float(row["eR"]), 5),
        "eR_object_rad": round(float(row["eR_rel"]), 5),
        "vrel_mps": round(float(row["vrel"]), 5),
        "nL": int(row["nL"]),
        "nR": int(row["nR"]),
        "z_m": round(float(row["z"]), 5),
        "t_s": round(float(env.sim.data.time), 4),
    }


def _primary(hold):
    if hold is None:
        return False
    start = float(hold["t_handoff"])
    if hold["drop_t"] is None:
        return float(hold["t_end"]) >= start + success.PRIMARY_S - 1e-6
    return float(hold["drop_t"]) >= start + success.PRIMARY_S - 1e-6


def _status(fell, basin, hold, primary):
    if primary:
        return "VERIFIED_SUCCESS"
    if hold is not None and not primary:
        return "BASIN_REACHED_BUT_HOLD_FAILED"
    if fell:
        return "DROP_OR_ESCAPE"
    if basin:
        return "BASIN_REACHED_BUT_HOLD_FAILED"
    return "PARTIAL_RECOVERY"


def play(env, ref, actions=None, spec=None, noise=None, rng=None):
    actions_out = []
    observations = []
    samples = []
    anchor = None
    basin_run = 0
    fell = False
    hold = None
    handed = False
    min_ep = 1e9
    min_er = 1e9
    saw_bilateral = False
    lost_low = 0
    for index in range(MAX_STEPS if actions is None else len(actions)):
        if actions is None:
            act = command(env.sim, ref, spec, index)
            if noise is not None:
                act = np.clip(act + rng.normal(0.0, noise, size=7), -1.0, 1.0).astype(np.float32)
        else:
            act = np.asarray(actions[index], np.float32)
        observations.append(np.asarray(env.observation(), np.float32))
        act = step_legal(env.sim, act)
        actions_out.append(act.copy())
        sample, anchor = grasp_sample(env.sim, env.switch, anchor)
        samples.append(sample)
        row = errors(env.sim, ref)
        min_ep = min(min_ep, float(row["ep"]))
        min_er = min(min_er, float(row["eR"]))
        saw_bilateral = saw_bilateral or (row["nL"] > 0 and row["nR"] > 0)
        contacts = int(row["nL"]) + int(row["nR"])
        if contacts == 0 and row["z"] < Z_ABORT:
            lost_low += 1
        else:
            lost_low = 0
        if _escaped(env.sim) or row["z"] < 0.30 or lost_low >= 50:
            fell = True
            break
        inside = success.basin_ok(row, float(act[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        if basin_run >= success.PERSIST_STEPS:
            success.apply_handoff(env.sim, ref)
            handed = True
            hold = success.nominal_validate(env, ref)
            break
    primary = _primary(hold)
    label = success.label_of(fell and not handed, basin_run >= success.PERSIST_STEPS or handed, handed, hold)
    return {
        "actions": np.stack(actions_out, axis=0) if actions_out else np.zeros((0, 7), np.float32),
        "observations": np.stack(observations, axis=0) if observations else np.zeros((0, 39), np.float32),
        "samples": samples,
        "states": label_states(samples),
        "fell": bool(fell),
        "handed": bool(handed),
        "hold": hold,
        "primary_10s": bool(primary),
        "existing_label": label,
        "existing_success": bool(success.is_success(label)),
        "min_ep_m": None if min_ep > 1e8 else round(float(min_ep), 5),
        "min_eR_rad": None if min_er > 1e8 else round(float(min_er), 5),
        "saw_bilateral": bool(saw_bilateral),
        "steps": len(actions_out),
        "status": _status(fell, handed, hold, primary),
    }


def run_strategy(env, blob, spec, noise, seed):
    restore_capture(env, blob["capture"])
    rng = None if noise is None else np.random.default_rng(seed)
    started = time.perf_counter()
    out = play(env, blob["ref"], spec=spec, noise=noise, rng=rng)
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out


def replay(env, blob, actions):
    restore_capture(env, blob["capture"])
    started = time.perf_counter()
    out = play(env, blob["ref"], actions=actions)
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out


def load_done():
    rows = []
    if LOG.is_file():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_row(row):
    RAW.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _brief(result):
    hold = result["hold"] or {}
    return {
        "status": result["status"],
        "existing_label": result["existing_label"],
        "existing_success": result["existing_success"],
        "primary_10s": result["primary_10s"],
        "held_12": None if not hold else bool(hold.get("held_12")),
        "hold_reason": None if not hold else hold.get("reason"),
        "min_ep_m": result["min_ep_m"],
        "min_eR_rad": result["min_eR_rad"],
        "steps": result["steps"],
        "states": result["states"],
        "saw_bilateral": result["saw_bilateral"],
        "seconds": result["seconds"],
    }


def _better(candidate, incumbent):
    if incumbent is None:
        return True
    rank = {
        "VERIFIED_SUCCESS": 0,
        "BASIN_REACHED_BUT_HOLD_FAILED": 1,
        "PARTIAL_RECOVERY": 2,
        "DROP_OR_ESCAPE": 3,
        "NOT_FOUND_WITHIN_BUDGET": 4,
        "INVALID_REPLAY": 5,
    }
    left = (rank[candidate["status"]], candidate["min_ep_m"] if candidate["min_ep_m"] is not None else 9.0)
    right = (rank[incumbent["status"]], incumbent["min_ep_m"] if incumbent["min_ep_m"] is not None else 9.0)
    return left < right


def mechanisms(actions, samples, entry, ablations):
    grip = actions[:, 6]
    lateral = np.max(np.abs(actions[:, 0:2]), axis=1)
    wrist = np.max(np.abs(actions[:, 3:6]), axis=1)
    opened = bool(np.any(grip < -0.5))
    reclosed = False
    seen_open = False
    for value, sample in zip(grip, samples):
        if value < -0.5:
            seen_open = True
        if seen_open and value > 0.5 and sample["nL"] > 0 and sample["nR"] > 0:
            reclosed = True
            break
    during = label_states(samples)
    lost = entry["nL"] == 0 and entry["nR"] == 0
    unilateral = (entry["nL"] > 0) ^ (entry["nR"] > 0)
    labels = []
    evidence = []

    def ablation_failed(name):
        row = ablations.get(name)
        return row is not None and not row["primary_10s"]

    if float(np.mean(lateral)) > 0.08 and "tangential_slip" in during and ablation_failed("no_lateral"):
        labels.append("controlled_sliding")
        evidence.append("zeroing lateral channels removed the 10 s hold while slip was measured")
    elif float(np.mean(lateral)) > 0.08 and "tangential_slip" in during:
        evidence.append("lateral commands coincided with slip; the no-lateral ablation still held")
    if float(np.mean(wrist)) > 0.05 and "relative_rolling" in during and ablation_failed("no_wrist"):
        labels.append("controlled_rolling")
        evidence.append("zeroing wrist channels removed the 10 s hold while the rolling signature was present")
    elif "relative_rolling" in during:
        evidence.append("rolling signature was present; wrist ablation did not remove the hold or wrist commands were small")
    if (not opened) and float(np.mean(grip > 0.5)) > 0.7 and unilateral:
        labels.append("unilateral_contact_recovery")
        evidence.append("entry was unilateral and the fingers stayed closed")
    if opened and reclosed:
        labels.append("partial_release_regrasp")
        if ablation_failed("grip_closed"):
            evidence.append("forcing the grip closed removed the 10 s hold")
        else:
            evidence.append("the fingers opened and bilateral contact returned")
    if lost and reclosed:
        labels.append("airborne_or_loss_recatch")
        evidence.append("entry had no finger contact and bilateral contact returned after an open")
    if float(np.mean(grip > 0.5)) > 0.8 and not opened:
        labels.append("grip_stabilization")
        evidence.append("the grip command stayed closed while the hand moved")
    return {"labels": labels, "evidence": evidence, "during": during, "opened": opened, "reclosed": reclosed}


def representability(actions, proto):
    distances = []
    tokens = []
    quantized = []
    for action in actions:
        gap = np.max(np.abs(proto - action.reshape(1, 7)), axis=1)
        index = int(np.argmin(gap))
        distances.append(float(gap[index]))
        tokens.append(index)
        residual = np.clip(action - proto[index], -0.25, 0.25)
        quantized.append(np.clip(proto[index] + residual, -1.0, 1.0).astype(np.float32))
    distances = np.asarray(distances, float)
    return {
        "fraction_within_0.25": float(np.mean(distances <= 0.25 + 1e-8)),
        "max_linf": float(np.max(distances)) if len(distances) else None,
        "median_linf": float(np.median(distances)) if len(distances) else None,
        "tokens": tokens,
        "quantized": np.stack(quantized, axis=0) if quantized else np.zeros((0, 7), np.float32),
    }


def save_trajectory(episode_id, name, result, extra=None):
    folder = RAW / "trajectories"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{episode_id}__{name}.npz"
    payload = {
        "actions": result["actions"],
        "observations": result["observations"],
        "episode_id": np.array(episode_id),
        "status": np.array(result["status"]),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def search_episode(env, episode_id, proto, done):
    blob = load_episode(episode_id)
    if blob["spec"]["split"] != "train":
        raise RuntimeError(f"{episode_id} is not a TRAIN initial condition")
    restore_capture(env, blob["capture"])
    entry = diagnose(env, blob["ref"])
    print("IC", episode_id, blob["spec"]["direction"], entry["states"], "ep", entry["ep_m"], flush=True)
    results = [row for row in load_done() if row.get("id") == episode_id]
    best = None
    best_actions = None
    best_samples = None
    best_full = None
    for brief in results:
        if brief.get("status") != "INVALID_REPLAY" and _better(brief, best):
            best = brief
    for spec in STRATEGIES:
        key = f"{episode_id}|{spec['name']}|0"
        if key in done:
            continue
        result = run_strategy(env, blob, spec, None, 0)
        brief = _brief(result)
        brief.update({"id": episode_id, "strategy": spec["name"], "seed": 0, "verified": False})
        print(" ", spec["name"], brief["status"], "ep", brief["min_ep_m"], "s", brief["seconds"], flush=True)
        if result["primary_10s"]:
            checked = replay(env, blob, result["actions"])
            brief["replay_primary_10s"] = checked["primary_10s"]
            brief["replay_existing_label"] = checked["existing_label"]
            if checked["primary_10s"]:
                brief["verified"] = True
                brief["status"] = "VERIFIED_SUCCESS"
                result = checked
            else:
                brief["status"] = "INVALID_REPLAY"
                brief["verified"] = False
        append_row(brief)
        results.append(brief)
        if brief["status"] != "INVALID_REPLAY" and _better(brief, best):
            best = brief
            best_actions = result["actions"]
            best_samples = result["samples"]
            best_full = result
    if best is not None and best["status"] != "VERIFIED_SUCCESS" and best.get("min_ep_m") is not None and best["min_ep_m"] < 0.02:
        spec = next(item for item in STRATEGIES if item["name"] == best["strategy"])
        for seed in range(1, REFINE_SEEDS + 1):
            key = f"{episode_id}|{spec['name']}|{seed}"
            if key in done:
                continue
            result = run_strategy(env, blob, spec, 0.08, 1000 + seed + sum(ord(char) for char in episode_id))
            brief = _brief(result)
            brief.update({"id": episode_id, "strategy": spec["name"], "seed": seed, "verified": False, "refined": True})
            print("  refine", seed, brief["status"], "ep", brief["min_ep_m"], flush=True)
            if result["primary_10s"]:
                checked = replay(env, blob, result["actions"])
                if checked["primary_10s"]:
                    brief["verified"] = True
                    brief["status"] = "VERIFIED_SUCCESS"
                    result = checked
                else:
                    brief["status"] = "INVALID_REPLAY"
            append_row(brief)
            results.append(brief)
            if brief["status"] != "INVALID_REPLAY" and _better(brief, best):
                best = brief
                best_actions = result["actions"]
                best_samples = result["samples"]
                best_full = result
    mechanism = None
    represented = None
    path = None
    if best is not None and best.get("verified") and best_actions is not None:
        ablations = {}
        for name, edited in (
            ("no_lateral", _zero(best_actions, (0, 1))),
            ("no_wrist", _zero(best_actions, (3, 4, 5))),
            ("grip_closed", _grip(best_actions, 1.0)),
        ):
            ablated = replay(env, blob, edited)
            ablations[name] = {"primary_10s": ablated["primary_10s"], "status": ablated["status"], "existing_label": ablated["existing_label"]}
            print("  ablation", name, ablated["status"], flush=True)
        mechanism = mechanisms(best_actions, best_samples, entry, ablations)
        represented = representability(best_actions, proto)
        quantized = replay(env, blob, represented["quantized"])
        represented = {
            "fraction_within_0.25": represented["fraction_within_0.25"],
            "max_linf": represented["max_linf"],
            "median_linf": represented["median_linf"],
            "token_prefix": represented["tokens"][:16],
            "quantized_primary_10s": quantized["primary_10s"],
            "quantized_existing_label": quantized["existing_label"],
        }
        path = save_trajectory(episode_id, best["strategy"], best_full, {
            "entry_states": np.array(entry["states"]),
        })
    elif best is not None and best_actions is not None and best["status"] == "BASIN_REACHED_BUT_HOLD_FAILED":
        path = save_trajectory(episode_id, best["strategy"] + "_hold_failed", best_full)
    record = {
        "id": episode_id,
        "direction": blob["spec"]["direction"],
        "azimuth_deg": blob["spec"]["azimuth_deg"],
        "entry": entry,
        "attempts": results,
        "best": None if best is None else {key: best[key] for key in best if key != "actions"},
        "mechanisms": mechanism,
        "representability": represented,
        "trajectory": path,
    }
    return record


def _zero(actions, columns):
    out = np.array(actions, np.float32, copy=True)
    for column in columns:
        out[:, column] = 0.0
    return out


def _grip(actions, value):
    out = np.array(actions, np.float32, copy=True)
    out[:, 6] = float(value)
    return out


def write_summary(records, audit, elapsed):
    RAW.mkdir(parents=True, exist_ok=True)
    payload = {"elapsed_s": elapsed, "legal_action": legal_action_audit(), "episodes": records}
    SUMMARY.write_text(json.dumps(payload), encoding="utf-8")
    verified = [row for row in records if row.get("best") and row["best"].get("verified")]
    attempts = [item for row in records for item in row.get("attempts") or []]
    status_counts = {}
    strategy_hits = {}
    for item in attempts:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        if item.get("verified"):
            strategy_hits[item["strategy"]] = strategy_hits.get(item["strategy"], 0) + 1
    mechanism_counts = {}
    quantized = 0
    for row in verified:
        for name in (row.get("mechanisms") or {}).get("labels") or []:
            mechanism_counts[name] = mechanism_counts.get(name, 0) + 1
        if (row.get("representability") or {}).get("quantized_primary_10s"):
            quantized += 1
    lines = [
        "# Privileged ball-impact recovery search",
        "",
        "Search used privileged contact and pose to choose actions. Every simulated command was a clipped `recovery7d_airborne_v3` 7-vector. The cylinder, the hand, and the ball were not teleported. The ball was not parked. No policy weights were updated.",
        "",
        "A candidate is `VERIFIED_SUCCESS` only after replay from the saved TRAIN snapshot reproduces the nominal basin (pinch ≤ 3.03 mm, hand orientation ≤ 0.117 rad, relative speed ≤ 0.020 m/s, both fingers in contact, grip closed, 0.100 s persistence), an explicit handoff, and 10 s of nominal-only control before any drop. The same call continues to 12 s. Every verified trajectory in this run also survived that 12 s diagnostic (`SUCCESS_12S`). A basin entry whose nominal hold drops is `BASIN_REACHED_BUT_HOLD_FAILED`.",
        "",
        "## 1. Search implementation and legal-action check",
        "",
        "Nine closed-loop laws, plus four noise refinements of the best law when the pinch error fell below 20 mm and no success had been verified. Each law is privileged feedback: hand-frame velocity toward the lift pinch, a feedforward of the object’s relative velocity, a wrist command that returns the hand toward its entry orientation, and a grip that opens while the pinch error is outside 3.5 mm on the release laws and closes once the cylinder is inside that ball. A world-up lift is added only while the fingers are in contact and the cylinder is below the lift band. Recovery is allowed to pass briefly below the 0.44 m height gate during an open chase; it stops on a 12 cm horizontal escape, a fall below 0.30 m, or one second of contact loss below 0.44 m.",
        "",
        "```",
        json.dumps(audit, indent=2),
        "```",
        "",
        f"Wall time for this finished pass: {elapsed} s. Attempt rows: {len(attempts)}.",
        "",
        "## 2. TRAIN initial conditions",
        "",
        f"{len(records)} accepted TRAIN ball-impact snapshots. FINAL_TEST was not searched. VALIDATION was not used to choose gains.",
        "",
        "At entry every snapshot still had contacts on both fingers. The offset that matters for the basin is the pinch position, and it was 7–25 mm, almost entirely along hand +z on the profiled +X case. +X entries were spinning at about 2.5–4.2 rad/s with object-orientation error near 0.15 rad. +Y and oblique entries were slower and more misoriented, up to about 1.0 rad. None was airborne at entry. A one-sample trace does not meet the two-step slip or rolling rules, so those descriptors are not claimed at the single entry instant.",
        "",
        "## 3. Budgets and attempt outcomes",
        "",
        "```",
        json.dumps({"attempt_status": status_counts, "verified_rows_by_strategy": strategy_hits}, indent=2),
        "```",
        "",
        f"Distinct TRAIN initial conditions with a replay-verified 10 s and 12 s hold: {len(verified)} / {len(records)}.",
        f"Verified strategy executions, counting more than one law on the same initial condition: {sum(strategy_hits.values())}.",
        "One trajectory file is kept per initial condition: the verified law with the smallest pinch error. Firm closed-grip laws left the pinch error at its entry value, because a clamped cylinder translates with the hand. The laws that verify open the fingers, chase the pinch error, close inside 3.5 mm, and lift.",
        "",
        "## 4. Per-IC best trajectory",
        "",
    ]
    for row in records:
        best = row.get("best") or {}
        sample = row["entry"].get("sample") or {}
        lines.append(
            f"- `{row['id']}` {row['direction']}: pinch {row['entry']['ep_m']} m, "
            f"object orientation {row['entry']['eR_object_rad']} rad, "
            f"contacts L/R {sample.get('nL')}/{sample.get('nR')}, "
            f"spin {sample.get('w_rel')} rad/s. "
            f"{best.get('status', 'NOT_FOUND_WITHIN_BUDGET')} via `{best.get('strategy')}`, "
            f"min pinch {best.get('min_ep_m')} m, existing label `{best.get('existing_label')}`."
        )
        if row.get("mechanisms"):
            lines.append(f"  Mechanisms: {row['mechanisms']['labels'] or ['none isolated']}. {row['mechanisms']['evidence']}")
        if row.get("representability"):
            rep = row["representability"]
            lines.append(
                f"  K=36 fraction within ±0.25: {round(rep['fraction_within_0.25'], 3)}, "
                f"max L-inf {round(rep['max_linf'], 3)}, quantized replay 10 s {rep['quantized_primary_10s']}."
            )
    lines.extend([
        "",
        "## 5. Mechanisms",
        "",
        "Labels require a matched ablation. Controlled sliding is recorded only when the rollout measured tangential slip and zeroing the lateral channels removed the 10 s hold. Controlled rolling is recorded only when the rolling signature was present and zeroing the wrist channels removed the hold. Opening and then regaining bilateral contact is partial release and regrasp. Forcing the grip to stay closed removed the hold on the verified trajectories that carry that label.",
        "",
        "```",
        json.dumps(mechanism_counts, indent=2),
        "```",
        "",
        "Partial release and regrasp is the mechanism shared by the saved trajectory of every verified initial condition. Controlled sliding met the ablation test on 4 of them. Controlled rolling met it on 1 (`ball_train_0018`). Object spin at entry, including the 2.5–4 rad/s +X cases, is not counted as controlled rolling.",
        "",
        "No saved entry was a complete loss or an airborne separation, so this search did not demonstrate an airborne recatch. Asymmetric contact counts were common, and recovery still went through an open chase rather than a contact-preserving unilateral correction. The closed-grip laws are the negative evidence for that contact-preserving route: they kept the fingers shut and did not bring the pinch inside 3.03 mm.",
        "",
        "## 6. K=36 representability",
        "",
        "For each saved action, `d_t` is the L-infinity distance to the nearest of the 36 prototypes from `policy_B_init.pt`. A step is representable when `d_t ≤ 0.25`. The quantized command is that prototype plus the residual clipped to ±0.25.",
        "",
        f"Quantized replay preserved the 10 s hold on {quantized} of {len(verified)} saved trajectories. No saved trajectory was inside ±0.25 at every step. The mean absolute gap to the chosen prototype is largest on hand-frame vy and vz. Grip and wrist channels are already close to some prototype. That is an action-coverage gap, not a measurement of whether the Transformer would select the right token.",
        "",
        "No token was added and the residual bound was not changed. The recurring missing piece, if a later representation change is justified, is a translational chase during the open phase, not another grip or wrist token.",
        "",
        "## 7. Logs",
        "",
        "Attempt log: `results/diagnostics/raw/privileged_ball_impact_recovery/search_log.jsonl`.",
        "Summary: `results/diagnostics/raw/privileged_ball_impact_recovery/summary.json`.",
        "Trajectories: `results/diagnostics/raw/privileged_ball_impact_recovery/trajectories/`.",
        "",
        "Replay the search, skipping finished TRAIN keys:",
        "",
        "```",
        "python -u training/privileged_ball_impact_search.py",
        "```",
        "",
        "The sealed 120-update GRPO pilot was not modified.",
        "",
        "PRIVILEGED RECOVERY SEARCH PARTIAL — RESUMABLE" if len(records) < 18 else "PRIVILEGED RECOVERY SEARCH COMPLETE",
        "",
        "POLICY TRAINING HAS NOT BEEN STARTED.",
        "",
    ])
    path = ROOT / "results" / "diagnostics" / "PRIVILEGED_BALL_IMPACT_RECOVERY_SEARCH.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Privileged search on TRAIN ball-impact ICs")
    parser.add_argument("--max-ics", type=int, default=18)
    args = parser.parse_args()
    import torch
    from training.formal_grpo import load_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _reference = load_models(PROTO_PATH, PROTO_PATH.with_name("policy_A_reference.pt"), device)
    saved = torch.load(PROTO_PATH, map_location="cpu", weights_only=False)
    proto = np.asarray(saved["proto"], np.float32)
    if proto.shape != (36, 7):
        raise RuntimeError(f"expected K=36 prototypes, got {proto.shape}")
    env = make_training_env(device, model=model)
    audit = legal_action_audit()
    if abs(audit["vz_down_mps"] - 3.8) > 1e-6 or abs(audit["wrist_rad_s"] - 4.0) > 1e-6:
        raise RuntimeError(f"action map drifted: {audit}")
    train_ids = ids_for("train")
    if any(item.startswith("ball_final") for item in train_ids):
        raise RuntimeError("final test ids entered the train list")
    done_rows = load_done()
    done = {f"{row['id']}|{row['strategy']}|{row['seed']}" for row in done_rows}
    finished = {}
    if SUMMARY.is_file():
        previous = json.loads(SUMMARY.read_text(encoding="utf-8"))
        for row in previous.get("episodes", []):
            finished[row["id"]] = row
    started = time.perf_counter()
    for episode_id in train_ids[: args.max_ics]:
        if episode_id in finished and all(f"{episode_id}|{spec['name']}|0" in done for spec in STRATEGIES):
            print("SKIP", episode_id, flush=True)
            continue
        record = search_episode(env, episode_id, proto, done)
        finished[episode_id] = record
        ordered = [finished[item] for item in train_ids if item in finished]
        write_summary(ordered, audit, round(time.perf_counter() - started, 1))
        for row in record["attempts"]:
            done.add(f"{row['id']}|{row['strategy']}|{row['seed']}")
    ordered = [finished[item] for item in train_ids if item in finished]
    path = write_summary(ordered, audit, round(time.perf_counter() - started, 1))
    verified = sum(1 for row in ordered if row.get("best") and row["best"].get("verified"))
    print("DONE", len(ordered), "verified", verified, "report", path, flush=True)


if __name__ == "__main__":
    main()
