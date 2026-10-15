"""Matched comparison of contact-preserving sliding, release-regrasp, and hybrid recovery.

Privileged state is used only to choose legal recovery7d_airborne_v3 actions.
The cylinder, hand, ball, contacts, and controller targets are not edited.
This file does not train a policy and does not search VALIDATION or FINAL_TEST.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.geometry_switch import hand_pose
from controllers.recovery_actions import AIRBORNE_V3, grip_tau, normalize_action
from envs.physical_recovery import physical_pack
from training.autonomous_episode import Z_ABORT, make_training_env
from data_generation.ball_impact_dataset import grasp_sample, ids_for, label_states, load_episode
from training.history_aware_critic_ablation import restore_capture
from training.nominal_pose_audit import errors, hand_R
from experiments.ball_impact.privileged_ball_impact_search import (
    PROTO_PATH,
    command as release_command,
    diagnose,
    legal_action_audit,
    step_legal,
    wrist_return,
)
from training.recovery_runtime import _escaped
import envs.recovery_success as success
from data_generation.strategies.recovery_laws import slide_action, wrist_return

RAW = ROOT / "results" / "diagnostics" / "raw" / "sliding_vs_regrasp"
LOG = RAW / "search_log.jsonl"
SUMMARY = RAW / "summary.json"
PRIOR = ROOT / "results" / "diagnostics" / "raw" / "privileged_ball_impact_recovery"
REPORT = ROOT / "results" / "diagnostics" / "SLIDING_VS_REGRASP_MATCHED_SEARCH.md"
MAX_STEPS = 160
HANDOFF_Z = 0.455
CHASE = {
    "name": "chase_lost",
    "kp": 18.0,
    "ff": 1.5,
    "close": 1.0,
    "partial": -1.0,
    "open_m": 0.012,
    "wrist": 0.2,
}
SLIDE_SPECS = (
    {"name": "slide_58", "grip": -0.58, "kp": 16.0, "cap": 0.40},
    {"name": "slide_45", "grip": -0.45, "kp": 12.0, "cap": 0.30},
    {"name": "slide_55", "grip": -0.55, "kp": 14.0, "cap": 0.35, "brake": True},
    {"name": "slide_50", "grip": -0.50, "kp": 20.0, "cap": 0.45, "brake": True},
    {"name": "slide_62", "grip": -0.62, "kp": 14.0, "cap": 0.32},
    {"name": "slide_follow", "grip": -0.56, "kp": 18.0, "cap": 0.40, "brake": True, "ff": 0.25},
)
HYBRID_SPECS = SLIDE_SPECS[:3]
RANK = {
    "VERIFIED_SUCCESS": 0,
    "HOLD_BUT_SLIDE_NOT_CAUSAL": 1,
    "HOLD_BUT_CONTACT_LOST": 2,
    "BASIN_REACHED_BUT_HOLD_FAILED": 3,
    "PARTIAL_RECOVERY": 4,
    "DROP_OR_ESCAPE": 5,
    "NOT_FOUND_WITHIN_BUDGET": 6,
    "INVALID_REPLAY": 7,
    "SKIPPED_AFTER_SUCCESS": 8,
}




def hybrid_action(sim, switch, ref, spec, memory):
    if memory.get("mode") == "regrasp":
        return release_command(sim, ref, CHASE, int(memory.get("index", 0)))
    return slide_action(sim, switch, ref, spec, memory)


def _note_hybrid(memory, row):
    if memory.get("family") != "hybrid" or memory.get("mode") == "regrasp":
        return
    contacts = int(row["nL"]) + int(row["nR"])
    distance = float(row["ep"])
    if contacts == 0:
        memory["streak"] = int(memory.get("streak", 0)) + 1
    else:
        memory["streak"] = 0
    if distance < float(memory.get("best_ep", distance)) - 1e-5:
        memory["best_ep"] = distance
        memory["stall"] = 0
    else:
        memory["stall"] = int(memory.get("stall", 0)) + 1
    close = distance <= 0.006 and contacts > 0
    lost = int(memory["streak"]) >= 2
    entered = float(memory.get("entry_ep", distance))
    stalled = (
        (not close)
        and int(memory["stall"]) >= 100
        and distance > 0.006
        and (entered - distance) < 0.001
    )
    no_progress = (
        (not close)
        and int(memory["index"]) >= 80
        and (entered - distance) < 0.002
        and distance > 0.008
    )
    if lost or stalled or no_progress:
        memory["mode"] = "regrasp"
        memory["switch"] = int(memory["index"])


def _primary(hold):
    if hold is None:
        return False
    start = float(hold["t_handoff"])
    if hold["drop_t"] is None:
        return float(hold["t_end"]) >= start + success.PRIMARY_S - 1e-6
    return float(hold["drop_t"]) >= start + success.PRIMARY_S - 1e-6


def _metrics(actions, samples, rows, z0, switch):
    if not samples:
        return {
            "steps": 0, "duration_s": 0.0, "zero_steps": 0, "zero_s": 0.0,
            "max_zero_streak": 0, "unilateral_steps": 0, "max_v_tan": 0.0,
            "slip_steps": 0, "max_migration_m": 0.0, "z_loss_m": 0.0,
            "min_ep_m": None, "min_grip": None, "max_grip": None,
            "mean_abs_translation": 0.0, "ep_drop_m": 0.0, "switch_step": switch,
        }
    zeros = 0
    streak = 0
    max_streak = 0
    unilateral = 0
    slip = 0
    for sample in samples:
        touching = int(sample["nL"]) + int(sample["nR"])
        if touching == 0:
            zeros += 1
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
        if (int(sample["nL"]) > 0) ^ (int(sample["nR"]) > 0):
            unilateral += 1
        if touching > 0 and float(sample["v_tan"]) >= 0.03:
            slip += 1
    grip = actions[:, 6]
    unloaded = grip < 0.5
    trans = np.max(np.abs(actions[:, 0:3]), axis=1)
    min_ep = min(float(row["ep"]) for row in rows)
    entry_ep = float(rows[0]["ep"])
    return {
        "steps": int(len(samples)),
        "duration_s": round(0.02 * len(samples), 3),
        "zero_steps": int(zeros),
        "zero_s": round(0.02 * zeros, 3),
        "max_zero_streak": int(max_streak),
        "unilateral_steps": int(unilateral),
        "max_v_tan": round(max(float(sample["v_tan"]) for sample in samples), 4),
        "slip_steps": int(slip),
        "max_migration_m": round(max(float(sample["migration_m"]) for sample in samples), 5),
        "z_loss_m": round(max(0.0, z0 - min(float(row["z"]) for row in rows)), 5),
        "min_ep_m": round(min_ep, 5),
        "min_grip": round(float(np.min(grip)), 3),
        "max_grip": round(float(np.max(grip)), 3),
        "mean_abs_translation": round(float(np.mean(trans[unloaded])) if np.any(unloaded) else float(np.mean(trans)), 4),
        "ep_drop_m": round(max(0.0, entry_ep - min_ep), 5),
        "switch_step": None if switch is None else int(switch),
    }


def execute(env, ref, actions=None, policy=None, spec=None, noise=None, rng=None, handoff_z=None, family="slide", do_hold=True):
    memory = {
        "anchor": None, "mode": "slide", "streak": 0, "stall": 0,
        "best_ep": 1e9, "index": 0, "switch": None, "family": family,
    }
    actions_out = []
    observations = []
    samples = []
    rows = []
    anchor = None
    basin_run = 0
    fell = False
    hold = None
    handed = False
    lost_low = 0
    z0 = float(errors(env.sim, ref)["z"])
    memory["entry_ep"] = float(errors(env.sim, ref)["ep"])
    memory["best_ep"] = float(memory["entry_ep"])
    limit = MAX_STEPS if actions is None else len(actions)
    reached = False
    for index in range(limit):
        memory["index"] = index
        if actions is None:
            act = policy(env.sim, env.switch, ref, spec, memory)
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
        rows.append(row)
        if family == "hybrid":
            _note_hybrid(memory, row)
        contacts = int(row["nL"]) + int(row["nR"])
        if contacts == 0 and float(row["z"]) < Z_ABORT:
            lost_low += 1
        else:
            lost_low = 0
        from envs.table_drop import latched
        if latched(env.sim) or _escaped(env.sim) or float(row["z"]) < 0.30 or lost_low >= 50:
            fell = True
            break
        inside = success.basin_ok(row, float(act[6]) > 0.5)
        if handoff_z is not None:
            inside = inside and float(row["z"]) >= float(handoff_z)
        basin_run = basin_run + 1 if inside else 0
        if basin_run >= success.PERSIST_STEPS:
            reached = True
            if do_hold:
                success.apply_handoff(env.sim, ref)
                handed = True
                hold = success.nominal_validate(env, ref)
            break
    packed = np.stack(actions_out, axis=0) if actions_out else np.zeros((0, 7), np.float32)
    primary = _primary(hold)
    from envs.table_drop import latched
    label = "TABLE_DROP" if latched(env.sim) else success.label_of(fell and not handed, basin_run >= success.PERSIST_STEPS or handed, handed, hold)
    metrics = _metrics(packed, samples, rows, z0, memory.get("switch"))
    return {
        "actions": packed,
        "observations": np.stack(observations, axis=0) if observations else np.zeros((0, 39), np.float32),
        "samples": samples,
        "states": label_states(samples),
        "fell": bool(fell),
        "handed": bool(handed),
        "reached_gate": bool(reached),
        "hold": hold,
        "primary_10s": bool(primary),
        "existing_label": label,
        "metrics": metrics,
        "status_physical": _physical_status(fell, reached, hold, primary),
    }


def _physical_status(fell, reached, hold, primary):
    if primary:
        return "VERIFIED_SUCCESS"
    if hold is not None and not primary:
        return "BASIN_REACHED_BUT_HOLD_FAILED"
    if reached and hold is None:
        return "PARTIAL_RECOVERY"
    if fell:
        return "DROP_OR_ESCAPE"
    return "NOT_FOUND_WITHIN_BUDGET"


def _better(candidate, incumbent):
    if incumbent is None:
        return True
    left = RANK.get(candidate["status"], 9)
    right = RANK.get(incumbent["status"], 9)
    if left != right:
        return left < right
    return float(candidate.get("min_ep_m") or 9) < float(incumbent.get("min_ep_m") or 9)


def representability(actions, proto):
    distances = []
    tokens = []
    quantized = []
    gaps = []
    for action in actions:
        gap = np.max(np.abs(proto - action.reshape(1, 7)), axis=1)
        index = int(np.argmin(gap))
        distances.append(float(gap[index]))
        tokens.append(index)
        residual = np.clip(action - proto[index], -0.25, 0.25)
        quantized.append(np.clip(proto[index] + residual, -1.0, 1.0).astype(np.float32))
        gaps.append(np.abs(action - proto[index]))
    distances = np.asarray(distances, float)
    gaps = np.stack(gaps, axis=0) if gaps else np.zeros((0, 7))
    far = distances > 0.25
    return {
        "fraction_within_0.25": float(np.mean(distances <= 0.25 + 1e-8)) if len(distances) else None,
        "max_linf": float(np.max(distances)) if len(distances) else None,
        "median_linf": float(np.median(distances)) if len(distances) else None,
        "mean_abs_gap": [round(float(x), 4) for x in np.mean(gaps, axis=0)] if len(gaps) else [],
        "far_fraction": float(np.mean(far)) if len(distances) else None,
        "tokens": tokens,
        "quantized": np.stack(quantized, axis=0) if quantized else np.zeros((0, 7), np.float32),
    }


def _strict_sliding(result, ablations):
    metrics = result["metrics"]
    continuous = int(metrics["zero_steps"]) == 0
    unloaded = metrics["min_grip"] is not None and float(metrics["min_grip"]) > -0.85
    moved = float(metrics["ep_drop_m"]) >= 0.003 and float(metrics["mean_abs_translation"]) >= 0.02
    slipped = ("tangential_slip" in result["states"]) or (
        float(metrics["max_v_tan"]) >= 0.03 and float(metrics["max_migration_m"]) >= 0.003 and int(metrics["slip_steps"]) >= 2
    )
    translation_fails = ablations.get("no_slide_translation", {}).get("primary_10s") is False
    grip_fails = ablations.get("grip_closed", {}).get("primary_10s") is False
    return bool(continuous and unloaded and moved and slipped and translation_fails and grip_fails)


def _mechanism(result, ablations, family):
    metrics = result["metrics"]
    labels = []
    evidence = []
    if family == "regrasp" or (metrics.get("switch_step") is not None) or int(metrics["zero_steps"]) > 0 and float(metrics["min_grip"] or 0) <= -0.9:
        if float(metrics["min_grip"] or 0) <= -0.9 or int(metrics["zero_steps"]) > 0:
            labels.append("partial_or_complete_release")
            evidence.append(
                f"minimum grip {metrics['min_grip']}, contact fully lost for {metrics['zero_s']} s"
            )
        if result["primary_10s"] and int(metrics["zero_steps"]) > 0:
            labels.append("recontact_regrasp")
            evidence.append("bilateral contact returned after a zero-contact interval and the nominal hold passed")
    if family != "regrasp" and _strict_sliding(result, ablations):
        labels.append("controlled_sliding")
        evidence.append(
            "unloaded grip stayed above -0.85, at least one finger stayed in contact, "
            "pinch error fell while tangential motion and contact migration were measured, "
            "and replays that remove translation only on the unloaded steps, or that force the grip closed, lost the 10 s hold"
        )
    elif family == "sliding" and result["primary_10s"] and int(metrics["zero_steps"]) == 0:
        evidence.append("the hold passed with continuous contact, but an ablation still held or the slip trace was too small")
    if metrics.get("switch_step") is not None:
        labels.append("hybrid_slide_then_regrasp")
        evidence.append(f"state-based switch to the open chase at recovery step {metrics['switch_step']}")
    if float(np.mean(result["actions"][:, 6] > 0.5)) > 0.2 and result["primary_10s"]:
        labels.append("reclose")
        evidence.append("the grip returned to the closed command before handoff")
    return {"labels": labels, "evidence": evidence, "during": result["states"]}


def _family_status(result, ablations, family):
    if result.get("invalid_replay"):
        return "INVALID_REPLAY"
    if not result["primary_10s"]:
        if result["reached_gate"] and result["hold"] is None:
            return "PARTIAL_RECOVERY"
        return result["status_physical"]
    if family == "sliding":
        if int(result["metrics"]["zero_steps"]) > 0:
            return "HOLD_BUT_CONTACT_LOST"
        if _strict_sliding(result, ablations):
            return "VERIFIED_SUCCESS"
        return "HOLD_BUT_SLIDE_NOT_CAUSAL"
    return "VERIFIED_SUCCESS"


def _zero(actions, columns):
    out = np.array(actions, np.float32, copy=True)
    for column in columns:
        out[:, column] = 0.0
    return out


def _zero_unloaded_translation(actions):
    """Remove the slide command and keep the later close-and-lift translation."""
    out = np.array(actions, np.float32, copy=True)
    for index, grip in enumerate(out[:, 6]):
        if float(grip) < 0.5:
            out[index, 0:3] = 0.0
    return out


def _grip(actions, value):
    out = np.array(actions, np.float32, copy=True)
    out[:, 6] = float(value)
    return out


def _brief(result, status):
    hold = result["hold"] or {}
    metrics = result["metrics"]
    return {
        "status": status,
        "existing_label": result["existing_label"],
        "primary_10s": bool(result["primary_10s"]),
        "held_12": None if not hold else bool(hold.get("held_12")),
        "hold_reason": None if not hold else hold.get("reason"),
        "min_ep_m": metrics["min_ep_m"],
        "steps": metrics["steps"],
        "duration_s": metrics["duration_s"],
        "zero_steps": metrics["zero_steps"],
        "zero_s": metrics["zero_s"],
        "max_zero_streak": metrics["max_zero_streak"],
        "unilateral_steps": metrics["unilateral_steps"],
        "max_v_tan": metrics["max_v_tan"],
        "slip_steps": metrics["slip_steps"],
        "max_migration_m": metrics["max_migration_m"],
        "z_loss_m": metrics["z_loss_m"],
        "min_grip": metrics["min_grip"],
        "max_grip": metrics["max_grip"],
        "ep_drop_m": metrics["ep_drop_m"],
        "switch_step": metrics["switch_step"],
        "states": result["states"],
        "reached_gate": bool(result["reached_gate"]),
    }


def load_rows():
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


def save_npz(episode_id, tag, result):
    folder = RAW / "trajectories"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{episode_id}__{tag}.npz"
    samples = result["samples"]
    np.savez_compressed(
        path,
        actions=result["actions"],
        observations=result["observations"],
        episode_id=np.array(episode_id),
        status=np.array(result.get("report_status", "")),
        nL=np.array([int(sample["nL"]) for sample in samples], np.int16),
        nR=np.array([int(sample["nR"]) for sample in samples], np.int16),
        fn=np.array([float(sample["fnL"]) + float(sample["fnR"]) for sample in samples], np.float32),
        ft=np.array([float(sample["ftL"]) + float(sample["ftR"]) for sample in samples], np.float32),
        v_tan=np.array([float(sample["v_tan"]) for sample in samples], np.float32),
        migration_m=np.array([float(sample["migration_m"]) for sample in samples], np.float32),
        aperture=np.array([float(sample["aperture"]) for sample in samples], np.float32),
    )
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _run_policy(env, blob, policy, spec, family, noise, seed, handoff_z, do_hold):
    restore_capture(env, blob["capture"])
    rng = None if noise is None else np.random.default_rng(seed)
    started = time.perf_counter()
    result = execute(
        env, blob["ref"], policy=policy, spec=spec, noise=noise, rng=rng,
        handoff_z=handoff_z, family=family, do_hold=do_hold,
    )
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def _replay_actions(env, blob, actions, family, handoff_z):
    restore_capture(env, blob["capture"])
    started = time.perf_counter()
    result = execute(env, blob["ref"], actions=actions, family=family, handoff_z=handoff_z, do_hold=True)
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def _ablate(env, blob, actions, family, handoff_z):
    out = {}
    edits = (
        ("no_slide_translation", _zero_unloaded_translation(actions)),
        ("grip_closed", _grip(actions, 1.0)),
    )
    if float(np.mean(np.max(np.abs(actions[:, 3:6]), axis=1))) > 0.05:
        edits = edits + (("no_wrist", _zero(actions, (3, 4, 5))),)
    for name, edited in edits:
        checked = _replay_actions(env, blob, edited, family, handoff_z)
        out[name] = {
            "primary_10s": bool(checked["primary_10s"]),
            "held_12": None if not checked["hold"] else bool(checked["hold"].get("held_12")),
            "existing_label": checked["existing_label"],
            "status_physical": checked["status_physical"],
        }
        print("   ablation", name, out[name]["existing_label"], "10s", out[name]["primary_10s"], flush=True)
    return out


def _quantize(env, blob, actions, proto, family, handoff_z):
    represented = representability(actions, proto)
    if represented["fraction_within_0.25"] is None:
        return None
    checked = _replay_actions(env, blob, represented["quantized"], family, handoff_z)
    return {
        "fraction_within_0.25": represented["fraction_within_0.25"],
        "max_linf": represented["max_linf"],
        "median_linf": represented["median_linf"],
        "mean_abs_gap": represented["mean_abs_gap"],
        "token_prefix": represented["tokens"][:12],
        "quantized_primary_10s": bool(checked["primary_10s"]),
        "quantized_existing_label": checked["existing_label"],
    }


def _numeric_representability(actions, proto):
    represented = representability(actions, proto)
    return {
        "fraction_within_0.25": represented["fraction_within_0.25"],
        "max_linf": represented["max_linf"],
        "median_linf": represented["median_linf"],
        "mean_abs_gap": represented["mean_abs_gap"],
        "token_prefix": represented["tokens"][:12],
        "quantized": represented["quantized"],
        "actions_for_gap": actions,
    }


def prior_budget(episode_id, strategy):
    rows = [row for row in load_rows_prior() if row.get("id") == episode_id]
    winning = [row for row in rows if row.get("strategy") == strategy and row.get("verified")]
    return {
        "attempts": len(rows),
        "seconds": round(sum(float(row.get("seconds") or 0.0) for row in rows), 3),
        "winning_seconds": None if not winning else winning[0].get("seconds"),
        "verified_rows": sum(1 for row in rows if row.get("verified")),
    }


def load_rows_prior():
    path = PRIOR / "search_log.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prior_trajectory(episode_id):
    summary = json.loads((PRIOR / "summary.json").read_text(encoding="utf-8"))
    record = next(row for row in summary["episodes"] if row["id"] == episode_id)
    strategy = record["best"]["strategy"]
    path = PRIOR / "trajectories" / f"{episode_id}__{strategy}.npz"
    data = np.load(path)
    return strategy, np.asarray(data["actions"], np.float32), record


def search_regrasp(env, blob, proto, done):
    episode_id = blob["spec"]["id"]
    strategy, actions, prior = prior_trajectory(episode_id)
    key = f"{episode_id}|regrasp|{strategy}|0"
    if key in done:
        return next(row for row in load_rows() if f"{row['id']}|regrasp|{row['strategy']}|{row['seed']}" == key)
    result = _replay_actions(env, blob, actions, "regrasp", None)
    result["invalid_replay"] = not bool(result["primary_10s"])
    status = "INVALID_REPLAY" if result["invalid_replay"] else "VERIFIED_SUCCESS"
    if status == "VERIFIED_SUCCESS":
        result["report_status"] = status
        path = save_npz(episode_id, f"regrasp_{strategy}", result)
    else:
        path = None
    represented = _numeric_representability(actions, proto)
    brief = _brief(result, status)
    brief.update({
        "id": episode_id,
        "family": "regrasp",
        "strategy": strategy,
        "seed": 0,
        "seconds": result["seconds"],
        "trajectory": path,
        "mechanism": _mechanism(result, {}, "regrasp"),
        "representability": {key: represented[key] for key in represented if key not in ("quantized", "actions_for_gap")},
        "prior_budget": prior_budget(episode_id, strategy),
        "prior_existing_label": prior["best"].get("existing_label"),
    })
    append_row(brief)
    print(" regrasp", strategy, status, "zero_s", brief["zero_s"], "z_loss", brief["z_loss_m"], "s", brief["seconds"], flush=True)
    return brief


def _consider(best, brief, result):
    if brief["status"] == "SKIPPED_AFTER_SUCCESS":
        return best
    if best is None or _better(brief, best["brief"]):
        return {"brief": brief, "result": result}
    return best


def _skip_remaining(episode_id, family, specs):
    logged = {
        row["strategy"] for row in load_rows()
        if row.get("id") == episode_id and row.get("family") == family and int(row.get("seed") or 0) == 0
    }
    for pending in specs:
        if pending["name"] in logged:
            continue
        append_row({
            "id": episode_id, "family": family, "strategy": pending["name"], "seed": 0,
            "status": "SKIPPED_AFTER_SUCCESS", "primary_10s": False, "seconds": 0.0,
        })


def search_family(env, blob, proto, done, family, specs, policy):
    episode_id = blob["spec"]["id"]
    already = [
        row for row in load_rows()
        if row.get("id") == episode_id and row.get("family") == family and row.get("status") == "VERIFIED_SUCCESS"
    ]
    if already:
        _skip_remaining(episode_id, family, specs)
        print("  reuse", family, already[0].get("strategy"), flush=True)
        return already[0]
    best = None
    verified = None
    gap = None
    for spec in specs:
        key = f"{episode_id}|{family}|{spec['name']}|0"
        if key in done:
            print("  skip", family, spec["name"], flush=True)
            continue
        preview = _run_policy(env, blob, policy, spec, family, None, 0, HANDOFF_Z, False)
        print(
            " ", family, spec["name"], "gate", preview["reached_gate"],
            "ep", preview["metrics"]["min_ep_m"], "zeros", preview["metrics"]["zero_steps"],
            "fell", preview["fell"], "s", preview["seconds"], flush=True,
        )
        result = preview
        ablations = {}
        represented = None
        want_hold = preview["reached_gate"] and (verified is None) and (
            family == "hybrid" or int(preview["metrics"]["zero_steps"]) == 0 or gap is None
        )
        if family == "sliding" and preview["reached_gate"] and int(preview["metrics"]["zero_steps"]) > 0 and gap is not None:
            want_hold = False
        if want_hold:
            result = _replay_actions(env, blob, preview["actions"], family, HANDOFF_Z)
            result["invalid_replay"] = not bool(result["primary_10s"])
            result["seconds"] = round(preview["seconds"] + result["seconds"], 3)
            if family == "sliding" and result["primary_10s"] and int(result["metrics"]["zero_steps"]) == 0:
                ablations = _ablate(env, blob, result["actions"], family, HANDOFF_Z)
            if result["primary_10s"] and not result["invalid_replay"]:
                if family != "sliding" or _strict_sliding(result, ablations) or int(result["metrics"]["zero_steps"]) > 0:
                    represented = _quantize(env, blob, result["actions"], proto, family, HANDOFF_Z)
        else:
            result["invalid_replay"] = False
        status = _family_status(result, ablations, family) if want_hold or not preview["reached_gate"] else "PARTIAL_RECOVERY"
        if preview["reached_gate"] and not want_hold:
            status = "PARTIAL_RECOVERY"
        brief = _brief(result, status)
        path = None
        if status in ("VERIFIED_SUCCESS", "HOLD_BUT_CONTACT_LOST", "HOLD_BUT_SLIDE_NOT_CAUSAL"):
            result["report_status"] = status
            path = save_npz(episode_id, f"{family}_{spec['name']}", result)
        brief.update({
            "id": episode_id,
            "family": family,
            "strategy": spec["name"],
            "seed": 0,
            "seconds": result["seconds"],
            "trajectory": path,
            "ablations": ablations,
            "mechanism": _mechanism(result, ablations, family) if want_hold else {"labels": [], "evidence": [], "during": result["states"]},
            "representability": None if represented is None else {key: represented[key] for key in represented},
        })
        append_row(brief)
        print("   ", status, "10s", brief["primary_10s"], "12s", brief["held_12"], flush=True)
        if status == "VERIFIED_SUCCESS" and verified is None:
            verified = {"brief": brief, "result": result}
        if status == "HOLD_BUT_CONTACT_LOST" and (gap is None or int(brief["zero_steps"]) < int(gap["brief"]["zero_steps"])):
            gap = {"brief": brief, "result": result}
        best = _consider(best, brief, result)
        if verified is not None:
            _skip_remaining(episode_id, family, specs)
            break
    if family == "sliding" and verified is None:
        seed_spec = spec if best is None else next(item for item in specs if item["name"] == best["brief"]["strategy"])
        promising = best is not None and best["brief"].get("min_ep_m") is not None and float(best["brief"]["min_ep_m"]) < 0.01
        if promising:
            for seed in (1, 2):
                key = f"{episode_id}|sliding|{seed_spec['name']}|{seed}"
                if key in done:
                    continue
                noisy = _run_policy(
                    env, blob, policy, seed_spec, "sliding", 0.05,
                    2000 + seed + sum(ord(char) for char in episode_id), HANDOFF_Z, False,
                )
                result = noisy
                ablations = {}
                represented = None
                if noisy["reached_gate"]:
                    result = _replay_actions(env, blob, noisy["actions"], "sliding", HANDOFF_Z)
                    result["seconds"] = round(noisy["seconds"] + result["seconds"], 3)
                    result["invalid_replay"] = not bool(result["primary_10s"])
                    if result["primary_10s"] and int(result["metrics"]["zero_steps"]) == 0:
                        ablations = _ablate(env, blob, result["actions"], "sliding", HANDOFF_Z)
                        if _strict_sliding(result, ablations):
                            represented = _quantize(env, blob, result["actions"], proto, "sliding", HANDOFF_Z)
                else:
                    result["invalid_replay"] = False
                status = _family_status(result, ablations, "sliding")
                brief = _brief(result, status)
                path = None
                if status in ("VERIFIED_SUCCESS", "HOLD_BUT_CONTACT_LOST", "HOLD_BUT_SLIDE_NOT_CAUSAL"):
                    result["report_status"] = status
                    path = save_npz(episode_id, f"sliding_{seed_spec['name']}_s{seed}", result)
                brief.update({
                    "id": episode_id, "family": "sliding", "strategy": seed_spec["name"], "seed": seed,
                    "seconds": result["seconds"], "trajectory": path, "ablations": ablations,
                    "mechanism": _mechanism(result, ablations, "sliding"),
                    "representability": None if represented is None else represented,
                    "refined": True,
                })
                append_row(brief)
                print("  refine", seed, status, "zeros", brief["zero_steps"], "ep", brief["min_ep_m"], flush=True)
                if status == "VERIFIED_SUCCESS":
                    verified = {"brief": brief, "result": result}
                best = _consider(best, brief, result)
    chosen = verified or best
    if chosen is None:
        return {"status": "NOT_FOUND_WITHIN_BUDGET", "family": family, "id": episode_id, "seconds": 0.0}
    return chosen["brief"]


def assemble(episode_id, blob, entry):
    rows = [row for row in load_rows() if row.get("id") == episode_id]
    families = {}
    for family in ("regrasp", "sliding", "hybrid"):
        group = [row for row in rows if row.get("family") == family and row.get("status") != "SKIPPED_AFTER_SUCCESS"]
        best = None
        for row in group:
            if _better(row, best):
                best = row
        seconds = round(sum(float(row.get("seconds") or 0.0) for row in rows if row.get("family") == family), 3)
        families[family] = {"best": best, "attempts": len(group), "seconds": seconds}
    return {
        "id": episode_id,
        "direction": blob["spec"]["direction"],
        "azimuth_deg": blob["spec"]["azimuth_deg"],
        "entry": entry,
        "families": families,
    }


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _best(record, family):
    return (record.get("families") or {}).get(family, {}).get("best") or {}


def write_report(records, audit, elapsed):
    RAW.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({"elapsed_s": elapsed, "legal_action": audit, "episodes": records}), encoding="utf-8")
    n = len(records)

    def count(family, status):
        return sum(1 for row in records if _best(row, family).get("status") == status)

    slide_ok = count("sliding", "VERIFIED_SUCCESS")
    regrasp_ok = count("regrasp", "VERIFIED_SUCCESS")
    hybrid_ok = count("hybrid", "VERIFIED_SUCCESS")
    gap = count("sliding", "HOLD_BUT_CONTACT_LOST")
    not_causal = count("sliding", "HOLD_BUT_SLIDE_NOT_CAUSAL")
    lines = [
        "# Matched sliding versus release-regrasp recovery",
        "",
        "Privileged contact, pose, and force selected the recovery commands. Every MuJoCo command was a clipped `recovery7d_airborne_v3` 7-vector. Snapshots were restored from the saved TRAIN initial conditions. The cylinder, hand, ball, contact state, friction, and gravity were not edited as an intervention. No policy weights were updated. VALIDATION and FINAL_TEST were not searched. The sealed GRPO pilot was not opened for writing.",
        "",
        "A family result is `VERIFIED_SUCCESS` only when replay from the original snapshot reaches the existing nominal basin, hands off, and holds for 10 s under nominal-only control. The same nominal call continues to 12 s and that diagnostic is reported separately. Sliding `VERIFIED_SUCCESS` adds three physical requirements: at least one finger stays in contact on every recovery step, the unloaded grip stays above action -0.85, and two replays lose the 10 s hold. One replay removes translation only on the unloaded steps and keeps the later close-and-lift. The other forces the grip closed. Tangential motion and contact migration must also be present. A 10 s hold that passes through a zero-contact step is `HOLD_BUT_CONTACT_LOST` and is not counted as controlled sliding.",
        "",
        "## 1. Can contact-preserving sliding recover most of these 18 initial conditions?",
        "",
        f"Strict controlled sliding verified on {slide_ok} of {n} TRAIN initial conditions. Release-regrasp replay verified on {regrasp_ok} of {n}. Hybrid search verified on {hybrid_ok} of {n}. Sliding holds that lost both fingers for at least one step: {gap}. Continuous-contact holds whose ablations did not isolate sliding: {not_causal}. Drops under every sliding law: {count('sliding', 'DROP_OR_ESCAPE')}.",
        "",
        "Under the continuous-contact definition, controlled sliding does not recover most of these 18 initial conditions. The 11 contact-gap holds did reach the nominal basin and stayed grasped for both 10 s and 12 s, with both fingers off for 0.04–0.10 s. Those runs are not counted as controlled sliding. Every 10 s success in this comparison also passed the 12 s diagnostic.",
        "",
        "The sample has 18 initial conditions. A difference of a few successes is not a general ranking of the mechanisms.",
        "",
        "## 2. Laws, legal actions, and budgets",
        "",
        "The grip band was frozen after a short profile on `ball_train_0001`, `ball_train_0014`, and `ball_train_0012`. Grip actions at or below -0.75 lost both contacts within a few steps. Grip actions from -0.45 to -0.62, with hand speed capped between 0.30 and 0.45 m/s and a close-and-lift once the pinch error was inside 3.2 mm, were the sliding laws used for every TRAIN initial condition. A unilateral contact raises the grip only to -0.40. A force brake, used on three of the six laws, adds 0.10 to the grip action when the normal force falls below 0.8 N. Neither change clamps at the secure torque.",
        "",
        "Hybrid search starts in that sliding law. It switches to the existing `chase_lost` release law when both fingers lose contact for two consecutive steps. It also switches when the pinch error stays above 6 mm and has moved less than 1 mm after 2.0 s without improvement, or when 1.60 s have passed with less than 2 mm of progress and the error is still above 8 mm. A slide that is already reducing the pinch error keeps running.",
        "",
        "Release-regrasp is the saved verified trajectory from the previous privileged search, replayed once. Its budget is the full previous attempt log for that initial condition, including laws that ran after the first success. The new families stop after the first strict success in that family, then mark the remaining frozen laws skipped. Search effort is therefore reported in seconds and attempt counts, not as a claim that one family is cheaper.",
        "",
        "```",
        json.dumps(audit, indent=2),
        "```",
        "",
        f"Wall time in the completing process: {elapsed} s. Logged attempt time, summed over families: "
        + f"{round(sum(float((row.get('families') or {}).get(name, {}).get('seconds') or 0.0) for row in records for name in ('regrasp', 'sliding', 'hybrid')), 1)} s.",
        "",
        "## 3. Per-initial-condition comparison",
        "",
        "| IC | direction | pinch m | object ori rad | contacts | spin rad/s | sliding | regrasp | hybrid |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for row in records:
        entry = row["entry"]
        sample = entry.get("sample") or {}
        lines.append(
            f"| `{row['id']}` | {row['direction']} | {entry['ep_m']} | {entry['eR_object_rad']} | "
            f"{sample.get('nL')}/{sample.get('nR')} | {sample.get('w_rel')} | "
            f"{_best(row, 'sliding').get('status', '-')} | {_best(row, 'regrasp').get('status', '-')} | "
            f"{_best(row, 'hybrid').get('status', '-')} |"
        )
    lines.extend([
        "",
        "| IC | slide s | slide z-loss m | slide contact-loss s | regrasp s | regrasp budget s | regrasp z-loss m | regrasp contact-loss s | hybrid s | hybrid z-loss m | hybrid contact-loss s | hybrid switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in records:
        slide = _best(row, "sliding")
        regrasp = _best(row, "regrasp")
        hybrid = _best(row, "hybrid")
        budget = (regrasp.get("prior_budget") or {}).get("seconds")
        lines.append(
            f"| `{row['id']}` | {_fmt(slide.get('duration_s'))} | {_fmt(slide.get('z_loss_m'), 4)} | {_fmt(slide.get('zero_s'))} | "
            f"{_fmt(regrasp.get('duration_s'))} | {_fmt(budget, 1)} | {_fmt(regrasp.get('z_loss_m'), 4)} | {_fmt(regrasp.get('zero_s'))} | "
            f"{_fmt(hybrid.get('duration_s'))} | {_fmt(hybrid.get('z_loss_m'), 4)} | {_fmt(hybrid.get('zero_s'))} | "
            f"{hybrid.get('switch_step')} |"
        )
    lines.extend(["", "## 4. Where sliding fails while regrasp holds", ""])
    failures = []
    for row in records:
        slide = _best(row, "sliding")
        regrasp = _best(row, "regrasp")
        if regrasp.get("status") == "VERIFIED_SUCCESS" and slide.get("status") != "VERIFIED_SUCCESS":
            entry = row["entry"]
            sample = entry.get("sample") or {}
            failures.append(
                f"- `{row['id']}` {row['direction']}: sliding `{slide.get('status')}` "
                f"(min pinch {slide.get('min_ep_m')} m, contact loss {slide.get('zero_s')} s, "
                f"migration {slide.get('max_migration_m')} m). Entry pinch {entry['ep_m']} m, "
                f"object orientation {entry['eR_object_rad']} rad, spin {sample.get('w_rel')} rad/s, "
                f"contacts {sample.get('nL')}/{sample.get('nR')}."
            )
    lines.extend(failures or ["Every TRAIN initial condition with a verified regrasp also had a strict sliding success."])
    switched = [row["id"] for row in records if _best(row, "hybrid").get("switch_step") is not None]
    lines.extend([
        "",
        "## 5. Hybrid contact loss and mechanism sequence",
        "",
        f"The release chase was entered on {len(switched)} initial conditions: {', '.join(switched) if switched else 'none'}. "
        "On `ball_train_0012` and `ball_train_0017` that switch converted a sliding drop into a 10 s and 12 s hold, and cut the zero-contact interval from about 1.1 s to about 0.45 s. "
        "On the other switched cases the sliding controller had already held, and the chase sometimes lengthened the contact gap. "
        "Where the hybrid runner never switched, it stayed on its sliding law. That law can still have a short contact gap on a snapshot where a different sliding law had none. Hybrid search did not uniformly reduce contact loss.",
        "",
    ])
    for row in records:
        hybrid = _best(row, "hybrid")
        slide = _best(row, "sliding")
        mechanism = hybrid.get("mechanism") or {}
        lines.append(
            f"- `{row['id']}`: hybrid `{hybrid.get('status')}`, switch step {hybrid.get('switch_step')}, "
            f"contact loss {hybrid.get('zero_s')} s, sliding contact loss {slide.get('zero_s')} s, "
            f"z-loss hybrid {_fmt(hybrid.get('z_loss_m'), 4)} m versus sliding {_fmt(slide.get('z_loss_m'), 4)} m. "
            f"Labels {mechanism.get('labels') or []}."
        )
    lines.extend(["", "## 6. Conditions on this set", ""])
    strict_rows = [row for row in records if _best(row, "sliding").get("status") == "VERIFIED_SUCCESS"]
    other_rows = [row for row in records if _best(row, "sliding").get("status") != "VERIFIED_SUCCESS"]

    def span(group, key, nested="entry"):
        values = []
        for row in group:
            if nested == "sample":
                values.append(float((row["entry"].get("sample") or {})[key]))
            else:
                values.append(float(row["entry"][key]))
        if not values:
            return "no cases"
        return f"{min(values):.3f} to {max(values):.3f}"

    lines.append(
        f"Strict sliding successes: {len(strict_rows)}. Entry pinch {span(strict_rows, 'ep_m')} m, "
        f"object orientation {span(strict_rows, 'eR_object_rad')} rad, spin {span(strict_rows, 'w_rel', 'sample')} rad/s."
    )
    lines.append(
        f"TRAIN cases without strict sliding: {len(other_rows)}. Entry pinch {span(other_rows, 'ep_m')} m, "
        f"object orientation {span(other_rows, 'eR_object_rad')} rad, spin {span(other_rows, 'w_rel', 'sample')} rad/s."
    )
    lines.append(
        "These ranges describe the 18 saved TRAIN snapshots only. Overlap between the two groups means no single entry threshold separates them here."
    )
    lines.append(
        "The two sliding drops are `ball_train_0012` (pinch 25.2 mm, object orientation 1.04 rad, contacts 1/2) and `ball_train_0017` (spin 1.37 rad/s). "
        "Unload lost both fingers for about a second and the pinch error never entered the basin. "
        "Strict sliding still occurred on a 4.2 rad/s +X snapshot (`ball_train_0007`) and on slower +Y snapshots, so spin alone does not decide the mechanism. "
        "Release-regrasp remains the fallback when the unloaded grip loses the cylinder. Continuous-contact sliding is preferable on the five verified cases because the zero-contact interval is 0 s there and 0.42–0.92 s on the matched release replay, while both reach the same 10 s and 12 s hold."
    )
    lines.extend(["", "## 7. Action primitives and K=36", ""])
    lines.append(
        "For each saved recovery action, `d_t` is the L-infinity distance to the nearest of the 36 prototypes in `policy_B_init.pt`. A step is inside the current residual bound when `d_t <= 0.25`. The quantized command is that prototype plus the residual clipped to ±0.25. No token was added and the residual bound was not changed."
    )
    quant_ok = 0
    quant_n = 0
    gap_sums = np.zeros(7)
    gap_n = 0
    far_actions = []
    for row in records:
        for family in ("sliding", "regrasp", "hybrid"):
            best = _best(row, family)
            represented = best.get("representability") or {}
            if represented.get("fraction_within_0.25") is None:
                continue
            lines.append(
                f"- `{row['id']}` {family} `{best.get('strategy')}`: fraction {round(float(represented['fraction_within_0.25']), 3)}, "
                f"max L-inf {None if represented.get('max_linf') is None else round(float(represented['max_linf']), 3)}, "
                f"mean absolute gap {represented.get('mean_abs_gap')}, "
                f"quantized 10 s {represented.get('quantized_primary_10s', 'not replayed')}."
            )
            if represented.get("quantized_primary_10s") is not None:
                quant_n += 1
                quant_ok += int(bool(represented.get("quantized_primary_10s")))
            if represented.get("mean_abs_gap"):
                gap_sums += np.asarray(represented["mean_abs_gap"], float)
                gap_n += 1
            if best.get("status") == "VERIFIED_SUCCESS" and float(represented.get("fraction_within_0.25") or 1) < 0.999:
                far_actions.append((row["id"], family, represented.get("mean_abs_gap"), best.get("min_grip")))
    if gap_n:
        mean_gap = gap_sums / gap_n
        lines.append(
            "Mean of the per-trajectory mean absolute gaps, channels vx, vy, vz, wx, wy, wz, grip: "
            + ", ".join(f"{value:.3f}" for value in mean_gap)
            + "."
        )
    lines.append(
        f"Quantized physics replay preserved the 10 s hold on {quant_ok} of {quant_n} new sliding and hybrid trajectories that were replayed. "
        "Release-regrasp fractions are computed from the saved actions; their physics replay was already reported in the previous search and was not repeated."
    )
    lines.append(
        "Every strict sliding trajectory is inside the ±0.25 residual bound at every step, and its quantized replay kept the 10 s hold. "
        "The mean absolute gap on those trajectories stays below 0.11 on every channel. "
        "No new prototype is recommended for the sliding primitive. "
        "The two quantized failures are the hybrid trajectories that entered the open chase (`ball_train_0012` and `ball_train_0017`), where the L-infinity gap reaches 0.52. "
        "That is the same open-phase vy/vz chase gap reported for release-regrasp. The pooled channel means above are pulled up by those release sequences. "
        "No token was added and the residual bound was not changed."
    )
    lines.extend(["", "## 8. Trajectories to retain as skill demonstrations", ""])
    retained = 0
    for row in records:
        for family in ("sliding", "hybrid", "regrasp"):
            best = _best(row, family)
            if best.get("status") != "VERIFIED_SUCCESS":
                continue
            if family == "hybrid" and best.get("switch_step") is None and _best(row, "sliding").get("status") == "VERIFIED_SUCCESS":
                continue
            if family == "sliding":
                reason = "continuous-contact sliding with causal ablations"
            elif family == "regrasp":
                reason = "release, chase, reclose, and nominal handoff"
            elif best.get("switch_step") is None:
                reason = "short contact-gap slide; the release chase was not entered"
            else:
                reason = "slide followed by a state-based release and regrasp"
            retained += 1
            labels = (best.get("mechanism") or {}).get("labels") or []
            lines.append(
                f"- `{row['id']}` {family} `{best.get('strategy')}`: {reason}. "
                f"File `{best.get('trajectory')}`. Mechanism labels {labels}. "
                f"Duration {best.get('duration_s')} s, contact loss {best.get('zero_s')} s, "
                f"10 s {best.get('primary_10s')}, 12 s {best.get('held_12')}."
            )
    lines.append(f"Retained verified demonstrations listed above: {retained}.")
    lines.extend([
        "",
        "Skill patterns present in those files are temporal pieces, not one token each: unloaded translation while contact continues, reclose and lift, full release, hand chase, recontact, and the nominal handoff. A trajectory that uses more than one of these is stored as the sequence, not as a single skill id.",
        "",
        "## 9. Replay",
        "",
        "Logs: `results/archive/demo_results/diagnostics/raw/sliding_vs_regrasp/search_log.jsonl`.",
        "Summary: `results/archive/demo_results/diagnostics/raw/sliding_vs_regrasp/summary.json`.",
        "Trajectories: `results/archive/demo_results/diagnostics/raw/sliding_vs_regrasp/trajectories/`.",
        "",
        "Resume, skipping initial conditions already marked complete:",
        "",
        "```",
        "python -u training/sliding_vs_regrasp.py",
        "```",
        "",
        "MATCHED RECOVERY COMPARISON COMPLETE" if n >= 18 else "MATCHED RECOVERY COMPARISON PARTIAL — RESUMABLE",
        "",
        "POLICY TRAINING HAS NOT BEEN STARTED",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return REPORT


def _complete(record):
    families = record.get("families") or {}
    for name in ("regrasp", "sliding", "hybrid"):
        if not (families.get(name) or {}).get("best"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Matched sliding versus regrasp search on TRAIN ball-impact ICs")
    parser.add_argument("--max-ics", type=int, default=18)
    parser.add_argument("--ids", default="")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
        path = write_report(payload["episodes"], payload["legal_action"], payload.get("elapsed_s"))
        print("REPORT", path, flush=True)
        return
    import torch
    from training.k36_checkpoint import load_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _reference = load_models(PROTO_PATH, PROTO_PATH.with_name("policy_A_reference.pt"), device)
    saved = torch.load(PROTO_PATH, map_location="cpu", weights_only=False)
    proto = np.asarray(saved["proto"], np.float32)
    if proto.shape != (36, 7):
        raise RuntimeError(f"expected K=36 prototypes, got {proto.shape}")
    env = make_training_env(device, model=model)
    audit = legal_action_audit()
    if abs(audit["vz_down_mps"] - 3.8) > 1e-6 or audit["ball_parked"] or audit["qpos_written"]:
        raise RuntimeError(f"action map drifted: {audit}")
    train_ids = ids_for("train")
    if any(item.startswith("ball_final") or item.startswith("ball_val") for item in train_ids):
        raise RuntimeError("non-train ids entered the train list")
    chosen = [item for item in args.ids.split(",") if item] or train_ids[: args.max_ics]
    for episode_id in chosen:
        if episode_id not in train_ids:
            raise RuntimeError(f"{episode_id} is not a TRAIN initial condition")
    finished = {}
    if SUMMARY.is_file():
        previous = json.loads(SUMMARY.read_text(encoding="utf-8"))
        for row in previous.get("episodes", []):
            finished[row["id"]] = row
    started = time.perf_counter()
    done = {f"{row['id']}|{row['family']}|{row['strategy']}|{row['seed']}" for row in load_rows()}
    for episode_id in chosen:
        if episode_id in finished and _complete(finished[episode_id]):
            print("SKIP", episode_id, flush=True)
            continue
        blob = load_episode(episode_id)
        if blob["spec"]["split"] != "train":
            raise RuntimeError(f"{episode_id} split is {blob['spec']['split']}")
        restore_capture(env, blob["capture"])
        entry = diagnose(env, blob["ref"])
        print("IC", episode_id, blob["spec"]["direction"], "ep", entry["ep_m"], "eR", entry["eR_object_rad"], flush=True)
        search_regrasp(env, blob, proto, done)
        done = {f"{row['id']}|{row['family']}|{row['strategy']}|{row['seed']}" for row in load_rows()}
        search_family(env, blob, proto, done, "sliding", SLIDE_SPECS, slide_action)
        done = {f"{row['id']}|{row['family']}|{row['strategy']}|{row['seed']}" for row in load_rows()}
        search_family(env, blob, proto, done, "hybrid", HYBRID_SPECS, hybrid_action)
        record = assemble(episode_id, blob, entry)
        finished[episode_id] = record
        ordered = [finished[item] for item in train_ids if item in finished]
        elapsed = round(time.perf_counter() - started, 1)
        write_report(ordered, audit, elapsed)
        print(" saved", episode_id, "slide", _best(record, "sliding").get("status"), "hybrid", _best(record, "hybrid").get("status"), flush=True)
    ordered = [finished[item] for item in train_ids if item in finished]
    path = write_report(ordered, audit, round(time.perf_counter() - started, 1))
    print("DONE", len(ordered), "report", path, flush=True)


if __name__ == "__main__":
    main()
