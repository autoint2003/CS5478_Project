"""Physical audit of Checkpoint 3 recovery on a Sliding-success / Recatch-fail IC.

Writes only under results/diagnostics/checkpoint3_skill_audit. Does not change
physics, the nominal controller, the codebook, observations, rewards, or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import hand_pose
from data_generation.strategies.vertical_recatch_search import _finish as vertical_finish
from data_generation.strategies.vertical_recatch_search import make_env as make_vertical_env
from data_generation.strategies.vertical_recatch_search import rollout as vertical_rollout
from data_generation.strategies.vertical_recatch_search import specs as vertical_specs
from data_generation.strategies.vertical_recatch_search import _summary as vertical_summary
from envs.outcomes.horizon_outcome import NOMINAL_S, finish_recovery, recovery_failed
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, terminal_reason
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results" / "diagnostics" / "checkpoint3_skill_audit"
CHECKPOINT = ROOT / "results" / "grpo" / "tree80" / "checkpoint_epoch_0003.pt"
DEFAULT_IC = "k48r_valid_00009"
SAMPLE_S = 0.04
BOTH_OFF_HOLD_S = 0.04  # two policy steps; single-frame gaps are not Recatch


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def find_slide_row(episode_id):
    for path in (
        ROOT / "data" / "controlled_sliding" / "manifest.jsonl",
        ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl",
        ROOT / "data" / "inward_sliding" / "manifest.jsonl",
    ):
        for row in load_jsonl(path):
            if row["id"] != episode_id:
                continue
            if row.get("reason") != "SUCCESS_12S":
                continue
            if row.get("strategy") == "controlled_slide":
                return row, path
    for path in (
        ROOT / "data" / "inward_sliding" / "manifest.jsonl",
        ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl",
    ):
        for row in load_jsonl(path):
            if row["id"] == episode_id and row.get("reason") == "SUCCESS_12S":
                return row, path
    raise RuntimeError(f"no SUCCESS_12S sliding tape for {episode_id}")


def find_vertical_fail(episode_id):
    hits = []
    budget = ROOT / "data" / "staging" / "recovery_expansion" / "budget_attempts.jsonl"
    if budget.is_file():
        for row in load_jsonl(budget):
            if row.get("id") == episode_id and row.get("strategy") == "vertical_recatch" and row.get("success") is False:
                hits.append({"source": str(budget.relative_to(ROOT)).replace("\\", "/"), **row})
    follow = ROOT / "data" / "staging" / "recovery_expansion" / "followup_attempts.jsonl"
    if follow.is_file():
        for row in load_jsonl(follow):
            if row.get("id") == episode_id and row.get("stage") == "vertical_tail_attempt" and row.get("success") is False:
                hits.append({"source": str(follow.relative_to(ROOT)).replace("\\", "/"), **row})
    return hits


def finger_contacts(sim):
    model, data, ids = sim.model, sim.data, sim.ids
    hand = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    origin = np.array(data.xpos[ids.hand_body], float)
    obj = int(ids.object_body)
    out = {"L": [], "R": []}
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
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        pos_h = hand.T @ (np.array(contact.pos, float) - origin)
        out[side].append({
            "pos_h": pos_h,
            "fn": float(wrench[0]),
            "ft": float(np.linalg.norm(wrench[1:3])),
            "pos_w": np.array(contact.pos, float).copy(),
        })
    return out


def sample_state(sim, ref, action=None, token=None, residual=None, phase="recovery"):
    pack = physical_pack(sim)
    pinch, _rel = hand_pose(sim)
    touch = finger_contacts(sim)

    def side(name):
        rows = touch[name]
        if not rows:
            return 0, 0.0, 0.0, np.full(3, np.nan)
        return (
            len(rows),
            float(sum(row["fn"] for row in rows)),
            float(sum(row["ft"] for row in rows)),
            np.mean([row["pos_h"] for row in rows], axis=0),
        )

    n_l, fn_l, ft_l, c_l = side("L")
    n_r, fn_r, ft_r, c_r = side("R")
    action = np.zeros(7) if action is None else np.asarray(action, float)
    residual = np.full(7, np.nan) if residual is None else np.asarray(residual, float)
    R_rel = np.asarray(pack["R_rel"], float)
    rot_angle = float(np.arccos(np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0)))
    return {
        "phase": phase,
        "t": float(sim.data.time),
        "token": None if token is None else int(token),
        "residual": residual,
        "action": action,
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "nL_raw": int(n_l),
        "nR_raw": int(n_r),
        "fnL": fn_l,
        "fnR": fn_r,
        "ftL": ft_l,
        "ftR": ft_r,
        "cL": c_l,
        "cR": c_r,
        "rh": np.asarray(pack["rh"], float).copy(),
        "v_rel_h": np.asarray(pack["v_rel_h"], float).copy(),
        "w_rel_h": np.asarray(pack["w_rel_h"], float).copy(),
        "obj_z": float(pack["obj_z"]),
        "aperture": float(pack["aperture"]),
        "tau": float(pack["tau"]),
        "pinch": np.asarray(pinch, float).copy(),
        "rot_angle": rot_angle,
        "ctrl_phase": str(sim.fsm.phase),
        "bilateral": int(pack["nL"]) > 0 and int(pack["nR"]) > 0,
        "both_off": int(pack["nL"]) == 0 and int(pack["nR"]) == 0,
    }


def flatten_row(row, t0):
    residual = row["residual"]
    action = row["action"]
    rh = row["rh"]
    v = row["v_rel_h"]
    w = row["w_rel_h"]
    c_l = row["cL"]
    c_r = row["cR"]
    pinch = row["pinch"]
    return {
        "phase": row["phase"],
        "t_abs": row["t"],
        "t_rel": row["t"] - t0,
        "token": "" if row["token"] is None else row["token"],
        "res0": residual[0], "res1": residual[1], "res2": residual[2],
        "res3": residual[3], "res4": residual[4], "res5": residual[5], "res6": residual[6],
        "act0": action[0], "act1": action[1], "act2": action[2],
        "act3": action[3], "act4": action[4], "act5": action[5], "act6": action[6],
        "nL": row["nL"], "nR": row["nR"], "bilateral": int(row["bilateral"]), "both_off": int(row["both_off"]),
        "fnL": row["fnL"], "fnR": row["fnR"], "ftL": row["ftL"], "ftR": row["ftR"],
        "cLx": c_l[0], "cLy": c_l[1], "cLz": c_l[2],
        "cRx": c_r[0], "cRy": c_r[1], "cRz": c_r[2],
        "rhx": rh[0], "rhy": rh[1], "rhz": rh[2],
        "vx": v[0], "vy": v[1], "vz": v[2],
        "wx": w[0], "wy": w[1], "wz": w[2],
        "obj_z": row["obj_z"], "aperture": row["aperture"], "tau": row["tau"],
        "pinching_x": pinch[0], "rot_angle": row["rot_angle"], "ctrl_phase": row["ctrl_phase"],
    }


def write_csv(path, rows, t0):
    flat = [flatten_row(row, t0) for row in rows]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def contact_events(rows, t0):
    events = []
    prev_l = rows[0]["nL"] > 0
    prev_r = rows[0]["nR"] > 0
    prev_both = rows[0]["both_off"]
    both_off_start = None
    for row in rows:
        left = row["nL"] > 0
        right = row["nR"] > 0
        t_rel = row["t"] - t0
        if left != prev_l:
            events.append({"t_rel": t_rel, "event": "left_on" if left else "left_off", "nL": row["nL"], "nR": row["nR"], "phase": row["phase"]})
        if right != prev_r:
            events.append({"t_rel": t_rel, "event": "right_on" if right else "right_off", "nL": row["nL"], "nR": row["nR"], "phase": row["phase"]})
        if row["both_off"] and not prev_both:
            both_off_start = t_rel
            events.append({"t_rel": t_rel, "event": "both_off_start", "nL": 0, "nR": 0, "phase": row["phase"]})
        if (not row["both_off"]) and prev_both and both_off_start is not None:
            events.append({
                "t_rel": t_rel, "event": "both_off_end",
                "duration_s": t_rel - both_off_start,
                "nL": row["nL"], "nR": row["nR"], "phase": row["phase"],
            })
            both_off_start = None
        prev_l, prev_r, prev_both = left, right, row["both_off"]
    return events


def classify(rows, t0, handoff_t):
    recovery = [row for row in rows if row["phase"] == "recovery" and row["t"] <= handoff_t + 1e-9]
    if not recovery:
        return {"label": "Unclear", "why": ["no recovery samples"]}
    both_off_intervals = []
    start = None
    for row in recovery:
        if row["both_off"] and start is None:
            start = row["t"]
        if (not row["both_off"]) and start is not None:
            both_off_intervals.append((start - t0, row["t"] - t0, row["t"] - start))
            start = None
    if start is not None:
        both_off_intervals.append((start - t0, recovery[-1]["t"] - t0, recovery[-1]["t"] - start))
    sustained = [item for item in both_off_intervals if item[2] >= BOTH_OFF_HOLD_S - 1e-9]
    bilateral_steps = sum(1 for row in recovery if row["bilateral"])
    any_contact_steps = sum(1 for row in recovery if row["nL"] > 0 or row["nR"] > 0)
    migration = 0.0
    contact_with_pos = [row for row in recovery if row["bilateral"] and np.isfinite(row["cL"]).all() and np.isfinite(row["cR"]).all()]
    if len(contact_with_pos) >= 2:
        mid = np.array([(row["cL"] + row["cR"]) * 0.5 for row in contact_with_pos])
        migration = float(np.max(np.linalg.norm(mid - mid[0], axis=1)))
    rh = np.array([row["rh"] for row in recovery])
    rh_travel = float(np.max(np.linalg.norm(rh - rh[0], axis=1))) if len(rh) else 0.0
    facts = {
        "recovery_steps": len(recovery),
        "bilateral_steps": bilateral_steps,
        "any_contact_steps": any_contact_steps,
        "both_off_intervals": [
            {"t_start": a, "t_end": b, "duration_s": c} for a, b, c in both_off_intervals
        ],
        "sustained_both_off_intervals": [
            {"t_start": a, "t_end": b, "duration_s": c} for a, b, c in sustained
        ],
        "max_contact_migration_m": migration,
        "max_object_travel_in_hand_m": rh_travel,
        "max_aperture": float(max(row["aperture"] for row in recovery)),
        "min_tau": float(min(row["tau"] for row in recovery)),
    }
    why = []
    label = "Unclear"
    if sustained:
        last_gap_end = sustained[-1][1]
        recontact = any(row["bilateral"] and (row["t"] - t0) >= last_gap_end - 1e-9 for row in recovery)
        if recontact:
            label = "Recatch"
            why.append(f"sustained both-off lasting >= {BOTH_OFF_HOLD_S:.2f}s then bilateral recontact before handoff")
        else:
            why.append("sustained both-off occurred but bilateral grasp was not clearly re-established before handoff")
    elif bilateral_steps == len(recovery) and migration >= 0.001:
        label = "Sliding"
        why.append("bilateral contact on every recovery sample with contact-point migration >= 1 mm")
    elif any_contact_steps == len(recovery) and migration >= 0.001:
        label = "Sliding"
        why.append("at least one finger contact on every recovery sample with contact-point migration >= 1 mm")
    elif any_contact_steps == len(recovery) and rh_travel >= 0.001:
        label = "Sliding"
        why.append("contact preserved with object motion relative to the hand >= 1 mm")
    elif both_off_intervals and not sustained:
        why.append("only sub-threshold both-off gaps; not counted as Recatch")
    else:
        why.append("contact pattern does not uniquely match Sliding or Recatch thresholds")
    if label == "Recatch" and sustained:
        gap_end = sustained[-1][1]
        after = [row for row in recovery if (row["t"] - t0) >= gap_end - 1e-9 and row["bilateral"]]
        after_migration = 0.0
        if len(after) >= 2 and all(np.isfinite(row["cL"]).all() and np.isfinite(row["cR"]).all() for row in after):
            mid = np.array([(row["cL"] + row["cR"]) * 0.5 for row in after])
            after_migration = float(np.max(np.linalg.norm(mid - mid[0], axis=1)))
        if after_migration >= 0.001 or (len(after) >= 10 and rh_travel >= 0.001):
            label = "Hybrid"
            why.append(
                "after recontact, bilateral contact continues with in-hand adjustment "
                f"(post-gap migration {after_migration:.4f} m)"
            )
            facts["post_recontact_migration_m"] = after_migration
    return {"label": label, "why": why, "facts": facts}


def roll_policy_traced(env, model, book, blob, device, limit_s):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    rows = [sample_state(env.sim, blob["ref"], phase="recovery")]
    lost_low = 0
    how = None
    cap = int(round(float(limit_s) / 0.02))
    for _ in range(cap):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        residual = mean[-1].detach().float().cpu().numpy()
        if token == HANDOFF:
            how = "early"
            rows.append(sample_state(env.sim, blob["ref"], action=np.zeros(7), token=token, residual=residual, phase="handoff"))
            break
        _res, action = decode_action(book, token, residual)
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        rows.append(sample_state(env.sim, blob["ref"], action=executed, token=token, residual=residual, phase="recovery"))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return rows, t0, {
                "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                "success": False,
                "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "nominal_elapsed": 0.0,
            }, None
    if how is None:
        how = "forced"
    handoff_t = float(env.sim.data.time)
    out = finish_recovery(env, blob["ref"], t0, how)
    # finish_recovery already ran nominal; re-record nominal frames by a second restore is needed for CSV.
    return rows, t0, out, handoff_t


def continue_nominal_traced(env, blob, t0, how, rows_so_far, handoff_t):
    """Restore is assumed already at handoff-ready state after policy loop; call finish path with sampling."""
    # Re-run from a fresh policy handoff state is awkward after finish_recovery mutated the sim.
    # Caller re-runs policy to handoff then this function.
    apply_handoff(env.sim, blob["ref"])
    t_hand = float(env.sim.data.time)
    rows = list(rows_so_far)
    next_log = t_hand
    while float(env.sim.data.time) - t_hand < NOMINAL_S - 1e-9:
        nominal_step(env.sim)
        now = float(env.sim.data.time)
        if now + 1e-9 >= next_log:
            rows.append(sample_state(env.sim, blob["ref"], phase="nominal"))
            next_log += 0.02
        if latched(env.sim):
            break
        pack = physical_pack(env.sim)
        if float(pack["obj_z"]) < 0.30:
            break
    out = finish_recovery.__wrapped__ if False else None
    # Label with an independent finish on a third restore elsewhere; here return samples only.
    return rows, t_hand


def roll_policy_full(env, model, book, blob, device, limit_s):
    """Deterministic policy recovery + 12 s nominal, with full traces. Matches eval action rule."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    rows = [sample_state(env.sim, blob["ref"], phase="recovery")]
    lost_low = 0
    how = None
    cap = int(round(float(limit_s) / 0.02))
    for _ in range(cap):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        residual = mean[-1].detach().float().cpu().numpy()
        if token == HANDOFF:
            how = "early"
            rows.append(sample_state(env.sim, blob["ref"], action=np.zeros(7), token=token, residual=residual, phase="handoff"))
            break
        _res, action = decode_action(book, token, residual)
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        rows.append(sample_state(env.sim, blob["ref"], action=executed, token=token, residual=residual, phase="recovery"))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            reason = terminal_reason(env.sim) or "DROP_DURING_RECOVERY"
            return rows, t0, {
                "reason": reason, "success": False, "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0, "nominal_elapsed": 0.0,
                "lift_completed": False,
            }, float(env.sim.data.time)
    if how is None:
        how = "forced"
    handoff_t = float(env.sim.data.time)
    # Nominal hold with sampling, using the same verify loop as horizon_outcome.nominal_verify.
    from envs.outcomes.horizon_outcome import LOST_STEPS, _bilateral, _dropped, _monitor
    from envs.ballistic_impact import z_tgt_of
    from training.recovery_execution import _escaped
    from training.reward_v2 import NominalHoldTrace

    handoff = apply_handoff(env.sim, blob["ref"])
    if abs(float(handoff["time_delta"])) > 1e-9 or abs(float(handoff["qpos_max_abs"])) > 1e-12:
        raise RuntimeError("handoff edited the physical state")
    t_hand = float(env.sim.data.time)
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    hold_trace = NominalHoldTrace(NOMINAL_S)
    if latched(env.sim):
        reason = "TABLE_DROP"
    next_sample = t_hand
    while reason is None and float(env.sim.data.time) - t_hand < NOMINAL_S - 1e-9:
        t_prev = float(env.sim.data.time)
        nominal_step(env.sim)
        row = _monitor(env.sim)
        end_row = row
        target = float(z_tgt_of(env.sim))
        bilateral = _bilateral(row)
        command_reached = float(env.sim.fsm.p_des[2]) >= target - 1e-9
        dropped = _dropped(row) or _escaped(env.sim)
        hold_trace.observe(float(env.sim.data.time) - t_prev, bilateral, dropped, command_reached)
        if bilateral and command_reached:
            lift_done = True
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        if dropped:
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not bilateral else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
        now = float(env.sim.data.time)
        if now + 1e-9 >= next_sample:
            rows.append(sample_state(env.sim, blob["ref"], phase="nominal"))
            next_sample += 0.02
    nominal_elapsed = float(env.sim.data.time) - t_hand
    recovery_elapsed = t_hand - t0
    if reason is None:
        if nominal_elapsed < NOMINAL_S - 1e-6:
            reason = "PHYSICAL_VERIFICATION_FAILED"
        elif end_row is None or not _bilateral(end_row) or _dropped(end_row) or _escaped(env.sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    out = {
        "reason": reason,
        "success": reason == "SUCCESS_12S",
        "handoff": how,
        "recovery_elapsed": recovery_elapsed,
        "nominal_elapsed": nominal_elapsed,
        "lift_completed": lift_done,
    }
    return rows, t0, out, handoff_t


def replay_slide_expert(env, blob, row):
    with np.load(ROOT / row["continuous_npz"]) as archive:
        actions = np.asarray(archive["actions"], np.float32)
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    rows = [sample_state(env.sim, blob["ref"], phase="recovery")]
    for action in actions:
        executed = step_legal(env.sim, action)
        rows.append(sample_state(env.sim, blob["ref"], action=executed, phase="recovery"))
    handoff_t = float(env.sim.data.time)
    out = finish_recovery(env, blob["ref"], t0, "recorded_end")
    return rows, t0, out, handoff_t


def replay_vertical_oracle(blob, book):
    env = make_vertical_env()
    spec = next(item for item in vertical_specs() if item["name"] == "follow_early")
    result = vertical_rollout(env, blob, book, spec)
    summary = vertical_summary(result)
    samples = []
    for row in result.get("rows") or []:
        cmd = np.asarray(row.get("cmd", np.zeros(3)), float)
        action = np.zeros(7)
        action[:3] = cmd
        samples.append({
            "phase": "recovery",
            "t": float(row.get("t", 0.0)),
            "token": None,
            "residual": np.full(7, np.nan),
            "action": action,
            "nL": int(row.get("nL", 0)),
            "nR": int(row.get("nR", 0)),
            "nL_raw": int(row.get("nL", 0)),
            "nR_raw": int(row.get("nR", 0)),
            "fnL": 0.0,
            "fnR": 0.0,
            "ftL": 0.0,
            "ftR": 0.0,
            "cL": np.full(3, np.nan),
            "cR": np.full(3, np.nan),
            "rh": np.zeros(3),
            "v_rel_h": np.asarray(row.get("v_rel", np.zeros(3)), float).reshape(-1)[:3],
            "w_rel_h": np.zeros(3),
            "obj_z": float(row.get("z", 0.0) or 0.0),
            "aperture": float(row.get("gap", 0.0) or 0.0),
            "tau": 0.0,
            "pinch": np.zeros(3),
            "rot_angle": float("nan"),
            "ctrl_phase": str(row.get("mode", "oracle")),
            "bilateral": int(row.get("nL", 0)) > 0 and int(row.get("nR", 0)) > 0,
            "both_off": int(row.get("nL", 0)) == 0 and int(row.get("nR", 0)) == 0,
        })
    out = {
        "reason": result.get("mode"),
        "success": False,
        "summary": summary,
        "max_linf": result.get("max_linf"),
    }
    if result.get("mode") == "handoff":
        finish, extra = vertical_finish(env.sim, blob["ref"])
        out = {
            "reason": finish.get("reason") if not finish.get("success") else (
                finish.get("reason") if extra == "HELD_EXTENSION" else extra
            ),
            "success": bool(finish.get("success") and extra == "HELD_EXTENSION"),
            "recovery_elapsed": finish.get("recovery_elapsed"),
            "nominal_elapsed": finish.get("nominal_elapsed"),
            "extension": extra,
            "handoff": finish.get("handoff"),
            "summary": summary,
            "max_linf": result.get("max_linf"),
        }
    return samples, out, result


def render_video(env, model, book, blob, device, limit_s, dest):
    cam = mjv_camera_from_preset({
        "lookat": np.array(env.sim.data.xpos[env.sim.ids.object_body], float),
        "distance": 0.55, "azimuth": 132.0, "elevation": -22.0,
    })
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    frames = []

    def grab(text):
        cam.lookat[:] = np.array(env.sim.data.xpos[env.sim.ids.object_body], float)
        renderer.update_scene(env.sim.data, cam)
        frames.append(caption(np.asarray(renderer.render(), np.uint8).copy(), text))

    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    next_frame = t0
    lost_low = 0
    how = None
    cap = int(round(float(limit_s) / 0.02))
    for _ in range(cap):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        pack = physical_pack(env.sim)
        now = float(env.sim.data.time)
        if now + 1e-9 >= next_frame:
            grab(f"ckpt3 recover {now - t0:.2f}s tok={token} n={int(pack['nL'])}/{int(pack['nR'])}")
            next_frame += SAMPLE_S
        if token == HANDOFF:
            how = "early"
            break
        _res, action = decode_action(book, token, mean[-1].detach().float().cpu().numpy())
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            grab(f"ckpt3 FAIL {terminal_reason(env.sim)}")
            renderer.close()
            write_frames_mp4(frames, dest, fps=25)
            shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
            return
    if how is None:
        how = "forced"
    apply_handoff(env.sim, blob["ref"])
    t_hand = float(env.sim.data.time)
    next_frame = t_hand
    while float(env.sim.data.time) - t_hand < NOMINAL_S - 1e-9:
        nominal_step(env.sim)
        now = float(env.sim.data.time)
        if now + 1e-9 >= next_frame:
            grab(f"ckpt3 hold {now - t_hand:.2f}/12s")
            next_frame += SAMPLE_S
        if latched(env.sim) or float(physical_pack(env.sim)["obj_z"]) < 0.30:
            break
    grab("ckpt3 end")
    renderer.close()
    write_frames_mp4(frames, dest, fps=25)
    shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)


def evidence_bundle(episode_id):
    slide_row, slide_manifest = find_slide_row(episode_id)
    fails = find_vertical_fail(episode_id)
    in_vertical = any(row["id"] == episode_id for row in load_jsonl(ROOT / "data" / "vertical_recatch" / "manifest.jsonl"))
    return {
        "episode_id": episode_id,
        "sliding_success": {
            "manifest": str(slide_manifest.relative_to(ROOT)).replace("\\", "/"),
            "row": {
                "id": slide_row["id"],
                "strategy": slide_row.get("strategy"),
                "mechanism": slide_row.get("mechanism"),
                "reason": slide_row.get("reason"),
                "success": slide_row.get("success"),
                "verified_success": slide_row.get("verified_success"),
                "snapshot": slide_row.get("snapshot"),
                "continuous_npz": slide_row.get("continuous_npz"),
                "token_residual_npz": slide_row.get("token_residual_npz"),
                "recovery_elapsed": slide_row.get("recovery_elapsed"),
                "nominal_elapsed": slide_row.get("nominal_elapsed"),
                "lift_completed": slide_row.get("lift_completed"),
                "split": slide_row.get("split"),
            },
        },
        "vertical_recatch_failures": fails,
        "in_vertical_recatch_manifest": in_vertical,
        "gap": {
            "failed_vertical_trajectory_npz": (
                "not saved; only attempt log rows exist under data/staging/recovery_expansion/"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=DEFAULT_IC)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-s", type=float, default=SELECTED_LIMIT_S)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    evidence = evidence_bundle(args.id)
    if not evidence["vertical_recatch_failures"]:
        raise RuntimeError(f"{args.id} has no recorded vertical_recatch failure")
    if evidence["in_vertical_recatch_manifest"]:
        raise RuntimeError(f"{args.id} is already a successful vertical_recatch demo; refuse")

    slide_row = find_slide_row(args.id)[0]
    with (ROOT / slide_row["snapshot"]).open("rb") as handle:
        blob = pickle.load(handle)
    device = torch.device(args.device)
    model, book = load_trained_k48(ROOT / args.checkpoint, device)
    model.eval()
    codebook = load_codebook()
    if not np.array_equal(book, codebook):
        raise RuntimeError("checkpoint codebook differs from configured codebook")
    env = make_training_env(device, model=model)

    # Cross-check official eval path.
    from envs.outcomes.horizon_outcome import roll_policy
    restore_capture(env, blob["capture"])
    eval_out = roll_policy(env, model, book, blob, device, cap_steps=200, limit_s=float(args.limit_s))

    policy_rows, t0, policy_out, handoff_t = roll_policy_full(
        env, model, book, blob, device, float(args.limit_s),
    )
    write_csv(OUT / f"{args.id}__checkpoint3_trace.csv", policy_rows, t0)
    events = contact_events(policy_rows, t0)
    (OUT / f"{args.id}__checkpoint3_contact_events.json").write_text(
        json.dumps(events, indent=2), encoding="utf-8",
    )
    skill = classify(policy_rows, t0, handoff_t)

    # Expert sliding replay for comparison.
    restore_capture(env, blob["capture"])
    slide_rows, slide_t0, slide_out, slide_handoff = replay_slide_expert(env, blob, slide_row)
    write_csv(OUT / f"{args.id}__sliding_expert_trace.csv", slide_rows, slide_t0)
    slide_skill = classify(slide_rows, slide_t0, slide_handoff)

    # Live vertical oracle replay (same snapshot; may differ from the historical attempt log).
    vert_samples, vert_out, vert_raw = replay_vertical_oracle(blob, book)
    if vert_samples:
        write_csv(OUT / f"{args.id}__vertical_oracle_replay_trace.csv", vert_samples, vert_samples[0]["t"])
        vert_events = contact_events(vert_samples, vert_samples[0]["t"])
        (OUT / f"{args.id}__vertical_oracle_contact_events.json").write_text(
            json.dumps(vert_events, indent=2), encoding="utf-8",
        )
    else:
        vert_events = []

    video = OUT / f"{args.id}__checkpoint3.mp4"
    render_video(env, model, book, blob, device, float(args.limit_s), video)

    tokens = [row["token"] for row in policy_rows if row["phase"] == "recovery" and row["token"] is not None]
    summary = {
        "checkpoint": args.checkpoint,
        "episode_id": args.id,
        "evidence": evidence,
        "official_eval": {
            "reason": eval_out.get("reason"),
            "success": bool(eval_out.get("success")),
            "handoff": eval_out.get("handoff"),
            "recovery_elapsed": eval_out.get("recovery_elapsed"),
            "nominal_elapsed": eval_out.get("nominal_elapsed"),
            "lift_completed": eval_out.get("lift_completed"),
        },
        "checkpoint3_traced": policy_out,
        "checkpoint3_skill": skill,
        "checkpoint3_tokens": tokens,
        "checkpoint3_handoff_token_present": any(token == HANDOFF for token in [
            row["token"] for row in policy_rows if row["token"] is not None
        ]),
        "sliding_expert_replay": {
            "reason": slide_out.get("reason"),
            "success": bool(slide_out.get("success")),
            "skill": slide_skill,
        },
        "vertical_oracle_live_replay": vert_out,
        "files": {
            "policy_csv": f"{args.id}__checkpoint3_trace.csv",
            "policy_events": f"{args.id}__checkpoint3_contact_events.json",
            "slide_csv": f"{args.id}__sliding_expert_trace.csv",
            "video": video.name,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    write_report(summary, events, vert_events)
    print("AUDIT_DONE", args.id, policy_out["reason"], skill["label"], flush=True)


def write_report(summary, events, vert_events):
    eid = summary["episode_id"]
    ev = summary["evidence"]
    slide = ev["sliding_success"]["row"]
    fails = ev["vertical_recatch_failures"]
    skill = summary["checkpoint3_skill"]
    facts = skill["facts"]
    lines = [
        "# Checkpoint 3 skill audit",
        "",
        f"Primary IC: `{eid}`. Checkpoint: `{summary['checkpoint']}`.",
        "",
        "This report separates direct physical measurements from skill classification and from unverified speculation.",
        "",
        "## 1. Matched Sliding-success / Recatch-fail IC",
        "",
        "### Direct evidence",
        "",
        f"- Sliding tape: `{ev['sliding_success']['manifest']}`",
        f"- Strategy `{slide['strategy']}`, mechanism `{slide['mechanism']}`",
        f"- Stored outcome: `reason={slide['reason']}`, `success={slide['success']}`, `verified_success={slide['verified_success']}`",
        f"- Snapshot: `{slide['snapshot']}`",
        f"- Continuous actions: `{slide['continuous_npz']}`",
        f"- Recovery `{slide['recovery_elapsed']:.3f}` s then nominal `{slide['nominal_elapsed']:.3f}` s, `lift_completed={slide['lift_completed']}`",
        "",
        "Recorded vertical Recatch failures on the same episode id:",
        "",
    ]
    for row in fails:
        lines.append(
            f"- `{row['source']}`: strategy/stage `{row.get('strategy') or row.get('stage')}`, "
            f"reason `{row.get('reason')}`, success `{row.get('success')}`, "
            f"t_off `{row.get('t_off')}`, t_pad `{row.get('t_pad')}`"
        )
    lines.extend([
        "",
        f"- Present in `data/vertical_recatch/manifest.jsonl`: `{ev['in_vertical_recatch_manifest']}`",
        "",
        "### Gap",
        "",
        f"- {ev['gap']['failed_vertical_trajectory_npz']}",
        "- The historical failure is therefore a logged oracle outcome on this id, not a saved failed npz that can be re-inspected frame-by-frame without a live oracle replay.",
        "",
        "### Other matched candidates (not the primary audit IC)",
        "",
        "Same pattern appears for valid ICs in `data/staging/recovery_expansion/budget_attempts.jsonl`:",
        "`inward_extra_250043` (DROP_AFTER_HANDOFF), `inward_extra_250051` (LOST_GRASP), `inward_extra_250131` (LOST_GRASP), and this IC `k48r_valid_00009` (LOST_GRASP). Train-side matches appear in `followup_attempts.jsonl` (`ball_train_v2_0012`, `0031`, `0036`, `0038`, `0077`).",
        "",
        "## 2. Checkpoint 3 deterministic evaluation",
        "",
        "Action rule matches `envs/outcomes/horizon_outcome.roll_policy` / `scripts/train.py --eval`: argmax token, residual mean, legal decode, then official 12 s nominal verify.",
        "",
        "### Direct observations",
        "",
        f"- Official `roll_policy`: `{summary['official_eval']['reason']}`, success `{summary['official_eval']['success']}`, handoff `{summary['official_eval']['handoff']}`, recovery `{summary['official_eval']['recovery_elapsed']}`, nominal `{summary['official_eval']['nominal_elapsed']}`",
        f"- Traced closed loop: `{summary['checkpoint3_traced']['reason']}`, success `{summary['checkpoint3_traced']['success']}`, handoff `{summary['checkpoint3_traced']['handoff']}`",
        f"- Recovery tokens: `{summary['checkpoint3_tokens']}`",
        f"- Trace CSV: `{summary['files']['policy_csv']}`",
        f"- Contact events: `{summary['files']['policy_events']}`",
        f"- Video: `{summary['files']['video']}`",
        "",
        "Contact events during the Checkpoint 3 recovery (excerpt):",
        "",
        "```",
    ])
    for event in events[:40]:
        lines.append(json.dumps(event))
    if len(events) > 40:
        lines.append(f"... ({len(events) - 40} more)")
    lines.extend([
        "```",
        "",
        "Measured recovery contact facts:",
        "",
        "```json",
        json.dumps(facts, indent=2),
        "```",
        "",
        "## 3. Skill classification",
        "",
        f"**Classification: `{skill['label']}`**",
        "",
        "Rules used:",
        f"- Recatch requires sustained both-finger-off for at least `{BOTH_OFF_HOLD_S:.2f}` s, then bilateral recontact before handoff.",
        "- Sliding requires contact preserved on every recovery sample plus contact migration or in-hand object travel of at least 1 mm.",
        "- A single-frame both-off gap is not Recatch.",
        "- Horizontal/Vertical Recatch naming follows the original grasp scene of the IC, not the cylinder motion direction. This IC is a vertical-cylinder sliding scene.",
        "",
        "Reasons:",
        "",
    ])
    for why in skill["why"]:
        lines.append(f"- {why}")
    lines.extend([
        "",
        "### Sliding expert on the same snapshot",
        "",
        f"- Replay outcome: `{summary['sliding_expert_replay']['reason']}`, success `{summary['sliding_expert_replay']['success']}`",
        f"- Expert classification: `{summary['sliding_expert_replay']['skill']['label']}`",
        f"- Expert facts: `{json.dumps(summary['sliding_expert_replay']['skill']['facts'], default=float)}`",
        "",
        "### Live vertical oracle replay on the same snapshot",
        "",
        "This is a fresh `follow_early` rollout, not the historical saved failure npz.",
        "",
        f"- Live outcome: `{json.dumps(summary['vertical_oracle_live_replay'], default=float)}`",
        "",
    ])
    if vert_events:
        lines.append("Live oracle contact events (excerpt):")
        lines.append("")
        lines.append("```")
        for event in vert_events[:40]:
            lines.append(json.dumps(event))
        lines.append("```")
        lines.append("")
    lines.extend([
        "## 4. Interpretation boundaries",
        "",
        "### Fact",
        "",
        f"- On `{eid}`, a verified Sliding demonstration reaches SUCCESS_12S after a full 12 s nominal hold.",
        "- The same episode id has a recorded vertical Recatch oracle failure in the expansion attempt log.",
        f"- Checkpoint 3 reaches `{summary['checkpoint3_traced']['reason']}` under the official deterministic eval rule.",
        f"- The contact timeline classification for Checkpoint 3 on this IC is `{skill['label']}`.",
        "",
        "### Classification, not a new physical measurement",
        "",
        "- The Sliding / Recatch / Hybrid label is derived from the contact thresholds above.",
        "- Video appearance and gripper opening alone were not used as the decision.",
        "",
        "### Speculation / not proven",
        "",
        "- That Checkpoint 3 'always' Recatches on the full validation set is not proven by this one-IC audit.",
        "- That Recatch is generally feasible on every Sliding-only IC is not proven. The historical vertical failure can be search-coverage limited (one or a few open-overchase laws) rather than a physical impossibility.",
        "- Without the original failed Recatch npz, the exact historical failure trajectory cannot be reconstructed bit-for-bit; the live oracle replay is the available substitute.",
        "",
        "## 5. Support check",
        "",
        "| Need | Status |",
        "|---|---|",
        "| Same snapshot for Sliding success | Available |",
        "| Sliding SUCCESS_12S after full nominal | Available in manifest + expert replay |",
        "| Recatch failure on same IC | Available as attempt log; failed npz missing |",
        "| Checkpoint 3 deterministic eval | Available via `roll_policy` |",
        "| Per-step token/residual/action/contact CSV | Written by this script |",
        "| Video | Written by this script |",
        "",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
