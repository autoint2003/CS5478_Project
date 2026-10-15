"""Privileged airborne-intercept search for legal K48 recatch demonstrations.

Commands are recovery7d_airborne_v3 actions executed by step_legal. The
cylinder, arm, and ball state are never rewritten. TABLE_DROP stays latched.
Outputs go only to results/diagnostics/airborne_recatch_search.
"""

from __future__ import annotations


import json
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from envs.outcomes.nominal_pose import capture_ref, closed_grip, errors, hand_R, rotvec
from controllers.gripper.controller import finger_opening
from controllers.recovery.action_map import AIRBORNE_V3, W_MAX, normalize_action
from envs.simulation.grasp_quality import body_twist
from envs.outcomes.physical_recovery import physical_pack

from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed

from experiments.ball_impact.privileged_ball_impact_search import step_legal
from envs.outcomes.table_drop import export_state, latched

OUT = ROOT / "results" / "diagnostics" / "airborne_recatch_search"
RAW = OUT / "raw"
TRAJ = OUT / "trajectories"
VIDEO = OUT / "videos"

# Measured on ball_train_v2_0051 with the live Cartesian tracker.
# Commands of 1 m/s are not achieved. Downward +3.8 m/s is not usable.
V_HAND_X = 0.70
V_HAND_Y = 0.45
V_DOWN = 0.60
V_UP = 0.55
W_USE = 0.40
PINCH_LOCAL = np.array([0.0, 0.0, 0.100])
TABLE_Z = 0.40
CYL_R = 0.018
CYL_HALF = 0.030
LOSS_STEPS = 2
HOLD_STEPS = 3
CONTROL_S = 0.02


def _rot(sim):
    return np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)


def _obj_rot(sim):
    return np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)


def _twist(sim, body):
    linear, angular = body_twist(sim.model, sim.data, body)
    return np.asarray(linear, float), np.asarray(angular, float)


def usable_hand_velocity(v_hand):
    out = np.array(v_hand, float)
    out[0] = float(np.clip(out[0], -V_HAND_X, V_HAND_X))
    out[1] = float(np.clip(out[1], -V_HAND_Y, V_HAND_Y))
    if out[2] >= 0.0:
        out[2] = min(float(out[2]), V_DOWN)
    else:
        out[2] = max(float(out[2]), -V_UP)
    return out


def predict_com(position, velocity, tau):
    gravity = np.array([0.0, 0.0, -9.81])
    return np.asarray(position, float) + np.asarray(velocity, float) * float(tau) + 0.5 * gravity * float(tau) ** 2


def time_to_table(position, velocity, axis):
    """Ballistic time until the cylinder's lowest point reaches the table top."""
    vertical = abs(float(axis[2]))
    radial = float(np.sqrt(max(0.0, 1.0 - vertical * vertical)))
    lowest = float(position[2]) - (CYL_R * radial + CYL_HALF * vertical)
    gap = lowest - TABLE_Z
    if gap <= 0.0:
        return 0.0
    vz = float(velocity[2])
    a = 0.5 * -9.81
    disc = vz * vz - 4.0 * a * gap
    if disc < 0.0:
        return 1.0e3
    root = float(np.sqrt(disc))
    times = [t for t in ((-vz - root) / (2.0 * a), (-vz + root) / (2.0 * a)) if t > 1.0e-4]
    return float(min(times)) if times else 1.0e3


def jaws_contain(local, opening):
    return bool(
        abs(float(local[1])) <= float(opening) + 0.006
        and abs(float(local[0])) <= 0.028
        and 0.045 <= float(local[2]) <= 0.125
    )


def sense(sim):
    rh = _rot(sim)
    ro = _obj_rot(sim)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    vh, wh = _twist(sim, sim.ids.hand_body)
    vo, wo = _twist(sim, sim.ids.object_body)
    pack = physical_pack(sim)
    opening = finger_opening(sim.data, sim.ids)
    local = rh.T @ (po - ph)
    return {
        "t": float(sim.data.time),
        "rh": rh,
        "ph": ph,
        "vh": vh,
        "wh": wh,
        "po": po,
        "vo": vo,
        "wo": wo,
        "axis": ro[:, 2].copy(),
        "local": local,
        "opening": float(opening),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "z": float(po[2]),
        "inside": jaws_contain(local, opening),
        "bilateral": int(pack["nL"]) > 0 and int(pack["nR"]) > 0,
        "latched": bool(latched(sim)),
        "ttc": time_to_table(po, vo, ro[:, 2]),
    }


def _action(sim, v_world, w_world, grip):
    rh = _rot(sim)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    v_hand = usable_hand_velocity(rh.T @ np.asarray(v_world, float))
    w_body = rd.T @ np.asarray(w_world, float)
    w_body = np.clip(w_body, -W_USE, W_USE)
    tau = float(np.interp(np.clip(grip, -1.0, 1.0), [-1.0, 1.0], [2.0, -18.0]))
    return normalize_action(v_hand, w_body, tau, AIRBORNE_V3)


def _align_rate(row):
    opening = row["rh"][:, 1]
    axis = row["axis"]
    couple = float(np.dot(np.cross(opening, axis), row["rh"][:, 2]))
    return float(np.clip(2.0 * couple, -W_USE, W_USE)) * row["rh"][:, 2]


def _clip_speed(vector, limit):
    out = np.asarray(vector, float)
    norm = float(np.linalg.norm(out))
    if norm > float(limit) > 0.0:
        out = out * (float(limit) / norm)
    return out


class InterceptPolicy:
    """Match the cylinder velocity, then add a small position correction.

    A raw position gain saturates the 0.7 m/s hand command and knocks the
    cylinder off the pads. While the fingers still touch it, the correction
    is capped at a few centimetres per second.
    """

    def __init__(self, spec, ref):
        self.spec = spec
        self.ref = ref
        self.loss_run = 0
        self.hold_run = 0
        self.had_loss = False
        self.captured = False
        self.stable_steps = 0
        self.origin = None

    def __call__(self, sim):
        row = sense(sim)
        if self.origin is None:
            self.origin = row["ph"].copy()
        self.loss_run = self.loss_run + 1 if (row["nL"] == 0 and row["nR"] == 0) else 0
        if self.loss_run >= LOSS_STEPS:
            self.had_loss = True
        self.hold_run = self.hold_run + 1 if row["bilateral"] else 0
        if self.had_loss and self.hold_run >= HOLD_STEPS and row["inside"]:
            self.captured = True
        if self.captured and row["bilateral"]:
            self.stable_steps += 1
        if self.captured and self.stable_steps >= HOLD_STEPS:
            correction = np.asarray(self.ref["p_nom"], float) - row["ph"]
            v_world = row["vo"] + _clip_speed(correction, 0.08)
            return _action(sim, v_world, np.zeros(3), 1.0), row
        future = predict_com(row["po"], row["vo"], self.spec["lead"])
        if not row["bilateral"]:
            future = future.copy()
            future[:2] = self.origin[:2] + np.clip(future[:2] - self.origin[:2], -0.06, 0.06)
        aim = future - row["rh"] @ PINCH_LOCAL
        correction = aim - row["ph"]
        if row["bilateral"] or row["inside"]:
            limit = float(self.spec["held_speed"])
            grip = float(self.spec["held_grip"])
        else:
            limit = 0.20 if float(np.linalg.norm(correction)) > 0.04 else 0.08
            grip = -1.0 if self.spec["mode"] == "intercept" else float(self.spec["held_grip"])
        v_world = float(self.spec["match"]) * row["vo"] + _clip_speed(correction, limit)
        w_world = _align_rate(row) if self.spec["align"] and not row["bilateral"] else np.zeros(3)
        return _action(sim, v_world, w_world, grip), row


def _sample(row, action):
    return {
        "t": round(row["t"], 4),
        "nL": row["nL"],
        "nR": row["nR"],
        "z": round(row["z"], 4),
        "opening": round(row["opening"], 4),
        "inside": bool(row["inside"]),
        "ttc": None if row["ttc"] > 10.0 else round(row["ttc"], 4),
        "local": [round(float(x), 4) for x in row["local"]],
        "vo": [round(float(x), 4) for x in row["vo"]],
        "vh": [round(float(x), 4) for x in row["vh"]],
        "po": [round(float(x), 4) for x in row["po"]],
        "ph": [round(float(x), 4) for x in row["ph"]],
        "action": [round(float(x), 4) for x in np.asarray(action, float)],
        "latched": bool(row["latched"]),
    }


def rollout(env, blob, spec, verify_nominal=False):
    restore_capture(env, blob["capture"])
    sim = env.sim
    t0 = float(sim.data.time)
    policy = InterceptPolicy(spec, blob["ref"])
    actions = []
    trace = []
    lost_low = 0
    reason = None
    physics = 0
    dt = float(sim.model.opt.timestep)
    min_gap = 1.0e9
    for _ in range(200):
        if float(sim.data.time) - t0 >= 4.0 - 1e-9:
            reason = "RECOVERY_LIMIT"
            break
        action, row = policy(sim)
        min_gap = min(min_gap, float(np.linalg.norm(row["local"] - PINCH_LOCAL)))
        trace.append(_sample(row, action))
        if row["latched"]:
            reason = "TABLE_DROP"
            break
        before = float(sim.data.time)
        step_legal(sim, action)
        physics += int(round((float(sim.data.time) - before) / dt))
        actions.append(np.asarray(action, np.float32))
        failed, lost_low = recovery_failed(sim, blob["ref"], lost_low)
        if latched(sim):
            reason = "TABLE_DROP"
            trace.append(_sample(sense(sim), action))
            break
        if failed:
            from training.recovery_execution import _escaped
            end = sense(sim)
            if latched(sim):
                reason = "TABLE_DROP"
            elif float(end["z"]) < 0.30:
                reason = "FELL_BELOW_0.30"
            elif _escaped(sim):
                reason = "ESCAPED_XY"
            else:
                reason = "LOST_CONTACT_TIMEOUT"
            trace.append(_sample(end, action))
            break
        if policy.captured and policy.stable_steps >= 15 and row["bilateral"]:
            err = errors(sim, blob["ref"])
            if float(err["ep"]) < 0.03 and float(err["vrel"]) < 0.08:
                reason = "READY"
                break
    if reason is None:
        reason = "RECOVERY_LIMIT"
    outcome = None
    still_held = bool(trace) and trace[-1]["nL"] > 0 and trace[-1]["nR"] > 0 and not latched(sim)
    if latched(sim):
        reason = "TABLE_DROP"
    if reason not in ("TABLE_DROP", "FELL_BELOW_0.30") and verify_nominal and (policy.captured or (still_held and reason == "RECOVERY_LIMIT")):
        before = float(sim.data.time)
        outcome = finish_recovery(env, blob["ref"], t0, "early")
        physics += int(round(max(0.0, float(sim.data.time) - before) / dt))
        if outcome["reason"] == "TABLE_DROP" or latched(sim):
            reason = "TABLE_DROP"
        elif outcome.get("success"):
            reason = "SUCCESS_12S"
        else:
            reason = "HANDOFF_" + str(outcome["reason"])
    elif reason is None:
        reason = "RECOVERY_LIMIT"
    return {
        "spec": spec,
        "reason": reason,
        "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 7), np.float32),
        "trace": trace,
        "physics_steps": int(physics),
        "had_airborne_loss": bool(policy.had_loss),
        "captured": bool(policy.captured),
        "min_pinch_gap_m": float(min_gap),
        "success": bool(outcome and outcome.get("success")),
        "outcome_reason": None if outcome is None else outcome["reason"],
        "table_drop": export_state(sim),
        "qpos_at_end": float(np.max(np.abs(sim.data.qpos))),
    }


def _rank(row):
    """Lower is better. Continuous success outranks a mere capture."""
    order = {
        "SUCCESS_12S": 0,
        "READY": 1,
    }
    base = order.get(row["reason"], 4)
    if str(row["reason"]).startswith("HANDOFF_"):
        base = 2
    if row["reason"] in ("TABLE_DROP", "FELL_BELOW_0.30", "ESCAPED_XY", "LOST_CONTACT_TIMEOUT"):
        base = 5
    if not row["had_airborne_loss"] and row["reason"] == "SUCCESS_12S":
        base = 3
    return (base, 0 if row["captured"] else 1, row["min_pinch_gap_m"], -row["physics_steps"])


def candidate_grid():
    """Short leads only. The measured free-fall window on these ICs is about 0.14 s."""
    grid = []
    for held_grip in (1.0, 0.2):
        for held_speed in (0.03, 0.06):
            grid.append({
                "mode": "closed",
                "lead": 0.0,
                "match": 1.0,
                "gain": 0.0,
                "align": False,
                "held_grip": held_grip,
                "held_speed": held_speed,
            })
    for lead in (0.04, 0.08, 0.12):
        for held_speed in (0.04, 0.08):
            for align in (False, True):
                grid.append({
                    "mode": "intercept",
                    "lead": float(lead),
                    "match": 1.0,
                    "gain": 0.0,
                    "align": bool(align),
                    "held_grip": 1.0,
                    "held_speed": float(held_speed),
                })
    return grid


def _refine(best):
    spec = dict(best["spec"])
    if spec["mode"] != "intercept":
        return []
    out = []
    for lead in (max(0.02, spec["lead"] - 0.02), spec["lead"] + 0.02):
        for held_speed in (0.03, 0.10):
            child = dict(spec)
            child["lead"] = float(lead)
            child["held_speed"] = float(held_speed)
            out.append(child)
    return out


def search_ic(env, blob, book, verify_nominal=True):
    from data_generation.rebuild_k48_recovery import legal_verify, legalize

    tried = []
    physics = 0
    best = None
    queue = candidate_grid()
    refined = False
    while queue and len(tried) < 40 and physics < 250000:
        spec = queue.pop(0)
        screening = rollout(env, blob, spec, verify_nominal=False)
        physics += screening["physics_steps"]
        tried.append({k: screening[k] for k in ("spec", "reason", "had_airborne_loss", "captured", "min_pinch_gap_m", "physics_steps")})
        if best is None or _rank(screening) < _rank(best):
            best = screening
        if screening["captured"] and not refined:
            refined = True
            queue = _refine(screening) + queue
        if screening["reason"] == "SUCCESS_12S":
            break
    verified = None
    quantized = None
    retained = best is not None and best["reason"] == "RECOVERY_LIMIT" and best["trace"] and best["trace"][-1]["nL"] > 0 and best["trace"][-1]["nR"] > 0
    if best is not None and verify_nominal and (best["captured"] or retained):
        verified = rollout(env, blob, best["spec"], verify_nominal=True)
        physics += verified["physics_steps"]
        best = verified
        if verified["success"] and verified["had_airborne_loss"]:
            tokens, residuals, decoded, packed = legalize(verified["actions"], book)
            legal = bool(packed["fraction_within_0.25"] is not None and float(packed["max_linf"]) <= 0.25 + 1e-8)
            q_out = None
            q_reason = "NOT_REPRESENTABLE"
            if legal:
                q_out, _obs, _exe, _contacts, qpos_error, phase_ok = legal_verify(env, blob, decoded)
                physics += int(round(12.0 / float(env.sim.model.opt.timestep))) + 10 * len(decoded)
                q_reason = q_out["reason"]
                legal = bool(q_out.get("success")) and qpos_error <= 1e-6 and phase_ok
            quantized = {
                "representable": bool(packed["fraction_within_0.25"] is not None and float(packed["max_linf"]) <= 0.25 + 1e-8),
                "fraction_within_0.25": packed["fraction_within_0.25"],
                "max_linf": packed["max_linf"],
                "legal_verify_reason": q_reason,
                "legal_verify_success": bool(q_out and q_out.get("success")),
                "accepted": bool(legal),
                "tokens": tokens.tolist(),
                "decoded": decoded,
            }
    return {
        "attempts": len(tried),
        "physics_steps": int(physics),
        "tried": tried,
        "best": best,
        "quantized": quantized,
    }


def _manifest_rows():
    rows = []
    path = ROOT / "data" / "recatch" / "manifest.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("split") == "train":
                rows.append(row)
    return rows


def _features(env, blob):
    restore_capture(env, blob["capture"])
    row = sense(env.sim)
    return {
        "z": row["z"],
        "vh": float(np.linalg.norm(row["vo"][:2])),
        "vz": float(row["vo"][2]),
        "spin": float(np.linalg.norm(row["wo"])),
        "contacts": int(row["nL"] + row["nR"]),
        "ttc": row["ttc"],
        "local_z": float(row["local"][2]),
        "opening": row["opening"],
    }


def choose_ics(env):
    rows = _manifest_rows()
    named = ["ball_train_v2_0051", "ball_train_v2_0226", "k48r_train_00135"]
    by_id = {}
    for row in rows:
        by_id.setdefault(row["id"], row)
    pool = []
    for index in range(0, len(rows), max(1, len(rows) // 16)):
        pool.append(rows[index])
    for name in named:
        if name in by_id:
            pool.append(by_id[name])
    measured = []
    seen = set()
    blobs = {}
    for row in pool:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        path = ROOT / row["snapshot"]
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        blobs[row["id"]] = (row, blob)
        feat = _features(env, blob)
        measured.append((row, feat))
    chosen = []
    for name in named:
        for row, feat in measured:
            if row["id"] == name:
                chosen.append((row, feat))
    rest = [item for item in measured if item[0]["id"] not in named]

    def vector(feat):
        return np.array([feat["vh"], feat["vz"], feat["spin"], feat["z"], feat["contacts"]], float)

    while rest and len(chosen) < 8:
        anchor = np.stack([vector(feat) for _row, feat in chosen]) if chosen else np.zeros((1, 5))
        rest.sort(key=lambda item: float(np.min(np.linalg.norm(anchor - vector(item[1]), axis=1))))
        chosen.append(rest.pop(-1))
    return [(row, feat, blobs[row["id"]][1]) for row, feat in chosen]


def _events(trace):
    loss = next((row["t"] for row in trace if row["nL"] == 0 and row["nR"] == 0), None)
    both = None
    seen_loss = False
    for row in trace:
        if row["nL"] == 0 and row["nR"] == 0:
            seen_loss = True
        elif seen_loss and row["nL"] > 0 and row["nR"] > 0 and both is None:
            both = row["t"]
    drop = next((row["t"] for row in trace if row["latched"]), None)
    return {"loss_t": loss, "recapture_t": both, "drop_t": drop}


def _save_trace(case_id, best, quantized):
    TRAJ.mkdir(parents=True, exist_ok=True)
    path = TRAJ / f"{case_id}__airborne_intercept_recatch.npz"
    payload = {
        "actions": best["actions"],
        "reason": np.asarray(best["reason"]),
    }
    if quantized and quantized.get("accepted"):
        payload["quantized_actions"] = quantized["decoded"]
        payload["tokens"] = np.asarray(quantized["tokens"], np.int64)
    np.savez(path, **payload)
    (TRAJ / f"{case_id}__trace.json").write_text(json.dumps({
        "spec": best["spec"],
        "reason": best["reason"],
        "events": _events(best["trace"]),
        "trace": best["trace"],
    }, indent=2), encoding="utf-8")
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _classify(best, quantized):
    if best is None:
        return "SEARCH_EXHAUSTED"
    if quantized and quantized.get("accepted"):
        return "LEGAL_AIRBORNE_RECATCH"
    if best["success"] and best["had_airborne_loss"] and quantized is not None and not quantized.get("accepted"):
        return "QUANTIZE_FAIL"
    if best["success"] and not best["had_airborne_loss"]:
        return "RETAINED_GRASP"
    if best["had_airborne_loss"] and best["captured"] and str(best["reason"]).startswith("HANDOFF_"):
        return "SLIP_OR_HANDOFF_FAIL"
    if best["had_airborne_loss"] and not best["captured"] and best["reason"] == "TABLE_DROP":
        return "TABLE_DROP_BEFORE_CAPTURE"
    if best["had_airborne_loss"] and best["captured"] and best["reason"] == "TABLE_DROP":
        return "SLIP_THEN_TABLE_DROP"
    if not best["had_airborne_loss"] and best["reason"] in ("TABLE_DROP", "FELL_BELOW_0.30", "ESCAPED_XY"):
        return "LOST_WHILE_COMMAND_STILL_CLOSED"
    if best["reason"] == "RECOVERY_LIMIT":
        return "SEARCH_BUDGET_NO_CAPTURE"
    return best["reason"]


def render_case(env, blob, actions, dest, title):
    from experiments.legacy.demo.demonstrations.demo_ballistic_recovery_comparison import caption
    from experiments.legacy.demo.demonstrations.demo_grav_reposition_comparison import write_frames_mp4
    from visualization.impact_view import mjv_camera_from_preset

    restore_capture(env, blob["capture"])
    sim = env.sim
    renderer = mujoco.Renderer(sim.model, 480, 640)
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    cam = mjv_camera_from_preset({"lookat": hand, "distance": 0.85, "azimuth": 270.0, "elevation": -20.0})
    frames = []
    next_t = float(sim.data.time)
    lost_low = 0
    t0 = float(sim.data.time)

    def grab(phase):
        nonlocal next_t
        if float(sim.data.time) + 1e-9 < next_t:
            return
        renderer.update_scene(sim.data, cam)
        frames.append(caption(renderer.render(), f"{title}  {phase}  t={sim.data.time:.2f}s"))
        next_t += 1.0 / 25.0

    grab("START")
    phase = "INTERCEPT"
    for action in actions:
        if float(sim.data.time) - t0 >= 4.0 - 1e-9 or latched(sim):
            break
        step_legal(sim, action)
        failed, lost_low = recovery_failed(sim, blob["ref"], lost_low)
        row = sense(sim)
        if row["bilateral"]:
            phase = "CAPTURE"
        if latched(sim) or failed:
            phase = "TABLE_DROP" if latched(sim) else "FAIL"
            grab(phase)
            break
        grab(phase)
    if not latched(sim) and phase == "CAPTURE":
        from envs.outcomes.recovery_success import apply_handoff, nominal_step
        apply_handoff(sim, blob["ref"])
        phase = "HANDOFF"
        hold_end = float(sim.data.time) + 12.0
        while float(sim.data.time) < hold_end - 1e-9 and not latched(sim):
            nominal_step(sim)
            if latched(sim):
                phase = "TABLE_DROP"
            grab(phase)
        if not latched(sim):
            phase = "SUCCESS_12S"
            grab(phase)
    renderer.close()
    if not frames:
        return None
    return write_frames_mp4(frames, dest)


def main():
    import torch
    from training.recovery_episode import make_training_env
    from training.k47_codebook import load_codebook
    from training.unified_k48 import load_trained_k48

    RAW.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cpu")
    model, _saved = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", device)
    model.eval()
    env = make_training_env(device, model=model)
    book = load_codebook()
    selected = choose_ics(env)
    records = []
    for row, feat, blob in selected:
        print("SEARCH", row["id"], row["strategy"], {k: round(feat[k], 3) if isinstance(feat[k], float) else feat[k] for k in feat}, flush=True)
        result = search_ic(env, blob, book, verify_nominal=True)
        best = result["best"]
        label = _classify(best, result["quantized"])
        tape = None
        if best is not None and (best["captured"] or best["had_airborne_loss"] or best["success"]):
            tape = _save_trace(row["id"], best, result["quantized"])
        record = {
            "id": row["id"],
            "strategy": row["strategy"],
            "snapshot": row["snapshot"],
            "features": feat,
            "label": label,
            "attempts": result["attempts"],
            "physics_steps": result["physics_steps"],
            "best_reason": None if best is None else best["reason"],
            "best_spec": None if best is None else best["spec"],
            "had_airborne_loss": bool(best and best["had_airborne_loss"]),
            "captured": bool(best and best["captured"]),
            "continuous_success": bool(best and best["success"]),
            "quantized_success": bool(result["quantized"] and result["quantized"].get("accepted")),
            "quantize": None if result["quantized"] is None else {k: result["quantized"][k] for k in ("representable", "fraction_within_0.25", "max_linf", "legal_verify_reason", "legal_verify_success", "accepted")},
            "events": None if best is None else _events(best["trace"]),
            "min_pinch_gap_m": None if best is None else best["min_pinch_gap_m"],
            "table_drop_latched": bool(best and best["table_drop"]["latched"]),
            "trajectory": tape,
            "tried": result["tried"],
        }
        records.append(record)
        print("  ", label, "loss", record["had_airborne_loss"], "capture", record["captured"], "steps", record["physics_steps"], flush=True)
        (RAW / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    videos = []
    for record, (_row, _feat, blob) in zip(records, selected):
        if record["trajectory"] is None:
            continue
        if record["label"] not in ("LEGAL_AIRBORNE_RECATCH", "QUANTIZE_FAIL", "TABLE_DROP_BEFORE_CAPTURE", "SLIP_OR_HANDOFF_FAIL", "SLIP_THEN_TABLE_DROP", "RETAINED_GRASP", "FELL_BELOW_0.30") and len(videos) >= 3:
            continue
        actions = np.load(ROOT / record["trajectory"])["actions"]
        dest = VIDEO / f"{record['id']}__{record['label']}.mp4"
        path = render_case(env, blob, actions, dest, record["label"])
        videos.append({"id": record["id"], "label": record["label"], "video": path})
        print("VIDEO", path, flush=True)
        if sum(1 for item in videos if item["video"]) >= 4:
            break
    summary = {
        "seconds": round(time.perf_counter() - started, 1),
        "ics": len(records),
        "physics_steps": int(sum(row["physics_steps"] for row in records)),
        "labels": {row["id"]: row["label"] for row in records},
        "videos": videos,
    }
    (RAW / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SEARCH_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
