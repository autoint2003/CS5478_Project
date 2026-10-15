"""Contact-aware recatch search from the post-impact snapshot.

Legal recovery7d_airborne_v3 actions only. No extra speed cap beyond the
action normalization. TABLE_DROP, the codebook, and the nominal controller
are unchanged. Writes only under results/diagnostics/contact_aware_recatch_search.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery_actions import AIRBORNE_V3, W_MAX, map_action, normalize_action
from training.airborne_intercept_recatch import PINCH_LOCAL, predict_com, sense
from training.history_aware_critic_ablation import restore_capture, take_capture
from training.horizon_outcome import finish_recovery, recovery_failed
from training.nominal_pose_audit import errors
from training.privileged_ball_impact_search import step_legal
from training.replay_core import raw_object_contacts
from training.table_drop import latched

OUT = ROOT / "results" / "diagnostics" / "contact_aware_recatch_search"
RAW = OUT / "raw"
TRAJ = OUT / "trajectories"
VIDEO = OUT / "videos"
N_SEG = 4
SEG_STEPS = 4
PLAN_STEPS = N_SEG * SEG_STEPS
EXECUTE_STEPS = SEG_STEPS
POP = 8
ELITES = 3
ITERS = 2
REPLAN_CYCLES = 5


def _load(path):
    with (ROOT / path).open("rb") as handle:
        return pickle.load(handle)


def _limits(sim):
    arm = np.array([87, 87, 87, 87, 12, 12, 12], float)
    return arm


def capability(env, blob):
    """Time series for saturated axis commands, including setpoint error."""
    from training.history_aware_critic_ablation import restore_capture as restore

    sim = env.sim
    series = {}
    commands = {
        "vx+": np.array([1, 0, 0, 0, 0, 0, 1], np.float32),
        "vy+": np.array([0, 1, 0, 0, 0, 0, 1], np.float32),
        "vz_down": np.array([0, 0, 1, 0, 0, 0, 1], np.float32),
        "vz_up": np.array([0, 0, -1, 0, 0, 0, 1], np.float32),
        "wz+": np.array([0, 0, 0, 0, 0, 1, 1], np.float32),
    }
    for name, action in commands.items():
        restore(env, blob["capture"])
        rows = []
        for _ in range(25):
            rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
            rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
            cmd = map_action(action, rd, rh, AIRBORNE_V3)
            p0 = np.array(sim.data.xpos[sim.ids.hand_body], float)
            step_legal(sim, action)
            p1 = np.array(sim.data.xpos[sim.ids.hand_body], float)
            vh = sense(sim)["vh"]
            force = np.array(sim.data.qfrc_actuator[:7], float)
            limit = _limits(sim)
            rows.append({
                "cmd_v": [round(float(x), 4) for x in cmd["v_world"]],
                "cmd_w": [round(float(x), 4) for x in cmd["w_world"]],
                "actual_v": [round(float(x), 4) for x in vh],
                "actual_speed": round(float(np.linalg.norm(vh)), 4),
                "step_disp": round(float(np.linalg.norm(p1 - p0)), 5),
                "sat_frac": round(float(np.mean(np.abs(force) > 0.95 * limit)), 3),
                "grip_ctrl": round(float(sim.data.ctrl[7]), 3),
            })
        speeds = [row["actual_speed"] for row in rows]
        series[name] = {
            "median_speed": float(np.median(speeds)),
            "max_speed": float(np.max(speeds)),
            "final_speed": float(speeds[-1]),
            "mean_step_disp_m": float(np.mean([row["step_disp"] for row in rows])),
            "mean_sat_frac": float(np.mean([row["sat_frac"] for row in rows])),
            "series": rows,
        }
    return series


def _events(trace):
    loss = None
    regain = None
    drop = None
    slip = None
    seen_loss = False
    for row in trace:
        separating = abs(row["local"][0]) > 0.02 or row["local"][2] > 0.112 or (row["nL"] == 0) != (row["nR"] == 0)
        if slip is None and separating:
            slip = row["t"]
        both_off = row["nL"] == 0 and row["nR"] == 0
        if both_off and loss is None:
            loss = row["t"]
            seen_loss = True
        elif seen_loss and row["nL"] > 0 and row["nR"] > 0 and regain is None:
            regain = row["t"]
        if row["latched"] and drop is None:
            drop = row["t"]
    return {"slip_t": slip, "loss_t": loss, "recapture_t": regain, "drop_t": drop, "end_t": None if not trace else trace[-1]["t"]}


def _score(trace):
    """Search rank only. Ending in a grasp dominates a later fall."""
    if not trace:
        return -1000.0
    end = trace[-1]
    if any(row["latched"] for row in trace):
        duration = float(end["t"] - trace[0]["t"])
        return -400.0 + 15.0 * min(duration, 0.4)
    bilateral = [row for row in trace if row["nL"] > 0 and row["nR"] > 0]
    end_held = end["nL"] > 0 and end["nR"] > 0
    pinch = float(np.linalg.norm(np.asarray(end["local"]) - PINCH_LOCAL))
    vrel = float(np.linalg.norm(np.asarray(end["vo"]) - np.asarray(end["vh"])))
    regained = _events(trace)["recapture_t"] is not None
    return (
        (250.0 if end_held else 0.0)
        + 100.0 * len(bilateral) / len(trace)
        + (80.0 if regained else 0.0)
        - 50.0 * pinch
        - 15.0 * min(vrel, 2.0)
    )


def _compact(row, action, cmd_v):
    return {
        "t": round(float(row["t"]), 4),
        "nL": int(row["nL"]),
        "nR": int(row["nR"]),
        "z": round(float(row["z"]), 4),
        "opening": round(float(row["opening"]), 4),
        "local": [round(float(x), 4) for x in row["local"]],
        "vo": [round(float(x), 4) for x in row["vo"]],
        "vh": [round(float(x), 4) for x in row["vh"]],
        "wo": [round(float(x), 3) for x in row["wo"]],
        "cmd_v": [round(float(x), 4) for x in cmd_v],
        "action": [round(float(x), 4) for x in np.asarray(action, float)],
        "latched": bool(row["latched"]),
        "ttc": None if row["ttc"] > 10 else round(float(row["ttc"]), 4),
    }


def play(env, blob, actions, capture=None):
    if capture is None:
        restore_capture(env, blob["capture"])
    else:
        restore_capture(env, capture)
    sim = env.sim
    t0 = float(sim.data.time)
    trace = []
    executed = []
    lost_low = 0
    reason = None
    dt = float(sim.model.opt.timestep)
    physics = 0
    for action in np.asarray(actions, np.float32):
        if float(sim.data.time) - t0 >= 4.0 - 1e-9:
            reason = "RECOVERY_LIMIT"
            break
        row = sense(sim)
        rh = row["rh"]
        rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        cmd = map_action(action, rd, rh, AIRBORNE_V3)
        trace.append(_compact(row, action, cmd["v_world"]))
        if row["latched"]:
            reason = "TABLE_DROP"
            break
        before = float(sim.data.time)
        step_legal(sim, action)
        physics += int(round((float(sim.data.time) - before) / dt))
        executed.append(np.asarray(action, np.float32))
        failed, lost_low = recovery_failed(sim, blob["ref"], lost_low)
        if latched(sim):
            reason = "TABLE_DROP"
            trace.append(_compact(sense(sim), action, cmd["v_world"]))
            break
        if failed:
            reason = "OTHER_FAILURE"
            trace.append(_compact(sense(sim), action, cmd["v_world"]))
            break
    if reason is None:
        reason = "HORIZON_END" if len(executed) < 200 else "RECOVERY_LIMIT"
    return {
        "reason": reason,
        "trace": trace,
        "actions": np.stack(executed, axis=0) if executed else np.zeros((0, 7), np.float32),
        "physics_steps": int(physics),
        "score": _score(trace),
        "events": _events(trace),
        "held_end": bool(trace) and trace[-1]["nL"] > 0 and trace[-1]["nR"] > 0 and not latched(sim),
    }


def _templates():
    close_up = np.array([0.0, 0.0, -0.7, 0.0, 0.0, 0.0, 1.0], np.float32)
    wrist_up = np.array([0.0, 0.0, -0.4, 0.0, 0.0, 0.8, 0.4], np.float32)
    open_down = np.array([0.0, 0.0, 0.45, 0.0, 0.0, 0.0, -1.0], np.float32)
    close_down = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0], np.float32)
    lateral = np.array([0.8, 0.0, -0.3, 0.7, 0.0, 0.0, 0.5], np.float32)
    hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32)
    rows = [
        np.stack([close_up, close_up, close_up, close_up]),
        np.stack([wrist_up, wrist_up, close_up, close_up]),
        np.stack([open_down, open_down, close_down, close_down]),
        np.stack([lateral, lateral, wrist_up, close_up]),
        np.stack([hold, hold, hold, hold]),
        np.stack([open_down, lateral, close_down, close_up]),
    ]
    return [row.reshape(-1) for row in rows]


def _decode(theta):
    segs = np.clip(np.asarray(theta, float).reshape(N_SEG, 7), -1.0, 1.0)
    return np.repeat(segs, SEG_STEPS, axis=0).astype(np.float32)


def cem_cycle(env, blob, capture, rng):
    mean = np.zeros(N_SEG * 7, float)
    std = np.full(N_SEG * 7, 0.55)
    templates = _templates()
    best = None
    history = []
    for iteration in range(ITERS):
        samples = []
        if iteration == 0:
            samples.extend(templates)
            while len(samples) < POP:
                samples.append(rng.uniform(-1.0, 1.0, size=mean.shape))
        else:
            while len(samples) < POP:
                draw = mean + std * rng.normal(size=mean.shape)
                samples.append(np.clip(draw, -1.0, 1.0))
        scored = []
        for theta in samples[:POP]:
            result = play(env, blob, _decode(theta), capture=capture)
            scored.append((result["score"], theta, result))
            history.append({
                "iteration": iteration,
                "score": result["score"],
                "reason": result["reason"],
                "events": result["events"],
                "physics_steps": result["physics_steps"],
                "held_end": result["held_end"],
            })
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = scored[:ELITES]
        mean = np.mean([item[1] for item in elites], axis=0)
        std = np.clip(np.std([item[1] for item in elites], axis=0) + 0.08, 0.05, 0.7)
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
    return best, history


def receding_cem(env, blob, rng):
    restore_capture(env, blob["capture"])
    capture = take_capture(env)
    executed = []
    cycles = []
    physics = 0
    for cycle in range(REPLAN_CYCLES):
        best, history = cem_cycle(env, blob, capture, rng)
        physics += sum(row["physics_steps"] for row in history)
        plan = _decode(best[1])
        piece = plan[:EXECUTE_STEPS]
        stepped = play(env, blob, piece, capture=capture)
        physics += stepped["physics_steps"]
        executed.append(stepped["actions"])
        cycles.append({"cycle": cycle, "best_score": best[0], "reason": stepped["reason"], "events": stepped["events"], "samples": history})
        if stepped["reason"] in ("TABLE_DROP", "OTHER_FAILURE") or not len(stepped["actions"]):
            break
        restore_capture(env, capture)
        play(env, blob, stepped["actions"], capture=capture)
        if latched(env.sim) or not stepped["held_end"] and stepped["reason"] == "TABLE_DROP":
            break
        row = sense(env.sim)
        if row["bilateral"] and abs(row["local"][0]) < 0.015 and abs(row["local"][2] - 0.10) < 0.02:
            if float(np.linalg.norm(row["vo"] - row["vh"])) < 0.12:
                capture = take_capture(env)
                break
        capture = take_capture(env)
    actions = np.concatenate(executed, axis=0) if executed else np.zeros((0, 7), np.float32)
    final = play(env, blob, actions)
    physics += final["physics_steps"]
    final["physics_steps"] = int(physics)
    final["cycles"] = cycles
    return final


def feedback_action(sim, mode):
    row = sense(sim)
    rh = row["rh"]
    v_obj = rh.T @ row["vo"]
    v_hand = v_obj.copy()
    w_body = np.zeros(3)
    down_slip = float(row["local"][2]) > 0.108 or float(v_obj[2]) > 0.06
    side_slip = abs(float(row["local"][0])) > 0.012 or abs(float(v_obj[0])) > 0.06
    if mode == "redirect" and (row["bilateral"] or row["inside"]):
        v_hand[2] += -0.55
        v_hand[0] += float(np.clip(-4.0 * row["local"][0], -0.6, 0.6))
        w_body[2] = float(np.clip(-6.0 * row["local"][0], -W_MAX, W_MAX))
        grip = 0.4 if side_slip else 1.0
    elif mode == "regrasp" and (down_slip or not row["bilateral"]):
        if row["inside"] and row["opening"] > 0.02:
            grip = 1.0
            v_hand = v_obj
        else:
            grip = -1.0
            v_hand[2] += 0.7
        w_body[1] = float(np.clip(row["wo"][1], -1.0, 1.0))
    else:
        if row["bilateral"] or row["inside"]:
            grip = 1.0
            v_hand[0] += float(np.clip(-3.0 * row["local"][0], -0.4, 0.4))
            v_hand[2] += float(np.clip(-3.0 * (row["local"][2] - 0.10), -0.4, 0.4))
        else:
            future = predict_com(row["po"], row["vo"], 0.08)
            aim = future - rh @ PINCH_LOCAL
            v_world = row["vo"] + np.clip(6.0 * (aim - row["ph"]), -0.8, 0.8)
            v_hand = rh.T @ v_world
            grip = -1.0
    return normalize_action(v_hand, w_body, float(np.interp(grip, [-1, 1], [2.0, -18.0])), AIRBORNE_V3)


def run_feedback(env, blob, mode, seconds=1.2):
    restore_capture(env, blob["capture"])
    actions = []
    t0 = float(env.sim.data.time)
    while float(env.sim.data.time) - t0 < seconds - 1e-9:
        if latched(env.sim):
            break
        actions.append(np.asarray(feedback_action(env.sim, mode), np.float32))
        step_legal(env.sim, actions[-1])
        if len(actions) >= 200:
            break
    result = play(env, blob, actions)
    result["mode"] = mode
    return result


def maybe_nominal(env, blob, result):
    if result["reason"] == "TABLE_DROP" or not result["held_end"]:
        result["nominal"] = None
        result["success"] = False
        return result
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    for action in result["actions"]:
        if latched(env.sim):
            break
        step_legal(env.sim, action)
    pre = errors(env.sim, blob["ref"])
    pre_row = sense(env.sim)
    outcome = None
    if not latched(env.sim) and pre_row["bilateral"]:
        outcome = finish_recovery(env, blob["ref"], t0, "early")
    result["nominal"] = None if outcome is None else {
        "reason": outcome["reason"],
        "success": bool(outcome.get("success")),
        "pre_ep": float(pre["ep"]),
        "pre_eR": float(pre["eR"]),
        "pre_vrel": float(pre["vrel"]),
        "pre_nL": int(pre["nL"]),
        "pre_nR": int(pre["nR"]),
        "pre_z": float(pre["z"]),
        "pre_local": [round(float(x), 4) for x in pre_row["local"]],
        "pre_opening": round(float(pre_row["opening"]), 4),
    }
    result["success"] = bool(outcome and outcome.get("success"))
    if result["success"]:
        result["score"] = 1.0e6
    return result


def quantify(env, blob, result, book):
    if not result.get("success"):
        return None
    from training.rebuild_k48_recovery import legal_verify, legalize

    _tokens, _residuals, decoded, packed = legalize(result["actions"], book)
    legal = packed["fraction_within_0.25"] is not None and float(packed["max_linf"]) <= 0.25 + 1e-8
    verified = None
    if legal:
        verified, _o, _e, _c, qpos, phase = legal_verify(env, blob, decoded)
    return {
        "representable": bool(legal),
        "max_linf": packed["max_linf"],
        "fraction_within_0.25": packed["fraction_within_0.25"],
        "legal_verify_reason": None if verified is None else verified["reason"],
        "legal_verify_success": bool(verified and verified.get("success") and qpos <= 1e-6 and phase),
    }


def choose(env):
    rows = []
    for line in (ROOT / "data" / "recatch" / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("split") == "train":
                rows.append(row)
    by_id = {}
    for row in rows:
        by_id.setdefault(row["id"], row)
    sample = [rows[index] for index in range(0, len(rows), max(1, len(rows) // 18))]
    for name in ("ball_train_v2_0051", "ball_train_v2_0226", "k48r_train_00135", "k48r_train_00150", "ball_train_v2_0098"):
        if name in by_id:
            sample.append(by_id[name])
    seen = set()
    measured = []
    for row in sample:
        if row["id"] in seen or not (ROOT / row["snapshot"]).is_file():
            continue
        seen.add(row["id"])
        blob = _load(row["snapshot"])
        restore_capture(env, blob["capture"])
        state = sense(env.sim)
        hand_v = state["rh"].T @ state["vo"]
        measured.append({
            "row": row,
            "blob": blob,
            "side": abs(float(hand_v[0])) + 4.0 * abs(float(state["local"][0])),
            "down": max(0.0, float(hand_v[2])) + 6.0 * max(0.0, float(state["local"][2]) - 0.105),
            "axis_z": abs(float(state["axis"][2])),
            "spin": float(np.linalg.norm(state["wo"])),
            "hand_v": [round(float(x), 4) for x in hand_v],
            "local": [round(float(x), 4) for x in state["local"]],
            "z": round(float(state["z"]), 4),
            "nL": state["nL"],
            "nR": state["nR"],
        })
    named = [item for item in measured if item["row"]["id"] in ("ball_train_v2_0051", "ball_train_v2_0226", "k48r_train_00135")]
    others = [item for item in measured if item not in named]
    side = max(others, key=lambda item: item["side"])
    down = max(others, key=lambda item: item["down"])
    chosen = named + [side, down]
    unique = []
    ids = set()
    for item in chosen:
        if item["row"]["id"] not in ids:
            ids.add(item["row"]["id"])
            unique.append(item)
    handoff = [item for item in measured if item["row"]["id"] in ("k48r_train_00150", "ball_train_v2_0098")]
    return unique, handoff


def _public(result):
    return {
        "reason": result["reason"],
        "score": result["score"],
        "events": result["events"],
        "physics_steps": result["physics_steps"],
        "held_end": result["held_end"],
        "success": bool(result.get("success")),
        "nominal": result.get("nominal"),
        "quantized": result.get("quantized"),
        "steps": int(len(result["actions"])),
    }


def _save(case_id, method, result):
    TRAJ.mkdir(parents=True, exist_ok=True)
    path = TRAJ / f"{case_id}__{method}.json"
    path.write_text(json.dumps({
        "events": result["events"],
        "reason": result["reason"],
        "nominal": result.get("nominal"),
        "trace": result["trace"],
        "actions": np.asarray(result["actions"]).round(5).tolist(),
    }), encoding="utf-8")
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _forces(env, blob, actions):
    restore_capture(env, blob["capture"])
    rows = []
    for action in actions[:40]:
        rows.append(raw_object_contacts(env.sim))
        if latched(env.sim):
            break
        step_legal(env.sim, action)
    peak = {"finger": 0.0, "table": 0.0}
    for contacts in rows:
        for item in contacts:
            if "finger" in item["kind"]:
                peak["finger"] = max(peak["finger"], item["fn"])
            if item["kind"] in ("object-table", "object-floor"):
                peak["table"] = max(peak["table"], item["fn"])
    return {key: round(value, 3) for key, value in peak.items()}


def main():
    import torch
    from training.airborne_intercept_recatch import rollout as old_rollout
    from training.autonomous_episode import make_training_env
    from training.unified_k47 import load_codebook
    from training.unified_k48 import load_trained_k48

    RAW.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cpu")
    model, _saved = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", device)
    model.eval()
    env = make_training_env(device, model=model)
    book = load_codebook()
    chosen, handoff = choose(env)
    probe = capability(env, chosen[0]["blob"])
    (RAW / "capability.json").write_text(json.dumps({key: {k: v for k, v in value.items() if k != "series"} | {"series_len": len(value["series"]), "speed_trace": [row["actual_speed"] for row in value["series"]]} for key, value in probe.items()}, indent=2), encoding="utf-8")
    rng = np.random.default_rng(7)
    lines = []
    champions = []
    for item in chosen:
        case_id = item["row"]["id"]
        blob = item["blob"]
        print("IC", case_id, "side", round(item["side"], 3), "down", round(item["down"], 3), "axis_z", round(item["axis_z"], 3), "hand_v", item["hand_v"], flush=True)
        old = old_rollout(env, blob, {"mode": "closed", "lead": 0.0, "match": 1.0, "gain": 0.0, "align": False, "held_grip": 1.0, "held_speed": 0.03}, verify_nominal=False)
        old_play = play(env, blob, old["actions"])
        old_play["success"] = False
        old_play["nominal"] = None
        geometry = {key: value for key, value in item.items() if key not in ("row", "blob")}
        record_a = {"id": case_id, "method": "A_old_tracking", "geometry": geometry, **_public(old_play)}
        record_a["trace_path"] = _save(case_id, "A_old_tracking", old_play)
        lines.append(record_a)
        print("  A", old_play["reason"], old_play["events"], "score", round(old_play["score"], 1), flush=True)
        best_b = None
        for mode in ("redirect", "regrasp", "combined"):
            trial = run_feedback(env, blob, mode)
            trial = maybe_nominal(env, blob, trial)
            trial["quantized"] = quantify(env, blob, trial, book)
            record = {"id": case_id, "method": f"B_{mode}", "geometry": geometry, **_public(trial)}
            record["trace_path"] = _save(case_id, f"B_{mode}", trial)
            record["peak_force_n"] = _forces(env, blob, trial["actions"])
            lines.append(record)
            print("  B", mode, trial["reason"], trial["events"], "held", trial["held_end"], "success", trial["success"], flush=True)
            if best_b is None or trial["score"] > best_b["score"]:
                best_b = trial
                best_b["method"] = f"B_{mode}"
        cem = receding_cem(env, blob, rng)
        cem = maybe_nominal(env, blob, cem)
        cem["quantized"] = quantify(env, blob, cem, book)
        record_c = {"id": case_id, "method": "C_cem", "geometry": geometry, **_public(cem)}
        record_c["trace_path"] = _save(case_id, "C_cem", cem)
        record_c["peak_force_n"] = _forces(env, blob, cem["actions"])
        record_c["cem_cycles"] = [{"cycle": row["cycle"], "best_score": row["best_score"], "reason": row["reason"], "events": row["events"]} for row in cem.get("cycles", [])]
        lines.append(record_c)
        print("  C", cem["reason"], cem["events"], "score", round(cem["score"], 1), "success", cem["success"], "phys", cem["physics_steps"], flush=True)
        champions.append((item, best_b, cem))
        (RAW / "candidates.jsonl").write_text("".join(json.dumps({k: v for k, v in row.items() if k != "geometry" or True}) + "\n" for row in lines), encoding="utf-8")
    handoff_rows = []
    for item in handoff:
        trial = run_feedback(env, blob := item["blob"], "combined", seconds=4.0)
        trial = maybe_nominal(env, blob, trial)
        handoff_rows.append({"id": item["row"]["id"], **_public(trial), "trace_path": _save(item["row"]["id"], "handoff_probe", trial)})
        print("HANDOFF", item["row"]["id"], trial.get("nominal"), trial["events"], flush=True)
    (RAW / "handoff.json").write_text(json.dumps(handoff_rows, indent=2), encoding="utf-8")
    from training.airborne_intercept_recatch import render_case

    videos = []
    for item, best_b, cem in champions[:4]:
        dest = VIDEO / f"{item['row']['id']}__{best_b['method']}.mp4"
        videos.append(render_case(env, item["blob"], best_b["actions"], dest, best_b["method"]))
    summary = {
        "seconds": round(time.perf_counter() - started, 1),
        "ics": [item["row"]["id"] for item, _b, _c in champions],
        "physics_steps": int(sum(row["physics_steps"] for row in lines)),
        "success_continuous": sum(1 for row in lines if row["success"]),
        "success_k48": sum(1 for row in lines if row.get("quantized") and row["quantized"].get("legal_verify_success")),
        "videos": videos,
    }
    (RAW / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("CONTACT_SEARCH_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
