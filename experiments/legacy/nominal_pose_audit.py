"""Nominal-pose return and long-horizon handoff audit.

Does not train and does not change the frozen geometry exit. The basin used
here is computed from an undisturbed nominal hold, then applied only inside
this study.
"""

from __future__ import annotations


import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from controllers.cartesian.tracking import TRACK
from envs.simulation.support_contacts import restore_supports, supports_of
from envs.initialization.grasp_seat import apply_spec
import models.airborne_v3 as v3
from controllers.recovery.geometry_switch import GeometrySwitch, hand_pose, relative_speed, rot_angle
from controllers.recovery.action_map import AIRBORNE_V2, AIRBORNE_V3, normalize_action
from envs.outcomes.physical_recovery import physical_pack
from training.recovery_episode import T_TASK, Z_ABORT, ball_substep, empty_ball, fresh_nominal_state, make_training_env, nominal_substep, seat_base

from envs.ballistic_impact import TAU_SEC, z_tgt_of
from models.historical.k33_checkpoint import load_policy
from data_generation.airborne_recatch_dataset import drop_supports, restore_to, step_mapped

from controllers.recovery.skill_schedules import schedules
from training.recovery_execution import TABLE_DROP, _escaped
from controllers.cartesian.velocity_step import tick_vw

RAW = ROOT / "results" / "diagnostics" / "raw" / "nominal_pose_audit"
SPEC_ZP6 = {"v_mps": 6.5, "z_off_m": 0.006, "phi_deg": 0.0}
HORIZON = 12.0
PERSIST = 5
SPEED_CAP = 0.08
SECURE_TAU = -18.0


from envs.outcomes.nominal_pose import capture_ref, closed_grip, errors, hand_R, rotvec


def in_basin(sim, ref, tol, grip_action=None):
    row = errors(sim, ref)
    grip_ok = closed_grip(sim) or (grip_action is not None and float(grip_action) > 0.5)
    ok = (
        row["nL"] > 0 and row["nR"] > 0
        and row["eR"] <= tol["ori"] and row["ep"] <= tol["pos"]
        and row["vrel"] <= tol["speed"] and grip_ok
    )
    return ok, row


def torch_tensor(array, device):
    import torch
    return torch.tensor(array, device=device)


def policy_action(env):
    import torch
    obs = env.observation()
    env._obs_seq.append(np.asarray(obs, np.float32))
    env._prev_seq.append(env._prev.copy())
    with torch.no_grad():
        logits, _res, act = env.model(
            torch_tensor(np.stack(env._obs_seq[-160:]), env.device),
            torch_tensor(np.stack(env._prev_seq[-160:]), env.device),
        )
    token = int(np.argmax(logits[-1].detach().float().cpu().numpy()))
    clipped = np.clip(act[-1].detach().float().cpu().numpy(), -1.0, 1.0).astype(np.float32)
    v3.step_v3(env.sim, clipped)
    env._prev = clipped
    return token, clipped


def nominal_step(sim):
    """Same nominal continuation as continue_long, without an absolute t=12 stop."""
    z_now = float(z_tgt_of(sim))
    lift_hold = str(sim.fsm.phase) == "lift" and float(sim.fsm.p_des[2]) >= z_now - 1e-9
    if lift_hold:
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, TRACK)
    else:
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()


def nominal_hold(env, ref, seconds, restore_orientation):
    sim = env.sim
    before = errors(sim, ref)
    if restore_orientation:
        sim.fsm.r_des = np.array(ref["r_des"], float).copy()
        sim.fsm.w_cmd = np.zeros(3)
        sim.fsm.w_des = np.zeros(3)
        sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
        sim.fsm.v_cmd = np.zeros(3)
        sim.fsm.v_des = np.zeros(3)
    after = errors(sim, ref)
    t0 = float(sim.data.time)
    lost = 0
    drop_t = None
    reason = None
    samples = []
    next_log = t0
    while float(sim.data.time) < t0 + seconds - 1e-9:
        nominal_step(sim)
        row = errors(sim, ref)
        if row["t"] + 1e-9 >= next_log:
            samples.append({k: row[k] for k in ("t", "eR", "ep", "vrel", "nL", "nR", "z")})
            next_log += 0.5
        both_off = row["nL"] == 0 and row["nR"] == 0
        lost = lost + 1 if both_off else 0
        if row["z"] < TABLE_DROP or row["z"] < Z_ABORT:
            drop_t = row["t"]
            reason = "drop"
            break
        if _escaped(sim):
            drop_t = row["t"]
            reason = "escape"
            break
        if lost >= 75:
            drop_t = row["t"]
            reason = "bilateral_lost"
            break
    t_end = float(sim.data.time)
    return {
        "t_handoff": t0,
        "before": before,
        "after": after,
        "drop_t": drop_t,
        "reason": reason,
        "held_10": bool(drop_t is None and t_end >= t0 + 10.0 - 1e-6),
        "held_12": bool(drop_t is None and t_end >= t0 + HORIZON - 1e-6),
        "samples": samples[-8:],
        "end": errors(sim, ref),
    }


def classify(recatch, pose, handed, hold):
    if not recatch:
        return "DROP_DURING_RECOVERY"
    if handed and not pose:
        return "BAD_ORIENTATION_AT_HANDOFF"
    if not pose:
        return "RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED"
    if not handed or hold is None:
        return "NO_NOMINAL_HANDOFF"
    if hold["held_12"]:
        return "SUCCESS_12S"
    if hold["held_10"]:
        return "SUCCESS_10S"
    if hold["drop_t"] is not None:
        delay = float(hold["drop_t"]) - float(hold["t_handoff"])
        return "DELAYED_DROP_AFTER_HANDOFF" if delay >= 5.0 else "DROP_AFTER_HANDOFF"
    return "NOMINAL_CONTROLLER_CANNOT_HOLD_RESTORED_GRASP"


def orient_command(sim, ref, correct_position):
    Rd = np.array(sim.fsm.r_des, float).reshape(3, 3)
    w_world = rotvec(ref["R_hand"] @ Rd.T)
    mag = float(np.linalg.norm(w_world))
    if mag > 1e-8:
        w_world = w_world / mag * min(2.0 * mag, 1.2)
    v_hand = np.zeros(3)
    if correct_position:
        p, _rel = hand_pose(sim)
        v_hand = np.clip(-6.0 * (p - ref["p_nom"]), -0.20, 0.20)
    return normalize_action(v_hand, Rd.T @ w_world, SECURE_TAU, AIRBORNE_V3)


def servo(env, ref, tol, correct_position, limit_s=4.0):
    t0 = float(env.sim.data.time)
    run = 0
    trace = []
    last = None
    while float(env.sim.data.time) < t0 + limit_s - 1e-9:
        act = orient_command(env.sim, ref, correct_position)
        v3.step_v3(env.sim, np.asarray(act, np.float32))
        ok, row = in_basin(env.sim, ref, tol, grip_action=1.0)
        last = row
        run = run + 1 if ok else 0
        if len(trace) < 6 or run == PERSIST:
            trace.append({k: row[k] for k in ("t", "eR", "ep", "vrel", "nL", "nR")})
        if run >= PERSIST:
            return True, row, trace
        if row["z"] < Z_ABORT:
            return False, row, trace
    return False, last, trace


def basin_ok(row, tol, grip):
    return (
        row["nL"] > 0 and row["nR"] > 0 and row["eR"] <= tol["ori"]
        and row["ep"] <= tol["pos"] and row["vrel"] <= tol["speed"] and float(grip) > 0.5
    )


def old_exit_ok(row):
    return (
        row["nL"] > 0 and row["nR"] > 0 and row["ep"] <= 0.0147
        and row["eR_rel"] <= 0.35 and row["vrel"] <= 0.08
    )


def roll_policy(env, ref, tol, cap, stop_on_basin=False):
    env.start_recovery()
    rows = []
    seen_loss = False
    stable_run = 0
    basin_run = 0
    old_run = 0
    fork = None
    basin = None
    old_exit = None
    fell = False
    for _ in range(cap):
        token, act = policy_action(env)
        row = errors(env.sim, ref)
        row["token"] = token
        row["action"] = [round(float(x), 4) for x in act]
        rows.append(row)
        if row["nL"] == 0 and row["nR"] == 0:
            seen_loss = True
            stable_run = 0
        elif seen_loss and float(act[6]) > 0.5 and row["nL"] > 0 and row["nR"] > 0:
            stable_run += 1
            if stable_run >= PERSIST and fork is None:
                fork = take_capture(env)
                fork["t"] = row["t"]
                fork["row"] = slim(row)
        else:
            stable_run = 0
        basin_run = basin_run + 1 if basin_ok(row, tol, act[6]) else 0
        if basin_run >= PERSIST and basin is None:
            basin = take_capture(env)
            basin["t"] = row["t"]
            basin["row"] = slim(row)
            if stop_on_basin:
                break
        old_run = old_run + 1 if old_exit_ok(row) else 0
        if old_run >= PERSIST and old_exit is None:
            old_exit = slim(row)
        if row["z"] < Z_ABORT:
            fell = True
            break
    held = (not fell) and bool(rows) and rows[-1]["nL"] > 0 and rows[-1]["nR"] > 0
    return {
        "rows": rows, "fork": fork, "basin": basin, "old_exit": old_exit,
        "fell": fell, "recatch": bool(held or basin is not None or fork is not None),
    }


def until_impact(env, spec):
    env.begin_nominal()
    ball = empty_ball()
    switch = env.switch
    ref = None
    while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            ball_substep(env.sim, env.gains, env.nominal, ball, spec["v_mps"], spec["z_off_m"], spec["phi_deg"], None)
            if float(env.sim.data.time) >= T_TASK - 1e-9:
                break
        fresh = switch.p0 is None
        row = switch.measure(env.sim)
        if fresh and switch.p0 is not None:
            ref = capture_ref(env.sim, switch)
        if ball["released"]:
            if ref is None or not ref.get("live"):
                ref = capture_ref(env.sim, switch, live=True)
                ref["live"] = True
            switch.update_entry(env.sim)
        if ball["released"] and row["z"] < Z_ABORT:
            break
        if (not ball["released"]) and float(env.sim.data.time) > 8.0:
            break
    return ref, ball


def undisturbed(env):
    env.begin_nominal()
    st = fresh_nominal_state()
    ref = None
    samples = []
    lift = []
    hold_t0 = None
    while float(env.sim.data.time) < 12.0:
        for _ in range(v3.N_SUB):
            nominal_substep(env.sim, env.gains, st)
        fresh = env.switch.p0 is None
        env.switch.measure(env.sim)
        if fresh and env.switch.p0 is not None:
            ref = capture_ref(env.sim, env.switch)
        if ref is None:
            continue
        row = errors(env.sim, ref)
        if not st["holding"]:
            lift.append(row)
            continue
        if hold_t0 is None:
            hold_t0 = float(env.sim.data.time)
            ref = capture_ref(env.sim, env.switch, live=True)
            continue
        if float(env.sim.data.time) >= hold_t0 + 0.05:
            samples.append(row)
        if float(env.sim.data.time) >= hold_t0 + 2.05:
            break
    if not samples:
        raise RuntimeError("undisturbed nominal hold produced no samples")
    eR = np.array([s["eR"] for s in samples])
    ep = np.array([s["ep"] for s in samples])
    vv = np.array([s["vrel"] for s in samples])
    rel = np.array([s["eR_rel"] for s in samples])
    lift_eR = np.array([s["eR"] for s in lift]) if lift else np.zeros(1)
    lift_ep = np.array([s["ep"] for s in lift]) if lift else np.zeros(1)
    return {
        "t_ref": ref["t"],
        "t_hold": hold_t0,
        "n": int(len(samples)),
        "n_lift": int(len(lift)),
        "lift_eR_max": float(lift_eR.max()),
        "lift_ep_max": float(lift_ep.max()),
        "eR_max": float(eR.max()),
        "eR_p95": float(np.quantile(eR, 0.95)),
        "eR_rel_max": float(rel.max()),
        "ep_max": float(ep.max()),
        "ep_p95": float(np.quantile(ep, 0.95)),
        "v_max": float(vv.max()),
        "p_nom": [round(float(x), 5) for x in ref["p_nom"]],
    }


def choose_tol(nom, sideways):
    ori = float(nom["eR_max"] + max(0.02, 3.0 * nom["eR_max"]))
    pos = float(nom["ep_max"] + max(0.003, 3.0 * nom["ep_max"]))
    if sideways is not None and sideways > 0.20 and ori > 0.5 * sideways:
        ori = float(0.5 * sideways)
    return {"ori": ori, "pos": pos, "speed": float(min(SPEED_CAP, max(0.02, 5.0 * nom["v_max"]))), "persist_steps": PERSIST}


def slim(row):
    keep = ("t", "eR", "eR_rel", "eR_des", "ep", "vrel", "nL", "nR", "z", "token", "action")
    return {k: row[k] for k in keep if k in row}


def hold_brief(hold):
    if hold is None:
        return None
    return {k: hold[k] for k in ("t_handoff", "held_10", "held_12", "drop_t", "reason", "before", "after", "end")}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_training_env()
    table = supports_of(env.sim)
    model, _proto, cap = load_policy(device)
    env.model = model.to(env.device)
    env.model.eval()
    print("NOMINAL", flush=True)
    nom = undisturbed(env)
    print(nom, flush=True)
    restore_supports(env.sim, table)
    print("ZP6", flush=True)
    ref, ball = until_impact(env, SPEC_ZP6)
    provisional = choose_tol(nom, None)
    roll = roll_policy(env, ref, provisional, 300, stop_on_basin=False)
    rows = roll["rows"]
    sideways = rows[-1]["eR"]
    tol = choose_tol(nom, sideways)
    print("TOL", tol, "sideways", round(sideways, 4), flush=True)
    run = 0
    earliest = None
    grow = None
    for row in rows:
        ok = (
            row["nL"] > 0 and row["nR"] > 0 and row["eR"] <= tol["ori"]
            and row["ep"] <= tol["pos"] and row["vrel"] <= tol["speed"] and row["action"][6] > 0.5
        )
        run = run + 1 if ok else 0
        if run >= PERSIST and earliest is None:
            earliest = row["t"]
        if grow is None and row["eR"] > max(0.2, 2.0 * tol["ori"]):
            grow = {"t": row["t"], "eR": row["eR"], "token": row["token"], "action": row["action"]}
    end = rows[-1]
    pose_cap = end["eR"] <= tol["ori"] and end["ep"] <= tol["pos"] and end["nL"] > 0 and end["nR"] > 0
    print("fork", None if roll["fork"] is None else roll["fork"]["row"], "earliest", earliest, "grow", grow, "eR_cap", round(end["eR"], 3), flush=True)
    print("BRANCH A hold-at-cap without restoring r_des", flush=True)
    hold_a = None
    if end["z"] >= Z_ABORT and end["nL"] > 0 and end["nR"] > 0:
        hold_a = nominal_hold(env, ref, HORIZON, restore_orientation=False)
    branches = {
        "A_current": {
            "pose_restored_at_cap": bool(pose_cap),
            "eR": end["eR"],
            "ep": end["ep"],
            "eR_des": end["eR_des"],
            "eR_rel": end["eR_rel"],
            "token": end["token"],
            "action": end["action"],
            "hold_if_forced": hold_brief(hold_a),
            "label": "RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED" if roll["recatch"] and not pose_cap else classify(roll["recatch"], pose_cap, pose_cap, hold_a),
        }
    }
    print(branches["A_current"]["label"], flush=True)

    def branch_from_fork(name, mode):
        print("BRANCH", name, flush=True)
        restore_capture(env, roll["fork"])
        if mode == "nominal":
            hold = nominal_hold(env, ref, HORIZON, restore_orientation=False)
            out = {
                "eR_at_handoff": hold["before"]["eR"],
                "ep_at_handoff": hold["before"]["ep"],
                "eR_des_before": hold["before"]["eR_des"],
                "hold": hold_brief(hold),
                "label": classify(True, hold["before"]["eR"] <= tol["ori"] and hold["before"]["ep"] <= tol["pos"], True, hold),
            }
        else:
            ok, row, trace = servo(env, ref, tol, correct_position=(mode == "privileged"))
            hold = nominal_hold(env, ref, HORIZON, restore_orientation=True) if ok else None
            out = {
                "reached": bool(ok),
                "trace": trace,
                "row": None if row is None else slim(row),
                "hold": hold_brief(hold),
                "label": classify(True, ok, ok, hold),
            }
        print(out["label"], flush=True)
        return out

    if roll["fork"] is None:
        branches["B_nominal_now"] = {"label": "DROP_DURING_RECOVERY"}
        branches["C_return"] = {"label": "DROP_DURING_RECOVERY"}
        branches["D_privileged"] = {"label": "DROP_DURING_RECOVERY"}
    else:
        branches["B_nominal_now"] = branch_from_fork("B", "nominal")
        branches["C_return"] = branch_from_fork("C", "orient")
        branches["D_privileged"] = branch_from_fork("D", "privileged")

    loss = next((r for r in rows if r["nL"] == 0 and r["nR"] == 0), None)
    recatch_row = None
    reclose_row = None
    if loss is not None:
        for r in rows:
            if r["t"] < loss["t"]:
                continue
            if recatch_row is None and r["nL"] > 0 and r["nR"] > 0:
                recatch_row = slim(r)
            if reclose_row is None and r["action"][6] > 0.5:
                reclose_row = slim(r)
    markers = {
        "impact": ball["t_impact"],
        "entry": env.switch.t_entry,
        "release": None if loss is None else loss["t"],
        "recatch": None if recatch_row is None else recatch_row["t"],
        "reclose": None if reclose_row is None else reclose_row["t"],
        "stable_bilateral": None if roll["fork"] is None else roll["fork"]["t"],
        "rotation_onset": grow,
        "step_cap": end["t"],
        "old_exit": roll["old_exit"],
    }

    def arm_reference(env_):
        env_.switch.capture_if_ready(env_.sim)
        if env_.switch.p0 is None:
            env_.switch.p0, env_.switch.r0 = hand_pose(env_.sim)
        return capture_ref(env_.sim, env_.switch)

    def enter_offset(spec):
        info = restore_to(env.sim, base, env.nominal, env.pads, env.pad_mu)
        drop_supports(env.sim)
        if not info.get("pose_ok", False):
            raise RuntimeError("offset restore failed")
        env.switch = GeometrySwitch()
        ref_case = arm_reference(env)
        apply_spec(env.sim, spec)
        st = fresh_nominal_state()
        while float(env.sim.data.time) < T_TASK - 1e-9 and not env.switch.entered:
            for _ in range(v3.N_SUB):
                nominal_substep(env.sim, TRACK, st)
            row = env.switch.measure(env.sim)
            env.switch.update_entry(env.sim)
            if row["z"] < Z_ABORT:
                break
        return ref_case, bool(env.switch.entered)

    def enter_air(spec):
        info = restore_to(env.sim, base, env.nominal, env.pads, env.pad_mu)
        drop_supports(env.sim)
        if not info.get("pose_ok", False):
            raise RuntimeError("carry restore failed")
        env.switch = GeometrySwitch()
        ref_case = arm_reference(env)
        t0 = float(env.sim.data.time)
        open_at = t0 + float(spec["duration_s"])
        opened = False
        open_hold = 0
        st = fresh_nominal_state()
        while float(env.sim.data.time) < T_TASK - 1e-9 and not env.switch.entered:
            for _ in range(v3.N_SUB):
                t = float(env.sim.data.time)
                if open_hold < 10 and t < open_at + 0.20:
                    v_h = np.zeros(3)
                    if t < open_at:
                        v_h[1] = float(spec["sign"]) * float(spec["speed"])
                        tau = -18.0
                    else:
                        opened = True
                        v_h[2] = 0.3
                        tau = 2.0
                    action = normalize_action(v_h, np.zeros(3), tau, AIRBORNE_V2)
                    step_mapped(env.sim, action, AIRBORNE_V2, TRACK, q_ref)
                    pack = physical_pack(env.sim)
                    if opened and int(pack["nL"]) == 0 and int(pack["nR"]) == 0:
                        open_hold += 1
                else:
                    nominal_substep(env.sim, TRACK, st)
            if env.switch.update_entry(env.sim):
                break
        return ref_case, bool(env.switch.entered)

    catalog = []

    def finish_case(name, kind, ref_case, out):
        basin = out["basin"]
        end_row = None if not out["rows"] else out["rows"][-1]
        pose = basin is not None
        hold = None
        row = basin["row"] if pose else end_row
        if pose:
            hold = nominal_hold(env, ref_case, HORIZON, restore_orientation=True)
        physical = bool(out["recatch"] and not out["fell"])
        label = classify(physical or pose, pose, pose, hold)
        catalog.append({
            "case": name,
            "kind": kind,
            "recatch": physical or pose,
            "pose": bool(pose),
            "eR": None if row is None else row["eR"],
            "ep": None if row is None else row["ep"],
            "t_recatch": None if out["fork"] is None else out["fork"]["t"],
            "t_recovery_end": None if row is None else row["t"],
            "old_exit_eR": None if out["old_exit"] is None else out["old_exit"]["eR"],
            "old_exit_ep": None if out["old_exit"] is None else out["old_exit"]["ep"],
            "handoff": None if hold is None else hold["t_handoff"],
            "held_10": None if hold is None else hold["held_10"],
            "held_12": None if hold is None else hold["held_12"],
            "drop_t": None if hold is None else hold["drop_t"],
            "label": label,
        })
        print(name, label, flush=True)

    def eval_impact(name, spec):
        restore_supports(env.sim, table)
        ref_case, _b = until_impact(env, spec)
        if ref_case is None or not env.switch.entered:
            catalog.append({"case": name, "kind": "impact", "recatch": False, "pose": False, "label": "DROP_DURING_RECOVERY"})
            print(name, "no entry", flush=True)
            return
        out = roll_policy(env, ref_case, tol, 300, stop_on_basin=True)
        finish_case(name, "impact", ref_case, out)

    def eval_air(name, speed):
        ref_case, entered = enter_air({"kind": "air", "speed": speed, "sign": 1.0, "duration_s": 0.08})
        if not entered:
            catalog.append({"case": name, "kind": "air", "recatch": False, "pose": False, "label": "DROP_DURING_RECOVERY"})
            print(name, "no entry", flush=True)
            return
        env.start_recovery()
        out = roll_policy(env, ref_case, tol, 80, stop_on_basin=True)
        finish_case(name, "air", ref_case, out)

    def eval_offset(name, dz, dx, dy, dvz):
        ref_case, entered = enter_offset({"kind": "offset", "dz": dz, "dx": dx, "dy": dy, "dvz": dvz})
        if not entered:
            catalog.append({"case": name, "kind": "offset", "recatch": False, "pose": False, "label": "DROP_DURING_RECOVERY"})
            print(name, "no entry", flush=True)
            return
        env.start_recovery()
        out = roll_policy(env, ref_case, tol, 80, stop_on_basin=True)
        finish_case(name, "offset", ref_case, out)

    def eval_skill(name):
        restore_supports(env.sim, table)
        ref_case, _b = until_impact(env, {"v_mps": 6.5, "z_off_m": -0.006, "phi_deg": 0.0})
        if ref_case is None or not env.switch.entered:
            catalog.append({"case": name, "kind": "skill", "recatch": False, "pose": False, "label": "DROP_DURING_RECOVERY"})
            return
        env.start_recovery()
        fell = False
        end_row = None
        run = 0
        basin = None
        for act in schedules()[name]:
            a3 = v3.v1_normalized_to_v3(np.asarray(act, np.float32))
            v3.step_v3(env.sim, a3)
            end_row = errors(env.sim, ref_case)
            end_row["action"] = [round(float(x), 4) for x in a3]
            run = run + 1 if basin_ok(end_row, tol, a3[6]) else 0
            if run >= PERSIST and basin is None:
                basin = {"t": end_row["t"], "row": slim(end_row)}
                break
            if end_row["z"] < Z_ABORT:
                fell = True
                break
        physical = end_row is not None and end_row["nL"] > 0 and end_row["nR"] > 0 and not fell
        out = {
            "rows": [] if end_row is None else [end_row],
            "fork": None if not physical else {"t": end_row["t"]},
            "basin": basin,
            "old_exit": None,
            "fell": fell,
            "recatch": physical or basin is not None,
        }
        finish_case(name, "skill", ref_case, out)

    print("REEVAL", flush=True)
    for name, spec in (
        ("impact_zm6", {"v_mps": 6.5, "z_off_m": -0.006, "phi_deg": 0.0}),
        ("impact_z3", {"v_mps": 6.5, "z_off_m": -0.003, "phi_deg": 0.0}),
        ("impact_zp6", SPEC_ZP6),
    ):
        eval_impact(name, spec)
    for name in ("wrist_reseat", "gravity_inward", "recatch_open20"):
        eval_skill(name)
    print("SEAT", flush=True)
    base, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    for name, speed in (("air_lat_pos", 0.30), ("air_fast_pos", 0.45)):
        eval_air(name, speed)
    for name, dz, dx, dy, dvz in (
        ("train_dz13.5", -13.5, 0.0, 0.0, 0.0),
        ("nl_dz14.6", -14.6, 0.0, 0.0, 0.0),
        ("nl_dvz_p", -13.5, 0.0, 0.0, 0.005),
        ("dz13.2", -13.2, 0.0, 0.0, 0.0),
        ("dz14.2", -14.2, 0.0, 0.0, 0.0),
        ("dvz_m", -13.5, 0.0, 0.0, -0.01),
        ("h_dz14.00", -14.00, 0.0, 0.0, 0.0),
        ("h_dz14.25", -14.25, 0.0, 0.0, 0.0),
        ("h_dy", -14.10, 0.0, 0.12, 0.0),
        ("h_dx", -14.70, 0.10, 0.0, 0.0),
    ):
        eval_offset(name, dz, dx, dy, dvz)

    payload = {
        "checkpoint": "results/archive/demo_results/diagnostics/raw/action_representation/ckpt_lateral_token.pt",
        "residual_cap": cap,
        "nominal": nom,
        "tolerance": tol,
        "zp6_steps": len(rows),
        "zp6_markers": markers,
        "zp6_fork_t": None if roll["fork"] is None else roll["fork"]["t"],
        "zp6_fork_row": None if roll["fork"] is None else roll["fork"]["row"],
        "zp6_earliest_basin_t": earliest,
        "zp6_rotation_onset": grow,
        "zp6_trace": [slim(r) for i, r in enumerate(rows) if i < 12 or i % 20 == 0 or i == len(rows) - 1],
        "branches": branches,
        "catalog": catalog,
    }
    text = json.dumps(payload)
    (RAW / "audit.json").write_text(text, encoding="utf-8")
    print("WROTE", RAW / "audit.json", flush=True)


if __name__ == "__main__":
    main()
