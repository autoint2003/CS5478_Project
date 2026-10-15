"""Causal audit of the 40 degree ball-impact recovery. Does not change controllers."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.geometry_switch import rot_angle
from controllers.recovery_actions import AIRBORNE_V3, map_action
from envs.airborne_obs import relative_state, rotvec
from envs.deterioration import body_twist
from training.ball_impact_recoverability import WIN, _cell, _fly, _oracle, _setup
from training.capture_geometry import capture_geometry
from training.horizon_outcome import NOMINAL_S, nominal_verify
from training.legal_recatch_transfer import _bottom
from training.nominal_pose_audit import hand_R
from training.privileged_ball_impact_search import step_legal
from training.recovery_ic_expansion import _table_touch
from training.recovery_runtime import make_snap, restore_ballistic
from training.recovery_success import apply_handoff, nominal_step
from training.replay_core import TABLE_TOP
from training.table_drop import latched, restore_state

OUT = ROOT / "results" / "diagnostics" / "ball_impact_handoff_causal_audit"
SRC = ROOT / "results" / "diagnostics" / "ball_impact_recoverability" / "success"


def _force(sim):
    import mujoco

    obj = int(sim.ids.object_body)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    normal = tangent = 0.0
    horizontal = np.zeros(2)
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        body_1 = int(sim.model.geom_bodyid[int(contact.geom1)])
        body_2 = int(sim.model.geom_bodyid[int(contact.geom2)])
        if obj not in (body_1, body_2):
            continue
        other = body_2 if body_1 == obj else body_1
        if other not in (left, right):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, index, wrench)
        normal += abs(float(wrench[0]))
        tangent += float(np.linalg.norm(wrench[1:3]))
        frame = np.asarray(contact.frame, float).reshape(3, 3)
        force = frame[0] * wrench[0] + frame[1] * wrench[1] + frame[2] * wrench[2]
        horizontal += force[:2]
    limits = np.asarray(sim.ids.ctrl_high[:7], float)
    tau = np.asarray(sim.data.ctrl[:7], float)
    saturated = int(np.sum(np.abs(tau) >= 0.98 * np.abs(limits)))
    return normal, tangent, horizontal, tau, saturated


def _obs(sim, phase):
    rel = relative_state(sim.model, sim.data, sim.ids)
    geo = capture_geometry(sim)
    _linear, angular = body_twist(sim.model, sim.data, sim.ids.object_body)
    normal, tangent, horizontal, tau, saturated = _force(sim)
    p_des = np.asarray(sim.fsm.p_des, float).reshape(-1)[:3]
    v_cmd = np.asarray(sim.fsm.v_cmd, float).reshape(3)
    v_des = np.asarray(sim.fsm.v_des, float).reshape(3)
    hand = np.asarray(rel["p_hand_w"], float)
    return {
        "t": float(sim.data.time),
        "phase": phase,
        "fsm": str(sim.fsm.phase),
        "p_hand": hand.copy(),
        "p_obj": np.asarray(rel["p_obj_w"], float).copy(),
        "v_hand": np.asarray(rel["v_hand_w"], float).copy(),
        "v_obj": np.asarray(rel["v_obj_w"], float).copy(),
        "p_rel": np.asarray(rel["p_rel_h"], float).copy(),
        "v_rel": np.asarray(rel["v_rel_h"], float).copy(),
        "w_obj": np.asarray(angular, float).copy(),
        "R_obj": np.array(sim.data.xmat[sim.ids.object_body], float).reshape(3, 3).copy(),
        "nL": int(geo["nL"]),
        "nR": int(geo["nR"]),
        "bottom": float(_bottom(sim)),
        "fn": float(normal),
        "ft": float(tangent),
        "fxy": horizontal.copy(),
        "p_err": hand - p_des,
        "p_des": p_des.copy(),
        "v_cmd": v_cmd.copy(),
        "v_des": v_des.copy(),
        "tau": tau.copy(),
        "saturated": saturated,
        "table": int(_table_touch(sim)),
        "drop": bool(latched(sim)),
        "qpos": np.array(sim.data.qpos, float).copy(),
        "gap": float(geo["inner_gap_m"]),
    }


def _restore(sim, snap):
    restore_ballistic(sim, snap)
    restore_state(sim, None)


def _play(sim, snap, actions, phase):
    _restore(sim, snap)
    rows = [_obs(sim, "ic")]
    commands = []
    for action in actions:
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        cmd = map_action(action, rd, rh, AIRBORNE_V3)
        step_legal(sim, action)
        row = _obs(sim, phase)
        row["v_world_cmd"] = np.asarray(cmd["v_world"], float).copy()
        rows.append(row)
        commands.append(np.asarray(cmd["v_world"], float))
        if latched(sim) or _bottom(sim) < TABLE_TOP:
            break
    return rows, commands


def _nominal(sim, ref, rows):
    before = {
        "p_des": np.asarray(sim.fsm.p_des, float).reshape(-1)[:3].copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "v_des": np.asarray(sim.fsm.v_des, float).copy(),
        "phase": str(sim.fsm.phase),
        "grip": None if sim.data.ctrl.size <= 7 else float(sim.data.ctrl[7]),
        "r_to_hand": float(rot_angle(np.asarray(sim.fsm.r_des, float), hand_R(sim))),
    }
    qpos = np.array(sim.data.qpos, float).copy()
    record = apply_handoff(sim, ref)
    after = {
        "p_des": np.asarray(sim.fsm.p_des, float).reshape(-1)[:3].copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "v_des": np.asarray(sim.fsm.v_des, float).copy(),
        "phase": str(sim.fsm.phase),
        "grip": None if sim.data.ctrl.size <= 7 else float(sim.data.ctrl[7]),
        "qpos_delta": float(np.max(np.abs(sim.data.qpos - qpos))),
        "r_to_saved": float(record["r_des_to_hand_rad"]),
    }
    t0 = float(sim.data.time)
    lost = 0
    lift_done = False
    reason = "TABLE_DROP" if latched(sim) else None
    end = None
    count = 0
    while reason is None and float(sim.data.time) - t0 < NOMINAL_S - 1e-9:
        nominal_step(sim)
        count += 1
        if count % 10 == 0:
            rows.append(_obs(sim, "nominal"))
        both = rows[-1]["nL"] > 0 and rows[-1]["nR"] > 0 if rows else False
        # The official check uses its own contact count. Recompute here.
        from training.horizon_outcome import _dropped, _monitor

        monitor = _monitor(sim)
        both = int(monitor["nL"]) > 0 and int(monitor["nR"]) > 0
        if both and float(sim.fsm.p_des[2]) >= float(sim.fsm.grasp_z + 0.0):
            from training.demo_ballistic_impact import z_tgt_of

            if float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-9:
                lift_done = True
        if latched(sim):
            reason = "TABLE_DROP"
            break
        if _dropped(monitor) or _escaped(sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = 0 if both else lost + 1
        if lost >= 75:
            reason = "LOST_GRASP"
            break
        end = monitor
    elapsed = float(sim.data.time) - t0
    if reason is None:
        if elapsed < NOMINAL_S - 1e-6:
            reason = "PHYSICAL_VERIFICATION_FAILED"
        elif end is None or int(end["nL"]) == 0 or int(end["nR"]) == 0:
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    return {
        "trace_reason": reason,
        "elapsed": round(elapsed, 3),
        "lift_done": bool(lift_done),
        "before": _plain(before),
        "after": _plain(after),
        "handoff": {key: record[key] for key in ("p_des", "v_cmd_norm", "grip_ctrl", "physics_edited", "qpos_max_abs", "r_des_to_hand_rad")},
    }


def _escaped(sim):
    from training.recovery_runtime import _escaped as escaped

    return escaped(sim)


def _plain(value):
    if isinstance(value, np.ndarray):
        return [round(float(x), 5) for x in value.reshape(-1)]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, float):
        return round(value, 5)
    return value


def _official(sim, snap, ref):
    _restore(sim, snap)
    class Env:
        pass
    env = Env()
    env.sim = sim
    out = nominal_verify(env, ref, seconds=NOMINAL_S)
    return {"reason": out.get("reason"), "elapsed": round(float(out.get("nominal_elapsed") or 0.0), 3), "end": out.get("end")}


def _metrics(rows, commands):
    ic = rows[0]
    last = rows[-1]
    nominal_rows = [row for row in rows if row["phase"] == "nominal"] or [last]
    recovery_rows = [row for row in rows if row["phase"] != "nominal"]
    end_recovery = recovery_rows[-1]

    def xy(row, key):
        return np.asarray(row[key][:2], float)

    def rel_xy(row):
        return float(np.linalg.norm(row["p_rel"][:2]))

    def vrel_xy(row):
        return float(np.linalg.norm(row["v_rel"][:2]))

    both = [int(row["nL"] > 0 and row["nR"] > 0) for row in recovery_rows]
    first_both = next((index for index, flag in enumerate(both) if flag), None)
    vcmd = np.array([row["v_cmd"] for row in nominal_rows], float) if nominal_rows else np.zeros((1, 3))
    pdes = np.array([row["p_des"] for row in nominal_rows], float) if nominal_rows else np.zeros((1, 3))
    hand = np.array([row["p_hand"] for row in nominal_rows], float)
    obj = np.array([row["p_obj"] for row in nominal_rows], float)
    return {
        "recovery_s": round(float(end_recovery["t"] - ic["t"]), 3),
        "d_hand_xy_recovery": [round(float(x), 4) for x in (xy(end_recovery, "p_hand") - xy(ic, "p_hand"))],
        "d_obj_xy_recovery": [round(float(x), 4) for x in (xy(end_recovery, "p_obj") - xy(ic, "p_obj"))],
        "rel_xy_ic": round(rel_xy(ic), 4),
        "rel_xy_handoff": round(rel_xy(end_recovery), 4),
        "vrel_xy_ic": round(vrel_xy(ic), 4),
        "vrel_xy_handoff": round(vrel_xy(end_recovery), 4),
        "vrel_z_ic": round(float(ic["v_rel"][2]), 4),
        "vrel_z_handoff": round(float(end_recovery["v_rel"][2]), 4),
        "n_ic": [ic["nL"], ic["nR"]],
        "n_handoff": [end_recovery["nL"], end_recovery["nR"]],
        "first_bilateral_s": None if first_both is None else round(float(recovery_rows[first_both]["t"] - ic["t"]), 3),
        "bottom_min": round(float(min(row["bottom"] for row in rows)), 4),
        "table_contacts": int(sum(row["table"] for row in rows)),
        "drop": any(row["drop"] for row in rows),
        "nominal_vcmd_max": round(float(np.max(np.abs(vcmd))), 5),
        "nominal_pdes_xy_span": round(float(np.max(np.linalg.norm(pdes[:, :2] - pdes[0, :2], axis=1))), 5) if len(pdes) else 0.0,
        "nominal_hand_xy_span": round(float(np.max(np.linalg.norm(hand[:, :2] - hand[0, :2], axis=1))), 5),
        "nominal_obj_xy_span": round(float(np.max(np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1))), 5),
        "nominal_perr_max": round(float(max(np.linalg.norm(row["p_err"]) for row in nominal_rows)), 5),
        "cmd_xy_max": 0.0 if not commands else round(float(np.max(np.linalg.norm(np.array(commands)[:, :2], axis=1))), 4),
        "cmd_z_max": 0.0 if not commands else round(float(np.max(np.abs(np.array(commands)[:, 2]))), 4),
        "saturated_steps": int(sum(row["saturated"] for row in rows)),
    }


def _save_series(path, rows):
    fields = [
        "t", "phase", "fsm", "nL", "nR", "bottom", "fn", "ft", "table", "drop", "saturated", "gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + [
            "phx", "phy", "phz", "pox", "poy", "poz", "vhx", "vhy", "vhz", "vox", "voy", "voz",
            "prel_x", "prel_y", "prel_z", "vrel_x", "vrel_y", "vrel_z",
            "pdes_x", "pdes_y", "pdes_z", "perr_x", "perr_y", "perr_z",
            "vcmd_x", "vcmd_y", "vcmd_z", "vdes_x", "vdes_y", "vdes_z", "wx", "wy", "wz",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "t": row["t"], "phase": row["phase"], "fsm": row["fsm"],
                "nL": row["nL"], "nR": row["nR"], "bottom": row["bottom"],
                "fn": row["fn"], "ft": row["ft"], "table": row["table"],
                "drop": int(row["drop"]), "saturated": row["saturated"], "gap": row["gap"],
                "phx": row["p_hand"][0], "phy": row["p_hand"][1], "phz": row["p_hand"][2],
                "pox": row["p_obj"][0], "poy": row["p_obj"][1], "poz": row["p_obj"][2],
                "vhx": row["v_hand"][0], "vhy": row["v_hand"][1], "vhz": row["v_hand"][2],
                "vox": row["v_obj"][0], "voy": row["v_obj"][1], "voz": row["v_obj"][2],
                "prel_x": row["p_rel"][0], "prel_y": row["p_rel"][1], "prel_z": row["p_rel"][2],
                "vrel_x": row["v_rel"][0], "vrel_y": row["v_rel"][1], "vrel_z": row["v_rel"][2],
                "pdes_x": row["p_des"][0], "pdes_y": row["p_des"][1], "pdes_z": row["p_des"][2],
                "perr_x": row["p_err"][0], "perr_y": row["p_err"][1], "perr_z": row["p_err"][2],
                "vcmd_x": row["v_cmd"][0], "vcmd_y": row["v_cmd"][1], "vcmd_z": row["v_cmd"][2],
                "vdes_x": row["v_des"][0], "vdes_y": row["v_des"][1], "vdes_z": row["v_des"][2],
                "wx": row["w_obj"][0], "wy": row["w_obj"][1], "wz": row["w_obj"][2],
            })


def _video(sim, rows, dest, title):
    from training.recatch_interface_comparison import _render

    _render(sim, {"rows": rows}, dest, title)


def _case(sim, snap, ref, actions, name, phase):
    rows, commands = _play(sim, snap, actions, phase)
    pre = make_snap(sim)
    traced = _nominal(sim, ref, rows)
    official = _official(sim, pre, ref)
    summary = _metrics(rows, commands)
    summary["name"] = name
    summary["trace_reason"] = traced["trace_reason"]
    summary["official_reason"] = official["reason"]
    summary["official_elapsed"] = official["elapsed"]
    summary["official_end"] = official["end"]
    summary["handoff_before"] = traced["before"]
    summary["handoff_after"] = traced["after"]
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    _save_series(OUT / "traces" / f"{name}.csv", rows)
    np.savez_compressed(
        OUT / "traces" / f"{name}.npz",
        t=np.array([row["t"] for row in rows]),
        p_hand=np.array([row["p_hand"] for row in rows]),
        p_obj=np.array([row["p_obj"] for row in rows]),
        v_hand=np.array([row["v_hand"] for row in rows]),
        v_obj=np.array([row["v_obj"] for row in rows]),
        p_rel=np.array([row["p_rel"] for row in rows]),
        v_rel=np.array([row["v_rel"] for row in rows]),
        nL=np.array([row["nL"] for row in rows]),
        nR=np.array([row["nR"] for row in rows]),
        bottom=np.array([row["bottom"] for row in rows]),
        p_des=np.array([row["p_des"] for row in rows]),
        v_cmd=np.array([row["v_cmd"] for row in rows]),
        phase=np.array([row["phase"] for row in rows]),
    )
    print(name, official["reason"], "rel", summary["rel_xy_ic"], "->", summary["rel_xy_handoff"], "vrel", summary["vrel_xy_ic"], "->", summary["vrel_xy_handoff"], flush=True)
    return summary, rows


def _scan(sim, snap, ref, actions, label):
    rows = []
    _restore(sim, snap)
    samples = [_obs(sim, "ic")]
    both_since = None
    for index, action in enumerate(actions):
        step_legal(sim, action)
        samples.append(_obs(sim, "recovery"))
        both = samples[-1]["nL"] > 0 and samples[-1]["nR"] > 0
        if both and both_since is None:
            both_since = index
    speeds = [float(np.linalg.norm(row["v_rel"][:2])) for row in samples]
    peak = int(np.argmax(speeds))
    falling = next((index for index in range(peak, len(speeds)) if speeds[index] < 0.7 * max(speeds[peak], 1e-6)), None)
    marks = {0, len(actions)}
    if both_since is not None:
        marks.add(max(0, both_since))
        marks.add(min(len(actions), both_since + 1))
    if falling is not None:
        marks.add(int(falling))
    for fraction in (0.25, 0.5, 0.75, 1.0):
        marks.add(int(round(fraction * len(actions))))
    # Later than the recorded tape: legal zero-velocity hold, still step_legal.
    extra = [10, 25, 50]
    out = []
    for mark in sorted(marks):
        if mark == 0:
            chosen = np.zeros((0, 7), np.float32)
        else:
            chosen = np.asarray(actions[:mark], np.float32)
        result = _handoff_at(sim, snap, ref, chosen, f"{label}_t{mark}")
        result["mark"] = mark
        result["extra_hold_steps"] = 0
        out.append(result)
        print("SCAN", label, mark, result["official_reason"], "vrel", result["vrel_xy"], "n", result["nL"], result["nR"], flush=True)
    for hold in extra:
        pad = np.zeros((hold, 7), np.float32)
        pad[:, 6] = 1.0
        chosen = np.concatenate([np.asarray(actions, np.float32), pad], axis=0)
        result = _handoff_at(sim, snap, ref, chosen, f"{label}_extra{hold}")
        result["mark"] = len(actions)
        result["extra_hold_steps"] = hold
        out.append(result)
        print("SCAN", label, "extra", hold, result["official_reason"], flush=True)
    return out, samples


def _handoff_at(sim, snap, ref, actions, name):
    _restore(sim, snap)
    for action in actions:
        step_legal(sim, action)
        if latched(sim) or _bottom(sim) < TABLE_TOP:
            return {"name": name, "official_reason": "TABLE_DURING_RECOVERY", "vrel_xy": None, "nL": 0, "nR": 0}
    state = _obs(sim, "handoff")
    pre = make_snap(sim)
    official = _official(sim, pre, ref)
    return {
        "name": name,
        "official_reason": official["reason"],
        "elapsed": official["elapsed"],
        "end_z": None if official["end"] is None else official["end"].get("z"),
        "vrel_xy": round(float(np.linalg.norm(state["v_rel"][:2])), 4),
        "vrel_z": round(float(state["v_rel"][2]), 4),
        "prel_xy": round(float(np.linalg.norm(state["p_rel"][:2])), 4),
        "prel_x": round(float(state["p_rel"][0]), 4),
        "nL": state["nL"],
        "nR": state["nR"],
        "bottom": round(state["bottom"], 4),
        "p_hand_y": round(float(state["p_hand"][1]), 4),
        "p_obj_y": round(float(state["p_obj"][1]), 4),
        "p_err": round(float(np.linalg.norm(state["p_err"])), 5),
        "v_obj": round(float(np.linalg.norm(state["v_obj"])), 4),
        "spin": round(float(np.linalg.norm(state["w_obj"])), 3),
    }


def _plot(series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for name, rows in series:
        t0 = rows[0]["t"]
        t = np.array([row["t"] - t0 for row in rows])
        axes[0].plot(t, [np.linalg.norm(row["p_rel"][:2]) for row in rows], label=name)
        axes[1].plot(t, [np.linalg.norm(row["v_rel"][:2]) for row in rows], label=name)
        axes[2].plot(t, [row["p_obj"][1] - rows[0]["p_obj"][1] for row in rows], label=name + " obj")
        axes[2].plot(t, [row["p_hand"][1] - rows[0]["p_hand"][1] for row in rows], linestyle="--", label=name + " hand")
    axes[0].set_ylabel("hand-frame lateral offset m")
    axes[1].set_ylabel("hand-frame lateral speed m/s")
    axes[2].set_ylabel("world Y displacement m")
    axes[2].set_xlabel("time from IC s")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "relative_state.png", dpi=120)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    sim, _base, ref, _target, _book = _setup()
    with (SRC / "snapshot.pkl").open("rb") as handle:
        snap = pickle.load(handle)
    decoded = np.load(SRC / "actions.npz")["decoded"].astype(np.float32)
    rh = None
    _restore(sim, snap)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    print("R_hand columns world", np.round(rh, 3), flush=True)
    zero = np.zeros_like(decoded)
    zero[:, 6] = 1.0
    lateral = decoded.copy()
    lateral[:, 2:6] = 0.0
    lateral[:, 6] = 1.0
    summaries = []
    stored = {}
    for name, actions in (
        ("A_immediate", np.zeros((0, 7), np.float32)),
        ("B_recovery", decoded),
        ("C_zero", zero),
        ("D_lateral", lateral),
    ):
        summary, rows = _case(sim, snap, ref, actions, name, "recovery" if name != "A_immediate" else "wait")
        summaries.append(summary)
        stored[name] = rows
        if name in ("A_immediate", "B_recovery", "C_zero", "D_lateral"):
            _video(sim, rows, OUT / "videos" / f"{name}.mp4", name)
    scan, _samples = _scan(sim, snap, ref, decoded, "az40")
    cell = _cell(35.0, 8.0, 0.0)
    psnap, info = _fly(sim, _base, cell)
    _restore(sim, psnap)
    rolled = _oracle(sim, _target, ref["r_des"], WIN)
    az35_actions = np.stack(rolled["actions"]).astype(np.float32) if rolled["actions"] else np.zeros((0, 7), np.float32)
    summary35, rows35 = _case(sim, psnap, ref, az35_actions, "E35_same_law", "recovery")
    summaries.append(summary35)
    scan35, _s35 = _scan(sim, psnap, ref, az35_actions, "az35")
    _plot([
        ("A immediate", stored["A_immediate"]),
        ("B recovery", stored["B_recovery"]),
        ("C zero", stored["C_zero"]),
        ("D lateral", stored["D_lateral"]),
        ("35 same law", rows35),
    ])
    fields = ["name", "official_reason", "official_elapsed", "recovery_s", "rel_xy_ic", "rel_xy_handoff", "vrel_xy_ic", "vrel_xy_handoff", "n_ic", "n_handoff", "first_bilateral_s", "d_hand_xy_recovery", "d_obj_xy_recovery", "nominal_vcmd_max", "nominal_pdes_xy_span", "nominal_hand_xy_span", "nominal_obj_xy_span", "cmd_xy_max", "cmd_z_max", "table_contacts", "bottom_min"]
    with (OUT / "MATCHED_CONTROL_COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    scan_fields = ["name", "mark", "extra_hold_steps", "official_reason", "elapsed", "vrel_xy", "vrel_z", "prel_xy", "prel_x", "nL", "nR", "bottom", "p_hand_y", "p_obj_y", "p_err", "v_obj", "spin", "end_z"]
    with (OUT / "HANDOFF_TIME_SCAN.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scan_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scan + scan35)
    (OUT / "facts.json").write_text(json.dumps({
        "R_hand": np.round(rh, 4).tolist(),
        "cases": summaries,
        "scan40": scan,
        "scan35": scan35,
        "az35_peak_n": info["peak_n"],
        "az35_oracle_phase": rolled["phase"],
    }, indent=2, default=str), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
