"""Move the braked airborne recatch onto the official step_legal interface.

Does not change step_legal, TABLE_DROP, the nominal controller, or K48.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery_actions import AIRBORNE_V3, normalize_action
from envs.airborne_obs import relative_state
from envs.deterioration import body_twist
from training.capture_geometry import capture_geometry
from training.demo_ballistic_impact import assert_noslip, make_sim
from training.full_3d_airborne_recatch_dataset import capture_nominal, capture_pads
from training.horizontal_grasp_offset_freefall_recatch import DT, grasp_settled
from training.privileged_ball_impact_search import step_legal
from training.recatch_braking_optimization import (
    BASE_SPEC,
    _advance,
    _bottom,
    _grip,
    _law,
    _pad_force,
    _pinch_target,
    _prepare,
    _trigger,
    run,
)
from training.recovery_runtime import gains_from_cfg, stamp
from training.replay_core import TABLE_TOP
from training.table_drop import latched, restore_state

OUT = ROOT / "results" / "diagnostics" / "legal_recatch_transfer"
WIN = {"trigger": "both", "accel": 6.0, "grip": -18.0, "lift": 0.20}


def _enable_supports(sim):
    """_prepare disables floor and table. This audit needs the original collision flags."""
    fresh, _cfg = make_sim()
    for name in ("floor", "table"):
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        src = mujoco.mj_name2id(fresh.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        sim.model.geom_contype[gid] = int(fresh.model.geom_contype[src])
        sim.model.geom_conaffinity[gid] = int(fresh.model.geom_conaffinity[src])
    return {
        name: {
            "contype": int(sim.model.geom_contype[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)]),
            "conaffinity": int(sim.model.geom_conaffinity[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)]),
        }
        for name in ("floor", "table")
    }


def _table_box(sim):
    gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    center = np.array(sim.model.geom_pos[gid], float)
    half = np.array(sim.model.geom_size[gid], float)
    return {
        "top": float(center[2] + half[2]),
        "x": [float(center[0] - half[0]), float(center[0] + half[0])],
        "y": [float(center[1] - half[1]), float(center[1] + half[1])],
    }


def _over_table(sim, box):
    pos = np.array(sim.data.xpos[sim.ids.object_body], float)
    return bool(box["x"][0] <= pos[0] <= box["x"][1] and box["y"][0] <= pos[1] <= box["y"][1])


def _ball_touch(sim):
    ball = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    if ball < 0:
        return 0
    obj = int(sim.ids.object_geom)
    n = 0
    for i in range(int(sim.data.ncon)):
        g = {int(sim.data.contact[i].geom1), int(sim.data.contact[i].geom2)}
        if ball in g and obj in g:
            n += 1
    return n


def _snap(sim):
    return {
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "time": float(sim.data.time),
        "p_des": np.array(sim.fsm.p_des, float).copy(),
        "r_des": np.array(sim.fsm.r_des, float).copy(),
        "v_des": np.array(sim.fsm.v_des, float).copy(),
        "v_cmd": np.array(sim.fsm.v_cmd, float).copy(),
        "w_des": np.array(sim.fsm.w_des, float).copy(),
        "w_cmd": np.array(sim.fsm.w_cmd, float).copy(),
    }


def _restore_live(sim, state):
    sim.data.qpos[:] = state["qpos"]
    sim.data.qvel[:] = state["qvel"]
    sim.data.ctrl[:] = state["ctrl"]
    sim.data.time = state["time"]
    sim.fsm.p_des = state["p_des"].copy()
    sim.fsm.r_des = state["r_des"].copy()
    sim.fsm.v_des = state["v_des"].copy()
    sim.fsm.v_cmd = state["v_cmd"].copy()
    sim.fsm.w_des = state["w_des"].copy()
    sim.fsm.w_cmd = state["w_cmd"].copy()
    mujoco.mj_forward(sim.model, sim.data)


def _sample(sim, t0, phase, v_des, tau):
    geo = capture_geometry(sim)
    rel = relative_state(sim.model, sim.data, sim.ids)
    fl, fr = _pad_force(sim)
    ph = np.array(rel["p_hand_w"], float)
    p_des = np.array(sim.fsm.p_des, float).reshape(-1)[:3]
    return {
        "t": float(sim.data.time) - t0,
        "phase": phase,
        "bottom": _bottom(sim),
        "zo": float(rel["p_obj_w"][2]),
        "xy": [float(rel["p_obj_w"][0]), float(rel["p_obj_w"][1])],
        "vh": np.array(rel["v_hand_w"], float),
        "vo": np.array(rel["v_obj_w"], float),
        "vrel_h": np.array(rel["v_rel_h"], float),
        "nL": int(geo["nL"]),
        "nR": int(geo["nR"]),
        "gap": float(geo["inner_gap_m"]),
        "between": int(bool(geo["between"])),
        "fn": [fl, fr],
        "p_err": float(np.linalg.norm(p_des - ph)),
        "p_des_z": float(p_des[2]),
        "p_hand_z": float(ph[2]),
        "v_des_z": float(np.asarray(v_des, float).reshape(-1)[2]),
        "tau": float(tau),
        "tau_j": np.array(sim.data.ctrl[:7], float),
        "qpos": np.array(sim.data.qpos, float).copy(),
    }


def _audit_existing(sim, snap, nominal, pads, pad_mu, box):
    rows = []
    for interface in ("C", "B", "A"):
        result = run(
            sim, snap, nominal, pads, pad_mu, WIN, interface=interface, drop_table=False, horizon=2.4,
        )
        bottom = float(np.min([r["bottom"] for r in result["rows"]]))
        over = []
        # The diagnostic stepper still parks the ball every physics step.
        rec = {
            "interface": interface,
            "labels": result["labels"],
            "stable": result["stable"],
            "min_bottom_m": result["min_bottom_m"],
            "end_zo_m": result["end_zo_m"],
            "end_voz": result["end_voz"],
            "end_n": result["end_n"],
            "table_drop": bool(latched(sim)),
            "table_top_m": box["top"],
            "bottom_clears_table": bool(result["min_bottom_m"] >= box["top"]),
            "ball_parked_every_step": True,
            "qpos_object_written": False,
            "gripper_qpos_written": False,
        }
        rows.append(rec)
        print("AUDIT", interface, rec["labels"], "bottom", rec["min_bottom_m"], "drop", rec["table_drop"], flush=True)
    return rows


def _chase_until_contact(sim, snap, nominal, pads, pad_mu):
    from training.multi_disturbance_airborne_feasibility import restore

    restore(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = DT
    sim._watch_drop = True
    restore_state(sim, None)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
        sim.set_guide_weld(True)
    t0 = float(sim.data.time)
    rel0 = relative_state(sim.model, sim.data, sim.ids)
    rh = np.array(rel0["p_rel_h"], float)
    phase = "chase"
    free_s = both_s = 0.0
    armed = False
    for _ in range(int(1.2 / DT)):
        geo = capture_geometry(sim)
        phase, free_s, both_s, armed = _advance(phase, geo, free_s, both_s, armed, DT)
        if WIN and _trigger("both", geo, both_s, free_s, armed):
            return _snap(sim), rh, float(sim.data.time) - t0
        hold = phase == "hold"
        p_star, v_star, _err = _law(
            sim, rh, BASE_SPEC["aim_down"], BASE_SPEC["k"], BASE_SPEC["v_clip"], hold, BASE_SPEC["corr_clip"],
        )
        tau, phase = _grip(phase, geo, BASE_SPEC["open_tau"])
        from training.recatch_braking_optimization import _step

        _step(sim, p_star, v_star, tau)
    raise RuntimeError("bilateral recontact was not reached with the table enabled")


def _brake_c(sim, state, rh, horizon=0.15):
    _restore_live(sim, state)
    t0 = float(sim.data.time)
    v_cmd_z = float(min(body_twist(sim.model, sim.data, sim.ids.hand_body)[0][2], 0.0))
    p_z = float(sim.data.xpos[sim.ids.hand_body][2])
    rows = []
    tape = []
    from training.recatch_braking_optimization import _step

    for _ in range(int(horizon / DT)):
        vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
        v_cmd_z = min(0.0, v_cmd_z + 6.0 * DT)
        p_z = p_z + v_cmd_z * DT
        p_star = _pinch_target(sim, rh, p_z)
        v_star = np.array([float(vo[0]), float(vo[1]), v_cmd_z], float)
        tau = -18.0
        _step(sim, p_star, v_star, tau)
        rows.append(_sample(sim, t0, "brake", v_star, tau))
        tape.append((v_star.copy(), tau))
    return rows, tape


def _brake_from_tape(sim, state, tape, interface):
    """Replay C's 2 ms command tape. A calls step_legal; B integrates p_des once per 20 ms."""
    from training.recatch_braking_optimization import _step
    from training.recatch_interface_comparison import _mapped_velocity

    _restore_live(sim, state)
    t0 = float(sim.data.time)
    rows = []
    p_state = np.array(state["p_des"], float).reshape(-1)[:3].copy()
    for i in range(0, len(tape), 10):
        v_star, tau = tape[i]
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        if interface == "A":
            action = normalize_action(rh.T @ v_star, np.zeros(3), tau, AIRBORNE_V3)
            step_legal(sim, action)
            rows.append(_sample(sim, t0, "A", v_star, tau))
            continue
        _action, v_apply, tau_apply = _mapped_velocity(sim, v_star, tau)
        p_state = p_state + v_apply * 0.02
        for _ in range(10):
            _step(sim, p_state, v_apply, tau_apply)
        rows.append(_sample(sim, t0, "B", v_apply, tau_apply))
    return rows


def _summarize_window(name, rows):
    if not rows:
        return {"name": name}
    vh = np.stack([r["vh"] for r in rows])
    vo = np.stack([r["vo"] for r in rows])
    t = np.array([r["t"] for r in rows], float)
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.02
    dt = np.maximum(dt, 1e-6)
    acc_h = np.diff(vh[:, 2], prepend=vh[0, 2]) / dt
    acc_o = np.diff(vo[:, 2], prepend=vo[0, 2]) / dt
    both = np.array([(r["nL"] > 0 and r["nR"] > 0) for r in rows])
    return {
        "name": name,
        "n": len(rows),
        "t1": round(float(rows[-1]["t"]), 4),
        "min_bottom_m": round(float(min(r["bottom"] for r in rows)), 4),
        "end_voz": round(float(rows[-1]["vo"][2]), 3),
        "end_vhz": round(float(rows[-1]["vh"][2]), 3),
        "mean_hand_az": round(float(np.mean(acc_h)), 2),
        "mean_obj_az": round(float(np.mean(acc_o)), 2),
        "max_p_err_m": round(float(max(r["p_err"] for r in rows)), 4),
        "mean_fn_n": round(float(np.mean([sum(r["fn"]) for r in rows])), 2),
        "both_fraction": round(float(np.mean(both)), 3),
        "end_n": [int(rows[-1]["nL"]), int(rows[-1]["nR"])],
        "max_finger_slip": round(float(np.max(np.abs([r["vrel_h"][2] for r in rows]))), 3),
        "max_pad_slip": round(float(np.max(np.abs([r["vrel_h"][0] for r in rows]))), 3),
        "peak_tau": round(float(np.max(np.abs(np.stack([r["tau_j"] for r in rows])))), 2),
    }


def _save_rows(name, rows):
    path = OUT / "traces" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in rows], np.float32),
        bottom=np.array([r["bottom"] for r in rows], np.float32),
        zo=np.array([r["zo"] for r in rows], np.float32),
        xy=np.array([r["xy"] for r in rows], np.float32),
        vh=np.stack([r["vh"] for r in rows]).astype(np.float32),
        vo=np.stack([r["vo"] for r in rows]).astype(np.float32),
        vrel_h=np.stack([r["vrel_h"] for r in rows]).astype(np.float32),
        nL=np.array([r["nL"] for r in rows], np.int16),
        nR=np.array([r["nR"] for r in rows], np.int16),
        gap=np.array([r["gap"] for r in rows], np.float32),
        fn=np.array([r["fn"] for r in rows], np.float32),
        p_err=np.array([r["p_err"] for r in rows], np.float32),
        v_des_z=np.array([r["v_des_z"] for r in rows], np.float32),
        tau=np.array([r["tau"] for r in rows], np.float32),
        tau_j=np.stack([r["tau_j"] for r in rows]).astype(np.float32),
        qpos=np.stack([r["qpos"] for r in rows]).astype(np.float32),
    )
    return str(path.relative_to(ROOT)).replace("\\", "/")


def run_legal(sim, snap, nominal, pads, pad_mu, spec, horizon=2.6):
    """Full episode through the official step_legal. Table stays on."""
    from training.multi_disturbance_airborne_feasibility import restore

    restore(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = DT
    restore_state(sim, None)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
        sim.set_guide_weld(True)
    t0 = float(sim.data.time)
    rel0 = relative_state(sim.model, sim.data, sim.ids)
    rh = np.array(rel0["p_rel_h"], float)
    phase = "chase"
    free_s = both_s = 0.0
    armed = False
    v_cmd_z = None
    p_z = None
    brake_t = None
    match_until = None
    zo_brake = None
    rows = []
    actions = []
    ball_hits = 0
    steps = int(horizon / 0.02)
    for _ in range(steps):
        geo = capture_geometry(sim)
        phase, free_s, both_s, armed = _advance(phase, geo, free_s, both_s, armed, 0.02)
        if spec is not None and phase in ("chase", "close") and _trigger(spec["trigger"], geo, both_s, free_s, armed):
            if v_cmd_z is None:
                vh = np.array(body_twist(sim.model, sim.data, sim.ids.hand_body)[0], float)
                vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
                v_cmd_z = float(min(vh[2], vo[2], 0.0))
                p_z = float(sim.data.xpos[sim.ids.hand_body][2])
                brake_t = float(sim.data.time) - t0
                zo_brake = float(sim.data.xpos[sim.ids.object_body][2])
                match_until = brake_t + float(spec["match_s"])
                phase = "brake"
        if phase in ("brake", "lift", "stable") and v_cmd_z is not None:
            vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
            both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
            now = float(sim.data.time) - t0
            if phase == "brake" and now < match_until:
                v_cmd_z = float(min(vo[2], 0.0))
            elif phase == "brake":
                v_cmd_z = min(0.0, float(v_cmd_z) + float(spec["accel"]) * 0.02)
                if both and v_cmd_z >= -0.02 and float(vo[2]) > -0.15:
                    phase = "lift"
            elif phase == "lift":
                v_cmd_z = min(float(spec["lift"]), float(v_cmd_z) + float(spec["accel"]) * 0.02)
                if both and float(vo[2]) > -0.02 and float(sim.data.xpos[sim.ids.object_body][2]) >= zo_brake + 0.03:
                    phase = "stable"
            if phase == "stable":
                v_cmd_z = 0.0
            p_z = float(p_z) + float(v_cmd_z) * 0.02
            v_star = np.array([float(vo[0]), float(vo[1]), float(v_cmd_z)], float)
            tau = float(spec["grip"])
        else:
            hold = phase == "hold"
            _p, v_star, _e = _law(
                sim, rh, BASE_SPEC["aim_down"], BASE_SPEC["k"], BASE_SPEC["v_clip"], hold, BASE_SPEC["corr_clip"],
            )
            if spec.get("pre_accel", 0.0) > 0.0 and _trigger("enter", geo, both_s, free_s, armed) and phase != "brake":
                v_star = np.array(v_star, float).copy()
                v_star[2] = min(0.0, float(v_star[2]) + float(spec["pre_accel"]) * 0.02)
            tau, phase = _grip(phase, geo, 2.0)
        hand = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        action = normalize_action(hand.T @ np.asarray(v_star, float), np.zeros(3), float(tau), AIRBORNE_V3)
        step_legal(sim, action)
        actions.append(np.array(action, np.float32))
        ball_hits += _ball_touch(sim)
        rows.append(_sample(sim, t0, phase, v_star, tau))
        if latched(sim):
            break
        if rows[-1]["bottom"] < TABLE_TOP and phase in ("brake", "lift", "stable"):
            break
        if phase == "stable" and sum(r["phase"] == "stable" for r in rows) >= int(1.0 / 0.02):
            break
    return _label_legal(rows, actions, ball_hits, brake_t, bool(latched(sim)))


def _label_legal(rows, actions, ball_hits, brake_t, dropped):
    off = False
    re = None
    for i, r in enumerate(rows):
        both = r["nL"] > 0 and r["nR"] > 0
        none = r["nL"] == 0 and r["nR"] == 0
        if none:
            off = True
        elif off and both and re is None:
            re = i
    bottom = float(min(r["bottom"] for r in rows))
    voz = np.array([r["vo"][2] for r in rows])
    vhz = np.array([r["vh"][2] for r in rows])
    labels = []
    if re is None:
        labels.append("NO_RECONTACT" if off else "NO_RELEASE")
    else:
        labels.append("RECONTACT")
        if np.any(voz[re:] > -0.05) and bottom >= TABLE_TOP:
            labels.append("BRAKED")
        zmin = re + int(np.argmin([r["zo"] for r in rows[re:]]))
        if max(r["zo"] for r in rows[zmin:]) >= rows[zmin]["zo"] + 0.03 and bottom >= TABLE_TOP:
            labels.append("LIFTED")
    stable_rows = [r for r in rows if r["phase"] == "stable"]
    stable = False
    if len(stable_rows) >= int(1.0 / 0.02) - 1 and "BRAKED" in labels and "LIFTED" in labels:
        window = stable_rows[: int(1.0 / 0.02)]
        frac = float(np.mean([r["nL"] > 0 and r["nR"] > 0 for r in window]))
        end = window[-1]
        stable = bool(frac >= 0.95 and end["bottom"] >= TABLE_TOP + 0.01 and end["vo"][2] > -0.05 and not dropped)
        if stable:
            labels.append("STABLE_AIRBORNE_HOLD")
    if bottom < TABLE_TOP:
        labels.append("CROSSED_TABLE")
    if dropped:
        labels.append("TABLE_DROP")
    stage = _failure_stage(rows, re, labels, bottom)
    return {
        "labels": labels,
        "stage": stage,
        "stable": stable,
        "recontact_t": None if re is None else round(float(rows[re]["t"]), 4),
        "brake_t": None if brake_t is None else round(float(brake_t), 4),
        "min_bottom_m": round(bottom, 4),
        "end_zo_m": round(float(rows[-1]["zo"]), 4),
        "end_voz": round(float(voz[-1]), 3),
        "end_vhz": round(float(vhz[-1]), 3),
        "end_n": [int(rows[-1]["nL"]), int(rows[-1]["nR"])],
        "ball_hits": int(ball_hits),
        "actions": actions,
        "rows": rows,
    }


def _failure_stage(rows, re, labels, bottom):
    if "STABLE_AIRBORNE_HOLD" in labels:
        return "STABLE_AIRBORNE_HOLD"
    if "TABLE_DROP" in labels and "BRAKED" not in labels:
        return "TABLE_DROP"
    if "NO_RELEASE" in labels:
        return "NO_RELEASE"
    if re is None:
        return "NO_RECONTACT"
    if "CROSSED_TABLE" in labels and "BRAKED" not in labels:
        end_n = rows[-1]["nL"] + rows[-1]["nR"]
        if end_n == 0 and rows[-1]["vh"][2] > rows[-1]["vo"][2] + 0.4:
            return "SLIP_DURING_BRAKE"
        return "CROSSED_DURING_BRAKE"
    if "LIFTED" in labels:
        return "LIFT_NOT_HELD"
    if "BRAKED" in labels:
        return "BRAKED_NO_LIFT"
    if rows[-1]["nL"] == 0 and rows[-1]["nR"] == 0:
        return "SLIP_DURING_BRAKE"
    return "BRAKE_INCOMPLETE"


def _grid():
    specs = []
    for trigger in ("both", "uni", "enter"):
        for accel in (1.5, 3.0, 6.0):
            for match_s in (0.0, 0.04, 0.08):
                for pre in (0.0, 2.0):
                    specs.append({
                        "trigger": trigger,
                        "accel": accel,
                        "match_s": match_s,
                        "pre_accel": pre,
                        "grip": -18.0,
                        "lift": 0.20,
                    })
    return specs


def _csv(path, rows):
    fields = ["name", "stage", "trigger", "accel", "match_s", "pre_accel", "recontact_t", "brake_t", "min_bottom_m", "end_zo_m", "end_voz", "end_vhz", "end_nL", "end_nR", "ball_hits"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _public(result, name, spec):
    return {
        "name": name,
        "stage": result["stage"],
        "labels": result["labels"],
        "trigger": spec["trigger"],
        "accel": spec["accel"],
        "match_s": spec["match_s"],
        "pre_accel": spec["pre_accel"],
        "recontact_t": result["recontact_t"],
        "brake_t": result["brake_t"],
        "min_bottom_m": result["min_bottom_m"],
        "end_zo_m": result["end_zo_m"],
        "end_voz": result["end_voz"],
        "end_vhz": result["end_vhz"],
        "end_nL": result["end_n"][0],
        "end_nR": result["end_n"][1],
        "ball_hits": result["ball_hits"],
        "stable": result["stable"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    sim, snap, nominal, pads, pad_mu = _prepare()
    supports = _enable_supports(sim)
    box = _table_box(sim)
    print("SUPPORTS", supports, "TABLE", box, flush=True)
    audit = _audit_existing(sim, snap, nominal, pads, pad_mu, box)
    state, rh, contact_t = _chase_until_contact(sim, snap, nominal, pads, pad_mu)
    print("CONTACT", round(contact_t, 4), "bottom", round(_bottom(sim), 4), flush=True)
    rows_c, tape = _brake_c(sim, state, rh, 0.15)
    rows_a = _brake_from_tape(sim, state, tape, "A")
    rows_b = _brake_from_tape(sim, state, tape, "B")
    matched = [
        _summarize_window("C", rows_c),
        _summarize_window("B", rows_b),
        _summarize_window("A", rows_a),
    ]
    _save_rows("matched_C", rows_c)
    _save_rows("matched_A", rows_a)
    _save_rows("matched_B", rows_b)
    print("MATCHED", json.dumps(matched), flush=True)
    searched = []
    best = None
    for spec in _grid():
        result = run_legal(sim, snap, nominal, pads, pad_mu, spec)
        name = f"A_{spec['trigger']}_a{spec['accel']}_m{spec['match_s']}_p{spec['pre_accel']}"
        pub = _public(result, name, spec)
        searched.append(pub)
        print("A", name, pub["stage"], "bottom", pub["min_bottom_m"], "re", pub["recontact_t"], flush=True)
        rank = (bool(result["stable"]), pub["min_bottom_m"], 0 if pub["stage"] == "SLIP_DURING_BRAKE" else 1)
        if best is None or rank > best[0]:
            best = (rank, name, result, spec)
        if result["stable"] and result["ball_hits"] == 0 and not latched(sim):
            break
    assert best is not None
    _csv(OUT / "SEARCH_RESULTS.csv", searched)
    best_name, best_result, best_spec = best[1], best[2], best[3]
    trace = _save_rows(best_name, best_result["rows"])
    summary = {
        "table": box,
        "audit_table_on": audit,
        "contact_t": contact_t,
        "matched_150ms": matched,
        "best_A": {**_public(best_result, best_name, best_spec), "trace": trace, "labels": best_result["labels"]},
        "k48": None,
        "success_12s": None,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BEST_A", best_name, best_result["stage"], best_result["labels"], flush=True)


if __name__ == "__main__":
    main()
