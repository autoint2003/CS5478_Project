"""No-table airborne recatch: historical chase versus the legal 20 ms interface.

Diagnostic only. Does not change the official controller, K48, TABLE_DROP, or
reward. Interface A copies the current step_legal setpoint rule locally.
Interface C may command the historical +18 N·m open torque. It does not write
finger qpos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery_actions import AIRBORNE_V3, map_action, normalize_action
from envs.airborne_obs import relative_state
from envs.deterioration import body_twist
from training.capture_geometry import capture_geometry
from training.demo_ballistic_impact import assert_noslip, make_sim
from training.full_3d_airborne_recatch_dataset import (
    TRACK,
    capture_nominal,
    capture_pads,
    drop_supports,
    grasp_settled,
)
from training.horizontal_grasp_offset_freefall_recatch import DT
from training.multi_disturbance_airborne_feasibility import restore
from training.recovery_runtime import gains_from_cfg, stamp

OUT = ROOT / "results" / "diagnostics" / "recatch_interface_comparison"
HOLD_S = 1.0
FREE_NEED_S = 0.020
GAP_TARGET = 0.049
HORIZON = 0.040
N_SUB = 10


def _limits(sim):
    return np.array(sim.model.actuator_ctrlrange[:7, 1], float)


def _vo(sim):
    return np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)


def _predict(sim, horizon):
    pos = np.array(sim.data.xpos[sim.ids.object_body], float)
    vel = _vo(sim)
    geo = capture_geometry(sim)
    touching = int(geo["nL"]) + int(geo["nR"]) > 0
    gravity = np.array(sim.model.opt.gravity, float)
    if touching:
        return pos + vel * horizon, vel.copy()
    return pos + vel * horizon + 0.5 * gravity * horizon * horizon, vel + gravity * horizon


def _sample(sim, t0, phase, tau, v_cmd, p_des, ctrl, lim):
    geo = capture_geometry(sim)
    rel = relative_state(sim.model, sim.data, sim.ids)
    vh = np.array(rel["v_hand_w"], float)
    vo = np.array(rel["v_obj_w"], float)
    sat = np.abs(np.asarray(ctrl[:7], float)) >= 0.95 * lim
    return {
        "t": float(sim.data.time) - t0,
        "phase": phase,
        "gap": float(geo["inner_gap_m"]),
        "finger_q": float(np.mean(geo["finger_qpos_m"])),
        "between": int(bool(geo["between"])),
        "nL": int(geo["nL"]),
        "nR": int(geo["nR"]),
        "cx": float(geo["center_hand_m"][0]),
        "cy": float(geo["center_hand_m"][1]),
        "cz": float(geo["center_hand_m"][2]),
        "axial": float(geo["axial_overlap_m"]),
        "vh": vh,
        "vo": vo,
        "tau": float(tau),
        "v_cmd": np.asarray(v_cmd, float).copy(),
        "p_err": float(np.linalg.norm(np.asarray(p_des, float) - rel["p_hand_w"])),
        "sat": int(np.sum(sat)),
        "tau_j": np.asarray(ctrl[:7], float).copy(),
        "qpos": np.array(sim.data.qpos, float).copy(),
        "zo": float(sim.data.xpos[sim.ids.object_body][2]),
    }


def _step(sim, p_des, v_des, tau):
    sim.fsm.p_des = np.asarray(p_des, float).copy()
    sim.fsm.v_des = np.asarray(v_des, float).copy()
    sim.fsm.v_cmd = np.asarray(v_des, float).copy()
    sim.fsm.w_des = np.zeros(3)
    sim.fsm.w_cmd = np.zeros(3)
    ctrl = apply_cartesian_ctrl(
        sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
        gripper_tau=float(tau), v_des=sim.fsm.v_des, w_des=np.zeros(3), **TRACK,
    )
    vz_before = None
    if getattr(sim, "_watch_drop", False):
        from training.table_drop import object_vz

        vz_before = object_vz(sim)
    mujoco.mj_step(sim.model, sim.data)
    if vz_before is not None:
        from training.table_drop import observe

        observe(sim, vz_before)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
    if getattr(sim, "_count_support", None) is not None:
        table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        floor = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        obj = int(sim.ids.object_geom)
        for index in range(int(sim.data.ncon)):
            geoms = {int(sim.data.contact[index].geom1), int(sim.data.contact[index].geom2)}
            if obj not in geoms:
                continue
            if table in geoms:
                sim._count_support["table"] += 1
            if floor in geoms:
                sim._count_support["floor"] += 1
    return ctrl


def _law(sim, rh, aim_down, k, v_clip, hold, spec_corr):
    """Privileged intercept. Feedforward uses the current cylinder speed."""
    rh_t = np.asarray(rh, float).copy()
    if hold:
        future, _vf = _predict(sim, 0.0)
        k_use, cap = 2.0, 1.5
    else:
        future, _vf = _predict(sim, HORIZON)
        k_use, cap = float(k), v_clip
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    p_des = future - rd @ rh_t
    p_des[2] -= float(aim_down)
    err = p_des - ph
    geo = capture_geometry(sim)
    one_side = (int(geo["nL"]) > 0) ^ (int(geo["nR"]) > 0)
    correction = float(k_use) * err
    # A one-sided hit means the dive is pushing the cylinder, not enclosing it.
    if one_side and not hold:
        correction = np.zeros(3)
    elif spec_corr is not None:
        correction = np.clip(correction, -float(spec_corr), float(spec_corr))
    v = _vo(sim) + correction
    if cap is not None:
        v = np.clip(v, -float(cap), float(cap))
    return p_des, v, err


def _capture_window(geo, free_s, armed):
    """Close only after a real opening and a real contact gap, once the cylinder is back in the pads."""
    return bool(
        armed
        and free_s >= 0.040
        and geo["between"]
        and 0.045 <= float(geo["inner_gap_m"]) <= 0.070
        and float(geo["axial_overlap_m"]) >= 0.004
        and abs(float(geo["center_hand_m"][0])) < 0.016
        and abs(float(geo["center_hand_m"][2]) - 0.098) < 0.020
    )


def _grip(phase, geo, open_tau):
    """Open to about 49 mm, then hold that gap. Full -18 N·m only after both pads are on."""
    qmin = min(float(v) for v in geo["finger_qpos_m"])
    both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
    gap = float(geo["inner_gap_m"])
    if qmin < 0.0:
        return float(open_tau), "chase"
    if phase == "hold":
        return -18.0, "hold"
    if both and phase == "close":
        return -18.0, "close"
    if phase == "close" and geo["between"] and qmin > 0.008:
        return -18.0, "close"
    if phase == "close":
        return 0.0, "chase"
    if gap < 0.040:
        return float(open_tau), phase
    if gap > 0.060:
        return -4.0, phase
    return 0.0, phase


def _advance(phase, geo, free_s, both_s, armed, dt):
    both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
    if not both:
        free_s += dt
    if float(geo["inner_gap_m"]) >= GAP_TARGET:
        armed = True
    if phase in ("open", "chase"):
        phase = "close" if _capture_window(geo, free_s, armed) else "chase"
    elif phase == "close":
        if both:
            both_s += dt
            if both_s >= 0.040:
                phase = "hold"
        else:
            both_s = 0.0
            if not geo["between"] or abs(float(geo["center_hand_m"][0])) > 0.020:
                phase = "chase"
    elif phase == "hold":
        if both or geo["between"]:
            both_s += dt
        else:
            both_s = 0.0
            phase = "chase"
    return phase, free_s, both_s, armed


def _restore(sim, snap, nominal, pads, pad_mu):
    restore(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = DT
    drop_supports(sim)
    if hasattr(sim, "park_ball"):
        sim.park_ball()


def _mapped_velocity(sim, v_world, tau_for_map):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    action = normalize_action(rh.T @ np.asarray(v_world, float), np.zeros(3), float(tau_for_map), AIRBORNE_V3)
    cmd = map_action(action, rd, rh, AIRBORNE_V3)
    return action, cmd["v_world"], float(cmd["tau"])


def rollout(sim, snap, nominal, pads, pad_mu, spec, already=False):
    """One closed-loop attempt. already=True keeps the simulator's current state."""
    if not already:
        _restore(sim, snap, nominal, pads, pad_mu)
    lim = _limits(sim)
    t0 = float(sim.data.time)
    rel0 = relative_state(sim.model, sim.data, sim.ids)
    rh = np.array(spec["rh"], float) if spec.get("rh") is not None else np.array(rel0["p_rel_h"], float)
    if spec["aim"] == "above" and spec.get("rh") is None:
        rh = rh + np.array([0.0, 0.0, 0.012])
    aim_down = float(spec.get("aim_down", 0.040 if spec["aim"] == "below" else 0.0))
    phase = "chase"
    free_s = 0.0
    both_s = 0.0
    armed = False
    hold_t = None
    re_t = None
    seen_free = False
    rows = []
    interface = spec["interface"]
    p_state = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    steps = int(spec.get("horizon_s", 2.2) / DT)
    i = 0
    while i < steps:
        geo = capture_geometry(sim)
        decision_dt = DT if interface in ("C", "C_v3speed") else N_SUB * DT
        phase, free_s, both_s, armed = _advance(phase, geo, free_s, both_s, armed, decision_dt)
        if free_s >= 0.040:
            seen_free = True
        if phase == "hold" and hold_t is None and seen_free and armed:
            hold_t = float(sim.data.time) - t0
        if re_t is None and int(geo["nL"]) > 0 and int(geo["nR"]) > 0 and seen_free and armed:
            re_t = float(sim.data.time) - t0
        hold = phase == "hold"
        p_star, v_star, _err = _law(
            sim, rh, aim_down, spec["k"], spec["v_clip"], hold, spec.get("corr_clip"),
        )
        tau, phase = _grip(phase, geo, spec["open_tau"])
        block = 1 if interface == "C" else N_SUB
        if interface == "A":
            action, v_apply, tau_apply = _mapped_velocity(sim, v_star, tau)
            p_anchor = np.array(sim.data.xpos[sim.ids.hand_body], float)
            for _ in range(block):
                sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, np.zeros(3), DT)
                ctrl = _step(sim, p_anchor, v_apply, tau_apply)
                rows.append(_sample(sim, t0, phase, tau_apply, v_apply, p_anchor, ctrl, lim))
                i += 1
            spec_action = action
        elif interface == "B":
            action, v_apply, tau_apply = _mapped_velocity(sim, v_star, tau)
            for _ in range(block):
                p_state = p_state + v_apply * DT
                sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, np.zeros(3), DT)
                ctrl = _step(sim, p_state, v_apply, tau_apply)
                rows.append(_sample(sim, t0, phase, tau_apply, v_apply, p_state, ctrl, lim))
                i += 1
            spec_action = action
        elif interface == "C_v3speed":
            _action, v_apply, _tau_mapped = _mapped_velocity(sim, v_star, tau)
            ctrl = _step(sim, p_star, v_apply, tau)
            rows.append(_sample(sim, t0, phase, tau, v_apply, p_star, ctrl, lim))
            i += 1
            spec_action = _action
        else:
            ctrl = _step(sim, p_star, v_star, tau)
            rows.append(_sample(sim, t0, phase, tau, v_star, p_star, ctrl, lim))
            i += 1
            spec_action = None
        if hold_t is not None and rows[-1]["t"] >= hold_t + HOLD_S:
            break
        if rows[-1]["t"] > 0.80 and phase == "chase" and float(rows[-1]["gap"]) < 0.030:
            break
        zo = float(sim.data.xpos[sim.ids.object_body][2])
        if zo < -0.05 and phase != "hold":
            break
    return _summarize(spec, rows, hold_t, re_t, seen_free, free_s, armed, spec_action)


def _summarize(spec, rows, hold_t, re_t, seen_free, free_s, armed, last_action):
    gap = np.array([r["gap"] for r in rows])
    between = np.array([r["between"] for r in rows])
    both = np.array([(r["nL"] > 0 and r["nR"] > 0) for r in rows])
    held = False
    hold_frac = 0.0
    re_row = None
    if re_t is not None:
        re_row = min(rows, key=lambda r: abs(r["t"] - re_t))
    vrel_re = None if re_row is None else [round(float(a - b), 3) for a, b in zip(re_row["vo"], re_row["vh"])]
    speed_ok = vrel_re is not None and abs(vrel_re[2]) <= 0.25
    if hold_t is not None:
        window = [r for r in rows if hold_t - 1e-9 <= r["t"] <= hold_t + HOLD_S + 1e-9]
        if window and window[-1]["t"] >= hold_t + HOLD_S - 0.02:
            hold_frac = float(np.mean([(r["nL"] > 0 and r["nR"] > 0) for r in window]))
            held = bool(
                hold_frac >= 0.95
                and window[-1]["nL"] > 0
                and window[-1]["nR"] > 0
                and seen_free
                and armed
                and float(np.max(gap)) >= GAP_TARGET - 0.001
                and speed_ok
            )
    if held:
        stage = "HELD_1S"
    elif float(np.max(gap)) < GAP_TARGET - 0.001:
        stage = "OPEN_FAILED"
    elif not seen_free:
        stage = "NEVER_RELEASED"
    elif not np.any(between & (np.array([r["t"] for r in rows]) > 0.05)):
        stage = "NEVER_BETWEEN"
    elif re_t is None:
        stage = "NO_RECONTACT"
    elif not speed_ok:
        stage = "SPEED_MISMATCH"
    else:
        stage = "HOLD_FAILED"
    vh = np.stack([r["vh"] for r in rows])
    vo = np.stack([r["vo"] for r in rows])
    tau = np.stack([r["tau_j"] for r in rows])
    close_rows = [r for r in rows if r["phase"] == "close"]
    return {
        "spec": {k: v for k, v in spec.items() if k != "horizon_s"},
        "stage": stage,
        "held_1s": held,
        "diagnostic_only": True,
        "legal_task": False,
        "airborne_release": bool(seen_free),
        "free_s": round(float(free_s), 4),
        "opened_to_49mm": bool(armed),
        "vrel_at_recontact": vrel_re,
        "recontact_t": None if re_t is None else round(float(re_t), 4),
        "hold_t": None if hold_t is None else round(float(hold_t), 4),
        "hold_bilateral_fraction": round(hold_frac, 3),
        "max_gap_m": round(float(np.max(gap)), 4),
        "gap_reached_49mm": bool(np.max(gap) >= GAP_TARGET - 0.001),
        "max_between_after_10ms": bool(np.any(between[1:])),
        "min_cz_m": round(float(np.min([r["cz"] for r in rows])), 4),
        "peak_hand_speed": round(float(np.max(np.linalg.norm(vh, axis=1))), 3),
        "peak_hand_vz": round(float(np.min(vh[:, 2])), 3),
        "peak_obj_vz": round(float(np.min(vo[:, 2])), 3),
        "sat_fraction": round(float(np.mean([r["sat"] > 0 for r in rows])), 3),
        "peak_tau_j2": round(float(np.max(np.abs(tau[:, 1]))), 2),
        "peak_tau_j4": round(float(np.max(np.abs(tau[:, 3]))), 2),
        "close_t": None if not close_rows else round(float(close_rows[0]["t"]), 4),
        "end_t": round(float(rows[-1]["t"]), 4),
        "end_gap_m": round(float(rows[-1]["gap"]), 4),
        "end_n": [int(rows[-1]["nL"]), int(rows[-1]["nR"])],
        "end_between": int(rows[-1]["between"]),
        "end_cz_m": round(float(rows[-1]["cz"]), 4),
        "end_vh": [round(float(x), 3) for x in rows[-1]["vh"]],
        "end_vo": [round(float(x), 3) for x in rows[-1]["vo"]],
        "end_obj_z": round(float(rows[-1]["zo"]), 4),
        "rows": rows,
    }


def _save_trace(name, result):
    rows = result["rows"]
    path = OUT / "traces" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in rows], np.float32),
        phase=np.array([r["phase"] for r in rows]),
        gap=np.array([r["gap"] for r in rows], np.float32),
        finger_q=np.array([r["finger_q"] for r in rows], np.float32),
        between=np.array([r["between"] for r in rows], np.int8),
        nL=np.array([r["nL"] for r in rows], np.int16),
        nR=np.array([r["nR"] for r in rows], np.int16),
        cx=np.array([r["cx"] for r in rows], np.float32),
        cy=np.array([r["cy"] for r in rows], np.float32),
        cz=np.array([r["cz"] for r in rows], np.float32),
        axial=np.array([r["axial"] for r in rows], np.float32),
        vh=np.stack([r["vh"] for r in rows]).astype(np.float32),
        vo=np.stack([r["vo"] for r in rows]).astype(np.float32),
        tau=np.array([r["tau"] for r in rows], np.float32),
        v_cmd=np.stack([r["v_cmd"] for r in rows]).astype(np.float32),
        p_err=np.array([r["p_err"] for r in rows], np.float32),
        sat=np.array([r["sat"] for r in rows], np.int8),
        tau_j=np.stack([r["tau_j"] for r in rows]).astype(np.float32),
        qpos=np.stack([r["qpos"] for r in rows]).astype(np.float32),
    )
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _render(sim, result, dest, title):
    from demo.demo_ballistic_recovery_comparison import caption
    from demo.demo_grav_reposition_comparison import write_frames_mp4
    from training.impact_visualization_utils import mjv_camera_from_preset

    rows = result["rows"]
    renderer = mujoco.Renderer(sim.model, 480, 640)
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    cam = mjv_camera_from_preset({"lookat": hand, "distance": 0.55, "azimuth": 90.0, "elevation": -10.0})
    frames = []
    next_t = 0.0
    qpos0 = np.array(sim.data.qpos, float).copy()
    qvel0 = np.array(sim.data.qvel, float).copy()
    for row in rows:
        if row["t"] + 1e-9 < next_t:
            continue
        sim.data.qpos[:] = row["qpos"]
        sim.data.qvel[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        renderer.update_scene(sim.data, cam)
        text = f"{title} {row['phase']} t={row['t']:.3f} gap={row['gap']*1e3:.0f}mm n={row['nL']}/{row['nR']}"
        frames.append(caption(renderer.render(), text))
        next_t += 1.0 / 25.0
    sim.data.qpos[:] = qpos0
    sim.data.qvel[:] = qvel0
    mujoco.mj_forward(sim.model, sim.data)
    renderer.close()
    return write_frames_mp4(frames, dest)


def _public(result, trace):
    out = {k: v for k, v in result.items() if k != "rows"}
    out["trace"] = trace
    return out


def _base_grid():
    """Historical intercept, but close only after a 49 mm opening and 40 ms of lost contact."""
    specs = []
    for aim, down, k, corr, tag in (
        ("above", 0.0, 8.0, 1.0, "above12_k8_corr1"),
        ("below", 0.015, 8.0, 1.0, "down15_k8_corr1"),
        ("below", 0.015, 8.0, None, "down15_k8_sumclip_only"),
    ):
        specs.append({
            "name": f"C_{tag}_open18",
            "interface": "C",
            "aim": aim,
            "aim_down": down,
            "k": k,
            "v_clip": 4.5,
            "corr_clip": corr,
            "open_tau": 18.0,
        })
    return specs


def _ladder(base):
    """Change one interface factor at a time, starting from a holding law."""
    common = {
        "aim": base["aim"],
        "aim_down": base.get("aim_down", 0.0),
        "k": base["k"],
        "v_clip": base["v_clip"],
        "corr_clip": base.get("corr_clip"),
    }
    return [
        {**common, "name": "L1_open_torque_2Nm", "interface": "C", "open_tau": 2.0},
        {**common, "name": "L2_v3_speed_clip_2ms", "interface": "C_v3speed", "open_tau": base["open_tau"]},
        {**common, "name": "L3_legal_B_integrate_pdes", "interface": "B", "open_tau": 2.0, "v_clip": None},
        {**common, "name": "L4_legal_A_reset_pdes", "interface": "A", "open_tau": 2.0, "v_clip": None},
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    sim, cfg = make_sim()
    assert_noslip(sim)
    gains = gains_from_cfg(cfg)
    sim._nom_gains = gains
    nominal = capture_nominal(sim)
    pads, pad_mu = capture_pads(sim)
    seated = grasp_settled(sim, gains)
    if seated is None:
        raise RuntimeError("settled horizontal grasp was not reached")
    drop_supports(sim)
    snap = stamp(sim)
    geo0 = capture_geometry(sim)
    origin = {
        "supports": "floor and table contype/conaffinity set to 0 after the seated lift",
        "obj_z": round(float(sim.data.xpos[sim.ids.object_body][2]), 4),
        "gap_m": round(float(geo0["inner_gap_m"]), 4),
        "finger_q": [round(float(v), 4) for v in geo0["finger_qpos_m"]],
        "n": [int(geo0["nL"]), int(geo0["nR"])],
        "between": bool(geo0["between"]),
        "center_hand_m": [round(float(v), 4) for v in geo0["center_hand_m"]],
        "open_torque_historical_nm": 18.0,
        "open_torque_legal_nm": 2.0,
        "qpos_write": False,
    }
    print("ORIGIN", json.dumps(origin), flush=True)
    results = []
    best = None
    for spec in _base_grid():
        result = rollout(sim, snap, nominal, pads, pad_mu, spec)
        trace = _save_trace(spec["name"], result)
        pub = _public(result, trace)
        results.append(pub)
        print("CASE", spec["name"], pub["stage"], "gap", pub["max_gap_m"], "free", pub["free_s"], "re", pub["recontact_t"], "vrel", pub["vrel_at_recontact"], "vh", pub["peak_hand_vz"], flush=True)
        if best is None or (pub["held_1s"], pub["recontact_t"] is not None, pub["max_gap_m"]) > (
            best["held_1s"], best["recontact_t"] is not None, best["max_gap_m"]
        ):
            best = pub
            best_rows = result
        if pub["held_1s"]:
            break
    ladder = []
    if best and best["held_1s"]:
        for spec in _ladder(best["spec"]):
            result = rollout(sim, snap, nominal, pads, pad_mu, spec)
            trace = _save_trace(spec["name"], result)
            pub = _public(result, trace)
            ladder.append(pub)
            print("LADDER", spec["name"], pub["stage"], "gap", pub["max_gap_m"], "re", pub["recontact_t"], flush=True)
            if (pub["held_1s"], pub["max_gap_m"]) > (best["held_1s"], best["max_gap_m"]):
                best = pub
                best_rows = result
    video = None
    if best_rows is not None:
        dest = OUT / "videos" / f"{best['spec']['name']}__{best['stage']}.mp4"
        video = _render(sim, best_rows, dest, best["spec"]["name"])
    summary = {"origin": origin, "grid": results, "ladder": ladder, "best": best["spec"]["name"] if best else None, "video": video}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", best["spec"]["name"] if best else None, best["stage"] if best else None, flush=True)


def impact_probe():
    """One real TRAIN impact, first without the table, then with TABLE_DROP watched."""
    import pickle

    import torch

    from training.autonomous_episode import make_training_env
    from training.history_aware_critic_ablation import restore_capture
    from training.table_drop import latched
    from training.unified_k48 import load_trained_k48

    path = ROOT / "data" / "snapshots" / "train" / "ball_train_v2_0002.pkl"
    device = torch.device("cpu")
    model, _saved = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", device)
    model.eval()
    env = make_training_env(device, model=model)
    with path.open("rb") as handle:
        blob = pickle.load(handle)
    spec = {
        "name": "impact_0002_C",
        "interface": "C",
        "aim": "below",
        "aim_down": 0.015,
        "k": 8.0,
        "v_clip": 4.5,
        "corr_clip": 1.0,
        "open_tau": 18.0,
        "rh": [0.0, 0.0, 0.098],
    }
    rows = []
    for table in (False, True):
        restore_capture(env, blob["capture"])
        env.sim._watch_drop = bool(table)
        if not table:
            drop_supports(env.sim)
        spec = dict(spec)
        spec["name"] = "impact_0002_C_" + ("table" if table else "notable")
        result = rollout(env.sim, None, None, None, None, spec, already=True)
        trace = _save_trace(spec["name"], result)
        pub = _public(result, trace)
        pub["table_collision"] = bool(table)
        pub["table_drop_latched"] = bool(latched(env.sim)) if table else False
        pub["legal_task"] = False
        rows.append(pub)
        print("IMPACT", spec["name"], pub["stage"], "gap", pub["max_gap_m"], "free", pub["free_s"], "re", pub["recontact_t"], "latch", pub["table_drop_latched"], flush=True)
    (OUT / "impact.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _one(spec):
    sim, cfg = make_sim()
    assert_noslip(sim)
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pads, pad_mu = capture_pads(sim)
    seated = grasp_settled(sim, gains)
    if seated is None:
        raise RuntimeError("settled horizontal grasp was not reached")
    drop_supports(sim)
    snap = stamp(sim)
    result = rollout(sim, snap, nominal, pads, pad_mu, spec)
    trace = _save_trace(spec["name"], result)
    pub = _public(result, trace)
    print("ONE", spec["name"], pub["stage"], "endz", pub["end_obj_z"], "frac", pub["hold_bilateral_fraction"], "endn", pub["end_n"], flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "impact":
        impact_probe()
    elif len(sys.argv) > 1 and sys.argv[1] == "l1":
        _one({
            "name": "L1_open_torque_2Nm",
            "interface": "C",
            "aim": "below",
            "aim_down": 0.015,
            "k": 8.0,
            "v_clip": 4.5,
            "corr_clip": 1.0,
            "open_tau": 2.0,
        })
    elif len(sys.argv) > 1 and sys.argv[1] == "ab":
        common = {
            "aim": "below",
            "aim_down": 0.015,
            "k": 8.0,
            "v_clip": None,
            "corr_clip": 1.0,
            "open_tau": 2.0,
        }
        _one({**common, "name": "L3_legal_B_integrate_pdes", "interface": "B"})
        _one({**common, "name": "L4_legal_A_reset_pdes", "interface": "A"})
    else:
        main()
