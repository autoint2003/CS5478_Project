"""Uncapped Cartesian chase from the 13.5 mm natural-loss state.

The ±1 m/s action-map cap is not applied. Torques still clip to the Panda
actuators. No policy training. The loss state is the same closed-hold replay
as the separation audit, checked against those scalars, then saved and reloaded.
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
from envs.airborne_obs import relative_state
from envs.deterioration import body_twist
from envs.physical_recovery import physical_pack
from training.demo_ballistic_impact import assert_noslip, make_sim
from training.full_3d_airborne_recatch_dataset import (
    TRACK,
    begin,
    capture_nominal,
    capture_pads,
    drop_supports,
    grasp_settled,
    hold_step,
    restore_to,
)
from training.long_context_recovery_pilot import jsonable
from training.recovery_runtime import gains_from_cfg, stamp

OUT = ROOT / "results" / "diagnostics" / "raw" / "uncapped_cartesian_airborne_chase"
DT = 0.002
G = 9.81
K_INT = 200.0
# Setpoint used only so the velocity servo saturates the actuators.
# It is not a claimed hand-speed target.
V_SAT = 30.0


def limits(sim):
    lim = np.array(sim.model.actuator_ctrlrange[:7, 1], float)
    q = np.array(sim.data.qpos[sim.ids.arm_jnt], float)
    qd = np.array(sim.data.qvel[sim.ids.arm_dof], float)
    lo = np.array([sim.model.jnt_range[int(j), 0] for j in sim.ids.arm_jnt])
    hi = np.array([sim.model.jnt_range[int(j), 1] for j in sim.ids.arm_jnt])
    margin = np.minimum(q - lo, hi - q)
    return lim, q, qd, margin


def step_cart(sim, p_des, v_des, tau):
    sim.fsm.p_des = np.asarray(p_des, float).copy()
    sim.fsm.v_des = np.asarray(v_des, float).copy()
    sim.fsm.v_cmd = np.asarray(v_des, float).copy()
    ctrl = apply_cartesian_ctrl(
        sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
        gripper_tau=float(tau), v_des=v_des, w_des=np.zeros(3), **TRACK,
    )
    mujoco.mj_step(sim.model, sim.data)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
    return ctrl


def target_hand(sim, pinch):
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    return po - Rd @ np.asarray(pinch, float)


def command(sim, pinch, kind, cap):
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    p_des = target_hand(sim, pinch)
    e = p_des - ph
    vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
    if kind == "maxdown":
        v = vo.copy()
        v[2] = float(vo[2]) - V_SAT
    else:
        v = vo + K_INT * e
    if cap is not None:
        v = np.clip(v, -float(cap), float(cap))
    return p_des, v, e, vo


def preset_fingers_open(sim):
    sim.data.qpos[sim.ids.finger_jnt] = 0.04
    sim.data.qvel[sim.ids.finger_dof] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def rollout(sim, pinch, kind, cap, open_mode):
    from training.relative_separation_chase_audit import robot_object_contacts

    t0 = float(sim.data.time)
    if open_mode == "preset":
        preset_fingers_open(sim)
    phase = "chase"
    both_hold = 0.0
    hold_t = None
    re_t = None
    rows = []
    prev_vh = None
    prev_vo = None
    lim0, _, _, _ = limits(sim)
    for _ in range(int(1.6 / DT)):
        o = physical_pack(sim)
        rel = relative_state(sim.model, sim.data, sim.ids)
        rh = rel["p_rel_h"]
        excess = float(rh[2] - pinch[2])
        aper = float(o["aperture"])
        vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
        vh = np.array(body_twist(sim.model, sim.data, sim.ids.hand_body)[0], float)
        vrel = vo - vh
        slow = abs(float(vrel[2])) < 0.15 and float(np.linalg.norm(vrel[:2])) < 0.20
        near = abs(excess) < 0.012 and abs(float(rh[0] - pinch[0])) < 0.012 and abs(float(rh[1] - pinch[1])) < 0.012
        both = int(o["nL"]) > 0 and int(o["nR"]) > 0
        if phase == "chase" and aper > 0.036 and near and slow:
            phase = "close"
        if phase == "close":
            if both:
                if re_t is None:
                    re_t = float(sim.data.time) - t0
                both_hold += DT
                if both_hold >= 0.04:
                    phase = "hold"
                    if hold_t is None:
                        hold_t = float(sim.data.time) - t0
            else:
                both_hold = 0.0
        if phase in ("close", "hold"):
            p_des, v_des, e, vo = command(sim, pinch, "intercept", cap)
            v_des = vo + 8.0 * e
            if cap is not None:
                v_des = np.clip(v_des, -float(cap), float(cap))
            if phase == "hold":
                v_des = vo + 2.0 * e
                if cap is not None:
                    v_des = np.clip(v_des, -float(cap), float(cap))
            tau = -18.0
        else:
            p_des, v_des, e, vo = command(sim, pinch, kind, cap)
            tau = 2.0
        ctrl = step_cart(sim, p_des, v_des, tau)
        vh2 = np.array(body_twist(sim.model, sim.data, sim.ids.hand_body)[0], float)
        vo2 = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
        az_h = 0.0 if prev_vh is None else float((vh2[2] - prev_vh) / DT)
        az_o = 0.0 if prev_vo is None else float((vo2[2] - prev_vo) / DT)
        prev_vh, prev_vo = float(vh2[2]), float(vo2[2])
        _, _, qd, margin = limits(sim)
        rel2 = relative_state(sim.model, sim.data, sim.ids)["p_rel_h"]
        po = float(sim.data.xpos[sim.ids.object_body][2])
        ph = float(sim.data.xpos[sim.ids.hand_body][2])
        sat = np.abs(ctrl[:7]) >= 0.95 * lim0
        rows.append({
            "t": float(sim.data.time) - t0,
            "zh": ph,
            "zo": po,
            "relz": po - ph,
            "excess": float(rel2[2] - pinch[2]),
            "vhz": float(vh2[2]),
            "voz": float(vo2[2]),
            "vrelz": float(vo2[2] - vh2[2]),
            "azh": az_h,
            "azo": az_o,
            "arel": az_o - az_h,
            "aper": float(physical_pack(sim)["aperture"]),
            "nL": int(physical_pack(sim)["nL"]),
            "nR": int(physical_pack(sim)["nR"]),
            "phase": phase,
            "cmdz": float(v_des[2]),
            "qd": qd.copy(),
            "tau": np.array(ctrl[:7], float).copy(),
            "sat": int(np.sum(sat)),
            "margin": margin.copy(),
            "ncon": len(robot_object_contacts(sim)),
        })
        if hold_t is not None and rows[-1]["t"] >= hold_t + 1.0:
            break
        if po < -0.15 or rows[-1]["t"] > 0.70 and phase == "chase":
            break
    return rows, re_t, hold_t


def pack(rows, re_t, hold_t, lim):
    # skip the first sample (az not yet differenced)
    sl = rows[1:]
    azh = np.array([r["azh"] for r in sl])
    azo = np.array([r["azo"] for r in sl])
    arel = np.array([r["arel"] for r in sl])
    ex = np.array([r["excess"] for r in rows])
    i_min = int(np.argmin(ex))
    below = azh < -(G + 0.3)
    dur = float(np.sum(below) * DT)
    both_tail = []
    if hold_t is not None:
        both_tail = [r for r in rows if r["t"] >= hold_t - 1e-9]
    held = bool(
        hold_t is not None
        and rows[-1]["t"] >= hold_t + 0.95
        and both_tail
        and all(r["nL"] > 0 and r["nR"] > 0 for r in both_tail)
    )
    qd = np.stack([r["qd"] for r in rows])
    tau = np.stack([r["tau"] for r in rows])
    margin = np.stack([r["margin"] for r in rows])
    # 20 ms table for the first 0.24 s
    table = []
    for r in rows:
        if abs(r["t"] / 0.02 - round(r["t"] / 0.02)) < 0.02 and r["t"] <= 0.24:
            table.append({
                "t": round(r["t"], 3),
                "excess_mm": round(r["excess"] * 1e3, 2),
                "relz_mm": round(r["relz"] * 1e3, 1),
                "vhz": round(r["vhz"], 3),
                "voz": round(r["voz"], 3),
                "vrelz": round(r["vrelz"], 3),
                "azh": round(r["azh"], 2),
                "azo": round(r["azo"], 2),
                "arel": round(r["arel"], 2),
                "aper_mm": round(r["aper"] * 1e3, 1),
                "cmdz": round(r["cmdz"], 2),
                "sat": int(r["sat"]),
                "n": [int(r["nL"]), int(r["nR"])],
                "ncon": int(r["ncon"]),
            })
    return {
        "t_end": round(float(rows[-1]["t"]), 3),
        "excess0_mm": round(float(rows[0]["excess"]) * 1e3, 2),
        "min_excess_mm": round(float(ex[i_min]) * 1e3, 2),
        "t_min_excess": round(float(rows[i_min]["t"]), 3),
        "vrel_at_min": round(float(rows[i_min]["vrelz"]), 3),
        "pushed_at_min": bool(rows[i_min]["ncon"] > 0),
        "peak_azh": round(float(np.min(azh)), 2),
        "peak_down_speed": round(float(np.min([r["vhz"] for r in rows])), 3),
        "peak_obj_speed": round(float(np.min([r["voz"] for r in rows])), 3),
        "dur_faster_than_g": round(dur, 3),
        "mean_arel_0_80ms": round(float(np.mean(arel[:40])), 2) if len(arel) >= 40 else None,
        "mean_azh_0_80ms": round(float(np.mean(azh[:40])), 2) if len(azh) >= 40 else None,
        "sat_steps": int(sum(r["sat"] > 0 for r in rows)),
        "sat_joint_steps": [int(np.sum(np.abs(tau[:, j]) >= 0.95 * lim[j])) for j in range(7)],
        "peak_tau": [round(float(x), 2) for x in np.max(np.abs(tau), axis=0)],
        "peak_qd": [round(float(x), 3) for x in np.max(np.abs(qd), axis=0)],
        "min_margin": [round(float(x), 3) for x in np.min(margin, axis=0)],
        "re_t": None if re_t is None else round(float(re_t), 3),
        "hold_t": None if hold_t is None else round(float(hold_t), 3),
        "held_1s": held,
        "end_n": [int(rows[-1]["nL"]), int(rows[-1]["nR"])],
        "end_excess_mm": round(float(rows[-1]["excess"]) * 1e3, 1),
        "end_z": round(float(rows[-1]["zo"]), 3),
        "table": table,
    }


def required_accel(s0, ydot0):
    """s0 is the +excess to close. y = z_obj - z_hand increases when the gap closes.

    ydot0 is initial d/dt(z_obj - z_hand) = voz - vhz.
    Constant a_rel = a_obj - a_hand over horizon T.
    """
    rows = []
    for T in (0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40):
        # need Δy = +s0, starting from ydot0 (negative if the gap is opening)
        a = 2.0 * (s0 - ydot0 * T) / (T * T)
        rows.append({
            "T": T,
            "a_rel": round(a, 2),
            "a_hand_if_obj_is_g": round(-G - a, 2),
        })
    return rows


def save_snap(path, snap):
    arrays = {}
    meta = {}
    for k, v in snap.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (float, int, bool, np.floating, np.integer)):
            meta[k] = v
        else:
            meta[k] = v
    np.savez_compressed(path, **arrays, meta_json=np.array(json.dumps(jsonable(meta))))


def main():
    from training.relative_separation_chase_audit import make_loss

    OUT.mkdir(parents=True, exist_ok=True)
    sim, cfg = make_sim()
    assert_noslip(sim)
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pads, pad_mu = capture_pads(sim)
    settled = grasp_settled(sim, gains)
    begin(sim, settled, nominal, pads, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0})
    for _ in range(40):
        hold_step(sim, gains)
    pinch = relative_state(sim.model, sim.data, sim.ids)["p_rel_h"].copy()
    base = stamp(sim)
    loss, info = make_loss(sim, gains, nominal, pads, pad_mu, base, float(pinch[2]))
    # Persist the MuJoCo state, then reload it before any chase.
    save_snap(OUT / "natural_loss_13_5.npz", loss)
    print("LOSS", info, flush=True)
    assert abs(info["excess_mm"] - 25.44) < 0.05
    assert abs(info["vo"][2] + 0.0459) < 0.002
    assert info["n_robot_contacts"] == 0
    assert abs(info["t_slip"] - 6.074) < 0.004

    # Joint velocity limit presence.
    limited = []
    for j in sim.ids.arm_jnt:
        j = int(j)
        limited.append({
            "limited": bool(sim.model.jnt_limited[j]),
            "range": [round(float(x), 3) for x in sim.model.jnt_range[j]],
        })

    jobs = [
        ("A_v1_0.08_intercept", "intercept", 0.08, "dynamic"),
        ("B_v2_1.0_intercept", "intercept", 1.0, "dynamic"),
        ("C_uncapped_intercept", "intercept", None, "dynamic"),
        ("C_uncapped_maxdown", "maxdown", None, "dynamic"),
        ("C_uncapped_intercept_preset_open", "intercept", None, "preset"),
        ("C_uncapped_maxdown_preset_open", "maxdown", None, "preset"),
    ]
    summary = {"loss": info, "pinch_mm": [round(float(x) * 1e3, 2) for x in pinch], "joints_limited": limited, "runs": {}}
    lim, _, _, _ = limits(sim)
    summary["tau_lim"] = [round(float(x), 1) for x in lim]
    for name, kind, cap, open_mode in jobs:
        restore_to(sim, loss, nominal, pads, pad_mu)
        drop_supports(sim)
        rows, re_t, hold_t = rollout(sim, pinch, kind, cap, open_mode)
        packed = pack(rows, re_t, hold_t, lim)
        summary["runs"][name] = packed
        np.savez_compressed(
            OUT / f"{name}.npz",
            t=np.array([r["t"] for r in rows]),
            excess=np.array([r["excess"] for r in rows]),
            relz=np.array([r["relz"] for r in rows]),
            vhz=np.array([r["vhz"] for r in rows]),
            voz=np.array([r["voz"] for r in rows]),
            azh=np.array([r["azh"] for r in rows]),
            azo=np.array([r["azo"] for r in rows]),
            arel=np.array([r["arel"] for r in rows]),
            aper=np.array([r["aper"] for r in rows]),
            cmdz=np.array([r["cmdz"] for r in rows]),
            sat=np.array([r["sat"] for r in rows]),
        )
        print(
            name,
            "min_ex", packed["min_excess_mm"],
            "t", packed["t_min_excess"],
            "peak_az", packed["peak_azh"],
            "dur_gt_g", packed["dur_faster_than_g"],
            "peak_vh", packed["peak_down_speed"],
            "mean_arel", packed["mean_arel_0_80ms"],
            "sat", packed["sat_steps"],
            "held", packed["held_1s"],
            "end_ex", packed["end_excess_mm"],
            flush=True,
        )
        for row in packed["table"][:8]:
            print(" ", row, flush=True)

    # Requirement from the reloaded snapshot, before any command.
    restore_to(sim, loss, nominal, pads, pad_mu)
    drop_supports(sim)
    vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
    vh = np.array(body_twist(sim.model, sim.data, sim.ids.hand_body)[0], float)
    s0 = float(info["excess_mm"]) / 1e3
    ydot0 = float(vo[2] - vh[2])
    summary["requirement"] = {
        "s0_m": round(s0, 5),
        "ydot0": round(ydot0, 4),
        "vhz0": round(float(vh[2]), 4),
        "voz0": round(float(vo[2]), 4),
        "horizons": required_accel(s0, ydot0),
    }
    print("REQ", summary["requirement"], flush=True)
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
