"""Legal omega_y=3 joint/actuator/tracking audit. Does not raise the bound."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.bias import arm_bias
from controllers.jacobian_controller import (
    jacobian_6d,
    ori_error_deg,
    rotation_error,
)
from controllers.residual import RECOVERY4D_W_HY_MAX
from envs.config_util import load_yaml, merge_sim_config
from envs.deterioration import body_twist
from training.demo_airborne_recovery import copy_masks, disable_object_table_only, prepare_high_parent
from training.demo_airborne_recapture import make_parent_sim, tick_4d
from training.demo_ballistic_impact import make_sim as make_ballistic_sim
from training.demo_ballistic_recovery import restore_ballistic
from training.demo_dynamic_recatch import EARLY_PKL, TAU_SEC
from training.demo_teleport_recovery_state import signed_rot_about_y_deg
from training.replay_core import freeze

OUT = ROOT / "results" / "diagnostics" / "dynamic_airborne_recatch"
RAW = OUT / "raw"
OMEGA = float(RECOVERY4D_W_HY_MAX)
MAX_DEG = 125.0


def dump(path: Path, obj) -> None:
    def fix(x):
        if isinstance(x, dict):
            return {str(k): fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [fix(v) for v in x]
        if isinstance(x, np.ndarray):
            return fix(x.tolist())
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fix(obj), indent=2), encoding="utf-8")


def model_limits(sim) -> dict:
    m = sim.model
    ids = sim.ids
    names = []
    for j in ids.arm_jnt:
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j))
        names.append(nm)
    jnt_range = np.array(m.jnt_range[ids.arm_jnt], float)
    jnt_limited = np.array(m.jnt_limited[ids.arm_jnt], int)
    vel = np.array(m.jnt_velocity[ids.arm_jnt], float) if hasattr(m, "jnt_velocity") else None
    ctrl = np.stack([ids.ctrl_low[:7], ids.ctrl_high[:7]], axis=1)
    force = np.array(m.actuator_forcerange[:7], float)
    ctrl_r = np.array(m.actuator_ctrlrange[:7], float)
    return {
        "xml": "assets/panda_torque.xml",
        "autolimits": True,
        "joint_names": names,
        "jnt_range": jnt_range.tolist(),
        "jnt_limited": jnt_limited.tolist(),
        "jnt_velocity": None if vel is None else vel.tolist(),
        "jnt_velocity_note": (
            "MuJoCo jnt_velocity==0 means unlimited. XML has no velocity= attributes."
        ),
        "actuator_ctrlrange": ctrl_r.tolist(),
        "actuator_forcerange": force.tolist(),
        "ids_ctrl_low_high": ctrl.tolist(),
        "xml_comment": {
            "joint_default_range": [-2.8973, 2.8973],
            "joint2": [-1.7628, 1.7628],
            "joint4": [-3.0718, -0.0698],
            "joint6": [-0.0175, 3.7525],
            "motors_1_4_Nm": 87.0,
            "motors_5_7_Nm": 12.0,
            "gripper_Nm": 50.0,
        },
    }


def wrench_tau(sim, gains, v_des, w_des):
    ids = sim.ids
    model, data = sim.model, sim.data
    p = data.xpos[ids.hand_body].copy()
    r = data.xmat[ids.hand_body].reshape(3, 3).copy()
    e_p = np.asarray(sim.fsm.p_des, float) - p
    e_o = rotation_error(np.asarray(sim.fsm.r_des, float).reshape(3, 3), r)
    e = np.concatenate([e_p, e_o])
    j = jacobian_6d(model, data, ids)
    qdot = np.array(data.qvel[ids.arm_dof], float)
    xdot = j @ qdot
    vd = np.asarray(v_des, float).reshape(3)
    wd = np.asarray(w_des, float).reshape(3)
    xdot_des = np.concatenate([vd, wd])
    kp = np.array(
        [gains["kp_pos"], gains["kp_pos"], gains["kp_pos"], gains["kp_ori"], gains["kp_ori"], gains["kp_ori"]]
    )
    kd = np.array(
        [gains["kd_pos"], gains["kd_pos"], gains["kd_pos"], gains["kd_ori"], gains["kd_ori"], gains["kd_ori"]]
    )
    wrench = kp * e + kd * (xdot_des - xdot)
    tau = j.T @ wrench + arm_bias(model, data, ids)
    q_ref = ids.home_qpos[ids.arm_jnt]
    lam = 1e-3
    j_pinv = j.T @ np.linalg.inv(j @ j.T + lam * np.eye(6))
    nproj = np.eye(7) - j_pinv @ j
    tau = tau + nproj @ (gains.get("kp_null", 4.0) * (q_ref - data.qpos[ids.arm_jnt]) - gains.get("kd_null", 0.8) * qdot)
    tau_clip = np.clip(tau, ids.ctrl_low[:7], ids.ctrl_high[:7])
    s = np.linalg.svd(j, compute_uv=False)
    s_w = np.linalg.svd(j[3:6, :], compute_uv=False)
    return {
        "tau_unclipped": tau,
        "tau_clipped": tau_clip,
        "J_cond": float(np.linalg.cond(j)),
        "J_smin": float(s.min()),
        "J_smax": float(s.max()),
        "Jw_smin": float(s_w.min()),
        "Jw_smax": float(s_w.max()),
        "Jw": j[3:6, :].copy(),
        "qdot": qdot,
        "q": np.array(data.qpos[ids.arm_jnt], float),
        "p_err": e_p,
        "ori_deg": float(ori_error_deg(np.asarray(sim.fsm.r_des, float).reshape(3, 3), r)),
    }


def sample(sim, gains, omega_y, Rh0, jnt_range, force_lim):
    ids = sim.ids
    r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    r = np.array(sim.data.xmat[ids.hand_body].reshape(3, 3), float)
    p = np.array(sim.data.xpos[ids.hand_body], float)
    p_des = np.asarray(sim.fsm.p_des, float)
    wh = body_twist(sim.model, sim.data, ids.hand_body)[1]
    y_des = r_des[:, 1]
    omega_axis = float(np.dot(wh, y_des))
    omega_hy_hand = float((r.T @ wh)[1])
    w_cmd = r_des @ np.array([0.0, omega_y, 0.0])
    info = wrench_tau(sim, gains, np.zeros(3), w_cmd)
    q = info["q"]
    qdot = info["qdot"]
    tau_u = info["tau_unclipped"]
    tau_c = np.array(sim.data.ctrl[:7], float)
    lim = np.array(force_lim, float)
    sat = np.abs(tau_u) >= (np.abs(lim) - 1e-6)
    mid = 0.5 * (jnt_range[:, 0] + jnt_range[:, 1])
    half = 0.5 * (jnt_range[:, 1] - jnt_range[:, 0])
    prox = np.abs(q - mid) / np.maximum(half, 1e-9)
    y_row = info["Jw"].T @ y_des  # joint contrib to omega about y_des if qdot=1
    return {
        "t": float(sim.data.time),
        "ang_deg": float(signed_rot_about_y_deg(Rh0, r)),
        "omega_y_cmd": float(omega_y),
        "omega_axis_actual": omega_axis,
        "omega_hy_hand": omega_hy_hand,
        "w_hand": np.array(wh, float).tolist(),
        "w_hand_norm": float(np.linalg.norm(wh)),
        "ori_err_deg": info["ori_deg"],
        "p_err": info["p_err"].tolist(),
        "p_err_norm": float(np.linalg.norm(info["p_err"])),
        "p_des": p_des.tolist(),
        "p_hand": p.tolist(),
        "q": q.tolist(),
        "qdot": qdot.tolist(),
        "qdot_abs_max": float(np.max(np.abs(qdot))),
        "tau_unclipped": tau_u.tolist(),
        "tau_applied": tau_c.tolist(),
        "tau_limit": lim.tolist(),
        "tau_util": (np.abs(tau_u) / np.maximum(np.abs(lim), 1e-9)).tolist(),
        "tau_util_max": float(np.max(np.abs(tau_u) / np.maximum(np.abs(lim), 1e-9))),
        "sat_unclipped": sat.astype(int).tolist(),
        "any_sat": bool(np.any(sat)),
        "joint_limit_prox": prox.tolist(),
        "joint_limit_prox_max": float(np.max(prox)),
        "J_cond": info["J_cond"],
        "J_smin": info["J_smin"],
        "Jw_smin": info["Jw_smin"],
        "joint_omega_y_map": y_row.tolist(),
        "joint_omega_y_share": (y_row * qdot).tolist(),
    }


def summarize(rows: list[dict], limits: dict) -> dict:
    if not rows:
        return {}
    qdot = np.array([r["qdot"] for r in rows], float)
    util = np.array([r["tau_util"] for r in rows], float)
    omega_a = np.array([r["omega_axis_actual"] for r in rows], float)
    cmd = np.array([r["omega_y_cmd"] for r in rows], float)
    moving = np.abs(cmd) > 0.5
    om = omega_a[moving] if np.any(moving) else omega_a
    ori = np.array([r["ori_err_deg"] for r in rows], float)
    perr = np.array([r["p_err_norm"] for r in rows], float)
    cond = np.array([r["J_cond"] for r in rows], float)
    prox = np.array([r["joint_limit_prox_max"] for r in rows], float)
    sat_n = int(sum(1 for r in rows if r["any_sat"]))
    qdot_abs_max_t = float(np.max(np.abs(qdot)))
    qdot_peak_j = np.max(np.abs(qdot), axis=0)
    util_peak_j = np.max(util, axis=0)
    # peak |qdot| row
    i_qd = int(np.argmax(np.max(np.abs(qdot), axis=1)))
    i_tau = int(np.argmax([r["tau_util_max"] for r in rows]))
    i_om = int(np.argmax(np.abs(omega_a))) if rows else 0
    marks = {}
    for tgt in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0):
        i = int(np.argmin([abs(r["ang_deg"] - tgt) for r in rows]))
        marks[str(tgt)] = rows[i]
    return {
        "n": len(rows),
        "omega_cmd": float(OMEGA),
        "omega_axis_min_while_cmd": float(np.min(om)),
        "omega_axis_max_while_cmd": float(np.max(om)),
        "omega_axis_median_while_cmd": float(np.median(om)),
        "ori_err_deg_max": float(np.max(ori)),
        "ori_err_deg_median": float(np.median(ori)),
        "p_err_m_max": float(np.max(perr)),
        "p_err_m_median": float(np.median(perr)),
        "J_cond_max": float(np.max(cond)),
        "J_cond_median": float(np.median(cond)),
        "joint_limit_prox_max": float(np.max(prox)),
        "qdot_abs_peak": qdot_abs_max_t,
        "qdot_peak_per_joint": qdot_peak_j.tolist(),
        "tau_util_peak_per_joint": util_peak_j.tolist(),
        "tau_util_peak": float(np.max(util_peak_j)),
        "n_steps_unclipped_sat": sat_n,
        "xml_jnt_velocity": limits["jnt_velocity"],
        "at_peak_qdot": rows[i_qd],
        "at_peak_tau_util": rows[i_tau],
        "at_peak_omega_axis": rows[i_om],
        "marks_deg": marks,
    }


def run_swing(sim, gains, restore_fn, label: str, limits: dict):
    restore_fn()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    jnt_range = np.array(limits["jnt_range"], float)
    force = np.abs(np.array(limits["actuator_forcerange"], float)[:, 1])
    rows = [sample(sim, gains, 0.0, Rh0, jnt_range, force)]
    timeout = float(sim.data.time) + np.deg2rad(MAX_DEG) / OMEGA + 1.0
    while float(sim.data.time) < timeout:
        tick_4d(sim, gains, OMEGA, 0.0, 0.0, TAU_SEC)
        r = sample(sim, gains, OMEGA, Rh0, jnt_range, force)
        rows.append(r)
        if abs(r["ang_deg"]) >= MAX_DEG - 0.5:
            break
    summ = summarize(rows, limits)
    summ["label"] = label
    summ["t0"] = rows[0]["t"]
    summ["t1"] = rows[-1]["t"]
    summ["ang_end"] = rows[-1]["ang_deg"]
    return rows, summ


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = {
        "kp_pos": float(cfg["cartesian"]["kp_pos"]),
        "kd_pos": float(cfg["cartesian"]["kd_pos"]),
        "kp_ori": float(cfg["cartesian"]["kp_ori"]),
        "kd_ori": float(cfg["cartesian"]["kd_ori"]),
        "kp_null": float(cfg["cartesian"].get("kp_null", 4.0)),
        "kd_null": float(cfg["cartesian"].get("kd_null", 0.8)),
    }

    sim_c = make_parent_sim()
    orig = copy_masks(sim_c)
    prep = prepare_high_parent(sim_c, cfg, gains, orig)
    if not prep.get("ok"):
        raise RuntimeError(prep)
    parent = prep["parent"]
    limits = model_limits(sim_c)

    def restore_cen():
        sim_c.load_snapshot(parent)
        mujoco.mj_forward(sim_c.model, sim_c.data)
        freeze(sim_c)
        disable_object_table_only(sim_c)

    rows_c, sum_c = run_swing(sim_c, gains, restore_cen, "high_clearance_centered", limits)

    if not EARLY_PKL.is_file():
        raise FileNotFoundError(EARLY_PKL)
    with EARLY_PKL.open("rb") as f:
        early = pickle.load(f)
    sim_e, _ = make_ballistic_sim()
    limits_e = model_limits(sim_e)

    def restore_e():
        restore_ballistic(sim_e, early)
        if hasattr(sim_e, "park_ball"):
            sim_e.park_ball()
        mujoco.mj_forward(sim_e.model, sim_e.data)
        freeze(sim_e)
        disable_object_table_only(sim_e)

    rows_e, sum_e = run_swing(sim_e, gains, restore_e, "center6_early", limits_e)

    r_eff = 0.099
    g = 9.81
    kin = []
    for w in (3.0, 4.0, 5.0, 6.0):
        vr = w * r_eff
        kin.append(
            {
                "omega": w,
                "r_eff_m": r_eff,
                "omega_r_mps": vr,
                "ideal_delta_z_m": (vr * vr) / (2.0 * g),
                "label": "IDEALIZED UPPER-SCALE ESTIMATE, not a simulation",
            }
        )

    out = {
        "omega_bound": OMEGA,
        "provenance": {
            "file": "controllers/residual.py",
            "lines": "104-106",
            "constant": "RECOVERY4D_W_HY_MAX = 3.0",
            "comment": (
                "recovery4d: independent of 7d dw_max=0.6. RULE 90 deg about hand-y in "
                "t_align_max=0.6 s needs ~2.62 rad/s; 3.0 rad/s reaches 90 deg in 0.52 s."
            ),
            "git_blame": "Not Committed Yet (working tree)",
            "yaml_duplicate": "config/nominal.yaml recovery.w_hy_max: 3.0 (not read by map_recovery4d)",
            "7d_dw_max": "config/nominal.yaml recovery.dw_max: [0.6, 0.6, 0.6]",
            "rule_t_align_max": "config/rule_based_recovery.yaml t_align_max: 0.6",
        },
        "limits": limits,
        "centered": sum_c,
        "center6_early": sum_e,
        "kinematic_scale": kin,
    }
    dump(RAW / "omega_authority_audit.json", out)
    np.savez_compressed(
        RAW / "omega_authority_centered.npz",
        t=np.array([r["t"] for r in rows_c]),
        ang=np.array([r["ang_deg"] for r in rows_c]),
        omega_cmd=np.array([r["omega_y_cmd"] for r in rows_c]),
        omega_axis=np.array([r["omega_axis_actual"] for r in rows_c]),
        ori_err=np.array([r["ori_err_deg"] for r in rows_c]),
        p_err=np.array([r["p_err_norm"] for r in rows_c]),
        qdot=np.array([r["qdot"] for r in rows_c]),
        tau_u=np.array([r["tau_unclipped"] for r in rows_c]),
        tau_a=np.array([r["tau_applied"] for r in rows_c]),
        tau_util=np.array([r["tau_util"] for r in rows_c]),
        J_cond=np.array([r["J_cond"] for r in rows_c]),
        prox=np.array([r["joint_limit_prox_max"] for r in rows_c]),
    )
    np.savez_compressed(
        RAW / "omega_authority_early.npz",
        t=np.array([r["t"] for r in rows_e]),
        ang=np.array([r["ang_deg"] for r in rows_e]),
        omega_cmd=np.array([r["omega_y_cmd"] for r in rows_e]),
        omega_axis=np.array([r["omega_axis_actual"] for r in rows_e]),
        ori_err=np.array([r["ori_err_deg"] for r in rows_e]),
        p_err=np.array([r["p_err_norm"] for r in rows_e]),
        qdot=np.array([r["qdot"] for r in rows_e]),
        tau_u=np.array([r["tau_unclipped"] for r in rows_e]),
        tau_a=np.array([r["tau_applied"] for r in rows_e]),
        tau_util=np.array([r["tau_util"] for r in rows_e]),
        J_cond=np.array([r["J_cond"] for r in rows_e]),
        prox=np.array([r["joint_limit_prox_max"] for r in rows_e]),
    )
    print("OMEGA", OMEGA, flush=True)
    print("CENTERED omega_axis", sum_c["omega_axis_min_while_cmd"], sum_c["omega_axis_max_while_cmd"], flush=True)
    print("CENTERED qdot_peak", sum_c["qdot_abs_peak"], sum_c["qdot_peak_per_joint"], flush=True)
    print("CENTERED tau_util", sum_c["tau_util_peak"], sum_c["tau_util_peak_per_joint"], "sat", sum_c["n_steps_unclipped_sat"], flush=True)
    print("CENTERED ori/p/cond/prox", sum_c["ori_err_deg_max"], sum_c["p_err_m_max"], sum_c["J_cond_max"], sum_c["joint_limit_prox_max"], flush=True)
    print("EARLY omega_axis", sum_e["omega_axis_min_while_cmd"], sum_e["omega_axis_max_while_cmd"], flush=True)
    print("EARLY qdot_peak", sum_e["qdot_abs_peak"], sum_e["qdot_peak_per_joint"], flush=True)
    print("EARLY tau_util", sum_e["tau_util_peak"], sum_e["tau_util_peak_per_joint"], "sat", sum_e["n_steps_unclipped_sat"], flush=True)
    print("EARLY ori/p/cond/prox", sum_e["ori_err_deg_max"], sum_e["p_err_m_max"], sum_e["J_cond_max"], sum_e["joint_limit_prox_max"], flush=True)
    print("jnt_velocity", limits["jnt_velocity"], flush=True)


if __name__ == "__main__":
    main()
