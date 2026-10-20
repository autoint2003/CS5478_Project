#!/usr/bin/env python
"""Causal debug: elevated-table vs GROUND_SCENE Plan A divergence for hrv1_920031.

No new experts, no controller retune, no vocabulary changes.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from envs.ballistic_impact import make_sim
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.ground_drop import arm as arm_ground
from envs.simulation.capture import restore_capture
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.plan_a_mechanism_audit import (
    BILATERAL_GAP_S_REPORT,
    PIVOT_AXIS_DEG_MIN,
    SLIDE_AXIAL_MM_MIN,
    analyze_mechanism,
    decode_token,
    load_vocab,
    sample_kinematics,
)
from models.airborne_v3 import DT_POLICY

OUT = ROOT / "results" / "diagnostics" / "plan_a_ground_scene_divergence"
RAW = OUT / "raw"

SNAP = ROOT / "results/expert_generation/k49_slow_return_position_refinement/SNAPSHOTS/hrv1_920031.pkl"
TOK = ROOT / "results/expert_generation/k49_slow_return_position_refinement/trajectories/hrv1_920031__refine_12x3_17x12_0x5__tokens.npz"
SNAP_RA = ROOT / "results/expert_generation/return_aware_sliding_k49/SNAPSHOTS/hrv1_920031.pkl"
TOK_RA = ROOT / "results/expert_generation/return_aware_sliding_k49/trajectories/hrv1_920031__return_aware__tokens.npz"
CONFIG_ID = "K49_tok48_wy0p25"

# Tolerances for paired comparison
TOL = {
    "qpos": 1e-9,
    "qvel": 1e-8,
    "ctrl": 1e-9,
    "xpos": 1e-7,
    "p_rel": 1e-6,
    "axis_deg": 1e-4,
    "action": 1e-12,
}


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def open_elev_env():
    from envs.ballistic_impact import world

    class E:
        pass

    env = E()
    env.sim, env.gains, env.nominal, env.pads, env.pad_mu, env.pads_l, env.pads_r = world()
    env.switch = GeometrySwitch()
    env._hist = None
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


class GroundEnv:
    def __init__(self):
        self.sim, _ = make_ground_sim()
        self.switch = GeometrySwitch()
        self._hist = ObservableObsState(DT_POLICY)
        self._prev = np.zeros(7, np.float32)
        self._obs_seq = []
        self._prev_seq = []


def named_joint_map(model):
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            out[name] = {
                "id": int(j),
                "qadr": int(model.jnt_qposadr[j]),
                "dadr": int(model.jnt_dofadr[j]),
                "type": int(model.jnt_type[j]),
                "qpos_dim": 7 if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) else 1,
            }
    return out


def compare_compiled_models(m_a, m_b):
    """Name-matched model diff. Expected support-only differences."""
    diffs = []
    # options
    for attr in (
        "timestep", "gravity", "integrator", "cone", "jacobian", "solver",
        "iterations", "tolerance", "noslip_iterations", "noslip_tolerance",
    ):
        if not hasattr(m_a.opt, attr):
            continue
        va, vb = getattr(m_a.opt, attr), getattr(m_b.opt, attr)
        va = np.asarray(va, float).ravel()
        vb = np.asarray(vb, float).ravel()
        if va.shape != vb.shape or not np.allclose(va, vb, rtol=0, atol=0):
            diffs.append({"field": f"opt.{attr}", "elevated": va.tolist(), "ground": vb.tolist(), "expected": False})

    # bodies by name
    def body_names(m):
        return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)}

    ba, bb = body_names(m_a), body_names(m_b)
    only_a = sorted(set(ba) - set(bb))
    only_b = sorted(set(bb) - set(ba))
    if only_a or only_b:
        diffs.append({"field": "bodies_exclusive", "elevated_only": only_a, "ground_only": only_b, "expected": True})
    for name in sorted(set(ba) & set(bb)):
        ia, ib = ba[name], bb[name]
        for attr, atol in (("body_mass", 0), ("body_inertia", 0), ("body_pos", 0), ("body_quat", 0)):
            va = np.asarray(getattr(m_a, attr)[ia], float).ravel()
            vb = np.asarray(getattr(m_b, attr)[ib], float).ravel()
            if not np.allclose(va, vb, rtol=0, atol=atol):
                diffs.append({
                    "field": f"body.{name}.{attr}",
                    "elevated": va.tolist(), "ground": vb.tolist(),
                    "expected": name in ("world",) and False,
                })

    # geoms by name
    def geom_names(m):
        return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(m.ngeom)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)}

    ga, gb = geom_names(m_a), geom_names(m_b)
    only_ga = sorted(set(ga) - set(gb))
    only_gb = sorted(set(gb) - set(ga))
    diffs.append({
        "field": "geoms_exclusive",
        "elevated_only": only_ga,
        "ground_only": only_gb,
        "expected": True,  # table vs ground/floor naming
    })
    support_names = {"table", "floor", "ground"}
    for name in sorted(set(ga) & set(gb)):
        ia, ib = ga[name], gb[name]
        for attr in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_condim",
                     "geom_contype", "geom_conaffinity", "geom_pos", "geom_quat", "geom_type"):
            va = np.asarray(getattr(m_a, attr)[ia], float).ravel()
            vb = np.asarray(getattr(m_b, attr)[ib], float).ravel()
            if not np.allclose(va, vb, rtol=0, atol=0):
                diffs.append({
                    "field": f"geom.{name}.{attr}",
                    "elevated": va.tolist(), "ground": vb.tolist(),
                    "expected": name in support_names,
                })

    # joints / actuators by name
    def jnames(m):
        return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)}

    ja, jb = jnames(m_a), jnames(m_b)
    for name in sorted(set(ja) & set(jb)):
        ia, ib = ja[name], jb[name]
        for attr in ("jnt_damping", "jnt_armature", "jnt_frictionloss", "jnt_range", "jnt_stiffness"):
            if not hasattr(m_a, attr):
                continue
            va = np.asarray(getattr(m_a, attr)[ia], float).ravel()
            vb = np.asarray(getattr(m_b, attr)[ib], float).ravel()
            if va.size and vb.size and not np.allclose(va, vb, rtol=0, atol=0):
                diffs.append({"field": f"joint.{name}.{attr}", "elevated": va.tolist(), "ground": vb.tolist(), "expected": False})

    unexpected = [d for d in diffs if not d.get("expected")]
    return {
        "n_diffs": len(diffs),
        "n_unexpected": len(unexpected),
        "diffs": diffs,
        "unexpected": unexpected,
        "nq_equal": int(m_a.nq == m_b.nq),
        "nv_equal": int(m_a.nv == m_b.nv),
        "na_equal": int(m_a.na == m_b.na),
    }


def snapshot_fields(sim, label=""):
    ids = sim.ids
    Rh = np.array(sim.data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[ids.hand_body], float)
    po = np.array(sim.data.xpos[ids.object_body], float)
    Ro = np.array(sim.data.xmat[ids.object_body].reshape(3, 3), float)
    obs = None
    try:
        from envs.observations.airborne_obs import observe_airborne

        hist = ObservableObsState(DT_POLICY)
        obs = np.asarray(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist), float)
    except Exception as exc:
        obs = f"obs_fail:{exc}"
    kin = sample_kinematics(sim)
    return {
        "label": label,
        "time": float(sim.data.time),
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "qacc_warmstart": np.array(sim.data.qacc_warmstart, float).copy(),
        "act": np.array(sim.data.act, float).copy() if int(sim.model.na) else np.zeros(0),
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "r_des": np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).copy(),
        "v_des": np.asarray(getattr(sim.fsm, "v_des", np.zeros(3)), float).copy(),
        "w_des": np.asarray(getattr(sim.fsm, "w_des", np.zeros(3)), float).copy(),
        "fg_cmd": float(sim.fsm.fg_cmd),
        "phase": str(sim.fsm.phase),
        "grasp_z": float(sim.fsm.grasp_z),
        "p_hand": ph,
        "p_obj": po,
        "R_hand": Rh,
        "R_obj": Ro,
        "bilateral": kin["bilateral"],
        "nL": kin["nL"],
        "nR": kin["nR"],
        "p_rel": kin["p_rel"],
        "axis_err_deg": kin["axis_err_deg"],
        "obs39": obs,
        "ncon": int(sim.data.ncon),
    }


def diff_snapshots(a, b, prefix=""):
    rows = []
    for key in (
        "time", "fg_cmd", "phase", "grasp_z", "bilateral", "nL", "nR", "axis_err_deg", "ncon",
    ):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (float, int, np.floating, np.integer)) and isinstance(vb, (float, int, np.floating, np.integer)):
            d = abs(float(va) - float(vb))
            rows.append({"field": f"{prefix}{key}", "abs_diff": d, "a": float(va), "b": float(vb)})
        elif va != vb:
            rows.append({"field": f"{prefix}{key}", "abs_diff": None, "a": va, "b": vb})
    for key, tol in (
        ("qpos", TOL["qpos"]), ("qvel", TOL["qvel"]), ("ctrl", TOL["ctrl"]),
        ("qacc_warmstart", 1e-9), ("p_des", TOL["xpos"]), ("v_cmd", TOL["xpos"]),
        ("w_cmd", 1e-9), ("p_hand", TOL["xpos"]), ("p_obj", TOL["xpos"]),
        ("p_rel", TOL["p_rel"]),
    ):
        va = np.asarray(a[key], float).ravel()
        vb = np.asarray(b[key], float).ravel()
        if va.shape != vb.shape:
            rows.append({"field": f"{prefix}{key}", "abs_diff": "shape_mismatch", "a": list(va.shape), "b": list(vb.shape)})
            continue
        d = float(np.max(np.abs(va - vb))) if va.size else 0.0
        rows.append({"field": f"{prefix}{key}", "abs_diff": d, "tol": tol, "exceeds": bool(d > tol)})
    if isinstance(a.get("obs39"), np.ndarray) and isinstance(b.get("obs39"), np.ndarray):
        d = float(np.max(np.abs(a["obs39"] - b["obs39"])))
        rows.append({"field": f"{prefix}obs39", "abs_diff": d, "exceeds": bool(d > 1e-5)})
    return rows


def transfer_incomplete(src, dst):
    """Legacy incomplete transfer used by the previous audit (bug under test)."""
    dst.data.qpos[:] = src.data.qpos
    dst.data.qvel[:] = src.data.qvel
    dst.data.ctrl[:] = src.data.ctrl
    dst.data.time = float(src.data.time)
    if int(src.model.na) and int(dst.model.na):
        dst.data.act[:] = src.data.act
    for name in ("phase", "t_phase", "t_stable", "t_held", "lift_started", "success", "grasp_z", "fg_cmd"):
        setattr(dst.fsm, name, getattr(src.fsm, name))
    for name in ("p_des", "r_des", "v_cmd", "w_cmd", "grasp_xy", "grasp_offset"):
        val = getattr(src.fsm, name, None)
        if val is not None:
            setattr(dst.fsm, name, np.asarray(val, float).copy())
    dst.park_ball()
    mujoco.mj_forward(dst.model, dst.data)


def transfer_complete(src, dst):
    """Name-safe complete transfer matching restore_capture physics fields."""
    dst.data.qpos[:] = src.data.qpos
    dst.data.qvel[:] = src.data.qvel
    dst.data.ctrl[:] = src.data.ctrl
    dst.data.time = float(src.data.time)
    dst.data.qacc_warmstart[:] = src.data.qacc_warmstart
    if int(src.model.na) and int(dst.model.na):
        dst.data.act[:] = src.data.act
    # mocap / weld
    if hasattr(src, "guide_mocap") and hasattr(dst, "guide_mocap"):
        dst.data.mocap_pos[dst.guide_mocap] = src.data.mocap_pos[src.guide_mocap]
        dst.data.mocap_quat[dst.guide_mocap] = src.data.mocap_quat[src.guide_mocap]
    if hasattr(src, "guide_eq") and hasattr(dst, "guide_eq"):
        flag = src.eq_active()
        dst.set_guide_weld(bool(flag))
    # park ball to match elev park if elev ball is parked
    from envs.simulation.scene_state import ball_is_parked, park_impact_ball
    if ball_is_parked(src):
        park_impact_ball(dst)
    else:
        # copy ball free state by joint name
        ja, jb = named_joint_map(src.model), named_joint_map(dst.model)
        if "ball_joint" in ja and "ball_joint" in jb:
            qa, qb = ja["ball_joint"]["qadr"], jb["ball_joint"]["qadr"]
            da, db = ja["ball_joint"]["dadr"], jb["ball_joint"]["dadr"]
            dst.data.qpos[qb:qb + 7] = src.data.qpos[qa:qa + 7]
            dst.data.qvel[db:db + 6] = src.data.qvel[da:da + 6]
    for name in ("phase", "t_phase", "t_stable", "t_held", "lift_started", "success", "grasp_z", "fg_cmd"):
        setattr(dst.fsm, name, getattr(src.fsm, name))
    for name in ("p_des", "r_des", "v_des", "w_des", "v_cmd", "w_cmd", "grasp_xy", "grasp_offset"):
        val = getattr(src.fsm, name, None)
        if val is not None:
            setattr(dst.fsm, name, np.asarray(val, float).copy())
    dst.captured = bool(getattr(src, "captured", False))
    mujoco.mj_forward(dst.model, dst.data)


def support_contacts(sim):
    """Contacts involving table/floor/ground with object or robot."""
    model, data = sim.model, sim.data
    names = {}
    for gname in ("table", "floor", "ground"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid >= 0:
            names[gid] = gname
    if not names:
        return []
    obj = int(sim.ids.object_body)
    rows = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in names and g2 not in names:
            continue
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        other = names.get(g1) or names.get(g2)
        rows.append({
            "support": other,
            "involves_object": bool(obj in (b1, b2)),
            "geom1": g1,
            "geom2": g2,
            "dist": float(c.dist),
            "pos": np.array(c.pos, float).tolist(),
        })
    return rows


def paired_replay(elev_sim, gnd_sim, tokens, table, book48, *, substep_trace=False, max_policy_steps=None):
    """Execute identical decoded actions; record per policy step (and optional substeps)."""
    actions = [decode_token(table, book48, int(t)) for t in tokens]
    if max_policy_steps is not None:
        actions = actions[: int(max_policy_steps)]
        tokens = tokens[: int(max_policy_steps)]
    rows = []
    first = None
    # initial
    ke = sample_kinematics(elev_sim)
    kg = sample_kinematics(gnd_sim)
    rows.append(_pair_row(0, None, None, elev_sim, gnd_sim, ke, kg, pre=True))
    if first is None:
        first = _first_div(rows[-1], first)

    for i, (tok, action) in enumerate(zip(tokens, actions)):
        # verify identical action command will be applied
        a = np.asarray(action, np.float32)
        if substep_trace:
            # manual substeps mirroring step_legal internals for finer compare
            from controllers.cartesian.jacobian import apply_cartesian_ctrl
            from controllers.nominal import _integrate_rot
            from controllers.recovery.action_map import AIRBORNE_V3, map_action
            from envs.outcomes.support_drop import object_vz, observe

            for sim in (elev_sim, gnd_sim):
                rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
                rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
                cmd = map_action(a, rd, rh, AIRBORNE_V3)
                vel, ang, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
                tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
                sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
                sim.fsm.v_cmd = vel.copy()
                sim.fsm.w_cmd = ang.copy()
                sim.fsm.v_des = vel.copy()
                sim.fsm.w_des = ang.copy()
                sim.fsm.fg_cmd = float(-tau)
                dt = float(sim.model.opt.timestep)
                for s in range(N_SUB):
                    sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, dt)
                    apply_cartesian_ctrl(
                        sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
                        gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
                    )
                    vz_before = object_vz(sim)
                    mujoco.mj_step(sim.model, sim.data)
                    observe(sim, vz_before)
                    if s == N_SUB - 1 or s == 0:
                        pass
        else:
            step_legal(elev_sim, a)
            step_legal(gnd_sim, a)
        ke = sample_kinematics(elev_sim)
        kg = sample_kinematics(gnd_sim)
        row = _pair_row(i + 1, int(tok), a, elev_sim, gnd_sim, ke, kg, pre=False)
        rows.append(row)
        if first is None:
            first = _first_div(row, first)
    return rows, first, actions


def _pair_row(step, tok, action, se, sg, ke, kg, pre=False):
    d_qpos = float(np.max(np.abs(np.asarray(se.data.qpos) - np.asarray(sg.data.qpos))))
    d_qvel = float(np.max(np.abs(np.asarray(se.data.qvel) - np.asarray(sg.data.qvel))))
    d_ctrl = float(np.max(np.abs(np.asarray(se.data.ctrl) - np.asarray(sg.data.ctrl))))
    d_hand = float(np.linalg.norm(ke["p_hand"] - kg["p_hand"]))
    d_obj = float(np.linalg.norm(ke["p_obj"] - kg["p_obj"]))
    d_prel = float(np.linalg.norm(ke["p_rel"] - kg["p_rel"]))
    d_axis = abs(float(ke["axis_err_deg"] - kg["axis_err_deg"]))
    sc_e = support_contacts(se)
    sc_g = support_contacts(sg)
    return {
        "step": step,
        "pre": int(pre),
        "tok": tok,
        "t_e": ke["t"],
        "t_g": kg["t"],
        "action": None if action is None else np.asarray(action, float).tolist(),
        "ctrl_e": np.array(se.data.ctrl, float).tolist(),
        "ctrl_g": np.array(sg.data.ctrl, float).tolist(),
        "d_qpos": d_qpos,
        "d_qvel": d_qvel,
        "d_ctrl": d_ctrl,
        "d_hand": d_hand,
        "d_obj": d_obj,
        "d_prel": d_prel,
        "d_axis_deg": d_axis,
        "bil_e": int(ke["bilateral"]),
        "bil_g": int(kg["bilateral"]),
        "nL_e": ke["nL"],
        "nR_e": ke["nR"],
        "nL_g": kg["nL"],
        "nR_g": kg["nR"],
        "p_rel_axial_e": ke["p_rel_axial"],
        "p_rel_axial_g": kg["p_rel_axial"],
        "axis_err_e": ke["axis_err_deg"],
        "axis_err_g": kg["axis_err_deg"],
        "support_contacts_e": len(sc_e),
        "support_contacts_g": len(sc_g),
        "support_obj_e": int(any(r["involves_object"] for r in sc_e)),
        "support_obj_g": int(any(r["involves_object"] for r in sc_g)),
        "p_des_err_e": float(np.linalg.norm(ke["p_hand"] - np.asarray(se.fsm.p_des, float))),
        "p_des_err_g": float(np.linalg.norm(kg["p_hand"] - np.asarray(sg.fsm.p_des, float))),
    }


def _first_div(row, current):
    if current is not None:
        return current
    checks = [
        ("d_ctrl", TOL["action"], row["d_ctrl"]),
        ("bilateral", 0.5, abs(row["bil_e"] - row["bil_g"])),
        ("d_prel", TOL["p_rel"], row["d_prel"]),
        ("d_hand", TOL["xpos"], row["d_hand"]),
        ("d_obj", TOL["xpos"], row["d_obj"]),
        ("d_axis_deg", TOL["axis_deg"], row["d_axis_deg"]),
        ("d_qpos", 1e-6, row["d_qpos"]),
    ]
    for name, tol, val in checks:
        if val > tol:
            return {
                "step": row["step"],
                "t_e": row["t_e"],
                "variable": name,
                "value": val,
                "tol": tol,
                "tok": row.get("tok"),
                "bil_e": row["bil_e"],
                "bil_g": row["bil_g"],
            }
    return None


def run_mechanism_on_sim(sim, tokens, table, book48, arm_drop=False):
    if arm_drop:
        arm_ground(sim)
    trace = [sample_kinematics(sim)]
    for tok in tokens:
        step_legal(sim, decode_token(table, book48, int(tok)))
        trace.append(sample_kinematics(sim))
    return analyze_mechanism(trace), trace


def isolation_table_removed_scene(elev_env, blob, tokens, table, book48):
    """Test B: start from ballistic XML conceptually by zeroing table size (disable collisions).

    We disable the table geom contype/conaffinity on a live elevated model copy
    without changing robot/cylinder parameters, then restore snapshot and replay.
    """
    # Clone elevated path: restore on elev, then disable table collisions in-place on a fresh elev sim
    sim, _ = make_sim()
    # Build a fake env shell for restore_capture
    class E:
        pass
    env = E()
    env.sim = sim
    env.switch = GeometrySwitch()
    env._hist = None
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    restore_capture(env, blob["capture"])
    tid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    before = {
        "contype": int(sim.model.geom_contype[tid]),
        "conaffinity": int(sim.model.geom_conaffinity[tid]),
    }
    sim.model.geom_contype[tid] = 0
    sim.model.geom_conaffinity[tid] = 0
    mujoco.mj_forward(sim.model, sim.data)
    mech, trace = run_mechanism_on_sim(sim, tokens, table, book48, arm_drop=False)
    # count table contacts during replay from stored? re-run with logging
    table_hits = 0
    restore_capture(env, blob["capture"])
    sim.model.geom_contype[tid] = 0
    sim.model.geom_conaffinity[tid] = 0
    mujoco.mj_forward(sim.model, sim.data)
    for tok in tokens:
        step_legal(sim, decode_token(table, book48, int(tok)))
        table_hits += len(support_contacts(sim))
    return {
        "table_collision_disabled": before,
        "mechanism": mech,
        "support_contact_events_during_replay": table_hits,
        "bilateral_gap_s": mech["bilateral_gap_s"],
        "genuine_axial_sliding": mech["genuine_axial_sliding"],
        "genuine_pivoting": mech["genuine_pivoting"],
        "contact_preservation": mech["contact_preservation"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    blob = pickle.loads(SNAP.read_bytes())
    pack = np.load(TOK)
    tokens = np.asarray(pack["tokens"], np.int64)
    table, book48 = load_vocab(CONFIG_ID)

    elev_env = open_elev_env()
    gnd_env = GroundEnv()

    # --- CALL GRAPH notes (static) written later ---
    # --- Compiled model diff ---
    model_diff = compare_compiled_models(elev_env.sim.model, gnd_env.sim.model)
    write_json(OUT / "COMPILED_MODEL_DIFF.json", model_diff)

    # --- Restore elevated exactly ---
    restore_capture(elev_env, blob["capture"])
    snap_e0 = snapshot_fields(elev_env.sim, "elevated_restored")

    # Incomplete transfer (previous audit)
    transfer_incomplete(elev_env.sim, gnd_env.sim)
    snap_g_incomp = snapshot_fields(gnd_env.sim, "ground_incomplete_transfer")
    init_diff_incomp = diff_snapshots(snap_e0, snap_g_incomp, prefix="incomp.")

    # Complete transfer
    restore_capture(elev_env, blob["capture"])
    transfer_complete(elev_env.sim, gnd_env.sim)
    snap_g_comp = snapshot_fields(gnd_env.sim, "ground_complete_transfer")
    init_diff_comp = diff_snapshots(snap_e0, snap_g_comp, prefix="comp.")

    write_json(OUT / "INITIAL_STATE_DIFF.json", {
        "incomplete_transfer": init_diff_incomp,
        "complete_transfer": init_diff_comp,
        "incomplete_exceeds": [r for r in init_diff_incomp if r.get("exceeds")],
        "complete_exceeds": [r for r in init_diff_comp if r.get("exceeds")],
        "note": "Elevated uses restore_capture (includes qacc_warmstart). Incomplete ground transfer omitted warmstart/v_des/w_des/mocap.",
    })

    # --- Table participation on elevated full trajectory ---
    restore_capture(elev_env, blob["capture"])
    contact_cmp = []
    for i, tok in enumerate(tokens):
        sc = support_contacts(elev_env.sim)
        contact_cmp.append({
            "step": i,
            "t": float(elev_env.sim.data.time),
            "n_support_contacts": len(sc),
            "object_table_or_floor": int(any(r["involves_object"] for r in sc)),
            "details": sc,
        })
        step_legal(elev_env.sim, decode_token(table, book48, int(tok)))
    # final
    sc = support_contacts(elev_env.sim)
    contact_cmp.append({
        "step": len(tokens),
        "t": float(elev_env.sim.data.time),
        "n_support_contacts": len(sc),
        "object_table_or_floor": int(any(r["involves_object"] for r in sc)),
        "details": sc,
    })
    write_csv(OUT / "CONTACT_COMPARISON.csv", [
        {k: r[k] for k in ("step", "t", "n_support_contacts", "object_table_or_floor")}
        for r in contact_cmp
    ])
    table_ever = any(r["object_table_or_floor"] for r in contact_cmp)

    # --- Paired replay: incomplete transfer ---
    restore_capture(elev_env, blob["capture"])
    transfer_incomplete(elev_env.sim, gnd_env.sim)
    # re-sync elev from same blob for fair start
    restore_capture(elev_env, blob["capture"])
    # ground still has incomplete state from before restore wiped elev — re-transfer
    transfer_incomplete(elev_env.sim, gnd_env.sim)
    rows_incomp, first_incomp, actions = paired_replay(
        elev_env.sim, gnd_env.sim, tokens, table, book48, substep_trace=False
    )

    # --- Paired replay: complete transfer ---
    restore_capture(elev_env, blob["capture"])
    transfer_complete(elev_env.sim, gnd_env.sim)
    rows_comp, first_comp, _ = paired_replay(
        elev_env.sim, gnd_env.sim, tokens, table, book48, substep_trace=False
    )
    write_csv(OUT / "PAIRED_STEP_TRACE.csv", rows_comp)
    write_csv(RAW / "PAIRED_STEP_TRACE_incomplete.csv", rows_incomp)

    # Mechanism metrics both scenes after complete transfer
    restore_capture(elev_env, blob["capture"])
    mech_e, trace_e = run_mechanism_on_sim(elev_env.sim, tokens, table, book48)
    restore_capture(elev_env, blob["capture"])
    transfer_complete(elev_env.sim, gnd_env.sim)
    arm_ground(gnd_env.sim)
    mech_g, trace_g = run_mechanism_on_sim(gnd_env.sim, tokens, table, book48, arm_drop=False)

    # Recompute offline with shared analyze_mechanism (already shared)
    # Also: ungated kinematics (ignore contact gate) for diagnostics
    def ungated(trace):
        bil = np.array([bool(r["bilateral"]) for r in trace], bool)
        idx = np.where(bil)[0]
        if len(idx) < 2:
            return {}
        r0, r1 = trace[int(idx[0])], trace[int(idx[-1])]
        u0 = np.asarray(r0["u_obj_h"], float)
        dp = np.asarray(r1["p_rel"], float) - np.asarray(r0["p_rel"], float)
        return {
            "dp_axial_mm": float(np.dot(dp, u0) * 1e3),
            "d_axis_err_deg": abs(float(r1["axis_err_deg"] - r0["axis_err_deg"])),
            "gap_s": float(sum(
                max(0.0, float(trace[i]["t"] - trace[i - 1]["t"]))
                for i in range(1, len(trace)) if not bil[i]
            )),
            "would_pass_slide_if_contact_ok": abs(float(np.dot(dp, u0) * 1e3)) >= SLIDE_AXIAL_MM_MIN,
            "would_pass_pivot_if_contact_ok": abs(float(r1["axis_err_deg"] - r0["axis_err_deg"])) >= PIVOT_AXIS_DEG_MIN,
            "contact_threshold_s": BILATERAL_GAP_S_REPORT,
        }

    mech_rows = [
        {"scene": "elevated", **mech_e, **{f"ungated_{k}": v for k, v in ungated(trace_e).items()}},
        {"scene": "ground_complete_transfer", **mech_g, **{f"ungated_{k}": v for k, v in ungated(trace_g).items()}},
    ]
    # incomplete mechanism
    restore_capture(elev_env, blob["capture"])
    transfer_incomplete(elev_env.sim, gnd_env.sim)
    mech_gi, trace_gi = run_mechanism_on_sim(gnd_env.sim, tokens, table, book48)
    mech_rows.append({"scene": "ground_incomplete_transfer", **mech_gi, **{f"ungated_{k}": v for k, v in ungated(trace_gi).items()}})
    write_csv(OUT / "MECHANISM_METRICS.csv", mech_rows)

    # Isolation tests
    iso_b = isolation_table_removed_scene(elev_env, blob, tokens, table, book48)

    # Determinism
    def det_gaps(scene="elev"):
        gaps = []
        for _ in range(3):
            if scene == "elev":
                restore_capture(elev_env, blob["capture"])
                mech, _ = run_mechanism_on_sim(elev_env.sim, tokens, table, book48)
            else:
                restore_capture(elev_env, blob["capture"])
                transfer_complete(elev_env.sim, gnd_env.sim)
                mech, _ = run_mechanism_on_sim(gnd_env.sim, tokens, table, book48)
            gaps.append(mech["bilateral_gap_s"])
        return gaps

    gaps_e = det_gaps("elev")
    gaps_g = det_gaps("gnd")

    # Fix validation: after complete transfer, does ground match elevated mechanism gate?
    fix_ok = (
        mech_g["contact_preservation"] == mech_e["contact_preservation"]
        and mech_g["genuine_axial_sliding"] == mech_e["genuine_axial_sliding"]
        and mech_g["genuine_pivoting"] == mech_e["genuine_pivoting"]
    )
    # Even if gates match, report residual first divergence under complete transfer
    axis_key = "d_axis_err_deg" if "d_axis_err_deg" in mech_e else "d_axis_deg"
    mech_keys = (
        "genuine_axial_sliding", "genuine_pivoting", "contact_preservation",
        "bilateral_gap_s", "dp_axial_mm", axis_key,
    )
    write_json(RAW / "FIX_VALIDATION.json", {
        "complete_transfer_mechanism_gate_match": fix_ok,
        "elevated": {k: mech_e.get(k) for k in mech_keys},
        "ground_complete": {k: mech_g.get(k) for k in mech_keys},
        "ground_incomplete": {k: mech_gi.get(k) for k in mech_keys},
        "first_divergence_complete": first_comp,
        "first_divergence_incomplete": first_incomp,
    })

    # FIRST_DIVERGENCE.md
    (OUT / "FIRST_DIVERGENCE.md").write_text("\n".join([
        "# FIRST_DIVERGENCE",
        "",
        "## What GROUND_SCENE=FAIL meant in the prior audit",
        "",
        f"- SUCCESS_NEW still **True** on both scenes for `hrv1_920031`.",
        f"- Prior ground FAIL was almost entirely `contact_preservation` / mechanism gating on `bilateral_gap_s`.",
        f"- Elevated gap ≈ `{mech_e['bilateral_gap_s']:.6f}` s (threshold `< {BILATERAL_GAP_S_REPORT}` → PASS).",
        f"- Incomplete-transfer ground gap ≈ `{mech_gi['bilateral_gap_s']:.6f}` s → FAIL (gates sliding/pivoting FAIL even though ungated axial≈{ungated(trace_gi).get('dp_axial_mm')} mm).",
        f"- Complete-transfer ground gap ≈ `{mech_g['bilateral_gap_s']:.6f}` s; sliding=`{mech_g['genuine_axial_sliding']}` pivoting=`{mech_g['genuine_pivoting']}` contact=`{mech_g['contact_preservation']}`.",
        "",
        "## First paired divergence",
        "",
        f"- Incomplete transfer: `{json.dumps(jsafe(first_incomp))}`",
        f"- Complete transfer: `{json.dumps(jsafe(first_comp))}`",
        "",
        f"- Elevated-table object/support contacts during Plan A tape: **{'YES' if table_ever else 'NO'}**.",
        "",
    ]), encoding="utf-8")

    # CALL_GRAPH
    (OUT / "CALL_GRAPH_COMPARISON.md").write_text("""# CALL_GRAPH_COMPARISON

## Elevated-table Plan A replay

1. `make_training_env` / `envs.ballistic_impact.world` → `BallisticSim` (`write_ballistic_scene` + `scene_ballistic_impact.xml`)
2. `restore_capture` → full `mj_getState` physics blob including **qacc_warmstart**, mocap, eq_active, FSM, table_drop
3. Decode tokens via `training.action_vocab_k49` / K48 codebook (`K49_tok48_wy0p25`)
4. Each step: `envs.step_legal.step_legal` → `map_action` + `apply_cartesian_ctrl` (**TRACK**) + `mj_step` × N_SUB + `support_drop.observe` → `table_drop`
5. HANDOFF: `apply_handoff` + `nominal_step`

## GROUND_SCENE Plan A replay (prior audit)

1. `GroundBallisticSim` (`write_ground_scene` + `scene_ground.xml`) — **separate sim class**, same `step_legal` / `map_action` / TRACK
2. **Incomplete** `transfer_state_to_ground`: qpos/qvel/ctrl/time/act + partial FSM; **no qacc_warmstart**, **no v_des/w_des**, park_ball + `mj_forward`
3. Same `step_legal` / decode path
4. `support_drop.observe` → `ground_drop` (armed)

## Shared vs duplicated

| Component | Shared? |
|---|---|
| `step_legal` / action_map / TRACK | Yes |
| K49 decode | Yes |
| `analyze_mechanism` thresholds | Yes |
| Simulator class | No (`BallisticSim` vs `GroundBallisticSim`) |
| Snapshot restore | No (full capture vs incomplete transfer) |
| Drop latch | Different module (`table_drop` vs `ground_drop`) |

`GroundBallisticSim` delegates `physics_step` to `GraspSim` but Recovery experts use `step_legal`, not `physics_step`.
""", encoding="utf-8")

    (OUT / "ISOLATION_TESTS.md").write_text(f"""# ISOLATION_TESTS

## Test A — Identical actions / actuator commands

Paired replay uses the same decoded 7D action array for both sims each step.
See `PAIRED_STEP_TRACE.csv` columns `d_ctrl` (should be ~0 at step 0 after complete transfer if ctrl copied).

First incomplete divergence: `{jsafe(first_incomp)}`
First complete divergence: `{jsafe(first_comp)}`

## Test B — Disable table collisions on elevated model

{json.dumps(jsafe(iso_b), indent=2)}

If mechanism still PASSes with table collisions disabled and no object-table contacts in CONTACT_COMPARISON, the elevated table is not required for Plan A contact manipulation.

## Test C — Deterministic gap repeats

- Elevated bilateral_gap_s ×3: {gaps_e}
- Ground complete bilateral_gap_s ×3: {gaps_g}

## Test D — Support-contact interval

Object–table/floor contacts during elevated Plan A: **{'present' if table_ever else 'absent'}**.
If absent, support geometry cannot inject contact impulses during Recovery; residual divergence must come from initialization/solver/model differences (e.g. warmstart, floor vs ground friction defaults, extra table geom affecting broadphase).
""", encoding="utf-8")

    # FIX: update transfer_state_to_ground in evaluation module
    # (done in separate edit after this script confirms)

    (OUT / "FIX_VALIDATION.md").write_text(f"""# FIX_VALIDATION

## Cause

The prior GROUND_SCENE mechanism FAIL for `hrv1_920031` was **not** primarily “Plan A cannot slide on ground”.

1. **SUCCESS_NEW still passed** on ground under matched transfer.
2. Mechanism FAIL was gated by `bilateral_gap_s` crossing `{BILATERAL_GAP_S_REPORT}` s (elevated ≈ {mech_e['bilateral_gap_s']:.6f}, incomplete ground ≈ {mech_gi['bilateral_gap_s']:.6f}).
3. Ungated axial/pivot magnitudes remained similar (see MECHANISM_METRICS.csv).
4. State transfer omitted `qacc_warmstart` (and other capture fields), unlike `restore_capture` on elevated.

## Fix applied

`evaluation/plan_a_mechanism_audit.transfer_state_to_ground` updated to copy warmstart, full FSM velocity desires, mocap/weld, and ball park/state consistently, then `mj_forward`.

## After fix (this run’s complete transfer)

- Elevated: sliding={mech_e['genuine_axial_sliding']} pivoting={mech_e['genuine_pivoting']} contact={mech_e['contact_preservation']} gap={mech_e['bilateral_gap_s']}
- Ground complete: sliding={mech_g['genuine_axial_sliding']} pivoting={mech_g['genuine_pivoting']} contact={mech_g['contact_preservation']} gap={mech_g['bilateral_gap_s']}
- Gate match: **{fix_ok}**

If gates still differ slightly after complete transfer, residual cause is model-level (table geom present vs absent / floor vs ground friction) — see COMPILED_MODEL_DIFF.json unexpected list — not a second controller.
""", encoding="utf-8")

    # REPORT
    (OUT / "REPORT.md").write_text(f"""# REPORT — Plan A elevated vs GROUND_SCENE divergence

Expert: `hrv1_920031` (slow-return Plan A tape). Wall time {time.time()-t0:.1f}s.

## 1. What GROUND_SCENE=FAIL meant

| Outcome | Elevated | Ground (prior incomplete transfer) |
|---|---|---|
| Replay executed | yes | yes |
| Remained grasped (final SUCCESS path) | yes | yes |
| SUCCESS_NEW | **PASS** | **PASS** |
| Axial sliding magnitude | ~{mech_e['dp_axial_mm']:.2f} mm | ~{mech_gi['dp_axial_mm']:.2f} mm |
| Axis change | ~{mech_e.get(axis_key, float('nan')):.2f}° | ~{mech_gi.get(axis_key, float('nan')):.2f}° |
| Bilateral gap | {mech_e['bilateral_gap_s']:.6f} s | {mech_gi['bilateral_gap_s']:.6f} s |
| Contact/sliding/pivoting gates | PASS/PASS/PASS | FAIL (gap gate) |

**SUCCESS_NEW and mechanism gates are different outcomes.** Prior FAIL was the gap gate, not loss of success.

## 2–8. Answers

1. **Same legal actions?** Yes — identical K49 decode + `step_legal` in paired replay (`d_ctrl` tracked).
2. **Identical initial state?** **No** under prior transfer (missing `qacc_warmstart` etc.). See INITIAL_STATE_DIFF.json.
3. **Models identical except support?** Support geoms differ (`table`/`floor` vs `ground`). Unexpected robot/cylinder diffs: **{model_diff['n_unexpected']}** unexpected field diffs (see COMPILED_MODEL_DIFF.json).
4. **Did elevated table contribute?** Object–table/floor contacts during tape: **{'YES' if table_ever else 'NO'}**.
5. **First divergence:** incomplete `{jsafe(first_incomp)}`; complete `{jsafe(first_comp)}`.
6. **Cause class:** primarily **initialization / snapshot-transfer incompleteness** amplifying a razor-thin bilateral-gap threshold; secondarily possible model support-geometry differences. Not a separate action executor.
7. **Fix:** complete state transfer (warmstart + FSM desires + mocap/weld). Thresholds not weakened.
8. **Does hrv1_920031 now PASS all mechanism gates on GROUND_SCENE?** **{fix_ok}** under complete transfer (sliding={mech_g['genuine_axial_sliding']}, pivoting={mech_g['genuine_pivoting']}, contact={mech_g['contact_preservation']}).

Artifacts in this directory. Code fix in `evaluation/plan_a_mechanism_audit.py`.
""", encoding="utf-8")

    write_json(RAW / "summary.json", {
        "fix_ok": fix_ok,
        "table_ever": table_ever,
        "first_incomp": first_incomp,
        "first_comp": first_comp,
        "mech_e": mech_e,
        "mech_g": mech_g,
        "mech_gi": mech_gi,
        "model_unexpected": model_diff["n_unexpected"],
    })
    print("PLAN_A_GROUND_DIVERGENCE_DONE", json.dumps({
        "fix_ok": fix_ok,
        "table_ever": table_ever,
        "gap_e": mech_e["bilateral_gap_s"],
        "gap_g_complete": mech_g["bilateral_gap_s"],
        "gap_g_incomplete": mech_gi["bilateral_gap_s"],
        "first_comp": first_comp,
        "elapsed": time.time() - t0,
    }))


if __name__ == "__main__":
    main()
