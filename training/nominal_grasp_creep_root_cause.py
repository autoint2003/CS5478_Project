"""Root-cause audit: why an undisturbed nominal pinch creeps in +hand-z.

Diagnostic copies of the runtime MjModel only. Official XML is not written.
No recovery, disturbance, RULE, or SAC.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg, ori_error_deg
from controllers.nominal import grasp_orientation
from envs.config_util import load_yaml, merge_sim_config
from envs.dynamics import set_finger_object_sliding_mu
from envs.physical_recovery import physical_pack
from training.demo_teleport_recovery_state import (
    PAIR_MU,
    advance_to_parent,
    make_parent_sim,
)
from training.replay_core import freeze, tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts

OUT = ROOT / "results" / "diagnostics" / "nominal_grasp_creep_root_cause"
RAW = OUT / "raw"
FIG = OUT / "figures"
HOLD_S = 10.0
TAU = -18.0
LOG_DT = 0.010


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")


def restore_snap(sim, snap: dict) -> None:
    sim.load_snapshot(snap)
    set_finger_object_sliding_mu(sim.model, sim.ids, PAIR_MU, data=None)
    mujoco.mj_forward(sim.model, sim.data)


def pack_snap(sim) -> dict:
    return sim.snapshot()


def robust_slope(t, y, t0=1.0, t1=10.0):
    m = (t >= t0) & (t <= t1)
    if int(m.sum()) < 10:
        return float("nan")
    tt, yy = t[m], y[m]
    n = len(tt)
    step = max(n // 400, 1)
    idx = np.arange(0, n, step)
    sl = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            dt = tt[idx[j]] - tt[idx[i]]
            if abs(dt) < 1e-9:
                continue
            sl.append((yy[idx[j]] - yy[idx[i]]) / dt)
    return float(np.median(sl)) if sl else float("nan")


def actuator_semantics(sim) -> dict:
    m, d, ids = sim.model, sim.data, sim.ids
    a = 7
    trn = int(m.actuator_trntype[a])
    names = {
        0: "JOINT",
        1: "JOINTINPARENT",
        2: "SLIDERCRANK",
        3: "TENDON",
        4: "SITE",
        5: "BODY",
    }
    gain = int(m.actuator_gaintype[a])
    bias = int(m.actuator_biastype[a])
    dynt = int(m.actuator_dyntype[a])
    ten = int(m.actuator_trnid[a, 0]) if trn == 3 else -1
    tendon = None
    if ten >= 0:
        w = []
        adr = int(m.tendon_adr[ten]) if hasattr(m, "tendon_adr") else 0
        num = int(m.tendon_num[ten]) if hasattr(m, "tendon_num") else 0
        # wrap numbers for fixed tendon
        if hasattr(m, "wrap_objid"):
            for k in range(num):
                w.append(
                    {
                        "type": int(m.wrap_type[adr + k]) if hasattr(m, "wrap_type") else None,
                        "objid": int(m.wrap_objid[adr + k]),
                        "coef": float(m.wrap_prm[adr + k]) if hasattr(m, "wrap_prm") else None,
                    }
                )
        tendon = {
            "id": ten,
            "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, ten),
            "length": float(d.ten_length[ten]) if m.ntendon else None,
            "velocity": float(d.ten_velocity[ten]) if m.ntendon else None,
            "wraps": w,
        }
    fj = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in ids.finger_jnt]
    return {
        "actuator_index": 7,
        "actuator_name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a),
        "trntype": trn,
        "trntype_name": names.get(trn, str(trn)),
        "gaintype": gain,
        "gaintype_name": "FIXED" if gain == 0 else str(gain),
        "biastype": bias,
        "biastype_name": "NONE" if bias == 0 else str(bias),
        "dyntype": dynt,
        "dyntype_name": "NONE" if dynt == 0 else str(dynt),
        "gear": np.array(m.actuator_gear[a], float).tolist(),
        "gainprm": np.array(m.actuator_gainprm[a], float)[:3].tolist(),
        "biasprm": np.array(m.actuator_biasprm[a], float)[:3].tolist(),
        "ctrlrange": np.array(m.actuator_ctrlrange[a], float).tolist(),
        "forcerange": np.array(m.actuator_forcerange[a], float).tolist(),
        "ctrl7": float(d.ctrl[7]),
        "actuator_force7": float(d.actuator_force[7]) if d.actuator_force.size > 7 else None,
        "meaning": (
            "motor on tendon 'split', gaintype=fixed, biastype=none: "
            "tendon force = gear[0]*ctrl[7]. "
            "fixed tendon maps coef*F onto finger_joint1 and finger_joint2 (slide). "
            "ctrl[7]=-18 with gear=1 => tendon force -18 N; coef=0.5 => about -9 N per finger joint."
        ),
        "tendon": tendon,
        "finger_joint_names": fj,
        "finger_qpos": np.array(d.qpos[ids.finger_jnt], float).tolist(),
        "finger_qvel": np.array(d.qvel[ids.finger_dof], float).tolist(),
        "finger_qfrc_actuator": np.array(d.qfrc_actuator[ids.finger_dof], float).tolist(),
        "xml_actuator": "assets/panda_torque.xml <motor name='actuator8' tendon='split' gear='1' ctrlrange='-50 50' forcerange='-50 50'/>",
        "xml_tendon": "assets/panda_torque.xml <fixed name='split'><joint finger_joint1 coef='0.5'/><joint finger_joint2 coef='0.5'/></fixed>",
        "xml_eq": "assets/panda_torque.xml equality joint finger_joint1~finger_joint2 (symmetry), not the grip motor",
    }


def live_contact_origin(sim) -> dict:
    m, ids = sim.model, sim.ids
    obj = int(ids.object_geom)
    pads = [
        g
        for g in range(m.ngeom)
        if int(m.geom_bodyid[g]) in (int(ids.left_body), int(ids.right_body))
        and int(m.geom_contype[g]) != 0
    ]
    cons, _ = extract_contacts(sim)
    return {
        "mujoco": mujoco.__version__,
        "integrator": int(m.opt.integrator),
        "integrator_name": "implicitfast" if int(m.opt.integrator) == 3 else str(int(m.opt.integrator)),
        "timestep": float(m.opt.timestep),
        "gravity": np.array(m.opt.gravity, float).tolist(),
        "iterations": int(m.opt.iterations),
        "tolerance": float(m.opt.tolerance),
        "ls_iterations": int(getattr(m.opt, "ls_iterations", 0) or 0),
        "solver": int(m.opt.solver),
        "cone": int(m.opt.cone),
        "impratio": float(m.opt.impratio),
        "noslip_iterations": int(getattr(m.opt, "noslip_iterations", 0) or 0),
        "object_geom": {
            "id": obj,
            "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, obj),
            "solref": np.array(m.geom_solref[obj], float).tolist(),
            "solimp": np.array(m.geom_solimp[obj], float).tolist(),
            "solmix": float(m.geom_solmix[obj]),
            "priority": int(m.geom_priority[obj]),
            "condim": int(m.geom_condim[obj]),
            "friction": np.array(m.geom_friction[obj], float).tolist(),
            "margin": float(m.geom_margin[obj]),
            "gap": float(m.geom_gap[obj]),
            "xml": 'assets/scene.xml geom object solref="0.01 1" condim="4" (solimp unspecified → MuJoCo default)',
        },
        "pad_geoms": [
            {
                "id": int(g),
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)),
                "solref": np.array(m.geom_solref[g], float).tolist(),
                "solimp": np.array(m.geom_solimp[g], float).tolist(),
                "solmix": float(m.geom_solmix[g]),
                "priority": int(m.geom_priority[g]),
                "condim": int(m.geom_condim[g]),
                "friction": np.array(m.geom_friction[g], float).tolist(),
                "margin": float(m.geom_margin[g]),
                "gap": float(m.geom_gap[g]),
            }
            for g in pads[:12]
        ],
        "pad_solref_xml": "assets/panda_torque.xml fingertip_pad_collision_* box geoms: no solref/solimp → MuJoCo default solref=(0.02,1)",
        "mixing_rule_code": (
            "Equal geom_priority and equal solmix mix solref/solimp as a weighted average. "
            "0.5*(0.01+0.02)=0.015 matches live contact.solref[0]."
        ),
        "n_live": len(cons),
        "live_solref": sorted({tuple(c["solref"]) for c in cons if c.get("solref")}),
        "live_solimp": sorted({tuple(np.round(c["solimp"], 6)) for c in cons if c.get("solimp")}),
        "live_mu": sorted({c["mu"] for c in cons}),
        "live_dim": sorted({c["dim"] for c in cons}),
        "live_margin_dist_sample": [c["dist"] for c in cons[:8]],
        "hand_z_world_column": np.array(sim.data.xmat[ids.hand_body].reshape(3, 3)[:, 2], float).tolist(),
        "grasp_orientation_hand_z": grasp_orientation()[:, 2].tolist(),
        "plus_hand_z_is_distal": "grasp_orientation() sets hand Z = world down; +r_h.z is toward fingertips",
    }


def apply_harder(sim) -> None:
    ids = sim.ids
    targets = [int(ids.object_geom)]
    targets += [
        g
        for g in range(sim.model.ngeom)
        if int(sim.model.geom_bodyid[g]) in (int(ids.left_body), int(ids.right_body))
    ]
    for g in targets:
        s = np.array(sim.model.geom_solref[g], float)
        s[0] = 0.005
        sim.model.geom_solref[g] = s
    mujoco.mj_forward(sim.model, sim.data)


def construct_ideal(sim) -> dict:
    """Privileged debug state: centered, aligned, zero velocity. Not training data."""
    ids = sim.ids
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    Rh = np.array(sim.data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[ids.hand_body], float)
    po = np.array(sim.data.xpos[ids.object_body], float)
    rh = Rh.T @ (po - ph)
    po_new = ph + Rh @ np.array([0.0, 0.0, rh[2]])
    adr = int(sim.model.jnt_qposadr[ids.object_jnt])
    quat_keep = np.array(sim.data.qpos[adr + 3 : adr + 7], float).copy()
    sim.data.qpos[adr : adr + 3] = po_new
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rh.reshape(9))
    sim.data.qpos[adr + 3 : adr + 7] = quat
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    freeze(sim)
    cons, mech = extract_contacts(sim)
    if mech["nL"] == 0 or mech["nR"] == 0:
        # keep natural object quaternion; only center in the hand xy plane
        sim.data.qpos[adr + 3 : adr + 7] = quat_keep
        sim.data.qvel[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        freeze(sim)
        cons, mech = extract_contacts(sim)
    return {
        "privileged": True,
        "nL": mech["nL"],
        "nR": mech["nR"],
        "rh": mech["rh"],
        "qvel_norm": float(np.linalg.norm(sim.data.qvel)),
        "n_contacts": len(cons),
    }


def overlay_nom(viewer, phase, t_hold, aperture, dez_mm, extra_title="NOMINAL STATIC PINCH"):
    if viewer is None or not hasattr(viewer, "add_overlay"):
        return
    try:
        pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
        viewer.add_overlay(pos, "TITLE", extra_title)
        viewer.add_overlay(pos, "PHASE", str(phase))
        viewer.add_overlay(pos, "HOLD t", f"{t_hold:.2f} s")
        viewer.add_overlay(pos, "APERTURE", f"{aperture:.5f} m")
        viewer.add_overlay(pos, "Delta r_h.z", f"{dez_mm:.2f} mm")
    except Exception:
        pass


def sample(sim, t_hold: float, ez0: float) -> dict:
    cons, mech = extract_contacts(sim)
    o = physical_pack(sim)
    ids = sim.ids
    Rh = np.array(sim.data.xmat[ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[ids.object_body].reshape(3, 3), float)
    ten_len = float(sim.data.ten_length[0]) if sim.model.ntendon else float("nan")
    ten_vel = float(sim.data.ten_velocity[0]) if sim.model.ntendon else float("nan")
    rhos = [c["rho"] for c in cons]
    fts = [c["Ft"] for c in cons]
    dists = [c["dist"] for c in cons]
    keys = []
    for c in cons:
        p = np.round(np.array(c["pos_w"], float), 4)
        keys.append((int(c["geom1"]), int(c["geom2"]), float(p[0]), float(p[1]), float(p[2])))
    return {
        "t_hold": t_hold,
        "t": float(sim.data.time),
        "ph": mech["ph"],
        "po": mech["po"],
        "Rh": Rh.reshape(9).tolist(),
        "Ro": Ro.reshape(9).tolist(),
        "vh": mech["vh"],
        "wh": mech["wh"],
        "vo": np.array(sim.data.cvel[ids.object_body][3:6], float).tolist(),
        "wo": np.array(sim.data.cvel[ids.object_body][:3], float).tolist(),
        "rh": mech["rh"],
        "e_z": float(mech["rh"][2]),
        "dez": float(mech["rh"][2] - ez0),
        "v_rel_h": mech["v_rel_h"],
        "w_rel_h": mech["w_rel_h"],
        "p_des": mech["p_des"],
        "pos_err": mech["pos_err"],
        "v_cmd": mech["v_cmd"],
        "ori_err_deg": float(ori_error_deg(np.asarray(sim.fsm.r_des).reshape(3, 3), Rh)),
        "aperture": float(o["aperture"]),
        "finger_qpos": np.array(sim.data.qpos[ids.finger_jnt], float).tolist(),
        "finger_qvel": np.array(sim.data.qvel[ids.finger_dof], float).tolist(),
        "finger_qfrc_act": np.array(sim.data.qfrc_actuator[ids.finger_dof], float).tolist(),
        "ten_length": ten_len,
        "ten_velocity": ten_vel,
        "ctrl7": float(sim.data.ctrl[7]),
        "act_force7": float(sim.data.actuator_force[7]),
        "nL": mech["nL"],
        "nR": mech["nR"],
        "Fn_L": float(sum(abs(c["Fn"]) for c in cons if c["side"] == "L")),
        "Fn_R": float(sum(abs(c["Fn"]) for c in cons if c["side"] == "R")),
        "Ft_sum": float(sum(fts)) if fts else 0.0,
        "rho_mean": float(np.nanmean(rhos)) if rhos else float("nan"),
        "rho_max": float(np.nanmax(rhos)) if rhos else float("nan"),
        "dist_mean": float(np.mean(dists)) if dists else float("nan"),
        "dist_min": float(np.min(dists)) if dists else float("nan"),
        "residual_norm": mech["residual_norm"],
        "F_contact_h": mech["F_contact_h"],
        "Fg_h": mech["Fg_h"],
        "a_lin_h": mech["a_lin_h"],
        "obj_z": float(o["obj_z"]),
        "hand_z": float(mech["ph"][2]),
        "contact_keys": keys,
        "n_contacts": len(cons),
        "mu_live": sorted({c["mu"] for c in cons}),
        "solref_live": [list(x) for x in sorted({tuple(c["solref"]) for c in cons if c.get("solref")})],
    }


def summarize(rows: list[dict], extra: dict | None = None) -> dict:
    t = np.array([r["t_hold"] for r in rows], float)
    ez = np.array([r["e_z"] for r in rows], float)
    hz = np.array([r["hand_z"] for r in rows], float)
    oz = np.array([r["obj_z"] for r in rows], float)
    ap = np.array([r["aperture"] for r in rows], float)
    vz = np.array([r["v_rel_h"][2] for r in rows], float)

    def d_at(ts):
        i = int(np.argmin(np.abs(t - ts)))
        return float(ez[i] - ez[0])

    m = (t >= 1.0) & (t <= 10.0)
    # churn
    lifetimes = defaultdict(int)
    births = deaths = 0
    prev = set()
    for r in rows:
        cur = set(tuple(k) if not isinstance(k, tuple) else k for k in r["contact_keys"])
        # keys may be lists after json; keep as tuples of numbers
        cur = set(tuple(x) for x in r["contact_keys"])
        if prev:
            births += len(cur - prev)
            deaths += len(prev - cur)
        for k in cur:
            lifetimes[k] += 1
        prev = cur
    nstep = max(len(rows) - 1, 1)
    dt_log = float(np.median(np.diff(t))) if len(t) > 1 else LOG_DT
    s = {
        "d_ez_1s_mm": 1e3 * d_at(1.0),
        "d_ez_2s_mm": 1e3 * d_at(2.0),
        "d_ez_5s_mm": 1e3 * d_at(5.0),
        "d_ez_10s_mm": 1e3 * d_at(10.0),
        "slope_ez_1_10_mm_s": 1e3 * robust_slope(t, ez),
        "median_vz_1_10": float(np.median(vz[m])) if m.any() else float("nan"),
        "d_hand_z_world_mm": 1e3 * float(hz[-1] - hz[0]),
        "d_obj_z_world_mm": 1e3 * float(oz[-1] - oz[0]),
        "d_obj_minus_hand_world_mm": 1e3 * float((oz[-1] - hz[-1]) - (oz[0] - hz[0])),
        "d_aperture_mm": 1e3 * float(ap[-1] - ap[0]),
        "median_aperture": float(np.median(ap)),
        "rho_mean_med": float(np.nanmedian([r["rho_mean"] for r in rows])),
        "rho_max_max": float(np.nanmax([r["rho_max"] for r in rows])),
        "Fn_L_med": float(np.median([r["Fn_L"] for r in rows])),
        "Fn_R_med": float(np.median([r["Fn_R"] for r in rows])),
        "residual_median": float(np.nanmedian([r["residual_norm"] for r in rows])),
        "dist_mean_med": float(np.nanmedian([r["dist_mean"] for r in rows])),
        "nL_range": [min(r["nL"] for r in rows), max(r["nL"] for r in rows)],
        "nR_range": [min(r["nR"] for r in rows), max(r["nR"] for r in rows)],
        "hand_pos_err_end_mm": [1e3 * x for x in rows[-1]["pos_err"]],
        "ori_err_end_deg": rows[-1]["ori_err_deg"],
        "ctrl7": rows[0]["ctrl7"],
        "act_force7_med": float(np.median([r["act_force7"] for r in rows])),
        "finger_qfrc_med": np.median([r["finger_qfrc_act"] for r in rows], axis=0).tolist(),
        "churn_births_per_s": births / max(t[-1] - t[0], 1e-6),
        "churn_deaths_per_s": deaths / max(t[-1] - t[0], 1e-6),
        "unique_contact_keys": len(lifetimes),
        "median_key_lifetime_s": float(np.median(list(lifetimes.values())) * dt_log) if lifetimes else 0.0,
        "mean_key_lifetime_s": float(np.mean(list(lifetimes.values())) * dt_log) if lifetimes else 0.0,
        "n_samples": len(rows),
        "solref_live": rows[len(rows) // 2]["solref_live"],
        "mu_live": rows[0]["mu_live"],
        "ez0_mm": 1e3 * float(ez[0]),
        "dropped_n_zero": any(r["nL"] == 0 and r["nR"] == 0 for r in rows),
    }
    if extra:
        s.update(extra)
    return s


def save_traj(tag: str, rows, summ):
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW / f"{tag}.npz",
        t_hold=np.array([r["t_hold"] for r in rows], float),
        e_z=np.array([r["e_z"] for r in rows], float),
        v_rel_z=np.array([r["v_rel_h"][2] for r in rows], float),
        hand_z=np.array([r["hand_z"] for r in rows], float),
        obj_z=np.array([r["obj_z"] for r in rows], float),
        pos_err=np.array([r["pos_err"] for r in rows], float),
        aperture=np.array([r["aperture"] for r in rows], float),
        ten_length=np.array([r["ten_length"] for r in rows], float),
        finger_qpos=np.array([r["finger_qpos"] for r in rows], float),
        Fn_L=np.array([r["Fn_L"] for r in rows], float),
        Fn_R=np.array([r["Fn_R"] for r in rows], float),
        rho_mean=np.array([r["rho_mean"] for r in rows], float),
        rho_max=np.array([r["rho_max"] for r in rows], float),
        Ft_sum=np.array([r["Ft_sum"] for r in rows], float),
        dist_mean=np.array([r["dist_mean"] for r in rows], float),
        residual=np.array([r["residual_norm"] for r in rows], float),
        nL=np.array([r["nL"] for r in rows], float),
        nR=np.array([r["nR"] for r in rows], float),
        n_contacts=np.array([r["n_contacts"] for r in rows], float),
        ctrl7=np.array([r["ctrl7"] for r in rows], float),
        act_force7=np.array([r["act_force7"] for r in rows], float),
    )
    slim_rows = [{k: v for k, v in r.items() if k != "contact_keys"} for r in rows[:: max(len(rows) // 40, 1)]]
    dump_json(RAW / f"{tag}.json", {"summary": summ, "sparse_rows": slim_rows})


def run_hold(
    snap,
    *,
    variant: str,
    interactive=False,
    playback=0.35,
    title="NOMINAL STATIC PINCH",
    show_construction=False,
):
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    if show_construction and interactive:
        return run_full_viewer(sim, cfg, playback, title)

    restore_snap(sim, snap)
    info = {"variant": variant}
    if variant == "harder":
        apply_harder(sim)
    elif variant.startswith("dt"):
        fac = float(variant.split("_")[1])
        sim.model.opt.timestep = float(sim.model.opt.timestep) / fac
        info["timestep"] = float(sim.model.opt.timestep)
    elif variant == "g0":
        sim.model.opt.gravity[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
    elif variant == "g0.5":
        sim.model.opt.gravity[:] = np.array(sim.model.opt.gravity, float) * 0.5
        mujoco.mj_forward(sim.model, sim.data)
    elif variant == "ideal":
        info["ideal"] = construct_ideal(sim)
    freeze(sim)
    if hasattr(sim.fsm, "v_des"):
        sim.fsm.v_des[:] = 0.0
    if hasattr(sim.fsm, "w_des"):
        sim.fsm.w_des[:] = 0.0

    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    n = int(round(HOLD_S / dt))
    stride = max(int(round(LOG_DT / dt)), 1)
    ez0 = float((np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3)).T @ (
        np.array(sim.data.xpos[sim.ids.object_body]) - np.array(sim.data.xpos[sim.ids.hand_body])
    ))[2])
    rows = [sample(sim, 0.0, ez0)]
    viewer = None
    cm = None
    pacer = None
    if interactive:
        import mujoco.viewer as mjviewer
        from training.demo_teleport_recovery_state import _cam
        from training.impact_demo_core import RENDER_HZ, RealtimePacer
        from training.impact_visualization_utils import apply_viewer_camera, enable_viewer_flags

        cm = mjviewer.launch_passive(sim.model, sim.data)
        viewer = cm.__enter__()
        apply_viewer_camera(viewer, _cam(sim))
        enable_viewer_flags(viewer, show_contact_points=False, show_contact_forces=False)
        pacer = RealtimePacer(playback, RENDER_HZ, dt)
        pacer.start(float(sim.data.time))
        pacer.note_sync(float(sim.data.time))
    try:
        for k in range(n):
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU, gains)
            t_hold = (k + 1) * dt
            if k % stride == 0 or k + 1 == n:
                rows.append(sample(sim, t_hold, ez0))
            if viewer is not None and pacer is not None and pacer.should_render(float(sim.data.time)):
                overlay_nom(
                    viewer,
                    "hold",
                    t_hold,
                    rows[-1]["aperture"],
                    1e3 * rows[-1]["dez"],
                    title,
                )
                viewer.sync()
                pacer.note_sync(float(sim.data.time))
                pacer.wait_if_ahead(float(sim.data.time))
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    break
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
    force_pts = {}
    for ts in (0.0, 1.0, 5.0, 10.0):
        i = int(np.argmin(np.abs(np.array([r["t_hold"] for r in rows]) - ts)))
        r = rows[i]
        force_pts[str(ts)] = {
            "t_hold": r["t_hold"],
            "F_contact_h": r["F_contact_h"],
            "Fg_h": r["Fg_h"],
            "a_lin_h": r["a_lin_h"],
            "residual_norm": r["residual_norm"],
            "rho_mean": r["rho_mean"],
            "nL": r["nL"],
            "nR": r["nR"],
        }
    extra = {
        "force_snapshots": force_pts,
        "live_origin": live_contact_origin(sim),
        "actuator": actuator_semantics(sim),
        "freeze": {
            "p_des": np.array(sim.fsm.p_des, float).tolist(),
            "rule": "p_des:=p_hand r_des:=R_hand v_cmd:=0 once then tick_vw zeros tau=-18",
        },
        "info": info,
    }
    # actuator/live at end of hold; also save start-of-hold copies from first restore in caller
    summ = summarize(rows, extra)
    return rows, summ


def run_full_viewer(sim, cfg, playback, title):
    import mujoco.viewer as mjviewer
    from training.demo_teleport_recovery_state import Z_AIR, PARENT_HOLD, _cam
    from training.impact_demo_core import RENDER_HZ, RealtimePacer
    from training.impact_visualization_utils import apply_viewer_camera, enable_viewer_flags

    dt = float(sim.model.opt.timestep)
    cm = mjviewer.launch_passive(sim.model, sim.data)
    viewer = cm.__enter__()
    apply_viewer_camera(viewer, _cam(sim))
    enable_viewer_flags(viewer, show_contact_points=False, show_contact_forces=False)
    pacer = RealtimePacer(playback, RENDER_HZ, dt)
    pacer.start(0.0)
    pacer.note_sync(0.0)
    air_ok = 0.0
    ez0 = None
    try:
        n_max = int(round(12.0 / dt))
        for _ in range(n_max):
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
            o = physical_pack(sim)
            t = float(sim.data.time)
            if pacer.should_render(t):
                overlay_nom(viewer, str(sim.fsm.phase), 0.0, float(o["aperture"]), 0.0, title)
                viewer.sync()
                pacer.note_sync(t)
                pacer.wait_if_ahead(t)
            if sim.fsm.phase == "lift" and sim.captured and float(o["obj_z"]) >= Z_AIR:
                if int(o["nL"]) > 0 and int(o["nR"]) > 0 and float(o["v_rel"]) < 0.05:
                    air_ok += dt
                else:
                    air_ok = 0.0
                if air_ok >= PARENT_HOLD and abs(float(o["e_x"])) < 0.003:
                    freeze(sim)
                    ez0 = float(o["e_z"]) if "e_z" in o else float(np.asarray(o["rh"])[2])
                    break
        if ez0 is None:
            freeze(sim)
            ez0 = float(physical_pack(sim)["rh"][2])
        gains = gains_from_cfg(cfg)
        n = int(round(HOLD_S / dt))
        for k in range(n):
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU, gains)
            t_hold = (k + 1) * dt
            o = physical_pack(sim)
            dez = 1e3 * (float(np.asarray(o["rh"])[2]) - ez0)
            if pacer.should_render(float(sim.data.time)):
                overlay_nom(viewer, "hold", t_hold, float(o["aperture"]), dez, title)
                viewer.sync()
                pacer.note_sync(float(sim.data.time))
                pacer.wait_if_ahead(float(sim.data.time))
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    break
    finally:
        cm.__exit__(None, None, None)
    return [], {"viewer_only": True}


def figures(metrics: dict):
    FIG.mkdir(parents=True, exist_ok=True)

    def L(tag):
        return np.load(RAW / f"{tag}.npz")

    b = L("natural_baseline")
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(b["t_hold"], 1e3 * (b["e_z"] - b["e_z"][0]))
    ax[0].set_ylabel("Delta r_h.z mm")
    ax[1].plot(b["t_hold"], b["v_rel_z"])
    ax[1].set_ylabel("v_rel_h.z m/s")
    ax[1].set_xlabel("hold t s")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figA_rh_vrel.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(b["t_hold"], 1e3 * (b["hand_z"] - b["hand_z"][0]), label="hand world z")
    ax.plot(b["t_hold"], 1e3 * (b["obj_z"] - b["obj_z"][0]), label="object world z")
    ax.plot(b["t_hold"], 1e3 * (b["e_z"] - b["e_z"][0]), label="r_h.z (hand frame)")
    ax.set_ylabel("Delta mm")
    ax.set_xlabel("hold t s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figB_world_vs_rel.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(b["t_hold"], 1e3 * b["aperture"])
    ax[0].set_ylabel("aperture mm")
    ax[1].plot(b["t_hold"], 1e3 * b["finger_qpos"][:, 0], label="j1")
    ax[1].plot(b["t_hold"], 1e3 * b["finger_qpos"][:, 1], label="j2")
    ax[1].set_ylabel("finger qpos mm")
    ax[1].legend()
    ax[2].plot(b["t_hold"], b["ten_length"])
    ax[2].set_ylabel("tendon length")
    ax[2].set_xlabel("hold t s")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figC_fingers.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(b["t_hold"], b["Fn_L"], label="Fn L")
    ax[0].plot(b["t_hold"], b["Fn_R"], label="Fn R")
    ax[0].legend()
    ax[0].set_ylabel("Fn N")
    ax[1].plot(b["t_hold"], b["rho_mean"], label="rho mean")
    ax[1].plot(b["t_hold"], b["rho_max"], ls=":", label="rho max")
    ax[1].set_ylabel("rho")
    ax[1].set_xlabel("hold t s")
    ax[1].legend()
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figD_fn_rho.png", dpi=130)
    plt.close(fig)

    h = L("natural_harder")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(b["t_hold"], 1e3 * (b["e_z"] - b["e_z"][0]), label="baseline 0.015")
    ax.plot(h["t_hold"], 1e3 * (h["e_z"] - h["e_z"][0]), label="harder 0.005")
    ax.set_ylabel("Delta r_h.z mm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figE_harder.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, lab in (
        ("natural_baseline", "dt=0.002"),
        ("natural_dt_2", "dt/2"),
        ("natural_dt_4", "dt/4"),
    ):
        z = L(tag)
        ax.plot(z["t_hold"], 1e3 * (z["e_z"] - z["e_z"][0]), label=lab)
    ax.set_ylabel("Delta r_h.z mm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figF_timestep.png", dpi=130)
    plt.close(fig)

    g0 = L("natural_g0")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(b["t_hold"], 1e3 * (b["e_z"] - b["e_z"][0]), label="1 g")
    ax.plot(g0["t_hold"], 1e3 * (g0["e_z"] - g0["e_z"][0]), label="0 g")
    if (RAW / "natural_g05.npz").exists():
        g5 = L("natural_g05")
        ax.plot(g5["t_hold"], 1e3 * (g5["e_z"] - g5["e_z"][0]), label="0.5 g")
    ax.set_ylabel("Delta r_h.z mm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figG_gravity.png", dpi=130)
    plt.close(fig)

    ideal = L("ideal_baseline")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(b["t_hold"], 1e3 * (b["e_z"] - b["e_z"][0]), label="natural grasp")
    ax.plot(ideal["t_hold"], 1e3 * (ideal["e_z"] - ideal["e_z"][0]), label="ideal static pinch")
    ax.set_ylabel("Delta r_h.z mm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figH_ideal.png", dpi=130)
    plt.close(fig)


def classify(m: dict) -> tuple[str, str]:
    b = m["natural_baseline"]
    hard = m["natural_harder"]
    g0 = m["natural_g0"]
    dt2 = m["natural_dt_2"]
    dt4 = m["natural_dt_4"]
    ideal = m["ideal_baseline"]
    sb, sh, sg, s2, s4, si = (
        b["slope_ez_1_10_mm_s"],
        hard["slope_ez_1_10_mm_s"],
        g0["slope_ez_1_10_mm_s"],
        dt2["slope_ez_1_10_mm_s"],
        dt4["slope_ez_1_10_mm_s"],
        ideal["slope_ez_1_10_mm_s"],
    )
    labels = []
    # A: hand world motion vs relative
    hand_explains = abs(b["d_hand_z_world_mm"]) > 0.5 * abs(b["d_ez_10s_mm"]) and abs(
        b["d_obj_minus_hand_world_mm"]
    ) < 0.3 * abs(b["d_ez_10s_mm"])
    # r_h.z is not world z; if relative motion is large, hand drift is not the relative cause
    rel_is_object = abs(b["d_ez_10s_mm"]) > 2.0 and abs(b["d_obj_minus_hand_world_mm"]) > 0.5 * abs(
        b["d_ez_10s_mm"]
    )
    if hand_explains and not rel_is_object:
        labels.append("A")
    if abs(b["d_aperture_mm"]) > 0.3 and abs(b["d_aperture_mm"]) > 0.2 * abs(b["d_ez_10s_mm"]):
        labels.append("B")
    g_killed = abs(sg) < 0.25 * abs(sb) or abs(sg) < 0.15
    hard_cut = abs(sh) < 0.6 * abs(sb)
    fingers_still = abs(b["d_aperture_mm"]) < 0.15
    if g_killed and hard_cut and fingers_still and rel_is_object:
        labels.append("C")
    dt_strong = abs(s4) < 0.5 * abs(sb) or abs(s2) < 0.5 * abs(sb)
    dt_flat = abs(s2 - sb) < 0.15 * abs(sb) and abs(s4 - sb) < 0.25 * abs(sb)
    if dt_strong:
        labels.append("D")
    if b["churn_births_per_s"] > 50 and b["median_key_lifetime_s"] < 0.05:
        labels.append("E")
    if abs(si) < 0.25 * abs(sb) and abs(sb) > 0.4:
        labels.append("F")
    if not labels:
        return "H. UNRESOLVED", f"slopes baseline={sb:.3f} harder={sh:.3f} g0={sg:.3f} dt/2={s2:.3f} ideal={si:.3f}"
    if labels == ["C"] or set(labels) <= {"C"} or (labels == ["C"] ):
        msg = "Creep persists with nearly fixed fingers; vanishes/shrinks without gravity; shrinks when solref timeconst is reduced; timestep does not remove it." if dt_flat else "Gravity-loaded and solref-sensitive; check D as well."
        return "C. CONTACT COMPLIANCE / REGULARIZATION" + (" + D" if "D" in labels else ""), msg
    if "C" in labels and "D" in labels:
        return "G. MIXED (C+D)", "Compliant contact plus timestep contribution."
    if "C" in labels and "A" in labels:
        return "G. MIXED (A+C)", "Hand tracking sag exists, but relative object-hand distal creep remains after accounting for world motion."
    code = "+".join(labels)
    return f"G. MIXED ({code})" if len(labels) > 1 else f"{labels[0]}", str(labels)


def write_md(metrics, classif, act0, origin0):
    p = OUT / "NOMINAL_GRASP_CREEP_ROOT_CAUSE.md"
    L = []
    a = L.append
    a("# Why the undisturbed nominal pinch creeps (+hand-z)")
    a("")
    a("Root-cause diagnostic. Official XML, recovery, reward, observation, and training were not changed.")
    a("")
    a("## Evidence classes")
    a("- VERIFIED CODE FACT")
    a("- RAW TRAJECTORY EVIDENCE")
    a("- USER VISUAL OBSERVATION (commands below; not claimed here)")
    a("- INTERPRETATION / HYPOTHESIS")
    a("")
    a("## 1. Minimal test (CODE)")
    a("")
    a("Ordinary `advance_to_parent`: centered cylinder, m=0.20 kg, pair mu=1.0, no offset/teleport/impact. Then freeze-here `p_des:=p_hand`, `r_des:=R_hand`, `v_cmd:=0`, `tau=-18`, hold 10 s, no lift.")
    a("")
    a("## 3–4. Baseline 10 s (RAW)")
    a("")
    b = metrics["natural_baseline"]
    a(f"- Δ r_h.z at 1/2/5/10 s: **{b['d_ez_1s_mm']:.2f} / {b['d_ez_2s_mm']:.2f} / {b['d_ez_5s_mm']:.2f} / {b['d_ez_10s_mm']:.2f} mm**")
    a(f"- robust slope 1–10 s: **{b['slope_ez_1_10_mm_s']:.3f} mm/s**")
    a(f"- median v_rel_h.z 1–10 s: **{b['median_vz_1_10']:.5f} m/s**")
    a(f"- Δ hand world z: {b['d_hand_z_world_mm']:.2f} mm; Δ object world z: {b['d_obj_z_world_mm']:.2f} mm")
    a(f"- Δ (object−hand) world z: {b['d_obj_minus_hand_world_mm']:.2f} mm")
    a(f"- Δ aperture: {b['d_aperture_mm']:.4f} mm")
    a(f"- rho mean/max: {b['rho_mean_med']:.3f} / {b['rho_max_max']:.3f}")
    a(f"- Fn L/R med: {b['Fn_L_med']:.2f} / {b['Fn_R_med']:.2f} N")
    a(f"- |ma-(Fc+Fg)| median: {b['residual_median']:.4f} N")
    a(f"- hand pos_err end mm: {b['hand_pos_err_end_mm']}")
    a("")
    a("**VERIFIED CODE FACT:** `grasp_orientation()` sets hand Z = world down; +r_h.z is distal / fingertips.")
    a("")
    a("If |Δ r_h.z| remains after subtracting world co-motion of hand and object, contact/object-relative motion is implicated rather than Cartesian sag alone.")
    a("")
    a("## 5. Grip actuator (CODE)")
    a("")
    a("```")
    a(json.dumps(act0, indent=2, default=str)[:4000])
    a("```")
    a("")
    a("## 9–10. Live contact and solref origin (CODE)")
    a("")
    a("```")
    a(json.dumps({k: origin0[k] for k in origin0 if k != "pad_geoms"}, indent=2, default=str)[:5000])
    a("```")
    a("")
    a("Pads inherit MuJoCo default `solref=(0.02,1)`. Cylinder XML sets `solref=(0.01,1)`. Equal-priority mix → live **0.015**. This was not a pair-specific authoring.")
    a("")
    a("condim=4, pyramidal cone, impratio=1, integrator=implicitfast, dt=0.002. Low rho does not imply zero tangential regularization velocity.")
    a("")
    a("## Matrix")
    a("")
    a("| run | slope mm/s | d10 mm | med vz | rho | aperture dmm | |res| |")
    a("|---|---|---|---|---|---|---|")
    for k, s in metrics.items():
        a(
            f"| {k} | {s['slope_ez_1_10_mm_s']:.3f} | {s['d_ez_10s_mm']:.2f} | {s['median_vz_1_10']:.5f} | "
            f"{s['rho_mean_med']:.3f} | {s['d_aperture_mm']:.4f} | {s['residual_median']:.4f} |"
        )
    a("")
    a("## 18. Contact churn (baseline RAW)")
    a("")
    a(f"- unique contact keys (1e-4 m grid): {b['unique_contact_keys']}")
    a(f"- births/deaths per s: {b['churn_births_per_s']:.1f} / {b['churn_deaths_per_s']:.1f}")
    a(f"- median/mean key lifetime: {b['median_key_lifetime_s']:.3f} / {b['mean_key_lifetime_s']:.3f} s")
    a("")
    a("## 20. Causal decision tree")
    a("")
    a(f"**{classif[0]}**")
    a("")
    a(classif[1])
    a("")
    a("Not a claim that MuJoCo friction is broken. Not a claim a real Panda would/wouldn't slip. Not a claim that solref=0.005 is correct. Official model was not changed.")
    a("")
    a("## Viewer")
    a("")
    a("```")
    a("python training/nominal_grasp_creep_root_cause.py --view natural --playback-speed 0.35")
    a("python training/nominal_grasp_creep_root_cause.py --view ideal --playback-speed 0.35")
    a("python training/nominal_grasp_creep_root_cause.py --view g0 --playback-speed 0.35")
    a("```")
    a("")
    a("Natural viewer includes approach/close/lift then 10 s freeze-here hold.")
    p.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--view", choices=("natural", "ideal", "g0"), default=None)
    p.add_argument("--playback-speed", type=float, default=0.35)
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    if args.view:
        cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
        if args.view == "natural":
            sim = make_parent_sim()
            run_full_viewer(sim, cfg, args.playback_speed, "NOMINAL STATIC PINCH")
            return 0
        sp = RAW / "state_natural.json"
        if not sp.exists():
            print("run headless audit first")
            return 2
        snap = json.loads(sp.read_text(encoding="utf-8"))
        snap = {k: (np.array(v, float) if isinstance(v, list) else v) for k, v in snap.items()}
        var = "ideal" if args.view == "ideal" else "g0"
        title = "IDEAL STATIC PINCH" if var == "ideal" else "NOMINAL ZERO-G"
        run_hold(snap, variant=var, interactive=True, playback=args.playback_speed, title=title)
        return 0

    print("natural parent ...", flush=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    parent = advance_to_parent(sim, cfg, mode="ZERO")
    if not parent.get("ok"):
        raise RuntimeError("failed nominal parent")
    freeze(sim)
    snap = pack_snap(sim)
    ser = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in snap.items()}
    dump_json(RAW / "state_natural.json", ser)
    dump_json(RAW / "actuator_at_parent.json", actuator_semantics(sim))
    dump_json(RAW / "contact_origin_at_parent.json", live_contact_origin(sim))
    act0 = actuator_semantics(sim)
    origin0 = live_contact_origin(sim)

    plan = [
        ("natural_baseline", "baseline", "NOMINAL STATIC PINCH"),
        ("natural_harder", "harder", "NOMINAL HARDER CONTACT"),
        ("natural_dt_2", "dt_2", "NOMINAL dt/2"),
        ("natural_dt_4", "dt_4", "NOMINAL dt/4"),
        ("natural_g0", "g0", "NOMINAL ZERO-G"),
        ("natural_g05", "g0.5", "NOMINAL 0.5g"),
        ("ideal_baseline", "ideal", "IDEAL STATIC PINCH"),
    ]
    metrics = {}
    for tag, var, title in plan:
        print("hold", tag, "...", flush=True)
        rows, summ = run_hold(snap, variant=var, interactive=False, title=title)
        save_traj(tag, rows, summ)
        metrics[tag] = summ
        print(
            " ",
            tag,
            "slope",
            round(summ["slope_ez_1_10_mm_s"], 3),
            "d10",
            round(summ["d_ez_10s_mm"], 2),
            "ap",
            round(summ["d_aperture_mm"], 4),
            "g_handz",
            round(summ["d_hand_z_world_mm"], 2),
            flush=True,
        )

    figures(metrics)
    classif = classify(metrics)
    dump_json(RAW / "metrics.json", {k: {kk: vv for kk, vv in s.items() if kk not in ("live_origin", "actuator", "sparse")} for k, s in metrics.items()})
    write_md(metrics, classif, act0, origin0)
    print("class", classif[0])
    print("wrote", OUT / "NOMINAL_GRASP_CREEP_ROOT_CAUSE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
