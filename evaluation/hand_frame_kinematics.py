"""Hand-frame relative pose, contact, and twist measurements.

Shared by Plan C grasp diagnostics and the historical Plan A mechanism audit.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from envs.contacts.frames import extract_contacts
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT


def sample_kinematics(sim) -> dict[str, Any]:
    """Hand-frame relative pose, contacts, and world motion observables."""
    ids = sim.ids
    ph = np.array(sim.data.xpos[ids.hand_body], float)
    Rh = np.array(sim.data.xmat[ids.hand_body].reshape(3, 3), float)
    po = np.array(sim.data.xpos[ids.object_body], float)
    Ro = np.array(sim.data.xmat[ids.object_body].reshape(3, 3), float)
    p_rel = Rh.T @ (po - ph)
    R_rel = Rh.T @ Ro
    u_obj_w = Ro[:, 2]
    u_obj_h = Rh.T @ u_obj_w
    u_obj_h = u_obj_h / max(float(np.linalg.norm(u_obj_h)), 1e-12)
    u_tgt = np.asarray(U_TGT, float)
    u_tgt = u_tgt / max(float(np.linalg.norm(u_tgt)), 1e-12)
    cos_a = float(np.clip(abs(np.dot(u_obj_h, u_tgt)), -1.0, 1.0))
    axis_err_deg = float(np.degrees(np.arccos(cos_a)))

    rows, _ = extract_contacts(sim)
    nL = sum(1 for r in rows if r.get("side") == "L")
    nR = sum(1 for r in rows if r.get("side") == "R")
    s_vals = []
    fn_sum = 0.0
    ft_sum = 0.0
    for r in rows:
        if r.get("side") not in ("L", "R"):
            continue
        pw = np.asarray(r.get("pos_w", r.get("pos", [np.nan] * 3)), float)
        if np.all(np.isfinite(pw)):
            s_vals.append(float(np.dot(pw - po, u_obj_w)))
        fn_sum += abs(float(r.get("Fn", r.get("fn", 0.0))))
        ft_sum += float(r.get("Ft", r.get("ft", 0.0)) or 0.0)

    from envs.simulation.grasp_quality import body_twist

    vh, wh = body_twist(sim.model, sim.data, ids.hand_body)
    vo, wo = body_twist(sim.model, sim.data, ids.object_body)
    v_rel_h = Rh.T @ (vo - vh)
    w_rel_h = Rh.T @ (wo - wh)
    v_axial = float(np.dot(v_rel_h, u_obj_h))

    m = measure(sim, U_TGT, P_TGT)
    finger_q = []
    for jname in ("finger_joint1", "finger_joint2"):
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            finger_q.append(float(sim.data.qpos[int(sim.model.jnt_qposadr[jid])]))

    return {
        "t": float(sim.data.time),
        "p_hand": ph.copy(),
        "p_obj": po.copy(),
        "p_rel": p_rel.copy(),
        "p_rel_axial": float(np.dot(p_rel, u_obj_h)),
        "p_rel_trans": float(np.linalg.norm(p_rel - np.dot(p_rel, u_obj_h) * u_obj_h)),
        "u_obj_h": u_obj_h.copy(),
        "axis_err_deg": axis_err_deg,
        "R_rel": R_rel.copy(),
        "nL": nL,
        "nR": nR,
        "bilateral": bool(nL > 0 and nR > 0),
        "contact_s_mean": float(np.mean(s_vals)) if s_vals else float("nan"),
        "fn_sum": fn_sum,
        "ft_sum": ft_sum,
        "v_rel_h": v_rel_h.copy(),
        "w_rel_h": w_rel_h.copy(),
        "v_axial": v_axial,
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "obj_z": float(po[2]),
        "tau_cmd": float(sim.fsm.fg_cmd) if hasattr(sim.fsm, "fg_cmd") else float("nan"),
        "finger_q": finger_q,
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "ncon": int(sim.data.ncon),
    }
