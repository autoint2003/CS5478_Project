"""Finger-cylinder contacts in the MuJoCo contact frame.

frame rows are the contact axes. mj_contactForce is the wrench on geom2.
The cylinder force uses that vector, or its opposite when the cylinder is geom1.
"""

from __future__ import annotations

import mujoco
import numpy as np

from envs.simulation.grasp_quality import body_twist


def _gname(model, gid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return str(name) if name else f"geom_{int(gid)}"


def _bname(model, bid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
    return str(name) if name else f"body_{int(bid)}"


def pad_span_hand(sim) -> dict:
    """Finger-pad AABB in the hand frame from live box geoms (not e_x/e_z by fiat).

    Finger bodies sit at hand +z ≈ 0.0584 m. Pad boxes are in the finger frame:
    main pad size (8.5, 4, 8.5) mm at finger (0, 5.5, 44.5) mm. Finger +z is the
    distal/fingertip direction, which is +hand-z. Pads face each other along ±y.
    Pad width is ±hand-x about 0. The useful capture box is that AABB.
    """
    model, data, ids = sim.model, sim.data, sim.ids
    Rh = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(data.xpos[ids.hand_body], float)
    xs, ys, zs = [], [], []
    for bid in (int(ids.left_body), int(ids.right_body)):
        for g in range(int(model.ngeom)):
            if int(model.geom_bodyid[g]) != bid:
                continue
            if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            sz = np.array(model.geom_size[g], float)
            xpos = np.array(data.geom_xpos[g], float)
            xmat = np.array(data.geom_xmat[g].reshape(3, 3), float)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for szz in (-1.0, 1.0):
                        pw = xpos + xmat @ (sz * np.array([sx, sy, szz]))
                        h = Rh.T @ (pw - ph)
                        xs.append(float(h[0]))
                        ys.append(float(h[1]))
                        zs.append(float(h[2]))
    if not xs:
        return {
            "x_lo": -0.0085,
            "x_hi": 0.0085,
            "y_lo": -0.02,
            "y_hi": 0.02,
            "z_lo": 0.094,
            "z_hi": 0.111,
            "center": [0.0, 0.0, 0.103],
        }
    return {
        "x_lo": min(xs),
        "x_hi": max(xs),
        "y_lo": min(ys),
        "y_hi": max(ys),
        "z_lo": min(zs),
        "z_hi": max(zs),
        "center": [0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)), 0.5 * (min(zs) + max(zs))],
    }


def extract_contacts(sim) -> tuple[list[dict], dict]:
    """Per cylinder-finger contact using installed MuJoCo contact.frame / mj_contactForce.

    VERIFIED CODE FACT (from MuJoCo docs + this call path, not the old Fn-only helper):
    - mjContact.frame is stored transposed: rows are axes.
      frame[0:3] = contact X = normal (world)
      frame[3:6] = contact Y = tangent1
      frame[6:9] = contact Z = tangent2
    - mj_contactForce -> wrench[0:3] force, wrench[3:6] torque, IN THAT CONTACT FRAME.
      wrench[0] = Fn (normal). wrench[1], wrench[2] = Ft1, Ft2.
    - Docs: efc contact force points from first toward second geom; mj_contactForce
      is the intuitive 3+3 in the contact frame. GitHub #1314: force is on geom[1]
      (Python geom2). We convert to force-on-cylinder and CHECK against m*a.
    """
    model, data, ids = sim.model, sim.data, sim.ids
    Rh = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(data.xpos[ids.hand_body], float)
    po = np.array(data.xpos[ids.object_body], float)
    obj_g = int(ids.object_geom)
    obj_b = int(ids.object_body)
    wr = np.zeros(6)
    acc = np.zeros(6)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, obj_b, acc, 0)
    a_ang_w, a_lin_w = acc[:3].copy(), acc[3:].copy()
    mass = float(model.body_mass[obj_b])
    g_w = np.array(model.opt.gravity, float)
    Fg_w = mass * g_w
    rows = []
    F_obj_w = np.zeros(3)
    Tau_obj_w = np.zeros(3)
    F_L = np.zeros(3)
    F_R = np.zeros(3)
    Tau_L = np.zeros(3)
    Tau_R = np.zeros(3)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        if obj_b not in (b1, b2):
            continue
        if ids.left_body not in (b1, b2) and ids.right_body not in (b1, b2):
            continue
        mujoco.mj_contactForce(model, data, i, wr)
        n = np.array(c.frame[0:3], float)
        t1 = np.array(c.frame[3:6], float)
        t2 = np.array(c.frame[6:9], float)
        fn, ft1, ft2 = float(wr[0]), float(wr[1]), float(wr[2])
        tau_c = np.array(wr[3:6], float)
        f_c = np.array([fn, ft1, ft2], float)
        # world force corresponding to mj_contactForce (on geom2 / geom[1])
        f_on_geom2_w = n * fn + t1 * ft1 + t2 * ft2
        # map onto cylinder
        if g2 == obj_g or b2 == obj_b:
            f_obj = f_on_geom2_w
            on_object_as = "geom2"
        else:
            f_obj = -f_on_geom2_w
            on_object_as = "geom1_negated"
        pos_w = np.array(c.pos, float)
        r_com = pos_w - po
        tau_obj = np.cross(r_com, f_obj)
        F_obj_w += f_obj
        Tau_obj_w += tau_obj
        side = "L" if ids.left_body in (b1, b2) else "R"
        if side == "L":
            F_L += f_obj
            Tau_L += tau_obj
        else:
            F_R += f_obj
            Tau_R += tau_obj
        mu_vec = np.array(c.friction, float)
        mu = float(mu_vec[0]) if mu_vec.size else float("nan")
        ft_mag = float(np.hypot(ft1, ft2))
        mu_fn = mu * abs(fn)
        rho = float(ft_mag / mu_fn) if mu_fn > 1e-9 else float("nan")
        pos_h = Rh.T @ (pos_w - ph)
        n_h = Rh.T @ n
        f_obj_h = Rh.T @ f_obj
        fric_g1 = np.array(model.geom_friction[g1], float)
        fric_g2 = np.array(model.geom_friction[g2], float)
        rows.append(
            {
                "i": i,
                "side": side,
                "geom1": g1,
                "geom2": g2,
                "geom1_name": _gname(model, g1),
                "geom2_name": _gname(model, g2),
                "body1": b1,
                "body2": b2,
                "body1_name": _bname(model, b1),
                "body2_name": _bname(model, b2),
                "pos_w": pos_w.tolist(),
                "pos_h": pos_h.tolist(),
                "pos_rel_com": r_com.tolist(),
                "n_w": n.tolist(),
                "t1_w": t1.tolist(),
                "t2_w": t2.tolist(),
                "n_h": n_h.tolist(),
                "Fn": fn,
                "Ft1": ft1,
                "Ft2": ft2,
                "Ft": ft_mag,
                "mu": mu,
                "friction_vec": mu_vec.tolist(),
                "geom1_friction": fric_g1.tolist(),
                "geom2_friction": fric_g2.tolist(),
                "mu_Fn": mu_fn,
                "rho": rho,
                "dist": float(c.dist),
                "dim": int(c.dim),
                "solref": np.array(c.solref, float).tolist() if hasattr(c, "solref") else None,
                "solimp": np.array(c.solimp, float).tolist() if hasattr(c, "solimp") else None,
                "wrench6": wr.tolist(),
                "mu_geom": float(c.mu) if hasattr(c, "mu") else None,
                "includemargin": float(c.includemargin) if hasattr(c, "includemargin") else None,
                "on_object_as": on_object_as,
                "f_obj_w": f_obj.tolist(),
                "f_obj_h": f_obj_h.tolist(),
                "tau_obj_w": tau_obj.tolist(),
            }
        )
    F_net_w = F_obj_w + Fg_w
    ma_w = mass * a_lin_w
    residual_w = ma_w - F_net_w
    pad = pad_span_hand(sim)
    vo, wo = body_twist(model, data, obj_b)
    vh, wh = body_twist(model, data, ids.hand_body)
    v_rel_h = Rh.T @ (vo - vh)
    w_rel_h = Rh.T @ (wo - wh)
    rh = Rh.T @ (po - ph)
    p_des = np.asarray(sim.fsm.p_des, float)
    r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    summary = {
        "t": float(data.time),
        "mass": mass,
        "g_w": g_w.tolist(),
        "g_h": (Rh.T @ g_w).tolist(),
        "Fg_w": Fg_w.tolist(),
        "Fg_h": (Rh.T @ Fg_w).tolist(),
        "F_contact_w": F_obj_w.tolist(),
        "F_contact_h": (Rh.T @ F_obj_w).tolist(),
        "F_L_h": (Rh.T @ F_L).tolist(),
        "F_R_h": (Rh.T @ F_R).tolist(),
        "Tau_contact_w": Tau_obj_w.tolist(),
        "Tau_contact_h": (Rh.T @ Tau_obj_w).tolist(),
        "Tau_L_h": (Rh.T @ Tau_L).tolist(),
        "Tau_R_h": (Rh.T @ Tau_R).tolist(),
        "F_net_h": (Rh.T @ F_net_w).tolist(),
        "a_lin_w": a_lin_w.tolist(),
        "a_ang_w": a_ang_w.tolist(),
        "a_lin_h": (Rh.T @ a_lin_w).tolist(),
        "ma_w": ma_w.tolist(),
        "ma_h": (Rh.T @ ma_w).tolist(),
        "residual_w": residual_w.tolist(),
        "residual_h": (Rh.T @ residual_w).tolist(),
        "residual_norm": float(np.linalg.norm(residual_w)),
        "rh": rh.tolist(),
        "v_rel_h": v_rel_h.tolist(),
        "w_rel_h": w_rel_h.tolist(),
        "ph": ph.tolist(),
        "po": po.tolist(),
        "vh": vh.tolist(),
        "wh": wh.tolist(),
        "p_des": p_des.tolist(),
        "pos_err": (p_des - ph).tolist(),
        "r_des": r_des.reshape(9).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).tolist(),
        "pad": pad,
        "n": len(rows),
        "nL": sum(1 for r in rows if r["side"] == "L"),
        "nR": sum(1 for r in rows if r["side"] == "R"),
        "cone": int(model.opt.cone),
        "impratio": float(model.opt.impratio),
        "noslip_iterations": int(getattr(model.opt, "noslip_iterations", 0) or 0),
        "mujoco": mujoco.__version__,
    }
    return rows, summary
