"""Current recovery runtime: 7D action, snapshot restore, t=12 physical oracle.

No classifier. No learned handoff. No SAC.
Physical success is bilateral retention through absolute t=12 under the nominal controller.
"""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.jacobian_controller import apply_cartesian_ctrl, gains_from_cfg
from controllers.nominal import _integrate_rot
from controllers.residual import (
    RECOVERY4D_TAU_OPEN,
    RECOVERY4D_TAU_SECURE,
    RECOVERY4D_V_HX_MAX,
    RECOVERY4D_W_HY_MAX,
)
from envs.deterioration import body_twist
from envs.dynamics import finger_pad_geom_ids, set_finger_object_sliding_mu, set_object_dynamics
from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from training.demo_ballistic_impact import (
    HAND_RISE_M,
    TAU_SEC,
    assert_noslip,
    ball_force_on_ball,
    z_tgt_of,
)
from training.impact_ball import apply_ball_free_state, incoming_hand_x
from training.replay_core import park_impact_ball, tick_vw
from training.write_ballistic_impact_scene import BALL_R, CYL_R

T_TASK = 12.0
Z_LIFT = 0.51
ESCAPE_XY = 0.12
N_SUB = 10
V_MAX = RECOVERY4D_V_HX_MAX
W_MAX = RECOVERY4D_W_HY_MAX


def tau_of(a_grip: float) -> float:
    a = float(np.clip(a_grip, -1.0, 1.0))
    return RECOVERY4D_TAU_OPEN + 0.5 * (a + 1.0) * (RECOVERY4D_TAU_SECURE - RECOVERY4D_TAU_OPEN)


def map7(action, r_des, r_hand):
    """v in the hand frame, w in the r_des frame, grip on the same tendon map as recovery4d."""
    a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
    if a.size < 7:
        raise ValueError("recovery7 action needs 7 components")
    v_h = a[0:3] * V_MAX
    w_b = a[3:6] * W_MAX
    rh = np.asarray(r_hand, float).reshape(3, 3)
    rd = np.asarray(r_des, float).reshape(3, 3)
    return rh @ v_h, rd @ w_b, tau_of(a[6]), v_h, w_b


def capture_pads(sim):
    left, right = finger_pad_geom_ids(sim.model, sim.ids)
    pads = left + right
    return pads, np.array([float(sim.model.geom_friction[g][0]) for g in pads], float).copy()


def capture_nominal(sim):
    bid, gid = sim.ids.object_body, sim.ids.object_geom
    return (
        float(sim.model.body_mass[bid]),
        np.array(sim.model.body_inertia[bid], float).copy(),
        np.array(sim.model.geom_friction[gid], float).copy(),
    )


def apply_dyn(sim, nominal, mass, friction):
    bid, gid = sim.ids.object_body, sim.ids.object_geom
    sim.model.body_mass[bid] = nominal[0]
    sim.model.body_inertia[bid] = nominal[1].copy()
    sim.model.geom_friction[gid] = nominal[2].copy()
    if mass is None and friction is None:
        return float(nominal[0]), float(nominal[2][0])
    set_object_dynamics(
        sim.model, sim.ids,
        nominal[0] if mass is None else mass,
        nominal[2][0] if friction is None else friction,
        sim.data,
    )
    return float(sim.model.body_mass[bid]), float(sim.model.geom_friction[gid][0])


def make_snap(sim) -> dict:
    s = sim.snapshot()
    s["friction"] = 1.0
    s["eq_active"] = int(sim.eq_active())
    s["mocap_pos"] = np.array(sim.data.mocap_pos, float).copy()
    s["mocap_quat"] = np.array(sim.data.mocap_quat, float).copy()
    s["z_tgt"] = z_tgt_of(sim)
    return s


def restore_ballistic(sim, snap) -> None:
    sim.load_snapshot(snap)
    on = bool(int(snap.get("eq_active", 0)))
    sim.set_guide_weld(on)
    if "mocap_pos" in snap:
        sim.data.mocap_pos[:] = np.asarray(snap["mocap_pos"], float)
        sim.data.mocap_quat[:] = np.asarray(snap["mocap_quat"], float)
    mujoco.mj_forward(sim.model, sim.data)


def restore_dyn(sim, snap, nominal, pad_ids, pad_mu, mass, friction) -> dict:
    restore_ballistic(sim, snap)
    qpos = np.array(sim.data.qpos, float).copy()
    qvel = np.array(sim.data.qvel, float).copy()
    ctrl = np.array(sim.data.ctrl, float).copy()
    t = float(sim.data.time)
    p_des = np.asarray(sim.fsm.p_des, float).copy()
    r_des = np.asarray(sim.fsm.r_des, float).copy()
    v_cmd = np.asarray(sim.fsm.v_cmd, float).copy()
    w_cmd = np.asarray(sim.fsm.w_cmd, float).copy()
    apply_dyn(sim, nominal, mass, friction)
    for g, mu in zip(pad_ids, pad_mu):
        sim.model.geom_friction[g][0] = float(mu)
    if friction is not None:
        set_finger_object_sliding_mu(sim.model, sim.ids, float(friction), sim.data)
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.data.ctrl[:] = ctrl
    sim.data.time = t
    sim.fsm.p_des = p_des.copy()
    sim.fsm.r_des = r_des.copy()
    sim.fsm.v_cmd = v_cmd.copy()
    sim.fsm.w_cmd = w_cmd.copy()
    mujoco.mj_forward(sim.model, sim.data)
    z = float(sim.data.xpos[sim.ids.object_body][2])
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    z_ref = float(snap.get("obj_z_ref", z))
    hand_ref = np.asarray(snap.get("hand_ref", hand), float)
    return {
        "pose_ok": bool(abs(z - z_ref) < 0.003 and np.linalg.norm(hand - hand_ref) < 0.003),
        "obj_z": z,
        "mass": float(sim.model.body_mass[sim.ids.object_body]),
        "friction": float(sim.model.geom_friction[sim.ids.object_geom][0]),
    }


def step7(sim, action, gains=None):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    v, w, tau, _vh, _wb = map7(action, rd, rh)
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    if gains is None:
        gains = gains_from_cfg(sim.cfg)
    dt = float(sim.model.opt.timestep)
    for _ in range(N_SUB):
        sim.fsm.p_des = sim.fsm.p_des + v * dt
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, w, dt)
        sim.fsm.v_cmd = v.copy()
        sim.fsm.w_cmd = w.copy()
        sim.fsm.v_des = v.copy()
        sim.fsm.w_des = w.copy()
        sim.fsm.fg_cmd = float(-tau)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=v, w_des=w, **gains,
        )
        mujoco.mj_step(sim.model, sim.data)
    return v, w, tau


def step_cmd(sim, kind, action):
    """Compatibility wrapper. kind '7' is the current action. kind '4' is refused."""
    if kind == "4":
        raise RuntimeError("recovery4d is not the training action; use recovery7")
    v, w, tau = step7(sim, action)
    return v, w, v, w, 0


def _end_state(sim) -> dict:
    o = physical_pack(sim)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    return {
        "t": float(sim.data.time),
        "obj_z": float(o["obj_z"]),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "xy": float(np.linalg.norm(po[:2] - ph[:2])),
    }


def _dropped(o) -> bool:
    return float(o["obj_z"]) < TABLE_DROP


def _escaped(sim) -> bool:
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    return float(np.linalg.norm((po - ph)[:2])) > ESCAPE_XY


def continue_long(sim, gains, t_end=T_TASK, label=""):
    """Unchanged nominal continuation. Early exit only after a persisted drop or escape."""
    dt = float(sim.model.opt.timestep)
    t0 = float(sim.data.time)
    t0_z = float(physical_pack(sim)["obj_z"])
    t_drop = None
    t_esc = None
    airborne_seen = bool(t0_z >= Z_AIR)
    n = int(round(max(0.0, t_end - t0) / dt)) + 50
    for _ in range(n):
        z_now = float(z_tgt_of(sim))
        lift_hold = str(sim.fsm.phase) == "lift" and float(sim.fsm.p_des[2]) >= z_now - 1e-9
        if lift_hold:
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        else:
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
        o = physical_pack(sim)
        if float(o["obj_z"]) >= Z_AIR:
            airborne_seen = True
        t = float(sim.data.time)
        if airborne_seen and _dropped(o) and t_drop is None:
            t_drop = t
        if _escaped(sim) and t_esc is None:
            t_esc = t
        if t_drop is not None and t >= t_drop + 0.15:
            break
        if t_esc is not None and t >= t_esc + 0.15:
            break
        if t >= t_end - 1e-9:
            break
    return {"label": label, "t0": t0, "t_drop": t_drop, "t_escape": t_esc, "end": _end_state(sim)}


def physical_of(end, t_drop, t_escape):
    if t_drop is not None:
        return 0, "DROP"
    if t_escape is not None:
        return 0, "ESCAPE"
    bilateral = int(end["nL"]) > 0 and int(end["nR"]) > 0
    lifted = float(end["obj_z"]) > Z_LIFT
    if bilateral and lifted:
        return 1, "RETAINED"
    if (not bilateral) and lifted:
        return 0, "LOSS_OF_GRASP"
    if bilateral and (not lifted):
        return 0, "FALL_BELOW_LIFT_HEIGHT"
    return 0, "OTHER_PHYSICAL_FAILURE"


def oracle_snap(sim, gains, snap, nominal, pad_ids, pad_mu, mass, friction) -> dict:
    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, mass, friction)
    park_impact_ball(sim)
    if not info["pose_ok"]:
        return {"y": 0, "outcome": "INVALID_RESTORE", "pose_ok": False, "end_z": info["obj_z"],
                "mass": info["mass"], "friction": info["friction"]}
    rec = continue_long(sim, gains, T_TASK, "oracle")
    y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
    return {
        "y": int(y),
        "outcome": "RETAINED_TO_12" if int(y) == 1 else kind,
        "pose_ok": True,
        "end_z": float(rec["end"]["obj_z"]),
        "end_nL": int(rec["end"]["nL"]),
        "end_nR": int(rec["end"]["nR"]),
        "mass": info["mass"],
        "friction": info["friction"],
    }


def _metrics(sim) -> dict:
    o = physical_pack(sim)
    rh = np.asarray(o["rh"], float)
    return {
        "rh_mm": [round(float(x) * 1e3, 2) for x in rh],
        "g_h": [round(float(x), 3) for x in np.asarray(o["g_h"], float)],
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "obj_z": round(float(o["obj_z"]), 4),
    }


def stamp(sim) -> dict:
    snap = make_snap(sim)
    snap["obj_z_ref"] = float(sim.data.xpos[sim.ids.object_body][2])
    snap["hand_ref"] = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    snap["metrics"] = _metrics(sim)
    return snap


def write_dyn_quiet(sim, nominal, mass, friction) -> None:
    bid = sim.ids.object_body
    if mass is not None:
        old = float(sim.model.body_mass[bid])
        sim.model.body_inertia[bid] = np.array(sim.model.body_inertia[bid], float) * (float(mass) / max(old, 1e-9))
        sim.model.body_mass[bid] = float(mass)
    if friction is not None:
        set_finger_object_sliding_mu(sim.model, sim.ids, float(friction), data=None)


def launch_offcenter(sim, v_hit: float, z_off: float, phi_deg: float) -> dict:
    u = incoming_hand_x(sim)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    vo, _wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    axis = Ro[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    u_perp = u - axis * float(np.dot(axis, u))
    nrm = float(np.linalg.norm(u_perp))
    if nrm < 1e-6:
        u_perp = u
        nrm = float(np.linalg.norm(u_perp))
    e_face = -u_perp / nrm
    e_tan = np.cross(axis, e_face)
    e_tan = e_tan / max(float(np.linalg.norm(e_tan)), 1e-12)
    phi = np.radians(float(phi_deg))
    T = 0.10
    g = np.array(sim.model.opt.gravity, float)
    po_pred = po + vo * T
    p_contact = po_pred + axis * float(z_off) + e_face * (CYL_R * np.cos(phi)) + e_tan * (CYL_R * np.sin(phi))
    n_out = e_face * np.cos(phi) + e_tan * np.sin(phi)
    p_ball = p_contact + n_out * BALL_R
    p_launch = p_ball - u * (float(v_hit) * T)
    v0 = (p_ball - p_launch) / T - 0.5 * g * T
    miss = float(np.linalg.norm(np.cross(p_ball - po_pred, u)))
    return {"p_launch": p_launch, "v0": v0, "miss_mm": miss * 1e3}


def run_impact(sim, gains, v_hit, z_off, phi, pads_l, pads_r, nominal, mass, friction, dyn_at="release") -> dict:
    del pads_l, pads_r
    assert_noslip(sim)
    sim.reset()
    apply_dyn(sim, nominal, None, None)
    if dyn_at == "start":
        write_dyn_quiet(sim, nominal, mass, friction)
    mujoco.mj_forward(sim.model, sim.data)
    dt = float(sim.model.opt.timestep)
    released = False
    parked = False
    t_impact = None
    t_last = None
    t_park = None
    pre = None
    snap_a = None
    launch = None
    holding = False
    hand_z0 = None
    z_tgt = None
    n_max = int(round(8.0 / dt))
    for _ in range(n_max):
        if (not holding) and sim.fsm.phase == "lift" and z_tgt is not None:
            if float(sim.fsm.p_des[2]) >= z_tgt - 1e-9:
                holding = True
        if holding:
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        else:
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
        t = float(sim.data.time)
        if sim.fsm.phase == "lift" and hand_z0 is None:
            hand_z0 = float(sim.data.xpos[sim.ids.hand_body][2])
            z_tgt = z_tgt_of(sim)
        o = physical_pack(sim)
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if (
            (not released)
            and sim.fsm.phase == "lift"
            and sim.captured
            and hand_z0 is not None
            and float(o["obj_z"]) >= Z_AIR
            and (hz - hand_z0) >= HAND_RISE_M
        ):
            pre = _metrics(sim)
            if dyn_at != "start":
                write_dyn_quiet(sim, nominal, mass, friction)
            launch = launch_offcenter(sim, v_hit, z_off, phi)
            sim.set_guide_weld(False)
            apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
            released = True
        if released and not parked:
            _fb, bc = ball_force_on_ball(sim)
            has = any(r["kind"] == "ball-cylinder" for r in bc)
            if has and t_impact is None:
                t_impact = t
            if has:
                t_last = t
            departed = t_impact is not None and t_last is not None and (not has) and t > t_last + 0.025
            stuck = t_impact is not None and t > t_impact + 0.35
            if departed or stuck:
                park_impact_ball(sim)
                parked = True
                t_park = float(sim.data.time)
        if parked and snap_a is None and t >= t_park + 0.040:
            snap_a = stamp(sim)
            break
        if float(o["obj_z"]) < 0.42 and released:
            break
    meta = {
        "v": float(v_hit),
        "z_off_mm": float(z_off) * 1e3,
        "phi_deg": float(phi),
        "t_impact": None if t_impact is None else round(float(t_impact), 4),
        "pre": pre,
        "early": None if snap_a is None else snap_a["metrics"],
    }
    return {"meta": meta, "snap": snap_a, "pre_rh": None if pre is None else pre["rh_mm"]}
