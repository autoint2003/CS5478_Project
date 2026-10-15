"""Stable horizontal recatch and vertical-contact helpers.

The Open, over-chase, close, and brake law stays distinct from the vertical
multi-axis chase. Experiment scripts call these functions.
"""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.geometry_switch import hand_pose
from controllers.jacobian_controller import apply_cartesian_ctrl
from controllers.recovery_actions import TAU_OPEN, grip_action, normalize_vz
from data_generation.ball_impact_dataset import cylinder_contacts
from data_generation.strategies.capture_geometry import PINCH_HAND, capture_geometry
from envs.airborne_obs import relative_state
from envs.ballistic_impact import TAU_SEC, assert_noslip, make_sim, z_tgt_of
from envs.deterioration import body_twist
from envs.grasp_seat import grasp_settled, restore_seated
from envs.horizon_outcome import NOMINAL_S, nominal_verify
from envs.airborne_obs import observe_airborne
from envs.observable_obs import ObservableObsState
from envs.recovery_success import nominal_step
from envs.step_legal import step_legal
from envs.table_drop import latched, object_vz, observe, restore_state
from training.full_3d_airborne_recatch_dataset import TRACK, capture_nominal, capture_pads, drop_supports
from training.impact_ball import apply_ball_free_state
from training.nominal_pose_audit import hand_R
from training.recovery_runtime import gains_from_cfg, make_snap, restore_ballistic, stamp
from training.replay_core import TABLE_TOP, park_impact_ball
from training.unified_k47 import load_codebook


def _nominal_fail(sim, snap, ref):
    """Official handoff plus the 12 s nominal window. This is the triviality test."""
    restore_ballistic(sim, snap)
    restore_state(sim, None)

    class Env:
        pass

    env = Env()
    env.sim = sim
    out = nominal_verify(env, ref, seconds=NOMINAL_S)
    return {
        "ok": bool(out.get("success")),
        "t": round(float(out.get("nominal_elapsed") or 0.0), 3),
        "why": out.get("reason"),
        "end": {"z": None if out.get("end") is None else out["end"].get("z")},
    }


def _action_from(sim, v_world, w_body, tau):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    v_h = rh.T @ np.asarray(v_world, float)
    from controllers.recovery_actions import grip_action, normalize_vz

    action = np.zeros(7, np.float32)
    action[0] = np.clip(v_h[0] / 1.0, -1.0, 1.0)
    action[1] = np.clip(v_h[1] / 1.0, -1.0, 1.0)
    action[2] = normalize_vz(float(v_h[2]))
    action[3:6] = np.clip(np.asarray(w_body, float) / 4.0, -1.0, 1.0)
    action[6] = grip_action(float(tau))
    return action


def _setup():
    sim, seated, _nominal, _pads, _pad_mu = _prepare()
    _enable_supports(sim)
    
    restore_seated(sim, seated, _nominal, _pads, _pad_mu)
    sim.park_ball()
    sim.set_guide_weld(True)
    restore_state(sim, None)
    ref = _ref_now(sim)
    target = relative_state(sim.model, sim.data, sim.ids)["p_rel_h"].copy()
    return sim, make_snap(sim), ref, target, load_codebook()


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


def _bottom(sim) -> float:
    gid = int(sim.ids.object_geom)
    pos = np.array(sim.data.geom_xpos[gid], float)
    rot = np.array(sim.data.geom_xmat[gid], float).reshape(3, 3)
    radius = float(sim.model.geom_size[gid][0])
    half = float(sim.model.geom_size[gid][1])
    axis_z = abs(float(rot[2, 2]))
    extent = axis_z * half + radius * float(np.sqrt(max(0.0, 1.0 - axis_z * axis_z)))
    return float(pos[2] - extent)


def _prepare():
    
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
    return sim, snap, nominal, pads, pad_mu


def _ref_now(sim):
    pose, relative = hand_pose(sim)
    return {
        "R_hand": hand_R(sim),
        "r_des": np.array(sim.fsm.r_des, float).copy(),
        "p_nom": np.array(pose, float).copy(),
        "R_rel": np.array(relative, float).copy(),
    }


def _apply_variant(sim, nominal, cell):
    if cell.get("mass_kg") is None and cell.get("friction") is None:
        return
    apply_dyn(sim, nominal, cell.get("mass_kg"), cell.get("friction"))
    if cell.get("friction") is not None:
        from envs.dynamics import set_finger_object_sliding_mu

        set_finger_object_sliding_mu(sim.model, sim.ids, float(cell["friction"]), sim.data)


def _flight_step(sim):
    """Nominal lift hold without the diagnostic ball park inside tick_vw.

    On this scene tick_vw writes the ball back to its park pose after every
    physics step, which deletes a launch. The arm command is the same hold.
    """
    import mujoco
    from controllers.jacobian_controller import apply_cartesian_ctrl
    from envs.ballistic_impact import TAU_SEC, z_tgt_of
    from training.full_3d_airborne_recatch_dataset import TRACK
    from envs.table_drop import object_vz, observe

    vz_before = object_vz(sim)
    z_now = float(z_tgt_of(sim))
    lift_hold = str(sim.fsm.phase) == "lift" and float(sim.fsm.p_des[2]) >= z_now - 1e-9
    if lift_hold:
        dt = float(sim.model.opt.timestep)
        tau = float(np.clip(TAU_SEC, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
        sim.fsm.p_des = np.asarray(sim.fsm.p_des, float) + np.zeros(3)
        sim.fsm.v_cmd = np.zeros(3)
        sim.fsm.w_cmd = np.zeros(3)
        sim.fsm.v_des = np.zeros(3)
        sim.fsm.w_des = np.zeros(3)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=np.zeros(3), w_des=np.zeros(3), **TRACK,
        )
        mujoco.mj_step(sim.model, sim.data)
    else:
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
    observe(sim, vz_before)


def _contacts(sim):
    """Finger forces on the cylinder, split into normal and friction."""
    import mujoco

    obj = int(sim.ids.object_geom)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    rows = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if obj not in (geom1, geom2):
            continue
        body1 = int(sim.model.geom_bodyid[geom1])
        body2 = int(sim.model.geom_bodyid[geom2])
        other = body2 if body1 == int(sim.ids.object_body) else body1
        # object_body vs geom body: use geom's body
        other = body2 if int(sim.model.geom_bodyid[geom1]) == int(sim.ids.object_body) else body1
        if other not in (left, right):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, index, wrench)
        normal = np.array(contact.frame[0:3], float)
        tan1 = np.array(contact.frame[3:6], float)
        tan2 = np.array(contact.frame[6:9], float)
        # MuJoCo's normal points from geom1 to geom2. The compression on geom1 is
        # opposite that normal. The static grasp must support +mg, so the sign is
        # checked against that equilibrium in the audit printout.
        on_first = -float(wrench[0]) * normal + float(wrench[1]) * tan1 + float(wrench[2]) * tan2
        on_cylinder = -(on_first if geom1 == obj else -on_first)
        normal_on_cylinder = -((-float(wrench[0]) * normal) if geom1 == obj else (float(wrench[0]) * normal))
        friction_on_cylinder = on_cylinder - normal_on_cylinder
        point = np.array(contact.pos, float)
        center = np.array(sim.data.xpos[sim.ids.object_body], float)
        rows.append({
            "side": "L" if other == left else "R",
            "pos": point,
            "normal_on_cylinder": normal_on_cylinder,
            "friction_on_cylinder": friction_on_cylinder,
            "fn": abs(float(wrench[0])),
            "ft": float(np.linalg.norm(wrench[1:3])),
            "fz_normal": float(normal_on_cylinder[2]),
            "fz_friction": float(friction_on_cylinder[2]),
            "offset": point - center,
        })
    return rows


def _pose(sim):
    rel = relative_state(sim.model, sim.data, sim.ids)
    geo = capture_geometry(sim)
    _linear, angular = body_twist(sim.model, sim.data, sim.ids.object_body)
    axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    forces = _contacts(sim)
    hand = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    pinch = np.array(sim.data.xpos[sim.ids.hand_body], float) + hand @ PINCH_HAND
    obj = np.array(rel["p_obj_w"], float)
    return {
        "p_obj": obj.copy(),
        "p_hand": np.array(rel["p_hand_w"], float).copy(),
        "pinch": pinch.copy(),
        "v_obj": np.array(rel["v_obj_w"], float).copy(),
        "v_hand": np.array(rel["v_hand_w"], float).copy(),
        "p_rel": np.array(rel["p_rel_h"], float).copy(),
        "v_rel": np.array(rel["v_rel_h"], float).copy(),
        "w": float(np.linalg.norm(angular)),
        "axis": axis.copy(),
        "tilt_deg": float(np.degrees(np.arctan2(abs(axis[2]), max(float(np.linalg.norm(axis[:2])), 1e-9)))),
        "nL": int(geo["nL"]),
        "nR": int(geo["nR"]),
        "gap": float(geo["inner_gap_m"]),
        "q": [float(v) for v in geo["finger_qpos_m"]],
        "between": bool(geo["between"]),
        "cz": float(geo["center_hand_m"][2]),
        "pad_z": [float(v) for v in geo["pad_z_m"]],
        "ready": bool(geo["ready_to_close"]),
        "bottom": float(_bottom(sim)),
        "fz_n": float(sum(item["fz_normal"] for item in forces)),
        "fz_f": float(sum(item["fz_friction"] for item in forces)),
        "fn": float(sum(item["fn"] for item in forces)),
        "contacts": forces,
        "qpos": np.array(sim.data.qpos, float).copy(),
        "grip": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else 0.0,
    }


def _aim(sim, speed, pitch_deg, axial, ecc):
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    opening = np.cross(up, axis)
    opening = opening / max(float(np.linalg.norm(opening)), 1e-9)
    radius = float(sim.model.geom_size[int(sim.ids.object_geom)][0])
    ball_r = float(sim.model.geom_size[int(sim.ball_geom)][0])
    # Upper surface, shifted along the axis and around the cross-section.
    p_contact = po + axis * float(axial) + opening * float(ecc) + up * np.sqrt(max(radius ** 2 - float(ecc) ** 2, 0.0))
    pitch = np.radians(float(pitch_deg))
    incoming = -up * np.cos(pitch) + axis * np.sin(pitch) * np.sign(axial if abs(axial) > 1e-6 else 1.0)
    incoming = incoming / max(float(np.linalg.norm(incoming)), 1e-9)
    v_imp = incoming * float(speed)
    p_ball = p_contact - incoming * (ball_r + 0.001)
    flight = 0.10
    g = np.array(sim.model.opt.gravity, float)
    v0 = v_imp - g * flight
    p_launch = p_ball - v0 * flight - 0.5 * g * flight * flight
    return {"p_launch": p_launch, "v0": v0, "v_imp": v_imp}


def _impact(sim, base, cell, after_restore=None):
    restore_ballistic(sim, base)
    if after_restore is not None:
        after_restore()
    restore_state(sim, None)
    sim.model.opt.timestep = 0.002
    sim.set_guide_weld(False)
    launch = _aim(sim, cell["speed"], cell["pitch"], cell["axial"], cell["ecc"])
    before = _pose(sim)
    apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
    if cylinder_contacts(sim)[0]:
        return None, "overlapped"
    t0 = float(sim.data.time)
    contact_t = None
    peak = 0.0
    impulse = 0.0
    worst_fz = 0.0
    rows = []
    unstable = None
    unstable_snap = None
    for _ in range(int(0.45 / 0.002)):
        _flight_step(sim)
        now = float(sim.data.time)
        hits, _other = cylinder_contacts(sim)
        if hits and contact_t is None:
            contact_t = now
        if hits:
            peak = max(peak, max(float(item["fn"]) for item in hits))
            impulse += sum(float(item["fn"]) for item in hits) * 0.002
        pose = _pose(sim)
        pose["t"] = now - t0
        if contact_t is not None and (len(rows) == 0 or now - t0 >= rows[-1]["t"] + 0.02):
            rows.append(pose)
        worst_fz = min(worst_fz, pose["fz_n"])
        if contact_t is None:
            if now > t0 + 0.30:
                break
            continue
        age = now - contact_t
        slipped = pose["p_rel"][2] > before["p_rel"][2] + 0.004 or pose["v_obj"][2] < -0.15
        lost_side = (pose["nL"] == 0) != (pose["nR"] == 0) or (pose["nL"] + pose["nR"] == 0)
        if age > 0.012 and (slipped or lost_side or pose["tilt_deg"] > before["tilt_deg"] + 8.0):
            if unstable is None or pose["p_rel"][2] > unstable["p_rel"][2]:
                unstable = pose
                unstable_snap = make_snap(sim)
        if latched(sim) or pose["bottom"] < TABLE_TOP + 0.02 or age > 0.22:
            break
    if contact_t is None:
        return None, "no_contact"
    pose = rows[-1] if rows else _pose(sim)
    horizontal = float(np.linalg.norm(pose["v_obj"][:2]))
    progressive = bool(
        peak >= 3.0
        and horizontal < 0.40
        and pose["bottom"] > 0.50
        and float(np.linalg.norm(pose["p_obj"][:2] - before["p_obj"][:2])) < 0.05
        and (pose["p_rel"][2] > before["p_rel"][2] + 0.003 or pose["v_obj"][2] < -0.12 or worst_fz < -0.3)
    )
    if unstable_snap is None:
        park_impact_ball(sim)
        if after_restore is not None:
            after_restore()
        unstable_snap = make_snap(sim)
    else:
        restore_ballistic(sim, unstable_snap)
        if after_restore is not None:
            after_restore()
        park_impact_ball(sim)
        unstable_snap = make_snap(sim)
    snap = unstable_snap
    return {
        "snap": snap,
        "before": before,
        "rows": rows,
        "pose": _pose(sim),
        "unstable": unstable,
        "contact_s": round(float(contact_t - t0), 3),
        "peak_n": round(float(peak), 2),
        "impulse": round(float(impulse), 3),
        "v_down": round(float(-launch["v_imp"][2]), 3),
        "worst_fz_normal": round(float(worst_fz), 3),
        "progressive": progressive,
        "horizontal": round(horizontal, 3),
    }, None


def _finish(sim, ref, extra=0.0):
    class Env:
        pass

    env = Env()
    env.sim = sim
    out = nominal_verify(env, ref, seconds=NOMINAL_S)
    extra_reason = None
    if out.get("success") and extra > 0.0:
        t0 = float(sim.data.time)
        while float(sim.data.time) < t0 + extra:
            nominal_step(sim)
            if latched(sim) or _bottom(sim) < TABLE_TOP:
                extra_reason = "FELL_DURING_EXTENSION"
                break
            geo = capture_geometry(sim)
            if int(geo["nL"]) == 0 or int(geo["nR"]) == 0:
                extra_reason = "LOST_DURING_EXTENSION"
                break
        if extra_reason is None:
            extra_reason = "HELD_EXTENSION"
    return out, extra_reason


MAIN_HALF = np.array([0.0085, 0.004, 0.0085])


MAIN_CENTER = np.array([0.0, 0.0055, 0.0445])


def _geom_name(sim, geom_id):
    import mujoco

    name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return name or f"geom{int(geom_id)}"


def _geom_kind(sim, geom_id):
    """Which finger collision piece this geom is.

    The large pad box is recessed. The stable grasp touches the inner face
    of the smaller fingertip boxes. Finger +Y points outward, so that face
    is the low-Y side.
    """
    import mujoco

    kind = int(sim.model.geom_type[geom_id])
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        return "MESH"
    size = np.array(sim.model.geom_size[geom_id], float)
    if np.allclose(size, MAIN_HALF, atol=1e-4):
        return "MAIN_PAD"
    pos = np.array(sim.model.geom_pos[geom_id], float)
    if float(pos[2]) > 0.046:
        return "CORNER_TIP"
    return "CORNER_PALM"


def _class_of(sim, geom_id, local):
    kind = _geom_kind(sim, geom_id)
    z = float(local[2])
    y = float(local[1])
    if kind == "MESH":
        if z < 0.036:
            return "UPPER_SLOPED_PART"
        return "OTHER"
    if kind in ("MAIN_PAD", "CORNER_PALM", "CORNER_TIP") and 0.034 <= z <= 0.055 and y < 0.005:
        return "LOWER_VERTICAL_PAD"
    return "EDGE_CONTACT"


def contacts_now(sim):
    """Every finger–cylinder contact, in the finger body frame."""
    import mujoco

    obj = int(sim.ids.object_geom)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    rows = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if obj not in (geom1, geom2):
            continue
        other_geom = geom2 if geom1 == obj else geom1
        body = int(sim.model.geom_bodyid[other_geom])
        if body not in (left, right):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, index, wrench)
        normal = np.array(contact.frame[0:3], float)
        tan1 = np.array(contact.frame[3:6], float)
        tan2 = np.array(contact.frame[6:9], float)
        on_first = -float(wrench[0]) * normal + float(wrench[1]) * tan1 + float(wrench[2]) * tan2
        on_cylinder = -(on_first if geom1 == obj else -on_first)
        normal_on = -((-float(wrench[0]) * normal) if geom1 == obj else (float(wrench[0]) * normal))
        friction_on = on_cylinder - normal_on
        point = np.array(contact.pos, float)
        origin = np.array(sim.data.xpos[body], float)
        rotation = np.array(sim.data.xmat[body].reshape(3, 3), float)
        local = rotation.T @ (point - origin)
        center = np.array(sim.data.xpos[sim.ids.object_body], float)
        rows.append({
            "side": "L" if body == left else "R",
            "geom": _geom_name(sim, other_geom),
            "kind": _geom_kind(sim, other_geom),
            "class": _class_of(sim, other_geom, local),
            "local": local,
            "offset": point - center,
            "normal": normal_on,
            "fn": abs(float(wrench[0])),
            "ft": float(np.linalg.norm(wrench[1:3])),
            "fz_n": float(normal_on[2]),
            "fz_f": float(friction_on[2]),
            "dist": float(contact.dist),
            "dim": int(contact.dim),
            "efc": int(contact.efc_address),
        })
    return rows


def _pad_band(sim):
    """Hand-frame z span of the main vertical pad, not the corner boxes."""
    import mujoco

    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    finger_axis = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)[:, 2]
    lows, highs = [], []
    for body in (int(sim.ids.left_body), int(sim.ids.right_body)):
        for geom_id in range(sim.model.ngeom):
            if int(sim.model.geom_bodyid[geom_id]) != body:
                continue
            if int(sim.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            size = np.array(sim.model.geom_size[geom_id], float)
            if not np.allclose(size, MAIN_HALF, atol=1e-4):
                continue
            origin = np.array(sim.data.xpos[body], float)
            rotation = np.array(sim.data.xmat[body].reshape(3, 3), float)
            center = origin + rotation @ MAIN_CENTER
            half_z = abs(float(rotation[:, 2] @ finger_axis)) * MAIN_HALF[2]
            cz = float(np.dot(center - hand, finger_axis))
            lows.append(cz - half_z)
            highs.append(cz + half_z)
    return float(np.min(lows)), float(np.max(highs))


def _class_force(rows):
    out = {name: 0.0 for name in ("LOWER_VERTICAL_PAD", "UPPER_SLOPED_PART", "EDGE_CONTACT", "OTHER")}
    sides = {name: set() for name in out}
    for row in rows:
        out[row["class"]] += row["fn"]
        if row["fn"] > 0.2:
            sides[row["class"]].add(row["side"])
    return out, {name: "".join(sorted(value)) for name, value in sides.items()}


def _wrist_sat(sim):
    force = np.array(sim.data.actuator_force, float)
    limits = np.array(sim.model.actuator_forcerange, float)
    peak = 0.0
    saturated = False
    for index in range(min(7, force.size)):
        limit = max(abs(float(limits[index, 0])), abs(float(limits[index, 1])))
        peak = max(peak, abs(float(force[index])))
        if limit > 0.0 and abs(float(force[index])) >= 0.95 * limit:
            saturated = True
    return peak, saturated


def _metrics(rows):
    def first(pred):
        for row in rows:
            if pred(row):
                return row
        return None

    opened = first(lambda row: row["gap"] > rows[0]["gap"] + 0.003)
    off = first(lambda row: row["nL"] == 0 and row["nR"] == 0)
    seen = False
    recontact = None
    pad_hit = None
    for row in rows:
        if row["nL"] == 0 and row["nR"] == 0:
            seen = True
        elif seen and recontact is None and row["nL"] > 0 and row["nR"] > 0:
            recontact = row
        if seen and pad_hit is None and row["pad_sides"] == "LR":
            pad_hit = row
    pad_run = 0
    sustained = None
    for row in rows:
        pad_run = pad_run + 1 if row["pad_sides"] == "LR" and row["fn_pad"] >= row["fn_slope"] else 0
        if pad_run >= 3 and sustained is None:
            sustained = row
    vz0 = first(lambda row: row["v_obj"][2] > -0.05 and row["t"] > 0.05)
    return {
        "t_open": None if opened is None else round(opened["t"], 3),
        "t_off": None if off is None else round(off["t"], 3),
        "t_re": None if recontact is None else round(recontact["t"], 3),
        "t_pad": None if pad_hit is None else round(pad_hit["t"], 3),
        "t_pad_hold": None if sustained is None else round(sustained["t"], 3),
        "t_vz0": None if vz0 is None else round(vz0["t"], 3),
        "re_class": "" if recontact is None else recontact["dominant"],
        "re_cz": None if recontact is None else round(recontact["cz"], 4),
        "re_gap_mm": None if recontact is None else round(1000.0 * recontact["gap"], 1),
        "re_fn_pad": None if recontact is None else round(recontact["fn_pad"], 2),
        "re_fn_slope": None if recontact is None else round(recontact["fn_slope"], 2),
        "min_cz": round(min(row["cz"] for row in rows), 4),
        "max_hand_drop": round(max(rows[0]["p_hand"][2] - row["p_hand"][2] for row in rows), 4),
        "max_down_speed": round(min(row["v_hand"][2] for row in rows), 3),
        "gap_max_mm": round(1000.0 * max(row["gap"] for row in rows), 1),
        "gap_min_mm": round(1000.0 * min(row["gap"] for row in rows), 1),
        "min_bottom": round(min(row["bottom"] for row in rows), 4),
        "min_fz_n": round(min(row["fz_n"] for row in rows), 2),
        "peak_fn": round(max(row["fn"] for row in rows), 2),
        "peak_class": max(rows, key=lambda row: row["fn"])["dominant"],
        "table": any(row["table"] for row in rows),
        "end_mode": rows[-1]["mode"],
        "end_bottom": round(rows[-1]["bottom"], 4),
        "end_cz": round(rows[-1]["cz"], 4),
        "end_gap_mm": round(1000.0 * rows[-1]["gap"], 1),
        "end_vz": round(rows[-1]["v_obj"][2], 3),
        "lifted": rows[-1]["bottom"] > min(row["bottom"] for row in rows) + 0.03 and rows[-1]["v_obj"][2] > -0.02,
        "sat": any(row["sat"] for row in rows),
    }


def rollout(sim, snap, choose, horizon, after_restore=None):
    restore_ballistic(sim, snap)
    if after_restore is not None:
        after_restore()
    restore_state(sim, None)
    t0 = float(sim.data.time)
    rows = []
    actions = []
    contacts = []
    observations = []
    hist = ObservableObsState(0.02)
    handoff_obs = None
    state = {"mode": "open"}
    while float(sim.data.time) - t0 < horizon - 1e-9:
        pose = _pose(sim)
        raw = contacts_now(sim)
        forces, sides = _class_force(raw)
        band = _pad_band(sim)
        pose["t"] = float(sim.data.time) - t0
        pose["cz"] = float(capture_geometry(sim)["center_hand_m"][2])
        pose["pad_lo"], pose["pad_hi"] = band
        pose["fn_pad"] = forces["LOWER_VERTICAL_PAD"]
        pose["fn_slope"] = forces["UPPER_SLOPED_PART"]
        pose["fn_edge"] = forces["EDGE_CONTACT"]
        pose["pad_sides"] = sides["LOWER_VERTICAL_PAD"]
        pose["slope_sides"] = sides["UPPER_SLOPED_PART"]
        pose["dominant"] = max(forces, key=forces.get) if sum(forces.values()) > 1e-6 else "NONE"
        pose["fn"] = float(sum(item["fn"] for item in raw))
        pose["fz_n"] = float(sum(item["fz_n"] for item in raw))
        peak, sat = _wrist_sat(sim)
        pose["tau_peak"] = peak
        pose["sat"] = sat
        pose["table"] = bool(latched(sim) or pose["bottom"] < TABLE_TOP)
        decision = choose(pose, raw, state)
        pose["mode"] = state["mode"]
        pose["cmd_z"] = float(decision["v"][2])
        pose["cmd_tau"] = float(decision["tau"])
        rows.append(pose)
        for item in raw:
            item = dict(item)
            item["t"] = pose["t"]
            item["cz"] = pose["cz"]
            item["gap"] = pose["gap"]
            item["mode"] = state["mode"]
            contacts.append(item)
        if pose["table"] or decision.get("stop"):
            handoff_obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            break
        observations.append(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist))
        action = _action_from(sim, decision["v"], np.zeros(3), decision["tau"])
        step_legal(sim, action)
        actions.append(np.asarray(action, np.float32))
    return {
        "rows": rows,
        "actions": actions,
        "contacts": contacts,
        "observations": observations,
        "handoff_obs": handoff_obs,
        "metrics": _metrics(rows) if rows else {},
    }


def staged_choose(spec):
    def choose(pose, _raw, state):
        both = pose["nL"] > 0 and pose["nR"] > 0
        pad = pose["pad_sides"] == "LR" and pose["fn_pad"] >= pose["fn_slope"]
        mode = state["mode"]
        xy = np.array([
            float(np.clip(pose["v_obj"][0], -0.25, 0.25)),
            float(np.clip(pose["v_obj"][1], -0.25, 0.25)),
        ])
        tau = spec["arrest"]
        vz = 0.0
        if mode == "open":
            tau = TAU_OPEN if pose["gap"] < spec["open_gap"] else spec["arrest"]
            vz = -spec["down"]
            if pose["gap"] >= spec["open_gap"]:
                mode = "chase"
        if mode == "chase":
            tau = spec["arrest"] if pose["gap"] >= spec["open_gap"] else TAU_OPEN
            vz = -spec["down"]
            deep_enough = pose["cz"] <= spec["stop_cz"]
            if deep_enough and pose["gap"] >= spec["close_gap"]:
                mode = "close"
            elif deep_enough:
                mode = "hold"
            elif pose["t"] > spec["chase_limit"]:
                mode = "fail"
        if mode == "hold":
            tau = TAU_OPEN if pose["gap"] < spec["open_gap"] else spec["arrest"]
            vz = _depth_vz(pose, spec)
            if pose["gap"] >= spec["close_gap"]:
                mode = "close"
        if mode == "close":
            tau = spec["hold_tau"] if pad else spec["close_tau"]
            vz = _depth_vz(pose, spec) if not spec["close_dive"] else float(min(pose["v_obj"][2] - 0.30, -0.20))
            state["pad"] = state.get("pad", 0) + 1 if pad else 0
            if state["pad"] >= spec["pad_steps"]:
                mode = "brake"
        elif mode == "brake":
            tau = spec["hold_tau"] if pad else spec["close_tau"]
            speed = float(pose["v_obj"][2])
            vz = 0.0 if speed >= -0.02 else min(0.0, speed + spec["brake"] * 0.02)
            vz = _blend_depth(vz, pose, spec)
            if not pad:
                state["pad"] = 0
                mode = "close"
            elif speed > -0.05 and pose["v_hand"][2] > -0.10 and state.get("pad", 0) >= spec["pad_steps"]:
                mode = "lift"
        elif mode == "lift":
            tau = spec["hold_tau"]
            risen = pose["bottom"] > float(spec.get("settle_bottom", 0.56)) and state.get("pad", 0) >= spec["pad_steps"]
            vz = 0.0 if risen else spec["lift"]
            if risen:
                mode = "settle"
        elif mode == "settle":
            tau = spec["hold_tau"]
            vz = 0.0
            quiet = pose["pad_sides"] == "LR" and abs(float(pose["v_obj"][2])) < 0.06 and pose["bottom"] > 0.52
            state["quiet"] = state.get("quiet", 0) + 1 if quiet else 0
            if state["quiet"] >= 8:
                mode = "handoff"
        elif mode == "fail":
            tau = spec["arrest"]
            vz = 0.0
        state["mode"] = mode
        v = np.clip(np.array([xy[0], xy[1], vz]), [-1.0, -1.0, -3.8], [1.0, 1.0, 1.0])
        return {"v": v, "tau": float(tau), "stop": mode in ("fail", "handoff")}

    return choose


def _depth_vz(pose, spec):
    """Match the cylinder, and stop driving the hand once the pad band is reached."""
    vz = float(np.clip(pose["v_obj"][2], -1.0, 0.4))
    if pose["cz"] < spec["stop_cz"] - 0.003:
        vz = max(vz, 0.05)
    elif pose["cz"] > spec["stop_cz"] + 0.004:
        vz = min(vz, -spec["down"])
    return vz


def _blend_depth(vz, pose, spec):
    if pose["cz"] < spec["stop_cz"] - 0.006:
        return max(vz, 0.0)
    return vz


def _base_spec():
    return {
        "open_gap": 0.044,
        "arrest": 0.0,
        "close_gap": 0.042,
        "down": 2.2,
        "stop_cz": 0.100,
        "close_dive": False,
        "close_tau": -8.0,
        "hold_tau": -18.0,
        "pad_steps": 2,
        "brake": 6.0,
        "lift": 0.08,
        "chase_limit": 0.35,
    }
