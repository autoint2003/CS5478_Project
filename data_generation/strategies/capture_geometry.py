"""Privileged capture geometry and a lead-intercept action.

Diagnosis only. The returned command is a legal recovery7d_airborne_v3
action. This module does not change the action map, the gripper torque
limits, or any mechanical parameter.
"""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.recovery_actions import AIRBORNE_V3, TAU_OPEN, TAU_SECURE, grip_tau, normalize_action
from envs.deterioration import body_twist
from envs.physical_recovery import physical_pack

CYLINDER_RADIUS = 0.018
CYLINDER_DIAMETER = 0.036
# Pad center in the hand frame: finger body at z=0.0584 plus pad local z=0.0445.
PINCH_HAND = np.array([0.0, 0.0, 0.1029], float)
ENTRY_GAP = CYLINDER_DIAMETER + 0.006


def _box_corners(model, data, geom_id):
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        return []
    center = np.array(data.geom_xpos[geom_id], float)
    rotation = np.array(data.geom_xmat[geom_id].reshape(3, 3), float)
    half = np.array(model.geom_size[geom_id], float)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(center + rotation @ (np.array([sx, sy, sz]) * half))
    return corners


def _finger_pad_corners(sim, body_id):
    model, data = sim.model, sim.data
    corners = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != int(body_id):
            continue
        corners.extend(_box_corners(model, data, geom_id))
    return corners


def capture_geometry(sim) -> dict:
    """Inner-face capture test. Hand–COM distance is not a grasp."""
    pack = physical_pack(sim)
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    rotation = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    opening_axis = rotation[:, 1]
    finger_axis = rotation[:, 2]
    lateral_axis = rotation[:, 0]
    obj = np.array(sim.data.xpos[sim.ids.object_body], float)
    obj_rot = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    axis = obj_rot[:, 2]
    left = _finger_pad_corners(sim, sim.ids.left_body)
    right = _finger_pad_corners(sim, sim.ids.right_body)
    if not left or not right:
        raise RuntimeError("finger pad boxes were not found")
    left_y = np.array([float(np.dot(p - hand, opening_axis)) for p in left])
    right_y = np.array([float(np.dot(p - hand, opening_axis)) for p in right])
    left_z = np.array([float(np.dot(p - hand, finger_axis)) for p in left])
    right_z = np.array([float(np.dot(p - hand, finger_axis)) for p in right])
    left_x = np.array([float(np.dot(p - hand, lateral_axis)) for p in left])
    right_x = np.array([float(np.dot(p - hand, lateral_axis)) for p in right])
    if float(np.mean(left_y)) >= float(np.mean(right_y)):
        positive_y, negative_y = left_y, right_y
        positive_z, negative_z = left_z, right_z
        positive_x, negative_x = left_x, right_x
    else:
        positive_y, negative_y = right_y, left_y
        positive_z, negative_z = right_z, left_z
        positive_x, negative_x = right_x, left_x
    inner_positive = float(np.min(positive_y))
    inner_negative = float(np.max(negative_y))
    gap = inner_positive - inner_negative
    center = obj - hand
    cy = float(np.dot(center, opening_axis))
    cz = float(np.dot(center, finger_axis))
    cx = float(np.dot(center, lateral_axis))
    dist_positive = inner_positive - cy
    dist_negative = cy - inner_negative
    between = dist_positive > 0.0 and dist_negative > 0.0
    pad_z = np.concatenate([positive_z, negative_z])
    pad_x = np.concatenate([positive_x, negative_x])
    z_lo, z_hi = float(np.min(pad_z)), float(np.max(pad_z))
    x_lo, x_hi = float(np.min(pad_x)), float(np.max(pad_x))
    # A vertical cylinder extends along the finger axis. Count it as axially
    # graspable when its body overlaps the pad span, not when only the COM does.
    axial_overlap = min(cz + CYLINDER_RADIUS, z_hi) - max(cz - CYLINDER_RADIUS, z_lo)
    lateral_overlap = min(cx + CYLINDER_RADIUS, x_hi) - max(cx - CYLINDER_RADIUS, x_lo)
    opening_alignment = abs(float(np.dot(axis, opening_axis)))
    orientation_ok = opening_alignment < 0.65
    clearance_positive = dist_positive - CYLINDER_RADIUS
    clearance_negative = dist_negative - CYLINDER_RADIUS
    # Closing while the barrel is outside either inner plane pushes it away.
    outside_and_close = (not between) and min(abs(dist_positive), abs(dist_negative)) < CYLINDER_RADIUS + 0.008
    off_pad = axial_overlap < 0.004 or lateral_overlap < 0.0
    v_rel = np.asarray(pack["v_rel_h"], float)
    outward = 0.0
    if dist_positive < dist_negative:
        outward = max(0.0, float(v_rel[1]))
    else:
        outward = max(0.0, -float(v_rel[1]))
    would_eject = bool(outside_and_close or off_pad or outward > 0.35 or not orientation_ok)
    ready = bool(
        between
        and orientation_ok
        and axial_overlap >= 0.004
        and lateral_overlap >= 0.0
        and gap >= CYLINDER_DIAMETER - 0.004
        and not would_eject
        and abs(float(v_rel[1])) < 0.30
    )
    qadr = [int(sim.model.jnt_qposadr[j]) for j in sim.ids.finger_jnt]
    qpos = [float(sim.data.qpos[a]) for a in qadr]
    return {
        "inner_gap_m": gap,
        "finger_qpos_m": qpos,
        "between": between,
        "dist_positive_m": dist_positive,
        "dist_negative_m": dist_negative,
        "clearance_positive_m": clearance_positive,
        "clearance_negative_m": clearance_negative,
        "center_hand_m": [cx, cy, cz],
        "pad_z_m": [z_lo, z_hi],
        "pad_x_m": [x_lo, x_hi],
        "axial_overlap_m": float(axial_overlap),
        "lateral_overlap_m": float(lateral_overlap),
        "axis_opening_alignment": opening_alignment,
        "orientation_ok": orientation_ok,
        "relative_velocity_hand_mps": v_rel.tolist(),
        "would_eject_if_closed": would_eject,
        "ready_to_close": ready,
        "gap_fits_36mm": gap >= CYLINDER_DIAMETER,
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "grip_ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else 0.0,
        "obj_z": float(obj[2]),
    }


def _object_velocity(sim):
    linear, angular = body_twist(sim.model, sim.data, sim.ids.object_body)
    return np.array(linear, float), np.array(angular, float)


def _predict(sim, horizon):
    """Future cylinder position. Contact blocks the free-fall assumption."""
    pack = physical_pack(sim)
    pos = np.array(sim.data.xpos[sim.ids.object_body], float)
    vel, _angular = _object_velocity(sim)
    touching = int(pack["nL"]) + int(pack["nR"]) > 0
    gravity = np.array(sim.model.opt.gravity, float)
    if touching:
        return pos + vel * horizon, vel.copy(), True
    return pos + vel * horizon + 0.5 * gravity * horizon * horizon, vel + gravity * horizon, False


def choose_grip_tau(geo: dict, want_close: bool) -> float:
    """Open with the legal +2 N·m command. Do not slam an empty jaw shut.

    A 20 ms pulse of the secure torque (-18 N·m) drives the finger slides
    through the q=0 stop. The meshes then lock, and the legal open torque
    cannot separate them. Full close is used only while both pads are on
    the cylinder, which stops the slides before that inversion.
    """
    qmin = min(float(v) for v in geo["finger_qpos_m"])
    held = bool(geo["nL"] > 0 and geo["nR"] > 0 and geo["between"] and qmin > 0.008)
    if qmin < 0.0:
        return TAU_OPEN
    if held:
        return TAU_SECURE
    if not want_close:
        return TAU_OPEN
    # Empty slides still have enough speed at q=0.02 to cross the stop.
    # Zero torque lets them coast. The cylinder, once between the pads,
    # is what stops a stronger close.
    if qmin < 0.018 and not geo["between"]:
        return grip_tau(-0.8)
    if geo["ready_to_close"] or geo["between"]:
        return grip_tau(-0.70)
    return TAU_OPEN


def lead_action(sim, mode: str, lead_m: float) -> tuple[np.ndarray, dict]:
    """Move the pinch ahead of the cylinder and close only inside the pads.

    mode is intercept, sync, or open_first. lead_m is the distance, along the
    cylinder motion, by which the pinch stays ahead of the current COM.
    """
    geo = capture_geometry(sim)
    hand_rot = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    pinch = hand + hand_rot @ PINCH_HAND
    obj = np.array(sim.data.xpos[sim.ids.object_body], float)
    obj_vel, _angular = _object_velocity(sim)
    touching = geo["nL"] + geo["nR"] > 0
    gravity = np.array(sim.model.opt.gravity, float)
    speed = float(np.linalg.norm(obj_vel))
    if touching or speed < 0.05:
        direction = np.array([0.0, 0.0, -1.0]) if speed < 0.05 else obj_vel / speed
        predicted = obj_vel.copy() if touching else obj_vel + gravity * 0.04
    else:
        direction = obj_vel / speed
        predicted = obj_vel + gravity * 0.04
    if float(np.dot(direction, np.array([0.0, 0.0, -1.0]))) < 0.0 and not touching:
        direction = -direction
    future, _future_vel, _contact_model = _predict(sim, 0.08)
    # Aim at the predicted COM, then keep the pinch lead_m further along the
    # motion so the jaws are already past the cylinder.
    desired = future + float(lead_m) * direction
    error = desired - pinch
    gain = 6.0 if mode == "sync" else 8.0
    limit = 0.35 if mode == "sync" else 0.80
    # Match the cylinder's current speed. The position term only trims the lead.
    # Do not replace this with a future speed, or the hand sprints away at release.
    command = obj_vel + np.clip(gain * error, -limit, limit)
    grasped = geo["nL"] > 0 and geo["nR"] > 0 and geo["between"]
    separation = float(np.dot(obj - pinch, direction))
    near = geo["between"] or 0.0 <= separation <= 0.05
    if near and not grasped:
        along_command = float(np.dot(command, direction))
        along_object = float(np.dot(obj_vel, direction))
        if along_command < along_object:
            command = command + direction * (along_object - along_command)
    if grasped:
        command = obj_vel + np.clip(3.0 * (obj - pinch), -0.12, 0.12)
    want_close = bool(geo["ready_to_close"] or grasped)
    if mode == "open_first" and geo["inner_gap_m"] < ENTRY_GAP:
        want_close = False
    tau = choose_grip_tau(geo, want_close)
    action = normalize_action(hand_rot.T @ command, np.zeros(3), tau, AIRBORNE_V3)
    geo["mode"] = mode
    geo["lead_m"] = float(lead_m)
    geo["pinch_z"] = float(pinch[2])
    geo["desired_z"] = float(desired[2])
    geo["command_world_mps"] = command.tolist()
    geo["predicted_world_mps"] = predicted.tolist()
    geo["contact_motion_model"] = "contact_hold" if touching else "ballistic"
    return action, geo
