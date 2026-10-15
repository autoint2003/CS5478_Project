"""Seated grasp construction used by training and data generation.

seat_horizontal places the horizontal cylinder on the table. grasp_settled
runs the nominal controller until the lift is stable. apply_offset and
apply_spec are the seated perturbations. The offset grid search stays in the
horizontal freefall experiment.
"""

from __future__ import annotations

import mujoco
import numpy as np

from envs.ballistic_impact import HAND_RISE_M, TAU_SEC, z_tgt_of
from envs.simulation.grasp_quality import body_twist
from envs.outcomes.physical_recovery import Z_AIR, physical_pack
from training.recovery_execution import restore_dyn, stamp
from controllers.cartesian.velocity_step import tick_vw
from envs.simulation.ballistic_scene import CYL_R

TABLE = 0.40
DT = 0.002


def nominal_physics_step(sim):
    sim.physics_step(None, in_recovery=False)
    sim.maybe_capture_reference()


def restore_seated(sim, snap, nominal, pad_ids, pad_mu):
    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, None, None)
    sim.data.xfrc_applied[:] = 0.0
    return info


def seat_horizontal(sim):
    q = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    d = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    sim.data.qpos[q:q + 3] = np.array([0.50, 0.0, TABLE + CYL_R + 0.001])
    s = np.sqrt(0.5)
    sim.data.qpos[q + 3:q + 7] = np.array([s, s, 0.0, 0.0])
    sim.data.qvel[d:d + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def axis_of(sim):
    axis = np.array(sim.data.xmat[sim.ids.object_body], float).reshape(3, 3)[:, 2]
    n = float(np.linalg.norm(axis))
    return axis / max(n, 1e-12)


def hold_step(sim, gains):
    z_now = float(z_tgt_of(sim))
    lift_hold = str(sim.fsm.phase) == "lift" and float(sim.fsm.p_des[2]) >= z_now - 1e-9
    if lift_hold:
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
    else:
        nominal_physics_step(sim)


def grasp_settled(sim, gains):
    sim.model.opt.timestep = DT
    sim.reset()
    seat_horizontal(sim)
    hand_z0 = None
    risen_at = None
    for _ in range(int(8.0 / DT)):
        if risen_at is not None and float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-4:
            hold_step(sim, gains)
        else:
            nominal_physics_step(sim)
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if sim.fsm.phase == "lift" and hand_z0 is None:
            hand_z0 = hz
        o = physical_pack(sim)
        risen = (
            hand_z0 is not None
            and (hz - hand_z0) >= HAND_RISE_M + 0.03
            and int(o["nL"]) > 0
            and int(o["nR"]) > 0
            and float(o["obj_z"]) >= Z_AIR
            and float(sim.data.ctrl[7]) <= -17.0
        )
        if risen and risen_at is None:
            risen_at = float(sim.data.time)
        vz = float(body_twist(sim.model, sim.data, sim.ids.object_body)[0][2])
        if risen_at is not None and float(sim.data.time) >= risen_at + 0.30 and abs(vz) < 0.02:
            return stamp(sim)
    return None


def apply_offset(sim, spec):
    q = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    sim.data.qpos[q] += float(spec["dx"]) * 1e-3
    sim.data.qpos[q + 1] += float(spec["dy"]) * 1e-3
    sim.data.qpos[q + 2] += float(spec["dz"]) * 1e-3
    pitch = float(spec.get("pitch", 0.0))
    if abs(pitch) > 1e-6:
        ang = np.radians(pitch) * 0.5
        extra = np.array([np.cos(ang), np.sin(ang), 0.0, 0.0])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, extra, np.array(sim.data.qpos[q + 3:q + 7], float))
        sim.data.qpos[q + 3:q + 7] = out
    mujoco.mj_forward(sim.model, sim.data)

def begin(sim, snap, nominal, pads, pad_mu, spec):
    info = restore_seated(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = DT
    if not info.get("pose_ok", False):
        return False
    apply_offset(sim, spec)
    return True

def apply_spec(sim, spec):
    apply_offset(sim, {
        "dx": spec.get("dx", 0.0),
        "dy": spec.get("dy", 0.0),
        "dz": spec["dz"],
        "pitch": spec.get("pitch", 0.0),
    })
    if spec.get("dq2"):
        j = int(sim.ids.arm_jnt[1])
        sim.data.qpos[int(sim.model.jnt_qposadr[j])] += float(spec["dq2"])
    if spec.get("dq4"):
        j = int(sim.ids.arm_jnt[3])
        sim.data.qpos[int(sim.model.jnt_qposadr[j])] += float(spec["dq4"])
    if spec.get("df"):
        for j in sim.ids.finger_jnt:
            sim.data.qpos[int(sim.model.jnt_qposadr[int(j)])] += float(spec["df"])
    if spec.get("dvz"):
        dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
        sim.data.qvel[dadr + 2] += float(spec["dvz"])
    mujoco.mj_forward(sim.model, sim.data)

def target_hand(sim, pinch):
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    return po - Rd @ np.asarray(pinch, float)
