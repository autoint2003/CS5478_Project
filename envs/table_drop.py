"""Latched post-impact table/floor landing.

The cylinder starts on the table during the initial grasp. This monitor is
called only from the recovery step and the post-handoff nominal step, so that
pre-impact contact is not a failure.

A bilateral finger grasp suppresses the event. A later regrasp does not clear
it. The latch is Python episode state, not an ``mjData`` field, and is copied
with the recovery snapshot.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Half the cylinder weight. One 2 ms step at this force is the smallest
# impulse accepted as a landing collision. Ten such steps (0.02 s) count as
# support when the landing does not reverse vertical speed, for example a
# cylinder that is already nearly at rest. This is far shorter than the old
# 1 s no-contact timer, so a table regrasp cannot finish before the latch.
SUPPORT_FRACTION = 0.5
SUPPORT_TIME_S = 0.02
QUIET_RESET_S = 0.02
# Upward speed change from the velocity before this contact episode.
# ball_train_v2_0051 reversed by about 1.5 m/s; 0.15 m/s is above a
# numerical contact and still fires on the impact step of a real bounce.
BOUNCE_DV = 0.15


def _fresh():
    return {
        "latched": False,
        "t": None,
        "kind": None,
        "support_s": 0.0,
        "quiet_s": 0.0,
        "in_contact": False,
        "vz_onset": 0.0,
        "impulse": 0.0,
        "dv": 0.0,
        "peak_up": 0.0,
        "z": None,
    }


def state_of(sim):
    state = getattr(sim, "_table_drop", None)
    if state is None:
        state = _fresh()
        sim._table_drop = state
    return state


def export_state(sim):
    return dict(state_of(sim))


def restore_state(sim, blob):
    """Install a private copy. A missing blob is a clean episode."""
    if not blob:
        sim._table_drop = _fresh()
        return
    state = _fresh()
    state.update(blob)
    sim._table_drop = state


def latched(sim):
    return bool(state_of(sim)["latched"])


def terminal_reason(sim, default="DROP_DURING_RECOVERY"):
    return "TABLE_DROP" if latched(sim) else default


def object_vz(sim):
    return float(sim.data.cvel[sim.ids.object_body][5])


def _landing_force(sim):
    """Upward Newtons from table or floor onto the cylinder, plus finger counts."""
    model, data, ids = sim.model, sim.data, sim.ids
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    n_l = 0
    n_r = 0
    upward = 0.0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if ids.object_body not in (body1, body2):
            continue
        other_body = body2 if body1 == ids.object_body else body1
        other_geom = geom2 if body1 == ids.object_body else geom1
        if other_body == ids.left_body:
            n_l += 1
        elif other_body == ids.right_body:
            n_r += 1
        if other_geom not in (table, floor):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = np.array(contact.frame[:3], float)
        # The contact normal points from geom1 toward geom2. Compression
        # pushes geom2 along +normal and geom1 along -normal.
        sign = -1.0 if body1 == ids.object_body else 1.0
        upward += max(0.0, sign * float(wrench[0]) * float(normal[2]))
    return n_l, n_r, float(upward)


def consider(state, dt, time, z, bilateral, upward, vz_before, vz_after, weight):
    """Pure update. ``state`` is private to one episode."""
    if state["latched"]:
        return state
    if bilateral:
        state["in_contact"] = False
        state["support_s"] = 0.0
        state["quiet_s"] = 0.0
        state["impulse"] = 0.0
        state["dv"] = 0.0
        return state
    threshold = SUPPORT_FRACTION * float(weight)
    if upward >= threshold:
        state["support_s"] = float(state["support_s"]) + float(dt)
        state["quiet_s"] = 0.0
    else:
        state["quiet_s"] = float(state["quiet_s"]) + float(dt)
        if state["quiet_s"] >= QUIET_RESET_S:
            state["support_s"] = 0.0
    if upward > 1e-3:
        if not state["in_contact"]:
            state["in_contact"] = True
            state["vz_onset"] = float(vz_before)
            state["impulse"] = 0.0
        state["impulse"] = float(state["impulse"]) + float(upward) * float(dt)
        state["peak_up"] = max(float(state["peak_up"]), float(upward))
        state["dv"] = float(vz_after) - float(state["vz_onset"])
        if state["dv"] >= BOUNCE_DV and state["impulse"] >= threshold * float(dt):
            _latch(state, "bounce", time, z)
            return state
    else:
        state["in_contact"] = False
    if float(state["support_s"]) >= SUPPORT_TIME_S:
        _latch(state, "support", time, z)
    return state


def observe(sim, vz_before):
    """Update from the physics step that just ran. ``vz_before`` is pre-step."""
    n_l, n_r, upward = _landing_force(sim)
    weight = float(sim.model.body_mass[sim.ids.object_body]) * float(abs(sim.model.opt.gravity[2]))
    return consider(
        state_of(sim),
        float(sim.model.opt.timestep),
        float(sim.data.time),
        float(sim.data.xpos[sim.ids.object_body][2]),
        n_l > 0 and n_r > 0,
        upward,
        vz_before,
        object_vz(sim),
        weight,
    )


def _latch(state, kind, time, z):
    state["latched"] = True
    state["kind"] = kind
    state["t"] = float(time)
    state["z"] = float(z)
