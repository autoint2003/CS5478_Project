"""Latched irreversible GROUND_DROP for GROUND_SCENE.

Semantics
---------
- Event name: ``GROUND_DROP`` (not a renamed TABLE_DROP string).
- Physical trigger: cylinder contact force against the ``ground`` plane geom,
  using the same bounce / sustained-support logic as the elevated-table latch.
- Arming: the latch is evaluated only while ``armed(sim)`` is true. Scene
  setup may rest the cylinder on the ground without counting as failure.
  Arm at Recovery start (and keep armed through post-handoff nominal verify).
- Bilateral finger grasp suppresses a new latch while held; a later regrasp
  does **not** clear an already latched GROUND_DROP.
- Geometric height guard ``GROUND_DROP_Z`` (see ``ground_scene``) is a secondary
  abort used by horizon logic; it is not the old table-height threshold.

The latch is Python episode state, copied with the recovery snapshot under
key ``ground_drop``.
"""

from __future__ import annotations

import mujoco
import numpy as np

from envs.simulation.ground_scene import GROUND_DROP_Z, GROUND_Z, SCENE_ID

# Same numerical landing criteria as table_drop (impulse / support timing).
SUPPORT_FRACTION = 0.5
SUPPORT_TIME_S = 0.02
QUIET_RESET_S = 0.02
BOUNCE_DV = 0.15


def _fresh():
    return {
        "latched": False,
        "armed": False,
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
        "scene_id": SCENE_ID,
        "ground_z": GROUND_Z,
        "ground_drop_z": GROUND_DROP_Z,
    }


def state_of(sim):
    state = getattr(sim, "_ground_drop", None)
    if state is None:
        state = _fresh()
        sim._ground_drop = state
    return state


def export_state(sim):
    return dict(state_of(sim))


def restore_state(sim, blob):
    if not blob:
        sim._ground_drop = _fresh()
        return
    state = _fresh()
    state.update(blob)
    sim._ground_drop = state


def arm(sim) -> None:
    """Enable GROUND_DROP monitoring (Recovery / post-handoff)."""
    state_of(sim)["armed"] = True


def disarm(sim) -> None:
    """Disable monitoring (scene construction / pre-Recovery). Does not clear latch."""
    state_of(sim)["armed"] = False


def armed(sim) -> bool:
    return bool(state_of(sim).get("armed", False))


def latched(sim) -> bool:
    return bool(state_of(sim)["latched"])


def terminal_reason(sim, default="DROP_DURING_RECOVERY"):
    return "GROUND_DROP" if latched(sim) else default


def object_vz(sim):
    return float(sim.data.cvel[sim.ids.object_body][5])


def height_guard_triggered(sim) -> bool:
    """Secondary geometric abort: COM below ground-relative guard while armed."""
    if not armed(sim):
        return False
    z = float(sim.data.xpos[sim.ids.object_body][2])
    return z < float(GROUND_DROP_Z)


def _landing_force(sim):
    """Upward Newtons from the ground plane onto the cylinder, plus finger counts."""
    model, data, ids = sim.model, sim.data, sim.ids
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground < 0:
        # Accept legacy name if a variant XML kept "floor".
        ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
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
        if other_geom != ground:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = np.array(contact.frame[:3], float)
        sign = -1.0 if body1 == ids.object_body else 1.0
        upward += max(0.0, sign * float(wrench[0]) * float(normal[2]))
    return n_l, n_r, float(upward)


def consider(state, dt, time, z, bilateral, upward, vz_before, vz_after, weight):
    if state["latched"]:
        return state
    if not state.get("armed", False):
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
    """Update from the physics step that just ran. No-op while disarmed."""
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
