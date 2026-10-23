"""Save and restore support-plane contact masks.

These masks are part of a seated-grasp snapshot. They are not a success label.
Elevated scenes use floor/table; GROUND_SCENE uses ground.
"""

from __future__ import annotations

import mujoco

SUPPORT_GEOM_NAMES = ("floor", "table", "ground")


def supports_of(sim):
    saved = []
    for name in SUPPORT_GEOM_NAMES:
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        saved.append((int(gid), int(sim.model.geom_contype[gid]), int(sim.model.geom_conaffinity[gid])))
    return saved


def restore_supports(sim, saved):
    """Restore by geom id only when the id still names a support plane on this model."""
    if not saved:
        return
    support_ids = {
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in SUPPORT_GEOM_NAMES
    }
    support_ids.discard(-1)
    for gid, contype, conaffinity in saved:
        gid = int(gid)
        if gid not in support_ids:
            # Foreign-scene geom ids (e.g. elevated floor/table → GROUND_SCENE).
            continue
        if gid < 0 or gid >= int(sim.model.ngeom):
            continue
        sim.model.geom_contype[gid] = int(contype)
        sim.model.geom_conaffinity[gid] = int(conaffinity)