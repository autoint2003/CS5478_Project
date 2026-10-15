"""Save and restore floor/table contact masks.

These masks are part of a seated-grasp snapshot. They are not a success label.
"""

from __future__ import annotations

import mujoco


def supports_of(sim):
    saved = []
    for name in ("floor", "table"):
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        saved.append((int(gid), int(sim.model.geom_contype[gid]), int(sim.model.geom_conaffinity[gid])))
    return saved


def restore_supports(sim, saved):
    for gid, contype, conaffinity in saved:
        sim.model.geom_contype[gid] = contype
        sim.model.geom_conaffinity[gid] = conaffinity
