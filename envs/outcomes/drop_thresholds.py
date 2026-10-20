"""Scene-aware geometric drop thresholds.

Elevated table keeps the historical ``TABLE_DROP = 0.44`` height gate.
``GROUND_SCENE`` uses ``GROUND_DROP_Z`` (cylinder rest height + margin), not a
renamed table top. Contact latch (``GROUND_DROP`` / ``TABLE_DROP``) remains the
primary irreversible failure; the height gate is secondary.
"""

from __future__ import annotations

from envs.outcomes.physical_recovery import TABLE_DROP
from envs.outcomes.support_drop import is_ground_scene
from envs.simulation.ground_scene import GROUND_DROP_Z


def geometric_drop_z(sim) -> float:
    if is_ground_scene(sim):
        return float(GROUND_DROP_Z)
    return float(TABLE_DROP)


def below_geometric_drop(sim, z: float | None = None) -> bool:
    if z is None:
        z = float(sim.data.xpos[sim.ids.object_body][2])
    return float(z) < geometric_drop_z(sim)
