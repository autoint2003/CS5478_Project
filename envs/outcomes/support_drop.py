"""Scene-aware drop latch facade.

Elevated-table Recovery keeps ``table_drop`` / ``TABLE_DROP``.
``GROUND_SCENE`` uses ``ground_drop`` / ``GROUND_DROP``.
"""

from __future__ import annotations

from envs.simulation.ground_scene import SCENE_ID as GROUND_SCENE_ID


def scene_id_of(sim) -> str:
    return str(getattr(sim, "scene_id", "") or "")


def is_ground_scene(sim) -> bool:
    return scene_id_of(sim) == GROUND_SCENE_ID


def object_vz(sim):
    if is_ground_scene(sim):
        from envs.outcomes.ground_drop import object_vz as gz

        return gz(sim)
    from envs.outcomes.table_drop import object_vz as tz

    return tz(sim)


def observe(sim, vz_before):
    if is_ground_scene(sim):
        from envs.outcomes.ground_drop import observe as go

        return go(sim, vz_before)
    from envs.outcomes.table_drop import observe as to

    return to(sim, vz_before)


def latched(sim) -> bool:
    if is_ground_scene(sim):
        from envs.outcomes.ground_drop import latched as gl

        return gl(sim)
    from envs.outcomes.table_drop import latched as tl

    return tl(sim)


def terminal_reason(sim, default="DROP_DURING_RECOVERY"):
    if is_ground_scene(sim):
        from envs.outcomes.ground_drop import terminal_reason as gt

        return gt(sim, default)
    from envs.outcomes.table_drop import terminal_reason as tt

    return tt(sim, default)


def export_state(sim):
    if is_ground_scene(sim):
        from envs.outcomes.ground_drop import export_state as ge

        return {"kind": "ground_drop", "state": ge(sim)}
    from envs.outcomes.table_drop import export_state as te

    return {"kind": "table_drop", "state": te(sim)}


def restore_state(sim, blob):
    if not blob:
        if is_ground_scene(sim):
            from envs.outcomes.ground_drop import restore_state as gr

            gr(sim, None)
        else:
            from envs.outcomes.table_drop import restore_state as tr

            tr(sim, None)
        return
    if isinstance(blob, dict) and "kind" in blob:
        kind = blob["kind"]
        state = blob.get("state")
        if kind == "ground_drop":
            from envs.outcomes.ground_drop import restore_state as gr

            gr(sim, state)
            return
        from envs.outcomes.table_drop import restore_state as tr

        tr(sim, state)
        return
    # Legacy captures stored the table_drop dict directly.
    if is_ground_scene(sim):
        from envs.outcomes.ground_drop import restore_state as gr

        gr(sim, blob)
    else:
        from envs.outcomes.table_drop import restore_state as tr

        tr(sim, blob)


def arm_recovery_drop(sim) -> None:
    """Call at Recovery start. No-op for elevated-table (always monitored in step)."""
    if is_ground_scene(sim):
        from envs.outcomes.ground_drop import arm

        arm(sim)
