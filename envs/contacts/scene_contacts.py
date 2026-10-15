"""Object, table, and ball contact rows used by replay and observation packing."""

from __future__ import annotations

import mujoco
import numpy as np

from envs.simulation.scene_state import TABLE_TOP


def geom_name(model, gid: int) -> str:
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return str(nm) if nm else f"geom{gid}"


def body_name(model, bid: int) -> str:
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
    return str(nm) if nm else f"body{bid}"


def cylinder_min_clearance(sim) -> float:
    gid = int(sim.ids.object_geom)
    r = float(sim.model.geom_size[gid][0])
    hh = float(sim.model.geom_size[gid][1])
    c = np.array(sim.data.geom_xpos[gid], dtype=float)
    axis = np.array(sim.data.geom_xmat[gid].reshape(3, 3)[:, 2], dtype=float)
    sxy = float(np.sqrt(max(0.0, 1.0 - axis[2] ** 2)))
    zs = [float((c + s * hh * axis)[2] - r * sxy) for s in (-1.0, 1.0)]
    return min(zs) - TABLE_TOP


def scene_contacts(sim) -> list[dict]:
    ids = sim.ids
    table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    floor = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    rows = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        pair = {g1, g2}
        wr = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, i, wr)
        kind = None
        if ids.object_geom in pair and table in pair:
            kind = "object_table"
        elif ids.object_geom in pair and floor in pair:
            kind = "object_floor"
        if kind is None:
            continue
        rows.append({"kind": kind, "g1": geom_name(sim.model, g1), "g2": geom_name(sim.model, g2), "fn": abs(float(wr[0]))})
    return rows


def disable_object_table(sim) -> dict:
    table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    obj = int(sim.ids.object_geom)
    fingers = [g for g in range(sim.model.ngeom)
               if int(sim.model.geom_bodyid[g]) in (sim.ids.left_body, sim.ids.right_body)]
    before = {
        "obj": (int(sim.model.geom_contype[obj]), int(sim.model.geom_conaffinity[obj])),
        "table": (int(sim.model.geom_contype[table]), int(sim.model.geom_conaffinity[table])),
    }
    for g in fingers:
        sim.model.geom_contype[g] = int(sim.model.geom_contype[g]) | 2
        sim.model.geom_conaffinity[g] = int(sim.model.geom_conaffinity[g]) | 2
    sim.model.geom_contype[obj] = 2
    sim.model.geom_conaffinity[obj] = 2
    mujoco.mj_forward(sim.model, sim.data)
    return {"before": before, "table_id": int(table), "obj_id": obj}


def isolate_ball_object_only(sim) -> dict:
    ball = int(sim.ball_geom)
    obj = int(sim.ids.object_geom)
    sim.model.geom_contype[obj] = int(sim.model.geom_contype[obj]) | 4
    sim.model.geom_conaffinity[obj] = int(sim.model.geom_conaffinity[obj]) | 4
    sim.model.geom_contype[ball] = 4
    sim.model.geom_conaffinity[ball] = 4
    mujoco.mj_forward(sim.model, sim.data)
    return {"ball": 4, "obj": int(sim.model.geom_contype[obj])}


def raw_object_contacts(sim) -> list[dict]:
    model, data, ids = sim.model, sim.data, sim.ids
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    out = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        if ids.object_body not in (b1, b2):
            continue
        other_b = b2 if b1 == ids.object_body else b1
        other_g = g2 if b1 == ids.object_body else g1
        kind = "object-other"
        if other_b == ids.left_body:
            kind = "object-left_finger"
        elif other_b == ids.right_body:
            kind = "object-right_finger"
        elif other_b == ids.hand_body:
            kind = "object-palm_hand"
        elif other_g == table:
            kind = "object-table"
        elif other_g == floor:
            kind = "object-floor"
        wr = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wr)
        out.append({"kind": kind, "fn": float(wr[0]), "g1": geom_name(model, g1), "g2": geom_name(model, g2)})
    return out


def dump_ball_contacts(sim) -> list[dict]:
    rows = []
    if not hasattr(sim, "ball_geom"):
        return rows
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if sim.ball_geom not in (g1, g2):
            continue
        other = g2 if g1 == sim.ball_geom else g1
        kind = "ball-other"
        if int(other) == int(sim.ids.object_geom):
            kind = "ball-cylinder"
        elif int(sim.model.geom_bodyid[other]) in (sim.ids.left_body, sim.ids.right_body):
            kind = "ball-finger"
        rows.append({"kind": kind, "g1": geom_name(sim.model, g1), "g2": geom_name(sim.model, g2)})
    return rows

