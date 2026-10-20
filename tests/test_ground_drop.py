"""Unit tests for GROUND_DROP (no elevated-table threshold)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.outcomes.ground_drop import (
    BOUNCE_DV,
    SUPPORT_FRACTION,
    SUPPORT_TIME_S,
    consider,
    _fresh,
)
from envs.simulation.ground_scene import GROUND_DROP_Z, SCENE_ID, write_ground_scene
from envs.simulation.ground_sim import make_ground_sim
from envs.outcomes.ground_drop import arm, disarm, latched, observe
import mujoco


DT = 0.002
WEIGHT = 0.20 * 9.81


def _check(name, cond):
    if not cond:
        raise RuntimeError(name)
    print("PASS", name, flush=True)


def test_unit():
    half = SUPPORT_FRACTION * WEIGHT
    held = _fresh()
    held["armed"] = True
    consider(held, DT, 1.0, 0.05, True, 40.0, -1.0, 0.5, WEIGHT)
    _check("bilateral ignores ground", not held["latched"])

    cold = _fresh()
    for _ in range(30):
        consider(cold, DT, 1.0, 0.03, False, half, 0.0, 0.0, WEIGHT)
    _check("disarmed never latches", not cold["latched"])

    brush = _fresh()
    brush["armed"] = True
    for step in range(9):
        consider(brush, DT, step * DT, 0.03, False, half, 0.0, 0.0, WEIGHT)
    _check("support short of window", not brush["latched"])
    consider(brush, DT, 9 * DT, 0.03, False, half, 0.0, 0.0, WEIGHT)
    _check("support latches", brush["latched"] and brush["kind"] == "support")

    bounce = _fresh()
    bounce["armed"] = True
    consider(bounce, DT, 1.0, 0.04, False, 30.0, -1.2, 0.3, WEIGHT)
    _check("bounce latches", bounce["latched"] and bounce["dv"] >= BOUNCE_DV)
    _check("scene id constant", SCENE_ID == "GROUND_SCENE")
    _check("ground drop z is not table gate", abs(GROUND_DROP_Z - 0.04) < 1e-9)


def test_scene_physics():
    write_ground_scene()
    sim, _ = make_ground_sim()
    table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    ground = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    _check("no table geom", table < 0)
    _check("ground geom present", ground >= 0)
    _check("scene_id set", sim.scene_id == SCENE_ID)
    disarm(sim)
    for _ in range(40):
        observe(sim, 0.0)
        mujoco.mj_step(sim.model, sim.data)
    _check("resting cylinder does not GROUND_DROP while disarmed", not latched(sim))


def main():
    test_unit()
    test_scene_physics()
    print("GROUND_DROP_TESTS_OK", flush=True)


if __name__ == "__main__":
    main()
