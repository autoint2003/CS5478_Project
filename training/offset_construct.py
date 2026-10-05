"""Construct settled airborne hand-x offset ICs for SAC training.

Privileged apply_rel_pose is IC-construction only. Not used in RecoveryEnv.step.
Does not read held-out eval_sets npz files.
"""

from __future__ import annotations

import numpy as np

from training.replay_core import (
    MASS,
    MU,
    disable_object_table,
    freeze,
    object_free_adr,
    obs_from_sim,
    tick_vw,
)
from controllers.jacobian_controller import gains_from_cfg
from envs.grasp_sim import GraspSim
from envs.physical_recovery import Z_AIR, physical_pack


def apply_rel_pose(sim, e_x: float) -> None:
    """Set object origin so hand-frame x equals e_x; keep current y,z in hand frame."""
    ids = sim.ids
    Rh = np.array(sim.data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[ids.hand_body], float)
    po = np.array(sim.data.xpos[ids.object_body], float)
    rh = Rh.T @ (po - ph)
    rh[0] = float(e_x)
    po_new = ph + Rh @ rh
    qadr, dadr = object_free_adr(sim)
    sim.data.qpos[qadr : qadr + 3] = po_new
    sim.data.qvel[dadr : dadr + 6] = 0.0
    import mujoco

    mujoco.mj_forward(sim.model, sim.data)


def lift_to_airborne(sim, cfg, max_time: float = 12.0) -> bool:
    n = int(round(max_time / sim.model.opt.timestep))
    for _ in range(n):
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
        if sim.fsm.phase == "lift" and sim.captured:
            z = float(sim.data.xpos[sim.ids.object_body][2])
            if z >= Z_AIR:
                freeze(sim)
                return True
    return False


def settle_hold(sim, cfg, duration: float = 0.25, tau: float = -18.0) -> None:
    gains = gains_from_cfg(cfg)
    n = int(round(duration / sim.model.opt.timestep))
    for _ in range(n):
        tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)


def ic_ok(o: dict, e_lo: float, e_hi: float) -> bool:
    ae = abs(float(o["e_x"]))
    return (
        e_lo <= ae <= e_hi
        and int(o["nL"]) > 0
        and int(o["nR"]) > 0
        and int(o["scene"]) == 0
        and float(o["obj_z"]) >= Z_AIR
        and float(o["clear"]) >= 0.02
        and float(o["v_rel"]) < 0.05
        and float(o.get("w_rel", 0.0)) < 0.5
    )


def make_one_training_ic(cfg: dict, rng: np.random.Generator) -> dict:
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    if not lift_to_airborne(sim, cfg):
        raise RuntimeError("failed to reach airborne lift")
    disable_object_table(sim)
    freeze(sim)
    for _ in range(24):
        e_x = float(rng.uniform(-0.0075, 0.0075))
        if abs(e_x) < 0.002:
            e_x = float(np.sign(e_x) or 1.0) * float(rng.uniform(0.002, 0.0075))
        apply_rel_pose(sim, e_x)
        settle_hold(sim, cfg, 0.25, -18.0)
        o = physical_pack(sim)
        if ic_ok(o, 0.002, 0.0085):
            snap = sim.snapshot()
            snap["source"] = "constructed_offset"
            snap["e_x"] = float(o["e_x"])
            snap["v_cmd"] = np.zeros(3)
            snap["w_cmd"] = np.zeros(3)
            return snap
        freeze(sim)
    raise RuntimeError("could not construct a valid offset IC")
