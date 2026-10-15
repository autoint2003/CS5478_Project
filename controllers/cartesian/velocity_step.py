"""One Cartesian velocity step and a hand-pose freeze.

tick_vw integrates the desired position and calls the Jacobian controller.
The RULE baseline and the nominal hold both use it.
"""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot


def freeze(sim):
    p = sim.data.xpos[sim.ids.hand_body].copy()
    r = sim.data.xmat[sim.ids.hand_body].reshape(3, 3).copy()
    sim.fsm.p_des = p.copy()
    sim.fsm.r_des = r.copy()
    sim.fsm.v_cmd[:] = 0
    sim.fsm.w_cmd[:] = 0
    return p, r


def tick_vw(sim, v_w, w_w, tau, gains):
    dt = float(sim.model.opt.timestep)
    v = np.asarray(v_w, float).reshape(3)
    w = np.asarray(w_w, float).reshape(3)
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = sim.fsm.p_des + v * dt
    sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, w, dt)
    sim.fsm.v_cmd = v.copy()
    sim.fsm.w_cmd = w.copy()
    apply_cartesian_ctrl(
        sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
        gripper_tau=tau, v_des=v, w_des=w, **gains,
    )
    mujoco.mj_step(sim.model, sim.data)
    if hasattr(sim, "park_ball"):
        sim.park_ball()

