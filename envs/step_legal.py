"""One 20 ms legal recovery step.

This is the official recovery transition. Diagnostic searches call it;
they do not own a second copy. The step does not write qpos, qvel, or
the ball pose, and it does not integrate p_des. p_des is reset to the
current hand position at the start of every 20 ms command.
"""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import AIRBORNE_V3, map_action

# Same values the recovery step used when it lived beside the search:
# natural_loss_recatch_transformer_v3.N_SUB and full_3d TRACK.
N_SUB = 10
TRACK = dict(kp_pos=1400.0, kd_pos=80.0, kp_ori=80.0, kd_ori=6.0, kp_null=2.0, kd_null=0.4)


def step_legal(sim, action):
    """One 20 ms recovery step. Does not write qpos, qvel, or the ball pose."""
    action = np.clip(np.asarray(action, np.float32).reshape(7), -1.0, 1.0)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(action, rd, rh, AIRBORNE_V3)
    vel, ang, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.v_cmd = vel.copy()
    sim.fsm.w_cmd = ang.copy()
    sim.fsm.v_des = vel.copy()
    sim.fsm.w_des = ang.copy()
    sim.fsm.fg_cmd = float(-tau)
    from envs.outcomes.table_drop import object_vz, observe

    dt = float(sim.model.opt.timestep)
    for _ in range(N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, dt)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
        )
        vz_before = object_vz(sim)
        mujoco.mj_step(sim.model, sim.data)
        observe(sim, vz_before)
    return action
