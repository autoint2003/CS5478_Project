"""Online episode: nominal, geometry entry, 39-D policy, stable exit, nominal.

The task clock is sim.data.time. Recovery does not reset it, and it does not
restore qpos, qvel, or a prepared snapshot. Future training should construct
episodes with make_training_env. This module does not train.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
from controllers.geometry_switch import (
    CONTROL_PERIOD_S,
    GeometrySwitch,
)
from envs.airborne_obs import OBS_DIM_AIR, observe_airborne
from envs.observable_obs import ObservableObsState
from envs.physical_recovery import Z_AIR, physical_pack
from training.demo_ballistic_impact import HAND_RISE_M, TAU_SEC, assert_noslip, ball_force_on_ball, z_tgt_of
from training.full_3d_airborne_recatch_dataset import TRACK, drop_supports, limits_of
from training.horizontal_grasp_offset_freefall_recatch import begin, grasp_settled, hold_step
from training.impact_ball import apply_ball_free_state
from training.observation_study_common import load_balanced
from training.phase_decoupled_recovery_pilot import world
from training.recovery_runtime import (
    T_TASK,
    apply_dyn,
    continue_long,
    launch_offcenter,
    physical_of,
    stamp,
    write_dyn_quiet,
)
from training.replay_core import park_impact_ball, tick_vw

Z_ABORT = 0.44


def nominal_substep(sim, gains, st):
    if (not st["holding"]) and sim.fsm.phase == "lift" and st["z_tgt"] is not None:
        if float(sim.fsm.p_des[2]) >= st["z_tgt"] - 1e-9:
            st["holding"] = True
    if st["holding"]:
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
    else:
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
    if sim.fsm.phase == "lift" and st["hand_z0"] is None:
        st["hand_z0"] = float(sim.data.xpos[sim.ids.hand_body][2])
        st["z_tgt"] = float(z_tgt_of(sim))
    return float(sim.data.time)


def fresh_nominal_state():
    return {"holding": False, "hand_z0": None, "z_tgt": None}


def empty_ball():
    return {
        "holding": False,
        "hand_z0": None,
        "z_tgt": None,
        "released": False,
        "parked": False,
        "shifted": False,
        "t_impact": None,
        "t_last": None,
        "t_park": None,
        "t_release": None,
    }


def ball_substep(sim, gains, nominal, ball, v_hit, z_off, phi, delta):
    nominal_substep(sim, gains, ball)
    t = float(sim.data.time)
    o = physical_pack(sim)
    hz = float(sim.data.xpos[sim.ids.hand_body][2])
    if (
        (not ball["released"])
        and sim.fsm.phase == "lift"
        and sim.captured
        and ball["hand_z0"] is not None
        and float(o["obj_z"]) >= Z_AIR
        and (hz - ball["hand_z0"]) >= HAND_RISE_M
    ):
        write_dyn_quiet(sim, nominal, None, None)
        launch = launch_offcenter(sim, v_hit, z_off, phi)
        sim.set_guide_weld(False)
        apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
        ball["released"] = True
        ball["t_release"] = float(sim.data.time)
    if ball["released"] and not ball["parked"]:
        _fb, bc = ball_force_on_ball(sim)
        has = any(r["kind"] == "ball-cylinder" for r in bc)
        if has and ball["t_impact"] is None:
            ball["t_impact"] = t
        if has:
            ball["t_last"] = t
        departed = (
            ball["t_impact"] is not None
            and ball["t_last"] is not None
            and (not has)
            and t > ball["t_last"] + 0.025
        )
        stuck = ball["t_impact"] is not None and t > ball["t_impact"] + 0.35
        if departed or stuck:
            park_impact_ball(sim)
            ball["parked"] = True
            ball["t_park"] = float(sim.data.time)
    if (
        delta is not None
        and ball["parked"]
        and (not ball["shifted"])
        and ball["t_park"] is not None
        and t >= ball["t_park"] + 0.040
    ):
        adr = int(sim.model.jnt_qposadr[int(sim.ids.object_jnt)])
        sim.data.qpos[adr:adr + 3] = sim.data.qpos[adr:adr + 3] + np.asarray(delta, float)
        mujoco.mj_forward(sim.model, sim.data)
        ball["shifted"] = True
    return t


def seat_base(sim, nominal, pads, pad_mu):
    seat = grasp_settled(sim, TRACK)
    if seat is None or not begin(sim, seat, nominal, pads, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0}):
        raise RuntimeError("seat failed")
    for _ in range(40):
        hold_step(sim, TRACK)
    drop_supports(sim)
    q_ref, _margin = limits_of(sim)
    return stamp(sim), q_ref


class AutonomousRecoveryEnv:
    """Nominal world, 39-D observation, geometry switch, and the frozen policy."""

    def __init__(self, device=None):
        torch.set_num_threads(1)
        self.sim, self.gains, self.nominal, self.pads, self.pad_mu, self.pads_l, self.pads_r = world()
        assert_noslip(self.sim)
        self.model, self.proto, self.saved, self.device = load_balanced(device)
        self.model.eval()
        self.switch = GeometrySwitch()
        self._hist = None
        self._prev = None
        self._obs_seq = None
        self._prev_seq = None

    def observation(self):
        hist = self._hist if self._hist is not None else ObservableObsState(v3.DT_POLICY)
        obs = observe_airborne(self.sim.model, self.sim.data, self.sim.ids, self.sim.fsm, hist)
        obs = np.asarray(obs, np.float32)
        if obs.shape != (OBS_DIM_AIR,):
            raise RuntimeError(f"observation {obs.shape}")
        return obs

    def begin_nominal(self):
        self.sim.reset()
        apply_dyn(self.sim, self.nominal, None, None)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.switch = GeometrySwitch()
        self._hist = None
        self._prev = None
        self._obs_seq = None
        self._prev_seq = None

    def nominal_steps(self, n, state):
        for _ in range(int(n)):
            nominal_substep(self.sim, self.gains, state)
            if float(self.sim.data.time) >= T_TASK - 1e-9:
                return

    def start_recovery(self):
        """Arm the policy on the live state. Does not restore or reset the clock."""
        if abs(float(v3.DT_POLICY) - CONTROL_PERIOD_S) > 1e-9:
            raise RuntimeError("policy step and geometry control period differ")
        self._hist = ObservableObsState(v3.DT_POLICY)
        self._prev = np.zeros(7, np.float32)
        self._obs_seq = []
        self._prev_seq = []

    def recovery_step(self):
        obs = self.observation()
        self._obs_seq.append(np.asarray(obs, np.float32))
        self._prev_seq.append(self._prev.copy())
        with torch.no_grad():
            pred = self.model(
                torch.tensor(np.stack(self._obs_seq), device=self.device),
                torch.tensor(np.stack(self._prev_seq), device=self.device),
            )[2][-1].float().cpu().numpy()
        act = np.clip(pred, -1.0, 1.0).astype(np.float32)
        v3.step_v3(self.sim, act)
        self._prev = act
        return act

    def finish_nominal(self, gains):
        rec = continue_long(self.sim, gains, T_TASK, "nominal")
        y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
        t12 = "RETAINED_TO_12" if y == 1 else kind
        return t12, float(self.sim.data.time)


def make_training_env(device=None):
    """Construct the autonomous training environment. Does not step or train."""
    return AutonomousRecoveryEnv(device)


def run_ball_episode(env, v_hit, z_off, phi, delta=None):
    """Nominal lift, online ball, geometry entry, policy, stable exit, t = 12."""
    env.begin_nominal()
    ball = empty_ball()
    switch = env.switch
    contact_z = None
    while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            ball_substep(env.sim, env.gains, env.nominal, ball, v_hit, z_off, phi, delta)
            if float(env.sim.data.time) >= T_TASK - 1e-9:
                break
        row = switch.measure(env.sim)
        if ball["t_impact"] is not None and contact_z is None and row["t"] + 1e-9 >= ball["t_impact"]:
            contact_z = row["z"]
        if ball["released"]:
            switch.update_entry(env.sim)
        if ball["released"] and row["z"] < Z_ABORT:
            break
        if (not ball["released"]) and float(env.sim.data.time) > 8.0:
            break
    out = {
        "t_impact": ball["t_impact"],
        "t_entry": switch.t_entry,
        "entry_reason": switch.entry_reason,
        "entered": bool(switch.entered),
        "qpos_reset": False,
        "clock_reset": False,
    }
    if not switch.entered:
        t12, t_end = env.finish_nominal(env.gains)
        out.update({
            "t12": t12,
            "t_end": t_end,
            "exit_kind": "no_entry",
            "t_exit": None,
            "e_pos_exit": None,
            "e_rot_exit": None,
            "vrel_exit": None,
            "steps": 0,
            "first_step_dt": None,
        })
        return out
    t_before = float(env.sim.data.time)
    q_before = np.array(env.sim.data.qpos, float).copy()
    env.start_recovery()
    if abs(float(env.sim.data.time) - t_before) > 1e-9:
        out["clock_reset"] = True
    if not np.allclose(env.sim.data.qpos, q_before):
        out["qpos_reset"] = True
    steps = 0
    exit_kind = "none"
    last = None
    first_dt = None
    while float(env.sim.data.time) < T_TASK - 1e-9:
        t0 = float(env.sim.data.time)
        env.recovery_step()
        steps += 1
        if first_dt is None:
            first_dt = float(env.sim.data.time) - t0
        last = switch.measure(env.sim)
        if last["z"] < Z_ABORT:
            exit_kind = "fall_abort"
            break
        if switch.update_exit(env.sim):
            exit_kind = "geometry"
            break
    if exit_kind == "fall_abort":
        t12, t_end = "DROP", float(env.sim.data.time)
    else:
        t12, t_end = env.finish_nominal(TRACK)
    out.update({
        "t12": t12,
        "t_end": t_end,
        "exit_kind": exit_kind,
        "t_exit": None if last is None else last["t"],
        "e_pos_exit": None if last is None else last["e_pos"],
        "e_rot_exit": None if last is None else last["e_rot"],
        "vrel_exit": None if last is None else last["vrel"],
        "steps": steps,
        "first_step_dt": first_dt,
        "contact_z": contact_z,
    })
    return out


def run_undisturbed(env):
    """Nominal grasp through absolute t = 12. Recovery must not start."""
    env.begin_nominal()
    st = fresh_nominal_state()
    switch = env.switch
    max_pos = 0.0
    max_rot = 0.0
    while float(env.sim.data.time) < T_TASK - 1e-9:
        env.nominal_steps(v3.N_SUB, st)
        row = switch.measure(env.sim)
        max_pos = max(max_pos, row["e_pos"])
        max_rot = max(max_rot, row["e_rot"])
        if switch.update_entry(env.sim):
            break
    return {
        "entered": bool(switch.entered),
        "t_entry": switch.t_entry,
        "max_pos": max_pos,
        "max_rot": max_rot,
        "t_end": float(env.sim.data.time),
        "t12_reached": float(env.sim.data.time) >= T_TASK - 1e-6,
    }
