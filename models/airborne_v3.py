"""39-D airborne policy and one v3 policy step.

load_airborne_policy and build_model load the causal Transformer.
step_v3 executes one recovery7d_airborne_v3 command. The historical demo
generator lives in experiments/legacy/airborne_v3_demos.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.cartesian.tracking import TRACK
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import (
    AIRBORNE_V2,
    AIRBORNE_V3,
    V_Z_DOWN,
    V_Z_UP,
    map_action,
    normalize_action,
    v1_normalized_to_v3,
    v2_normalized_to_v3,
)
from envs.initialization.grasp_seat import target_hand
from envs.observations.airborne_obs import OBS_DIM_AIR
from envs.outcomes.physical_recovery import physical_pack
from envs.simulation.grasp_quality import body_twist

DT_POLICY = 0.02
N_SUB = 10
K_INT = 200.0
HOLD = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)


def assert_map():
    """Canonical horizontal grasp: hand +z is world down."""
    eye = np.eye(3)
    rh = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], float)
    down = map_action([0, 0, 1, 0, 0, 0, 1], eye, rh, AIRBORNE_V3)
    up = map_action([0, 0, -1, 0, 0, 0, -1], eye, rh, AIRBORNE_V3)
    if abs(float(down["v_world"][2]) + V_Z_DOWN) > 1e-6:
        raise RuntimeError(f"v3 +1 is not world down {down['v_world']}")
    if abs(float(up["v_world"][2]) - V_Z_UP) > 1e-6:
        raise RuntimeError(f"v3 -1 is not world up {up['v_world']}")
    full_v2 = np.array([0, 0, 1, 0, 0, 0, 1], np.float32)
    as_v3 = v2_normalized_to_v3(full_v2)
    if abs(float(as_v3[2]) - 1.0) < 1e-3:
        raise RuntimeError("v2 normalized +1 was reused as v3 +1")
    back = map_action(as_v3, eye, rh, AIRBORNE_V3)["v_world"]
    src = map_action(full_v2, eye, rh, AIRBORNE_V2)["v_world"]
    if not np.allclose(back, src, atol=1e-5):
        raise RuntimeError(f"v2->v3 changed the physical command {src} {back}")
    return {
        "version": AIRBORNE_V3,
        "vx_vy": [-1.0, 1.0],
        "vz_hand_positive_m_s": V_Z_DOWN,
        "vz_hand_negative_m_s": -V_Z_UP,
        "world_down_when_hand_z_is_world_down": -V_Z_DOWN,
        "world_up_when_hand_z_is_world_down": V_Z_UP,
        "v2_full_down_rewritten_a2": round(float(as_v3[2]), 4),
        "wx_wy_wz": 4.0,
        "grip_open_nm": 2.0,
        "grip_secure_nm": -18.0,
    }


def step_v3(sim, action, stop_on_loss=False):
    """One policy step. Setpoint is rebased to the hand, then v_des is held.

    Integrating p_des at the full downward command runs the setpoint past the
    cylinder. Rebasing each policy step and holding the intercept velocity is
    what keeps the 20 ms chase on the verified grasp.
    """
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(action, rd, rh, AIRBORNE_V3)
    v, w, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.v_cmd = v.copy()
    sim.fsm.w_cmd = w.copy()
    sim.fsm.v_des = v.copy()
    sim.fsm.w_des = w.copy()
    sim.fsm.fg_cmd = float(-tau)
    dt = float(sim.model.opt.timestep)
    for _ in range(N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, w, dt)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=v, w_des=w, **TRACK,
        )
        mujoco.mj_step(sim.model, sim.data)
        if hasattr(sim, "park_ball"):
            sim.park_ball()
        if stop_on_loss:
            o = physical_pack(sim)
            if int(o["nL"]) == 0 or int(o["nR"]) == 0:
                break
    cmd["tau"] = tau
    return cmd


def intercept_action(sim, pinch, phase):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    p_des = target_hand(sim, pinch)
    e = p_des - np.array(sim.data.xpos[sim.ids.hand_body], float)
    vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
    if phase == "hold":
        v = vo + 2.0 * e
        tau = -18.0
    elif phase == "close":
        v = vo + 8.0 * e
        tau = -18.0
    else:
        v = vo + K_INT * e
        tau = 2.0
    v = np.array(v, float)
    v[0] = float(np.clip(v[0], -1.0, 1.0))
    v[1] = float(np.clip(v[1], -1.0, 1.0))
    v[2] = float(np.clip(v[2], -V_Z_DOWN, V_Z_UP))
    v_h = rh.T @ v
    return normalize_action(v_h, np.zeros(3), tau, AIRBORNE_V3), v


def build_model(proto, device):
    import torch

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("proto", torch.tensor(np.asarray(proto, np.float32)))
            d = 64
            self.in_proj = torch.nn.Linear(OBS_DIM_AIR + 7, d)
            self.pos = torch.nn.Embedding(160, d)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=128, dropout=0.0,
                batch_first=True, activation="relu",
            )
            self.enc = torch.nn.TransformerEncoder(layer, num_layers=2)
            self.logits = torch.nn.Linear(d, len(proto))
            self.res = torch.nn.Linear(d, 7)
            torch.nn.init.zeros_(self.res.weight)
            torch.nn.init.zeros_(self.res.bias)

        def forward(self, obs, prev, pad_mask=None):
            x = torch.cat([obs, prev], dim=-1)
            t = x.shape[-2]
            h = self.in_proj(x) + self.pos(torch.arange(t, device=x.device))
            causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
            if x.dim() == 2:
                ctx = self.enc(h.unsqueeze(0), mask=causal)[0]
            else:
                ctx = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
            logits = self.logits(ctx)
            res = 0.25 * torch.tanh(self.res(ctx))
            tok = torch.argmax(logits, dim=-1)
            act = torch.clamp(self.proto[tok] + res, -1.0, 1.0)
            return logits, res, act

    model = Policy().to(device)
    return model


def load_airborne_policy(path, device):
    """Load a checkpoint trained for the current observation width.

    A checkpoint whose first layer was built for a different observation,
    including the retired 45-D vector, is refused. Weights are not resized.
    """
    import torch

    saved = torch.load(path, map_location=device, weights_only=False)
    if "obs_dim" in saved and int(saved["obs_dim"]) != OBS_DIM_AIR:
        raise RuntimeError(
            f"{path} records obs_dim {saved['obs_dim']}; the active observation is {OBS_DIM_AIR}-D"
        )
    proto = np.asarray(saved["proto"], np.float32)
    model = build_model(proto, device)
    weight = saved["model"]["in_proj.weight"]
    expected = tuple(model.in_proj.weight.shape)
    if tuple(weight.shape) != expected:
        raise RuntimeError(
            f"{path} in_proj weight is {tuple(weight.shape)}; "
            f"the active model expects {expected} "
            f"({OBS_DIM_AIR}-D observation plus the 7-D previous command). "
            "The checkpoint is not adapted."
        )
    model.load_state_dict(saved["model"])
    model.eval()
    return model, proto, saved

