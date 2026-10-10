"""Ground-truth airborne observation. State only, no images.

The historical 27-D observable is kept as a prefix and is not redefined.
Its hand-velocity channels still use the 0.08 m/s scale, so a chase faster
than that saturates them. The airborne block carries the relative object
state, and the hand twist, at scales that cover the demonstrated catch.
"""

from __future__ import annotations

import numpy as np

from envs.deterioration import body_twist
from envs.observable_obs import OBS_NAMES, observe_observable
from envs.observable_reward import read_tactile

OBS_VERSION = "obs_gt_airborne_v2"

# Metres, m/s, rad, rad/s. Chosen so the verified catch (about 1 m/s, a few
# centimetres of relative motion) sits inside (-1, 1) after scaling.
SCALE_PREL = 0.25
SCALE_VREL = 2.0
SCALE_ROT = np.pi
SCALE_WREL = 8.0

AIR_NAMES = [
    "p_rel_hx",
    "p_rel_hy",
    "p_rel_hz",
    "v_rel_hx",
    "v_rel_hy",
    "v_rel_hz",
    "rot_rel_x",
    "rot_rel_y",
    "rot_rel_z",
    "w_rel_hx",
    "w_rel_hy",
    "w_rel_hz",
    "v_hand_hx",
    "v_hand_hy",
    "v_hand_hz",
    "w_hand_hx",
    "w_hand_hy",
    "w_hand_hz",
]

OBS_NAMES_AIR = list(OBS_NAMES) + AIR_NAMES
OBS_DIM_AIR = len(OBS_NAMES_AIR)


def rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, float).reshape(3, 3)
    c = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        float,
    )
    s = 2.0 * np.sin(ang)
    if abs(s) < 1e-8:
        return np.zeros(3)
    return axis * (ang / s)


def relative_state(model, data, ids) -> dict:
    """Object and hand twists in the current hand frame. MuJoCo bodies, not an estimate."""
    Rh = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(data.xpos[ids.hand_body], float)
    po = np.array(data.xpos[ids.object_body], float)
    Ro = np.array(data.xmat[ids.object_body].reshape(3, 3), float)
    vo, wo = body_twist(model, data, ids.object_body)
    vh, wh = body_twist(model, data, ids.hand_body)
    return {
        "p_rel_h": Rh.T @ (po - ph),
        "v_rel_h": Rh.T @ (vo - vh),
        "rot_rel": rotvec(Rh.T @ Ro),
        "w_rel_h": Rh.T @ (wo - wh),
        "v_hand_h": Rh.T @ vh,
        "w_hand_h": Rh.T @ wh,
        "v_hand_w": np.asarray(vh, float).copy(),
        "v_obj_w": np.asarray(vo, float).copy(),
        "p_hand_w": ph.copy(),
        "p_obj_w": po.copy(),
    }


def observe_airborne(model, data, ids, fsm, hist) -> np.ndarray:
    _tac_touch, tac = read_tactile(model, data, ids)
    base = observe_observable(model, data, ids, fsm, tac, hist)
    rel = relative_state(model, data, ids)
    extra = np.array(
        [
            *rel["p_rel_h"] / SCALE_PREL,
            *rel["v_rel_h"] / SCALE_VREL,
            *rel["rot_rel"] / SCALE_ROT,
            *rel["w_rel_h"] / SCALE_WREL,
            *rel["v_hand_h"] / SCALE_VREL,
            *rel["w_hand_h"] / SCALE_WREL,
        ],
        np.float32,
    )
    obs = np.concatenate([base, np.clip(extra, -1.0, 1.0)]).astype(np.float32)
    if obs.shape[0] != OBS_DIM_AIR:
        raise RuntimeError("airborne obs dim mismatch")
    return obs
