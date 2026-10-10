"""Versioned 7-D recovery actions.

recovery7d_local_v1 is the historical map: hand-frame translation at ±0.08 m/s.
recovery7d_airborne_v2 uses the same rotation and grip maps, and raises the
translational command range to ±1.0 m/s on vx, vy, and vz.
recovery7d_airborne_v3 keeps vx and vy at ±1.0 m/s and keeps the upward
hand-frame vz authority at 1.0 m/s. Positive hand-frame vz, which is world
downward on the horizontal grasp, extends to 3.8 m/s.

Normalized actions from one version must not be fed through another. Convert
physical velocity first. These maps only produce a Cartesian command. Joint
limits, actuator torque limits, and MuJoCo dynamics still determine the motion.
"""

from __future__ import annotations

import numpy as np

from controllers.residual import (
    RECOVERY4D_TAU_OPEN,
    RECOVERY4D_TAU_SECURE,
    RECOVERY4D_W_HY_MAX,
)

LOCAL_V1 = "recovery7d_local_v1"
AIRBORNE_V2 = "recovery7d_airborne_v2"
AIRBORNE_V3 = "recovery7d_airborne_v3"

V_LOCAL = 0.08  # m/s, hand frame, historical recovery7 / map7
V_AIRBORNE = 1.0  # m/s, hand frame, symmetric v2 range
# v3 hand-frame vz. On the horizontal grasp, hand +z is world −z, so the
# positive channel is the downward command that the cap study set to 3.8 m/s.
# Negative hand-frame vz stays at the v2 upward authority of 1.0 m/s.
V_Z_DOWN = 3.8
V_Z_UP = 1.0
W_MAX = float(RECOVERY4D_W_HY_MAX)  # rad/s in the r_des frame
TAU_OPEN = float(RECOVERY4D_TAU_OPEN)  # +2 N·m at grip action -1
TAU_SECURE = float(RECOVERY4D_TAU_SECURE)  # -18 N·m at grip action +1

V_OF = {
    LOCAL_V1: V_LOCAL,
    AIRBORNE_V2: V_AIRBORNE,
    AIRBORNE_V3: V_AIRBORNE,
}


def grip_tau(a_grip: float) -> float:
    a = float(np.clip(a_grip, -1.0, 1.0))
    return TAU_OPEN + 0.5 * (a + 1.0) * (TAU_SECURE - TAU_OPEN)


def grip_action(tau: float) -> float:
    span = TAU_SECURE - TAU_OPEN
    a = 2.0 * (float(tau) - TAU_OPEN) / span - 1.0
    return float(np.clip(a, -1.0, 1.0))


def hand_translation(action, version: str) -> np.ndarray:
    """Normalized translation channels -> hand-frame metres per second."""
    if version not in V_OF:
        raise ValueError(f"unknown recovery action version {version}")
    a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
    if version != AIRBORNE_V3:
        return a[0:3] * float(V_OF[version])
    vz = float(a[2]) * V_Z_DOWN if float(a[2]) >= 0.0 else float(a[2]) * V_Z_UP
    return np.array([float(a[0]) * V_AIRBORNE, float(a[1]) * V_AIRBORNE, vz], float)


def normalize_vz(vz_hand: float) -> float:
    """Hand-frame vz (m/s) -> v3 action. Positive is the 3.8 m/s downward side."""
    vz = float(vz_hand)
    if vz >= 0.0:
        return float(np.clip(vz / V_Z_DOWN, 0.0, 1.0))
    return float(np.clip(vz / V_Z_UP, -1.0, 0.0))


def map_action(action, r_des, r_hand, version: str) -> dict:
    """Normalized 7-vector -> world linear velocity, world angular velocity, grip torque.

    v_world = R_hand @ v_hand
    w_world = R_des  @ (a[3:6] * 4 rad/s)
    Components are vx, vy, vz in the hand frame and wx, wy, wz in the r_des frame.
    v3 does not scale vz by a single number: +1 maps to +3.8 m/s and -1 maps to -1.0 m/s.
    """
    if version not in V_OF:
        raise ValueError(f"unknown recovery action version {version}")
    a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
    if a.size < 7:
        raise ValueError("recovery action needs 7 components")
    v_scale = float(V_OF[version])
    v_h = hand_translation(a, version)
    w_b = a[3:6] * W_MAX
    rh = np.asarray(r_hand, float).reshape(3, 3)
    rd = np.asarray(r_des, float).reshape(3, 3)
    return {
        "version": version,
        "a": a[:7].copy(),
        "v_hand": v_h.copy(),
        "w_body": w_b.copy(),
        "v_world": rh @ v_h,
        "w_world": rd @ w_b,
        "tau": grip_tau(a[6]),
        "v_scale": v_scale,
    }


def normalize_action(v_hand, w_body, tau, version: str) -> np.ndarray:
    """Physical hand-frame v, r_des-frame w, and grip torque -> normalized action.

    Translation outside the version's command range is clipped. That clip is the
    action interface, not a change to the torque limits.
    """
    v = np.asarray(v_hand, float).reshape(3)
    a = np.zeros(7, np.float32)
    if version == AIRBORNE_V3:
        a[0] = float(np.clip(v[0] / V_AIRBORNE, -1.0, 1.0))
        a[1] = float(np.clip(v[1] / V_AIRBORNE, -1.0, 1.0))
        a[2] = normalize_vz(v[2])
    else:
        v_scale = float(V_OF[version])
        a[0:3] = np.clip(v / v_scale, -1.0, 1.0)
    a[3:6] = np.clip(np.asarray(w_body, float).reshape(3) / W_MAX, -1.0, 1.0)
    a[6] = grip_action(tau)
    return a


def _translate_version(action, src: str, dst: str) -> np.ndarray:
    """Same physical command, written in another version's normalization."""
    a = np.asarray(action, float).reshape(-1)
    v_h = hand_translation(a, src)
    w_b = np.clip(a[3:6], -1.0, 1.0) * W_MAX
    out = normalize_action(v_h, w_b, grip_tau(a[6]), dst)
    return out


def v2_normalized_to_v3(action) -> np.ndarray:
    """v2 normalized action -> v3. A v2 downward command of 1 m/s is not v3's +1."""
    return _translate_version(action, AIRBORNE_V2, AIRBORNE_V3)


def v1_normalized_to_v3(action) -> np.ndarray:
    return _translate_version(action, LOCAL_V1, AIRBORNE_V3)


def v1_normalized_to_v2(action) -> np.ndarray:
    """Same physical command, rewritten for the airborne scale.

    a_v2 translation = a_v1 * 0.08 / 1.0. Rotation and grip channels match.
    """
    a = np.asarray(action, float).reshape(-1)
    out = np.zeros(7, np.float32)
    out[0:3] = np.clip(a[0:3] * (V_LOCAL / V_AIRBORNE), -1.0, 1.0)
    out[3:7] = np.clip(a[3:7], -1.0, 1.0)
    return out
