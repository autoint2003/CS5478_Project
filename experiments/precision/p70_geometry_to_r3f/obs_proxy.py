"""Deployable observation proxies — no privileged object pose / e_pos.

Actor obs (ObservableObs) has no object COM error. Controllers must not
secretly read working_geometry e_pos. Proxies used here:

- contact_present_L/R (indices 2,3)
- estimate_valid, e_hat_x (tactile, post-contact only)
- aperture, ap_dot
- hand linear speed magnitude from v_hx,v_hy,v_z_world

Privileged e_pos is logged for *analysis only*.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# OBS_NAMES indices from envs.observations.observable_obs
IX_E_HAT = 0
IX_EST_VALID = 1
IX_CONTACT_L = 2
IX_CONTACT_R = 3
IX_APERTURE = 12
IX_AP_DOT = 13
IX_VHX = 18
IX_VHY = 19
IX_VZ = 20
IX_TRACK_Z = 25


def last_obs(env) -> np.ndarray:
    if not env.obs_seq:
        return np.zeros(27, dtype=np.float32)
    return np.asarray(env.obs_seq[-1], dtype=np.float32).reshape(-1)


def obs_features(obs: np.ndarray) -> dict[str, Any]:
    o = np.asarray(obs, float).reshape(-1)
    if o.size < 26:
        o = np.pad(o, (0, 26 - o.size))
    cL = float(o[IX_CONTACT_L]) > 0.5
    cR = float(o[IX_CONTACT_R]) > 0.5
    vh = float(np.linalg.norm([o[IX_VHX], o[IX_VHY], o[IX_VZ]]))
    return {
        "e_hat_x": float(o[IX_E_HAT]),
        "estimate_valid": float(o[IX_EST_VALID]) > 0.5,
        "contact_L": cL,
        "contact_R": cR,
        "any_contact_obs": cL or cR,
        "aperture": float(o[IX_APERTURE]),
        "ap_dot": float(o[IX_AP_DOT]),
        "hand_speed": vh,
        "track_z": float(o[IX_TRACK_Z]),
    }


def alignment_proxy_deployable(obs: np.ndarray) -> dict[str, Any]:
    """Pre-contact alignment proxy candidates (deployable).

    None of these equal e_pos. Used only as soft signals for Variant 2.
    """
    f = obs_features(obs)
    # Heuristic score: prefer slow approach + moderate aperture (closing)
    # Higher score ⇒ more "ready to grip" in proprioceptive sense.
    ready_score = (
        -2.0 * f["hand_speed"]
        + 0.5 * (1.0 - abs(f["aperture"]))  # aperture already scaled
        - 0.3 * abs(f["track_z"])
    )
    return {
        **f,
        "ready_score": float(ready_score),
        "note": "NOT e_pos; proprioceptive readiness only",
    }
