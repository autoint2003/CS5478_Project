"""Free camera used by recovery videos. Display only."""

from __future__ import annotations

import mujoco
import numpy as np


def mjv_camera_from_preset(params: dict) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat[:] = np.asarray(params["lookat"], float)
    cam.distance = float(params["distance"])
    cam.azimuth = float(params["azimuth"])
    cam.elevation = float(params["elevation"])
    return cam
