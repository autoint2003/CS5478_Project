"""Geometry hysteresis for recovery entry and exit.

Thresholds are loaded from config/geometry_switch.yaml. Position error is the
only entry signal. Exit is a stable-grasp region, not a return to the lift pose.
After exit the handoff is one-way for the rest of the episode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.config_util import load_yaml
from envs.deterioration import body_twist
from envs.physical_recovery import physical_pack

_CFG = load_yaml(Path(__file__).resolve().parents[1] / "config" / "geometry_switch.yaml")

CONTROL_PERIOD_S = float(_CFG["control_period_s"])
ENTRY_POSITION_THRESHOLD_M = float(_CFG["entry_position_threshold_m"])
ENTRY_PERSISTENCE_S = float(_CFG["entry_persistence_s"])
EXIT_POSITION_THRESHOLD_M = float(_CFG["exit_position_threshold_m"])
EXIT_ORIENTATION_THRESHOLD_RAD = float(_CFG["exit_orientation_threshold_rad"])
EXIT_RELATIVE_SPEED_THRESHOLD_MPS = float(_CFG["exit_relative_speed_threshold_mps"])
EXIT_STABILITY_WINDOW_S = float(_CFG["exit_stability_window_s"])

# Three samples span 40 ms. Five samples cover the 100 ms exit window.
ENTRY_PERSISTENCE_STEPS = int(round(ENTRY_PERSISTENCE_S / CONTROL_PERIOD_S)) + 1
EXIT_STABILITY_STEPS = int(round(EXIT_STABILITY_WINDOW_S / CONTROL_PERIOD_S))


def hand_pose(sim):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    return rh.T @ (po - ph), rh.T @ ro


def rot_angle(r0, r):
    err = r0.T @ r
    c = float(np.clip(0.5 * (np.trace(err) - 1.0), -1.0, 1.0))
    return float(np.arccos(c))


def relative_speed(sim):
    vo, _wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, _wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    return float(np.linalg.norm(np.asarray(vo, float) - np.asarray(vh, float)))


class GeometrySwitch:
    """Records the lift grasp, then entry and one-way exit on later samples."""

    def __init__(self):
        self.p0 = None
        self.r0 = None
        self.pos_run = 0
        self.exit_run = 0
        self.entered = False
        self.exited = False
        self.t_entry = None
        self.t_exit = None
        self.entry_reason = None

    def capture_if_ready(self, sim):
        if self.p0 is None and sim.captured and str(sim.fsm.phase) == "lift":
            self.p0, self.r0 = hand_pose(sim)

    def measure(self, sim):
        self.capture_if_ready(sim)
        o = physical_pack(sim)
        t = float(sim.data.time)
        if self.p0 is None:
            e_pos, e_rot = 0.0, 0.0
        else:
            p, r = hand_pose(sim)
            e_pos = float(np.linalg.norm(p - self.p0))
            e_rot = rot_angle(self.r0, r)
        return {
            "t": t,
            "e_pos": e_pos,
            "e_rot": e_rot,
            "vrel": relative_speed(sim),
            "nL": int(o["nL"]),
            "nR": int(o["nR"]),
            "z": float(o["obj_z"]),
        }

    def update_entry(self, sim):
        """One nominal control step. Position error only. Returns True on entry."""
        row = self.measure(sim)
        if self.entered or self.exited:
            return False
        hit = self.p0 is not None and row["e_pos"] > ENTRY_POSITION_THRESHOLD_M
        self.pos_run = self.pos_run + 1 if hit else 0
        if self.pos_run >= ENTRY_PERSISTENCE_STEPS:
            self.entered = True
            self.t_entry = row["t"]
            self.entry_reason = "position"
            self.exit_run = 0
            return True
        return False

    def update_exit(self, sim):
        """One recovery control step. Returns True when nominal should resume."""
        row = self.measure(sim)
        if (not self.entered) or self.exited:
            return False
        stable = (
            row["nL"] > 0
            and row["nR"] > 0
            and row["e_pos"] <= EXIT_POSITION_THRESHOLD_M
            and row["e_rot"] <= EXIT_ORIENTATION_THRESHOLD_RAD
            and row["vrel"] <= EXIT_RELATIVE_SPEED_THRESHOLD_MPS
        )
        self.exit_run = self.exit_run + 1 if stable else 0
        if self.exit_run >= EXIT_STABILITY_STEPS:
            self.exited = True
            self.t_exit = row["t"]
            return True
        return False
