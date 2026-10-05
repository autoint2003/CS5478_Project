"""Shared impact-ball launch and lifecycle. Display and diagnostics must use this.

Does not change cylinder/robot physics, RULE, reward, or held-out eval ICs.
Ball qpos/qvel are written only at: park, one launch, park-after-impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from envs.physical_recovery import Z_AIR
from training.replay_core import (
    BALL_R,
    G_HOLD,
    disable_object_table,
    dump_ball_contacts,
    freeze,
    isolate_ball_object_only,
    park_impact_ball,
)

# Old visualization launch: extra gap beyond surface radii (NOT used for new fly-in).
LEGACY_LAUNCH_GAP_M = 0.070

STATES = (
    "PARKED",
    "READY",
    "FLYING",
    "CONTACT",
    "DEPARTED",
    "PARKED_AFTER_IMPACT",
)


def ball_pose_twist(sim) -> dict:
    q = int(sim.ball_qadr)
    d = int(sim.ball_dadr)
    return {
        "pos": np.array(sim.data.qpos[q : q + 3], float).copy(),
        "quat": np.array(sim.data.qpos[q + 3 : q + 7], float).copy(),
        "vel": np.array(sim.data.qvel[d : d + 3], float).copy(),
        "omega": np.array(sim.data.qvel[d + 3 : d + 6], float).copy(),
    }


def incoming_hand_x(sim) -> np.ndarray:
    """Same impact direction as the previous live launch: +hand-x through the object."""
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    u = Rh[:, 0].copy()
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        raise RuntimeError("hand-x direction is degenerate")
    return u / n


def surface_radii(sim) -> float:
    r_obj = float(sim.model.geom_size[int(sim.ids.object_geom)][0])
    return r_obj + float(BALL_R)


def ball_hits_cylinder(sim) -> bool:
    return any(r["kind"] == "ball-cylinder" for r in dump_ball_contacts(sim))


def ballistic_launch_state(
    sim,
    v_impact: float,
    d_launch: float,
    *,
    legacy_gap: bool = False,
) -> dict:
    """Initial free-flight state aimed at the current object center along +hand-x.

    v_impact is the intended along-track speed (CLI --impact-speed). Gravity is
    compensated so the ball center arrives at the surface point at time T.
    d_launch is an engineering free-flight distance, not a calibrated constant.
    """
    u = incoming_hand_x(sim)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    r_surf = surface_radii(sim)
    g = np.array(sim.model.opt.gravity, float)
    if legacy_gap:
        dist = r_surf + float(LEGACY_LAUNCH_GAP_M)
        p_launch = po - u * dist
        v0 = u * float(v_impact)
        T = float(LEGACY_LAUNCH_GAP_M) / max(float(v_impact), 1e-9)
        p_hit = po - u * r_surf
        return {
            "legacy": True,
            "p_target": po.copy(),
            "p_hit": p_hit,
            "p_launch": p_launch,
            "v0": v0,
            "direction": u,
            "d_launch": float(dist),
            "d_launch_user": float(LEGACY_LAUNCH_GAP_M),
            "T_hit_s": T,
            "r_surf": r_surf,
            "v_impact_requested": float(v_impact),
            "gravity": g.copy(),
        }
    d_launch = float(d_launch)
    if d_launch <= r_surf + 0.05:
        raise ValueError(f"d_launch={d_launch} too small vs surface radii {r_surf}")
    p_target = po.copy()
    p_hit = p_target - u * r_surf
    p_launch = p_target - u * d_launch
    dist_hit = float(np.linalg.norm(p_hit - p_launch))
    v_in = float(v_impact)
    T = dist_hit / max(v_in, 1e-9)
    v0 = (p_hit - p_launch) / T - 0.5 * g * T
    return {
        "legacy": False,
        "p_target": p_target,
        "p_hit": p_hit,
        "p_launch": p_launch,
        "v0": v0,
        "direction": u,
        "d_launch": d_launch,
        "d_launch_user": d_launch,
        "T_hit_s": T,
        "r_surf": r_surf,
        "v_impact_requested": v_in,
        "gravity": g.copy(),
        "v0_speed": float(np.linalg.norm(v0)),
    }


def apply_ball_free_state(sim, pos: np.ndarray, vel: np.ndarray) -> None:
    q = int(sim.ball_qadr)
    d = int(sim.ball_dadr)
    sim.data.qpos[q : q + 3] = np.asarray(pos, float).reshape(3)
    sim.data.qpos[q + 3 : q + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    sim.data.qvel[d : d + 3] = np.asarray(vel, float).reshape(3)
    sim.data.qvel[d + 3 : d + 6] = 0.0
    sim.data.qacc[d : d + 6] = 0.0
    sim.data.qacc_warmstart[d : d + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


@dataclass
class ImpactBallDriver:
    """PARKED → FLYING → CONTACT → DEPARTED → PARKED_AFTER_IMPACT."""

    sim: object
    v_impact: float = 9.0
    d_launch: float = 0.30
    park_clear_steps: int = 10
    park_sep_m: float = 0.04
    legacy: bool = False
    state: str = "PARKED"
    launch_meta: dict = field(default_factory=dict)
    t_launch: float | None = None
    t_first_contact: float | None = None
    t_last_contact: float | None = None
    t_park: float | None = None
    n_qpos_writes: int = 0
    n_park_calls: int = 0
    pre_contact_vel: np.ndarray | None = None
    pre_contact_pos: np.ndarray | None = None
    pre_contact_omega: np.ndarray | None = None
    pre_contact_cyl: dict | None = None
    last_free_vel: np.ndarray | None = None
    last_free_pos: np.ndarray | None = None
    last_free_omega: np.ndarray | None = None
    last_free_cyl: dict | None = None
    clear_steps: int = 0
    pos_hist: list = field(default_factory=list)

    def _write_count(self) -> None:
        self.n_qpos_writes += 1

    def park_now(self, after_impact: bool) -> None:
        park_impact_ball(self.sim)
        self.n_park_calls += 1
        self._write_count()
        self.state = "PARKED_AFTER_IMPACT" if after_impact else "PARKED"
        if after_impact:
            self.t_park = float(self.sim.data.time)

    def keep_parked(self) -> None:
        if self.state not in ("PARKED", "READY"):
            raise RuntimeError(f"keep_parked in state {self.state}")
        self.sim.park_ball()
        self.n_park_calls += 1
        self._write_count()
        self.state = "PARKED"

    def launch(self) -> dict:
        if self.state not in ("PARKED", "READY"):
            raise RuntimeError(f"launch from {self.state}")
        meta = ballistic_launch_state(
            self.sim, self.v_impact, self.d_launch, legacy_gap=self.legacy
        )
        apply_ball_free_state(self.sim, meta["p_launch"], meta["v0"])
        self._write_count()
        self.launch_meta = meta
        self.t_launch = float(self.sim.data.time)
        self.state = "FLYING"
        st = ball_pose_twist(self.sim)
        self.last_free_pos = st["pos"]
        self.last_free_vel = st["vel"]
        self.last_free_omega = st["omega"]
        return meta

    def note_after_step(self) -> None:
        """Call once after each mj_step during FLYING/CONTACT/DEPARTED. No qpos writes."""
        if self.state not in ("FLYING", "CONTACT", "DEPARTED"):
            return
        st = ball_pose_twist(self.sim)
        self.pos_hist.append(st["pos"].copy())
        hit = ball_hits_cylinder(self.sim)
        if hit:
            if self.state == "FLYING":
                self.pre_contact_vel = (
                    None if self.last_free_vel is None else self.last_free_vel.copy()
                )
                self.pre_contact_pos = (
                    None if self.last_free_pos is None else self.last_free_pos.copy()
                )
                self.pre_contact_omega = (
                    None if self.last_free_omega is None else self.last_free_omega.copy()
                )
                self.pre_contact_cyl = self.last_free_cyl
                self.t_first_contact = float(self.sim.data.time)
                self.state = "CONTACT"
            self.t_last_contact = float(self.sim.data.time)
            self.clear_steps = 0
        else:
            self.last_free_vel = st["vel"].copy()
            self.last_free_pos = st["pos"].copy()
            self.last_free_omega = st["omega"].copy()
            j = int(self.sim.ids.object_jnt)
            adr = int(self.sim.model.jnt_dofadr[j])
            po = np.array(self.sim.data.xpos[self.sim.ids.object_body], float)
            vo = np.array(self.sim.data.qvel[adr : adr + 6], float)
            self.last_free_cyl = {"pos": po.copy(), "twist6": vo.copy()}
            if self.state == "CONTACT":
                self.clear_steps += 1
                sep = float(np.linalg.norm(st["pos"] - po)) - surface_radii(self.sim)
                if self.clear_steps >= self.park_clear_steps and sep >= self.park_sep_m:
                    self.state = "DEPARTED"

    def maybe_park_after_impact(self) -> bool:
        if self.state != "DEPARTED":
            return False
        self.park_now(after_impact=True)
        return True

    def measured(self) -> dict:
        v = self.pre_contact_vel
        return {
            "state": self.state,
            "t_launch": self.t_launch,
            "t_first_contact": self.t_first_contact,
            "t_last_contact": self.t_last_contact,
            "t_park": self.t_park,
            "n_qpos_writes": self.n_qpos_writes,
            "n_park_calls": self.n_park_calls,
            "launch": {
                k: (val.tolist() if isinstance(val, np.ndarray) else val)
                for k, val in self.launch_meta.items()
            },
            "pre_contact_vel": None if v is None else np.asarray(v, float).tolist(),
            "pre_contact_speed": None if v is None else float(np.linalg.norm(v)),
            "pre_contact_pos": None
            if self.pre_contact_pos is None
            else np.asarray(self.pre_contact_pos, float).tolist(),
            "pre_contact_omega": None
            if self.pre_contact_omega is None
            else np.asarray(self.pre_contact_omega, float).tolist(),
            "dt_launch_to_contact": None
            if self.t_launch is None or self.t_first_contact is None
            else float(self.t_first_contact - self.t_launch),
        }


def run_to_airborne_hold(sim, gains, hold_s: float = 0.20, lift_extra: float = 0.25, timeout: float = 12.0) -> None:
    """Centered airborne freeze-hold with ball parked. Shared by viz tests/diagnostics."""
    from training.impact_visualization_utils import tick_vw_park

    extra = 0.0
    dt = float(sim.model.opt.timestep)
    while True:
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
        sim.park_ball()
        z = float(sim.data.xpos[sim.ids.object_body][2])
        if sim.fsm.phase == "lift" and sim.captured and z >= Z_AIR:
            extra += dt
            if extra >= lift_extra:
                freeze(sim)
                disable_object_table(sim)
                isolate_ball_object_only(sim)
                break
        if float(sim.data.time) > timeout:
            raise RuntimeError("failed to reach airborne centered hold")
    n = max(1, int(round(hold_s / dt)))
    for _ in range(n):
        tick_vw_park(sim, np.zeros(3), np.zeros(3), G_HOLD, gains, park=False)
        sim.park_ball()


def fly_until_parked(sim, driver: ImpactBallDriver, gains, timeout: float = 1.0) -> None:
    from training.impact_visualization_utils import tick_vw_park

    t0 = float(sim.data.time)
    while float(sim.data.time) - t0 < timeout:
        if driver.state == "PARKED_AFTER_IMPACT":
            return
        tick_vw_park(sim, np.zeros(3), np.zeros(3), G_HOLD, gains, park=False)
        driver.note_after_step()
        driver.maybe_park_after_impact()
    if driver.state != "PARKED_AFTER_IMPACT":
        raise RuntimeError(f"ball not parked after flight, state={driver.state}")

