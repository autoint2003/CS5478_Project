"""Offline privileged closed-loop grip force / slip controllers (legal a[6] only).

Uses mj_contactForce. Does not alter actuators, friction, or runtime observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from controllers.recovery.action_map import AIRBORNE_V3, W_MAX, grip_tau, normalize_action
from envs.ballistic_impact import PAIR_MU
from envs.outcomes.physical_recovery import physical_pack
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT

DT = 0.02


def contact_forces(sim) -> dict[str, Any]:
    obj = int(sim.ids.object_body)
    left, right = int(sim.ids.left_body), int(sim.ids.right_body)
    fn_L = ft_L = fn_R = ft_R = 0.0
    nL = nR = 0
    contacts = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        if other not in (left, right):
            continue
        force = np.zeros(6, float)
        mujoco.mj_contactForce(sim.model, sim.data, i, force)
        fr = np.asarray(c.frame, float).reshape(3, 3)
        fn = abs(float(force[0]))
        ft = float(np.linalg.norm(force[1:3]))
        side = "L" if other == left else "R"
        contacts.append(
            {
                "side": side,
                "pos": np.asarray(c.pos, float).tolist(),
                "fn": fn,
                "ft": ft,
                "normal_world": fr[0].tolist(),
            }
        )
        if side == "L":
            nL += 1
            fn_L += fn
            ft_L += ft
        else:
            nR += 1
            fn_R += fn
            ft_R += ft
    return {
        "fn_L": fn_L,
        "fn_R": fn_R,
        "ft_L": ft_L,
        "ft_R": ft_R,
        "fn_tot": fn_L + fn_R,
        "ft_tot": ft_L + ft_R,
        "nL": nL,
        "nR": nR,
        "bilateral": nL > 0 and nR > 0,
        "asymmetry": abs(fn_L - fn_R),
        "contacts": contacts,
        "mu": float(PAIR_MU),
    }


def sample_state(sim) -> dict[str, Any]:
    pack = physical_pack(sim)
    m = measure(sim, U_TGT, P_TGT)
    cf = contact_forces(sim)
    p = np.asarray(m["p_c_h"], float)
    e = p - np.asarray(P_TGT, float)
    v = np.asarray(pack["v_rel_h"], float)
    g = np.asarray(pack["g_h"], float)
    return {
        "p_c_h": p.tolist(),
        "e_signed_mm": (e * 1e3).tolist(),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "v_rel_h": v.tolist(),
        "v_x": float(v[0]),
        "v_y": float(v[1]),
        "v_z": float(v[2]),
        "v_rel_norm": float(np.linalg.norm(v)),
        "w_rel_norm": float(np.linalg.norm(pack["w_rel_h"])),
        "g_h": g.tolist(),
        "g_x": float(g[0]),
        "fn_L": cf["fn_L"],
        "fn_R": cf["fn_R"],
        "fn_tot": cf["fn_tot"],
        "ft_tot": cf["ft_tot"],
        "bilateral": cf["bilateral"],
        "nL": cf["nL"],
        "nR": cf["nR"],
        "asymmetry": cf["asymmetry"],
        "contacts": cf["contacts"],
        "grip_ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float("nan"),
    }


def make_action(v_hand, w_body, grip_a: float) -> np.ndarray:
    return np.asarray(
        normalize_action(
            np.asarray(v_hand, float),
            np.asarray(w_body, float),
            grip_tau(float(grip_a)),
            AIRBORNE_V3,
        ),
        np.float32,
    )


@dataclass
class ForcePController:
    """Maps target ΣFn → legal a[6]. Higher a → higher Fn (closing)."""

    kp: float = 0.08
    a_min: float = -0.85
    a_max: float = 1.0
    da_max: float = 0.12  # per 20 ms step
    fn_filter: float = 0.35  # EMA on measured Fn
    a: float = 1.0
    fn_filt: float = 0.0
    saturated: bool = False

    def reset(self, a0: float = 1.0, fn0: float = 0.0) -> None:
        self.a = float(np.clip(a0, self.a_min, self.a_max))
        self.fn_filt = float(fn0)
        self.saturated = False

    def step(self, fn_meas: float, fn_target: float) -> tuple[float, dict]:
        self.fn_filt = (1.0 - self.fn_filter) * self.fn_filt + self.fn_filter * float(fn_meas)
        err = self.fn_filt - float(fn_target)  # positive ⇒ too much force ⇒ open (↓a)
        da = -self.kp * err
        da = float(np.clip(da, -self.da_max, self.da_max))
        a_new = float(np.clip(self.a + da, self.a_min, self.a_max))
        self.saturated = a_new in (self.a_min, self.a_max) and abs(da) > 1e-9
        self.a = a_new
        return self.a, {
            "fn_filt": self.fn_filt,
            "fn_target": float(fn_target),
            "fn_err": err,
            "da": da,
            "a": self.a,
            "saturated": self.saturated,
            "tau": float(grip_tau(self.a)),
        }


@dataclass
class SlipOuterLoop:
    """Adjusts Fn_target from relative-velocity error along a desired hand-frame axis.

    Default axis is −hand-x with v_target=−1 mm/s (validated closed-loop setting).
    Gains / hysteresis unchanged from the calibrated controller.
    """

    vx_target: float = -0.001  # m/s signed speed along slip_axis
    fn_min: float = 3.5
    fn_max: float = 12.0
    kn: float = 800.0  # N per (m/s) — unload if too slow in desire direction
    hysteresis_n: float = 0.4
    fn_target: float = 6.0
    stick_count: int = 0
    slip_count: int = 0
    # Projection axis for v_along = v·axis. Default +hand-x so v_along≡v_x and
    # vx_target=−1 mm/s matches the validated closed-loop (desire motion −x).
    # For general desire direction d, set axis = −d with the same negative target.
    slip_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def reset(self, fn0: float = 6.0, slip_axis: tuple[float, float, float] | None = None) -> None:
        self.fn_target = float(np.clip(fn0, self.fn_min, self.fn_max))
        self.stick_count = 0
        self.slip_count = 0
        if slip_axis is not None:
            a = np.asarray(slip_axis, float)
            n = float(np.linalg.norm(a))
            self.slip_axis = tuple((a / max(n, 1e-12)).tolist())

    def step(self, v_along: float, bilateral: bool) -> tuple[float, dict]:
        if not bilateral:
            # emergency firm-up
            self.fn_target = min(self.fn_max, self.fn_target + 1.5)
            return self.fn_target, {"mode": "loss_protect", "fn_target": self.fn_target}

        # Desire: v_along ≈ vx_target. err > 0 ⇒ not sliding enough along axis ⇒ unload.
        err = float(v_along) - float(self.vx_target)
        if abs(v_along) < 0.0004 and err > 0:
            self.stick_count += 1
            self.slip_count = 0
        elif abs(v_along) > 0.008:
            self.slip_count += 1
            self.stick_count = 0
        else:
            self.stick_count = max(0, self.stick_count - 1)
            self.slip_count = max(0, self.slip_count - 1)

        dfn = -self.kn * err * DT
        # hysteresis: only unload after sustained sticking; firm after sustained fast slip
        if self.stick_count >= 3:
            dfn = min(dfn, -self.hysteresis_n)
        if self.slip_count >= 2 or v_along < self.vx_target - 0.005:
            dfn = max(dfn, self.hysteresis_n)

        self.fn_target = float(np.clip(self.fn_target + dfn, self.fn_min, self.fn_max))
        return self.fn_target, {
            "mode": "track",
            "v_along": float(v_along),
            "vx": float(v_along),  # backward-compatible alias
            "vx_target": float(self.vx_target),
            "vx_err": err,
            "dfn": dfn,
            "fn_target": self.fn_target,
            "stick_count": self.stick_count,
            "slip_count": self.slip_count,
            "slip_axis": list(self.slip_axis),
        }


@dataclass
class DualLoopController:
    force: ForcePController = field(default_factory=ForcePController)
    slip: SlipOuterLoop = field(default_factory=SlipOuterLoop)
    use_slip: bool = True

    def reset(
        self,
        a0: float = 1.0,
        fn0: float = 10.0,
        fn_target: float = 6.0,
        slip_axis: tuple[float, float, float] | None = None,
    ) -> None:
        self.force.reset(a0, fn0)
        self.slip.reset(fn_target, slip_axis=slip_axis)

    def step(self, state: dict, *, fn_target_override: float | None = None) -> tuple[np.ndarray, dict]:
        if fn_target_override is not None:
            fn_tgt = float(fn_target_override)
            slip_meta = {"mode": "override", "fn_target": fn_tgt}
        elif self.use_slip:
            axis = np.asarray(self.slip.slip_axis, float)
            v = np.asarray(state.get("v_rel_h", [state["v_x"], state.get("v_y", 0.0), state.get("v_z", 0.0)]), float)
            v_along = float(np.dot(v, axis))
            fn_tgt, slip_meta = self.slip.step(v_along, state["bilateral"])
        else:
            fn_tgt = self.slip.fn_target
            slip_meta = {"mode": "force_only", "fn_target": fn_tgt}
        a, fmeta = self.force.step(state["fn_tot"], fn_tgt)
        action = make_action([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], a)
        return action, {"force": fmeta, "slip": slip_meta, "a": a, "fn_target": fn_tgt}


def tilt_actions(target_deg: float, sign: float = 1.0, rate_scale: float = 1.0) -> list:
    omega = float(W_MAX) * float(rate_scale)
    dur = float(np.clip(np.radians(target_deg) / max(omega, 1e-6), 0.1, 2.5))
    n = max(1, int(round(dur / DT)))
    a = make_action([0, 0, 0], [0, sign * omega, 0], 1.0)
    return [a] * n


def untilt_actions(n_tilt: int, sign: float = 1.0, rate_scale: float = 1.0) -> list:
    omega = float(W_MAX) * float(rate_scale)
    a = make_action([0, 0, 0], [0, -sign * omega, 0], 1.0)
    return [a] * n_tilt + [make_action([0, 0, 0], [0, 0, 0], 1.0)] * 8
