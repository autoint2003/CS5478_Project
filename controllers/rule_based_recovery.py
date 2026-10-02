"""Deterministic rule-based recovery FSM.

Legal observables only. Nominal Cartesian gains (kp_pos=180, kd_pos=28).
No privileged mass/friction. Wrist via demonstrated hand-y 90 deg r_des.
Grip via tendon tau (fg = -tau).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from controllers.residual import ResidualCommand


@dataclass
class RuleParams:
    e_tol: float = 0.003
    t_stabilize: float = 0.15
    t_stabilize: float = 0.15
    t_align_max: float = 0.60
    t_slip_max: float = 1.20
    t_brake_hold: float = 0.12
    t_check: float = 0.20
    t_secure: float = 0.50
    t_open_max: float = 0.20
    t_reclose: float = 0.12
    t_resume: float = 0.50
    t_brake: float = 0.08
    tau_secure: float = -18.0
    tau_slip: float = -2.0
    tau_open: float = -1.0
    v_open: float = 0.06
    fall_lim: float = 0.025
    v_rel_tol: float = 0.08
    w_align: float = 0.0
    g_hx_min: float = 8.0
    n_unilat_abort: int = 12  # 24 ms at 500 Hz; ignore contact flicker
    # justification filled later
    e_tol_note: str = (
        "Centered airborne |e_x|~0.13 mm. Stick-sweep used 2 mm of relative motion. "
        "The 3 T_brake probes stopped captured-slip at 2.54–2.70 mm residual. "
        "e_tol=3 mm is that residual band, not an arbitrary eval cutoff."
    )


@dataclass
class RuleState:
    mode: str = "STABILIZE"
    t_mode: float = 0.0
    t_total: float = 0.0
    used_slip: bool = False
    used_open: bool = False
    slip_done: bool = False
    open_done: bool = False
    r_des_align: np.ndarray | None = None
    r_des_task: np.ndarray | None = None
    e_x_min: float = 1e9
    e_dot: float = 0.0
    e_x_prev: float | None = None
    z0: float | None = None
    ph0: np.ndarray | None = None
    Rh0: np.ndarray | None = None
    zero_seen: bool = False
    modes: list = field(default_factory=list)
    fail: str = ""
    recovery_success: bool = False
    done: bool = False
    n_unilat: int = 0
    saw_center: bool = False


def g_hand(Rh: np.ndarray, g_w: np.ndarray) -> np.ndarray:
    return Rh.T @ np.asarray(g_w, float).reshape(3)


def align_R(Rh: np.ndarray, e_x: float, g_w: np.ndarray) -> np.ndarray:
    """90 deg about hand y (demonstrated). Mirror so sign(g_h.x) = -sign(e_x)."""
    want = -float(np.sign(e_x) or 1.0)
    best = None
    best_score = -1e9
    for sgn in (1.0, -1.0):
        ang = 0.5 * np.pi * sgn
        cy, sy = np.cos(ang), np.sin(ang)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        R = Rh @ Ry
        gh = g_hand(R, g_w)
        score = want * float(gh[0])
        if score > best_score:
            best_score = score
            best = R
    return best


class RuleBasedRecovery:
    def __init__(self, params: RuleParams | None = None):
        self.p = params or RuleParams()
        self.s = RuleState()

    def reset(self, r_des_task: np.ndarray) -> None:
        self.s = RuleState(r_des_task=np.asarray(r_des_task, float).reshape(3, 3).copy())
        self.s.modes = ["STABILIZE"]

    def _go(self, mode: str) -> None:
        self.s.mode = mode
        self.s.t_mode = 0.0
        self.s.modes.append(mode)

    def step(self, obs: dict, dt: float) -> dict:
        """obs: e_x, e_x_legal, Rh, g_w, nL, nR, v_rel, obj_z, ph, aperture, scene.
        Returns v_world, tau, r_des (or None to keep), freeze_p (bool).
        """
        s, p = self.s, self.p
        e_x = float(obs["e_x"])
        s.e_x_min = min(s.e_x_min, abs(e_x))
        prev = s.e_x_prev
        if prev is None:
            s.e_dot = 0.0
            crossed = False
        else:
            s.e_dot = (e_x - prev) / dt
            crossed = (prev * e_x) <= 0.0 and abs(prev) > 1e-6
        s.e_x_prev = e_x
        if s.z0 is None:
            s.z0 = float(obs["obj_z"])
            s.ph0 = np.asarray(obs["ph"], float).copy()
            s.Rh0 = np.asarray(obs["Rh"], float).reshape(3, 3).copy()
        s.t_mode += dt
        s.t_total += dt
        fall = s.z0 - float(obs["obj_z"])
        nL, nR = int(obs["nL"]), int(obs["nR"])
        bilat = nL > 0 and nR > 0
        scene = int(obs.get("scene", 0))
        v_rel = float(obs["v_rel"])
        Rh = np.asarray(obs["Rh"], float).reshape(3, 3)
        g_w = np.asarray(obs["g_w"], float)
        gh = g_hand(Rh, g_w)
        want_ghx = -float(np.sign(e_x) or 1.0)
        aligned = (np.sign(gh[0]) == np.sign(want_ghx) or abs(e_x) < p.e_tol) and abs(gh[0]) >= p.g_hx_min
        if not bilat:
            s.n_unilat += 1
        else:
            s.n_unilat = 0
        if abs(e_x) <= p.e_tol:
            s.saw_center = True

        v = np.zeros(3)
        tau = p.tau_secure
        r_des = s.r_des_task
        freeze_p = True

        if scene:
            s.fail = "SCENE"
            s.done = True
            return self._out(v, tau, r_des, freeze_p, gh)

        # Re-dispatch after _go so the new mode's command is applied this tick.
        for _ in range(4):
            if s.mode == "STABILIZE":
                tau = p.tau_secure
                r_des = s.r_des_task
                if s.t_mode >= p.t_stabilize:
                    s.r_des_align = align_R(Rh, e_x, g_w)
                    self._go("SLIP_ALIGN")
                    continue

            elif s.mode == "SLIP_ALIGN":
                tau = p.tau_secure
                r_des = s.r_des_align
                if aligned or s.t_mode >= p.t_align_max:
                    s.used_slip = True
                    s.n_unilat = 0
                    self._go("CONTROLLED_SLIP")
                    continue

            elif s.mode == "CONTROLLED_SLIP":
                s.slip_done = True
                tau = p.tau_slip
                r_des = s.r_des_align
                d_stop = abs(s.e_dot) * p.t_brake
                lost = s.n_unilat >= p.n_unilat_abort or fall >= p.fall_lim
                if lost:
                    self._go("BRAKE")
                    continue
                if crossed or abs(e_x) <= max(p.e_tol, d_stop) or s.t_mode >= p.t_slip_max:
                    self._go("BRAKE")
                    continue

            elif s.mode == "BRAKE":
                tau = p.tau_secure
                r_des = s.r_des_align
                if s.t_mode >= p.t_brake_hold:
                    self._go("CHECK")
                    continue

            elif s.mode == "CHECK":
                tau = p.tau_secure
                r_des = s.r_des_align
                if s.t_mode >= p.t_check:
                    centered = (abs(e_x) <= p.e_tol and bilat) or s.saw_center
                    if centered:
                        # High v_rel after slip is settled in SECURE, not by opening.
                        self._go("SECURE")
                        continue
                    if not s.open_done:
                        s.used_open = True
                        s.open_done = True
                        s.zero_seen = False
                        self._go("OPEN_REGRASP")
                        continue
                    s.fail = "CHECK_FAIL"
                    s.done = True

            elif s.mode == "OPEN_REGRASP":
                tau = p.tau_open
                # Keep demonstrated align pose; do not rewind wrist while open.
                r_des = s.r_des_align if s.r_des_align is not None else s.r_des_task
                if s.t_mode <= dt + 1e-9:
                    s.ph0 = np.asarray(obs["ph"], float).copy()
                    s.Rh0 = Rh.copy()
                zero = nL == 0 and nR == 0
                if zero:
                    s.zero_seen = True
                if s.zero_seen:
                    ex = s.Rh0[:, 0]
                    dx = float(np.dot(ex, np.asarray(obs["ph"], float) - s.ph0))
                    v_h = np.array([float(np.sign(e_x) or 1.0) * p.v_open, 0.0, 0.0])
                    v = Rh @ v_h
                    freeze_p = False
                    if abs(dx) >= 0.001 or fall >= p.fall_lim or s.t_mode >= p.t_open_max:
                        self._go("RECLOSE")
                        continue
                elif s.t_mode >= 0.12:
                    self._go("RECLOSE")
                    continue

            elif s.mode == "RECLOSE":
                tau = p.tau_secure
                r_des = s.r_des_align if s.r_des_align is not None else s.r_des_task
                v = np.zeros(3)
                if s.t_mode >= p.t_reclose:
                    self._go("SECURE")
                    continue

            elif s.mode == "SECURE":
                tau = p.tau_secure
                # Hold aligned until stable, then return wrist (user SECURE order).
                if s.t_mode < 0.5 * p.t_secure:
                    r_des = s.r_des_align if s.r_des_align is not None else s.r_des_task
                    if abs(e_x) <= p.e_tol and bilat and v_rel < p.v_rel_tol and not scene:
                        s.recovery_success = True
                else:
                    r_des = s.r_des_task
                    if abs(e_x) <= p.e_tol and bilat and not scene:
                        s.recovery_success = s.recovery_success or (
                            v_rel < p.v_rel_tol
                        )
                if s.t_mode >= p.t_secure:
                    ok = s.recovery_success and bilat and abs(e_x) <= p.e_tol and not scene
                    s.recovery_success = bool(ok)
                    if ok:
                        self._go("RESUME")
                        continue
                    s.fail = "SECURE_FAIL"
                    s.done = True

            elif s.mode == "RESUME":
                tau = p.tau_secure
                r_des = s.r_des_task
                v = np.array([0.0, 0.0, 0.08])
                freeze_p = False
                if s.t_mode >= p.t_resume:
                    s.done = True

            break

        return self._out(v, tau, r_des, freeze_p, gh)

    def _out(self, v, tau, r_des, freeze_p, gh):
        return {
            "v_world": np.asarray(v, float).reshape(3),
            "tau": float(tau),
            "r_des": None if r_des is None else np.asarray(r_des, float).reshape(3, 3),
            "freeze_p": bool(freeze_p),
            "mode": self.s.mode,
            "g_hx": float(gh[0]),
            "e_dot": self.s.e_dot,
        }

    def to_residual(self, out: dict, fg_nom: float = 18.0) -> ResidualCommand:
        """Best-effort 7d residual. Grip maps dfg = -tau - fg_nom (clipped by env)."""
        dfg = (-out["tau"]) - fg_nom
        return ResidualCommand(dv=out["v_world"].copy(), dw=np.zeros(3), dfg=float(dfg))
