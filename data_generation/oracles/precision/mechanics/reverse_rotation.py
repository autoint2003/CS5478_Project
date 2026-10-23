"""Reverse the verified finger-support → vertical path under hold-mode grip.

Forward (frozen): Stage A (−wz pad-up) → Stage B (fingertips-up) → axial slip.
Return: B⁻¹ (tip back to pad-up) → A⁻¹ (unspin to fingertips-down) with NO intentional slip.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from controllers.recovery.action_map import W_MAX
from data_generation.oracles.precision.mechanics.finger_vertical import (
    U_OPEN_HAND,
    Z_ABORT,
    opening_alignment,
    vertical_gate_report,
)
from data_generation.oracles.precision.mechanics.finger_surface import (
    finger_inner_normals_world,
    support_force_decomposition,
)
from data_generation.oracles.precision.mechanics.dual_loop import (
    DT,
    ForcePController,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.corrections.orientation import _height_hold_v_hand
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R

UP = np.array([0.0, 0.0, 1.0], float)
G_WORLD = np.array([0.0, 0.0, -9.81], float)


def rotmat_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Rodrigues inverse: R → (unit axis, angle rad in [0, π])."""
    R = np.asarray(R, float)
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    ang = float(np.arccos(tr))
    if ang < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if ang > np.pi - 1e-6:
        # near 180°: pick largest diagonal eigenvector approx
        w, v = np.linalg.eigh(0.5 * (R + R.T))
        axis = v[:, int(np.argmax(w))]
        return axis / max(float(np.linalg.norm(axis)), 1e-12), ang
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return axis, ang


def omega_body_toward_R(sim, R_target: np.ndarray, rate_scale: float) -> tuple[np.ndarray, float]:
    """Legal r_des-frame ω that body-right-multiplies Rh toward R_target.

    Kinematic model (same as orientation_planner): Rh_new ≈ Rh @ exp([ω_body×]).
    Actions: w_world = R_des @ w_body.
    """
    Rh = np.array(hand_R(sim), float, copy=True)
    Rd = np.array(sim.fsm.r_des, float, copy=True).reshape(3, 3)
    R_tgt = np.asarray(R_target, float).reshape(3, 3)
    # Desired body delta: Rh @ R_delta ≈ R_tgt ⇒ R_delta ≈ Rh.T @ R_tgt
    R_delta = Rh.T @ R_tgt
    # orthonormalize lightly
    u, _, vt = np.linalg.svd(R_delta)
    R_delta = u @ vt
    if np.linalg.det(R_delta) < 0:
        u[:, -1] *= -1
        R_delta = u @ vt
    axis_body, ang = rotmat_to_axis_angle(R_delta)
    if ang < 1e-4:
        return np.zeros(3), 0.0
    # Command body rate; map through r_des so physical ≈ Rh rate when Rd≈Rh
    omega = float(W_MAX) * float(rate_scale)
    # Prefer expressing ω in r_des: if we want ω_hand_body, and w_world = Rd @ w_body,
    # with Rd≈Rh: w_body ≈ Rh.T @ (Rh @ (axis_body * omega)) = axis_body * omega when aligned.
    w_hand = axis_body * omega  # rates about hand body axes
    w_world = Rh @ w_hand
    w_body = Rd.T @ w_world
    return w_body, float(np.degrees(ang))


def snapshot_rich(sim, *, e_ref: np.ndarray | None = None) -> dict[str, Any]:
    s = sample_state(sim)
    e = np.asarray(s["e_signed_mm"], float)
    oa = opening_alignment(sim)
    kin = finger_inner_normals_world(sim)
    sup = support_force_decomposition(sim)
    Rh = np.array(hand_R(sim), float, copy=True)
    z = float(sim.data.xpos[sim.ids.object_body][2])
    out = {
        "e_signed_mm": e.tolist(),
        "theta_deg": s["e_theta_deg"],
        "p_c_h": s["p_c_h"],
        "fn_L": s["fn_L"],
        "fn_R": s["fn_R"],
        "fn_tot": s["fn_tot"],
        "bilateral": s["bilateral"],
        "v_rel_h": s["v_rel_h"],
        "v_rel_norm": s["v_rel_norm"],
        "obj_z": z,
        "world_height_margin": z - Z_ABORT,
        "hand_R": Rh.tolist(),
        "qpos": np.asarray(sim.data.qpos, float).tolist(),
        "alpha_open_up_deg": oa["alpha_open_up_deg"],
        "g_dot_d_inward": oa["g_dot_d_inward"],
        "u_open_world": oa["u_open_world"],
        "left_align_up": kin["left_align_up"],
        "right_align_up": kin["right_align_up"],
        "Fn_z_L": sup["sides"]["L"]["Fn_z_normal"],
        "Fn_z_R": sup["sides"]["R"]["Fn_z_normal"],
        "Ft_z_L": sup["sides"]["L"]["Ft_z_friction"],
        "Ft_z_R": sup["sides"]["R"]["Ft_z_friction"],
    }
    if e_ref is not None:
        er = np.asarray(e_ref, float)
        out["delta_ex_mm"] = float(e[0] - er[0])
        out["delta_ey_mm"] = float(e[1] - er[1])
        out["delta_ez_mm"] = float(e[2] - er[2])
        out["delta_theta_deg"] = float(s["e_theta_deg"])  # absolute; caller may subtract
    return out


def _log_row(sim, *, phase: str, e_post: np.ndarray, th_post: float, rate: float, ang_err: float, fn_tgt: float) -> dict:
    s = sample_state(sim)
    e = np.asarray(s["e_signed_mm"], float)
    oa = opening_alignment(sim)
    kin = finger_inner_normals_world(sim)
    sup = support_force_decomposition(sim)
    return {
        "phase": phase,
        "e_signed_mm": e.tolist(),
        "theta_deg": s["e_theta_deg"],
        "delta_ex_mm": float(e[0] - e_post[0]),
        "delta_ey_mm": float(e[1] - e_post[1]),
        "delta_ez_mm": float(e[2] - e_post[2]),
        "delta_theta_deg": float(s["e_theta_deg"] - th_post),
        "fn_L": s["fn_L"],
        "fn_R": s["fn_R"],
        "fn_tot": s["fn_tot"],
        "fn_target": fn_tgt,
        "v_rel_norm": s["v_rel_norm"],
        "v_rel_h": s["v_rel_h"],
        "bilateral": s["bilateral"],
        "alpha_open_up_deg": oa["alpha_open_up_deg"],
        "g_dot_u_open": float(np.dot(G_WORLD, np.asarray(oa["u_open_world"], float))),
        "g_dot_d_inward": oa["g_dot_d_inward"],
        "left_align_up": kin["left_align_up"],
        "right_align_up": kin["right_align_up"],
        "Fn_z_L": sup["sides"]["L"]["Fn_z_normal"],
        "Fn_z_R": sup["sides"]["R"]["Fn_z_normal"],
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "world_height_margin": float(sim.data.xpos[sim.ids.object_body][2]) - Z_ABORT,
        "rate_scale": rate,
        "ang_err_deg": ang_err,
    }


class HoldModeGrip:
    """Force regulation without intentional slip; boost FN if relative motion rises."""

    def __init__(
        self,
        *,
        fn_hold: float = 13.0,
        fn_min: float = 10.0,
        fn_max: float = 16.0,
        v_slip_boost: float = 0.0015,
        kp: float = 0.14,
        da_max: float = 0.15,
        a0: float = 0.85,
    ):
        self.fn_hold = float(fn_hold)
        self.fn_min = float(fn_min)
        self.fn_max = float(fn_max)
        self.v_slip_boost = float(v_slip_boost)
        self.ctl = ForcePController(kp=kp, da_max=da_max, a_min=-0.5, a_max=1.0)
        self._fn_tgt = float(fn_hold)
        self._a0 = float(a0)
        self._inited = False

    def reset(self, fn0: float) -> None:
        self.ctl.reset(self._a0, fn0)
        self._fn_tgt = self.fn_hold
        self._inited = True

    def step(self, state: dict) -> tuple[float, float, dict]:
        if not self._inited:
            self.reset(state["fn_tot"])
        v = float(state["v_rel_norm"])
        if not state["bilateral"]:
            self._fn_tgt = min(self.fn_max, self._fn_tgt + 1.0)
        elif v > self.v_slip_boost:
            self._fn_tgt = min(self.fn_max, self._fn_tgt + 0.35)
        elif v < 0.5 * self.v_slip_boost:
            self._fn_tgt = max(self.fn_min, 0.95 * self._fn_tgt + 0.05 * self.fn_hold)
        a, meta = self.ctl.step(state["fn_tot"], self._fn_tgt)
        return a, self._fn_tgt, {**meta, "hold_fn_tgt": self._fn_tgt, "v_rel_norm": v}


def track_orientation_waypoints(
    env,
    blob: dict,
    waypoints: list[np.ndarray],
    *,
    phase: str,
    rate_scale: float = 0.12,
    fn_hold: float = 13.0,
    ang_tol_deg: float = 3.0,
    max_steps_per_wp: int = 80,
    e_post: np.ndarray | None = None,
    th_post: float | None = None,
    contact_aware: bool = False,
    ez_abort_mm: float = 6.0,
    dx_abort_mm: float = 4.0,
    v_slow: float = 0.002,
) -> dict[str, Any]:
    """Closed-loop track a sequence of hand_R targets under hold-mode grip."""
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    if e_post is None:
        e_post = np.asarray(s0["e_signed_mm"], float)
    else:
        e_post = np.asarray(e_post, float)
    if th_post is None:
        th_post = float(s0["e_theta_deg"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    hold = HoldModeGrip(fn_hold=fn_hold)
    hold.reset(s0["fn_tot"])
    actions, series = [], []
    lost = 0
    rate = float(rate_scale)
    phase_ez0 = float(e_post[2])

    for wi, R_tgt in enumerate(waypoints):
        R_tgt = np.asarray(R_tgt, float).reshape(3, 3)
        for _ in range(max_steps_per_wp):
            w_body, ang_err = omega_body_toward_R(env.sim, R_tgt, rate)
            st = sample_state(env.sim)
            if contact_aware:
                if st["v_rel_norm"] > v_slow:
                    rate = max(0.04, rate * 0.85)
                    # pause rotation briefly while boosting grip
                    w_body = np.zeros(3)
                else:
                    rate = min(float(rate_scale), rate * 1.02)
            a_g, fn_tgt, _ = hold.step(st)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), w_body, a_g)
            step_legal(env.sim, a)
            actions.append(a.copy())
            failed, lost = recovery_failed(env.sim, blob["ref"], lost)
            st2 = sample_state(env.sim)
            e2 = np.asarray(st2["e_signed_mm"], float)
            z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
            if len(actions) % 4 == 0 or ang_err < ang_tol_deg:
                row = _log_row(
                    env.sim,
                    phase=f"{phase}_wp{wi}",
                    e_post=e_post,
                    th_post=th_post,
                    rate=rate,
                    ang_err=ang_err,
                    fn_tgt=fn_tgt,
                )
                row["t"] = len(actions) * DT
                row["waypoint_idx"] = wi
                series.append(row)
            if failed or latched(env.sim) or not st2["bilateral"]:
                return {
                    "ok": False,
                    "reason": "contact_loss",
                    "phase": phase,
                    "waypoint_idx": wi,
                    "actions": actions,
                    "series": series,
                    "capture": take_capture(env),
                    "snapshot": snapshot_rich(env.sim, e_ref=e_post),
                }
            if z < Z_ABORT:
                return {
                    "ok": False,
                    "reason": "world_height",
                    "phase": phase,
                    "actions": actions,
                    "series": series,
                    "capture": take_capture(env),
                    "snapshot": snapshot_rich(env.sim, e_ref=e_post),
                }
            if abs(float(e2[0]) - float(e_post[0])) > dx_abort_mm:
                return {
                    "ok": False,
                    "reason": "lateral_drift",
                    "phase": phase,
                    "actions": actions,
                    "series": series,
                    "capture": take_capture(env),
                    "snapshot": snapshot_rich(env.sim, e_ref=e_post),
                }
            if abs(float(e2[2]) - phase_ez0) > ez_abort_mm:
                return {
                    "ok": False,
                    "reason": "axial_regression",
                    "phase": phase,
                    "actions": actions,
                    "series": series,
                    "capture": take_capture(env),
                    "snapshot": snapshot_rich(env.sim, e_ref=e_post),
                    "ez_now": float(e2[2]),
                    "ez_post_slip": phase_ez0,
                }
            if ang_err <= ang_tol_deg:
                break
        else:
            # waypoint timeout — continue to next if reasonably close
            _, ang_err = omega_body_toward_R(env.sim, R_tgt, rate)
            if ang_err > 2.0 * ang_tol_deg:
                return {
                    "ok": False,
                    "reason": "waypoint_timeout",
                    "phase": phase,
                    "waypoint_idx": wi,
                    "ang_err_deg": ang_err,
                    "actions": actions,
                    "series": series,
                    "capture": take_capture(env),
                    "snapshot": snapshot_rich(env.sim, e_ref=e_post),
                }

    # settle
    for _ in range(10):
        st = sample_state(env.sim)
        a_g, fn_tgt, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
    sf = sample_state(env.sim)
    snap = snapshot_rich(env.sim, e_ref=e_post)
    return {
        "ok": bool(sf["bilateral"]),
        "reason": "ok" if sf["bilateral"] else "settle_loss",
        "phase": phase,
        "actions": actions,
        "series": series,
        "capture": take_capture(env),
        "snapshot": snap,
        "e_final": sf["e_signed_mm"],
        "theta_final": sf["e_theta_deg"],
        "delta_ez_mm": float(np.asarray(sf["e_signed_mm"], float)[2] - e_post[2]),
        "delta_ex_mm": float(np.asarray(sf["e_signed_mm"], float)[0] - e_post[0]),
    }


def build_reverse_waypoints(
    forward_waypoints: list[dict],
    *,
    phase_from: str,
    phase_to: str,
    stride: int = 1,
) -> list[np.ndarray]:
    """Take forward Rh samples in [phase_from, phase_to] and reverse them."""
    mats = []
    for w in forward_waypoints:
        if w["phase"] in (phase_from, phase_to) or (
            phase_from == "A" and w["phase"].startswith("A")
        ) or (phase_from == "B" and w["phase"].startswith("B")):
            mats.append(np.asarray(w["hand_R"], float))
    if not mats:
        # filter by exact phase tags
        mats = [np.asarray(w["hand_R"], float) for w in forward_waypoints if w["phase"] == phase_from]
    mats = mats[:: max(1, stride)]
    return list(reversed(mats))


def record_forward_with_waypoints(
    env,
    blob: dict,
    *,
    support_axis: str = "-wz",
    support_angle_deg: float = 90.0,
    support_rate: float = 0.18,
    tip_rate: float = 0.15,
    wp_every: int = 5,
    run_slip_fn: Callable | None = None,
) -> dict[str, Any]:
    """Execute frozen forward path while logging hand_R waypoints + post-slip hold state."""
    from data_generation.oracles.precision.mechanics.finger_vertical import (
        stage_finger_support,
        stage_fingertips_up,
    )

    restore_capture(env, blob["capture"])
    start_snap = snapshot_rich(env.sim)
    waypoints: list[dict] = []
    waypoints.append(
        {
            "phase": "start",
            "hand_R": start_snap["hand_R"],
            "e_signed_mm": start_snap["e_signed_mm"],
            "alpha_open_up_deg": start_snap["alpha_open_up_deg"],
        }
    )

    # --- Stage A with waypoint logging via action replay instrumentation ---
    a = stage_finger_support(
        env, blob, axis_name=support_axis, angle_deg=support_angle_deg, rate_scale=support_rate
    )
    if not a.get("ok"):
        return {"ok": False, "reason": f"stage_A_failed:{a.get('reason')}"}

    # Replay A actions from start to sample dense Rh (stage_finger_support already ran;
    # re-run from blob to log)
    restore_capture(env, blob["capture"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    ctl = ForcePController(kp=0.12, da_max=0.10)
    ctl.reset(1.0, sample_state(env.sim)["fn_tot"])
    for i, act in enumerate(a["actions"]):
        step_legal(env.sim, act)
        if i % wp_every == 0 or i == len(a["actions"]) - 1:
            sn = snapshot_rich(env.sim)
            waypoints.append(
                {
                    "phase": "A",
                    "step": i + 1,
                    "hand_R": sn["hand_R"],
                    "e_signed_mm": sn["e_signed_mm"],
                    "alpha_open_up_deg": sn["alpha_open_up_deg"],
                    "left_align_up": sn["left_align_up"],
                    "Fn_z_L": sn["Fn_z_L"],
                    "Fn_z_R": sn["Fn_z_R"],
                }
            )
    mid_cap = take_capture(env)
    mid_snap = snapshot_rich(env.sim)
    waypoints.append(
        {
            "phase": "A_end",
            "hand_R": mid_snap["hand_R"],
            "e_signed_mm": mid_snap["e_signed_mm"],
            "alpha_open_up_deg": mid_snap["alpha_open_up_deg"],
        }
    )

    mid = {"capture": mid_cap, "ref": blob["ref"]}
    b = stage_fingertips_up(env, mid, rate_scale=tip_rate)
    if not b.get("ok"):
        return {"ok": False, "reason": f"stage_B_failed:{b.get('reason')}", "waypoints": waypoints}

    restore_capture(env, mid_cap)
    for i, act in enumerate(b["actions"]):
        step_legal(env.sim, act)
        if i % wp_every == 0 or i == len(b["actions"]) - 1:
            sn = snapshot_rich(env.sim)
            waypoints.append(
                {
                    "phase": "B",
                    "step": i + 1,
                    "hand_R": sn["hand_R"],
                    "e_signed_mm": sn["e_signed_mm"],
                    "alpha_open_up_deg": sn["alpha_open_up_deg"],
                    "left_align_up": sn["left_align_up"],
                    "Fn_z_L": sn["Fn_z_L"],
                    "Fn_z_R": sn["Fn_z_R"],
                }
            )
    vert_cap = take_capture(env)
    vert_snap = snapshot_rich(env.sim)
    waypoints.append(
        {
            "phase": "B_end",
            "hand_R": vert_snap["hand_R"],
            "e_signed_mm": vert_snap["e_signed_mm"],
            "alpha_open_up_deg": vert_snap["alpha_open_up_deg"],
        }
    )

    slip_actions = []
    post_slip = None
    if run_slip_fn is not None:
        slip_res = run_slip_fn(env, {"capture": vert_cap, "ref": blob["ref"]}, vert_snap)
        if not slip_res.get("ok"):
            return {
                "ok": False,
                "reason": f"slip_failed:{slip_res.get('reason') or slip_res.get('abort_reason')}",
                "waypoints": waypoints,
            }
        slip_actions = list(slip_res.get("actions") or [])
        post_slip = slip_res
        sn = snapshot_rich(env.sim)
        waypoints.append(
            {
                "phase": "post_slip_hold",
                "hand_R": sn["hand_R"],
                "e_signed_mm": sn["e_signed_mm"],
                "alpha_open_up_deg": sn["alpha_open_up_deg"],
            }
        )

    forward_actions = list(a["actions"]) + list(b["actions"]) + slip_actions
    return {
        "ok": True,
        "support_axis": support_axis,
        "waypoints": waypoints,
        "actions_forward": forward_actions,
        "actions_A": a["actions"],
        "actions_B": b["actions"],
        "actions_slip": slip_actions,
        "capture_start": blob["capture"],
        "capture_A_end": mid_cap,
        "capture_B_end": vert_cap,
        "capture_post_slip": take_capture(env) if post_slip else vert_cap,
        "snapshot_start": start_snap,
        "snapshot_A_end": mid_snap,
        "snapshot_B_end": vert_snap,
        "snapshot_post_slip": snapshot_rich(env.sim) if post_slip else vert_snap,
        "stage_A": {k: v for k, v in a.items() if k not in ("actions", "waypoints", "capture")},
        "stage_B": {k: v for k, v in b.items() if k not in ("actions", "series", "capture", "last_stable_capture")},
        "slip": None
        if not post_slip
        else {k: v for k, v in post_slip.items() if k not in ("actions", "series", "capture")},
    }


def run_reverse_restore(
    env,
    post_slip_blob: dict,
    forward_waypoints: list[dict],
    *,
    rate_scale: float = 0.12,
    fn_hold: float = 13.0,
    contact_aware: bool = False,
    variant: str = "exact",
) -> dict[str, Any]:
    """B⁻¹ then A⁻¹ from post-slip hold state."""
    e_post = np.asarray(snapshot_rich(env.sim)["e_signed_mm"], float)
    # Ensure we're at post_slip
    restore_capture(env, post_slip_blob["capture"])
    e_post = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    th_post = float(sample_state(env.sim)["e_theta_deg"])
    post0 = snapshot_rich(env.sim, e_ref=e_post)

    # B waypoints: forward B samples + B_end, reversed → tip back toward A_end
    wp_B = []
    for w in forward_waypoints:
        if w["phase"] in ("B", "B_end", "A_end"):
            wp_B.append(np.asarray(w["hand_R"], float))
    # Keep unique-ish progression: from B_end back to A_end
    # forward order in list: ... A_end, B..., B_end
    # reverse: B_end ... A_end
    idx_a_end = next(i for i, w in enumerate(forward_waypoints) if w["phase"] == "A_end")
    idx_b_end = next(i for i, w in enumerate(forward_waypoints) if w["phase"] == "B_end")
    seg_B = forward_waypoints[idx_a_end : idx_b_end + 1]
    wp_B = [np.asarray(w["hand_R"], float) for w in reversed(seg_B)]
    # subsample for tractable tracking
    if len(wp_B) > 40:
        stride = max(1, len(wp_B) // 40)
        wp_B = wp_B[::stride]
        if not np.allclose(wp_B[-1], np.asarray(seg_B[0]["hand_R"], float)):
            wp_B.append(np.asarray(seg_B[0]["hand_R"], float))

    rB = track_orientation_waypoints(
        env,
        post_slip_blob,
        wp_B,
        phase="B_inv",
        rate_scale=rate_scale,
        fn_hold=fn_hold,
        e_post=e_post,
        th_post=th_post,
        contact_aware=contact_aware,
    )
    phase_B = {
        "ok": rB.get("ok"),
        "reason": rB.get("reason"),
        "delta_ez_mm": rB.get("delta_ez_mm"),
        "delta_ex_mm": rB.get("delta_ex_mm"),
        "snapshot": rB.get("snapshot"),
        "n_steps": len(rB.get("actions") or []),
    }
    if not rB.get("ok"):
        return {
            "ok": False,
            "variant": variant,
            "reason": f"B_inv:{rB.get('reason')}",
            "post_slip": post0,
            "phase_B_inv": phase_B,
            "phase_A_inv": None,
            "actions": rB.get("actions") or [],
            "series": rB.get("series") or [],
            "capture": rB.get("capture"),
            "loss_locus": "B_inv",
        }

    # Intermediate after B⁻¹
    mid_blob = {"capture": rB["capture"], "ref": post_slip_blob["ref"]}
    mid_snap = rB.get("snapshot")

    idx_start = next(i for i, w in enumerate(forward_waypoints) if w["phase"] == "start")
    seg_A = forward_waypoints[idx_start : idx_a_end + 1]
    wp_A = [np.asarray(w["hand_R"], float) for w in reversed(seg_A)]
    if len(wp_A) > 40:
        stride = max(1, len(wp_A) // 40)
        wp_A = wp_A[::stride]
        wp_A.append(np.asarray(seg_A[0]["hand_R"], float))

    rA = track_orientation_waypoints(
        env,
        mid_blob,
        wp_A,
        phase="A_inv",
        rate_scale=rate_scale,
        fn_hold=fn_hold,
        e_post=e_post,
        th_post=th_post,
        contact_aware=contact_aware,
    )
    phase_A = {
        "ok": rA.get("ok"),
        "reason": rA.get("reason"),
        "delta_ez_mm": rA.get("delta_ez_mm"),
        "delta_ex_mm": rA.get("delta_ex_mm"),
        "snapshot": rA.get("snapshot"),
        "n_steps": len(rA.get("actions") or []),
    }
    actions = list(rB.get("actions") or []) + list(rA.get("actions") or [])
    series = list(rB.get("series") or []) + list(rA.get("series") or [])
    ok = bool(rA.get("ok"))
    final = rA.get("snapshot") if ok else rA.get("snapshot") or mid_snap
    return {
        "ok": ok,
        "variant": variant,
        "reason": "ok" if ok else f"A_inv:{rA.get('reason')}",
        "post_slip": post0,
        "phase_B_inv": phase_B,
        "phase_mid": mid_snap,
        "phase_A_inv": phase_A,
        "final": final,
        "actions": actions,
        "series": series,
        "capture": rA.get("capture") if ok else rA.get("capture") or rB.get("capture"),
        "loss_locus": None if ok else "A_inv",
        "delta_ez_total_mm": None
        if final is None
        else float(final["e_signed_mm"][2] - e_post[2]),
        "delta_ex_total_mm": None
        if final is None
        else float(final["e_signed_mm"][0] - e_post[0]),
    }
