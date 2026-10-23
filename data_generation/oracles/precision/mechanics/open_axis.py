"""Endpoint-only axial roll: roll about u_open only when fingertips-down.

Phases:
  0 tilt → fingertips-down (α≥175°)   [no roll]
  1 axial roll (support prep)           [roll only]
  2 tilt → fingertips-up (α≤5°)       [no roll]
  3 inward axial slip (external)
  4 tilt → fingertips-down            [no roll]
  5 inverse axial roll                [roll only]
  6 residual tilt to nominal-compatible [no roll]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.recovery.action_map import W_MAX
from data_generation.oracles.precision.mechanics.finger_vertical import U_OPEN_HAND, Z_ABORT
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
from data_generation.oracles.precision.mechanics.reverse_rotation import HoldModeGrip
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R

UP = np.array([0.0, 0.0, 1.0], float)
G_WORLD = np.array([0.0, 0.0, -9.81], float)
DOWN = -UP

ALPHA_UP_GATE_DEG = 5.0
ALPHA_DOWN_GATE_DEG = 175.0
ROLL_BUDGET_DURING_TILT_DEG_S = 3.0  # |ω_roll| max allowed during tilt (diagnostic)


def u_open_world(sim) -> np.ndarray:
    Rh = np.array(hand_R(sim), float, copy=True)
    u = Rh @ U_OPEN_HAND
    return u / max(float(np.linalg.norm(u)), 1e-12)


def alpha_up_deg(sim) -> float:
    u = u_open_world(sim)
    c = float(np.clip(np.dot(u, UP), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def gates(sim) -> dict[str, Any]:
    a = alpha_up_deg(sim)
    return {
        "alpha_up_deg": a,
        "fingertips_up": bool(a <= ALPHA_UP_GATE_DEG),
        "fingertips_down": bool(a >= ALPHA_DOWN_GATE_DEG),
    }


def decompose_omega_world(omega_w: np.ndarray, u_w: np.ndarray) -> dict[str, Any]:
    o = np.asarray(omega_w, float).reshape(3)
    u = np.asarray(u_w, float).reshape(3)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    roll_s = float(np.dot(o, u))
    o_roll = roll_s * u
    o_tilt = o - o_roll
    return {
        "omega_world": o.tolist(),
        "omega_roll": o_roll.tolist(),
        "omega_tilt": o_tilt.tolist(),
        "omega_roll_signed": roll_s,
        "omega_roll_norm": abs(roll_s),
        "omega_tilt_norm": float(np.linalg.norm(o_tilt)),
    }


def project_out_roll(omega_w: np.ndarray, u_w: np.ndarray) -> np.ndarray:
    o = np.asarray(omega_w, float).reshape(3)
    u = np.asarray(u_w, float).reshape(3)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    return o - float(np.dot(o, u)) * u


def world_omega_to_body(sim, omega_w: np.ndarray) -> np.ndarray:
    Rd = np.array(sim.fsm.r_des, float, copy=True).reshape(3, 3)
    return Rd.T @ np.asarray(omega_w, float).reshape(3)


def tilt_omega_toward(
    sim,
    target_dir_w: np.ndarray,
    rate_scale: float,
    preferred_axis_w: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Pure tilt: ω ∥ u_open × target (zero roll by construction).

    When current and target are antiparallel, ``cross(u, target)`` vanishes / is
    numerical noise. Pass ``preferred_axis_w`` (from support-aware selection) or
    fall back to hand_x projected ⟂ u_open.
    """
    u = u_open_world(sim)
    t = np.asarray(target_dir_w, float)
    t = t / max(float(np.linalg.norm(t)), 1e-12)
    c = float(np.clip(np.dot(u, t), -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-4:
        return np.zeros(3), 0.0, decompose_omega_world(np.zeros(3), u)
    axis = np.cross(u, t)
    n = float(np.linalg.norm(axis))
    antiparallel = c <= -0.92 or n < 0.08
    if antiparallel:
        if preferred_axis_w is not None:
            axis = np.asarray(preferred_axis_w, float).reshape(3)
        else:
            Rh = hand_R(sim)
            axis = Rh[:, 0]
        axis = axis - float(np.dot(axis, u)) * u
        n = float(np.linalg.norm(axis))
        if n < 1e-8:
            return np.zeros(3), float(np.degrees(ang)), decompose_omega_world(np.zeros(3), u)
        axis = axis / n
        # Choose sign so instantaneous motion reduces angle to target
        tilt_tgt = t - float(np.dot(t, u)) * u
        tn = float(np.linalg.norm(tilt_tgt))
        if tn > 1e-8:
            tilt_tgt = tilt_tgt / tn
            if float(np.dot(np.cross(axis, u), tilt_tgt)) < 0.0:
                axis = -axis
    else:
        if n < 1e-8:
            return np.zeros(3), float(np.degrees(ang)), decompose_omega_world(np.zeros(3), u)
        axis = axis / n
    omega_w = axis * (float(W_MAX) * float(rate_scale))
    omega_w = project_out_roll(omega_w, u)
    decomp = decompose_omega_world(omega_w, u)
    return world_omega_to_body(sim, omega_w), float(np.degrees(ang)), decomp


def roll_omega_about_open(sim, sign: float, rate_scale: float) -> tuple[np.ndarray, dict]:
    """Pure axial roll about u_open."""
    u = u_open_world(sim)
    omega_w = u * (float(sign) * float(W_MAX) * float(rate_scale))
    decomp = decompose_omega_world(omega_w, u)
    return world_omega_to_body(sim, omega_w), decomp


def reference_pad_vector(sim) -> np.ndarray:
    """Hand +y in world — pad-facing reference for roll angle."""
    Rh = np.array(hand_R(sim), float, copy=True)
    return Rh[:, 1].copy()


def roll_angle_about_open_deg(sim, y0_w: np.ndarray, u0_w: np.ndarray) -> float:
    """Signed roll of current hand_y vs initial, in plane ⟂ u_open (use current u)."""
    u = u_open_world(sim)
    y = reference_pad_vector(sim)
    # Project initial y0 onto plane ⟂ current u for comparison in current plane
    y0p = y0_w - float(np.dot(y0_w, u)) * u
    yp = y - float(np.dot(y, u)) * u
    n0 = float(np.linalg.norm(y0p))
    n1 = float(np.linalg.norm(yp))
    if n0 < 1e-8 or n1 < 1e-8:
        return 0.0
    y0p /= n0
    yp /= n1
    c = float(np.clip(np.dot(y0p, yp), -1.0, 1.0))
    s = float(np.dot(u, np.cross(y0p, yp)))
    return float(np.degrees(np.arctan2(s, c)))


def phase_boundary_log(sim, *, phase: str, e_ref: np.ndarray | None = None, **extra) -> dict[str, Any]:
    s = sample_state(sim)
    e = np.asarray(s["e_signed_mm"], float)
    g = gates(sim)
    kin = finger_inner_normals_world(sim)
    sup = support_force_decomposition(sim)
    u = u_open_world(sim)
    out = {
        "phase": phase,
        "alpha_up_deg": g["alpha_up_deg"],
        "fingertips_up": g["fingertips_up"],
        "fingertips_down": g["fingertips_down"],
        "u_open_world": u.tolist(),
        "e_signed_mm": e.tolist(),
        "theta_deg": s["e_theta_deg"],
        "fn_L": s["fn_L"],
        "fn_R": s["fn_R"],
        "fn_tot": s["fn_tot"],
        "bilateral": s["bilateral"],
        "v_rel_h": s["v_rel_h"],
        "v_rel_norm": s["v_rel_norm"],
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "world_height_margin": float(sim.data.xpos[sim.ids.object_body][2]) - Z_ABORT,
        "left_align_up": kin["left_align_up"],
        "right_align_up": kin["right_align_up"],
        "Fn_z_L": sup["sides"]["L"]["Fn_z_normal"],
        "Fn_z_R": sup["sides"]["R"]["Fn_z_normal"],
        "qpos": np.asarray(sim.data.qpos, float)[:9].tolist(),
        **extra,
    }
    if e_ref is not None:
        er = np.asarray(e_ref, float)
        out["delta_ex_mm"] = float(e[0] - er[0])
        out["delta_ez_mm"] = float(e[2] - er[2])
    return out


def _assert_gate(sim, *, require_down: bool = False, require_up: bool = False, phase: str) -> dict | None:
    g = gates(sim)
    if require_down and not g["fingertips_down"]:
        return {"ok": False, "reason": f"assert_fail_not_down@{phase}", "alpha_up_deg": g["alpha_up_deg"]}
    if require_up and not g["fingertips_up"]:
        return {"ok": False, "reason": f"assert_fail_not_up@{phase}", "alpha_up_deg": g["alpha_up_deg"]}
    return None


def run_tilt_phase(
    env,
    blob: dict,
    *,
    target_dir_w: np.ndarray,
    phase: str,
    rate_scale: float = 0.12,
    fn_hold: float = 13.0,
    max_time_s: float = 10.0,
    ang_tol_deg: float = 5.0,
    contact_aware: bool = True,
    e_ref: np.ndarray | None = None,
    y0_w: np.ndarray | None = None,
    allow_roll: bool = False,
    roll_accum0_deg: float = 0.0,
    z_ref_boost: float = 0.0,
    height_kp: float = 2.5,
    support_guard: bool = False,
    support_side: str | None = None,
    min_support_fn: float = 0.8,
    max_vrel_support: float = 0.12,
) -> dict[str, Any]:
    """Closed-loop tilt toward target u_open direction; roll projected out unless allow_roll.

    When ``support_guard`` is True (DX2 S1-protected rotation), abort on support-side
    force collapse / uncontrolled slip / latch — not merely on transient non-bilateral
    contact (one-sided mechanical support is expected under gravity).
    """
    from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode

    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    if e_ref is None:
        e_ref = np.asarray(s0["e_signed_mm"], float)
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2]) + float(z_ref_boost)
    hold = HoldModeGrip(fn_hold=fn_hold)
    hold.reset(s0["fn_tot"])
    actions, series = [], []
    lost = 0
    rate = float(rate_scale)
    max_roll_cmd = 0.0
    fail = None
    side = support_side
    saw_support_load = False

    for i in range(int(max_time_s / DT)):
        w_body, ang_err, decomp = tilt_omega_toward(env.sim, target_dir_w, rate)
        if not allow_roll:
            # Re-project after body mapping numerical error
            u = u_open_world(env.sim)
            ow = np.array(env.sim.fsm.r_des, float).reshape(3, 3) @ w_body
            ow = project_out_roll(ow, u)
            w_body = world_omega_to_body(env.sim, ow)
            decomp = decompose_omega_world(ow, u)
        max_roll_cmd = max(max_roll_cmd, float(decomp["omega_roll_norm"]))
        if contact_aware:
            vrel = float(sample_state(env.sim)["v_rel_norm"])
            if vrel > 0.008:
                rate = max(0.06, rate * 0.9)
            elif vrel < 0.003:
                rate = min(float(rate_scale), rate * 1.01)
            # Do not zero ω — that stalls tilt under gravity-assisted slip.
        st = sample_state(env.sim)
        a_g, fn_tgt, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref, kp=height_kp), w_body, a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        st2 = sample_state(env.sim)
        g = gates(env.sim)
        mode = classify_contact_mode(env.sim) if support_guard else None
        if support_guard and mode is not None:
            side = side or mode.get("support_side")
            if side == "L":
                fn_sup = float(mode.get("fn_L") or 0.0)
                fnz_sup = float(mode.get("Fn_z_L") or 0.0)
            elif side == "R":
                fn_sup = float(mode.get("fn_R") or 0.0)
                fnz_sup = float(mode.get("Fn_z_R") or 0.0)
            else:
                fn_sup = max(float(mode.get("fn_L") or 0.0), float(mode.get("fn_R") or 0.0))
                fnz_sup = max(float(mode.get("Fn_z_L") or 0.0), float(mode.get("Fn_z_R") or 0.0))
            if fn_sup >= min_support_fn or fnz_sup > 1.5:
                saw_support_load = True
        else:
            fn_sup = float(st2.get("fn_tot") or 0.0)
            fnz_sup = 0.0
        roll_acc = roll_accum0_deg
        if y0_w is not None:
            roll_acc = roll_angle_about_open_deg(env.sim, y0_w, u_open_world(env.sim))
        if (i + 1) % 4 == 0 or ang_err < ang_tol_deg:
            row = {
                "t": (i + 1) * DT,
                "phase": phase,
                "alpha_up_deg": g["alpha_up_deg"],
                "ang_err_deg": ang_err,
                "omega_roll_norm": decomp["omega_roll_norm"],
                "omega_tilt_norm": decomp["omega_tilt_norm"],
                "omega_roll_signed": decomp["omega_roll_signed"],
                "roll_angle_deg": roll_acc,
                "e_signed_mm": st2["e_signed_mm"],
                "delta_ez_mm": float(np.asarray(st2["e_signed_mm"], float)[2] - e_ref[2]),
                "delta_ex_mm": float(np.asarray(st2["e_signed_mm"], float)[0] - e_ref[0]),
                "fn_tot": st2["fn_tot"],
                "fn_target": fn_tgt,
                "v_rel_norm": st2["v_rel_norm"],
                "bilateral": st2["bilateral"],
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "rate_scale": rate,
            }
            if support_guard and mode is not None:
                row.update({
                    "support_side": side,
                    "fn_support": fn_sup,
                    "Fn_z_support": fnz_sup,
                    "one_sided_mechanical": bool(mode.get("one_sided_mechanical")),
                    "contact_mode": mode.get("mode"),
                })
            series.append(row)
        if failed or latched(env.sim):
            fail = "contact_loss"
            break
        if support_guard:
            # S1-protected: allow one-sided; abort only on support collapse / free slip.
            tot = float(st2.get("fn_L") or 0.0) + float(st2.get("fn_R") or 0.0)
            if saw_support_load and fn_sup < min_support_fn and tot < 1.5:
                fail = "support_lost"
                break
            if not saw_support_load and tot < 1.0 and i > int(0.4 / DT):
                fail = "support_lost"
                break
            if float(st2.get("v_rel_norm") or 0.0) > float(max_vrel_support):
                fail = "uncontrolled_slip"
                break
        elif not st2["bilateral"]:
            fail = "contact_loss"
            break
        if float(env.sim.data.xpos[env.sim.ids.object_body][2]) < Z_ABORT:
            fail = "world_height"
            break
        if not allow_roll and decomp["omega_roll_norm"] > 0.15:
            # Should be near-zero after projection; hard fail if large
            fail = "roll_leak_during_tilt"
            break
        if ang_err <= ang_tol_deg:
            break
    else:
        fail = fail or "timeout"

    # settle
    st2 = sample_state(env.sim)
    for _ in range(8):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref, kp=height_kp), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        st2 = sample_state(env.sim)

    boundary = phase_boundary_log(
        env.sim,
        phase=phase + "_end",
        e_ref=e_ref,
        max_omega_roll_cmd=max_roll_cmd,
        roll_angle_deg=roll_angle_about_open_deg(env.sim, y0_w, u_open_world(env.sim)) if y0_w is not None else None,
    )
    if support_guard:
        mode_exit = classify_contact_mode(env.sim)
        contact_ok = bool(
            mode_exit.get("bilateral")
            or mode_exit.get("one_sided_mechanical")
            or (
                (side == "L" and float(mode_exit.get("fn_L") or 0.0) >= min_support_fn)
                or (side == "R" and float(mode_exit.get("fn_R") or 0.0) >= min_support_fn)
            )
        )
        ok = fail is None and contact_ok
    else:
        mode_exit = None
        ok = fail is None and st2["bilateral"]
    return {
        "ok": ok,
        "reason": "ok" if ok else fail,
        "phase": phase,
        "actions": actions,
        "series": series,
        "capture": take_capture(env),
        "boundary": boundary,
        "max_omega_roll_cmd": max_roll_cmd,
        "alpha_final": gates(env.sim)["alpha_up_deg"],
        "support_side": side,
        "support_guard": bool(support_guard),
        "contact_mode_exit": mode_exit,
    }


def run_roll_phase(
    env,
    blob: dict,
    *,
    sign: float,
    angle_deg: float,
    phase: str,
    rate_scale: float = 0.15,
    fn_hold: float = 13.0,
    require_down: bool = True,
    y0_w: np.ndarray | None = None,
    e_ref: np.ndarray | None = None,
) -> dict[str, Any]:
    """Axial roll about u_open; only legal when fingertips-down."""
    restore_capture(env, blob["capture"])
    if require_down:
        bad = _assert_gate(env.sim, require_down=True, phase=phase)
        if bad:
            return {**bad, "phase": phase, "actions": [], "series": [], "capture": take_capture(env)}

    s0 = sample_state(env.sim)
    if e_ref is None:
        e_ref = np.asarray(s0["e_signed_mm"], float)
    if y0_w is None:
        y0_w = reference_pad_vector(env.sim)
    roll0 = roll_angle_about_open_deg(env.sim, y0_w, u_open_world(env.sim))
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    hold = HoldModeGrip(fn_hold=fn_hold)
    hold.reset(s0["fn_tot"])
    actions, series = [], []
    lost = 0
    target = float(angle_deg)  # signed delta from roll0
    # Integrate until |roll - roll0 - sign*angle| small
    goal = roll0 + float(sign) * abs(float(angle_deg))
    n_max = max(1, int(round(np.radians(abs(angle_deg)) / max(float(W_MAX) * rate_scale, 1e-6) / DT))) + 40

    for i in range(n_max):
        if require_down and not gates(env.sim)["fingertips_down"]:
            return {
                "ok": False,
                "reason": "lost_down_gate_during_roll",
                "phase": phase,
                "actions": actions,
                "series": series,
                "capture": take_capture(env),
                "boundary": phase_boundary_log(env.sim, phase=phase + "_abort", e_ref=e_ref),
            }
        w_body, decomp = roll_omega_about_open(env.sim, sign, rate_scale)
        st = sample_state(env.sim)
        a_g, fn_tgt, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), w_body, a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        st2 = sample_state(env.sim)
        roll = roll_angle_about_open_deg(env.sim, y0_w, u_open_world(env.sim))
        if (i + 1) % 3 == 0:
            series.append(
                {
                    "t": (i + 1) * DT,
                    "phase": phase,
                    "alpha_up_deg": gates(env.sim)["alpha_up_deg"],
                    "roll_angle_deg": roll,
                    "omega_roll_signed": decomp["omega_roll_signed"],
                    "omega_tilt_norm": decomp["omega_tilt_norm"],
                    "e_signed_mm": st2["e_signed_mm"],
                    "delta_ez_mm": float(np.asarray(st2["e_signed_mm"], float)[2] - e_ref[2]),
                    "fn_tot": st2["fn_tot"],
                    "v_rel_norm": st2["v_rel_norm"],
                    "bilateral": st2["bilateral"],
                }
            )
        if failed or latched(env.sim) or not st2["bilateral"]:
            return {
                "ok": False,
                "reason": "contact_loss",
                "phase": phase,
                "actions": actions,
                "series": series,
                "capture": take_capture(env),
            }
        # Progress toward goal (handle wrap roughly within ±180)
        err = goal - roll
        while err > 180:
            err -= 360
        while err < -180:
            err += 360
        if abs(err) < 4.0:
            break

    for _ in range(8):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())

    roll_f = roll_angle_about_open_deg(env.sim, y0_w, u_open_world(env.sim))
    bil = bool(sample_state(env.sim)["bilateral"])
    gate_ok = bool(gates(env.sim)["fingertips_down"]) if require_down else True
    return {
        "ok": bool(bil and gate_ok),
        "reason": "ok" if (bil and gate_ok) else ("lost_bilateral" if not bil else "lost_down_gate"),
        "phase": phase,
        "actions": actions,
        "series": series,
        "capture": take_capture(env),
        "boundary": phase_boundary_log(
            env.sim, phase=phase + "_end", e_ref=e_ref, roll_angle_deg=roll_f, roll0_deg=roll0, roll_goal_deg=goal
        ),
        "roll0_deg": roll0,
        "roll_final_deg": roll_f,
        "roll_delta_deg": roll_f - roll0,
        "sign": sign,
        "angle_cmd_deg": angle_deg,
    }


def choose_roll_sign_for_support(sim) -> float:
    """State-dependent roll sign; keep legacy −1.0 as default when ambiguous.

    Prefers the sign that increases the max pad ``n·UP`` alignment (support potential).
    """
    kin = finger_inner_normals_world(sim)
    # Legacy successful Plan-B path used −1.0; keep as default candidate winner on ties.
    legacy = -1.0
    align_R = float(kin.get("right_align_up") or 0.0)
    align_L = float(kin.get("left_align_up") or 0.0)
    # If one pad already more upward-facing, prefer rolling to strengthen that side.
    if abs(align_R - align_L) < 1e-3:
        return legacy
    # Positive roll about u_open (right-hand rule along fingertips). Empirical mapping:
    # legacy −1 remains default; only flip when left pad is clearly the support candidate.
    if align_L > align_R + 0.05:
        return +1.0
    return legacy
