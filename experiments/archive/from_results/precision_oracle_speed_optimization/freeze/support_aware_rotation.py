"""Support-aware antiparallel-safe tilt planner for Precision dz flips.

Root cause addressed: ``tilt_omega_toward`` uses ``axis = cross(u, target)``, which
vanishes (or becomes numerical noise) when fingertips are antiparallel to UP/DOWN.
This module enumerates physically meaningful axes, scores short MuJoCo rollouts from
identical captures, and executes a fixed world-axis tilt with the winner.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from controllers.recovery.action_map import W_MAX
from data_generation.oracles.precision.contact_modes import (
    MODE_ONE_SIDED_SUPPORT,
    classify_contact_mode,
    make_s1_state,
)
from data_generation.oracles.precision.endpoint_only_axial_roll import (
    DOWN,
    UP,
    gates,
    project_out_roll,
    u_open_world,
    world_omega_to_body,
)
from data_generation.oracles.precision.finger_surface_support import (
    finger_inner_normals_world,
    support_force_decomposition,
)
from data_generation.oracles.precision.force_slip_control import DT, make_action, sample_state
from data_generation.oracles.precision.orientation_planner import _height_hold_v_hand
from data_generation.oracles.precision.reverse_supported_rotation import HoldModeGrip
from data_generation.oracles.precision.precision_types import FN_HOLD, Z_ABORT
from data_generation.oracles.precision.rotation_speed_profile import (
    contact_rate_guard,
    scheduled_rate,
)
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R

G_WORLD = np.array([0.0, 0.0, -9.81], float)
ANTIPARALLEL_DOT = -0.92  # |u·target| above this → treat as antiparallel risk
NOISY_CROSS_NORM = 0.08
MIN_SUPPORT_FNZ = 1.0  # short-rollout soft gate (cost); hard fail stays contact_loss


def _unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.zeros(3)


def project_axis_perp_u(axis_w: np.ndarray, u_w: np.ndarray) -> np.ndarray | None:
    a = np.asarray(axis_w, float).reshape(3)
    u = _unit(u_w)
    a = a - float(np.dot(a, u)) * u
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        return None
    return a / n


def generate_tilt_axis_candidates(sim, target_dir_w: np.ndarray) -> list[dict[str, Any]]:
    """Physically meaningful axes ⟂ u_open; both signs evaluated separately later."""
    u = u_open_world(sim)
    t = _unit(target_dir_w)
    Rh = hand_R(sim)
    kin = finger_inner_normals_world(sim)
    mode = classify_contact_mode(sim)
    cands: list[dict[str, Any]] = []

    def add(name: str, axis: np.ndarray, priority: float):
        a = project_axis_perp_u(axis, u)
        if a is None:
            return
        # Dedup by direction (ignore sign here)
        for c in cands:
            if abs(float(np.dot(c["axis"], a))) > 0.98:
                if priority > c["priority"]:
                    c.update({"name": name, "axis": a, "priority": priority})
                return
        cands.append({"name": name, "axis": a, "priority": priority})

    cross = np.cross(u, t)
    cross_n = float(np.linalg.norm(cross))
    antiparallel = float(np.dot(u, t)) <= ANTIPARALLEL_DOT
    # Demote pure geometric cross near DOWN↔UP so support-compatible axes win
    # short-horizon evaluations (avoids false-OK unsafe flips).
    if cross_n >= NOISY_CROSS_NORM and not antiparallel:
        add("cross_u_target", cross, priority=2.0 if float(np.dot(u, t)) > -0.3 else 1.2)
    elif cross_n >= 1e-4:
        add("cross_u_target_noisy", cross, priority=0.2)

    add("hand_x", Rh[:, 0], priority=2.4)
    add("hand_y", Rh[:, 1], priority=1.6)
    add("gravity_cross_u", np.cross(G_WORLD, u), priority=2.6)

    nR = np.asarray(kin["right_inner_n_world"], float)
    nL = np.asarray(kin["left_inner_n_world"], float)
    add("nR_cross_u", np.cross(nR, u), priority=2.2)
    add("nL_cross_u", np.cross(nL, u), priority=2.2)

    if mode.get("support_side") == "R":
        add("supportR_cross_u", np.cross(nR, u), priority=3.0)
    elif mode.get("support_side") == "L":
        add("supportL_cross_u", np.cross(nL, u), priority=3.0)

    cands.sort(key=lambda c: -c["priority"])
    return cands


def _support_fnz(sim, side: str | None) -> float:
    m = classify_contact_mode(sim)
    if side == "L":
        return float(m["Fn_z_L"])
    if side == "R":
        return float(m["Fn_z_R"])
    return max(float(m["Fn_z_L"]), float(m["Fn_z_R"]))


def _gravity_press_score(sim, support_side: str | None) -> float:
    """Higher when gravity loads the supporting pad normal (presses cylinder into it)."""
    if not support_side:
        return 0.0
    kin = finger_inner_normals_world(sim)
    n = np.asarray(
        kin["right_inner_n_world"] if support_side == "R" else kin["left_inner_n_world"], float
    )
    # Force-on-cylinder from gravity is along G; pad pushes back along +n (inner).
    # Prefer n having upward component while Fn_z on that side is positive.
    return float(np.dot(n, UP)) * max(_support_fnz(sim, support_side), 0.0)


def short_axis_rollout(
    env,
    capture,
    *,
    axis_w: np.ndarray,
    sign: float,
    target_dir_w: np.ndarray,
    horizon_s: float = 1.6,
    rate_scale: float = 0.05,
    z_ref_boost: float = 0.08,
    support_side_hint: str | None = None,
) -> dict[str, Any]:
    """Bounded counterfactual rollout; caller must restore capture after."""
    restore_capture(env, capture)
    axis0 = _unit(axis_w)
    t = _unit(target_dir_w)
    u0 = u_open_world(env.sim)
    ang0 = float(np.degrees(np.arccos(np.clip(np.dot(u0, t), -1.0, 1.0))))
    mode0 = classify_contact_mode(env.sim)
    side = support_side_hint or mode0.get("support_side")
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    fnz0 = max(_support_fnz(env.sim, side), 0.0)
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    e0 = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    fail = None
    min_fnz = 1e9
    sum_press = 0.0
    max_vrel = 0.0
    n_steps = 0

    for i in range(max(1, int(horizon_s / DT))):
        u = u_open_world(env.sim)
        ang = float(np.degrees(np.arccos(np.clip(np.dot(u, t), -1.0, 1.0))))
        ow = project_out_roll(axis0 * (float(sign) * float(W_MAX) * float(rate_scale)), u)
        wb = world_omega_to_body(env.sim, ow)
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        step_legal(env.sim, make_action(_height_hold_v_hand(env.sim, z0 + z_ref_boost, kp=6.0), wb, a_g))
        n_steps += 1
        st2 = sample_state(env.sim)
        max_vrel = max(max_vrel, float(st2.get("v_rel_norm") or 0.0))
        fnz = _support_fnz(env.sim, side)
        min_fnz = min(min_fnz, fnz)
        sum_press += _gravity_press_score(env.sim, side)
        if latched(env.sim) or not st2.get("bilateral"):
            fail = "contact_loss"
            break
        if float(env.sim.data.xpos[env.sim.ids.object_body][2]) < Z_ABORT:
            fail = "world_height"
            break
        if ang <= 8.0:
            break

    u1 = u_open_world(env.sim)
    ang1 = float(np.degrees(np.arccos(np.clip(np.dot(u1, t), -1.0, 1.0))))
    e1 = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    progress = ang0 - ang1
    mode1 = classify_contact_mode(env.sim)
    min_fnz_f = float(min_fnz if min_fnz < 1e8 else 0.0)
    # Cost: lower is better — support persistence outweighs raw angular progress.
    if fail:
        cost = 1e6 + (1e5 if fail == "contact_loss" else 0.0)
        cost -= 2.0 * max(progress, 0.0)
    else:
        support_penalty = 0.0
        if fnz0 > 3.0 and min_fnz_f < max(MIN_SUPPORT_FNZ, 0.35 * fnz0):
            support_penalty = 40.0 + 8.0 * (max(MIN_SUPPORT_FNZ, 0.35 * fnz0) - min_fnz_f)
        cost = (
            -10.0 * progress
            + 90.0 * max_vrel
            + 0.5 * float(np.linalg.norm(e1 - e0))
            - 4.0 * max(min_fnz_f, -5.0)
            - 1.2 * (sum_press / max(n_steps, 1))
            - (8.0 if mode1.get("one_sided_mechanical") else 0.0)
            + support_penalty
        )
    support_ok = min_fnz_f >= MIN_SUPPORT_FNZ or min_fnz_f >= 0.45 * max(fnz0, 0.1)
    return {
        "ok": (
            fail is None
            and progress > 2.0
            and bool(sample_state(env.sim).get("bilateral"))
            and support_ok
        ),
        "fail": fail,
        "cost": float(cost),
        "progress_deg": float(progress),
        "ang0": ang0,
        "ang1": ang1,
        "max_vrel": max_vrel,
        "min_fnz_support": min_fnz_f,
        "mean_press": float(sum_press / max(n_steps, 1)),
        "n_steps": n_steps,
        "support_side": side,
        "mode_exit": mode1.get("mode"),
        "alpha_exit": float(gates(env.sim)["alpha_up_deg"]),
    }


def select_support_aware_tilt_axis(
    env,
    capture,
    target_dir_w: np.ndarray,
    *,
    support_side_hint: str | None = None,
    horizon_s: float = 1.6,
    rate_scale: float = 0.05,
) -> dict[str, Any]:
    """Evaluate axis×sign candidates from identical capture; restore when done."""
    restore_capture(env, capture)
    mode0 = classify_contact_mode(env.sim)
    side = support_side_hint or mode0.get("support_side")
    raw = generate_tilt_axis_candidates(env.sim, target_dir_w)
    trials = []
    best = None
    for c in raw:
        for sign in (+1.0, -1.0):
            restore_capture(env, capture)
            r = short_axis_rollout(
                env,
                capture,
                axis_w=c["axis"],
                sign=sign,
                target_dir_w=target_dir_w,
                horizon_s=horizon_s,
                rate_scale=rate_scale,
                support_side_hint=side,
            )
            row = {
                "name": c["name"],
                "priority": c["priority"],
                "axis": np.asarray(c["axis"], float).tolist(),
                "sign": sign,
                **{k: v for k, v in r.items()},
            }
            trials.append(row)
            if r["ok"] and (best is None or r["cost"] < best["cost"]):
                best = row
    restore_capture(env, capture)
    # Diagnostic fallback only — never treat as executable without ok.
    if best is None and trials:
        viable = [
            t
            for t in trials
            if t.get("progress_deg", 0) > 2.0 and t.get("fail") is None
        ]
        best = min(viable, key=lambda t: t["cost"]) if viable else min(trials, key=lambda t: t["cost"])
    return {
        "ok": bool(best and best.get("ok")),
        "selected": best,
        "trials": trials,
        "support_side_hint": side,
        "mode_entry": mode0,
        "antiparallel": bool(
            float(np.dot(u_open_world(env.sim), _unit(target_dir_w))) <= ANTIPARALLEL_DOT
        ),
    }


def run_fixed_axis_tilt(
    env,
    blob: dict,
    *,
    target_dir_w: np.ndarray,
    axis_w: np.ndarray,
    sign: float,
    phase: str,
    rate_scale: float = 0.14,
    fn_hold: float = FN_HOLD,
    max_time_s: float = 18.0,
    ang_tol_deg: float = 6.0,
    z_ref_boost: float = 0.08,
    height_kp: float = 6.5,
    support_side_hint: str | None = None,
    pre_lift_s: float = 0.0,
    cruise: float | None = None,
    peak: float | None = None,
    floor: float | None = None,
    accel_s: float | None = None,
    use_speed_profile: bool = True,
) -> dict[str, Any]:
    """Closed-loop tilt using a fixed world rotation axis (single-axis primitive)."""
    restore_capture(env, blob["capture"])
    axis0 = _unit(axis_w)
    t = _unit(target_dir_w)
    u0 = u_open_world(env.sim)
    ang0 = float(np.degrees(np.arccos(np.clip(np.dot(u0, t), -1.0, 1.0))))
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    hold = HoldModeGrip(fn_hold=fn_hold)
    hold.reset(sample_state(env.sim)["fn_tot"])
    actions: list = []
    series: list = []
    fail = None
    path_len = 0.0
    prev_u = u0.copy()
    signed_angle = 0.0
    side = support_side_hint or classify_contact_mode(env.sim).get("support_side")
    cruise_r = float(cruise if cruise is not None else rate_scale)
    peak_r = float(peak if peak is not None else max(cruise_r * 1.25, cruise_r))
    floor_r = float(floor if floor is not None else min(0.06, cruise_r))
    accel = float(accel_s if accel_s is not None else 0.35)
    peak_cmd = 0.0
    sum_rate = 0.0

    # Optional pre-lift so return-DOWN starts with margin above Z_ABORT.
    z_cmd = z0 + float(z_ref_boost)
    for _ in range(int(max(0.0, pre_lift_s) / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_cmd, kp=height_kp), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        if latched(env.sim) or not sample_state(env.sim).get("bilateral"):
            fail = "contact_loss"
            break
    if fail:
        return {
            "ok": False,
            "reason": fail,
            "phase": phase,
            "actions": actions,
            "series": series,
            "capture": take_capture(env),
            "axis_w": axis0.tolist(),
            "sign": float(sign),
            "ang0_deg": ang0,
            "ang1_deg": ang0,
            "signed_angle_rad": 0.0,
            "path_length_u": 0.0,
            "n_actions": len(actions),
            "sim_s": len(actions) * DT,
            "support_side": side,
            "mode_exit": classify_contact_mode(env.sim),
            "omega_body_cmd_scale": float(sign) * float(rate_scale),
            "rate_profile": {"cruise": cruise_r, "peak": peak_r, "floor": floor_r},
        }
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    z_cmd = max(z0 + float(z_ref_boost), Z_ABORT + 0.06)
    t_rot0 = 0.0

    for i in range(int(max_time_s / DT)):
        u = u_open_world(env.sim)
        ang = float(np.degrees(np.arccos(np.clip(np.dot(u, t), -1.0, 1.0))))
        path_len += float(np.linalg.norm(u - prev_u))
        signed_angle += float(np.dot(np.cross(prev_u, u), axis0))
        prev_u = u.copy()
        alpha = float(gates(env.sim)["alpha_up_deg"])
        t_s = t_rot0 + i * DT
        if use_speed_profile:
            rate = scheduled_rate(
                t_s=t_s, ang_err_deg=ang, ang0_deg=ang0,
                cruise=cruise_r, peak=peak_r, floor=floor_r, accel_s=accel,
            )
            rate, guard = contact_rate_guard(
                env.sim, rate, support_side=side, floor=floor_r,
            )
        else:
            # Legacy slow schedule (baseline comparison).
            rate = float(rate_scale)
            if ang < 40.0:
                rate = min(rate, 0.04)
            elif ang < 80.0:
                rate = min(rate, 0.055)
            guard = {"guard": "legacy"}
        peak_cmd = max(peak_cmd, rate)
        sum_rate += rate
        zb = float(z_ref_boost)
        if ang < 100.0:
            zb = max(zb, 0.08)
        if ang < 40.0:
            zb = max(zb, 0.10)
        z_tgt = max(z0 + zb, Z_ABORT + 0.05)
        ow = project_out_roll(axis0 * (float(sign) * float(W_MAX) * rate), u)
        wb = world_omega_to_body(env.sim, ow)
        st = sample_state(env.sim)
        a_g, fn_tgt, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_tgt, kp=height_kp), wb, a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        st2 = sample_state(env.sim)
        if (i + 1) % 5 == 0 or ang <= ang_tol_deg:
            m = classify_contact_mode(env.sim)
            series.append(
                {
                    "t": (i + 1) * DT,
                    "phase": phase,
                    "ang_err_deg": ang,
                    "alpha_up_deg": alpha,
                    "v_rel_norm": st2["v_rel_norm"],
                    "bilateral": st2["bilateral"],
                    "Fn_z_L": m["Fn_z_L"],
                    "Fn_z_R": m["Fn_z_R"],
                    "support_side": m["support_side"],
                    "mode": m["mode"],
                    "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                    "rate": rate,
                    "guard": guard.get("guard"),
                }
            )
        if latched(env.sim) or not st2.get("bilateral"):
            fail = "contact_loss"
            break
        if float(env.sim.data.xpos[env.sim.ids.object_body][2]) < Z_ABORT:
            fail = "world_height"
            break
        if ang <= ang_tol_deg:
            break
    else:
        fail = fail or "timeout"

    for _ in range(6):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_cmd, kp=height_kp), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())

    u1 = u_open_world(env.sim)
    ang1 = float(np.degrees(np.arccos(np.clip(np.dot(u1, t), -1.0, 1.0))))
    mode = classify_contact_mode(env.sim)
    n_rot = max(1, len(actions) - 6)
    ok = fail is None and ang1 <= ang_tol_deg + 2.0 and bool(sample_state(env.sim).get("bilateral"))
    return {
        "ok": ok,
        "reason": "ok" if ok else fail,
        "phase": phase,
        "actions": actions,
        "series": series,
        "capture": take_capture(env),
        "axis_w": axis0.tolist(),
        "sign": float(sign),
        "ang0_deg": ang0,
        "ang1_deg": ang1,
        "signed_angle_rad": float(signed_angle),
        "path_length_u": float(path_len),
        "n_actions": len(actions),
        "sim_s": len(actions) * DT,
        "support_side": mode.get("support_side") or support_side_hint,
        "mode_exit": mode,
        "omega_body_cmd_scale": float(sign) * float(cruise_r),
        "rate_profile": {
            "cruise": cruise_r,
            "peak": peak_r,
            "floor": floor_r,
            "accel_s": accel,
            "use_speed_profile": use_speed_profile,
            "peak_cmd": peak_cmd,
            "mean_cmd": sum_rate / n_rot,
        },
    }


def plan_and_execute_support_tilt(
    env,
    blob: dict,
    target_dir_w: np.ndarray,
    *,
    phase: str,
    cfg=None,
    support_side_hint: str | None = None,
    rate_scale: float | None = None,
    require_mechanical_s1: bool = False,
) -> dict[str, Any]:
    """Select support-aware axis+sign, then execute fixed-axis tilt."""
    restore_capture(env, blob["capture"])
    live = make_s1_state(env.sim, roll_sign=None, force_valid=False)
    if require_mechanical_s1 and not live.get("force_ok"):
        return {
            "ok": False,
            "reason": "no_verified_mechanical_s1",
            "phase": phase,
            "actions": [],
            "capture": blob["capture"],
            "s1_live": live,
        }
    side = support_side_hint or live.get("support_side")
    cruise = float(
        rate_scale
        if rate_scale is not None
        else getattr(cfg, "rate_tilt_cruise", None) or getattr(cfg, "rate_protected_s1", 0.14) or 0.14
    )
    cruise = float(np.clip(cruise, 0.05, 0.28))
    peak = float(getattr(cfg, "rate_tilt_peak", cruise * 1.3) or cruise * 1.3)
    floor = float(getattr(cfg, "rate_tilt_floor", 0.06) or 0.06)
    accel = float(getattr(cfg, "tilt_accel_s", 0.35) or 0.35)
    returning_down = float(np.dot(_unit(target_dir_w), DOWN)) > 0.9
    going_up = float(np.dot(_unit(target_dir_w), UP)) > 0.9
    up_tol = float(getattr(cfg, "dz_up_tol_deg", 12.0) or 12.0)
    ang_tol = up_tol if going_up else 8.0
    pre_lift = float(getattr(cfg, "dz_return_pre_lift_s", 0.25) or 0.25) if returning_down else 0.0

    sel = select_support_aware_tilt_axis(
        env,
        blob["capture"],
        target_dir_w,
        support_side_hint=side,
        horizon_s=1.2,
        rate_scale=min(cruise, 0.08),
    )
    chosen = sel.get("selected")
    sel_summary = {
        "ok": sel.get("ok"),
        "antiparallel": sel.get("antiparallel"),
        "chosen_name": (chosen or {}).get("name"),
        "chosen_sign": (chosen or {}).get("sign"),
        "chosen_cost": (chosen or {}).get("cost"),
        "chosen_progress": (chosen or {}).get("progress_deg"),
        "n_trials": len(sel.get("trials") or []),
        "top_trials": sorted(sel.get("trials") or [], key=lambda t: t["cost"])[:8],
    }
    if not chosen or not sel.get("ok"):
        return {
            "ok": False,
            "reason": "no_safe_rotation_candidate",
            "phase": phase,
            "actions": [],
            "capture": blob["capture"],
            "selection": sel_summary,
            "planner": "support_aware_fixed_axis",
        }

    # Slightly conservative cruise on return-DOWN (longer lever / drop risk).
    cruise_exec = cruise * (0.85 if returning_down else 1.0)
    peak_exec = peak * (0.85 if returning_down else 1.0)
    use_prof = bool(getattr(cfg, "use_tilt_speed_profile", True))
    exe = run_fixed_axis_tilt(
        env,
        blob,
        target_dir_w=target_dir_w,
        axis_w=np.asarray(chosen["axis"], float),
        sign=float(chosen["sign"]),
        phase=phase,
        rate_scale=cruise_exec,
        support_side_hint=side,
        z_ref_boost=0.12 if returning_down else 0.08,
        height_kp=9.0 if returning_down else 7.0,
        max_time_s=18.0,
        ang_tol_deg=ang_tol,
        pre_lift_s=pre_lift,
        cruise=cruise_exec,
        peak=peak_exec,
        floor=floor,
        accel_s=accel,
        use_speed_profile=use_prof,
    )
    sel_summary["chosen_axis"] = chosen.get("axis")
    exe["selection"] = sel_summary
    exe["planner"] = "support_aware_fixed_axis"
    return exe


def verify_mechanical_s1(sim, s1_state: dict | None = None) -> dict[str, Any]:
    """Strict live S1 check — stale flags / bilateral-only force_valid override rejected."""
    m = classify_contact_mode(sim)
    live = make_s1_state(sim, roll_sign=(s1_state or {}).get("roll_sign"), force_valid=False)
    ok = bool(m.get("one_sided_mechanical") and m.get("bilateral") and live.get("force_ok"))
    return {
        "ok": ok,
        "live": live,
        "mode": m,
        "reason": None if ok else (
            "not_bilateral" if not m.get("bilateral") else "not_one_sided_mechanical_support"
        ),
    }
