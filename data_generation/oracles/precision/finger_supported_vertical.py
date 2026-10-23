"""Multi-stage finger-support → fingertips-up → axial slip (offline Oracle).

Stage A: ±wz to intermediate finger-pad-up support (not a slip pose).
Stage B: continue reorientation until u_open_world ≈ world_up (α ≤ 5°).
Stage C: only then closed-loop inward axial slip.

u_open := +hand_z (palm → fingertips). d_inward := −u_open.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.recovery.action_map import W_MAX
from data_generation.oracles.precision.finger_surface_support import (
    finger_inner_normals_world,
    support_force_decomposition,
    waypoint_rotate_axis,
)
from data_generation.oracles.precision.force_slip_control import (
    DT,
    ForcePController,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import _height_hold_v_hand
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R
from evaluation.success_criteria import P_TGT

UP = np.array([0.0, 0.0, 1.0], float)
G_WORLD = np.array([0.0, 0.0, -9.81], float)
U_OPEN_HAND = np.array([0.0, 0.0, 1.0], float)  # +hand_z
ALPHA_GATE_DEG = 5.0
Z_ABORT = 0.44
REORI_DX_MM = 5.0
REORI_DTHETA_DEG = 8.0


def opening_alignment(sim) -> dict[str, Any]:
    """World-frame opening alignment vs world-up; gravity vs inward desire."""
    Rh = np.array(hand_R(sim), float, copy=True)
    u_w = Rh @ U_OPEN_HAND
    u_w = u_w / max(float(np.linalg.norm(u_w)), 1e-12)
    d_in_w = -u_w
    c = float(np.clip(np.dot(u_w, UP), -1.0, 1.0))
    alpha_deg = float(np.degrees(np.arccos(c)))
    g_dot = float(np.dot(G_WORLD, d_in_w))
    return {
        "u_open_hand": U_OPEN_HAND.tolist(),
        "u_open_world": u_w.tolist(),
        "d_inward_world": d_in_w.tolist(),
        "alpha_open_up_deg": alpha_deg,
        "cos_alpha": c,
        "g_dot_d_inward": g_dot,
        "fingertips_up_ok": bool(c >= float(np.cos(np.radians(ALPHA_GATE_DEG)))),
        "gravity_assists_inward": bool(g_dot > 0.0),
        "hand_R": Rh.tolist(),
    }


def vertical_gate_report(sim, *, e0_x: float | None = None) -> dict[str, Any]:
    s = sample_state(sim)
    oa = opening_alignment(sim)
    kin = finger_inner_normals_world(sim)
    sup = support_force_decomposition(sim)
    z = float(sim.data.xpos[sim.ids.object_body][2])
    e = np.asarray(s["e_signed_mm"], float)
    return {
        **oa,
        "e_signed_mm": e.tolist(),
        "theta_deg": s["e_theta_deg"],
        "fn_L": s["fn_L"],
        "fn_R": s["fn_R"],
        "fn_tot": s["fn_tot"],
        "bilateral": s["bilateral"],
        "v_rel_h": s["v_rel_h"],
        "v_rel_norm": s["v_rel_norm"],
        "obj_z": z,
        "world_height_margin": z - Z_ABORT,
        "ground_clear": bool(z >= Z_ABORT and not latched(sim)),
        "delta_ex_mm": None if e0_x is None else float(e[0] - e0_x),
        "left_align_up": kin["left_align_up"],
        "right_align_up": kin["right_align_up"],
        "Fn_z_L": sup["sides"]["L"]["Fn_z_normal"],
        "Fn_z_R": sup["sides"]["R"]["Fn_z_normal"],
        "Ft_z_L": sup["sides"]["L"]["Ft_z_friction"],
        "Ft_z_R": sup["sides"]["R"]["Ft_z_friction"],
        "P_TGT": np.asarray(P_TGT, float).tolist(),
    }


def _omega_body_align_open_up(sim, rate_scale: float) -> tuple[np.ndarray, float]:
    """Legal r_des-frame ω that rotates u_open_world toward world_up."""
    Rh = np.array(hand_R(sim), float, copy=True)
    Rd = np.array(sim.fsm.r_des, float, copy=True).reshape(3, 3)
    u = Rh @ U_OPEN_HAND
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    c = float(np.clip(np.dot(u, UP), -1.0, 1.0))
    alpha = float(np.arccos(c))
    if alpha < 1e-4:
        return np.zeros(3), 0.0
    axis_w = np.cross(u, UP)
    n = float(np.linalg.norm(axis_w))
    if n < 1e-8:
        # antiparallel: pick any orthogonal axis
        helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis_w = np.cross(u, helper)
        n = float(np.linalg.norm(axis_w))
    axis_w = axis_w / max(n, 1e-12)
    omega = float(W_MAX) * float(rate_scale)
    w_world = axis_w * omega
    w_body = Rd.T @ w_world
    # Keep within legal |a[3:6]| ≤ 1 after normalize (clip in make_action)
    return w_body, float(np.degrees(alpha))


def stage_finger_support(
    env,
    blob: dict,
    *,
    axis_name: str,
    angle_deg: float = 90.0,
    rate_scale: float = 0.18,
    support_side: str | None = None,
) -> dict[str, Any]:
    """Stage A: proven ±wz pad-up intermediate (no slip)."""
    side = support_side or ("L" if axis_name.startswith("+") else "R")
    if axis_name == "+wz":
        side = "L"
    elif axis_name == "-wz":
        side = "R"
    tr = waypoint_rotate_axis(
        env,
        blob,
        axis_name=axis_name,
        angle_deg=angle_deg,
        rate_scale=rate_scale,
        support_side=side,
        dx_abort_mm=REORI_DX_MM,
        dtheta_abort_deg=REORI_DTHETA_DEG,
    )
    if tr.get("capture") is not None:
        restore_capture(env, tr["capture"])
        tr["opening_after"] = opening_alignment(env.sim)
        tr["gate_after"] = vertical_gate_report(env.sim, e0_x=float((tr.get("e0") or [0])[0]))
    tr["stage"] = "A_finger_support"
    return tr


def stage_fingertips_up(
    env,
    blob: dict,
    *,
    rate_scale: float = 0.15,
    alpha_gate_deg: float = ALPHA_GATE_DEG,
    max_time_s: float = 8.0,
    fn_target: float = 12.0,
    dx_abort_mm: float = REORI_DX_MM,
    dtheta_abort_deg: float = REORI_DTHETA_DEG,
    settle_steps: int = 8,
) -> dict[str, Any]:
    """Stage B: closed-loop tip until opening points world-up (α ≤ gate)."""
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    e0 = np.asarray(s0["e_signed_mm"], float)
    th0 = float(s0["e_theta_deg"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    oa0 = opening_alignment(env.sim)
    if oa0["fingertips_up_ok"]:
        return {
            "ok": True,
            "reason": "already_fingertips_up",
            "stage": "B_fingertips_up",
            "n_steps": 0,
            "actions": [],
            "series": [],
            "opening0": oa0,
            "opening_final": oa0,
            "gate_final": vertical_gate_report(env.sim, e0_x=float(e0[0])),
            "capture": take_capture(env),
            "e0": e0.tolist(),
            "e_final": e0.tolist(),
        }

    ctl = ForcePController(kp=0.12, da_max=0.10)
    ctl.reset(1.0, s0["fn_tot"])
    actions, series = [], []
    lost = 0
    n_max = int(max_time_s / DT)
    last_stable = {
        "capture": take_capture(env),
        "opening": oa0,
        "gate": vertical_gate_report(env.sim, e0_x=float(e0[0])),
        "step": 0,
    }
    fail_reason = None

    for i in range(n_max):
        w_body, alpha = _omega_body_align_open_up(env.sim, rate_scale)
        st = sample_state(env.sim)
        a_g, _ = ctl.step(st["fn_tot"], fn_target)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), w_body, a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        st2 = sample_state(env.sim)
        e2 = np.asarray(st2["e_signed_mm"], float)
        oa = opening_alignment(env.sim)
        z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        kin = finger_inner_normals_world(env.sim)
        sup = support_force_decomposition(env.sim)

        if (i + 1) % 5 == 0 or i == 0:
            series.append(
                {
                    "step": i + 1,
                    "t": (i + 1) * DT,
                    "alpha_open_up_deg": oa["alpha_open_up_deg"],
                    "g_dot_d_inward": oa["g_dot_d_inward"],
                    "e_signed_mm": e2.tolist(),
                    "theta": st2["e_theta_deg"],
                    "delta_ex_mm": float(e2[0] - e0[0]),
                    "delta_theta_deg": float(st2["e_theta_deg"] - th0),
                    "left_align_up": kin["left_align_up"],
                    "right_align_up": kin["right_align_up"],
                    "Fn_z_L": sup["sides"]["L"]["Fn_z_normal"],
                    "Fn_z_R": sup["sides"]["R"]["Fn_z_normal"],
                    "Ft_z_L": sup["sides"]["L"]["Ft_z_friction"],
                    "Ft_z_R": sup["sides"]["R"]["Ft_z_friction"],
                    "fn_L": st2["fn_L"],
                    "fn_R": st2["fn_R"],
                    "bilateral": st2["bilateral"],
                    "v_rel_norm": st2["v_rel_norm"],
                    "obj_z": z,
                    "w_body": w_body.tolist(),
                }
            )

        if failed or latched(env.sim) or not st2["bilateral"]:
            fail_reason = "contact_loss"
            break
        if z < Z_ABORT:
            fail_reason = "world_height"
            break
        if abs(float(e2[0]) - float(e0[0])) > dx_abort_mm:
            fail_reason = "lateral_drift"
            break
        if abs(float(st2["e_theta_deg"]) - th0) > dtheta_abort_deg:
            fail_reason = "theta_drift"
            break

        last_stable = {
            "capture": take_capture(env),
            "opening": oa,
            "gate": vertical_gate_report(env.sim, e0_x=float(e0[0])),
            "step": i + 1,
        }
        if oa["alpha_open_up_deg"] <= alpha_gate_deg:
            # settle
            for _ in range(settle_steps):
                st = sample_state(env.sim)
                a_g, _ = ctl.step(st["fn_tot"], fn_target)
                a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
                step_legal(env.sim, a)
                actions.append(a.copy())
            oa_f = opening_alignment(env.sim)
            sf = sample_state(env.sim)
            return {
                "ok": bool(oa_f["fingertips_up_ok"] and sf["bilateral"] and oa_f["gravity_assists_inward"]),
                "reason": "ok" if oa_f["fingertips_up_ok"] else "alpha_not_met_after_settle",
                "stage": "B_fingertips_up",
                "n_steps": len(actions),
                "actions": actions,
                "series": series,
                "opening0": oa0,
                "opening_final": oa_f,
                "gate_final": vertical_gate_report(env.sim, e0_x=float(e0[0])),
                "e0": e0.tolist(),
                "e_final": sf["e_signed_mm"],
                "theta0": th0,
                "theta_final": sf["e_theta_deg"],
                "delta_ex_mm": float(np.asarray(sf["e_signed_mm"], float)[0] - e0[0]),
                "delta_theta_deg": float(sf["e_theta_deg"] - th0),
                "capture": take_capture(env),
                "min_alpha_reached_deg": oa_f["alpha_open_up_deg"],
            }

    # Failed transition — report last stable
    return {
        "ok": False,
        "reason": fail_reason or "timeout_alpha_not_reached",
        "stage": "B_fingertips_up",
        "n_steps": len(actions),
        "actions": actions,
        "series": series,
        "opening0": oa0,
        "opening_final": last_stable["opening"],
        "gate_final": last_stable["gate"],
        "last_stable_capture": last_stable["capture"],
        "last_stable_step": last_stable["step"],
        "e0": e0.tolist(),
        "e_final": last_stable["gate"].get("e_signed_mm"),
        "delta_ex_mm": last_stable["gate"].get("delta_ex_mm"),
        "min_alpha_reached_deg": last_stable["opening"].get("alpha_open_up_deg"),
        "capture": last_stable["capture"],
        "diagnosis": {
            "fail_reason": fail_reason or "timeout_alpha_not_reached",
            "alpha0_deg": oa0["alpha_open_up_deg"],
            "alpha_last_stable_deg": last_stable["opening"].get("alpha_open_up_deg"),
            "g_dot_last": last_stable["opening"].get("g_dot_d_inward"),
            "note": (
                "Transition failure (not axial-slip failure). "
                "wz alone cannot tip u_open; Stage B uses closed-loop ω = R_des.T @ (u×up)."
            ),
        },
    }


def run_support_then_vertical(
    env,
    blob: dict,
    *,
    support_axis: str,
    support_angle_deg: float = 90.0,
    support_rate: float = 0.18,
    tip_rate: float = 0.15,
) -> dict[str, Any]:
    """Full Stage A → Stage B from a common IC blob."""
    a = stage_finger_support(
        env, blob, axis_name=support_axis, angle_deg=support_angle_deg, rate_scale=support_rate
    )
    if not a.get("ok"):
        return {
            "ok": False,
            "reason": f"stage_A_failed:{a.get('reason')}",
            "support_axis": support_axis,
            "stage_A": {k: v for k, v in a.items() if k not in ("actions", "waypoints")},
            "stage_B": None,
            "actions": a.get("actions") or [],
        }
    mid = {"capture": a["capture"], "ref": blob["ref"]}
    b = stage_fingertips_up(env, mid, rate_scale=tip_rate)
    actions = list(a.get("actions") or []) + list(b.get("actions") or [])
    return {
        "ok": bool(b.get("ok")),
        "reason": b.get("reason") if b.get("ok") else f"stage_B_failed:{b.get('reason')}",
        "support_axis": support_axis,
        "stage_A": {k: v for k, v in a.items() if k not in ("actions", "waypoints")},
        "stage_B": {k: v for k, v in b.items() if k not in ("actions", "series", "last_stable_capture")},
        "stage_B_series": b.get("series"),
        "actions": actions,
        "capture": b.get("capture") or a.get("capture"),
        "gate_final": b.get("gate_final"),
        "opening_final": b.get("opening_final"),
        "diagnosis": b.get("diagnosis"),
        "stage_A_full": a,
        "stage_B_full": b,
    }
