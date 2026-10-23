"""Finger-inner-surface support diagnostics and wx-axis reorientation (offline).

Pads face each other along ±hand-y (see envs/contacts/frames.pad_span_hand).
Left finger at +y: inner normal ≈ −hand_y (into the grasp).
Right finger at −y: inner normal ≈ +hand_y.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.recovery.action_map import W_MAX
from data_generation.oracles.precision.mechanics.dual_loop import (
    DT,
    ForcePController,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.corrections.orientation import _height_hold_v_hand
from envs.contacts.frames import extract_contacts, pad_span_hand
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R

UP = np.array([0.0, 0.0, 1.0], float)
REORI_DX_MM = 3.0
REORI_DTHETA_DEG = 3.0


def ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def finger_inner_normals_hand() -> dict[str, Any]:
    """Kinematic pad-facing normals in the hand frame (geometry convention)."""
    return {
        "left_inner_n_hand": [0.0, -1.0, 0.0],
        "right_inner_n_hand": [0.0, 1.0, 0.0],
        "note": "Panda pads face each other along ±hand-y; left at +y faces −y.",
    }


def finger_inner_normals_world(sim) -> dict[str, Any]:
    Rh = hand_R(sim)
    kin = finger_inner_normals_hand()
    nL = Rh @ np.asarray(kin["left_inner_n_hand"], float)
    nR = Rh @ np.asarray(kin["right_inner_n_hand"], float)
    return {
        "left_inner_n_world": nL.tolist(),
        "right_inner_n_world": nR.tolist(),
        "left_align_up": float(np.dot(nL, UP)),
        "right_align_up": float(np.dot(nR, UP)),
        "hand_x_world": Rh[:, 0].tolist(),
        "hand_y_world": Rh[:, 1].tolist(),
        "hand_z_world": Rh[:, 2].tolist(),
        "pad": pad_span_hand(sim),
    }


def support_force_decomposition(sim) -> dict[str, Any]:
    """Decompose contact force-on-cylinder into normal/tangential world-Z support."""
    rows, summ = extract_contacts(sim)
    out_sides = {}
    for side in ("L", "R"):
        rs = [r for r in rows if r["side"] == side]
        Fn_z = Ft_z = F_z = 0.0
        n_z_w = []
        for r in rs:
            n = np.asarray(r["n_w"], float)
            f = np.asarray(r["f_obj_w"], float)  # force on cylinder
            f_n = n * float(np.dot(f, n))
            f_t = f - f_n
            Fn_z += float(f_n[2])
            Ft_z += float(f_t[2])
            F_z += float(f[2])
            n_z_w.append(float(n[2]))
        out_sides[side] = {
            "n_contacts": len(rs),
            "F_z_total": F_z,
            "Fn_z_normal": Fn_z,
            "Ft_z_friction": Ft_z,
            "mean_contact_n_z": float(np.mean(n_z_w)) if n_z_w else None,
            "normal_support_meaningful": bool(Fn_z > 0.3),  # N, provisional
        }
    kin = finger_inner_normals_world(sim)
    return {
        "sides": out_sides,
        "F_contact_w": summ.get("F_contact_w"),
        "Fg_w": summ.get("Fg_w"),
        "mass": summ.get("mass"),
        "kinematics": kin,
        "weight": abs(float(summ["Fg_w"][2])) if summ.get("Fg_w") else None,
        "note": (
            "Fn_z_normal is the world-Z component of the normal part of force-on-cylinder. "
            "Meaningful upward normal support requires Fn_z > 0, not merely large |F| or friction."
        ),
    }


def probe_action_omega_mapping(env, blob: dict, *, n_steps: int = 40, rate: float = 0.20) -> dict[str, Any]:
    """Measure actual hand rotation under legal wx/wy/wz actions."""
    results = {}
    omega = float(W_MAX) * float(rate)
    for name, w_body in (
        ("+wx", [omega, 0.0, 0.0]),
        ("-wx", [-omega, 0.0, 0.0]),
        ("+wy", [0.0, omega, 0.0]),
        ("-wy", [0.0, -omega, 0.0]),
        ("+wz", [0.0, 0.0, omega]),
        ("-wz", [0.0, 0.0, -omega]),
    ):
        restore_capture(env, blob["capture"])
        # Copy: hand_R/xmat can alias live MjData buffers.
        Rh0 = np.array(hand_R(env.sim), float, copy=True)
        Rd0 = np.array(env.sim.fsm.r_des, float, copy=True).reshape(3, 3)
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        ctl = ForcePController(kp=0.12, da_max=0.10)
        ctl.reset(1.0, sample_state(env.sim)["fn_tot"])
        for _ in range(n_steps):
            st = sample_state(env.sim)
            a_g, _ = ctl.step(st["fn_tot"], 12.0)
            step_legal(env.sim, make_action(_height_hold_v_hand(env.sim, z_ref), w_body, a_g))
        Rh1 = np.array(hand_R(env.sim), float, copy=True)
        Rd1 = np.array(env.sim.fsm.r_des, float, copy=True).reshape(3, 3)
        nL0 = -(Rh0[:, 1])
        nR0 = Rh0[:, 1]
        nL1 = -(Rh1[:, 1])
        nR1 = Rh1[:, 1]
        results[name] = {
            "w_body_cmd": w_body,
            "hand_ang_x_deg": ang_deg(Rh0[:, 0], Rh1[:, 0]),
            "hand_ang_y_deg": ang_deg(Rh0[:, 1], Rh1[:, 1]),
            "hand_ang_z_deg": ang_deg(Rh0[:, 2], Rh1[:, 2]),
            "rdes_ang_x_deg": ang_deg(Rd0[:, 0], Rd1[:, 0]),
            "rdes_ang_y_deg": ang_deg(Rd0[:, 1], Rd1[:, 1]),
            "rdes_ang_z_deg": ang_deg(Rd0[:, 2], Rd1[:, 2]),
            "hand_tracks_rdes_y_err_deg": ang_deg(Rh1[:, 1], Rd1[:, 1]),
            "hand_tracks_rdes_z_err_deg": ang_deg(Rh1[:, 2], Rd1[:, 2]),
            "left_align_up_0": float(np.dot(nL0, UP)),
            "left_align_up_1": float(np.dot(nL1, UP)),
            "right_align_up_0": float(np.dot(nR0, UP)),
            "right_align_up_1": float(np.dot(nR1, UP)),
            "delta_left_align_up": float(np.dot(nL1, UP) - np.dot(nL0, UP)),
            "delta_right_align_up": float(np.dot(nR1, UP) - np.dot(nR0, UP)),
        }
    # Interpret which command lifts which finger
    best_L = max(results.items(), key=lambda kv: kv[1]["delta_left_align_up"])
    best_R = max(results.items(), key=lambda kv: kv[1]["delta_right_align_up"])
    return {
        "W_MAX": float(W_MAX),
        "rate": rate,
        "n_steps": n_steps,
        "trials": results,
        "best_for_left_up": {"action": best_L[0], **best_L[1]},
        "best_for_right_up": {"action": best_R[0], **best_R[1]},
        "geometry": finger_inner_normals_hand(),
        "note": "a[3:6] are ω in r_des frame; w_world = R_des @ w_body (action_map).",
    }


def waypoint_rotate_axis(
    env,
    blob: dict,
    *,
    axis_name: str,
    angle_deg: float,
    rate_scale: float = 0.15,
    fn_target: float = 12.0,
    waypoint_deg: float = 10.0,
    dx_abort_mm: float = REORI_DX_MM,
    dtheta_abort_deg: float = REORI_DTHETA_DEG,
    support_side: str | None = None,
) -> dict[str, Any]:
    """Slow signed wrist rotation about a legal axis with support diagnostics."""
    axis_map = {
        "+wx": np.array([1.0, 0.0, 0.0]),
        "-wx": np.array([-1.0, 0.0, 0.0]),
        "+wy": np.array([0.0, 1.0, 0.0]),
        "-wy": np.array([0.0, -1.0, 0.0]),
        "+wz": np.array([0.0, 0.0, 1.0]),
        "-wz": np.array([0.0, 0.0, -1.0]),
    }
    if axis_name not in axis_map:
        return {"ok": False, "reason": f"unknown_axis_{axis_name}"}
    axis = axis_map[axis_name]
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    e0 = np.asarray(s0["e_signed_mm"], float)
    th0 = float(s0["e_theta_deg"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    kin0 = finger_inner_normals_world(env.sim)
    sup0 = support_force_decomposition(env.sim)

    omega = float(W_MAX) * float(rate_scale)
    n_total = max(1, int(round(np.radians(angle_deg) / max(omega, 1e-6) / DT)))
    n_wp = max(1, int(round(np.radians(waypoint_deg) / max(omega, 1e-6) / DT)))
    ctl = ForcePController(kp=0.12, da_max=0.10)
    ctl.reset(1.0, s0["fn_tot"])
    actions, waypoints = [], []
    lost = 0

    for i in range(n_total):
        st = sample_state(env.sim)
        a_g, _ = ctl.step(st["fn_tot"], fn_target)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), axis * omega, a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        st2 = sample_state(env.sim)
        e2 = np.asarray(st2["e_signed_mm"], float)
        if failed or latched(env.sim) or not st2["bilateral"]:
            return {
                "ok": False,
                "reason": "contact_loss",
                "axis_name": axis_name,
                "angle_deg": angle_deg,
                "rate_scale": rate_scale,
                "n_steps": i + 1,
                "actions": actions,
                "waypoints": waypoints,
                "support0": sup0,
            }
        dx = abs(float(e2[0]) - float(e0[0]))
        dth = abs(float(st2["e_theta_deg"]) - th0)
        if dx > dx_abort_mm or dth > dtheta_abort_deg:
            return {
                "ok": False,
                "reason": "reori_drift_abort",
                "axis_name": axis_name,
                "angle_deg": angle_deg,
                "rate_scale": rate_scale,
                "n_steps": i + 1,
                "actions": actions,
                "waypoints": waypoints,
                "e0": e0.tolist(),
                "e_now": e2.tolist(),
                "delta_ex_mm": float(e2[0] - e0[0]),
                "delta_theta_deg": float(st2["e_theta_deg"] - th0),
                "support0": {"sides": sup0["sides"], "weight": sup0["weight"]},
                "support_now": support_force_decomposition(env.sim),
                "kinematics_now": finger_inner_normals_world(env.sim),
                "capture": take_capture(env),
            }
        if (i + 1) % n_wp == 0 or i == n_total - 1:
            kin = finger_inner_normals_world(env.sim)
            sup = support_force_decomposition(env.sim)
            waypoints.append(
                {
                    "step": i + 1,
                    "angle_approx_deg": (i + 1) * float(np.degrees(omega * DT)),
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
                    "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                }
            )

    sf = sample_state(env.sim)
    e_f = np.asarray(sf["e_signed_mm"], float)
    kin_f = finger_inner_normals_world(env.sim)
    sup_f = support_force_decomposition(env.sim)
    side = support_side or ("L" if kin_f["left_align_up"] >= kin_f["right_align_up"] else "R")
    ok = (
        sf["bilateral"]
        and abs(float(e_f[0]) - float(e0[0])) <= dx_abort_mm
        and abs(float(sf["e_theta_deg"]) - th0) <= dtheta_abort_deg
    )
    return {
        "ok": ok,
        "reason": "ok" if ok else "end_out_of_budget",
        "axis_name": axis_name,
        "angle_deg": angle_deg,
        "rate_scale": rate_scale,
        "support_side": side,
        "n_steps": n_total,
        "actions": actions,
        "waypoints": waypoints,
        "e0": e0.tolist(),
        "e_final": e_f.tolist(),
        "theta0": th0,
        "theta_final": sf["e_theta_deg"],
        "delta_ex_mm": float(e_f[0] - e0[0]),
        "delta_theta_deg": float(sf["e_theta_deg"] - th0),
        "kinematics0": kin0,
        "kinematics_final": kin_f,
        "support0": {"sides": sup0["sides"], "weight": sup0["weight"]},
        "support_final": {"sides": sup_f["sides"], "weight": sup_f["weight"]},
        "selected_Fn_z": sup_f["sides"][side]["Fn_z_normal"],
        "selected_Ft_z": sup_f["sides"][side]["Ft_z_friction"],
        "selected_align_up": kin_f["left_align_up"] if side == "L" else kin_f["right_align_up"],
        "normal_support_ok": bool(sup_f["sides"][side]["Fn_z_normal"] > 0.3),
        "min_obj_z": float(min((w["obj_z"] for w in waypoints), default=z_ref)),
        "capture": take_capture(env),
    }


def search_finger_support(
    env,
    blob: dict,
    *,
    candidate: str,
    angles: tuple[float, ...] = (30.0, 45.0, 60.0, 75.0, 90.0),
    rates: tuple[float, ...] = (0.12, 0.15, 0.20),
) -> dict[str, Any]:
    """candidate in {L, R}; pick axis sign from mapping probe if provided via axes."""
    # Default axes: ±wx are the orthogonal candidates; include ±wz as fallback
    if candidate == "L":
        axes = ("+wx", "-wx", "+wz", "-wz")
        side = "L"
    else:
        axes = ("+wx", "-wx", "+wz", "-wz")
        side = "R"
    trials = []
    best = None
    for axis_name in axes:
        for ang in angles:
            for rate in rates:
                tr = waypoint_rotate_axis(
                    env,
                    blob,
                    axis_name=axis_name,
                    angle_deg=ang,
                    rate_scale=rate,
                    support_side=side,
                )
                summary = {k: v for k, v in tr.items() if k not in ("actions", "waypoints", "capture")}
                trials.append(summary)
                print(
                    f"  [{candidate}] {axis_name}@{ang}° r={rate}: ok={tr.get('ok')} "
                    f"align={tr.get('selected_align_up')} Fn_z={tr.get('selected_Fn_z')} "
                    f"Ft_z={tr.get('selected_Ft_z')} Δex={tr.get('delta_ex_mm')} "
                    f"reason={tr.get('reason')}",
                    flush=True,
                )
                if not tr.get("ok"):
                    continue
                # Score: normal support, alignment with up, preserve lateral/theta
                score = (
                    40.0 * float(tr.get("selected_Fn_z") or 0.0)
                    + 20.0 * float(tr.get("selected_align_up") or 0.0)
                    - 2.0 * abs(float(tr.get("delta_ex_mm") or 0.0))
                    - 2.0 * abs(float(tr.get("delta_theta_deg") or 0.0))
                    - 0.05 * ang
                )
                summary["score"] = score
                if best is None or score > best["score"]:
                    best = {**tr, "score": score}
    return {
        "candidate": candidate,
        "support_side": side,
        "n_trials": len(trials),
        "trials": trials,
        "best": None if best is None else {k: v for k, v in best.items() if k not in ("actions", "waypoints")},
        "best_full": best,
        "any_normal_support": bool(best and best.get("normal_support_ok")),
    }
