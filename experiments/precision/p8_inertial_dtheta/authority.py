"""Rotational authority diagnostics and short legal probes (experimental)."""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.recovery.action_map import AIRBORNE_V3, W_MAX, grip_tau, normalize_action
from data_generation.oracles.precision.mechanics.dual_loop import sample_state
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.grasp_quality import body_twist
from envs.simulation.ground_scene import CYL_IXX, CYL_IZZ, CYL_M, CYL_R
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R
from experiments.precision.oscillation_assisted_reorientation.metrics import omega_rel_hand
from experiments.precision.p7_oscillatory_dtheta.geometry import (
    body_axis_for_correction,
    signed_theta_to_target,
)

DT = 0.02  # control step (step_legal)


def hand_omega_along_body(sim, body_ax: np.ndarray) -> float:
    Rh = hand_R(sim)
    _, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    aw = Rh @ np.asarray(body_ax, float)
    return float(np.dot(wh, aw))


def obj_omega_along_body(sim, body_ax: np.ndarray) -> float:
    Rh = hand_R(sim)
    _, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    aw = Rh @ np.asarray(body_ax, float)
    return float(np.dot(wo, aw))


def rel_omega_along_axis_h(sim, axis_h: np.ndarray) -> float:
    wr = omega_rel_hand(sim)
    return float(np.dot(wr, np.asarray(axis_h, float)))


def friction_torque_estimate(fn_tot: float, mu: float = 1.0) -> float:
    """Rough static friction torque bound μ N R (N·m) for opposing pads."""
    return float(mu) * float(fn_tot) * float(CYL_R)


def inertial_torque_estimate(I: float, alpha: float) -> float:
    return float(I) * float(alpha)


def authority_static_audit(sim) -> dict[str, Any]:
    th = signed_theta_to_target(sim)
    body_ax, ax_name = body_axis_for_correction(sim)
    st = sample_state(sim)
    fn = float(st.get("fn_tot") or 0)
    # Relevant inertia: tilt about diameter ≈ Ixx
    I = float(CYL_IXX)
    return {
        "W_MAX_rad_s": float(W_MAX),
        "control_dt_s": DT,
        "sim_timestep_s": float(sim.model.opt.timestep),
        "n_substeps_per_cmd": 10,
        "ori_tracking": {"kp_ori": 80.0, "kd_ori": 6.0},
        "note_command": (
            "Actions command r_des angular rate (map_action → w_des), not direct joint ω. "
            "Achieved hand ω/α depend on orientation tracking; not proportional to command step."
        ),
        "body_axis_name": ax_name,
        "body_axis": body_ax.tolist(),
        "axis_h": th["axis_h"],
        "signed_theta_deg": th["signed_theta_deg"],
        "abs_theta_deg": th["abs_theta_deg"],
        "cyl_mass_kg": float(CYL_M),
        "cyl_radius_m": float(CYL_R),
        "cyl_Ixx": float(CYL_IXX),
        "cyl_Izz": float(CYL_IZZ),
        "I_relevant": I,
        "fn_L": float(st.get("fn_L") or 0),
        "fn_R": float(st.get("fn_R") or 0),
        "fn_tot": fn,
        "tau_friction_est_Nm": friction_torque_estimate(fn),
        "alpha_needed_to_match_friction_rad_s2": (
            friction_torque_estimate(fn) / max(I, 1e-12)
        ),
        "bilateral": bool(st.get("bilateral")),
        "geom_friction_object": None,
    }


def run_accel_probe(
    env,
    blob: dict,
    *,
    label: str,
    w_cmd_rad_s: float,
    grip: float,
    n_steps: int,
    profile: str = "step",  # step | ramp
    direction: float = 1.0,
) -> dict[str, Any]:
    restore_capture(env, blob["capture"])
    th0 = signed_theta_to_target(env.sim)
    body_ax, ax_name = body_axis_for_correction(env.sim)
    body_cmd = body_ax * float(np.sign(direction) or 1.0)
    axis_h = np.asarray(th0["axis_h"], float)
    series: list[dict] = []
    w_prev = hand_omega_along_body(env.sim, body_cmd)
    dropped = lost = False
    actions: list = []

    for i in range(n_steps):
        if profile == "ramp":
            scale = min(1.0, (i + 1) / max(1.0, n_steps * 0.5))
        else:
            scale = 1.0
        w_des = body_cmd * (float(w_cmd_rad_s) * scale)
        a = np.asarray(
            normalize_action(np.zeros(3), w_des, grip_tau(float(grip)), AIRBORNE_V3),
            np.float32,
        )
        step_legal(env.sim, a)
        actions.append(a.copy())
        w_now = hand_omega_along_body(env.sim, body_cmd)
        alpha = (w_now - w_prev) / DT
        w_prev = w_now
        th = signed_theta_to_target(env.sim)
        st = sample_state(env.sim)
        row = {
            "i": i,
            "t": (i + 1) * DT,
            "w_cmd": float(w_cmd_rad_s * scale * np.sign(direction)),
            "w_hand": w_now,
            "w_obj": obj_omega_along_body(env.sim, body_cmd),
            "alpha_hand": alpha,
            "w_rel": rel_omega_along_axis_h(env.sim, axis_h),
            "theta_abs": th["abs_theta_deg"],
            "signed_theta": th["signed_theta_deg"],
            "dx": th["dx_mm"],
            "dz": th["dz_mm"],
            "e_pos": th["e_pos_mm"],
            "fn_tot": float(st.get("fn_tot") or 0),
            "fn_L": float(st.get("fn_L") or 0),
            "fn_R": float(st.get("fn_R") or 0),
            "bilateral": bool(st.get("bilateral")),
            "grip_cmd": float(grip),
        }
        series.append(row)
        failed, _ = recovery_failed(env.sim, blob["ref"], 0)
        if latched(env.sim) or failed:
            dropped = True
            break
        if not row["bilateral"] and row["fn_tot"] < 0.4:
            lost = True
            break

    th1 = signed_theta_to_target(env.sim)
    alphas = [abs(s["alpha_hand"]) for s in series]
    ws = [abs(s["w_hand"]) for s in series]
    wrels = [abs(s["w_rel"]) for s in series]
    return {
        "label": label,
        "profile": profile,
        "w_cmd_rad_s": float(w_cmd_rad_s),
        "grip_cmd": float(grip),
        "direction": float(direction),
        "body_axis_name": ax_name,
        "n_steps": len(series),
        "peak_w_hand": max(ws) if ws else 0.0,
        "peak_alpha_hand": max(alphas) if alphas else 0.0,
        "peak_w_rel": max(wrels) if wrels else 0.0,
        "mean_w_hand": float(np.mean(ws)) if ws else 0.0,
        "cmd_vs_achieved_w_ratio": (max(ws) / max(abs(w_cmd_rad_s), 1e-6)) if ws else 0.0,
        "d_theta_abs": float(th0["abs_theta_deg"] - th1["abs_theta_deg"]),
        "d_signed": float(th0["signed_theta_deg"] - th1["signed_theta_deg"]),
        "theta0": float(th0["abs_theta_deg"]),
        "theta1": float(th1["abs_theta_deg"]),
        "signed0": float(th0["signed_theta_deg"]),
        "signed1": float(th1["signed_theta_deg"]),
        "dx0": float(th0["dx_mm"]),
        "dx1": float(th1["dx_mm"]),
        "dz0": float(th0["dz_mm"]),
        "dz1": float(th1["dz_mm"]),
        "e_pos0": float(th0["e_pos_mm"]),
        "e_pos1": float(th1["e_pos_mm"]),
        "fn0": float(series[0]["fn_tot"]) if series else None,
        "fn1": float(series[-1]["fn_tot"]) if series else None,
        "bilateral_end": bool(th1["bilateral"]),
        "dropped": dropped,
        "contact_loss": lost,
        "useful_rel_rotation": bool((max(wrels) if wrels else 0.0) > 0.15),
        "series": series,
        "actions": actions,
        "capture": take_capture(env),
    }
