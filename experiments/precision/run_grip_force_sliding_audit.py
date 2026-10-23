#!/usr/bin/env python
"""Contact-force audit + controlled gravity-slip ID at fixed tilt (Plan A post-pivot).

Does not optimize SUCCESS_NEW. Does not modify gripper controller.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import (  # noqa: E402
    TAU_OPEN,
    TAU_SECURE,
    W_MAX,
    grip_action,
    grip_tau,
)
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.primitives import (  # noqa: E402
    _action_vw,
    _wrist,
    pivot_segment_metrics,
)
from envs.ballistic_impact import PAIR_MU  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.physical_recovery import physical_pack  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world, hand_R, measure, p_com_hand  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

PRIOR = ROOT / "results/diagnostics/position_preserving_pivot"
STATE_A = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_a__ball_train_v2_0070__stable.pkl"
OUT = ROOT / "results/diagnostics/grip_force_sliding_audit"

DT = 0.02
FPS = 25
TILT_TARGET_DEG = 60.0
SETTLE_S = 0.30
OBSERVE_S = 0.80
BRAKE_S = 0.40
# Grip actions from firm → light (a=+1 secure … a=-1 open)
GRIP_LADDER = (1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.48, float(grip_action(-5.0)), -0.70, -0.85, -1.0)


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


def parse_ppivot_tag(tag: str) -> dict:
    rest = tag[len("ppiv_v") :]
    pre_w, after_w = rest.split("_w", 1)
    vx_s, vz_s = pre_w.split("_", 1)
    pre_g, after_g = after_w.split("_g", 1)
    wx_s, wy_s = pre_g.split("_", 1)
    grip_s, dur_s = after_g.split("_d", 1)
    return {k: float(v) for k, v in zip(("vx", "vz", "wx", "wy", "grip", "dur"), (vx_s, vz_s, wx_s, wy_s, grip_s, dur_s))}


def pivot_actions(tag: str) -> list:
    p = parse_ppivot_tag(tag)
    a_p = _action_vw([p["vx"], 0.0, p["vz"]], [p["wx"], p["wy"], 0.0], p["grip"])
    a_b = _wrist(0.0, 0.0, 1.0)
    return [a_p] * max(1, int(round(p["dur"] / DT))) + [a_b] * 8


def grip_implementation_audit(sim) -> dict:
    """Static documentation of how a[6] reaches actuators (no controller edits)."""
    act_name = None
    try:
        act_name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 7)
    except Exception:
        act_name = "actuator8?"
    tendon = None
    if sim.model.nu > 7:
        # tendon actuator: trntype
        try:
            trn = int(sim.model.actuator_trntype[7])
            trnid = int(sim.model.actuator_trnid[7, 0])
            if trn == int(mujoco.mjtTrn.mjTRN_TENDON):
                tendon = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_TENDON, trnid)
        except Exception:
            pass
    return {
        "action_channel": "a[6] in legal 7D, range to [-1,1]",
        "mapping": "grip_tau(a)=TAU_OPEN+(a+1)/2*(TAU_SECURE-TAU_OPEN)",
        "TAU_OPEN_Nm": float(TAU_OPEN),
        "TAU_SECURE_Nm": float(TAU_SECURE),
        "physical_meaning": (
            "a=+1 → τ=-18 N·m (closing / secure); a=-1 → τ=+2 N·m (opening). "
            "Command is tendon motor control torque, not a force setpoint and not a position command."
        ),
        "step_legal_path": (
            "map_action → tau=grip_tau(a[6]) → clip to ctrl_low/high[7] → "
            "apply_cartesian_ctrl(..., gripper_tau=tau) → data.ctrl[7]=tau → mj_step ×10"
        ),
        "actuator_index": 7,
        "actuator_name": act_name,
        "tendon_name": tendon,
        "ctrl_range": [float(sim.ids.ctrl_low[7]), float(sim.ids.ctrl_high[7])],
        "finger_symmetry": "Single tendon 'split' drives both finger joints — symmetric, not independent fingers",
        "saturation": "ctrl clipped to [-50,50] in XML; recovery map only uses [-18,+2]",
        "fsm_fg_cmd": "sim.fsm.fg_cmd = -tau (positive squeeze effort mirror); not the contact force",
        "PAIR_MU_sliding": float(PAIR_MU),
        "note_unload": (
            "Historical 'unload' uses a≈-0.3 (τ≈-5) or a=-0.48 (τ≈-3.2), lighter than secure a=1 (τ=-18). "
            "Whether measured pad normal force drops must be checked via mj_contactForce, not inferred from a[6]."
        ),
        "ladder_preview": [{"a": float(a), "tau": float(grip_tau(a))} for a in GRIP_LADDER],
    }


def contact_force_detail(sim) -> dict:
    """Per-contact normals/tangents via mj_contactForce; world-frame arrows."""
    obj = int(sim.ids.object_body)
    left, right = int(sim.ids.left_body), int(sim.ids.right_body)
    mu = float(PAIR_MU)
    details = []
    fn_L = ft_L = fn_R = ft_R = 0.0
    nL = nR = 0
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
        # Contact frame: first row = normal (geom1→geom2)
        fr = np.asarray(c.frame, float).reshape(3, 3)
        n_w = fr[0]
        t1_w, t2_w = fr[1], fr[2]
        fn_signed = float(force[0])
        ft_vec = float(force[1]) * t1_w + float(force[2]) * t2_w
        fn = abs(fn_signed)
        ft = float(np.linalg.norm(force[1:3]))
        f_w = fn_signed * n_w + ft_vec
        side = "L" if other == left else "R"
        util = ft / max(mu * fn, 1e-9)
        entry = {
            "side": side,
            "pos": np.asarray(c.pos, float).tolist(),
            "fn": fn,
            "ft": ft,
            "fn_signed": fn_signed,
            "normal_world": n_w.tolist(),
            "f_world": f_w.tolist(),
            "ft_world": ft_vec.tolist(),
            "friction_util_ft_over_muFn": util,
            "dist": float(c.dist),
            "mu_ref": mu,
        }
        details.append(entry)
        if side == "L":
            nL += 1
            fn_L += fn
            ft_L += ft
        else:
            nR += 1
            fn_R += fn
            ft_R += ft
    return {
        "nL": nL,
        "nR": nR,
        "bilateral": nL > 0 and nR > 0,
        "fn_L": fn_L,
        "fn_R": fn_R,
        "ft_L": ft_L,
        "ft_R": ft_R,
        "fn_tot": fn_L + fn_R,
        "ft_tot": ft_L + ft_R,
        "mu": mu,
        "contacts": details,
    }


def grip_actuator_force(sim) -> float:
    af = np.asarray(sim.data.actuator_force, float)
    return float(af[7]) if af.size > 7 else float("nan")


def sample(sim) -> dict:
    pack = physical_pack(sim)
    m = measure(sim, U_TGT, P_TGT)
    cf = contact_force_detail(sim)
    p = np.asarray(m["p_c_h"], float)
    e = p - np.asarray(P_TGT, float)
    g_h = np.asarray(pack["g_h"], float)
    v = np.asarray(pack["v_rel_h"], float)
    aper = float(np.mean(sim.data.qpos[sim.ids.finger_jnt]))
    return {
        "e_signed_mm": (e * 1e3).tolist(),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "p_c_h": p.tolist(),
        "g_h": g_h.tolist(),
        "g_x": float(g_h[0]),
        "v_rel_h": v.tolist(),
        "v_x": float(v[0]),
        "v_y": float(v[1]),
        "v_z": float(v[2]),
        "v_rel_norm": float(np.linalg.norm(v)),
        "w_rel_norm": float(np.linalg.norm(pack["w_rel_h"])),
        "grip_ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float("nan"),
        "grip_af": grip_actuator_force(sim),
        "aperture_m": aper,
        "fn_L": cf["fn_L"],
        "fn_R": cf["fn_R"],
        "ft_L": cf["ft_L"],
        "ft_R": cf["ft_R"],
        "fn_tot": cf["fn_tot"],
        "ft_tot": cf["ft_tot"],
        "bilateral": cf["bilateral"],
        "nL": cf["nL"],
        "nR": cf["nR"],
        "contacts": cf["contacts"],
        "ground_drop": bool(latched(sim)),
    }


def tilt_to_target(env, blob, target_deg=TILT_TARGET_DEG, sign=+1.0):
    """Open-loop wy to ~target_deg; return actions + end metrics."""
    omega = float(W_MAX)
    dur = float(np.clip(np.radians(target_deg) / omega, 0.1, 1.2))
    n = max(1, int(round(dur / DT)))
    a = _wrist(0.0, sign * omega, 1.0)
    acts = [a] * n + [_wrist(0.0, 0.0, 1.0)] * 8
    restore_capture(env, blob["capture"])
    s0 = sample(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    lost = 0
    series = []
    for i, act in enumerate(acts):
        step_legal(env.sim, act)
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        if i % 2 == 0 or i == len(acts) - 1:
            s = sample(env.sim)
            s["step"] = i
            s["t_rel"] = (i + 1) * DT
            series.append({k: s[k] for k in s if k != "contacts"})
        if failed or latched(env.sim):
            return {"ok": False, "reason": "DROP_DURING_TILT", "actions": acts, "series": series}
    s1 = sample(env.sim)
    dp = (np.asarray(s1["p_c_h"], float) - p0) * 1e3
    return {
        "ok": bool(s1["bilateral"] and s1["g_x"] < -4.0),
        "actions": acts,
        "n_tilt": n,
        "dur_s": dur,
        "sign": sign,
        "g_x0": s0["g_x"],
        "g_x1": s1["g_x"],
        "theta0": s0["e_theta_deg"],
        "theta1": s1["e_theta_deg"],
        "e_signed0": s0["e_signed_mm"],
        "e_signed1": s1["e_signed_mm"],
        "delta_p_rel_mm": dp.tolist(),
        "substantial_shift": float(np.linalg.norm(dp)) > 5.0,
        "bilateral": s1["bilateral"],
        "series_tail": series[-6:],
        "end_sample": {k: s1[k] for k in s1 if k != "contacts"},
    }


def classify_mode(series: list, p0: np.ndarray, desire_dpx_sign: float) -> str:
    if not series:
        return "no_data"
    if any(not r["bilateral"] or r["ground_drop"] for r in series):
        if series[-1]["nL"] == 0 and series[-1]["nR"] == 0:
            return "complete_grasp_loss"
        return "partial_contact_loss"
    vx = [r["v_x"] for r in series]
    vz = [r["v_z"] for r in series]
    wn = [r["w_rel_norm"] for r in series]
    p1 = np.asarray(series[-1]["p_c_h"], float)
    dp = (p1 - p0) * 1e3
    max_vx = max(abs(x) for x in vx)
    max_vz = max(abs(z) for z in vz)
    max_w = max(wn)
    # sticking: tiny relative motion
    if max_vx < 0.01 and max_vz < 0.01 and max_w < 0.3 and abs(dp[0]) < 0.5:
        return "sticking"
    # rolling/pivot: high angular, modest COM x
    if max_w > 1.5 and abs(dp[0]) < 2.0:
        return "rolling_or_pivoting"
    # useful controlled hand-x
    if (dp[0] * desire_dpx_sign) >= 2.0 and abs(dp[2]) < abs(dp[0]) + 2.0 and max_vx < 0.25:
        return "controlled_sliding_hand_x"
    # uncontrolled
    if max_vx > 0.35 or max_vz > 0.35 or abs(dp[2]) > 8.0:
        return "uncontrolled_slipping"
    if abs(dp[0]) >= 1.0 and abs(dp[2]) > abs(dp[0]):
        return "motion_dominated_by_hand_z"
    if abs(dp[0]) >= 1.0:
        return "partial_hand_x_motion"
    return "sticking_or_ineffective"


def run_grip_trial(env, tilt_blob, grip_a: float, desire_sign: float) -> dict:
    """Independent trial: reset to tilted snap, set grip, hold orientation, observe."""
    restore_capture(env, tilt_blob["capture"])
    s0 = sample(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    # settle under grip with zero wrist/trans
    a_hold = _wrist(0.0, 0.0, grip_a)
    series = []
    lost = 0
    n_settle = max(1, int(round(SETTLE_S / DT)))
    n_obs = max(1, int(round(OBSERVE_S / DT)))
    stopped = None
    for i in range(n_settle + n_obs):
        step_legal(env.sim, a_hold)
        failed, lost = recovery_failed(env.sim, tilt_blob["ref"], lost)
        s = sample(env.sim)
        s["step"] = i
        s["t_rel"] = (i + 1) * DT
        s["phase"] = "settle" if i < n_settle else "observe"
        s["grip_a"] = grip_a
        s["grip_tau_cmd"] = float(grip_tau(grip_a))
        series.append({k: s[k] for k in s if k != "contacts"})
        if failed or latched(env.sim) or (not s["bilateral"] and i > 3):
            stopped = "contact_loss_or_drop"
            break
        # stop if runaway
        if i >= n_settle and (s["v_rel_norm"] > 0.5 or abs(s["v_z"]) > 0.4):
            stopped = "uncontrolled_motion"
            break
    mode = classify_mode(series, p0, desire_sign)
    s_end = series[-1]
    dp = (np.asarray(s_end["p_c_h"], float) - p0) * 1e3
    # mean forces after settle
    obs = [r for r in series if r.get("phase") == "observe"] or series[-5:]
    return {
        "grip_a": grip_a,
        "tau_cmd": float(grip_tau(grip_a)),
        "stopped": stopped,
        "mode": mode,
        "fn_L_mean": float(np.mean([r["fn_L"] for r in obs])),
        "fn_R_mean": float(np.mean([r["fn_R"] for r in obs])),
        "fn_tot_mean": float(np.mean([r["fn_tot"] for r in obs])),
        "ft_tot_mean": float(np.mean([r["ft_tot"] for r in obs])),
        "fn_tot_end": s_end["fn_tot"],
        "grip_ctrl_end": s_end["grip_ctrl"],
        "grip_af_end": s_end["grip_af"],
        "aperture_end": s_end["aperture_m"],
        "v_x_mean": float(np.mean([r["v_x"] for r in obs])),
        "v_z_mean": float(np.mean([r["v_z"] for r in obs])),
        "v_x_max_abs": float(max(abs(r["v_x"]) for r in obs)),
        "v_z_max_abs": float(max(abs(r["v_z"]) for r in obs)),
        "delta_p_rel_mm": dp.tolist(),
        "dpx_mm": float(dp[0]),
        "dpy_mm": float(dp[1]),
        "dpz_mm": float(dp[2]),
        "theta0": s0["e_theta_deg"],
        "theta_end": s_end["e_theta_deg"],
        "e_pos0": s0["e_pos_mm"],
        "e_pos_end": s_end["e_pos_mm"],
        "e_signed0": s0["e_signed_mm"],
        "e_signed_end": s_end["e_signed_mm"],
        "g_x": s0["g_x"],
        "bilateral_end": s_end["bilateral"],
        "series": series,
    }


def brake_and_restore(env, tilt_blob, grip_a: float, tilt_meta: dict, desire_sign: float) -> dict:
    """After a promising observe window: regrip, brake, untilt; measure surviving Δp."""
    restore_capture(env, tilt_blob["capture"])
    s0 = sample(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    a_light = _wrist(0.0, 0.0, grip_a)
    n_obs = max(1, int(round(OBSERVE_S / DT)))
    lost = 0
    for _ in range(max(1, int(round(SETTLE_S / DT))) + n_obs):
        step_legal(env.sim, a_light)
        failed, lost = recovery_failed(env.sim, tilt_blob["ref"], lost)
        if failed:
            return {"ok": False, "reason": "lost_before_brake"}
    s_peak = sample(env.sim)
    dp_peak = (np.asarray(s_peak["p_c_h"], float) - p0) * 1e3
    # regrip + brake
    a_sec = _wrist(0.0, 0.0, 1.0)
    for _ in range(max(1, int(round(BRAKE_S / DT)))):
        step_legal(env.sim, a_sec)
        failed, lost = recovery_failed(env.sim, tilt_blob["ref"], lost)
        if failed:
            return {"ok": False, "reason": "lost_during_brake", "dpx_peak": float(dp_peak[0])}
    s_brake = sample(env.sim)
    dp_brake = (np.asarray(s_brake["p_c_h"], float) - p0) * 1e3
    # untilt
    sign = float(tilt_meta["sign"])
    n_tilt = int(tilt_meta["n_tilt"])
    a_ret = _wrist(0.0, -sign * float(W_MAX), 1.0)
    for _ in range(n_tilt):
        step_legal(env.sim, a_ret)
        failed, lost = recovery_failed(env.sim, tilt_blob["ref"], lost)
        if failed:
            return {
                "ok": False,
                "reason": "lost_during_untilt",
                "dpx_peak": float(dp_peak[0]),
                "dpx_brake": float(dp_brake[0]),
            }
    for _ in range(8):
        step_legal(env.sim, a_sec)
    s_end = sample(env.sim)
    dp_end = (np.asarray(s_end["p_c_h"], float) - p0) * 1e3
    useful = (float(dp_end[0]) * desire_sign) >= 2.0 and abs(float(dp_end[2])) < 8.0
    return {
        "ok": True,
        "grip_a": grip_a,
        "dpx_peak_mm": float(dp_peak[0]),
        "dpz_peak_mm": float(dp_peak[2]),
        "dpx_after_brake_mm": float(dp_brake[0]),
        "dpz_after_brake_mm": float(dp_brake[2]),
        "dpx_after_restore_mm": float(dp_end[0]),
        "dpz_after_restore_mm": float(dp_end[2]),
        "delta_p_after_restore_mm": dp_end.tolist(),
        "theta_end": s_end["e_theta_deg"],
        "e_signed_end": s_end["e_signed_mm"],
        "e_pos_end": s_end["e_pos_mm"],
        "bilateral_end": s_end["bilateral"],
        "useful_surviving_hand_x": useful,
    }


def make_plots(trials: list, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return {"error": str(e)}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    a = [t["grip_a"] for t in trials]
    tau = [t["tau_cmd"] for t in trials]
    fn = [t["fn_tot_mean"] for t in trials]
    vx = [t["v_x_mean"] for t in trials]
    vz = [t["v_z_mean"] for t in trials]
    dpx = [t["dpx_mm"] for t in trials]
    modes = [t["mode"] for t in trials]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(a, fn, "o-", label="fn_tot mean")
    ax.set_xlabel("grip action a[6]")
    ax.set_ylabel("measured ΣFn (N)")
    ax.set_title("Grip command vs measured normal force")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    p = out_dir / "grip_vs_fn.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["grip_vs_fn"] = str(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fn, vx, "o", label="v_x mean")
    ax.plot(fn, vz, "s", label="v_z mean")
    for i, m in enumerate(modes):
        ax.annotate(m[:6], (fn[i], vx[i]), fontsize=7, alpha=0.7)
    ax.set_xlabel("measured ΣFn (N)")
    ax.set_ylabel("hand-frame relative velocity (m/s)")
    ax.set_title("Normal force vs relative velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "fn_vs_vrel.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["fn_vs_vrel"] = str(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(a, dpx, "o-", color="C0")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("grip action a[6]")
    ax.set_ylabel("Δp_rel.x (mm)")
    ax.set_title("Grip command vs surviving hand-x COM displacement")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    p = out_dir / "grip_vs_dpx.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["grip_vs_dpx"] = str(p)

    # COM traces for a mid and light trial
    fig, ax = plt.subplots(figsize=(7, 4))
    for t in trials:
        if t["grip_a"] in (1.0, -0.48, -0.85) or t["mode"] not in ("sticking", "sticking_or_ineffective"):
            xs = [r["t_rel"] for r in t["series"]]
            pxs = [(np.asarray(r["p_c_h"])[0] - np.asarray(t["series"][0]["p_c_h"])[0]) * 1e3 for r in t["series"]]
            ax.plot(xs, pxs, label=f"a={t['grip_a']:+.2f}/{t['mode'][:10]}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Δp_x (mm)")
    ax.set_title("Hand-x COM displacement traces")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    p = out_dir / "dpx_traces.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["dpx_traces"] = str(p)

    # tau vs fn
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tau, fn, "o-")
    ax.set_xlabel("τ_cmd (N·m)")
    ax.set_ylabel("ΣFn (N)")
    ax.set_title("Commanded tendon torque vs measured contact normal force")
    ax.grid(True, alpha=0.3)
    p = out_dir / "tau_vs_fn.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["tau_vs_fn"] = str(p)
    return paths


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    bar = min(out.shape[0], 16 + 15 * len(lines))
    out[:bar] = (10, 10, 10)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 3
        for line in lines:
            dr.text((5, y), line, fill=(255, 255, 255))
            y += 15
        return np.asarray(im)
    except Exception:
        return out


def render_diagnostic_video(env, tilt_blob, trial: dict, dest: Path, title: str):
    """Show real geoms + contact force arrows + overlays for one grip trial."""
    restore_capture(env, tilt_blob["capture"])
    p0 = np.asarray(sample(env.sim)["p_c_h"], float)
    ptgt = np.asarray(P_TGT, float)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 480, 640)
    rh = mujoco.Renderer(env.sim.model, 480, 640)
    # Enable contact viz flags on scene options if available
    for r in (rw, rh):
        try:
            r.scene.flags  # noqa
        except Exception:
            pass
    cam_w = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.55, "azimuth": 145, "elevation": -20})
    cam_h = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.35, "azimuth": 145, "elevation": -25})
    frames = []
    grip_a = trial["grip_a"]
    a_hold = _wrist(0.0, 0.0, grip_a)
    n_total = max(1, int(round((SETTLE_S + OBSERVE_S) / DT)))

    def cams():
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.55 * po + 0.45 * ph
        cam_w.distance = 0.50
        cam_w.azimuth = 140.0
        cam_w.elevation = -18.0
        look = ph + Rh @ np.array([0.0, 0.0, 0.08])
        pos = ph + Rh @ np.array([0.22, -0.28, 0.14])
        vec = pos - look
        dist = float(np.linalg.norm(vec))
        cam_h.lookat[:] = look
        cam_h.distance = max(dist, 0.25)
        cam_h.azimuth = float(np.degrees(np.arctan2(vec[1], vec[0])))
        cam_h.elevation = float(np.degrees(np.arcsin(np.clip(vec[2] / max(dist, 1e-9), -1, 1))))

    def draw(scn):
        scn.ngeom = 0
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        # COM cyan, target COM yellow (hand-frame target expressed in world)
        add_sphere(scn, po, [0.1, 0.95, 1.0, 1.0], 0.006)
        add_sphere(scn, ph + Rh @ ptgt, [1.0, 0.9, 0.1, 1.0], 0.005)
        # axes: green=cyl axis, red=hand-x desire direction
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.2, 1.0], 0.0025)
        hx = Rh[:, 0]
        add_capsule(scn, po, po - 0.06 * hx, [1.0, 0.2, 0.2, 1.0], 0.003)  # desire −hand-x
        # contact forces
        cf = contact_force_detail(env.sim)
        for c in cf["contacts"][:8]:
            pos = np.asarray(c["pos"], float)
            add_sphere(scn, pos, [1.0, 0.5, 0.0, 1.0], 0.0035)
            n = np.asarray(c["normal_world"], float)
            scale_n = 0.002 * c["fn"]
            add_capsule(scn, pos, pos + scale_n * n, [0.2, 0.4, 1.0, 1.0], 0.002)  # normal blue
            ft_w = np.asarray(c["ft_world"], float)
            if np.linalg.norm(ft_w) > 1e-6:
                add_capsule(scn, pos, pos + 0.004 * ft_w, [1.0, 0.2, 0.9, 1.0], 0.002)  # tangent magenta

    def frame(phase):
        cams()
        s = sample(env.sim)
        dp = (np.asarray(s["p_c_h"], float) - p0) * 1e3
        lines = [
            f"{phase} {title} t={float(env.sim.data.time)-t0:.2f}s a={grip_a:+.2f} τ={grip_tau(grip_a):+.1f}",
            f"Fn L/R={s['fn_L']:.1f}/{s['fn_R']:.1f}N  Ft={s['ft_tot']:.1f}  ctrl7={s['grip_ctrl']:.1f} af={s['grip_af']:.1f}",
            f"θ={s['e_theta_deg']:.1f} e_x={s['e_signed_mm'][0]:+.1f} Δp_x={dp[0]:+.1f} Δp_z={dp[2]:+.1f} "
            f"vx={s['v_x']:+.3f} bil={int(s['bilateral'])}",
            "markers: cyan=COM yellow=P_TGT green=cylAxis red=desire-x orange=contact blue=Fn magenta=Ft",
        ]
        rw.update_scene(env.sim.data, cam_w)
        try:
            rw.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        except Exception:
            pass
        draw(rw.scene)
        iw = overlay(np.asarray(rw.render(), np.uint8), ["WORLD"] + lines)
        rh.update_scene(env.sim.data, cam_h)
        draw(rh.scene)
        ih = overlay(np.asarray(rh.render(), np.uint8), ["HAND-REL close-up"] + lines)
        return np.concatenate([iw, ih], axis=1)

    # presentation
    img = frame("BEFORE")
    for _ in range(int(3 * FPS)):
        frames.append(img)
    dup = max(1, int(round((DT / 0.25) * FPS)))
    for i in range(n_total):
        step_legal(env.sim, a_hold)
        phase = "SETTLE" if i * DT < SETTLE_S else "OBSERVE"
        img = frame(phase)
        for _ in range(dup):
            frames.append(img)
    img = frame("AFTER")
    for _ in range(int(3 * FPS)):
        frames.append(img)
    rw.close()
    rh.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "duration_s": len(frames) / FPS}


def main():
    t_wall = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(parents=True, exist_ok=True)

    search = json.loads((PRIOR / "search.json").read_text(encoding="utf-8"))
    tag = search["A_raw"]["best"]["tag"]
    state = pickle.loads(STATE_A.read_bytes())
    env = make_env()
    blob0 = {"capture": state["capture"], "ref": state["ref"]}

    audit = grip_implementation_audit(env.sim)
    (OUT / "gripper_implementation.json").write_text(json.dumps(jsafe(audit), indent=2), encoding="utf-8")
    print("=== Gripper implementation ===", flush=True)
    print(json.dumps(audit, indent=2)[:800], flush=True)

    # Pivot
    piv = pivot_actions(tag)
    met = pivot_segment_metrics(env, blob0, piv, record_series=False)
    print(f"Pivot θ {met['theta0']:.3f}->{met['theta1']:.3f} Δp={met['delta_p_rel_mm']:.3f}", flush=True)
    restore_capture(env, blob0["capture"])
    for a in piv:
        step_legal(env.sim, a)
    post = {"capture": take_capture(env), "ref": state["ref"]}
    s_post = sample(env.sim)
    desire_sign = -1.0 if s_post["e_signed_mm"][0] > 0 else 1.0
    print(f"Post-pivot e={s_post['e_signed_mm']} desire_dpx_sign={desire_sign}", flush=True)

    # Prefer tilt sign with g_x < 0
    print("=== Tilt to ~60° ===", flush=True)
    best_tilt = None
    for sign in (+1.0, -1.0):
        tm = tilt_to_target(env, post, TILT_TARGET_DEG, sign)
        print(f"  sign={sign:+.0f} ok={tm['ok']} g_x={tm.get('g_x1')} dp={tm.get('delta_p_rel_mm')} shift={tm.get('substantial_shift')}", flush=True)
        if tm["ok"] and (best_tilt is None or tm["g_x1"] < best_tilt["g_x1"]):
            best_tilt = tm
    if best_tilt is None:
        # accept best bilateral even if g_x weak
        for sign in (+1.0, -1.0):
            tm = tilt_to_target(env, post, TILT_TARGET_DEG, sign)
            if tm.get("bilateral") and (best_tilt is None or tm["g_x1"] < best_tilt.get("g_x1", 99)):
                best_tilt = tm
    if best_tilt is None:
        raise RuntimeError("tilt failed both signs")

    restore_capture(env, post["capture"])
    for a in best_tilt["actions"]:
        step_legal(env.sim, a)
    tilt_cap = take_capture(env)
    tilt_blob = {"capture": tilt_cap, "ref": state["ref"]}
    s_tilt = sample(env.sim)
    (OUT / "snapshots" / "A_post_pivot_tilt60.pkl").write_bytes(
        pickle.dumps({"capture": tilt_cap, "ref": state["ref"], "tilt": best_tilt, "sample": {k: s_tilt[k] for k in s_tilt if k != "contacts"}})
    )
    print(f"Tilted snap g_x={s_tilt['g_x']:.2f} θ={s_tilt['e_theta_deg']:.2f} e={s_tilt['e_signed_mm']}", flush=True)

    # No-tilt unload control from post-pivot (not tilted)
    print("=== Controls + grip ladder ===", flush=True)
    ctrl_notilt = run_grip_trial(env, post, -0.48, desire_sign)
    print(f"  no-tilt unload a=-0.48 mode={ctrl_notilt['mode']} fn={ctrl_notilt['fn_tot_mean']:.1f} dpx={ctrl_notilt['dpx_mm']:.2f}", flush=True)

    trials = []
    for ga in GRIP_LADDER:
        tr = run_grip_trial(env, tilt_blob, float(ga), desire_sign)
        trials.append(tr)
        print(
            f"  a={ga:+.2f} τ={tr['tau_cmd']:+.1f} Fn={tr['fn_tot_mean']:.1f} "
            f"vx={tr['v_x_mean']:+.3f} vz={tr['v_z_mean']:+.3f} "
            f"dpx={tr['dpx_mm']:+.2f} dpz={tr['dpz_mm']:+.2f} mode={tr['mode']}",
            flush=True,
        )

    # Firm / mid / light named controls already in ladder
    table = [
        {
            "grip_a": t["grip_a"],
            "tau_cmd": t["tau_cmd"],
            "fn_L": t["fn_L_mean"],
            "fn_R": t["fn_R_mean"],
            "fn_tot": t["fn_tot_mean"],
            "ft_tot": t["ft_tot_mean"],
            "grip_ctrl_end": t["grip_ctrl_end"],
            "grip_af_end": t["grip_af_end"],
            "aperture_m": t["aperture_end"],
            "v_x_mean": t["v_x_mean"],
            "v_z_mean": t["v_z_mean"],
            "dpx_mm": t["dpx_mm"],
            "dpz_mm": t["dpz_mm"],
            "mode": t["mode"],
            "theta_end": t["theta_end"],
            "e_pos_end": t["e_pos_end"],
        }
        for t in trials
    ]
    (OUT / "force_table.json").write_text(json.dumps(jsafe(table), indent=2), encoding="utf-8")
    trial_summaries = []
    for t in trials:
        row = {k: v for k, v in t.items() if k != "series"}
        row["n_series"] = len(t["series"])
        trial_summaries.append(row)
    (OUT / "trials.json").write_text(json.dumps(jsafe(trial_summaries), indent=2), encoding="utf-8")

    plots = make_plots(trials, OUT / "plots")
    print(f"Plots: {plots}", flush=True)

    # Braking only if useful hand-x at observe
    brake_results = []
    for t in trials:
        if t["mode"] == "controlled_sliding_hand_x" or (
            (t["dpx_mm"] * desire_sign) >= 2.0 and abs(t["dpz_mm"]) < 8.0 and t["bilateral_end"]
        ):
            print(f"  Brake/restore for a={t['grip_a']}...", flush=True)
            br = brake_and_restore(env, tilt_blob, t["grip_a"], best_tilt, desire_sign)
            brake_results.append(br)
            print(f"    {br}", flush=True)

    # Videos: firm, best-motion, lightest bilateral
    vid_trials = []
    firm = next(t for t in trials if t["grip_a"] == 1.0)
    vid_trials.append(("firm", firm))
    # pick largest useful |dpx| in desire direction among bilateral
    ranked = sorted(trials, key=lambda t: (t["dpx_mm"] * desire_sign) - 0.3 * abs(t["dpz_mm"]), reverse=True)
    vid_trials.append(("best_motion", ranked[0]))
    light = next((t for t in reversed(trials) if t["bilateral_end"]), trials[-1])
    if light["grip_a"] != ranked[0]["grip_a"]:
        vid_trials.append(("light", light))

    videos = {}
    for name, tr in vid_trials:
        print(f"Rendering video {name} a={tr['grip_a']}...", flush=True)
        videos[name] = render_diagnostic_video(
            env, tilt_blob, tr, OUT / "videos" / f"A_grip_{name}_a{tr['grip_a']:+.2f}.mp4", f"grip a={tr['grip_a']:+.2f}"
        )

    # Conclusion
    fn_secure = next(t["fn_tot_mean"] for t in trials if t["grip_a"] == 1.0)
    fn_light = trials[-1]["fn_tot_mean"]
    unload_reduces = fn_light < 0.7 * fn_secure
    has_controlled = any(t["mode"] == "controlled_sliding_hand_x" for t in trials)
    has_z_redirect = any(t["mode"] in ("motion_dominated_by_hand_z", "uncontrolled_slipping") for t in trials)
    always_stick = all(t["mode"] in ("sticking", "sticking_or_ineffective") for t in trials if t["bilateral_end"])
    abrupt_loss = any(t["mode"] in ("partial_contact_loss", "complete_grasp_loss") for t in trials)

    if has_controlled and any(b.get("useful_surviving_hand_x") for b in brake_results):
        conclusion = "controllable_grip_force_window_exists"
    elif has_controlled:
        conclusion = "controlled_slide_seen_but_does_not_survive_brake_restore"
    elif abrupt_loss and not unload_reduces:
        conclusion = "grip_force_control_too_coarse_or_unstable"
    elif has_z_redirect and unload_reduces:
        conclusion = "contact_geometry_redirects_motion"
    elif always_stick and not unload_reduces:
        conclusion = "normal_force_remains_high_despite_unload_command"
    elif always_stick:
        conclusion = "available_legal_controls_cannot_produce_useful_lateral_correction"
    else:
        conclusion = "evidence_insufficient_or_mixed"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t_wall, 1),
        "pivot": {"theta0": met["theta0"], "theta1": met["theta1"], "delta_p_rel_mm": met["delta_p_rel_mm"]},
        "post_pivot_e_signed_mm": s_post["e_signed_mm"],
        "desire_dpx_sign": desire_sign,
        "tilt": {k: best_tilt[k] for k in best_tilt if k not in ("actions", "series_tail", "end_sample")},
        "tilt_end_sample": {k: s_tilt[k] for k in s_tilt if k != "contacts"},
        "no_tilt_control": {k: v for k, v in ctrl_notilt.items() if k != "series"},
        "force_table": table,
        "brake_results": brake_results,
        "plots": plots,
        "videos": videos,
        "fn_secure": fn_secure,
        "fn_lightest": fn_light,
        "unload_reduces_measured_fn": unload_reduces,
        "conclusion": conclusion,
        "gripper_audit_path": str(OUT / "gripper_implementation.json"),
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    # REPORT
    lines = [
        "# Grip-Force Sliding Audit — Plan A Post-Pivot",
        "",
        f"Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s",
        "",
        "## 1. Gripper-control implementation (audit)",
        "",
        f"- **Command:** legal `a[6] ∈ [-1,1]` → tendon motor torque `τ = grip_tau(a)`.",
        f"- **Map:** a=+1 → τ={TAU_SECURE} N·m (secure close); a=-1 → τ={TAU_OPEN} N·m (open).",
        f"- **Actuator:** `{audit['actuator_name']}` on tendon `{audit['tendon_name']}` (XML motor, gear=1).",
        f"- **Symmetry:** one tendon splits to both finger joints — **not** independent finger force control.",
        f"- **Path:** `step_legal` → `map_action` → `apply_cartesian_ctrl(gripper_tau=τ)` → `data.ctrl[7]` → `mj_step`.",
        f"- **Not** a contact-force setpoint, finger position target, or velocity mode.",
        f"- **Saturation:** recovery uses τ∈[{TAU_SECURE},{TAU_OPEN}]; XML allows ±50.",
        f"- **`fsm.fg_cmd = -τ`** is a bookkeeping mirror, not measured contact force.",
        f"- Sliding friction reference PAIR_MU={PAIR_MU}.",
        f"- Full dump: `gripper_implementation.json`.",
        "",
        "**Does unload reduce measured Fn?** See force table — compare a=+1 vs lighter a (must use `mj_contactForce`, not τ).",
        "",
        f"Unload reduces measured Fn: **{unload_reduces}** (secure ΣFn={fn_secure:.1f} N → lightest ΣFn={fn_light:.1f} N).",
        "",
        "## 2. Setup",
        "",
        f"- Verified pivot replay: θ {met['theta0']:.3f}→{met['theta1']:.3f}°, Δp_rel={met['delta_p_rel_mm']:.3f} mm.",
        f"- Post-pivot e_signed≈{s_post['e_signed_mm']}; desire Δp_x sign={desire_sign}.",
        f"- Fixed tilt ≈{TILT_TARGET_DEG}° sign={best_tilt['sign']:+.0f}: g_x {best_tilt['g_x0']:.2f}→{best_tilt['g_x1']:.2f}.",
        f"- Tilt-induced Δp_rel={best_tilt['delta_p_rel_mm']} mm; substantial_shift={best_tilt['substantial_shift']}.",
        "",
        "## 3. Force table (fixed tilt, independent trials)",
        "",
        "| a[6] | τ_cmd | ΣFn | Fn_L | Fn_R | ΣFt | v_x | v_z | Δp_x | Δp_z | mode |",
        "|------|-------|-----|------|------|-----|-----|-----|------|------|------|",
    ]
    for t in table:
        lines.append(
            f"| {t['grip_a']:+.2f} | {t['tau_cmd']:+.1f} | {t['fn_tot']:.1f} | {t['fn_L']:.1f} | {t['fn_R']:.1f} | "
            f"{t['ft_tot']:.1f} | {t['v_x_mean']:+.3f} | {t['v_z_mean']:+.3f} | "
            f"{t['dpx_mm']:+.2f} | {t['dpz_mm']:+.2f} | {t['mode']} |"
        )
    lines += [
        "",
        f"No-tilt unload control (a=-0.48): mode={ctrl_notilt['mode']} Fn={ctrl_notilt['fn_tot_mean']:.1f} "
        f"dpx={ctrl_notilt['dpx_mm']:+.2f} dpz={ctrl_notilt['dpz_mm']:+.2f}",
        "",
        "## 4. Force–motion relationship",
        "",
        f"Plots: {plots}",
        "",
        "## 5. Braking / restore",
        "",
    ]
    if not brake_results:
        lines.append("No trial met the useful hand-x criterion for brake/restore testing.")
    else:
        for b in brake_results:
            lines.append(f"- {b}")
    lines += [
        "",
        "## 6. Videos",
        "",
    ]
    for k, v in videos.items():
        lines.append(f"- `{k}`: {v.get('path')} ({v.get('duration_s')} s)")
    lines += [
        "",
        "Visualization legend: cyan=COM, yellow=P_TGT (hand-frame target in world), green=cylinder axis,",
        "red=desired −hand-x, orange=contact points, blue=normal force, magenta=tangential force.",
        "",
        "## 7. Conclusion",
        "",
        f"**{conclusion}**",
        "",
        "Supported distinctions:",
        f"- Controllable grip window for useful hand-x sliding: {has_controlled}",
        f"- Measured Fn drops with unload command: {unload_reduces}",
        f"- Motion redirected into hand-z / uncontrolled: {has_z_redirect}",
        f"- Abrupt contact loss on light grip: {abrupt_loss}",
        f"- Always sticking when bilateral retained: {always_stick}",
        "",
        "Stopped after contact-force identification. No SUCCESS_NEW optimization / no Plan B/C.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(jsafe({"conclusion": conclusion, "unload_reduces_fn": unload_reduces, "wall_s": summary["wall_s"]}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
