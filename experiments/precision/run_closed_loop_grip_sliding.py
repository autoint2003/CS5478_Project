#!/usr/bin/env python
"""Closed-loop grip force + precision slip control (Plan A post-pivot).

Offline privileged contact feedback → legal a[6] only. No SUCCESS_NEW optimization.
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

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import (  # noqa: E402
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
    tilt_actions,
    untilt_actions,
)
from data_generation.oracles.precision.corrections.contact_primitives import (  # noqa: E402
    _action_vw,
    _wrist,
    pivot_segment_metrics,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world, hand_R  # noqa: E402
from evaluation.success_criteria import P_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

PRIOR = ROOT / "results/diagnostics/position_preserving_pivot"
STATE_A = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_a__ball_train_v2_0070__stable.pkl"
OUT = ROOT / "results/diagnostics/closed_loop_grip_sliding"

FPS = 25
TILT_DEG = 60.0
DESIRE_DPX_SIGN = -1.0
MILESTONES_MM = (2.0, 5.0, 10.0, 14.0)
MAX_SLIP_S = 12.0
HOLD_STABLE_S = 0.15


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
    return [a_p] * max(1, int(round(p["dur"] / DT))) + [_wrist(0.0, 0.0, 1.0)] * 8


def classify_motion(series, p0, dpx_end) -> str:
    if not series:
        return "no_data"
    if any((not r["bilateral"]) for r in series[-3:]):
        if series[-1]["nL"] == 0 and series[-1]["nR"] == 0:
            return "complete_grasp_loss"
        return "partial_contact_loss"
    vx = [abs(r["v_x"]) for r in series]
    wn = [r["w_rel_norm"] for r in series]
    dpz = abs((np.asarray(series[-1]["p_c_h"])[2] - p0[2]) * 1e3)
    if max(vx) < 0.0005 and abs(dpx_end) < 0.5:
        return "sticking"
    if max(wn) > 1.5 and abs(dpx_end) < 2.0:
        return "rolling"
    if (dpx_end * DESIRE_DPX_SIGN) >= 1.5 and series[-1]["bilateral"] and dpz < abs(dpx_end) + 3.0:
        return "controlled_translation"
    if abs(dpx_end) >= 1.0 and dpz > abs(dpx_end):
        return "coupled_translation_rotation_or_z"
    if max(vx) > 0.05 and not series[-1]["bilateral"]:
        return "uncontrolled"
    return "partial_or_ineffective"


def run_force_calibration(env, blob, targets=(4.0, 5.0, 6.0, 8.0, 10.0), hold_s=1.0) -> list:
    rows = []
    for fn_tgt in targets:
        restore_capture(env, blob["capture"])
        s0 = sample_state(env.sim)
        ctl = ForcePController(kp=0.10, da_max=0.15)
        ctl.reset(a0=0.5, fn0=s0["fn_tot"])
        series = []
        n = max(1, int(round(hold_s / DT)))
        for i in range(n):
            s = sample_state(env.sim)
            a, meta = ctl.step(s["fn_tot"], fn_tgt)
            step_legal(env.sim, make_action([0, 0, 0], [0, 0, 0], a))
            s2 = sample_state(env.sim)
            series.append(
                {
                    "t": (i + 1) * DT,
                    "fn": s2["fn_tot"],
                    "fn_L": s2["fn_L"],
                    "fn_R": s2["fn_R"],
                    "a": a,
                    "tau": meta["tau"],
                    "target": fn_tgt,
                    "bilateral": s2["bilateral"],
                    "asymmetry": s2["asymmetry"],
                }
            )
        tail = series[len(series) // 2 :]
        fn_ss = float(np.mean([r["fn"] for r in tail]))
        rows.append(
            {
                "fn_target": fn_tgt,
                "fn_ss": fn_ss,
                "ss_err": fn_ss - fn_tgt,
                "overshoot": float(max(r["fn"] for r in series) - fn_tgt),
                "settled": abs(fn_ss - fn_tgt) < 1.5 and all(r["bilateral"] for r in tail),
                "a_ss": float(np.mean([r["a"] for r in tail])),
                "asymmetry_ss": float(np.mean([r["asymmetry"] for r in tail])),
                "series": series,
            }
        )
        print(f"  calib Fn*={fn_tgt:.1f} → Fn_ss={fn_ss:.2f} err={fn_ss-fn_tgt:+.2f} settled={rows[-1]['settled']}", flush=True)
    return rows


def reorient_compare(env, post_blob, sign=+1.0) -> dict:
    """Compare fixed-grip / force-regulated / slow+regulated reorientation drift."""
    variants = []

    def run_variant(name, rate_scale, use_force, fn_target=8.0):
        restore_capture(env, post_blob["capture"])
        s0 = sample_state(env.sim)
        p0 = np.asarray(s0["p_c_h"], float)
        acts_tilt = tilt_actions(TILT_DEG, sign, rate_scale)
        ctl = ForcePController(kp=0.10, da_max=0.10)
        ctl.reset(1.0, s0["fn_tot"])
        series = []
        lost = 0
        for i, a_tilt in enumerate(acts_tilt):
            if use_force:
                s = sample_state(env.sim)
                a_g, _ = ctl.step(s["fn_tot"], fn_target)
                # merge: use tilt wrist from a_tilt, grip from controller
                a = a_tilt.copy()
                a[6] = make_action([0, 0, 0], [0, 0, 0], a_g)[6]
            else:
                a = a_tilt
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, post_blob["ref"], lost)
            s = sample_state(env.sim)
            dp = (np.asarray(s["p_c_h"], float) - p0) * 1e3
            if i % 2 == 0 or i == len(acts_tilt) - 1:
                series.append(
                    {
                        "t": (i + 1) * DT,
                        "g_x": s["g_x"],
                        "fn": s["fn_tot"],
                        "dpx": float(dp[0]),
                        "dpy": float(dp[1]),
                        "dpz": float(dp[2]),
                        "theta": s["e_theta_deg"],
                        "bilateral": s["bilateral"],
                        "a6": float(a[6]),
                    }
                )
            if failed or latched(env.sim):
                return {"name": name, "ok": False, "reason": "DROP", "series": series}
        s1 = sample_state(env.sim)
        dp = (np.asarray(s1["p_c_h"], float) - p0) * 1e3
        return {
            "name": name,
            "ok": bool(s1["bilateral"] and s1["g_x"] < -4.0),
            "n_steps": len(acts_tilt),
            "rate_scale": rate_scale,
            "use_force": use_force,
            "g_x0": s0["g_x"],
            "g_x1": s1["g_x"],
            "theta0": s0["e_theta_deg"],
            "theta1": s1["e_theta_deg"],
            "delta_p_rel_mm": dp.tolist(),
            "dpx": float(dp[0]),
            "dpz": float(dp[2]),
            "dp_norm": float(np.linalg.norm(dp)),
            "fn_end": s1["fn_tot"],
            "series": series,
        }

    variants.append(run_variant("fixed_grip_nominal", 1.0, False))
    variants.append(run_variant("force_reg_Fn8", 1.0, True, 8.0))
    variants.append(run_variant("force_reg_Fn10_slow", 0.45, True, 10.0))
    variants.append(run_variant("force_reg_Fn6_slow", 0.45, True, 6.0))
    # pick best by ok and minimal |dp|
    ok_v = [v for v in variants if v.get("ok")]
    best = min(ok_v, key=lambda v: v["dp_norm"]) if ok_v else variants[0]
    return {"variants": variants, "best": best["name"], "best_row": {k: best[k] for k in best if k != "series"}}


def build_tilted_snapshot(env, post_blob, reori_best: dict) -> dict:
    """Replay best reorientation variant to create tilted IC."""
    name = reori_best["best"]
    row = next(v for v in reori_best["variants"] if v["name"] == name)
    restore_capture(env, post_blob["capture"])
    rate = row["rate_scale"]
    use_force = row["use_force"]
    acts = tilt_actions(TILT_DEG, +1.0, rate)
    ctl = ForcePController(kp=0.10, da_max=0.10)
    s0 = sample_state(env.sim)
    ctl.reset(1.0, s0["fn_tot"])
    fn_tgt = 10.0 if "Fn10" in name else (6.0 if "Fn6" in name else 8.0)
    for a_tilt in acts:
        if use_force:
            s = sample_state(env.sim)
            a_g, _ = ctl.step(s["fn_tot"], fn_tgt)
            a = a_tilt.copy()
            a[6] = make_action([0, 0, 0], [0, 0, 0], a_g)[6]
        else:
            a = a_tilt
        step_legal(env.sim, a)
    # short hold
    for _ in range(8):
        s = sample_state(env.sim)
        if use_force:
            a_g, _ = ctl.step(s["fn_tot"], fn_tgt)
            step_legal(env.sim, make_action([0, 0, 0], [0, 0, 0], a_g))
        else:
            step_legal(env.sim, make_action([0, 0, 0], [0, 0, 0], 1.0))
    cap = take_capture(env)
    s = sample_state(env.sim)
    return {
        "capture": cap,
        "ref": post_blob["ref"],
        "reori_name": name,
        "n_tilt": len(acts),
        "rate_scale": rate,
        "sample": {k: s[k] for k in s if k != "contacts"},
    }


def open_loop_fixed_grip(env, tilt_blob, grip_a: float, duration_s: float) -> dict:
    restore_capture(env, tilt_blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    series = []
    n = max(1, int(round(duration_s / DT)))
    lost = 0
    for i in range(n):
        step_legal(env.sim, make_action([0, 0, 0], [0, 0, 0], grip_a))
        failed, lost = recovery_failed(env.sim, tilt_blob["ref"], lost)
        s = sample_state(env.sim)
        dp = (np.asarray(s["p_c_h"], float) - p0) * 1e3
        series.append(
            {
                "t": (i + 1) * DT,
                "fn": s["fn_tot"],
                "fn_L": s["fn_L"],
                "fn_R": s["fn_R"],
                "fn_target": float("nan"),
                "a": grip_a,
                "vx": s["v_x"],
                "vz": s["v_z"],
                "dpx": float(dp[0]),
                "dpy": float(dp[1]),
                "dpz": float(dp[2]),
                "theta": s["e_theta_deg"],
                "e_pos": s["e_pos_mm"],
                "e_signed": s["e_signed_mm"],
                "bilateral": s["bilateral"],
                "phase": "slip",
            }
        )
        if failed or not s["bilateral"]:
            break
    dpx = series[-1]["dpx"]
    mode = classify_motion(
        [
            {
                "bilateral": r["bilateral"],
                "nL": 1 if r["bilateral"] else 0,
                "nR": 1 if r["bilateral"] else 0,
                "v_x": r["vx"],
                "w_rel_norm": 0.0,
                "p_c_h": (p0 + np.array([r["dpx"], r["dpy"], r["dpz"]]) * 1e-3).tolist(),
            }
            for r in series
        ],
        p0,
        dpx,
    )
    return {
        "method": "open_loop",
        "grip_a": grip_a,
        "dpx_mm": dpx,
        "dpz_mm": series[-1]["dpz"],
        "mode": mode,
        "series": series,
        "theta_end": series[-1]["theta"],
        "e_pos_end": series[-1]["e_pos"],
        "e_signed_end": series[-1]["e_signed"],
        "bilateral_end": series[-1]["bilateral"],
        "reached": (dpx * DESIRE_DPX_SIGN) >= 2.0,
        "stop_reason": "contact_loss" if not series[-1]["bilateral"] else "timeout",
        "brake": None,
        "restore": None,
    }


def run_slip_trial(
    env,
    tilt_blob,
    *,
    use_slip: bool,
    vx_target: float,
    fn_init: float,
    milestone_mm: float | None,
    max_s: float = MAX_SLIP_S,
    brake: bool = True,
    restore: bool = False,
    n_tilt: int = 30,
    rate_scale: float = 1.0,
) -> dict:
    restore_capture(env, tilt_blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    ctl = DualLoopController(
        force=ForcePController(kp=0.10, da_max=0.12, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=vx_target, fn_min=3.8, fn_max=11.0, kn=900.0),
        use_slip=use_slip,
    )
    ctl.reset(a0=0.2, fn0=s0["fn_tot"], fn_target=fn_init)
    if not use_slip:
        ctl.slip.fn_target = fn_init

    series = []
    lost = 0
    n_max = max(1, int(round(max_s / DT)))
    reached = False
    stop_reason = "timeout"
    for i in range(n_max):
        s = sample_state(env.sim)
        if use_slip:
            a, meta = ctl.step(s)
        else:
            a, meta = ctl.step(s, fn_target_override=fn_init)
        step_legal(env.sim, a)
        failed, lost = recovery_failed(env.sim, tilt_blob["ref"], lost)
        s2 = sample_state(env.sim)
        dp = (np.asarray(s2["p_c_h"], float) - p0) * 1e3
        row = {
            "t": (i + 1) * DT,
            "fn": s2["fn_tot"],
            "fn_L": s2["fn_L"],
            "fn_R": s2["fn_R"],
            "fn_target": meta["fn_target"],
            "a": meta["a"],
            "vx": s2["v_x"],
            "vz": s2["v_z"],
            "dpx": float(dp[0]),
            "dpy": float(dp[1]),
            "dpz": float(dp[2]),
            "theta": s2["e_theta_deg"],
            "e_pos": s2["e_pos_mm"],
            "e_signed": s2["e_signed_mm"],
            "bilateral": s2["bilateral"],
            "phase": "slip",
        }
        series.append(row)
        if failed or latched(env.sim) or not s2["bilateral"]:
            stop_reason = "contact_loss"
            break
        if milestone_mm is not None and (float(dp[0]) * DESIRE_DPX_SIGN) >= milestone_mm:
            reached = True
            stop_reason = "milestone"
            break

    dpx = series[-1]["dpx"]
    mode = classify_motion(
        [
            {
                "bilateral": r["bilateral"],
                "nL": 1 if r["bilateral"] else 0,
                "nR": 1 if r["bilateral"] else 0,
                "v_x": r["vx"],
                "w_rel_norm": 0.0,
                "p_c_h": (p0 + np.array([r["dpx"], r["dpy"], r["dpz"]]) * 1e-3).tolist(),
            }
            for r in series
        ],
        p0,
        dpx,
    )

    brake_meta = None
    if brake and series[-1]["bilateral"] and reached:
        # firm + hold ≥150 ms
        ctl_b = ForcePController(kp=0.15, da_max=0.2)
        ctl_b.reset(series[-1]["a"], series[-1]["fn"])
        n_brake = max(1, int(round(0.40 / DT)))
        n_hold = max(1, int(round(HOLD_STABLE_S / DT)))
        for j in range(n_brake + n_hold):
            s = sample_state(env.sim)
            a_g, _ = ctl_b.step(s["fn_tot"], 14.0)  # firm
            step_legal(env.sim, make_action([0, 0, 0], [0, 0, 0], a_g))
            s2 = sample_state(env.sim)
            dp = (np.asarray(s2["p_c_h"], float) - p0) * 1e3
            series.append(
                {
                    "t": series[-1]["t"] + DT,
                    "fn": s2["fn_tot"],
                    "fn_L": s2["fn_L"],
                    "fn_R": s2["fn_R"],
                    "fn_target": 14.0,
                    "a": a_g,
                    "vx": s2["v_x"],
                    "vz": s2["v_z"],
                    "dpx": float(dp[0]),
                    "dpy": float(dp[1]),
                    "dpz": float(dp[2]),
                    "theta": s2["e_theta_deg"],
                    "e_pos": s2["e_pos_mm"],
                    "e_signed": s2["e_signed_mm"],
                    "bilateral": s2["bilateral"],
                    "phase": "brake" if j < n_brake else "hold",
                }
            )
            if not s2["bilateral"]:
                break
        hold_rows = [r for r in series if r["phase"] == "hold"]
        brake_meta = {
            "dpx_after_brake": series[-1]["dpx"],
            "dpz_after_brake": series[-1]["dpz"],
            "vx_end": series[-1]["vx"],
            "hold_bilateral": all(r["bilateral"] for r in hold_rows) if hold_rows else False,
            "theta": series[-1]["theta"],
            "e_signed": series[-1]["e_signed"],
        }

    restore_meta = None
    if restore and brake_meta and brake_meta.get("hold_bilateral"):
        ret = untilt_actions(n_tilt, +1.0, rate_scale)
        ctl_r = ForcePController(kp=0.12, da_max=0.12)
        ctl_r.reset(series[-1]["a"], series[-1]["fn"])
        for a_ret in ret:
            s = sample_state(env.sim)
            a_g, _ = ctl_r.step(s["fn_tot"], 12.0)
            a = a_ret.copy()
            a[6] = make_action([0, 0, 0], [0, 0, 0], a_g)[6]
            step_legal(env.sim, a)
            s2 = sample_state(env.sim)
            dp = (np.asarray(s2["p_c_h"], float) - p0) * 1e3
            series.append(
                {
                    "t": series[-1]["t"] + DT,
                    "fn": s2["fn_tot"],
                    "fn_L": s2["fn_L"],
                    "fn_R": s2["fn_R"],
                    "fn_target": 12.0,
                    "a": a_g,
                    "vx": s2["v_x"],
                    "vz": s2["v_z"],
                    "dpx": float(dp[0]),
                    "dpy": float(dp[1]),
                    "dpz": float(dp[2]),
                    "theta": s2["e_theta_deg"],
                    "e_pos": s2["e_pos_mm"],
                    "e_signed": s2["e_signed_mm"],
                    "bilateral": s2["bilateral"],
                    "phase": "restore",
                }
            )
            if not s2["bilateral"]:
                break
        restore_meta = {
            "dpx_after_restore": series[-1]["dpx"],
            "dpz_after_restore": series[-1]["dpz"],
            "theta": series[-1]["theta"],
            "e_signed": series[-1]["e_signed"],
            "e_pos": series[-1]["e_pos"],
            "bilateral": series[-1]["bilateral"],
            "survives": series[-1]["bilateral"] and (series[-1]["dpx"] * DESIRE_DPX_SIGN) >= 1.5,
        }

    return {
        "use_slip": use_slip,
        "vx_target": vx_target,
        "fn_init": fn_init,
        "milestone_mm": milestone_mm,
        "reached": reached,
        "stop_reason": stop_reason,
        "mode": mode,
        "dpx_mm": dpx,
        "dpz_mm": series[-1]["dpz"],
        "theta_end": series[-1]["theta"],
        "e_pos_end": series[-1]["e_pos"],
        "e_signed_end": series[-1]["e_signed"],
        "bilateral_end": series[-1]["bilateral"],
        "brake": brake_meta,
        "restore": restore_meta,
        "series": series,
    }


def make_plots(calib, comparisons, milestones, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return {"error": str(e)}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Force tracking
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in calib:
        ax.plot([r["t"] for r in c["series"]], [r["fn"] for r in c["series"]], label=f"tgt={c['fn_target']:.0f}")
        ax.axhline(c["fn_target"], ls="--", alpha=0.3)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("ΣFn (N)")
    ax.set_title("Force controller: target vs measured")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    p = out_dir / "force_tracking.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["force_tracking"] = str(p)

    # Open vs closed loop dpx
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, tr in comparisons.items():
        ax.plot([r["t"] for r in tr["series"]], [r["dpx"] for r in tr["series"]], label=name)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Δp_x (mm)")
    ax.set_title("Open-loop vs closed-loop hand-x displacement")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    p = out_dir / "ol_vs_cl_dpx.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["ol_vs_cl_dpx"] = str(p)

    # Velocity tracking for dual-loop
    if "force_plus_slip" in comparisons:
        tr = comparisons["force_plus_slip"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["t"] for r in tr["series"]], [r["vx"] * 1e3 for r in tr["series"]], label="v_x")
        ax.axhline(tr.get("vx_target", -0.001) * 1e3, ls="--", color="k", label="target")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("v_x (mm/s)")
        ax.set_title("Slip-velocity tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = out_dir / "vx_tracking.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["vx_tracking"] = str(p)

    # Milestone traces
    if milestones:
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in milestones:
            ax.plot([r["t"] for r in m["series"]], [r["dpx"] for r in m["series"]], label=f"{m['milestone_mm']}mm/{m['stop_reason']}")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Δp_x (mm)")
        ax.set_title("Milestone slip attempts")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        p = out_dir / "milestones_dpx.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["milestones_dpx"] = str(p)

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


def render_video(env, tilt_blob, trial: dict, dest: Path, title: str):
    restore_capture(env, tilt_blob["capture"])
    # Re-exec series actions approximately by replaying controller from series a values
    p0 = np.asarray(sample_state(env.sim)["p_c_h"], float)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 440, 600)
    rh = mujoco.Renderer(env.sim.model, 440, 600)
    cam_w = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.5, "azimuth": 140, "elevation": -18})
    cam_h = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.35, "azimuth": 140, "elevation": -22})
    frames = []
    series = trial["series"]

    def cams():
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.55 * po + 0.45 * ph
        cam_w.distance = 0.5
        cam_w.azimuth = 138
        cam_w.elevation = -18
        look = ph + Rh @ np.array([0.0, 0.0, 0.08])
        pos = ph + Rh @ np.array([0.22, -0.28, 0.14])
        vec = pos - look
        dist = float(np.linalg.norm(vec))
        cam_h.lookat[:] = look
        cam_h.distance = max(dist, 0.22)
        cam_h.azimuth = float(np.degrees(np.arctan2(vec[1], vec[0])))
        cam_h.elevation = float(np.degrees(np.arcsin(np.clip(vec[2] / max(dist, 1e-9), -1, 1))))

    def draw(scn):
        from data_generation.oracles.precision.mechanics.dual_loop import contact_forces

        scn.ngeom = 0
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.1, 0.95, 1.0, 1.0], 0.006)
        add_sphere(scn, ph + Rh @ np.asarray(P_TGT, float), [1.0, 0.9, 0.1, 1.0], 0.005)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.2, 1.0], 0.0025)
        add_capsule(scn, po, po - 0.06 * Rh[:, 0], [1.0, 0.2, 0.2, 1.0], 0.003)
        cf = contact_forces(env.sim)
        for c in cf["contacts"][:6]:
            pos = np.asarray(c["pos"], float)
            add_sphere(scn, pos, [1.0, 0.5, 0.0, 1.0], 0.003)
            n = np.asarray(c["normal_world"], float)
            add_capsule(scn, pos, pos + 0.002 * c["fn"] * n, [0.2, 0.4, 1.0, 1.0], 0.002)

    def frame(row):
        cams()
        lines = [
            f"{row.get('phase','RUN')} {title} t={row['t']:.2f}",
            f"a={row['a']:+.2f} Fn*={row.get('fn_target',0):.1f} Fn={row['fn']:.1f} L/R={row.get('fn_L',0):.1f}/{row.get('fn_R',0):.1f}",
            f"vx={row['vx']*1e3:+.1f}mm/s Δp_x={row['dpx']:+.1f} Δp_z={row['dpz']:+.1f} θ={row['theta']:.1f} bil={int(row['bilateral'])}",
            "cyan=COM yellow=P_TGT green=axis red=desire-x orange=contact blue=Fn",
        ]
        rw.update_scene(env.sim.data, cam_w)
        draw(rw.scene)
        iw = overlay(np.asarray(rw.render(), np.uint8), ["WORLD"] + lines)
        rh.update_scene(env.sim.data, cam_h)
        draw(rh.scene)
        ih = overlay(np.asarray(rh.render(), np.uint8), ["HAND-REL"] + lines)
        return np.concatenate([iw, ih], axis=1)

    img = frame({"phase": "BEFORE", "t": 0, "a": 0, "fn": 0, "fn_target": 0, "fn_L": 0, "fn_R": 0, "vx": 0, "dpx": 0, "dpz": 0, "theta": 0, "bilateral": True})
    for _ in range(int(3 * FPS)):
        frames.append(img)

    # Replay by stepping with recorded a; slow for slip/brake
    for row in series:
        step_legal(env.sim, make_action([0, 0, 0], [0, 0, 0], row["a"]))
        # refresh row metrics lightly
        s = sample_state(env.sim)
        dp = (np.asarray(s["p_c_h"], float) - p0) * 1e3
        row_v = {
            **row,
            "fn": s["fn_tot"],
            "fn_L": s["fn_L"],
            "fn_R": s["fn_R"],
            "vx": s["v_x"],
            "dpx": float(dp[0]),
            "dpz": float(dp[2]),
            "theta": s["e_theta_deg"],
            "bilateral": s["bilateral"],
        }
        img = frame(row_v)
        slow = row.get("phase") in ("slip", "brake") or abs(row_v["vx"]) > 0.005
        dup = max(1, int(round((DT / (0.25 if slow else 0.5)) * FPS)))
        for _ in range(dup):
            frames.append(img)

    img = frame({**series[-1], "phase": "AFTER"})
    for _ in range(int(3 * FPS)):
        frames.append(img)
    rw.close()
    rh.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "duration_s": len(frames) / FPS}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(parents=True, exist_ok=True)

    search = json.loads((PRIOR / "search.json").read_text(encoding="utf-8"))
    tag = search["A_raw"]["best"]["tag"]
    state = pickle.loads(STATE_A.read_bytes())
    env = make_env()
    blob0 = {"capture": state["capture"], "ref": state["ref"]}

    # IC
    piv = pivot_actions(tag)
    met = pivot_segment_metrics(env, blob0, piv, record_series=False)
    print(f"Pivot θ {met['theta0']:.3f}->{met['theta1']:.3f} Δp={met['delta_p_rel_mm']:.3f}", flush=True)
    restore_capture(env, blob0["capture"])
    for a in piv:
        step_legal(env.sim, a)
    post = {"capture": take_capture(env), "ref": state["ref"]}
    s_post = sample_state(env.sim)
    print(f"Post-pivot e={s_post['e_signed_mm']} θ={s_post['e_theta_deg']:.3f}", flush=True)
    (OUT / "snapshots" / "A_post_pivot.pkl").write_bytes(
        pickle.dumps({"capture": post["capture"], "ref": post["ref"], "geom": {k: s_post[k] for k in s_post if k != "contacts"}})
    )

    # Reorientation drift
    print("=== Reorientation drift comparison ===", flush=True)
    reori = reorient_compare(env, post, sign=+1.0)
    for v in reori["variants"]:
        print(
            f"  {v['name']}: ok={v.get('ok')} dp={v.get('delta_p_rel_mm')} g_x={v.get('g_x1')}",
            flush=True,
        )
    print(f"  best={reori['best']}", flush=True)
    (OUT / "reorientation_compare.json").write_text(
        json.dumps(jsafe({"best": reori["best"], "variants": [{k: v[k] for k in v if k != "series"} for v in reori["variants"]]}), indent=2),
        encoding="utf-8",
    )

    tilt = build_tilted_snapshot(env, post, reori)
    (OUT / "snapshots" / "A_tilted.pkl").write_bytes(pickle.dumps({"capture": tilt["capture"], "ref": tilt["ref"], "meta": {k: tilt[k] for k in tilt if k != "capture"}}))
    print(f"Tilted via {tilt['reori_name']}: g_x={tilt['sample']['g_x']:.2f} e={tilt['sample']['e_signed_mm']}", flush=True)
    tilt_blob = {"capture": tilt["capture"], "ref": tilt["ref"]}

    # Force calibration at tilted pose
    print("=== Force calibration ===", flush=True)
    calib = run_force_calibration(env, tilt_blob)
    n_ok = sum(1 for c in calib if c["settled"])
    force_ok = n_ok >= 3
    print(f"  settled targets: {n_ok}/{len(calib)} force_ok={force_ok}", flush=True)
    (OUT / "force_calibration.json").write_text(
        json.dumps(jsafe([{k: v for k, v in c.items() if k != "series"} for c in calib]), indent=2),
        encoding="utf-8",
    )

    if not force_ok:
        # Still try a minimal slip attempt but report limitation
        print("Force regulation weak — continuing limited slip tests", flush=True)

    # Comparisons at fixed tilt
    print("=== Open vs closed loop ===", flush=True)
    comparisons = {}
    comparisons["open_loop_a-0.30"] = open_loop_fixed_grip(env, tilt_blob, -0.30, 2.0)
    print(f"  OL: dpx={comparisons['open_loop_a-0.30']['dpx_mm']:.2f} mode={comparisons['open_loop_a-0.30']['mode']}", flush=True)

    comparisons["force_only"] = run_slip_trial(
        env, tilt_blob, use_slip=False, vx_target=-0.001, fn_init=5.5, milestone_mm=2.0, max_s=4.0, brake=True, restore=False, n_tilt=tilt["n_tilt"], rate_scale=tilt["rate_scale"]
    )
    print(f"  force_only: dpx={comparisons['force_only']['dpx_mm']:.2f} reached={comparisons['force_only']['reached']} mode={comparisons['force_only']['mode']}", flush=True)

    comparisons["force_plus_slip"] = run_slip_trial(
        env, tilt_blob, use_slip=True, vx_target=-0.001, fn_init=5.5, milestone_mm=2.0, max_s=6.0, brake=True, restore=False, n_tilt=tilt["n_tilt"], rate_scale=tilt["rate_scale"]
    )
    print(f"  dual: dpx={comparisons['force_plus_slip']['dpx_mm']:.2f} reached={comparisons['force_plus_slip']['reached']} mode={comparisons['force_plus_slip']['mode']}", flush=True)

    # Try -2 mm/s if -1 worked at all
    if comparisons["force_plus_slip"]["dpx_mm"] * DESIRE_DPX_SIGN >= 1.0:
        comparisons["force_plus_slip_vx2"] = run_slip_trial(
            env, tilt_blob, use_slip=True, vx_target=-0.002, fn_init=5.0, milestone_mm=2.0, max_s=6.0, brake=True, restore=False, n_tilt=tilt["n_tilt"], rate_scale=tilt["rate_scale"]
        )
        print(f"  dual vx2: dpx={comparisons['force_plus_slip_vx2']['dpx_mm']:.2f} mode={comparisons['force_plus_slip_vx2']['mode']}", flush=True)

    # Progressive milestones only if 2mm reachable
    milestones = []
    can_progress = comparisons["force_plus_slip"].get("reached") or (comparisons["force_plus_slip"]["dpx_mm"] * DESIRE_DPX_SIGN >= 1.8)
    if can_progress:
        print("=== Milestones ===", flush=True)
        for m_mm in MILESTONES_MM:
            tr = run_slip_trial(
                env,
                tilt_blob,
                use_slip=True,
                vx_target=-0.001,
                fn_init=5.2,
                milestone_mm=m_mm,
                max_s=MAX_SLIP_S,
                brake=True,
                restore=(m_mm <= 5.0),  # restore only for short successes
                n_tilt=tilt["n_tilt"],
                rate_scale=tilt["rate_scale"],
            )
            milestones.append(tr)
            print(
                f"  {m_mm}mm: reached={tr['reached']} dpx={tr['dpx_mm']:.2f} brake={tr.get('brake')} restore={tr.get('restore')} mode={tr['mode']}",
                flush=True,
            )
            if not tr["reached"] or not (tr.get("brake") or {}).get("hold_bilateral", False):
                print("  stop advancing milestones", flush=True)
                break
            # repeatability of 2mm
            if m_mm == 2.0:
                reps = []
                for k in range(3):
                    r = run_slip_trial(
                        env, tilt_blob, use_slip=True, vx_target=-0.001, fn_init=5.2, milestone_mm=2.0, max_s=6.0, brake=True, restore=False, n_tilt=tilt["n_tilt"], rate_scale=tilt["rate_scale"]
                    )
                    reps.append({"k": k, "reached": r["reached"], "dpx": r["dpx_mm"], "mode": r["mode"]})
                (OUT / "repeat_2mm.json").write_text(json.dumps(jsafe(reps), indent=2), encoding="utf-8")
                print(f"  2mm repeats: {reps}", flush=True)
                if sum(1 for r in reps if r["reached"]) < 2:
                    print("  2mm not repeatable — stop", flush=True)
                    break
    else:
        print("=== Milestones skipped (2mm not achieved) ===", flush=True)

    plots = make_plots(calib, comparisons, milestones, OUT / "plots")
    print(f"Plots: {plots}", flush=True)

    # Videos
    videos = {}
    print("Rendering videos...", flush=True)
    videos["force_plus_slip"] = render_video(
        env, tilt_blob, comparisons["force_plus_slip"], OUT / "videos" / "A_closed_loop_slip.mp4", "CL force+slip"
    )
    videos["open_loop"] = render_video(
        env, tilt_blob, comparisons["open_loop_a-0.30"], OUT / "videos" / "A_open_loop_slip.mp4", "OL a=-0.30"
    )
    if milestones:
        videos["best_milestone"] = render_video(
            env, tilt_blob, milestones[0], OUT / "videos" / "A_milestone_best.mp4", f"milestone {milestones[0]['milestone_mm']}mm"
        )

    # Recommendation — desire −x; best = most desire-aligned among bilateral endings
    cands = [comparisons["open_loop_a-0.30"], comparisons["force_only"], comparisons["force_plus_slip"]]
    bil = [c for c in cands if c.get("bilateral_end", True)]
    best = max(bil or cands, key=lambda c: c["dpx_mm"] * DESIRE_DPX_SIGN)
    method_name = (
        "force_plus_slip"
        if best is comparisons["force_plus_slip"]
        else ("force_only" if best is comparisons["force_only"] else "open_loop")
    )

    cl_improves = (comparisons["force_plus_slip"]["dpx_mm"] * DESIRE_DPX_SIGN) > (
        comparisons["open_loop_a-0.30"]["dpx_mm"] * DESIRE_DPX_SIGN + 0.5
    )
    milestone_ok = any(m.get("reached") and (m.get("brake") or {}).get("hold_bilateral") for m in milestones)

    if milestone_ok and cl_improves:
        recommendation = "adopt_closed_loop_force_slip_in_precision_oracle_with_limits"
    elif cl_improves and not milestone_ok:
        recommendation = "closed_loop_helps_slightly_but_cannot_reach_2mm_stable_stop"
    elif force_ok and not cl_improves:
        recommendation = "force_loop_tracks_but_does_not_enable_useful_lateral_slip"
    else:
        recommendation = "do_not_adopt_closed_loop_as_primary_precision_mechanism_yet"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "pivot": {"theta0": met["theta0"], "theta1": met["theta1"], "delta_p_rel_mm": met["delta_p_rel_mm"]},
        "post_pivot_e_signed_mm": s_post["e_signed_mm"],
        "reorientation": reori["best_row"],
        "tilt": {k: tilt[k] for k in ("reori_name", "n_tilt", "rate_scale", "sample")},
        "force_calibration_settled": n_ok,
        "force_regulation_ok": force_ok,
        "comparisons": {k: {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in comparisons.items()},
        "milestones": [{kk: vv for kk, vv in m.items() if kk != "series"} for m in milestones],
        "best_surviving": {
            "method": method_name,
            "dpx_mm": best["dpx_mm"],
            "dpz_mm": best.get("dpz_mm"),
            "theta_end": best.get("theta_end"),
            "e_pos_end": best.get("e_pos_end"),
            "mode": best.get("mode"),
        },
        "closed_loop_improves_vs_open_loop": cl_improves,
        "recommendation": recommendation,
        "plots": plots,
        "videos": videos,
    }
    (OUT / "comparison_series.json").write_text(
        json.dumps(jsafe({k: v["series"] for k, v in comparisons.items()}), indent=2), encoding="utf-8"
    )
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    report = f"""# Closed-Loop Grip Force and Precision Slip Control — Plan A

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

Offline privileged `mj_contactForce` → legal `a[6]` only. No SUCCESS_NEW optimization. No physics/actuator changes.

## 1. Initial condition

- Pivot verified: θ {met['theta0']:.3f}→{met['theta1']:.3f}°, Δp_rel={met['delta_p_rel_mm']:.3f} mm
- Post-pivot e_signed≈{s_post['e_signed_mm']}, θ={s_post['e_theta_deg']:.3f}°
- Snapshot: `snapshots/A_post_pivot.pkl`

## 2. Reorientation drift (~60°)

| Variant | ok | Δp_rel (mm) | g_x | θ |
|---------|----|-------------|-----|---|
"""
    for v in reori["variants"]:
        report += f"| {v['name']} | {v.get('ok')} | {v.get('delta_p_rel_mm')} | {v.get('g_x1')} | {v.get('theta1')} |\n"
    report += f"\nSelected tilt IC: **{reori['best']}** → `snapshots/A_tilted.pkl`\n\n"
    report += "## 3. Force inner-loop calibration\n\n"
    report += "| Fn* | Fn_ss | err | settled | a_ss | asymmetry |\n|-----|-------|-----|---------|------|----------|\n"
    for c in calib:
        report += f"| {c['fn_target']:.1f} | {c['fn_ss']:.2f} | {c['ss_err']:+.2f} | {c['settled']} | {c['a_ss']:+.2f} | {c['asymmetry_ss']:.2f} |\n"
    report += f"\nForce regulation usable: **{force_ok}**\n\n"
    report += "## 4. Open-loop vs closed-loop\n\n"
    for k, v in comparisons.items():
        report += (
            f"- **{k}**: Δp_x={v['dpx_mm']:+.2f} mm, Δp_z={v.get('dpz_mm', float('nan')):+.2f} mm, "
            f"mode={v.get('mode')}, reached_2mm={v.get('reached')}, "
            f"bilateral_end={v.get('bilateral_end')}, stop={v.get('stop_reason')}\n"
        )
    report += f"\nClosed-loop improves vs open-loop (≥0.5 mm): **{cl_improves}**\n\n"
    report += "## 5. Progressive milestones\n\n"
    if not milestones:
        report += "Skipped — controller did not achieve a stable ~2 mm correct-sign displacement with hold.\n\n"
    else:
        for m in milestones:
            report += (
                f"- target {m['milestone_mm']} mm: reached={m['reached']}, dpx={m['dpx_mm']:+.2f}, "
                f"mode={m['mode']}, brake={m.get('brake')}, restore={m.get('restore')}\n"
            )
        report += "\n"
    report += "## 6. Best surviving correction\n\n"
    report += f"{summary['best_surviving']}\n\n"
    report += "## 7. Plots & videos\n\n"
    report += f"Plots: {plots}\n\n"
    for k, v in videos.items():
        report += f"- `{k}`: {v.get('path')} ({v.get('duration_s')} s)\n"
    report += f"""
## 8. Recommendation

**{recommendation}**

Interpretation:
- Inner-loop force tracking: {'usable' if force_ok else 'not reliable'}
- Outer-loop slip for ≥2 mm stable stop: {'yes' if milestone_ok else 'no'}
- Improvement over ~1 mm open-loop unload: {'yes' if cl_improves else 'no / marginal'}

Do **not** treat this as physical impossibility of lateral slip; it is a statement about this controller + coupled tendon + contact geometry under the tested tilt.

Stopped after this focused experiment. No bulk Oracle / SFT / GRPO.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": recommendation, "best": summary["best_surviving"], "wall_s": summary["wall_s"]}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
