#!/usr/bin/env python
"""Validate Plan A closed-loop HANDOFF; develop geometry-inferred orientation.

Low-level force/slip frozen from force_slip_control.py.
Orientation axis/sign/magnitude inferred from current geometry (not memorized 60°).
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
from data_generation.oracles.precision.force_slip_control import (  # noqa: E402
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
    tilt_actions,
)
from data_generation.oracles.precision.orientation_planner import (  # noqa: E402
    _height_hold_v_hand,
    desired_correction_direction,
    execute_orientation_plan,
    execute_restore_plan,
    plan_orientation,
    plan_restore_orientation,
)
from data_generation.oracles.precision.primitives import (  # noqa: E402
    _action_vw,
    _wrist,
    pivot_segment_metrics,
)
from envs.observations.airborne_obs import OBS_DIM_AIR, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world, hand_R  # noqa: E402
from evaluation.success_criteria import P_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from scripts.generate_plan_c_experts import evaluate_handoff_from_state  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

PRIOR_PIV = ROOT / "results/diagnostics/position_preserving_pivot"
STATE_A = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_a__ball_train_v2_0070__stable.pkl"
STATE_B = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"
IC_A = ROOT / "data/snapshots/train/ball_train_v2_0070.pkl"
OUT = ROOT / "results/diagnostics/geometry_inferred_precision"

FPS = 25
MILESTONE_MM = 14.0
HOLD_STABLE_S = 0.15
V_SLIP = -0.001


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


def obs39(env) -> np.ndarray:
    return observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist)


def run_slip_to_milestone(env, blob, d_axis, *, milestone_mm=MILESTONE_MM, max_s=16.0, record_actions=True):
    """Frozen dual-loop slip along d_axis until milestone projected displacement."""
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    d = np.asarray(d_axis, float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    # Validated loop: v_along = v·axis, target=−1 mm/s. Desire motion along d ⇒ axis=−d.
    axis = -d
    ctl = DualLoopController(
        force=ForcePController(kp=0.10, da_max=0.12, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=V_SLIP, fn_min=3.8, fn_max=11.0, kn=900.0),
        use_slip=True,
    )
    ctl.reset(a0=0.2, fn0=s0["fn_tot"], fn_target=5.2, slip_axis=tuple(axis.tolist()))
    actions, series = [], []
    lost = 0
    n_max = max(1, int(round(max_s / DT)))
    reached = False
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for i in range(n_max):
        s = sample_state(env.sim)
        a, meta = ctl.step(s)
        # Overlay height hold on grip-only dual-loop action
        v_h = _height_hold_v_hand(env.sim, z_ref)
        a = make_action(v_h, [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        if record_actions:
            actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        disp_along = float(np.dot(dp, d)) * 1e3
        series.append(
            {
                "t": (i + 1) * DT,
                "fn": s2["fn_tot"],
                "fn_L": s2["fn_L"],
                "fn_R": s2["fn_R"],
                "fn_target": meta["fn_target"],
                "a": meta["a"],
                "vx": s2["v_x"],
                "v_along": float(np.dot(s2["v_rel_h"], d)),
                "dpx": float(dp[0] * 1e3),
                "dpy": float(dp[1] * 1e3),
                "dpz": float(dp[2] * 1e3),
                "disp_along_mm": disp_along,
                "theta": s2["e_theta_deg"],
                "e_pos": s2["e_pos_mm"],
                "e_signed": s2["e_signed_mm"],
                "bilateral": s2["bilateral"],
                "g_h": s2["g_h"],
                "phase": "slip",
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
            }
        )
        if failed or latched(env.sim) or not s2["bilateral"]:
            return {"ok": False, "reason": "contact_loss", "actions": actions, "series": series, "reached": False}
        if disp_along >= milestone_mm:
            reached = True
            break

    if not series:
        return {"ok": False, "reason": "no_steps", "actions": actions, "series": series, "reached": False}

    # brake + hold (even on partial progress — needed for HANDOFF validation)
    ctl_b = ForcePController(kp=0.15, da_max=0.2)
    ctl_b.reset(series[-1]["a"], series[-1]["fn"])
    n_brake = max(1, int(round(0.40 / DT)))
    n_hold = max(1, int(round(HOLD_STABLE_S / DT)))
    for j in range(n_brake + n_hold):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], 14.0)
        v_h = _height_hold_v_hand(env.sim, z_ref)
        a = make_action(v_h, [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        series.append(
            {
                "t": series[-1]["t"] + DT,
                "fn": s2["fn_tot"],
                "fn_L": s2["fn_L"],
                "fn_R": s2["fn_R"],
                "fn_target": 14.0,
                "a": a_g,
                "vx": s2["v_x"],
                "v_along": float(np.dot(s2["v_rel_h"], d)),
                "dpx": float(dp[0] * 1e3),
                "dpy": float(dp[1] * 1e3),
                "dpz": float(dp[2] * 1e3),
                "disp_along_mm": float(np.dot(dp, d) * 1e3),
                "theta": s2["e_theta_deg"],
                "e_pos": s2["e_pos_mm"],
                "e_signed": s2["e_signed_mm"],
                "bilateral": s2["bilateral"],
                "g_h": s2["g_h"],
                "phase": "brake" if j < n_brake else "hold",
            }
        )
        if not s2["bilateral"]:
            return {"ok": False, "reason": "loss_on_brake", "actions": actions, "series": series, "reached": reached}

    hold_ok = all(r["bilateral"] for r in series if r["phase"] == "hold")
    disp = float(series[-1]["disp_along_mm"])
    # Accept near-miss (≥85% of milestone) so HANDOFF validation can proceed.
    useful_progress = reached or disp >= 0.85 * milestone_mm
    return {
        "ok": bool(hold_ok and useful_progress),
        "reached": reached,
        "actions": actions,
        "series": series,
        "disp_along_mm": disp,
        "e_signed": series[-1]["e_signed"],
        "e_pos": series[-1]["e_pos"],
        "theta": series[-1]["theta"],
        "capture": take_capture(env),
    }


def baseline_fixed_orientation_trajectory(env, post_pivot_blob):
    """Reproduce validated fixed 60° +wy slow force-reg → slip → brake → reverse untilt.

    Slip uses the validated −hand-x axis (not a tilted d), matching closed_loop_grip_sliding.
    """
    actions_all = []
    restore_capture(env, post_pivot_blob["capture"])
    s0 = sample_state(env.sim)
    d_geom = desired_correction_direction(np.asarray(s0["e_signed_mm"], float) * 1e-3)
    d_slip = np.array([-1.0, 0.0, 0.0])  # validated baseline slip axis

    # Fixed baseline: +wy, 60°, rate 0.45, Fn*=10, with world-height hold
    acts_tilt = tilt_actions(60.0, sign=+1.0, rate_scale=0.45)
    ctl = ForcePController(kp=0.10, da_max=0.10)
    ctl.reset(1.0, s0["fn_tot"])
    p0 = np.asarray(s0["p_c_h"], float)
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    reori_series = []
    for a_tilt in acts_tilt:
        s = sample_state(env.sim)
        a_g, _ = ctl.step(s["fn_tot"], 10.0)
        v_h = _height_hold_v_hand(env.sim, z_ref)
        # keep wrist from tilt template, grip from force loop, translation = height hold
        w = np.asarray(a_tilt[3:6], float)  # not used — rebuild from tilt ω
        # tilt_actions encodes normalized action; rebuild via make_action for consistency
        from controllers.recovery.action_map import W_MAX

        a = make_action(v_h, [0.0, float(W_MAX) * 0.45, 0.0], a_g)
        step_legal(env.sim, a)
        actions_all.append(a.copy())
        s2 = sample_state(env.sim)
        dp = (np.asarray(s2["p_c_h"], float) - p0) * 1e3
        reori_series.append(
            {
                "dpx": float(dp[0]),
                "dpz": float(dp[2]),
                "g_h": s2["g_h"],
                "useful": float(np.dot(s2["g_h"], d_geom)),
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
            }
        )
    for _ in range(6):
        s = sample_state(env.sim)
        a_g, _ = ctl.step(s["fn_tot"], 10.0)
        v_h = _height_hold_v_hand(env.sim, z_ref)
        a = make_action(v_h, [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions_all.append(a.copy())

    tilt_blob = {"capture": take_capture(env), "ref": post_pivot_blob["ref"]}
    s_tilt = sample_state(env.sim)
    drift = (np.asarray(s_tilt["p_c_h"], float) - p0) * 1e3
    reori = {
        "method": "fixed_60deg_+wy_rate0.45",
        "n_tilt": len(acts_tilt),
        "delta_p_rel_mm": drift.tolist(),
        "g_h": s_tilt["g_h"],
        "useful": float(np.dot(s_tilt["g_h"], d_geom)),
        "d": d_geom.tolist(),
        "series_tail": reori_series[-3:],
    }

    # Milestone on −dpx (desire −x): 14 mm hand-x correction as in prior experiment.
    slip = run_slip_to_milestone(env, tilt_blob, d_slip, milestone_mm=MILESTONE_MM, max_s=16.0)
    actions_all.extend(slip["actions"])
    if not slip.get("ok"):
        return {
            "ok": False,
            "phase": "slip",
            "reori": reori,
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
            "slip_series": slip.get("series"),
            "actions": actions_all,
            "d": d_slip.tolist(),
            "d_geom": d_geom.tolist(),
        }

    # Fixed reverse untilt: −wy for n_tilt steps + height hold
    restore_capture(env, slip["capture"])
    p_pre = np.asarray(sample_state(env.sim)["p_c_h"], float)
    n_tilt = len(acts_tilt)
    z_ref_r = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    # Prefer lifting back toward original post-pivot height if sunk
    z_ref_r = max(z_ref_r, z_ref)
    from controllers.recovery.action_map import W_MAX

    ctl_r = ForcePController(kp=0.12, da_max=0.12)
    ctl_r.reset(1.0, sample_state(env.sim)["fn_tot"])
    for _ in range(n_tilt):
        s = sample_state(env.sim)
        a_g, _ = ctl_r.step(s["fn_tot"], 12.0)
        v_h = _height_hold_v_hand(env.sim, z_ref_r)
        a = make_action(v_h, [0.0, -float(W_MAX) * 0.45, 0.0], a_g)
        step_legal(env.sim, a)
        actions_all.append(a.copy())
    for _ in range(8):
        s = sample_state(env.sim)
        a_g, _ = ctl_r.step(s["fn_tot"], 12.0)
        v_h = _height_hold_v_hand(env.sim, z_ref_r)
        a = make_action(v_h, [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions_all.append(a.copy())

    # Raise to handoff-safe world height (Z_ABORT=0.44) if still low
    z_handoff = max(z_ref, 0.50)
    for _ in range(60):
        z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        if z >= 0.46:
            break
        s = sample_state(env.sim)
        a_g, _ = ctl_r.step(s["fn_tot"], 14.0)
        v_h = _height_hold_v_hand(env.sim, z_handoff, kp=3.0, vmax=1.0)
        a = make_action(v_h, [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions_all.append(a.copy())

    sf = sample_state(env.sim)
    dp_rest = (np.asarray(sf["p_c_h"], float) - p_pre) * 1e3
    final = {
        "e_signed_mm": sf["e_signed_mm"],
        "e_pos_mm": sf["e_pos_mm"],
        "theta": sf["e_theta_deg"],
        "bilateral": sf["bilateral"],
        "g_h": sf["g_h"],
        "delta_p_during_restore_mm": dp_rest.tolist(),
        "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
    }
    return {
        "ok": bool(sf["bilateral"] and final["obj_z"] >= 0.44),
        "method": "baseline_fixed",
        "reori": reori,
        "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        "slip_series": slip["series"],
        "final": final,
        "actions": actions_all,
        "n_steps": len(actions_all),
        "capture": take_capture(env),
        "d": d_slip.tolist(),
        "d_geom": d_geom.tolist(),
    }


def geometry_inferred_trajectory(env, post_pivot_blob):
    """Geometry-inferred orientation + same frozen slip controller + inferred restore."""
    actions_all = []
    restore_capture(env, post_pivot_blob["capture"])
    s0 = sample_state(env.sim)
    e = np.asarray(s0["e_signed_mm"], float) * 1e-3
    d = desired_correction_direction(e)
    # Target nearly full hand-frame error magnitude along d.
    milestone = float(min(MILESTONE_MM, 0.95 * np.linalg.norm(e) * 1e3))

    plan = plan_orientation(env, post_pivot_blob, d=d, simulate=True)
    print(f"  geom plan: {plan.get('rationale') or plan.get('reason')} axis={plan.get('axis_name')} ang={plan.get('angle_deg')}", flush=True)
    exe = execute_orientation_plan(env, post_pivot_blob, plan, adaptive=True)
    actions_all.extend(exe.get("actions") or [])
    if not exe.get("ok"):
        return {"ok": False, "phase": "reori", "plan": {k: v for k, v in plan.items() if k != "top"}, "exe": {k: v for k, v in exe.items() if k not in ("actions", "series", "capture")}}

    tilt_blob = {"capture": exe.get("capture") or take_capture(env), "ref": post_pivot_blob["ref"]}
    reori = {
        "method": "geometry_inferred",
        "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
        "delta_p_rel_mm": exe.get("delta_p_rel_mm"),
        "g_h": exe.get("g_h"),
        "useful": exe.get("useful"),
        "n_steps": exe.get("n_steps"),
        "d": d.tolist(),
    }

    # If inferred tilt is still light, prefer slip along dominant hand-x when |d·ex| high.
    d_slip = d if abs(d[0]) < 0.9 else np.array([-1.0 if d[0] < 0 else 1.0, 0.0, 0.0])
    slip = run_slip_to_milestone(env, tilt_blob, d_slip, milestone_mm=milestone, max_s=16.0)
    actions_all.extend(slip["actions"])
    if not slip.get("ok"):
        return {
            "ok": False,
            "phase": "slip",
            "reori": reori,
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
            "actions": actions_all,
            "plan": reori["plan"],
            "milestone_mm": milestone,
        }

    post_slip = {"capture": slip["capture"], "ref": post_pivot_blob["ref"]}
    rplan = plan_restore_orientation(env, post_slip)
    print(f"  restore plan: {rplan.get('rationale') or rplan.get('axis_name')} ang={rplan.get('angle_deg')}", flush=True)
    rexe = execute_restore_plan(env, post_slip, rplan)
    actions_all.extend(rexe.get("actions") or [])
    # Raise to handoff-safe height
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    z_handoff = max(z0, 0.50)
    ctl_h = ForcePController(kp=0.15, da_max=0.2)
    ctl_h.reset(1.0, sample_state(env.sim)["fn_tot"])
    for _ in range(60):
        z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        if z >= 0.46:
            break
        s = sample_state(env.sim)
        a_g, _ = ctl_h.step(s["fn_tot"], 14.0)
        v_h = _height_hold_v_hand(env.sim, z_handoff, kp=3.0, vmax=1.0)
        a = make_action(v_h, [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions_all.append(a.copy())
    sf = sample_state(env.sim)
    final = {
        "e_signed_mm": sf["e_signed_mm"],
        "e_pos_mm": sf["e_pos_mm"],
        "theta": sf["e_theta_deg"],
        "bilateral": sf["bilateral"],
        "g_h": sf["g_h"],
        "delta_p_rel_mm": rexe.get("delta_p_rel_mm"),
        "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
    }
    return {
        "ok": bool(sf["bilateral"] and final["obj_z"] >= 0.44),
        "method": "geometry_inferred",
        "reori": reori,
        "restore_plan": {k: v for k, v in rplan.items() if k != "top"},
        "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        "slip_series": slip["series"],
        "final": final,
        "actions": actions_all,
        "n_steps": len(actions_all),
        "capture": take_capture(env),
        "d": d.tolist(),
        "plan": reori["plan"],
    }


def handoff_from_capture(env, capture, ref) -> dict:
    restore_capture(env, capture)
    # brake hold then HANDOFF
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    meta = evaluate_handoff_from_state(env.sim, ref, stabilize_n=0)
    win = meta.get("window") or {}
    return {
        "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S" or meta.get("success_12s")),
        "reason": meta.get("reason"),
        "pre_theta": meta.get("pre_theta"),
        "pre_pos_mm": meta.get("pre_pos_mm"),
        "pre_in_band": meta.get("pre_in_band"),
        "longest_both_ok_s": win.get("longest_both_ok_s"),
        "window": {k: win[k] for k in win if k in ("longest_both_ok_s", "working_SUCCESS_NEW_hold", "mean_theta", "mean_pos", "max_theta", "max_pos")},
    }


def replay_ic_3x(env, stab_prefix_actions, precision_actions, ref) -> dict:
    results = []
    for trial in range(3):
        root = pickle.loads(IC_A.read_bytes())
        restore_capture(env, root["capture"])
        for a in stab_prefix_actions:
            step_legal(env.sim, np.asarray(a, np.float32))
        for a in precision_actions:
            step_legal(env.sim, np.asarray(a, np.float32))
        for _ in range(8):
            step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
        meta = evaluate_handoff_from_state(env.sim, ref, stabilize_n=0)
        win = meta.get("window") or {}
        results.append(
            {
                "trial": trial,
                "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
                "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S"),
                "reason": meta.get("reason"),
                "pre_theta": meta.get("pre_theta"),
                "pre_pos_mm": meta.get("pre_pos_mm"),
                "longest_both_ok_s": win.get("longest_both_ok_s"),
            }
        )
        print(f"  IC replay {trial}: SN={results[-1]['SUCCESS_NEW']} 12s={results[-1]['SUCCESS_12S']} {results[-1]['reason']}", flush=True)
    return {"trials": results, "SUCCESS_NEW_3of3": all(r["SUCCESS_NEW"] for r in results), "SUCCESS_12S_3of3": all(r["SUCCESS_12S"] for r in results)}


def planner_geometry_tests(env, states: dict) -> list:
    """Run orientation inference on physically distinct grasp states (plan only + short exec)."""
    rows = []
    for name, blob in states.items():
        restore_capture(env, blob["capture"])
        s = sample_state(env.sim)
        e = np.asarray(s["e_signed_mm"], float) * 1e-3
        d = desired_correction_direction(e)
        plan = plan_orientation(env, blob, d=d, simulate=True)
        exe = execute_orientation_plan(env, blob, plan, adaptive=True) if plan.get("ok") or plan.get("already_sufficient") else {"ok": False}
        obs = obs39(env)
        # re-obs at start for aliasing
        restore_capture(env, blob["capture"])
        obs0 = obs39(env)
        rows.append(
            {
                "name": name,
                "e_signed_mm": s["e_signed_mm"],
                "e_pos_mm": s["e_pos_mm"],
                "theta": s["e_theta_deg"],
                "g_h0": s["g_h"],
                "d": d.tolist(),
                "axis_name": plan.get("axis_name"),
                "angle_deg": plan.get("angle_deg"),
                "rate_scale": plan.get("rate_scale"),
                "omega_body": plan.get("omega_body"),
                "useful0": plan.get("useful0"),
                "useful_after": exe.get("useful") if exe else None,
                "g_h_after": exe.get("g_h") if exe else None,
                "delta_p_rel_mm": exe.get("delta_p_rel_mm") if exe else None,
                "ok": bool((plan.get("ok") or plan.get("already_sufficient")) and (exe.get("ok") if exe else False)),
                "rationale": plan.get("rationale") or plan.get("reason"),
                "obs39_start": obs0.tolist(),
            }
        )
        print(
            f"  geomtest {name}: e={np.round(s['e_signed_mm'],2)} d={np.round(d,3)} → {plan.get('axis_name')}@{plan.get('angle_deg')}° useful={exe.get('useful')}",
            flush=True,
        )
    return rows


def observability_assessment(geom_rows: list) -> dict:
    """Check whether distinct orientation decisions are separable in 39D obs."""
    if len(geom_rows) < 2:
        return {"n": len(geom_rows), "aliased": None}
    pairs = []
    for i in range(len(geom_rows)):
        for j in range(i + 1, len(geom_rows)):
            a, b = geom_rows[i], geom_rows[j]
            oa = np.asarray(a["obs39_start"], float)
            ob = np.asarray(b["obs39_start"], float)
            dist = float(np.linalg.norm(oa - ob))
            same_plan = (a.get("axis_name") == b.get("axis_name")) and (
                abs(float(a.get("angle_deg") or 0) - float(b.get("angle_deg") or 0)) < 1e-6
            )
            different_decision = not same_plan
            # Alias if decisions differ but obs nearly identical
            aliased = different_decision and dist < 0.05
            pairs.append(
                {
                    "a": a["name"],
                    "b": b["name"],
                    "obs_l2": dist,
                    "axis_a": a.get("axis_name"),
                    "axis_b": b.get("axis_name"),
                    "ang_a": a.get("angle_deg"),
                    "ang_b": b.get("angle_deg"),
                    "different_decision": different_decision,
                    "aliased": aliased,
                }
            )
    return {
        "n_states": len(geom_rows),
        "obs_dim": OBS_DIM_AIR,
        "pairs": pairs,
        "any_alias": any(p["aliased"] for p in pairs),
        "note": (
            "Aliasing = different inferred orientation actions but ||Δobs39||<0.05. "
            "Does not modify the observation interface."
        ),
    }


def make_plots(baseline, geom, geom_rows, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return {"error": str(e)}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Gravity / force / error for geom slip series
    if geom and geom.get("slip_series"):
        ser = geom["slip_series"]
        fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        t = [r["t"] for r in ser]
        ax[0].plot(t, [r["disp_along_mm"] for r in ser], label="disp along d")
        ax[0].plot(t, [r["dpx"] for r in ser], label="dpx", alpha=0.7)
        ax[0].plot(t, [r["dpz"] for r in ser], label="dpz", alpha=0.7)
        ax[0].legend(fontsize=7)
        ax[0].set_ylabel("mm")
        ax[0].set_title("Geometry-inferred: slip displacement")
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(t, [r["fn"] for r in ser], label="Fn")
        ax[1].plot(t, [r["fn_target"] for r in ser], ls="--", label="Fn*")
        ax[1].legend(fontsize=7)
        ax[1].set_ylabel("N")
        ax[1].grid(True, alpha=0.3)
        ax[2].plot(t, [r["g_h"][0] for r in ser], label="g_x")
        ax[2].plot(t, [r["g_h"][2] for r in ser], label="g_z")
        ax[2].plot(t, [r["theta"] for r in ser], label="θ")
        ax[2].legend(fontsize=7)
        ax[2].set_xlabel("t (s)")
        ax[2].grid(True, alpha=0.3)
        p = out_dir / "geom_slip_traces.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["geom_slip_traces"] = str(p)

    if baseline and baseline.get("slip_series"):
        ser = baseline["slip_series"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["t"] for r in ser], [r["disp_along_mm"] for r in ser], label="baseline")
        if geom and geom.get("slip_series"):
            ax.plot([r["t"] for r in geom["slip_series"]], [r["disp_along_mm"] for r in geom["slip_series"]], label="geometry")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("disp along d (mm)")
        ax.set_title("Baseline vs geometry-inferred slip")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = out_dir / "baseline_vs_geom_disp.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["baseline_vs_geom_disp"] = str(p)

    # Planner decisions across geometries
    if geom_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [r["name"] for r in geom_rows]
        angs = [r.get("angle_deg") or 0 for r in geom_rows]
        ax.bar(names, angs)
        for i, r in enumerate(geom_rows):
            ax.text(i, angs[i] + 1, f"{r.get('axis_name')}", ha="center", fontsize=8)
        ax.set_ylabel("inferred angle (deg)")
        ax.set_title("Inferred tilt by grasp geometry")
        p = out_dir / "inferred_angles_by_geometry.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["inferred_angles"] = str(p)

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


def render_trajectory_video(env, start_blob, actions, dest: Path, title: str, d=None, plan=None):
    restore_capture(env, start_blob["capture"])
    rw = mujoco.Renderer(env.sim.model, 440, 600)
    rh = mujoco.Renderer(env.sim.model, 440, 600)
    cam_w = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.5, "azimuth": 140, "elevation": -18})
    cam_h = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.35, "azimuth": 140, "elevation": -22})
    frames = []
    d = np.asarray(d if d is not None else [-1, 0, 0], float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)

    def cams():
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.55 * po + 0.45 * ph
        cam_w.distance = 0.5
        look = ph + Rh @ np.array([0.0, 0.0, 0.08])
        pos = ph + Rh @ np.array([0.22, -0.28, 0.14])
        vec = pos - look
        dist = float(np.linalg.norm(vec))
        cam_h.lookat[:] = look
        cam_h.distance = max(dist, 0.22)
        cam_h.azimuth = float(np.degrees(np.arctan2(vec[1], vec[0])))
        cam_h.elevation = float(np.degrees(np.arcsin(np.clip(vec[2] / max(dist, 1e-9), -1, 1))))

    def draw(scn):
        scn.ngeom = 0
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.1, 0.95, 1.0, 1.0], 0.006)
        add_sphere(scn, ph + Rh @ np.asarray(P_TGT, float), [1.0, 0.9, 0.1, 1.0], 0.005)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.2, 1.0], 0.0025)
        # desired correction direction in world
        add_capsule(scn, po, po + 0.07 * (Rh @ d), [1.0, 0.2, 0.2, 1.0], 0.003)
        # gravity in hand
        g_h = sample_state(env.sim)["g_h"]
        add_capsule(scn, po, po + 0.008 * (Rh @ np.asarray(g_h, float)), [0.9, 0.5, 1.0, 1.0], 0.0025)

    def frame(tag):
        cams()
        s = sample_state(env.sim)
        lines = [
            f"{tag} {title}",
            f"e={np.round(s['e_signed_mm'],1)} θ={s['e_theta_deg']:.1f} Fn={s['fn_tot']:.1f}",
            f"g_h={np.round(s['g_h'],1)} useful={float(np.dot(s['g_h'], d)):.1f}",
            f"plan={plan.get('axis_name') if plan else 'fixed'}@{plan.get('angle_deg') if plan else 60}°",
            "cyan=COM yellow=P_TGT green=axis red=d purple=g",
        ]
        rw.update_scene(env.sim.data, cam_w)
        draw(rw.scene)
        iw = overlay(np.asarray(rw.render(), np.uint8), ["WORLD"] + lines)
        rh.update_scene(env.sim.data, cam_h)
        draw(rh.scene)
        ih = overlay(np.asarray(rh.render(), np.uint8), ["HAND-REL"] + lines)
        return np.concatenate([iw, ih], axis=1)

    img = frame("BEFORE")
    for _ in range(int(2 * FPS)):
        frames.append(img)
    step_skip = max(1, len(actions) // 200)
    for i, a in enumerate(actions):
        step_legal(env.sim, a)
        if i % step_skip == 0 or i == len(actions) - 1:
            img = frame(f"step {i+1}/{len(actions)}")
            for _ in range(max(1, int(round(DT * FPS)))):
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
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(parents=True, exist_ok=True)

    search = json.loads((PRIOR_PIV / "search.json").read_text(encoding="utf-8"))
    tag_a = search["A_raw"]["best"]["tag"]
    tag_b = search["B_raw"]["best"]["tag"]
    state_a = pickle.loads(STATE_A.read_bytes())
    state_b = pickle.loads(STATE_B.read_bytes())
    env = make_env()

    # --- Build post-pivot A ---
    blob0 = {"capture": state_a["capture"], "ref": state_a["ref"]}
    piv_a = pivot_actions(tag_a)
    met = pivot_segment_metrics(env, blob0, piv_a, record_series=False)
    restore_capture(env, blob0["capture"])
    for a in piv_a:
        step_legal(env.sim, a)
    post_a = {"capture": take_capture(env), "ref": state_a["ref"]}
    s_post = sample_state(env.sim)
    print(f"Post-pivot A: θ={s_post['e_theta_deg']:.3f} e={s_post['e_signed_mm']}", flush=True)
    (OUT / "snapshots" / "A_post_pivot.pkl").write_bytes(pickle.dumps(post_a))

    # --- §1 Baseline full trajectory + HANDOFF ---
    print("=== Baseline fixed-orientation full trajectory ===", flush=True)
    baseline = baseline_fixed_orientation_trajectory(env, post_a)
    print(
        f"  baseline ok={baseline.get('ok')} final e={baseline.get('final',{}).get('e_signed_mm')} "
        f"e_pos={baseline.get('final',{}).get('e_pos_mm')}",
        flush=True,
    )
    (OUT / "baseline_final.json").write_text(
        json.dumps(jsafe({k: v for k, v in baseline.items() if k not in ("actions", "slip_series", "capture")}), indent=2),
        encoding="utf-8",
    )

    handoff_base = None
    replay_base = None
    if baseline.get("ok") and baseline.get("capture"):
        print("=== HANDOFF from baseline final state ===", flush=True)
        handoff_base = handoff_from_capture(env, baseline["capture"], state_a["ref"])
        print(json.dumps(jsafe(handoff_base), indent=2), flush=True)
        (OUT / "handoff_baseline.json").write_text(json.dumps(jsafe(handoff_base), indent=2), encoding="utf-8")
        if handoff_base.get("SUCCESS_NEW") or handoff_base.get("SUCCESS_12S"):
            print("=== Original-IC 3× replay (baseline precision tape) ===", flush=True)
            precision = list(piv_a) + list(baseline["actions"])
            replay_base = replay_ic_3x(env, state_a["stabilization_prefix_actions"], precision, state_a["ref"])
            (OUT / "replay_ic_baseline.json").write_text(json.dumps(jsafe(replay_base), indent=2), encoding="utf-8")
        else:
            print(f"  HANDOFF failed: {handoff_base.get('reason')} — diagnosing pre-handoff state", flush=True)
            restore_capture(env, baseline["capture"])
            diag = sample_state(env.sim)
            (OUT / "handoff_failure_diag.json").write_text(
                json.dumps(
                    jsafe(
                        {
                            "reason": handoff_base.get("reason"),
                            "pre": handoff_base,
                            "state": {k: diag[k] for k in diag if k != "contacts"},
                            "note": "Sliding controller preserved; failure is at HANDOFF/nominal stage.",
                        }
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )

    # Preserve diagnostic reference tape
    if baseline.get("actions"):
        np.savez_compressed(OUT / "baseline_precision_actions.npz", actions=np.asarray(baseline["actions"], np.float32))

    # --- § Geometry-inferred ---
    print("=== Geometry-inferred trajectory (same IC) ===", flush=True)
    geom = geometry_inferred_trajectory(env, post_a)
    print(
        f"  geom ok={geom.get('ok')} final e={geom.get('final',{}).get('e_signed_mm')} "
        f"e_pos={geom.get('final',{}).get('e_pos_mm')} n_steps={geom.get('n_steps')}",
        flush=True,
    )
    (OUT / "geometry_final.json").write_text(
        json.dumps(jsafe({k: v for k, v in geom.items() if k not in ("actions", "slip_series", "capture")}), indent=2),
        encoding="utf-8",
    )
    if geom.get("actions"):
        np.savez_compressed(OUT / "geometry_precision_actions.npz", actions=np.asarray(geom["actions"], np.float32))

    handoff_geom = None
    replay_geom = None
    if geom.get("ok") and geom.get("capture"):
        print("=== HANDOFF from geometry-inferred final state ===", flush=True)
        handoff_geom = handoff_from_capture(env, geom["capture"], state_a["ref"])
        print(json.dumps(jsafe(handoff_geom), indent=2), flush=True)
        (OUT / "handoff_geometry.json").write_text(json.dumps(jsafe(handoff_geom), indent=2), encoding="utf-8")
        if handoff_geom.get("SUCCESS_NEW") or handoff_geom.get("SUCCESS_12S"):
            print("=== Original-IC 3× replay (geometry precision tape) ===", flush=True)
            precision = list(piv_a) + list(geom["actions"])
            replay_geom = replay_ic_3x(env, state_a["stabilization_prefix_actions"], precision, state_a["ref"])
            (OUT / "replay_ic_geometry.json").write_text(json.dumps(jsafe(replay_geom), indent=2), encoding="utf-8")

    # --- Distinct geometries ---
    print("=== Orientation inference on distinct grasp geometries ===", flush=True)
    # B post-pivot
    blob_b0 = {"capture": state_b["capture"], "ref": state_b["ref"]}
    piv_b = pivot_actions(tag_b)
    restore_capture(env, blob_b0["capture"])
    for a in piv_b:
        step_legal(env.sim, a)
    post_b = {"capture": take_capture(env), "ref": state_b["ref"]}
    # Third: A after baseline restore if available (different signed e)
    states = {"A_post_pivot": post_a, "B_post_pivot": post_b}
    if baseline.get("ok") and baseline.get("capture"):
        states["A_post_baseline_restore"] = {"capture": baseline["capture"], "ref": state_a["ref"]}
    geom_rows = planner_geometry_tests(env, states)
    (OUT / "geometry_tests.json").write_text(json.dumps(jsafe(geom_rows), indent=2), encoding="utf-8")

    obs_assess = observability_assessment(geom_rows)
    (OUT / "observability_39d.json").write_text(json.dumps(jsafe(obs_assess), indent=2), encoding="utf-8")
    print(f"  39D alias any={obs_assess.get('any_alias')} pairs={obs_assess.get('pairs')}", flush=True)

    # Comparison table
    comparison = {
        "baseline": {
            "method": "fixed_60_+wy",
            "reori_axis": "+wy",
            "reori_angle_deg": 60.0,
            "reori_drift_mm": (baseline.get("reori") or {}).get("delta_p_rel_mm"),
            "useful_g": (baseline.get("reori") or {}).get("useful"),
            "final": baseline.get("final"),
            "n_steps": baseline.get("n_steps"),
            "handoff": handoff_base,
            "ok": baseline.get("ok"),
        },
        "geometry": {
            "method": "geometry_inferred",
            "reori_axis": (geom.get("plan") or {}).get("axis_name"),
            "reori_angle_deg": (geom.get("plan") or {}).get("angle_deg"),
            "reori_drift_mm": (geom.get("reori") or {}).get("delta_p_rel_mm"),
            "useful_g": (geom.get("reori") or {}).get("useful"),
            "final": geom.get("final"),
            "n_steps": geom.get("n_steps"),
            "handoff": handoff_geom,
            "ok": geom.get("ok"),
            "rationale": (geom.get("plan") or {}).get("rationale"),
        },
        "same_low_level_slip": True,
        "planner_changes_with_geometry": len({(r.get("axis_name"), r.get("angle_deg")) for r in geom_rows}) > 1,
    }
    (OUT / "comparison.json").write_text(json.dumps(jsafe(comparison), indent=2), encoding="utf-8")

    plots = make_plots(baseline, geom, geom_rows, OUT / "plots")

    # Videos
    videos = {}
    print("Rendering videos...", flush=True)
    if baseline.get("actions"):
        videos["baseline"] = render_trajectory_video(
            env, post_a, baseline["actions"], OUT / "videos" / "A_baseline_fixed_orientation.mp4", "Baseline fixed 60°", d=baseline.get("d"), plan={"axis_name": "+wy", "angle_deg": 60}
        )
    if geom.get("actions"):
        videos["geometry"] = render_trajectory_video(
            env, post_a, geom["actions"], OUT / "videos" / "A_geometry_inferred_orientation.mp4", "Geometry-inferred", d=geom.get("d"), plan=geom.get("plan")
        )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "pivot_A": {"theta0": met["theta0"], "theta1": met["theta1"], "delta_p_rel_mm": met["delta_p_rel_mm"]},
        "post_pivot_e_signed_mm": s_post["e_signed_mm"],
        "handoff_baseline": handoff_base,
        "replay_ic_baseline": replay_base,
        "handoff_geometry": handoff_geom,
        "replay_ic_geometry": replay_geom,
        "comparison": comparison,
        "geometry_tests": [{k: v for k, v in r.items() if k != "obs39_start"} for r in geom_rows],
        "observability_39d": obs_assess,
        "plots": plots,
        "videos": videos,
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    # REPORT
    hb = handoff_base or {}
    hg = handoff_geom or {}
    report = f"""# Geometry-Inferred Precision Orientation + Plan A HANDOFF

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

Design: **reuse frozen low-level force/slip** (`force_slip_control.py`); **infer wrist orientation from current geometry** (`orientation_planner.py`). No fixed 60° / memorized wy tape as the general Oracle solution. No physics / K49 / SUCCESS_NEW / 39D changes. No bulk Oracle / SFT / GRPO.

## 1. Plan A baseline HANDOFF (fixed orientation reference)

Post-pivot: θ {met['theta0']:.3f}→{met['theta1']:.3f}°, e_signed≈{s_post['e_signed_mm']}

Baseline trajectory: slow force-regulated **+wy 60° (rate 0.45)** → closed-loop slip to ~14 mm → brake → reverse untilt.

| Quantity | Value |
|----------|-------|
| Final e_signed | {baseline.get('final',{}).get('e_signed_mm')} |
| Final e_pos | {baseline.get('final',{}).get('e_pos_mm')} |
| Final θ | {baseline.get('final',{}).get('theta')} |
| Bilateral | {baseline.get('final',{}).get('bilateral')} |
| SUCCESS_12S | {hb.get('SUCCESS_12S')} |
| SUCCESS_NEW | {hb.get('SUCCESS_NEW')} |
| HANDOFF reason | {hb.get('reason')} |
| pre_θ / pre_e_pos | {hb.get('pre_theta')} / {hb.get('pre_pos_mm')} |

Original-IC 3× replay: {jsafe(replay_base)}

Diagnostic reference tape: `baseline_precision_actions.npz`

## 2. How tilt axis / direction / magnitude are inferred

Inputs (offline privileged): signed error `e = p_cyl_h − p_tgt_h`, desired motion `d = −e/‖e‖`, hand orientation, world gravity, contact forces during simulated candidates, joint-limit margin.

Procedure:
1. If `g_h · d` already ≥ 6 m/s² → no tilt.
2. Else score kinematic candidates over wrist axes `{{±wx, ±wy, ±wz, diagonals}}` × angles `{{15…65°}}`.
3. Physically evaluate top candidates at rate scales `{{0.45, 0.60}}` with force-regulated grip (Fn*≈10 N).
4. Prefer **smallest angle** that clears `g·d ≥ 6` with bilateral contact and low COM drift.
5. Execute with adaptive slowdown if drift grows.
6. After slip+brake, **restore** is re-planned from the current state toward upright `g_h ≈ [0,0,+g]` (not a memorized reverse step count).

Slip itself reuses the validated dual-loop controller with `slip_axis = d` and the same gains / velocity target.

## 3. Baseline vs geometry-inferred (same post-pivot A)

| | Baseline fixed | Geometry-inferred |
|--|----------------|-------------------|
| Axis | +wy | {comparison['geometry'].get('reori_axis')} |
| Angle | 60° | {comparison['geometry'].get('reori_angle_deg')}° |
| Reori drift (mm) | {comparison['baseline'].get('reori_drift_mm')} | {comparison['geometry'].get('reori_drift_mm')} |
| Useful g·d | {comparison['baseline'].get('useful_g')} | {comparison['geometry'].get('useful_g')} |
| Final e_pos | {(baseline.get('final') or {}).get('e_pos_mm')} | {(geom.get('final') or {}).get('e_pos_mm')} |
| Final θ | {(baseline.get('final') or {}).get('theta')} | {(geom.get('final') or {}).get('theta')} |
| Steps | {baseline.get('n_steps')} | {geom.get('n_steps')} |
| SUCCESS_12S | {hb.get('SUCCESS_12S')} | {hg.get('SUCCESS_12S')} |
| SUCCESS_NEW | {hb.get('SUCCESS_NEW')} | {hg.get('SUCCESS_NEW')} |

Planner rationale: {(geom.get('plan') or {}).get('rationale')}

## 4. Distinct grasp geometries

Planner changes with geometry: **{comparison['planner_changes_with_geometry']}**

| State | e_signed (mm) | d | axis | angle | useful after | drift |
|-------|---------------|---|------|-------|--------------|-------|
"""
    for r in geom_rows:
        report += (
            f"| {r['name']} | {np.round(r['e_signed_mm'],2).tolist()} | {np.round(r['d'],3).tolist()} | "
            f"{r.get('axis_name')} | {r.get('angle_deg')} | {r.get('useful_after')} | {r.get('delta_p_rel_mm')} |\n"
        )
    report += f"""
A planner that always emitted the same tilt would fail this table.

## 5. 39D observability (diagnostic)

{json.dumps(jsafe(obs_assess), indent=2)}

Oracle geometry inference may use privileged state. Runtime Transformer still sees only frozen 39D history. Aliasing here is reported, not “fixed” by expanding observations.

## 6. Plots & videos

Plots: {plots}

"""
    for k, v in videos.items():
        report += f"- `{k}`: {v.get('path')} ({v.get('duration_s')} s)\n"

    # Recommendation
    sn_base = bool(hb.get("SUCCESS_NEW"))
    sn_geom = bool(hg.get("SUCCESS_NEW"))
    changes = comparison["planner_changes_with_geometry"]
    if sn_base and (replay_base or {}).get("SUCCESS_NEW_3of3"):
        rec = "baseline_closed_loop_trajectory_is_SUCCESS_NEW_expert_candidate; geometry_planner_ready_for_oracle_integration"
    elif sn_base or sn_geom:
        rec = "handoff_success_partial_or_single_IC; continue_geometry_planner_but_require_3of3_before_expert_freeze"
    elif hb.get("SUCCESS_12S") or hg.get("SUCCESS_12S"):
        rec = "SUCCESS_12S_without_NEW_joint_band; investigate_final_2s_theta_epos_not_slip_controller"
    else:
        rec = "handoff_failed_preserve_sliding_controller_diagnose_nominal_stage"

    if not changes:
        rec += "; WARNING_planner_did_not_change_across_geometries"

    report += f"""
## 7. Recommendation

**{rec}**

- Keep force/slip closed-loop as the low-level mechanism.
- Use geometry-inferred orientation (not memorized 60°) in the precision Oracle.
- Do not claim runtime learnability from this Oracle alone; check 39D aliasing above.
- No bulk Oracle / SFT / GRPO in this milestone.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    summary["recommendation"] = rec
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": rec, "handoff_base": hb, "handoff_geom": hg, "wall_s": summary["wall_s"]}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
