#!/usr/bin/env python
"""Transfer geometry-inferred precision mechanism to Plan B hard_910220.

Transfers the *mechanism* (force/slip + orientation planner + height hold),
not a memorized Plan A action tape. Bounded experiment — no bulk Oracle/SFT/GRPO.
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

from controllers.recovery.action_map import W_MAX  # noqa: E402
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import decode_tok, load_vocab  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import (  # noqa: E402
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.corrections.orientation import (  # noqa: E402
    _height_hold_v_hand,
    desired_correction_direction,
    execute_orientation_plan,
    execute_restore_plan,
    plan_orientation,
    plan_restore_orientation,
    score_gravity_alignment,
)
from data_generation.oracles.precision.corrections.contact_primitives import (  # noqa: E402
    _action_vw,
    _wrist,
    pivot_segment_metrics,
)
from data_generation.oracles.precision.mechanics.grasp_stabilization import hold_and_confirm_stable  # noqa: E402
from envs.observations.airborne_obs import OBS_DIM_AIR, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_hand, cylinder_axis_world, hand_R  # noqa: E402
from evaluation.success_criteria import P_TGT, WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from evaluation.handoff_from_state import evaluate_handoff_from_state  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

# --- Paths (do not overwrite Plan A geometry_inferred_precision/) ---
STATE_B = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"
IC_B = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
POST_B = ROOT / "results/diagnostics/plan_b_posta_plan_a_correction/hard_910220/hard_910220__post_recatch.pkl"
PRIOR_PIV = ROOT / "results/diagnostics/position_preserving_pivot"
PLAN_A_SUMMARY = ROOT / "results/diagnostics/geometry_inferred_precision/summary.json"
OUT = ROOT / "results/diagnostics/plan_b_geometry_precision_transfer"

FPS = 25
V_SLIP = -0.001  # validated Plan A gain structure
HOLD_STABLE_S = 0.15
# Unchanged Plan A low-level gains
FORCE_KP = 0.10
FORCE_DA = 0.12
SLIP_KN = 900.0
FN_MIN, FN_MAX = 3.8, 11.0
FN_INIT = 5.2
FN_BRAKE = 14.0


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


def decompose_error(sim) -> dict:
    s = sample_state(sim)
    e = np.asarray(s["e_signed_mm"], float) * 1e-3
    d = desired_correction_direction(e)
    u = cylinder_axis_hand(sim)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    e_axial = float(np.dot(e, u))
    e_lat_vec = e - e_axial * u
    e_lat = float(np.linalg.norm(e_lat_vec))
    g = np.asarray(s["g_h"], float)
    sc = score_gravity_alignment(g, d)
    return {
        "e_signed_mm": s["e_signed_mm"],
        "e_pos_mm": s["e_pos_mm"],
        "e_theta_deg": s["e_theta_deg"],
        "e_hand_mm": (e * 1e3).tolist(),
        "e_hx_mm": float(e[0] * 1e3),
        "e_hy_mm": float(e[1] * 1e3),
        "e_hz_mm": float(e[2] * 1e3),
        "u_hand": u.tolist(),
        "e_axial_mm": e_axial * 1e3,
        "e_lateral_mm": e_lat * 1e3,
        "e_lateral_vec_mm": (e_lat_vec * 1e3).tolist(),
        "d": d.tolist(),
        "g_h": g.tolist(),
        "useful0": sc["useful"],
        "unwanted0": sc["unwanted"],
        "bilateral": s["bilateral"],
        "fn_L": s["fn_L"],
        "fn_R": s["fn_R"],
        "v_rel_norm": s["v_rel_norm"],
        "w_rel_norm": s["w_rel_norm"],
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "hand_z": float(sim.data.xpos[sim.ids.hand_body][2]),
        "single_direction_sufficient": bool(e_lat > 0.5 * abs(e_axial) or abs(e_axial) > 0.5 * e_lat),
        "note": (
            "If both |e_axial| and e_lateral are large, one gravity direction along d "
            "may still reduce total e_pos; sequential corrections only if first leaves e_pos>12."
        ),
    }


def reconstruct_stable_b(env, table, book48) -> dict:
    """Replay IC → active-recatch tokens → verify stable hold ≥150 ms."""
    root = pickle.loads(IC_B.read_bytes())
    state_cached = pickle.loads(STATE_B.read_bytes())
    tokens = list(state_cached.get("stabilization_prefix_tokens") or [32, 12, 12, 0, 0, 0, 0, 0])
    # Prefer post-recatch capture if available (same tokens), else replay from IC
    if POST_B.exists():
        post = pickle.loads(POST_B.read_bytes())
        restore_capture(env, post["capture"])
        ref = post.get("ref") or root["ref"]
        tokens = list(post.get("b1_tokens") or tokens)
        source = "post_recatch_snapshot"
    else:
        restore_capture(env, root["capture"])
        for tok in tokens:
            step_legal(env.sim, decode_tok(table, book48, int(tok)))
        ref = root["ref"]
        source = "ic_token_replay"
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    # Remake capture pre-hold for branching
    if POST_B.exists():
        restore_capture(env, pickle.loads(POST_B.read_bytes())["capture"])
    else:
        restore_capture(env, root["capture"])
        for tok in tokens:
            step_legal(env.sim, decode_tok(table, book48, int(tok)))
    s = sample_state(env.sim)
    cap = take_capture(env)
    # Also record continuous actions for exact IC replay (decoded tokens)
    restore_capture(env, root["capture"])
    prefix_actions = []
    for tok in tokens:
        a = decode_tok(table, book48, int(tok))
        step_legal(env.sim, a)
        prefix_actions.append(np.asarray(a, np.float32).copy())
    # sync to post-recatch capture if used
    if POST_B.exists():
        restore_capture(env, cap)
    return {
        "capture": cap,
        "ref": ref,
        "stabilization_prefix_tokens": tokens,
        "stabilization_prefix_actions": [a.tolist() for a in prefix_actions],
        "stability": stab,
        "geom": {k: s[k] for k in s if k != "contacts"},
        "source": source,
        "ic_id": "hard_910220",
        "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
        "hand_z": float(env.sim.data.xpos[env.sim.ids.hand_body][2]),
    }


def handoff_eval(env, capture, ref) -> dict:
    restore_capture(env, capture)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
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
        "window": {
            k: win[k]
            for k in win
            if k in ("longest_both_ok_s", "working_SUCCESS_NEW_hold", "mean_theta", "mean_pos", "max_theta", "max_pos")
        },
        "obj_z_pre": z0,
    }


def run_slip(env, blob, d_axis, *, milestone_mm: float, max_s: float = 16.0, z_ref: float | None = None):
    """Plan A dual-loop gains; slip along d; height hold from Plan B z_ref."""
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    d = np.asarray(d_axis, float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    axis = -d  # validated convention: v_along=v·axis, target=−1 mm/s ⇒ motion along d
    if z_ref is None:
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=V_SLIP, fn_min=FN_MIN, fn_max=FN_MAX, kn=SLIP_KN),
        use_slip=True,
    )
    ctl.reset(a0=0.2, fn0=s0["fn_tot"], fn_target=FN_INIT, slip_axis=tuple(axis.tolist()))
    actions, series = [], []
    lost = 0
    reached = False
    n_max = max(1, int(round(max_s / DT)))
    for i in range(n_max):
        s = sample_state(env.sim)
        _, meta = ctl.step(s)
        v_h = _height_hold_v_hand(env.sim, z_ref)
        a = make_action(v_h, [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        disp = float(np.dot(dp, d)) * 1e3
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
                "disp_along_mm": disp,
                "theta": s2["e_theta_deg"],
                "e_pos": s2["e_pos_mm"],
                "e_signed": s2["e_signed_mm"],
                "bilateral": s2["bilateral"],
                "g_h": s2["g_h"],
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "phase": "slip",
            }
        )
        if failed or latched(env.sim) or not s2["bilateral"]:
            return {"ok": False, "reason": "contact_loss", "actions": actions, "series": series, "reached": False}
        if disp >= milestone_mm:
            reached = True
            break
    if not series:
        return {"ok": False, "reason": "no_steps", "actions": actions, "series": series, "reached": False}

    ctl_b = ForcePController(kp=0.15, da_max=0.2)
    ctl_b.reset(series[-1]["a"], series[-1]["fn"])
    n_brake = max(1, int(round(0.40 / DT)))
    n_hold = max(1, int(round(HOLD_STABLE_S / DT)))
    for j in range(n_brake + n_hold):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], FN_BRAKE)
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
                "fn_target": FN_BRAKE,
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
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "phase": "brake" if j < n_brake else "hold",
            }
        )
        if not s2["bilateral"]:
            return {"ok": False, "reason": "loss_on_brake", "actions": actions, "series": series, "reached": reached}

    hold_ok = all(r["bilateral"] for r in series if r["phase"] == "hold")
    disp = float(series[-1]["disp_along_mm"])
    useful = reached or disp >= 0.85 * milestone_mm
    return {
        "ok": bool(hold_ok and useful),
        "reached": reached,
        "actions": actions,
        "series": series,
        "disp_along_mm": disp,
        "e_signed": series[-1]["e_signed"],
        "e_pos": series[-1]["e_pos"],
        "theta": series[-1]["theta"],
        "obj_z": series[-1]["obj_z"],
        "capture": take_capture(env),
        "min_obj_z": float(min(r["obj_z"] for r in series)),
    }


def raise_to_handoff_height(env, actions_all, z_target: float = 0.50):
    ctl = ForcePController(kp=0.15, da_max=0.2)
    ctl.reset(1.0, sample_state(env.sim)["fn_tot"])
    for _ in range(60):
        z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        if z >= 0.46:
            break
        s = sample_state(env.sim)
        a_g, _ = ctl.step(s["fn_tot"], FN_BRAKE)
        v_h = _height_hold_v_hand(env.sim, z_target, kp=3.0, vmax=1.0)
        a = make_action(v_h, [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions_all.append(a.copy())


def _lateral_desire(decomp: dict) -> np.ndarray:
    """Desire direction in the hand xy-plane (drop axial/hand-z for sequential stage 1)."""
    e = np.asarray(decomp["e_signed_mm"], float) * 1e-3
    e_xy = np.array([e[0], e[1], 0.0])
    n = float(np.linalg.norm(e_xy))
    if n < 1e-6:
        return desired_correction_direction(e)
    return -e_xy / n


def precision_trajectory(
    env,
    start_blob,
    *,
    label: str,
    milestone_mm: float | None = None,
    d_override: np.ndarray | None = None,
    g_useful_min: float | None = None,
):
    """Geometry-inferred reorient → slip → brake → restore → raise. Plan A gains."""
    actions_all = []
    restore_capture(env, start_blob["capture"])
    decomp = decompose_error(env.sim)
    d = np.asarray(d_override if d_override is not None else decomp["d"], float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    e_norm = float(np.linalg.norm(np.asarray(decomp["e_signed_mm"], float)))
    # Need e_pos ≤ 12 ⇒ at least (e_pos−12) mm along d; ask for margin
    need_mm = max(3.0, e_norm - WORKING_POS_MM + 2.0)
    milestone = float(milestone_mm if milestone_mm is not None else min(10.0, need_mm))
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])

    print(f"  [{label}] e_pos={decomp['e_pos_mm']:.2f} d={np.round(d,3)} milestone={milestone:.1f}mm z_ref={z_ref:.3f}", flush=True)

    plan_kw = {"d": d, "simulate": True}
    if g_useful_min is not None:
        plan_kw["g_useful_min"] = g_useful_min
    plan = plan_orientation(env, start_blob, **plan_kw)
    print(f"  [{label}] orient: {plan.get('rationale') or plan.get('reason')} axis={plan.get('axis_name')} ang={plan.get('angle_deg')} useful={plan.get('useful')}", flush=True)
    if not (plan.get("ok") or plan.get("already_sufficient")):
        return {
            "ok": False,
            "phase": "plan",
            "label": label,
            "decomp": decomp,
            "plan": {k: v for k, v in plan.items() if k != "top"},
            "reachable_best_useful": plan.get("useful"),
            "note": "Wrist interface may be unable to align g with full d; see sequential fallback.",
        }

    exe = execute_orientation_plan(env, start_blob, plan, adaptive=True)
    actions_all.extend(exe.get("actions") or [])
    if not exe.get("ok"):
        return {
            "ok": False,
            "phase": "reori",
            "label": label,
            "decomp": decomp,
            "plan": {k: v for k, v in plan.items() if k != "top"},
            "exe": {k: v for k, v in exe.items() if k not in ("actions", "series", "capture")},
        }

    tilt_blob = {"capture": exe.get("capture") or take_capture(env), "ref": start_blob["ref"]}
    z_tilt = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    z_ref_slip = max(z_ref, z_tilt)  # preserve Plan B height, not Plan A absolute

    # Progressive: short probe then remaining displacement along updated d
    short_mm = min(3.0, milestone)
    slip = run_slip(env, tilt_blob, d, milestone_mm=short_mm, max_s=16.0, z_ref=z_ref_slip)
    print(
        f"  [{label}] slip probe={short_mm}mm ok={slip.get('ok')} reached={slip.get('reached')} "
        f"disp={slip.get('disp_along_mm')} e_pos={slip.get('e_pos')} z={slip.get('obj_z')}",
        flush=True,
    )
    if not slip.get("ok"):
        return {
            "ok": False,
            "phase": "slip",
            "label": label,
            "decomp": decomp,
            "plan": {k: v for k, v in plan.items() if k != "top"},
            "reori": {"delta_p_rel_mm": exe.get("delta_p_rel_mm"), "g_h": exe.get("g_h"), "useful": exe.get("useful")},
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
            "actions": actions_all + list(slip.get("actions") or []),
            "milestone_attempted_mm": short_mm,
        }
    actions_all.extend(slip["actions"])
    tilt_blob = {"capture": slip["capture"], "ref": start_blob["ref"]}

    restore_capture(env, slip["capture"])
    e_now = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    if float(np.linalg.norm(e_now)) > WORKING_POS_MM:
        d = desired_correction_direction(e_now * 1e-3)
        rem = max(2.0, float(np.linalg.norm(e_now)) - WORKING_POS_MM + 2.0)
        rem = min(rem, milestone)
        slip2 = run_slip(env, tilt_blob, d, milestone_mm=rem, max_s=16.0, z_ref=z_ref_slip)
        print(
            f"  [{label}] slip remain={rem:.1f}mm ok={slip2.get('ok')} disp={slip2.get('disp_along_mm')} "
            f"e_pos={slip2.get('e_pos')}",
            flush=True,
        )
        if slip2.get("ok"):
            actions_all.extend(slip2["actions"])
            slip = {
                **{k: v for k, v in slip2.items() if k != "series"},
                "series": (slip.get("series") or []) + (slip2.get("series") or []),
                "probe_disp_mm": slip.get("disp_along_mm"),
            }
        else:
            # keep probe success; may still be enough after restore
            slip = {**slip, "extend_failed": {k: v for k, v in slip2.items() if k not in ("actions", "series", "capture")}}

    post_slip = {"capture": slip["capture"], "ref": start_blob["ref"]}
    restore_capture(env, post_slip["capture"])
    s_slip = sample_state(env.sim)
    in_band_post_slip = bool(
        s_slip["e_theta_deg"] <= WORKING_THETA_DEG and s_slip["e_pos_mm"] <= WORKING_POS_MM and s_slip["bilateral"]
    )
    print(
        f"  [{label}] post-slip e_pos={s_slip['e_pos_mm']:.2f} θ={s_slip['e_theta_deg']:.2f} in_band={in_band_post_slip}",
        flush=True,
    )

    restore_meta = {"skipped": False}
    if in_band_post_slip:
        # Do not undo a HANDOFF-ready pose with a large reverse tilt
        print(f"  [{label}] skip restore — already suitable for HANDOFF", flush=True)
        restore_meta = {"skipped": True, "reason": "already_in_band_post_slip"}
    else:
        rplan = plan_restore_orientation(env, post_slip)
        print(f"  [{label}] restore: {rplan.get('rationale') or rplan.get('axis_name')} ang={rplan.get('angle_deg')}", flush=True)
        rexe = execute_restore_plan(env, post_slip, rplan)
        actions_all.extend(rexe.get("actions") or [])
        restore_meta = {k: v for k, v in rplan.items() if k != "top"}
        restore_meta["exe_delta_p_rel_mm"] = rexe.get("delta_p_rel_mm")

    raise_to_handoff_height(env, actions_all, z_target=max(z_ref, 0.50))

    sf = sample_state(env.sim)
    final = {
        "e_signed_mm": sf["e_signed_mm"],
        "e_pos_mm": sf["e_pos_mm"],
        "theta": sf["e_theta_deg"],
        "bilateral": sf["bilateral"],
        "g_h": sf["g_h"],
        "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
        "in_band": bool(sf["e_theta_deg"] <= WORKING_THETA_DEG and sf["e_pos_mm"] <= WORKING_POS_MM and sf["bilateral"]),
    }
    return {
        "ok": bool(sf["bilateral"] and final["obj_z"] >= 0.44),
        "label": label,
        "decomp": decomp,
        "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
        "reori": {"delta_p_rel_mm": exe.get("delta_p_rel_mm"), "g_h": exe.get("g_h"), "useful": exe.get("useful"), "n_steps": exe.get("n_steps")},
        "restore_plan": restore_meta,
        "post_slip": {
            "e_pos_mm": s_slip["e_pos_mm"],
            "theta": s_slip["e_theta_deg"],
            "e_signed_mm": s_slip["e_signed_mm"],
            "in_band": in_band_post_slip,
        },
        "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        "slip_series": slip.get("series"),
        "final": final,
        "actions": actions_all,
        "n_steps": len(actions_all),
        "capture": take_capture(env),
        "d": d.tolist(),
        "milestone_mm": milestone,
        "gains_changed_from_plan_a": False,
        "gains": {"FORCE_KP": FORCE_KP, "SLIP_KN": SLIP_KN, "V_SLIP": V_SLIP, "FN_INIT": FN_INIT},
    }


def optional_second_correction(env, blob, actions_so_far: list) -> dict:
    """One bounded extra correction if e_pos still > 12."""
    restore_capture(env, blob["capture"])
    s = sample_state(env.sim)
    if s["e_pos_mm"] <= WORKING_POS_MM and s["e_theta_deg"] <= WORKING_THETA_DEG:
        return {"needed": False, "reason": "already_in_band", "actions": []}
    if s["e_pos_mm"] <= WORKING_POS_MM:
        return {"needed": False, "reason": "position_ok", "actions": []}
    print(f"  [second] residual e_pos={s['e_pos_mm']:.2f} — one bounded correction", flush=True)
    tr = precision_trajectory(env, blob, label="second", milestone_mm=min(6.0, s["e_pos_mm"] - WORKING_POS_MM + 2.0))
    return {"needed": True, "trajectory": {k: v for k, v in tr.items() if k not in ("actions", "slip_series")}, "actions": tr.get("actions") or [], "ok": tr.get("ok"), "capture": tr.get("capture"), "final": tr.get("final")}


def replay_ic_3x(env, table, book48, tokens, precision_actions, ref) -> dict:
    results = []
    for trial in range(3):
        root = pickle.loads(IC_B.read_bytes())
        restore_capture(env, root["capture"])
        for tok in tokens:
            step_legal(env.sim, decode_tok(table, book48, int(tok)))
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
        print(f"  IC replay {trial}: SN={results[-1]['SUCCESS_NEW']} {results[-1]['reason']}", flush=True)
    return {"trials": results, "SUCCESS_NEW_3of3": all(r["SUCCESS_NEW"] for r in results), "mode": "exact_frozen_actions"}


def observability_vs_plan_a(env, blob_b, obs_b) -> dict:
    """Compare Plan B 39D to Plan A post-pivot if available."""
    out = {"obs_dim": OBS_DIM_AIR, "plan_b_obs_norm": float(np.linalg.norm(obs_b))}
    a_snap = ROOT / "results/diagnostics/geometry_inferred_precision/snapshots/A_post_pivot.pkl"
    if a_snap.exists():
        a = pickle.loads(a_snap.read_bytes())
        restore_capture(env, a["capture"])
        obs_a = obs39(env)
        out["plan_a_post_pivot_l2"] = float(np.linalg.norm(obs_a - obs_b))
        out["distinguishable"] = out["plan_a_post_pivot_l2"] > 0.05
        out["note"] = "If L2 small, A/B orientation decisions may be hard to imitate from 39D alone."
    else:
        out["plan_a_snapshot_missing"] = True
    return out


def make_plots(traj, decomp, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return {"error": str(e)}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if traj and traj.get("slip_series"):
        ser = traj["slip_series"]
        fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        t = [r["t"] for r in ser]
        ax[0].plot(t, [r["disp_along_mm"] for r in ser], label="along d")
        ax[0].plot(t, [r["dpx"] for r in ser], label="dpx", alpha=0.7)
        ax[0].plot(t, [r["dpz"] for r in ser], label="dpz", alpha=0.7)
        ax[0].legend(fontsize=7)
        ax[0].set_ylabel("mm")
        ax[0].set_title("Plan B slip displacement")
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(t, [r["fn"] for r in ser], label="Fn")
        ax[1].plot(t, [r["fn_target"] for r in ser], ls="--", label="Fn*")
        ax[1].legend(fontsize=7)
        ax[1].set_ylabel("N")
        ax[1].grid(True, alpha=0.3)
        ax[2].plot(t, [r["e_pos"] for r in ser], label="e_pos")
        ax[2].plot(t, [r["theta"] for r in ser], label="θ")
        ax[2].plot(t, [r["obj_z"] for r in ser], label="obj_z")
        ax[2].axhline(WORKING_POS_MM, ls="--", alpha=0.4)
        ax[2].legend(fontsize=7)
        ax[2].set_xlabel("t (s)")
        ax[2].grid(True, alpha=0.3)
        p = out_dir / "plan_b_slip_traces.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["slip_traces"] = str(p)

    # Error decomposition bar
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = ["e_hx", "e_hy", "e_hz", "e_axial", "e_lateral"]
    vals = [decomp["e_hx_mm"], decomp["e_hy_mm"], decomp["e_hz_mm"], decomp["e_axial_mm"], decomp["e_lateral_mm"]]
    ax.bar(names, vals)
    ax.set_ylabel("mm")
    ax.set_title("Plan B signed error decomposition")
    ax.grid(True, alpha=0.3, axis="y")
    p = out_dir / "error_decomposition.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["error_decomp"] = str(p)
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


def render_video(env, start_blob, actions, dest: Path, title: str, d=None, plan=None):
    restore_capture(env, start_blob["capture"])
    rw = mujoco.Renderer(env.sim.model, 440, 600)
    rh = mujoco.Renderer(env.sim.model, 440, 600)
    cam_w = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.5, "azimuth": 140, "elevation": -18})
    cam_h = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.35, "azimuth": 140, "elevation": -22})
    frames = []
    d = np.asarray(d if d is not None else [-1, 0, 0], float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])

    def cams():
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.55 * po + 0.45 * ph
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
        add_capsule(scn, po, po + 0.07 * (Rh @ d), [1.0, 0.2, 0.2, 1.0], 0.003)
        g_h = sample_state(env.sim)["g_h"]
        add_capsule(scn, po, po + 0.008 * (Rh @ np.asarray(g_h, float)), [0.9, 0.5, 1.0, 1.0], 0.0025)

    def frame(tag):
        cams()
        s = sample_state(env.sim)
        z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        lines = [
            f"{tag} {title}",
            f"e={np.round(s['e_signed_mm'],1)} θ={s['e_theta_deg']:.1f} e_pos={s['e_pos_mm']:.1f}",
            f"Fn={s['fn_tot']:.1f} g·d={float(np.dot(s['g_h'], d)):.1f} z={z:.3f} margin={z-0.44:.3f}",
            f"plan={plan.get('axis_name') if plan else '?'}@{plan.get('angle_deg') if plan else '?'}°",
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
    step_skip = max(1, len(actions) // 220)
    for i, a in enumerate(actions):
        step_legal(env.sim, a)
        if i % step_skip == 0 or i == len(actions) - 1:
            img = frame(f"{i+1}/{len(actions)}")
            for _ in range(max(1, int(round(DT * FPS)))):
                frames.append(img)
    img = frame("AFTER")
    for _ in range(int(3 * FPS)):
        frames.append(img)
    rw.close()
    rh.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "duration_s": len(frames) / FPS, "z0": z0}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(parents=True, exist_ok=True)

    # Preserve pointer to Plan A results (do not modify)
    plan_a_ref = None
    if PLAN_A_SUMMARY.exists():
        plan_a_ref = json.loads(PLAN_A_SUMMARY.read_text(encoding="utf-8"))
        plan_a_ref = {
            "path": str(PLAN_A_SUMMARY),
            "handoff_baseline_SN": (plan_a_ref.get("handoff_baseline") or {}).get("SUCCESS_NEW"),
            "replay_3of3": (plan_a_ref.get("replay_ic_baseline") or {}).get("SUCCESS_NEW_3of3"),
            "note": "Plan A artifacts left untouched under geometry_inferred_precision/",
        }

    table, book48 = load_vocab()
    env = make_env()

    # --- §2 Reconstruct stable ---
    print("=== Reconstruct Plan B stable state ===", flush=True)
    stable = reconstruct_stable_b(env, table, book48)
    print(
        f"  stable={stable['stability'].get('stable')} source={stable['source']} "
        f"θ={stable['geom']['e_theta_deg']:.2f} e_pos={stable['geom']['e_pos_mm']:.2f} z={stable['obj_z']:.3f}",
        flush=True,
    )
    (OUT / "snapshots" / "B_stable.pkl").write_bytes(
        pickle.dumps({k: stable[k] for k in ("capture", "ref", "stabilization_prefix_tokens", "stabilization_prefix_actions", "stability", "geom", "source")})
    )
    (OUT / "stable_state.json").write_text(
        json.dumps(jsafe({k: stable[k] for k in stable if k != "capture"}), indent=2), encoding="utf-8"
    )

    blob0 = {"capture": stable["capture"], "ref": stable["ref"]}

    # --- §3 Direct HANDOFF baseline ---
    print("=== Direct HANDOFF baseline (no precision) ===", flush=True)
    baseline = handoff_eval(env, stable["capture"], stable["ref"])
    print(json.dumps(jsafe(baseline), indent=2), flush=True)
    (OUT / "handoff_direct_baseline.json").write_text(json.dumps(jsafe(baseline), indent=2), encoding="utf-8")

    # --- §4 Error decomposition ---
    restore_capture(env, stable["capture"])
    decomp = decompose_error(env.sim)
    print(f"  decomp: e={decomp['e_signed_mm']} axial={decomp['e_axial_mm']:.2f} lat={decomp['e_lateral_mm']:.2f} d={np.round(decomp['d'],3)}", flush=True)
    (OUT / "error_decomposition.json").write_text(json.dumps(jsafe(decomp), indent=2), encoding="utf-8")
    obs_b0 = obs39(env)

    # Optional pivot (improves θ; may change e)
    search = json.loads((PRIOR_PIV / "search.json").read_text(encoding="utf-8"))
    tag_b = search["B_raw"]["best"]["tag"]
    piv = pivot_actions(tag_b)
    met = pivot_segment_metrics(env, blob0, piv, record_series=False)
    print(f"  optional pivot: θ {met['theta0']:.2f}->{met['theta1']:.2f} Δp={met['delta_p_rel_mm']:.2f}", flush=True)
    restore_capture(env, blob0["capture"])
    for a in piv:
        step_legal(env.sim, a)
    post_piv = {"capture": take_capture(env), "ref": stable["ref"]}
    decomp_piv = decompose_error(env.sim)
    (OUT / "snapshots" / "B_post_pivot.pkl").write_bytes(pickle.dumps(post_piv))
    (OUT / "error_decomposition_post_pivot.json").write_text(json.dumps(jsafe(decomp_piv), indent=2), encoding="utf-8")

    # --- §5–9 Precision from stable (primary) and post-pivot (secondary) ---
    trials = []
    best = None

    # Prefer start with better chance: if θ already OK, start from stable; also try post-pivot
    starts = [("from_stable", blob0, decomp)]
    if decomp_piv["e_pos_mm"] < decomp["e_pos_mm"] + 2.0:
        starts.append(("from_post_pivot", post_piv, decomp_piv))
    else:
        starts.append(("from_post_pivot", post_piv, decomp_piv))

    def _finalize_trial(tr, label, blob_start):
        nonlocal best
        if not tr.get("ok"):
            trials.append({k: v for k, v in tr.items() if k not in ("actions", "slip_series", "capture")})
            return False
        actions = list(tr["actions"])
        cap = tr["capture"]
        # Second correction only if not already in-band (avoid undoing a good post-slip state)
        if not (tr.get("final") or {}).get("in_band"):
            second = optional_second_correction(env, {"capture": cap, "ref": stable["ref"]}, actions)
            if second.get("needed") and second.get("ok"):
                actions.extend(second["actions"])
                cap = second["capture"]
                tr["second"] = second.get("trajectory")
                tr["final"] = second.get("final") or tr["final"]
            elif second.get("needed"):
                tr["second"] = {k: v for k, v in second.items() if k not in ("actions", "capture", "trajectory")}
                tr["second_failed"] = True
        print(f"  [{label}] final e_pos={tr['final']['e_pos_mm']:.2f} θ={tr['final']['theta']:.2f} in_band={tr['final'].get('in_band')}", flush=True)
        print(f"=== HANDOFF after {label} ===", flush=True)
        h = handoff_eval(env, cap, stable["ref"])
        print(json.dumps(jsafe(h), indent=2), flush=True)
        tr["handoff"] = h
        tr["actions"] = actions
        tr["capture"] = cap
        trials.append({k: v for k, v in tr.items() if k not in ("actions", "slip_series", "capture")})
        trials[-1]["n_actions"] = len(actions)
        if h.get("SUCCESS_NEW"):
            best = tr
            return True
        if best is None or (tr.get("final") and tr["final"]["e_pos_mm"] < (best.get("final") or {}).get("e_pos_mm", 1e9)):
            best = tr
        return False

    for label, blob, dec in starts:
        print(f"=== Precision trajectory: {label} (full d) ===", flush=True)
        tr = precision_trajectory(env, blob, label=label)
        if _finalize_trial(tr, label, blob):
            break

        # Sequential lateral-first (Plan-A-like gravity along −hand-x; often easier restore)
        print(f"=== Sequential lateral-first: {label} ===", flush=True)
        d_lat = _lateral_desire(dec)
        e_hx = abs(dec["e_hx_mm"])
        e_hz = abs(dec["e_hz_mm"])
        if e_hz < WORKING_POS_MM:
            e_hx_max = float(np.sqrt(max(0.0, WORKING_POS_MM**2 - e_hz**2)))
            stage1_mm = max(3.0, e_hx - e_hx_max + 1.5)
        else:
            stage1_mm = max(5.0, 0.6 * e_hx)
        stage1_mm = min(stage1_mm, 12.0)
        tr1 = precision_trajectory(
            env, blob, label=f"{label}_lat", d_override=d_lat, milestone_mm=stage1_mm, g_useful_min=8.5
        )
        if not tr1.get("ok"):
            trials.append({k: v for k, v in tr1.items() if k not in ("actions", "slip_series", "capture")})
            continue
        actions_seq = list(tr1["actions"])
        cap1 = tr1["capture"]
        restore_capture(env, cap1)
        s1 = sample_state(env.sim)
        tr1["stage"] = "lateral"
        if s1["e_pos_mm"] > WORKING_POS_MM and s1["e_theta_deg"] <= WORKING_THETA_DEG:
            print(f"  residual after lat: e_pos={s1['e_pos_mm']:.2f} — second geom correction", flush=True)
            tr2 = precision_trajectory(
                env,
                {"capture": cap1, "ref": stable["ref"]},
                label=f"{label}_res",
                milestone_mm=min(8.0, s1["e_pos_mm"] - WORKING_POS_MM + 2.0),
                g_useful_min=7.0,
            )
            if tr2.get("ok"):
                actions_seq.extend(tr2["actions"])
                tr1["residual_stage"] = {k: v for k, v in tr2.items() if k not in ("actions", "slip_series", "capture")}
                tr1["final"] = tr2["final"]
                tr1["capture"] = tr2["capture"]
                tr1["actions"] = actions_seq
            else:
                tr1["residual_stage"] = {k: v for k, v in tr2.items() if k not in ("actions", "slip_series", "capture")}
                tr1["actions"] = actions_seq
                tr1["capture"] = cap1
        else:
            tr1["actions"] = actions_seq
            tr1["capture"] = cap1
        tr1["ok"] = bool(tr1.get("final") and tr1["final"].get("bilateral") and tr1["final"].get("obj_z", 0) >= 0.44)
        tr1["strategy"] = "sequential_lateral_then_residual"
        if _finalize_trial(tr1, f"{label}_sequential", blob):
            break

    # If no SUCCESS_NEW but best in-band, still try handoff already done
    replay = None
    if best and (best.get("handoff") or {}).get("SUCCESS_NEW"):
        print("=== Original-IC 3× exact replay ===", flush=True)
        # Include pivot if best started from post-pivot
        prefix_prec = list(best["actions"])
        if best.get("label") == "from_post_pivot":
            prefix_prec = list(piv) + prefix_prec
        replay = replay_ic_3x(env, table, book48, stable["stabilization_prefix_tokens"], prefix_prec, stable["ref"])
        (OUT / "replay_ic.json").write_text(json.dumps(jsafe(replay), indent=2), encoding="utf-8")
        np.savez_compressed(OUT / "best_precision_actions.npz", actions=np.asarray(best["actions"], np.float32))
        if best.get("label") == "from_post_pivot":
            np.savez_compressed(OUT / "best_full_precision_with_pivot.npz", actions=np.asarray(prefix_prec, np.float32))

    if best and best.get("actions"):
        (OUT / "snapshots" / "B_best_final.pkl").write_bytes(pickle.dumps({"capture": best["capture"], "ref": stable["ref"], "label": best.get("label")}))

    # Observability
    restore_capture(env, blob0["capture"])
    obs_audit = observability_vs_plan_a(env, blob0, obs_b0)
    # Also compare required decisions: Plan A wanted −x / +wy; Plan B wants d with z
    obs_audit["plan_b_d"] = decomp["d"]
    obs_audit["plan_b_e"] = decomp["e_signed_mm"]
    if best:
        obs_audit["plan_b_inferred_axis"] = (best.get("plan") or {}).get("axis_name")
        obs_audit["plan_b_inferred_angle"] = (best.get("plan") or {}).get("angle_deg")
    (OUT / "observability_39d.json").write_text(json.dumps(jsafe(obs_audit), indent=2), encoding="utf-8")

    plots = make_plots(best, decomp, OUT / "plots")
    videos = {}
    if best and best.get("actions"):
        print("Rendering video...", flush=True)
        start = post_piv if best.get("label") == "from_post_pivot" else blob0
        # video of precision only from start blob
        videos["best"] = render_video(
            env,
            start,
            best["actions"],
            OUT / "videos" / "B_geometry_precision_transfer.mp4",
            f"Plan B {best.get('label')}",
            d=best.get("d") or decomp["d"],
            plan=best.get("plan"),
        )

    # Comparison with Plan A mechanism
    comparison = {
        "plan_a_preserved": plan_a_ref,
        "plan_b_direct_handoff": baseline,
        "plan_b_best": {
            "label": None if not best else best.get("label"),
            "final": None if not best else best.get("final"),
            "plan_axis": None if not best else (best.get("plan") or {}).get("axis_name"),
            "plan_angle": None if not best else (best.get("plan") or {}).get("angle_deg"),
            "handoff": None if not best else best.get("handoff"),
            "gains_changed_from_plan_a": False if not best else best.get("gains_changed_from_plan_a"),
            "n_steps": None if not best else best.get("n_steps"),
        },
        "trials": trials,
        "replay_ic": replay,
        "transfer_notes": (
            "Same DualLoopController / ForcePController gains as Plan A. "
            "Orientation axis/angle and slip direction d recomputed from Plan B state. "
            "Height target derived from Plan B z_ref, not Plan A absolute height."
        ),
    }
    (OUT / "comparison.json").write_text(json.dumps(jsafe(comparison), indent=2), encoding="utf-8")
    (OUT / "trials.json").write_text(json.dumps(jsafe(trials), indent=2), encoding="utf-8")

    sn = bool(best and (best.get("handoff") or {}).get("SUCCESS_NEW"))
    sn33 = bool(replay and replay.get("SUCCESS_NEW_3of3"))
    if sn and sn33:
        rec = "plan_b_geometry_transfer_SUCCESS_NEW_3of3"
    elif sn:
        rec = "plan_b_SUCCESS_NEW_from_branch_but_check_ic_replay"
    elif best and best.get("final", {}).get("in_band"):
        rec = "in_band_but_handoff_failed_diagnose_nominal"
    elif best and best.get("ok"):
        rec = "partial_correction_e_pos_still_above_12_or_handoff_fail"
    else:
        rec = "transfer_failed_document_physical_or_control_limits"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "ic_id": "hard_910220",
        "plan_a_preserved": plan_a_ref,
        "stable": {k: stable[k] for k in stable if k != "capture"},
        "direct_handoff": baseline,
        "error_decomposition": decomp,
        "error_decomposition_post_pivot": decomp_piv,
        "pivot": {"theta0": met["theta0"], "theta1": met["theta1"], "delta_p_rel_mm": met["delta_p_rel_mm"], "tag": tag_b},
        "best_label": None if not best else best.get("label"),
        "best_final": None if not best else best.get("final"),
        "best_handoff": None if not best else best.get("handoff"),
        "best_plan": None if not best else best.get("plan"),
        "replay_ic": replay,
        "observability_39d": obs_audit,
        "plots": plots,
        "videos": videos,
        "recommendation": rec,
        "gains_unchanged_from_plan_a": True,
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    # REPORT
    bf = (best or {}).get("final") or {}
    bh = (best or {}).get("handoff") or {}
    bp = (best or {}).get("plan") or {}
    report = f"""# Plan B Geometry-Inferred Precision Transfer — hard_910220

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

**Transfer the mechanism, not a Plan A tape.** Plan A artifacts under `geometry_inferred_precision/` are preserved untouched.

Low-level gains unchanged from Plan A (`FORCE_KP={FORCE_KP}`, `SLIP_KN={SLIP_KN}`, `V_SLIP={V_SLIP}`). Orientation axis/angle and slip direction `d` recomputed from Plan B state. Height hold uses Plan B `z_ref`.

## 1. Stable state

| Quantity | Value |
|----------|-------|
| Source | {stable['source']} |
| Stable hold | {stable['stability'].get('stable')} |
| θ / e_pos | {stable['geom']['e_theta_deg']:.2f}° / {stable['geom']['e_pos_mm']:.2f} mm |
| e_signed | {stable['geom']['e_signed_mm']} |
| obj_z | {stable['obj_z']:.3f} m |
| Prefix tokens | {stable['stabilization_prefix_tokens']} |

Snapshot: `snapshots/B_stable.pkl`

## 2. Direct HANDOFF baseline

| SUCCESS_12S | SUCCESS_NEW | reason | pre_θ | pre_e_pos |
|-------------|-------------|--------|-------|-----------|
| {baseline.get('SUCCESS_12S')} | {baseline.get('SUCCESS_NEW')} | {baseline.get('reason')} | {baseline.get('pre_theta')} | {baseline.get('pre_pos_mm')} |

## 3. Error decomposition

| Component | mm |
|-----------|-----|
| e_hx / e_hy / e_hz | {decomp['e_hx_mm']:.2f} / {decomp['e_hy_mm']:.2f} / {decomp['e_hz_mm']:.2f} |
| e_axial (along u_hand) | {decomp['e_axial_mm']:.2f} |
| e_lateral | {decomp['e_lateral_mm']:.2f} |
| d = −e/‖e‖ | {np.round(decomp['d'], 4).tolist()} |
| g·d before tilt | {decomp['useful0']:.2f} |

Both axial (~hand-z) and lateral (~hand-x) are substantial → single gravity direction along `d` is the first strategy; second correction only if residual e_pos>12.

Optional pivot: θ {met['theta0']:.2f}→{met['theta1']:.2f}°, post-pivot e_pos={decomp_piv['e_pos_mm']:.2f} mm.

## 4. Geometry-inferred orientation (best trial)

| | Value |
|--|-------|
| Start | {best.get('label') if best else 'none'} |
| Axis / angle | {bp.get('axis_name')} / {bp.get('angle_deg')}° |
| Rate | {bp.get('rate_scale')} |
| Useful g·d | {bp.get('useful') or (best or {}).get('reori', {}).get('useful')} |
| Rationale | {bp.get('rationale')} |

## 5. Best complete trajectory

| | Value |
|--|-------|
| Final e_signed | {bf.get('e_signed_mm')} |
| Final e_pos / θ | {bf.get('e_pos_mm')} / {bf.get('theta')} |
| In-band | {bf.get('in_band')} |
| obj_z | {bf.get('obj_z')} |
| SUCCESS_12S / NEW | {bh.get('SUCCESS_12S')} / {bh.get('SUCCESS_NEW')} |
| HANDOFF reason | {bh.get('reason')} |
| IC 3/3 | {None if not replay else replay.get('SUCCESS_NEW_3of3')} |

## 6. Trials summary

"""
    for t in trials:
        report += (
            f"- **{t.get('label')}**: ok={t.get('ok')} phase={t.get('phase')} "
            f"final_e_pos={((t.get('final') or {}).get('e_pos_mm'))} "
            f"SN={((t.get('handoff') or {}).get('SUCCESS_NEW'))} "
            f"axis={((t.get('plan') or {}).get('axis_name'))}@{(t.get('plan') or {}).get('angle_deg')}\n"
        )
    report += f"""
## 7. Transfer vs Plan A

- Plan A SUCCESS_NEW 3/3 preserved: {plan_a_ref}
- Gains changed: **False** (same DualLoop / ForceP structure)
- What changed: `d`, inferred tilt axis/angle, Plan-B-derived `z_ref`
- Observability: {json.dumps(jsafe(obs_audit), indent=2)}

## 8. Plots & videos

Plots: {plots}

"""
    for k, v in videos.items():
        report += f"- `{k}`: {v.get('path')} ({v.get('duration_s')} s)\n"

    report += f"""
## 9. Transfer failure / adaptation notes

If SUCCESS_NEW is False after sequential lateral-first staging, the typical Plan B blocker is a residual
**hand-z / cylinder-axial** component after a successful lateral (−hand-x) gravity slip. Deep full-`d`
tilts can reduce e_pos but often destroy θ; restore and Cartesian axial assists tend to undo gains or
lose bilateral contact. Document any probe outcomes under `adaptation_probes.json` rather than
claiming a memorized Plan B tape.

## 10. Recommendation

**{rec}**

Stopped after this focused Plan B milestone. No bulk Oracle / SFT / GRPO.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": rec, "best_final": bf, "handoff": bh, "replay": replay, "wall_s": summary["wall_s"]}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
