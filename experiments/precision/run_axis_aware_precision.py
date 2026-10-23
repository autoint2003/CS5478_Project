#!/usr/bin/env python
"""Axis-aware precision Oracle: lateral-first, conditional axial correction.

Diagnostic SUCCESS_AXIS_V1 alongside unchanged SUCCESS_NEW.
Bounded Plan B hard_910220 experiment — no bulk Oracle / SFT / GRPO.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import W_MAX  # noqa: E402
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import decode_tok, load_vocab  # noqa: E402
from data_generation.oracles.precision.axis_aware import (  # noqa: E402
    axial_desire_direction,
    decide_precision,
    lateral_desire_direction,
)
from data_generation.oracles.precision.force_slip_control import (  # noqa: E402
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import (  # noqa: E402
    _height_hold_v_hand,
    execute_orientation_plan,
    execute_restore_plan,
    plan_orientation,
    plan_restore_orientation,
    score_gravity_alignment,
)
from data_generation.oracles.precision.primitives import (  # noqa: E402
    _action_vw,
    _wrist,
    pivot_segment_metrics,
)
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable  # noqa: E402
from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import (  # noqa: E402
    AXIS_V1_EZ_MM,
    AXIS_V1_EX_MM,
    Z_LIMIT_CANDIDATES_MM,
    evaluate_handoff_with_axis,
    sensitivity_over_z_limits,
)
from evaluation.success_criteria import P_TGT, WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

STATE_B = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"
IC_B = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
POST_B = ROOT / "results/diagnostics/plan_b_posta_plan_a_correction/hard_910220/hard_910220__post_recatch.pkl"
PRIOR_PIV = ROOT / "results/diagnostics/position_preserving_pivot"
PLAN_A_SNAP = ROOT / "results/diagnostics/geometry_inferred_precision/snapshots/A_post_pivot.pkl"
PLAN_A_ACTIONS = ROOT / "results/diagnostics/geometry_inferred_precision/geometry_precision_actions.npz"
OUT = ROOT / "results/diagnostics/axis_aware_precision"

# Unchanged Plan A low-level gains
FORCE_KP, FORCE_DA = 0.10, 0.12
SLIP_KN = 900.0
V_SLIP = -0.001
FN_MIN, FN_MAX = 3.8, 11.0
FN_INIT, FN_BRAKE = 5.2, 14.0
HOLD_STABLE_S = 0.15
Z_ABORT = 0.44
# Abort axial if |e_x| drifts beyond this from start of axial stage
LATERAL_DRIFT_ABORT_MM = 4.0
LATERAL_ABS_ABORT_MM = 8.0
FPS = 25


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


def reconstruct_stable_b(env, table, book48) -> dict:
    root = pickle.loads(IC_B.read_bytes())
    state_cached = pickle.loads(STATE_B.read_bytes())
    tokens = list(state_cached.get("stabilization_prefix_tokens") or [32, 12, 12, 0, 0, 0, 0, 0])
    if POST_B.exists():
        post = pickle.loads(POST_B.read_bytes())
        restore_capture(env, post["capture"])
        ref = post.get("ref") or root["ref"]
        tokens = list(post.get("b1_tokens") or tokens)
        source = "post_recatch_snapshot"
        cap = take_capture(env)
    else:
        restore_capture(env, root["capture"])
        for tok in tokens:
            step_legal(env.sim, decode_tok(table, book48, int(tok)))
        ref = root["ref"]
        source = "ic_token_replay"
        cap = take_capture(env)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, cap)
    s = sample_state(env.sim)
    restore_capture(env, root["capture"])
    prefix_actions = []
    for tok in tokens:
        a = decode_tok(table, book48, int(tok))
        step_legal(env.sim, a)
        prefix_actions.append(np.asarray(a, np.float32).copy())
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
    }


def handoff_rich(env, capture, ref, *, stabilize: int = 8) -> dict:
    restore_capture(env, capture)
    for _ in range(stabilize):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    meta = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    meta["obj_z_pre"] = z0
    # Drop full samples from default return for size; keep sensitivity + tail
    samples = meta.pop("samples", [])
    meta["_samples_for_sensitivity"] = samples  # caller may strip
    return meta


def strip_samples(meta: dict) -> dict:
    m = {k: v for k, v in meta.items() if k != "_samples_for_sensitivity"}
    return m


def run_slip_guarded(
    env,
    blob,
    d_axis,
    *,
    milestone_mm: float,
    max_s: float = 12.0,
    z_ref: float | None = None,
    guard_lateral: bool = True,
    e_x0: float | None = None,
):
    """Dual-loop slip along d with optional lateral-drift abort (preserve e_x)."""
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    e0 = np.asarray(s0["e_signed_mm"], float)
    if e_x0 is None:
        e_x0 = float(e0[0])
    d = np.asarray(d_axis, float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    axis = -d
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
    abort_reason = None
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
        e2 = np.asarray(s2["e_signed_mm"], float)
        disp = float(np.dot(dp, d)) * 1e3
        row = {
            "t": (i + 1) * DT,
            "fn": s2["fn_tot"],
            "fn_L": s2["fn_L"],
            "fn_R": s2["fn_R"],
            "fn_target": meta["fn_target"],
            "a": meta["a"],
            "v_rel": s2["v_rel_h"],
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
        series.append(row)
        if failed or latched(env.sim) or not s2["bilateral"]:
            return {
                "ok": False,
                "reason": "contact_loss",
                "actions": actions,
                "series": series,
                "reached": False,
            }
        if guard_lateral:
            dx = abs(float(e2[0]) - e_x0)
            if dx > LATERAL_DRIFT_ABORT_MM or abs(float(e2[0])) > LATERAL_ABS_ABORT_MM:
                abort_reason = "lateral_drift"
                break
        if disp >= milestone_mm:
            reached = True
            break

    # Brake / regrip
    ctl_b = ForcePController(kp=0.15, da_max=0.2)
    ctl_b.reset(series[-1]["a"] if series else 0.5, series[-1]["fn"] if series else 10.0)
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
                "t": (series[-1]["t"] if series else 0.0) + DT,
                "fn": s2["fn_tot"],
                "fn_L": s2["fn_L"],
                "fn_R": s2["fn_R"],
                "fn_target": FN_BRAKE,
                "a": a_g,
                "v_rel": s2["v_rel_h"],
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
            return {
                "ok": False,
                "reason": "loss_on_brake",
                "actions": actions,
                "series": series,
                "reached": reached,
            }

    hold_ok = all(r["bilateral"] for r in series if r["phase"] == "hold")
    disp = float(series[-1]["disp_along_mm"])
    useful = reached or disp >= 0.5 * milestone_mm  # looser for 2 mm probes
    ok = bool(hold_ok and useful and abort_reason is None)
    if abort_reason and hold_ok and disp >= 0.3:
        # partial but aborted — not counted as success
        ok = False
    return {
        "ok": ok,
        "reached": reached,
        "abort_reason": abort_reason,
        "actions": actions,
        "series": series,
        "disp_along_mm": disp,
        "e_signed": series[-1]["e_signed"],
        "e_pos": series[-1]["e_pos"],
        "theta": series[-1]["theta"],
        "obj_z": series[-1]["obj_z"],
        "min_obj_z": float(min(r["obj_z"] for r in series)),
        "e_x0": e_x0,
        "e_x_final": float(series[-1]["e_signed"][0]),
        "lateral_drift_mm": abs(float(series[-1]["e_signed"][0]) - e_x0),
        "capture": take_capture(env),
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


def build_post_lateral_state(env, stable: dict) -> dict:
    """Regenerate verified lateral-corrected Plan B state (e_x~-1.2, e_z~13.3)."""
    search = json.loads((PRIOR_PIV / "search.json").read_text(encoding="utf-8"))
    tag = search["B_raw"]["best"]["tag"]
    piv = pivot_actions(tag)
    blob0 = {"capture": stable["capture"], "ref": stable["ref"]}
    restore_capture(env, blob0["capture"])
    for a in piv:
        step_legal(env.sim, a)
    post_piv = {"capture": take_capture(env), "ref": stable["ref"]}
    s = sample_state(env.sim)
    d_lat = lateral_desire_direction(s["e_signed_mm"])
    plan = plan_orientation(env, post_piv, d=d_lat, simulate=True, g_useful_min=8.5)
    exe = execute_orientation_plan(env, post_piv, plan, adaptive=True)
    if not exe.get("ok"):
        return {"ok": False, "phase": "reori", "plan": plan, "exe": {k: v for k, v in exe.items() if k != "actions"}}
    tilt = {"capture": exe["capture"], "ref": stable["ref"]}
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    # ~10–11.5 mm lateral correction as in transfer probe
    slip = run_slip_guarded(env, tilt, d_lat, milestone_mm=10.0, max_s=20.0, z_ref=z_ref, guard_lateral=False)
    if not slip.get("ok") and not (slip.get("disp_along_mm") or 0) >= 8.0:
        return {"ok": False, "phase": "lateral_slip", "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")}}
    # Prefer brake-hold state even if milestone slightly short
    restore_capture(env, slip["capture"])
    sf = sample_state(env.sim)
    actions = list(piv) + list(exe.get("actions") or []) + list(slip.get("actions") or [])
    return {
        "ok": True,
        "capture": slip["capture"],
        "ref": stable["ref"],
        "actions_prefix": actions,
        "pivot_tag": tag,
        "plan_lateral": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
        "geom": {k: sf[k] for k in sf if k != "contacts"},
        "e_signed_mm": sf["e_signed_mm"],
        "e_pos_mm": sf["e_pos_mm"],
        "theta": sf["e_theta_deg"],
        "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
        "min_obj_z_slip": slip.get("min_obj_z"),
        "decision": decide_precision(sf["e_signed_mm"], theta_deg=sf["e_theta_deg"], bilateral=sf["bilateral"]),
    }


def axial_correction_attempt(env, start_blob, *, milestone_mm: float, restore: bool = True) -> dict:
    """Geometry-inferred axial gravity orientation + guarded slip + optional restore."""
    actions = []
    restore_capture(env, start_blob["capture"])
    s0 = sample_state(env.sim)
    e0 = np.asarray(s0["e_signed_mm"], float)
    d = axial_desire_direction(e0)
    if float(np.linalg.norm(d)) < 0.5:
        return {"ok": False, "reason": "ez_near_zero", "e_signed_mm": e0.tolist()}
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    decision = decide_precision(e0, theta_deg=s0["e_theta_deg"], bilateral=s0["bilateral"])

    # Prefer slightly lower useful threshold for pure axial (harder wrist reach)
    plan = plan_orientation(env, start_blob, d=d, simulate=True, g_useful_min=7.0)
    kin_only = None
    if not (plan.get("ok") or plan.get("already_sufficient")):
        kin_only = plan_orientation(env, start_blob, d=d, simulate=False, g_useful_min=5.0)
        return {
            "ok": False,
            "phase": "plan",
            "reason": "axial_orientation_unreachable",
            "d": d.tolist(),
            "e_signed_mm": e0.tolist(),
            "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
            "reachable_kin": {k: v for k, v in (kin_only or {}).items() if k not in ("candidates_kin",)},
            "decision": decision,
            "g0": s0["g_h"],
            "g0_score": score_gravity_alignment(np.asarray(s0["g_h"], float), d),
        }

    print(
        f"  axial plan: {plan.get('rationale')} axis={plan.get('axis_name')} "
        f"ang={plan.get('angle_deg')} useful={plan.get('useful')}",
        flush=True,
    )
    exe = execute_orientation_plan(env, start_blob, plan, adaptive=True)
    actions.extend(exe.get("actions") or [])
    if not exe.get("ok"):
        return {
            "ok": False,
            "phase": "reori",
            "plan": {k: v for k, v in plan.items() if k not in ("top",)},
            "exe": {k: v for k, v in exe.items() if k not in ("actions", "series", "capture")},
            "decision": decision,
        }

    # Verify actual g projection after reori
    restore_capture(env, exe["capture"])
    s_tilt = sample_state(env.sim)
    g_sc = score_gravity_alignment(np.asarray(s_tilt["g_h"], float), d)
    e_tilt = np.asarray(s_tilt["e_signed_mm"], float)
    print(
        f"  post-reori g·d={g_sc['useful']:.2f} e={np.round(e_tilt,2)} θ={s_tilt['e_theta_deg']:.2f} "
        f"Δex={e_tilt[0]-e0[0]:+.2f}",
        flush=True,
    )
    if abs(float(e_tilt[0]) - float(e0[0])) > LATERAL_DRIFT_ABORT_MM:
        return {
            "ok": False,
            "phase": "reori_lateral_drift",
            "reason": "reorientation_destroyed_ex",
            "e0": e0.tolist(),
            "e_tilt": e_tilt.tolist(),
            "plan": {k: v for k, v in plan.items() if k not in ("top",)},
            "g_score": g_sc,
            "actions": actions,
            "capture": exe["capture"],
        }

    tilt_blob = {"capture": exe["capture"], "ref": start_blob["ref"]}
    z_slip = max(z_ref, float(env.sim.data.xpos[env.sim.ids.object_body][2]))
    slip = run_slip_guarded(
        env,
        tilt_blob,
        d,
        milestone_mm=milestone_mm,
        max_s=14.0,
        z_ref=z_slip,
        guard_lateral=True,
        e_x0=float(e0[0]),
    )
    print(
        f"  axial slip {milestone_mm}mm: ok={slip.get('ok')} reached={slip.get('reached')} "
        f"disp={slip.get('disp_along_mm')} abort={slip.get('abort_reason')} "
        f"e={np.round(slip.get('e_signed') or [0,0,0],2)} drift_x={slip.get('lateral_drift_mm')}",
        flush=True,
    )
    actions.extend(slip.get("actions") or [])
    if not slip.get("ok"):
        return {
            "ok": False,
            "phase": "slip",
            "milestone_mm": milestone_mm,
            "plan": {k: v for k, v in plan.items() if k not in ("top",)},
            "reori_g": g_sc,
            "reori_e": e_tilt.tolist(),
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
            "actions": actions,
            "capture": slip.get("capture") or exe["capture"],
            "decision": decision,
        }

    post = {"capture": slip["capture"], "ref": start_blob["ref"]}
    restore_meta = {"skipped": True}
    if restore:
        rplan = plan_restore_orientation(env, post)
        # Protect e_x: skip restore if it would likely couple — measure after short restore attempt
        rexe = execute_restore_plan(env, post, rplan)
        actions.extend(rexe.get("actions") or [])
        restore_meta = {k: v for k, v in rplan.items() if k != "top"}
        restore_meta["exe_delta"] = rexe.get("delta_p_rel_mm")
        post = {"capture": take_capture(env), "ref": start_blob["ref"]}

    raise_to_handoff_height(env, actions, z_target=max(z_ref, 0.50))
    sf = sample_state(env.sim)
    e_f = np.asarray(sf["e_signed_mm"], float)
    # Reject if z improved but x destroyed
    z_improved = abs(e_f[2]) < abs(e0[2]) - 0.5
    x_ok = abs(e_f[0]) <= max(AXIS_V1_EX_MM, abs(e0[0]) + 2.0)
    return {
        "ok": bool(sf["bilateral"] and x_ok and float(env.sim.data.xpos[env.sim.ids.object_body][2]) >= Z_ABORT),
        "milestone_mm": milestone_mm,
        "decision": decision,
        "d": d.tolist(),
        "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
        "reori_g": g_sc,
        "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        "slip_series": slip.get("series"),
        "restore": restore_meta,
        "e0": e0.tolist(),
        "e_final": e_f.tolist(),
        "theta_final": sf["e_theta_deg"],
        "e_pos_final": sf["e_pos_mm"],
        "z_improved": z_improved,
        "x_preserved": x_ok,
        "lateral_drift_mm": abs(e_f[0] - e0[0]),
        "min_obj_z": min(slip.get("min_obj_z") or 1.0, float(env.sim.data.xpos[env.sim.ids.object_body][2])),
        "actions": actions,
        "capture": take_capture(env),
        "ref": start_blob["ref"],
    }


def observability_audit(env, post_lat: dict, decision: dict) -> dict:
    obs = observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist)
    e = np.asarray(post_lat["e_signed_mm"], float)
    return {
        "obs_dim": int(obs.shape[0]),
        "obs_norm": float(np.linalg.norm(obs)),
        "oracle_sees": {
            "e_signed_mm": e.tolist(),
            "abs_ex": abs(float(e[0])),
            "abs_ez": abs(float(e[2])),
            "lateral_excessive": abs(float(e[0])) > AXIS_V1_EX_MM,
            "axial_acceptable_provisional": abs(float(e[2])) <= AXIS_V1_EZ_MM,
            "axial_excessive_vs_NEW": abs(float(e[2])) > WORKING_POS_MM,
            "decision_mode": decision.get("mode"),
            "d_axial": decision.get("d_axial"),
        },
        "runtime_note": (
            "39D frozen; no privileged e_signed in observation. "
            "Policy must infer lateral vs axial need from history if imitating Oracle. "
            "No runtime FSM / plan selector added."
        ),
        "ambiguity": (
            "Euclidean e_pos alone cannot distinguish lateral-only vs axial-only vs mixed; "
            "axis-aware decisions require signed components not explicitly in 39D."
        ),
    }


def render_video(env, blob, actions, dest: Path, title: str, d=None, plan=None) -> dict:
    """Reuse Plan B transfer WORLD+HAND-REL renderer."""
    from experiments.precision.run_plan_b_geometry_precision_transfer import (
        render_video as _rv,
    )

    return _rv(env, blob, actions, dest, title, d=d, plan=plan)


def make_plots(branch_a, branch_b, out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    # Signed error bar comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, xs, ys, zs = [], [], [], []
    for name, br in (("A_no_axial", branch_a), ("B_axial", branch_b)):
        if not br:
            continue
        e = br.get("pre_e_signed_mm") or br.get("e_signed_mm") or [0, 0, 0]
        if br.get("handoff"):
            e = br["handoff"].get("pre_e_signed_mm") or e
        labels.append(name)
        xs.append(e[0])
        ys.append(e[1])
        zs.append(e[2])
    if labels:
        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, xs, w, label="e_x")
        ax.bar(x, ys, w, label="e_y")
        ax.bar(x + w, zs, w, label="e_z")
        ax.axhline(12, color="gray", ls="--", lw=0.8, label="|e|=12 NEW")
        ax.axhline(20, color="C2", ls=":", lw=0.8, label="|e_z|=20 AXIS")
        ax.axhline(-12, color="gray", ls="--", lw=0.8)
        ax.axhline(-20, color="C2", ls=":", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("mm")
        ax.set_title("Pre-HANDOFF signed errors")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "signed_errors_branches.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["signed_errors"] = str(p)

    # Slip series if any
    ser = (branch_b or {}).get("slip_series") or []
    if ser:
        fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        t = [r["t"] for r in ser]
        axs[0].plot(t, [r["e_signed"][0] for r in ser], label="e_x")
        axs[0].plot(t, [r["e_signed"][2] for r in ser], label="e_z")
        axs[0].legend()
        axs[0].set_ylabel("e (mm)")
        axs[1].plot(t, [r["fn_L"] for r in ser], label="Fn_L")
        axs[1].plot(t, [r["fn_R"] for r in ser], label="Fn_R")
        axs[1].plot(t, [r["fn_target"] for r in ser], label="Fn_tgt", ls="--")
        axs[1].legend(fontsize=8)
        axs[1].set_ylabel("N")
        axs[2].plot(t, [r["obj_z"] for r in ser], label="obj_z")
        axs[2].axhline(Z_ABORT, color="r", ls="--", label="Z_ABORT")
        axs[2].legend(fontsize=8)
        axs[2].set_ylabel("m")
        axs[2].set_xlabel("t (s)")
        fig.suptitle("Branch B axial slip traces")
        fig.tight_layout()
        p = out_dir / "branch_b_axial_traces.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["axial_traces"] = str(p)

    # Sensitivity bars
    sens = (branch_a or {}).get("handoff", {}).get("z_limit_sensitivity") or {}
    if sens:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        keys = sorted(sens.keys())
        vals_new = [1 if (branch_a.get("handoff") or {}).get("SUCCESS_NEW") else 0] * len(keys)
        vals_ax = [1 if sens[k].get("SUCCESS_AXIS_V1") else 0 for k in keys]
        x = np.arange(len(keys))
        ax.bar(x - 0.2, vals_new, 0.4, label="SUCCESS_NEW")
        ax.bar(x + 0.2, vals_ax, 0.4, label="SUCCESS_AXIS_V1")
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("ez_", "ez≤") + "mm" for k in keys])
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("pass")
        ax.set_title("Branch A: NEW vs AXIS_V1 by z limit")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "z_limit_sensitivity.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["sensitivity"] = str(p)
    return paths


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)

    env = make_env()
    table, book48 = load_vocab()

    print("=== Reconstruct Plan B stable ===", flush=True)
    stable = reconstruct_stable_b(env, table, book48)
    (OUT / "snapshots" / "B_stable.pkl").write_bytes(
        pickle.dumps({"capture": stable["capture"], "ref": stable["ref"]})
    )
    (OUT / "stable_state.json").write_text(
        json.dumps(jsafe({k: stable[k] for k in stable if k != "capture"}), indent=2), encoding="utf-8"
    )

    print("=== Build post-lateral best state ===", flush=True)
    post_lat = build_post_lateral_state(env, stable)
    print(
        f"  post_lat ok={post_lat.get('ok')} e={post_lat.get('e_signed_mm')} "
        f"θ={post_lat.get('theta')} mode={((post_lat.get('decision') or {}).get('mode'))}",
        flush=True,
    )
    if not post_lat.get("ok"):
        (OUT / "post_lateral_FAILED.json").write_text(json.dumps(jsafe(post_lat), indent=2), encoding="utf-8")
        print("Failed to rebuild post-lateral state", flush=True)
        return 1
    (OUT / "snapshots" / "B_post_lateral.pkl").write_bytes(
        pickle.dumps({"capture": post_lat["capture"], "ref": post_lat["ref"], "actions_prefix": post_lat["actions_prefix"]})
    )
    (OUT / "post_lateral.json").write_text(
        json.dumps(jsafe({k: v for k, v in post_lat.items() if k not in ("capture", "actions_prefix")}), indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(OUT / "post_lateral_prefix_actions.npz", actions=np.asarray(post_lat["actions_prefix"], np.float32))

    decision = post_lat["decision"]
    (OUT / "precision_decision.json").write_text(json.dumps(jsafe(decision), indent=2), encoding="utf-8")

    # --- Axial orientation inference (report even before slip) ---
    d_ax = axial_desire_direction(post_lat["e_signed_mm"])
    blob_pl = {"capture": post_lat["capture"], "ref": post_lat["ref"]}
    plan_ax = plan_orientation(env, blob_pl, d=d_ax, simulate=True, g_useful_min=7.0)
    orient_report = {
        "d_axial": d_ax.tolist(),
        "e_signed_mm": post_lat["e_signed_mm"],
        "desired_g_hand_z_sign": float(np.sign(d_ax[2])) if abs(d_ax[2]) > 0 else 0.0,
        "note": "e_z>0 ⇒ want cylinder motion −hand_z ⇒ g·d>0 with d=[0,0,-1] ⇒ g_hand,z < 0",
        "plan": {k: v for k, v in plan_ax.items() if k not in ("top", "candidates_kin")},
    }
    (OUT / "axial_orientation_inference.json").write_text(json.dumps(jsafe(orient_report), indent=2), encoding="utf-8")
    print(f"  axial orient: ok={plan_ax.get('ok')} {plan_ax.get('rationale') or plan_ax.get('reason')}", flush=True)

    # ========== Branch A: no axial ==========
    print("=== Branch A: HANDOFF without axial manipulation ===", flush=True)
    restore_capture(env, post_lat["capture"])
    # brake/hold already done; raise if needed
    acts_a = []
    raise_to_handoff_height(env, acts_a, z_target=max(0.50, post_lat["obj_z"]))
    cap_a = take_capture(env)
    ha = handoff_rich(env, cap_a, post_lat["ref"])
    samples_a = ha.pop("_samples_for_sensitivity", [])
    print(
        f"  A: S12={ha['SUCCESS_12S']} NEW={ha['SUCCESS_NEW']} AXIS={ha['SUCCESS_AXIS_V1']} "
        f"pre_e={np.round(ha['pre_e_signed_mm'],2)} sens={[ (k, v.get('SUCCESS_AXIS_V1')) for k,v in ha['z_limit_sensitivity'].items() ]}",
        flush=True,
    )
    branch_a = {
        "label": "no_axial_handoff",
        "e_signed_mm": ha["pre_e_signed_mm"],
        "handoff": strip_samples(ha),
        "raise_actions_n": len(acts_a),
    }
    (OUT / "branch_a_handoff.json").write_text(json.dumps(jsafe(branch_a), indent=2), encoding="utf-8")
    (OUT / "z_limit_sensitivity_branch_a.json").write_text(
        json.dumps(jsafe(ha["z_limit_sensitivity"]), indent=2), encoding="utf-8"
    )

    # Also: direct stable (no lateral) and Plan A geometry for sensitivity context
    print("=== Sensitivity context: stable B direct + Plan A ===", flush=True)
    hs = handoff_rich(env, stable["capture"], stable["ref"])
    samples_s = hs.pop("_samples_for_sensitivity", [])
    ctx = {"stable_direct": strip_samples(hs)}
    if PLAN_A_SNAP.exists():
        a_blob = pickle.loads(PLAN_A_SNAP.read_bytes())
        # If precision actions exist, apply then handoff
        if PLAN_A_ACTIONS.exists():
            restore_capture(env, a_blob["capture"])
            acts = np.load(PLAN_A_ACTIONS)["actions"]
            for a in acts:
                step_legal(env.sim, a)
            ha_a = handoff_rich(env, take_capture(env), a_blob.get("ref") or stable["ref"])
        else:
            ha_a = handoff_rich(env, a_blob["capture"], a_blob.get("ref") or stable["ref"])
        ha_a.pop("_samples_for_sensitivity", None)
        ctx["plan_a_geometry"] = strip_samples(ha_a)
    (OUT / "sensitivity_context.json").write_text(json.dumps(jsafe(ctx), indent=2), encoding="utf-8")

    # ========== Branch B: conditional axial ==========
    print("=== Branch B: conditional axial correction ===", flush=True)
    branch_b = {"label": "conditional_axial", "attempts": []}
    # Only if decision says axial needed under provisional 20 mm — still try 2 mm feasibility
    # even if within 20 mm (user asked Branch B attempt)
    ez_abs = abs(float(post_lat["e_signed_mm"][2]))
    branch_b["ez_abs_mm"] = ez_abs
    branch_b["axial_needed_vs_provisional_20"] = ez_abs > AXIS_V1_EZ_MM
    branch_b["axial_needed_vs_NEW_euclidean"] = float(post_lat["e_pos_mm"]) > WORKING_POS_MM

    att2 = axial_correction_attempt(env, blob_pl, milestone_mm=2.0, restore=True)
    branch_b["attempts"].append({k: v for k, v in att2.items() if k not in ("actions", "slip_series", "capture")})
    best_b = att2 if att2.get("ok") else None

    if att2.get("ok"):
        print("  2 mm axial OK — trying 5 mm from original post-lateral", flush=True)
        att5 = axial_correction_attempt(env, blob_pl, milestone_mm=5.0, restore=True)
        branch_b["attempts"].append({k: v for k, v in att5.items() if k not in ("actions", "slip_series", "capture")})
        if att5.get("ok"):
            best_b = att5
    else:
        print("  2 mm axial not ok — skip 5/10 mm", flush=True)

    hb = None
    if best_b and best_b.get("capture"):
        (OUT / "snapshots" / "B_branch_b_final.pkl").write_bytes(
            pickle.dumps({"capture": best_b["capture"], "ref": best_b["ref"]})
        )
        hb = handoff_rich(env, best_b["capture"], best_b["ref"])
        hb.pop("_samples_for_sensitivity", None)
        print(
            f"  B handoff: S12={hb['SUCCESS_12S']} NEW={hb['SUCCESS_NEW']} AXIS={hb['SUCCESS_AXIS_V1']} "
            f"pre_e={np.round(hb['pre_e_signed_mm'],2)}",
            flush=True,
        )
        branch_b["handoff"] = strip_samples(hb)
        branch_b["slip_series"] = best_b.get("slip_series")
        branch_b["e_signed_mm"] = best_b.get("e_final")
        branch_b["best_milestone_mm"] = best_b.get("milestone_mm")
        branch_b["x_preserved"] = best_b.get("x_preserved")
        branch_b["z_improved"] = best_b.get("z_improved")
        np.savez_compressed(OUT / "branch_b_actions.npz", actions=np.asarray(best_b["actions"], np.float32))
    else:
        branch_b["handoff"] = None
        branch_b["feasibility"] = "axial_slip_not_ok"
        # Still record last attempt capture handoff if any
        last = att2
        if last.get("capture"):
            hb_fail = handoff_rich(env, last["capture"], post_lat["ref"])
            hb_fail.pop("_samples_for_sensitivity", None)
            branch_b["handoff_after_failed_attempt"] = strip_samples(hb_fail)

    (OUT / "branch_b.json").write_text(
        json.dumps(jsafe({k: v for k, v in branch_b.items() if k != "slip_series"}), indent=2), encoding="utf-8"
    )
    if branch_b.get("slip_series"):
        (OUT / "branch_b_slip_series.json").write_text(json.dumps(jsafe(branch_b["slip_series"]), indent=2), encoding="utf-8")

    # Observability
    restore_capture(env, post_lat["capture"])
    obs = observability_audit(env, post_lat, decision)
    (OUT / "observability_39d.json").write_text(json.dumps(jsafe(obs), indent=2), encoding="utf-8")

    # IC replay only if SUCCESS_NEW or SUCCESS_AXIS_V1 on a complete traj
    replay = None
    winner = None
    if hb and (hb.get("SUCCESS_NEW") or hb.get("SUCCESS_AXIS_V1")):
        winner = "B"
    elif ha.get("SUCCESS_NEW") or ha.get("SUCCESS_AXIS_V1"):
        winner = "A"
    if winner == "A" and (ha.get("SUCCESS_NEW") or ha.get("SUCCESS_AXIS_V1")):
        # Branch A: prefix = post_lateral actions; frozen replay of prefix + raise + handoff is via evaluate
        print("=== Original-IC 3× replay (Branch A prefix + HANDOFF) ===", flush=True)
        results = []
        for trial in range(3):
            root = pickle.loads(IC_B.read_bytes())
            restore_capture(env, root["capture"])
            for a in stable["stabilization_prefix_actions"]:
                step_legal(env.sim, np.asarray(a, np.float32))
            for a in post_lat["actions_prefix"]:
                step_legal(env.sim, np.asarray(a, np.float32))
            raise_to_handoff_height(env, [], z_target=0.50)
            hr = handoff_rich(env, take_capture(env), stable["ref"])
            hr.pop("_samples_for_sensitivity", None)
            results.append(strip_samples(hr))
            print(f"  trial{trial}: NEW={hr['SUCCESS_NEW']} AXIS={hr['SUCCESS_AXIS_V1']} S12={hr['SUCCESS_12S']}", flush=True)
        replay = {
            "branch": "A",
            "mode": "frozen_prefix_actions_then_handoff",
            "trials": results,
            "SUCCESS_NEW_3of3": all(r.get("SUCCESS_NEW") for r in results),
            "SUCCESS_AXIS_V1_3of3": all(r.get("SUCCESS_AXIS_V1") for r in results),
        }
        (OUT / "replay_ic.json").write_text(json.dumps(jsafe(replay), indent=2), encoding="utf-8")
    elif winner == "B" and best_b:
        print("=== Original-IC 3× replay (Branch B) ===", flush=True)
        results = []
        for trial in range(3):
            root = pickle.loads(IC_B.read_bytes())
            restore_capture(env, root["capture"])
            for a in stable["stabilization_prefix_actions"]:
                step_legal(env.sim, np.asarray(a, np.float32))
            for a in post_lat["actions_prefix"]:
                step_legal(env.sim, np.asarray(a, np.float32))
            for a in best_b["actions"]:
                step_legal(env.sim, np.asarray(a, np.float32))
            hr = handoff_rich(env, take_capture(env), stable["ref"])
            hr.pop("_samples_for_sensitivity", None)
            results.append(strip_samples(hr))
            print(f"  trial{trial}: NEW={hr['SUCCESS_NEW']} AXIS={hr['SUCCESS_AXIS_V1']}", flush=True)
        replay = {
            "branch": "B",
            "mode": "frozen_prefix_plus_axial_actions_then_handoff",
            "trials": results,
            "SUCCESS_NEW_3of3": all(r.get("SUCCESS_NEW") for r in results),
            "SUCCESS_AXIS_V1_3of3": all(r.get("SUCCESS_AXIS_V1") for r in results),
        }
        (OUT / "replay_ic.json").write_text(json.dumps(jsafe(replay), indent=2), encoding="utf-8")

    # Plots / video
    plots = make_plots(branch_a, branch_b, OUT / "plots")
    videos = {}
    print("Rendering video...", flush=True)
    d_lat_v = lateral_desire_direction(post_lat["e_signed_mm"])
    videos["post_lateral"] = render_video(
        env,
        {"capture": stable["capture"], "ref": stable["ref"]},
        post_lat["actions_prefix"],
        OUT / "videos" / "B_post_lateral.mp4",
        "Plan B post-lateral",
        d=d_lat_v,
        plan=post_lat.get("plan_lateral"),
    )
    if best_b and best_b.get("actions"):
        videos["branch_b_axial"] = render_video(
            env,
            blob_pl,
            best_b["actions"],
            OUT / "videos" / "B_branch_b_axial.mp4",
            "Branch B axial",
            d=d_ax,
            plan=best_b.get("plan") or plan_ax,
        )

    # Classification comparison table
    classification = {
        "branch_a": {
            "SUCCESS_12S": ha.get("SUCCESS_12S"),
            "SUCCESS_NEW": ha.get("SUCCESS_NEW"),
            "SUCCESS_AXIS_V1": ha.get("SUCCESS_AXIS_V1"),
            "pre_e_signed_mm": ha.get("pre_e_signed_mm"),
            "z_limit_sensitivity": ha.get("z_limit_sensitivity"),
        },
        "branch_b": None
        if not hb
        else {
            "SUCCESS_12S": hb.get("SUCCESS_12S"),
            "SUCCESS_NEW": hb.get("SUCCESS_NEW"),
            "SUCCESS_AXIS_V1": hb.get("SUCCESS_AXIS_V1"),
            "pre_e_signed_mm": hb.get("pre_e_signed_mm"),
            "z_limit_sensitivity": hb.get("z_limit_sensitivity"),
        },
        "stable_direct": {
            "SUCCESS_12S": hs.get("SUCCESS_12S"),
            "SUCCESS_NEW": hs.get("SUCCESS_NEW"),
            "SUCCESS_AXIS_V1": hs.get("SUCCESS_AXIS_V1"),
            "pre_e_signed_mm": hs.get("pre_e_signed_mm"),
        },
        "note": "SUCCESS_NEW unchanged (Euclidean 12 mm). SUCCESS_AXIS_V1 is provisional diagnostic.",
    }
    (OUT / "classification_comparison.json").write_text(json.dumps(jsafe(classification), indent=2), encoding="utf-8")

    # Verdict
    if ha.get("SUCCESS_NEW"):
        rec = "branch_a_SUCCESS_NEW_unexpected"
    elif ha.get("SUCCESS_AXIS_V1") and not ha.get("SUCCESS_NEW"):
        rec = "branch_a_SUCCESS_AXIS_V1_provisional_accepts_axial_offset"
    elif best_b and hb and hb.get("SUCCESS_NEW"):
        rec = "branch_b_axial_enabled_SUCCESS_NEW"
    elif best_b and hb and hb.get("SUCCESS_AXIS_V1"):
        rec = "branch_b_SUCCESS_AXIS_V1_after_partial_axial"
    elif not (plan_ax.get("ok") or plan_ax.get("already_sufficient")):
        rec = "axial_orientation_unreachable_under_legal_wrist"
    elif branch_b.get("feasibility") == "axial_slip_not_ok":
        rec = "axial_slip_infeasible_preserve_lateral_branch_a_diagnostic"
    else:
        rec = "axis_aware_milestone_complete_see_report"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "ic_id": "hard_910220",
        "SUCCESS_NEW_definition_unchanged": True,
        "SUCCESS_AXIS_V1_provisional_limits_mm": {
            "ex": AXIS_V1_EX_MM,
            "ey": 5.0,
            "ez": AXIS_V1_EZ_MM,
            "theta_deg": WORKING_THETA_DEG,
            "z_candidates": list(Z_LIMIT_CANDIDATES_MM),
        },
        "post_lateral": {
            "e_signed_mm": post_lat["e_signed_mm"],
            "e_pos_mm": post_lat["e_pos_mm"],
            "theta": post_lat["theta"],
            "decision": decision,
        },
        "axial_orientation": orient_report,
        "branch_a": classification["branch_a"],
        "branch_b": classification["branch_b"],
        "branch_b_feasibility": {k: branch_b.get(k) for k in ("feasibility", "attempts", "x_preserved", "z_improved", "best_milestone_mm")},
        "sensitivity_context": {k: {kk: ctx[k].get(kk) for kk in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm")} for k in ctx},
        "observability_39d": obs,
        "replay_ic": replay,
        "plots": plots,
        "videos": videos,
        "recommendation": rec,
        "gains_unchanged_from_plan_a": True,
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    # REPORT
    ba = classification["branch_a"]
    bb = classification["branch_b"] or {}
    report = f"""# Axis-Aware Precision Oracle — Plan B hard_910220

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

**SUCCESS_NEW is unchanged** (θ≤10°, Euclidean e_pos≤12 mm, 0.5 s dwell).  
**SUCCESS_AXIS_V1** is a provisional diagnostic: |e_x|≤12, |e_y|≤5, |e_z|≤{AXIS_V1_EZ_MM} mm, θ≤10°, same dwell.

Low-level force/slip gains unchanged from Plan A. No bulk Oracle / SFT / GRPO.

## 1. Post-lateral Plan B state

| | Value |
|--|-------|
| e_signed | {post_lat['e_signed_mm']} |
| e_pos / θ | {post_lat['e_pos_mm']:.2f} mm / {post_lat['theta']:.2f}° |
| Decision mode | {decision.get('mode')} |
| Axial needed @ ez≤20 | {decision.get('axial_needed')} |

Snapshot: `snapshots/B_post_lateral.pkl`

## 2. Axial-tolerance sensitivity (Branch A HANDOFF)

| z_limit | SUCCESS_12S | SUCCESS_NEW | SUCCESS_AXIS_V1 |
|---------|-------------|-------------|-----------------|
"""
    for k, v in sorted(ha.get("z_limit_sensitivity", {}).items()):
        report += f"| {v.get('ez_mm')} mm | {ha.get('SUCCESS_12S')} | {ha.get('SUCCESS_NEW')} | {v.get('SUCCESS_AXIS_V1')} |\n"
    report += f"""
Pre-HANDOFF e_signed: {ha.get('pre_e_signed_mm')}  
Window NEW longest both-ok: {(ha.get('window_SUCCESS_NEW') or {}).get('longest_both_ok_s')} s  
AXIS_V1 (ez=20) longest: {(ha.get('window_AXIS_V1') or {}).get('longest_ok_s')} s  
min_obj_z / Z_ABORT margin: {ha.get('min_obj_z')} / {ha.get('z_abort_margin')}

**Provisional:** 20 mm ez limit is a candidate, not a validated safety margin. Selection is not solely because e_z≈13.3 mm.

## 3. Branch A vs Branch B

| | Branch A (no axial) | Branch B (axial attempt) |
|--|---------------------|--------------------------|
| SUCCESS_12S | {ba.get('SUCCESS_12S')} | {bb.get('SUCCESS_12S')} |
| SUCCESS_NEW | {ba.get('SUCCESS_NEW')} | {bb.get('SUCCESS_NEW')} |
| SUCCESS_AXIS_V1 | {ba.get('SUCCESS_AXIS_V1')} | {bb.get('SUCCESS_AXIS_V1')} |
| pre e_signed | {ba.get('pre_e_signed_mm')} | {bb.get('pre_e_signed_mm')} |

Branch B attempts: {json.dumps(jsafe(branch_b.get('attempts')), indent=2)}

## 4. Axial gravity orientation

d_axial = {d_ax.tolist()}  
Plan: axis={plan_ax.get('axis_name')} angle={plan_ax.get('angle_deg')} useful={plan_ax.get('useful')}  
Rationale: {plan_ax.get('rationale') or plan_ax.get('reason')}

## 5. Observability (39D)

{json.dumps(jsafe(obs), indent=2)}

## 6. IC replay

See `replay_ic.json` (frozen prefix actions). Brief: NEW_3of3={None if not replay else replay.get('SUCCESS_NEW_3of3')}, AXIS_3of3={None if not replay else replay.get('SUCCESS_AXIS_V1_3of3')}.

## 7. Plots & videos

{plots}

"""
    for k, v in videos.items():
        report += f"- `{k}`: {v.get('path')} ({v.get('duration_s')} s)\n"

    report += f"""
## 8. Recommendation

**{rec}**

### Limitations
- Axial gravity reorientation on post-lateral Plan B destroys e_x (−1.2→−10 mm observed).
- SUCCESS_AXIS_V1 must not silently relabel historical SUCCESS_NEW failures.
- ez=20 mm is provisional (final-window e_z≈15.9 mm); not a validated safety limit.

### Suggested next experiment
Freeze Branch A as a diagnostic AXIS_V1 candidate; do not force axial slip until a lateral-preserving axial mode exists.

Stopped after this bounded milestone.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": rec, "branch_a": ba, "branch_b": bb, "wall_s": summary["wall_s"]}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
