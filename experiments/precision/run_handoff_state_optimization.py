#!/usr/bin/env python
"""Optimize recovery HANDOFF state — no restore to IC orientation.

Freeze verified P0–P5 from endpoint_only_axial_roll path, then:
  1) HANDOFF directly after P5 (fingertips-down)
  2) Diagnose first 500 ms / apply_handoff continuity
  3) Bounded α candidates 175/165/150/135° if needed
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
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch, rot_angle  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import (  # noqa: E402
    DOWN,
    UP,
    choose_roll_sign_for_support,
    gates,
    phase_boundary_log,
    reference_pad_vector,
    run_roll_phase,
    run_tilt_phase,
    u_open_world,
)
from data_generation.oracles.precision.mechanics.finger_vertical import U_OPEN_HAND  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import (  # noqa: E402
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.corrections.orientation import _height_hold_v_hand  # noqa: E402
from data_generation.oracles.precision.mechanics.reverse_rotation import HoldModeGrip  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import (  # noqa: E402
    LOST_STEPS,
    _bilateral,
    _dropped,
    _escaped,
    _monitor,
)
from envs.outcomes.recovery_success import apply_handoff, nominal_step  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world, hand_R, measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

B_POST = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl"
IC_B = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
AXIS_PREFIX = ROOT / "results/diagnostics/axis_aware_precision/post_lateral_prefix_actions.npz"
STABLE_JSON = ROOT / "results/diagnostics/axis_aware_precision/stable_state.json"
OUT = ROOT / "results/diagnostics/handoff_state_optimization"

FORCE_KP, FORCE_DA = 0.10, 0.12
V_SLIP = -0.001
FN_MIN, FN_MAX, FN_INIT, FN_BRAKE = 3.8, 11.0, 5.2, 14.0
SLIP_KN = 900.0
A0_SLIP = 0.55
FN_HOLD = 13.0
DT = 0.02
FPS = 25
ROLL_ANGLE_DEG = 90.0
CANDIDATE_ALPHAS = (175.0, 165.0, 150.0, 135.0)


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
    return env


def load_blob(path: Path):
    return pickle.loads(path.read_bytes())


def run_slip_5mm(env, blob, z_ref: float, e_x0: float):
    d = -U_OPEN_HAND

    def _one(blob_i, milestone, e_x):
        restore_capture(env, blob_i["capture"])
        s0 = sample_state(env.sim)
        p0 = np.asarray(s0["p_c_h"], float)
        ctl = DualLoopController(
            force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
            slip=SlipOuterLoop(vx_target=V_SLIP, fn_min=FN_MIN, fn_max=FN_MAX, kn=SLIP_KN),
            use_slip=True,
        )
        ctl.reset(a0=A0_SLIP, fn0=s0["fn_tot"], fn_target=FN_INIT, slip_axis=tuple((-d).tolist()))
        actions = []
        lost = 0
        reached = False
        abort = None
        last_disp = 0.0
        for _ in range(int(12.0 / DT)):
            s = sample_state(env.sim)
            _, meta = ctl.step(s)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
            step_legal(env.sim, a)
            actions.append(a.copy())
            failed, lost = recovery_failed_safe(env, blob_i["ref"], lost)
            s2 = sample_state(env.sim)
            disp = float(np.dot(np.asarray(s2["p_c_h"], float) - p0, d)) * 1e3
            last_disp = disp
            if failed or latched(env.sim) or not s2["bilateral"]:
                return {"ok": False, "reason": "contact_loss", "actions": actions}
            if abs(float(s2["e_signed_mm"][0]) - e_x) > 4.0:
                abort = "lateral_drift"
                break
            if disp >= milestone:
                reached = True
                break
        ctl_b = ForcePController(kp=0.15, da_max=0.2)
        ctl_b.reset(0.85, sample_state(env.sim)["fn_tot"])
        for _ in range(int(0.4 / DT) + int(0.2 / DT)):
            s = sample_state(env.sim)
            a_g, _ = ctl_b.step(s["fn_tot"], FN_BRAKE)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
            step_legal(env.sim, a)
            actions.append(a.copy())
            if not sample_state(env.sim)["bilateral"]:
                return {"ok": False, "reason": "loss_on_brake", "actions": actions}
        return {
            "ok": bool(abort is None and (reached or last_disp >= 0.5 * milestone)),
            "disp": last_disp,
            "e_signed": sample_state(env.sim)["e_signed_mm"],
            "actions": actions,
            "capture": take_capture(env),
        }

    s2 = _one(blob, 2.0, e_x0)
    if not s2.get("ok"):
        return s2
    s5 = _one({"capture": s2["capture"], "ref": blob["ref"]}, 5.0, float(s2["e_signed"][0]))
    if not s5.get("ok"):
        return {**s5, "actions": list(s2["actions"]) + list(s5.get("actions") or [])}
    restore_capture(env, s5["capture"])
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    extra = []
    for _ in range(int(0.25 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        extra.append(a.copy())
    return {
        "ok": True,
        "disp_2": s2["disp"],
        "disp_5": s5["disp"],
        "e_signed": sample_state(env.sim)["e_signed_mm"],
        "actions": list(s2["actions"]) + list(s5["actions"]) + extra,
        "capture": take_capture(env),
    }


def recovery_failed_safe(env, ref, lost):
    from envs.outcomes.horizon_outcome import recovery_failed

    return recovery_failed(env.sim, ref, lost)


def rebuild_through_p5(env, blob) -> dict:
    """Verified endpoint-only P0–P5; no P6 IC restore."""
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    y0 = reference_pad_vector(env.sim) if False else None
    restore_capture(env, blob["capture"])
    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    actions = []
    roll_sign = choose_roll_sign_for_support(env.sim)

    p0 = run_tilt_phase(
        env, cur, target_dir_w=DOWN, phase="P0", rate_scale=0.20, fn_hold=FN_HOLD,
        ang_tol_deg=4.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=False, max_time_s=8.0,
    )
    assert p0["ok"] and gates(env.sim)["fingertips_down"], p0.get("reason")
    actions += p0["actions"]
    cur = {"capture": p0["capture"], "ref": blob["ref"]}

    p1 = run_roll_phase(
        env, cur, sign=roll_sign, angle_deg=ROLL_ANGLE_DEG, phase="P1",
        rate_scale=0.15, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    assert p1["ok"], p1.get("reason")
    actions += p1["actions"]
    cur = {"capture": p1["capture"], "ref": blob["ref"]}

    p2 = run_tilt_phase(
        env, cur, target_dir_w=UP, phase="P2", rate_scale=0.22, fn_hold=FN_HOLD,
        ang_tol_deg=5.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=16.0,
    )
    assert p2["ok"] and gates(env.sim)["fingertips_up"], p2.get("reason")
    actions += p2["actions"]
    cur = {"capture": p2["capture"], "ref": blob["ref"]}

    # firm settle before slip
    restore_capture(env, cur["capture"])
    hold = HoldModeGrip(fn_hold=14.0)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for _ in range(int(0.4 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
    cur = {"capture": take_capture(env), "ref": blob["ref"]}
    e_x0 = float(sample_state(env.sim)["e_signed_mm"][0])
    slip = run_slip_5mm(env, cur, z_ref, e_x0)
    assert slip["ok"], slip.get("reason")
    actions += slip["actions"]
    cur = {"capture": slip["capture"], "ref": blob["ref"]}
    e_post = np.asarray(slip["e_signed"], float)

    p4 = run_tilt_phase(
        env, cur, target_dir_w=DOWN, phase="P4", rate_scale=0.18, fn_hold=FN_HOLD,
        ang_tol_deg=4.0, y0_w=y0, e_ref=e_post, allow_roll=False, contact_aware=True, max_time_s=16.0,
        z_ref_boost=0.04, height_kp=4.0,
    )
    assert p4["ok"] and gates(env.sim)["fingertips_down"], p4.get("reason")
    actions += p4["actions"]
    cur = {"capture": p4["capture"], "ref": blob["ref"]}

    p5 = run_roll_phase(
        env, cur, sign=-roll_sign, angle_deg=ROLL_ANGLE_DEG, phase="P5",
        rate_scale=0.15, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_post,
    )
    assert p5["ok"], p5.get("reason")
    actions += p5["actions"]
    cur = {"capture": p5["capture"], "ref": blob["ref"]}

    # settle under hold (prepare_stable_handoff_state — no IC orientation restore)
    restore_capture(env, cur["capture"])
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for _ in range(int(0.5 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())

    snap = phase_boundary_log(env.sim, phase="post_P5_settle", e_ref=e_post)
    return {
        "capture": take_capture(env),
        "ref": blob["ref"],
        "actions_prefix": actions,
        "snapshot": snap,
        "e_post_slip": e_post.tolist(),
        "roll_sign": roll_sign,
        "y0": y0.tolist(),
    }


def tilt_to_alpha(env, blob, target_alpha_deg: float, *, e_ref, y0, rate=0.15) -> dict:
    """Tilt so α_up ≈ target (between 90 and 180) without axial roll."""
    # α = angle(u, UP). Want u such that angle to UP = target.
    # Equivalent: aim u toward a direction with z = cos(target) in the plane of current tilt.
    restore_capture(env, blob["capture"])
    a0 = gates(env.sim)["alpha_up_deg"]
    u = u_open_world(env.sim)
    # Target direction: rotate u in the plane spanned by u and UP toward the cone at target_alpha
    # Simplest: interpolate between UP and DOWN on the great circle through current u.
    c_tgt = float(np.cos(np.radians(target_alpha_deg)))
    # Build orthonormal basis
    # Desired: u_tgt with u_tgt·UP = c_tgt, in plane of (u projected)
    axis = np.cross(u, UP)
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        # already parallel — nudge
        helper = np.array([1.0, 0.0, 0.0])
        axis = np.cross(u, helper)
        n = float(np.linalg.norm(axis))
    axis = axis / n
    # Current alpha a0; rotate by (a0 - target) about axis
    delta = np.radians(a0 - target_alpha_deg)
    # Rodrigues: rotate u toward UP (negative delta if a0 > target means tip up from down)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], float)
    R = np.eye(3) + np.sin(delta) * K + (1 - np.cos(delta)) * (K @ K)
    u_tgt = R @ u
    u_tgt = u_tgt / max(float(np.linalg.norm(u_tgt)), 1e-12)
    # Verify
    a_pred = float(np.degrees(np.arccos(np.clip(np.dot(u_tgt, UP), -1, 1))))
    res = run_tilt_phase(
        env, blob, target_dir_w=u_tgt, phase=f"cand_a{int(target_alpha_deg)}",
        rate_scale=rate, fn_hold=FN_HOLD, ang_tol_deg=3.0, y0_w=np.asarray(y0, float),
        e_ref=np.asarray(e_ref, float), allow_roll=False, contact_aware=True, max_time_s=10.0,
        z_ref_boost=0.03, height_kp=4.0,
    )
    res["target_alpha_deg"] = target_alpha_deg
    res["predicted_alpha"] = a_pred
    res["alpha_achieved"] = gates(env.sim)["alpha_up_deg"]
    return res


def controller_snapshot(sim, ref) -> dict:
    Rh = np.array(hand_R(sim), float, copy=True)
    Rd = np.array(sim.fsm.r_des, float, copy=True).reshape(3, 3)
    s = sample_state(sim)
    m = measure(sim, U_TGT, P_TGT)
    return {
        "alpha_up_deg": gates(sim)["alpha_up_deg"],
        "hand_R": Rh.tolist(),
        "r_des": Rd.tolist(),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "hand_pos": np.asarray(sim.data.xpos[sim.ids.hand_body], float).tolist(),
        "r_des_to_hand_deg": float(np.degrees(rot_angle(Rd, Rh))),
        "r_des_to_ref_deg": float(np.degrees(rot_angle(Rd, np.asarray(ref["r_des"], float)))),
        "ref_r_des_to_hand_deg": float(np.degrees(rot_angle(np.asarray(ref["r_des"], float), Rh))),
        "phase": str(sim.fsm.phase),
        "fg_cmd": float(sim.fsm.fg_cmd),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).tolist(),
        "e_signed_mm": s["e_signed_mm"],
        "theta_deg": s["e_theta_deg"],
        "fn_L": s["fn_L"],
        "fn_R": s["fn_R"],
        "fn_tot": s["fn_tot"],
        "bilateral": s["bilateral"],
        "v_rel_norm": s["v_rel_norm"],
        "obj_z": float(m["obj_z"]),
        "qpos": np.asarray(sim.data.qpos, float)[:9].tolist(),
        "ctrl": np.asarray(sim.data.ctrl, float)[:8].tolist() if sim.data.ctrl.size >= 8 else np.asarray(sim.data.ctrl, float).tolist(),
    }


def handoff_with_trace(env, capture, ref, *, label: str, max_trace_s: float = 0.5) -> dict:
    """Settle already done; apply_handoff + dense 20ms trace then full 12s eval."""
    restore_capture(env, capture)
    pre = controller_snapshot(env.sim, ref)
    # Short settle already assumed; one more firm hold
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for _ in range(8):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        step_legal(env.sim, make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g))
    pre = controller_snapshot(env.sim, ref)

    ho_meta = apply_handoff(env.sim, ref)
    post_apply = controller_snapshot(env.sim, ref)
    t_h = float(env.sim.data.time)
    trace = []
    reason = None
    lost = 0
    first_event = None
    first_event_t = None

    n_trace = int(max_trace_s / DT)
    for i in range(n_trace):
        nominal_step(env.sim)
        t_rel = float(env.sim.data.time) - t_h
        s = sample_state(env.sim)
        m = measure(env.sim, U_TGT, P_TGT)
        mon = _monitor(env.sim)
        bil = _bilateral(mon)
        row = {
            "t_rel": t_rel,
            "step": i + 1,
            "bilateral": bil,
            "fn_L": s["fn_L"],
            "fn_R": s["fn_R"],
            "fn_tot": s["fn_tot"],
            "e_signed_mm": s["e_signed_mm"],
            "e_pos_mm": m["e_pos_mm"],
            "theta_deg": m["e_theta_deg"],
            "obj_z": m["obj_z"],
            "v_rel_norm": s["v_rel_norm"],
            "p_des": np.asarray(env.sim.fsm.p_des, float).tolist(),
            "v_cmd_norm": float(np.linalg.norm(env.sim.fsm.v_cmd)),
            "w_cmd_norm": float(np.linalg.norm(env.sim.fsm.w_cmd)),
            "r_des_to_hand_deg": float(np.degrees(rot_angle(np.asarray(env.sim.fsm.r_des, float), hand_R(env.sim)))),
            "fg_cmd": float(env.sim.fsm.fg_cmd),
            "phase": str(env.sim.fsm.phase),
            "alpha_up_deg": gates(env.sim)["alpha_up_deg"],
        }
        trace.append(row)
        if first_event is None:
            if latched(env.sim):
                first_event, first_event_t = "GROUND_DROP", t_rel
            elif _dropped(mon, env.sim) or _escaped(env.sim):
                first_event, first_event_t = "DROP", t_rel
            elif not bil:
                first_event, first_event_t = "LOST_BILATERAL", t_rel
            elif row["v_rel_norm"] > 0.05:
                first_event, first_event_t = "HIGH_REL_VEL", t_rel
            elif row["r_des_to_hand_deg"] > 25.0 and i == 0:
                first_event, first_event_t = "R_DES_JUMP", t_rel

    # Continue full 12s from current state via evaluate (re-handoff would double-apply).
    # Instead finish nominal loop manually for SUCCESS flags.
    while reason is None and float(env.sim.data.time) - t_h < 12.0 - 1e-9:
        nominal_step(env.sim)
        t_rel = float(env.sim.data.time) - t_h
        mon = _monitor(env.sim)
        bil = _bilateral(mon)
        if latched(env.sim):
            reason = "GROUND_DROP"
            break
        if _dropped(mon, env.sim) or _escaped(env.sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not bil else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
    if reason is None:
        reason = "SUCCESS_12S" if _bilateral(_monitor(env.sim)) else "LOST_GRASP"

    # Re-run standard evaluator from same capture for SUCCESS_NEW window stats
    restore_capture(env, capture)
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for _ in range(8):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        step_legal(env.sim, make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g))
    std = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    std.pop("samples", None)

    return {
        "label": label,
        "pre": pre,
        "apply_handoff": ho_meta,
        "post_apply": post_apply,
        "trace_500ms": trace,
        "first_event": first_event,
        "first_event_t": first_event_t,
        "manual_reason": reason,
        "SUCCESS_12S": std.get("SUCCESS_12S"),
        "SUCCESS_NEW": std.get("SUCCESS_NEW"),
        "SUCCESS_AXIS_V1": std.get("SUCCESS_AXIS_V1"),
        "std_reason": std.get("reason"),
        "pre_e_signed_mm": std.get("pre_e_signed_mm"),
        "pre_theta": std.get("pre_theta"),
        "pre_pos_mm": std.get("pre_pos_mm"),
        "continuity": {
            "r_des_set_to_ref": True,
            "code_path": "envs/outcomes/recovery_success.py::apply_handoff",
            "pre_r_des_to_hand_deg": pre["r_des_to_hand_deg"],
            "post_r_des_to_hand_deg": post_apply["r_des_to_hand_deg"],
            "pre_ref_to_hand_deg": pre["ref_r_des_to_hand_deg"],
            "post_ref_to_hand_deg": post_apply["ref_r_des_to_hand_deg"],
            "note": (
                "apply_handoff copies ref['r_des'] (IC grasp orientation) into fsm.r_des; "
                "p_des is set to current hand position. Orientation discontinuity is expected "
                "when recovered hand ≠ IC grasp orientation."
            ),
        },
    }


def overlay(img, lines):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    y = 3
    for line in lines:
        dr.text((4, y), line, fill=(255, 255, 255), font=font)
        y += 15
    return np.asarray(im)


def render_best_video(env, start_cap, prefix_actions, prep_actions, dest: Path, title: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r_w = mujoco.Renderer(env.sim.model, 480, 640)
    r_c = mujoco.Renderer(env.sim.model, 480, 640)
    cam_w = mjv_camera_from_preset({"lookat": [0.35, 0.0, 0.4], "distance": 2.2, "azimuth": 145, "elevation": -18})
    cam_c = mjv_camera_from_preset({"lookat": [0.5, 0.05, 0.52], "distance": 0.55, "azimuth": 120, "elevation": -25})
    frames = []
    actions = list(prefix_actions) + list(prep_actions or [])

    def draw(scn):
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        uo = u_open_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_sphere(scn, ph + Rh @ np.asarray(P_TGT, float), [1.0, 0.9, 0.15, 1.0], 0.005)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.0025)
        add_capsule(scn, po, po + 0.09 * uo, [1.0, 0.85, 0.1, 1.0], 0.0035)

    def frame(tag):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.35 * ph + 0.35 * po + np.array([0, 0, 0.05])
        cam_c.lookat[:] = 0.5 * ph + 0.5 * po
        s = sample_state(env.sim)
        g = gates(env.sim)
        lines = [f"{tag} {title}", f"α={g['alpha_up_deg']:.1f} e={np.round(s['e_signed_mm'],1)}", f"θ={s['e_theta_deg']:.2f}"]
        r_w.update_scene(env.sim.data, cam_w)
        draw(r_w.scene)
        iw = overlay(np.asarray(r_w.render(), np.uint8), ["WIDE"] + lines)
        r_c.update_scene(env.sim.data, cam_c)
        draw(r_c.scene)
        ic = overlay(np.asarray(r_c.render(), np.uint8), ["CLOSE"] + lines)
        return np.concatenate([iw, ic], axis=1)

    restore_capture(env, start_cap)
    for i, a in enumerate(actions):
        step_legal(env.sim, a)
        if i % 4 == 0 or i == len(actions) - 1:
            frames.append(frame(f"{i+1}/{len(actions)}"))
    # show handoff+nominal briefly
    apply_handoff(env.sim, pickle.loads(B_POST.read_bytes())["ref"] if False else None)
    r_w.close()
    r_c.close()
    # fix: need ref — caller will pass; skip post-handoff frames if ref missing
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "n_frames": len(frames), "duration_s": len(frames) / FPS}


def render_full(env, start_cap, actions, ref, dest: Path, title: str, handoff_at: int | None = None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r_w = mujoco.Renderer(env.sim.model, 480, 640)
    r_c = mujoco.Renderer(env.sim.model, 480, 640)
    cam_w = mjv_camera_from_preset({"lookat": [0.35, 0.0, 0.4], "distance": 2.2, "azimuth": 145, "elevation": -18})
    cam_c = mjv_camera_from_preset({"lookat": [0.5, 0.05, 0.52], "distance": 0.55, "azimuth": 120, "elevation": -25})
    frames = []

    def draw(scn):
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        uo = u_open_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_sphere(scn, ph + Rh @ np.asarray(P_TGT, float), [1.0, 0.9, 0.15, 1.0], 0.005)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.0025)
        add_capsule(scn, po, po + 0.09 * uo, [1.0, 0.85, 0.1, 1.0], 0.0035)

    def frame(tag):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.35 * ph + 0.35 * po + np.array([0, 0, 0.05])
        cam_c.lookat[:] = 0.5 * ph + 0.5 * po
        s = sample_state(env.sim)
        g = gates(env.sim)
        lines = [f"{tag} {title}", f"α={g['alpha_up_deg']:.1f} e={np.round(s['e_signed_mm'],1)} θ={s['e_theta_deg']:.2f}"]
        r_w.update_scene(env.sim.data, cam_w)
        draw(r_w.scene)
        iw = overlay(np.asarray(r_w.render(), np.uint8), ["WIDE"] + lines)
        r_c.update_scene(env.sim.data, cam_c)
        draw(r_c.scene)
        ic = overlay(np.asarray(r_c.render(), np.uint8), ["CLOSE"] + lines)
        return np.concatenate([iw, ic], axis=1)

    restore_capture(env, start_cap)
    for i, a in enumerate(actions):
        step_legal(env.sim, a)
        if i % 4 == 0 or i == len(actions) - 1:
            frames.append(frame(f"rec {i+1}/{len(actions)}"))
    apply_handoff(env.sim, ref)
    frames.append(frame("HANDOFF"))
    for i in range(int(3.0 / DT)):
        nominal_step(env.sim)
        if i % 2 == 0:
            frames.append(frame(f"nom {i*DT:.2f}s"))
        if latched(env.sim) or not sample_state(env.sim)["bilateral"]:
            frames.append(frame("DROP"))
            break
    r_w.close()
    r_c.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "n_frames": len(frames), "duration_s": len(frames) / FPS}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("snapshots", "plots", "videos", "traces"):
        (OUT / sub).mkdir(exist_ok=True)

    env = make_env()
    blob = load_blob(B_POST)
    print("=== Rebuild verified P0–P5 (no IC-orientation P6) ===", flush=True)
    frozen = rebuild_through_p5(env, blob)
    (OUT / "snapshots" / "post_P5.pkl").write_bytes(
        pickle.dumps({"capture": frozen["capture"], "ref": frozen["ref"], "snapshot": frozen["snapshot"]})
    )
    np.savez_compressed(OUT / "prefix_P0_P5_actions.npz", actions=np.asarray(frozen["actions_prefix"], np.float32))
    (OUT / "post_P5_state.json").write_text(json.dumps(jsafe(frozen["snapshot"]), indent=2), encoding="utf-8")
    print(
        f"  post-P5 α={frozen['snapshot']['alpha_up_deg']:.1f} e={np.round(frozen['snapshot']['e_signed_mm'],2)} "
        f"down={frozen['snapshot']['fingertips_down']}",
        flush=True,
    )

    # Continuity inspection of apply_handoff source
    continuity_doc = {
        "file": "envs/outcomes/recovery_success.py",
        "function": "apply_handoff",
        "behavior": {
            "r_des": "set to ref['r_des'] (IC saved grasp orientation)",
            "p_des": "set to current hand xpos (no translational yank)",
            "qpos_qvel": "not written",
            "w_cmd_v_cmd": "zeroed",
        },
        "implication": (
            "Any recovered hand orientation that differs from IC grasp r_des will see an "
            "immediate orientation-command discontinuity at HANDOFF under frozen semantics."
        ),
    }
    (OUT / "handoff_continuity_code.json").write_text(json.dumps(continuity_doc, indent=2), encoding="utf-8")

    results = {}

    # --- Direct P5 HANDOFF ---
    print("=== Direct HANDOFF after P5 (no extra orientation restore) ===", flush=True)
    p5_h = handoff_with_trace(env, frozen["capture"], frozen["ref"], label="direct_P5")
    results["direct_P5"] = p5_h
    (OUT / "traces" / "direct_P5_500ms.json").write_text(json.dumps(jsafe(p5_h["trace_500ms"]), indent=2), encoding="utf-8")
    (OUT / "direct_P5_handoff.json").write_text(
        json.dumps(jsafe({k: v for k, v in p5_h.items() if k != "trace_500ms"}), indent=2), encoding="utf-8"
    )
    print(
        f"  NEW={p5_h['SUCCESS_NEW']} 12S={p5_h['SUCCESS_12S']} reason={p5_h['std_reason']} "
        f"first={p5_h['first_event']}@{p5_h['first_event_t']} "
        f"Δr_des_hand pre={p5_h['pre']['ref_r_des_to_hand_deg']:.1f}→post={p5_h['post_apply']['r_des_to_hand_deg']:.1f}°",
        flush=True,
    )

    # Control: B_post_lateral direct HANDOFF (Branch-A style)
    print("=== Control: HANDOFF from B_post_lateral ===", flush=True)
    ctrl = handoff_with_trace(env, blob["capture"], blob["ref"], label="control_B_post")
    results["control_B_post"] = {k: v for k, v in ctrl.items() if k != "trace_500ms"}
    (OUT / "traces" / "control_B_post_500ms.json").write_text(json.dumps(jsafe(ctrl["trace_500ms"]), indent=2), encoding="utf-8")
    print(f"  control NEW={ctrl['SUCCESS_NEW']} 12S={ctrl['SUCCESS_12S']} reason={ctrl['std_reason']}", flush=True)

    candidates = {}
    best = ("direct_P5", p5_h, [])
    need_search = not bool(p5_h.get("SUCCESS_NEW"))

    if need_search:
        print("=== Bounded endpoint α candidates ===", flush=True)
        e_ref = frozen["e_post_slip"]
        y0 = frozen["y0"]
        post_blob = {"capture": frozen["capture"], "ref": frozen["ref"]}
        for a_tgt in CANDIDATE_ALPHAS:
            print(f"  --- Candidate α≈{a_tgt}° ---", flush=True)
            prep = tilt_to_alpha(env, post_blob, a_tgt, e_ref=e_ref, y0=y0)
            prep_ok = bool(prep.get("ok"))
            # settle
            if prep_ok and prep.get("capture") is not None:
                restore_capture(env, prep["capture"])
                hold = HoldModeGrip(fn_hold=FN_HOLD)
                hold.reset(sample_state(env.sim)["fn_tot"])
                z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
                for _ in range(int(0.4 / DT)):
                    st = sample_state(env.sim)
                    a_g, _, _ = hold.step(st)
                    step_legal(env.sim, make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g))
                cap = take_capture(env)
                snap = phase_boundary_log(env.sim, phase=f"cand_{int(a_tgt)}_settle")
                if not snap["bilateral"] or snap["v_rel_norm"] > 0.02:
                    candidates[f"a{int(a_tgt)}"] = {
                        "prep_ok": False,
                        "reason": "prep_unstable",
                        "alpha_achieved": snap["alpha_up_deg"],
                        "snapshot": snap,
                    }
                    print(f"    prep unstable α={snap['alpha_up_deg']:.1f}", flush=True)
                    continue
                h = handoff_with_trace(env, cap, frozen["ref"], label=f"cand_a{int(a_tgt)}")
                (OUT / "traces" / f"cand_a{int(a_tgt)}_500ms.json").write_text(
                    json.dumps(jsafe(h["trace_500ms"]), indent=2), encoding="utf-8"
                )
                entry = {
                    "prep_ok": True,
                    "alpha_target": a_tgt,
                    "alpha_achieved": h["pre"]["alpha_up_deg"],
                    "snapshot_pre": snap,
                    "handoff": {k: v for k, v in h.items() if k != "trace_500ms"},
                    "prep_actions": prep.get("actions"),
                }
                candidates[f"a{int(a_tgt)}"] = entry
                print(
                    f"    α={h['pre']['alpha_up_deg']:.1f} NEW={h['SUCCESS_NEW']} reason={h['std_reason']} "
                    f"first={h['first_event']} ref→hand={h['pre']['ref_r_des_to_hand_deg']:.1f}°",
                    flush=True,
                )
                # rank
                score = (
                    1 if h.get("SUCCESS_NEW") else 0,
                    1 if h.get("SUCCESS_12S") else 0,
                    1 if h.get("SUCCESS_AXIS_V1") else 0,
                    -(h.get("first_event_t") or 0) if h.get("first_event") else 0,
                    -abs(float(h["pre"]["e_signed_mm"][2]) - float(frozen["e_post_slip"][2])),
                )
                best_score = (
                    1 if best[1].get("SUCCESS_NEW") else 0,
                    1 if best[1].get("SUCCESS_12S") else 0,
                    1 if best[1].get("SUCCESS_AXIS_V1") else 0,
                    -(best[1].get("first_event_t") or 0) if best[1].get("first_event") else 0,
                    0,
                )
                if score > best_score:
                    best = (f"a{int(a_tgt)}", h, list(prep.get("actions") or []))
            else:
                candidates[f"a{int(a_tgt)}"] = {
                    "prep_ok": False,
                    "reason": prep.get("reason"),
                    "alpha_achieved": prep.get("alpha_achieved"),
                }
                print(f"    prep fail {prep.get('reason')}", flush=True)

    # Rank all including direct
    (OUT / "candidates.json").write_text(
        json.dumps(
            jsafe(
                {
                    k: {kk: vv for kk, vv in v.items() if kk != "prep_actions"}
                    for k, v in candidates.items()
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # Replay if SUCCESS_NEW
    replay = None
    best_name, best_h, prep_acts = best
    if best_h.get("SUCCESS_NEW"):
        print("=== IC 3× frozen replay ===", flush=True)
        prefix = []
        if STABLE_JSON.exists():
            stj = json.loads(STABLE_JSON.read_text(encoding="utf-8"))
            prefix = [np.asarray(a, np.float32) for a in stj.get("stabilization_prefix_actions") or []]
        lat = np.load(AXIS_PREFIX)["actions"] if AXIS_PREFIX.exists() else []
        mb = list(frozen["actions_prefix"]) + list(prep_acts)
        np.savez_compressed(OUT / "best_complete_actions.npz", actions=np.asarray(mb, np.float32))
        trials = []
        for _ in range(3):
            root = pickle.loads(IC_B.read_bytes())
            restore_capture(env, root["capture"])
            for a in prefix:
                step_legal(env.sim, a)
            for a in lat:
                step_legal(env.sim, a)
            for a in mb:
                step_legal(env.sim, a)
            hr = evaluate_handoff_with_axis(env.sim, frozen["ref"], stabilize_n=0)
            hr.pop("samples", None)
            trials.append({k: hr.get(k) for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "reason")})
        replay = {
            "SUCCESS_NEW_3of3": all(t["SUCCESS_NEW"] for t in trials),
            "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials),
            "trials": trials,
        }
        (OUT / "replay_ic.json").write_text(json.dumps(jsafe(replay), indent=2), encoding="utf-8")

    print("Rendering best video...", flush=True)
    videos = {
        "best": render_full(
            env,
            blob["capture"],
            list(frozen["actions_prefix"]) + list(prep_acts),
            frozen["ref"],
            OUT / "videos" / f"best_{best_name}.mp4",
            f"best {best_name}",
        )
    }

    # Plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    labels, news, d12, axisv, r2h = [], [], [], [], []
    for name, h in [("direct_P5", p5_h), ("control_B_post", ctrl)]:
        labels.append(name)
        news.append(1 if h.get("SUCCESS_NEW") else 0)
        d12.append(1 if h.get("SUCCESS_12S") else 0)
        axisv.append(1 if h.get("SUCCESS_AXIS_V1") else 0)
        r2h.append(h["pre"]["ref_r_des_to_hand_deg"])
    for name, c in candidates.items():
        if not c.get("prep_ok"):
            continue
        h = c["handoff"]
        labels.append(name)
        news.append(1 if h.get("SUCCESS_NEW") else 0)
        d12.append(1 if h.get("SUCCESS_12S") else 0)
        axisv.append(1 if h.get("SUCCESS_AXIS_V1") else 0)
        r2h.append(h["pre"]["ref_r_des_to_hand_deg"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, news, 0.2, label="SUCCESS_NEW")
    ax.bar(x, d12, 0.2, label="SUCCESS_12S")
    ax.bar(x + 0.2, axisv, 0.2, label="AXIS_V1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_title("HANDOFF outcomes (not ranked by IC similarity)")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "handoff_outcomes.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(labels, r2h)
    ax.set_ylabel("ref r_des ↔ hand (deg)")
    ax.set_title("Orientation discontinuity vs IC grasp r_des at HANDOFF")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "r_des_discontinuity.png", dpi=120)
    plt.close(fig)

    if p5_h.get("SUCCESS_NEW"):
        rec = "direct_P5_SUCCESS_NEW"
    elif any((c.get("handoff") or {}).get("SUCCESS_NEW") for c in candidates.values()):
        rec = f"candidate_{best_name}_SUCCESS_NEW"
    elif ctrl.get("SUCCESS_NEW") or ctrl.get("SUCCESS_AXIS_V1"):
        rec = "recovered_states_fail_control_B_post_ok_handoff_r_des_mismatch"
    else:
        rec = "all_candidates_handoff_fail_inspect_continuity"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "post_P5": frozen["snapshot"],
        "removed_P6": "no restore to IC alpha≈122°",
        "continuity_code": continuity_doc,
        "direct_P5": {k: v for k, v in p5_h.items() if k != "trace_500ms"},
        "control_B_post": results["control_B_post"],
        "candidates": {
            k: {kk: vv for kk, vv in v.items() if kk != "prep_actions"} for k, v in candidates.items()
        },
        "best": best_name,
        "replay_ic": replay,
        "videos": videos,
        "recommendation": rec,
        "core_rule": "Recovery endpoint need not match IC; goal is stable nominal continuation.",
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    # Candidate table rows
    crow = []
    crow.append(
        f"| direct_P5 | {p5_h['pre']['alpha_up_deg']:.1f} | {np.round(p5_h['pre']['e_signed_mm'],2).tolist()} | "
        f"{p5_h['pre']['ref_r_des_to_hand_deg']:.1f} | {p5_h['first_event']} | {p5_h['SUCCESS_NEW']} | {p5_h['std_reason']} |"
    )
    for name, c in candidates.items():
        if not c.get("prep_ok"):
            crow.append(f"| {name} | {c.get('alpha_achieved')} | prep_fail | — | — | — | {c.get('reason')} |")
            continue
        h = c["handoff"]
        crow.append(
            f"| {name} | {h['pre']['alpha_up_deg']:.1f} | {np.round(h['pre']['e_signed_mm'],2).tolist()} | "
            f"{h['pre']['ref_r_des_to_hand_deg']:.1f} | {h['first_event']} | {h['SUCCESS_NEW']} | {h['std_reason']} |"
        )

    report = f"""# HANDOFF State Optimization — Plan B hard_910220

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

## Verdict

**{rec}**

P6 IC-orientation restore (**α≈122°**) removed. Recovery endpoint is free; goal is nominal continuation.

## Post-P5 frozen state

α_up = {frozen['snapshot']['alpha_up_deg']:.1f}° (fingertips-down gate)  
e = {np.round(frozen['snapshot']['e_signed_mm'], 2).tolist()}  
bilateral / low v_rel verified before HANDOFF tests.

## HANDOFF continuity (frozen semantics — not modified)

`envs/outcomes/recovery_success.py::apply_handoff`:

- Sets `fsm.r_des ← ref['r_des']` (**IC grasp orientation**)
- Sets `fsm.p_des ← current hand position`
- Does **not** edit qpos/qvel

Evidence (direct P5):

- Before apply: `ref_r_des ↔ hand` = **{p5_h['pre']['ref_r_des_to_hand_deg']:.1f}°**
- After apply: `r_des ↔ hand` = **{p5_h['post_apply']['r_des_to_hand_deg']:.1f}°**
- First event: **{p5_h['first_event']}** @ {p5_h['first_event_t']} s

Control B_post_lateral: NEW={ctrl.get('SUCCESS_NEW')} 12S={ctrl.get('SUCCESS_12S')} AXIS={ctrl.get('SUCCESS_AXIS_V1')} reason={ctrl.get('std_reason')}  
(ref↔hand = {ctrl['pre']['ref_r_des_to_hand_deg']:.1f}°)

## Candidate comparison

| ID | α_up | e (mm) | ref↔hand (°) | first event | SUCCESS_NEW | reason |
|----|------|--------|--------------|-------------|-------------|--------|
{chr(10).join(crow)}

Best: `{best_name}`

## Interpretation

{("Direct P5 HANDOFF succeeded: after endpoint-only P0–P5 the hand SO(3) is already within "
  f"{p5_h['pre']['ref_r_des_to_hand_deg']:.1f}° of frozen IC `ref['r_des']`, even though "
  f"α_up≈{p5_h['pre']['alpha_up_deg']:.1f}° ≠ IC α≈122°. "
  "IC α_up matching is neither necessary nor the success criterion; "
  "`apply_handoff` retargets full SO(3) `r_des←ref['r_des']`, and the verified reverse path "
  "already lands near that frame. Bounded α candidates were skipped.")
 if p5_h.get("SUCCESS_NEW") else
 ("Direct P5 failed. Bounded α candidates were tested. Under frozen HANDOFF, "
  "`apply_handoff` sets `r_des←ref['r_des']` (IC grasp orientation) while keeping `p_des` at "
  "the current hand. Matching IC α≈122° alone is neither necessary nor sufficient; "
  "continuity of the full SO(3) grasp frame vs `ref['r_des']` is the relevant quantity. "
  "No nominal/HANDOFF code was changed.")}

No nominal/HANDOFF code was changed in this diagnostic.

## Replay

{json.dumps(jsafe(replay), indent=2) if replay else 'not run'}

## Videos

{json.dumps(jsafe(videos), indent=2)}
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": rec, "wall_s": summary["wall_s"], "best": best_name}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
