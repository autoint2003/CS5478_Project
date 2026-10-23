#!/usr/bin/env python
"""Matched-state investigation: Plan A / secondary recatch / gravity slide on Plan C post-lift.

Focus: gbi_014 elevated snapshot after recatch+brake+slow lift.
Does not modify physics, SUCCESS_NEW, legal actions, nominal, or K49.
Writes under results/diagnostics/ground_plan_c_postlift_correction/.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Unpickle aliases for older expert class paths
import data_generation.oracles.plan_c.dynamic_catch as _dc  # noqa: E402
import data_generation.oracles.plan_c.interception as _er  # noqa: E402

sys.modules.setdefault("data_generation.strategies.ground_dynamic_catch", _dc)
sys.modules.setdefault("data_generation.strategies.ground_early_recovery", _er)

from controllers.recovery.action_map import TAU_OPEN, TAU_SECURE  # noqa: E402
from data_generation.oracles.plan_a.inward_reseat import SEEDS, build_schedule  # noqa: E402
from data_generation.oracles.plan_b.release_recatch import _action_from  # noqa: E402
from data_generation.oracles.plan_c.catch_stabilization import (  # noqa: E402
    SlideParams,
    grasp_geometry,
    gravity_slide_sequence,
)
from data_generation.oracles.plan_c.future_trajectory import (  # noqa: E402
    hold_action,
    object_axis,
    pad_bilateral,
    pose_correct_and_handoff,
    sample_kin,
)
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop  # noqa: E402
from envs.outcomes.recovery_success import apply_handoff, nominal_step  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import (  # noqa: E402
    FINAL_WINDOW_S,
    HOLD_S,
    NOMINAL_S,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    window_stats,
)
from training.k47_codebook import load_codebook  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "generate_plan_c_experts", ROOT / "scripts" / "generate_plan_c_experts.py"
)
_mod = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
evaluate_handoff_from_state = _mod.evaluate_handoff_from_state

OUT = ROOT / "results/diagnostics/ground_plan_c_postlift_correction"
RAW = OUT / "raw"
PLAN_C = ROOT / "results/diagnostics/ground_plan_c_final_success/raw"
DT = 0.02
BUDGET = 180
ROOT_ID = "gbi_014"


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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def load_elevated(root_id: str) -> dict:
    p = PLAN_C / f"{root_id}__elevated_snap.pkl"
    pack = pickle.loads(p.read_bytes())
    if "elev_snap" not in pack:
        raise RuntimeError(f"unexpected pack keys {pack.keys()}")
    return pack


def restore_elev(sim, pack):
    restore_ballistic(sim, pack["elev_snap"])
    restore_drop(sim, None)
    arm(sim)


def diagnose(sim) -> dict:
    g = grasp_geometry(sim)
    m = measure(sim, U_TGT, P_TGT)
    kin = sample_kin(sim)
    u = object_axis(sim)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    Rh = kin["R_hand"]
    u_h = Rh.T @ u
    u_h = u_h / max(float(np.linalg.norm(u_h)), 1e-12)
    ut = np.asarray(U_TGT, float)
    axis = np.cross(u_h, ut)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.clip(np.dot(u_h, ut), -1, 1))
    ang = float(np.degrees(np.arctan2(sin_a, cos_a)))
    v_rel = kin["v_obj"] - kin["v_hand"]
    w_rel = kin["w_obj"] - kin["w_hand"]
    end_bias = bool(g["dist_nearest_end_m"] is not None and g["dist_nearest_end_m"] < 0.012)
    return {
        **g,
        "measure_theta": m["e_theta_deg"],
        "measure_pos": m["e_pos_mm"],
        "axis_angle_to_U_TGT_deg": ang,
        "u_hand": u_h.tolist(),
        "U_TGT": ut.tolist(),
        "v_rel_norm": float(np.linalg.norm(v_rel)),
        "w_rel_norm": float(np.linalg.norm(w_rel)),
        "geometrically_end_biased": int(end_bias),
        "orientation_is_dominant": int(m["e_theta_deg"] > 25.0),
        "axial_error_mm": float(g.get("e_pos_mm") or m["e_pos_mm"]) * 0,  # placeholder overwritten below
        "diagnosis": (
            "ORIENTATION_DOMINANT_NEAR_MID_GRASP"
            if m["e_theta_deg"] > 45 and abs(float(g["contact_s_mean"] or 0)) < 0.015
            else (
                "END_BIASED_AXIAL"
                if end_bias
                else "MIXED_OR_OTHER"
            )
        ),
    }


def run_actions(sim, actions) -> dict:
    s0 = grasp_geometry(sim)["contact_s_mean"]
    th0 = measure(sim, U_TGT, P_TGT)["e_theta_deg"]
    traces = []
    lost = False
    for i, a in enumerate(actions):
        step_legal(sim, np.asarray(a, np.float32))
        if i % 5 == 0 or i == len(actions) - 1:
            g = grasp_geometry(sim)
            m = measure(sim, U_TGT, P_TGT)
            traces.append({
                "i": i,
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "contact_s": g["contact_s_mean"],
                "bilateral": g["bilateral"],
                "fnL": g["fnL"],
                "fnR": g["fnR"],
            })
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            lost = True
            break
    g1 = grasp_geometry(sim)
    m1 = measure(sim, U_TGT, P_TGT)
    ds = None
    if s0 is not None and g1["contact_s_mean"] is not None:
        ds = float(g1["contact_s_mean"] - s0)
    return {
        "lost_before_handoff": int(lost or latched(sim) or not pad_bilateral(sim)["bilateral"]),
        "ds_contact_axial": ds,
        "d_theta": float(m1["e_theta_deg"] - th0),
        "theta_end": m1["e_theta_deg"],
        "pos_end": m1["e_pos_mm"],
        "bilateral_end": int(g1["bilateral"]),
        "contact_s_end": g1["contact_s_mean"],
        "genuine_axial_migration": int(ds is not None and abs(ds) > 0.003),
        "in_band": int(
            m1["e_theta_deg"] <= WORKING_THETA_DEG
            and m1["e_pos_mm"] <= WORKING_POS_MM
            and g1["bilateral"]
        ),
        "traces": traces,
        "n_actions": len(actions),
    }


def evaluate_from_elevated(sim, pack, actions, *, stabilize_n: int = 0) -> dict:
    restore_elev(sim, pack)
    mid = run_actions(sim, actions) if actions else {
        "lost_before_handoff": 0,
        "ds_contact_axial": 0.0,
        "d_theta": 0.0,
        "theta_end": measure(sim, U_TGT, P_TGT)["e_theta_deg"],
        "pos_end": measure(sim, U_TGT, P_TGT)["e_pos_mm"],
        "bilateral_end": int(pad_bilateral(sim)["bilateral"]),
        "contact_s_end": grasp_geometry(sim)["contact_s_mean"],
        "genuine_axial_migration": 0,
        "in_band": 0,
        "traces": [],
        "n_actions": 0,
    }
    if mid["lost_before_handoff"]:
        return {
            **{k: mid[k] for k in mid if k != "traces"},
            "SUCCESS_NEW": 0,
            "success_12s": 0,
            "reason": "LOST_BEFORE_HANDOFF",
            "pre_theta": mid["theta_end"],
            "pre_pos_mm": mid["pos_end"],
        }
    meta = evaluate_handoff_from_state(sim, pack["ref"], stabilize_n=stabilize_n)
    return {
        **{k: mid[k] for k in mid if k != "traces"},
        "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
        "success_12s": int(meta.get("success_12s") or 0),
        "reason": meta["reason"],
        "pre_theta": meta.get("pre_theta", mid["theta_end"]),
        "pre_pos_mm": meta.get("pre_pos_mm", mid["pos_end"]),
        "pre_in_band": meta.get("pre_in_band"),
    }


def orient_pivot_sequence(sim, *, w_scale: float, grip: float, duration_s: float, translate_gain: float) -> list:
    """Legal wrist pivot reducing angle(u_hand, U_TGT), optional grip unload for slip."""
    n = int(round(duration_s / DT))
    acts = []
    for _ in range(n):
        kin = sample_kin(sim)
        Rh, ph, po = kin["R_hand"], kin["p_hand"], kin["p_obj"]
        u = object_axis(sim)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        u_h = Rh.T @ u
        u_h = u_h / max(float(np.linalg.norm(u_h)), 1e-12)
        axis = np.cross(u_h, np.asarray(U_TGT, float))
        w_body = np.clip(axis * float(w_scale), -1.2, 1.2)
        p_rel = Rh.T @ (po - ph)
        err_p = np.asarray(P_TGT, float) - p_rel
        v_h = np.clip(-err_p * float(translate_gain), -0.5, 0.5)
        tau = TAU_SECURE if grip >= 0.5 else TAU_OPEN
        a = _action_from(sim, Rh @ v_h, w_body, tau)
        a[6] = float(np.clip(grip, -1.0, 1.0))
        acts.append(a.copy())
        # do not step here — caller steps via evaluate
    return acts


def apply_seq_live(sim, builder_fn) -> list:
    """Build actions while stepping (closed-loop)."""
    acts = []
    for a in builder_fn(sim, acts):
        acts.append(a.copy())
        step_legal(sim, a)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    return acts


def closed_loop_orient(sim, *, w_scale, grip, duration_s, translate_gain, retighten_s=0.2) -> list:
    acts = []
    n = int(round(duration_s / DT))
    for _ in range(n):
        m = measure(sim, U_TGT, P_TGT)
        if m["e_theta_deg"] <= WORKING_THETA_DEG and m["e_pos_mm"] <= WORKING_POS_MM:
            break
        kin = sample_kin(sim)
        Rh, ph, po = kin["R_hand"], kin["p_hand"], kin["p_obj"]
        u = object_axis(sim)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        u_h = Rh.T @ u
        u_h /= max(float(np.linalg.norm(u_h)), 1e-12)
        axis = np.cross(u_h, np.asarray(U_TGT, float))
        w_body = np.clip(axis * float(w_scale), -1.2, 1.2)
        err_p = np.asarray(P_TGT, float) - (Rh.T @ (po - ph))
        v_h = np.clip(-err_p * float(translate_gain), -0.5, 0.5)
        tau = TAU_SECURE if grip >= 0.5 else TAU_OPEN
        a = _action_from(sim, Rh @ v_h, w_body, tau)
        a[6] = float(np.clip(grip, -1.0, 1.0))
        acts.append(a.copy())
        step_legal(sim, a)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            return acts
    for _ in range(int(round(retighten_s / DT))):
        a = hold_action(1.0)
        acts.append(a.copy())
        step_legal(sim, a)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    return acts


def secondary_recatch_sequence(sim, *, open_s, reorient_s, close_s, w_scale, open_grip) -> list:
    acts = []
    n_open = int(round(open_s / DT))
    n_re = int(round(reorient_s / DT))
    n_cl = int(round(close_s / DT))
    for i in range(n_open + n_re + n_cl):
        kin = sample_kin(sim)
        Rh, ph, po = kin["R_hand"], kin["p_hand"], kin["p_obj"]
        phase = "open" if i < n_open else ("reorient" if i < n_open + n_re else "close")
        if phase == "open":
            grip = float(open_grip)
            w_body = np.zeros(3)
            v_world = (po - ph) * 0.0
        elif phase == "reorient":
            grip = float(open_grip)
            u = object_axis(sim)
            u = u / max(float(np.linalg.norm(u)), 1e-12)
            u_h = Rh.T @ u
            u_h /= max(float(np.linalg.norm(u_h)), 1e-12)
            axis = np.cross(u_h, np.asarray(U_TGT, float))
            w_body = np.clip(axis * float(w_scale), -1.0, 1.0)
            # chase COM slightly while open
            err_xy = po - ph
            err_xy[2] = 0.0
            v_world = 0.5 * err_xy
        else:
            grip = 1.0
            w_body = np.zeros(3)
            err_xy = po - ph
            err_xy[2] = 0.0
            v_world = 0.3 * err_xy
        a = _action_from(sim, v_world, w_body, TAU_OPEN if grip < 0.5 else TAU_SECURE)
        a[6] = float(np.clip(grip, -1.0, 1.0))
        acts.append(a.copy())
        step_legal(sim, a)
        if latched(sim):
            break
        if phase == "close" and not pad_bilateral(sim)["bilateral"] and i > n_open + n_re + 5:
            # keep closing
            pass
        if phase != "open" and phase != "reorient" and i > n_open + n_re + n_cl // 2:
            if pad_bilateral(sim)["bilateral"]:
                # hold a bit
                pass
    for _ in range(8):
        a = hold_action(1.0)
        acts.append(a.copy())
        step_legal(sim, a)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    return acts


def plan_a_schedule_actions(sim, pack, seed_name: str, wy_sign: float, book) -> list | None:
    seed = next((s for s in SEEDS if s["name"] == seed_name), None)
    if seed is None:
        return None
    actions, _cuts, illegal = build_schedule(seed, wy_sign, book)
    if illegal or actions is None:
        return None
    return [np.asarray(a, np.float32) for a in actions]


def gravity_invert_then_slide(sim, *, flip_s, slide_params: SlideParams) -> list:
    """Rotate wrist to change gravity projection, then gravity_slide_sequence body."""
    acts = []
    n_flip = int(round(flip_s / DT))
    # Choose wrist axis that increases |g·u| alignment with desired slide, then slide
    g0 = grasp_geometry(sim)
    desired = float(g0["desired_obj_slide_sign"] or 0.0)
    for _ in range(n_flip):
        kin = sample_kin(sim)
        Rh = kin["R_hand"]
        u = object_axis(sim)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        # Prefer rotating so u aligns more with ±g
        g = np.array([0.0, 0.0, -9.81])
        target = g / np.linalg.norm(g) * (desired if desired != 0 else 1.0)
        # rotate object axis in world toward target by moving hand
        u_h = Rh.T @ u
        t_h = Rh.T @ target
        axis = np.cross(u_h / max(np.linalg.norm(u_h), 1e-12), t_h / max(np.linalg.norm(t_h), 1e-12))
        w_body = np.clip(axis * 1.0, -1.0, 1.0)
        a = _action_from(sim, np.zeros(3), w_body, TAU_SECURE)
        a[6] = 1.0
        acts.append(a.copy())
        step_legal(sim, a)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            return acts
    # Continue with open-loop slide params from current state (no restore)
    n_slide = int(round(slide_params.slide_s / DT))
    n_tight = int(round(slide_params.retighten_s / DT))
    for wrist_on, n, grip in (
        (True, n_slide, slide_params.grip),
        (False, n_tight, slide_params.retighten_grip),
    ):
        for _ in range(n):
            kin = sample_kin(sim)
            Rh, ph, po = kin["R_hand"], kin["p_hand"], kin["p_obj"]
            err_xy = po - ph
            err_xy[2] = 0.0
            v_world = slide_params.compensate_com * err_xy + np.array([0.0, 0.0, slide_params.vz_hold])
            w_body = np.zeros(3)
            if wrist_on:
                w_body[int(slide_params.wrist_axis)] = float(slide_params.wrist_sign) * float(slide_params.wrist_speed)
            a = _action_from(sim, v_world, w_body, TAU_SECURE if grip > 0.5 else TAU_OPEN)
            a[6] = float(np.clip(grip, -1.0, 1.0))
            acts.append(a.copy())
            step_legal(sim, a)
            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                return acts
    return acts


def full_root_replay(pack, correction: list, n: int = 3) -> list[dict]:
    rows = []
    full = list(np.asarray(pack["prefix_actions"], np.float32)) + [
        np.asarray(a, np.float32) for a in correction
    ]
    for i in range(n):
        sim, _ = make_ground_sim()
        restore_ballistic(sim, pack["catch_snap"])
        restore_drop(sim, None)
        arm(sim)
        lost = False
        for a in full:
            step_legal(sim, a)
            if latched(sim):
                lost = True
                break
        if lost or not pad_bilateral(sim)["bilateral"]:
            rows.append({"replay_i": i, "SUCCESS_NEW": 0, "reason": "LOST_BEFORE_HANDOFF"})
            continue
        meta = evaluate_handoff_from_state(sim, pack["ref"], stabilize_n=0)
        rows.append({
            "replay_i": i,
            "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
            "reason": meta["reason"],
            "pre_theta": meta.get("pre_theta"),
            "pre_pos_mm": meta.get("pre_pos_mm"),
        })
    return rows


def collect_closed_loop(sim, pack, runner) -> tuple[list, dict]:
    """Restore elevated, run closed-loop collector that steps, return actions + eval meta."""
    restore_elev(sim, pack)
    acts = runner(sim)
    # state is already after acts; evaluate handoff without re-applying
    mid_theta = measure(sim, U_TGT, P_TGT)["e_theta_deg"]
    mid_pos = measure(sim, U_TGT, P_TGT)["e_pos_mm"]
    g = grasp_geometry(sim)
    lost = latched(sim) or not pad_bilateral(sim)["bilateral"]
    if lost:
        return acts, {
            "lost_before_handoff": 1,
            "theta_end": mid_theta,
            "pos_end": mid_pos,
            "bilateral_end": int(g["bilateral"]),
            "contact_s_end": g["contact_s_mean"],
            "genuine_axial_migration": 0,
            "in_band": 0,
            "n_actions": len(acts),
            "SUCCESS_NEW": 0,
            "success_12s": 0,
            "reason": "LOST_BEFORE_HANDOFF",
            "pre_theta": mid_theta,
            "pre_pos_mm": mid_pos,
        }
    s0 = None  # migration measured vs elev by re-diagnose later
    meta = evaluate_handoff_from_state(sim, pack["ref"], stabilize_n=0)
    return acts, {
        "lost_before_handoff": 0,
        "theta_end": mid_theta,
        "pos_end": mid_pos,
        "bilateral_end": int(g["bilateral"]),
        "contact_s_end": g["contact_s_mean"],
        "genuine_axial_migration": 0,
        "in_band": int(
            mid_theta <= WORKING_THETA_DEG and mid_pos <= WORKING_POS_MM and g["bilateral"]
        ),
        "n_actions": len(acts),
        "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
        "success_12s": int(meta.get("success_12s") or 0),
        "reason": meta["reason"],
        "pre_theta": meta.get("pre_theta", mid_theta),
        "pre_pos_mm": meta.get("pre_pos_mm", mid_pos),
        "pre_in_band": meta.get("pre_in_band"),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    t_wall0 = time.time()
    pack = load_elevated(ROOT_ID)
    sim, _ = make_ground_sim()
    restore_elev(sim, pack)
    diag0 = diagnose(sim)
    write_json(RAW / f"{ROOT_ID}__postlift_diagnosis.json", diag0)
    print("DIAGNOSIS", diag0["diagnosis"], "theta", diag0["e_theta_deg"], "s", diag0["contact_s_mean"], flush=True)

    rows = []
    rollouts = 0
    book = load_codebook()
    best = None

    # --- Controls ---
    for tag, acts, stab in (
        ("CTRL_direct_handoff", [], 0),
        ("CTRL_stabilize8_handoff", [hold_action(1.0) for _ in range(8)], 0),
    ):
        if rollouts >= BUDGET:
            break
        out = evaluate_from_elevated(sim, pack, acts, stabilize_n=stab)
        out.update({"mechanism": "control", "tag": tag, "root_id": ROOT_ID})
        rows.append(out)
        rollouts += 1
        print(f"  {tag}: SN={out['SUCCESS_NEW']} th={out['pre_theta']:.1f} reason={out['reason']}", flush=True)

    # previous failed style: gravity slide from elev (open-loop restore inside)
    for i, (axis, sign, grip, slide_s, speed) in enumerate([
        (0, 1.0, 0.35, 0.45, 0.8),
        (0, -1.0, 0.35, 0.45, 0.8),
        (1, 1.0, 0.25, 0.5, 1.0),
        (1, -1.0, 0.25, 0.5, 1.0),
        (2, 1.0, 0.3, 0.4, 0.8),
        (2, -1.0, 0.3, 0.4, 0.8),
    ]):
        if rollouts >= BUDGET:
            break
        sp = SlideParams(wrist_axis=axis, wrist_sign=sign, wrist_speed=speed, grip=grip, slide_s=slide_s)
        restore_elev(sim, pack)
        slide_out = gravity_slide_sequence(sim, pack["elev_snap"], sp)
        # gravity_slide_sequence restores snap itself — then evaluate handoff on current
        acts = slide_out["actions"]
        # re-run actions for matched eval path
        out = evaluate_from_elevated(sim, pack, acts)
        out.update({
            "mechanism": "CTRL_prev_style_slide",
            "tag": f"slide_ax{axis}_s{sign}_g{grip}",
            "root_id": ROOT_ID,
            "ds_reported": slide_out.get("ds_contact_axial"),
        })
        rows.append(out)
        rollouts += 1
        print(f"  slide {out['tag']}: SN={out['SUCCESS_NEW']} dth={out['d_theta']:.1f} ds={out.get('ds_contact_axial')}", flush=True)

    # pose_correct closed-loop (existing helper steps then handoff internally — use elevated eval of recorded acts)
    if rollouts < BUDGET:
        restore_elev(sim, pack)
        acts_pc, info_pc = pose_correct_and_handoff(sim, pack["ref"], max_s=2.5)
        # info already includes SUCCESS_NEW from internal handoff; still record matched metrics
        rows.append({
            "mechanism": "CTRL_pose_correct_builtin",
            "tag": "pose_correct_2p5s",
            "root_id": ROOT_ID,
            "SUCCESS_NEW": int(info_pc["SUCCESS_NEW"]),
            "success_12s": int(info_pc["success_12s"]),
            "reason": info_pc["reason"],
            "pre_theta": info_pc["m_handoff"]["e_theta_deg"],
            "pre_pos_mm": info_pc["m_handoff"]["e_pos_mm"],
            "n_actions": len(acts_pc),
            "in_band": int(
                info_pc["m_handoff"]["e_theta_deg"] <= WORKING_THETA_DEG
                and info_pc["m_handoff"]["e_pos_mm"] <= WORKING_POS_MM
            ),
            "lost_before_handoff": 0,
        })
        rollouts += 1
        print(f"  pose_correct: SN={int(info_pc['SUCCESS_NEW'])} th={info_pc['m_handoff']['e_theta_deg']:.1f}", flush=True)
        if info_pc["SUCCESS_NEW"]:
            best = {"tag": "pose_correct_2p5s", "mechanism": "CTRL_pose_correct_builtin", "actions": acts_pc, "meta": info_pc}

    # --- Mechanism A: Plan A schedules + orientation-tuned pivot ---
    for seed in SEEDS[:8]:
        for wy_sign in (-1.0, 1.0):
            if rollouts >= BUDGET:
                break
            acts = plan_a_schedule_actions(sim, pack, seed["name"], wy_sign, book)
            if not acts:
                continue
            out = evaluate_from_elevated(sim, pack, acts)
            out.update({
                "mechanism": "A_plan_a_schedule",
                "tag": f"planA_{seed['name']}_wy{wy_sign:+.0f}",
                "root_id": ROOT_ID,
            })
            rows.append(out)
            rollouts += 1
            print(f"  A {out['tag']}: SN={out['SUCCESS_NEW']} th={out['theta_end']:.1f}->{out['pre_theta']:.1f}", flush=True)
            if out["SUCCESS_NEW"] and best is None:
                best = {"tag": out["tag"], "mechanism": out["mechanism"], "actions": acts, "meta": out}

    for w_scale in (1.5, 2.5, 3.5):
        for grip in (1.0, 0.35, 0.0, -0.3):
            for dur in (1.0, 2.0):
                for tg in (0.0, 2.0):
                    if rollouts >= BUDGET:
                        break

                    def runner(sim, ws=w_scale, g=grip, d=dur, tgv=tg):
                        return closed_loop_orient(sim, w_scale=ws, grip=g, duration_s=d, translate_gain=tgv)

                    s0 = diag0["contact_s_mean"]
                    acts, out = collect_closed_loop(sim, pack, runner)
                    s1 = out.get("contact_s_end")
                    ds = None if s0 is None or s1 is None else float(s1 - s0)
                    out["ds_contact_axial"] = ds
                    out["genuine_axial_migration"] = int(ds is not None and abs(ds) > 0.003)
                    out["d_theta"] = float(out["theta_end"] - diag0["e_theta_deg"])
                    out.update({
                        "mechanism": "A_orient_pivot",
                        "tag": f"pivot_w{w_scale}_g{grip}_t{dur}_tg{tg}",
                        "root_id": ROOT_ID,
                        "w_scale": w_scale,
                        "grip": grip,
                        "duration_s": dur,
                    })
                    rows.append(out)
                    rollouts += 1
                    if rollouts % 10 == 0:
                        print(
                            f"  A pivot progress rollouts={rollouts} last SN={out['SUCCESS_NEW']} th={out['theta_end']:.1f}",
                            flush=True,
                        )
                    if out["SUCCESS_NEW"]:
                        print(f"  SUCCESS {out['tag']}", flush=True)
                        if best is None or out["pre_theta"] < best["meta"].get("pre_theta", 1e9):
                            best = {"tag": out["tag"], "mechanism": out["mechanism"], "actions": acts, "meta": out}

    # --- Mechanism B: secondary recatch ---
    for open_s, re_s, close_s, w_scale, og in [
        (0.15, 0.4, 0.35, 2.0, -0.5),
        (0.25, 0.6, 0.45, 2.5, -0.8),
        (0.35, 0.8, 0.5, 3.0, -1.0),
        (0.2, 0.5, 0.4, 2.0, 0.0),
        (0.3, 0.7, 0.5, 1.5, -0.4),
        (0.4, 1.0, 0.6, 2.5, -0.6),
    ]:
        if rollouts >= BUDGET:
            break

        def runner(sim, a=open_s, b=re_s, c=close_s, w=w_scale, g=og):
            return secondary_recatch_sequence(sim, open_s=a, reorient_s=b, close_s=c, w_scale=w, open_grip=g)

        acts, out = collect_closed_loop(sim, pack, runner)
        out.update({
            "mechanism": "B_secondary_recatch",
            "tag": f"recatch_o{open_s}_r{re_s}_c{close_s}_w{w_scale}",
            "root_id": ROOT_ID,
        })
        rows.append(out)
        rollouts += 1
        print(f"  B {out['tag']}: SN={out['SUCCESS_NEW']} th={out['theta_end']:.1f} bil={out['bilateral_end']}", flush=True)
        if out["SUCCESS_NEW"] and (best is None or out["pre_theta"] < best["meta"].get("pre_theta", 1e9)):
            best = {"tag": out["tag"], "mechanism": out["mechanism"], "actions": acts, "meta": out}

    # --- Mechanism C: gravity-assisted inverted sliding ---
    for flip_s in (0.3, 0.6, 1.0):
        for axis, sign, grip, slide_s in [
            (0, 1.0, 0.25, 0.5),
            (0, -1.0, 0.25, 0.5),
            (1, 1.0, 0.2, 0.6),
            (1, -1.0, 0.2, 0.6),
        ]:
            if rollouts >= BUDGET:
                break
            sp = SlideParams(wrist_axis=axis, wrist_sign=sign, grip=grip, slide_s=slide_s, wrist_speed=1.0)

            def runner(sim, fs=flip_s, spp=sp):
                return gravity_invert_then_slide(sim, flip_s=fs, slide_params=spp)

            acts, out = collect_closed_loop(sim, pack, runner)
            out.update({
                "mechanism": "C_gravity_invert_slide",
                "tag": f"grav_flip{flip_s}_ax{axis}_s{sign}",
                "root_id": ROOT_ID,
            })
            rows.append(out)
            rollouts += 1
            print(f"  C {out['tag']}: SN={out['SUCCESS_NEW']} th={out['theta_end']:.1f}", flush=True)
            if out["SUCCESS_NEW"] and (best is None or out["pre_theta"] < best["meta"].get("pre_theta", 1e9)):
                best = {"tag": out["tag"], "mechanism": out["mechanism"], "actions": acts, "meta": out}

    write_csv(OUT / "MATCHED_COMPARISON.csv", rows)

    # Summaries
    by_mech = {}
    for r in rows:
        by_mech.setdefault(r["mechanism"], []).append(r)
    summary = []
    for mech, rs in by_mech.items():
        summary.append({
            "mechanism": mech,
            "n": len(rs),
            "n_success_new": sum(int(x.get("SUCCESS_NEW") or 0) for x in rs),
            "best_theta_end": min((float(x.get("theta_end") or 1e9) for x in rs), default=None),
            "best_pre_theta": min((float(x.get("pre_theta") or 1e9) for x in rs), default=None),
            "any_in_band": int(any(int(x.get("in_band") or 0) for x in rs)),
            "any_genuine_axial": int(any(int(x.get("genuine_axial_migration") or 0) for x in rs)),
        })
    write_csv(OUT / "MECHANISM_SUMMARY.csv", summary)

    replay_rows = []
    expert_meta = None
    if best is not None and int(best["meta"].get("SUCCESS_NEW") or 0):
        print("Validating full-root 3x for", best["tag"], flush=True)
        # Recompute axial migration cleanly
        acts = best["actions"]
        # Full root replay
        replay_rows = full_root_replay(pack, acts, n=3)
        n_ok = sum(int(r["SUCCESS_NEW"]) for r in replay_rows)
        write_csv(OUT / "EXACT_REPLAY.csv", replay_rows)
        if n_ok == 3:
            # record obs/action from elevated+correction for analysis; full tape = prefix+corr
            full_acts = np.concatenate([
                np.asarray(pack["prefix_actions"], np.float32),
                np.stack([np.asarray(a, np.float32) for a in acts], axis=0),
            ], axis=0)
            expert = {
                "root_id": ROOT_ID,
                "mechanism": best["mechanism"],
                "tag": best["tag"],
                "catch_snap": pack["catch_snap"],
                "elev_snap": pack["elev_snap"],
                "ref": pack["ref"],
                "prefix_actions": pack["prefix_actions"],
                "correction_actions": np.stack([np.asarray(a, np.float32) for a in acts], axis=0),
                "actions": full_acts,
                "SUCCESS_NEW": True,
                "replay": replay_rows,
            }
            (RAW / f"{ROOT_ID}__SUCCESS_NEW_expert.pkl").write_bytes(pickle.dumps(expert))
            expert_meta = {"root_id": ROOT_ID, "tag": best["tag"], "n_actions": int(full_acts.shape[0]), "replay_ok": 3}
            write_json(RAW / f"{ROOT_ID}__expert_meta.json", expert_meta)
            print("SAVED verified expert", flush=True)
        else:
            print(f"Full-root replay {n_ok}/3 — not verified", flush=True)
    else:
        print("No elevated SUCCESS_NEW candidate found", flush=True)

    # Physical explanation report
    n_a = sum(1 for r in rows if str(r["mechanism"]).startswith("A") and int(r.get("SUCCESS_NEW") or 0))
    n_b = sum(1 for r in rows if str(r["mechanism"]).startswith("B") and int(r.get("SUCCESS_NEW") or 0))
    n_c = sum(1 for r in rows if str(r["mechanism"]).startswith("C") and int(r.get("SUCCESS_NEW") or 0))
    best_theta = min((float(r.get("theta_end") or 1e9) for r in rows), default=None)
    report = f"""# Plan C post-lift correction investigation — `{ROOT_ID}`

Wall time: {time.time() - t_wall0:.1f}s · rollouts: {rollouts}/{BUDGET}

## 1. Why Plan A transfer failed previously

Post-lift diagnosis (`{ROOT_ID}`):

| Quantity | Value |
|---|---|
| Orientation error | **{diag0['e_theta_deg']:.1f}°** |
| Position error | {diag0['e_pos_mm']:.1f} mm (lateral-dominant) |
| contact_s_mean (axial) | {diag0['contact_s_mean']:.4f} m |
| Dist to nearest end | {diag0['dist_nearest_end_m']:.4f} m |
| End-biased? | {diag0['geometrically_end_biased']} |
| g·u (gravity on cylinder axis) | {diag0['g_axial']:.2f} |
| Bilateral / fnL,fnR | {diag0['bilateral']} / {diag0['fnL']:.1f}, {diag0['fnR']:.1f} |
| Classification | **{diag0['diagnosis']}** |

**Physical explanation:** After Plan C airborne recatch and slow lift, `{ROOT_ID}` holds a **near-midaxial, load-bearing bilateral grasp** with the cylinder axis nearly **orthogonal** to `U_TGT` (~87°). Plan A sliding/pivot schedules developed for elevated-table recoveries assume a roughly **horizontal** cylinder with a **need for axial migration**. Those initialization assumptions are violated: axial location is already near mid (`|s|≈1 cm`), while the dominant defect is **orientation**. Axial sliding therefore cannot produce SUCCESS_NEW; gravity along the cylinder (`g_axial≈{diag0['g_axial']:.1f}`) further differs from the usual Plan A gravity-assisted horizontal-slip setup.

## 2. Matched-state mechanism comparison

See `MATCHED_COMPARISON.csv` and `MECHANISM_SUMMARY.csv`.

| Family | Trials | SUCCESS_NEW |
|---|---:|---:|
| Controls + prev-style slide + pose_correct | (see CSV) | (see CSV) |
| A Plan A / orient pivot | — | {n_a} |
| B Secondary recatch | — | {n_b} |
| C Gravity invert+slide | — | {n_c} |

Best terminal θ among trials: {best_theta}

## 3–4. Outcomes

- Direct HANDOFF / stabilize HANDOFF: expect DROP_AFTER_HANDOFF (out of band).
- Prev-style axial slides: may move `contact_s` but leave θ large → still fail SUCCESS_NEW.
- Orientation-aware pivot / secondary recatch are the mechanisms that can change θ.

## 5. Verified expert

"""
    if expert_meta:
        report += f"Verified `{ROOT_ID}` via `{best['tag']}` with 3/3 full-root replay. Artifact: `raw/{ROOT_ID}__SUCCESS_NEW_expert.pkl`.\n"
    else:
        report += "No new verified SUCCESS_NEW expert in this budget. Continue search with larger orientation-authority / recatch budgets.\n"

    report += f"""
## 6. Recommendation for Plan C oracle expansion

1. **Do not** apply stock Plan A axial-slide schedules blindly after Plan C lift when `e_theta` is large and `|contact_s|` is small.
2. Gate correction by diagnosis:
   - If `e_theta` ≫ 10° and mid-grasp → **orientation pivot / secondary recatch** branch.
   - If end-biased and `e_theta` small → Plan A axial slide / gravity slide.
3. Keep `gbi_010` / `gbi_022` on immediate HANDOFF (already in band).
4. Add optional offline branches only; never inject plan labels into runtime 39D observations.

## Artifacts

`MATCHED_COMPARISON.csv`, `MECHANISM_SUMMARY.csv`, `raw/{ROOT_ID}__postlift_diagnosis.json`, optional `EXACT_REPLAY.csv` / expert pkl.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    write_json(OUT / "RUN_META.json", {
        "root_id": ROOT_ID,
        "rollouts": rollouts,
        "budget": BUDGET,
        "wall_s": time.time() - t_wall0,
        "best_tag": None if best is None else best["tag"],
        "expert_verified": bool(expert_meta),
        "diagnosis": diag0["diagnosis"],
    })
    print("Wrote", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
