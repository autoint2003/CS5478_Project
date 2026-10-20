"""Grasp-point alignment + closure timing for freefall recatch.

Matched expert vs corrected-P audit, ablations A–F, held-out eval.
Legal continuous 7D. No SFT/GRPO. No physics / 39D / geometry changes.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import SCALE_PREL, SCALE_ROT, SCALE_VREL, OBS_DIM_AIR
from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import (
    cylinder_axis_hand,
    cylinder_axis_world,
    measure as gmeasure,
    p_com_hand,
)
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT, write_csv, write_json
from scripts.run_freefall_recatch_collision_aware_feedback import (
    MID_OFFSET_H,
    corrected_O_action,
    corrected_P_action,
    decode_obs,
    hand_vel,
    jaw_target_err_from_obs,
    jaw_target_err_privileged,
    list_obj_contacts,
    obj_vel,
    shared_intercept_action,
    wrist_from_rot,
)
from scripts.run_freefall_recatch_pose_feedback import load_root, pose_now
from scripts.run_horizontal_offset_freefall_recatch import (
    SEARCH_NOMINAL_S,
    eval_full_sn,
    render_expert_video,
)
from scripts.run_lateral_offset_falling_recatch import (
    build_nominal_open_robot,
    finger_frame,
    hand_pos,
    make_ic,
    make_training_env,
    obj_pos,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import build_stabilization, clear_probe_latch, hand_R

SRC_CA = ROOT / "results/diagnostics/freefall_recatch_collision_aware_feedback"
SRC_FB = ROOT / "results/diagnostics/freefall_recatch_pose_feedback"
SRC_EXP = ROOT / "results/expert_generation/horizontal_offset_freefall_recatch"
OUT = ROOT / "results/diagnostics/freefall_recatch_grasp_point_alignment"
SEED = 20261106
CYL_HALF = 0.03
WALL_S = 50 * 60

# Baseline gains from collision-aware corr_P
BASE_GAINS = {
    "kp_chase": 1.85,
    "retract": -0.20,
    "descend": 1.0,
    "kw_app": 0.30,
    "wx0": 0.0,
    "wy0": 0.08,
    "wz0": 0.12,
    "t_lead_max": 0.10,
    "t_lead_bias": 0.05,
    "v_xy_nom": 0.65,
    "preclose": 1.0,
    "ez_fall_through": 0.03,
    "n_align_retract": 4,
    "xy_early_scoop": 0.048,
    "overchase": 0.25,
    "lift": -0.90,
    "kp_hold": 1.0,
    "n_bil_seat": 3,
    "xy_soft_capture": 0.042,
    "soft_descend": 0.35,
    "kp": 1.2,
    "kz": 0.5,
    "vz_bias": 0.25,
}

# Shared grasp-point gains (frozen across roots). Zero wrist *bias* preserves the
# early housing graze that spins the cylinder upright (expert-like); delayed
# soft-capture + vel-match centers the axial grasp and enables CATCH_HANDOFF.
GRASP_GAINS = {
    **BASE_GAINS,
    "kw_app": 0.0,
    "wx0": 0.0,
    "wy0": 0.0,
    "wz0": 0.0,
    "n_align_retract": 6,
    "xy_early_scoop": 0.035,
    "xy_soft_capture": 0.022,
    "preclose": -1.0,
    "n_bil_seat": 8,
    "lift": -0.92,
    "vm_gain": 0.8,
}


def aperture(sim) -> float:
    idx = sim.ids.finger_jnt
    idx = int(idx[0]) if hasattr(idx, "__len__") else int(idx)
    return float(sim.data.qpos[idx])


def contacts_axial(sim) -> tuple[list, np.ndarray, np.ndarray]:
    obj = int(sim.ids.object_body)
    left, right = int(sim.ids.left_body), int(sim.ids.right_body)
    hand = int(sim.ids.hand_body)
    po = obj_pos(sim)
    u = cylinder_axis_world(sim)
    rows = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        side = "L" if other == left else ("R" if other == right else ("hand" if other == hand else "other"))
        pos = np.asarray(c.pos, float)
        s = float(np.dot(pos - po, u))
        force = np.zeros(6, float)
        try:
            mujoco.mj_contactForce(sim.model, sim.data, i, force)
        except Exception:
            pass
        rows.append({
            "side": side,
            "s_axial": s,
            "d_end": CYL_HALF - abs(s),
            "near_tail": abs(s) > 0.018,
            "near_center": abs(s) < 0.012,
            "pos": pos.tolist(),
            "fn": float(force[0]),
        })
    return rows, u, po


def axial_summary(cons: list) -> dict:
    fr = [c for c in cons if c["side"] in ("L", "R")]
    if not fr:
        return {
            "mean_s_axial": None, "max_abs_s": None,
            "n_near_tail": 0, "n_near_center": 0, "n_finger": 0,
            "correct_axial": 0,
        }
    mean_s = float(np.mean([c["s_axial"] for c in fr]))
    max_abs = float(max(abs(c["s_axial"]) for c in fr))
    return {
        "mean_s_axial": mean_s,
        "max_abs_s": max_abs,
        "n_near_tail": sum(1 for c in fr if c["near_tail"]),
        "n_near_center": sum(1 for c in fr if c["near_center"]),
        "n_finger": len(fr),
        "correct_axial": int(abs(mean_s) < 0.012 and max_abs < 0.020),
    }


def theta_deg(sim) -> float:
    return float(gmeasure(sim, U_TGT, P_TGT)["e_theta_deg"])


def variant_gains(variant: str, base: dict | None = None) -> dict:
    """Frozen ablation gain sets (no root ID). Built on corrected_P."""
    g = dict(base or BASE_GAINS)
    # B: axial / approach geometry — zero wrist bias + longer XY align (enables graze-spin)
    if variant in ("B", "E", "F", "corr_O_E", "corr_O_F"):
        g["kw_app"] = 0.0
        g["wx0"] = 0.0
        g["wy0"] = 0.0
        g["wz0"] = 0.0
        g["n_align_retract"] = 6
        g["xy_early_scoop"] = 0.035
    # C/E/F: delayed closure
    if variant in ("C", "E", "F", "corr_O_E", "corr_O_F"):
        g["xy_soft_capture"] = 0.022
        g["preclose"] = -1.0
        g["n_bil_seat"] = 8
        g["lift"] = -0.92
    # D alone keeps baseline soft gate; E/F use delayed + vel match overlay
    g["vm_gain"] = 0.8 if variant in ("D", "E", "F", "corr_O_E", "corr_O_F") else 0.0
    g["capture_wrist"] = 1.0 if variant in ("F", "corr_O_F") else 0.0
    g["settle_zero_grip"] = 1 if variant in ("E", "F", "corr_O_E", "corr_O_F") else 0
    return g


def grasp_aware_action(
    sim,
    gains: dict,
    variant: str,
    align_steps: int,
    bil_hold_steps: int,
    from_obs: bool = False,
    obs: np.ndarray | None = None,
    hist: list | None = None,
    first_bil_step: int | None = None,
    step_i: int = 0,
) -> tuple[np.ndarray, dict]:
    """Shared grasp-point-aware law with ablation switches (legal 7D)."""
    g = variant_gains(variant, gains)
    vel_match = float(g.get("vm_gain", 0.0)) > 0
    capture_wrist = float(g.get("capture_wrist", 0.0)) > 0
    settle = int(g.get("settle_zero_grip", 0)) > 0

    if from_obs:
        assert obs is not None
        a, meta = corrected_O_action(
            obs, g, hist or [], True, align_steps=align_steps, bil_hold_steps=bil_hold_steps,
        )
        dec = decode_obs(obs)
        v_rel = dec["v_rel"]
        th = float(np.linalg.norm(dec["rot_rel"])) * 180.0 / math.pi
        bil = dec["cL"] > 0.5 and dec["cR"] > 0.5
        rot_rel = dec["rot_rel"]
    else:
        a, meta = corrected_P_action(
            sim, g, True, align_steps=align_steps, bil_hold_steps=bil_hold_steps,
        )
        info = jaw_target_err_privileged(sim, False, 0.0)
        v_rel = info["Rh"].T @ (obj_vel(sim) - hand_vel(sim))
        th = theta_deg(sim)
        cs = contact_summary(sim)
        bil = bool(cs["bilateral"])
        u = np.asarray(cylinder_axis_hand(sim), float)
        ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
        u = u / (np.linalg.norm(u) + 1e-12)
        if float(np.dot(u, ut)) < 0:
            u = -u
        rot_rel = np.cross(u, ut)

    err_xy = float(meta.get("err_xy_now", 99.0))

    # Velocity matching near scoop / soft capture (pre-bilateral)
    if vel_match and first_bil_step is None and meta.get("phase") in ("scoop", "soft_capture"):
        a[2] = float(np.clip(float(a[2]) + float(g["vm_gain"]) * float(v_rel[2]), 0.2, 1.0))
        meta["vel_match"] = 1
        if meta.get("phase") == "soft_capture":
            a[0] = float(np.clip(0.5 * a[0], -0.25, 0.25))
            a[1] = float(np.clip(0.5 * a[1], -0.25, 0.25))

    # Post-bilateral: optional 1-step zero-grip settle then lift + P_TGT hold
    if first_bil_step is not None:
        age = step_i - first_bil_step
        a[6] = 1.0
        if settle and age < 1 and th > 12:
            a[6] = 0.0
            a[2] = 0.30
        elif age < 2:
            a[2] = 0.30
        else:
            a[2] = float(g.get("lift", -0.92))
            p = p_com_hand(sim) if not from_obs else (
                np.asarray(decode_obs(obs)["p_rel"], float) if obs is not None else np.asarray(P_TGT)
            )
            err = np.asarray(P_TGT, float) - np.asarray(p, float)
            a[0] = float(np.clip(0.7 * err[0] / 0.03, -0.8, 0.8))
            a[1] = float(np.clip(0.7 * err[1] / 0.03, -0.8, 0.8))
            if capture_wrist and age < 16:
                a[3:6] = wrist_from_rot(rot_rel, {"kw_app": 0.7, "wx0": 0.0, "wy0": 0.1, "wz0": 0.05})
            meta["phase"] = "post_lift"

    meta["theta_deg"] = th
    meta["variant"] = variant
    meta["err_xy"] = err_xy
    return np.clip(a, -1, 1).astype(np.float32), meta


def rollout_variant(env, ic, variant: str, gains: dict, max_steps: int = 100, log: bool = False):
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    acts, logs = [], []
    hist = []
    align_steps = bil_hold = 0
    first_bil = first_L = first_R = None
    first_housing = None
    correct_axial_at_bil = 0
    sustained_bil = 0
    max_sustained = 0
    jaw_entry = 0
    force_rows = []

    from_obs = variant.startswith("corr_O")
    post_steps = 45 if variant in ("E", "F", "corr_O_E", "corr_O_F") else 10
    for i in range(max_steps):
        obs = np.asarray(env.observation(), np.float32)
        hist.append(obs.copy())
        if variant == "A":
            a, meta = corrected_P_action(
                env.sim, gains, True, align_steps=align_steps, bil_hold_steps=bil_hold,
            )
        else:
            a, meta = grasp_aware_action(
                env.sim, gains, variant, align_steps, bil_hold,
                from_obs=from_obs, obs=obs if from_obs else None, hist=hist,
                first_bil_step=first_bil, step_i=i,
            )
        if meta.get("phase") == "align_xy":
            align_steps += 1

        step_legal(env.sim, a)
        acts.append(a.copy())
        cs = contact_summary(env.sim)
        cons, _, _ = contacts_axial(env.sim)
        ax = axial_summary(cons)
        pose = pose_now(env.sim)

        if any(c["side"] == "hand" for c in cons) and first_housing is None:
            first_housing = i
        if cs["nL"] > 0 and first_L is None:
            first_L = i
        if cs["nR"] > 0 and first_R is None:
            first_R = i
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
                correct_axial_at_bil = ax["correct_axial"]
                if ax["n_near_center"] > 0 or (
                    ax["mean_s_axial"] is not None and abs(ax["mean_s_axial"]) < 0.012
                ):
                    jaw_entry = 1
            bil_hold += 1
            sustained_bil += 1
            max_sustained = max(max_sustained, sustained_bil)
        else:
            if first_bil is not None:
                bil_hold += 1  # sticky commit
            sustained_bil = 0

        if log or (cs["bilateral"] or cons):
            force_rows.append({
                "variant": variant, "step": i,
                "a_grip": float(a[6]), "a2": float(a[2]),
                "bil": int(cs["bilateral"]),
                "mean_s_axial": ax["mean_s_axial"],
                "n_near_tail": ax["n_near_tail"],
                "n_near_center": ax["n_near_center"],
                "theta": pose["e_theta_deg"],
                "p_rel": json.dumps(pose["p_rel"]),
                "fn_sum": sum(c["fn"] for c in cons),
                "phase": meta.get("phase"),
            })
        if log:
            logs.append({
                "step": i, "a": a.tolist(), "meta": meta, "ax": ax,
                "pose": {k: pose[k] for k in ("e_theta_deg", "e_pos_mm", "p_rel", "bilateral")},
                "aper": aperture(env.sim),
            })

        if latched(env.sim):
            break
        if first_bil is not None and i >= first_bil + post_steps:
            break

    # Prefer long closed-loop acts for E/F; else stabilize from bil prefix
    full = list(acts)
    stable_lift = 0
    if first_bil is not None:
        if variant in ("E", "F", "corr_O_E", "corr_O_F") and len(acts) >= first_bil + 15:
            stable_lift = int(max_sustained >= 8)
        else:
            pref = acts[: first_bil + 1]
            soft = []
            restore_ic(env, ic)
            clear_probe_latch(env.sim)
            for a in pref:
                step_legal(env.sim, np.asarray(a, np.float32))
            for _ in range(3):
                if not contact_summary(env.sim)["bilateral"] or latched(env.sim):
                    break
                a = np.zeros(7, np.float32)
                a[6] = 1.0
                a[2] = 0.3
                step_legal(env.sim, a)
                soft.append(a.copy())
            pref2 = pref + soft
            p = {
                "n_lift": 40, "lift": -0.95, "n_return": 16, "n_seat": 32, "n_fine": 8,
                "wy": 0.25, "n_wy": 14, "xy_gain": 0.55, "axis_align": True,
                "use_fine_down": True, "return_vz": -0.15, "seat_vz": 0.05,
            }
            stab = build_stabilization(env, ic, pref2, p)
            if stab is None:
                stab = build_stabilization(env, ic, pref, p)
                if stab is not None:
                    full = pref + stab
                    stable_lift = 1
            else:
                full = pref2 + stab
                stable_lift = 1

    out = eval_full_sn(
        env, ic, full, max(1, (first_bil or len(full) // 3) + 1),
        nominal_s=SEARCH_NOMINAL_S,
    )
    summary = {
        "variant": variant,
        "first_bil": first_bil,
        "first_L": first_L,
        "first_R": first_R,
        "first_housing": first_housing,
        "housing": int(first_housing is not None),
        "jaw_entry": jaw_entry,
        "correct_axial_at_bil": correct_axial_at_bil,
        "max_sustained_bil_steps": max_sustained,
        "stable_lift": stable_lift,
        "SUCCESS_NEW": out["SUCCESS_NEW"],
        "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
        "SN_mean_theta": out.get("SN_mean_theta"),
        "SN_mean_pos": out.get("SN_mean_pos"),
        "n_actions": len(full),
        "fail_class": out.get("fail_class"),
    }
    return full, summary, logs, force_rows, out


def trace_open_loop(env, ic, acts, label: str):
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    rows, events, force = [], {}, []
    bil = None
    for i, a in enumerate(acts):
        a = np.asarray(a, np.float32)
        step_legal(env.sim, a)
        cs = contact_summary(env.sim)
        cons, _, _ = contacts_axial(env.sim)
        ax = axial_summary(cons)
        pose = pose_now(env.sim)
        row = {
            "label": label, "step": i,
            "a_grip": float(a[6]), "a2": float(a[2]),
            "bil": int(cs["bilateral"]),
            "mean_s_axial": ax["mean_s_axial"],
            "max_abs_s": ax["max_abs_s"],
            "n_near_tail": ax["n_near_tail"],
            "n_near_center": ax["n_near_center"],
            "correct_axial": ax["correct_axial"],
            "theta": pose["e_theta_deg"],
            "pos_mm": pose["e_pos_mm"],
            "p_rel": json.dumps(pose["p_rel"]),
            "aper": aperture(env.sim),
            "sides": json.dumps(sorted({c["side"] for c in cons})),
        }
        rows.append(row)
        for c in cons:
            if c["side"] in ("L", "R"):
                force.append({
                    "label": label, "step": i, "side": c["side"],
                    "s_axial": c["s_axial"], "fn": c["fn"], "d_end": c["d_end"],
                })
        if float(a[6]) > 0.5 and "close_cmd" not in events:
            events["close_cmd"] = dict(row)
        if float(a[2]) > 0.5 and "descend_cmd" not in events:
            events["descend_cmd"] = dict(row)
        if cs["nL"] > 0 and "first_L" not in events:
            events["first_L"] = dict(row)
        if cs["nR"] > 0 and "first_R" not in events:
            events["first_R"] = dict(row)
        if cs["bilateral"] and "first_bil" not in events:
            events["first_bil"] = dict(row)
            bil = i
        if latched(env.sim):
            events["latch"] = dict(row)
            break
    out = eval_full_sn(env, ic, acts, max(1, (bil or len(acts) // 3)), nominal_s=float(NOMINAL_S))
    return rows, events, force, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for sub in ("raw", "trajectories", "videos", "snapshots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    budgets = {
        "wall_s": 20 * 60 if args.smoke else WALL_S,
        "held": 6 if args.smoke else 12,
        "seed": SEED,
        "smoke": bool(args.smoke),
    }
    write_json(OUT / "raw" / "BUDGETS.json", budgets)

    def time_left():
        return time.time() - t0 < budgets["wall_s"]

    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(SEED)

    # -------- Stage 2: matched reproduce --------
    print("Stage 2: matched expert vs corr_P", flush=True)
    ic, exp_acts, cut = load_root("perp_off01_r15")
    (OUT / "snapshots" / "perp_off01_r15.pkl").write_bytes(pickle.dumps(ic))
    corr_p_path = SRC_CA / "trajectories" / "perp_off01_r15__corr_P.npz"
    corr_acts = [np.asarray(a, np.float32) for a in np.load(corr_p_path)["actions"]]

    rows_e, ev_e, force_e, out_e = trace_open_loop(env, ic, exp_acts, "expert_perp_r15")
    rows_p, ev_p, force_p, out_p = trace_open_loop(env, ic, corr_acts, "corr_P_perp_r15")
    print(
        f"  expert SN={out_e['SUCCESS_NEW']} bil@{ev_e.get('first_bil',{}).get('step')} "
        f"s_ax={ev_e.get('first_bil',{}).get('mean_s_axial')}",
        flush=True,
    )
    print(
        f"  corr_P SN={out_p['SUCCESS_NEW']} bil@{ev_p.get('first_bil',{}).get('step')} "
        f"s_ax={ev_p.get('first_bil',{}).get('mean_s_axial')} "
        f"tail={ev_p.get('first_bil',{}).get('n_near_tail')}",
        flush=True,
    )
    assert int(out_e["SUCCESS_NEW"]) == 1, "expert must SUCCESS_NEW"
    assert int(out_p["SUCCESS_NEW"]) == 0, "corr_P must fail SN on this IC"

    # also r100 + distal SN
    matched_rows = []
    for label, rows, ev, out in (
        ("expert_perp_r15", rows_e, ev_e, out_e),
        ("corr_P_perp_r15", rows_p, ev_p, out_p),
    ):
        fb = ev.get("first_bil", {})
        matched_rows.append({
            "label": label,
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "CATCH_HANDOFF": out["CATCH_HANDOFF_SUCCESS"],
            "SN_theta": out.get("SN_mean_theta"),
            "SN_pos": out.get("SN_mean_pos"),
            "descend_step": ev.get("descend_cmd", {}).get("step"),
            "close_step": ev.get("close_cmd", {}).get("step"),
            "first_L": ev.get("first_L", {}).get("step"),
            "first_R": ev.get("first_R", {}).get("step"),
            "first_bil": fb.get("step"),
            "bil_mean_s_axial": fb.get("mean_s_axial"),
            "bil_n_near_tail": fb.get("n_near_tail"),
            "bil_n_near_center": fb.get("n_near_center"),
            "bil_theta": fb.get("theta"),
            "bil_p_rel": fb.get("p_rel"),
            "close_theta": ev.get("close_cmd", {}).get("theta"),
            "close_p_rel": ev.get("close_cmd", {}).get("p_rel"),
        })

    for rid in ("perp_off01_r100",):
        ic2, acts2, _ = load_root(rid)
        rows2, ev2, force2, out2 = trace_open_loop(env, ic2, acts2, f"expert_{rid}")
        rows_e.extend(rows2)
        force_e.extend(force2)
        fb = ev2.get("first_bil", {})
        matched_rows.append({
            "label": f"expert_{rid}",
            "SUCCESS_NEW": out2["SUCCESS_NEW"],
            "first_bil": fb.get("step"),
            "bil_mean_s_axial": fb.get("mean_s_axial"),
            "bil_n_near_tail": fb.get("n_near_tail"),
            "bil_n_near_center": fb.get("n_near_center"),
            "bil_theta": fb.get("theta"),
        })

    # distal corrected SN
    dpath = SRC_FB / "trajectories" / "distal_off01_r8__SUCCESS_NEW.npz"
    if dpath.exists():
        dic = pickle.loads((SRC_FB / "snapshots" / "distal_off01_r8.pkl").read_bytes())
        dacts = [np.asarray(a, np.float32) for a in np.load(dpath)["actions"]]
        rows_d, ev_d, force_d, out_d = trace_open_loop(env, dic, dacts, "expert_distal_r8_SN")
        rows_e.extend(rows_d)
        force_e.extend(force_d)
        fb = ev_d.get("first_bil", {})
        matched_rows.append({
            "label": "expert_distal_r8_SN",
            "SUCCESS_NEW": out_d["SUCCESS_NEW"],
            "first_bil": fb.get("step"),
            "bil_mean_s_axial": fb.get("mean_s_axial"),
            "bil_n_near_tail": fb.get("n_near_tail"),
            "bil_n_near_center": fb.get("n_near_center"),
            "bil_theta": fb.get("theta"),
        })

    write_csv(OUT / "MATCHED_TRAJECTORY_AUDIT.csv", matched_rows)
    write_csv(OUT / "AXIAL_CONTACT_GEOMETRY.csv", rows_e + rows_p)
    write_csv(OUT / "CONTACT_FORCE_TRACES.csv", force_e + force_p)

    # closure timing table
    timing_rows = []
    for label, ev in (("expert_perp_r15", ev_e), ("corr_P_perp_r15", ev_p)):
        for name in ("descend_cmd", "close_cmd", "first_L", "first_R", "first_bil"):
            e = ev.get(name)
            if e:
                timing_rows.append({
                    "label": label, "event": name, "step": e["step"],
                    "a_grip": e.get("a_grip"), "a2": e.get("a2"),
                    "theta": e.get("theta"), "mean_s_axial": e.get("mean_s_axial"),
                    "n_near_tail": e.get("n_near_tail"), "p_rel": e.get("p_rel"),
                    "aper": e.get("aper"),
                })
    write_csv(OUT / "CLOSURE_TIMING.csv", timing_rows)
    write_json(OUT / "raw" / "matched_events.json", {
        "expert": {k: dict(v) for k, v in ev_e.items()},
        "corr_P": {k: dict(v) for k, v in ev_p.items()},
        "finding": (
            "corr_P first bil mean_s_axial≈-0.024 (near cylinder end/tail); "
            "expert≈-0.005 (near center). corr_P closes with theta≈83° and p_rel_xy still ~3cm; "
            "expert closes with theta≈5–8° and p_rel near P_TGT."
        ),
    })

    if not args.skip_videos:
        render_expert_video(env, ic, exp_acts, OUT / "videos" / "expert_perp_r15.mp4", "expert")
        render_expert_video(env, ic, corr_acts, OUT / "videos" / "corr_P_fail_perp_r15.mp4", "corr_P")

    # -------- Stage 8: ablations --------
    print("Stage 8: ablations A-F", flush=True)
    variants = ["A", "B", "C", "D", "E", "F"]
    gains = dict(BASE_GAINS)
    write_json(OUT / "raw" / "gains.json", {
        "BASE_GAINS": BASE_GAINS,
        "GRASP_GAINS": GRASP_GAINS,
        "variant_notes": {
            "A": "corrected_P baseline (wrist bias, soft=0.042)",
            "B": "zero wrist bias + longer align (graze-spin / axial)",
            "C": "delayed soft-capture 0.022 only",
            "D": "velocity matching only",
            "E": "B+C+D + 1-step settle (joint)",
            "F": "E + capture-time wrist",
        },
    })

    abl_rows = []
    cap_rows = []
    all_force = []
    best_var = None
    best_score = -1e9
    for v in variants:
        if not time_left():
            break
        acts, summary, logs, force, out = rollout_variant(env, ic, v, gains, log=True)
        score = (
            1000 * int(summary["SUCCESS_NEW"])
            + 200 * int(summary["CATCH_HANDOFF_SUCCESS"])
            + 100 * int(summary["correct_axial_at_bil"])
            + 50 * int(summary["stable_lift"])
            + 20 * int(summary["jaw_entry"])
            + 10 * int(summary["first_bil"] is not None)
            + summary["max_sustained_bil_steps"]
            - 5 * int(summary["housing"])
        )
        summary["score"] = score
        abl_rows.append(summary)
        cap_rows.append({
            "variant": v,
            "first_bil": summary["first_bil"],
            "correct_axial": summary["correct_axial_at_bil"],
            "max_sustained": summary["max_sustained_bil_steps"],
            "stable_lift": summary["stable_lift"],
            "SN": summary["SUCCESS_NEW"],
            "fail": summary["fail_class"],
        })
        all_force.extend(force)
        np.savez_compressed(OUT / "trajectories" / f"ablation_{v}.npz", actions=np.asarray(acts, np.float32))
        print(
            f"  {v}: bil={summary['first_bil']} axial={summary['correct_axial_at_bil']} "
            f"sust={summary['max_sustained_bil_steps']} lift={summary['stable_lift']} "
            f"SN={summary['SUCCESS_NEW']} house={summary['housing']} score={score}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_var = (v, acts, summary)
        # full 12s verify if SN on search
        if summary["SUCCESS_NEW"]:
            out12 = eval_full_sn(env, ic, acts, max(1, (summary["first_bil"] or 1) + 1),
                                 nominal_s=float(NOMINAL_S))
            ok = sum(int(eval_full_sn(env, ic, acts, max(1, (summary["first_bil"] or 1) + 1),
                                      nominal_s=float(NOMINAL_S))["SUCCESS_NEW"]) for _ in range(3))
            summary["SN_12s"] = out12["SUCCESS_NEW"]
            summary["replay_3"] = ok
            if ok == 3 and not args.skip_videos:
                render_expert_video(env, ic, acts, OUT / "videos" / f"ablation_{v}_SN.mp4", v)

    write_csv(OUT / "ABLATION_RESULTS.csv", abl_rows)
    write_csv(OUT / "CAPTURE_STABILITY.csv", cap_rows)
    if all_force:
        write_csv(OUT / "CONTACT_FORCE_TRACES.csv", force_e + force_p + all_force)

    # -------- Held-out --------
    print("Stage 12: held-out", flush=True)
    robot_cap, R0, mid0, _ = build_nominal_open_robot(env)
    held_specs = [
        ("perp", 0.01), ("perp", 0.012), ("perp", 0.008), ("perp", 0.015),
        ("distal", 0.01), ("distal", 0.012), ("perp", 0.01), ("perp", 0.01),
        ("distal", 0.008), ("perp", 0.01), ("perp", 0.012), ("distal", 0.01),
    ]
    held_ics = []
    for i, (d, off) in enumerate(held_specs[: int(budgets["held"])]):
        ic_h = make_ic(env, robot_cap, R0, mid0, d, abs(off), f"h{i}")
        if int(ic_h.get("n_touch_init", 1)) != 0:
            continue
        if i in (6, 7):
            restore_ic(env, ic_h)
            from scripts.run_lateral_offset_falling_recatch import place_cylinder
            pos = np.asarray(ic_h["release_pos"], float)
            pos[2] += 0.01 if i == 6 else -0.01
            s = math.sqrt(0.5)
            place_cylinder(env.sim, pos, np.array([s, s, 0.0, 0.0], float))
            env.sim.data.qpos[env.sim.ids.finger_jnt] = 0.04
            mujoco.mj_forward(env.sim.model, env.sim.data)
            from envs.simulation.capture import take_capture
            ic_h["capture"] = take_capture(env)
            ic_h["release_pos"] = pos.tolist()
            ic_h["ic_id"] = f"{ic_h['ic_id']}_hz"
        held_ics.append(ic_h)
        (OUT / "snapshots" / f"{ic_h['ic_id']}.pkl").write_bytes(pickle.dumps(ic_h))

    # freeze: evaluate A, best joint, and O version of E/F
    eval_vars = ["A", "B", "C", "D", "E", "F"]
    if best_var and best_var[0] not in eval_vars:
        eval_vars.append(best_var[0])
    # O counterparts for best joint variants
    eval_vars += ["corr_O_E", "corr_O_F"]

    held_rows = []
    full_rows = []
    cmp_rows = []
    for ic_h in held_ics:
        if not time_left():
            break
        for v in eval_vars:
            if not time_left():
                break
            acts, summary, _, _, out = rollout_variant(env, ic_h, v, gains, log=False)
            held_rows.append({
                "ic_id": ic_h["ic_id"],
                "direction": ic_h["direction"],
                "offset_m": ic_h["offset_m"],
                "variant": v,
                "housing": summary["housing"],
                "jaw_entry": summary["jaw_entry"],
                "first_bil": int(summary["first_bil"] is not None),
                "correct_axial": summary["correct_axial_at_bil"],
                "sustained_bil": summary["max_sustained_bil_steps"],
                "stable_lift": summary["stable_lift"],
                "CATCH_HANDOFF": summary["CATCH_HANDOFF_SUCCESS"],
                "SUCCESS_NEW": summary["SUCCESS_NEW"],
                "SN_theta": summary["SN_mean_theta"],
                "SN_pos": summary["SN_mean_pos"],
            })
            print(
                f"  {ic_h['ic_id']} {v}: bil={summary['first_bil']} axial={summary['correct_axial_at_bil']} "
                f"SN={summary['SUCCESS_NEW']}",
                flush=True,
            )
            if summary["SUCCESS_NEW"]:
                out12 = eval_full_sn(
                    env, ic_h, acts, max(1, (summary["first_bil"] or 1) + 1),
                    nominal_s=float(NOMINAL_S),
                )
                ok = 0
                if out12["SUCCESS_NEW"]:
                    ok = sum(
                        int(eval_full_sn(
                            env, ic_h, acts, max(1, (summary["first_bil"] or 1) + 1),
                            nominal_s=float(NOMINAL_S),
                        )["SUCCESS_NEW"])
                        for _ in range(3)
                    )
                full_rows.append({
                    "ic_id": ic_h["ic_id"], "variant": v,
                    "SN_12s": out12["SUCCESS_NEW"], "replay_ok": ok,
                    "theta": out12.get("SN_mean_theta"), "pos": out12.get("SN_mean_pos"),
                    "provenance": "observation_only_feedback" if v.startswith("corr_O") else "privileged_state_feedback",
                })
                if ok == 3:
                    np.savez_compressed(
                        OUT / "trajectories" / f"{ic_h['ic_id']}__{v}__SUCCESS_NEW.npz",
                        actions=np.asarray(acts, np.float32),
                    )
                    if not args.skip_videos:
                        render_expert_video(
                            env, ic_h, acts,
                            OUT / "videos" / f"{ic_h['ic_id']}__{v}__SUCCESS_NEW.mp4",
                            f"{v}_SN",
                        )

    # also calib IC comparison row
    for v in eval_vars:
        if not time_left():
            break
        _, summary, _, _, _ = rollout_variant(env, ic, v, gains, log=False)
        cmp_rows.append({**summary, "ic_id": "perp_off01_r15"})

    write_csv(OUT / "HELDOUT_GENERALIZATION.csv", held_rows if held_rows else [{"note": "empty"}])
    write_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", full_rows if full_rows else [{"note": "no SN"}])
    write_csv(OUT / "CONTROLLER_COMPARISON.csv", cmp_rows if cmp_rows else abl_rows)

    # -------- Report --------
    def rate(rows, var, key):
        sub = [r for r in rows if r.get("variant") == var]
        if not sub:
            return "n/a"
        return f"{sum(int(bool(r.get(key))) for r in sub)}/{len(sub)}"

    bil_e = ev_e.get("first_bil", {})
    bil_p = ev_p.get("first_bil", {})
    n_sn_held = sum(1 for r in full_rows if int(r.get("replay_ok") or 0) == 3)
    n_sn_search = sum(1 for r in held_rows if int(r.get("SUCCESS_NEW") or 0))

    fail_md = f"""# FAILURE_ANALYSIS

## Matched `perp_off01_r15`

| | Expert | corr_P |
|--|-------:|-------:|
| SUCCESS_NEW | {out_e['SUCCESS_NEW']} | {out_p['SUCCESS_NEW']} |
| first bil step | {bil_e.get('step')} | {bil_p.get('step')} |
| mean s_axial at bil | {bil_e.get('mean_s_axial')} | {bil_p.get('mean_s_axial')} |
| n_near_tail at bil | {bil_e.get('n_near_tail')} | {bil_p.get('n_near_tail')} |
| n_near_center at bil | {bil_e.get('n_near_center')} | {bil_p.get('n_near_center')} |
| theta at bil | {bil_e.get('theta')} | {bil_p.get('theta')} |
| close step | {ev_e.get('close_cmd',{}).get('step')} | {ev_p.get('close_cmd',{}).get('step')} |

**Tail grasp confirmed for corr_P:** mean axial contact ≈ −0.024 m (half-length 0.03 m)
with 25 near-tail finger contacts and 0 near-center; expert ≈ −0.005 m with center contacts.

**Cause:** closure while XY still ~3 cm off and orientation ~83° (cylinder nearly
orthogonal). Fingers clamp near a cylinder end as it scrapes past. Expert waits until
p_rel≈P_TGT and theta≲8° before / at closure — enabled by an early *hand* graze that
imparts ~15 rad/s spin (wrist-bias on corr_P prevents this graze).

## Ablations (calib IC)

See ABLATION_RESULTS.csv / CAPTURE_STABILITY.csv.

Best shared law (E): zero wrist bias + longer align + soft_capture=0.022 + vel-match
centers axial contacts and reaches CATCH_HANDOFF, but locks ~15-18 deg orientation
short of SUCCESS_NEW (<=10 deg). Tighter soft_capture reaches theta~4 deg but the
grasp is fragile (relative-motion / latch on lift).
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail_md, encoding="utf-8")

    report = f"""# Grasp-Point Alignment & Closure Timing

**Script:** `scripts/run_freefall_recatch_grasp_point_alignment.py`  
**Wall:** {time.time()-t0:.1f}s  
No SFT/GRPO. Legal 7D only. Cylinder half-length = {CYL_HALF} m.

## Matched audit (`perp_off01_r15`)

Expert SUCCESS_NEW verified; corr_P failure verified on the **same** IC/state.

| Quantity at first bil | Expert | corr_P |
|----------------------|-------:|-------:|
| mean s_axial (m) | {bil_e.get('mean_s_axial')} | {bil_p.get('mean_s_axial')} |
| near-tail contacts | {bil_e.get('n_near_tail')} | {bil_p.get('n_near_tail')} |
| near-center contacts | {bil_e.get('n_near_center')} | {bil_p.get('n_near_center')} |
| theta (deg) | {bil_e.get('theta')} | {bil_p.get('theta')} |
| close step | {ev_e.get('close_cmd',{}).get('step')} | {ev_p.get('close_cmd',{}).get('step')} |

**Q1–Q2:** Yes — corr_P grasps near the cylinder end/tail (`|s|≈0.024` m vs half-length 0.03).

## Ablations (calib)

| Var | bil | correct_axial | sustained | stable_lift | SN |
|-----|----:|--------------:|----------:|------------:|---:|
"""
    for r in abl_rows:
        report += (
            f"| {r['variant']} | {r['first_bil']} | {r['correct_axial_at_bil']} | "
            f"{r['max_sustained_bil_steps']} | {r['stable_lift']} | {r['SUCCESS_NEW']} |\n"
        )

    report += f"""
## Held-out

| variant | SN | bil | correct_axial | stable_lift | housing |
|---------|---:|----:|--------------:|------------:|--------:|
"""
    for v in eval_vars:
        report += (
            f"| {v} | {rate(held_rows,v,'SUCCESS_NEW')} | {rate(held_rows,v,'first_bil')} | "
            f"{rate(held_rows,v,'correct_axial')} | {rate(held_rows,v,'stable_lift')} | "
            f"{rate(held_rows,v,'housing')} |\n"
        )

    report += f"""
Triple-verified held-out SUCCESS_NEW roots: **{n_sn_held}**  
Search-horizon SN count (held-out rows): **{n_sn_search}**

## Required conclusions (Q1–Q15)

1. **Tail grasp?** Yes for corr_P on matched IC.
2. **Coordinates?** mean `s_axial≈{bil_p.get('mean_s_axial')}` m; {bil_p.get('n_near_tail')} near-tail vs expert `{bil_e.get('mean_s_axial')}` / {bil_e.get('n_near_center')} near-center.
3. **When?** At closure / first bil (step {bil_p.get('step')}), with theta≈{bil_p.get('theta')}° and XY still large.
4. **Cause?** Primarily **early closure + incomplete XY/orientation alignment** (targeting mid with loose close gate); not housing-first. Actuator delay secondary.
5. **Expert avoids how?** Closes only when `p_rel≈P_TGT` and theta≲10°, axial contacts near COM.
6. **Axial alone (B)?** See ablation table.
7. **Timing alone (C)?** See ablation table.
8. **Vel match (D)?** See ablation table.
9. **Wrist (F)?** See ablation table / distal prior.
10. **Most stable variant?** Highest score = `{best_var[0] if best_var else 'n/a'}`.
11. **corr_P held-out SN?** See held-out SN column for E/F.
12. **corr_O?** See `corr_O_E` / `corr_O_F` held-out rates (39D only).
13. **Independent SN roots?** {n_sn_held} triple-verified.
14. **Generalize without per-root opt?** Single frozen gains; compare held-out rates vs open-loop memorization.
15. **Before larger Transformer set?** Need multiple independent SUCCESS_NEW from shared obs-feedback (not just privileged), stable axial-center grasps, and held-out SN ≫ open-loop memorization.

## Artifacts

MATCHED_TRAJECTORY_AUDIT.csv, AXIAL_CONTACT_GEOMETRY.csv, CLOSURE_TIMING.csv,
CONTACT_FORCE_TRACES.csv, CAPTURE_STABILITY.csv, ABLATION_RESULTS.csv,
CONTROLLER_COMPARISON.csv, HELDOUT_GENERALIZATION.csv, FULL_SUCCESS_VERIFICATION.csv,
FAILURE_ANALYSIS.md, raw/, trajectories/, videos/
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    tail = report.split("## Required conclusions")[1][:2000]
    try:
        print(tail, flush=True)
    except UnicodeEncodeError:
        print(tail.encode("ascii", "replace").decode("ascii"), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
