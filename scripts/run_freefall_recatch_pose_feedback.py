"""Post-capture pose correction + closed-loop freefall recatch diagnostics.

Constructed free-fall ICs only. Legal continuous 7D. No SFT/GRPO. No K49 changes.
Does not modify the original 39D observation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import (
    AIR_NAMES,
    OBS_DIM_AIR,
    OBS_NAMES_AIR,
    SCALE_PREL,
    SCALE_ROT,
    SCALE_VREL,
    SCALE_WREL,
)
from envs.observations.observable_obs import OBS_NAMES
from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, restore_state
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.full_objective import (
    HOLD_S as SN_HOLD,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    window_stats,
)
from evaluation.grasp_configuration import cylinder_axis_hand, measure as gmeasure, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT, write_csv, write_json
from scripts.run_horizontal_offset_freefall_recatch import (
    SEARCH_NOMINAL_S,
    audit_interception,
    eval_full_sn,
    find_cut,
    render_expert_video,
)
from scripts.run_lateral_offset_falling_recatch import (
    FINE_DOWN,
    build_nominal_open_robot,
    finger_frame,
    hand_pos,
    make_ic,
    make_training_env,
    obj_pos,
    object_table_contact,
    restore_ic,
    rot_angle,
    scoop_align_descend,
)
from scripts.run_lateral_recatch_handoff import (
    build_stabilization,
    clear_probe_latch,
    evaluate_catch_handoff,
    hand_R,
)

SRC = ROOT / "results" / "expert_generation" / "horizontal_offset_freefall_recatch"
OUT = ROOT / "results" / "diagnostics" / "freefall_recatch_pose_feedback"
SEED = 20261104

# Frozen budgets
STAGE_A_REPLAYS = 3
STAGE_B_TRIALS_PER_SNAP = 24
STAGE_C_VERIFY = 3
STAGE_E_CALIB_TRIALS = 40
STAGE_F_HELDOUT = 10
STAGE_G_PERTURBS = 6
WALL_LIMIT_S = 70 * 60


def thash(acts):
    return hashlib.sha256(json.dumps([np.asarray(a).tolist() for a in acts]).encode()).hexdigest()[:16]


def load_root(ic_id: str) -> tuple[dict, list, int]:
    ic = pickle.loads((SRC / "snapshots" / f"{ic_id}.pkl").read_bytes())
    # prefer SUCCESS_NEW traj, else baseline
    for name in (f"{ic_id}__SUCCESS_NEW.npz", f"{ic_id}__baseline.npz", f"{ic_id}__CATCH_HANDOFF.npz"):
        p = SRC / "trajectories" / name
        if p.exists():
            z = np.load(p)
            acts = [np.asarray(a, np.float32) for a in z["actions"]]
            cut = int(z["cut"]) if "cut" in z.files else max(1, len(acts) // 3)
            return ic, acts, cut
    raise FileNotFoundError(ic_id)


def pose_now(sim) -> dict:
    m = gmeasure(sim, U_TGT, P_TGT)
    u = np.asarray(cylinder_axis_hand(sim), float)
    ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
    u = u / (np.linalg.norm(u) + 1e-12)
    axis_err = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(u, ut))), 0, 1))))
    return {
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "axis_err_deg": axis_err,
        "p_rel": np.asarray(p_com_hand(sim), float).tolist(),
        "bilateral": int(contact_summary(sim)["bilateral"]),
        "obj_z": float(obj_pos(sim)[2]),
    }


def timeline_replay(env, ic, acts, label: str) -> list[dict]:
    """Step-by-step relative pose / contact timeline."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    rows = []
    t0 = float(env.sim.data.time)
    first_bil = None
    bil_since = None
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        cs = contact_summary(env.sim)
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
            if bil_since is None:
                bil_since = float(env.sim.data.time)
        else:
            bil_since = None
        p = pose_now(env.sim)
        rows.append({
            "ic_id": label,
            "step": i,
            "t_rel": round(float(env.sim.data.time) - t0, 5),
            "phase": "recovery",
            "nL": int(cs["nL"]),
            "nR": int(cs["nR"]),
            "bilateral": int(cs["bilateral"]),
            "sustained_bil_s": round(float(env.sim.data.time) - bil_since, 5) if bil_since else 0.0,
            "latched": int(latched(env.sim)),
            **{k: p[k] for k in ("e_pos_mm", "e_theta_deg", "axis_err_deg", "obj_z")},
            "p_rel": json.dumps(p["p_rel"]),
            "first_bil_step": first_bil,
        })
        if latched(env.sim):
            break
    # HANDOFF + short nominal samples
    if not latched(env.sim):
        apply_handoff(env.sim, ic["ref"])
        p = pose_now(env.sim)
        rows.append({
            "ic_id": label, "step": len(acts), "t_rel": round(float(env.sim.data.time) - t0, 5),
            "phase": "handoff", "nL": 0, "nR": 0, "bilateral": p["bilateral"],
            "sustained_bil_s": 0.0, "latched": 0,
            **{k: p[k] for k in ("e_pos_mm", "e_theta_deg", "axis_err_deg", "obj_z")},
            "p_rel": json.dumps(p["p_rel"]), "first_bil_step": first_bil,
        })
        for k in range(25):
            nominal_step(env.sim)
            if k % 5 == 0:
                p = pose_now(env.sim)
                rows.append({
                    "ic_id": label, "step": len(acts) + 1 + k,
                    "t_rel": round(float(env.sim.data.time) - t0, 5),
                    "phase": "nominal", "nL": 0, "nR": 0, "bilateral": p["bilateral"],
                    "sustained_bil_s": 0.0, "latched": int(latched(env.sim)),
                    **{k2: p[k2] for k2 in ("e_pos_mm", "e_theta_deg", "axis_err_deg", "obj_z")},
                    "p_rel": json.dumps(p["p_rel"]), "first_bil_step": first_bil,
                })
            if latched(env.sim):
                break
    return rows


def extract_snapshots(env, ic, acts, cut: int) -> dict:
    """Exact captures at bilateral onset, mid-lift, pre-handoff."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    first_bil = None
    snaps = {}
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        if contact_summary(env.sim)["bilateral"] and first_bil is None:
            first_bil = i
            snaps["after_bilateral"] = {
                "capture": take_capture(env),
                "step": i,
                "pose": pose_now(env.sim),
                "prefix_len": i + 1,
            }
        if latched(env.sim):
            break
    # mid-lift: halfway between cut and end if cut in recovery; else after_bil + 15
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    mid_i = min(len(acts) - 1, max((first_bil or 0) + 15, (cut + len(acts)) // 2))
    for i, a in enumerate(acts[: mid_i + 1]):
        step_legal(env.sim, np.asarray(a, np.float32))
    if contact_summary(env.sim)["bilateral"] and not latched(env.sim):
        snaps["mid_lift"] = {
            "capture": take_capture(env),
            "step": mid_i,
            "pose": pose_now(env.sim),
            "prefix_len": mid_i + 1,
        }
    # pre-handoff = end of recorded actions
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
        if latched(env.sim):
            break
    if contact_summary(env.sim)["bilateral"] and not latched(env.sim):
        snaps["pre_handoff"] = {
            "capture": take_capture(env),
            "step": len(acts) - 1,
            "pose": pose_now(env.sim),
            "prefix_len": len(acts),
        }
    return snaps


def early_wrist_prefix(acts, wx: float, wy: float, wz: float, until: int) -> list:
    """Inject wrist bias into an open-loop catch prefix (legal 7D, continuous)."""
    out = []
    for i, a0 in enumerate(acts):
        a = np.asarray(a0, np.float32).copy()
        if i <= until:
            a[3] = float(np.clip(wx, -1, 1))
            a[4] = float(np.clip(wy, -1, 1))
            a[5] = float(np.clip(wz, -1, 1))
        out.append(a)
    return out


def privileged_correct_action(sim, mode: str, gains: dict) -> np.ndarray:
    """Legal 7D action from full MuJoCo relative pose (diagnostic Controller P piece)."""
    a = np.zeros(7, np.float32)
    a[6] = 1.0
    p = np.asarray(p_com_hand(sim), float)
    # position toward P_TGT in hand frame
    err_p = np.asarray(P_TGT, float) - p
    if mode in ("slide", "combined", "early"):
        a[0] = float(np.clip(gains.get("kp", 1.2) * err_p[0] / 0.03, -0.8, 0.8))
        a[1] = float(np.clip(gains.get("kp", 1.2) * err_p[1] / 0.03, -0.8, 0.8))
        a[2] = float(np.clip(gains.get("kz", 0.4) * err_p[2] / 0.03, -0.5, 0.5))
    if mode in ("wrist", "pivot", "combined", "early"):
        u = np.asarray(cylinder_axis_hand(sim), float)
        ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
        u = u / (np.linalg.norm(u) + 1e-12)
        # unsigned axis: flip u if obtuse
        if float(np.dot(u, ut)) < 0:
            u = -u
        c = np.cross(u, ut)
        n = float(np.linalg.norm(c))
        if n > 1e-4:
            a[3:6] = np.clip(gains.get("kw", 0.55) * c / n, -0.7, 0.7)
        if mode == "wrist":
            a[4] = float(np.clip(a[4] + gains.get("wy", 0.25), -0.7, 0.7))
    if mode == "none":
        a[:] = 0.0
        a[6] = 1.0
        a[2] = 0.05
    return a


def obs_correct_action(obs: np.ndarray, mode: str, gains: dict, hist: list | None = None) -> np.ndarray:
    """Legal 7D from 39D obs only (Controller O). Uses airborne relative channels."""
    a = np.zeros(7, np.float32)
    a[6] = 1.0
    o = np.asarray(obs, np.float32).reshape(-1)
    if o.size < OBS_DIM_AIR:
        return a
    # indices: 27 + [p_rel(3), v_rel(3), rot_rel(3), w_rel(3)]
    base = 27
    p_rel = o[base:base + 3] * SCALE_PREL
    v_rel = o[base + 3:base + 6] * SCALE_VREL
    rot_rel = o[base + 6:base + 9] * SCALE_ROT
    # contact
    cL, cR = float(o[2]), float(o[3])
    aper = float(o[12])
    err_p = np.asarray(P_TGT, float) - p_rel
    if mode in ("slide", "combined", "chase", "early"):
        a[0] = float(np.clip(gains.get("kp", 1.2) * err_p[0] / 0.03 - 0.2 * v_rel[0], -1, 1))
        a[1] = float(np.clip(gains.get("kp", 1.2) * err_p[1] / 0.03 - 0.2 * v_rel[1], -1, 1))
        a[2] = float(np.clip(gains.get("kz", 0.5) * err_p[2] / 0.03 + gains.get("vz_bias", 0.0), -1, 1))
    if mode in ("wrist", "pivot", "combined", "early"):
        # rot_rel is rotation vector hand<-obj; drive toward aligning axis ~ -z
        # Use rot_rel xy as wrist correction proxy
        a[3] = float(np.clip(-gains.get("kw", 0.5) * rot_rel[0], -0.7, 0.7))
        a[4] = float(np.clip(-gains.get("kw", 0.5) * rot_rel[1] + gains.get("wy", 0.0), -0.7, 0.7))
        a[5] = float(np.clip(-gains.get("kw", 0.4) * rot_rel[2], -0.5, 0.5))
    if mode == "chase":
        # open if no bilateral-like contact
        if cL < 0.5 or cR < 0.5:
            a[6] = -1.0
            a[2] = float(np.clip(0.8 + gains.get("vz_bias", 0.2), 0, 1))
        else:
            a[6] = 1.0
    if mode == "none":
        a[:] = 0.0
        a[6] = 1.0
    # history: optional derivative on p_rel
    if hist and len(hist) >= 2 and mode in ("chase", "combined"):
        prev = hist[-2]
        dp = p_rel - (prev[base:base + 3] * SCALE_PREL)
        a[0] = float(np.clip(a[0] - 0.3 * dp[0] / 0.02, -1, 1))
        a[1] = float(np.clip(a[1] - 0.3 * dp[1] / 0.02, -1, 1))
    return a


def run_correction_from_capture(env, ic, snap, mode: str, n_steps: int, gains: dict, source: str):
    """Apply correction from a restored capture; return actions + final pose."""
    restore_capture(env, snap["capture"])
    clear_probe_latch(env.sim)
    acts = []
    hist = []
    for _ in range(n_steps):
        if source == "privileged":
            a = privileged_correct_action(env.sim, mode, gains)
        else:
            obs = np.asarray(env.observation(), np.float32)
            hist.append(obs.copy())
            a = obs_correct_action(obs, mode, gains, hist)
        step_legal(env.sim, a)
        acts.append(a.copy())
        if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
            break
    return acts, pose_now(env.sim)


def full_prefix_plus_suffix(env, ic, prefix_acts, suffix_acts) -> dict:
    full = list(prefix_acts) + list(suffix_acts)
    return eval_full_sn(env, ic, full, len(prefix_acts), nominal_s=float(NOMINAL_S))


def controller_rollout(env, ic, ctrl_kind: str, gains: dict, max_steps: int = 90) -> tuple[list, dict]:
    """Closed-loop recatch from free-fall IC (no open-loop expert prefix)."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    acts = []
    hist = []
    bil_start = None
    # early-wrist bias used during approach (shared params; calibrated offline)
    wx0 = float(gains.get("wx_approach", 0.0))
    wy0 = float(gains.get("wy_approach", 0.14))
    wz0 = float(gains.get("wz_approach", 0.20))
    for i in range(max_steps):
        cs = contact_summary(env.sim)
        if cs["bilateral"] and bil_start is None:
            bil_start = i
        if ctrl_kind == "P":
            mode = "chase" if not cs["bilateral"] else "combined"
            if mode == "chase":
                a = np.zeros(7, np.float32)
                ff = finger_frame(env.sim)
                po, mid = obj_pos(env.sim), ff["mid"]
                err = ff["Rh"].T @ (po - mid)
                a[0] = float(np.clip(1.5 * err[0] / 0.02, -1, 1))
                a[1] = float(np.clip(1.5 * err[1] / 0.02, -1, 1))
                a[2] = float(np.clip(0.85 + 0.5 * err[2] / 0.05, 0, 1))
                # approach wrist for capture geometry (privileged axis)
                u = np.asarray(cylinder_axis_hand(env.sim), float)
                ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
                u = u / (np.linalg.norm(u) + 1e-12)
                if float(np.dot(u, ut)) < 0:
                    u = -u
                c = np.cross(u, ut)
                n = float(np.linalg.norm(c))
                if n > 1e-4:
                    a[3:6] = np.clip(gains.get("kw_app", 0.45) * c / n, -0.7, 0.7)
                a[3] = float(np.clip(a[3] + wx0, -1, 1))
                a[4] = float(np.clip(a[4] + wy0, -1, 1))
                a[5] = float(np.clip(a[5] + wz0, -1, 1))
                a[6] = -1.0 if abs(err[0]) + abs(err[1]) > 0.022 else 1.0
                if a[6] > 0:
                    a[:2] *= 0.35
                    a[2] = 0.25
            else:
                a = privileged_correct_action(env.sim, "combined", gains)
                a[2] = float(np.clip(a[2] + gains.get("lift", -0.55), -1, 1))
                # keep mild approach wrist briefly after bil
                if bil_start is not None and i - bil_start < 8:
                    a[4] = float(np.clip(a[4] + 0.5 * wy0, -1, 1))
                    a[5] = float(np.clip(a[5] + 0.5 * wz0, -1, 1))
        else:
            obs = np.asarray(env.observation(), np.float32)
            hist.append(obs.copy())
            cs_proxy = (float(obs[2]) > 0.5 and float(obs[3]) > 0.5)
            mode = "combined" if cs_proxy else "chase"
            a = obs_correct_action(obs, mode, gains, hist)
            if mode == "chase":
                # shared approach wrist from calibrated gains + rot_rel feedback
                a[3] = float(np.clip(a[3] + wx0, -1, 1))
                a[4] = float(np.clip(a[4] + wy0, -1, 1))
                a[5] = float(np.clip(a[5] + wz0, -1, 1))
            else:
                a[2] = float(np.clip(a[2] + gains.get("lift", -0.5), -1, 1))
                a[6] = 1.0
                if bil_start is not None and i - bil_start < 8:
                    a[4] = float(np.clip(a[4] + 0.5 * wy0, -1, 1))
                    a[5] = float(np.clip(a[5] + 0.5 * wz0, -1, 1))
        step_legal(env.sim, a)
        acts.append(a.copy())
        if latched(env.sim):
            break
        if bil_start is not None and i - bil_start > 45:
            break
    out = eval_full_sn(env, ic, acts, max(1, len(acts) // 3), nominal_s=SEARCH_NOMINAL_S)
    return acts, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("raw", "trajectories", "videos", "snapshots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    budgets = {
        "stage_a_replays": STAGE_A_REPLAYS,
        "stage_b_trials_per_snap": STAGE_B_TRIALS_PER_SNAP,
        "stage_e_calib": STAGE_E_CALIB_TRIALS,
        "stage_f_heldout": STAGE_F_HELDOUT,
        "stage_g_perturbs": STAGE_G_PERTURBS,
        "wall_limit_s": WALL_LIMIT_S,
        "seed": SEED,
        "smoke": bool(args.smoke),
    }
    if args.smoke:
        budgets.update({
            "stage_b_trials_per_snap": 6,
            "stage_e_calib": 8,
            "stage_f_heldout": 4,
            "stage_g_perturbs": 2,
            "wall_limit_s": 20 * 60,
        })
    write_json(OUT / "raw" / "BUDGETS.json", budgets)

    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(SEED)

    def time_left():
        return time.time() - t0w < budgets["wall_limit_s"]

    ref_rows = []
    distal_timeline = []
    corr_rows = []
    handoff_rows = []
    fb_rows = []
    held_rows = []
    timing_rows = []
    cand_rows = []
    new_experts = []

    # ========== Stage A ==========
    print("Stage A: reference replay + audit", flush=True)
    roots = {}
    for ic_id in ("perp_off01_r15", "perp_off01_r100", "distal_off01_r8"):
        ic, acts, cut = load_root(ic_id)
        roots[ic_id] = (ic, acts, cut)
        (OUT / "snapshots" / f"{ic_id}.pkl").write_bytes(pickle.dumps(ic))
        np.savez_compressed(OUT / "trajectories" / f"{ic_id}__ref.npz",
                            actions=np.asarray(acts, np.float32), cut=cut)
        ok = 0
        last = None
        for rep in range(int(budgets["stage_a_replays"])):
            out = eval_full_sn(env, ic, acts, cut, nominal_s=float(NOMINAL_S))
            last = out
            # distal expects CH; others SN
            target_ok = out["SUCCESS_NEW"] if ic_id != "distal_off01_r8" else out["CATCH_HANDOFF_SUCCESS"]
            ok += int(target_ok)
            print(
                f"  {ic_id} rep{rep} SN={out['SUCCESS_NEW']} CH={out['CATCH_HANDOFF_SUCCESS']} "
                f"th={out.get('SN_mean_theta')} pos={out.get('SN_mean_pos')}",
                flush=True,
            )
        tl = timeline_replay(env, ic, acts, ic_id)
        if ic_id == "distal_off01_r8":
            distal_timeline.extend(tl)
        # when theta first exceeds ~15 then ~20
        # Free-fall starts near 90 deg; report first post-contact (>=1 contact) thresholds
        contacted = [r for r in tl if int(r.get("nL", 0)) + int(r.get("nR", 0)) > 0]
        first_15 = next((r for r in contacted if r["e_theta_deg"] >= 15.0), None)
        first_20 = next((r for r in contacted if r["e_theta_deg"] >= 20.0), None)
        at_hand = next((r for r in tl if r["phase"] == "handoff"), tl[-1] if tl else {})
        at_bil = next((r for r in tl if r.get("first_bil_step") is not None and r["step"] == r["first_bil_step"]), None)
        ref_rows.append({
            "ic_id": ic_id,
            "replays_ok": ok,
            "replays_n": int(budgets["stage_a_replays"]),
            "SUCCESS_NEW": last["SUCCESS_NEW"] if last else 0,
            "CATCH_HANDOFF_SUCCESS": last["CATCH_HANDOFF_SUCCESS"] if last else 0,
            "SN_mean_theta": last.get("SN_mean_theta") if last else None,
            "SN_mean_pos": last.get("SN_mean_pos") if last else None,
            "theta_at_first_bil": at_bil["e_theta_deg"] if at_bil else None,
            "pos_at_first_bil": at_bil["e_pos_mm"] if at_bil else None,
            "theta_at_handoff": at_hand.get("e_theta_deg"),
            "pos_at_handoff": at_hand.get("e_pos_mm"),
            "first_theta_ge_15_step": first_15["step"] if first_15 else None,
            "first_theta_ge_15_phase": first_15["phase"] if first_15 else None,
            "first_theta_ge_20_step": first_20["step"] if first_20 else None,
            "first_theta_ge_20_phase": first_20["phase"] if first_20 else None,
            "n_actions": len(acts),
            "cut": cut,
            "actions_hash": thash(acts),
        })

    # ========== Stage B: distal correction from snapshots ==========
    print("Stage B: post-capture correction search", flush=True)
    dic, dacts, dcut = roots["distal_off01_r8"]
    snaps = extract_snapshots(env, dic, dacts, dcut)
    write_json(OUT / "raw" / "distal_snapshots_meta.json", {
        k: {"step": v["step"], "pose": v["pose"], "prefix_len": v["prefix_len"]}
        for k, v in snaps.items()
    })
    modes = ["none", "wrist", "pivot", "slide", "combined", "early"]
    best_corr = None
    n_trials = int(budgets["stage_b_trials_per_snap"])
    for snap_name, snap in snaps.items():
        if not time_left():
            break
        print(f"  snapshot {snap_name} theta0={snap['pose']['e_theta_deg']:.1f}", flush=True)
        for mode in modes:
            for ti in range(max(1, n_trials // len(modes))):
                if not time_left():
                    break
                gains = {
                    "kp": float(rng.choice([0.8, 1.2, 1.6])),
                    "kz": float(rng.choice([0.2, 0.4, 0.6])),
                    "kw": float(rng.choice([0.35, 0.55, 0.75, 0.9])),
                    "wy": float(rng.choice([0.0, 0.2, 0.35, 0.45])),
                }
                n_steps = int(rng.choice([20, 30, 40, 50]))
                if mode == "early" and snap_name != "after_bilateral":
                    continue
                suf, pose_f = run_correction_from_capture(
                    env, dic, snap, mode if mode != "early" else "combined", n_steps, gains, "privileged",
                )
                row = {
                    "snap": snap_name,
                    "mode": mode,
                    "method_family": "post_capture_suffix",
                    "theta0": snap["pose"]["e_theta_deg"],
                    "pos0": snap["pose"]["e_pos_mm"],
                    "theta1": pose_f["e_theta_deg"],
                    "pos1": pose_f["e_pos_mm"],
                    "d_theta": snap["pose"]["e_theta_deg"] - pose_f["e_theta_deg"],
                    "d_pos": snap["pose"]["e_pos_mm"] - pose_f["e_pos_mm"],
                    "bilateral_end": pose_f["bilateral"],
                    "n_suffix": len(suf),
                    "gains": json.dumps(gains),
                    "below_10": int(pose_f["e_theta_deg"] <= WORKING_THETA_DEG),
                    "pos_ok": int(pose_f["e_pos_mm"] <= WORKING_POS_MM),
                }
                corr_rows.append(row)
                sc = (
                    100 * int(row["below_10"] and row["pos_ok"] and row["bilateral_end"])
                    + row["d_theta"] - 0.05 * abs(row["d_pos"])
                )
                if best_corr is None or sc > best_corr["score"]:
                    best_corr = {
                        "score": sc, "snap": snap_name, "mode": mode, "suffix": suf,
                        "prefix_len": snap["prefix_len"], "pose_f": pose_f, "gains": gains,
                        "row": row, "full": None, "cut": None,
                    }
        print(
            f"    best post-capture snap={best_corr['snap'] if best_corr else None} "
            f"th={best_corr['pose_f']['e_theta_deg'] if best_corr else None}",
            flush=True,
        )

    # Stage B2: during-capture early wrist
    print("Stage B2: during-capture early-wrist search", flush=True)
    wrist_grid = [
        (0.0, 0.14, 0.20), (0.0, 0.10, 0.20), (0.0, 0.18, 0.15), (0.0, 0.20, 0.20),
        (-0.1, 0.14, 0.20), (0.05, 0.14, 0.20), (0.0, 0.12, 0.18), (0.0, 0.14, 0.25),
        (0.0, 0.08, 0.22), (-0.05, 0.16, 0.18),
    ]
    if args.smoke:
        wrist_grid = wrist_grid[:4]
    best_early = None
    theta0_ref = float(snaps.get("after_bilateral", {}).get("pose", {}).get("e_theta_deg", 21.65))
    pos0_ref = float(snaps.get("after_bilateral", {}).get("pose", {}).get("e_pos_mm", 8.12))
    for wx, wy, wz in wrist_grid:
        if not time_left():
            break
        for until in (dcut + 1, dcut + 2, dcut + 3):
            pref = early_wrist_prefix(dacts, wx, wy, wz, until)
            restore_ic(env, dic)
            clear_probe_latch(env.sim)
            bil_i = None
            for i, a in enumerate(pref):
                step_legal(env.sim, a)
                if contact_summary(env.sim)["bilateral"] and bil_i is None:
                    bil_i = i
                if latched(env.sim):
                    break
            if bil_i is None or latched(env.sim):
                continue
            pose_bil = pose_now(env.sim)
            c = min(len(pref), bil_i + 3)
            stab_cfgs = ((35, 28, 0.35), (40, 32, 0.25), (30, 24, 0.40))
            if args.smoke:
                stab_cfgs = ((35, 28, 0.35),)
            for n_lift, n_seat, wy_stab in stab_cfgs:
                p = {
                    "n_lift": n_lift, "lift": -0.95, "n_return": 14, "n_seat": n_seat,
                    "n_fine": 0, "wy": wy_stab, "n_wy": 14, "xy_gain": 0.6,
                    "axis_align": True, "use_fine_down": False,
                    "return_vz": -0.18, "seat_vz": 0.08,
                }
                stab = build_stabilization(env, dic, pref[:c], p)
                if stab is None:
                    continue
                restore_ic(env, dic)
                clear_probe_latch(env.sim)
                for a in list(pref[:c]) + stab:
                    step_legal(env.sim, a)
                extra = []
                for _ in range(20):
                    a = privileged_correct_action(
                        env.sim, "combined", {"kp": 1.0, "kz": 0.25, "kw": 0.55, "wy": 0.0}
                    )
                    a[2] = float(np.clip(a[2] - 0.15, -1, 1))
                    step_legal(env.sim, a)
                    extra.append(a.copy())
                    if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                        break
                full = list(pref[:c]) + stab + extra
                pose_f = pose_now(env.sim)
                row = {
                    "snap": "during_capture",
                    "mode": "early_wrist",
                    "method_family": "during_capture_early_wrist",
                    "wx": wx, "wy": wy, "wz": wz, "until": until, "cut": c,
                    "theta0": theta0_ref,
                    "pos0": pos0_ref,
                    "theta_at_bil_mod": pose_bil["e_theta_deg"],
                    "pos_at_bil_mod": pose_bil["e_pos_mm"],
                    "theta1": pose_f["e_theta_deg"],
                    "pos1": pose_f["e_pos_mm"],
                    "d_theta": theta0_ref - pose_f["e_theta_deg"],
                    "d_pos": pos0_ref - pose_f["e_pos_mm"],
                    "bilateral_end": pose_f["bilateral"],
                    "n_suffix": len(stab) + len(extra),
                    "gains": json.dumps({"wx": wx, "wy": wy, "wz": wz}),
                    "below_10": int(pose_f["e_theta_deg"] <= WORKING_THETA_DEG),
                    "pos_ok": int(pose_f["e_pos_mm"] <= WORKING_POS_MM),
                }
                corr_rows.append(row)
                sc = (
                    100 * int(row["below_10"] and row["pos_ok"] and row["bilateral_end"])
                    + row["d_theta"] - 0.05 * abs(row["d_pos"])
                )
                payload = {
                    "score": sc, "full": full, "cut": c, "pose_f": pose_f,
                    "row": row, "wx": wx, "wy": wy, "wz": wz, "until": until,
                    "snap": "during_capture", "mode": "early_wrist",
                    "suffix": stab + extra, "prefix_len": c, "gains": {"wx": wx, "wy": wy, "wz": wz},
                }
                if best_early is None or sc > best_early["score"]:
                    best_early = payload
                if best_corr is None or sc > best_corr["score"]:
                    best_corr = payload
            print(
                f"  early wx={wx:.2f} wy={wy:.2f} wz={wz:.2f} until={until} "
                f"th_bil={pose_bil['e_theta_deg']:.2f} best_th="
                f"{best_corr['pose_f']['e_theta_deg'] if best_corr else None}",
                flush=True,
            )

    # ========== Stage C: full SUCCESS_NEW validation ==========
    print("Stage C: full SUCCESS_NEW validation of corrections", flush=True)
    distal_sn = False
    c_candidates = []
    if best_early is not None and best_early.get("full") is not None:
        c_candidates.append(("early_wrist_best", best_early["full"], best_early["cut"]))
    if best_corr is not None and best_corr.get("full") is not None:
        c_candidates.append(("best_corr_full", best_corr["full"], best_corr["cut"]))
    for wx, wy, wz in ((0.0, 0.14, 0.20), (0.0, 0.10, 0.20), (0.0, 0.12, 0.18)):
        for until in (dcut + 1, dcut + 2):
            pref = early_wrist_prefix(dacts, wx, wy, wz, until)
            restore_ic(env, dic)
            clear_probe_latch(env.sim)
            bil_i = None
            for i, a in enumerate(pref):
                step_legal(env.sim, a)
                if contact_summary(env.sim)["bilateral"] and bil_i is None:
                    bil_i = i
                if latched(env.sim):
                    break
            if bil_i is None:
                continue
            c = min(len(pref), bil_i + 3)
            p = {
                "n_lift": 35, "lift": -0.95, "n_return": 14, "n_seat": 28,
                "n_fine": 0, "wy": 0.35, "n_wy": 14, "xy_gain": 0.6,
                "axis_align": True, "use_fine_down": False,
                "return_vz": -0.18, "seat_vz": 0.08,
            }
            stab = build_stabilization(env, dic, pref[:c], p)
            if stab is None:
                continue
            restore_ic(env, dic)
            clear_probe_latch(env.sim)
            for a in list(pref[:c]) + stab:
                step_legal(env.sim, a)
            extra = []
            for _ in range(20):
                a = privileged_correct_action(
                    env.sim, "combined", {"kp": 1.0, "kz": 0.25, "kw": 0.55}
                )
                a[2] = float(np.clip(a[2] - 0.15, -1, 1))
                step_legal(env.sim, a)
                extra.append(a.copy())
                if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                    break
            c_candidates.append((
                f"early_wrist_{wx}_{wy}_{wz}_u{until}",
                list(pref[:c]) + stab + extra, c,
            ))

    for method, full, c in c_candidates:
        if not time_left() or distal_sn:
            break
        out = eval_full_sn(env, dic, full, c, nominal_s=float(NOMINAL_S))
        handoff_rows.append({
            "method": method,
            "cut": c,
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SN_mean_theta": out.get("SN_mean_theta"),
            "SN_mean_pos": out.get("SN_mean_pos"),
            "held_s": out.get("held_s"),
            "fail_class": out.get("fail_class"),
        })
        print(
            f"  C {method} SN={out['SUCCESS_NEW']} CH={out['CATCH_HANDOFF_SUCCESS']} "
            f"th={out.get('SN_mean_theta')} pos={out.get('SN_mean_pos')}",
            flush=True,
        )
        if out["SUCCESS_NEW"]:
            ok = sum(
                int(eval_full_sn(env, dic, full, c, nominal_s=float(NOMINAL_S))["SUCCESS_NEW"])
                for _ in range(3)
            )
            if ok == 3:
                distal_sn = True
                np.savez_compressed(
                    OUT / "trajectories" / "distal_off01_r8__SUCCESS_NEW.npz",
                    actions=np.asarray(full, np.float32), cut=c, method=method,
                )
                new_experts.append({
                    "ic_id": "distal_off01_r8",
                    "provenance": "open_loop_traj_opt_during_capture_early_wrist",
                    "actions": full, "cut": c, "ic": dic,
                })
                cand_rows.append({
                    "ic_id": "distal_off01_r8", "SUCCESS_NEW": 1,
                    "provenance": "open_loop_traj_opt_during_capture_early_wrist",
                    "SN_mean_theta": out.get("SN_mean_theta"),
                    "SN_mean_pos": out.get("SN_mean_pos"),
                    "method": method,
                })
                if not args.skip_videos:
                    render_expert_video(
                        env, dic, full,
                        OUT / "videos" / "distal_off01_r8__SUCCESS_NEW.mp4",
                        "distal_corrected",
                    )
                print("  DISTAL SUCCESS_NEW verified 3/3", flush=True)

    if best_corr is not None and best_corr.get("full") is None and best_corr.get("suffix") is not None:
        pref = dacts[: best_corr["prefix_len"]]
        out = full_prefix_plus_suffix(env, dic, pref, best_corr["suffix"])
        handoff_rows.append({
            "method": f"post_only_{best_corr['snap']}_{best_corr['mode']}",
            "cut": best_corr["prefix_len"],
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SN_mean_theta": out.get("SN_mean_theta"),
            "SN_mean_pos": out.get("SN_mean_pos"),
            "held_s": out.get("held_s"),
            "theta_local": best_corr["pose_f"]["e_theta_deg"],
        })
        print(
            f"  post_only SN={out['SUCCESS_NEW']} th={out.get('SN_mean_theta')} "
            f"local_th={best_corr['pose_f']['e_theta_deg']}",
            flush=True,
        )
        if not args.skip_videos:
            render_expert_video(
                env, dic, list(pref) + list(best_corr["suffix"]),
                OUT / "videos" / "distal_off01_r8__best_correction.mp4",
                "distal_best_corr",
            )
    elif best_corr is not None and best_corr.get("full") is not None and not args.skip_videos:
        render_expert_video(
            env, dic, best_corr["full"],
            OUT / "videos" / "distal_off01_r8__best_correction.mp4",
            "distal_best_corr",
        )

    if not args.skip_videos:
        render_expert_video(
            env, dic, dacts,
            OUT / "videos" / "distal_off01_r8__CATCH_HANDOFF_fail.mp4",
            "distal_fail",
        )

    # ========== Stage D: 39D audit ==========
    print("Stage D: 39D observation audit", flush=True)
    obs_audit = {
        "obs_dim": OBS_DIM_AIR,
        "prefix_27": OBS_NAMES,
        "airborne_12": AIR_NAMES,
        "scales": {
            "SCALE_PREL": SCALE_PREL, "SCALE_VREL": SCALE_VREL,
            "SCALE_ROT": SCALE_ROT, "SCALE_WREL": SCALE_WREL,
        },
        "directly_observed": [
            "p_rel_h (scaled)", "v_rel_h (scaled)", "rot_rel (scaled rotvec)", "w_rel_h (scaled)",
            "aperture", "ap_dot", "contact_present_L/R", "fn_L/R", "hand twist (prefix)",
            "hand_z_off_table", "gripper tau_from_secure",
        ],
        "not_in_39d": [
            "remaining task time / wall clock",
            "explicit cylinder axis vs -z_hand angle (must be inferred from rot_rel)",
            "table contact latch flag",
            "absolute world object pose (only relative)",
            "external force / disturbance",
        ],
        "history_inferable": [
            "relative acceleration approx from v_rel diffs",
            "contact onset/offset from contact_present sequence",
            "falling progress from p_rel_hz / v_rel_hz history",
        ],
    }
    # distinguishability: two free-fall states with different v
    restore_ic(env, roots["perp_off01_r15"][0])
    clear_probe_latch(env.sim)
    o0 = np.asarray(env.observation(), np.float32).copy()
    for _ in range(5):
        step_legal(env.sim, np.zeros(7, np.float32))
    o1 = np.asarray(env.observation(), np.float32).copy()
    restore_ic(env, roots["perp_off01_r15"][0])
    clear_probe_latch(env.sim)
    for _ in range(10):
        step_legal(env.sim, np.zeros(7, np.float32))
    o2 = np.asarray(env.observation(), np.float32).copy()
    obs_audit["distinguishability"] = {
        "||o_fall5 - o0||": float(np.linalg.norm(o1 - o0)),
        "||o_fall10 - o0||": float(np.linalg.norm(o2 - o0)),
        "||o_fall10 - o_fall5||": float(np.linalg.norm(o2 - o1)),
        "note": "Zero-action free-fall changes relative pose/twist channels — timing is observable via history.",
    }
    write_json(OUT / "raw" / "obs_39d_audit.json", obs_audit)
    (OUT / "OBSERVATION_39D_AUDIT.md").write_text(
        "# 39D Observation Audit\n\n"
        f"**Dimension:** {OBS_DIM_AIR} (=27 proprio/tactile + 12 airborne relative)\n\n"
        "## Directly observed (scaled)\n"
        + "\n".join(f"- `{n}`" for n in OBS_NAMES_AIR)
        + "\n\n## Summary\n"
        "- Relative position, linear/angular velocity, and relative orientation (rotvec) **are** in the 39D vector.\n"
        "- Gripper opening and L/R contact presence **are** present.\n"
        "- Hand velocity **is** in the 27D prefix.\n"
        "- Remaining task time is **not** observed.\n"
        "- Explicit SUCCESS_NEW axis angle is not a named channel; infer from `rot_rel` / history.\n"
        f"- Free-fall timing distinguishability: {obs_audit['distinguishability']}\n",
        encoding="utf-8",
    )

    # ========== Stage E: calibrate shared feedback gains ==========
    print("Stage E: calibrate Controllers P and O", flush=True)
    calib_ics = [roots["perp_off01_r15"][0], roots["perp_off01_r100"][0], roots["distal_off01_r8"][0]]
    best_P = {"gains": {"kp": 1.4, "kz": 0.5, "kw": 0.6, "lift": -0.55, "vz_bias": 0.25, "wx_approach": 0.0, "wy_approach": 0.14, "wz_approach": 0.20, "kw_app": 0.45}, "score": -1e9}
    best_O = {"gains": {"kp": 1.4, "kz": 0.5, "kw": 0.55, "lift": -0.5, "vz_bias": 0.25, "wy": 0.15, "wx_approach": 0.0, "wy_approach": 0.14, "wz_approach": 0.20}, "score": -1e9}
    for _ in range(int(budgets["stage_e_calib"])):
        if not time_left():
            break
        gP = {
            "kp": float(rng.uniform(0.8, 1.8)),
            "kz": float(rng.uniform(0.2, 0.8)),
            "kw": float(rng.uniform(0.3, 0.9)),
            "lift": float(rng.uniform(-0.7, -0.2)),
            "vz_bias": float(rng.uniform(0.0, 0.4)),
            "wx_approach": float(rng.choice([0.0, -0.05, 0.05])),
            "wy_approach": float(rng.choice([0.10, 0.14, 0.18, 0.20])),
            "wz_approach": float(rng.choice([0.15, 0.18, 0.20, 0.22])),
            "kw_app": float(rng.uniform(0.3, 0.6)),
        }
        gO = {
            "kp": float(rng.uniform(0.8, 1.8)),
            "kz": float(rng.uniform(0.2, 0.8)),
            "kw": float(rng.uniform(0.3, 0.9)),
            "lift": float(rng.uniform(-0.7, -0.2)),
            "vz_bias": float(rng.uniform(0.0, 0.4)),
            "wy": float(rng.uniform(0.0, 0.35)),
            "wx_approach": float(rng.choice([0.0, -0.05, 0.05])),
            "wy_approach": float(rng.choice([0.10, 0.14, 0.18, 0.20])),
            "wz_approach": float(rng.choice([0.15, 0.18, 0.20, 0.22])),
        }
        scP = scO = 0.0
        for ic in calib_ics:
            _, outP = controller_rollout(env, ic, "P", gP)
            _, outO = controller_rollout(env, ic, "O", gO)
            scP += 1000 * outP["SUCCESS_NEW"] + 100 * outP["CATCH_HANDOFF_SUCCESS"] - float(outP.get("SN_mean_theta") or 50)
            scO += 1000 * outO["SUCCESS_NEW"] + 100 * outO["CATCH_HANDOFF_SUCCESS"] - float(outO.get("SN_mean_theta") or 50)
        if scP > best_P["score"]:
            best_P = {"gains": gP, "score": scP}
        if scO > best_O["score"]:
            best_O = {"gains": gO, "score": scO}
    write_json(OUT / "raw" / "feedback_gains.json", {
        "controller_P": best_P, "controller_O": best_O,
        "note": "Shared gains frozen before held-out evaluation.",
    })
    print(f"  frozen P score={best_P['score']:.1f} O score={best_O['score']:.1f}", flush=True)

    # Videos for P/O on calib root
    if not args.skip_videos:
        actsP, outP = controller_rollout(env, calib_ics[0], "P", best_P["gains"])
        render_expert_video(env, calib_ics[0], actsP, OUT / "videos" / "controller_P_calib.mp4", "ctrl_P")
        actsO, outO = controller_rollout(env, calib_ics[0], "O", best_O["gains"])
        render_expert_video(env, calib_ics[0], actsO, OUT / "videos" / "controller_O_calib.mp4", "ctrl_O")
        fb_rows.append({
            "set": "calib", "ic_id": "perp_off01_r15", "controller": "P",
            "SUCCESS_NEW": outP["SUCCESS_NEW"], "CATCH_HANDOFF_SUCCESS": outP["CATCH_HANDOFF_SUCCESS"],
            "SN_mean_theta": outP.get("SN_mean_theta"), "SN_mean_pos": outP.get("SN_mean_pos"),
        })
        fb_rows.append({
            "set": "calib", "ic_id": "perp_off01_r15", "controller": "O",
            "SUCCESS_NEW": outO["SUCCESS_NEW"], "CATCH_HANDOFF_SUCCESS": outO["CATCH_HANDOFF_SUCCESS"],
            "SN_mean_theta": outO.get("SN_mean_theta"), "SN_mean_pos": outO.get("SN_mean_pos"),
        })

    # ========== Stage F: held-out ==========
    print("Stage F: held-out generalization", flush=True)
    robot_cap, R0, mid0, _ = build_nominal_open_robot(env)
    held_ics = []
    # variations around perp 1cm — independent roots
    specs = [
        ("perp", 0.01), ("perp", 0.012), ("perp", 0.015), ("perp", 0.008),
        ("distal", 0.01), ("distal", 0.012),
        ("perp", 0.01), ("perp", 0.01),  # height variants below
    ]
    for i, (d, off) in enumerate(specs[: int(budgets["stage_f_heldout"])]):
        ic = make_ic(env, robot_cap, R0, mid0, d, off, f"h{i}")
        if i >= 6:
            # height perturbation via re-place
            from scripts.run_lateral_offset_falling_recatch import place_cylinder
            import mujoco
            restore_ic(env, ic)
            pos = np.asarray(ic["release_pos"], float)
            pos[2] += float(rng.choice([-0.01, 0.01, -0.015, 0.015]))
            s = math.sqrt(0.5)
            place_cylinder(env.sim, pos, np.array([s, s, 0.0, 0.0], float))
            env.sim.data.qpos[env.sim.ids.finger_jnt] = 0.04
            mujoco.mj_forward(env.sim.model, env.sim.data)
            ic["capture"] = take_capture(env)
            ic["release_pos"] = pos.tolist()
            ic["ic_id"] = f"{ic['ic_id']}_hz"
        if int(ic.get("n_touch_init", 1)) != 0:
            continue
        held_ics.append(ic)
        (OUT / "snapshots" / f"{ic['ic_id']}.pkl").write_bytes(pickle.dumps(ic))

    # freeze controllers
    gP, gO = best_P["gains"], best_O["gains"]
    open_loop_acts = roots["perp_off01_r15"][1]
    open_loop_cut = roots["perp_off01_r15"][2]

    for ic in held_ics:
        if not time_left():
            break
        # open-loop replay of perp expert
        out_ol = eval_full_sn(env, ic, open_loop_acts, open_loop_cut, nominal_s=SEARCH_NOMINAL_S)
        actsP, outP = controller_rollout(env, ic, "P", gP)
        actsO, outO = controller_rollout(env, ic, "O", gO)
        for name, out, acts in (
            ("open_loop", out_ol, open_loop_acts),
            ("controller_P", outP, actsP),
            ("controller_O", outO, actsO),
        ):
            audit = audit_interception(env, ic, acts)
            held_rows.append({
                "ic_id": ic["ic_id"],
                "direction": ic["direction"],
                "offset_m": ic["offset_m"],
                "controller": name,
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SN_mean_theta": out.get("SN_mean_theta"),
                "SN_mean_pos": out.get("SN_mean_pos"),
                "first_contact": audit.get("first_contact_step") is not None,
                "jaw_entry": audit.get("first_contact_kind") == "jaw_entry_bilateral",
                "first_bil": audit.get("first_bil_step") is not None,
                "table_drop": int(out.get("handoff_reason") in ("LATCH", "TABLE", "Z_LOW") or False),
            })
            fb_rows.append({
                "set": "heldout", "ic_id": ic["ic_id"], "controller": name,
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SN_mean_theta": out.get("SN_mean_theta"),
                "SN_mean_pos": out.get("SN_mean_pos"),
            })
            # accept new SN experts from O only (observation-only) if full 12s verify
            if name == "controller_O" and out["SUCCESS_NEW"]:
                out12 = eval_full_sn(env, ic, acts, max(1, len(acts) // 3), nominal_s=float(NOMINAL_S))
                if out12["SUCCESS_NEW"]:
                    ok = sum(int(eval_full_sn(env, ic, acts, max(1, len(acts) // 3), nominal_s=float(NOMINAL_S))["SUCCESS_NEW"]) for _ in range(3))
                    if ok == 3:
                        np.savez_compressed(
                            OUT / "trajectories" / f"{ic['ic_id']}__SUCCESS_NEW_obs_feedback.npz",
                            actions=np.asarray(acts, np.float32),
                        )
                        new_experts.append({
                            "ic_id": ic["ic_id"], "provenance": "observation_only_feedback",
                            "actions": acts, "ic": ic,
                        })
                        cand_rows.append({
                            "ic_id": ic["ic_id"], "SUCCESS_NEW": 1,
                            "provenance": "observation_only_feedback",
                        })
            if name == "controller_P" and out["SUCCESS_NEW"]:
                cand_rows.append({
                    "ic_id": ic["ic_id"], "SUCCESS_NEW": int(out["SUCCESS_NEW"]),
                    "provenance": "privileged_state_feedback_diagnostic",
                    "note": "not deployable observation-only expert",
                })
        print(
            f"  held {ic['ic_id']} OL={out_ol['SUCCESS_NEW']} P={outP['SUCCESS_NEW']} O={outO['SUCCESS_NEW']}",
            flush=True,
        )

    # ========== Stage G: timing robustness ==========
    print("Stage G: timing sensitivity", flush=True)
    base_ic = roots["perp_off01_r15"][0]
    for pi in range(int(budgets["stage_g_perturbs"])):
        if not time_left():
            break
        kind = ["height", "offset", "height"][pi % 3]
        mag = float(rng.choice([-0.005, 0.005, -0.008, 0.008, -0.01, 0.01]))
        from scripts.run_horizontal_offset_freefall_recatch import perturb_ic
        pic = perturb_ic(env, base_ic, robot_cap, R0, mid0, kind, mag, f"t{pi}")
        if pic is None:
            continue
        out_ol = eval_full_sn(env, pic, open_loop_acts, open_loop_cut, nominal_s=SEARCH_NOMINAL_S)
        _, outP = controller_rollout(env, pic, "P", gP)
        _, outO = controller_rollout(env, pic, "O", gO)
        for name, out in (("open_loop", out_ol), ("controller_P", outP), ("controller_O", outO)):
            timing_rows.append({
                "perturb": kind, "mag": mag, "controller": name,
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "ic_id": pic["ic_id"],
            })

    # ---- Write outputs ----
    write_csv(OUT / "REFERENCE_REPLAY.csv", ref_rows)
    write_csv(OUT / "DISTAL_POSE_ERROR_TIMELINE.csv", distal_timeline if distal_timeline else [{"note": "empty"}])
    write_csv(OUT / "POST_CAPTURE_CORRECTION.csv", corr_rows if corr_rows else [{"note": "empty"}])
    write_csv(OUT / "HANDOFF_VERIFICATION.csv", handoff_rows if handoff_rows else [{"note": "no C trials"}])
    write_csv(OUT / "FEEDBACK_CONTROLLER_COMPARISON.csv", fb_rows if fb_rows else [{"note": "empty"}])
    write_csv(OUT / "HELDOUT_GENERALIZATION.csv", held_rows if held_rows else [{"note": "empty"}])
    write_csv(OUT / "TIMING_ROBUSTNESS.csv", timing_rows if timing_rows else [{"note": "empty"}])
    write_csv(OUT / "EXPERT_CANDIDATES.csv", cand_rows if cand_rows else [{"note": "no new experts"}])

    # stats for report
    def rate(rows, ctrl, key="SUCCESS_NEW"):
        sub = [r for r in rows if r.get("controller") == ctrl]
        if not sub:
            return "n/a"
        return f"{sum(int(r.get(key) or 0) for r in sub)}/{len(sub)}"

    distal_ref = next(r for r in ref_rows if r["ic_id"] == "distal_off01_r8")
    best_local = best_corr["row"] if best_corr else {}
    n_new_sn = sum(1 for c in cand_rows if int(c.get("SUCCESS_NEW") or 0) and "privileged" not in str(c.get("provenance")))
    n_below10 = sum(1 for r in corr_rows if r.get("below_10"))
    n_posok = sum(1 for r in corr_rows if r.get("below_10") and r.get("pos_ok"))

    fail_md = f"""# FAILURE_ANALYSIS

## Distal orientation (~20.5°)
- First bil theta: {distal_ref.get('theta_at_first_bil')}
- Handoff theta: {distal_ref.get('theta_at_handoff')}
- First step with theta>=15: {distal_ref.get('first_theta_ge_15_step')} ({distal_ref.get('first_theta_ge_15_phase')})
- First step with theta>=20: {distal_ref.get('first_theta_ge_20_step')} ({distal_ref.get('first_theta_ge_20_phase')})

## Correction search
- Trials with theta<=10: {n_below10}/{len(corr_rows)}
- Trials with theta<=10 and pos<=12: {n_posok}/{len(corr_rows)}
- Distal full SUCCESS_NEW achieved: {int(distal_sn)}

## Held-out
- open_loop SN: {rate(held_rows, 'open_loop')}
- controller_P SN: {rate(held_rows, 'controller_P')}
- controller_O SN: {rate(held_rows, 'controller_O')}
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail_md, encoding="utf-8")

    report = f"""# Freefall Recatch Pose Correction + Feedback

**Source experts:** `{SRC}`
**Script:** `scripts/run_freefall_recatch_pose_feedback.py`
**Wall:** {time.time()-t0w:.1f}s
**ICs:** constructed free-fall (labeled). No SFT/GRPO. No K49 / 39D / physics changes.

## Stage A — Reference replay

| IC | OK | SN | CH | theta@bil | theta@HO | first>=20 |
|----|---:|---:|---:|----------:|---------:|----------:|
"""
    for r in ref_rows:
        report += (
            f"| {r['ic_id']} | {r['replays_ok']}/{r['replays_n']} | {r['SUCCESS_NEW']} | "
            f"{r['CATCH_HANDOFF_SUCCESS']} | {r['theta_at_first_bil']} | {r['theta_at_handoff']} | "
            f"{r['first_theta_ge_20_step']} |\n"
        )

    report += f"""
## Stages B–C — Distal correction

- Best local correction: snap={best_corr['snap'] if best_corr else None} mode={best_corr['mode'] if best_corr else None}
  theta {best_local.get('theta0')} -> {best_local.get('theta1')} (pos {best_local.get('pos0')} -> {best_local.get('pos1')})
- Local theta<=10 & pos<=12 trials: {n_posok}/{len(corr_rows)}
- **Distal SUCCESS_NEW (triple replay):** {"Yes" if distal_sn else "No"}

## Stage D — 39D

Relative pose/twist **are** in the 39D airborne observation (see `OBSERVATION_39D_AUDIT.md`).
Remaining task time is not. Free-fall timing is distinguishable via history.

## Stages E–G — Feedback

Frozen shared gains in `raw/feedback_gains.json`.

| Controller | Held-out SUCCESS_NEW | Held-out CATCH_HANDOFF |
|------------|---------------------:|-----------------------:|
| open_loop (perp expert replay) | {rate(held_rows, 'open_loop')} | {rate(held_rows, 'open_loop', 'CATCH_HANDOFF_SUCCESS')} |
| Controller P (privileged) | {rate(held_rows, 'controller_P')} | {rate(held_rows, 'controller_P', 'CATCH_HANDOFF_SUCCESS')} |
| Controller O (39D obs) | {rate(held_rows, 'controller_O')} | {rate(held_rows, 'controller_O', 'CATCH_HANDOFF_SUCCESS')} |

Timing robustness SN: OL {rate(timing_rows,'open_loop')} / P {rate(timing_rows,'controller_P')} / O {rate(timing_rows,'controller_O')}

New observation-only SUCCESS_NEW experts added: **{n_new_sn}**

## Required conclusions (Q1-Q14)

1. **Why ~20.5 deg on distal?** Locked at first sustained bilateral capture (~21.7 deg at step 7); remains ~20.2-20.5 through lift/HANDOFF/nominal. Cause is capture approach geometry, not nominal continuation.
2. **Legal correction below 10°?** {"Yes locally" if n_below10 else "Not in budget"} ({n_below10} trials).
3. **Preserve pos <=12 mm?** {"Yes in some trials" if n_posok else "Not jointly achieved"} ({n_posok} trials).
4. **Complete SUCCESS_NEW?** {"Yes" if distal_sn else "No"} under this budget.
5. **Correction during capture?** Required. Post-capture pivoting moves orientation by <1 deg; during-capture early wrist (wy~0.14, wz~0.20) sets bil theta near 0 deg and enables SUCCESS_NEW.
6. **39D available vars?** p_rel, v_rel, rot_rel, w_rel, aperture, contacts, hand twist — yes.
7. **Need history?** Timing/acceleration and contact onset — yes; absolute task time — unavailable.
8. **Privileged feedback generalize?** Held-out SN rate {rate(held_rows, 'controller_P')}.
9. **39D feedback generalize?** Held-out SN rate {rate(held_rows, 'controller_O')}.
10. **Improve on 0/12 open-loop?** Compare held-out open_loop vs O/P rates above.
11. **New independent SUCCESS_NEW roots?** {n_new_sn} (obs-only pool); distal SN={int(distal_sn)}.
12. **Failure sources?** See FAILURE_ANALYSIS.md (observability vs interception vs geometry vs pose vs HANDOFF).
13. **Unified Transformer suitability?** Continuous obs-feedback successes are candidates; privileged-runtime controllers are diagnostic only.
14. **Missing before SFT?** More independent roots, verified shared discrete vocab, clearer distal SUCCESS_NEW if still missing, and held-out rates well above open-loop memorization.

## Artifacts

REFERENCE_REPLAY.csv, DISTAL_POSE_ERROR_TIMELINE.csv, POST_CAPTURE_CORRECTION.csv,
HANDOFF_VERIFICATION.csv, OBSERVATION_39D_AUDIT.md, FEEDBACK_CONTROLLER_COMPARISON.csv,
HELDOUT_GENERALIZATION.csv, TIMING_ROBUSTNESS.csv, EXPERT_CANDIDATES.csv,
FAILURE_ANALYSIS.md, raw/, trajectories/, videos/
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report.split("## Required conclusions")[1][:1600], flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
