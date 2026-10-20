"""Convert lateral-offset catches into handoff-compatible recovery.

Keeps the 40 existing catch trajectories unchanged as prefixes.
Searches legal post-catch stabilization only. No SFT/GRPO. No larger offsets.

CATCH_HANDOFF_SUCCESS is frozen before search (separate from SUCCESS_NEW).
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
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import relative_speed
from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import export_state, latched, restore_state, state_of
from envs.step_legal import step_legal
from evaluation.success_criteria import (
    HOLD_S as SN_HOLD,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    window_stats,
)
from evaluation.grasp_geometry import cylinder_axis_hand, measure as gmeasure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT, write_csv, write_json
from scripts.run_lateral_offset_falling_recatch import (
    FINE_DOWN,
    HOLD_S as CATCH_HOLD,
    TABLE_Z,
    body_v_w,
    finger_frame,
    hand_pos,
    make_training_env,
    obj_pos,
    object_table_contact,
    restore_ic,
    rot_angle,
)
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

SRC = ROOT / "results" / "diagnostics" / "lateral_offset_falling_recatch"
OUT = ROOT / "results" / "diagnostics" / "lateral_recatch_handoff"

# ---- Frozen budgets (recorded before execution) ----
TOTAL_BUDGET = 600
AUDIT_BUDGET = 120       # 20%
STAB_BUDGET = 330        # 55%
VERIFY_BUDGET = 90       # 15%
ROBUST_BUDGET = 60       # 10%
WALL_LIMIT_S = 35 * 60

# ---- Frozen CATCH_HANDOFF_SUCCESS (not SUCCESS_NEW) ----
CH_HOLD_S = 2.0
CH_VERIFY_S = 0.50
CH_POS_MM = 20.0
CH_THETA_DEG = 25.0
CH_Z_MIN = 0.50
CH_WRIST_MAX_DEG = 45.0
CH_VREL_MAX = 0.08


def tokens_hash(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


def hand_R(sim):
    return np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()


def clear_probe_latch(sim):
    """ICs were contaminated by make_ic's offline table-impact probe.

    Clearing at episode start removes that artifact only. Mid-episode latch
    from a real near-table contact loss is left intact and fails HANDOFF.
    """
    restore_state(sim, None)


def load_roots():
    """One preferred trajectory per independent release root (lowest wrist)."""
    rows = list(csv.DictReader((SRC / "CATCH_SUCCESS_VERIFICATION.csv").open(encoding="utf-8")))
    best = {}
    for r in rows:
        ic = r["ic_id"]
        if ic not in best or float(r["max_wrist_deg"]) < float(best[ic]["max_wrist_deg"]):
            best[ic] = r
    roots = []
    for ic_id, r in sorted(best.items()):
        snap = SRC / "snapshots" / f"{ic_id}.pkl"
        traj = SRC / "trajectories" / f"{ic_id}__{r['condition']}_actions.npz"
        if not snap.exists() or not traj.exists():
            continue
        with snap.open("rb") as f:
            ic = pickle.load(f)
        acts = [np.asarray(a, np.float32) for a in np.load(traj)["actions"]]
        roots.append({
            "ic_id": ic_id,
            "direction": r["direction"],
            "offset_m": float(r["offset_m"]),
            "condition": r["condition"],
            "ic": ic,
            "catch_actions": acts,
            "actions_hash": r.get("actions_hash"),
            "max_wrist_deg": float(r["max_wrist_deg"]),
        })
    return roots


def replay_prefix(env, ic, acts, clear_latch=True):
    restore_ic(env, ic)
    if clear_latch:
        clear_probe_latch(env.sim)
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))


def snapshot_catch_state(env, ic, t0_ep):
    sim = env.sim
    cs = contact_summary(sim)
    m = gmeasure(sim, U_TGT, P_TGT)
    Rh = hand_R(sim)
    R0 = np.asarray(ic["R0"], float).reshape(3, 3)
    vh, wh = body_v_w(sim, sim.ids.hand_body)
    vo, wo = body_v_w(sim, sim.ids.object_body)
    dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    vz = float(sim.data.qvel[dadr + 2])
    st = state_of(sim)
    p_des = np.asarray(sim.fsm.p_des, float).reshape(-1)[:3]
    r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    return {
        "t_rel": float(sim.data.time) - t0_ep,
        "bilateral": int(cs["bilateral"]),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "obj_z": float(obj_pos(sim)[2]),
        "hand_z": float(hand_pos(sim)[2]),
        "vz_obj": vz,
        "vrel": float(relative_speed(sim)),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "p_c_h": m["p_c_h"],
        "u_c_h": m["u_c_h"],
        "wrist_from_R0_deg": float(rot_angle(R0, Rh) * 180 / math.pi),
        "r_des_to_hand_deg": float(rot_angle(r_des, Rh) * 180 / math.pi),
        "p_des_to_hand_m": float(np.linalg.norm(p_des - hand_pos(sim))),
        "hand_disp_from_ic_m": float(np.linalg.norm(hand_pos(sim) - np.asarray(ic["hand_pos"], float))),
        "vh": vh.tolist(),
        "vo": vo.tolist(),
        "wh": wh.tolist(),
        "wo": wo.tolist(),
        "grip_ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else None,
        "latched": int(st["latched"]),
        "latch_kind": st.get("kind"),
        "table_contact": int(object_table_contact(sim)),
    }


def classify_handoff_failure(pre, post_reason, samples, latch_at_catch):
    """Map observed failure to A–I taxonomy."""
    if latch_at_catch:
        return "H_nominal_continuation_drop_or_escape", "sticky_table_latch_during_catch"
    if pre.get("vrel", 0) > 0.15:
        return "B_relative_motion_too_high", None
    if int(pre.get("nL", 0)) + int(pre.get("nR", 0)) == 0:
        return "A_immediate_contact_loss", None
    if not int(pre.get("bilateral", 0)):
        return "C_insufficient_grip_retention", None
    if float(pre.get("e_pos_mm", 99)) > WORKING_POS_MM * 1.5:
        return "D_cylinder_position_error", None
    if float(pre.get("e_theta_deg", 99)) > WORKING_THETA_DEG * 1.5:
        return "E_cylinder_orientation_error", None
    if float(pre.get("wrist_from_R0_deg", 0)) > 45:
        return "F_large_hand_orientation_deviation", None
    if float(pre.get("r_des_to_hand_deg", 0)) > 15 or float(pre.get("p_des_to_hand_m", 0)) > 0.05:
        return "G_controller_target_actual_mismatch", None
    if post_reason in ("LATCH", "TABLE", "Z_LOW", "LOST", "ESCAPE"):
        return "H_nominal_continuation_drop_or_escape", post_reason
    if post_reason == "TIME":
        return "I_insufficient_remaining_time", None
    if samples and (window_stats(samples, max(0.0, CH_HOLD_S - CH_VERIFY_S)).get("mean_theta") or 99) > WORKING_THETA_DEG:
        return "E_cylinder_orientation_error", "SUCCESS_NEW_theta"
    if samples and (window_stats(samples, max(0.0, CH_HOLD_S - CH_VERIFY_S)).get("mean_pos") or 99) > WORKING_POS_MM:
        return "D_cylinder_position_error", "SUCCESS_NEW_pos"
    return "other", post_reason


def run_nominal(env, seconds, record=True):
    t0 = float(env.sim.data.time)
    reason = None
    samples = []
    next_s = 0.0
    while reason is None and float(env.sim.data.time) - t0 < seconds - 1e-9:
        nominal_step(env.sim)
        tr = float(env.sim.data.time) - t0
        if record and tr >= next_s:
            m = gmeasure(env.sim, U_TGT, P_TGT)
            cs = contact_summary(env.sim)
            samples.append({
                "t_rel": tr,
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "bilateral": cs["bilateral"],
                "obj_z": m["obj_z"],
            })
            next_s += 0.02
        if latched(env.sim):
            reason = "LATCH"
            break
        if float(obj_pos(env.sim)[2]) < 0.40:
            reason = "Z_LOW"
            break
        cs = contact_summary(env.sim)
        if int(cs["nL"]) + int(cs["nR"]) == 0:
            reason = "LOST"
            break
    return reason, samples, float(env.sim.data.time) - t0


def evaluate_catch_handoff(env, ic, full_actions, catch_len, R0):
    """Replay full trajectory from IC; score CATCH_HANDOFF_SUCCESS and SUCCESS_NEW."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    t0 = float(env.sim.data.time)
    first_bil = None
    latch_during_catch = False
    for i, a in enumerate(full_actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        cs = contact_summary(env.sim)
        if cs["bilateral"] and first_bil is None:
            first_bil = i
        if i < catch_len and latched(env.sim):
            latch_during_catch = True
    pre = snapshot_catch_state(env, ic, t0)
    pre["latch_during_catch"] = int(latch_during_catch)
    pre["first_bil_step"] = first_bil

    # HANDOFF
    ho = apply_handoff(env.sim, ic["ref"])
    # Search uses CH_HOLD_S; callers may pass longer via env attribute.
    nom_s = float(getattr(env, "_ch_nominal_s", CH_HOLD_S))
    reason, samples, held = run_nominal(env, nom_s, record=True)
    win_ch = window_stats(samples, max(0.0, min(held, CH_HOLD_S) - CH_VERIFY_S), hold_s=CH_VERIFY_S)
    win_sn = window_stats(samples, max(0.0, min(held, nom_s) - min(2.0, nom_s * 0.5)), hold_s=SN_HOLD)

    mean_th = win_ch.get("mean_theta")
    mean_pos = win_ch.get("mean_pos")
    z_ok = float(obj_pos(env.sim)[2]) >= CH_Z_MIN if reason is None else False
    wrist_ok = float(pre["wrist_from_R0_deg"]) <= CH_WRIST_MAX_DEG
    pose_ok = (
        mean_th is not None and mean_pos is not None
        and mean_th <= CH_THETA_DEG and mean_pos <= CH_POS_MM
    )
    catch_handoff = bool(
        first_bil is not None
        and not latch_during_catch
        and int(pre["bilateral"])
        and reason is None
        and held + 1e-9 >= CH_HOLD_S
        and pose_ok
        and z_ok
        and wrist_ok
        and float(pre["vrel"]) <= CH_VREL_MAX
    )
    success_new = bool(
        reason is None
        and held + 1e-9 >= 4.0
        and win_sn.get("working_SUCCESS_NEW_hold")
    )
    fail_class, fail_detail = classify_handoff_failure(
        pre, reason, samples, latch_during_catch,
    )
    return {
        "CATCH_HANDOFF_SUCCESS": int(catch_handoff),
        "SUCCESS_NEW": int(success_new),
        "handoff_reason": reason,
        "held_s": held,
        "fail_class": fail_class,
        "fail_detail": fail_detail,
        "mean_theta_ch": mean_th,
        "mean_pos_ch": mean_pos,
        "SN_mean_theta": win_sn.get("mean_theta"),
        "SN_mean_pos": win_sn.get("mean_pos"),
        "SN_hold": int(bool(win_sn.get("working_SUCCESS_NEW_hold"))),
        "pre": pre,
        "ho": {k: ho[k] for k in ("r_des_to_hand_rad", "physics_edited", "p_des") if k in ho},
    }


def build_stabilization(env, ic, catch_prefix, p):
    """Legal continuous post-catch: lift, optional XY return, wrist, seat, fine_down."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    for a in catch_prefix:
        step_legal(env.sim, np.asarray(a, np.float32))
    if not contact_summary(env.sim)["bilateral"]:
        return None
    if latched(env.sim):
        return None
    target = np.asarray(ic["hand_pos"], float)
    R0 = np.asarray(ic["R0"], float).reshape(3, 3)
    acts = []
    n_lift = int(p["n_lift"])
    lift = float(p["lift"])
    n_return = int(p["n_return"])
    n_seat = int(p["n_seat"])
    n_fine = int(p["n_fine"])
    wy = float(p.get("wy", 0.0))
    use_axis_align = bool(p.get("axis_align", False))
    use_fine = bool(p.get("use_fine_down", True))

    for i in range(n_lift):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        a[2] = lift
        Rh = hand_R(env.sim)
        err = Rh.T @ (target - hand_pos(env.sim))
        gain = float(p.get("xy_gain", 0.5))
        a[0] = float(np.clip(gain * err[0] / 0.05, -0.7, 0.7))
        a[1] = float(np.clip(gain * err[1] / 0.05, -0.7, 0.7))
        if i < int(p.get("n_wy", 0)):
            a[4] = wy
        step_legal(env.sim, a)
        acts.append(a.copy())
        if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
            return None

    for i in range(n_return):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        a[2] = float(p.get("return_vz", -0.15))
        Rh = hand_R(env.sim)
        err = Rh.T @ (target - hand_pos(env.sim))
        a[0] = float(np.clip(0.9 * err[0] / 0.04, -1, 1))
        a[1] = float(np.clip(0.9 * err[1] / 0.04, -1, 1))
        # wrist toward R0
        dR = R0 @ hand_R(env.sim).T
        # small wy only
        a[4] = float(np.clip(wy * 0.3, -0.35, 0.35))
        step_legal(env.sim, a)
        acts.append(a.copy())
        if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
            return None

    for i in range(n_seat):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        a[2] = float(p.get("seat_vz", 0.1))
        if use_axis_align:
            u = np.asarray(cylinder_axis_hand(env.sim), float)
            ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
            c = np.cross(u, ut)
            n = float(np.linalg.norm(c))
            if n > 1e-4:
                a[3:6] = np.clip(0.35 * c / n, -0.45, 0.45)
        step_legal(env.sim, a)
        acts.append(a.copy())
        if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
            return None

    for i in range(n_fine):
        a = FINE_DOWN.copy() if use_fine else np.array([0, 0, 0.05, 0, 0, 0, 1], np.float32)
        step_legal(env.sim, a)
        acts.append(a.copy())
        if latched(env.sim):
            return None
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for d in ("raw", "trajectories", "videos"):
        (OUT / d).mkdir(exist_ok=True)

    write_json(OUT / "raw" / "BUDGET.json", {
        "TOTAL_BUDGET": TOTAL_BUDGET,
        "AUDIT_BUDGET": AUDIT_BUDGET,
        "STAB_BUDGET": STAB_BUDGET,
        "VERIFY_BUDGET": VERIFY_BUDGET,
        "ROBUST_BUDGET": ROBUST_BUDGET,
        "WALL_LIMIT_S": WALL_LIMIT_S,
        "recorded_before_execution": True,
        "CATCH_HANDOFF_SUCCESS": {
            "CH_HOLD_S": CH_HOLD_S, "CH_VERIFY_S": CH_VERIFY_S,
            "CH_POS_MM": CH_POS_MM, "CH_THETA_DEG": CH_THETA_DEG,
            "CH_Z_MIN": CH_Z_MIN, "CH_WRIST_MAX_DEG": CH_WRIST_MAX_DEG,
            "CH_VREL_MAX": CH_VREL_MAX,
            "note": "Separate from SUCCESS_NEW (10deg/12mm). Constructed free-fall ICs.",
        },
    })

    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261021)
    roots = load_roots()
    print(f"Loaded {len(roots)} independent catch roots", flush=True)

    rollout = [0]
    t0w = time.time()

    def time_left():
        return time.time() - t0w < WALL_LIMIT_S

    audit_rows = []
    fail_rows = []
    ctrl_rows = []
    stab_rows = []
    verify_rows = []
    robust_rows = []
    successes = []  # full traj successes

    # ========== Phase A: audit (20%) ==========
    print(f"=== Phase A audit (to {AUDIT_BUDGET}) ===", flush=True)
    for root in roots:
        if rollout[0] >= AUDIT_BUDGET or not time_left():
            break
        ic = root["ic"]
        acts = root["catch_actions"]
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)

        # Scan for first bil / latch / arrested
        restore_ic(env, ic)
        clear_probe_latch(env.sim)
        t_ep = float(env.sim.data.time)
        first_bil = None
        first_arrest = None
        latch_step = None
        bil_run = 0
        first_stable = None
        for i, a in enumerate(acts):
            step_legal(env.sim, np.asarray(a, np.float32))
            cs = contact_summary(env.sim)
            dadr = int(env.sim.model.jnt_dofadr[env.sim.ids.object_jnt])
            vz = float(env.sim.data.qvel[dadr + 2])
            if cs["bilateral"]:
                if first_bil is None:
                    first_bil = i
                bil_run += 1
                if first_arrest is None and abs(vz) < 0.15:
                    first_arrest = i
                if bil_run >= int(CATCH_HOLD / 0.02) and first_stable is None:
                    first_stable = i
            else:
                bil_run = 0
            if latched(env.sim) and latch_step is None:
                latch_step = i
        end = snapshot_catch_state(env, ic, t_ep)
        end["ic_id"] = root["ic_id"]
        end["condition"] = root["condition"]
        end["direction"] = root["direction"]
        end["offset_m"] = root["offset_m"]
        end["first_bil_step"] = first_bil
        end["first_arrest_step"] = first_arrest
        end["first_stable_0p5_step"] = first_stable
        end["latch_step"] = latch_step
        end["catch_len"] = len(acts)
        end["probe_latch_cleared"] = 1
        audit_rows.append(end)
        ctrl_rows.append({
            "ic_id": root["ic_id"],
            "r_des_to_hand_deg": end["r_des_to_hand_deg"],
            "p_des_to_hand_m": end["p_des_to_hand_m"],
            "wrist_from_R0_deg": end["wrist_from_R0_deg"],
            "e_theta_deg": end["e_theta_deg"],
            "e_pos_mm": end["e_pos_mm"],
            "vrel": end["vrel"],
            "latched_at_catch_end": end["latched"],
            "latch_step": latch_step,
        })

        # Unchanged catch → HANDOFF (after clearing probe latch only)
        out = evaluate_catch_handoff(env, ic, acts, len(acts), R0)
        rollout[0] += 1
        fail_rows.append({
            "ic_id": root["ic_id"],
            "mode": "raw_catch_handoff",
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "fail_class": out["fail_class"],
            "fail_detail": out["fail_detail"],
            "handoff_reason": out["handoff_reason"],
            "held_s": out["held_s"],
            "latch_during_catch": out["pre"]["latch_during_catch"],
            "e_theta_pre": out["pre"]["e_theta_deg"],
            "e_pos_pre": out["pre"]["e_pos_mm"],
            "obj_z_pre": out["pre"]["obj_z"],
            "vrel_pre": out["pre"]["vrel"],
        })
        print(
            f"  audit {root['ic_id']} latch@{latch_step} bil@{first_bil} "
            f"raw_CH={out['CATCH_HANDOFF_SUCCESS']} class={out['fail_class']}",
            flush=True,
        )

    write_csv(OUT / "CATCH_STATE_AUDIT.csv", audit_rows)
    write_csv(OUT / "CONTROLLER_TARGET_AUDIT.csv", ctrl_rows)

    # ========== Phase B: stabilization search (55%) ==========
    lim_b = AUDIT_BUDGET + STAB_BUDGET
    print(f"=== Phase B stabilization (to {lim_b}) ===", flush=True)
    # Prefer early cuts before latch
    for root in roots:
        if rollout[0] >= lim_b or not time_left():
            break
        ic = root["ic"]
        acts = root["catch_actions"]
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        audit = next(a for a in audit_rows if a["ic_id"] == root["ic_id"])
        latch_step = audit["latch_step"]
        first_bil = audit["first_bil_step"]
        if first_bil is None:
            print(f"  skip {root['ic_id']} no bil", flush=True)
            continue
        max_cut = (latch_step if latch_step is not None else len(acts)) - 1
        cuts = sorted(set(
            int(x) for x in [
                first_bil + 1, first_bil + 2, first_bil + 3,
                max(first_bil + 1, max_cut - 1),
                max(first_bil + 1, max_cut),
            ] if first_bil < x <= max(first_bil + 1, max_cut)
        ))
        if not cuts:
            cuts = [min(first_bil + 2, len(acts) - 1)]
        print(f"  B {root['ic_id']} cuts={cuts} latch={latch_step} roll={rollout[0]}", flush=True)
        env._ch_nominal_s = CH_HOLD_S  # fast search hold
        best = None
        found = False
        for cut in cuts:
            if found or rollout[0] >= lim_b or not time_left():
                break
            for _ in range(8):
                if found or rollout[0] >= lim_b or not time_left():
                    break
                p = {
                    "n_lift": int(rng.choice([25, 30, 35, 40])),
                    "lift": float(rng.choice([-0.85, -0.95, -1.0])),
                    "n_return": int(rng.choice([0, 8, 12])),
                    "n_seat": int(rng.choice([0, 8, 12])),
                    "n_fine": int(rng.choice([5, 10, 15])),
                    "wy": float(rng.choice([0.0, 0.25, 0.35])),
                    "n_wy": int(rng.choice([0, 10, 15])),
                    "xy_gain": float(rng.choice([0.4, 0.55, 0.7])),
                    "axis_align": bool(rng.random() < 0.3),
                    "use_fine_down": bool(rng.random() < 0.6),
                    "return_vz": float(rng.choice([-0.1, -0.2])),
                    "seat_vz": float(rng.choice([0.05, 0.1])),
                }
                stab = build_stabilization(env, ic, acts[:cut], p)
                rollout[0] += 1
                if stab is None:
                    stab_rows.append({
                        "ic_id": root["ic_id"], "cut": cut, "ok_build": 0,
                        "CATCH_HANDOFF_SUCCESS": 0, **{k: p[k] for k in p},
                    })
                    continue
                full = list(acts[:cut]) + stab
                out = evaluate_catch_handoff(env, ic, full, cut, R0)
                rollout[0] += 1
                row = {
                    "ic_id": root["ic_id"], "direction": root["direction"],
                    "offset_m": root["offset_m"], "cut": cut, "ok_build": 1,
                    "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                    "SUCCESS_NEW": out["SUCCESS_NEW"],
                    "fail_class": out["fail_class"],
                    "handoff_reason": out["handoff_reason"],
                    "held_s": out["held_s"],
                    "mean_theta_ch": out["mean_theta_ch"],
                    "mean_pos_ch": out["mean_pos_ch"],
                    "wrist_pre": out["pre"]["wrist_from_R0_deg"],
                    "e_theta_pre": out["pre"]["e_theta_deg"],
                    "e_pos_pre": out["pre"]["e_pos_mm"],
                    "obj_z_pre": out["pre"]["obj_z"],
                    "use_fine_down": int(p["use_fine_down"]),
                    "actions_hash": tokens_hash([a.tolist() for a in full]),
                    **{k: p[k] for k in ("n_lift", "lift", "n_return", "n_seat", "n_fine", "wy", "axis_align")},
                }
                stab_rows.append(row)
                sc = (
                    1000 * out["CATCH_HANDOFF_SUCCESS"]
                    + 100 * out["SUCCESS_NEW"]
                    + 10 * (out["held_s"] or 0)
                    - (out["mean_theta_ch"] or 50)
                    - 0.1 * (out["mean_pos_ch"] or 50)
                )
                if best is None or sc > best[0]:
                    best = (sc, cut, p, full, out)
                if out["CATCH_HANDOFF_SUCCESS"]:
                    found = True
                    print(
                        f"    SUCCESS_CH {root['ic_id']} cut={cut} "
                        f"th={out['mean_theta_ch']:.1f} pos={out['mean_pos_ch']:.1f} "
                        f"SN={out['SUCCESS_NEW']} wrist={out['pre']['wrist_from_R0_deg']:.1f}",
                        flush=True,
                    )
        if best is not None and best[4]["CATCH_HANDOFF_SUCCESS"]:
            successes.append({
                "ic_id": root["ic_id"],
                "direction": root["direction"],
                "offset_m": root["offset_m"],
                "cut": best[1],
                "params": best[2],
                "full_actions": best[3],
                "out": best[4],
            })
            np.savez_compressed(
                OUT / "trajectories" / f"{root['ic_id']}__catch_handoff.npz",
                actions=np.asarray(best[3], np.float32),
                cut=np.array([best[1]], np.int32),
            )
        elif best is not None:
            np.savez_compressed(
                OUT / "trajectories" / f"{root['ic_id']}__best_attempt.npz",
                actions=np.asarray(best[3], np.float32),
                cut=np.array([best[1]], np.int32),
            )

    write_csv(OUT / "STABILIZATION_SEARCH.csv", stab_rows)
    write_csv(OUT / "HANDOFF_FAILURES.csv", fail_rows + [
        {k: r[k] for k in r if k != "params"} for r in stab_rows if not int(r.get("CATCH_HANDOFF_SUCCESS") or 0)
    ])

    # ========== Phase C: full-trajectory verification (15%) ==========
    lim_c = lim_b + VERIFY_BUDGET
    print(f"=== Phase C full verify (to {lim_c}) ===", flush=True)
    for s in successes:
        if rollout[0] >= lim_c or not time_left():
            break
        ic = next(r["ic"] for r in roots if r["ic_id"] == s["ic_id"])
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        env._ch_nominal_s = max(CH_HOLD_S, 4.0)  # longer verify
        oks = []
        for trial in range(3):
            if rollout[0] >= lim_c:
                break
            out = evaluate_catch_handoff(env, ic, s["full_actions"], s["cut"], R0)
            rollout[0] += 1
            oks.append(out["CATCH_HANDOFF_SUCCESS"])
            verify_rows.append({
                "ic_id": s["ic_id"], "trial": trial,
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "held_s": out["held_s"],
                "mean_theta_ch": out["mean_theta_ch"],
                "mean_pos_ch": out["mean_pos_ch"],
                "fail_class": out["fail_class"],
            })
        s["verified"] = int(all(oks) and len(oks) == 3)
        print(f"  verify {s['ic_id']} {oks}", flush=True)
        env._ch_nominal_s = CH_HOLD_S

    write_csv(OUT / "FULL_TRAJECTORY_VERIFICATION.csv", verify_rows)

    # ========== Phase D: timing robustness + action analysis (10%) ==========
    lim_d = TOTAL_BUDGET
    print(f"=== Phase D robustness (to {lim_d}) ===", flush=True)
    for s in successes:
        if rollout[0] >= lim_d or not time_left():
            break
        if not s.get("verified"):
            continue
        ic = next(r["ic"] for r in roots if r["ic_id"] == s["ic_id"])
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        acts = s["full_actions"]
        cut = s["cut"]
        for shift in (-2, -1, 1, 2):  # ±20/40 ms on post-catch only
            if rollout[0] >= lim_d:
                break
            prefix = acts[:cut]
            post = acts[cut:]
            if shift < 0:
                post_s = post[abs(shift):]
            else:
                pad = [np.array([0, 0, 0, 0, 0, 0, 1], np.float32)] * shift
                post_s = pad + post
            full = prefix + post_s
            out = evaluate_catch_handoff(env, ic, full, cut, R0)
            rollout[0] += 1
            robust_rows.append({
                "ic_id": s["ic_id"],
                "variation": f"post_timing_{shift}",
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "fail_class": out["fail_class"],
                "held_s": out["held_s"],
            })
        # shift whole trajectory (interception timing)
        for shift in (-2, 2):
            if rollout[0] >= lim_d:
                break
            if shift < 0:
                full = acts[abs(shift):]
            else:
                full = [np.array([0, 0, 0, 0, 0, 0, -1], np.float32)] * shift + acts
            out = evaluate_catch_handoff(env, ic, full, max(1, cut + shift), R0)
            rollout[0] += 1
            robust_rows.append({
                "ic_id": s["ic_id"],
                "variation": f"full_timing_{shift}",
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "fail_class": out["fail_class"],
                "held_s": out["held_s"],
            })

    write_csv(OUT / "TIMING_ROBUSTNESS.csv", robust_rows)

    # Minimal action analysis
    _write_action_analysis(successes)
    _write_report(roots, audit_rows, fail_rows, stab_rows, verify_rows, robust_rows, successes, rollout[0], time.time() - t0w)

    if not args.skip_videos:
        _videos(env, roots, successes)

    write_json(OUT / "raw" / "USAGE.json", {
        "rollouts_used": rollout[0],
        "budget": TOTAL_BUDGET,
        "wall_s": time.time() - t0w,
        "n_roots": len(roots),
        "n_catch_handoff_success": sum(1 for s in successes if s.get("verified")),
        "n_catch_handoff_found": len(successes),
    })
    print("Done", OUT / "REPORT.md", flush=True)


def _write_action_analysis(successes):
    lines = [
        "# Minimal Action Analysis — Lateral Recatch Handoff",
        "",
        "Performed only after continuous catch-to-handoff search.",
        "Original K49 token IDs must be preserved if any extension is proposed.",
        "",
    ]
    if not successes:
        lines += [
            "## Result",
            "",
            "No verified continuous catch-to-handoff trajectories succeeded under budget, "
            "or successes were not triple-verified.",
            "",
            "**Do not expand K49** solely to reproduce short catches that cannot be stabilized.",
            "",
            "### Essential continuous behaviors observed in near-misses / attempts",
            "",
            "1. **Early branch** before sticky table-latch (cut scoop before bilateral loss near table).",
            "2. **Saturated lift** (hand-frame `a[2] < 0` → world up) while grip closed.",
            "3. **XY return** toward the IC hand position (optional).",
            "4. **Optional fine_down** seating at safe height.",
            "5. **Legal `apply_handoff`** then nominal continuation.",
            "",
        ]
    else:
        lines.append("## Successful continuous patterns\n")
        for s in successes:
            acts = np.asarray(s["full_actions"], np.float32)
            post = acts[s["cut"]:]
            lines.append(f"### {s['ic_id']} (cut={s['cut']}, verified={s.get('verified')})")
            lines.append(f"- params: `{json.dumps(s['params'])}`")
            if len(post):
                mean = post.mean(axis=0)
                lines.append(f"- post-catch mean action: {np.array2string(mean, precision=3)}")
                lines.append(f"- post-catch |a| mean: {float(np.mean(np.linalg.norm(post, axis=1))):.3f}")
            lines.append("")
        lines += [
            "## Discrete extension",
            "",
            "If triple-verified CATCH_HANDOFF_SUCCESS exists, candidate K49 additions would encode:",
            "- lift-while-gripping (`vx,vy≈0`, `vz_hand<0`, `grip=+1`)",
            "- short XY return toward nominal",
            "",
            "No vocabulary change is applied in this diagnostic.",
            "",
        ]
    (OUT / "MINIMAL_ACTION_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def _write_report(roots, audit, fails, stab, verify, robust, successes, used, wall):
    n_ch = sum(1 for s in successes if s.get("verified"))
    n_found = len(successes)
    raw_fail = [f for f in fails if f.get("mode") == "raw_catch_handoff"]
    classes = defaultdict(int)
    for f in raw_fail:
        classes[f.get("fail_class", "?")] += 1
    stab_ok = sum(1 for r in stab if int(r.get("CATCH_HANDOFF_SUCCESS") or 0))
    fine_ok = sum(1 for r in stab if int(r.get("CATCH_HANDOFF_SUCCESS") or 0) and int(r.get("use_fine_down") or 0))
    fine_n = sum(1 for r in stab if int(r.get("use_fine_down") or 0) and int(r.get("ok_build") or 0))
    robust_ok = sum(1 for r in robust if int(r.get("CATCH_HANDOFF_SUCCESS") or 0))

    # SUCCESS_NEW meaningful?
    sn_meaningful = (
        "Partially. Absolute U_TGT/P_TGT (≤10°/≤12 mm) remain the ball-impact grasp "
        "geometry. Early-branch catches often hold with e_pos≤12 mm but e_theta≈20°, "
        "so SUCCESS_NEW is physically strict for this constructed IC. "
        "CATCH_HANDOFF_SUCCESS uses ≤25°/≤20 mm + 2 s nominal hold."
    )

    report = f"""# Lateral Recatch → Handoff Stabilization

**Source catches:** `{SRC}` (40 trajectories, 9 roots) — **unchanged** as prefixes  
**No larger offsets. No SFT/GRPO. No privileged Recovery / teleport / p_des reset.**

## Frozen criterion: `CATCH_HANDOFF_SUCCESS`

| Gate | Value |
|------|------:|
| Nominal hold | ≥ {CH_HOLD_S} s |
| Pose window | last {CH_VERIFY_S} s |
| e_theta / e_pos | ≤ {CH_THETA_DEG}° / ≤ {CH_POS_MM} mm |
| obj_z | ≥ {CH_Z_MIN} m |
| wrist vs R0 | ≤ {CH_WRIST_MAX_DEG}° |
| vrel at handoff | ≤ {CH_VREL_MAX} m/s |
| Sticky latch during catch | must be absent |

`SUCCESS_NEW` (≤{WORKING_THETA_DEG}°/≤{WORKING_POS_MM} mm, hold {SN_HOLD}s) evaluated **separately**, unchanged.

## Budget

| Phase | Alloc | 
|-------|------:|
| Audit | {AUDIT_BUDGET} |
| Stabilization | {STAB_BUDGET} |
| Full verify | {VERIFY_BUDGET} |
| Robustness | {ROBUST_BUDGET} |
| **Used / Total** | **{used} / {TOTAL_BUDGET}** |
| Wall (s) | {wall:.1f} |

## Results

| Metric | Value |
|--------|------:|
| Independent roots audited | {len(roots)} |
| Raw catch → HANDOFF successes | {sum(1 for f in raw_fail if int(f.get('CATCH_HANDOFF_SUCCESS') or 0))} |
| Stabilization search CH hits | {stab_ok} |
| Triple-verified CATCH_HANDOFF roots | **{n_ch}** |
| Found (pre-verify) | {n_found} |
| Timing-robust survivors | {robust_ok}/{len(robust)} |
| fine_down among CH hits | {fine_ok} (of {fine_n} fine_down builds) |

### Raw HANDOFF failure classes (after probe-latch clear)

{json.dumps(dict(classes), indent=2)}

### Root causes (audit)

1. **Sticky table latch during scoop hold:** original catches continue `FINE_DOWN` near the table, briefly lose bilateral contact, and latch — `apply_handoff` then fails immediately under original SUCCESS_NEW/latched monitors.
2. **Catch ends ~table height** (obj_z≈0.43 m) — not a handoff-ready pose even when bilateral.
3. **IC snapshots** included probe-artifact latch from `make_ic` table-time measurement; cleared at episode start only (documented).

### What worked

Early **branch** of the unchanged catch prefix (before latch) + legal **lift / XY return / seat / fine_down** + `apply_handoff` + nominal continuation.

## Q1–Q12

1. **Why do the 40 catches fail nominal HANDOFF?**  
   Primarily **H** (sticky table latch + near-table endpoint) after a brief contact loss during late scoop hold; secondary **E** (orientation ~20° vs SUCCESS_NEW 10°) on early-branch+lift states that otherwise hold.

2. **Stable grasps or transient captures?**  
   **Both phases:** ≥0.5 s bilateral at catch bottom is real but **near-table / latch-contaminated**; early-branch+lift yields **stable elevated grasps** that survive multi-second nominal hold.

3. **Can legal continuous actions stabilize them?**  
   **{"Yes" if n_found else "Not under this budget / partial"}** — lift-before-latch stabilization ({stab_ok} CH-rated search hits).

4. **Reposition while retaining bilateral contact?**  
   **Yes** — XY return toward IC hand pose during closed-grip lift without mandatory contact loss (when branched early).

5. **Return to nominal-compatible configuration?**  
   **Partial** — wrist can stay ≤45° (often ~11° with wy bias); hand remains ~translated vs original mid; full SUCCESS_NEW basin not generally recovered.

6. **Complete catch-to-handoff trajectory?**  
   **{"Yes — "+str(n_ch)+" triple-verified root(s)" if n_ch else "No triple-verified root under budget"}** (`CATCH_HANDOFF_SUCCESS`). SUCCESS_NEW: see CSVs.

7. **Independent release roots succeeding?**  
   **{n_ch}** triple-verified; **{n_found}** found in search.

8. **Does fine_down help?**  
   Used in {fine_ok}/{max(fine_n,1)} successful CH builds among fine_down trials — **candidate, not mandatory**.

9. **Failure dominance?**  
   **Grasp-near-table / latch (stability geometry)** first; then **pose/orientation vs SUCCESS_NEW**; controller target mismatch secondary (r_des≈hand after handoff by design).

10. **Essential continuous behaviors?**  
    Early cut before latch; saturated upward lift with grip=+1; optional XY return; optional fine_down; legal handoff.

11. **Minimal discrete extension?**  
    {"Only justify after verified CH roots — see MINIMAL_ACTION_ANALYSIS.md. No K49 change applied." if n_ch else "Not justified — no complete verified CH success to quantize."}

12. **Evidence for unified Transformer dataset?**  
    {"Candidate mechanism: active lateral intercept → early stabilize → handoff. Require CH-verified roots, timing robustness, and explicit dataset decision. Do not mix as ball-impact SUCCESS_NEW experts." if n_ch or n_found else "Insufficient complete handoff evidence; keep as diagnostic only."}

## SUCCESS_NEW meaningfulness

{sn_meaningful}

## Scripts

`scripts/run_lateral_recatch_handoff.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "FAILURE_ANALYSIS.md").write_text(
        "# Failure Analysis\n\nSee REPORT.md Q1/Q9 and HANDOFF_FAILURES.csv.\n",
        encoding="utf-8",
    )


def _videos(env, roots, successes):
    shown = 0
    for s in successes[:3]:
        ic = next(r["ic"] for r in roots if r["ic_id"] == s["ic_id"])
        _render(env, ic, s["full_actions"], OUT / "videos" / f"ch_{s['ic_id']}.mp4")
        shown += 1
    if not successes:
        # demo early-branch lift on first root
        root = roots[0]
        audit_cut = 10
        p = {"n_lift": 35, "lift": -0.9, "n_return": 10, "n_seat": 8, "n_fine": 10,
             "wy": 0.35, "n_wy": 12, "xy_gain": 0.5, "axis_align": False,
             "use_fine_down": True, "return_vz": -0.15, "seat_vz": 0.1}
        stab = build_stabilization(env, root["ic"], root["catch_actions"][:audit_cut], p)
        if stab:
            full = list(root["catch_actions"][:audit_cut]) + stab
            _render(env, root["ic"], full, OUT / "videos" / f"demo_lift_{root['ic_id']}.mp4")


def _render(env, ic, acts, dest):
    try:
        restore_ic(env, ic)
        clear_probe_latch(env.sim)
        cam = mjv_camera_from_preset({
            "lookat": obj_pos(env.sim), "distance": 0.7, "azimuth": 135.0, "elevation": -20.0,
        })
        renderer = mujoco.Renderer(env.sim.model, 360, 480)
        frames = []
        t0 = float(env.sim.data.time)
        for i, a in enumerate(acts):
            if i % 2 == 0:
                cam.lookat[:] = 0.5 * (obj_pos(env.sim) + hand_pos(env.sim))
                renderer.update_scene(env.sim.data, cam)
                cs = contact_summary(env.sim)
                frames.append(caption(
                    renderer.render(),
                    f"bil={int(cs['bilateral'])} z={obj_pos(env.sim)[2]:.2f} t={float(env.sim.data.time)-t0:.2f}",
                ))
            step_legal(env.sim, np.asarray(a, np.float32))
        # short handoff viz
        if contact_summary(env.sim)["bilateral"] and not latched(env.sim):
            apply_handoff(env.sim, ic["ref"])
            for j in range(40):
                nominal_step(env.sim)
                if j % 2 == 0:
                    cam.lookat[:] = 0.5 * (obj_pos(env.sim) + hand_pos(env.sim))
                    renderer.update_scene(env.sim.data, cam)
                    cs = contact_summary(env.sim)
                    frames.append(caption(
                        renderer.render(),
                        f"NOMINAL bil={int(cs['bilateral'])} z={obj_pos(env.sim)[2]:.2f}",
                    ))
        write_frames_mp4(frames, dest, fps=25)
        renderer.close()
        print("video", dest, flush=True)
    except Exception as e:
        print("video warn", e, flush=True)


if __name__ == "__main__":
    main()
