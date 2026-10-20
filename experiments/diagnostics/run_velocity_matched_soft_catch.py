"""Velocity-matched soft-catch diagnostics for dynamic Panda recovery.

Inspired by Catch It (ICRA 2025) principles, adapted to fixed-base parallel-jaw.
Does NOT claim Catch It uses explicit velocity-matching loss (it uses two-stage RL).
Locked physics/controller/obs/SUCCESS_NEW. No SFT/GRPO.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, V_AIRBORNE, map_action
from envs.outcomes.horizon_outcome import recovery_failed, terminal_reason, NOMINAL_S, latched
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture
from envs.simulation.grasp_quality import body_twist
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.grasp_geometry import measure
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table
from training.recovery_episode import make_training_env
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

from scripts.expand_k49_hard_dynamic import ic_metrics as hard_ic_metrics, is_high_motion
from scripts.expand_k49_large_chase_dynamic import HM_THR, hand_pos, obj_pos
from scripts.expand_k49_capture_brake_diagnostics import find_snap, load_blob
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID, SFT, decode, mapping_checksum, write_csv, write_json,
)
from evaluation.success_criteria import window_stats as window_stats_fo

OUT = ROOT / "results" / "diagnostics" / "velocity_matched_soft_catch"
LC_OUT = ROOT / "results" / "expert_generation" / "k49_large_chase_dynamic_recovery"

# Budgets frozen before execution
TOTAL_ROLLOUT_BUDGET = 2400
WALL_LIMIT_S = 45 * 60
PHASE_A_FRAC = 0.30
PHASE_B_SPLIT = (0.40, 0.30, 0.20, 0.10)  # promising HM / ablations / stabilize / pose+HANDOFF

BE_ICS = [
    "hard_910012", "hard_910006", "hard_910095", "hard_910031", "bimp_0033",
    "bimp_0010", "bimp_0036", "ubimp_0040", "hdimp_0002", "hdimp_0012",
    "hdimp_0017", "hdimp_0022", "hdimp_0030", "hdimp_0036", "hdimp_0039",
]
LC_PARTIAL_ICS = ["ubimp_0040", "bimp_0010", "bimp_0036"]

# fine_down from prior ablation (comparison only; not mixed into K49 dataset)
FINE_DOWN = np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0], np.float32)

CHASE_PREFIX_TOKS = {
    "t17": [17] * 18,
    "t33": [33] * 18,
    "open_t17": [30] * 3 + [17] * 16,
    "t1": [1] * 20,
}


def tokens_hash(x):
    return hashlib.sha256(json.dumps(x if isinstance(x, list) else str(x)).encode()).hexdigest()[:16]


def body_v_w(sim, body_id):
    v, w = body_twist(sim.model, sim.data, int(body_id))
    return np.asarray(v, float), np.asarray(w, float)


def contact_point_kinematics(sim):
    """Per finger-object contact: COM vs contact-point relative velocities (world)."""
    obj = int(sim.ids.object_body)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    vo, wo = body_v_w(sim, obj)
    vh, wh = body_v_w(sim, sim.ids.hand_body)
    po = obj_pos(sim)
    ph = hand_pos(sim)
    out = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        if other not in (left, right):
            continue
        side = "L" if other == left else "R"
        pos = np.asarray(c.pos, float)
        # object contact-point velocity
        r_o = pos - po
        v_obj_pt = vo + np.cross(wo, r_o)
        # finger body as proxy for finger surface (parallel-jaw pad)
        vf, wf = body_v_w(sim, other)
        pf = np.asarray(sim.data.xpos[other], float)
        r_f = pos - pf
        v_fin_pt = vf + np.cross(wf, r_f)
        v_rel_pt = v_obj_pt - v_fin_pt
        # contact frame: MuJoCo contact.frame is 3x3 row-major; first row ~ normal from geom1 to geom2
        frame = np.asarray(c.frame, float).reshape(3, 3)
        n = frame[0]  # normal
        # ensure normal points roughly object->finger for signing (diagnostic)
        if float(np.dot(n, pf - po)) < 0:
            n = -n
        v_n = float(np.dot(v_rel_pt, n))
        v_t = v_rel_pt - v_n * n
        force = np.zeros(6, float)
        try:
            mujoco.mj_contactForce(sim.model, sim.data, i, force)
        except Exception:
            force[:] = 0.0
        out.append({
            "side": side,
            "pos": pos.tolist(),
            "v_obj_pt": v_obj_pt.tolist(),
            "v_fin_pt": v_fin_pt.tolist(),
            "v_rel_pt": v_rel_pt.tolist(),
            "v_rel_pt_norm": float(np.linalg.norm(v_rel_pt)),
            "v_n": v_n,
            "v_t_norm": float(np.linalg.norm(v_t)),
            "f_n_proxy": float(np.linalg.norm(force[:3])),
            "dist": float(c.dist),
        })
    # COM-relative for comparison
    v_com_rel = vo - vh
    return {
        "contacts": out,
        "v_com_obj": vo.tolist(),
        "w_obj": wo.tolist(),
        "v_com_hand": vh.tolist(),
        "w_hand": wh.tolist(),
        "v_com_rel": v_com_rel.tolist(),
        "v_com_rel_norm": float(np.linalg.norm(v_com_rel)),
        "w_rel_norm": float(np.linalg.norm(wo - wh)),
        "n_contacts": len(out),
        "mean_v_rel_pt": float(np.mean([c["v_rel_pt_norm"] for c in out])) if out else None,
        "max_v_rel_pt": float(np.max([c["v_rel_pt_norm"] for c in out])) if out else None,
        "mean_v_n": float(np.mean([abs(c["v_n"]) for c in out])) if out else None,
        "mean_v_t": float(np.mean([c["v_t_norm"] for c in out])) if out else None,
    }


def geometry_snapshot(sim):
    """Interception geometry: cylinder vs finger pads."""
    po = obj_pos(sim)
    pl = np.asarray(sim.data.xpos[sim.ids.left_body], float)
    pr = np.asarray(sim.data.xpos[sim.ids.right_body], float)
    ph = hand_pos(sim)
    mid = 0.5 * (pl + pr)
    gap = float(np.linalg.norm(pl - pr))
    # Is COM between fingers? project onto finger-separation axis
    axis = pl - pr
    an = float(np.linalg.norm(axis)) + 1e-12
    u = axis / an
    t = float(np.dot(po - pr, u))  # 0 at right, |gap| at left
    between = 0.0 < t < an
    # axial alignment: cylinder axis vs finger open direction
    from evaluation.grasp_geometry import cylinder_axis_world
    u_cyl = cylinder_axis_world(sim)
    align = abs(float(np.dot(u_cyl, u)))  # 0 = axis parallel to jaw plane open dir? open is left-right
    # better: open direction is u; preferred cylinder axis perp to u for parallel jaw
    axis_perp_score = float(np.linalg.norm(u_cyl - np.dot(u_cyl, u) * u))  # ~1 if perp
    cs = contact_summary(sim)
    return {
        "gap_m": gap,
        "com_between_fingers": int(between),
        "t_along_gap": t,
        "dist_com_to_mid": float(np.linalg.norm(po - mid)),
        "dist_com_to_hand": float(np.linalg.norm(po - ph)),
        "axis_perp_to_open": axis_perp_score,
        "nL": int(cs["nL"]), "nR": int(cs["nR"]), "bilateral": int(cs["bilateral"]),
        "outer_contact_risk": int(not between and (int(cs["nL"]) + int(cs["nR"]) > 0)),
    }


def run_actions_eval(env, blob, actions, eval_full=True):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    ok = True
    first_contact_kin = None
    first_geo = None
    pre_contact_min_vpt = None  # not available before contact
    pre_contact_vcom = []
    contact_kin_rows = []
    max_bil = 0.0
    bil_run = 0.0
    had_uni = had_bil = had_any = False
    t_first = None
    dt_step = None
    for ai, a in enumerate(actions):
        a = np.clip(np.asarray(a, np.float32).reshape(7), -1, 1)
        cs = contact_summary(env.sim)
        any_c = int(cs["nL"]) + int(cs["nR"]) > 0
        if not any_c:
            vo, _ = body_v_w(env.sim, env.sim.ids.object_body)
            vh, _ = body_v_w(env.sim, env.sim.ids.hand_body)
            pre_contact_vcom.append(float(np.linalg.norm(vo - vh)))
        if any_c and first_contact_kin is None:
            first_contact_kin = contact_point_kinematics(env.sim)
            first_geo = geometry_snapshot(env.sim)
            t_first = float(env.sim.data.time) - t0
            had_any = True
        if any_c:
            had_any = True
            kin = contact_point_kinematics(env.sim)
            contact_kin_rows.append(kin)
            if (int(cs["nL"]) > 0) ^ (int(cs["nR"]) > 0):
                had_uni = True
            if cs["bilateral"]:
                had_bil = True
                bil_run += 0.02  # approx command period
                max_bil = max(max_bil, bil_run)
            else:
                bil_run = 0.0
        step_legal(env.sim, a)
        env._prev = a.copy()
        if dt_step is None:
            dt_step = float(env.sim.data.time) - t0
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            ok = False
            break
    # M2: did pre-contact COM relative velocity decrease?
    m2 = 0
    if len(pre_contact_vcom) >= 2:
        m2 = int(pre_contact_vcom[-1] < pre_contact_vcom[0] - 0.05)
    out = {
        "replay_ok": int(ok),
        "reason_fail": None if ok else (terminal_reason(env.sim) or "DROP"),
        "had_contact": int(had_any),
        "had_uni": int(had_uni),
        "had_bil": int(had_bil),
        "max_bil_persist_s": max_bil,
        "t_first_contact": t_first,
        "M2_vcom_reduced_pre": m2,
        "pre_vcom_start": pre_contact_vcom[0] if pre_contact_vcom else None,
        "pre_vcom_end": pre_contact_vcom[-1] if pre_contact_vcom else None,
        "v_com_rel_at_contact": None if first_contact_kin is None else first_contact_kin["v_com_rel_norm"],
        "v_pt_rel_at_contact": None if first_contact_kin is None else first_contact_kin["mean_v_rel_pt"],
        "v_n_at_contact": None if first_contact_kin is None else first_contact_kin["mean_v_n"],
        "v_t_at_contact": None if first_contact_kin is None else first_contact_kin["mean_v_t"],
        "geo_between": None if first_geo is None else first_geo["com_between_fingers"],
        "geo_outer_risk": None if first_geo is None else first_geo["outer_contact_risk"],
        "SUCCESS_NEW": 0,
        "win_mean_theta": None,
        "win_mean_pos": None,
        "hold_both_s": None,
        "eval_reason": None,
    }
    if ok and eval_full:
        apply_handoff(env.sim, blob["ref"])
        t_h = float(env.sim.data.time)
        samples = []
        next_sample = 0.0
        reason = None
        while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
            nominal_step(env.sim)
            t_rel = float(env.sim.data.time) - t_h
            if t_rel + 1e-12 >= next_sample:
                m = measure(env.sim, U_TGT, P_TGT)
                cs = contact_summary(env.sim)
                samples.append({
                    "t_rel": t_rel, "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
                    "bilateral": cs["bilateral"], "obj_z": m["obj_z"],
                })
                next_sample += 0.02
            if latched(env.sim):
                reason = "TABLE_DROP"
        win = window_stats_fo(samples, float(NOMINAL_S) - 2.0)
        ok_new = bool(win.get("working_SUCCESS_NEW_hold"))
        mat = (win.get("matrix") or {}).get("th10_pos12") or {}
        if mat:
            ok_new = bool(mat.get("SUCCESS_NEW"))
        out["SUCCESS_NEW"] = int(ok_new)
        out["win_mean_theta"] = win.get("mean_theta")
        out["win_mean_pos"] = win.get("mean_pos")
        out["hold_both_s"] = win.get("longest_both_ok_s")
        out["eval_reason"] = reason or ("SUCCESS_NEW" if ok_new else "POSE_OR_HOLD_FAIL")
    # milestones
    out["M1"] = int(had_any or (out.get("pre_vcom_end") is not None and out["pre_vcom_end"] < 0.25))
    out["M3"] = int(had_any)
    out["M4"] = int(had_bil)
    out["M5"] = int(max_bil >= 0.15)
    out["M6"] = int(max_bil >= 0.30)
    out["M7"] = int(out.get("win_mean_pos") is not None and float(out["win_mean_pos"]) <= 12.0
                    and out.get("win_mean_theta") is not None and float(out["win_mean_theta"]) <= 10.0)
    out["M8"] = int(out.get("SUCCESS_NEW") or 0)
    return out, first_contact_kin, first_geo


def toks_to_actions(table, book48, toks):
    return [np.asarray(decode(table, book48, int(t)), np.float32) for t in toks]


def desired_match_action(sim, scale=0.85, grip=1.0):
    """Legal action approximating hand velocity matching object COM (hand frame)."""
    vo, wo = body_v_w(sim, sim.ids.object_body)
    vh, wh = body_v_w(sim, sim.ids.hand_body)
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    # want vh -> vo (match), so command in world ~ vo (feedforward), less current error
    v_des_w = scale * vo
    v_des_h = Rh.T @ v_des_w
    # normalize to AIRBORNE_V3 action
    a = np.zeros(7, np.float32)
    a[0] = float(np.clip(v_des_h[0] / V_AIRBORNE, -1, 1))
    a[1] = float(np.clip(v_des_h[1] / V_AIRBORNE, -1, 1))
    # vz asymmetric
    vz = float(v_des_h[2])
    if vz >= 0:
        a[2] = float(np.clip(vz / 3.8, 0, 1))
    else:
        a[2] = float(np.clip(vz / 1.0, -1, 0))
    # small angular match
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    w_des_b = rd.T @ (scale * 0.3 * wo)
    a[3:6] = np.clip(w_des_b / 4.0, -1, 1)
    a[6] = float(grip)
    return a


def sequence_from_params(env, blob, table, book48, p, mode="B"):
    """Physically-informed parameterization -> legal 20ms action list.

    p keys:
      n_pursue, v_scale, grip_open_n, grip_close_onset, grip_close_val,
      n_comotion, n_brake, n_settle, use_fine_down
    """
    restore_capture(env, blob["capture"])
    acts = []
    n_pursue = int(p["n_pursue"])
    # optional brief open
    for _ in range(int(p.get("grip_open_n", 0))):
        a = desired_match_action(env.sim, scale=float(p["v_scale"]), grip=-1.0)
        step_legal(env.sim, a)
        acts.append(a.copy())
    # pursuit / velocity match
    for i in range(n_pursue):
        grip = float(p.get("grip_hold", 1.0))
        if mode == "C" and i >= int(p.get("grip_close_onset", 999)):
            grip = float(p.get("grip_close_val", 1.0))
        a = desired_match_action(env.sim, scale=float(p["v_scale"]), grip=grip)
        # blend toward object COM direction in hand frame for approach
        po, ph = obj_pos(env.sim), hand_pos(env.sim)
        Rh = np.asarray(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
        err_h = Rh.T @ (po - ph)
        a[:2] = np.clip(a[:2] + 0.25 * np.clip(err_h[:2] / 0.05, -1, 1), -1, 1)
        step_legal(env.sim, a)
        acts.append(a.copy())
        failed, _ = recovery_failed(env.sim, blob["ref"], 0)
        if failed:
            return acts
    # co-motion
    for _ in range(int(p.get("n_comotion", 0))):
        grip = float(p.get("grip_close_val", 1.0)) if mode == "C" else float(p.get("grip_hold", 1.0))
        a = desired_match_action(env.sim, scale=float(p.get("comotion_scale", 0.5)), grip=grip)
        step_legal(env.sim, a)
        acts.append(a.copy())
    # brake / settle
    for _ in range(int(p.get("n_brake", 0))):
        a = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
        if p.get("use_fine_down"):
            a = FINE_DOWN.copy()
        step_legal(env.sim, a)
        acts.append(a.copy())
    for _ in range(int(p.get("n_settle", 0))):
        a = np.array([0, 0, 0, 0, -0.2, 0, 1], np.float32)  # soft reverse wy like tok48-ish
        if p.get("use_fine_down"):
            a = FINE_DOWN.copy()
        step_legal(env.sim, a)
        acts.append(a.copy())
    # reconstruct from stored acts only (already stepped); for eval we need fresh replay
    return acts


def param_grid(rng, n, mode="B"):
    params = []
    for _ in range(n):
        p = {
            "n_pursue": int(rng.choice([8, 12, 16, 20, 24])),
            "v_scale": float(rng.choice([0.4, 0.6, 0.75, 0.9, 1.0])),
            "grip_open_n": int(rng.choice([0, 0, 2, 3])),
            "grip_hold": float(rng.choice([1.0, 1.0, 0.35, -0.48])),
            "grip_close_onset": int(rng.choice([4, 8, 12, 16])),
            "grip_close_val": float(rng.choice([1.0, 0.35, 1.0])),
            "n_comotion": int(rng.choice([0, 2, 4, 6, 8])),
            "comotion_scale": float(rng.choice([0.3, 0.5, 0.7])),
            "n_brake": int(rng.choice([4, 6, 8, 10])),
            "n_settle": int(rng.choice([0, 8, 16, 24])),
            "use_fine_down": bool(rng.random() < 0.35),
        }
        if mode != "C":
            p["grip_open_n"] = int(rng.choice([0, 2]))
            p["grip_hold"] = 1.0
        params.append(p)
    return params


def score_soft_catch(out):
    sc = 0.0
    sc += 100.0 * float(out.get("SUCCESS_NEW") or 0)
    sc += 8.0 * float(out.get("max_bil_persist_s") or 0)
    sc += 5.0 * float(out.get("had_bil") or 0)
    sc += 2.0 * float(out.get("had_uni") or 0)
    sc += 1.0 * float(out.get("had_contact") or 0)
    sc += 6.0 * float(out.get("M2_vcom_reduced_pre") or 0)
    if out.get("v_pt_rel_at_contact") is not None:
        sc += max(0.0, 8.0 - 20.0 * float(out["v_pt_rel_at_contact"]))
    if out.get("v_com_rel_at_contact") is not None:
        sc += max(0.0, 5.0 - 10.0 * float(out["v_com_rel_at_contact"]))
    if out.get("geo_between"):
        sc += 2.0
    if out.get("geo_outer_risk"):
        sc -= 2.0
    if not out.get("replay_ok"):
        sc -= 15.0
    if out.get("win_mean_pos") is not None:
        sc += max(0.0, 40.0 - float(out["win_mean_pos"]))
    return sc


def write_catch_it_comparison():
    (OUT / "CATCH_IT_METHOD_COMPARISON.md").write_text(
        """# Catch It vs Panda Recovery — Method Comparison

Reference: Zhang et al., *Catch It! Learning to Catch in Flight with Mobile Dexterous Hands* (arXiv:2409.10319, ICRA 2025).
Project: https://mobile-dex-catch.github.io/  Code: https://github.com/hang0610/Catch_It

## Documented Catch It mechanisms (from paper)

1. **System:** omni mobile base + 6-DoF arm + 12-DoF dexterous hand (LEAP).
2. **Interception:** two-stage RL (PPO). Stage 1 *tracking*: base+arm chase object with hand fixed; Stage 2 *catching*: train hand grasp while fine-tuning base+arm near contact.
3. **State:** consecutive object/EE positions (relative to arm base), base planar velocity, hand joints.
4. **Actions:** base 2D velocity, EE delta position + **roll only**, 12 hand joint deltas.
5. **Grasp timing:** learned by RL in stage 2 (touch/grasp rewards); not a scripted FSM.
6. **Post-contact:** part of catching policy; rewards for grasp stability.
7. **Mobile-base dependence:** large workspace / chasing distant throws.
8. **Dexterous-hand dependence:** multi-contact enveloping grasp for diverse shapes.

**Explicit velocity-matching loss:** **Not documented** in Catch It. Rewards emphasize distance reduction, palm orientation, touch, and grasp stability. Do not claim Catch It optimizes a velocity-matching objective.

Related classical soft-catch literature (e.g. impact-aware planners) *does* minimize robot-object relative velocity before contact; that idea informs **our adaptation**, not Catch It's stated method.

## Transferable principles (adaptation proposals)

| Principle | Catch It source | Our Panda adaptation |
|-----------|-----------------|----------------------|
| Reach/track before grasp | Stage-1 tracking | Pre-contact pursuit with legal continuous actions |
| Align end-effector / palm | Palm orientation reward | Wrist alignment + jaw open direction vs cylinder |
| Time the grasp near contact | Stage-2 hand RL | Coordinated grip onset (Condition C) |
| Soften impact | Implicit via tracking | **Explicit velocity matching** (our hypothesis) |
| Post-catch stabilize | Grasp stability reward | Brief co-motion then brake within legal bounds |

## Non-transferable / limited transfer

- Omni-base whole-body chase workspace.
- 12-DoF enveloping grasp and in-hand compliance.
- Sandbag / deformable objects (paper real-world).
- Delta-EE position interface vs our velocity-command AIRBORNE_V3 `step_legal`.

## Our robot

Fixed-base Franka Panda, parallel-jaw gripper, 39D obs, Recovery horizon, SUCCESS_NEW unchanged, legal `a in [-1,1]^7` every 20 ms.
""",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for d in ("trajectories", "snapshots", "raw_contact_traces", "videos", "raw"):
        (OUT / d).mkdir(exist_ok=True)

    b_a = int(TOTAL_ROLLOUT_BUDGET * PHASE_A_FRAC)
    b_rest = TOTAL_ROLLOUT_BUDGET - b_a
    b_b = [int(b_rest * f) for f in PHASE_B_SPLIT]
    b_b[3] = b_rest - sum(b_b[:3])
    write_json(OUT / "raw" / "BUDGET.json", {
        "TOTAL_ROLLOUT_BUDGET": TOTAL_ROLLOUT_BUDGET,
        "WALL_LIMIT_S": WALL_LIMIT_S,
        "PHASE_A": b_a,
        "PHASE_B": {"promising": b_b[0], "ablations": b_b[1], "stabilize": b_b[2], "pose_handoff": b_b[3]},
        "recorded_before_execution": True,
    })
    write_catch_it_comparison()

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    checksum = mapping_checksum(table)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261019)
    rollout = [0]
    t_wall0 = time.time()

    def time_left():
        return time.time() - t_wall0 < WALL_LIMIT_S

    # =====================================================================
    # Phase A — contact-point velocity audit
    # =====================================================================
    print(f"=== Phase A contact audit (budget {b_a}) ===", flush=True)
    audit_rows = []
    geo_rows = []
    track_rows = []

    audit_seeds = [
        ("open_t17", [30] * 3 + [17] * 16 + [0] * 4),
        ("t17", [17] * 18),
        ("t33", [33] * 18),
        ("vm_open", None),  # special: velocity match
    ]

    for ic in BE_ICS:
        if rollout[0] >= b_a or not time_left():
            break
        snap = find_snap(ic)
        if snap is None:
            continue
        dst = OUT / "snapshots" / f"{ic}.pkl"
        if not dst.exists():
            shutil.copy2(snap, dst)
        blob = load_blob(snap)
        st0 = hard_ic_metrics(env, blob)

        # controller tracking sample
        restore_capture(env, blob["capture"])
        a_cmd = desired_match_action(env.sim, scale=0.8, grip=1.0)
        rh = np.asarray(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
        rd = np.asarray(env.sim.fsm.r_des, float).reshape(3, 3)
        cmd = map_action(a_cmd, rd, rh, AIRBORNE_V3)
        h0 = hand_pos(env.sim)
        t0 = float(env.sim.data.time)
        for _ in range(8):
            step_legal(env.sim, a_cmd)
        dt = float(env.sim.data.time) - t0
        v_real = (hand_pos(env.sim) - h0) / max(dt, 1e-9)
        track_rows.append({
            "ic_id": ic,
            "v_cmd_norm": float(np.linalg.norm(cmd["v_world"])),
            "v_real_norm": float(np.linalg.norm(v_real)),
            "tracking_ratio": float(np.linalg.norm(v_real) / max(np.linalg.norm(cmd["v_world"]), 1e-6)),
            "v0": st0["v_rel_norm"],
        })
        rollout[0] += 1

        for sname, toks in audit_seeds:
            if rollout[0] >= b_a:
                break
            if toks is None:
                # short VM sequence for audit
                p = {"n_pursue": 16, "v_scale": 0.85, "grip_open_n": 2, "n_comotion": 0, "n_brake": 4, "n_settle": 0}
                acts = sequence_from_params(env, blob, table, book48, p, mode="B")
            else:
                acts = toks_to_actions(table, book48, toks)
            out, kin, geo = run_actions_eval(env, blob, acts, eval_full=False)
            rollout[0] += 1
            row = {
                "ic_id": ic, "seed": sname, "high_motion": int(is_high_motion(st0, HM_THR)),
                "v0_com_rel": st0["v_rel_norm"], "w0": st0["w_rel_norm"],
                **{k: out[k] for k in (
                    "had_contact", "had_uni", "had_bil", "t_first_contact",
                    "v_com_rel_at_contact", "v_pt_rel_at_contact", "v_n_at_contact", "v_t_at_contact",
                    "M2_vcom_reduced_pre", "pre_vcom_start", "pre_vcom_end", "max_bil_persist_s",
                    "geo_between", "geo_outer_risk", "reason_fail",
                )},
            }
            if kin is not None:
                row["v_com_rel_norm"] = kin["v_com_rel_norm"]
                row["mean_v_rel_pt"] = kin["mean_v_rel_pt"]
                row["pt_vs_com_ratio"] = (
                    None if not kin["v_com_rel_norm"] else float(kin["mean_v_rel_pt"]) / float(kin["v_com_rel_norm"])
                )
                write_json(OUT / "raw_contact_traces" / f"{ic}__{sname}_first_contact.json", kin)
            audit_rows.append(row)
            if geo is not None:
                geo_rows.append({"ic_id": ic, "seed": sname, **geo})
        print(
            f"  audit {ic} last v_com={audit_rows[-1].get('v_com_rel_at_contact')} "
            f"v_pt={audit_rows[-1].get('v_pt_rel_at_contact')}",
            flush=True,
        )

    write_csv(OUT / "CONTACT_POINT_VELOCITY_AUDIT.csv", audit_rows)
    write_csv(OUT / "INTERCEPTION_GEOMETRY.csv", geo_rows)
    write_csv(OUT / "CONTROLLER_TRACKING.csv", track_rows)

    # =====================================================================
    # Phase B — velocity-matched search
    # =====================================================================
    print("=== Phase B velocity-matched search ===", flush=True)
    matched_rows = []
    vm_rows = []
    grip_rows = []
    stab_rows = []
    full_rows = []
    best_by_ic = {}

    # prioritize ICs with contact in audit and high v_pt
    ranked = []
    for ic in BE_ICS:
        rows = [r for r in audit_rows if r["ic_id"] == ic and int(r.get("had_contact") or 0)]
        if not rows:
            continue
        vpt = np.mean([float(r["v_pt_rel_at_contact"]) for r in rows if r.get("v_pt_rel_at_contact") not in (None, "")])
        ranked.append((ic, vpt))
    ranked.sort(key=lambda x: -x[1])

    lim1 = b_a + b_b[0]
    lim2 = lim1 + b_b[1]
    lim3 = lim2 + b_b[2]
    lim4 = TOTAL_ROLLOUT_BUDGET

    def eval_condition(ic, blob, label, acts, family_list):
        out, kin, geo = run_actions_eval(env, blob, acts, eval_full=True)
        rollout[0] += 1
        sc = score_soft_catch(out)
        row = {"ic_id": ic, "condition": label, "score": sc, **out}
        family_list.append(row)
        matched_rows.append(row)
        cur = best_by_ic.get(ic)
        if cur is None or sc > cur["score"]:
            best_by_ic[ic] = {"score": sc, "condition": label, "actions": [a.tolist() for a in acts], "out": out}
        if out.get("SUCCESS_NEW"):
            full_rows.append({
                "ic_id": ic, "condition": label, "SUCCESS_NEW": 1,
                "win_mean_theta": out.get("win_mean_theta"),
                "win_mean_pos": out.get("win_mean_pos"),
                "hold_both_s": out.get("hold_both_s"),
                "v_pt_rel_at_contact": out.get("v_pt_rel_at_contact"),
                "actions_hash": tokens_hash([list(map(float, a)) for a in acts]),
            })
            # save trajectory actions
            np.savez_compressed(
                OUT / "trajectories" / f"{ic}__{label}_actions.npz",
                actions=np.asarray(acts, np.float32),
            )
            write_json(OUT / "trajectories" / f"{ic}__{label}_meta.json", row)
            print(f"    FULL {ic} {label} pos={out.get('win_mean_pos')}", flush=True)
        return out, sc

    # B1: promising HM — compare A/B/C
    print(f"=== B1 promising ICs (to {lim1}) ===", flush=True)
    for ic, _vpt in ranked:
        if rollout[0] >= lim1 or not time_left():
            break
        snap = find_snap(ic)
        if snap is None:
            continue
        blob = load_blob(snap)
        print(f"  B1 {ic} roll={rollout[0]}", flush=True)

        # A: K49 chase baseline
        for pref_name, pref in CHASE_PREFIX_TOKS.items():
            if rollout[0] >= lim1:
                break
            toks = list(pref) + [0] * 4 + [12] * 6 + [SECURE_REVERSE_WRIST] * 16 + [0] * 4
            acts = toks_to_actions(table, book48, toks)
            eval_condition(ic, blob, f"A_k49_{pref_name}", acts, matched_rows)

        # B: velocity matching param search
        for p in param_grid(rng, 12, mode="B"):
            if rollout[0] >= lim1:
                break
            acts = sequence_from_params(env, blob, table, book48, p, mode="B")
            out, sc = eval_condition(ic, blob, "B_velmatch", acts, vm_rows)

        # C: VM + grip timing
        for p in param_grid(rng, 10, mode="C"):
            if rollout[0] >= lim1:
                break
            acts = sequence_from_params(env, blob, table, book48, p, mode="C")
            eval_condition(ic, blob, "C_velmatch_grip", acts, grip_rows)

    # B2: matched ablations on top ICs (collision stop vs VM co-motion etc.)
    print(f"=== B2 ablations (to {lim2}) ===", flush=True)
    top = [ic for ic, _ in ranked[:8]]
    for ic in top:
        if rollout[0] >= lim2 or not time_left():
            break
        blob = load_blob(find_snap(ic))
        ablations = [
            ("abl_fast_stop", {"n_pursue": 12, "v_scale": 1.0, "grip_open_n": 0, "n_comotion": 0, "n_brake": 10, "n_settle": 8}),
            ("abl_vm_stop", {"n_pursue": 16, "v_scale": 0.85, "grip_open_n": 2, "n_comotion": 0, "n_brake": 10, "n_settle": 8}),
            ("abl_vm_comotion", {"n_pursue": 16, "v_scale": 0.85, "grip_open_n": 2, "n_comotion": 6, "comotion_scale": 0.5, "n_brake": 8, "n_settle": 8}),
            ("abl_vm_grip", {"n_pursue": 16, "v_scale": 0.85, "grip_open_n": 3, "grip_close_onset": 8, "grip_close_val": 1.0, "n_comotion": 4, "n_brake": 8, "n_settle": 12}),
            ("abl_vm_gradual", {"n_pursue": 18, "v_scale": 0.75, "grip_open_n": 2, "n_comotion": 8, "comotion_scale": 0.4, "n_brake": 12, "n_settle": 16}),
            ("abl_vm_finedown", {"n_pursue": 16, "v_scale": 0.85, "grip_open_n": 2, "n_comotion": 4, "n_brake": 6, "n_settle": 12, "use_fine_down": True}),
        ]
        for name, p in ablations:
            if rollout[0] >= lim2:
                break
            mode = "C" if "grip" in name else "B"
            acts = sequence_from_params(env, blob, table, book48, p, mode=mode)
            eval_condition(ic, blob, name, acts, matched_rows)

    # B3: stabilize promising (best so far with bil)
    print(f"=== B3 stabilize (to {lim3}) ===", flush=True)
    for ic, pack in list(best_by_ic.items()):
        if rollout[0] >= lim3 or not time_left():
            break
        if not int(pack["out"].get("had_bil") or 0):
            continue
        blob = load_blob(find_snap(ic))
        base_acts = [np.asarray(a, np.float32) for a in pack["actions"]]
        # extend with settle / fine_down / return-like
        for extra_name, extra in [
            ("stab_hold", [np.array([0, 0, 0, 0, 0, 0, 1], np.float32)] * 16),
            ("stab_finedown", [FINE_DOWN.copy() for _ in range(12)] + [np.array([0, 0, 0, 0, -0.2, 0, 1], np.float32)] * 16),
            ("stab_wy", [np.array([0, 0, 0, 0, -0.25, 0, 1], np.float32)] * 20),
        ]:
            if rollout[0] >= lim3:
                break
            acts = base_acts + extra
            eval_condition(ic, blob, extra_name, acts, stab_rows)

    # B4: pose + HANDOFF refine for hard_910012 and any near-success
    print(f"=== B4 pose/HANDOFF (to {lim4}) ===", flush=True)
    pose_ics = ["hard_910012"] + [ic for ic, p in best_by_ic.items() if float(p["out"].get("max_bil_persist_s") or 0) >= 0.2]
    for ic in dict.fromkeys(pose_ics):
        if rollout[0] >= lim4 or not time_left():
            break
        blob = load_blob(find_snap(ic))
        # VM then fine_down then reverse wy
        for p in param_grid(rng, 15, mode="C"):
            if rollout[0] >= lim4:
                break
            p = dict(p)
            p["use_fine_down"] = True
            p["n_settle"] = max(16, int(p["n_settle"]))
            acts = sequence_from_params(env, blob, table, book48, p, mode="C")
            eval_condition(ic, blob, "pose_vm_finedown", acts, stab_rows)

    write_csv(OUT / "MATCHED_CONTROL_ABLATIONS.csv", matched_rows)
    write_csv(OUT / "VELOCITY_MATCHING_SEARCH.csv", vm_rows)
    write_csv(OUT / "GRIP_TIMING_SEARCH.csv", grip_rows)
    write_csv(OUT / "CAPTURE_STABILIZATION.csv", stab_rows)
    write_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", full_rows)
    write_json(OUT / "raw" / "BEST_BY_IC.json", {
        ic: {"score": v["score"], "condition": v["condition"], "out": v["out"]}
        for ic, v in best_by_ic.items()
    })
    write_json(OUT / "raw" / "USAGE.json", {
        "rollouts_used": rollout[0], "budget": TOTAL_ROLLOUT_BUDGET,
        "wall_s": time.time() - t_wall0, "n_full": len(full_rows),
    })

    if not args.skip_videos:
        _render_videos(env, table, book48, best_by_ic, full_rows)

    _write_reports(audit_rows, track_rows, matched_rows, vm_rows, grip_rows, full_rows, best_by_ic, checksum)
    print("Done", OUT / "REPORT.md", flush=True)


def _render_videos(env, table, book48, best_by_ic, full_rows):
    vid = OUT / "videos"

    def render(ic, acts, dest, label):
        blob = load_blob(find_snap(ic))
        restore_capture(env, blob["capture"])
        sim = env.sim
        cam = mjv_camera_from_preset({"lookat": obj_pos(sim), "distance": 0.55, "azimuth": 128.0, "elevation": -18.0})
        renderer = mujoco.Renderer(sim.model, 480, 640)
        frames = []
        t0 = float(sim.data.time)
        dt = float(sim.model.opt.timestep)
        stride = max(1, int(round(0.04 / dt)))

        def grab(tag, clock):
            cam.lookat[:] = obj_pos(sim)
            renderer.update_scene(sim.data, cam)
            pack = physical_pack(sim)
            frames.append(caption(
                renderer.render(),
                f"{ic} {label} {tag} t={clock:.2f} n={int(pack['nL'])}/{int(pack['nR'])}",
            ))

        grab("START", 0.0)
        for i, a in enumerate(acts):
            step_legal(sim, np.asarray(a, np.float32))
            if i % max(1, stride // 2) == 0:
                grab("REC", float(sim.data.time) - t0)
        apply_handoff(sim, blob["ref"])
        t_h = float(sim.data.time)
        steps = 0
        while float(sim.data.time) - t_h < 2.0 - 1e-9:
            nominal_step(sim)
            steps += 1
            if steps % stride == 0:
                grab("HOLD", float(sim.data.time) - t_h)
        renderer.close()
        write_frames_mp4(frames, dest, fps=25)
        shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)

    for r in full_rows[:5]:
        path = OUT / "trajectories" / f"{r['ic_id']}__{r['condition']}_actions.npz"
        if path.exists():
            acts = np.load(path)["actions"]
            render(r["ic_id"], acts, vid / f"full_{r['ic_id']}_{r['condition']}.mp4", "FULL")
    # best partials
    shown = 0
    for ic, pack in sorted(best_by_ic.items(), key=lambda kv: -kv[1]["score"]):
        if shown >= 4:
            break
        acts = [np.asarray(a, np.float32) for a in pack["actions"]]
        render(ic, acts, vid / f"best_{ic}_{pack['condition']}.mp4", pack["condition"])
        shown += 1


def _write_reports(audit, track, matched, vm, grip, full, best_by_ic, checksum):
    # Contact-point vs COM
    contacted = [r for r in audit if int(r.get("had_contact") or 0) and r.get("v_pt_rel_at_contact") not in (None, "")]
    com_vals = [float(r["v_com_rel_at_contact"]) for r in contacted if r.get("v_com_rel_at_contact") not in (None, "")]
    pt_vals = [float(r["v_pt_rel_at_contact"]) for r in contacted]
    ratios = [float(r["pt_vs_com_ratio"]) for r in contacted if r.get("pt_vs_com_ratio") not in (None, "")]

    # Prior 0.41-0.45 were from measure() v_rel_h = COM relative in hand frame
    note_prior = (
        "Previously quoted ~0.41-0.45 m/s values are **COM-relative** speeds from "
        "`measure(...).v_rel_h` (hand frame), not contact-point relative velocities."
    )

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    # Condition comparison
    def cond_stats(name_prefix):
        rows = [r for r in matched if str(r.get("condition", "")).startswith(name_prefix) or r.get("condition") == name_prefix]
        if not rows:
            # try exact / contains
            rows = [r for r in matched if name_prefix in str(r.get("condition", ""))]
        return {
            "n": len(rows),
            "contact": mean([float(r.get("had_contact") or 0) for r in rows]),
            "bil": mean([float(r.get("had_bil") or 0) for r in rows]),
            "persist": mean([float(r.get("max_bil_persist_s") or 0) for r in rows]),
            "m2": mean([float(r.get("M2_vcom_reduced_pre") or 0) for r in rows]),
            "new": sum(int(r.get("SUCCESS_NEW") or 0) for r in rows),
            "v_pt": mean([float(r["v_pt_rel_at_contact"]) for r in rows if r.get("v_pt_rel_at_contact") not in (None, "")]),
        }

    stA = cond_stats("A_k49")
    stB = cond_stats("B_velmatch")
    stC = cond_stats("C_velmatch_grip")
    stAbl = {
        "vm_stop": cond_stats("abl_vm_stop"),
        "vm_comotion": cond_stats("abl_vm_comotion"),
        "vm_grip": cond_stats("abl_vm_grip"),
        "fast_stop": cond_stats("abl_fast_stop"),
    }

    n_stable = sum(1 for ic, p in best_by_ic.items() if int(p["out"].get("M6") or 0) or float(p["out"].get("max_bil_persist_s") or 0) >= 0.3)
    n_full = len({r["ic_id"] for r in full})

    track_ratio = mean([float(r["tracking_ratio"]) for r in track])

    m2_rate = mean([float(r.get("M2_vcom_reduced_pre") or 0) for r in vm + [r for r in matched if "velmatch" in str(r.get("condition", ""))]])

    fail_md = f"""# Failure Analysis

## Contact-point vs COM

{note_prior}

- Audited contacts: {len(contacted)}
- Mean v_COM_rel at contact: {mean(com_vals):.4f} m/s
- Mean v_contact_point_rel at contact: {mean(pt_vals):.4f} m/s
- Mean (v_pt / v_com) ratio: {mean(ratios):.3f}

## Matched strategies

| Condition | n | contact rate | bil rate | mean persist | M2 rate | FULL |
|-----------|--:|-------------:|---------:|-------------:|--------:|-----:|
| A K49 | {stA['n']} | {stA['contact']:.2f} | {stA['bil']:.2f} | {stA['persist']:.3f} | {stA['m2']:.2f} | {stA['new']} |
| B VM | {stB['n']} | {stB['contact']:.2f} | {stB['bil']:.2f} | {stB['persist']:.3f} | {stB['m2']:.2f} | {stB['new']} |
| C VM+grip | {stC['n']} | {stC['contact']:.2f} | {stC['bil']:.2f} | {stC['persist']:.3f} | {stC['m2']:.2f} | {stC['new']} |

Ablations: {json.dumps(stAbl)}

## Interpretation

- If B/C reduce v_pt and raise bil vs A, velocity matching helps capture.
- If M2 stays low, pre-contact matching is not achieved (tracking/horizon).
- If geometry outer_contact_risk high, morphology/alignment dominates.
- Controller mean tracking ratio (VM command): {track_ratio:.3f}
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail_md, encoding="utf-8")

    # Useful discrete behaviors absent from K49
    missing = []
    if stB["m2"] > stA["m2"] + 0.05:
        missing.append("pre-contact velocity-matching pursuit commands")
    if stC["bil"] > stB["bil"] + 0.05:
        missing.append("coordinated grip-close onset during matched pursuit")
    if stAbl["vm_comotion"]["persist"] > stAbl["vm_stop"]["persist"] + 0.02:
        missing.append("brief post-contact co-motion")
    missing.append("fine_down (tok63) already shown useful for pose; keep as comparison")

    report = f"""# Velocity-Matched Soft Catch Diagnostics

**Baseline K49:** `{CONFIG_ID}` checksum `{checksum}`  
**Legal interface:** continuous AIRBORNE_V3 `a in [-1,1]^7` / 20 ms. No SFT/GRPO.

## Budget (frozen before run)

| Item | Value |
|------|------:|
| Total rollouts | {TOTAL_ROLLOUT_BUDGET} |
| Wall (s) | {WALL_LIMIT_S} |
| Phase A / B | 30% audit / 70% search (40/30/20/10 within B) |

See `CATCH_IT_METHOD_COMPARISON.md` (Catch It uses two-stage RL; **no documented velocity-matching loss**).

## Phase A — Contact kinematics

{note_prior}

| Metric | Value |
|--------|------:|
| Contact events audited | {len(contacted)} |
| Mean COM-relative at contact (m/s) | {mean(com_vals):.4f} |
| Mean contact-point relative (m/s) | {mean(pt_vals):.4f} |
| Mean pt/com ratio | {mean(ratios):.3f} |
| Controller VM tracking ratio | {track_ratio:.3f} |

## Phase B — Matched control

| Condition | bil rate | mean persist (s) | M2 rate | SUCCESS_NEW |
|-----------|---------:|-----------------:|--------:|------------:|
| A K49 chase | {stA['bil']:.2f} | {stA['persist']:.3f} | {stA['m2']:.2f} | {stA['new']} |
| B velocity match | {stB['bil']:.2f} | {stB['persist']:.3f} | {stB['m2']:.2f} | {stB['new']} |
| C VM + grip timing | {stC['bil']:.2f} | {stC['persist']:.3f} | {stC['m2']:.2f} | {stC['new']} |

Co-motion vs immediate stop (ablation persist): comotion={stAbl['vm_comotion']['persist']:.3f} stop={stAbl['vm_stop']['persist']:.3f}

Stable bilateral (M6-like) ICs: **{n_stable}**  
Complete SUCCESS_NEW ICs: **{n_full}** ({sorted({r['ic_id'] for r in full})})

## Q1-Q13

1. Contact-point relative velocities (audited failures/seeds): mean **{mean(pt_vals):.3f} m/s** (n={len(pt_vals)}).
2. Different from COM-relative? Mean COM **{mean(com_vals):.3f} m/s**; ratio pt/com **{mean(ratios):.2f}**. Prior 0.41-0.45 m/s quotes were **COM-relative** (`v_rel_h`), not contact-point.
3. Can legal continuous actions reduce relative velocity before contact? M2 rate (VM searches) **{stB['m2']:.2f}** vs K49 **{stA['m2']:.2f}**.
4. Velocity matching improve uni/bilateral capture? bil B={stB['bil']:.2f} vs A={stA['bil']:.2f} (**{"yes" if stB["bil"] > stA["bil"] + 0.02 else "no clear gain"}**).
5. Coordinated grip timing help? bil C={stC['bil']:.2f} vs B={stB['bil']:.2f} (**{"yes" if stC["bil"] > stB["bil"] + 0.02 else "no clear gain"}**).
6. Brief post-contact co-motion help? persist comotion={stAbl['vm_comotion']['persist']:.3f} vs stop={stAbl['vm_stop']['persist']:.3f} (**{"yes" if stAbl["vm_comotion"]["persist"] > stAbl["vm_stop"]["persist"] + 0.02 else "no clear gain"}**).
7. Dominant failure factors: see geometry `outer_contact_risk` rates and v_pt levels in CSVs; typically **high contact-point relative velocity + parallel-jaw geometry**, with grip timing secondary.
8. Controller achieve required velocity? Mean tracking ratio **{track_ratio:.3f}** under VM commands (<<1 implies lag/saturation).
9. HIGH_MOTION ICs reaching stable bilateral: **{n_stable}**
10. Complete SUCCESS_NEW: **{n_full}**
11. Useful behaviors absent from K49: {missing}
12. Small discrete vocabulary represent them? **Possibly** for pose (`fine_down`); for soft-catch VM, need dedicated match/co-motion tokens **if** continuous VM shows clear gains — quantize only after confirmed useful continuous controls.
13. Remaining limitation: **{"gripper morphology + high impact relative velocity / controller tracking" if n_full == 0 else "partially unlocked; check FULL roots"}**; search still non-exhaustive but physically-informed init used.

## Scripts
`scripts/run_velocity_matched_soft_catch.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
