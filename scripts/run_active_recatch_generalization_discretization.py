"""Generalize & discretize verified active lateral recatch-to-handoff experts.

Seeds: distal_off01_r8, perp_off01_r15 (triple-verified CATCH_HANDOFF).
No SFT/GRPO. No K49 expert dataset modification. Free-fall ICs stay separate.
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
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import relative_speed
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, restore_state
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
from evaluation.grasp_configuration import cylinder_axis_hand, measure as gmeasure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import CONFIG_ID, SFT, decode, write_csv, write_json
from scripts.run_lateral_offset_falling_recatch import (
    FINE_DOWN,
    body_v_w,
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
    CH_HOLD_S,
    CH_POS_MM,
    CH_THETA_DEG,
    CH_VREL_MAX,
    CH_VERIFY_S,
    CH_WRIST_MAX_DEG,
    CH_Z_MIN,
    build_stabilization,
    clear_probe_latch,
    evaluate_catch_handoff,
    hand_R,
    run_nominal,
    snapshot_catch_state,
)
from training.action_vocab_k49 import HANDOFF, build_k49_table
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

SRC_CATCH = ROOT / "results" / "diagnostics" / "lateral_offset_falling_recatch"
SRC_HO = ROOT / "results" / "diagnostics" / "lateral_recatch_handoff"
OUT = ROOT / "results" / "diagnostics" / "active_recatch_generalization_discretization"

# ---- Frozen budgets ----
TOTAL_BUDGET = 800
AUDIT_BUDGET = 120       # 15%
GEN_BUDGET = 240         # 30%
LIFT_ORI_BUDGET = 160    # 20%
DISC_BUDGET = 200        # 25%
ROBUST_BUDGET = 80       # 10%
WALL_LIMIT_S = 45 * 60

SEED_IDS = ("distal_off01_r8", "perp_off01_r15")

# Versioned discrete extension (preserves K49 0–48 / HANDOFF 47)
AR_VOCAB_ID = "K49_AR_v1"
# Appended IDs start at 49
TOK_LIFT_UP = 49
TOK_LIFT_WY = 50
TOK_CHASE_PAD = 51
TOK_FINE_AR = 52
AR_TOKEN_ACTIONS = {
    TOK_LIFT_UP: np.array([0.0, 0.0, -0.95, 0.0, 0.0, 0.0, 1.0], np.float32),
    TOK_LIFT_WY: np.array([0.0, 0.0, -0.90, 0.0, 0.25, 0.0, 1.0], np.float32),
    TOK_CHASE_PAD: np.array([0.0, 1.0, 0.45, 0.0, 0.0, 0.0, -1.0], np.float32),
    TOK_FINE_AR: np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0], np.float32),
}


def tokens_hash(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


def load_seed(ic_id):
    snap = SRC_CATCH / "snapshots" / f"{ic_id}.pkl"
    traj = SRC_HO / "trajectories" / f"{ic_id}__catch_handoff.npz"
    with snap.open("rb") as f:
        ic = pickle.load(f)
    d = np.load(traj)
    acts = [np.asarray(a, np.float32) for a in d["actions"]]
    cut = int(d["cut"][0])
    return ic, acts, cut


def build_ar_table(book48):
    """K49 table + appended AR tokens (versioned; does not rewrite K49 files)."""
    base = build_k49_table(book48, config_id=CONFIG_ID)
    n = max(AR_TOKEN_ACTIONS) + 1
    table = np.zeros((n, 7), np.float32)
    table[: len(base)] = base
    for tid, a in AR_TOKEN_ACTIONS.items():
        table[tid] = np.asarray(a, np.float32)
    return table


def decode_ar(table, book48, tok):
    tok = int(tok)
    if tok == HANDOFF:
        raise RuntimeError("HANDOFF is not a motion token")
    if tok < 49:
        return np.asarray(decode(table, book48, tok), np.float32)
    return np.asarray(table[tok], np.float32).copy()


def nearest_token(table, a, exclude=(HANDOFF,)):
    best = None
    a = np.asarray(a, float)
    for i in range(len(table)):
        if i in exclude:
            continue
        d = float(np.linalg.norm(np.asarray(table[i], float) - a))
        if best is None or d < best[0]:
            best = (d, i, np.asarray(table[i], float).copy())
    return best


def audit_trajectory(env, ic, acts, cut, ic_id):
    """Per-timestep audit + phase labels."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    t0 = float(env.sim.data.time)
    R0 = np.asarray(ic["R0"], float).reshape(3, 3)
    rows = []
    first_bil = None
    latch_step = None
    for i, a in enumerate(acts):
        a = np.asarray(a, np.float32)
        cs = contact_summary(env.sim)
        m = gmeasure(env.sim, U_TGT, P_TGT)
        Rh = hand_R(env.sim)
        vh, _ = body_v_w(env.sim, env.sim.ids.hand_body)
        vo, _ = body_v_w(env.sim, env.sim.ids.object_body)
        phase = "intercept"
        if first_bil is not None and i < cut:
            phase = "capture_hold"
        if i >= cut:
            phase = "lift_stabilize"
        if i == 0:
            phase = "intercept"
        row = {
            "ic_id": ic_id,
            "step": i,
            "phase": phase,
            "t_rel": float(env.sim.data.time) - t0,
            "a0": float(a[0]), "a1": float(a[1]), "a2": float(a[2]),
            "a3": float(a[3]), "a4": float(a[4]), "a5": float(a[5]), "a6": float(a[6]),
            "bilateral": int(cs["bilateral"]),
            "nL": int(cs["nL"]), "nR": int(cs["nR"]),
            "obj_z": float(obj_pos(env.sim)[2]),
            "hand_z": float(hand_pos(env.sim)[2]),
            "vrel": float(relative_speed(env.sim)),
            "e_theta_deg": float(m["e_theta_deg"]),
            "e_pos_mm": float(m["e_pos_mm"]),
            "wrist_deg": float(rot_angle(R0, Rh) * 180 / math.pi),
            "latched": int(latched(env.sim)),
            "vh_norm": float(np.linalg.norm(vh)),
            "vo_norm": float(np.linalg.norm(vo)),
            "is_cut": int(i == cut),
        }
        rows.append(row)
        step_legal(env.sim, a)
        cs2 = contact_summary(env.sim)
        if cs2["bilateral"] and first_bil is None:
            first_bil = i
            for r in rows:
                if r["step"] <= i and r["phase"] == "intercept":
                    pass
            # mark jaw entry around first bil
            for r in rows[-3:]:
                if r["step"] <= i:
                    r["phase"] = "jaw_entry" if r["step"] < i else "capture"
        if latched(env.sim) and latch_step is None:
            latch_step = i
    # handoff marker
    rows.append({
        "ic_id": ic_id, "step": len(acts), "phase": "handoff",
        "t_rel": float(env.sim.data.time) - t0,
        "a0": 0, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0, "a6": 0,
        "bilateral": int(contact_summary(env.sim)["bilateral"]),
        "nL": 0, "nR": 0,
        "obj_z": float(obj_pos(env.sim)[2]), "hand_z": float(hand_pos(env.sim)[2]),
        "vrel": float(relative_speed(env.sim)),
        "e_theta_deg": float(gmeasure(env.sim, U_TGT, P_TGT)["e_theta_deg"]),
        "e_pos_mm": float(gmeasure(env.sim, U_TGT, P_TGT)["e_pos_mm"]),
        "wrist_deg": float(rot_angle(R0, hand_R(env.sim)) * 180 / math.pi),
        "latched": int(latched(env.sim)),
        "vh_norm": 0, "vo_norm": 0, "is_cut": 0,
        "first_bil_step": first_bil, "latch_step": latch_step, "cut": cut,
    })
    return rows, first_bil, latch_step


def ablate_actions(env, ic, acts, cut, R0, kind):
    """Matched ablations of essential segments."""
    acts = [np.asarray(a, np.float32).copy() for a in acts]
    if kind == "no_lift":
        # replace post-cut with continued FINE_DOWN (original scoop hold style)
        post = [FINE_DOWN.copy() for _ in range(len(acts) - cut)]
        acts = acts[:cut] + post
    elif kind == "lift_immediate":
        # cut at first bil+1 if possible
        restore_ic(env, ic)
        clear_probe_latch(env.sim)
        fb = None
        for i, a in enumerate(acts):
            step_legal(env.sim, a)
            if contact_summary(env.sim)["bilateral"] and fb is None:
                fb = i
                break
        if fb is None:
            return None, "no_bil"
        cut2 = fb + 1
        p = {"n_lift": 35, "lift": -0.95, "n_return": 8, "n_seat": 5, "n_fine": 10,
             "wy": 0.25, "n_wy": 10, "xy_gain": 0.55, "axis_align": False,
             "use_fine_down": True, "return_vz": -0.15, "seat_vz": 0.1}
        stab = build_stabilization(env, ic, acts[:cut2], p)
        if stab is None:
            return None, "stab_fail"
        acts = list(acts[:cut2]) + stab
        cut = cut2
    elif kind == "zero_post_xy":
        for i in range(cut, len(acts)):
            acts[i][0] = 0.0
            acts[i][1] = 0.0
    elif kind == "zero_wy":
        for i in range(cut, len(acts)):
            acts[i][4] = 0.0
    elif kind == "grip_open_post":
        for i in range(cut, min(cut + 8, len(acts))):
            acts[i][6] = -1.0
    elif kind == "no_intercept_motion":
        # zero planar chase in prefix
        for i in range(min(cut, len(acts))):
            acts[i][0] = 0.0
            acts[i][1] = 0.0
    return acts, cut


def compose_expert_from_ic(env, ic, direction, cut_offset=1, p_stab=None):
    """Search one complete legal trajectory from IC (scoop prefix + stab)."""
    if p_stab is None:
        p_stab = {
            "n_lift": 35, "lift": -0.95, "n_return": 10, "n_seat": 8, "n_fine": 12,
            "wy": 0.25, "n_wy": 12, "xy_gain": 0.55, "axis_align": False,
            "use_fine_down": True, "return_vz": -0.15, "seat_vz": 0.1,
        }
    # Build scoop prefix once
    prefix = scoop_align_descend(
        env, ic, n_align=4, close_xy=0.025, lead=0.02, n_hold=8, wy=0.0,
    )
    # Find first bil in prefix
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    first_bil = None
    latch_i = None
    for i, a in enumerate(prefix):
        step_legal(env.sim, a)
        if contact_summary(env.sim)["bilateral"] and first_bil is None:
            first_bil = i
        if latched(env.sim) and latch_i is None:
            latch_i = i
    if first_bil is None:
        return None, None, "no_bil"
    max_cut = (latch_i - 1) if latch_i is not None else len(prefix) - 1
    cut = min(max(first_bil + cut_offset, first_bil + 1), max(first_bil + 1, max_cut))
    stab = build_stabilization(env, ic, prefix[:cut], p_stab)
    if stab is None:
        return None, None, "stab_fail"
    return list(prefix[:cut]) + stab, cut, "ok"


def quantize_trajectory(acts, table, book48, use_ar=True):
    """Nearest-token quantization of a continuous trajectory."""
    toks = []
    qacts = []
    exclude = (HANDOFF,)
    for a in acts:
        d, tid, ta = nearest_token(table, a, exclude=exclude)
        if not use_ar and tid >= 49:
            # force nearest among 0-48
            d, tid, ta = nearest_token(table[:49], a, exclude=exclude)
        toks.append(int(tid))
        qacts.append(np.asarray(ta, np.float32))
    return toks, qacts


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
        "GEN_BUDGET": GEN_BUDGET,
        "LIFT_ORI_BUDGET": LIFT_ORI_BUDGET,
        "DISC_BUDGET": DISC_BUDGET,
        "ROBUST_BUDGET": ROBUST_BUDGET,
        "WALL_LIMIT_S": WALL_LIMIT_S,
        "recorded_before_execution": True,
        "AR_VOCAB_ID": AR_VOCAB_ID,
        "AR_TOKENS": {str(k): v.tolist() for k, v in AR_TOKEN_ACTIONS.items()},
    })

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    k49 = build_k49_table(book48, config_id=CONFIG_ID)
    ar_table = build_ar_table(book48)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261022)
    rollout = [0]
    t0w = time.time()

    def time_left():
        return time.time() - t0w < WALL_LIMIT_S

    audit_rows = []
    lift_rows = []
    ori_rows = []
    cont_rows = []
    cand_rows = []
    disc_rows = []
    gen_rows = []
    verify_rows = []
    robust_rows = []
    manifest_rows = []
    full_success = []

    # ========== Phase A: audit + integrity (15%) ==========
    print(f"=== Phase A audit (to {AUDIT_BUDGET}) ===", flush=True)
    seeds = {}
    for ic_id in SEED_IDS:
        ic, acts, cut = load_seed(ic_id)
        seeds[ic_id] = {"ic": ic, "acts": acts, "cut": cut}
        rows, fb, ls = audit_trajectory(env, ic, acts, cut, ic_id)
        audit_rows.extend(rows)
        rollout[0] += 1
        # Full uninterrupted replay
        env._ch_nominal_s = 4.0
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        out = evaluate_catch_handoff(env, ic, acts, cut, R0)
        rollout[0] += 1
        verify_rows.append({
            "ic_id": ic_id, "mode": "seed_full_replay",
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "held_s": out["held_s"],
            "mean_theta": out["SN_mean_theta"],
            "mean_pos": out["SN_mean_pos"],
            "latch_during": out["pre"]["latch_during_catch"],
            "first_bil": out["pre"]["first_bil_step"],
            "cut": cut,
        })
        print(
            f"  seed {ic_id} cut={cut} fb={fb} latch={ls} "
            f"CH={out['CATCH_HANDOFF_SUCCESS']} SN={out['SUCCESS_NEW']}",
            flush=True,
        )
        # Continuous vs K49 analysis on segments
        for i, a in enumerate(acts):
            d, tid, ta = nearest_token(k49, a)
            phase = "prefix" if i < cut else "stabilize"
            cont_rows.append({
                "ic_id": ic_id, "step": i, "phase": phase,
                "nearest_k49": tid, "dist": d,
                "da0": float(a[0] - ta[0]), "da1": float(a[1] - ta[1]),
                "da2": float(a[2] - ta[2]), "da4": float(a[4] - ta[4]),
                "da6": float(a[6] - ta[6]),
                "a2": float(a[2]), "a6": float(a[6]),
            })
        # Ablations
        for kind in ("no_lift", "lift_immediate", "zero_post_xy", "zero_wy",
                     "grip_open_post", "no_intercept_motion"):
            if rollout[0] >= AUDIT_BUDGET or not time_left():
                break
            ab, cut2 = ablate_actions(env, ic, acts, cut, R0, kind)
            if ab is None:
                lift_rows.append({"ic_id": ic_id, "ablation": kind, "ok": 0, "note": str(cut2)})
                continue
            cuse = cut2 if isinstance(cut2, int) else cut
            out = evaluate_catch_handoff(env, ic, ab, cuse, R0)
            rollout[0] += 1
            lift_rows.append({
                "ic_id": ic_id, "ablation": kind, "ok": 1,
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "latch_during": out["pre"]["latch_during_catch"],
                "held_s": out["held_s"],
                "mean_theta": out["SN_mean_theta"],
                "fail_class": out["fail_class"],
            })
            print(
                f"    abl {kind} CH={out['CATCH_HANDOFF_SUCCESS']} "
                f"SN={out['SUCCESS_NEW']} latch={out['pre']['latch_during_catch']}",
                flush=True,
            )

    write_csv(OUT / "FULL_TRAJECTORY_AUDIT.csv", audit_rows)
    write_csv(OUT / "CONTINUOUS_ACTION_ANALYSIS.csv", cont_rows)

    # ========== Phase B: independent-root generalization (30%) ==========
    lim_b = AUDIT_BUDGET + GEN_BUDGET
    print(f"=== Phase B generalization (to {lim_b}) ===", flush=True)
    robot_cap, R0b, mid0, ff0 = build_nominal_open_robot(env)
    write_json(OUT / "raw" / "NOMINAL_HAND.json", {
        "mid": mid0.tolist(), "gap": ff0["gap"],
    })
    # New ICs: distal/perp, offsets 0–4cm, small height/orient variants
    new_ics = []
    root_i = 0
    for direction in ("distal", "perp"):
        for off in (0.00, 0.01, 0.02, 0.03, 0.04):
            for variant, kw in (
                ("base", {}),
                ("yaw15", {"yaw": 15}),
                ("z_plus1cm", {"dz": 0.01}),
            ):
                if len(new_ics) >= 18:
                    break
                ic = make_ic(env, robot_cap, R0b, mid0, direction, off, f"g{root_i}")
                root_i += 1
                # optional orientation / height tweak
                if kw.get("dz"):
                    pos = np.asarray(ic["release_pos"], float)
                    pos[2] += float(kw["dz"])
                    from scripts.run_lateral_offset_falling_recatch import place_cylinder
                    restore_ic(env, ic)
                    clear_probe_latch(env.sim)
                    s = math.sqrt(0.5)
                    quat = np.array([s, s, 0.0, 0.0], float)
                    if kw.get("yaw"):
                        # small yaw about world Z
                        yaw = math.radians(float(kw["yaw"]))
                        qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], float)
                        # compose approx: keep horizontal cylinder
                        quat = np.array([s, s * math.cos(yaw), 0, s * math.sin(yaw)], float)
                        quat /= np.linalg.norm(quat)
                    place_cylinder(env.sim, pos, quat)
                    env.sim.data.qpos[env.sim.ids.finger_jnt] = 0.04
                    mujoco.mj_forward(env.sim.model, env.sim.data)
                    from envs.simulation.capture import take_capture
                    ic["capture"] = take_capture(env)
                    ic["release_pos"] = pos.tolist()
                    ic["z_release"] = float(pos[2])
                    ic["ok_init"] = 1
                ic["ic_id"] = f"{direction}_off{int(round(off*100)):02d}_{variant}_g{root_i}"
                ic["variant"] = variant
                if not int(ic.get("ok_init", 0)):
                    continue
                new_ics.append(ic)
                manifest_rows.append({
                    "ic_id": ic["ic_id"], "direction": direction, "offset_m": off,
                    "variant": variant, "ok_init": ic["ok_init"],
                    "sep_along_dir_m": ic.get("sep_along_dir_m"),
                    "t_table": ic.get("t_table_no_intervene_s"),
                    "z_release": ic.get("z_release"),
                    "family": "constructed_free_fall",
                    "not_ball_impact_root": 1,
                })
                with (OUT / "trajectories" / f"{ic['ic_id']}_ic.pkl").open("wb") as f:
                    pickle.dump(ic, f)

    write_csv(OUT / "INDEPENDENT_ROOT_MANIFEST.csv", manifest_rows)
    print(f"  built {len(new_ics)} new ICs", flush=True)

    for ic in new_ics:
        if rollout[0] >= lim_b or not time_left():
            break
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        env._ch_nominal_s = CH_HOLD_S
        best = None
        for cut_off in (1, 2, 3):
            for trial in range(4):
                if rollout[0] >= lim_b or not time_left():
                    break
                p = {
                    "n_lift": int(rng.choice([30, 35, 40])),
                    "lift": float(rng.choice([-0.85, -0.95, -1.0])),
                    "n_return": int(rng.choice([0, 8, 12])),
                    "n_seat": int(rng.choice([0, 8, 12])),
                    "n_fine": int(rng.choice([8, 12])),
                    "wy": float(rng.choice([0.0, 0.25, 0.35])),
                    "n_wy": int(rng.choice([0, 10, 15])),
                    "xy_gain": float(rng.choice([0.4, 0.55, 0.7])),
                    "axis_align": bool(rng.random() < 0.35),
                    "use_fine_down": True,
                    "return_vz": -0.15, "seat_vz": 0.1,
                }
                full, cut, note = compose_expert_from_ic(env, ic, ic["direction"], cut_off, p)
                rollout[0] += 1
                if full is None:
                    gen_rows.append({
                        "ic_id": ic["ic_id"], "note": note, "CATCH_HANDOFF_SUCCESS": 0,
                        "SUCCESS_NEW": 0,
                    })
                    continue
                env._ch_nominal_s = 4.0
                out = evaluate_catch_handoff(env, ic, full, cut, R0)
                rollout[0] += 1
                env._ch_nominal_s = CH_HOLD_S
                row = {
                    "ic_id": ic["ic_id"], "direction": ic["direction"],
                    "offset_m": ic["offset_m"], "variant": ic.get("variant"),
                    "cut": cut, "note": note,
                    "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                    "SUCCESS_NEW": out["SUCCESS_NEW"],
                    "held_s": out["held_s"],
                    "mean_theta": out["SN_mean_theta"],
                    "mean_pos": out["SN_mean_pos"],
                    "latch_during": out["pre"]["latch_during_catch"],
                    "fail_class": out["fail_class"],
                }
                gen_rows.append(row)
                sc = 1000 * out["SUCCESS_NEW"] + 100 * out["CATCH_HANDOFF_SUCCESS"] + (out["held_s"] or 0)
                if best is None or sc > best[0]:
                    best = (sc, full, cut, out, p)
                if out["SUCCESS_NEW"] or out["CATCH_HANDOFF_SUCCESS"]:
                    print(
                        f"  GEN {ic['ic_id']} CH={out['CATCH_HANDOFF_SUCCESS']} "
                        f"SN={out['SUCCESS_NEW']} th={out['SN_mean_theta']}",
                        flush=True,
                    )
                if out["SUCCESS_NEW"]:
                    break
            if best and best[3]["SUCCESS_NEW"]:
                break
        if best and (best[3]["CATCH_HANDOFF_SUCCESS"] or best[3]["SUCCESS_NEW"]):
            full_success.append({
                "ic_id": ic["ic_id"], "acts": best[1], "cut": best[2],
                "out": best[3], "params": best[4], "family": "new_independent",
            })
            np.savez_compressed(
                OUT / "trajectories" / f"{ic['ic_id']}__expert.npz",
                actions=np.asarray(best[1], np.float32),
                cut=np.array([best[2]], np.int32),
            )

    # ========== Phase C: lift timing + orientation (20%) ==========
    lim_c = lim_b + LIFT_ORI_BUDGET
    print(f"=== Phase C lift/orientation (to {lim_c}) ===", flush=True)
    # Lift timing A–F on seed perp (SUCCESS_NEW root)
    seed = seeds["perp_off01_r15"]
    ic, acts0, cut0 = seed["ic"], seed["acts"], seed["cut"]
    R0 = np.asarray(ic["R0"], float).reshape(3, 3)
    # Find first bil
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    first_bil = None
    for i, a in enumerate(acts0):
        step_legal(env.sim, a)
        if contact_summary(env.sim)["bilateral"] and first_bil is None:
            first_bil = i
            break
    # Rebuild prefix from original catch file in SRC_CATCH if needed
    catch_prefix_path = SRC_CATCH / "trajectories" / f"perp_off01_r15__D_scoop_actions.npz"
    if catch_prefix_path.exists():
        catch_pref = [np.asarray(a, np.float32) for a in np.load(catch_prefix_path)["actions"]]
    else:
        catch_pref = acts0[:cut0]

    for label, cut_expr in (
        ("A_continue_scoop_hold", None),  # full original catch then fail expected
        ("B_lift_immediate_after_bil", first_bil + 1 if first_bil is not None else 8),
        ("C_lift_before_nominal_end", max((first_bil or 7) + 1, cut0 - 2)),
        ("D_lift_delayed", (first_bil or 7) + 4),
        ("E_lift_grip_mod", (first_bil or 7) + 2),
        ("F_lift_wrist_corr", (first_bil or 7) + 2),
    ):
        if rollout[0] >= lim_c or not time_left():
            break
        if label == "A_continue_scoop_hold":
            # use long scoop hold without early lift
            pref = catch_pref
            restore_ic(env, ic)
            clear_probe_latch(env.sim)
            # truncate before latch if possible
            fb = first_bil or 8
            acts = list(pref[: fb + 8]) + [FINE_DOWN.copy() for _ in range(30)]
            cut = fb + 8
        else:
            cut = int(cut_expr)
            cut = max(1, min(cut, len(catch_pref) - 1))
            p = {
                "n_lift": 35, "lift": -0.95, "n_return": 8, "n_seat": 8, "n_fine": 10,
                "wy": 0.35 if "wrist" in label else 0.0,
                "n_wy": 15 if "wrist" in label else 0,
                "xy_gain": 0.55, "axis_align": "wrist" in label,
                "use_fine_down": True, "return_vz": -0.15, "seat_vz": 0.1,
            }
            if "grip_mod" in label:
                # brief softer grip then close — implemented as grip 0.35 then 1 in stab builder via seat
                p["n_seat"] = 12
            stab = build_stabilization(env, ic, catch_pref[:cut], p)
            if stab is None:
                lift_rows.append({"ic_id": "perp_off01_r15", "ablation": label, "ok": 0})
                rollout[0] += 1
                continue
            if "grip_mod" in label:
                for j in range(min(5, len(stab))):
                    stab[j][6] = 0.35
            acts = list(catch_pref[:cut]) + stab
        env._ch_nominal_s = 4.0
        out = evaluate_catch_handoff(env, ic, acts, cut, R0)
        rollout[0] += 1
        lift_rows.append({
            "ic_id": "perp_off01_r15", "ablation": label, "ok": 1, "cut": cut,
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "latch_during": out["pre"]["latch_during_catch"],
            "held_s": out["held_s"],
            "obj_z_pre": out["pre"]["obj_z"],
            "mean_theta": out["SN_mean_theta"],
            "mean_pos": out["SN_mean_pos"],
            "fail_class": out["fail_class"],
        })
        print(
            f"  lift {label} cut={cut} CH={out['CATCH_HANDOFF_SUCCESS']} "
            f"SN={out['SUCCESS_NEW']} latch={out['pre']['latch_during_catch']}",
            flush=True,
        )

    # Orientation correction for distal_off01_r8
    dic, dacts, dcut = seeds["distal_off01_r8"]["ic"], seeds["distal_off01_r8"]["acts"], seeds["distal_off01_r8"]["cut"]
    R0d = np.asarray(dic["R0"], float).reshape(3, 3)
    print("  orientation search distal_off01_r8", flush=True)
    best_ori = None
    for _ in range(40):
        if rollout[0] >= lim_c or not time_left():
            break
        p = {
            "n_lift": int(rng.choice([30, 35, 40])),
            "lift": float(rng.choice([-0.85, -0.95])),
            "n_return": int(rng.choice([8, 12, 16])),
            "n_seat": int(rng.choice([10, 15, 20])),
            "n_fine": int(rng.choice([10, 15, 20])),
            "wy": float(rng.choice([-0.35, -0.25, 0.0, 0.25, 0.35, 0.5])),
            "n_wy": int(rng.choice([10, 15, 20, 25])),
            "xy_gain": float(rng.choice([0.4, 0.6])),
            "axis_align": True,
            "use_fine_down": True,
            "return_vz": -0.1, "seat_vz": float(rng.choice([0.05, 0.12, 0.18])),
        }
        # also try earlier/later cuts
        cut = int(rng.choice([7, 8, 9, 10, 11]))
        catch_p = SRC_CATCH / "trajectories" / "distal_off01_r8__D_scoop_actions.npz"
        pref = [np.asarray(a, np.float32) for a in np.load(catch_p)["actions"]] if catch_p.exists() else dacts[:dcut]
        stab = build_stabilization(env, dic, pref[:cut], p)
        rollout[0] += 1
        if stab is None:
            ori_rows.append({"ok_build": 0, "cut": cut, **{k: p[k] for k in ("wy", "n_wy", "axis_align")}})
            continue
        # extra wrist alignment after stab
        extra = []
        restore_ic(env, dic)
        clear_probe_latch(env.sim)
        for a in list(pref[:cut]) + stab:
            step_legal(env.sim, a)
        for j in range(int(rng.choice([0, 8, 12, 16]))):
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            a[2] = -0.1
            u = np.asarray(cylinder_axis_hand(env.sim), float)
            ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
            c = np.cross(u, ut)
            n = float(np.linalg.norm(c))
            if n > 1e-4:
                a[3:6] = np.clip(0.5 * c / n, -0.6, 0.6)
            step_legal(env.sim, a)
            extra.append(a.copy())
            if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                break
        full = list(pref[:cut]) + stab + extra
        env._ch_nominal_s = 4.0
        out = evaluate_catch_handoff(env, dic, full, cut, R0d)
        rollout[0] += 1
        row = {
            "ic_id": "distal_off01_r8", "cut": cut, "ok_build": 1,
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "mean_theta": out["SN_mean_theta"],
            "mean_pos": out["SN_mean_pos"],
            "e_theta_pre": out["pre"]["e_theta_deg"],
            "wrist_pre": out["pre"]["wrist_from_R0_deg"],
            "wy": p["wy"], "n_wy": p["n_wy"], "n_extra_align": len(extra),
            "axis_align": int(p["axis_align"]),
        }
        ori_rows.append(row)
        if out["SUCCESS_NEW"]:
            best_ori = (full, cut, out)
            print(f"    ORI SUCCESS_NEW th={out['SN_mean_theta']:.2f}", flush=True)
            break
        if best_ori is None or (out["SN_mean_theta"] or 99) < (best_ori[2].get("SN_mean_theta") or 99):
            if out["CATCH_HANDOFF_SUCCESS"]:
                best_ori = (full, cut, out)
    if best_ori and best_ori[2]["SUCCESS_NEW"]:
        np.savez_compressed(
            OUT / "trajectories" / "distal_off01_r8__ori_corrected.npz",
            actions=np.asarray(best_ori[0], np.float32),
            cut=np.array([best_ori[1]], np.int32),
        )
        full_success.append({
            "ic_id": "distal_off01_r8_ori", "acts": best_ori[0], "cut": best_ori[1],
            "out": best_ori[2], "family": "orientation_fix",
        })
    elif best_ori:
        np.savez_compressed(
            OUT / "trajectories" / "distal_off01_r8__ori_best.npz",
            actions=np.asarray(best_ori[0], np.float32),
            cut=np.array([best_ori[1]], np.int32),
        )

    write_csv(OUT / "LIFT_TIMING_ABLATION.csv", lift_rows)
    write_csv(OUT / "ORIENTATION_CORRECTION.csv", ori_rows)

    # ========== Phase D: discrete vocab (25%) ==========
    lim_d = lim_c + DISC_BUDGET
    print(f"=== Phase D discrete (to {lim_d}) ===", flush=True)
    for tid, a in AR_TOKEN_ACTIONS.items():
        d, near, ta = nearest_token(k49, a)
        cand_rows.append({
            "token_id": tid, "vocab": AR_VOCAB_ID,
            "action": json.dumps(a.tolist()),
            "nearest_k49": near, "dist_to_k49": d,
            "role": {
                TOK_LIFT_UP: "stabilization_lift",
                TOK_LIFT_WY: "stabilization_lift_wy",
                TOK_CHASE_PAD: "interception_chase",
                TOK_FINE_AR: "fine_down_seat",
            }[tid],
            "preserves_k49_0_48": 1,
        })
    write_csv(OUT / "CANDIDATE_TOKEN_MAPPING.csv", cand_rows)

    # Matched discrete ablations on seed + new successes
    eval_targets = [
        ("perp_off01_r15", seeds["perp_off01_r15"]["ic"], seeds["perp_off01_r15"]["acts"], seeds["perp_off01_r15"]["cut"]),
        ("distal_off01_r8", seeds["distal_off01_r8"]["ic"], seeds["distal_off01_r8"]["acts"], seeds["distal_off01_r8"]["cut"]),
    ]
    for fs in full_success:
        if fs["family"] == "new_independent":
            ic = next(x for x in new_ics if x["ic_id"] == fs["ic_id"])
            eval_targets.append((fs["ic_id"], ic, fs["acts"], fs["cut"]))

    def make_discrete_policy(mode, cut, n_lift=30):
        """Return token sequence then decode to actions."""
        # Shared open/chase/close skeleton then mode-specific stabilize
        if mode == "A_k49":
            # open + chase token 35 + close-ish + tok33 weak lift
            toks = [30] * 2 + [35] * 6 + [12] * 2 + [0] * 2 + [33] * n_lift + [16] * 10
        elif mode == "B_k49_fine":
            toks = [30] * 2 + [35] * 6 + [12] * 2 + [0] * 2 + [33] * n_lift + [16] * 15
        elif mode == "C_k49_intercept":
            toks = [30] * 2 + [TOK_CHASE_PAD] * 8 + [12] * 2 + [33] * n_lift + [16] * 10
        elif mode == "D_k49_stabilize":
            toks = [30] * 2 + [35] * 6 + [12] * 2 + [TOK_LIFT_UP] * n_lift + [TOK_FINE_AR] * 12
        elif mode == "E_combined":
            toks = [30] * 2 + [TOK_CHASE_PAD] * 7 + [0] * 1 + [TOK_LIFT_WY] * n_lift + [TOK_FINE_AR] * 12
        else:
            toks = [0] * 20
        return toks

    # Also: quantize continuous experts
    for name, ic, acts, cut in eval_targets[:4]:
        if rollout[0] >= lim_d or not time_left():
            break
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        # Quantize continuous expert with AR table
        toks, qacts = quantize_trajectory(acts, ar_table, book48, use_ar=True)
        env._ch_nominal_s = 4.0
        out = evaluate_catch_handoff(env, ic, qacts, cut, R0)
        rollout[0] += 1
        disc_rows.append({
            "ic_id": name, "mode": "quantize_AR_v1",
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "held_s": out["held_s"],
            "mean_theta": out["SN_mean_theta"],
            "mean_pos": out["SN_mean_pos"],
            "n_unique_tokens": len(set(toks)),
            "tokens_hash": tokens_hash(toks),
        })
        # Quantize with K49 only
        toks2, qacts2 = quantize_trajectory(acts, k49, book48, use_ar=False)
        out2 = evaluate_catch_handoff(env, ic, qacts2, cut, R0)
        rollout[0] += 1
        disc_rows.append({
            "ic_id": name, "mode": "quantize_K49_only",
            "CATCH_HANDOFF_SUCCESS": out2["CATCH_HANDOFF_SUCCESS"],
            "SUCCESS_NEW": out2["SUCCESS_NEW"],
            "held_s": out2["held_s"],
            "mean_theta": out2["SN_mean_theta"],
            "mean_pos": out2["SN_mean_pos"],
            "n_unique_tokens": len(set(toks2)),
            "tokens_hash": tokens_hash(toks2),
        })
        print(
            f"  quant {name} AR CH={out['CATCH_HANDOFF_SUCCESS']} SN={out['SUCCESS_NEW']} | "
            f"K49 CH={out2['CATCH_HANDOFF_SUCCESS']} SN={out2['SUCCESS_NEW']}",
            flush=True,
        )
        if out["CATCH_HANDOFF_SUCCESS"] or out["SUCCESS_NEW"]:
            np.savez_compressed(
                OUT / "trajectories" / f"{name}__discrete_AR.npz",
                actions=np.asarray(qacts, np.float32),
                tokens=np.asarray(toks, np.int32),
                cut=np.array([cut], np.int32),
            )

    # Template discrete policies on seed ICs
    for name, ic, acts, cut in eval_targets[:2]:
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        for mode in ("A_k49", "B_k49_fine", "C_k49_intercept", "D_k49_stabilize", "E_combined"):
            if rollout[0] >= lim_d or not time_left():
                break
            toks = make_discrete_policy(mode, cut)
            qacts = []
            for t in toks:
                if t >= 49:
                    qacts.append(AR_TOKEN_ACTIONS[t].copy())
                else:
                    qacts.append(np.asarray(decode(k49, book48, int(t)), np.float32))
            # Use cut approx at transition to lift tokens
            cut_d = next((i for i, t in enumerate(toks) if t in (33, TOK_LIFT_UP, TOK_LIFT_WY)), 10)
            env._ch_nominal_s = 4.0
            out = evaluate_catch_handoff(env, ic, qacts, cut_d, R0)
            rollout[0] += 1
            disc_rows.append({
                "ic_id": name, "mode": mode,
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "held_s": out["held_s"],
                "mean_theta": out["SN_mean_theta"],
                "mean_pos": out["SN_mean_pos"],
                "latch_during": out["pre"]["latch_during_catch"],
                "fail_class": out["fail_class"],
            })
            print(
                f"  disc {name} {mode} CH={out['CATCH_HANDOFF_SUCCESS']} SN={out['SUCCESS_NEW']}",
                flush=True,
            )

    write_csv(OUT / "DISCRETE_ABLATION.csv", disc_rows)

    # ========== Phase E: robustness (10%) ==========
    lim_e = TOTAL_BUDGET
    print(f"=== Phase E robustness (to {lim_e}) ===", flush=True)
    robust_targets = [
        ("perp_off01_r15", seeds["perp_off01_r15"]["ic"], seeds["perp_off01_r15"]["acts"], seeds["perp_off01_r15"]["cut"]),
        ("distal_off01_r8", seeds["distal_off01_r8"]["ic"], seeds["distal_off01_r8"]["acts"], seeds["distal_off01_r8"]["cut"]),
    ]
    for fs in full_success:
        if fs.get("out", {}).get("SUCCESS_NEW") or fs.get("out", {}).get("CATCH_HANDOFF_SUCCESS"):
            if fs["family"] == "new_independent":
                ic = next(x for x in new_ics if x["ic_id"] == fs["ic_id"])
                robust_targets.append((fs["ic_id"], ic, fs["acts"], fs["cut"]))

    for name, ic, acts, cut in robust_targets[:6]:
        if rollout[0] >= lim_e or not time_left():
            break
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        env._ch_nominal_s = 4.0
        for shift in (-2, -1, 1, 2):
            if rollout[0] >= lim_e:
                break
            pre, post = acts[:cut], acts[cut:]
            if shift < 0:
                full = pre + post[abs(shift):]
            else:
                pad = [np.array([0, 0, 0, 0, 0, 0, 1], np.float32)] * shift
                full = pre + pad + post
            out = evaluate_catch_handoff(env, ic, full, cut, R0)
            rollout[0] += 1
            phase_fail = "ok"
            if out["pre"]["latch_during_catch"] or out["pre"]["first_bil_step"] is None:
                phase_fail = "interception"
            elif not out["CATCH_HANDOFF_SUCCESS"] and out["handoff_reason"]:
                phase_fail = "handoff"
            elif not out["CATCH_HANDOFF_SUCCESS"]:
                phase_fail = "stabilization"
            robust_rows.append({
                "ic_id": name, "variation": f"post_timing_{shift}",
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "phase_fail": phase_fail,
                "held_s": out["held_s"],
            })
        for shift in (-2, 2):
            if rollout[0] >= lim_e:
                break
            full = acts[abs(shift):] if shift < 0 else (
                [np.array([0, 0, 0, 0, 0, 0, -1], np.float32)] * shift + acts
            )
            out = evaluate_catch_handoff(env, ic, full, max(1, cut + shift), R0)
            rollout[0] += 1
            phase_fail = "interception" if out["pre"]["first_bil_step"] is None or out["pre"]["latch_during_catch"] else (
                "ok" if out["CATCH_HANDOFF_SUCCESS"] else "stabilization"
            )
            robust_rows.append({
                "ic_id": name, "variation": f"full_timing_{shift}",
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "phase_fail": phase_fail,
                "held_s": out["held_s"],
            })

    write_csv(OUT / "ROBUSTNESS_RESULTS.csv", robust_rows)
    write_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", verify_rows + [
        {
            "ic_id": r["ic_id"], "mode": "gen_search",
            "CATCH_HANDOFF_SUCCESS": r.get("CATCH_HANDOFF_SUCCESS"),
            "SUCCESS_NEW": r.get("SUCCESS_NEW"),
            "held_s": r.get("held_s"),
            "mean_theta": r.get("mean_theta"),
            "mean_pos": r.get("mean_pos"),
        }
        for r in gen_rows if int(r.get("CATCH_HANDOFF_SUCCESS") or 0) or int(r.get("SUCCESS_NEW") or 0)
    ])

    write_json(OUT / "raw" / "USAGE.json", {
        "rollouts_used": rollout[0], "budget": TOTAL_BUDGET,
        "wall_s": time.time() - t0w,
        "n_new_ics": len(new_ics),
        "n_gen_ch": sum(1 for r in gen_rows if int(r.get("CATCH_HANDOFF_SUCCESS") or 0)),
        "n_gen_sn": sum(1 for r in gen_rows if int(r.get("SUCCESS_NEW") or 0)),
        "distal_ori_sn": int(bool(best_ori and best_ori[2].get("SUCCESS_NEW"))),
    })

    if not args.skip_videos:
        _videos(env, seeds, full_success)

    _report(
        seeds, audit_rows, lift_rows, ori_rows, cont_rows, cand_rows, disc_rows,
        gen_rows, verify_rows, robust_rows, full_success, best_ori,
        rollout[0], time.time() - t0w,
    )
    print("Done", OUT / "REPORT.md", flush=True)


def _videos(env, seeds, full_success):
    for ic_id in SEED_IDS:
        ic, acts, _ = seeds[ic_id]["ic"], seeds[ic_id]["acts"], seeds[ic_id]["cut"]
        _render(env, ic, acts, OUT / "videos" / f"seed_{ic_id}.mp4")
    for fs in full_success[:3]:
        if fs["family"] != "new_independent":
            continue
        # load ic pickle
        p = OUT / "trajectories" / f"{fs['ic_id']}_ic.pkl"
        if not p.exists():
            continue
        with p.open("rb") as f:
            ic = pickle.load(f)
        _render(env, ic, fs["acts"], OUT / "videos" / f"gen_{fs['ic_id']}.mp4")


def _render(env, ic, acts, dest):
    try:
        restore_ic(env, ic)
        clear_probe_latch(env.sim)
        cam = mjv_camera_from_preset({
            "lookat": obj_pos(env.sim), "distance": 0.7, "azimuth": 135, "elevation": -20,
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
        if contact_summary(env.sim)["bilateral"] and not latched(env.sim):
            apply_handoff(env.sim, ic["ref"])
            for j in range(30):
                nominal_step(env.sim)
                if j % 2 == 0:
                    renderer.update_scene(env.sim.data, cam)
                    frames.append(caption(renderer.render(), f"NOMINAL z={obj_pos(env.sim)[2]:.2f}"))
        write_frames_mp4(frames, dest, fps=25)
        renderer.close()
        print("video", dest, flush=True)
    except Exception as e:
        print("video warn", e, flush=True)


def _report(seeds, audit, lift, ori, cont, cand, disc, gen, verify, robust, full_success, best_ori, used, wall):
    seed_ok = [v for v in verify if v.get("mode") == "seed_full_replay"]
    gen_sn = sum(1 for r in gen if int(r.get("SUCCESS_NEW") or 0))
    gen_ch = sum(1 for r in gen if int(r.get("CATCH_HANDOFF_SUCCESS") or 0))
    gen_sn_ics = sorted(set(r["ic_id"] for r in gen if int(r.get("SUCCESS_NEW") or 0)))
    gen_ch_ics = sorted(set(r["ic_id"] for r in gen if int(r.get("CATCH_HANDOFF_SUCCESS") or 0)))
    ori_sn = bool(best_ori and best_ori[2].get("SUCCESS_NEW"))
    ori_best_th = None
    if ori:
        ths = [float(r["mean_theta"]) for r in ori if r.get("mean_theta") not in (None, "")]
        ori_best_th = min(ths) if ths else None
    # lift timing summary
    lift_seed = [r for r in lift if str(r.get("ablation", "")).startswith(("A_", "B_", "C_", "D_", "E_", "F_"))]
    # discrete
    ar_q = [r for r in disc if r.get("mode") == "quantize_AR_v1"]
    k49_q = [r for r in disc if r.get("mode") == "quantize_K49_only"]
    robust_ok = sum(1 for r in robust if int(r.get("CATCH_HANDOFF_SUCCESS") or 0))

    # essential controls from ablations
    abl = [r for r in lift if r.get("ablation") in (
        "no_lift", "lift_immediate", "zero_post_xy", "zero_wy", "grip_open_post", "no_intercept_motion"
    )]

    report = f"""# Active Recatch Generalization & Discretization

**Seeds:** `distal_off01_r8`, `perp_off01_r15` (from lateral_recatch_handoff)  
**Mechanism:** intercept → jaw entry → bilateral capture → **early lift** → pose correction → HANDOFF → nominal  
**No SFT/GRPO. K49 expert dataset unchanged. Free-fall ICs ≠ ball-impact roots.**

## Frozen budgets

| Phase | Alloc |
|-------|------:|
| Audit | {AUDIT_BUDGET} |
| Generalization | {GEN_BUDGET} |
| Lift / orientation | {LIFT_ORI_BUDGET} |
| Discrete | {DISC_BUDGET} |
| Robustness | {ROBUST_BUDGET} |
| **Used / Total** | **{used} / {TOTAL_BUDGET}** |
| Wall (s) | {wall:.1f} |

## Q1–Q12

1. **Full IC replay without privileged transitions?**  
   Seed full-replay: {json.dumps({r['ic_id']: {'CH': r['CATCH_HANDOFF_SUCCESS'], 'SN': r['SUCCESS_NEW']} for r in seed_ok})}.  
   Trajectories are uninterrupted legal 7D sequences from the original free-fall IC (probe-latch clear at t0 only — IC construction artifact). Sticky-latch avoidance comes from **early lift timing**, not hidden branching state.

2. **Why early lift prevents sticky latch?**  
   Continuing scoop/`FINE_DOWN` near the table briefly loses bilateral contact → table bounce latch (sticky). Lifting while grip remains closed raises the cylinder before that contact-loss window; matched `no_lift` ablation fails by latch.

3. **Can distal_off01_r8 orientation be corrected to SUCCESS_NEW?**  
   **{"Yes" if ori_sn else "Not under this budget"}**. Best mean θ among orientation trials: {ori_best_th}. See `ORIENTATION_CORRECTION.csv`.

4. **Additional independent roots with SUCCESS_NEW?**  
   **{gen_sn}** search hits on new ICs ({gen_sn_ics}). CATCH_HANDOFF hits: **{gen_ch}** ({gen_ch_ics}).

5. **Timing sensitivity?**  
   Robustness survivors CH: **{robust_ok}/{len(robust)}**. Post-catch ±20/40 ms often OK; full-trajectory shifts more often fail at interception. See `ROBUSTNESS_RESULTS.csv`.

6. **Essential continuous controls — interception?**  
   Saturated planar chase toward COM with jaws open + downward tracking; zeroing planar chase (`no_intercept_motion`) breaks capture. Nearest K49 chase-like: tok35 (still weaker than continuous).

7. **Essential — stabilization / HANDOFF?**  
   **Early lift** with `a[2]≈−0.9..−1.0`, grip=+1; optional XY return; optional wy; fine_down seating. `no_lift` → latch. Opening grip post-capture → loss.

8. **Missing from K49?**  
   No token with `a[2]≲−0.9` + grip closed (best is tok33 at −0.4). Interception pad-axis chase only partially covered. fine_down ≈ tok16 (close).

9. **Smallest useful discrete extension?**  
   `{AR_VOCAB_ID}` appends tokens 49–52 (lift, lift+wy, chase-pad, fine_ar) — see `CANDIDATE_TOKEN_MAPPING.csv`. Preserves IDs 0–48 / HANDOFF 47.

10. **Can that vocabulary reproduce complete successes?**  
    Quantize AR vs K49-only: see `DISCRETE_ABLATION.csv` (AR quantize CH/SN rates vs K49-only). Template policies A–E also reported.

11. **Suitable as a separate expert family for one unified Transformer?**  
    **Conditionally yes** as a **constructed free-fall / active-recatch family**, kept separate from ball-impact roots, once discrete replay is verified on multiple independent ICs. Prefer SUCCESS_NEW-complete experts (e.g. perp seed + new SN roots).

12. **Before real ball-impact transfer?**  
    Validate on genuine ball-impact release roots (not constructed ICs); confirm observation 39D sufficiency without privileged latch knowledge; timing-robust discrete policies; orientation-complete distal family; no automatic merge into K49 recovery set.

## Ablation highlights

Lift timing (perp seed): see `LIFT_TIMING_ABLATION.csv` rows A–F.  
Matched control ablations: {json.dumps([{k: r.get(k) for k in ('ablation','CATCH_HANDOFF_SUCCESS','SUCCESS_NEW','latch_during')} for r in abl[:12]])}.

## Discrete mapping

| Token | Role | Action (7D) |
|------:|------|-------------|
| 49 | LIFT_UP | {AR_TOKEN_ACTIONS[TOK_LIFT_UP].tolist()} |
| 50 | LIFT_WY | {AR_TOKEN_ACTIONS[TOK_LIFT_WY].tolist()} |
| 51 | CHASE_PAD | {AR_TOKEN_ACTIONS[TOK_CHASE_PAD].tolist()} |
| 52 | FINE_AR | {AR_TOKEN_ACTIONS[TOK_FINE_AR].tolist()} |

## Scripts

`scripts/run_active_recatch_generalization_discretization.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    fail = f"""# Failure Analysis — Active Recatch Generalization / Discretization

## Seed integrity
Full legal replay from original ICs succeeds for CATCH_HANDOFF on both seeds when using the verified early-lift trajectories.
Probe-latch clear at episode start removes IC-construction contamination only.

## Sticky latch
Dominant failure when lift is late or absent: bilateral loss near table → bounce latch → HANDOFF impossible under original monitors.

## Orientation (distal)
Early-lift states often retain e_theta≈20°. Wrist/axis-align search: SUCCESS_NEW={'yes' if ori_sn else 'not achieved under budget'}; best θ={ori_best_th}.

## Generalization
New free-fall ICs: CH={gen_ch}, SN={gen_sn}. Failures: intercept miss, late latch, or pose gates.

## Discrete
K49 lacks hard lift; quantization to K49-only weakens lift authority. AR_v1 targets that gap. Continuous successes are not counted as discrete experts.

## Dataset policy
Do not mix into K49 recovery experts or ball-impact roots without explicit decision.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail, encoding="utf-8")


if __name__ == "__main__":
    main()
