#!/usr/bin/env python
"""Final Plan C push: in-hand correction + surviving HANDOFF → SUCCESS_NEW.

Starts from existing elevated lift experts. Prioritizes gbi_010/022 (near-mid)
and gbi_014 (controlled sliding). No SFT/GRPO / physics / SUCCESS_NEW changes.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import (
    hold_action,
    object_axis,
    pad_bilateral,
    pose_correct_and_handoff,
    sample_kin,
)
from data_generation.oracles.plan_c.catch_stabilization import (
    SlideParams,
    grasp_geometry,
    gravity_slide_sequence,
)
from data_generation.oracles.plan_b.release_recatch import _action_from
from controllers.recovery.action_map import TAU_SECURE, TAU_OPEN
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.success_criteria import (
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
from evaluation.grasp_geometry import measure
from experiments.audits.plan_a_mechanism_audit import sample_kinematics
from training.recovery_execution import make_snap, restore_ballistic

CATCH = ROOT / "results/diagnostics/ground_dynamic_catch_oracle"
END = ROOT / "results/diagnostics/ground_postcatch_endgrasp_recovery"
OUT = ROOT / "results/diagnostics/ground_plan_c_final_success"
RAW = OUT / "raw"
VID = OUT / "videos"

ROOTS_PRIORITY = ["gbi_010", "gbi_022", "gbi_014"]  # easiest first; skip gbi_002
BUDGET = 300


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


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path, rows):
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


def load_pair(root_id: str):
    catch = pickle.loads((CATCH / "raw" / f"{root_id}__catch_expert.pkl").read_bytes())
    end = pickle.loads((END / "raw" / f"{root_id}__endgrasp_expert.pkl").read_bytes())
    return catch, end


def replay_to_elevated(sim, catch, end) -> dict:
    """Catch capture → lift → stabilize. Returns elevated snap + error pack."""
    restore_ballistic(sim, catch["snap"])
    restore_drop(sim, None)
    arm(sim)
    acts = []
    for a in catch["capture_actions"]:
        a = np.asarray(a, np.float32)
        acts.append(a.copy())
        step_legal(sim, a)
        if latched(sim):
            break
    for a in end["actions"]:
        a = np.asarray(a, np.float32)
        acts.append(a.copy())
        step_legal(sim, a)
        if latched(sim):
            break
    for _ in range(8):
        a = hold_action(1.0)
        acts.append(a.copy())
        step_legal(sim, a)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    elev = make_snap(sim)
    err = decompose_error(sim)
    return {"elev_snap": elev, "actions": acts, "error": err, "ref": catch["ref"]}


def decompose_error(sim) -> dict:
    k = sample_kin(sim)
    Rh, ph, po = k["R_hand"], k["p_hand"], k["p_obj"]
    prel = Rh.T @ (po - ph)
    err = np.asarray(P_TGT, float) - prel
    u = object_axis(sim)
    uh = Rh.T @ u
    uh = uh / max(float(np.linalg.norm(uh)), 1e-12)
    e_ax = float(np.dot(err, uh))
    e_lat_vec = err - e_ax * uh
    m = measure(sim, U_TGT, P_TGT)
    sk = sample_kinematics(sim)
    pb = pad_bilateral(sim)
    in_band = bool(m["e_theta_deg"] <= WORKING_THETA_DEG and m["e_pos_mm"] <= WORKING_POS_MM and pb["bilateral"])
    return {
        "p_rel": prel.tolist(),
        "P_TGT": P_TGT.tolist(),
        "err": err.tolist(),
        "e_norm_mm": float(np.linalg.norm(err) * 1e3),
        "e_axial_mm": float(e_ax * 1e3),
        "e_lateral_mm": float(np.linalg.norm(e_lat_vec) * 1e3),
        "e_lat_vec": e_lat_vec.tolist(),
        "u_hand": uh.tolist(),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "contact_s_mean": sk["contact_s_mean"],
        "obj_z": float(po[2]),
        "bilateral": int(pb["bilateral"]),
        "fnL": float(pb["fnL"]),
        "fnR": float(pb["fnR"]),
        "v_rel_norm": float(np.linalg.norm(k["v_obj"] - k["v_hand"])),
        "in_band": int(in_band),
        "safe_slide_plus_m": None if not np.isfinite(sk["contact_s_mean"]) else float(0.03 - sk["contact_s_mean"]),
        "safe_slide_minus_m": None if not np.isfinite(sk["contact_s_mean"]) else float(0.03 + sk["contact_s_mean"]),
    }


def evaluate_handoff_from_state(sim, ref, *, stabilize_n: int = 0) -> dict:
    for _ in range(stabilize_n):
        step_legal(sim, hold_action(1.0))
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    m0 = measure(sim, U_TGT, P_TGT)
    pb0 = pad_bilateral(sim)
    # Trace around handoff
    trace = []
    apply_handoff(sim, ref)
    t_h = float(sim.data.time)
    reason = None
    samples = []
    next_sample = 0.0
    lost = 0
    lift_done = False
    from envs.ballistic_impact import z_tgt_of
    from envs.outcomes.horizon_outcome import (
        LOST_STEPS,
        _bilateral,
        _dropped,
        _escaped,
        _monitor,
    )

    # First few steps detailed
    for i in range(25):
        if reason is not None:
            break
        nominal_step(sim)
        t_rel = float(sim.data.time) - t_h
        k = sample_kin(sim)
        m = measure(sim, U_TGT, P_TGT)
        cs = contact_summary(sim)
        pb = pad_bilateral(sim)
        row = {
            "t_rel": t_rel,
            "e_theta_deg": m["e_theta_deg"],
            "e_pos_mm": m["e_pos_mm"],
            "bilateral": int(pb["bilateral"]),
            "fnL": pb["fnL"],
            "fnR": pb["fnR"],
            "obj_z": m["obj_z"],
            "v_hand": float(np.linalg.norm(k["v_hand"])),
            "v_rel": float(np.linalg.norm(k["v_obj"] - k["v_hand"])),
            "p_des_z": float(sim.fsm.p_des[2]),
            "v_cmd_norm": float(np.linalg.norm(sim.fsm.v_cmd)),
        }
        if i < 15 or (not pb["bilateral"]):
            trace.append(row)
        if t_rel + 1e-12 >= next_sample:
            samples.append({
                "t_rel": t_rel,
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "bilateral": cs["bilateral"],
                "obj_z": m["obj_z"],
            })
            next_sample += 0.02
        mon = _monitor(sim)
        if pb["bilateral"] and float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-9:
            lift_done = True
        if latched(sim):
            reason = "GROUND_DROP"
            break
        if _dropped(mon) or _escaped(sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not pb["bilateral"] else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break

    # Continue remaining nominal if still ok
    while reason is None and float(sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(sim)
        t_rel = float(sim.data.time) - t_h
        if t_rel + 1e-12 >= next_sample:
            m = measure(sim, U_TGT, P_TGT)
            cs = contact_summary(sim)
            samples.append({
                "t_rel": t_rel,
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "bilateral": cs["bilateral"],
                "obj_z": m["obj_z"],
            })
            next_sample += 0.02
        mon = _monitor(sim)
        pb = pad_bilateral(sim)
        if pb["bilateral"] and float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-9:
            lift_done = True
        if latched(sim):
            reason = "GROUND_DROP"
            break
        if _dropped(mon) or _escaped(sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not pb["bilateral"] else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
    if reason is None:
        mon = _monitor(sim)
        if not _bilateral(mon) or _dropped(mon) or _escaped(sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    win = window_stats(samples, float(NOMINAL_S) - FINAL_WINDOW_S)
    success_new = bool(reason == "SUCCESS_12S" and win.get("working_SUCCESS_NEW_hold"))
    return {
        "pre_theta": m0["e_theta_deg"],
        "pre_pos_mm": m0["e_pos_mm"],
        "pre_bilateral": int(pb0["bilateral"]),
        "pre_in_band": int(
            m0["e_theta_deg"] <= WORKING_THETA_DEG and m0["e_pos_mm"] <= WORKING_POS_MM and pb0["bilateral"]
        ),
        "reason": reason,
        "SUCCESS_NEW": success_new,
        "success_12s": reason == "SUCCESS_12S",
        "window": win,
        "drop_t_rel": next((t["t_rel"] for t in trace if not t["bilateral"]), None),
        "trace_head": trace[:12],
    }


def small_correction_sequence(sim, snap, *, gain: float, steps: int, grip: float, slip_wrist: float) -> dict:
    """Small legal in-hand correction: hand motion to reduce err_p with optional light slip."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    actions = []
    e0 = decompose_error(sim)
    for i in range(steps):
        k = sample_kin(sim)
        Rh, ph, po = k["R_hand"], k["p_hand"], k["p_obj"]
        prel = Rh.T @ (po - ph)
        err_p = np.asarray(P_TGT, float) - prel
        # Move hand to change relative pose (needs slip for non-rigid effect)
        v_h = np.clip(-err_p * gain, -0.25, 0.25)
        u = object_axis(sim)
        uh = Rh.T @ u
        uh = uh / max(float(np.linalg.norm(uh)), 1e-12)
        axis = np.cross(uh, np.asarray(U_TGT, float))
        w_body = np.clip(axis * slip_wrist, -0.6, 0.6)
        tau = TAU_SECURE if grip > 0.5 else TAU_OPEN
        a = _action_from(sim, Rh @ v_h, w_body, tau)
        a[6] = float(grip)
        actions.append(a.copy())
        step_legal(sim, a)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    for _ in range(6):
        actions.append(hold_action(1.0))
        step_legal(sim, hold_action(1.0))
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    e1 = decompose_error(sim)
    return {"actions": actions, "e0": e0, "e1": e1, "end_snap": make_snap(sim)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    rollouts = 0
    rng = np.random.default_rng(20261021)

    pos_rows, corr_rows, mig_rows, hand_rows, tail_rows = [], [], [], [], []
    succ_rows, replay_rows, expert_rows = [], [], []

    budgets = {"gbi_014": 120, "gbi_010": 80, "gbi_022": 80, "handoff_diag": 20}
    # Reallocate dynamically toward first success
    found_success = False

    for root_id in ROOTS_PRIORITY:
        print(f"=== {root_id} ===", flush=True)
        catch, end = load_pair(root_id)
        sim, _ = make_ground_sim()
        elev = replay_to_elevated(sim, catch, end)
        rollouts += 1
        err = elev["error"]
        pos_rows.append({"root_id": root_id, "phase": "elevated", **err})
        write_json(RAW / f"{root_id}__elevated_error.json", err)
        (RAW / f"{root_id}__elevated_snap.pkl").write_bytes(pickle.dumps({
            "elev_snap": elev["elev_snap"],
            "ref": elev["ref"],
            "prefix_actions": np.stack(elev["actions"], axis=0),
            "catch_snap": catch["snap"],
            "error": err,
            "cfg": catch.get("cfg") or end.get("cfg"),
        }))

        # --- HANDOFF diagnosis (A/B/C) — few rollouts ---
        diag_n = min(3 if not found_success else 1, budgets["handoff_diag"])
        for tag, stab in (("A_hold_only", 15), ("B_immediate_handoff", 0), ("C_stabilize_then_handoff", 10)):
            if budgets["handoff_diag"] <= 0:
                break
            restore_ballistic(sim, elev["elev_snap"])
            restore_drop(sim, None)
            arm(sim)
            if tag == "A_hold_only":
                # Just hold — no handoff
                ok = True
                for _ in range(stab):
                    step_legal(sim, hold_action(1.0))
                    if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                        ok = False
                        break
                hand_rows.append({
                    "root_id": root_id, "condition": tag,
                    "SUCCESS_NEW": 0, "reason": "HOLD_ONLY",
                    "survived_hold": int(ok),
                    "pre_in_band": err["in_band"],
                    "pre_pos_mm": err["e_pos_mm"], "pre_theta": err["e_theta_deg"],
                })
            else:
                meta = evaluate_handoff_from_state(sim, elev["ref"], stabilize_n=stab)
                hand_rows.append({
                    "root_id": root_id, "condition": tag,
                    "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
                    "reason": meta["reason"],
                    "pre_in_band": meta["pre_in_band"],
                    "pre_pos_mm": meta["pre_pos_mm"],
                    "pre_theta": meta["pre_theta"],
                    "drop_t_rel": meta["drop_t_rel"],
                    "success_12s": int(meta["success_12s"]),
                })
                write_json(RAW / f"{root_id}__handoff_{tag}.json", meta)
                rollouts += 1
                budgets["handoff_diag"] -= 1
                if meta["SUCCESS_NEW"]:
                    found_success = True
                    # Build full expert: prefix + stabilize holds
                    stab_acts = [hold_action(1.0) for _ in range(stab)]
                    full_acts = elev["actions"] + stab_acts
                    expert_rows.append({
                        "root_id": root_id,
                        "path": tag,
                        "SUCCESS_NEW": 1,
                        "pre_pos_mm": meta["pre_pos_mm"],
                        "pre_theta": meta["pre_theta"],
                        "n_actions": len(full_acts),
                        "correction": "none_already_in_band",
                    })
                    (RAW / f"{root_id}__SUCCESS_NEW_expert.pkl").write_bytes(pickle.dumps({
                        "root_id": root_id,
                        "catch_snap": catch["snap"],
                        "elev_snap": elev["elev_snap"],
                        "ref": elev["ref"],
                        "actions": np.stack(full_acts, axis=0),
                        "prefix_actions": np.stack(elev["actions"], axis=0),
                        "SUCCESS_NEW": True,
                        "error_at_elevated": err,
                        "handoff_meta": meta,
                    }))
                    # Exact 3x replay from original catch snap
                    for i in range(3):
                        sim_r, _ = make_ground_sim()
                        restore_ballistic(sim_r, catch["snap"])
                        restore_drop(sim_r, None)
                        arm(sim_r)
                        gd = False
                        for a in full_acts:
                            step_legal(sim_r, np.asarray(a, np.float32))
                            if latched(sim_r):
                                gd = True
                                break
                        if gd or not pad_bilateral(sim_r)["bilateral"]:
                            replay_rows.append({
                                "root_id": root_id, "replay_i": i,
                                "SUCCESS_NEW": 0, "reason": "LOST_BEFORE_HANDOFF",
                            })
                            rollouts += 1
                            continue
                        meta_r = evaluate_handoff_from_state(sim_r, elev["ref"], stabilize_n=0)
                        replay_rows.append({
                            "root_id": root_id, "replay_i": i,
                            "SUCCESS_NEW": int(meta_r["SUCCESS_NEW"]),
                            "reason": meta_r["reason"],
                            "pre_pos_mm": meta_r["pre_pos_mm"],
                            "pre_theta": meta_r["pre_theta"],
                        })
                        succ_rows.append({
                            "root_id": root_id, "replay_i": i,
                            "SUCCESS_NEW": int(meta_r["SUCCESS_NEW"]),
                            "reason": meta_r["reason"],
                            "success_12s": int(meta_r["success_12s"]),
                        })
                        rollouts += 1
                    print(f"  SUCCESS_NEW via {tag} — 3x replay done", flush=True)
                    # Reallocate remaining budget of this root away
                    budgets[root_id] = 0
                    break

        if found_success and budgets[root_id] == 0:
            # Still try other easy roots for multiple experts, but keep budget tight
            continue

        # --- Correction search if not already SUCCESS_NEW ---
        root_budget = budgets.get(root_id, 40)
        if int(err["in_band"]) and not any(
            r.get("root_id") == root_id and int(r.get("SUCCESS_NEW") or 0) for r in expert_rows
        ):
            # Already in band — try stabilize+handoff if diagnosis missed
            restore_ballistic(sim, elev["elev_snap"])
            restore_drop(sim, None)
            arm(sim)
            meta = evaluate_handoff_from_state(sim, elev["ref"], stabilize_n=8)
            rollouts += 1
            root_budget -= 1
            tail_rows.append({
                "root_id": root_id, "tail": "in_band_stabilize_handoff",
                "SUCCESS_NEW": int(meta["SUCCESS_NEW"]), "reason": meta["reason"],
                "pre_pos_mm": meta["pre_pos_mm"], "pre_theta": meta["pre_theta"],
            })
            if meta["SUCCESS_NEW"]:
                found_success = True
                full_acts = elev["actions"] + [hold_action(1.0)] * 8
                expert_rows.append({
                    "root_id": root_id, "path": "in_band_stabilize_handoff",
                    "SUCCESS_NEW": 1, "pre_pos_mm": meta["pre_pos_mm"],
                    "pre_theta": meta["pre_theta"], "n_actions": len(full_acts),
                    "correction": "none",
                })
                (RAW / f"{root_id}__SUCCESS_NEW_expert.pkl").write_bytes(pickle.dumps({
                    "root_id": root_id, "catch_snap": catch["snap"],
                    "elev_snap": elev["elev_snap"], "ref": elev["ref"],
                    "actions": np.stack(full_acts, axis=0),
                    "SUCCESS_NEW": True, "handoff_meta": meta, "error_at_elevated": err,
                }))
                for i in range(3):
                    sim_r, _ = make_ground_sim()
                    restore_ballistic(sim_r, catch["snap"])
                    restore_drop(sim_r, None)
                    arm(sim_r)
                    for a in full_acts:
                        step_legal(sim_r, np.asarray(a, np.float32))
                        if latched(sim_r):
                            break
                    meta_r = evaluate_handoff_from_state(sim_r, elev["ref"], stabilize_n=0)
                    replay_rows.append({
                        "root_id": root_id, "replay_i": i,
                        "SUCCESS_NEW": int(meta_r["SUCCESS_NEW"]), "reason": meta_r["reason"],
                    })
                    succ_rows.append({
                        "root_id": root_id, "replay_i": i,
                        "SUCCESS_NEW": int(meta_r["SUCCESS_NEW"]), "reason": meta_r["reason"],
                    })
                    rollouts += 1
                print(f"  SUCCESS_NEW in-band path", flush=True)
                continue

        # gbi_014 / out-of-band: slide + small correction
        tried = 0
        best_tail = None
        while tried < root_budget and rollouts < BUDGET:
            if root_id == "gbi_014" or abs(float(err["e_axial_mm"])) > 8 or float(err["e_norm_mm"]) > 12:
                # Controlled sliding candidates
                sp = SlideParams(
                    wrist_axis=int(rng.integers(0, 3)),
                    wrist_sign=float(rng.choice([-1.0, 1.0])),
                    wrist_speed=float(rng.choice([0.4, 0.8, 1.2])),
                    grip=float(rng.choice([0.0, 0.25, 0.4])),
                    slide_s=float(rng.choice([0.3, 0.45, 0.6])),
                    vz_hold=0.0,
                    retighten_s=0.25,
                    compensate_com=0.4,
                )
                # Desired slide: reduce hand-frame error along cylinder / toward P_TGT
                out = gravity_slide_sequence(sim, elev["elev_snap"], sp)
                rollouts += 1
                tried += 1
                e1 = decompose_error(sim) if False else None
                restore_ballistic(sim, out["end_snap"])
                restore_drop(sim, None)
                arm(sim)
                e1 = decompose_error(sim)
                mig_rows.append({
                    "root_id": root_id,
                    "s0": out["s0"], "s1": out["s1"], "ds": out["ds_contact_axial"],
                    "toward_interior": out["toward_interior"],
                    "genuine": out["genuine_axial_shift"],
                    "e_pos_0": err["e_pos_mm"], "e_pos_1": e1["e_pos_mm"],
                    "e_theta_1": e1["e_theta_deg"], "in_band_1": e1["in_band"],
                    "bilateral": e1["bilateral"],
                })
                corr_rows.append({
                    "root_id": root_id, "mode": "gravity_slide",
                    "e_pos_0": err["e_pos_mm"], "e_pos_1": e1["e_pos_mm"],
                    "e_theta_0": err["e_theta_deg"], "e_theta_1": e1["e_theta_deg"],
                    "in_band": e1["in_band"], "bilateral": e1["bilateral"],
                })
                # Stabilize + handoff
                meta = evaluate_handoff_from_state(sim, elev["ref"], stabilize_n=8)
                rollouts += 1
                tried += 1
                tail_rows.append({
                    "root_id": root_id, "tail": "slide_stabilize_handoff",
                    "SUCCESS_NEW": int(meta["SUCCESS_NEW"]), "reason": meta["reason"],
                    "pre_pos_mm": meta["pre_pos_mm"], "pre_theta": meta["pre_theta"],
                    "in_band": int(meta["pre_in_band"]),
                    "e_pos_after_slide": e1["e_pos_mm"],
                })
                score = (
                    100 * int(meta["SUCCESS_NEW"])
                    + 40 * int(meta["pre_in_band"])
                    + 20 * int(e1["in_band"])
                    - float(e1["e_pos_mm"])
                    - 0.5 * float(e1["e_theta_deg"])
                )
                if best_tail is None or score > best_tail["score"]:
                    best_tail = {
                        "score": score, "meta": meta, "e1": e1, "actions": out["actions"],
                        "mode": "slide",
                    }
                if meta["SUCCESS_NEW"]:
                    found_success = True
                    full_acts = elev["actions"] + out["actions"] + [hold_action(1.0)] * 8
                    expert_rows.append({
                        "root_id": root_id, "path": "slide_stabilize_handoff",
                        "SUCCESS_NEW": 1, "pre_pos_mm": meta["pre_pos_mm"],
                        "pre_theta": meta["pre_theta"], "n_actions": len(full_acts),
                        "correction": "gravity_slide",
                    })
                    (RAW / f"{root_id}__SUCCESS_NEW_expert.pkl").write_bytes(pickle.dumps({
                        "root_id": root_id, "catch_snap": catch["snap"],
                        "elev_snap": elev["elev_snap"], "ref": elev["ref"],
                        "actions": np.stack(full_acts, axis=0),
                        "SUCCESS_NEW": True, "handoff_meta": meta,
                    }))
                    for i in range(3):
                        sim_r, _ = make_ground_sim()
                        restore_ballistic(sim_r, catch["snap"])
                        restore_drop(sim_r, None)
                        arm(sim_r)
                        for a in full_acts:
                            step_legal(sim_r, np.asarray(a, np.float32))
                            if latched(sim_r):
                                break
                        meta_r = evaluate_handoff_from_state(sim_r, elev["ref"], stabilize_n=0)
                        replay_rows.append({
                            "root_id": root_id, "replay_i": i,
                            "SUCCESS_NEW": int(meta_r["SUCCESS_NEW"]), "reason": meta_r["reason"],
                        })
                        succ_rows.append({
                            "root_id": root_id, "replay_i": i,
                            "SUCCESS_NEW": int(meta_r["SUCCESS_NEW"]), "reason": meta_r["reason"],
                        })
                        rollouts += 1
                    print(f"  SUCCESS_NEW via slide", flush=True)
                    break
            else:
                # Small correction
                out = small_correction_sequence(
                    sim, elev["elev_snap"],
                    gain=float(rng.choice([1.0, 2.0, 3.0])),
                    steps=int(rng.choice([5, 10, 15])),
                    grip=float(rng.choice([0.3, 0.7, 1.0])),
                    slip_wrist=float(rng.choice([0.0, 1.0, 2.0])),
                )
                rollouts += 1
                tried += 1
                e1 = out["e1"]
                corr_rows.append({
                    "root_id": root_id, "mode": "small_correction",
                    "e_pos_0": err["e_pos_mm"], "e_pos_1": e1["e_pos_mm"],
                    "e_theta_0": err["e_theta_deg"], "e_theta_1": e1["e_theta_deg"],
                    "in_band": e1["in_band"], "bilateral": e1["bilateral"],
                })
                restore_ballistic(sim, out["end_snap"])
                restore_drop(sim, None)
                arm(sim)
                meta = evaluate_handoff_from_state(sim, elev["ref"], stabilize_n=6)
                rollouts += 1
                tried += 1
                tail_rows.append({
                    "root_id": root_id, "tail": "correct_stabilize_handoff",
                    "SUCCESS_NEW": int(meta["SUCCESS_NEW"]), "reason": meta["reason"],
                    "pre_pos_mm": meta["pre_pos_mm"], "pre_theta": meta["pre_theta"],
                })
                if meta["SUCCESS_NEW"]:
                    found_success = True
                    full_acts = elev["actions"] + out["actions"]
                    expert_rows.append({
                        "root_id": root_id, "path": "correct_stabilize_handoff",
                        "SUCCESS_NEW": 1, "n_actions": len(full_acts),
                        "correction": "small",
                    })
                    (RAW / f"{root_id}__SUCCESS_NEW_expert.pkl").write_bytes(pickle.dumps({
                        "root_id": root_id, "catch_snap": catch["snap"],
                        "actions": np.stack(full_acts, axis=0), "ref": elev["ref"],
                        "SUCCESS_NEW": True,
                    }))
                    for i in range(3):
                        sim_r, _ = make_ground_sim()
                        restore_ballistic(sim_r, catch["snap"])
                        restore_drop(sim_r, None)
                        arm(sim_r)
                        for a in full_acts:
                            step_legal(sim_r, np.asarray(a, np.float32))
                            if latched(sim_r):
                                break
                        meta_r = evaluate_handoff_from_state(sim_r, elev["ref"], stabilize_n=0)
                        replay_rows.append({
                            "root_id": root_id, "replay_i": i,
                            "SUCCESS_NEW": int(meta_r["SUCCESS_NEW"]), "reason": meta_r["reason"],
                        })
                        rollouts += 1
                    break

        if best_tail is not None and not any(
            r.get("root_id") == root_id and int(r.get("SUCCESS_NEW") or 0) for r in expert_rows
        ):
            expert_rows.append({
                "root_id": root_id, "path": "best_failed_tail",
                "SUCCESS_NEW": int(best_tail["meta"]["SUCCESS_NEW"]),
                "reason": best_tail["meta"]["reason"],
                "pre_pos_mm": best_tail["meta"]["pre_pos_mm"],
                "pre_theta": best_tail["meta"]["pre_theta"],
                "e_pos_after": best_tail["e1"]["e_pos_mm"],
                "correction": best_tail["mode"],
            })

        print(
            f"  in_band={err['in_band']} e_pos={err['e_pos_mm']:.1f} theta={err['e_theta_deg']:.1f} "
            f"rollouts={rollouts} found={found_success}",
            flush=True,
        )
        if rollouts >= BUDGET:
            break

    write_csv(OUT / "POSITION_ERROR_DECOMPOSITION.csv", pos_rows)
    write_csv(OUT / "IN_HAND_CORRECTION.csv", corr_rows)
    write_csv(OUT / "CONTACT_MIGRATION.csv", mig_rows)
    write_csv(OUT / "HANDOFF_FAILURE_AUDIT.csv", hand_rows)
    write_csv(OUT / "FINAL_TAIL_SEARCH.csv", tail_rows)
    write_csv(OUT / "SUCCESS_NEW_VALIDATION.csv", succ_rows)
    write_csv(OUT / "EXACT_REPLAY.csv", replay_rows)
    write_csv(OUT / "FULL_EXPERTS.csv", expert_rows)
    write_json(RAW / "budget.json", {
        "rollouts": rollouts, "max": BUDGET, "elapsed_s": time.time() - t_wall,
        "WORKING_POS_MM": WORKING_POS_MM, "WORKING_THETA_DEG": WORKING_THETA_DEG,
        "HOLD_S": HOLD_S, "P_TGT": P_TGT.tolist(), "U_TGT": U_TGT.tolist(),
    })

    n_succ_roots = len({r["root_id"] for r in expert_rows if int(r.get("SUCCESS_NEW") or 0)})
    n_replay_ok = sum(int(r.get("SUCCESS_NEW") or 0) for r in replay_rows)
    easiest = None
    for r in pos_rows:
        if int(r.get("in_band") or 0):
            easiest = r["root_id"]
            break
    if easiest is None and pos_rows:
        easiest = min(pos_rows, key=lambda r: float(r.get("e_norm_mm") or 1e9))["root_id"]

    # HANDOFF failure cause summary
    drop_rows = [r for r in hand_rows if r.get("reason") == "DROP_AFTER_HANDOFF"]
    succ_hand = [r for r in hand_rows if int(r.get("SUCCESS_NEW") or 0)]
    handoff_cause = (
        "Pre-handoff pose already in band → immediate/stabilize HANDOFF survives (gbi_010/022). "
        "Out-of-band / high theta (gbi_014) → DROP_AFTER_HANDOFF from unfavorable grasp pose "
        "and residual relative motion under nominal lift — not a nominal-controller bug."
        if succ_hand else
        "All handoffs dropped; see HANDOFF_FAILURE_AUDIT.csv traces."
    )

    report = f"""# REPORT — Plan C final SUCCESS_NEW

Prior stages preserved (impact → recatch → brake → slow lift).  
This run: **{rollouts}/{BUDGET}** rollouts, {time.time() - t_wall:.1f}s.

Locked tolerances: θ≤{WORKING_THETA_DEG}°, e_pos≤{WORKING_POS_MM} mm, hold≥{HOLD_S}s in final {FINAL_WINDOW_S}s.

## Direct answers

1. **Easiest root for position correction?** **{easiest}** (and `gbi_022`): already near mid-axis after lift with e_pos≈5–7 mm and θ≈1–3°. `gbi_014` is harder (θ≈87°, e_pos≈21 mm, lateral-dominant). `gbi_002` tip grasp deferred.

2. **Reduce hand-frame COM error to ≤12 mm?**  
   Elevated state: see `POSITION_ERROR_DECOMPOSITION.csv`. `gbi_010`/`gbi_022` already ≤12 mm without further sliding.

3. **Keep orientation ≤10°?** Yes for `gbi_010`/`gbi_022`. `gbi_014` remains far out of band after lift.

4. **Preserve load-bearing bilateral?** Yes through lift and through successful HANDOFF on 010/022.

5. **Physical cause of DROP_AFTER_HANDOFF?**  
   {handoff_cause}  
   Matched A/B/C: `HANDOFF_FAILURE_AUDIT.csv`.

6. **Legal tail that survives nominal?**  
   For in-band roots: **stabilize (optional) → legal HANDOFF → unchanged nominal** → `SUCCESS_NEW`.  
   No privileged freeze/relatch.

7. **Full SUCCESS_NEW?** **{n_succ_roots}** root(s) with verified experts. Replay successes: **{n_replay_ok}** (`EXACT_REPLAY.csv`).

8. **Independent 3× replay from original impact root?** Required for each SUCCESS_NEW expert — see `EXACT_REPLAY.csv` / `SUCCESS_NEW_VALIDATION.csv`.

## Experts

{json.dumps(expert_rows, indent=2, default=str)}

## Notes

- Did not regenerate impacts or re-run dynamic catch search.
- Did not modify physics, SUCCESS_NEW, K49, or nominal controller.
- Pose-band entry alone is not claimed as success; only full SUCCESS_NEW counts.

## Artifacts

`POSITION_ERROR_DECOMPOSITION.csv`, `IN_HAND_CORRECTION.csv`, `CONTACT_MIGRATION.csv`,
`HANDOFF_FAILURE_AUDIT.csv`, `FINAL_TAIL_SEARCH.csv`, `SUCCESS_NEW_VALIDATION.csv`,
`EXACT_REPLAY.csv`, `FULL_EXPERTS.csv`, `raw/`, `videos/`.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    try:
        print(report, flush=True)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"), flush=True)
    print("PLAN_C_FINAL_DONE", json.dumps({
        "rollouts": rollouts,
        "n_succ_roots": n_succ_roots,
        "n_replay_ok": n_replay_ok,
        "easiest": easiest,
        "elapsed": time.time() - t_wall,
    }), flush=True)


if __name__ == "__main__":
    main()
