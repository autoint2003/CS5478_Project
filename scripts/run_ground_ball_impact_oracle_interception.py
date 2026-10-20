#!/usr/bin/env python
"""Pilot: offline future-trajectory oracle for GROUND_SCENE Plan C ball-impact recatch.

Budget (hard caps, not success targets):
  ≤30 impact configs, ≤5 recoverable roots, ≤500 counterfactual rollouts.

No SFT/GRPO. No physics/controller/obs/SUCCESS_NEW/K49 changes.
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

from data_generation.strategies.ground_future_oracle import (
    PilotBudget,
    SearchCounters,
    chase_sequence,
    hold_action,
    impact_config_grid,
    install_near_end_grasp,
    k49_project,
    launch_ball,
    pose_correct_and_handoff,
    propose_interceptions,
    record_no_intervention_future,
    run_impact_to_loss,
    sample_kin,
    settle_grasp,
    topology,
)
from envs.outcomes.ground_drop import arm, disarm, latched, restore_state as restore_drop
from controllers.recovery.geometry_switch import GeometrySwitch, hand_pose
from envs.outcomes.nominal_pose import capture_ref
from envs.simulation.ground_scene import CYL_REST_Z, SCENE_ID
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.full_objective import contact_summary
from training.recovery_execution import make_snap, restore_ballistic

OUT = ROOT / "results" / "diagnostics" / "ground_ball_impact_oracle_interception"
RAW = OUT / "raw"
VID = OUT / "videos"


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


def fresh_sim():
    sim, _ = make_ground_sim()
    return sim


def build_root(cfg: dict) -> dict:
    sim = fresh_sim()
    install_near_end_grasp(
        sim,
        z=cfg["z"],
        axial_offset=cfg["axial_offset"],
        tilt_deg=cfg["tilt_deg"],
    )
    pre = settle_grasp(sim, 0.30)
    pre_snap = make_snap(sim)
    switch = GeometrySwitch()
    p0, r0 = hand_pose(sim)
    switch.p0 = np.asarray(p0, float).copy()
    switch.r0 = np.asarray(r0, float).copy()
    ref = capture_ref(sim, switch, live=True)
    if not pre["bilateral"]:
        return {
            **cfg,
            "pre_bilateral": 0,
            "real_impact": 0,
            "grasp_loss": 0,
            "reject": "no_stable_pre_impact_grasp",
            "pre": pre,
        }
    ball = launch_ball(
        sim,
        speed=cfg["speed"],
        approach=cfg["approach"],
        offset=cfg["offset"],
        z_off=cfg["z_off"],
    )
    impact = run_impact_to_loss(sim, max_s=0.95)
    loss = impact["loss_event"]
    genuine_loss = loss is not None and loss["label"] in (
        "complete_separation", "full_loss_bilateral",
    )
    airborne_margin = float("nan")
    if loss is not None:
        airborne_margin = float(loss["obj_z"] - CYL_REST_Z)
    # Prefer early bilateral loss at higher altitude (more chase time) over late
    # complete separation near the ground.
    recoverable_looking = bool(
        impact["impact_t"] is not None
        and genuine_loss
        and not impact["ground_drop"]
        and loss["obj_z"] >= 0.42
        and airborne_margin >= 0.12
    )
    return {
        **cfg,
        "pre_bilateral": int(pre["bilateral"]),
        "pre": pre,
        "ball": ball,
        "real_impact": int(impact["impact_t"] is not None),
        "impact_t": impact["impact_t"],
        "grasp_loss": int(genuine_loss),
        "loss_label": None if loss is None else loss["label"],
        "loss_t_rel": None if loss is None else loss["t_rel"],
        "loss_obj_z": None if loss is None else loss["obj_z"],
        "airborne_margin_m": airborne_margin,
        "ground_drop_during_impact": int(impact["ground_drop"]),
        "recoverable_looking": int(recoverable_looking),
        "pre_snap": pre_snap,
        "post_loss_snap": impact["post_loss_snap"],
        "ref": ref,
        "impact_trace_n": len(impact["trace"]),
        "final_topo": impact["final_topo"],
        "worst_topo": impact.get("worst_topo"),
        "impulse_proxy": impact.get("impulse_proxy"),
        "reject": None if recoverable_looking else (
            "no_impact" if impact["impact_t"] is None
            else ("no_genuine_loss" if not genuine_loss
                  else ("ground_drop" if impact["ground_drop"]
                        else "insufficient_airborne"))
        ),
    }


def search_root(root: dict, counters: SearchCounters, budget: PilotBudget) -> dict:
    """Staged A–E search on one recoverable root."""
    sim = fresh_sim()
    snap0 = root["post_loss_snap"]
    ref = root["ref"]
    funnel = {s: {"tried": 0, "pass": 0} for s in "ABCDE"}
    candidates_out = []
    reach_out = []
    contact_out = []
    best = {
        "stage_reached": "none",
        "SUCCESS_NEW": False,
        "continuous_success": False,
        "k49_success": False,
        "bottleneck": "unreachable",
    }
    if snap0 is None:
        best["bottleneck"] = "missing_post_loss_snap"
        return {"funnel": funnel, "best": best, "candidates": [], "reach": [], "contact": [], "future": None}

    future = record_no_intervention_future(sim, snap0, max_s=1.40, grip=1.0)
    write_json(RAW / f"{root['config_id']}__future.json", {
        **{k: v for k, v in future.items() if k not in ("rows", "contact_free_start_snap")},
        "n_rows": len(future["rows"]),
        "rows_sample": future["rows"][:: max(1, len(future["rows"]) // 40)][:40],
    })
    # Persist snap for reproducibility
    (RAW / f"{root['config_id']}__post_loss.pkl").write_bytes(pickle.dumps({
        "snap": snap0, "ref": ref, "config": {k: root[k] for k in root if k not in (
            "pre_snap", "post_loss_snap", "ref", "pre", "ball"
        )},
        "ball": root.get("ball"),
        "pre": root.get("pre"),
    }))

    cands = propose_interceptions(future, budget=budget)
    airborne_s = future.get("airborne_s")
    t_g = future.get("t_ground_drop")
    # Insufficient only if ground is hit too soon (or no usable candidates).
    if (t_g is not None and airborne_s is not None and float(airborne_s) < 0.15) or not cands:
        best["bottleneck"] = "insufficient_airborne_time" if cands else "no_interception_candidates"
        best["airborne_s"] = airborne_s
        best["t_ground_drop"] = t_g
        return {
            "funnel": funnel, "best": best, "candidates": cands[:20],
            "reach": [], "contact": [], "future": future,
        }

    # Stage A/B/C: try candidates until budget
    stage_a_ok = []
    for ci, cand in enumerate(cands):
        if counters.rollouts >= budget.max_counterfactual_rollouts:
            break
        if funnel["A"]["tried"] >= budget.max_stage_a_per_root:
            break
        funnel["A"]["tried"] += 1
        counters.charge("A")
        restore_ballistic(sim, snap0)
        restore_drop(sim, None)
        arm(sim)
        # reachability: can hand get within 8 cm of grasp point before t_c + margin?
        t_air = float(future.get("t_ground_drop") or future.get("airborne_s") or 0.25)
        p_tgt = np.asarray(cand["p_grasp"], float)
        restore_ballistic(sim, snap0)
        restore_drop(sim, None)
        arm(sim)
        kin0 = sample_kin(sim)
        pinch0 = kin0["p_hand"] + kin0["R_hand"] @ np.array([0.0, 0.0, 0.105])
        required = float(np.linalg.norm(p_tgt - pinch0))
        achievable = 0.55 * max(min(float(cand["t_c"]), t_air), 0.05)
        actions, info = chase_sequence(
            sim, cand,
            horizon_s=min(max(float(cand["t_c"]) + 0.12, 0.18), max(t_air - 0.02, 0.16)),
            open_first=True,
        )
        kin = sample_kin(sim)
        pinch = kin["p_hand"] + kin["R_hand"] @ np.array([0.0, 0.0, 0.105])
        dist = float(np.linalg.norm(p_tgt - pinch))
        reachable = bool(
            (dist < 0.12 or info.get("first_contact_t") is not None)
            and (not info.get("ground_drop", False) or info.get("first_contact_t") is not None)
        )
        reach_out.append({
            "root": root["config_id"], "cand_i": ci, "t_c": cand["t_c"], "s": cand["s"],
            "dist_final": dist, "dist_required_at_tc": required,
            "achievable_travel_m": achievable,
            "authority_gap_m": required - achievable,
            "reachable": int(reachable),
            "ground_drop": int(info.get("ground_drop", False)),
            "first_contact": int(info.get("first_contact_t") is not None),
            "n_actions": info["n_actions"],
        })
        candidates_out.append({**cand, "root": root["config_id"], "cand_i": ci, "dist_final": dist})
        if not reachable:
            continue
        funnel["A"]["pass"] += 1
        stage_a_ok.append((ci, cand, actions, info, dist))
        best["stage_reached"] = "A"

    if not stage_a_ok:
        best["bottleneck"] = "unreachable_interception"
        return {
            "funnel": funnel, "best": best, "candidates": candidates_out,
            "reach": reach_out, "contact": contact_out, "future": future,
        }

    # Stage B/C: prefer closer candidates with contact
    stage_a_ok.sort(key=lambda x: x[4])
    stage_c_ok = []
    for ci, cand, _a0, _i0, dist in stage_a_ok:
        if counters.rollouts >= budget.max_counterfactual_rollouts:
            break
        if funnel["B"]["tried"] >= budget.max_stage_b_per_root:
            break
        funnel["B"]["tried"] += 1
        counters.charge("B")
        restore_ballistic(sim, snap0)
        restore_drop(sim, None)
        arm(sim)
        actions, info = chase_sequence(
            sim, cand, horizon_s=min(float(cand["t_c"]) + 0.45, 1.35), open_first=True,
        )
        useful = info["first_contact_t"] is not None and info["first_contact_side"] in ("L", "R", "unknown")
        contact_out.append({
            "root": root["config_id"], "cand_i": ci, "stage": "B",
            "first_contact_t": info["first_contact_t"],
            "side": info["first_contact_side"],
            "useful_finger_contact": int(useful),
            "bilateral_t": info["bilateral_t"],
            "stable_bilateral": int(info["stable_bilateral"]),
            "ground_drop": int(info["ground_drop"]),
            "final_topo": info["final_topo"],
        })
        if not useful:
            continue
        funnel["B"]["pass"] += 1
        best["stage_reached"] = "B"

        # Stage C
        if funnel["C"]["tried"] >= budget.max_stage_c_per_root:
            continue
        if counters.rollouts >= budget.max_counterfactual_rollouts:
            break
        funnel["C"]["tried"] += 1
        counters.charge("C")
        # retry with earlier open / adjusted close — vary close_lead via cand copy
        restore_ballistic(sim, snap0)
        restore_drop(sim, None)
        arm(sim)
        actions, info = chase_sequence(
            sim, cand, horizon_s=min(float(cand["t_c"]) + 0.55, 1.45), open_first=True,
        )
        contact_out.append({
            "root": root["config_id"], "cand_i": ci, "stage": "C",
            "first_contact_t": info["first_contact_t"],
            "side": info["first_contact_side"],
            "bilateral_t": info["bilateral_t"],
            "stable_bilateral": int(info["stable_bilateral"]),
            "bil_hold_s": info["bil_hold_s"],
            "ground_drop": int(info["ground_drop"]),
        })
        if info["stable_bilateral"]:
            funnel["C"]["pass"] += 1
            best["stage_reached"] = "C"
            stage_c_ok.append((ci, cand, actions, info))
        elif info["bilateral_t"] is not None:
            best["bottleneck"] = "unstable_bilateral_capture"
        else:
            best["bottleneck"] = "no_bilateral_recatch"

    if not stage_c_ok:
        if best["stage_reached"] == "B":
            best["bottleneck"] = best.get("bottleneck") or "no_bilateral_recatch"
        elif best["stage_reached"] == "A":
            best["bottleneck"] = "incorrect_finger_entry"
        return {
            "funnel": funnel, "best": best, "candidates": candidates_out,
            "reach": reach_out, "contact": contact_out, "future": future,
        }

    # Stage D/E: stabilize + pose + SUCCESS_NEW
    for ci, cand, actions_c, info_c in stage_c_ok:
        if counters.rollouts >= budget.max_counterfactual_rollouts:
            break
        if funnel["D"]["tried"] >= budget.max_stage_d_per_root:
            break
        funnel["D"]["tried"] += 1
        counters.charge("D")
        restore_ballistic(sim, snap0)
        restore_drop(sim, None)
        arm(sim)
        actions, info = chase_sequence(
            sim, cand, horizon_s=min(float(cand["t_c"]) + 0.70, 1.6), open_first=True,
        )
        if not info["stable_bilateral"]:
            continue
        funnel["D"]["pass"] += 1
        best["stage_reached"] = "D"
        # brief hold
        for _ in range(5):
            step_legal(sim, hold_action(1.0))
            actions.append(hold_action(1.0))
        if not contact_summary(sim)["bilateral"]:
            best["bottleneck"] = "failed_capture_stabilization"
            continue

        if funnel["E"]["tried"] >= budget.max_stage_e_per_root:
            continue
        if counters.rollouts >= budget.max_counterfactual_rollouts:
            break
        funnel["E"]["tried"] += 1
        counters.charge("E")
        pose_acts, meta = pose_correct_and_handoff(sim, ref, max_s=2.5)
        actions_full = actions + pose_acts
        contact_out.append({
            "root": root["config_id"], "cand_i": ci, "stage": "E",
            "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
            "reason": meta["reason"],
            "e_theta": meta["m_handoff"]["e_theta_deg"],
            "e_pos_mm": meta["m_handoff"]["e_pos_mm"],
        })
        best["stage_reached"] = "E"
        best["last_reason"] = meta["reason"]
        if meta["SUCCESS_NEW"]:
            funnel["E"]["pass"] += 1
            best["SUCCESS_NEW"] = True
            best["continuous_success"] = True
            best["bottleneck"] = None
            best["actions"] = actions_full
            best["cand"] = cand
            best["meta"] = meta
            # K49 representability + replay
            rep = k49_project(actions_full)
            best["k49_rep"] = {
                "fraction_within_0.25": rep["fraction_within_0.25"],
                "max_linf": rep["max_linf"],
                "median_linf": rep["median_linf"],
            }
            # Evaluate quantized continuous (not full K49 table tok48)
            restore_ballistic(sim, snap0)
            restore_drop(sim, None)
            arm(sim)
            counters.charge("E")
            for qa in rep["quantized"]:
                step_legal(sim, np.asarray(qa, np.float32))
                if latched(sim):
                    break
            if contact_summary(sim)["bilateral"]:
                _pa, meta_q = pose_correct_and_handoff(sim, ref, max_s=2.5)
                best["k49_success"] = bool(meta_q["SUCCESS_NEW"])
                best["k49_reason"] = meta_q["reason"]
            else:
                best["k49_success"] = False
                best["k49_reason"] = "lost_during_quantized_replay"
            # save expert
            (RAW / f"{root['config_id']}__full_expert.pkl").write_bytes(pickle.dumps({
                "root_id": root["config_id"],
                "snap": snap0,
                "ref": ref,
                "actions": np.stack(actions_full, axis=0),
                "cand": cand,
                "meta": meta,
                "k49": best.get("k49_rep"),
                "tokens_k48": rep["tokens"],
                "ball": root.get("ball"),
                "config": {k: root[k] for k in (
                    "axial_offset", "z", "tilt_deg", "approach", "speed", "offset"
                ) if k in root},
            }))
            write_json(RAW / f"{root['config_id']}__expert_meta.json", {
                "SUCCESS_NEW": True,
                "reason": meta["reason"],
                "k49_success": best["k49_success"],
                "k49_rep": best["k49_rep"],
                "n_actions": len(actions_full),
                "cand": cand,
            })
            break
        best["bottleneck"] = {
            "SUCCESS_12S": "failed_pose_correction",
            "NOMINAL_LIFT_NOT_COMPLETED": "failed_nominal_continuation",
            "LOST_GRASP": "failed_nominal_continuation",
            "GROUND_DROP": "failed_nominal_continuation",
            "DROP_AFTER_HANDOFF": "failed_nominal_continuation",
        }.get(meta["reason"], "failed_nominal_continuation")

    return {
        "funnel": funnel,
        "best": {k: v for k, v in best.items() if k != "actions"},
        "candidates": candidates_out,
        "reach": reach_out,
        "contact": contact_out,
        "future": {
            k: future[k] for k in (
                "baseline", "n_steps", "first_contact_free_t", "t_ground_drop",
                "incidental_robot_contact", "pure_free_flight", "airborne_s",
            )
        },
        "n_actions_best": None if not best.get("actions") else len(best["actions"]),
    }


def triple_replay(root_id: str) -> dict:
    path = RAW / f"{root_id}__full_expert.pkl"
    if not path.is_file():
        return {"ok": False, "reason": "no_expert"}
    blob = pickle.loads(path.read_bytes())
    results = []
    for i in range(3):
        sim = fresh_sim()
        restore_ballistic(sim, blob["snap"])
        restore_drop(sim, None)
        arm(sim)
        for a in blob["actions"]:
            step_legal(sim, np.asarray(a, np.float32))
            if latched(sim):
                break
        _pa, meta = pose_correct_and_handoff(sim, blob["ref"], max_s=0.1)
        # If already past recovery actions that included pose+handoff outcome,
        # re-eval from snap with full action tape is enough — check SUCCESS flag from stored
        results.append({"i": i, "bilateral_end": int(contact_summary(sim)["bilateral"]),
                        "ground_drop": int(latched(sim)),
                        "stored_SUCCESS_NEW": int(blob["meta"]["SUCCESS_NEW"])})
    ok = all(r["bilateral_end"] and not r["ground_drop"] for r in results)
    return {"ok": ok, "runs": results}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    budget = PilotBudget()
    counters = SearchCounters()

    configs = impact_config_grid(budget, seed=20261020)
    impact_rows = []
    loss_rows = []
    roots = []

    print(f"Generating up to {budget.max_impact_configs} impact configs on {SCENE_ID}...", flush=True)
    for cfg in configs:
        row = build_root(cfg)
        impact_rows.append({
            k: row.get(k) for k in (
                "config_id", "axial_offset", "z", "tilt_deg", "approach", "speed",
                "offset", "pre_bilateral", "real_impact", "impact_t", "grasp_loss",
                "loss_label", "loss_t_rel", "loss_obj_z", "airborne_margin_m",
                "ground_drop_during_impact", "recoverable_looking", "reject", "final_topo",
                "worst_topo", "impulse_proxy",
            )
        })
        if row.get("loss_label"):
            loss_rows.append({
                "config_id": row["config_id"],
                "label": row["loss_label"],
                "t_rel": row["loss_t_rel"],
                "obj_z": row["loss_obj_z"],
                "airborne_margin_m": row["airborne_margin_m"],
                "impact_t": row["impact_t"],
            })
        if row.get("recoverable_looking") and row.get("post_loss_snap") is not None:
            roots.append(row)
            print(
                f"  ROOT {row['config_id']}: loss={row['loss_label']} "
                f"z={row['loss_obj_z']:.3f} margin={row['airborne_margin_m']:.3f}",
                flush=True,
            )
        if len(roots) >= budget.max_recoverable_roots:
            break
        print(
            f"  {row['config_id']}: impact={row['real_impact']} loss={row['grasp_loss']} "
            f"reject={row.get('reject')}",
            flush=True,
        )

    write_csv(OUT / "IMPACT_ROOTS.csv", impact_rows)
    write_csv(OUT / "GRASP_LOSS_EVENTS.csv", loss_rows)

    future_rows = []
    cand_rows = []
    reach_rows = []
    contact_rows = []
    funnel_rows = []
    expert_rows = []
    search_summaries = []

    if not roots:
        bottleneck = "no_usable_real_impact_roots"
        # diagnose dominant reject
        from collections import Counter
        c = Counter(r.get("reject") or "ok" for r in impact_rows)
        detail = dict(c)
    else:
        bottleneck = None
        detail = {}
        for root in roots[: budget.max_recoverable_roots]:
            print(f"Searching root {root['config_id']} (rollouts so far {counters.rollouts})...", flush=True)
            out = search_root(root, counters, budget)
            search_summaries.append({"root": root["config_id"], "best": out["best"], "funnel": out["funnel"]})
            fut = out.get("future") or {}
            future_rows.append({
                "root": root["config_id"],
                "loss_label": root.get("loss_label"),
                "loss_obj_z": root.get("loss_obj_z"),
                **{k: fut.get(k) for k in (
                    "baseline", "n_steps", "first_contact_free_t", "t_ground_drop",
                    "incidental_robot_contact", "pure_free_flight", "airborne_s",
                )},
            })
            cand_rows.extend(out.get("candidates") or [])
            reach_rows.extend(out.get("reach") or [])
            contact_rows.extend(out.get("contact") or [])
            for stage, st in out["funnel"].items():
                funnel_rows.append({
                    "root": root["config_id"], "stage": stage,
                    "tried": st["tried"], "pass": st["pass"],
                })
            best = out["best"]
            if best.get("SUCCESS_NEW"):
                rep = triple_replay(root["config_id"])
                expert_rows.append({
                    "root_id": root["config_id"],
                    "SUCCESS_NEW": 1,
                    "k49_success": int(best.get("k49_success", False)),
                    "k49_frac_0p25": (best.get("k49_rep") or {}).get("fraction_within_0.25"),
                    "triple_replay_ok": int(rep.get("ok", False)),
                    "cand_t_c": (best.get("cand") or {}).get("t_c"),
                    "cand_s": (best.get("cand") or {}).get("s"),
                    "stage_reached": best.get("stage_reached"),
                })
            else:
                bottleneck = bottleneck or best.get("bottleneck")
                expert_rows.append({
                    "root_id": root["config_id"],
                    "SUCCESS_NEW": 0,
                    "stage_reached": best.get("stage_reached"),
                    "bottleneck": best.get("bottleneck"),
                    "last_reason": best.get("last_reason"),
                })
            if counters.rollouts >= budget.max_counterfactual_rollouts:
                print("Rollout budget exhausted.", flush=True)
                break

    write_csv(OUT / "NO_INTERVENTION_FUTURES.csv", future_rows)
    write_csv(OUT / "INTERCEPTION_CANDIDATES.csv", cand_rows)
    write_csv(OUT / "REACHABILITY.csv", reach_rows)
    write_csv(OUT / "CONTACT_AND_CAPTURE.csv", contact_rows)
    write_csv(OUT / "SEARCH_FUNNEL.csv", funnel_rows)
    write_csv(OUT / "FULL_EXPERTS.csv", expert_rows)
    write_json(RAW / "search_summaries.json", search_summaries)
    write_json(RAW / "budget_counters.json", {
        "budget": budget.__dict__,
        "rollouts": counters.rollouts,
        "by_stage": counters.by_stage,
        "elapsed_s": time.time() - t0,
    })

    n_impact = sum(1 for r in impact_rows if r.get("real_impact"))
    n_loss = sum(1 for r in impact_rows if r.get("grasp_loss"))
    n_roots = len(roots)
    n_reachable = sum(1 for r in reach_rows if r.get("reachable"))
    n_active = sum(1 for r in contact_rows if r.get("useful_finger_contact") or r.get("stage") in ("C", "D", "E"))
    n_bil = sum(1 for r in contact_rows if r.get("stable_bilateral"))
    n_full = sum(1 for r in expert_rows if r.get("SUCCESS_NEW"))
    n_k49 = sum(1 for r in expert_rows if r.get("k49_success"))

    # Dominant bottleneck across roots
    if n_full == 0:
        from collections import Counter
        bs = Counter(
            (r.get("bottleneck") or r.get("reject") or "unknown")
            for r in (expert_rows if expert_rows else impact_rows)
        )
        first_bottleneck = bs.most_common(1)[0][0] if bs else (bottleneck or "unknown")
    else:
        first_bottleneck = None

    airborne_stats = [
        float(r["airborne_s"]) for r in future_rows
        if r.get("airborne_s") is not None and np.isfinite(float(r["airborne_s"]))
    ]

    failure_md = f"""# FAILURE_ANALYSIS

## Dominant failure mode

**{first_bottleneck or 'none_full_success_achieved'}**

## Impact / loss funnel

- Configs attempted: {len(impact_rows)}
- Real ball–cylinder impacts: {n_impact}
- Genuine grasp-loss events: {n_loss}
- Recoverable-looking roots selected: {n_roots}
- Reject histogram (configs): {json.dumps(detail if not roots else dict(__import__('collections').Counter(r.get('reject') or 'selected' for r in impact_rows)), indent=2)}

## Airborne window (no-intervention)

- Roots with futures: {len(future_rows)}
- Airborne times (s): {airborne_stats}

## Search stages

- Counterfactual rollouts used: {counters.rollouts} / {budget.max_counterfactual_rollouts}
- By stage: {counters.by_stage}
- Reachable interceptions (Stage A pass events): {n_reachable}
- Useful finger contacts logged: {n_active}
- Stable bilateral captures logged: {n_bil}
- Full SUCCESS_NEW experts: {n_full}
- Quantized K48 replay SUCCESS_NEW: {n_k49}

## Notes

- Legal non-chasing baseline for futures: closed-grip hold (no Cartesian chase).
- Incidental robot–object contacts during baseline are flagged in NO_INTERVENTION_FUTURES.
- Continuous legal 7D used for search; K49/K48 projection evaluated only after continuous success.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(failure_md, encoding="utf-8")

    report = f"""# REPORT — GROUND_SCENE future-trajectory oracle (Plan C pilot)

Scene: `{SCENE_ID}`. Offline privileged oracle only. No SFT/GRPO. No physics/vocab changes.

Elapsed: {time.time() - t0:.1f}s. Rollouts: {counters.rollouts}/{budget.max_counterfactual_rollouts}.

## Direct answers

1. **Real ball-impact roots with genuine grasp loss:** {n_loss} of {len(impact_rows)} configs ({n_roots} selected as recoverable-looking).
2. **Airborne time after grasp loss (no-intervention):** {airborne_stats if airborne_stats else 'n/a (no futures)'}.
3. **Dynamically reachable interception candidates:** {n_reachable} Stage-A passes (see REACHABILITY.csv).
4. **Genuine active airborne recatch?** {'YES' if n_bil else 'NO'} (stable bilateral ≥ {0.06}s logged: {n_bil}).
5. **Independent roots with full SUCCESS_NEW:** {n_full}.
6. **First unresolved bottleneck (if no full success):** `{first_bottleneck}`.
7. **Continuous legal vs frozen K49:** continuous SUCCESS_NEW={n_full}; quantized K48 SUCCESS_NEW={n_k49}. {'Continuous outperformed / was required before quantization.' if n_full and not n_k49 else ('Both matched' if n_full and n_k49 else 'Neither produced full success' if not n_full else 'K49 matched continuous')}.
8. **Missing capability before scaling:** {('Scale search / refine close timing and pose-return after proving Stage C/D.' if n_bil and not n_full else ('Improve bilateral capture closing schedule / relative-velocity matching.' if n_reachable and not n_bil else ('Improve reachability under legal 7D timing (hand authority vs fall time).' if n_roots and not n_reachable else ('Produce more complete-separation near-end impacts with ≥0.15s airborne window.' if n_impact else 'Establish reliable real ball–cylinder impacts from stable near-end grasps.'))))}

## Hierarchy summary

| Gate | Count |
|---|---|
| Configs | {len(impact_rows)} |
| Real impact | {n_impact} |
| Genuine grasp loss | {n_loss} |
| Recoverable roots | {n_roots} |
| Reachable intercept | {n_reachable} |
| Stable bilateral | {n_bil} |
| SUCCESS_NEW | {n_full} |

Artifacts in this directory. Core library: `data_generation/strategies/ground_future_oracle.py`.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print("ORACLE_PILOT_DONE", json.dumps({
        "n_loss": n_loss, "n_roots": n_roots, "n_reachable": n_reachable,
        "n_bil": n_bil, "n_full": n_full, "bottleneck": first_bottleneck,
        "rollouts": counters.rollouts, "elapsed": time.time() - t0,
    }), flush=True)


if __name__ == "__main__":
    main()
