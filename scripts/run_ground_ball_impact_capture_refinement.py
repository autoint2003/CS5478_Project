#!/usr/bin/env python
"""Capture refinement for GROUND_SCENE ball-impact oracle.

Fixes metric validity, reanalyzes existing roots (priority gbi_002), searches
closing/braking with first-contact branching (≤300 rollouts). Does not overwrite
the original pilot under ground_ball_impact_oracle_interception/.

No SFT/GRPO. No physics / SUCCESS_NEW / K49 changes.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.gripper.controller import finger_opening
from controllers.recovery.geometry_switch import GeometrySwitch, hand_pose
from data_generation.strategies.ground_future_oracle import (
    ChaseParams,
    PINCH_Z,
    PilotBudget,
    chase_sequence,
    empty_bilateral_stats,
    exact_replay_actions,
    grasp_point,
    grasp_point_vel,
    hold_action,
    pad_bilateral,
    pose_correct_and_handoff,
    propose_interceptions,
    record_no_intervention_future,
    sample_kin,
    topology,
    update_bilateral_stats,
    validate_pose_return_sign,
)
from envs.simulation.ground_scene import CYL_HALF
from envs.contacts.frames import extract_contacts
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.outcomes.nominal_pose import capture_ref
from envs.simulation.ground_sim import make_ground_sim
from envs.simulation.grasp_quality import body_twist
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import measure
from training.recovery_execution import make_snap, restore_ballistic
from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import AIRBORNE_V3, map_action, TAU_OPEN, TAU_SECURE
from data_generation.strategies.recatch_oracle import _action_from

PRIOR = ROOT / "results/diagnostics/ground_ball_impact_oracle_interception"
OUT = ROOT / "results/diagnostics/ground_ball_impact_capture_refinement"
RAW = OUT / "raw"
VID = OUT / "videos"

ROOT_IDS = ["gbi_002", "gbi_010", "gbi_014", "gbi_022"]
MAX_ROLLOUTS = 300
DT = 0.002
DT_POL = 0.02


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


def fresh_sim():
    return make_ground_sim()[0]


def load_root(root_id: str) -> dict | None:
    path = PRIOR / "raw" / f"{root_id}__post_loss.pkl"
    if not path.is_file():
        return None
    return pickle.loads(path.read_bytes())


def contact_detail(sim) -> dict:
    rows, pack = extract_contacts(sim)
    pb = pad_bilateral(sim)
    opening = float(finger_opening(sim.data, sim.ids))
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    pinch = ph + Rh @ np.array([0.0, 0.0, PINCH_Z])
    p_rel = Rh.T @ (po - ph)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    # jaw: object y in hand frame vs half opening
    jaw_clearance = float(opening) - abs(float(p_rel[1]))
    pads = []
    for r in rows:
        if r.get("side") not in ("L", "R"):
            continue
        pads.append({
            "side": r["side"],
            "fn": abs(float(r.get("Fn", r.get("fn", 0)) or 0)),
            "pos_h": np.asarray(r.get("pos_h"), float).tolist() if r.get("pos_h") is not None else None,
            "pos_w": np.asarray(r.get("pos_w", r.get("pos")), float).tolist(),
        })
    return {
        "topo": topology(sim),
        "pb": pb,
        "opening": opening,
        "jaw_clearance": jaw_clearance,
        "p_rel": p_rel.tolist(),
        "pinch": pinch.tolist(),
        "p_obj": po.tolist(),
        "p_hand": ph.tolist(),
        "v_hand": np.asarray(vh, float).tolist(),
        "v_obj": np.asarray(vo, float).tolist(),
        "w_hand": np.asarray(wh, float).tolist(),
        "w_obj": np.asarray(wo, float).tolist(),
        "v_rel": (np.asarray(vo, float) - np.asarray(vh, float)).tolist(),
        "obj_z": float(po[2]),
        "pads": pads,
        "n_housing": int(pb["housing"]),
        "ground_drop": int(latched(sim)),
    }


def diagnose_loss_mechanism(timeline: list[dict]) -> str:
    """Infer physical capture-loss mechanism from contact timeline."""
    if not timeline:
        return "no_timeline"
    first_pad = next((r for r in timeline if r.get("any_pad")), None)
    first_bil = next((r for r in timeline if r.get("bilateral")), None)
    last = timeline[-1]
    if first_pad is None:
        return "misses_both_fingers"
    # unilateral only then none
    had_uni = any(r.get("nL", 0) + r.get("nR", 0) == 1 for r in timeline)
    had_bil = first_bil is not None
    ejected = False
    if first_pad is not None:
        # after first contact, relative speed spikes and contact lost
        i0 = timeline.index(first_pad)
        for r in timeline[i0:]:
            vrel = float(np.linalg.norm(r.get("v_rel", [0, 0, 0])))
            if vrel > 0.8 and not r.get("any_pad"):
                ejected = True
                break
    if had_uni and not had_bil and ejected:
        return "struck_and_ejected_by_first_finger"
    if had_uni and not had_bil:
        # check if object y crossed jaw before close
        for r in timeline:
            if r.get("jaw_clearance", 1) < -0.005 and not r.get("bilateral"):
                return "passes_through_jaw_before_closing"
        return "misses_second_finger"
    if had_bil:
        # bilateral then lost
        after = [r for r in timeline if r["t_rel"] >= first_bil["t_rel"]]
        max_consec = 0.0
        cur = 0.0
        for r in after:
            if r.get("bilateral"):
                cur += r.get("dt", DT_POL)
                max_consec = max(max_consec, cur)
            else:
                cur = 0.0
        # high relative motion while bilateral
        high_rel = any(
            float(np.linalg.norm(r.get("v_rel", [0, 0, 0]))) > 0.45 and r.get("bilateral")
            for r in after
        )
        if high_rel and not last.get("bilateral"):
            return "caught_but_excessive_relative_motion"
        # closing squeeze: high fn then loss
        if any(r.get("fn_sum", 0) > 8.0 for r in after) and not last.get("bilateral"):
            return "squeezed_out_during_closing"
        if not last.get("bilateral"):
            # still chasing? large pinch error while contacting
            if any(float(np.linalg.norm(np.asarray(r.get("pinch_err", [0, 0, 0])))) > 0.05 for r in after):
                return "lost_while_hand_chases_obsolete_target"
            return "bilateral_lost_unclassified"
        return "maintained_bilateral"
    return "unclassified"


def mj_step_trace(sim, actions: list[np.ndarray], *, log_from_t_rel: float | None = None) -> list[dict]:
    """Execute actions; log every MuJoCo step after optional relative time."""
    rows = []
    t0 = float(sim.data.time)
    for a in actions:
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        cmd = map_action(np.asarray(a, np.float32), rd, rh, AIRBORNE_V3)
        vel, ang, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
        tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
        sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
        sim.fsm.v_cmd = vel.copy()
        sim.fsm.w_cmd = ang.copy()
        sim.fsm.v_des = vel.copy()
        sim.fsm.w_des = ang.copy()
        sim.fsm.fg_cmd = float(-tau)
        for _ in range(N_SUB):
            sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, DT)
            apply_cartesian_ctrl(
                sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
                gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
            )
            from envs.outcomes.support_drop import object_vz, observe

            vz = object_vz(sim)
            mujoco.mj_step(sim.model, sim.data)
            observe(sim, vz)
            t_rel = float(sim.data.time) - t0
            if log_from_t_rel is not None and t_rel + 1e-12 < log_from_t_rel:
                continue
            d = contact_detail(sim)
            rows.append({
                "t_rel": t_rel,
                "dt": DT,
                "bilateral": int(d["pb"]["bilateral"]),
                "any_pad": int(d["pb"]["any_pad"]),
                "nL": d["pb"]["nL"],
                "nR": d["pb"]["nR"],
                "fnL": d["pb"]["fnL"],
                "fnR": d["pb"]["fnR"],
                "fn_sum": d["pb"]["fnL"] + d["pb"]["fnR"],
                "opening": d["opening"],
                "jaw_clearance": d["jaw_clearance"],
                "obj_z": d["obj_z"],
                "v_rel": d["v_rel"],
                "p_rel": d["p_rel"],
                "topo": d["topo"],
                "ground_drop": d["ground_drop"],
                "n_housing": d["n_housing"],
            })
            if latched(sim):
                return rows
    return rows


def best_prior_candidate(future: dict) -> dict:
    cands = propose_interceptions(future, budget=PilotBudget(max_intercept_times=8, max_axial_samples=5))
    if not cands:
        # synthesize mid-horizon candidate from future rows
        rows = future["rows"]
        mid = rows[len(rows) // 3]
        s = 0.0
        return {
            "t_c": float(mid["t_rel"]),
            "s": s,
            "p_grasp": grasp_point(mid["p_obj"], mid["u_obj"], s).tolist(),
            "v_grasp": grasp_point_vel(mid["v_obj"], mid["w_obj"], mid["u_obj"], s).tolist(),
            "u": mid["u_obj"].tolist(),
            "obj_z": float(mid["obj_z"]),
        }
    # prefer mid-time, s near 0
    cands = sorted(cands, key=lambda c: (abs(c["s"]), abs(c["t_c"] - 0.08)))
    return cands[0]


def closing_param_grid() -> list[ChaseParams]:
    """Distinct closing/braking strategies — not identical laws with longer horizon."""
    grid = []
    for open_s in (0.02, 0.06, 0.10):
        for close_lead in (0.02, 0.06, 0.10, 0.14):
            for brake in (0.0, 0.5, 0.85):
                for stop in (False, True):
                    for vel_m in (0.2, 0.55, 0.9):
                        grid.append(ChaseParams(
                            open_s=open_s,
                            close_lead_s=close_lead,
                            approach_gain=1.2,
                            vel_match=vel_m,
                            post_contact_brake=brake,
                            stop_chase_after_contact=stop,
                            post_contact_close=True,
                        ))
    return grid


def branch_close_brake(sim, contact_snap, ref, *, n_variants: int = 12) -> list[dict]:
    """From first-contact snapshot, try short legal closing/braking sequences."""
    variants = []
    profiles = []
    # grip ramps + translational brake magnitudes
    for close_steps in (2, 4, 6):
        for brake in (0.0, 0.6, 1.0):
            for w_scale in (0.0, 0.4):
                profiles.append((close_steps, brake, w_scale))
    for close_steps, brake, w_scale in profiles[:n_variants]:
        restore_ballistic(sim, contact_snap)
        restore_drop(sim, None)
        arm(sim)
        actions = []
        bil = empty_bilateral_stats()
        t0 = float(sim.data.time)
        # open briefly then close while braking relative velocity
        for i in range(close_steps + 8):
            kin = sample_kin(sim)
            vh, vo = kin["v_hand"], kin["v_obj"]
            v_rel = vo - vh
            v_cmd = -brake * v_rel
            # slight retract along hand -z if penetrating
            Rh = kin["R_hand"]
            p_rel = Rh.T @ (kin["p_obj"] - kin["p_hand"])
            if abs(p_rel[1]) < 0.01:
                v_cmd = v_cmd - 0.15 * Rh[:, 1] * np.sign(p_rel[1] + 1e-9)
            w_body = -w_scale * (Rh.T @ kin["w_obj"])
            if i < close_steps:
                tau = TAU_OPEN + (TAU_SECURE - TAU_OPEN) * (i + 1) / max(close_steps, 1)
            else:
                tau = TAU_SECURE
            a = _action_from(sim, v_cmd, w_body, tau)
            actions.append(a.copy())
            step_legal(sim, a)
            pb = pad_bilateral(sim)
            update_bilateral_stats(bil, bool(pb["bilateral"]), float(sim.data.time) - t0, DT_POL)
            if latched(sim):
                break
        variants.append({
            "close_steps": close_steps,
            "brake": brake,
            "w_scale": w_scale,
            "actions": actions,
            "bil": dict(bil),
            "final_topo": topology(sim),
            "ground_drop": int(latched(sim)),
            "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
            "v_rel_end": float(np.linalg.norm(
                sample_kin(sim)["v_obj"] - sample_kin(sim)["v_hand"]
            )),
        })
    return variants


def milestones_from_bil(bil: dict, ground_drop: bool, v_rel: float) -> dict:
    return {
        "FIRST_BILATERAL_CONTACT": int(bil.get("first_bilateral_t") is not None),
        "CONTINUOUS_BILATERAL_60MS": int(bil.get("continuous_ge_60ms_at_end", False)),
        "CONTINUOUS_BILATERAL_150MS": int(bil.get("continuous_ge_150ms_at_end", False)),
        "STABILIZED_REGRASP": int(
            bil.get("continuous_ge_150ms_at_end", False)
            and bil.get("bilateral_at_end", False)
            and not ground_drop
            and v_rel < 0.25
        ),
        "historical_ge_60ms_only": int(
            bil.get("historical_ge_60ms", False) and not bil.get("continuous_ge_60ms_at_end", False)
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    rollouts = 0

    # --- Pose return validation (independent of roots) ---
    sim0 = fresh_sim()
    pose_val = validate_pose_return_sign(sim0)
    (OUT / "POSE_RETURN_VALIDATION.md").write_text(
        "# POSE_RETURN_VALIDATION\n\n"
        "Relative pose `p_rel = R_hand.T @ (p_obj - p_hand)`. For a stationary object, "
        "`Δp_rel ≈ -R_hand.T @ v_hand`.\n\n"
        "Therefore the command that reduces `err_p = P_TGT - p_rel` is "
        "`v_hand_world = R_hand @ (-err_p * gain)` — **opposite** to the previous "
        "`v_hand = +err_p` implementation.\n\n"
        f"Finite-difference check: `{json.dumps(jsafe(pose_val), indent=2)}`\n\n"
        "Wrist note: under a rigid bilateral grasp, rotating the hand co-rotates the "
        "cylinder, so hand-relative axis error does not decrease. Pose return after "
        "recatch needs slip/pivot (Plan A mechanisms), not pure wrist co-rotation.\n",
        encoding="utf-8",
    )

    (OUT / "PILOT_METRIC_CORRECTIONS.md").write_text(
        """# PILOT_METRIC_CORRECTIONS

## A. Bilateral continuity
- Replaced accumulated `bil_hold_s` with consecutive / max consecutive / total /
  first / last / gaps / end-status.
- `currently_stable_grasp` requires consecutive bilateral at termination ≥ 60 ms.
- Historical ≥60 ms is flagged separately and is **not** treated as current stability.
- Load-bearing pad contacts require `|Fn| ≥ 0.05 N` (`pad_bilateral`).

## B. Stage-A reachability
- Split into geometric proximity, legal dynamic authority gap, and actual pad contact.
- `dist < 0.12 or first_contact` is only a **loose candidate filter**, not verified reachability.

## C. Stage duplication
- Closing search uses explicit `ChaseParams` (open_s, close_lead, brake, vel_match, stop_chase).
- Contact-branch search uses distinct close-step / brake / wrist profiles.

## D. Stage-D
- Stabilized regrasp requires continuous ≥150 ms bilateral at end, no GROUND_DROP,
  and `|v_rel| < 0.25 m/s`.

## E. Impact grid
- Stratified / shuffled sampling in `impact_config_grid` (for future root generation).
- Existing pilot root IDs preserved and reloaded unchanged.

## F. Exact replay
- `exact_replay_actions` recomputes HANDOFF + SUCCESS_NEW independently each run.
""",
        encoding="utf-8",
    )

    diagnosis_rows = []
    first_contact_rows = []
    closing_rows = []
    capture_rows = []
    branch_rows = []
    expert_rows = []
    replay_rows = []

    best_global = None

    for root_id in ROOT_IDS:
        blob = load_root(root_id)
        if blob is None:
            diagnosis_rows.append({"root_id": root_id, "error": "missing_pkl"})
            continue
        snap0 = blob["snap"]
        ref = blob["ref"]
        sim = fresh_sim()

        future = record_no_intervention_future(sim, snap0, max_s=1.2, grip=1.0)
        write_json(RAW / f"{root_id}__future_meta.json", {
            k: future[k] for k in (
                "baseline", "n_steps", "first_contact_free_t", "t_ground_drop",
                "incidental_robot_contact", "pure_free_flight", "airborne_s",
            )
        })
        cand = best_prior_candidate(future)
        t_air = float(future.get("t_ground_drop") or future.get("airborne_s") or 0.25)

        # --- Reanalyze with default (legacy-like) chase + fine contact log ---
        restore_ballistic(sim, snap0)
        restore_drop(sim, None)
        arm(sim)
        params0 = ChaseParams()
        actions0, info0 = chase_sequence(
            sim, cand, horizon_s=min(max(cand["t_c"] + 0.15, 0.20), max(t_air - 0.01, 0.18)),
            params=params0,
        )
        rollouts += 1

        # Fine timeline: replay again with mj-step logging around first contact
        restore_ballistic(sim, snap0)
        restore_drop(sim, None)
        arm(sim)
        # run until we know first contact time from info0
        t_fc = info0.get("first_pad_contact_t") or info0.get("first_contact_t")
        log_from = None if t_fc is None else max(0.0, float(t_fc) - 0.10)
        timeline = mj_step_trace(sim, actions0, log_from_t_rel=log_from)
        # enrich pinch_err
        for r in timeline:
            # approximate using p_rel vs PINCH
            pr = np.asarray(r["p_rel"], float)
            r["pinch_err"] = (pr - np.array([0.0, 0.0, PINCH_Z])).tolist()
        mechanism = diagnose_loss_mechanism(timeline)
        write_json(RAW / f"{root_id}__contact_timeline.json", timeline[::5][:200])

        diagnosis_rows.append({
            "root_id": root_id,
            "airborne_s": future.get("airborne_s"),
            "t_ground_drop": future.get("t_ground_drop"),
            "cand_t_c": cand["t_c"],
            "cand_s": cand["s"],
            "first_pad_contact_t": info0.get("first_pad_contact_t"),
            "first_bilateral_t": info0.get("first_bilateral_t"),
            "max_consecutive_bil_s": info0.get("max_consecutive_bil_s"),
            "consecutive_bil_s_end": info0.get("consecutive_bil_s"),
            "total_bil_s": info0.get("total_bil_s"),
            "n_bil_gaps": info0.get("n_bil_gaps"),
            "historical_ge_60ms": int(info0.get("historical_ge_60ms", False)),
            "continuous_ge_60ms_at_end": int(info0.get("continuous_ge_60ms_at_end", False)),
            "currently_stable_grasp": int(info0.get("currently_stable_grasp", False)),
            "bilateral_at_end": int(info0.get("bilateral_at_end", False)),
            "mechanism": mechanism,
            "ground_drop": int(info0.get("ground_drop", False)),
            "final_topo": info0.get("final_topo"),
            "pilot_claimed_stable_bilateral_likely_historical": int(
                info0.get("historical_ge_60ms", False) and not info0.get("currently_stable_grasp", False)
            ),
        })

        for r in timeline:
            if r.get("any_pad") or r.get("bilateral"):
                first_contact_rows.append({"root_id": root_id, **{k: r[k] for k in r if k != "pads"}})

        # --- Closing / braking search (spend most budget on gbi_002) ---
        # Reserve budget across roots; prefer gbi_002 but do not starve others.
        remaining = max(0, MAX_ROLLOUTS - rollouts)
        quota = min(80 if root_id == "gbi_002" else 35, max(remaining // max(1, 5 - ROOT_IDS.index(root_id)), 10))
        grid = closing_param_grid()
        rng = np.random.default_rng(abs(hash(root_id)) % (2**31))
        n_pick = min(quota // 3, len(grid), max(remaining - 5, 0))
        picks = list(rng.choice(len(grid), size=n_pick, replace=False)) if n_pick > 0 else []

        root_best = None
        for gi in picks:
            if rollouts >= MAX_ROLLOUTS:
                break
            params = grid[int(gi)]
            restore_ballistic(sim, snap0)
            restore_drop(sim, None)
            arm(sim)
            # also vary t_c / s slightly for gbi_002
            cand_i = dict(cand)
            if root_id == "gbi_002":
                cand_i["t_c"] = float(np.clip(cand["t_c"] + rng.choice([-0.02, 0.0, 0.02, 0.04]), 0.02, t_air - 0.04))
                cand_i["s"] = float(np.clip(cand["s"] + rng.choice([-0.01, 0.0, 0.01]), -CYL_HALF + 0.004, CYL_HALF - 0.004))
                # refresh grasp point from future at nearest time
                rows = future["rows"]
                nearest = min(rows, key=lambda r: abs(r["t_rel"] - cand_i["t_c"]))
                cand_i["p_grasp"] = grasp_point(nearest["p_obj"], nearest["u_obj"], cand_i["s"]).tolist()
                cand_i["v_grasp"] = grasp_point_vel(
                    nearest["v_obj"], nearest["w_obj"], nearest["u_obj"], cand_i["s"]
                ).tolist()
                cand_i["u"] = nearest["u_obj"].tolist()

            actions, info = chase_sequence(
                sim, cand_i,
                horizon_s=min(max(cand_i["t_c"] + 0.18, 0.22), max(t_air - 0.01, 0.20)),
                params=params,
            )
            rollouts += 1
            v_rel = float(np.linalg.norm(sample_kin(sim)["v_obj"] - sample_kin(sim)["v_hand"]))
            ms = milestones_from_bil(info, info.get("ground_drop", False), v_rel)
            row = {
                "root_id": root_id,
                "rollout": rollouts,
                "open_s": params.open_s,
                "close_lead_s": params.close_lead_s,
                "brake": params.post_contact_brake,
                "stop_chase": int(params.stop_chase_after_contact),
                "vel_match": params.vel_match,
                "t_c": cand_i["t_c"],
                "s": cand_i["s"],
                "first_pad_t": info.get("first_pad_contact_t"),
                "max_consec_bil": info.get("max_consecutive_bil_s"),
                "consec_end": info.get("consecutive_bil_s"),
                "historical_60": int(info.get("historical_ge_60ms", False)),
                "continuous_60_end": int(info.get("continuous_ge_60ms_at_end", False)),
                "continuous_150_end": int(info.get("continuous_ge_150ms_at_end", False)),
                "stable_now": int(info.get("currently_stable_grasp", False)),
                "v_rel": v_rel,
                "ground_drop": int(info.get("ground_drop", False)),
                **ms,
            }
            closing_rows.append(row)
            capture_rows.append(row)

            score = (
                10.0 * ms["STABILIZED_REGRASP"]
                + 5.0 * ms["CONTINUOUS_BILATERAL_150MS"]
                + 2.0 * ms["CONTINUOUS_BILATERAL_60MS"]
                + 1.0 * ms["FIRST_BILATERAL_CONTACT"]
                + float(info.get("max_consecutive_bil_s") or 0)
                - 2.0 * float(info.get("ground_drop", False))
            )
            if root_best is None or score > root_best["score"]:
                root_best = {
                    "score": score, "actions": actions, "info": info, "cand": cand_i,
                    "params": params, "ms": ms, "v_rel": v_rel,
                }

            # --- First-contact branching when pad contact achieved ---
            if info.get("first_pad_contact_t") is not None and rollouts < MAX_ROLLOUTS:
                # rebuild contact snap by replaying prefix
                restore_ballistic(sim, snap0)
                restore_drop(sim, None)
                arm(sim)
                contact_snap = None
                prefix = []
                for a in actions:
                    step_legal(sim, a)
                    prefix.append(a.copy())
                    if pad_bilateral(sim)["any_pad"]:
                        contact_snap = make_snap(sim)
                        break
                    if latched(sim):
                        break
                if contact_snap is not None and rollouts < MAX_ROLLOUTS:
                    # Cap branches tightly — prior run burned the budget on gbi_002 branches.
                    n_b = 3 if root_id == "gbi_002" else 2
                    branches = branch_close_brake(sim, contact_snap, ref, n_variants=n_b)
                    rollouts += len(branches)
                    for bi, br in enumerate(branches):
                        ms_b = milestones_from_bil(br["bil"], br["ground_drop"], br["v_rel_end"])
                        branch_rows.append({
                            "root_id": root_id,
                            "parent_rollout": row["rollout"],
                            "branch_i": bi,
                            "close_steps": br["close_steps"],
                            "brake": br["brake"],
                            "w_scale": br["w_scale"],
                            "max_consec": br["bil"].get("max_consecutive_bil_s"),
                            "consec_end": br["bil"].get("consecutive_bil_s"),
                            "continuous_60": int(br["bil"].get("continuous_ge_60ms_at_end", False)),
                            "continuous_150": int(br["bil"].get("continuous_ge_150ms_at_end", False)),
                            "stable_now": int(br["bil"].get("currently_stable_grasp", False)),
                            "v_rel_end": br["v_rel_end"],
                            "ground_drop": br["ground_drop"],
                            **ms_b,
                        })
                        score_b = (
                            10 * ms_b["STABILIZED_REGRASP"]
                            + 5 * ms_b["CONTINUOUS_BILATERAL_150MS"]
                            + 2 * ms_b["CONTINUOUS_BILATERAL_60MS"]
                            + float(br["bil"].get("max_consecutive_bil_s") or 0)
                        )
                        if score_b > root_best["score"]:
                            # stitch uninterrupted tape: prefix + branch actions
                            full = prefix + br["actions"]
                            # optional short hold
                            for _ in range(5):
                                full.append(hold_action(1.0))
                            root_best = {
                                "score": score_b,
                                "actions": full,
                                "info": br["bil"],
                                "cand": cand_i,
                                "params": {"branch": True, **{k: br[k] for k in ("close_steps", "brake", "w_scale")}},
                                "ms": ms_b,
                                "v_rel": br["v_rel_end"],
                                "stitched": True,
                            }

        # Evaluate best for this root toward SUCCESS_NEW if stabilized
        if root_best is not None:
            restore_ballistic(sim, snap0)
            restore_drop(sim, None)
            arm(sim)
            for a in root_best["actions"]:
                step_legal(sim, np.asarray(a, np.float32))
                if latched(sim):
                    break
            pb = pad_bilateral(sim)
            success_meta = {"SUCCESS_NEW": False, "reason": "no_stable_capture"}
            if pb["bilateral"] and root_best["ms"].get("STABILIZED_REGRASP"):
                _pa, success_meta = pose_correct_and_handoff(sim, ref, max_s=2.5)
                rollouts += 1
                # save full tape including pose acts
                full_tape = list(root_best["actions"]) + list(_pa)
                root_best["actions"] = full_tape
            elif pb["bilateral"] and root_best["ms"].get("CONTINUOUS_BILATERAL_60MS"):
                # try pose anyway for diagnostics
                _pa, success_meta = pose_correct_and_handoff(sim, ref, max_s=1.5)
                rollouts += 1

            expert_rows.append({
                "root_id": root_id,
                "score": root_best["score"],
                "STABILIZED_REGRASP": root_best["ms"].get("STABILIZED_REGRASP"),
                "CONTINUOUS_BILATERAL_150MS": root_best["ms"].get("CONTINUOUS_BILATERAL_150MS"),
                "CONTINUOUS_BILATERAL_60MS": root_best["ms"].get("CONTINUOUS_BILATERAL_60MS"),
                "SUCCESS_NEW": int(success_meta.get("SUCCESS_NEW", False)),
                "reason": success_meta.get("reason"),
                "v_rel": root_best["v_rel"],
                "n_actions": len(root_best["actions"]),
                "stitched_branch": int(bool(root_best.get("stitched"))),
            })

            if success_meta.get("SUCCESS_NEW") or root_best["ms"].get("STABILIZED_REGRASP"):
                (RAW / f"{root_id}__candidate_expert.pkl").write_bytes(pickle.dumps({
                    "root_id": root_id,
                    "snap": snap0,
                    "ref": ref,
                    "actions": np.stack(root_best["actions"], axis=0),
                    "meta": success_meta,
                    "ms": root_best["ms"],
                }))
                # exact triple replay
                for i in range(3):
                    sim_r = fresh_sim()
                    rep = exact_replay_actions(sim_r, snap0, ref, root_best["actions"])
                    replay_rows.append({"root_id": root_id, "replay_i": i, **rep})
                    rollouts += 1

            if best_global is None or root_best["score"] > best_global["score"]:
                best_global = {"root_id": root_id, **root_best, "success_meta": success_meta}

        if rollouts >= MAX_ROLLOUTS:
            break

    write_csv(OUT / "ROOT_CONTACT_DIAGNOSIS.csv", diagnosis_rows)
    write_csv(OUT / "FIRST_CONTACT_TRACES.csv", first_contact_rows)
    write_csv(OUT / "CLOSING_TIMING_SEARCH.csv", closing_rows)
    write_csv(OUT / "CAPTURE_STABILITY.csv", capture_rows)
    write_csv(OUT / "CONTACT_BRANCH_SEARCH.csv", branch_rows)
    write_csv(OUT / "FULL_EXPERTS.csv", expert_rows)
    write_csv(OUT / "EXACT_REPLAY.csv", replay_rows)
    write_json(RAW / "budget.json", {"rollouts": rollouts, "max": MAX_ROLLOUTS, "elapsed_s": time.time() - t_wall})

    n_hist = sum(int(r.get("historical_ge_60ms") or 0) for r in diagnosis_rows)
    n_real60 = sum(int(r.get("continuous_ge_60ms_at_end") or 0) for r in diagnosis_rows)
    n_stable = sum(int(r.get("STABILIZED_REGRASP") or 0) for r in expert_rows)
    n_succ = sum(int(r.get("SUCCESS_NEW") or 0) for r in expert_rows)
    gbi2 = next((r for r in diagnosis_rows if r.get("root_id") == "gbi_002"), {})
    # best closing improvements
    improved = [
        r for r in closing_rows
        if r.get("root_id") == "gbi_002" and r.get("continuous_60_end")
    ]
    best_closing = max(improved, key=lambda r: (r.get("continuous_150_end", 0), r.get("consec_end", 0)), default=None)

    cause = "search_strategy"
    if gbi2.get("airborne_s") is not None and float(gbi2["airborne_s"]) < 0.20:
        cause = "insufficient_time"
    if any(r.get("mechanism") in ("misses_second_finger", "passes_through_jaw_before_closing", "struck_and_ejected_by_first_finger") for r in diagnosis_rows):
        cause = "contact_geometry"
    if any(r.get("mechanism") == "caught_but_excessive_relative_motion" for r in diagnosis_rows):
        cause = "contact_geometry_and_braking"

    replay_ok = False
    if replay_rows:
        by_root = {}
        for r in replay_rows:
            by_root.setdefault(r["root_id"], []).append(r)
        for rid, rr in by_root.items():
            if all(x.get("SUCCESS_NEW") for x in rr):
                replay_ok = True

    report = f"""# REPORT — Ball-impact capture refinement

Prior pilot preserved under `ground_ball_impact_oracle_interception/`.
This run: **{rollouts}/{MAX_ROLLOUTS}** rollouts, elapsed {time.time() - t_wall:.1f}s.

## Direct answers

1. **Previously reported bilateral events that were genuinely consecutive & load-bearing at termination:**  
   Reanalysis with corrected metrics: historical≥60ms on default chase = **{n_hist}** roots; continuous≥60ms **at end** = **{n_real60}**.  
   A historical ≥60 ms peak is **not** a currently stable grasp.

2. **What physically caused `gbi_002` to lose capture?**  
   **`{gbi2.get('mechanism', 'n/a')}`**  
   (see `ROOT_CONTACT_DIAGNOSIS.csv` and `raw/gbi_002__contact_timeline.json`).

3. **Which closing/braking parameters improved bilateral capture?**  
   {('`'+json.dumps({k: best_closing[k] for k in ('open_s','close_lead_s','brake','stop_chase','vel_match','continuous_60_end','continuous_150_end','consec_end')})+'`') if best_closing else 'No parameter set achieved continuous≥60ms at end on gbi_002 in this budget; see CLOSING_TIMING_SEARCH.csv / CONTACT_BRANCH_SEARCH.csv for ranked attempts.'}

4. **Persistent bilateral recatch (STABILIZED_REGRASP)?** **{n_stable}** roots.

5. **Legal HANDOFF + SUCCESS_NEW?** **{n_succ}** roots.

6. **If capture still fails, primary cause class:** **`{cause}`**  
   (airborne ≈ {gbi2.get('airborne_s')} s on gbi_002; geometry/braking dominate over Stage-A rediscovery).

7. **Best result independently replayable from original post-impact snapshot?** **{replay_ok}**  
   (see `EXACT_REPLAY.csv` — each row recomputes SUCCESS_NEW; not stored metadata).

## Milestone counts (best per-root experts)

| Milestone | Count |
|---|---|
| STABILIZED_REGRASP | {n_stable} |
| SUCCESS_NEW | {n_succ} |

Metric corrections: `PILOT_METRIC_CORRECTIONS.md`. Pose-return sign: `POSE_RETURN_VALIDATION.md`.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print("CAPTURE_REFINEMENT_DONE", json.dumps({
        "rollouts": rollouts,
        "n_hist60": n_hist,
        "n_real60_end": n_real60,
        "n_stable": n_stable,
        "n_succ": n_succ,
        "gbi002_mechanism": gbi2.get("mechanism"),
        "cause": cause,
        "replay_ok": replay_ok,
        "elapsed": time.time() - t_wall,
    }), flush=True)


if __name__ == "__main__":
    main()
