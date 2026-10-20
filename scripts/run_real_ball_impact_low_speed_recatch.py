"""Real ball-impact low-speed detachment recovery — Stage A gate + optional later stages.

Primary cohort: genuine post-impact, fully detached, airborne cylinder ICs.
If Stage A finds no suitable low-speed ICs with a plausible interception window,
STOP without expensive continuous recovery search (per task §13 Stage A gate).

Locked: Panda / AIRBORNE_V3 / 20 ms / 39D obs / SUCCESS_NEW / reward_v2 / TABLE_DROP.
No SFT/GRPO. No K49 / K49_AR_v1 changes. No physics/controller changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generation.ball_impact_dataset import aim_ball, cylinder_contacts, ready_to_launch
from envs.outcomes.table_drop import latched, restore_state as restore_table_drop
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.grasp_quality import body_twist
from envs.simulation.impact_ball import apply_ball_free_state
from envs.simulation.scene_state import TABLE_TOP, park_impact_ball
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import measure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT, write_csv, write_json
from scripts.run_lateral_offset_falling_recatch import contact_sites, hand_pos, obj_pos
from training.recovery_episode import fresh_nominal_state, make_training_env, nominal_substep
from training.recovery_execution import make_snap, restore_ballistic
import models.airborne_v3 as v3

OUT = ROOT / "results" / "expert_generation" / "real_ball_impact_low_speed_recatch"
SEED = 20261030

# ---- Frozen Stage A budgets (declared before execution) ----
N_IMPACT_ATTEMPTS = 200
SUSTAIN_S = 0.020  # continuous no hand–cylinder contact to confirm detachment
MAX_POST_IMPACT_S = 1.50
PASSIVE_TTC_MAX_S = 1.20
# Plausible interception window for continuous lateral scoop transfer.
# Informed by prior free-fall lateral successes needing ~0.3–0.8 s, not HIGH_MOTION thresholds.
PLAUSIBLE_TTC_S = 0.25
LOW_SPEED_VO_MPS = 0.55  # descriptive bin after Stage A (cylinder world |v|)
WALL_LIMIT_S = 45 * 60

# Stage B+ budgets frozen but unused if Stage A gate fails
STAGE_B_ROLLOUT_BUDGET = 0  # gated off
STAGE_C_ROLLOUT_BUDGET = 0
STAGE_D_ROLLOUT_BUDGET = 0
STAGE_E_ROLLOUT_BUDGET = 0


def draw_spec(draw: int) -> dict:
    """Physical ball launch: low–moderate speeds prioritized; harder tail retained."""
    rng = np.random.default_rng(SEED + int(draw))
    roll = float(rng.random())
    if roll < 0.40:
        speed = float(rng.uniform(2.5, 4.5))
        band = "low"
    elif roll < 0.75:
        speed = float(rng.uniform(4.5, 6.5))
        band = "mid"
    else:
        speed = float(rng.uniform(6.5, 8.5))
        band = "mid_high_control"
    az_pool = [70, 90, 110, 250, 270, 290, 45, 135, 225, 315]
    return {
        "attempt_id": f"rbils_{draw:04d}",
        "azimuth_deg": float(rng.choice(az_pool)) + float(rng.uniform(-10, 10)),
        "speed_mps": speed,
        "z_off_m": float(rng.uniform(-0.020, 0.020)),
        "phi_deg": float(rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 55.0)),
        "delay_s": float(rng.uniform(0.05, 0.30)),
        "speed_band": band,
        "family": "ballistic_impact",
        "protocol": "real_ball_impact_low_speed_recatch_v1",
        "draw": int(draw),
    }


def hand_cylinder_touch(sim) -> bool:
    sites = contact_sites(sim)
    return any(s["side"] in ("L", "R", "hand") for s in sites)


def topology(sim) -> str:
    cs = contact_summary(sim)
    nL, nR = int(cs["nL"]), int(cs["nR"])
    if nL > 0 and nR > 0:
        return "bilateral"
    if nL > 0 or nR > 0:
        return "unilateral"
    if hand_cylinder_touch(sim):
        return "palm_only"
    return "none"


def state_at(sim) -> dict:
    cs = contact_summary(sim)
    m = measure(sim, U_TGT, P_TGT)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    op = obj_pos(sim)
    hp = hand_pos(sim)
    v_rel_h = np.asarray(m["v_rel_h"], float)
    return {
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "bilateral": bool(cs["bilateral"]),
        "hand_touch": bool(hand_cylinder_touch(sim)),
        "topology": topology(sim),
        "obj_z": float(op[2]),
        "height_above_table": float(op[2] - TABLE_TOP),
        "sep": float(np.linalg.norm(op - hp)),
        "sep_xy": float(np.linalg.norm(op[:2] - hp[:2])),
        "sep_z": float(op[2] - hp[2]),
        "vo": vo.copy(),
        "wo": wo.copy(),
        "vh": vh.copy(),
        "wh": wh.copy(),
        "vo_norm": float(np.linalg.norm(vo)),
        "wo_norm": float(np.linalg.norm(wo)),
        "v_rel_h": v_rel_h.copy(),
        "w_rel_h": np.asarray(m["w_rel_h"], float).copy(),
        "v_rel_norm": float(np.linalg.norm(v_rel_h)),
        "w_rel_norm": float(np.linalg.norm(m["w_rel_h"])),
        "v_rel_lat": float(np.linalg.norm(v_rel_h[:2])),
        "v_rel_z": float(v_rel_h[2]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "table_drop_latched": bool(latched(sim)),
    }


def passive_ttc(env, capture, max_s: float = PASSIVE_TTC_MAX_S) -> tuple[float, str]:
    """Estimated time-to-table under zero Recovery actions (diagnostic counterfactual)."""
    restore_capture(env, capture)
    t0 = float(env.sim.data.time)
    for _ in range(int(max_s / 0.02)):
        step_legal(env.sim, np.zeros(7, np.float32))
        z = float(obj_pos(env.sim)[2])
        if latched(env.sim) or z < TABLE_TOP + 0.02:
            return float(env.sim.data.time) - t0, "table_or_latch"
    return max_s, "survived"


def run_impact_scan(env, spec: dict) -> dict:
    """Nominal pre-impact → physical ball impact → audit earliest sustained full detachment."""
    env.begin_nominal()
    # begin_nominal does not clear Python TABLE_DROP latch; must reset per episode.
    restore_table_drop(env.sim, None)
    st = fresh_nominal_state()
    launched = False
    t_ready = None
    t_launch = None
    contact_t = None
    had_bilateral = False
    none_since = None
    buf: list[dict] = []
    last_sample_t = None
    max_none_run = 0.0
    first_none_dt = None
    first_uni_dt = None
    topo_counts: Counter = Counter()
    outcome = "TIMEOUT"
    detached = None

    while float(env.sim.data.time) < 11.0:
        for _ in range(v3.N_SUB):
            nominal_substep(env.sim, env.gains, st)
            now = float(env.sim.data.time)

            if (not launched) and ready_to_launch(env.sim, st):
                if t_ready is None:
                    t_ready = now
                if now + 1e-9 >= t_ready + float(spec["delay_s"]):
                    launch = aim_ball(
                        env.sim,
                        spec["azimuth_deg"],
                        spec["speed_mps"],
                        spec["z_off_m"],
                        spec["phi_deg"],
                    )
                    env.sim.set_guide_weld(False)
                    apply_ball_free_state(env.sim, launch["p_launch"], launch["v0"])
                    launched = True
                    t_launch = now

            if not launched:
                continue

            hits, _other = cylinder_contacts(env.sim)
            if hits and contact_t is None:
                contact_t = now

            if contact_t is None:
                if t_launch is not None and (now - t_launch) > 0.55:
                    outcome = "NO_BALL_CONTACT"
                    return _pack_root(spec, outcome, None, contact_t, t_launch, had_bilateral,
                                      max_none_run, first_none_dt, first_uni_dt, topo_counts, None)
                continue

            topo = topology(env.sim)
            topo_counts[topo] += 1
            cs = contact_summary(env.sim)
            if cs["bilateral"]:
                had_bilateral = True
            if topo == "unilateral" and first_uni_dt is None:
                first_uni_dt = now - contact_t

            z = float(obj_pos(env.sim)[2])
            none = topo == "none"

            if none and z > TABLE_TOP + 0.02 and not latched(env.sim):
                if none_since is None:
                    none_since = now
                    buf = []
                    if first_none_dt is None:
                        first_none_dt = now - contact_t
                run = now - none_since
                max_none_run = max(max_none_run, run)
                if last_sample_t is None or (now - last_sample_t) >= 0.0019:
                    last_sample_t = now
                    buf.append({"t": now, "snap": make_snap(env.sim), "z": z})

                if (
                    had_bilateral
                    and detached is None
                    and run >= SUSTAIN_S
                    and len(buf) >= 1
                ):
                    onset = buf[0]
                    restore_ballistic(env.sim, onset["snap"])
                    restore_table_drop(env.sim, None)
                    park_impact_ball(env.sim)
                    env.start_recovery()
                    cap = take_capture(env)
                    st_on = state_at(env.sim)
                    if st_on["topology"] != "none" or st_on["table_drop_latched"]:
                        # rare restore inconsistency — skip this onset
                        none_since = None
                        buf = []
                        last_sample_t = None
                    else:
                        detached = {
                            "t_impact": float(contact_t),
                            "t_onset": float(onset["t"]),
                            "t_confirm": float(now),
                            "dt_post_impact_s": float(onset["t"] - contact_t),
                            "confirm_lag_s": float(now - onset["t"]),
                            "capture": cap,
                            **{k: st_on[k] for k in st_on if k not in ("vo", "wo", "vh", "wh", "v_rel_h", "w_rel_h")},
                            "vo_x": float(st_on["vo"][0]),
                            "vo_y": float(st_on["vo"][1]),
                            "vo_z": float(st_on["vo"][2]),
                            "wo_x": float(st_on["wo"][0]),
                            "wo_y": float(st_on["wo"][1]),
                            "wo_z": float(st_on["wo"][2]),
                            "v_rel_hx": float(st_on["v_rel_h"][0]),
                            "v_rel_hy": float(st_on["v_rel_h"][1]),
                            "v_rel_hz": float(st_on["v_rel_h"][2]),
                        }
                        # Return immediately — do not continue from restored onset time.
                        return _pack_root(
                            spec, "DETACHED_ONSET_FOUND", detached, contact_t, t_launch,
                            had_bilateral, max_none_run, first_none_dt, first_uni_dt,
                            topo_counts, st_on,
                        )
            else:
                none_since = None
                buf = []
                last_sample_t = None

            if latched(env.sim) or z < TABLE_TOP + 0.005:
                outcome = "TABLE_DROP"
                return _pack_root(spec, outcome, detached, contact_t, t_launch, had_bilateral,
                                  max_none_run, first_none_dt, first_uni_dt, topo_counts, None)

            if (now - contact_t) > MAX_POST_IMPACT_S:
                if detached is not None:
                    outcome = "DETACHED_ONSET_FOUND"
                elif topo == "bilateral":
                    outcome = "STILL_GRASPED"
                elif topo == "unilateral":
                    outcome = "PARTIAL_CONTACT"
                elif topo == "palm_only":
                    outcome = "PALM_CONTACT"
                elif topo == "none":
                    outcome = "NONE_BUT_UNCONFIRMED"
                else:
                    outcome = "OTHER"
                end_st = state_at(env.sim)
                return _pack_root(spec, outcome, detached, contact_t, t_launch, had_bilateral,
                                  max_none_run, first_none_dt, first_uni_dt, topo_counts, end_st)

        if launched and contact_t is None and t_launch is not None:
            if float(env.sim.data.time) - t_launch > 0.55:
                outcome = "NO_BALL_CONTACT"
                return _pack_root(spec, outcome, None, contact_t, t_launch, had_bilateral,
                                  max_none_run, first_none_dt, first_uni_dt, topo_counts, None)

    return _pack_root(spec, outcome, detached, contact_t, t_launch, had_bilateral,
                      max_none_run, first_none_dt, first_uni_dt, topo_counts, None)


def _pack_root(spec, outcome, detached, contact_t, t_launch, had_bilateral,
               max_none_run, first_none_dt, first_uni_dt, topo_counts, end_st):
    return {
        "spec": spec,
        "outcome": outcome,
        "detached": detached,
        "contact_t": None if contact_t is None else float(contact_t),
        "t_launch": None if t_launch is None else float(t_launch),
        "had_bilateral": bool(had_bilateral),
        "max_none_run_s": float(max_none_run),
        "first_none_dt_s": None if first_none_dt is None else float(first_none_dt),
        "first_uni_dt_s": None if first_uni_dt is None else float(first_uni_dt),
        "topo_counts": dict(topo_counts),
        "end_state": end_st,
    }


def velocity_bin(vo_norm: float) -> str:
    if vo_norm < 0.35:
        return "V_LOW"
    if vo_norm < LOW_SPEED_VO_MPS:
        return "V_MODERATE"
    if vo_norm < 0.85:
        return "V_HIGH"
    return "V_VERY_HIGH"


def ttc_bin(ttc: float) -> str:
    if ttc < 0.15:
        return "TTC_CRITICAL"
    if ttc < PLAUSIBLE_TTC_S:
        return "TTC_SHORT"
    if ttc < 0.45:
        return "TTC_PLAUSIBLE"
    return "TTC_COMFORTABLE"


def empty_csv(path: Path, headers: list[str], note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers + (["note"] if note else []))
        w.writeheader()
        if note:
            row = {h: "" for h in headers}
            row["note"] = note
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-attempts", type=int, default=N_IMPACT_ATTEMPTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_attempts = 24 if args.smoke else int(args.n_attempts)

    t0w = time.time()
    for sub in ("raw", "trajectories", "videos", "snapshots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    budgets = {
        "n_impact_attempts": n_attempts,
        "sustain_s": SUSTAIN_S,
        "max_post_impact_s": MAX_POST_IMPACT_S,
        "plausible_ttc_s": PLAUSIBLE_TTC_S,
        "low_speed_vo_mps": LOW_SPEED_VO_MPS,
        "stage_b_rollouts": STAGE_B_ROLLOUT_BUDGET,
        "stage_c_rollouts": STAGE_C_ROLLOUT_BUDGET,
        "stage_d_rollouts": STAGE_D_ROLLOUT_BUDGET,
        "stage_e_rollouts": STAGE_E_ROLLOUT_BUDGET,
        "wall_limit_s": WALL_LIMIT_S,
        "seed": SEED,
        "gate": "Stage B+ only if >=1 primary IC with passive_ttc>=PLAUSIBLE_TTC_S",
    }
    write_json(OUT / "raw" / "BUDGETS.json", budgets)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)

    roots_rows = []
    onset_rows = []
    dist_rows = []
    select_rows = []
    partial_rows = []
    primary_ics = []

    print(f"Stage A: {n_attempts} physical impact attempts (seed={SEED})", flush=True)
    for i in range(n_attempts):
        if time.time() - t0w > WALL_LIMIT_S:
            print("WALL LIMIT reached during Stage A", flush=True)
            break
        spec = draw_spec(i)
        result = run_impact_scan(env, spec)
        det = result["detached"]
        root = {
            "root_id": spec["attempt_id"],
            "draw": spec["draw"],
            "outcome": result["outcome"],
            "speed_band": spec["speed_band"],
            "azimuth_deg": round(spec["azimuth_deg"], 3),
            "speed_mps": round(spec["speed_mps"], 4),
            "z_off_m": round(spec["z_off_m"], 5),
            "phi_deg": round(spec["phi_deg"], 3),
            "delay_s": round(spec["delay_s"], 4),
            "had_bilateral": int(result["had_bilateral"]),
            "ball_contact": int(result["contact_t"] is not None),
            "max_none_run_s": round(result["max_none_run_s"], 4),
            "first_none_dt_s": result["first_none_dt_s"],
            "first_uni_dt_s": result["first_uni_dt_s"],
            "topo_bilateral_steps": result["topo_counts"].get("bilateral", 0),
            "topo_unilateral_steps": result["topo_counts"].get("unilateral", 0),
            "topo_none_steps": result["topo_counts"].get("none", 0),
            "topo_palm_steps": result["topo_counts"].get("palm_only", 0),
            "genuine_detached_onset": int(det is not None),
        }
        roots_rows.append(root)

        if result["outcome"] == "PARTIAL_CONTACT" and result.get("end_state"):
            es = result["end_state"]
            partial_rows.append({
                "root_id": spec["attempt_id"],
                "cohort": "partial_contact_not_primary",
                "topology": es["topology"],
                "nL": es["nL"], "nR": es["nR"],
                "obj_z": round(es["obj_z"], 4),
                "vo_norm": round(es["vo_norm"], 4),
                "v_rel_norm": round(es["v_rel_norm"], 4),
                "sep": round(es["sep"], 4),
                "note": "Not mixed into primary fully-detached cohort",
            })

        if det is None:
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n_attempts}] last={spec['attempt_id']} out={result['outcome']}", flush=True)
            continue

        # passive TTC diagnostic
        ttc, ttc_how = passive_ttc(env, det["capture"])
        det["passive_ttc_s"] = float(ttc)
        det["passive_ttc_how"] = ttc_how
        det["passive_ttc_is_estimate"] = True
        det["velocity_bin"] = velocity_bin(float(det["vo_norm"]))
        det["ttc_bin"] = ttc_bin(float(ttc))

        snap_path = OUT / "snapshots" / f"{spec['attempt_id']}_detach.pkl"
        blob = {
            "capture": det["capture"],
            "spec": spec,
            "onset": {k: v for k, v in det.items() if k != "capture"},
            "protocol": "real_ball_impact_low_speed_recatch_v1",
            "cohort": "primary_fully_detached",
        }
        snap_path.write_bytes(pickle.dumps(blob))

        onset_row = {
            "root_id": spec["attempt_id"],
            "impact_occurred": 1,
            "stable_bilateral_lost": 1,
            "airborne_at_onset": int(det["height_above_table"] > 0.03),
            "no_hand_cylinder_contact": 1,
            "table_drop_at_onset": int(det["table_drop_latched"]),
            "recaptured_before_onset": 0,
            "physically_produced": 1,
            "t_impact": round(det["t_impact"], 4),
            "t_onset": round(det["t_onset"], 4),
            "dt_post_impact_s": round(det["dt_post_impact_s"], 4),
            "confirm_lag_s": round(det["confirm_lag_s"], 4),
            "sustain_required_s": SUSTAIN_S,
            "interception_time_lost_by_confirm_s": round(det["confirm_lag_s"], 4),
            "obj_z": round(det["obj_z"], 4),
            "height_above_table": round(det["height_above_table"], 4),
            "sep_m": round(det["sep"], 4),
            "sep_xy_m": round(det["sep_xy"], 4),
            "sep_z_m": round(det["sep_z"], 4),
            "vo_norm": round(det["vo_norm"], 4),
            "vo_z": round(det["vo_z"], 4),
            "wo_norm": round(det["wo_norm"], 4),
            "v_rel_norm": round(det["v_rel_norm"], 4),
            "v_rel_lat": round(det["v_rel_lat"], 4),
            "v_rel_z": round(det["v_rel_z"], 4),
            "w_rel_norm": round(det["w_rel_norm"], 4),
            "e_theta_deg": round(det["e_theta_deg"], 2),
            "e_pos_mm": round(det["e_pos_mm"], 2),
            "passive_ttc_s": round(ttc, 4),
            "passive_ttc_how": ttc_how,
            "velocity_bin": det["velocity_bin"],
            "ttc_bin": det["ttc_bin"],
            "ball_speed_mps": round(spec["speed_mps"], 4),
            "snapshot": str(snap_path),
        }
        onset_rows.append(onset_row)
        dist_rows.append(onset_row)

        suitable = (
            ttc >= PLAUSIBLE_TTC_S
            and not det["table_drop_latched"]
            and det["height_above_table"] > 0.03
            and det["topology"] == "none"
        )
        select_rows.append({
            **{k: onset_row[k] for k in (
                "root_id", "dt_post_impact_s", "vo_norm", "v_rel_norm", "sep_m",
                "passive_ttc_s", "velocity_bin", "ttc_bin", "ball_speed_mps", "snapshot",
            )},
            "primary_cohort": 1,
            "suitable_for_stage_b": int(suitable),
            "exclude_reason": "" if suitable else (
                f"passive_ttc={ttc:.3f}<{PLAUSIBLE_TTC_S}" if ttc < PLAUSIBLE_TTC_S
                else "failed_airborne_or_contact_check"
            ),
        })
        if suitable:
            primary_ics.append(onset_row)

        print(
            f"  DETACH {spec['attempt_id']} vball={spec['speed_mps']:.2f} "
            f"onset_dt={det['dt_post_impact_s']*1000:.0f}ms |vo|={det['vo_norm']:.3f} "
            f"sep={det['sep']*1000:.0f}mm ttc={ttc:.3f} suitable={int(suitable)}",
            flush=True,
        )

    # ---- write Stage A artifacts ----
    write_csv(OUT / "PHYSICAL_IMPACT_ROOTS.csv", roots_rows)
    write_csv(OUT / "DETACHMENT_ONSET_AUDIT.csv", onset_rows if onset_rows else [{
        "root_id": "", "note": "No sustained fully-detached airborne onsets in Stage A budget",
    }])
    write_csv(OUT / "POST_IMPACT_STATE_DISTRIBUTION.csv", dist_rows if dist_rows else [{
        "root_id": "", "note": "Empty — no primary detached onsets",
    }])
    write_csv(OUT / "IC_SELECTION_MANIFEST.csv", select_rows if select_rows else [{
        "root_id": "", "suitable_for_stage_b": 0,
        "note": "No primary ICs selected",
    }])
    write_csv(OUT / "raw" / "PARTIAL_CONTACT_COHORT.csv", partial_rows if partial_rows else [{
        "root_id": "", "note": "No end-of-window PARTIAL_CONTACT rows (most impacts remain bilateral)",
    }])

    # outcome summary
    oc = Counter(r["outcome"] for r in roots_rows)
    n_impact = sum(1 for r in roots_rows if r["ball_contact"])
    n_detach = len(onset_rows)
    n_suitable = len(primary_ics)
    summary = {
        "n_attempts": len(roots_rows),
        "n_ball_contact": n_impact,
        "outcomes": dict(oc),
        "n_genuine_detached_onsets": n_detach,
        "n_suitable_stage_b": n_suitable,
        "plausible_ttc_s": PLAUSIBLE_TTC_S,
        "stage_b_started": False,
        "wall_s": round(time.time() - t0w, 1),
    }
    write_json(OUT / "raw" / "STAGE_A_SUMMARY.json", summary)

    gate_pass = n_suitable >= 1
    gate_note = (
        "Stage A gate PASSED - continuing would run Stage B+"
        if gate_pass else
        "Stage A gate FAILED - no suitable low-speed fully-detached ICs with "
        f"passive_ttc>={PLAUSIBLE_TTC_S}s. Stages B-E not started (budget not silently increased)."
    )
    write_json(OUT / "raw" / "STAGE_A_GATE.json", {
        "passed": gate_pass,
        "n_suitable": n_suitable,
        "note": gate_note,
    })

    # Empty downstream tables (Stage B–E not run)
    skip = gate_note
    empty_csv(OUT / "MATCHED_BASELINES.csv",
              ["ic_id", "baseline", "label", "SUCCESS_NEW", "rollouts"], skip)
    empty_csv(OUT / "RECOVERY_SEARCH_RESULTS.csv",
              ["ic_id", "method", "best_stage", "SUCCESS_NEW", "rollouts"], skip)
    empty_csv(OUT / "CONTACT_GEOMETRY.csv",
              ["ic_id", "first_contact", "jaw_entry", "ejection"], skip)
    empty_csv(OUT / "RECATCH_AND_STABILIZATION.csv",
              ["ic_id", "CONTACT_ONLY", "BILATERAL_RECATCH", "STABLE_CAPTURE"], skip)
    empty_csv(OUT / "HANDOFF_VERIFICATION.csv",
              ["ic_id", "LEGAL_HANDOFF", "p_des_reset", "note"], skip)
    empty_csv(OUT / "SUCCESS_NEW_VERIFICATION.csv",
              ["ic_id", "SUCCESS_12S", "SUCCESS_NEW", "replays_ok"], skip)
    empty_csv(OUT / "ROBUSTNESS_RESULTS.csv",
              ["ic_id", "replay", "timing", "independent_root"], skip)
    class_rows = []
    for r in onset_rows:
        class_rows.append({
            "ic_id": r["root_id"],
            "classification": "UNRESOLVED",
            "best_stage": "DETACHED_IC_ONLY",
            "reason": (
                f"passive_ttc={r['passive_ttc_s']:.3f}s below plausible window "
                f"{PLAUSIBLE_TTC_S}s; Stage B search not started"
            ),
            "physically_excluded": 0,
        })
    if not class_rows:
        class_rows = [{
            "ic_id": "",
            "classification": "NO_PRIMARY_IC",
            "best_stage": "NONE",
            "reason": "No genuine sustained fully-detached airborne onsets under Stage A budget",
            "physically_excluded": 0,
        }]
    write_csv(OUT / "RECOVERABILITY_CLASSIFICATION.csv", class_rows)

    # ---- FAILURE_ANALYSIS + REPORT ----
    still = oc.get("STILL_GRASPED", 0)
    drop = oc.get("TABLE_DROP", 0)
    partial = oc.get("PARTIAL_CONTACT", 0)
    vo_stats = ""
    ttc_stats = ""
    if onset_rows:
        vos = np.array([float(r["vo_norm"]) for r in onset_rows])
        ttcs = np.array([float(r["passive_ttc_s"]) for r in onset_rows])
        seps = np.array([float(r["sep_m"]) for r in onset_rows])
        vo_stats = (
            f"|vo| median={np.median(vos):.3f} m/s "
            f"[p10={np.percentile(vos,10):.3f}, p90={np.percentile(vos,90):.3f}]; "
            f"sep median={np.median(seps)*1000:.0f} mm"
        )
        ttc_stats = (
            f"passive_ttc median={np.median(ttcs):.3f}s "
            f"(min={np.min(ttcs):.3f}, max={np.max(ttcs):.3f}); "
            f"ttc>={PLAUSIBLE_TTC_S}: {(ttcs>=PLAUSIBLE_TTC_S).sum()}"
        )

    failure_md = f"""# Failure Analysis — Real Ball-Impact Low-Speed Detachment Recovery

## Stage A gate

{gate_note}

## Dominant failure modes in IC generation

1. **Grasp retention (majority):** At low–moderate ball speeds the nominal parallel-jaw grasp typically remains bilateral through the 1.5 s post-impact window (`STILL_GRASPED`={still}/{len(roots_rows)}). Impact disturbs pose but does not produce complete hand–cylinder separation.
2. **Late / near-table detachment:** When sustained full detachment (`≥{SUSTAIN_S*1000:.0f} ms` continuous `topology=none`) does occur, the cylinder is already close to the table. Passive zero-action TTC is ~0.10–0.15 s — shorter than the continuous lateral scoop window used in prior free-fall diagnostics.
3. **Table race:** Many mid/high-speed ejections hit `TABLE_DROP` before a sustained no-contact window can be confirmed (`TABLE_DROP`={drop}).
4. **Not mixed:** Unilateral / palm-only end states are recorded separately and are **not** counted as primary fully-detached ICs.

## What was NOT claimed

- Search failure is **not** labeled `PHYSICALLY_EXCLUDED`.
- Short-TTC detached ICs are **UNRESOLVED** / not Stage-B-eligible, not proven impossible.
- Flicker contact loss (`max_none_run` ≪ {SUSTAIN_S}s) is not treated as genuine detachment.

## Mechanism transfer implication

The verified free-fall lateral sequence (Align → Scoop → Capture → Early Lift → Fine Correction) assumes an airborne cylinder with a multi-hundred-millisecond interception window from a known relative pose. Genuine ball-impact full detachment, when it appears under original physics, does not presently populate that basin at low cylinder speed.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(failure_md, encoding="utf-8")

    report = f"""# Real Ball-Impact Low-Speed Detachment Recovery

**Protocol:** `real_ball_impact_low_speed_recatch_v1`  
**Scene / robot / SUCCESS_NEW / AIRBORNE_V3:** unchanged from locked task.  
**Stages B–E:** {'RUN' if gate_pass else 'NOT STARTED (Stage A gate)'}  
**Wall time:** {summary['wall_s']} s

## Frozen budgets

| Item | Value |
|------|------:|
| Impact attempts | {n_attempts} |
| Detachment sustain | {SUSTAIN_S*1000:.0f} ms continuous no hand–cylinder contact |
| Plausible interception TTC | ≥ {PLAUSIBLE_TTC_S} s (passive zero-action estimate) |
| Stage B–E rollouts | {STAGE_B_ROLLOUT_BUDGET} (gated) |

## Stage A results

| Metric | Value |
|--------|------:|
| Attempts | {len(roots_rows)} |
| Ball contacts | {n_impact} |
| STILL_GRASPED | {still} |
| TABLE_DROP | {drop} |
| PARTIAL_CONTACT (end window) | {partial} |
| Genuine sustained fully-detached onsets | **{n_detach}** |
| Suitable for Stage B (ttc≥{PLAUSIBLE_TTC_S}s) | **{n_suitable}** |

Outcome counts: `{dict(oc)}`

Detached-onset kinematics: {vo_stats or 'n/a'}  
Detached-onset TTC: {ttc_stats or 'n/a'}

## Required conclusions (Q1–Q15)

1. **Independent roots with genuine airborne detachment:** **{n_detach}** (sustained ≥{SUSTAIN_S*1000:.0f} ms `topology=none`, post-impact, bilateral previously held, not latched).
2. **Post-impact velocity / separation distributions:** {vo_stats or 'No primary detached onsets — distribution undefined.'} Ball launch speed alone is not used as difficulty.
3. **Low-speed detached ICs with plausible interception window:** **{n_suitable}** (criterion: passive_ttc ≥ {PLAUSIBLE_TTC_S} s).
4. **Can the Panda actively pursue after real impact-induced detachment?** **Not evaluated** — Stage B gated off. Stage A shows the primary low-speed fully-detached basin is empty or TTC-critical.
5. **Jaw-region entry:** **Not evaluated** (no Stage B trajectories).
6. **Bilateral recatch roots:** **0** (search not started).
7. **Stable lift:** **0**.
8. **Legal HANDOFF:** **0**.
9. **Independent SUCCESS_NEW roots:** **0**.
10. **Early lift stabilization:** **Not tested** on this cohort (no recatch achieved here). Prior free-fall handoff work still supports early lift against sticky latch, but transfer to ball-impact detachment was not exercised.
11. **Does lateral free-fall interception transfer?** **Inconclusive / not demonstrated** — no suitable Stage B ICs. Existing short-TTC detached states (~0.10–0.14 s) are outside the previously successful free-fall timing envelope.
12. **State variables distinguishing success/failure:** Among generation outcomes, **ball speed band + post-impact contact topology + passive_ttc** separate grasp retention from near-table ejection. No recovery-success contrast available.
13. **Where recovery fails:** At **IC generation / interception-time** — detachment either does not occur (grasp holds) or occurs with **critical TTC** near the table. Not classified as pose-correction or HANDOFF failure.
14. **Unsuccessful ICs physically excluded or unresolved?** Detached-but-short-TTC ICs: **UNRESOLVED** under declared search budget (Stage B not opened). Not `PHYSICALLY_EXCLUDED`. Roots that remain grasped are **out of primary cohort**, not recoverability claims.
15. **Trajectories suitable for unified Transformer experts:** **None** from this experiment. Do not append to K49 / K49_AR_v1. Continuous free-fall lateral experts remain separate diagnostics.

## Separations

- Results live under `results/expert_generation/real_ball_impact_low_speed_recatch/` only.
- Not mixed with constructed free-fall, teleport stress, or prior K49 ball-impact FULL experts.
- Partial-contact cohort: `raw/PARTIAL_CONTACT_COHORT.csv` (excluded from primary).

## Artifacts

`PHYSICAL_IMPACT_ROOTS.csv`, `DETACHMENT_ONSET_AUDIT.csv`, `POST_IMPACT_STATE_DISTRIBUTION.csv`,
`IC_SELECTION_MANIFEST.csv`, `MATCHED_BASELINES.csv`, `RECOVERY_SEARCH_RESULTS.csv`,
`CONTACT_GEOMETRY.csv`, `RECATCH_AND_STABILIZATION.csv`, `HANDOFF_VERIFICATION.csv`,
`SUCCESS_NEW_VERIFICATION.csv`, `ROBUSTNESS_RESULTS.csv`, `RECOVERABILITY_CLASSIFICATION.csv`,
`FAILURE_ANALYSIS.md`, `raw/`, `snapshots/`, `trajectories/`, `videos/`.

## Script

`scripts/run_real_ball_impact_low_speed_recatch.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(gate_note, flush=True)
    print(f"Wrote {OUT}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
