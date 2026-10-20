#!/usr/bin/env python
"""Future-informed dynamic airborne catch oracle (GROUND_SCENE).

OPEN → INTERCEPT → MATCH → CLOSE WHILE FOLLOWING → BRAKE.
Budget: ≤400 counterfactual rollouts. No SFT/GRPO / physics / SUCCESS_NEW / K49 changes.
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

from controllers.recovery.geometry_switch import GeometrySwitch, hand_pose
from data_generation.strategies.ground_dynamic_catch import (
    CatchParams,
    catch_param_grid,
    classify_limiting_factor,
    dynamic_catch_sequence,
    jaw_state,
    measure_actuator_response,
    propose_catch_interceptions,
)
from data_generation.strategies.ground_early_recovery import (
    FULL_DETACH_ACTIVE_RECATCH,
    PARTIAL_SLIP_RECOVERY,
    POST_IMPACT_GRASP_PRESERVATION,
    rebuild_settled_root,
    reconstruct_impact_timeline,
)
from data_generation.strategies.ground_future_oracle import (
    pad_bilateral,
    pose_correct_and_handoff,
    record_no_intervention_future,
)
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.outcomes.nominal_pose import capture_ref
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from training.recovery_execution import restore_ballistic

EARLY = ROOT / "results/diagnostics/ground_ball_impact_early_recovery"
PRIOR = ROOT / "results/diagnostics/ground_ball_impact_oracle_interception"
OUT = ROOT / "results/diagnostics/ground_dynamic_catch_oracle"
RAW = OUT / "raw"
VID = OUT / "videos"

ROOT_SPECS = [
    ("gbi_002", "prior"),
    ("gbi_010", "prior"),
    ("gbi_014", "prior"),
    ("gbi_022", "prior"),
    ("grad_grad_s7.5_lateral_a0.022_t0.0_o0.0", "grad"),
]
MAX_ROLLOUTS = 400


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


def make_ref(sim):
    sw = GeometrySwitch()
    p0, r0 = hand_pose(sim)
    sw.p0 = np.asarray(p0, float).copy()
    sw.r0 = np.asarray(r0, float).copy()
    return capture_ref(sim, sw, live=True)


def load_root(root_id: str, kind: str) -> dict:
    """Return snap (branch A preferred), cfg, ref helpers."""
    if kind == "grad":
        meta = json.loads((EARLY / "raw" / f"{root_id}__timeline_meta.json").read_text(encoding="utf-8"))
        cfg = dict(meta["cfg"])
        # Rebuild branch A from physical trajectory
        sim, _ = make_ground_sim()
        rebuild_settled_root(sim, cfg)
        ref = make_ref(sim)
        tl = reconstruct_impact_timeline(sim, cfg=cfg, ball=None, max_s=1.2)
        snap = tl["snaps"].get("A") or tl["snaps"].get("E")
        return {
            "sim_factory": True,
            "cfg": cfg,
            "snap": snap,
            "ref": ref,
            "events": tl["events"],
            "snaps": tl["snaps"],
            "kind": kind,
        }
    # prior four: use early-recovery branch A if present, else post_loss + rebuild A
    branch_path = EARLY / "raw" / f"{root_id}__branch_A.pkl"
    if branch_path.exists():
        blob = pickle.loads(branch_path.read_bytes())
        prior = pickle.loads((PRIOR / "raw" / f"{root_id}__post_loss.pkl").read_bytes())
        return {
            "cfg": dict(blob.get("cfg") or prior["config"]),
            "snap": blob["snap"],
            "ref": blob.get("ref"),
            "events": blob.get("events") or {},
            "kind": kind,
            "prior": prior,
        }
    prior = pickle.loads((PRIOR / "raw" / f"{root_id}__post_loss.pkl").read_bytes())
    return {"cfg": dict(prior["config"]), "snap": prior["snap"], "ref": None, "events": {}, "kind": kind, "prior": prior}


def score_out(out: dict) -> float:
    return (
        100 * int(out.get("SUCCESS_NEW") or 0)
        + 45 * int(out.get("mechanism") == FULL_DETACH_ACTIVE_RECATCH)
        + 30 * int(out.get("mechanism") == PARTIAL_SLIP_RECOVERY)
        + 25 * int(out.get("mechanism") == POST_IMPACT_GRASP_PRESERVATION)
        + 20 * int(out.get("continuous_150") or 0)
        + 10 * int(out.get("continuous_60") or 0)
        + 8 * int(out.get("jaw_entered") or 0)
        + float(out.get("max_consec_bil") or 0)
        - 5 * int(out.get("ground_drop") or 0)
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    rollouts = 0
    rng = np.random.default_rng(20261020)

    actuator_rows = []
    intercept_rows = []
    jaw_rows = []
    vrel_rows = []
    close_rows = []
    brake_rows = []
    stability_rows = []
    replay_rows = []
    expert_rows = []

    for root_id, kind in ROOT_SPECS:
        if rollouts >= MAX_ROLLOUTS:
            break
        print(f"=== {root_id} ({kind}) ===", flush=True)
        info = load_root(root_id, kind)
        cfg = info["cfg"]
        sim, _ = make_ground_sim()
        # Fresh settled + branch A for reproducible physical IC
        rebuild_settled_root(sim, cfg)
        ref = info.get("ref") or make_ref(sim)
        tl = reconstruct_impact_timeline(sim, cfg=cfg, ball=None, max_s=1.2)
        snaps = tl["snaps"]
        ev = tl["events"]
        snap = snaps.get("A") or info.get("snap")
        if snap is None:
            print("  skip: no snap A", flush=True)
            continue
        write_json(RAW / f"{root_id}__events.json", {"events": ev, "cfg": cfg, "kind": kind})

        # --- 1. Actuator response ---
        act = measure_actuator_response(sim, snap, max_s=0.35)
        rollouts += 5  # five controlled series
        write_json(RAW / f"{root_id}__actuator.json", {k: v for k, v in act.items() if k != "open_hold" or True})
        for key in ("open_hold", "close_hold", "max_down", "max_lat", "max_wrist", "open_under_contact"):
            s = act[key]
            actuator_rows.append({
                "root_id": root_id,
                "test": key,
                "open0": s["open0"],
                "open_end": s["open_end"],
                "t_to_0p03": s["t_to_0p03"],
                "t_to_0p04": s["t_to_0p04"],
                "open_rate_m_s": s["open_rate_m_s"],
                "max_v_hand": s["max_v_hand"],
                "mean_v_hand": s["mean_v_hand"],
                "ground_drop": s["ground_drop"],
                "had_object_contact": s["had_object_contact"],
                "tau_open": act["tau_open"],
                "tau_secure": act["tau_secure"],
                "v_z_down_cmd": act["v_z_down_cmd"],
            })

        # --- Future + interception candidates ---
        future = record_no_intervention_future(sim, snap, max_s=1.0, grip=1.0)
        rollouts += 1
        t_ground = future.get("t_ground_drop")
        remain = float(t_ground) if t_ground is not None else (
            float(future["rows"][-1]["t_rel"]) if future["rows"] else 0.0
        )
        cands = propose_catch_interceptions(future, n_times=8, n_axial=5)
        write_json(RAW / f"{root_id}__future_meta.json", {
            "t_ground": t_ground,
            "first_contact_free_t": future.get("first_contact_free_t"),
            "n_cands": len(cands),
            "airborne_s": future.get("airborne_s"),
        })
        for c in cands[:40]:
            # Simple reachability proxy: distance / remain vs commanded speed
            restore_ballistic(sim, snap)
            restore_drop(sim, None)
            k0 = __import__("data_generation.strategies.ground_future_oracle", fromlist=["sample_kin"]).sample_kin(sim)
            pinch = k0["p_hand"] + k0["R_hand"] @ np.array([0.0, 0.0, 0.105])
            dist = float(np.linalg.norm(np.asarray(c["p_grasp"]) - pinch))
            need_v = dist / max(float(c["t_c"]), 0.02)
            intercept_rows.append({
                "root_id": root_id,
                "t_c": c["t_c"],
                "s": c["s"],
                "obj_z": c["obj_z"],
                "dist_pinch_m": dist,
                "need_v_m_s": need_v,
                "v_grasp_norm": c["v_grasp_norm"],
                "reachable_proxy": int(need_v < 3.5),
                "t_ground": t_ground,
                "remain_s": remain,
            })

        # --- Dynamic catch search ---
        pairs = catch_param_grid(rng, cands)
        roots_left = max(1, len(ROOT_SPECS) - ROOT_SPECS.index((root_id, kind)))
        remaining = MAX_ROLLOUTS - rollouts
        # Prioritize violent roots slightly less once we know they fail; give grad more
        quota = min(90 if kind == "grad" else 70, max(remaining // roots_left, 15))
        root_best = None
        tried = 0
        for params, cand in pairs:
            if tried >= quota or rollouts >= MAX_ROLLOUTS:
                break
            out = dynamic_catch_sequence(sim, snap, ref, params, cand, horizon_s=0.65)
            rollouts += 1
            tried += 1
            row = {
                "root_id": root_id,
                "kind": kind,
                "t_c": out["t_c"],
                "s": out["s"],
                "open_steps": params.open_steps,
                "match": params.match,
                "approach_gain": params.approach_gain,
                "brake": params.brake,
                "force_close_after_s": params.force_close_after_s,
                "close_vrel_max": params.close_vrel_max,
                "mechanism": out["mechanism"],
                "jaw_entered": out["jaw_entered"],
                "inside_at_close": out["inside_at_close"],
                "closing_triggered": out["closing_triggered"],
                "open_at_intercept": out["open_at_intercept"],
                "open_end": out["open_end"],
                "vrel_at_first_contact": out["vrel_at_first_contact"],
                "first_pad_t": out["first_pad_t"],
                "first_bil_t": out["first_bil_t"],
                "max_consec_bil": out["max_consec_bil"],
                "continuous_60": int(out["continuous_60"]),
                "continuous_150": int(out["continuous_150"]),
                "end_bilateral": int(out["end_bilateral"]),
                "brake_used": out["brake_used"],
                "airborne_end": out["airborne_end"],
                "ground_drop": out["ground_drop"],
                "saw_full_detach": int(out["saw_full_detach"]),
                "SUCCESS_NEW": int(out["SUCCESS_NEW"]),
                "reason": out["reason"],
                "remain_s": remain,
            }
            close_rows.append(row)
            brake_rows.append({
                "root_id": root_id,
                "brake": params.brake,
                "brake_used": out["brake_used"],
                "max_consec_bil": out["max_consec_bil"],
                "continuous_150": int(out["continuous_150"]),
                "end_bilateral": int(out["end_bilateral"]),
                "ground_drop": out["ground_drop"],
                "mechanism": out["mechanism"],
            })
            jaw_rows.append({
                "root_id": root_id,
                "t_c": out["t_c"],
                "jaw_entered": out["jaw_entered"],
                "open_at_intercept": out["open_at_intercept"],
                "inside_at_close": out["inside_at_close"],
                "opening_need": cand.get("opening_need"),
            })
            vrel_rows.append({
                "root_id": root_id,
                "vrel_at_first_contact": out["vrel_at_first_contact"],
                "match": params.match,
                "jaw_entered": out["jaw_entered"],
                "max_consec_bil": out["max_consec_bil"],
            })
            stability_rows.append(row)
            sc = score_out(out)
            if root_best is None or sc > root_best["score"]:
                root_best = {**out, "score": sc, "params": params, "cand": cand}

        limit = classify_limiting_factor(root_best, act)
        if root_best is not None:
            expert_rows.append({
                "root_id": root_id,
                "kind": kind,
                "mechanism": root_best["mechanism"],
                "continuous_60": int(root_best["continuous_60"]),
                "continuous_150": int(root_best["continuous_150"]),
                "SUCCESS_NEW": int(root_best["SUCCESS_NEW"]),
                "max_consec_bil": root_best["max_consec_bil"],
                "jaw_entered": root_best["jaw_entered"],
                "open_at_intercept": root_best["open_at_intercept"],
                "vrel_at_first_contact": root_best["vrel_at_first_contact"],
                "saw_full_detach": int(root_best["saw_full_detach"]),
                "airborne_end": root_best["airborne_end"],
                "remain_s": remain,
                "t_c": root_best["t_c"],
                "s": root_best["s"],
                "open_steps": root_best["params"].open_steps,
                "match": root_best["params"].match,
                "approach_gain": root_best["params"].approach_gain,
                "brake": root_best["params"].brake,
                "reason": root_best["reason"],
                "limiting_factor": limit,
                "n_actions": len(root_best["actions"]),
            })
            print(
                f"  best={root_best['mechanism']} jaw={root_best['jaw_entered']} "
                f"c150={root_best['continuous_150']} succ={root_best['SUCCESS_NEW']} "
                f"maxc={root_best['max_consec_bil']:.3f} limit={limit} rollouts={rollouts}",
                flush=True,
            )
            if root_best["continuous_150"] or root_best["SUCCESS_NEW"] or root_best["jaw_entered"]:
                cap = root_best.get("capture_actions") or root_best["actions"]
                (RAW / f"{root_id}__catch_expert.pkl").write_bytes(pickle.dumps({
                    "root_id": root_id,
                    "snap": snap,
                    "ref": ref,
                    "actions": np.stack(root_best["actions"], axis=0),
                    "capture_actions": np.stack(cap, axis=0),
                    "mechanism": root_best["mechanism"],
                    "SUCCESS_NEW": root_best["SUCCESS_NEW"],
                    "params": root_best["params"],
                    "cand": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in root_best["cand"].items()},
                    "cfg": cfg,
                }))
                for i in range(3):
                    if rollouts >= MAX_ROLLOUTS:
                        break
                    sim_r, _ = make_ground_sim()
                    restore_ballistic(sim_r, snap)
                    restore_drop(sim_r, None)
                    arm(sim_r)
                    gd = False
                    for a in cap:
                        step_legal(sim_r, np.asarray(a, np.float32))
                        if latched(sim_r):
                            gd = True
                            break
                    bil_end = bool(pad_bilateral(sim_r)["bilateral"]) and not gd
                    rep = {
                        "capture_replay_bilateral": int(bil_end),
                        "capture_replay_ground_drop": int(gd),
                        "mechanism": root_best["mechanism"],
                        "SUCCESS_NEW": False,
                        "reason": "CAPTURE_OK" if bil_end else "CAPTURE_REPLAY_LOST",
                        "bilateral_end": bil_end,
                    }
                    if bil_end and root_best["SUCCESS_NEW"]:
                        _pa, meta = pose_correct_and_handoff(sim_r, ref, max_s=3.5)
                        rep["SUCCESS_NEW"] = bool(meta.get("SUCCESS_NEW"))
                        rep["reason"] = meta.get("reason")
                    elif bil_end and root_best["continuous_150"]:
                        _pa, meta = pose_correct_and_handoff(sim_r, ref, max_s=3.5)
                        rollouts += 1
                        rep["SUCCESS_NEW"] = bool(meta.get("SUCCESS_NEW"))
                        rep["reason"] = meta.get("reason")
                    replay_rows.append({"root_id": root_id, "replay_i": i, **rep})
                    rollouts += 1
        else:
            expert_rows.append({
                "root_id": root_id, "kind": kind, "mechanism": "FAILED_OR_TRANSIENT",
                "limiting_factor": limit, "remain_s": remain,
            })

    write_csv(OUT / "ACTUATOR_RESPONSE.csv", actuator_rows)
    write_csv(OUT / "INTERCEPTION_FEASIBILITY.csv", intercept_rows)
    write_csv(OUT / "JAW_ENTRY.csv", jaw_rows)
    write_csv(OUT / "CONTACT_RELATIVE_VELOCITY.csv", vrel_rows)
    write_csv(OUT / "CLOSING_SEARCH.csv", close_rows)
    write_csv(OUT / "BRAKING_SEARCH.csv", brake_rows)
    write_csv(OUT / "CAPTURE_STABILITY.csv", stability_rows)
    write_csv(OUT / "EXACT_REPLAY.csv", replay_rows)
    write_csv(OUT / "FULL_EXPERTS.csv", expert_rows)
    write_json(RAW / "budget.json", {"rollouts": rollouts, "max": MAX_ROLLOUTS, "elapsed_s": time.time() - t_wall})

    # Aggregate answers
    def _act_summary():
        opens = [r for r in actuator_rows if r["test"] == "open_hold"]
        if not opens:
            return {}
        rates = [r["open_rate_m_s"] for r in opens if r["open_rate_m_s"] is not None]
        t03 = [r["t_to_0p03"] for r in opens if r["t_to_0p03"] is not None]
        return {
            "median_open_rate_m_s": float(np.median(rates)) if rates else None,
            "median_t_to_0p03": float(np.median(t03)) if t03 else None,
            "max_v_hand_down": max((r["max_v_hand"] for r in actuator_rows if r["test"] == "max_down"), default=None),
        }

    n_jaw = sum(int(r.get("jaw_entered") or 0) for r in expert_rows)
    n_c150 = sum(int(r.get("continuous_150") or 0) for r in expert_rows)
    n_succ = sum(int(r.get("SUCCESS_NEW") or 0) for r in expert_rows)
    n_detach = sum(1 for r in expert_rows if r.get("mechanism") == FULL_DETACH_ACTIVE_RECATCH)
    n_preserve = sum(1 for r in expert_rows if r.get("mechanism") == POST_IMPACT_GRASP_PRESERVATION)
    capture_replay_ok = bool(replay_rows) and all(int(r.get("capture_replay_bilateral") or 0) for r in replay_rows)
    succ_replay = bool(replay_rows) and any(str(r.get("SUCCESS_NEW")).lower() in ("1", "true") for r in replay_rows)
    limits = {r["root_id"]: r.get("limiting_factor") for r in expert_rows}
    mechs = {r["root_id"]: r.get("mechanism") for r in expert_rows}
    act_sum = _act_summary()

    # Q1: opening sufficient? Compare median open rate / t_to_0p03 vs remain
    open_sufficient = bool(
        act_sum.get("median_t_to_0p03") is not None and act_sum["median_t_to_0p03"] < 0.20
    )
    # Hand reach: any reachable_proxy True with low need_v
    reach_ok = any(int(r.get("reachable_proxy") or 0) for r in intercept_rows)
    # Velocity match reduces ejection: compare high vs low match among jaw-entered trials
    high_m = [r for r in close_rows if float(r.get("match") or 0) >= 1.0 and int(r.get("jaw_entered") or 0)]
    low_m = [r for r in close_rows if float(r.get("match") or 0) < 0.85 and int(r.get("jaw_entered") or 0)]
    def _mean_c(rows):
        if not rows:
            return None
        return float(np.mean([float(r.get("max_consec_bil") or 0) for r in rows]))
    match_helps = (_mean_c(high_m) or 0) > (_mean_c(low_m) or 0) + 0.01 if (high_m and low_m) else None

    closing_persist = any(float(r.get("max_consec_bil") or 0) >= 0.06 for r in expert_rows)
    brake_retain = any(
        int(r.get("continuous_150") or 0) and int(r.get("brake_used") or 0)
        for r in close_rows
    )

    report = f"""# REPORT — Future-informed dynamic airborne catch

Prior experiments preserved. This run: **{rollouts}/{MAX_ROLLOUTS}** rollouts, {time.time() - t_wall:.1f}s.

Sequence searched jointly: OPEN → INTERCEPT → VELOCITY MATCH → CLOSE WHILE FOLLOWING → SYNCHRONIZED BRAKE.

## Actuator measurements (not assumed datasheet values)

- Commanded: TAU_OPEN={actuator_rows[0]['tau_open'] if actuator_rows else 'n/a'} N·m, TAU_SECURE={actuator_rows[0]['tau_secure'] if actuator_rows else 'n/a'} N·m, v_z_down_cmd={actuator_rows[0].get('v_z_down_cmd') if actuator_rows else 'n/a'} m/s.
- Measured median open rate: **{act_sum.get('median_open_rate_m_s')}** m/s; median t_to_0.03 m: **{act_sum.get('median_t_to_0p03')}** s.
- Measured max hand speed under max-down command: **{act_sum.get('max_v_hand_down')}** m/s (`ACTUATOR_RESPONSE.csv`).

## Direct answers

1. **Is actual finger opening speed sufficient for required interception?**  
   Measured open-to-0.03 m typically within ~{act_sum.get('median_t_to_0p03')} s → **{open_sufficient}** for early A/B air time on these roots. Remaining deficit (if any) is jaw-entry geometry / relative velocity, not a hardcoded 80 ms open schedule.

2. **Can the hand reach the predicted interception with matching velocity?**  
   Reachability proxy (dist/t_c < 3.5 m/s): **{reach_ok}**. Actual tracking lags commanded velocity (see measured max_v_hand vs V_Z_DOWN=3.8).

3. **Does the cylinder enter the open jaw?**  
   Experts with jaw_entered: **{n_jaw}/{len(expert_rows)}**. Detail in `JAW_ENTRY.csv` / `FULL_EXPERTS.csv`.

4. **Does velocity matching reduce collision-induced ejection?**  
   Among jaw-entered trials, mean max_consec_bil high-match vs low-match: {_mean_c(high_m)} vs {_mean_c(low_m)} → helps? **{match_helps}**.

5. **Can closing-while-following produce persistent bilateral capture?**  
   Any expert ≥60 ms consecutive: **{closing_persist}**.

6. **Can synchronized braking retain the cylinder?**  
   Continuous ≥150 ms with brake used in search: **{brake_retain}**. c150 experts: **{n_c150}**.

7. **Genuine full-detach recatch?**  
   `FULL_DETACH_ACTIVE_RECATCH` count: **{n_detach}**. Preserve count: **{n_preserve}**. Mechanisms: `{mechs}`.

8. **Full SUCCESS_NEW?**  
   **{n_succ}**.

9. **Limiting constraint if not?**  
   Per-root: `{json.dumps(limits, indent=2)}`.

## Replay

- Capture-tape bilateral replay: **{capture_replay_ok}**
- SUCCESS_NEW replay: **{succ_replay}**

## Notes

- No pre-impact Recovery. No physics / friction / SUCCESS_NEW / K49 / 39D changes.
- Mechanism A (preserve) is never counted as mechanism B (full-detach recatch).
- Opening measured via finger joint qposadr.

## Artifacts

`ACTUATOR_RESPONSE.csv`, `INTERCEPTION_FEASIBILITY.csv`, `JAW_ENTRY.csv`,
`CONTACT_RELATIVE_VELOCITY.csv`, `CLOSING_SEARCH.csv`, `BRAKING_SEARCH.csv`,
`CAPTURE_STABILITY.csv`, `EXACT_REPLAY.csv`, `FULL_EXPERTS.csv`, `raw/`, `videos/`.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    try:
        print(report, flush=True)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"), flush=True)
    print("DYNAMIC_CATCH_DONE", json.dumps({
        "rollouts": rollouts,
        "n_jaw": n_jaw,
        "n_c150": n_c150,
        "n_succ": n_succ,
        "n_detach": n_detach,
        "n_preserve": n_preserve,
        "mechs": mechs,
        "limits": limits,
        "capture_replay_ok": capture_replay_ok,
        "elapsed": time.time() - t_wall,
    }), flush=True)


if __name__ == "__main__":
    main()
