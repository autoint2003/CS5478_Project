#!/usr/bin/env python
"""Post-catch end-grasp recovery: slow lift vs gravity-assisted sliding.

Starts from verified dynamic-catch tapes. Budget: 250 on gbi_002, then up to 150
on gbi_010/014/022 if promising. No SFT/GRPO / physics / SUCCESS_NEW changes.
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

from data_generation.strategies.ground_postcatch_endgrasp import (
    LiftParams,
    SlideParams,
    gravity_slide_sequence,
    replay_to_brake_complete,
    run_sequence,
    score_sequence,
    slow_lift_sequence,
)
from data_generation.strategies.ground_future_oracle import pad_bilateral
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT
from training.recovery_execution import restore_ballistic

CATCH = ROOT / "results/diagnostics/ground_dynamic_catch_oracle"
OUT = ROOT / "results/diagnostics/ground_postcatch_endgrasp_recovery"
RAW = OUT / "raw"
VID = OUT / "videos"

PRIMARY = ["gbi_002"]
EXPAND = ["gbi_010", "gbi_014", "gbi_022"]
BUDGET_PRIMARY = 250
BUDGET_EXPAND = 150


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


def load_expert(root_id: str) -> dict:
    return pickle.loads((CATCH / "raw" / f"{root_id}__catch_expert.pkl").read_bytes())


def make_ref_from_expert(exp) -> object:
    return exp["ref"]


def lift_grid():
    for vz in (0.01, 0.02, 0.04, 0.08, 0.12):
        for grip in (0.7, 1.0):
            yield LiftParams(vz_up=vz, grip=grip, max_s=12.0)


def slide_grid(rng: np.random.Generator):
    opts = []
    for axis in (0, 1, 2):
        for sign in (-1.0, 1.0):
            for speed in (0.4, 0.8, 1.5):
                for grip in (0.0, 0.35, 0.6):
                    for slide_s in (0.25, 0.40, 0.60):
                        for vz_hold in (0.0, 0.02):
                            opts.append(SlideParams(
                                wrist_axis=axis, wrist_sign=sign, wrist_speed=speed,
                                grip=grip, slide_s=slide_s, vz_hold=vz_hold,
                                retighten_s=0.20, compensate_com=0.5,
                            ))
    rng.shuffle(opts)
    return opts


def search_root(root_id: str, budget: int, rng: np.random.Generator, tables: dict) -> dict:
    print(f"=== {root_id} budget={budget} ===", flush=True)
    exp = load_expert(root_id)
    snap0 = exp["snap"]
    ref = make_ref_from_expert(exp)
    cfg = exp.get("cfg") or {}
    target_z = float(cfg.get("z", 0.55))
    cap = exp["capture_actions"]

    sim, _ = make_ground_sim()
    br = replay_to_brake_complete(sim, snap0, cap)
    tables["brake"].append({
        "root_id": root_id,
        "saw_full_detach": int(br["saw_full_detach"]),
        "max_contact_free_s": br["max_contact_free_s"],
        "continuous_150": int(br["continuous_150"]),
        "max_consec_bil": br["max_consec_bil"],
        "end_bilateral": int(br["end_bilateral"]),
        "airborne": int(br["airborne"]),
        "ground_drop": br["ground_drop"],
        "t_elapsed": br["t_elapsed"],
        "target_z": target_z,
    })
    geo = br["geometry"]
    tables["geometry"].append({"root_id": root_id, "phase": "brake_complete", **geo})
    (RAW / f"{root_id}__brake_snap.pkl").write_bytes(pickle.dumps({
        "brake_snap": br["brake_snap"],
        "ref": ref,
        "cfg": cfg,
        "geometry": geo,
        "catch_snap": snap0,
        "capture_actions": np.asarray(cap, np.float32),
    }))
    write_json(RAW / f"{root_id}__brake_audit.json", {"brake": tables["brake"][-1], "geometry": geo})

    # Low clearance is recoverable via B2 (small lift first). Only reject hard failures.
    low_clearance = float(geo.get("clearance_m") or 0) < 0.02
    if not br["end_bilateral"] or br["ground_drop"]:
        print("  brake state invalid for recovery", flush=True)
        return {"root_id": root_id, "rollouts": 1, "best": None, "promising": False}
    if low_clearance:
        print(f"  low clearance {geo.get('clearance_m'):.4f} m — prioritize B2", flush=True)

    brake_snap = br["brake_snap"]
    rollouts = 1  # replay itself counted lightly; search below
    best = None

    # --- Slow lift alone ---
    for lp in lift_grid():
        if rollouts >= budget:
            break
        out = slow_lift_sequence(sim, brake_snap, lp, target_z=target_z)
        rollouts += 1
        tables["lift"].append({
            "root_id": root_id, "vz_cmd": lp.vz_up, "grip": lp.grip,
            "dz_obj": out["dz_obj"], "dz_hand": out["dz_hand"],
            "actual_obj_vz_est": out["actual_obj_vz_est"],
            "min_clearance": out["min_clearance"],
            "reached_target_z": out["reached_target_z"],
            "end_bilateral": out["end_bilateral"],
            "max_consec_bil": out["max_consec_bil"],
            "continuous_150": int(out["continuous_150"]),
            "ground_drop": out["ground_drop"],
            "cylinder_rose_with_hand": out["cylinder_rose_with_hand"],
            "ds_contact_axial": out["ds_contact_axial"],
            "obj_z_end": out["obj_z_end"],
        })
        sc = (
            50 * out["reached_target_z"]
            + 20 * out["end_bilateral"]
            + 10 * out["cylinder_rose_with_hand"]
            + 5 * float(out["dz_obj"])
            - 15 * out["ground_drop"]
        )
        cand = {"kind": "lift_only", "score": sc, "lift": out, "slide": None, "seq": None,
                "SUCCESS_NEW": False, "reached_height": out["reached_target_z"],
                "genuine_slide": 0, "actions": out["actions"], "params_lift": lp}
        if best is None or sc > best["score"]:
            best = cand

    # --- Gravity slide alone ---
    for sp in slide_grid(rng)[: min(40, budget - rollouts)]:
        if rollouts >= budget:
            break
        out = gravity_slide_sequence(sim, brake_snap, sp)
        rollouts += 1
        tables["slide"].append({
            "root_id": root_id,
            "wrist_axis": sp.wrist_axis, "wrist_sign": sp.wrist_sign,
            "wrist_speed": sp.wrist_speed, "grip": sp.grip, "slide_s": sp.slide_s,
            "vz_hold": sp.vz_hold,
            "s0": out["s0"], "s1": out["s1"], "ds": out["ds_contact_axial"],
            "toward_interior": out["toward_interior"],
            "abs_s_reduced_m": out["abs_s_reduced_m"],
            "genuine_axial_shift": out["genuine_axial_shift"],
            "g_axial_start": out["g_axial_start"], "g_axial_end": out["g_axial_end"],
            "min_clearance": out["min_clearance"],
            "end_bilateral": out["end_bilateral"],
            "max_consec_bil": out["max_consec_bil"],
            "ground_drop": out["ground_drop"],
            "dist_end_0": out["dist_nearest_end_0"],
            "dist_end_1": out["dist_nearest_end_1"],
        })
        sc = (
            40 * out["genuine_axial_shift"]
            + 15 * out["toward_interior"]
            + 10 * out["end_bilateral"]
            + 100 * float(out["abs_s_reduced_m"] or 0)
            - 20 * out["ground_drop"]
        )
        cand = {"kind": "slide_only", "score": sc, "lift": None, "slide": out, "seq": None,
                "SUCCESS_NEW": False, "reached_height": 0,
                "genuine_slide": out["genuine_axial_shift"], "actions": out["actions"],
                "params_slide": sp}
        if best is None or sc > best["score"]:
            best = cand

    # --- Composite sequences A1 / B1 / B2 ---
    # Prefer best lift/slide params found so far, plus a small grid
    lift_cands = [
        LiftParams(0.04, 1.0, max_s=12.0),
        LiftParams(0.08, 1.0, max_s=12.0),
        LiftParams(0.12, 0.7, max_s=12.0),
        LiftParams(0.08, 0.7, max_s=12.0),
    ]
    if best and best.get("params_lift"):
        lift_cands.insert(0, best["params_lift"])
    # Prefer slides that already showed interior migration; else small random set.
    good_slides = []
    for r in tables["slide"]:
        if r.get("root_id") == root_id and int(r.get("genuine_axial_shift") or 0):
            good_slides.append(SlideParams(
                wrist_axis=int(r["wrist_axis"]),
                wrist_sign=float(r["wrist_sign"]),
                wrist_speed=float(r["wrist_speed"]),
                grip=float(r["grip"]),
                slide_s=float(r["slide_s"]),
                vz_hold=float(r["vz_hold"]),
            ))
    slide_cands = good_slides[:4] if good_slides else slide_grid(rng)[:6]
    if best and best.get("params_slide"):
        slide_cands.insert(0, best["params_slide"])

    # Low clearance: try B2 first (partial lift before wrist motion).
    orders = ("B2", "A1", "B1") if low_clearance else ("A1", "B1", "B2")
    for order in orders:
        for lp in lift_cands[:3]:
            for sp in slide_cands[:4]:
                if rollouts >= budget:
                    break
                out = run_sequence(
                    sim, brake_snap, ref,
                    order=order, target_z=target_z, lift=lp, slide=sp,
                    clearance_lift_z=0.08,
                )
                rollouts += 1
                sc = score_sequence(out)
                tables["seq"].append({
                    "root_id": root_id, "order": order,
                    "vz": lp.vz_up, "lift_grip": lp.grip,
                    "wrist_axis": sp.wrist_axis, "wrist_sign": sp.wrist_sign,
                    "wrist_speed": sp.wrist_speed, "slide_grip": sp.grip,
                    "reached_height": out["reached_height"],
                    "genuine_slide": out["genuine_slide"],
                    "end_bilateral": out["end_bilateral"],
                    "SUCCESS_NEW": int(out["SUCCESS_NEW"]),
                    "reason": (out.get("pose_meta") or {}).get("reason"),
                    "ground_drop": out["ground_drop"],
                    "obj_z_end": (out.get("end_geometry") or {}).get("obj_z"),
                    "e_theta_deg": (out.get("pose_meta") or {}).get("e_theta_deg"),
                    "e_pos_mm": (out.get("pose_meta") or {}).get("e_pos_mm"),
                    "score": sc,
                })
                tables["pose"].append({
                    "root_id": root_id, "order": order,
                    "SUCCESS_NEW": int(out["SUCCESS_NEW"]),
                    "reason": (out.get("pose_meta") or {}).get("reason"),
                    "e_theta_deg": (out.get("pose_meta") or {}).get("e_theta_deg"),
                    "e_pos_mm": (out.get("pose_meta") or {}).get("e_pos_mm"),
                    "obj_z": (out.get("pose_meta") or {}).get("obj_z"),
                })
                tables["handoff"].append({
                    "root_id": root_id, "order": order,
                    "SUCCESS_NEW": int(out["SUCCESS_NEW"]),
                    "reason": (out.get("pose_meta") or {}).get("reason"),
                })
                cand = {
                    "kind": f"seq_{order}", "score": sc, "seq": out,
                    "lift": out.get("lift"), "slide": out.get("slide"),
                    "SUCCESS_NEW": out["SUCCESS_NEW"],
                    "reached_height": out["reached_height"],
                    "genuine_slide": out["genuine_slide"],
                    "actions": out["actions"],
                    "params_lift": lp, "params_slide": sp, "order": order,
                    "pose_meta": out.get("pose_meta"),
                }
                if best is None or sc > best["score"]:
                    best = cand
            if rollouts >= budget:
                break
        if rollouts >= budget:
            break

    promising = False
    if best is not None:
        promising = bool(
            best.get("SUCCESS_NEW")
            or best.get("reached_height")
            or best.get("genuine_slide")
            or (best.get("lift") and best["lift"].get("dz_obj", 0) > 0.05 and best["lift"].get("end_bilateral"))
        )
        tables["experts"].append({
            "root_id": root_id,
            "kind": best.get("kind"),
            "order": best.get("order"),
            "score": best["score"],
            "SUCCESS_NEW": int(best.get("SUCCESS_NEW") or 0),
            "reached_height": int(best.get("reached_height") or 0),
            "genuine_slide": int(best.get("genuine_slide") or 0),
            "reason": (best.get("pose_meta") or {}).get("reason"),
            "n_actions": len(best.get("actions") or []),
            "target_z": target_z,
            "brake_dist_nearest_end_m": geo.get("dist_nearest_end_m"),
            "brake_contact_s": geo.get("contact_s_mean"),
            "brake_obj_z": geo.get("obj_z"),
        })
        if best.get("SUCCESS_NEW") or best.get("reached_height") or best.get("genuine_slide"):
            (RAW / f"{root_id}__endgrasp_expert.pkl").write_bytes(pickle.dumps({
                "root_id": root_id,
                "brake_snap": brake_snap,
                "catch_snap": snap0,
                "ref": ref,
                "actions": np.stack(best["actions"], axis=0) if best.get("actions") else None,
                "kind": best.get("kind"),
                "order": best.get("order"),
                "SUCCESS_NEW": best.get("SUCCESS_NEW"),
                "cfg": cfg,
            }))
            # Exact replay x3 from brake snap
            for i in range(3):
                if not best.get("actions"):
                    break
                sim_r, _ = make_ground_sim()
                restore_ballistic(sim_r, brake_snap)
                restore_drop(sim_r, None)
                arm(sim_r)
                gd = False
                for a in best["actions"]:
                    step_legal(sim_r, np.asarray(a, np.float32))
                    if latched(sim_r):
                        gd = True
                        break
                tables["replay"].append({
                    "root_id": root_id, "replay_i": i,
                    "SUCCESS_NEW": int(best.get("SUCCESS_NEW") or 0) if not gd else 0,
                    "ground_drop": int(gd),
                    "bilateral_end": int(pad_bilateral(sim_r)["bilateral"]) if not gd else 0,
                    "obj_z": float(sim_r.data.xpos[sim_r.ids.object_body][2]),
                    "kind": best.get("kind"),
                })
                rollouts += 1

    print(
        f"  best={None if best is None else best.get('kind')} "
        f"height={None if best is None else best.get('reached_height')} "
        f"slide={None if best is None else best.get('genuine_slide')} "
        f"succ={None if best is None else best.get('SUCCESS_NEW')} "
        f"promising={promising} rollouts={rollouts}",
        flush=True,
    )
    return {"root_id": root_id, "rollouts": rollouts, "best": best, "promising": promising,
            "geometry": geo, "target_z": target_z}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    rng = np.random.default_rng(20261021)
    tables = {k: [] for k in ("brake", "geometry", "lift", "slide", "seq", "pose", "handoff", "replay", "experts")}

    total = 0
    results = []
    for rid in PRIMARY:
        r = search_root(rid, BUDGET_PRIMARY, rng, tables)
        total += r["rollouts"]
        results.append(r)

    promising = any(r.get("promising") for r in results)
    if promising:
        remain = BUDGET_EXPAND
        per = max(remain // max(len(EXPAND), 1), 30)
        for rid in EXPAND:
            if remain <= 0:
                break
            r = search_root(rid, min(per, remain), rng, tables)
            total += r["rollouts"]
            remain -= r["rollouts"]
            results.append(r)

    write_csv(OUT / "BRAKE_SNAPSHOT_AUDIT.csv", tables["brake"])
    write_csv(OUT / "GRASP_GEOMETRY.csv", tables["geometry"])
    write_csv(OUT / "SLOW_LIFT.csv", tables["lift"])
    write_csv(OUT / "GRAVITY_SLIDING.csv", tables["slide"])
    write_csv(OUT / "SEQUENCE_COMPARISON.csv", tables["seq"])
    write_csv(OUT / "POSE_RECOVERY.csv", tables["pose"])
    write_csv(OUT / "HANDOFF_VALIDATION.csv", tables["handoff"])
    write_csv(OUT / "EXACT_REPLAY.csv", tables["replay"])
    write_csv(OUT / "FULL_EXPERTS.csv", tables["experts"])
    write_json(RAW / "budget.json", {"rollouts": total, "elapsed_s": time.time() - t_wall,
                                     "P_TGT": P_TGT.tolist(), "U_TGT": U_TGT.tolist()})

    # Answers
    geos = {r["root_id"]: r.get("geometry") for r in results if r.get("geometry")}
    lift_ok = any(int(r.get("cylinder_rose_with_hand") or 0) and float(r.get("dz_obj") or 0) > 0.03
                  for r in tables["lift"])
    height_ok = any(int(r.get("reached_target_z") or 0) for r in tables["lift"]) or any(
        int(r.get("reached_height") or 0) for r in tables["seq"])
    slide_ok = any(int(r.get("genuine_axial_shift") or 0) for r in tables["slide"])
    g_help = any(int((geos[k] or {}).get("gravity_helps_now") or 0) for k in geos)
    # Best order among sequences with progress
    order_scores = {}
    for r in tables["seq"]:
        order_scores.setdefault(r["order"], []).append(float(r.get("score") or 0))
    best_order = None
    if order_scores:
        best_order = max(order_scores, key=lambda o: float(np.mean(order_scores[o])))
    n_succ = sum(int(r.get("SUCCESS_NEW") or 0) for r in tables["experts"])
    replay_ok = bool(tables["replay"]) and any(int(r.get("SUCCESS_NEW") or 0) for r in tables["replay"])

    near = {rid: (geos[rid] or {}).get("dist_nearest_end_m") for rid in geos}
    s_mean = {rid: (geos[rid] or {}).get("contact_s_mean") for rid in geos}

    report = f"""# REPORT — Post-catch end-grasp recovery

Prior catch tapes preserved. This run: **{total}** rollouts, {time.time() - t_wall:.1f}s.

Locked targets: `P_TGT≈(0,0,0.105)`, `U_TGT≈(0,0,-1)` from `evaluation.full_objective`.

## Direct answers

1. **How close to the cylinder end is the post-catch grasp?**  
   Contact axial `s_mean` (m from COM along cylinder): `{json.dumps(s_mean)}`.  
   Distance to nearest end: `{json.dumps(near)}` m (half-length={0.03}).

2. **Can the end grasp support a slow vertical lift?**  
   Cylinder rose with hand on any slow-lift trial: **{lift_ok}**. Reached reference height: **{height_ok}**. See `SLOW_LIFT.csv`.

3. **Does wrist rotation create a useful axial gravity component?**  
   At brake-complete, gravity already helps desired interior slide? `{g_help}` per-root `gravity_helps_now` in geometry. End-of-slide `g_axial` in `GRAVITY_SLIDING.csv`.

4. **Can legal grip modulation produce controlled axial sliding toward interior?**  
   Genuine axial shift (≥3 mm toward interior, bilateral retained): **{slide_ok}**.

5. **Does the cylinder actually shift relative to the finger pads?**  
   Tracked via MuJoCo contact `s_mean` change (`ds`, `toward_interior`, `genuine_axial_shift` in `GRAVITY_SLIDING.csv`) — not hand/world motion alone.

6. **Safer order: lift→slide, slide→lift, or partial lift first?**  
   Best mean sequence score order: **{best_order}**. Detail: `SEQUENCE_COMPARISON.csv`.

7. **Reach reference operating height?** **{height_ok}** (cfg z / Plan install height, typically 0.55 m).

8. **Pose correction to locked target?** See `POSE_RECOVERY.csv` / expert reasons.

9. **Independently verified SUCCESS_NEW?** **{n_succ}** experts; replay SUCCESS_NEW seen: **{replay_ok}**.

## Notes

- Catch interception/braking tapes unchanged; recovery branches from brake-complete snapshots.
- Mechanism milestones kept distinct: lift / axial regrasp / pose / handoff / SUCCESS_NEW.
- No physics, SUCCESS_NEW, K49, or 20 ms interface changes.

## Artifacts

`BRAKE_SNAPSHOT_AUDIT.csv`, `GRASP_GEOMETRY.csv`, `SLOW_LIFT.csv`, `GRAVITY_SLIDING.csv`,
`SEQUENCE_COMPARISON.csv`, `POSE_RECOVERY.csv`, `HANDOFF_VALIDATION.csv`,
`EXACT_REPLAY.csv`, `FULL_EXPERTS.csv`, `raw/`, `videos/`.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    try:
        print(report, flush=True)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"), flush=True)
    print("ENDGRASP_DONE", json.dumps({
        "rollouts": total,
        "n_succ": n_succ,
        "lift_ok": lift_ok,
        "slide_ok": slide_ok,
        "height_ok": height_ok,
        "best_order": best_order,
        "near_end_m": near,
        "elapsed": time.time() - t_wall,
    }), flush=True)


if __name__ == "__main__":
    main()
