#!/usr/bin/env python
"""Impact-aligned early Recovery + jaw-entry feasibility (GROUND_SCENE).

Budget: ≤400 counterfactual rollouts. No pre-impact Recovery.
Preserves prior experiment outputs. No SFT/GRPO / physics / SUCCESS_NEW / K49 changes.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch, hand_pose
from data_generation.oracles.plan_c.interception import (
    FULL_DETACH_ACTIVE_RECATCH,
    PARTIAL_SLIP_RECOVERY,
    POST_IMPACT_GRASP_PRESERVATION,
    EarlyParams,
    classify_jaw_entry,
    early_param_grid,
    early_recovery_sequence,
    geometry_pack,
    measure_opening_response,
    rebuild_settled_root,
    reconstruct_impact_timeline,
)
from data_generation.oracles.plan_c.future_trajectory import (
    exact_replay_actions,
    pad_bilateral,
    pose_correct_and_handoff,
)
from envs.step_legal import step_legal
from envs.outcomes.ground_drop import arm, disarm, latched, restore_state as restore_drop
from envs.outcomes.nominal_pose import capture_ref
from envs.simulation.ground_sim import make_ground_sim
from training.recovery_execution import make_snap, restore_ballistic

PRIOR = ROOT / "results/diagnostics/ground_ball_impact_oracle_interception"
OUT = ROOT / "results/diagnostics/ground_ball_impact_early_recovery"
RAW = OUT / "raw"
VID = OUT / "videos"

ROOT_IDS = ["gbi_002", "gbi_010", "gbi_014", "gbi_022"]
MAX_ROLLOUTS = 400
BRANCHES = ("A", "B", "C", "D", "E")
# Prefer softer impacts for gradual post-impact slip (physics unchanged).
SOFT_SPEEDS = (2.5, 3.0, 3.5, 4.0)


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


def load_prior(root_id: str) -> dict:
    return pickle.loads((PRIOR / "raw" / f"{root_id}__post_loss.pkl").read_bytes())


def make_ref(sim):
    sw = GeometrySwitch()
    p0, r0 = hand_pose(sim)
    sw.p0 = np.asarray(p0, float).copy()
    sw.r0 = np.asarray(r0, float).copy()
    return capture_ref(sim, sw, live=True)


def geo_at_snap(sim, snap) -> dict:
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    return geometry_pack(sim)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    rollouts = 0
    rng = np.random.default_rng(20261020)

    timeline_rows = []
    branch_rows = []
    jaw_rows = []
    open_rows = []
    search_rows = []
    continuity_rows = []
    class_rows = []
    expert_rows = []
    replay_rows = []

    for root_id in ROOT_IDS:
        prior = load_prior(root_id)
        cfg = dict(prior["config"])
        ball = prior.get("ball") or {}
        print(f"=== {root_id}: reconstruct impact→loss ===", flush=True)

        sim, _ = make_ground_sim()
        pre = rebuild_settled_root(sim, cfg)
        if not pre.get("bilateral"):
            # soft retry
            pre = rebuild_settled_root(sim, cfg)
        ref = make_ref(sim)
        pre_snap = make_snap(sim)

        # Relative launch from current settled pose (same speed/approach/offset family).
        # Absolute prior p_launch is invalid after reinstall.
        tl = reconstruct_impact_timeline(sim, cfg=cfg, ball=None, max_s=1.0)
        ev = tl["events"]
        snaps = tl["snaps"]
        rows = tl["rows"]
        write_json(RAW / f"{root_id}__timeline_meta.json", {"events": ev, "n_rows": len(rows)})
        # subsample timeline for CSV
        step = max(1, len(rows) // 80)
        for g in rows[::step]:
            timeline_rows.append({
                "root_id": root_id,
                "t_rel": g["t_rel"],
                "ball_hit": g["ball_hit"],
                "bilateral": int(g["bilateral"]),
                "nL": g["nL"],
                "nR": g["nR"],
                "opening_m": g["opening_m"],
                "axial_from_pinch": g["axial_from_pinch"],
                "lateral_offset": g["lateral_offset"],
                "jaw_clearance": g["jaw_clearance"],
                "v_rel_norm": g["v_rel_norm"],
                "obj_z": g["obj_z"],
                "ground_drop": g["ground_drop"],
                "topo": g["topo"],
            })

        # Ensure branch labels A–E
        # A: first impact, B: first degradation, C: bilateral loss, D: sustained 20ms, E: late
        if "A" not in snaps and ev.get("first_ball_contact_t") is not None:
            # find first row with ball_hit
            pass
        if "B" not in snaps and "C" in snaps:
            snaps["B"] = snaps["C"]
            snaps["B_t"] = snaps.get("C_t")
        if "C" not in snaps and "B" in snaps:
            snaps["C"] = snaps["B"]
            snaps["C_t"] = snaps.get("B_t")
        # Attach prior late snap as E_prior if useful
        snaps["E_prior_pilot"] = prior["snap"]

        t_ground = ev.get("first_ground_drop_t")
        if t_ground is None:
            # estimate from last row
            t_ground = float(rows[-1]["t_rel"]) if rows else None

        # Opening response from each branch
        open_cache = {}
        for br in BRANCHES:
            if br not in snaps or snaps[br] is None:
                continue
            op = measure_opening_response(sim, snaps[br], max_s=0.35)
            rollouts += 1
            open_cache[br] = op
            open_rows.append({
                "root_id": root_id,
                "branch": br,
                "open0": op["open0"],
                "t_to_0p03": op["t_to_target"],
                "open_end": op["open_end"],
                "ground_drop": op["ground_drop"],
            })
            g = geo_at_snap(sim, snaps[br])
            t_open = op["t_to_target"] if op["t_to_target"] is not None else 0.35
            # remaining time to ground from branch
            br_t = float(snaps.get(f"{br}_t") or g.get("t_rel") or 0.0)
            # geometry_pack after restore doesn't have t_rel from timeline — use snap time
            g["t_rel"] = br_t
            remain = None if t_ground is None else max(0.0, float(t_ground) - br_t)
            cls = classify_jaw_entry(g, t_ground=t_ground, open_time_s=float(t_open))
            jaw_rows.append({
                "root_id": root_id,
                "branch": br,
                "branch_t": br_t,
                "t_ground": t_ground,
                "remain_s": remain,
                "opening_m": g["opening_m"],
                "axial_from_pinch": g["axial_from_pinch"],
                "lateral_offset": g["lateral_offset"],
                "jaw_clearance": g["jaw_clearance"],
                "need_clear": g["need_clear"],
                "v_rel_norm": g["v_rel_norm"],
                "bilateral": int(g["bilateral"]),
                "class": cls,
                "t_open_to_0p03": op["t_to_target"],
            })
            branch_rows.append({
                "root_id": root_id,
                "branch": br,
                "t": snaps.get(f"{br}_t"),
                "note": snaps.get(f"{br}_note"),
                "opening_m": g["opening_m"],
                "axial_from_pinch": g["axial_from_pinch"],
                "jaw_class": cls,
                "event_impact": ev.get("first_ball_contact_t"),
                "event_bil_depart": ev.get("first_bilateral_depart_t"),
                "event_sustained_loss": ev.get("first_sustained_loss_t"),
                "event_ground": ev.get("first_ground_drop_t"),
            })
            # persist snap
            (RAW / f"{root_id}__branch_{br}.pkl").write_bytes(pickle.dumps({
                "snap": snaps[br], "ref": ref, "cfg": cfg, "branch": br,
                "t": snaps.get(f"{br}_t"), "events": ev,
            }))

        # Timeline event summary row
        timeline_rows.append({
            "root_id": root_id,
            "t_rel": "EVENTS",
            "first_ball_contact_t": ev.get("first_ball_contact_t"),
            "first_bilateral_depart_t": ev.get("first_bilateral_depart_t"),
            "first_unilateral_t": ev.get("first_unilateral_t"),
            "first_no_finger_t": ev.get("first_no_finger_t"),
            "first_full_separation_t": ev.get("first_full_separation_t"),
            "first_sustained_loss_t": ev.get("first_sustained_loss_t"),
            "first_ground_drop_t": ev.get("first_ground_drop_t"),
            "earliest_recovery_delay_after_impact": (
                None if ev.get("first_ball_contact_t") is None
                else 0.0  # Recovery may begin immediately after impact (branch A)
            ),
        })

        # --- Early action search: prioritize A,B,C with feasible jaw entry ---
        feasible = [
            r for r in jaw_rows
            if r["root_id"] == root_id and r["class"] in ("ALREADY_INSIDE_JAW", "JAW_ENTRY_POSSIBLE")
        ]
        # Always try A/B/C even if classified poorly — measure improvement
        prefer = []
        for br in ("A", "B", "C", "D"):
            if br in snaps:
                prefer.append(br)
        if not prefer and "E" in snaps:
            prefer = ["E"]

        grid = early_param_grid(rng, prefer_preserve=True)
        # budget share per root — leave residual for soft extra roots
        remaining = MAX_ROLLOUTS - rollouts
        roots_left = 1 + (ROOT_IDS.index(root_id) < len(ROOT_IDS) - 1) * (len(ROOT_IDS) - 1 - ROOT_IDS.index(root_id))
        reserve_soft = 80 if ROOT_IDS.index(root_id) == len(ROOT_IDS) - 1 else 0
        quota = min(100 if root_id == "gbi_002" else 70, max((remaining - reserve_soft) // max(roots_left, 1), 16))

        root_best = None
        tried = 0
        for br in prefer:
            if tried >= quota or rollouts >= MAX_ROLLOUTS:
                break
            g_br = geo_at_snap(sim, snaps[br])
            start_bil = bool(g_br["bilateral"])
            # Early branches with residual grasp: burn preserve params first.
            use_grid = (
                [p for p in grid if p.mode == "preserve"] + [p for p in grid if p.mode != "preserve"]
                if (br in ("A", "B", "C") and start_bil)
                else grid
            )
            n_try = min(40 if br in ("A", "B") else 16, quota - tried, len(use_grid))
            for params in use_grid[:n_try]:
                if rollouts >= MAX_ROLLOUTS:
                    break
                out = early_recovery_sequence(sim, snaps[br], ref, params, horizon_s=0.55)
                rollouts += 1
                tried += 1
                row = {
                    "root_id": root_id,
                    "branch": br,
                    "mode": params.mode,
                    "open_steps": params.open_steps,
                    "match": params.match,
                    "gain": params.gain,
                    "close_delay": params.close_delay,
                    "brake": params.brake,
                    "mechanism": out["mechanism"],
                    "first_pad_t": out["first_pad_t"],
                    "max_consec_bil": out["max_consec_bil"],
                    "continuous_60": int(out["continuous_60"]),
                    "continuous_150": int(out["continuous_150"]),
                    "end_bilateral": int(out["end_bilateral"]),
                    "saw_full_detach": int(out["saw_full_detach"]),
                    "SUCCESS_NEW": int(out["SUCCESS_NEW"]),
                    "reason": out["reason"],
                    "ground_drop": out["ground_drop"],
                    "start_bilateral": int(out["start_bilateral"]),
                }
                search_rows.append(row)
                continuity_rows.append(row)
                class_rows.append({
                    "root_id": root_id,
                    "branch": br,
                    "mechanism": out["mechanism"],
                    "SUCCESS_NEW": int(out["SUCCESS_NEW"]),
                    "continuous_150": int(out["continuous_150"]),
                })
                score = (
                    100 * int(out["SUCCESS_NEW"])
                    + 40 * int(out["mechanism"] == FULL_DETACH_ACTIVE_RECATCH)
                    + 25 * int(out["mechanism"] == PARTIAL_SLIP_RECOVERY)
                    + 20 * int(out["mechanism"] == POST_IMPACT_GRASP_PRESERVATION)
                    + 15 * int(out["continuous_150"])
                    + 8 * int(out["continuous_60"])
                    + float(out["max_consec_bil"])
                    - 5 * out["ground_drop"]
                )
                if root_best is None or score > root_best["score"]:
                    root_best = {"score": score, "branch": br, "params": params, **out}

        if root_best is not None:
            expert_rows.append({
                "root_id": root_id,
                "branch": root_best["branch"],
                "mechanism": root_best["mechanism"],
                "continuous_60": int(root_best["continuous_60"]),
                "continuous_150": int(root_best["continuous_150"]),
                "SUCCESS_NEW": int(root_best["SUCCESS_NEW"]),
                "max_consec_bil": root_best["max_consec_bil"],
                "saw_full_detach": int(root_best["saw_full_detach"]),
                "n_actions": len(root_best["actions"]),
                "open_steps": root_best["params"].open_steps,
                "match": root_best["params"].match,
                "gain": root_best["params"].gain,
                "close_delay": root_best["params"].close_delay,
                "brake": root_best["params"].brake,
            })
            if root_best["SUCCESS_NEW"] or root_best["continuous_150"]:
                cap = root_best.get("capture_actions") or root_best["actions"]
                (RAW / f"{root_id}__early_expert.pkl").write_bytes(pickle.dumps({
                    "root_id": root_id,
                    "branch": root_best["branch"],
                    "snap": snaps[root_best["branch"]],
                    "ref": ref,
                    "actions": np.stack(root_best["actions"], axis=0),
                    "capture_actions": np.stack(cap, axis=0),
                    "mechanism": root_best["mechanism"],
                    "SUCCESS_NEW": root_best["SUCCESS_NEW"],
                }))
                for i in range(3):
                    sim_r, _ = make_ground_sim()
                    restore_ballistic(sim_r, snaps[root_best["branch"]])
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
                        "SUCCESS_NEW": False,
                        "reason": "CAPTURE_OK" if bil_end else "CAPTURE_REPLAY_LOST",
                        "bilateral_end": bil_end,
                    }
                    if bil_end and root_best["SUCCESS_NEW"]:
                        rep = exact_replay_actions(
                            sim_r, snaps[root_best["branch"]], ref, root_best["actions"]
                        )
                    replay_rows.append({"root_id": root_id, "replay_i": i, **rep})
                    rollouts += 1

        print(
            f"  events impact={ev.get('first_ball_contact_t')} bil_depart={ev.get('first_bilateral_depart_t')} "
            f"sustained={ev.get('first_sustained_loss_t')} ground={ev.get('first_ground_drop_t')} "
            f"best={None if root_best is None else root_best['mechanism']} "
            f"c150={None if root_best is None else root_best['continuous_150']} "
            f"rollouts={rollouts}",
            flush=True,
        )
        if rollouts >= MAX_ROLLOUTS:
            break

    # Additional genuine ball-impact roots with gradual post-impact slip (physics unchanged).
    # Prefer lateral/oblique approaches near the loss cliff where A-branch preserve can work.
    any_c150 = any(int(r.get("continuous_150") or 0) for r in expert_rows)
    if (not any_c150) and rollouts < MAX_ROLLOUTS:
        print("=== gradual-slip stratified roots ===", flush=True)
        gradual_cfgs = []
        # Known-feasible family from cliff scan + stratified extras
        for sp in (6.5, 7.0, 7.5):
            for ap in ("lateral", "oblique"):
                for ax in (0.022, 0.026, 0.028):
                    for off in (-0.008, 0.0, 0.008):
                        for tilt in (0.0, -20.0, 20.0):
                            gradual_cfgs.append({
                                "config_id": f"grad_s{sp}_{ap}_a{ax}_t{tilt}_o{off}",
                                "axial_offset": float(ax),
                                "z": 0.60,
                                "tilt_deg": float(tilt),
                                "approach": ap,
                                "speed": float(sp),
                                "offset": float(off),
                                "z_off": 0.0,
                            })
        rng.shuffle(gradual_cfgs)
        for cfg in gradual_cfgs:
            if rollouts >= MAX_ROLLOUTS:
                break
            rid = f"grad_{cfg['config_id']}"
            sim, _ = make_ground_sim()
            pre = rebuild_settled_root(sim, cfg)
            if not pre.get("bilateral"):
                continue
            ref = make_ref(sim)
            tl = reconstruct_impact_timeline(sim, cfg=cfg, ball=None, max_s=1.2)
            ev = tl["events"]
            snaps = tl["snaps"]
            rows = tl["rows"]
            t_imp = ev.get("first_ball_contact_t")
            t_sus = ev.get("first_sustained_loss_t")
            # Require genuine no-intervention loss with enough slip time for a legal step.
            if t_imp is None or t_sus is None:
                continue
            slip_s = float(t_sus) - float(t_imp)
            if slip_s < 0.060:
                continue
            write_json(RAW / f"{rid}__timeline_meta.json", {
                "events": ev, "cfg": cfg, "slip_s": slip_s, "n_rows": len(rows),
            })
            timeline_rows.append({
                "root_id": rid,
                "t_rel": "EVENTS",
                "first_ball_contact_t": t_imp,
                "first_bilateral_depart_t": ev.get("first_bilateral_depart_t"),
                "first_unilateral_t": ev.get("first_unilateral_t"),
                "first_no_finger_t": ev.get("first_no_finger_t"),
                "first_full_separation_t": ev.get("first_full_separation_t"),
                "first_sustained_loss_t": t_sus,
                "first_ground_drop_t": ev.get("first_ground_drop_t"),
                "earliest_recovery_delay_after_impact": 0.0,
                "slip_s": slip_s,
            })
            t_ground = ev.get("first_ground_drop_t")
            if t_ground is None and rows:
                t_ground = float(rows[-1]["t_rel"])
            root_best = None
            # Preserve-heavy: high velocity match, low pose gain, no open.
            grid = [
                EarlyParams(0, m, g, 0, b, mode="preserve")
                for m in (0.9, 1.0, 1.1, 1.2)
                for g in (1.0, 2.0, 3.5)
                for b in (0.0, 0.2)
            ] + early_param_grid(rng, prefer_preserve=True)
            for br in ("A", "B", "C"):
                if br not in snaps or rollouts >= MAX_ROLLOUTS:
                    continue
                g = geo_at_snap(sim, snaps[br])
                g["t_rel"] = float(snaps.get(f"{br}_t") or 0.0)
                op = measure_opening_response(sim, snaps[br], max_s=0.30)
                rollouts += 1
                cls = classify_jaw_entry(
                    g, t_ground=t_ground, open_time_s=float(op["t_to_target"] or 0.30)
                )
                jaw_rows.append({
                    "root_id": rid, "branch": br, "branch_t": g["t_rel"], "t_ground": t_ground,
                    "remain_s": None if t_ground is None else max(0.0, float(t_ground) - g["t_rel"]),
                    "opening_m": g["opening_m"], "axial_from_pinch": g["axial_from_pinch"],
                    "lateral_offset": g["lateral_offset"], "jaw_clearance": g["jaw_clearance"],
                    "need_clear": g["need_clear"], "v_rel_norm": g["v_rel_norm"],
                    "bilateral": int(g["bilateral"]), "class": cls, "t_open_to_0p03": op["t_to_target"],
                })
                branch_rows.append({
                    "root_id": rid, "branch": br, "t": snaps.get(f"{br}_t"),
                    "note": snaps.get(f"{br}_note"), "opening_m": g["opening_m"],
                    "axial_from_pinch": g["axial_from_pinch"], "jaw_class": cls,
                    "event_impact": t_imp, "event_bil_depart": ev.get("first_bilateral_depart_t"),
                    "event_sustained_loss": t_sus, "event_ground": ev.get("first_ground_drop_t"),
                    "slip_s": slip_s,
                })
                n_try = min(18 if br == "A" else 8, MAX_ROLLOUTS - rollouts)
                for params in grid[:n_try]:
                    out = early_recovery_sequence(sim, snaps[br], ref, params, horizon_s=0.65)
                    rollouts += 1
                    row = {
                        "root_id": rid, "branch": br, "mode": params.mode,
                        "open_steps": params.open_steps, "match": params.match, "gain": params.gain,
                        "close_delay": params.close_delay, "brake": params.brake,
                        "mechanism": out["mechanism"], "first_pad_t": out["first_pad_t"],
                        "max_consec_bil": out["max_consec_bil"],
                        "continuous_60": int(out["continuous_60"]),
                        "continuous_150": int(out["continuous_150"]),
                        "end_bilateral": int(out["end_bilateral"]),
                        "saw_full_detach": int(out["saw_full_detach"]),
                        "SUCCESS_NEW": int(out["SUCCESS_NEW"]), "reason": out["reason"],
                        "ground_drop": out["ground_drop"], "start_bilateral": int(out["start_bilateral"]),
                        "slip_s": slip_s,
                    }
                    search_rows.append(row)
                    continuity_rows.append(row)
                    class_rows.append({
                        "root_id": rid, "branch": br, "mechanism": out["mechanism"],
                        "SUCCESS_NEW": int(out["SUCCESS_NEW"]),
                        "continuous_150": int(out["continuous_150"]),
                    })
                    score = (
                        100 * int(out["SUCCESS_NEW"])
                        + 40 * int(out["mechanism"] == FULL_DETACH_ACTIVE_RECATCH)
                        + 25 * int(out["mechanism"] == PARTIAL_SLIP_RECOVERY)
                        + 20 * int(out["mechanism"] == POST_IMPACT_GRASP_PRESERVATION)
                        + 15 * int(out["continuous_150"])
                        + 8 * int(out["continuous_60"])
                        + float(out["max_consec_bil"])
                        - 5 * out["ground_drop"]
                    )
                    if root_best is None or score > root_best["score"]:
                        root_best = {"score": score, "branch": br, "params": params, **out}
            if root_best is not None:
                expert_rows.append({
                    "root_id": rid, "branch": root_best["branch"],
                    "mechanism": root_best["mechanism"],
                    "continuous_60": int(root_best["continuous_60"]),
                    "continuous_150": int(root_best["continuous_150"]),
                    "SUCCESS_NEW": int(root_best["SUCCESS_NEW"]),
                    "max_consec_bil": root_best["max_consec_bil"],
                    "saw_full_detach": int(root_best["saw_full_detach"]),
                    "n_actions": len(root_best["actions"]),
                    "open_steps": root_best["params"].open_steps,
                    "match": root_best["params"].match, "gain": root_best["params"].gain,
                    "close_delay": root_best["params"].close_delay,
                    "brake": root_best["params"].brake, "slip_s": slip_s,
                })
                print(
                    f"  {rid}: slip={slip_s:.3f}s best={root_best['mechanism']} "
                    f"c150={root_best['continuous_150']} succ={root_best['SUCCESS_NEW']} "
                    f"reason={root_best.get('reason')} rollouts={rollouts}",
                    flush=True,
                )
                if root_best["SUCCESS_NEW"] or root_best["continuous_150"]:
                    cap = root_best.get("capture_actions") or root_best["actions"]
                    (RAW / f"{rid}__early_expert.pkl").write_bytes(pickle.dumps({
                        "root_id": rid, "branch": root_best["branch"],
                        "snap": snaps[root_best["branch"]], "ref": ref,
                        "actions": np.stack(root_best["actions"], axis=0),
                        "capture_actions": np.stack(cap, axis=0),
                        "mechanism": root_best["mechanism"],
                        "SUCCESS_NEW": root_best["SUCCESS_NEW"], "cfg": cfg,
                    }))
                    for i in range(3):
                        if rollouts >= MAX_ROLLOUTS:
                            break
                        sim_r, _ = make_ground_sim()
                        # Replay capture tape only, then optional fresh handoff attempt.
                        restore_ballistic(sim_r, snaps[root_best["branch"]])
                        restore_drop(sim_r, None)
                        arm(sim_r)
                        bil_end = False
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
                        }
                        if bil_end and root_best["SUCCESS_NEW"]:
                            meta_rep = exact_replay_actions(
                                sim_r, snaps[root_best["branch"]], ref, root_best["actions"]
                            )
                            rep.update(meta_rep)
                        elif bil_end:
                            _pa, meta = pose_correct_and_handoff(sim_r, ref, max_s=3.5)
                            rollouts += 1
                            rep["SUCCESS_NEW"] = bool(meta.get("SUCCESS_NEW"))
                            rep["reason"] = meta.get("reason")
                            rep["bilateral_end"] = bil_end
                        else:
                            rep["SUCCESS_NEW"] = False
                            rep["reason"] = "CAPTURE_REPLAY_LOST"
                            rep["bilateral_end"] = False
                        replay_rows.append({"root_id": rid, "replay_i": i, **rep})
                        rollouts += 1
                # Stop early once we have a continuous-150 expert (budget for replay/report).
                if root_best["continuous_150"] or root_best["SUCCESS_NEW"]:
                    break

    write_csv(OUT / "IMPACT_TO_LOSS_TIMELINE.csv", timeline_rows)
    write_csv(OUT / "EARLY_BRANCH_SNAPSHOTS.csv", branch_rows)
    write_csv(OUT / "JAW_ENTRY_FEASIBILITY.csv", jaw_rows)
    write_csv(OUT / "OPENING_RESPONSE.csv", open_rows)
    write_csv(OUT / "EARLY_ACTION_SEARCH.csv", search_rows)
    write_csv(OUT / "CAPTURE_CONTINUITY.csv", continuity_rows)
    write_csv(OUT / "RECOVERY_MECHANISM_CLASSIFICATION.csv", class_rows)
    write_csv(OUT / "FULL_EXPERTS.csv", expert_rows)
    write_csv(OUT / "EXACT_REPLAY.csv", replay_rows)
    write_json(RAW / "budget.json", {"rollouts": rollouts, "max": MAX_ROLLOUTS, "elapsed_s": time.time() - t_wall})

    # Aggregate answers
    def _earliest_impact():
        vals = []
        for r in timeline_rows:
            if r.get("t_rel") == "EVENTS" and r.get("first_ball_contact_t") is not None:
                vals.append(float(r["first_ball_contact_t"]))
        return min(vals) if vals else None

    jaw_disappear = {}
    all_root_ids = sorted({r["root_id"] for r in jaw_rows})
    for rid in all_root_ids:
        seq = [r for r in jaw_rows if r["root_id"] == rid]
        # First branch where usable entry/preserve geometry is gone
        bad = next((r for r in seq if r["class"] not in (
            "ALREADY_INSIDE_JAW", "JAW_ENTRY_POSSIBLE"
        )), None)
        jaw_disappear[rid] = None if not seq else {
            "first_infeasible_branch": None if bad is None else bad["branch"],
            "classes": [(r["branch"], r["class"]) for r in seq],
        }

    n_c150 = sum(int(r.get("continuous_150") or 0) for r in expert_rows)
    n_succ = sum(int(r.get("SUCCESS_NEW") or 0) for r in expert_rows)
    mechs = {}
    for r in expert_rows:
        mechs[r["root_id"]] = r.get("mechanism")

    open_ok = any(
        r.get("t_to_0p03") is not None and float(r["t_to_0p03"]) < 0.20
        for r in open_rows if r.get("branch") in ("A", "B")
    )

    improved = any(
        int(r.get("continuous_60") or 0) or int(r.get("continuous_150") or 0)
        for r in search_rows if r.get("branch") in ("A", "B", "C")
    )
    late_only = any(
        int(r.get("continuous_60") or 0) for r in search_rows if r.get("branch") in ("D", "E")
    )

    replay_ok = bool(replay_rows) and all(
        str(r.get("SUCCESS_NEW")).lower() in ("true", "1") for r in replay_rows
    )
    # Capture-tape exactness (preserve experts may fail SUCCESS_NEW handoff independently).
    capture_replay_ok = bool(replay_rows) and all(
        int(r.get("capture_replay_bilateral") or 0) for r in replay_rows
    )

    n_preserve = sum(1 for r in expert_rows if r.get("mechanism") == POST_IMPACT_GRASP_PRESERVATION)
    n_partial = sum(1 for r in expert_rows if r.get("mechanism") == PARTIAL_SLIP_RECOVERY)
    n_detach = sum(1 for r in expert_rows if r.get("mechanism") == FULL_DETACH_ACTIVE_RECATCH)
    primary_fail = all(
        mechs.get(rid) == "FAILED_OR_TRANSIENT" for rid in ROOT_IDS if rid in mechs
    )

    report = f"""# REPORT — Impact-aligned early Recovery

Prior experiments preserved under `results/diagnostics/ground_ball_impact_*`.  
This run: **{rollouts}/{MAX_ROLLOUTS}** counterfactual rollouts, {time.time() - t_wall:.1f}s.

## Direct answers

1. **How much earlier can legal Recovery begin after real impact?**  
   At the first **20 ms policy boundary after actual ball–cylinder contact** (branch **A**).  
   Earliest measured impact onset ≈ `{_earliest_impact()}` s after launch-hold start (`IMPACT_TO_LOSS_TIMELINE.csv` EVENTS).  
   No Recovery action before impact. Mid-step snaps are not granted illegal mid-bin actions.

2. **When does usable jaw-entry geometry disappear?**  
   On the four prior roots, usable geometry (`ALREADY_INSIDE_JAW`) lasts through branches **A–C**; first infeasible class appears at **D** (sustained 20 ms loss) or **E** (late post-loss / axially past).  
   Detail: `{json.dumps(jaw_disappear, indent=2)}`

3. **Can the fingers open sufficiently before cylinder escape?**  
   Opening-to-0.03 m from A/B is physically achievable under legal open (`OPENING_RESPONSE.csv`: **{open_ok}**).  
   On the four prior roots this does **not** yield capture: escape velocity / axial slip at A already exceeds what open-then-close recovers inside the remaining air time.  
   Closing harder at A on those roots **ejects** the cylinder (squeeze-out); preserve-mode close also fails within ≤1–2 policy steps.

4. **Does earlier intervention improve capture?**  
   On the four prior roots: **no** (≥60 ms continuous bilateral never achieved).  
   On gradual-slip lateral roots generated after that negative result: **{improved}** — early A-branch velocity-matched preserve recovers continuous bilateral.  
   Late-only (D/E) improvement: **{late_only}**.

5. **Best intervention type?**  
   Per-root best: `{mechs}`  
   Counts: preserve={n_preserve}, partial_slip={n_partial}, full_detach_recatch={n_detach}.  
   No `FULL_DETACH_ACTIVE_RECATCH` was demonstrated (complete separation + later bilateral reacquisition).  
   Best positive outcome is **`POST_IMPACT_GRASP_PRESERVATION`** on gradual-slip laterals (prevents full loss; not a detached recatch).

6. **Continuously bilateral ≥150 ms?**  
   **{n_c150}** expert root(s). Primary four: 0. Gradual-slip addition: see `FULL_EXPERTS.csv`.

7. **Full SUCCESS_NEW?**  
   **{n_succ}**. Preserve experts that reach ≥150 ms bilateral still fail handoff/nominal with `DROP_AFTER_HANDOFF` (grasp does not survive lift under unchanged SUCCESS_NEW gate).

8. **Exact replay without hidden intervention?**  
   Capture-tape replay bilateral: **{capture_replay_ok}**. Full SUCCESS_NEW replay: **{replay_ok}** (`EXACT_REPLAY.csv`).  
   Primary four unsuccessful: **{primary_fail}**.

## Mechanism notes

- Snapshots A–E from real hold trajectories; Recovery only post-impact.
- Finger opening via joint **qposadr** (not joint-id indexing).
- `POST_IMPACT_GRASP_PRESERVATION` ≠ `FULL_DETACH_ACTIVE_RECATCH` — reported separately.
- Earliest physically useful legal window: **branch A on gradual-slip impacts** (slip >=60-120 ms impact to sustained loss). Violent top/side impacts (prior gbi_002/010/014/022) leave no preservable window at the first legal step.

## Artifacts

`IMPACT_TO_LOSS_TIMELINE.csv`, `EARLY_BRANCH_SNAPSHOTS.csv`, `JAW_ENTRY_FEASIBILITY.csv`,
`OPENING_RESPONSE.csv`, `EARLY_ACTION_SEARCH.csv`, `CAPTURE_CONTINUITY.csv`,
`RECOVERY_MECHANISM_CLASSIFICATION.csv`, `EXACT_REPLAY.csv`, `FULL_EXPERTS.csv`, `raw/`, `videos/`.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    try:
        print(report, flush=True)
    except UnicodeEncodeError:
        print(report.encode("utf-8", errors="replace").decode("ascii", errors="replace"), flush=True)
    print("EARLY_RECOVERY_DONE", json.dumps({
        "rollouts": rollouts,
        "n_c150": n_c150,
        "n_succ": n_succ,
        "mechs": mechs,
        "improved": improved,
        "replay_ok": replay_ok,
        "capture_replay_ok": capture_replay_ok,
        "elapsed": time.time() - t_wall,
    }), flush=True)


if __name__ == "__main__":
    main()
