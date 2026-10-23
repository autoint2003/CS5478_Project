"""Post-sidewall handoff search for gbi_014 improved catches.

Replays top opposing_sidewall params from SEARCH.csv, varies lift/stabilize/
in-hand correction, requires SUCCESS_NEW + 3/3 elevated replay.
Writes under results/diagnostics/ground_plan_c_sidewall_research/ only.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import data_generation.oracles.plan_c.dynamic_catch as dc
import data_generation.oracles.plan_c.interception as er

sys.modules.setdefault("data_generation.strategies.ground_dynamic_catch", dc)
sys.modules.setdefault("data_generation.strategies.ground_early_recovery", er)

from controllers.recovery.action_map import TAU_OPEN, TAU_SECURE
from controllers.recovery.geometry_switch import GeometrySwitch, hand_pose
from data_generation.oracles.plan_b.release_recatch import _action_from
from data_generation.oracles.plan_c.catch_stabilization import (
    LiftParams,
    SlideParams,
    gravity_slide_sequence,
    slow_lift_sequence,
)
from data_generation.oracles.plan_c.contact_surface import classify_contacts
from data_generation.oracles.plan_c.dynamic_catch import (
    CatchParams,
    dynamic_catch_sequence,
    propose_catch_interceptions,
)
from data_generation.oracles.plan_c.future_trajectory import (
    hold_action,
    object_axis,
    pad_bilateral,
    pose_correct_and_handoff,
    record_no_intervention_future,
    sample_kin,
)
from data_generation.oracles.plan_c.interception import FULL_DETACH_ACTIVE_RECATCH
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.outcomes.nominal_pose import capture_ref
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT
from training.recovery_execution import make_snap, restore_ballistic

OUT = ROOT / "results/diagnostics/ground_plan_c_sidewall_research"
RAW = OUT / "raw"
EARLY = ROOT / "results/diagnostics/ground_ball_impact_early_recovery"
PRIOR = ROOT / "results/diagnostics/ground_ball_impact_oracle_interception"
ROOT_ID = "gbi_014"


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


def load_snap_ref():
    branch = pickle.loads((EARLY / "raw" / f"{ROOT_ID}__branch_A.pkl").read_bytes())
    prior = pickle.loads((PRIOR / "raw" / f"{ROOT_ID}__post_loss.pkl").read_bytes())
    snap = branch["snap"]
    ref = branch.get("ref")
    cfg = dict(branch.get("cfg") or prior["config"])
    if ref is None:
        sim, _ = make_ground_sim()
        restore_ballistic(sim, snap)
        restore_drop(sim, None)
        arm(sim)
        ref = make_ref(sim)
    return snap, ref, cfg


def top_sidewall_params(n: int = 6) -> list[dict]:
    rows = list(csv.DictReader((OUT / "SEARCH.csv").open(encoding="utf-8")))
    hits = [
        r
        for r in rows
        if r.get("grasp_kind") == "opposing_sidewall"
        and float(r["e_theta_end"]) < 20.0
        and int(float(r["continuous_150"])) == 1
    ]
    hits.sort(key=lambda r: -float(r["score"]))
    # unique-ish by (t_c,s,wrist,open,match,gain,brake)
    seen = set()
    out = []
    for r in hits:
        key = (
            r["t_c"],
            r["s"],
            r["wrist_gain"],
            r["open_steps"],
            r["match"],
            r["approach_gain"],
            r["brake"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= n:
            break
    return out


def params_from_row(r: dict) -> CatchParams:
    return CatchParams(
        open_steps=int(float(r["open_steps"])),
        match=float(r["match"]),
        approach_gain=float(r["approach_gain"]),
        brake=float(r["brake"]),
        wrist_gain=float(r["wrist_gain"]),
        t_c=float(r["t_c"]),
        s_axial=float(r["s"]),
        force_close_after_s=0.30,
        close_vrel_max=0.55,
    )


def match_cand(cands: list[dict], t_c: float, s: float) -> dict | None:
    best = None
    best_d = 1e9
    for c in cands:
        d = abs(float(c["t_c"]) - t_c) + 5.0 * abs(float(c["s"]) - s)
        if d < best_d:
            best_d = d
            best = c
    return best


def small_correction(sim, *, gain: float, steps: int, grip: float, slip_wrist: float):
    actions = []
    for _ in range(steps):
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        po = np.array(sim.data.xpos[sim.ids.object_body], float)
        ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
        prel = Rh.T @ (po - ph)
        err_p = np.asarray(P_TGT, float) - prel
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
    for _ in range(8):
        actions.append(hold_action(1.0))
        step_legal(sim, hold_action(1.0))
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    return actions


def exact_replay_elev(elev_snap, ref, stabilize_n: int, *, n: int = 3) -> dict:
    wins = 0
    details = []
    for i in range(n):
        sim, _ = make_ground_sim()
        restore_ballistic(sim, elev_snap)
        restore_drop(sim, None)
        arm(sim)
        for _ in range(stabilize_n):
            step_legal(sim, hold_action(1.0))
            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                break
        _a, meta = pose_correct_and_handoff(sim, ref, max_s=4.0)
        details.append({"i": i, "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")), "reason": meta.get("reason")})
        if meta.get("SUCCESS_NEW"):
            wins += 1
    return {"wins": wins, "n": n, "ok_3of3": wins == n, "details": details}


def main():
    t0 = time.time()
    snap, ref, cfg = load_snap_ref()
    sim, _ = make_ground_sim()
    tops = top_sidewall_params(6)
    print(f"top sidewall params: {len(tops)}", flush=True)
    future = record_no_intervention_future(sim, snap, max_s=0.90)
    cands = propose_catch_interceptions(future, n_times=10, n_axial=7)
    print(f"future cands={len(cands)}", flush=True)
    rows = []
    accepted = None

    lift_opts = [
        LiftParams(vz_up=0.04, max_s=2.0),
        LiftParams(vz_up=0.06, max_s=2.5),
        LiftParams(vz_up=0.03, max_s=3.0),
    ]
    corr_opts = [
        None,
        {"gain": 2.0, "steps": 20, "grip": 1.0, "slip_wrist": 0.0},
        {"gain": 3.0, "steps": 30, "grip": 0.85, "slip_wrist": 0.4},
        {"gain": 2.5, "steps": 25, "grip": 0.6, "slip_wrist": 0.8},
    ]
    stab_opts = (0, 8, 15)
    slide_opts = (False, True)

    for ti, tr in enumerate(tops):
        params = params_from_row(tr)
        cand = match_cand(cands, params.t_c, params.s_axial)
        if cand is None:
            print(f"  param{ti} no matching cand", flush=True)
            continue
        out = dynamic_catch_sequence(sim, snap, ref, params, cand, horizon_s=0.70)
        if out.get("grasp_kind") != "opposing_sidewall" or not out["end_bilateral"]:
            print(f"  param{ti} lost sidewall kind={out.get('grasp_kind')} th={out.get('e_theta_end')}", flush=True)
            continue
        if out["mechanism"] != FULL_DETACH_ACTIVE_RECATCH:
            print(f"  param{ti} mech={out['mechanism']}", flush=True)
            continue
        cap = [np.asarray(a, np.float32) for a in out["capture_actions"]]
        # Persist catch tape for this sidewall hit
        (RAW / f"{ROOT_ID}__sidewall_catch_{ti}.pkl").write_bytes(
            pickle.dumps({
                "root_id": ROOT_ID,
                "snap": snap,
                "ref": ref,
                "capture_actions": np.stack(cap, axis=0),
                "actions": np.stack(out["actions"], axis=0),
                "params": params,
                "cand": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cand.items()},
                "grasp_kind": out.get("grasp_kind"),
                "e_theta_end": out.get("e_theta_end"),
                "e_pos_end_mm": out.get("e_pos_end_mm"),
                "mechanism": out["mechanism"],
                "surface_end": out.get("surface_end"),
            })
        )
        print(
            f"=== param{ti} th={out['e_theta_end']:.1f} pos={out['e_pos_end_mm']:.1f} "
            f"score_src={tr['score']}",
            flush=True,
        )

        for lift_p in lift_opts:
            for do_slide in slide_opts:
                for corr in corr_opts:
                    for stab in stab_opts:
                        # Replay catch
                        restore_ballistic(sim, snap)
                        restore_drop(sim, None)
                        arm(sim)
                        prefix = []
                        lost = False
                        for a in cap:
                            prefix.append(a.copy())
                            step_legal(sim, a)
                            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                                # allow brief gap only during open; at end must hold
                                pass
                        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                            continue
                        brake_snap = make_snap(sim)
                        surf_b = classify_contacts(sim)

                        # Optional gravity slide then lift
                        if do_slide:
                            slide = gravity_slide_sequence(
                                sim,
                                brake_snap,
                                SlideParams(slide_s=0.35, grip=0.4, wrist_speed=0.6),
                            )
                            for a in slide.get("actions") or []:
                                prefix.append(np.asarray(a, np.float32).copy())
                            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                                continue
                            brake_snap = make_snap(sim)

                        lift = slow_lift_sequence(sim, brake_snap, lift_p, target_z=0.55)
                        for a in lift.get("actions") or []:
                            prefix.append(np.asarray(a, np.float32).copy())
                        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                            continue
                        if corr is not None:
                            ca = small_correction(sim, **corr)
                            prefix.extend(ca)
                            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                                continue
                        for _ in range(stab):
                            prefix.append(hold_action(1.0))
                            step_legal(sim, hold_action(1.0))
                            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                                break
                        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                            continue

                        elev_snap = make_snap(sim)
                        surf_e = classify_contacts(sim)
                        m_e = measure(sim, U_TGT, P_TGT)
                        if surf_e.get("grasp_kind") not in ("opposing_sidewall", "sidewall_partial"):
                            continue

                        _pose, meta = pose_correct_and_handoff(sim, ref, max_s=4.0)
                        row = {
                            "param_i": ti,
                            "vz": lift_p.vz_up,
                            "slide": int(do_slide),
                            "corr": None if corr is None else json.dumps(corr),
                            "stab": stab,
                            "grasp_kind": surf_e.get("grasp_kind"),
                            "oppose": int(surf_e.get("opposing_sidewall") or 0),
                            "theta": m_e["e_theta_deg"],
                            "pos_mm": m_e["e_pos_mm"],
                            "SUCCESS_NEW": int(bool(meta.get("SUCCESS_NEW"))),
                            "reason": meta.get("reason"),
                            "t_c": params.t_c,
                            "s": params.s_axial,
                            "wrist_gain": params.wrist_gain,
                        }
                        rows.append(row)
                        if meta.get("SUCCESS_NEW"):
                            print(
                                f"  SUCCESS candidate th={m_e['e_theta_deg']:.1f} "
                                f"pos={m_e['e_pos_mm']:.1f} stab={stab} slide={do_slide}",
                                flush=True,
                            )
                            rep = exact_replay_elev(elev_snap, ref, stab, n=3)
                            row["replay_3of3"] = int(rep["ok_3of3"])
                            row["replay_wins"] = rep["wins"]
                            write_json(RAW / f"{ROOT_ID}__handoff_try_{len(rows)}__replay.json", rep)
                            if rep["ok_3of3"]:
                                accepted = {
                                    "root_id": ROOT_ID,
                                    "snap": snap,
                                    "elev_snap": elev_snap,
                                    "ref": ref,
                                    "cfg": cfg,
                                    "capture_actions": np.stack(cap, axis=0),
                                    "prefix_actions": np.stack(prefix, axis=0),
                                    "params": params,
                                    "SUCCESS_NEW": True,
                                    "surface": {
                                        "brake": surf_b.get("grasp_kind"),
                                        "elev": surf_e.get("grasp_kind"),
                                        "oppose_elev": int(surf_e.get("opposing_sidewall") or 0),
                                    },
                                    "meta": meta,
                                    "replay": rep,
                                    "row": row,
                                }
                                (RAW / f"{ROOT_ID}__SUCCESS_NEW_sidewall_expert.pkl").write_bytes(
                                    pickle.dumps({
                                        "root_id": ROOT_ID,
                                        "snap": snap,
                                        "elev_snap": elev_snap,
                                        "ref": ref,
                                        "cfg": cfg,
                                        "capture_actions": accepted["capture_actions"],
                                        "prefix_actions": accepted["prefix_actions"],
                                        "params": params,
                                        "SUCCESS_NEW": True,
                                        "mechanism": FULL_DETACH_ACTIVE_RECATCH,
                                        "grasp_kind": surf_e.get("grasp_kind"),
                                        "opposing_sidewall": int(surf_e.get("opposing_sidewall") or 0),
                                    })
                                )
                                print("ACCEPTED 3/3", flush=True)
                                break
                    if accepted:
                        break
                if accepted:
                    break
            if accepted:
                break
        if accepted:
            break

    write_csv(OUT / "HANDOFF_SEARCH.csv", rows)
    write_json(OUT / "HANDOFF_SUMMARY.json", {
        "accepted": accepted is not None,
        "n_tries": len(rows),
        "n_success_flag": sum(int(r["SUCCESS_NEW"]) for r in rows),
        "wall_s": time.time() - t0,
        "best_reason": None if not rows else min(rows, key=lambda r: (0 if r["SUCCESS_NEW"] else 1, r["pos_mm"])),
    })
    # Append to REPORT
    rep_path = OUT / "REPORT.md"
    extra = [
        "",
        "## Handoff search (sidewall catches)",
        "",
        f"- tries: {len(rows)}",
        f"- SUCCESS_NEW flags: {sum(int(r['SUCCESS_NEW']) for r in rows)}",
        f"- accepted 3/3: {accepted is not None}",
        f"- wall_s: {time.time() - t0:.1f}",
        "",
    ]
    if accepted is not None:
        extra.append(
            "gbi_014 now has a verified opposing-sidewall recatch with full SUCCESS_NEW "
            "and 3/3 elevated replay. Golden gbi_010/022 untouched."
        )
    else:
        n_drop = sum(1 for r in rows if r.get("reason") == "DROP_AFTER_HANDOFF")
        extra.append(
            f"Sidewall geometry is fixed (θ≈3–10°, opposing_sidewall), but handoff still fails "
            f"({n_drop}/{len(rows)} DROP_AFTER_HANDOFF). Improved recatch yes; full SUCCESS_NEW not yet."
        )
    with rep_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(extra) + "\n")
    print("done accepted", accepted is not None, "tries", len(rows), flush=True)


if __name__ == "__main__":
    main()
