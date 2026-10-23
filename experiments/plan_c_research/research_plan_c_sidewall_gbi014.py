"""Re-search gbi_014 for opposing-sidewall Plan C recatch → SUCCESS_NEW.

Writes only under results/diagnostics/ground_plan_c_sidewall_research/.
Does not modify existing catch/SUCCESS experts for gbi_010/022/014.
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

import data_generation.oracles.plan_c.dynamic_catch as dc
import data_generation.oracles.plan_c.interception as er

sys.modules.setdefault("data_generation.strategies.ground_dynamic_catch", dc)
sys.modules.setdefault("data_generation.strategies.ground_early_recovery", er)

from controllers.recovery.geometry_switch import GeometrySwitch, hand_pose
from data_generation.oracles.plan_c.contact_surface import classify_contacts, score_catch_out
from data_generation.oracles.plan_c.dynamic_catch import (
    CatchParams,
    catch_param_grid,
    dynamic_catch_sequence,
    propose_catch_interceptions,
)
from data_generation.oracles.plan_c.future_trajectory import (
    pad_bilateral,
    pose_correct_and_handoff,
    record_no_intervention_future,
)
from data_generation.oracles.plan_c.interception import FULL_DETACH_ACTIVE_RECATCH
from data_generation.oracles.plan_c.catch_stabilization import LiftParams, slow_lift_sequence
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.outcomes.nominal_pose import capture_ref
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT
from training.recovery_execution import make_snap, restore_ballistic

EARLY = ROOT / "results/diagnostics/ground_ball_impact_early_recovery"
PRIOR = ROOT / "results/diagnostics/ground_ball_impact_oracle_interception"
OUT = ROOT / "results/diagnostics/ground_plan_c_sidewall_research"
RAW = OUT / "raw"
ROOT_ID = "gbi_014"
MAX_ROLLOUTS = 220
TOP_LIFT = 12


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


def load_root(root_id: str) -> dict:
    """Same physical root as the original dynamic-catch oracle (branch A)."""
    branch_path = EARLY / "raw" / f"{root_id}__branch_A.pkl"
    prior = pickle.loads((PRIOR / "raw" / f"{root_id}__post_loss.pkl").read_bytes())
    if branch_path.exists():
        blob = pickle.loads(branch_path.read_bytes())
        return {
            "cfg": dict(blob.get("cfg") or prior["config"]),
            "snap": blob["snap"],
            "ref": blob.get("ref"),
            "prior": prior,
        }
    return {"cfg": dict(prior["config"]), "snap": prior["snap"], "ref": None, "prior": prior}


def exact_replay_from_elevated(elev_snap, ref, *, n: int = 3) -> dict:
    """3× independent: restore elevated bilateral snap → pose_correct_and_handoff."""
    wins = 0
    details = []
    for i in range(n):
        sim, _ = make_ground_sim()
        restore_ballistic(sim, elev_snap)
        restore_drop(sim, None)
        arm(sim)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            details.append({"i": i, "SUCCESS_NEW": False, "reason": "BAD_ELEV_SNAP"})
            continue
        _acts, meta = pose_correct_and_handoff(sim, ref, max_s=4.0)
        details.append({"i": i, "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")), "reason": meta.get("reason")})
        if meta.get("SUCCESS_NEW"):
            wins += 1
    return {"wins": wins, "n": n, "ok_3of3": wins == n, "details": details}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(20261021)
    info = load_root(ROOT_ID)
    cfg = info["cfg"]
    sim, _ = make_ground_sim()

    print("=== physical root", ROOT_ID, flush=True)
    snap = info["snap"]
    ref = info.get("ref")
    if ref is None:
        restore_ballistic(sim, snap)
        restore_drop(sim, None)
        arm(sim)
        ref = make_ref(sim)

    # Ensure post-loss airborne snap
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    future = record_no_intervention_future(sim, snap, max_s=0.90)
    cands = propose_catch_interceptions(future, n_times=10, n_axial=7)
    print(f"future rows={len(future.get('rows') or [])} cands={len(cands)} t_free={future.get('first_contact_free_t')}", flush=True)

    pairs = catch_param_grid(rng, cands)
    search_rows = []
    sidewall_hits = []
    rollouts = 0
    best = None

    for params, cand in pairs:
        if rollouts >= MAX_ROLLOUTS:
            break
        out = dynamic_catch_sequence(sim, snap, ref, params, cand, horizon_s=0.70)
        rollouts += 1
        sc = score_catch_out(out)
        row = {
            "rollout": rollouts,
            "score": sc,
            "mechanism": out["mechanism"],
            "grasp_kind": out.get("grasp_kind"),
            "opposing_sidewall": out.get("opposing_sidewall"),
            "e_theta_end": out.get("e_theta_end"),
            "e_pos_end_mm": out.get("e_pos_end_mm"),
            "continuous_150": int(out["continuous_150"]),
            "max_consec_bil": out["max_consec_bil"],
            "end_bilateral": int(out["end_bilateral"]),
            "SUCCESS_NEW": int(out["SUCCESS_NEW"]),
            "vrel_at_first_contact": out["vrel_at_first_contact"],
            "t_c": out["t_c"],
            "s": out["s"],
            "open_steps": params.open_steps,
            "match": params.match,
            "approach_gain": params.approach_gain,
            "wrist_gain": params.wrist_gain,
            "brake": params.brake,
            "force_close_after_s": params.force_close_after_s,
            "n_sidewall": (out.get("surface_end") or {}).get("n_sidewall"),
            "n_end_face": (out.get("surface_end") or {}).get("n_end_face"),
            "reason": out.get("reason"),
        }
        search_rows.append(row)
        if best is None or sc > best["score"]:
            best = {**out, "score": sc, "params": params, "cand": cand}
        kind = out.get("grasp_kind")
        if (
            out["mechanism"] == FULL_DETACH_ACTIVE_RECATCH
            and kind in ("opposing_sidewall", "sidewall_partial")
            and out["end_bilateral"]
            and float(out.get("e_theta_end") or 99) < 45.0
        ):
            sidewall_hits.append({**out, "score": sc, "params": params, "cand": cand})
        if rollouts % 25 == 0:
            print(
                f"  rollouts={rollouts} best_kind={(best or {}).get('grasp_kind')} "
                f"best_th={(best or {}).get('e_theta_end')} best_sc={(best or {}).get('score'):.1f} "
                f"sidewall_hits={len(sidewall_hits)}",
                flush=True,
            )

    write_csv(OUT / "SEARCH.csv", search_rows)
    write_json(OUT / "BEST_CATCH.json", {
        "score": None if best is None else best["score"],
        "mechanism": None if best is None else best.get("mechanism"),
        "grasp_kind": None if best is None else best.get("grasp_kind"),
        "e_theta_end": None if best is None else best.get("e_theta_end"),
        "e_pos_end_mm": None if best is None else best.get("e_pos_end_mm"),
        "opposing_sidewall": None if best is None else best.get("opposing_sidewall"),
        "surface_end": None if best is None else best.get("surface_end"),
        "continuous_150": None if best is None else best.get("continuous_150"),
        "SUCCESS_NEW": None if best is None else best.get("SUCCESS_NEW"),
        "params": None if best is None else asdict_params(best["params"]),
        "cand": None if best is None else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best["cand"].items()},
        "n_sidewall_hits": len(sidewall_hits),
        "rollouts": rollouts,
    })

    # Rank sidewall hits for lift / handoff
    sidewall_hits.sort(key=lambda o: o["score"], reverse=True)
    lift_rows = []
    accepted = None
    for hi, hit in enumerate(sidewall_hits[:TOP_LIFT]):
        # Replay capture to brake, then slow lift, then pose+handoff
        restore_ballistic(sim, snap)
        restore_drop(sim, None)
        arm(sim)
        prefix = []
        for a in hit["capture_actions"]:
            a = np.asarray(a, np.float32)
            prefix.append(a.copy())
            step_legal(sim, a)
            if latched(sim):
                break
        surf_brake = classify_contacts(sim)
        m_brake = measure(sim, U_TGT, P_TGT)
        brake_snap = make_snap(sim)
        if not pad_bilateral(sim)["bilateral"] or latched(sim):
            lift_rows.append({
                "hit": hi,
                "stage": "brake",
                "ok": 0,
                "grasp_kind": surf_brake.get("grasp_kind"),
                "reason": "lost_at_brake",
            })
            continue
        lift = slow_lift_sequence(sim, brake_snap, LiftParams(vz_up=0.05, max_s=2.5), target_z=0.55)
        for a in lift.get("actions") or []:
            prefix.append(np.asarray(a, np.float32).copy())
        surf_lift = classify_contacts(sim)
        m_lift = measure(sim, U_TGT, P_TGT)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            lift_rows.append({
                "hit": hi,
                "stage": "lift",
                "ok": 0,
                "grasp_kind": surf_lift.get("grasp_kind"),
                "e_theta": m_lift["e_theta_deg"],
                "reason": "lost_at_lift",
            })
            continue
        elev_snap = make_snap(sim)
        # Pose correct + legal HANDOFF + unchanged nominal (SUCCESS_NEW)
        pose_acts, meta = pose_correct_and_handoff(sim, ref, max_s=4.0)
        for a in pose_acts:
            prefix.append(np.asarray(a, np.float32).copy())
        succ = bool(meta.get("SUCCESS_NEW"))
        surf_end = classify_contacts(sim) if pad_bilateral(sim)["bilateral"] else {"grasp_kind": "none"}
        row = {
            "hit": hi,
            "stage": "handoff",
            "ok": int(succ),
            "grasp_kind_brake": surf_brake.get("grasp_kind"),
            "grasp_kind_lift": surf_lift.get("grasp_kind"),
            "grasp_kind_end": surf_end.get("grasp_kind"),
            "oppose_brake": int(surf_brake.get("opposing_sidewall") or 0),
            "oppose_lift": int(surf_lift.get("opposing_sidewall") or 0),
            "theta_brake": m_brake["e_theta_deg"],
            "theta_lift": m_lift["e_theta_deg"],
            "SUCCESS_NEW": int(succ),
            "reason": meta.get("reason"),
            "score": hit["score"],
            "t_c": hit["t_c"],
            "s": hit["s"],
            "wrist_gain": hit["params"].wrist_gain,
        }
        lift_rows.append(row)
        print(
            f"  lift_hit={hi} succ={succ} kind={surf_lift.get('grasp_kind')} "
            f"th={m_lift['e_theta_deg']:.1f} reason={meta.get('reason')}",
            flush=True,
        )
        # Accept opposing sidewall or partial sidewall (unusual but valid); reject end-face.
        lift_kind = surf_lift.get("grasp_kind")
        if succ and lift_kind in ("opposing_sidewall", "sidewall_partial"):
            rep = exact_replay_from_elevated(elev_snap, ref, n=3)
            row["replay_3of3"] = int(rep["ok_3of3"])
            row["replay_wins"] = rep["wins"]
            write_json(RAW / f"{ROOT_ID}__candidate_{hi}__replay.json", rep)
            if rep["ok_3of3"]:
                accepted = {
                    "root_id": ROOT_ID,
                    "prefix_actions": np.stack(prefix, axis=0),
                    "capture_actions": np.stack(hit["capture_actions"], axis=0),
                    "snap": snap,
                    "elev_snap": elev_snap,
                    "ref": ref,
                    "cfg": cfg,
                    "mechanism": hit["mechanism"],
                    "SUCCESS_NEW": True,
                    "params": hit["params"],
                    "cand": hit["cand"],
                    "surface_brake": {k: surf_brake[k] for k in (
                        "grasp_kind", "opposing_sidewall", "n_sidewall", "n_end_face", "contact_s_mean"
                    ) if k in surf_brake},
                    "surface_lift": {k: surf_lift[k] for k in (
                        "grasp_kind", "opposing_sidewall", "n_sidewall", "n_end_face", "contact_s_mean"
                    ) if k in surf_lift},
                    "score": hit["score"],
                    "replay": rep,
                }
                (RAW / f"{ROOT_ID}__SUCCESS_NEW_sidewall_expert.pkl").write_bytes(
                    pickle.dumps({
                        "root_id": ROOT_ID,
                        "snap": snap,
                        "elev_snap": elev_snap,
                        "ref": ref,
                        "prefix_actions": accepted["prefix_actions"],
                        "capture_actions": accepted["capture_actions"],
                        "mechanism": hit["mechanism"],
                        "SUCCESS_NEW": True,
                        "params": hit["params"],
                        "cand": {
                            k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in hit["cand"].items()
                        },
                        "cfg": cfg,
                        "grasp_kind": surf_lift.get("grasp_kind"),
                        "opposing_sidewall": int(surf_lift.get("opposing_sidewall") or 0),
                    })
                )
                print("ACCEPTED SUCCESS_NEW 3/3 sidewall expert", flush=True)
                break

    write_csv(OUT / "LIFT_HANDOFF.csv", lift_rows)

    # Baseline: classify old catch expert for comparison (read-only)
    old = pickle.loads(
        (ROOT / "results/diagnostics/ground_dynamic_catch_oracle/raw/gbi_014__catch_expert.pkl").read_bytes()
    )
    restore_ballistic(sim, old["snap"])
    restore_drop(sim, None)
    arm(sim)
    for a in old["capture_actions"]:
        step_legal(sim, np.asarray(a, np.float32))
    old_audit = classify_contacts(sim)
    old_m = measure(sim, U_TGT, P_TGT)

    report = [
        "# gbi_014 sidewall re-search",
        "",
        f"- rollouts: {rollouts}",
        f"- sidewall candidate hits (mech+kind+θ<45): {len(sidewall_hits)}",
        f"- best catch kind: {(best or {}).get('grasp_kind')} θ={(best or {}).get('e_theta_end')}",
        f"- old catch kind: {old_audit.get('grasp_kind')} θ={old_m['e_theta_deg']:.1f}",
        f"- accepted SUCCESS_NEW 3/3: {accepted is not None}",
        "",
        "## Verdict",
        "",
    ]
    if accepted is not None:
        report.append(
            "gbi_014 obtained a genuine improved opposing-sidewall (or sidewall) recatch "
            "with full SUCCESS_NEW and 3/3 exact replay. Expert saved under raw/."
        )
    elif sidewall_hits:
        report.append(
            "Improved sidewall recatch candidates were found at catch/brake, but none "
            "completed SUCCESS_NEW with 3/3 replay within this budget. See LIFT_HANDOFF.csv."
        )
    else:
        report.append(
            "No opposing-sidewall FULL_DETACH recatch with θ<45° found in this budget. "
            "Wrong-surface end-face pinch remains the limiting geometry for gbi_014; "
            "secondary release-recatch is the next fallback."
        )
    report.append("")
    report.append(f"wall_time_s: {time.time() - t0:.1f}")
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(OUT / "SUMMARY.json", {
        "rollouts": rollouts,
        "n_sidewall_hits": len(sidewall_hits),
        "accepted": accepted is not None,
        "best_grasp_kind": None if best is None else best.get("grasp_kind"),
        "best_e_theta_end": None if best is None else best.get("e_theta_end"),
        "old_grasp_kind": old_audit.get("grasp_kind"),
        "old_e_theta": old_m["e_theta_deg"],
        "wall_s": time.time() - t0,
    })
    print("done", OUT, "accepted", accepted is not None, flush=True)


def asdict_params(p: CatchParams) -> dict:
    from dataclasses import asdict

    return asdict(p)


if __name__ == "__main__":
    main()
