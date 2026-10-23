#!/usr/bin/env python
"""Earliest-divergence diagnosis: elevated vs GROUND_SCENE for failed Plan B roots."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.datasets.recovery_rebuild import resolve_stored_path  # noqa: E402
from envs.ballistic_impact import elevated_world, world  # noqa: E402
from envs.contacts.frames import extract_contacts  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched, terminal_reason  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from evaluation.handoff_from_state import evaluate_handoff_from_state  # noqa: E402

MIG = ROOT / "results" / "diagnostics" / "plan_ab_table_free_migration"
OUT = ROOT / "results" / "diagnostics" / "plan_b_table_free_recovery"
BAK = MIG / "pre_update_unified_manifest.jsonl"


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


def make_env(elev: bool):
    class E:
        pass

    env = E()
    env.sim, *_ = elevated_world() if elev else world()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


def sample(sim):
    ids = sim.ids
    rows, _ = extract_contacts(sim)
    nL = sum(1 for r in rows if r.get("side") == "L")
    nR = sum(1 for r in rows if r.get("side") == "R")
    oq = int(sim.model.jnt_qposadr[ids.object_jnt])
    od = int(sim.model.jnt_dofadr[ids.object_jnt])
    return {
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "obj_pos": np.array(sim.data.qpos[oq : oq + 3], float).copy(),
        "obj_v": np.array(sim.data.qvel[od : od + 3], float).copy(),
        "hand": np.array(sim.data.xpos[ids.hand_body], float).copy(),
        "p_des": np.array(sim.fsm.p_des, float).copy(),
        "fg": float(sim.fsm.fg_cmd),
        "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else 0.0,
        "nL": nL,
        "nR": nR,
        "z": float(sim.data.xpos[ids.object_body][2]),
        "latched": bool(latched(sim)),
        "phase": str(sim.fsm.phase),
    }


def diverge(a, b):
    return {
        "dq": float(np.max(np.abs(a["qpos"] - b["qpos"]))),
        "dv": float(np.max(np.abs(a["qvel"] - b["qvel"]))),
        "dobj": float(np.linalg.norm(a["obj_pos"] - b["obj_pos"])),
        "dobj_v": float(np.linalg.norm(a["obj_v"] - b["obj_v"])),
        "dhand": float(np.linalg.norm(a["hand"] - b["hand"])),
        "dpdes": float(np.linalg.norm(a["p_des"] - b["p_des"])),
        "dnL": int(a["nL"] - b["nL"]),
        "dnR": int(a["nR"] - b["nR"]),
        "dz": float(a["z"] - b["z"]),
        "dfg": float(a["fg"] - b["fg"]),
        "dctrl7": float(a["ctrl7"] - b["ctrl7"]),
        "latched_diff": bool(a["latched"] != b["latched"]),
    }


def meaningful(d):
    return (
        d["dobj"] > 1e-5
        or d["dhand"] > 1e-5
        or abs(d["dz"]) > 1e-4
        or d["dnL"] != 0
        or d["dnR"] != 0
        or d["latched_diff"]
        or d["dobj_v"] > 1e-4
    )


def load_uref():
    out = {}
    with BAK.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("plan_family") == "B":
                out[r["expert_id"]] = r
    return out


def load_mig():
    out = {}
    with (MIG / "migration_results.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("plan_family") == "B":
                out[r["expert_id"]] = r
    return out


def diagnose_one(eid: str, uref: dict) -> dict:
    row = uref[eid]
    ref = row["initial_physical_state_ref"]
    blob = pickle.loads(resolve_stored_path(ref["snapshot"]).read_bytes())
    if "capture" not in blob:
        return {"expert_id": eid, "error": "flat_snap_unsupported_in_paired_diag"}
    acts = np.asarray(np.load(resolve_stored_path(ref["continuous_npz"]))["actions"], np.float32)
    elev = make_env(True)
    gnd = make_env(False)
    restore_capture(elev, blob["capture"])
    restore_capture(gnd, blob["capture"])
    s0e, s0g = sample(elev.sim), sample(gnd.sim)
    d0 = diverge(s0e, s0g)
    ic_match = (
        d0["dq"] < 1e-9
        and d0["dv"] < 1e-9
        and d0["dobj"] < 1e-12
        and d0["dhand"] < 1e-12
        and abs(d0["dfg"]) < 1e-12
        and d0["dnL"] == 0
        and d0["dnR"] == 0
    )
    first = None
    traces = []
    lost_e = lost_g = 0
    elev_fail = gnd_fail = None
    elev_done = gnd_done = False
    for i, a in enumerate(acts):
        if not elev_done:
            step_legal(elev.sim, a)
            se = sample(elev.sim)
            fe, lost_e = recovery_failed(elev.sim, blob["ref"], lost_e)
            if fe and elev_fail is None:
                elev_fail = {"step": i, "reason": terminal_reason(elev.sim), "z": se["z"]}
                elev_done = True
        else:
            se = sample(elev.sim)
            fe = True
        if not gnd_done:
            step_legal(gnd.sim, a)
            sg = sample(gnd.sim)
            fg, lost_g = recovery_failed(gnd.sim, blob["ref"], lost_g)
            if fg and gnd_fail is None:
                gnd_fail = {"step": i, "reason": terminal_reason(gnd.sim), "z": sg["z"]}
                gnd_done = True
        else:
            sg = sample(gnd.sim)
            fg = True
        d = diverge(se, sg)
        if first is None and meaningful(d):
            first = {"step": i, "diverge": d, "elev": _slim(se), "gnd": _slim(sg)}
        if i < 3 or i % 10 == 0 or (first and i == first["step"]) or fe or fg:
            traces.append(
                {
                    "step": i,
                    "elev": _slim(se),
                    "gnd": _slim(sg),
                    "diverge": d,
                    "elev_fail": bool(fe),
                    "gnd_fail": bool(fg),
                }
            )
        if elev_done and gnd_done:
            break
    hand_elev = hand_gnd = None
    if elev_fail is None:
        hand_elev = evaluate_handoff_from_state(elev.sim, blob["ref"], stabilize_n=0)
    if gnd_fail is None:
        hand_gnd = evaluate_handoff_from_state(gnd.sim, blob["ref"], stabilize_n=0)
    # phase classification at ground fail
    phase = "unknown"
    if gnd_fail:
        step = gnd_fail["step"]
        # grip from early actions
        grip0 = float(acts[0, 6]) if len(acts) else 0.0
        if grip0 < -0.5 and step <= 15:
            phase = "release_or_early_drop"
        elif step < len(acts) * 0.5:
            phase = "chase_or_mid_flight"
        else:
            phase = "late_recatch_or_brake"
    elif hand_gnd is not None:
        if hand_gnd.get("SUCCESS_NEW"):
            phase = "success_new"
        elif hand_gnd.get("reason") == "SUCCESS_12S":
            phase = "handoff_success_12s_not_new"
        else:
            phase = f"handoff_{hand_gnd.get('reason')}"

    return {
        "expert_id": eid,
        "mechanism": row.get("contact_grasp_mechanism"),
        "n_actions": int(len(acts)),
        "ic_match": bool(ic_match),
        "ic_diverge": d0,
        "ic_elev": _slim(s0e),
        "ic_gnd": _slim(s0g),
        "first_meaningful_diverge": first,
        "elev_fail": elev_fail,
        "gnd_fail": gnd_fail,
        "hand_elev": _hand_slim(hand_elev),
        "hand_gnd": _hand_slim(hand_gnd),
        "failure_phase": phase,
        "dominant_hypothesis": _hypothesis(ic_match, first, elev_fail, gnd_fail, hand_gnd),
        "trace_sparse": traces,
    }


def _slim(s):
    return {
        "z": s["z"],
        "nL": s["nL"],
        "nR": s["nR"],
        "latched": s["latched"],
        "fg": s["fg"],
        "ctrl7": s["ctrl7"],
        "obj_pos": s["obj_pos"].tolist(),
        "obj_v": s["obj_v"].tolist(),
        "hand": s["hand"].tolist(),
    }


def _hand_slim(h):
    if h is None:
        return None
    return {
        "SUCCESS_NEW": h.get("SUCCESS_NEW"),
        "reason": h.get("reason"),
        "pre_pos_mm": h.get("pre_pos_mm"),
        "pre_theta": h.get("pre_theta"),
        "success_12s": h.get("success_12s") if "success_12s" in h else int(h.get("reason") == "SUCCESS_12S"),
    }


def _hypothesis(ic_match, first, elev_fail, gnd_fail, hand_gnd):
    if not ic_match:
        return "state_migration_mismatch"
    if elev_fail and gnd_fail and elev_fail["step"] == gnd_fail["step"]:
        return "original_tape_invalid_both_scenes"
    if elev_fail and gnd_fail and elev_fail["reason"] == "TABLE_DROP" and gnd_fail["step"] >= elev_fail["step"]:
        return "table_as_failure_surface_tape_invalid_without_table"
    if elev_fail is None and gnd_fail is not None:
        return "genuine_physics_diff_ground_only_fail"
    if hand_gnd and hand_gnd.get("reason") == "SUCCESS_12S" and not hand_gnd.get("SUCCESS_NEW"):
        return "tape_ok_success_new_hold_fail"
    if first and first["step"] <= 2 and first["diverge"]["dobj"] > 1e-4:
        return "numerical_contact_sensitivity"
    return "original_action_tape_no_longer_valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--expert-id", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    uref = load_uref()
    mig = load_mig()

    # stratified sample
    if args.expert_id:
        ids = [args.expert_id]
    else:
        buckets = {
            "FAIL_REPLAY:release_long": [],
            "FAIL_REPLAY:chase_lost": [],
            "FAIL_REPLAY:release_short": [],
            "FAIL_SUCCESS_12S_NOT_NEW:follow_early": [],
            "FAIL_SUCCESS_12S_NOT_NEW:reference": [],
            "FAIL_SUCCESS_12S_NOT_NEW:shallow_stop": [],
            "FAIL_SUCCESS_12S_NOT_NEW:later_close": [],
            "PASS:follow_early": [],
        }
        for eid, r in mig.items():
            key = f"{r['status']}:{r.get('mechanism')}"
            if key in buckets:
                buckets[key].append(eid)
        ids = []
        per = max(1, args.limit // max(1, len([b for b in buckets.values() if b])))
        for key, lst in buckets.items():
            ids.extend(lst[:per])
        ids = ids[: args.limit]

    results = []
    for i, eid in enumerate(ids):
        print(f"[{i+1}/{len(ids)}] {eid} ...", flush=True)
        try:
            out = diagnose_one(eid, uref)
            out["mig_status"] = mig.get(eid, {}).get("status")
            out["mig_mechanism"] = mig.get(eid, {}).get("mechanism")
        except Exception as exc:
            out = {"expert_id": eid, "error": f"{type(exc).__name__}: {exc}"}
        results.append(out)
        print(
            f"  hyp={out.get('dominant_hypothesis')} phase={out.get('failure_phase')} "
            f"ic_match={out.get('ic_match')} elev_fail={out.get('elev_fail')} gnd_fail={out.get('gnd_fail')}",
            flush=True,
        )

    from collections import Counter

    hyp = Counter(r.get("dominant_hypothesis") for r in results if "dominant_hypothesis" in r)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(results),
        "hypothesis_counts": dict(hyp),
        "ids": ids,
    }
    (OUT / "divergence_sample.jsonl").write_text(
        "\n".join(json.dumps(jsafe(r), ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )
    (OUT / "divergence_summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    print(json.dumps(summary["hypothesis_counts"], indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
