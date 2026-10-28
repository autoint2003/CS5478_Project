#!/usr/bin/env python
"""Paired MuJoCo feasibility: Gravity Pivot & Coupled DX/DZ vs production baselines.

Does not modify production Oracle or datasets. Writes results under
results/diagnostics/precision_coupled_feasibility/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.corrections.axial_inward import correct_dz_inward  # noqa: E402
from data_generation.oracles.precision.handoff import probe_direct_handoff  # noqa: E402
from data_generation.oracles.precision.mechanics.support_transfer import correct_dx_support_transfer  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import correct_dtheta, run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.coupled_feasibility.braking_probe import probe_inward_braking  # noqa: E402
from experiments.precision.coupled_feasibility.coupled_dx_dz import coupled_dx_dz  # noqa: E402
from experiments.precision.coupled_feasibility.gravity_pivot import gravity_aligned_pivot  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/precision_coupled_feasibility"
AUDIT = ROOT / "results/diagnostics/precision_candidate_audit/CLASSIFICATION.jsonl"
KNOWN = [
    "B__hard_910026__post_b1_v2__e3a32a342c39f52b",
    "B__hard_910220__0412621ee978627d",
    "HARD_IC__hard_910026__634ebeb7f0c6e4a2",  # noqa: spelling — hash from stage2 IC id
]


def jsafe(o: Any) -> Any:
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


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def cfg() -> PrecisionConfig:
    return PrecisionConfig(evaluate_handoff=True, use_verified_lateral_fallback=False)


def load_rows() -> list[dict]:
    return [json.loads(l) for l in AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()]


def strip_acts(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("actions", "series", "series_tail")}


def run_orientation_pair(env, row: dict, pcfg: PrecisionConfig) -> dict:
    blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
    from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible

    elig = check_stage1_eligible(
        env,
        {"capture": blob["capture"], "ref": blob["ref"], "sample_id": row.get("sample_id")},
        allow_live_verify=True,
    )
    if not elig.eligible:
        return {
            "sample_id": row["sample_id"],
            "experiment": "orientation",
            "status": "STAGE1_INELIGIBLE",
            "stage1_eligibility": elig.to_dict(),
            "rejected_reason": f"STAGE1_INELIGIBLE:{elig.reason}",
        }
    plan = str(row["plan_family"])
    restore_capture(env, blob["capture"])
    base = correct_dtheta(env, {"capture": blob["capture"], "ref": blob["ref"]}, pcfg, plan)
    restore_capture(env, blob["capture"])
    cand = gravity_aligned_pivot(env, blob)
    # Optional handoff from each end state
    hand_b = hand_c = None
    if base.get("ok") and base.get("capture"):
        hand_b = probe_direct_handoff(env, base["capture"], blob["ref"])
    if cand.get("ok") and cand.get("capture"):
        hand_c = probe_direct_handoff(env, cand["capture"], blob["ref"])
    return {
        "sample_id": row["sample_id"],
        "experiment": "orientation",
        "baseline": strip_acts(base),
        "candidate": strip_acts(cand),
        "baseline_handoff": {
            "SUCCESS_NEW": bool((hand_b or {}).get("SUCCESS_NEW")),
            "SUCCESS_12S": bool((hand_b or {}).get("SUCCESS_12S")),
            "reason": (hand_b or {}).get("reason"),
        }
        if hand_b
        else None,
        "candidate_handoff": {
            "SUCCESS_NEW": bool((hand_c or {}).get("SUCCESS_NEW")),
            "SUCCESS_12S": bool((hand_c or {}).get("SUCCESS_12S")),
            "reason": (hand_c or {}).get("reason"),
        }
        if hand_c
        else None,
        "dtheta0": (row.get("physics_entry") or {}).get("dtheta_deg"),
        "baseline_improve": (base.get("errors") or {}).get("dtheta_deg")
        if base.get("errors")
        else None,
        "candidate_improve_deg": cand.get("d_theta_improve_deg"),
    }


def run_position_pair(env, row: dict, pcfg: PrecisionConfig) -> dict:
    blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
    from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible

    elig = check_stage1_eligible(
        env,
        {"capture": blob["capture"], "ref": blob["ref"], "sample_id": row.get("sample_id")},
        allow_live_verify=True,
    )
    if not elig.eligible:
        return {
            "sample_id": row["sample_id"],
            "experiment": "position",
            "status": "STAGE1_INELIGIBLE",
            "stage1_eligibility": elig.to_dict(),
            "rejected_reason": f"STAGE1_INELIGIBLE:{elig.reason}",
        }
    restore_capture(env, blob["capture"])
    # Baseline: ST dx then dz inward if needed (no VL)
    acts_b = []
    st = correct_dx_support_transfer(env, {"capture": blob["capture"], "ref": blob["ref"]}, pcfg)
    logs_b = {"dx": strip_acts(st)}
    cur = {"capture": st.get("capture") or blob["capture"], "ref": blob["ref"]}
    acts_b += st.get("actions") or []
    dz = None
    if st.get("ok"):
        restore_capture(env, cur["capture"])
        err = measure_errors(env.sim)
        if abs(float(err["dz_mm"])) > float(pcfg.ez_tol_mm) * 0.25 or float(err["e_pos_mm"]) > float(pcfg.working_pos_mm):
            dz = correct_dz_inward(env, cur, pcfg)
            logs_b["dz"] = strip_acts(dz)
            acts_b += dz.get("actions") or []
            cur = {"capture": dz.get("capture") or cur["capture"], "ref": blob["ref"]}
    err_b = measure_errors(env.sim)
    hand_b = probe_direct_handoff(env, cur["capture"], blob["ref"]) if err_b.get("bilateral") else None

    restore_capture(env, blob["capture"])
    coup = coupled_dx_dz(env, blob, cfg=pcfg)
    err_c = coup.get("errors1") or measure_errors(env.sim)
    hand_c = None
    if coup.get("ok") and coup.get("capture"):
        hand_c = probe_direct_handoff(env, coup["capture"], blob["ref"])

    return {
        "sample_id": row["sample_id"],
        "experiment": "position",
        "baseline_ok": bool(st.get("ok") and (dz is None or dz.get("ok"))),
        "baseline_dx_ok": bool(st.get("ok")),
        "baseline_dz_ok": None if dz is None else bool(dz.get("ok")),
        "baseline_errors": err_b,
        "baseline_logs": logs_b,
        "baseline_handoff": {
            "SUCCESS_NEW": bool((hand_b or {}).get("SUCCESS_NEW")),
            "SUCCESS_12S": bool((hand_b or {}).get("SUCCESS_12S")),
        }
        if hand_b
        else None,
        "candidate": strip_acts(coup),
        "candidate_handoff": {
            "SUCCESS_NEW": bool((hand_c or {}).get("SUCCESS_NEW")),
            "SUCCESS_12S": bool((hand_c or {}).get("SUCCESS_12S")),
        }
        if hand_c
        else None,
        "candidate_errors": err_c,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-orient", type=int, default=0)
    ap.add_argument("--limit-pos", type=int, default=25)
    ap.add_argument("--skip-handoff-heavy", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    by_id = {r["sample_id"]: r for r in rows}
    pcfg = cfg()
    env = make_env()

    # --- Orientation cohort: dtheta failures + known ---
    dtheta_fail = [
        r
        for r in rows
        if r.get("class") == "PRECISION_REQUIRED_BUT_FAILED"
        and (
            ((r.get("precision") or {}).get("first_failed_primitive") == "dtheta")
            or any("dtheta:" in str(f) for f in ((r.get("precision") or {}).get("failures") or []))
        )
    ]
    # Deduplicate to true dtheta failure strings
    dtheta_fail = [
        r
        for r in dtheta_fail
        if any(str(f).startswith("dtheta:") for f in ((r.get("precision") or {}).get("failures") or []))
        or ((r.get("precision") or {}).get("first_failed_primitive") == "dtheta"
            and any("dtheta" in str(f) for f in ((r.get("precision") or {}).get("failures") or [])))
    ]
    # Simpler filter
    dtheta_fail = [
        r
        for r in rows
        if any(str(f).startswith("dtheta:") for f in ((r.get("precision") or {}).get("failures") or []))
    ]
    if args.limit_orient:
        dtheta_fail = dtheta_fail[: args.limit_orient]

    orient_rows = []
    print(f"orientation pairs: {len(dtheta_fail)}", flush=True)
    for i, row in enumerate(dtheta_fail):
        print(f"[orient {i+1}/{len(dtheta_fail)}] {row['sample_id']}", flush=True)
        try:
            orient_rows.append(run_orientation_pair(env, row, pcfg))
        except Exception as exc:
            orient_rows.append({"sample_id": row["sample_id"], "error": str(exc), "traceback": traceback.format_exc(limit=4)})

    # --- Position cohort: ST failures + regression controls ---
    st_fail = [
        r
        for r in rows
        if r.get("class") == "PRECISION_REQUIRED_BUT_FAILED"
        and ((r.get("precision") or {}).get("first_failed_primitive") in ("support_transfer", "DX1", "dz"))
    ]
    successes = [r for r in rows if r.get("class") == "PRECISION_REQUIRED_AND_SUCCESSFUL"]
    pos_cohort = st_fail[: args.limit_pos] + [by_id[k] for k in KNOWN if k in by_id] + successes[:5]
    # unique by sample_id
    seen = set()
    pos_unique = []
    for r in pos_cohort:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        pos_unique.append(r)

    pos_rows = []
    print(f"position pairs: {len(pos_unique)}", flush=True)
    for i, row in enumerate(pos_unique):
        print(f"[pos {i+1}/{len(pos_unique)}] {row['sample_id']}", flush=True)
        try:
            pos_rows.append(run_position_pair(env, row, pcfg))
        except Exception as exc:
            pos_rows.append({"sample_id": row["sample_id"], "error": str(exc), "traceback": traceback.format_exc(limit=4)})

    # --- Braking diagnosis on successes with dz ---
    brake_rows = []
    brake_targets = [r for r in successes if "dz" in str((r.get("precision") or {}).get("executed_correction_skills"))][:6]
    # also a couple ST fails with large |dz|
    for r in rows:
        if r.get("class") != "PRECISION_REQUIRED_BUT_FAILED":
            continue
        dz = abs(float(((r.get("physics_entry") or {}).get("e_signed_mm") or [0, 0, 0])[2]))
        if dz > 15:
            brake_targets.append(r)
        if len(brake_targets) >= 10:
            break
    seen_b = set()
    for i, row in enumerate(brake_targets):
        if row["sample_id"] in seen_b:
            continue
        seen_b.add(row["sample_id"])
        print(f"[brake {len(seen_b)}] {row['sample_id']}", flush=True)
        try:
            blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
            brake_rows.append({"sample_id": row["sample_id"], **probe_inward_braking(env, blob, pcfg)})
        except Exception as exc:
            brake_rows.append({"sample_id": row["sample_id"], "error": str(exc)})

    # Persist
    (OUT / "ORIENTATION_PAIRS.json").write_text(json.dumps(jsafe(orient_rows), indent=2), encoding="utf-8")
    (OUT / "POSITION_PAIRS.json").write_text(json.dumps(jsafe(pos_rows), indent=2), encoding="utf-8")
    (OUT / "BRAKING_PROBES.json").write_text(json.dumps(jsafe(brake_rows), indent=2), encoding="utf-8")

    # CSV summary
    csv_path = OUT / "PAIRED_RESULTS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "experiment",
                "sample_id",
                "baseline_ok",
                "candidate_ok",
                "baseline_SUCCESS_NEW",
                "candidate_SUCCESS_NEW",
                "baseline_SUCCESS_12S",
                "candidate_SUCCESS_12S",
                "notes",
            ]
        )
        for r in orient_rows:
            if r.get("error"):
                w.writerow(["orientation", r["sample_id"], "", "", "", "", "", "", r["error"]])
                continue
            b_ok = bool((r.get("baseline") or {}).get("ok"))
            c_ok = bool((r.get("candidate") or {}).get("ok"))
            w.writerow(
                [
                    "orientation",
                    r["sample_id"],
                    b_ok,
                    c_ok,
                    (r.get("baseline_handoff") or {}).get("SUCCESS_NEW"),
                    (r.get("candidate_handoff") or {}).get("SUCCESS_NEW"),
                    (r.get("baseline_handoff") or {}).get("SUCCESS_12S"),
                    (r.get("candidate_handoff") or {}).get("SUCCESS_12S"),
                    (r.get("candidate") or {}).get("reason") or (r.get("candidate") or {}).get("motion_class"),
                ]
            )
        for r in pos_rows:
            if r.get("error"):
                w.writerow(["position", r["sample_id"], "", "", "", "", "", "", r["error"]])
                continue
            w.writerow(
                [
                    "position",
                    r["sample_id"],
                    r.get("baseline_ok"),
                    bool((r.get("candidate") or {}).get("ok")),
                    (r.get("baseline_handoff") or {}).get("SUCCESS_NEW"),
                    (r.get("candidate_handoff") or {}).get("SUCCESS_NEW"),
                    (r.get("baseline_handoff") or {}).get("SUCCESS_12S"),
                    (r.get("candidate_handoff") or {}).get("SUCCESS_12S"),
                    (r.get("candidate") or {}).get("reason"),
                ]
            )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_orientation": len(orient_rows),
        "orient_baseline_ok": sum(1 for r in orient_rows if (r.get("baseline") or {}).get("ok")),
        "orient_candidate_ok": sum(1 for r in orient_rows if (r.get("candidate") or {}).get("ok")),
        "orient_cand_better_theta": sum(
            1
            for r in orient_rows
            if (r.get("candidate") or {}).get("d_theta_improve_deg") is not None
            and float((r.get("candidate") or {}).get("d_theta_improve_deg") or 0) >= 2.0
            and not (r.get("baseline") or {}).get("ok")
        ),
        "n_position": len(pos_rows),
        "pos_baseline_ok": sum(1 for r in pos_rows if r.get("baseline_ok")),
        "pos_candidate_ok": sum(1 for r in pos_rows if (r.get("candidate") or {}).get("ok")),
        "pos_candidate_infeasible": sum(1 for r in pos_rows if (r.get("candidate") or {}).get("infeasible") or (r.get("candidate") or {}).get("reason", "").startswith("no_feasible") or "not_representable" in str((r.get("candidate") or {}).get("reason"))),
        "n_brake": len(brake_rows),
        "brake_mean_residual_mm": float(
            np.mean([abs(float(r.get("residual_at_end_mm") or 0)) for r in brake_rows if "residual_at_end_mm" in r] or [0])
        ),
    }
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
