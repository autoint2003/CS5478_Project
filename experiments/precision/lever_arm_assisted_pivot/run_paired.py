#!/usr/bin/env python
"""Paired A/B/C/D LAP feasibility from identical MuJoCo IC snapshots.

A — Baseline existing Pivot (correct_dtheta)
B — Lever only (axial offset, grip maintained)
C — Full LAP (lever + protected rotation + controlled release)
D — Reverse lever + same protected release protocol

Does not modify production Oracle / datasets. verified_lateral_fallback=false.
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
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import correct_dtheta  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.lever_arm_assisted_pivot.cohorts import paired_cohort  # noqa: E402
from experiments.precision.lever_arm_assisted_pivot.geometry import measure_lever_geometry  # noqa: E402
from experiments.precision.lever_arm_assisted_pivot.lap_primitive import (  # noqa: E402
    run_lap,
    run_lever_only,
)
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/lever_arm_assisted_pivot"


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


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("actions", "series", "capture")}


def summarize(cond: str, res: dict) -> dict:
    return {
        "condition": cond,
        "ok": bool(res.get("ok")),
        "failure": res.get("failure") or res.get("reason"),
        "phase_failed": res.get("phase_failed"),
        "relative_rotation_deg": res.get("relative_rotation_deg"),
        "d_theta_improve_deg": res.get("d_theta_improve_deg"),
        "delta_p_rel_mm": res.get("delta_p_rel_mm"),
        "motion_class": res.get("motion_class"),
        "diag_3mm_ok": res.get("diag_3mm_ok"),
        "n_actions": len(res.get("actions") or []),
        "errors0_dtheta": (res.get("errors0") or {}).get("dtheta_deg"),
        "errors1_dtheta": (res.get("errors1") or res.get("errors") or {}).get("dtheta_deg"),
        "final_alpha": ((res.get("final_posture") or {}).get("alpha_up_deg")),
        "s1_reentry_candidate": res.get("s1_reentry_candidate"),
        "mechanism": res.get("mechanism"),
    }


def run_one(env, row: dict, pcfg: PrecisionConfig, *, lever_mm: float, grip_end: float, release_s: float) -> dict:
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
            "ic_path": row["ic_path"],
            "class": row.get("class"),
            "status": "STAGE1_INELIGIBLE",
            "stage1_eligibility": elig.to_dict(),
            "rejected_reason": f"STAGE1_INELIGIBLE:{elig.reason}",
        }
    plan = str(row.get("plan_family") or "A")
    restore_capture(env, blob["capture"])
    geo_ic = measure_lever_geometry(env.sim)

    # A — baseline Pivot
    restore_capture(env, blob["capture"])
    base = correct_dtheta(env, {"capture": blob["capture"], "ref": blob["ref"]}, pcfg, plan)
    # relative rotation from baseline metrics if present
    met = (base.get("metrics") or {})
    base_rel = met.get("d_axis_deg")
    base_out = {
        **strip(base),
        "relative_rotation_deg": base_rel,
        "d_theta_improve_deg": None,
        "delta_p_rel_mm": met.get("delta_p_rel_mm"),
        "motion_class": met.get("motion_class") or met.get("label"),
        "errors0": {"dtheta_deg": geo_ic.get("dtheta_deg")},
        "errors1": base.get("errors") or {},
    }
    if base.get("errors") and geo_ic.get("dtheta_deg") is not None:
        base_out["d_theta_improve_deg"] = float(geo_ic["dtheta_deg"]) - float(base["errors"].get("dtheta_deg") or geo_ic["dtheta_deg"])

    # B — lever only (explicit offset attempt)
    restore_capture(env, blob["capture"])
    b = run_lever_only(env, blob, lever_mm=lever_mm)

    # C — LAP: search-based lever (None) so Phase A can auto-skip existing large levers
    restore_capture(env, blob["capture"])
    c = run_lap(env, blob, lever_mm=None, grip_end=grip_end, release_s=release_s)

    # D — reverse-sign forced lever + same release protocol
    restore_capture(env, blob["capture"])
    d = run_lap(env, blob, lever_mm=-float(lever_mm), lever_sign=-1.0, grip_end=grip_end, release_s=release_s)

    # Persist action traces (pickle of actions lists)
    trace_dir = OUT / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    sid = row["sample_id"]
    for tag, res in (("A", base), ("B", b), ("C", c), ("D", d)):
        (trace_dir / f"{sid}__{tag}.json").write_text(
            json.dumps(
                jsafe(
                    {
                        "sample_id": sid,
                        "condition": tag,
                        "summary": summarize(tag, res if tag != "A" else base_out),
                        "result": strip(res if tag != "A" else base_out),
                        "n_actions": len(res.get("actions") or []),
                        "phases": (res.get("phases") if tag in "CD" else None),
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        # actions as npy for video replay
        acts = res.get("actions") or []
        if acts:
            np.save(trace_dir / f"{sid}__{tag}__actions.npy", np.asarray(acts, np.float32))

    sum_a = summarize("A", {**base_out, "actions": base.get("actions") or []})
    sum_b = summarize("B", b)
    sum_c = summarize("C", c)
    sum_d = summarize("D", d)
    return {
        "sample_id": sid,
        "ic_path": row["ic_path"],
        "class": row.get("class"),
        "plan_family": plan,
        "entry_dx_mm": row.get("_cohort_dx_mm"),
        "entry_dz_mm": row.get("_cohort_dz_mm"),
        "entry_dtheta_deg": row.get("_cohort_dtheta_deg"),
        "geo_ic": geo_ic,
        "A": sum_a,
        "B": sum_b,
        "C": sum_c,
        "D": sum_d,
        "A_full": strip(base_out),
        "B_full": strip(b),
        "C_full": strip(c),
        "D_full": strip(d),
        "lever_mm": lever_mm,
        "grip_end": grip_end,
        "release_s": release_s,
    }


def write_paired_csv(rows: list[dict], path: Path) -> None:
    flat = []
    for r in rows:
        for cond in ("A", "B", "C", "D"):
            s = r.get(cond) or {}
            flat.append(
                {
                    "sample_id": r["sample_id"],
                    "class": r.get("class"),
                    "entry_dx_mm": r.get("entry_dx_mm"),
                    "entry_dz_mm": r.get("entry_dz_mm"),
                    "entry_dtheta_deg": r.get("entry_dtheta_deg"),
                    "lever_mm": r.get("lever_mm"),
                    **s,
                }
            )
    if not flat:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)


def append_contact_csv(rows: list[dict], path: Path) -> None:
    """Extend CONTACT_FORCE_ANALYSIS with LAP C end-state forces."""
    extra = []
    for r in rows:
        for cond, key in (("C", "C_full"), ("D", "D_full")):
            full = r.get(key) or {}
            geo = full.get("geo1") or full.get("geo") or {}
            prim = geo.get("primary") or {}
            extra.append(
                {
                    "sample_id": r["sample_id"],
                    "phase": f"LAP_{cond}_end",
                    "support_side": geo.get("support_side"),
                    "Fn_z_L": geo.get("Fn_z_L"),
                    "Fn_z_R": geo.get("Fn_z_R"),
                    "fn_L": geo.get("fn_L"),
                    "fn_R": geo.get("fn_R"),
                    "axial_lever_mm": prim.get("axial_lever_mm"),
                    "tau_g": prim.get("tau_g_along_des"),
                    "tau_c": prim.get("tau_c_along_des"),
                    "relative_rotation_deg": full.get("relative_rotation_deg"),
                    "dtheta_change_deg": -(full.get("d_theta_improve_deg") or 0),
                    "rotation_origin": full.get("motion_class"),
                    "torque_hypothesis": full.get("failure"),
                }
            )
    if not extra:
        return
    exists = path.exists()
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(extra[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(extra)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--lever-mm", type=float, default=8.0)
    ap.add_argument("--grip-end", type=float, default=-0.15)
    ap.add_argument("--release-s", type=float, default=2.5)
    ap.add_argument("--sweep", action="store_true", help="small lever/grip sweep on first 4 ICs")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    cohort = paired_cohort(n_dtheta=8, n_large_dz=6, n_success=4)[: args.limit]
    env = make_env()
    pcfg = cfg()
    assert pcfg.use_verified_lateral_fallback is False

    results = []
    print(f"LAP paired: {len(cohort)} ICs @ {datetime.now(timezone.utc).isoformat()}", flush=True)
    for i, row in enumerate(cohort):
        print(f"[{i+1}/{len(cohort)}] {row['sample_id']}", flush=True)
        try:
            results.append(
                run_one(
                    env,
                    row,
                    pcfg,
                    lever_mm=args.lever_mm,
                    grip_end=args.grip_end,
                    release_s=args.release_s,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "sample_id": row["sample_id"],
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                }
            )

    sweep_rows = []
    if args.sweep and cohort:
        print("bounded sweep on first min(4,n) ICs", flush=True)
        for row in cohort[: min(4, len(cohort))]:
            for lev in (5.0, 8.0, 12.0):
                for ge, rs in ((-0.05, 2.0), (-0.25, 3.0)):
                    try:
                        restore_capture(env, pickle.loads((ROOT / row["ic_path"]).read_bytes())["capture"])
                        blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
                        c = run_lap(env, blob, lever_mm=lev, grip_end=ge, release_s=rs)
                        sweep_rows.append(
                            {
                                "sample_id": row["sample_id"],
                                "lever_mm": lev,
                                "grip_end": ge,
                                "release_s": rs,
                                **summarize("C_sweep", c),
                            }
                        )
                    except Exception as exc:
                        sweep_rows.append({"sample_id": row["sample_id"], "lever_mm": lev, "error": str(exc)})

    (OUT / "PAIRED_RESULTS.json").write_text(json.dumps(jsafe(results), indent=2), encoding="utf-8")
    write_paired_csv(results, OUT / "PAIRED_RESULTS.csv")
    append_contact_csv(results, OUT / "CONTACT_FORCE_ANALYSIS.csv")
    if sweep_rows:
        with (OUT / "SWEEP_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader()
            w.writerows(sweep_rows)
    print(f"wrote {OUT / 'PAIRED_RESULTS.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
