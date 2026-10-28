#!/usr/bin/env python
"""Paired A/B/C/D OAR single-cycle feasibility from identical IC snapshots."""

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
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.oscillation_assisted_reorientation.cohorts import oar_cohort  # noqa: E402
from experiments.precision.oscillation_assisted_reorientation.oar_primitive import (  # noqa: E402
    run_multi_cycle,
    run_oar_cycle,
)
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/oscillation_assisted_reorientation"


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


def summarize(res: dict) -> dict:
    return {
        "condition": res.get("condition"),
        "direction": res.get("direction"),
        "ok": bool(res.get("ok")),
        "failure": res.get("failure"),
        "net_relative_rotation_deg": res.get("net_relative_rotation_deg"),
        "peak_relative_rotation_deg": res.get("peak_relative_rotation_deg"),
        "rel_after_forward_deg": res.get("rel_after_forward_deg"),
        "d_theta_improve_deg": res.get("d_theta_improve_deg"),
        "dtheta0_deg": res.get("dtheta0_deg"),
        "dtheta1_deg": res.get("dtheta1_deg"),
        "delta_p_rel_mm": res.get("delta_p_rel_mm"),
        "diag_3mm_ok": res.get("diag_3mm_ok"),
        "tracked": res.get("tracked"),
        "undid_on_reverse": res.get("undid_on_reverse"),
        "bilateral_end": res.get("bilateral_end"),
        "peak_hand_tilt_deg": res.get("peak_hand_tilt_deg"),
        "n_actions": len(res.get("actions") or []),
        "amplitude_deg": res.get("amplitude_deg"),
        "half_period_s": res.get("half_period_s"),
    }


def run_ic(env, row: dict, *, amp: float, half: float, grip_loose: float) -> dict:
    blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
    sid = row["sample_id"]
    from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible

    elig = check_stage1_eligible(
        env,
        {"capture": blob["capture"], "ref": blob["ref"], "sample_id": sid},
        allow_live_verify=True,
    )
    if not elig.eligible:
        return {
            "sample_id": sid,
            "ic_path": row["ic_path"],
            "status": "STAGE1_INELIGIBLE",
            "stage1_eligibility": elig.to_dict(),
            "rejected_reason": f"STAGE1_INELIGIBLE:{elig.reason}",
            "results": {},
        }
    results = {}
    # A/B/C × both directions; D × both grip-phase directions via direction sign unused → run once + flip via condition C-like: use direction only for A–C
    for cond in ("A", "B", "C", "D"):
        for direction in (1.0, -1.0):
            if cond == "D" and direction < 0:
                # D: second pass swaps loose/lock by reusing condition B/C grip timeline with opposite mapping:
                # emulate reverse grip phase by swapping grip_loose/grip_lock via negative direction flag in wrapper
                pass
            restore_capture(env, blob["capture"])
            key = f"{cond}_dir{int(direction)}"
            try:
                if cond == "D" and direction < 0:
                    # grip-only with swapped phase: use condition B profile labels but zero ω already in D;
                    # swap by calling with grip_loose/lock exchanged
                    res = run_oar_cycle(
                        env,
                        blob,
                        condition="D",
                        direction=1.0,
                        amplitude_deg=amp,
                        half_period_s=half,
                        grip_loose=0.95,
                        grip_lock=grip_loose,
                    )
                    res["condition"] = "D"
                    res["direction"] = -1.0
                    res["note"] = "grip_phase_swapped"
                else:
                    res = run_oar_cycle(
                        env,
                        blob,
                        condition=cond,  # type: ignore[arg-type]
                        direction=direction,
                        amplitude_deg=amp,
                        half_period_s=half,
                        grip_loose=grip_loose,
                    )
                results[key] = summarize(res)
                # traces
                tdir = OUT / "traces"
                tdir.mkdir(parents=True, exist_ok=True)
                (tdir / f"{sid}__{key}.json").write_text(
                    json.dumps(
                        jsafe(
                            {
                                "sample_id": sid,
                                "summary": results[key],
                                "series": res.get("series"),
                                "ic": res.get("ic"),
                                "final": res.get("final"),
                            }
                        ),
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                acts = res.get("actions") or []
                if acts:
                    np.save(tdir / f"{sid}__{key}__actions.npy", np.asarray(acts, np.float32))
                results[key]["_full_ok"] = bool(res.get("ok"))
                results[key]["_net"] = res.get("net_relative_rotation_deg")
            except Exception as exc:
                results[key] = {"condition": cond, "direction": direction, "error": str(exc), "traceback": traceback.format_exc(limit=4)}

    # Best single-cycle among A/B/C (for multi-cycle gate)
    best_key, best_net, best_ok = None, 0.0, False
    for k, v in results.items():
        if v.get("error"):
            continue
        net = abs(float(v.get("net_relative_rotation_deg") or 0))
        imp = float(v.get("d_theta_improve_deg") or 0)
        score = net + max(0.0, imp)
        if v.get("ok") or (net >= 1.5 and imp >= 1.0):
            if score > best_net:
                best_net, best_key, best_ok = score, k, bool(v.get("ok"))

    return {
        "sample_id": sid,
        "ic_path": row["ic_path"],
        "class": row.get("class"),
        "entry_dx_mm": row.get("_dx"),
        "entry_dz_mm": row.get("_dz"),
        "entry_dtheta_deg": row.get("_dtheta"),
        "amp_deg": amp,
        "half_period_s": half,
        "results": results,
        "best_key": best_key,
        "best_ok": best_ok,
    }


def flat_rows(paired: list[dict]) -> list[dict]:
    out = []
    for r in paired:
        for key, s in (r.get("results") or {}).items():
            out.append(
                {
                    "sample_id": r["sample_id"],
                    "class": r.get("class"),
                    "entry_dx_mm": r.get("entry_dx_mm"),
                    "entry_dz_mm": r.get("entry_dz_mm"),
                    "entry_dtheta_deg": r.get("entry_dtheta_deg"),
                    "amp_deg": r.get("amp_deg"),
                    "half_period_s": r.get("half_period_s"),
                    "trial_key": key,
                    **{k: v for k, v in s.items() if not str(k).startswith("_")},
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--amp", type=float, default=12.0)
    ap.add_argument("--half", type=float, default=0.6)
    ap.add_argument("--grip-loose", type=float, default=0.05)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--multi-if-ok", action="store_true", default=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    cohort = oar_cohort(n_dtheta=5, n_large_dz=3, n_success=2)[: args.limit]
    env = make_env()

    paired = []
    print(f"OAR single-cycle: {len(cohort)} ICs @ {datetime.now(timezone.utc).isoformat()}", flush=True)
    for i, row in enumerate(cohort):
        print(f"[{i+1}/{len(cohort)}] {row['sample_id']}", flush=True)
        try:
            paired.append(run_ic(env, row, amp=args.amp, half=args.half, grip_loose=args.grip_loose))
        except Exception as exc:
            paired.append({"sample_id": row["sample_id"], "error": str(exc), "traceback": traceback.format_exc(limit=6)})

    # Bounded sweep on first 2 ICs if requested
    sweep = []
    if args.sweep and cohort:
        for row in cohort[:2]:
            for amp in (8.0, 12.0, 16.0):
                for half in (0.45, 0.7):
                    for loose in (0.05, -0.2):
                        blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
                        for cond in ("B", "C"):
                            for direction in (1.0, -1.0):
                                restore_capture(env, blob["capture"])
                                res = run_oar_cycle(
                                    env, blob, condition=cond, direction=direction,  # type: ignore[arg-type]
                                    amplitude_deg=amp, half_period_s=half, grip_loose=loose,
                                )
                                sweep.append(
                                    {
                                        "sample_id": row["sample_id"],
                                        "amp": amp,
                                        "half": half,
                                        "grip_loose": loose,
                                        **summarize(res),
                                    }
                                )

    # Multi-cycle only if any single-cycle ok
    multi = []
    any_ok = any(bool(r.get("best_ok")) for r in paired if not r.get("error"))
    # Also gate on persistent net+improve even if ok flag strict
    candidates = []
    for r in paired:
        if r.get("error"):
            continue
        for k, s in (r.get("results") or {}).items():
            if s.get("error"):
                continue
            if bool(s.get("ok")) or (
                abs(float(s.get("net_relative_rotation_deg") or 0)) >= 1.5
                and float(s.get("d_theta_improve_deg") or 0) >= 1.0
                and bool(s.get("bilateral_end"))
            ):
                candidates.append((r, k, s))
    if args.multi_if_ok and candidates:
        print(f"multi-cycle on {len(candidates)} promising trials", flush=True)
        for r, k, s in candidates[:6]:
            blob = pickle.loads((ROOT / r["ic_path"]).read_bytes())
            cond = str(s.get("condition") or k[0])
            direction = float(s.get("direction") or 1.0)
            restore_capture(env, blob["capture"])
            mc = run_multi_cycle(
                env,
                blob,
                condition=cond,  # type: ignore[arg-type]
                n_cycles=3,
                direction=direction,
                amplitude_deg=float(r.get("amp_deg") or args.amp),
                half_period_s=float(r.get("half_period_s") or args.half),
                grip_loose=args.grip_loose,
            )
            multi.append(
                {
                    "sample_id": r["sample_id"],
                    "from_trial": k,
                    "ok": mc.get("ok"),
                    "n_cycles_run": mc.get("n_cycles_run"),
                    "net_relative_rotation_deg": mc.get("net_relative_rotation_deg"),
                    "d_theta_improve_total": mc.get("d_theta_improve_total"),
                    "delta_p_rel_mm": mc.get("delta_p_rel_mm"),
                    "failure": mc.get("failure"),
                    "cycles": mc.get("cycles"),
                }
            )
            if mc.get("actions"):
                np.save(OUT / "traces" / f"{r['sample_id']}__multi__actions.npy", np.asarray(mc["actions"], np.float32))
    else:
        print("multi-cycle skipped: no single-cycle persistent corrective rotation", flush=True)

    (OUT / "PAIRED_SINGLE_CYCLE.json").write_text(json.dumps(jsafe(paired), indent=2), encoding="utf-8")
    flat = flat_rows(paired)
    if flat:
        with (OUT / "SINGLE_CYCLE_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
    if sweep:
        with (OUT / "SWEEP_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            w.writerows(sweep)
    if multi:
        (OUT / "MULTI_CYCLE_RESULTS.json").write_text(json.dumps(jsafe(multi), indent=2), encoding="utf-8")
        # CSV summary
        mflat = [{k: v for k, v in m.items() if k != "cycles"} for m in multi]
        with (OUT / "MULTI_CYCLE_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(mflat[0].keys()))
            w.writeheader()
            w.writerows(mflat)
    else:
        # explicit empty marker
        (OUT / "MULTI_CYCLE_RESULTS.csv").write_text(
            "note,multi_cycle_skipped_no_single_cycle_success\n", encoding="utf-8"
        )

    print(f"wrote {OUT / 'SINGLE_CYCLE_RESULTS.csv'} any_ok={any_ok} candidates={len(candidates)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
