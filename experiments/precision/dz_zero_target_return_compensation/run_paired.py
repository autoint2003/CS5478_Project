#!/usr/bin/env python
"""Paired V0/V1/V2 DZ zero-target / return-compensation on seven B2 exits.

Does not modify production Oracle. Writes results/diagnostics/dz_zero_target_return_compensation/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.dz_zero_target_return_compensation.experimental_dz import (  # noqa: E402
    EXP_DEADBAND_MM,
    EXP_MAX_SAFE_TRAVEL_MM,
    patched_inward_slip_factory,
)
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/dz_zero_target_return_compensation"
EXIT_CACHE = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
HARD_IDS = [
    "hard_910026",
    "hard_910132",
    "hard_910024",
    "hard_910263",
    "hard_910277",
    "hard_910084",
    "hard_910305",
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


def load_blob(hid: str) -> dict:
    blob = pickle.loads((EXIT_CACHE / f"{hid}.pkl").read_bytes())
    meta = json.loads((EXIT_CACHE / f"{hid}__meta.json").read_text(encoding="utf-8"))
    return {**blob, "sample_id": meta.get("raw_id"), "hard_id": hid, "meta": meta}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


@contextmanager
def patch_dz_variant(variant: str):
    """Patch inward slip + wrap correct_dz_inward for post-return retry (V1/V2)."""
    import data_generation.oracles.precision.corrections.axial_inward as ain
    import data_generation.oracles.precision.mechanics.axial_sliding as asl
    import data_generation.oracles.precision.state_machine as sm

    ctx: dict[str, Any] = {
        "timeline": [],
        "deadband_mm": EXP_DEADBAND_MM,
        "max_safe_travel_mm": EXP_MAX_SAFE_TRAVEL_MM,
        "retry_count": 0,
        "variant": variant,
    }
    orig_slip = asl._inward_slip_staged
    orig_ain_slip = ain._inward_slip_staged
    orig_dz = sm.correct_dz_inward
    orig_ain_dz = ain.correct_dz_inward

    if variant == "V0":
        yield ctx
        return

    patched_slip = patched_inward_slip_factory(variant, ctx)
    asl._inward_slip_staged = patched_slip
    ain._inward_slip_staged = patched_slip

    def wrapped_dz(env, blob, cfg, *, s1_state=None):
        # Capture pre-return dz via monitoring slip result in ctx
        r = orig_ain_dz(env, blob, cfg, s1_state=s1_state)
        actions = list(r.get("actions") or [])
        ctx["first_dz"] = {k: v for k, v in r.items() if k not in ("actions", "capture")}
        slip_m = (ctx.get("last_slip") or {}).get("markers") or {}
        dz_before_return = slip_m.get("dz_after_slip_stop")
        restore_capture(env, r.get("capture") or blob["capture"])
        err_after = measure_errors(env.sim)
        dz_after_return = float(err_after["dz_mm"])
        actual_drift = None if dz_before_return is None else float(dz_after_return - float(dz_before_return))
        ctx["return_metrics"] = {
            "dz_before_return": dz_before_return,
            "dz_after_return": dz_after_return,
            "actual_return_drift_mm": actual_drift,
            "predicted_return_drift_mm": (ctx.get("last_slip") or {}).get("predicted_return_drift_mm"),
            "e_pos_after_return": float(err_after["e_pos_mm"]),
            "dtheta_after_return": float(err_after["dtheta_deg"]),
            "bilateral": bool(err_after.get("bilateral")),
        }

        # Post-return re-evaluation + bounded retry (V1/V2)
        need = (
            err_after.get("bilateral")
            and float(err_after["e_pos_mm"]) > float(cfg.working_pos_mm)
            and abs(float(err_after["dz_mm"])) > float(ctx["deadband_mm"])
            and ctx["retry_count"] < 1
        )
        # Safety: do not retry if |ex| unsafe for inward
        if need and abs(float(err_after["dx_mm"])) > float(getattr(cfg, "dz_safe_ex_mm", 6.0) or 6.0):
            ctx["retry_blocked"] = f"lateral_ex_too_large:{err_after['dx_mm']:.2f}"
            need = False

        if need:
            ctx["retry_count"] += 1
            # V2: use measured drift from this episode's return (past, not future)
            if variant == "V2" and actual_drift is not None and abs(actual_drift) >= 0.5:
                ctx["force_target_on_retry"] = float(np.clip(-actual_drift, -8.0, 8.0))
            else:
                ctx["force_target_on_retry"] = 0.0

            # For retry, temporarily force V1-style target via ctx for slip factory
            # Re-bind slip to honor force_target
            def retry_slip(env2, blob2, z_ref, e_x0, *, milestone_mm=12.0, cfg=None):
                from experiments.precision.dz_zero_target_return_compensation.experimental_dz import (
                    experimental_inward_slip,
                )

                tgt = float(ctx.get("force_target_on_retry", 0.0))
                res = experimental_inward_slip(
                    env2,
                    blob2,
                    cfg or PrecisionConfig(),
                    dz_target_mm=tgt,
                    deadband_mm=float(ctx["deadband_mm"]),
                    max_safe_travel_mm=float(ctx["max_safe_travel_mm"]),
                    velocity_aware_brake=True,
                    timeline=ctx.setdefault("timeline", []),
                )
                res["disp_total_mm"] = res.get("disp_along_mm")
                res["reached"] = bool(res.get("goal_reached"))
                res["brake_ok"] = True
                res["regrip_ok"] = True
                res["errors"] = res.get("errors1")
                ctx["last_slip_retry"] = {k: v for k, v in res.items() if k not in ("actions", "timeline", "capture")}
                return res

            asl._inward_slip_staged = retry_slip
            ain._inward_slip_staged = retry_slip
            try:
                r2 = orig_ain_dz(
                    env,
                    {"capture": take_capture(env), "ref": blob.get("ref")},
                    cfg,
                    s1_state=None,
                )
            finally:
                asl._inward_slip_staged = patched_slip
                ain._inward_slip_staged = patched_slip

            if r2.get("ok"):
                actions += list(r2.get("actions") or [])
                r = {
                    **{k: v for k, v in r.items() if k not in ("actions", "capture")},
                    "ok": True,
                    "actions": actions,
                    "capture": r2.get("capture") or r.get("capture"),
                    "retry": {k: v for k, v in r2.items() if k not in ("actions", "capture")},
                    "experimental_retry": True,
                }
                restore_capture(env, r["capture"])
                err3 = measure_errors(env.sim)
                ctx["return_metrics_after_retry"] = {
                    "dz": float(err3["dz_mm"]),
                    "e_pos": float(err3["e_pos_mm"]),
                    "dtheta": float(err3["dtheta_deg"]),
                }
            else:
                # Keep first result; record failed retry
                ctx["retry_failed"] = r2.get("reason") or r2.get("stop_reason")
                r = {**r, "actions": actions, "experimental_retry_failed": ctx["retry_failed"]}

        r["experimental_ctx"] = {
            "variant": variant,
            "retry_count": ctx["retry_count"],
            "return_metrics": ctx.get("return_metrics"),
            "last_slip": ctx.get("last_slip"),
            "budget_max_safe_travel_mm": ctx["max_safe_travel_mm"],
            "deadband_mm": ctx["deadband_mm"],
        }
        return r

    sm.correct_dz_inward = wrapped_dz
    ain.correct_dz_inward = wrapped_dz
    try:
        yield ctx
    finally:
        asl._inward_slip_staged = orig_slip
        ain._inward_slip_staged = orig_ain_slip
        sm.correct_dz_inward = orig_dz
        ain.correct_dz_inward = orig_ain_dz


def pack_result(hid: str, variant: str, res, ctx: dict | None, entry: dict) -> dict:
    hand = res.handoff or {}
    ee = res.errors_exit or {}
    logs = res.primitive_logs or {}
    exp = None
    for key in ("dz", "dz_inward"):
        if isinstance(logs.get(key), dict) and logs[key].get("experimental_ctx"):
            exp = logs[key]["experimental_ctx"]
            break
    # Also search nested
    if exp is None:
        for v in logs.values():
            if isinstance(v, dict) and v.get("experimental_ctx"):
                exp = v["experimental_ctx"]
                break
    rm = (ctx or {}).get("return_metrics") or (exp or {}).get("return_metrics") or {}
    slip = (ctx or {}).get("last_slip") or (exp or {}).get("last_slip") or {}
    markers = slip.get("markers") or {}
    return {
        "hard_id": hid,
        "variant": variant,
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "status": res.status,
        "dropped": not bool(ee.get("bilateral", True)),
        "final_dx_mm": float(ee.get("dx_mm") or 0),
        "final_dy_mm": float(ee.get("dy_mm") or 0),
        "final_dz_mm": float(ee.get("dz_mm") or 0),
        "final_e_pos_mm": float(ee.get("e_pos_mm") or 0),
        "final_dtheta_deg": float(ee.get("dtheta_deg") or 0),
        "entry_dz_mm": entry.get("dz"),
        "entry_e_pos_mm": entry.get("e_pos"),
        "dz_at_brake_start": markers.get("dz_at_brake_start"),
        "dz_after_slip_stop": markers.get("dz_after_slip_stop"),
        "dz_before_return": rm.get("dz_before_return"),
        "dz_after_return": rm.get("dz_after_return"),
        "predicted_return_drift_mm": rm.get("predicted_return_drift_mm")
        or slip.get("predicted_return_drift_mm"),
        "actual_return_drift_mm": rm.get("actual_return_drift_mm"),
        "overshoot_mm": None
        if markers.get("dz_after_slip_stop") is None
        else float(markers.get("dz_after_slip_stop") or 0),  # signed residual vs 0
        "pre_return_target_mm": slip.get("pre_return_target_mm"),
        "compensation_applied": slip.get("compensation_applied"),
        "retry_count": (ctx or {}).get("retry_count") or (exp or {}).get("retry_count") or 0,
        "contact_loss": "CONTACT" in str(res.failures or []),
        "n_actions": len(res.actions or []),
        "correction_duration_s": float(len(res.actions or [])) * 0.02,
        "max_safe_travel_mm": (ctx or {}).get("max_safe_travel_mm")
        if variant != "V0"
        else 12.0,
        "phases": list(res.phases or []),
        "failures": list(res.failures or []),
        "brake_mode": markers.get("brake_mode"),
        "slip_stop_reason": slip.get("stop_reason"),
    }


def run_one(env, hid: str, pcfg: PrecisionConfig) -> dict:
    blob = load_blob(hid)
    print(f"\n=== {hid} ===", flush=True)
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    entry = {
        "dx": float(err0["dx_mm"]),
        "dy": float(err0["dy_mm"]),
        "dz": float(err0["dz_mm"]),
        "e_pos": float(err0["e_pos_mm"]),
        "dtheta": float(err0["dtheta_deg"]),
    }
    case = {"hard_id": hid, "entry": entry, "variants": {}}
    all_timeline_rows = []

    for variant in ("V0", "V1", "V2"):
        print(f"  [{hid}] {variant}...", flush=True)
        try:
            with patch_dz_variant(variant) as ctx:
                restore_capture(env, blob["capture"])
                res = run_precision(
                    env,
                    {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob["sample_id"]},
                    "B",
                    pcfg,
                )
                packed = pack_result(hid, variant, res, ctx, entry)
                case["variants"][variant] = packed
                np.savez_compressed(
                    OUT / "raw" / f"{hid}__{variant}__actions.npz",
                    actions=np.asarray(res.actions or [], np.float32),
                )
                for row in ctx.get("timeline") or []:
                    all_timeline_rows.append({"hard_id": hid, "variant": variant, **row})
                # For V0, also dump a coarse end marker
                if variant == "V0":
                    all_timeline_rows.append(
                        {
                            "hard_id": hid,
                            "variant": "V0",
                            "t": packed["correction_duration_s"],
                            "phase": "exit",
                            "dz_mm": packed["final_dz_mm"],
                            "e_pos_mm": packed["final_e_pos_mm"],
                            "dtheta_deg": packed["final_dtheta_deg"],
                        }
                    )
                print(
                    f"    NEW={packed['SUCCESS_NEW']} 12S={packed['SUCCESS_12S']} "
                    f"dz={packed['final_dz_mm']:.2f} e_pos={packed['final_e_pos_mm']:.2f} "
                    f"retry={packed['retry_count']}",
                    flush=True,
                )
        except Exception as exc:
            case["variants"][variant] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"    ERROR {exc}", flush=True)

    case["timeline_rows"] = all_timeline_rows
    return case


def write_reports(cases: list[dict], paired_rows: list[dict]) -> None:
    audit = """# DZ_CONTROL_CODE_AUDIT

## Production (V0) inward DZ

| Step | Location | Behavior |
|---|---|---|
| Axial error | hand-frame signed `e_z` (mm) via `measure_errors` / DualLoop `e_signed_mm[2]` | +ez = object COM offset along hand open |
| Target | implicit 0 | desire `-U_OPEN_HAND` when inward |
| Stopping tol | `axial_sliding._inward_slip_staged` | `axis_tol=clip(sqrt(12²-ex²-ey²),3,8)`, `use_euclidean_stop=False` |
| Max travel | `cfg.inward_slip_mm` default **12** | safety cap — observed `MAX_SAFE_TRAVEL` on hard_910305 |
| Slip | DualLoop force/slip | |
| Brake | fixed 0.45s + 0.25s regrip after loop exit | not velocity-scheduled to a pose target |
| Return | `correct_dz_inward` return DOWN | can change ez (drift) |
| Phase done | slip ok → return → SM `done_dz=True` | blocks second DZ even if `e_pos>12` while `|ez|<20` |

## Residual ~15–16 mm (hard_910305)

From prior diagnostic + V0 logs: slip often ends by **`MAX_SAFE_TRAVEL` (12 mm)**, not by the 3–8 mm band; return/regrip then leaves final `|dz|~15–16`. So tolerance formula alone does **not** fully explain the residual — travel budget + return drift + SM no-retry co-cause.

## Experimental V1 / V2

- Target: explicit `dz_target_mm` (default 0); deadband ~1.5 mm
- Decoupled from SUCCESS_NEW 12 mm threshold
- `max_safe_travel_mm=20` (reported budget delta vs production 12)
- Velocity-aware brake using `|v_axial|*t_brake`
- V2: decision-time return probe (simulate+restore) → signed pre-return target `-Δ̂ez`; fallback to zero if unreliable; post-return bounded retry using measured drift
"""
    (OUT / "DZ_CONTROL_CODE_AUDIT.md").write_text(audit, encoding="utf-8")

    # RETURN_DRIFT_ANALYSIS
    lines = [
        "# RETURN_DRIFT_ANALYSIS",
        "",
        "| hard_id | var | dz_before_return | dz_after_return | predicted | actual | compensation |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in paired_rows:
        lines.append(
            f"| `{r.get('hard_id')}` | {r.get('variant')} | {r.get('dz_before_return')} | "
            f"{r.get('dz_after_return')} | {r.get('predicted_return_drift_mm')} | "
            f"{r.get('actual_return_drift_mm')} | {r.get('compensation_applied')} |"
        )
    (OUT / "RETURN_DRIFT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CONTACT
    cl = [
        "# CONTACT_STABILITY_ANALYSIS",
        "",
        "| hard_id | var | dropped | contact_loss_flag | SUCCESS_12S | retry |",
        "|---|---|---|---|---|---:|",
    ]
    for r in paired_rows:
        cl.append(
            f"| `{r.get('hard_id')}` | {r.get('variant')} | {r.get('dropped')} | "
            f"{r.get('contact_loss')} | {r.get('SUCCESS_12S')} | {r.get('retry_count')} |"
        )
    (OUT / "CONTACT_STABILITY_ANALYSIS.md").write_text("\n".join(cl) + "\n", encoding="utf-8")

    # REGRESSION
    rg = [
        "# REGRESSION_RESULTS",
        "",
        "Focus: preserve `hard_910026` SUCCESS_NEW; improve 910305/910084 if possible.",
        "",
        "| hard_id | V0 NEW | V1 NEW | V2 NEW | V0 dz | V1 dz | V2 dz |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    by = {}
    for r in paired_rows:
        by.setdefault(r["hard_id"], {})[r["variant"]] = r
    for hid in HARD_IDS:
        m = by.get(hid) or {}
        rg.append(
            f"| `{hid}` | {(m.get('V0') or {}).get('SUCCESS_NEW')} | {(m.get('V1') or {}).get('SUCCESS_NEW')} | "
            f"{(m.get('V2') or {}).get('SUCCESS_NEW')} | {(m.get('V0') or {}).get('final_dz_mm')} | "
            f"{(m.get('V1') or {}).get('final_dz_mm')} | {(m.get('V2') or {}).get('final_dz_mm')} |"
        )
    v0_026 = (by.get("hard_910026") or {}).get("V0") or {}
    v1_026 = (by.get("hard_910026") or {}).get("V1") or {}
    v2_026 = (by.get("hard_910026") or {}).get("V2") or {}
    rg += [
        "",
        f"- hard_910026 V0 NEW={v0_026.get('SUCCESS_NEW')} V1={v1_026.get('SUCCESS_NEW')} V2={v2_026.get('SUCCESS_NEW')}",
        "",
    ]
    (OUT / "REGRESSION_RESULTS.md").write_text("\n".join(rg), encoding="utf-8")

    def count(var, key="SUCCESS_NEW"):
        return sum(1 for r in paired_rows if r.get("variant") == var and r.get(key) is True)

    recovers = []
    for hid in HARD_IDS:
        m = by.get(hid) or {}
        if hid == "hard_910026":
            continue
        for var in ("V1", "V2"):
            if not (m.get("V0") or {}).get("SUCCESS_NEW") and (m.get(var) or {}).get("SUCCESS_NEW"):
                recovers.append((var, hid))
    regress_026 = []
    if v0_026.get("SUCCESS_NEW") and not v1_026.get("SUCCESS_NEW"):
        regress_026.append("V1")
    if v0_026.get("SUCCESS_NEW") and not v2_026.get("SUCCESS_NEW"):
        regress_026.append("V2")

    prefer = "neither"
    if count("V2") > count("V1") and count("V2") > count("V0") and not regress_026:
        prefer = "V2"
    elif count("V1") >= count("V2") and count("V1") > count("V0") and "V1" not in regress_026:
        prefer = "V1"

    adopt = prefer in ("V1", "V2") and not regress_026 and (count(prefer) > count("V0"))

    fr = f"""# FINAL_RECOMMENDATION — Zero-Target / Return-Compensated DZ

Production Oracle **not** modified in this experiment.

## Counts (7 B2 ICs)

| Variant | SUCCESS_NEW | SUCCESS_12S |
|---|---:|---:|
| V0 production | {count('V0')}/7 | {count('V0','SUCCESS_12S')}/7 |
| V1 zero-target | {count('V1')}/7 | {count('V1','SUCCESS_12S')}/7 |
| V2 return-compensated | {count('V2')}/7 | {count('V2','SUCCESS_12S')}/7 |

- Recoveries of prior fails: {recovers or 'none'}
- Regressions of hard_910026: {regress_026 or 'none'}
- Preferred experimental controller: **{prefer}**

## Recommendation: **{'CONDITIONAL ADOPT ' + prefer if adopt else 'DO NOT INTEGRATE into production'}**

Acceptance requires SUCCESS_NEW improvement, no 910026 regression, legal actions, reproducible.
Budget note: V1/V2 use `max_safe_travel_mm={EXP_MAX_SAFE_TRAVEL_MM}` vs production 12 mm.

If V1 ≥ V2 on SUCCESS_NEW, prefer simpler zero-target (V1).
"""
    (OUT / "FINAL_RECOMMENDATION.md").write_text(fr, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="", help="comma-separated hard_ids")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    ids = HARD_IDS
    if args.only:
        ids = [x.strip() for x in args.only.split(",") if x.strip()]

    env = make_env()
    pcfg = cfg()
    cases = []
    paired = []
    timelines = []
    for hid in ids:
        if not (EXIT_CACHE / f"{hid}.pkl").is_file():
            print(f"missing exit {hid}", flush=True)
            continue
        case = run_one(env, hid, pcfg)
        cases.append(jsafe({k: v for k, v in case.items() if k != "timeline_rows"}))
        timelines.extend(case.get("timeline_rows") or [])
        for var in ("V0", "V1", "V2"):
            v = case.get("variants", {}).get(var) or {}
            if "error" not in v:
                paired.append(v)

    (OUT / "PAIRED_CASES.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    write_csv(OUT / "PAIRED_RESULTS.csv", paired)
    write_csv(OUT / "DZ_TIMELINES.csv", timelines)
    write_reports(cases, paired)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(ids),
        "v0_new": sum(1 for r in paired if r.get("variant") == "V0" and r.get("SUCCESS_NEW")),
        "v1_new": sum(1 for r in paired if r.get("variant") == "V1" and r.get("SUCCESS_NEW")),
        "v2_new": sum(1 for r in paired if r.get("variant") == "V2" and r.get("SUCCESS_NEW")),
    }
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
