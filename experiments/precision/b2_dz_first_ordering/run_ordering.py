#!/usr/bin/env python
"""S0–S3 DZ-first ordering on seven verified B2 Stage-1 exits.

S0 = production run_precision (unchanged).
S1–S3 = experimental ordering; shared zero-target DZ (no MAX_SAFE_TRAVEL).
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
from data_generation.oracles.precision.entry_conditions import assess_need  # noqa: E402
from data_generation.oracles.precision.mechanics.support_transfer import (  # noqa: E402
    correct_dx_support_transfer,
    restore_fingertips_down,
)
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.handoff import run_final_down_handoff_gate  # noqa: E402
from data_generation.oracles.precision.state_machine import correct_dtheta, run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.b2_dz_first_ordering.coords_and_orientation import (  # noqa: E402
    analyze_dx_gravity,
    prepare_favorable_orientation,
)
from experiments.precision.b2_dz_first_ordering.dz_controller import run_zero_target_dz  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/b2_dz_first_ordering"
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
PRIMARY = ["hard_910305", "hard_910084", "hard_910026"]


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


def err_pack(env) -> dict:
    e = measure_errors(env.sim)
    return {
        "dx": float(e["dx_mm"]),
        "dy": float(e["dy_mm"]),
        "dz": float(e["dz_mm"]),
        "e_pos": float(e["e_pos_mm"]),
        "dtheta": float(e["dtheta_deg"]),
        "bilateral": bool(e.get("bilateral")),
    }


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


def _entry_settle(env, blob, cfg, actions: list) -> dict:
    restore_capture(env, blob["capture"])
    for _ in range(int(0.20 / 0.02)):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        step_legal(env.sim, a)
        actions.append(a.copy())
    return {"capture": take_capture(env), "ref": blob["ref"]}


def run_S0(env, blob, pcfg) -> dict:
    restore_capture(env, blob["capture"])
    res = run_precision(
        env,
        {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob.get("sample_id")},
        "B",
        pcfg,
    )
    hand = res.handoff or {}
    return {
        "strategy": "S0",
        "status": res.status,
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "phases": list(res.phases or []),
        "failures": list(res.failures or []),
        "n_actions": len(res.actions or []),
        "actions": list(res.actions or []),
        "errors_exit": err_pack(env) if res.capture is None else None,
        "final": {
            "dx": float((res.errors_exit or {}).get("dx_mm") or 0),
            "dy": float((res.errors_exit or {}).get("dy_mm") or 0),
            "dz": float((res.errors_exit or {}).get("dz_mm") or 0),
            "e_pos": float((res.errors_exit or {}).get("e_pos_mm") or 0),
            "dtheta": float((res.errors_exit or {}).get("dtheta_deg") or 0),
            "bilateral": bool((res.errors_exit or {}).get("bilateral", True)),
        },
        "phase_traces": [],
        "dz_n_attempts": None,
        "dz_total_disp_mm": None,
        "used_max_safe_travel": None,
        "orientation": None,
        "dtheta_after_dx": None,
        "max_dtheta_deterioration": None,
    }


def run_experimental(env, blob, pcfg, strategy: str) -> dict:
    """S1 / S2 / S3 custom ordering with shared zero-target DZ."""
    actions: list = []
    phases: list[str] = []
    failures: list[str] = []
    traces: list[dict] = []
    timeline: list = []
    budget = int(pcfg.global_action_budget)

    elig = check_stage1_eligible(
        env,
        {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob.get("sample_id")},
        allow_live_verify=True,
    )
    if not elig.eligible:
        return {
            "strategy": strategy,
            "status": "STAGE1_INELIGIBLE",
            "SUCCESS_NEW": False,
            "SUCCESS_12S": False,
            "failures": [f"STAGE1_INELIGIBLE:{elig.reason}"],
            "actions": [],
            "n_actions": 0,
            "phase_traces": [],
        }

    cur = _entry_settle(env, blob, pcfg, actions)
    phases.append("entry_settle")
    init = err_pack(env)
    grav0 = analyze_dx_gravity(env.sim)
    traces.append({"phase": "init", **init, **{k: grav0[k] for k in ("adverse_g_along_dx_offset_mps2", "favorable", "offset_points_down_score")}})

    need = assess_need(measure_errors(env.sim), pcfg)
    orientation_info = None
    dtheta0 = float(init["dtheta"])

    # Preserve dθ first when needed (safety / existing skill dependency)
    if need.get("dtheta_needed"):
        phases.append("correct_dtheta")
        rd = correct_dtheta(env, cur, pcfg, "B")
        actions += list(rd.get("actions") or [])
        if not rd.get("ok"):
            failures.append(f"dtheta:{rd.get('reason')}")
            return _fail(strategy, phases, failures, actions, traces, "PRECISION_PRIMITIVE_FAILED")
        cur = {"capture": rd["capture"], "ref": blob["ref"]}
        traces.append({"phase": "after_dtheta", **err_pack(env)})
        need = assess_need(measure_errors(env.sim), pcfg)

    # S2/S3: favorable orientation before DZ
    if strategy in ("S2", "S3") and (need.get("dz_needed") or abs(float(err_pack(env)["dz"])) > 2.0):
        phases.append("favorable_orientation")
        orientation_info = prepare_favorable_orientation(env, cur)
        actions += list(orientation_info.get("actions") or [])
        if orientation_info.get("ok") and orientation_info.get("capture"):
            cur = {"capture": orientation_info["capture"], "ref": blob["ref"]}
        traces.append(
            {
                "phase": "after_orientation",
                **err_pack(env),
                "favorable": (orientation_info.get("after") or {}).get("favorable"),
                "adverse": (orientation_info.get("after") or {}).get("adverse_g_along_dx_offset_mps2"),
                "orient_skipped": orientation_info.get("skipped"),
                "orient_reason": orientation_info.get("reason"),
            }
        )
        need = assess_need(measure_errors(env.sim), pcfg)

    dz_stats = {"n_attempts": 0, "total_disp_mm": 0.0, "used_max_safe_travel": False}
    dtheta_before_dx = None
    dtheta_after_dx = None

    # DZ first (coarse)
    if need.get("dz_needed") or abs(float(err_pack(env)["dz"])) > 0.25 * float(pcfg.ez_tol_mm):
        # Production dependency: inward DZ needs |ex|<=dz_safe_ex. If violated, report and optionally ST first.
        ex = abs(float(err_pack(env)["dx"]))
        dz_safe = float(getattr(pcfg, "dz_safe_ex_mm", 6.0) or 6.0)
        if ex > dz_safe:
            # Explicit dependency: cannot legally run inward DZ with large ex
            phases.append("dz_blocked_ex_too_large")
            failures.append(f"dz_requires_prior_dx_safe_band:ex={ex:.2f}>{dz_safe:.2f}")
            # Fall back: do DX (ST) first then DZ — but label that S1/S2 could not pure-reorder
            phases.append("fallback_dx_before_dz_due_to_safety")
            rst = correct_dx_support_transfer(env, cur, pcfg)
            actions += list(rst.get("actions") or [])
            if not rst.get("ok") and not rst.get("skipped"):
                failures.append(f"support_transfer_dx:{rst.get('reason')}")
                return _fail(strategy, phases, failures, actions, traces, "PRECISION_PRIMITIVE_FAILED", orientation_info)
            cur = {"capture": rst.get("capture") or take_capture(env), "ref": blob["ref"]}
            traces.append({"phase": "after_fallback_dx", **err_pack(env)})
            # restore not required if S1 kept for dz — keep S1 for inward
            s1 = rst.get("s1_state")
        else:
            s1 = None

        phases.append("dz_coarse_zero_target")
        rem = max(100, budget - len(actions))
        dzr = run_zero_target_dz(
            env,
            cur,
            pcfg,
            dz_target_mm=0.0,
            action_budget_left=rem,
            timeline=timeline,
            label="dz_coarse",
        )
        actions += list(dzr.get("actions") or [])
        dz_stats["n_attempts"] += int(dzr.get("n_attempts") or 0)
        dz_stats["total_disp_mm"] += float(dzr.get("total_disp_mm") or 0)
        if not dzr.get("ok"):
            failures.append(f"dz_coarse:{dzr.get('reason') or dzr.get('stop_reason')}")
            return _fail(strategy, phases, failures, actions, traces, "PRECISION_PRIMITIVE_FAILED", orientation_info, dz_stats)
        cur = {"capture": dzr["capture"], "ref": blob["ref"]}
        traces.append({"phase": "after_dz_coarse", **err_pack(env), "n_attempts": dzr.get("n_attempts"), "disp": dzr.get("total_disp_mm")})

        # Return DOWN before DX if tipped (DX ST expects workable posture; ST establishes S1 itself)
        if not measure_errors(env.sim).get("fingertips_down"):
            phases.append("restore_DOWN_before_dx")
            rd = restore_fingertips_down(env, cur, pcfg, rate_scale=float(pcfg.rate_restore_down))
            actions += list(rd.get("actions") or [])
            if not rd.get("ok"):
                failures.append(f"restore_DOWN:{rd.get('reason')}")
                return _fail(strategy, phases, failures, actions, traces, "PRECISION_PRIMITIVE_FAILED", orientation_info, dz_stats)
            cur = {"capture": rd["capture"], "ref": blob["ref"]}
            traces.append({"phase": "after_return_before_dx", **err_pack(env)})

    need = assess_need(measure_errors(env.sim), pcfg)
    dtheta_before_dx = float(err_pack(env)["dtheta"])

    # DX (production ST)
    if need.get("dx_needed") or abs(float(err_pack(env)["dx"])) > float(pcfg.ex_tol_mm) * 0.25:
        phases.append("dx_support_transfer")
        rst = correct_dx_support_transfer(env, cur, pcfg)
        actions += list(rst.get("actions") or [])
        if not rst.get("ok") and not rst.get("skipped"):
            failures.append(f"support_transfer_dx:{rst.get('reason')}")
            return _fail(strategy, phases, failures, actions, traces, "PRECISION_PRIMITIVE_FAILED", orientation_info, dz_stats)
        cur = {"capture": rst.get("capture") or take_capture(env), "ref": blob["ref"]}
        dtheta_after_dx = float(err_pack(env)["dtheta"])
        traces.append(
            {
                "phase": "after_dx",
                **err_pack(env),
                "dtheta_delta_during_dx": dtheta_after_dx - dtheta_before_dx,
            }
        )
        # Keep S1 for optional fine DZ under support if S3
        s1_after_dx = rst.get("s1_state")
    else:
        s1_after_dx = None
        dtheta_after_dx = dtheta_before_dx

    # S3: final DZ refinement
    if strategy == "S3":
        need = assess_need(measure_errors(env.sim), pcfg)
        ep = err_pack(env)
        if ep["e_pos"] > float(pcfg.working_pos_mm) or abs(ep["dz"]) > DEADBAND():
            if abs(ep["dx"]) > float(getattr(pcfg, "dz_safe_ex_mm", 6.0) or 6.0):
                phases.append("dz_fine_skipped_ex_unsafe")
            elif len(actions) < budget - 50:
                phases.append("dz_fine_zero_target")
                # May need UP — use production inward path only if we have S1; else our slip assumes current posture
                # For tipped S1 after DX, experimental slip along open still works when UP-ish; if DOWN, run_zero_target still uses signed desire
                dzf = run_zero_target_dz(
                    env,
                    cur,
                    pcfg,
                    dz_target_mm=0.0,
                    action_budget_left=budget - len(actions),
                    timeline=timeline,
                    label="dz_fine",
                )
                actions += list(dzf.get("actions") or [])
                dz_stats["n_attempts"] += int(dzf.get("n_attempts") or 0)
                dz_stats["total_disp_mm"] += float(dzf.get("total_disp_mm") or 0)
                if not dzf.get("ok"):
                    failures.append(f"dz_fine:{dzf.get('reason') or dzf.get('stop_reason')}")
                    # non-fatal if grasp retained — continue to handoff attempt
                    if not measure_errors(env.sim).get("bilateral"):
                        return _fail(strategy, phases, failures, actions, traces, "PRECISION_PRIMITIVE_FAILED", orientation_info, dz_stats)
                else:
                    cur = {"capture": dzf["capture"], "ref": blob["ref"]}
                traces.append({"phase": "after_dz_fine", **err_pack(env)})

    # Final DOWN + handoff
    phases.append("final_down_gate")
    gate = run_final_down_handoff_gate(env, cur, pcfg, actions)
    if not gate.get("ok"):
        failures.append(f"final_down_gate:{gate.get('failure_code')}:{gate.get('reason')}")
        return _fail(strategy, phases, failures, actions, traces, "PRECISION_PRIMITIVE_FAILED", orientation_info, dz_stats, dtheta_before_dx, dtheta_after_dx)
    cur = {"capture": gate["capture"], "ref": blob["ref"]}
    restore_capture(env, cur["capture"])
    hand = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=pcfg.stabilize_handoff_n)
    restore_capture(env, cur["capture"])
    final = err_pack(env)
    traces.append({"phase": "final", **final})
    success = bool(hand.get("SUCCESS_NEW"))
    status = "PRECISION_EXECUTED_SUCCESS" if success else "PRECISION_EXECUTED_HANDOFF_FAIL"
    if hand and not hand.get("SUCCESS_NEW"):
        failures.append(f"handoff_NEW_fail:12S={hand.get('SUCCESS_12S')}")
    return {
        "strategy": strategy,
        "status": status,
        "SUCCESS_NEW": success,
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "phases": phases,
        "failures": failures,
        "n_actions": len(actions),
        "actions": actions,
        "final": final,
        "phase_traces": traces,
        "dz_n_attempts": dz_stats["n_attempts"],
        "dz_total_disp_mm": dz_stats["total_disp_mm"],
        "used_max_safe_travel": False,
        "orientation": {k: v for k, v in (orientation_info or {}).items() if k != "actions"},
        "dtheta_before_dx": dtheta_before_dx,
        "dtheta_after_dx": dtheta_after_dx,
        "dtheta_delta_during_dx": None
        if dtheta_before_dx is None or dtheta_after_dx is None
        else float(dtheta_after_dx - dtheta_before_dx),
        "init": init,
        "grav0": grav0,
        "timeline": timeline,
    }


def DEADBAND():
    return 1.5


def _fail(strategy, phases, failures, actions, traces, status, orientation=None, dz_stats=None, dth0=None, dth1=None):
    return {
        "strategy": strategy,
        "status": status,
        "SUCCESS_NEW": False,
        "SUCCESS_12S": False,
        "phases": phases,
        "failures": failures,
        "n_actions": len(actions),
        "actions": actions,
        "final": traces[-1] if traces else {},
        "phase_traces": traces,
        "dz_n_attempts": (dz_stats or {}).get("n_attempts"),
        "dz_total_disp_mm": (dz_stats or {}).get("total_disp_mm"),
        "used_max_safe_travel": False,
        "orientation": {k: v for k, v in (orientation or {}).items() if k != "actions"} if orientation else None,
        "dtheta_before_dx": dth0,
        "dtheta_after_dx": dth1,
        "dtheta_delta_during_dx": None if dth0 is None or dth1 is None else float(dth1 - dth0),
    }


def row_from(hid: str, packed: dict) -> dict:
    fin = packed.get("final") or {}
    return {
        "hard_id": hid,
        "strategy": packed.get("strategy"),
        "SUCCESS_NEW": packed.get("SUCCESS_NEW"),
        "SUCCESS_12S": packed.get("SUCCESS_12S"),
        "status": packed.get("status"),
        "final_dx": fin.get("dx"),
        "final_dy": fin.get("dy"),
        "final_dz": fin.get("dz"),
        "final_e_pos": fin.get("e_pos"),
        "final_dtheta": fin.get("dtheta"),
        "n_actions": packed.get("n_actions"),
        "dz_n_attempts": packed.get("dz_n_attempts"),
        "dz_total_disp_mm": packed.get("dz_total_disp_mm"),
        "used_max_safe_travel": packed.get("used_max_safe_travel"),
        "dtheta_delta_during_dx": packed.get("dtheta_delta_during_dx"),
        "orient_favorable_after": ((packed.get("orientation") or {}).get("favorable_after")),
        "orient_reason": ((packed.get("orientation") or {}).get("reason")),
        "failures": "|".join(packed.get("failures") or []),
        "phases": "|".join(packed.get("phases") or []),
    }


def run_case(env, hid: str, pcfg: PrecisionConfig) -> dict:
    blob = load_blob(hid)
    print(f"\n=== {hid} ===", flush=True)
    restore_capture(env, blob["capture"])
    elig = check_stage1_eligible(
        env, {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob["sample_id"]}, allow_live_verify=True
    )
    case = {
        "hard_id": hid,
        "stage1_eligible": elig.to_dict(),
        "stage1_SUCCESS_12S": bool((elig.to_dict() or {}).get("SUCCESS_12S") or elig.eligible),
        "variants": {},
    }
    phase_rows = []
    for strategy in ("S0", "S1", "S2", "S3"):
        print(f"  [{hid}] {strategy}...", flush=True)
        try:
            if strategy == "S0":
                packed = run_S0(env, blob, pcfg)
            else:
                packed = run_experimental(env, blob, pcfg, strategy)
            # strip actions for JSON size; save npz
            acts = packed.pop("actions", [])
            tl = packed.pop("timeline", [])
            np.savez_compressed(OUT / "raw" / f"{hid}__{strategy}__actions.npz", actions=np.asarray(acts, np.float32))
            case["variants"][strategy] = jsafe(packed)
            for tr in packed.get("phase_traces") or []:
                phase_rows.append({"hard_id": hid, "strategy": strategy, **tr})
            for row in tl:
                phase_rows.append({"hard_id": hid, "strategy": strategy, "trace": "dz_timeline", **row})
            print(
                f"    NEW={packed.get('SUCCESS_NEW')} 12S={packed.get('SUCCESS_12S')} "
                f"dz={((packed.get('final') or {}).get('dz'))} e_pos={((packed.get('final') or {}).get('e_pos'))} "
                f"dth_dxΔ={packed.get('dtheta_delta_during_dx')}",
                flush=True,
            )
        except Exception as exc:
            case["variants"][strategy] = {"strategy": strategy, "status": "ERROR", "error": str(exc), "traceback": traceback.format_exc()}
            print(f"    ERROR {exc}", flush=True)
    case["phase_rows"] = phase_rows
    return case


def write_reports(cases: list, order_rows: list) -> None:
    # COORDINATE
    (OUT / "COORDINATE_AND_GRAVITY_ANALYSIS.md").write_text(
        """# COORDINATE_AND_GRAVITY_ANALYSIS

## Conventions

| Symbol | Meaning |
|---|---|
| DX / `e_signed_mm[0]` | Hand-frame lateral COM residual (mm) |
| DZ / `e_signed_mm[2]` | Hand-frame axial COM residual (mm) |
| Hand X | Lateral axis (pad); Hand Z | Axial / open |
| `g_world` | `(0,0,-9.81)` |
| `g_h = R_handᵀ g_world` | Gravity in hand frame |
| `u_dx_offset_hand = sign(DX)·x̂` | Direction of existing lateral offset |
| `adverse = g_h · u_dx_offset_hand` | >0 ⇒ gravity loads further into the offset side |
| `offset_down = −u_dx_world·ẑ` | >0 ⇒ offset direction points downward in world |

## Favorable orientation (S2/S3)

`favorable := (adverse ≤ 2 m/s²) ∧ (offset_down ≤ 0.35)`

Implemented as a **feasible region**, not Coupled-style `g·d ≥ 6` along a 2D desire.
Preparation: modest legal roll; skip if no improvement or contact risk.
If unreachable, recorded as `unreachable_favorable` without forcing.

## Axial slip

Experimental DZ desire: reduce signed `(ez − target)` along `±u_open` (production inward = `−u_open` when ez>target).
""",
        encoding="utf-8",
    )

    lines = [
        "# CONTACT_STABILITY_ANALYSIS",
        "",
        "| hard_id | strategy | NEW | 12S | status | bilateral_final | failures |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in order_rows:
        lines.append(
            f"| `{r.get('hard_id')}` | {r.get('strategy')} | {r.get('SUCCESS_NEW')} | {r.get('SUCCESS_12S')} | "
            f"`{r.get('status')}` | — | {r.get('failures')} |"
        )
    (OUT / "CONTACT_STABILITY_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    by = {}
    for r in order_rows:
        by.setdefault(r["hard_id"], {})[r["strategy"]] = r

    def cnt(s):
        return sum(1 for r in order_rows if r.get("strategy") == s and r.get("SUCCESS_NEW") is True)

    recovers = []
    for hid, m in by.items():
        if hid == "hard_910026":
            continue
        for s in ("S1", "S2", "S3"):
            if not (m.get("S0") or {}).get("SUCCESS_NEW") and (m.get(s) or {}).get("SUCCESS_NEW"):
                recovers.append((s, hid))
    reg = []
    m026 = by.get("hard_910026") or {}
    for s in ("S1", "S2", "S3"):
        if (m026.get("S0") or {}).get("SUCCESS_NEW") and not (m026.get(s) or {}).get("SUCCESS_NEW"):
            reg.append(s)

    prefer = "none"
    if cnt("S1") >= max(cnt("S2"), cnt("S3")) and cnt("S1") > cnt("S0") and "S1" not in reg:
        prefer = "S1"
    elif cnt("S2") > cnt("S1") and cnt("S2") > cnt("S0") and "S2" not in reg:
        prefer = "S2"
    elif cnt("S3") > cnt("S0") and "S3" not in reg:
        prefer = "S3"

    adopt = prefer != "none" and not reg
    fr = f"""# FINAL_RECOMMENDATION — DZ-First Ordering (P4 Revised)

Production Oracle **unchanged**.

## SUCCESS_NEW counts

| Strategy | SUCCESS_NEW |
|---|---:|
| S0 production | {cnt('S0')}/7 |
| S1 DZ→DX | {cnt('S1')}/7 |
| S2 orient→DZ→DX | {cnt('S2')}/7 |
| S3 orient→DZ→DX→DZ fine | {cnt('S3')}/7 |

- Recoveries: {recovers or 'none'}
- Regressions of hard_910026: {reg or 'none'}
- Prefer: **{prefer}**

## Causal notes

- S0 vs S1: order alone (shared experimental DZ on S1–S3; S0 uses production DZ).
- Experimental DZ: **no `MAX_SAFE_TRAVEL` stop**; multi-attempt zero-target with contact/lateral/no-progress/time stops only.
- If `\|ex\| > dz_safe_ex`, pure DZ-first is **blocked** and falls back to DX-then-DZ with explicit phase label.

## Recommendation: **{'CONDITIONAL ADOPT ' + prefer if adopt else 'DO NOT ADOPT DZ-first into production'}**

Do not generalize beyond these seven verified B2 ICs.
"""
    (OUT / "FINAL_RECOMMENDATION.md").write_text(fr, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--primary-first", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    ids = HARD_IDS
    if args.only:
        ids = [x.strip() for x in args.only.split(",") if x.strip()]
    elif args.primary_first:
        ids = PRIMARY + [h for h in HARD_IDS if h not in PRIMARY]

    env = make_env()
    pcfg = cfg()
    cases = []
    order_rows = []
    phase_rows = []
    for hid in ids:
        case = run_case(env, hid, pcfg)
        phase_rows.extend(case.pop("phase_rows", []))
        cases.append(jsafe(case))
        for s in ("S0", "S1", "S2", "S3"):
            v = case.get("variants", {}).get(s) or {}
            if v.get("status") == "ERROR":
                order_rows.append({"hard_id": hid, "strategy": s, "status": "ERROR", "SUCCESS_NEW": False})
            else:
                order_rows.append(row_from(hid, v))

    (OUT / "ORDERING_CASES.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    write_csv(OUT / "ORDERING_RESULTS.csv", order_rows)
    write_csv(OUT / "PHASE_ERROR_TRACES.csv", phase_rows)
    write_reports(cases, order_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ids": ids,
        "s0_new": sum(1 for r in order_rows if r.get("strategy") == "S0" and r.get("SUCCESS_NEW")),
        "s1_new": sum(1 for r in order_rows if r.get("strategy") == "S1" and r.get("SUCCESS_NEW")),
        "s2_new": sum(1 for r in order_rows if r.get("strategy") == "S2" and r.get("SUCCESS_NEW")),
        "s3_new": sum(1 for r in order_rows if r.get("strategy") == "S3" and r.get("SUCCESS_NEW")),
    }
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
