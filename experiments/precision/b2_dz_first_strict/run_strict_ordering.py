#!/usr/bin/env python
"""Strict DZ-first S0–S3: no threshold skip, no silent DX-before-DZ fallback.

Audit columns: DZ_FIRST_EXECUTED, DZ_FIRST_BLOCK_REASON, ACTUAL_PHASE_SEQUENCE.
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
from data_generation.oracles.precision.corrections.axial_inward import check_dz_inward_entry  # noqa: E402
from data_generation.oracles.precision.entry_conditions import assess_need  # noqa: E402
from data_generation.oracles.precision.handoff import run_final_down_handoff_gate  # noqa: E402
from data_generation.oracles.precision.mechanics.support_transfer import (  # noqa: E402
    correct_dx_support_transfer,
    restore_fingertips_down,
)
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
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
from experiments.precision.b2_dz_first_strict.dz_controller_v2 import (  # noqa: E402
    EPS_DZ_MM,
    run_strict_dz,
)
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/b2_dz_first_strict"
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


def _entry_settle(env, blob, actions: list) -> dict:
    """Restore Stage-1 exit only.

    A prior 0.2s grip=1 settle was shown to destabilize hard_910305 DZ
    (abs-improve then contact loss) while the identical controller succeeds
    from the raw exit capture. Do not reintroduce without a case-level audit.
    """
    restore_capture(env, blob["capture"])
    return {"capture": blob["capture"], "ref": blob["ref"]}


def run_S0(env, blob, pcfg) -> dict:
    restore_capture(env, blob["capture"])
    e0 = err_pack(env)
    res = run_precision(
        env,
        {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob.get("sample_id")},
        "B",
        pcfg,
    )
    hand = res.handoff or {}
    phases = list(res.phases or [])
    return {
        "strategy": "S0",
        "status": res.status,
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "phases": phases,
        "ACTUAL_PHASE_SEQUENCE": "|".join(phases),
        "failures": list(res.failures or []),
        "n_actions": len(res.actions or []),
        "actions": list(res.actions or []),
        "init": e0,
        "final": {
            "dx": float((res.errors_exit or {}).get("dx_mm") or 0),
            "dy": float((res.errors_exit or {}).get("dy_mm") or 0),
            "dz": float((res.errors_exit or {}).get("dz_mm") or 0),
            "e_pos": float((res.errors_exit or {}).get("e_pos_mm") or 0),
            "dtheta": float((res.errors_exit or {}).get("dtheta_deg") or 0),
            "bilateral": bool((res.errors_exit or {}).get("bilateral", True)),
        },
        "DZ_nonzero": abs(float(e0["dz"])) > EPS_DZ_MM,
        "DZ_attempted": "dz" in "|".join(phases).lower() or "inward" in "|".join(phases).lower(),
        "DZ_FIRST_EXECUTED": None,
        "DZ_FIRST_BLOCK_REASON": "N/A_S0_PRODUCTION",
        "dz_termination_reason": None,
        "axial_correction_occurred": None,
        "dx_before_dz": None,
        "order_respected": None,
        "phase_traces": [],
    }


def run_experimental(env, blob, pcfg, strategy: str, *, bypass_ex_gate: bool) -> dict:
    actions: list = []
    phases: list[str] = []
    failures: list[str] = []
    traces: list[dict] = []
    timeline: list = []
    budget = int(pcfg.global_action_budget)

    audit = {
        "DZ_nonzero": False,
        "DZ_attempted": False,
        "DZ_FIRST_EXECUTED": False,
        "DZ_FIRST_BLOCK_REASON": "",
        "axial_correction_occurred": False,
        "dz0": None,
        "dz1": None,
        "dz_termination_reason": None,
        "dx_executed_before_dz": False,
        "order_respected": False,
        "numerically_zero_dz": False,
    }

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
            **audit,
            "ACTUAL_PHASE_SEQUENCE": "",
        }

    cur = _entry_settle(env, blob, actions)
    phases.append("entry_settle")
    init = err_pack(env)
    grav0 = analyze_dx_gravity(env.sim)
    traces.append({"phase": "init", **init, **{k: grav0[k] for k in ("adverse_g_along_dx_offset_mps2", "favorable", "offset_points_down_score")}})

    need = assess_need(measure_errors(env.sim), pcfg)
    orientation_info = None

    # Preserve dθ when needed (existing skill dependency; not a DZ skip)
    if need.get("dtheta_needed"):
        phases.append("correct_dtheta")
        rd = correct_dtheta(env, cur, pcfg, "B")
        actions += list(rd.get("actions") or [])
        if not rd.get("ok"):
            failures.append(f"dtheta:{rd.get('reason')}")
            audit["DZ_FIRST_EXECUTED"] = False
            audit["DZ_FIRST_BLOCK_REASON"] = f"DTHETA_FAILED_BEFORE_DZ:{rd.get('reason')}"
            audit["order_respected"] = False
            return _fail(strategy, phases, failures, actions, traces, audit, "DZ_FIRST_NOT_EXECUTED", init)
        cur = {"capture": rd["capture"], "ref": blob["ref"]}
        traces.append({"phase": "after_dtheta", **err_pack(env)})
        need = assess_need(measure_errors(env.sim), pcfg)

    # S2/S3 favorable orientation — not gated on dz_needed / ez_tol
    if strategy in ("S2", "S3"):
        dz_now = abs(float(err_pack(env)["dz"]))
        if dz_now > EPS_DZ_MM:
            phases.append("favorable_orientation")
            orientation_info = prepare_favorable_orientation(env, cur)
            actions += list(orientation_info.get("actions") or [])
            if orientation_info.get("ok") and orientation_info.get("capture"):
                cur = {"capture": orientation_info["capture"], "ref": blob["ref"]}
            traces.append(
                {
                    "phase": "after_orientation",
                    **err_pack(env),
                    "orient_skipped": orientation_info.get("skipped"),
                    "orient_reason": orientation_info.get("reason"),
                }
            )

    # --- Strict DZ first ---
    ep = err_pack(env)
    dz_signed = float(ep["dz"])
    audit["dz0"] = dz_signed
    audit["DZ_nonzero"] = abs(dz_signed) > EPS_DZ_MM

    if not audit["DZ_nonzero"]:
        audit["numerically_zero_dz"] = True
        audit["DZ_FIRST_EXECUTED"] = True  # vacuously: nothing to correct before DX
        audit["DZ_FIRST_BLOCK_REASON"] = "NUMERICALLY_ZERO"
        audit["order_respected"] = True
        phases.append("dz_skip_numerically_zero")
        traces.append({"phase": "dz_numerically_zero", **ep})
    else:
        # Record whether production entry gate would block (audit), then attempt per bypass flag
        entry = check_dz_inward_entry(env.sim, pcfg)
        entry_reasons = list(entry.get("reasons") or [])
        ex_block = any("lateral_ex_too_large" in r for r in entry_reasons)
        if not entry.get("ok") and not bypass_ex_gate:
            # Do NOT silently run DX first
            phases.append("DZ_BLOCKED_BY_EX_GATE" if ex_block else "DZ_ENTRY_REJECT")
            audit["DZ_attempted"] = False
            audit["DZ_FIRST_EXECUTED"] = False
            audit["DZ_FIRST_BLOCK_REASON"] = (
                "DZ_BLOCKED_BY_EX_GATE" if ex_block else "DZ_ENTRY_REJECT:" + ";".join(entry_reasons)
            )
            audit["order_respected"] = False
            failures.append(f"DZ_FIRST_NOT_EXECUTED:{audit['DZ_FIRST_BLOCK_REASON']}")
            traces.append({"phase": "dz_blocked", "entry_reasons": entry_reasons, **ep})
            return _fail(
                strategy,
                phases,
                failures,
                actions,
                traces,
                audit,
                "DZ_FIRST_NOT_EXECUTED",
                init,
                orientation_info,
            )

        phases.append("dz_strict_zero_target")
        audit["DZ_attempted"] = True
        rem = max(100, budget - len(actions))
        dzr = run_strict_dz(
            env,
            cur,
            pcfg,
            bypass_ex_gate=bypass_ex_gate or (not entry.get("ok") and bypass_ex_gate),
            do_return_down=True,
            action_budget_left=rem,
            timeline=timeline,
            label="dz_coarse",
        )
        # If gate was bypassed for audit mode, still attempt
        if dzr.get("blocked"):
            phases.append(str(dzr.get("block_reason") or "DZ_BLOCKED"))
            audit["DZ_FIRST_EXECUTED"] = False
            audit["DZ_FIRST_BLOCK_REASON"] = str(dzr.get("block_reason") or "DZ_BLOCKED")
            audit["order_respected"] = False
            failures.append(f"DZ_FIRST_NOT_EXECUTED:{audit['DZ_FIRST_BLOCK_REASON']}")
            actions += list(dzr.get("actions") or [])
            return _fail(
                strategy,
                phases,
                failures,
                actions,
                traces,
                audit,
                "DZ_FIRST_NOT_EXECUTED",
                init,
                orientation_info,
            )

        actions += list(dzr.get("actions") or [])
        audit["DZ_FIRST_EXECUTED"] = True
        audit["DZ_FIRST_BLOCK_REASON"] = (
            "ENTRY_BYPASSED_EX_GATE" if dzr.get("entry_bypassed_ex") else ""
        )
        audit["axial_correction_occurred"] = bool(dzr.get("axial_correction_occurred"))
        audit["dz1"] = dzr.get("dz1")
        audit["dz_termination_reason"] = dzr.get("stop_reason")
        audit["dx_executed_before_dz"] = False
        audit["order_respected"] = True
        traces.append(
            {
                "phase": "after_dz_strict",
                **err_pack(env),
                "dz0": dzr.get("dz0"),
                "dz1": dzr.get("dz1"),
                "stop_reason": dzr.get("stop_reason"),
                "n_attempts": dzr.get("n_attempts"),
                "axial": dzr.get("axial_correction_occurred"),
                "improved_abs": dzr.get("improved_abs_after_return"),
            }
        )
        if not dzr.get("ok"):
            # DZ was executed (ordering valid) but failed physically
            failures.append(f"dz_strict:{dzr.get('stop_reason') or dzr.get('reason')}")
            if not measure_errors(env.sim).get("bilateral"):
                return _fail(
                    strategy,
                    phases,
                    failures,
                    actions,
                    traces,
                    audit,
                    "PRECISION_PRIMITIVE_FAILED",
                    init,
                    orientation_info,
                )
        cur = {"capture": dzr.get("capture") or take_capture(env), "ref": blob["ref"]}

        from data_generation.oracles.precision.mechanics.open_axis import gates as _gates

        if not _gates(env.sim).get("fingertips_down"):
            phases.append("restore_DOWN_before_dx")
            rd = restore_fingertips_down(env, cur, pcfg, rate_scale=float(pcfg.rate_restore_down))
            actions += list(rd.get("actions") or [])
            if rd.get("ok") and rd.get("capture") is not None:
                cur = {"capture": rd["capture"], "ref": blob["ref"]}
            elif measure_errors(env.sim).get("bilateral") and (
                bool(dzr.get("improved_abs_after_return")) or abs(float(dzr.get("dz1") or 99)) <= 1.0
            ):
                # DZ already reached ~0 under UP; a second DOWN restore can trip
                # world_height (seen on hard_910305). Keep grasp and proceed to DX ST,
                # which establishes its own support posture.
                failures.append(f"restore_DOWN_soft:{rd.get('reason')}")
                phases.append("restore_DOWN_soft_continue")
                cur = {"capture": take_capture(env), "ref": blob["ref"]}
            else:
                failures.append(f"restore_DOWN:{rd.get('reason')}")
                return _fail(
                    strategy,
                    phases,
                    failures,
                    actions,
                    traces,
                    audit,
                    "PRECISION_PRIMITIVE_FAILED",
                    init,
                    orientation_info,
                )

    # DX after DZ (or after numeric-zero skip).
    # Do NOT force DX on tiny residuals (EPS): on hard_910305, DZ leaves dx≈0.5 mm
    # and a forced ST then fails world_height from residual UP posture.
    # assess_need uses working/ex tolerances — not the 12 mm SUCCESS band alone.
    need = assess_need(measure_errors(env.sim), pcfg)
    dtheta_before_dx = float(err_pack(env)["dtheta"])
    if need.get("dx_needed"):
        phases.append("dx_support_transfer")
        rst = correct_dx_support_transfer(env, cur, pcfg)
        actions += list(rst.get("actions") or [])
        if not rst.get("ok") and not rst.get("skipped"):
            failures.append(f"support_transfer_dx:{rst.get('reason')}")
            return _fail(
                strategy,
                phases,
                failures,
                actions,
                traces,
                audit,
                "PRECISION_PRIMITIVE_FAILED",
                init,
                orientation_info,
            )
        cur = {"capture": rst.get("capture") or take_capture(env), "ref": blob["ref"]}
        dtheta_after_dx = float(err_pack(env)["dtheta"])
        traces.append(
            {
                "phase": "after_dx",
                **err_pack(env),
                "dtheta_delta_during_dx": dtheta_after_dx - dtheta_before_dx,
            }
        )
    else:
        dtheta_after_dx = dtheta_before_dx

    # S3 refinement
    if strategy == "S3":
        ep2 = err_pack(env)
        if abs(float(ep2["dz"])) > EPS_DZ_MM and len(actions) < budget - 50:
            phases.append("dz_fine_strict")
            dzf = run_strict_dz(
                env,
                cur,
                pcfg,
                bypass_ex_gate=bypass_ex_gate,
                do_return_down=True,
                action_budget_left=budget - len(actions),
                timeline=timeline,
                label="dz_fine",
            )
            actions += list(dzf.get("actions") or [])
            if dzf.get("blocked"):
                phases.append(f"dz_fine_blocked:{dzf.get('block_reason')}")
            elif not dzf.get("ok") and not measure_errors(env.sim).get("bilateral"):
                failures.append(f"dz_fine:{dzf.get('stop_reason')}")
                return _fail(
                    strategy,
                    phases,
                    failures,
                    actions,
                    traces,
                    audit,
                    "PRECISION_PRIMITIVE_FAILED",
                    init,
                    orientation_info,
                )
            else:
                cur = {"capture": dzf.get("capture") or take_capture(env), "ref": blob["ref"]}
            traces.append({"phase": "after_dz_fine", **err_pack(env), "stop": dzf.get("stop_reason")})

    phases.append("final_down_gate")
    gate = run_final_down_handoff_gate(env, cur, pcfg, actions)
    if not gate.get("ok"):
        failures.append(f"final_down_gate:{gate.get('failure_code')}:{gate.get('reason')}")
        return _fail(
            strategy,
            phases,
            failures,
            actions,
            traces,
            audit,
            "PRECISION_PRIMITIVE_FAILED",
            init,
            orientation_info,
            dtheta_before_dx,
            dtheta_after_dx,
        )
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
        "ACTUAL_PHASE_SEQUENCE": "|".join(phases),
        "failures": failures,
        "n_actions": len(actions),
        "actions": actions,
        "final": final,
        "init": init,
        "phase_traces": traces,
        "timeline": timeline,
        "dtheta_before_dx": dtheta_before_dx,
        "dtheta_after_dx": dtheta_after_dx,
        "dtheta_delta_during_dx": float(dtheta_after_dx - dtheta_before_dx),
        "orientation": {k: v for k, v in (orientation_info or {}).items() if k != "actions"},
        "grav0": grav0,
        "used_max_safe_travel": False,
        **audit,
    }


def _fail(
    strategy,
    phases,
    failures,
    actions,
    traces,
    audit,
    status,
    init=None,
    orientation=None,
    dth0=None,
    dth1=None,
):
    return {
        "strategy": strategy,
        "status": status,
        "SUCCESS_NEW": False,
        "SUCCESS_12S": False,
        "phases": phases,
        "ACTUAL_PHASE_SEQUENCE": "|".join(phases),
        "failures": failures,
        "n_actions": len(actions),
        "actions": actions,
        "final": traces[-1] if traces else {},
        "init": init,
        "phase_traces": traces,
        "orientation": {k: v for k, v in (orientation or {}).items() if k != "actions"} if orientation else None,
        "dtheta_before_dx": dth0,
        "dtheta_after_dx": dth1,
        "dtheta_delta_during_dx": None if dth0 is None or dth1 is None else float(dth1 - dth0),
        "used_max_safe_travel": False,
        **audit,
    }


def audit_row(hid: str, packed: dict) -> dict:
    fin = packed.get("final") or {}
    init = packed.get("init") or {}
    return {
        "hard_id": hid,
        "strategy": packed.get("strategy"),
        "init_dx": init.get("dx"),
        "init_dz": init.get("dz"),
        "DZ_nonzero": packed.get("DZ_nonzero"),
        "DZ_attempted": packed.get("DZ_attempted"),
        "axial_correction_occurred": packed.get("axial_correction_occurred"),
        "dz0": packed.get("dz0") if packed.get("dz0") is not None else init.get("dz"),
        "dz1": packed.get("dz1"),
        "dz_termination_reason": packed.get("dz_termination_reason"),
        "dx_executed_before_dz": packed.get("dx_executed_before_dz"),
        "DZ_FIRST_EXECUTED": packed.get("DZ_FIRST_EXECUTED"),
        "DZ_FIRST_BLOCK_REASON": packed.get("DZ_FIRST_BLOCK_REASON"),
        "ACTUAL_PHASE_SEQUENCE": packed.get("ACTUAL_PHASE_SEQUENCE") or "|".join(packed.get("phases") or []),
        "order_respected": packed.get("order_respected"),
        "final_dx": fin.get("dx"),
        "final_dy": fin.get("dy"),
        "final_dz": fin.get("dz"),
        "final_dtheta": fin.get("dtheta"),
        "final_e_pos": fin.get("e_pos"),
        "SUCCESS_12S": packed.get("SUCCESS_12S"),
        "SUCCESS_NEW": packed.get("SUCCESS_NEW"),
        "status": packed.get("status"),
        "failures": "|".join(packed.get("failures") or []),
        "n_actions": packed.get("n_actions"),
        "dtheta_delta_during_dx": packed.get("dtheta_delta_during_dx"),
    }


def run_case(env, hid: str, pcfg: PrecisionConfig, *, bypass_ex_gate: bool) -> dict:
    blob = load_blob(hid)
    print(f"\n=== {hid} (bypass_ex_gate={bypass_ex_gate}) ===", flush=True)
    restore_capture(env, blob["capture"])
    elig = check_stage1_eligible(
        env, {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob["sample_id"]}, allow_live_verify=True
    )
    # Gate audit snapshot at Stage-1 exit
    entry0 = check_dz_inward_entry(env.sim, pcfg)
    case = {
        "hard_id": hid,
        "stage1_eligible": elig.to_dict(),
        "entry_at_exit": jsafe({k: entry0[k] for k in entry0 if k != "errors"}),
        "init_errors": err_pack(env),
        "variants": {},
    }
    phase_rows = []
    for strategy in ("S0", "S1", "S2", "S3"):
        print(f"  [{hid}] {strategy}...", flush=True)
        try:
            if strategy == "S0":
                packed = run_S0(env, blob, pcfg)
            else:
                packed = run_experimental(env, blob, pcfg, strategy, bypass_ex_gate=bypass_ex_gate)
            acts = packed.pop("actions", [])
            tl = packed.pop("timeline", [])
            np.savez_compressed(OUT / "raw" / f"{hid}__{strategy}__actions.npz", actions=np.asarray(acts, np.float32))
            case["variants"][strategy] = jsafe(packed)
            for tr in packed.get("phase_traces") or []:
                phase_rows.append({"hard_id": hid, "strategy": strategy, **tr})
            print(
                f"    NEW={packed.get('SUCCESS_NEW')} DZ_FIRST={packed.get('DZ_FIRST_EXECUTED')} "
                f"block={packed.get('DZ_FIRST_BLOCK_REASON')} status={packed.get('status')}",
                flush=True,
            )
        except Exception as exc:
            case["variants"][strategy] = {
                "strategy": strategy,
                "status": "ERROR",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "DZ_FIRST_EXECUTED": False,
                "DZ_FIRST_BLOCK_REASON": f"ERROR:{exc}",
                "ACTUAL_PHASE_SEQUENCE": "",
            }
            print(f"    ERROR {exc}", flush=True)
    case["phase_rows"] = phase_rows
    return case


def write_invalidity_audit() -> None:
    (OUT / "PREVIOUS_ORDERING_INVALIDITY_AUDIT.md").write_text(
        """# PREVIOUS_ORDERING_INVALIDITY_AUDIT

Prior `b2_dz_first_ordering` S1–S3 results **do not** establish whether strict DZ-first works.

## Invalidating mechanisms found in `run_ordering.py`

1. **Threshold-based DZ skip**
   ```python
   if need.get("dz_needed") or abs(dz) > 0.25 * ez_tol_mm:
   ```
   Allowed skipping DZ when assess_need / tolerance said "not needed", even if signed DZ ≠ 0.

2. **Silent DX-before-DZ fallback**
   ```python
   if ex > dz_safe:
       phases.append("fallback_dx_before_dz_due_to_safety")
       # then DX, then DZ
   ```
   Cases with large |ex| were labeled as S1/S2/S3 but executed DX-first.

3. **Deadband early exit in experimental DZ** (`DEADBAND_MM = 1.5`) equated "inside band" with correction complete.

4. **Wrong-direction growth on hard_910305**: experimental slip from fingertips-DOWN without UP staging grew |DZ|; single attempt then contact loss.

## Consequence

Any SUCCESS/FAIL counts from that package mix:
- genuine DZ-first attempts,
- skipped DZ,
- DX-first fallbacks,
- and physically invalid DOWN slips.

This strict package re-runs with those mechanisms removed.
""",
        encoding="utf-8",
    )


def write_gate_audit(cases: list) -> None:
    lines = [
        "# DZ_GATE_AUDIT",
        "",
        "`dz_safe_ex_mm` (default 6) comes from `check_dz_inward_entry` / historical Stage-1 ICs that failed when inward DZ ran with large lateral offset.",
        "",
        "This audit records the gate state at each Stage-1 exit and whether strict runs used `bypass_ex_gate`.",
        "",
        "| hard_id | init_dx | init_dz | entry_ok | reasons | would_block_ex |",
        "|---|---:|---:|---|---|---|",
    ]
    for c in cases:
        e = c.get("init_errors") or {}
        ent = c.get("entry_at_exit") or {}
        reasons = ent.get("reasons") or []
        ex_b = any("lateral_ex_too_large" in str(r) for r in reasons)
        lines.append(
            f"| `{c.get('hard_id')}` | {e.get('dx')} | {e.get('dz')} | {ent.get('ok')} | "
            f"{';'.join(map(str, reasons)) or '—'} | {ex_b} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `would_block_ex` and `bypass_ex_gate=false`: case is `DZ_FIRST_NOT_EXECUTED` (not a completed DZ-first failure).",
            "- If `bypass_ex_gate=true`: we test whether contact state still supports legal DZ despite large DX; contact/actuator aborts remain enforced.",
            "",
        ]
    )
    (OUT / "DZ_GATE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def write_final(cases: list, rows: list) -> None:
    def cnt(s, key="SUCCESS_NEW"):
        return sum(1 for r in rows if r.get("strategy") == s and r.get(key) is True)

    def executed(s):
        return sum(1 for r in rows if r.get("strategy") == s and r.get("DZ_FIRST_EXECUTED") in (True, "True", "true", 1))

    lines = [
        "# FINAL_REPORT — Strict DZ-First",
        "",
        "Production Oracle unchanged. Strict package removes threshold skip and silent DX-first fallback.",
        "",
        "## Counts (all 7 ICs)",
        "",
        "| Strategy | SUCCESS_NEW | SUCCESS_12S | DZ_FIRST_EXECUTED |",
        "|---|---:|---:|---:|",
        f"| S0 | {cnt('S0')}/7 | {cnt('S0','SUCCESS_12S')}/7 | n/a |",
        f"| S1 | {cnt('S1')}/7 | {cnt('S1','SUCCESS_12S')}/7 | {executed('S1')}/7 |",
        f"| S2 | {cnt('S2')}/7 | {cnt('S2','SUCCESS_12S')}/7 | {executed('S2')}/7 |",
        f"| S3 | {cnt('S3')}/7 | {cnt('S3','SUCCESS_12S')}/7 | {executed('S3')}/7 |",
        "",
        "## Validity rule",
        "",
        "A case is evidence about DZ-first **only if** `DZ_FIRST_EXECUTED` is true (or initial DZ numerically zero).",
        "Blocked cases are `DZ_FIRST_NOT_EXECUTED` and must not be counted as completed DZ-first failures.",
        "",
        "## Per-case",
        "",
        "| hard_id | S | NEW | DZ_FIRST | block | phases (abbrev) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        seq = str(r.get("ACTUAL_PHASE_SEQUENCE") or "")
        if len(seq) > 80:
            seq = seq[:77] + "..."
        lines.append(
            f"| `{r.get('hard_id')}` | {r.get('strategy')} | {r.get('SUCCESS_NEW')} | "
            f"{r.get('DZ_FIRST_EXECUTED')} | {r.get('DZ_FIRST_BLOCK_REASON') or '—'} | `{seq}` |"
        )
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    ap.add_argument(
        "--bypass-ex-gate",
        action="store_true",
        help="Attempt DZ even when |ex|>dz_safe_ex_mm (still enforce contact safety).",
    )
    ap.add_argument("--gate-only", action="store_true", help="Write gate audit from exits without full strategies.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    write_invalidity_audit()

    ids = HARD_IDS
    if args.only:
        ids = [x.strip() for x in args.only.split(",") if x.strip()]

    env = make_env()
    pcfg = cfg()

    if args.gate_only:
        cases = []
        for hid in ids:
            blob = load_blob(hid)
            restore_capture(env, blob["capture"])
            entry0 = check_dz_inward_entry(env.sim, pcfg)
            cases.append(
                {
                    "hard_id": hid,
                    "init_errors": err_pack(env),
                    "entry_at_exit": jsafe({k: entry0[k] for k in entry0 if k != "errors"}),
                }
            )
        write_gate_audit(cases)
        print("gate audit only", flush=True)
        return 0

    cases = []
    rows = []
    phase_rows = []
    for hid in ids:
        case = run_case(env, hid, pcfg, bypass_ex_gate=bool(args.bypass_ex_gate))
        phase_rows.extend(case.pop("phase_rows", []))
        cases.append(jsafe(case))
        for s in ("S0", "S1", "S2", "S3"):
            v = case.get("variants", {}).get(s) or {}
            rows.append(audit_row(hid, v) if v.get("status") != "ERROR" else {
                "hard_id": hid,
                "strategy": s,
                "status": "ERROR",
                "SUCCESS_NEW": False,
                "DZ_FIRST_EXECUTED": False,
                "DZ_FIRST_BLOCK_REASON": v.get("DZ_FIRST_BLOCK_REASON"),
                "ACTUAL_PHASE_SEQUENCE": "",
                "failures": v.get("error"),
            })

    (OUT / "STRICT_CASES.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    write_csv(OUT / "STRICT_ORDERING_RESULTS.csv", rows)
    write_csv(OUT / "ACTUAL_PHASE_SEQUENCES.csv", [
        {
            "hard_id": r["hard_id"],
            "strategy": r["strategy"],
            "DZ_FIRST_EXECUTED": r.get("DZ_FIRST_EXECUTED"),
            "DZ_FIRST_BLOCK_REASON": r.get("DZ_FIRST_BLOCK_REASON"),
            "ACTUAL_PHASE_SEQUENCE": r.get("ACTUAL_PHASE_SEQUENCE"),
            "order_respected": r.get("order_respected"),
            "SUCCESS_NEW": r.get("SUCCESS_NEW"),
        }
        for r in rows
    ])
    write_csv(OUT / "PHASE_ERROR_TRACES.csv", phase_rows)
    write_gate_audit(cases)
    write_final(cases, rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ids": ids,
        "bypass_ex_gate": bool(args.bypass_ex_gate),
        "s0_new": sum(1 for r in rows if r.get("strategy") == "S0" and r.get("SUCCESS_NEW")),
        "s1_new": sum(1 for r in rows if r.get("strategy") == "S1" and r.get("SUCCESS_NEW")),
        "s2_new": sum(1 for r in rows if r.get("strategy") == "S2" and r.get("SUCCESS_NEW")),
        "s3_new": sum(1 for r in rows if r.get("strategy") == "S3" and r.get("SUCCESS_NEW")),
        "s1_dz_first_executed": sum(
            1 for r in rows if r.get("strategy") == "S1" and r.get("DZ_FIRST_EXECUTED") in (True, "True", True)
        ),
    }
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
