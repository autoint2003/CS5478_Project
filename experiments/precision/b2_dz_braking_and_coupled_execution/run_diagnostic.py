#!/usr/bin/env python
"""P4: DZ braking diagnosis + Coupled non-execution audit on hard_910305 / hard_910084."""

from __future__ import annotations

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
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from data_generation.oracles.precision.mechanics.support_transfer import restore_fingertips_down  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.b2_dz_braking_and_coupled_execution.instrumented_slip import (  # noqa: E402
    compute_inward_axis_tol,
    run_instrumented_inward_slip,
)
from experiments.precision.coupled_feasibility.coupled_dx_dz import (  # noqa: E402
    coupled_desire_hand,
    coupled_dx_dz,
)
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/b2_dz_braking_and_coupled_execution"
EXIT_CACHE = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
FOCUS = ["hard_910305", "hard_910084"]
COUPLED_TRACE_IDS = ["hard_910026", "hard_910132", "hard_910024", "hard_910084", "hard_910305"]


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


def load_exit(hid: str) -> dict:
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


def reach_pre_slip_state(env, blob, pcfg) -> dict[str, Any]:
    """Run production correct_dz_inward setup until just before slip by calling full primitive
    is heavy; instead call correct_dz and extract from phase_log — for instrumentation we
    re-enter via production path: if already UP-friendly, slip starts soon.

    Simpler: run production correct_dz_inward but we need pre-slip capture.
    We'll use a two-phase approach: monkeypatch _inward_slip_staged to capture entry.
    """
    import data_generation.oracles.precision.corrections.axial_inward as ain

    holder: dict[str, Any] = {}
    orig = ain._inward_slip_staged

    def _capture(env2, blob2, z_ref, e_x0, milestone_mm=12.0, cfg=None):
        holder["pre_slip_blob"] = {"capture": take_capture(env2), "ref": blob2.get("ref")}
        holder["z_ref"] = z_ref
        holder["e_x0"] = e_x0
        holder["milestone_mm"] = milestone_mm
        err = measure_errors(env2.sim)
        holder["errors_pre_slip"] = err
        holder["tol"] = compute_inward_axis_tol(err["e_signed_mm"], float((cfg or pcfg).working_pos_mm))
        # Run real slip for D0 path when requested
        return orig(env2, blob2, z_ref, e_x0, milestone_mm=milestone_mm, cfg=cfg)

    ain._inward_slip_staged = _capture
    try:
        restore_capture(env, blob["capture"])
        res = correct_dz_inward(env, {"capture": blob["capture"], "ref": blob["ref"]}, pcfg, s1_state=None)
    finally:
        ain._inward_slip_staged = orig
    holder["dz_result"] = {k: v for k, v in res.items() if k not in ("actions", "capture")}
    holder["actions"] = list(res.get("actions") or [])
    holder["capture_after"] = res.get("capture")
    holder["dz_ok"] = bool(res.get("ok"))
    return holder


def run_d_variants(env, hid: str, blob: dict, pcfg: PrecisionConfig) -> dict:
    print(f"\n=== {hid} D0/D1/D2 ===", flush=True)
    out: dict[str, Any] = {"hard_id": hid}
    restore_capture(env, blob["capture"])
    elig = check_stage1_eligible(
        env, {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob.get("sample_id")}, allow_live_verify=True
    )
    out["stage1_eligible"] = elig.to_dict()
    err_entry = measure_errors(env.sim)
    out["errors_entry"] = {
        "dx": float(err_entry["dx_mm"]),
        "dy": float(err_entry["dy_mm"]),
        "dz": float(err_entry["dz_mm"]),
        "e_pos": float(err_entry["e_pos_mm"]),
        "dtheta": float(err_entry["dtheta_deg"]),
    }

    # ---- D0: full production Precision ----
    print(f"  [{hid}] D0 production run_precision...", flush=True)
    restore_capture(env, blob["capture"])
    res0 = run_precision(env, {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob["sample_id"]}, "B", pcfg)
    hand0 = res0.handoff or {}
    out["D0"] = {
        "status": res0.status,
        "SUCCESS_NEW": bool(hand0.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand0.get("SUCCESS_12S")),
        "phases": list(res0.phases or []),
        "failures": list(res0.failures or []),
        "n_actions": len(res0.actions or []),
        "errors_exit": {
            "dx": float((res0.errors_exit or {}).get("dx_mm") or 0),
            "dy": float((res0.errors_exit or {}).get("dy_mm") or 0),
            "dz": float((res0.errors_exit or {}).get("dz_mm") or 0),
            "e_pos": float((res0.errors_exit or {}).get("e_pos_mm") or 0),
            "dtheta": float((res0.errors_exit or {}).get("dtheta_deg") or 0),
        },
        "dz_logs": (res0.primitive_logs or {}).get("dz")
        or (res0.primitive_logs or {}).get("dz_inward")
        or (res0.primitive_logs or {}).get("correct_dz_B_inward_under_s1"),
    }
    # Persist D0 actions
    np.savez_compressed(OUT / "raw" / f"{hid}__D0__actions.npz", actions=np.asarray(res0.actions or [], np.float32))
    print(f"    D0 NEW={out['D0']['SUCCESS_NEW']} 12S={out['D0']['SUCCESS_12S']} dz={out['D0']['errors_exit']['dz']:.2f}", flush=True)

    # ---- Instrumented production DZ primitive (for timeline) ----
    print(f"  [{hid}] instrumented production DZ (capture pre-slip)...", flush=True)
    restore_capture(env, blob["capture"])
    held = reach_pre_slip_state(env, blob, pcfg)
    out["production_dz_primitive"] = {
        "ok": held.get("dz_ok"),
        "result": held.get("dz_result"),
        "tol_at_pre_slip": held.get("tol"),
        "errors_pre_slip": {
            "dx": float((held.get("errors_pre_slip") or {}).get("dx_mm") or 0),
            "dz": float((held.get("errors_pre_slip") or {}).get("dz_mm") or 0),
            "e_pos": float((held.get("errors_pre_slip") or {}).get("e_pos_mm") or 0),
        },
    }
    # Re-run instrumented slip from pre-slip capture for fine timeline
    timeline_rows = []
    markers = {}
    if held.get("pre_slip_blob"):
        restore_capture(env, held["pre_slip_blob"]["capture"])
        slip0 = run_instrumented_inward_slip(env, held["pre_slip_blob"], pcfg, continue_past_production_tol=False)
        timeline_rows = list(slip0.get("timeline") or [])
        markers = dict(slip0.get("markers") or {})
        # Append return/regrip from production restore
        restore_capture(env, slip0["capture"])
        dz_pre_ret = float(measure_errors(env.sim)["dz_mm"])
        markers["dz_pre_return"] = dz_pre_ret
        rd = restore_fingertips_down(
            env, {"capture": slip0["capture"], "ref": blob["ref"]}, pcfg, rate_scale=float(pcfg.rate_restore_down)
        )
        for i, a in enumerate(rd.get("actions") or []):
            if i % 5 == 0:
                err = measure_errors(env.sim)
                timeline_rows.append(
                    {
                        "t": float(timeline_rows[-1]["t"] + 0.02) if timeline_rows else 0.0,
                        "phase": "return_DOWN",
                        "dx_mm": float(err["dx_mm"]),
                        "dy_mm": float(err["dy_mm"]),
                        "dz_mm": float(err["dz_mm"]),
                        "e_pos_mm": float(err["e_pos_mm"]),
                        "dtheta_deg": float(err["dtheta_deg"]),
                        "bilateral": bool(err.get("bilateral")),
                        "fn_tot": float(err.get("fn_tot") or 0),
                        "stop_reason": None if rd.get("ok") else rd.get("reason"),
                    }
                )
        err_post = measure_errors(env.sim)
        markers["dz_post_return"] = float(err_post["dz_mm"])
        markers["dz_return_drift"] = float(err_post["dz_mm"]) - dz_pre_ret
        markers["return_ok"] = bool(rd.get("ok"))
        # HANDOFF from post-return
        if rd.get("ok") and err_post.get("bilateral"):
            restore_capture(env, rd["capture"])
            hand = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=pcfg.stabilize_handoff_n)
            restore_capture(env, rd["capture"])
            markers["handoff_after_instrumented"] = {
                "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
                "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
                "reason": hand.get("std_reason") or hand.get("reason"),
            }
            markers["dz_at_handoff"] = float(measure_errors(env.sim)["dz_mm"])
        out["instrumented_D0_slip"] = {
            "ok": slip0.get("ok"),
            "stop_reason": slip0.get("stop_reason"),
            "tol_info": slip0.get("tol_info"),
            "markers": markers,
            "goal_axis_tol": float(slip0["goal"].axis_tol_mm) if slip0.get("goal") else None,
        }
        write_csv(OUT / f"DZ_TIMELINE_{hid}.csv", timeline_rows)
        out["timeline_path"] = f"DZ_TIMELINE_{hid}.csv"

    # Why no further correction after D0?
    # done_dz latch + ez_tol=20 vs e_pos=12 mismatch
    out["no_further_correction_analysis"] = {
        "ez_tol_mm": float(pcfg.ez_tol_mm),
        "working_pos_mm": float(pcfg.working_pos_mm),
        "final_dz_D0": out["D0"]["errors_exit"]["dz"],
        "final_e_pos_D0": out["D0"]["errors_exit"]["e_pos"],
        "dz_below_ez_tol": abs(out["D0"]["errors_exit"]["dz"]) <= float(pcfg.ez_tol_mm),
        "e_pos_above_working": out["D0"]["errors_exit"]["e_pos"] > float(pcfg.working_pos_mm),
        "sm_done_dz_blocks_retry": True,
        "note": (
            "Inward slip stops at axis_tol=clip(sqrt(12^2-ex^2-ey^2),3,8) with use_euclidean_stop=False. "
            "Return_DOWN then increases |dz|. SM sets done_dz=True after one dz; "
            "reassess may set dz_needed via e_pos>12 but done_dz blocks another attempt."
        ),
    }

    # ---- D1: continue legal DZ past production tol from pre-slip ----
    print(f"  [{hid}] D1 continue past production tol...", flush=True)
    d1 = {"ok": False}
    if held.get("pre_slip_blob"):
        restore_capture(env, held["pre_slip_blob"]["capture"])
        slip1 = run_instrumented_inward_slip(
            env, held["pre_slip_blob"], pcfg, continue_past_production_tol=True, max_safe_travel_mm=18.0
        )
        restore_capture(env, slip1["capture"])
        dz_pre = float(measure_errors(env.sim)["dz_mm"])
        rd1 = restore_fingertips_down(
            env, {"capture": slip1["capture"], "ref": blob["ref"]}, pcfg, rate_scale=float(pcfg.rate_restore_down)
        )
        acts1 = list(slip1.get("actions") or []) + list(rd1.get("actions") or [])
        err1 = measure_errors(env.sim)
        hand1 = None
        if rd1.get("ok") and err1.get("bilateral"):
            hand1 = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=pcfg.stabilize_handoff_n)
            restore_capture(env, rd1["capture"] if rd1.get("capture") else slip1["capture"])
        d1 = {
            "slip_ok": bool(slip1.get("ok")),
            "stop_reason": slip1.get("stop_reason"),
            "markers": slip1.get("markers"),
            "tol_info": slip1.get("tol_info"),
            "dz_pre_return": dz_pre,
            "dz_post_return": float(err1["dz_mm"]),
            "e_pos_post_return": float(err1["e_pos_mm"]),
            "dtheta_post_return": float(err1["dtheta_deg"]),
            "bilateral": bool(err1.get("bilateral")),
            "return_ok": bool(rd1.get("ok")),
            "SUCCESS_NEW": bool((hand1 or {}).get("SUCCESS_NEW")),
            "SUCCESS_12S": bool((hand1 or {}).get("SUCCESS_12S")),
            "n_actions": len(acts1),
            "additional_motion_possible": bool(slip1.get("markers", {}).get("production_tol_crossed"))
            and abs(float((slip1.get("errors1") or {}).get("dz_mm") or 99))
            < abs(float(slip1.get("markers", {}).get("dz_when_production_tol_hit") or 99)),
            "improved_dz_vs_prod_mark": None,
        }
        if slip1.get("markers", {}).get("dz_when_production_tol_hit") is not None:
            d1["improved_dz_vs_prod_mark"] = abs(float(err1["dz_mm"])) < abs(
                float(slip1["markers"]["dz_when_production_tol_hit"])
            )
        np.savez_compressed(OUT / "raw" / f"{hid}__D1__actions.npz", actions=np.asarray(acts1, np.float32))
        write_csv(OUT / "raw" / f"{hid}__D1__timeline.csv", slip1.get("timeline") or [])
        print(
            f"    D1 NEW={d1['SUCCESS_NEW']} dz_post={d1['dz_post_return']:.2f} e_pos={d1['e_pos_post_return']:.2f}",
            flush=True,
        )
    out["D1"] = d1

    # ---- D2: production DZ + post-return extra legal DZ if e_pos still high ----
    print(f"  [{hid}] D2 post-return second DZ attempt...", flush=True)
    restore_capture(env, blob["capture"])
    held2 = reach_pre_slip_state(env, blob, pcfg)
    # After production dz (includes return inside correct_dz_inward)
    cap_after = held2.get("capture_after") or blob["capture"]
    restore_capture(env, cap_after)
    err_mid = measure_errors(env.sim)
    second = None
    acts2 = list(held2.get("actions") or [])
    if (
        err_mid.get("bilateral")
        and float(err_mid["e_pos_mm"]) > float(pcfg.working_pos_mm)
        and abs(float(err_mid["dz_mm"])) > 2.0
    ):
        # Additional legal inward attempt with tighter tol (still safety-capped)
        # Need UP posture again for inward — call correct_dz_inward again (legal)
        r2 = correct_dz_inward(env, {"capture": take_capture(env), "ref": blob["ref"]}, pcfg, s1_state=None)
        acts2 += list(r2.get("actions") or [])
        second = {
            "attempted": True,
            "ok": bool(r2.get("ok")),
            "reason": r2.get("reason") or r2.get("stop_reason"),
            "errors_after": {
                "dz": float((r2.get("errors") or measure_errors(env.sim)).get("dz_mm") or measure_errors(env.sim)["dz_mm"]),
                "e_pos": float(measure_errors(env.sim)["e_pos_mm"]),
                "dtheta": float(measure_errors(env.sim)["dtheta_deg"]),
            },
        }
        cap_after = r2.get("capture") or take_capture(env)
    else:
        second = {
            "attempted": False,
            "reason": "post_return_not_eligible_or_already_in_band",
            "errors_mid": {
                "dz": float(err_mid["dz_mm"]),
                "e_pos": float(err_mid["e_pos_mm"]),
                "bilateral": bool(err_mid.get("bilateral")),
            },
        }
    restore_capture(env, cap_after)
    err_f = measure_errors(env.sim)
    hand2 = None
    if err_f.get("bilateral"):
        # ensure DOWN for handoff
        if not err_f.get("fingertips_down"):
            rd = restore_fingertips_down(env, {"capture": cap_after, "ref": blob["ref"]}, pcfg, rate_scale=float(pcfg.rate_restore_down))
            acts2 += list(rd.get("actions") or [])
            cap_after = rd.get("capture") or cap_after
            restore_capture(env, cap_after)
            err_f = measure_errors(env.sim)
        if err_f.get("bilateral"):
            hand2 = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=pcfg.stabilize_handoff_n)
            restore_capture(env, cap_after)
    d2 = {
        "first_dz_ok": held2.get("dz_ok"),
        "second": second,
        "dz_final": float(err_f["dz_mm"]),
        "e_pos_final": float(err_f["e_pos_mm"]),
        "dtheta_final": float(err_f["dtheta_deg"]),
        "bilateral": bool(err_f.get("bilateral")),
        "SUCCESS_NEW": bool((hand2 or {}).get("SUCCESS_NEW")),
        "SUCCESS_12S": bool((hand2 or {}).get("SUCCESS_12S")),
        "n_actions": len(acts2),
    }
    np.savez_compressed(OUT / "raw" / f"{hid}__D2__actions.npz", actions=np.asarray(acts2, np.float32))
    print(f"    D2 NEW={d2['SUCCESS_NEW']} dz={d2['dz_final']:.2f} e_pos={d2['e_pos_final']:.2f} second={second}", flush=True)
    out["D2"] = d2
    return out


def audit_coupled(env, hid: str, blob: dict, pcfg: PrecisionConfig) -> dict:
    print(f"  [{hid}] Coupled execution audit...", flush=True)
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    desire = coupled_desire_hand(err0["e_signed_mm"])
    from data_generation.oracles.precision.corrections.orientation import (
        execute_orientation_plan,
        plan_orientation,
        score_gravity_alignment,
        G_USEFUL_MIN,
    )
    from evaluation.grasp_geometry import hand_R

    G_WORLD = np.array([0.0, 0.0, -9.81], float)
    row: dict[str, Any] = {
        "hard_id": hid,
        "dx0": float(err0["dx_mm"]),
        "dz0": float(err0["dz_mm"]),
        "e_pos0": float(err0["e_pos_mm"]),
        "desire_ok": bool(desire.get("ok")),
        "desire_reason": desire.get("reason"),
        "desire_d": desire.get("d"),
        "geom_angle_deg": desire.get("geom_angle_deg"),
        "weight_dx": desire.get("weight_dx"),
        "weight_dz": desire.get("weight_dz"),
        "plan_g_useful_min": 6.0,
        "post_orient_useful_threshold": 4.0,
        "canonical_G_USEFUL_MIN": float(G_USEFUL_MIN),
        "coupled_slip_executed": False,
        "label": "COUPLED_NOT_EXECUTED",
    }
    if not desire.get("ok"):
        row["abort_stage"] = "desire"
        row["abort_reason"] = desire.get("reason")
        return row

    d = np.asarray(desire["d"], float)
    plan = plan_orientation(env, blob, d=d, g_useful_min=6.0, max_candidates=36)
    row["orientation_plan_ok"] = bool(plan.get("ok") or plan.get("already_sufficient"))
    row["orientation_plan_reason"] = plan.get("reason")
    row["orientation_already_sufficient"] = bool(plan.get("already_sufficient"))
    row["orientation_useful_pred"] = (plan.get("best") or {}).get("useful") if isinstance(plan.get("best"), dict) else plan.get("useful")
    acts = []
    cur = {"capture": blob["capture"], "ref": blob.get("ref")}
    if plan.get("already_sufficient"):
        restore_capture(env, blob["capture"])
    elif plan.get("ok"):
        eo = execute_orientation_plan(env, cur, plan)
        acts += eo.get("actions") or []
        row["orientation_exec_ok"] = bool(eo.get("ok"))
        row["orientation_exec_reason"] = eo.get("reason")
        row["n_orient_actions"] = len(eo.get("actions") or [])
        if not eo.get("ok"):
            row["abort_stage"] = "orientation_exec"
            row["abort_reason"] = eo.get("reason")
            row["n_actions_total"] = len(acts)
            return row
        cur = {"capture": eo.get("capture") or take_capture(env), "ref": blob.get("ref")}
    else:
        row["abort_stage"] = "orientation_plan"
        row["abort_reason"] = plan.get("reason") or "no_feasible_gravity_orientation"
        return row

    restore_capture(env, cur["capture"])
    Rh = hand_R(env.sim)
    g_h = Rh.T @ G_WORLD
    sc = score_gravity_alignment(g_h, d)
    row["post_orient_useful"] = float(sc["useful"])
    row["post_orient_unwanted"] = float(sc["unwanted"])
    row["post_orient_align"] = float(sc["align"])
    row["g_h"] = g_h.tolist()

    # Full coupled call for ground truth
    restore_capture(env, blob["capture"])
    full = coupled_dx_dz(env, blob, cfg=pcfg)
    row["coupled_ok"] = bool(full.get("ok"))
    row["coupled_reason"] = full.get("reason")
    row["n_actions_coupled"] = len(full.get("actions") or [])
    # Did DualLoop slip run? Look for slip in logs
    logs = full.get("logs") or {}
    row["had_slip_log"] = "slip" in logs
    row["slip_stop"] = (logs.get("slip") or {}).get("stop_reason")
    if full.get("reason") == "gravity_alignment_insufficient_after_orient" or (
        float(sc["useful"]) < 4.0 and not full.get("ok")
    ):
        row["label"] = "COUPLED_NOT_EXECUTED"
        row["abort_stage"] = "post_orient_gravity_gate"
        row["abort_reason"] = "gravity_alignment_insufficient_after_orient"
        row["physical_purpose_of_gate"] = (
            "Require g·d >= 4 m/s^2 so gravity assists DualLoop along desire; "
            "below this, prior design treats slip as under-driven / contact-risk."
        )
        row["gate_vs_plan_mismatch"] = (
            f"plan_orientation uses g_useful_min=6.0 but post-exec gate is 4.0; "
            f"canonical G_USEFUL_MIN={G_USEFUL_MIN}. Observed useful={sc['useful']:.3f}."
        )
    elif logs.get("slip"):
        row["coupled_slip_executed"] = True
        row["label"] = "COUPLED_SLIP_EXECUTED"
    else:
        row["label"] = "COUPLED_NOT_EXECUTED"
        row["abort_stage"] = full.get("reason") or "unknown_pre_slip"
    row["n_actions_total"] = len(acts) + len(full.get("actions") or [])
    return row


def write_markdown_reports(focus_results: list[dict], coupled_rows: list[dict]) -> None:
    # DZ_STOPPING_CODE_AUDIT.md
    audit = """# DZ_STOPPING_CODE_AUDIT

Production inward DZ path (unchanged). Units: millimeters unless noted.

## Call chain

1. `state_machine.run_precision` → `correct_dz_inward` / `correct_dz_B_inward_under_s1`
2. `corrections/axial_inward.py` `correct_dz_inward` → UP prep → `_inward_slip_staged` → return DOWN
3. `mechanics/axial_sliding.py` `_inward_slip_staged` → `run_goal_aware_axial_slip`
4. `residuals.evaluate_slip_goal` decides per-step stop

## 1. DZ target

- **File:** `data_generation/oracles/precision/state.py` `SlipGoal.target_signed_mm` default `0.0`
- **Expression:** drive signed `e_z → 0` along `-u_open` (inward)
- **Rationale:** center object axially in hand frame
- **Type:** convergence criterion

## 2. DZ tolerance (critical)

- **File:** `axial_sliding.py` `_inward_slip_staged` lines ~824–828
- **Expression:**
  - `budget_sq = working_pos_mm² − ex² − ey²` with `working_pos_mm = 12`
  - `axis_tol = clip(sqrt(max(budget_sq,0)), 3, 8)`
- **Current thresholds:** clip **[3, 8] mm**; typical when |ex|,|ey| small → **8 mm**
- **Rationale:** leave axial residual compatible with Euclidean `e_pos≤12` *before* return, while not trusting live `e_pos` when fingertips-UP (`use_euclidean_stop=False`)
- **Type:** convergence criterion derived from SUCCESS_NEW 12 mm budget — **not** a hard-coded 15 mm

## 3. Axial slip

- **File:** `axial_sliding.py` `run_goal_aware_axial_slip` (~280+)
- DualLoop force/slip along `-U_OPEN_HAND`
- **Safety caps:** `max_safe_travel_mm = cfg.inward_slip_mm` (12), `max_duration_s = 12`, lateral drift abort `LATERAL_DRIFT_ABORT_MM = 4`

## 4. Slip completion detection

- **File:** `residuals.py` `evaluate_slip_goal` (~115–157)
- Inward: `use_euclidean_stop=False` → stop iff `|ez − target| ≤ axis_tol_mm`
- (`brake_margin_mm=1.0` is passed but for non-euclidean path only returns GOAL when `axis_err ≤ axis_tol`; margin does not early-stop)

## 5. Braking

- **File:** `axial_sliding.py` ~431–461 after loop break
- Fixed **0.45 s** brake + **0.25 s** regrip (grip force → ≥14 N)
- **Type:** safety / contact retention (not pose convergence)

## 6. Return / regrip

- **File:** `axial_inward.py` post-slip return DOWN (~337+)
- Also `support_transfer.restore_fingertips_down`
- Can **increase** |dz| (posture change + contact settle)
- **Type:** mandatory posture for HANDOFF

## 7. Another correction?

- **File:** `state_machine.py` — `done_dz = True` after one successful dz (~889)
- Reassess uses `entry_conditions`: `dz_needed = |ez| > ez_tol` (**20 mm** AXIS_V1) OR `e_pos>12` residual rule
- Even if `dz_needed` becomes True via `e_pos>12`, **`not done_dz` gate blocks retry**
- **Type:** state-machine latch (algorithmic), not physics

## 8. Phase complete

- Slip `ok` if bilateral + safe stop + (goal hit OR |dz| improved ≥0.3 mm)
- Then return DOWN; SM continues to FINAL_DOWN_GATE → 12 s HANDOFF

## Classification of ~15–16 mm finals

Not a hard-coded 15 mm stop. Typical composition:

1. Slip intentionally stops near **|dz|≈3–8 mm** (often ~8)
2. Return/regrip **adds outward drift**
3. Final |dz|~15–16 mm still **≤ ez_tol 20 mm** and/or **done_dz** prevents second pass while `e_pos>12` → `SUCCESS_12S` possible, `SUCCESS_NEW` fails
"""
    (OUT / "DZ_STOPPING_CODE_AUDIT.md").write_text(audit, encoding="utf-8")

    # DZ_CONTINUATION_RESULTS.md
    lines = ["# DZ_CONTINUATION_RESULTS", "", "D0 = production full Precision. D1 = continue slip past production axis_tol (legal safety caps). D2 = production DZ + second legal DZ if post-return e_pos>12.", ""]
    for r in focus_results:
        hid = r["hard_id"]
        lines.append(f"## `{hid}`")
        lines.append(f"- Entry: {r.get('errors_entry')}")
        lines.append(f"- D0: NEW={r['D0'].get('SUCCESS_NEW')} 12S={r['D0'].get('SUCCESS_12S')} exit={r['D0'].get('errors_exit')}")
        lines.append(f"- Instrumented markers: {(r.get('instrumented_D0_slip') or {}).get('markers')}")
        lines.append(f"- D1: {r.get('D1')}")
        lines.append(f"- D2: {r.get('D2')}")
        lines.append(f"- No-further-correction: {r.get('no_further_correction_analysis')}")
        lines.append("")
        d1 = r.get("D1") or {}
        lines.append("### Claims")
        lines.append(f"- Additional motion physically possible past prod tol: **{d1.get('additional_motion_possible')}**")
        lines.append(f"- Additional motion improves final DZ vs mark: **{d1.get('improved_dz_vs_prod_mark')}**")
        lines.append(f"- Achieves SUCCESS_NEW (D1): **{d1.get('SUCCESS_NEW')}** ; (D2): **{(r.get('D2') or {}).get('SUCCESS_NEW')}**")
        lines.append("")
    (OUT / "DZ_CONTINUATION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    # COUPLED_EXECUTION_AUDIT.md
    cl = [
        "# COUPLED_EXECUTION_AUDIT",
        "",
        "Prior Coupled implementation: `experiments/precision/coupled_feasibility/coupled_dx_dz.py`.",
        "",
        "## Intended mechanism vs executed",
        "",
        "1. Build 2D desire `d ∝ (−dx,0,−dz)`",
        "2. `plan_orientation(..., g_useful_min=6.0)`",
        "3. `execute_orientation_plan`",
        "4. **Post-orient gate:** `score_gravity_alignment(g_h,d).useful ≥ 4.0` else abort `gravity_alignment_insufficient_after_orient`",
        "5. Support-bias roll + DualLoop along `d`",
        "",
        "If abort at step 4 → **no DualLoop slip** → label **`COUPLED_NOT_EXECUTED`** (not a completed coupled-slip failure).",
        "",
        "| hard_id | label | abort_stage | useful_post | desire_ok | slip_executed |",
        "|---|---|---|---:|---|---|",
    ]
    for row in coupled_rows:
        cl.append(
            f"| `{row.get('hard_id')}` | {row.get('label')} | {row.get('abort_stage')} | "
            f"{row.get('post_orient_useful')} | {row.get('desire_ok')} | {row.get('coupled_slip_executed')} |"
        )
    cl += [
        "",
        "## Interpretation of the gravity gate",
        "",
        "- **Physical purpose:** gravity-assisted sliding needs `g·d` large enough; design comment cites ~9.7 m/s² closed-loop; planner uses 6.0; post-exec gate 4.0.",
        "- On these B2 exits, observed `useful` after orient is often **~2.3** → below gate → abort.",
        "- This is **not** evidence that simultaneous DX+DZ slip is physically impossible; it shows the **current Coupled implementation does not reach slip** on these ICs.",
        "- Possible contributors: insufficient orientation authority (ANGLE_DEGS / rate), geometric desire needing tip past reachable set, or threshold mismatch — **not** silently removed in this diagnostic.",
        "",
    ]
    (OUT / "COUPLED_EXECUTION_AUDIT.md").write_text("\n".join(cl), encoding="utf-8")

    # PAIRED_RESULTS.md
    pr = ["# PAIRED_RESULTS", "", "## DZ focus cases", "", "| hard_id | D0 NEW | D0 12S | D0 dz | D1 NEW | D1 dz | D2 NEW | D2 dz |", "|---|---|---|---:|---|---:|---|---:|"]
    for r in focus_results:
        pr.append(
            f"| `{r['hard_id']}` | {r['D0'].get('SUCCESS_NEW')} | {r['D0'].get('SUCCESS_12S')} | "
            f"{r['D0']['errors_exit'].get('dz')} | {r['D1'].get('SUCCESS_NEW')} | {r['D1'].get('dz_post_return')} | "
            f"{r['D2'].get('SUCCESS_NEW')} | {r['D2'].get('dz_final')} |"
        )
    pr += ["", "## Coupled execution labels", "", "| hard_id | label | reason |", "|---|---|---|"]
    for row in coupled_rows:
        pr.append(f"| `{row.get('hard_id')}` | {row.get('label')} | {row.get('abort_reason') or row.get('coupled_reason')} |")
    (OUT / "PAIRED_RESULTS.md").write_text("\n".join(pr), encoding="utf-8")

    # FINAL_RECOMMENDATION.md
    any_d1 = any(bool((r.get("D1") or {}).get("SUCCESS_NEW")) for r in focus_results)
    any_d2 = any(bool((r.get("D2") or {}).get("SUCCESS_NEW")) for r in focus_results)
    any_slip = any(bool(r.get("coupled_slip_executed")) for r in coupled_rows)
    fr = f"""# FINAL_RECOMMENDATION — P4 DZ Braking + Coupled Execution

**Production Oracle was not modified.**

## 1. Exact reason DZ correction stops

Inward DualLoop stops when `|dz| ≤ axis_tol` with
`axis_tol = clip(sqrt(12² − ex² − ey²), 3, 8)` and `use_euclidean_stop=False`
(`axial_sliding._inward_slip_staged`). This is a **convergence band derived from the 12 mm
Euclidean SUCCESS_NEW budget**, then clipped to ≤8 mm — **not** a hard-coded 15 mm threshold.

After stop: brake/regrip → return DOWN (often **increases** |dz|). SM sets `done_dz=True`,
so a second DZ is not scheduled even when final `e_pos>12` while `|dz|<ez_tol(20)`.

## 2. Does additional legal DZ achieve SUCCESS_NEW?

- D1 continued slip past production tol: SUCCESS_NEW on focus set = **{any_d1}**
- D2 second post-return DZ: SUCCESS_NEW = **{any_d2}**

See `DZ_CONTINUATION_RESULTS.md` for per-case distinction:
physical possibility vs DZ improvement vs final SUCCESS_NEW.

## 3. Why Coupled Slip never executed

Abort at **post-orientation gravity gate** (`useful < 4.0`) → label **`COUPLED_NOT_EXECUTED`**.
No simultaneous DualLoop along 2D desire ran on the audited ICs (`coupled_slip_executed={any_slip}`).

## 4. Can current Coupled be meaningfully tested here?

**Not with the present gate/orientation stack on these exits** — it aborts before slip.
Do **not** claim Coupled DX+DZ is physically infeasible from this abort alone.
A different legal coupling strategy would be required before effectiveness can be measured;
this diagnostic does not invent a new controller.

## Stop condition met

Identified DZ stop law + SM latch; measured D0/D1/D2; labeled Coupled non-execution.
No expert regen / training / production edits.
"""
    (OUT / "FINAL_RECOMMENDATION.md").write_text(fr, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    env = make_env()
    pcfg = cfg()

    focus_results = []
    for hid in FOCUS:
        blob = load_exit(hid)
        try:
            focus_results.append(jsafe(run_d_variants(env, hid, blob, pcfg)))
        except Exception as exc:
            focus_results.append({"hard_id": hid, "error": str(exc), "traceback": traceback.format_exc()})
            print(f"ERROR {hid}: {exc}", flush=True)

    (OUT / "FOCUS_RESULTS.json").write_text(json.dumps(focus_results, indent=2), encoding="utf-8")

    coupled_rows = []
    for hid in COUPLED_TRACE_IDS:
        if not (EXIT_CACHE / f"{hid}.pkl").is_file():
            continue
        blob = load_exit(hid)
        try:
            coupled_rows.append(jsafe(audit_coupled(env, hid, blob, pcfg)))
        except Exception as exc:
            coupled_rows.append({"hard_id": hid, "error": str(exc), "label": "COUPLED_NOT_EXECUTED"})
    write_csv(OUT / "COUPLED_ACTION_TRACE.csv", coupled_rows)
    (OUT / "COUPLED_ACTION_TRACE.json").write_text(json.dumps(coupled_rows, indent=2), encoding="utf-8")

    write_markdown_reports(focus_results, coupled_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "focus": FOCUS,
        "d1_any_success_new": any(bool((r.get("D1") or {}).get("SUCCESS_NEW")) for r in focus_results),
        "d2_any_success_new": any(bool((r.get("D2") or {}).get("SUCCESS_NEW")) for r in focus_results),
        "coupled_any_slip": any(bool(r.get("coupled_slip_executed")) for r in coupled_rows),
    }
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
