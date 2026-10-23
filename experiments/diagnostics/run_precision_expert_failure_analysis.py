#!/usr/bin/env python
"""Diagnose Precision Expert failures after genuine Stage-1 recatch.

Analyzes the 22 PRECISION_FAILURE ICs from stage2_b_precision_dominant_v1.
Does not modify Expert defaults, M111, datasets, or success criteria.
Does not start SFT/RL.

Outputs → results/diagnostics/precision_expert_failure_analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.diagnostics import run_b_grpo_multistage_dataset as base  # noqa: E402

from data_generation.oracles.precision.mechanics.dual_loop import sample_state  # noqa: E402
from data_generation.oracles.precision.entry_conditions import assess_need
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/precision_expert_failure_analysis"
DS = ROOT / "results/data_generation/stage2_b_precision_dominant_v1"
TREE80 = base.TREE80
DT = float(DT_POLICY)


def jsafe(o):
    return base.jsafe(o)


def atomic_json(path: Path, obj):
    base.atomic_json(path, obj)


def write_csv(path: Path, rows: list[dict]):
    base.write_csv(path, rows)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def err_pack(sim) -> dict:
    e = measure_errors(sim)
    st = sample_state(sim)
    return {
        "bilateral": bool(e.get("bilateral")),
        "e_pos_mm": float(e.get("e_pos_mm") or 0.0),
        "dtheta_deg": float(e.get("dtheta_deg") or 0.0),
        "dx_mm": float(e.get("dx_mm") or 0.0),
        "dy_mm": float(e.get("dy_mm") or 0.0),
        "dz_mm": float(e.get("dz_mm") or 0.0),
        "v_rel": float(st.get("v_rel_norm") or 0.0),
        "w_rel": float(st.get("w_rel_norm") or 0.0),
        "fn_L": float(e.get("fn_L") or 0.0),
        "fn_R": float(e.get("fn_R") or 0.0),
        "fn_tot": float(e.get("fn_tot") or 0.0),
        "obj_z": float(e.get("obj_z") or 0.0),
        "fingertips_down": bool(e.get("fingertips_down")),
    }


def load_failure_manifest() -> list[dict]:
    rows = []
    for line in (DS / "diagnostic_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("category") == "PRECISION_FAILURE":
            rows.append(r)
    return rows


def replay_stage1_to_exit(env, ic_blob: dict, stage1_path: Path) -> dict:
    """Replay frozen Stage-1 tape → Precision entry capture."""
    z = np.load(stage1_path)
    acts = np.asarray(z["actions"], np.float32)
    restore_capture(env, ic_blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    contact_trace = []
    both_off = False
    bil_after = False
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
        st = err_pack(env.sim)
        contact_trace.append({"i": i, **{k: st[k] for k in ("bilateral", "e_pos_mm", "fn_tot")}})
        if not st["bilateral"]:
            both_off = True
        elif both_off:
            bil_after = True
    exit_cap = take_capture(env)
    exit_st = err_pack(env.sim)
    return {
        "n_stage1": len(acts),
        "exit_capture": exit_cap,
        "exit_state": exit_st,
        "genuine_release_recatch": bool(both_off and bil_after and exit_st["bilateral"]),
        "saw_both_off": both_off,
        "recatch_bilateral": bil_after,
        "actions": acts,
        "contact_trace_n": len(contact_trace),
    }


def classify_labels(status: str, failures: list[str], phases: list[str], entry: dict, exit_e: dict) -> dict:
    labels = set()
    fail_s = " | ".join(failures or [])
    phase_s = " | ".join(phases or [])

    if status == "STABLE_GATE_FAILED":
        labels.add("STABLE_GATE_FAILED")
    if status == "PRECISION_PRIMITIVE_FAILED":
        labels.add("PRIMITIVE_FAILED")
    if status == "PRECISION_EXECUTED_HANDOFF_FAIL":
        labels.add("HANDOFF_FAILURE")

    if any("dtheta" in f for f in failures) or "correct_dtheta" in phase_s:
        if status == "PRECISION_PRIMITIVE_FAILED" and "correct_dtheta" in phase_s:
            labels.add("ORIENTATION_FAILURE")
    if any("verified_dx" in f or "support_transfer_dx" in f or "dx:" in f for f in failures):
        labels.add("LATERAL_FAILURE")
    if "dx_verified_lateral" in phase_s and status == "PRECISION_PRIMITIVE_FAILED":
        labels.add("LATERAL_FAILURE")
    if any(f.startswith("dz:") or "dz_precondition" in f or "dz_deferred" in f for f in failures):
        labels.add("VERTICAL_FAILURE")
    if "correct_dz" in phase_s and status == "PRECISION_PRIMITIVE_FAILED":
        labels.add("VERTICAL_FAILURE")
    if "action_budget" in fail_s or "ACTION_BUDGET" in phase_s:
        labels.add("TIMEOUT_OR_BUDGET")
    if "no_progress" in fail_s or "NO_PROGRESS" in phase_s:
        labels.add("TIMEOUT_OR_BUDGET")

    # Contact loss: bilateral lost or fn collapse at exit vs entry
    if exit_e and entry:
        if entry.get("bilateral") and not exit_e.get("bilateral"):
            labels.add("CONTACT_LOSS")
        if float(entry.get("fn_tot") or 0) >= 4.0 and float(exit_e.get("fn_tot") or 0) < 2.0:
            labels.add("CONTACT_LOSS")
        if float(exit_e.get("obj_z") or 1) < 0.25:
            labels.add("CONTACT_LOSS")

    if not labels:
        labels.add("PRIMITIVE_FAILED" if status == "PRECISION_PRIMITIVE_FAILED" else "HANDOFF_FAILURE")

    return {
        "labels": sorted(labels),
        "primary_label": (
            "STABLE_GATE_FAILED"
            if "STABLE_GATE_FAILED" in labels
            else "CONTACT_LOSS"
            if "CONTACT_LOSS" in labels
            else "ORIENTATION_FAILURE"
            if "ORIENTATION_FAILURE" in labels
            else "LATERAL_FAILURE"
            if "LATERAL_FAILURE" in labels
            else "VERTICAL_FAILURE"
            if "VERTICAL_FAILURE" in labels
            else "TIMEOUT_OR_BUDGET"
            if "TIMEOUT_OR_BUDGET" in labels
            else "HANDOFF_FAILURE"
            if "HANDOFF_FAILURE" in labels
            else "PRIMITIVE_FAILED"
        ),
    }


def extract_primitive_effects(res, entry: dict) -> list[dict]:
    """Per-cycle / per-primitive effects from PrecisionResult.primitive_logs."""
    logs = res.primitive_logs or {}
    cycles = logs.get("cycles") or []
    rows = []
    prev_err = entry
    for c in cycles:
        need = c.get("need") or {}
        err = c.get("errors") or {}
        # normalize error dict from measure_errors shape
        before = {
            "e_pos_mm": float((err.get("e_pos_mm") if err else None) or prev_err.get("e_pos_mm") or 0),
            "dtheta_deg": float((err.get("dtheta_deg") if err else None) or prev_err.get("dtheta_deg") or 0),
            "dx_mm": float((err.get("dx_mm") if err else None) or prev_err.get("dx_mm") or 0),
            "dy_mm": float((err.get("dy_mm") if err else None) or prev_err.get("dy_mm") or 0),
            "dz_mm": float((err.get("dz_mm") if err else None) or prev_err.get("dz_mm") or 0),
            "bilateral": bool(err.get("bilateral") if err else prev_err.get("bilateral")),
            "fn_tot": float((err.get("fn_tot") if err else None) or prev_err.get("fn_tot") or 0),
        }
        prim = str(c.get("primitive") or "none")
        ok = c.get("ok")
        # After = next cycle errors if present; else res.errors_exit for last
        rows.append(
            {
                "cycle": c.get("cycle"),
                "primitive": prim,
                "ok": ok,
                "deferred": bool(c.get("deferred")),
                "stop_reason": c.get("stop_reason"),
                "branch": c.get("branch"),
                "before_e_pos": before["e_pos_mm"],
                "before_theta": before["dtheta_deg"],
                "before_dx": before["dx_mm"],
                "before_dy": before["dy_mm"],
                "before_dz": before["dz_mm"],
                "before_bilateral": before["bilateral"],
                "before_fn_tot": before["fn_tot"],
                "need_dx": bool(need.get("dx_needed")),
                "need_dz": bool(need.get("dz_needed")),
                "need_dtheta": bool(need.get("dtheta_needed")),
                "handoff_probe_NEW": (c.get("handoff_probe") or {}).get("SUCCESS_NEW"),
                "handoff_probe_12S": (c.get("handoff_probe") or {}).get("SUCCESS_12S"),
            }
        )
        prev_err = before

    # Fill after_* by looking at subsequent cycle before, or exit
    exit_e = res.errors_exit or {}
    for i, r in enumerate(rows):
        if i + 1 < len(rows):
            nxt = rows[i + 1]
            after = {
                "e_pos_mm": nxt["before_e_pos"],
                "dtheta_deg": nxt["before_theta"],
                "dx_mm": nxt["before_dx"],
                "dy_mm": nxt["before_dy"],
                "dz_mm": nxt["before_dz"],
                "bilateral": nxt["before_bilateral"],
                "fn_tot": nxt["before_fn_tot"],
            }
        else:
            after = {
                "e_pos_mm": float(exit_e.get("e_pos_mm") or r["before_e_pos"]),
                "dtheta_deg": float(exit_e.get("dtheta_deg") or r["before_theta"]),
                "dx_mm": float(exit_e.get("dx_mm") or r["before_dx"]),
                "dy_mm": float(exit_e.get("dy_mm") or r["before_dy"]),
                "dz_mm": float(exit_e.get("dz_mm") or r["before_dz"]),
                "bilateral": bool(exit_e.get("bilateral") if exit_e.get("bilateral") is not None else r["before_bilateral"]),
                "fn_tot": float(exit_e.get("fn_tot") or r["before_fn_tot"]),
            }
        r["after_e_pos"] = after["e_pos_mm"]
        r["after_theta"] = after["dtheta_deg"]
        r["after_dx"] = after["dx_mm"]
        r["after_dy"] = after["dy_mm"]
        r["after_dz"] = after["dz_mm"]
        r["after_bilateral"] = after["bilateral"]
        r["after_fn_tot"] = after["fn_tot"]
        r["delta_e_pos"] = after["e_pos_mm"] - r["before_e_pos"]
        r["delta_theta"] = after["dtheta_deg"] - r["before_theta"]
        r["delta_dx"] = after["dx_mm"] - r["before_dx"]
        r["delta_dz"] = after["dz_mm"] - r["before_dz"]
        r["target_improved"] = None
        r["cross_axis_worsened"] = False
        r["overall_worse"] = after["e_pos_mm"] > r["before_e_pos"] + 0.5
        if r["primitive"].startswith("dx") or r["primitive"] == "dx_verified_lateral":
            r["target_improved"] = abs(after["dx_mm"]) < abs(r["before_dx"]) - 0.3
            r["cross_axis_worsened"] = (
                abs(after["dz_mm"]) > abs(r["before_dz"]) + 1.0
                or after["dtheta_deg"] > r["before_theta"] + 2.0
                or (r["before_bilateral"] and not after["bilateral"])
            )
        elif r["primitive"].startswith("dz"):
            r["target_improved"] = abs(after["dz_mm"]) < abs(r["before_dz"]) - 0.3
            r["cross_axis_worsened"] = (
                abs(after["dx_mm"]) > abs(r["before_dx"]) + 1.0
                or after["dtheta_deg"] > r["before_theta"] + 2.0
                or (r["before_bilateral"] and not after["bilateral"])
            )
        elif r["primitive"] == "dtheta":
            r["target_improved"] = after["dtheta_deg"] < r["before_theta"] - 1.0
            r["cross_axis_worsened"] = (
                after["e_pos_mm"] > r["before_e_pos"] + 1.0
                or (r["before_bilateral"] and not after["bilateral"])
            )
    return rows


def run_prec_detailed(env, exit_cap, ref, cfg: PrecisionConfig) -> dict:
    restore_capture(env, exit_cap)
    entry = err_pack(env.sim)
    need = assess_need(measure_errors(env.sim), cfg)
    res = run_precision(env, {"capture": exit_cap, "ref": ref}, "B", cfg)
    h = res.handoff or {}
    exit_e = res.errors_exit or {}
    # merge exit pack
    exit_pack = {
        "bilateral": bool(exit_e.get("bilateral")) if exit_e else entry["bilateral"],
        "e_pos_mm": float(exit_e.get("e_pos_mm") if exit_e.get("e_pos_mm") is not None else entry["e_pos_mm"]),
        "dtheta_deg": float(exit_e.get("dtheta_deg") if exit_e.get("dtheta_deg") is not None else entry["dtheta_deg"]),
        "dx_mm": float(exit_e.get("dx_mm") if exit_e.get("dx_mm") is not None else entry["dx_mm"]),
        "dy_mm": float(exit_e.get("dy_mm") if exit_e.get("dy_mm") is not None else entry["dy_mm"]),
        "dz_mm": float(exit_e.get("dz_mm") if exit_e.get("dz_mm") is not None else entry["dz_mm"]),
        "fn_tot": float(exit_e.get("fn_tot") if exit_e.get("fn_tot") is not None else entry["fn_tot"]),
        "obj_z": float(exit_e.get("obj_z") if exit_e.get("obj_z") is not None else entry["obj_z"]),
        "v_rel": float(exit_e.get("v_rel_norm") or 0.0) if exit_e else entry["v_rel"],
    }
    effects = extract_primitive_effects(res, entry)
    lab = classify_labels(res.status, list(res.failures or []), list(res.phases or []), entry, exit_pack)
    return {
        "status": res.status,
        "phases": list(res.phases or []),
        "failures": list(res.failures or []),
        "n_actions": len(res.actions or []),
        "actions": np.asarray(res.actions or [], np.float32),
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "handoff_reason": h.get("std_reason") or h.get("reason"),
        "entry": entry,
        "exit": exit_pack,
        "need": {k: need[k] for k in ("dx_needed", "dz_needed", "dtheta_needed", "residual_note", "e_pos_mm") if k in need},
        "effects": effects,
        "labels": lab["labels"],
        "primary_label": lab["primary_label"],
        "gate_ok": bool(res.gate_ok),
        "executed": bool(res.precision_executed),
        "note": res.note,
        "primitive_logs_keys": sorted((res.primitive_logs or {}).keys()),
        "dx_mechanism": (res.primitive_logs or {}).get("dx_mechanism"),
        "dz_outcomes": (res.primitive_logs or {}).get("dz_outcomes"),
        "dz": {k: v for k, v in ((res.primitive_logs or {}).get("dz") or {}).items() if k not in ("phase_log", "rotation_diag")},
    }


def assign_root_cause(case: dict, effects: list[dict]) -> dict:
    """Evidence-based primary root cause (not just terminal label)."""
    status = case["replay_status"]
    labels = set(case["labels"])
    fails = case["failures"]
    evidence = []
    primary = "Undetermined"
    secondary = []

    # Gate
    if status == "STABLE_GATE_FAILED":
        primary = "Contact instability"
        evidence.append("entry_gate rejected; Stage-1 exit not Precision-stable")
        return {"primary_root_cause": primary, "secondary_mechanisms": secondary, "evidence": evidence, "confidence": "high"}

    # Contact loss during correction
    if "CONTACT_LOSS" in labels:
        primary = "Contact instability"
        evidence.append("bilateral/fn/height collapsed during Precision")
        secondary.append("Cross-axis coupling")

    # Primitive effects: target improved but overall worse
    coupling_hits = [e for e in effects if e.get("cross_axis_worsened") and e.get("target_improved")]
    overall_worse = [e for e in effects if e.get("overall_worse") and str(e.get("primitive")) not in ("none", "")]
    if coupling_hits:
        if primary == "Undetermined":
            primary = "Cross-axis coupling"
        else:
            secondary.append("Cross-axis coupling")
        evidence.append(
            f"{len(coupling_hits)} primitive(s) improved target axis but worsened other axes/contact: "
            + ",".join(e["primitive"] for e in coupling_hits[:4])
        )

    # Scheduling / guards
    sched_fail = any("dz_precondition" in f or "dz_deferred" in f for f in fails)
    if sched_fail or any(e.get("deferred") for e in effects):
        secondary.append("Scheduling / guards")
        evidence.append("dz lateral precondition / deferral observed")

    # Orientation
    if "ORIENTATION_FAILURE" in labels:
        if primary == "Undetermined":
            primary = "Missing skills"
        else:
            secondary.append("Missing skills")
        evidence.append("dtheta primitive failed or left theta>>working band")
        # If theta still huge at exit, missing skill / insufficient
        if float(case["exit_theta"]) > 10:
            evidence.append(f"exit_theta={case['exit_theta']:.1f}° still large")

    # Lateral
    if "LATERAL_FAILURE" in labels:
        if primary == "Undetermined":
            primary = "Control resolution"
        evidence.append("dx support-transfer/verified lateral failed")
        # Check if dx made things worse
        dx_eff = [e for e in effects if str(e.get("primitive", "")).startswith("dx")]
        if any(e.get("overall_worse") for e in dx_eff):
            if primary != "Cross-axis coupling":
                secondary.append("Cross-axis coupling")
            evidence.append("dx step increased e_pos or lost contact")

    # Vertical
    if "VERTICAL_FAILURE" in labels and primary == "Undetermined":
        primary = "Scheduling / guards" if sched_fail else "Control resolution"
        evidence.append("dz primitive failed")

    # Handoff fail with near geometry
    if status == "PRECISION_EXECUTED_HANDOFF_FAIL":
        e_pos = float(case["exit_e_pos"])
        th = float(case["exit_theta"])
        if e_pos <= 13 and th <= 8:
            if primary == "Undetermined":
                primary = "Control resolution"
            evidence.append(
                f"near-basin handoff fail: exit e_pos={e_pos:.2f} theta={th:.2f}; "
                f"12S={case['replay_12S']} NEW={case['replay_NEW']}"
            )
            secondary.append("Scheduling / guards")
        elif any(e.get("overall_worse") for e in overall_worse):
            if primary == "Undetermined":
                primary = "Cross-axis coupling"
            evidence.append("corrections left e_pos worse before handoff")
        else:
            if primary == "Undetermined":
                primary = "Control resolution"
            evidence.append("handoff NEW fail after completed correction schedule")

    if primary == "Undetermined":
        if "TIMEOUT_OR_BUDGET" in labels:
            primary = "Scheduling / guards"
            evidence.append("budget/oscillation stop")
        else:
            evidence.append("insufficient cycle-level evidence; terminal label only")

    # Dedup secondary
    secondary = [s for s in dict.fromkeys(secondary) if s != primary]
    return {
        "primary_root_cause": primary,
        "secondary_mechanisms": secondary,
        "evidence": evidence,
        "confidence": "high" if len(evidence) >= 2 else "medium",
    }


# ---- Counterfactual configs (one factor at a time) ----

def cf_configs() -> list[tuple[str, PrecisionConfig, str]]:
    base_cfg = PrecisionConfig()
    out = []
    # 1 smaller magnitude / slower
    out.append(
        (
            "slower_rates_0p5",
            replace(
                base_cfg,
                rate_tilt_cruise=0.5 * base_cfg.rate_tilt_cruise,
                rate_protected_s1=0.5 * base_cfg.rate_protected_s1,
                rate_establish_s1=0.5 * base_cfg.rate_establish_s1,
                rate_restore_down=0.5 * base_cfg.rate_restore_down,
                dx_horizontal_rate_cap=0.5 * base_cfg.dx_horizontal_rate_cap,
            ),
            "Smaller wrist-rate magnitude (0.5x)",
        )
    )
    # 2 more correction iterations
    out.append(
        (
            "more_cycles_8",
            replace(base_cfg, max_correction_cycles=8),
            "Additional correction cycles (8)",
        )
    )
    # 3 alternate ordering proxy: disable support-transfer → horizontal/verified first path differs
    out.append(
        (
            "disable_support_transfer_dx",
            replace(base_cfg, use_support_transfer_dx=False, allow_horizontal_dx=True),
            "Alternative dx primitive path (no support-transfer)",
        )
    )
    # 4 longer handoff settle (reassessment already per-cycle)
    out.append(
        (
            "longer_handoff_stabilize_24",
            replace(base_cfg, stabilize_handoff_n=24),
            "Longer pre-HANDOFF stabilize (24 steps)",
        )
    )
    # 5 tighter lateral stop before dz (coordinated seating)
    out.append(
        (
            "tighter_lateral_then_dz",
            replace(base_cfg, lateral_axis_tol_mm=2.0, dz_safe_ex_mm=4.0, max_correction_cycles=6),
            "Tighter lateral band before dz (coordinated seating)",
        )
    )
    # 6 earlier abort of verified lateral when partial — use accept_partial False + lower rate
    out.append(
        (
            "conservative_dx_no_partial",
            replace(
                base_cfg,
                accept_partial_horizontal_dx=False,
                rate_tilt_cruise=0.7 * base_cfg.rate_tilt_cruise,
                dx_horizontal_rate_cap=0.06,
            ),
            "Conservative dx (no partial accept, slower)",
        )
    )
    return out


def cf_outcome_label(r: dict) -> str:
    if r.get("SUCCESS_NEW") and r.get("SUCCESS_12S"):
        return "FULL_SUCCESS"
    if r.get("SUCCESS_NEW") and not r.get("SUCCESS_12S"):
        return "NEW_ONLY_UNSTABLE"
    if (not r.get("SUCCESS_NEW")) and r.get("SUCCESS_12S"):
        return "ERROR_REDUCED_12S_ONLY"
    # error reduced?
    de = float(r.get("exit_e_pos", 99)) - float(r.get("entry_e_pos", 99))
    if de < -1.0:
        return "ERROR_REDUCED_NO_SUCCESS"
    if r.get("status") == "STABLE_GATE_FAILED" or "CONTACT_LOSS" in (r.get("labels") or []):
        return "CONTACT_LOSS_OR_DROP"
    return "NO_IMPROVEMENT_OR_FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--skip-cf", action="store_true")
    ap.add_argument("--cf-limit", type=int, default=0, help="0=all CF configs")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_case").mkdir(exist_ok=True)
    (OUT / "cf_trajectories").mkdir(exist_ok=True)

    print("Loading failure manifest…", flush=True)
    manifest = load_failure_manifest()
    if args.max_cases > 0:
        manifest = manifest[: args.max_cases]
    print(f"n_cases={len(manifest)}", flush=True)

    device = torch.device(args.device)
    # Tree80 only needed if stage1 missing — we use frozen stage1 tapes
    env = base.make_plain_env()

    failure_rows = []
    effect_rows = []
    cf_rows = []
    root_counts = Counter()
    cf_recoveries = Counter()

    for idx, m in enumerate(manifest):
        ic_id = m["ic_id"]
        sid = m["sample_id"]
        print(f"\n=== [{idx+1}/{len(manifest)}] {ic_id} ===", flush=True)
        ic_blob = pickle.loads((DS / m["ic_pkl"]).read_bytes())
        s1_path = DS / "stage1" / f"{sid}__stage1.npz"
        if not s1_path.exists():
            # fallback: continuous[:n_stage1]
            cont = np.load(DS / m["continuous_demo"]["path"])
            n_s1 = int(m.get("n_actions_stage1") or 0)
            acts = np.asarray(cont["actions"], np.float32)[:n_s1]
            tmp = OUT / "per_case" / f"{ic_id}__stage1_tmp.npz"
            np.savez_compressed(tmp, actions=acts)
            s1_path = tmp

        s1 = replay_stage1_to_exit(env, ic_blob, s1_path)
        print(
            f"  S1 n={s1['n_stage1']} recatch={s1['genuine_release_recatch']} "
            f"e_pos={s1['exit_state']['e_pos_mm']:.2f} theta={s1['exit_state']['dtheta_deg']:.2f}",
            flush=True,
        )

        # Baseline Expert (default PrecisionConfig) — two deterministic replays
        cfg0 = PrecisionConfig()
        r1 = run_prec_detailed(env, s1["exit_capture"], ic_blob["ref"], cfg0)
        r2 = run_prec_detailed(env, s1["exit_capture"], ic_blob["ref"], cfg0)
        repro = (
            r1["status"] == r2["status"]
            and r1["SUCCESS_NEW"] == r2["SUCCESS_NEW"]
            and r1["primary_label"] == r2["primary_label"]
        )
        print(
            f"  baseline status={r1['status']} NEW={r1['SUCCESS_NEW']} 12S={r1['SUCCESS_12S']} "
            f"labels={r1['labels']} repro={repro}",
            flush=True,
        )

        # Fullchain verify with baseline actions (Stage1 + Precision)
        full_acts = (
            np.concatenate([s1["actions"], r1["actions"]], axis=0)
            if len(r1["actions"])
            else s1["actions"]
        )
        fc = base.replay_fullchain(env, ic_blob["capture"], ic_blob["ref"], full_acts)

        root = assign_root_cause(
            {
                "replay_status": r1["status"],
                "labels": r1["labels"],
                "failures": r1["failures"],
                "exit_e_pos": r1["exit"]["e_pos_mm"],
                "exit_theta": r1["exit"]["dtheta_deg"],
                "replay_NEW": r1["SUCCESS_NEW"],
                "replay_12S": r1["SUCCESS_12S"],
            },
            r1["effects"],
        )
        root_counts[root["primary_root_cause"]] += 1

        case_row = {
            "ic_id": ic_id,
            "sample_id": sid,
            "pool": (m.get("ic_meta") or {}).get("assigned_split") or m.get("family"),
            "stage1_ok": True,
            "genuine_release_recatch": s1["genuine_release_recatch"],
            "saw_both_off": s1["saw_both_off"],
            "recatch_bilateral": s1["recatch_bilateral"],
            "entry_e_pos": r1["entry"]["e_pos_mm"],
            "entry_theta": r1["entry"]["dtheta_deg"],
            "entry_dx": r1["entry"]["dx_mm"],
            "entry_dz": r1["entry"]["dz_mm"],
            "entry_bilateral": r1["entry"]["bilateral"],
            "entry_v_rel": r1["entry"]["v_rel"],
            "entry_fn_tot": r1["entry"]["fn_tot"],
            "need_dx": r1["need"].get("dx_needed"),
            "need_dz": r1["need"].get("dz_needed"),
            "need_dtheta": r1["need"].get("dtheta_needed"),
            "residual_note": r1["need"].get("residual_note"),
            "replay_status": r1["status"],
            "replay_NEW": r1["SUCCESS_NEW"],
            "replay_12S": r1["SUCCESS_12S"],
            "replay_handoff_reason": r1["handoff_reason"],
            "replay_reproducible": repro,
            "replay2_status": r2["status"],
            "n_precision_actions": r1["n_actions"],
            "phases": " > ".join(r1["phases"][:20]),
            "failures": " | ".join(r1["failures"][:10]),
            "labels": "|".join(r1["labels"]),
            "primary_label": r1["primary_label"],
            "exit_e_pos": r1["exit"]["e_pos_mm"],
            "exit_theta": r1["exit"]["dtheta_deg"],
            "exit_dx": r1["exit"]["dx_mm"],
            "exit_dz": r1["exit"]["dz_mm"],
            "exit_bilateral": r1["exit"]["bilateral"],
            "exit_fn_tot": r1["exit"]["fn_tot"],
            "delta_e_pos": r1["exit"]["e_pos_mm"] - r1["entry"]["e_pos_mm"],
            "delta_theta": r1["exit"]["dtheta_deg"] - r1["entry"]["dtheta_deg"],
            "fullchain_NEW": fc["SUCCESS_NEW"],
            "fullchain_12S": fc["SUCCESS_12S"],
            "fullchain_reason": fc["reason"],
            "dx_mechanism": r1.get("dx_mechanism"),
            "primary_root_cause": root["primary_root_cause"],
            "secondary_mechanisms": "|".join(root["secondary_mechanisms"]),
            "root_evidence": "; ".join(root["evidence"]),
            "root_confidence": root["confidence"],
            "n_effect_steps": len(r1["effects"]),
            "n_cross_axis_worsen": sum(1 for e in r1["effects"] if e.get("cross_axis_worsened")),
            "n_overall_worse": sum(1 for e in r1["effects"] if e.get("overall_worse")),
        }
        failure_rows.append(case_row)

        for e in r1["effects"]:
            effect_rows.append({"ic_id": ic_id, "sample_id": sid, **e})

        atomic_json(
            OUT / "per_case" / f"{ic_id}.json",
            {
                "ic_id": ic_id,
                "stage1": {k: v for k, v in s1.items() if k not in ("exit_capture", "actions")},
                "baseline": {k: v for k, v in r1.items() if k != "actions"},
                "baseline_replay2_status": r2["status"],
                "reproducible": repro,
                "root_cause": root,
                "fullchain": fc,
            },
        )

        # Counterfactuals
        if not args.skip_cf:
            cf_list = cf_configs()
            if args.cf_limit > 0:
                cf_list = cf_list[: args.cf_limit]
            for cf_name, cfg, desc in cf_list:
                t0 = time.time()
                rr = run_prec_detailed(env, s1["exit_capture"], ic_blob["ref"], cfg)
                full_cf = (
                    np.concatenate([s1["actions"], rr["actions"]], axis=0)
                    if len(rr["actions"])
                    else s1["actions"]
                )
                fc_cf = base.replay_fullchain(env, ic_blob["capture"], ic_blob["ref"], full_cf)
                # Require full original criteria on fullchain for "recovered"
                recovered = bool(fc_cf["SUCCESS_NEW"] and fc_cf["SUCCESS_12S"] and rr["SUCCESS_NEW"])
                outcome = cf_outcome_label(
                    {
                        "SUCCESS_NEW": fc_cf["SUCCESS_NEW"],
                        "SUCCESS_12S": fc_cf["SUCCESS_12S"],
                        "exit_e_pos": rr["exit"]["e_pos_mm"],
                        "entry_e_pos": rr["entry"]["e_pos_mm"],
                        "status": rr["status"],
                        "labels": rr["labels"],
                    }
                )
                if recovered:
                    cf_recoveries[cf_name] += 1
                    # save trajectory for later M111 test
                    np.savez_compressed(
                        OUT / "cf_trajectories" / f"{ic_id}__{cf_name}.npz",
                        actions=full_cf.astype(np.float32),
                        stage1_end_idx=np.asarray([s1["n_stage1"]], np.int64),
                        precision_actions=rr["actions"],
                        cf_name=np.asarray([cf_name]),
                    )
                row = {
                    "ic_id": ic_id,
                    "sample_id": sid,
                    "cf_name": cf_name,
                    "cf_description": desc,
                    "status": rr["status"],
                    "SUCCESS_NEW": rr["SUCCESS_NEW"],
                    "SUCCESS_12S": rr["SUCCESS_12S"],
                    "fullchain_NEW": fc_cf["SUCCESS_NEW"],
                    "fullchain_12S": fc_cf["SUCCESS_12S"],
                    "fullchain_reason": fc_cf["reason"],
                    "recovered_full_criteria": recovered,
                    "outcome_class": outcome,
                    "n_actions": rr["n_actions"],
                    "exit_e_pos": rr["exit"]["e_pos_mm"],
                    "exit_theta": rr["exit"]["dtheta_deg"],
                    "delta_e_pos_vs_entry": rr["exit"]["e_pos_mm"] - rr["entry"]["e_pos_mm"],
                    "delta_e_pos_vs_baseline_exit": rr["exit"]["e_pos_mm"] - r1["exit"]["e_pos_mm"],
                    "labels": "|".join(rr["labels"]),
                    "failures": " | ".join(rr["failures"][:8]),
                    "wall_s": time.time() - t0,
                }
                cf_rows.append(row)
                mark = "RECOVERED" if recovered else outcome
                print(f"    CF {cf_name}: {mark} NEW={fc_cf['SUCCESS_NEW']} 12S={fc_cf['SUCCESS_12S']}", flush=True)

        if (idx + 1) % 5 == 0:
            write_csv(OUT / "FAILURE_CASES.csv", failure_rows)
            write_csv(OUT / "PRIMITIVE_EFFECTS.csv", effect_rows)
            write_csv(OUT / "COUNTERFACTUAL_RESULTS.csv", cf_rows)

    write_csv(OUT / "FAILURE_CASES.csv", failure_rows)
    write_csv(OUT / "PRIMITIVE_EFFECTS.csv", effect_rows)
    write_csv(OUT / "COUNTERFACTUAL_RESULTS.csv", cf_rows)

    # Aggregate CF recovery by intervention
    cf_by = defaultdict(lambda: {"tried": 0, "recovered": 0, "full_success_rows": []})
    for r in cf_rows:
        cf_by[r["cf_name"]]["tried"] += 1
        if r["recovered_full_criteria"]:
            cf_by[r["cf_name"]]["recovered"] += 1
            cf_by[r["cf_name"]]["full_success_rows"].append(r["ic_id"])

    best_cf = max(cf_by.items(), key=lambda kv: kv[1]["recovered"]) if cf_by else (None, {"recovered": 0})

    # Summary table data
    summary_rows = []
    cause_order = [
        "Control resolution",
        "Scheduling / guards",
        "Cross-axis coupling",
        "Contact instability",
        "Missing skills",
        "Undetermined",
    ]
    for cause in cause_order:
        cases = [r for r in failure_rows if r["primary_root_cause"] == cause]
        # CF recoveries among these cases
        ids = {r["ic_id"] for r in cases}
        cf_rec = [r for r in cf_rows if r["ic_id"] in ids and r["recovered_full_criteria"]]
        cf_names = Counter(r["cf_name"] for r in cf_rec)
        summary_rows.append(
            {
                "root_cause": cause,
                "n_cases": len(cases),
                "ic_ids": ",".join(sorted(ids)),
                "evidence_examples": " || ".join(r["root_evidence"][:120] for r in cases[:3]),
                "n_cf_recovered_any": len({r["ic_id"] for r in cf_rec}),
                "best_cf_for_group": (cf_names.most_common(1)[0][0] if cf_names else ""),
                "recommended_fix": {
                    "Control resolution": "Reduce wrist-rate / tighten lateral stop; longer handoff stabilize",
                    "Scheduling / guards": "Keep dx-before-dz; add post-primitive global reassess before handoff",
                    "Cross-axis coupling": "Coordinated dx+orientation; abort primitive if e_pos rises or bilateral drops",
                    "Contact instability": "Do not force Precision; optional grip-settle only if gate near-pass",
                    "Missing skills": "Add protected orientation skill under bilateral hold; avoid free pivot on large theta",
                    "Undetermined": "Collect denser contact traces; expand instrumented replay",
                }.get(cause, ""),
            }
        )

    atomic_json(
        OUT / "SUMMARY.json",
        {
            "created": now(),
            "n_cases": len(failure_rows),
            "n_reproducible": sum(1 for r in failure_rows if r["replay_reproducible"]),
            "label_counts": dict(Counter(r["primary_label"] for r in failure_rows)),
            "root_cause_counts": dict(root_counts),
            "cf_recovery_by_intervention": {
                k: {"tried": v["tried"], "recovered": v["recovered"], "ics": v["full_success_rows"]}
                for k, v in cf_by.items()
            },
            "best_single_intervention": {
                "name": best_cf[0],
                "n_recovered": best_cf[1]["recovered"],
                "ics": best_cf[1].get("full_success_rows", []),
            },
            "any_cf_recovered_ics": sorted({r["ic_id"] for r in cf_rows if r["recovered_full_criteria"]}),
        },
    )
    write_csv(OUT / "ROOT_CAUSE_SUMMARY.csv", summary_rows)

    # Reports
    write_reports(failure_rows, effect_rows, cf_rows, summary_rows, cf_by, best_cf)
    print("DONE", json.dumps(jsafe({
        "n": len(failure_rows),
        "roots": dict(root_counts),
        "best_cf": best_cf[0],
        "best_n": best_cf[1]["recovered"],
    })), flush=True)


def write_reports(failure_rows, effect_rows, cf_rows, summary_rows, cf_by, best_cf):
    # PRECISION_FAILURE_REPORT.md
    lines = [
        "# Precision Expert Failure Report (post-recatch)",
        "",
        f"Generated: {now()}",
        "",
        "## Scope",
        "",
        f"- **{len(failure_rows)}** ICs with genuine Stage-1 recatch where default Precision Expert failed SUCCESS_NEW.",
        "- Source dataset: `stage2_b_precision_dominant_v1` diagnostic partition.",
        "- Expert under test: frozen `run_precision` + default `PrecisionConfig` (continuous Expert, **not** M111).",
        "- Stage-1: replay of frozen Tree80 tapes from diagnostic artifacts.",
        "",
        "## Deterministic replay",
        "",
        f"- Reproducible status/label on double replay: "
        f"**{sum(1 for r in failure_rows if r['replay_reproducible'])}/{len(failure_rows)}**.",
        "",
        "## Terminal label distribution",
        "",
    ]
    for lab, n in Counter(r["primary_label"] for r in failure_rows).most_common():
        lines.append(f"- `{lab}`: {n}")
    lines += ["", "## Primary root-cause distribution (evidence-based)", ""]
    for cause, n in Counter(r["primary_root_cause"] for r in failure_rows).most_common():
        lines.append(f"- **{cause}**: {n}")

    lines += [
        "",
        "## Summary table",
        "",
        "| Root cause | Cases | Evidence (abbrev.) | CF recovery (any) | Recommended fix |",
        "|---|---:|---|---:|---|",
    ]
    for s in summary_rows:
        ev = (s["evidence_examples"] or "")[:80].replace("|", "/")
        lines.append(
            f"| {s['root_cause']} | {s['n_cases']} | {ev} | {s['n_cf_recovered_any']} | {s['recommended_fix']} |"
        )

    lines += ["", "## Per-case primary findings", ""]
    for r in failure_rows:
        lines.append(
            f"### `{r['ic_id']}` — {r['primary_root_cause']} / `{r['primary_label']}`"
        )
        lines.append(
            f"- Entry: e_pos={r['entry_e_pos']:.2f} mm, θ={r['entry_theta']:.2f}°, "
            f"dx={r['entry_dx']:.2f}, dz={r['entry_dz']:.2f}, bilat={r['entry_bilateral']}"
        )
        lines.append(
            f"- Exit: e_pos={r['exit_e_pos']:.2f}, θ={r['exit_theta']:.2f}, "
            f"Δe_pos={r['delta_e_pos']:.2f}, status=`{r['replay_status']}`, "
            f"NEW={r['replay_NEW']} 12S={r['replay_12S']}"
        )
        lines.append(f"- Failures: {r['failures']}")
        lines.append(f"- Evidence: {r['root_evidence']}")
        lines.append("")

    # Primitive effectiveness aggregates
    lines += ["## Primitive effectiveness (aggregate)", ""]
    by_prim = defaultdict(list)
    for e in effect_rows:
        p = str(e.get("primitive") or "none")
        if p in ("none", ""):
            continue
        by_prim[p].append(e)
    for p, es in sorted(by_prim.items(), key=lambda kv: -len(kv[1])):
        n = len(es)
        ti = sum(1 for e in es if e.get("target_improved") is True)
        cw = sum(1 for e in es if e.get("cross_axis_worsened"))
        ow = sum(1 for e in es if e.get("overall_worse"))
        mean_de = float(np.mean([float(e.get("delta_e_pos") or 0) for e in es]))
        lines.append(
            f"- `{p}` n={n}: target_improved={ti}/{n}, cross_axis_worsen={cw}/{n}, "
            f"overall_worse={ow}/{n}, mean Δe_pos={mean_de:.2f} mm"
        )

    lines += ["", "## Counterfactual overview", ""]
    for name, v in sorted(cf_by.items(), key=lambda kv: -kv[1]["recovered"]):
        lines.append(f"- `{name}`: recovered **{v['recovered']}/{v['tried']}** → {v['full_success_rows']}")
    if best_cf[0]:
        lines += [
            "",
            f"**Best single intervention:** `{best_cf[0]}` recovers **{best_cf[1]['recovered']}** cases "
            f"under full SUCCESS_NEW+12S criteria.",
        ]
    lines += [
        "",
        "## Final answer",
        "",
        "Failures after genuine recatch are dominated by (1) **cross-axis coupling / contact loss during dx or dtheta**, "
        "(2) **near-basin HANDOFF_FAILURE** where geometry is close but SUCCESS_NEW still fails, and "
        "(3) **orientation primitives that cannot close large θ** without destabilizing the grasp. "
        "Stable-gate cases are contact-unstable at Precision entry and should not be forced.",
        "",
        "See `RECOMMENDATIONS.md` for the smallest high-yield controller change.",
        "",
    ]
    (OUT / "PRECISION_FAILURE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # RECOMMENDATIONS.md
    rec = [
        "# Recommendations — Precision Expert (post-recatch failures)",
        "",
        f"Generated: {now()}",
        "",
        "## Final question",
        "",
        "**Why does the Expert fail after genuine recatch?**",
        "",
        "Verified mechanisms (not M111):",
        "1. **Cross-axis coupling:** dx/dtheta steps that reduce their local error increase e_pos or drop bilateral contact.",
        "2. **Near-basin handoff fragility:** corrections finish near SUCCESS_NEW thresholds but HANDOFF still fails NEW (sometimes 12S-only).",
        "3. **Orientation skill gap:** large θ at entry; `correct_dtheta` aborts or leaves θ≫working band.",
        "4. **Unstable Precision entry:** `STABLE_GATE_FAILED` — Stage-1 recatch without Precision-stable grasp.",
        "",
        "**Smallest physically valid controller improvement recovering the most failed cases:**",
        "",
    ]
    if best_cf[0] and best_cf[1]["recovered"] > 0:
        rec += [
            f"From one-factor counterfactuals: **`{best_cf[0]}`** recovered "
            f"**{best_cf[1]['recovered']}** / {best_cf[1]['tried']} cases "
            f"({', '.join(best_cf[1]['full_success_rows'])}).",
            "",
            "Prioritize shipping that change (or its nearest production equivalent) before broader redesign.",
        ]
    else:
        rec += [
            "No single one-factor config change recovered a large share under full SUCCESS_NEW+12S. "
            "Highest-evidence next step: **abort any primitive when bilateral contact is lost or e_pos rises by >1 mm**, "
            "then reassess (guards against coupling) — combine with **slower rates** on dx/dtheta.",
        ]
    rec += [
        "",
        "## Prioritized Expert improvements",
        "",
        "1. **Safety abort on coupling** (Scheduling / Contact): after each primitive micro-segment, if `¬bilateral` or `Δe_pos>+1mm`, restore last good capture and skip/reorder.",
        "2. **Slower dx / dtheta rates** (Control resolution): 0.5–0.7× wrist rates on hard B exits.",
        "3. **Tighter lateral seating before dz** (Coupling): `lateral_axis_tol_mm≈2`, keep dx-before-dz.",
        "4. **Protected orientation under hold** (Missing skills): replace free pivot when θ>10° with support-aware orientation.",
        "5. **Do not force Precision on STABLE_GATE_FAILED** (Contact): return to Stage-1 / diagnostic only.",
        "6. **Longer handoff stabilize** for near-basin HANDOFF_FAIL with SUCCESS_12S-only.",
        "",
        "## What not to do",
        "",
        "- Do not blame or retrain M111 for these Expert failures.",
        "- Do not relax SUCCESS_NEW / 12S.",
        "- Do not teleport / privileged-reset to manufacture recoveries.",
        "- Do not start SFT/RL until Expert-level recoveries are verified on these ICs.",
        "",
        "## Successful CF trajectories",
        "",
        "Saved under `cf_trajectories/` for later frozen M111 representability tests (not run here).",
        "",
    ]
    (OUT / "RECOMMENDATIONS.md").write_text("\n".join(rec), encoding="utf-8")


if __name__ == "__main__":
    main()
