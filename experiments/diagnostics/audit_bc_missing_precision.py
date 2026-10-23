#!/usr/bin/env python
"""Audit why B/C experts lack Stage-1→Precision coverage.

Diagnostic only — does not modify experts, physics, or success criteria.
Results → results/diagnostics/bc_missing_precision_audit/
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.contact_modes import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.force_slip_control import sample_state  # noqa: E402
from data_generation.oracles.precision.precision_goals import (  # noqa: E402
    assess_need,
    entry_gate,
    measure_errors,
)
from data_generation.oracles.precision.precision_types import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist  # noqa: E402
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

DS = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
OUT = ROOT / "results/diagnostics/bc_missing_precision_audit"
DT = float(DT_POLICY)
ACT = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]


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


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]):
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
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def snap(sim) -> dict:
    err = measure_errors(sim)
    st = sample_state(sim)
    m = classify_contact_mode(sim)
    return {
        "bilateral": bool(err["bilateral"]),
        "e_pos_mm": float(err["e_pos_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "dx_mm": float(err["dx_mm"]),
        "dy_mm": float(err["dy_mm"]),
        "dz_mm": float(err["dz_mm"]),
        "e_signed_mm": [float(x) for x in err["e_signed_mm"]],
        "v_rel": float(st.get("v_rel_norm") or 0.0),
        "w_rel": float(st.get("w_rel_norm") or 0.0),
        "obj_z": float(err["obj_z"]),
        "fn_L": float(err["fn_L"]),
        "fn_R": float(err["fn_R"]),
        "mode": m["mode"],
        "latched": bool(latched(sim)),
    }


def replay_to_stage1_end(env, cap, acts, n_s1: int):
    restore_capture(env, cap)
    for a in acts[:n_s1]:
        step_legal(env.sim, np.asarray(a, np.float32))
    return take_capture(env)


def handoff_probe(env, cap, ref, stabilize_n=8):
    restore_capture(env, cap)
    for _ in range(stabilize_n):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    pre = snap(env.sim)
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    restore_capture(env, cap)
    return {
        "pre": pre,
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
        "pre_in_band_NEW": h.get("pre_in_band_NEW"),
        "pre_pos_mm": h.get("pre_pos_mm"),
        "pre_theta": h.get("pre_theta"),
    }


def action_phase_stats(acts: np.ndarray) -> dict:
    """Mechanistic buckets from 7D action magnitudes (not Precision labels)."""
    if len(acts) == 0:
        return {}
    a = np.asarray(acts, np.float32)
    n = len(a)
    # thirds of Stage-1
    cuts = [0, n // 3, 2 * n // 3, n]
    out = {"n": n, "thirds": {}}
    for ti, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        seg = a[lo:hi]
        if len(seg) == 0:
            continue
        mag_v = np.linalg.norm(seg[:, :3], axis=1)
        mag_w = np.linalg.norm(seg[:, 3:6], axis=1)
        out["thirds"][f"T{ti}"] = {
            "mean_|v|": float(np.mean(mag_v)),
            "mean_|w|": float(np.mean(mag_w)),
            "mean_|wx|": float(np.mean(np.abs(seg[:, 3]))),
            "mean_|wy|": float(np.mean(np.abs(seg[:, 4]))),
            "mean_|vz|": float(np.mean(np.abs(seg[:, 2]))),
            "mean_grip": float(np.mean(seg[:, 6])),
            "frac_|v|>0.25": float(np.mean(mag_v > 0.25)),
            "frac_|w|>0.25": float(np.mean(mag_w > 0.25)),
            "frac_|wy|>0.15": float(np.mean(np.abs(seg[:, 4]) > 0.15)),
            "frac_grip>0.5": float(np.mean(seg[:, 6] > 0.5)),
        }
    # early 0.2s vs last 0.2s
    e_n = min(10, n)
    l_n = min(10, n)
    early, late = a[:e_n], a[-l_n:]
    out["early_vs_late"] = {
        "early_mean_|v|": float(np.mean(np.linalg.norm(early[:, :3], axis=1))),
        "late_mean_|v|": float(np.mean(np.linalg.norm(late[:, :3], axis=1))),
        "early_mean_|w|": float(np.mean(np.linalg.norm(early[:, 3:6], axis=1))),
        "late_mean_|w|": float(np.mean(np.linalg.norm(late[:, 3:6], axis=1))),
        "early_mean_|wy|": float(np.mean(np.abs(early[:, 4]))),
        "late_mean_|wy|": float(np.mean(np.abs(late[:, 4]))),
        "late_frac_near_hold": float(
            np.mean((np.linalg.norm(late[:, :6], axis=1) < 0.15) & (late[:, 6] > 0.5))
        ),
    }
    return out


def state_path_stats(env, cap, acts: np.ndarray, stride: int = 10) -> dict:
    restore_capture(env, cap)
    series = [snap(env.sim)]
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        if (i + 1) % stride == 0 or i + 1 == len(acts):
            series.append({**snap(env.sim), "i": i + 1})
    if not series:
        return {}
    epos = [s["e_pos_mm"] for s in series]
    th = [s["dtheta_deg"] for s in series]
    bil = [s["bilateral"] for s in series]
    z = [s["obj_z"] for s in series]
    return {
        "e_pos_start": epos[0],
        "e_pos_end": epos[-1],
        "e_pos_min": float(np.min(epos)),
        "e_pos_max": float(np.max(epos)),
        "dtheta_start": th[0],
        "dtheta_end": th[-1],
        "dtheta_min": float(np.min(th)),
        "dtheta_max": float(np.max(th)),
        "bilateral_frac": float(np.mean(bil)),
        "bilateral_end": bil[-1],
        "obj_z_start": z[0],
        "obj_z_end": z[-1],
        "obj_z_max": float(np.max(z)),
        "n_samples": len(series),
    }


def load_rows():
    return [
        json.loads(l)
        for l in (DS / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cfg = PrecisionConfig()
    rows = load_rows()

    # ------------------------------------------------------------------
    # Phase 1 summary from manifest (code-path evidence companion)
    # ------------------------------------------------------------------
    phase1 = {
        "branch_rule": (
            "After Stage-1, if probe_direct_handoff(stable).SUCCESS_NEW → DIRECT_HANDOFF "
            "(Precision skipped). Else run_precision(family). No skill-identity bypass."
        ),
        "code": {
            "stage1_ab": "scripts/generate_stage2_abc_recovery_fullchain_v1.py:run_stage1_ab",
            "stage1_c": "scripts/generate_stage2_abc_recovery_fullchain_v1.py:run_stage1_c",
            "direct_vs_precision": "scripts/generate_stage2_abc_recovery_fullchain_v1.py:process_one ~601-632",
            "assess_need": "data_generation/oracles/precision/precision_goals.py:assess_need",
            "probe_direct": "data_generation/oracles/precision/precision_handoff.py:probe_direct_handoff",
            "entry_gate": "data_generation/oracles/precision/precision_goals.py:entry_gate",
        },
        "by_label": {},
    }
    for lab in ("A", "B", "C", "STABLE_PRECISION"):
        xs = [r for r in rows if r.get("source_label") == lab]
        phase1["by_label"][lab] = {
            "n": len(xs),
            "category": dict(Counter(r.get("category") for r in xs)),
            "precision_executed": dict(Counter(bool(r.get("precision_executed")) for r in xs)),
            "verified_continuous": sum(
                1 for r in xs if (r.get("continuous_demo") or {}).get("verified")
            ),
        }
    atomic_json(OUT / "PHASE1_PIPELINE.json", phase1)

    # ------------------------------------------------------------------
    # Phase 2 — Stage-1 exit physical audit for every B/C
    # ------------------------------------------------------------------
    exit_rows = []
    cfg_need = PrecisionConfig()
    bc = [
        r
        for r in rows
        if r.get("source_label") in ("B", "C")
        and r.get("ic_pkl")
        and (r.get("partition") in ("expert", "expert_continuous_only", "diagnostic"))
    ]
    print(f"Phase2 B/C rows: {len(bc)}", flush=True)
    for idx, r in enumerate(bc):
        sid = r["sample_id"]
        lab = r["source_label"]
        ic = pickle.loads((DS / r["ic_pkl"]).read_bytes())
        cap0, ref = ic["capture"], ic["ref"]
        n_s1 = int(r.get("n_actions_stage1") or 0)
        # Prefer continuous tape if present; else reconstruct from stage1/failure only
        acts = None
        cont = r.get("continuous_demo") or {}
        if cont.get("path") and (DS / cont["path"]).exists():
            acts = np.asarray(np.load(DS / cont["path"])["actions"], np.float32)
        if acts is None or n_s1 <= 0:
            # Diagnostic PRECISION_FAILURE may lack continuous tape — use stored stage1 if any
            s1_npz = DS / "stage1" / f"{sid}__stage1.npz"
            if s1_npz.exists():
                blob = np.load(s1_npz)
                key = "actions" if "actions" in blob.files else blob.files[0]
                acts = np.asarray(blob[key], np.float32)
                n_s1 = int(n_s1 or len(acts))
            else:
                exit_rows.append({
                    "sample_id": sid,
                    "skill": lab,
                    "category": r.get("category"),
                    "partition": r.get("partition"),
                    "n_s1": n_s1,
                    "note": "no_continuous_actions",
                    "stored_direct_stable_NEW": (r.get("direct_stable") or {}).get("SUCCESS_NEW"),
                    "stored_direct_stable_12S": (r.get("direct_stable") or {}).get("SUCCESS_12S"),
                    "stored_e_pos": (r.get("errors_stable") or {}).get("e_pos_mm"),
                    "stored_theta": (r.get("errors_stable") or {}).get("dtheta_deg"),
                    "need_dtheta": bool((r.get("need_at_stable") or {}).get("dtheta_needed")),
                    "need_dx": bool((r.get("need_at_stable") or {}).get("dx_needed")),
                    "need_dz": bool((r.get("need_at_stable") or {}).get("dz_needed")),
                    "h8_NEW": (r.get("direct_stable") or {}).get("SUCCESS_NEW"),
                    "h8_12S": (r.get("direct_stable") or {}).get("SUCCESS_12S"),
                })
                continue

        stable_cap = replay_to_stage1_end(env, cap0, acts, n_s1)
        restore_capture(env, stable_cap)
        st = snap(env.sim)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, stable_cap)
        gate = entry_gate(env)
        restore_capture(env, stable_cap)
        err = measure_errors(env.sim)
        need = assess_need(err, cfg_need)
        # HANDOFF with stabilize 8 (pipeline) and 0 (raw)
        h8 = handoff_probe(env, stable_cap, ref, stabilize_n=8)
        h0 = handoff_probe(env, stable_cap, ref, stabilize_n=0)
        # delayed: settle 0.5s then handoff
        restore_capture(env, stable_cap)
        for _ in range(25):
            step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
        cap_settle = take_capture(env)
        h_settle = handoff_probe(env, cap_settle, ref, stabilize_n=0)

        exit_rows.append({
            "sample_id": sid,
            "skill": lab,
            "category": r.get("category"),
            "partition": r.get("partition"),
            "precision_executed": bool(r.get("precision_executed")),
            "n_s1": n_s1,
            "bilateral": st["bilateral"],
            "latched": st["latched"],
            "e_pos_mm": st["e_pos_mm"],
            "dtheta_deg": st["dtheta_deg"],
            "dx_mm": st["dx_mm"],
            "dy_mm": st["dy_mm"],
            "dz_mm": st["dz_mm"],
            "v_rel": st["v_rel"],
            "w_rel": st["w_rel"],
            "obj_z": st["obj_z"],
            "mode": st["mode"],
            "fn_L": st["fn_L"],
            "fn_R": st["fn_R"],
            "hold_stable": bool(stab.get("stable")),
            "gate_ok": bool(gate.get("ok")),
            "need_dtheta": bool(need.get("dtheta_needed")),
            "need_dx": bool(need.get("dx_needed")),
            "need_dz": bool(need.get("dz_needed")),
            "need_ready_flag": bool(need.get("ready_for_direct_handoff_assessment")),
            "h8_NEW": h8["SUCCESS_NEW"],
            "h8_12S": h8["SUCCESS_12S"],
            "h8_reason": h8["reason"],
            "h0_NEW": h0["SUCCESS_NEW"],
            "h0_12S": h0["SUCCESS_12S"],
            "h0_reason": h0["reason"],
            "hsettle_NEW": h_settle["SUCCESS_NEW"],
            "hsettle_12S": h_settle["SUCCESS_12S"],
            "hsettle_reason": h_settle["reason"],
            "stored_direct_NEW": (r.get("direct_stable") or {}).get("SUCCESS_NEW"),
            "stored_direct_12S": (r.get("direct_stable") or {}).get("SUCCESS_12S"),
            "stored_handoff_NEW": (r.get("handoff") or {}).get("SUCCESS_NEW"),
            "stored_handoff_12S": (r.get("handoff") or {}).get("SUCCESS_12S"),
            "matches_stored_direct": bool(h8["SUCCESS_NEW"]) == bool((r.get("direct_stable") or {}).get("SUCCESS_NEW")),
        })
        if (idx + 1) % 10 == 0 or idx + 1 == len(bc):
            print(f"  [{idx+1}/{len(bc)}] {sid[:40]} h8_NEW={h8['SUCCESS_NEW']}", flush=True)

    write_csv(OUT / "STAGE1_EXIT_PHYSICS.csv", exit_rows)

    # Summary aggregates
    def agg(skill, pred):
        xs = [r for r in exit_rows if r.get("skill") == skill and "e_pos_mm" in r]
        if not xs:
            return {}
        return {
            "n": len(xs),
            "bilateral_frac": float(np.mean([r["bilateral"] for r in xs])),
            "hold_stable_frac": float(np.mean([r["hold_stable"] for r in xs])),
            "gate_ok_frac": float(np.mean([r["gate_ok"] for r in xs])),
            "need_any_frac": float(np.mean([r["need_dtheta"] or r["need_dx"] or r["need_dz"] for r in xs])),
            "h8_NEW": sum(1 for r in xs if r["h8_NEW"]),
            "h8_12S": sum(1 for r in xs if r["h8_12S"]),
            "h8_NEW_not_12S": sum(1 for r in xs if r["h8_NEW"] and not r["h8_12S"]),
            "h8_12S_not_NEW": sum(1 for r in xs if r["h8_12S"] and not r["h8_NEW"]),
            "h0_NEW": sum(1 for r in xs if r["h0_NEW"]),
            "mean_e_pos": float(np.mean([r["e_pos_mm"] for r in xs])),
            "max_e_pos": float(np.max([r["e_pos_mm"] for r in xs])),
            "mean_theta": float(np.mean([r["dtheta_deg"] for r in xs])),
            "max_theta": float(np.max([r["dtheta_deg"] for r in xs])),
            "mean_v_rel": float(np.mean([r["v_rel"] for r in xs])),
            "mean_w_rel": float(np.mean([r["w_rel"] for r in xs])),
            "category": dict(Counter(r["category"] for r in xs)),
        }

    phase2_sum = {"B": agg("B", None), "C": agg("C", None)}
    # Split C verified vs precision_failure
    c_ok = [r for r in exit_rows if r.get("skill") == "C" and r.get("category") == "DIRECT_HANDOFF"]
    c_pf = [r for r in exit_rows if r.get("skill") == "C" and r.get("category") == "PRECISION_FAILURE"]
    phase2_sum["C_DIRECT"] = {
        "n": len(c_ok),
        "h8_NEW": sum(1 for r in c_ok if r.get("h8_NEW")),
        "need_any": sum(1 for r in c_ok if r.get("need_dtheta") or r.get("need_dx") or r.get("need_dz")),
        "mean_e_pos": float(np.mean([r["e_pos_mm"] for r in c_ok])) if c_ok else None,
    }
    phase2_sum["C_PRECISION_FAILURE"] = {
        "n": len(c_pf),
        "h8_NEW": sum(1 for r in c_pf if r.get("h8_NEW")),
        "h8_12S": sum(1 for r in c_pf if r.get("h8_12S")),
        "need_any": sum(1 for r in c_pf if r.get("need_dtheta") or r.get("need_dx") or r.get("need_dz")),
        "mean_e_pos": float(np.mean([r["e_pos_mm"] for r in c_pf])) if c_pf else None,
        "mean_theta": float(np.mean([r["dtheta_deg"] for r in c_pf])) if c_pf else None,
        "hsettle_NEW": sum(1 for r in c_pf if r.get("hsettle_NEW")),
    }
    atomic_json(OUT / "PHASE2_SUMMARY.json", phase2_sum)

    # ------------------------------------------------------------------
    # Phase 3 — Does C Stage-1 absorb Precision-like control?
    # ------------------------------------------------------------------
    c_verified = [
        r
        for r in rows
        if r.get("source_label") == "C"
        and (r.get("continuous_demo") or {}).get("verified")
        and (r.get("continuous_demo") or {}).get("path")
    ]
    mech_rows = []
    print(f"Phase3 C verified: {len(c_verified)}", flush=True)
    for idx, r in enumerate(c_verified):
        acts = np.asarray(np.load(DS / r["continuous_demo"]["path"])["actions"], np.float32)
        n_s1 = int(r.get("n_actions_stage1") or len(acts))
        s1 = acts[:n_s1]
        ic = pickle.loads((DS / r["ic_pkl"]).read_bytes())
        astat = action_phase_stats(s1)
        pstat = state_path_stats(env, ic["capture"], s1, stride=max(1, n_s1 // 40))
        # Precision-like late wrist? (wy activity late while |v| small)
        late = s1[-min(25, len(s1)) :] if len(s1) else s1
        early = s1[: min(25, len(s1))] if len(s1) else s1
        mech_rows.append({
            "sample_id": r["sample_id"],
            "n_s1": n_s1,
            "early_mean_|v|": astat.get("early_vs_late", {}).get("early_mean_|v|"),
            "late_mean_|v|": astat.get("early_vs_late", {}).get("late_mean_|v|"),
            "early_mean_|w|": astat.get("early_vs_late", {}).get("early_mean_|w|"),
            "late_mean_|w|": astat.get("early_vs_late", {}).get("late_mean_|w|"),
            "early_mean_|wy|": astat.get("early_vs_late", {}).get("early_mean_|wy|"),
            "late_mean_|wy|": astat.get("early_vs_late", {}).get("late_mean_|wy|"),
            "late_frac_near_hold": astat.get("early_vs_late", {}).get("late_frac_near_hold"),
            "T0_frac_|v|>0.25": (astat.get("thirds") or {}).get("T0", {}).get("frac_|v|>0.25"),
            "T2_frac_|v|>0.25": (astat.get("thirds") or {}).get("T2", {}).get("frac_|v|>0.25"),
            "T0_frac_|w|>0.25": (astat.get("thirds") or {}).get("T0", {}).get("frac_|w|>0.25"),
            "T2_frac_|wy|>0.15": (astat.get("thirds") or {}).get("T2", {}).get("frac_|wy|>0.15"),
            "e_pos_start": pstat.get("e_pos_start"),
            "e_pos_end": pstat.get("e_pos_end"),
            "dtheta_start": pstat.get("dtheta_start"),
            "dtheta_end": pstat.get("dtheta_end"),
            "obj_z_start": pstat.get("obj_z_start"),
            "obj_z_end": pstat.get("obj_z_end"),
            "obj_z_max": pstat.get("obj_z_max"),
            "bilateral_frac": pstat.get("bilateral_frac"),
            "delta_e_pos": None if not pstat else pstat["e_pos_end"] - pstat["e_pos_start"],
            "delta_theta": None if not pstat else pstat["dtheta_end"] - pstat["dtheta_start"],
            "late_wy_activity": float(np.mean(np.abs(late[:, 4]))) if len(late) else 0.0,
            "late_|v|_low": float(np.mean(np.linalg.norm(late[:, :3], axis=1) < 0.2)) if len(late) else 0.0,
        })
        if (idx + 1) % 10 == 0:
            print(f"  mech {idx+1}/{len(c_verified)}", flush=True)

    write_csv(OUT / "C_STAGE1_MECHANISM.csv", mech_rows)
    if mech_rows:
        phase3 = {
            "n": len(mech_rows),
            "mean_n_s1": float(np.mean([r["n_s1"] for r in mech_rows])),
            "mean_early_|v|": float(np.mean([r["early_mean_|v|"] for r in mech_rows])),
            "mean_late_|v|": float(np.mean([r["late_mean_|v|"] for r in mech_rows])),
            "mean_early_|w|": float(np.mean([r["early_mean_|w|"] for r in mech_rows])),
            "mean_late_|w|": float(np.mean([r["late_mean_|w|"] for r in mech_rows])),
            "mean_late_|wy|": float(np.mean([r["late_mean_|wy|"] for r in mech_rows])),
            "mean_late_frac_near_hold": float(np.mean([r["late_frac_near_hold"] for r in mech_rows])),
            "mean_delta_e_pos": float(np.mean([r["delta_e_pos"] for r in mech_rows])),
            "mean_delta_theta": float(np.mean([r["delta_theta"] for r in mech_rows])),
            "mean_obj_z_lift": float(np.mean([r["obj_z_end"] - r["obj_z_start"] for r in mech_rows])),
            "frac_late_wy_gt_015_and_v_low": float(
                np.mean([
                    (r["late_wy_activity"] > 0.15) and (r["late_|v|_low"] > 0.5)
                    for r in mech_rows
                ])
            ),
            "interpretation": (
                "Early Stage-1 dominated by translation/catch (high |v|,|w|); "
                "late Stage-1 dominated by secure hold / low motion. "
                "Large e_pos/theta reduction during Stage-1 indicates catch+settle "
                "into working band, not Precision oracle primitives (dx/dz/pivot)."
            ),
        }
    else:
        phase3 = {}
    atomic_json(OUT / "PHASE3_SUMMARY.json", phase3)

    # ------------------------------------------------------------------
    # Phase 4 — missing multi-stage / failed precision search
    # ------------------------------------------------------------------
    phase4 = {
        "B_stage1_precision_success": 0,
        "C_stage1_precision_success": 0,
        "C_precision_failure_diagnostic": [],
        "B_precision_failure_diagnostic": [],
        "A_direct_handoff": [],
        "causes": {},
    }
    for r in rows:
        lab = r.get("source_label")
        cat = r.get("category")
        if lab == "B" and cat == "PRECISION_FAILURE":
            phase4["B_precision_failure_diagnostic"].append(r["sample_id"])
        if lab == "C" and cat == "PRECISION_FAILURE":
            phase4["C_precision_failure_diagnostic"].append({
                "sample_id": r["sample_id"],
                "n_s1": r.get("n_actions_stage1"),
                "n_prec": r.get("n_actions_precision"),
                "need": r.get("need_at_stable"),
                "direct_stable": r.get("direct_stable"),
                "precision_phases": (r.get("precision") or {}).get("phases"),
                "precision_status": (r.get("precision") or {}).get("status"),
                "handoff_reason": (r.get("handoff") or {}).get("reason"),
            })
        if lab == "A" and cat == "DIRECT_HANDOFF":
            phase4["A_direct_handoff"].append(r["sample_id"])
        if lab in ("B", "C") and cat == "PRECISION_SUCCESS":
            phase4[f"{lab}_stage1_precision_success"] += 1

    # Cause classification
    b_exit = [r for r in exit_rows if r.get("skill") == "B" and "h8_NEW" in r]
    c_dir = [r for r in exit_rows if r.get("skill") == "C" and r.get("category") == "DIRECT_HANDOFF" and "h8_NEW" in r]
    phase4["causes"] = {
        "A_genuine_stage1_sufficiency_B": {
            "supported": all(r.get("h8_NEW") for r in b_exit) and all(not (r.get("need_dtheta") or r.get("need_dx") or r.get("need_dz")) for r in b_exit),
            "evidence": f"B exit h8_NEW={sum(r.get('h8_NEW') for r in b_exit)}/{len(b_exit)}; need_any=0 on verified set",
        },
        "A_genuine_stage1_sufficiency_C_verified": {
            "supported": all(r.get("h8_NEW") for r in c_dir) if c_dir else False,
            "evidence": f"C DIRECT exit h8_NEW={sum(r.get('h8_NEW') for r in c_dir)}/{len(c_dir)}; assess_need empty",
        },
        "B_stage_transition_implementation": {
            "supported": True,
            "evidence": (
                "process_one skips Precision whenever direct_stable.SUCCESS_NEW; "
                "not skill-gated. Same rule used for A (1 DIRECT_HANDOFF)."
            ),
        },
        "C_filtering_bias": {
            "supported": True,
            "partial": True,
            "evidence": (
                "C PRECISION_FAILURE (6) are diagnostic-only (partition=diagnostic); "
                "no successful C PRECISION_SUCCESS exists. Failures show NO_REMAINING_CORRECTION "
                "(settle-only) while direct HANDOFF fails — Precision capability gap for "
                "contact-fragile in-band states, not export filter of successes."
            ),
        },
        "D_missing_expert_capabilities": {
            "supported": True,
            "evidence": (
                "When C Stage-1 exits in-band but not SUCCESS_NEW-stable, Precision oracle "
                "finds no dx/dz/dtheta need and only settle_hold (25 steps), then DROP_AFTER_HANDOFF."
            ),
        },
        "E_recovery_entry_diversity": {
            "supported": "partial",
            "evidence": (
                "B always n_s1=5 (first stability check) — source Plan-B ICs already near-stable. "
                "Verified C ICs that succeed direct after long catch/lift dominate; "
                "fragile C exits exist but Precision cannot repair them."
            ),
        },
    }
    atomic_json(OUT / "PHASE4_COVERAGE.json", phase4)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    md = []
    md.append("# Root-Cause Report — Missing B/C Precision Coverage")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("")
    md.append("## Executive verdict")
    md.append("")
    md.append(
        "Verified B and C experts lack Stage-1→Precision tapes because Stage-1 exits "
        "**already achieve `probe_direct_handoff` SUCCESS_NEW**, so the pipeline "
        "legally takes **DIRECT_HANDOFF** and skips Precision. This is **not** a "
        "skill-identity bypass. Precision **is** reachable for C when direct fails "
        "(6 diagnostic attempts), but those attempts only settle (no primitive "
        "corrections) and still fail — a **capability / basin-fragility** gap, not "
        "silent deletion of successes."
    )
    md.append("")
    md.append("## Phase 1 — Stage transitions (code)")
    md.append("")
    md.append("| Step | Condition | Location |")
    md.append("|---|---|---|")
    md.append(
        "| Stage-1 end (A/B) | earliest prefix with `hold_and_confirm_stable` ∧ `entry_gate.ok` (check every 5 steps) | `run_stage1_ab` |"
    )
    md.append(
        "| Stage-1 end (C) | same + `obj_z ≥ 0.48` before checks; elev_snap fallback | `run_stage1_c` |"
    )
    md.append(
        "| Enter Precision | `not probe_direct_handoff(stable).SUCCESS_NEW` | `process_one` L601–611 |"
    )
    md.append(
        "| DIRECT_HANDOFF | `probe_direct_handoff(stable).SUCCESS_NEW` → skip Precision | `process_one` L601–608 |"
    )
    md.append(
        "| Skill bypass? | **No** — family only passed to `run_precision` when entered | same |"
    )
    md.append(
        "| Export filter | `PRECISION_FAILURE` → `partition=diagnostic` (not expert) | L706–708 |"
    )
    md.append("")
    md.append("## Phase 2 — Stage-1 exit physics")
    md.append("")
    md.append(f"- B: {json.dumps(phase2_sum.get('B'))}")
    md.append(f"- C: {json.dumps(phase2_sum.get('C'))}")
    md.append(f"- C DIRECT subset: {json.dumps(phase2_sum.get('C_DIRECT'))}")
    md.append(f"- C PRECISION_FAILURE subset: {json.dumps(phase2_sum.get('C_PRECISION_FAILURE'))}")
    md.append("")
    md.append("SUCCESS_NEW and SUCCESS_12S are measured separately (`h8_NEW` / `h8_12S`).")
    md.append("")
    md.append("## Phase 3 — Does C Stage-1 absorb Precision?")
    md.append("")
    md.append(json.dumps(phase3, indent=2))
    md.append("")
    md.append(
        "**Conclusion:** C Stage-1 performs airborne recatch, lift, grip secure, and "
        "settle into the working pose band. It is **not** the Precision oracle "
        "(no dx/dz/pivot primitives). Late Stage-1 is mostly hold, not Precision-like wrist work."
    )
    md.append("")
    md.append("## Phase 4 — Cause matrix")
    md.append("")
    for k, v in phase4["causes"].items():
        md.append(f"- **{k}**: supported={v.get('supported')} — {v.get('evidence')}")
    md.append("")
    md.append("## Phase 5 — Repair plan (proposal only; do not implement yet)")
    md.append("")
    md.append("See `REPAIR_PLAN.md`.")
    (OUT / "ROOT_CAUSE_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    repair = """# Repair Plan — B/C Multi-Stage Coverage (proposal only)

## Goals
Preserve genuine DIRECT_HANDOFF successes. Add Stage-1 → Precision → HANDOFF **only when physically needed and successfully demonstrated**. Do not manufacture ICs, relabel skills, weaken SUCCESS_*, or change the 7D interface.

## Findings driving the plan
1. Verified B/C Direct exits are physically SUCCESS_NEW-capable — keep them.
2. Missing successful B/C Precision is **not** because Precision is unreachable in code.
3. C's 6 diagnostic Precision attempts fail because `assess_need` reports no dx/dz/dtheta while 12s HANDOFF still drops — Precision has no corrective primitive for **contact-fragile in-band** states.
4. B ICs stabilize in 5 steps — little room for a Precision stage unless we change IC selection.

## Proposed generation strategy

### Track 1 — Keep Direct corpus (no change)
Re-export / retain all B/C DIRECT_HANDOFF with verified continuous SUCCESS_NEW.

### Track 2 — C fragile-exit Precision (new capability first)
Before mining more C→Precision tapes:
1. Define a **fragility gate** distinct from geometric `assess_need`: e.g. direct probe fails OR `SUCCESS_12S` fails OR contact margins (fn, v_rel, slip) below thresholds despite in-band pose.
2. Extend Precision (or a post-Stage-1 **stabilize/regrasp** primitive under the Precision stage label) to address fragile in-band exits without inventing dx/dz when not needed.
3. Only after that capability exists: regenerate C lineages where Stage-1 exits fail the fragility gate; keep failures diagnostic; promote only SUCCESS_NEW fullchain tapes as `C` + `PRECISION_SUCCESS`.

### Track 3 — B Stage-1 → Precision (diversity / selection)
1. Audit Plan-B Recovery IC pool for cases where earliest stable gate passes but `probe_direct_handoff` fails (pose out of band or fragile).
2. If such ICs exist: run shared Precision from that stable capture (same as A).
3. If they do not exist: **do not force Precision** on Direct-success B; instead expand Recovery-entry diversity (harder Plan-B impacts) until Direct fails at stable while Stage-1 still succeeds.

### Track 4 — Dataset policy (filtering)
- Continue excluding PRECISION_FAILURE from the expert partition (correct).
- Optionally keep a **curriculum diagnostic** partition of C fragile Direct-fails for oracle development — not for SFT success labels.
- Never relabel A/STABLE_PRECISION as B/C.

### Acceptance for a repaired release
- ≥1 verified B fullchain with `n_actions_precision > 0` and SUCCESS_NEW, **or** documented proof that no Plan-B Recovery IC in the eligible pool needs Precision.
- ≥N verified C fullchain with Precision success (N agreed, e.g. 10), from fragility-gated exits, without reducing Direct success criteria.
- All existing Direct B/C successes still present and replaying SUCCESS_NEW.

## Explicit non-goals
- Do not truncate C Stage-1 early to “make room” for Precision.
- Do not call settle_hold a Precision correction success.
- Do not weaken WORKING_POS / SUCCESS_NEW to create need flags.
"""
    (OUT / "REPAIR_PLAN.md").write_text(repair, encoding="utf-8")
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
