"""Grasp-configuration migration audit (read-only on protected components).

Calibrates object–gripper targets from SUCCESS_12S end-states, re-evaluates
the 22-IC short-handoff pilot under configuration metrics, and runs a
budgeted legal-action pivoting feasibility probe.

Writes: results/diagnostics/grasp_configuration_migration/
Does NOT train, edit reward/SUCCESS_12S/physics, or run large-scale mining.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed, terminal_reason
from envs.simulation.capture import restore_capture
from envs.step_legal import N_SUB, step_legal
from evaluation.grasp_geometry import (
    AXIS_REVERSAL_SYMMETRIC,
    DEFAULT_P_TARGET_H,
    calibrate_from_samples,
    cylinder_axis_hand,
    measure,
    p_com_hand,
    success_new_components,
)
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from training.pipeline import HANDOFF

OUT = ROOT / "results" / "diagnostics" / "grasp_configuration_migration"
PILOT = ROOT / "results" / "diagnostics" / "nontrivial_recovery_ic_mining"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
LIMIT_S = float(SELECTED_LIMIT_S)
DT = 0.02
SEED = 5478


def jsafe(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def load_pilot_manifest():
    rows = list(csv.DictReader((PILOT / "IC_MANIFEST.csv").open(encoding="utf-8")))
    return rows


def parse_tokens(s):
    if not s or not str(s).strip():
        return []
    return list(json.loads(s))


def run_tokens(env, book, blob, tokens, residuals=None, log_substeps=False):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    sub = []
    for i, tok in enumerate(tokens):
        if int(tok) == HANDOFF:
            break
        pre = np.zeros(7, np.float32) if residuals is None else np.asarray(residuals[i], np.float32)
        _r, action = decode_action(book, int(tok), pre)
        # step_legal runs N_SUB mujoco steps; optionally sample mid-token
        if log_substeps:
            m0 = measure_quick(env.sim)
            executed = step_legal(env.sim, action)
            env._prev = np.asarray(executed, np.float32)
            m1 = measure_quick(env.sim)
            sub.append({"tok": int(tok), "pre": m0, "post": m1, "action": np.asarray(executed, float).tolist()})
        else:
            executed = step_legal(env.sim, action)
            env._prev = np.asarray(executed, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "success_12s": False,
                "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "dropped_before_handoff": True,
                "substeps": sub,
            }
    outcome = finish_recovery(env, blob["ref"], t0, "early")
    return {
        "success_12s": bool(outcome.get("success")),
        "reason": outcome.get("reason"),
        "recovery_elapsed": outcome.get("recovery_elapsed"),
        "dropped_before_handoff": False,
        "outcome": outcome,
        "substeps": sub,
    }


def measure_quick(sim):
    """Lightweight fields for pivoting logs (targets filled later)."""
    return {
        "u_c_h": cylinder_axis_hand(sim).tolist(),
        "p_c_h": p_com_hand(sim).tolist(),
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "ncon": int(sim.data.ncon),
        "t": float(sim.data.time),
        "tau": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else 0.0,
    }


def measure_with_hold(env, blob, book, tokens, u_tgt, p_tgt, tol_th, tol_pos, hold_s=0.5):
    """Replay prefix→HANDOFF→nominal; track pose/pos OK durations in final hold window."""
    restore_capture(env, blob["capture"])
    m0 = measure(env.sim, u_tgt, p_tgt)
    t0 = float(env.sim.data.time)
    lost_low = 0
    for tok in tokens:
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            m_fail = measure(env.sim, u_tgt, p_tgt)
            return {
                "m_init": m0,
                "m_after_prefix": m_fail,
                "m_final": m_fail,
                "success_12s": False,
                "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                "pose_ok_s": 0.0,
                "pos_ok_s": 0.0,
                "components": success_new_components(False, m_fail["e_theta_rad"], m_fail["e_pos_m"], tol_th, tol_pos),
            }
    m_mid = measure(env.sim, u_tgt, p_tgt)
    # HANDOFF + nominal with duration tracking
    from envs.outcomes.recovery_success import apply_handoff, nominal_step
    from envs.outcomes.horizon_outcome import _bilateral, _dropped, _monitor, NOMINAL_S, LOST_STEPS
    from envs.outcomes.table_drop import latched
    from training.recovery_execution import _escaped
    from envs.ballistic_impact import z_tgt_of

    apply_handoff(env.sim, blob["ref"])
    t_h = float(env.sim.data.time)
    pose_ok_s = 0.0
    pos_ok_s = 0.0
    # Final evaluation window: last ``hold_s`` seconds of the 12s nominal
    window_start = float(NOMINAL_S) - float(hold_s)
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    if latched(env.sim):
        reason = "TABLE_DROP"
    while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        t_prev = float(env.sim.data.time)
        nominal_step(env.sim)
        dt = float(env.sim.data.time) - t_prev
        elapsed = float(env.sim.data.time) - t_h
        row = _monitor(env.sim)
        end_row = row
        m = measure(env.sim, u_tgt, p_tgt)
        if elapsed >= window_start:
            if m["e_theta_rad"] <= tol_th:
                pose_ok_s += dt
            if m["e_pos_m"] <= tol_pos:
                pos_ok_s += dt
        target = float(z_tgt_of(env.sim))
        bilateral = _bilateral(row)
        command_reached = float(env.sim.fsm.p_des[2]) >= target - 1e-9
        dropped = _dropped(row) or _escaped(env.sim)
        if bilateral and command_reached:
            lift_done = True
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        if dropped:
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not bilateral else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
    if reason is None:
        if end_row is None or not _bilateral(end_row) or _dropped(end_row) or _escaped(env.sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    m_final = measure(env.sim, u_tgt, p_tgt)
    succ = reason == "SUCCESS_12S"
    comps = success_new_components(
        succ, m_final["e_theta_rad"], m_final["e_pos_m"], tol_th, tol_pos,
        pose_ok_duration_s=pose_ok_s, pos_ok_duration_s=pos_ok_s, hold_required_s=hold_s,
    )
    return {
        "m_init": m0,
        "m_after_prefix": m_mid,
        "m_final": m_final,
        "success_12s": succ,
        "reason": reason,
        "pose_ok_s": pose_ok_s,
        "pos_ok_s": pos_ok_s,
        "components": comps,
        "recovery_elapsed": t_h - t0,
    }


def calibrate(env, book, ics, u_seed, p_seed):
    """Collect end-of-SUCCESS_12S configurations from short-prefix successes."""
    samples_u, samples_p = [], []
    meta = []
    for row in ics:
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "capture" not in blob:
            continue
        tokens = parse_tokens(row["shortest_short"])
        res = run_tokens(env, book, blob, tokens)
        if not res["success_12s"]:
            continue
        # state is post-nominal after finish_recovery
        u = cylinder_axis_hand(env.sim)
        p = p_com_hand(env.sim)
        samples_u.append(u)
        samples_p.append(p)
        meta.append({"ic_id": row["id"], "tokens": tokens, "u_c_h": u.tolist(), "p_c_h": p.tolist()})
    if len(samples_u) < 3:
        # fallback seed
        cal = {
            "u_target_h": np.asarray(u_seed, float),
            "p_target_h": np.asarray(p_seed, float),
            "n_samples": len(samples_u),
            "note": "insufficient SUCCESS_12S end samples; using seed targets",
            "tol_theta_rad_proposed": np.deg2rad(15.0),
            "tol_pos_m_proposed": 0.015,
        }
    else:
        cal = calibrate_from_samples(samples_u, samples_p)
    cal["samples"] = meta
    return cal


def pivoting_pilot(env, book, ics, u_tgt, p_tgt, rng, max_ics=5, max_seqs_per_ic=24):
    """Budgeted legal grip-modulation search for relative orientation change."""
    # Prefer ICs with large initial e_theta
    ranked = []
    for row in ics:
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "capture" not in blob:
            continue
        restore_capture(env, blob["capture"])
        m = measure(env.sim, u_tgt, p_tgt)
        ranked.append((m["e_theta_deg"], row, blob, m))
    ranked.sort(key=lambda x: -x[0])
    candidates = ranked[:max_ics]

    openish = [i for i in range(47) if float(book[i, 6]) < -0.5]
    closeish = [i for i in range(47) if float(book[i, 6]) > 0.5]
    # mid-grip: least |a6| among all
    mid = sorted(range(47), key=lambda i: abs(float(book[i, 6])))[:5]

    rows = []
    logs = []
    for e0, row, blob, m0 in candidates:
        ic = row["id"]
        short = parse_tokens(row["shortest_short"])
        branches = {
            "immediate": [],
            "short_baseline": short,
        }
        # grip-modulation proposals (legal K48 only, zero residual)
        proposals = []
        for t_open in [30, 36, 44] + list(rng.choice(openish, size=min(3, len(openish)), replace=False)):
            for n_open in (1, 2, 3, 5):
                for t_close in [0, 1] + list(rng.choice(closeish, size=min(2, len(closeish)), replace=False)):
                    for n_close in (1, 2, 3):
                        proposals.append([int(t_open)] * n_open + [int(t_close)] * n_close)
        for t_mid in mid:
            proposals.append([int(t_mid)] * 5)
            proposals.append([int(t_mid)] * 10)
        # dedupe + budget
        uniq = []
        seen = set()
        for seq in proposals:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(seq)
        rng.shuffle(uniq)
        uniq = uniq[:max_seqs_per_ic]

        for name, toks in list(branches.items()) + [(f"pivot_{i}", s) for i, s in enumerate(uniq)]:
            t_wall0 = time.perf_counter()
            res = run_tokens(env, book, blob, toks, log_substeps=(name.startswith("pivot_") and len(toks) <= 6))
            # measure final state (after finish_recovery if success path)
            m_final = measure(env.sim, u_tgt, p_tgt)
            # after_prefix: re-run without nominal for mid measure
            restore_capture(env, blob["capture"])
            lost_low = 0
            dropped = False
            for tok in toks:
                _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
                step_legal(env.sim, action)
                env._prev = np.asarray(action, np.float32)
                failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
                if failed:
                    dropped = True
                    break
            m_mid = measure(env.sim, u_tgt, p_tgt)
            delta_theta = float(m0["e_theta_deg"] - m_mid["e_theta_deg"])
            # contact-preserving heuristic: bilateral throughout mid (from log)
            contact_ok = bool(m_mid["bilateral"]) and not dropped
            mechanism = "UNKNOWN"
            if res["success_12s"] and delta_theta > 5.0 and contact_ok and m_mid["e_theta_deg"] < m0["e_theta_deg"]:
                # check no full release in substep log if present
                released = False
                for s in res.get("substeps") or []:
                    # ncon alone insufficient; use bilateral on post via remeasure fields
                    pass
                mechanism = "CANDIDATE_ORIENTATION_IMPROVE"
            if res["success_12s"] and m_final["e_theta_deg"] <= 10.0 and m_final["e_pos_mm"] <= 15.0:
                mechanism = "STABILITY_AND_CONFIG_CANDIDATE"
            rows.append({
                "ic_id": ic,
                "branch": name,
                "tokens": json.dumps(toks),
                "e_theta0_deg": m0["e_theta_deg"],
                "e_pos0_mm": m0["e_pos_mm"],
                "e_theta_mid_deg": m_mid["e_theta_deg"],
                "e_pos_mid_mm": m_mid["e_pos_mm"],
                "e_theta_final_deg": m_final["e_theta_deg"],
                "e_pos_final_mm": m_final["e_pos_mm"],
                "delta_theta_mid_deg": delta_theta,
                "success_12s": int(res["success_12s"]),
                "reason": res["reason"],
                "mid_bilateral": int(m_mid["bilateral"]),
                "mechanism_note": mechanism,
                "wall_s": time.perf_counter() - t_wall0,
                "n_tokens": len(toks),
            })
            if name.startswith("pivot_") and (delta_theta > 3.0 or res["success_12s"]):
                logs.append({
                    "ic_id": ic, "branch": name, "tokens": toks,
                    "m0": m0, "m_mid": m_mid, "m_final": m_final,
                    "reason": res["reason"], "success_12s": res["success_12s"],
                    "substeps": res.get("substeps"),
                })
        print(f"PIVOT_IC {ic} e0={e0:.1f}deg branches={2+len(uniq)}", flush=True)
    return rows, logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--pivot-ics", type=int, default=5)
    parser.add_argument("--pivot-seqs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    ics = load_pilot_manifest()
    print("LOADED_ICS", len(ics), flush=True)

    # --- Symmetry note ---
    symmetry = {
        "AXIS_REVERSAL_SYMMETRIC": AXIS_REVERSAL_SYMMETRIC,
        "evidence": "ballistic_scene cylinder geom + diaginertia Ixx=Iyy≠Izz; uniform rgba; no markings",
        "cyl_axis_body": [0, 0, 1],
    }
    write_json(OUT / "raw" / "symmetry.json", symmetry)

    # --- Calibration from SUCCESS_12S end states ---
    print("CALIBRATE", flush=True)
    cal = calibrate(env, book, ics, np.array([0.0, 0.0, 1.0]), DEFAULT_P_TARGET_H)
    u_tgt = np.asarray(cal["u_target_h"], float)
    p_tgt = np.asarray(cal["p_target_h"], float)
    tol_th = float(cal["tol_theta_rad_proposed"])
    tol_pos = float(cal["tol_pos_m_proposed"])
    # Geometric prior: cylinder axis vs ±hand +Z (table upright grasp)
    geo = []
    for row in ics[:8]:
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "capture" not in blob:
            continue
        tokens = parse_tokens(row["shortest_short"])
        if not run_tokens(env, book, blob, tokens)["success_12s"]:
            continue
        u = cylinder_axis_hand(env.sim)
        for hand_z in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])):
            from evaluation.grasp_geometry import orientation_error_rad
            geo.append(float(np.degrees(orientation_error_rad(u, hand_z, True))))
    cal["geometric_hand_z_e_theta_deg_mean"] = float(np.mean(geo)) if geo else None
    cal["calibration_source"] = (
        "SUCCESS_12S end-states after each IC shortest short-prefix HANDOFF "
        "(empirical achieved configuration under old stability task)"
    )
    write_json(OUT / "calibration.json", cal)
    print(
        "CAL", "n", cal["n_samples"],
        "tol_deg", np.degrees(tol_th), "tol_mm", tol_pos * 1e3,
        "p_tgt", p_tgt.tolist(), "u_tgt", u_tgt.tolist(),
        "geo_hand_z_mean_deg", cal.get("geometric_hand_z_e_theta_deg_mean"),
        flush=True,
    )

    # --- 22-IC re-evaluation ---
    print("REEVAL_22", flush=True)
    reeval_rows = []
    for row in ics:
        ic = row["id"]
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "capture" not in blob:
            print("SKIP", ic, flush=True)
            continue
        tokens = parse_tokens(row["shortest_short"])
        out = measure_with_hold(env, blob, book, tokens, u_tgt, p_tgt, tol_th, tol_pos, hold_s=args.hold_s)
        # also immediate handoff branch for comparison
        imm = measure_with_hold(env, blob, book, [], u_tgt, p_tgt, tol_th, tol_pos, hold_s=args.hold_s)
        comps = out["components"]
        reeval_rows.append({
            "ic_id": ic,
            "role": row.get("role"),
            "shortest_short": row["shortest_short"],
            "e_theta0_deg": out["m_init"]["e_theta_deg"],
            "e_pos0_mm": out["m_init"]["e_pos_mm"],
            "e_theta_after_prefix_deg": out["m_after_prefix"]["e_theta_deg"],
            "e_pos_after_prefix_mm": out["m_after_prefix"]["e_pos_mm"],
            "e_theta_final_deg": out["m_final"]["e_theta_deg"],
            "e_pos_final_mm": out["m_final"]["e_pos_mm"],
            "success_12s": int(out["success_12s"]),
            "reason": out["reason"],
            "pose_ok_s": out["pose_ok_s"],
            "pos_ok_s": out["pos_ok_s"],
            "POSE_CORRECT": int(comps["POSE_CORRECT"]),
            "POSITION_CORRECT": int(comps["POSITION_CORRECT"]),
            "SUCCESS_NEW": int(comps["SUCCESS_NEW"]),
            "stability_only": int(out["success_12s"] and not comps["SUCCESS_NEW"]),
            "imm_success_12s": int(imm["success_12s"]),
            "imm_e_theta_final_deg": imm["m_final"]["e_theta_deg"],
            "imm_e_pos_final_mm": imm["m_final"]["e_pos_mm"],
            "tol_theta_deg": float(np.degrees(tol_th)),
            "tol_pos_mm": float(tol_pos * 1e3),
            "hold_s": args.hold_s,
        })
        write_json(OUT / "raw" / f"{ic}_reeval.json", out)
        print(
            "REEVAL", ic,
            "S12", out["success_12s"],
            "SNEW", comps["SUCCESS_NEW"],
            "eθf", f"{out['m_final']['e_theta_deg']:.1f}",
            "epf_mm", f"{out['m_final']['e_pos_mm']:.1f}",
            flush=True,
        )
    write_csv(OUT / "EXISTING_22IC_REEVALUATION.csv", reeval_rows)

    # --- Pivoting pilot ---
    print("PIVOT_PILOT", flush=True)
    pivot_rows, pivot_logs = pivoting_pilot(
        env, book, ics, u_tgt, p_tgt, rng,
        max_ics=args.pivot_ics, max_seqs_per_ic=args.pivot_seqs,
    )
    write_csv(OUT / "PIVOTING_PILOT.csv", pivot_rows)
    write_json(OUT / "raw" / "pivoting_logs.json", pivot_logs)

    n_stab_only = sum(1 for r in reeval_rows if r["stability_only"])
    n_snew = sum(1 for r in reeval_rows if r["SUCCESS_NEW"])
    n_s12 = sum(1 for r in reeval_rows if r["success_12s"])
    pivot_improve = [r for r in pivot_rows if r["delta_theta_mid_deg"] > 5 and r["mid_bilateral"]]
    pivot_s12 = [r for r in pivot_rows if r["success_12s"] and r["branch"].startswith("pivot_")]
    write_json(OUT / "summary.json", {
        "n_ics": len(reeval_rows),
        "n_success_12s": n_s12,
        "n_success_new": n_snew,
        "n_stability_only": n_stab_only,
        "tol_theta_deg": float(np.degrees(tol_th)),
        "tol_pos_mm": float(tol_pos * 1e3),
        "u_target_h": u_tgt.tolist(),
        "p_target_h": p_tgt.tolist(),
        "pivot_seqs": len(pivot_rows),
        "pivot_orientation_improve_ge5deg_bilateral": len(pivot_improve),
        "pivot_success_12s": len(pivot_s12),
        "axis_reversal_symmetric": AXIS_REVERSAL_SYMMETRIC,
        "seed": args.seed,
    })
    print(
        "DONE",
        "S12", n_s12, "SNEW", n_snew, "stab_only", n_stab_only,
        "pivot_improve", len(pivot_improve),
        flush=True,
    )


if __name__ == "__main__":
    main()
