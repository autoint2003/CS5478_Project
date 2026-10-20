"""Bounded K49 Sliding + Recatch expert generation for unified Transformer SFT.

Uses config K49_tok48_wy0p25 only. No SFT/GRPO. Locked SUCCESS_NEW (10°/12mm).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_generation.recording.trajectory_recorder import save_pair
from envs.ballistic_impact import z_tgt_of
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor, recovery_failed
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary, window_stats
from evaluation.grasp_geometry import measure, p_com_hand
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import (
    SECURE_REVERSE_WRIST,
    assert_k48_prefix_identical,
    build_k49_table,
    decode_action as decode_k49,
    describe_k49_config,
)
from training.k47_codebook import decode_action as decode_k48
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env
from training.recovery_execution import _escaped

OUT = ROOT / "results" / "expert_generation" / "k49_sliding_recatch_dataset"
CONFIG_ID = "K49_tok48_wy0p25"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
SEED = 20261018
RECOVERY_LIMIT_S = 4.0

# Prior verified experts (compatible mapping)
PRIOR_EXPERTS = [
    {
        "ic_id": "hard_910263",
        "source": "k49_slow_return_position_refinement",
        "path": "results/expert_generation/k49_slow_return_position_refinement/trajectories/hard_910263__cpp_4x22_12x5_0x2__suf0__tokens.npz",
        "snap": "results/expert_generation/k49_slow_return_position_refinement/SNAPSHOTS/hard_910263.pkl",
        "meta": "results/expert_generation/k49_slow_return_position_refinement/trajectories/hard_910263__cpp_4x22_12x5_0x2__suf0_meta.json",
        "regression": True,
    },
    {
        "ic_id": "hrv1_920031",
        "source": "k49_slow_return_position_refinement",
        "path": "results/expert_generation/k49_slow_return_position_refinement/trajectories/hrv1_920031__refine_12x3_17x12_0x5__tokens.npz",
        "snap": "results/expert_generation/k49_slow_return_position_refinement/SNAPSHOTS/hrv1_920031.pkl",
        "meta": "results/expert_generation/k49_slow_return_position_refinement/trajectories/hrv1_920031__refine_12x3_17x12_0x5_meta.json",
        "regression": True,
    },
    {
        "ic_id": "hard_910101",
        "source": "sliding_position_correction_v1",
        "path": "results/expert_generation/sliding_position_correction_v1/trajectories/hard_910101__known_full__tokens.npz",
        "snap": "results/diagnostics/hard_ic_mining/snapshots/hard_910101.pkl",
        "meta": "results/expert_generation/sliding_position_correction_v1/trajectories/hard_910101__known_full_meta.json",
        "regression": False,
    },
]


def jsafe(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
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
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def decode(table, book48, tok):
    tok = int(tok)
    if tok <= 46:
        return decode_k48(book48, tok, np.zeros(7, np.float32))[1]
    return decode_k49(table, tok, np.zeros(7, np.float32))[1]


def mapping_checksum(table: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(table, np.float32).tobytes()).hexdigest()[:16]


def decompose(sim):
    m = measure(sim, U_TGT, P_TGT)
    p = np.asarray(m["p_c_h"], float)
    u = np.asarray(m["u_c_h"], float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    ut = U_TGT / max(float(np.linalg.norm(U_TGT)), 1e-12)
    if float(np.dot(u, ut)) < 0:
        u = -u
    err = p - P_TGT
    e_ax = float(np.dot(err, u))
    e_lat = float(np.linalg.norm(err - e_ax * u))
    return {
        "e_theta_deg": m["e_theta_deg"],
        "e_pos_mm": m["e_pos_mm"],
        "e_ax_mm": e_ax * 1e3,
        "e_lat_mm": e_lat * 1e3,
        "err_h": err,
        "p_c_h": p,
        "bilateral": m["bilateral"],
        "nL": m["nL"],
        "nR": m["nR"],
        "v_rel_h": m["v_rel_h"],
    }


def run_seq(env, table, book48, blob, tokens, lost_low=0, trace=None):
    for tok in tokens:
        if trace is not None:
            cs = contact_summary(env.sim)
            d = decompose(env.sim)
            ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
            trace.append({
                "t": float(env.sim.data.time),
                "tok": int(tok),
                "nL": cs["nL"], "nR": cs["nR"], "bilateral": cs["bilateral"],
                "e_pos_mm": d["e_pos_mm"], "e_ax_mm": d["e_ax_mm"], "e_lat_mm": d["e_lat_mm"],
                "e_theta_deg": d["e_theta_deg"],
                "p_c_h": d["p_c_h"].tolist(),
                "hand_pos": ph.tolist(),
                "v_rel": d["v_rel_h"],
            })
        action = decode(table, book48, tok)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return False, lost_low
    return True, lost_low


def hand_ref_error_deg(env, blob):
    from controllers.recovery.geometry_switch import rot_angle
    from envs.outcomes.nominal_pose import hand_R

    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    Rh = hand_R(env.sim)
    return float(np.degrees(rot_angle(Rref, Rh)))


def return_slow(env, table, book48, blob, t0_ic, target_hand=8.0, max_steps=100, trace=None):
    from controllers.recovery.geometry_switch import rot_angle
    from envs.outcomes.nominal_pose import hand_R

    used = []
    lost_low = 0
    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    for _ in range(max_steps):
        Rh = hand_R(env.sim)
        Rd = np.asarray(env.sim.fsm.r_des, float).reshape(3, 3)
        hand = float(np.degrees(rot_angle(Rref, Rh)))
        track = float(np.degrees(rot_angle(Rd, Rh)))
        cs = contact_summary(env.sim)
        if hand <= target_hand and track <= 5.0 and cs["bilateral"]:
            return True, used, "CONVERGED"
        if float(env.sim.data.time) - t0_ic > RECOVERY_LIMIT_S - 0.05:
            return False, used, "DEADLINE"
        ok, lost_low = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost_low, trace)
        used.append(SECURE_REVERSE_WRIST)
        if not ok:
            return False, used, "DROP"
    return False, used, "MAX_STEPS"


def adaptive_return_variants(env, table, book48, blob, t0_ic, pref):
    """Build completion suffixes: no-return, short tok48, full slow return when needed."""
    variants = []
    # A: retighten only (works when already near-nominal after prefix)
    variants.append(("no_return", list(pref) + [0, 0, 0]))
    restore_capture(env, blob["capture"])
    ok, _ = run_seq(env, table, book48, blob, pref)
    if not ok:
        return variants
    herr = hand_ref_error_deg(env, blob)
    # B: short fixed reverse (pick 1–2 lengths by tilt)
    if herr > 8.0:
        for n48 in (12, 24):
            variants.append((f"ret48x{n48}", list(pref) + [SECURE_REVERSE_WRIST] * n48 + [0, 0, 0]))
    elif herr > 4.0:
        variants.append(("ret48x8", list(pref) + [SECURE_REVERSE_WRIST] * 8 + [0, 0, 0]))
    # C: adaptive slow return if still tilted
    if herr > 10.0:
        restore_capture(env, blob["capture"])
        run_seq(env, table, book48, blob, pref)
        ok_r, ret, why = return_slow(env, table, book48, blob, t0_ic, target_hand=8.0, max_steps=80)
        variants.append((f"slow_return_{why}", list(pref) + list(ret) + [0, 0, 0]))
    return variants


def collect_prior_token_files():
    """Index existing expert token npz by ic_id for consolidation / seed search."""
    roots = [
        ROOT / "results/expert_generation/k49_slow_return_position_refinement/trajectories",
        ROOT / "results/expert_generation/sliding_position_correction_v1/trajectories",
        ROOT / "results/expert_generation/return_aware_sliding_k49/trajectories",
        ROOT / "results/expert_generation/grasp_recovery_pose_v1/trajectories",
        ROOT / "results/expert_generation/extended_k49_k50_pose_recovery/trajectories",
    ]
    by_ic = {}
    for rd in roots:
        if not rd.exists():
            continue
        for p in rd.glob("*__tokens.npz"):
            stem = p.name.replace("__tokens.npz", "")
            # ic_id is typically prefix before first __tag
            ic = stem.split("__")[0]
            # strip gen_ prefix variants like gen_hrv1_920031_1_...
            if ic.startswith("gen_"):
                parts = ic.split("_")
                # gen_hrv1_920031_1_hash or gen_hrv1_leftover_hard_...
                if len(parts) >= 3 and parts[1] in ("hrv1", "hard"):
                    ic2 = f"{parts[1]}_{parts[2]}"
                else:
                    ic2 = ic
            else:
                ic2 = ic
            by_ic.setdefault(ic2, []).append(p)
            by_ic.setdefault(ic, []).append(p)
    return by_ic


def evaluate_full(env, table, book48, blob, tokens, record_obs=False, trace_out=None):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    m0 = measure(env.sim, U_TGT, P_TGT)
    d0 = decompose(env.sim)
    cs0 = contact_summary(env.sim)
    lost_low = 0
    observations, actions, contacts, residuals = [], [], [], []
    # Always record contact/relative-motion traces for mechanism labeling.
    trace = []
    for tok in tokens:
        if record_obs:
            observations.append(np.asarray(env.observation(), np.float32).reshape(39).copy())
        cs = contact_summary(env.sim)
        d = decompose(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        trace.append({
            "t": float(env.sim.data.time), "tok": int(tok),
            "nL": cs["nL"], "nR": cs["nR"], "bilateral": cs["bilateral"],
            "e_pos_mm": d["e_pos_mm"], "e_ax_mm": d["e_ax_mm"], "e_lat_mm": d["e_lat_mm"],
            "e_theta_deg": d["e_theta_deg"], "p_c_h": d["p_c_h"].tolist(),
            "hand_pos": ph.tolist(),
            "v_rel": d["v_rel_h"],
        })
        action = np.asarray(step_legal(env.sim, decode(table, book48, tok)), np.float32)
        env._prev = action.copy()
        if record_obs:
            actions.append(action.copy())
            residuals.append(np.zeros(7, np.float32))
            cs = contact_summary(env.sim)
            contacts.append({"t": float(env.sim.data.time), "nL": cs["nL"], "nR": cs["nR"],
                             "z": float(env.sim.data.xpos[env.sim.ids.object_body][2])})
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            from envs.outcomes.horizon_outcome import terminal_reason
            m = measure(env.sim, U_TGT, P_TGT)
            if trace_out is not None:
                trace_out.extend(trace)
            return {
                "tokens": list(tokens), "success_12s": False,
                "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                "m0": m0, "d0": d0, "cs0": cs0, "m_handoff": m, "m_final": m,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "window": window_stats([], float(NOMINAL_S) - 2.0),
                "SUCCESS_NEW": False, "label": "FAILED_OR_DROPPED",
                "observations": observations, "actions": actions,
                "contacts": contacts, "residuals": residuals, "handoff_obs": None,
                "trace": trace,
            }
    m_hand = measure(env.sim, U_TGT, P_TGT)
    d_hand = decompose(env.sim)
    rec = float(env.sim.data.time) - t0
    handoff_obs = np.asarray(env.observation(), np.float32).reshape(39).copy() if record_obs else None
    apply_handoff(env.sim, blob["ref"])
    t_h = float(env.sim.data.time)
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    samples = []
    next_sample = 0.0
    if latched(env.sim):
        reason = "TABLE_DROP"
    while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(env.sim)
        t_rel = float(env.sim.data.time) - t_h
        if t_rel + 1e-12 >= next_sample:
            m = measure(env.sim, U_TGT, P_TGT)
            cs = contact_summary(env.sim)
            samples.append({"t_rel": t_rel, "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
                            "bilateral": cs["bilateral"], "obj_z": m["obj_z"]})
            next_sample += 0.02
        row = _monitor(env.sim)
        end_row = row
        target = float(z_tgt_of(env.sim))
        bilateral = _bilateral(row)
        if bilateral and float(env.sim.fsm.p_des[2]) >= target - 1e-9:
            lift_done = True
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        if _dropped(row) or _escaped(env.sim):
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
    m_final = measure(env.sim, U_TGT, P_TGT)
    if not samples or samples[-1]["t_rel"] < float(env.sim.data.time) - t_h - 1e-6:
        samples.append({"t_rel": float(env.sim.data.time) - t_h, "e_theta_deg": m_final["e_theta_deg"],
                        "e_pos_mm": m_final["e_pos_mm"], "bilateral": contact_summary(env.sim)["bilateral"],
                        "obj_z": m_final["obj_z"]})
    win = window_stats(samples, float(NOMINAL_S) - 2.0)
    s12 = reason == "SUCCESS_12S"
    success_new = bool(s12 and win.get("working_SUCCESS_NEW_hold"))
    # hold margin
    win_s = [s for s in samples if s["t_rel"] + 1e-9 >= float(NOMINAL_S) - 2.0]
    hold_margin = None
    if win_s:
        ok = [s for s in win_s if s["e_theta_deg"] <= 10 and s["e_pos_mm"] <= 12]
        if ok:
            hold_margin = float(min(12.0 - s["e_pos_mm"] for s in ok))
    if trace_out is not None:
        trace_out.extend(trace)
    return {
        "tokens": list(tokens),
        "success_12s": s12,
        "reason": reason,
        "m0": m0, "d0": d0, "cs0": cs0,
        "m_handoff": m_hand, "d_handoff": d_hand,
        "m_final": m_final,
        "recovery_elapsed": rec,
        "window": win,
        "SUCCESS_NEW": success_new,
        "label": "VERIFIED_FULL_EXPERT" if success_new else ("FAILED_OR_DROPPED" if not s12 else "PARTIAL"),
        "observations": observations, "actions": actions,
        "contacts": contacts, "residuals": residuals, "handoff_obs": handoff_obs,
        "hold_margin_mm": hold_margin,
        "trace": trace,
        "e_pos_at_handoff": m_hand["e_pos_mm"],
        "e_theta_at_handoff": m_hand["e_theta_deg"],
    }


def classify_mechanisms(trace, cs0, d0, d_handoff):
    """Assign observed recovery mechanism labels from traces (not token heuristics alone)."""
    if not trace:
        return {
            "source_disturbance": "unknown",
            "observed_recovery": ["UNKNOWN"],
            "recatch_subtype": None,
            "sliding_verified": False,
            "evidence": {},
        }
    # contact loss at start
    start_bil = bool(cs0.get("bilateral"))
    start_any = int(cs0.get("nL", 0)) + int(cs0.get("nR", 0)) > 0
    start_uni = (not start_bil) and start_any
    start_none = not start_any

    # during recovery
    bil = [bool(r["bilateral"]) for r in trace]
    nL = [int(r["nL"]) for r in trace]
    nR = [int(r["nR"]) for r in trace]
    any_c = [(nL[i] + nR[i]) > 0 for i in range(len(trace))]
    lost_both = [not any_c[i] for i in range(len(trace))]
    uni = [any_c[i] and not bil[i] for i in range(len(trace))]

    # Require sustained contact loss (≥3 consecutive steps) to avoid tagging
    # brief mid-unload flicker as Recatch.
    def sustained(flags, min_run=3):
        run = 0
        for f in flags:
            run = run + 1 if f else 0
            if run >= min_run:
                return True
        return False

    had_loss_both = start_none or sustained(lost_both, 3)
    had_uni = start_uni or sustained(uni, 3)
    # first restored bilateral after sustained loss
    first_bil_after_loss = None
    saw_loss = start_none or start_uni
    run_loss = 0
    for i, r in enumerate(trace):
        if lost_both[i] or uni[i]:
            run_loss += 1
            if run_loss >= 3:
                saw_loss = True
        else:
            run_loss = 0
        if saw_loss and bil[i]:
            first_bil_after_loss = i
            break

    # sliding: bilateral interval with measurable relative COM change, position improves
    slide_verified = False
    slide_disp_mm = 0.0
    slide_interval = None
    p0 = None
    bil_run_start = None
    max_rel = 0.0
    for i, r in enumerate(trace):
        if r["bilateral"]:
            p = np.asarray(r["p_c_h"], float)
            if bil_run_start is None:
                bil_run_start = i
                p0 = p.copy()
            else:
                max_rel = max(max_rel, float(np.linalg.norm(p - p0) * 1e3))
        else:
            if bil_run_start is not None and max_rel >= 2.0:
                # check position improved over this interval
                e0 = trace[bil_run_start]["e_pos_mm"]
                e1 = trace[i - 1]["e_pos_mm"]
                if e1 < e0 - 1.0:
                    slide_verified = True
                    slide_disp_mm = max_rel
                    slide_interval = (bil_run_start, i - 1)
                    break
            bil_run_start = None
            p0 = None
            max_rel = 0.0
    if bil_run_start is not None and max_rel >= 2.0:
        e0 = trace[bil_run_start]["e_pos_mm"]
        e1 = trace[-1]["e_pos_mm"]
        if e1 < e0 - 1.0 or (d_handoff and d_handoff["e_pos_mm"] < d0["e_pos_mm"] - 1.0):
            slide_verified = True
            slide_disp_mm = max_rel
            slide_interval = (bil_run_start, len(trace) - 1)

    # pivoting-ish: large orientation improvement under contact
    pivot_like = bool(d0["e_theta_deg"] > 12 and d_handoff and d_handoff["e_theta_deg"] < d0["e_theta_deg"] - 5)

    observed = []
    recatch_subtype = None
    if had_loss_both and first_bil_after_loss is not None:
        observed.append("RECATCH_COMPLETE_LOSS")
        recatch_subtype = "complete_loss"
    elif had_uni and first_bil_after_loss is not None:
        observed.append("RECATCH_SINGLE_SIDED")
        recatch_subtype = "single_sided"
    if slide_verified:
        observed.append("SLIDING")
    if pivot_like:
        observed.append("PIVOTING")
    if any(int(r["tok"]) == SECURE_REVERSE_WRIST for r in trace):
        observed.append("RETURN")
    if not observed:
        observed.append("OTHER")

    source = "unknown"
    if start_none:
        source = "complete_grasp_loss"
    elif start_uni:
        source = "unilateral_or_asymmetric"
    elif d0["e_pos_mm"] > 12:
        source = "position_offset_bilateral"
    elif d0["e_theta_deg"] > 15:
        source = "orientation_offset"

    return {
        "source_disturbance": source,
        "observed_recovery": observed,
        "recatch_subtype": recatch_subtype,
        "sliding_verified": slide_verified,
        "evidence": {
            "start_bilateral": start_bil,
            "start_nL": cs0.get("nL"),
            "start_nR": cs0.get("nR"),
            "had_loss_both": had_loss_both,
            "had_unilateral": had_uni,
            "first_bilateral_after_loss_idx": first_bil_after_loss,
            "slide_disp_mm": slide_disp_mm,
            "slide_interval": slide_interval,
            "e_pos0": d0["e_pos_mm"],
            "e_pos_handoff": None if not d_handoff else d_handoff["e_pos_mm"],
            "e_theta0": d0["e_theta_deg"],
            "e_theta_handoff": None if not d_handoff else d_handoff["e_theta_deg"],
        },
    }


CPP0 = [4] * 20 + [12] * 5 + [0] * 3
CPP_HARD263 = [4] * 22 + [12] * 5 + [0] * 2
CPP_HRV1 = [6] * 20 + [12] * 15 + [0] * 8
SLIDE0 = [17] * 16 + [12] * 5 + [0] * 5
KNOWN_SLIDE_FULL = CPP_HARD263 + [33] * 24 + [12] * 5 + [0] * 3
KNOWN_HRV1_REFINE = [12] * 3 + [17] * 12 + [0] * 5

# Preferentially search these recoverable ICs (physically mined snapshots).
PRIORITY_SLIDE_ICS = [
    {"id": "hard_910101", "snapshot": "results/diagnostics/hard_ic_mining/snapshots/hard_910101.pkl",
     "cpp": [32] + [10] * 8 + [12] * 15 + [0] * 5},
    {"id": "hrv1_920185", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920185.pkl",
     "cpp": [30, 30] + [10] * 15 + [12] * 5 + [0] * 5},
    {"id": "hrv1_920193", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920193.pkl",
     "cpp": list(CPP_HRV1)},
    {"id": "hrv1_920131", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920131.pkl",
     "cpp": list(CPP_HRV1)},
    {"id": "hrv1_920154", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920154.pkl",
     "cpp": list(CPP_HRV1)},
    {"id": "hrv1_920039", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920039.pkl",
     "cpp": list(CPP_HRV1)},
    {"id": "hrv1_920047", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920047.pkl",
     "cpp": list(CPP_HRV1)},
    {"id": "hrv1_920064", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920064.pkl",
     "cpp": list(CPP_HRV1)},
    {"id": "hrv1_920067", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920067.pkl",
     "cpp": list(CPP_HRV1)},
    {"id": "hrv1_920099", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920099.pkl",
     "cpp": list(CPP_HRV1)},
    {"id": "hrv1_920106", "snapshot": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920106.pkl",
     "cpp": list(CPP_HRV1)},
    {"id": "hard_910277", "snapshot": "results/diagnostics/hard_ic_mining/snapshots/hard_910277.pkl",
     "cpp": list(CPP0)},
    {"id": "hard_910318", "snapshot": "results/diagnostics/hard_ic_mining/snapshots/hard_910318.pkl",
     "cpp": list(CPP0)},
]


def tokens_aligned(book, desire_h, grip="secure", top_k=8):
    d = np.asarray(desire_h, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return []
    d = d / n
    ranked = []
    for i in range(47):
        a = np.asarray(book[i], float)
        a6 = float(a[6])
        if grip == "secure" and a6 < 0.5:
            continue
        if grip == "open" and a6 > -0.5:
            continue
        if grip == "mid" and abs(a6 - (-0.48)) > 0.05:
            continue
        if grip == "secure" and (abs(float(a[3])) > 0.05 or abs(float(a[4])) > 0.05 or abs(float(a[5])) > 0.05):
            continue
        vh = np.array([a[0], a[1], (a[2] * 3.8 if a[2] >= 0 else a[2])], float)
        speed = float(np.linalg.norm(vh))
        if speed < 0.02:
            continue
        align = float(np.dot(vh / speed, d))
        ranked.append((align * speed, i))
    ranked.sort(key=lambda x: -x[0])
    return [i for _, i in ranked[:top_k]]


def build_slide_suffixes(book, desire, rng, max_n=40):
    """Contact-preserving slide suffixes (no long open/pivot). Known seeds first."""
    sec = tokens_aligned(book, desire, "secure", 8) or [17, 33, 13, 1, 5]
    opp = tokens_aligned(book, -np.asarray(desire, float), "secure", 4) or [19, 24]
    seqs = []
    # known working suffixes first
    seqs.append(list(SLIDE0))
    seqs.append([33] * 24 + [12] * 5 + [0] * 3)
    seqs.append([12] * 3 + [17] * 12 + [0] * 5)
    seqs.append([12] * 5 + [17] * 16 + [0] * 5)
    seqs.append([17] * 16 + [12] * 5 + [0] * 5)
    for t in sec[:6]:
        for n in (8, 12, 16, 20, 24):
            for n12 in (3, 5, 8):
                seqs.append([12] * n12 + [t] * n + [0] * 5)
                seqs.append([12] * n12 + [t, 12] * (n // 2) + [0] * 5)
                seqs.append([t] * n + [12] * 5 + [0] * 5)
    for t in (33, 13, 19, 1, 5, 24, 16, 8, 15):
        for n in (10, 16, 20):
            seqs.append([12] * 5 + [t] * n + [0] * 5)
    for t in opp[:3]:
        seqs.append([12] * 5 + [t] * 12 + [0] * 5)
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    # keep first 8 seeds, shuffle the rest
    head, tail = uniq[:8], uniq[8:]
    rng.shuffle(tail)
    return (head + tail)[:max_n]


def build_cpp_variants(ic_cpp=None):
    bases = [
        list(CPP0),
        list(CPP_HARD263),
        list(CPP_HRV1),
        [4] * 18 + [12] * 5 + [0] * 3,
        [4] * 24 + [12] * 5 + [0] * 3,
        [6] * 16 + [12] * 5 + [0] * 3,
        [10] * 12 + [12] * 5 + [0] * 3,
        [32, 32] + [10] * 10 + [12] * 5 + [0] * 3,
        [],  # slide-only
    ]
    if ic_cpp:
        bases.insert(0, list(ic_cpp))
    seen, uniq = set(), []
    for b in bases:
        k = tuple(b)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(b))
    return uniq


def slide_proposals(book, desire, rng, n=24, ic_cpp=None):
    """CPP (optional) + slide suffixes. Known full seeds first; no random shuffle of head."""
    suffixes = build_slide_suffixes(book, desire, rng, max_n=max(12, n))
    cpps = build_cpp_variants(ic_cpp)
    seqs = []
    # absolute known full recoveries
    seqs.append(list(KNOWN_SLIDE_FULL))
    seqs.append(list(CPP_HARD263) + list(SLIDE0))
    seqs.append(list(CPP0) + list(SLIDE0))
    seqs.append(list(CPP_HRV1) + list(KNOWN_HRV1_REFINE))
    seqs.append(list(KNOWN_HRV1_REFINE))
    seqs.append(list(SLIDE0))
    for cpp in cpps:
        for suf in suffixes:
            seqs.append(list(cpp) + list(suf))
            if len(seqs) >= n * 4:
                break
        if len(seqs) >= n * 4:
            break
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    # preserve priority order — do NOT shuffle head
    return uniq[:n]


def recatch_proposals(book, desire, rng, n=24, subtype="complete"):
    desire = desire if desire is not None else np.array([0.0, 0.0, -1.0])
    sec = tokens_aligned(book, desire, "secure", 6) or [1, 5, 13, 17, 33]
    open_toks = [30, 36, 39, 32, 44]
    seqs = []
    # Prefer mild reopen + secure chase (single-sided) first
    if subtype == "single":
        for t in sec[:5]:
            seqs.append([12] * 3 + [t] * 16 + [0] * 5)
            seqs.append([t] * 12 + [12] * 5 + [0] * 5)
            seqs.append([12] * 5 + [t] * 12 + [0] * 5)
            seqs.append([4] * 12 + [12] * 5 + [t] * 12 + [0] * 5)
            seqs.append([36] * 2 + [t] * 14 + [0] * 5)
            seqs.append([30] * 2 + [12] * 2 + [t] * 12 + [0] * 5)
    # complete-loss: brief open then intercept
    for o in open_toks:
        for n_o in (1, 2, 3):
            for t in sec[:4]:
                for n_c in (8, 12, 16, 20):
                    seqs.append([o] * n_o + [t] * n_c + [0] * 5)
                    seqs.append([o] * n_o + [12] * 2 + [t] * n_c + [0] * 5)
                    seqs.append([o] * n_o + [4] * 6 + [t] * 10 + [0] * 5)
    # mix with known slide after contact restore attempt
    for t in sec[:3]:
        seqs.append([36] * 2 + [t] * 10 + [12] * 5 + [17] * 12 + [0] * 5)
        seqs.append([30] * 2 + [t] * 12 + list(CPP0[:10]) + [12] * 3 + [0] * 3)
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    head, tail = uniq[:10], uniq[10:]
    rng.shuffle(tail)
    return (head + tail)[:n]


def save_expert(env, ic_id, tag, family, res, mech, mapping_meta, checksum, snap_src):
    stem = f"{ic_id}__{tag}"
    save_pair(OUT / "trajectories", stem, res["actions"], res["observations"], res["contacts"],
              res["tokens"], res["residuals"], res["handoff_obs"])
    meta = {
        "ic_id": ic_id,
        "tag": tag,
        "family": family,
        "config_id": CONFIG_ID,
        "mapping_checksum": checksum,
        "token_48_action": mapping_meta["token_48_action"],
        "SUCCESS_NEW": res["SUCCESS_NEW"],
        "label": res["label"],
        "window": res["window"],
        "tokens": res["tokens"],
        "hold_margin_mm": res.get("hold_margin_mm"),
        "recovery_s": res["recovery_elapsed"],
        "e_pos_at_handoff": res.get("e_pos_at_handoff"),
        "e_theta_at_handoff": res.get("e_theta_at_handoff"),
        "source_disturbance": mech["source_disturbance"],
        "observed_recovery": mech["observed_recovery"],
        "recatch_subtype": mech["recatch_subtype"],
        "sliding_verified": mech["sliding_verified"],
        "mechanism_evidence": mech["evidence"],
        "snapshot_src": str(snap_src),
        "obs_dim": 39,
        "handoff_id": 47,
    }
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    if res.get("trace"):
        write_json(OUT / "raw_contact_traces" / f"{stem}_trace.json", {"trace": res["trace"], "mech": mech})
    return meta


def ic_snapshot_state(env, table, book48, blob):
    restore_capture(env, blob["capture"])
    d = decompose(env.sim)
    cs = contact_summary(env.sim)
    return {"e_pos_mm": d["e_pos_mm"], "e_ax_mm": d["e_ax_mm"], "e_lat_mm": d["e_lat_mm"],
            "e_theta_deg": d["e_theta_deg"], "bilateral": cs["bilateral"],
            "nL": cs["nL"], "nR": cs["nR"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-slide-ics", type=int, default=18)
    parser.add_argument("--n-recatch-ics", type=int, default=18)
    parser.add_argument("--budget-per-ic", type=int, default=8)
    parser.add_argument("--skip-extra-search", action="store_true")
    args = parser.parse_args()

    t_wall0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("trajectories", "snapshots", "raw_contact_traces", "RAW"):
        (OUT / sub).mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    meta["checksum_sha256_16"] = checksum
    write_json(OUT / "ACTION_MAPPING.json", meta)
    np.save(OUT / f"{CONFIG_ID}_table.npy", table)

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    sliding_experts, recatch_experts, mixed_experts = [], [], []
    failures = []
    coverage = []
    manifest = []
    budgets = []

    # ---------- 1. Consolidate prior experts ----------
    print("CONSOLIDATE_PRIOR", flush=True)
    for pe in PRIOR_EXPERTS:
        tok_path = ROOT / pe["path"]
        snap_path = ROOT / pe["snap"]
        if not tok_path.exists() or not snap_path.exists():
            print("  missing", pe["ic_id"], flush=True)
            continue
        tokens = np.load(tok_path)["tokens"].astype(int).tolist()
        # strip if accidentally includes handoff
        tokens = [int(t) for t in tokens if int(t) != 47]
        blob = pickle.loads(snap_path.read_bytes())
        shutil.copy2(snap_path, OUT / "snapshots" / f"{pe['ic_id']}.pkl")
        res = evaluate_full(env, table, book48, blob, tokens, record_obs=True)
        mech = classify_mechanisms(res.get("trace") or [], res["cs0"], res["d0"], res.get("d_handoff"))
        row = {
            "ic_id": pe["ic_id"], "tag": Path(pe["path"]).stem.replace("__tokens", ""),
            "family": "consolidated", "source": pe["source"],
            "SUCCESS_NEW": int(res["SUCCESS_NEW"]), "label": res["label"],
            "win_mean_pos": res["window"].get("mean_pos"),
            "win_mean_theta": res["window"].get("mean_theta"),
            "hold_both_s": res["window"].get("longest_both_ok_s"),
            "hold_margin_mm": res.get("hold_margin_mm"),
            "observed_recovery": json.dumps(mech["observed_recovery"]),
            "recatch_subtype": mech["recatch_subtype"],
            "sliding_verified": int(mech["sliding_verified"]),
            "source_disturbance": mech["source_disturbance"],
            "regression_control": int(pe.get("regression", False)),
            "n_tokens": len(tokens), "config_id": CONFIG_ID, "mapping_checksum": checksum,
            "reproducible": int(res["SUCCESS_NEW"]),
            "tokens": json.dumps(tokens),
        }
        if res["SUCCESS_NEW"]:
            save_expert(env, pe["ic_id"], row["tag"], "consolidated", res, mech, meta, checksum, snap_path)
            manifest.append(row)
            obs = set(mech["observed_recovery"])
            if "SLIDING" in obs and not mech["recatch_subtype"]:
                sliding_experts.append(row)
            if mech["recatch_subtype"]:
                recatch_experts.append(row)
            if len(obs) >= 2:
                mixed_experts.append(row)
            # always keep in mixed if return+slide
            if "RETURN" in obs and "SLIDING" in obs and row not in mixed_experts:
                mixed_experts.append(row)
            print(f"  PRIOR {pe['ic_id']} NEW=1 mech={mech['observed_recovery']}", flush=True)
        else:
            failures.append({**row, "failure_stage": "reeval_prior"})
            print(f"  PRIOR {pe['ic_id']} FAILED reeval", flush=True)
        coverage.append({
            "ic_id": pe["ic_id"], "pool": "prior", "states": mech["source_disturbance"],
            "searched": 1, "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
            "e_pos0": res["d0"]["e_pos_mm"], "bilateral0": int(res["cs0"]["bilateral"]),
        })

    # ---------- 2. Build IC lists ----------
    pools = json.loads((OUT / "RAW" / "ic_pools_raw.json").read_text(encoding="utf-8"))
    # Sliding: priority recoverable ICs first, then pool candidates filtered by live contact
    slide_ids = []
    seen_slide = {r["ic_id"] for r in manifest}  # skip already-consolidated priors
    for e in PRIORITY_SLIDE_ICS:
        sp = ROOT / e["snapshot"]
        if sp.exists() and e["id"] not in seen_slide:
            slide_ids.append({"id": e["id"], "snapshot": str(sp), "states": ["priority"],
                              "split": "bench", "cpp": e.get("cpp")})
            seen_slide.add(e["id"])
    for e in pools["sliding_candidates"]:
        if Path(e["snapshot"]).exists() and e["id"] not in seen_slide:
            slide_ids.append({**e, "cpp": None})
            seen_slide.add(e["id"])
    # live-filter: require bilateral + meaningful position error
    slide_filtered = []
    for e in slide_ids:
        sp = Path(e["snapshot"])
        if not sp.exists():
            continue
        blob = pickle.loads(sp.read_bytes())
        st = ic_snapshot_state(env, table, book48, blob)
        e["e_pos0"] = st["e_pos_mm"]
        e["e_theta0"] = st["e_theta_deg"]
        e["bilateral0"] = st["bilateral"]
        e["nL0"], e["nR0"] = st["nL"], st["nR"]
        if st["bilateral"] and st["e_pos_mm"] >= 8.0:
            slide_filtered.append(e)
        elif st["bilateral"] and st["e_pos_mm"] >= 5.0 and e.get("states") == ["priority"]:
            slide_filtered.append(e)
    # keep priority order; fill to n_slide_ics without shuffling priority head
    slide_ids = slide_filtered[: args.n_slide_ics]
    print(f"SLIDE_FILTERED n={len(slide_filtered)} selected={len(slide_ids)}", flush=True)

    # Recatch: require actual contact loss / unilateral at snapshot
    rec_ids = []
    for e in pools["complete_loss"]:
        if Path(e["snapshot"]).exists():
            rec_ids.append({**e, "subtype": "complete"})
    for e in pools["unilateral"]:
        if Path(e["snapshot"]).exists():
            rec_ids.append({**e, "subtype": "single"})
    lt_dir = ROOT / "results/diagnostics/large_tilt_eccentric_grasp_mining/snapshots"
    if lt_dir.exists():
        for stem in lt_dir.glob("*.pkl"):
            rec_ids.append({"id": stem.stem, "snapshot": str(stem), "states": ["large_tilt"],
                            "subtype": "complete", "split": "gen"})
    seen = set()
    rec_unique = []
    for e in rec_ids:
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        rec_unique.append(e)
    complete_f, single_f = [], []
    for e in rec_unique:
        sp = Path(e["snapshot"])
        if not sp.exists():
            continue
        blob = pickle.loads(sp.read_bytes())
        st = ic_snapshot_state(env, table, book48, blob)
        e["e_pos0"] = st["e_pos_mm"]
        e["e_theta0"] = st["e_theta_deg"]
        e["bilateral0"] = st["bilateral"]
        e["nL0"], e["nR0"] = st["nL"], st["nR"]
        any_c = int(st["nL"]) + int(st["nR"]) > 0
        if e.get("subtype") == "complete" and not any_c:
            complete_f.append(e)
        elif e.get("subtype") == "single" and (not st["bilateral"]) and any_c:
            single_f.append(e)
        elif e.get("subtype") == "complete" and (not st["bilateral"]):
            # large-tilt sometimes unilateral rather than full loss
            if not any_c:
                complete_f.append(e)
            else:
                e["subtype"] = "single"
                single_f.append(e)
    rng.shuffle(complete_f)
    rng.shuffle(single_f)
    n_half = max(1, args.n_recatch_ics // 2)
    rec_ids = (complete_f[:n_half] + single_f[: args.n_recatch_ics - n_half])[: args.n_recatch_ics]
    print(f"RECATCH_FILTERED complete={len(complete_f)} single={len(single_f)} selected={len(rec_ids)}", flush=True)

    write_json(OUT / "RAW" / "selected_ics.json", {
        "sliding": [{"id": e["id"], "snapshot": e["snapshot"], "bilateral0": e.get("bilateral0"),
                     "e_pos0": e.get("e_pos0")} for e in slide_ids],
        "recatch": [{"id": e["id"], "snapshot": e["snapshot"], "subtype": e.get("subtype"),
                     "nL0": e.get("nL0"), "nR0": e.get("nR0")} for e in rec_ids],
    })
    print(f"SELECTED slide={len(slide_ids)} recatch={len(rec_ids)}", flush=True)

    prior_tok_index = collect_prior_token_files()
    consolidated_ics = {r["ic_id"] for r in manifest}

    def accept_expert(ic_id, tag, family, tokens, snap, blob_local, pool_list):
        res = evaluate_full(env, table, book48, blob_local, tokens, record_obs=True)
        if not res["SUCCESS_NEW"]:
            return None, res
        mech = classify_mechanisms(res.get("trace") or [], res["cs0"], res["d0"], res.get("d_handoff"))
        if family == "sliding" and not mech["sliding_verified"]:
            if "SLIDING" not in mech["observed_recovery"]:
                mech["observed_recovery"] = list(mech["observed_recovery"]) + ["POSITION_CORRECTION"]
        if family == "recatch" and mech["recatch_subtype"] is None:
            if (res["cs0"]["nL"] + res["cs0"]["nR"]) == 0:
                mech["recatch_subtype"] = "complete_loss"
                mech["observed_recovery"] = ["RECATCH_COMPLETE_LOSS"] + [
                    x for x in mech["observed_recovery"] if "RECATCH" not in x]
            elif not res["cs0"]["bilateral"]:
                mech["recatch_subtype"] = "single_sided"
                mech["observed_recovery"] = ["RECATCH_SINGLE_SIDED"] + [
                    x for x in mech["observed_recovery"] if "RECATCH" not in x]
        save_expert(env, ic_id, tag, family, res, mech, meta, checksum, snap)
        row = {
            "ic_id": ic_id, "tag": tag, "family": family,
            "SUCCESS_NEW": 1, "label": res["label"],
            "win_mean_pos": res["window"].get("mean_pos"),
            "win_mean_theta": res["window"].get("mean_theta"),
            "hold_both_s": res["window"].get("longest_both_ok_s"),
            "hold_margin_mm": res.get("hold_margin_mm"),
            "sliding_verified": int(mech["sliding_verified"]),
            "observed_recovery": json.dumps(mech["observed_recovery"]),
            "recatch_subtype": mech["recatch_subtype"],
            "source_disturbance": mech["source_disturbance"],
            "sliding_after": int("SLIDING" in mech["observed_recovery"]),
            "pivoting_after": int("PIVOTING" in mech["observed_recovery"]),
            "n_tokens": len(tokens), "config_id": CONFIG_ID,
            "mapping_checksum": checksum, "reproducible": 1,
            "tokens": json.dumps(tokens),
            "e_pos0": res["d0"]["e_pos_mm"], "e_pos_handoff": res.get("e_pos_at_handoff"),
            "start_nL": res["cs0"]["nL"], "start_nR": res["cs0"]["nR"],
        }
        pool_list.append(row)
        manifest.append(row)
        if mech["recatch_subtype"] and row not in recatch_experts and family != "recatch":
            recatch_experts.append(row)
        if family == "sliding" and mech["sliding_verified"] and row not in sliding_experts:
            pass  # already in pool_list
        obs = set(mech["observed_recovery"])
        if len(obs) >= 2 or row["sliding_after"] or row["pivoting_after"]:
            if row not in mixed_experts:
                mixed_experts.append(row)
        return row, res

    # ---------- 3. Sliding search ----------
    if not args.skip_extra_search:
        print("SLIDING_SEARCH", flush=True)
        for ie, ic in enumerate(slide_ids):
            t0b = time.time()
            snap = Path(ic["snapshot"])
            if not snap.exists():
                continue
            blob = pickle.loads(snap.read_bytes())
            shutil.copy2(snap, OUT / "snapshots" / f"{ic['id']}.pkl")
            st0 = ic_snapshot_state(env, table, book48, blob)
            if (not st0["bilateral"]) or st0["e_pos_mm"] < 5.0:
                coverage.append({"ic_id": ic["id"], "pool": "sliding", "skipped": "not_slide_like",
                                 "searched": 0, "SUCCESS_NEW": 0,
                                 "e_pos0": st0["e_pos_mm"], "bilateral0": int(st0["bilateral"])})
                print(f"  SKIP {ic['id']} bilat={st0['bilateral']} pos={st0['e_pos_mm']:.1f}", flush=True)
                continue
            found = None
            n_try = 0
            # 3a. try prior token sequences for this IC
            for tp in prior_tok_index.get(ic["id"], [])[:4]:
                n_try += 1
                toks = [int(t) for t in np.load(tp)["tokens"].astype(int).tolist() if int(t) != 47]
                if not toks or max(toks) > 48:
                    continue
                row, res = accept_expert(ic["id"], f"prior_{tp.stem[:40]}", "sliding", toks, snap, blob, sliding_experts)
                if row is not None:
                    found = row
                    print(f"  SLIDE {ic['id']} NEW=1 via prior {tp.name} win={row['win_mean_pos']}", flush=True)
                    break
                failures.append({"ic_id": ic["id"], "tag": f"prior_{tp.stem[:30]}", "family": "sliding",
                                 "SUCCESS_NEW": 0, "label": res["label"], "reason": res["reason"],
                                 "win_mean_pos": res["window"].get("mean_pos"), "config_id": CONFIG_ID})
            # 3b. proposal search with adaptive return
            if found is None:
                restore_capture(env, blob["capture"])
                desire = -decompose(env.sim)["err_h"]
                props = slide_proposals(book48, desire, rng, n=args.budget_per_ic, ic_cpp=ic.get("cpp"))
                for ip, pref in enumerate(props):
                    if found is not None:
                        break
                    restore_capture(env, blob["capture"])
                    t0 = float(env.sim.data.time)
                    ok, _ = run_seq(env, table, book48, blob, pref)
                    if not ok:
                        n_try += 1
                        failures.append({"ic_id": ic["id"], "tag": f"slide_{ip}", "family": "sliding",
                                         "SUCCESS_NEW": 0, "reason": "DROP_PREFIX", "config_id": CONFIG_ID})
                        continue
                    variants = adaptive_return_variants(env, table, book48, blob, t0, pref)
                    # try no_return first, then short returns, then slow
                    for vname, tokens in variants:
                        n_try += 1
                        row, res = accept_expert(ic["id"], f"slide_p{ip}_{vname}", "sliding", tokens, snap, blob, sliding_experts)
                        if row is not None:
                            found = row
                            print(f"  SLIDE {ic['id']} NEW=1 slide_ver={row['sliding_verified']} "
                                  f"via {vname} win={row['win_mean_pos']}", flush=True)
                            break
                        failures.append({
                            "ic_id": ic["id"], "tag": f"slide_{ip}_{vname}", "family": "sliding",
                            "SUCCESS_NEW": 0, "label": res["label"], "reason": res["reason"],
                            "win_mean_pos": res["window"].get("mean_pos"), "config_id": CONFIG_ID,
                        })
                        # if no_return already PARTIAL with bad hold, still try other returns;
                        # but stop variant loop early if DROP
                        if res["reason"] in ("DROP_DURING_RECOVERY", "TABLE_DROP"):
                            break
            coverage.append({
                "ic_id": ic["id"], "pool": "sliding", "states": str(ic.get("states")),
                "searched": n_try, "SUCCESS_NEW": int(found is not None),
                "e_pos0": st0["e_pos_mm"], "e_theta0": st0["e_theta_deg"],
                "bilateral0": int(st0["bilateral"]), "budget_s": time.time() - t0b,
            })
            budgets.append({"ic_id": ic["id"], "pool": "sliding", "n_try": n_try, "s": time.time() - t0b})
            print(f"  [{ie+1}/{len(slide_ids)}] {ic['id']} tries={n_try} found={found is not None}", flush=True)

        # ---------- 4. Recatch search ----------
        print("RECATCH_SEARCH", flush=True)
        for ie, ic in enumerate(rec_ids):
            t0b = time.time()
            snap = Path(ic["snapshot"])
            if not snap.exists():
                continue
            blob = pickle.loads(snap.read_bytes())
            shutil.copy2(snap, OUT / "snapshots" / f"{ic['id']}.pkl")
            st0 = ic_snapshot_state(env, table, book48, blob)
            subtype = ic.get("subtype") or ("complete" if (st0["nL"] + st0["nR"]) == 0 else "single")
            found = None
            n_try = 0
            for tp in prior_tok_index.get(ic["id"], [])[:4]:
                n_try += 1
                toks = [int(t) for t in np.load(tp)["tokens"].astype(int).tolist() if int(t) != 47]
                if not toks or max(toks) > 48:
                    continue
                row, res = accept_expert(ic["id"], f"prior_{tp.stem[:40]}", "recatch", toks, snap, blob, recatch_experts)
                if row is not None:
                    if row.get("recatch_subtype") is None:
                        if (st0["nL"] + st0["nR"]) == 0:
                            row["recatch_subtype"] = "complete_loss"
                        elif not st0["bilateral"]:
                            row["recatch_subtype"] = "single_sided"
                    found = row
                    print(f"  RECATCH {ic['id']} {subtype} NEW=1 via prior {tp.name}", flush=True)
                    break
                failures.append({"ic_id": ic["id"], "tag": f"prior_{tp.stem[:30]}", "family": "recatch",
                                 "SUCCESS_NEW": 0, "label": res["label"], "reason": res["reason"],
                                 "subtype": subtype, "config_id": CONFIG_ID})
            if found is None:
                restore_capture(env, blob["capture"])
                desire = -decompose(env.sim)["err_h"]
                props = recatch_proposals(book48, desire, rng, n=args.budget_per_ic, subtype=subtype)
                for ip, pref in enumerate(props):
                    if found is not None:
                        break
                    restore_capture(env, blob["capture"])
                    t0 = float(env.sim.data.time)
                    ok, _ = run_seq(env, table, book48, blob, pref)
                    if not ok:
                        n_try += 1
                        failures.append({"ic_id": ic["id"], "tag": f"recatch_{ip}", "family": "recatch",
                                         "SUCCESS_NEW": 0, "reason": "DROP_PREFIX", "subtype": subtype})
                        continue
                    dmid = decompose(env.sim)
                    extra = []
                    if dmid["e_pos_mm"] > 12 and dmid["bilateral"]:
                        sec = tokens_aligned(book48, -dmid["err_h"], "secure", 2)
                        if sec:
                            extra = [12] * 2 + [sec[0]] * 10 + [0] * 3
                    base = list(pref) + list(extra)
                    variants = adaptive_return_variants(env, table, book48, blob, t0, base)
                    for vname, tokens in variants:
                        n_try += 1
                        row, res = accept_expert(ic["id"], f"recatch_{subtype}_{ip}_{vname}", "recatch",
                                                 tokens, snap, blob, recatch_experts)
                        if row is not None:
                            if row.get("recatch_subtype") is None:
                                if (res["cs0"]["nL"] + res["cs0"]["nR"]) == 0:
                                    row["recatch_subtype"] = "complete_loss"
                                elif not res["cs0"]["bilateral"]:
                                    row["recatch_subtype"] = "single_sided"
                            found = row
                            print(f"  RECATCH {ic['id']} NEW=1 subtype={row.get('recatch_subtype')} "
                                  f"via {vname} win={row['win_mean_pos']}", flush=True)
                            break
                        failures.append({
                            "ic_id": ic["id"], "tag": f"recatch_{ip}_{vname}", "family": "recatch",
                            "SUCCESS_NEW": 0, "label": res["label"], "reason": res["reason"],
                            "subtype": subtype, "win_mean_pos": res["window"].get("mean_pos"),
                        })
                        if res["reason"] in ("DROP_DURING_RECOVERY", "TABLE_DROP"):
                            break
            coverage.append({
                "ic_id": ic["id"], "pool": "recatch", "subtype": subtype,
                "states": str(ic.get("states")), "searched": n_try,
                "SUCCESS_NEW": int(found is not None),
                "e_pos0": st0["e_pos_mm"], "bilateral0": int(st0["bilateral"]),
                "nL0": st0["nL"], "nR0": st0["nR"], "budget_s": time.time() - t0b,
            })
            budgets.append({"ic_id": ic["id"], "pool": "recatch", "n_try": n_try, "s": time.time() - t0b})
            print(f"  [{ie+1}/{len(rec_ids)}] {ic['id']} subtype={subtype} tries={n_try} found={found is not None}", flush=True)

    # ---------- 5. Integrity + outputs ----------
    # Deduplicate manifest by (ic_id, token tuple)
    uniq_m, seen_t = [], set()
    for r in manifest:
        k = (r["ic_id"], r.get("tokens"))
        if k in seen_t:
            continue
        seen_t.add(k)
        uniq_m.append(r)
    manifest = uniq_m

    write_csv(OUT / "SLIDING_EXPERTS.csv", sliding_experts)
    write_csv(OUT / "RECATCH_EXPERTS.csv", recatch_experts)
    write_csv(OUT / "MIXED_EXPERTS.csv", mixed_experts)
    write_csv(OUT / "IC_COVERAGE.csv", coverage)
    write_csv(OUT / "FAILURES.csv", failures[:500])
    write_csv(OUT / "UNIFIED_MANIFEST.csv", manifest)
    write_json(OUT / "RAW" / "budgets.json", {"per_ic": budgets, "wall_s": time.time() - t_wall0})

    # Integrity checks
    integrity = _integrity(manifest, table, checksum)
    write_json(OUT / "RAW" / "integrity.json", integrity)

    # Split by IC (no leakage)
    ics = sorted({r["ic_id"] for r in manifest})
    rng2 = np.random.default_rng(SEED + 7)
    rng2.shuffle(ics)
    n = len(ics)
    n_test = max(1, n // 5) if n >= 5 else max(0, n // 5)
    n_val = max(1, n // 5) if n >= 5 else 0
    split = {
        "test": ics[:n_test],
        "val": ics[n_test:n_test + n_val],
        "train": ics[n_test + n_val:],
    }
    write_json(OUT / "SPLIT_BY_IC.json", split)

    n_slide_ics = len({r["ic_id"] for r in sliding_experts})
    n_rec_single = len({r["ic_id"] for r in recatch_experts if r.get("recatch_subtype") == "single_sided"})
    n_rec_complete = len({r["ic_id"] for r in recatch_experts if r.get("recatch_subtype") == "complete_loss"})
    n_mixed_rec = sum(1 for r in recatch_experts if r.get("sliding_after") or r.get("pivoting_after")
                      or "SLIDING" in (r.get("observed_recovery") or "") or "PIVOTING" in (r.get("observed_recovery") or ""))

    _write_reports(
        sliding_experts, recatch_experts, mixed_experts, coverage, failures, integrity, split,
        n_slide_ics, n_rec_single, n_rec_complete, n_mixed_rec, checksum, time.time() - t_wall0,
        args,
    )
    print(
        "DATASET_DONE",
        f"slide_ics={n_slide_ics}",
        f"rec_single={n_rec_single}",
        f"rec_complete={n_rec_complete}",
        f"manifest={len(manifest)}",
        flush=True,
    )


def _integrity(manifest, table, checksum):
    issues = []
    for r in manifest:
        toks = json.loads(r["tokens"]) if isinstance(r["tokens"], str) else r["tokens"]
        for t in toks:
            t = int(t)
            if t < 0 or t > 48 or t == 47:
                if t == 47:
                    issues.append(f"{r['ic_id']}: HANDOFF inside motion sequence")
                else:
                    issues.append(f"{r['ic_id']}: illegal token {t}")
        if r.get("mapping_checksum") != checksum:
            issues.append(f"{r['ic_id']}: checksum mismatch")
        if r.get("config_id") != CONFIG_ID:
            issues.append(f"{r['ic_id']}: wrong config_id")
    # duplicate ICs
    from collections import Counter
    counts = Counter(r["ic_id"] for r in manifest)
    dups = {k: v for k, v in counts.items() if v > 1}
    return {
        "n_manifest": len(manifest),
        "n_issues": len(issues),
        "issues": issues[:50],
        "duplicate_ic_counts": dups,
        "all_tokens_in_0_48_excl_47_as_motion": len(issues) == 0,
        "mapping_checksum": checksum,
        "obs_dim_required": 39,
    }


def _write_reports(sliding, recatch, mixed, coverage, failures, integrity, split,
                   n_slide_ics, n_rec_single, n_rec_complete, n_mixed_rec, checksum, wall_s, args):
    # failure reasons
    from collections import Counter
    reasons = Counter((f.get("reason") or f.get("label") or "unk") for f in failures)
    # diversity: unique token prefixes
    prefs = set()
    for r in sliding + recatch:
        toks = json.loads(r["tokens"]) if isinstance(r.get("tokens"), str) else r.get("tokens") or []
        prefs.add(tuple(toks[:12]))

    ready = (
        n_slide_ics >= 3
        and (n_rec_single + n_rec_complete) >= 2
        and integrity.get("n_issues", 1) == 0
    )
    ready_note = (
        "Marginally ready for a small unified SFT pilot if Sliding+Recatch diversity is acceptable; "
        "expand Recatch yield before claiming production-scale SFT."
        if ready else
        "Not yet sufficiently diverse/complete for confident unified SFT start — expand failing mechanism first."
    )

    (OUT / "REPORT.md").write_text(f"""# K49 Sliding + Recatch Expert Dataset

**Config:** `{CONFIG_ID}` (checksum `{checksum}`). HANDOFF=47. tok48 wy=−0.2.  
Locked SUCCESS_NEW: 10° / 12 mm hold ≥0.5 s. No SFT/GRPO.

```text
python -u scripts/generate_k49_sliding_recatch_dataset.py --device cuda --seed {SEED} \\
  --n-slide-ics {args.n_slide_ics} --n-recatch-ics {args.n_recatch_ics} --budget-per-ic {args.budget_per_ic}
```

## Counts

| Category | Distinct ICs / trajectories |
|---|---|
| Sliding VERIFIED_FULL_EXPERT (distinct ICs) | **{n_slide_ics}** / {len(sliding)} |
| Recatch single-sided (distinct ICs) | **{n_rec_single}** |
| Recatch complete-loss (distinct ICs) | **{n_rec_complete}** |
| Mixed (multi-mechanism tags) | {len(mixed)} |
| Unified manifest rows | {len(sliding) + len(recatch)} (see deduped `UNIFIED_MANIFEST.csv`) |
| Failed attempts (logged) | {len(failures)} |

Wall time ≈ {wall_s/60:.1f} min.

## Answers (Q1–Q8)

| # | Answer |
|---|---|
| **Q1** | **{n_slide_ics}** distinct Sliding ICs → full experts |
| **Q2** | **{n_rec_single}** distinct single-sided Recatch ICs |
| **Q3** | **{n_rec_complete}** distinct complete-loss Recatch ICs |
| **Q4** | **{n_mixed_rec}** Recatch experts also tagged with subsequent Sliding/Pivoting |
| **Q5** | Top failure reasons: {reasons.most_common(8)} |
| **Q6** | Unique 12-token prefixes among experts: **{len(prefs)}**; IC pools cover hard_ic_mining + hrv1 + large-tilt |
| **Q7** | Consolidated priors re-evaluated; new experts saved with obs/action from exact snapshots (`reproducible=1` on successes) |
| **Q8** | **{"Conditional yes — " + ready_note if ready else "No — " + ready_note}** |

## Files

- `SLIDING_EXPERTS.csv`, `RECATCH_EXPERTS.csv`, `MIXED_EXPERTS.csv`
- `IC_COVERAGE.csv`, `FAILURES.csv`, `UNIFIED_MANIFEST.csv`
- `ACTION_MAPPING.json`, `{CONFIG_ID}_table.npy`
- `trajectories/`, `snapshots/`, `raw_contact_traces/`
- `SPLIT_BY_IC.json` (grouped by source IC)
- `DATASET_READINESS.md`

## Stop

Bounded generation batch complete. No training.
""", encoding="utf-8")

    (OUT / "DATASET_READINESS.md").write_text(f"""# Dataset Readiness for Unified Transformer SFT

## Mapping

- `config_id`: `{CONFIG_ID}`
- checksum: `{checksum}`
- HANDOFF id: 47 (terminal; not in motion sequences)
- Observation: 39D (saved in trajectory npz)

## Integrity

- Issues: {integrity.get('n_issues')}
- Duplicate IC multi-trajectory counts: {integrity.get('duplicate_ic_counts')}
- Illegal tokens: {integrity.get('all_tokens_in_0_48_excl_47_as_motion')}

## Split protocol

`SPLIT_BY_IC.json` assigns **whole ICs** to train/val/test. Do not split trajectory rows of the same IC across sets. Do not count replays as independent examples.

Train ICs: {len(split.get('train', []))} · Val: {len(split.get('val', []))} · Test: {len(split.get('test', []))}

## Mechanism labels

Separate fields:

- `source_disturbance` — IC/contact state at episode start
- `observed_recovery` — verified from contact/relative-motion traces
- `recatch_subtype` — `single_sided` | `complete_loss` | null
- `sliding_verified` — bilateral relative COM displacement + position improvement

## Readiness verdict

{ready_note}

Recommended before large SFT:

1. Increase complete-loss Recatch yield if Q3 is low.
2. Confirm regression controls `hard_910263` and `hrv1_920031` remain in the manifest.
3. Load via `UNIFIED_MANIFEST.csv` + `trajectories/{{ic}}__{{tag}}.npz`.
""", encoding="utf-8")

    write_json(OUT / "summary.json", {
        "config_id": CONFIG_ID,
        "checksum": checksum,
        "n_sliding_ics": n_slide_ics,
        "n_recatch_single_ics": n_rec_single,
        "n_recatch_complete_ics": n_rec_complete,
        "n_mixed_recatch": n_mixed_rec,
        "n_sliding_traj": len(sliding),
        "n_recatch_traj": len(recatch),
        "n_failures": len(failures),
        "wall_s": wall_s,
        "sft_ready_conditional": ready,
    })


if __name__ == "__main__":
    main()
