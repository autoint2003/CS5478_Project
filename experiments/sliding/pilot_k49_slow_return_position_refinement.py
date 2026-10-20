"""Bounded Sliding/CPP refinement with fixed K49_tok48_wy0p25 slow return.

Target: hard_910263 VERIFIED_FULL_EXPERT under locked 10°/12mm SUCCESS_NEW.
No new tokens. No physics/reward changes. No SFT/GRPO.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import rot_angle
from data_generation.recording.trajectory_recorder import save_pair
from envs.ballistic_impact import z_tgt_of
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor, recovery_failed
from envs.outcomes.nominal_pose import hand_R
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary, window_stats
from evaluation.grasp_geometry import measure
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

OUT = ROOT / "results" / "expert_generation" / "k49_slow_return_position_refinement"
CONFIG_ID = "K49_tok48_wy0p25"
SNAP = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910263.pkl"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
PRIOR_FULL = ROOT / "results/diagnostics/k49_slow_reverse_token48/FULL_RECOVERY_RESULTS.csv"
SEED = 20261018
RECOVERY_LIMIT_S = 4.0

CPP0 = [4] * 20 + [12] * 5 + [0] * 3
SLIDE0 = [17] * 16 + [12] * 5 + [0] * 5
RETIGHTEN = [0, 0, 0]

EXTRA_ICS = [
    {"id": "hrv1_920031", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920031.pkl",
     "cpp": [6] * 20 + [12] * 15 + [0] * 8},
    {"id": "hard_910101", "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910101.pkl",
     "cpp": [32] + [10] * 8 + [12] * 15 + [0] * 5},
    {"id": "hrv1_920185", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920185.pkl",
     "cpp": [30, 30] + [10] * 15 + [12] * 5 + [0] * 5},
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
        "err_h": err.tolist(),
        "p_c_h": p.tolist(),
        "u_c_h": u.tolist(),
        "bilateral": m["bilateral"],
        "nL": m["nL"],
        "nR": m["nR"],
        "v_rel_h": m["v_rel_h"],
    }


def rich(sim, blob, label=""):
    d = decompose(sim)
    Rh = hand_R(sim)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    cs = contact_summary(sim)
    return {
        "stage": label,
        "t": float(sim.data.time),
        **d,
        "hand_to_ref_deg": float(np.degrees(rot_angle(Rref, Rh))),
        "rdes_to_ref_deg": float(np.degrees(rot_angle(Rref, Rd))),
        "hand_to_rdes_deg": float(np.degrees(rot_angle(Rd, Rh))),
        "bilateral_cs": cs["bilateral"],
    }


def run_seq(env, table, book48, blob, tokens, lost_low=0):
    for tok in tokens:
        action = decode(table, book48, tok)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return False, lost_low
    return True, lost_low


def return_slow(env, table, book48, blob, t0_ic, target_hand=8.0, max_steps=120):
    used = []
    lost_low = 0
    stages = []
    for i in range(max_steps):
        st = rich(env.sim, blob, f"ret_{i}")
        stages.append(st)
        if (st["hand_to_ref_deg"] <= target_hand and st["hand_to_rdes_deg"] <= 5.0
                and st["bilateral"]):
            return True, used, stages, "CONVERGED"
        if float(env.sim.data.time) - t0_ic > RECOVERY_LIMIT_S - 0.05:
            return False, used, stages, "DEADLINE"
        ok, lost_low = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost_low)
        used.append(SECURE_REVERSE_WRIST)
        if not ok:
            return False, used, stages, "DROP"
    st = rich(env.sim, blob, "ret_end")
    stages.append(st)
    ok = st["hand_to_ref_deg"] <= target_hand + 1.0
    return ok, used, stages, "CONVERGED" if ok else "MAX_STEPS"


def evaluate_full(env, table, book48, blob, tokens, record_obs=False):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    m0 = measure(env.sim, U_TGT, P_TGT)
    lost_low = 0
    observations, actions, contacts, residuals = [], [], [], []
    for tok in tokens:
        if record_obs:
            observations.append(np.asarray(env.observation(), np.float32).reshape(39).copy())
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
            return {
                "tokens": list(tokens), "success_12s": False,
                "reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY",
                "m0": m0, "m_handoff": m, "m_final": m,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "window": window_stats([], float(NOMINAL_S) - 2.0),
                "SUCCESS_NEW": False, "label": "FAILED_OR_DROPPED",
                "observations": observations, "actions": actions,
                "contacts": contacts, "residuals": residuals, "handoff_obs": None,
                "pre_handoff": None,
            }
    pre = rich(env.sim, blob, "pre_handoff")
    m_hand = measure(env.sim, U_TGT, P_TGT)
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
            d = decompose(env.sim)
            samples.append({
                "t_rel": t_rel, "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
                "e_ax_mm": d["e_ax_mm"], "e_lat_mm": d["e_lat_mm"],
                "bilateral": cs["bilateral"], "obj_z": m["obj_z"],
            })
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
        d = decompose(env.sim)
        samples.append({
            "t_rel": float(env.sim.data.time) - t_h, "e_theta_deg": m_final["e_theta_deg"],
            "e_pos_mm": m_final["e_pos_mm"], "e_ax_mm": d["e_ax_mm"], "e_lat_mm": d["e_lat_mm"],
            "bilateral": contact_summary(env.sim)["bilateral"], "obj_z": m_final["obj_z"],
        })
    win = window_stats(samples, float(NOMINAL_S) - 2.0)
    # hold margin: min (12 - e_pos) over frames where both ok
    hold_margin = None
    win_samps = [s for s in samples if s["t_rel"] + 1e-9 >= float(NOMINAL_S) - 2.0]
    if win_samps:
        ok_mask = [(s["e_theta_deg"] <= 10.0 and s["e_pos_mm"] <= 12.0) for s in win_samps]
        if any(ok_mask):
            margins = [12.0 - s["e_pos_mm"] for s, ok in zip(win_samps, ok_mask) if ok]
            hold_margin = float(min(margins)) if margins else None
    s12 = reason == "SUCCESS_12S"
    success_new = bool(s12 and win.get("working_SUCCESS_NEW_hold"))
    # final window axial/lateral means
    if win_samps:
        win_ax = float(np.mean([s["e_ax_mm"] for s in win_samps]))
        win_lat = float(np.mean([s["e_lat_mm"] for s in win_samps]))
    else:
        win_ax = win_lat = None
    return {
        "tokens": list(tokens),
        "success_12s": s12,
        "reason": reason,
        "m0": m0,
        "m_handoff": m_hand,
        "m_final": m_final,
        "pre_handoff": pre,
        "recovery_elapsed": rec,
        "window": win,
        "SUCCESS_NEW": success_new,
        "label": "VERIFIED_FULL_EXPERT" if success_new else ("FAILED_OR_DROPPED" if not s12 else "PARTIAL"),
        "observations": observations,
        "actions": actions,
        "contacts": contacts,
        "residuals": residuals,
        "handoff_obs": handoff_obs,
        "e_pos_at_handoff": m_hand["e_pos_mm"],
        "e_theta_at_handoff": m_hand["e_theta_deg"],
        "hold_margin_mm": hold_margin,
        "win_mean_ax_mm": win_ax,
        "win_mean_lat_mm": win_lat,
        "nominal_samples_tail": win_samps[-5:] if win_samps else [],
    }


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
        if abs(float(a[3])) > 0.05 or abs(float(a[4])) > 0.05 or abs(float(a[5])) > 0.05:
            continue
        vh = np.array([a[0], a[1], (a[2] * 3.8 if a[2] >= 0 else a[2])], float)
        speed = float(np.linalg.norm(vh))
        if speed < 0.02:
            continue
        align = float(np.dot(vh / speed, d))
        ranked.append((align * speed, i, align, speed))
    ranked.sort(key=lambda x: -x[0])
    return ranked[:top_k]


def build_slide_variants(book, desire_h, rng):
    """Bounded local variants around baseline SLIDE0."""
    sec = [t for _, t, *_ in tokens_aligned(book, desire_h, "secure", 8)]
    opp = [t for _, t, *_ in tokens_aligned(book, -np.asarray(desire_h, float), "secure", 4)]
    # axial desire: reduce |e_ax| when e_ax negative => move object +u => hand -u under mid
    seqs = []
    # baseline
    seqs.append(("base", list(SLIDE0)))
    # duration sweeps on tok17
    for n in (10, 12, 14, 16, 18, 20, 22, 24, 28):
        for n12 in (3, 5, 8):
            for n0 in (3, 5, 8):
                seqs.append((f"17x{n}_12x{n12}_0x{n0}", [17] * n + [12] * n12 + [0] * n0))
                seqs.append((f"12x{n12}_17x{n}_0x{n0}", [12] * n12 + [17] * n + [0] * n0))
                seqs.append((f"12x{n12}_alt17_{n}_0x{n0}", [12] * n12 + [17, 12] * (n // 2) + [0] * n0))
    # other secure translate tokens
    for t in sec[:6]:
        for n in (12, 16, 20, 24):
            seqs.append((f"t{t}x{n}", [12] * 5 + [t] * n + [0] * 5))
            seqs.append((f"t{t}x{n}_mid", [12] * 5 + [t, 12] * (n // 2) + [0] * 5))
            seqs.append((f"t{t}x{n}_first", [t] * n + [12] * 5 + [0] * 5))
    # unload longer / shorter
    for n12 in (2, 4, 6, 10, 12):
        seqs.append((f"unload{n12}_17x16", [12] * n12 + [17] * 16 + [0] * 5))
    # opposite direction control
    for t in opp[:3]:
        seqs.append((f"opp_t{t}", [12] * 5 + [t] * 12 + [0] * 5))
    # known axial-ish
    for t in (33, 13, 19, 1, 5, 24, 16):
        for n in (8, 12, 16, 20):
            seqs.append((f"ax_t{t}x{n}", [12] * 5 + [t] * n + [0] * 5))
            seqs.append((f"ax_alt_t{t}x{n}", [12] * 3 + [t, 12] * (n // 2) + [0] * 4))
    # retighten timing: slide then longer/shorter 0
    for n0 in (2, 4, 6, 10):
        seqs.append((f"base_rt{n0}", [17] * 16 + [12] * 5 + [0] * n0))

    seen, uniq = set(), []
    for name, s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((name, list(k)))
    rng.shuffle(uniq)
    return uniq


def build_cpp_variants(rng):
    """Small local CPP variations around [4]*20+[12]*5+[0]*3."""
    vars_ = [("cpp_base", list(CPP0))]
    for n4 in (16, 18, 20, 22, 24):
        for n12 in (3, 5, 8):
            for n0 in (2, 3, 5):
                vars_.append((f"cpp_4x{n4}_12x{n12}_0x{n0}", [4] * n4 + [12] * n12 + [0] * n0))
    # mix with tok6 (pure +wy)
    for n6 in (10, 15, 20):
        vars_.append((f"cpp_6x{n6}", [6] * n6 + [12] * 5 + [0] * 3))
        vars_.append((f"cpp_4x10_6x{n6}", [4] * 10 + [6] * n6 + [12] * 5 + [0] * 3))
    # tok10 coupled
    for n10 in (8, 12, 16):
        vars_.append((f"cpp_10x{n10}", [10] * n10 + [12] * 5 + [0] * 3))
        vars_.append((f"cpp_4x12_10x{n10}", [4] * 12 + [10] * n10 + [12] * 5 + [0] * 3))
    rng.shuffle(vars_)
    return vars_


def pack_cand(tag, family, tokens, res, extra=None):
    pre = res.get("pre_handoff") or {}
    row = {
        "tag": tag,
        "family": family,
        "config_id": CONFIG_ID,
        "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
        "success_12s": int(res["success_12s"]),
        "label": res["label"],
        "reason": res["reason"],
        "win_mean_theta": res["window"].get("mean_theta"),
        "win_mean_pos": res["window"].get("mean_pos"),
        "win_mean_ax": res.get("win_mean_ax_mm"),
        "win_mean_lat": res.get("win_mean_lat_mm"),
        "hold_both_s": res["window"].get("longest_both_ok_s"),
        "hold_margin_mm": res.get("hold_margin_mm"),
        "e_pos_at_handoff": res.get("e_pos_at_handoff"),
        "e_theta_at_handoff": res.get("e_theta_at_handoff"),
        "e_ax_at_handoff": pre.get("e_ax_mm"),
        "e_lat_at_handoff": pre.get("e_lat_mm"),
        "hand_to_ref": pre.get("hand_to_ref_deg"),
        "recovery_s": res["recovery_elapsed"],
        "n_tokens": len(tokens),
        "n_tok48": sum(1 for t in tokens if int(t) == SECURE_REVERSE_WRIST),
        "tokens": json.dumps(tokens),
    }
    if extra:
        row.update(extra)
    return row


def save_expert(env, ic_id, source, res):
    if not res.get("observations"):
        return
    stem = f"{ic_id}__{source}"
    save_pair(OUT / "trajectories", stem, res["actions"], res["observations"], res["contacts"],
              res["tokens"], res["residuals"], res["handoff_obs"])
    write_json(OUT / "trajectories" / f"{stem}_meta.json", {
        "ic_id": ic_id, "source": source, "config_id": CONFIG_ID,
        "SUCCESS_NEW": res["SUCCESS_NEW"], "label": res["label"],
        "window": res["window"], "tokens": res["tokens"],
        "hold_margin_mm": res.get("hold_margin_mm"),
    })


def analyze_baseline(env, table, book48, blob):
    """Stage-wise decomposition of the prior best wy0p25 trajectory."""
    stages = []
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    stages.append(rich(env.sim, blob, "IC"))
    run_seq(env, table, book48, blob, CPP0)
    stages.append(rich(env.sim, blob, "after_CPP"))
    run_seq(env, table, book48, blob, SLIDE0)
    stages.append(rich(env.sim, blob, "after_Sliding"))
    run_seq(env, table, book48, blob, RETIGHTEN)
    stages.append(rich(env.sim, blob, "before_Return"))
    ok_r, ret, _, why = return_slow(env, table, book48, blob, t0)
    stages.append(rich(env.sim, blob, "after_Return"))
    run_seq(env, table, book48, blob, RETIGHTEN)
    stages.append(rich(env.sim, blob, "pre_HANDOFF"))
    tokens = CPP0 + SLIDE0 + RETIGHTEN + ret + RETIGHTEN
    full = evaluate_full(env, table, book48, blob, tokens, record_obs=False)
    win = full["window"]
    decomp_rows = []
    for s in stages:
        decomp_rows.append({
            "stage": s["stage"],
            "t": s["t"],
            "e_theta_deg": s["e_theta_deg"],
            "e_pos_mm": s["e_pos_mm"],
            "e_ax_mm": s["e_ax_mm"],
            "e_lat_mm": s["e_lat_mm"],
            "hand_to_ref_deg": s["hand_to_ref_deg"],
            "bilateral": int(s["bilateral"]),
        })
    # final window summary row
    decomp_rows.append({
        "stage": "final_window_mean",
        "t": None,
        "e_theta_deg": win.get("mean_theta"),
        "e_pos_mm": win.get("mean_pos"),
        "e_ax_mm": full.get("win_mean_ax_mm"),
        "e_lat_mm": full.get("win_mean_lat_mm"),
        "hand_to_ref_deg": None,
        "bilateral": None,
        "hold_both_s": win.get("longest_both_ok_s"),
        "SUCCESS_NEW": int(full["SUCCESS_NEW"]),
        "return_why": why,
        "n_tok48": len(ret),
    })
    write_csv(OUT / "POSITION_ERROR_DECOMPOSITION.csv", decomp_rows)
    write_json(OUT / "RAW" / "baseline_stages.json", {"stages": stages, "full_summary": {
        "SUCCESS_NEW": full["SUCCESS_NEW"], "win": win,
        "handoff_pos": full.get("e_pos_at_handoff"),
        "blocker": _blocker(full),
    }})
    print("BASELINE_BLOCKER", _blocker(full), "win_pos", win.get("mean_pos"),
          "win_ax", full.get("win_mean_ax_mm"), "win_lat", full.get("win_mean_lat_mm"), flush=True)
    return stages, full, tokens, why


def _blocker(full):
    if full["SUCCESS_NEW"]:
        return "NONE"
    if not full["success_12s"]:
        return f"NO_S12:{full['reason']}"
    win = full["window"]
    th = win.get("mean_theta")
    pos = win.get("mean_pos")
    hold = win.get("longest_both_ok_s") or 0
    if th is not None and th > 10:
        return "ORIENTATION_WINDOW"
    if pos is not None and pos > 12:
        return f"POSITION_WINDOW({pos:.2f}mm)"
    if hold < 0.5:
        return f"HOLD_TOO_SHORT({hold:.2f}s)"
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--budget-slide", type=int, default=70)
    parser.add_argument("--budget-cpp", type=int, default=24)
    parser.add_argument("--skip-extra", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("RAW", "SNAPSHOTS", "trajectories"):
        (OUT / sub).mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    meta = describe_k49_config(CONFIG_ID, book48)
    write_json(OUT / "ACTION_MAPPING.json", meta)
    np.save(OUT / f"{CONFIG_ID}_table.npy", table)

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    blob = pickle.loads(SNAP.read_bytes())
    shutil.copy2(SNAP, OUT / "SNAPSHOTS" / "hard_910263.pkl")

    print("ANALYZE_BASELINE", CONFIG_ID, "tok48", meta["token_48_action"], flush=True)
    stages, base_full, base_tokens, base_why = analyze_baseline(env, table, book48, blob)

    cand_rows = []
    experts = []
    best = None

    # baseline as candidate
    cand_rows.append(pack_cand("baseline_wy0p25", "baseline", base_tokens, base_full,
                               {"return_why": base_why}))

    # --- Sliding search with fixed CPP0 ---
    print("SLIDE_SEARCH", flush=True)
    restore_capture(env, blob["capture"])
    run_seq(env, table, book48, blob, CPP0)
    desire = -np.asarray(decompose(env.sim)["err_h"], float)
    slides = build_slide_variants(book48, desire, rng)[: args.budget_slide]
    for i, (name, slide) in enumerate(slides):
        tokens_prefix = CPP0 + slide + RETIGHTEN
        restore_capture(env, blob["capture"])
        t0 = float(env.sim.data.time)
        ok, _ = run_seq(env, table, book48, blob, tokens_prefix)
        if not ok:
            cand_rows.append({"tag": name, "family": "slide", "SUCCESS_NEW": 0, "label": "DROP_PREFIX",
                              "config_id": CONFIG_ID})
            continue
        post_slide = rich(env.sim, blob, "post_slide")
        ok_r, ret, _, why = return_slow(env, table, book48, blob, t0)
        tokens = tokens_prefix + ret + RETIGHTEN
        res = evaluate_full(env, table, book48, blob, tokens, record_obs=False)
        row = pack_cand(name, "slide", tokens, res, {
            "return_why": why,
            "e_pos_post_slide": post_slide["e_pos_mm"],
            "e_ax_post_slide": post_slide["e_ax_mm"],
            "e_lat_post_slide": post_slide["e_lat_mm"],
            "e_pos_post_return_pre_eval": None,
        })
        cand_rows.append(row)
        if i % 8 == 0 or res["SUCCESS_NEW"] or (
            res["success_12s"] and (res["window"].get("mean_pos") or 99) < 12.2
        ):
            print(
                f"  S[{i}] {name}: post_slide={post_slide['e_pos_mm']:.2f} "
                f"handoff={res.get('e_pos_at_handoff')} win={res['window'].get('mean_pos')} "
                f"NEW={res['SUCCESS_NEW']} hold={res['window'].get('longest_both_ok_s')}",
                flush=True,
            )
        if res["SUCCESS_NEW"]:
            best = {**res, "tokens": tokens, "tag": name, "family": "slide"}
            break
        if best is None or _better(res, best):
            best = {**res, "tokens": tokens, "tag": name, "family": "slide"}

    # --- Joint CPP+slide if no NEW yet ---
    if not (best and best.get("SUCCESS_NEW")):
        print("CPP_SLIDE_SEARCH", flush=True)
        # take top slide by win_pos among S12
        slide_ok = [r for r in cand_rows if r.get("family") == "slide" and r.get("success_12s")
                    and r.get("win_mean_pos") is not None]
        slide_ok.sort(key=lambda r: float(r["win_mean_pos"]))
        top_slides = []
        for r in slide_ok[:5]:
            top_slides.append(json.loads(r["tokens"]))
        # extract slide portion: after CPP0 length
        slide_suffixes = []
        for toks in top_slides:
            # tokens = CPP0 + slide + RETIGHTEN + return...; take between CPP0 and first 48
            try:
                i0 = len(CPP0)
                # find first 48
                i48 = next(i for i, t in enumerate(toks) if int(t) == SECURE_REVERSE_WRIST)
                # back up past RETIGHTEN zeros
                suf = toks[i0:i48]
                while suf and suf[-1] == 0:
                    suf = suf[:-1]
                # include a short retighten in suffix rebuild
                slide_suffixes.append(suf)
            except StopIteration:
                pass
        if not slide_suffixes:
            slide_suffixes = [SLIDE0 + RETIGHTEN]

        cpp_vars = build_cpp_variants(rng)[: args.budget_cpp]
        for j, (cname, cpp) in enumerate(cpp_vars):
            for k, suf in enumerate(slide_suffixes[:3]):
                # suf may already include retighten; ensure ends with some 0s
                slide_part = list(suf)
                if not slide_part or slide_part[-1] != 0:
                    slide_part = slide_part + RETIGHTEN
                tokens_prefix = list(cpp) + slide_part
                restore_capture(env, blob["capture"])
                t0 = float(env.sim.data.time)
                ok, _ = run_seq(env, table, book48, blob, tokens_prefix)
                if not ok:
                    continue
                ok_r, ret, _, why = return_slow(env, table, book48, blob, t0)
                tokens = tokens_prefix + ret + RETIGHTEN
                res = evaluate_full(env, table, book48, blob, tokens, record_obs=False)
                tag = f"{cname}__suf{k}"
                row = pack_cand(tag, "cpp_slide", tokens, res, {"return_why": why})
                cand_rows.append(row)
                if res["SUCCESS_NEW"] or (j % 6 == 0 and k == 0):
                    print(
                        f"  C[{j}.{k}] {tag}: win={res['window'].get('mean_pos')} "
                        f"NEW={res['SUCCESS_NEW']} handoff={res.get('e_pos_at_handoff')}",
                        flush=True,
                    )
                if res["SUCCESS_NEW"]:
                    best = {**res, "tokens": tokens, "tag": tag, "family": "cpp_slide"}
                    break
                if best is None or _better(res, best):
                    best = {**res, "tokens": tokens, "tag": tag, "family": "cpp_slide"}
            if best and best.get("SUCCESS_NEW"):
                break

    write_csv(OUT / "CANDIDATE_SEARCH.csv", cand_rows)

    # Replay + robustness if SUCCESS_NEW
    h263_full = False
    if best and best.get("SUCCESS_NEW"):
        print("REPLAY_AND_ROBUSTNESS", flush=True)
        res1 = evaluate_full(env, table, book48, blob, best["tokens"], record_obs=True)
        res2 = evaluate_full(env, table, book48, blob, best["tokens"], record_obs=False)
        replay_ok = bool(res1["SUCCESS_NEW"] and res2["SUCCESS_NEW"])
        write_json(OUT / "RAW" / "replay.json", {
            "run1_NEW": res1["SUCCESS_NEW"], "run2_NEW": res2["SUCCESS_NEW"],
            "win1": res1["window"], "win2": res2["window"],
            "hold_margin": res1.get("hold_margin_mm"),
        })
        # timing sensitivity: ±1/±2 tok48, ±1 slide duration
        sens = []
        toks = list(best["tokens"])
        # find return block
        idxs48 = [i for i, t in enumerate(toks) if int(t) == SECURE_REVERSE_WRIST]
        if idxs48:
            for dn in (-2, -1, 1, 2):
                t2 = list(toks)
                if dn < 0:
                    # remove |dn| tok48 from end of block
                    for _ in range(abs(dn)):
                        if idxs48:
                            i = idxs48.pop()
                            if i < len(t2) and t2[i] == SECURE_REVERSE_WRIST:
                                t2.pop(i)
                                idxs48 = [i for i, t in enumerate(t2) if int(t) == SECURE_REVERSE_WRIST]
                else:
                    # insert dn tok48 at end of block
                    last = idxs48[-1]
                    for _ in range(dn):
                        t2.insert(last + 1, SECURE_REVERSE_WRIST)
                        last += 1
                r = evaluate_full(env, table, book48, blob, t2, record_obs=False)
                sens.append({"var": f"tok48_delta_{dn}", "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                             "win_pos": r["window"].get("mean_pos"), "hold": r["window"].get("longest_both_ok_s")})
        write_csv(OUT / "RAW" / "timing_sensitivity.csv", sens)
        robust = replay_ok and all(s["SUCCESS_NEW"] for s in sens if s["var"] in ("tok48_delta_-1", "tok48_delta_1"))
        if replay_ok:
            h263_full = True
            save_expert(env, "hard_910263", best.get("tag", "best"), res1)
            experts.append(pack_cand(best.get("tag", "best"), best.get("family", "best"),
                                     best["tokens"], res1, {"replay_ok": int(replay_ok), "robust_pm1": int(robust)}))
            write_json(OUT / "RAW" / "robustness.json", {"replay_ok": replay_ok, "robust_pm1_tok48": robust, "sens": sens})
            print("HARD263_FULL", "robust", robust, "margin", res1.get("hold_margin_mm"), flush=True)
    else:
        print("NO_FULL_EXPERT best_win", None if best is None else best.get("window", {}).get("mean_pos"), flush=True)
        if best is not None:
            write_json(OUT / "RAW" / "best_partial.json", {
                "tag": best.get("tag"), "win": best.get("window"),
                "handoff_pos": best.get("e_pos_at_handoff"),
                "tokens": best.get("tokens"),
            })

    # Extra ICs with same method (bounded) if we want Q6 — always try small budget
    extra_experts = []
    if not args.skip_extra:
        print("EXTRA_ICS", flush=True)
        for spec in EXTRA_ICS:
            path = ROOT / spec["path"]
            if not path.exists():
                continue
            bl = pickle.loads(path.read_bytes())
            shutil.copy2(path, OUT / "SNAPSHOTS" / f"{spec['id']}.pkl")
            restore_capture(env, bl["capture"])
            run_seq(env, table, book48, bl, spec["cpp"])
            desire = -np.asarray(decompose(env.sim)["err_h"], float)
            slides = build_slide_variants(book48, desire, rng)[:12]
            found = None
            for name, slide in slides:
                pref = list(spec["cpp"]) + slide + RETIGHTEN
                restore_capture(env, bl["capture"])
                t0 = float(env.sim.data.time)
                ok, _ = run_seq(env, table, book48, bl, pref)
                if not ok:
                    continue
                ok_r, ret, _, why = return_slow(env, table, book48, bl, t0, max_steps=100)
                tokens = pref + ret + RETIGHTEN
                res = evaluate_full(env, table, book48, bl, tokens, record_obs=True)
                if res["SUCCESS_NEW"]:
                    found = (name, tokens, res)
                    break
            if found:
                name, tokens, res = found
                save_expert(env, spec["id"], f"refine_{name}", res)
                extra_experts.append(pack_cand(f"{spec['id']}_{name}", "extra", tokens, res, {"ic_id": spec["id"]}))
                print(f"  EXTRA {spec['id']} NEW=1", flush=True)
            else:
                # evaluate cpp-only + slow return
                restore_capture(env, bl["capture"])
                t0 = float(env.sim.data.time)
                run_seq(env, table, book48, bl, spec["cpp"] + RETIGHTEN)
                ok_r, ret, _, _ = return_slow(env, table, book48, bl, t0, max_steps=100)
                tokens = list(spec["cpp"]) + RETIGHTEN + ret + RETIGHTEN
                res = evaluate_full(env, table, book48, bl, tokens, record_obs=False)
                print(f"  EXTRA {spec['id']} NEW={res['SUCCESS_NEW']} win={res['window'].get('mean_pos')}", flush=True)
                if res["SUCCESS_NEW"]:
                    res2 = evaluate_full(env, table, book48, bl, tokens, record_obs=True)
                    save_expert(env, spec["id"], "cpp_return", res2)
                    extra_experts.append(pack_cand(f"{spec['id']}_cpp_return", "extra", tokens, res2, {"ic_id": spec["id"]}))

    experts.extend(extra_experts)
    write_csv(OUT / "FULL_EXPERTS.csv", experts)
    _write_report(stages, base_full, cand_rows, experts, h263_full, best)
    print("PILOT_DONE", OUT, "h263_full", h263_full, "n_experts", len(experts), flush=True)


def _better(a, b):
    """Prefer SUCCESS_NEW, then lower win_pos among S12."""
    if a.get("SUCCESS_NEW") and not b.get("SUCCESS_NEW"):
        return True
    if b.get("SUCCESS_NEW"):
        return False
    if a.get("success_12s") and not b.get("success_12s"):
        return True
    if not a.get("success_12s"):
        return False
    pa = a.get("window", {}).get("mean_pos")
    pb = b.get("window", {}).get("mean_pos")
    if pa is None:
        return False
    if pb is None:
        return True
    return float(pa) < float(pb)


def _write_report(stages, base_full, cand_rows, experts, h263_full, best):
    win = base_full["window"]
    pre = base_full.get("pre_handoff") or {}
    # find stages
    by = {s["stage"]: s for s in stages}

    def _sg(stage, key):
        v = (by.get(stage) or {}).get(key)
        if v is None:
            return "—"
        try:
            return f"{float(v):.2f}"
        except Exception:
            return str(v)

    after_cpp_pos = _sg("after_CPP", "e_pos_mm")
    slide_success = any(r.get("SUCCESS_NEW") for r in cand_rows if r.get("family") == "slide")
    cpp_needed = (not slide_success) and any(r.get("SUCCESS_NEW") for r in cand_rows if r.get("family") == "cpp_slide")
    s12 = [r for r in cand_rows if r.get("success_12s") and r.get("win_mean_pos") is not None]
    best_win = min(s12, key=lambda r: float(r["win_mean_pos"])) if s12 else None
    n_other = sum(1 for e in experts if e.get("ic_id") and e.get("ic_id") != "hard_910263")
    n_full = len(experts)

    robust = None
    rob_path = OUT / "RAW" / "robustness.json"
    if rob_path.exists():
        robust = json.loads(rob_path.read_text(encoding="utf-8"))

    lines = [
        "# K49 Slow Return Position Refinement",
        "",
        f"**Mapping:** `{CONFIG_ID}` (tok48 wy=−0.2). Historical wy1p0 preserved elsewhere.",
        "No new tokens. No SFT/GRPO. Locked SUCCESS_NEW: 10° / 12 mm hold.",
        "",
        "```text",
        "python -u scripts/pilot_k49_slow_return_position_refinement.py --device cuda --seed 20261018",
        "```",
        "",
        "## Baseline stages (`hard_910263`)",
        "",
        "| Stage | e_pos (mm) | Axial (mm) | Lateral (mm) | θ (°) | hand↔ref |",
        "|---|---:|---:|---:|---:|---:|",
        f"| after CPP | {_sg('after_CPP','e_pos_mm')} | {_sg('after_CPP','e_ax_mm')} | {_sg('after_CPP','e_lat_mm')} | {_sg('after_CPP','e_theta_deg')} | {_sg('after_CPP','hand_to_ref_deg')} |",
        f"| after Sliding | {_sg('after_Sliding','e_pos_mm')} | {_sg('after_Sliding','e_ax_mm')} | {_sg('after_Sliding','e_lat_mm')} | {_sg('after_Sliding','e_theta_deg')} | {_sg('after_Sliding','hand_to_ref_deg')} |",
        f"| before Return | {_sg('before_Return','e_pos_mm')} | {_sg('before_Return','e_ax_mm')} | {_sg('before_Return','e_lat_mm')} | {_sg('before_Return','e_theta_deg')} | {_sg('before_Return','hand_to_ref_deg')} |",
        f"| after Return | {_sg('after_Return','e_pos_mm')} | {_sg('after_Return','e_ax_mm')} | {_sg('after_Return','e_lat_mm')} | {_sg('after_Return','e_theta_deg')} | {_sg('after_Return','hand_to_ref_deg')} |",
        f"| HANDOFF | {base_full.get('e_pos_at_handoff')} | {pre.get('e_ax_mm')} | {pre.get('e_lat_mm')} | {base_full.get('e_theta_at_handoff')} | {pre.get('hand_to_ref_deg')} |",
        f"| final window | {win.get('mean_pos')} | {base_full.get('win_mean_ax_mm')} | {base_full.get('win_mean_lat_mm')} | {win.get('mean_theta')} | — |",
        "",
        f"**Blocker:** `{_blocker(base_full)}` (hold_both={win.get('longest_both_ok_s')} s).",
        "",
        "## Search",
        "",
        "- Sliding variants: family `slide` in `CANDIDATE_SEARCH.csv`",
        "- Joint CPP+slide: family `cpp_slide`",
        f"- Best candidate win_pos: {None if best_win is None else best_win.get('win_mean_pos')} (`{None if best_win is None else best_win.get('tag')}`)",
        "",
        "## Answers",
        "",
        "| # | Answer |",
        "|---|---|",
        f"| **Q1** | Final-window mean axial ≈ **{base_full.get('win_mean_ax_mm')} mm**, lateral ≈ **{base_full.get('win_mean_lat_mm')} mm** (baseline wy0p25). |",
        f"| **Q2** | {'**Yes** — Sliding candidate achieved SUCCESS_NEW.' if slide_success else '**Not to SUCCESS_NEW** via Sliding alone in this budget.'} |",
        f"| **Q3** | {'**Yes**, CPP joint search contributed the full expert.' if cpp_needed else ('**No** (Sliding alone sufficed).' if slide_success else '**Tried**; no SUCCESS_NEW from pivoting refine alone.')} |",
        f"| **Q4** | **{h263_full}** |",
        f"| **Q5** | {robust} |",
        f"| **Q6** | Additional full experts (other ICs): **{n_other}**; total saved **{n_full}**. |",
        "",
        "## Selection note",
        "",
        f"config_id on all saved experts: `{CONFIG_ID}`.",
        "",
        "## Stop",
        "",
        "Bounded refinement complete. No training.",
        "",
    ]
    # silence unused
    _ = after_cpp_pos
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    write_json(OUT / "summary.json", {
        "config_id": CONFIG_ID,
        "hard_910263_full_expert": h263_full,
        "n_full_experts": n_full,
        "baseline_blocker": _blocker(base_full),
        "baseline_win_pos": win.get("mean_pos"),
        "baseline_win_ax": base_full.get("win_mean_ax_mm"),
        "baseline_win_lat": base_full.get("win_mean_lat_mm"),
        "slide_success": slide_success,
        "cpp_needed": cpp_needed,
        "best_tag": None if best is None else best.get("tag"),
        "best_win_pos": None if best is None else (best.get("window") or {}).get("mean_pos"),
        "robustness": robust,
    })


if __name__ == "__main__":
    main()
