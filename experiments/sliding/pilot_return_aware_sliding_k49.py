"""Return-aware sliding + closed-loop K49 expert refinement (bounded pilot).

Diagnose Return position drift on hard_910263, then compare Strategies A/B/C.
No new tokens. No physics/reward/SUCCESS_12S/checkpoint changes. No SFT/GRPO.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import rot_angle
from data_generation.recording.trajectory_recorder import save_pair
from envs.ballistic_impact import z_tgt_of
from envs.contacts.frames import extract_contacts
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor, recovery_failed
from envs.outcomes.nominal_pose import hand_R
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary, window_stats
from evaluation.grasp_geometry import measure
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import HANDOFF, SECURE_REVERSE_WRIST, build_k49_table, decode_action as decode_k49
from training.k47_codebook import decode_action as decode_k48
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env
from training.recovery_execution import _escaped

OUT = ROOT / "results" / "expert_generation" / "return_aware_sliding_k49"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
SNAP = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910263.pkl"
SEED = 20261018
RECOVERY_LIMIT_S = 4.0

CPP = [4] * 20 + [12] * 5 + [0] * 3
SLIDE_BASE = [17] * 16 + [12] * 5 + [0] * 5
RETIGHTEN = [0, 0, 0]

# Extra ICs after a strategy shows benefit
EXTRA_ICS = [
    {"id": "hrv1_920185", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920185.pkl",
     "prefix": [30, 30] + [10] * 15 + [12] * 5 + [0] * 5},
    {"id": "hrv1_920193", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920193.pkl",
     "prefix": [32] + [6] * 8 + [12] * 5 + [0] * 5},
    {"id": "hard_910277", "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910277.pkl",
     "prefix": [44, 44] + [10] * 8 + [12] * 5 + [0] * 5},
    {"id": "hrv1_920031", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920031.pkl",
     "prefix": [6] * 20 + [12] * 15 + [0] * 8},  # regression control (known full)
    {"id": "hard_910101", "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910101.pkl",
     "prefix": [32] + [10] * 8 + [12] * 15 + [0] * 5},  # regression control
]


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
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def decode(table, book48, token, pre):
    tok = int(token)
    if tok <= 46:
        return decode_k48(book48, tok, pre)
    return decode_k49(table, tok, pre)


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
    e_lat_vec = err - e_ax * u
    e_lat = float(np.linalg.norm(e_lat_vec))
    return {
        "e_theta_deg": m["e_theta_deg"],
        "e_pos_mm": m["e_pos_mm"],
        "e_ax_mm": e_ax * 1e3,
        "e_lat_mm": e_lat * 1e3,
        "axial_frac": abs(e_ax) / max(float(np.linalg.norm(err)), 1e-12),
        "p_c_h": p.tolist(),
        "u_c_h": u.tolist(),
        "err_h": err.tolist(),
        "v_rel_h": m["v_rel_h"],
        "w_rel_h": m["w_rel_h"],
        "bilateral": m["bilateral"],
        "nL": m["nL"],
        "nR": m["nR"],
    }


def rich_snapshot(sim, blob, label=""):
    d = decompose(sim)
    cs_rows, _ = extract_contacts(sim)
    Rh = hand_R(sim)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    # contact summary forces / pad locations
    Fn_L = Ft_L = Fn_R = Ft_R = 0.0
    pos_L, pos_R = [], []
    for r in cs_rows:
        fn = abs(float(r.get("Fn", 0.0)))
        ft = float(r.get("Ft", 0.0))
        if r["side"] == "L":
            Fn_L += fn
            Ft_L += ft
            pos_L.append(r["pos_h"])
        else:
            Fn_R += fn
            Ft_R += ft
            pos_R.append(r["pos_h"])
    cL = np.mean(pos_L, axis=0) if pos_L else [np.nan] * 3
    cR = np.mean(pos_R, axis=0) if pos_R else [np.nan] * 3
    # grip command / actuator
    tau_cmd = float(getattr(sim.fsm, "fg_cmd", np.nan))
    # finger actuator qpos if available
    try:
        qL = float(sim.data.qpos[sim.ids.finger_qposadr[0]]) if hasattr(sim.ids, "finger_qposadr") else float("nan")
        qR = float(sim.data.qpos[sim.ids.finger_qposadr[1]]) if hasattr(sim.ids, "finger_qposadr") else float("nan")
    except Exception:
        qL = qR = float("nan")
    return {
        "label": label,
        "t": float(sim.data.time),
        **d,
        "hand_to_ref_deg": float(np.degrees(rot_angle(Rref, Rh))),
        "rdes_to_ref_deg": float(np.degrees(rot_angle(Rref, Rd))),
        "hand_to_rdes_deg": float(np.degrees(rot_angle(Rd, Rh))),
        "Fn_L": Fn_L, "Ft_L": Ft_L, "Fn_R": Fn_R, "Ft_R": Ft_R,
        "cL_h": list(cL), "cR_h": list(cR),
        "n_contacts": len(cs_rows),
        "tau_cmd": tau_cmd,
        "q_finger_L": qL, "q_finger_R": qR,
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
    }


def run_seq(env, table, book48, blob, tokens, lost_low=0):
    for tok in tokens:
        _r, action = decode(table, book48, tok, np.zeros(7, np.float32))
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return False, lost_low
    return True, lost_low


def evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=False):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    m0 = measure(env.sim, U_TGT, P_TGT)
    lost_low = 0
    observations, actions, contacts, residuals = [], [], [], []
    for tok in tokens:
        if record_obs:
            observations.append(np.asarray(env.observation(), np.float32).reshape(39).copy())
        _r, action = decode(table, book48, tok, np.zeros(7, np.float32))
        action = np.asarray(step_legal(env.sim, action), np.float32)
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
                "e_pos_at_handoff": m["e_pos_mm"], "e_theta_at_handoff": m["e_theta_deg"],
            }
    m_hand = measure(env.sim, U_TGT, P_TGT)
    pre_h = rich_snapshot(env.sim, blob, "pre_handoff")
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
    return {
        "tokens": list(tokens),
        "success_12s": s12,
        "reason": reason,
        "m0": m0,
        "m_handoff": m_hand,
        "m_final": m_final,
        "pre_handoff_rich": pre_h,
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
    }


def return_until(env, table, book48, blob, t0_ic, max_steps=30,
                 tol_hand=8.0, tol_rdes=8.0, tol_track=5.0, record=False):
    used = []
    trace = []
    lost_low = 0
    for i in range(max_steps):
        st = rich_snapshot(env.sim, blob, f"ret_{i}")
        if record:
            trace.append(st)
        if (st["hand_to_ref_deg"] <= tol_hand and st["rdes_to_ref_deg"] <= tol_rdes
                and st["hand_to_rdes_deg"] <= tol_track and st["bilateral"]):
            return True, used, trace, "CONVERGED"
        if float(env.sim.data.time) - t0_ic > RECOVERY_LIMIT_S - 0.05:
            return False, used, trace, "DEADLINE"
        ok, lost_low = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost_low)
        used.append(SECURE_REVERSE_WRIST)
        if not ok:
            return False, used, trace, "DROP"
    st = rich_snapshot(env.sim, blob, "ret_end")
    if record:
        trace.append(st)
    ok = st["hand_to_ref_deg"] <= tol_hand and st["rdes_to_ref_deg"] <= tol_rdes
    return ok, used, trace, "CONVERGED" if ok else "MAX_STEPS"


def classify_drift(trace0, trace1):
    """Classify return drift mechanism from start/end rich snapshots + intermediate."""
    if not trace0 or not trace1:
        return {"class": "UNKNOWN", "notes": "empty trace"}
    a0, a1 = trace0[0], trace1[-1] if isinstance(trace1, list) else trace1
    if isinstance(trace1, list) and len(trace1) > 1:
        a1 = trace1[-1]
    d_ax = float(a1["e_ax_mm"] - a0["e_ax_mm"])
    d_lat = float(a1["e_lat_mm"] - a0["e_lat_mm"])
    d_pos = float(a1["e_pos_mm"] - a0["e_pos_mm"])
    # relative motion magnitudes
    v_mags = [float(np.linalg.norm(s["v_rel_h"])) for s in (trace1 if isinstance(trace1, list) else [a1])]
    w_mags = [float(np.linalg.norm(s["w_rel_h"])) for s in (trace1 if isinstance(trace1, list) else [a1])]
    lost = any(not s["bilateral"] for s in (trace1 if isinstance(trace1, list) else [a1]))
    # contact centroid migration
    def cent(s):
        cL = np.asarray(s["cL_h"], float)
        cR = np.asarray(s["cR_h"], float)
        if np.any(~np.isfinite(cL)) or np.any(~np.isfinite(cR)):
            return None
        return 0.5 * (cL + cR)
    c0, c1 = cent(a0), cent(a1)
    mig = float(np.linalg.norm(c1 - c0)) if c0 is not None and c1 is not None else float("nan")
    # object rotation in hand: change in u_c_h
    u0 = np.asarray(a0["u_c_h"], float)
    u1 = np.asarray(a1["u_c_h"], float)
    du = float(np.degrees(np.arccos(np.clip(abs(np.dot(u0, u1)), 0, 1))))
    notes = []
    classes = []
    if lost:
        classes.append("D")
        notes.append("contact loss during return")
    if abs(d_ax) >= 1.5 and abs(d_ax) >= abs(d_lat) * 0.8:
        classes.append("A")
        notes.append(f"axial Δ={d_ax:.2f}mm dominates")
    if abs(d_lat) >= 1.5 and (abs(d_lat) >= abs(d_ax) * 0.8 or (np.isfinite(mig) and mig > 0.002)):
        classes.append("B")
        notes.append(f"lateral Δ={d_lat:.2f}mm mig={mig*1e3 if np.isfinite(mig) else float('nan'):.2f}mm")
    if du >= 2.0 and max(w_mags) > 0.15:
        classes.append("C")
        notes.append(f"object axis rotate {du:.1f}° in hand")
    if not classes:
        classes.append("E")
        notes.append("mixed/small components")
    label = "+".join(sorted(set(classes))) if len(classes) > 1 else classes[0]
    if len(classes) > 1:
        label = "E(" + "+".join(sorted(set(classes))) + ")"
    return {
        "class": label,
        "d_pos_mm": d_pos,
        "d_ax_mm": d_ax,
        "d_lat_mm": d_lat,
        "contact_centroid_mig_mm": mig * 1e3 if np.isfinite(mig) else None,
        "u_axis_delta_deg": du,
        "max_v_rel": max(v_mags) if v_mags else None,
        "max_w_rel": max(w_mags) if w_mags else None,
        "any_contact_loss": lost,
        "notes": "; ".join(notes),
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
        if grip == "mid" and abs(a6 - (-0.48)) > 0.05:
            continue
        vh = np.array([a[0], a[1], (a[2] * 3.8 if a[2] >= 0 else a[2])], float)
        speed = float(np.linalg.norm(vh))
        if speed < 0.02:
            continue
        # reject wrist rotation for pure slide
        if abs(float(a[3])) > 0.05 or abs(float(a[4])) > 0.05 or abs(float(a[5])) > 0.05:
            continue
        align = float(np.dot(vh / speed, d))
        ranked.append((align * speed, i, align, speed))
    ranked.sort(key=lambda x: -x[0])
    return ranked[:top_k]


def slide_candidates(book, desire_h, rng, max_n=40):
    sec = [t for _, t, *_ in tokens_aligned(book, desire_h, "secure", 8)]
    opp = [t for _, t, *_ in tokens_aligned(book, -np.asarray(desire_h, float), "secure", 4)]
    seqs = []
    for t in sec[:6]:
        for n in (8, 12, 16, 20, 24, 28):
            seqs.append([12] * 5 + [t] * n + [0] * 5)
            seqs.append([t] * n + [12] * 5 + [0] * 5)
            seqs.append([12] * 3 + [t, 12] * (n // 2) + [0] * 5)
    # known baseline
    seqs.append(list(SLIDE_BASE))
    # overcompensation: longer / stronger
    for t in sec[:4]:
        seqs.append([12] * 5 + [t] * 32 + [0] * 8)
        seqs.append([t] * 32 + [0] * 8)
    for t in (33, 17, 13, 19, 1, 5, 24):
        for n in (10, 16, 22, 28):
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
    rng.shuffle(uniq)
    return uniq[:max_n]


def pack_row(strategy, tag, tokens, res, extra=None):
    row = {
        "strategy": strategy,
        "tag": tag,
        "s12": int(res["success_12s"]),
        "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
        "label": res["label"],
        "reason": res["reason"],
        "win_mean_theta": res["window"].get("mean_theta"),
        "win_mean_pos": res["window"].get("mean_pos"),
        "hold_both_s": res["window"].get("longest_both_ok_s"),
        "e_pos_at_handoff": res.get("e_pos_at_handoff"),
        "e_theta_at_handoff": res.get("e_theta_at_handoff"),
        "hand_to_ref_at_handoff": (res.get("pre_handoff_rich") or {}).get("hand_to_ref_deg"),
        "e_ax_at_handoff": (res.get("pre_handoff_rich") or {}).get("e_ax_mm"),
        "e_lat_at_handoff": (res.get("pre_handoff_rich") or {}).get("e_lat_mm"),
        "recovery_s": res["recovery_elapsed"],
        "n_tokens": len(tokens),
        "tokens": json.dumps(tokens),
        "uses_48": int(SECURE_REVERSE_WRIST in tokens),
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
        "ic_id": ic_id, "source": source, "vocab": "K49",
        "SUCCESS_NEW": res["SUCCESS_NEW"], "label": res["label"],
        "window": res["window"], "tokens": res["tokens"],
    })


def video(env, table, book48, blob, tokens, dest):
    try:
        restore_capture(env, blob["capture"])
        frames = []
        renderer = mujoco.Renderer(env.sim.model, 480, 480)

        def grab():
            mujoco.mj_forward(env.sim.model, env.sim.data)
            try:
                renderer.update_scene(env.sim.data, camera="side")
            except Exception:
                renderer.update_scene(env.sim.data)
            frames.append(renderer.render().copy())

        grab()
        for tok in tokens:
            _r, action = decode(table, book48, tok, np.zeros(7, np.float32))
            step_legal(env.sim, action)
            grab()
        apply_handoff(env.sim, blob["ref"])
        for i in range(40):
            nominal_step(env.sim)
            if i % 2 == 0:
                grab()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.mimsave(str(dest), frames[::2], fps=10)
        except Exception:
            np.savez_compressed(dest.with_suffix(".npz"), frames=np.stack(frames[::4]))
    except Exception as exc:
        (dest.parent / "VIDEO_ERROR.txt").write_text(str(exc), encoding="utf-8")


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def diagnose_return(env, table, book48, blob):
    """Substep-resolution Return drift diagnosis on hard_910263."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    ok, _ = run_seq(env, table, book48, blob, CPP + SLIDE_BASE + RETIGHTEN)
    assert ok
    before = rich_snapshot(env.sim, blob, "before_return")
    # step tok48 one-by-one with rich snapshots
    trace = [before]
    lost_low = 0
    for i in range(25):
        ok, lost_low = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost_low)
        st = rich_snapshot(env.sim, blob, f"tok48_{i+1}")
        trace.append(st)
        if not ok:
            break
        if (st["hand_to_ref_deg"] <= 8 and st["rdes_to_ref_deg"] <= 8
                and st["hand_to_rdes_deg"] <= 5 and st["bilateral"]):
            break
    after = trace[-1]
    mech = classify_drift(trace[:1], trace)
    # per-step deltas
    rows = []
    for i in range(1, len(trace)):
        a, b = trace[i - 1], trace[i]
        rows.append({
            "step": i,
            "e_pos_mm": b["e_pos_mm"],
            "d_e_pos": b["e_pos_mm"] - a["e_pos_mm"],
            "e_ax_mm": b["e_ax_mm"],
            "d_e_ax": b["e_ax_mm"] - a["e_ax_mm"],
            "e_lat_mm": b["e_lat_mm"],
            "d_e_lat": b["e_lat_mm"] - a["e_lat_mm"],
            "hand_to_ref": b["hand_to_ref_deg"],
            "rdes_to_ref": b["rdes_to_ref_deg"],
            "v_rel": float(np.linalg.norm(b["v_rel_h"])),
            "w_rel": float(np.linalg.norm(b["w_rel_h"])),
            "bilateral": int(b["bilateral"]),
            "Fn_L": b["Fn_L"], "Fn_R": b["Fn_R"],
            "Ft_L": b["Ft_L"], "Ft_R": b["Ft_R"],
            "n_contacts": b["n_contacts"],
        })
    write_csv(OUT / "RAW" / "return_drift_trace.csv", rows)
    write_json(OUT / "RAW" / "return_drift_before_after.json", {"before": before, "after": after, "mechanism": mech})
    write_json(OUT / "RAW" / "return_drift_full_trace.json", {"trace": trace, "mechanism": mech})
    return before, after, mech, trace


def strategy_A(env, table, book48, blob, rng, budget=36):
    """Sliding overcompensation then full Return."""
    restore_capture(env, blob["capture"])
    run_seq(env, table, book48, blob, CPP)
    desire = -(np.asarray(decompose(env.sim)["err_h"], float))
    cands = slide_candidates(book48, desire, rng, max_n=budget)
    rows = []
    best = None
    for i, slide in enumerate(cands):
        seq_prefix = CPP + slide + RETIGHTEN
        restore_capture(env, blob["capture"])
        t0 = float(env.sim.data.time)
        ok, _ = run_seq(env, table, book48, blob, seq_prefix)
        if not ok:
            rows.append({"strategy": "A", "tag": f"slide_{i}", "ok_prefix": 0, "SUCCESS_NEW": 0,
                         "label": "DROP_PREFIX", "e_pos_post_slide": None})
            continue
        post_slide = rich_snapshot(env.sim, blob, "post_slide")
        ok_r, ret_toks, _, why = return_until(env, table, book48, blob, t0, max_steps=28)
        post_ret = rich_snapshot(env.sim, blob, "post_ret")
        tokens = seq_prefix + ret_toks
        res = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=False)
        row = pack_row("A", f"slide_{i}", tokens, res, {
            "ok_return": int(ok_r), "return_why": why,
            "e_pos_post_slide": post_slide["e_pos_mm"],
            "e_ax_post_slide": post_slide["e_ax_mm"],
            "e_lat_post_slide": post_slide["e_lat_mm"],
            "e_pos_post_return": post_ret["e_pos_mm"],
            "drift_mm": post_ret["e_pos_mm"] - post_slide["e_pos_mm"],
            "slide_tokens": json.dumps(slide),
        })
        rows.append(row)
        print(f"  A[{i}] slide_pos={post_slide['e_pos_mm']:.2f} ret_pos={post_ret['e_pos_mm']:.2f} "
              f"NEW={res['SUCCESS_NEW']} win_pos={res['window'].get('mean_pos')}", flush=True)
        if res["SUCCESS_NEW"] and (best is None or res["window"].get("mean_pos", 99) < best["window"].get("mean_pos", 99)):
            best = {**res, "tokens": tokens, "tag": f"A_slide_{i}"}
        elif best is None or (
            res["success_12s"] and (res.get("e_pos_at_handoff") or 99) < (best.get("e_pos_at_handoff") or 99)
            and not best.get("SUCCESS_NEW")
        ):
            if not (best and best.get("SUCCESS_NEW")):
                best = {**res, "tokens": tokens, "tag": f"A_slide_{i}"}
    return rows, best


def strategy_B(env, table, book48, blob, rng, budget=28):
    """Segmented Return with intermediate legal position correction."""
    # after CPP+baseline slide+retighten, return in chunks with optional correction
    desire_tokens = [17, 33, 13, 19, 1, 5, 24, 16]
    chunk_lens = [4, 6, 8, 10]
    corr_lens = [0, 4, 6, 8, 10]
    rows = []
    best = None
    plans = []
    for cl in chunk_lens:
        for n_chunks in (1, 2, 3):
            for corr_n in corr_lens:
                for ct in desire_tokens[:5]:
                    plans.append((cl, n_chunks, corr_n, ct))
    rng.shuffle(plans)
    plans = plans[:budget]
    # also: measure desire after each chunk dynamically (closed-loop) — handled inside
    for i, (cl, n_chunks, corr_n, ct) in enumerate(plans):
        restore_capture(env, blob["capture"])
        t0 = float(env.sim.data.time)
        prefix = CPP + SLIDE_BASE + RETIGHTEN
        ok, lost = run_seq(env, table, book48, blob, prefix)
        if not ok:
            rows.append({"strategy": "B", "tag": f"seg_{i}", "SUCCESS_NEW": 0, "label": "DROP_PREFIX"})
            continue
        used = list(prefix)
        dropped = False
        for ch in range(n_chunks):
            # return chunk
            for _ in range(cl):
                st = rich_snapshot(env.sim, blob, "")
                if st["hand_to_ref_deg"] <= 8 and st["rdes_to_ref_deg"] <= 8 and st["hand_to_rdes_deg"] <= 5:
                    break
                if float(env.sim.data.time) - t0 > RECOVERY_LIMIT_S - 0.08:
                    break
                ok, lost = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost)
                used.append(SECURE_REVERSE_WRIST)
                if not ok:
                    dropped = True
                    break
            if dropped:
                break
            # intermediate correction if e_pos rising / still high
            st = rich_snapshot(env.sim, blob, "")
            if corr_n > 0 and st["e_pos_mm"] > 10.0:
                # closed-loop: pick translate aligned to -err
                err = np.asarray(st["err_h"], float)
                ranked = tokens_aligned(book48, -err, "secure", 4)
                tok = ranked[0][1] if ranked else ct
                # unload briefly then translate then retighten
                corr = [12] * 2 + [tok] * corr_n + [0] * 3
                ok, lost = run_seq(env, table, book48, blob, corr, lost)
                used.extend(corr)
                if not ok:
                    dropped = True
                    break
        if dropped:
            rows.append(pack_row("B", f"seg_{i}", used, {
                "success_12s": False, "SUCCESS_NEW": False, "label": "FAILED_OR_DROPPED",
                "reason": "DROP", "window": window_stats([], float(NOMINAL_S) - 2.0),
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "e_pos_at_handoff": None, "e_theta_at_handoff": None, "pre_handoff_rich": {},
            }, {"plan": f"cl{cl}_n{n_chunks}_corr{corr_n}_ct{ct}"}))
            continue
        # finish return
        ok_r, ret_toks, _, why = return_until(env, table, book48, blob, t0, max_steps=20)
        used.extend(ret_toks)
        # stabilize
        ok, lost = run_seq(env, table, book48, blob, RETIGHTEN, lost)
        used.extend(RETIGHTEN)
        res = evaluate_tokens_vocab(env, table, book48, blob, used, record_obs=False)
        row = pack_row("B", f"seg_{i}", used, res, {
            "ok_return": int(ok_r), "return_why": why,
            "plan": f"cl{cl}_n{n_chunks}_corr{corr_n}_ct{ct}",
        })
        rows.append(row)
        print(f"  B[{i}] plan={row['plan']} handoff_pos={res.get('e_pos_at_handoff')} "
              f"NEW={res['SUCCESS_NEW']} win_pos={res['window'].get('mean_pos')}", flush=True)
        if res["SUCCESS_NEW"] and (best is None or res["window"].get("mean_pos", 99) < best["window"].get("mean_pos", 99)):
            best = {**res, "tokens": used, "tag": f"B_seg_{i}"}
        elif best is None or (
            res["success_12s"] and not (best and best.get("SUCCESS_NEW"))
            and (res.get("e_pos_at_handoff") or 99) < (best.get("e_pos_at_handoff") or 99)
        ):
            if not (best and best.get("SUCCESS_NEW")):
                best = {**res, "tokens": used, "tag": f"B_seg_{i}"}
    return rows, best


def strategy_C(env, table, book48, blob, rng, budget=24):
    """Return (partial/full) before final Sliding."""
    rows = []
    best = None
    return_ns = [8, 12, 16, 19, 22]
    slide_ns = [8, 12, 16, 20, 24]
    slide_toks = [17, 33, 13, 19, 1, 5]
    plans = []
    for rn in return_ns:
        for sn in slide_ns:
            for st in slide_toks:
                plans.append((rn, sn, st, False))  # no second return
                plans.append((rn, sn, st, True))   # second return after slide
    rng.shuffle(plans)
    plans = plans[:budget]
    for i, (rn, sn, st, second_ret) in enumerate(plans):
        restore_capture(env, blob["capture"])
        t0 = float(env.sim.data.time)
        used = list(CPP) + RETIGHTEN
        ok, lost = run_seq(env, table, book48, blob, used)
        if not ok:
            rows.append({"strategy": "C", "tag": f"rbs_{i}", "SUCCESS_NEW": 0, "label": "DROP_CPP"})
            continue
        # partial/full return first
        for _ in range(rn):
            st_snap = rich_snapshot(env.sim, blob, "")
            if st_snap["hand_to_ref_deg"] <= 8 and st_snap["rdes_to_ref_deg"] <= 8:
                break
            if float(env.sim.data.time) - t0 > RECOVERY_LIMIT_S - 0.1:
                break
            ok, lost = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost)
            used.append(SECURE_REVERSE_WRIST)
            if not ok:
                break
        if not ok:
            rows.append({"strategy": "C", "tag": f"rbs_{i}", "SUCCESS_NEW": 0, "label": "DROP_RETURN1"})
            continue
        mid = rich_snapshot(env.sim, blob, "mid")
        # slide at (possibly) more nominal wrist
        err = np.asarray(mid["err_h"], float)
        ranked = tokens_aligned(book48, -err, "secure", 3)
        tok = ranked[0][1] if ranked else st
        slide = [12] * 3 + [tok] * sn + [0] * 4
        ok, lost = run_seq(env, table, book48, blob, slide, lost)
        used.extend(slide)
        if not ok:
            rows.append({"strategy": "C", "tag": f"rbs_{i}", "SUCCESS_NEW": 0, "label": "DROP_SLIDE"})
            continue
        if second_ret:
            ok_r, ret_toks, _, why = return_until(env, table, book48, blob, t0, max_steps=18)
            used.extend(ret_toks)
        else:
            ok_r, why = True, "SKIP"
        used.extend(RETIGHTEN)
        run_seq(env, table, book48, blob, RETIGHTEN, lost)
        res = evaluate_tokens_vocab(env, table, book48, blob, used, record_obs=False)
        row = pack_row("C", f"rbs_{i}", used, res, {
            "ok_return": int(ok_r), "return_why": why,
            "plan": f"rn{rn}_sn{sn}_st{tok}_2nd{int(second_ret)}",
            "hand_mid": mid["hand_to_ref_deg"],
            "e_pos_mid": mid["e_pos_mm"],
        })
        rows.append(row)
        print(f"  C[{i}] plan={row['plan']} handoff_pos={res.get('e_pos_at_handoff')} "
              f"NEW={res['SUCCESS_NEW']} win_pos={res['window'].get('mean_pos')}", flush=True)
        if res["SUCCESS_NEW"] and (best is None or res["window"].get("mean_pos", 99) < best["window"].get("mean_pos", 99)):
            best = {**res, "tokens": used, "tag": f"C_rbs_{i}"}
        elif best is None or (
            res["success_12s"] and not (best and best.get("SUCCESS_NEW"))
            and (res.get("e_pos_at_handoff") or 99) < (best.get("e_pos_at_handoff") or 99)
        ):
            if not (best and best.get("SUCCESS_NEW")):
                best = {**res, "tokens": used, "tag": f"C_rbs_{i}"}
    return rows, best


def closed_loop_refine(env, table, book48, blob, seed_tokens, rng, horizon=6, iters=12):
    """Receding-horizon local refinement around a seed sequence using legal K49 tokens."""
    motion = list(range(47)) + [SECURE_REVERSE_WRIST]
    best_tokens = list(seed_tokens)
    best_res = evaluate_tokens_vocab(env, table, book48, blob, best_tokens, record_obs=False)
    best_score = _guide(best_res)
    rows = []
    for it in range(iters):
        cand = list(best_tokens)
        if len(cand) < 4:
            break
        # mutate a short window near the end (before stabilize)
        end = max(0, len(cand) - 3)
        start = max(0, end - horizon)
        for j in range(start, end):
            if rng.random() < 0.45:
                cand[j] = int(rng.choice(motion))
        # optionally append short correction+return
        if rng.random() < 0.35:
            cand = cand + [12, int(rng.choice([17, 33, 13, 1])), SECURE_REVERSE_WRIST, 0]
        res = evaluate_tokens_vocab(env, table, book48, blob, cand, record_obs=False)
        sc = _guide(res)
        rows.append(pack_row("CL", f"it_{it}", cand, res, {"guide": sc}))
        print(f"  CL[{it}] guide={sc:.3f} NEW={res['SUCCESS_NEW']} "
              f"win_pos={res['window'].get('mean_pos')} handoff={res.get('e_pos_at_handoff')}", flush=True)
        if sc > best_score or (res["SUCCESS_NEW"] and not best_res.get("SUCCESS_NEW")):
            best_score = sc
            best_tokens = cand
            best_res = {**res, "tokens": cand, "tag": f"CL_it_{it}"}
            if res["SUCCESS_NEW"]:
                break
    return rows, best_res if best_res.get("SUCCESS_NEW") or best_score > _guide(
        evaluate_tokens_vocab(env, table, book48, blob, seed_tokens)
    ) else None


def _guide(res):
    """Privileged ranking: prefer SUCCESS_NEW, then low handoff pos with low hand-ref error."""
    if res.get("SUCCESS_NEW"):
        return 1000.0 - float(res["window"].get("mean_pos") or 99)
    if not res.get("success_12s"):
        return -100.0
    pos = float(res.get("e_pos_at_handoff") or 99)
    th = float(res.get("e_theta_at_handoff") or 99)
    hand = float((res.get("pre_handoff_rich") or {}).get("hand_to_ref_deg") or 90)
    # reward being under thresholds with nominal wrist
    return 50.0 - pos - 0.05 * th - 0.1 * hand


def run_extra_ics(env, table, book48, best_strategy_fn, rng):
    """Apply best approach pattern to a few more ICs + regression controls."""
    experts = []
    failed = []
    for spec in EXTRA_ICS:
        path = ROOT / spec["path"]
        if not path.exists():
            continue
        blob = pickle.loads(path.read_bytes())
        shutil.copy2(path, OUT / "SNAPSHOTS" / f"{spec['id']}.pkl")
        # baseline prefix → return → HANDOFF
        restore_capture(env, blob["capture"])
        t0 = float(env.sim.data.time)
        prefix = list(spec["prefix"])
        ok, lost = run_seq(env, table, book48, blob, prefix)
        if not ok:
            failed.append({"ic_id": spec["id"], "tag": "prefix_drop", "SUCCESS_NEW": 0})
            continue
        # measure if return needed
        st = rich_snapshot(env.sim, blob, "")
        used = list(prefix)
        if st["hand_to_ref_deg"] > 10:
            ok_r, ret_toks, _, why = return_until(env, table, book48, blob, t0, max_steps=25)
            used.extend(ret_toks)
            # if pos high after return, try short closed-loop correction+re-return
            st2 = rich_snapshot(env.sim, blob, "")
            if st2["e_pos_mm"] > 11.5:
                err = np.asarray(st2["err_h"], float)
                ranked = tokens_aligned(book48, -err, "secure", 3)
                if ranked:
                    corr = [12] * 2 + [ranked[0][1]] * 8 + [0] * 3
                    ok, lost = run_seq(env, table, book48, blob, corr, lost)
                    used.extend(corr)
                    ok_r2, ret2, _, _ = return_until(env, table, book48, blob, t0, max_steps=15)
                    used.extend(ret2)
        used.extend(RETIGHTEN)
        res = evaluate_tokens_vocab(env, table, book48, blob, used, record_obs=True)
        row = {"ic_id": spec["id"], **pack_row("EXTRA", "return_aware", used, res)}
        print(f"  EXTRA {spec['id']} NEW={res['SUCCESS_NEW']} win_pos={res['window'].get('mean_pos')} "
              f"handoff={res.get('e_pos_at_handoff')}", flush=True)
        if res["SUCCESS_NEW"]:
            save_expert(env, spec["id"], "return_aware", res)
            experts.append(row)
        else:
            failed.append(row)
    # Second pass for failed extras: A-style overcompensation
    for spec in EXTRA_ICS:
        path = ROOT / spec["path"]
        if not path.exists():
            continue
        if any(e["ic_id"] == spec["id"] for e in experts):
            continue
        blob = pickle.loads(path.read_bytes())
        restore_capture(env, blob["capture"])
        run_seq(env, table, book48, blob, spec["prefix"])
        desire = -np.asarray(decompose(env.sim)["err_h"], float)
        cands = slide_candidates(book48, desire, rng, max_n=10)
        for j, slide in enumerate(cands[:8]):
            tokens = list(spec["prefix"]) + slide + RETIGHTEN
            restore_capture(env, blob["capture"])
            t0 = float(env.sim.data.time)
            ok, _ = run_seq(env, table, book48, blob, tokens)
            if not ok:
                continue
            ok_r, ret_toks, _, _ = return_until(env, table, book48, blob, t0, max_steps=25)
            tokens = tokens + ret_toks
            res = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=True)
            print(f"  EXTRA-A {spec['id']}[{j}] NEW={res['SUCCESS_NEW']} "
                  f"win_pos={res['window'].get('mean_pos')}", flush=True)
            if res["SUCCESS_NEW"]:
                save_expert(env, spec["id"], f"A_extra_{j}", res)
                experts.append({"ic_id": spec["id"], **pack_row("EXTRA_A", f"slide_{j}", tokens, res)})
                break
            failed.append({"ic_id": spec["id"], **pack_row("EXTRA_A", f"slide_{j}", tokens, res)})
    return experts, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--budget-a", type=int, default=36)
    parser.add_argument("--budget-b", type=int, default=28)
    parser.add_argument("--budget-c", type=int, default=24)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-extra", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("RAW", "SNAPSHOTS", "VIDEOS", "trajectories"):
        (OUT / sub).mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48)
    assert np.allclose(table[:47], book48)
    assert abs(float(table[48, 4]) + float(book48[4, 4])) < 1e-5

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    blob = pickle.loads(SNAP.read_bytes())
    shutil.copy2(SNAP, OUT / "SNAPSHOTS" / "hard_910263.pkl")
    np.save(OUT / "codebook_k49_table.npy", table)

    # --- 1. Diagnose ---
    print("DIAGNOSE_RETURN_DRIFT", flush=True)
    before, after, mech, trace = diagnose_return(env, table, book48, blob)
    print(f"  before_pos={before['e_pos_mm']:.2f} ax={before['e_ax_mm']:.2f} lat={before['e_lat_mm']:.2f}", flush=True)
    print(f"  after_pos={after['e_pos_mm']:.2f} ax={after['e_ax_mm']:.2f} lat={after['e_lat_mm']:.2f}", flush=True)
    print(f"  mechanism={mech}", flush=True)

    all_rows = []
    experts = []
    failed = []
    best_global = None

    # Baseline: CPP+slide→HANDOFF and CPP+slide+return→HANDOFF
    print("BASELINES", flush=True)
    for tag, toks in [
        ("base_slide_handoff", CPP + SLIDE_BASE),
        ("base_slide_return", None),
    ]:
        if tag == "base_slide_return":
            restore_capture(env, blob["capture"])
            t0 = float(env.sim.data.time)
            run_seq(env, table, book48, blob, CPP + SLIDE_BASE + RETIGHTEN)
            ok_r, ret_toks, _, why = return_until(env, table, book48, blob, t0, max_steps=28)
            toks = CPP + SLIDE_BASE + RETIGHTEN + ret_toks
        res = evaluate_tokens_vocab(env, table, book48, blob, toks, record_obs=False)
        row = pack_row("BASE", tag, toks, res)
        all_rows.append(row)
        print(f"  {tag} NEW={res['SUCCESS_NEW']} win_pos={res['window'].get('mean_pos')} "
              f"handoff={res.get('e_pos_at_handoff')}", flush=True)

    # --- Strategy A ---
    print("STRATEGY_A", flush=True)
    rows_a, best_a = strategy_A(env, table, book48, blob, rng, budget=args.budget_a)
    all_rows.extend(rows_a)
    if best_a and best_a.get("SUCCESS_NEW"):
        best_global = best_a
        print(f"  A FOUND FULL: {best_a.get('tag')}", flush=True)

    # --- Strategy B ---
    print("STRATEGY_B", flush=True)
    rows_b, best_b = strategy_B(env, table, book48, blob, rng, budget=args.budget_b)
    all_rows.extend(rows_b)
    if best_b and best_b.get("SUCCESS_NEW"):
        if best_global is None or (best_b["window"].get("mean_pos") or 99) < (best_global["window"].get("mean_pos") or 99):
            best_global = best_b
        print(f"  B FOUND FULL: {best_b.get('tag')}", flush=True)

    # --- Strategy C ---
    print("STRATEGY_C", flush=True)
    rows_c, best_c = strategy_C(env, table, book48, blob, rng, budget=args.budget_c)
    all_rows.extend(rows_c)
    if best_c and best_c.get("SUCCESS_NEW"):
        if best_global is None or (best_c["window"].get("mean_pos") or 99) < (best_global["window"].get("mean_pos") or 99):
            best_global = best_c
        print(f"  C FOUND FULL: {best_c.get('tag')}", flush=True)

    # Seed for closed-loop: best non-full by handoff pos among successful S12
    seed = None
    for b in (best_a, best_b, best_c):
        if b is None:
            continue
        if seed is None or _guide(b) > _guide(seed):
            seed = b
    if seed is None:
        seed = {"tokens": CPP + SLIDE_BASE + RETIGHTEN + [SECURE_REVERSE_WRIST] * 19}

    print("CLOSED_LOOP", flush=True)
    rows_cl, best_cl = closed_loop_refine(env, table, book48, blob, seed["tokens"], rng, horizon=6, iters=14)
    all_rows.extend(rows_cl)
    if best_cl and best_cl.get("SUCCESS_NEW"):
        best_global = best_cl
        print(f"  CL FOUND FULL: {best_cl.get('tag')}", flush=True)

    # If we have a full expert for hard_910263, record it
    h263_full = False
    if best_global and best_global.get("SUCCESS_NEW"):
        res_rec = evaluate_tokens_vocab(env, table, book48, blob, best_global["tokens"], record_obs=True)
        if res_rec["SUCCESS_NEW"]:
            h263_full = True
            save_expert(env, "hard_910263", best_global.get("tag", "best"), res_rec)
            experts.append({"ic_id": "hard_910263", **pack_row(
                best_global.get("tag", "best").split("_")[0], best_global.get("tag", "best"),
                best_global["tokens"], res_rec)})
            if not args.skip_videos:
                video(env, table, book48, blob, best_global["tokens"],
                      OUT / "VIDEOS" / "hard_910263_full_expert.mp4")

    # Collect failed candidates (S12 but not NEW, or drops) — keep top by guide
    for r in all_rows:
        if r.get("SUCCESS_NEW"):
            continue
        failed.append(r)

    # Extra ICs if any strategy showed benefit (lower handoff pos after return vs baseline ~15mm)
    benefit = False
    for r in all_rows:
        if r.get("strategy") in ("A", "B", "C", "CL") and r.get("success_12s" if "success_12s" in r else "s12"):
            ep = r.get("e_pos_at_handoff")
            if ep is not None and float(ep) <= 12.5:
                benefit = True
                break
    if h263_full:
        benefit = True

    extra_experts, extra_failed = [], []
    if not args.skip_extra and benefit:
        print("EXTRA_ICS", flush=True)
        extra_experts, extra_failed = run_extra_ics(env, table, book48, None, rng)
        experts.extend(extra_experts)
        failed.extend(extra_failed)
    elif not args.skip_extra:
        print("EXTRA_ICS_SKIPPED_NO_CLEAR_BENEFIT_STILL_RUNNING_CONTROLS", flush=True)
        # still run regression controls with return-aware pattern
        extra_experts, extra_failed = run_extra_ics(env, table, book48, None, rng)
        experts.extend(extra_experts)
        failed.extend(extra_failed)

    write_csv(OUT / "STRATEGY_COMPARISON.csv", all_rows)
    write_csv(OUT / "FULL_EXPERTS.csv", experts)
    # trim failed to avoid huge files
    failed_sorted = sorted(failed, key=lambda r: (
        0 if r.get("s12") or r.get("success_12s") else 1,
        float(r.get("e_pos_at_handoff") or r.get("win_mean_pos") or 999),
    ))
    write_csv(OUT / "FAILED_CANDIDATES.csv", failed_sorted[:200])
    write_json(OUT / "RAW" / "all_strategy_rows.json", all_rows)
    write_json(OUT / "RAW" / "mechanism.json", mech)

    # Write reports
    _write_reports(before, after, mech, all_rows, experts, h263_full, benefit)

    if not args.skip_videos:
        video(env, table, book48, blob, CPP + SLIDE_BASE,
              OUT / "VIDEOS" / "hard_910263_baseline_slide.mp4")
        base_ret = CPP + SLIDE_BASE + RETIGHTEN + [SECURE_REVERSE_WRIST] * 19
        video(env, table, book48, blob, base_ret,
              OUT / "VIDEOS" / "hard_910263_baseline_return.mp4")

    print("PILOT_DONE", OUT, "h263_full", h263_full, "n_experts", len(experts), flush=True)


def _write_reports(before, after, mech, all_rows, experts, h263_full, benefit):
    # RETURN_DRIFT_MECHANISM.md
    (OUT / "RETURN_DRIFT_MECHANISM.md").write_text(f"""# Return Drift Mechanism — hard_910263

## Setup

Start: CPP `[4]×20+[12]×5+[0]×3` + Sliding `{SLIDE_BASE}` + retighten `[0]×3`.
Return: K49 token **48** (secure −wy) until hand/r_des near nominal.

## Measured endpoints

| Stage | e_pos (mm) | Axial (mm) | Lateral (mm) | hand↔ref (°) | bilateral |
|---|---:|---:|---:|---:|---|
| Before Return | {before['e_pos_mm']:.2f} | {before['e_ax_mm']:.2f} | {before['e_lat_mm']:.2f} | {before['hand_to_ref_deg']:.2f} | {before['bilateral']} |
| After Return | {after['e_pos_mm']:.2f} | {after['e_ax_mm']:.2f} | {after['e_lat_mm']:.2f} | {after['hand_to_ref_deg']:.2f} | {after['bilateral']} |
| Δ | {after['e_pos_mm']-before['e_pos_mm']:.2f} | {after['e_ax_mm']-before['e_ax_mm']:.2f} | {after['e_lat_mm']-before['e_lat_mm']:.2f} | {after['hand_to_ref_deg']-before['hand_to_ref_deg']:.2f} | |

## Classification

- **Label:** `{mech['class']}`
- **Notes:** {mech['notes']}
- Contact-centroid migration: {mech.get('contact_centroid_mig_mm')} mm
- Cylinder-axis Δ in hand: {mech.get('u_axis_delta_deg'):.2f}°
- Any contact loss: {mech.get('any_contact_loss')}
- Max ‖v_rel‖ / ‖w_rel‖: {mech.get('max_v_rel')} / {mech.get('max_w_rel')}

## Interpretation

Codes: A=axial sliding, B=lateral contact migration, C=object rotation about contact, D=contact loss/recapture, E=combination.

See `RAW/return_drift_trace.csv` for per-token-step axial/lateral/contact force series.
""", encoding="utf-8")

    def best_of(strat):
        cand = [r for r in all_rows if r.get("strategy") == strat and r.get("SUCCESS_NEW")]
        if cand:
            return min(cand, key=lambda r: float(r.get("win_mean_pos") or 99))
        cand = [r for r in all_rows if r.get("strategy") == strat and (r.get("s12") or r.get("success_12s"))]
        if not cand:
            return None
        return min(cand, key=lambda r: float(r.get("e_pos_at_handoff") or r.get("win_mean_pos") or 99))

    ba, bb, bc, bcl = best_of("A"), best_of("B"), best_of("C"), best_of("CL")
    n_full = len(experts)
    a_survives = bool(ba and ba.get("SUCCESS_NEW"))
    b_ok = bool(bb and bb.get("SUCCESS_NEW"))
    c_ok = bool(bc and bc.get("SUCCESS_NEW"))

    (OUT / "REPORT.md").write_text(f"""# Return-Aware Sliding K49 — Pilot Report

**No new tokens. K49 token 48 unchanged. HANDOFF=47. No SFT/GRPO.**

```text
python -u scripts/pilot_return_aware_sliding_k49.py --device cuda --seed 20261018
```

## Drift diagnosis (Q1)

Before→after uninterrupted tok48 return on `hard_910263`:
**{before['e_pos_mm']:.2f} → {after['e_pos_mm']:.2f} mm** (Δ {after['e_pos_mm']-before['e_pos_mm']:+.2f}).
Mechanism class **`{mech['class']}`**: {mech['notes']}.

Details: `RETURN_DRIFT_MECHANISM.md`, `RAW/return_drift_trace.csv`.

## Strategy comparison (hard_910263)

| Strategy | Best SUCCESS_NEW? | Best handoff e_pos | Best window pos | Notes |
|---|---|---:|---:|---|
| A overcompensation | {bool(a_survives)} | {_fmt(ba, 'e_pos_at_handoff')} | {_fmt(ba, 'win_mean_pos')} | {_tag(ba)} |
| B segmented return | {bool(b_ok)} | {_fmt(bb, 'e_pos_at_handoff')} | {_fmt(bb, 'win_mean_pos')} | {_tag(bb)} |
| C return-before-slide | {bool(c_ok)} | {_fmt(bc, 'e_pos_at_handoff')} | {_fmt(bc, 'win_mean_pos')} | {_tag(bc)} |
| CL closed-loop | {bool(bcl and bcl.get('SUCCESS_NEW'))} | {_fmt(bcl, 'e_pos_at_handoff')} | {_fmt(bcl, 'win_mean_pos')} | {_tag(bcl)} |

Full table: `STRATEGY_COMPARISON.csv`.

## hard_910263 VERIFIED_FULL_EXPERT?

**{h263_full}**

## Full experts this pilot

n = **{n_full}**

See `FULL_EXPERTS.csv`, `trajectories/`.

## Answers (Q1–Q7)

| # | Answer |
|---|---|
| **Q1** | `{mech['class']}` — {mech['notes']} (Δpos={mech.get('d_pos_mm'):.2f} mm; Δax={mech.get('d_ax_mm'):.2f}; Δlat={mech.get('d_lat_mm'):.2f}). |
| **Q2** | {"Yes" if a_survives else "No clear full-objective survival in this budget"} — best A handoff pos {_fmt(ba, 'e_pos_at_handoff')} mm. |
| **Q3** | {"Yes" if b_ok else "Not to SUCCESS_NEW in this budget"} — best B handoff {_fmt(bb, 'e_pos_at_handoff')} mm. |
| **Q4** | {"Yes" if c_ok else "Not to SUCCESS_NEW in this budget"} — best C handoff {_fmt(bc, 'e_pos_at_handoff')} mm. |
| **Q5** | **{h263_full}** |
| **Q6** | **{n_full}** full-objective experts saved (including controls if they reconfirmed). |
| **Q7** | {"Reproducible obs/action pairs saved under trajectories/; still thin for SFT until hard_910263 is solved and diversity grows." if not h263_full else "Yes for saved experts (legal K49 tokens + obs); expand before SFT."} |

## Stop

Bounded pilot complete. No training. No physics/vocab/reward changes.
""", encoding="utf-8")

    write_json(OUT / "summary.json", {
        "mechanism": mech,
        "before_pos_mm": before["e_pos_mm"],
        "after_pos_mm": after["e_pos_mm"],
        "hard_910263_full_expert": h263_full,
        "n_full_experts": n_full,
        "benefit_flag": benefit,
        "best_A_new": bool(a_survives),
        "best_B_new": bool(b_ok),
        "best_C_new": bool(c_ok),
    })


def _fmt(row, key):
    if row is None:
        return "—"
    v = row.get(key)
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def _tag(row):
    if row is None:
        return "no S12 candidate"
    return f"tag={row.get('tag')} NEW={row.get('SUCCESS_NEW')}"


if __name__ == "__main__":
    main()
