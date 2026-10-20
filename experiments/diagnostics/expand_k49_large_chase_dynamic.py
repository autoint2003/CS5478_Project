"""Large-displacement dynamic chase-and-recatch discovery.

Phase A: motion audit of existing SUCCESS_NEW + HIGH_MOTION failures; freeze LARGE_CHASE.
Phase B: targeted search for LARGE_CHASE / HIGH_MOTION / complete-contact-loss FULL experts.

Locked K49_tok48_wy0p25. SUCCESS_NEW unchanged. No SFT/GRPO.
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
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_generation.datasets.ball_impact import run_episode
from data_generation.datasets.recovery_rebuild import immediate_handoff
from data_generation.recording.trajectory_recorder import save_pair
from data_generation.oracles.plan_b.vertical_recatch import make_env
from envs.outcomes.horizon_outcome import NOMINAL_S, recovery_failed, terminal_reason
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.grasp_geometry import measure
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import (
    SECURE_REVERSE_WRIST,
    build_k49_table,
    describe_k49_config,
)
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

from scripts.expand_k49_hard_dynamic import (
    CAPTURE_SEED,
    failure_stage,
    ic_metrics as hard_ic_metrics,
    is_high_motion,
    mechanism_tags,
)
from scripts.generate_k49_ball_impact_dynamic_recovery import (
    classify_ic_regime,
    draw_lateral_spec,
)
from scripts.generate_k49_dynamic_recatch_experts import type2_proposals
from scripts.generate_k49_in_recovery_recatch import (
    find_first_bilateral,
    pose_correction_proposals,
    regrasp_proposals,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    classify_mechanisms,
    decode,
    decompose,
    evaluate_full,
    mapping_checksum,
    run_seq,
    slide_proposals,
    write_csv,
    write_json,
)
from scripts.generate_k49_type3_capture_brake import run_traced_with_snapshots

OUT = ROOT / "results" / "expert_generation" / "k49_large_chase_dynamic_recovery"
HARD = ROOT / "results" / "expert_generation" / "k49_hard_dynamic_expansion"
UNIFIED_BASE = HARD / "UNIFIED_MANIFEST.csv"
PHYS = HARD / "PHYSICAL_DIFFICULTY_METRICS.csv"
POOL = HARD / "HARD_DYNAMIC_IC_POOL.csv"

# Budgets recorded BEFORE execution (Phase B)
TOTAL_ROLLOUT_BUDGET = 4500
WALL_LIMIT_S = 75 * 60
PHASE_FRAC = (0.40, 0.40, 0.20)  # prior HM / new impacts / refine
TARGET_LARGE_CHASE = 5
TARGET_HIGH_MOTION = 3
TARGET_CONTACT_LOSS = 3

# Frozen HIGH_MOTION (unchanged from hard-dynamic expansion)
HM_THR = {
    "v_rel_norm_mps": 0.12,
    "w_rel_norm_radps": 2.5,
    "v_lat_mps": 0.10,
}

SEVEN = [
    "bimp_0001", "bimp_0026", "bimp_0055",
    "hard_910024", "hard_910028", "hard_910030", "hard_910042",
]

TRANSLATE_TOKS = (1, 5, 8, 13, 15, 16, 17, 19, 24, 33)
OPEN_TOKS = (30, 32, 36, 39, 41, 44)
BRAKE_TOK = 12


def tokens_hash(toks):
    return hashlib.sha256(json.dumps([int(x) for x in toks]).encode()).hexdigest()[:16]


def load_blob(path):
    return pickle.loads(Path(path).read_bytes())


def hand_pos(sim):
    return np.asarray(sim.data.xpos[sim.ids.hand_body], float).copy()


def obj_pos(sim):
    return np.asarray(sim.data.xpos[sim.ids.object_body], float).copy()


def hand_quat(sim):
    # free joint / body xquat
    return np.asarray(sim.data.xquat[sim.ids.hand_body], float).copy()


def ang_speed_from_quat(q0, q1, dt):
    """Approximate angular speed (rad/s) from consecutive body quaternions."""
    if dt <= 1e-12:
        return 0.0
    # relative rotation angle
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    # q_rel = q1 * conj(q0)
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    wr = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
    wr = float(np.clip(abs(wr), -1.0, 1.0))
    ang = 2.0 * float(np.arccos(wr))
    if ang > np.pi:
        ang = 2 * np.pi - ang
    return ang / dt


def wrist_angle_delta(q0, q1):
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    wr = float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
    ang = 2.0 * float(np.arccos(wr))
    if ang > np.pi:
        ang = 2 * np.pi - ang
    return ang


def sample_state(sim, t0):
    cs = contact_summary(sim)
    m = measure(sim, U_TGT, P_TGT)
    d = decompose(sim)
    ph = hand_pos(sim)
    po = obj_pos(sim)
    return {
        "t": float(sim.data.time) - t0,
        "nL": int(cs["nL"]), "nR": int(cs["nR"]), "bilateral": bool(cs["bilateral"]),
        "hand": ph, "obj": po,
        "sep": float(np.linalg.norm(po - ph)),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_lat_mm": float(d["e_lat_mm"]),
        "e_ax_mm": float(d["e_ax_mm"]),
        "v_rel": np.asarray(m["v_rel_h"], float),
        "w_rel": np.asarray(m["w_rel_h"], float),
        "obj_z": float(m["obj_z"]),
        "hand_q": hand_quat(sim),
    }


def motion_metrics_from_trace(trace, toks, onset_bilateral):
    """Aggregate physical motion metrics from dense per-token samples."""
    if not trace:
        return {}
    h0 = trace[0]["hand"]
    o0 = trace[0]["obj"]
    path = 0.0
    max_disp = 0.0
    max_lat = 0.0
    max_vert = 0.0
    peak_v = 0.0
    peak_w = 0.0
    wrist_tot = 0.0
    max_obj_disp = 0.0
    max_rel_disp = float(np.linalg.norm(trace[0]["obj"] - trace[0]["hand"]))
    max_tilt = float(trace[0]["e_theta_deg"])
    saw_loss = int(trace[0]["nL"]) + int(trace[0]["nR"]) == 0
    complete_loss = saw_loss or (not onset_bilateral and int(trace[0]["nL"]) + int(trace[0]["nR"]) == 0)
    first_recatch_i = None
    first_bil_i = None
    t_recatch = None
    t_bil = None
    hand_at_recatch = None
    hand_at_end = trace[-1]["hand"]

    for i in range(len(trace)):
        r = trace[i]
        disp = float(np.linalg.norm(r["hand"] - h0))
        max_disp = max(max_disp, disp)
        max_lat = max(max_lat, float(np.hypot(r["hand"][0] - h0[0], r["hand"][1] - h0[1])))
        max_vert = max(max_vert, abs(float(r["hand"][2] - h0[2])))
        max_obj_disp = max(max_obj_disp, float(np.linalg.norm(r["obj"] - o0)))
        rel = float(np.linalg.norm(r["obj"] - r["hand"]))
        max_rel_disp = max(max_rel_disp, rel)
        max_tilt = max(max_tilt, float(r["e_theta_deg"]))
        if int(r["nL"]) + int(r["nR"]) == 0:
            complete_loss = True
            saw_loss = True
        if i > 0:
            dt = max(float(r["t"] - trace[i - 1]["t"]), 1e-6)
            dh = float(np.linalg.norm(r["hand"] - trace[i - 1]["hand"]))
            path += dh
            peak_v = max(peak_v, dh / dt)
            peak_w = max(peak_w, ang_speed_from_quat(trace[i - 1]["hand_q"], r["hand_q"], dt))
            wrist_tot += wrist_angle_delta(trace[i - 1]["hand_q"], r["hand_q"])
        # first meaningful recatch: any contact after loss, or first bilateral restore
        any_c = int(r["nL"]) + int(r["nR"]) > 0
        if saw_loss and any_c and first_recatch_i is None:
            first_recatch_i = i
            t_recatch = float(r["t"])
            hand_at_recatch = r["hand"].copy()
        if r["bilateral"] and first_bil_i is None and (saw_loss or not onset_bilateral):
            first_bil_i = i
            t_bil = float(r["t"])
            if hand_at_recatch is None:
                hand_at_recatch = r["hand"].copy()
                t_recatch = float(r["t"])
                first_recatch_i = i
        # if never lost contact, first "recatch" is first bilateral still-ok (local)
        if not saw_loss and onset_bilateral and first_recatch_i is None and i == 0:
            pass

    # if never lost contact: treat first sample as "recatch" = local correction
    if first_recatch_i is None:
        # earliest bilateral after onset, or end
        for i, r in enumerate(trace):
            if r["bilateral"]:
                first_recatch_i = i
                t_recatch = float(r["t"])
                hand_at_recatch = r["hand"].copy()
                break
        if hand_at_recatch is None:
            hand_at_recatch = hand_at_end.copy()
            t_recatch = float(trace[-1]["t"])
            first_recatch_i = len(trace) - 1

    disp_to_recatch = float(np.linalg.norm(hand_at_recatch - h0))
    # chase duration: onset -> first recatch
    chase_s = float(t_recatch or 0.0)

    # mechanism flags from physics (not exclusive)
    had_open = any(int(t) in OPEN_TOKS for t in toks)
    had_translate = sum(1 for t in toks if int(t) in TRANSLATE_TOKS) >= 6
    had_brake = BRAKE_TOK in [int(t) for t in toks]
    had_return = SECURE_REVERSE_WRIST in [int(t) for t in toks]
    # contact-preserving: never lost all contact and max_disp smallish handled later
    contact_preserving = (not complete_loss) and onset_bilateral
    short_range = disp_to_recatch < 0.025 and max_disp < 0.035
    dynamic_chase = (max_disp >= 0.04 or disp_to_recatch >= 0.035) and (complete_loss or not onset_bilateral or had_translate)
    capture_brake = had_brake and (had_open or complete_loss)
    # sliding/pivoting after recatch: translation/pivot tokens after first_recatch_i
    after = toks[first_recatch_i:] if first_recatch_i is not None else []
    slide_pivot_after = sum(1 for t in after if int(t) in TRANSLATE_TOKS or int(t) in (4, 6, 10)) >= 8
    slow_return = had_return
    # stabilization: many 0/12 near end
    settle = sum(1 for t in toks[-16:] if int(t) in (0, 12)) >= 8

    bil_restored = any(r["bilateral"] for r in trace) and (
        complete_loss or not onset_bilateral or (first_bil_i is not None and first_bil_i > 0)
    )
    if onset_bilateral and not complete_loss:
        bil_restored = True  # maintained

    return {
        "hand_onset": h0.tolist(),
        "hand_recatch": hand_at_recatch.tolist(),
        "hand_handoff": hand_at_end.tolist(),
        "obj_onset": o0.tolist(),
        "hand_max_disp_m": max_disp,
        "hand_path_m": path,
        "hand_max_lat_m": max_lat,
        "hand_max_vert_m": max_vert,
        "hand_peak_v_mps": peak_v,
        "hand_peak_w_radps": peak_w,
        "wrist_rot_rad": wrist_tot,
        "t_to_recatch_s": chase_s,
        "t_to_bilateral_s": float(t_bil) if t_bil is not None else None,
        "disp_to_recatch_m": disp_to_recatch,
        "obj_max_disp_m": max_obj_disp,
        "rel_disp_onset_m": float(np.linalg.norm(o0 - h0)),
        "rel_disp_max_m": max_rel_disp,
        "max_tilt_deg": max_tilt,
        "complete_contact_loss": int(complete_loss),
        "bilateral_restored_by_recovery": int(bool(bil_restored)),
        "onset_bilateral": int(onset_bilateral),
        "mech_contact_preserving": int(contact_preserving and not dynamic_chase),
        "mech_short_range_recatch": int(short_range),
        "mech_dynamic_chase": int(dynamic_chase),
        "mech_capture_brake": int(capture_brake),
        "mech_slide_pivot_after": int(slide_pivot_after),
        "mech_slow_return": int(slow_return),
        "mech_stabilize_before_handoff": int(settle),
        "n_trace": len(trace),
    }


def replay_motion(env, table, book48, blob, tokens):
    """Replay legal tokens; return motion metrics + success flags."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    cs0 = contact_summary(env.sim)
    m0 = measure(env.sim, U_TGT, P_TGT)
    d0 = decompose(env.sim)
    onset_bil = bool(cs0["bilateral"])
    v0 = float(np.linalg.norm(m0["v_rel_h"]))
    w0 = float(np.linalg.norm(m0["w_rel_h"]))
    vlat0 = float(np.hypot(m0["v_rel_h"][0], m0["v_rel_h"][1]))
    trace = []
    lost_low = 0
    ok = True
    for tok in tokens:
        st = sample_state(env.sim, t0)
        st["tok"] = int(tok)
        trace.append(st)
        action = decode(table, book48, tok)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            ok = False
            break
    # final state
    if ok:
        st = sample_state(env.sim, t0)
        st["tok"] = -1
        trace.append(st)
    metrics = motion_metrics_from_trace(trace, tokens, onset_bil)
    metrics.update({
        "replay_ok": int(ok),
        "v0": v0, "w0": w0, "v_lat0": vlat0,
        "e_theta0": float(m0["e_theta_deg"]), "e_pos0": float(m0["e_pos_mm"]),
        "nL0": int(cs0["nL"]), "nR0": int(cs0["nR"]),
        "reason_fail": None if ok else (terminal_reason(env.sim) or "DROP"),
    })
    success_new = False
    win = {}
    if ok:
        res = evaluate_full(env, table, book48, blob, tokens, record_obs=False)
        success_new = bool(res.get("SUCCESS_NEW"))
        win = res.get("window") or {}
        metrics["SUCCESS_NEW"] = int(success_new)
        metrics["win_mean_theta"] = win.get("mean_theta")
        metrics["win_mean_pos"] = win.get("mean_pos")
        metrics["hold_both_s"] = win.get("longest_both_ok_s")
        metrics["eval_reason"] = res.get("reason")
    else:
        metrics["SUCCESS_NEW"] = 0
    return metrics, trace


def chase_proposals(book, desire, rng, n=40, has_contact=True):
    """Sustained legal translation toward cylinder + capture timing variants.

    Prefer contact-preserving long slides when contact remains; open-chase only
    when needed. Always end with close/brake/return options.
    """
    props = []
    # Contact-preserving long translation (no open) — primary for unilateral/bilateral
    for t in (17, 33, 13, 1, 5, 19, 8, 15):
        for n_ch in (10, 14, 18, 22, 26):
            props.append([t] * n_ch + [BRAKE_TOK] * 6 + [0] * 8)
            props.append([t] * n_ch + [BRAKE_TOK] * 5 + [SECURE_REVERSE_WRIST] * 24 + [0] * 6)
            props.append([t] * n_ch + [4] * 16 + [BRAKE_TOK] * 5 + [SECURE_REVERSE_WRIST] * 20 + [0] * 6)
            props.append([t] * (n_ch // 2) + [12] * 3 + [t] * (n_ch // 2) + [0] * 6 + [SECURE_REVERSE_WRIST] * 16)
    # Chase then capture-brake (brief open late)
    for t in (17, 33, 1, 5, 13):
        for n_ch in (12, 16, 20, 24):
            props.append([t] * n_ch + [30] * 3 + [0] * 6 + [BRAKE_TOK] * 8 + [0] * 8)
            props.append([t] * n_ch + [30] * 3 + [0] * 6 + [BRAKE_TOK] * 6 + [SECURE_REVERSE_WRIST] * 20 + [0] * 6)
            props.append([t] * 8 + [30] * 2 + [t] * 10 + [0] * 8 + [BRAKE_TOK] * 6 + [4] * 12 + [0] * 6)
    # Open-first only if no contact
    if not has_contact:
        for t in (17, 33, 1, 5):
            props.append([30] * 4 + [t] * 18 + [0] * 8 + [BRAKE_TOK] * 8 + [SECURE_REVERSE_WRIST] * 20)
            props.append([30] * 5 + [t] * 22 + [0] * 10 + [BRAKE_TOK] * 6 + [4] * 16 + [0] * 6)
            props.append([32] * 3 + [t] * 16 + [0] * 8 + [BRAKE_TOK] * 6 + [0] * 8)
    # Desire-aware slides + limited type2
    props.extend(slide_proposals(book, desire, rng, n=10))
    if not has_contact:
        props.extend(type2_proposals(book, desire, rng, n=8, open_prefix=[30]))
    props.extend(regrasp_proposals(book, desire, rng, n=6))
    # Known successful local seeds extended with long translate prefixes (forced travel)
    for t in (17, 33, 1):
        props.append([t] * 16 + list(CAPTURE_SEED) + [SECURE_REVERSE_WRIST] * 16)
        props.append([t] * 20 + [10] * 12 + [12] * 5 + [SECURE_REVERSE_WRIST] * 24 + [0] * 6)
        props.append([t] * 14 + [32] + [10] * 10 + [12] * 12 + [0] * 8)
    seen, uniq = set(), []
    for s in props:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    # Prefer longer translation counts
    uniq.sort(key=lambda s: -sum(1 for t in s if t in TRANSLATE_TOKS))
    head, mid, tail = uniq[: max(1, n // 3)], uniq[n // 3: 2 * n // 3], uniq[2 * n // 3:]
    rng.shuffle(mid)
    rng.shuffle(tail)
    return (head + mid + tail)[:n]


def freeze_large_chase(success_rows, fail_probe_rows):
    """Freeze LARGE_CHASE from measured displacement distributions BEFORE new search outcomes.

    Primary signal is displacement-to-first-recatch (interception travel), not full-episode
    max displacement (which is inflated by post-recatch Sliding/Return).
    """
    succ_disp = np.array([float(r["hand_max_disp_m"]) for r in success_rows], float)
    succ_rec = np.array([float(r["disp_to_recatch_m"]) for r in success_rows], float)
    fail_disp = np.array([float(r.get("hand_max_disp_m") or 0) for r in fail_probe_rows], float) if fail_probe_rows else np.array([0.0])

    def q(a, p):
        return float(np.percentile(a, p)) if len(a) else 0.0

    quantiles = {
        "success_max_disp_p50_m": q(succ_disp, 50),
        "success_max_disp_p75_m": q(succ_disp, 75),
        "success_max_disp_p90_m": q(succ_disp, 90),
        "success_max_disp_p95_m": q(succ_disp, 95),
        "success_max_disp_max_m": float(np.max(succ_disp)) if len(succ_disp) else 0.0,
        "success_disp_to_recatch_p50_m": q(succ_rec, 50),
        "success_disp_to_recatch_p90_m": q(succ_rec, 90),
        "success_disp_to_recatch_p95_m": q(succ_rec, 95),
        "success_disp_to_recatch_max_m": float(np.max(succ_rec)) if len(succ_rec) else 0.0,
        "fail_probe_max_disp_p50_m": q(fail_disp, 50),
        "fail_probe_max_disp_p90_m": q(fail_disp, 90),
    }
    # Cylinder radius ~1.5–2 cm. Existing SUCCESS_NEW disp_to_recatch p90 ≈ 1 cm
    # (local adjustment). Meaningful chase: hand travels multiple cylinder radii
    # before first restored contact. Floor at 3.5 cm (~2 radii), above p95 of
    # existing to-recatch displacements. Secondary: large onset-to-peak hand
    # displacement AND nontrivial to-recatch (>=2 cm) to avoid counting pure
    # post-recatch Return/Sliding as chase.
    thr_recatch = float(max(
        0.035,
        quantiles["success_disp_to_recatch_p95_m"] + 0.015,
        3.0 * max(quantiles["success_disp_to_recatch_p50_m"], 0.005),
    ))
    thr_recatch = float(min(thr_recatch, 0.08))
    thr_max = float(max(0.06, thr_recatch + 0.02))
    thr_max = float(min(thr_max, 0.10))
    return {
        "threshold_hand_max_disp_m": thr_max,
        "threshold_disp_to_recatch_m": thr_recatch,
        "threshold_max_requires_recatch_m": 0.02,
        "rule": (
            "LARGE_CHASE if (disp_to_recatch_m >= thr_recatch) OR "
            "(hand_max_disp_m >= thr_max AND disp_to_recatch_m >= 0.02). "
            "Primary metric is interception displacement, not path length or "
            "post-recatch Return travel."
        ),
        "quantiles": quantiles,
        "interpretation": (
            f"thr_recatch={thr_recatch:.3f} m hand travel to first restored contact "
            f"(~{thr_recatch/0.015:.1f}x cylinder radius); "
            f"thr_max={thr_max:.3f} m peak displacement only counts with >=2 cm "
            "to-recatch so post-catch Sliding/Return alone is not LARGE_CHASE. "
            "Existing SUCCESS_NEW to-recatch p90 is ~1 cm (short-range); this thr "
            "is well above that while remaining reachable in the ~4 s Recovery horizon."
        ),
    }


def is_large_chase(metrics, thr_pack):
    d_rec = float(metrics.get("disp_to_recatch_m") or 0)
    d_max = float(metrics.get("hand_max_disp_m") or 0)
    if d_rec >= thr_pack["threshold_disp_to_recatch_m"]:
        return True
    need = float(thr_pack.get("threshold_max_requires_recatch_m") or 0.02)
    return d_max >= thr_pack["threshold_hand_max_disp_m"] and d_rec >= need


def classify_failure_mode(trace, ok, bil_restored, success_new, eval_reason):
    """A–E taxonomy from task §5."""
    if not trace:
        return "A"
    if success_new:
        return "OK"
    min_sep = min(float(r["sep"]) for r in trace)
    saw_any = any(int(r["nL"]) + int(r["nR"]) > 0 for r in trace)
    saw_uni = any((int(r["nL"]) > 0) ^ (int(r["nR"]) > 0) for r in trace)
    saw_bil = any(bool(r["bilateral"]) for r in trace)
    # A: never approached
    if min_sep > 0.08 and not saw_any:
        return "A"
    if not saw_any:
        return "B" if min_sep < 0.06 else "A"
    if saw_any and not saw_bil:
        return "C"
    if saw_bil and not ok:
        return "D"
    if saw_bil and ok and not success_new:
        return "E"
    if eval_reason in ("DROP_AFTER_HANDOFF", "LOST_GRASP"):
        return "E"
    return "C" if saw_uni else "B"


def render_expert_video(env, blob, tokens, table, book48, dest, overlay):
    restore_capture(env, blob["capture"])
    sim = env.sim
    look = obj_pos(sim)
    cam = mjv_camera_from_preset(
        {"lookat": look, "distance": 0.55, "azimuth": 128.0, "elevation": -18.0}
    )
    renderer = mujoco.Renderer(sim.model, 480, 640)
    frames = []
    t0 = float(sim.data.time)
    dt = float(sim.model.opt.timestep)
    stride = max(1, int(round(0.04 / dt)))

    def grab(label, clock):
        cam.lookat[:] = obj_pos(sim)
        renderer.update_scene(sim.data, cam)
        pack = physical_pack(sim)
        dH = float(overlay.get("hand_max_disp_m") or 0)
        text = (
            f"{overlay.get('ic_id','')} {label} t={clock:.2f}s "
            f"n={int(pack['nL'])}/{int(pack['nR'])} "
            f"dH={dH:.3f}m "
            f"NEW={overlay.get('SUCCESS_NEW',0)}"
        )
        frames.append(caption(renderer.render(), text))

    grab("START", 0.0)
    for i, tok in enumerate(tokens):
        step_legal(sim, decode(table, book48, tok))
        if i % max(1, stride // 2) == 0 or i == len(tokens) - 1:
            grab("REC", float(sim.data.time) - t0)
    apply_handoff(sim, blob["ref"])
    t_h = float(sim.data.time)
    steps = 0
    while float(sim.data.time) - t_h < 2.5 - 1e-9:
        nominal_step(sim)
        steps += 1
        if steps % stride == 0:
            grab("HOLD", float(sim.data.time) - t_h)
    grab("END", float(sim.data.time) - t_h)
    renderer.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = write_frames_mp4(frames, dest, fps=25)
    shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
    return path


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------

def phase_a(env, table, book48, args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    (OUT / "trajectories").mkdir(exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    (OUT / "RAW").mkdir(exist_ok=True)

    unified = list(csv.DictReader(UNIFIED_BASE.open(encoding="utf-8")))
    print(f"Phase A: auditing {len(unified)} SUCCESS_NEW trajectories", flush=True)

    success_rows = []
    seven_rows = []
    for i, row in enumerate(unified):
        ic = row["ic_id"]
        toks = json.loads(row["tokens"])
        snap = Path(row["snapshot"])
        if not snap.exists():
            print(f"  SKIP missing snap {ic}", flush=True)
            continue
        blob = load_blob(snap)
        metrics, trace = replay_motion(env, table, book48, blob, toks)
        rec = {
            "ic_id": ic,
            "root_episode": row.get("root_episode") or ic,
            "source_dataset": row.get("source_dataset") or "",
            "n_tokens": len(toks),
            "tokens_hash": tokens_hash(toks),
            **{k: metrics[k] for k in metrics if k not in ("hand_onset", "hand_recatch", "hand_handoff", "obj_onset")},
            "hand_onset": json.dumps(metrics.get("hand_onset")),
            "hand_recatch": json.dumps(metrics.get("hand_recatch")),
            "hand_handoff": json.dumps(metrics.get("hand_handoff")),
            "high_motion_onset": int(
                metrics["v0"] >= HM_THR["v_rel_norm_mps"]
                or metrics["w0"] >= HM_THR["w_rel_norm_radps"]
                or metrics["v_lat0"] >= HM_THR["v_lat_mps"]
            ),
            "snapshot": str(snap),
            "tokens": row["tokens"],
        }
        success_rows.append(rec)
        if ic in SEVEN:
            seven_rows.append({**rec, "tokens": row["tokens"]})
        if (i + 1) % 10 == 0:
            print(f"  audited {i+1}/{len(unified)}", flush=True)

    write_csv(OUT / "EXISTING_SUCCESS_MOTION_AUDIT.csv", success_rows)
    write_csv(OUT / "SEVEN_DYNAMIC_SUCCESS_AUDIT.csv", seven_rows)

    # HIGH_MOTION failure audit: 42 measured ICs from PHYSICAL_DIFFICULTY
    phys = list(csv.DictReader(PHYS.open(encoding="utf-8")))
    pool = {r["ic_id"]: r for r in csv.DictReader(POOL.open(encoding="utf-8"))}
    hm_ics = []
    seen = set()
    for r in phys:
        ic = r["ic_id"]
        if ic in seen:
            continue
        v, w, vl = float(r["v_rel_norm"]), float(r["w_rel_norm"]), float(r["v_lat"])
        if v >= HM_THR["v_rel_norm_mps"] or w >= HM_THR["w_rel_norm_radps"] or vl >= HM_THR["v_lat_mps"]:
            seen.add(ic)
            hm_ics.append(r)

    print(f"Phase A: auditing {len(hm_ics)} HIGH_MOTION ICs", flush=True)
    fail_rows = []
    fail_probe_motion = []
    snap_dirs = [
        HARD / "snapshots",
        ROOT / "results/expert_generation/k49_ball_impact_dynamic_recovery/snapshots",
        ROOT / "results/diagnostics/hard_ic_mining/snapshots",
    ]

    def find_snap(ic):
        for d in snap_dirs:
            p = d / f"{ic}.pkl"
            if p.exists():
                return p
        pr = pool.get(ic)
        if pr and Path(pr.get("snapshot") or "").exists():
            return Path(pr["snapshot"])
        return None

    for r in hm_ics:
        ic = r["ic_id"]
        snap = find_snap(ic)
        if snap is None:
            fail_rows.append({
                "ic_id": ic, "high_motion": 1, "audit_status": "NO_SNAPSHOT",
                "v0": r["v_rel_norm"], "w0": r["w_rel_norm"], "v_lat0": r["v_lat"],
            })
            continue
        blob = load_blob(snap)
        st = hard_ic_metrics(env, blob)
        # probe: zero hold, CAPTURE_SEED, and best short chase
        probes = {
            "zero16": [0] * 16,
            "capture_seed": list(CAPTURE_SEED),
            "chase_t17": [17] * 16 + [30] * 4 + [0] * 6 + [12] * 6,
            "chase_t33": [33] * 16 + [30] * 4 + [0] * 6 + [12] * 6,
            "chase_t1": [1] * 20 + [30] * 5 + [0] * 8 + [12] * 6,
            "open_chase": [30] * 4 + [17] * 18 + [0] * 8 + [12] * 6,
        }
        best = None
        t_escape = None
        max_hand = 0.0
        best_sep = 1e9
        best_uni = False
        best_bil = False
        earliest_restore = None
        fail_mode = "A"
        for name, seq in probes.items():
            metrics, trace = replay_motion(env, table, book48, blob, seq)
            max_hand = max(max_hand, float(metrics.get("hand_max_disp_m") or 0))
            if trace:
                best_sep = min(best_sep, min(float(x["sep"]) for x in trace))
                if any((int(x["nL"]) > 0) ^ (int(x["nR"]) > 0) for x in trace):
                    best_uni = True
                if any(bool(x["bilateral"]) for x in trace):
                    best_bil = True
                    for x in trace:
                        if x["bilateral"]:
                            earliest_restore = float(x["t"]) if earliest_restore is None else min(earliest_restore, float(x["t"]))
                            break
            mode = classify_failure_mode(
                trace, bool(metrics.get("replay_ok")), best_bil,
                bool(metrics.get("SUCCESS_NEW")), metrics.get("eval_reason"),
            )
            # prefer progress order OK>E>D>C>B>A
            order = {"OK": 0, "E": 1, "D": 2, "C": 3, "B": 4, "A": 5}
            if best is None or order.get(mode, 9) < order.get(fail_mode, 9):
                best = name
                fail_mode = mode
            if not metrics.get("replay_ok"):
                # escape time approx last trace time
                if trace:
                    te = float(trace[-1]["t"])
                    t_escape = te if t_escape is None else min(t_escape, te)
            fail_probe_motion.append({
                "ic_id": ic, "probe": name,
                "hand_max_disp_m": metrics.get("hand_max_disp_m"),
                "disp_to_recatch_m": metrics.get("disp_to_recatch_m"),
                "SUCCESS_NEW": metrics.get("SUCCESS_NEW"),
            })

        # kinematic reachability bound: under legal translate for remaining horizon,
        # estimate max hand travel from peak probe speed * remaining time (validated by probes)
        remaining_s = 4.0
        # from probes, peak hand speed observed
        reach_bound_m = max_hand  # validated by legal rollouts (conservative: achieved max)

        # plausible interception: not A, and (best_sep < 0.05 or best_uni/bil) and escape not instant
        plausible = fail_mode in ("B", "C", "D", "E") or (best_sep < 0.055 and (t_escape is None or t_escape > 0.15))

        fail_rows.append({
            "ic_id": ic,
            "root_episode": ic,
            "high_motion": 1,
            "audit_status": "OK",
            "v0": st["v_rel_norm"], "w0": st["w_rel_norm"], "v_lat0": st["v_lat"],
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "nL0": st["nL"], "nR0": st["nR"], "bilateral0": int(st["bilateral"]),
            "obj_z0": st["obj_z"],
            "hand_onset": json.dumps(hand_pos(env.sim).tolist()) if False else "",
            "best_probe": best,
            "failure_class": fail_mode,
            "t_escape_s": t_escape,
            "max_hand_disp_legal_m": max_hand,
            "best_sep_m": best_sep if best_sep < 1e8 else None,
            "best_unilateral": int(best_uni),
            "best_bilateral": int(best_bil),
            "earliest_restore_s": earliest_restore,
            "reach_bound_m": reach_bound_m,
            "reach_bound_assumption": "max hand displacement achieved under legal probe rollouts (not analytic)",
            "plausible_interception": int(plausible),
            "snapshot": str(snap),
        })

    write_csv(OUT / "HIGH_MOTION_FAILURE_AUDIT.csv", fail_rows)

    # Freeze LARGE_CHASE BEFORE Phase B
    thr_pack = freeze_large_chase(success_rows, fail_probe_motion)
    write_json(OUT / "RAW" / "LARGE_CHASE_THRESHOLD_FREEZE.json", thr_pack)
    md = OUT / "CHASE_DISPLACEMENT_THRESHOLDS.md"
    md.write_text(
        "# LARGE_CHASE displacement thresholds (frozen before Phase B search)\n\n"
        f"**HIGH_MOTION (unchanged):** v≥{HM_THR['v_rel_norm_mps']} m/s OR "
        f"w≥{HM_THR['w_rel_norm_radps']} rad/s OR v_lat≥{HM_THR['v_lat_mps']} m/s\n\n"
        f"**LARGE_CHASE thr_max:** {thr_pack['threshold_hand_max_disp_m']:.4f} m\n"
        f"**LARGE_CHASE thr_recatch:** {thr_pack['threshold_disp_to_recatch_m']:.4f} m\n\n"
        f"**Rule:** {thr_pack['rule']}\n\n"
        f"**Interpretation:** {thr_pack['interpretation']}\n\n"
        "## Quantiles (existing SUCCESS_NEW hand motion)\n\n"
        + "\n".join(f"- `{k}`: {v:.4f}" for k, v in thr_pack["quantiles"].items())
        + "\n\nNeither diagnostic changes SUCCESS_NEW.\n",
        encoding="utf-8",
    )
    # annotate success rows with LARGE_CHASE
    for r in success_rows:
        r["large_chase"] = int(is_large_chase(r, thr_pack))
    write_csv(OUT / "EXISTING_SUCCESS_MOTION_AUDIT.csv", success_rows)
    for r in seven_rows:
        r["large_chase"] = int(is_large_chase(r, thr_pack))
        r["pursuit_vs_local"] = (
            "ACTIVE_PURSUIT" if float(r["disp_to_recatch_m"]) >= 0.03 or float(r["hand_max_disp_m"]) >= 0.04
            else "LOCAL_CORRECTION"
        )
    write_csv(OUT / "SEVEN_DYNAMIC_SUCCESS_AUDIT.csv", seven_rows)

    n_lc = sum(int(r["large_chase"]) for r in success_rows)
    n_plaus = sum(int(r.get("plausible_interception") or 0) for r in fail_rows)
    print(
        f"Phase A done. LARGE_CHASE thr={thr_pack['threshold_hand_max_disp_m']:.3f}m; "
        f"existing LARGE_CHASE successes={n_lc}/{len(success_rows)}; "
        f"plausible HM failures={n_plaus}/{len(fail_rows)}",
        flush=True,
    )
    return {
        "success_rows": success_rows,
        "seven_rows": seven_rows,
        "fail_rows": fail_rows,
        "thr_pack": thr_pack,
        "hm_ics": hm_ics,
    }


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------

def save_full_expert(env, table, book48, blob, ic_id, root, tag, toks, st, thr_pack,
                     source, checksum, snap_src, regimes, round_id, mapping_meta):
    res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
    if not res["SUCCESS_NEW"]:
        return None
    metrics, _ = replay_motion(env, table, book48, blob, toks)
    hm = is_high_motion(st, HM_THR)
    lc = is_large_chase(metrics, thr_pack)
    tags = mechanism_tags(toks, st, hm)
    if lc:
        tags.append("LARGE_CHASE")
    if metrics.get("complete_contact_loss"):
        tags.append("COMPLETE_CONTACT_LOSS")
    stem = f"{ic_id}__{tag}"
    save_pair(
        OUT / "trajectories", stem,
        res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    meta = {
        "ic_id": ic_id, "root_episode": root, "tag": tag,
        "expert_type": "LARGE_CHASE_FULL_EXPERT",
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "SUCCESS_NEW": True, "window": res["window"], "tokens": toks,
        "high_motion": hm, "large_chase": lc,
        "mechanism_tags": tags, "regimes": regimes,
        "source": source, "round": round_id,
        "motion": {k: metrics[k] for k in (
            "hand_max_disp_m", "disp_to_recatch_m", "hand_path_m",
            "obj_max_disp_m", "t_to_recatch_s", "complete_contact_loss",
            "v0", "w0", "v_lat0",
        ) if k in metrics},
        "obs_dim": 39, "handoff_id": 47,
    }
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    dst = OUT / "snapshots" / f"{ic_id}.pkl"
    if not dst.exists():
        if Path(snap_src).exists():
            shutil.copy2(snap_src, dst)
        else:
            dst.write_bytes(pickle.dumps(blob))
    return {
        "ic_id": ic_id, "root_episode": root, "tag": tag,
        "expert_type": "LARGE_CHASE_FULL_EXPERT", "SUCCESS_NEW": 1,
        "win_mean_theta": res["window"].get("mean_theta"),
        "win_mean_pos": res["window"].get("mean_pos"),
        "hold_both_s": res["window"].get("longest_both_ok_s"),
        "n_tokens": len(toks), "tokens": json.dumps(toks),
        "tokens_hash": tokens_hash(toks),
        "high_motion": int(hm), "large_chase": int(lc),
        "complete_contact_loss": int(metrics.get("complete_contact_loss") or 0),
        "hand_max_disp_m": metrics.get("hand_max_disp_m"),
        "disp_to_recatch_m": metrics.get("disp_to_recatch_m"),
        "obj_max_disp_m": metrics.get("obj_max_disp_m"),
        "mechanism_tags": json.dumps(tags),
        "regimes": json.dumps(regimes),
        "source": source, "round": round_id,
        "v0": st["v_rel_norm"], "w0": st["w_rel_norm"], "v_lat0": st["v_lat"],
        "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
        "nL0": st["nL"], "nR0": st["nR"], "bilateral0": int(st["bilateral"]),
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "snapshot": str(dst),
    }


def search_ic(env, table, book48, blob, ic_id, root, st, thr_pack, rng,
              rollout_counter, rollout_limit, budgets, logs, checksum, mapping_meta,
              source, round_id, regimes, solved_roots, per_ic_cap=120):
    """Chase-focused multi-stage search with selective event branching."""
    if ic_id in solved_roots or rollout_counter[0] >= rollout_limit:
        return None
    ic_start = rollout_counter[0]
    has_contact = int(st["nL"]) + int(st["nR"]) > 0
    restore_capture(env, blob["capture"])
    desire = -decompose(env.sim)["err_h"]
    props = chase_proposals(book48, desire, rng, n=budgets["a"], has_contact=has_contact)
    # short-prefix seeds for transfer comparison (tagged short_prefix)
    short_seeds = [
        ("short_cap", list(CAPTURE_SEED)),
        ("short_open", [30] + [0] * 12),
        ("short_t17", [17] * 8 + [0] * 8),
        ("short_t33", [33] * 8 + [12] * 5 + [0] * 8),
    ]
    for name, s in reversed(short_seeds):
        props.insert(0, s)

    intercept_rows, capture_rows, pose_rows, failures = logs
    full = None
    promising_prefs = []  # (score, pref, kind)

    def try_tokens(tag, toks, short_prefix=0):
        nonlocal full
        if rollout_counter[0] >= rollout_limit or rollout_counter[0] - ic_start >= per_ic_cap:
            return False
        if full is not None:
            return True
        rollout_counter[0] += 1
        metrics, trace = replay_motion(env, table, book48, blob, toks)
        lc = is_large_chase(metrics, thr_pack)
        hm = is_high_motion(st, HM_THR)
        row = {
            "ic_id": ic_id, "tag": tag, "round": round_id,
            "n_tokens": len(toks), "large_chase": int(lc), "high_motion": int(hm),
            "hand_max_disp_m": metrics.get("hand_max_disp_m"),
            "disp_to_recatch_m": metrics.get("disp_to_recatch_m"),
            "complete_contact_loss": int(metrics.get("complete_contact_loss") or 0),
            "SUCCESS_NEW": int(metrics.get("SUCCESS_NEW") or 0),
            "replay_ok": metrics.get("replay_ok"),
            "reason": metrics.get("eval_reason") or metrics.get("reason_fail"),
            "short_prefix": int(short_prefix),
        }
        if int(metrics.get("complete_contact_loss") or 0) or not st["bilateral"]:
            intercept_rows.append(row)
        elif metrics.get("bilateral_restored_by_recovery"):
            capture_rows.append(row)
        else:
            pose_rows.append(row)

        # track promising bilateral / reduced-sep prefixes for branching
        if metrics.get("replay_ok") and (
            metrics.get("bilateral_restored_by_recovery")
            or float(metrics.get("disp_to_recatch_m") or 0) >= 0.025
            or float(metrics.get("hand_max_disp_m") or 0) >= 0.04
        ):
            sc = float(metrics.get("hand_max_disp_m") or 0) + 2.0 * float(metrics.get("disp_to_recatch_m") or 0)
            if metrics.get("bilateral_restored_by_recovery"):
                sc += 1.0
            promising_prefs.append((sc, list(toks), tag))

        if metrics.get("SUCCESS_NEW"):
            snap_src = OUT / "snapshots" / f"{ic_id}.pkl"
            if not snap_src.exists():
                snap_src.write_bytes(pickle.dumps(blob))
            erow = save_full_expert(
                env, table, book48, blob, ic_id, root, tag, toks, st,
                thr_pack, source, checksum, snap_src, regimes, round_id, mapping_meta,
            )
            if erow is not None:
                if erow["large_chase"] or erow["high_motion"] or erow["complete_contact_loss"]:
                    full = erow
                    solved_roots.add(ic_id)
                    print(
                        f"    FULL {ic_id} LC={erow['large_chase']} HM={erow['high_motion']} "
                        f"CL={erow['complete_contact_loss']} "
                        f"dH={float(erow['hand_max_disp_m']):.3f}m "
                        f"dR={float(erow['disp_to_recatch_m']):.3f}m",
                        flush=True,
                    )
                    return True
                failures.append({**row, "reject": "easy_small_motion"})
                return False
        failures.append(row)
        return False

    # Transfer prior dynamic SUCCESS_NEW token sequences (+ long translate prefixes)
    prior_full = HARD / "FULL_EXPERTS.csv"
    if prior_full.exists():
        for pr in csv.DictReader(prior_full.open(encoding="utf-8")):
            try:
                pt = json.loads(pr["tokens"])
            except Exception:
                continue
            try_tokens(f"xfer_{pr['ic_id']}", list(pt), short_prefix=1)
            for t in (17, 33, 1, 5):
                try_tokens(f"xfer_{pr['ic_id']}_ch{t}", [t] * 14 + list(pt), short_prefix=0)
                try_tokens(
                    f"xfer_{pr['ic_id']}_ch{t}ret",
                    [t] * 14 + list(pt) + [SECURE_REVERSE_WRIST] * 16 + [0] * 4,
                    short_prefix=0,
                )
            if full or rollout_counter[0] - ic_start >= per_ic_cap:
                break

    for ia, seq in enumerate(props):
        if rollout_counter[0] >= rollout_limit or rollout_counter[0] - ic_start >= per_ic_cap or full:
            break
        is_short = ia < len(short_seeds)
        # base + return / settle
        try_tokens(f"a{ia}_raw", list(seq), short_prefix=int(is_short))
        try_tokens(f"a{ia}_ret", list(seq) + [SECURE_REVERSE_WRIST] * 24 + [0] * 6, short_prefix=0)
        if full:
            break
        # Selective event branch only if sequence survives and reaches contact/min-sep
        restore_capture(env, blob["capture"])
        lost_low = 0
        snaps_at = {}
        for ti, tok in enumerate(seq):
            act = decode(table, book48, tok)
            step_legal(env.sim, act)
            env._prev = np.asarray(act, np.float32)
            cs = contact_summary(env.sim)
            sep = float(np.linalg.norm(obj_pos(env.sim) - hand_pos(env.sim)))
            if "min_sep" not in snaps_at or sep < snaps_at["min_sep"][0]:
                snaps_at["min_sep"] = (sep, ti, list(seq[: ti + 1]))
            if cs["bilateral"] and "first_bil" not in snaps_at:
                snaps_at["first_bil"] = (sep, ti, list(seq[: ti + 1]))
            failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
            if failed:
                break
        else:
            # only branch from first_bil or close min_sep
            branch_keys = []
            if "first_bil" in snaps_at:
                branch_keys.append("first_bil")
            if "min_sep" in snaps_at and snaps_at["min_sep"][0] < 0.05:
                branch_keys.append("min_sep")
            for bk in branch_keys[:1]:  # at most one event type per seq
                if rollout_counter[0] - ic_start >= per_ic_cap:
                    break
                pref = snaps_at[bk][2]
                restore_capture(env, blob["capture"])
                run_seq(env, table, book48, blob, pref)
                desire_b = -decompose(env.sim)["err_h"]
                # capture-brake continuations from chase peak
                for cont_name, cont in [
                    ("brake", [0] * 4 + [BRAKE_TOK] * 8 + [SECURE_REVERSE_WRIST] * 20 + [0] * 6),
                    ("close_brake", [0] * 8 + [BRAKE_TOK] * 6 + [4] * 12 + [SECURE_REVERSE_WRIST] * 16),
                ]:
                    try_tokens(f"a{ia}_{bk}_{cont_name}", list(pref) + cont)
                for ib, corr in enumerate(pose_correction_proposals(book48, desire_b, rng, n=budgets["b"])[: budgets["b"]]):
                    try_tokens(f"a{ia}_{bk}_c{ib}", list(pref) + list(corr) + [SECURE_REVERSE_WRIST] * 16 + [0] * 4)
                    if full:
                        break

    # Extra refine on top promising prefs: extend translate then return
    if full is None and promising_prefs:
        promising_prefs.sort(key=lambda x: -x[0])
        for sc, pref, ptag in promising_prefs[:4]:
            if rollout_counter[0] - ic_start >= per_ic_cap or rollout_counter[0] >= rollout_limit:
                break
            for t in (17, 33, 1):
                ext = list(pref) + [t] * 8 + [BRAKE_TOK] * 5 + [SECURE_REVERSE_WRIST] * 20 + [0] * 4
                try_tokens(f"refine_{ptag}_{t}", ext)
                if full:
                    break

    return full


def phase_b(env, table, book48, audit, args, checksum, mapping_meta):
    thr_pack = audit["thr_pack"]
    fail_rows = audit["fail_rows"]
    rng = np.random.default_rng(20261019)

    # record budgets
    budget_doc = {
        "TOTAL_ROLLOUT_BUDGET": TOTAL_ROLLOUT_BUDGET,
        "WALL_LIMIT_S": WALL_LIMIT_S,
        "PHASE_FRAC": PHASE_FRAC,
        "targets": {
            "LARGE_CHASE": TARGET_LARGE_CHASE,
            "HIGH_MOTION": TARGET_HIGH_MOTION,
            "COMPLETE_CONTACT_LOSS": TARGET_CONTACT_LOSS,
        },
        "recorded_before_execution": True,
    }
    write_json(OUT / "RAW" / "PHASE_B_BUDGET.json", budget_doc)
    round_budgets = [int(TOTAL_ROLLOUT_BUDGET * f) for f in PHASE_FRAC]
    round_budgets[2] = TOTAL_ROLLOUT_BUDGET - sum(round_budgets[:2])

    rollout_counter = [0]
    t0 = time.time()
    solved = set()
    full_experts = []
    intercept_rows, capture_rows, pose_rows, failures = [], [], [], []
    logs = (intercept_rows, capture_rows, pose_rows, failures)
    new_impact_log = []
    coverage = []

    def time_left():
        return time.time() - t0 < WALL_LIMIT_S

    # --- Phase B1: prior plausible HIGH_MOTION failures (40%) ---
    b1_limit = round_budgets[0]
    print(f"=== Phase B1 prior HM (budget {b1_limit}) ===", flush=True)
    candidates = [
        r for r in fail_rows
        if int(r.get("plausible_interception") or 0) == 1
        and r.get("snapshot")
        and r.get("failure_class") in ("B", "C", "D", "E")
    ]
    # secondary: other non-A HM with residual contact
    others = [
        r for r in fail_rows
        if r.get("snapshot") and r.get("audit_status") == "OK"
        and r.get("failure_class") in ("B", "C", "D", "E")
        and r not in candidates
    ]
    candidates = candidates + others
    candidates.sort(key=lambda r: (
        0 if int(r.get("plausible_interception") or 0) else 1,
        {"E": 0, "D": 1, "C": 2, "B": 3}.get(r.get("failure_class"), 9),
        float(r.get("best_sep_m") or 1),
    ))

    for row in candidates:
        if rollout_counter[0] >= b1_limit or not time_left():
            break
        if len([e for e in full_experts if e["large_chase"]]) >= TARGET_LARGE_CHASE and \
           len([e for e in full_experts if e["high_motion"]]) >= TARGET_HIGH_MOTION and \
           len([e for e in full_experts if e["complete_contact_loss"]]) >= TARGET_CONTACT_LOSS:
            break
        ic = row["ic_id"]
        if ic in SEVEN or ic in solved:
            continue
        snap = Path(row["snapshot"])
        blob = load_blob(snap)
        st = hard_ic_metrics(env, blob)
        regimes = classify_ic_regime(blob, st)
        before = rollout_counter[0]
        print(f"  B1 search {ic} v={st['v_rel_norm']:.2f} nL/R={st['nL']}/{st['nR']} "
              f"class={row.get('failure_class')} roll={rollout_counter[0]}", flush=True)
        # Spend more budget on D/E (bilateral-ish) than C
        cap = args.per_ic_cap + (40 if row.get("failure_class") in ("D", "E") else 0)
        erow = search_ic(
            env, table, book48, blob, ic, ic, st, thr_pack, rng,
            rollout_counter, b1_limit, {"a": args.budget_a + (6 if row.get("failure_class") in ("D", "E") else 0), "b": args.budget_b},
            logs, checksum, mapping_meta, "prior_hm_failure", "B1", regimes, solved,
            per_ic_cap=cap,
        )
        coverage.append({"ic_id": ic, "round": "B1", "full": int(erow is not None),
                         "searched": rollout_counter[0] - before})
        if erow:
            full_experts.append(erow)

    # --- Phase B2: new large-displacement impacts (40%) ---
    b2_limit = b1_limit + round_budgets[1]
    print(f"=== Phase B2 new impacts (budget to {b2_limit}) ===", flush=True)
    new_items = []
    for i in range(args.max_new_impacts):
        if not time_left() or rollout_counter[0] >= b2_limit:
            break
        spec = draw_lateral_spec(9000 + i)
        # bias toward substantial but interceptable displacement
        if float(rng.random()) < 0.6:
            spec["speed_mps"] = float(rng.uniform(6.5, 9.0))
        if float(rng.random()) < 0.5:
            spec["z_off_m"] = float(rng.uniform(-0.02, 0.02))
        eid = f"lcimp_{i:04d}"
        spec["attempt_id"] = eid
        evidence, blob = run_episode(env, spec)
        log = {
            "root_id": eid, "category": evidence.get("category"),
            "azimuth_deg": spec["azimuth_deg"], "speed_mps": spec["speed_mps"],
            "phi_deg": spec["phi_deg"], "z_off_m": spec["z_off_m"],
            "peak_fn": evidence.get("peak_normal_force_n"),
        }
        if blob is None or evidence.get("category") != "valid_impact_recovery_ic":
            log["accepted"] = 0
            new_impact_log.append(log)
            continue
        nom = immediate_handoff(env, blob)
        log["nominal_class"] = nom["nominal_class"]
        if nom["nominal_class"] != "RECOVERY_CANDIDATE":
            log["accepted"] = 0
            new_impact_log.append(log)
            continue
        path = OUT / "snapshots" / f"{eid}.pkl"
        path.write_bytes(pickle.dumps(blob))
        st = hard_ic_metrics(env, blob)
        # Escape probe: brief open then free — measures cylinder travel if grasp weakens
        metrics, _ = replay_motion(env, table, book48, blob, [30] * 3 + [0] * 18)
        rollout_counter[0] += 1
        regimes = classify_ic_regime(blob, st)
        hm = int(is_high_motion(st, HM_THR))
        obj_d = float(metrics.get("obj_max_disp_m") or 0)
        log.update({
            "accepted": 1, "high_motion": hm,
            "v0": st["v_rel_norm"], "w0": st["w_rel_norm"], "v_lat0": st["v_lat"],
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "nL": st["nL"], "nR": st["nR"], "bilateral": int(st["bilateral"]),
            "zero_probe_obj_disp": obj_d,
            "zero_probe_hand_disp": metrics.get("hand_max_disp_m"),
            "probe_reason": metrics.get("reason_fail") or metrics.get("eval_reason"),
            "regimes": json.dumps(regimes), "snapshot": str(path),
        })
        # reject immediate irreversible: obj already very low / huge speed
        if st["obj_z"] < 0.32 or st["v_rel_norm"] > 0.85:
            log["accepted"] = 0
            log["reject"] = "irreversible_escape"
            new_impact_log.append(log)
            continue
        # Prefer nontrivial escape or HM / unilateral — skip near-static bilateral soft impacts
        nontrivial = (
            hm or (st["nL"] + st["nR"] < 2) or obj_d >= 0.02
            or float(st["e_pos_mm"]) >= 12.0 or float(st["v_lat"]) >= 0.08
        )
        if not nontrivial:
            log["accepted"] = 0
            log["reject"] = "too_static"
            new_impact_log.append(log)
            continue
        new_impact_log.append(log)
        new_items.append({
            "ic_id": eid, "root": eid, "blob": blob, "st": st,
            "regimes": regimes, "snap": path,
            "obj_disp_probe": obj_d,
            "hm": hm,
        })
        print(f"  NEW {eid} HM={hm} v={st['v_rel_norm']:.2f} obj_d={obj_d:.3f}", flush=True)

    write_csv(OUT / "NEW_IMPACT_ROOTS.csv", new_impact_log)
    # prefer larger object displacement probes that are still HM or unilateral
    new_items.sort(key=lambda x: (-x["obj_disp_probe"], -x["hm"]))
    for item in new_items:
        if rollout_counter[0] >= b2_limit or not time_left():
            break
        before = rollout_counter[0]
        print(f"  B2 search {item['ic_id']} HM={item['hm']} obj_d={item['obj_disp_probe']:.3f} "
              f"roll={rollout_counter[0]}", flush=True)
        erow = search_ic(
            env, table, book48, item["blob"], item["ic_id"], item["root"], item["st"],
            thr_pack, rng, rollout_counter, b2_limit, {"a": args.budget_a + 2, "b": args.budget_b},
            logs, checksum, mapping_meta, "new_large_impact", "B2", item["regimes"], solved,
            per_ic_cap=args.per_ic_cap,
        )
        coverage.append({"ic_id": item["ic_id"], "round": "B2", "full": int(erow is not None),
                         "searched": rollout_counter[0] - before})
        if erow:
            full_experts.append(erow)

    # --- Phase B3: refine partial interceptions (20%) ---
    b3_limit = TOTAL_ROLLOUT_BUDGET
    print(f"=== Phase B3 refine (budget to {b3_limit}) ===", flush=True)
    # ICs that got bilateral in logs but not SUCCESS_NEW
    refine_ics = {}
    for r in capture_rows + intercept_rows:
        if int(r.get("SUCCESS_NEW") or 0):
            continue
        if r.get("reason") in (None, "DROP_AFTER_HANDOFF", "LOST_GRASP", "SUCCESS_12S"):
            refine_ics.setdefault(r["ic_id"], r)
    for ic, _ in list(refine_ics.items())[:40]:
        if rollout_counter[0] >= b3_limit or not time_left():
            break
        if ic in solved:
            continue
        snap = None
        for d in [OUT / "snapshots", HARD / "snapshots",
                  ROOT / "results/expert_generation/k49_ball_impact_dynamic_recovery/snapshots",
                  ROOT / "results/diagnostics/hard_ic_mining/snapshots"]:
            p = d / f"{ic}.pkl"
            if p.exists():
                snap = p
                break
        if snap is None:
            continue
        blob = load_blob(snap)
        st = hard_ic_metrics(env, blob)
        regimes = classify_ic_regime(blob, st)
        before = rollout_counter[0]
        print(f"  B3 refine {ic} roll={rollout_counter[0]}", flush=True)
        erow = search_ic(
            env, table, book48, blob, ic, ic, st, thr_pack, rng,
            rollout_counter, b3_limit, {"a": args.budget_a + 4, "b": args.budget_b + 1},
            logs, checksum, mapping_meta, "refine_interception", "B3", regimes, solved,
            per_ic_cap=args.per_ic_cap + 20,
        )
        coverage.append({"ic_id": ic, "round": "B3", "full": int(erow is not None),
                         "searched": rollout_counter[0] - before})
        if erow:
            full_experts.append(erow)

    write_csv(OUT / "INTERCEPTION_SEARCH.csv", intercept_rows[:3000])
    write_csv(OUT / "CAPTURE_SEARCH.csv", capture_rows[:3000])
    write_csv(OUT / "POSE_CORRECTION_SEARCH.csv", pose_rows[:3000])
    write_csv(OUT / "FAILURE_ANALYSIS.csv", failures[:3000])
    write_csv(OUT / "FULL_EXPERTS.csv", full_experts)
    write_csv(OUT / "ROOT_COVERAGE.csv", coverage)

    provenance = []
    for e in full_experts:
        provenance.append({
            "ic_id": e["ic_id"], "root_episode": e["root_episode"],
            "source": e["source"], "round": e["round"],
            "large_chase": e["large_chase"], "high_motion": e["high_motion"],
            "complete_contact_loss": e["complete_contact_loss"],
            "tokens_hash": e["tokens_hash"], "snapshot": e["snapshot"],
        })
    write_csv(OUT / "ROOT_PROVENANCE.csv", provenance)

    wall = time.time() - t0
    write_json(OUT / "RAW" / "PHASE_B_USAGE.json", {
        "rollouts_used": rollout_counter[0],
        "rollout_budget": TOTAL_ROLLOUT_BUDGET,
        "wall_s": wall,
        "wall_limit_s": WALL_LIMIT_S,
        "n_full": len(full_experts),
        "n_large_chase": sum(int(e["large_chase"]) for e in full_experts),
        "n_high_motion": sum(int(e["high_motion"]) for e in full_experts),
        "n_contact_loss": sum(int(e["complete_contact_loss"]) for e in full_experts),
    })
    return full_experts, logs, rollout_counter[0], wall


def finalize(env, table, book48, audit, full_experts, args, checksum):
    thr_pack = audit["thr_pack"]
    success_rows = audit["success_rows"]
    seven_rows = audit["seven_rows"]
    fail_rows = audit["fail_rows"]

    # videos: seven prior, best new LC, best HM partials, representative fails
    print("Rendering videos...", flush=True)
    vid_dir = OUT / "videos"
    for r in seven_rows:
        snap = Path(r["snapshot"])
        blob = load_blob(snap)
        toks = json.loads(r["tokens"])
        overlay = {
            "ic_id": r["ic_id"],
            "hand_max_disp_m": r["hand_max_disp_m"],
            "SUCCESS_NEW": r.get("SUCCESS_NEW", 1),
        }
        render_expert_video(
            env, blob, toks, table, book48,
            vid_dir / f"prior_{r['ic_id']}_success.mp4", overlay,
        )

    # best new experts
    ranked = sorted(full_experts, key=lambda e: (-int(e["large_chase"]), -float(e["hand_max_disp_m"] or 0)))
    for e in ranked[:5]:
        blob = load_blob(e["snapshot"])
        toks = json.loads(e["tokens"])
        render_expert_video(
            env, blob, toks, table, book48,
            vid_dir / f"new_{e['ic_id']}_LC{e['large_chase']}_HM{e['high_motion']}.mp4",
            {"ic_id": e["ic_id"], "hand_max_disp_m": e["hand_max_disp_m"], "SUCCESS_NEW": 1},
        )

    # representative failed interception (plausible but not solved)
    solved = {e["ic_id"] for e in full_experts}
    fails_vid = [r for r in fail_rows if int(r.get("plausible_interception") or 0) and r["ic_id"] not in solved and r.get("snapshot")]
    for r in fails_vid[:3]:
        blob = load_blob(r["snapshot"])
        seq = [17] * 16 + [30] * 4 + [0] * 6 + [12] * 6
        render_expert_video(
            env, blob, seq, table, book48,
            vid_dir / f"fail_{r['ic_id']}_class{r.get('failure_class')}.mp4",
            {"ic_id": r["ic_id"], "hand_max_disp_m": r.get("max_hand_disp_legal_m"), "SUCCESS_NEW": 0},
        )

    # unified manifest update
    base = list(csv.DictReader(UNIFIED_BASE.open(encoding="utf-8")))
    existing_roots = {r["root_episode"] for r in base}
    existing_ics = {r["ic_id"] for r in base}
    added = []
    for e in full_experts:
        if e["root_episode"] in existing_roots:
            continue
        added.append(e)
        base.append({
            "ic_id": e["ic_id"], "root_episode": e["root_episode"],
            "source_dataset": "k49_large_chase_dynamic_recovery",
            "source_path": str(OUT / "FULL_EXPERTS.csv"),
            "SUCCESS_NEW": 1,
            "win_mean_theta": e["win_mean_theta"], "win_mean_pos": e["win_mean_pos"],
            "hold_both_s": e["hold_both_s"], "expert_type": e["expert_type"],
            "n_tokens": e["n_tokens"], "tokens_hash": e["tokens_hash"],
            "tokens": e["tokens"], "provenance": "large_chase_dynamic",
            "snapshot": e["snapshot"],
        })
    write_csv(OUT / "UPDATED_UNIFIED_MANIFEST.csv", base)

    # stats for report
    seven_disp = [float(r["hand_max_disp_m"]) for r in seven_rows]
    seven_rec = [float(r["disp_to_recatch_m"]) for r in seven_rows]
    n_exist_lc = sum(int(r["large_chase"]) for r in success_rows)
    n_plaus = sum(int(r.get("plausible_interception") or 0) for r in fail_rows)
    n_new_lc = sum(int(e["large_chase"]) for e in full_experts)
    n_new_hm = sum(int(e["high_motion"]) for e in full_experts)
    n_new_cl = sum(int(e["complete_contact_loss"]) for e in full_experts)
    n_new_roots = len(added)

    # bil capture but pose fail
    pose_fail_ics = sorted({
        r["ic_id"] for r in csv.DictReader((OUT / "FAILURE_ANALYSIS.csv").open(encoding="utf-8"))
        if Path(OUT / "FAILURE_ANALYSIS.csv").stat().st_size > 0
    }) if (OUT / "FAILURE_ANALYSIS.csv").exists() and (OUT / "FAILURE_ANALYSIS.csv").stat().st_size > 10 else []
    # better: from fail_rows class E + capture search non-NEW
    e_pose = [r["ic_id"] for r in fail_rows if r.get("failure_class") == "E"]
    try:
        cap = list(csv.DictReader((OUT / "CAPTURE_SEARCH.csv").open(encoding="utf-8")))
        bil_pose_fail = sorted({r["ic_id"] for r in cap if int(r.get("SUCCESS_NEW") or 0) == 0 and r.get("reason") in (
            "DROP_AFTER_HANDOFF", "LOST_GRASP", "SUCCESS_12S", None, ""
        )})
    except Exception:
        bil_pose_fail = e_pose

    usage = json.loads((OUT / "RAW" / "PHASE_B_USAGE.json").read_text(encoding="utf-8")) if (OUT / "RAW" / "PHASE_B_USAGE.json").exists() else {}

    # short vs sustained: compare intercept success rates from INTERCEPTION_SEARCH
    short, long_ = [], []
    short_rate = long_rate = 0.0
    sustained_helped = False
    try:
        inter = list(csv.DictReader((OUT / "INTERCEPTION_SEARCH.csv").open(encoding="utf-8")))
        short = [r for r in inter if int(r.get("short_prefix") or 0) == 1]
        long_ = [r for r in inter if int(r.get("short_prefix") or 0) == 0]

        def rate(rows):
            if not rows:
                return 0.0
            return sum(int(r.get("SUCCESS_NEW") or 0) for r in rows) / len(rows)

        short_rate, long_rate = rate(short), rate(long_)
        sustained_helped = long_rate > short_rate
    except Exception:
        pass

    report = f"""# Large-Displacement Dynamic Chase-and-Recatch

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**SUCCESS_NEW locked. No SFT/GRPO.**  
**HIGH_MOTION unchanged:** v≥{HM_THR['v_rel_norm_mps']} / w≥{HM_THR['w_rel_norm_radps']} / v_lat≥{HM_THR['v_lat_mps']}  
**LARGE_CHASE frozen:** max_disp≥{thr_pack['threshold_hand_max_disp_m']:.4f} m OR disp_to_recatch≥{thr_pack['threshold_disp_to_recatch_m']:.4f} m

## Budgets (recorded before Phase B)

| Item | Value |
|------|------:|
| Rollout budget | {TOTAL_ROLLOUT_BUDGET} |
| Wall limit (s) | {WALL_LIMIT_S} |
| Used rollouts | {usage.get('rollouts_used')} |
| Used wall (s) | {usage.get('wall_s')} |
| Split | 40% prior HM / 40% new impacts / 20% refine |

## Phase A summary

| Metric | Value |
|--------|------:|
| Existing SUCCESS_NEW traj audited | {len(success_rows)} |
| Existing LARGE_CHASE successes | {n_exist_lc} |
| HIGH_MOTION ICs audited | {len(fail_rows)} |
| Plausible interception (B–E) | {n_plaus} |

### Seven dynamic successes — hand motion (m)

| IC | max_disp | disp_to_recatch | obj_max_disp | pursuit |
|----|--------:|----------------:|-------------:|---------|
"""
    for r in seven_rows:
        report += (
            f"| {r['ic_id']} | {float(r['hand_max_disp_m']):.4f} | "
            f"{float(r['disp_to_recatch_m']):.4f} | {float(r['obj_max_disp_m']):.4f} | "
            f"{r.get('pursuit_vs_local')} |\n"
        )
    report += f"""
Seven mean max_disp = {float(np.mean(seven_disp)):.4f} m; max = {float(np.max(seven_disp)):.4f} m.

## Phase B yield

| Metric | Value |
|--------|------:|
| New FULL experts | {len(full_experts)} |
| New LARGE_CHASE FULL | {n_new_lc} / target {TARGET_LARGE_CHASE} |
| New HIGH_MOTION FULL | {n_new_hm} / target {TARGET_HIGH_MOTION} |
| New complete-contact-loss FULL | {n_new_cl} / target {TARGET_CONTACT_LOSS} |
| New independent roots added | {n_new_roots} |

## Q1–Q13

1. Seven prior hand max_disp (m): {[round(x,4) for x in seven_disp]}; mean={float(np.mean(seven_disp)):.4f}; to_recatch={[round(x,4) for x in seven_rec]}
2. Short rightward recatch representative? **{"YES" if float(np.mean(seven_disp)) < thr_pack["threshold_hand_max_disp_m"] else "MIXED"}** — mean max_disp {float(np.mean(seven_disp)):.4f} m vs LARGE_CHASE thr {thr_pack["threshold_hand_max_disp_m"]:.4f} m; local_correction count = {sum(1 for r in seven_rows if r.get("pursuit_vs_local")=="LOCAL_CORRECTION")}/7
3. Existing SUCCESS_NEW that are LARGE_CHASE: **{n_exist_lc}** / {len(success_rows)}
4. LARGE_CHASE threshold: **{thr_pack["threshold_hand_max_disp_m"]:.4f} m** (max disp) / **{thr_pack["threshold_disp_to_recatch_m"]:.4f} m** (to recatch). {thr_pack["interpretation"]}
5. Failed HIGH_MOTION with plausible interception: **{n_plaus}** / {len(fail_rows)}
6. Sustained translation vs short-prefix: short NEW-rate={short_rate:.4f} (n={len(short)}), long NEW-rate={long_rate:.4f} (n={len(long_)}); improved=**{sustained_helped}**
7. New LARGE_CHASE FULL: **{n_new_lc}**
8. New HIGH_MOTION FULL: **{n_new_hm}**
9. New complete-contact-loss chase-recatch FULL: **{n_new_cl}**
10. Bilateral capture but pose/HANDOFF fail (sample ICs): {bil_pose_fail[:20]}
11. Physical escape (class A) vs search-limited (B–E among unsolved): A={sum(1 for r in fail_rows if r.get("failure_class")=="A")}, B–E plausible unsolved={sum(1 for r in fail_rows if int(r.get("plausible_interception") or 0) and r["ic_id"] not in solved)}
12. New independent successful roots added to unified: **{n_new_roots}**
13. Final verified dataset: traj **{len(base)}**, ICs **{len({r["ic_id"] for r in base})}**, roots **{len({r["root_episode"] for r in base})}**

## Scripts
`scripts/expand_k49_large_chase_dynamic.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print("Wrote", OUT / "REPORT.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "all"], default="all")
    ap.add_argument("--budget-a", type=int, default=18)
    ap.add_argument("--budget-b", type=int, default=3)
    ap.add_argument("--per-ic-cap", type=int, default=100)
    ap.add_argument("--max-new-impacts", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    mapping_meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    env = make_training_env(device, model=model)

    # freeze budget note at start
    write_json(OUT / "RAW" / "RUN_CONFIG.json", {
        "CONFIG_ID": CONFIG_ID,
        "checksum": checksum,
        "HM_THR": HM_THR,
        "TOTAL_ROLLOUT_BUDGET": TOTAL_ROLLOUT_BUDGET,
        "WALL_LIMIT_S": WALL_LIMIT_S,
        "PHASE_FRAC": PHASE_FRAC,
        "phase": args.phase,
    })

    audit_path = OUT / "RAW" / "PHASE_A_AUDIT.pkl"
    if args.phase in ("A", "all"):
        audit = phase_a(env, table, book48, args)
        audit_path.write_bytes(pickle.dumps({
            "thr_pack": audit["thr_pack"],
            "success_rows": audit["success_rows"],
            "seven_rows": audit["seven_rows"],
            "fail_rows": audit["fail_rows"],
        }))
    else:
        audit = pickle.loads(audit_path.read_bytes())

    if args.phase in ("B", "all"):
        full_experts, logs, n_roll, wall = phase_b(
            env, table, book48, audit, args, checksum, mapping_meta,
        )
        finalize(env, table, book48, audit, full_experts, args, checksum)


if __name__ == "__main__":
    main()
