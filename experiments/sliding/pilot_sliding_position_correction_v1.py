"""Bounded sliding-based position correction + composite expert pilot.

Locked working SUCCESS_NEW: 10 deg / 12 mm hold in final 2s + SUCCESS_12S.
No physics/reward/K48/obs changes. No SFT/GRPO.
"""

from __future__ import annotations

import argparse
import csv
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import AIRBORNE_V3, grip_tau, map_action
from data_generation.recording.trajectory_recorder import save_pair
from envs.contacts.frames import extract_contacts
from envs.outcomes.horizon_outcome import recovery_failed, terminal_reason
from envs.outcomes.table_drop import object_vz, observe
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.full_objective import (
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    evaluate_tokens,
    guide_score_after_prefix,
)
from evaluation.grasp_configuration import cylinder_axis_hand, measure, p_com_hand
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.recovery_episode import make_training_env

OUT = ROOT / "results" / "expert_generation" / "sliding_position_correction_v1"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
SEED = 20261018
RECOVERY_LIMIT_S = 4.0
MAX_TOKENS = int(RECOVERY_LIMIT_S / 0.02)

CPP_HARD263 = [4] * 20 + [12] * 5 + [0] * 3
CPP_HRV1 = [6] * 20 + [12] * 15 + [0] * 8

# POSE_ONLY / priority ICs from prior pilot + CPP seeds
IC_SPECS = [
    {
        "id": "hard_910263",
        "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910263.pkl",
        "priority": "primary",
        "cpp": CPP_HARD263,
        "short": [32, 32],
        "pose_only_prefix": [32, 32, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0],
    },
    {
        "id": "hrv1_920185",
        "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920185.pkl",
        "priority": "pose_only",
        "short": [30, 30],
        "pose_only_prefix": [30, 30, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0],
    },
    {
        "id": "hrv1_920193",
        "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920193.pkl",
        "priority": "pose_only",
        "short": [32],
        "pose_only_prefix": [32, 35, 6, 6, 6, 6, 6, 6, 6, 6, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0],
    },
    {
        "id": "hard_910277",
        "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910277.pkl",
        "priority": "pose_only",
        "short": [44, 44],
        "pose_only_prefix": [44, 44, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0],
    },
    {
        "id": "hrv1_leftover_hard_910318",
        "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_leftover_hard_910318.pkl",
        "priority": "pose_only",
        "short": [30],
        "pose_only_prefix": [30, 6, 6, 6, 6, 6, 6, 6, 6, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0],
    },
    {
        "id": "hrv1_920154",
        "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920154.pkl",
        "priority": "pose_only",
        "short": [32, 32],
        "pose_only_prefix": [32, 32, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0],
    },
    {
        "id": "hrv1_920031",
        "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920031.pkl",
        "priority": "cpp_seed",
        "cpp": CPP_HRV1,
        "short": [30],
    },
    {
        "id": "hard_910101",
        "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910101.pkl",
        "priority": "known_full",
        "short": [32],
        "known_full": [32, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0],
    },
    {
        "id": "gen_hrv1_920031_1_74e897d20146",
        "path": "results/diagnostics/large_tilt_eccentric_grasp_mining/snapshots/gen_hrv1_920031_1_74e897d20146.pkl",
        "priority": "recatch",
        "short": [36, 40],
        "recatch": [4] * 12 + [36] * 5 + [0] * 8,
    },
    {
        "id": "hrv1_920131",
        "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920131.pkl",
        "priority": "position_only",
        "short": [41],
    },
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


def load_blob(path: Path) -> dict:
    blob = pickle.loads(path.read_bytes())
    assert "capture" in blob and "ref" in blob
    return blob


def decompose_pose(sim) -> dict:
    """Hand-frame position error into axial / lateral vs cylinder axis."""
    m = measure(sim, U_TGT, P_TGT)
    p = np.asarray(m["p_c_h"], float)
    u = np.asarray(m["u_c_h"], float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    # axis-symmetric: align u with target
    ut = U_TGT / max(float(np.linalg.norm(U_TGT)), 1e-12)
    if float(np.dot(u, ut)) < 0:
        u = -u
    err = p - P_TGT
    e_ax = float(np.dot(err, u))
    e_lat_vec = err - e_ax * u
    e_lat = float(np.linalg.norm(e_lat_vec))
    rows, _ = extract_contacts(sim)
    cL = [r for r in rows if r["side"] == "L"]
    cR = [r for r in rows if r["side"] == "R"]
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)

    def grasp_s(contacts):
        # signed distance of contact along cylinder axis from COM (object frame Z)
        Ro = np.asarray(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
        axis_w = Ro @ np.array([0.0, 0.0, 1.0])
        axis_w = axis_w / max(float(np.linalg.norm(axis_w)), 1e-12)
        if not contacts:
            return None
        ss = [float(np.dot(np.asarray(r["pos_w"], float) - po, axis_w)) for r in contacts]
        return float(np.mean(ss))

    return {
        "e_theta_deg": m["e_theta_deg"],
        "e_pos_mm": m["e_pos_mm"],
        "p_c_h_mm": (p * 1e3).tolist(),
        "p_target_h_mm": (P_TGT * 1e3).tolist(),
        "err_h_mm": (err * 1e3).tolist(),
        "u_c_h": u.tolist(),
        "e_axial_mm": e_ax * 1e3,
        "e_lateral_mm": e_lat * 1e3,
        "e_lateral_vec_h_mm": (e_lat_vec * 1e3).tolist(),
        "axial_fraction": abs(e_ax) / max(float(np.linalg.norm(err)), 1e-12),
        "nL": len(cL),
        "nR": len(cR),
        "bilateral": bool(cL and cR),
        "grasp_s_L_m": grasp_s(cL),
        "grasp_s_R_m": grasp_s(cR),
        "v_rel_h": m["v_rel_h"],
        "w_rel_h": m["w_rel_h"],
        "desired_slide_h": (-err).tolist(),  # reduce COM error in hand frame
        "desired_axial_h": ((-e_ax) * u).tolist(),
        "obj_z": m["obj_z"],
    }


def run_tokens_to_state(env, book, blob, tokens):
    """Execute tokens from IC; return mid-state capture + decomp (no HANDOFF)."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    for tok in tokens:
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return None
    elapsed = float(env.sim.data.time) - t0
    remaining = RECOVERY_LIMIT_S - elapsed
    cap = take_capture(env)
    return {
        "capture": cap,
        "ref": blob["ref"],
        "elapsed": elapsed,
        "remaining": remaining,
        "decomp": decompose_pose(env.sim),
        "tokens": list(tokens),
        "prev": np.asarray(env._prev, np.float32).copy() if env._prev is not None else np.zeros(7, np.float32),
    }


def tokens_aligned_to_dir(book, desire_h, grip_set, top_k=8):
    """Rank motion tokens by hand-frame translation alignment with desire_h."""
    d = np.asarray(desire_h, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return []
    d = d / n
    ranked = []
    for i in range(47):
        a = np.asarray(book[i], float)
        a6 = float(a[6])
        if grip_set == "mid" and abs(a6 - (-0.48)) > 0.05:
            continue
        if grip_set == "secure" and a6 < 0.5:
            continue
        if grip_set == "open" and a6 > -0.5:
            continue
        if grip_set == "any":
            pass
        # approximate v_hand from codebook (identity R for ranking)
        vh = np.array([a[0] * 1.0, a[1] * 1.0, (a[2] * 3.8 if a[2] >= 0 else a[2] * 1.0)], float)
        speed = float(np.linalg.norm(vh))
        if speed < 0.02:
            continue
        align = float(np.dot(vh / speed, d))
        ranked.append((align * speed, i, align, speed, float(a6)))
    ranked.sort(key=lambda x: -x[0])
    return ranked[:top_k]


def slide_proposals(book, desire_h, rng, max_n=60):
    """Build legal slide/reposition candidate suffixes."""
    seqs = []
    mid_toks = [t for _, t, *_ in tokens_aligned_to_dir(book, desire_h, "mid", top_k=6)]
    sec_toks = [t for _, t, *_ in tokens_aligned_to_dir(book, desire_h, "secure", top_k=8)]
    open_toks = [t for _, t, *_ in tokens_aligned_to_dir(book, desire_h, "open", top_k=6)]
    # pure mid (tok12) + translation secure/open mixes
    # Family S1: mid unload hold + aligned translation under mid (if any mid-trans tokens)
    # Most mid tokens are only tok12 (zero translation). So: alternate 12 with secure translate.
    for t in sec_toks[:6]:
        for n in (5, 8, 12, 16):
            for n_u in (3, 5, 8):
                seqs.append([12] * n_u + [t] * n + [0] * 5)
                seqs.append([12] * n_u + [t, 12] * (n // 2) + [0] * 5)
                seqs.append([t] * n + [12] * 5 + [0] * 5)  # secure translate then mid
    # Family S2: open briefly then secure translate (recatch-ish reposition)
    for o in (36, 39, 30)[:3]:
        for t in sec_toks[:4]:
            seqs.append([o] * 2 + [t] * 10 + [0] * 5)
            seqs.append([o] * 3 + [12] * 3 + [t] * 8 + [0] * 5)
    # Family S3: inertial secure jerk along desire then brake
    for t in sec_toks[:5]:
        seqs.append([t] * 15 + [0] * 8)
        seqs.append([t] * 8 + [14] * 5 + [0] * 5)
    # Family S4: opposite direction control (in case desire sign wrong)
    opp = (-np.asarray(desire_h, float)).tolist()
    for t in [x for _, x, *_ in tokens_aligned_to_dir(book, opp, "secure", top_k=4)]:
        seqs.append([12] * 5 + [t] * 10 + [0] * 5)
    # Family S5: hand-z dominant known tokens
    for t in (33, 8, 15, 32, 42, 44, 1, 13, 3):
        seqs.append([12] * 5 + [t] * 10 + [0] * 5)
        seqs.append([t] * 12 + [0] * 5)
    # mutations
    base_pool = list(seqs)
    rng.shuffle(base_pool)
    for b in base_pool[:20]:
        s = list(b)
        if len(s) > 4:
            s[int(rng.integers(0, len(s)))] = int(rng.integers(0, 47))
        seqs.append(s)
    # dedupe
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s[:MAX_TOKENS])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    rng.shuffle(uniq)
    return uniq[:max_n]


def step_legal_traced(sim, action):
    action = np.clip(np.asarray(action, np.float32).reshape(7), -1.0, 1.0)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(action, rd, rh, AIRBORNE_V3)
    vel, ang, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.v_cmd = vel.copy()
    sim.fsm.w_cmd = ang.copy()
    sim.fsm.v_des = vel.copy()
    sim.fsm.w_des = ang.copy()
    sim.fsm.fg_cmd = float(-tau)
    dt = float(sim.model.opt.timestep)
    rows = []
    for _ in range(N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, dt)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
        )
        vz_before = object_vz(sim)
        mujoco.mj_step(sim.model, sim.data)
        observe(sim, vz_before)
        m = measure(sim, U_TGT, P_TGT)
        cs = contact_summary(sim)
        p = np.asarray(m["p_c_h"], float)
        u = np.asarray(m["u_c_h"], float)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        rows.append(
            {
                "p_c_h": p.tolist(),
                "u_c_h": u.tolist(),
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "nL": cs["nL"],
                "nR": cs["nR"],
                "v_rel_h": m["v_rel_h"],
                "w_rel_h": m["w_rel_h"],
                "tau": tau,
            }
        )
    return action, rows


def classify_slide_phase(sub_rows, p0, u0) -> dict:
    """Distinguish sliding vs pivoting vs release from substep traces."""
    if not sub_rows:
        return {"label": "NO_DATA", "dp_mm": 0, "dp_axial_mm": 0, "frac_bi": 0, "zero_frac": 1}
    p0 = np.asarray(p0, float)
    u0 = np.asarray(u0, float)
    u0 = u0 / max(float(np.linalg.norm(u0)), 1e-12)
    ps = np.array([r["p_c_h"] for r in sub_rows], float)
    us = np.array([r["u_c_h"] for r in sub_rows], float)
    bi = np.array([1 if r["nL"] > 0 and r["nR"] > 0 else 0 for r in sub_rows])
    zero = np.array([1 if r["nL"] + r["nR"] == 0 else 0 for r in sub_rows])
    dp = ps[-1] - ps[0]
    dp_ax = float(np.dot(dp, u0))
    dp_lat = float(np.linalg.norm(dp - dp_ax * u0))
    # orientation change
    def ang(a, b):
        a = a / max(float(np.linalg.norm(a)), 1e-12)
        b = b / max(float(np.linalg.norm(b)), 1e-12)
        return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))

    d_theta = ang(us[0], us[-1])
    # rotation-only COM motion estimate: if fixed contact, COM arc ~ R*dtheta
    # cylinder radius ~0.018; if d_theta small and dp_ax large -> sliding
    frac_bi = float(bi.mean())
    zero_frac = float(zero.mean())
    max_zero_streak = 0
    cur = 0
    for z in zero:
        if z:
            cur += 1
            max_zero_streak = max(max_zero_streak, cur)
        else:
            cur = 0
    # each sub ~2ms
    max_zero_ms = max_zero_streak * 2.0
    dp_norm = float(np.linalg.norm(dp)) * 1e3
    label = "AMBIGUOUS"
    if max_zero_ms >= 40 and dp_norm >= 3:
        label = "RELEASE_RECATCH_LIKE"
    elif zero_frac > 0.15:
        label = "PARTIAL_LOSS_OR_SLIP"
    elif frac_bi >= 0.85 and abs(dp_ax) * 1e3 >= 3 and d_theta < 8:
        label = "CONTACT_PRESERVING_SLIDING"
    elif frac_bi >= 0.85 and d_theta >= 8 and abs(dp_ax) * 1e3 < 3:
        label = "PIVOT_LIKE_LITTLE_AXIAL_SLIDE"
    elif frac_bi >= 0.7 and dp_norm >= 3 and d_theta >= 5:
        label = "COUPLED_SLIDE_PIVOT"
    elif dp_norm < 2:
        label = "NO_MEANINGFUL_TRANSLATION"
    return {
        "label": label,
        "dp_mm": dp_norm,
        "dp_axial_mm": abs(dp_ax) * 1e3,
        "dp_lateral_mm": dp_lat * 1e3,
        "d_theta_deg": d_theta,
        "frac_bi": frac_bi,
        "zero_frac": zero_frac,
        "max_zero_ms": max_zero_ms,
        "signed_dp_axial_mm": dp_ax * 1e3,
    }


def run_prefix_traced(env, book, blob, tokens):
    restore_capture(env, blob["capture"])
    lost_low = 0
    all_sub = []
    m0 = measure(env.sim, U_TGT, P_TGT)
    p0 = m0["p_c_h"]
    u0 = m0["u_c_h"]
    for tok in tokens:
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        action, rows = step_legal_traced(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        all_sub.extend(rows)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {"ok": False, "reason": terminal_reason(env.sim) or "DROP", "sub": all_sub, "slide": classify_slide_phase(all_sub, p0, u0)}
    m1 = measure(env.sim, U_TGT, P_TGT)
    return {
        "ok": True,
        "m0": m0,
        "m1": m1,
        "sub": all_sub,
        "slide": classify_slide_phase(all_sub, p0, u0),
        "decomp1": decompose_pose(env.sim),
    }


def composite_candidates(prefix, book, desire, rng, max_n=50):
    """prefix (e.g. CPP) + sliding suffixes, and alternate orders."""
    out = []
    slides = slide_proposals(book, desire, rng, max_n=40)
    for s in slides:
        out.append(list(prefix) + list(s))
        # slide then remaining brake already in s
    # short then slide
    return out[:max_n]


def save_traj(stem, res, meta):
    if not res.get("observations"):
        return
    folder = OUT / "trajectories"
    cont, tokp = save_pair(
        folder, stem, res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    write_json(folder / f"{stem}_meta.json", {**meta, "continuous": str(cont.relative_to(ROOT)), "tokens_path": str(tokp.relative_to(ROOT))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-ics", type=int, default=10)
    parser.add_argument("--guide-budget", type=int, default=55)
    parser.add_argument("--terminal-budget", type=int, default=16)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("RAW", "SNAPSHOTS", "VIDEOS", "trajectories"):
        (OUT / sub).mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    # cost
    demo = load_blob(ROOT / IC_SPECS[0]["path"])
    t0 = time.perf_counter()
    guide_score_after_prefix(env, book, demo, CPP_HARD263[:5])
    t_g = time.perf_counter() - t0
    t0 = time.perf_counter()
    evaluate_tokens(env, book, demo, [32, 32])
    t_f = time.perf_counter() - t0
    write_json(OUT / "RAW" / "search_budget.json", {
        "guide_s": t_g, "full_s": t_f, "guide_budget": args.guide_budget,
        "terminal_budget": args.terminal_budget, "seed": args.seed,
        "working_theta_deg": WORKING_THETA_DEG, "working_pos_mm": WORKING_POS_MM,
    })
    print("COST", round(t_g, 3), round(t_f, 3), flush=True)

    # grip audit
    grip_rows = []
    for i in range(47):
        a = np.asarray(book[i], float)
        grip_rows.append({"token": i, "a6": float(a[6]), "tau": grip_tau(a[6]),
                          "vh_approx": [float(a[0]), float(a[1]), float(a[2] * (3.8 if a[2] >= 0 else 1.0))]})
    write_csv(OUT / "RAW" / "k48_token_grip_map.csv", grip_rows)

    decomp_rows = []
    search_rows = []
    full_experts = []
    failures = []
    slide_verified = []

    specs = IC_SPECS[: args.max_ics]
    for spec in specs:
        ic = spec["id"]
        path = ROOT / spec["path"]
        if not path.exists():
            print("MISSING", ic, flush=True)
            continue
        print("IC", ic, flush=True)
        blob = load_blob(path)
        dest = OUT / "SNAPSHOTS" / f"{ic}.pkl"
        if not dest.exists():
            shutil.copy2(path, dest)

        # --- decomposition at IC and after key prefixes ---
        restore_capture(env, blob["capture"])
        d0 = decompose_pose(env.sim)
        d0_row = {"ic_id": ic, "stage": "t0", **{k: d0[k] for k in d0 if k != "desired_slide_h"}}
        d0_row["desired_slide_h"] = d0["desired_slide_h"]
        decomp_rows.append(d0_row)

        prefixes = {}
        if spec.get("cpp"):
            prefixes["cpp"] = spec["cpp"]
        if spec.get("short"):
            prefixes["short"] = spec["short"]
        if spec.get("pose_only_prefix"):
            prefixes["pose_only"] = spec["pose_only_prefix"]
        if spec.get("recatch"):
            prefixes["recatch"] = spec["recatch"]
        if spec.get("known_full"):
            prefixes["known_full"] = spec["known_full"]

        mid_states = {}
        for name, toks in prefixes.items():
            mid = run_tokens_to_state(env, book, blob, toks)
            if mid is None:
                decomp_rows.append({"ic_id": ic, "stage": f"{name}_FAILED", "e_theta_deg": None, "e_pos_mm": None})
                continue
            mid_states[name] = mid
            d = mid["decomp"]
            decomp_rows.append({
                "ic_id": ic, "stage": name, "elapsed_s": mid["elapsed"], "remaining_s": mid["remaining"],
                **{k: d[k] for k in ("e_theta_deg", "e_pos_mm", "e_axial_mm", "e_lateral_mm", "axial_fraction",
                                      "nL", "nR", "bilateral", "grasp_s_L_m", "grasp_s_R_m")},
                "err_h_mm": d["err_h_mm"], "desired_slide_h": d["desired_slide_h"],
            })
            write_json(OUT / "RAW" / f"{ic}_{name}_pre_handoff.json", {
                "tokens": toks, "elapsed": mid["elapsed"], "remaining": mid["remaining"], "decomp": d,
            })

        # baselines full eval
        attempted = []
        best = None

        def consider(res, source, slide_meta=None):
            nonlocal best
            attempted.append({
                "source": source, "label": res["label"], "s12": res["success_12s"], "new": res["SUCCESS_NEW"],
                "win_th": res["window"].get("mean_theta"), "win_pos": res["window"].get("mean_pos"),
                "tokens": res["tokens"], "rec_s": res["recovery_elapsed"],
                "slide": None if not slide_meta else slide_meta,
            })
            if res["SUCCESS_NEW"]:
                score = 1e6 - float(res["window"]["mean_theta"]) - 0.05 * float(res["window"]["mean_pos"])
            elif res["success_12s"]:
                # prefer lower pos among good orientation
                th = float(res["window"]["mean_theta"] or 180)
                po = float(res["window"]["mean_pos"] or 1e3)
                score = 1e3 - (0 if th <= WORKING_THETA_DEG else 50) - po - 0.2 * th
            else:
                score = -1e3
            if best is None or score > best[0]:
                best = (score, res, source, slide_meta)

        # Counterfactuals A/D style
        consider(evaluate_tokens(env, book, blob, []), "A_immediate")
        if "short" in prefixes:
            consider(evaluate_tokens(env, book, blob, prefixes["short"]), "D_short")
        if "cpp" in prefixes:
            consider(evaluate_tokens(env, book, blob, prefixes["cpp"]), "A_cpp")
        if "pose_only" in prefixes:
            consider(evaluate_tokens(env, book, blob, prefixes["pose_only"]), "pose_only_prefix")
        if "recatch" in prefixes:
            consider(evaluate_tokens(env, book, blob, prefixes["recatch"]), "F_recatch")
        if "known_full" in prefixes:
            consider(evaluate_tokens(env, book, blob, prefixes["known_full"]), "known_full")

        # Choose desire from best mid state with good orientation / bad position
        desire = d0["desired_slide_h"]
        base_prefix = prefixes.get("cpp") or prefixes.get("pose_only") or prefixes.get("short") or []
        if "cpp" in mid_states and mid_states["cpp"]["decomp"]["e_theta_deg"] < 12:
            desire = mid_states["cpp"]["decomp"]["desired_slide_h"]
            base_prefix = prefixes["cpp"]
            print("  desire_from cpp e_pos", round(mid_states["cpp"]["decomp"]["e_pos_mm"], 2),
                  "axial", round(mid_states["cpp"]["decomp"]["e_axial_mm"], 2),
                  "lat", round(mid_states["cpp"]["decomp"]["e_lateral_mm"], 2),
                  "remain", round(mid_states["cpp"]["remaining"], 2), flush=True)
        elif "pose_only" in mid_states:
            desire = mid_states["pose_only"]["decomp"]["desired_slide_h"]
            base_prefix = prefixes["pose_only"]
            print("  desire_from pose_only e_pos", round(mid_states["pose_only"]["decomp"]["e_pos_mm"], 2), flush=True)

        # remaining time check
        used = 0.02 * len(base_prefix)
        remain_tokens = max(0, MAX_TOKENS - len(base_prefix))
        print("  remain_tokens", remain_tokens, flush=True)

        # Build composite candidates
        cands = []
        # B: CPP → slide
        cands.extend([(list(base_prefix) + s, "B_cpp_slide") for s in slide_proposals(book, desire, rng, max_n=35)])
        # E: slide → pivot (tilt mid brake) then maybe more slide
        for s in slide_proposals(book, desire, rng, max_n=10):
            for tilt in (4, 6, 10):
                cands.append((list(s) + [tilt] * 10 + [12] * 8 + [0] * 5, "E_slide_pivot"))
        # F: recatch → reposition
        if "recatch" in prefixes:
            for s in slide_proposals(book, desire, rng, max_n=8):
                cands.append((list(prefixes["recatch"]) + list(s), "F_recatch_reposition"))
        # G: general mutations of base+slide
        for _ in range(15):
            s = list(base_prefix) + [int(rng.integers(0, 47)) for _ in range(int(rng.integers(5, 20)))] + [0] * 5
            cands.append((s[:MAX_TOKENS], "G_general"))
        # D: short → slide
        if "short" in prefixes:
            for s in slide_proposals(book, desire, rng, max_n=10):
                cands.append((list(prefixes["short"]) + list(s), "D_short_slide"))

        # trim by remaining length
        cands2 = []
        for seq, tag in cands:
            if len(seq) <= MAX_TOKENS:
                cands2.append((seq, tag))
        rng.shuffle(cands2)
        cands2 = cands2[: args.guide_budget]

        # guide rank by position reduction priority
        ranked = []
        for seq, tag in cands2:
            g = guide_score_after_prefix(env, book, blob, seq)
            if not g["ok"]:
                continue
            # prioritize low pos if theta already ok-ish
            score = -0.15 * g["e_theta"] - 1.0 * g["e_pos"] + (3.0 if g["bilateral"] else -15.0)
            if g["e_theta"] < WORKING_THETA_DEG + 5:
                score += 10.0
            if g["e_pos"] < WORKING_POS_MM:
                score += 20.0
            ranked.append((score, seq, tag, g))
        ranked.sort(key=lambda x: -x[0])

        term_used = 0
        for score, seq, tag, g in ranked[: args.terminal_budget]:
            # trace slide segment = suffix after base_prefix when applicable
            slide_meta = None
            if tag.startswith("B_") or tag.startswith("D_"):
                pref_len = len(base_prefix) if tag.startswith("B_") else len(prefixes.get("short") or [])
                suffix = seq[pref_len:]
                # restore after prefix then trace suffix
                mid = run_tokens_to_state(env, book, blob, seq[:pref_len])
                if mid is not None and suffix:
                    # build temp blob at mid for tracing suffix only
                    restore_capture(env, mid["capture"])
                    tr = run_prefix_traced(env, book, {"capture": mid["capture"], "ref": blob["ref"]}, suffix)
                    slide_meta = tr.get("slide")
                    if slide_meta and slide_meta.get("label") == "CONTACT_PRESERVING_SLIDING":
                        slide_verified.append({"ic_id": ic, "tag": tag, **slide_meta, "tokens_suffix": suffix})
            res = evaluate_tokens(env, book, blob, seq)
            consider(res, tag, slide_meta)
            term_used += 1
            search_rows.append({
                "ic_id": ic, "source": tag, "label": res["label"], "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
                "s12": int(res["success_12s"]),
                "win_theta": res["window"].get("mean_theta"), "win_pos": res["window"].get("mean_pos"),
                "guide_theta": g["e_theta"], "guide_pos": g["e_pos"],
                "slide_label": None if not slide_meta else slide_meta.get("label"),
                "slide_dp_axial_mm": None if not slide_meta else slide_meta.get("dp_axial_mm"),
                "n_tokens": len(seq), "tokens": json.dumps(seq),
            })
            if res["SUCCESS_NEW"]:
                print("  FULL_EXPERT", tag, "th", round(res["window"]["mean_theta"], 2),
                      "pos", round(res["window"]["mean_pos"], 2), flush=True)
                # local refine
                for _ in range(3):
                    mut = list(seq)
                    i = int(rng.integers(max(0, len(mut) - 15), len(mut)))
                    mut[i] = int(rng.integers(0, 47))
                    rr = evaluate_tokens(env, book, blob, mut)
                    consider(rr, tag + "_refine", slide_meta)
                break

        write_json(OUT / "RAW" / f"{ic}_attempts.json", {"attempted": attempted})

        if best:
            res, src, sm = best[1], best[2], best[3]
            row = {
                "ic_id": ic, "source": src, "label": res["label"], "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
                "success_12s": int(res["success_12s"]),
                "win_mean_theta": res["window"].get("mean_theta"),
                "win_mean_pos": res["window"].get("mean_pos"),
                "hold_both_s": res["window"].get("longest_both_ok_s"),
                "recovery_s": res["recovery_elapsed"],
                "n_tokens": len(res["tokens"]),
                "tokens": json.dumps(res["tokens"]),
                "slide_label": None if not sm else sm.get("label"),
                "slide_dp_axial_mm": None if not sm else sm.get("dp_axial_mm"),
            }
            # replay + save if S12
            if res["success_12s"]:
                full = evaluate_tokens(env, book, blob, res["tokens"], record_obs=True)
                full["label"] = res["label"]
                full["SUCCESS_NEW"] = res["SUCCESS_NEW"]
                replay = evaluate_tokens(env, book, blob, res["tokens"])
                row["replay_ok"] = int(replay["SUCCESS_NEW"] == res["SUCCESS_NEW"] and replay["success_12s"] == res["success_12s"])
                save_traj(f"{ic}__{src}", full, {
                    "ic_id": ic, "source": src, "label": res["label"], "SUCCESS_NEW": res["SUCCESS_NEW"],
                    "seed": args.seed, "snapshot": str(dest.relative_to(ROOT)),
                    "mechanism_diag": None if not sm else sm.get("label"),
                    "working_theta_deg": WORKING_THETA_DEG, "working_pos_mm": WORKING_POS_MM,
                })
                if res["SUCCESS_NEW"]:
                    full_experts.append(row)
                else:
                    failures.append({**row, "note": "best not full expert"})
            else:
                row["replay_ok"] = 0
                failures.append({**row, "note": res["reason"]})
            search_rows.append({**row, "guide_theta": None, "guide_pos": None})

    write_csv(OUT / "POSITION_ERROR_DECOMPOSITION.csv", decomp_rows)
    write_csv(OUT / "COMPOSITE_SEARCH_RESULTS.csv", search_rows)
    write_csv(OUT / "FULL_EXPERTS.csv", full_experts)
    write_json(OUT / "RAW" / "slide_verified_events.json", slide_verified)
    write_json(OUT / "RAW" / "failures.json", failures)

    # markdown notes will be written by follow-up; dump summary
    summary = {
        "seed": args.seed,
        "n_ics": len(specs),
        "n_full_experts": len(full_experts),
        "n_verified_cps_events": sum(1 for s in slide_verified if s.get("label") == "CONTACT_PRESERVING_SLIDING"),
        "full_expert_ids": [r["ic_id"] for r in full_experts],
        "working_theta_deg": WORKING_THETA_DEG,
        "working_pos_mm": WORKING_POS_MM,
    }
    write_json(OUT / "summary.json", summary)
    print("PILOT_DONE", OUT, "full", len(full_experts), "cps_events", summary["n_verified_cps_events"], flush=True)


if __name__ == "__main__":
    main()
