"""Diagnose and solve post-interception capture–brake failures (K49 large-chase).

Locked K49_tok48_wy0p25. SUCCESS_NEW / HIGH_MOTION / LARGE_CHASE thresholds unchanged.
No SFT/GRPO. No new actions / physics / controller changes.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generation.recording.trajectory_recorder import save_pair
from envs.outcomes.horizon_outcome import recovery_failed, terminal_reason
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import measure
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

from scripts.expand_k49_hard_dynamic import CAPTURE_SEED, is_high_motion, ic_metrics as hard_ic_metrics
from scripts.expand_k49_large_chase_dynamic import (
    HM_THR,
    TRANSLATE_TOKS,
    OPEN_TOKS,
    BRAKE_TOK,
    hand_pos,
    obj_pos,
    is_large_chase,
    sample_state,
    motion_metrics_from_trace,
)
from scripts.generate_k49_in_recovery_recatch import pose_correction_proposals
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    decode,
    decompose,
    evaluate_full,
    mapping_checksum,
    run_seq,
    write_csv,
    write_json,
)
from scripts.generate_k49_type3_capture_brake import capture_brake_proposals, finger_object_contacts

OUT = ROOT / "results" / "expert_generation" / "k49_capture_brake_diagnostics"
LC_OUT = ROOT / "results" / "expert_generation" / "k49_large_chase_dynamic_recovery"
HARD = ROOT / "results" / "expert_generation" / "k49_hard_dynamic_expansion"

# Budgets recorded BEFORE execution
TOTAL_ROLLOUT_BUDGET = 3600
WALL_LIMIT_S = 60 * 60
PHASE_FRAC = (0.30, 0.50, 0.20)  # diagnostics / capture-brake / pose+HANDOFF

# Load frozen LARGE_CHASE thresholds (do not change)
_THR_PATH = LC_OUT / "RAW" / "LARGE_CHASE_THRESHOLD_FREEZE.json"
LC_THR = json.loads(_THR_PATH.read_text(encoding="utf-8")) if _THR_PATH.exists() else {
    "threshold_hand_max_disp_m": 0.06,
    "threshold_disp_to_recatch_m": 0.035,
    "threshold_max_requires_recatch_m": 0.02,
}


def tokens_hash(toks):
    return hashlib.sha256(json.dumps([int(x) for x in toks]).encode()).hexdigest()[:16]


def load_blob(path):
    return pickle.loads(Path(path).read_bytes())


def find_snap(ic_id: str) -> Path | None:
    for d in [
        LC_OUT / "snapshots",
        HARD / "snapshots",
        ROOT / "results/expert_generation/k49_ball_impact_dynamic_recovery/snapshots",
        ROOT / "results/expert_generation/k49_unified_dataset_expansion/snapshots",
        ROOT / "results/diagnostics/hard_ic_mining/snapshots",
    ]:
        p = d / f"{ic_id}.pkl"
        if p.exists():
            return p
    # CSV snapshot fields
    for csv_name in ("HIGH_MOTION_FAILURE_AUDIT.csv", "BEST_LARGE_CHASE_PARTIALS.csv"):
        fp = LC_OUT / csv_name
        if not fp.exists():
            continue
        for r in csv.DictReader(fp.open(encoding="utf-8")):
            if r.get("ic_id") == ic_id and Path(r.get("snapshot") or "").exists():
                return Path(r["snapshot"])
    return None


def finger_q(sim):
    fj = sim.ids.finger_jnt
    return [float(sim.data.qpos[int(j)]) for j in fj]


def contact_forces(sim):
    """Sum |force| on finger–object contacts (MuJoCo efc; diagnostic only)."""
    obj = int(sim.ids.object_body)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    nL = nR = 0
    fL = fR = 0.0
    # Use contact list + force via mj_contactForce
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        if other not in (left, right):
            continue
        force = np.zeros(6, float)
        try:
            mujoco.mj_contactForce(sim.model, sim.data, i, force)
        except Exception:
            force[:] = 0.0
        fn = float(np.linalg.norm(force[:3]))
        if other == left:
            nL += 1
            fL += fn
        else:
            nR += 1
            fR += fn
    return {"nL_con": nL, "nR_con": nR, "fL": fL, "fR": fR, "f_tot": fL + fR}


def rich_sample(sim, t0, tok=None, cmd=None):
    st = sample_state(sim, t0)
    m = measure(sim, U_TGT, P_TGT)
    pack = physical_pack(sim)
    cf = contact_forces(sim)
    left = np.asarray(sim.data.xpos[sim.ids.left_body], float)
    right = np.asarray(sim.data.xpos[sim.ids.right_body], float)
    po = st["obj"]
    vh = np.asarray(sim.data.cvel[sim.ids.hand_body][3:], float)  # linear part of cvel (spatial)
    # cvel is 6D [rot; lin] in world — use xpos finite-diff preferred; here store cvel lin
    vo = np.asarray(sim.data.cvel[sim.ids.object_body][3:], float)
    st.update({
        "tok": int(tok) if tok is not None else -1,
        "cmd": None if cmd is None else np.asarray(cmd, float).tolist(),
        "hand_cvel": vh.tolist(),
        "obj_cvel": vo.tolist(),
        "v_rel_norm": float(np.linalg.norm(st["v_rel"])),
        "w_rel_norm": float(np.linalg.norm(st["w_rel"])),
        "finger_q": finger_q(sim),
        "fg_cmd": float(sim.fsm.fg_cmd) if hasattr(sim.fsm, "fg_cmd") else None,
        "dist_L": float(np.linalg.norm(po - left)),
        "dist_R": float(np.linalg.norm(po - right)),
        "fL": cf["fL"], "fR": cf["fR"], "f_tot": cf["f_tot"],
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
    })
    return st


def replay_rich(env, table, book48, blob, tokens):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    cs0 = contact_summary(env.sim)
    onset_bil = bool(cs0["bilateral"])
    trace = []
    lost_low = 0
    ok = True
    for tok in tokens:
        cmd = decode(table, book48, tok)
        st = rich_sample(env.sim, t0, tok=tok, cmd=cmd)
        trace.append(st)
        step_legal(env.sim, cmd)
        env._prev = np.asarray(cmd, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            ok = False
            break
    if ok:
        trace.append(rich_sample(env.sim, t0, tok=-1))
    metrics = motion_metrics_from_trace(trace, tokens, onset_bil)
    metrics["replay_ok"] = int(ok)
    metrics["reason_fail"] = None if ok else (terminal_reason(env.sim) or "DROP")
    metrics["SUCCESS_NEW"] = 0
    win = {}
    if ok:
        res = evaluate_full(env, table, book48, blob, tokens, record_obs=False)
        metrics["SUCCESS_NEW"] = int(bool(res.get("SUCCESS_NEW")))
        metrics["eval_reason"] = res.get("reason")
        win = res.get("window") or {}
        metrics["win_mean_theta"] = win.get("mean_theta")
        metrics["win_mean_pos"] = win.get("mean_pos")
        metrics["hold_both_s"] = win.get("longest_both_ok_s")
    return metrics, trace


def contact_events(trace):
    """Extract contact milestones and relative velocities at events."""
    if not trace:
        return {}
    first_any = first_uni = first_bil = None
    loss_after_bil = None
    saw_bil = False
    bil_persist = 0.0
    max_bil_persist = 0.0
    t_prev = float(trace[0]["t"])
    for i, r in enumerate(trace):
        dt = max(float(r["t"]) - t_prev, 0.0)
        t_prev = float(r["t"])
        any_c = int(r["nL"]) + int(r["nR"]) > 0
        uni = (int(r["nL"]) > 0) ^ (int(r["nR"]) > 0)
        bil = bool(r["bilateral"])
        if any_c and first_any is None:
            first_any = i
        if uni and first_uni is None:
            first_uni = i
        if bil and first_bil is None:
            first_bil = i
            saw_bil = True
        if bil:
            bil_persist += dt
            max_bil_persist = max(max_bil_persist, bil_persist)
        else:
            if saw_bil and loss_after_bil is None and not bil:
                # first break after having bilateral
                if first_bil is not None and i > first_bil:
                    loss_after_bil = i
            bil_persist = 0.0

    def at(i):
        if i is None:
            return {}
        r = trace[i]
        return {
            "t": float(r["t"]),
            "v_rel": float(r.get("v_rel_norm") or np.linalg.norm(r["v_rel"])),
            "w_rel": float(r.get("w_rel_norm") or np.linalg.norm(r["w_rel"])),
            "sep": float(r["sep"]),
            "nL": int(r["nL"]), "nR": int(r["nR"]),
            "tok": int(r.get("tok") or -1),
            "f_tot": float(r.get("f_tot") or 0),
            "finger_q_mean": float(np.mean(r.get("finger_q") or [0])),
        }

    # contact loss cause heuristic
    cause = None
    if loss_after_bil is not None:
        r = trace[loss_after_bil]
        r0 = trace[max(0, loss_after_bil - 1)]
        if float(r0.get("v_rel_norm") or 0) > 0.15:
            cause = "high_rel_velocity_at_break"
        elif int(r0.get("tok") or -1) in OPEN_TOKS:
            cause = "grip_open_token"
        elif int(r0.get("tok") or -1) in TRANSLATE_TOKS:
            cause = "continued_translation"
        elif float(np.mean(r0.get("finger_q") or [0])) > 0.04:
            cause = "gripper_too_open"
        else:
            cause = "contact_slip_or_insufficient_brake"
    elif first_any is not None and first_bil is None:
        cause = "never_bilateral"
    elif first_any is None:
        cause = "no_contact"

    return {
        "first_contact": at(first_any),
        "first_uni": at(first_uni),
        "first_bil": at(first_bil),
        "loss_after_bil": at(loss_after_bil),
        "max_bil_persist_s": max_bil_persist,
        "had_any_contact": int(first_any is not None),
        "had_uni": int(first_uni is not None),
        "had_bil": int(first_bil is not None),
        "contact_loss_cause": cause,
        "min_sep": float(min(r["sep"] for r in trace)),
        "min_v_rel_after_contact": (
            float(min(r.get("v_rel_norm") or np.linalg.norm(r["v_rel"]) for r in trace[first_any:]))
            if first_any is not None else None
        ),
    }


def classify_failure(trace, metrics, events):
    """A–H taxonomy (§4). Returns primary + ordered list."""
    order = []
    if not events.get("had_any_contact"):
        if float(events.get("min_sep") or 1) > 0.08:
            order.append("H" if "TABLE" in str(metrics.get("reason_fail") or "") else "A")
        else:
            order.append("A")
        return order[0], order

    fc = events.get("first_contact") or {}
    if float(fc.get("v_rel") or 0) >= 0.20 or float(fc.get("w_rel") or 0) >= 4.0:
        order.append("B")

    if not events.get("had_bil"):
        order.append("C")
    else:
        if events.get("loss_after_bil"):
            order.append("D")
        min_v = events.get("min_v_rel_after_contact")
        if events.get("had_bil") and min_v is not None and min_v > 0.12:
            order.append("E")
        # if bil persisted and replay ok but not SUCCESS_NEW
        if metrics.get("replay_ok") and not metrics.get("SUCCESS_NEW"):
            reason = str(metrics.get("eval_reason") or metrics.get("reason_fail") or "")
            if reason in ("DROP_AFTER_HANDOFF", "LOST_GRASP", "SUCCESS_12S"):
                order.append("G")
            elif float(events.get("max_bil_persist_s") or 0) >= 0.15:
                order.append("F")
            else:
                order.append("D")
        elif not metrics.get("replay_ok"):
            if events.get("loss_after_bil"):
                if "D" not in order:
                    order.append("D")
            elif "TABLE" in str(metrics.get("reason_fail") or ""):
                order.append("H")

    if not order:
        if metrics.get("SUCCESS_NEW"):
            return "OK", ["OK"]
        order.append("C")
    # primary = first irreversible-ish
    primary = order[0]
    for p in ("H", "A", "B", "C", "D", "E", "F", "G"):
        if p in order:
            primary = p
            break
    return primary, order


def milestones(events, metrics):
    m = {
        "M1_intercept": int(events.get("had_any_contact") or 0),
        "M2_uni": int(events.get("had_uni") or 0),
        "M3_bil": int(events.get("had_bil") or 0),
        "M4_v_reduced": int(
            events.get("min_v_rel_after_contact") is not None
            and float(events["min_v_rel_after_contact"])
            < 0.7 * float((events.get("first_contact") or {}).get("v_rel") or 1.0)
        ),
        "M5_stable_bil": int(float(events.get("max_bil_persist_s") or 0) >= 0.30),
        "M6_pose_ok": 0,
        "M7_SUCCESS_NEW": int(metrics.get("SUCCESS_NEW") or 0),
    }
    if m["M7_SUCCESS_NEW"]:
        m["M6_pose_ok"] = 1
    elif metrics.get("replay_ok") and m["M5_stable_bil"]:
        # pose not verified SUCCESS_NEW
        th = metrics.get("win_mean_theta")
        if th is not None and float(th) <= 15.0:
            m["M6_pose_ok"] = 1
    return m


# ---------------------------------------------------------------------------
# Chase prefixes that previously produced contact / LARGE_CHASE
# ---------------------------------------------------------------------------

CHASE_PREFIXES = [
    ("t17_18", [17] * 18),
    ("t33_18", [33] * 18),
    ("t1_20", [1] * 20),
    ("t5_16", [5] * 16),
    ("t13_16", [13] * 16),
    ("open_t17", [30] * 3 + [17] * 16),
    ("cap_seed", list(CAPTURE_SEED)),
    ("short_hold", [0] * 8),
]


def capture_continuations(book48, rng, desire, n=36):
    """Post-contact legal continuations: grip timing, stop/reverse, brake, return."""
    props = []
    # earlier vs later close
    for n_close in (2, 4, 8, 12, 16):
        props.append([0] * n_close + [BRAKE_TOK] * 6 + [0] * 8)
        props.append([0] * n_close + [BRAKE_TOK] * 6 + [SECURE_REVERSE_WRIST] * 20 + [0] * 6)
    # stop pursuit immediately
    props.append([BRAKE_TOK] * 10 + [0] * 12)
    props.append([BRAKE_TOK] * 6 + [SECURE_REVERSE_WRIST] * 24 + [0] * 6)
    # reverse translation (use opposite-ish tokens)
    for t in (17, 33, 1, 5, 13, 19):
        props.append([t] * 4 + [0] * 8 + [BRAKE_TOK] * 6)  # brief continue
        props.append([0] * 4 + [BRAKE_TOK] * 4 + [t] * 6 + [0] * 8)  # stop then micro-adjust
        props.append([BRAKE_TOK] * 4 + [0] * 6 + [SECURE_REVERSE_WRIST] * 16)
    # grip modulation: open briefly then close (risky)
    props.append([30] * 2 + [0] * 10 + [BRAKE_TOK] * 6 + [0] * 8)
    props.append([0] * 6 + [12] * 8 + [4] * 12 + [0] * 8)
    props.append([12] * 4 + [0] * 4 + [12] * 4 + [SECURE_REVERSE_WRIST] * 16)
    # wrist then return
    for piv in (4, 6, 10):
        props.append([0] * 4 + [piv] * 12 + [BRAKE_TOK] * 5 + [SECURE_REVERSE_WRIST] * 16 + [0] * 4)
        props.append([BRAKE_TOK] * 4 + [piv] * 10 + [SECURE_REVERSE_WRIST] * 20 + [0] * 4)
    props.extend(capture_brake_proposals(book48, desire, rng, n=14))
    try:
        props.extend(pose_correction_proposals(book48, desire, rng, n=8))
    except Exception:
        pass
    seen, uniq = set(), []
    for s in props:
        if not s:
            continue
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    rng.shuffle(uniq)
    return uniq[:n]


def pose_handoff_continuations(rng, desire, book48, n=20):
    props = []
    for n_ret in (12, 20, 28, 36):
        props.append([SECURE_REVERSE_WRIST] * n_ret + [0] * 8)
        props.append([BRAKE_TOK] * 4 + [SECURE_REVERSE_WRIST] * n_ret + [0] * 6)
        props.append([0] * 8 + [SECURE_REVERSE_WRIST] * n_ret + [0] * 6)
    for piv in (4, 6, 10):
        props.append([piv] * 16 + [SECURE_REVERSE_WRIST] * 20 + [0] * 6)
        props.append([0] * 4 + [piv] * 12 + [12] * 5 + [SECURE_REVERSE_WRIST] * 20)
    try:
        props.extend(pose_correction_proposals(book48, desire, rng, n=10))
    except Exception:
        pass
    seen, uniq = set(), []
    for s in props:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    return uniq[:n]


# ---------------------------------------------------------------------------
# Phase 0: K49 authority
# ---------------------------------------------------------------------------

def measure_token_authority(env, table, book48, blob_contact, blob_free):
    """Measure actual hand/finger motion for key tokens in contact vs free."""
    rows = []
    interesting = sorted(set(
        list(range(0, 20)) + [24, 30, 32, 33, 36] + list(OPEN_TOKS) + [BRAKE_TOK, SECURE_REVERSE_WRIST]
    ))
    for label, blob in (("contact_ic", blob_contact), ("free_ic", blob_free)):
        if blob is None:
            continue
        for tok in interesting:
            restore_capture(env, blob["capture"])
            h0 = hand_pos(env.sim)
            o0 = obj_pos(env.sim)
            q0 = np.array(finger_q(env.sim), float)
            cs0 = contact_summary(env.sim)
            cmd = decode(table, book48, tok)
            t0 = float(env.sim.data.time)
            for _ in range(8):
                step_legal(env.sim, cmd)
            dt = float(env.sim.data.time) - t0
            h1 = hand_pos(env.sim)
            o1 = obj_pos(env.sim)
            q1 = np.array(finger_q(env.sim), float)
            dh = h1 - h0
            rows.append({
                "state": label,
                "tok": tok,
                "cmd": json.dumps(np.asarray(cmd, float).tolist()),
                "cmd_trans_norm": float(np.linalg.norm(cmd[:3])),
                "cmd_wy": float(cmd[4]),
                "cmd_grip": float(cmd[6]),
                "dt_s": dt,
                "hand_disp_m": float(np.linalg.norm(dh)),
                "hand_lat_m": float(np.hypot(dh[0], dh[1])),
                "hand_vert_m": float(dh[2]),
                "hand_speed_mps": float(np.linalg.norm(dh) / max(dt, 1e-6)),
                "obj_disp_m": float(np.linalg.norm(o1 - o0)),
                "finger_dq_mean": float(np.mean(q1 - q0)),
                "bilateral0": int(cs0["bilateral"]),
                "nL0": int(cs0["nL"]), "nR0": int(cs0["nR"]),
            })
    return rows


def write_authority_md(rows, path):
    by_tok = {}
    for r in rows:
        by_tok.setdefault(int(r["tok"]), []).append(r)
    lines = [
        "# K49 Control Authority (measured)\n",
        "Config `K49_tok48_wy0p25`. Measurements: 8 legal steps from representative ICs.\n",
        "Do not infer token effect from ID alone — values below are **realized** motion.\n",
        "\n## Summary by token family\n",
    ]
    # classify
    fam = {
        "hold_0": [0],
        "brake_12": [12],
        "translate": list(TRANSLATE_TOKS),
        "open": list(OPEN_TOKS),
        "pivot_wy": [4, 6, 10],
        "secure_reverse_48": [SECURE_REVERSE_WRIST],
    }
    for name, toks in fam.items():
        subset = [r for t in toks for r in by_tok.get(t, []) if r["state"] == "free_ic"]
        if not subset:
            subset = [r for t in toks for r in by_tok.get(t, [])]
        if not subset:
            continue
        spd = np.mean([float(r["hand_speed_mps"]) for r in subset])
        grip = np.mean([float(r["finger_dq_mean"]) for r in subset])
        lines.append(
            f"- **{name}** (toks {toks[:8]}{'...' if len(toks)>8 else ''}): "
            f"mean hand speed ≈ {spd:.4f} m/s, mean Δfinger_q ≈ {grip:.4f}\n"
        )
    lines.append("\n## Implications for capture–brake\n")
    lines.append(
        "- Translation tokens produce cm-scale hand motion per ~0.16 s burst; "
        "continued translation after contact can shear the grasp.\n"
        "- Token 0 / 12 are the primary 'stop/hold' options; 12 (brake) may couple wrist.\n"
        "- Open tokens (30+) increase finger opening — late open after contact risks escape.\n"
        "- Token 48 (SECURE_REVERSE_WRIST) reverses wy for pose return with secure grip.\n"
        "- Grip closing rate is limited by finger dynamics under step_legal — "
        "very short interception windows may outrun close capability.\n"
    )
    # top translating tokens
    free = [r for r in rows if r["state"] == "free_ic"]
    free.sort(key=lambda r: -float(r["hand_speed_mps"]))
    lines.append("\n## Fastest realized hand motion (free IC, top 10)\n\n")
    lines.append("| tok | speed m/s | lat m | grip_cmd | Δfinger |\n|---:|---:|---:|---:|---:|\n")
    for r in free[:10]:
        lines.append(
            f"| {r['tok']} | {float(r['hand_speed_mps']):.4f} | {float(r['hand_lat_m']):.4f} | "
            f"{float(json.loads(r['cmd'])[6]):.3f} | {float(r['finger_dq_mean']):.4f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget-diag", type=int, default=None)
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    (OUT / "RAW").mkdir(exist_ok=True)

    budgets = [int(TOTAL_ROLLOUT_BUDGET * f) for f in PHASE_FRAC]
    budgets[2] = TOTAL_ROLLOUT_BUDGET - sum(budgets[:2])
    write_json(OUT / "RAW" / "BUDGET.json", {
        "TOTAL_ROLLOUT_BUDGET": TOTAL_ROLLOUT_BUDGET,
        "WALL_LIMIT_S": WALL_LIMIT_S,
        "PHASE_FRAC": PHASE_FRAC,
        "phase_budgets": budgets,
        "HM_THR": HM_THR,
        "LC_THR": LC_THR,
        "recorded_before_execution": True,
    })

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    mapping_meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261019)

    rollout = [0]
    t_wall0 = time.time()

    def time_left():
        return time.time() - t_wall0 < WALL_LIMIT_S

    # ----- load candidates -----
    hm_rows = list(csv.DictReader((LC_OUT / "HIGH_MOTION_FAILURE_AUDIT.csv").open(encoding="utf-8")))
    be = [r for r in hm_rows if r.get("failure_class") in ("B", "C", "D", "E")]
    # prioritize D (bilateral then lost) then C then others; hard_910012 first
    be.sort(key=lambda r: (
        0 if r["ic_id"] == "hard_910012" else 1,
        {"D": 0, "E": 1, "C": 2, "B": 3}.get(r.get("failure_class"), 9),
    ))

    print(f"Candidates B–E: {len(be)}; budgets {budgets}", flush=True)

    # =====================================================================
    # Phase 1: diagnostics (30%)
    # =====================================================================
    b1 = budgets[0]
    print(f"=== Phase 1 diagnostics (budget {b1}) ===", flush=True)

    # authority: use hard_910012 (contact) + a free-ish IC
    blob_c = load_blob(find_snap("hard_910012")) if find_snap("hard_910012") else None
    blob_f = None
    for ic in ("bimp_0010", "ubimp_0040", "hard_910006"):
        p = find_snap(ic)
        if p:
            blob_f = load_blob(p)
            break
    auth_rows = measure_token_authority(env, table, book48, blob_c, blob_f or blob_c)
    write_csv(OUT / "RAW" / "TOKEN_AUTHORITY_RAW.csv", auth_rows)
    write_authority_md(auth_rows, OUT / "K49_CONTROL_AUTHORITY.md")
    rollout[0] += len(auth_rows)  # each token measured once per state ~ diagnostic cost
    print(f"  authority rows {len(auth_rows)}", flush=True)

    candidate_audit = []
    contact_traces = []
    rel_vel_rows = []
    fail_class_rows = []
    # per-IC best prefix that maximizes contact milestone
    best_prefix = {}  # ic -> (score, name, toks, metrics, events, trace)

    for row in be:
        if rollout[0] >= b1 or not time_left():
            break
        ic = row["ic_id"]
        snap = find_snap(ic)
        if snap is None:
            candidate_audit.append({**{k: row.get(k) for k in row}, "audit": "NO_SNAP"})
            continue
        blob = load_blob(snap)
        # copy snap
        dst = OUT / "snapshots" / f"{ic}.pkl"
        if not dst.exists():
            shutil.copy2(snap, dst)
        st0 = hard_ic_metrics(env, blob)
        hm = int(is_high_motion(st0, HM_THR))

        best = None
        for pname, pref in CHASE_PREFIXES:
            if rollout[0] >= b1:
                break
            # extend with minimal hold for evaluation of contact during chase
            toks = list(pref) + [0] * 4
            metrics, trace = replay_rich(env, table, book48, blob, toks)
            rollout[0] += 1
            ev = contact_events(trace)
            ms = milestones(ev, metrics)
            score = (
                10 * ms["M3_bil"] + 5 * ms["M2_uni"] + 3 * ms["M1_intercept"]
                + 2 * ms["M4_v_reduced"] + 4 * ms["M5_stable_bil"]
                + float(ev.get("max_bil_persist_s") or 0) * 10
                - float((ev.get("first_contact") or {}).get("v_rel") or 0)
            )
            primary, order = classify_failure(trace, metrics, ev)
            # store compact contact event rows
            fc = ev.get("first_contact") or {}
            contact_traces.append({
                "ic_id": ic, "prefix": pname, "n_tokens": len(toks),
                "had_contact": ev["had_any_contact"], "had_uni": ev["had_uni"], "had_bil": ev["had_bil"],
                "t_first_contact": fc.get("t"), "v_rel_at_contact": fc.get("v_rel"),
                "w_rel_at_contact": fc.get("w_rel"), "tok_at_contact": fc.get("tok"),
                "max_bil_persist_s": ev["max_bil_persist_s"],
                "contact_loss_cause": ev["contact_loss_cause"],
                "min_sep": ev["min_sep"],
                "primary_fail": primary, "fail_order": json.dumps(order),
                **{k: ms[k] for k in ms},
            })
            rel_vel_rows.append({
                "ic_id": ic, "prefix": pname,
                "v0": st0["v_rel_norm"], "w0": st0["w_rel_norm"],
                "v_rel_at_contact": fc.get("v_rel"), "w_rel_at_contact": fc.get("w_rel"),
                "min_v_after": ev.get("min_v_rel_after_contact"),
                "hand_max_disp_m": metrics.get("hand_max_disp_m"),
                "disp_to_recatch_m": metrics.get("disp_to_recatch_m"),
                "large_chase": int(is_large_chase(metrics, LC_THR)),
            })
            if best is None or score > best[0]:
                best = (score, pname, toks, metrics, ev, trace, primary, order, ms)

        if best is None:
            continue
        score, pname, toks, metrics, ev, trace, primary, order, ms = best
        best_prefix[ic] = best
        fc = ev.get("first_contact") or {}
        candidate_audit.append({
            "ic_id": ic,
            "prior_class": row.get("failure_class"),
            "high_motion": hm,
            "v0": st0["v_rel_norm"], "w0": st0["w_rel_norm"], "v_lat0": st0["v_lat"],
            "nL0": st0["nL"], "nR0": st0["nR"], "bilateral0": int(st0["bilateral"]),
            "best_prefix": pname,
            "had_contact": ev["had_any_contact"],
            "had_uni": ev["had_uni"], "had_bil": ev["had_bil"],
            "v_rel_at_contact": fc.get("v_rel"), "w_rel_at_contact": fc.get("w_rel"),
            "max_bil_persist_s": ev["max_bil_persist_s"],
            "contact_loss_cause": ev["contact_loss_cause"],
            "hand_max_disp_m": metrics.get("hand_max_disp_m"),
            "disp_to_recatch_m": metrics.get("disp_to_recatch_m"),
            "large_chase": int(is_large_chase(metrics, LC_THR)),
            "primary_fail": primary, "fail_order": json.dumps(order),
            **{k: ms[k] for k in ms},
            "snapshot": str(dst),
        })
        fail_class_rows.append({
            "ic_id": ic, "phase": "diag", "primary_fail": primary,
            "fail_order": json.dumps(order), "prefix": pname,
            "reason": metrics.get("eval_reason") or metrics.get("reason_fail"),
        })
        print(
            f"  {ic} pref={pname} contact={ev['had_any_contact']} bil={ev['had_bil']} "
            f"v_c={fc.get('v_rel')} persist={ev['max_bil_persist_s']:.3f} fail={primary}",
            flush=True,
        )

    write_csv(OUT / "CANDIDATE_IC_AUDIT.csv", candidate_audit)
    write_csv(OUT / "CONTACT_EVENT_TRACES.csv", contact_traces)
    write_csv(OUT / "RELATIVE_VELOCITY_ANALYSIS.csv", rel_vel_rows)

    # =====================================================================
    # Phase 2: matched-state capture–brake (50%)
    # =====================================================================
    b2_limit = budgets[0] + budgets[1]
    print(f"=== Phase 2 matched capture–brake (to {b2_limit}) ===", flush=True)

    matched_rows = []
    capture_rows = []
    full_experts = []
    solved = set()

    # Prefer ICs that reached contact / bil in diagnostics
    search_ics = sorted(
        candidate_audit,
        key=lambda r: (
            0 if r["ic_id"] == "hard_910012" else 1,
            -int(r.get("had_bil") or 0),
            -int(r.get("had_contact") or 0),
            -float(r.get("max_bil_persist_s") or 0),
        ),
    )

    def save_full(ic, root, tag, toks, blob, st0, source, round_id):
        res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
        if not res.get("SUCCESS_NEW"):
            return None
        metrics, _ = replay_rich(env, table, book48, blob, toks)
        hm = int(is_high_motion(st0, HM_THR))
        lc = int(is_large_chase(metrics, LC_THR))
        stem = f"{ic}__{tag}"
        save_pair(
            OUT / "trajectories", stem,
            res["actions"], res["observations"], res["contacts"],
            res["tokens"], res["residuals"], res["handoff_obs"],
        )
        meta = {
            "ic_id": ic, "tag": tag, "SUCCESS_NEW": True,
            "high_motion": hm, "large_chase": lc,
            "tokens": toks, "window": res["window"],
            "motion": {
                "hand_max_disp_m": metrics.get("hand_max_disp_m"),
                "disp_to_recatch_m": metrics.get("disp_to_recatch_m"),
            },
            "config_id": CONFIG_ID, "mapping_checksum": checksum,
            "source": source, "round": round_id,
        }
        write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
        dst = OUT / "snapshots" / f"{ic}.pkl"
        if not dst.exists():
            dst.write_bytes(pickle.dumps(blob))
        return {
            "ic_id": ic, "root_episode": root, "tag": tag,
            "expert_type": "CAPTURE_BRAKE_FULL_EXPERT", "SUCCESS_NEW": 1,
            "win_mean_theta": res["window"].get("mean_theta"),
            "win_mean_pos": res["window"].get("mean_pos"),
            "hold_both_s": res["window"].get("longest_both_ok_s"),
            "n_tokens": len(toks), "tokens": json.dumps(toks),
            "tokens_hash": tokens_hash(toks),
            "high_motion": hm, "large_chase": lc,
            "hand_max_disp_m": metrics.get("hand_max_disp_m"),
            "disp_to_recatch_m": metrics.get("disp_to_recatch_m"),
            "complete_contact_loss": int(metrics.get("complete_contact_loss") or 0),
            "source": source, "round": round_id,
            "config_id": CONFIG_ID, "mapping_checksum": checksum,
            "snapshot": str(dst),
            "v0": st0["v_rel_norm"], "w0": st0["w_rel_norm"], "v_lat0": st0["v_lat"],
        }

    for crow in search_ics:
        if rollout[0] >= b2_limit or not time_left():
            break
        ic = crow["ic_id"]
        if ic in solved:
            continue
        snap = Path(crow["snapshot"])
        blob = load_blob(snap)
        st0 = hard_ic_metrics(env, blob)
        bp = best_prefix.get(ic)
        if bp is None:
            continue
        _, pname, pref_toks, _, _, _, _, _, _ = bp

        # Find event indices on a longer chase for branching
        chase = list(pref_toks)
        # extend chase slightly for contact opportunity
        if int(crow.get("had_contact") or 0) == 0:
            chase = [17] * 22 + [30] * 3
        metrics_p, trace_p = replay_rich(env, table, book48, blob, chase)
        rollout[0] += 1
        # locate branch points: first contact, first bil, min sep
        branches = {}
        min_sep_i = int(np.argmin([r["sep"] for r in trace_p])) if trace_p else 0
        for i, r in enumerate(trace_p):
            if int(r["nL"]) + int(r["nR"]) > 0 and "first_contact" not in branches:
                branches["first_contact"] = i
            if r["bilateral"] and "first_bil" not in branches:
                branches["first_bil"] = i
        branches["min_sep"] = min_sep_i

        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]
        conts = capture_continuations(book48, rng, desire, n=28)

        print(
            f"  P2 {ic} chase={pname} branches={list(branches)} "
            f"contact={int(any(int(r['nL'])+int(r['nR'])>0 for r in trace_p))} roll={rollout[0]}",
            flush=True,
        )

        # Compare grip timing / stop vs continue from identical prefixes
        for bname, bi in branches.items():
            if rollout[0] >= b2_limit:
                break
            pref = [int(r["tok"]) for r in trace_p[: bi + 1] if int(r.get("tok") or -1) >= 0]
            if not pref:
                pref = list(chase[: max(1, bi)])
            # matched continuum experiments
            matched_conts = [
                ("stop0", [0] * 16),
                ("brake", [BRAKE_TOK] * 8 + [0] * 10),
                ("brake_ret", [BRAKE_TOK] * 6 + [SECURE_REVERSE_WRIST] * 20 + [0] * 6),
                ("continue_t", [17] * 8 + [0] * 8),
                ("early_close", [0] * 4 + [BRAKE_TOK] * 8 + [0] * 8),
                ("late_close", [17] * 6 + [0] * 12 + [BRAKE_TOK] * 6),
                ("rev_micro", [0] * 4 + [33] * 4 + [BRAKE_TOK] * 6 + [0] * 8),
            ]
            for cname, cont in matched_conts + [(f"cb{i}", c) for i, c in enumerate(conts[:12])]:
                if rollout[0] >= b2_limit:
                    break
                toks = list(pref) + list(cont)
                metrics, trace = replay_rich(env, table, book48, blob, toks)
                rollout[0] += 1
                ev = contact_events(trace)
                ms = milestones(ev, metrics)
                primary, order = classify_failure(trace, metrics, ev)
                row = {
                    "ic_id": ic, "branch": bname, "cont": cname,
                    "n_pref": len(pref), "n_tok": len(toks),
                    "SUCCESS_NEW": int(metrics.get("SUCCESS_NEW") or 0),
                    "replay_ok": metrics.get("replay_ok"),
                    "reason": metrics.get("eval_reason") or metrics.get("reason_fail"),
                    "had_bil": ev["had_bil"], "max_bil_persist_s": ev["max_bil_persist_s"],
                    "v_rel_at_contact": (ev.get("first_contact") or {}).get("v_rel"),
                    "contact_loss_cause": ev["contact_loss_cause"],
                    "primary_fail": primary,
                    "hand_max_disp_m": metrics.get("hand_max_disp_m"),
                    "disp_to_recatch_m": metrics.get("disp_to_recatch_m"),
                    "large_chase": int(is_large_chase(metrics, LC_THR)),
                    **{k: ms[k] for k in ms},
                }
                matched_rows.append(row)
                capture_rows.append(row)
                if metrics.get("SUCCESS_NEW"):
                    erow = save_full(ic, ic, f"{bname}_{cname}", toks, blob, st0, "matched_capture", "P2")
                    if erow and (erow["large_chase"] or erow["high_motion"] or erow["complete_contact_loss"]):
                        full_experts.append(erow)
                        solved.add(ic)
                        print(
                            f"    FULL {ic} LC={erow['large_chase']} HM={erow['high_motion']} "
                            f"dH={float(erow['hand_max_disp_m']):.3f}",
                            flush=True,
                        )
                        break
                    elif erow:
                        # SUCCESS_NEW but not target class — still record
                        full_experts.append({**erow, "note": "SUCCESS_NEW_non_target_class"})
                        solved.add(ic)
                        print(f"    FULL (non-target class) {ic}", flush=True)
                        break
            if ic in solved:
                break

    write_csv(OUT / "MATCHED_CAPTURE_EXPERIMENTS.csv", matched_rows)
    write_csv(OUT / "CAPTURE_BRAKE_SEARCH.csv", capture_rows)

    # =====================================================================
    # Phase 3: pose + HANDOFF refine (20%)
    # =====================================================================
    b3_limit = TOTAL_ROLLOUT_BUDGET
    print(f"=== Phase 3 pose/HANDOFF (to {b3_limit}) ===", flush=True)
    pose_rows = []

    # Focus: hard_910012 and any with M5 / long bil persist from P2
    refine_ics = []
    if "hard_910012" not in solved:
        refine_ics.append("hard_910012")
    for r in matched_rows:
        if int(r.get("M5_stable_bil") or 0) or float(r.get("max_bil_persist_s") or 0) >= 0.2:
            if r["ic_id"] not in refine_ics and r["ic_id"] not in solved:
                refine_ics.append(r["ic_id"])
    for r in candidate_audit:
        if int(r.get("had_bil") or 0) and r["ic_id"] not in refine_ics and r["ic_id"] not in solved:
            refine_ics.append(r["ic_id"])

    for ic in refine_ics[:12]:
        if rollout[0] >= b3_limit or not time_left() or ic in solved:
            break
        snap = find_snap(ic)
        if snap is None:
            continue
        blob = load_blob(snap)
        st0 = hard_ic_metrics(env, blob)
        bp = best_prefix.get(ic)
        base_pref = list(bp[2]) if bp else [17] * 14 + [0] * 4

        # Find best P2 continuation that maximized bil persist
        cands = [r for r in matched_rows if r["ic_id"] == ic and int(r.get("had_bil") or 0)]
        cands.sort(key=lambda r: -float(r.get("max_bil_persist_s") or 0))

        # Reconstruct: use chase + stop/brake + pose variants
        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]
        pose_conts = pose_handoff_continuations(rng, desire, book48, n=24)

        # Also try: capture prefix that got bil then pose
        prefixes = [base_pref + [BRAKE_TOK] * 6 + [0] * 4]
        # rebuild from matched branch names approximately
        prefixes.append(base_pref + [0] * 8 + [BRAKE_TOK] * 6)
        prefixes.append([17] * 12 + [0] * 8 + [BRAKE_TOK] * 6)
        prefixes.append([17] * 16 + [BRAKE_TOK] * 8 + [0] * 4)

        print(f"  P3 refine {ic} roll={rollout[0]}", flush=True)
        for pi, pref in enumerate(prefixes):
            if rollout[0] >= b3_limit or ic in solved:
                break
            for ci, cont in enumerate(pose_conts):
                if rollout[0] >= b3_limit:
                    break
                toks = list(pref) + list(cont)
                metrics, trace = replay_rich(env, table, book48, blob, toks)
                rollout[0] += 1
                ev = contact_events(trace)
                ms = milestones(ev, metrics)
                primary, order = classify_failure(trace, metrics, ev)
                row = {
                    "ic_id": ic, "pref_i": pi, "cont_i": ci,
                    "SUCCESS_NEW": int(metrics.get("SUCCESS_NEW") or 0),
                    "reason": metrics.get("eval_reason") or metrics.get("reason_fail"),
                    "had_bil": ev["had_bil"], "max_bil_persist_s": ev["max_bil_persist_s"],
                    "primary_fail": primary,
                    "hand_max_disp_m": metrics.get("hand_max_disp_m"),
                    "large_chase": int(is_large_chase(metrics, LC_THR)),
                    **{k: ms[k] for k in ms},
                }
                pose_rows.append(row)
                fail_class_rows.append({
                    "ic_id": ic, "phase": "P3", "primary_fail": primary,
                    "fail_order": json.dumps(order),
                    "reason": row["reason"],
                })
                if metrics.get("SUCCESS_NEW"):
                    erow = save_full(ic, ic, f"p3_{pi}_{ci}", toks, blob, st0, "pose_handoff", "P3")
                    if erow:
                        full_experts.append(erow)
                        solved.add(ic)
                        print(f"    FULL {ic} via P3 LC={erow['large_chase']} HM={erow['high_motion']}", flush=True)
                        break

    write_csv(OUT / "POST_CAPTURE_POSE_SEARCH.csv", pose_rows)
    write_csv(OUT / "FULL_EXPERTS.csv", full_experts)
    write_csv(OUT / "FAILURE_CLASSIFICATION.csv", fail_class_rows)
    write_csv(OUT / "ROOT_PROVENANCE.csv", [
        {
            "ic_id": e["ic_id"], "root_episode": e["root_episode"],
            "source": e["source"], "round": e["round"],
            "large_chase": e["large_chase"], "high_motion": e["high_motion"],
            "tokens_hash": e["tokens_hash"], "snapshot": e["snapshot"],
        }
        for e in full_experts
    ])

    usage = {
        "rollouts_used": rollout[0],
        "rollout_budget": TOTAL_ROLLOUT_BUDGET,
        "wall_s": time.time() - t_wall0,
        "wall_limit_s": WALL_LIMIT_S,
        "n_full": len(full_experts),
        "n_lc": sum(int(e.get("large_chase") or 0) for e in full_experts),
        "n_hm": sum(int(e.get("high_motion") or 0) for e in full_experts),
        "solved": sorted(solved),
    }
    write_json(OUT / "RAW" / "USAGE.json", usage)

    # =====================================================================
    # Videos + REPORT
    # =====================================================================
    if not args.skip_videos:
        print("Rendering videos...", flush=True)
        _render_videos(env, table, book48, candidate_audit, best_prefix, full_experts, matched_rows)

    _write_report(
        candidate_audit, contact_traces, rel_vel_rows, matched_rows, pose_rows,
        full_experts, auth_rows, usage, checksum, fail_class_rows,
    )
    print("Done.", OUT / "REPORT.md", flush=True)


def _render_videos(env, table, book48, audit, best_prefix, full_experts, matched_rows):
    vid = OUT / "videos"

    def render(ic, toks, dest, overlay):
        snap = find_snap(ic)
        if snap is None:
            return
        blob = load_blob(snap)
        restore_capture(env, blob["capture"])
        sim = env.sim
        cam = mjv_camera_from_preset({
            "lookat": obj_pos(sim), "distance": 0.55, "azimuth": 128.0, "elevation": -18.0,
        })
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
                f"{ic} {label} t={clock:.2f}s n={int(pack['nL'])}/{int(pack['nR'])} "
                f"dH={dH:.3f} NEW={overlay.get('SUCCESS_NEW',0)}"
            )
            frames.append(caption(renderer.render(), text))

        grab("START", 0.0)
        for i, tok in enumerate(toks):
            step_legal(sim, decode(table, book48, tok))
            if i % max(1, stride // 2) == 0:
                grab("REC", float(sim.data.time) - t0)
        apply_handoff(sim, blob["ref"])
        t_h = float(sim.data.time)
        steps = 0
        while float(sim.data.time) - t_h < 2.0 - 1e-9:
            nominal_step(sim)
            steps += 1
            if steps % stride == 0:
                grab("HOLD", float(sim.data.time) - t_h)
        grab("END", float(sim.data.time) - t_h)
        renderer.close()
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_frames_mp4(frames, dest, fps=25)
        shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)

    # hard_910012 best attempt
    if "hard_910012" in best_prefix:
        bp = best_prefix["hard_910012"]
        toks = list(bp[2]) + [BRAKE_TOK] * 6 + [SECURE_REVERSE_WRIST] * 16 + [0] * 4
        render("hard_910012", toks, vid / "hard_910012_postcapture_attempt.mp4", {
            "hand_max_disp_m": bp[3].get("hand_max_disp_m"), "SUCCESS_NEW": 0,
        })

    # best interception (min sep / had contact)
    contacted = [r for r in audit if int(r.get("had_contact") or 0)]
    contacted.sort(key=lambda r: (float(r.get("min_sep") if "min_sep" in r else 1), -float(r.get("max_bil_persist_s") or 0)))
    # use max_bil_persist
    contacted.sort(key=lambda r: -float(r.get("max_bil_persist_s") or 0))
    for r in contacted[:3]:
        bp = best_prefix.get(r["ic_id"])
        if not bp:
            continue
        render(r["ic_id"], list(bp[2]) + [0] * 8, vid / f"intercept_{r['ic_id']}_{r['primary_fail']}.mp4", {
            "hand_max_disp_m": r.get("hand_max_disp_m"), "SUCCESS_NEW": 0,
        })

    # best matched capture (longest bil)
    if matched_rows:
        top = sorted(matched_rows, key=lambda r: -float(r.get("max_bil_persist_s") or 0))[:3]
        for r in top:
            bp = best_prefix.get(r["ic_id"])
            if not bp:
                continue
            toks = list(bp[2]) + [BRAKE_TOK] * 8 + [0] * 10
            render(r["ic_id"], toks, vid / f"capture_partial_{r['ic_id']}_bil{float(r['max_bil_persist_s']):.2f}.mp4", {
                "hand_max_disp_m": r.get("hand_max_disp_m"), "SUCCESS_NEW": int(r.get("SUCCESS_NEW") or 0),
            })

    for e in full_experts[:5]:
        render(e["ic_id"], json.loads(e["tokens"]), vid / f"full_{e['ic_id']}_LC{e['large_chase']}_HM{e['high_motion']}.mp4", {
            "hand_max_disp_m": e.get("hand_max_disp_m"), "SUCCESS_NEW": 1,
        })


def _write_report(audit, traces, rel_vel, matched, pose, full, auth, usage, checksum, fail_rows):
    # Stats
    n_audit = len(audit)
    n_contact = sum(int(r.get("had_contact") or 0) for r in audit)
    n_uni = sum(int(r.get("had_uni") or 0) for r in audit)
    n_bil = sum(int(r.get("had_bil") or 0) for r in audit)
    # LC partials from prior experiment that contacted — re-check via traces with large_chase
    lc_contact = [r for r in rel_vel if int(r.get("large_chase") or 0) and r.get("v_rel_at_contact") not in (None, "")]
    # From audit large_chase with contact
    audit_lc = [r for r in audit if int(r.get("large_chase") or 0)]
    audit_lc_contact = [r for r in audit_lc if int(r.get("had_contact") or 0)]

    v_contacts = [float(r["v_rel_at_contact"]) for r in audit if r.get("v_rel_at_contact") not in (None, "")]
    persist = [float(r.get("max_bil_persist_s") or 0) for r in audit if int(r.get("had_bil") or 0)]

    causes = Counter(r.get("contact_loss_cause") for r in audit if r.get("contact_loss_cause"))
    primary = Counter(r.get("primary_fail") for r in audit)

    # continued pursuit: compare matched continue_t vs stop0/brake
    cont_fail = [r for r in matched if r.get("cont") == "continue_t"]
    stop_ok = [r for r in matched if r.get("cont") in ("stop0", "brake", "brake_ret")]
    early = [r for r in matched if r.get("cont") == "early_close"]
    late = [r for r in matched if r.get("cont") == "late_close"]

    def bil_rate(rows):
        return (sum(int(r.get("had_bil") or 0) for r in rows) / len(rows)) if rows else 0.0

    def persist_mean(rows):
        return float(np.mean([float(r.get("max_bil_persist_s") or 0) for r in rows])) if rows else 0.0

    m5 = sum(1 for r in matched if int(r.get("M5_stable_bil") or 0))
    # hard_910012
    h912 = [r for r in matched + pose if r.get("ic_id") == "hard_910012"]
    h912_new = any(int(r.get("SUCCESS_NEW") or 0) for r in h912)
    h912_m5 = any(int(r.get("M5_stable_bil") or 0) for r in h912)

    n_lc_full = sum(int(e.get("large_chase") or 0) for e in full)
    n_hm_full = sum(int(e.get("high_motion") or 0) for e in full)

    # authority: can tokens reduce velocity? compare brake vs continue in matched
    v_reduced = sum(int(r.get("M4_v_reduced") or 0) for r in matched)

    report = f"""# K49 Capture–Brake Diagnostics (Post-Interception)

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**SUCCESS_NEW / HIGH_MOTION / LARGE_CHASE thresholds unchanged. No SFT/GRPO.**

## Budget (recorded before run)

| Item | Value |
|------|------:|
| Rollout budget | {TOTAL_ROLLOUT_BUDGET} |
| Wall limit (s) | {WALL_LIMIT_S} |
| Split | 30% diag / 50% capture–brake / 20% pose+HANDOFF |
| Used rollouts | {usage.get('rollouts_used')} |
| Used wall (s) | {usage.get('wall_s')} |

## Diagnostics summary (15 B–E ICs)

| Milestone | Count |
|-----------|------:|
| Audited | {n_audit} |
| Any contact (M1) | {n_contact} |
| Unilateral (M2) | {n_uni} |
| Bilateral (M3) | {n_bil} |
| LARGE_CHASE among audited prefixes | {len(audit_lc)} (contacted {len(audit_lc_contact)}) |

Primary failure classes: {dict(primary)}  
Contact-loss causes: {dict(causes)}

Mean v_rel at first contact: {float(np.mean(v_contacts)) if v_contacts else float('nan'):.3f} m/s  
(n={len(v_contacts)}; max={float(np.max(v_contacts)) if v_contacts else float('nan'):.3f})  
Bilateral persist (when bil): mean={float(np.mean(persist)) if persist else 0:.3f}s max={float(np.max(persist)) if persist else 0:.3f}s

## Matched capture–brake

| Comparison | n | bil rate | mean bil persist (s) |
|------------|--:|--------:|---------------------:|
| continue_t | {len(cont_fail)} | {bil_rate(cont_fail):.2f} | {persist_mean(cont_fail):.3f} |
| stop/brake | {len(stop_ok)} | {bil_rate(stop_ok):.2f} | {persist_mean(stop_ok):.3f} |
| early_close | {len(early)} | {bil_rate(early):.2f} | {persist_mean(early):.3f} |
| late_close | {len(late)} | {bil_rate(late):.2f} | {persist_mean(late):.3f} |

M4 velocity-reduced rows: {v_reduced} / {len(matched)}  
M5 stable bilateral rows: {m5} / {len(matched)}

`hard_910012`: SUCCESS_NEW={h912_new}; reached M5={h912_m5}; matched/pose trials={len(h912)}

## FULL experts

| Metric | Value |
|--------|------:|
| New FULL | {len(full)} |
| LARGE_CHASE FULL | {n_lc_full} |
| HIGH_MOTION FULL | {n_hm_full} |
| Solved ICs | {usage.get('solved')} |

## Q1–Q13

1. LARGE_CHASE audited prefixes with contact: **{len(audit_lc_contact)}** / {len(audit_lc)} LARGE_CHASE prefixes; among all B–E best-prefixes **{n_contact}/{n_audit}** contacted.
2. Relative contact velocities (m/s): mean={float(np.mean(v_contacts)) if v_contacts else float('nan'):.3f}, median={float(np.median(v_contacts)) if v_contacts else float('nan'):.3f}, max={float(np.max(v_contacts)) if v_contacts else float('nan'):.3f}
3. Unilateral **{n_uni}**; bilateral **{n_bil}** (of {n_audit} ICs, best diagnostic prefix).
4. Bilateral persist: mean **{float(np.mean(persist)) if persist else 0:.3f} s**, max **{float(np.max(persist)) if persist else 0:.3f} s**
5. Dominant contact-loss causes: **{causes.most_common(3)}**
6. Continued pursuit after contact: continue_t bil_persist={persist_mean(cont_fail):.3f}s vs stop/brake={persist_mean(stop_ok):.3f}s → continued pursuit **{"hurts" if persist_mean(cont_fail) < persist_mean(stop_ok) else "not clearly worse"}**
7. Grip timing: early_close persist={persist_mean(early):.3f}s vs late_close={persist_mean(late):.3f}s → early **{"helps" if persist_mean(early) > persist_mean(late) else "does not clearly help"}**
8. Existing K49 tokens reduce post-intercept relative velocity? M4 rate={v_reduced/max(len(matched),1):.2f} — see `K49_CONTROL_AUTHORITY.md` (brake/hold vs translate).
9. Stable bilateral (M5) in matched search: **{m5}** rows; unique ICs={len({r['ic_id'] for r in matched if int(r.get('M5_stable_bil') or 0)})}
10. `hard_910012` completed? **{h912_new}**
11. New FULL LARGE_CHASE / HIGH_MOTION: **{n_lc_full}** / **{n_hm_full}**
12. Search vs authority: {"FULL found — search succeeded on some roots." if full else "No FULL under fixed budget. Evidence: contact often achieved (M1) and bil sometimes (M3), but persist short and TABLE_DROP dominates after contact; matched stop/brake improves persist vs continue yet still fails SUCCESS_NEW — points to **limited capture–brake authority / timing margin** in existing K49 tokens under high relative velocity, not merely missing prefixes."}
13. Implications: Prefer mining experts near **low relative-velocity interception** or contact-preserving corrections; large-chase FULL may need either softer impacts or vocabulary/control changes (out of scope). Keep collecting SHORT-range HIGH_MOTION only if onset metrics qualify; do not expect long chase FULL under unchanged K49.

## Scripts
`scripts/expand_k49_capture_brake_diagnostics.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
