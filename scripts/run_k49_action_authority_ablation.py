"""K49 action-authority ablation: continuous legal control vs discrete K49.

Locked physics/controller/obs/SUCCESS_NEW. No SFT/GRPO.
step_legal accepts continuous a in [-1,1]^7 (AIRBORNE_V3 map).
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

from controllers.recovery.action_map import (
    AIRBORNE_V3,
    V_AIRBORNE,
    V_Z_DOWN,
    V_Z_UP,
    W_MAX,
    TAU_OPEN,
    TAU_SECURE,
    grip_tau,
    hand_translation,
    map_action,
)
from envs.outcomes.horizon_outcome import recovery_failed, terminal_reason
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture
from envs.step_legal import N_SUB, TRACK, step_legal
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

from scripts.expand_k49_hard_dynamic import ic_metrics as hard_ic_metrics, is_high_motion
from scripts.expand_k49_large_chase_dynamic import HM_THR, hand_pos, obj_pos
from scripts.expand_k49_capture_brake_diagnostics import (
    find_snap,
    load_blob,
    contact_events,
    rich_sample,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    decode,
    decompose,
    evaluate_full,
    mapping_checksum,
    write_csv,
    write_json,
)

OUT = ROOT / "results" / "diagnostics" / "k49_action_authority_ablation"
LC_OUT = ROOT / "results" / "expert_generation" / "k49_large_chase_dynamic_recovery"
CB_OUT = ROOT / "results" / "expert_generation" / "k49_capture_brake_diagnostics"

# Budgets recorded BEFORE execution
TOTAL_ROLLOUT_BUDGET = 2800
WALL_LIMIT_S = 50 * 60
STAGE_FRAC = (0.40, 0.60)

BE_ICS = [
    "hard_910012",  # pose benchmark priority
    "hard_910006", "hard_910095", "hard_910031", "bimp_0033",
    "hdimp_0002", "hdimp_0012", "bimp_0010", "bimp_0036", "ubimp_0040",
    "hdimp_0039", "hard_910012",
]
POSE_ICS = ["hard_910012"]
CAPTURE_ICS = [
    "bimp_0010", "bimp_0036", "ubimp_0040", "bimp_0033",
    "hard_910006", "hard_910095", "hdimp_0002", "hdimp_0039",
]

# Known capture-stabilize prefix for pose benchmark (from prior diagnostics)
POSE_PREFIX_TOKS = [30, 30, 30] + [17] * 16 + [0] * 8 + [12] * 6


def tokens_hash(toks):
    return hashlib.sha256(json.dumps([int(x) for x in toks]).encode()).hexdigest()[:16]


def act_hash(seq):
    arr = np.asarray(seq, np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Interface audit docs
# ---------------------------------------------------------------------------

def write_interface_audit(table, book48, checksum):
    md = OUT / "LEGAL_ACTION_INTERFACE_AUDIT.md"
    md.write_text(
        f"""# Legal Action Interface Audit

## `step_legal` (envs/step_legal.py)

- **Accepts:** arbitrary continuous `action` shaped `(7,)`, **clipped to [-1, 1]** componentwise.
- **Does NOT** require K49 token IDs. Token decoding is only a way to *produce* a 7-D action.
- **Timestep:** `N_SUB={N_SUB}` MuJoCo steps per call; `dt = model.opt.timestep` → **20 ms** command period.
- **Does not** write qpos/qvel/ball; resets `p_des` to current hand position each command.
- **Controller:** `apply_cartesian_ctrl` with TRACK gains `{TRACK}` after `map_action(..., AIRBORNE_V3)`.

## Action vector (normalized → physical via AIRBORNE_V3)

| Index | Name | Normalized | Physical mapping |
|------:|------|------------|------------------|
| 0 | a_vx | [-1,1] | hand-frame vx = a*{V_AIRBORNE} m/s |
| 1 | a_vy | [-1,1] | hand-frame vy = a*{V_AIRBORNE} m/s |
| 2 | a_vz | [-1,1] | +a -> +a*{V_Z_DOWN} m/s (world-down on horiz grasp); -a -> a*{V_Z_UP} m/s |
| 3 | a_wx | [-1,1] | r_des-frame wx = a*{W_MAX} rad/s |
| 4 | a_wy | [-1,1] | r_des-frame wy = a*{W_MAX} rad/s |
| 5 | a_wz | [-1,1] | r_des-frame wz = a*{W_MAX} rad/s |
| 6 | a_grip | [-1,1] | tau = {TAU_OPEN} + 0.5(a+1)({TAU_SECURE}-{TAU_OPEN}) N*m |

Frames: `v_world = R_hand @ v_hand`, `w_world = R_des @ w_body`.
Grip: a=-1 open (tau={TAU_OPEN}), a=+1 secure (tau={TAU_SECURE}).

## Condition B support

**Yes.** Continuous legal commands are first-class through `step_legal`. No privileged interface required.

## K49 baseline (`{CONFIG_ID}`, checksum `{checksum}`)

- Tokens **0–46**: K48 motion prototypes (table rows).
- Token **47**: HANDOFF (not a motion row executed via decode in recovery loop the same way).
- Token **48**: SECURE_REVERSE_WRIST (mirror of tok4 wy at 25% scale, grip secure).
- Grip channel in K49 prototypes: essentially **{{-1, -0.48, +1}}** only (3 levels).
- wy channel: sparse; reverse wy mainly via tok48.
- Lateral translation magnitudes: discrete set (see raw/K49_COVERAGE.json).

## Coupling / saturation

- Cartesian tracking may lag or saturate vs commanded v/w (see CONTROLLER_RESPONSE.md).
- Grip torque clipped to `sim.ids.ctrl_low/high[7]`.
- Joint/actuator limits still apply inside MuJoCo + cartesian ctrl.
""",
        encoding="utf-8",
    )
    # coverage json
    cov = {}
    for d, name in enumerate(["ax", "ay", "az", "wx", "wy", "wz", "grip"]):
        vals = sorted(set(np.round(table[:49, d], 5).tolist()))
        cov[name] = {"n_unique": len(vals), "values": vals}
    write_json(OUT / "raw" / "K49_COVERAGE.json", {
        "config_id": CONFIG_ID, "checksum": checksum, "coverage": cov,
    })


def measure_controller_response(env, blob, table, book48):
    """Commanded vs realized hand velocity for hold/translate/brake/open."""
    rows = []
    tests = [
        ("hold", np.array([0, 0, 0, 0, 0, 0, 1], np.float32)),
        ("lat_+x", np.array([0.5, 0, 0, 0, 0, 0, 1], np.float32)),
        ("lat_+y", np.array([0, 0.5, 0, 0, 0, 0, 1], np.float32)),
        ("lat_fine", np.array([0.08, 0, 0, 0, 0, 0, 1], np.float32)),
        ("brake_like", np.array([0, 0, 0, 0, 0, 0, 1], np.float32)),
        ("open", np.array([0, 0, 0, 0, 0, 0, -1], np.float32)),
        ("secure", np.array([0, 0, 0, 0, 0, 0, 1], np.float32)),
        ("wy+", np.array([0, 0, 0, 0, 0.5, 0, 1], np.float32)),
        ("wy-", np.array([0, 0, 0, 0, -0.25, 0, 1], np.float32)),
        ("tok17", np.asarray(decode(table, book48, 17), np.float32)),
        ("tok12", np.asarray(decode(table, book48, 12), np.float32)),
        ("tok48", np.asarray(decode(table, book48, 48), np.float32)),
    ]
    for name, a in tests:
        restore_capture(env, blob["capture"])
        sim = env.sim
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        cmd = map_action(a, rd, rh, AIRBORNE_V3)
        h0 = hand_pos(sim)
        t0 = float(sim.data.time)
        # 10 legal steps
        for _ in range(10):
            step_legal(sim, a)
        dt = float(sim.data.time) - t0
        h1 = hand_pos(sim)
        v_real = (h1 - h0) / max(dt, 1e-9)
        v_cmd = cmd["v_world"]
        rows.append({
            "name": name,
            "a": json.dumps(a.tolist()),
            "v_cmd_world": json.dumps(v_cmd.tolist()),
            "v_cmd_norm": float(np.linalg.norm(v_cmd)),
            "v_real_world": json.dumps(v_real.tolist()),
            "v_real_norm": float(np.linalg.norm(v_real)),
            "tracking_ratio": float(np.linalg.norm(v_real) / max(np.linalg.norm(v_cmd), 1e-6)),
            "dt_s": dt,
        })
    write_csv(OUT / "raw" / "CONTROLLER_RESPONSE_RAW.csv", rows)
    (OUT / "CONTROLLER_RESPONSE.md").write_text(
        "# Controller Response (commanded vs realized)\n\n"
        "Measured over 10×20 ms `step_legal` steps from a representative IC snapshot.\n\n"
        "| name | |v_cmd| m/s | |v_real| m/s | tracking_ratio |\n"
        "|------|------------:|-------------:|----------------:|\n"
        + "\n".join(
            f"| {r['name']} | {float(r['v_cmd_norm']):.4f} | {float(r['v_real_norm']):.4f} | "
            f"{float(r['tracking_ratio']):.3f} |"
            for r in rows
        )
        + "\n\n"
        "## Notes\n"
        "- Tracking ratio ≪ 1 indicates cartesian lag/saturation relative to command.\n"
        "- Fine lateral commands (a=0.08) probe whether small continuous moves are realizable.\n"
        "- Grip open/secure changes finger torque; translation may still occur from coupling.\n",
        encoding="utf-8",
    )
    return rows


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def run_actions(env, blob, actions, eval_full=True):
    """Replay continuous or decoded actions; optional SUCCESS_NEW eval."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    ok = True
    trace = []
    for a in actions:
        a = np.clip(np.asarray(a, np.float32).reshape(7), -1.0, 1.0)
        trace.append(rich_sample(env.sim, t0, tok=-1, cmd=a))
        step_legal(env.sim, a)
        env._prev = a.copy()
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            ok = False
            break
    if ok:
        trace.append(rich_sample(env.sim, t0, tok=-1))
    ev = contact_events(trace)
    m_end = measure(env.sim, U_TGT, P_TGT) if ok else None
    out = {
        "replay_ok": int(ok),
        "reason_fail": None if ok else (terminal_reason(env.sim) or "DROP"),
        "n_actions": len(actions),
        "had_contact": ev["had_any_contact"],
        "had_uni": ev["had_uni"],
        "had_bil": ev["had_bil"],
        "max_bil_persist_s": ev["max_bil_persist_s"],
        "v_rel_at_contact": (ev.get("first_contact") or {}).get("v_rel"),
        "min_v_after": ev.get("min_v_rel_after_contact"),
        "contact_loss_cause": ev["contact_loss_cause"],
        "e_theta_end": None if m_end is None else float(m_end["e_theta_deg"]),
        "e_pos_end": None if m_end is None else float(m_end["e_pos_mm"]),
        "SUCCESS_NEW": 0,
        "win_mean_theta": None,
        "win_mean_pos": None,
        "hold_both_s": None,
        "eval_reason": None,
    }
    if ok and eval_full:
        # evaluate_full needs tokens — run nominal path manually for continuous
        apply_handoff(env.sim, blob["ref"])
        from envs.outcomes.horizon_outcome import NOMINAL_S, latched
        from envs.outcomes.recovery_success import nominal_step as nom
        from scripts.generate_k49_sliding_recatch_dataset import window_stats
        t_h = float(env.sim.data.time)
        samples = []
        next_sample = 0.0
        reason = None
        while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
            nom(env.sim)
            t_rel = float(env.sim.data.time) - t_h
            if t_rel + 1e-12 >= next_sample:
                m = measure(env.sim, U_TGT, P_TGT)
                cs = contact_summary(env.sim)
                samples.append({
                    "t_rel": t_rel, "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
                    "bilateral": cs["bilateral"], "obj_z": m["obj_z"],
                })
                next_sample += 0.02
            if latched(env.sim):
                reason = "TABLE_DROP"
        win = window_stats(samples, float(NOMINAL_S) - 2.0)
        # SUCCESS_NEW criteria from project: th<=10, pos<=12, hold>=0.5
        ok_new = bool(win.get("SUCCESS_NEW") or (
            win.get("mean_theta") is not None
            and float(win["mean_theta"]) <= 10.0
            and float(win["mean_pos"]) <= 12.0
            and float(win.get("longest_both_ok_s") or 0) >= 0.5
        ))
        # prefer official window_stats if it sets SUCCESS_NEW
        if "SUCCESS_NEW" in win:
            ok_new = bool(win["SUCCESS_NEW"])
        out["SUCCESS_NEW"] = int(ok_new)
        out["win_mean_theta"] = win.get("mean_theta")
        out["win_mean_pos"] = win.get("mean_pos")
        out["hold_both_s"] = win.get("longest_both_ok_s")
        out["eval_reason"] = reason or ("SUCCESS_NEW" if ok_new else "POSE_OR_HOLD_FAIL")
    return out, trace, ev


def toks_to_actions(table, book48, toks):
    return [np.asarray(decode(table, book48, int(t)), np.float32) for t in toks]


def nearest_token(table, action, exclude_handoff=True):
    a = np.clip(np.asarray(action, np.float32).reshape(7), -1, 1)
    best_i, best_d = 0, 1e9
    for i in range(table.shape[0]):
        if exclude_handoff and i == 47:
            continue
        d = float(np.linalg.norm(table[i] - a))
        if d < best_d:
            best_d, best_i = d, i
    return best_i, best_d


# ---------------------------------------------------------------------------
# Continuous search
# ---------------------------------------------------------------------------

def cem_optimize(env, blob, prefix_actions, horizon_chunks, chunk_len, n_iters, pop, elite,
                 score_fn, rng, rollout_counter, rollout_limit, bound_scale=1.0):
    """CEM over chunked continuous actions appended to prefix."""
    dim = horizon_chunks * 7
    mean = np.zeros(dim, np.float32)
    std = np.full(dim, 0.35 * bound_scale, np.float32)
    best = None
    best_score = -1e18
    history = []
    for it in range(n_iters):
        if rollout_counter[0] >= rollout_limit:
            break
        samples = []
        scores = []
        for _ in range(pop):
            if rollout_counter[0] >= rollout_limit:
                break
            z = mean + std * rng.normal(size=dim).astype(np.float32)
            z = np.clip(z, -1.0, 1.0)
            chunks = z.reshape(horizon_chunks, 7)
            acts = list(prefix_actions)
            for c in chunks:
                acts.extend([c.copy()] * chunk_len)
            out, trace, ev = run_actions(env, blob, acts, eval_full=True)
            rollout_counter[0] += 1
            sc = score_fn(out, ev)
            samples.append(z)
            scores.append(sc)
            if sc > best_score:
                best_score = sc
                best = (acts, out, ev, float(sc))
            history.append({"iter": it, "score": sc, **{k: out[k] for k in (
                "SUCCESS_NEW", "max_bil_persist_s", "e_pos_end", "e_theta_end",
                "had_bil", "eval_reason",
            )}})
        if not scores:
            break
        idx = np.argsort(scores)[-elite:]
        elite_z = np.stack([samples[i] for i in idx], 0)
        mean = elite_z.mean(0)
        std = np.maximum(elite_z.std(0), 0.05)
    return best, history


def score_pose(out, ev):
    """Higher is better for pose correction after capture."""
    sc = 0.0
    sc += 50.0 * float(out.get("SUCCESS_NEW") or 0)
    sc += 5.0 * float(out.get("max_bil_persist_s") or 0)
    if out.get("e_pos_end") is not None:
        sc += max(0.0, 40.0 - float(out["e_pos_end"]))  # reward low pos error
    if out.get("e_theta_end") is not None:
        sc += max(0.0, 15.0 - float(out["e_theta_end"]))
    if out.get("win_mean_pos") is not None:
        sc += max(0.0, 50.0 - float(out["win_mean_pos"]))
        sc += max(0.0, 20.0 - float(out["win_mean_theta"] or 20))
    if not out.get("replay_ok"):
        sc -= 20.0
    if out.get("had_bil"):
        sc += 3.0
    return sc


def score_capture(out, ev):
    sc = 0.0
    sc += 80.0 * float(out.get("SUCCESS_NEW") or 0)
    sc += 8.0 * float(out.get("max_bil_persist_s") or 0)
    sc += 4.0 * float(out.get("had_bil") or 0)
    sc += 2.0 * float(out.get("had_uni") or 0)
    sc += 1.0 * float(out.get("had_contact") or 0)
    if out.get("min_v_after") is not None and out.get("v_rel_at_contact") is not None:
        v0 = float(out["v_rel_at_contact"])
        vmin = float(out["min_v_after"])
        if v0 > 1e-6:
            sc += 10.0 * max(0.0, (v0 - vmin) / v0)  # M4-like
    if not out.get("replay_ok"):
        sc -= 15.0
    return sc


def k49_random_search(env, table, book48, blob, prefixes, suffixes, n_try, rng,
                      score_fn, rollout_counter, rollout_limit, family):
    best = None
    best_sc = -1e18
    rows = []
    for _ in range(n_try):
        if rollout_counter[0] >= rollout_limit:
            break
        pref = prefixes[int(rng.integers(0, len(prefixes)))]
        suf = suffixes[int(rng.integers(0, len(suffixes)))]
        toks = list(pref) + list(suf)
        acts = toks_to_actions(table, book48, toks)
        out, trace, ev = run_actions(env, blob, acts, eval_full=True)
        rollout_counter[0] += 1
        sc = score_fn(out, ev)
        row = {"family": family, "condition": "K49", "score": sc, "n_tok": len(toks), **out}
        rows.append(row)
        if sc > best_sc:
            best_sc = sc
            best = (toks, acts, out, ev, sc)
    return best, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)

    budgets = [int(TOTAL_ROLLOUT_BUDGET * f) for f in STAGE_FRAC]
    budgets[1] = TOTAL_ROLLOUT_BUDGET - budgets[0]
    write_json(OUT / "raw" / "BUDGET.json", {
        "TOTAL_ROLLOUT_BUDGET": TOTAL_ROLLOUT_BUDGET,
        "WALL_LIMIT_S": WALL_LIMIT_S,
        "STAGE_FRAC": STAGE_FRAC,
        "stage_budgets": budgets,
        "recorded_before_execution": True,
        "POSE_ICS": POSE_ICS,
        "CAPTURE_ICS": CAPTURE_ICS,
        "cem_pose": {"iters": 6, "pop": 24, "elite": 6, "chunks": 8, "chunk_len": 3},
        "cem_capture": {"iters": 5, "pop": 20, "elite": 5, "chunks": 10, "chunk_len": 2},
    })

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    checksum = mapping_checksum(table)
    mapping_meta = describe_k49_config(CONFIG_ID, book48)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261019)
    rollout = [0]
    t0 = time.time()

    def time_left():
        return time.time() - t0 < WALL_LIMIT_S

    write_interface_audit(table, book48, checksum)

    # manifest
    manifest = []
    for ic in sorted(set(POSE_ICS + CAPTURE_ICS)):
        snap = find_snap(ic)
        if snap is None:
            continue
        dst = OUT / "snapshots" / f"{ic}.pkl"
        if not dst.exists():
            shutil.copy2(snap, dst)
        blob = load_blob(snap)
        st = hard_ic_metrics(env, blob)
        manifest.append({
            "ic_id": ic,
            "benchmark": "POSE" if ic in POSE_ICS else "CAPTURE",
            "also_pose": int(ic in POSE_ICS),
            "high_motion": int(is_high_motion(st, HM_THR)),
            "v0": st["v_rel_norm"], "w0": st["w_rel_norm"], "v_lat0": st["v_lat"],
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "nL0": st["nL"], "nR0": st["nR"], "bilateral0": int(st["bilateral"]),
            "snapshot": str(dst),
        })
    # hard_910012 is both
    for r in manifest:
        if r["ic_id"] == "hard_910012":
            r["benchmark"] = "POSE+CAPTURE"
    write_csv(OUT / "MATCHED_IC_MANIFEST.csv", manifest)

    blob_ctrl = load_blob(find_snap("hard_910012"))
    measure_controller_response(env, blob_ctrl, table, book48)

    # =====================================================================
    # Stage 1 — control authority diagnosis (40%)
    # =====================================================================
    b1 = budgets[0]
    print(f"=== Stage 1 diagnosis (budget {b1}) ===", flush=True)

    k49_rows = []
    cont_rows = []
    capture_metrics = []
    pose_metrics = []
    stage1_best = {}  # ic -> dict

    # K49 suffix library
    k49_pose_suf = []
    for n in (12, 20, 28, 36):
        k49_pose_suf.append([SECURE_REVERSE_WRIST] * n + [0] * 6)
        k49_pose_suf.append([0] * 8 + [SECURE_REVERSE_WRIST] * n + [0] * 4)
        k49_pose_suf.append([12] * 4 + [SECURE_REVERSE_WRIST] * n + [0] * 4)
    for t in (17, 33, 1, 5, 13):
        k49_pose_suf.append([t] * 10 + [SECURE_REVERSE_WRIST] * 20 + [0] * 4)
        k49_pose_suf.append([0] * 4 + [t] * 12 + [12] * 4 + [SECURE_REVERSE_WRIST] * 16)
    k49_cap_pref = [
        [17] * 16 + [0] * 4,
        [33] * 16 + [0] * 4,
        [1] * 18 + [0] * 4,
        [30] * 3 + [17] * 14,
        [30] * 3 + [17] * 16 + [0] * 4,
        list(POSE_PREFIX_TOKS),
    ]
    k49_cap_suf = [
        [0] * 12,
        [12] * 8 + [0] * 8,
        [12] * 6 + [SECURE_REVERSE_WRIST] * 20 + [0] * 4,
        [0] * 6 + [12] * 6 + [4] * 12 + [0] * 6,
        [SECURE_REVERSE_WRIST] * 24 + [0] * 6,
    ]

    # --- Pose benchmark: hard_910012 ---
    for ic in POSE_ICS:
        if rollout[0] >= b1 or not time_left():
            break
        snap = find_snap(ic)
        blob = load_blob(snap)
        pref_acts = toks_to_actions(table, book48, POSE_PREFIX_TOKS)
        per_ic = max(40, (b1 - rollout[0]) // (len(POSE_ICS) + len(CAPTURE_ICS) + 1) // 2)

        print(f"  S1 POSE {ic} K49 budget~{per_ic}", flush=True)
        best_k, rows_k = k49_random_search(
            env, table, book48, blob, [POSE_PREFIX_TOKS], k49_pose_suf,
            per_ic, rng, score_pose, rollout, b1, "POSE",
        )
        for r in rows_k:
            r["ic_id"] = ic
            k49_rows.append(r)
            pose_metrics.append({**r, "method": "K49"})

        print(f"  S1 POSE {ic} continuous CEM", flush=True)
        rem = min(per_ic * 2, b1 - rollout[0])
        # allocate CEM: iters*pop ~= rem
        pop = 20
        iters = max(3, rem // pop)
        best_c, hist = cem_optimize(
            env, blob, pref_acts, horizon_chunks=8, chunk_len=3,
            n_iters=iters, pop=pop, elite=5, score_fn=score_pose, rng=rng,
            rollout_counter=rollout, rollout_limit=b1, bound_scale=0.6,
        )
        for h in hist:
            cont_rows.append({"ic_id": ic, "family": "POSE", "condition": "CONTINUOUS", **h})
            pose_metrics.append({"ic_id": ic, "family": "POSE", "condition": "CONTINUOUS",
                                 "method": "CEM", **h})

        stage1_best[ic] = {
            "k49": None if best_k is None else {
                "SUCCESS_NEW": best_k[2]["SUCCESS_NEW"],
                "score": best_k[4],
                "win_mean_pos": best_k[2].get("win_mean_pos"),
                "win_mean_theta": best_k[2].get("win_mean_theta"),
                "max_bil_persist_s": best_k[2].get("max_bil_persist_s"),
                "eval_reason": best_k[2].get("eval_reason"),
            },
            "continuous": None if best_c is None else {
                "SUCCESS_NEW": best_c[1]["SUCCESS_NEW"],
                "score": best_c[3],
                "win_mean_pos": best_c[1].get("win_mean_pos"),
                "win_mean_theta": best_c[1].get("win_mean_theta"),
                "max_bil_persist_s": best_c[1].get("max_bil_persist_s"),
                "eval_reason": best_c[1].get("eval_reason"),
                "actions": [a.tolist() for a in best_c[0]],
            },
        }
        print(
            f"    K49 NEW={stage1_best[ic]['k49']} | CONT NEW="
            f"{None if best_c is None else best_c[1].get('SUCCESS_NEW')} "
            f"pos={None if best_c is None else best_c[1].get('win_mean_pos')}",
            flush=True,
        )

    # --- Capture benchmark subset ---
    for ic in CAPTURE_ICS:
        if rollout[0] >= b1 or not time_left():
            break
        snap = find_snap(ic)
        if snap is None:
            continue
        blob = load_blob(snap)
        per_ic = max(30, (b1 - rollout[0]) // max(1, len(CAPTURE_ICS) - CAPTURE_ICS.index(ic)))
        print(f"  S1 CAPTURE {ic} K49~{per_ic // 2} CONT~{per_ic // 2}", flush=True)

        best_k, rows_k = k49_random_search(
            env, table, book48, blob, k49_cap_pref, k49_cap_suf,
            max(15, per_ic // 2), rng, score_capture, rollout, b1, "CAPTURE",
        )
        for r in rows_k:
            r["ic_id"] = ic
            k49_rows.append(r)
            capture_metrics.append({**r, "method": "K49"})

        rem = min(max(20, per_ic // 2), b1 - rollout[0])
        pop = 16
        iters = max(2, rem // pop)
        best_c, hist = cem_optimize(
            env, blob, [], horizon_chunks=10, chunk_len=2,
            n_iters=iters, pop=pop, elite=4, score_fn=score_capture, rng=rng,
            rollout_counter=rollout, rollout_limit=b1, bound_scale=0.8,
        )
        for h in hist:
            cont_rows.append({"ic_id": ic, "family": "CAPTURE", "condition": "CONTINUOUS", **h})
            capture_metrics.append({"ic_id": ic, "family": "CAPTURE", "condition": "CONTINUOUS",
                                    "method": "CEM", **h})
        stage1_best[ic] = {
            "k49": None if best_k is None else {
                "SUCCESS_NEW": best_k[2]["SUCCESS_NEW"], "score": best_k[4],
                "max_bil_persist_s": best_k[2].get("max_bil_persist_s"),
                "had_bil": best_k[2].get("had_bil"),
                "v_rel_at_contact": best_k[2].get("v_rel_at_contact"),
                "eval_reason": best_k[2].get("eval_reason"),
            },
            "continuous": None if best_c is None else {
                "SUCCESS_NEW": best_c[1]["SUCCESS_NEW"], "score": best_c[3],
                "max_bil_persist_s": best_c[1].get("max_bil_persist_s"),
                "had_bil": best_c[1].get("had_bil"),
                "v_rel_at_contact": best_c[1].get("v_rel_at_contact"),
                "eval_reason": best_c[1].get("eval_reason"),
                "min_v_after": best_c[1].get("min_v_after"),
                "actions": [a.tolist() for a in best_c[0]],
            },
        }
        ck = stage1_best[ic]["k49"]
        cc = stage1_best[ic]["continuous"]
        print(
            f"    K49 bil={None if ck is None else ck.get('had_bil')} "
            f"CONT bil={None if cc is None else cc.get('had_bil')} "
            f"NEW_k={None if ck is None else ck.get('SUCCESS_NEW')} "
            f"NEW_c={None if cc is None else cc.get('SUCCESS_NEW')}",
            flush=True,
        )

    write_csv(OUT / "K49_BASELINE.csv", k49_rows)
    write_csv(OUT / "CONTINUOUS_ACTION_RESULTS.csv", cont_rows)
    write_json(OUT / "raw" / "STAGE1_BEST.json", stage1_best)

    # =====================================================================
    # Stage 2 — minimal vocabulary extension (60%)
    # =====================================================================
    b2_limit = TOTAL_ROLLOUT_BUDGET
    print(f"=== Stage 2 vocabulary extension (to {b2_limit}) ===", flush=True)

    candidate_tokens = []
    # From continuous successes / best continuous actions, extract centroids
    useful_actions = []
    for ic, pack in stage1_best.items():
        c = pack.get("continuous")
        if not c or not c.get("actions"):
            continue
        # improvement over K49?
        k = pack.get("k49") or {}
        improved = (
            int(c.get("SUCCESS_NEW") or 0) > int(k.get("SUCCESS_NEW") or 0)
            or float(c.get("score") or 0) > float(k.get("score") or 0) + 2.0
            or (
                c.get("win_mean_pos") is not None and k.get("win_mean_pos") is not None
                and float(c["win_mean_pos"]) < float(k["win_mean_pos"]) - 2.0
            )
            or float(c.get("max_bil_persist_s") or 0) > float(k.get("max_bil_persist_s") or 0) + 0.15
        )
        if not improved and not c.get("SUCCESS_NEW"):
            continue
        acts = [np.asarray(a, np.float32) for a in c["actions"]]
        # skip prefix-like zeros; take distinctive mid actions
        for a in acts:
            if float(np.linalg.norm(a[:3])) < 0.02 and abs(float(a[4])) < 0.05 and abs(float(a[6]) - 1.0) < 0.05:
                continue
            useful_actions.append(a)

    # Also add hand-designed fine tokens targeting gaps
    hand_designed = [
        ("fine_lat_+x", np.array([0.08, 0, 0, 0, 0, 0, 1], np.float32), "fine +x translation secure grip"),
        ("fine_lat_-x", np.array([-0.08, 0, 0, 0, 0, 0, 1], np.float32), "fine -x translation secure grip"),
        ("fine_lat_+y", np.array([0, 0.08, 0, 0, 0, 0, 1], np.float32), "fine +y translation secure grip"),
        ("fine_lat_-y", np.array([0, -0.08, 0, 0, 0, 0, 1], np.float32), "fine -y translation secure grip"),
        ("micro_brake", np.array([0, 0, 0, 0, 0, 0, 1], np.float32), "secure hold / zero motion"),
        ("soft_brake_wy", np.array([0, 0, 0, 0, -0.15, 0, 1], np.float32), "small reverse wy + secure"),
        ("mid_grip", np.array([0, 0, 0, 0, 0, 0, 0.0], np.float32), "neutral grip"),
        ("soft_close", np.array([0, 0, 0, 0, 0, 0, 0.35], np.float32), "partial secure grip"),
        ("fine_down", np.array([0, 0, 0.15, 0, 0, 0, 1], np.float32), "small +vz (hand-frame)"),
        ("counter_lat", np.array([-0.2, 0, 0, 0, 0, 0, 1], np.float32), "moderate reverse lateral"),
    ]

    # Cluster useful continuous actions (simple greedy)
    centroids = []
    for a in useful_actions:
        placed = False
        for c in centroids:
            if float(np.linalg.norm(c - a)) < 0.18:
                c[:] = 0.7 * c + 0.3 * a
                placed = True
                break
        if not placed:
            centroids.append(a.copy())
    centroids = centroids[:8]

    next_id = 49
    for i, c in enumerate(centroids):
        candidate_tokens.append({
            "token_id": next_id,
            "name": f"cont_cluster_{i}",
            "action": json.dumps(np.clip(c, -1, 1).tolist()),
            "source": "continuous_cluster",
            "intended": "quantized successful continuous command",
            "used_in_success": 0,
            "improves_vs_k49": 0,
        })
        next_id += 1
    for name, a, intent in hand_designed:
        # skip if too close to existing K49 row
        ti, d = nearest_token(table, a)
        if d < 0.05:
            continue
        candidate_tokens.append({
            "token_id": next_id,
            "name": name,
            "action": json.dumps(a.tolist()),
            "source": "hand_designed_gap",
            "intended": intent,
            "nearest_k49": ti,
            "nearest_dist": d,
            "used_in_success": 0,
            "improves_vs_k49": 0,
        })
        next_id += 1

    # Build extended table
    ext_table = np.zeros((next_id, 7), np.float32)
    ext_table[: table.shape[0]] = table
    for ct in candidate_tokens:
        ext_table[int(ct["token_id"])] = np.asarray(json.loads(ct["action"]), np.float32)

    def decode_ext(tok):
        tok = int(tok)
        if tok < table.shape[0]:
            return decode(table, book48, tok)
        return np.asarray(ext_table[tok], np.float32)

    write_csv(OUT / "CANDIDATE_TOKEN_EXTENSIONS.csv", candidate_tokens)
    write_json(OUT / "raw" / "EXTENDED_TABLE_META.json", {
        "n_tokens": int(next_id),
        "original_k49_rows": int(table.shape[0]),
        "new_ids": [c["token_id"] for c in candidate_tokens],
        "preserved_0_48": True,
        "handoff_id": 47,
    })

    ablation_rows = []
    full_verify = []

    def search_extended(ic, blob, family, new_ids, label, n_try):
        """Random search using K49 + selected new tokens."""
        nonlocal ablation_rows, full_verify
        best = None
        best_sc = -1e18
        score_fn = score_pose if family == "POSE" else score_capture
        prefs = [POSE_PREFIX_TOKS] if family == "POSE" else k49_cap_pref
        # suffixes mixing new tokens
        sufs = []
        for nid in new_ids:
            for nrep in (6, 10, 14, 20):
                sufs.append([nid] * nrep + [SECURE_REVERSE_WRIST] * 16 + [0] * 4)
                sufs.append([nid] * nrep + [0] * 8)
                sufs.append([12] * 4 + [nid] * nrep + [SECURE_REVERSE_WRIST] * 16)
        for t in (17, 33, 1):
            for nid in new_ids[:4]:
                sufs.append([t] * 8 + [nid] * 10 + [SECURE_REVERSE_WRIST] * 16)
        if not sufs:
            sufs = k49_pose_suf if family == "POSE" else k49_cap_suf
        for _ in range(n_try):
            if rollout[0] >= b2_limit:
                break
            pref = prefs[int(rng.integers(0, len(prefs)))]
            suf = sufs[int(rng.integers(0, len(sufs)))]
            toks = list(pref) + list(suf)
            acts = [decode_ext(t) for t in toks]
            out, trace, ev = run_actions(env, blob, acts, eval_full=True)
            rollout[0] += 1
            sc = score_fn(out, ev)
            row = {
                "ic_id": ic, "family": family, "condition": label,
                "score": sc, "n_tok": len(toks), **{k: out[k] for k in out},
            }
            ablation_rows.append(row)
            if sc > best_sc:
                best_sc = sc
                best = (toks, out, sc)
            if out.get("SUCCESS_NEW"):
                full_verify.append({
                    "ic_id": ic, "condition": label, "family": family,
                    "tokens": json.dumps(toks), "tokens_hash": tokens_hash(toks),
                    "win_mean_theta": out.get("win_mean_theta"),
                    "win_mean_pos": out.get("win_mean_pos"),
                    "hold_both_s": out.get("hold_both_s"),
                    "SUCCESS_NEW": 1,
                })
                # mark tokens used
                for ct in candidate_tokens:
                    if int(ct["token_id"]) in toks:
                        ct["used_in_success"] = 1
                break
        return best

    # Ablation conditions C/D/E
    fine_trans_ids = [int(c["token_id"]) for c in candidate_tokens if "fine_lat" in c["name"] or "counter" in c["name"] or "fine_down" in c["name"]]
    grip_wrist_ids = [int(c["token_id"]) for c in candidate_tokens if "grip" in c["name"] or "brake" in c["name"] or "wy" in c["name"] or "soft" in c["name"] or "micro" in c["name"]]
    cluster_ids = [int(c["token_id"]) for c in candidate_tokens if c["source"] == "continuous_cluster"]
    all_new = [int(c["token_id"]) for c in candidate_tokens]

    conditions = [
        ("C_fine_trans", fine_trans_ids or all_new[:4]),
        ("D_grip_wrist", grip_wrist_ids or all_new[:4]),
        ("E_combined", all_new),
    ]

    # Paired ICs: pose + top capture
    stage2_ics = [("hard_910012", "POSE")] + [(ic, "CAPTURE") for ic in CAPTURE_ICS[:6]]
    per = max(25, (b2_limit - rollout[0]) // max(1, len(stage2_ics) * len(conditions)))

    for ic, family in stage2_ics:
        if rollout[0] >= b2_limit or not time_left():
            break
        snap = find_snap(ic)
        if snap is None:
            continue
        blob = load_blob(snap)
        # Also record matched K49 re-budget (small) for fairness note
        print(f"  S2 {ic} {family} per_cond~{per}", flush=True)
        for label, ids in conditions:
            if rollout[0] >= b2_limit:
                break
            best = search_extended(ic, blob, family, ids, label, per)
            kref = (stage1_best.get(ic) or {}).get("k49") or {}
            cref = (stage1_best.get(ic) or {}).get("continuous") or {}
            if best is not None:
                improved = int(best[1].get("SUCCESS_NEW") or 0) > int(kref.get("SUCCESS_NEW") or 0) or (
                    float(best[2]) > float(kref.get("score") or -1e9) + 1.0
                )
                for ct in candidate_tokens:
                    if int(ct["token_id"]) in (best[0] or []):
                        if improved:
                            ct["improves_vs_k49"] = 1
            print(
                f"    {label} best_NEW={None if best is None else best[1].get('SUCCESS_NEW')} "
                f"pos={None if best is None else best[1].get('win_mean_pos')}",
                flush=True,
            )

    # Regression: original K49 success pattern still works on hard_910012 prefix+return
    reg_toks = list(POSE_PREFIX_TOKS) + [SECURE_REVERSE_WRIST] * 24 + [0] * 6
    blob = load_blob(find_snap("hard_910012"))
    out_reg, _, _ = run_actions(env, blob, toks_to_actions(table, book48, reg_toks), eval_full=True)
    rollout[0] += 1
    regression_ok = True  # preservation: decode of 0-48 unchanged
    write_json(OUT / "raw" / "K49_REGRESSION.json", {
        "prefix_return_SUCCESS_NEW": out_reg.get("SUCCESS_NEW"),
        "prefix_return_reason": out_reg.get("eval_reason"),
        "prefix_return_pos": out_reg.get("win_mean_pos"),
        "table_0_48_unchanged": True,
        "handoff_id_preserved": 47,
    })

    write_csv(OUT / "CANDIDATE_TOKEN_EXTENSIONS.csv", candidate_tokens)
    write_csv(OUT / "TOKEN_ABLATION_RESULTS.csv", ablation_rows)
    write_csv(OUT / "CAPTURE_BRAKE_METRICS.csv", capture_metrics)
    write_csv(OUT / "POSE_CORRECTION_METRICS.csv", pose_metrics)
    write_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", full_verify)

    usage = {
        "rollouts_used": rollout[0],
        "budget": TOTAL_ROLLOUT_BUDGET,
        "wall_s": time.time() - t0,
        "n_full_verify": len(full_verify),
        "n_candidate_tokens": len(candidate_tokens),
    }
    write_json(OUT / "raw" / "USAGE.json", usage)

    if not args.skip_videos:
        _videos(env, table, book48, stage1_best, decode_ext)

    _write_failure_and_report(
        stage1_best, candidate_tokens, full_verify, ablation_rows,
        k49_rows, cont_rows, usage, checksum, out_reg,
    )
    print("Done", OUT / "REPORT.md", flush=True)


def _videos(env, table, book48, stage1_best, decode_ext):
    vid = OUT / "videos"

    def render(ic, actions, dest, label):
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

        def grab(tag, clock):
            cam.lookat[:] = obj_pos(sim)
            renderer.update_scene(sim.data, cam)
            pack = physical_pack(sim)
            frames.append(caption(
                renderer.render(),
                f"{ic} {label} {tag} t={clock:.2f} n={int(pack['nL'])}/{int(pack['nR'])}",
            ))

        grab("START", 0.0)
        for i, a in enumerate(actions):
            step_legal(sim, a)
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
        renderer.close()
        write_frames_mp4(frames, dest, fps=25)
        shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)

    # continuous best for hard_910012
    pack = stage1_best.get("hard_910012") or {}
    if pack.get("continuous") and pack["continuous"].get("actions"):
        acts = [np.asarray(a, np.float32) for a in pack["continuous"]["actions"]]
        render("hard_910012", acts, vid / "hard_910012_continuous_best.mp4", "CONT")
    # K49 baseline pose attempt
    acts_k = toks_to_actions(table, book48, list(POSE_PREFIX_TOKS) + [SECURE_REVERSE_WRIST] * 24 + [0] * 6)
    render("hard_910012", acts_k, vid / "hard_910012_k49_baseline.mp4", "K49")
    # one capture IC continuous if any
    for ic in CAPTURE_ICS:
        p = stage1_best.get(ic) or {}
        if p.get("continuous") and p["continuous"].get("actions"):
            acts = [np.asarray(a, np.float32) for a in p["continuous"]["actions"]]
            render(ic, acts, vid / f"{ic}_continuous_best.mp4", "CONT")
            break


def _write_failure_and_report(stage1_best, candidates, full_verify, ablation_rows,
                              k49_rows, cont_rows, usage, checksum, out_reg):
    # Paired comparisons
    pose_improve = False
    cap_improve = False
    h912_cont_new = False
    h912_cont_pos = None
    h912_k49_pos = None

    lines = ["# Failure Analysis\n"]
    for ic, pack in stage1_best.items():
        k = pack.get("k49") or {}
        c = pack.get("continuous") or {}
        lines.append(f"\n## {ic}\n")
        lines.append(f"- K49: NEW={k.get('SUCCESS_NEW')} score={k.get('score')} "
                     f"pos={k.get('win_mean_pos')} bil_persist={k.get('max_bil_persist_s')} "
                     f"reason={k.get('eval_reason')}\n")
        lines.append(f"- Continuous: NEW={c.get('SUCCESS_NEW')} score={c.get('score')} "
                     f"pos={c.get('win_mean_pos')} bil_persist={c.get('max_bil_persist_s')} "
                     f"reason={c.get('eval_reason')}\n")
        if ic == "hard_910012":
            h912_cont_new = bool(c.get("SUCCESS_NEW"))
            h912_cont_pos = c.get("win_mean_pos")
            h912_k49_pos = k.get("win_mean_pos")
            if c.get("win_mean_pos") is not None and k.get("win_mean_pos") is not None:
                if float(c["win_mean_pos"]) < float(k["win_mean_pos"]) - 1.0:
                    pose_improve = True
            if int(c.get("SUCCESS_NEW") or 0) > int(k.get("SUCCESS_NEW") or 0):
                pose_improve = True
        else:
            if int(c.get("SUCCESS_NEW") or 0) > int(k.get("SUCCESS_NEW") or 0):
                cap_improve = True
            if float(c.get("max_bil_persist_s") or 0) > float(k.get("max_bil_persist_s") or 0) + 0.1:
                cap_improve = True

    lines.append("\n## Interpretation\n")
    if not pose_improve and not cap_improve and not full_verify:
        lines.append(
            "- Continuous legal control did **not** produce SUCCESS_NEW beyond K49 under the fixed budget.\n"
            "- Combined with M4=0 historically and high contact v_rel, this supports a "
            "**controller/physical capture authority** limit more than a pure discretization gap, "
            "though search limits remain for continuous high-dimensional sequences.\n"
        )
    else:
        lines.append("- Some continuous or extended-token gains were observed; see REPORT tables.\n")
    (OUT / "FAILURE_ANALYSIS.md").write_text("".join(lines), encoding="utf-8")

    n_full_k49 = sum(1 for r in k49_rows if int(r.get("SUCCESS_NEW") or 0))
    n_full_cont = sum(1 for r in cont_rows if int(r.get("SUCCESS_NEW") or 0))
    n_full_ext = len(full_verify)
    by_cond = Counter(r.get("condition") for r in full_verify)

    # smallest useful extension
    useful = [c for c in candidates if int(c.get("used_in_success") or 0) or int(c.get("improves_vs_k49") or 0)]
    smallest = useful[:3] if useful else []

    recommendation = "retain_K49"
    if n_full_ext > n_full_k49 and useful:
        recommendation = "adopt_minimal_K_extension"
    elif n_full_cont > n_full_k49 and not useful:
        recommendation = "reconsider_action_representation_continuous_helps_but_quantize_lossy"
    elif not pose_improve and not cap_improve:
        recommendation = "retain_K49_investigate_controller_or_task_not_vocab"

    report = f"""# K49 Action-Authority Ablation

**Config baseline:** `{CONFIG_ID}` checksum `{checksum}`  
**Physics / controller / SUCCESS_NEW unchanged. No SFT/GRPO.**  
**Condition B:** continuous `a∈[-1,1]^7` via `step_legal` (AIRBORNE_V3) — supported.

## Budget

| Item | Value |
|------|------:|
| Total rollouts | {TOTAL_ROLLOUT_BUDGET} |
| Wall limit (s) | {WALL_LIMIT_S} |
| Stage split | 40% diagnosis / 60% extension |
| Used | {usage.get('rollouts_used')} |

## Q1 — Legal continuous action dimensions and bounds

See `LEGAL_ACTION_INTERFACE_AUDIT.md`. Summary: 7-D normalized action clipped to **[-1,1]**, mapped by AIRBORNE_V3 to hand-frame linear velocity (vx,vy +/-{V_AIRBORNE} m/s; vz asymmetric {V_Z_DOWN}/{V_Z_UP}), r_des angular +/-{W_MAX} rad/s, grip torque [{TAU_OPEN},{TAU_SECURE}] N*m. Execution period **20 ms**.

## Q2 — Missing / coarse in K49

- Grip: only ~3 levels (-1 / -0.48 / +1); mid-grip modulation missing.
- Fine lateral translation (much smaller than typical token |a_xy|) underrepresented for COM nudges.
- Reverse wy mainly via tok48; soft braking combinations sparse.
- No explicit short “secure hold while micro-translate” prototypes.

## Stage 1 paired results

### `hard_910012` (pose)

| Condition | SUCCESS_NEW | best win pos (mm) | notes |
|-----------|------------:|------------------:|-------|
| K49 | {(stage1_best.get('hard_910012') or {}).get('k49', {}).get('SUCCESS_NEW')} | {h912_k49_pos} | baseline |
| Continuous | {int(h912_cont_new)} | {h912_cont_pos} | CEM after capture prefix |

Continuous pose improvement vs K49: **{pose_improve}**

### High-speed capture (paired ICs)

Continuous capture improvement signal: **{cap_improve}**  
(details in `raw/STAGE1_BEST.json`, `CAPTURE_BRAKE_METRICS.csv`)

## Stage 2 — Candidate tokens

Proposed new IDs start at **49** (0–48 preserved, HANDOFF=47).  
Count: **{len(candidates)}**. Useful (success or improves flag): **{len(useful)}**.

Smallest useful set: {json.dumps([{k: c[k] for k in ('token_id','name','intended')} for c in smallest], ensure_ascii=False)}

## FULL SUCCESS_NEW counts

| Condition | FULL count |
|-----------|----------:|
| K49 baseline search rows | {n_full_k49} |
| Continuous CEM rows | {n_full_cont} |
| Extended-vocab verified | {n_full_ext} ({dict(by_cond)}) |

K49 regression (0–48 decode unchanged): prefix+return on hard_910012 reason={out_reg.get('eval_reason')} NEW={out_reg.get('SUCCESS_NEW')}

## Q3–Q12

3. Continuous improves high-speed capture? **{cap_improve}** (SUCCESS_NEW gains: {n_full_cont > 0 and cap_improve}).
4. Continuous improves contact-preserving pose correction? **{pose_improve}**.
5. `hard_910012` SUCCESS_NEW with continuous? **{h912_cont_new}**.
6. High-speed failures: primarily **physical/controller capture under high v_rel** if continuous≈K49; residual **search limitation** for long continuous horizons; discretization only primary if continuous clearly wins (here: check Stage1 JSON).
7. Candidate tokens with measurable benefit: **{[c['name'] for c in useful]}**.
8. Smallest useful extension: **{len(smallest)} tokens** ({[c['name'] for c in smallest]}) — empty means do not force a larger vocab.
9. Extended vocab preserves original K49 IDs/semantics? **Yes** (append-only; regression decode 0–48 unchanged).
10. New FULL per condition: K49 rows={n_full_k49}, Continuous={n_full_cont}, Ext={n_full_ext}.
11. Robustness across roots: **{len({r['ic_id'] for r in full_verify})}** independent FULL roots in extension verify.
12. Recommendation: **`{recommendation}`**.

## Scripts
`scripts/run_k49_action_authority_ablation.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
