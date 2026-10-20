"""Paired experiment: return arm to nominal pose before HANDOFF (hard_910263).

Tests whether COM regression after sliding is driven by HANDOFF snapping
r_des to the saved nominal while the hand is still tilted.

Does not modify physics, K48, HANDOFF semantics, reward, or SUCCESS_12S.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, map_action
from controllers.recovery.geometry_switch import rot_angle
from envs.outcomes.horizon_outcome import NOMINAL_S, recovery_failed
from envs.outcomes.nominal_pose import errors as pose_errors
from envs.outcomes.nominal_pose import hand_R
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.full_objective import (
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    evaluate_tokens,
    window_stats,
)
from evaluation.grasp_configuration import measure
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.recovery_episode import make_training_env
from envs.ballistic_impact import z_tgt_of
from envs.outcomes.horizon_outcome import LOST_STEPS, _bilateral, _dropped, _monitor
from envs.outcomes.table_drop import latched
from training.recovery_execution import _escaped
from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from envs.outcomes.table_drop import object_vz, observe
from envs.step_legal import N_SUB, TRACK

OUT = ROOT / "results" / "diagnostics" / "return_to_nominal_pose_before_handoff"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
SNAP = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910263.pkl"
SEED = 20261018

CPP = [4] * 20 + [12] * 5 + [0] * 3
# Best B_cpp_slide suffix from sliding pilot (after CPP)
SLIDE = [17] * 16 + [12] * 5 + [0] * 5
BASE = CPP + SLIDE  # Branch A sequence
RECOVERY_LIMIT_S = 4.0


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


def snapshot_stage(env, blob, label, t0, extra=None):
    sim = env.sim
    m = measure(sim, U_TGT, P_TGT)
    cs = contact_summary(sim)
    pe = pose_errors(sim, blob["ref"])
    Rh = hand_R(sim)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    Rhand_ref = np.asarray(blob["ref"]["R_hand"], float).reshape(3, 3)
    row = {
        "stage": label,
        "t_abs": float(sim.data.time),
        "t_since_ic": float(sim.data.time) - t0,
        "e_theta_deg": m["e_theta_deg"],
        "e_pos_mm": m["e_pos_mm"],
        "nL": cs["nL"],
        "nR": cs["nR"],
        "bilateral": cs["bilateral"],
        "obj_z": m["obj_z"],
        "hand_to_ref_rdes_deg": float(np.degrees(rot_angle(Rref, Rh))),
        "hand_to_ref_Rhand_deg": float(np.degrees(rot_angle(Rhand_ref, Rh))),
        "rdes_to_ref_rdes_deg": float(np.degrees(rot_angle(Rref, Rd))),
        "hand_to_live_rdes_deg": float(np.degrees(rot_angle(Rd, Rh))),
        "ep_nom_m": pe["ep"],
        "eR_hand_ref_rad": pe["eR"],
        "vrel": pe["vrel"],
        "qpos": np.asarray(sim.data.qpos, float).copy().tolist(),
        "qvel_norm": float(np.linalg.norm(sim.data.qvel)),
        "p_des": np.asarray(sim.fsm.p_des, float).reshape(-1)[:3].tolist(),
        "p_hand": np.asarray(sim.data.xpos[sim.ids.hand_body], float).tolist(),
        "r_des": Rd.tolist(),
        "R_hand": Rh.tolist(),
        "v_rel_h": m["v_rel_h"],
        "w_rel_h": m["w_rel_h"],
        "grip_ctrl": None if sim.data.ctrl.size <= 7 else float(sim.data.ctrl[7]),
    }
    if extra:
        row.update(extra)
    return row


def run_tokens(env, book, blob, tokens, t0, lost_low, stage_rows, stage_name_prefix, log_every=1):
    for i, tok in enumerate(tokens):
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if (i + 1) % log_every == 0 or i == len(tokens) - 1:
            stage_rows.append(
                snapshot_stage(env, blob, f"{stage_name_prefix}_tok{i}_id{tok}", t0, {"token": int(tok), "tok_i": i})
            )
        if failed:
            return False, lost_low, terminal_reason_safe(env)
    return True, lost_low, None


def terminal_reason_safe(env):
    from envs.outcomes.horizon_outcome import terminal_reason

    return terminal_reason(env.sim) or "DROP"


def step_legal_sub(sim, action, sub_log):
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
        sub_log.append(
            {
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "nL": cs["nL"],
                "nR": cs["nR"],
                "hand_to_ref_rdes_deg": None,  # filled by caller optionally
            }
        )
    return action


def codebook_wrist_inventory(book):
    rows = []
    for i in range(47):
        a = np.asarray(book[i], float)
        rows.append({
            "token": i,
            "a6": float(a[6]),
            "wx": float(a[3]),
            "wy": float(a[4]),
            "wz": float(a[5]),
            "secure": bool(a[6] > 0.5),
            "mid": bool(abs(a[6] + 0.48) < 0.05),
            "open": bool(a[6] < -0.5),
        })
    return rows


def greedy_return_tokens(env, book, blob, t0, lost_low, stage_rows, max_steps=80, secure_only=True, tol_deg=5.0):
    """Pick legal tokens that reduce hand↔ref[r_des] orientation error.

    Returns (ok, lost_low, tokens_used, sub_trace, stuck_reason).
    """
    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    used = []
    sub_trace = []
    # Candidate pool
    cands = []
    for i in range(47):
        a = np.asarray(book[i], float)
        if secure_only and a[6] < 0.5:
            continue
        if not secure_only and a[6] > -0.4:  # allow mid+open+secure when not secure_only
            pass
        cands.append(i)
    if not cands:
        cands = list(range(47))

    for step in range(max_steps):
        Rh = hand_R(env.sim)
        err0 = float(np.degrees(rot_angle(Rref, Rh)))
        if err0 <= tol_deg:
            stage_rows.append(snapshot_stage(env, blob, "return_converged", t0, {"hand_err_deg": err0}))
            return True, lost_low, used, sub_trace, None

        # remaining time
        elapsed = float(env.sim.data.time) - t0
        if elapsed > RECOVERY_LIMIT_S - 0.05:
            return False, lost_low, used, sub_trace, "RECOVERY_DEADLINE"

        best = None
        # One-step lookahead on a cloned... we can't clone easily; try each candidate
        # from current state by save/restore capture
        cap = take_capture(env)
        prev = np.asarray(env._prev, np.float32).copy() if env._prev is not None else np.zeros(7, np.float32)
        for tok in cands:
            restore_capture(env, cap)
            env._prev = prev.copy()
            _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
            step_legal(env.sim, action)
            Rh2 = hand_R(env.sim)
            err1 = float(np.degrees(rot_angle(Rref, Rh2)))
            m = measure(env.sim, U_TGT, P_TGT)
            cs = contact_summary(env.sim)
            # score: reduce orientation error; keep bilateral; don't explode pos
            score = (err0 - err1) - 0.01 * m["e_pos_mm"] + (2.0 if cs["bilateral"] else -20.0)
            if best is None or score > best[0]:
                best = (score, tok, err1, m["e_pos_mm"], cs["bilateral"])

        restore_capture(env, cap)
        env._prev = prev.copy()
        if best is None or best[0] <= 0.05:
            # no improving token
            stage_rows.append(snapshot_stage(env, blob, "return_stuck", t0, {
                "hand_err_deg": err0, "best_score": None if not best else best[0],
                "best_tok": None if not best else best[1],
            }))
            return False, lost_low, used, sub_trace, "NO_IMPROVING_TOKEN"

        tok = int(best[1])
        _r, action = decode_action(book, tok, np.zeros(7, np.float32))
        action = step_legal_sub(env.sim, action, sub_trace)
        env._prev = np.asarray(action, np.float32)
        used.append(tok)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        stage_rows.append(snapshot_stage(env, blob, f"return_step{step}_tok{tok}", t0, {
            "token": tok, "pred_score": best[0], "hand_err_deg": float(np.degrees(rot_angle(Rref, hand_R(env.sim)))),
        }))
        if failed:
            return False, lost_low, used, sub_trace, "DROP_DURING_RETURN"

    Rh = hand_R(env.sim)
    errf = float(np.degrees(rot_angle(Rref, Rh)))
    return errf <= tol_deg, lost_low, used, sub_trace, None if errf <= tol_deg else "MAX_STEPS"


def run_nominal_logged(env, blob, t_h, samples, stage_rows, t0, sample_dt=0.02):
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    next_sample = 0.0
    # dense early samples for regression onset
    early_log_times = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 12.0]
    early_i = 0
    if latched(env.sim):
        reason = "TABLE_DROP"
    while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(env.sim)
        t_rel = float(env.sim.data.time) - t_h
        if t_rel + 1e-12 >= next_sample:
            m = measure(env.sim, U_TGT, P_TGT)
            cs = contact_summary(env.sim)
            samples.append({
                "t_rel": t_rel,
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "bilateral": cs["bilateral"],
                "obj_z": m["obj_z"],
            })
            next_sample += sample_dt
        while early_i < len(early_log_times) and t_rel + 1e-9 >= early_log_times[early_i]:
            stage_rows.append(snapshot_stage(env, blob, f"nominal_t{early_log_times[early_i]:.2f}", t0, {
                "t_rel_handoff": t_rel,
            }))
            early_i += 1
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
    return reason


def run_branch(env, book, blob, branch_name, return_mode):
    """return_mode: None | 'secure' | 'any_grip'"""
    restore_capture(env, blob["capture"])
    prev0 = blob["capture"].get("prev")
    env._prev = np.zeros(7, np.float32) if prev0 is None else np.asarray(prev0, np.float32)
    t0 = float(env.sim.data.time)
    stage_rows = []
    stage_rows.append(snapshot_stage(env, blob, "IC", t0))

    lost_low = 0
    ok, lost_low, why = run_tokens(env, book, blob, CPP, t0, lost_low, stage_rows, "cpp", log_every=5)
    if not ok:
        return {"branch": branch_name, "ok": False, "reason": why, "stages": stage_rows}
    stage_rows.append(snapshot_stage(env, blob, "after_cpp", t0))

    ok, lost_low, why = run_tokens(env, book, blob, SLIDE, t0, lost_low, stage_rows, "slide", log_every=4)
    if not ok:
        return {"branch": branch_name, "ok": False, "reason": why, "stages": stage_rows}
    stage_rows.append(snapshot_stage(env, blob, "after_slide", t0))

    # re-tighten already ends SLIDE with [0]*5; explicit extra tighten
    ok, lost_low, why = run_tokens(env, book, blob, [0, 0, 0], t0, lost_low, stage_rows, "retighten", log_every=1)
    if not ok:
        return {"branch": branch_name, "ok": False, "reason": why, "stages": stage_rows}
    stage_rows.append(snapshot_stage(env, blob, "after_retighten", t0))

    return_tokens = []
    sub_trace = []
    return_meta = {"attempted": False, "converged": False, "stuck_reason": None}
    if return_mode:
        stage_rows.append(snapshot_stage(env, blob, "before_return", t0))
        # Branch C: grasp check
        cs = contact_summary(env.sim)
        m = measure(env.sim, U_TGT, P_TGT)
        grasp_ok = cs["bilateral"] and m["e_theta_deg"] < 15.0
        return_meta["grasp_check_bilateral"] = cs["bilateral"]
        return_meta["grasp_check_theta"] = m["e_theta_deg"]
        return_meta["grasp_check_pass"] = grasp_ok
        if return_mode == "secure" or (return_mode == "any_grip" and grasp_ok) or return_mode == "any_grip":
            secure_only = return_mode == "secure"
            ok_r, lost_low, return_tokens, sub_trace, stuck = greedy_return_tokens(
                env, book, blob, t0, lost_low, stage_rows,
                max_steps=60, secure_only=secure_only, tol_deg=5.0,
            )
            return_meta.update({
                "attempted": True,
                "converged": ok_r,
                "stuck_reason": stuck,
                "n_return_tokens": len(return_tokens),
                "return_tokens": return_tokens,
            })
            stage_rows.append(snapshot_stage(env, blob, "after_return", t0, {"return_converged": ok_r}))
            # annotate subtrace with hand err
            Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
            # subtrace already has metrics; add summary
            if sub_trace:
                return_meta["sub_n"] = len(sub_trace)
                return_meta["sub_theta0"] = sub_trace[0]["e_theta_deg"]
                return_meta["sub_theta1"] = sub_trace[-1]["e_theta_deg"]
                return_meta["sub_pos0"] = sub_trace[0]["e_pos_mm"]
                return_meta["sub_pos1"] = sub_trace[-1]["e_pos_mm"]
                return_meta["sub_d_theta"] = float(sub_trace[-1]["e_theta_deg"] - sub_trace[0]["e_theta_deg"])
                return_meta["sub_d_pos"] = float(sub_trace[-1]["e_pos_mm"] - sub_trace[0]["e_pos_mm"])
                bi = sum(1 for s in sub_trace if s["nL"] > 0 and s["nR"] > 0) / len(sub_trace)
                return_meta["sub_frac_bilateral"] = bi

    stage_rows.append(snapshot_stage(env, blob, "before_handoff", t0))
    handoff_audit = apply_handoff(env.sim, blob["ref"])
    stage_rows.append(snapshot_stage(env, blob, "after_handoff", t0, {
        "handoff_r_des_to_hand_rad": handoff_audit["r_des_to_hand_rad"],
        "handoff_r_des_to_saved_rad": handoff_audit["r_des_to_saved_rad"],
    }))

    samples = []
    t_h = float(env.sim.data.time)
    reason = run_nominal_logged(env, blob, t_h, samples, stage_rows, t0)
    m_final = measure(env.sim, U_TGT, P_TGT)
    win = window_stats(samples, float(NOMINAL_S) - 2.0)
    s12 = reason == "SUCCESS_12S"
    success_new = bool(s12 and win.get("working_SUCCESS_NEW_hold"))

    # find onset of pos growth after handoff
    onset = None
    if samples:
        p0 = samples[0]["e_pos_mm"]
        for s in samples:
            if s["e_pos_mm"] > p0 + 1.0:
                onset = s["t_rel"]
                break

    return {
        "branch": branch_name,
        "ok": True,
        "reason": reason,
        "success_12s": s12,
        "SUCCESS_NEW": success_new,
        "label": (
            "VERIFIED_FULL_EXPERT" if success_new else
            ("FAILED_OR_DROPPED" if not s12 else "PARTIAL")
        ),
        "win_mean_theta": win.get("mean_theta"),
        "win_mean_pos": win.get("mean_pos"),
        "win_max_theta": win.get("max_theta"),
        "win_max_pos": win.get("max_pos"),
        "hold_both_s": win.get("longest_both_ok_s"),
        "handoff_audit": handoff_audit,
        "return_meta": return_meta,
        "pos_growth_onset_s_after_handoff": onset,
        "e_pos_at_handoff": next(r["e_pos_mm"] for r in reversed(stage_rows) if r["stage"] == "before_handoff"),
        "e_theta_at_handoff": next(r["e_theta_deg"] for r in reversed(stage_rows) if r["stage"] == "before_handoff"),
        "hand_err_at_handoff_deg": next(r["hand_to_ref_rdes_deg"] for r in reversed(stage_rows) if r["stage"] == "before_handoff"),
        "hand_to_rdes_at_handoff_deg": next(r["hand_to_live_rdes_deg"] for r in reversed(stage_rows) if r["stage"] == "before_handoff"),
        "stages": stage_rows,
        "sub_return": sub_trace,
        "nominal_samples_head": samples[:20],
        "base_tokens": BASE,
        "return_tokens": return_tokens,
        "full_tokens": BASE + [0, 0, 0] + return_tokens,
    }


def render_video(env, book, blob, tokens, dest, label):
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
            _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
            step_legal(env.sim, action)
            grab()
        apply_handoff(env.sim, blob["ref"])
        for i in range(60):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RAW").mkdir(exist_ok=True)
    (OUT / "SNAPSHOTS").mkdir(exist_ok=True)
    (OUT / "VIDEOS").mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    blob = pickle.loads(SNAP.read_bytes())
    # copy snapshot
    import shutil
    shutil.copy2(SNAP, OUT / "SNAPSHOTS" / "hard_910263.pkl")

    inv = codebook_wrist_inventory(book)
    write_csv(OUT / "RAW" / "k48_wrist_inventory.csv", inv)
    secure_w = [r for r in inv if r["secure"] and (abs(r["wy"]) > 0.05 or abs(r["wx"]) > 0.05)]
    write_json(OUT / "RAW" / "secure_wrist_tokens.json", {
        "secure_nonzero_w": secure_w,
        "note": "K48 has only +wy secure wrist tokens (4,6,10); no secure reverse-wy",
    })

    # IC hand vs ref
    restore_capture(env, blob["capture"])
    write_json(OUT / "RAW" / "ic_pose_audit.json", snapshot_stage(env, blob, "IC_audit", float(env.sim.data.time)))

    branches = [
        ("A_existing_slide_handoff", None),
        ("B_return_secure_then_handoff", "secure"),
        ("C_return_anygrip_then_handoff", "any_grip"),
    ]
    results = []
    all_stages = []
    for name, mode in branches:
        print("BRANCH", name, flush=True)
        res = run_branch(env, book, blob, name, mode)
        write_json(OUT / "RAW" / f"{name}.json", {k: v for k, v in res.items() if k not in ("stages", "sub_return")})
        write_json(OUT / "RAW" / f"{name}_stages.json", res.get("stages"))
        if res.get("sub_return"):
            write_json(OUT / "RAW" / f"{name}_return_substeps.json", res["sub_return"][::5])  # downsample
        results.append(res)
        for s in res.get("stages") or []:
            all_stages.append({"branch": name, **{k: v for k, v in s.items() if k not in ("qpos", "r_des", "R_hand")}})

        # also full-objective check via evaluate_tokens for Branch A exact BASE
        if name.startswith("A_"):
            ev = evaluate_tokens(env, book, blob, BASE)
            write_json(OUT / "RAW" / "A_evaluate_tokens_BASE.json", {
                "SUCCESS_NEW": ev["SUCCESS_NEW"], "s12": ev["success_12s"],
                "win": ev["window"], "label": ev["label"],
            })

    paired = []
    for res in results:
        paired.append({
            "branch": res["branch"],
            "success_12s": int(res.get("success_12s") or 0),
            "SUCCESS_NEW": int(res.get("SUCCESS_NEW") or 0),
            "reason": res.get("reason"),
            "win_mean_theta": res.get("win_mean_theta"),
            "win_mean_pos": res.get("win_mean_pos"),
            "hold_both_s": res.get("hold_both_s"),
            "e_pos_at_handoff": res.get("e_pos_at_handoff"),
            "e_theta_at_handoff": res.get("e_theta_at_handoff"),
            "hand_err_at_handoff_deg": res.get("hand_err_at_handoff_deg"),
            "hand_to_live_rdes_at_handoff_deg": res.get("hand_to_rdes_at_handoff_deg"),
            "handoff_r_des_to_hand_rad": None if not res.get("handoff_audit") else res["handoff_audit"]["r_des_to_hand_rad"],
            "pos_growth_onset_s": res.get("pos_growth_onset_s_after_handoff"),
            "return_converged": None if not res.get("return_meta") else res["return_meta"].get("converged"),
            "return_stuck_reason": None if not res.get("return_meta") else res["return_meta"].get("stuck_reason"),
            "return_n_tokens": None if not res.get("return_meta") else res["return_meta"].get("n_return_tokens"),
            "sub_d_pos_during_return": None if not res.get("return_meta") else res["return_meta"].get("sub_d_pos"),
            "sub_d_theta_during_return": None if not res.get("return_meta") else res["return_meta"].get("sub_d_theta"),
        })
    write_csv(OUT / "PAIRED_RESULTS.csv", paired)
    write_csv(OUT / "STAGE_METRICS.csv", all_stages)

    if not args.skip_videos:
        render_video(env, book, blob, BASE, OUT / "VIDEOS" / "A_slide_handoff.mp4", "A")
        # B/C may have return tokens
        for res in results:
            if res["branch"].startswith("B_") or res["branch"].startswith("C_"):
                toks = res.get("full_tokens") or BASE
                render_video(env, book, blob, toks, OUT / "VIDEOS" / f"{res['branch']}.mp4", res["branch"])

    write_json(OUT / "summary.json", {"paired": paired, "seed": args.seed})
    print("DONE", OUT, flush=True)
    for p in paired:
        print(p["branch"], "NEW", p["SUCCESS_NEW"], "pos_win", p["win_mean_pos"],
              "hand_err", p["hand_err_at_handoff_deg"], "onset", p["pos_growth_onset_s"],
              "stuck", p["return_stuck_reason"], flush=True)


if __name__ == "__main__":
    main()
