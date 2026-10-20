"""K49 secure reverse-wrist + full-objective expert pilot.

Appends token 48 = exact secure -wy mirror of K48 token 4.
HANDOFF remains 47. Original tokens 0-46 unchanged.
No training; no physics/reward/SUCCESS_12S changes.
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
    guide_score_after_prefix,
    window_stats,
)
from evaluation.grasp_configuration import measure
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import (
    HANDOFF,
    SECURE_REVERSE_WRIST,
    assert_k48_prefix_identical,
    build_k49_table,
    build_k50_table,
    decode_action as decode_k49,
    logit_layout,
    motion_token_ids,
    save_vocab_artifacts,
)
from training.k47_codebook import decode_action as decode_k48
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env
from envs.outcomes.horizon_outcome import NOMINAL_S, recovery_failed
from envs.ballistic_impact import z_tgt_of
from envs.outcomes.horizon_outcome import LOST_STEPS, _bilateral, _dropped, _monitor
from envs.outcomes.table_drop import latched
from training.recovery_execution import _escaped

OUT = ROOT / "results" / "expert_generation" / "extended_k49_k50_pose_recovery"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
SNAP = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910263.pkl"
SEED = 20261018
RECOVERY_LIMIT_S = 4.0

CPP = [4] * 20 + [12] * 5 + [0] * 3
SLIDE = [17] * 16 + [12] * 5 + [0] * 5
BASE = CPP + SLIDE
RETIGHTEN = [0, 0, 0]

# Additional ICs for bounded expert expansion
EXTRA_ICS = [
    {"id": "hrv1_920185", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920185.pkl",
     "prefix": [30, 30] + [10] * 15 + [12] * 5 + [0] * 5},
    {"id": "hrv1_920193", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920193.pkl",
     "prefix": [32] + [6] * 8 + [12] * 5 + [0] * 5},
    {"id": "hard_910277", "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910277.pkl",
     "prefix": [44, 44] + [10] * 8 + [12] * 5 + [0] * 5},
    {"id": "hrv1_920031", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920031.pkl",
     "prefix": [6] * 20 + [12] * 15 + [0] * 8},
    {"id": "hard_910101", "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910101.pkl",
     "prefix": [32] + [10] * 8 + [12] * 15 + [0] * 5},
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


def stage_metrics(env, blob, label):
    sim = env.sim
    m = measure(sim, U_TGT, P_TGT)
    cs = contact_summary(sim)
    Rh = hand_R(sim)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    return {
        "stage": label,
        "t": float(sim.data.time),
        "e_theta_deg": m["e_theta_deg"],
        "e_pos_mm": m["e_pos_mm"],
        "nL": cs["nL"],
        "nR": cs["nR"],
        "bilateral": cs["bilateral"],
        "hand_to_ref_deg": float(np.degrees(rot_angle(Rref, Rh))),
        "rdes_to_ref_deg": float(np.degrees(rot_angle(Rref, Rd))),
        "hand_to_rdes_deg": float(np.degrees(rot_angle(Rd, Rh))),
        "obj_z": m["obj_z"],
        "v_rel_h": m["v_rel_h"],
        "w_rel_h": m["w_rel_h"],
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
    """Like evaluate_tokens but supports token 48 via K49 table."""
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
            }
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
    # onset of pos growth
    onset = None
    if samples:
        p0 = samples[0]["e_pos_mm"]
        for s in samples:
            if s["e_pos_mm"] > p0 + 1.0:
                onset = s["t_rel"]
                break
    return {
        "tokens": list(tokens),
        "success_12s": s12,
        "reason": reason,
        "m0": m0,
        "m_handoff": m_hand,
        "m_final": m_final,
        "recovery_elapsed": rec,
        "window": win,
        "SUCCESS_NEW": success_new,
        "label": "VERIFIED_FULL_EXPERT" if success_new else ("FAILED_OR_DROPPED" if not s12 else "PARTIAL"),
        "observations": observations,
        "actions": actions,
        "contacts": contacts,
        "residuals": residuals,
        "handoff_obs": handoff_obs,
        "pos_growth_onset_s": onset,
        "e_pos_at_handoff": m_hand["e_pos_mm"],
        "e_theta_at_handoff": m_hand["e_theta_deg"],
        "nominal_samples_head": samples[:15],
    }


def return_with_token48(env, table, book48, blob, t0_ic, max_steps=40, tol_hand_deg=8.0, tol_rdes_deg=8.0, tol_track_deg=5.0):
    """Apply token 48 until hand and live r_des near ref, or max_steps."""
    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    used = []
    stages = []
    lost_low = 0
    for i in range(max_steps):
        st = stage_metrics(env, blob, f"return_{i}")
        stages.append(st)
        if (st["hand_to_ref_deg"] <= tol_hand_deg and st["rdes_to_ref_deg"] <= tol_rdes_deg
                and st["hand_to_rdes_deg"] <= tol_track_deg and st["bilateral"]):
            return True, used, stages, "CONVERGED"
        if float(env.sim.data.time) - t0_ic > RECOVERY_LIMIT_S - 0.05:
            return False, used, stages, "DEADLINE"
        ok, lost_low = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost_low)
        used.append(SECURE_REVERSE_WRIST)
        if not ok:
            return False, used, stages, "DROP"
        # lost_low already updated by run_seq
    st = stage_metrics(env, blob, "return_end")
    stages.append(st)
    ok = st["hand_to_ref_deg"] <= tol_hand_deg and st["rdes_to_ref_deg"] <= tol_rdes_deg
    return ok, used, stages, "CONVERGED" if ok else "MAX_STEPS"


def write_action_audit(out_dir, book48, table):
    rows = []
    for i in [0, 4, 6, 10, 12, 33, 30, 32]:
        a = book48[i]
        rows.append({
            "token": i,
            "a": a.tolist(),
            "a6": float(a[6]),
            "wy": float(a[4]),
            "wx": float(a[3]),
            "vz_cmd_norm": float(a[2]),
            "note": {
                0: "secure zero motion",
                4: "secure +wy (+vx)",
                6: "secure +wy pure",
                10: "secure +wx/+wy",
                12: "mid unload zero motion",
                33: "secure -vz translate",
                30: "open +wy",
                32: "open translate tiny w",
            }.get(i, ""),
        })
    rows.append({
        "token": 47, "a": None, "a6": None, "wy": None, "note": "HANDOFF (not a motion code)",
    })
    rows.append({
        "token": 48, "a": table[48].tolist(), "a6": float(table[48, 6]), "wy": float(table[48, 4]),
        "note": "K49 NEW: exact mirror of token 4 (secure -wy)",
    })
    write_csv(out_dir / "RAW" / "action_mapping_rows.csv", rows)
    neg_secure = [i for i in range(47) if book48[i, 6] > 0.5 and book48[i, 4] < -0.05]
    write_json(out_dir / "RAW" / "action_mapping_summary.json", {
        "secure_positive_wy_tokens": [i for i in range(47) if book48[i, 6] > 0.5 and book48[i, 4] > 0.05],
        "secure_negative_wy_tokens_k48": neg_secure,
        "k48_has_secure_reverse_wy": len(neg_secure) > 0,
        "token_48": table[48].tolist(),
        "token_4": book48[4].tolist(),
        "handoff_id": HANDOFF,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--try-k50", action="store_true", help="also build/test mid reverse token 49")
    parser.add_argument("--max-extra-ics", type=int, default=5)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("RAW", "SNAPSHOTS", "VIDEOS", "trajectories"):
        (OUT / sub).mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48)
    assert_k48_prefix_identical(table, book48)
    meta = save_vocab_artifacts(OUT, include_k50=args.try_k50)
    write_json(OUT / "vocab_meta.json", {**meta, "logit_layout_k48": logit_layout("K48"),
                                         "logit_layout_k49": logit_layout("K49")})
    write_action_audit(OUT, book48, table)

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    blob = pickle.loads(SNAP.read_bytes())
    shutil.copy2(SNAP, OUT / "SNAPSHOTS" / "hard_910263.pkl")

    # --- Physical reverse test from post retighten state ---
    print("PHYS_REVERSE_TEST", flush=True)
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    ok, _ = run_seq(env, table, book48, blob, BASE + RETIGHTEN)
    assert ok
    before = stage_metrics(env, blob, "before_return")
    write_json(OUT / "RAW" / "before_return_state.json", before)

    # A: K48-only return attempt (tok33 like prior)
    restore_capture(env, blob["capture"])
    run_seq(env, table, book48, blob, BASE + RETIGHTEN)
    ok_a, _ = run_seq(env, table, book48, blob, [33] * 20)
    after_a = stage_metrics(env, blob, "after_k48_tok33x20")
    # B: token 48 alone
    restore_capture(env, blob["capture"])
    run_seq(env, table, book48, blob, BASE + RETIGHTEN)
    hand0 = before["hand_to_ref_deg"]
    stages_b = [before]
    lost_low = 0
    for i in range(25):
        run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost_low)
        st = stage_metrics(env, blob, f"tok48_step{i}")
        stages_b.append(st)
    after_b = stages_b[-1]
    # Confirm direction: hand_to_ref should decrease
    write_json(OUT / "RAW" / "physical_reverse_trace.json", {
        "before": before,
        "after_k48_tok33": after_a,
        "after_tok48_x25": after_b,
        "hand_err_delta_tok48": float(after_b["hand_to_ref_deg"] - before["hand_to_ref_deg"]),
        "rdes_err_delta_tok48": float(after_b["rdes_to_ref_deg"] - before["rdes_to_ref_deg"]),
        "steps": stages_b,
    })
    print("  before hand_err", round(before["hand_to_ref_deg"], 2),
          "after48", round(after_b["hand_to_ref_deg"], 2),
          "after33", round(after_a["hand_to_ref_deg"], 2), flush=True)

    need_k50 = False
    if after_b["hand_to_ref_deg"] > 15 and args.try_k50:
        need_k50 = True
    # If secure reverse works, K49 sufficient
    vocab_final = "K50" if need_k50 else "K49"
    if need_k50:
        table = build_k50_table(book48)

    # --- Full objective paired ---
    print("FULL_OBJECTIVE", flush=True)
    paired_rows = []

    # A: original K48 base -> HANDOFF
    res_a = evaluate_tokens_vocab(env, table, book48, blob, BASE)
    paired_rows.append({"branch": "A_k48_cpp_slide_handoff", **_pack(res_a), "hand_err_pre_handoff": None})

    # B: base + retighten + tok48 return + HANDOFF
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    run_seq(env, table, book48, blob, BASE + RETIGHTEN)
    ok_r, ret_toks, ret_stages, why = return_with_token48(env, table, book48, blob, t0, max_steps=30)
    pre_h = stage_metrics(env, blob, "pre_handoff_B")
    write_json(OUT / "RAW" / "branch_B_return_stages.json", ret_stages)
    seq_b = BASE + RETIGHTEN + ret_toks
    res_b = evaluate_tokens_vocab(env, table, book48, blob, seq_b)
    paired_rows.append({
        "branch": "B_k49_cpp_slide_return48_handoff", **_pack(res_b),
        "return_converged": ok_r, "return_why": why, "n_return": len(ret_toks),
        "hand_err_pre_handoff": pre_h["hand_to_ref_deg"],
        "rdes_err_pre_handoff": pre_h["rdes_to_ref_deg"],
        "track_err_pre_handoff": pre_h["hand_to_rdes_deg"],
        "e_pos_pre_handoff": pre_h["e_pos_mm"],
        "e_theta_pre_handoff": pre_h["e_theta_deg"],
    })
    print("  B NEW", res_b["SUCCESS_NEW"], "win_pos", res_b["window"].get("mean_pos"),
          "hand_err", round(pre_h["hand_to_ref_deg"], 2), "onset", res_b.get("pos_growth_onset_s"), flush=True)

    # C: alternatives — tok48 counts, with brake, with tok33 mix
    alt_seqs = []
    for n in (15, 20, 25, 30):
        alt_seqs.append(("C_return48x" + str(n), BASE + RETIGHTEN + [SECURE_REVERSE_WRIST] * n + [0] * 5))
        alt_seqs.append(("C_return48x" + str(n) + "_brake", BASE + RETIGHTEN + [SECURE_REVERSE_WRIST] * n + [0] * 8))
        alt_seqs.append(("C_return48x" + str(n) + "_t33", BASE + RETIGHTEN + [SECURE_REVERSE_WRIST] * n + [33] * 5 + [0] * 5))
    # D: slide then return shorter slide
    short_slide = CPP + [17] * 8 + [12] * 5 + [0] * 5 + RETIGHTEN + [SECURE_REVERSE_WRIST] * 22 + [0] * 5
    alt_seqs.append(("C_shortslide_return", short_slide))
    # E: general — mutate return length
    for _ in range(8):
        n = int(rng.integers(12, 28))
        extra = [int(x) for x in rng.choice([0, 14, 33, SECURE_REVERSE_WRIST], size=5)]
        alt_seqs.append((f"E_general_n{n}", BASE + RETIGHTEN + [SECURE_REVERSE_WRIST] * n + extra + [0] * 3))

    best_full = res_b if res_b["SUCCESS_NEW"] else None
    full_obj_rows = list(paired_rows)
    for name, seq in alt_seqs:
        r = evaluate_tokens_vocab(env, table, book48, blob, seq)
        row = {"branch": name, **_pack(r)}
        full_obj_rows.append(row)
        if r["SUCCESS_NEW"]:
            print("  FULL", name, "pos", r["window"].get("mean_pos"), "th", r["window"].get("mean_theta"), flush=True)
            if best_full is None or (r["window"]["mean_pos"] < best_full["window"]["mean_pos"]):
                best_full = r
                best_full["branch"] = name

    write_csv(OUT / "RETURN_TO_NOMINAL_RESULTS.csv", paired_rows)
    write_csv(OUT / "FULL_OBJECTIVE_RESULTS.csv", full_obj_rows)

    # Save best expert for hard_910263 if any
    experts = []
    if best_full and best_full["SUCCESS_NEW"]:
        full = evaluate_tokens_vocab(env, table, book48, blob, best_full["tokens"], record_obs=True)
        _save_expert(env, "hard_910263", best_full.get("branch", "best"), full)
        experts.append(_expert_row("hard_910263", best_full.get("branch", "best"), full))
        replay = evaluate_tokens_vocab(env, table, book48, blob, best_full["tokens"])
        experts[-1]["replay_ok"] = int(replay["SUCCESS_NEW"] and replay["success_12s"])

    # --- Expand to other ICs ---
    print("EXTRA_ICS", flush=True)
    failures = []
    for spec in EXTRA_ICS[: args.max_extra_ics]:
        path = ROOT / spec["path"]
        if not path.exists():
            continue
        print(" IC", spec["id"], flush=True)
        shutil.copy2(path, OUT / "SNAPSHOTS" / f"{spec['id']}.pkl")
        bl = pickle.loads(path.read_bytes())
        # baseline prefix handoff
        base_p = list(spec["prefix"])
        r0 = evaluate_tokens_vocab(env, table, book48, bl, base_p)
        # with return48
        restore_capture(env, bl["capture"])
        t0 = float(env.sim.data.time)
        ok, _ = run_seq(env, table, book48, bl, base_p + RETIGHTEN)
        if not ok:
            failures.append({"ic_id": spec["id"], "note": "prefix_drop"})
            continue
        ok_r, ret_toks, _, why = return_with_token48(env, table, book48, bl, t0, max_steps=30)
        seq = base_p + RETIGHTEN + ret_toks + [0] * 3
        r1 = evaluate_tokens_vocab(env, table, book48, bl, seq)
        # also try fixed 20/25 return
        candidates = [
            ("prefix", base_p, r0),
            ("prefix_return", seq, r1),
        ]
        for n in (18, 22, 26):
            s = base_p + RETIGHTEN + [SECURE_REVERSE_WRIST] * n + [0] * 5
            rr = evaluate_tokens_vocab(env, table, book48, bl, s)
            candidates.append((f"return48x{n}", s, rr))
        best = max(candidates, key=lambda x: (
            int(x[2]["SUCCESS_NEW"]),
            -float(x[2]["window"]["mean_pos"] or 1e3) if x[2]["success_12s"] else -1e6,
        ))
        tag, sbest, rbest = best
        full_obj_rows.append({"branch": f"{spec['id']}__{tag}", **_pack(rbest)})
        if rbest["SUCCESS_NEW"]:
            full = evaluate_tokens_vocab(env, table, book48, bl, sbest, record_obs=True)
            _save_expert(env, spec["id"], tag, full)
            experts.append(_expert_row(spec["id"], tag, full))
            print("  FULL_EXPERT", spec["id"], tag, flush=True)
        else:
            failures.append({
                "ic_id": spec["id"], "best_tag": tag,
                "win_theta": rbest["window"].get("mean_theta"),
                "win_pos": rbest["window"].get("mean_pos"),
                "s12": rbest["success_12s"], "return_why": why,
            })

    write_csv(OUT / "FULL_OBJECTIVE_RESULTS.csv", full_obj_rows)
    write_csv(OUT / "EXPERT_TRAJECTORIES.csv", experts)
    write_json(OUT / "RAW" / "failures.json", failures)

    n_full = sum(1 for e in experts if e.get("SUCCESS_NEW") == 1)
    # hard_910263 specifically
    h263 = any(e["ic_id"] == "hard_910263" and e["SUCCESS_NEW"] == 1 for e in experts)
    summary = {
        "vocab_final": vocab_final,
        "token_48": table[48].tolist(),
        "n_full_experts": n_full,
        "hard_910263_full_expert": h263,
        "physical_hand_err_before": before["hand_to_ref_deg"],
        "physical_hand_err_after_tok48x25": after_b["hand_to_ref_deg"],
        "branch_B": paired_rows[1] if len(paired_rows) > 1 else None,
        "seed": args.seed,
    }
    write_json(OUT / "summary.json", summary)

    if not args.skip_videos and best_full and best_full.get("SUCCESS_NEW"):
        _video(env, table, book48, blob, best_full["tokens"], OUT / "VIDEOS" / "hard_910263_best.mp4")
    _video(env, table, book48, blob, BASE, OUT / "VIDEOS" / "hard_910263_A_baseline.mp4")
    if len(paired_rows) > 1:
        _video(env, table, book48, blob, seq_b, OUT / "VIDEOS" / "hard_910263_B_return.mp4")

    print("PILOT_DONE", OUT, "vocab", vocab_final, "full", n_full, "h263", h263, flush=True)


def _pack(res):
    return {
        "success_12s": int(res["success_12s"]),
        "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
        "label": res["label"],
        "reason": res["reason"],
        "win_mean_theta": res["window"].get("mean_theta"),
        "win_mean_pos": res["window"].get("mean_pos"),
        "hold_both_s": res["window"].get("longest_both_ok_s"),
        "e_pos_at_handoff": res.get("e_pos_at_handoff"),
        "e_theta_at_handoff": res.get("e_theta_at_handoff"),
        "pos_growth_onset_s": res.get("pos_growth_onset_s"),
        "recovery_s": res["recovery_elapsed"],
        "n_tokens": len(res["tokens"]),
        "tokens": json.dumps(res["tokens"]),
    }


def _expert_row(ic_id, source, res):
    return {
        "ic_id": ic_id,
        "source": source,
        "label": res["label"],
        "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
        "success_12s": int(res["success_12s"]),
        "win_mean_theta": res["window"].get("mean_theta"),
        "win_mean_pos": res["window"].get("mean_pos"),
        "hold_both_s": res["window"].get("longest_both_ok_s"),
        "recovery_s": res["recovery_elapsed"],
        "n_tokens": len(res["tokens"]),
        "tokens": json.dumps(res["tokens"]),
        "uses_token_48": int(SECURE_REVERSE_WRIST in res["tokens"]),
    }


def _save_expert(env, ic_id, source, res):
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


def _video(env, table, book48, blob, tokens, dest):
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
        for i in range(50):
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


if __name__ == "__main__":
    main()
