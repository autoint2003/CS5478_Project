"""Compare discrete K49 tok48 angular speeds (100% / 50% / 25%).

Speed is encoded in the token mapping (named config_id), executed via step_legal.
No continuous ω scaling inside trajectories. No SFT/GRPO. No pulse interleaving.
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

from controllers.recovery.action_map import AIRBORNE_V3, map_action
from controllers.recovery.geometry_switch import rot_angle
from envs.ballistic_impact import z_tgt_of
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor, recovery_failed
from envs.outcomes.nominal_pose import hand_R
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.grasp_quality import body_twist
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary, window_stats
from evaluation.grasp_geometry import measure
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import (
    DEFAULT_K49_CONFIG_ID,
    HANDOFF,
    K49_TOK48_SPEED_CONFIGS,
    SECURE_REVERSE_WRIST,
    assert_k48_prefix_identical,
    build_k49_table,
    decode_action as decode_k49,
    describe_k49_config,
    save_k49_speed_configs,
)
from training.k47_codebook import decode_action as decode_k48
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env
from training.recovery_execution import _escaped

OUT = ROOT / "results" / "diagnostics" / "k49_slow_reverse_token48"
SNAP_IC = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910263.pkl"
PRERETURN = ROOT / "results/diagnostics/k49_return_axial_slip_mechanism/SNAPSHOTS/hard_910263_prereturn.pkl"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"

CPP = [4] * 20 + [12] * 5 + [0] * 3
SLIDE = [17] * 16 + [12] * 5 + [0] * 5
RETIGHTEN = [0, 0, 0]
PREFIX = CPP + SLIDE + RETIGHTEN
RECOVERY_LIMIT_S = 4.0
SEED = 20261018

# Configs under test (ordered)
SPEED_CONFIGS = [
    "K49_tok48_wy1p0",
    "K49_tok48_wy0p5",
    "K49_tok48_wy0p25",
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


def axial(sim):
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
        "e_ax_mm": e_ax * 1e3,
        "e_lat_mm": e_lat * 1e3,
        "e_pos_mm": m["e_pos_mm"],
        "e_theta_deg": m["e_theta_deg"],
        "bilateral": m["bilateral"],
        "nL": m["nL"],
        "nR": m["nR"],
    }


def stage(sim, blob, label=""):
    ax = axial(sim)
    cs = contact_summary(sim)
    Rh = hand_R(sim)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    return {
        "label": label,
        "t": float(sim.data.time),
        **ax,
        "hand_to_ref_deg": float(np.degrees(rot_angle(Rref, Rh))),
        "rdes_to_ref_deg": float(np.degrees(rot_angle(Rref, Rd))),
        "hand_to_rdes_deg": float(np.degrees(rot_angle(Rd, Rh))),
        "w_hand_norm": float(np.linalg.norm(wh)),
        "v_hand_norm": float(np.linalg.norm(vh)),
        "Fn_proxy_nL": cs["nL"],
        "Fn_proxy_nR": cs["nR"],
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


def evaluate_full(env, table, book48, blob, tokens):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    m0 = measure(env.sim, U_TGT, P_TGT)
    lost_low = 0
    for tok in tokens:
        action = decode(table, book48, tok)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
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
                "pre_handoff": None,
            }
    pre = stage(env.sim, blob, "pre_handoff")
    m_hand = measure(env.sim, U_TGT, P_TGT)
    rec = float(env.sim.data.time) - t0
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
        "pre_handoff": pre,
        "recovery_elapsed": rec,
        "window": win,
        "SUCCESS_NEW": success_new,
        "label": "VERIFIED_FULL_EXPERT" if success_new else ("FAILED_OR_DROPPED" if not s12 else "PARTIAL"),
        "e_pos_at_handoff": m_hand["e_pos_mm"],
        "e_theta_at_handoff": m_hand["e_theta_deg"],
    }


def return_until(env, table, book48, blob, t0_ic, target_hand_deg, tol_hand=1.0,
                 tol_rdes=8.0, tol_track=5.0, max_steps=200):
    """Apply tok48 until hand orientation <= target_hand_deg (or deadline)."""
    used = []
    stages = []
    lost_low = 0
    w_samples = []
    a_samples = []
    prev_w = None
    dt_tok = 0.02
    for i in range(max_steps):
        st = stage(env.sim, blob, f"ret_{i}")
        stages.append(st)
        w_samples.append(st["w_hand_norm"])
        if prev_w is not None:
            a_samples.append(abs(st["w_hand_norm"] - prev_w) / dt_tok)
        prev_w = st["w_hand_norm"]
        # converge: hand near target (matched to baseline end) AND tracking ok
        if (st["hand_to_ref_deg"] <= target_hand_deg + tol_hand
                and st["hand_to_rdes_deg"] <= tol_track and st["bilateral"]):
            return True, used, stages, "CONVERGED", w_samples, a_samples
        if float(env.sim.data.time) - t0_ic > RECOVERY_LIMIT_S - 0.05:
            return False, used, stages, "DEADLINE", w_samples, a_samples
        ok, lost_low = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost_low)
        used.append(SECURE_REVERSE_WRIST)
        if not ok:
            return False, used, stages, "DROP", w_samples, a_samples
    st = stage(env.sim, blob, "ret_end")
    stages.append(st)
    ok = st["hand_to_ref_deg"] <= target_hand_deg + tol_hand
    return ok, used, stages, "CONVERGED" if ok else "MAX_STEPS", w_samples, a_samples


def ensure_prereturn(env, table, book48, blob_ic):
    if PRERETURN.exists():
        pre = pickle.loads(PRERETURN.read_bytes())
        return pre
    restore_capture(env, blob_ic["capture"])
    ok, _ = run_seq(env, table, book48, blob_ic, PREFIX)
    assert ok
    return {"capture": take_capture(env), "ref": blob_ic["ref"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("RAW", "ACTION_MAPPING_CONFIGS", "SNAPSHOTS"):
        (OUT / sub).mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    catalog = save_k49_speed_configs(OUT / "ACTION_MAPPING_CONFIGS", book48)
    write_json(OUT / "ACTION_MAPPING_CONFIGS" / "catalog_summary.json", catalog)

    # Verify historical default unchanged vs prior mirror
    t_hist = build_k49_table(book48, config_id=DEFAULT_K49_CONFIG_ID)
    t_legacy = build_k49_table(book48)  # default args
    assert np.allclose(t_hist[48], t_legacy[48])
    assert_k48_prefix_identical(t_hist, book48)
    for cid in SPEED_CONFIGS:
        assert_k48_prefix_identical(build_k49_table(book48, config_id=cid), book48)
        assert abs(build_k49_table(book48, config_id=cid)[47].sum()) < 1e-9  # HANDOFF row unused

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    blob_ic = pickle.loads(SNAP_IC.read_bytes())
    shutil.copy2(SNAP_IC, OUT / "SNAPSHOTS" / "hard_910263_ic.pkl")

    # Use historical table for prefix to reach identical pre-return
    table_hist = build_k49_table(book48, config_id=DEFAULT_K49_CONFIG_ID)
    pre = ensure_prereturn(env, table_hist, book48, blob_ic)
    with (OUT / "SNAPSHOTS" / "hard_910263_prereturn.pkl").open("wb") as f:
        pickle.dump(pre, f)

    restore_capture(env, pre["capture"])
    before = stage(env.sim, pre, "prereturn")
    write_json(OUT / "RAW" / "prereturn_state.json", before)
    print("PRERETURN", before["e_ax_mm"], before["hand_to_ref_deg"], flush=True)

    # Commanded |w| from map_action for each config (diagnostic print)
    for cid in SPEED_CONFIGS:
        table = build_k49_table(book48, config_id=cid)
        a = decode(table, book48, SECURE_REVERSE_WRIST)
        rh = np.eye(3)
        rd = np.eye(3)
        cmd = map_action(a, rd, rh, AIRBORNE_V3)
        print(cid, "tok48", a, "w_cmd", cmd["w_world"], "tau", cmd["tau"], flush=True)

    # --- Pass 1: baseline 100% to get matched target orientation ---
    print("BASELINE_RETURN", flush=True)
    restore_capture(env, pre["capture"])
    t0 = float(env.sim.data.time)
    # Use generous target then record achieved hand
    ok, used, stages, why, w_s, a_s = return_until(
        env, table_hist, book48, pre, t0, target_hand_deg=8.0, tol_hand=0.0, max_steps=80,
    )
    target_hand = stages[-1]["hand_to_ref_deg"] if stages else 8.0
    # Prefer the converged value; if deadline, use last
    if why == "CONVERGED":
        target_hand = stages[-1]["hand_to_ref_deg"]
    write_json(OUT / "RAW" / "baseline_return_end.json", {
        "ok": ok, "why": why, "n_tok48": len(used), "target_hand_deg_for_match": target_hand,
        "end": stages[-1] if stages else None,
    })
    print(f"  baseline why={why} n={len(used)} hand={target_hand:.3f}", flush=True)

    # Matched-orientation target: use baseline final hand (or 8° if tighter)
    match_hand = float(target_hand)

    speed_rows = []
    full_rows = []
    results_by_cfg = {}

    for cid in SPEED_CONFIGS:
        print(f"SPEED_TEST {cid}", flush=True)
        table = build_k49_table(book48, config_id=cid)
        meta = describe_k49_config(cid, book48)
        restore_capture(env, pre["capture"])
        t0 = float(env.sim.data.time)
        before_i = stage(env.sim, pre, "before")
        # For 100%: allow converge to same as baseline; for others match baseline hand
        tol = 1.0 if cid != DEFAULT_K49_CONFIG_ID else 0.5
        ok_r, ret_toks, stgs, why_r, w_samp, a_samp = return_until(
            env, table, book48, pre, t0,
            target_hand_deg=match_hand, tol_hand=tol, max_steps=220,
        )
        after_i = stgs[-1] if stgs else stage(env.sim, pre, "after")
        # per-token stage csv
        write_csv(OUT / "RAW" / f"{cid}_return_stages.csv", stgs)

        d_ax = after_i["e_ax_mm"] - before_i["e_ax_mm"]
        d_lat = after_i["e_lat_mm"] - before_i["e_lat_mm"]
        d_pos = after_i["e_pos_mm"] - before_i["e_pos_mm"]
        n_ret = len(ret_toks)
        dt_ret = after_i["t"] - before_i["t"]
        row = {
            "config_id": cid,
            "wy_scale": meta["wy_scale"],
            "token_48_wy": meta["token_48_wy"],
            "token_48_action": json.dumps(meta["token_48_action"]),
            "ok_return": int(ok_r),
            "return_why": why_r,
            "n_tok48": n_ret,
            "return_time_s": dt_ret,
            "w_hand_mean": float(np.mean(w_samp)) if w_samp else None,
            "w_hand_max": float(np.max(w_samp)) if w_samp else None,
            "alpha_hand_mean": float(np.mean(a_samp)) if a_samp else None,
            "hand0_deg": before_i["hand_to_ref_deg"],
            "hand1_deg": after_i["hand_to_ref_deg"],
            "rdes1_deg": after_i["rdes_to_ref_deg"],
            "track1_deg": after_i["hand_to_rdes_deg"],
            "e_ax0_mm": before_i["e_ax_mm"],
            "e_ax1_mm": after_i["e_ax_mm"],
            "d_e_ax_mm": d_ax,
            "e_lat0_mm": before_i["e_lat_mm"],
            "e_lat1_mm": after_i["e_lat_mm"],
            "d_e_lat_mm": d_lat,
            "e_pos0_mm": before_i["e_pos_mm"],
            "e_pos1_mm": after_i["e_pos_mm"],
            "d_e_pos_mm": d_pos,
            "e_theta0": before_i["e_theta_deg"],
            "e_theta1": after_i["e_theta_deg"],
            "bilateral_end": int(after_i["bilateral"]),
            "match_hand_target_deg": match_hand,
        }
        # recovery budget from IC time (prefix already consumed in wall clock of sim)
        restore_capture(env, blob_ic["capture"])
        t_ic = float(env.sim.data.time)
        row["t_ic"] = t_ic
        row["t_prereturn"] = before_i["t"]
        row["t_after_return"] = after_i["t"]
        row["recovery_used_s"] = after_i["t"] - t_ic
        row["recovery_remaining_s"] = RECOVERY_LIMIT_S - (after_i["t"] - t_ic)
        speed_rows.append(row)
        print(
            f"  {cid}: d_ax={d_ax:.3f} n={n_ret} dt={dt_ret:.3f}s hand={after_i['hand_to_ref_deg']:.2f} "
            f"why={why_r} remain={row['recovery_remaining_s']:.3f}s",
            flush=True,
        )

        # Full trajectory: CPP+SLIDE+RETIGHTEN+return+stabilize  (tokens from IC)
        # Prefix uses historical motion tokens 0-46 (identical across configs)
        tokens_full = list(PREFIX) + list(ret_toks) + list(RETIGHTEN)
        # Need to re-run return from IC with this table to get exact token count consistently
        restore_capture(env, blob_ic["capture"])
        t0_ic = float(env.sim.data.time)
        ok_p, lost = run_seq(env, table, book48, blob_ic, PREFIX)
        if not ok_p:
            full = {"SUCCESS_NEW": False, "success_12s": False, "reason": "DROP_PREFIX",
                    "window": window_stats([], float(NOMINAL_S) - 2.0), "label": "FAILED_OR_DROPPED",
                    "recovery_elapsed": 0, "e_pos_at_handoff": None, "e_theta_at_handoff": None,
                    "pre_handoff": None, "tokens": PREFIX}
        else:
            ok_r2, ret2, _, why2, _, _ = return_until(
                env, table, book48, blob_ic, t0_ic, match_hand, tol_hand=tol, max_steps=220,
            )
            tokens_full = list(PREFIX) + list(ret2) + list(RETIGHTEN)
            full = evaluate_full(env, table, book48, blob_ic, tokens_full)
            full["return_why"] = why2
            full["ok_return"] = ok_r2

        pre_h = full.get("pre_handoff") or {}
        frow = {
            "config_id": cid,
            "wy_scale": meta["wy_scale"],
            "SUCCESS_NEW": int(full["SUCCESS_NEW"]),
            "success_12s": int(full["success_12s"]),
            "label": full["label"],
            "reason": full["reason"],
            "win_mean_theta": full["window"].get("mean_theta"),
            "win_mean_pos": full["window"].get("mean_pos"),
            "hold_both_s": full["window"].get("longest_both_ok_s"),
            "e_pos_at_handoff": full.get("e_pos_at_handoff"),
            "e_theta_at_handoff": full.get("e_theta_at_handoff"),
            "e_ax_at_handoff": pre_h.get("e_ax_mm"),
            "e_lat_at_handoff": pre_h.get("e_lat_mm"),
            "hand_to_ref_at_handoff": pre_h.get("hand_to_ref_deg"),
            "recovery_s": full["recovery_elapsed"],
            "n_tokens": len(full["tokens"]),
            "n_tok48": sum(1 for t in full["tokens"] if int(t) == SECURE_REVERSE_WRIST),
            "return_why": full.get("return_why"),
            "tokens": json.dumps(full["tokens"]),
        }
        full_rows.append(frow)
        results_by_cfg[cid] = {"speed": row, "full": frow, "meta": meta}
        print(
            f"  FULL {cid}: NEW={frow['SUCCESS_NEW']} S12={frow['success_12s']} "
            f"win_pos={frow['win_mean_pos']} handoff_pos={frow['e_pos_at_handoff']} "
            f"n48={frow['n_tok48']} rec_s={frow['recovery_s']}",
            flush=True,
        )

    write_csv(OUT / "TOKEN48_SPEED_COMPARISON.csv", speed_rows)
    write_csv(OUT / "FULL_RECOVERY_RESULTS.csv", full_rows)
    write_json(OUT / "RAW" / "results_by_cfg.json", results_by_cfg)

    selected = _select(speed_rows, full_rows)
    write_json(OUT / "SELECTED_CONFIG.json", selected)
    # Copy selected table to a clear path
    sel_id = selected["config_id"]
    shutil.copy2(
        OUT / "ACTION_MAPPING_CONFIGS" / f"{sel_id}_table.npy",
        OUT / "ACTION_MAPPING_CONFIGS" / "SELECTED_table.npy",
    )
    shutil.copy2(
        OUT / "ACTION_MAPPING_CONFIGS" / f"{sel_id}_meta.json",
        OUT / "ACTION_MAPPING_CONFIGS" / "SELECTED_meta.json",
    )
    _write_report(speed_rows, full_rows, selected, match_hand, before)
    print("SELECTED", sel_id, selected.get("reason"), flush=True)
    print("DIAG_DONE", OUT, flush=True)


def _select(speed_rows, full_rows):
    """Prefer SUCCESS_NEW; among those prefer higher speed (shorter); else best margin."""
    by = {r["config_id"]: r for r in full_rows}
    sp = {r["config_id"]: r for r in speed_rows}
    # Any SUCCESS_NEW?
    winners = [cid for cid, r in by.items() if r.get("SUCCESS_NEW")]
    if winners:
        # prefer higher wy_scale (faster / shorter) among winners
        winners.sort(key=lambda c: -float(sp[c]["wy_scale"]))
        cid = winners[0]
        return {
            "config_id": cid,
            "wy_scale": sp[cid]["wy_scale"],
            "reason": f"SUCCESS_NEW achieved; prefer faster among successes ({cid})",
            "SUCCESS_NEW": True,
        }
    # None succeeded — report failure; still pick least-bad by handoff pos after return
    # Prefer 50% if axial improvement similar to 25% with much less time
    scored = []
    for cid, r in by.items():
        s = sp[cid]
        # lower |d_e_ax| better, need ok_return, remaining time > 0.2
        score = (
            0 if s.get("ok_return") else 1,
            abs(float(s.get("d_e_ax_mm") or 99)),
            float(s.get("return_time_s") or 99),
        )
        scored.append((score, cid))
    scored.sort()
    cid = scored[0][1]
    # Decision rule from task: prefer 50% if comparable quality & shorter than 25%
    s50 = sp.get("K49_tok48_wy0p5")
    s25 = sp.get("K49_tok48_wy0p25")
    if s50 and s25 and s50.get("ok_return") and s25.get("ok_return"):
        ax50 = abs(float(s50["d_e_ax_mm"]))
        ax25 = abs(float(s25["d_e_ax_mm"]))
        # comparable if within 0.5 mm; prefer 50% for time
        if ax50 <= ax25 + 0.5 and float(s50["return_time_s"]) < 0.85 * float(s25["return_time_s"]):
            cid = "K49_tok48_wy0p5"
            reason = (
                f"No SUCCESS_NEW; 50% axial |Δ|={ax50:.2f} comparable to 25% {ax25:.2f} "
                f"with shorter return ({s50['return_time_s']:.2f}s vs {s25['return_time_s']:.2f}s)"
            )
            return {"config_id": cid, "wy_scale": 0.5, "reason": reason, "SUCCESS_NEW": False}
    return {
        "config_id": cid,
        "wy_scale": sp[cid]["wy_scale"],
        "reason": (
            f"No SUCCESS_NEW for any speed; selected least axial drift among feasible returns ({cid}). "
            "Lower speed alone does not solve hard_910263."
        ),
        "SUCCESS_NEW": False,
    }


def _write_report(speed_rows, full_rows, selected, match_hand, before):
    lines = [
        "# K49 Slow Reverse Token 48 — Speed Comparison",
        "",
        "**No pulse interleaving. Speed encoded in named discrete tok48 mappings. `step_legal` only.**",
        "",
        "```text",
        "python -u scripts/diagnose_k49_slow_reverse_token48.py --device cuda",
        "```",
        "",
        "## Configurations (tokens 0–47 + HANDOFF=47 unchanged)",
        "",
        "| config_id | wy_scale | tok48 wy | historical |",
        "|---|---:|---:|---|",
    ]
    for cid, cfg in K49_TOK48_SPEED_CONFIGS.items():
        meta = describe_k49_config(cid)
        lines.append(
            f"| `{cid}` | {cfg['wy_scale']} | {meta['token_48_wy']:.4f} | {cfg['historical_default']} |"
        )
    lines += [
        "",
        f"Matched hand target (from 100% converge): **{match_hand:.2f}°**.",
        f"Pre-return: e_ax={before['e_ax_mm']:.2f} mm, hand={before['hand_to_ref_deg']:.2f}°.",
        "",
        "## Return comparison (same pre-Return snapshot)",
        "",
        "| config | n×48 | time (s) | mean \|w\| | Δe_ax (mm) | e_pos end | hand end | remain (s) | why |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in speed_rows:
        lines.append(
            f"| `{r['config_id']}` | {r['n_tok48']} | {r['return_time_s']:.3f} | "
            f"{r['w_hand_mean']:.3f} | {r['d_e_ax_mm']:.3f} | {r['e_pos1_mm']:.2f} | "
            f"{r['hand1_deg']:.2f} | {r['recovery_remaining_s']:.3f} | {r['return_why']} |"
        )
    lines += [
        "",
        "## Full recovery (CPP→Slide→Return→HANDOFF→nominal)",
        "",
        "| config | S12 | NEW | win θ / pos | handoff pos | n48 | rec (s) |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in full_rows:
        lines.append(
            f"| `{r['config_id']}` | {r['success_12s']} | **{r['SUCCESS_NEW']}** | "
            f"{r['win_mean_theta']} / {r['win_mean_pos']} | {r['e_pos_at_handoff']} | "
            f"{r['n_tok48']} | {r['recovery_s']:.3f} |"
        )
    lines += [
        "",
        "## Selection",
        "",
        f"**Selected:** `{selected['config_id']}` (wy_scale={selected['wy_scale']})",
        "",
        f"Reason: {selected['reason']}",
        "",
        f"SUCCESS_NEW under selected mapping: **{selected['SUCCESS_NEW']}**",
        "",
        "Artifacts: `ACTION_MAPPING_CONFIGS/{config_id}_table.npy`, `SELECTED_table.npy`.",
        "",
        "## Stop",
        "",
        "Speed comparison complete. No SFT/GRPO. Historical `K49_tok48_wy1p0` preserved.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "summary.json", {
        "selected": selected,
        "match_hand_deg": match_hand,
        "speed_rows": speed_rows,
        "full_rows": full_rows,
    })


if __name__ == "__main__":
    main()
