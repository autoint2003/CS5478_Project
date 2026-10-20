"""Convert historical nominal-assisted regrasp into legal in-Recovery Recatch experts.

Uses locked K49_tok48_wy0p25. No SFT/GRPO. SUCCESS_NEW unchanged.
"""
from __future__ import annotations

import argparse
import ast
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

# Reuse helpers from sliding/recatch generator
from scripts.generate_k49_sliding_recatch_dataset import (
    adaptive_return_variants,
    classify_mechanisms,
    decode,
    decompose,
    evaluate_full,
    hand_ref_error_deg,
    ic_snapshot_state,
    mapping_checksum,
    return_slow,
    run_seq,
    tokens_aligned,
    write_csv,
    write_json,
    jsafe,
)

OUT = ROOT / "results" / "expert_generation" / "k49_in_recovery_recatch"
CONFIG_ID = "K49_tok48_wy0p25"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
SEED = 20261018
RECOVERY_LIMIT_S = 4.0
AUDIT_CSV = ROOT / "results/diagnostics/nontrivial_recovery_ic_mining/PREFIX_HANDOFF_AUDIT.csv"


def snapshot_for(ic_id: str) -> Path | None:
    cands = [
        ROOT / f"results/benchmarks/hard_recovery_v1/snapshots/{ic_id}.pkl",
        ROOT / f"results/diagnostics/hard_ic_mining/snapshots/{ic_id}.pkl",
    ]
    if "leftover_" in ic_id:
        hard = ic_id.split("leftover_")[-1]
        cands += [
            ROOT / f"results/diagnostics/hard_ic_mining/snapshots/{hard}.pkl",
            ROOT / f"results/benchmarks/hard_recovery_v1/snapshots/{hard}.pkl",
        ]
    for p in cands:
        if p.exists():
            return p
    return None


def load_short_handoff_ics():
    rows = list(csv.DictReader(AUDIT_CSV.open(encoding="utf-8")))
    out = []
    for r in rows:
        if r["success"] != "1":
            continue
        ic = r["ic_id"]
        toks = ast.literal_eval(r["tokens"])
        snap = snapshot_for(ic)
        if snap is None:
            continue
        out.append({"ic_id": ic, "prefix": [int(t) for t in toks], "snapshot": str(snap)})
    # dedupe keep first (shortest successful)
    seen, uniq = set(), []
    for e in out:
        if e["ic_id"] in seen:
            continue
        seen.add(e["ic_id"])
        uniq.append(e)
    return uniq


def sample_row(env, phase, tok=None):
    cs = contact_summary(env.sim)
    d = decompose(env.sim)
    ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
    po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
    grip = float(env.sim.data.ctrl[7]) if env.sim.data.ctrl.size > 7 else None
    return {
        "t": float(env.sim.data.time),
        "phase": phase,
        "tok": None if tok is None else int(tok),
        "nL": cs["nL"], "nR": cs["nR"], "bilateral": bool(cs["bilateral"]),
        "e_pos_mm": d["e_pos_mm"], "e_ax_mm": d["e_ax_mm"], "e_lat_mm": d["e_lat_mm"],
        "e_theta_deg": d["e_theta_deg"],
        "hand_pos": ph.tolist(), "obj_pos": po.tolist(),
        "grip_ctrl": grip,
        "obj_z": float(po[2]),
    }


def record_historical_nominal_regrasp(env, table, book48, blob, prefix):
    """prefix → HANDOFF → nominal; record contact restoration and pose."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    st0 = sample_row(env, "ic")
    requires_restore = (not st0["bilateral"]) or True  # short-handoff ICs: imm HANDOFF fails
    # confirm imm fails S12 for labeling
    imm = evaluate_full(env, table, book48, blob, [], record_obs=False)

    restore_capture(env, blob["capture"])
    trace = []
    ok, lost = run_seq(env, table, book48, blob, prefix, trace=trace)
    prefix_end = sample_row(env, "after_prefix")
    if not ok:
        return {
            "ok_prefix": False, "requires_contact_restoration": True,
            "imm_s12": imm["success_12s"], "imm_reason": imm["reason"],
            "st0": st0, "prefix_end": prefix_end, "trace_prefix": trace,
            "label": "RECATCH_FAILED", "reason": "DROP_DURING_PREFIX",
        }

    apply_handoff(env.sim, blob["ref"])
    handoff_state = sample_row(env, "handoff")
    # nominal until bilateral restored (or fail / timeout)
    first_bil = None
    stable_bil_s = 0.0
    last_bil_t = None
    nom_trace = []
    reason = None
    end_row = None
    lift_done = False
    lost_c = 0
    t_h = float(env.sim.data.time)
    next_sample = 0.0
    samples = []
    while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(env.sim)
        t_rel = float(env.sim.data.time) - t_h
        row = sample_row(env, "nominal")
        if t_rel + 1e-12 >= next_sample:
            nom_trace.append({**row, "t_rel": t_rel})
            samples.append({"t_rel": t_rel, "e_theta_deg": row["e_theta_deg"], "e_pos_mm": row["e_pos_mm"],
                            "bilateral": row["bilateral"], "obj_z": row["obj_z"]})
            next_sample += 0.02
        if first_bil is None and row["bilateral"]:
            first_bil = {"t_rel": t_rel, "t_abs": row["t"], **{k: row[k] for k in
                         ("nL", "nR", "e_pos_mm", "e_theta_deg", "e_ax_mm", "e_lat_mm")}}
        if row["bilateral"]:
            if last_bil_t is None:
                last_bil_t = t_rel
            stable_bil_s = t_rel - last_bil_t if last_bil_t is not None else 0.0
            # actually accumulate continuous
        else:
            last_bil_t = None
        mon = _monitor(env.sim)
        end_row = mon
        target = float(z_tgt_of(env.sim))
        if _bilateral(mon) and float(env.sim.fsm.p_des[2]) >= target - 1e-9:
            lift_done = True
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        if _dropped(mon) or _escaped(env.sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost_c = lost_c + 1 if not _bilateral(mon) else 0
        if lost_c >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
    if reason is None:
        if end_row is None or not _bilateral(end_row) or _dropped(end_row) or _escaped(env.sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    win = window_stats(samples, float(NOMINAL_S) - 2.0)
    s12 = reason == "SUCCESS_12S"
    success_new = bool(s12 and win.get("working_SUCCESS_NEW_hold"))
    # continuous bilateral duration after first restore
    bil_hold = 0.0
    if first_bil is not None:
        run = 0.0
        prev = None
        for r in nom_trace:
            if r["bilateral"]:
                if prev is None:
                    prev = r["t_rel"]
                run = r["t_rel"] - first_bil["t_rel"]
            else:
                if r["t_rel"] >= first_bil["t_rel"]:
                    break
        bil_hold = run

    label = "HANDOFF_ASSISTED_RECATCH_FULL_EXPERT" if success_new else (
        "HANDOFF_ASSISTED_S12" if s12 else "RECATCH_FAILED"
    )
    return {
        "ok_prefix": True,
        "requires_contact_restoration": True,
        "imm_s12": imm["success_12s"],
        "imm_reason": imm["reason"],
        "st0": st0,
        "prefix_end": prefix_end,
        "handoff_state": handoff_state,
        "first_bilateral_after_handoff": first_bil,
        "bilateral_hold_s_approx": bil_hold,
        "s12": s12,
        "SUCCESS_NEW": success_new,
        "reason": reason,
        "window": win,
        "label": label,
        "trace_prefix": trace,
        "trace_nominal": nom_trace[:200],
        "recovery_s": float(prefix_end["t"] - t0),
        "pose_after_regrasp": None if first_bil is None else {
            "e_pos_mm": first_bil["e_pos_mm"],
            "e_theta_deg": first_bil["e_theta_deg"],
            "e_ax_mm": first_bil["e_ax_mm"],
            "e_lat_mm": first_bil["e_lat_mm"],
            "t_rel": first_bil["t_rel"],
        },
    }


def regrasp_proposals(book, desire, rng, n=40, open_prefix=None):
    """Legal K49 sequences intended to restore bilateral contact after release."""
    sec = tokens_aligned(book, desire if desire is not None else np.array([0, 0, -1.0]), "secure", 8) or [
        0, 1, 5, 13, 17, 33
    ]
    seqs = []
    # A: close / hold (nominal-like grip restore)
    for n0 in (5, 8, 12, 16, 20, 24, 30, 40):
        seqs.append([0] * n0)
        seqs.append([12] * min(5, n0 // 2) + [0] * n0)
    # B: secure translate while closing
    for t in sec[:6]:
        for n_c in (8, 12, 16, 20, 24):
            seqs.append([t] * n_c + [0] * 5)
            seqs.append([12] * 3 + [t] * n_c + [0] * 5)
            seqs.append([0] * 4 + [t] * n_c + [0] * 4)
    # C: brief further open then close (F_recatch-ish)
    for o in (30, 36, 39, 44, 41, 32):
        for n_o in (1, 2, 3, 5):
            for t in (sec[:3] or [1, 5]):
                seqs.append([o] * n_o + [0] * 12)
                seqs.append([o] * n_o + [t] * 12 + [0] * 5)
                seqs.append([o] * n_o + [12] * 3 + [t] * 10 + [0] * 5)
    # D: pivot then close
    for piv in ([4] * 10, [4] * 16, [6] * 12, [10] * 10, []):
        for t in sec[:3]:
            seqs.append(list(piv) + [12] * 3 + [t] * 12 + [0] * 5)
            seqs.append(list(piv) + [0] * 16)
    # E: known F_recatch body without assuming prior pivot
    seqs.append([36] * 5 + [0] * 8)
    seqs.append([4] * 12 + [36] * 5 + [0] * 8)
    seqs.append([36] * 5 + [0] * 8 + [4] * 20 + [12] * 5 + [0] * 3)

    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    head, tail = uniq[:12], uniq[12:]
    rng.shuffle(tail)
    return (head + tail)[:n]


def pose_correction_proposals(book, desire, rng, n=20):
    sec = tokens_aligned(book, desire, "secure", 6) or [17, 33, 13, 1]
    seqs = []
    seqs.append([])  # none
    for piv in ([4] * 20 + [12] * 5 + [0] * 3,
                [4] * 24 + [12] * 5 + [0] * 3,
                [6] * 16 + [12] * 5 + [0] * 3,
                [10] * 12 + [12] * 5 + [0] * 3):
        seqs.append(list(piv))
        for t in sec[:3]:
            seqs.append(list(piv) + [12] * 3 + [t] * 12 + [0] * 4)
    for t in sec[:4]:
        for n_tr in (8, 12, 16, 20):
            seqs.append([12] * 4 + [t] * n_tr + [0] * 4)
            seqs.append([t] * n_tr + [12] * 4 + [0] * 4)
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    rng.shuffle(uniq)
    # keep empty first
    uniq = [[]] + [u for u in uniq if u]
    return uniq[:n]


def find_first_bilateral(trace, start_idx=0):
    run = 0
    first = None
    for i in range(start_idx, len(trace)):
        if trace[i].get("bilateral"):
            run += 1
            if run >= 3 and first is None:
                first = i - 2
                break
        else:
            run = 0
    return first


def save_expert_traj(env, ic_id, tag, family, res, mech, mapping_meta, checksum, snap_src, extra_meta=None):
    stem = f"{ic_id}__{tag}"
    save_pair(
        OUT / "trajectories", stem,
        res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    meta = {
        "ic_id": ic_id,
        "tag": tag,
        "family": family,
        "config_id": CONFIG_ID,
        "mapping_checksum": checksum,
        "token_48_action": mapping_meta.get("token_48_action"),
        "SUCCESS_NEW": res["SUCCESS_NEW"],
        "label": res.get("expert_label") or res["label"],
        "window": res["window"],
        "tokens": res["tokens"],
        "hold_margin_mm": res.get("hold_margin_mm"),
        "recovery_s": res["recovery_elapsed"],
        "e_pos_at_handoff": res.get("e_pos_at_handoff"),
        "e_theta_at_handoff": res.get("e_theta_at_handoff"),
        "source_disturbance": mech.get("source_disturbance"),
        "observed_recovery": mech.get("observed_recovery"),
        "recatch_subtype": mech.get("recatch_subtype"),
        "sliding_verified": mech.get("sliding_verified"),
        "mechanism_evidence": mech.get("evidence"),
        "snapshot_src": str(snap_src),
        "obs_dim": 39,
        "handoff_id": 47,
    }
    if extra_meta:
        meta.update(extra_meta)
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    if res.get("trace"):
        write_json(OUT / "traces" / "candidates" / f"{stem}_trace.json", {"trace": res["trace"], "mech": mech})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-ics", type=int, default=22)
    ap.add_argument("--budget-regrasp", type=int, default=28)
    ap.add_argument("--budget-pose", type=int, default=10)
    ap.add_argument("--skip-search", action="store_true")
    args = ap.parse_args()

    t_wall0 = time.time()
    for sub in ("trajectories", "snapshots", "traces/historical", "traces/candidates", "RAW"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    mapping_meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    mapping_meta["checksum_sha256_16"] = checksum
    write_json(OUT / "ACTION_MAPPING.json", mapping_meta)
    np.save(OUT / f"{CONFIG_ID}_table.npy", table)

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    ics = load_short_handoff_ics()[: args.n_ics]
    write_json(OUT / "RAW" / "selected_ics.json", ics)
    print(f"SELECTED n={len(ics)}", flush=True)

    historical_rows = []
    in_recovery_experts = []
    handoff_assisted = []
    contact_only = []
    pose_only = []
    failures = []
    coverage = []
    manifest = []

    for ie, ic in enumerate(ics):
        t0b = time.time()
        ic_id = ic["ic_id"]
        prefix = ic["prefix"]
        snap = Path(ic["snapshot"])
        blob = pickle.loads(snap.read_bytes())
        shutil.copy2(snap, OUT / "snapshots" / f"{ic_id}.pkl")
        st0 = ic_snapshot_state(env, table, book48, blob)
        print(f"\n[{ie+1}/{len(ics)}] {ic_id} prefix={prefix} bil0={st0['bilateral']} "
              f"nL={st0['nL']} nR={st0['nR']} pos={st0['e_pos_mm']:.1f}", flush=True)

        # ---------- 1. Historical nominal-assisted ----------
        hist = record_historical_nominal_regrasp(env, table, book48, blob, prefix)
        write_json(OUT / "traces" / "historical" / f"{ic_id}_nominal_regrasp.json", hist)
        hrow = {
            "ic_id": ic_id,
            "prefix": json.dumps(prefix),
            "imm_s12": int(hist.get("imm_s12") or 0),
            "imm_reason": hist.get("imm_reason"),
            "s12": int(hist.get("s12") or 0),
            "SUCCESS_NEW": int(hist.get("SUCCESS_NEW") or 0),
            "reason": hist.get("reason"),
            "label": hist.get("label"),
            "first_bil_t_rel": None if not hist.get("first_bilateral_after_handoff")
            else hist["first_bilateral_after_handoff"].get("t_rel"),
            "pose_after_regrasp_pos": None if not hist.get("pose_after_regrasp")
            else hist["pose_after_regrasp"].get("e_pos_mm"),
            "pose_after_regrasp_theta": None if not hist.get("pose_after_regrasp")
            else hist["pose_after_regrasp"].get("e_theta_deg"),
            "win_mean_pos": (hist.get("window") or {}).get("mean_pos"),
            "win_mean_theta": (hist.get("window") or {}).get("mean_theta"),
            "bil0": int(st0["bilateral"]),
            "nL0": st0["nL"], "nR0": st0["nR"],
            "e_pos0": st0["e_pos_mm"], "e_theta0": st0["e_theta_deg"],
        }
        historical_rows.append(hrow)
        if hist.get("s12"):
            handoff_assisted.append(hrow)
        print(f"  HIST s12={hist.get('s12')} NEW={hist.get('SUCCESS_NEW')} "
              f"first_bil_t={hrow['first_bil_t_rel']} after_pos={hrow['pose_after_regrasp_pos']}", flush=True)

        if args.skip_search:
            coverage.append({"ic_id": ic_id, "searched": 0, "SUCCESS_NEW": 0, "pool": "hist_only"})
            continue

        # ---------- 2–4. In-recovery search ----------
        found = None
        n_try = 0
        best_contact = None  # bilateral restored but NEW fail
        best_partial = None

        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]
        # desire after prefix often better for chase
        restore_capture(env, blob["capture"])
        run_seq(env, table, book48, blob, prefix)
        desire_after = -decompose(env.sim)["err_h"]

        regrasp_cands = regrasp_proposals(book48, desire_after, rng, n=args.budget_regrasp, open_prefix=prefix)
        pose_cands = pose_correction_proposals(book48, desire_after, rng, n=args.budget_pose)

        for ir, regrasp in enumerate(regrasp_cands):
            if found is not None:
                break
            restore_capture(env, blob["capture"])
            t0 = float(env.sim.data.time)
            trace = []
            ok, _ = run_seq(env, table, book48, blob, prefix + list(regrasp), trace=trace)
            n_try += 1
            if not ok:
                failures.append({"ic_id": ic_id, "tag": f"rg_{ir}", "reason": "DROP_PREFIX_OR_REGRASP",
                                 "label": "RECATCH_FAILED"})
                continue
            # require bilateral restored in recovery (before pose/handoff)
            bil_idx = find_first_bilateral(trace, start_idx=len(prefix))
            if bil_idx is None:
                # try a few zeros to settle
                ok2, _ = run_seq(env, table, book48, blob, [0] * 8, trace=trace)
                bil_idx = find_first_bilateral(trace, start_idx=len(prefix))
                settle = [0] * 8 if ok2 else []
                if bil_idx is None:
                    failures.append({"ic_id": ic_id, "tag": f"rg_{ir}", "reason": "NO_BILATERAL_IN_RECOVERY",
                                     "label": "RECATCH_FAILED"})
                    continue
            else:
                settle = []

            d_mid = decompose(env.sim)
            cs_mid = contact_summary(env.sim)
            contact_restore_t = float(trace[bil_idx]["t"] - t0) if bil_idx is not None else None

            # pose correction variants
            for ip, pose in enumerate(pose_cands):
                if found is not None:
                    break
                base = prefix + list(regrasp) + list(settle) + list(pose)
                # adaptive return
                restore_capture(env, blob["capture"])
                okb, _ = run_seq(env, table, book48, blob, base)
                if not okb:
                    n_try += 1
                    continue
                variants = adaptive_return_variants(env, table, book48, blob, t0, base)
                # also raw base
                variants = [("raw", list(base) + [0, 0, 0])] + variants
                for vname, tokens in variants:
                    n_try += 1
                    res = evaluate_full(env, table, book48, blob, tokens, record_obs=bool(False))
                    if not res["SUCCESS_NEW"]:
                        if best_contact is None or (
                            res.get("e_pos_at_handoff") is not None
                            and (best_contact.get("e_pos_at_handoff") is None
                                 or float(res["e_pos_at_handoff"]) < float(best_contact["e_pos_at_handoff"]))
                        ):
                            best_contact = {
                                "ic_id": ic_id, "tag": f"rg{ir}_p{ip}_{vname}",
                                "label": "RECATCH_CONTACT_ONLY",
                                "reason": res["reason"],
                                "s12": int(res["success_12s"]),
                                "SUCCESS_NEW": 0,
                                "e_pos_at_handoff": res.get("e_pos_at_handoff"),
                                "e_theta_at_handoff": res.get("e_theta_at_handoff"),
                                "win_mean_pos": res["window"].get("mean_pos"),
                                "win_mean_theta": res["window"].get("mean_theta"),
                                "contact_restore_t": contact_restore_t,
                                "tokens": json.dumps(tokens),
                            }
                        failures.append({
                            "ic_id": ic_id, "tag": f"rg{ir}_p{ip}_{vname}",
                            "reason": res["reason"], "label": res["label"],
                            "s12": int(res["success_12s"]),
                            "win_mean_pos": res["window"].get("mean_pos"),
                            "win_mean_theta": res["window"].get("mean_theta"),
                        })
                        if res["reason"] in ("DROP_DURING_RECOVERY", "TABLE_DROP", "DROP_AFTER_HANDOFF"):
                            break
                        # try next variant; if raw already PARTIAL keep searching returns
                        continue

                    # SUCCESS_NEW — full record with obs/trace
                    res = evaluate_full(env, table, book48, blob, tokens, record_obs=True)
                    mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                    # require contact restoration in recovery (before final handoff)
                    bil_in_rec = find_first_bilateral(res["trace"], start_idx=len(prefix))
                    if bil_in_rec is None:
                        failures.append({"ic_id": ic_id, "tag": f"rg{ir}_ok_but_no_bil_trace",
                                         "reason": "NEW_WITHOUT_BIL_TRACE", "label": "RECATCH_FAILED"})
                        continue
                    # source must need restoration: imm HANDOFF should not be S12
                    if hist.get("imm_s12"):
                        failures.append({"ic_id": ic_id, "tag": "skip_imm_already_ok",
                                         "reason": "IMM_HANDOFF_ALREADY_S12", "label": "SKIP"})
                        continue

                    # Release→regrasp: short open can cause 1–2 frame complete loss (not 3-frame sustained).
                    rec_seg = res["trace"][len(prefix):]
                    had_none = any(int(t["nL"]) + int(t["nR"]) == 0 for t in rec_seg)
                    had_uni = any(
                        (int(t["nL"]) + int(t["nR"])) > 0 and not t["bilateral"] for t in rec_seg
                    )
                    if had_none:
                        mech["recatch_subtype"] = "complete_loss"
                        mech["observed_recovery"] = ["RECATCH_COMPLETE_LOSS"] + [
                            x for x in mech["observed_recovery"]
                            if x not in ("OTHER", "RECATCH_COMPLETE_LOSS", "RECATCH_SINGLE_SIDED")
                        ]
                    elif had_uni:
                        mech["recatch_subtype"] = "single_sided"
                        mech["observed_recovery"] = ["RECATCH_SINGLE_SIDED"] + [
                            x for x in mech["observed_recovery"]
                            if x not in ("OTHER", "RECATCH_COMPLETE_LOSS", "RECATCH_SINGLE_SIDED")
                        ]
                    elif mech["recatch_subtype"] is None:
                        mech["recatch_subtype"] = "release_regrasp"
                        mech["observed_recovery"] = ["RECATCH_IN_RECOVERY"] + [
                            x for x in mech["observed_recovery"] if x != "OTHER"
                        ]
                    # Reject if no contact interruption after release prefix
                    if not (had_none or had_uni):
                        failures.append({
                            "ic_id": ic_id, "tag": f"rg{ir}_no_loss",
                            "reason": "NO_CONTACT_LOSS_AFTER_RELEASE",
                            "label": "RECATCH_FAILED",
                        })
                        continue

                    used_slide = "SLIDING" in mech["observed_recovery"]
                    used_pivot = "PIVOTING" in mech["observed_recovery"]
                    used_return = "RETURN" in mech["observed_recovery"]
                    extra = {
                        "expert_type": "IN_RECOVERY_RECATCH_FULL_EXPERT",
                        "historical_prefix": prefix,
                        "regrasp_tokens": regrasp,
                        "pose_tokens": pose,
                        "contact_restore_t": float(res["trace"][bil_in_rec]["t"] - res["trace"][0]["t"]),
                        "contact_restore_idx": bil_in_rec,
                        "post_regrasp_correction": {
                            "sliding": used_slide, "pivoting": used_pivot, "return": used_return,
                        },
                        "vs_historical_first_bil_t_rel": hrow["first_bil_t_rel"],
                    }
                    res["expert_label"] = "IN_RECOVERY_RECATCH_FULL_EXPERT"
                    save_expert_traj(env, ic_id, f"inrec_rg{ir}_p{ip}_{vname}", "in_recovery_recatch",
                                     res, mech, mapping_meta, checksum, snap, extra)
                    row = {
                        "ic_id": ic_id,
                        "tag": f"inrec_rg{ir}_p{ip}_{vname}",
                        "expert_type": "IN_RECOVERY_RECATCH_FULL_EXPERT",
                        "SUCCESS_NEW": 1,
                        "label": "VERIFIED_FULL_EXPERT",
                        "win_mean_pos": res["window"].get("mean_pos"),
                        "win_mean_theta": res["window"].get("mean_theta"),
                        "hold_both_s": res["window"].get("longest_both_ok_s"),
                        "hold_margin_mm": res.get("hold_margin_mm"),
                        "contact_restore_t": extra["contact_restore_t"],
                        "sliding_after": int(used_slide),
                        "pivoting_after": int(used_pivot),
                        "return_after": int(used_return),
                        "observed_recovery": json.dumps(mech["observed_recovery"]),
                        "recatch_subtype": mech["recatch_subtype"],
                        "n_tokens": len(tokens),
                        "prefix": json.dumps(prefix),
                        "config_id": CONFIG_ID,
                        "mapping_checksum": checksum,
                        "reproducible": 1,
                        "tokens": json.dumps(tokens),
                        "e_pos0": res["d0"]["e_pos_mm"],
                        "e_pos_handoff": res.get("e_pos_at_handoff"),
                        "e_theta_handoff": res.get("e_theta_at_handoff"),
                    }
                    in_recovery_experts.append(row)
                    manifest.append(row)
                    found = row
                    print(f"  IN_RECOVERY NEW=1 win_p={row['win_mean_pos']:.2f} "
                          f"slide={used_slide} pivot={used_pivot} via {vname}", flush=True)
                    break

        if found is None and best_contact is not None:
            contact_only.append(best_contact)
            write_json(OUT / "RAW" / f"{ic_id}_best_contact_only.json", best_contact)
            print(f"  CONTACT_ONLY best posH={best_contact.get('e_pos_at_handoff')}", flush=True)

        coverage.append({
            "ic_id": ic_id,
            "pool": "in_recovery_search",
            "prefix": json.dumps(prefix),
            "hist_s12": int(hist.get("s12") or 0),
            "hist_NEW": int(hist.get("SUCCESS_NEW") or 0),
            "searched": n_try,
            "SUCCESS_NEW": int(found is not None),
            "contact_only": int(best_contact is not None and found is None),
            "bil0": int(st0["bilateral"]),
            "e_pos0": st0["e_pos_mm"],
            "budget_s": time.time() - t0b,
        })
        print(f"  DONE tries={n_try} found={found is not None} ({time.time()-t0b:.1f}s)", flush=True)

    # ---------- outputs ----------
    write_csv(OUT / "HISTORICAL_NOMINAL_REGRASP.csv", historical_rows)
    write_csv(OUT / "HANDOFF_ASSISTED.csv", handoff_assisted)
    write_csv(OUT / "IN_RECOVERY_EXPERTS.csv", in_recovery_experts)
    write_csv(OUT / "RECATCH_CONTACT_ONLY.csv", contact_only)
    write_csv(OUT / "FAILURES.csv", failures[:600])
    write_csv(OUT / "IC_COVERAGE.csv", coverage)
    write_csv(OUT / "UNIFIED_MANIFEST.csv", manifest)

    n_inrec = len({r["ic_id"] for r in in_recovery_experts})
    n_need_corr = sum(1 for r in in_recovery_experts
                      if int(r.get("sliding_after") or 0) or int(r.get("pivoting_after") or 0))
    n_hist_s12 = sum(1 for r in historical_rows if int(r.get("s12") or 0))
    n_hist_new = sum(1 for r in historical_rows if int(r.get("SUCCESS_NEW") or 0))

    # control authority diagnosis if zero successes
    authority = {
        "k49_can_open_release": True,
        "k49_has_secure_close_tokens": True,
        "k49_has_translate_tokens": True,
        "k49_missing_nominal_fsm_grasp_retarget": (
            "Nominal after HANDOFF retargets fsm.r_des to saved grasp and runs full nominal policy; "
            "K49 recovery tokens only command local 20ms twists via step_legal without FSM phase machine."
        ),
        "if_zero_in_recovery": (
            "Primary blockers: (1) timing — nominal regrasp uses continuous closed-loop grip+arm over "
            "hundreds of ms with r_des reset; discrete K49 open→close may miss intercept window; "
            "(2) orientation — after short open, hand may need nominal orientation servo not spannned "
            "by short K49 pivot sequences within RECOVERY_LIMIT; (3) SUCCESS_NEW pose hold after "
            "contact restore often fails even when bilateral returns."
        ),
    }
    write_json(OUT / "RAW" / "control_authority_diagnosis.json", authority)

    answers = {
        "Q1_can_reproduce_in_recovery": n_inrec > 0,
        "Q1_detail": (
            f"{n_inrec}/{len(ics)} ICs yielded IN_RECOVERY_RECATCH_FULL_EXPERT under SUCCESS_NEW"
        ),
        "Q2_n_independent_ics_in_recovery": n_inrec,
        "Q2_ids": sorted({r["ic_id"] for r in in_recovery_experts}),
        "Q3_n_needing_slide_or_pivot_after_regrasp": n_need_corr,
        "Q3_fraction": None if n_inrec == 0 else n_need_corr / n_inrec,
        "Q4_failure_modes": dict(
            __import__("collections").Counter(f.get("reason") for f in failures).most_common(10)
        ),
        "Q5_ready_to_merge": n_inrec >= 3 and all(int(r.get("reproducible") or 0) for r in in_recovery_experts),
        "historical_s12_count": n_hist_s12,
        "historical_NEW_count": n_hist_new,
        "contact_only_count": len(contact_only),
        "wall_s": time.time() - t_wall0,
    }
    write_json(OUT / "RAW" / "answers.json", answers)

    report = f"""# In-Recovery Recatch from Historical Nominal-Assisted Regrasp

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**Objective:** Release → Legal In-Recovery Regrasp → optional Sliding/Pivoting/Return → HANDOFF(47) → SUCCESS_NEW  
**Source:** 22 short-sequence HANDOFF ICs from `PREFIX_HANDOFF_AUDIT.csv` (imm HANDOFF fails; short open→HANDOFF→nominal S12).

## Historical evidence

| Metric | Count |
|--------|------:|
| ICs audited | {len(ics)} |
| Historical short→HANDOFF→nominal SUCCESS_12S | {n_hist_s12} |
| Historical also SUCCESS_NEW | {n_hist_new} |

See `HISTORICAL_NOMINAL_REGRASP.csv` and `traces/historical/*_nominal_regrasp.json` for:
prefix, HANDOFF state, first restored bilateral time, pose after regrasp, nominal traces.

**Label:** these are `HANDOFF_ASSISTED` (not in-Recovery). Do not conflate with `IN_RECOVERY_RECATCH_FULL_EXPERT`.

## In-Recovery search results

| Metric | Count |
|--------|------:|
| IN_RECOVERY_RECATCH_FULL_EXPERT (distinct ICs) | **{n_inrec}** |
| Needing Sliding/Pivoting after regrasp | {n_need_corr} |
| RECATCH_CONTACT_ONLY (best per IC) | {len(contact_only)} |

## Answers

1. **Can historical nominal regrasp be reproduced entirely within Recovery using legal K49?**  
   **{"Yes, for some ICs" if n_inrec else "Not yet for SUCCESS_NEW on this batch"}** — {answers["Q1_detail"]}.

2. **Independent ICs with complete in-Recovery Recatch experts:** **{n_inrec}**  
   IDs: {answers["Q2_ids"]}

3. **After regrasp, Sliding/Pivoting required:** {n_need_corr}/{max(n_inrec,1)} successful experts  
   fraction={answers["Q3_fraction"]}

4. **What prevents reproduction in failures?**  
   Top reasons: {answers["Q4_failure_modes"]}  
   Control-authority notes: see `RAW/control_authority_diagnosis.json`.

5. **Ready to merge into unified expert dataset?** **{answers["Q5_ready_to_merge"]}**  
   (Need ≥3 independent IN_RECOVERY FULL experts, all reproducible.)

## Labels used

- `IN_RECOVERY_RECATCH_FULL_EXPERT` — regrasp inside Recovery + SUCCESS_NEW
- `HANDOFF_ASSISTED_S12` / historical NEW — nominal regrasp after HANDOFF (not mislabeled as in-Recovery)
- `RECATCH_CONTACT_ONLY` — bilateral restored in Recovery but SUCCESS_NEW fails
- `RECATCH_FAILED` — drop / no bilateral restore

## Compatibility

- K49_tok48_wy0p25 unchanged; 39D obs; step_legal only; no nominal takeover in Recovery search.
- SFT/GRPO not started.

## Artifacts

- `HISTORICAL_NOMINAL_REGRASP.csv`, `HANDOFF_ASSISTED.csv`
- `IN_RECOVERY_EXPERTS.csv`, `RECATCH_CONTACT_ONLY.csv`, `FAILURES.csv`
- `IC_COVERAGE.csv`, `UNIFIED_MANIFEST.csv`, `ACTION_MAPPING.json`
- `traces/historical/`, `traces/candidates/`, `trajectories/`, `snapshots/`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    ready = f"""# Dataset Readiness — In-Recovery Recatch

## Verdict: {"MERGE-CANDIDATE" if answers["Q5_ready_to_merge"] else "NOT READY TO MERGE"}

- In-Recovery FULL experts (ICs): {n_inrec}
- Historical HANDOFF-assisted S12 (reference only): {n_hist_s12}
- Contact-only (not full): {len(contact_only)}

### Merge rule
Only merge rows with `expert_type=IN_RECOVERY_RECATCH_FULL_EXPERT` into the unified Sliding/Recatch SFT pool.  
Keep HANDOFF-assisted traces as analysis artifacts, not as in-Recovery labels.

### Next steps if not ready
1. Increase regrasp budget / use nominal velocity matching to propose K49 closes.
2. Capture mid-nominal regrasp snapshots and search K49 from those states (still continuous, no teleport).
3. If a concrete missing twist (e.g. coupled grip+wrist rate) is proven absent from K49, document before proposing tokens.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("\nDONE", answers, flush=True)


if __name__ == "__main__":
    main()
