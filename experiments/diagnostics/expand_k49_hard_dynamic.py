"""Targeted expansion: hard dynamic ball-impact SUCCESS_NEW experts.

Locked K49_tok48_wy0p25. No SFT/GRPO. Finite 3-round budget.
Only nontrivial dynamic roots (fail simple baselines) count toward the 20-root target.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_generation.datasets.ball_impact import run_episode
from data_generation.datasets.recovery_rebuild import immediate_handoff
from data_generation.recording.trajectory_recorder import save_pair
from envs.simulation.capture import restore_capture, take_capture
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.grasp_geometry import measure
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import (
    SECURE_REVERSE_WRIST,
    assert_k48_prefix_identical,
    build_k49_table,
    describe_k49_config,
)
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env

from scripts.generate_k49_ball_impact_dynamic_recovery import (
    classify_ic_regime,
    draw_lateral_spec,
)
from scripts.generate_k49_dynamic_recatch_experts import (
    find_bilateral_after_loss,
    run_traced,
    type2_proposals,
)
from scripts.generate_k49_in_recovery_recatch import (
    find_first_bilateral,
    pose_correction_proposals,
    regrasp_proposals,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    classify_mechanisms,
    decompose,
    evaluate_full,
    hand_ref_error_deg,
    mapping_checksum,
    run_seq,
    slide_proposals,
    write_csv,
    write_json,
)
from scripts.generate_k49_type3_capture_brake import run_traced_with_snapshots

OUT = ROOT / "results" / "expert_generation" / "k49_hard_dynamic_expansion"
UNIFIED = ROOT / "results" / "expert_generation" / "k49_unified_dataset_expansion"
BALL = ROOT / "results" / "expert_generation" / "k49_ball_impact_dynamic_recovery"
HARD_SNAPS = ROOT / "results" / "diagnostics" / "hard_ic_mining" / "snapshots"
SEED = 20261019
CAPTURE_SEED = [30, 12, 12, 8] + [0] * 7

# Explicit finite budget (recorded before search)
TOTAL_ROLLOUT_BUDGET = 5500
WALL_LIMIT_S = 90 * 60  # 90 minutes
ROUND_FRAC = (0.35, 0.40, 0.25)
TARGET_NEW_HARD_ROOTS = 20


def tokens_hash(toks):
    return hashlib.sha256(json.dumps([int(x) for x in toks]).encode()).hexdigest()[:16]


def ic_metrics(env, blob):
    restore_capture(env, blob["capture"])
    cs = contact_summary(env.sim)
    m = measure(env.sim, U_TGT, P_TGT)
    d = decompose(env.sim)
    v = np.asarray(m["v_rel_h"], float)
    w = np.asarray(m["w_rel_h"], float)
    return {
        "nL": int(cs["nL"]), "nR": int(cs["nR"]), "bilateral": bool(cs["bilateral"]),
        "e_theta_deg": float(m["e_theta_deg"]), "e_pos_mm": float(m["e_pos_mm"]),
        "e_ax_mm": float(d["e_ax_mm"]), "e_lat_mm": float(d["e_lat_mm"]),
        "v_rel_norm": float(np.linalg.norm(v)),
        "w_rel_norm": float(np.linalg.norm(w)),
        "v_lat": float(np.hypot(v[0], v[1])),
        "v_vert": float(v[2]),
        "v_rel_h": v.tolist(),
        "w_rel_h": w.tolist(),
        "obj_z": float(m["obj_z"]),
    }


def freeze_thresholds(metrics_rows):
    """Freeze HIGH_MOTION thresholds from observed distribution BEFORE success known."""
    vn = np.array([float(r["v_rel_norm"]) for r in metrics_rows], float)
    wn = np.array([float(r["w_rel_norm"]) for r in metrics_rows], float)
    vl = np.array([float(r["v_lat"]) for r in metrics_rows], float)
    # Use 60th percentile of disturbed pool as HIGH_MOTION cut (report units)
    thr = {
        "v_rel_norm_mps": float(np.percentile(vn, 60)) if len(vn) else 0.15,
        "w_rel_norm_radps": float(np.percentile(wn, 60)) if len(wn) else 3.0,
        "v_lat_mps": float(np.percentile(vl, 60)) if len(vl) else 0.12,
        "rule": "HIGH_MOTION if (v_rel_norm >= thr_v) OR (w_rel_norm >= thr_w) OR (v_lat >= thr_lat)",
        "percentile": 60,
        "n_pool": len(metrics_rows),
        "units": {"v_rel_norm": "m/s", "w_rel_norm": "rad/s", "v_lat": "m/s"},
    }
    # floors so trivial zero-motion never qualifies
    thr["v_rel_norm_mps"] = max(thr["v_rel_norm_mps"], 0.12)
    thr["w_rel_norm_radps"] = max(thr["w_rel_norm_radps"], 2.5)
    thr["v_lat_mps"] = max(thr["v_lat_mps"], 0.10)
    return thr


def is_high_motion(st, thr):
    return (
        float(st["v_rel_norm"]) >= thr["v_rel_norm_mps"]
        or float(st["w_rel_norm"]) >= thr["w_rel_norm_radps"]
        or float(st["v_lat"]) >= thr["v_lat_mps"]
    )


def failure_stage(env, table, book48, blob, tokens):
    """Classify best partial failure stage A–G for a candidate sequence."""
    ok, tr = run_traced(env, table, book48, blob, tokens)
    if not tr:
        return "G", tr
    saw_any = any(int(r["nL"]) + int(r["nR"]) > 0 for r in tr)
    saw_uni = any((int(r["nL"]) > 0) ^ (int(r["nR"]) > 0) for r in tr)
    bil_i = find_first_bilateral(tr, 0) or find_bilateral_after_loss(tr)
    if not saw_any:
        return "A", tr  # never intercepted
    if not ok and not bil_i:
        # had some contact then lost
        if saw_uni:
            return "C", tr
        return "B", tr
    if bil_i is None:
        return "C" if saw_uni else "B", tr
    # bilateral restored
    res = evaluate_full(env, table, book48, blob, tokens, record_obs=False)
    if res["SUCCESS_NEW"]:
        return "OK", tr
    if res["reason"] in ("TABLE_DROP", "DROP_DURING_RECOVERY"):
        return "G", tr
    if res["reason"] in ("DROP_AFTER_HANDOFF", "LOST_GRASP") and res.get("success_12s") is False:
        # check if pose was ok at handoff-ish
        last = tr[-1]
        if float(last["e_theta_deg"]) <= 12 and float(last["e_pos_mm"]) <= 15:
            return "F", tr
        return "D", tr
    if res.get("success_12s") and not res["SUCCESS_NEW"]:
        return "E", tr  # stabilized but pose hold failed
    # bilateral but not stable
    last = tr[min(bil_i + 5, len(tr) - 1)]
    if float(np.linalg.norm(last["v_rel_h"])) > 0.15 or float(np.linalg.norm(last["w_rel_h"])) > 4:
        return "D", tr
    if float(last["e_theta_deg"]) > 10 or float(last["e_pos_mm"]) > 12:
        return "E", tr
    return "F", tr


def load_seed_library():
    seeds = []
    for path in (
        UNIFIED / "CANONICAL_MANIFEST.csv",
        BALL / "FULL_EXPERTS.csv",
        ROOT / "results/expert_generation/k49_dynamic_recatch_experts/TYPE2_EXPERTS.csv",
        ROOT / "results/expert_generation/k49_lost_grasp_recovery/TYPE3_FULL_EXPERTS.csv",
    ):
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if not r.get("tokens"):
                continue
            try:
                seeds.append(json.loads(r["tokens"]))
            except Exception:
                pass
    generics = [
        CAPTURE_SEED,
        [30] + [0] * 8,
        [30, 12, 12, 8] + [17] * 16 + [0] * 12,
        [30] + [17] * 14 + [0] * 10,
        [30] + [33] * 14 + [0] * 10,
        [36] * 3 + [12] * 4 + [17] * 16 + [0] * 10,
        [32] + [10] * 12 + [12] * 5 + [0] * 8,
        [4] * 20 + [12] * 5 + [0] * 5,
        [4] * 24 + [12] * 5 + [17] * 16 + [0] * 8,
        [6] * 16 + [12] * 5 + [0] * 8,
        [1] * 12 + [0] * 8,  # chase-ish
        [5] * 12 + [0] * 8,
        [13] * 12 + [0] * 8,
        [17] * 20 + [12] * 5 + [0] * 8,
        [33] * 20 + [12] * 5 + [SECURE_REVERSE_WRIST] * 20 + [0] * 8,
        [30] + [8] * 10 + [12] * 5 + [0] * 10,
        [30] + [15] * 10 + [12] * 5 + [0] * 10,
        [44] + [0] * 16,
        # longer multi-stage templates
        [30] + [17] * 10 + [12] * 4 + [4] * 16 + [12] * 5 + [0] * 8 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
        [30, 12, 12] + [33] * 12 + [12] * 4 + [6] * 12 + [12] * 5 + [0] * 8 + [SECURE_REVERSE_WRIST] * 20 + [0] * 8,
    ]
    seen, out = set(), []
    for s in seeds + generics:
        if not s:
            continue
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        out.append(list(k))
    return out


def long_stage_proposals(book, desire, rng, n=24):
    """Longer multi-stage chase→capture→manip→return proposals."""
    props = []
    props.extend(type2_proposals(book, desire, rng, n=n // 2, open_prefix=[30]))
    props.extend(regrasp_proposals(book, desire, rng, n=n // 2))
    # chase translations then soft close
    for t in (1, 5, 8, 13, 15, 17, 19, 33):
        props.append([t] * 12 + [12] * 4 + [0] * 12)
        props.append([30] + [t] * 14 + [12] * 5 + [0] * 12)
        props.append([30] + [t] * 10 + [12] * 4 + [4] * 16 + [12] * 5 + [0] * 8)
        props.append([30] + [t] * 10 + [12] * 4 + [6] * 12 + [12] * 5 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8)
    props.extend(slide_proposals(book, desire, rng, n=8))
    seen, uniq = set(), []
    for s in props:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    rng.shuffle(uniq)
    return uniq[:n]


def search_hard_ic(env, table, book48, blob, rng, seeds, budget_a, budget_b, rollout_counter, rollout_limit):
    """Multi-stage search; returns best SUCCESS_NEW or None. Counts rollouts."""
    if rollout_counter[0] >= rollout_limit:
        return None, [], 0
    restore_capture(env, blob["capture"])
    desire = -decompose(env.sim)["err_h"]
    cs0 = contact_summary(env.sim)
    has_contact = int(cs0["nL"]) + int(cs0["nR"]) > 0
    props = []
    props.extend(seeds[:14])
    if has_contact:
        # Contact-preserving chase / slide / pivot before any open
        props.extend(slide_proposals(book48, desire, rng, n=max(10, budget_a // 3)))
        for t in (1, 5, 8, 13, 15, 17, 19, 33):
            props.append([t] * 14 + [12] * 5 + [0] * 12)
            props.append([t] * 10 + [12] * 4 + [4] * 16 + [12] * 5 + [0] * 8)
            props.append([12] * 4 + [t] * 16 + [SECURE_REVERSE_WRIST] * 20 + [0] * 8)
        props.append([4] * 20 + [12] * 5 + [17] * 16 + [SECURE_REVERSE_WRIST] * 20 + [0] * 8)
    props.extend(long_stage_proposals(book48, desire, rng, n=max(10, budget_a // 2)))
    if not has_contact or float(rng.random()) < 0.7:
        props.extend(type2_proposals(book48, desire, rng, n=budget_a // 3, open_prefix=[30]))
    seen, uniq = set(), []
    for s in props:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    head, tail = uniq[:18], uniq[18:]
    rng.shuffle(tail)
    uniq = (head + tail)[:budget_a]

    trials = []
    n_try = 0
    best = None

    for ia, seq in enumerate(uniq):
        if rollout_counter[0] + n_try >= rollout_limit:
            break
        variants = [
            ("raw", list(seq)),
            ("settle16", list(seq) + [0] * 16),
            ("settle32", list(seq) + [0] * 32),
            ("ret20", list(seq) + [SECURE_REVERSE_WRIST] * 20 + [0] * 8),
            ("ret32", list(seq) + [SECURE_REVERSE_WRIST] * 32 + [0] * 8),
        ]
        restore_capture(env, blob["capture"])
        ok, _ = run_seq(env, table, book48, blob, seq)
        n_try += 1
        if not ok:
            trials.append({"tag": f"a{ia}", "NEW": 0, "reason": "DROP", "s12": 0, "stage": "G"})
            continue
        # branch: post-prefix pose corrections (Stage C)
        restore_capture(env, blob["capture"])
        run_seq(env, table, book48, blob, seq)
        desire_b = -decompose(env.sim)["err_h"]
        for ib, corr in enumerate(pose_correction_proposals(book48, desire_b, rng, n=budget_b)[:budget_b]):
            if corr:
                variants.append((f"b{ib}", list(seq) + list(corr) + [0] * 12))
                variants.append((
                    f"b{ib}ret",
                    list(seq) + list(corr) + [SECURE_REVERSE_WRIST] * 20 + [0] * 8,
                ))

        for vname, toks in variants:
            if rollout_counter[0] + n_try >= rollout_limit:
                break
            n_try += 1
            res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
            if res["SUCCESS_NEW"]:
                stage = "OK"
            elif res["reason"] in ("TABLE_DROP", "DROP_DURING_RECOVERY"):
                stage = "G"
            elif res["reason"] in ("DROP_AFTER_HANDOFF", "LOST_GRASP"):
                stage = "F"
            elif res.get("success_12s"):
                stage = "E"
            else:
                stage = "D"
            row = {
                "tag": f"a{ia}_{vname}", "NEW": int(res["SUCCESS_NEW"]),
                "s12": int(res["success_12s"]), "reason": res["reason"],
                "stage": stage,
                "win_th": res["window"].get("mean_theta"),
                "win_pos": res["window"].get("mean_pos"),
                "n_tokens": len(toks),
            }
            trials.append(row)
            if res["SUCCESS_NEW"]:
                best = {"tokens": toks, "tag": f"a{ia}_{vname}", "res": res, "n_try": n_try}
                rollout_counter[0] += n_try
                return best, trials, n_try

        # failure-driven branching on best non-NEW long prefix
        if ok and best is None and len(seq) >= 8:
            if rollout_counter[0] + n_try + 5 >= rollout_limit:
                continue
            ok2, tr2, snaps = run_traced_with_snapshots(env, table, book48, blob, seq)
            n_try += 1
            # branch at first unilateral, first bilateral, peak |w|
            idxs = []
            for i, r in enumerate(tr2):
                if (int(r["nL"]) > 0) ^ (int(r["nR"]) > 0):
                    idxs.append(("uni", i))
                    break
            bil = find_first_bilateral(tr2, 0) or find_bilateral_after_loss(tr2)
            if bil is not None:
                idxs.append(("bil", bil))
            if tr2:
                wi = int(np.argmax([float(np.linalg.norm(r["w_rel_h"])) for r in tr2]))
                idxs.append(("peak_w", wi))
            for bname, bidx in idxs[:2]:
                if rollout_counter[0] + n_try >= rollout_limit:
                    break
                prefix = list(seq[: bidx + 1])
                restore_capture(env, snaps[bidx]["capture"])
                desire_c = -decompose(env.sim)["err_h"]
                for corr in pose_correction_proposals(book48, desire_c, rng, n=4)[:4]:
                    if not corr:
                        continue
                    toks = prefix + list(corr) + [0] * 10 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8
                    n_try += 1
                    res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
                    trials.append({
                        "tag": f"a{ia}_br_{bname}", "NEW": int(res["SUCCESS_NEW"]),
                        "s12": int(res["success_12s"]), "reason": res["reason"],
                        "stage": "OK" if res["SUCCESS_NEW"] else "BR",
                        "win_th": res["window"].get("mean_theta"),
                        "win_pos": res["window"].get("mean_pos"),
                        "n_tokens": len(toks),
                    })
                    if res["SUCCESS_NEW"]:
                        best = {"tokens": toks, "tag": f"a{ia}_br_{bname}", "res": res, "n_try": n_try}
                        rollout_counter[0] += n_try
                        return best, trials, n_try

    rollout_counter[0] += n_try
    return best, trials, n_try


def mechanism_tags(tokens, st0, high_motion):
    toks = [int(t) for t in tokens]
    tags = []
    if high_motion:
        tags.append("HIGH_ANGULAR_MOTION" if st0["w_rel_norm"] >= st0["v_rel_norm"] * 10 else "HIGH_MOTION")
    if st0["nL"] + st0["nR"] == 0:
        tags.append("COMPLETE_CONTACT_LOSS")
    if (st0["nL"] > 0) ^ (st0["nR"] > 0):
        tags.append("UNILATERAL")
    if any(t in (30, 32, 36, 39, 41, 44) for t in toks):
        tags.append("RECATCH")
    if any(t in (1, 5, 8, 13, 15, 17, 19, 33) for t in toks[:20]):
        tags.append("CHASE")
    if sum(1 for t in toks if t in (17, 33, 13, 1, 5, 19)) >= 8:
        tags.append("SLIDING")
    if sum(1 for t in toks if t in (4, 6, 10)) >= 8:
        tags.append("PIVOTING")
    if SECURE_REVERSE_WRIST in toks:
        tags.append("RETURN")
    if 12 in toks and any(t in (30, 32, 36) for t in toks):
        tags.append("CAPTURE_BRAKE")
    if "RECATCH" in tags and ("SLIDING" in tags or "PIVOTING" in tags):
        tags.append("RECATCH_PLUS_MANIP")
    return tags


def save_expert(env, table, book48, blob, ic_id, root, tag, toks, st, thr, simple_ok,
                source, mapping_meta, checksum, snap_src, regimes, round_id):
    res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
    if not res["SUCCESS_NEW"]:
        return None
    hm = is_high_motion(st, thr)
    tags = mechanism_tags(toks, st, hm)
    stem = f"{ic_id}__{tag}"
    save_pair(
        OUT / "trajectories", stem,
        res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
    meta = {
        "ic_id": ic_id, "root_episode": root, "tag": tag,
        "expert_type": "HARD_DYNAMIC_FULL_EXPERT",
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "SUCCESS_NEW": True, "window": res["window"], "tokens": toks,
        "high_motion": hm, "mechanism_tags": tags, "regimes": regimes,
        "newly_vs_simple": int(not simple_ok), "source": source,
        "round": round_id, "snapshot_src": str(snap_src),
        "obs_dim": 39, "handoff_id": 47,
        "v0": st["v_rel_norm"], "w0": st["w_rel_norm"], "v_lat0": st["v_lat"],
    }
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    if res.get("trace"):
        write_json(OUT / "raw_contact_traces" / f"{stem}_trace.json", {
            "trace": res["trace"], "mech": mech, "tags": tags,
        })
    dst = OUT / "snapshots" / f"{ic_id}.pkl"
    if not dst.exists():
        if Path(snap_src).exists():
            shutil.copy2(snap_src, dst)
        else:
            dst.write_bytes(pickle.dumps(blob))
    return {
        "ic_id": ic_id, "root_episode": root, "tag": tag,
        "expert_type": "HARD_DYNAMIC_FULL_EXPERT", "SUCCESS_NEW": 1,
        "win_mean_theta": res["window"].get("mean_theta"),
        "win_mean_pos": res["window"].get("mean_pos"),
        "hold_both_s": res["window"].get("longest_both_ok_s"),
        "n_tokens": len(toks), "tokens": json.dumps(toks),
        "tokens_hash": tokens_hash(toks),
        "high_motion": int(hm), "mechanism_tags": json.dumps(tags),
        "regimes": json.dumps(regimes),
        "newly_vs_simple": int(not simple_ok),
        "source": source, "round": round_id,
        "v0": st["v_rel_norm"], "w0": st["w_rel_norm"], "v_lat0": st["v_lat"],
        "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
        "nL0": st["nL"], "nR0": st["nR"], "bilateral0": int(st["bilateral"]),
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "snapshot": str(dst), "recatch_plus_manip": int("RECATCH_PLUS_MANIP" in tags),
    }


def audit_baseline():
    canon = list(csv.DictReader((UNIFIED / "CANONICAL_MANIFEST.csv").open(encoding="utf-8")))
    ics = {r["ic_id"] for r in canon}
    roots = {r["root_episode"] for r in canon}
    ball_roots = {r["root_episode"] for r in canon
                  if str(r["root_episode"]).startswith(("bimp_", "ubimp"))
                  or "ball" in str(r.get("source") or r.get("source_dataset") or "").lower()}
    # mapping / handoff checks
    bad_map = 0
    for r in canon:
        toks = json.loads(r["tokens"])
        if any(int(x) < 0 or int(x) > 48 for x in toks):
            bad_map += 1
    splits = list(csv.DictReader((UNIFIED / "TRAIN_VAL_TEST_SPLITS.csv").open(encoding="utf-8")))
    by_root = {}
    for r in splits:
        by_root.setdefault(r["root_episode"], set()).add(r["split"])
    leak = sum(1 for v in by_root.values() if len(v) > 1)
    # reconcile 53→68
    audit = {
        "n_traj": len(canon), "n_unique_ics": len(ics), "n_unique_roots": len(roots),
        "n_ball_impact_roots": len(ball_roots),
        "ball_roots": sorted(ball_roots),
        "bad_token_map": bad_map,
        "split_leak_roots": leak,
        "reconcile_note": (
            f"68 traj = prior SUCCESS_NEW rows + expansion variants; "
            f"50 unique roots (26 prior + 24 new). Ball historical roots among the 50: {len(ball_roots)} "
            f"(not all are 'newly generated')."
        ),
        "unique_roots": sorted(roots),
    }
    # freeze copy
    write_csv(OUT / "BASELINE_MANIFEST_AUDIT.csv", [
        {"key": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v}
        for k, v in audit.items()
    ])
    shutil.copy2(UNIFIED / "CANONICAL_MANIFEST.csv", OUT / "RAW" / "FROZEN_BASELINE_CANONICAL_MANIFEST.csv")
    return audit, roots, ball_roots


def collect_pool(solved_roots, env):
    """Build hard dynamic IC pool from failed ball + unsolved ubimp + hard high-v."""
    pool = []
    # 1) previously searched unsolved ball
    cov = list(csv.DictReader((BALL / "ROOT_EPISODE_COVERAGE.csv").open(encoding="utf-8")))
    for r in cov:
        rid = r["root_id"]
        if rid in solved_roots or str(r.get("full")) == "1":
            continue
        sp = BALL / "snapshots" / f"{rid}.pkl"
        if not sp.exists():
            continue
        pool.append({"ic_id": rid, "root_episode": rid, "snap": sp,
                     "origin": "failed_ball_searched", "prior_full": 0})
    # 2) accepted ubimp not in solved
    if (UNIFIED / "NEW_PHYSICAL_ROOTS.csv").exists():
        for r in csv.DictReader((UNIFIED / "NEW_PHYSICAL_ROOTS.csv").open(encoding="utf-8")):
            rid = r["root_id"]
            if rid in solved_roots or str(r.get("accepted")) != "1":
                continue
            sp = UNIFIED / "snapshots" / f"{rid}.pkl"
            if not sp.exists():
                continue
            pool.append({"ic_id": rid, "root_episode": rid, "snap": sp,
                         "origin": "failed_ubimp", "prior_full": 0})
    # 3) hard snaps with high motion potential not solved
    if HARD_SNAPS.exists():
        for sp in sorted(HARD_SNAPS.glob("hard_*.pkl")):
            if sp.stem in solved_roots:
                continue
            if any(p["ic_id"] == sp.stem for p in pool):
                continue
            pool.append({"ic_id": sp.stem, "root_episode": sp.stem, "snap": sp,
                         "origin": "hard_ic_mining", "prior_full": 0})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", type=int, default=TARGET_NEW_HARD_ROOTS)
    ap.add_argument("--rollout-budget", type=int, default=TOTAL_ROLLOUT_BUDGET)
    ap.add_argument("--wall-s", type=int, default=WALL_LIMIT_S)
    ap.add_argument("--max-new-impacts", type=int, default=40)
    ap.add_argument("--budget-a", type=int, default=20)
    ap.add_argument("--budget-b", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.rollout_budget = 400
        args.wall_s = 600
        args.max_new_impacts = 6
        args.target = 3
        args.budget_a = 10
        args.budget_b = 4

    t0w = time.time()
    for sub in ("trajectories", "snapshots", "raw_contact_traces", "RAW", "branch_snapshots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED + 11)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=== AUDIT baseline ===", flush=True)
    audit, solved_roots, hist_ball_roots = audit_baseline()
    print(json.dumps({k: audit[k] for k in ("n_traj", "n_unique_ics", "n_unique_roots",
                                            "n_ball_impact_roots", "bad_token_map", "split_leak_roots")},
                     indent=2), flush=True)
    write_json(OUT / "RAW" / "budget.json", {
        "total_rollout_budget": args.rollout_budget,
        "wall_limit_s": args.wall_s,
        "round_frac": ROUND_FRAC,
        "target_new_hard_roots": args.target,
        "recorded_before_search": True,
    })

    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    mapping_meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    mapping_meta["checksum_sha256_16"] = checksum
    write_json(OUT / "ACTION_MAPPING.json", mapping_meta)

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    seeds = load_seed_library()

    # ---------- measure pool & freeze thresholds ----------
    print("=== Build HARD pool + freeze thresholds ===", flush=True)
    raw_pool = collect_pool(solved_roots, env)
    # measure all (cap for smoke)
    measure_cap = 80 if not args.smoke else 25
    metrics_rows = []
    measured = []
    for item in raw_pool[:measure_cap]:
        try:
            blob = pickle.loads(item["snap"].read_bytes())
        except Exception:
            continue
        if "capture" not in blob or "ref" not in blob:
            continue
        st = ic_metrics(env, blob)
        # skip already-easy stable tiny errors without motion
        if st["v_rel_norm"] < 0.05 and st["w_rel_norm"] < 1.0 and st["bilateral"] and st["e_pos_mm"] < 14 and st["e_theta_deg"] < 8:
            continue
        ev = blob.get("evidence") or {}
        regimes = classify_ic_regime(blob, st) if isinstance(ev, dict) else []
        row = {
            **item,
            "blob": blob,
            "st": st,
            "regimes": regimes,
            "peak_fn": ev.get("peak_normal_force_n") if isinstance(ev, dict) else None,
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "v0": st["v_rel_norm"], "w0": st["w_rel_norm"], "v_lat0": st["v_lat"],
            "v_vert0": st["v_vert"],
            "nL": st["nL"], "nR": st["nR"], "bilateral": int(st["bilateral"]),
        }
        measured.append(row)
        metrics_rows.append({
            "ic_id": item["ic_id"], "origin": item["origin"],
            "v_rel_norm": st["v_rel_norm"], "w_rel_norm": st["w_rel_norm"],
            "v_lat": st["v_lat"], "v_vert": st["v_vert"],
            "e_theta_deg": st["e_theta_deg"], "e_pos_mm": st["e_pos_mm"],
            "nL": st["nL"], "nR": st["nR"], "bilateral": int(st["bilateral"]),
            "peak_fn": row["peak_fn"],
        })

    thr = freeze_thresholds(metrics_rows)
    write_json(OUT / "RAW" / "high_motion_thresholds.json", thr)
    print("FROZEN HIGH_MOTION thr", thr, flush=True)

    for row in measured:
        row["high_motion"] = int(is_high_motion(row["st"], thr))
        row["contact_class"] = (
            "NONE" if row["st"]["nL"] + row["st"]["nR"] == 0 else
            ("UNI" if (row["st"]["nL"] > 0) ^ (row["st"]["nR"] > 0) else "BIL")
        )

    write_csv(OUT / "PHYSICAL_DIFFICULTY_METRICS.csv", [
        {k: v for k, v in r.items() if k not in ("blob", "st", "snap", "regimes")}
        | {"regimes": json.dumps(r["regimes"]), "high_motion": r["high_motion"],
           "contact_class": r["contact_class"]}
        for r in measured
    ])

    # diagnose prior failure stages with a short probe (counts as rollouts)
    rollout_counter = [0]
    pool_rows = []
    for row in measured:
        # quick stage probe with capture-brake seed
        stage, _ = failure_stage(env, table, book48, row["blob"], CAPTURE_SEED)
        rollout_counter[0] += 2
        row["prior_best_stage"] = stage
        pool_rows.append({
            "ic_id": row["ic_id"], "root_episode": row["root_episode"],
            "origin": row["origin"], "high_motion": row["high_motion"],
            "contact_class": row["contact_class"],
            "prior_best_stage": stage,
            "v0": row["v0"], "w0": row["w0"], "v_lat0": row["v_lat0"],
            "e_theta0": row["e_theta0"], "e_pos0": row["e_pos0"],
            "nL": row["nL"], "nR": row["nR"], "bilateral": row["bilateral"],
            "regimes": json.dumps(row["regimes"]),
            "snapshot": str(row["snap"]),
        })
    write_csv(OUT / "HARD_DYNAMIC_IC_POOL.csv", pool_rows)

    n_hm = sum(1 for r in measured if r["high_motion"])
    n_failed_prev = sum(1 for r in measured if r["origin"].startswith("failed"))
    print(f"POOL n={len(measured)} HIGH_MOTION={n_hm} from_failed={n_failed_prev}", flush=True)

    # Prefer boundary recoverability: HIGH_MOTION but not ballistic escape,
    # and prior stages C–F. Extreme v/w often irreversible within horizon.
    stage_pri = {"C": 0, "D": 1, "E": 2, "F": 3, "B": 4, "A": 5, "G": 6, "OK": 9}

    def recoverability_rank(r):
        v, w = float(r["v0"]), float(r["w0"])
        # sweet spot: above HIGH_MOTION floor, below escape
        if 0.12 <= v <= 0.55 and w <= 15:
            band = 0
        elif v < 0.12 and (r["high_motion"] or w >= thr["w_rel_norm_radps"]):
            band = 1  # angular-dominated
        elif v <= 0.75:
            band = 2
        else:
            band = 3  # likely irreversible
        contact_bonus = 0 if r["st"]["nL"] + r["st"]["nR"] > 0 else 1
        return (band, contact_bonus, stage_pri.get(r["prior_best_stage"], 8), -min(v, 0.55))

    measured.sort(key=recoverability_rank)

    round_budgets = [int(args.rollout_budget * f) for f in ROUND_FRAC]
    # fix rounding
    round_budgets[2] = args.rollout_budget - sum(round_budgets[:2])

    baselines = []
    intercept_rows = []
    capture_rows = []
    post_rows = []
    handoff_rows = []
    full_experts = []
    failures = []
    coverage = []
    new_impact_log = []
    new_hard_roots = set()
    transfer_wins = 0
    multistage_wins = 0

    def time_left():
        return (time.time() - t0w) < args.wall_s

    def try_ic(row, round_id, round_limit, local_budget_a=None):
        nonlocal transfer_wins, multistage_wins
        if not time_left() or len(new_hard_roots) >= args.target:
            return False
        if rollout_counter[0] >= round_limit:
            return False
        ic_id = row["ic_id"]
        if ic_id in solved_roots or ic_id in new_hard_roots:
            return False
        blob, st = row["blob"], row["st"]
        t0 = time.time()

        # matched baselines — require all simple fail for nontrivial target
        res_a = evaluate_full(env, table, book48, blob, [], record_obs=False)
        res_b = evaluate_full(env, table, book48, blob, [0] * 16, record_obs=False)
        res_c = evaluate_full(env, table, book48, blob, [30] + [0] * 8, record_obs=False)
        res_d = evaluate_full(env, table, book48, blob, CAPTURE_SEED, record_obs=False)
        rollout_counter[0] += 4
        simple_ok = res_a["SUCCESS_NEW"] or res_b["SUCCESS_NEW"] or res_c["SUCCESS_NEW"]
        baselines.append({
            "ic_id": ic_id, "round": round_id, "high_motion": row["high_motion"],
            "A_NEW": int(res_a["SUCCESS_NEW"]), "B_NEW": int(res_b["SUCCESS_NEW"]),
            "C_NEW": int(res_c["SUCCESS_NEW"]), "D_NEW": int(res_d["SUCCESS_NEW"]),
            "simple_solves": int(simple_ok),
            "v0": st["v_rel_norm"], "w0": st["w_rel_norm"],
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
        })
        if simple_ok:
            coverage.append({
                "ic_id": ic_id, "round": round_id, "skip": "simple_solves",
                "full": 0, "high_motion": row["high_motion"],
            })
            return False  # do not count easy toward hard-dynamic target

        ba = local_budget_a or args.budget_a
        print(f"  [{round_id}] {ic_id} HM={row['high_motion']} stage={row.get('prior_best_stage')} "
              f"v={st['v_rel_norm']:.2f} w={st['w_rel_norm']:.2f} "
              f"th={st['e_theta_deg']:.1f} pos={st['e_pos_mm']:.1f} "
              f"new_hard={len(new_hard_roots)}/{args.target} roll={rollout_counter[0]}",
              flush=True)

        before = rollout_counter[0]
        best, trials, n_try = search_hard_ic(
            env, table, book48, blob, rng, seeds, ba, args.budget_b,
            rollout_counter, round_limit,
        )
        for t in trials:
            failures.append({"ic_id": ic_id, "round": round_id, **t})
            if "br_" in str(t.get("tag", "")):
                capture_rows.append({"ic_id": ic_id, **t})
            elif t.get("n_tokens", 0) > 40:
                post_rows.append({"ic_id": ic_id, **t})
            else:
                intercept_rows.append({"ic_id": ic_id, **t})
            if t.get("NEW"):
                handoff_rows.append({"ic_id": ic_id, **t})

        got = 0
        if best is not None:
            # classify transfer vs multistage by tag
            if "br_" in best["tag"] or "b" in best["tag"].split("_")[-1]:
                multistage_wins += 1
            elif best["tag"].endswith("raw") and best["n_try"] <= 5:
                transfer_wins += 1
            else:
                multistage_wins += 1
            erow = save_expert(
                env, table, book48, blob, ic_id, ic_id, best["tag"], best["tokens"],
                st, thr, False, row["origin"], mapping_meta, checksum, row["snap"],
                row["regimes"], round_id,
            )
            rollout_counter[0] += 1
            if erow is not None:
                full_experts.append(erow)
                new_hard_roots.add(ic_id)
                solved_roots.add(ic_id)
                got = 1
                print(f"    FULL HARD NEW root th={erow['win_mean_theta']:.1f} "
                      f"pos={erow['win_mean_pos']:.1f} tags={erow['mechanism_tags']}",
                      flush=True)
        coverage.append({
            "ic_id": ic_id, "round": round_id, "full": got,
            "high_motion": row["high_motion"], "prior_stage": row.get("prior_best_stage"),
            "searched": rollout_counter[0] - before, "budget_s": time.time() - t0,
            "simple_solves": 0,
        })
        return got == 1

    # ===== ROUND 1: existing hard / failed partials (C–F preferred) =====
    r1_limit = round_budgets[0]
    print(f"=== ROUND 1 existing hard (budget {r1_limit}) ===", flush=True)
    r1 = [r for r in measured if r["origin"].startswith("failed") or r["prior_best_stage"] in ("C", "D", "E", "F")]
    r1 = r1 or measured
    for row in r1:
        if rollout_counter[0] >= r1_limit or not time_left() or len(new_hard_roots) >= args.target:
            break
        try_ic(row, "R1", r1_limit)

    # ===== ROUND 2: new physical impacts =====
    r2_start = rollout_counter[0]
    r2_limit = r1_limit + round_budgets[1]
    print(f"=== ROUND 2 new impacts (budget to {r2_limit}) ===", flush=True)
    new_measured = []
    for i in range(args.max_new_impacts):
        if not time_left() or len(new_hard_roots) >= args.target:
            break
        spec = draw_lateral_spec(8000 + i)
        # bias harder but not extreme
        if float(rng.random()) < 0.55:
            spec["speed_mps"] = float(rng.uniform(7.5, 9.5))
        eid = f"hdimp_{i:04d}"
        spec["attempt_id"] = eid
        evidence, blob = run_episode(env, spec)
        log = {
            "root_id": eid, "category": evidence.get("category"),
            "azimuth_deg": spec["azimuth_deg"], "speed_mps": spec["speed_mps"],
            "phi_deg": spec["phi_deg"], "z_off_m": spec["z_off_m"],
            "peak_fn": evidence.get("peak_normal_force_n"),
            "launch_speed": evidence.get("launch_speed"),
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
        st = ic_metrics(env, blob)
        regimes = classify_ic_regime(blob, st)
        hm = int(is_high_motion(st, thr))
        log.update({
            "accepted": 1, "high_motion": hm,
            "v0": st["v_rel_norm"], "w0": st["w_rel_norm"], "v_lat0": st["v_lat"],
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "nL": st["nL"], "nR": st["nR"], "bilateral": int(st["bilateral"]),
            "regimes": json.dumps(regimes), "snapshot": str(path),
        })
        new_impact_log.append(log)
        metrics_rows.append({
            "ic_id": eid, "origin": "new_impact",
            "v_rel_norm": st["v_rel_norm"], "w_rel_norm": st["w_rel_norm"],
            "v_lat": st["v_lat"], "v_vert": st["v_vert"],
            "e_theta_deg": st["e_theta_deg"], "e_pos_mm": st["e_pos_mm"],
            "nL": st["nL"], "nR": st["nR"], "bilateral": int(st["bilateral"]),
            "peak_fn": log["peak_fn"],
        })
        stage, _ = failure_stage(env, table, book48, blob, CAPTURE_SEED)
        rollout_counter[0] += 2
        item = {
            "ic_id": eid, "root_episode": eid, "snap": path, "blob": blob, "st": st,
            "origin": "new_impact", "regimes": regimes, "high_motion": hm,
            "prior_best_stage": stage, "v0": st["v_rel_norm"], "w0": st["w_rel_norm"],
            "v_lat0": st["v_lat"], "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "nL": st["nL"], "nR": st["nR"], "bilateral": int(st["bilateral"]),
            "contact_class": (
                "NONE" if st["nL"] + st["nR"] == 0 else
                ("UNI" if (st["nL"] > 0) ^ (st["nR"] > 0) else "BIL")
            ),
        }
        new_measured.append(item)
        print(f"  NEW impact {eid} HM={hm} v={st['v_rel_norm']:.2f} stage={stage}", flush=True)

    write_csv(OUT / "NEW_IMPACT_ROOTS.csv", new_impact_log)
    # Prefer boundary HIGH_MOTION new impacts (not irreversible escapes)
    new_measured.sort(key=recoverability_rank)
    for row in new_measured:
        if rollout_counter[0] >= r2_limit or not time_left() or len(new_hard_roots) >= args.target:
            break
        try_ic(row, "R2", r2_limit, local_budget_a=args.budget_a + 4)

    # ===== ROUND 3: refine unresolved promising =====
    r3_limit = args.rollout_budget
    print(f"=== ROUND 3 refine (budget to {r3_limit}) ===", flush=True)
    # candidates: covered but not full, stages C–F or high motion
    tried = {c["ic_id"] for c in coverage}
    refine = []
    for row in measured + new_measured:
        if row["ic_id"] in new_hard_roots or row["ic_id"] in solved_roots:
            continue
        if row["ic_id"] in tried and any(
            c["ic_id"] == row["ic_id"] and int(c.get("full") or 0) == 0 for c in coverage
        ):
            if row.get("prior_best_stage") in ("C", "D", "E", "F") or row["high_motion"]:
                refine.append(row)
        elif row["ic_id"] not in tried and (row["high_motion"] or row.get("prior_best_stage") in ("C", "D", "E", "F")):
            refine.append(row)
    for row in refine:
        if rollout_counter[0] >= r3_limit or not time_left() or len(new_hard_roots) >= args.target:
            break
        try_ic(row, "R3", r3_limit, local_budget_a=args.budget_a + 8)

    # ---------- finalize ----------
    write_csv(OUT / "MATCHED_BASELINES.csv", baselines)
    write_csv(OUT / "INTERCEPTION_SEARCH.csv", intercept_rows[:2000])
    write_csv(OUT / "CAPTURE_BRAKE_SEARCH.csv", capture_rows[:2000])
    write_csv(OUT / "POST_RECATCH_MANIPULATION.csv", post_rows[:2000])
    write_csv(OUT / "HANDOFF_SEARCH.csv", handoff_rows[:500])
    write_csv(OUT / "FULL_EXPERTS.csv", full_experts)
    write_csv(OUT / "FAILURES.csv", failures[:2500])
    write_csv(OUT / "ROOT_COVERAGE.csv", coverage)

    # unified = frozen baseline + new hard
    baseline_canon = list(csv.DictReader((OUT / "RAW" / "FROZEN_BASELINE_CANONICAL_MANIFEST.csv").open(encoding="utf-8")))
    unified = list(baseline_canon)
    for r in full_experts:
        unified.append({
            "ic_id": r["ic_id"], "root_episode": r["root_episode"],
            "source_dataset": "k49_hard_dynamic_expansion",
            "SUCCESS_NEW": 1, "tokens": r["tokens"], "n_tokens": r["n_tokens"],
            "win_mean_theta": r["win_mean_theta"], "win_mean_pos": r["win_mean_pos"],
            "hold_both_s": r["hold_both_s"], "provenance": "hard_dynamic_expansion",
            "expert_type": r["expert_type"], "tokens_hash": r["tokens_hash"],
            "snapshot": r["snapshot"],
        })
    write_csv(OUT / "UNIFIED_MANIFEST.csv", unified)

    # rewrite physical metrics with all
    write_csv(OUT / "PHYSICAL_DIFFICULTY_METRICS.csv", metrics_rows)

    all_roots = {r["root_episode"] for r in unified}
    all_ics = {r["ic_id"] for r in unified}
    n_hm_pool = sum(1 for r in measured + new_measured if r.get("high_motion"))
    n_hm_full = sum(1 for r in full_experts if int(r.get("high_motion") or 0) == 1)
    n_recatch_manip = sum(1 for r in full_experts if int(r.get("recatch_plus_manip") or 0) == 1)
    n_contact = sum(1 for r in coverage if int(r.get("full") or 0) == 1)  # approx
    # better from failures stages
    stage_ctr = Counter(f.get("stage") for f in failures if f.get("stage"))
    fail_reason = Counter(f.get("reason") for f in failures)

    # contact restoration stats among searched
    n_uni_or_bil = sum(1 for f in failures if f.get("stage") in ("C", "D", "E", "F", "OK"))
    n_pose_needed = sum(1 for f in failures if f.get("stage") in ("E", "F"))

    answers = {
        "Q1_n_hard_dynamic_ics_before": len(measured),
        "Q2_n_from_failed_impact_roots": n_failed_prev,
        "Q3_n_new_independent_impact_episodes": sum(1 for r in new_impact_log if int(r.get("accepted") or 0) == 1),
        "Q3_n_new_impact_attempts": len(new_impact_log),
        "Q4_n_HIGH_MOTION_measured": n_hm_pool,
        "Q4_thresholds": thr,
        "Q5_n_restored_uni_or_bil_partials": n_uni_or_bil,
        "Q6_n_stable_but_pose_needed_partials": n_pose_needed,
        "Q7_n_recatch_plus_slide_pivot": n_recatch_manip,
        "Q8_n_SUCCESS_NEW": len(full_experts),
        "Q9_n_newly_vs_simple": sum(1 for r in full_experts if int(r.get("newly_vs_simple") or 0) == 1),
        "Q10_n_new_independent_successful_roots": len(new_hard_roots),
        "Q11_n_hard_dynamic_roots_in_unified": len(hist_ball_roots | new_hard_roots),
        "Q11_n_ball_like_roots_unified": len({
            r for r in all_roots
            if str(r).startswith(("bimp_", "ubimp", "hdimp_", "hard_"))
        }),
        "Q12_dominant_failures": dict(fail_reason.most_common(8)),
        "Q12_stage_distribution": dict(stage_ctr.most_common(10)),
        "Q13_transfer_wins": transfer_wins,
        "Q13_multistage_wins": multistage_wins,
        "Q13_longer_search_helped": multistage_wins > transfer_wins,
        "Q14_n_traj": len(unified),
        "Q14_n_ics": len(all_ics),
        "Q14_n_roots": len(all_roots),
        "Q15_gap_to_20": max(0, args.target - len(new_hard_roots)),
        "n_HIGH_MOTION_FULL": n_hm_full,
        "rollouts_used": rollout_counter[0],
        "rollout_budget": args.rollout_budget,
        "wall_s": time.time() - t0w,
        "wall_limit_s": args.wall_s,
        "new_hard_root_ids": sorted(new_hard_roots),
        "baseline_audit": {k: audit[k] for k in ("n_traj", "n_unique_ics", "n_unique_roots", "n_ball_impact_roots")},
    }
    write_json(OUT / "RAW" / "answers.json", answers)

    report = f"""# Hard Dynamic Ball-Impact Expansion

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**SUCCESS_NEW locked. No SFT/GRPO.**  
**Budget:** {args.rollout_budget} rollouts / {args.wall_s}s wall (used {rollout_counter[0]} / {answers['wall_s']:.1f}s)

## Baseline freeze (verified)

| Metric | Value |
|--------|------:|
| Trajectories | {audit['n_traj']} |
| Unique ICs | {audit['n_unique_ics']} |
| Unique roots | {audit['n_unique_roots']} |
| Ball-impact roots (historical) | {audit['n_ball_impact_roots']} |
| Bad token map | {audit['bad_token_map']} |
| Split leakage roots | {audit['split_leak_roots']} |

{audit['reconcile_note']}

## HIGH_MOTION thresholds (frozen before success)

- v_rel_norm ≥ **{thr['v_rel_norm_mps']:.4f} m/s**
- w_rel_norm ≥ **{thr['w_rel_norm_radps']:.4f} rad/s**
- v_lat ≥ **{thr['v_lat_mps']:.4f} m/s**
- Rule: {thr['rule']} (p{thr['percentile']} of pool n={thr['n_pool']})

## Yield

| Metric | Value |
|--------|------:|
| New HARD FULL experts | **{len(full_experts)}** |
| New independent hard roots | **{len(new_hard_roots)}** / target {args.target} |
| HIGH_MOTION FULL | {n_hm_full} |
| Recatch+Slide/Pivot | {n_recatch_manip} |
| Transfer-like wins | {transfer_wins} |
| Multi-stage/branch wins | {multistage_wins} |
| Gap to 20 | **{answers['Q15_gap_to_20']}** |

## Q1–Q15

1. Hard dynamic ICs available before batch: **{answers['Q1_n_hard_dynamic_ics_before']}**
2. From previously failed impact roots: **{answers['Q2_n_from_failed_impact_roots']}**
3. New independent impact episodes: **{answers['Q3_n_new_independent_impact_episodes']}** (attempts {answers['Q3_n_new_impact_attempts']})
4. Actually HIGH_MOTION (measured): **{answers['Q4_n_HIGH_MOTION_measured']}**
5. Partials restoring uni/bilateral (search stages C–F/OK): **{answers['Q5_n_restored_uni_or_bil_partials']}**
6. Stable grasp but pose still needed (E/F): **{answers['Q6_n_stable_but_pose_needed_partials']}**
7. Recatch + Sliding/Pivoting FULL: **{answers['Q7_n_recatch_plus_slide_pivot']}**
8. Complete SUCCESS_NEW: **{answers['Q8_n_SUCCESS_NEW']}**
9. Newly vs simple baselines: **{answers['Q9_n_newly_vs_simple']}**
10. New independent successful roots: **{answers['Q10_n_new_independent_successful_roots']}** {answers['new_hard_root_ids']}
11. Hard/ball-like dynamic roots in unified after dedup: **{answers['Q11_n_hard_dynamic_roots_in_unified']}** (ball-hist ∪ new)
12. Dominant failures: {answers['Q12_dominant_failures']} — stages {answers['Q12_stage_distribution']}
13. Longer multi-stage vs transfer: multistage={multistage_wins} transfer={transfer_wins} → helped={answers['Q13_longer_search_helped']}
14. Total dataset size: traj **{answers['Q14_n_traj']}**, ICs **{answers['Q14_n_ics']}**, roots **{answers['Q14_n_roots']}**
15. Remaining gap to 20-root target: **{answers['Q15_gap_to_20']}**

## Scripts
`scripts/expand_k49_hard_dynamic.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    ready = f"""# Dataset Readiness — Hard Dynamic Expansion

## Verdict: {"PARTIAL" if len(new_hard_roots) < args.target else "TARGET MET"}

- New nontrivial hard-dynamic SUCCESS_NEW roots: **{len(new_hard_roots)}** / {args.target}
- HIGH_MOTION FULL: {n_hm_full}
- Unified total roots after merge: **{len(all_roots)}**
- Easy STABLE_POSE_ERROR successes were **excluded** from the 20-root target count

### Merge
Append `FULL_EXPERTS.csv` into the unified canonical with `expert_type=HARD_DYNAMIC_FULL_EXPERT`.
Split by `root_episode`. Do not start SFT/GRPO here.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("\nDONE", json.dumps({k: answers[k] for k in answers if k not in ("new_hard_root_ids", "Q4_thresholds", "baseline_audit")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
