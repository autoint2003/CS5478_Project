"""Generate Type-3 Lost-Grasp Recovery experts under locked K49_tok48_wy0p25.

Type 3 = recovery from lost/unreliable grasp at Recovery onset (not eccentric-pivot).
No SFT/GRPO. SUCCESS_NEW unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_generation.recording.trajectory_recorder import save_pair
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
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

from scripts.generate_k49_dynamic_recatch_experts import (
    PIVOT_TOKS,
    SLIDE_HINT,
    find_bilateral_after_loss,
    has_post_manipulation,
    rich_state,
    run_traced,
    type2_proposals,
)
from scripts.generate_k49_in_recovery_recatch import (
    find_first_bilateral,
    load_short_handoff_ics,
    pose_correction_proposals,
    regrasp_proposals,
    snapshot_for,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    adaptive_return_variants,
    classify_mechanisms,
    decode,
    decompose,
    evaluate_full,
    hand_ref_error_deg,
    mapping_checksum,
    run_seq,
    write_csv,
    write_json,
)

OUT = ROOT / "results" / "expert_generation" / "k49_lost_grasp_recovery"
SEED = 20261019

# Known capture-brake / type2 seeds
CAPTURE_SEEDS = {
    "hrv1_920031": [[30, 12, 12, 8] + [0] * 7],
}
TYPE2_SEEDS_DIR = ROOT / "results/expert_generation/k49_dynamic_recatch_experts"


def root_episode_id(ic_id: str) -> str:
    """Collapse leftover aliases and branch suffixes to root episode."""
    s = ic_id.replace("hrv1_leftover_", "")
    # hard_910xxx / hrv1_920xxx
    if s.startswith("hard_") or s.startswith("hrv1_"):
        parts = s.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2]) if parts[0] == "hrv1" else "_".join(parts[:2])
    return s


def probe_secure(env, table, book48, blob, n_steps=10):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    z0 = float(measure(env.sim, U_TGT, P_TGT)["obj_z"])
    lost_low = 0
    bil_frac = 0
    from envs.outcomes.horizon_outcome import recovery_failed
    for i in range(n_steps):
        cs = contact_summary(env.sim)
        if cs["bilateral"]:
            bil_frac += 1
        action = decode(table, book48, 0)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "failed": True,
                "steps": i + 1,
                "bil_frac": bil_frac / max(1, i + 1),
                "dz": float(measure(env.sim, U_TGT, P_TGT)["obj_z"]) - z0,
            }
    dz = float(measure(env.sim, U_TGT, P_TGT)["obj_z"]) - z0
    return {
        "failed": False,
        "steps": n_steps,
        "bil_frac": bil_frac / n_steps,
        "dz": dz,
    }


def classify_ic(env, table, book48, blob):
    restore_capture(env, blob["capture"])
    cs = contact_summary(env.sim)
    m = measure(env.sim, U_TGT, P_TGT)
    d = decompose(env.sim)
    nL, nR = int(cs["nL"]), int(cs["nR"])
    bil = bool(cs["bilateral"])
    v = float(np.linalg.norm(m["v_rel_h"]))
    w = float(np.linalg.norm(m["w_rel_h"]))
    probe = probe_secure(env, table, book48, blob, n_steps=12)

    if probe["failed"] and (nL + nR) == 0:
        # already lost and escapes under secure — may still be recoverable with chase
        cls = "COMPLETE_CONTACT_LOSS"
        if abs(probe["dz"]) > 0.08:
            cls = "IRREVERSIBLE_ESCAPE"
    elif (nL + nR) == 0:
        cls = "COMPLETE_CONTACT_LOSS"
    elif bil and (probe["failed"] or probe["bil_frac"] < 0.5 or v > 0.25 or w > 8.0 or d["e_pos_mm"] > 20):
        cls = "UNSTABLE_BILATERAL"
    elif (not bil) and (nL + nR) > 0:
        cls = "UNILATERAL_LOSS"
    elif bil and not probe["failed"] and probe["bil_frac"] >= 0.75 and v < 0.15 and w < 5.0:
        cls = "STABLE_GRASP"  # not Type3 onset
    else:
        # borderline bilateral with moderate motion
        cls = "UNSTABLE_BILATERAL" if (v > 0.12 or w > 4.0 or d["e_pos_mm"] > 14) else "STABLE_GRASP"

    # irreversible if already far below / failed quickly with no contact
    if probe["failed"] and probe["steps"] <= 3 and (nL + nR) == 0 and abs(probe.get("dz", 0)) > 0.05:
        cls = "IRREVERSIBLE_ESCAPE"

    return {
        "lost_class": cls,
        "is_type3_onset": cls in ("UNILATERAL_LOSS", "UNSTABLE_BILATERAL", "COMPLETE_CONTACT_LOSS"),
        "nL": nL, "nR": nR, "bilateral": bil,
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_ax_mm": float(d["e_ax_mm"]),
        "e_lat_mm": float(d["e_lat_mm"]),
        "v_rel_norm": v, "w_rel_norm": w,
        "v_rel_h": np.asarray(m["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(m["w_rel_h"], float).tolist(),
        "probe_failed": int(probe["failed"]),
        "probe_bil_frac": probe["bil_frac"],
        "probe_dz": probe["dz"],
        "obj_z": float(m["obj_z"]),
    }


def load_type2_seed_tokens():
    p = TYPE2_SEEDS_DIR / "TYPE2_EXPERTS.csv"
    out = defaultdict(list)
    if not p.exists():
        return out
    for r in csv.DictReader(p.open(encoding="utf-8")):
        try:
            toks = json.loads(r["tokens"])
        except Exception:
            continue
        out[r["ic_id"]].append(toks)
    return out


def stage_a_proposals(book, desire, rng, n=28, seeds=None):
    seqs = []
    if seeds:
        for s in seeds:
            seqs.append(list(s))
            # local variants
            seqs.append(list(s) + [0] * 8)
            seqs.append(list(s)[:4] + [0] * 16)
            seqs.append([12] * 3 + list(s) + [0] * 5)
    seqs += regrasp_proposals(book, desire, rng, n=max(8, n // 2))
    # capture-brake style
    seqs.append([30, 12, 12, 8] + [0] * 10)
    seqs.append([30, 12, 12, 8, 17] + [0] * 10)
    seqs.append([0] * 20)
    seqs.append([12] * 5 + [0] * 16)
    seqs.append([36] * 3 + [0] * 16)
    seqs.append([30] + [17] * 8 + [0] * 12)
    seqs.append([30] + [33] * 8 + [0] * 12)
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    head, tail = uniq[:10], uniq[10:]
    rng.shuffle(tail)
    return (head + tail)[:n]


def stage_b_proposals(book, desire, rng, n=16):
    return pose_correction_proposals(book, desire, rng, n=n)


def sublabel_from_trace(trace, bil_i):
    if bil_i is None:
        return "TYPE3_RECATCH_FAILED"
    post, info = has_post_manipulation(trace, bil_i)
    piv = info.get("post_pivot_n", 0)
    sld = info.get("post_slide_n", 0)
    if piv >= 8 and sld >= 8:
        return "TYPE3_RECATCH_SLIDING_PIVOTING"
    if piv >= 8:
        return "TYPE3_RECATCH_PIVOTING"
    if sld >= 8:
        return "TYPE3_RECATCH_SLIDING"
    return "TYPE3_RECATCH_ONLY"


def save_expert(env, ic_id, tag, res, mech, mapping_meta, checksum, snap, extra):
    stem = f"{ic_id}__{tag}"
    save_pair(
        OUT / "trajectories", stem,
        res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    meta = {
        "ic_id": ic_id, "tag": tag, "family": "type3_lost_grasp",
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "SUCCESS_NEW": res["SUCCESS_NEW"], "label": res["label"],
        "window": res["window"], "tokens": res["tokens"],
        "snapshot_src": str(snap), "obs_dim": 39, "handoff_id": 47,
        **extra,
    }
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    if res.get("trace"):
        write_json(OUT / "raw_contact_traces" / f"{stem}_trace.json", {
            "trace": res["trace"], "mech": mech, "extra": extra,
        })
    return meta


def mine_lost_grasp_ics(env, table, book48, max_scan=220, max_keep=24):
    pools = []
    for d in [
        ROOT / "results/benchmarks/hard_recovery_v1/snapshots",
        ROOT / "results/diagnostics/hard_ic_mining/snapshots",
        ROOT / "results/expert_generation/k49_in_recovery_recatch/snapshots",
        ROOT / "results/expert_generation/k49_dynamic_recatch_experts/snapshots",
    ]:
        if d.exists():
            pools.extend(sorted(d.glob("*.pkl")))

    seen_root = set()
    scored = []
    scanned = 0
    for sp in pools:
        if scanned >= max_scan:
            break
        ic = sp.stem.replace("hrv1_leftover_", "")
        root = root_episode_id(ic)
        if root in seen_root and ic not in ("hrv1_920031", "hard_910101", "hard_910318", "hrv1_920099"):
            # allow a few known hard names even if root seen via leftover
            pass
        try:
            blob = pickle.loads(sp.read_bytes())
            if "ref" not in blob or "capture" not in blob:
                continue
            info = classify_ic(env, table, book48, blob)
        except Exception:
            continue
        scanned += 1
        if not info["is_type3_onset"]:
            continue
        if info["lost_class"] == "IRREVERSIBLE_ESCAPE":
            continue
        # score: prefer loss + nontrivial pose + recoverable-ish (not huge v)
        # Prefer unilateral/unstable that are still near the hand; demote far complete loss.
        score = 0.0
        score += {
            "UNILATERAL_LOSS": 45,
            "UNSTABLE_BILATERAL": 40,
            "COMPLETE_CONTACT_LOSS": 15,
        }[info["lost_class"]]
        score += min(info["e_theta_deg"], 40) * 0.4
        # pose error helps nontriviality but large pos often means escaped
        if info["e_pos_mm"] <= 25:
            score += info["e_pos_mm"] * 0.4
        else:
            score -= (info["e_pos_mm"] - 25) * 0.8
        score += min(info["v_rel_norm"], 0.6) * 8
        if info["probe_failed"] and info["lost_class"] != "COMPLETE_CONTACT_LOSS":
            score += 8
        if info["v_rel_norm"] > 1.0 or info["obj_z"] < 0.06 or info["e_pos_mm"] > 45:
            score -= 50  # likely irreversible / unreachable
        if info["lost_class"] == "COMPLETE_CONTACT_LOSS" and info["e_pos_mm"] > 28:
            continue  # skip far free-flight snapshots
        scored.append({
            "ic_id": ic, "root_episode": root, "snapshot": str(sp),
            "score": score, **info,
        })

    scored.sort(key=lambda x: -x["score"])
    # diversify by root
    picked, roots = [], set()
    for e in scored:
        if e["root_episode"] in roots:
            continue
        roots.add(e["root_episode"])
        picked.append(e)
        if len(picked) >= max_keep:
            break
    return picked, scanned, scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-a", type=int, default=22)
    ap.add_argument("--budget-b", type=int, default=12)
    ap.add_argument("--max-ics", type=int, default=16)
    ap.add_argument("--max-scan", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("trajectories", "snapshots", "raw_contact_traces", "RAW"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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

    type2_seeds = load_type2_seed_tokens()
    short_ics = {e["ic_id"]: e for e in load_short_handoff_ics()}

    print("Mining lost-grasp ICs...", flush=True)
    picked, n_scan, all_scored = mine_lost_grasp_ics(
        env, table, book48, max_scan=args.max_scan, max_keep=args.max_ics
    )
    # always include 031 for classification check + known seed
    forced = []
    for force_id in ("hrv1_920031", "hard_910318", "hrv1_920099", "hard_910101", "hard_910220"):
        if any(e["ic_id"] == force_id for e in picked):
            continue
        sp = snapshot_for(force_id)
        if sp is None:
            continue
        blob = pickle.loads(sp.read_bytes())
        info = classify_ic(env, table, book48, blob)
        forced.append({
            "ic_id": force_id, "root_episode": root_episode_id(force_id),
            "snapshot": str(sp), "score": -1, **info,
        })
    # put forced first so smoke/truncate retains known seeds + hard cases
    picked = forced + [e for e in picked if e["ic_id"] not in {f["ic_id"] for f in forced}]
    if args.smoke:
        picked = picked[:6]

    write_json(OUT / "RAW" / "mine_all_scored.json", all_scored[:80])
    print(f"SELECTED {len(picked)} (scanned {n_scan})", flush=True)

    ic_rows = []
    baselines = []
    prefixes = []
    post_corr = []
    full_experts = []
    failures = []
    coverage = []
    manifest = []
    simple_ok = []
    n_rollouts = 0

    for ie, ic in enumerate(picked):
        t0b = time.time()
        ic_id = ic["ic_id"]
        snap = Path(ic["snapshot"])
        blob = pickle.loads(snap.read_bytes())
        shutil.copy2(snap, OUT / "snapshots" / f"{ic_id}.pkl")
        root_ep = ic.get("root_episode") or root_episode_id(ic_id)
        info = {k: ic[k] for k in ic if k not in ("snapshot",)}
        # refresh classification
        info.update(classify_ic(env, table, book48, blob))
        info["root_episode"] = root_ep
        ic_rows.append({
            "ic_id": ic_id, "root_episode": root_ep,
            "lost_class": info["lost_class"], "is_type3_onset": int(info["is_type3_onset"]),
            "nL": info["nL"], "nR": info["nR"], "bilateral": int(info["bilateral"]),
            "e_theta0": info["e_theta_deg"], "e_pos0": info["e_pos_mm"],
            "v_rel": info["v_rel_norm"], "w_rel": info["w_rel_norm"],
            "probe_failed": info["probe_failed"], "probe_bil_frac": info["probe_bil_frac"],
            "snapshot": str(snap),
        })
        print(f"\n[{ie+1}/{len(picked)}] {ic_id} class={info['lost_class']} "
              f"th={info['e_theta_deg']:.1f} pos={info['e_pos_mm']:.1f} "
              f"L/R={info['nL']}/{info['nR']} bil={info['bilateral']}", flush=True)

        if not info["is_type3_onset"]:
            print("  skip: not Type3 onset", flush=True)
            coverage.append({
                "ic_id": ic_id, "root_episode": root_episode_id(ic_id),
                "lost_class": info["lost_class"], "searched": 0,
                "stage_a": 0, "full": 0, "skip": "not_type3_onset",
                "budget_s": time.time() - t0b,
            })
            continue

        # ----- matched baselines -----
        n_try = 0
        # A immediate HANDOFF
        res_a = evaluate_full(env, table, book48, blob, [], record_obs=False)
        n_try += 1
        # B immediate secure
        res_b = evaluate_full(env, table, book48, blob, [0] * 16, record_obs=False)
        n_try += 1
        # C short release-close
        pref = short_ics.get(ic_id, {}).get("prefix") or [30]
        simple = list(pref) + [0] * 8
        res_c = evaluate_full(env, table, book48, blob, simple, record_obs=False)
        n_try += 1
        # D known capture-brake if any
        res_d = None
        d_toks = None
        for s in CAPTURE_SEEDS.get(ic_id, []) + type2_seeds.get(ic_id, [])[:1]:
            d_toks = list(s)
            res_d = evaluate_full(env, table, book48, blob, d_toks, record_obs=False)
            n_try += 1
            break
        if res_d is None:
            # generic capture-brake seed
            d_toks = [30, 12, 12, 8] + [0] * 7
            res_d = evaluate_full(env, table, book48, blob, d_toks, record_obs=False)
            n_try += 1

        base_row = {
            "ic_id": ic_id, "lost_class": info["lost_class"],
            "e_theta0": info["e_theta_deg"], "e_pos0": info["e_pos_mm"],
            "A_NEW": int(res_a["SUCCESS_NEW"]), "A_s12": int(res_a["success_12s"]),
            "A_reason": res_a["reason"],
            "B_NEW": int(res_b["SUCCESS_NEW"]), "B_s12": int(res_b["success_12s"]),
            "B_reason": res_b["reason"],
            "C_NEW": int(res_c["SUCCESS_NEW"]), "C_s12": int(res_c["success_12s"]),
            "C_reason": res_c["reason"],
            "D_NEW": int(res_d["SUCCESS_NEW"]), "D_s12": int(res_d["success_12s"]),
            "D_reason": res_d["reason"],
            "D_tokens": json.dumps(d_toks),
        }
        baselines.append(base_row)

        simple_solves = res_a["SUCCESS_NEW"] or res_b["SUCCESS_NEW"] or res_c["SUCCESS_NEW"]
        if simple_solves:
            simple_ok.append({"ic_id": ic_id, "how": (
                "A" if res_a["SUCCESS_NEW"] else ("B" if res_b["SUCCESS_NEW"] else "C")
            )})
            print(f"  simple baseline already NEW — keep as SIMPLE, still try Type3 search lightly", flush=True)

        # If D already NEW and onset is lost-grasp, this IS a Type3 expert candidate
        best_full = None
        if res_d["SUCCESS_NEW"] and info["is_type3_onset"]:
            ok, tr = run_traced(env, table, book48, blob, d_toks)
            n_try += 1
            bil_i = find_first_bilateral(tr, 0)
            if bil_i is None:
                bil_i = find_bilateral_after_loss(tr)
            if bil_i is not None:
                sub = sublabel_from_trace(tr, bil_i)
                res = evaluate_full(env, table, book48, blob, d_toks, record_obs=True)
                n_try += 1
                mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                extra = {
                    "expert_type": "TYPE3_LOST_GRASP_FULL_EXPERT",
                    "sublabel": sub,
                    "lost_class": info["lost_class"],
                    "root_episode": root_episode_id(ic_id),
                    "source": "capture_brake_or_type2_seed",
                    "onset": info,
                }
                save_expert(env, ic_id, f"seed_{sub}", res, mech, mapping_meta, checksum, snap, extra)
                row = {
                    "ic_id": ic_id, "tag": f"seed_{sub}", "root_episode": root_episode_id(ic_id),
                    "expert_type": "TYPE3_LOST_GRASP_FULL_EXPERT", "sublabel": sub,
                    "lost_class": info["lost_class"], "SUCCESS_NEW": 1,
                    "win_mean_pos": res["window"].get("mean_pos"),
                    "win_mean_theta": res["window"].get("mean_theta"),
                    "hold_both_s": res["window"].get("longest_both_ok_s"),
                    "bilateral_i": bil_i, "n_tokens": len(d_toks),
                    "tokens": json.dumps(d_toks), "config_id": CONFIG_ID,
                    "mapping_checksum": checksum, "newly_vs_simple": int(not simple_solves),
                    "reproducible": 1,
                }
                full_experts.append(row)
                manifest.append(row)
                best_full = row
                print(f"  SEED FULL {sub} NEW=1", flush=True)

        # ----- Stage A search -----
        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]
        seeds = list(CAPTURE_SEEDS.get(ic_id, []))
        seeds += type2_seeds.get(ic_id, [])[:2]
        props_a = stage_a_proposals(book48, desire, rng, n=args.budget_a, seeds=seeds)
        # reduce budget if simple already solves and we already have full
        if best_full is not None and simple_solves:
            props_a = props_a[:4]
        elif best_full is not None:
            props_a = props_a[:8]

        stage_a_hit = None
        print(f"  STAGE_A n={len(props_a)}", flush=True)
        for ia, seq in enumerate(props_a):
            if best_full is not None and ia > 6:
                break
            n_try += 1
            n_rollouts += 1
            ok, tr = run_traced(env, table, book48, blob, seq)
            bil_i = find_first_bilateral(tr, 0)
            if bil_i is None:
                bil_i = find_bilateral_after_loss(tr)
            if bil_i is None or not ok:
                failures.append({
                    "ic_id": ic_id, "tag": f"a_{ia}", "stage": "A",
                    "reason": "NO_BILATERAL" if ok else "DROP",
                })
                continue
            # controllable: bilateral run + reduced speed
            after = tr[bil_i: min(len(tr), bil_i + 8)]
            bil_run = sum(1 for r in after if r["bilateral"])
            w_bil = float(np.linalg.norm(tr[bil_i]["w_rel_h"]))
            prefixes.append({
                "ic_id": ic_id, "tag": f"a_{ia}", "bilateral_i": bil_i,
                "bil_run": bil_run, "w_at_bil": w_bil,
                "theta_at_bil": tr[bil_i]["e_theta_deg"],
                "pos_at_bil": tr[bil_i]["e_pos_mm"],
                "n_tokens": len(seq), "tokens": json.dumps(seq),
            })
            if bil_run < 3:
                continue
            stage_a_hit = {"seq": seq, "bil_i": bil_i, "tr": tr, "ia": ia}
            # quick full eval of stage A alone
            for settle in ([0] * 3, [0] * 12, [0] * 24):
                toks = list(seq) + list(settle)
                n_try += 1
                res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
                if res["SUCCESS_NEW"] and info["is_type3_onset"]:
                    ok2, tr2 = run_traced(env, table, book48, blob, toks)
                    bil2 = find_first_bilateral(tr2, 0) or find_bilateral_after_loss(tr2)
                    sub = sublabel_from_trace(tr2, bil2)
                    res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
                    mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                    extra = {
                        "expert_type": "TYPE3_LOST_GRASP_FULL_EXPERT",
                        "sublabel": sub, "lost_class": info["lost_class"],
                        "root_episode": root_episode_id(ic_id), "source": "stage_a",
                    }
                    save_expert(env, ic_id, f"a{ia}_{sub}", res, mech, mapping_meta, checksum, snap, extra)
                    row = {
                        "ic_id": ic_id, "tag": f"a{ia}_{sub}", "root_episode": root_episode_id(ic_id),
                        "expert_type": "TYPE3_LOST_GRASP_FULL_EXPERT", "sublabel": sub,
                        "lost_class": info["lost_class"], "SUCCESS_NEW": 1,
                        "win_mean_pos": res["window"].get("mean_pos"),
                        "win_mean_theta": res["window"].get("mean_theta"),
                        "hold_both_s": res["window"].get("longest_both_ok_s"),
                        "bilateral_i": bil2, "n_tokens": len(toks),
                        "tokens": json.dumps(toks), "config_id": CONFIG_ID,
                        "mapping_checksum": checksum,
                        "newly_vs_simple": int(not simple_solves), "reproducible": 1,
                    }
                    full_experts.append(row)
                    manifest.append(row)
                    best_full = row
                    print(f"  STAGE_A FULL {sub}", flush=True)
                    break
            if best_full is not None:
                break

            # ----- Stage B from this prefix -----
            restore_capture(env, blob["capture"])
            run_seq(env, table, book48, blob, seq)
            desire_b = -decompose(env.sim)["err_h"]
            props_b = stage_b_proposals(book48, desire_b, rng, n=args.budget_b)
            print(f"    STAGE_B after a_{ia} n={len(props_b)} th={tr[bil_i]['e_theta_deg']:.1f} "
                  f"pos={tr[bil_i]['e_pos_mm']:.1f}", flush=True)
            for ib, corr in enumerate(props_b):
                n_try += 1
                n_rollouts += 1
                base = list(seq) + list(corr)
                variants = [("raw", base + [0] * 3)]
                restore_capture(env, blob["capture"])
                run_seq(env, table, book48, blob, base)
                if hand_ref_error_deg(env, blob) > 8:
                    variants += adaptive_return_variants(env, table, book48, blob, float(env.sim.data.time), base)
                    variants.append(("ret12", base + [SECURE_REVERSE_WRIST] * 12 + [0] * 5))
                for vname, toks in variants:
                    n_try += 1
                    res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
                    ok2, tr2 = run_traced(env, table, book48, blob, toks)
                    bil2 = find_first_bilateral(tr2, 0) or find_bilateral_after_loss(tr2)
                    sub = sublabel_from_trace(tr2, bil2)
                    post_corr.append({
                        "ic_id": ic_id, "a_tag": f"a_{ia}", "b_tag": f"b_{ib}_{vname}",
                        "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
                        "s12": int(res["success_12s"]),
                        "win_th": res["window"].get("mean_theta"),
                        "win_pos": res["window"].get("mean_pos"),
                        "sublabel": sub, "reason": res["reason"],
                        "n_tokens": len(toks),
                    })
                    if not res["SUCCESS_NEW"]:
                        failures.append({
                            "ic_id": ic_id, "tag": f"a{ia}_b{ib}_{vname}", "stage": "B",
                            "reason": res["reason"], "s12": int(res["success_12s"]),
                            "win_th": res["window"].get("mean_theta"),
                            "win_pos": res["window"].get("mean_pos"),
                        })
                        continue
                    if bil2 is None or not info["is_type3_onset"]:
                        continue
                    res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
                    mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                    extra = {
                        "expert_type": "TYPE3_LOST_GRASP_FULL_EXPERT",
                        "sublabel": sub, "lost_class": info["lost_class"],
                        "root_episode": root_episode_id(ic_id), "source": "stage_a_b",
                    }
                    save_expert(env, ic_id, f"a{ia}b{ib}_{vname}_{sub}", res, mech,
                                mapping_meta, checksum, snap, extra)
                    row = {
                        "ic_id": ic_id, "tag": f"a{ia}b{ib}_{vname}_{sub}",
                        "root_episode": root_episode_id(ic_id),
                        "expert_type": "TYPE3_LOST_GRASP_FULL_EXPERT", "sublabel": sub,
                        "lost_class": info["lost_class"], "SUCCESS_NEW": 1,
                        "win_mean_pos": res["window"].get("mean_pos"),
                        "win_mean_theta": res["window"].get("mean_theta"),
                        "hold_both_s": res["window"].get("longest_both_ok_s"),
                        "bilateral_i": bil2, "n_tokens": len(toks),
                        "tokens": json.dumps(toks), "config_id": CONFIG_ID,
                        "mapping_checksum": checksum,
                        "newly_vs_simple": int(not simple_solves), "reproducible": 1,
                    }
                    full_experts.append(row)
                    manifest.append(row)
                    best_full = row
                    print(f"  STAGE_B FULL {sub}", flush=True)
                    break
                if best_full is not None:
                    break
            if best_full is not None:
                break
            # one stage-A hit with B search is enough per IC before next A
            if stage_a_hit is not None and ia >= 4 and best_full is None:
                # continue a few more A
                pass

        # update baseline with Type3 result
        base_row["T3_NEW"] = int(best_full is not None)
        base_row["T3_sublabel"] = None if not best_full else best_full["sublabel"]
        base_row["T3_win_th"] = None if not best_full else best_full["win_mean_theta"]
        base_row["T3_win_pos"] = None if not best_full else best_full["win_mean_pos"]
        base_row["best"] = (
            "T3" if best_full else (
                "D" if res_d["SUCCESS_NEW"] else (
                    "C" if res_c["SUCCESS_NEW"] else (
                        "B" if res_b["SUCCESS_NEW"] else (
                            "A" if res_a["SUCCESS_NEW"] else "NONE"))))
        )

        coverage.append({
            "ic_id": ic_id, "root_episode": root_episode_id(ic_id),
            "lost_class": info["lost_class"], "searched": n_try,
            "stage_a": int(stage_a_hit is not None), "full": int(best_full is not None),
            "simple_baseline_new": int(simple_solves),
            "newly_vs_simple": int(best_full is not None and not simple_solves),
            "budget_s": time.time() - t0b,
        })
        print(f"  DONE full={best_full is not None} tries={n_try} ({time.time()-t0b:.1f}s)", flush=True)

    # ----- outputs -----
    write_csv(OUT / "LOST_GRASP_ICS.csv", ic_rows)
    write_csv(OUT / "BASELINE_COMPARISON.csv", baselines)
    write_csv(OUT / "RECATCH_PREFIXES.csv", prefixes)
    write_csv(OUT / "POST_RECATCH_CORRECTIONS.csv", post_corr)
    write_csv(OUT / "TYPE3_FULL_EXPERTS.csv", full_experts)
    write_csv(OUT / "FAILURES.csv", failures[:1200])
    write_csv(OUT / "ROOT_EPISODE_COVERAGE.csv", coverage)
    write_csv(OUT / "UNIFIED_MANIFEST.csv", manifest)
    write_csv(OUT / "SIMPLE_BASELINE_OK.csv", simple_ok)
    write_json(OUT / "RAW" / "n_rollouts.json", {"n_rollouts": n_rollouts, "wall_s": time.time() - t0w})

    # counts
    roots_lost = sorted({r["root_episode"] for r in ic_rows if int(r.get("is_type3_onset") or 0) == 1})
    n_bil_restore = len({p["ic_id"] for p in prefixes})
    need_corr = [p for p in prefixes if float(p["theta_at_bil"]) > 10 or float(p["pos_at_bil"]) > 12]
    n_full = len({r["ic_id"] for r in full_experts})
    n_new = sum(1 for r in full_experts if int(r.get("newly_vs_simple") or 0) == 1)
    n_new_ics = len({r["ic_id"] for r in full_experts if int(r.get("newly_vs_simple") or 0) == 1})
    sub_ctr = Counter(r["sublabel"] for r in full_experts)

    # prior type1/2 coverage estimate
    n1 = 0
    p1 = ROOT / "results/expert_generation/k49_sliding_recatch_dataset/UNIFIED_MANIFEST.csv"
    if p1.exists():
        n1 = len({r["ic_id"] for r in csv.DictReader(p1.open(encoding="utf-8"))})
    n2 = 0
    p2 = ROOT / "results/expert_generation/k49_dynamic_recatch_experts/TYPE2_EXPERTS.csv"
    if p2.exists():
        n2 = len({r["ic_id"] for r in csv.DictReader(p2.open(encoding="utf-8"))})

    # failure reasons among non-full type3-onset
    unsolved = [c for c in coverage if int(c.get("full") or 0) == 0 and c.get("skip") != "not_type3_onset"]
    fail_why = Counter()
    for c in unsolved:
        ic = c["ic_id"]
        fs = [f for f in failures if f["ic_id"] == ic]
        if not fs:
            fail_why["no_stage_a_or_budget"] += 1
        else:
            fail_why[fs[-1].get("reason", "unknown")] += 1

    answers = {
        "Q1_n_independent_lost_grasp_roots": len(roots_lost),
        "Q1_root_ids": roots_lost,
        "Q2_n_ics_bilateral_restored": n_bil_restore,
        "Q3_n_stable_recatch_need_slide_pivot": len({p["ic_id"] for p in need_corr}),
        "Q4_n_type3_full_ics": n_full,
        "Q4_sublabels": dict(sub_ctr),
        "Q5_n_newly_solved_vs_simple": n_new_ics,
        "Q5_newly_ids": sorted({r["ic_id"] for r in full_experts if int(r.get("newly_vs_simple") or 0) == 1}),
        "Q6_unsolved_n": len(unsolved),
        "Q6_failure_reasons": dict(fail_why.most_common(12)),
        "Q7_ready_unified_SFT": bool(n1 >= 3 and n2 >= 2 and n_full >= 3),
        "Q7_counts": {"type1_ics_approx": n1, "type2_ics": n2, "type3_lost_grasp_ics": n_full},
        "n_ics_searched": len([c for c in coverage if c.get("skip") != "not_type3_onset"]),
        "n_scan": n_scan,
        "wall_s": time.time() - t0w,
        "n_rollouts": n_rollouts,
    }
    if n_full < 3:
        answers["Q7_ready_unified_SFT"] = False
        answers["Q7_note"] = (
            f"Type1≈{n1}, Type2={n2}, Type3-lost-grasp={n_full}. "
            "Want ≥3 independent Type3 lost-grasp FULL experts before unified SFT."
        )
    else:
        answers["Q7_note"] = (
            f"Type1≈{n1}, Type2={n2}, Type3-lost-grasp={n_full} independent ICs — "
            "mechanism diversity present; still verify episode-grouped splits."
        )
    write_json(OUT / "RAW" / "answers.json", answers)

    report = f"""# Type 3 Lost-Grasp Recovery Experts

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**SUCCESS_NEW:** locked 10° / 12 mm / ≥0.5 s hold  
**Type 3 redefined:** lost/unreliable grasp at Recovery onset (eccentric/Δθ not required).  
**No SFT/GRPO.**

## Yield

| Metric | Value |
|--------|------:|
| Independent lost-grasp root episodes (mined/selected) | {len(roots_lost)} |
| ICs with Stage-A bilateral restore | {n_bil_restore} |
| TYPE3_LOST_GRASP_FULL_EXPERT ICs | **{n_full}** |
| Newly solved vs simple A/B/C baselines | **{n_new_ics}** |
| Sublabels | {dict(sub_ctr)} |

## Q1–Q7

1. **Independent lost-grasp roots:** **{len(roots_lost)}** — {roots_lost}

2. **ICs restored to stable bilateral (Stage A prefixes):** **{n_bil_restore}**

3. **Stable Recatch still needing pose correction (θ>10 or pos>12 at bil):** **{len({p['ic_id'] for p in need_corr})}**

4. **Complete SUCCESS_NEW Type3 experts:** **{n_full}**  
   Sublabels: {dict(sub_ctr)}

5. **Newly solved vs simple Recatch/grip/HANDOFF:** **{n_new_ics}** {answers['Q5_newly_ids']}

6. **Remaining failures:** {answers['Q6_failure_reasons']}  
   Unsolved searched ICs: {len(unsolved)}

7. **Ready for unified Transformer SFT?** **{answers['Q7_ready_unified_SFT']}** — {answers['Q7_note']}

## Note on `hrv1_920031`

Classification of `[30,12,12,8,0×7]` depends on onset `lost_class` in `LOST_GRASP_ICS.csv`.  
If onset is `STABLE_GRASP`, it is **not** labeled Type3 even if SUCCESS_NEW.

## Artifacts

`LOST_GRASP_ICS.csv`, `BASELINE_COMPARISON.csv`, `RECATCH_PREFIXES.csv`,  
`POST_RECATCH_CORRECTIONS.csv`, `TYPE3_FULL_EXPERTS.csv`, `FAILURES.csv`,  
`ROOT_EPISODE_COVERAGE.csv`, `UNIFIED_MANIFEST.csv`, `DATASET_READINESS.md`,  
`trajectories/`, `snapshots/`, `raw_contact_traces/`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    ready = f"""# Dataset Readiness — Lost-Grasp Type3

## Verdict: {"YES (conditional)" if answers["Q7_ready_unified_SFT"] else "NOT YET"}

- Type1 (sliding/recatch pool) ICs ≈ {n1}
- Type2 ICs = {n2}
- Type3 lost-grasp FULL ICs = {n_full}
- Newly solved vs simple baselines = {n_new_ics}

### Merge rule
Append `TYPE3_FULL_EXPERTS.csv` / `UNIFIED_MANIFEST.csv` rows with `expert_type=TYPE3_LOST_GRASP_FULL_EXPERT`.
Split by `root_episode`. Do not treat branch snapshots as new roots.
Keep onset `lost_class` as diagnostic metadata only (not in 39D obs).
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("\nDONE", answers, flush=True)


if __name__ == "__main__":
    main()
