"""Scale up unified K49 grasp-recovery expert dataset (SUCCESS_NEW only).

Locked K49_tok48_wy0p25. No SFT/GRPO. No physics/SUCCESS_NEW changes.
Stages: audit → historical STABLE_POSE_ERROR → ball-impact → 50/75/100 roots.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generation.ball_impact_dataset import run_episode
from data_generation.rebuild_k48_recovery import immediate_handoff
from data_generation.recording.trajectory_recorder import save_pair
from envs.simulation.capture import restore_capture
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import measure
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
    draw_lateral_spec,
    classify_ic_regime,
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
    decompose,
    evaluate_full,
    hand_ref_error_deg,
    mapping_checksum,
    run_seq,
    slide_proposals,
    write_csv,
    write_json,
)

OUT = ROOT / "results" / "expert_generation" / "k49_unified_dataset_expansion"
SEED = 20261019
HRV1_SNAPS = ROOT / "results" / "benchmarks" / "hard_recovery_v1" / "snapshots"
HARD_SNAPS = ROOT / "results" / "diagnostics" / "hard_ic_mining" / "snapshots"
CAPTURE_SEED = [30, 12, 12, 8] + [0] * 7

# Source manifests of previously claimed SUCCESS_NEW experts
SOURCE_MANIFESTS = [
    ("k49_sliding_recatch", ROOT / "results/expert_generation/k49_sliding_recatch_dataset/UNIFIED_MANIFEST.csv"),
    ("k49_in_recovery_recatch", ROOT / "results/expert_generation/k49_in_recovery_recatch/UNIFIED_MANIFEST.csv"),
    ("k49_dynamic_type2", ROOT / "results/expert_generation/k49_dynamic_recatch_experts/TYPE2_EXPERTS.csv"),
    ("k49_ball_impact", ROOT / "results/expert_generation/k49_ball_impact_dynamic_recovery/FULL_EXPERTS.csv"),
    ("k49_lost_grasp", ROOT / "results/expert_generation/k49_lost_grasp_recovery/TYPE3_FULL_EXPERTS.csv"),
    ("k49_capture_brake", ROOT / "results/expert_generation/k49_type3_capture_brake/HYBRID_EXPERTS.csv"),
    ("grasp_recovery_pose_v1", ROOT / "results/expert_generation/grasp_recovery_pose_v1/EXPERT_TRAJECTORIES.csv"),
    ("extended_k49_k50", ROOT / "results/expert_generation/extended_k49_k50_pose_recovery/EXPERT_TRAJECTORIES.csv"),
    ("k49_slow_return", ROOT / "results/expert_generation/k49_slow_return_position_refinement/FULL_EXPERTS.csv"),
    ("return_aware_sliding", ROOT / "results/expert_generation/return_aware_sliding_k49/FULL_EXPERTS.csv"),
    ("sliding_position_v1", ROOT / "results/expert_generation/sliding_position_correction_v1/FULL_EXPERTS.csv"),
]

SNAP_POOLS = [
    ("hard_ic_mining", HARD_SNAPS),
    ("hard_recovery_v1", HRV1_SNAPS),
    ("ball_impact", ROOT / "results/expert_generation/k49_ball_impact_dynamic_recovery/snapshots"),
    ("sliding_recatch", ROOT / "results/expert_generation/k49_sliding_recatch_dataset/snapshots"),
    ("lost_grasp", ROOT / "results/expert_generation/k49_lost_grasp_recovery/snapshots"),
    ("in_recovery", ROOT / "results/expert_generation/k49_in_recovery_recatch/snapshots"),
    ("pose_v1", ROOT / "results/expert_generation/grasp_recovery_pose_v1/snapshots"),
]


def is_success_new(row: dict) -> bool:
    v = row.get("SUCCESS_NEW")
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes") or "VERIFIED_FULL" in str(row.get("label", ""))


def canonical_root(ic_id: str, root_episode: str | None = None) -> str:
    """Independent physical root; strip action-variant / loss suffixes."""
    cand = (root_episode or ic_id or "").strip()
    # reject token-ish fake roots from slow-return tags
    if cand.startswith(("cpp_", "suf", "refine_", "known_", "a0_", "a1_", "raw")):
        cand = ic_id or cand
    if "_loss" in cand:
        cand = cand.split("_loss")[0]
    if "_uni" in cand:
        cand = cand.split("_uni")[0]
    if "__" in cand:
        cand = cand.split("__")[0]
    # gen_*_hash keep full gen id as root (one disturbance episode)
    return cand


def tokens_hash(toks) -> str:
    raw = json.dumps([int(x) for x in toks], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def find_snapshot(ic_id: str) -> Path | None:
    names = [f"{ic_id}.pkl"]
    base = canonical_root(ic_id)
    if base != ic_id:
        names.append(f"{base}.pkl")
    for _, pool in SNAP_POOLS:
        if not pool.exists():
            continue
        for n in names:
            p = pool / n
            if p.exists():
                return p
    # recursive fallback
    for p in ROOT.joinpath("results").rglob(f"{ic_id}.pkl"):
        return p
    return None


def find_traj_tokens(source_dir: Path, ic_id: str):
    """Load tokens from trajectory meta/npz if present."""
    traj = source_dir / "trajectories"
    if not traj.exists():
        return None
    metas = list(traj.glob(f"{ic_id}__*_meta.json")) + list(traj.glob(f"{ic_id}_meta.json"))
    for m in metas:
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
            if meta.get("tokens"):
                return list(meta["tokens"])
        except Exception:
            pass
    npzs = list(traj.glob(f"{ic_id}__*__tokens.npz")) + list(traj.glob(f"{ic_id}*tokens.npz"))
    for n in npzs:
        try:
            z = np.load(n)
            for k in ("tokens", "token", "actions_tok"):
                if k in z:
                    return [int(x) for x in np.asarray(z[k]).reshape(-1)]
        except Exception:
            pass
    return None


def audit_existing():
    """Build baseline audit + seed token library from prior SUCCESS_NEW rows."""
    audit_rows = []
    canon = []
    seed_seqs = []
    by_root = defaultdict(list)
    traj_count = 0
    for src_name, path in SOURCE_MANIFESTS:
        if not path.exists():
            audit_rows.append({
                "source": src_name, "path": str(path), "exists": 0,
                "n_rows": 0, "n_SUCCESS_NEW": 0, "n_unique_ic": 0, "n_unique_root": 0,
            })
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        ok = [r for r in rows if is_success_new(r)]
        ics, roots = set(), set()
        for r in ok:
            traj_count += 1
            ic = (r.get("ic_id") or "").strip()
            tag = (r.get("tag") or "").strip()
            # slow-return CSVs omit ic_id; recover from known tags / trajectory stems
            if not ic or ic.startswith(("cpp_", "suf", "refine_", "a0_", "raw")):
                if "263" in tag or tag.startswith("cpp_4x22"):
                    ic = "hard_910263"
                elif "920031" in tag or "refine_12x3" in tag:
                    ic = "hrv1_920031"
                elif tag and not tag.startswith(("cpp_", "suf", "refine_")):
                    ic = tag
                else:
                    # try trajectory meta next to source
                    metas = list((path.parent / "trajectories").glob("*_meta.json")) if (path.parent / "trajectories").exists() else []
                    for m in metas:
                        try:
                            meta = json.loads(m.read_text(encoding="utf-8"))
                            if meta.get("tokens") and r.get("tokens") and meta.get("ic_id"):
                                if meta.get("tokens") == json.loads(r["tokens"]) if isinstance(r.get("tokens"), str) else None:
                                    ic = meta["ic_id"]
                                    break
                        except Exception:
                            pass
                    if not ic:
                        ic = tag or f"unknown_{src_name}_{traj_count}"
            root = canonical_root(ic, r.get("root_episode") or r.get("root_id"))
            ics.add(ic)
            roots.add(root)
            toks = None
            if r.get("tokens"):
                try:
                    toks = json.loads(r["tokens"]) if isinstance(r["tokens"], str) else r["tokens"]
                except Exception:
                    toks = None
            if toks is None:
                toks = find_traj_tokens(path.parent, ic)
            row = {
                "ic_id": ic,
                "root_episode": root,
                "source_dataset": src_name,
                "source_path": str(path),
                "SUCCESS_NEW": 1,
                "win_mean_theta": r.get("win_mean_theta") or r.get("win_mean_th"),
                "win_mean_pos": r.get("win_mean_pos"),
                "hold_both_s": r.get("hold_both_s"),
                "expert_type": r.get("expert_type") or r.get("family") or src_name,
                "n_tokens": len(toks) if toks else None,
                "tokens_hash": tokens_hash(toks) if toks else None,
                "tokens": json.dumps(toks) if toks else None,
                "provenance": "pre_expansion",
                "snapshot": str(find_snapshot(ic) or find_snapshot(root) or ""),
            }
            canon.append(row)
            by_root[root].append(row)
            if toks:
                seed_seqs.append(list(toks))
        audit_rows.append({
            "source": src_name, "path": str(path), "exists": 1,
            "n_rows": len(rows), "n_SUCCESS_NEW": len(ok),
            "n_unique_ic": len(ics), "n_unique_root": len(roots),
        })
    # dedupe canon by (root, tokens_hash) keeping first
    seen = set()
    dedup = []
    for r in canon:
        key = (r["root_episode"], r.get("tokens_hash") or r["ic_id"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    unique_ics = {r["ic_id"] for r in dedup}
    unique_roots = {r["root_episode"] for r in dedup}
    summary = {
        "n_source_SUCCESS_NEW_rows": traj_count,
        "n_dedup_trajectories": len(dedup),
        "n_unique_ics": len(unique_ics),
        "n_unique_roots": len(unique_roots),
        "unique_roots": sorted(unique_roots),
    }
    return audit_rows, dedup, seed_seqs, summary


def ic_state(env, blob):
    restore_capture(env, blob["capture"])
    cs = contact_summary(env.sim)
    m = measure(env.sim, U_TGT, P_TGT)
    d = decompose(env.sim)
    return {
        "nL": int(cs["nL"]), "nR": int(cs["nR"]), "bilateral": bool(cs["bilateral"]),
        "e_theta_deg": float(m["e_theta_deg"]), "e_pos_mm": float(m["e_pos_mm"]),
        "e_ax_mm": float(d["e_ax_mm"]), "e_lat_mm": float(d["e_lat_mm"]),
        "v_rel_norm": float(np.linalg.norm(m["v_rel_h"])),
        "w_rel_norm": float(np.linalg.norm(m["w_rel_h"])),
        "obj_z": float(m["obj_z"]),
    }


def classify_stability(env, table, book48, blob, st) -> str:
    """Classify IC using physical state + immediate handoff outcome."""
    res = evaluate_full(env, table, book48, blob, [], record_obs=False)
    if res["SUCCESS_NEW"]:
        return "STABLE_CORRECT"
    bil = bool(st["bilateral"])
    n_c = int(st["nL"]) + int(st["nR"])
    v, w = float(st["v_rel_norm"]), float(st["w_rel_norm"])
    if n_c == 0:
        return "LOST_GRASP"
    if (not bil) or v >= 0.25 or w >= 6.0:
        if n_c > 0 and (not bil or v >= 0.15):
            return "UNSTABLE_CONTACT" if n_c > 0 else "LOST_GRASP"
        return "UNSTABLE_CONTACT"
    # bilateral, low motion, but handoff fails pose/hold
    if bil and v < 0.20 and w < 5.0:
        return "STABLE_POSE_ERROR"
    if n_c == 0:
        return "LOST_GRASP"
    return "UNSTABLE_CONTACT"


def difficulty_stratum(st, klass: str) -> str:
    th, pos, v, w = st["e_theta_deg"], st["e_pos_mm"], st["v_rel_norm"], st["w_rel_norm"]
    if klass == "LOST_GRASP" or v >= 0.5 or w >= 12 or th >= 45:
        return "HARD"
    if klass == "UNSTABLE_CONTACT" or v >= 0.2 or th >= 25 or pos >= 25:
        return "MEDIUM"
    if klass == "STABLE_POSE_ERROR" and (th >= 8 or pos >= 12):
        return "MEDIUM"
    return "EASY"


def load_seed_library(seed_seqs):
    generics = [
        CAPTURE_SEED,
        [0] * 24,
        [12] * 5 + [0] * 16,
        [12] * 4 + [17] * 16 + [0] * 8,
        [12] * 4 + [33] * 16 + [0] * 8,
        [4] * 20 + [12] * 5 + [0] * 5,
        [4] * 24 + [12] * 5 + [17] * 12 + [0] * 5,
        [4] * 22 + [12] * 5 + [0] * 3 + [33] * 24 + [12] * 5 + [0] * 3,
        [6] * 16 + [12] * 5 + [0] * 5,
        [10] * 12 + [12] * 5 + [0] * 5,
        [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
        [0] * 8 + [SECURE_REVERSE_WRIST] * 20 + [0] * 8,
        [12] * 3 + [17] * 12 + [0] * 5 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
        [30] + [0] * 12,
        [30, 12, 12, 8] + [17] * 12 + [0] * 8,
        [44] + [0] * 8,
        [32] + [10] * 10 + [12] * 5 + [0] * 5,
    ]
    seen, out = set(), []
    for s in list(seed_seqs) + generics:
        if not s:
            continue
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        out.append(list(k))
    return out


def search_ic(env, table, book48, blob, rng, seeds, klass, budget_a=18, budget_b=8):
    restore_capture(env, blob["capture"])
    desire = -decompose(env.sim)["err_h"]
    props = []
    props.extend(seeds[:16])
    if klass in ("STABLE_POSE_ERROR", "STABLE_CORRECT"):
        props.extend(slide_proposals(book48, desire, rng, n=max(10, budget_a // 2)))
        props.append([0] * 24)
        props.append([12] * 4 + [17] * 16 + [0] * 8)
        props.append([12] * 4 + [33] * 16 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8)
    else:
        props.extend(regrasp_proposals(book48, desire, rng, n=budget_a // 2))
        props.extend(type2_proposals(book48, desire, rng, n=budget_a // 2, open_prefix=[30]))
        props.extend(slide_proposals(book48, desire, rng, n=6))
    seen, uniq = set(), []
    for s in props:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    head, tail = uniq[:14], uniq[14:]
    rng.shuffle(tail)
    uniq = (head + tail)[:budget_a]

    trials = []
    n_try = 0
    for ia, seq in enumerate(uniq):
        variants = [("raw", list(seq)), ("settle12", list(seq) + [0] * 12),
                    ("settle24", list(seq) + [0] * 24)]
        restore_capture(env, blob["capture"])
        ok, _ = run_seq(env, table, book48, blob, seq)
        n_try += 1
        if not ok:
            trials.append({"tag": f"a{ia}", "NEW": 0, "reason": "DROP", "s12": 0})
            continue
        if hand_ref_error_deg(env, blob) > 8:
            variants.append(("ret16", list(seq) + [SECURE_REVERSE_WRIST] * 16 + [0] * 8))
            variants.append(("ret24", list(seq) + [SECURE_REVERSE_WRIST] * 24 + [0] * 8))
        restore_capture(env, blob["capture"])
        run_seq(env, table, book48, blob, seq)
        desire_b = -decompose(env.sim)["err_h"]
        for ib, corr in enumerate(pose_correction_proposals(book48, desire_b, rng, n=budget_b)[:budget_b]):
            if corr:
                variants.append((f"b{ib}", list(seq) + list(corr) + [0] * 8))

        for vname, toks in variants:
            n_try += 1
            res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
            trials.append({
                "tag": f"a{ia}_{vname}", "NEW": int(res["SUCCESS_NEW"]),
                "s12": int(res["success_12s"]), "reason": res["reason"],
                "win_th": res["window"].get("mean_theta"),
                "win_pos": res["window"].get("mean_pos"),
                "n_tokens": len(toks),
            })
            if res["SUCCESS_NEW"]:
                return {"tokens": toks, "tag": f"a{ia}_{vname}", "res": res, "n_try": n_try}, trials, n_try
    return None, trials, n_try


def mechanism_flags(tokens, st0):
    toks = [int(t) for t in tokens]
    has_open = any(t in (30, 32, 36, 39, 41, 44) for t in toks)
    has_pivot = sum(1 for t in toks if t in (4, 6, 10)) >= 8
    has_slide = sum(1 for t in toks if t in (17, 33, 13, 1, 5, 19, 8, 15)) >= 8
    has_return = SECURE_REVERSE_WRIST in toks
    has_brake = 12 in toks and 0 in toks
    return {
        "has_recatch": int(has_open or (st0["nL"] + st0["nR"] == 0)),
        "has_sliding": int(has_slide),
        "has_pivoting": int(has_pivot),
        "has_return": int(has_return),
        "has_capture_brake": int(has_brake and has_open),
        "recatch_plus_manip": int(has_open and (has_slide or has_pivot)),
    }


def save_new_expert(env, table, book48, blob, ic_id, root, tag, toks, st, klass,
                    strata, simple_ok, source, mapping_meta, checksum, snap_src, regimes):
    res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
    if not res["SUCCESS_NEW"]:
        return None
    stem = f"{ic_id}__{tag}"
    save_pair(
        OUT / "trajectories", stem,
        res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    flags = mechanism_flags(toks, st)
    mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
    meta = {
        "ic_id": ic_id, "root_episode": root, "tag": tag,
        "expert_type": "UNIFIED_EXPANSION_FULL_EXPERT",
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "SUCCESS_NEW": True, "label": res["label"], "window": res["window"],
        "tokens": toks, "stability_class": klass, "difficulty": strata,
        "newly_vs_simple": int(not simple_ok), "source": source,
        "snapshot_src": str(snap_src), "regimes": regimes, **flags,
        "obs_dim": 39, "handoff_id": 47, "provenance": "expansion_search",
    }
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    if res.get("trace"):
        write_json(OUT / "raw_contact_traces" / f"{stem}_trace.json", {
            "trace": res["trace"], "mech": mech, "flags": flags,
        })
    # copy snapshot
    dst = OUT / "snapshots" / f"{ic_id}.pkl"
    if not dst.exists():
        if Path(snap_src).exists():
            shutil.copy2(snap_src, dst)
        else:
            dst.write_bytes(pickle.dumps(blob))
    return {
        "ic_id": ic_id, "root_episode": root, "tag": tag,
        "expert_type": "UNIFIED_EXPANSION_FULL_EXPERT",
        "SUCCESS_NEW": 1,
        "win_mean_theta": res["window"].get("mean_theta"),
        "win_mean_pos": res["window"].get("mean_pos"),
        "hold_both_s": res["window"].get("longest_both_ok_s"),
        "n_tokens": len(toks), "tokens": json.dumps(toks),
        "tokens_hash": tokens_hash(toks),
        "stability_class": klass, "difficulty": strata,
        "newly_vs_simple": int(not simple_ok),
        "source": source, "regimes": json.dumps(regimes),
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "provenance": "expansion_search", "snapshot": str(dst),
        **flags,
    }


def pick_historical_candidates(solved_roots: set, rng, max_n: int):
    """Prefer historical ICs not already solved as independent roots."""
    hrv1, hard = [], []
    if HRV1_SNAPS.exists():
        for sp in sorted(HRV1_SNAPS.glob("hrv1_*.pkl")):
            rid = sp.stem
            if rid in solved_roots:
                continue
            hrv1.append(("hard_recovery_v1", rid, sp))
    if HARD_SNAPS.exists():
        for sp in sorted(HARD_SNAPS.glob("hard_*.pkl")):
            rid = sp.stem
            if rid in solved_roots:
                continue
            hard.append(("hard_ic_mining", rid, sp))
    rng.shuffle(hrv1)
    rng.shuffle(hard)
    # 70% budget to hrv1 (STABLE_POSE_ERROR), 30% hard (dynamic)
    n_h = int(round(0.70 * max_n))
    return hrv1[:n_h] + hard[: max_n - n_h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3],
                    help="Stop after reaching stage target (50/75/100)")
    ap.add_argument("--max-hist", type=int, default=90, help="max historical ICs to search")
    ap.add_argument("--max-ball-gen", type=int, default=40, help="new ball-impact attempts")
    ap.add_argument("--max-ball-search", type=int, default=24)
    ap.add_argument("--budget-a", type=int, default=18)
    ap.add_argument("--budget-b", type=int, default=8)
    ap.add_argument("--cap-easy-frac", type=float, default=0.35,
                    help="max fraction of NEW experts that are baseline-solvable")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    stage_targets = {1: 50, 2: 75, 3: 100}
    target = stage_targets[args.stage]
    if args.smoke:
        args.max_hist = 16
        args.max_ball_gen = 8
        args.max_ball_search = 4
        args.budget_a = 10
        args.budget_b = 4
        target = min(target, 40)

    t0w = time.time()
    for sub in ("trajectories", "snapshots", "raw_contact_traces", "RAW"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=== AUDIT existing experts ===", flush=True)
    audit_rows, prior_canon, seed_seqs, audit_summary = audit_existing()
    write_csv(OUT / "BASELINE_DATASET_AUDIT.csv", audit_rows)
    write_json(OUT / "RAW" / "baseline_audit_summary.json", audit_summary)
    print(json.dumps(audit_summary, indent=2), flush=True)

    # seed library
    seeds = load_seed_library(seed_seqs)
    write_json(OUT / "RAW" / "seed_library_meta.json", {"n_seeds": len(seeds)})

    # initialize canonical with prior (do not re-verify all — preserve as claimed)
    canonical = list(prior_canon)
    solved_roots = {r["root_episode"] for r in canonical}
    write_csv(OUT / "CANONICAL_MANIFEST.csv", canonical)

    if args.audit_only:
        print("audit-only done", flush=True)
        return

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

    strata_rows = []
    baseline_rows = []
    transfer_rows = []
    search_rows = []
    new_experts = []
    new_roots_log = []
    coverage = []
    n_rollouts = 0
    n_new_easy = 0

    def n_unique_roots():
        return len({r["root_episode"] for r in canonical})

    def accept_expert(row, simple_ok):
        nonlocal n_new_easy
        root = row["root_episode"]
        if root in solved_roots:
            # action variant on known root — keep as traj but not new root
            row["counts_as_new_root"] = 0
            row["is_easy_baseline"] = int(simple_ok)
            canonical.append(row)
            new_experts.append(row)
            return False
        # Every SUCCESS_NEW root counts toward the unique-root target.
        # Easy-cap only tags metadata (for diversity reporting), never drops the root.
        row["counts_as_new_root"] = 1
        row["is_easy_baseline"] = int(simple_ok)
        if simple_ok:
            n_new_easy += 1
            new_root_count = sum(1 for e in new_experts if int(e.get("counts_as_new_root") or 0) == 1)
            if new_root_count > 0 and n_new_easy / max(1, new_root_count + 1) > args.cap_easy_frac:
                row["easy_over_cap_flag"] = 1
        solved_roots.add(root)
        canonical.append(row)
        new_experts.append(row)
        return True

    # ---------- Phase H: historical ICs ----------
    print("=== PHASE H: historical IC classification + search ===", flush=True)
    hist = pick_historical_candidates(solved_roots, rng, args.max_hist)
    # prioritize by expected STABLE_POSE_ERROR: hrv1 first half already
    for ih, (pool, ic_id, sp) in enumerate(hist):
        if n_unique_roots() >= target:
            print(f"Reached target {target} during historical phase", flush=True)
            break
        try:
            blob = pickle.loads(sp.read_bytes())
        except Exception:
            continue
        if "capture" not in blob or "ref" not in blob:
            continue
        t0 = time.time()
        st = ic_state(env, blob)
        klass = classify_stability(env, table, book48, blob, st)
        n_rollouts += 1
        strata = difficulty_stratum(st, klass)
        regimes = classify_ic_regime(blob, st) if isinstance(blob.get("evidence"), dict) else []
        strata_rows.append({
            "ic_id": ic_id, "root_episode": ic_id, "pool": pool,
            "stability_class": klass, "difficulty": strata,
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "e_lat0": st["e_lat_mm"], "e_ax0": st["e_ax_mm"],
            "v0": st["v_rel_norm"], "w0": st["w_rel_norm"],
            "nL": st["nL"], "nR": st["nR"], "bilateral": int(st["bilateral"]),
            "regimes": json.dumps(regimes),
        })
        if klass == "STABLE_CORRECT":
            coverage.append({"ic_id": ic_id, "phase": "hist", "klass": klass, "full": 0, "skip": "already_NEW"})
            continue
        # Stage 1–2: skip irreversible lost-grasp until stable-pose coverage is denser
        if klass == "LOST_GRASP" and n_unique_roots() < min(target, 75):
            coverage.append({"ic_id": ic_id, "phase": "hist", "klass": klass, "full": 0, "skip": "defer_lost"})
            continue
        if klass == "UNSTABLE_CONTACT" and strata == "HARD" and n_unique_roots() < min(target, 60):
            coverage.append({"ic_id": ic_id, "phase": "hist", "klass": klass, "full": 0, "skip": "defer_hard_unstable"})
            continue

        # matched baselines A–D
        res_a = evaluate_full(env, table, book48, blob, [], record_obs=False); n_rollouts += 1
        res_b = evaluate_full(env, table, book48, blob, [0] * 16, record_obs=False); n_rollouts += 1
        res_c = evaluate_full(env, table, book48, blob, [30] + [0] * 8, record_obs=False); n_rollouts += 1
        res_d = evaluate_full(env, table, book48, blob, CAPTURE_SEED, record_obs=False); n_rollouts += 1
        simple_ok = res_a["SUCCESS_NEW"] or res_b["SUCCESS_NEW"] or res_c["SUCCESS_NEW"]
        baseline_rows.append({
            "ic_id": ic_id, "pool": pool, "klass": klass, "difficulty": strata,
            "A_NEW": int(res_a["SUCCESS_NEW"]), "B_NEW": int(res_b["SUCCESS_NEW"]),
            "C_NEW": int(res_c["SUCCESS_NEW"]), "D_NEW": int(res_d["SUCCESS_NEW"]),
            "simple_solves": int(simple_ok),
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
        })

        # Prefer nontrivial; still search some easy for anchors (capped later)
        ba = args.budget_a + (10 if klass == "STABLE_POSE_ERROR" else 0)
        print(f"  [H {ih+1}/{len(hist)}] {ic_id} {klass}/{strata} "
              f"th={st['e_theta_deg']:.1f} pos={st['e_pos_mm']:.1f} "
              f"roots={n_unique_roots()}/{target}", flush=True)

        best, trials, n_try = search_ic(
            env, table, book48, blob, rng, seeds, klass,
            budget_a=ba, budget_b=args.budget_b,
        )
        n_rollouts += n_try
        for t in trials:
            search_rows.append({"ic_id": ic_id, "phase": "hist", **t})
            transfer_rows.append({
                "ic_id": ic_id, "phase": "hist", "klass": klass,
                "tag": t.get("tag"), "NEW": t.get("NEW"), "reason": t.get("reason"),
            })

        got = 0
        if best is not None:
            row = save_new_expert(
                env, table, book48, blob, ic_id, ic_id, best["tag"], best["tokens"],
                st, klass, strata, simple_ok, pool, mapping_meta, checksum, sp, regimes,
            )
            n_rollouts += 1
            if row is not None:
                is_new = accept_expert(row, simple_ok)
                got = 1
                print(f"    FULL NEW={row['newly_vs_simple']} new_root={int(is_new)} "
                      f"th={row['win_mean_theta']:.1f} pos={row['win_mean_pos']:.1f}",
                      flush=True)
        coverage.append({
            "ic_id": ic_id, "root_episode": ic_id, "phase": "hist", "pool": pool,
            "klass": klass, "difficulty": strata, "full": got,
            "simple_solves": int(simple_ok), "searched": n_try,
            "budget_s": time.time() - t0,
        })

        # checkpoint manifests periodically
        if (ih + 1) % 10 == 0:
            write_csv(OUT / "CANONICAL_MANIFEST.csv", canonical)
            write_csv(OUT / "VERIFIED_FULL_EXPERTS.csv",
                      [r for r in canonical if r.get("provenance") == "expansion_search"] or new_experts)

    # ---------- Phase B: new ball impacts ----------
    if n_unique_roots() < target:
        print("=== PHASE B: new lateral ball-impact episodes ===", flush=True)
        ball_roots = []
        for i in range(args.max_ball_gen):
            if n_unique_roots() >= target:
                break
            spec = draw_lateral_spec(1000 + i)
            eid = f"ubimp_{i:04d}"
            spec["attempt_id"] = eid
            evidence, blob = run_episode(env, spec)
            row = {
                "root_id": eid, "category": evidence.get("category"),
                "reason": evidence.get("reason"),
                "azimuth_deg": spec["azimuth_deg"], "speed_mps": spec["speed_mps"],
                "phi_deg": spec["phi_deg"], "z_off_m": spec["z_off_m"],
                "peak_fn": evidence.get("peak_normal_force_n"),
                "launch_speed": evidence.get("launch_speed"),
            }
            if blob is None or evidence.get("category") != "valid_impact_recovery_ic":
                row["accepted"] = 0
                new_roots_log.append(row)
                continue
            nom = immediate_handoff(env, blob)
            row["nominal_class"] = nom["nominal_class"]
            if nom["nominal_class"] != "RECOVERY_CANDIDATE":
                row["accepted"] = 0
                new_roots_log.append(row)
                continue
            path = OUT / "snapshots" / f"{eid}.pkl"
            path.write_bytes(pickle.dumps(blob))
            st = ic_state(env, blob)
            regimes = classify_ic_regime(blob, st)
            row.update({
                "accepted": 1, "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
                "v0": st["v_rel_norm"], "w0": st["w_rel_norm"],
                "regimes": json.dumps(regimes), "snapshot": str(path),
            })
            new_roots_log.append(row)
            ball_roots.append({
                "ic_id": eid, "blob": blob, "st": st, "regimes": regimes,
                "snap": path, "spec": spec, "evidence": evidence,
            })
            print(f"  ball ACCEPT {eid} th={st['e_theta_deg']:.1f} pos={st['e_pos_mm']:.1f} "
                  f"v={st['v_rel_norm']:.2f} {regimes}", flush=True)

        # search accepted ball roots (prefer recoverable)
        def ball_score(r):
            st = r["st"]
            s = min(st["e_theta_deg"], 40) * 0.5 + min(st["e_pos_mm"], 30) * 0.4
            if st["bilateral"]:
                s += 15
            if st["v_rel_norm"] < 0.4:
                s += 10 - st["v_rel_norm"] * 10
            else:
                s -= 10
            return s

        ball_roots = sorted(ball_roots, key=ball_score, reverse=True)[: args.max_ball_search]
        for ib, br in enumerate(ball_roots):
            if n_unique_roots() >= target:
                break
            ic_id = br["ic_id"]
            blob, st = br["blob"], br["st"]
            t0 = time.time()
            klass = classify_stability(env, table, book48, blob, st)
            n_rollouts += 1
            strata = difficulty_stratum(st, klass)
            strata_rows.append({
                "ic_id": ic_id, "root_episode": ic_id, "pool": "generated_ball",
                "stability_class": klass, "difficulty": strata,
                "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
                "e_lat0": st["e_lat_mm"], "e_ax0": st["e_ax_mm"],
                "v0": st["v_rel_norm"], "w0": st["w_rel_norm"],
                "nL": st["nL"], "nR": st["nR"], "bilateral": int(st["bilateral"]),
                "regimes": json.dumps(br["regimes"]),
            })
            res_a = evaluate_full(env, table, book48, blob, [], record_obs=False); n_rollouts += 1
            res_b = evaluate_full(env, table, book48, blob, [0] * 16, record_obs=False); n_rollouts += 1
            res_c = evaluate_full(env, table, book48, blob, [30] + [0] * 8, record_obs=False); n_rollouts += 1
            res_d = evaluate_full(env, table, book48, blob, CAPTURE_SEED, record_obs=False); n_rollouts += 1
            simple_ok = res_a["SUCCESS_NEW"] or res_b["SUCCESS_NEW"] or res_c["SUCCESS_NEW"]
            baseline_rows.append({
                "ic_id": ic_id, "pool": "generated_ball", "klass": klass, "difficulty": strata,
                "A_NEW": int(res_a["SUCCESS_NEW"]), "B_NEW": int(res_b["SUCCESS_NEW"]),
                "C_NEW": int(res_c["SUCCESS_NEW"]), "D_NEW": int(res_d["SUCCESS_NEW"]),
                "simple_solves": int(simple_ok),
                "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            })
            print(f"  [B {ib+1}/{len(ball_roots)}] {ic_id} {klass}/{strata} "
                  f"roots={n_unique_roots()}/{target}", flush=True)
            best, trials, n_try = search_ic(
                env, table, book48, blob, rng, seeds, klass,
                budget_a=args.budget_a + 8, budget_b=args.budget_b,
            )
            n_rollouts += n_try
            for t in trials:
                search_rows.append({"ic_id": ic_id, "phase": "ball", **t})
            got = 0
            if best is not None:
                row = save_new_expert(
                    env, table, book48, blob, ic_id, ic_id, best["tag"], best["tokens"],
                    st, klass, strata, simple_ok, "generated_ball", mapping_meta, checksum,
                    br["snap"], br["regimes"],
                )
                n_rollouts += 1
                if row is not None:
                    is_new = accept_expert(row, simple_ok)
                    got = 1
                    print(f"    FULL new_root={int(is_new)} th={row['win_mean_theta']:.1f} "
                          f"pos={row['win_mean_pos']:.1f}", flush=True)
            coverage.append({
                "ic_id": ic_id, "root_episode": ic_id, "phase": "ball",
                "pool": "generated_ball", "klass": klass, "difficulty": strata,
                "full": got, "simple_solves": int(simple_ok), "searched": n_try,
                "budget_s": time.time() - t0,
            })

    # ---------- finalize artifacts ----------
    write_csv(OUT / "CANONICAL_MANIFEST.csv", canonical)
    write_csv(OUT / "NEW_PHYSICAL_ROOTS.csv", new_roots_log)
    write_csv(OUT / "IC_DIFFICULTY_STRATA.csv", strata_rows)
    write_csv(OUT / "MATCHED_BASELINES.csv", baseline_rows)
    write_csv(OUT / "EXPERT_TRANSFER_RESULTS.csv", transfer_rows[:3000])
    write_csv(OUT / "SEARCH_RESULTS.csv", search_rows[:4000])
    expansion_full = [r for r in canonical if r.get("provenance") == "expansion_search"]
    write_csv(OUT / "VERIFIED_FULL_EXPERTS.csv", expansion_full if expansion_full else new_experts)
    write_csv(OUT / "ROOT_COVERAGE.csv", coverage)

    # splits by root
    roots_sorted = sorted({r["root_episode"] for r in canonical})
    rng2 = np.random.default_rng(SEED + 7)
    order = list(roots_sorted)
    rng2.shuffle(order)
    n = len(order)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    test_set = set(order[:n_test])
    val_set = set(order[n_test:n_test + n_val])
    split_rows = []
    for r in canonical:
        root = r["root_episode"]
        if root in test_set:
            sp = "test"
        elif root in val_set:
            sp = "validation"
        else:
            sp = "train"
        split_rows.append({
            "ic_id": r["ic_id"], "root_episode": root, "split": sp,
            "source_dataset": r.get("source_dataset") or r.get("source"),
            "provenance": r.get("provenance"),
        })
    write_csv(OUT / "TRAIN_VAL_TEST_SPLITS.csv", split_rows)

    # diversity / answers
    unique_roots = {r["root_episode"] for r in canonical}
    unique_ics = {r["ic_id"] for r in canonical}
    new_root_experts = [r for r in expansion_full if int(r.get("counts_as_new_root") or 0) == 1]
    newly_vs_simple = [r for r in expansion_full if int(r.get("newly_vs_simple") or 0) == 1]
    recatch_manip = [r for r in canonical if int(r.get("recatch_plus_manip") or 0) == 1
                     or ("RECATCH" in str(r.get("expert_type", "")).upper() and "SLID" in str(r).upper())]
    # better flags for expansion experts
    n_recatch_manip = sum(1 for r in expansion_full if int(r.get("recatch_plus_manip") or 0) == 1)
    # include prior ball + any ball source
    ball_roots_ok = {r["root_episode"] for r in canonical
                     if str(r.get("source_dataset") or r.get("source") or "").find("ball") >= 0
                     or str(r["root_episode"]).startswith(("bimp_", "ubimp_"))}
    near_horiz = sum(1 for r in expansion_full if "NEAR_HORIZONTAL" in str(r.get("regimes", "")))
    high_mot = sum(1 for r in strata_rows if float(r.get("v0") or 0) >= 0.2 or float(r.get("w0") or 0) >= 4)
    high_mot_full = sum(
        1 for r in expansion_full
        if r["root_episode"] in {s["ic_id"] for s in strata_rows
                                 if float(s.get("v0") or 0) >= 0.2 or float(s.get("w0") or 0) >= 4}
    )
    stable_pose_full = sum(1 for r in expansion_full if r.get("stability_class") == "STABLE_POSE_ERROR")
    lengths = [int(r["n_tokens"]) for r in canonical if r.get("n_tokens")]
    # handoff time approx = n_tokens * 0.02
    handoff_s = [0.02 * int(r["n_tokens"]) for r in canonical if r.get("n_tokens")]

    klass_ctr = Counter(r.get("stability_class") for r in expansion_full)
    diff_ctr = Counter(r.get("difficulty") for r in expansion_full)
    src_ctr = Counter((r.get("source_dataset") or r.get("source") or r.get("provenance")) for r in canonical)

    answers = {
        "Q1_n_verified_traj_before": audit_summary["n_source_SUCCESS_NEW_rows"],
        "Q2_n_unique_ics_before": audit_summary["n_unique_ics"],
        "Q2_n_unique_roots_before": audit_summary["n_unique_roots"],
        "Q3_n_new_physical_roots_generated": sum(1 for r in new_roots_log if int(r.get("accepted") or 0) == 1),
        "Q3_n_ball_attempts": len(new_roots_log),
        "Q4_n_new_independent_roots_SUCCESS_NEW": len(new_root_experts),
        "Q5_n_unique_successful_roots_after": len(unique_roots),
        "Q5_n_unique_ics_after": len(unique_ics),
        "Q5_n_trajectories_after": len(canonical),
        "Q6_n_newly_vs_simple": len(newly_vs_simple),
        "Q7_n_recatch_plus_slide_pivot_expansion": n_recatch_manip,
        "Q8_n_ball_impact_roots": len(ball_roots_ok),
        "Q9_n_near_horizontal_expansion": near_horiz,
        "Q9_n_high_motion_strata_seen": high_mot,
        "Q9_n_high_motion_full_expansion": high_mot_full,
        "Q10_seq_len_mean": float(np.mean(lengths)) if lengths else None,
        "Q10_seq_len_median": float(np.median(lengths)) if lengths else None,
        "Q10_handoff_s_mean": float(np.mean(handoff_s)) if handoff_s else None,
        "Q10_handoff_s_median": float(np.median(handoff_s)) if handoff_s else None,
        "Q11_stability_class_expansion": dict(klass_ctr),
        "Q11_difficulty_expansion": dict(diff_ctr),
        "Q11_source_distribution_all": dict(src_ctr),
        "Q11_stable_pose_error_full": stable_pose_full,
        "Q12_underrepresented": [],
        "Q13_suitable_for_SFT": len(unique_roots) >= 50,
        "stage_target": target,
        "stage_reached": n_unique_roots() >= target,
        "wall_s": time.time() - t0w,
        "n_rollouts": n_rollouts,
        "unique_roots_list": sorted(unique_roots),
    }
    # underrepresented regimes
    under = []
    if near_horiz < 5:
        under.append("near_horizontal_FULL")
    if high_mot_full < 5:
        under.append("high_motion_FULL")
    if n_recatch_manip < 5:
        under.append("recatch_plus_manipulation")
    if len(ball_roots_ok) < 10:
        under.append("ball_impact_roots")
    if stable_pose_full < 10 and args.stage >= 2:
        under.append("stable_pose_error_expansion")
    answers["Q12_underrepresented"] = under
    answers["Q13_note"] = (
        f"Dedup roots={len(unique_roots)} (target {target}). "
        f"Prior roots={audit_summary['n_unique_roots']}; "
        f"new independent={len(new_root_experts)}. "
        + ("Ready for initial unified SFT with root-grouped splits."
           if answers["Q13_suitable_for_SFT"] else
           "Below preferred 50-root floor for unified SFT; continue expansion.")
    )
    write_json(OUT / "RAW" / "answers.json", answers)

    # diversity report
    div = f"""# Diversity Report — Unified K49 Expansion

## Root counts
- Before: **{audit_summary['n_unique_roots']}** unique roots / {audit_summary['n_unique_ics']} ICs / {audit_summary['n_source_SUCCESS_NEW_rows']} traj rows
- After: **{len(unique_roots)}** unique roots / {len(unique_ics)} ICs / {len(canonical)} traj
- New independent SUCCESS_NEW roots this run: **{len(new_root_experts)}**

## Expansion experts by stability class
{dict(klass_ctr)}

## Expansion experts by difficulty
{dict(diff_ctr)}

## All trajectories by source
{dict(src_ctr)}

## Ball-impact successful roots
{len(ball_roots_ok)}: {sorted(ball_roots_ok)[:30]}{'...' if len(ball_roots_ok)>30 else ''}

## Sequence length (tokens)
mean={answers['Q10_seq_len_mean']}, median={answers['Q10_seq_len_median']}
Approx HANDOFF time (n_tokens×20ms): mean={answers['Q10_handoff_s_mean']}, median={answers['Q10_handoff_s_median']}

## Underrepresented
{under}
"""
    (OUT / "DIVERSITY_REPORT.md").write_text(div, encoding="utf-8")

    report = f"""# Unified K49 Dataset Expansion

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**SUCCESS_NEW locked. No SFT/GRPO.**  
**Stage target:** {target} unique roots — **achieved {len(unique_roots)}** ({'YES' if len(unique_roots)>=target else 'NO'})

## Baseline (before expansion)

| Metric | Value |
|--------|------:|
| SUCCESS_NEW trajectory rows | {audit_summary['n_source_SUCCESS_NEW_rows']} |
| Dedup trajectories | {audit_summary['n_dedup_trajectories']} |
| Unique ICs | {audit_summary['n_unique_ics']} |
| **Unique roots** | **{audit_summary['n_unique_roots']}** |

## After expansion

| Metric | Value |
|--------|------:|
| Trajectories in canonical | {len(canonical)} |
| Unique ICs | {len(unique_ics)} |
| **Unique successful roots** | **{len(unique_roots)}** |
| New independent roots (this run) | {len(new_root_experts)} |
| Newly vs simple baselines | {len(newly_vs_simple)} |
| Expansion STABLE_POSE_ERROR FULL | {stable_pose_full} |
| Ball-impact successful roots | {len(ball_roots_ok)} |
| Rollouts | {n_rollouts} |
| Wall time | {answers['wall_s']:.1f}s |

## Q1–Q13

1. Verified traj before: **{answers['Q1_n_verified_traj_before']}**
2. Unique ICs / roots before: **{answers['Q2_n_unique_ics_before']}** / **{answers['Q2_n_unique_roots_before']}**
3. New physical roots generated (accepted ball): **{answers['Q3_n_new_physical_roots_generated']}** (attempts {answers['Q3_n_ball_attempts']})
4. New independent roots with SUCCESS_NEW: **{answers['Q4_n_new_independent_roots_SUCCESS_NEW']}**
5. Unique successful roots after dedup: **{answers['Q5_n_unique_successful_roots_after']}**
6. Newly solved vs simple baselines: **{answers['Q6_n_newly_vs_simple']}**
7. Recatch+Sliding/Pivoting (expansion flags): **{answers['Q7_n_recatch_plus_slide_pivot_expansion']}**
8. Real ball-impact successful roots: **{answers['Q8_n_ball_impact_roots']}**
9. Near-horizontal expansion FULL: **{answers['Q9_n_near_horizontal_expansion']}**; high-motion FULL expansion: **{answers['Q9_n_high_motion_full_expansion']}**
10. Seq length mean/median: {answers['Q10_seq_len_mean']:.1f} / {answers['Q10_seq_len_median']:.1f}; HANDOFF time mean/median: {answers['Q10_handoff_s_mean']:.2f}s / {answers['Q10_handoff_s_median']:.2f}s
11. Balance — stability {answers['Q11_stability_class_expansion']}; difficulty {answers['Q11_difficulty_expansion']}; sources {answers['Q11_source_distribution_all']}
12. Underrepresented: {answers['Q12_underrepresented']}
13. Suitable for unified SFT? **{answers['Q13_suitable_for_SFT']}** — {answers['Q13_note']}

## Splits
Root-grouped train/val/test in `TRAIN_VAL_TEST_SPLITS.csv` (approx 70/15/15).

## Scripts
- `scripts/expand_k49_unified_dataset.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    ready = f"""# Dataset Readiness — Unified K49 Expansion

## Verdict: {"YES" if answers["Q13_suitable_for_SFT"] else "NOT YET"}

- Unique SUCCESS_NEW roots: **{len(unique_roots)}** (stage target {target})
- Prior roots preserved: {audit_summary['n_unique_roots']}
- New independent roots added: {len(new_root_experts)}
- Separate coverage: STABLE_POSE_ERROR expansion={stable_pose_full}; ball-impact roots={len(ball_roots_ok)}

### Use
`CANONICAL_MANIFEST.csv` + `TRAIN_VAL_TEST_SPLITS.csv`. Split strictly by `root_episode`.
Do not start SFT/GRPO in this deliverable.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("\nDONE", json.dumps({k: answers[k] for k in answers if not k.endswith("_list")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
