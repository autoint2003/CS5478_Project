"""Build and freeze Hard Recovery Benchmark v1 (no training, no GRPO peek).

Pipeline:
  1) Collect candidates (new draws + optional round-1 leftovers)
  2) Leakage / near-dup filter vs all train/dev/valid/hard-selected
  3) SFT-only difficulty screen (freeze hard definition)
  4) Oracle recoverability (verified vs unverified)
  5) Write frozen manifests + protocol + hash
  6) ONLY THEN evaluate SFT and GRPO pairwise

Writes under results/benchmarks/hard_recovery_v1/.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.datasets.ball_impact import run_episode
from data_generation.datasets.skill_quotas import _fingerprint, _try_sliding
from data_generation.datasets.recovery_rebuild import (
    _known_specs,
    _reserved_keys,
    _scenario_key,
    immediate_handoff,
)
from data_generation.oracles.plan_b.vertical_recatch import (
    _finish,
    _summary,
    rollout as vertical_rollout,
    specs as vertical_specs,
)
from data_generation.validation.impact_labels import cylinder_first, near
from envs.outcomes.horizon_outcome import roll_policy
from evaluation.classify_validation_skills import classify_mechanism
from experiments.sliding.diagnose_sft_sliding_root_cause import free_rollout
from experiments.sliding.mine_hard_ics import (
    STOCH_SEEDS,
    load_known_fingerprints,
)
from models.k48_checkpoint import load_trained_k48
from training.adaptive_hybrid import load_train_snapshots
from training.k47_codebook import load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from evaluation import skill_traces as audit

OUT = ROOT / "results" / "benchmarks" / "hard_recovery_v1"
HARD_MINE = ROOT / "results" / "diagnostics" / "hard_ic_mining"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
GRPO = ROOT / "results/grpo/adaptive_g/full_run_v2_natural_sampling/checkpoint_epoch_0005.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
NEW_SEED_OFFSET = 920_000


def json_safe(obj):
    """Convert numpy / nested structures to JSON-serializable Python types."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row)) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(json_safe(r)) + "\n" for r in rows), encoding="utf-8")


def slim_oracle_payload(outcome: dict | None) -> dict:
    if not outcome:
        return {}
    skip = {"executed", "observations", "contacts", "tokens", "residuals", "handoff_obs", "samples"}
    return {k: json_safe(v) for k, v in outcome.items() if k not in skip}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def family_id_from_row(row: dict) -> str:
    key = row.get("scenario_key")
    if isinstance(key, list):
        return "scen:" + hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12]
    if row.get("azimuth_deg") is not None:
        az = int(round(float(row["azimuth_deg"]) / 45.0) * 45) % 360
        sp = int(round(float(row.get("speed_mps") or 0)))
        return f"az{az:03d}_v{sp}"
    return f"id:{row['id']}"


def load_forbidden() -> dict:
    """Snapshots/fingerprints/ids that must not enter the benchmark."""
    snaps, fps, ids = set(), set(), set()

    def add_manifest(path: Path):
        for row in read_jsonl(path):
            ids.add(row["id"])
            snaps.add(str(row["snapshot"]).replace("\\", "/"))
            if row.get("fingerprint"):
                fps.add(row["fingerprint"])
            try:
                blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
                fps.add(_fingerprint(blob))
            except Exception:
                pass

    # Heuristic train (adaptive) + classic + valid
    for row in load_train_snapshots():
        ids.add(row["id"])
        snaps.add(str(row["snapshot"]).replace("\\", "/"))
        try:
            fps.add(_fingerprint(pickle.loads((ROOT / row["snapshot"]).read_bytes())))
        except Exception:
            pass
    add_manifest(ROOT / "data/valid_skill_quotas.jsonl")
    add_manifest(ROOT / "data/grpo_initial_conditions.jsonl")
    # Merged GRPO manifests
    add_manifest(ROOT / "results/grpo/adaptive_g/datasets/merged_train.jsonl")
    add_manifest(ROOT / "results/grpo/adaptive_g/datasets/merged_validation.jsonl")
    add_manifest(ROOT / "results/grpo/adaptive_g/datasets/hard_test.jsonl")
    # Hard mining selected
    for name in ("selected_train_manifest.jsonl", "selected_dev_manifest.jsonl", "selected_test_manifest.jsonl"):
        add_manifest(HARD_MINE / name)
    # All data snapshots train/valid fingerprints
    fps |= load_known_fingerprints()
    return {"snapshots": snaps, "fingerprints": fps, "ids": ids}


def generate_new(env, attempts: int, known_fp: set[str], seed_offset: int) -> list[dict]:
    snap_dir = OUT / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    specs = _known_specs()
    reserved = _reserved_keys()
    # Also reserve round-1 hard scenario keys
    for row in read_jsonl(HARD_MINE / "candidate_manifest.jsonl"):
        if row.get("scenario_key"):
            reserved.add(tuple(row["scenario_key"]) if isinstance(row["scenario_key"], list) else row["scenario_key"])
    candidates = []
    draw = seed_offset
    ran = 0
    while ran < attempts:
        rng = np.random.default_rng(seed_offset + (draw - seed_offset))
        roll = float(rng.random())
        if roll < 0.55:
            speed = float(rng.uniform(9.2, 10.2))
        elif roll < 0.85:
            speed = float(rng.uniform(8.4, 9.2))
        else:
            speed = float(rng.uniform(7.0, 8.4))
        z_off = float(rng.uniform(-0.006, 0.006)) if rng.random() < 0.4 else float(rng.uniform(-0.022, 0.022))
        phi = float(rng.uniform(-22.0, 22.0)) if rng.random() < 0.45 else float(rng.choice([-1.0, 1.0]) * rng.uniform(22.0, 55.0))
        spec = {
            "azimuth_deg": float(rng.uniform(0.0, 360.0)),
            "speed_mps": speed,
            "z_off_m": z_off,
            "phi_deg": phi,
            "delay_s": float(rng.uniform(0.0, 0.45)),
            "direction": "continuous",
            "family": "ballistic_impact",
            "protocol": "hard_recovery_benchmark_v1",
            "draw": int(draw),
            "generation_seed": int(seed_offset + (draw - seed_offset)),
        }
        draw += 1
        key = _scenario_key(spec)
        if any(near(spec, item) for item in specs) or key in reserved:
            ran += 1
            continue
        eid = f"hrv1_{draw - 1:06d}"
        spec["split"] = "benchmark"
        spec["attempt_id"] = eid
        print("GEN", eid, flush=True)
        evidence, blob = run_episode(env, spec)
        evidence, blob = cylinder_first(evidence, blob)
        ran += 1
        if blob is None or evidence.get("category") != "valid_impact_recovery_ic":
            continue
        nominal = immediate_handoff(env, blob)
        if nominal["nominal_class"] != "RECOVERY_CANDIDATE":
            continue
        fp = _fingerprint(blob)
        if fp in known_fp:
            continue
        known_fp.add(fp)
        reserved.add(key)
        blob["spec"]["id"] = eid
        blob["spec"]["protocol"] = "hard_recovery_benchmark_v1"
        rel = f"results/benchmarks/hard_recovery_v1/snapshots/{eid}.pkl"
        with (ROOT / rel).open("wb") as handle:
            pickle.dump(blob, handle)
        row = {
            "id": eid,
            "draw": draw - 1,
            "generation_seed": spec["generation_seed"],
            "provenance": "round2_new",
            "parent_scenario": list(key),
            "scenario_key": list(key),
            "snapshot": rel.replace("\\", "/"),
            "snapshot_sim_time": float(blob["capture"]["physics"]["time"]) if "capture" in blob else None,
            "fingerprint": fp,
            "family_id": family_id_from_row({"scenario_key": list(key), **spec}),
            "source_grasp_configuration": "vertical",
            "azimuth_deg": spec["azimuth_deg"],
            "speed_mps": spec["speed_mps"],
            "z_off_m": spec["z_off_m"],
            "phi_deg": spec["phi_deg"],
            "delay_s": spec["delay_s"],
            "physics_parameters": {k: spec[k] for k in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")},
            "disturbance_parameters": {k: spec[k] for k in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")},
            "states": evidence.get("states"),
            "nominal_class": nominal["nominal_class"],
            "nominal_reason": nominal["reason"],
            "source_trajectory_id": eid,
        }
        candidates.append(row)
        append_jsonl(OUT / "generation_log.jsonl", {"event": "accept", **row})
        print("CAND", eid, flush=True)
    return candidates


def import_leftovers(forbidden: dict, known_fp: set[str]) -> list[dict]:
    selected_ids = set()
    for name in ("selected_train_manifest.jsonl", "selected_dev_manifest.jsonl", "selected_test_manifest.jsonl"):
        selected_ids |= {r["id"] for r in read_jsonl(HARD_MINE / name)}
    out = []
    snap_dir = OUT / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for row in read_jsonl(HARD_MINE / "candidate_manifest.jsonl"):
        if row["id"] in selected_ids or row["id"] in forbidden["ids"]:
            continue
        snap = str(row["snapshot"]).replace("\\", "/")
        if snap in forbidden["snapshots"]:
            continue
        fp = row.get("fingerprint")
        try:
            blob = pickle.loads((ROOT / snap).read_bytes())
            fp = _fingerprint(blob)
        except Exception:
            continue
        if fp in forbidden["fingerprints"] or fp in known_fp:
            continue
        # Copy snapshot into benchmark tree for self-contained provenance
        eid = f"hrv1_leftover_{row['id']}"
        rel = f"results/benchmarks/hard_recovery_v1/snapshots/{eid}.pkl"
        shutil.copy2(ROOT / snap, ROOT / rel)
        known_fp.add(fp)
        out.append({
            "id": eid,
            "draw": row.get("draw"),
            "generation_seed": row.get("draw"),
            "provenance": "round1_leftover",
            "parent_scenario": row.get("scenario_key"),
            "scenario_key": row.get("scenario_key"),
            "original_hard_id": row["id"],
            "snapshot": rel,
            "snapshot_sim_time": None,
            "fingerprint": fp,
            "family_id": family_id_from_row(row),
            "source_grasp_configuration": "vertical",
            "azimuth_deg": row.get("azimuth_deg"),
            "speed_mps": row.get("speed_mps"),
            "z_off_m": row.get("z_off_m"),
            "phi_deg": row.get("phi_deg"),
            "delay_s": row.get("delay_s"),
            "physics_parameters": {k: row.get(k) for k in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")},
            "disturbance_parameters": {k: row.get(k) for k in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")},
            "states": row.get("states"),
            "nominal_class": row.get("nominal_class"),
            "nominal_reason": row.get("nominal_reason"),
            "source_trajectory_id": row["id"],
        })
    return out


def screen_sft(env, model, book, device, row: dict) -> dict:
    blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
    det = roll_policy(env, model, book, blob, device, cap_steps=200, limit_s=SELECTED_LIMIT_S)
    traced = free_rollout(env, model, book, blob, device, SELECTED_LIMIT_S, stochastic=False)
    mech = classify_mechanism(traced["samples"], traced["t0"], traced["handoff_t"])
    stoch = []
    for seed in STOCH_SEEDS:
        run = free_rollout(env, model, book, blob, device, SELECTED_LIMIT_S, stochastic=True, seed=seed)
        stoch.append({
            "seed": seed,
            "success": bool(run["outcome"].get("success")),
            "reason": run["outcome"].get("reason"),
        })
    rate = float(np.mean([s["success"] for s in stoch]))
    hard = (not bool(det.get("success"))) or rate < (2.0 / 3.0)
    return {
        **row,
        "sft_det_success": bool(det.get("success")),
        "sft_det_reason": det.get("reason"),
        "sft_det_mechanism": mech["mechanism"],
        "sft_det_handoff": traced.get("handoff_t"),
        "sft_stoch_rate": rate,
        "sft_stoch_successes": int(sum(s["success"] for s in stoch)),
        "hard_by_sft": hard,
        "difficulty_rule": "det_fail_or_stoch_lt_2_of_3",
    }


def oracle_one(env, book, row: dict) -> dict:
    blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
    evidence_dir = OUT / "oracle_evidence" / row["id"]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    found = False
    strategy = None
    reason = None
    mechanism_tag = None
    slide = _try_sliding(env, book, blob)
    if slide and slide.get("ok"):
        found = True
        strategy = slide.get("oracle_strategy") or "sliding"
        reason = "SUCCESS_12S"
        mechanism_tag = "Contact-Preserving Adjustment"
        (evidence_dir / "sliding.json").write_text(
            json.dumps(slim_oracle_payload(slide), indent=2), encoding="utf-8",
        )
    if not found:
        for spec in list(vertical_specs()):
            result = vertical_rollout(env, blob, book, spec)
            hold, _extra = _finish(env.sim, blob["ref"])
            summary = _summary(result)
            if bool(hold.get("success")):
                found = True
                strategy = f"vertical_recatch:{spec.get('name')}"
                reason = hold.get("reason") or "SUCCESS_12S"
                mechanism_tag = "Release-Recatch"
                (evidence_dir / f"vertical_{spec.get('name', 'spec')}.json").write_text(
                    json.dumps(json_safe({"hold": hold, "summary": summary}), indent=2), encoding="utf-8",
                )
                break
    if not found and int(row.get("sft_stoch_successes") or 0) > 0:
        found = True
        strategy = "sft_stochastic"
        reason = "SUCCESS_12S"
        mechanism_tag = "Unknown-from-stoch"
        (evidence_dir / "sft_stochastic.json").write_text(
            json.dumps({
                "sft_stoch_successes": row.get("sft_stoch_successes"),
                "sft_stoch_rate": row.get("sft_stoch_rate"),
                "note": "K48 SFT stochastic success under fixed seeds; not a dedicated oracle schedule",
            }, indent=2),
            encoding="utf-8",
        )
    return {
        **row,
        "oracle_found": found,
        "oracle_strategy": strategy,
        "oracle_reason": reason,
        "oracle_mechanism_tag": mechanism_tag,
        "recoverability": "verified_recoverable" if found else "unverified_challenge",
    }


def eval_policy(name, ckpt, env, model, book, device, rows):
    per = []
    for row in rows:
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        print("EVAL", name, row["id"], flush=True)
        det = roll_policy(env, model, book, blob, device, cap_steps=200, limit_s=SELECTED_LIMIT_S)
        samples, t0, traced, handoff_t = audit.roll_policy_full(
            env, model, book, blob, device, SELECTED_LIMIT_S,
        )
        mech = classify_mechanism(samples, t0, handoff_t)
        # Contact migration / both-off facts from classifier
        facts = mech.get("facts") or {}
        per.append({
            "ic_id": row["id"],
            "family_id": row.get("family_id"),
            "provenance": row.get("provenance"),
            "recoverability": row.get("recoverability"),
            "source_grasp": row.get("source_grasp_configuration"),
            "hard_by_sft": row.get("hard_by_sft"),
            "success": bool(traced.get("success")),
            "reason": traced.get("reason"),
            "mechanism": mech.get("mechanism"),
            "mechanism_confidence": mech.get("confidence"),
            "handoff_t": handoff_t,
            "recovery_elapsed": traced.get("recovery_elapsed") if isinstance(traced, dict) else None,
            "max_contact_migration_m": facts.get("max_contact_migration_m"),
            "post_recontact_migration_m": facts.get("post_recontact_migration_m"),
            "both_off_max_s": max(
                (iv.get("duration_s") or 0.0 for iv in (facts.get("sustained_both_off_intervals") or [])),
                default=0.0,
            ),
        })
        # Optional short video for discordant later
        dest = OUT / "traces" / name / row["id"]
        dest.mkdir(parents=True, exist_ok=True)
        audit.write_csv(dest / "trace.csv", samples, t0)
        (dest / "contact_events.json").write_text(
            json.dumps(audit.contact_events(samples, t0), indent=2), encoding="utf-8",
        )
        (dest / "summary.json").write_text(json.dumps(per[-1], indent=2, default=float), encoding="utf-8")
    return per


def mcnemar_exact(n01: int, n10: int) -> float:
    """Two-sided exact McNemar (binomial) for discordant pairs."""
    from math import comb
    n = n01 + n10
    if n == 0:
        return 1.0
    # P(X<=min) * 2 under Bin(n,0.5), capped at 1
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * p))


def bootstrap_delta(pairs: list[tuple[int, int]], n_boot: int = 2000, seed: int = 0):
    """pairs of (sft_success, grpo_success) as 0/1."""
    rng = np.random.default_rng(seed)
    n = len(pairs)
    if n == 0:
        return {"delta": None, "ci95": [None, None]}
    arr = np.asarray(pairs, dtype=np.int64)
    def delta(a):
        grpo_only = int(np.sum((a[:, 0] == 0) & (a[:, 1] == 1)))
        sft_only = int(np.sum((a[:, 0] == 1) & (a[:, 1] == 0)))
        return (grpo_only - sft_only) / len(a)
    point = delta(arr)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(delta(arr[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float(point), "ci95": [float(lo), float(hi)], "n": n}


def family_bootstrap_delta(rows_sft, rows_grpo, n_boot=2000, seed=1):
    by_fam = defaultdict(list)
    gmap = {r["ic_id"]: r for r in rows_grpo}
    for r in rows_sft:
        by_fam[r["family_id"]].append((int(r["success"]), int(gmap[r["ic_id"]]["success"])))
    families = list(by_fam.keys())
    rng = np.random.default_rng(seed)
    def delta_from_fams(fams):
        pairs = []
        for f in fams:
            pairs.extend(by_fam[f])
        a = np.asarray(pairs, dtype=np.int64)
        grpo_only = int(np.sum((a[:, 0] == 0) & (a[:, 1] == 1)))
        sft_only = int(np.sum((a[:, 0] == 1) & (a[:, 1] == 0)))
        return (grpo_only - sft_only) / max(len(a), 1)
    point = delta_from_fams(families)
    boots = []
    for _ in range(n_boot):
        drawn = [families[i] for i in rng.integers(0, len(families), size=len(families))]
        boots.append(delta_from_fams(drawn))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float(point), "ci95": [float(lo), float(hi)], "n_families": len(families)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-attempts", type=int, default=200)
    parser.add_argument("--include-leftovers", action="store_true", default=True)
    parser.add_argument("--no-leftovers", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--resume-from-screened", action="store_true",
                        help="Reuse screened_manifest.jsonl; skip generate+screen")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    include_leftovers = args.include_leftovers and not args.no_leftovers

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    (OUT / "oracle_evidence").mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    sft_model, _ = load_trained_k48(SFT, device)
    sft_model.eval()
    env = make_training_env(device, model=sft_model)

    screened_path = OUT / "screened_manifest.jsonl"
    cand_path = OUT / "candidate_manifest.jsonl"

    if args.resume_from_screened and screened_path.is_file():
        screened = read_jsonl(screened_path)
        candidates = read_jsonl(cand_path) if cand_path.is_file() else screened
        print("RESUME_SCREENED", len(screened), flush=True)
        forbidden = load_forbidden()
        print("FORBIDDEN", len(forbidden["fingerprints"]), "fps", len(forbidden["ids"]), "ids", flush=True)
    else:
        forbidden = load_forbidden()
        known_fp = set(forbidden["fingerprints"])
        print("FORBIDDEN", len(forbidden["fingerprints"]), "fps", len(forbidden["ids"]), "ids", flush=True)

        # --- Phase generate ---
        if args.skip_generate and cand_path.is_file():
            candidates = read_jsonl(cand_path)
            print("REUSE_CANDIDATES", len(candidates), flush=True)
        else:
            if cand_path.is_file():
                cand_path.unlink()
            candidates = []
            if include_leftovers:
                left = import_leftovers(forbidden, known_fp)
                print("LEFTOVERS", len(left), flush=True)
                candidates.extend(left)
            new = generate_new(env, args.generate_attempts, known_fp, NEW_SEED_OFFSET)
            print("NEW", len(new), flush=True)
            candidates.extend(new)
            write_jsonl(cand_path, candidates)

        # Leakage audit
        leaks = []
        clean = []
        for row in candidates:
            if row["fingerprint"] in forbidden["fingerprints"]:
                leaks.append({"id": row["id"], "why": "fingerprint"})
                continue
            if row["snapshot"] in forbidden["snapshots"]:
                leaks.append({"id": row["id"], "why": "snapshot"})
                continue
            clean.append(row)
        (OUT / "leakage_audit.md").write_text(
            "# Leakage audit\n\n"
            f"- Forbidden fingerprints loaded: {len(forbidden['fingerprints'])}\n"
            f"- Candidates before filter: {len(candidates)}\n"
            f"- Rejected: {len(leaks)}\n"
            f"- Kept: {len(clean)}\n"
            f"- Reject samples: {leaks[:20]}\n"
            f"- Note: exact fingerprint + snapshot path vs train/valid/hard-selected; "
            f"family_id=scenario_key hash for cluster-aware stats.\n",
            encoding="utf-8",
        )
        candidates = clean
        write_jsonl(cand_path, candidates)

        # --- SFT screen (no GRPO) ---
        screened = []
        for row in candidates:
            print("SCREEN", row["id"], flush=True)
            screened.append(screen_sft(env, sft_model, book, device, row))
        write_jsonl(screened_path, screened)
        write_jsonl(OUT / "all_valid_candidates_unfiltered_by_hard.jsonl", screened)

    hard = [r for r in screened if r["hard_by_sft"]]
    easy = [r for r in screened if not r["hard_by_sft"]]
    print("SCREEN_DONE", "hard", len(hard), "easy_kept_for_reference", len(easy), flush=True)

    # --- Oracle only on hard; incremental progress for resume ---
    progress_path = OUT / "oracle_progress.jsonl"
    done_map = {r["id"]: r for r in read_jsonl(progress_path)}
    verified, unverified = [], []
    for row in hard:
        if row["id"] in done_map and done_map[row["id"]].get("oracle_found") is not None:
            scored = done_map[row["id"]]
            print("ORACLE_SKIP", row["id"], scored.get("recoverability"), flush=True)
        else:
            print("ORACLE", row["id"], flush=True)
            scored = oracle_one(env, book, row)
            append_jsonl(progress_path, scored)
        if scored["oracle_found"]:
            verified.append(scored)
        else:
            unverified.append(scored)
    write_jsonl(OUT / "verified_manifest.jsonl", verified)
    write_jsonl(OUT / "unverified_manifest.jsonl", unverified)

    # Families
    fam = defaultdict(list)
    for row in verified + unverified:
        fam[row["family_id"]].append(row["id"])
    family_rows = [{"family_id": k, "n": len(v), "ic_ids": v} for k, v in sorted(fam.items())]
    write_jsonl(OUT / "family_manifest.jsonl", family_rows)

    # Freeze protocol + hashes BEFORE GRPO eval
    protocol = {
        "name": "hard_recovery_v1",
        "frozen_before_grpo_eval": True,
        "sft_checkpoint": str(SFT),
        "difficulty_rule": "SFT det fail OR stoch < 2/3 (seeds 11/22/33)",
        "success_criterion": "SUCCESS_12S unchanged",
        "n_candidates": len(screened),
        "n_hard": len(hard),
        "n_verified": len(verified),
        "n_unverified": len(unverified),
        "n_families": len(family_rows),
        "source_grasp_note": "Parent scenarios are vertical-grasp ballistic; mechanism labels are trajectory-based",
        "horizontal_coverage": "Under-represented as parent grasp; see generation_audit.md",
        "exclude": "All GRPO/SFT train+valid+hard selected ICs",
    }
    (OUT / "benchmark_protocol.md").write_text(
        "# Benchmark protocol\n\n```json\n" + json.dumps(protocol, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    hashes = {
        "verified_manifest.jsonl": file_sha256(OUT / "verified_manifest.jsonl") if verified else None,
        "unverified_manifest.jsonl": file_sha256(OUT / "unverified_manifest.jsonl") if unverified else None,
        "candidate_manifest.jsonl": file_sha256(cand_path),
        "screened_manifest.jsonl": file_sha256(OUT / "screened_manifest.jsonl"),
    }
    (OUT / "manifest_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print("FROZEN", json.dumps(protocol), flush=True)

    if args.skip_eval:
        return

    # --- Paired eval (GRPO only after freeze) ---
    bench = verified + unverified
    if not bench:
        print("EMPTY_BENCHMARK", flush=True)
        return

    grpo_model, _ = load_trained_k48(GRPO, device)
    grpo_model.eval()
    # Fresh envs bound to each model
    env_sft = make_training_env(device, model=sft_model)
    env_grpo = make_training_env(device, model=grpo_model)
    sft_res = eval_policy("sft", SFT, env_sft, sft_model, book, device, bench)
    grpo_res = eval_policy("grpo_e5", GRPO, env_grpo, grpo_model, book, device, bench)

    gmap = {r["ic_id"]: r for r in grpo_res}
    paired = []
    for s in sft_res:
        g = gmap[s["ic_id"]]
        if s["success"] and g["success"]:
            tag = "both_success"
        elif s["success"] and not g["success"]:
            tag = "SFT_only"
        elif (not s["success"]) and g["success"]:
            tag = "GRPO_only"
        else:
            tag = "both_failure"
        paired.append({
            "ic_id": s["ic_id"],
            "family_id": s["family_id"],
            "provenance": s["provenance"],
            "recoverability": s["recoverability"],
            "source_grasp": s["source_grasp"],
            "sft_success": int(s["success"]),
            "grpo_success": int(g["success"]),
            "transition": tag,
            "sft_mechanism": s["mechanism"],
            "grpo_mechanism": g["mechanism"],
            "sft_reason": s["reason"],
            "grpo_reason": g["reason"],
            "sft_handoff": s["handoff_t"],
            "grpo_handoff": g["handoff_t"],
            "sft_migration_m": s.get("max_contact_migration_m"),
            "grpo_migration_m": g.get("max_contact_migration_m"),
            "early_handoff_grpo": (
                s["handoff_t"] is not None and g["handoff_t"] is not None and float(g["handoff_t"]) + 1e-9 < float(s["handoff_t"])
            ) if s["handoff_t"] is not None and g["handoff_t"] is not None else None,
        })

    with (OUT / "paired_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0].keys()))
        writer.writeheader()
        writer.writerows(paired)

    # Mechanism transitions
    mech_rows = []
    for r in paired:
        mech_rows.append({
            "ic_id": r["ic_id"],
            "sft_mechanism": r["sft_mechanism"],
            "grpo_mechanism": r["grpo_mechanism"],
            "transition": r["transition"],
            "early_handoff_grpo": r["early_handoff_grpo"],
        })
    with (OUT / "mechanism_transition.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mech_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mech_rows)

    def stratum_stats(items):
        n = len(items)
        sft_s = sum(r["sft_success"] for r in items)
        g_s = sum(r["grpo_success"] for r in items)
        go = sum(1 for r in items if r["transition"] == "GRPO_only")
        so = sum(1 for r in items if r["transition"] == "SFT_only")
        pairs = [(r["sft_success"], r["grpo_success"]) for r in items]
        boot = bootstrap_delta(pairs)
        return {
            "n": n,
            "sft_success": sft_s,
            "grpo_success": g_s,
            "sft_rate": None if n == 0 else sft_s / n,
            "grpo_rate": None if n == 0 else g_s / n,
            "GRPO_only": go,
            "SFT_only": so,
            "both_success": sum(1 for r in items if r["transition"] == "both_success"),
            "both_failure": sum(1 for r in items if r["transition"] == "both_failure"),
            "delta": boot["delta"],
            "delta_ci95": boot["ci95"],
            "mcnemar_p": mcnemar_exact(go, so),
        }

    stats = {
        "all": stratum_stats(paired),
        "verified": stratum_stats([r for r in paired if r["recoverability"] == "verified_recoverable"]),
        "unverified": stratum_stats([r for r in paired if r["recoverability"] == "unverified_challenge"]),
        "family_clustered": family_bootstrap_delta(sft_res, grpo_res),
        "early_handoff_among_grpo_only": {
            "n_grpo_only": sum(1 for r in paired if r["transition"] == "GRPO_only"),
            "n_with_earlier_handoff": sum(
                1 for r in paired if r["transition"] == "GRPO_only" and r["early_handoff_grpo"]
            ),
        },
        "manifest_hashes": hashes,
        "counts": protocol,
    }
    (OUT / "statistical_analysis.json").write_text(json.dumps(json_safe(stats), indent=2), encoding="utf-8")

    # Family stats
    fam_stats = []
    by_f = defaultdict(list)
    for r in paired:
        by_f[r["family_id"]].append(r)
    for fid, items in sorted(by_f.items()):
        fam_stats.append({"family_id": fid, **{k: stratum_stats(items)[k] for k in ("n", "sft_success", "grpo_success", "GRPO_only", "SFT_only", "delta")}})
    with (OUT / "family_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fam_stats[0].keys()) if fam_stats else ["family_id"])
        writer.writeheader()
        writer.writerows(fam_stats)

    # Failure case studies for discordant
    lines = ["# Failure / gain case studies\n"]
    for r in paired:
        if r["transition"] not in ("GRPO_only", "SFT_only"):
            continue
        lines.append(f"## {r['ic_id']} ({r['transition']})")
        lines.append(f"- Recoverability: {r['recoverability']}")
        lines.append(f"- SFT: success={r['sft_success']} reason={r['sft_reason']} mech={r['sft_mechanism']} handoff={r['sft_handoff']}")
        lines.append(f"- GRPO: success={r['grpo_success']} reason={r['grpo_reason']} mech={r['grpo_mechanism']} handoff={r['grpo_handoff']}")
        lines.append(f"- Earlier GRPO handoff: {r['early_handoff_grpo']}")
        lines.append(f"- Traces: `traces/sft/{r['ic_id']}/`, `traces/grpo_e5/{r['ic_id']}/`")
        lines.append("")
    (OUT / "failure_case_studies.md").write_text("\n".join(lines), encoding="utf-8")

    # Mechanism retention questions
    def mech_keep(src_mech: str):
        rows = [r for r in paired if r["sft_mechanism"] == src_mech and r["sft_success"]]
        keep = sum(1 for r in rows if r["grpo_success"])
        return {"n_sft_success": len(rows), "n_grpo_also": keep}

    sliding_keep = mech_keep("Contact-Preserving Sliding")
    # classifiers may use several labels
    for alt in ("Contact-Preserving Adjustment", "Sliding", "contact_preserving_sliding"):
        if sliding_keep["n_sft_success"] == 0:
            sliding_keep = mech_keep(alt)
    recatch_keep = mech_keep("Release–Recatch")
    for alt in ("Release-Recatch", "Recatch", "release_recatch"):
        if recatch_keep["n_sft_success"] == 0:
            recatch_keep = mech_keep(alt)

    grpo_only_rows = [r for r in paired if r["transition"] == "GRPO_only"]
    sft_only_rows = [r for r in paired if r["transition"] == "SFT_only"]
    source_counts = Counter(r.get("source_grasp") for r in paired)
    prov_counts = Counter(r.get("provenance") for r in paired)
    mech_sft = Counter(r["sft_mechanism"] for r in paired if r["sft_success"])
    mech_grpo = Counter(r["grpo_mechanism"] for r in paired if r["grpo_success"])
    mech_grpo_only = Counter(r["grpo_mechanism"] for r in grpo_only_rows)

    report = []
    report.append("# Hard Recovery Benchmark v1 — EVALUATION REPORT\n")
    report.append("## Freeze protocol\n")
    report.append("- Benchmark screened and recoverability-labeled **before** GRPO checkpoint evaluation.")
    report.append(f"- SFT checkpoint: `{SFT}`")
    report.append(f"- GRPO checkpoint: `{GRPO}`")
    report.append(f"- Manifest hashes: `{json.dumps(hashes)}`")
    report.append("")
    report.append("## 1. Benchmark size\n")
    report.append(f"- Candidates screened: **{len(screened)}**")
    report.append(f"- Hard (SFT det fail or stoch < 2/3): **{len(hard)}**")
    report.append(f"- Independent families (`family_id`): **{len(family_rows)}**")
    report.append(f"- Provenance: {dict(prov_counts)}")
    report.append("")
    report.append("## 2. Verified vs Unverified\n")
    report.append(f"- Verified-recoverable: **{len(verified)}**")
    report.append(f"- Unverified challenge: **{len(unverified)}**")
    report.append("- Search failure ≠ physical impossibility.")
    report.append("")
    report.append("## 3. Source grasp coverage\n")
    report.append(f"- Counts by `source_grasp_configuration`: {dict(source_counts)}")
    report.append("- **Horizontal parent grasp: under-represented** (generation uses vertical-cylinder ballistic parents). See `generation_audit.md`.")
    report.append("- Sliding / Vertical / Horizontal here are **source labels**, not policy skill switches.")
    report.append("")
    report.append("## 4. Leakage\n")
    report.append(f"- Forbidden fingerprints: {len(forbidden['fingerprints'])}; rejected exact overlaps: see `leakage_audit.md`.")
    report.append("- Leftovers exclude hard-mining selected train/dev/test; new draws use seed offset 920000 and reserved scenario keys.")
    report.append("")
    report.append("## 5. Success rates (paired, identical snapshots)\n")
    for name in ("all", "verified", "unverified"):
        s = stats[name]
        report.append(
            f"- **{name}**: N={s['n']}; SFT {s['sft_success']}/{s['n']} ({s['sft_rate']}); "
            f"GRPO {s['grpo_success']}/{s['n']} ({s['grpo_rate']})"
        )
    report.append("")
    report.append("## 6–7. Paired transitions and Δ\n")
    a = stats["all"]
    report.append("| | GRPO success | GRPO failure |")
    report.append("|---|---:|---:|")
    report.append(f"| SFT success | {a['both_success']} | {a['SFT_only']} |")
    report.append(f"| SFT failure | {a['GRPO_only']} | {a['both_failure']} |")
    report.append("")
    report.append(f"- GRPO_only: **{a['GRPO_only']}**; SFT_only: **{a['SFT_only']}**")
    report.append(f"- Δ = (GRPO_only − SFT_only) / N = **{a['delta']:.4f}**")
    report.append(f"- Paired bootstrap 95% CI: **{a['delta_ci95']}**")
    report.append(f"- Exact McNemar p: **{a['mcnemar_p']}**")
    report.append(f"- Family-clustered bootstrap: **{stats['family_clustered']}**")
    report.append("")
    report.append("## 8. Mechanism retention / change\n")
    report.append(f"- SFT success mechanisms: {dict(mech_sft)}")
    report.append(f"- GRPO success mechanisms: {dict(mech_grpo)}")
    report.append(f"- GRPO_only mechanisms: {dict(mech_grpo_only)}")
    report.append(f"- Sliding-source SFT successes retained by GRPO: {sliding_keep}")
    report.append(f"- Recatch-source SFT successes retained by GRPO: {recatch_keep}")
    report.append("- Do **not** infer forgetting from label counts alone; see `mechanism_transition.csv` and traces.")
    report.append("")
    report.append("## 9. HANDOFF timing among GRPO_only\n")
    eh = stats["early_handoff_among_grpo_only"]
    report.append(f"- GRPO_only with earlier HANDOFF than SFT: {eh['n_with_earlier_handoff']} / {eh['n_grpo_only']}")
    report.append("")
    report.append("## 10. Verdict\n")
    if a["n"] < 30 or len(family_rows) < 15:
        report.append("- **Sample / family coverage limited** — treat Δ as exploratory.")
    if len(verified) < max(10, a["n"] // 5):
        report.append("- **Verified-recoverable coverage is thin** — many failures may be search-budget limited.")
    if a["GRPO_only"] > a["SFT_only"] and (a["mcnemar_p"] is None or a["mcnemar_p"] < 0.05):
        report.append("- Evidence **supports** a GRPO paired gain on this SFT-hard set (check CI and family bootstrap).")
    elif a["GRPO_only"] < a["SFT_only"]:
        report.append("- Evidence **does not support** GRPO advantage; SFT_only ≥ GRPO_only.")
    else:
        report.append("- Evidence is **inconclusive** on GRPO hard-recovery generalization (small discordant set or non-significant McNemar).")
    report.append("")
    report.append("## Artifacts\n")
    report.append("- `paired_results.csv`, `mechanism_transition.csv`, `family_statistics.csv`")
    report.append("- `statistical_analysis.json`, `failure_case_studies.md`")
    report.append("- `traces/sft/`, `traces/grpo_e5/`, `oracle_evidence/`")
    (OUT / "EVALUATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("EVAL_DONE", json.dumps(stats["all"]), flush=True)


if __name__ == "__main__":
    main()
