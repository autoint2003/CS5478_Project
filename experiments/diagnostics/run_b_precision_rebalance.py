#!/usr/bin/env python
"""Rebalance B experts toward Precision-required recovery.

Phases:
  1) Diagnose Tree80→Precision failures from stage2_b_grpo_multistage_v1
  2) Improved Precision (dx-before-dz safe-band) revalidation on recoverable exits
  3) Expand genuine ball-impact ICs; keep Precision-required successes
  4) Build Precision-dominant training subset (~20% Direct / 80% Precision target)
  5) Frozen M111 ρ=0.35 retention on new Precision experts

Does not: start SFT/RL, modify C experts, edit frozen prior datasets, retrain GRPO,
or modify M111 from held-out failures. Original stage2_b_grpo_multistage_v1 kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse Tree80 / Precision pipeline helpers from the prior validation script.
from experiments.diagnostics import run_b_grpo_multistage_dataset as base  # noqa: E402

from data_generation.datasets.ball_impact import run_episode  # noqa: E402
from data_generation.datasets.recovery_rebuild import (  # noqa: E402
    _known_specs,
    _reserved_keys,
    _scenario_key,
    immediate_handoff,
)
from data_generation.datasets.skill_quotas import _fingerprint  # noqa: E402
from data_generation.validation.impact_labels import cylinder_first, near  # noqa: E402
from training.action_vocab_k49 import build_k49_table  # noqa: E402
from training.skill_aware_action_vocab import (  # noqa: E402
    DEFAULT_WEIGHTS,
    load_motion_codebook,
)

HARD_DRAW_START = 910_000


def draw_hard_spec(draw: int) -> dict:
    """Harder-biased legal continuous ball-impact launch (same family as hard_ic_mining)."""
    rng = np.random.default_rng(HARD_DRAW_START + int(draw))
    roll = float(rng.random())
    if roll < 0.55:
        speed = float(rng.uniform(9.2, 10.2))
    elif roll < 0.85:
        speed = float(rng.uniform(8.4, 9.2))
    else:
        speed = float(rng.uniform(7.0, 8.4))
    if float(rng.random()) < 0.4:
        z_off = float(rng.uniform(-0.006, 0.006))
    else:
        z_off = float(rng.uniform(-0.022, 0.022))
    if float(rng.random()) < 0.45:
        phi = float(rng.uniform(-22.0, 22.0))
    else:
        phi = float(rng.choice([-1.0, 1.0]) * rng.uniform(22.0, 55.0))
    return {
        "azimuth_deg": float(rng.uniform(0.0, 360.0)),
        "speed_mps": speed,
        "z_off_m": z_off,
        "phi_deg": phi,
        "delay_s": float(rng.uniform(0.0, 0.45)),
        "direction": "continuous",
        "family": "ballistic_impact",
        "protocol": "b_precision_rebalance_mine_v1",
        "draw": int(draw),
    }

VAL = ROOT / "results/diagnostics/b_precision_rebalance"
PRIOR_VAL = ROOT / "results/diagnostics/b_grpo_multistage_validation"
PRIOR_DS = ROOT / "results/data_generation/stage2_b_grpo_multistage_v1"
DS_OUT = ROOT / "results/data_generation/stage2_b_precision_dominant_v1"
HARD = ROOT / "results/diagnostics/hard_ic_mining"
HRV1 = ROOT / "results/benchmarks/hard_recovery_v1/snapshots"
TREE80 = base.TREE80
M111_META = base.M111_META
DATASET_VERSION = "stage2_b_precision_dominant_v1"
ORACLE_TAG = "tree80_e3_plus_precision_v24_dx_before_dz"

# Held-out lineages (frozen; never enter training subset)
HELDOUT_SPLITS = {"test", "dev"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_gen_log() -> list[dict]:
    path = PRIOR_VAL / "GENERATION_LOG.csv"
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _truthy(v) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes")


def _f(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def classify_failure(row: dict) -> dict:
    """Phase-1 taxonomy for one PRECISION_FAILURE row."""
    status = str(row.get("precision_status") or "")
    phases = str(row.get("precision_phases") or "")
    e_pos = _f(row.get("exit_e_pos"), 0.0) or 0.0
    theta = _f(row.get("exit_theta"), 0.0) or 0.0
    n_prec = int(_f(row.get("n_precision"), 0) or 0)
    need_dx = _truthy(row.get("need_dx"))
    need_dz = _truthy(row.get("need_dz"))
    need_dtheta = _truthy(row.get("need_dtheta"))
    direct_reason = str(row.get("direct_reason") or "")

    labels: list[str] = []
    recoverable = False
    notes = []

    if status == "STABLE_GATE_FAILED":
        labels.append("A_outside_recoverable_basin")
        notes.append("entry_gate rejected unstable grasp; do not force Precision")
    elif status == "PRECISION_EXECUTED_HANDOFF_FAIL":
        if "NO_REMAINING_CORRECTION" in phases and n_prec < 50:
            labels.append("D_correction_insufficient_or_missed_residual")
            notes.append("geometry assessed ready but SUCCESS_NEW still fails")
            recoverable = True
        elif n_prec > 200:
            labels.append("D_correction_insufficient")
            notes.append("long correction ran; e_pos improved but NEW handoff failed")
            recoverable = True
        else:
            labels.append("E_handoff_fail_after_partial")
            recoverable = True
    elif status == "PRECISION_PRIMITIVE_FAILED":
        if "correct_dz_inward" in phases and n_prec == 0:
            labels.append("C_scheduler_gap_dz_before_dx")
            notes.append(
                "dz scheduled while |ex| likely > dz_safe_ex; aborted with 0 actions "
                "(AXIS ex_tol=12 vs dz_safe=6). Fix: dx-before-dz."
            )
            recoverable = True
        elif "correct_dtheta" in phases and n_prec > 0 and need_dtheta:
            labels.append("D_orientation_correction_insufficient")
            notes.append("pivot executed but failed to reach theta band / lost grasp")
            recoverable = theta < 25.0 and e_pos < 25.0
        elif "dx_verified_lateral" in phases:
            labels.append("D_lateral_correction_insufficient")
            notes.append("support-transfer + verified lateral failed or destabilized")
            recoverable = e_pos < 25.0 and theta < 15.0
        elif "correct_dz" in phases:
            labels.append("D_axial_correction_insufficient")
            recoverable = True
        else:
            labels.append("E_other_primitive_fail")
            recoverable = e_pos < 22.0 and theta < 20.0
    else:
        labels.append("E_other")

    if _truthy(row.get("genuine_release_recatch")) and _truthy(row.get("exit_bilateral")):
        if status != "STABLE_GATE_FAILED":
            notes.append("Stage-1 exit physically recaught + bilateral")
    else:
        labels.append("B_contact_unstable")
        recoverable = False

    # Borderline Direct (12S ok, NEW fail) still Precision-required under SUCCESS_NEW
    if direct_reason == "SUCCESS_12S":
        notes.append("Direct SUCCESS_12S but not SUCCESS_NEW — Precision-required under NEW")

    return {
        "ic_id": row.get("ic_id"),
        "precision_status": status,
        "direct_reason": direct_reason,
        "n_precision": n_prec,
        "exit_e_pos": e_pos,
        "exit_theta": theta,
        "need_dx": need_dx,
        "need_dz": need_dz,
        "need_dtheta": need_dtheta,
        "phases": phases[:240],
        "taxonomy": "|".join(labels) if labels else "E_other",
        "plausibly_recoverable_with_better_precision": recoverable,
        "notes": "; ".join(notes),
    }


def phase1_diagnose() -> dict:
    rows = _load_gen_log()
    fails = [r for r in rows if r.get("category") == "PRECISION_FAILURE"]
    diagnoses = [classify_failure(r) for r in fails]
    by_tax = Counter()
    for d in diagnoses:
        for t in str(d["taxonomy"]).split("|"):
            by_tax[t] += 1
    prim = [d for d in diagnoses if d["precision_status"] == "PRECISION_PRIMITIVE_FAILED"]
    summary = {
        "n_precision_failures": len(fails),
        "n_primitive_failed": len(prim),
        "n_stable_gate": sum(1 for d in diagnoses if d["precision_status"] == "STABLE_GATE_FAILED"),
        "n_handoff_fail": sum(1 for d in diagnoses if d["precision_status"] == "PRECISION_EXECUTED_HANDOFF_FAIL"),
        "n_plausibly_recoverable": sum(1 for d in diagnoses if d["plausibly_recoverable_with_better_precision"]),
        "taxonomy_counts": dict(by_tax),
        "primitive_failed_ics": [d["ic_id"] for d in prim],
        "recoverable_ics": [
            d["ic_id"] for d in diagnoses if d["plausibly_recoverable_with_better_precision"]
        ],
        "root_cause_summary": [
            "Dominant mode (6/11 primitive): dz scheduled under e_pos>12 while |ex| in (dz_safe=6, AXIS_ex=12]; "
            "check_dz_inward_entry rejects → PRECISION_PRIMITIVE_FAILED with 0 actions.",
            "Fix applied: assess_need forces dx when dz needed and |ex|>dz_safe_ex; scheduler defers dz on lateral precondition.",
            "2 STABLE_GATE_FAILED: Stage-1 exit outside Precision basin — not forced.",
            "3 HANDOFF_FAIL: correction ran / geometry near-ready but SUCCESS_NEW still fails — residual grip/axis.",
            "Large-theta pivot failures (101/263/277) and catastrophic dx (305/327) less likely without new primitives.",
        ],
    }
    base.write_csv(VAL / "PHASE1_FAILURE_DIAGNOSIS.csv", diagnoses)
    base.atomic_json(VAL / "PHASE1_SUMMARY.json", summary)
    return {"summary": summary, "diagnoses": diagnoses, "fail_rows": fails}


def load_extra_hrv1_ball_ics(exclude: set[str]) -> list[dict]:
    """Optional extra Recovery-entry candidates from hard_recovery_v1 (ball-impact leftovers)."""
    out = []
    if not HRV1.exists():
        return out
    for p in sorted(HRV1.glob("*.pkl")):
        ic_id = p.stem
        if ic_id in exclude:
            continue
        # Prefer leftover / hard_ prefixes that look like ball-impact provenance
        if not (
            "hard_" in ic_id
            or "bimp" in ic_id
            or "leftover" in ic_id
            or ic_id.startswith("hrv1_")
        ):
            continue
        try:
            blob = pickle.loads(p.read_bytes())
        except Exception:
            continue
        if "capture" not in blob or "ref" not in blob:
            continue
        out.append(
            {
                "ic_id": ic_id,
                "pool": "hrv1_ball_impact",
                "source": "hard_recovery_v1_snapshot",
                "capture": blob["capture"],
                "ref": blob["ref"],
                "handcrafted_stage1": None,
                "meta": {"azimuth_deg": None, "speed_mps": None, "assigned_split": "train"},
            }
        )
    return out


def mine_new_ball_impacts(env, n_attempts: int, start_id: int) -> list[dict]:
    """Generate additional genuine ball-impact Recovery-entry captures (no teleports)."""
    out = []
    specs = _known_specs()
    reserved = _reserved_keys()
    known_fp: set[str] = set()
    for p in (HARD / "snapshots").glob("hard_*.pkl"):
        try:
            known_fp.add(_fingerprint(pickle.loads(p.read_bytes())))
        except Exception:
            pass
    draw = int(start_id)
    ran = 0
    mine_dir = VAL / "mined_snapshots"
    mine_dir.mkdir(parents=True, exist_ok=True)
    while ran < n_attempts:
        spec = draw_hard_spec(draw)
        draw += 1
        ran += 1
        key = _scenario_key(spec)
        if any(near(spec, item) for item in specs) or key in reserved:
            continue
        eid = f"hard_{draw - 1:06d}"
        if (HARD / "snapshots" / f"{eid}.pkl").exists() or (mine_dir / f"{eid}.pkl").exists():
            continue
        spec["split"] = "rebalance_mine"
        spec["attempt_id"] = eid
        print(f"  MINE {eid} speed={spec['speed_mps']:.2f} az={spec['azimuth_deg']:.1f}", flush=True)
        try:
            evidence, blob = run_episode(env, spec)
            evidence, blob = cylinder_first(evidence, blob)
        except Exception as exc:
            print(f"    mine error: {exc}", flush=True)
            continue
        if blob is None or evidence.get("category") != "valid_impact_recovery_ic":
            continue
        nominal = immediate_handoff(env, blob)
        if nominal.get("nominal_class") != "RECOVERY_CANDIDATE":
            continue
        fp = _fingerprint(blob)
        if fp in known_fp:
            continue
        known_fp.add(fp)
        blob["spec"] = dict(blob.get("spec") or {})
        blob["spec"]["id"] = eid
        blob["spec"]["protocol"] = "b_precision_rebalance_mine_v1"
        pickle.dump(blob, (mine_dir / f"{eid}.pkl").open("wb"))
        out.append(
            {
                "ic_id": eid,
                "pool": "hard_ball_impact_rebalance",
                "source": "ball_impact_rebalance_mining",
                "capture": blob["capture"],
                "ref": blob["ref"],
                "handcrafted_stage1": None,
                "meta": {
                    "azimuth_deg": spec.get("azimuth_deg"),
                    "speed_mps": spec.get("speed_mps"),
                    "z_off_m": spec.get("z_off_m"),
                    "phi_deg": spec.get("phi_deg"),
                    "delay_s": spec.get("delay_s"),
                    "assigned_split": "train",
                    "fingerprint": fp,
                },
            }
        )
        print(f"    KEEP {eid} states={evidence.get('states')}", flush=True)
    return out


def diversify_select(rows: list[dict], k: int, *, key_fn) -> list[dict]:
    """Greedy diversity by (azimuth bucket, speed bucket, e_pos bucket)."""
    if k <= 0 or not rows:
        return []
    if len(rows) <= k:
        return list(rows)

    def feat(r):
        meta = r.get("ic_meta") or r.get("meta") or {}
        az = meta.get("azimuth_deg")
        sp = meta.get("speed_mps")
        if az is None:
            az = hash(str(r.get("ic_id"))) % 360
        if sp is None:
            sp = 8.0
        e = _f(r.get("exit_e_pos"), 15.0) or 15.0
        return (int(float(az) // 45), int(float(sp) // 1.0), int(e // 3.0))

    buckets = defaultdict(list)
    for r in rows:
        buckets[feat(r)].append(r)
    selected = []
    # Round-robin across buckets
    while len(selected) < k and any(buckets.values()):
        for b in list(buckets.keys()):
            if len(selected) >= k:
                break
            if buckets[b]:
                selected.append(buckets[b].pop(0))
        # prune empty
        buckets = defaultdict(list, {k_: v for k_, v in buckets.items() if v})
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-hard", type=int, default=0, help="0=all hard snapshots")
    ap.add_argument("--max-hrv1", type=int, default=40)
    ap.add_argument("--mine-attempts", type=int, default=120, help="new ball-impact attempts")
    ap.add_argument("--mine-start-id", type=int, default=920000)
    ap.add_argument("--skip-mine", action="store_true")
    ap.add_argument("--skip-m111", action="store_true")
    ap.add_argument("--direct-quota-frac", type=float, default=0.20)
    ap.add_argument("--max-direct", type=int, default=8)
    ap.add_argument("--phase1-only", action="store_true")
    args = ap.parse_args()

    VAL.mkdir(parents=True, exist_ok=True)
    for sub in ("ics", "continuous", "stage1", "precision", "outcomes", "failures", "tokens_m111", "tokens_k49"):
        (DS_OUT / sub).mkdir(parents=True, exist_ok=True)
    dirs = {k: DS_OUT / k for k in ("ics", "continuous", "stage1", "precision", "outcomes", "failures", "tokens_m111", "tokens_k49")}

    # Route base persistence into the new versioned dataset (do not touch prior DS).
    base.DS_OUT = DS_OUT
    base.DATASET_VERSION = DATASET_VERSION
    base.ORACLE_TAG = ORACLE_TAG
    base.VAL = VAL

    print("Phase1 diagnose prior Precision failures…", flush=True)
    p1 = phase1_diagnose()
    print(json.dumps(p1["summary"], indent=2), flush=True)
    if args.phase1_only:
        return

    device = torch.device(args.device)
    print("Loading Tree80 E3…", flush=True)
    model, book = base.load_trained_k48(TREE80, device)
    env_pol = base.make_training_env(device, model=model)
    env_plain = base.make_plain_env()

    # ---- Phase 2–3: revalidate prior Direct-fail + expand ----
    hard_ics = base.load_hard_snapshots()
    if args.max_hard > 0:
        hard_ics = hard_ics[: args.max_hard]
    known_ids = {ic["ic_id"] for ic in hard_ics}

    # Prioritize previously failed Precision ICs first
    fail_ids = [d["ic_id"] for d in p1["diagnoses"]]
    hard_by = {ic["ic_id"]: ic for ic in hard_ics}
    ordered = []
    for i in fail_ids:
        if i in hard_by:
            ordered.append(hard_by.pop(i))
    ordered.extend(hard_by.values())

    expand_ics = list(ordered)
    if not args.skip_mine and args.mine_attempts > 0:
        print(f"Phase3 mine new ball-impacts: attempts={args.mine_attempts}", flush=True)
        mined = mine_new_ball_impacts(env_pol, args.mine_attempts, args.mine_start_id)
        print(f"  mined valid Recovery-entry ICs: {len(mined)}", flush=True)
        expand_ics.extend(mined)

    if args.max_hrv1 > 0:
        hrv = load_extra_hrv1_ball_ics(known_ids | {ic["ic_id"] for ic in expand_ics})
        hrv = hrv[: args.max_hrv1]
        print(f"Phase3 extra hrv1 ICs: {len(hrv)}", flush=True)
        expand_ics.extend(hrv)

    funnel = Counter()
    gen_rows = []
    expert_manifest = []
    diagnostic_manifest = []
    precision_experts = []
    direct_experts = []

    # Carry forward handcrafted Direct (small quota later) — revalidate once
    print("Revalidate handcrafted B Direct (pool for small Direct quota)…", flush=True)
    hand_pool = []
    for ic in base.load_handcrafted_b_direct():
        out = base.copy_handcrafted_expert(dirs, ic, env_plain)
        if out and out.get("partition") == "expert":
            out["dataset_version"] = DATASET_VERSION
            out["oracle_tag"] = ORACLE_TAG
            hand_pool.append(out)
            funnel["handcrafted_direct_ok"] += 1
        else:
            funnel["handcrafted_direct_fail"] += 1

    print(f"Phase2/3 process {len(expand_ics)} ICs with improved Precision…", flush=True)
    for i, ic in enumerate(expand_ics):
        print(f"=== [{i+1}/{len(expand_ics)}] {ic['ic_id']} ({ic['pool']}) ===", flush=True)
        row = base.process_tree80_ic(env_pol, env_plain, model, book, device, ic)
        funnel["tested"] += 1
        funnel["stage1_ok"] += int(bool(row.get("stage1_ok")))
        funnel["genuine_recatch"] += int(bool(row.get("genuine_release_recatch")))
        funnel["direct_NEW"] += int(bool(row.get("direct_NEW")))
        if row.get("direct_NEW") is False and row.get("stage1_ok"):
            funnel["direct_fail_given_s1ok"] += 1
            funnel["prec_NEW_given_direct_fail"] += int(bool(row.get("prec_NEW")))
        funnel[str(row.get("category"))] += 1

        gen_rows.append({k: v for k, v in row.items() if not k.startswith("_")})

        if row["partition"] == "expert" and row["category"] == "TREE80_PRECISION":
            out = base.save_expert_row(dirs, ic, row, expert_kind="tree80_precision")
            expert_manifest.append(out)
            precision_experts.append(out)
            print(f"  SAVED PRECISION NEW={out['handoff']['SUCCESS_NEW']}", flush=True)
        elif row["partition"] == "expert" and row["category"] == "TREE80_DIRECT":
            out = base.save_expert_row(dirs, ic, row, expert_kind="tree80_direct")
            direct_experts.append(out)
            print(f"  SAVED DIRECT (pool) NEW={out['handoff']['SUCCESS_NEW']}", flush=True)
        else:
            out = base.save_expert_row(dirs, ic, row, expert_kind="diagnostic")
            diagnostic_manifest.append(out)
            print(f"  diagnostic {row.get('category')} status={row.get('precision_status')}", flush=True)

        if (i + 1) % 10 == 0:
            base.write_csv(VAL / "GENERATION_LOG.csv", gen_rows)

    base.write_csv(VAL / "GENERATION_LOG.csv", gen_rows)

    # How many of the prior 16 failures recovered?
    recovered = []
    for d in p1["diagnoses"]:
        ic_id = d["ic_id"]
        hit = next((e for e in precision_experts if e.get("ic_id") == ic_id), None)
        recovered.append(
            {
                "ic_id": ic_id,
                "prior_status": d["precision_status"],
                "prior_taxonomy": d["taxonomy"],
                "recovered": hit is not None,
                "plausible": d["plausibly_recoverable_with_better_precision"],
            }
        )
    base.write_csv(VAL / "PHASE2_RECOVERY.csv", recovered)
    n_rec = sum(1 for r in recovered if r["recovered"])
    print(f"Phase2 recovered prior failures: {n_rec}/{len(recovered)}", flush=True)

    # ---- Phase 4: Precision-dominant subset ----
    # Hold out lineages marked test/dev in hard mining meta
    def is_heldout(e: dict) -> bool:
        meta = e.get("ic_meta") or {}
        sp = str(meta.get("assigned_split") or "").lower()
        return sp in HELDOUT_SPLITS

    prec_train = [e for e in precision_experts if not is_heldout(e)]
    prec_hold = [e for e in precision_experts if is_heldout(e)]
    # Dedup by lineage
    by_lin = {}
    for e in prec_train:
        by_lin[e["lineage_id"]] = e
    prec_train = list(by_lin.values())

    n_prec = len(prec_train)
    # Target Direct ≈ 20% of subset → Direct = Prec * frac/(1-frac)
    frac = float(args.direct_quota_frac)
    target_direct = int(round(n_prec * frac / max(1e-6, 1.0 - frac)))
    target_direct = max(2, min(int(args.max_direct), target_direct if n_prec else 2))

    # Build Direct pool: prefer handcrafted diversity + a few Tree80 Direct (not overlapping Precision ICs)
    prec_ic_ids = {e["ic_id"] for e in precision_experts}
    tree_direct_pool = [e for e in direct_experts if e["ic_id"] not in prec_ic_ids and not is_heldout(e)]
    # Diversity select
    n_hand = min(len(hand_pool), max(1, target_direct // 2 + target_direct % 2))
    n_tree = min(len(tree_direct_pool), max(0, target_direct - n_hand))
    # If still short, take more handcrafted
    if n_hand + n_tree < target_direct:
        n_hand = min(len(hand_pool), target_direct - n_tree)
    hand_sel = diversify_select(hand_pool, n_hand, key_fn=lambda e: e["ic_id"])
    tree_sel = diversify_select(tree_direct_pool, n_tree, key_fn=lambda e: e["ic_id"])
    direct_sel = hand_sel + tree_sel

    selected = prec_train + direct_sel
    # Write selected training subset manifest (all Precision train + small Direct)
    # Also keep held-out Precision separately
    train_manifest = []
    for e in selected:
        e = dict(e)
        e["subset"] = "train_precision_dominant"
        e["dataset_version"] = DATASET_VERSION
        train_manifest.append(e)
    hold_manifest = []
    for e in prec_hold:
        e = dict(e)
        e["subset"] = "heldout_precision"
        hold_manifest.append(e)

    # Persist unused Direct experts as reference-only (not in training subset)
    ref_direct = []
    sel_ids = {e["sample_id"] for e in direct_sel}
    for e in hand_pool + direct_experts:
        if e["sample_id"] not in sel_ids:
            ee = dict(e)
            ee["subset"] = "reference_direct_not_in_train"
            ref_direct.append(ee)

    with (DS_OUT / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for e in train_manifest + hold_manifest:
            f.write(json.dumps(base.jsafe(e)) + "\n")
    with (DS_OUT / "reference_direct_manifest.jsonl").open("w", encoding="utf-8") as f:
        for e in ref_direct:
            f.write(json.dumps(base.jsafe(e)) + "\n")
    with (DS_OUT / "diagnostic_manifest.jsonl").open("w", encoding="utf-8") as f:
        for e in diagnostic_manifest:
            f.write(json.dumps(base.jsafe(e)) + "\n")

    raw_dist = {
        "n_tested": int(funnel["tested"]),
        "stage1_ok": int(funnel["stage1_ok"]),
        "genuine_recatch": int(funnel["genuine_recatch"]),
        "tree80_direct_raw": len(direct_experts),
        "tree80_precision_raw": len(precision_experts),
        "precision_failures": int(funnel.get("PRECISION_FAILURE", 0)),
        "stage1_failures": int(funnel.get("STAGE1_FAILURE", 0)),
        "handcrafted_direct_ok": int(funnel["handcrafted_direct_ok"]),
        "prior_failures_recovered": n_rec,
        "funnel": dict(funnel),
    }
    selected_dist = {
        "n_train_total": len(train_manifest),
        "n_train_precision": len(prec_train),
        "n_train_direct": len(direct_sel),
        "n_train_direct_handcrafted": len(hand_sel),
        "n_train_direct_tree80": len(tree_sel),
        "frac_direct": (len(direct_sel) / len(train_manifest)) if train_manifest else None,
        "frac_precision": (len(prec_train) / len(train_manifest)) if train_manifest else None,
        "n_heldout_precision": len(hold_manifest),
        "n_reference_direct_excluded": len(ref_direct),
        "target_direct_frac": frac,
        "limitation": None
        if len(prec_train) >= 8
        else (
            "Physical Precision yield remains limited; subset is Precision-dominant among "
            "available verified successes but absolute Precision count is still small."
        ),
    }
    base.atomic_json(
        VAL / "PHASE4_DISTRIBUTIONS.json",
        {"raw_physical_generation": raw_dist, "selected_training_subset": selected_dist},
    )
    base.atomic_json(DS_OUT / "SUMMARY.json", {
        "dataset_version": DATASET_VERSION,
        "created": _now(),
        "oracle_tag": ORACLE_TAG,
        "raw": raw_dist,
        "selected": selected_dist,
        "phase1": p1["summary"],
        "original_datasets_unchanged": [
            str(PRIOR_DS.relative_to(ROOT)).replace("\\", "/"),
            str(PRIOR_VAL.relative_to(ROOT)).replace("\\", "/"),
        ],
    })

    print(
        f"Phase4 selected train: Prec={len(prec_train)} Direct={len(direct_sel)} "
        f"(frac_direct={selected_dist['frac_direct']})",
        flush=True,
    )

    # ---- Phase 5: M111 on Precision experts (all verified, incl heldout) ----
    m111_summary = None
    if not args.skip_m111 and precision_experts:
        print("Phase5 M111 representability on Precision experts…", flush=True)
        cb = load_motion_codebook(M111_META)
        w = DEFAULT_WEIGHTS
        k49_table = build_k49_table(config_id=base.K49_CONFIG)

        m_rows = []
        for e in precision_experts:
            cont = np.load(DS_OUT / e["continuous_demo"]["path"])
            acts = np.asarray(cont["actions"], np.float32)
            ic_blob = pickle.loads((DS_OUT / e["ic_pkl"]).read_bytes())
            qm = base.quantize_and_replay(
                env_plain, ic_blob["capture"], ic_blob["ref"], acts, cb.prototypes, cb.residual, w, "M111"
            )
            qk = base.quantize_k49(env_plain, ic_blob["capture"], ic_blob["ref"], acts, k49_table)
            # stage-split error if precision length known
            n_s1 = int(e.get("n_actions_stage1") or 0)
            n_pr = int(e.get("n_actions_precision") or max(0, len(acts) - n_s1))
            stage_err = {}
            if n_s1 > 0 and len(qm.get("recon_actions", [])) == len(acts):
                ra = np.asarray(qm["recon_actions"], np.float32)
                s1_err = float(np.mean(np.linalg.norm(ra[:n_s1] - acts[:n_s1], axis=1))) if n_s1 else None
                pr_err = (
                    float(np.mean(np.linalg.norm(ra[n_s1 : n_s1 + n_pr] - acts[n_s1 : n_s1 + n_pr], axis=1)))
                    if n_pr
                    else None
                )
                stage_err = {"m111_stage1_mean_l2": s1_err, "m111_precision_mean_l2": pr_err}
            # save tokens
            sid = e["sample_id"]
            np.savez_compressed(
                dirs["tokens_m111"] / f"{sid}__m111.npz",
                tokens=qm["tokens"],
                residuals=qm["residuals"],
                recon_actions=qm["recon_actions"],
            )
            np.savez_compressed(
                dirs["tokens_k49"] / f"{sid}__k49.npz",
                tokens=qk.get("tokens", np.zeros(0, np.int64)),
                residuals=qk.get("residuals", np.zeros((0, 7), np.float32)),
                recon_actions=qk.get("recon_actions", np.zeros((0, 7), np.float32)),
            )
            m_rows.append(
                {
                    "sample_id": sid,
                    "ic_id": e["ic_id"],
                    "category": e["category"],
                    "continuous_NEW": True,
                    "M111_NEW": qm["SUCCESS_NEW"],
                    "M111_12S": qm["SUCCESS_12S"],
                    "M111_clip": qm["clipped_frac"],
                    "M111_mean_wlinf": qm["mean_recon_wlinf"],
                    "M111_n_tok": qm["n_unique_tokens"],
                    "K49_NEW": qk.get("SUCCESS_NEW"),
                    "K49_12S": qk.get("SUCCESS_12S"),
                    "K49_clip": qk.get("clipped_frac"),
                    **stage_err,
                    "sat_frac_per_dim": json.dumps(qm.get("sat_frac_per_dim")),
                }
            )
            print(f"  {sid[:48]} M111={qm['SUCCESS_NEW']} K49={qk.get('SUCCESS_NEW')}", flush=True)

        m111_summary = {
            "M": 111,
            "rho": cb.residual.to_list(),
            "handoff_id": int(cb.handoff_id),
            "n_precision_experts": len(m_rows),
            "M111_NEW": sum(1 for r in m_rows if r["M111_NEW"]),
            "K49_NEW": sum(1 for r in m_rows if r["K49_NEW"]),
            "retention": (sum(1 for r in m_rows if r["M111_NEW"]) / len(m_rows)) if m_rows else None,
            "mean_clip_frac": float(np.mean([r["M111_clip"] for r in m_rows])) if m_rows else None,
            "mean_precision_stage_l2": float(
                np.mean([r["m111_precision_mean_l2"] for r in m_rows if r.get("m111_precision_mean_l2") is not None])
            )
            if any(r.get("m111_precision_mean_l2") is not None for r in m_rows)
            else None,
        }
        base.write_csv(VAL / "M111_PRECISION.csv", m_rows)
        base.atomic_json(VAL / "M111_SUMMARY.json", m111_summary)

    # ---- Report ----
    md = [
        "# B Precision-Dominant Rebalance",
        "",
        f"Generated: {_now()}",
        "",
        "## Phase 1 — Failure diagnosis",
        f"- Precision failures analyzed: {p1['summary']['n_precision_failures']}",
        f"- PRIMITIVE_FAILED: {p1['summary']['n_primitive_failed']}",
        f"- Plausibly recoverable: {p1['summary']['n_plausibly_recoverable']}",
        f"- Taxonomy: `{json.dumps(p1['summary']['taxonomy_counts'])}`",
        "",
        "## Phase 2 — Improved Precision",
        f"- Prior failures recovered: **{n_rec}/{len(recovered)}**",
        "- Change: dx-before-dz when `|ex|>dz_safe_ex` (assess_need + scheduler deferral)",
        "",
        "## Phase 3–4 — Distributions",
        "### Raw physical generation",
        f"```json\n{json.dumps(raw_dist, indent=2)}\n```",
        "### Selected training subset",
        f"```json\n{json.dumps(selected_dist, indent=2)}\n```",
        "",
        "## Phase 5 — M111",
        f"```json\n{json.dumps(m111_summary, indent=2)}\n```",
        "",
        "## Dataset",
        f"- Path: `{DS_OUT.relative_to(ROOT).as_posix()}`",
        f"- Train Precision: {len(prec_train)} | Train Direct: {len(direct_sel)} | Held-out Precision: {len(hold_manifest)}",
        "",
        "## Limitations",
        selected_dist.get("limitation")
        or "Absolute Precision yield bounded by genuine Tree80 recatch + Direct-fail + recoverable basin.",
        "",
        "No SFT/RL started. Prior datasets unchanged.",
    ]
    (VAL / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    (DS_OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    base.atomic_json(
        VAL / "RECOMMENDATION.json",
        {
            "ready_for_precision_dominant_packing_pilot": bool(len(prec_train) >= 8 and selected_dist.get("frac_precision", 0) >= 0.6),
            "n_train_precision": len(prec_train),
            "n_train_direct": len(direct_sel),
            "m111_precision_retention": None if not m111_summary else m111_summary.get("retention"),
            "notes": [
                "Prefer this subset over Direct-heavy stage2_b_grpo_multistage_v1 for B Precision objective.",
                "Do not start SFT/RL until M111 retention on Precision is reviewed.",
                "Original Direct-heavy corpus preserved untouched.",
            ],
        },
    )
    print("DONE", json.dumps({
        "prec_train": len(prec_train),
        "direct_train": len(direct_sel),
        "recovered_failures": n_rec,
        "m111": m111_summary,
    }), flush=True)


if __name__ == "__main__":
    main()
