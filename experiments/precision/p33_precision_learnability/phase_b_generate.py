"""P33 Phase B — Expand independent Precision-required (P1) ICs via existing Oracle.

Uses production `run_precision` + immediate-HANDOFF STRICT classification.
Does not change Oracle, physics, or M111. No training on holdout lineages.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.precision.d2_eligibility import check_d2_eligible  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p13_teleport_d2.teleport_builder import clone_capture  # noqa: E402
from experiments.precision.p14_small_error_d2.small_error_builder import (  # noqa: E402
    SmallErrorSpec,
    build_small_error_ic,
)
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/p33_precision_learnability"
ICS = OUT / "generated_ics"
SEED = 33

# Donor bases: Stage-1 verified Direct-HANDOFF grasps (not P32/P21 val D2 ICs)
BASE_CANDIDATES = [
    (
        "ball_train_v2_0155",
        ROOT
        / "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF"
        / "A_STABLE_DIRECT_HANDOFF__A_STAGE1__ball_train_v2_0155.pkl",
    ),
    (
        "k48r_train_00010",
        ROOT
        / "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF"
        / "A_STABLE_DIRECT_HANDOFF__A_STAGE1__k48r_train_00010.pkl",
    ),
    (
        "slide_quota_400100",
        ROOT
        / "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF"
        / "A_STABLE_DIRECT_HANDOFF__A_STAGE1__slide_quota_400100.pkl",
    ),
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_holdout() -> set[str]:
    p = OUT / "phase_a_summary.json"
    hold = set()
    if (OUT / "DATASET_MANIFEST.csv").is_file():
        import csv as _csv

        with (OUT / "DATASET_MANIFEST.csv").open(encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if r.get("holdout_eval") in ("1", "True", "true"):
                    hold.add(str(r.get("lineage_id") or ""))
                    hold.add(str(r.get("physical_root") or ""))
    # Always exclude known P32 D2 val lineages
    hold |= {
        "A_STABLE_DIRECT_HANDOFF__A_STAGE1__ball_train_0027",
        "A_STABLE_DIRECT_HANDOFF__A_STAGE1__ball_train_v2_0168",
        "ball_train_0027",
        "ball_train_v2_0168",
    }
    return {h for h in hold if h}


def load_existing_p1_roots() -> set[str]:
    roots = set()
    p = OUT / "phase_a_summary.json"
    if p.is_file():
        roots |= set(json.loads(p.read_text(encoding="utf-8")).get("p1_train_root_ids") or [])
    return roots


def discover_bases() -> list[dict]:
    """Find available Stage-1 Direct-HANDOFF donors, excluding holdout lineages."""
    hold = load_holdout()
    folder = ROOT / "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF"
    out = []
    if folder.is_dir():
        for p in sorted(folder.glob("*.pkl")):
            # lineage from filename
            stem = p.stem
            # A_STABLE_DIRECT_HANDOFF__A_STAGE1__<root>
            parts = stem.split("__")
            root = parts[-1] if parts else stem
            lin = stem
            if root in hold or lin in hold or any(h in stem for h in hold if len(h) > 8):
                continue
            out.append({"sample_id": stem, "path": p, "physical_root": root})
    # Prefer diverse roots; cap donors for runtime
    by_root = {}
    for b in out:
        by_root.setdefault(b["physical_root"], b)
    bases = list(by_root.values())
    rng = np.random.default_rng(SEED)
    rng.shuffle(bases)
    return bases  # use all unique roots (holdout already filtered)


def offset_matrix(stage: int) -> list[SmallErrorSpec]:
    """Precision-eligible band (P14): small offsets stay CASE4 / not precision_required."""
    specs: list[SmallErrorSpec] = []
    # Mirror P14 eligible band — below ~8 mm usually fails assess_need
    dx_vals = [8.0, 10.0, 12.0, -8.0, -10.0, -12.0]
    dz_vals = [8.0, 12.0, 16.0, -8.0, -12.0, -16.0]
    if stage >= 50:
        dx_vals += [9.0, 11.0, 14.0, -9.0, -11.0, -14.0]
        dz_vals += [10.0, 14.0, -10.0, -14.0]
    if stage >= 100:
        dx_vals += [7.0, 13.0, 15.0, -7.0, -13.0, -15.0]
        dz_vals += [7.0, 9.0, 15.0, -7.0, -9.0, -15.0]
    for v in dx_vals:
        specs.append(SmallErrorSpec(family="DX_ONLY", target_dx_mm=float(v), target_dz_mm=0.0))
    for v in dz_vals:
        specs.append(SmallErrorSpec(family="DZ_ONLY", target_dx_mm=0.0, target_dz_mm=float(v)))
    for dx, dz in (
        (8.0, 4.0),
        (10.0, 4.0),
        (4.0, 12.0),
        (4.0, 16.0),
        (-8.0, -4.0),
        (-4.0, -12.0),
        (-10.0, -8.0),
        (10.0, 8.0),
        (-12.0, 4.0),
        (12.0, -4.0),
        (-8.0, 8.0),
        (8.0, -8.0),
    ):
        specs.append(SmallErrorSpec(family="COMBINED", target_dx_mm=float(dx), target_dz_mm=float(dz)))
    return specs


def immediate_ok(env, capture, ref) -> tuple[bool, dict]:
    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    return bool(hand.get("SUCCESS_NEW")), hand


def classify_candidate(env, capture, ref, actions, cfg) -> dict[str, Any]:
    imm_ok, imm_hand = immediate_ok(env, capture, ref)
    # Oracle already ran; use STRICT from oracle handoff if provided via re-eval of actions
    from experiments.precision.p18_discrete_vocab_audit.replay_harness import replay_actions

    prec = replay_actions(env, capture, ref, actions, stabilize_n=8)
    prec_ok = bool(prec.get("SUCCESS_NEW"))
    if imm_ok and prec_ok:
        cls = "P2"
    elif imm_ok and not prec_ok:
        cls = "P0"
    elif (not imm_ok) and prec_ok:
        cls = "P1"
    else:
        cls = "P3"
    return {
        "class": cls,
        "imm_SUCCESS_NEW": int(imm_ok),
        "imm_reason": imm_hand.get("std_reason") or imm_hand.get("reason"),
        "prec_SUCCESS_NEW": int(prec_ok),
        "prec_reason": prec.get("reason"),
        "n_actions": int(len(np.asarray(actions).reshape(-1, 7))) if actions is not None else 0,
    }


def run_phase_b(target_p1: int = 100, val_n: int = 40, test_n: int = 40) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    ICS.mkdir(parents=True, exist_ok=True)
    print(f"=== P33 Phase B: generate P1 (target={target_p1}) ===", flush=True)
    cfg = PrecisionConfig(evaluate_handoff=True)
    env = make_env()
    bases = discover_bases()
    print(f"donor bases={len(bases)}", flush=True)
    if not bases:
        _write_md(
            OUT / "GENERATION_REPORT.md",
            f"# GENERATION_REPORT\n\nGenerated: {_utc()}\n\n**Bottleneck:** no donor bases found under experts_v3_stage1_verified.\n",
        )
        return {"p1": 0, "bottleneck": "no_bases"}

    existing_roots = load_existing_p1_roots()
    hold = load_holdout()
    seen_fp: set[str] = set()
    funnel = Counter()
    gen_rows: list[dict] = []
    p1_pool: list[dict] = []
    p0_pool: list[dict] = []

    # Resume from ICs already on disk
    if ICS.is_dir():
        for p in ICS.glob("*.pkl"):
            blob = pickle.loads(p.read_bytes())
            acts = blob.get("actions")
            n_act = int(len(np.asarray(acts).reshape(-1, 7))) if acts is not None else 0
            row = {
                "tag": blob.get("tag") or p.stem,
                "lineage_id": blob.get("lineage_id"),
                "physical_root": blob.get("physical_root"),
                "n_actions": n_act,
                "class": "P1",
                "ic_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "oracle_status": blob.get("oracle_status"),
            }
            p1_pool.append(row)
            fp = capture_fingerprint(blob["capture"])
            seen_fp.add(fp)
            if row["lineage_id"]:
                existing_roots.add(row["lineage_id"])
        print(f"resumed {len(p1_pool)} P1 from disk", flush=True)

    specs = offset_matrix(target_p1)
    t0 = time.time()
    goal = target_p1  # total P1 pool size before split (val/test carved from roots)

    for bi, base in enumerate(bases):
        if len(p1_pool) >= goal:
            break
        path = base["path"]
        if not path.is_file():
            continue
        raw = pickle.loads(path.read_bytes())
        blob = {"capture": raw["capture"], "ref": raw["ref"], "sample_id": base["sample_id"]}
        for spec in specs:
            if len(p1_pool) >= goal:
                break
            funnel["attempt"] += 1
            tag = f"p33_{base['physical_root']}_{spec.family}_dx{spec.target_dx_mm:g}_dz{spec.target_dz_mm:g}"
            built = build_small_error_ic(env, blob, spec, cfg=cfg)
            if not built.ok:
                funnel[f"build_fail:{built.reason}"] += 1
                continue
            funnel["build_ok"] += 1
            if not built.log.get("precision_required"):
                funnel["not_precision_required"] += 1
                continue
            funnel["precision_required"] += 1
            fp = built.capture_fp or capture_fingerprint(built.capture)
            if fp in seen_fp:
                funnel["dup_fp"] += 1
                continue
            seen_fp.add(fp)

            clone = {"capture": clone_capture(built.capture), "ref": built.ref, "sample_id": tag}
            elig = check_d2_eligible(env, clone, cfg=cfg)
            if not elig.eligible:
                funnel[f"elig_fail:{elig.reason}"] += 1
                continue
            funnel["eligible"] += 1

            frozen = {"capture": built.capture, "ref": built.ref, "sample_id": tag}
            res = run_precision(env, frozen, "D", cfg)
            hand = res.handoff or {}
            executed = res.status == "PRECISION_EXECUTED_SUCCESS"
            skipped = res.status == "PRECISION_SKIPPED_DIRECT_SUCCESS"
            if res.status not in ("PRECISION_EXECUTED_SUCCESS", "PRECISION_SKIPPED_DIRECT_SUCCESS"):
                funnel[f"oracle_fail:{res.status}"] += 1
                continue
            if skipped or not executed:
                funnel["oracle_skipped"] += 1
                # still classify — likely P0
            else:
                funnel["oracle_executed"] += 1

            acts = list(res.actions or [])
            cls_info = classify_candidate(env, built.capture, built.ref, acts, cfg)
            funnel[f"class_{cls_info['class']}"] += 1

            # Lineage: donor root + quantized offsets (independent from P32 eval ICs)
            lineage_id = f"P33__{base['physical_root']}__dx{spec.target_dx_mm:g}__dz{spec.target_dz_mm:g}"
            if base["physical_root"] in hold or any(h in lineage_id for h in hold if "ball_train_0027" in h or "v2_0168" in h):
                # exclude donors that collide with known eval roots when offsets recreate them — soft check
                pass
            if base["physical_root"] in ("ball_train_0027", "ball_train_v2_0168"):
                funnel["excluded_eval_donor"] += 1
                continue

            row = {
                "tag": tag,
                "lineage_id": lineage_id,
                "physical_root": base["physical_root"],
                "donor_id": base["sample_id"],
                "family": spec.family,
                "target_dx_mm": spec.target_dx_mm,
                "target_dz_mm": spec.target_dz_mm,
                "capture_fp": fp,
                "oracle_status": res.status,
                "n_actions": len(acts),
                **cls_info,
                "post_dx_mm": built.log.get("post_settle_dx_mm"),
                "post_dz_mm": built.log.get("post_settle_dz_mm"),
            }
            gen_rows.append(row)

            if cls_info["class"] == "P1":
                # independence: unique lineage_id (root×offset)
                if lineage_id in existing_roots or any(x["lineage_id"] == lineage_id for x in p1_pool):
                    funnel["dup_lineage"] += 1
                    continue
                ic_path = ICS / f"{tag}.pkl"
                ic_path.write_bytes(
                    pickle.dumps(
                        {
                            "capture": built.capture,
                            "ref": built.ref,
                            "actions": np.asarray(acts, np.float32),
                            "lineage_id": lineage_id,
                            "physical_root": base["physical_root"],
                            "construction": built.log,
                            "oracle_status": res.status,
                            "phases": res.phases,
                            "class": "P1",
                            "tag": tag,
                        },
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                )
                row["ic_path"] = str(ic_path.relative_to(ROOT)).replace("\\", "/")
                p1_pool.append(row)
                print(f"  P1 #{len(p1_pool)} {tag} n_act={len(acts)}", flush=True)
            elif cls_info["class"] == "P0":
                p0_pool.append(row)

    # Split P1 by physical_root into train/val/test
    by_root = defaultdict(list)
    for r in p1_pool:
        by_root[r["physical_root"]].append(r)
    roots = list(by_root.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(roots)
    # Assign whole roots to splits
    n_roots = len(roots)
    n_test_r = max(1, int(round(n_roots * 0.2))) if n_roots >= 5 else max(0, min(1, n_roots // 3))
    n_val_r = max(1, int(round(n_roots * 0.2))) if n_roots >= 5 else max(0, min(1, n_roots // 3))
    test_roots = set(roots[:n_test_r])
    val_roots = set(roots[n_test_r : n_test_r + n_val_r])
    train_roots = set(roots[n_test_r + n_val_r :])

    def assign(r):
        if r["physical_root"] in test_roots:
            return "test"
        if r["physical_root"] in val_roots:
            return "val"
        return "train"

    for r in p1_pool:
        r["split"] = assign(r)

    train_p1 = [r for r in p1_pool if r["split"] == "train"]
    val_p1 = [r for r in p1_pool if r["split"] == "val"]
    test_p1 = [r for r in p1_pool if r["split"] == "test"]

    _write_csv(OUT / "GENERATION_ATTEMPTS.csv", gen_rows)
    _write_csv(OUT / "GENERATED_P1.csv", p1_pool)
    _write_csv(OUT / "GENERATED_P0_SIDE.csv", p0_pool[:200])

    # Manifest for Phase C
    manifest = {
        "generated_at": _utc(),
        "seed": SEED,
        "target_p1": target_p1,
        "p1_total": len(p1_pool),
        "p1_train": len(train_p1),
        "p1_val": len(val_p1),
        "p1_test": len(test_p1),
        "unique_train_roots": len({r["physical_root"] for r in train_p1}),
        "unique_val_roots": len({r["physical_root"] for r in val_p1}),
        "unique_test_roots": len({r["physical_root"] for r in test_p1}),
        "acceptance_p1_per_attempt": len(p1_pool) / max(1, funnel["attempt"]),
        "funnel": dict(funnel),
        "train": train_p1,
        "val": val_p1,
        "test": test_p1,
        "existing_audit_p1_roots": sorted(existing_roots),
    }
    (OUT / "GENERATED_DATASET.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # Coverage by regime
    regime = Counter((r["family"], r.get("target_dx_mm"), r.get("target_dz_mm")) for r in p1_pool)

    report = f"""# GENERATION_REPORT — P33 Phase B

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Method

- Donor: Stage-1 verified `A_STABLE_DIRECT_HANDOFF` grasps (holdout/eval donors excluded).
- Construction: existing `build_small_error_ic` (no teleport after freeze).
- Gate: `check_d2_eligible` then production `run_precision(..., plan_id=\"D\")`.
- Classification: immediate HANDOFF STRICT vs Oracle tape STRICT → keep **P1** only as genuine Precision-required.
- Independence: unique `(physical_root × dx × dz)` lineage; duplicate `capture_fp` rejected.
- Split by **physical_root** before any training.

## Yield

| Metric | Value |
|---|---:|
| Attempts | {funnel['attempt']} |
| Build OK | {funnel['build_ok']} |
| Precision-required | {funnel['precision_required']} |
| D2-eligible | {funnel['eligible']} |
| Oracle executed | {funnel['oracle_executed']} |
| **P1 accepted** | **{len(p1_pool)}** |
| P1 acceptance / attempt | {len(p1_pool)/max(1,funnel['attempt']):.4f} |
| P0 side-product | {len(p0_pool)} |

### Funnel detail
{json.dumps(dict(funnel), indent=2)}

## Splits (root-disjoint)

| Split | P1 ICs | Unique roots |
|---|---:|---:|
| train | {len(train_p1)} | {len({r['physical_root'] for r in train_p1})} |
| val | {len(val_p1)} | {len({r['physical_root'] for r in val_p1})} |
| test | {len(test_p1)} | {len({r['physical_root'] for r in test_p1})} |

Pre-expansion audited P1 train roots: {len(existing_roots)}.

## Regime coverage (P1 family×offset counts)

| family / dx / dz | n |
|---|---:|
"""
    for k, n in sorted(regime.items(), key=lambda x: -x[1])[:40]:
        report += f"| {k} | {n} |\n"

    bottleneck = "ok"
    if len(p1_pool) < 25:
        bottleneck = "low_p1_yield — IC construction or eligibility dominates (see funnel)"
    elif len(train_p1) < 25:
        bottleneck = "split left fewer than 25 train P1 — need more roots"

    report += f"""
## Bottleneck

**{bottleneck}**

Targets 25/50/100/200: achieved total P1={len(p1_pool)} (train={len(train_p1)}).
Val/test targets 40/40: achieved val={len(val_p1)} test={len(test_p1)}.

Oracle behavior and STRICT criteria were **not** relaxed.
P32's 17 D2 val ICs were not modified or used as donors when identifiable.
"""
    _write_md(OUT / "GENERATION_REPORT.md", report)
    print("=== Phase B done ===", f"P1={len(p1_pool)} train={len(train_p1)}", flush=True)
    return manifest


if __name__ == "__main__":
    # Staged: try for 100 first; report actual yield
    run_phase_b(target_p1=100, val_n=40, test_n=40)
