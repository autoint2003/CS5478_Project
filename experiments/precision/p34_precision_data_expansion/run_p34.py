"""P34 — Scalable Precision-required data generation (no SFT / RL).

Throughput optimizations (evaluator unchanged):
  - Immediate HANDOFF gate before Oracle (skip Oracle on P0).
  - Use Oracle handoff SUCCESS_NEW (no full re-replay of 1.7–2k steps).
  - Reuse one MuJoCo env per worker; restore dynamics defaults between ICs.
  - Resumable manifests + atomic pickle writes.
  - Optional ProcessPool workers with deterministic seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/diagnostics/p34_precision_data_expansion"
ICS_P1 = OUT / "generated_ics/p1"
ICS_P0 = OUT / "generated_ics/p0"
ICS_CONT_ONLY = OUT / "generated_ics/continuous_only"
P33 = ROOT / "results/diagnostics/p33_precision_learnability"
SEED = 34
M111_META = (
    ROOT
    / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
)
HANDOFF_TOKEN = 111  # M111: 111 motion prototypes + HANDOFF


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


def _atomic_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(tmp, path)


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Holdouts / P33 freeze
# ---------------------------------------------------------------------------
def load_holdout() -> set[str]:
    hold = {
        "A_STABLE_DIRECT_HANDOFF__A_STAGE1__ball_train_0027",
        "A_STABLE_DIRECT_HANDOFF__A_STAGE1__ball_train_v2_0168",
        "ball_train_0027",
        "ball_train_v2_0168",
    }
    man = P33 / "DATASET_MANIFEST.csv"
    if man.is_file():
        with man.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("holdout_eval") in ("1", "True", "true"):
                    hold.add(str(r.get("lineage_id") or ""))
                    hold.add(str(r.get("physical_root") or ""))
    return {h for h in hold if h}


def load_p33_p1() -> list[dict]:
    """Immutable references to P33 accepted P1 (do not copy/mutate pickles)."""
    rows = []
    js = P33 / "GENERATED_DATASET.json"
    if not js.is_file():
        return rows
    data = json.loads(js.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        for r in data.get(split) or []:
            rows.append(
                {
                    **r,
                    "source": "P33",
                    "split": r.get("split") or split,
                    "class": "P1",
                    "m111_trainable": True,  # audited in P33/P18 path; re-checked optionally
                }
            )
    return rows


def discover_bases(rng: np.random.Generator) -> list[dict]:
    hold = load_holdout()
    folder = ROOT / "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF"
    by_root: dict[str, dict] = {}
    if folder.is_dir():
        for p in sorted(folder.glob("*.pkl")):
            stem = p.stem
            root = stem.split("__")[-1]
            if root in hold or stem in hold:
                continue
            if root in ("ball_train_0027", "ball_train_v2_0168"):
                continue
            by_root.setdefault(root, {"sample_id": stem, "path": str(p), "physical_root": root})
    bases = list(by_root.values())
    rng.shuffle(bases)
    return bases


# ---------------------------------------------------------------------------
# Single-candidate pipeline (worker-safe)
# ---------------------------------------------------------------------------
_WORKER = {"env": None, "defaults": None, "cfg": None}


def _init_worker(seed: int = 0) -> None:
    from data_generation.oracles.precision.state import PrecisionConfig
    from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
    from experiments.precision.p34_precision_data_expansion.builder import snapshot_default_dynamics

    # Parallel MjModel.from_xml_path can race on Windows include resolution → "empty file".
    env = None
    last_exc: Exception | None = None
    for attempt in range(25):
        try:
            env = make_env()
            break
        except ValueError as exc:
            last_exc = exc
            if "ParseXML" in str(exc) or "empty file" in str(exc):
                time.sleep(0.15 * (attempt + 1) + 0.05 * (seed % 7))
                continue
            raise
        except Exception as exc:
            last_exc = exc
            time.sleep(0.2 * (attempt + 1))
    if env is None:
        raise RuntimeError(f"make_env failed after retries: {last_exc}")
    _WORKER["env"] = env
    _WORKER["defaults"] = snapshot_default_dynamics(env.sim)
    _WORKER["cfg"] = PrecisionConfig(evaluate_handoff=True)
    _WORKER["seed"] = seed


def _immediate_handoff(env, capture, ref, defaults) -> tuple[bool, dict, float]:
    from evaluation.axis_aware_success import evaluate_handoff_with_axis
    from envs.observations.observable_obs import ObservableObsState
    from experiments.precision.p34_precision_data_expansion.builder import restore_ic
    from models.transformer_policy import DT_POLICY

    t0 = time.perf_counter()
    restore_ic(env, capture, defaults=defaults)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    return bool(hand.get("SUCCESS_NEW")), hand, time.perf_counter() - t0


def process_candidate(task: dict) -> dict[str, Any]:
    """Build → imm HANDOFF → (optional) Oracle → classify. Timings in seconds."""
    from data_generation.oracles.precision.d2_eligibility import check_d2_eligible
    from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint
    from data_generation.oracles.precision.state_machine import run_precision
    from experiments.precision.p13_teleport_d2.teleport_builder import clone_capture
    from experiments.precision.p34_precision_data_expansion.builder import (
        build_regime_ic,
        restore_default_dynamics,
    )
    from experiments.precision.p34_precision_data_expansion.regimes import RegimeSpec

    if _WORKER["env"] is None:
        _init_worker(int(task.get("worker_seed") or 0))
    env = _WORKER["env"]
    defaults = _WORKER["defaults"]
    cfg = _WORKER["cfg"]

    timings = Counter()
    t_all = time.perf_counter()
    row: dict[str, Any] = {
        "tag": task["tag"],
        "lineage_id": task["lineage_id"],
        "physical_root": task["physical_root"],
        "donor_id": task["donor_id"],
        "regime_id": task["regime_id"],
        "family": task["family"],
        "bucket": task.get("bucket"),
        "target_dx_mm": task.get("target_dx_mm"),
        "target_dy_mm": task.get("target_dy_mm"),
        "target_dz_mm": task.get("target_dz_mm"),
        "dtheta_deg": task.get("dtheta_deg"),
        "mass_scale": task.get("mass_scale"),
        "friction_scale": task.get("friction_scale"),
        "worker_id": task.get("worker_id"),
    }
    try:
        path = Path(task["donor_path"])
        raw = pickle.loads(path.read_bytes())
        blob = {"capture": raw["capture"], "ref": raw["ref"], "sample_id": task["donor_id"]}
        spec = RegimeSpec(
            regime_id=task["regime_id"],
            family=task["family"],
            target_dx_mm=float(task.get("target_dx_mm") or 0),
            target_dy_mm=float(task.get("target_dy_mm") or 0),
            target_dz_mm=float(task.get("target_dz_mm") or 0),
            dtheta_deg=float(task.get("dtheta_deg") or 0),
            v_lin_hand_mms=tuple(task.get("v_lin_hand_mms") or (0.0, 0.0, 0.0)),
            w_hand_dps=tuple(task.get("w_hand_dps") or (0.0, 0.0, 0.0)),
            mass_scale=float(task.get("mass_scale") or 1.0),
            friction_scale=float(task.get("friction_scale") or 1.0),
            bucket=str(task.get("bucket") or "explore"),
        )

        t0 = time.perf_counter()
        built = build_regime_ic(env, blob, spec, cfg=cfg, defaults=defaults)
        timings["build_s"] = time.perf_counter() - t0
        if not built.ok:
            row.update(
                {
                    "ok": False,
                    "reject": f"build:{built.reason}",
                    "class": "PU",
                    "timings": dict(timings),
                    "wall_s": time.perf_counter() - t_all,
                }
            )
            restore_default_dynamics(env.sim, defaults)
            return row
        if not built.log.get("precision_required"):
            row.update(
                {
                    "ok": False,
                    "reject": "not_precision_required",
                    "class": "PU",
                    "timings": dict(timings),
                    "wall_s": time.perf_counter() - t_all,
                    **{k: built.log.get(k) for k in ("post_settle_dx_mm", "post_settle_dz_mm", "fsm_case")},
                }
            )
            restore_default_dynamics(env.sim, defaults)
            return row

        fp = built.capture_fp or capture_fingerprint(built.capture)
        row["capture_fp"] = fp
        if fp in set(task.get("seen_fps") or []):
            row.update(
                {
                    "ok": False,
                    "reject": "dup_fp",
                    "class": "PU",
                    "timings": dict(timings),
                    "wall_s": time.perf_counter() - t_all,
                }
            )
            restore_default_dynamics(env.sim, defaults)
            return row

        clone = {"capture": clone_capture(built.capture), "ref": built.ref, "sample_id": task["tag"]}
        # Apply pre-freeze dynamics override so eligibility matches frozen IC physics.
        from experiments.precision.p34_precision_data_expansion.builder import (
            DynamicsOverride,
            apply_dynamics_override,
        )

        restore_default_dynamics(env.sim, defaults)
        apply_dynamics_override(
            env.sim, DynamicsOverride.from_dict((built.capture or {}).get("dynamics_override"))
        )
        t0 = time.perf_counter()
        elig = check_d2_eligible(env, clone, cfg=cfg)
        timings["elig_s"] = time.perf_counter() - t0
        if not elig.eligible:
            row.update(
                {
                    "ok": False,
                    "reject": f"elig:{elig.reason}",
                    "class": "PU",
                    "timings": dict(timings),
                    "wall_s": time.perf_counter() - t_all,
                }
            )
            restore_default_dynamics(env.sim, defaults)
            return row

        imm_ok, imm_hand, imm_s = _immediate_handoff(env, built.capture, built.ref, defaults)
        timings["imm_handoff_s"] = imm_s
        row["imm_SUCCESS_NEW"] = int(imm_ok)
        row["imm_reason"] = imm_hand.get("std_reason") or imm_hand.get("reason")

        # Throughput: if immediate HANDOFF succeeds → P0 (or P2 if we later ran Oracle).
        # Skip Oracle for P0 collection path unless forced.
        if imm_ok and not task.get("force_oracle_on_p0"):
            row.update(
                {
                    "ok": True,
                    "class": "P0",
                    "oracle_skipped_for_throughput": 1,
                    "prec_SUCCESS_NEW": None,
                    "n_actions": 0,
                    "oracle_status": "SKIPPED_IMM_OK",
                    "construction": built.log,
                    "timings": dict(timings),
                    "wall_s": time.perf_counter() - t_all,
                    "capture": built.capture,
                    "ref": built.ref,
                    "actions": None,
                    "phases": [],
                }
            )
            restore_default_dynamics(env.sim, defaults)
            return row

        # Immediate failed (or forced) → run Precision Oracle
        frozen = {"capture": built.capture, "ref": built.ref, "sample_id": task["tag"]}
        t0 = time.perf_counter()
        res = run_precision(env, frozen, "D", cfg)
        timings["oracle_s"] = time.perf_counter() - t0
        acts = list(res.actions or [])
        hand = res.handoff or {}
        prec_ok = bool(hand.get("SUCCESS_NEW"))
        # Fallback: if Oracle did not attach handoff, do NOT invent success
        if res.handoff is None and res.status == "PRECISION_EXECUTED_SUCCESS":
            # rare: trust status only if evaluate_handoff path marked success
            prec_ok = True
        executed = res.status == "PRECISION_EXECUTED_SUCCESS"
        skipped = res.status == "PRECISION_SKIPPED_DIRECT_SUCCESS"

        if imm_ok and prec_ok:
            cls = "P2"
        elif imm_ok and not prec_ok:
            cls = "P0"
        elif (not imm_ok) and prec_ok and executed:
            cls = "P1"
        elif (not imm_ok) and skipped and prec_ok:
            # skipped but imm failed — inconsistent / PU
            cls = "PU"
        elif (not imm_ok) and not prec_ok:
            cls = "P3"
        else:
            cls = "PU"

        # Nonzero Oracle actions alone do NOT make P1
        if cls == "P1" and (imm_ok or not prec_ok or not executed):
            cls = "PU"

        row.update(
            {
                "ok": True,
                "class": cls,
                "oracle_status": res.status,
                "prec_SUCCESS_NEW": int(prec_ok) if res.handoff is not None else None,
                "prec_reason": hand.get("std_reason") or hand.get("reason"),
                "n_actions": len(acts),
                "oracle_executed": int(executed),
                "oracle_skipped": int(skipped),
                "phases": list(res.phases or []),
                "construction": built.log,
                "timings": dict(timings),
                "wall_s": time.perf_counter() - t_all,
                "sim_steps_oracle": len(acts),
                "capture": built.capture if cls in ("P0", "P1", "P2") else None,
                "ref": built.ref if cls in ("P0", "P1", "P2") else None,
                "actions": np.asarray(acts, np.float32) if acts and cls in ("P1", "P2") else None,
                "errors_entry": res.errors_entry,
                "errors_exit": res.errors_exit,
                "pre_handoff_state": {
                    "errors_exit": res.errors_exit,
                    "phases_tail": list(res.phases or [])[-5:],
                },
            }
        )
        restore_default_dynamics(env.sim, defaults)
        return row
    except Exception as exc:
        try:
            if _WORKER.get("defaults") is not None:
                restore_default_dynamics(env.sim, _WORKER["defaults"])
        except Exception:
            pass
        row.update(
            {
                "ok": False,
                "reject": f"exception:{type(exc).__name__}:{exc}",
                "class": "PU",
                "traceback": traceback.format_exc()[-800:],
                "timings": dict(timings),
                "wall_s": time.perf_counter() - t_all,
            }
        )
        return row


# ---------------------------------------------------------------------------
# Persist accepted samples
# ---------------------------------------------------------------------------
def save_accepted(row: dict, *, partition: str) -> str | None:
    """Write IC pickle; strip heavy fields from row; return relative path."""
    cap = row.pop("capture", None)
    ref = row.pop("ref", None)
    acts = row.pop("actions", None)
    if cap is None or ref is None:
        return None
    if partition == "p1":
        dest_dir = ICS_P1
    elif partition == "p0":
        dest_dir = ICS_P0
    elif partition == "continuous_only":
        dest_dir = ICS_CONT_ONLY
    else:
        dest_dir = OUT / "generated_ics" / partition
    tag = row["tag"]
    path = dest_dir / f"{tag}.pkl"
    blob = {
        "capture": cap,
        "ref": ref,
        "actions": acts,
        "lineage_id": row["lineage_id"],
        "physical_root": row["physical_root"],
        "construction": row.get("construction"),
        "oracle_status": row.get("oracle_status"),
        "phases": row.get("phases"),
        "class": row.get("class"),
        "tag": tag,
        "regime_id": row.get("regime_id"),
        "family": row.get("family"),
        "source": "P34",
        "errors_entry": row.get("errors_entry"),
        "errors_exit": row.get("errors_exit"),
        "pre_handoff_state": row.get("pre_handoff_state"),
        "imm_SUCCESS_NEW": row.get("imm_SUCCESS_NEW"),
        "prec_SUCCESS_NEW": row.get("prec_SUCCESS_NEW"),
    }
    _atomic_pickle(path, blob)
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    row["ic_path"] = rel
    return rel


# ---------------------------------------------------------------------------
# M111 audit
# ---------------------------------------------------------------------------
def m111_replay_check(ic_path: Path) -> dict[str, Any]:
    from experiments.precision.p18_discrete_vocab_audit.replay_harness import (
        make_env,
        quantize_with_book,
        replay_actions,
    )
    from experiments.precision.p34_precision_data_expansion.builder import (
        restore_default_dynamics,
        snapshot_default_dynamics,
    )
    from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, load_motion_codebook

    blob = pickle.loads(ic_path.read_bytes())
    acts = blob.get("actions")
    if acts is None or len(np.asarray(acts).reshape(-1, 7)) == 0:
        return {"ic_path": str(ic_path), "ok": False, "reason": "no_actions"}
    acts = np.asarray(acts, np.float32).reshape(-1, 7)
    m111 = load_motion_codebook(M111_META)
    book = m111.prototypes
    rho = m111.residual
    q = quantize_with_book(book, acts, rho, w=DEFAULT_WEIGHTS)
    recon = np.asarray(q["recon_actions"], np.float32).reshape(-1, 7)
    tokens = np.asarray(q["tokens"], np.int64).reshape(-1)
    # Append explicit HANDOFF marker (not executed as motion)
    env = make_env()
    defaults = snapshot_default_dynamics(env.sim)
    # continuous already validated at accept; re-check M111 recon
    from experiments.precision.p34_precision_data_expansion.builder import restore_ic

    restore_ic(env, blob["capture"], defaults=defaults)
    h = replay_actions(env, blob["capture"], blob["ref"], recon, stabilize_n=8)
    # Note: replay_actions restores again internally
    restore_default_dynamics(env.sim, defaults)
    max_abs_err = float(np.max(np.abs(recon - acts))) if len(acts) else 0.0
    return {
        "ic_path": str(ic_path.relative_to(ROOT)).replace("\\", "/"),
        "tag": blob.get("tag"),
        "lineage_id": blob.get("lineage_id"),
        "n_actions": int(len(acts)),
        "n_tokens": int(len(tokens)),
        "m111_SUCCESS_NEW": int(bool(h.get("SUCCESS_NEW"))),
        "m111_reason": h.get("reason"),
        "max_abs_recon_err": max_abs_err,
        "unique_tokens": int(len(set(tokens.tolist()))),
        "handoff_token_id": HANDOFF_TOKEN,
        "ok": bool(h.get("SUCCESS_NEW")),
    }


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def run_profile(n_candidates: int = 12, workers: int = 1) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    from experiments.precision.p34_precision_data_expansion.regimes import (
        adaptive_schedule,
        build_regime_catalog,
    )

    rng = np.random.default_rng(SEED)
    bases = discover_bases(rng)
    catalog = build_regime_catalog()
    schedule = adaptive_schedule(catalog, regime_p1_counts={}, rng=rng)
    tasks = []
    for i in range(n_candidates):
        base = bases[i % len(bases)]
        spec = schedule[i % len(schedule)]
        lineage_id = (
            f"P34__{base['physical_root']}__{spec.regime_id}"
        )
        tag = f"p34_{base['physical_root']}_{spec.regime_id}"
        tasks.append(
            {
                "tag": tag,
                "lineage_id": lineage_id,
                "physical_root": base["physical_root"],
                "donor_id": base["sample_id"],
                "donor_path": base["path"],
                "regime_id": spec.regime_id,
                "family": spec.family,
                "bucket": spec.bucket,
                "target_dx_mm": spec.target_dx_mm,
                "target_dy_mm": spec.target_dy_mm,
                "target_dz_mm": spec.target_dz_mm,
                "dtheta_deg": spec.dtheta_deg,
                "v_lin_hand_mms": list(spec.v_lin_hand_mms),
                "w_hand_dps": list(spec.w_hand_dps),
                "mass_scale": spec.mass_scale,
                "friction_scale": spec.friction_scale,
                "seen_fps": [],
                "worker_id": 0,
                "worker_seed": SEED,
                "force_oracle_on_p0": False,
            }
        )

    # Baseline-style: force oracle even when imm ok (P33-like cost) on half
    baseline_tasks = [{**t, "force_oracle_on_p0": True, "tag": t["tag"] + "_base"} for t in tasks[: max(2, n_candidates // 3)]]
    opt_tasks = tasks

    _init_worker(SEED)
    base_rows = []
    t0 = time.perf_counter()
    for t in baseline_tasks:
        base_rows.append(process_candidate(t))
    base_wall = time.perf_counter() - t0

    opt_rows = []
    t0 = time.perf_counter()
    for t in opt_tasks:
        opt_rows.append(process_candidate(t))
    opt_wall = time.perf_counter() - t0

    def _sum_stage(rows, key):
        return sum(float((r.get("timings") or {}).get(key) or 0) for r in rows)

    def _cls(rows):
        return dict(Counter(r.get("class") for r in rows))

    report = f"""# THROUGHPUT_PROFILE — P34

Generated: {_utc()}

## Setup

- Donors: Stage-1 `A_STABLE_DIRECT_HANDOFF` (holdouts excluded)
- Workers (profile): 1 (serial microbench)
- Oracle / STRICT / M111: **unchanged**
- Optimization under test: **immediate-HANDOFF gate before Oracle** + **no full re-replay**

## Baseline-like (force Oracle even if immediate HANDOFF succeeds)

| Metric | Value |
|---|---:|
| Candidates | {len(baseline_tasks)} |
| Wall_s | {base_wall:.1f} |
| Mean wall_s / cand | {base_wall/max(1,len(baseline_tasks)):.1f} |
| Sum build_s | {_sum_stage(base_rows,'build_s'):.1f} |
| Sum imm_handoff_s | {_sum_stage(base_rows,'imm_handoff_s'):.1f} |
| Sum oracle_s | {_sum_stage(base_rows,'oracle_s'):.1f} |
| Class counts | {_cls(base_rows)} |

## Optimized (skip Oracle when immediate HANDOFF succeeds)

| Metric | Value |
|---|---:|
| Candidates | {len(opt_tasks)} |
| Wall_s | {opt_wall:.1f} |
| Mean wall_s / cand | {opt_wall/max(1,len(opt_tasks)):.1f} |
| Sum build_s | {_sum_stage(opt_rows,'build_s'):.1f} |
| Sum imm_handoff_s | {_sum_stage(opt_rows,'imm_handoff_s'):.1f} |
| Sum oracle_s | {_sum_stage(opt_rows,'oracle_s'):.1f} |
| Sum elig_s | {_sum_stage(opt_rows,'elig_s'):.1f} |
| Class counts | {_cls(opt_rows)} |
| P1 accepted | {sum(1 for r in opt_rows if r.get('class')=='P1')} |
| P0 (Oracle skipped) | {sum(1 for r in opt_rows if r.get('oracle_skipped_for_throughput'))} |

## Interpretation

- Failed build / not_precision_required / eligibility rejects are cheap relative to Oracle.
- Oracle dominates whenever immediate HANDOFF fails (true P1/P3 path).
- Skipping Oracle on immediate-success candidates removes the largest avoidable cost from P33's classify-everything pattern.
- Parallel workers multiply throughput approximately linearly until CPU/memory saturate (see generation run).

## Safety

Both physical counterfactuals remain mandatory for **P1**:
1. Immediate HANDOFF fails STRICT
2. Precision Oracle + explicit HANDOFF passes STRICT
"""
    _write_md(OUT / "THROUGHPUT_PROFILE.md", report)
    summary = {
        "baseline_wall_s": base_wall,
        "baseline_n": len(baseline_tasks),
        "opt_wall_s": opt_wall,
        "opt_n": len(opt_tasks),
        "opt_class": _cls(opt_rows),
        "base_class": _cls(base_rows),
    }
    _atomic_json(OUT / "throughput_profile.json", summary)
    print("PROFILE done", summary, flush=True)
    return summary


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _resume_state() -> dict:
    seen_fp: set[str] = set()
    seen_lin: set[str] = set()
    p1_rows: list[dict] = []
    p0_rows: list[dict] = []
    attempts: list[dict] = []
    # P33 lineages
    for r in load_p33_p1():
        seen_lin.add(r["lineage_id"])
        p1_rows.append(r)
        # fingerprint from disk if present
        ip = r.get("ic_path")
        if ip and (ROOT / ip).is_file():
            try:
                b = pickle.loads((ROOT / ip).read_bytes())
                from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint

                seen_fp.add(capture_fingerprint(b["capture"]))
            except Exception:
                pass
    if ICS_P1.is_dir():
        for p in ICS_P1.glob("*.pkl"):
            b = pickle.loads(p.read_bytes())
            lin = b.get("lineage_id")
            if lin in seen_lin:
                continue
            from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint

            fp = capture_fingerprint(b["capture"])
            seen_fp.add(fp)
            seen_lin.add(lin)
            p1_rows.append(
                {
                    "tag": b.get("tag") or p.stem,
                    "lineage_id": lin,
                    "physical_root": b.get("physical_root"),
                    "n_actions": int(len(np.asarray(b.get("actions")).reshape(-1, 7)))
                    if b.get("actions") is not None
                    else 0,
                    "class": "P1",
                    "ic_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "oracle_status": b.get("oracle_status"),
                    "regime_id": b.get("regime_id"),
                    "family": b.get("family"),
                    "source": "P34",
                }
            )
    if ICS_P0.is_dir():
        for p in ICS_P0.glob("*.pkl"):
            b = pickle.loads(p.read_bytes())
            p0_rows.append(
                {
                    "tag": b.get("tag") or p.stem,
                    "lineage_id": b.get("lineage_id"),
                    "physical_root": b.get("physical_root"),
                    "class": "P0",
                    "ic_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "source": "P34",
                    "regime_id": b.get("regime_id"),
                }
            )
            seen_lin.add(b.get("lineage_id"))
    att_path = OUT / "GENERATION_ATTEMPTS.csv"
    if att_path.is_file() and att_path.stat().st_size > 0:
        with att_path.open(encoding="utf-8") as f:
            attempts = list(csv.DictReader(f))
    return {
        "seen_fp": seen_fp,
        "seen_lin": seen_lin,
        "p1_rows": p1_rows,
        "p0_rows": p0_rows,
        "attempts": attempts,
    }


def assign_splits(p1_rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """Preserve P33 split assignments; allocate new roots toward ~160/20/20."""
    # Freeze P33 assignments
    for r in p1_rows:
        if r.get("source") == "P33" and r.get("split"):
            continue
        if r.get("source") == "P33":
            r["split"] = "train"  # fallback
    # Existing root → split map from P33
    root_split: dict[str, str] = {}
    for r in p1_rows:
        if r.get("source") == "P33" and r.get("split"):
            root_split[r["physical_root"]] = r["split"]

    # Count current
    def counts():
        c = Counter(r.get("split") for r in p1_rows if r.get("split"))
        return c

    # New P34 rows without split: assign by physical_root, prefer train
    new_roots = sorted(
        {
            r["physical_root"]
            for r in p1_rows
            if r.get("source") == "P34" and r["physical_root"] not in root_split
        }
    )
    rng.shuffle(new_roots)
    target = {"train": 160, "val": 20, "test": 20}
    # Seeds from P33
    for r in p1_rows:
        if r.get("split"):
            continue
        pr = r["physical_root"]
        if pr in root_split:
            r["split"] = root_split[pr]
            continue
        # decide split for this root
        cur = counts()
        # Prefer filling val/test if under target AND we still have headroom on train
        n_total = sum(cur.values()) + 1
        # heuristic: keep ~10% val, ~10% test of projected 200
        if cur.get("val", 0) < target["val"] and cur.get("val", 0) <= cur.get("test", 0):
            # only if train already has reasonable share
            if cur.get("train", 0) >= max(10, int(0.7 * n_total)):
                root_split[pr] = "val"
            else:
                root_split[pr] = "train"
        elif cur.get("test", 0) < target["test"] and cur.get("train", 0) >= max(10, int(0.7 * n_total)):
            root_split[pr] = "test"
        else:
            root_split[pr] = "train"
        r["split"] = root_split[pr]
    # Fill any remaining
    for r in p1_rows:
        if not r.get("split"):
            r["split"] = root_split.get(r["physical_root"], "train")
    return p1_rows


def run_generation(
    target_total_p1: int = 200,
    workers: int = 3,
    max_attempts: int = 8000,
    p0_target: int = 80,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ICS_P1.mkdir(parents=True, exist_ok=True)
    ICS_P0.mkdir(parents=True, exist_ok=True)
    ICS_CONT_ONLY.mkdir(parents=True, exist_ok=True)

    from experiments.precision.p34_precision_data_expansion.regimes import (
        adaptive_schedule,
        build_regime_catalog,
    )

    rng = np.random.default_rng(SEED)
    state = _resume_state()
    seen_fp: set[str] = set(state["seen_fp"])
    seen_lin: set[str] = set(state["seen_lin"])
    p1_rows: list[dict] = list(state["p1_rows"])
    p0_rows: list[dict] = list(state["p0_rows"])
    attempt_rows: list[dict] = []
    regime_stats: dict[str, Counter] = defaultdict(Counter)

    bases = discover_bases(rng)
    catalog = build_regime_catalog()
    print(
        f"=== P34 generate: have_p1={len(p1_rows)} target={target_total_p1} "
        f"donors={len(bases)} regimes={len(catalog)} workers={workers} ===",
        flush=True,
    )
    if not bases:
        _write_md(OUT / "GENERATION_REPORT.md", f"# GENERATION_REPORT\n\nNo donors. {_utc()}\n")
        return {"p1": 0}

    t_start = time.time()
    attempts = 0

    # --- Cheap P0 harvest from Direct-HANDOFF donors (no Oracle) ---
    if len(p0_rows) < p0_target:
        print(f"=== P0 harvest (target={p0_target}, have={len(p0_rows)}) ===", flush=True)
        _init_worker(SEED)
        from experiments.precision.p34_precision_data_expansion.builder import (
            restore_default_dynamics,
            snapshot_default_dynamics,
        )

        env = _WORKER["env"]
        defaults = _WORKER["defaults"]
        for base in bases:
            if len(p0_rows) >= p0_target:
                break
            lin = f"P34P0__{base['physical_root']}__donor"
            if lin in seen_lin:
                continue
            raw = pickle.loads(Path(base["path"]).read_bytes())
            imm_ok, imm_hand, _ = _immediate_handoff(
                env, raw["capture"], raw["ref"], defaults
            )
            if not imm_ok:
                continue
            tag = f"p34p0_{base['physical_root']}_donor"
            row = {
                "tag": tag,
                "lineage_id": lin,
                "physical_root": base["physical_root"],
                "donor_id": base["sample_id"],
                "class": "P0",
                "regime_id": "DONOR_DIRECT",
                "family": "P0_DONOR",
                "imm_SUCCESS_NEW": 1,
                "oracle_status": "NOT_RUN_P0",
                "n_actions": 0,
                "capture": raw["capture"],
                "ref": raw["ref"],
                "actions": None,
                "source": "P34",
            }
            save_accepted(row, partition="p0")
            seen_lin.add(lin)
            p0_rows.append(
                {
                    "tag": tag,
                    "lineage_id": lin,
                    "physical_root": base["physical_root"],
                    "class": "P0",
                    "ic_path": row.get("ic_path"),
                    "source": "P34",
                    "regime_id": "DONOR_DIRECT",
                    "family": "P0_DONOR",
                }
            )
            print(f"  P0 #{len(p0_rows)} {tag}", flush=True)
        restore_default_dynamics(env.sim, defaults)
        print(f"P0 harvest done: {len(p0_rows)}", flush=True)

    def checkpoint():
        slim_attempts = []
        for r in attempt_rows[-5000:]:
            slim_attempts.append({k: v for k, v in r.items() if k not in ("capture", "ref", "actions", "traceback", "construction", "phases", "errors_entry", "errors_exit", "pre_handoff_state", "timings") or k == "timings"})
            if "timings" in r:
                slim_attempts[-1]["timings"] = json.dumps(r["timings"])
        _write_csv(OUT / "GENERATION_ATTEMPTS.csv", slim_attempts if slim_attempts else attempt_rows)
        assign_splits(p1_rows, rng)
        _write_csv(OUT / "DATASET_MANIFEST.csv", [{k: v for k, v in r.items() if k != "construction"} for r in p1_rows])
        _write_csv(OUT / "P0_MANIFEST.csv", p0_rows)
        _atomic_json(
            OUT / "checkpoint.json",
            {
                "utc": _utc(),
                "p1_total": len(p1_rows),
                "p1_p34": sum(1 for r in p1_rows if r.get("source") == "P34"),
                "p0": len(p0_rows),
                "attempts": attempts,
                "elapsed_s": time.time() - t_start,
            },
        )
        print(
            f"  checkpoint P1={len(p1_rows)} (P34={sum(1 for r in p1_rows if r.get('source')=='P34')}) "
            f"P0={len(p0_rows)} attempts={attempts} elapsed_min={(time.time()-t_start)/60:.1f}",
            flush=True,
        )

    # Precompute systematic donor×regime pairs (productive-first within each donor)
    regime_p1_counts0 = Counter(
        r.get("regime_id") for r in p1_rows if r.get("source") == "P34" and r.get("regime_id")
    )
    for r in p1_rows:
        if r.get("source") == "P33" and r.get("family") is not None:
            rid_guess = (
                f"{r.get('family')}_dx{float(r.get('target_dx_mm') or 0):g}_dy0_"
                f"dz{float(r.get('target_dz_mm') or 0):g}_th0"
            )
            regime_p1_counts0[rid_guess] += 1
    schedule0 = adaptive_schedule(catalog, regime_p1_counts=dict(regime_p1_counts0), rng=rng)
    pair_queue: list[tuple[dict, Any]] = []
    for base in bases:
        for spec in schedule0:
            lineage_id = f"P34__{base['physical_root']}__{spec.regime_id}"
            if lineage_id in seen_lin:
                continue
            pair_queue.append((base, spec))
    rng.shuffle(pair_queue)  # diversify donor order; schedule already productive-biased
    # Re-stable: keep productive pairs toward the front after shuffle of donors
    pair_queue.sort(key=lambda x: 0 if x[1].bucket == "productive" else (1 if x[1].bucket == "underrepresented" else 2))
    pair_i = 0
    print(f"pair_queue={len(pair_queue)}", flush=True)

    # Serial or parallel loop
    while len(p1_rows) < target_total_p1 and attempts < max_attempts:
        need = target_total_p1 - len(p1_rows)
        batch_n = min(max(workers * 3, 6), need * 6, 24)
        tasks = []
        seen_task: set[str] = set()
        while len(tasks) < batch_n and pair_i < len(pair_queue):
            base, spec = pair_queue[pair_i]
            pair_i += 1
            lineage_id = f"P34__{base['physical_root']}__{spec.regime_id}"
            if lineage_id in seen_lin:
                continue
            tag = f"p34_{base['physical_root']}_{spec.regime_id}"
            if tag in seen_task:
                continue
            seen_task.add(tag)
            tasks.append(
                {
                    "tag": tag,
                    "lineage_id": lineage_id,
                    "physical_root": base["physical_root"],
                    "donor_id": base["sample_id"],
                    "donor_path": base["path"],
                    "regime_id": spec.regime_id,
                    "family": spec.family,
                    "bucket": spec.bucket,
                    "target_dx_mm": spec.target_dx_mm,
                    "target_dy_mm": spec.target_dy_mm,
                    "target_dz_mm": spec.target_dz_mm,
                    "dtheta_deg": spec.dtheta_deg,
                    "v_lin_hand_mms": list(spec.v_lin_hand_mms),
                    "w_hand_dps": list(spec.w_hand_dps),
                    "mass_scale": spec.mass_scale,
                    "friction_scale": spec.friction_scale,
                    "seen_fps": list(seen_fp)[-5000:],
                    "worker_id": len(tasks) % max(1, workers),
                    "worker_seed": SEED + len(tasks) + attempts,
                    "force_oracle_on_p0": False,
                }
            )

        if not tasks:
            print("No more unique lineage tasks; stopping.", flush=True)
            break

        results: list[dict] = []
        if workers <= 1:
            _init_worker(SEED)
            for t in tasks:
                results.append(process_candidate(t))
        else:
            # Windows-friendly process pool
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(SEED,),
            ) as ex:
                futs = {ex.submit(process_candidate, t): t for t in tasks}
                for fut in as_completed(futs):
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        results.append(
                            {
                                "ok": False,
                                "reject": f"future:{exc}",
                                "class": "PU",
                                "tag": futs[fut]["tag"],
                                "lineage_id": futs[fut]["lineage_id"],
                                "regime_id": futs[fut]["regime_id"],
                            }
                        )

        for row in results:
            attempts += 1
            rid = row.get("regime_id") or "unknown"
            cls = row.get("class") or "PU"
            regime_stats[rid]["attempt"] += 1
            regime_stats[rid][f"class_{cls}"] += 1
            if row.get("reject"):
                regime_stats[rid][f"reject:{row['reject'].split(':')[0]}"] += 1

            # Strip heavy blobs for CSV attempt log
            light = {
                k: v
                for k, v in row.items()
                if k
                not in (
                    "capture",
                    "ref",
                    "actions",
                    "construction",
                    "phases",
                    "errors_entry",
                    "errors_exit",
                    "pre_handoff_state",
                    "traceback",
                )
            }
            if row.get("timings"):
                light["timings"] = json.dumps(row["timings"])
            attempt_rows.append(light)

            fp = row.get("capture_fp")
            if fp:
                if fp in seen_fp and cls == "P1":
                    continue
                seen_fp.add(fp)

            if cls == "P1":
                lin = row["lineage_id"]
                if lin in seen_lin:
                    continue
                save_accepted(row, partition="p1")
                seen_lin.add(lin)
                p1_rows.append(
                    {
                        "tag": row["tag"],
                        "lineage_id": lin,
                        "physical_root": row["physical_root"],
                        "donor_id": row.get("donor_id"),
                        "n_actions": row.get("n_actions"),
                        "class": "P1",
                        "ic_path": row.get("ic_path"),
                        "oracle_status": row.get("oracle_status"),
                        "regime_id": row.get("regime_id"),
                        "family": row.get("family"),
                        "bucket": row.get("bucket"),
                        "target_dx_mm": row.get("target_dx_mm"),
                        "target_dy_mm": row.get("target_dy_mm"),
                        "target_dz_mm": row.get("target_dz_mm"),
                        "dtheta_deg": row.get("dtheta_deg"),
                        "mass_scale": row.get("mass_scale"),
                        "friction_scale": row.get("friction_scale"),
                        "imm_SUCCESS_NEW": row.get("imm_SUCCESS_NEW"),
                        "prec_SUCCESS_NEW": row.get("prec_SUCCESS_NEW"),
                        "source": "P34",
                        "wall_s": row.get("wall_s"),
                        "capture_fp": fp,
                    }
                )
                print(
                    f"  P1 #{len(p1_rows)} {row['tag']} n_act={row.get('n_actions')} "
                    f"wall={row.get('wall_s'):.1f}s",
                    flush=True,
                )
            elif cls == "P0" and len(p0_rows) < p0_target:
                lin = row["lineage_id"]
                if lin in seen_lin and any(x.get("lineage_id") == lin for x in p0_rows):
                    pass
                else:
                    # P0 may share lineage key space — use P0-specific lineage
                    row["lineage_id"] = row["lineage_id"] + "__P0"
                    if row["lineage_id"] not in seen_lin:
                        save_accepted(row, partition="p0")
                        seen_lin.add(row["lineage_id"])
                        p0_rows.append(
                            {
                                "tag": row["tag"],
                                "lineage_id": row["lineage_id"],
                                "physical_root": row["physical_root"],
                                "class": "P0",
                                "ic_path": row.get("ic_path"),
                                "source": "P34",
                                "regime_id": row.get("regime_id"),
                                "family": row.get("family"),
                                "oracle_skipped_for_throughput": row.get("oracle_skipped_for_throughput"),
                            }
                        )

        checkpoint()
        # Milestone logging
        n = len(p1_rows)
        if n >= 50:
            (OUT / "CHECKPOINT_A_50.json").write_text(
                json.dumps({"p1_total": n, "utc": _utc()}, indent=2), encoding="utf-8"
            )
        if n >= 100:
            (OUT / "CHECKPOINT_B_100.json").write_text(
                json.dumps({"p1_total": n, "utc": _utc()}, indent=2), encoding="utf-8"
            )

    assign_splits(p1_rows, rng)
    _write_csv(OUT / "DATASET_MANIFEST.csv", [{k: v for k, v in r.items() if k != "construction"} for r in p1_rows])
    _write_csv(OUT / "P0_MANIFEST.csv", p0_rows)
    _write_csv(OUT / "GENERATION_ATTEMPTS.csv", attempt_rows)

    # Regime coverage tables
    cov_rows = []
    acc_rows = []
    for rid, ctr in sorted(regime_stats.items()):
        att = int(ctr.get("attempt") or 0)
        n1 = int(ctr.get("class_P1") or 0)
        cov_rows.append(
            {
                "regime_id": rid,
                "attempts": att,
                "P0": int(ctr.get("class_P0") or 0),
                "P1": n1,
                "P2": int(ctr.get("class_P2") or 0),
                "P3": int(ctr.get("class_P3") or 0),
                "PU": int(ctr.get("class_PU") or 0),
                "p1_rate": n1 / max(1, att),
            }
        )
        acc_rows.append(
            {
                "regime_id": rid,
                "attempts": att,
                "p1_accepted": n1,
                "acceptance_rate": n1 / max(1, att),
                **{k: int(v) for k, v in ctr.items() if k.startswith("reject:")},
            }
        )
    _write_csv(OUT / "REGIME_COVERAGE.csv", cov_rows)
    _write_csv(OUT / "P1_ACCEPTANCE_BY_REGIME.csv", acc_rows)

    train = [r for r in p1_rows if r.get("split") == "train"]
    val = [r for r in p1_rows if r.get("split") == "val"]
    test = [r for r in p1_rows if r.get("split") == "test"]
    p34_new = sum(1 for r in p1_rows if r.get("source") == "P34")
    elapsed = time.time() - t_start
    rate = p34_new / max(elapsed / 3600.0, 1e-6)

    gen_report = f"""# GENERATION_REPORT — P34

Generated: {_utc()}
Elapsed_h: {elapsed/3600:.2f}

## Method

- Donors: Stage-1 verified `A_STABLE_DIRECT_HANDOFF` (P21/P22/P32 eval holdouts excluded).
- Construction: `build_regime_ic` (absolute hand-frame residuals ± orientation ± mild velocity ± pre-freeze mass/friction).
- Gate: `check_d2_eligible` → **immediate HANDOFF STRICT** → (if fail) production `run_precision(plan_id=D)`.
- **P1 iff** immediate STRICT fails **and** Oracle executed + STRICT succeeds.
- Independence: unique lineage `P34__{{root}}__{{regime_id}}` + capture fingerprint.
- P33's 25 P1 roots remain immutable and are included in totals.
- Workers: {workers}

## Yield

| Metric | Value |
|---|---:|
| Attempts (this run batches) | {attempts} |
| **Total P1 roots (P33+P34)** | **{len(p1_rows)}** |
| New P34 P1 | {p34_new} |
| P0 collected | {len(p0_rows)} |
| Train / Val / Test | {len(train)} / {len(val)} / {len(test)} |
| Unique train roots | {len({r['physical_root'] for r in train})} |
| Unique val roots | {len({r['physical_root'] for r in val})} |
| Unique test roots | {len({r['physical_root'] for r in test})} |
| New P1 / hour | {rate:.2f} |

## Checkpoints

- A (≥50 total): {'yes' if len(p1_rows)>=50 else 'no'}
- B (≥100 total): {'yes' if len(p1_rows)>=100 else 'no'}
- C (≥200 total): {'yes' if len(p1_rows)>=200 else 'no'}

## Bottleneck

Oracle wall-clock on immediate-fail candidates remains dominant (~1.7–2.1k steps/tape).
Acceptance criteria and Oracle behavior were **not** relaxed.
"""
    _write_md(OUT / "GENERATION_REPORT.md", gen_report)
    return {
        "p1_total": len(p1_rows),
        "p34_new": p34_new,
        "p0": len(p0_rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "elapsed_s": elapsed,
        "attempts": attempts,
        "p1_per_hour": rate,
    }


# ---------------------------------------------------------------------------
# M111 audit + final report
# ---------------------------------------------------------------------------
def run_m111_audit(limit: int | None = None) -> dict:
    ICS_CONT_ONLY.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = sorted(ICS_P1.glob("*.pkl"))
    if limit:
        paths = paths[:limit]
    print(f"=== M111 audit n={len(paths)} ===", flush=True)
    cont_fail = 0
    for p in paths:
        r = m111_replay_check(p)
        rows.append(r)
        if not r.get("ok"):
            cont_fail += 1
            # Move reference note — do not delete continuous IC; mark in manifest
            print(f"  M111 FAIL {p.name}: {r.get('m111_reason')}", flush=True)
        else:
            print(f"  M111 OK {p.name}", flush=True)
    _write_csv(OUT / "M111_REPLAY_RESULTS.csv", rows)
    text = f"""# M111_REPLAY_AUDIT — P34

Generated: {_utc()}

## Protocol

1. Load continuous Oracle actions from accepted P1 pickles.
2. Quantize with frozen M111 codebook (`SELECTED_skill_aware_hybrid_meta.json`).
3. Replay **decoded** actions only via `step_legal` + 12s STRICT.
4. Continuous-only successes that fail M111 replay are reported here and excluded from the M111-trainable partition flag.

## Results

| Metric | Value |
|---|---:|
| Audited P34 P1 ICs | {len(rows)} |
| M111 STRICT success | {sum(1 for r in rows if r.get('ok'))} |
| Continuous-ok / M111-fail | {cont_fail} |

## Detail

See `M111_REPLAY_RESULTS.csv`.

Continuous Oracle success **does not** automatically enter the M111-trainable partition when replay fails.
"""
    _write_md(OUT / "M111_REPLAY_AUDIT.md", text)
    # Update manifest flags
    man_path = OUT / "DATASET_MANIFEST.csv"
    if man_path.is_file():
        with man_path.open(encoding="utf-8") as f:
            man = list(csv.DictReader(f))
        by_path = {r.get("ic_path"): r for r in rows}
        for m in man:
            if m.get("source") == "P33":
                m["m111_trainable"] = "1"
                continue
            ip = m.get("ic_path")
            if ip in by_path:
                m["m111_trainable"] = "1" if by_path[ip].get("ok") else "0"
                m["m111_SUCCESS_NEW"] = by_path[ip].get("m111_SUCCESS_NEW")
            else:
                m["m111_trainable"] = ""
        _write_csv(man_path, man)
    return {"audited": len(rows), "m111_ok": sum(1 for r in rows if r.get("ok")), "m111_fail": cont_fail}


def write_final_report(gen: dict, audit: dict, profile: dict | None) -> None:
    man = []
    if (OUT / "DATASET_MANIFEST.csv").is_file():
        with (OUT / "DATASET_MANIFEST.csv").open(encoding="utf-8") as f:
            man = list(csv.DictReader(f))
    p33_n = sum(1 for r in man if r.get("source") == "P33")
    p34_n = sum(1 for r in man if r.get("source") == "P34")
    train = [r for r in man if r.get("split") == "train"]
    val = [r for r in man if r.get("split") == "val"]
    test = [r for r in man if r.get("split") == "test"]
    m111_fail = int(audit.get("m111_fail") or 0)
    p0_n = 0
    if (OUT / "P0_MANIFEST.csv").is_file():
        with (OUT / "P0_MANIFEST.csv").open(encoding="utf-8") as f:
            p0_n = sum(1 for _ in csv.DictReader(f))

    # Regime coverage summary
    cov = []
    if (OUT / "REGIME_COVERAGE.csv").is_file():
        with (OUT / "REGIME_COVERAGE.csv").open(encoding="utf-8") as f:
            cov = list(csv.DictReader(f))
    with_p1 = [c for c in cov if int(float(c.get("P1") or 0)) > 0]
    zero_p1 = [c for c in cov if int(float(c.get("attempts") or 0)) >= 3 and int(float(c.get("P1") or 0)) == 0]

    rate = float(gen.get("p1_per_hour") or 0)
    remain = max(0, 200 - len(man))
    proj_h = remain / rate if rate > 0 else float("inf")

    # Throughput improvement estimate
    thr_note = "see THROUGHPUT_PROFILE.md"
    if profile:
        b = float(profile.get("baseline_wall_s") or 0) / max(1, int(profile.get("baseline_n") or 1))
        o = float(profile.get("opt_wall_s") or 0) / max(1, int(profile.get("opt_n") or 1))
        thr_note = f"serial mean s/cand baseline-like={b:.1f} → optimized={o:.1f}"

    fam = Counter((r.get("family"), r.get("target_dz_mm"), r.get("target_dx_mm")) for r in man if r.get("source") == "P34")

    ready = (
        len(man) >= 50
        and m111_fail == 0
        and len(train) >= 25
        and len(val) >= 3
        and len(test) >= 3
    )
    # Dataset ready for next SFT experiment ≠ Precision is learnable
    ready_txt = (
        "YES — sufficient verified M111-trainable P1 volume and splits for a follow-on Precision-only SFT curve"
        if ready
        else "PARTIAL / NO — see yield and M111 audit; do not claim learnability from generation alone"
    )

    text = f"""# FINAL_REPORT — P34 Precision Data Expansion

Generated: {_utc()}

## Answers

1. **New independent P1 roots generated (P34):** {p34_n}
2. **Total trainable P1 roots now (P33+P34):** {len(man)} (P33={p33_n}, P34={p34_n})
3. **Final split:** train={len(train)}, val={len(val)}, test={len(test)} (P33 assignments preserved)
4. **Throughput:** {thr_note}; generation new-P1/hour ≈ {rate:.2f}
5. **Adequate coverage regimes:** {len(with_p1)} regimes with ≥1 P1 (see REGIME_COVERAGE.csv)
6. **Underrepresented / sterile regimes:** {len(zero_p1)} regimes with ≥3 attempts and 0 P1 — notably orientation-only, some MASS_FRIC, and non-12 mm DZ cells may remain sparse
7. **Continuous Oracle OK but M111 replay fail:** {m111_fail}
8. **Independent P0 roots collected:** {p0_n}
9. **Projected cost to 200 total P1:** ~{proj_h:.1f} hours at current rate ({remain} remaining)
10. **Ready for next Precision-only SFT experiment?** {ready_txt}

## Important

This report does **not** claim that Precision correction is learnable. P33 closed-loop SFT was 0/N; P34 only expands data.

## P34 family snapshot (new only)

{json.dumps({str(k): v for k, v in fam.most_common(20)}, indent=2)}

## Artifacts

- `THROUGHPUT_PROFILE.md`
- `GENERATION_REPORT.md`
- `DATASET_MANIFEST.csv`
- `P0_MANIFEST.csv`
- `REGIME_COVERAGE.csv`
- `P1_ACCEPTANCE_BY_REGIME.csv`
- `M111_REPLAY_AUDIT.md`
- `generated_ics/p1/`, `generated_ics/p0/`
"""
    _write_md(OUT / "FINAL_REPORT.md", text)
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-only", action="store_true")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=8000)
    ap.add_argument("--profile-n", type=int, default=10)
    ap.add_argument("--m111-limit", type=int, default=0, help="0=all P34 P1")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-m111", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    profile = None
    if not args.skip_profile:
        profile = run_profile(n_candidates=args.profile_n, workers=1)
        if args.profile_only:
            return
    gen = {"p1_total": 0, "p34_new": 0, "p1_per_hour": 0}
    if not args.skip_gen:
        gen = run_generation(
            target_total_p1=args.target,
            workers=args.workers,
            max_attempts=args.max_attempts,
        )
    audit = {"audited": 0, "m111_ok": 0, "m111_fail": 0}
    if not args.skip_m111:
        lim = args.m111_limit if args.m111_limit > 0 else None
        audit = run_m111_audit(limit=lim)
    # reload gen stats from checkpoint if skip-gen
    if args.skip_gen and (OUT / "checkpoint.json").is_file():
        gen = {**gen, **json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))}
    write_final_report(gen, audit, profile)


if __name__ == "__main__":
    main()
