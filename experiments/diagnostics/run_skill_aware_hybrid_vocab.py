#!/usr/bin/env python
"""Construct and validate a skill-aware hybrid action vocabulary.

Hybrid decoder (matches deployed policy):
    a = clip(c_k + clip(r, -rho, rho), -1, 1)

Oracle reconstruction uses expert actions to choose (k, r); this is NOT learned-policy eval.
Does not overwrite K49, alter experts, physics, or success criteria. No RL/SFT.

Results → results/diagnostics/skill_aware_action_vocab_design/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import sample_state  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.corrections.contact_primitives import _wrist  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.action_vocab_k49 import CAP as K49_CAP  # noqa: E402
from training.action_vocab_k49 import HANDOFF as K49_HANDOFF  # noqa: E402
from training.action_vocab_k49 import build_k49_table  # noqa: E402
from training.skill_aware_action_vocab import (  # noqa: E402
    ACT_DIMS,
    DEFAULT_WEIGHTS,
    EARLY_CATCH_N,
    M_MAX,
    ResidualBudget,
    MotionCodebook,
    action_key,
    batch_nearest_wlinf,
    dedupe_prototypes,
    hybrid_decode,
    oracle_reconstruct,
    phase_labels,
    quantize_trajectory,
    weighted_linf,
)

DS = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
OUT = ROOT / "results/diagnostics/skill_aware_action_vocab_design"
K49_CONFIG = "K49_tok48_wy0p25"
SEED = 5478
DT = float(DT_POLICY)

# Phase proxy thresholds (construction-only).
PHASE_TAU = {
    "P_catch": 0.95,
    "P_lift": 0.90,
    "P_precision": 0.90,
    "P_s1_short": 0.95,
    "P_settle": 0.95,
}
PHASE_EPS = {
    "P_catch": 0.05,
    "P_lift": 0.08,
    "P_precision": 0.08,
    "P_s1_short": 0.08,
    "P_settle": 0.08,
}

# Soft priority for which phases to cover first when growing the book.
PHASE_PRIORITY = [
    ("C", "P_catch"),
    ("C", "P_lift"),
    ("A", "P_precision"),
    ("STABLE_PRECISION", "P_precision"),
    ("A", "P_s1_short"),
    ("B", "P_s1_short"),
    ("A", "P_settle"),
    ("B", "P_settle"),
    ("C", "P_settle"),
    ("STABLE_PRECISION", "P_settle"),
]

SKILLS = ("A", "B", "C", "STABLE_PRECISION")


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


# ---------------------------------------------------------------------------
# Dataset / splits
# ---------------------------------------------------------------------------


def load_expert_rows() -> list[dict]:
    rows = [
        json.loads(l)
        for l in (DS / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    out = []
    for r in rows:
        if r.get("partition") not in ("expert", "expert_continuous_only"):
            continue
        if r.get("source_label") not in SKILLS:
            continue
        cd = r.get("continuous_demo") or {}
        if not cd.get("verified"):
            continue
        if not bool((r.get("handoff") or {}).get("SUCCESS_NEW")) and not cd.get("replay_SUCCESS_NEW"):
            continue
        out.append(r)
    return out


def stratified_lineage_split(
    rows: list[dict], seed: int = SEED, frac_construct: float = 0.6
) -> tuple[list[dict], list[dict], dict]:
    """60/40 construction/held-out by lineage, stratified by skill. Construction-only for centers."""
    rng = np.random.RandomState(seed)
    by_skill: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_skill[r["source_label"]].append(r)

    construct, held = [], []
    detail = {}
    for skill in SKILLS:
        xs = list(by_skill.get(skill, []))
        # Deterministic order then shuffle
        xs.sort(key=lambda r: r["lineage_id"])
        rng.shuffle(xs)
        n = len(xs)
        if n == 0:
            detail[skill] = {"n": 0, "construct": 0, "heldout": 0}
            continue
        n_tr = int(round(frac_construct * n))
        if n >= 2:
            n_tr = min(max(1, n_tr), n - 1)
        else:
            n_tr = n  # single sample → construction (document leakage risk)
        construct.extend(xs[:n_tr])
        held.extend(xs[n_tr:])
        detail[skill] = {
            "n": n,
            "construct": n_tr,
            "heldout": n - n_tr,
            "construct_lineages": [r["lineage_id"] for r in xs[:n_tr]],
            "heldout_lineages": [r["lineage_id"] for r in xs[n_tr:]],
        }
    rng.shuffle(construct)
    rng.shuffle(held)
    meta = {
        "seed": seed,
        "frac_construct": frac_construct,
        "n_construct": len(construct),
        "n_heldout": len(held),
        "by_skill": detail,
        "rule": "stratified lineage 60/40 by source_label; centers from construct only",
    }
    return construct, held, meta


def load_traj(row: dict) -> dict:
    ic = pickle.loads((DS / row["ic_pkl"]).read_bytes())
    cont = np.load(DS / row["continuous_demo"]["path"])
    acts = np.asarray(cont["actions"], np.float32)
    s1 = int(row.get("n_actions_stage1") or (int(cont["stage1_end_idx"][0]) if "stage1_end_idx" in cont.files else 0))
    n_prec = int(row.get("n_actions_precision") or 0)
    skill = row["source_label"]
    phases = phase_labels(len(acts), s1, skill, n_precision=n_prec)
    return {
        "sample_id": row["sample_id"],
        "lineage_id": row["lineage_id"],
        "skill": skill,
        "capture": ic["capture"],
        "ref": ic["ref"],
        "actions": acts,
        "stage1_end": s1,
        "n_precision": n_prec,
        "phases": phases,
        "k49_verified": bool((row.get("k49_demo") or {}).get("verified")),
        "k49_path": (row.get("k49_demo") or {}).get("path"),
    }


# ---------------------------------------------------------------------------
# Codebook construction
# ---------------------------------------------------------------------------


def collect_phase_pools(trajs: list[dict]) -> dict[tuple[str, str], list[np.ndarray]]:
    pools: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for tr in trajs:
        for a, ph in zip(tr["actions"], tr["phases"]):
            if ph.startswith("P_"):
                pools[(tr["skill"], ph)].append(np.asarray(a, np.float32))
    return pools


def subsample_pool(acts: list[np.ndarray], max_n: int, seed: int) -> np.ndarray:
    if not acts:
        return np.zeros((0, 7), np.float32)
    arr = np.asarray(acts, np.float32)
    if len(arr) <= max_n:
        return arr
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(arr), size=max_n, replace=False)
    return arr[idx]


def phase_proxy_coverage(
    book: np.ndarray,
    acts: np.ndarray,
    rho: ResidualBudget,
    eps: float,
    w=None,
) -> dict:
    if len(acts) == 0:
        return {"n": 0, "covered": 0, "rate": 1.0, "mean_wlinf": 0.0, "clip_frac": 0.0}
    book = np.asarray(book, np.float32).reshape(-1, 7)
    acts = np.asarray(acts, np.float32).reshape(-1, 7)
    w = DEFAULT_WEIGHTS if w is None else np.asarray(w, dtype=np.float64)
    tok, _ = batch_nearest_wlinf(book, acts, w=w)
    proto = book[tok]
    raw = acts - proto
    rbudget = rho.as_array()
    res = np.clip(raw, -rbudget, rbudget)
    recon = np.clip(proto + res, -1.0, 1.0)
    err = recon - acts
    wl = np.max(w * np.abs(err), axis=1)
    clipped = np.any(np.abs(raw) > rbudget + 1e-8, axis=1)
    n = len(acts)
    covered = int(np.sum(wl <= eps + 1e-9))
    return {
        "n": n,
        "covered": covered,
        "rate": covered / n,
        "mean_wlinf": float(np.mean(wl)),
        "clip_frac": float(np.mean(clipped)),
    }


def greedy_add_center(
    book: list[np.ndarray],
    candidates: np.ndarray,
    rho: ResidualBudget,
    w,
    *,
    min_dist: float,
    catch_boost: bool = False,
    score_cap: int = 600,
    seed: int = 0,
) -> tuple[np.ndarray | None, dict]:
    """Pick candidate maximizing uncovered mass under current book (weighted FPS)."""
    if len(candidates) == 0:
        return None, {"reason": "empty"}
    best_score, best_a, best_meta = -1.0, None, {}
    seen = set()
    uniq = []
    for a in candidates:
        k = action_key(a, nd=3)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(np.asarray(a, np.float32).reshape(7))
    cand = np.asarray(uniq, np.float32)
    if len(cand) > score_cap:
        rng = np.random.RandomState(seed)
        # Keep high |wx|/|vz| actions when catch_boost
        if catch_boost:
            mag = 2.0 * np.abs(cand[:, 3]) + 1.5 * np.abs(cand[:, 2]) + np.abs(cand[:, 1])
            top = np.argsort(-mag)[: score_cap // 2]
            rest = rng.choice(len(cand), size=min(score_cap // 2, len(cand)), replace=False)
            idx = np.unique(np.concatenate([top, rest]))[:score_cap]
            cand = cand[idx]
        else:
            cand = cand[rng.choice(len(cand), size=score_cap, replace=False)]

    book_arr = np.asarray(book, np.float32) if book else np.zeros((0, 7), np.float32)
    rho_arr = rho.as_array()
    w = np.asarray(w, dtype=np.float64).reshape(7)

    # Precompute current nearest distance for each cand point
    if len(book_arr):
        d_cur = np.array(
            [min(weighted_linf(p, b, w) for p in book_arr) for b in cand], dtype=np.float64
        )
    else:
        d_cur = np.full(len(cand), 1e9, dtype=np.float64)

    for ai, a in enumerate(cand):
        if book and float(np.min(np.max(w * np.abs(book_arr - a), axis=1))) < min_dist:
            continue
        d_a = np.max(w * np.abs(cand - a), axis=1)  # weighted linf to all cand
        improve = d_a + 1e-9 < d_cur
        if not np.any(improve):
            score = 0.0
        else:
            raw = np.abs(cand - a)
            beyond = np.maximum(raw - rho_arr, 0.0)
            mass = beyond.sum(axis=1)
            boost = np.ones(len(cand), dtype=np.float64)
            if catch_boost:
                boost = boost + 3.0 * beyond[:, 3] + 2.0 * beyond[:, 2] + 1.5 * beyond[:, 1]
            score = float(np.sum((d_cur[improve] - d_a[improve]) + boost[improve] * (1.0 + mass[improve])))
        if book:
            score += 0.25 * float(np.min(np.max(w * np.abs(book_arr - a), axis=1)))
        else:
            score += 0.25
        if score > best_score:
            best_score, best_a = score, a.copy()
            best_meta = {"score": float(score), "key": action_key(a)}
    if best_a is None:
        return None, {"reason": "all_too_close"}
    return best_a, best_meta


def build_shared_core(
    pools: dict[tuple[str, str], list[np.ndarray]],
    rho: ResidualBudget,
    w,
    *,
    target_n: int = 32,
    seed: int = SEED,
) -> tuple[list[np.ndarray], list[dict]]:
    """Shared patterns: secure-hold, small corrections, common translations across skills."""
    shared_phases = ("P_settle", "P_s1_short", "P_lift")
    bags = []
    for (sk, ph), acts in pools.items():
        if ph in shared_phases or (ph == "P_precision" and sk in ("A", "STABLE_PRECISION")):
            take = acts if ph != "P_precision" else acts[::5]
            bags.extend(take)
    arr = subsample_pool(bags, max_n=4000, seed=seed)
    if len(arr) == 0:
        hold = [np.array([0, 0, 0, 0, 0, 0, 1], np.float32)]
        return hold, [{"bucket": "shared", "reason": "empty_pool_seed"}]

    book: list[np.ndarray] = []
    seeds = [
        np.array([0, 0, 0, 0, 0, 0, 1], np.float32),
        np.array([0, 0, 0, 0, 0, 0, -1], np.float32),
        np.array([0, 0, 0, 0, 0, 0, 0], np.float32),
    ]
    for s in seeds:
        book.append(s)

    selection_log: list[dict] = []
    while len(book) < target_n:
        a, meta = greedy_add_center(
            book, arr, rho, w, min_dist=0.12, catch_boost=False, seed=seed + len(book)
        )
        if a is None:
            break
        book.append(a)
        selection_log.append({"bucket": "shared", **meta, "M_after": len(book)})
    return dedupe_prototypes(book, min_dist=0.08, w=w), selection_log


def _hard_actions(book_arr: np.ndarray, acts: np.ndarray, rho: ResidualBudget, eps: float, w) -> np.ndarray:
    if len(acts) == 0:
        return acts
    cov = phase_proxy_coverage(book_arr, acts, rho, eps, w=w)
    # Recompute mask via batch
    tok, _ = batch_nearest_wlinf(book_arr, acts, w=w)
    proto = book_arr[tok]
    raw = acts - proto
    res = np.clip(raw, -rho.as_array(), rho.as_array())
    recon = np.clip(proto + res, -1.0, 1.0)
    wl = np.max(np.asarray(w).reshape(7) * np.abs(recon - acts), axis=1)
    hard = acts[wl > eps]
    return hard if len(hard) else acts


def construct_codebook(
    construct_trajs: list[dict],
    rho: ResidualBudget,
    w=None,
    *,
    m_max: int = M_MAX,
    seed: int = SEED,
) -> tuple[MotionCodebook, dict]:
    w = DEFAULT_WEIGHTS if w is None else np.asarray(w, dtype=np.float64)
    pools = collect_phase_pools(construct_trajs)
    pool_arr = {
        k: subsample_pool(v, max_n=1200, seed=seed + (abs(hash(k[0] + k[1])) % 10000))
        for k, v in pools.items()
    }

    shared, shared_log = build_shared_core(pools, rho, w, target_n=24, seed=seed)
    book = list(shared)
    selection_log = list(shared_log)
    print(f"    shared_core M={len(book)} rho={rho.label()}", flush=True)

    def coverage_report(book_list):
        ba = np.asarray(book_list, np.float32)
        rep = {}
        for (sk, ph), acts in pool_arr.items():
            eps = PHASE_EPS.get(ph, 0.08)
            rep[f"{sk}/{ph}"] = phase_proxy_coverage(ba, acts, rho, eps, w=w)
        return rep

    stuck = 0
    while len(book) < m_max:
        cov = coverage_report(book)
        target = None
        for sk, ph in PHASE_PRIORITY:
            key = f"{sk}/{ph}"
            if key not in cov or cov[key]["n"] == 0:
                continue
            tau = PHASE_TAU.get(ph, 0.9)
            if cov[key]["rate"] + 1e-12 < tau:
                target = (sk, ph)
                break
        if target is None:
            break

        sk, ph = target
        acts = pool_arr.get((sk, ph), np.zeros((0, 7), np.float32))
        eps = PHASE_EPS.get(ph, 0.08)
        ba = np.asarray(book, np.float32)
        cand = _hard_actions(ba, acts, rho, eps, w)
        catch_boost = sk == "C" and ph == "P_catch"
        a, meta = greedy_add_center(
            book,
            cand,
            rho,
            w,
            min_dist=0.10 if catch_boost else 0.12,
            catch_boost=catch_boost,
            seed=seed + len(book),
            score_cap=400,
        )
        if a is None:
            stuck += 1
            if stuck >= 3:
                break
            continue
        stuck = 0
        before = cov[f"{sk}/{ph}"]["rate"]
        book.append(a)
        after_cov = phase_proxy_coverage(np.asarray(book, np.float32), acts, rho, eps, w=w)
        selection_log.append({
            "bucket": f"{sk}/{ph}",
            "reason": "phase_under_coverage",
            "coverage_before": before,
            "coverage_after": after_cov["rate"],
            "tau": PHASE_TAU.get(ph, 0.9),
            "M_after": len(book),
            **meta,
        })
        if len(book) % 8 == 0:
            print(
                f"    grow M={len(book)} focus={sk}/{ph} cov={after_cov['rate']:.3f}/{PHASE_TAU.get(ph,0.9)}",
                flush=True,
            )

    book = dedupe_prototypes(book, min_dist=0.06, w=w)
    cov = coverage_report(book)
    catch_key = "C/P_catch"
    force_steps = 0
    while (
        len(book) < m_max
        and catch_key in cov
        and cov[catch_key]["n"] > 0
        and cov[catch_key]["rate"] < PHASE_TAU["P_catch"]
        and force_steps < 40
    ):
        acts = pool_arr[("C", "P_catch")]
        cand = _hard_actions(np.asarray(book, np.float32), acts, rho, PHASE_EPS["P_catch"], w)
        a, meta = greedy_add_center(
            book, cand, rho, w, min_dist=0.08, catch_boost=True, seed=seed + 1000 + force_steps, score_cap=400
        )
        if a is None:
            break
        book.append(a)
        cov = coverage_report(book)
        force_steps += 1
        selection_log.append({
            "bucket": "C/P_catch",
            "reason": "force_catch_coverage",
            "coverage_after": cov[catch_key]["rate"],
            "M_after": len(book),
            **meta,
        })

    protos = np.asarray(book, np.float32)
    final_cov = coverage_report(book)
    unmet = {
        k: v
        for k, v in final_cov.items()
        if v["n"] > 0 and v["rate"] < PHASE_TAU.get(k.split("/")[1], 0.9) - 1e-12
    }
    print(f"    done M={len(protos)} unmet={list(unmet.keys())}", flush=True)
    cb = MotionCodebook(
        prototypes=protos,
        residual=rho,
        weights=w,
        meta={
            "construction": "skill_phase_greedy_wlinf",
            "M": int(len(protos)),
            "unmet_phases": unmet,
            "phase_coverage": final_cov,
        },
    )
    return cb, {"selection_log": selection_log, "phase_coverage": final_cov, "unmet": unmet}


# ---------------------------------------------------------------------------
# Physics replay
# ---------------------------------------------------------------------------


def snap_state(sim) -> dict:
    err = measure_errors(sim)
    m = classify_contact_mode(sim)
    st = sample_state(sim)
    return {
        "e_pos_mm": float(err["e_pos_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "bilateral": bool(err["bilateral"]),
        "v_rel": float(st.get("v_rel_norm") or 0.0),
        "obj_z": float(err["obj_z"]),
        "mode": m["mode"],
        "latched": bool(latched(sim)),
    }


def handoff_eval(env, ref) -> dict:
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
    }


def replay_actions(env, cap, ref, actions: np.ndarray, *, early_contact_trace: bool = False) -> dict:
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    first_bilateral_loss = None
    first_latch = None
    early = []
    n = len(actions)
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
        if early_contact_trace and i < EARLY_CATCH_N:
            st = snap_state(env.sim)
            early.append({"i": i + 1, "t": (i + 1) * DT, **st})
            if first_bilateral_loss is None and not st["bilateral"]:
                first_bilateral_loss = i + 1
            if first_latch is None and st["latched"]:
                first_latch = i + 1
        elif i % 10 == 9 or i + 1 == n:
            st = snap_state(env.sim)
            if first_bilateral_loss is None and not st["bilateral"]:
                first_bilateral_loss = i + 1
            if first_latch is None and st["latched"]:
                first_latch = i + 1
    h = handoff_eval(env, ref)
    fail = None
    if not h["SUCCESS_NEW"]:
        if first_latch is not None:
            fail = "support_drop_latched"
        elif first_bilateral_loss is not None:
            fail = "bilateral_loss"
        else:
            fail = h["reason"] or "SUCCESS_FAIL"
    return {
        **h,
        "first_bilateral_loss_i": first_bilateral_loss,
        "first_latch_i": first_latch,
        "fail_category": fail,
        "early_trace": early,
        "n": n,
    }


def quant_stats_for_traj(q: dict, phases: list[str]) -> dict:
    acts_err = []
    per_dim = {d: [] for d in ACT_DIMS}
    per_phase = defaultdict(list)
    sat_phase = defaultdict(int)
    phase_n = Counter()
    tokens = q["tokens"]
    for i, info in enumerate(q["infos"]):
        ph = phases[i] if i < len(phases) else "P_other"
        phase_n[ph] += 1
        per_phase[ph].append(info["recon_wlinf"])
        if info["clipped"]:
            sat_phase[ph] += 1
        for di, d in enumerate(ACT_DIMS):
            per_dim[d].append(abs(float(info["err"][di])))
        acts_err.append(info["recon_wlinf"])
    tok_counts = Counter(int(t) for t in tokens)
    return {
        "mean_recon_wlinf": float(np.mean(acts_err)) if acts_err else 0.0,
        "max_recon_wlinf": float(np.max(acts_err)) if acts_err else 0.0,
        "clipped_frac": q["clipped_frac"],
        "mae_per_dim": {d: float(np.mean(per_dim[d])) if per_dim[d] else 0.0 for d in ACT_DIMS},
        "sat_frac_per_dim": {d: float(q["sat_frac_per_dim"][i]) for i, d in enumerate(ACT_DIMS)},
        "mean_abs_residual": {d: float(q["mean_abs_residual"][i]) for i, d in enumerate(ACT_DIMS)},
        "phase_mean_wlinf": {ph: float(np.mean(vs)) for ph, vs in per_phase.items()},
        "phase_sat_frac": {ph: sat_phase[ph] / max(phase_n[ph], 1) for ph in phase_n},
        "token_hist": dict(tok_counts),
        "n_unique_tokens": len(tok_counts),
    }


def evaluate_vocab_on_trajs(
    env,
    trajs: list[dict],
    book: np.ndarray,
    rho: ResidualBudget,
    w,
    *,
    tag: str,
    early_c: bool = True,
) -> tuple[list[dict], dict]:
    rows = []
    skill_stats = {s: {"n": 0, "new": 0, "s12": 0, "bilateral_loss": 0, "latched": 0, "other": 0} for s in SKILLS}
    token_global = Counter()
    skill_token = {s: Counter() for s in SKILLS}
    early_wx_mae = []
    early_vz_mae = []

    for idx, tr in enumerate(trajs):
        t0 = time.time()
        q = quantize_trajectory(book, tr["actions"], rho, w=w)
        # Verify decoder identity: recon == hybrid_decode(proto, residual)
        if len(q["tokens"]):
            for i in range(min(3, len(q["tokens"]))):
                proto = book[int(q["tokens"][i])]
                dec = hybrid_decode(proto, q["residuals"][i], rho)
                if float(np.max(np.abs(dec - q["recon_actions"][i]))) > 1e-6:
                    raise RuntimeError("hybrid decoder mismatch vs oracle recon")

        phys = replay_actions(
            env,
            tr["capture"],
            tr["ref"],
            q["recon_actions"],
            early_contact_trace=early_c and tr["skill"] == "C",
        )
        qs = quant_stats_for_traj(q, tr["phases"])
        token_global.update(qs["token_hist"])
        skill_token[tr["skill"]].update(qs["token_hist"])

        # Early C dim errors
        early_detail = None
        if tr["skill"] == "C" and len(tr["actions"]) > 0:
            n_e = min(EARLY_CATCH_N, len(tr["actions"]))
            ac = tr["actions"][:n_e]
            ar = q["recon_actions"][:n_e]
            err = ar - ac
            early_detail = {
                "mae_wx": float(np.mean(np.abs(err[:, 3]))),
                "mae_vz": float(np.mean(np.abs(err[:, 2]))),
                "mae_vy": float(np.mean(np.abs(err[:, 1]))),
                "max_wlinf": float(max(weighted_linf(ar[i], ac[i], w) for i in range(n_e))),
                "clip_frac": float(np.mean([q["infos"][i]["clipped"] for i in range(n_e)])),
            }
            early_wx_mae.append(early_detail["mae_wx"])
            early_vz_mae.append(early_detail["mae_vz"])

        sk = tr["skill"]
        skill_stats[sk]["n"] += 1
        if phys["SUCCESS_NEW"]:
            skill_stats[sk]["new"] += 1
        if phys["SUCCESS_12S"]:
            skill_stats[sk]["s12"] += 1
        fc = phys["fail_category"]
        if fc == "bilateral_loss":
            skill_stats[sk]["bilateral_loss"] += 1
        elif fc == "support_drop_latched":
            skill_stats[sk]["latched"] += 1
        elif fc:
            skill_stats[sk]["other"] += 1

        rows.append({
            "tag": tag,
            "sample_id": tr["sample_id"],
            "lineage_id": tr["lineage_id"],
            "skill": sk,
            "n_actions": tr["actions"].shape[0],
            "SUCCESS_NEW": phys["SUCCESS_NEW"],
            "SUCCESS_12S": phys["SUCCESS_12S"],
            "reason": phys["reason"],
            "fail_category": phys["fail_category"],
            "first_bilateral_loss_i": phys["first_bilateral_loss_i"],
            "first_latch_i": phys["first_latch_i"],
            "first_div_t": (
                None
                if phys["first_bilateral_loss_i"] is None and phys["first_latch_i"] is None
                else DT * min(
                    x for x in [phys["first_bilateral_loss_i"], phys["first_latch_i"]] if x is not None
                )
            ),
            "mean_recon_wlinf": qs["mean_recon_wlinf"],
            "max_recon_wlinf": qs["max_recon_wlinf"],
            "clipped_frac": qs["clipped_frac"],
            "n_unique_tokens": qs["n_unique_tokens"],
            **{f"mae_{d}": qs["mae_per_dim"][d] for d in ACT_DIMS},
            **{f"sat_{d}": qs["sat_frac_per_dim"][d] for d in ACT_DIMS},
            **{f"absres_{d}": qs["mean_abs_residual"][d] for d in ACT_DIMS},
            "early_mae_wx": None if early_detail is None else early_detail["mae_wx"],
            "early_mae_vz": None if early_detail is None else early_detail["mae_vz"],
            "early_mae_vy": None if early_detail is None else early_detail["mae_vy"],
            "early_max_wlinf": None if early_detail is None else early_detail["max_wlinf"],
            "wall_s": time.time() - t0,
        })
        if (idx + 1) % 5 == 0 or idx + 1 == len(trajs):
            print(f"  [{tag}] {idx+1}/{len(trajs)} last={tr['sample_id'][:40]} NEW={phys['SUCCESS_NEW']}", flush=True)

    summary = {"by_skill": {}, "token_utilization": dict(token_global), "n_tokens_used": len(token_global)}
    for s in SKILLS:
        st = skill_stats[s]
        n = st["n"]
        k = st["new"]
        lo, hi = wilson_ci(k, n)
        summary["by_skill"][s] = {
            **st,
            "rate_NEW": (k / n) if n else float("nan"),
            "rate_12S": (st["s12"] / n) if n else float("nan"),
            "ci95_NEW": [lo, hi],
            "n_unique_tokens": len(skill_token[s]),
            "top_tokens": skill_token[s].most_common(8),
        }
    if early_wx_mae:
        summary["C_early"] = {
            "mean_mae_wx": float(np.mean(early_wx_mae)),
            "mean_mae_vz": float(np.mean(early_vz_mae)),
        }
    # Skill distinction: Jensen-Shannon-ish via token overlap
    summary["skill_token_overlap"] = {}
    for s1 in SKILLS:
        for s2 in SKILLS:
            if s1 >= s2:
                continue
            a, b = set(skill_token[s1]), set(skill_token[s2])
            if not a and not b:
                continue
            inter = len(a & b)
            union = len(a | b) or 1
            summary["skill_token_overlap"][f"{s1}|{s2}"] = {
                "jaccard": inter / union,
                "shared": inter,
                "union": union,
            }
    return rows, summary


def evaluate_k49_baseline(env, trajs: list[dict], table: np.ndarray, *, tag: str) -> tuple[list[dict], dict]:
    """Frozen K49 hybrid oracle recon with CAP=0.25 (skip HANDOFF row)."""
    # Build motion-only view for nearest: use full table but skip row 47 in nearest_proto clone
    motion_idx = [i for i in range(len(table)) if i != K49_HANDOFF]
    book = table[np.asarray(motion_idx)]
    # Map book row → original token id
    id_map = {j: motion_idx[j] for j in range(len(motion_idx))}

    rows = []
    skill_stats = {s: {"n": 0, "new": 0, "s12": 0, "bilateral_loss": 0, "latched": 0, "other": 0} for s in SKILLS}

    for idx, tr in enumerate(trajs):
        # Reconstruct with K49 CAP using L2 nearest on motion rows (historical behavior)
        acts = np.asarray(tr["actions"], np.float32).reshape(-1, 7)
        d = np.linalg.norm(acts[:, None, :] - book[None, :, :], axis=2)
        tok = np.argmin(d, axis=1)
        proto = book[tok]
        raw = acts - proto
        res = np.clip(raw, -K49_CAP, K49_CAP).astype(np.float32)
        recon_a = np.clip(proto + res, -1.0, 1.0).astype(np.float32)
        clipped = int(np.sum(np.any(np.abs(raw) > K49_CAP + 1e-8, axis=1)))
        phys = replay_actions(env, tr["capture"], tr["ref"], recon_a, early_contact_trace=tr["skill"] == "C")
        sk = tr["skill"]
        skill_stats[sk]["n"] += 1
        if phys["SUCCESS_NEW"]:
            skill_stats[sk]["new"] += 1
        if phys["SUCCESS_12S"]:
            skill_stats[sk]["s12"] += 1
        fc = phys["fail_category"]
        if fc == "bilateral_loss":
            skill_stats[sk]["bilateral_loss"] += 1
        elif fc == "support_drop_latched":
            skill_stats[sk]["latched"] += 1
        elif fc:
            skill_stats[sk]["other"] += 1
        rows.append({
            "tag": tag,
            "sample_id": tr["sample_id"],
            "skill": sk,
            "SUCCESS_NEW": phys["SUCCESS_NEW"],
            "SUCCESS_12S": phys["SUCCESS_12S"],
            "fail_category": phys["fail_category"],
            "clipped_frac": clipped / max(len(tr["actions"]), 1),
            "vocab": "K49",
        })
        if (idx + 1) % 5 == 0 or idx + 1 == len(trajs):
            print(f"  [{tag}] {idx+1}/{len(trajs)} NEW={phys['SUCCESS_NEW']}", flush=True)

    summary = {"by_skill": {}, "vocab": "K49", "cap": K49_CAP, "config": K49_CONFIG}
    for s in SKILLS:
        st = skill_stats[s]
        n = st["n"]
        k = st["new"]
        lo, hi = wilson_ci(k, n)
        summary["by_skill"][s] = {
            **st,
            "rate_NEW": (k / n) if n else float("nan"),
            "rate_12S": (st["s12"] / n) if n else float("nan"),
            "ci95_NEW": [lo, hi],
        }
    return rows, summary


# ---------------------------------------------------------------------------
# Residual grid / Pareto selection
# ---------------------------------------------------------------------------


def residual_candidates() -> list[ResidualBudget]:
    """Grid of scalar and per-dim residual budgets (legal [-1,1] interface)."""
    out = []
    for s in (0.15, 0.20, 0.25, 0.35):
        out.append(ResidualBudget.scalar(s))
    # Per-dim: elevate wx/vz (and mild vy) for catch, keep others moderate
    for base, wx, vz, vy in (
        (0.20, 0.40, 0.40, 0.30),
        (0.25, 0.40, 0.40, 0.30),
        (0.25, 0.55, 0.55, 0.40),
        (0.20, 0.55, 0.55, 0.35),
    ):
        r = np.full(7, base, dtype=np.float64)
        r[1] = vy
        r[2] = vz
        r[3] = wx
        out.append(ResidualBudget(rho=r))
    return out


def proxy_score_from_coverage(cov: dict) -> dict:
    """Aggregate construction proxy: catch-first, then mean of taus."""
    catch = cov.get("C/P_catch", {}).get("rate", 0.0)
    unmet = 0
    rates = []
    for k, v in cov.items():
        if v.get("n", 0) == 0:
            continue
        ph = k.split("/")[1]
        tau = PHASE_TAU.get(ph, 0.9)
        rates.append(min(1.0, v["rate"] / tau))
        if v["rate"] < tau:
            unmet += 1
    return {
        "catch_rate": catch,
        "mean_tau_ratio": float(np.mean(rates)) if rates else 0.0,
        "n_unmet": unmet,
    }


def select_pareto(candidates: list[dict]) -> tuple[dict | None, list[dict]]:
    """Select among construction-physics candidates with Pareto tradeoff.

    Primary: maximize C SUCCESS_NEW rate on construction.
    Among near-tied C (±1 success or rate within 0.02): prefer higher min(A, SP) rate,
    then lower mean residual saturation, then lower M, then smaller mean rho.
    """
    if not candidates:
        return None, []

    def key(c):
        by = c["physics_summary"]["by_skill"]
        c_rate = by["C"]["rate_NEW"]
        c_n = by["C"]["new"]
        a_rate = by["A"]["rate_NEW"]
        sp_rate = by["STABLE_PRECISION"]["rate_NEW"]
        b_rate = by["B"]["rate_NEW"]
        sat = float(c.get("mean_clipped_frac", 1.0))
        m = int(c["M"])
        mean_rho = float(np.mean(c["rho"]))
        # higher better for rates; negate sat/M/rho
        return (c_rate, c_n, min(a_rate, sp_rate), b_rate, -sat, -m, -mean_rho)

    ranked = sorted(candidates, key=key, reverse=True)
    best = ranked[0]
    # Mark Pareto frontier on (C_rate, -M, -mean_rho) roughly
    frontier = []
    best_c = -1.0
    for c in sorted(candidates, key=lambda x: -x["physics_summary"]["by_skill"]["C"]["rate_NEW"]):
        cr = c["physics_summary"]["by_skill"]["C"]["rate_NEW"]
        if cr >= best_c - 1e-12:
            # keep if not dominated on M/rho among equal C
            frontier.append(c)
            best_c = max(best_c, cr)
    return best, ranked


def passes_construction_gates(summary: dict, k49_summary: dict) -> dict:
    by = summary["by_skill"]
    k49 = k49_summary["by_skill"]
    # Construction gates are selection filters; held-out gates are final.
    gates = {
        "C_ge_0p80": by["C"]["rate_NEW"] >= 0.80 - 1e-12,
        "B_ok": by["B"]["rate_NEW"] >= 0.99 - 1e-12 or by["B"]["n"] == 0,
        "A_vs_k49": by["A"]["rate_NEW"] + 1e-12 >= k49["A"]["rate_NEW"] - (1.0 / max(by["A"]["n"], 1)),
        "SP_vs_k49": by["STABLE_PRECISION"]["rate_NEW"] + 1e-12
        >= k49["STABLE_PRECISION"]["rate_NEW"] - (1.0 / max(by["STABLE_PRECISION"]["n"], 1)),
    }
    gates["all"] = all(gates.values())
    return gates


def passes_heldout_gates(summary: dict, k49_summary: dict, delta: float = 1.0) -> dict:
    by = summary["by_skill"]
    k49 = k49_summary["by_skill"]
    gates = {
        "C_ge_0p80": by["C"]["rate_NEW"] >= 0.80 - 1e-12,
        "B_retains": by["B"]["new"] >= k49["B"]["new"] or by["B"]["rate_NEW"] >= 0.99 - 1e-12,
        "A_no_regress": by["A"]["new"] + 1e-12 >= k49["A"]["new"] - delta,
        "SP_no_regress": by["STABLE_PRECISION"]["new"] + 1e-12 >= k49["STABLE_PRECISION"]["new"] - delta,
    }
    gates["all"] = all(gates.values())
    return gates


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_reports(
    *,
    split_meta,
    tradeoff_rows,
    selected,
    construct_rows,
    held_rows,
    k49_construct_sum,
    k49_held_sum,
    held_sum,
    construct_sum,
    gates_c,
    gates_h,
    limiting,
):
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "CAPACITY_RESIDUAL_TRADEOFF.csv", tradeoff_rows)
    write_csv(OUT / "CONSTRUCT_PHYSICS_REPLAY.csv", construct_rows)
    write_csv(OUT / "HELDOUT_PHYSICS_REPLAY.csv", held_rows)

    atomic_json(OUT / "SPLIT.json", split_meta)
    atomic_json(
        OUT / "SELECTED.json",
        {
            "selected": selected,
            "construction_gates": gates_c,
            "heldout_gates": gates_h,
            "limiting": limiting,
        },
    )

    # Per-skill comparison table
    comp = []
    for split_name, sa, sk in (
        ("construct", construct_sum, k49_construct_sum),
        ("heldout", held_sum, k49_held_sum),
    ):
        for s in SKILLS:
            a = sa["by_skill"][s]
            b = sk["by_skill"][s]
            comp.append({
                "split": split_name,
                "skill": s,
                "hybrid_n": a["n"],
                "hybrid_NEW": a["new"],
                "hybrid_rate": a["rate_NEW"],
                "hybrid_ci95_lo": a["ci95_NEW"][0],
                "hybrid_ci95_hi": a["ci95_NEW"][1],
                "hybrid_12S": a["s12"],
                "hybrid_bilateral_loss": a["bilateral_loss"],
                "k49_NEW": b["new"],
                "k49_rate": b["rate_NEW"],
                "k49_ci95_lo": b["ci95_NEW"][0],
                "k49_ci95_hi": b["ci95_NEW"][1],
                "delta_NEW": a["new"] - b["new"],
            })
    write_csv(OUT / "PER_SKILL_VS_K49.csv", comp)

    ready = bool(gates_h.get("all"))
    md = []
    md.append("# Skill-Aware Hybrid Action Vocabulary — Validation Report")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("")
    md.append("## Selected configuration")
    if selected:
        md.append(f"- **M** = {selected['M']} motion prototypes (+ 1 HANDOFF id={selected['M']})")
        md.append(f"- **ρ** = `{selected['rho_label']}` → {selected['rho']}")
        md.append(f"- Weights (assign) = {selected.get('weights')}")
        md.append(f"- Artifacts: `{selected.get('prototypes_path')}`")
    else:
        md.append("- **No configuration selected** (see limiting analysis).")
    md.append("")
    md.append("## Held-out acceptance gates")
    for k, v in gates_h.items():
        md.append(f"- `{k}`: **{v}**")
    md.append("")
    md.append("## Per-skill SUCCESS_NEW (held-out)")
    md.append("")
    md.append("| Skill | Hybrid | K49 | Δ | 95% CI (hybrid) |")
    md.append("|---|---:|---:|---:|---|")
    for s in SKILLS:
        a = held_sum["by_skill"][s]
        b = k49_held_sum["by_skill"][s]
        md.append(
            f"| {s} | {a['new']}/{a['n']} ({a['rate_NEW']:.3f}) | {b['new']}/{b['n']} ({b['rate_NEW']:.3f}) | "
            f"{a['new']-b['new']:+d} | [{a['ci95_NEW'][0]:.3f}, {a['ci95_NEW'][1]:.3f}] |"
        )
    md.append("")
    md.append("## Construction-set SUCCESS_NEW (selection)")
    md.append("")
    md.append("| Skill | Hybrid | K49 |")
    md.append("|---|---:|---:|")
    for s in SKILLS:
        a = construct_sum["by_skill"][s]
        b = k49_construct_sum["by_skill"][s]
        md.append(f"| {s} | {a['new']}/{a['n']} ({a['rate_NEW']:.3f}) | {b['new']}/{b['n']} ({b['rate_NEW']:.3f}) |")
    md.append("")
    md.append("## Capacity–residual tradeoff (construction physics)")
    md.append("")
    md.append("| cand | M | ρ | C_NEW | A_NEW | SP_NEW | B_NEW | clip_frac | catch_proxy |")
    md.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in tradeoff_rows:
        md.append(
            f"| {r.get('cand_id')} | {r.get('M')} | {r.get('rho_label')} | {r.get('C_NEW')} | "
            f"{r.get('A_NEW')} | {r.get('SP_NEW')} | {r.get('B_NEW')} | {r.get('mean_clipped_frac')} | {r.get('catch_proxy')} |"
        )
    md.append("")
    md.append("## Recommendation")
    if ready:
        md.append(
            "**Representation is ready for policy-training data packing** "
            "(oracle hybrid representability gates passed on held-out). "
            "This does **not** claim a trained policy can predict tokens/residuals."
        )
    else:
        md.append(
            "**Not ready for policy training** under the agreed gates. "
            f"Limiting: `{jsafe(limiting)}`. "
            "Do not silently relax thresholds; a hybrid continuous-C Stage-1 fallback "
            "requires separate approval."
        )
    md.append("")
    md.append("## Notes")
    md.append("- Oracle reconstruction = expert action → nearest prototype + clipped residual.")
    md.append("- Frozen K49 baseline unchanged (`K49_tok48_wy0p25`, CAP=0.25).")
    md.append("- No RL/SFT; experts/physics/success criteria unmodified.")
    (OUT / "HELDOUT_VALIDATION_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "RECOMMENDATION.md").write_text(
        "\n".join(
            [
                "# Recommendation",
                "",
                "READY" if ready else "NOT_READY",
                "",
                md[-7] if len(md) > 7 else "",
                "",
                f"Selected M={None if not selected else selected['M']}, "
                f"rho={None if not selected else selected.get('rho_label')}",
                f"Held-out gates: {json.dumps(gates_h)}",
            ]
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Fewer residual candidates / smaller physics set smoke test")
    ap.add_argument("--skip-k49-construct", action="store_true")
    ap.add_argument("--skip-k49-heldout", action="store_true")
    ap.add_argument("--reuse-k49", action="store_true", help="Reuse K49_*_SUMMARY.json if present")
    ap.add_argument("--max-physics-candidates", type=int, default=6)
    args = ap.parse_args()
    if args.reuse_k49:
        args.skip_k49_construct = True
        args.skip_k49_heldout = True

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT / "candidates").mkdir(parents=True, exist_ok=True)

    print("Loading experts…", flush=True)
    rows = load_expert_rows()
    construct_rows, held_rows_m, split_meta = stratified_lineage_split(rows, seed=SEED)
    atomic_json(OUT / "SPLIT.json", split_meta)
    print(
        f"Experts={len(rows)} construct={len(construct_rows)} heldout={len(held_rows_m)}",
        flush=True,
    )

    print("Loading trajectories…", flush=True)
    construct_trajs = [load_traj(r) for r in construct_rows]
    held_trajs = [load_traj(r) for r in held_rows_m]

    env = make_env()
    k49_table = build_k49_table(config_id=K49_CONFIG)
    w = DEFAULT_WEIGHTS.copy()

    # K49 baselines on both splits
    print("K49 baseline on construction…", flush=True)
    if args.skip_k49_construct and (OUT / "K49_CONSTRUCT_SUMMARY.json").exists():
        k49_c_sum = json.loads((OUT / "K49_CONSTRUCT_SUMMARY.json").read_text(encoding="utf-8"))
        k49_c_rows = []
        print("  reused K49_CONSTRUCT_SUMMARY.json", flush=True)
    else:
        k49_c_rows, k49_c_sum = evaluate_k49_baseline(env, construct_trajs, k49_table, tag="K49_construct")
        write_csv(OUT / "K49_CONSTRUCT_PHYSICS.csv", k49_c_rows)
        atomic_json(OUT / "K49_CONSTRUCT_SUMMARY.json", k49_c_sum)

    print("K49 baseline on held-out…", flush=True)
    if args.skip_k49_heldout and (OUT / "K49_HELDOUT_SUMMARY.json").exists():
        k49_h_sum = json.loads((OUT / "K49_HELDOUT_SUMMARY.json").read_text(encoding="utf-8"))
        k49_h_rows = []
        print("  reused K49_HELDOUT_SUMMARY.json", flush=True)
    else:
        k49_h_rows, k49_h_sum = evaluate_k49_baseline(env, held_trajs, k49_table, tag="K49_heldout")
        write_csv(OUT / "K49_HELDOUT_PHYSICS.csv", k49_h_rows)
        atomic_json(OUT / "K49_HELDOUT_SUMMARY.json", k49_h_sum)

    # Build + proxy-screen candidates
    rhos = residual_candidates()
    if args.quick:
        rhos = [ResidualBudget.scalar(0.25), ResidualBudget(rho=np.array([0.25, 0.30, 0.40, 0.40, 0.25, 0.25, 0.25]))]

    proxy_candidates = []
    print(f"Constructing codebooks for {len(rhos)} residual budgets…", flush=True)
    for i, rho in enumerate(rhos):
        cb, clog = construct_codebook(construct_trajs, rho, w=w, m_max=M_MAX, seed=SEED + i)
        prox = proxy_score_from_coverage(clog["phase_coverage"])
        cand_id = f"c{i:02d}_M{cb.M}_{rho.label().replace('=', '').replace(',', '_').replace('[', '').replace(']', '')[:48]}"
        art_dir = OUT / "candidates" / cand_id
        meta = cb.save(art_dir, stem="codebook")
        atomic_json(art_dir / "construction_log.json", clog)
        proxy_candidates.append({
            "cand_id": cand_id,
            "M": cb.M,
            "rho": rho.to_list(),
            "rho_label": rho.label(),
            "weights": [float(x) for x in w],
            "proxy": prox,
            "unmet": clog["unmet"],
            "prototypes_path": meta["prototypes_path"],
            "book": cb.prototypes,
            "residual": rho,
            "phase_coverage": clog["phase_coverage"],
        })
        print(
            f"  {cand_id}: M={cb.M} catch_proxy={prox['catch_rate']:.3f} unmet={prox['n_unmet']}",
            flush=True,
        )

    # Rank by proxy for physics shortlist: catch coverage, unmet, then prefer moderate M/rho
    def proxy_rank(c):
        mean_rho = float(np.mean(c["rho"]))
        return (
            c["proxy"]["catch_rate"],
            c["proxy"]["mean_tau_ratio"],
            -c["proxy"]["n_unmet"],
            -abs(mean_rho - 0.25),  # mild preference near interface-familiar scale, not hard
            -c["M"],
        )

    proxy_ranked = sorted(proxy_candidates, key=proxy_rank, reverse=True)
    shortlist = proxy_ranked[: max(1, args.max_physics_candidates)]
    # Always include best catch + a high-rho and low-rho extreme if not present
    for prefer in proxy_ranked:
        if prefer["cand_id"] not in {c["cand_id"] for c in shortlist}:
            if prefer["M"] <= 64 or float(np.mean(prefer["rho"])) <= 0.20:
                shortlist.append(prefer)
        if len(shortlist) >= args.max_physics_candidates + 2:
            break

    atomic_json(
        OUT / "PROXY_CANDIDATES.json",
        [{k: v for k, v in c.items() if k not in ("book", "residual")} for c in proxy_ranked],
    )

    # Physics on construction shortlist
    physics_candidates = []
    all_construct_replay_rows = []
    tradeoff_rows = []
    print(f"Physics replay on {len(shortlist)} construction candidates…", flush=True)
    for c in shortlist:
        print(f"== Physics {c['cand_id']} ==", flush=True)
        rows_p, summ = evaluate_vocab_on_trajs(
            env, construct_trajs, c["book"], c["residual"], w, tag=c["cand_id"], early_c=True
        )
        for r in rows_p:
            r["cand_id"] = c["cand_id"]
            r["M"] = c["M"]
            r["rho_label"] = c["rho_label"]
        all_construct_replay_rows.extend(rows_p)
        mean_clip = float(np.mean([r["clipped_frac"] for r in rows_p])) if rows_p else 1.0
        gates = passes_construction_gates(summ, k49_c_sum)
        entry = {
            **{k: v for k, v in c.items() if k not in ("book", "residual")},
            "physics_summary": summ,
            "mean_clipped_frac": mean_clip,
            "construction_gates": gates,
        }
        physics_candidates.append(entry)
        tradeoff_rows.append({
            "cand_id": c["cand_id"],
            "M": c["M"],
            "rho_label": c["rho_label"],
            "rho": json.dumps(c["rho"]),
            "catch_proxy": c["proxy"]["catch_rate"],
            "n_unmet_proxy": c["proxy"]["n_unmet"],
            "C_NEW": f"{summ['by_skill']['C']['new']}/{summ['by_skill']['C']['n']}",
            "C_rate": summ["by_skill"]["C"]["rate_NEW"],
            "A_NEW": f"{summ['by_skill']['A']['new']}/{summ['by_skill']['A']['n']}",
            "SP_NEW": f"{summ['by_skill']['STABLE_PRECISION']['new']}/{summ['by_skill']['STABLE_PRECISION']['n']}",
            "B_NEW": f"{summ['by_skill']['B']['new']}/{summ['by_skill']['B']['n']}",
            "mean_clipped_frac": mean_clip,
            "gates_all": gates["all"],
            "C_early_mae_wx": (summ.get("C_early") or {}).get("mean_mae_wx"),
            "C_early_mae_vz": (summ.get("C_early") or {}).get("mean_mae_vz"),
        })
        atomic_json(OUT / "candidates" / c["cand_id"] / "construct_physics_summary.json", summ)

    write_csv(OUT / "CONSTRUCT_PHYSICS_REPLAY.csv", all_construct_replay_rows)
    write_csv(OUT / "CAPACITY_RESIDUAL_TRADEOFF.csv", tradeoff_rows)
    atomic_json(
        OUT / "PHYSICS_CANDIDATES.json",
        [{k: v for k, v in e.items() if k != "book"} for e in physics_candidates],
    )

    # Prefer candidates passing construction gates; else best Pareto among all
    passing = [e for e in physics_candidates if e["construction_gates"]["all"]]
    pool = passing if passing else physics_candidates
    selected_entry, ranked = select_pareto(pool)
    limiting = None
    if not passing:
        # diagnose limiting skill/phase on best effort
        best_effort = ranked[0] if ranked else None
        limiting = {
            "status": "no_candidate_passed_construction_gates",
            "best_effort_cand": None if best_effort is None else best_effort["cand_id"],
            "best_effort_gates": None if best_effort is None else best_effort["construction_gates"],
            "best_effort_C": None
            if best_effort is None
            else best_effort["physics_summary"]["by_skill"]["C"],
            "unmet_proxy": None if best_effort is None else best_effort.get("unmet"),
            "note": "Held-out will still run once on best-effort selection; do not redesign from held-out.",
        }

    assert selected_entry is not None
    # Reload book array
    sel_book = np.load(ROOT / selected_entry["prototypes_path"]).astype(np.float32)
    sel_rho = ResidualBudget.from_spec(selected_entry["rho"])
    # Freeze selected
    frozen = MotionCodebook(
        prototypes=sel_book,
        residual=sel_rho,
        weights=w,
        meta={
            "cand_id": selected_entry["cand_id"],
            "selected_on": "construction_physics_pareto",
            "M_max": M_MAX,
            "k49_baseline": K49_CONFIG,
            "seed": SEED,
        },
    )
    fr_meta = frozen.save(OUT / "artifacts", stem="SELECTED_skill_aware_hybrid")
    selected_public = {
        **fr_meta,
        "cand_id": selected_entry["cand_id"],
        "construction_gates": selected_entry["construction_gates"],
        "construct_physics": selected_entry["physics_summary"]["by_skill"],
        "mean_clipped_frac": selected_entry["mean_clipped_frac"],
        "rho_label": selected_entry["rho_label"],
        "rho": selected_entry["rho"],
        "weights": selected_entry["weights"],
        "prototypes_path": fr_meta["prototypes_path"],
    }
    atomic_json(OUT / "SELECTED.json", selected_public)

    # Copy selection log
    src_log = OUT / "candidates" / selected_entry["cand_id"] / "construction_log.json"
    if src_log.exists():
        (OUT / "artifacts" / "PROTOTYPE_SELECTION_LOG.json").write_text(
            src_log.read_text(encoding="utf-8"), encoding="utf-8"
        )

    print("Held-out physics on SELECTED (single shot)…", flush=True)
    held_replay, held_sum = evaluate_vocab_on_trajs(
        env, held_trajs, sel_book, sel_rho, w, tag="SELECTED_heldout", early_c=True
    )
    write_csv(OUT / "HELDOUT_PHYSICS_REPLAY.csv", held_replay)
    atomic_json(OUT / "HELDOUT_SUMMARY.json", held_sum)

    gates_c = selected_entry["construction_gates"]
    gates_h = passes_heldout_gates(held_sum, k49_h_sum, delta=1.0)
    if not gates_h["all"] and limiting is None:
        # identify limiting skill
        lim_skill = None
        for s in ("C", "A", "STABLE_PRECISION", "B"):
            # map gate names
            pass
        limiting = {
            "status": "heldout_gates_failed",
            "gates": gates_h,
            "heldout_by_skill": held_sum["by_skill"],
            "failures": [
                {
                    "sample_id": r["sample_id"],
                    "skill": r["skill"],
                    "fail_category": r["fail_category"],
                    "first_div_t": r["first_div_t"],
                    "early_mae_wx": r.get("early_mae_wx"),
                    "early_mae_vz": r.get("early_mae_vz"),
                }
                for r in held_replay
                if not r["SUCCESS_NEW"]
            ],
        }
    atomic_json(OUT / "REMAINING_FAILURES.json", limiting or {"status": "none", "gates": gates_h})

    # Also dump construct summary for selected
    construct_sum = selected_entry["physics_summary"]

    write_reports(
        split_meta=split_meta,
        tradeoff_rows=tradeoff_rows,
        selected=selected_public,
        construct_rows=all_construct_replay_rows,
        held_rows=held_replay,
        k49_construct_sum=k49_c_sum,
        k49_held_sum=k49_h_sum,
        held_sum=held_sum,
        construct_sum=construct_sum,
        gates_c=gates_c,
        gates_h=gates_h,
        limiting=limiting,
    )

    # Revised design spec stamp
    (OUT / "DESIGN_SPEC_REVISED.md").write_text(
        "\n".join(
            [
                "# Revised Design Spec — Skill-Aware Hybrid Vocabulary",
                "",
                "Status: **implemented & validated** (see HELDOUT_VALIDATION_REPORT.md).",
                "",
                "## Hybrid decoder (deployed form)",
                "",
                "$$a_t = \\mathrm{clip}(c_{k_t} + \\mathrm{clip}(r_t, -\\rho, \\rho), -1, 1)$$",
                "",
                "with optional per-dimension $\\rho\\in\\mathbb{R}_+^7$. HANDOFF is token id $M$, not a prototype.",
                "",
                "## Capacity–residual selection",
                "",
                "Do **not** minimize $M$ by inflating $\\rho$. Construction procedure:",
                "1. Build phase-aware codebook for each $\\rho$ on the construction split only.",
                "2. Screen by phase-stratified weighted-L∞ proxy (C `P_catch` prioritized).",
                "3. Physics-replay a shortlist; select via Pareto/rank on",
                "   (C SUCCESS_NEW, min(A, SP) SUCCESS_NEW, B, −saturation, −M, −mean ρ).",
                "4. Freeze; evaluate **once** on held-out lineages.",
                "",
                f"## Selected",
                "",
                f"- M = {selected_public['M']}",
                f"- ρ = {selected_public['rho_label']} {selected_public['rho']}",
                f"- Held-out gates passed: {gates_h.get('all')}",
                "",
                f"K49 baseline frozen: `{K49_CONFIG}` CAP={K49_CAP}.",
            ]
        ),
        encoding="utf-8",
    )

    print("DONE", json.dumps(jsafe({
        "M": selected_public["M"],
        "rho": selected_public["rho_label"],
        "heldout_gates": gates_h,
        "heldout_C": held_sum["by_skill"]["C"],
        "heldout_A": held_sum["by_skill"]["A"],
        "heldout_SP": held_sum["by_skill"]["STABLE_PRECISION"],
        "heldout_B": held_sum["by_skill"]["B"],
    })), flush=True)


if __name__ == "__main__":
    main()
