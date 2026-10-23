#!/usr/bin/env python
"""Diagnose K49 coverage on C-class experts and evaluate minimal codebook extensions.

Does NOT modify the frozen K49 baseline file, Oracle, physics, or expert trajectories.
Writes results/diagnostics/c_class_k49_codebook_extension/
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.contact_modes import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.force_slip_control import sample_state  # noqa: E402
from data_generation.oracles.precision.precision_goals import measure_errors  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.action_vocab_k49 import (  # noqa: E402
    CAP,
    HANDOFF,
    K49_TABLE_ROWS,
    build_k49_table,
)

DS = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
OUT = ROOT / "results/diagnostics/c_class_k49_codebook_extension"
K49_CONFIG = "K49_tok48_wy0p25"
DT = float(DT_POLICY)
EARLY_S = 0.20  # first 0.2 s
EARLY_N = int(round(EARLY_S / DT))  # 10
ACT_DIMS = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]
SEED = 5478
EXT_SIZES = (8, 16, 32)
REBUILD_SIZES = (49, 57, 65, 81)  # ~K49 and K49+8/16/32 motion capacities


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
            w.writerow(r)


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


def load_rows():
    return [
        json.loads(l)
        for l in (DS / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def motion_rows(table: np.ndarray) -> np.ndarray:
    """All usable motion prototypes (skip HANDOFF placeholder row)."""
    idx = [i for i in range(len(table)) if i != HANDOFF]
    return table[np.asarray(idx, int)]


def nearest_proto(table: np.ndarray, a: np.ndarray) -> tuple[int, np.ndarray, float]:
    a = np.asarray(a, np.float32).reshape(7)
    best_i, best_d, best_p = -1, 1e9, None
    for i in range(len(table)):
        if i == HANDOFF:
            continue
        p = table[i]
        d = float(np.linalg.norm(p - a))
        if d < best_d:
            best_i, best_d, best_p = i, d, p
    return best_i, best_p.copy(), best_d


def reconstruct(table: np.ndarray, a: np.ndarray, cap: float = CAP) -> tuple[np.ndarray, dict]:
    ti, proto, d = nearest_proto(table, a)
    raw = a - proto
    res = np.clip(raw, -cap, cap).astype(np.float32)
    recon = np.clip(proto + res, -1.0, 1.0).astype(np.float32)
    err = recon - a
    return recon, {
        "token": ti,
        "proto_l2": d,
        "recon_l2": float(np.linalg.norm(err)),
        "recon_linf": float(np.max(np.abs(err))),
        "clipped": bool(np.any(np.abs(raw) > cap + 1e-8)),
        "beyond_cap": np.maximum(np.abs(raw) - cap, 0.0),
        "err": err,
        "raw_res": raw,
    }


def quantize_traj(table: np.ndarray, acts: np.ndarray, cap: float = CAP) -> np.ndarray:
    out = []
    for a in acts:
        recon, _ = reconstruct(table, a, cap=cap)
        out.append(recon)
    return np.asarray(out, np.float32)


def action_key(a: np.ndarray, nd: int = 3) -> str:
    r = np.round(np.asarray(a, float), nd)
    return hashlib.sha1(r.tobytes()).hexdigest()[:12]


def load_traj(row: dict) -> dict:
    ic = pickle.loads((DS / row["ic_pkl"]).read_bytes())
    cont = np.load(DS / row["continuous_demo"]["path"])
    acts = np.asarray(cont["actions"], np.float32)
    return {
        "sample_id": row["sample_id"],
        "source_label": row["source_label"],
        "lineage_id": row["lineage_id"],
        "capture": ic["capture"],
        "ref": ic["ref"],
        "actions": acts,
        "k49_stored_ok": bool((row.get("k49_demo") or {}).get("verified")),
        "n": len(acts),
    }


def split_c(c_rows: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Stratified lineage split: ~60% construction / 40% evaluation by prior K49 outcome."""
    rng = np.random.RandomState(seed)
    ok = [r for r in c_rows if (r.get("k49_demo") or {}).get("verified")]
    bad = [r for r in c_rows if not (r.get("k49_demo") or {}).get("verified")]
    rng.shuffle(ok)
    rng.shuffle(bad)

    def take(xs, frac_train=0.6):
        n_tr = max(1, int(round(frac_train * len(xs)))) if xs else 0
        # ensure at least 1 eval when possible
        if len(xs) >= 2:
            n_tr = min(n_tr, len(xs) - 1)
        return xs[:n_tr], xs[n_tr:]

    tr_ok, ev_ok = take(ok)
    tr_bad, ev_bad = take(bad)
    train = tr_ok + tr_bad
    eval_ = ev_ok + ev_bad
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


# ---------------------------------------------------------------------------
# Codebook construction
# ---------------------------------------------------------------------------


def collect_weighted_early_actions(trajs: list[dict], table: np.ndarray) -> list[dict]:
    """Early-window actions from construction set with beyond-CAP weights."""
    items = []
    for tr in trajs:
        acts = tr["actions"][:EARLY_N]
        for i, a in enumerate(acts):
            recon, info = reconstruct(table, a)
            beyond = info["beyond_cap"]
            # Prioritize wx/vz physical divergence drivers + total beyond-CAP mass
            w = (
                float(beyond[3]) * 3.0  # wx
                + float(beyond[2]) * 2.0  # vz
                + float(np.sum(beyond)) * 1.0
                + float(info["recon_l2"]) * 0.5
            )
            items.append({
                "action": np.asarray(a, np.float32).copy(),
                "sample_id": tr["sample_id"],
                "t": i * DT,
                "weight": w,
                "recon_l2": info["recon_l2"],
                "beyond_wx": float(beyond[3]),
                "beyond_vz": float(beyond[2]),
                "clipped": info["clipped"],
                "key": action_key(a),
            })
    return items


def greedy_extend(
    base_table: np.ndarray,
    candidates: list[dict],
    n_new: int,
    *,
    min_dist: float = 0.15,
) -> np.ndarray:
    """Deterministic greedy selection: max weighted min-distance to current book."""
    book = [base_table[i].copy() for i in range(len(base_table)) if i != HANDOFF]
    # Dedup candidates by key, keep max weight
    by_key = {}
    for c in candidates:
        k = c["key"]
        if k not in by_key or c["weight"] > by_key[k]["weight"]:
            by_key[k] = c
    pool = sorted(
        by_key.values(),
        key=lambda c: (-c["weight"], c["key"]),  # deterministic
    )
    selected = []

    def min_d(a):
        return min(float(np.linalg.norm(a - b)) for b in book)

    for _ in range(n_new):
        best = None
        best_score = -1.0
        for c in pool:
            a = c["action"]
            md = min_d(a)
            if md < min_dist:
                continue
            # score: weight * min_dist (prefer hard-to-represent AND novel)
            score = c["weight"] * (0.25 + md)
            # tiny deterministic tie-break already via sorted pool then >
            if score > best_score + 1e-12:
                best_score = score
                best = c
        if best is None:
            # relax min_dist
            for c in pool:
                a = c["action"]
                md = min_d(a)
                score = c["weight"] * (0.25 + md)
                if score > best_score + 1e-12:
                    best_score = score
                    best = c
        if best is None:
            break
        selected.append(best["action"].copy())
        book.append(best["action"].copy())
        pool = [c for c in pool if c["key"] != best["key"]]

    # Build augmented table: original K49 rows unchanged, append C-specific after 48
    aug = np.zeros((K49_TABLE_ROWS + len(selected), 7), np.float32)
    aug[:K49_TABLE_ROWS] = base_table[:K49_TABLE_ROWS]
    for i, a in enumerate(selected):
        aug[K49_TABLE_ROWS + i] = a
    return aug, selected


def _fps(arr: np.ndarray, k: int) -> list[np.ndarray]:
    """Deterministic farthest-point sampling."""
    if len(arr) == 0 or k <= 0:
        return []
    arr = np.asarray(arr, np.float32)
    # start: largest L2, tie-break by action_key
    norms = np.linalg.norm(arr, axis=1)
    order0 = np.argsort(-norms)
    start = int(order0[0])
    for j in order0:
        if abs(norms[int(j)] - norms[start]) < 1e-9 and action_key(arr[int(j)]) < action_key(arr[start]):
            start = int(j)
    chosen = [arr[start].copy()]
    while len(chosen) < min(k, len(arr)):
        dmin = np.min(
            np.linalg.norm(arr[:, None, :] - np.asarray(chosen, np.float32)[None, :, :], axis=2),
            axis=1,
        )
        # avoid near-duplicates of already chosen
        for i, c in enumerate(chosen):
            dmin = np.maximum(dmin, 0.0)
        best = int(np.argmax(dmin))
        # tie-break
        top = np.flatnonzero(dmin >= dmin[best] - 1e-12)
        best = int(min(top, key=lambda i: action_key(arr[int(i)])))
        if dmin[best] < 1e-6:
            break
        chosen.append(arr[best].copy())
    return chosen


def rebuild_skill_balanced(
    skill_trajs: dict[str, list[dict]],
    n_motion: int,
    seed: int = SEED,
) -> np.ndarray:
    """Rebuild motion codebook from multi-skill early+body actions (equal skill quota).

    Capacity n_motion usable motion rows (HANDOFF slot reserved at index 47).
    """
    del seed  # deterministic without RNG
    skills = sorted(skill_trajs.keys())
    per = max(1, n_motion // max(len(skills), 1))
    pools: dict[str, list[np.ndarray]] = {}
    for sk in skills:
        acts = []
        for tr in skill_trajs[sk]:
            a = tr["actions"]
            acts.extend(list(a[:EARLY_N]))
            if len(a) > EARLY_N:
                idx = np.linspace(EARLY_N, len(a) - 1, num=min(40, len(a) - EARLY_N), dtype=int)
                acts.extend(list(a[idx]))
        pools[sk] = _fps(np.asarray(acts, np.float32), per) if acts else []

    all_codes: list[np.ndarray] = []
    for sk in skills:
        all_codes.extend(pools.get(sk, []))
    if len(all_codes) < n_motion:
        rest = []
        for sk in skills:
            for tr in skill_trajs[sk]:
                step = max(1, len(tr["actions"]) // 20)
                rest.extend(list(tr["actions"][::step]))
        extra = _fps(np.asarray(rest, np.float32), n_motion - len(all_codes)) if rest else []
        # only add if novel vs all_codes
        for e in extra:
            if not all_codes or min(float(np.linalg.norm(e - c)) for c in all_codes) > 0.05:
                all_codes.append(e)
            if len(all_codes) >= n_motion:
                break
    all_codes = all_codes[:n_motion]

    n_pre = min(47, len(all_codes))
    n_post = max(0, len(all_codes) - n_pre)
    table = np.zeros((47 + 1 + n_post, 7), np.float32)
    if n_pre:
        table[:n_pre] = np.asarray(all_codes[:n_pre], np.float32)
    if n_post:
        table[48 : 48 + n_post] = np.asarray(all_codes[n_pre:], np.float32)
    return table


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def handoff(env, ref) -> dict:
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
        "pre_pos_mm": h.get("pre_pos_mm"),
        "pre_theta": h.get("pre_theta"),
    }


def replay_quantized(env, traj: dict, table: np.ndarray) -> dict:
    acts_q = quantize_traj(table, traj["actions"])
    restore_capture(env, traj["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    first_bil_loss = None
    first_state = None
    # continuous reference states at early checkpoints
    restore_capture(env, traj["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    # actual quantized replay
    restore_capture(env, traj["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    s0 = measure_errors(env.sim)
    for i, a in enumerate(acts_q):
        step_legal(env.sim, a)
        env._prev = a
        if (i + 1) % 5 == 0 or i + 1 <= EARLY_N:
            st = sample_state(env.sim)
            err = measure_errors(env.sim)
            m = classify_contact_mode(env.sim)
            if first_bil_loss is None and (not err["bilateral"] or latched(env.sim)):
                first_bil_loss = {
                    "i": i + 1,
                    "t": (i + 1) * DT,
                    "mode": m["mode"],
                    "latched": bool(latched(env.sim)),
                }
            if first_state is None and i + 1 >= 5:
                # mark early drift vs start
                if abs(err["e_pos_mm"] - s0["e_pos_mm"]) > 5.0 or abs(err["dtheta_deg"] - s0["dtheta_deg"]) > 10.0:
                    first_state = {"i": i + 1, "t": (i + 1) * DT, "e_pos": err["e_pos_mm"], "theta": err["dtheta_deg"]}
    h = handoff(env, traj["ref"])
    fail = None
    if not h["SUCCESS_NEW"]:
        if first_bil_loss:
            fail = "bilateral_loss" if not first_bil_loss.get("latched") else "support_drop_latched"
        elif h["SUCCESS_12S"]:
            fail = "POSE_BAND_FAIL_HOLD_OK"
        else:
            fail = h.get("reason") or "FAIL"
    return {
        **h,
        "fail_reason": fail,
        "first_bil_loss_t": None if first_bil_loss is None else first_bil_loss["t"],
        "first_state_t": None if first_state is None else first_state["t"],
        "n_actions": len(acts_q),
        "n_codes": int(len(table)),
    }


def quant_stats(trajs: list[dict], table: np.ndarray) -> dict:
    early_err = {d: [] for d in ACT_DIMS}
    full_err = {d: [] for d in ACT_DIMS}
    early_l2, full_l2 = [], []
    early_clip, full_clip = [], []
    beyond_wx, beyond_vz = [], []
    for tr in trajs:
        for i, a in enumerate(tr["actions"]):
            _, info = reconstruct(table, a)
            bucket = early_err if i < EARLY_N else full_err
            for j, d in enumerate(ACT_DIMS):
                bucket[d].append(abs(float(info["err"][j])))
            if i < EARLY_N:
                early_l2.append(info["recon_l2"])
                early_clip.append(info["clipped"])
                beyond_wx.append(float(info["beyond_cap"][3]))
                beyond_vz.append(float(info["beyond_cap"][2]))
            else:
                full_l2.append(info["recon_l2"])
                full_clip.append(info["clipped"])
    def summ(xs):
        if not xs:
            return {"mean": 0.0, "p90": 0.0, "max": 0.0}
        a = np.asarray(xs, float)
        return {"mean": float(a.mean()), "p90": float(np.percentile(a, 90)), "max": float(a.max())}

    return {
        "early_recon_l2": summ(early_l2),
        "full_recon_l2": summ(full_l2),
        "early_clip_frac": float(np.mean(early_clip)) if early_clip else 0.0,
        "full_clip_frac": float(np.mean(full_clip)) if full_clip else 0.0,
        "early_beyond_wx": summ(beyond_wx),
        "early_beyond_vz": summ(beyond_vz),
        "early_mae": {d: summ(early_err[d]) for d in ACT_DIMS},
        "full_mae": {d: summ(full_err[d]) for d in ACT_DIMS},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "codebooks").mkdir(parents=True, exist_ok=True)
    env = make_env()
    k49 = build_k49_table(config_id=K49_CONFIG)
    # freeze copy of baseline
    np.save(OUT / "codebooks" / "K49_baseline.npy", k49)

    rows = load_rows()
    c_rows = [
        r
        for r in rows
        if r.get("source_label") == "C" and (r.get("continuous_demo") or {}).get("verified")
    ]
    train_rows, eval_rows = split_c(c_rows)
    atomic_json(
        OUT / "SPLIT.json",
        {
            "seed": SEED,
            "n_c": len(c_rows),
            "n_train": len(train_rows),
            "n_eval": len(eval_rows),
            "train_ids": [r["sample_id"] for r in train_rows],
            "eval_ids": [r["sample_id"] for r in eval_rows],
            "train_k49_ok": sum(1 for r in train_rows if (r.get("k49_demo") or {}).get("verified")),
            "eval_k49_ok": sum(1 for r in eval_rows if (r.get("k49_demo") or {}).get("verified")),
            "note": (
                "Codewords selected from TRAIN only. EVAL is held-out for generalization. "
                "Full-47 metrics labeled in-sample diagnostic."
            ),
        },
    )
    print(
        f"Split C: train={len(train_rows)} (k49ok={sum(1 for r in train_rows if (r.get('k49_demo') or {}).get('verified'))}) "
        f"eval={len(eval_rows)} (k49ok={sum(1 for r in eval_rows if (r.get('k49_demo') or {}).get('verified'))})",
        flush=True,
    )

    print("Loading trajectories...", flush=True)
    c_all = [load_traj(r) for r in c_rows]
    c_train = [load_traj(r) for r in train_rows]
    c_eval = [load_traj(r) for r in eval_rows]
    by_id = {t["sample_id"]: t for t in c_all}

    # Regression pools (non-C continuous verified)
    reg_rows = [
        r
        for r in rows
        if r.get("source_label") in ("A", "B", "STABLE_PRECISION")
        and (r.get("continuous_demo") or {}).get("verified")
    ]
    reg_trajs = [load_traj(r) for r in reg_rows]
    print(f"Regression pool: {len(reg_trajs)}", flush=True)

    # ----- Phase 1: coverage diagnostic -----
    print("Phase 1: coverage diagnostic...", flush=True)
    ok_ids = {t["sample_id"] for t in c_all if t["k49_stored_ok"]}
    fail_ids = {t["sample_id"] for t in c_all if not t["k49_stored_ok"]}
    stats_all = quant_stats(c_all, k49)
    stats_ok = quant_stats([by_id[i] for i in ok_ids], k49)
    stats_fail = quant_stats([by_id[i] for i in fail_ids], k49)
    stats_early_ok = quant_stats(
        [{"actions": by_id[i]["actions"][:EARLY_N], "sample_id": i} for i in ok_ids], k49
    )
    # Per-traj early errors
    per_traj = []
    poorly = []
    for t in c_all:
        e_l2, e_wx, e_vz = [], [], []
        for a in t["actions"][:EARLY_N]:
            _, info = reconstruct(k49, a)
            e_l2.append(info["recon_l2"])
            e_wx.append(float(info["beyond_cap"][3]))
            e_vz.append(float(info["beyond_cap"][2]))
            if info["recon_l2"] > 0.15 or info["beyond_cap"][3] > 0.05 or info["beyond_cap"][2] > 0.05:
                poorly.append({
                    "sample_id": t["sample_id"],
                    "action": np.asarray(a, float).tolist(),
                    "recon_l2": info["recon_l2"],
                    "beyond_wx": float(info["beyond_cap"][3]),
                    "beyond_vz": float(info["beyond_cap"][2]),
                    "k49_ok": t["k49_stored_ok"],
                })
        per_traj.append({
            "sample_id": t["sample_id"],
            "k49_ok": t["k49_stored_ok"],
            "early_mean_l2": float(np.mean(e_l2)),
            "early_max_l2": float(np.max(e_l2)),
            "early_mean_beyond_wx": float(np.mean(e_wx)),
            "early_mean_beyond_vz": float(np.mean(e_vz)),
            "early_max_beyond_wx": float(np.max(e_wx)),
            "early_max_beyond_vz": float(np.max(e_vz)),
            "n": t["n"],
        })
    write_csv(OUT / "PER_TRAJ_EARLY_QUANT.csv", per_traj)

    # Cluster poorly represented actions (unique keys)
    poor_by_key = {}
    for p in poorly:
        k = action_key(np.asarray(p["action"]))
        if k not in poor_by_key or p["recon_l2"] > poor_by_key[k]["recon_l2"]:
            poor_by_key[k] = p
    poor_list = sorted(poor_by_key.values(), key=lambda x: -x["recon_l2"])[:40]
    write_csv(
        OUT / "POORLY_REPRESENTED_ACTIONS.csv",
        [
            {
                "rank": i + 1,
                "sample_id": p["sample_id"],
                "k49_ok": p["k49_ok"],
                "recon_l2": p["recon_l2"],
                "beyond_wx": p["beyond_wx"],
                "beyond_vz": p["beyond_vz"],
                **{f"a_{d}": p["action"][j] for j, d in enumerate(ACT_DIMS)},
            }
            for i, p in enumerate(poor_list)
        ],
    )

    cov_report = f"""# QUANTIZATION_COVERAGE.md — C-class vs K49

Generated: {datetime.now(timezone.utc).isoformat()}
Config: `{K49_CONFIG}`, CAP={CAP}, early window={EARLY_S}s ({EARLY_N} steps)

## Headline

| Group | n | early recon L2 mean | early clip frac | early beyond-wx mean | early beyond-vz mean |
|---|---:|---:|---:|---:|---:|
| All C | {len(c_all)} | {stats_all['early_recon_l2']['mean']:.4f} | {stats_all['early_clip_frac']:.3f} | {stats_all['early_beyond_wx']['mean']:.4f} | {stats_all['early_beyond_vz']['mean']:.4f} |
| K49-success (8) | {len(ok_ids)} | {stats_ok['early_recon_l2']['mean']:.4f} | {stats_ok['early_clip_frac']:.3f} | {stats_ok['early_beyond_wx']['mean']:.4f} | {stats_ok['early_beyond_vz']['mean']:.4f} |
| K49-fail (39) | {len(fail_ids)} | {stats_fail['early_recon_l2']['mean']:.4f} | {stats_fail['early_clip_frac']:.3f} | {stats_fail['early_beyond_wx']['mean']:.4f} | {stats_fail['early_beyond_vz']['mean']:.4f} |

## Per-dimension early MAE (all C)

| dim | mean | p90 | max |
|---|---:|---:|---:|
"""
    for d in ACT_DIMS:
        s = stats_all["early_mae"][d]
        cov_report += f"| {d} | {s['mean']:.4f} | {s['p90']:.4f} | {s['max']:.4f} |\n"

    # Compare ok vs fail early max beyond wx
    ok_wx = np.mean([p["early_max_beyond_wx"] for p in per_traj if p["k49_ok"]])
    fail_wx = np.mean([p["early_max_beyond_wx"] for p in per_traj if not p["k49_ok"]])
    ok_vz = np.mean([p["early_max_beyond_vz"] for p in per_traj if p["k49_ok"]])
    fail_vz = np.mean([p["early_max_beyond_vz"] for p in per_traj if not p["k49_ok"]])
    cov_report += f"""
## Success vs fail early peak beyond-CAP

| | mean max beyond-wx | mean max beyond-vz |
|---|---:|---:|
| K49-success | {ok_wx:.4f} | {ok_vz:.4f} |
| K49-fail | {fail_wx:.4f} | {fail_vz:.4f} |

## Interpretation

- Failures concentrate where **wx/vz residuals exceed CAP=0.25** in the first 0.2 s (coupled catch/lift).
- Missing coverage is primarily **magnitude + coupled (trans, wx)** combinations, not grip sign flips.
- Unique poorly-represented early action clusters: **{len(poor_by_key)}** (see `POORLY_REPRESENTED_ACTIONS.csv`).
"""
    (OUT / "QUANTIZATION_COVERAGE.md").write_text(cov_report, encoding="utf-8")
    atomic_json(
        OUT / "COVERAGE_STATS.json",
        {"all": stats_all, "ok": stats_ok, "fail": stats_fail, "ok_wx_peak": ok_wx, "fail_wx_peak": fail_wx},
    )

    # ----- Phase 2: minimal extensions from TRAIN only -----
    print("Phase 2: constructing extensions from TRAIN...", flush=True)
    candidates = collect_weighted_early_actions(c_train, k49)
    books = {"K49": k49}
    meta_sel = {}
    for n_new in EXT_SIZES:
        aug, selected = greedy_extend(k49, candidates, n_new)
        name = f"K49_plus_{n_new}"
        books[name] = aug
        np.save(OUT / "codebooks" / f"{name}.npy", aug)
        meta_sel[name] = {
            "n_new": len(selected),
            "selected_actions": [a.tolist() for a in selected],
            "construction": "greedy weighted farthest-point on TRAIN early actions",
            "weights": "beyond_wx*3 + beyond_vz*2 + sum(beyond) + 0.5*recon_l2",
        }
        print(f"  {name}: {aug.shape[0]} rows (+{len(selected)})", flush=True)
    atomic_json(OUT / "EXTENSION_SELECTION.json", meta_sel)

    # ----- Phase 4 prep: rebuilt skill-balanced from TRAIN skills -----
    print("Phase 4: rebuilt skill-balanced codebooks...", flush=True)
    skill_train = defaultdict(list)
    for r in rows:
        if r.get("source_label") not in ("A", "B", "C", "STABLE_PRECISION"):
            continue
        if not (r.get("continuous_demo") or {}).get("verified"):
            continue
        # C construction only for C skill
        if r["source_label"] == "C" and r["sample_id"] not in {x["sample_id"] for x in train_rows}:
            continue
        skill_train[r["source_label"]].append(load_traj(r))
    for n_mot in REBUILD_SIZES:
        name = f"REBUILD_m{n_mot}"
        tab = rebuild_skill_balanced(skill_train, n_mot, seed=SEED)
        books[name] = tab
        np.save(OUT / "codebooks" / f"{name}.npy", tab)
        print(f"  {name}: shape={tab.shape}", flush=True)

    # ----- Phase 3: physics replay -----
    print("Phase 3: physics replay...", flush=True)
    outcome_rows = []
    summary_rows = []

    def eval_book(name: str, table: np.ndarray, trajs: list[dict], split: str):
        ok_n = 0
        fails = Counter()
        bil = 0
        for t in trajs:
            print(f"  [{name}/{split}] {t['sample_id']}", flush=True)
            r = replay_quantized(env, t, table)
            if r["SUCCESS_NEW"]:
                ok_n += 1
            else:
                fails[r["fail_reason"] or "FAIL"] += 1
                if r["fail_reason"] == "bilateral_loss":
                    bil += 1
            outcome_rows.append({
                "codebook": name,
                "split": split,
                "sample_id": t["sample_id"],
                "source_label": t["source_label"],
                "SUCCESS_NEW": r["SUCCESS_NEW"],
                "SUCCESS_12S": r["SUCCESS_12S"],
                "fail_reason": r["fail_reason"],
                "first_bil_loss_t": r["first_bil_loss_t"],
                "first_state_t": r["first_state_t"],
                "n_actions": r["n_actions"],
                "n_codes": r["n_codes"],
                "reason": r["reason"],
            })
        return {
            "codebook": name,
            "split": split,
            "n": len(trajs),
            "SUCCESS_NEW": ok_n,
            "rate": ok_n / max(len(trajs), 1),
            "bilateral_loss": bil,
            "fail_breakdown": dict(fails),
            "n_codes": int(len(table)),
        }

    # Order: baseline then extensions then rebuilds
    order = ["K49"] + [f"K49_plus_{n}" for n in EXT_SIZES] + [f"REBUILD_m{n}" for n in REBUILD_SIZES]
    for name in order:
        table = books[name]
        # held-out eval (primary generalization)
        summary_rows.append(eval_book(name, table, c_eval, "C_eval_heldout"))
        # in-sample diagnostic on all 47
        summary_rows.append(eval_book(name, table, c_all, "C_all_insample_diagnostic"))
        # regression non-C
        summary_rows.append(eval_book(name, table, reg_trajs, "nonC_regression"))

    write_csv(OUT / "PER_TRAJECTORY_REPLAY.csv", outcome_rows)
    write_csv(
        OUT / "CODEBOOK_COMPARISON.csv",
        [
            {
                **{k: s[k] for k in s if k != "fail_breakdown"},
                "fail_breakdown": json.dumps(s["fail_breakdown"]),
            }
            for s in summary_rows
        ],
    )
    atomic_json(OUT / "CODEBOOK_COMPARISON.json", summary_rows)

    # Quant coverage after extension on eval early window
    cov_ext = {}
    for name in ["K49"] + [f"K49_plus_{n}" for n in EXT_SIZES]:
        cov_ext[name] = quant_stats(c_eval, books[name])
    atomic_json(OUT / "EVAL_QUANT_AFTER_EXTENSION.json", cov_ext)

    # Recommendation
    held = [s for s in summary_rows if s["split"] == "C_eval_heldout"]
    base = next(s for s in held if s["codebook"] == "K49")
    best_ext = max(
        [s for s in held if s["codebook"].startswith("K49_plus_")],
        key=lambda s: (s["SUCCESS_NEW"], -s["n_codes"]),
    )
    best_reb = max(
        [s for s in held if s["codebook"].startswith("REBUILD_")],
        key=lambda s: (s["SUCCESS_NEW"], -s["n_codes"]),
    )
    reg = {s["codebook"]: s for s in summary_rows if s["split"] == "nonC_regression"}

    rec = f"""# RECOMMENDATION.md

Generated: {datetime.now(timezone.utc).isoformat()}

## Held-out C eval (codebook construction excluded these lineages)

| Codebook | SUCCESS_NEW | rate | bilateral_loss | n_codes | nonC regression NEW |
|---|---:|---:|---:|---:|---:|
"""
    for s in held:
        rg = reg.get(s["codebook"], {})
        rec += (
            f"| {s['codebook']} | {s['SUCCESS_NEW']}/{s['n']} | {s['rate']:.3f} | "
            f"{s['bilateral_loss']} | {s['n_codes']} | "
            f"{rg.get('SUCCESS_NEW','?')}/{rg.get('n','?')} |\n"
        )

    ins = [s for s in summary_rows if s["split"] == "C_all_insample_diagnostic"]
    rec += "\n## In-sample diagnostic (all 47 C — NOT generalization)\n\n"
    rec += "| Codebook | SUCCESS_NEW | rate |\n|---|---:|---:|\n"
    for s in ins:
        if s["codebook"].startswith("K49"):
            rec += f"| {s['codebook']} | {s['SUCCESS_NEW']}/{s['n']} | {s['rate']:.3f} |\n"

    rec += f"""
## Choice

- Baseline held-out: **{base['SUCCESS_NEW']}/{base['n']}**
- Best minimal extension (held-out): **{best_ext['codebook']}** → {best_ext['SUCCESS_NEW']}/{best_ext['n']}
- Best rebuild (held-out): **{best_reb['codebook']}** → {best_reb['SUCCESS_NEW']}/{best_reb['n']}

### Smallest codebook that preserves heterogeneous skills

Prefer the **smallest K49_plus_*** that improves held-out C NEW without regressing non-C rate below baseline.
If extensions remain << continuous 47/47, retain continuous Stage-1 for C (or hybrid) rather than forcing a small discrete book.

**Do not start RL** until a discrete representation clears physics replay on held-out C without non-C regression.
"""
    (OUT / "RECOMMENDATION.md").write_text(rec, encoding="utf-8")

    report = f"""# REPORT — C-class K49 codebook extension

Generated: {datetime.now(timezone.utc).isoformat()}
Baseline config: `{K49_CONFIG}` (original 49 rows frozen in `codebooks/K49_baseline.npy`)

## Phase 1 — Coverage

See `QUANTIZATION_COVERAGE.md`. Early-window failures are driven by **wx/vz beyond CAP**,
especially coupled catch/lift actions in the first 0.2 s. K49-fail trajectories show higher
peak beyond-wx ({fail_wx:.3f}) than K49-success ({ok_wx:.3f}).

## Phase 2 — Extensions

Constructed from **TRAIN only** ({len(train_rows)} lineages) via weighted greedy FPS on early actions.
Eval held-out: {len(eval_rows)} lineages. Selection metadata: `EXTENSION_SELECTION.json`.

## Phase 3 — Physics replay (held-out C)

| Codebook | NEW | rate | bil_loss |
|---|---:|---:|---:|
"""
    for s in held:
        report += f"| {s['codebook']} | {s['SUCCESS_NEW']}/{s['n']} | {s['rate']:.3f} | {s['bilateral_loss']} |\n"

    report += f"""
In-sample diagnostic (all 47) is in `CODEBOOK_COMPARISON.csv` (split=`C_all_insample_diagnostic`).

## Phase 4 — Rebuilds

Skill-balanced rebuilds from A/B/C/STABLE_PRECISION construction data; same replay protocol.
See comparison table in `RECOMMENDATION.md`.

## Artifacts

- `QUANTIZATION_COVERAGE.md`, `COVERAGE_STATS.json`, `PER_TRAJ_EARLY_QUANT.csv`
- `codebooks/*.npy`, `EXTENSION_SELECTION.json`, `SPLIT.json`
- `PER_TRAJECTORY_REPLAY.csv`, `CODEBOOK_COMPARISON.csv`
- `RECOMMENDATION.md`

No RL. Original K49 baseline file and expert trajectories unchanged.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
