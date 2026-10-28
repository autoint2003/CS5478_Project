"""Phase 2–3 — R0/R1/R2/R3 repairs and vocabulary ablation."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.precision.p18_discrete_vocab_audit.replay_harness import (
    make_env,
    quantize_with_book,
    replay_actions,
)
from experiments.precision.p34_5_m111_precision_repair.io_util import (
    atomic_json,
    load_csv,
    utc,
    write_csv,
    write_md,
)
from experiments.precision.p34_5_m111_precision_repair.paths import (
    HANDOFF_M111,
    M111_META,
    M111_REPLAY,
    MOTION_K,
    OUT,
    ROOT,
)
from experiments.precision.p34_precision_data_expansion.builder import (
    restore_default_dynamics,
    snapshot_default_dynamics,
)
from training.skill_aware_action_vocab import (
    ACT_DIMS,
    DEFAULT_WEIGHTS,
    ResidualBudget,
    hybrid_decode,
    load_motion_codebook,
    nearest_proto,
    quantize_trajectory,
)


def _rows() -> tuple[list[dict], list[dict]]:
    rows = load_csv(M111_REPLAY)
    fails = [r for r in rows if str(r.get("ok")).lower() not in ("1", "true")]
    oks = [r for r in rows if str(r.get("ok")).lower() in ("1", "true")]
    return fails, oks


def _abs(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def load_book(meta: Path | None = None):
    cb = load_motion_codebook(meta or M111_META)
    return np.asarray(cb.prototypes, np.float32), cb.residual, cb


# ---------------------------------------------------------------------------
# Quantizers / reconstructors
# ---------------------------------------------------------------------------
def recon_r0(book, rho, acts: np.ndarray) -> np.ndarray:
    return quantize_trajectory(book, acts, rho, w=DEFAULT_WEIGHTS)["recon_actions"]


def recon_r1_two_stage(book, rho, acts: np.ndarray) -> np.ndarray:
    """R1: motion nearest on dims 0–5 (grip weight 0), then residual incl. grip.

    Does not change prototypes or ρ — only nearest-neighbor weights for token choice.
    """
    w = DEFAULT_WEIGHTS.copy()
    w[6] = 0.0
    return quantize_trajectory(book, acts, rho, w=w)["recon_actions"]


def recon_r1_grip_aware(book, rho, acts: np.ndarray) -> np.ndarray:
    """R1b: raise grip weight for prototype selection (still same book/ρ)."""
    w = DEFAULT_WEIGHTS.copy()
    w[6] = 2.0
    return quantize_trajectory(book, acts, rho, w=w)["recon_actions"]


def recon_r2_temporal(book, rho, acts: np.ndarray, *, max_extra: int = 1) -> tuple[np.ndarray, dict]:
    """R2: for steps where residual saturates, try 2-step existing-token approx.

    Greedy: if single-token recon L∞ error on any dim > 0.05, search pairs (t1,t2)
    whose mean action best matches the continuous command (same decision rate ×2 steps).
    Reports extra steps.
    """
    acts = np.asarray(acts, np.float32).reshape(-1, 7)
    book = np.asarray(book, np.float32).reshape(-1, 7)
    rho_a = ResidualBudget.from_spec(rho).as_array()
    out: list[np.ndarray] = []
    extra = 0
    n_pair = 0
    # Precompute a small candidate set: all prototypes (111) is fine for pairs~12k
    for a in acts:
        ti, proto, _ = nearest_proto(book, a, w=DEFAULT_WEIGHTS)
        raw = a - proto
        res = np.clip(raw, -rho_a, rho_a)
        recon = np.clip(proto + res, -1, 1).astype(np.float32)
        if float(np.max(np.abs(recon - a))) <= 0.05 or max_extra <= 0:
            out.append(recon)
            continue
        # search best pair mean
        best = recon
        best_err = float(np.max(np.abs(recon - a)))
        # Restrict search to top-20 nearest for speed
        w = DEFAULT_WEIGHTS
        d = np.max(w * np.abs(book - a), axis=1)
        top = np.argsort(d)[:20]
        for i in top:
            for j in top:
                mean = 0.5 * (book[i] + book[j])
                # allow residual on mean toward a
                raw2 = a - mean
                res2 = np.clip(raw2, -rho_a, rho_a)
                cand = np.clip(mean + res2, -1, 1).astype(np.float32)
                err = float(np.max(np.abs(cand - a)))
                if err < best_err:
                    best_err = err
                    best = cand
                    # emit two half-steps as book[i]+res_i and book[j]+res_j approximating mean
                    # Practical: emit two decoded steps that average to best
                    r_i = np.clip(a - book[i], -rho_a, rho_a)
                    r_j = np.clip(a - book[j], -rho_a, rho_a)
                    a_i = np.clip(book[i] + r_i, -1, 1).astype(np.float32)
                    a_j = np.clip(book[j] + r_j, -1, 1).astype(np.float32)
                    if float(np.max(np.abs(0.5 * (a_i + a_j) - a))) <= best_err + 1e-6:
                        best = (a_i, a_j)
        if isinstance(best, tuple):
            out.extend(best)
            extra += 1
            n_pair += 1
        else:
            out.append(best)
    return np.asarray(out, np.float32), {"extra_steps": extra, "n_pair_replaced": n_pair, "out_len": len(out)}


def build_extended_book(
    book: np.ndarray,
    rho,
    fail_paths: list[Path],
    *,
    n_add: int,
    seed: int = 5478,
) -> tuple[np.ndarray, list[dict]]:
    """R3/V1/V2: cluster saturated continuous actions; append centroids as new prototypes."""
    book = np.asarray(book, np.float32).reshape(-1, 7)
    rho_a = ResidualBudget.from_spec(rho).as_array()
    residuals = []
    for p in fail_paths:
        acts = np.asarray(pickle.loads(p.read_bytes())["actions"], np.float32).reshape(-1, 7)
        q = quantize_trajectory(book, acts, rho, w=DEFAULT_WEIGHTS)
        recon = q["recon_actions"]
        err = np.abs(acts - recon)
        # keep steps with large error
        mask = np.max(err, axis=1) > 0.05
        if not np.any(mask):
            continue
        residuals.append(acts[mask])
    if not residuals:
        return book.copy(), []
    pool = np.concatenate(residuals, axis=0)
    # greedy farthest-point / simple k-medoids-ish
    rng = np.random.default_rng(seed)
    # start with worst grip extremes and motion outliers
    scores = np.max(np.abs(pool - 0), axis=1)  # magnitude
    order = np.argsort(-scores)
    chosen: list[np.ndarray] = []
    meta: list[dict] = []
    for idx in order:
        a = pool[idx].astype(np.float32)
        # skip if already covered by book+rho or chosen
        ok = False
        for b in list(book) + chosen:
            raw = a - b
            recon = np.clip(b + np.clip(raw, -rho_a, rho_a), -1, 1)
            if float(np.max(np.abs(recon - a))) <= 0.03:
                ok = True
                break
        if ok:
            continue
        chosen.append(a.copy())
        meta.append(
            {
                "new_id": int(len(book) + len(chosen) - 1),
                "proto": [float(x) for x in a],
                "source": "fail_pool_farthest",
            }
        )
        if len(chosen) >= n_add:
            break
    if not chosen:
        return book.copy(), []
    ext = np.vstack([book, np.asarray(chosen, np.float32)])
    return ext, meta


def recon_with_book(book, rho, acts, w=None):
    return quantize_trajectory(book, acts, rho, w=w if w is not None else DEFAULT_WEIGHTS)[
        "recon_actions"
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_recon_fn(
    env,
    paths: list[Path],
    recon_fn: Callable[[np.ndarray], np.ndarray],
    *,
    label: str,
) -> list[dict]:
    rows = []
    for p in paths:
        blob = pickle.loads(p.read_bytes())
        acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
        extra_meta = {}
        out = recon_fn(acts)
        if isinstance(out, tuple):
            recon, extra_meta = out
        else:
            recon = out
        h = replay_actions(env, blob["capture"], blob["ref"], recon)
        # approx error
        # align lengths if R2 extended
        if len(recon) == len(acts):
            max_err = float(np.max(np.abs(recon - acts)))
            mean_err = float(np.mean(np.abs(recon - acts)))
        else:
            # compare mean of pairs roughly: downsample
            max_err = float("nan")
            mean_err = float("nan")
        rows.append(
            {
                "label": label,
                "tag": blob.get("tag") or p.stem,
                "lineage_id": blob.get("lineage_id"),
                "ic_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "SUCCESS_NEW": int(bool(h.get("SUCCESS_NEW"))),
                "SUCCESS_12S": int(bool(h.get("SUCCESS_12S"))),
                "reason": h.get("reason"),
                "pre_pos_mm": h.get("pre_pos_mm"),
                "pre_theta": h.get("pre_theta"),
                "n_actions_in": int(len(acts)),
                "n_actions_out": int(len(recon)),
                "max_abs_recon_err": max_err,
                "mean_abs_recon_err": mean_err,
                **{f"extra_{k}": v for k, v in extra_meta.items()},
            }
        )
        print(
            f"  [{label}] {rows[-1]['tag'][:48]} NEW={rows[-1]['SUCCESS_NEW']} "
            f"12={rows[-1]['SUCCESS_12S']}",
            flush=True,
        )
    return rows


def run_repairs(
    *,
    diag_limit: int | None = None,
    skip_r2: bool = False,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fails, oks = _rows()
    if diag_limit is not None:
        fails = fails[:diag_limit]
        oks = oks[: max(10, diag_limit)]
    fail_paths = [_abs(r["ic_path"]) for r in fails]
    ok_paths = [_abs(r["ic_path"]) for r in oks]

    book, rho, cb = load_book()
    env = make_env()
    defaults = snapshot_default_dynamics(env.sim)

    ablation: list[dict] = []
    all_detail: list[dict] = []

    def summarize(label: str, vocab_size: int, n_add: int, fail_rows, ok_rows):
        n_fail_ok = sum(1 for r in fail_rows if r["SUCCESS_NEW"])
        n_ok_ok = sum(1 for r in ok_rows if r["SUCCESS_NEW"])
        n_ok_reg = sum(1 for r in ok_rows if not r["SUCCESS_NEW"])
        row = {
            "candidate": label,
            "vocab_motion": vocab_size,
            "handoff_id": vocab_size,
            "n_added": n_add,
            "fail_cohort_n": len(fail_rows),
            "fail_recovered": n_fail_ok,
            "fail_recovery_rate": n_fail_ok / max(1, len(fail_rows)),
            "ok_cohort_n": len(ok_rows),
            "ok_preserved": n_ok_ok,
            "ok_regressed": n_ok_reg,
            "mean_max_abs_err_fail": float(
                np.nanmean([r["max_abs_recon_err"] for r in fail_rows])
            ),
        }
        ablation.append(row)
        print(
            f"== {label}: recovered {n_fail_ok}/{len(fail_rows)}; "
            f"preserved {n_ok_ok}/{len(ok_rows)} (regressed {n_ok_reg}) ==",
            flush=True,
        )
        return row

    # R0
    print("=== R0 original M111 ===", flush=True)
    r0_f = eval_recon_fn(env, fail_paths, lambda a: recon_r0(book, rho, a), label="R0")
    r0_o = eval_recon_fn(env, ok_paths, lambda a: recon_r0(book, rho, a), label="R0_ok")
    summarize("V0_R0_M111", MOTION_K, 0, r0_f, r0_o)
    all_detail.extend(r0_f + r0_o)

    # R1 two-stage
    print("=== R1 two-stage (w_grip=0 for NN) ===", flush=True)
    r1_f = eval_recon_fn(env, fail_paths, lambda a: recon_r1_two_stage(book, rho, a), label="R1")
    r1_o = eval_recon_fn(env, ok_paths, lambda a: recon_r1_two_stage(book, rho, a), label="R1_ok")
    summarize("R1_two_stage", MOTION_K, 0, r1_f, r1_o)
    all_detail.extend(r1_f + r1_o)

    # R1b grip-aware weights
    print("=== R1b grip-aware weights ===", flush=True)
    r1b_f = eval_recon_fn(env, fail_paths, lambda a: recon_r1_grip_aware(book, rho, a), label="R1b")
    r1b_o = eval_recon_fn(env, ok_paths, lambda a: recon_r1_grip_aware(book, rho, a), label="R1b_ok")
    summarize("R1b_grip_weight", MOTION_K, 0, r1b_f, r1b_o)
    all_detail.extend(r1b_f + r1b_o)

    # R2 temporal (expensive on long tapes — optional)
    if not skip_r2:
        print("=== R2 temporal pair approx ===", flush=True)

        def r2fn(a):
            recon, meta = recon_r2_temporal(book, rho, a, max_extra=1)
            return recon, meta

        r2_f = eval_recon_fn(env, fail_paths, r2fn, label="R2")
        # ok cohort: only spot-check first 15 for cost
        r2_o = eval_recon_fn(env, ok_paths[:15], r2fn, label="R2_ok")
        summarize("R2_temporal", MOTION_K, 0, r2_f, r2_o)
        all_detail.extend(r2_f + r2_o)

    # R3 / V1: +8 prototypes
    print("=== V1 R3 +8 prototypes ===", flush=True)
    book_v1, meta_v1 = build_extended_book(book, rho, fail_paths, n_add=8)
    atomic_json(
        OUT / "vocab_v1_meta.json",
        {
            "M": int(book_v1.shape[0]),
            "handoff_id": int(book_v1.shape[0]),
            "added": meta_v1,
            "old_handoff": HANDOFF_M111,
            "id_map_note": "IDs 0..110 unchanged; new motion IDs 111..M-1; HANDOFF=M",
        },
    )
    np.save(OUT / "vocab_v1_prototypes.npy", book_v1)
    v1_f = eval_recon_fn(
        env, fail_paths, lambda a: recon_with_book(book_v1, rho, a), label="V1"
    )
    v1_o = eval_recon_fn(env, ok_paths, lambda a: recon_with_book(book_v1, rho, a), label="V1_ok")
    summarize("V1_M111_plus_8", int(book_v1.shape[0]), len(meta_v1), v1_f, v1_o)
    all_detail.extend(v1_f + v1_o)

    # V2: +16 if V1 insufficient
    need_v2 = sum(1 for r in v1_f if r["SUCCESS_NEW"]) < int(0.7 * len(v1_f))
    book_v2, meta_v2 = None, []
    if need_v2 or True:  # always compute V2 for ablation table
        print("=== V2 R3 +16 prototypes ===", flush=True)
        book_v2, meta_v2 = build_extended_book(book, rho, fail_paths, n_add=16)
        atomic_json(
            OUT / "vocab_v2_meta.json",
            {
                "M": int(book_v2.shape[0]),
                "handoff_id": int(book_v2.shape[0]),
                "added": meta_v2,
                "old_handoff": HANDOFF_M111,
            },
        )
        np.save(OUT / "vocab_v2_prototypes.npy", book_v2)
        v2_f = eval_recon_fn(
            env, fail_paths, lambda a: recon_with_book(book_v2, rho, a), label="V2"
        )
        v2_o = eval_recon_fn(
            env, ok_paths, lambda a: recon_with_book(book_v2, rho, a), label="V2_ok"
        )
        summarize("V2_M111_plus_16", int(book_v2.shape[0]), len(meta_v2), v2_f, v2_o)
        all_detail.extend(v2_f + v2_o)

    restore_default_dynamics(env.sim, defaults)
    write_csv(OUT / "VOCABULARY_ABLATION.csv", ablation)
    write_csv(OUT / "REPAIR_DETAIL.csv", all_detail)

    # Candidates doc
    best = max(ablation, key=lambda r: (r["fail_recovered"], -r["ok_regressed"], -r["n_added"]))
    # Prefer zero regression
    safe = [r for r in ablation if r["ok_regressed"] == 0]
    chosen = max(safe, key=lambda r: (r["fail_recovered"], -r["n_added"])) if safe else best

    cand_md = f"""# VOCABULARY_CANDIDATES — P34.5

Generated: {utc()}

## Design rules

- Preserve motion IDs **0..110** and their decoded controls exactly.
- HANDOFF remains a distinct non-motion token; under extension `HANDOFF = M_new`.
- Additions are continuous expert actions from the 38-fail development pool that are
  not approximable within ρ=0.35 of any existing prototype.

## Candidates

| ID | Description | Motion M | Added | Notes |
|---|---|---:|---:|---|
| V0 | Original M111 | 111 | 0 | R0 baseline |
| R1 | Two-stage NN (w_grip=0) | 111 | 0 | Quantization policy only |
| R1b | Grip weight ↑ for NN | 111 | 0 | Quantization policy only |
| R2 | Temporal 2-step approx | 111 | 0 | May lengthen sequences |
| V1 | M111 + 8 fail-derived protos | {111 + len(meta_v1)} | {len(meta_v1)} | See vocab_v1_meta.json |
| V2 | M111 + 16 fail-derived protos | {111 + len(meta_v2)} | {len(meta_v2)} | See vocab_v2_meta.json |

## Provisional selection (dev set)

**{chosen['candidate']}** — recovered {chosen['fail_recovered']}/{chosen['fail_cohort_n']},
regressed {chosen['ok_regressed']}/{chosen['ok_cohort_n']}.

Freeze before independent validation. Do not claim generalization from the 38 alone.
"""
    write_md(OUT / "VOCABULARY_CANDIDATES.md", cand_md)
    atomic_json(OUT / "chosen_vocab.json", chosen)
    return {
        "ablation": ablation,
        "chosen": chosen,
        "book_v1": book_v1,
        "meta_v1": meta_v1,
        "book_v2": book_v2,
        "meta_v2": meta_v2,
        "book": book,
        "rho": rho,
    }
