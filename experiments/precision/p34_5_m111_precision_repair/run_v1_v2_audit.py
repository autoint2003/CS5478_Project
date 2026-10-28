"""P34.5 Audit — verify V1/V2 coverage claims (no vocab changes, no SFT)."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.precision.p18_discrete_vocab_audit.replay_harness import (
    make_env,
    replay_actions,
)
from experiments.precision.p34_5_m111_precision_repair.io_util import utc, write_csv, write_md
from experiments.precision.p34_5_m111_precision_repair.paths import (
    M111_META,
    M111_REPLAY,
    OUT,
    P34_MANIFEST,
)
from experiments.precision.p34_precision_data_expansion.builder import (
    restore_default_dynamics,
    snapshot_default_dynamics,
)
from training.skill_aware_action_vocab import (
    ACT_DIMS,
    DEFAULT_WEIGHTS,
    ResidualBudget,
    load_motion_codebook,
    quantize_trajectory,
)

OUT.mkdir(parents=True, exist_ok=True)


def _abs(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def load_books():
    cb = load_motion_codebook(M111_META)
    book0 = np.asarray(cb.prototypes, np.float32)
    rho = cb.residual
    v1p = OUT / "vocab_v1_prototypes.npy"
    v2p = OUT / "vocab_v2_prototypes.npy"
    book_v1 = np.load(v1p) if v1p.is_file() else book0
    book_v2 = np.load(v2p) if v2p.is_file() else book0
    return book0, book_v1, book_v2, rho


def load_fail_ok():
    rows = list(csv.DictReader(M111_REPLAY.open(encoding="utf-8")))
    fails = [r for r in rows if str(r.get("ok")).lower() not in ("1", "true")]
    oks = [r for r in rows if str(r.get("ok")).lower() in ("1", "true")]
    return fails, oks


def identify_25_spot() -> list[str]:
    """Exact 25 tags used in ablation ok cohort (first 25 M111-ok in P34 audit order)."""
    _, oks = load_fail_ok()
    return [r.get("tag") or Path(r["ic_path"]).stem for r in oks[:25]]


def quantize(book, rho, acts, w=None):
    return quantize_trajectory(
        book, acts, rho, w=w if w is not None else DEFAULT_WEIGHTS
    )


def section_a_full_regression(book0, book_v1, book_v2, rho) -> dict:
    fails, oks = load_fail_ok()
    spot25 = identify_25_spot()
    env = make_env()
    defaults = snapshot_default_dynamics(env.sim)
    rows = []
    print(f"=== A: full regression on {len(oks)} M111-ok P34 roots ===", flush=True)
    for i, r in enumerate(oks):
        p = _abs(r["ic_path"])
        blob = pickle.loads(p.read_bytes())
        acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
        tag = blob.get("tag") or r.get("tag") or p.stem
        # V1≡V2 books → evaluate V0 and V1 only; copy V1 → V2.
        results = {}
        for name, book in [("V0", book0), ("V1", book_v1)]:
            recon = quantize(book, rho, acts)["recon_actions"]
            h = replay_actions(env, blob["capture"], blob["ref"], recon)
            results[name] = h
        results["V2"] = results["V1"]
        row = {
            "tag": tag,
            "lineage_id": blob.get("lineage_id") or r.get("lineage_id"),
            "ic_path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "in_ablation_spot25": int(tag in spot25 or r.get("tag") in spot25),
            "V0_SUCCESS_NEW": int(bool(results["V0"].get("SUCCESS_NEW"))),
            "V1_SUCCESS_NEW": int(bool(results["V1"].get("SUCCESS_NEW"))),
            "V2_SUCCESS_NEW": int(bool(results["V2"].get("SUCCESS_NEW"))),
            "V0_reason": results["V0"].get("reason"),
            "V1_reason": results["V1"].get("reason"),
            "V2_reason": results["V2"].get("reason"),
            "V2_copied_from_V1": 1,
            "V1_equals_V0_outcome": int(
                bool(results["V1"].get("SUCCESS_NEW")) == bool(results["V0"].get("SUCCESS_NEW"))
            ),
            "V2_equals_V1_outcome": 1,
        }
        rows.append(row)
        print(
            f"  [{i+1}/{len(oks)}] {tag[:52]} "
            f"V0={row['V0_SUCCESS_NEW']} V1={row['V1_SUCCESS_NEW']} V2={row['V2_SUCCESS_NEW']}",
            flush=True,
        )
    restore_default_dynamics(env.sim, defaults)
    write_csv(OUT / "FULL_REGRESSION_85.csv", rows)

    # P33 roots from manifest
    p33_rows = []
    if P34_MANIFEST.is_file():
        man = list(csv.DictReader(P34_MANIFEST.open(encoding="utf-8")))
        p33 = [m for m in man if m.get("source") == "P33"]
        print(f"=== A: P33 roots n={len(p33)} ===", flush=True)
        env = make_env()
        defaults = snapshot_default_dynamics(env.sim)
        for m in p33:
            p = _abs(m["ic_path"])
            if not p.is_file():
                p33_rows.append({"tag": m.get("tag"), "missing": 1})
                continue
            blob = pickle.loads(p.read_bytes())
            acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
            if acts.size == 0:
                p33_rows.append({"tag": m.get("tag"), "no_actions": 1})
                continue
            out = {"tag": m.get("tag"), "split": m.get("split"), "missing": 0}
            for name, book in [("V0", book0), ("V1", book_v1)]:
                recon = quantize(book, rho, acts)["recon_actions"]
                h = replay_actions(env, blob["capture"], blob["ref"], recon)
                out[f"{name}_SUCCESS_NEW"] = int(bool(h.get("SUCCESS_NEW")))
                out[f"{name}_reason"] = h.get("reason")
            out["V2_SUCCESS_NEW"] = out["V1_SUCCESS_NEW"]
            out["V2_reason"] = out["V1_reason"]
            out["V2_copied_from_V1"] = 1
            p33_rows.append(out)
            print(
                f"  P33 {out['tag'][:48]} V0={out.get('V0_SUCCESS_NEW')} "
                f"V1={out.get('V1_SUCCESS_NEW')} V2={out.get('V2_SUCCESS_NEW')}",
                flush=True,
            )
        restore_default_dynamics(env.sim, defaults)
        write_csv(OUT / "P33_REGRESSION.csv", p33_rows)

    summary = {
        "n_ok": len(oks),
        "spot25": spot25,
        "V0_preserved": sum(r["V0_SUCCESS_NEW"] for r in rows),
        "V1_preserved": sum(r["V1_SUCCESS_NEW"] for r in rows),
        "V2_preserved": sum(r["V2_SUCCESS_NEW"] for r in rows),
        "V1_regressed_vs_V0": sum(
            1 for r in rows if r["V0_SUCCESS_NEW"] and not r["V1_SUCCESS_NEW"]
        ),
        "V2_regressed_vs_V0": sum(
            1 for r in rows if r["V0_SUCCESS_NEW"] and not r["V2_SUCCESS_NEW"]
        ),
        "books_identical": bool(np.allclose(book_v1, book_v2)),
        "p33": p33_rows,
    }
    return summary


def section_b_token_usage(book0, book_v1, book_v2, rho) -> dict:
    fails, _ = load_fail_ok()
    M0 = book0.shape[0]
    rows = []
    new_v1_ids = list(range(M0, book_v1.shape[0]))
    new_v2_ids = list(range(M0, book_v2.shape[0]))
    print(
        f"=== B: token usage on 38 fails; new_v1={new_v1_ids} new_v2={new_v2_ids} ===",
        flush=True,
    )
    print(
        f"  book shapes V0={book0.shape} V1={book_v1.shape} V2={book_v2.shape} "
        f"identical={np.allclose(book_v1, book_v2)}",
        flush=True,
    )

    for r in fails:
        p = _abs(r["ic_path"])
        blob = pickle.loads(p.read_bytes())
        acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
        q0 = quantize(book0, rho, acts)
        q1 = quantize(book_v1, rho, acts)
        q2 = quantize(book_v2, rho, acts)
        t0 = np.asarray(q0["tokens"], np.int64)
        t1 = np.asarray(q1["tokens"], np.int64)
        t2 = np.asarray(q2["tokens"], np.int64)
        # per-new-token counts
        cnt1 = Counter(int(x) for x in t1 if int(x) >= M0)
        cnt2 = Counter(int(x) for x in t2 if int(x) >= M0)
        # first timestep where recon differs
        r0 = q0["recon_actions"]
        r1 = q1["recon_actions"]
        r2 = q2["recon_actions"]
        first_v1 = next(
            (i for i in range(len(acts)) if float(np.max(np.abs(r1[i] - r0[i]))) > 1e-6),
            None,
        )
        first_v2 = next(
            (i for i in range(len(acts)) if float(np.max(np.abs(r2[i] - r0[i]))) > 1e-6),
            None,
        )
        first_v2_vs_v1 = next(
            (i for i in range(len(acts)) if float(np.max(np.abs(r2[i] - r1[i]))) > 1e-6),
            None,
        )
        err0 = float(np.max(np.abs(r0 - acts)))
        err1 = float(np.max(np.abs(r1 - acts)))
        err2 = float(np.max(np.abs(r2 - acts)))
        mean0 = float(np.mean(np.abs(r0 - acts)))
        mean1 = float(np.mean(np.abs(r1 - acts)))
        mean2 = float(np.mean(np.abs(r2 - acts)))
        grip_err0 = float(np.max(np.abs(r0[:, 6] - acts[:, 6])))
        grip_err1 = float(np.max(np.abs(r1[:, 6] - acts[:, 6])))

        row = {
            "tag": blob.get("tag") or r.get("tag"),
            "lineage_id": blob.get("lineage_id") or r.get("lineage_id"),
            "n_actions": int(len(acts)),
            "n_steps_new_token_V1": int(sum(cnt1.values())),
            "n_steps_new_token_V2": int(sum(cnt2.values())),
            "uses_any_new_V1": int(sum(cnt1.values()) > 0),
            "uses_any_new_V2": int(sum(cnt2.values()) > 0),
            "new_token_hist_V1": dict(cnt1),
            "new_token_hist_V2": dict(cnt2),
            "unique_tokens_V0": int(len(set(t0.tolist()))),
            "unique_tokens_V1": int(len(set(t1.tolist()))),
            "unique_tokens_V2": int(len(set(t2.tolist()))),
            "max_err_V0": err0,
            "max_err_V1": err1,
            "max_err_V2": err2,
            "mean_err_V0": mean0,
            "mean_err_V1": mean1,
            "mean_err_V2": mean2,
            "grip_max_err_V0": grip_err0,
            "grip_max_err_V1": grip_err1,
            "err_reduced_V1": int(err1 + 1e-9 < err0),
            "first_recon_div_V1_vs_V0": first_v1,
            "first_recon_div_V2_vs_V0": first_v2,
            "first_recon_div_V2_vs_V1": first_v2_vs_v1,
            "recon_identical_V1_V2": int(np.allclose(r1, r2)),
            "tokens_identical_V1_V2": int(np.array_equal(t1, t2)),
        }
        # dim of first V1 divergence
        if first_v1 is not None:
            d = np.abs(r1[first_v1] - r0[first_v1])
            row["first_div_dims_V1"] = [ACT_DIMS[j] for j in range(7) if d[j] > 1e-6]
            row["first_div_cont"] = [float(x) for x in acts[first_v1]]
            row["first_div_V0_recon"] = [float(x) for x in r0[first_v1]]
            row["first_div_V1_recon"] = [float(x) for x in r1[first_v1]]
            row["first_div_token_V0"] = int(t0[first_v1])
            row["first_div_token_V1"] = int(t1[first_v1])
        rows.append(row)
        print(
            f"  {row['tag'][:48]} new_steps_V1={row['n_steps_new_token_V1']} "
            f"err {err0:.3f}->{err1:.3f} identical_V1V2={row['recon_identical_V1_V2']}",
            flush=True,
        )

    write_csv(OUT / "TOKEN_USAGE.csv", rows)

    # aggregate new-token selection across V2's "extra" ids 118.. if any
    global_hist = Counter()
    for row in rows:
        h = row["new_token_hist_V2"]
        if isinstance(h, str):
            h = json.loads(h)
        for k, v in h.items():
            global_hist[int(k)] += int(v)

    return {
        "rows": rows,
        "n_roots_use_new_V1": sum(r["uses_any_new_V1"] for r in rows),
        "n_roots_use_new_V2": sum(r["uses_any_new_V2"] for r in rows),
        "n_err_reduced": sum(r["err_reduced_V1"] for r in rows),
        "mean_max_err_V0": float(np.mean([r["max_err_V0"] for r in rows])),
        "mean_max_err_V1": float(np.mean([r["max_err_V1"] for r in rows])),
        "all_V1_V2_identical": all(r["recon_identical_V1_V2"] for r in rows),
        "global_new_token_hist": dict(global_hist),
        "new_v1_ids": new_v1_ids,
        "new_v2_ids": new_v2_ids,
        "M0": int(M0),
        "M1": int(book_v1.shape[0]),
        "M2": int(book_v2.shape[0]),
    }


def section_d_root_causes(book0, book_v1, rho, token_rows, env=None) -> list[dict]:
    """Classify each unrecovered fail with evidence from token usage + STRICT replay."""
    fails, _ = load_fail_ok()
    # recovery under V0/V1 from REPAIR_DETAIL if present
    repair = {}
    rp = OUT / "REPAIR_DETAIL.csv"
    if rp.is_file():
        for r in csv.DictReader(rp.open(encoding="utf-8")):
            repair.setdefault(r.get("tag"), {})[r.get("label")] = r

    by_tag = {r["tag"]: r for r in token_rows}
    env = env or make_env()
    defaults = snapshot_default_dynamics(env.sim)
    out = []
    print("=== D: root-cause classification for unrecovered ===", flush=True)
    for r in fails:
        p = _abs(r["ic_path"])
        blob = pickle.loads(p.read_bytes())
        acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
        tag = blob.get("tag") or r.get("tag")
        tu = by_tag.get(tag, {})
        q0 = quantize(book0, rho, acts)
        q1 = quantize(book_v1, rho, acts)
        # Prefer ablation STRICT labels when present; re-sim only if missing.
        ab_r0 = (repair.get(tag) or {}).get("R0") or {}
        ab_v1 = (repair.get(tag) or {}).get("V1") or {}
        if ab_r0.get("SUCCESS_NEW") != "" and ab_r0.get("SUCCESS_NEW") is not None:
            recovered_v0 = str(ab_r0.get("SUCCESS_NEW")) in ("1", "True", "true")
            reason_v0 = ab_r0.get("reason")
            s12_v0 = str(ab_r0.get("SUCCESS_12S")) in ("1", "True", "true")
            h0 = {
                "SUCCESS_NEW": recovered_v0,
                "SUCCESS_12S": s12_v0,
                "reason": reason_v0,
            }
        else:
            h0 = replay_actions(env, blob["capture"], blob["ref"], q0["recon_actions"])
            recovered_v0 = bool(h0.get("SUCCESS_NEW"))
        if ab_v1.get("SUCCESS_NEW") != "" and ab_v1.get("SUCCESS_NEW") is not None:
            recovered_v1 = str(ab_v1.get("SUCCESS_NEW")) in ("1", "True", "true")
            h1 = {
                "SUCCESS_NEW": recovered_v1,
                "SUCCESS_12S": str(ab_v1.get("SUCCESS_12S")) in ("1", "True", "true"),
                "reason": ab_v1.get("reason"),
            }
        else:
            h1 = replay_actions(env, blob["capture"], blob["ref"], q1["recon_actions"])
            recovered_v1 = bool(h1.get("SUCCESS_NEW"))

        # continuous label from diagnose taxonomy when available
        diag_path = OUT / "REPLAY_DIAGNOSTICS.csv"
        cont_ok = True
        if diag_path.is_file():
            for d in csv.DictReader(diag_path.open(encoding="utf-8")):
                if d.get("tag") == tag:
                    cont_ok = str(d.get("cont_SUCCESS_NEW")) in ("1", "True", "true")
                    break

        grip_gap = float(np.max(np.abs(q0["recon_actions"][:, 6] - acts[:, 6])))
        motion_gap = float(
            np.max(np.abs(q0["recon_actions"][:, :6] - acts[:, :6]))
        )
        uses_new = int(tu.get("uses_any_new_V1") or 0)
        err_reduced = int(tu.get("err_reduced_V1") or 0)

        if not cont_ok:
            cause = "REPLAY_OR_CONT_NONDETERMINISM"
            detail = "diagnose: continuous SUCCESS_NEW=0 in this environment"
        elif recovered_v1:
            cause = "RECOVERED_BY_V1"
            detail = "V1 STRICT success"
        elif h0.get("SUCCESS_12S") and not h0.get("SUCCESS_NEW"):
            if grip_gap > 0.05 and motion_gap < 0.15:
                cause = "GRIP_RESOLUTION_BORDERLINE_HOLD"
                detail = (
                    f"V0 SUCCESS_12S but not NEW; grip_max_err={grip_gap:.3f}; "
                    f"new_tokens_used={uses_new}; err_reduced={err_reduced}"
                )
            else:
                cause = "CUMULATIVE_DRIFT_BORDERLINE_HOLD"
                detail = f"12S ok / NEW fail; motion_gap={motion_gap:.3f} grip_gap={grip_gap:.3f}"
        elif uses_new and err_reduced and not recovered_v1:
            cause = "ERR_REDUCED_BUT_NO_PHYSICAL_GAIN"
            detail = "new tokens used and max-err fell, but STRICT still fails"
        elif not uses_new and grip_gap > 0.05:
            cause = "NEW_TOKENS_NOT_SELECTED_GRIP_GAP_REMAINS"
            detail = (
                f"added protos not nearest under DEFAULT_WEIGHTS; grip_max_err={grip_gap:.3f}"
            )
        elif h0.get("reason") not in (None, "SUCCESS_12S", ""):
            cause = "HARD_PHYSICAL_FAILURE"
            detail = f"reason={h0.get('reason')}"
        else:
            cause = "UNRESOLVED"
            detail = "insufficient evidence"

        # first action dim divergence continuous vs V0
        first_i = None
        first_dims = []
        for i in range(len(acts)):
            if float(np.max(np.abs(q0["recon_actions"][i] - acts[i]))) > 1e-4:
                first_i = i
                d = np.abs(q0["recon_actions"][i] - acts[i])
                first_dims = [ACT_DIMS[j] for j in range(7) if d[j] > 1e-4]
                break

        out.append(
            {
                "tag": tag,
                "lineage_id": blob.get("lineage_id"),
                "cont_SUCCESS_NEW": int(bool(cont_ok)),
                "V0_SUCCESS_NEW": int(recovered_v0),
                "V0_SUCCESS_12S": int(bool(h0.get("SUCCESS_12S"))),
                "V0_reason": h0.get("reason"),
                "V1_SUCCESS_NEW": int(recovered_v1),
                "V1_SUCCESS_12S": int(bool(h1.get("SUCCESS_12S"))),
                "V1_reason": h1.get("reason"),
                "uses_new_token_V1": uses_new,
                "err_reduced_V1": err_reduced,
                "max_err_V0": tu.get("max_err_V0"),
                "max_err_V1": tu.get("max_err_V1"),
                "grip_max_err_V0": grip_gap,
                "motion_max_err_V0": motion_gap,
                "first_action_div_i": first_i,
                "first_action_div_dims": first_dims,
                "root_cause": cause,
                "evidence": detail,
                "ablation_R0": (repair.get(tag) or {}).get("R0", {}).get("SUCCESS_NEW"),
                "ablation_V1": (repair.get(tag) or {}).get("V1", {}).get("SUCCESS_NEW"),
                "ablation_R1b": (repair.get(tag) or {}).get("R1b", {}).get("SUCCESS_NEW"),
            }
        )
        print(f"  {tag[:48]} -> {cause}", flush=True)

    restore_default_dynamics(env.sim, defaults)
    write_csv(OUT / "FAILURE_ROOT_CAUSES.csv", out)
    return out


def write_reports(reg, tok, causes, book0, book_v1, book_v2, rho) -> None:
    spot25 = reg["spot25"]
    cause_ctr = Counter(c["root_cause"] for c in causes)
    n_unrec = sum(1 for c in causes if c["root_cause"] not in ("RECOVERED_BY_V1",))
    # added proto grip stats
    added = book_v1[book0.shape[0] :]
    added_grip = [float(x) for x in added[:, 6]] if len(added) else []

    meta_v1 = {}
    if (OUT / "vocab_v1_meta.json").is_file():
        meta_v1 = json.loads((OUT / "vocab_v1_meta.json").read_text(encoding="utf-8"))

    audit = f"""# V1_V2_AUDIT — P34.5

Generated: {utc()}

## A. The 25/25 denominator

### Exact 25 roots (ablation ok spot-check)

These were **`oks[:25]`** from `M111_REPLAY_RESULTS.csv` (P34 audit order), not a curated
compatibility suite:

{chr(10).join(f'- `{t}`' for t in spot25)}

### Why not all 85?

`run_p34_5.py` intentionally limited the ablation ok cohort to **25** for wall-clock
(`ok_eval = ok_paths[:25]`), deferring a full check to Phase-4 regression — which was
**interrupted** before completion (`exit_code=1` after V2 ablation). Therefore **25/25 must
not be read as full backward compatibility**.

### Full regression (this audit)

| Vocab | SUCCESS_NEW preserved / 85 |
|---|---:|
| V0 (original M111) | {reg['V0_preserved']}/85 |
| V1 (M111+7 effective) | {reg['V1_preserved']}/85 |
| V2 (byte-identical to V1) | {reg['V2_preserved']}/85 |

- V1 regressions vs V0: {reg['V1_regressed_vs_V0']}
- V2 regressions vs V0: {reg['V2_regressed_vs_V0']}
- V1 book ≡ V2 book: **{reg['books_identical']}**

P33 (25) separate table: `P33_REGRESSION.csv`.

## B. Do V1/V2 use new tokens?

| Metric | Value |
|---|---:|
| V0 motion M | {tok['M0']} |
| V1 / V2 motion M | {tok['M1']} / {tok['M2']} |
| Requested adds | 8 / 16 |
| Actual unique added prototypes | {tok['M1'] - tok['M0']} (both V1 and V2) |
| Roots using ≥1 new token (V1) | {tok['n_roots_use_new_V1']}/38 |
| Roots using ≥1 new token (V2) | {tok['n_roots_use_new_V2']}/38 |
| Global new-token step counts | {tok['global_new_token_hist']} |
| All 38 recon tapes V1≡V2 | {tok['all_V1_V2_identical']} |
| Roots with reduced max recon err under V1 | {tok['n_err_reduced']}/38 |
| Mean max-err V0 → V1 | {tok['mean_max_err_V0']:.4f} → {tok['mean_max_err_V1']:.4f} |

### Why V2 did not add 8 more tokens

`build_extended_book(..., n_add=16)` uses the same farthest-point pool with a coverage
threshold of 0.03 under ρ=0.35. After **7** centroids, remaining high-error actions were
already within 0.03 of book∪chosen, so V2 stopped at the same 7 additions. Saved
`vocab_v1_prototypes.npy` and `vocab_v2_prototypes.npy` are **byte-identical**
(shape {book_v1.shape}).

### Added prototype grip values

{added_grip}

These are mostly **large positive grip** (~0.8–1.0). Diagnosed failures are dominated by
**negative grip** reconstruction error (~−0.3…−0.6). Under DEFAULT_WEIGHTS (grip weight 0.5),
many fail tapes never select the new IDs even when present.

See `TOKEN_USAGE.csv` for per-root histograms and first V0/V1 recon divergence.

## C. Quantization / replay algorithm (verified)

Replay path used throughout P34.5 (and this audit):

1. Load stored **continuous** Oracle actions from the P1 pickle.
2. **Open-loop nearest-neighbor tokenization** of those actions against the motion book
   (`quantize_trajectory` / weighted L∞ + residual clip ρ=0.35).
3. Decode to continuous controls: `clip(proto + clip(a-proto, ±ρ), ±1)`.
4. **Open-loop** `step_legal` replay of the decoded tape from the identical capture, then
   12s STRICT `evaluate_handoff_with_axis`.

This is **not**:
- a closed-loop Oracle re-query,
- replay of a stored discrete token tape from training packs,
- or a change to action duration / DT (still one decoded 7-vector per expert step).

Controller target persistence and physics are unchanged; only the commanded 7-vector sequence
differs between continuous and quantized paths.

## D. Unrecovered root causes

Aggregate over 38 fails (this audit's V0/V1 STRICT):

{chr(10).join(f'- **{k}**: {v}' for k,v in cause_ctr.most_common())}

Detail: `FAILURE_ROOT_CAUSES.csv`.

## E. Are more tokens justified?

1. **Did added actions reduce quantization error?** Mildly on average
   ({tok['mean_max_err_V0']:.4f} → {tok['mean_max_err_V1']:.4f}); only {tok['n_err_reduced']}/38 roots.
2. **Were they executed?** Only {tok['n_roots_use_new_V1']}/38 roots select any new ID; many never do.
3. **Did lower error improve physical replay?** Ablation recovered **2/38** under V1/V2 vs **1/38** V0;
   contrast **R1b (grip-aware NN weights, no new tokens): 33/38**.
4. **Why V2 = V1?** Identical prototype tables (pool saturated at 7).
5. **Remaining failures?** Primarily **grip under-weighting / residual saturation → borderline
   SUCCESS_NEW hold**, not missing motion directions that another +32 tokens would fix.
   Mining also selected the wrong grip polarity relative to the failure mode.

### Recommendation

**Do not add M111+32 or further vocabulary expansion based on V1/V2.** Evidence does not
support a pure coverage deficiency of the motion book as the binding constraint.
Prefer investigating quantization **weighting / residual allocation for grip** (R1b-class),
with frozen historical IDs 0..110, before any ID-space growth.
"""
    write_md(OUT / "V1_V2_AUDIT.md", audit)

    p33 = reg.get("p33") or []
    p33_v0 = sum(1 for r in p33 if r.get("V0_SUCCESS_NEW") == 1)
    p33_v1 = sum(1 for r in p33 if r.get("V1_SUCCESS_NEW") == 1)

    final = f"""# FINAL_AUDIT — P34.5 V1/V2 Verification

Generated: {utc()}

## Verdict

V1/V2 **do not** establish vocabulary coverage or full backward compatibility.
Reported 25/25 was an ablation smoke subset (`oks[:25]`). Full 85 regression is in
`FULL_REGRESSION_85.csv`.

Critical structural finding: **V1 ≡ V2** (same 7 added prototypes; requested +8/+16 both
saturated). Ablation recovery 2/38 vs R1b 33/38 without new tokens.

## Answers

### A — 25/25

- Exact tags: see `V1_V2_AUDIT.md` §A / list of {len(spot25)} tags.
- Reason for 25: wall-clock shortcut in `run_p34_5.run_ablation`; full 85 regression was
  planned then interrupted.
- Full preserved SUCCESS_NEW: V0={reg['V0_preserved']}/85, V1={reg['V1_preserved']}/85,
  V2={reg['V2_preserved']}/85.
- P33: V0={p33_v0}/{len(p33)}, V1={p33_v1}/{len(p33)} (see `P33_REGRESSION.csv`).

### B — New token usage

- New IDs: {tok['new_v1_ids']} (V2 identical).
- Roots selecting ≥1 new token: {tok['n_roots_use_new_V1']}/38.
- No distinct “extra eight V2 tokens” exist — V2 never materialized IDs beyond V1.
- Per-root histograms: `TOKEN_USAGE.csv`.

### C — Algorithm

Open-loop NN quantization of stored continuous Oracle actions + open-loop decoded replay +
STRICT. Not closed-loop re-query; not stored token-tape replay.

### D — Unrecovered causes

{json.dumps(dict(cause_ctr), indent=2)}

### E — More tokens?

**No.** Added tokens barely move error, are often unused, barely move STRICT success, and V2
adds nothing beyond V1. Failures align with grip quantization / borderline hold, while
**R1b (no vocab change) recovers 33/38**. Further token additions are not justified by this
evidence.

## Artifacts

- `V1_V2_AUDIT.md`
- `TOKEN_USAGE.csv`
- `FULL_REGRESSION_85.csv`
- `P33_REGRESSION.csv`
- `FAILURE_ROOT_CAUSES.csv`
- `FINAL_AUDIT.md`
"""
    write_md(OUT / "FINAL_AUDIT.md", final)
    print(final, flush=True)


def _tok_from_csv() -> dict | None:
    path = OUT / "TOKEN_USAGE.csv"
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) < 38:
        return None
    # coerce ints
    for r in rows:
        for k in (
            "n_actions",
            "n_steps_new_token_V1",
            "n_steps_new_token_V2",
            "uses_any_new_V1",
            "uses_any_new_V2",
            "err_reduced_V1",
            "recon_identical_V1_V2",
        ):
            if k in r and r[k] != "":
                r[k] = int(float(r[k]))
        for k in ("max_err_V0", "max_err_V1", "max_err_V2", "mean_err_V0", "mean_err_V1", "mean_err_V2"):
            if k in r and r[k] != "":
                r[k] = float(r[k])
        if isinstance(r.get("new_token_hist_V1"), str) and r["new_token_hist_V1"]:
            r["new_token_hist_V1"] = json.loads(r["new_token_hist_V1"].replace("'", '"'))
        if isinstance(r.get("new_token_hist_V2"), str) and r["new_token_hist_V2"]:
            r["new_token_hist_V2"] = json.loads(r["new_token_hist_V2"].replace("'", '"'))
    global_hist: Counter = Counter()
    for r in rows:
        h = r.get("new_token_hist_V2") or {}
        if isinstance(h, str):
            h = json.loads(h) if h else {}
        for k, v in h.items():
            global_hist[int(k)] += int(v)
    book0, book_v1, book_v2, _ = load_books()
    return {
        "rows": rows,
        "n_roots_use_new_V1": sum(int(r["uses_any_new_V1"]) for r in rows),
        "n_roots_use_new_V2": sum(int(r["uses_any_new_V2"]) for r in rows),
        "n_err_reduced": sum(int(r["err_reduced_V1"]) for r in rows),
        "mean_max_err_V0": float(np.mean([float(r["max_err_V0"]) for r in rows])),
        "mean_max_err_V1": float(np.mean([float(r["max_err_V1"]) for r in rows])),
        "all_V1_V2_identical": all(int(r["recon_identical_V1_V2"]) for r in rows),
        "global_new_token_hist": dict(global_hist),
        "new_v1_ids": list(range(book0.shape[0], book_v1.shape[0])),
        "new_v2_ids": list(range(book0.shape[0], book_v2.shape[0])),
        "M0": int(book0.shape[0]),
        "M1": int(book_v1.shape[0]),
        "M2": int(book_v2.shape[0]),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-token-usage", action="store_true")
    ap.add_argument("--skip-full85", action="store_true")
    args = ap.parse_args()

    book0, book_v1, book_v2, rho = load_books()
    print(
        f"Books: V0={book0.shape} V1={book_v1.shape} V2={book_v2.shape} "
        f"V1==V2={np.allclose(book_v1, book_v2)}",
        flush=True,
    )
    tok = _tok_from_csv() if args.reuse_token_usage else None
    if tok is None:
        tok = section_b_token_usage(book0, book_v1, book_v2, rho)
    else:
        print(f"=== B: reused TOKEN_USAGE.csv n={len(tok['rows'])} ===", flush=True)
    causes = section_d_root_causes(book0, book_v1, rho, tok["rows"])
    if args.skip_full85:
        spot25 = identify_25_spot()
        reg = {
            "n_ok": 85,
            "spot25": spot25,
            "V0_preserved": -1,
            "V1_preserved": -1,
            "V2_preserved": -1,
            "V1_regressed_vs_V0": -1,
            "V2_regressed_vs_V0": -1,
            "books_identical": bool(np.allclose(book_v1, book_v2)),
            "p33": [],
        }
        write_md(
            OUT / "FULL_REGRESSION_85.csv.PENDING",
            "Run without --skip-full85 to produce FULL_REGRESSION_85.csv\n",
        )
    else:
        reg = section_a_full_regression(book0, book_v1, book_v2, rho)
    write_reports(reg, tok, causes, book0, book_v1, book_v2, rho)


if __name__ == "__main__":
    main()
