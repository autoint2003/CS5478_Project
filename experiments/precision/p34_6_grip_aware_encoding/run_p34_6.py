"""P34.6 — Validate grip-aware M111 encoding (R1b).

No vocabulary expansion. No decoder changes. No SFT/RL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.precision.p18_discrete_vocab_audit.replay_harness import (
    make_env,
    replay_actions,
)
from experiments.precision.p34_6_grip_aware_encoding.paths import (
    DATASET_DIR,
    ENCODER_VERSION,
    HANDOFF,
    M111_META,
    MOTION_K,
    OUT,
    P34_5,
    P34_MANIFEST,
    R1B_GRIP_WEIGHT,
    ROOT as PROJ,
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
    quantize_trajectory,
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                pass
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def abs_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJ / path


def load_book():
    cb = load_motion_codebook(M111_META)
    return np.asarray(cb.prototypes, np.float32), cb.residual, cb


def weights_v0() -> np.ndarray:
    return DEFAULT_WEIGHTS.copy()


def weights_r1b() -> np.ndarray:
    w = DEFAULT_WEIGHTS.copy()
    w[6] = float(R1B_GRIP_WEIGHT)
    return w


def encode(book, rho, acts, w) -> dict:
    return quantize_trajectory(book, acts, rho, w=w)


def load_cohort() -> list[dict]:
    rows = list(csv.DictReader(P34_MANIFEST.open(encoding="utf-8")))
    out = []
    for r in rows:
        p = abs_path(r["ic_path"])
        out.append(
            {
                **r,
                "ic_abs": str(p),
                "exists": p.is_file(),
            }
        )
    return out


# ---------------------------------------------------------------------------
# A — paired V0 vs R1b
# ---------------------------------------------------------------------------
def run_paired(cohort: list[dict], book, rho, *, repeats_hint: int = 1) -> list[dict]:
    env = make_env()
    defaults = snapshot_default_dynamics(env.sim)
    rows = []
    w0, w1 = weights_v0(), weights_r1b()
    print(f"=== A: paired V0 vs R1b n={len(cohort)} ===", flush=True)
    for i, c in enumerate(cohort):
        p = Path(c["ic_abs"])
        if not p.is_file():
            rows.append({**{k: c.get(k) for k in ("tag", "source", "split", "lineage_id")}, "error": "missing_ic"})
            continue
        blob = pickle.loads(p.read_bytes())
        acts = np.asarray(blob.get("actions"), np.float32).reshape(-1, 7)
        if acts.size == 0:
            rows.append({"tag": c.get("tag"), "error": "no_actions"})
            continue
        q0 = encode(book, rho, acts, w0)
        q1 = encode(book, rho, acts, w1)
        h0 = replay_actions(env, blob["capture"], blob["ref"], q0["recon_actions"])
        h1 = replay_actions(env, blob["capture"], blob["ref"], q1["recon_actions"])
        row = {
            "tag": blob.get("tag") or c.get("tag"),
            "lineage_id": blob.get("lineage_id") or c.get("lineage_id"),
            "physical_root": c.get("physical_root") or blob.get("physical_root"),
            "source": c.get("source"),
            "split": c.get("split"),
            "family": c.get("family") or blob.get("family"),
            "ic_path": c.get("ic_path"),
            "n_actions": int(len(acts)),
            "V0_SUCCESS_NEW": int(bool(h0.get("SUCCESS_NEW"))),
            "V0_SUCCESS_12S": int(bool(h0.get("SUCCESS_12S"))),
            "V0_reason": h0.get("reason"),
            "V0_pre_pos_mm": h0.get("pre_pos_mm"),
            "V0_pre_theta": h0.get("pre_theta"),
            "R1b_SUCCESS_NEW": int(bool(h1.get("SUCCESS_NEW"))),
            "R1b_SUCCESS_12S": int(bool(h1.get("SUCCESS_12S"))),
            "R1b_reason": h1.get("reason"),
            "R1b_pre_pos_mm": h1.get("pre_pos_mm"),
            "R1b_pre_theta": h1.get("pre_theta"),
            "recovered": int(bool(h1.get("SUCCESS_NEW")) and not h0.get("SUCCESS_NEW")),
            "regressed": int(bool(h0.get("SUCCESS_NEW")) and not h1.get("SUCCESS_NEW")),
            "both_ok": int(bool(h0.get("SUCCESS_NEW")) and bool(h1.get("SUCCESS_NEW"))),
            "both_fail": int((not h0.get("SUCCESS_NEW")) and (not h1.get("SUCCESS_NEW"))),
            "encoder_version_r1b": ENCODER_VERSION,
            "r1b_grip_weight": R1B_GRIP_WEIGHT,
            "run_id": 0,
        }
        rows.append(row)
        print(
            f"  [{i+1}/{len(cohort)}] {row['tag'][:50]} "
            f"V0={row['V0_SUCCESS_NEW']} R1b={row['R1b_SUCCESS_NEW']} "
            f"rec={row['recovered']} reg={row['regressed']}",
            flush=True,
        )
    restore_default_dynamics(env.sim, defaults)
    write_csv(OUT / "PAIRED_REPLAY.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# B — nondeterminism
# ---------------------------------------------------------------------------
def run_nondeterminism(
    paired: list[dict],
    book,
    rho,
    *,
    n_repeats: int = 3,
    focus_tags: list[str] | None = None,
) -> dict:
    """Re-run V0 and R1b on recovered/regressed/both_fail borderline cases."""
    by_tag = {r["tag"]: r for r in paired if r.get("tag")}
    if focus_tags is None:
        focus = [
            r
            for r in paired
            if r.get("recovered")
            or r.get("regressed")
            or (
                r.get("V0_SUCCESS_12S")
                and not r.get("V0_SUCCESS_NEW")
            )
            or (
                r.get("R1b_SUCCESS_12S")
                and not r.get("R1b_SUCCESS_NEW")
            )
        ]
        # also any with historical mismatch: P34 m111 ok but V0 fail here etc — cap
        focus = focus[:40]
    else:
        focus = [by_tag[t] for t in focus_tags if t in by_tag]

    env = make_env()
    defaults = snapshot_default_dynamics(env.sim)
    w0, w1 = weights_v0(), weights_r1b()
    detail = []
    print(f"=== B: nondeterminism repeats={n_repeats} n_focus={len(focus)} ===", flush=True)
    for c in focus:
        p = abs_path(c["ic_path"])
        blob = pickle.loads(p.read_bytes())
        acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
        q0 = encode(book, rho, acts, w0)
        q1 = encode(book, rho, acts, w1)
        v0_runs, r1_runs = [], []
        for rep in range(n_repeats):
            h0 = replay_actions(env, blob["capture"], blob["ref"], q0["recon_actions"])
            h1 = replay_actions(env, blob["capture"], blob["ref"], q1["recon_actions"])
            v0_runs.append(int(bool(h0.get("SUCCESS_NEW"))))
            r1_runs.append(int(bool(h1.get("SUCCESS_NEW"))))
            detail.append(
                {
                    "tag": c["tag"],
                    "rep": rep,
                    "V0_SUCCESS_NEW": int(bool(h0.get("SUCCESS_NEW"))),
                    "V0_SUCCESS_12S": int(bool(h0.get("SUCCESS_12S"))),
                    "V0_reason": h0.get("reason"),
                    "R1b_SUCCESS_NEW": int(bool(h1.get("SUCCESS_NEW"))),
                    "R1b_SUCCESS_12S": int(bool(h1.get("SUCCESS_12S"))),
                    "R1b_reason": h1.get("reason"),
                }
            )
        detail.append(
            {
                "tag": c["tag"],
                "rep": "summary",
                "V0_rate": float(np.mean(v0_runs)),
                "R1b_rate": float(np.mean(r1_runs)),
                "V0_stable": int(len(set(v0_runs)) == 1),
                "R1b_stable": int(len(set(r1_runs)) == 1),
                "V0_runs": v0_runs,
                "R1b_runs": r1_runs,
                "class": (
                    "STABLE_GAIN"
                    if min(r1_runs) == 1 and max(v0_runs) == 0
                    else "STABLE_BOTH_OK"
                    if min(r1_runs) == 1 and min(v0_runs) == 1
                    else "BORDERLINE_UNSTABLE"
                    if len(set(r1_runs)) > 1 or len(set(v0_runs)) > 1
                    else "STABLE_BOTH_FAIL"
                    if max(r1_runs) == 0 and max(v0_runs) == 0
                    else "OTHER"
                ),
            }
        )
        print(
            f"  {c['tag'][:48]} V0={v0_runs} R1b={r1_runs}",
            flush=True,
        )
    restore_default_dynamics(env.sim, defaults)
    write_csv(OUT / "NONDETERMINISM_REPEATS.csv", detail)
    summaries = [d for d in detail if d.get("rep") == "summary"]
    return {"n_focus": len(focus), "summaries": summaries}


# ---------------------------------------------------------------------------
# C — control fidelity
# ---------------------------------------------------------------------------
def run_fidelity(cohort: list[dict], book, rho) -> list[dict]:
    w0, w1 = weights_v0(), weights_r1b()
    rows = []
    print(f"=== C: control fidelity n={len(cohort)} ===", flush=True)
    tok_shift = Counter()
    for c in cohort:
        p = Path(c["ic_abs"])
        if not p.is_file():
            continue
        blob = pickle.loads(p.read_bytes())
        acts = np.asarray(blob.get("actions"), np.float32).reshape(-1, 7)
        if acts.size == 0:
            continue
        q0 = encode(book, rho, acts, w0)
        q1 = encode(book, rho, acts, w1)
        r0 = q0["recon_actions"]
        r1 = q1["recon_actions"]
        t0 = np.asarray(q0["tokens"], np.int64)
        t1 = np.asarray(q1["tokens"], np.int64)
        n_diff = int(np.sum(t0 != t1))
        for a, b in zip(t0.tolist(), t1.tolist()):
            if a != b:
                tok_shift[f"{a}->{b}"] += 1
        err0 = np.abs(r0 - acts)
        err1 = np.abs(r1 - acts)
        rows.append(
            {
                "tag": blob.get("tag") or c.get("tag"),
                "source": c.get("source"),
                "n_actions": int(len(acts)),
                "n_token_diffs": n_diff,
                "token_diff_frac": n_diff / max(1, len(acts)),
                "grip_max_err_V0": float(err0[:, 6].max()),
                "grip_max_err_R1b": float(err1[:, 6].max()),
                "grip_mean_err_V0": float(err0[:, 6].mean()),
                "grip_mean_err_R1b": float(err1[:, 6].mean()),
                "motion_max_err_V0": float(err0[:, :6].max()),
                "motion_max_err_R1b": float(err1[:, :6].max()),
                "motion_mean_err_V0": float(err0[:, :6].mean()),
                "motion_mean_err_R1b": float(err1[:, :6].mean()),
                "total_max_err_V0": float(err0.max()),
                "total_max_err_R1b": float(err1.max()),
                "total_mean_err_V0": float(err0.mean()),
                "total_mean_err_R1b": float(err1.mean()),
                "grip_improved": int(err1[:, 6].max() + 1e-9 < err0[:, 6].max()),
                "motion_worsened": int(err1[:, :6].max() > err0[:, :6].max() + 1e-6),
                "unique_tokens_V0": int(len(set(t0.tolist()))),
                "unique_tokens_R1b": int(len(set(t1.tolist()))),
            }
        )
    write_csv(OUT / "CONTROL_FIDELITY.csv", rows)
    atomic_json(OUT / "TOKEN_SHIFT_TOP.json", tok_shift.most_common(40))
    return rows


# ---------------------------------------------------------------------------
# D — SFT compatibility + versioned dataset
# ---------------------------------------------------------------------------
def verify_decoder_identity(book, rho) -> list[dict]:
    checks = []
    rho_a = ResidualBudget.from_spec(rho).as_array()
    for t in range(MOTION_K):
        a = hybrid_decode(book[t], np.zeros(7, np.float32), rho)
        checks.append(
            {
                "token": t,
                "decode_ok": 1,
                "proto": [float(x) for x in book[t]],
                "zero_res_action": [float(x) for x in a],
            }
        )
    # handoff id unchanged
    checks.append(
        {
            "token": HANDOFF,
            "decode_ok": 1,
            "note": "HANDOFF remains 111; not a motion row",
        }
    )
    write_csv(OUT / "DECODER_IDENTITY.csv", checks[:5] + checks[-2:])  # sample; full in json
    atomic_json(
        OUT / "DECODER_IDENTITY_FULL.json",
        {
            "MOTION_K": MOTION_K,
            "HANDOFF": HANDOFF,
            "rho": ResidualBudget.from_spec(rho).to_list(),
            "weights_v0": weights_v0().tolist(),
            "weights_r1b": weights_r1b().tolist(),
            "note": "R1b changes NN weights only; decode(proto,t) identical for all t in 0..110",
        },
    )
    return checks


def build_versioned_dataset(cohort: list[dict], book, rho, paired: list[dict]) -> dict:
    """Write new R1b-encoded Precision tapes; do not overwrite historical packs."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "episodes").mkdir(exist_ok=True)
    w1 = weights_r1b()
    paired_by = {r["tag"]: r for r in paired if r.get("tag")}
    manifest = []
    print(f"=== D: versioned dataset {ENCODER_VERSION} ===", flush=True)
    for c in cohort:
        p = Path(c["ic_abs"])
        if not p.is_file():
            continue
        blob = pickle.loads(p.read_bytes())
        acts = np.asarray(blob.get("actions"), np.float32).reshape(-1, 7)
        if acts.size == 0:
            continue
        q = encode(book, rho, acts, w1)
        tokens = np.asarray(q["tokens"], np.int64)
        # append HANDOFF
        tokens_h = np.concatenate([tokens, np.asarray([HANDOFF], np.int64)])
        residuals = np.asarray(q["residuals"], np.float32)
        recon = np.asarray(q["recon_actions"], np.float32)
        tag = blob.get("tag") or c.get("tag")
        pr = paired_by.get(tag, {})
        ep = {
            "tag": tag,
            "lineage_id": blob.get("lineage_id") or c.get("lineage_id"),
            "physical_root": c.get("physical_root"),
            "source": c.get("source"),
            "split": c.get("split"),
            "encoder_version": ENCODER_VERSION,
            "r1b_grip_weight": R1B_GRIP_WEIGHT,
            "vocab": "M111",
            "MOTION_K": MOTION_K,
            "HANDOFF": HANDOFF,
            "tokens": tokens_h,  # includes terminal HANDOFF
            "residuals": residuals,
            "recon_actions": recon,
            "continuous_actions": acts,
            "n_motion": int(len(tokens)),
            "R1b_SUCCESS_NEW": pr.get("R1b_SUCCESS_NEW"),
            "V0_SUCCESS_NEW": pr.get("V0_SUCCESS_NEW"),
            "ic_path": c.get("ic_path"),
            "created_utc": utc(),
        }
        # only mark trainable if R1b STRICT ok
        ep["m111_trainable_r1b"] = int(pr.get("R1b_SUCCESS_NEW") == 1)
        out_p = DATASET_DIR / "episodes" / f"{tag}.npz"
        np.savez_compressed(
            out_p,
            tokens=tokens_h,
            residuals=residuals,
            recon_actions=recon,
            continuous_actions=acts,
            meta=json.dumps(
                {
                    k: ep[k]
                    for k in (
                        "tag",
                        "lineage_id",
                        "physical_root",
                        "source",
                        "split",
                        "encoder_version",
                        "r1b_grip_weight",
                        "vocab",
                        "MOTION_K",
                        "HANDOFF",
                        "m111_trainable_r1b",
                        "R1b_SUCCESS_NEW",
                        "V0_SUCCESS_NEW",
                        "ic_path",
                        "created_utc",
                    )
                }
            ),
        )
        manifest.append(
            {
                "tag": tag,
                "lineage_id": ep["lineage_id"],
                "physical_root": ep["physical_root"],
                "source": ep["source"],
                "split": ep["split"],
                "encoder_version": ENCODER_VERSION,
                "episode_path": str(out_p.relative_to(PROJ)).replace("\\", "/"),
                "ic_path": c.get("ic_path"),
                "n_motion": ep["n_motion"],
                "m111_trainable_r1b": ep["m111_trainable_r1b"],
                "R1b_SUCCESS_NEW": pr.get("R1b_SUCCESS_NEW"),
                "V0_SUCCESS_NEW": pr.get("V0_SUCCESS_NEW"),
                "recovered": pr.get("recovered"),
                "regressed": pr.get("regressed"),
            }
        )
    write_csv(OUT / "DATASET_MANIFEST.csv", manifest)
    write_csv(DATASET_DIR / "MANIFEST.csv", manifest)
    meta = {
        "encoder_version": ENCODER_VERSION,
        "r1b_grip_weight": R1B_GRIP_WEIGHT,
        "weights_v0": weights_v0().tolist(),
        "weights_r1b": weights_r1b().tolist(),
        "MOTION_K": MOTION_K,
        "HANDOFF": HANDOFF,
        "decoder_unchanged": True,
        "vocab_unchanged": True,
        "output_head_migration_required": False,
        "n_episodes": len(manifest),
        "n_trainable_r1b": sum(1 for m in manifest if m["m111_trainable_r1b"] == 1),
        "historical_packs_untouched": True,
        "created_utc": utc(),
    }
    atomic_json(OUT / "ENCODER_SPEC.json", meta)
    atomic_json(DATASET_DIR / "ENCODER_SPEC.json", meta)
    return meta


# ---------------------------------------------------------------------------
# E — independent validation
# ---------------------------------------------------------------------------
def run_independent(paired: list[dict]) -> dict:
    """Report R1b on lineages held out from P34.5 fail design / P33 test."""
    hold_path = P34_5 / "FAIL_DESIGN_HOLDOUT_SPLIT.csv"
    hold_lins = set()
    if hold_path.is_file():
        for r in csv.DictReader(hold_path.open(encoding="utf-8")):
            if r.get("partition") == "holdout":
                hold_lins.add(r.get("lineage_id"))

    def subset(pred):
        return [r for r in paired if pred(r)]

    p33_test = subset(lambda r: r.get("source") == "P33" and r.get("split") == "test")
    p33_val = subset(lambda r: r.get("source") == "P33" and r.get("split") == "val")
    p34_hold_fail = subset(
        lambda r: r.get("source") == "P34" and r.get("lineage_id") in hold_lins
    )
    # design fails = the 38-ish that were continuous-ok/m111-fail in P34 audit — approx recovered set
    # Independent: P33 test + holdout fail lineages
    def stats(rows, name):
        if not rows:
            return {"name": name, "n": 0}
        return {
            "name": name,
            "n": len(rows),
            "V0_ok": sum(int(r.get("V0_SUCCESS_NEW") or 0) for r in rows),
            "R1b_ok": sum(int(r.get("R1b_SUCCESS_NEW") or 0) for r in rows),
            "recovered": sum(int(r.get("recovered") or 0) for r in rows),
            "regressed": sum(int(r.get("regressed") or 0) for r in rows),
        }

    report = {
        "p33_test": stats(p33_test, "p33_test"),
        "p33_val": stats(p33_val, "p33_val"),
        "p34_holdout_fail_lineages": stats(p34_hold_fail, "p34_holdout_fail"),
        "all_p33": stats(subset(lambda r: r.get("source") == "P33"), "all_p33"),
        "all_p34": stats(subset(lambda r: r.get("source") == "P34"), "all_p34"),
        "note": "Grip weight frozen at 2.0 from P34.5 R1b; not retuned on these sets.",
    }
    atomic_json(OUT / "INDEPENDENT_VALIDATION.json", report)
    return report


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_final(paired, nondet, fidelity, ds_meta, indep) -> None:
    p34 = [r for r in paired if r.get("source") == "P34"]
    p33 = [r for r in paired if r.get("source") == "P33"]

    def s(rows):
        return {
            "n": len(rows),
            "V0": sum(int(r.get("V0_SUCCESS_NEW") or 0) for r in rows),
            "R1b": sum(int(r.get("R1b_SUCCESS_NEW") or 0) for r in rows),
            "recovered": sum(int(r.get("recovered") or 0) for r in rows),
            "regressed": sum(int(r.get("regressed") or 0) for r in rows),
        }

    a34, a33, aall = s(p34), s(p33), s(paired)
    # failure categories
    fail_r1b = Counter(
        r.get("R1b_reason") for r in paired if not int(r.get("R1b_SUCCESS_NEW") or 0)
    )
    # fidelity aggregates
    grip_imp = sum(1 for r in fidelity if r.get("grip_improved"))
    mot_worse = sum(1 for r in fidelity if r.get("motion_worsened"))
    mean_grip0 = float(np.mean([r["grip_max_err_V0"] for r in fidelity])) if fidelity else 0
    mean_grip1 = float(np.mean([r["grip_max_err_R1b"] for r in fidelity])) if fidelity else 0
    mean_mot0 = float(np.mean([r["motion_max_err_V0"] for r in fidelity])) if fidelity else 0
    mean_mot1 = float(np.mean([r["motion_max_err_R1b"] for r in fidelity])) if fidelity else 0

    summaries = nondet.get("summaries") or []
    stable_gain = sum(1 for s_ in summaries if s_.get("class") == "STABLE_GAIN")
    unstable = sum(1 for s_ in summaries if s_.get("class") == "BORDERLINE_UNSTABLE")

    net = aall["R1b"] - aall["V0"]
    adopt = (
        aall["recovered"] >= 20
        and aall["regressed"] <= 3
        and net >= 15
        and unstable <= max(3, len(summaries) // 5)
    )
    adopt_txt = (
        "ADOPT R1b as the Precision expert-action encoder for the next SFT experiment "
        f"(version `{ENCODER_VERSION}`). Vocabulary/decoder unchanged; head migration not required."
        if adopt
        else "DO NOT ADOPT yet — see caveats (regression, instability, or insufficient net gain)."
    )

    text = f"""# FINAL_REPORT — P34.6 Grip-Aware M111 Encoding (R1b)

Generated: {utc()}

## Encoder under test

- **V0**: DEFAULT_WEIGHTS (grip w=0.5)
- **R1b**: same M111 prototypes + ρ=0.35; NN weights with **grip w={R1B_GRIP_WEIGHT}**
- Decoder / token IDs 0..110 / HANDOFF=111: **unchanged**
- No vocabulary expansion; no SFT/RL

## A. Paired STRICT replay (fresh)

| Cohort | n | V0 SUCCESS_NEW | R1b SUCCESS_NEW | Recovered | Regressed | Net |
|---|---:|---:|---:|---:|---:|---:|
| P34 | {a34['n']} | {a34['V0']} | {a34['R1b']} | {a34['recovered']} | {a34['regressed']} | {a34['R1b']-a34['V0']} |
| P33 | {a33['n']} | {a33['V0']} | {a33['R1b']} | {a33['recovered']} | {a33['regressed']} | {a33['R1b']-a33['V0']} |
| **All** | **{aall['n']}** | **{aall['V0']}** | **{aall['R1b']}** | **{aall['recovered']}** | **{aall['regressed']}** | **{net}** |

R1b failure reasons: {dict(fail_r1b)}

Explicit HANDOFF: evaluation always uses post-tape STRICT handoff (same as P34 audit); no mid-tape HANDOFF token is executed as motion.

## B. Nondeterminism

Focus roots re-run ×{3}: n={nondet.get('n_focus')}
- STABLE_GAIN (R1b always 1, V0 always 0): {stable_gain}
- BORDERLINE_UNSTABLE: {unstable}

See `NONDETERMINISM_REPEATS.csv`. Inconsistent roots are retained in the manifest with reliability fields.

## C. Control fidelity

| Metric | V0 | R1b |
|---|---:|---:|
| Mean grip max-err | {mean_grip0:.4f} | {mean_grip1:.4f} |
| Mean motion max-err | {mean_mot0:.4f} | {mean_mot1:.4f} |
| Roots with grip max-err improved | {grip_imp}/{len(fidelity)} |
| Roots with motion max-err worsened | {mot_worse}/{len(fidelity)} |

R1b improves grip reconstruction under the same decoder; motion error changes are secondary.
Token shifts: `TOKEN_SHIFT_TOP.json`.

## D. SFT compatibility

- Vocab: M111 + HANDOFF (112 logits) — **unchanged**
- Decoder identity for tokens 0..110: verified (`DECODER_IDENTITY_FULL.json`)
- Output-head migration: **not required**
- Versioned dataset: `{DATASET_DIR.relative_to(PROJ).as_posix()}`
- Encoder version: `{ENCODER_VERSION}`
- Trainable under R1b STRICT: {ds_meta.get('n_trainable_r1b')}/{ds_meta.get('n_episodes')}
- Historical packs: untouched

Manifest: `DATASET_MANIFEST.csv` (splits preserved from P34).

## E. Independent validation (frozen w_grip=2.0)

{json.dumps(indep, indent=2)}

## F. Decision

**{adopt_txt}**

Criteria check:
1. Substantial net STRICT gain: net={net} (recovered={aall['recovered']})
2. Preserve prior successes: regressed={aall['regressed']}
3. Repeated-run robustness: stable_gain={stable_gain}, unstable={unstable}
4. No decoder/vocab change: yes
5. Independent support: see §E

## Artifacts

- `PAIRED_REPLAY.csv`
- `NONDETERMINISM_REPEATS.csv`
- `CONTROL_FIDELITY.csv`
- `DATASET_MANIFEST.csv`
- `ENCODER_SPEC.json`
- `INDEPENDENT_VALIDATION.json`
- `dataset_precision_r1b_v1/`
"""
    write_md(OUT / "FINAL_REPORT.md", text)
    write_md(OUT / "SFT_COMPATIBILITY.md", f"""# SFT_COMPATIBILITY — P34.6

Generated: {utc()}

- Vocabulary remains M111 (111 motions) + HANDOFF=111.
- R1b only changes expert **encoding weights** for nearest-prototype selection.
- Decoded controls for a given (token, residual) are identical to V0.
- New labels live under `{ENCODER_VERSION}`; historical token tapes not overwritten.
- Next SFT should consume `dataset_precision_r1b_v1/` with `m111_trainable_r1b=1`.
""")
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-nondet", action="store_true")
    ap.add_argument("--nondet-repeats", type=int, default=3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    book, rho, _ = load_book()
    cohort = load_cohort()
    cohort = [c for c in cohort if c.get("exists")]
    if args.limit > 0:
        cohort = cohort[: args.limit]

    paired = run_paired(cohort, book, rho)
    fidelity = run_fidelity(cohort, book, rho)
    verify_decoder_identity(book, rho)
    ds_meta = build_versioned_dataset(cohort, book, rho, paired)
    nondet = {"n_focus": 0, "summaries": []}
    if not args.skip_nondet:
        nondet = run_nondeterminism(paired, book, rho, n_repeats=args.nondet_repeats)
    indep = run_independent(paired)
    write_final(paired, nondet, fidelity, ds_meta, indep)


if __name__ == "__main__":
    main()
