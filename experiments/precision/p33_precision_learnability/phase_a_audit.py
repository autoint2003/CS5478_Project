"""P33 Phase A — Precision data supply audit (physical P0/P1/P2/P3 classification)."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.expert_pipeline.loader import load_expert_record, load_manifest  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import (  # noqa: E402
    make_env,
    replay_actions,
)
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/p33_precision_learnability"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P23 = ROOT / "results/diagnostics/p23_c_d2_sft"
FULLCHAIN = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
OFFICIAL = ROOT / "data/experts_v2_official"


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


def _d2_start(env, capture) -> None:
    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)


def probe_state(env, capture, ref) -> dict[str, Any]:
    _d2_start(env, capture)
    sim = env.sim
    m = measure(sim, U_TGT, P_TGT)
    cs = contact_summary(sim)
    err = measure_errors(sim)
    return {
        "e_pos_mm": float(m.get("e_pos_mm") or 0.0),
        "e_theta_deg": float(m.get("e_theta_deg") or 0.0),
        "dx_mm": float(err.get("dx_mm") or 0.0),
        "dz_mm": float(err.get("dz_mm") or 0.0),
        "dtheta_deg": float(err.get("dtheta_deg") or 0.0),
        "bilateral": int(bool(cs.get("bilateral"))),
        "nL": cs.get("nL"),
        "nR": cs.get("nR"),
        "fsm_phase": str(sim.fsm.phase),
        "capture_fp": capture_fingerprint(capture),
    }


def immediate_handoff(env, capture, ref) -> dict[str, Any]:
    _d2_start(env, capture)
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    return {
        "imm_SUCCESS_NEW": int(bool(hand.get("SUCCESS_NEW"))),
        "imm_SUCCESS_12S": int(bool(hand.get("SUCCESS_12S"))),
        "imm_reason": hand.get("std_reason") or hand.get("reason"),
    }


def expert_precision_handoff(env, capture, ref, actions) -> dict[str, Any]:
    acts = np.asarray(actions, np.float32).reshape(-1, 7) if actions is not None else np.zeros((0, 7), np.float32)
    r = replay_actions(env, capture, ref, acts, stabilize_n=8)
    norms = np.linalg.norm(acts, axis=1) if len(acts) else np.array([])
    return {
        "prec_SUCCESS_NEW": int(bool(r["SUCCESS_NEW"])),
        "prec_SUCCESS_12S": int(bool(r["SUCCESS_12S"])),
        "prec_reason": r.get("reason"),
        "n_actions": int(len(acts)),
        "n_nonzero_actions": int(np.sum(norms > 1e-4)) if len(norms) else 0,
        "precision_executed_proxy": int(np.sum(norms > 1e-4) > 0) if len(norms) else 0,
    }


def classify(imm_ok: bool, prec_ok: bool, evidence: bool) -> str:
    if not evidence:
        return "PU"
    if imm_ok and prec_ok:
        return "P2"
    if imm_ok and not prec_ok:
        return "P0"  # immediate succeeds (Precision may fail or not needed)
    if (not imm_ok) and prec_ok:
        return "P1"
    return "P3"


def eval_holdout_lineages() -> set[str]:
    """Lineages used in P21/P22/P23/P32 evaluation — never train."""
    out: set[str] = set()
    p21 = json.loads((P21 / "SFT_FULL_HORIZON_MANIFEST.json").read_text(encoding="utf-8"))
    for r in p21["trajectories"]:
        if r.get("split") == "val":
            out.add(str(r.get("lineage_id")))
    p23 = json.loads((P23 / "COMBINED_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    for r in p23["trajectories"]:
        if r.get("split") == "val":
            out.add(str(r.get("lineage_id")))
    for x in p23.get("d2_val_lineages") or []:
        out.add(str(x))
    return out


def load_fullchain_index() -> dict[str, dict]:
    path = FULLCHAIN / "manifest.jsonl"
    if not path.is_file():
        return {}
    idx = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = row.get("traj_id") or row.get("expert_id") or row.get("id")
        if tid:
            idx[str(tid)] = row
        # also index by continuous demo stem
        lin = row.get("lineage_id")
        if lin and lin not in idx:
            idx[str(lin)] = row
    return idx


def resolve_p21_prec_capture(row: dict, fc_idx: dict) -> tuple[Any, Any, np.ndarray | None, str]:
    """Return capture, ref, continuous actions, note."""
    tid = row["traj_id"]
    # try match fullchain by traj_id substring / lineage
    cand = None
    for k, v in fc_idx.items():
        if tid == k or tid in k or k in tid or row.get("lineage_id") == v.get("lineage_id"):
            if v.get("ic_pkl") or v.get("ic_path"):
                cand = v
                if row.get("lineage_id") == v.get("lineage_id"):
                    break
    if cand is None:
        return None, None, None, "NO_FULLCHAIN_IC"
    ic_rel = cand.get("ic_pkl") or cand.get("ic_path")
    ic_path = FULLCHAIN / ic_rel
    if not ic_path.is_file():
        return None, None, None, f"MISSING_IC:{ic_rel}"
    blob = pickle.loads(ic_path.read_bytes())
    capture, ref = blob.get("capture"), blob.get("ref")
    acts = None
    demo = cand.get("continuous_demo") or {}
    dpath = demo.get("path")
    if dpath:
        dp = FULLCHAIN / dpath
        if dp.is_file():
            demo_blob = pickle.loads(dp.read_bytes()) if dp.suffix == ".pkl" else None
            if isinstance(demo_blob, dict):
                acts = demo_blob.get("actions") or demo_blob.get("actions_7d")
            # sometimes npz
    if acts is None:
        # use packaged NPZ only for token stats later; continuous from fullchain phases
        acts = blob.get("actions") or (blob.get("meta") or {}).get("actions")
    return capture, ref, (np.asarray(acts, np.float32) if acts is not None else None), "ok"


def audit_official(env, holdout: set[str]) -> list[dict]:
    rows = []
    man = load_manifest(OFFICIAL / "manifest.jsonl")
    for row in man:
        bid = str(row.get("branch_id") or "")
        if bid not in ("A_STABLE_PRECISION", "B_STABLE_PRECISION"):
            continue
        alias = "A2" if bid.startswith("A_") else "B2"
        skipped = bool(row.get("precision_skipped"))
        try:
            rec = load_expert_record(row)
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": row.get("expert_id") or row.get("attempt_id"),
                    "source": "experts_v2_official",
                    "alias": alias,
                    "branch_id": bid,
                    "lineage_id": row.get("source_ic_id"),
                    "precision_skipped_meta": int(skipped),
                    "class": "PU",
                    "note": f"load_fail:{exc}",
                    "holdout_eval": int(str(row.get("source_ic_id")) in holdout),
                }
            )
            continue
        capture = rec.get("capture")
        ref = rec.get("ref")
        acts = rec.get("actions_7d")
        if acts is None:
            acts = rec.get("actions")
        if capture is None or ref is None:
            rows.append(
                {
                    "candidate_id": row.get("expert_id"),
                    "source": "experts_v2_official",
                    "alias": alias,
                    "branch_id": bid,
                    "lineage_id": row.get("source_ic_id"),
                    "precision_skipped_meta": int(skipped),
                    "class": "PU",
                    "note": "MISSING_CAPTURE",
                    "holdout_eval": int(str(row.get("source_ic_id")) in holdout),
                }
            )
            continue
        st = probe_state(env, capture, ref)
        imm = immediate_handoff(env, capture, ref)
        prec = expert_precision_handoff(env, capture, ref, acts)
        # If meta says skipped and actions empty/near-zero, treat as direct continuation
        cls = classify(bool(imm["imm_SUCCESS_NEW"]), bool(prec["prec_SUCCESS_NEW"]), True)
        # Override: skipped with imm success → P0 (not genuine Precision)
        if skipped and imm["imm_SUCCESS_NEW"]:
            cls = "P0"
        rows.append(
            {
                "candidate_id": row.get("expert_id") or row.get("attempt_id"),
                "source": "experts_v2_official",
                "alias": alias,
                "branch_id": bid,
                "lineage_id": str(row.get("source_ic_id")),
                "physical_root": str(row.get("source_ic_id")),
                "precision_skipped_meta": int(skipped),
                "precision_executed_meta": int(bool(rec.get("precision_executed"))),
                **st,
                **imm,
                **prec,
                "class": cls,
                "holdout_eval": int(str(row.get("source_ic_id")) in holdout or any(h in str(row.get("source_ic_id")) for h in holdout)),
                "n_actions_manifest": row.get("n_actions"),
                "SUCCESS_NEW_meta": row.get("SUCCESS_NEW"),
                "note": "ok",
            }
        )
        print(f"  official {alias} {row.get('expert_id')}: class={cls} skip={skipped}", flush=True)
    return rows


def audit_p21_a_prec(env, holdout: set[str]) -> list[dict]:
    rows = []
    p21 = json.loads((P21 / "SFT_FULL_HORIZON_MANIFEST.json").read_text(encoding="utf-8"))
    fc_idx = load_fullchain_index()
    prec = [r for r in p21["trajectories"] if r.get("includes_precision") and r.get("family") == "A"]
    for r in prec:
        capture, ref, acts, note = resolve_p21_prec_capture(r, fc_idx)
        lin = str(r.get("lineage_id"))
        base = {
            "candidate_id": r["traj_id"],
            "source": "p21_a_precision",
            "alias": "A_fullchain_prec",
            "branch_id": "P21_A_INCLUDES_PRECISION",
            "lineage_id": lin,
            "physical_root": lin,
            "split_declared": r.get("split"),
            "holdout_eval": int(lin in holdout or r.get("split") == "val"),
            "npz_relpath": r.get("npz_relpath"),
        }
        if capture is None:
            rows.append({**base, "class": "PU", "note": note})
            print(f"  p21 {r['traj_id'][:40]} PU {note}", flush=True)
            continue
        st = probe_state(env, capture, ref)
        imm = immediate_handoff(env, capture, ref)
        # For fullchain, continuous demo may include Stage-1+Precision; if missing, PU for prec arm
        if acts is None:
            cls = "PU"
            prec = {"prec_SUCCESS_NEW": None, "prec_SUCCESS_12S": None, "prec_reason": "NO_ACTIONS", "n_actions": 0, "n_nonzero_actions": 0, "precision_executed_proxy": 0}
        else:
            prec = expert_precision_handoff(env, capture, ref, acts)
            # Fullchain actions from recovery IC: replaying full tape is valid expert
            cls = classify(bool(imm["imm_SUCCESS_NEW"]), bool(prec["prec_SUCCESS_NEW"]), True)
        rows.append({**base, **st, **imm, **prec, "class": cls, "note": note, "precision_skipped_meta": 0})
        print(f"  p21 {r['traj_id'][:40]} class={cls}", flush=True)
    return rows


def audit_d2(env, holdout: set[str]) -> list[dict]:
    rows = []
    p23 = json.loads((P23 / "COMBINED_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    d2 = [r for r in p23["trajectories"] if r.get("family") == "D2"]
    ic_dirs = [
        ROOT / "results/diagnostics/p14_small_error_d2/ics",
        ROOT / "results/diagnostics/p15_forced_precision_d2/ics",
        ROOT / "results/diagnostics/p16_precision_skill_expansion/ics",
    ]

    def find_ic(tid: str) -> Path | None:
        stem = tid.split("__")[-1]
        for folder in ic_dirs:
            if not folder.is_dir():
                continue
            for p in folder.glob("*.pkl"):
                if p.stem == stem or stem in p.stem or p.stem in tid:
                    return p
        return None

    for r in d2:
        tid = r["traj_id"]
        lin = str(r.get("lineage_id"))
        found = find_ic(tid)
        base = {
            "candidate_id": tid,
            "source": "p23_d2",
            "alias": "D2",
            "branch_id": "D2",
            "lineage_id": lin,
            "physical_root": lin,
            "split_declared": r.get("split"),
            "holdout_eval": int(lin in holdout or r.get("split") == "val" or any(h in lin for h in holdout)),
            "npz_relpath": r.get("npz_relpath"),
            "source_corpus": r.get("source_corpus"),
        }
        if found is None:
            rows.append({**base, "class": "PU", "note": "IC_PKL_NOT_FOUND"})
            continue
        blob = pickle.loads(found.read_bytes())
        capture, ref = blob.get("capture"), blob.get("ref")
        acts = blob.get("actions") or (blob.get("meta") or {}).get("actions")
        if capture is None or ref is None:
            rows.append({**base, "class": "PU", "note": "MISSING_CAPTURE"})
            continue
        st = probe_state(env, capture, ref)
        imm = immediate_handoff(env, capture, ref)
        prec = expert_precision_handoff(env, capture, ref, acts)
        cls = classify(bool(imm["imm_SUCCESS_NEW"]), bool(prec["prec_SUCCESS_NEW"]), True)
        rows.append(
            {
                **base,
                **st,
                **imm,
                **prec,
                "class": cls,
                "note": "ok",
                "ic_path": str(found.relative_to(ROOT)).replace("\\", "/"),
                "precision_skipped_meta": 0,
            }
        )
        print(f"  d2 {tid[:48]} class={cls} split={r.get('split')}", flush=True)
    return rows


def run_phase_a() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== P33 Phase A: Precision data audit ===", flush=True)
    holdout = eval_holdout_lineages()
    env = make_env()
    all_rows: list[dict] = []
    print("--- official A2/B2 ---", flush=True)
    all_rows.extend(audit_official(env, holdout))
    print("--- P21 A includes_precision ---", flush=True)
    all_rows.extend(audit_p21_a_prec(env, holdout))
    print("--- P23 D2 ---", flush=True)
    all_rows.extend(audit_d2(env, holdout))

    _write_csv(OUT / "DATASET_MANIFEST.csv", all_rows)

    by_src_cls = Counter((r.get("source"), r.get("class")) for r in all_rows)
    by_cls = Counter(r.get("class") for r in all_rows)
    # Unique roots for P1 trainable (not holdout)
    p1 = [r for r in all_rows if r.get("class") == "P1"]
    p1_train = [r for r in p1 if not r.get("holdout_eval")]
    p1_roots = {r.get("physical_root") or r.get("lineage_id") for r in p1}
    p1_train_roots = {r.get("physical_root") or r.get("lineage_id") for r in p1_train}

    p0 = [r for r in all_rows if r.get("class") == "P0"]
    executed = [r for r in all_rows if int(r.get("n_nonzero_actions") or 0) > 0 or int(r.get("precision_executed_meta") or 0)]
    skipped = [r for r in all_rows if int(r.get("precision_skipped_meta") or 0) == 1]

    md = f"""# DATA_AUDIT — P33 Phase A

Generated: {_utc()}

## Scope

Physical re-evaluation of existing Precision-related candidates:
- Official `A_STABLE_PRECISION` (A2) / `B_STABLE_PRECISION` (B2)
- P21 A-family `includes_precision=true`
- P23 curated D2 (train+val)

Classification (12 s STRICT = `SUCCESS_NEW`):
- **P0**: immediate HANDOFF succeeds (and Precision not required for success; skipped B2 mapped here when imm ok)
- **P1**: immediate fails; Precision/expert tape succeeds — **genuine Precision-required**
- **P2**: both succeed (multi-valid)
- **P3**: Precision/expert fails (with or without imm fail)
- **PU**: missing capture/actions/evidence

Eval holdout lineages (P21/P22/P23/P32 val): **{len(holdout)}** ids — marked `holdout_eval=1`, not trainable.

## Counts (trajectories)

| Class | n |
|---|---:|
| P0 | {by_cls.get('P0',0)} |
| P1 | {by_cls.get('P1',0)} |
| P2 | {by_cls.get('P2',0)} |
| P3 | {by_cls.get('P3',0)} |
| PU | {by_cls.get('PU',0)} |
| **Total audited** | **{len(all_rows)}** |

### By source × class

| source / class | n |
|---|---:|
"""
    for (src, cls), n in sorted(by_src_cls.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        md += f"| {src} / {cls} | {n} |\n"

    md += f"""
## Unique root IC lineages

| Pool | Unique roots | Trajectory rows |
|---|---:|---:|
| All P1 | {len(p1_roots)} | {len(p1)} |
| **P1 trainable (not holdout)** | **{len(p1_train_roots)}** | **{len(p1_train)}** |
| All P0 | {len({r.get('physical_root') for r in p0})} | {len(p0)} |
| Precision-executed proxy (nonzero acts) | — | {len(executed)} |
| Meta precision_skipped | — | {len(skipped)} |

## Pre-expansion answer (Q1)

**Independent physically verified P1 training ICs before expansion: {len(p1_train_roots)}**
(trajectory rows={len(p1_train)}; holdout P1 rows={len(p1)-len(p1_train)}).

## Data-quality findings

1. Official A2 is tiny; B2 is skip-heavy — many B2 classify as **P0** after live immediate HANDOFF.
2. P1 is the only primary source of genuine closed-loop Precision correction demos.
3. P2 admits multiple valid strategies (imm HANDOFF *or* correction); do not treat all nonzero Oracle actions as required.
4. Sliding windows / repeated rollouts from the same root **do not** count as independent ICs.
5. Train/val must split by `physical_root` / `lineage_id` before any SFT.

## Next

Phase B may begin only after this audit. Target staged independent **P1** roots: 25 → 50 → 100 → 200,
plus held-out val/test (40 each) with disjoint lineages. Historical P32 17 D2 val ICs remain untouched.

See `DATASET_MANIFEST.csv`.
"""
    _write_md(OUT / "DATA_AUDIT.md", md)
    summary = {
        "n_audited": len(all_rows),
        "by_class": dict(by_cls),
        "p1_train_roots": len(p1_train_roots),
        "p1_train_rows": len(p1_train),
        "p1_total_roots": len(p1_roots),
        "holdout_lineages": len(holdout),
        "p1_train_root_ids": sorted(p1_train_roots),
    }
    (OUT / "phase_a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== Phase A done ===", summary, flush=True)
    return summary


if __name__ == "__main__":
    run_phase_a()
