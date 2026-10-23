#!/usr/bin/env python
"""Consolidate Stage-2 A/B/C mining into results/data_generation/stage2_abc_final/.

Preserves provenance. Does not start SFT/RL. Does not modify K49 / physics / success.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V1 = ROOT / "results/data_generation/stage2_abc_mining/v1"
V2 = ROOT / "results/data_generation/stage2_abc_mining/v2"
OUT = ROOT / "results/data_generation/stage2_abc_final"
K49_CONFIG = "K49_tok48_wy0p25"
HANDOFF = 47


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


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsafe(row), ensure_ascii=False) + "\n")


def classify_row(o: dict) -> str:
    cat = o.get("sft_category")
    k49 = o.get("k49_demo") or {}
    cont = o.get("continuous_demo") or {}
    if cat == "DIRECT_HANDOFF":
        return "DIRECT_HANDOFF"
    if cat == "INTERMEDIATE_CURRICULUM":
        return "INTERMEDIATE_CURRICULUM"
    if cat == "NONTRIVIAL_PRECISION":
        if k49.get("nontrivial_representability") or k49.get("verified"):
            return "NONTRIVIAL_K49_SUCCESS"
        if cont.get("verified") or cont.get("replay_SUCCESS_NEW"):
            return "CONTINUOUS_ONLY_SUCCESS"
        # fall through if labels incomplete
        if (o.get("precision") or {}).get("SUCCESS_NEW"):
            return "CONTINUOUS_ONLY_SUCCESS"
    return "ORACLE_FAILURE"


def source_episode(o: dict) -> str:
    meta_keys = ("ic_id", "hard_id", "root_id", "source_episode", "episode")
    for k in meta_keys:
        if o.get(k):
            return str(o[k])
    sid = str(o.get("sample_id") or "")
    parts = sid.split("__")
    if len(parts) >= 2:
        return parts[1]
    return sid


def load_outcomes() -> list[dict]:
    rows = []
    for base, pool in ((V1, "v1"), (V2, "v2")):
        od = base / "outcomes"
        if not od.exists():
            continue
        for p in sorted(od.glob("*.json")):
            o = json.loads(p.read_text(encoding="utf-8"))
            o["_pool"] = pool
            o["_outcome_path"] = str(p.relative_to(ROOT))
            rows.append(o)
    return rows


def resolve_ic_path(o: dict) -> Path | None:
    pool = V1 if o["_pool"] == "v1" else V2
    rel = o.get("ic_pkl")
    if rel:
        cand = pool / rel
        if cand.exists():
            return cand
        cand = ROOT / rel
        if cand.exists():
            return cand
    sid = o.get("sample_id")
    if sid:
        for base in (pool, V1, V2):
            cand = base / "ics" / f"{sid}.pkl"
            if cand.exists():
                return cand
    return None


def resolve_artifact(o: dict, key_path: str, kind: str) -> Path | None:
    """kind in continuous|tokens; key_path from demo dict."""
    pool = V1 if o["_pool"] == "v1" else V2
    demo = o.get(key_path) or {}
    rel = demo.get("path")
    if rel:
        for base in (pool, ROOT):
            cand = (base / rel) if base is pool else Path(rel)
            if not cand.is_absolute():
                cand = base / rel
            if cand.exists():
                return cand
    sid = o.get("sample_id")
    if not sid:
        return None
    if kind == "continuous":
        for base in (pool, V1, V2):
            for cand in (base / "continuous").glob(f"{sid}*"):
                return cand
    if kind == "tokens":
        for base in (pool, V1, V2):
            for cand in (base / "tokens").glob(f"{sid}*"):
                return cand
    return None


def copy_bytes(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def assign_splits(rows: list[dict], existing: dict | None) -> dict[str, str]:
    """Map source_episode -> split. Prefer v1 RL splits; else hash-stable 80/10/10."""
    ep_to_split = {}
    if existing:
        ep_to_split.update(existing)
    # Fill missing
    for o in rows:
        ep = source_episode(o)
        if ep in ep_to_split:
            continue
        h = int(hashlib.sha256(ep.encode()).hexdigest()[:8], 16) % 100
        ep_to_split[ep] = "train" if h < 80 else ("val" if h < 90 else "holdout")
    return ep_to_split


def load_v1_splits() -> dict[str, str]:
    out = {}
    man = V1 / "rl/rl_ic_manifest.jsonl"
    if not man.exists():
        return out
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("source_episode") and r.get("split"):
            out[str(r["source_episode"])] = str(r["split"])
    return out


def finalize():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in (
        "ics",
        "continuous",
        "tokens",
        "outcomes",
        "failures",
        "obs",
        "videos",
        "diagnostics",
    ):
        (OUT / sub).mkdir(exist_ok=True)

    rows = load_outcomes()
    # Dedup by sample_id preferring v2 then richer category
    rank = {
        "NONTRIVIAL_K49_SUCCESS": 5,
        "CONTINUOUS_ONLY_SUCCESS": 4,
        "DIRECT_HANDOFF": 3,
        "INTERMEDIATE_CURRICULUM": 2,
        "ORACLE_FAILURE": 1,
    }
    by_id = {}
    for o in rows:
        sid = o.get("sample_id")
        if not sid:
            continue
        final_cat = classify_row(o)
        o["_final_category"] = final_cat
        prev = by_id.get(sid)
        if prev is None or rank[final_cat] > rank[prev["_final_category"]] or (
            rank[final_cat] == rank[prev["_final_category"]] and o["_pool"] == "v2"
        ):
            by_id[sid] = o
    rows = list(by_id.values())

    ep_to_split = assign_splits(rows, load_v1_splits())
    manifest_path = OUT / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    counts = Counter()
    plan_counts = defaultdict(Counter)
    genuine_abc = Counter()
    hard_ic = Counter()
    integrity = {"missing_ic": [], "missing_continuous": [], "missing_tokens": [], "ok": 0}

    for o in sorted(rows, key=lambda r: r.get("sample_id") or ""):
        sid = o["sample_id"]
        plan = o.get("plan_provenance") or "UNKNOWN"
        final_cat = o["_final_category"]
        counts[final_cat] += 1
        plan_counts[plan][final_cat] += 1
        if plan == "HARD_IC":
            hard_ic[final_cat] += 1
        elif plan in ("A", "B", "C"):
            genuine_abc[(plan, final_cat)] += 1

        ep = source_episode(o)
        split = ep_to_split.get(ep, "train")

        # Copy IC
        ic_src = resolve_ic_path(o)
        ic_dst = OUT / "ics" / f"{sid}.pkl"
        if ic_src is None:
            integrity["missing_ic"].append(sid)
        else:
            copy_bytes(ic_src, ic_dst)

        cont_dst = None
        tok_dst = None
        if final_cat in ("NONTRIVIAL_K49_SUCCESS", "CONTINUOUS_ONLY_SUCCESS", "DIRECT_HANDOFF"):
            cs = resolve_artifact(o, "continuous_demo", "continuous")
            if cs:
                cont_dst = OUT / "continuous" / f"{sid}__continuous.npz"
                copy_bytes(cs, cont_dst)
            elif final_cat != "DIRECT_HANDOFF":
                integrity["missing_continuous"].append(sid)
            ts = resolve_artifact(o, "k49_demo", "tokens")
            if ts:
                tok_dst = OUT / "tokens" / f"{sid}__k49.npz"
                copy_bytes(ts, tok_dst)
            elif final_cat == "NONTRIVIAL_K49_SUCCESS":
                integrity["missing_tokens"].append(sid)

        if final_cat == "ORACLE_FAILURE":
            fdst = OUT / "failures" / f"{sid}__failure.json"
            atomic_json(
                fdst,
                {
                    "sample_id": sid,
                    "plan_provenance": plan,
                    "precision": o.get("precision"),
                    "direct": o.get("direct"),
                    "errors_entry": o.get("errors_entry"),
                    "source_pool": o["_pool"],
                },
            )

        # Slim outcome record
        out_row = {
            "sample_id": sid,
            "plan_provenance": plan,
            "oracle_route": o.get("oracle_route"),
            "source_kind": o.get("source_kind"),
            "source_episode": ep,
            "source_pool": o["_pool"],
            "genuine_stage1_plan_b": o.get("genuine_stage1_plan_b"),
            "final_category": final_cat,
            "sft_category_raw": o.get("sft_category"),
            "split": split,
            "capture_fp": o.get("capture_fp"),
            "ic_pkl": f"ics/{sid}.pkl" if ic_src else None,
            "continuous_npz": str(cont_dst.relative_to(OUT)) if cont_dst else None,
            "tokens_npz": str(tok_dst.relative_to(OUT)) if tok_dst else None,
            "errors_entry": o.get("errors_entry"),
            "direct": o.get("direct"),
            "precision": o.get("precision"),
            "continuous_demo": o.get("continuous_demo"),
            "k49_demo": o.get("k49_demo"),
            "handoff_token": HANDOFF,
            "k49_config": K49_CONFIG,
            "source_outcome": o.get("_outcome_path"),
        }
        atomic_json(OUT / "outcomes" / f"{sid}.json", out_row)
        append_jsonl(manifest_path, out_row)
        if ic_src and (
            final_cat == "ORACLE_FAILURE"
            or final_cat == "DIRECT_HANDOFF"
            or (cont_dst is not None or final_cat == "DIRECT_HANDOFF")
        ):
            integrity["ok"] += 1

    # Splits
    split_eps = defaultdict(set)
    for ep, sp in ep_to_split.items():
        split_eps[sp].add(ep)
    overlaps = {
        "train_val": sorted(split_eps["train"] & split_eps["val"]),
        "train_holdout": sorted(split_eps["train"] & split_eps["holdout"]),
        "val_holdout": sorted(split_eps["val"] & split_eps["holdout"]),
    }
    atomic_json(
        OUT / "splits.json",
        {
            "by_episode": ep_to_split,
            "counts": {k: len(v) for k, v in split_eps.items()},
            "overlaps": overlaps,
            "disjoint": all(len(v) == 0 for v in overlaps.values()),
        },
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "by_final_category": dict(counts),
        "by_plan_category": {p: dict(c) for p, c in plan_counts.items()},
        "genuine_abc": {f"{a}/{b}": n for (a, b), n in genuine_abc.items()},
        "hard_ic_separate": dict(hard_ic),
        "integrity": {
            "ok_rows": integrity["ok"],
            "missing_ic_n": len(integrity["missing_ic"]),
            "missing_continuous_n": len(integrity["missing_continuous"]),
            "missing_tokens_n": len(integrity["missing_tokens"]),
            "missing_ic": integrity["missing_ic"][:20],
            "missing_continuous": integrity["missing_continuous"][:20],
            "missing_tokens": integrity["missing_tokens"][:20],
        },
        "splits_disjoint": all(len(v) == 0 for v in overlaps.values()),
        "k49_config": K49_CONFIG,
        "handoff_token": HANDOFF,
    }
    atomic_json(OUT / "SUMMARY.json", summary)
    atomic_json(OUT / "integrity_report.json", summary["integrity"])
    return summary


def check_integrity() -> dict:
    """Loadable integrity checker for the final dataset."""
    man = OUT / "manifest.jsonl"
    assert man.exists(), "manifest missing"
    rows = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
    issues = []
    for r in rows:
        sid = r["sample_id"]
        if r.get("ic_pkl"):
            if not (OUT / r["ic_pkl"]).exists():
                issues.append(f"missing_ic:{sid}")
        if r["final_category"] == "NONTRIVIAL_K49_SUCCESS":
            if not r.get("tokens_npz") or not (OUT / r["tokens_npz"]).exists():
                issues.append(f"missing_tokens:{sid}")
            if not r.get("continuous_npz") or not (OUT / r["continuous_npz"]).exists():
                issues.append(f"missing_continuous:{sid}")
            k49 = r.get("k49_demo") or {}
            if not (k49.get("nontrivial_representability") or k49.get("verified")):
                issues.append(f"k49_label_mismatch:{sid}")
        if r["final_category"] == "CONTINUOUS_ONLY_SUCCESS":
            # Must not be counted as K49
            k49 = r.get("k49_demo") or {}
            if k49.get("nontrivial_representability"):
                issues.append(f"continuous_only_but_k49_flag:{sid}")
        if r.get("plan_provenance") == "HARD_IC" and r.get("genuine_stage1_plan_b"):
            issues.append(f"hard_ic_marked_genuine_b:{sid}")
    splits = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    report = {
        "n_rows": len(rows),
        "n_issues": len(issues),
        "issues_head": issues[:50],
        "splits_disjoint": splits.get("disjoint"),
        "ok": len(issues) == 0 and bool(splits.get("disjoint")),
    }
    atomic_json(OUT / "integrity_check.json", report)
    return report


if __name__ == "__main__":
    s = finalize()
    c = check_integrity()
    print(json.dumps({"summary": s, "check": c}, indent=2))
