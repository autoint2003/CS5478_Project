"""P46 — Pack non-holdout Tree80 B M111 experts into SFT chunks; merge with P45-B pools."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p46_b_tree80_expansion.paths import (  # noqa: E402
    M111_REP,
    OUT,
    P44_EVAL,
    P45,
    SEED,
    TREE80_DS,
)
from models.m111_checkpoint import HANDOFF, MOTION_K  # noqa: E402
from training.m111_dataset import WINDOW, chunk_episode  # noqa: E402

SKILLS = ("A1", "B1", "C", "D")
B_WEIGHT = 3.0
EARLY_WEIGHT = 2.0


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
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def root_id(tid: str) -> str:
    t = re.sub(r"^(A|B|C)__", "", tid)
    t = re.sub(r"__([0-9a-f]{12,})$", "", t)
    for p in ("plan_a__", "plan_b__", "plan_c__"):
        if t.startswith(p):
            t = t[len(p) :]
            break
    return t


def holdout_b_roots() -> set[str]:
    rows = csv.DictReader(P44_EVAL.open(encoding="utf-8"))
    return {
        root_id(r["traj_id"])
        for r in rows
        if r.get("skill") == "B1" and r.get("tag") == "holdout"
    }


def load_m111_pass() -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(M111_REP.open(encoding="utf-8")):
        ok = str(r.get("M111_12S", "")).lower() in ("1", "true")
        out[r["sample_id"]] = {**r, "pass": ok}
    return out


def build_episode(obs: np.ndarray, tokens: np.ndarray, residuals: np.ndarray, recon: np.ndarray):
    n = len(tokens)
    if len(obs) != n or len(residuals) != n or len(recon) != n:
        raise ValueError(f"length mismatch obs={len(obs)} tok={n} res={len(residuals)} recon={len(recon)}")
    # Drop terminal HANDOFF if present (motion-only SFT like P44/P45)
    if n and int(tokens[-1]) == HANDOFF:
        n -= 1
        obs, tokens, residuals, recon = obs[:n], tokens[:n], residuals[:n], recon[:n]
    if n == 0:
        return None
    prev = np.zeros((n, 7), np.float32)
    if n > 1:
        prev[1:] = recon[: n - 1]
    mask = (tokens != HANDOFF) & (tokens >= 0) & (tokens < MOTION_K)
    if not bool(mask.any()):
        return None
    return {
        "observations": np.asarray(obs, np.float32),
        "prev_actions": prev,
        "tokens": np.asarray(tokens, np.int64),
        "residuals": np.asarray(residuals, np.float32),
        "loss_mask": np.asarray(mask, np.bool_),
    }


def pack_tree80_b(*, include_kinds: tuple[str, ...] = ("tree80_direct", "tree80_precision")) -> tuple[list, list]:
    held = holdout_b_roots()
    m111_pass = load_m111_pass()
    man = [json.loads(l) for l in (TREE80_DS / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks: list[dict] = []
    rows: list[dict] = []
    for r in man:
        sid = r["sample_id"]
        kind = r["expert_kind"]
        ic = r["ic_id"]
        row = {
            "sample_id": sid,
            "ic_id": ic,
            "expert_kind": kind,
            "category": r.get("category"),
            "admitted": 0,
        }
        if kind not in include_kinds:
            row["reject"] = "kind_excluded"
            rows.append(row)
            continue
        if root_id(ic) in held or ic in held:
            row["reject"] = "holdout_overlap"
            rows.append(row)
            continue
        meta = m111_pass.get(sid)
        if meta is None or not meta.get("pass"):
            row["reject"] = "m111_12S_fail"
            rows.append(row)
            continue
        cont = np.load(TREE80_DS / r["continuous_demo"]["path"])
        tokz = np.load(TREE80_DS / "tokens_m111" / f"{sid}__m111.npz")
        ep = build_episode(
            np.asarray(cont["observations"], np.float32),
            np.asarray(tokz["tokens"], np.int64),
            np.asarray(tokz["residuals"], np.float32),
            np.asarray(tokz["recon_actions"], np.float32),
        )
        if ep is None:
            row["reject"] = "empty_motion"
            rows.append(row)
            continue
        n_ch = 0
        for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
            mask = np.asarray(ch["loss_mask"], dtype=np.bool_)
            mask &= np.asarray(ch["tokens"]) != HANDOFF
            mask &= (np.asarray(ch["tokens"]) >= 0) & (np.asarray(ch["tokens"]) < MOTION_K)
            if not bool(mask.any()):
                continue
            w = B_WEIGHT * (EARLY_WEIGHT if ci == 0 else 1.0)
            chunks.append(
                {
                    **ch,
                    "loss_mask": mask,
                    "traj_id": f"plan_b__tree80__{ic}",
                    "skill": "B1",
                    "split": "train",
                    "chunk_index": ci,
                    "n_tokens": int(mask.sum()),
                    "sample_weight": float(w),
                    "source": "p46_tree80_b",
                    "expert_kind": kind,
                    "ic_id": ic,
                    "SUCCESS_12S": 1,
                    "SUCCESS_NEW": int(str(meta.get("M111_NEW", "True")).lower() in ("1", "true")),
                }
            )
            n_ch += 1
        row.update({"admitted": 1, "n_chunks": n_ch, "n_steps": int(ep["observations"].shape[0])})
        rows.append(row)
        print(f"  ADMIT {kind} {ic} steps={row['n_steps']} chunks={n_ch}", flush=True)
    return chunks, rows


def merge_pools(tree_b: list[dict]) -> dict[str, list]:
    """P45-B A/C/D + (P45-B B ∪ Tree80 B)."""
    pools = {}
    for sk in SKILLS:
        base = pickle.loads((P45 / "dataset" / f"chunks_{sk}_P45B.pkl").read_bytes())
        pools[sk] = list(base)
    pools["B1"] = list(pools["B1"]) + list(tree_b)
    return pools


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset").mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("Packing Tree80 B (non-holdout, M111_12S)…", flush=True)
    tree_b, rows = pack_tree80_b()
    write_csv(OUT / "TREE80_B_ADMIT.csv", rows)
    (OUT / "dataset" / "chunks_B1_tree80_only.pkl").write_bytes(pickle.dumps(tree_b))

    pools = merge_pools(tree_b)
    for sk, chs in pools.items():
        (OUT / "dataset" / f"chunks_{sk}_P46.pkl").write_bytes(pickle.dumps(chs))
        print(f"pool {sk}: {len(chs)}", flush=True)

    admitted = [r for r in rows if int(r.get("admitted") or 0) == 1]
    summary = {
        "generated": utc(),
        "n_admitted": len(admitted),
        "by_kind": dict(Counter(r["expert_kind"] for r in admitted)),
        "n_tree80_chunks": len(tree_b),
        "n_tree80_steps": int(sum(int(r.get("n_steps") or 0) for r in admitted)),
        "pool_sizes": {sk: len(pools[sk]) for sk in SKILLS},
        "p45b_B1": len(pickle.loads((P45 / "dataset" / "chunks_B1_P45B.pkl").read_bytes())),
        "holdout_excluded": sum(1 for r in rows if r.get("reject") == "holdout_overlap"),
        "m111_fail_excluded": sum(1 for r in rows if r.get("reject") == "m111_12S_fail"),
        "note": "Handcrafted B (=P44 holdout) excluded; Tree80 Direct/Precision with M111_12S only.",
    }
    (OUT / "DATA_PACK_REPORT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# DATA_PACK_REPORT — P46 Tree80 B expansion",
        "",
        f"Generated: {summary['generated']}",
        "",
        f"- Admitted Tree80 B episodes: **{summary['n_admitted']}** "
        f"({summary['by_kind']})",
        f"- Tree80 chunks: {summary['n_tree80_chunks']} (steps={summary['n_tree80_steps']})",
        f"- Holdout handcrafted excluded: {summary['holdout_excluded']}",
        f"- M111_12S fail excluded: {summary['m111_fail_excluded']}",
        f"- Merged pools: `{summary['pool_sizes']}` (P45-B B1 was {summary['p45b_B1']})",
        "",
        "Source: `stage2_b_grpo_multistage_v1` (frozen Tree80 E3 rollout + Precision). "
        "No K48 GRPO retrain.",
        "",
    ]
    (OUT / "DATA_PACK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    argparse.ArgumentParser().parse_args()
    run()


if __name__ == "__main__":
    main()
