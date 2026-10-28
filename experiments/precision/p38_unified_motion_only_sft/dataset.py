"""Gate 2 — Motion-only unified A/B/C/D dataset (HANDOFF stripped from targets)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.precision.p27_history_multiskill_transformer.run_p27 import load_npz_flex
from experiments.precision.p38_unified_motion_only_sft.paths import (
    OUT,
    P21,
    P21_MAN,
    P23,
    P23_MAN,
    P25,
    P35,
    P35_MAN,
    ROOT,
)
from models.m111_checkpoint import HANDOFF, MOTION_K
from training.m111_dataset import WINDOW, chunk_episode

GATE2 = OUT / "gate2_dataset"


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


def motion_episode_from_npz(z: dict) -> dict | None:
    """Strip terminal HANDOFF; mask any remaining HANDOFF from loss."""
    toks = np.asarray(z["tokens"], np.int64)
    if len(toks) and int(toks[-1]) == HANDOFF:
        n = len(toks) - 1
    else:
        n = len(toks)
    if n <= 0:
        return None
    obs = np.asarray(z["observations"][:n], np.float32)
    prev = np.asarray(z["prev_actions"][:n], np.float32)
    tokens = toks[:n].astype(np.int64)
    resid = np.asarray(z["residuals"][:n], np.float32)
    mask = np.ones(n, dtype=np.bool_) & (tokens != HANDOFF)
    # also drop illegal tokens if any
    mask &= (tokens >= 0) & (tokens < MOTION_K)
    if not bool(mask.any()):
        return None
    return {
        "observations": obs,
        "prev_actions": prev,
        "tokens": tokens,
        "residuals": resid,
        "loss_mask": mask,
    }


def _chunk_rows(ep: dict, meta: dict) -> list[dict]:
    out = []
    for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
        mask = np.asarray(ch["loss_mask"], dtype=np.bool_) & (
            np.asarray(ch["tokens"]) != HANDOFF
        )
        mask &= (np.asarray(ch["tokens"]) >= 0) & (np.asarray(ch["tokens"]) < MOTION_K)
        if not bool(mask.any()):
            continue
        out.append(
            {
                **ch,
                "loss_mask": mask,
                **meta,
                "chunk_index": ci,
                "n_tokens": int(mask.sum()),
                "sample_weight": 1.0,
            }
        )
    return out


def build_skill_pool(skill: str, split: str) -> tuple[list[dict], dict]:
    """Return (windowed samples, census dict)."""
    samples: list[dict] = []
    traj_ids: list[str] = []
    tok_hist: Counter = Counter()
    n_motion_total = 0

    if skill in ("A1", "B1", "C"):
        fam = {"A1": "A", "B1": "B", "C": "C"}[skill]
        man = json.loads(P21_MAN.read_text(encoding="utf-8"))
        rows = [r for r in man["trajectories"] if r["family"] == fam and r["split"] == split]
        for r in rows:
            z = load_npz_flex(P21, r["npz_relpath"])
            ep = motion_episode_from_npz(z)
            if ep is None:
                continue
            traj_ids.append(r["traj_id"])
            n_motion_total += int(ep["loss_mask"].sum())
            for t in ep["tokens"][ep["loss_mask"]].tolist():
                tok_hist[int(t)] += 1
            samples.extend(
                _chunk_rows(
                    ep,
                    {
                        "traj_id": r["traj_id"],
                        "skill": skill,
                        "skill_group": skill,
                        "family": fam,
                        "lineage_id": r.get("lineage_id"),
                        "split": split,
                        "source": "p21",
                    },
                )
            )
        # Optional C dagger corrections on train only (same as P31)
        if skill == "C" and split == "train":
            p25p = P25 / "VERIFIED_CORRECTIONS_MANIFEST.json"
            if p25p.is_file():
                corr = json.loads(p25p.read_text(encoding="utf-8")).get("corrections") or []
                for c in corr:
                    if c.get("split") != "train":
                        continue
                    try:
                        ep_raw = load_npz_flex(
                            P25, c["npz_relpath"], allow_nonzero_prev0=True
                        )
                    except Exception:
                        continue
                    ep = motion_episode_from_npz(ep_raw)
                    if ep is None:
                        continue
                    tid = str(c.get("traj_id") or c.get("lineage_id"))
                    traj_ids.append(tid)
                    n_motion_total += int(ep["loss_mask"].sum())
                    for t in ep["tokens"][ep["loss_mask"]].tolist():
                        tok_hist[int(t)] += 1
                    samples.extend(
                        _chunk_rows(
                            ep,
                            {
                                "traj_id": tid,
                                "skill": "C",
                                "skill_group": "C",
                                "family": "C",
                                "lineage_id": c.get("lineage_id"),
                                "split": "train",
                                "source": "p25_corr",
                            },
                        )
                    )
    elif skill == "D":
        rows = [
            r
            for r in csv.DictReader(P35_MAN.open(encoding="utf-8"))
            if r.get("class") == "P1" and r.get("split") == split
        ]
        for r in rows:
            z = dict(np.load(P35 / r["npz_relpath"], allow_pickle=False))
            # np.load returns NpzFile-like; convert arrays
            z = {k: z[k] for k in z}
            ep = motion_episode_from_npz(z)
            if ep is None:
                continue
            traj_ids.append(r["traj_id"])
            n_motion_total += int(ep["loss_mask"].sum())
            for t in ep["tokens"][ep["loss_mask"]].tolist():
                tok_hist[int(t)] += 1
            samples.extend(
                _chunk_rows(
                    ep,
                    {
                        "traj_id": r["traj_id"],
                        "skill": "D",
                        "skill_group": "D",
                        "family": "D",
                        "lineage_id": r.get("lineage_id"),
                        "split": split,
                        "source": "p35_p1",
                    },
                )
            )
    else:
        raise ValueError(skill)

    census = {
        "skill": skill,
        "split": split,
        "n_independent_roots": len(set(traj_ids)),
        "n_trajectories": len(traj_ids),
        "n_chunks": len(samples),
        "n_motion_tokens": n_motion_total,
        "top_tokens": tok_hist.most_common(12),
        "n_unique_motion_tokens": len(tok_hist),
    }
    return samples, census


def build_all_pools() -> dict[str, Any]:
    GATE2.mkdir(parents=True, exist_ok=True)
    skills = ("A1", "B1", "C", "D")
    train: dict[str, list] = {}
    val: dict[str, list] = {}
    census_rows = []
    for sk in skills:
        tr, c_tr = build_skill_pool(sk, "train")
        va, c_va = build_skill_pool(sk, "val")
        # D also has test split — keep val for selection; do not put test in train
        train[sk] = tr
        val[sk] = va
        census_rows.append(c_tr)
        census_rows.append(c_va)
        print(
            f"  {sk} train: roots={c_tr['n_independent_roots']} chunks={c_tr['n_chunks']} "
            f"mot_tok={c_tr['n_motion_tokens']} | val: roots={c_va['n_independent_roots']} "
            f"chunks={c_va['n_chunks']}",
            flush=True,
        )
    write_csv(GATE2 / "DATASET_CENSUS.csv", census_rows)
    # Leakage check: traj_id overlap train/val
    leaks = []
    for sk in skills:
        tr_ids = {s["traj_id"] for s in train[sk]}
        va_ids = {s["traj_id"] for s in val[sk]}
        inter = tr_ids & va_ids
        if inter:
            leaks.append({"skill": sk, "n_overlap": len(inter), "examples": list(inter)[:5]})
    write_csv(GATE2 / "SPLIT_LEAKAGE.csv", leaks)
    meta = {
        "skills": list(skills),
        "window": WINDOW,
        "handoff_stripped": True,
        "motion_k": MOTION_K,
        "logit_k_retained": 112,
        "note": "112-way head retained; HANDOFF never in loss_mask; masked at inference",
        "leaks": leaks,
        "census": census_rows,
    }
    (GATE2 / "DATASET_META.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return {"train": train, "val": val, "census": census_rows, "leaks": leaks}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_all_pools()
