"""Phase 2 — Select 16 C train ICs + held-aside C/ABD validation ICs (train pool only)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p58_fast_c_grpo.paths import (
    OUT,
    P56_C_AUDIT,
    P57A_PROGRESS,
    SEED,
    VAL_ABD_PER,
    VAL_C_N,
)
from experiments.precision.p58_fast_c_grpo.stage_audit import run_audit


def _fam(tid: str) -> str:
    t = tid.lower()
    if "__gbi_" in t or t.startswith("c__gbi") or "gbi_" in t:
        # cluster by first gbi id digit groups
        return "gbi"
    return "plan_c" if "plan_c" in t else "other"


def select() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = run_audit()
    per = {r["ic_id"]: r for r in audit["per_ic"]}
    rows = [r for r in csv.DictReader(P57A_PROGRESS.open(encoding="utf-8")) if r["class"] == "C"]
    by_id = {r["ic_id"]: r for r in rows}

    p56_partial = set()
    if P56_C_AUDIT.is_file():
        a = json.loads(P56_C_AUDIT.read_text(encoding="utf-8"))
        for r in a.get("per_ic") or []:
            if r.get("useful_partial_despite_zero"):
                p56_partial.add(r["ic_id"])

    hard_cand = []
    for tid, r in by_id.items():
        s = int(r["success_relaxed_count"])
        if s != 0:
            continue
        rc = int(r["recatch_count"])
        st = int(r["stable_hold_count"])
        score = rc + 2 * st + (5 if tid in p56_partial else 0)
        hard_cand.append((score, tid))
    hard_cand.sort(reverse=True)
    # diversify families
    hard = []
    fam_count: dict[str, int] = defaultdict(int)
    for score, tid in hard_cand:
        f = _fam(tid)
        if fam_count[f] >= 4 and len(hard) < 6:
            continue
        hard.append(tid)
        fam_count[f] += 1
        if len(hard) >= 6:
            break
    while len(hard) < 6 and hard_cand:
        for _, tid in hard_cand:
            if tid not in hard:
                hard.append(tid)
            if len(hard) >= 6:
                break

    mid_cand = []
    for tid, r in by_id.items():
        if tid in hard:
            continue
        s = int(r["success_relaxed_count"])
        if 1 <= s <= 10:
            mid_cand.append((s, int(r["recatch_count"]), tid))
    mid_cand.sort()  # prefer lower occasional success first, then diversify
    mid = []
    fam_count = defaultdict(int)
    for s, rc, tid in mid_cand:
        f = _fam(tid)
        if fam_count[f] >= 4:
            continue
        mid.append(tid)
        fam_count[f] += 1
        if len(mid) >= 6:
            break
    for _, _, tid in mid_cand:
        if len(mid) >= 6:
            break
        if tid not in mid and tid not in hard:
            mid.append(tid)

    easy_cand = sorted(
        (
            (int(r["success_relaxed_count"]), tid)
            for tid, r in by_id.items()
            if tid not in hard and tid not in mid
        ),
        reverse=True,
    )
    easy = []
    fam_count = defaultdict(int)
    for s, tid in easy_cand:
        f = _fam(tid)
        if fam_count[f] >= 3 and len(easy) < 4:
            continue
        easy.append(tid)
        fam_count[f] += 1
        if len(easy) >= 4:
            break
    for s, tid in easy_cand:
        if len(easy) >= 4:
            break
        if tid not in easy:
            easy.append(tid)

    train_ids = hard + mid + easy
    # Pad from remaining by ascending success if short
    if len(train_ids) < 16:
        rest = sorted(
            ((int(r["success_relaxed_count"]), tid) for tid, r in by_id.items() if tid not in train_ids)
        )
        for _, tid in rest:
            train_ids.append(tid)
            if len(train_ids) >= 16:
                break
    train_ids = train_ids[:16]
    assert len(train_ids) == 16, len(train_ids)
    # Rebuild buckets for reporting
    hard = train_ids[:6]
    mid = train_ids[6:12]
    easy = train_ids[12:16]
    train_set = set(train_ids)

    # Validation C: 8 from remaining, mix of zero / mid / higher
    remain = [r for r in rows if r["ic_id"] not in train_set]
    remain_sorted = sorted(remain, key=lambda r: int(r["success_relaxed_count"]))
    val_c = []
    # 3 hard-ish, 3 mid, 2 easier from remainder
    val_c.extend([r["ic_id"] for r in remain_sorted[:3]])
    mid_r = remain_sorted[len(remain_sorted) // 3 : 2 * len(remain_sorted) // 3]
    val_c.extend([r["ic_id"] for r in mid_r[:3]])
    val_c.extend([r["ic_id"] for r in remain_sorted[-2:]])
    # unique pad
    seen = set()
    val_c_u = []
    for t in val_c:
        if t not in seen:
            seen.add(t)
            val_c_u.append(t)
    for r in remain_sorted:
        if len(val_c_u) >= VAL_C_N:
            break
        if r["ic_id"] not in seen:
            seen.add(r["ic_id"])
            val_c_u.append(r["ic_id"])
    val_c_ids = val_c_u[:VAL_C_N]

    ds = build_p54_dataset(seed=SEED)
    lookup = {(s.skill, s.traj_id): s for s in ds["schedule"]}
    # ABD val: 2 per class from schedule, not C-focused
    rng = np.random.default_rng(SEED + 58)
    val_abd = {}
    for sk, lab in (("A1", "A"), ("B1", "B"), ("D", "D")):
        pool = [s for s in ds["schedule"] if s.skill == sk]
        rng.shuffle(pool)
        val_abd[lab] = [s.traj_id for s in pool[:VAL_ABD_PER]]

    selected = {
        "seed": SEED,
        "train_c": {
            "hard_zero_partial": hard,
            "intermediate": mid,
            "successful_preserve": easy,
            "all": train_ids,
        },
        "val_c": val_c_ids,
        "val_abd": val_abd,
        "per_ic_stats": {
            tid: {
                "bucket": (
                    "hard"
                    if tid in hard
                    else "intermediate"
                    if tid in mid
                    else "successful"
                    if tid in easy
                    else "val"
                ),
                "family": _fam(tid),
                **{k: per[tid][k] for k in ("p_succ", "p_recatch", "p_stable", "p_handoff") if tid in per},
            }
            for tid in train_ids + val_c_ids
        },
        "notes": "All ICs from P54 train schedule; no held-out final eval ICs used for selection.",
    }
    (OUT / "SELECTED_C_ICS.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected


if __name__ == "__main__":
    s = select()
    print(json.dumps({k: s[k] for k in ("train_c", "val_c", "val_abd")}, indent=2))
