"""Construct rebalanced P54 train IC set (A↑ B↓ C/D keep)."""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from envs.simulation.capture import restore_capture
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import _is_held, holdout_ids, root_id
from experiments.precision.p53_independent_grpo64.full_dataset import build_uncapped_pools
from experiments.precision.p54_rebalanced_grpo64.paths import (
    OUT,
    P53_MANIFEST,
    ROOT,
    SEED,
    SKILLS,
    TARGET_A,
    TARGET_B,
)

CLASS_LABEL = {"A1": "A", "B1": "B", "C": "C", "D": "D"}


def _fam_from_id(tid: str) -> str:
    for k in ("ball", "slide", "inward", "k48r"):
        if k in tid:
            return k
    return "other"


def _normalize_root(tid: str) -> str:
    t = re.sub(r"^(plan_[abc]__|A__|B__|C__)", "", tid)
    t = re.sub(r"__([0-9a-f]{8,})$", "", t)
    return t


def load_p53_schedule() -> dict[str, list[dict]]:
    man = json.loads(P53_MANIFEST.read_text(encoding="utf-8"))
    by: dict[str, list[dict]] = {s: [] for s in SKILLS}
    for r in man["train_ics"]:
        by[r["skill"]].append(r)
    return by


def _validate_start(start: MixedStart) -> tuple[bool, str]:
    try:
        env = make_env()
        if start.kind == "ep":
            assert start.ep is not None
            restore_start(env, start.ep)
        else:
            assert start.capture is not None
            restore_capture(env, start.capture)
        # one physics step probe
        z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        if not np.isfinite(z):
            return False, "nonfinite_obj_z"
        return True, "ok"
    except Exception as e:
        return False, f"restore_fail:{type(e).__name__}:{e}"


def collect_a_candidates(*, held: set[str]) -> list[dict[str, Any]]:
    """A recovery starts: P21 train eps + plan_a source snapshots not yet covered."""
    out: list[dict[str, Any]] = []
    seen_roots: set[str] = set()

    # 1) P21 train A episodes (non-holdout)
    for j in collect_jobs():
        if j["skill"] != "A1" or j.get("split") != "train":
            continue
        if _is_held(j["traj_id"], held):
            continue
        root = _normalize_root(j["traj_id"])
        if root in seen_roots:
            continue
        seen_roots.add(root)
        out.append(
            {
                "traj_id": j["traj_id"],
                "root": root,
                "family": _fam_from_id(j["traj_id"]),
                "kind": "ep",
                "ep": j["ep"],
                "source": "p21_train",
                "start": MixedStart(
                    skill="A1",
                    traj_id=j["traj_id"],
                    family="A",
                    kind="ep",
                    ep=j["ep"],
                ),
            }
        )

    # 2) plan_a stage1 terminal → source_snapshot recovery IC
    pa = ROOT / "results/data_generation/plan_a_stage1_terminals_v1/ics"
    if pa.is_dir():
        for p in sorted(pa.glob("*.pkl")):
            blob = pickle.loads(p.read_bytes())
            root = str(blob.get("physical_root_id") or "")
            eid = str(blob.get("unified_expert_id") or f"plan_a__{root}")
            snap = blob.get("source_snapshot")
            if not root or root in seen_roots:
                continue
            if _is_held(eid, held) or _is_held(root, held) or _is_held(str(snap or ""), held):
                continue
            sp = ROOT / str(snap) if snap else None
            if sp is None or not sp.is_file():
                alt = (
                    ROOT
                    / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
                    / f"{root}.pkl"
                )
                sp = alt if alt.is_file() else None
            if sp is None or not sp.is_file():
                continue
            ic = pickle.loads(sp.read_bytes())
            if "capture" not in ic or "ref" not in ic:
                continue
            seen_roots.add(root)
            tid = eid if eid.startswith("plan_a__") else f"plan_a__{root}"
            out.append(
                {
                    "traj_id": tid,
                    "root": root,
                    "family": _fam_from_id(root),
                    "kind": "capture",
                    "path": str(sp),
                    "source": "plan_a_source_snapshot",
                    "start": MixedStart(
                        skill="A1",
                        traj_id=tid,
                        family=_fam_from_id(root),
                        kind="capture",
                        capture=ic["capture"],
                        ref=ic["ref"],
                    ),
                }
            )

    # 3) unified A train snapshots
    uni = ROOT / "data/unified_experts/manifest.jsonl"
    if uni.is_file():
        for line in uni.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("plan_family") != "A" or r.get("split") != "train":
                continue
            eid = str(r["expert_id"])
            root = str(r.get("physical_root_id") or _normalize_root(eid))
            if root in seen_roots:
                continue
            if _is_held(eid, held) or _is_held(root, held):
                continue
            ref = r.get("initial_physical_state_ref") or {}
            snap = ref.get("snapshot")
            sp = ROOT / str(snap) if snap else None
            if sp is None or not sp.is_file():
                continue
            ic = pickle.loads(sp.read_bytes())
            if "capture" not in ic or "ref" not in ic:
                continue
            seen_roots.add(root)
            out.append(
                {
                    "traj_id": eid,
                    "root": root,
                    "family": _fam_from_id(root),
                    "kind": "capture",
                    "path": str(sp),
                    "source": "unified_a_train",
                    "start": MixedStart(
                        skill="A1",
                        traj_id=eid,
                        family=_fam_from_id(root),
                        kind="capture",
                        capture=ic["capture"],
                        ref=ic["ref"],
                    ),
                }
            )
    return out


def select_a(target: int = TARGET_A, *, seed: int = SEED) -> tuple[list[MixedStart], dict]:
    held = holdout_ids()["A1"]
    cands = collect_a_candidates(held=held)
    rng = np.random.default_rng(seed)
    # diversify: shuffle within family then round-robin
    by_fam: dict[str, list] = defaultdict(list)
    for c in cands:
        by_fam[c["family"]].append(c)
    for fam in by_fam:
        rng.shuffle(by_fam[fam])
    order = []
    fams = sorted(by_fam.keys())
    while len(order) < len(cands):
        progressed = False
        for fam in fams:
            if by_fam[fam]:
                order.append(by_fam[fam].pop())
                progressed = True
        if not progressed:
            break

    selected: list[MixedStart] = []
    meta_rows = []
    rejected = []
    for c in order:
        if len(selected) >= target:
            break
        ok, reason = _validate_start(c["start"])
        row = {
            "traj_id": c["traj_id"],
            "root": c["root"],
            "family": c["family"],
            "kind": c["kind"],
            "source": c["source"],
            "valid": ok,
            "reason": reason,
        }
        if ok:
            selected.append(c["start"])
            meta_rows.append(row)
        else:
            rejected.append(row)

    audit = {
        "target": target,
        "candidates": len(cands),
        "selected": len(selected),
        "rejected": rejected,
        "family_selected": dict(Counter(m["family"] for m in meta_rows)),
        "source_selected": dict(Counter(m["source"] for m in meta_rows)),
        "limitation": None
        if len(selected) >= target
        else (
            f"Only {len(selected)} valid distinct non-holdout A recovery ICs available "
            f"(target {target}). Did not invent/relabel non-A ICs. "
            f"Candidates discovered={len(cands)}; rejected={len(rejected)}."
        ),
        "rows": meta_rows,
    }
    return selected, audit


def select_b(
    p53_b_starts: list[MixedStart],
    target: int = TARGET_B,
    *,
    seed: int = SEED,
) -> tuple[list[MixedStart], dict, list[dict]]:
    """Stratified subsample of P53 B pool; preserve full B manifest."""
    rng = np.random.default_rng(seed + 17)
    full_manifest = [
        {
            "traj_id": s.traj_id,
            "family": s.family,
            "kind": s.kind,
            "skill": "B1",
        }
        for s in p53_b_starts
    ]
    by_fam: dict[str, list[MixedStart]] = defaultdict(list)
    for s in p53_b_starts:
        by_fam[str(s.family)].append(s)
    for fam in by_fam:
        rng.shuffle(by_fam[fam])

    # Keep all minority families first, fill remainder from ball
    selected: list[MixedStart] = []
    rationale = []
    minorities = ["inward", "slide", "k48r"]
    for fam in minorities:
        pool = by_fam.get(fam) or []
        take = list(pool)  # keep all
        selected.extend(take)
        rationale.append({"family": fam, "available": len(pool), "taken": len(take), "policy": "keep_all"})
    remain = max(0, target - len(selected))
    ball = by_fam.get("ball") or []
    # Prefer mixing: use P53 progress difficulty when available
    hard_ids = set()
    prog = OUT.parent / "p53_independent_grpo64/FULL_EPOCH_PROGRESS.csv"
    # path fix
    prog = ROOT / "results/experiments/p53_independent_grpo64/FULL_EPOCH_PROGRESS.csv"
    if prog.is_file():
        import csv

        for r in csv.DictReader(prog.open(encoding="utf-8")):
            if r.get("skill") != "B1":
                continue
            if int(r.get("success_relaxed_count") or 0) == 0:
                hard_ids.add(r["ic_id"])
    hard = [s for s in ball if s.traj_id in hard_ids]
    easy = [s for s in ball if s.traj_id not in hard_ids]
    rng.shuffle(hard)
    rng.shuffle(easy)
    # ~40% hard among ball slots when possible
    n_hard = min(len(hard), max(1, int(0.4 * remain)))
    take_ball = hard[:n_hard] + easy[: max(0, remain - n_hard)]
    if len(take_ball) < remain:
        leftover = [s for s in ball if s not in take_ball]
        take_ball.extend(leftover[: remain - len(take_ball)])
    selected.extend(take_ball[:remain])
    rationale.append(
        {
            "family": "ball",
            "available": len(ball),
            "taken": len(take_ball[:remain]),
            "n_hard": n_hard,
            "policy": "stratified_hard40_then_easy",
        }
    )
    # trim if over
    selected = selected[:target]
    audit = {
        "target": target,
        "available_p53": len(p53_b_starts),
        "selected": len(selected),
        "family_selected": dict(Counter(str(s.family) for s in selected)),
        "rationale": rationale,
        "seed": seed + 17,
    }
    return selected, audit, full_manifest


def build_p54_dataset(*, seed: int = SEED) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    # Rebuild P53-equivalent uncapped pools for MixedStart objects
    pools = build_uncapped_pools(per_class_val=0, seed=seed)  # no carve — we control counts
    # Actually per_class_val=0 might break; use val=0 carefully
    # Re-get with val=0: looking at build_pools, n_v = min(0, ...) = 0, train = all. Good.

    p53_by = load_p53_schedule()
    # Map traj_id -> MixedStart from uncapped train pools
    lookup: dict[tuple[str, str], MixedStart] = {}
    for sk in SKILLS:
        for s in pools["train"].get(sk) or []:
            lookup[(sk, s.traj_id)] = s

    # C/D: exact P53 train lists
    starts_c: list[MixedStart] = []
    for r in p53_by["C"]:
        s = lookup.get(("C", r["traj_id"]))
        if s is None:
            raise RuntimeError(f"missing C IC {r['traj_id']}")
        starts_c.append(s)
    starts_d: list[MixedStart] = []
    for r in p53_by["D"]:
        s = lookup.get(("D", r["traj_id"]))
        if s is None:
            raise RuntimeError(f"missing D IC {r['traj_id']}")
        starts_d.append(s)

    # B from P53 list
    p53_b_starts = []
    for r in p53_by["B1"]:
        s = lookup.get(("B1", r["traj_id"]))
        if s is None:
            raise RuntimeError(f"missing B IC {r['traj_id']}")
        p53_b_starts.append(s)
    starts_b, b_audit, b_full = select_b(p53_b_starts, TARGET_B, seed=seed)

    # A expansion
    starts_a, a_audit = select_a(TARGET_A, seed=seed)

    schedule = starts_a + starts_b + starts_c + starts_d
    rng = np.random.default_rng(seed)
    order = np.arange(len(schedule))
    rng.shuffle(order)
    schedule = [schedule[int(i)] for i in order]

    counts = {
        "A1": len(starts_a),
        "B1": len(starts_b),
        "C": len(starts_c),
        "D": len(starts_d),
    }
    manifest = {
        "seed": seed,
        "group_size": 64,
        "n_train_ics": len(schedule),
        "n_train_per_class": counts,
        "n_expected_groups": len(schedule),
        "n_expected_rollouts": len(schedule) * 64,
        "targets": {"A1": TARGET_A, "B1": TARGET_B, "C": 68, "D": 93},
        "a_expansion": a_audit,
        "b_reduction": b_audit,
        "train_ics": [
            {
                "index": i,
                "skill": s.skill,
                "class": CLASS_LABEL[s.skill],
                "traj_id": s.traj_id,
                "family": s.family,
                "kind": s.kind,
            }
            for i, s in enumerate(schedule)
        ],
        "excluded": {"heldout": {k: sorted(v) for k, v in holdout_ids().items()}},
        "notes": [
            "Init training from P47T (not P53) to isolate dataset composition.",
            "C/D unchanged from P53 train manifests.",
            "B full pool preserved in P54_B_FULL_MANIFEST.json for rotation.",
        ],
    }
    (OUT / "P54_TRAIN_IC_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "P54_B_FULL_MANIFEST.json").write_text(json.dumps(b_full, indent=2), encoding="utf-8")

    # leakage / duplicate audit
    tids = [s.traj_id for s in schedule]
    roots = [_normalize_root(t) for t in tids]
    held_all = holdout_ids()
    leak = []
    for s in schedule:
        if _is_held(s.traj_id, held_all[s.skill]):
            leak.append(s.traj_id)
    audit_md = [
        "# P54_DATASET_AUDIT",
        "",
        "## Counts",
        "",
        f"| Class | Target | Actual |",
        f"|-------|-------:|-------:|",
        f"| A | {TARGET_A} | {counts['A1']} |",
        f"| B | {TARGET_B} | {counts['B1']} |",
        f"| C | 68 | {counts['C']} |",
        f"| D | 93 | {counts['D']} |",
        f"| **Total** | **341** | **{len(schedule)}** |",
        "",
        "## A expansion",
        "",
        f"- Candidates discovered: {a_audit['candidates']}",
        f"- Selected valid: {a_audit['selected']}",
        f"- Family mix: {a_audit['family_selected']}",
        f"- Sources: {a_audit['source_selected']}",
        f"- Limitation: {a_audit['limitation'] or 'none (target met)'}",
        "",
        "## B reduction",
        "",
        f"- From P53 pool: {b_audit['available_p53']} → {b_audit['selected']}",
        f"- Family mix: {b_audit['family_selected']}",
        f"- Rationale: {json.dumps(b_audit['rationale'])}",
        f"- Full B manifest: `P54_B_FULL_MANIFEST.json` ({len(b_full)} ICs)",
        "",
        "## Validity / leakage / duplicates",
        "",
        f"- Duplicate traj_ids: {len(tids) - len(set(tids))}",
        f"- Duplicate roots: {len(roots) - len(set(roots))}",
        f"- Holdout leaks: {len(leak)} {leak[:5]}",
        f"- Each selected A/B IC restore-validated: yes (A explicitly; B/C/D from P53-validated pool)",
        "",
        "## C/D",
        "",
        "- Unchanged relative to P53 train IC lists.",
        "",
    ]
    (OUT / "P54_DATASET_AUDIT.md").write_text("\n".join(audit_md), encoding="utf-8")

    return {
        "manifest": manifest,
        "schedule": schedule,
        "counts": counts,
        "a_audit": a_audit,
        "b_audit": b_audit,
        "lookup": {(s.skill, s.traj_id): s for s in schedule},
    }


if __name__ == "__main__":
    d = build_p54_dataset()
    print(json.dumps(d["counts"], indent=2))
    print("A limitation:", d["a_audit"].get("limitation"))
    print("n", d["manifest"]["n_train_ics"])
