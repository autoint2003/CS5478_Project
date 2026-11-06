"""Phase 4 — Expand C training ICs (no overlap with train/dev/final-test)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.precision.p51_four_class_mixed_ppo.ic_pools import holdout_ids, root_id
from experiments.precision.p53_independent_grpo64.full_dataset import build_uncapped_pools
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p62_grpo_optimization.eval_dev import _flat_starts, _fresh_final_starts
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor
from experiments.precision.p63_transformer_critic_ppo.paths import (
    EXPAND_C_N,
    OUT,
    P62_EPOCH3,
    P62_FINAL,
    SEED,
)
from experiments.precision.p63_transformer_critic_ppo.rollout import make_env


def _blocked_ids() -> set[str]:
    """All ICs that must not enter the expanded C train set."""
    blocked: set[str] = set()
    # P54 train (includes existing 68 C)
    ds = build_p54_dataset(seed=SEED)
    for s in ds["schedule"]:
        blocked.add(s.traj_id)
        blocked.add(root_id(s.traj_id))
    # P61/P62 development heldout
    flat, _ = _flat_starts(4)
    for s in flat:
        blocked.add(s.traj_id)
        blocked.add(root_id(s.traj_id))
    # P62 final test
    if P62_FINAL.is_file():
        final = json.loads(P62_FINAL.read_text(encoding="utf-8"))
        for row in final.get("ic_ids") or []:
            blocked.add(row["traj_id"])
            blocked.add(root_id(row["traj_id"]))
    else:
        flat_f, _ = _fresh_final_starts({s.traj_id for s in flat}, 4)
        for s in flat_f:
            blocked.add(s.traj_id)
            blocked.add(root_id(s.traj_id))
    # All P44 holdouts
    held = holdout_ids()
    for sk, ids in held.items():
        blocked |= set(ids)
    return blocked


def _is_blocked(tid: str, blocked: set[str]) -> bool:
    if tid in blocked:
        return True
    rid = root_id(tid)
    if rid in blocked:
        return True
    for b in blocked:
        if tid in b or b in tid or rid in b or b in rid:
            return True
    return False


def _family(tid: str) -> str:
    t = tid.lower()
    if "gbi_" in t:
        # coarse family by gbi id prefix
        for part in t.replace("__", "_").split("_"):
            if part.startswith("gbi") and part != "gbi":
                return part[:8]
            if part.isdigit() and len(part) >= 3:
                return f"gbi_{part[:3]}"
        return "gbi"
    if "plan_c" in t:
        return "plan_c"
    return "other"


def select_expanded_c(*, n: int = EXPAND_C_N, seed: int = SEED) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    blocked = _blocked_ids()
    pools = build_uncapped_pools(per_class_val=4, seed=seed)
    # Eligible C = all train+val C from uncapped that are not blocked
    candidates = []
    for split in ("train", "val"):
        for s in pools[split].get("C") or []:
            if _is_blocked(s.traj_id, blocked):
                continue
            candidates.append(s)

    # Deduplicate
    seen = set()
    uniq = []
    for s in candidates:
        if s.traj_id in seen:
            continue
        seen.add(s.traj_id)
        uniq.append(s)

    rng = np.random.default_rng(seed + 6301)
    # Diversify by family
    by_fam: dict[str, list] = defaultdict(list)
    for s in uniq:
        by_fam[_family(s.traj_id)].append(s)
    for fam in by_fam:
        rng.shuffle(by_fam[fam])

    selected = []
    fams = list(by_fam.keys())
    rng.shuffle(fams)
    # round-robin families
    while len(selected) < n and any(by_fam[f] for f in fams):
        for f in fams:
            if len(selected) >= n:
                break
            if by_fam[f]:
                selected.append(by_fam[f].pop())

    # Restore-validate (reset must succeed)
    _actor, book, rho, _ = load_actor(P62_EPOCH3, "cpu")
    env = make_env(book, rho)
    validated = []
    rejected = []
    for s in selected:
        try:
            env.reset(s)
            validated.append(s)
        except Exception as e:
            rejected.append({"traj_id": s.traj_id, "error": f"{type(e).__name__}: {e}"})

    # Top up if rejections
    leftovers = [s for s in uniq if s.traj_id not in {x.traj_id for x in validated}]
    rng.shuffle(leftovers)
    for s in leftovers:
        if len(validated) >= n:
            break
        if _is_blocked(s.traj_id, blocked):
            continue
        try:
            env.reset(s)
            validated.append(s)
        except Exception as e:
            rejected.append({"traj_id": s.traj_id, "error": f"{type(e).__name__}: {e}"})

    validated = validated[:n]
    manifest_path = OUT / "C_DATA_MANIFEST.jsonl"
    rows = []
    with manifest_path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(validated):
            row = {
                "index": i,
                "skill": "C",
                "traj_id": s.traj_id,
                "family": _family(s.traj_id),
                "kind": s.kind,
                "source": "uncapped_p53_pools_minus_p54_train_minus_dev_final_holdout",
                "provenance": {
                    "excluded_p54_train": True,
                    "excluded_p61_p62_dev": True,
                    "excluded_p62_final_test": True,
                    "excluded_p44_holdout": True,
                    "final_test_outcomes_not_used": True,
                },
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    summary = {
        "n_requested": n,
        "n_candidates": len(uniq),
        "n_selected": len(validated),
        "n_rejected_reset": len(rejected),
        "families": dict(
            sorted(
                {
                    r["family"]: sum(1 for x in rows if x["family"] == r["family"]) for r in rows
                }.items()
            )
        ),
        "rejected": rejected[:20],
        "manifest": str(manifest_path),
    }
    (OUT / "C_DATA_EXPANSION_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "starts": validated,
        "summary": summary,
        "rows": rows,
    }


def load_expanded_c_starts():
    """Rebuild MixedStart objects for the written manifest.

    Prefer newly generated Plan-C experts (``generate_c_experts``) when present;
    fall back to leftover uncapped-pool ICs.
    """
    from experiments.precision.p63_transformer_critic_ppo.generate_c_experts import (
        load_new_c_mixed_starts,
    )

    generated = load_new_c_mixed_starts()
    if generated:
        return generated

    path = OUT / "C_DATA_MANIFEST.jsonl"
    if not path.is_file():
        return select_expanded_c()["starts"]
    wanted = [json.loads(ln)["traj_id"] for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    pools = build_uncapped_pools(per_class_val=4, seed=SEED)
    lookup = {}
    for split in ("train", "val"):
        for s in pools[split].get("C") or []:
            lookup[s.traj_id] = s
    return [lookup[t] for t in wanted if t in lookup]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=EXPAND_C_N)
    args = ap.parse_args()
    out = select_expanded_c(n=args.n)
    print(json.dumps(out["summary"], indent=2))
