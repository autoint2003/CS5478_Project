"""Phase 3 — training exposure audit (P63-A + P64)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p63_transformer_ppo_c_grasp.dataset import schedule_branch_a
from experiments.precision.p64_ppo_multiround.paths import OUT as P64_OUT
from experiments.precision.p65_ppo_continuation.paths import OUT, SEED


def run_exposure() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = list(build_p54_dataset(seed=SEED)["schedule"])
    pool_by = Counter({"A1": 0, "B1": 0, "C": 0, "D": 0})
    for s in pool:
        pool_by[s.skill] += 1

    # P63-A fixed stratified pilot
    sched_a = schedule_branch_a(max_ics=64)
    p63 = Counter(s.skill for s in sched_a)
    # 4 updates × 8 traj
    p63_traj = {k: v * 8 * 4 for k, v in p63.items()}

    # P64 exposure jsonl
    exp_path = P64_OUT / "IC_EXPOSURE.jsonl"
    p64_ic = Counter()
    p64_unique = defaultdict(set)
    p64_traj = Counter()
    if exp_path.is_file():
        for ln in exp_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            by = row.get("by_skill") or {}
            n_traj = int(row.get("n_traj_per_ic") or 8)
            for sk, n in by.items():
                p64_ic[sk] += int(n)
                p64_traj[sk] += int(n) * n_traj
            for tid in row.get("traj_ids") or []:
                # skill unknown from id alone — count unique overall
                p64_unique["ALL"].add(tid)
            # better: zip skills from by counts is lossy; re-derive from traj list length
            # store traj ids with step already; unique per class needs skill — approximate via schedule lookup
    lookup = {s.traj_id: s.skill for s in pool}
    uniq_by = defaultdict(set)
    if exp_path.is_file():
        for ln in exp_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            for tid in row.get("traj_ids") or []:
                sk = lookup.get(tid)
                if sk:
                    uniq_by[sk].add(tid)

    # effective PPO updates
    p63_updates = 4
    p64_updates = 0
    if exp_path.is_file():
        p64_updates = sum(1 for ln in exp_path.read_text(encoding="utf-8").splitlines() if ln.strip())

    report = {
        "nominal_pool": {
            "A": pool_by["A1"],
            "B": pool_by["B1"],
            "C": pool_by["C"],
            "D": pool_by["D"],
            "total": sum(pool_by.values()),
            "target_note": "A60/B100/C68/D93 = 321",
        },
        "p63_ppo_a": {
            "unique_ics_by_skill": dict(p63),
            "traj_by_skill": p63_traj,
            "updates": p63_updates,
            "total_traj": sum(p63_traj.values()),
            "sampling": "fixed first-16-per-class each update (repeated)",
        },
        "p64": {
            "collection_rows": p64_updates,
            "ics_sampled_slots_by_skill": dict(p64_ic),
            "unique_ics_by_skill": {k: len(v) for k, v in uniq_by.items()},
            "traj_by_skill": dict(p64_traj),
            "total_traj": sum(p64_traj.values()),
            "sampling": "randomized stratified 16/class per update",
        },
        "combined_traj_approx": {
            sk: p63_traj.get(sk, 0) + p64_traj.get(sk, 0) for sk in ("A1", "B1", "C", "D")
        },
    }
    # under-exposure flags: unique C << pool C
    flags = []
    for sk, lab in (("A1", "A"), ("C", "C")):
        u = report["p64"]["unique_ics_by_skill"].get(sk, 0)
        pool_n = report["nominal_pool"][lab]
        if u < max(8, pool_n // 4):
            flags.append(f"{lab}_unique_ics_low_in_p64:{u}/{pool_n}")
    report["flags"] = flags

    (OUT / "TRAINING_EXPOSURE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# TRAINING_EXPOSURE",
        "",
        f"Nominal pool: {report['nominal_pool']}",
        "",
        "## P63 PPO-A",
        "",
        f"- unique ICs/class: {report['p63_ppo_a']['unique_ics_by_skill']}",
        f"- traj/class: {report['p63_ppo_a']['traj_by_skill']}",
        f"- {report['p63_ppo_a']['sampling']}",
        "",
        "## P64",
        "",
        f"- unique ICs/class: {report['p64']['unique_ics_by_skill']}",
        f"- traj/class: {report['p64']['traj_by_skill']}",
        f"- collections: {report['p64']['collection_rows']}",
        f"- {report['p64']['sampling']}",
        "",
        f"**Flags:** {flags}",
        "",
        "No sampling-weight changes in P65.",
        "",
    ]
    (OUT / "TRAINING_EXPOSURE.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_exposure(), indent=2))
