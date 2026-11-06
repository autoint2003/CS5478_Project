"""Training schedules: original 321 vs +32 new C (matched pilot IC budgets)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p63_transformer_critic_ppo.expand_c import load_expanded_c_starts
from experiments.precision.p63_transformer_ppo_c_grasp.paths import (
    C_MANIFEST,
    OUT,
    PILOT_ICS,
    SEED,
)


def original_schedule(seed: int = SEED) -> list:
    return list(build_p54_dataset(seed=seed)["schedule"])


def expanded_c_starts() -> list:
    return load_expanded_c_starts()


def stratified_pilot(schedule: list, max_ics: int = PILOT_ICS) -> list:
    by = {"A1": [], "B1": [], "C": [], "D": []}
    for st in schedule:
        by[st.skill].append(st)
    per = max(1, max_ics // 4)
    out = []
    for sk in ("A1", "B1", "C", "D"):
        out.extend(by[sk][:per])
    return out[:max_ics]


def schedule_branch_a(*, max_ics: int = PILOT_ICS) -> list:
    return stratified_pilot(original_schedule(), max_ics)


def schedule_branch_b(*, max_ics: int = PILOT_ICS) -> tuple[list, int]:
    """Same IC count as A; swap half of C slots for new C ICs."""
    base = schedule_branch_a(max_ics=max_ics)
    expanded = expanded_c_starts()
    non_c = [s for s in base if s.skill != "C"]
    orig_c = [s for s in base if s.skill == "C"]
    n_c = len(orig_c)
    n_new = min(len(expanded), max(1, n_c // 2))
    c_mix = orig_c[: max(0, n_c - n_new)] + expanded[:n_new]
    sched = (non_c + c_mix)[: len(base)]
    while len(sched) < len(base) and expanded:
        sched.append(expanded[len(sched) % len(expanded)])
    return sched, n_new


def schedule_branch_c(*, max_ics: int = PILOT_ICS) -> tuple[list, int]:
    """Identical IC schedule to Branch B (reward differs, not data)."""
    return schedule_branch_b(max_ics=max_ics)


def sync_c_manifest() -> Path:
    """Copy C_DATA_MANIFEST into this experiment's results tree."""
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "C_DATA_MANIFEST.jsonl"
    if C_MANIFEST.is_file():
        shutil.copy2(C_MANIFEST, dest)
        # Prefer generated experts manifest if richer
        gen = C_MANIFEST.parent / "c_experts" / "C_NEW_IC_MANIFEST.jsonl"
        if gen.is_file():
            # Merge note file
            rows_gen = [json.loads(ln) for ln in gen.read_text(encoding="utf-8").splitlines() if ln.strip()]
            note = {
                "source_primary": str(gen),
                "n_new_c": len(rows_gen),
                "also_copied": str(C_MANIFEST),
            }
            (OUT / "C_DATA_SOURCE.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
            shutil.copy2(gen, dest)
    return dest


def full_pool_counts() -> dict:
    """Expected A60/B100/C100/D93 ≈ 353 after +32 C."""
    sched = original_schedule()
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    cmap = {"A1": "A", "B1": "B", "C": "C", "D": "D"}
    for s in sched:
        counts[cmap[s.skill]] += 1
    n_new = len(expanded_c_starts())
    return {
        "original": counts,
        "original_total": sum(counts.values()),
        "new_c": n_new,
        "expanded_C": counts["C"] + n_new,
        "expanded_total": sum(counts.values()) + n_new,
        "target": {"A": 60, "B": 100, "C": 100, "D": 93, "total": 353},
    }
