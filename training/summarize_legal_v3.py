"""Count legal-search outcomes and store candidate token residuals.

Does not modify the continuous trajectory files and does not train.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results/diagnostics/raw/legal_recovery_v2/search.jsonl"
TRAJ = ROOT / "results/diagnostics/raw/legal_recovery_v2/trajectories"
OUT = ROOT / "results/diagnostics/raw/legal_recovery_v2/search_summary_v3.json"
TOKENS = ROOT / "results/diagnostics/raw/legal_recovery_v2/candidate_tokens.jsonl"


def none_intervals(obs):
    none = (obs[:, 2] <= 0.5) & (obs[:, 3] <= 0.5)
    count = 0
    previous = False
    for flag in none:
        if flag and not previous:
            count += 1
        previous = bool(flag)
    return count


def main():
    proto = np.asarray(torch.load(ROOT / "results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"], np.float32)
    extra = np.load(ROOT / "results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy")
    candidate = np.concatenate([proto, extra], axis=0)
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    holds = [row for row in rows if row.get("primary_10s")]
    by_family = Counter(row["family"] for row in holds)
    by_strategy = Counter(row["strategy"] for row in holds)
    ic_ids = sorted({row["id"] for row in holds})
    new_holds = [row for row in holds if int(row["id"].rsplit("_", 1)[-1]) >= 40]
    new_ics = sorted({row["id"] for row in new_holds})
    searched_new = sorted({row["id"] for row in rows if int(row["id"].rsplit("_", 1)[-1]) >= 40})
    labels = Counter(row.get("label") for row in rows if int(row["id"].rsplit("_", 1)[-1]) >= 40)
    token_rows = []
    multi = 0
    slide_k36 = 0
    release_inside = 0
    for row in holds:
        path = TRAJ / f"{row['id']}__{row['strategy']}.npz"
        if not path.is_file():
            continue
        blob = np.load(path)
        actions = blob["actions"].astype(np.float32)
        obs = blob["observations"].astype(np.float32)
        gap = np.max(np.abs(candidate[None] - actions[:, None]), axis=2)
        nearest = gap.argmin(1)
        linf = gap.min(1)
        residual = np.clip(actions - candidate[nearest], -0.25, 0.25)
        gaps = none_intervals(obs)
        if gaps >= 2:
            multi += 1
        if row["family"] == "sliding" and row.get("k36", {}).get("representable"):
            slide_k36 += 1
        if row["family"] == "regrasp" and float(linf.max()) <= 0.25 + 1e-6:
            release_inside += 1
        np.savez_compressed(
            TRAJ / f"{row['id']}__{row['strategy']}__candidate_k47.npz",
            tokens=nearest.astype(np.int16),
            residuals=residual.astype(np.float32),
        )
        token_rows.append({
            "id": row["id"],
            "strategy": row["strategy"],
            "family": row["family"],
            "steps": int(len(actions)),
            "max_linf": round(float(linf.max()), 4),
            "inside_0.25": bool(float(linf.max()) <= 0.25 + 1e-6),
            "none_intervals": gaps,
            "held_12": row.get("held_12"),
            "legacy_t12": row.get("legacy_t12"),
            "codebook": "tested_k47_not_adopted",
        })
    with TOKENS.open("w", encoding="utf-8") as handle:
        for item in token_rows:
            handle.write(json.dumps(item) + "\n")
    summary = {
        "trials": len(rows),
        "holds": len(holds),
        "hold_ics": len(ic_ids),
        "by_family": dict(by_family),
        "by_strategy": dict(by_strategy),
        "new_ics_searched": len(searched_new),
        "new_holds": len(new_holds),
        "new_hold_ics": len(new_ics),
        "new_labels": dict(labels),
        "slide_holds_marked_k36": slide_k36,
        "release_inside_candidate": release_inside,
        "holds_with_two_or_more_loss_intervals": multi,
        "token_rows": len(token_rows),
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
