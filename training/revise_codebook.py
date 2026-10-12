"""Fit a revised codebook on non-held-out TRAIN successes. Does not replay or train."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results/diagnostics/raw/legal_recovery_v2/search.jsonl"
TRAJ = ROOT / "results/diagnostics/raw/legal_recovery_v2/trajectories"
OUT = ROOT / "results/diagnostics/raw/legal_recovery_v2/revised_codebook.json"


def phase_of(action, contact):
    grip = float(action[6])
    horizontal = float(max(abs(action[0]), abs(action[1])))
    open_grip = grip < -0.5
    closed = grip > 0.5
    if contact == "none" and open_grip and horizontal >= 0.35:
        return "airborne_or_open_lateral_chase"
    if contact == "none" and open_grip:
        return "open_while_no_contact"
    if open_grip and float(np.max(np.abs(action[:3]))) < 0.15:
        return "open_nearly_stationary"
    if open_grip and float(np.max(np.abs(action[:3]))) < 0.40:
        return "open_small_translation"
    if open_grip and horizontal >= 0.40:
        return "lateral_chase_open"
    if open_grip:
        return "open_other_translation"
    if closed and contact == "bilateral":
        return "reclose_or_stabilize"
    if closed:
        return "closed_reposition"
    return "partial_grip"


def greedy(points, cap=0.25, limit=24):
    uncovered = np.ones(len(points), dtype=bool)
    centers = []
    while uncovered.any() and len(centers) < limit:
        active = np.flatnonzero(uncovered)
        best_i, best_n = int(active[0]), -1
        block = 4000
        for start in range(0, len(active), block):
            chunk = points[active[start:start + block]]
            # Count coverage against all uncovered points.
            pending = points[uncovered]
            counts = np.sum(np.max(np.abs(pending[None] - chunk[:, None]), axis=2) <= cap, axis=1)
            local = int(np.argmax(counts))
            if int(counts[local]) > best_n:
                best_n = int(counts[local])
                best_i = int(active[start + local])
        centers.append(points[best_i].copy())
        uncovered[np.max(np.abs(points - points[best_i]), axis=1) <= cap] = False
        print("center", len(centers), "covers", best_n, "left", int(uncovered.sum()), flush=True)
    return np.stack(centers).astype(np.float32) if centers else np.zeros((0, 7), np.float32)


def main():
    proto = np.asarray(torch.load(ROOT / "results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"], np.float32)
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("primary_10s")]
    sequences = {}
    far_fit = []
    phase_counts = Counter()
    for row in rows:
        actions = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")["actions"].astype(np.float32)
        obs = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")["observations"].astype(np.float32)
        sequences[(row["id"], row["strategy"], row["family"])] = (actions, obs)
        number = int(row["id"].rsplit("_", 1)[-1])
        if number % 5 == 0:
            continue
        gap = np.max(np.abs(proto[None] - actions[:, None]), axis=2).min(1)
        if not np.any(gap > 0.25):
            continue
        both = (obs[:, 2] > 0.5) & (obs[:, 3] > 0.5)
        none = (obs[:, 2] <= 0.5) & (obs[:, 3] <= 0.5)
        picked = actions[gap > 0.25]
        far_fit.append(picked)
        for index in np.flatnonzero(gap > 0.25):
            contact = "bilateral" if both[index] else "none" if none[index] else "unilateral"
            phase_counts[phase_of(actions[index], contact)] += 1
    points = np.unique(np.round(np.concatenate(far_fit), 2), axis=0).astype(np.float32)
    print("unique_far", len(points), "phases", dict(phase_counts), flush=True)
    extra = greedy(points)
    candidate = np.concatenate([proto, extra], axis=0)

    def covered(held):
        inside = total = 0
        missing = []
        for key, (actions, _obs) in sequences.items():
            number = int(key[0].rsplit("_", 1)[-1])
            if (number % 5 == 0) != held:
                continue
            total += 1
            linf = np.max(np.abs(candidate[None] - actions[:, None]), axis=2).min(1)
            if float(linf.max()) <= 0.25 + 1e-6:
                inside += 1
            else:
                missing.append({"id": key[0], "strategy": key[1], "family": key[2], "max_linf": round(float(linf.max()), 3)})
        return inside, total, missing

    fit_in, fit_n, fit_miss = covered(False)
    hold_in, hold_n, hold_miss = covered(True)
    payload = {
        "extra": np.round(extra, 3).tolist(),
        "candidate_k": int(candidate.shape[0]),
        "fit_inside": [fit_in, fit_n],
        "held_out_inside": [hold_in, hold_n],
        "fit_missing": fit_miss[:30],
        "held_out_missing": hold_miss[:30],
        "fit_missing_n": len(fit_miss),
        "held_out_missing_n": len(hold_miss),
        "far_phase": dict(phase_counts),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(ROOT / "results/diagnostics/raw/legal_recovery_v2/revised_extra.npy", extra)
    print("K", candidate.shape[0], "fit", fit_in, fit_n, "hold", hold_in, hold_n, flush=True)


if __name__ == "__main__":
    main()
