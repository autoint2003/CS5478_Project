"""K=36 and K=47 residual coverage for every successful v2 trajectory.

Does not replay and does not train.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/search.jsonl"
TRAJ = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/trajectories"
OUT = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/coverage_all_holds.json"
NAMES = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]


def phase_of(action, contact):
    grip = float(action[6])
    trans = np.abs(action[:3])
    horizontal = float(max(trans[0], trans[1]))
    open_grip = grip < -0.5
    closed = grip > 0.5
    if contact == "none" and open_grip and horizontal >= 0.35:
        return "airborne_or_open_lateral_chase"
    if contact == "none" and open_grip:
        return "open_while_no_contact"
    if open_grip and float(np.max(trans)) < 0.15:
        return "open_nearly_stationary"
    if open_grip and float(np.max(trans)) < 0.40:
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


def score(actions, proto):
    gap = np.max(np.abs(proto[None] - actions[:, None]), axis=2)
    linf = gap.min(1)
    nearest = gap.argmin(1)
    return nearest, linf


def main():
    proto = np.asarray(torch.load(ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"], np.float32)
    extra = np.load(ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy")
    candidate = np.concatenate([proto, extra], axis=0)
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("primary_10s")]
    far_phase = Counter()
    dim_over = Counter()
    records = []
    for row in rows:
        path = TRAJ / f"{row['id']}__{row['strategy']}.npz"
        blob = np.load(path)
        actions = blob["actions"].astype(np.float32)
        obs = blob["observations"].astype(np.float32)
        _near36, linf36 = score(actions, proto)
        near47, linf47 = score(actions, candidate)
        both = (obs[:, 2] > 0.5) & (obs[:, 3] > 0.5)
        none = (obs[:, 2] <= 0.5) & (obs[:, 3] <= 0.5)
        for index in np.flatnonzero(linf47 > 0.25 + 1e-6):
            contact = "bilateral" if both[index] else "none" if none[index] else "unilateral"
            kind = phase_of(actions[index], contact)
            far_phase[kind] += 1
            resid = np.abs(actions[index] - candidate[near47[index]])
            for dim, value in enumerate(resid):
                if value > 0.25 + 1e-6:
                    dim_over[NAMES[dim]] += 1
        number = int(row["id"].rsplit("_", 1)[-1])
        records.append({
            "id": row["id"],
            "strategy": row["strategy"],
            "family": row["family"],
            "held_out": number % 5 == 0,
            "steps": int(len(actions)),
            "k36_max_linf": round(float(linf36.max()), 4),
            "k36_frac_outside": round(float(np.mean(linf36 > 0.25 + 1e-6)), 4),
            "k36_inside": bool(float(linf36.max()) <= 0.25 + 1e-6),
            "k47_max_linf": round(float(linf47.max()), 4),
            "k47_frac_outside": round(float(np.mean(linf47 > 0.25 + 1e-6)), 4),
            "k47_inside": bool(float(linf47.max()) <= 0.25 + 1e-6),
            "k36_replay_10s": row.get("k36_replay_10s"),
            "held_12": row.get("held_12"),
            "legacy_t12": row.get("legacy_t12"),
        })
    payload = {
        "holds": len(records),
        "k36_inside": sum(1 for item in records if item["k36_inside"]),
        "k47_inside": sum(1 for item in records if item["k47_inside"]),
        "k47_far_phase": dict(far_phase),
        "k47_dim_over": dict(dim_over),
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("holds", len(records), "k36", payload["k36_inside"], "k47", payload["k47_inside"])
    print("far", dict(far_phase))
    print("dim", dict(dim_over))


if __name__ == "__main__":
    main()
