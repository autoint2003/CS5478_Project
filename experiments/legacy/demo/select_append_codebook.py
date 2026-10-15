"""Append-only codebook search. Does not replay or train.

New centers are rejected when they would change the nearest token of a
physically verified K=36 or K=47 trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
TRAJ = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/trajectories"
LOG = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/search.jsonl"
REPLAY = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/replay_k47_all.jsonl"
TOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/candidate_tokens.jsonl"
OUT = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/append_codebook.json"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def linf(actions, book):
    return np.max(np.abs(book[None] - actions[:, None]), axis=2).min(1)


def phase_of(action, contact):
    grip = float(action[6])
    horizontal = float(max(abs(float(action[0])), abs(float(action[1]))))
    open_grip = grip < -0.5
    if contact == "none" and open_grip and horizontal >= 0.35:
        return "airborne_or_lateral_chase"
    if contact == "none" and open_grip and abs(float(action[1])) >= 0.15:
        return "open_small_vy"
    if contact == "none" and open_grip and abs(float(action[2])) >= 0.15:
        return "open_vz"
    if contact == "none" and open_grip:
        return "open_while_no_contact"
    if open_grip and float(np.max(np.abs(action[:3]))) < 0.15:
        return "open_nearly_stationary"
    if open_grip:
        return "open_translation"
    if grip > 0.5:
        return "closed"
    return "partial_grip"


def main():
    proto = np.asarray(torch.load(ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"], np.float32)
    extra = np.load(ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy").astype(np.float32)
    book47 = np.concatenate([proto, extra], axis=0)
    holds = [row for row in load_jsonl(LOG) if row.get("primary_10s")]
    replay = {f"{row['id']}|{row['strategy']}": row for row in load_jsonl(REPLAY)}
    gaps = {f"{row['id']}|{row['strategy']}": int(row.get("none_intervals") or 0) for row in load_jsonl(TOK)}
    sequences = {}
    for row in holds:
        key = f"{row['id']}|{row['strategy']}"
        blob = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")
        actions = blob["actions"].astype(np.float32)
        obs = blob["observations"].astype(np.float32)
        dist47 = linf(actions, book47)
        sequences[key] = {
            "id": row["id"],
            "strategy": row["strategy"],
            "family": row["family"],
            "actions": actions,
            "obs": obs,
            "dist47": dist47,
            "inside47": bool(float(dist47.max()) <= 0.25 + 1e-6),
            "verified47": replay[key].get("primary_10s") is True,
            "multi": gaps.get(key, 0) >= 2,
            "held_out": int(row["id"].rsplit("_", 1)[-1]) % 5 == 0,
        }
    verified_ics = {item["id"] for item in sequences.values() if item["verified47"]}
    priority_ids = sorted({item["id"] for item in sequences.values() if item["id"] not in verified_ics})
    protected = []
    protected_dist = []
    for item in sequences.values():
        if not item["verified47"]:
            continue
        protected.append(item["actions"])
        protected_dist.append(item["dist47"])
    protected_actions = np.concatenate(protected, axis=0)
    protected_dist = np.concatenate(protected_dist, axis=0)

    def steals(center, book):
        current = protected_dist if book.shape[0] == book47.shape[0] else linf(protected_actions, book)
        proposed = np.max(np.abs(protected_actions - center.reshape(1, 7)), axis=1)
        return bool(np.any(proposed + 1e-8 < current))

    far_pool = []
    phase_counts = {}
    for item in sequences.values():
        if item["inside47"]:
            continue
        actions = item["actions"]
        obs = item["obs"]
        both = (obs[:, 2] > 0.5) & (obs[:, 3] > 0.5)
        none = (obs[:, 2] <= 0.5) & (obs[:, 3] <= 0.5)
        for index in np.flatnonzero(item["dist47"] > 0.25 + 1e-6):
            contact = "bilateral" if both[index] else "none" if none[index] else "unilateral"
            kind = phase_of(actions[index], contact)
            phase_counts[kind] = phase_counts.get(kind, 0) + 1
            if item["id"] in priority_ids:
                far_pool.append(actions[index])
    if not far_pool:
        far_pool = [item["actions"][index] for item in sequences.values() if not item["inside47"] for index in np.flatnonzero(item["dist47"] > 0.25)]
    points = np.unique(np.round(np.stack(far_pool), 2), axis=0).astype(np.float32)
    print("priority_ics", len(priority_ids), "far_points", len(points), "phases", phase_counts, flush=True)

    def coverage(book):
        inside_keys = []
        for key, item in sequences.items():
            if float(linf(item["actions"], book).max()) <= 0.25 + 1e-6:
                inside_keys.append(key)
        return inside_keys

    def ic_gain(book, center):
        trial = np.concatenate([book, center.reshape(1, 7)], axis=0)
        gained = set()
        gained_keys = []
        for key, item in sequences.items():
            if float(linf(item["actions"], book).max()) <= 0.25 + 1e-6:
                continue
            if float(linf(item["actions"], trial).max()) <= 0.25 + 1e-6:
                gained_keys.append(key)
                if item["id"] in priority_ids:
                    gained.add(item["id"])
        return gained, gained_keys

    selected = []
    book = book47.copy()
    for _round in range(8):
        best = None
        best_score = (0, 0)
        for center in points:
            if steals(center, book):
                continue
            gained_ics, gained_keys = ic_gain(book, center)
            # Prefer new priority ICs, then any newly covered trajectory.
            score = (len(gained_ics), len(gained_keys))
            if score > best_score:
                best_score = score
                best = (center, gained_ics, gained_keys)
        if best is None or best_score[0] == 0:
            print("stop", best_score, flush=True)
            break
        center, gained_ics, gained_keys = best
        selected.append(center)
        book = np.concatenate([book, center.reshape(1, 7)], axis=0)
        print("add", len(selected), "new_ics", len(gained_ics), "new_traj", len(gained_keys), "center", np.round(center, 3).tolist(), flush=True)

    # Compact reference: K=47 itself, and K=47 plus the safe additions.
    candidates = {"k36": proto, "k47": book47, "append": book}
    summary = {"priority_ics": priority_ids, "phases_on_oor": phase_counts, "added": np.round(np.stack(selected), 3).tolist() if selected else [], "candidates": {}}
    for name, candidate in candidates.items():
        keys = coverage(candidate)
        items = [sequences[key] for key in keys]
        ics = {item["id"] for item in items}
        priority_hit = sorted(ics & set(priority_ids))
        summary["candidates"][name] = {
            "k": int(candidate.shape[0]),
            "trajectories": len(keys),
            "ics": len(ics),
            "slides": sum(1 for item in items if item["family"] == "sliding"),
            "regrasp": sum(1 for item in items if item["family"] == "regrasp"),
            "multi": sum(1 for item in items if item["multi"]),
            "priority_ics_covered": priority_hit,
            "k36_tokens_unchanged": bool(np.allclose(candidate[:36], proto)),
        }
        print(name, summary["candidates"][name]["k"], summary["candidates"][name]["trajectories"], summary["candidates"][name]["ics"], "priority", len(priority_hit), flush=True)
    if selected:
        np.save(ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/append_extra.npy", np.stack(selected).astype(np.float32))
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
