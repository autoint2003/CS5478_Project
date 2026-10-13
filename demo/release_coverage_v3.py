"""Release-action gaps, held-out codebook, and launch-entry statistics.

Does not train and does not replay.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/search.jsonl"
TRAJ = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/trajectories"
ATTEMPTS = ROOT / "demo_results/diagnostics/raw/ball_impact_dataset_v2/attempts.jsonl"
PROTO_PATH = ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt"
OUT = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/coverage_v3.json"
NAMES = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]


def quantize(actions, proto):
    gap = np.max(np.abs(proto[None] - actions[:, None]), axis=2)
    nearest = gap.argmin(1)
    linf = gap.min(1)
    return nearest, linf


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


def greedy(points, cap=0.25):
    uncovered = np.ones(len(points), dtype=bool)
    centers = []
    while uncovered.any():
        best_i, best_n = 0, -1
        active = np.flatnonzero(uncovered)
        for index in active:
            count = int(np.sum(uncovered & (np.max(np.abs(points - points[index]), axis=1) <= cap)))
            if count > best_n:
                best_n, best_i = count, int(index)
        centers.append(points[best_i].copy())
        uncovered[np.max(np.abs(points - points[best_i]), axis=1) <= cap] = False
        if len(centers) > 30:
            break
    return np.stack(centers).astype(np.float32) if centers else np.zeros((0, 7), np.float32)


def main():
    proto = np.asarray(torch.load(PROTO_PATH, map_location="cpu", weights_only=False)["proto"], np.float32)
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [row for row in rows if row.get("primary_10s")]
    release = [row for row in ok if row["family"] == "regrasp"]
    slide = [row for row in ok if row["family"] == "sliding"]
    phase_counts = Counter()
    far_phase = Counter()
    dim_over = Counter()
    before = Counter()
    sequences = {}
    for row in release:
        actions = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")["actions"].astype(np.float32)
        obs = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")["observations"].astype(np.float32)
        nearest, linf = quantize(actions, proto)
        both = (obs[:, 2] > 0.5) & (obs[:, 3] > 0.5)
        none = (obs[:, 2] <= 0.5) & (obs[:, 3] <= 0.5)
        seen_loss = False
        seen_return = False
        for index, action in enumerate(actions):
            contact = "bilateral" if both[index] else "none" if none[index] else "unilateral"
            if contact == "none":
                seen_loss = True
            if seen_loss and contact == "bilateral":
                seen_return = True
            when = "after_recontact" if seen_return else "during_or_before_first_loss" if seen_loss else "before_loss"
            kind = phase_of(action, contact)
            phase_counts[kind] += 1
            before[f"{kind}|{when}|{contact}"] += 1
            if linf[index] > 0.25 + 1e-6:
                far_phase[kind] += 1
                resid = np.abs(action - proto[nearest[index]])
                for dim, value in enumerate(resid):
                    if value > 0.25 + 1e-6:
                        dim_over[NAMES[dim]] += 1
        sequences[(row["id"], row["strategy"])] = actions

    fit_ids = []
    hold_ids = []
    for row in release:
        number = int(row["id"].rsplit("_", 1)[-1])
        (hold_ids if number % 5 == 0 else fit_ids).append(row["id"])
    fit_ids = sorted(set(fit_ids))
    hold_ids = sorted(set(hold_ids))
    far_fit = []
    for key, actions in sequences.items():
        if key[0] not in fit_ids:
            continue
        _nearest, linf = quantize(actions, proto)
        if np.any(linf > 0.25):
            far_fit.append(actions[linf > 0.25])
    points = np.unique(np.round(np.concatenate(far_fit), 3), axis=0).astype(np.float32)
    extra = greedy(points)
    candidate = np.concatenate([proto, extra], axis=0)

    def covered(subset_ids):
        inside = total = 0
        missing = []
        for key, actions in sequences.items():
            if key[0] not in subset_ids:
                continue
            total += 1
            linf = np.max(np.abs(candidate[None] - actions[:, None]), axis=2).min(1)
            if float(linf.max()) <= 0.25 + 1e-6:
                inside += 1
            else:
                missing.append({"id": key[0], "strategy": key[1], "max_linf": round(float(linf.max()), 3)})
        return inside, total, missing

    slide_changed = []
    for row in slide:
        actions = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")["actions"].astype(np.float32)
        old, _old_linf = quantize(actions, proto)
        new, new_linf = quantize(actions, candidate)
        recon_old = proto[old]
        recon_new = candidate[new]
        if not np.allclose(recon_old, recon_new, atol=1e-5):
            slide_changed.append(row["id"] + " " + row["strategy"])
        if float(new_linf.max()) > 0.25 + 1e-6:
            slide_changed.append("OUT " + row["id"])

    # 0021 timeline
    path = TRAJ / "ball_train_v2_0021__release_long.npz"
    actions = np.load(path)["actions"].astype(np.float32)
    obs = np.load(path)["observations"].astype(np.float32)
    both = (obs[:, 2] > 0.5) & (obs[:, 3] > 0.5)
    none = (obs[:, 2] <= 0.5) & (obs[:, 3] <= 0.5)
    timeline = []
    state = "bilateral" if both[0] else "none" if none[0] else "unilateral"
    start = 0
    for index in range(1, len(both) + 1):
        cur = None if index == len(both) else ("bilateral" if both[index] else "none" if none[index] else "unilateral")
        if cur != state:
            chunk = actions[start:index]
            timeline.append({
                "state": state,
                "start": start,
                "end": index,
                "mean_action": [round(float(x), 3) for x in chunk.mean(0)],
                "grip_min": round(float(chunk[:, 6].min()), 3),
                "grip_max": round(float(chunk[:, 6].max()), 3),
            })
            state = cur
            start = index

    launches = [json.loads(line) for line in ATTEMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_bin = defaultdict(lambda: [0, 0])
    by_phi = defaultdict(lambda: [0, 0])
    by_speed = defaultdict(lambda: [0, 0])
    epos = {"entry": [], "miss": []}
    force = {"entry": [], "miss": []}
    for row in launches:
        azimuth = float(row.get("azimuth_deg") or 0.0) % 360.0
        entered = row.get("category") == "valid_impact_recovery_ic"
        by_bin[int(azimuth // 30) % 12][0] += 1
        by_bin[int(azimuth // 30) % 12][1] += int(entered)
        phi = abs(float((row.get("spec") or {}).get("phi_deg", 0.0)))
        phi_bin = "near_normal" if phi < 20 else "oblique" if phi < 45 else "tangential"
        by_phi[phi_bin][0] += 1
        by_phi[phi_bin][1] += int(entered)
        speed = float(row.get("speed_mps") or 0.0)
        speed_bin = "under_6" if speed < 6 else "6_to_7" if speed < 7 else "over_7"
        by_speed[speed_bin][0] += 1
        by_speed[speed_bin][1] += int(entered)
        key = "entry" if entered else "miss"
        epos[key].append(float(row.get("max_e_pos_m") or 0.0))
        force[key].append(float(row.get("peak_normal_force_n") or 0.0))
    bin6 = [row for row in launches if 180 <= float(row.get("azimuth_deg") or 0) < 210]
    payload = {
        "release_trajectories": len(release),
        "slide_trajectories": len(slide),
        "phase_counts": dict(phase_counts),
        "far_phase": dict(far_phase),
        "dim_over": dict(dim_over),
        "when": dict(before),
        "fit_ics": fit_ids,
        "held_out_ics": hold_ids,
        "extra": np.round(extra, 4).tolist(),
        "candidate_k": int(candidate.shape[0]),
        "fit_release_coverage": covered(*[None]) if False else None,
    }
    fit_in, fit_n, fit_miss = covered(set(fit_ids))
    hold_in, hold_n, hold_miss = covered(set(hold_ids))
    all_in, all_n, all_miss = covered(set(fit_ids) | set(hold_ids))
    payload.update({
        "fit_release_inside": [fit_in, fit_n],
        "held_out_release_inside": [hold_in, hold_n],
        "held_out_missing": hold_miss,
        "all_release_inside": [all_in, all_n],
        "all_release_missing": all_miss,
        "slide_quantization_changed": slide_changed,
        "timeline_0021": timeline,
        "launch_bins": {str(k): {"attempts": v[0], "accepted": v[1]} for k, v in sorted(by_bin.items())},
        "launch_phi": {k: {"attempts": v[0], "accepted": v[1]} for k, v in by_phi.items()},
        "launch_speed": {k: {"attempts": v[0], "accepted": v[1]} for k, v in by_speed.items()},
        "median_epos_entry": float(np.median(epos["entry"])) if epos["entry"] else None,
        "median_epos_miss": float(np.median(epos["miss"])) if epos["miss"] else None,
        "median_fn_entry": float(np.median(force["entry"])) if force["entry"] else None,
        "median_fn_miss": float(np.median(force["miss"])) if force["miss"] else None,
        "bin_180_210_attempts": len(bin6),
        "bin_180_210_accepted": sum(1 for row in bin6 if row.get("category") == "valid_impact_recovery_ic"),
        "bin_180_210_median_epos": float(np.median([float(row.get("max_e_pos_m") or 0) for row in bin6])) if bin6 else None,
        "bin_180_210_median_fn": float(np.median([float(row.get("peak_normal_force_n") or 0) for row in bin6])) if bin6 else None,
    })
    del payload["fit_release_coverage"]
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy", extra)
    print("extra", extra.shape[0], "fit", fit_in, fit_n, "hold", hold_in, hold_n, "all", all_in, all_n)
    print("slide_changed", slide_changed)
    print("far_phase", dict(far_phase))
    print("dim", dict(dim_over))
    print("phi", payload["launch_phi"], "speed", payload["launch_speed"])
    print("bin6", payload["bin_180_210_attempts"], payload["bin_180_210_median_epos"], payload["bin_180_210_median_fn"])
    print("0021", timeline)


if __name__ == "__main__":
    main()
