"""Summarize v2 launch yield before and after the near-normal proposal change."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = ROOT / "results/diagnostics/raw/ball_impact_dataset_v2/attempts.jsonl"
OUT = ROOT / "results/diagnostics/raw/ball_impact_dataset_v2/launch_stats_v3.json"


def force_pair(row):
    hits = ((row.get("contact") or {}).get("hits") if isinstance(row.get("contact"), dict) else None)
    if hits is None:
        hits = row.get("contact") or []
    if isinstance(hits, dict):
        hits = [hits]
    fn = ft = 0.0
    normal = None
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        fn += float(hit.get("fn") or 0.0)
        ft += float(hit.get("ft") or 0.0)
        if normal is None and hit.get("normal") is not None:
            normal = hit.get("normal")
    return fn, ft, normal


def pack(rows):
    bins = defaultdict(lambda: [0, 0])
    phi = defaultdict(lambda: [0, 0])
    speed = defaultdict(lambda: [0, 0])
    epos = {"entry": [], "miss": []}
    fns = {"entry": [], "miss": []}
    fts = {"entry": [], "miss": []}
    for row in rows:
        azimuth = float(row.get("azimuth_deg") or 0.0) % 360.0
        entered = row.get("category") == "valid_impact_recovery_ic"
        bin_id = int(azimuth // 30) % 12
        bins[bin_id][0] += 1
        bins[bin_id][1] += int(entered)
        angle = abs(float((row.get("spec") or {}).get("phi_deg", 0.0)))
        name = "near_normal" if angle < 20 else "oblique" if angle < 45 else "tangential"
        phi[name][0] += 1
        phi[name][1] += int(entered)
        value = float(row.get("speed_mps") or 0.0)
        band = "under_6" if value < 6 else "6_to_7" if value < 7 else "over_7"
        speed[band][0] += 1
        speed[band][1] += int(entered)
        key = "entry" if entered else "miss"
        epos[key].append(float(row.get("max_e_pos_m") or 0.0))
        fn, ft, _normal = force_pair(row)
        if row.get("peak_normal_force_n") is not None:
            fns[key].append(float(row["peak_normal_force_n"]))
        fts[key].append(ft)
    return {
        "attempts": len(rows),
        "accepted": sum(1 for row in rows if row.get("category") == "valid_impact_recovery_ic"),
        "categories": dict(__import__("collections").Counter(row.get("category") for row in rows)),
        "bins": {str(k): {"attempts": v[0], "accepted": v[1]} for k, v in sorted(bins.items())},
        "phi": {k: {"attempts": v[0], "accepted": v[1]} for k, v in phi.items()},
        "speed": {k: {"attempts": v[0], "accepted": v[1]} for k, v in speed.items()},
        "median_epos_entry": None if not epos["entry"] else float(np.median(epos["entry"])),
        "median_epos_miss": None if not epos["miss"] else float(np.median(epos["miss"])),
        "median_fn_entry": None if not fns["entry"] else float(np.median(fns["entry"])),
        "median_fn_miss": None if not fns["miss"] else float(np.median(fns["miss"])),
        "median_ft_entry": None if not fts["entry"] else float(np.median(fts["entry"])),
        "median_ft_miss": None if not fts["miss"] else float(np.median(fts["miss"])),
    }


def main():
    rows = [json.loads(line) for line in ATTEMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Batches b0001-b0003 are the first 216 logged launches. Later batches use the near-normal proposal.
    early = [row for row in rows if row.get("batch_id") in {"b0001", "b0002", "b0003"}]
    late = [row for row in rows if row.get("batch_id") in {"b0004", "b0005", "b0006"}]
    accepted = [row for row in rows if row.get("category") == "valid_impact_recovery_ic"]
    payload = {
        "v2_attempts": len(rows),
        "v2_accepted": len(accepted),
        "early": pack(early),
        "late": pack(late),
        "all": pack(rows),
        "accepted_azimuth_min": None if not accepted else round(min(float(row["azimuth_deg"]) for row in accepted), 2),
        "accepted_azimuth_max": None if not accepted else round(max(float(row["azimuth_deg"]) for row in accepted), 2),
        "accepted_speed_min": None if not accepted else round(min(float(row["speed_mps"]) for row in accepted), 3),
        "accepted_speed_max": None if not accepted else round(max(float(row["speed_mps"]) for row in accepted), 3),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("attempts", len(rows), "accepted", len(accepted))
    print("early", payload["early"]["accepted"], "/", payload["early"]["attempts"], payload["early"]["phi"])
    print("late", payload["late"]["accepted"], "/", payload["late"]["attempts"], payload["late"]["phi"])
    print("bins", payload["all"]["bins"])


if __name__ == "__main__":
    main()
