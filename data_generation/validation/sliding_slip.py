"""Flag saved sliding tapes whose finger-axis slip is missing or very small.

Does not run the 12 s hold. The matched-IC audit already checked that hold.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.rebuild_k48_recovery import _open_env
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from data_generation.validation.sliding_mechanism import bilateral_mask, release_regrasp, sample, slip_direction

OUT = ROOT / "results" / "diagnostics" / "sliding_mechanism_audit"


def status(record):
    if record["release_then_regrasp"]:
        return "release_then_regrasp"
    if record["light_steps"] == 0 or record["path_mm"] < 0.5:
        return "no_in_hand_slip"
    if record["inward_rhx_mm"] < 0.5:
        return "weak_net_correction"
    return "confirmed"


def main():
    env = _open_env()
    rows = []
    for folder, strategy in (("controlled_sliding", "controlled_slide"), ("inward_sliding", "inward_reseat")):
        for line in (ROOT / "data" / folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_strategy"] = strategy
                rows.append(row)
    found = []
    for index, row in enumerate(rows, start=1):
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        with np.load(ROOT / row["continuous_npz"]) as archive:
            actions = np.asarray(archive["actions"], np.float32)
        restore_capture(env, blob["capture"])
        traced = [sample(env.sim, blob["ref"], np.zeros(7))]
        for action in actions:
            step_legal(env.sim, action)
            traced.append(sample(env.sim, blob["ref"], action))
        start, end = traced[0], traced[-1]
        _direction, path = slip_direction(traced)
        command = np.stack([item["action"] for item in traced[1:]])
        light = int(np.sum(command[:, 6] < 0.5))
        record = {
            "id": row["id"],
            "split": row["split"],
            "strategy": row["_strategy"],
            "snapshot": row["snapshot"],
            "inward_rhx_mm": round(1000.0 * (abs(float(start["rh"][0])) - abs(float(end["rh"][0]))), 3),
            "path_mm": round(1000.0 * path, 3),
            "bilateral_fraction": round(float(np.mean(bilateral_mask(traced))), 4),
            "release_then_regrasp": release_regrasp(traced),
            "light_steps": light,
            "p_nom_x": round(float(start["p_nom_x"]), 6),
        }
        record["mechanism_status"] = status(record)
        found.append(record)
        if record["mechanism_status"] != "confirmed":
            print("FLAG", record["mechanism_status"], record["id"], record["strategy"], record["inward_rhx_mm"], record["path_mm"], flush=True)
        elif index % 20 == 0:
            print("scan", index, flush=True)
    (OUT / "all_tape_slip.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
    print("SCAN_DONE", len(found), flush=True)


if __name__ == "__main__":
    main()
