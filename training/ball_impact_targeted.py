"""Versioned TRAIN ball-impact generator.

v1 snapshots, the v1 manifest, and BALL_IMPACT_DATASET_REBUILD.md stay where
they are. New launches go to demo_results/diagnostics/raw/ball_impact_dataset_v2.
VALIDATION and FINAL_TEST are not written. This module does not train.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.ball_impact_dataset import (
    RAW as RAW_V1,
    load_log as load_v1,
    physics_record,
    run_episode,
)
from training.autonomous_episode import make_training_env

RAW = ROOT / "results" / "diagnostics" / "raw" / "ball_impact_dataset_v2"
ATTEMPTS = RAW / "attempts.jsonl"
STATE = RAW / "state.json"
REPORT = RAW / "aggregate.md"
PROTOCOL = "ball_impact_v2"
SPEEDS = (5.5, 6.5, 7.5, 8.5, 9.5)


def load_rows():
    rows = []
    if ATTEMPTS.is_file():
        for line in ATTEMPTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_state():
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"next_draw": 0}


def save_state(state):
    RAW.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state), encoding="utf-8")


def append_row(row):
    RAW.mkdir(parents=True, exist_ok=True)
    with ATTEMPTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def v1_specs():
    specs = []
    if not (RAW_V1 / "attempts.jsonl").is_file():
        return specs
    for row in load_v1():
        spec = row.get("spec") or {}
        if "azimuth_deg" not in spec:
            continue
        specs.append(spec)
    return specs


def near(spec, other):
    daz = abs((float(spec["azimuth_deg"]) - float(other["azimuth_deg"]) + 180.0) % 360.0 - 180.0)
    return (
        daz < 2.0
        and abs(float(spec["speed_mps"]) - float(other["speed_mps"])) < 0.20
        and abs(float(spec["z_off_m"]) - float(other["z_off_m"])) < 0.002
        and abs(float(spec["phi_deg"]) - float(other["phi_deg"])) < 5.0
        and abs(float(spec["delay_s"]) - float(other.get("delay_s", 0.0))) < 0.05
    )


def propose(rows, prior, draw):
    rng = np.random.default_rng(10_000 + int(draw))
    accepts = np.zeros(12)
    tries = np.zeros(12)
    near_tries = np.zeros(12)
    speed_tries = {}
    for row in rows:
        azimuth = float(row.get("azimuth_deg", 0.0)) % 360.0
        bin_id = int(azimuth // 30.0) % 12
        tries[bin_id] += 1
        if abs(float((row.get("spec") or {}).get("phi_deg", 99.0))) < 20.0:
            near_tries[bin_id] += 1
        if row.get("category") == "valid_impact_recovery_ic":
            accepts[bin_id] += 1
        speed_tries[(bin_id, round(float(row.get("speed_mps", 0.0)), 1))] = speed_tries.get(
            (bin_id, round(float(row.get("speed_mps", 0.0)), 1)), 0
        ) + 1

    def rank(bin_id):
        dead = 1 if near_tries[bin_id] >= 8 and accepts[bin_id] == 0 else 0
        return (dead, int(accepts[bin_id]), int(tries[bin_id]), bin_id)

    bin_id = sorted(range(12), key=rank)[0]
    ordered = sorted(SPEEDS, key=lambda speed: speed_tries.get((bin_id, round(float(speed), 1)), 0))
    speed = float(ordered[0] + rng.uniform(-0.15, 0.15))
    spec = {
        "split": "train",
        "protocol": PROTOCOL,
        "azimuth_deg": float(bin_id * 30.0 + rng.uniform(0.0, 30.0)),
        "speed_mps": speed,
        "z_off_m": float(rng.uniform(-0.020, 0.020)),
        "phi_deg": float(rng.uniform(-12.0, 12.0) if rng.random() < 0.8 else rng.choice([-50.0, 50.0]) + rng.uniform(-5.0, 5.0)),
        "delay_s": float(rng.uniform(0.0, 0.40)),
        "draw": int(draw),
        "direction": "continuous",
    }
    pool = prior + [row.get("spec", row) for row in rows]
    if any(near(spec, item) for item in pool if "azimuth_deg" in item):
        spec["azimuth_deg"] = float((spec["azimuth_deg"] + 17.0) % 360.0)
    if any(near(spec, item) for item in pool if "azimuth_deg" in item):
        spec["skip"] = "near_duplicate_launch"
    return spec


def cylinder_first(evidence, blob):
    other = evidence.get("first_other")
    contact_t = evidence.get("contact_t")
    if evidence.get("category") != "valid_impact_recovery_ic":
        return evidence, blob
    if other is not None and contact_t is not None and float(other["t"]) < float(contact_t) - 1e-6:
        evidence = dict(evidence)
        evidence["category"] = "robot_or_environment_before_cylinder"
        evidence["reason"] = (
            f"{other.get('kind')} contact at t={other.get('t')} s preceded the cylinder contact at t={contact_t} s"
        )
        return evidence, None
    if other is not None and contact_t is not None and abs(float(other["t"]) - float(contact_t)) <= 1e-6:
        evidence = dict(evidence)
        evidence["simultaneous_other"] = True
    return evidence, blob


def save_blob(blob, evidence):
    folder = RAW / "accepted" / "train"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{evidence['id']}.pkl"
    with path.open("wb") as handle:
        pickle.dump(blob, handle)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def file_counts():
    v1 = ROOT / "results" / "diagnostics" / "raw" / "ball_impact_dataset" / "accepted"
    v2 = RAW / "accepted"
    def nfiles(folder):
        return len(list(folder.glob("*.pkl"))) if folder.is_dir() else 0
    return {
        "v1_train": nfiles(v1 / "train"),
        "v1_validation": nfiles(v1 / "validation"),
        "v1_final_test": nfiles(v1 / "final_test"),
        "v2_train": nfiles(v2 / "train"),
    }


def write_batch(batch_id, batch_rows, seconds):
    folder = RAW / "batches"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"batch_id": batch_id, "seconds": round(seconds, 2), "rows": len(batch_rows)}
    (folder / f"{batch_id}.json").write_text(json.dumps({"meta": payload, "rows": batch_rows}, indent=2), encoding="utf-8")
    accepted = sum(1 for row in batch_rows if row.get("category") == "valid_impact_recovery_ic")
    lines = [
        f"# Targeted batch {batch_id}",
        "",
        f"Attempts {len(batch_rows)}. Accepted TRAIN {accepted}. Seconds {round(seconds, 1)}.",
        "",
        "VALIDATION and FINAL_TEST were not written.",
        "",
    ]
    (folder / f"{batch_id}.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def write_aggregate(rows, seconds, ran):
    counts = {}
    for row in rows:
        counts[row.get("category", "unknown")] = counts.get(row.get("category"), 0) + 1
    accepted = [row for row in rows if row.get("category") == "valid_impact_recovery_ic"]
    azimuths = [float(row["azimuth_deg"]) for row in accepted]
    speeds = [float(row["speed_mps"]) for row in accepted]
    heights = [float(row["spec"]["z_off_m"]) for row in accepted]
    files = file_counts()
    missing = [row["id"] for row in accepted if row.get("path") and not (ROOT / row["path"]).is_file()]
    fidelity = [float(row.get("snapshot_qpos_error", 1.0)) for row in accepted]
    lines = [
        "# Targeted ball-impact generation",
        "",
        "Protocol `ball_impact_v2`. New TRAIN launches only. The v1 manifest and sealed validation and final-test pickles were not edited.",
        "",
        "v1 assigned azimuths before generation: TRAIN 0/45/90, VALIDATION 135/180, FINAL_TEST 225/270/315. That restriction is the split table in `training/ball_impact_dataset.py`, not a sampler bug. v2 draws a continuous azimuth in every 30° bin and keeps those sealed splits unchanged.",
        "",
        "Held-out protocol, not yet generated: draw indices with `index % 10 >= 8` are skipped. Indices ending in 8 are reserved for a future validation block and indices ending in 9 for a future final-test block. They are not copied from v1.",
        "",
        "## Counts",
        "",
        f"v2 attempts {len(rows)}. v2 accepted TRAIN {len(accepted)}.",
        f"Funnel {counts}.",
        f"Snapshot files {files}. Log rows whose pickle is missing: {missing or 'none'}.",
        f"v1 pickle counts remain train {files['v1_train']}, validation {files['v1_validation']}, final test {files['v1_final_test']}.",
        f"Restore qpos error max {None if not fidelity else max(fidelity):.3e}." if fidelity else "No accepted v2 snapshot yet.",
        "",
        "## Coverage of accepted v2 impacts",
        "",
        f"Azimuth count {len(azimuths)}. Min {None if not azimuths else round(min(azimuths), 1)}. Max {None if not azimuths else round(max(azimuths), 1)}.",
        f"Speed min/max {None if not speeds else round(min(speeds), 2)} / {None if not speeds else round(max(speeds), 2)} m/s.",
        f"Axial offset min/max {None if not heights else round(min(heights), 4)} / {None if not heights else round(max(heights), 4)} m.",
        "",
        f"This process ran {ran} launches in {round(seconds, 1)} s.",
        "",
        "Cylinder contact is a MuJoCo ball-cylinder geom pair with a contact force in newtons (`mj_contactForce`). A robot or environment contact on an earlier timestep rejects the launch. A same-timestep extra contact is recorded and does not by itself reject a cylinder hit.",
        "",
        "TARGETED BALL-IMPACT GENERATION: PARTIAL",
        "",
        "POLICY TRAINING HAS NOT BEEN STARTED",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def generate(env, attempts):
    rows = load_rows()
    state = load_state()
    prior = v1_specs()
    batch_id = f"b{len(list((RAW / 'batches').glob('b*.json'))) + 1:04d}" if (RAW / "batches").is_dir() else "b0001"
    batch = []
    started = time.perf_counter()
    ran = 0
    draw = int(state.get("next_draw", 0))
    while ran < attempts:
        while draw % 10 >= 8:
            draw += 1
        spec = propose(rows, prior, draw)
        spec["batch_id"] = batch_id
        draw += 1
        if spec.get("skip"):
            row = {
                "category": "near_duplicate_launch",
                "reason": spec["skip"],
                "spec": spec,
                "split": "train",
                "azimuth_deg": spec["azimuth_deg"],
                "speed_mps": spec["speed_mps"],
                "protocol": PROTOCOL,
                "id": None,
                "path": None,
            }
        else:
            n_acc = sum(1 for item in rows if item.get("category") == "valid_impact_recovery_ic") + 1
            spec["attempt_id"] = f"ball_train_v2_{n_acc:04d}"
            evidence, blob = run_episode(env, spec)
            evidence, blob = cylinder_first(evidence, blob)
            evidence["split"] = "train"
            evidence["protocol"] = PROTOCOL
            evidence["azimuth_deg"] = spec["azimuth_deg"]
            evidence["speed_mps"] = spec["speed_mps"]
            evidence["batch_id"] = batch_id
            evidence["draw"] = spec["draw"]
            if blob is None:
                evidence["id"] = None
                evidence["path"] = None
            else:
                evidence["id"] = spec["attempt_id"]
                blob["spec"]["id"] = spec["attempt_id"]
                evidence["path"] = save_blob(blob, evidence)
            row = evidence
        append_row(row)
        rows.append(row)
        batch.append(row)
        ran += 1
        state["next_draw"] = draw
        save_state(state)
        print("V2", row.get("category"), row.get("id"), "az", round(float(spec["azimuth_deg"]), 1), flush=True)
    elapsed = time.perf_counter() - started
    write_batch(batch_id, batch, elapsed)
    write_aggregate(rows, elapsed, ran)
    print("BATCH", batch_id, "accepted_files", file_counts(), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Append targeted TRAIN ball-impact launches")
    parser.add_argument("--attempts", type=int, default=24)
    args = parser.parse_args()
    from training.formal_grpo import load_models
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(
        ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_B_init.pt",
        ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_A_reference.pt",
        device,
    )
    env = make_training_env(device, model=model)
    physics = physics_record(env.sim)
    if not physics["source_constants_match_model"]:
        raise RuntimeError(f"scene constants drifted: {physics}")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "physics.json").write_text(json.dumps(physics, indent=2), encoding="utf-8")
    generate(env, args.attempts)


if __name__ == "__main__":
    main()
