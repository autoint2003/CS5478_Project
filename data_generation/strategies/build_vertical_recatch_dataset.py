"""Build data/vertical_recatch from original sliding initial conditions.

Does not train, and does not write into data/recatch or data/regrasp.
Snapshots stay where the sliding manifests already point. A new copy would
overwrite those shared files.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.airborne_obs import observe_airborne
from envs.observable_obs import ObservableObsState
from training.history_aware_critic_ablation import restore_capture
from envs.horizon_outcome import nominal_verify
from data_generation.oracle_search import representability
from envs.step_legal import step_legal
from training.unified_k47 import load_codebook
from data_generation.strategies.vertical_recatch_search import (
    DT,
    OUT,
    _finish,
    _summary,
    load_blob,
    make_env,
    rollout,
    specs,
)

DATA = ROOT / "data" / "vertical_recatch"


def _manifest_rows():
    rows = []
    for folder in ("controlled_sliding", "inward_sliding"):
        path = ROOT / "data" / folder / "manifest.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["pool"] = folder
                rows.append(row)
    path = ROOT / "data" / "grpo_initial_conditions.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["pool"] = "grpo"
            row.setdefault("strategy", None)
            rows.append(row)
    return rows


def _family(spec):
    azimuth = int(round(float(spec["azimuth_deg"]) / 45.0) * 45) % 360
    speed = int(round(float(spec["speed_mps"])))
    return f"az{azimuth:03d}_v{speed:d}"


def _split_of(family, families):
    index = families.index(family)
    if index % 5 == 0:
        return "test"
    if index % 5 == 1:
        return "valid"
    return "train"


def _choose(rows):
    """Spread azimuths before any recovery rollout."""
    chosen = []
    seen = set()
    blobs = {}

    def take(row):
        if row["id"] in seen:
            return False
        blob = load_blob(ROOT / row["snapshot"])
        spec = blob.get("spec") or {}
        if "azimuth_deg" not in spec:
            return False
        seen.add(row["id"])
        blobs[row["id"]] = blob
        chosen.append(row)
        return True

    per_bin = Counter()
    for row in rows:
        if row["pool"] == "grpo" or len(chosen) >= 16:
            continue
        bin_name = None
        blob = load_blob(ROOT / row["snapshot"])
        spec = blob.get("spec") or {}
        if "azimuth_deg" not in spec or row["id"] in seen:
            continue
        bin_name = int(round(float(spec["azimuth_deg"]) / 45.0) * 45) % 360
        if per_bin[bin_name] >= 2:
            continue
        per_bin[bin_name] += 1
        seen.add(row["id"])
        blobs[row["id"]] = blob
        chosen.append(row)
    grpo_n = 0
    for row in rows:
        if row["pool"] != "grpo" or grpo_n >= 8:
            continue
        if take(row):
            grpo_n += 1
    if "ball_train_v2_0070" not in seen:
        for row in rows:
            if row["id"] == "ball_train_v2_0070" and take(row):
                break
    families = sorted({_family(blobs[row["id"]]["spec"]) for row in chosen})
    for row in chosen:
        row["impact_family"] = _family(blobs[row["id"]]["spec"])
        row["split"] = _split_of(row["impact_family"], families)
    return chosen, blobs, families


def _replay(env, blob, book, actions):
    restore_capture(env, blob["capture"])
    sim = env.sim
    hist = ObservableObsState(DT)
    observations = []
    for action in np.asarray(actions, np.float32):
        observations.append(np.asarray(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist), np.float32))
        step_legal(sim, action)
    handoff = np.asarray(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist), np.float32)
    out, extra = _finish(sim, blob["ref"])
    packed = representability(np.asarray(actions, np.float32), book)
    return out, extra, observations, handoff, packed


def _nominal(env, blob):
    restore_capture(env, blob["capture"])

    class Holder:
        pass

    holder = Holder()
    holder.sim = env.sim
    out = nominal_verify(holder, blob["ref"])
    return out.get("reason")


def _slide(env, blob, row):
    path = row.get("continuous_npz")
    if not path:
        return "NO_SLIDING_TAPE"
    with np.load(ROOT / path) as archive:
        if "actions" not in archive.files:
            return "NO_SLIDING_TAPE"
        actions = np.asarray(archive["actions"], np.float32)
    restore_capture(env, blob["capture"])

    class Holder:
        pass

    holder = Holder()
    holder.sim = env.sim
    for action in actions:
        step_legal(env.sim, action)
    out = nominal_verify(holder, blob["ref"])
    return out.get("reason")


def _accept(summary):
    return (
        summary["mode"] == "handoff"
        and summary["t_off"] is not None
        and summary["pad_class"] == "LOWER_VERTICAL_PAD"
        and summary["min_bottom"] > 0.40
    )


def _different(actions, previous):
    if previous is None or len(actions) < 5 or len(previous) < 5:
        return True
    n = min(len(actions), len(previous))
    return float(np.mean(np.abs(actions[:n] - previous[:n]))) > 0.04


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (DATA / split).mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    book = load_codebook()
    chosen, blobs, families = _choose(_manifest_rows())
    print("families", families, flush=True)
    print("n", len(chosen), Counter(row["split"] for row in chosen), flush=True)
    manifest = []
    boundary = []
    for row in chosen:
        blob = blobs[row["id"]]
        nominal = _nominal(env, blob)
        slide = _slide(env, blob, row)
        print(row["id"], row["split"], row["impact_family"], "nominal", nominal, "slide", slide, flush=True)
        kept = None
        saved = 0
        for spec in specs():
            result = rollout(env, blob, book, spec)
            summary = _summary(result)
            summary.update({
                "id": row["id"],
                "strategy": spec["name"],
                "nominal": nominal,
                "slide": slide,
                "split": row["split"],
                "family": row["impact_family"],
            })
            if not _accept(summary) or not _different(result["actions"], kept):
                summary["formal"] = None
                boundary.append(summary)
                print(" ", spec["name"], summary["mode"], "off", summary["t_off"], "pad", summary["t_pad"], "minb", summary["min_bottom"], flush=True)
                continue
            out, extra, observations, handoff, packed = _replay(env, blob, book, result["actions"])
            summary["formal"] = out.get("reason")
            summary["extension"] = extra
            summary["max_linf"] = None if packed["max_linf"] is None else round(float(packed["max_linf"]), 4)
            ok = bool(out.get("success") and extra == "HELD_EXTENSION" and summary["max_linf"] is not None and summary["max_linf"] <= 0.25)
            print(" ", spec["name"], summary["formal"], extra, "linf", summary["max_linf"], "ok", ok, flush=True)
            if not ok or len(observations) != len(result["actions"]):
                boundary.append(summary)
                continue
            split = row["split"]
            stem = f"{row['id']}__{spec['name']}"
            continuous = DATA / split / f"{stem}.npz"
            tokens = DATA / split / f"{stem}__tokens.npz"
            np.savez_compressed(
                continuous,
                actions=np.asarray(result["actions"], np.float32),
                observations=np.stack(observations, axis=0),
                handoff_obs=handoff,
            )
            np.savez_compressed(
                tokens,
                tokens=np.asarray(packed["tokens"], np.int64),
                residuals=np.asarray(result["actions"], np.float32) - book[np.asarray(packed["tokens"], np.int64)],
            )
            manifest.append({
                "id": row["id"],
                "split": split,
                "strategy": spec["name"],
                "family": "vertical_recatch",
                "impact_family": row["impact_family"],
                "mechanism": "open_overchase_close_brake",
                "continuous_npz": str(continuous.relative_to(ROOT)).replace("\\", "/"),
                "token_residual_npz": str(tokens.relative_to(ROOT)).replace("\\", "/"),
                "snapshot": row["snapshot"].replace("\\", "/"),
                "source_snapshot": row["snapshot"].replace("\\", "/"),
                "handoff_label": "SUCCESS_12S",
                "nominal_class": nominal,
                "sliding_class": slide,
                "provenance": "vertical_recatch",
                "source_kind": row["pool"],
                "sample_weight": 1.0,
                "verified_success": True,
                "azimuth_deg": round(float(blob["spec"]["azimuth_deg"]), 3),
                "speed_mps": round(float(blob["spec"]["speed_mps"]), 3),
                "z_off_m": round(float(blob["spec"]["z_off_m"]), 4),
                "phi_deg": round(float(blob["spec"]["phi_deg"]), 3),
                "k48_linf": summary["max_linf"],
                "extension": extra,
                "formal": out.get("reason"),
                "t_off": summary["t_off"],
                "t_re": summary["t_re"],
                "t_pad": summary["t_pad"],
                "min_bottom": summary["min_bottom"],
                "end_bottom": summary["end_bottom"],
                "max_down": summary["max_down"],
                "max_open": summary["max_open"],
                "max_right": summary["max_right"],
                "vh0": summary["vh0"],
            })
            kept = result["actions"]
            saved += 1
            if saved >= 2:
                break
        if saved == 0:
            boundary.append({
                "id": row["id"],
                "split": row["split"],
                "nominal": nominal,
                "slide": slide,
                "family": row["impact_family"],
                "saved": 0,
            })
    with (DATA / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item) + "\n")
    with (DATA / "boundary.jsonl").open("w", encoding="utf-8") as handle:
        for item in boundary:
            handle.write(json.dumps(item) + "\n")
    counts = Counter(item["split"] for item in manifest)
    print("saved", len(manifest), dict(counts), flush=True)


if __name__ == "__main__":
    main()
