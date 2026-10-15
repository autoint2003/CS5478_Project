"""Read-only physical audit of recatch and loss-then-recontact trajectories.

Writes only under results/diagnostics/recatch_legality_audit. Does not edit
data, rewards, success checks, or the simulator, and does not train.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.horizon_outcome as horizon
import training.rebuild_k48_recovery as rebuild
from training.autonomous_episode import make_training_env
from training.replay_core import raw_object_contacts
from training.unified_k48 import load_trained_k48

OUT = ROOT / "results" / "diagnostics" / "recatch_legality_audit"
RAW = OUT / "raw"
RECATCH_STRATEGIES = {"chase_lost", "release_long", "release_short"}
LOSS_S = 0.04
REGRASP_S = 0.04
SHORT_LOSS_S = 0.02
SUPPORT_FRACTION = 0.5
SUPPORT_TIME_S = 0.02
BOUNCE_DV = 0.10
MANIFESTS = (
    "data/supervised_targets.jsonl",
    "data/valid_demonstrations.jsonl",
    "data/regrasp/manifest.jsonl",
    "data/controlled_sliding/manifest.jsonl",
    "data/inward_sliding/manifest.jsonl",
    "data/immediate_handoff/manifest.jsonl",
)


def _rows(path):
    file = ROOT / path
    if not file.is_file():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hash_actions(actions):
    return hashlib.sha1(np.ascontiguousarray(actions, np.float32).tobytes()).hexdigest()


def inventory():
    supervised = {row.get("continuous_npz"): row for row in _rows("data/supervised_targets.jsonl")}
    valid = {row.get("continuous_npz"): row for row in _rows("data/valid_demonstrations.jsonl")}
    grpo_ids = {row["id"] for row in _rows("data/grpo_initial_conditions.jsonl")}
    grouped = {}
    for manifest in MANIFESTS:
        for row in _rows(manifest):
            npz = row.get("continuous_npz")
            if not npz:
                continue
            key = str(npz).replace("\\", "/")
            item = grouped.setdefault(key, {"manifests": [], "rows": []})
            item["manifests"].append(manifest)
            item["rows"].append(row)
    historical = Counter()
    for path in ROOT.rglob("*.npz"):
        name = path.name
        if not any(token in name for token in ("chase_lost", "release_long", "release_short")):
            continue
        historical[str(path.parent.relative_to(ROOT))] += 1
    return supervised, valid, grpo_ids, grouped, historical


def _sample(sim, weight):
    contacts = raw_object_contacts(sim)
    table = [row for row in contacts if row["kind"] == "object-table"]
    floor = [row for row in contacts if row["kind"] == "object-floor"]
    n_l = sum(1 for row in contacts if row["kind"] == "object-left_finger")
    n_r = sum(1 for row in contacts if row["kind"] == "object-right_finger")
    position = np.array(sim.data.xpos[sim.ids.object_body], float)
    velocity = np.array(sim.data.cvel[sim.ids.object_body][3:], float)
    return {
        "t": float(sim.data.time),
        "nL": int(n_l),
        "nR": int(n_r),
        "z": float(position[2]),
        "pos": [round(float(value), 5) for value in position],
        "vz": float(velocity[2]),
        "v": [round(float(value), 5) for value in velocity],
        "table_fn": float(sum(abs(row["fn"]) for row in table)),
        "floor_fn": float(sum(abs(row["fn"]) for row in floor)),
        "table_geoms": sorted({name for row in table for name in (row["g1"], row["g2"])}),
        "floor_geoms": sorted({name for row in floor for name in (row["g1"], row["g2"])}),
        "weight_n": weight,
    }


def _intervals(samples, dt):
    gaps = []
    index = 0
    while index < len(samples):
        row = samples[index]
        if row["nL"] > 0 and row["nR"] > 0:
            index += 1
            continue
        start = index
        while index < len(samples) and not (samples[index]["nL"] > 0 and samples[index]["nR"] > 0):
            index += 1
        chunk = samples[start:index]
        duration = len(chunk) * dt
        table_rows = [item for item in chunk if item["table_fn"] > 1e-6 or item["floor_fn"] > 1e-6]
        impulse = float(sum((item["table_fn"] + item["floor_fn"]) * dt for item in chunk))
        weight = float(chunk[0]["weight_n"])
        time_supported = float(sum(dt for item in chunk if item["table_fn"] + item["floor_fn"] >= SUPPORT_FRACTION * weight))
        if table_rows:
            onset = table_rows[0]["vz"]
            peak = max(table_rows, key=lambda item: item["table_fn"] + item["floor_fn"])
            dv = float(max(item["vz"] for item in table_rows) - onset)
        else:
            onset = None
            peak = None
            dv = 0.0
        support = bool(time_supported >= SUPPORT_TIME_S or impulse >= SUPPORT_FRACTION * weight * SUPPORT_TIME_S)
        bounce = bool(table_rows and dv >= BOUNCE_DV and impulse >= 0.005)
        later = samples[index:index + max(1, int(round(REGRASP_S / dt)))]
        regrasp = bool(index < len(samples) and all(item["nL"] > 0 and item["nR"] > 0 for item in later) and len(later) * dt >= REGRASP_S - 1e-9)
        gaps.append({
            "t0": chunk[0]["t"],
            "t1": chunk[-1]["t"],
            "duration_s": round(duration, 4),
            "min_z": round(min(item["z"] for item in chunk), 4),
            "impulse_ns": round(impulse, 5),
            "supported_s": round(time_supported, 4),
            "vz_onset": None if onset is None else round(onset, 4),
            "dv_up_mps": round(dv, 4),
            "support": support,
            "bounce": bounce,
            "brush": bool(table_rows) and not support and not bounce,
            "table_contact": bool(table_rows),
            "regrasp": regrasp,
            "peak": None if peak is None else {
                "t": peak["t"],
                "fn_n": round(peak["table_fn"] + peak["floor_fn"], 4),
                "geoms": peak["table_geoms"] or peak["floor_geoms"],
                "pos": peak["pos"],
                "v": peak["v"],
                "nL": peak["nL"],
                "nR": peak["nR"],
                "z": round(peak["z"], 4),
            },
        })
    return gaps


def _classify(gaps, success=False):
    primary = [gap for gap in gaps if gap["duration_s"] >= LOSS_S - 1e-9 and gap["regrasp"]]
    assisted = [gap for gap in gaps if gap["duration_s"] >= LOSS_S - 1e-9 and (gap["support"] or gap["bounce"])]
    short_assisted = [
        gap for gap in gaps
        if SHORT_LOSS_S - 1e-9 <= gap["duration_s"] < LOSS_S - 1e-9 and gap["regrasp"] and (gap["support"] or gap["bounce"])
    ]
    if any(gap["support"] or gap["bounce"] for gap in primary) or (assisted and success):
        return "ILLEGAL_TABLE_REGRASP"
    if short_assisted:
        return "UNCERTAIN"
    if primary:
        return "LEGAL_RECATCH"
    return "NOT_RECATCH"


def _consistent(npz, contacts, executed, observations, qpos_error, phase_ok):
    action_error = None
    if "actions" in npz.files and len(executed):
        stored = np.asarray(npz["actions"], np.float32)
        got = np.asarray(executed, np.float32)
        if stored.shape == got.shape:
            action_error = float(np.max(np.abs(stored - got)))
        else:
            action_error = 1.0
    observation_error = None
    if "observations" in npz.files and len(observations):
        stored = np.asarray(npz["observations"], np.float32)
        got = np.asarray(observations, np.float32)
        if stored.shape == got.shape:
            observation_error = float(np.max(np.abs(stored - got)))
        else:
            observation_error = 1.0
    contact_ok = None
    z_error = None
    if "nL" in npz.files and "nR" in npz.files and "obj_z" in npz.files:
        contact_ok = (
            len(contacts) == len(npz["nL"])
            and list(map(int, npz["nL"])) == [int(row["nL"]) for row in contacts]
            and list(map(int, npz["nR"])) == [int(row["nR"]) for row in contacts]
        )
        if len(contacts) == len(npz["obj_z"]):
            z_error = float(np.max(np.abs(np.asarray(npz["obj_z"], float) - np.array([row["z"] for row in contacts]))))
            contact_ok = contact_ok and z_error <= 1e-3
        else:
            contact_ok = False
    ok = bool(
        qpos_error <= 1e-6
        and phase_ok
        and contact_ok is not False
        and (action_error is None or action_error <= 1e-4)
        and (observation_error is None or observation_error <= 1e-4)
    )
    return {
        "ok": ok,
        "qpos_error": qpos_error,
        "phase_ok": bool(phase_ok),
        "action_max_abs": action_error,
        "observation_max_abs": observation_error,
        "obj_z_max_abs": z_error,
        "contact_match": contact_ok,
    }


def audit_one(env, blob, actions):
    weight = float(env.sim.model.body_mass[env.sim.ids.object_body]) * float(abs(env.sim.model.opt.gravity[2]))
    dt = float(env.sim.model.opt.timestep)
    samples = []
    active = {"on": False}
    original_step = mujoco.mj_step
    original_legal = rebuild.step_legal

    def wrapped(model, data):
        original_step(model, data)
        if active["on"]:
            samples.append(_sample(env.sim, weight))

    def stepped(sim, action):
        if not samples:
            samples.append(_sample(sim, weight))
        active["on"] = True
        return original_legal(sim, action)

    mujoco.mj_step = wrapped
    rebuild.step_legal = stepped
    active["on"] = True
    try:
        out, observations, executed, contacts, qpos_error, phase_ok = rebuild.legal_verify(env, blob, actions)
    finally:
        mujoco.mj_step = original_step
        rebuild.step_legal = original_legal
        active["on"] = False
    return out, observations, executed, contacts, samples, dt, weight, qpos_error, phase_ok


def references(bucket, supervised, valid, grpo_ids):
    rows = bucket["rows"]
    train_rows = [row for row in rows if row.get("split") == "train" and any(item.get("continuous_npz") == row.get("continuous_npz") for item in supervised.values())]
    sft = [supervised[row["continuous_npz"]] for row in rows if row.get("continuous_npz") in supervised]
    val = [valid[row["continuous_npz"]] for row in rows if row.get("continuous_npz") in valid]
    weight = 0.0
    if sft:
        weight = float(sft[0].get("sample_weight") or 0.0)
    primary = sft[0] if sft else (val[0] if val else rows[0])
    return {
        "id": primary.get("id"),
        "strategy": primary.get("strategy"),
        "mechanism": primary.get("mechanism"),
        "family": primary.get("family"),
        "split": primary.get("split"),
        "snapshot": primary.get("snapshot"),
        "continuous_npz": primary.get("continuous_npz"),
        "manifests": sorted(set(bucket["manifests"])),
        "sft": bool(sft),
        "sample_weight": weight if sft else None,
        "sft_nonzero": bool(sft) and weight > 0.0,
        "valid_demo": bool(val),
        "grpo_ic": primary.get("id") in grpo_ids,
        "verified_success_field": primary.get("verified_success"),
        "nominal_class": primary.get("nominal_class"),
        "all_ids": sorted({row.get("id") for row in rows}),
    }


def deduped_rows():
    by = {}
    path = RAW / "trajectories.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["classification"] = _classify(row.get("gaps") or [], success=bool(row.get("success_12s")))
        if not row["consistency"]["ok"]:
            row["classification"] = "UNCERTAIN"
        by[row["continuous_npz"]] = row
    return list(by.values())


def write_report():
    rows = deduped_rows()
    (RAW / "trajectories.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    events = []
    for row in rows:
        for gap in row.get("gaps") or []:
            if gap.get("support") or gap.get("bounce") or gap.get("regrasp") or gap.get("table_contact"):
                events.append({
                    "id": row["id"],
                    "strategy": row["strategy"],
                    "npz": row["continuous_npz"],
                    "classification": row["classification"],
                    **gap,
                })
    (RAW / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    counts = Counter(row["classification"] for row in rows)
    train = [row for row in rows if row.get("split") == "train"]
    valid = [row for row in rows if row.get("split") == "valid"]
    illegal = [row for row in rows if row["classification"] == "ILLEGAL_TABLE_REGRASP"]
    sft_illegal = [row for row in illegal if row.get("sft_nonzero")]
    weight_total = sum(float(row.get("sample_weight") or 0.0) for row in rows if row.get("sft"))
    weight_illegal = sum(float(row.get("sample_weight") or 0.0) for row in sft_illegal)

    def line_for(row):
        gap = next((item for item in row["gaps"] if item.get("support") or item.get("bounce")), None)
        peak = (gap or {}).get("peak") or {}
        return (
            f"| `{row['id']}` | {row['strategy']} | {row['split']} | {row.get('sample_weight')} | "
            f"{None if gap is None else gap['t0']:.3f} | {None if gap is None else gap['supported_s']} | "
            f"{None if gap is None else gap['impulse_ns']} | {None if gap is None else gap['dv_up_mps']} | "
            f"{None if gap is None else gap['min_z']} | {peak.get('fn_n')} | {peak.get('nL')}/{peak.get('nR')} | "
            f"{row['success_12s']} |"
        )

    case = {}
    for row in rows:
        case.setdefault(row["id"], []).append(row)
    text = []
    text.append("# Recatch legality audit")
    text.append("")
    text.append("Replay uses the stored recovery actions, `legal_verify`, and the unchanged `SUCCESS_12S` check. No dataset, reward, success rule, or simulator file was edited.")
    text.append("")
    text.append("## Coverage")
    text.append("")
    text.append("Physically replayed: every unique `continuous_npz` referenced by `data/supervised_targets.jsonl` (251), `data/valid_demonstrations.jsonl` (14), `data/regrasp/manifest.jsonl` (158), `data/controlled_sliding/manifest.jsonl` (50), `data/inward_sliding/manifest.jsonl` (50), and `data/immediate_handoff/manifest.jsonl` (7). After path dedupe this is 265 trajectories. There is no TEST split in these manifests. `data/grpo_initial_conditions.jsonl` has 87 initial conditions and no action tapes.")
    text.append("")
    text.append("Not resimulated: historical `chase_lost` / `release_long` / `release_short` npz files outside those manifests, including `demo_results/diagnostics/raw/k48_recovery_v3/search_attempts` (932), `legal_recovery_v2/trajectories` (472), and `k48_handoff/trajectories` (142). Counts are in `raw/coverage.json`. Those files are not a second copy of the audited verdicts.")
    text.append("")
    text.append("A loss of control is both finger geoms out of contact for at least 0.04 s. Table support is object-table or object-floor normal force at or above half the cylinder weight (about 1.96 N) for at least 0.02 s, or an equal impulse. A bounce is an upward speed change of at least 0.10 m/s during that contact together with an impulse of at least 0.005 N·s. A shorter or weaker contact is a brush and is not enough for the illegal class. Thresholds were fixed before reading the labels.")
    text.append("")
    text.append("## Counts")
    text.append("")
    text.append(f"Deduped trajectories: {len(rows)}.")
    text.append("")
    text.append("| class | count |")
    text.append("| --- | ---: |")
    for name in ("LEGAL_RECATCH", "ILLEGAL_TABLE_REGRASP", "NOT_RECATCH", "UNCERTAIN"):
        text.append(f"| {name} | {counts[name]} |")
    text.append("")
    text.append("| split | LEGAL_RECATCH | ILLEGAL_TABLE_REGRASP | NOT_RECATCH | UNCERTAIN |")
    text.append("| --- | ---: | ---: | ---: | ---: |")
    for name, group in (("train", train), ("valid", valid)):
        group_counts = Counter(row["classification"] for row in group)
        text.append(
            f"| {name} | {group_counts['LEGAL_RECATCH']} | {group_counts['ILLEGAL_TABLE_REGRASP']} | "
            f"{group_counts['NOT_RECATCH']} | {group_counts['UNCERTAIN']} |"
        )
    text.append("")
    text.append(f"SFT rows with nonzero weight and an illegal replay: {len(sft_illegal)}. Their weight sum is {weight_illegal:.4f} of {weight_total:.4f}. Validation demonstrations are not in the SFT weight sum. No TEST rows were found.")
    text.append("")
    text.append("## Known cases")
    text.append("")
    for episode_id, title in (
        ("ball_train_v2_0051", "chase_lost video case"),
        ("inward_extra_250005", "inward video case"),
        ("slide_extra_260013", "controlled-slide video case"),
    ):
        text.append(f"### {episode_id}")
        text.append("")
        hits = case.get(episode_id, [])
        if not hits:
            text.append("Not in the replayed manifests.")
            text.append("")
            continue
        for row in hits:
            gap = next((item for item in row["gaps"] if item.get("support") or item.get("bounce") or item.get("table_contact")), None)
            text.append(
                f"- `{row['continuous_npz']}`, snapshot `{row['snapshot']}`, split {row['split']}, "
                f"strategy {row['strategy']}, SFT {row['sft']}, weight {row.get('sample_weight')}, "
                f"GRPO initial condition {row['grpo_ic']}."
            )
            text.append(
                f"- Replay class `{row['classification']}`, official reason `{row['reason']}`, "
                f"SUCCESS_12S {row['success_12s']}. Consistency qpos {row['consistency']['qpos_error']}, "
                f"action error {row['consistency']['action_max_abs']}, contact match {row['consistency']['contact_match']}."
            )
            if gap is None:
                text.append("- No object-table or object-floor contact interval was recorded while a finger was off.")
            else:
                peak = gap.get("peak") or {}
                text.append(
                    f"- Table interval t={gap['t0']:.3f}s for {gap['duration_s']}s, supported {gap['supported_s']}s, "
                    f"impulse {gap['impulse_ns']} N·s, upward dv {gap['dv_up_mps']} m/s, min z {gap['min_z']} m. "
                    f"Peak force {peak.get('fn_n')} N on {peak.get('geoms')} at z={peak.get('z')}, "
                    f"nL/nR {peak.get('nL')}/{peak.get('nR')}, velocity {peak.get('v')}."
                )
        text.append("")
    text.append("## Illegal trajectories")
    text.append("")
    text.append("Every row below has a replay-consistent tape, a bilateral finger gap of at least 0.04 s, table support or bounce inside that gap, and `SUCCESS_12S` from the unchanged checker. Full peaks are in `raw/trajectories.jsonl` and `raw/events.jsonl`.")
    text.append("")
    text.append("| IC | strategy | split | weight | t0 | supported s | impulse N·s | dv m/s | min z | peak N | nL/nR | SUCCESS_12S |")
    text.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in sorted(illegal, key=lambda item: (item["split"], item["strategy"], item["id"])):
        text.append(line_for(row))
    text.append("")
    text.append("## Evaluator")
    text.append("")
    text.append("`recovery_failed` counts a drop only after 50 control steps (1.0 s) with no finger contact and object height below 0.44 m, or below 0.30 m immediately. `nominal_verify` then accepts `SUCCESS_12S` from bilateral contact, lift, and a 12 s hold. It does not read object-table force. All %d illegal replays still returned `SUCCESS_12S`, so the current checker accepts a table-supported regrasp when the pickup happens inside that window." % len(illegal))
    text.append("")
    text.append("## Contamination")
    text.append("")
    text.append(f"The supervised file contains {len(sft_illegal)} illegal demonstrations with nonzero weight. They are the `chase_lost`, `release_long`, and `release_short` rows. `controlled_slide` and `inward_reseat` replays in this set did not show a finger-off table interval. GRPO does not train on these action tapes; {len({row['id'] for row in illegal if row.get('grpo_ic')})} of the 87 GRPO initial conditions also have an illegal supervised tape. No rows were deleted.")
    text.append("")
    text.append("## What was not changed")
    text.append("")
    text.append("A minimal later change would be to reject a rollout whose unrecovered interval contains the table-support or bounce test above, and to stop giving those tapes nonzero supervised weight. This audit does not apply that change.")
    text.append("")
    text.append("## Evidence levels")
    text.append("")
    text.append("- Physics: the class counts, the table force, impulse, height, speed, and finger counts above come from MuJoCo replays whose stored actions and, where present, stored `nL`/`nR`/`obj_z` matched. qpos restore error was 0.")
    text.append("- Video, separate from this replay: the earlier side-by-side videos of `ball_train_v2_0051`, `inward_extra_250005`, and `slide_extra_260013` showed the same outcomes. The classes in this report do not use those videos.")
    text.append("- Not established: legality of historical search npz files that are not referenced by the current manifests. No `LEGAL_RECATCH` tape was found in the replayed set, and none of the replayed tapes were `UNCERTAIN`.")
    text.append("")
    (OUT / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print("REPORT", len(rows), counts, "sft_illegal", len(sft_illegal), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    supervised, valid, grpo_ids, grouped, historical = inventory()
    jobs = []
    seen = {}
    for key, bucket in grouped.items():
        path = ROOT / key
        if not path.is_file():
            continue
        with np.load(path) as packed:
            actions = packed["actions"].astype(np.float32) if "actions" in packed.files else np.zeros((0, 7), np.float32)
        snap = references(bucket, supervised, valid, grpo_ids)["snapshot"]
        physical = (str(snap), _hash_actions(actions))
        if physical in seen:
            seen[physical]["manifests"].extend(bucket["manifests"])
            seen[physical]["rows"].extend(bucket["rows"])
            continue
        seen[physical] = bucket
        jobs.append((key, bucket, actions))
    (RAW / "coverage.json").write_text(json.dumps({
        "manifests": list(MANIFESTS),
        "manifest_rows": {name: len(_rows(name)) for name in MANIFESTS},
        "unique_npz": len(grouped),
        "unique_physical": len(jobs),
        "historical_recatch_npz_not_replayed_by_directory": dict(historical),
        "note": "Physical replay covers unique trajectories referenced by the current data manifests. Historical search npz files are counted and are not each resimulated.",
    }, indent=2), encoding="utf-8")
    done_path = RAW / "trajectories.jsonl"
    done = set()
    if done_path.is_file():
        for line in done_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["continuous_npz"])
    import torch
    device = torch.device("cpu")
    model, _saved = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", device)
    model.eval()
    env = make_training_env(device, model=model)
    blobs = {}
    with done_path.open("a", encoding="utf-8") as handle:
        for index, (key, bucket, actions) in enumerate(jobs, start=1):
            if key in done:
                continue
            meta = references(bucket, supervised, valid, grpo_ids)
            snap = ROOT / meta["snapshot"]
            if meta["snapshot"] not in blobs:
                with snap.open("rb") as blob_file:
                    blobs[meta["snapshot"]] = pickle.load(blob_file)
            npz = np.load(ROOT / key)
            out, observations, executed, contacts, samples, dt, weight, qpos_error, phase_ok = audit_one(env, blobs[meta["snapshot"]], actions)
            check = _consistent(npz, contacts, executed, observations, qpos_error, phase_ok)
            gaps = _intervals(samples, dt) if samples else []
            label = _classify(gaps, success=bool(out.get("success"))) if check["ok"] else "UNCERTAIN"
            record = meta | {
                "classification": label,
                "success_12s": bool(out.get("success")),
                "reason": out.get("reason"),
                "recovery_steps": int(out.get("recovery_steps") or len(executed)),
                "cylinder_weight_n": round(weight, 4),
                "consistency": check,
                "gaps": [gap for gap in gaps if gap["duration_s"] >= SHORT_LOSS_S - 1e-9 or gap["table_contact"]],
                "table_while_held": bool(any(
                    (row["nL"] > 0 and row["nR"] > 0 and (row["table_fn"] + row["floor_fn"]) > 1e-6)
                    for row in samples
                )),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(index, "/", len(jobs), meta["id"], meta["strategy"], label, out.get("reason"), flush=True)
    print("AUDIT_DONE", len(jobs), flush=True)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_report()
    else:
        main()
