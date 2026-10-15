"""Read-only pre-training audit of data/. Does not train or write into data/."""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.validation.impact_labels import near
from data_generation.rebuild_k48_recovery import _load_jsonl, _scenario_key
from training.k47_codebook import CAP, episode_arrays, load_codebook
from training.unified_k48 import HANDOFF, episode_tensors

DATA = ROOT / "data"
OUT = ROOT / "results/archive/demo_results/diagnostics/raw/pretrain_audit"
RECATCH = {"chase_lost", "release_long", "release_short"}
SLIDE = {"controlled_slide"}
INWARD = {"inward_reseat"}
DT = 0.02
LIMIT_S = 4.0


def _rows(path):
    return _load_jsonl(path)


def _family(strategy):
    if strategy in RECATCH:
        return "recatch"
    if strategy in SLIDE:
        return "controlled_sliding"
    if strategy in INWARD:
        return "inward_sliding"
    if strategy == "immediate_handoff":
        return "control"
    return strategy or "unknown"


def _spec_from_row(row):
    if row.get("azimuth_deg") is None or row.get("speed_mps") is None:
        return None
    return {
        "azimuth_deg": row["azimuth_deg"],
        "speed_mps": row["speed_mps"],
        "z_off_m": row.get("z_off_m", 0.0),
        "phi_deg": row.get("phi_deg", 0.0),
        "delay_s": row.get("delay_s", 0.0),
    }


def _spec_from_blob(path):
    with path.open("rb") as handle:
        blob = pickle.load(handle)
    spec = blob.get("spec") if isinstance(blob, dict) else None
    if isinstance(spec, dict) and spec.get("azimuth_deg") is not None:
        return spec
    ref = blob.get("ref") if isinstance(blob, dict) else None
    if isinstance(ref, dict) and ref.get("azimuth_deg") is not None:
        return ref
    return None


def static_report():
    book = load_codebook()
    train = _rows(DATA / "supervised_targets.jsonl")
    valid = _rows(DATA / "valid_demonstrations.jsonl")
    grpo = _rows(DATA / "grpo_initial_conditions.jsonl")
    recovery_eval = _rows(DATA / "valid_recovery.jsonl")
    candidates = {row["id"]: row for row in _rows(ROOT / "results/archive/demo_results/diagnostics/raw/k48_recovery_v3/candidates.jsonl")}
    demos = [("train", row) for row in train] + [("valid", row) for row in valid]
    problems = []

    # Encoding, duration, HANDOFF assembly.
    encode = Counter()
    labels = Counter()
    families = Counter()
    over_limit = []
    missing_handoff_obs = []
    token_mismatch = []
    residual_mismatch = []
    action_mismatch = []
    handoff_in_motion = []
    obs_bad = []
    weight_by_family = defaultdict(float)
    weight_by_mechanism = defaultdict(float)
    rows_by_family = Counter()
    for split, row in demos:
        if split != row.get("split"):
            problems.append(f"split field {row.get('id')} {row.get('split')} != file {split}")
        family = _family(row.get("strategy"))
        labels[row.get("handoff_label")] += 1
        families[family] += 1
        if row.get("nominal_class") == "RECOVERY_CANDIDATE":
            rows_by_family[family] += 1
            weight_by_family[family] += float(row.get("sample_weight") or 0.0)
            weight_by_mechanism[row.get("mechanism") or "unknown"] += float(row.get("sample_weight") or 0.0)
        path = ROOT / row["continuous_npz"]
        packed = np.load(path)
        if row.get("handoff_label") == "immediate_handoff":
            obs, prev, tokens, residuals = episode_tensors(row, book)
            if list(tokens) != [HANDOFF] or residuals.shape != (1, 7):
                token_mismatch.append(row["id"])
            continue
        if "actions" not in packed.files:
            obs_bad.append(f"{row['id']} npz has no actions")
            continue
        actions = np.asarray(packed["actions"], np.float32)
        seconds = len(actions) * DT
        if row.get("nominal_class") == "RECOVERY_CANDIDATE" and (len(actions) > 200 or seconds > LIMIT_S + 1e-9):
            over_limit.append((row["id"], row.get("strategy"), len(actions), round(seconds, 3)))
        if row.get("handoff_label") == "early_handoff" and "handoff_obs" not in packed.files:
            missing_handoff_obs.append(row["id"])
        try:
            obs, prev, actions_ep, tokens, residuals = episode_arrays(row, book)
        except RuntimeError as exc:
            obs_bad.append(str(exc))
            continue
        if int(np.any(tokens == HANDOFF)):
            handoff_in_motion.append(row["id"])
        decoded = book[tokens] + residuals
        if float(np.max(np.abs(decoded - actions_ep))) > 1e-5:
            action_mismatch.append((row["id"], row.get("strategy"), float(np.max(np.abs(decoded - actions_ep)))))
        if float(np.max(np.abs(residuals))) > CAP + 1e-5:
            residual_mismatch.append(row["id"])
        nearest = np.argmax(-np.max(np.abs(actions_ep[:, None, :] - book[None, :, :]), axis=2), axis=1)
        linf = np.max(np.abs(actions_ep - book[nearest]), axis=1)
        if int(np.any(tokens != nearest)) or float(np.max(linf)) > CAP + 1e-5:
            token_mismatch.append((row["id"], row.get("strategy"), float(np.max(linf))))
        assembled = episode_tensors(row, book)
        if row.get("handoff_label") == "early_handoff":
            if int(assembled[2][-1]) != HANDOFF or int(np.sum(assembled[2] == HANDOFF)) != 1:
                problems.append(f"early handoff token assembly {row['id']}")
        elif int(np.sum(assembled[2] == HANDOFF)) != 0:
            problems.append(f"recorded_end gained a handoff token {row['id']}")

    # Unique ICs and overlap on the 251 train rows.
    recovery_train = [row for row in train if row.get("nominal_class") == "RECOVERY_CANDIDATE"]
    control_train = [row for row in train if row.get("nominal_class") != "RECOVERY_CANDIDATE"]
    by_id = defaultdict(set)
    for row in recovery_train:
        by_id[row["id"]].add(_family(row.get("strategy")))
    family_ids = {
        "recatch": {row["id"] for row in recovery_train if _family(row.get("strategy")) == "recatch"},
        "controlled_sliding": {row["id"] for row in recovery_train if _family(row.get("strategy")) == "controlled_sliding"},
        "inward_sliding": {row["id"] for row in recovery_train if _family(row.get("strategy")) == "inward_sliding"},
    }
    overlap = {
        "recatch_and_controlled": sorted(family_ids["recatch"] & family_ids["controlled_sliding"]),
        "recatch_and_inward": sorted(family_ids["recatch"] & family_ids["inward_sliding"]),
        "controlled_and_inward": sorted(family_ids["controlled_sliding"] & family_ids["inward_sliding"]),
        "all_three": sorted(family_ids["recatch"] & family_ids["controlled_sliding"] & family_ids["inward_sliding"]),
    }

    # Scenario isolation.
    spec_for = {}
    for split, row in demos:
        spec = _spec_from_row(row) or _spec_from_row(candidates.get(row["id"], {}))
        if spec is None and (ROOT / row["snapshot"]).is_file():
            spec = _spec_from_blob(ROOT / row["snapshot"])
        if spec is not None:
            spec_for[(split, row["id"], row.get("strategy"))] = spec
    train_specs = []
    valid_specs = []
    seen_train = set()
    seen_valid = set()
    for (split, episode_id, _strategy), spec in spec_for.items():
        key = (split, episode_id)
        if split == "train" and key not in seen_train:
            seen_train.add(key)
            train_specs.append((episode_id, spec))
        if split == "valid" and key not in seen_valid:
            seen_valid.add(key)
            valid_specs.append((episode_id, spec))
    scenario_hits = []
    for vid, vspec in valid_specs:
        vkey = _scenario_key(vspec)
        for tid, tspec in train_specs:
            if _scenario_key(tspec) == vkey or near(vspec, tspec):
                scenario_hits.append({"valid": vid, "train": tid, "same_key": _scenario_key(tspec) == vkey})
    path_cross = []
    id_cross = []
    train_ids = {row["id"] for row in train}
    valid_ids = {row["id"] for row in valid}
    id_cross = sorted(train_ids & valid_ids)
    for split, row in demos:
        snap = str(row.get("snapshot") or "").replace("\\", "/")
        if split == "train" and "/valid/" in snap:
            path_cross.append(row["id"])
        if split == "valid" and "/train/" in snap:
            path_cross.append(row["id"])
        if not (ROOT / row["snapshot"]).is_file():
            problems.append(f"missing snapshot {row['id']} {snap}")
        if not (ROOT / row["continuous_npz"]).is_file():
            problems.append(f"missing npz {row['id']}")

    # GRPO membership.
    demo_success_ids = {row["id"] for row in recovery_train}
    valid_success_ids = {row["id"] for row in valid if row.get("nominal_class") == "RECOVERY_CANDIDATE"}
    grpo_nominal = [row["id"] for row in grpo if row.get("nominal_class") == "NOMINAL_SOLVABLE" or row["id"] in {item["id"] for item in control_train}]
    grpo_proven = [row["id"] for row in grpo if row["id"] in demo_success_ids]
    grpo_unproven = [row["id"] for row in grpo if row["id"] not in demo_success_ids]
    grpo_in_valid = [row["id"] for row in grpo if row["id"] in valid_ids or row.get("split") == "valid"]
    grpo_missing_snap = [row["id"] for row in grpo if not (ROOT / row["snapshot"]).is_file()]
    eval_in_train = [row["id"] for row in recovery_eval if row["id"] in train_ids]

    report = {
        "counts": {
            "sft_rows": len(train),
            "sft_recovery_rows": len(recovery_train),
            "sft_control_rows": len(control_train),
            "sft_unique_recovery_ics": len(by_id),
            "valid_rows": len(valid),
            "valid_unique_ics": len(valid_ids),
            "grpo": len(grpo),
            "valid_recovery": len(recovery_eval),
        },
        "family_rows_train_recovery": {key: sum(1 for row in recovery_train if _family(row.get("strategy")) == key) for key in ("recatch", "controlled_sliding", "inward_sliding")},
        "family_unique_ics": {key: len(value) for key, value in family_ids.items()},
        "overlap_counts": {key: len(value) for key, value in overlap.items()},
        "overlap_ids": overlap,
        "handoff_labels": dict(labels),
        "encode": {
            "over_4s": over_limit,
            "missing_handoff_obs": missing_handoff_obs,
            "token_or_linf": token_mismatch[:20],
            "token_or_linf_count": len(token_mismatch),
            "action_decode_mismatch_count": len(action_mismatch),
            "residual_over_cap": residual_mismatch,
            "handoff_token_inside_motion_file": handoff_in_motion,
            "obs_errors": obs_bad[:10],
        },
        "weights": {
            "by_family": {key: round(value, 6) for key, value in weight_by_family.items()},
            "by_mechanism": {key: round(value, 6) for key, value in weight_by_mechanism.items()},
            "recovery_rows": dict(rows_by_family),
            "note": "An epoch visits every row once. sample_weight scales that row's loss. Each mechanism's weights sum to 1, so row count does not set the mechanism's total weight.",
        },
        "isolation": {
            "train_valid_same_id": id_cross,
            "snapshot_path_cross": path_cross,
            "scenario_near_or_same_key": scenario_hits,
            "specs_found_train_ics": len(seen_train),
            "specs_found_valid_ics": len(seen_valid),
            "train_ics_missing_spec": sorted(train_ids - {item[0] for item in train_specs})[:30],
            "valid_ics_missing_spec": sorted(valid_ids - {item[0] for item in valid_specs}),
        },
        "grpo": {
            "proven_by_train_recovery_demo": len(grpo_proven),
            "without_train_recovery_demo": grpo_unproven,
            "nominal_solvable_ids": grpo_nominal,
            "valid_ids": grpo_in_valid,
            "missing_snapshot": grpo_missing_snap,
            "duplicate_ids": len(grpo) - len({row["id"] for row in grpo}),
        },
        "valid_recovery_also_in_train": eval_in_train,
        "problems": problems,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "static.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("counts", "family_rows_train_recovery", "family_unique_ics", "overlap_counts", "handoff_labels", "weights", "grpo")}, indent=2), flush=True)
    print("ISOLATION", json.dumps(report["isolation"]), flush=True)
    print("ENCODE", json.dumps(report["encode"]), flush=True)
    print("PROBLEMS", problems, flush=True)
    return report


def replay():
    from data_generation.rebuild_k48_recovery import _load_blob, _open_env, immediate_handoff, legal_verify

    book = load_codebook()
    env = _open_env()
    train = _rows(DATA / "supervised_targets.jsonl")
    valid = _rows(DATA / "valid_demonstrations.jsonl")
    grpo = _rows(DATA / "grpo_initial_conditions.jsonl")
    path = OUT / "replay.jsonl"
    done = {row.get("key") for row in _rows(path)} if path.is_file() else set()
    demos = [("train", row) for row in train] + [("valid", row) for row in valid]
    for split, row in demos:
        if row.get("nominal_class") != "RECOVERY_CANDIDATE":
            continue
        key = f"demo|{split}|{row['id']}|{row['continuous_npz']}"
        if key in done:
            continue
        blob = _load_blob(row["snapshot"])
        actions = np.load(ROOT / row["continuous_npz"])["actions"]
        out, _obs, executed, _contacts, qpos_error, phase_ok = legal_verify(env, blob, actions)
        record = {
            "key": key,
            "kind": "demo",
            "split": split,
            "id": row["id"],
            "strategy": row.get("strategy"),
            "reason": out.get("reason"),
            "success": bool(out.get("success")),
            "nominal_elapsed": out.get("nominal_elapsed"),
            "recovery_elapsed": out.get("recovery_elapsed"),
            "lift_completed": out.get("lift_completed"),
            "qpos_error": qpos_error,
            "phase_restored": phase_ok,
            "steps": len(executed),
            "stored_steps": int(len(actions)),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        done.add(key)
        print("DEMO", split, row["id"], row.get("strategy"), record["reason"], flush=True)
    seen_ic = set()
    ic_rows = []
    for split, row in demos:
        if row.get("nominal_class") != "RECOVERY_CANDIDATE" or row["id"] in seen_ic:
            continue
        seen_ic.add(row["id"])
        ic_rows.append(row)
    for row in grpo:
        if row["id"] not in seen_ic:
            seen_ic.add(row["id"])
            ic_rows.append(row)
    for row in ic_rows:
        key = f"immediate|{row['id']}|{row['snapshot']}"
        if key in done:
            continue
        blob = _load_blob(row["snapshot"])
        out = immediate_handoff(env, blob)
        record = {
            "key": key,
            "kind": "immediate",
            "id": row["id"],
            "reason": out.get("reason"),
            "nominal_class": out.get("nominal_class"),
            "success": bool(out.get("success")),
            "qpos_error": out.get("qpos_error"),
            "phase_restored": out.get("phase_restored"),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        done.add(key)
        print("IMMEDIATE", row["id"], record["nominal_class"], record["reason"], flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "static"
    if mode == "static":
        static_report()
    elif mode == "replay":
        replay()
    else:
        raise SystemExit(mode)
