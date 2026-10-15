"""Replay historical sliding demonstrations and summarize the slide_50 search.

Writes only under results/archive/demo_results/diagnostics/raw/k48_sliding_audit.
Does not modify datasets, physics, or checkpoints, and does not open FINAL.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.ball_impact_dataset import grasp_sample
from training.history_aware_critic_ablation import restore_capture
from envs.horizon_outcome import finish_recovery, replay_commands
from training.nominal_pose_audit import errors
from experiments.ball_impact.privileged_ball_impact_search import representability, step_legal
from training.unified_k47 import CAP, load_codebook

OUT = ROOT / "results/archive/demo_results/diagnostics/raw/k48_sliding_audit"
SLIDE_DIR = ROOT / "results/archive/demo_results/diagnostics/raw/sliding_vs_regrasp/trajectories"
V1 = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset/accepted/train"
SEARCH = ROOT / "results/archive/demo_results/diagnostics/raw/k48_recovery_v3/search.jsonl"
ATTEMPTS = ROOT / "results/archive/demo_results/diagnostics/raw/k48_recovery_v3/search_attempts"
AUDIT = ROOT / "results/archive/demo_results/diagnostics/raw/k48_handoff/nominal_audit/train_immediate.jsonl"
SUPERVISED = ROOT / "results/archive/demo_results/diagnostics/raw/k48_handoff/supervised_targets.jsonl"


def _load(path):
    if "final" in str(path).lower():
        raise RuntimeError(f"refusing {path}")
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _done(path):
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def immediate(env, blob):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    out = finish_recovery(env, blob["ref"], t0, "early")
    return {
        "reason": out["reason"],
        "success": bool(out.get("success")),
        "nominal_elapsed": out.get("nominal_elapsed"),
        "lift_completed": out.get("lift_completed"),
    }


def traced_replay(env, blob, actions):
    from envs.horizon_outcome import _fail_recovery, recovery_failed

    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    samples = []
    anchor = None
    grip = []
    translation = []
    lost_low = 0
    failed = False
    for action in np.asarray(actions, np.float32):
        if float(env.sim.data.time) - t0 >= 4.0 - 1e-9:
            break
        sample, anchor = grasp_sample(env.sim, env.switch, anchor)
        row = errors(env.sim, blob["ref"])
        sample["ep"] = float(row["ep"])
        sample["z"] = float(row["z"])
        samples.append(sample)
        grip.append(float(np.asarray(action).reshape(-1)[6]))
        translation.append(float(np.max(np.abs(np.asarray(action).reshape(-1)[:3]))))
        step_legal(env.sim, action)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            break
    if failed:
        out = _fail_recovery(env.sim, blob["ref"], t0)
    else:
        out = finish_recovery(env, blob["ref"], t0, "recorded_end")
    return out, _slide_metrics(samples, grip, translation)


def _slide_metrics(samples, grip, translation):
    if not samples:
        return {"steps": 0}
    zero = 0
    bilateral = 0
    slip = 0
    for sample in samples:
        touching = int(sample["nL"]) + int(sample["nR"])
        if touching == 0:
            zero += 1
        if int(sample["nL"]) > 0 and int(sample["nR"]) > 0:
            bilateral += 1
        if touching > 0 and float(sample["v_tan"]) >= 0.03:
            slip += 1
    entry = float(samples[0]["ep"])
    best = min(float(sample["ep"]) for sample in samples)
    return {
        "steps": len(samples),
        "duration_s": round(0.02 * len(samples), 3),
        "zero_contact_steps": zero,
        "bilateral_steps": bilateral,
        "slip_steps": slip,
        "max_v_tan": round(max(float(sample["v_tan"]) for sample in samples), 4),
        "max_migration_m": round(max(float(sample["migration_m"]) for sample in samples), 5),
        "min_grip": round(min(grip), 3) if grip else None,
        "mean_translation": round(float(np.mean(translation)), 4) if translation else None,
        "ep_drop_m": round(max(0.0, entry - best), 5),
        "continuous_contact": zero == 0 and len(samples) > 0,
        "slip_observed": slip >= 2 and max(float(sample["v_tan"]) for sample in samples) >= 0.03 and max(float(sample["migration_m"]) for sample in samples) >= 0.003,
        "pinch_moved": (entry - best) >= 0.003,
    }


def classify(immediate_row, replay_row, metrics, representable):
    if immediate_row.get("success"):
        return "NOMINAL_SOLVABLE_DEMO"
    if not representable:
        return "PRIVILEGED_OR_INCOMPATIBLE"
    if replay_row.get("reason") != "SUCCESS_12S":
        return "SHORT_HORIZON_ONLY"
    if metrics.get("continuous_contact") and metrics.get("slip_observed") and metrics.get("pinch_moved"):
        return "VERIFIED_LEGAL_SLIDING_RECOVERY"
    return "SHORT_HORIZON_ONLY"


def historical(env, book):
    destination = OUT / "historical_replays.jsonl"
    done = _done(destination)
    files = sorted(SLIDE_DIR.glob("*sliding_slide*.npz"))
    for path in files:
        episode_id = path.name.split("__", 1)[0]
        key = path.name
        if key in done:
            continue
        snapshot = V1 / f"{episode_id}.pkl"
        if not snapshot.is_file():
            _append(destination, {"key": key, "id": episode_id, "class": "INSUFFICIENT_EVIDENCE", "reason": "snapshot missing"})
            continue
        blob = _load(snapshot)
        actions = np.load(path)["actions"].astype(np.float32)
        packed = representability(actions, book)
        representable = float(packed["fraction_within_0.25"]) >= 1.0 - 1e-8
        immediate_row = immediate(env, blob)
        replay_actions = packed["quantized"] if representable else actions
        official, metrics = traced_replay(env, blob, replay_actions)
        label = classify(immediate_row, official, metrics, representable)
        row = {
            "key": key,
            "id": episode_id,
            "snapshot": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            "trajectory": str(path.relative_to(ROOT)).replace("\\", "/"),
            "steps": int(len(actions)),
            "duration_s": round(0.02 * len(actions), 3),
            "representable": representable,
            "max_linf": packed["max_linf"],
            "immediate": immediate_row,
            "replay_reason": official["reason"],
            "replay_success": bool(official.get("success")),
            "recovery_elapsed": official.get("recovery_elapsed"),
            "nominal_elapsed": official.get("nominal_elapsed"),
            "metrics": metrics,
            "class": label,
            "replayed": "k48_quantized" if representable else "exact_saved_commands",
        }
        _append(destination, row)
        print("HIST", episode_id, label, official["reason"], "immediate", immediate_row["reason"], flush=True)


def k48_examples(env, book):
    destination = OUT / "k48_slide_examples.jsonl"
    done = _done(destination)
    rows = [json.loads(line) for line in SUPERVISED.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = []
    for row in rows:
        if row.get("strategy") == "slide_50" and row.get("source_kind") == "historical" and row.get("handoff_label") == "recorded_end":
            chosen.append(row)
        if len(chosen) >= 3:
            break
    if len(chosen) < 3:
        for row in rows:
            if row.get("strategy") == "slide_50" and row.get("id") == "ball_train_v2_0001":
                chosen.append(row)
                break
    for row in chosen:
        key = row["continuous_npz"]
        if key in done:
            continue
        blob = _load(ROOT / row["snapshot"])
        actions = np.load(ROOT / row["continuous_npz"])["actions"].astype(np.float32)
        packed = representability(actions, book)
        representable = float(packed["fraction_within_0.25"]) >= 1.0 - 1e-8
        immediate_row = immediate(env, blob)
        official, metrics = traced_replay(env, blob, actions)
        label = classify(immediate_row, official, metrics, representable or True)
        if immediate_row.get("success"):
            label = "NOMINAL_SOLVABLE_DEMO"
        _append(destination, {
            "key": key,
            "id": row["id"],
            "trajectory": row["continuous_npz"],
            "snapshot": row["snapshot"],
            "mechanism": row.get("mechanism"),
            "handoff_label": row.get("handoff_label"),
            "immediate": immediate_row,
            "replay_reason": official["reason"],
            "metrics": metrics,
            "representable": representable,
            "class": label,
        })
        print("K48", row["id"], label, official["reason"], flush=True)


def failure_traces():
    rows = [json.loads(line) for line in SEARCH.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("strategy") == "slide_50"]
    summary = {
        "n": len(rows),
        "reason": dict(Counter(row.get("reason") for row in rows)),
        "proposal_stop": dict(Counter(row.get("proposal_stop") for row in rows)),
        "legal_replay": dict(Counter(bool(row.get("legal_replay")) for row in rows)),
    }
    traces = []
    for row in rows:
        path = ATTEMPTS / f"{row['id']}__slide_50.npz"
        item = {
            "id": row["id"],
            "reason": row.get("reason"),
            "proposal_stop": row.get("proposal_stop"),
            "proposal_steps": row.get("proposal_steps"),
            "legal_replay": bool(row.get("legal_replay")),
            "fraction_within_0.25": row.get("fraction_within_0.25"),
        }
        if path.is_file():
            packed = np.load(path)
            actions = packed["actions"]
            nL = packed["nL"] if "nL" in packed.files else None
            nR = packed["nR"] if "nR" in packed.files else None
            item["saved_steps"] = int(len(actions))
            item["min_grip"] = round(float(np.min(actions[:, 6])), 3) if len(actions) else None
            item["mean_translation"] = round(float(np.mean(np.max(np.abs(actions[:, :3]), axis=1))), 4) if len(actions) else None
            if nL is not None:
                bilateral = (nL > 0) & (nR > 0)
                zero = (nL + nR) == 0
                item["bilateral_steps"] = int(np.sum(bilateral))
                item["zero_contact_steps"] = int(np.sum(zero))
                item["continuous_contact"] = bool(len(nL) and int(np.sum(zero)) == 0)
        traces.append(item)
    (OUT / "slide50_summary.json").write_text(json.dumps({"summary": summary, "traces": traces}, indent=2), encoding="utf-8")
    print("SLIDE50", json.dumps(summary), flush=True)


def failure_cases(env):
    from data_generation.rebuild_k48_recovery import propose_actions
    from experiments.sliding.sliding_vs_regrasp import slide_action

    destination = OUT / "failure_replays.jsonl"
    if destination.exists() and sum(1 for line in destination.read_text(encoding="utf-8").splitlines() if line.strip()) >= 5:
        return
    wanted = [
        ("ball_train_0023", "proposal_short"),
        ("ball_train_v2_0049", "proposal_lost_contact"),
        ("k48r_train_00032", "proposal_long_exact"),
        ("k48r_train_00042", "saved_cap_then_drop"),
        ("k48r_train_00152", "saved_drop_during"),
    ]
    candidates = {}
    for line in (ROOT / "results/archive/demo_results/diagnostics/raw/k48_recovery_v3/candidates.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            candidates[row["id"]] = row
    spec = {"name": "slide_50", "grip": -0.50, "kp": 20.0, "cap": 0.45, "brake": True}
    destination.write_text("", encoding="utf-8")
    for episode_id, tag in wanted:
        meta = candidates[episode_id]
        blob = _load(ROOT / meta["snapshot"])
        if tag.startswith("saved"):
            actions = np.load(ATTEMPTS / f"{episode_id}__slide_50.npz")["actions"]
            official, metrics = traced_replay(env, blob, actions)
            source = "saved_decoded_attempt"
        else:
            actions, stop = propose_actions(env, blob, "slide", spec)
            official, metrics = traced_replay(env, blob, actions)
            source = f"reproposed_closed_loop stop={stop}"
        row = {
            "id": episode_id,
            "tag": tag,
            "source": source,
            "snapshot": meta["snapshot"],
            "speed_mps": meta.get("speed_mps"),
            "azimuth_deg": meta.get("azimuth_deg"),
            "z_off_m": meta.get("z_off_m"),
            "reason": official["reason"],
            "recovery_elapsed": official.get("recovery_elapsed"),
            "nominal_elapsed": official.get("nominal_elapsed"),
            "metrics": metrics,
            "steps": int(len(actions)),
        }
        _append(destination, row)
        print("FAIL", episode_id, official["reason"], metrics.get("bilateral_steps"), metrics.get("slip_observed"), flush=True)


def main():
    import torch
    from training.autonomous_episode import make_training_env
    from training.unified_k48 import load_trained_k48

    OUT.mkdir(parents=True, exist_ok=True)
    model, _book = load_trained_k48(
        ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt",
        torch.device("cpu"),
    )
    env = make_training_env(torch.device("cpu"), model=model)
    book = load_codebook()
    failure_traces()
    historical(env, book)
    k48_examples(env, book)
    failure_cases(env)
    print("SLIDING_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
