"""Regression tests for the latched post-impact TABLE_DROP rule."""

from __future__ import annotations


import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture

from envs.outcomes.horizon_outcome import finish_recovery
from experiments.ball_impact.privileged_ball_impact_search import step_legal
from training.reward_v2 import reward_components
from envs.outcomes.table_drop import (
    BOUNCE_DV,
    SUPPORT_FRACTION,
    SUPPORT_TIME_S,
    _fresh,
    consider,
    export_state,
    latched,
    restore_state,
)

DT = 0.002
WEIGHT = 0.20 * 9.81


def _check(name, cond):
    if not cond:
        raise RuntimeError(name)
    print("PASS", name, flush=True)


def test_boundaries():
    half = SUPPORT_FRACTION * WEIGHT

    held = _fresh()
    consider(held, DT, 1.0, 0.43, True, 40.0, -1.0, 0.5, WEIGHT)
    _check("bilateral grasp ignores table force", not held["latched"])

    noise = _fresh()
    consider(noise, DT, 1.0, 0.43, False, 0.01, -0.2, 0.2, WEIGHT)
    _check("tiny force and speed change do not latch", not noise["latched"])

    brush = _fresh()
    for step in range(9):
        consider(brush, DT, step * DT, 0.43, False, half, 0.0, 0.0, WEIGHT)
    _check("0.018 s of support is not yet a landing", not brush["latched"])
    consider(brush, DT, 9 * DT, 0.43, False, half, 0.0, 0.0, WEIGHT)
    _check("0.020 s of support latches", brush["latched"] and brush["kind"] == "support")

    gap = _fresh()
    for step in range(5):
        consider(gap, DT, step * DT, 0.43, False, half, 0.0, 0.0, WEIGHT)
    for step in range(10):
        consider(gap, DT, 1.0 + step * DT, 0.43, False, 0.0, 0.0, 0.0, WEIGHT)
    consider(gap, DT, 2.0, 0.43, False, half, 0.0, 0.0, WEIGHT)
    _check("a quiet gap resets an unfinished support count", not gap["latched"])

    bounce = _fresh()
    consider(bounce, DT, 3.29, 0.43, False, 30.0, -1.2, 0.3, WEIGHT)
    _check("one-step bounce latches without waiting for support", bounce["latched"] and bounce["kind"] == "bounce")
    _check("bounce dv meets the threshold", bounce["dv"] >= BOUNCE_DV)
    consider(bounce, DT, 3.40, 0.50, True, 0.0, 0.0, 0.0, WEIGHT)
    _check("regrasp does not clear the latch", bounce["latched"])

    side = _fresh()
    consider(side, DT, 1.0, 0.50, False, 0.0, -0.4, 0.4, WEIGHT)
    _check("upward speed change without table force is not a landing", not side["latched"])


def test_reward_routes_existing_formulas():
    limit = 4.0
    during = reward_components("TABLE_DROP", 1.0, limit, q_nominal=0.0, lift_completed=False)
    drop = reward_components("DROP_DURING_RECOVERY", 1.0, limit)
    _check("recovery TABLE_DROP uses the existing drop return", during["return"] == drop["return"])
    _check("recovery TABLE_DROP is not success", during["terminal"] < 0.0)
    after = reward_components("TABLE_DROP", 1.0, limit, q_nominal=0.5, lift_completed=True)
    handoff = reward_components("DROP_AFTER_HANDOFF", 1.0, limit, q_nominal=0.5, lift_completed=True)
    _check("post-handoff TABLE_DROP uses the existing handoff-failure return", after["return"] == handoff["return"])
    success = reward_components("SUCCESS_12S", 1.0, limit, q_nominal=1.0, lift_completed=True)
    _check("TABLE_DROP return stays below SUCCESS_12S", after["return"] < success["return"])


def _manifest_index():
    found = {}
    for path in (
        ROOT / "data" / "recatch" / "manifest.jsonl",
        ROOT / "data" / "controlled_sliding" / "manifest.jsonl",
        ROOT / "data" / "inward_sliding" / "manifest.jsonl",
        ROOT / "data" / "supervised_targets.jsonl",
    ):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("id"), row.get("strategy"))
            tape = row.get("continuous_npz") or row.get("trajectory")
            if key[0] and key[1] and tape and (key not in found or "continuous_npz" not in found[key]):
                row = dict(row)
                row["continuous_npz"] = tape
                found[key] = row
    return found


def _load_case(index, case_id, strategy):
    row = index[(case_id, strategy)]
    with (ROOT / row["snapshot"]).open("rb") as handle:
        blob = pickle.load(handle)
    with np.load(ROOT / row["continuous_npz"]) as packed:
        actions = packed["actions"].astype(np.float32)
    return row, blob, actions


def _verify(env, blob, actions):
    import data_generation.rebuild_k48_recovery as rebuild
    out, _obs, _exe, _contacts, qpos_error, phase_ok = rebuild.legal_verify(env, blob, actions)
    return out, float(qpos_error), bool(phase_ok)


def test_replays(env, index):
    results = []
    must_fail = (
        ("ball_train_v2_0051", "chase_lost"),
        ("ball_train_v2_0226", "release_long"),
        ("k48r_train_00135", "release_short"),
    )
    must_pass = (
        ("inward_extra_250005", "inward_reseat"),
        ("slide_extra_260013", "controlled_slide"),
    )
    extras = []
    for (case_id, strategy), row in index.items():
        if strategy == "inward_reseat" and case_id != "inward_extra_250005" and len([item for item in extras if item[1] == strategy]) < 1:
            extras.append((case_id, strategy))
        if strategy == "controlled_slide" and case_id != "slide_extra_260013" and len([item for item in extras if item[1] == strategy]) < 1:
            extras.append((case_id, strategy))
    for case_id, strategy in must_fail:
        row, blob, actions = _load_case(index, case_id, strategy)
        out, qpos_error, phase_ok = _verify(env, blob, actions)
        evidence = export_state(env.sim)
        record = {
            "id": case_id,
            "strategy": strategy,
            "npz": row["continuous_npz"],
            "reason": out["reason"],
            "success": bool(out["success"]),
            "qpos_error": qpos_error,
            "phase_ok": phase_ok,
            "table_drop": evidence,
            "recovery_elapsed": out.get("recovery_elapsed"),
        }
        results.append(record)
        _check(f"{case_id} {strategy} is TABLE_DROP", out["reason"] == "TABLE_DROP" and not out["success"])
        _check(f"{case_id} replay restored", qpos_error <= 1e-6 and phase_ok)
        print("  event", evidence["kind"], "t", evidence["t"], "z", evidence["z"], "dv", round(evidence["dv"], 3), flush=True)
    for case_id, strategy in must_pass + tuple(extras):
        row, blob, actions = _load_case(index, case_id, strategy)
        out, qpos_error, phase_ok = _verify(env, blob, actions)
        record = {
            "id": case_id,
            "strategy": strategy,
            "npz": row["continuous_npz"],
            "reason": out["reason"],
            "success": bool(out["success"]),
            "qpos_error": qpos_error,
            "phase_ok": phase_ok,
            "table_drop": export_state(env.sim),
        }
        results.append(record)
        _check(f"{case_id} {strategy} stays SUCCESS_12S", out["reason"] == "SUCCESS_12S" and out["success"])
        _check(f"{case_id} did not latch", not latched(env.sim))
    return results


def test_bounce_then_regrasp(env, index):
    _row, blob, _actions = _load_case(index, "inward_extra_250005", "inward_reseat")
    restore_capture(env, blob["capture"])
    _check("inward snapshot starts clean", not latched(env.sim))
    open_grip = np.array([0, 0, -1, 0, 0, 0, -1], np.float32)
    close_grip = np.array([0, 0, 1, 0, 0, 0, 1], np.float32)
    for _ in range(200):
        step_legal(env.sim, open_grip)
        if latched(env.sim):
            break
    evidence = export_state(env.sim)
    _check("opened grasp bounces or lands on the table", evidence["latched"])
    for _ in range(40):
        step_legal(env.sim, close_grip)
    _check("closing the gripper after the landing keeps TABLE_DROP", latched(env.sim))
    t0 = float(blob["capture"]["physics"]["time"]) if "physics" in blob["capture"] else float(env.sim.data.time)
    out = finish_recovery(env, blob["ref"], t0, "early")
    _check("regrasp after a bounce cannot finish as SUCCESS_12S", out["reason"] == "TABLE_DROP" and not out["success"])
    return {"kind": evidence["kind"], "t": evidence["t"], "z": evidence["z"], "dv": evidence["dv"], "reason": out["reason"]}


def test_drop_after_handoff(env, index):
    _row, blob, actions = _load_case(index, "slide_extra_260013", "controlled_slide")
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    for action in actions:
        step_legal(env.sim, action)
    _check("slide recovery itself does not land", not latched(env.sim))
    open_grip = np.array([0, 0, -1, 0, 0, 0, -1], np.float32)
    for _ in range(250):
        step_legal(env.sim, open_grip)
        if latched(env.sim):
            break
    _check("opening after the recovery lands", latched(env.sim))
    out = finish_recovery(env, blob["ref"], t0, "early")
    _check("post-recovery landing is TABLE_DROP rather than SUCCESS_12S", out["reason"] == "TABLE_DROP" and not out["success"])
    return {"t": export_state(env.sim)["t"], "kind": export_state(env.sim)["kind"], "reason": out["reason"]}


def test_snapshots(env, index):
    from experiments.handoff.benchmark_k48_prefix_branching import restore_snapshot, take_snapshot

    _row, blob, actions = _load_case(index, "ball_train_v2_0051", "chase_lost")
    restore_capture(env, blob["capture"])
    _check("old IC capture has no inherited latch", not latched(env.sim))
    clean = take_snapshot(env)
    landed = None
    for action in actions:
        step_legal(env.sim, action)
        if latched(env.sim) and landed is None:
            landed = take_snapshot(env)
            break
    _check("chase_lost snapshot was taken after TABLE_DROP", landed is not None and landed["capture"]["table_drop"]["latched"])
    physics = mujoco.MjData(env.sim.model)
    mujoco.mj_copyData(physics, env.sim.model, env.sim.data)
    restore_state(env.sim, None)
    _check("explicit reset clears the latch", not latched(env.sim))
    mujoco.mj_copyData(env.sim.data, env.sim.model, physics)
    _check("restoring mjData does not resurrect TABLE_DROP", not latched(env.sim))
    restore_snapshot(env, landed)
    _check("landed prefix snapshot restores TABLE_DROP", latched(env.sim))
    mujoco.mj_copyData(env.sim.data, env.sim.model, clean["data"])
    _check("replacing mjData leaves the landed latch set", latched(env.sim))
    restore_snapshot(env, clean)
    _check("clean prefix snapshot does not inherit the other fork", not latched(env.sim))
    restore_snapshot(env, landed)
    _check("the landed fork is unchanged by restoring the clean fork", latched(env.sim))
    bare = take_capture(env)
    bare.pop("table_drop")
    restore_capture(env, bare)
    _check("a capture without the field restores as not dropped", not latched(env.sim))
    other = export_state(env.sim)
    restore_snapshot(env, landed)
    other["latched"] = False
    _check("mutating an exported copy does not clear the live latch", latched(env.sim))


def main():
    test_boundaries()
    test_reward_routes_existing_formulas()
    text = (ROOT / "training" / "recovery_episode.py").read_text(encoding="utf-8")
    _check("initial-grasp stepper does not monitor table drop", "observe(" not in text and "table_drop" not in text)
    import torch
    from training.recovery_episode import make_training_env
    from training.unified_k48 import load_trained_k48

    device = torch.device("cpu")
    model, _saved = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", device)
    model.eval()
    env = make_training_env(device, model=model)
    index = _manifest_index()
    replayed = test_replays(env, index)
    bounce = test_bounce_then_regrasp(env, index)
    after = test_drop_after_handoff(env, index)
    test_snapshots(env, index)
    import os
    import tempfile
    out = Path(os.environ["TABLE_DROP_OUT"]) if os.environ.get("TABLE_DROP_OUT") else Path(tempfile.mkdtemp(prefix="table_drop_"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "replays": replayed,
        "bounce_then_regrasp": bounce,
        "drop_after_handoff": after,
    }, indent=2), encoding="utf-8")
    print("TABLE_DROP_TESTS_OK", flush=True)


if __name__ == "__main__":
    main()
