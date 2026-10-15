"""Physics regression.

Two replays of one tape check determinism. tests/fixtures/physics_state_baseline.npz
is a later-change baseline from this tree, not a pre-refactor ground truth.
Does not train and does not rewrite manifests, checkpoints, or videos.
"""

from __future__ import annotations


import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from data_generation.strategies.vertical_recatch_search import load_blob, make_env
from envs.observations.airborne_obs import observe_airborne
from envs.contacts.frames import extract_contacts
from envs.outcomes.horizon_outcome import nominal_verify
from envs.observations.observable_obs import ObservableObsState
from envs.step_legal import step_legal

from training.k47_codebook import decode_action, load_codebook

ATOL = 1e-4
STATE_ATOL = 1e-8
BASELINE = ROOT / "tests" / "fixtures" / "physics_state_baseline.npz"
BASELINE_META = ROOT / "tests" / "fixtures" / "physics_state_baseline.json"
SCENE_CONFIG = "config/simulation/nominal.yaml"


def _check(name, cond, detail=""):
    if not cond:
        raise RuntimeError(f"{name} {detail}")
    print("PASS", name, detail, flush=True)


def _row(manifest, episode, strategy):
    chosen = None
    for line in Path(manifest).open(encoding="utf-8"):
        item = json.loads(line)
        if item.get("id") == episode and (strategy is None or item.get("strategy") == strategy):
            chosen = item
            break
    if chosen is None:
        raise RuntimeError(f"missing {episode} {strategy} in {manifest}")
    return chosen


def _restore(env, blob):
    if isinstance(blob, dict) and "capture" in blob:
        restore_capture(env, blob["capture"])
        return blob
    from training.recovery_execution import restore_ballistic

    restore_ballistic(env.sim, blob)
    return blob


def _replay(env, blob, actions, persistent_hist):
    _restore(env, blob)
    sim = env.sim
    hist = ObservableObsState(0.02)
    observations = []
    executed = []
    qpos = []
    qvel = []
    contacts = []
    control = []
    for action in np.asarray(actions, np.float32):
        state = hist if persistent_hist else ObservableObsState(0.02)
        observations.append(np.asarray(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, state), np.float32))
        executed.append(np.asarray(step_legal(sim, action), np.float32))
        qpos.append(np.array(sim.data.qpos, float).copy())
        qvel.append(np.array(sim.data.qvel, float).copy())
        rows, _summary = extract_contacts(sim)
        contacts.append([
            (
                int(row["geom1"]),
                int(row["geom2"]),
                round(float(row["Fn"]), 6),
                str(row["side"]),
                tuple(round(float(value), 6) for value in row["n_w"]),
            )
            for row in rows
        ])
        control.append((
            str(sim.fsm.phase),
            np.array(sim.fsm.p_des, float).copy(),
            np.array(sim.fsm.r_des, float).reshape(-1).copy(),
            np.array(sim.fsm.v_cmd, float).copy(),
            np.array(sim.fsm.w_cmd, float).copy(),
        ))
    return np.stack(observations), np.stack(executed), qpos, qvel, contacts, control


def _compare_tape(env, row, persistent_hist):
    blob = load_blob(ROOT / row["snapshot"])
    with np.load(ROOT / row["continuous_npz"]) as archive:
        actions = np.asarray(archive["actions"], np.float32)
        saved_obs = np.asarray(archive["observations"], np.float32)
    obs, executed, qpos, qvel, contacts, control = _replay(env, blob, actions, persistent_hist)
    obs2, executed2, qpos2, qvel2, contacts2, control2 = _replay(env, blob, actions, persistent_hist)
    _check(row["id"] + " action", np.allclose(executed, actions, atol=1e-5), str(np.max(np.abs(executed - actions))))
    _check(row["id"] + " replay", np.allclose(executed, executed2, atol=1e-6))
    qpos_err = max(float(np.max(np.abs(a - b))) for a, b in zip(qpos, qpos2))
    qvel_err = max(float(np.max(np.abs(a - b))) for a, b in zip(qvel, qvel2))
    _check(row["id"] + " qpos", qpos_err < 1e-8, str(qpos_err))
    _check(row["id"] + " qvel", qvel_err < 1e-8, str(qvel_err))
    _check(row["id"] + " contact", contacts == contacts2)
    obs_err = float(np.max(np.abs(obs - obs2)))
    _check(row["id"] + " observation replay", obs_err <= STATE_ATOL, str(obs_err))
    same_control = all(
        a[0] == b[0]
        and np.allclose(a[1], b[1], atol=STATE_ATOL)
        and np.allclose(a[2], b[2], atol=STATE_ATOL)
        and np.allclose(a[3], b[3], atol=STATE_ATOL)
        and np.allclose(a[4], b[4], atol=STATE_ATOL)
        for a, b in zip(control, control2)
    )
    _check(row["id"] + " controller", same_control)
    if obs.shape == saved_obs.shape:
        err = float(np.max(np.abs(obs - saved_obs)))
        _check(row["id"] + " observation", err <= ATOL, str(err))
    else:
        raise RuntimeError(f"{row['id']} obs shape {obs.shape} saved {saved_obs.shape}")
    return blob, actions, obs, qpos, qvel, contacts, control


def _case_rows():
    rows = []
    for key, folder in (
        ("controlled_sliding", "data/controlled_sliding"),
        ("inward_sliding", "data/inward_sliding"),
        ("horizontal_recatch", "data/recatch"),
    ):
        manifest = ROOT / folder / "manifest.jsonl"
        rows.append((key, json.loads(manifest.open(encoding="utf-8").readline())))
    rows.append(("vertical_recatch", _row(ROOT / "data/vertical_recatch/manifest.jsonl", "ball_train_v2_0070", "follow_early")))
    return rows


def _object_rows(rows):
    packed = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        packed[index] = row
    return packed


def _record(prefix, obs, actions, qpos, qvel, contacts, control):
    phase, p_des, r_des, v_cmd, w_cmd = zip(*control)
    return {
        prefix + "obs": np.asarray(obs, np.float32),
        prefix + "actions": np.asarray(actions, np.float32),
        prefix + "qpos": np.stack(qpos),
        prefix + "qvel": np.stack(qvel),
        prefix + "contacts": _object_rows(contacts),
        prefix + "phase": np.array(phase),
        prefix + "p_des": np.stack(p_des),
        prefix + "r_des": np.stack(r_des),
        prefix + "v_cmd": np.stack(v_cmd),
        prefix + "w_cmd": np.stack(w_cmd),
    }


def _baseline_meta(rows):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return {
        "role": "Baseline for later code changes. This is the current verified tree, not a pre-refactor ground truth.",
        "git_head": head,
        "working_tree": "dirty" if dirty else "clean",
        "parent_note": "HEAD 85a117b is the round-2 rollback commit. Round 3 and this cleanup are uncommitted on that commit.",
        "mujoco": mujoco.__version__,
        "scene_config": SCENE_CONFIG,
        "sim_config": "config/simulation/sim.yaml",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_atol_vs_saved_tape": ATOL,
        "state_atol": STATE_ATOL,
        "contact_force_decimals": 6,
        "contact_fields": ["geom1", "geom2", "Fn", "side", "n_w"],
        "cases": [
            {
                "key": key,
                "id": row.get("id"),
                "strategy": row.get("strategy"),
                "snapshot": row.get("snapshot"),
                "continuous_npz": row.get("continuous_npz"),
            }
            for key, row in rows
        ],
    }


def _compare_baseline(saved, prefix, obs, qpos, qvel, contacts, control):
    _check(prefix + "baseline obs", np.allclose(obs, saved[prefix + "obs"], atol=STATE_ATOL))
    _check(prefix + "baseline qpos", np.allclose(np.stack(qpos), saved[prefix + "qpos"], atol=STATE_ATOL))
    _check(prefix + "baseline qvel", np.allclose(np.stack(qvel), saved[prefix + "qvel"], atol=STATE_ATOL))
    _check(prefix + "baseline contact", contacts == list(saved[prefix + "contacts"]))
    phase, p_des, r_des, v_cmd, w_cmd = zip(*control)
    _check(prefix + "baseline phase", list(phase) == list(saved[prefix + "phase"]))
    _check(prefix + "baseline p_des", np.allclose(np.stack(p_des), saved[prefix + "p_des"], atol=STATE_ATOL))
    _check(prefix + "baseline r_des", np.allclose(np.stack(r_des), saved[prefix + "r_des"], atol=STATE_ATOL))
    _check(prefix + "baseline v_cmd", np.allclose(np.stack(v_cmd), saved[prefix + "v_cmd"], atol=STATE_ATOL))
    _check(prefix + "baseline w_cmd", np.allclose(np.stack(w_cmd), saved[prefix + "w_cmd"], atol=STATE_ATOL))


def test_state_baseline(records):
    """Compare with the saved state tape, or write it when explicitly requested.

    A missing fixture is not treated as proof that the current run matches the
    pre-refactor controller.
    """
    if os.environ.get("WRITE_PHYSICS_BASELINE") == "1":
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BASELINE, **records)
        BASELINE_META.write_text(json.dumps(_baseline_meta(_case_rows()), indent=2), encoding="utf-8")
        _check("baseline written", BASELINE.is_file(), str(BASELINE))
        return
    _check("baseline fixture", BASELINE.is_file() and BASELINE_META.is_file(), str(BASELINE))
    meta = json.loads(BASELINE_META.read_text(encoding="utf-8"))
    _check("baseline role", "not a pre-refactor ground truth" in meta["role"])
    _check("baseline mujoco", meta["mujoco"] == mujoco.__version__, meta["mujoco"])
    with np.load(BASELINE, allow_pickle=True) as saved:
        for key, _row_item in _case_rows():
            prefix = key + "_"
            _compare_baseline(
                saved,
                prefix,
                records[prefix + "obs"],
                list(records[prefix + "qpos"]),
                list(records[prefix + "qvel"]),
                list(records[prefix + "contacts"]),
                list(zip(
                    records[prefix + "phase"],
                    records[prefix + "p_des"],
                    records[prefix + "r_des"],
                    records[prefix + "v_cmd"],
                    records[prefix + "w_cmd"],
                )),
            )


def test_vertical():
    row = _row(ROOT / "data/vertical_recatch/manifest.jsonl", "ball_train_v2_0070", "follow_early")
    env = make_env()
    blob, actions, obs, qpos, qvel, contacts, control = _compare_tape(env, row, persistent_hist=True)
    records = _record("vertical_recatch_", obs, actions, qpos, qvel, contacts, control)
    with np.load(ROOT / row["token_residual_npz"]) as packed:
        tokens = np.asarray(packed["tokens"])
        residuals = np.asarray(packed["residuals"], np.float32)
    book = load_codebook()
    decoded = []
    for token, residual in zip(tokens, residuals):
        _resid, action = decode_action(book, int(token), residual)
        decoded.append(action)
    decoded = np.stack(decoded)
    # The stored tape includes only motion steps. A final HANDOFF token has no action.
    motion = decoded[: len(actions)]
    err = float(np.max(np.abs(motion - actions)))
    _check("k48 quantized replay", err <= 1e-5, str(err))
    restore_capture(env, blob["capture"])
    nominal = nominal_verify(env, blob["ref"])
    _check("nominal failure", nominal["reason"] != "SUCCESS_12S", str(nominal["reason"]))
    env = make_env()
    restore_capture(env, blob["capture"])
    for action in actions:
        step_legal(env.sim, action)
    held = nominal_verify(env, blob["ref"])
    _check("vertical SUCCESS_12S", held["reason"] == "SUCCESS_12S", str(held["reason"]))
    env = make_env()
    restore_capture(env, blob["capture"])
    for action in actions:
        step_legal(env.sim, action)
    held_again = nominal_verify(env, blob["ref"])
    _check("vertical outcome replay", held_again["reason"] == held["reason"], held_again["reason"])
    return records


def test_other_tapes():
    env = make_env()
    records = {}
    for key, row in _case_rows():
        if key == "vertical_recatch":
            continue
        _blob, actions, obs, qpos, qvel, contacts, control = _compare_tape(env, row, persistent_hist=True)
        records.update(_record(key + "_", obs, actions, qpos, qvel, contacts, control))
    return records


def test_manifests():
    categories = (
        "data/controlled_sliding/manifest.jsonl",
        "data/inward_sliding/manifest.jsonl",
        "data/recatch/manifest.jsonl",
        "data/vertical_recatch/manifest.jsonl",
        "data/regrasp/manifest.jsonl",
        "data/immediate_handoff/manifest.jsonl",
        "data/supervised_targets.jsonl",
        "data/valid_demonstrations.jsonl",
        "data/grpo_initial_conditions.jsonl",
        "data/valid_recovery.jsonl",
        "data/nominal_solvable_control.jsonl",
        "data/entry.json",
    )
    for relative in categories:
        path = ROOT / relative
        _check("manifest exists " + relative, path.is_file(), str(path))
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            _check("entry categories", "vertical_recatch" in json.dumps(payload))
            continue
        row = json.loads(path.open(encoding="utf-8").readline())
        if "continuous_npz" in row:
            with np.load(ROOT / row["continuous_npz"]) as archive:
                if "immediate_handoff" in relative:
                    _check(relative + " handoff only", "handoff_obs" in archive.files and "actions" not in archive.files)
                else:
                    _check(relative + " actions", "actions" in archive.files and "observations" in archive.files)
        if relative.endswith("immediate_handoff/manifest.jsonl"):
            _check("immediate handoff row", row.get("category", row.get("family", "immediate_handoff")) != "recatch")


def test_checkpoint():
    import torch
    from models.k48_policy import build_policy
    from training.unified_k48 import load_trained_k48

    candidates = [
        ROOT / "results/grpo/v2/checkpoint_u0783.pt",
        ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt",
        ROOT / "results/sft/checkpoint_epoch_0040.pt",
    ]
    path = next((item for item in candidates if item.is_file()), None)
    _check("checkpoint file", path is not None, str(path))
    model, book = load_trained_k48(path, torch.device("cpu"))
    keys = set(model.state_dict())
    for name in ("in_proj.weight", "pos.weight", "logits.weight", "res.weight", "proto"):
        _check("state " + name, any(key == name or key.startswith(name) for key in keys))
    fresh = build_policy(book, torch.device("cpu"), 0.25, handoff=True)
    _check("logit width", model.logits.out_features == fresh.logits.out_features == 48)
    obs = torch.zeros(1, 4, 39)
    prev = torch.zeros(1, 4, 7)
    with torch.no_grad():
        first = model(obs, prev)[0]
        second = model(obs, prev)[0]
    _check("forward repeat", torch.allclose(first, second))
    _check("forward finite", bool(torch.isfinite(first).all()))


def test_table_drop_rule():
    from tests.test_table_drop import test_boundaries, test_reward_routes_existing_formulas

    test_boundaries()
    test_reward_routes_existing_formulas()


def main():
    test_manifests()
    test_table_drop_rule()
    test_checkpoint()
    records = test_other_tapes()
    records.update(test_vertical())
    test_state_baseline(records)
    print("PHYSICS_REGRESSION_OK", flush=True)


if __name__ == "__main__":
    main()
