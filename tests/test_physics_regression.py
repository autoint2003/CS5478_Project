"""Physics regression after the layout move.

Replays saved action tapes through envs.step_legal. Does not train and does
not rewrite manifests, checkpoints, or videos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.strategies.vertical_recatch_search import load_blob, make_env
from envs.airborne_obs import observe_airborne
from envs.contacts import extract_contacts
from envs.horizon_outcome import nominal_verify
from envs.observable_obs import ObservableObsState
from envs.step_legal import step_legal
from training.history_aware_critic_ablation import restore_capture
from training.unified_k47 import decode_action, load_codebook

ATOL = 1e-4


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
    from training.recovery_runtime import restore_ballistic

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
        contacts.append([(int(row["geom1"]), int(row["geom2"]), round(float(row["Fn"]), 6), row["side"]) for row in rows])
        control.append((
            str(sim.fsm.phase),
            np.array(sim.fsm.p_des, float).copy(),
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
    same_control = all(
        a[0] == b[0] and np.allclose(a[1], b[1]) and np.allclose(a[2], b[2]) and np.allclose(a[3], b[3])
        for a, b in zip(control, control2)
    )
    _check(row["id"] + " controller", same_control)
    if obs.shape == saved_obs.shape:
        err = float(np.max(np.abs(obs - saved_obs)))
        _check(row["id"] + " observation", err <= ATOL, str(err))
    else:
        raise RuntimeError(f"{row['id']} obs shape {obs.shape} saved {saved_obs.shape}")
    return blob, actions


def test_vertical():
    row = _row(ROOT / "data/vertical_recatch/manifest.jsonl", "ball_train_v2_0070", "follow_early")
    env = make_env()
    blob, actions = _compare_tape(env, row, persistent_hist=True)
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


def test_other_tapes():
    env = make_env()
    for folder in ("data/controlled_sliding", "data/inward_sliding", "data/recatch"):
        manifest = ROOT / folder / "manifest.jsonl"
        first = json.loads(manifest.open(encoding="utf-8").readline())
        _compare_tape(env, first, persistent_hist=True)


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
    test_other_tapes()
    test_vertical()
    print("PHYSICS_REGRESSION_OK", flush=True)


if __name__ == "__main__":
    main()
