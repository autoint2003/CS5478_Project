"""Replay recovery demonstrations and check the constructed training batches.

Writes handoff_obs into an existing continuous npz only when that array is
missing. Motion actions, tokens, and residuals are compared before and after
and are not regenerated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.rebuild_k48_recovery import _load_blob, _load_jsonl, _open_env, legal_verify
from training.unified_k47 import episode_arrays, load_codebook
from training.unified_k48 import (
    HANDOFF,
    SELECTED_LIMIT_S,
    collate,
    episode_tensors,
    load_expanded,
    recorded_end_includes_handoff,
    supervised_loss,
)

DATA = ROOT / "data"
OUT = ROOT / "demo_results/diagnostics/raw/pretrain_audit"
CKPT = ROOT / "demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt"


def _demo_rows():
    rows = []
    for name in ("supervised_targets.jsonl", "valid_demonstrations.jsonl"):
        for row in _load_jsonl(DATA / name):
            if row.get("nominal_class") == "RECOVERY_CANDIDATE":
                rows.append(row)
    return rows


def _store_handoff_obs(path: Path, handoff_obs: np.ndarray):
    packed = np.load(path)
    saved = {key: np.array(packed[key]) for key in packed.files}
    before = {key: np.array(value) for key, value in saved.items() if key != "handoff_obs"}
    saved["handoff_obs"] = np.asarray(handoff_obs, np.float32).reshape(39)
    np.savez_compressed(path, **saved)
    after = np.load(path)
    for key, value in before.items():
        if key not in after.files or not np.array_equal(after[key], value):
            raise RuntimeError(f"{path} changed {key} while storing handoff_obs")


def _predicate_tests():
    if recorded_end_includes_handoff(0) or recorded_end_includes_handoff(200):
        raise RuntimeError("a 4 s tape must not gain a HANDOFF target")
    if not recorded_end_includes_handoff(199) or not recorded_end_includes_handoff(75):
        raise RuntimeError("a tape inside the 4 s limit must keep a HANDOFF target")
    folder = OUT / "limit_case"
    folder.mkdir(parents=True, exist_ok=True)
    actions = np.zeros((200, 7), np.float32)
    observations = np.zeros((200, 39), np.float32)
    tokens = np.zeros(200, np.int64)
    residuals = np.zeros((200, 7), np.float32)
    path = folder / "at_limit.npz"
    np.savez_compressed(path, actions=actions, observations=observations, tokens=tokens, residuals=residuals)
    book = load_codebook()
    row = {
        "id": "at_limit_synthetic",
        "handoff_label": "recorded_end",
        "nominal_class": "RECOVERY_CANDIDATE",
        "continuous_npz": str(path.relative_to(ROOT)).replace("\\", "/"),
        "token_residual_npz": str(path.relative_to(ROOT)).replace("\\", "/"),
    }
    obs, _prev, tok, _res = episode_tensors(row, book)
    if len(tok) != 200 or int(np.sum(tok == HANDOFF)) != 0 or obs.shape[0] != 200:
        raise RuntimeError("the 4 s case invented a HANDOFF supervision step")


def replay(env, book):
    rows = _demo_rows()
    records = []
    for row in rows:
        path = ROOT / row["continuous_npz"]
        stored = np.load(path)
        stored_actions = np.asarray(stored["actions"], np.float32).copy()
        stored_obs = np.asarray(stored["observations"], np.float32).copy()
        had_handoff = "handoff_obs" in stored.files
        stored_handoff = np.asarray(stored["handoff_obs"], np.float32).reshape(39).copy() if had_handoff else None
        token_path = ROOT / row["token_residual_npz"]
        token_before = np.load(token_path)
        tokens_before = np.asarray(token_before["tokens"]).copy()
        residuals_before = np.asarray(token_before["residuals"]).copy()
        blob = _load_blob(row["snapshot"])
        out, observations, executed, _contacts, qpos_error, phase_ok = legal_verify(env, blob, stored_actions)
        executed = np.asarray(executed, np.float32)
        observations = np.asarray(observations, np.float32)
        handoff_obs = np.asarray(out["handoff_obs"], np.float32).reshape(39)
        action_err = float(np.max(np.abs(executed - stored_actions))) if len(executed) == len(stored_actions) else 1.0
        obs_err = float(np.max(np.abs(observations - stored_obs))) if observations.shape == stored_obs.shape else 1.0
        if row.get("handoff_label") == "recorded_end":
            if stored_handoff is None or float(np.max(np.abs(stored_handoff - handoff_obs))) > 1e-5:
                _store_handoff_obs(path, handoff_obs)
        token_after = np.load(token_path)
        if not np.array_equal(token_after["tokens"], tokens_before) or not np.array_equal(token_after["residuals"], residuals_before):
            raise RuntimeError(f"{row['id']} token file changed")
        if not np.array_equal(np.load(path)["actions"], stored_actions):
            raise RuntimeError(f"{row['id']} actions changed")
        records.append({
            "id": row["id"],
            "split": row.get("split"),
            "strategy": row.get("strategy"),
            "handoff_label": row.get("handoff_label"),
            "reason": out.get("reason"),
            "success": bool(out.get("success")),
            "steps": int(len(stored_actions)),
            "recovery_elapsed": out.get("recovery_elapsed"),
            "nominal_elapsed": out.get("nominal_elapsed"),
            "qpos_error": qpos_error,
            "phase_restored": phase_ok,
            "action_err": action_err,
            "obs_err": obs_err,
            "handoff_matches_last_obs": bool(np.allclose(handoff_obs, stored_obs[-1])),
            "inside_limit": bool(recorded_end_includes_handoff(len(stored_actions))),
            "had_handoff_obs": had_handoff,
        })
        print("REPLAY", row.get("split"), row["id"], row.get("strategy"), out.get("reason"), len(stored_actions), flush=True)
    return records


def _batch_report(book):
    rows = _demo_rows()
    train = [row for row in _load_jsonl(DATA / "supervised_targets.jsonl")]
    early = [row for row in rows if row.get("handoff_label") == "early_handoff"]
    recorded = [row for row in rows if row.get("handoff_label") == "recorded_end"]
    motion_mismatch = []
    handoff_bad = []
    shift_bad = []
    for row in rows:
        stored = np.load(ROOT / row["continuous_npz"])
        actions = np.asarray(stored["actions"], np.float32)
        observations = np.asarray(stored["observations"], np.float32)
        file_tokens = np.asarray(np.load(ROOT / row["token_residual_npz"])["tokens"], np.int64)
        file_residuals = np.asarray(np.load(ROOT / row["token_residual_npz"])["residuals"], np.float32)
        obs, prev, tokens, residuals = episode_tensors(row, book)
        decoded = book[file_tokens] + file_residuals
        if float(np.max(np.abs(decoded - actions))) > 1e-5:
            motion_mismatch.append(row["id"])
        expect_handoff = row.get("handoff_label") == "early_handoff" or (
            row.get("handoff_label") == "recorded_end" and recorded_end_includes_handoff(len(actions))
        )
        if expect_handoff:
            if int(tokens[-1]) != HANDOFF or int(np.sum(tokens == HANDOFF)) != 1:
                handoff_bad.append(row["id"])
            if not np.array_equal(tokens[:-1], file_tokens) or not np.allclose(residuals[:-1], file_residuals):
                motion_mismatch.append(row["id"] + ":prefix")
            if float(np.max(np.abs(residuals[-1]))) != 0.0:
                handoff_bad.append(row["id"] + ":residual")
            if not np.allclose(obs[:-1], observations) or not np.allclose(obs[-1], np.asarray(stored["handoff_obs"], np.float32).reshape(39)):
                shift_bad.append(row["id"] + ":obs")
            if not np.allclose(prev[0], 0.0) or not np.allclose(prev[1:-1], actions[:-1]) or not np.allclose(prev[-1], actions[-1]):
                shift_bad.append(row["id"] + ":prev")
        else:
            if int(np.sum(tokens == HANDOFF)) != 0 or len(tokens) != len(actions):
                handoff_bad.append(row["id"] + ":limit")
    device = torch.device("cpu")
    model, _loaded = load_expanded(CKPT, device)
    model.train()
    masked_handoff = 0
    supervised_handoff = 0
    supervised_motion = 0
    batches = 0
    invariant = True
    recovery_handoff = 0
    immediate_handoff = 0
    for start in range(0, len(train), 8):
        batch = train[start:start + 8]
        obs, prev, tokens, residuals, mask, weight = collate(batch, book, device)
        handoff = tokens == HANDOFF
        supervised_handoff += int((handoff & mask).sum())
        supervised_motion += int((~handoff & mask).sum())
        masked_handoff += int((handoff & ~mask).sum())
        recovery_handoff += int(sum(row.get("nominal_class") == "RECOVERY_CANDIDATE" for row in batch))
        immediate_handoff += int(sum(row.get("handoff_label") == "immediate_handoff" for row in batch))
        _loss, token_a, resid_a = supervised_loss(model, obs, prev, tokens, residuals, weight, mask)
        changed = residuals.clone()
        changed[handoff] = 5.0
        _loss2, token_b, resid_b = supervised_loss(model, obs, prev, tokens, changed, weight, mask)
        if abs(float(resid_a) - float(resid_b)) > 1e-8 or abs(float(token_a) - float(token_b)) > 1e-8:
            invariant = False
        batches += 1
    return {
        "training_rows": len(train),
        "training_batches": batches,
        "recovery_rows": len(rows),
        "early_handoff_rows": len(early),
        "recorded_end_rows": len(recorded),
        "recorded_end_with_handoff_target": sum(
            1 for row in recorded if recorded_end_includes_handoff(len(np.load(ROOT / row["continuous_npz"])["actions"]))
        ),
        "recorded_end_at_limit": sum(
            1 for row in recorded if not recorded_end_includes_handoff(len(np.load(ROOT / row["continuous_npz"])["actions"]))
        ),
        "motion_mismatch": motion_mismatch,
        "handoff_bad": handoff_bad,
        "shift_bad": shift_bad,
        "supervised_handoff_tokens": supervised_handoff,
        "supervised_motion_tokens": supervised_motion,
        "masked_handoff_tokens": masked_handoff,
        "recovery_rows_in_training_batches": recovery_handoff,
        "immediate_rows_in_training_batches": immediate_handoff,
        "residual_loss_ignores_handoff": invariant,
    }


def main():
    _predicate_tests()
    env = _open_env()
    book = load_codebook()
    records = replay(env, book)
    batch = _batch_report(book)
    summary = {
        "replays": len(records),
        "success": sum(1 for row in records if row["success"] and row["reason"] == "SUCCESS_12S"),
        "action_mismatch": sum(1 for row in records if row["action_err"] > 1e-5),
        "obs_mismatch": sum(1 for row in records if row["obs_err"] > 1e-5),
        "qpos_nonzero": sum(1 for row in records if row["qpos_error"] > 1e-8 or not row["phase_restored"]),
        "nominal_short": sum(1 for row in records if (row.get("nominal_elapsed") or 0) < 11.99),
        "recorded_end": sum(1 for row in records if row["handoff_label"] == "recorded_end"),
        "recorded_end_inside_limit": sum(1 for row in records if row["handoff_label"] == "recorded_end" and row["inside_limit"]),
        "handoff_equals_last_stored_obs": sum(1 for row in records if row["handoff_matches_last_obs"]),
        "max_steps": max(row["steps"] for row in records),
        "batch": batch,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "handoff_fix_replay.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    (OUT / "handoff_fix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
