"""Clean observation check, then a non-overwriting continued SFT.

The original checkpoint at results/sft/current/checkpoint_epoch_0040.pt is only read.
New weights go to results/diagnostics/horizontal_recatch_root_cause/repair_sft/.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from data_generation.strategies.recatch_oracle import _setup
from envs.observations.airborne_obs import OBS_NAMES_AIR
from envs.outcomes.horizon_outcome import recovery_failed
from envs.step_legal import step_legal
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.dataset import episode_tensors
from training.k47_codebook import CAP, load_codebook, read_jsonl
from training.k48_config import ROOT
from training.recovery_episode import make_training_env
from training.train_sft import supervised_loss
from visualization.audit_sft_horizontal_followup import MeanArgmax, file_sha, score_sequence
from visualization.diagnose_horizontal_recatch import (
    CKPT,
    begin,
    load_expert,
    manifest_by_id,
    prepare_horizontal,
    rollout_policy,
)

REPAIR = ROOT / "results/diagnostics/horizontal_recatch_root_cause/repair_sft"
RAW = ROOT / "results/diagnostics/horizontal_recatch_root_cause/raw"


def _json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def write(name, payload):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    path.write_text(json.dumps(_json(payload), indent=2), encoding="utf-8")
    print("WROTE", name, flush=True)


def load_blob(path, seated_ref):
    with Path(path).open("rb") as handle:
        blob = pickle.load(handle)
    if "ref" not in blob:
        blob = dict(blob)
        blob["ref"] = seated_ref
    return blob


def clean_replay(env, blob, decoded, stored_obs, stored_handoff):
    """One observation() per step, matching a rollout."""
    begin(env, blob)
    lost_low = 0
    live = []
    for action in np.asarray(decoded, np.float32):
        live.append(np.asarray(env.observation(), np.float32).copy())
        done = step_legal(env.sim, action)
        env._prev = np.asarray(done, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            break
    handoff = None if len(live) != len(decoded) else np.asarray(env.observation(), np.float32).copy()
    live_arr = np.stack(live).astype(np.float32)
    n = min(len(live_arr), len(stored_obs))
    diff = np.max(np.abs(live_arr[:n] - stored_obs[:n]), axis=0)
    step0 = np.abs(live_arr[0] - stored_obs[0])
    handoff_linf = None
    if handoff is not None and stored_handoff is not None:
        handoff_linf = float(np.max(np.abs(handoff - stored_handoff)))
    return {
        "completed": handoff is not None,
        "live_steps": int(len(live_arr)),
        "stored_steps": int(len(stored_obs)),
        "step0_linf": float(np.max(step0)),
        "traj_linf": float(np.max(np.abs(live_arr[:n] - stored_obs[:n]))),
        "dim_step0": step0,
        "dim_traj": diff,
        "handoff_linf": handoff_linf,
        "obs": live_arr,
        "handoff_obs": handoff,
    }


def collect(env, book, seated_ref):
    tapes = manifest_by_id()
    prepared = {row["id"]: blob for row, blob in prepare_horizontal(seated_ref)}
    rows = []
    captured = {}
    dim_step0 = np.zeros(39, np.float64)
    dim_traj = np.zeros(39, np.float64)
    for meta in tapes.values():
        blob = prepared.get(meta["id"])
        if blob is None:
            blob = load_blob(ROOT / meta["snapshot"], seated_ref)
        _meta, _actions, tokens, residuals, decoded = load_expert({"id": meta["id"]}, book)
        stored = np.load(ROOT / meta["continuous_npz"])
        stored_obs = stored["observations"].astype(np.float32)
        stored_handoff = stored["handoff_obs"].astype(np.float32) if "handoff_obs" in stored.files else None
        replay = clean_replay(env, blob, decoded, stored_obs, stored_handoff)
        dim_step0 = np.maximum(dim_step0, replay["dim_step0"])
        dim_traj = np.maximum(dim_traj, replay["dim_traj"])
        prev = np.zeros((len(replay["obs"]), 7), np.float32)
        if len(replay["obs"]) > 1:
            prev[1:] = np.asarray(decoded[: len(replay["obs"]) - 1], np.float32)
        item = {
            "id": meta["id"],
            "split": meta["split"],
            "completed": replay["completed"],
            "step0_linf": replay["step0_linf"],
            "traj_linf": replay["traj_linf"],
            "handoff_linf": replay["handoff_linf"],
            "live_steps": replay["live_steps"],
            "stored_steps": replay["stored_steps"],
        }
        rows.append(item)
        if meta["split"] == "train" and replay["completed"]:
            captured[meta["id"]] = {
                "obs": replay["obs"],
                "prev": prev,
                "tokens": np.asarray(tokens, np.int64),
                "residuals": np.asarray(residuals, np.float32),
                "handoff_obs": replay["handoff_obs"],
                "handoff_prev": np.asarray(decoded[-1], np.float32),
            }
        print("ALIGN", meta["split"], meta["id"], "step0", round(replay["step0_linf"], 6), "traj", round(replay["traj_linf"], 6), "handoff", replay["handoff_linf"], flush=True)
    names = list(OBS_NAMES_AIR)
    report = {
        "rows": rows,
        "train_step0_linf_max": float(max(row["step0_linf"] for row in rows if row["split"] == "train")),
        "train_traj_linf_max": float(max(row["traj_linf"] for row in rows if row["split"] == "train")),
        "worst_step0_dims": [
            {"index": int(index), "name": names[index], "linf": float(dim_step0[index])}
            for index in np.argsort(-dim_step0)[:8]
        ],
        "worst_traj_dims": [
            {"index": int(index), "name": names[index], "linf": float(dim_traj[index])}
            for index in np.argsort(-dim_traj)[:8]
        ],
    }
    write("sft_obs_alignment_clean.json", report)
    return report, captured


def stack_batch(seqs, device):
    length = max(item[0].shape[0] for item in seqs)
    obs = torch.zeros(len(seqs), length, 39, device=device)
    prev = torch.zeros(len(seqs), length, 7, device=device)
    tokens = torch.zeros(len(seqs), length, dtype=torch.long, device=device)
    residuals = torch.zeros(len(seqs), length, 7, device=device)
    mask = torch.zeros(len(seqs), length, dtype=torch.bool, device=device)
    weights = torch.zeros(len(seqs), device=device)
    for index, (obs_i, prev_i, tok_i, res_i, weight) in enumerate(seqs):
        n = obs_i.shape[0]
        obs[index, :n] = torch.tensor(obs_i)
        prev[index, :n] = torch.tensor(prev_i)
        tokens[index, :n] = torch.tensor(tok_i)
        residuals[index, :n] = torch.tensor(res_i)
        mask[index, :n] = True
        weights[index] = float(weight)
    return obs, prev, tokens, residuals, mask, weights


def horizontal_targets(captured):
    packed = []
    for item in captured.values():
        obs = np.concatenate([item["obs"], item["handoff_obs"].reshape(1, 39)], axis=0)
        prev = np.concatenate([item["prev"], item["handoff_prev"].reshape(1, 7)], axis=0)
        tokens = np.concatenate([item["tokens"], np.array([HANDOFF], np.int64)])
        residuals = np.concatenate([item["residuals"], np.zeros((1, 7), np.float32)], axis=0)
        packed.append((obs, prev, tokens, residuals, 1.0))
    return packed


def step0_mean(model, seqs, token_index=None):
    model.eval()
    values = []
    with torch.no_grad():
        for obs, prev, tokens, _residuals, _weight in seqs:
            scored = score_sequence(model, obs, prev, tokens, _residuals, {"t_open": None})
            if token_index is None:
                values.append(scored[0]["p"])
            else:
                values.append(scored[0]["p"] if scored[0]["token"] == token_index else scored[0]["p"])
    model.train()
    return float(np.mean(values)), float(np.median(values))


def load_mix(book):
    supervised = read_jsonl(ROOT / "data/supervised_targets.jsonl")
    base = []
    vertical = []
    sliding = []
    manifest = []
    for row in supervised:
        obs, prev, tokens, residuals = episode_tensors(row, book)
        item = (obs, prev, tokens, residuals, float(row.get("sample_weight") or 0.0))
        base.append(item)
        manifest.append({"id": row["id"], "family": row.get("family"), "source": "supervised_targets"})
        if row.get("family") == "vertical_recatch":
            vertical.append(item)
        elif row.get("family") == "sliding":
            sliding.append(item)
    horizontal = []
    for meta in manifest_by_id().values():
        if meta.get("split") != "train":
            continue
        labeled = dict(meta)
        labeled["handoff_label"] = "early_handoff"
        obs, prev, tokens, residuals = episode_tensors(labeled, book)
        if int(tokens[0]) != 34 or int(tokens[-1]) != HANDOFF:
            raise RuntimeError(f"{meta['id']} repair target is not token 34 then HANDOFF")
        horizontal.append((obs, prev, tokens, residuals, 1.0))
        manifest.append({"id": meta["id"], "family": "horizontal_recatch", "handoff_label": "early_handoff", "sample_weight": 1.0, "source": "recatch_manifest"})
    return base + horizontal, horizontal, vertical, sliding, manifest


def probe(model, seqs):
    model.eval()
    step0 = []
    ce = []
    resid = []
    for obs, prev, tokens, residuals, _weight in seqs:
        scored = score_sequence(model, obs, prev, tokens, residuals, {"t_open": None})
        motion = [row for row in scored if row["token"] != HANDOFF]
        step0.append(scored[0])
        ce.append(float(np.mean([row["ce"] for row in motion])))
        resid.append(float(np.mean([row["resid_l2"] for row in motion])))
    model.train()
    return {
        "mean_p": float(np.mean([row["p"] for row in step0])),
        "median_p": float(np.median([row["p"] for row in step0])),
        "mean_ce": float(np.mean([row["ce"] for row in step0])),
        "argmax_match": int(sum(row["argmax"] == row["token"] for row in step0)),
        "n": len(step0),
        "trajectory_ce": float(np.mean(ce)),
        "trajectory_resid_l2": float(np.mean(resid)),
    }


def train_and_save(book, rows, horizontal, vertical, sliding):
    device = torch.device("cpu")
    model, _loaded = load_trained_k48(CKPT, device)
    curve = [{"epoch": 0, "horizontal": probe(model, horizontal), "vertical": probe(model, vertical), "sliding": probe(model, sliding[:40])}]
    print("REPAIR_EPOCH", 0, curve[0]["horizontal"]["median_p"], curve[0]["vertical"]["mean_p"], flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(1)
    model.train()
    for epoch in range(1, 41):
        order = rng.permutation(len(rows))
        total = 0.0
        seen = 0
        for start in range(0, len(rows), 8):
            batch = [rows[int(index)] for index in order[start:start + 8]]
            obs, prev, tokens, residuals, mask, weight = stack_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            loss, _token_loss, _resid_loss = supervised_loss(model, obs, prev, tokens, residuals, weight, mask)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            seen += 1
        point = {
            "epoch": epoch,
            "loss": total / max(seen, 1),
            "horizontal": probe(model, horizontal),
            "vertical": probe(model, vertical),
            "sliding": probe(model, sliding[:40]),
        }
        curve.append(point)
        print("REPAIR_EPOCH", epoch, round(point["loss"], 4), "h", round(point["horizontal"]["median_p"], 4), "v", round(point["vertical"]["mean_p"], 4), flush=True)
        if point["horizontal"]["median_p"] >= 0.9 and point["horizontal"]["argmax_match"] == point["horizontal"]["n"]:
            break
    REPAIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "proto": book,
            "res_cap": CAP,
            "handoff": HANDOFF,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "epoch": curve[-1]["epoch"],
            "source": "results/sft/current/checkpoint_epoch_0040.pt",
        },
        REPAIR / "checkpoint_repair.pt",
    )
    model.eval()
    return model, curve


def category_of(row):
    cats = set(row.get("categories") or [])
    if "horizontal_recatch" in cats:
        return "horizontal"
    if "vertical_recatch" in cats:
        return "vertical"
    return "sliding"


def evaluate(env, model, book, seated_ref, tag):
    device = next(model.parameters()).device
    model.eval()
    results = []
    for row, blob in prepare_horizontal(seated_ref):
        outcome = rollout_policy(env, blob, model, book, MeanArgmax(), device)
        results.append({"split": "train", "category": "horizontal", "id": row["id"], "reason": outcome["reason"], "success": outcome["success"], "first_token": outcome["first_token"]})
        print(tag, "train", "horizontal", row["id"], outcome["reason"], outcome["first_token"], flush=True)
    valid = read_jsonl(ROOT / "data/valid_skill_quotas.jsonl")
    for row in valid:
        blob = load_blob(ROOT / row["snapshot"], seated_ref)
        outcome = rollout_policy(env, blob, model, book, MeanArgmax(), device)
        cat = category_of(row)
        results.append({"split": "valid", "category": cat, "id": row["id"], "reason": outcome["reason"], "success": outcome["success"], "first_token": outcome["first_token"]})
        print(tag, "valid", cat, row["id"], outcome["reason"], outcome["first_token"], flush=True)
    return results


def main():
    before = file_sha(CKPT)
    book = load_codebook()
    rows, horizontal, vertical, sliding, manifest = load_mix(book)
    REPAIR.mkdir(parents=True, exist_ok=True)
    (REPAIR / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    print("MIX", len(rows), "horizontal", len(horizontal), flush=True)
    device = torch.device("cpu")
    original, _loaded = load_trained_k48(CKPT, device)
    original.eval()
    _sim, _base, seated_ref, _target, _code = _setup()
    env = make_training_env(device, model=original)
    original_eval = evaluate(env, original, book, seated_ref, "ORIGINAL")
    write("sft_repair_original_eval.json", original_eval)
    repaired, curve = train_and_save(book, rows, horizontal, vertical, sliding)
    write("sft_repair_curve.json", curve)
    repaired_eval = evaluate(env, repaired, book, seated_ref, "REPAIR")
    write("sft_repair_eval.json", repaired_eval)
    after = file_sha(CKPT)
    write("sft_checkpoint_guard_repair.json", {"before": before, "after": after, "unchanged": before == after})
    if before != after:
        raise RuntimeError("original SFT checkpoint changed")
    print("REPAIR_DONE", flush=True)


if __name__ == "__main__":
    main()
