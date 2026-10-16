"""SFT follow-up for the Horizontal Recatch audit.

Does not modify physics, the nominal controller, the K48 codebook, reward_v2,
SUCCESS_12S, or the saved SFT checkpoint. New weights go under
results/diagnostics/horizontal_recatch_root_cause/overfit/.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from data_generation.strategies.recatch_oracle import _setup
from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed
from envs.outcomes.table_drop import export_state
from envs.step_legal import step_legal
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.dataset import episode_tensors
from training.k47_codebook import CAP, decode_action, load_codebook, read_jsonl
from training.k48_config import ROOT
from training.recovery_episode import make_training_env
from training.train_sft import supervised_loss
from visualization.diagnose_horizontal_recatch import (
    CKPT,
    begin,
    load_expert,
    manifest_by_id,
    play,
    policy_distribution,
    prepare_horizontal,
    rollout_policy,
)

OUT = ROOT / "results/diagnostics/horizontal_recatch_root_cause/raw"
OVERFIT_DIR = ROOT / "results/diagnostics/horizontal_recatch_root_cause/overfit"
FOCUS = ("ref_s6_p10_a0_e0", "s6_p15_a+0.012_e0.008", "s7_p8_a+0.012_e0.008")
TREE_IDS = ("s6_p8_a-0.012_e0.000", "s7_p8_a+0.012_e0.008")


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
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(_json(payload), indent=2), encoding="utf-8")
    print("WROTE", path.name, flush=True)


def file_sha(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


class MeanArgmax:
    def choice(self, n, p=None):
        return int(np.argmax(np.asarray(p)))

    def normal(self, loc, scale, size=None):
        return np.zeros(size, np.float32)


def phase_of(step, meta):
    t = step * 0.02
    t_open = meta.get("t_open")
    t_re = meta.get("t_re")
    t_brake = meta.get("t_brake")
    if t_open is None:
        return "unmarked"
    if t < float(t_open):
        return "before_open"
    if t_re is not None and t < float(t_re):
        return "open_to_recontact"
    if t_brake is not None and t < float(t_brake):
        return "recontact_to_brake"
    return "after_brake"


def window_stats(steps, n):
    chunk = steps[:n]
    if not chunk:
        return None
    return {
        "n": len(chunk),
        "mean_ce": float(np.mean([row["ce"] for row in chunk])),
        "mean_p": float(np.mean([row["p"] for row in chunk])),
        "mean_resid_l2": float(np.mean([row["resid_l2"] for row in chunk])),
        "mean_action_loss": float(np.mean([row["action_loss"] for row in chunk])),
    }


def score_sequence(model, obs, prev, tokens, residuals, meta):
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, pred, _act = model(
            torch.tensor(obs, device=device),
            torch.tensor(prev, device=device),
        )
        log_p = torch.log_softmax(logits.float(), dim=-1)
        probs = log_p.exp()
    rows = []
    for index in range(obs.shape[0]):
        token = int(tokens[index])
        logit = logits[index].float()
        probability = float(probs[index, token])
        ce = float(-log_p[index, token])
        rank = 1 + int(torch.sum(logit > logit[token]))
        if token == HANDOFF:
            resid_l2 = 0.0
            action_loss = ce
        else:
            resid_l2 = float(torch.linalg.vector_norm(pred[index].float() - torch.tensor(residuals[index])).cpu())
            action_loss = ce + float(torch.mean((pred[index].float() - torch.tensor(residuals[index])) ** 2))
        rows.append({
            "step": index,
            "phase": "handoff_target" if token == HANDOFF else phase_of(index, meta),
            "token": token,
            "p": probability,
            "rank": rank,
            "ce": ce,
            "resid_l2": resid_l2,
            "p_handoff": float(probs[index, HANDOFF]),
            "argmax": int(torch.argmax(logit)),
            "p_argmax": float(probs[index, int(torch.argmax(logit))]),
            "action_loss": action_loss,
        })
    return rows


def summarize_steps(rows):
    phases = defaultdict(list)
    for row in rows:
        phases[row["phase"]].append(row)
    return {
        "n": len(rows),
        "windows": {str(n): window_stats(rows, n) for n in (1, 3, 5, 10, len(rows))},
        "phases": {
            name: {
                "n": len(group),
                "mean_ce": float(np.mean([row["ce"] for row in group])),
                "mean_p": float(np.mean([row["p"] for row in group])),
                "mean_resid_l2": float(np.mean([row["resid_l2"] for row in group])),
            }
            for name, group in phases.items()
        },
        "step0": rows[0] if rows else None,
    }


def census(book):
    supervised = read_jsonl(ROOT / "data/supervised_targets.jsonl")
    valid_demo = read_jsonl(ROOT / "data/valid_demonstrations.jsonl") if (ROOT / "data/valid_demonstrations.jsonl").is_file() else []
    recatch = list(manifest_by_id().values())
    supervised_ids = {row["id"] for row in supervised}
    supervised_snaps = {row.get("snapshot") for row in supervised}
    horizontal_train = [row for row in recatch if row.get("split") == "train"]
    horizontal_valid = [row for row in recatch if row.get("split") == "valid"]

    by_family = defaultdict(lambda: {"rows": 0, "ids": set(), "snaps": set(), "weight": 0.0, "steps": 0, "tokens": Counter(), "step0": Counter(), "handoff_targets": 0, "effective_step0": 0.0})
    token_rows = Counter()
    token34 = []
    decode_errors = []
    weight_zero = 0
    for row in supervised:
        family = row.get("family") or "none"
        bucket = by_family[family]
        bucket["rows"] += 1
        bucket["ids"].add(row["id"])
        bucket["snaps"].add(row.get("snapshot"))
        weight = float(row.get("sample_weight") or 0.0)
        bucket["weight"] += weight
        if weight == 0.0:
            weight_zero += 1
        obs, prev, tokens, residuals = episode_tensors(row, book)
        blob = np.load(ROOT / row["continuous_npz"])
        if "actions" not in blob.files:
            n = int(len(tokens))
            bucket["steps"] += n
            bucket["handoff_targets"] += int(np.sum(tokens == HANDOFF))
            share = weight / max(n, 1)
            bucket["effective_step0"] += share
            bucket["step0"][int(tokens[0])] += 1
            for token in tokens:
                bucket["tokens"][int(token)] += 1
                token_rows[int(token)] += 1
            continue
        actions = blob["actions"].astype(np.float32)
        if int(tokens[-1]) != HANDOFF:
            motion_tokens = tokens
            motion_residuals = residuals
        else:
            motion_tokens = tokens[:-1]
            motion_residuals = residuals[:-1]
        if len(motion_tokens) != len(actions):
            decode_errors.append({"id": row["id"], "strategy": row.get("strategy"), "token_len": int(len(motion_tokens)), "action_len": int(len(actions))})
        else:
            worst = 0.0
            for token, residual, action in zip(motion_tokens, motion_residuals, actions):
                _clipped, decoded = decode_action(book, int(token), residual)
                worst = max(worst, float(np.max(np.abs(decoded - action))))
            if worst > 1e-5:
                decode_errors.append({"id": row["id"], "strategy": row.get("strategy"), "linf": worst})
        n = int(len(tokens))
        bucket["steps"] += n
        bucket["handoff_targets"] += int(np.sum(tokens == HANDOFF))
        share = weight / n
        bucket["effective_step0"] += share
        bucket["step0"][int(tokens[0])] += 1
        for token in tokens:
            bucket["tokens"][int(token)] += 1
            token_rows[int(token)] += 1
        hits = np.flatnonzero(tokens == 34)
        for hit in hits[:8]:
            token34.append({
                "id": row["id"],
                "family": family,
                "strategy": row.get("strategy"),
                "step": int(hit),
                "weight": weight,
                "n_steps": n,
                "step_share": share,
            })
        if len(hits) > 8:
            token34.append({"id": row["id"], "family": family, "extra_token34": int(len(hits) - 8)})

    def pack(bucket):
        return {
            "rows": bucket["rows"],
            "unique_ids": len(bucket["ids"]),
            "unique_snapshots": len(bucket["snaps"]),
            "sample_weight_sum": bucket["weight"],
            "target_steps": bucket["steps"],
            "handoff_targets": bucket["handoff_targets"],
            "samples_per_epoch": bucket["rows"],
            "step0_effective_weight_sum": bucket["effective_step0"],
            "step0_tokens": dict(bucket["step0"]),
            "top_tokens": bucket["tokens"].most_common(8),
            "token34": int(bucket["tokens"][34]),
        }

    horizontal_chain = []
    for row in recatch:
        blob = np.load(ROOT / row["continuous_npz"])
        packed = np.load(ROOT / row["token_residual_npz"])
        tokens = packed["tokens"].astype(np.int64)
        residuals = packed["residuals"].astype(np.float32)
        actions = blob["actions"].astype(np.float32)
        worst = 0.0
        for token, residual, action in zip(tokens, residuals, actions):
            _clipped, decoded = decode_action(book, int(token), residual)
            worst = max(worst, float(np.max(np.abs(decoded - action))))
        as_written = episode_tensors(row, book)
        early = dict(row)
        early["handoff_label"] = "early_handoff"
        early_error = None
        early_len = None
        try:
            early_len = int(episode_tensors(early, book)[2].shape[0])
        except RuntimeError as exc:
            early_error = str(exc)
        horizontal_chain.append({
            "id": row["id"],
            "split": row["split"],
            "strategy": row["strategy"],
            "provenance": row.get("provenance"),
            "handoff_label": row.get("handoff_label"),
            "sample_weight": row.get("sample_weight"),
            "in_supervised_id": row["id"] in supervised_ids,
            "in_supervised_snapshot": row.get("snapshot") in supervised_snaps,
            "in_valid_demonstrations": row["id"] in {item["id"] for item in valid_demo},
            "actions": int(len(actions)),
            "stored_obs": list(blob["observations"].shape),
            "token34_count": int(np.sum(tokens == 34)),
            "token34_step0": bool(int(tokens[0]) == 34),
            "token_min": int(tokens.min()),
            "token_max": int(tokens.max()),
            "decode_linf": worst,
            "targets_with_written_label": int(as_written[2].shape[0]),
            "written_label_includes_handoff": bool(np.any(as_written[2] == HANDOFF)),
            "targets_if_early_handoff": early_len,
            "early_handoff_error": early_error,
            "npz_has_handoff_obs": "handoff_obs" in blob.files,
            "prev0_is_zero_under_episode_arrays": True,
        })

    payload = {
        "supervised_rows": len(supervised),
        "supervised_unique_ids": len(supervised_ids),
        "supervised_unique_snapshots": len(supervised_snaps),
        "supervised_row_meaning": "one JSONL row is one expert tape; train_supervised shuffles these rows and collates whole tapes, not windows and not unique snapshots",
        "batches_per_epoch": int(np.ceil(len(supervised) / 8)),
        "weight_zero_rows": weight_zero,
        "sample_weight_sum": float(sum(float(row.get("sample_weight") or 0.0) for row in supervised)),
        "families": {name: pack(bucket) for name, bucket in sorted(by_family.items())},
        "strategies": dict(Counter(row.get("strategy") for row in supervised)),
        "handoff_labels": dict(Counter(row.get("handoff_label") for row in supervised)),
        "token34_in_supervised_targets": int(token_rows[34]),
        "token34_examples": token34[:40],
        "decode_mismatches": decode_errors,
        "horizontal_train_in_supervised": sum(row["split"] == "train" and row["in_supervised_id"] for row in horizontal_chain),
        "horizontal_valid_in_supervised": sum(row["split"] == "valid" and row["in_supervised_id"] for row in horizontal_chain),
        "horizontal_rows": horizontal_chain,
        "codebook_token_34": book[34].tolist(),
        "codebook_token_36": book[36].tolist(),
        "valid_demonstration_rows": len(valid_demo),
        "horizontal_train_ids": [row["id"] for row in horizontal_train],
        "horizontal_valid_ids": [row["id"] for row in horizontal_valid],
    }
    write("sft_manifest_census.json", payload)
    return payload


def stored_teacher_force(model, book, census_payload):
    supervised = read_jsonl(ROOT / "data/supervised_targets.jsonl")
    grouped = defaultdict(list)
    step0 = defaultdict(list)
    for row in supervised:
        obs, prev, tokens, residuals = episode_tensors(row, book)
        scored = score_sequence(model, obs, prev, tokens, residuals, row)
        family = row.get("family") or "none"
        summary = summarize_steps(scored)
        summary["id"] = row["id"]
        summary["strategy"] = row.get("strategy")
        summary["weight"] = float(row.get("sample_weight") or 0.0)
        grouped[family].append(summary)
        step0[family].append(summary["step0"])
    family_report = {}
    for family, rows in grouped.items():
        family_report[family] = {
            "tapes": len(rows),
            "mean_ce": float(np.mean([row["windows"][str(row["n"])]["mean_ce"] for row in rows])),
            "mean_step0_p": float(np.mean([row["step0"]["p"] for row in rows])),
            "median_step0_p": float(np.median([row["step0"]["p"] for row in rows])),
            "mean_step0_ce": float(np.mean([row["step0"]["ce"] for row in rows])),
            "mean_step0_rank": float(np.mean([row["step0"]["rank"] for row in rows])),
            "step0_argmax_match": int(sum(row["step0"]["argmax"] == row["step0"]["token"] for row in rows)),
            "weight_sum": float(sum(row["weight"] for row in rows)),
        }
    horizontal = []
    for row in census_payload["horizontal_rows"]:
        meta = manifest_by_id()[row["id"]]
        obs, prev, tokens, residuals = episode_tensors(meta, book)
        scored = score_sequence(model, obs, prev, tokens, residuals, meta)
        horizontal.append({"id": meta["id"], "split": meta["split"], "stored_label": meta["handoff_label"], **summarize_steps(scored), "steps": scored})
    payload = {"trained_families": family_report, "horizontal_stored_obs": horizontal}
    write("sft_teacher_forced_stored.json", payload)
    return payload


def live_teacher_force(env, model, book, seated_ref):
    tapes = manifest_by_id()
    prepared = {row["id"]: blob for row, blob in prepare_horizontal(seated_ref)}
    valid = []
    for row in tapes.values():
        if row.get("split") != "valid":
            continue
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if "ref" not in blob:
            blob = dict(blob)
            blob["ref"] = seated_ref
        valid.append((row, blob))
    reports = []
    captured = {}
    for meta, blob in [(tapes[key], prepared[key]) for key in prepared] + valid:
        _meta, actions, tokens, residuals, decoded = load_expert({"id": meta["id"]}, book)
        t0 = begin(env, blob)
        lost_low = 0
        obs_rows = []
        prev_rows = []
        live_steps = []
        how = None
        for index, (token, residual, action) in enumerate(zip(tokens, residuals, decoded)):
            obs_now = np.asarray(env.observation(), np.float32).copy()
            prev_now = np.asarray(env._prev, np.float32).copy()
            _obs, probs, mean, _entropy = policy_distribution(model, env, next(model.parameters()).device)
            ce = float(-np.log(max(float(probs[int(token)]), 1e-12)))
            rank = 1 + int(np.sum(probs > probs[int(token)]))
            resid_l2 = float(np.linalg.norm(mean - residual))
            live_steps.append({
                "step": index,
                "phase": phase_of(index, meta),
                "token": int(token),
                "p": float(probs[int(token)]),
                "rank": rank,
                "ce": ce,
                "resid_l2": resid_l2,
                "p_handoff": float(probs[HANDOFF]),
                "argmax": int(np.argmax(probs)),
                "p_argmax": float(np.max(probs)),
                "action_loss": ce + float(np.mean((mean - residual) ** 2)),
            })
            obs_rows.append(obs_now)
            prev_rows.append(prev_now)
            if float(env.sim.data.time) - t0 >= 4.0 - 1e-9:
                how = "limit"
                break
            done = step_legal(env.sim, action)
            env._prev = np.asarray(done, np.float32)
            failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
            if failed:
                how = "drop"
                break
        handoff_obs = None if how == "drop" else np.asarray(env.observation(), np.float32).copy()
        stored = np.load(ROOT / meta["continuous_npz"])
        stored_obs = stored["observations"].astype(np.float32)
        n = min(len(obs_rows), len(stored_obs))
        obs_linf = float(np.max(np.abs(np.stack(obs_rows[:n]) - stored_obs[:n]))) if n else None
        stored_prev = np.zeros_like(actions)
        if len(actions) > 1:
            stored_prev[1:] = actions[:-1]
        prev_linf = float(np.max(np.abs(np.stack(prev_rows[:n]) - stored_prev[:n]))) if n else None
        item = {
            "id": meta["id"],
            "split": meta["split"],
            "strategy": meta["strategy"],
            "replay": how or "completed",
            "steps": live_steps,
            **summarize_steps(live_steps),
            "obs_linf_vs_stored": obs_linf,
            "prev_linf_vs_stored": prev_linf,
            "stored_steps": int(len(stored_obs)),
            "live_steps": len(obs_rows),
            "handoff_obs_captured": handoff_obs is not None,
        }
        reports.append(item)
        if meta["split"] == "train" and handoff_obs is not None:
            captured[meta["id"]] = {
                "obs": np.stack(obs_rows).astype(np.float32),
                "prev": np.stack(prev_rows).astype(np.float32),
                "tokens": np.asarray(tokens[: len(obs_rows)], np.int64),
                "residuals": np.asarray(residuals[: len(obs_rows)], np.float32),
                "handoff_obs": handoff_obs,
                "handoff_prev": np.asarray(env._prev, np.float32).copy(),
            }
        print("LIVE", meta["split"], meta["id"], item["replay"], "obs", obs_linf, "p0", item["step0"]["p"], flush=True)
    write("sft_teacher_forced_live.json", reports)
    return reports, captured


def force_tokens(env, blob, tokens, residuals_or_none, model, book, residual_source):
    _meta_unused = residuals_or_none
    t0 = begin(env, blob)
    lost_low = 0
    how = None
    deviations = []
    for token in tokens:
        if float(env.sim.data.time) - t0 >= 4.0 - 1e-9:
            how = "forced"
            break
        _obs, _probs, mean, _entropy = policy_distribution(model, env, next(model.parameters()).device)
        pre = mean if residual_source == "policy_mean" else residuals_or_none
        if residual_source != "policy_mean":
            raise RuntimeError(residual_source)
        _residual, action = decode_action(book, int(token), pre)
        expert_residual, expert_action = decode_action(book, int(token), residuals_or_none[len(deviations)] if False else pre)
        del expert_residual, expert_action
        done = step_legal(env.sim, action)
        env._prev = np.asarray(done, np.float32)
        deviations.append(float(np.max(np.abs(action))))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how = "drop"
            break
    if how == "drop":
        from envs.outcomes.horizon_outcome import _fail_recovery

        outcome = _fail_recovery(env.sim, blob["ref"], t0)
    else:
        outcome = finish_recovery(env, blob["ref"], t0, "forced" if how is None else how)
    return {
        "reason": outcome["reason"],
        "success": bool(outcome.get("success")),
        "steps": len(deviations),
        "recovery_elapsed": float(outcome.get("recovery_elapsed") or 0.0),
        "latch_kind": export_state(env.sim)["kind"],
    }


def expert_token_policy_mean(env, blob, tokens, expert_residuals, model, book):
    t0 = begin(env, blob)
    lost_low = 0
    how = None
    dev = []
    for index, token in enumerate(tokens):
        if float(env.sim.data.time) - t0 >= 4.0 - 1e-9:
            how = "forced"
            break
        _obs, _probs, mean, _entropy = policy_distribution(model, env, next(model.parameters()).device)
        _residual, action = decode_action(book, int(token), mean)
        _expert_residual, expert_action = decode_action(book, int(token), expert_residuals[index])
        dev.append(float(np.max(np.abs(action - expert_action))))
        done = step_legal(env.sim, action)
        env._prev = np.asarray(done, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how = "drop"
            break
    if how == "drop":
        from envs.outcomes.horizon_outcome import _fail_recovery

        outcome = _fail_recovery(env.sim, blob["ref"], t0)
    else:
        outcome = finish_recovery(env, blob["ref"], t0, "forced" if how is None else how)
    return {
        "reason": outcome["reason"],
        "success": bool(outcome.get("success")),
        "steps": len(dev),
        "max_action_deviation": max(dev) if dev else None,
        "recovery_elapsed": float(outcome.get("recovery_elapsed") or 0.0),
        "latch_kind": export_state(env.sim)["kind"],
    }


def handoff_at(env, blob, actions, k):
    t0 = begin(env, blob)
    lost_low = 0
    how = None
    executed = 0
    for action in np.asarray(actions[:k], np.float32):
        if float(env.sim.data.time) - t0 >= 4.0 - 1e-9:
            how = "forced"
            break
        done = step_legal(env.sim, action)
        env._prev = np.asarray(done, np.float32)
        executed += 1
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how = "drop"
            break
    if how == "drop":
        from envs.outcomes.horizon_outcome import _fail_recovery

        outcome = _fail_recovery(env.sim, blob["ref"], t0)
    else:
        outcome = finish_recovery(env, blob["ref"], t0, "forced" if how is None else how)
    return {
        "k": int(k),
        "executed": executed,
        "reason": outcome["reason"],
        "success": bool(outcome.get("success")),
        "recovery_elapsed": float(outcome.get("recovery_elapsed") or 0.0),
    }


def interventions(env, model, book, seated_ref):
    tapes = manifest_by_id()
    prepared = prepare_horizontal(seated_ref)
    results = []
    for row, blob in prepared:
        meta, _actions, tokens, residuals, decoded = load_expert({"id": row["id"]}, book)
        item = {"id": row["id"], "strategy": meta["strategy"], "tape_steps": int(len(decoded))}
        item["A_expert"] = {"reason": "see expert_replay_train.json", "success": True}
        for k in (1, 3, 5):
            outcome = rollout_policy(env, blob, model, book, MeanArgmax(), next(model.parameters()).device, prefix_actions=decoded[:k])
            item[f"D{k}_deterministic"] = outcome
            print("PREFIX", row["id"], k, outcome["reason"], outcome.get("first_token"), flush=True)
        if row["id"] in FOCUS:
            stochastic = {}
            for k in (1, 3, 5):
                draws = []
                for seed in (11, 22, 33, 44):
                    outcome = rollout_policy(
                        env, blob, model, book, np.random.default_rng(seed), next(model.parameters()).device, prefix_actions=decoded[:k],
                    )
                    draws.append({"seed": seed, "reason": outcome["reason"], "success": outcome["success"], "first_token": outcome["first_token"]})
                stochastic[str(k)] = draws
            item["stochastic_prefix"] = stochastic
            item["E_expert_token_policy_mean"] = expert_token_policy_mean(env, blob, tokens, residuals, model, book)
            item["F_handoff_timing"] = [handoff_at(env, blob, decoded, k) for k in (1, 10, len(decoded))]
            print("E", row["id"], item["E_expert_token_policy_mean"]["reason"], flush=True)
            print("F", row["id"], [row_f["reason"] for row_f in item["F_handoff_timing"]], flush=True)
        results.append(item)
    write("sft_prefix_interventions.json", results)
    return results


def sequence_step0(model, captured):
    model.eval()
    rows = []
    for item_id, item in captured.items():
        scored = score_sequence(model, item["obs"], item["prev"], item["tokens"], item["residuals"], {"t_open": None})
        rows.append({"id": item_id, "step0": scored[0], "mean_ce": float(np.mean([row["ce"] for row in scored])), "mean_resid_l2": float(np.mean([row["resid_l2"] for row in scored]))})
    return rows


def overfit(model_state, book, captured, env, seated_ref):
    device = torch.device("cpu")
    model, _book = load_trained_k48(CKPT, device)
    before = sequence_step0(model, {key: captured[key] for key in FOCUS})
    batch = []
    metas = []
    for item_id in FOCUS:
        item = captured[item_id]
        obs = np.concatenate([item["obs"], item["handoff_obs"].reshape(1, 39)], axis=0)
        prev = np.concatenate([item["prev"], item["handoff_prev"].reshape(1, 7)], axis=0)
        tokens = np.concatenate([item["tokens"], np.array([HANDOFF], np.int64)])
        residuals = np.concatenate([item["residuals"], np.zeros((1, 7), np.float32)], axis=0)
        batch.append((obs, prev, tokens, residuals))
        metas.append(item_id)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    curve = []
    model.train()
    for epoch in range(1, 81):
        packed = batch
        length = max(item[0].shape[0] for item in packed)
        obs = torch.zeros(len(packed), length, 39, device=device)
        prev = torch.zeros(len(packed), length, 7, device=device)
        tokens = torch.zeros(len(packed), length, dtype=torch.long, device=device)
        residuals = torch.zeros(len(packed), length, 7, device=device)
        mask = torch.zeros(len(packed), length, dtype=torch.bool, device=device)
        for index, (obs_i, prev_i, tok_i, res_i) in enumerate(packed):
            n = obs_i.shape[0]
            obs[index, :n] = torch.tensor(obs_i)
            prev[index, :n] = torch.tensor(prev_i)
            tokens[index, :n] = torch.tensor(tok_i)
            residuals[index, :n] = torch.tensor(res_i)
            mask[index, :n] = True
        weight = torch.ones(len(packed), device=device)
        opt.zero_grad(set_to_none=True)
        loss, token_loss, resid_loss = supervised_loss(model, obs, prev, tokens, residuals, weight, mask)
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == 80:
            probe = sequence_step0(model, {key: captured[key] for key in FOCUS})
            point = {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "token_loss": float(token_loss),
                "resid_loss": float(resid_loss),
                "step0_p34": [row["step0"]["p"] for row in probe],
                "mean_ce": [row["mean_ce"] for row in probe],
                "mean_resid_l2": [row["mean_resid_l2"] for row in probe],
            }
            curve.append(point)
            print("OVERFIT", point["epoch"], point["loss"], point["step0_p34"], flush=True)
            model.train()
    after = sequence_step0(model, {key: captured[key] for key in FOCUS})
    OVERFIT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"proto": book, "res_cap": CAP, "handoff": HANDOFF, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epoch": 80, "source": str(CKPT), "ids": list(FOCUS)},
        OVERFIT_DIR / "checkpoint_overfit.pt",
    )
    model.eval()
    tapes = manifest_by_id()
    prepared = {row["id"]: blob for row, blob in prepare_horizontal(seated_ref)}
    rollouts = []
    for item_id in FOCUS:
        blob = prepared[item_id]
        meta, _actions, _tokens, _residuals, decoded = load_expert({"id": item_id}, book)
        expert = play(env, blob, decoded, observe=False)
        det = rollout_policy(env, blob, model, book, MeanArgmax(), device)
        stoch = []
        for seed in (11, 22, 33, 44):
            outcome = rollout_policy(env, blob, model, book, np.random.default_rng(seed), device)
            stoch.append({"seed": seed, "reason": outcome["reason"], "success": outcome["success"], "first_token": outcome["first_token"]})
        rollouts.append({
            "id": item_id,
            "strategy": meta["strategy"],
            "expert_replay": {"reason": expert["reason"], "success": expert["success"]},
            "deterministic": {"reason": det["reason"], "success": det["success"], "first_token": det["first_token"]},
            "stochastic": stoch,
        })
        print("OVERFIT_ROLL", item_id, det["reason"], det["first_token"], flush=True)
    payload = {"before": before, "curve": curve, "after": after, "rollouts": rollouts, "checkpoint": "results/diagnostics/horizontal_recatch_root_cause/overfit/checkpoint_overfit.pt"}
    write("sft_overfit.json", payload)
    return payload


def tree_successes(book):
    path = ROOT / "results/grpo/adaptive_hybrid/ic_log.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tapes = manifest_by_id()
    found = []
    for row in rows:
        if row.get("primary_category") != "horizontal_recatch":
            continue
        meta, _actions, tokens, residuals, decoded = load_expert({"id": row["id"]}, book)
        for branch in ("tree_uniform", "tree_entropy"):
            for leaf in row.get(branch) or []:
                if leaf.get("label") != "SUCCESS_12S":
                    continue
                suffix = [int(token) for token in leaf["suffix_tokens"]]
                start = int(leaf["branch_step"])
                expert_slice = [int(token) for token in tokens[start:start + len(suffix)]]
                found.append({
                    "id": row["id"],
                    "strategy": meta["strategy"],
                    "branch": branch,
                    "branch_step": start,
                    "branch_t": leaf.get("branch_t"),
                    "trunk_token": leaf.get("trunk_token"),
                    "prefix_source": leaf.get("prefix_source"),
                    "suffix_tokens": suffix,
                    "suffix_has_token_34": 34 in suffix,
                    "expert_tokens_at_same_index": expert_slice,
                    "token_agreement": int(sum(left == right for left, right in zip(suffix, expert_slice))),
                    "suffix_len": len(suffix),
                    "expert_len": int(len(tokens)),
                    "handoff_t": leaf.get("handoff_t"),
                    "expert_handoff_t": float(len(tokens) * 0.02),
                    "updated": row["updated"],
                    "resample_success": row["resample_success"],
                    "log_has_residual": False,
                    "log_has_trunk_tokens": False,
                })
    payload = {
        "log_rows": len(rows),
        "log_bytes": path.stat().st_size,
        "horizontal_rows": sum(row.get("primary_category") == "horizontal_recatch" for row in rows),
        "successes": found,
        "note": "Independent trunk token lists and residuals were not stored. The policy weights at these visits were not saved. Suffix labels are the trainer's SUCCESS_12S results.",
    }
    write("sft_tree_successes.json", payload)
    return payload


def main():
    before = file_sha(CKPT)
    book = load_codebook()
    census_payload = census(book)
    device = torch.device("cpu")
    model, _saved = load_trained_k48(CKPT, device)
    model.eval()
    stored_teacher_force(model, book, census_payload)
    _sim, _base, seated_ref, _target, _code = _setup()
    env = make_training_env(device, model=model)
    _reports, captured = live_teacher_force(env, model, book, seated_ref)
    interventions(env, model, book, seated_ref)
    tree_successes(book)
    overfit(None, book, captured, env, seated_ref)
    after = file_sha(CKPT)
    write("sft_checkpoint_guard.json", {"before": before, "after": after, "unchanged": before == after, "overfit_exists": (OVERFIT_DIR / "checkpoint_overfit.pt").is_file()})
    if before != after:
        raise RuntimeError("original SFT checkpoint changed")
    print("FOLLOWUP_DONE", flush=True)


if __name__ == "__main__":
    main()
