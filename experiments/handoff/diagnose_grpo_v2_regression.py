"""Read-only diagnosis: Balanced SFT vs Adaptive-G GRPO v2 E5.

Writes under results/diagnostics/grpo_v2_regression/. Never trains or overwrites
protected checkpoints.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.yaml_config import load_yaml
from evaluation import skill_traces as audit
from evaluation.classify_validation_skills import classify_mechanism, prepare_blob
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.adaptive_grpo import (
    _action_surrogate,
    _prepare_blob,
    apply_group_update,
    explore_ic_independent_g,
    independent_rollout,
)
from training.adaptive_hybrid import loss_members
from training.adaptive_independent_g import load_merged_rows
from training.k47_codebook import categorical_kl, group_advantage, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from training.grpo import step_logprob

OUT = ROOT / "results" / "diagnostics" / "grpo_v2_regression"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
E5 = ROOT / "results/grpo/adaptive_g/full_run_v2_natural_sampling/checkpoint_epoch_0005.pt"
V2 = ROOT / "results/grpo/adaptive_g/full_run_v2_natural_sampling"
VALID = ROOT / "results/grpo/adaptive_g/datasets/merged_validation.jsonl"
TRAIN = ROOT / "results/grpo/adaptive_g/datasets/merged_train.jsonl"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
CFG = ROOT / "experiments/config/training/k48/unified_k48_recovery_v3_adaptive_g.yaml"


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def eval_checkpoint(name, ckpt, rows, book, device, seated_ref, limit_s):
    model, saved = load_trained_k48(ckpt, device)
    if not np.array_equal(saved, book):
        raise RuntimeError(f"codebook mismatch for {name}")
    model.eval()
    env = make_training_env(device, model=model)
    per_ic = []
    for row in rows:
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "ref" not in blob:
            try:
                blob = prepare_blob(blob, seated_ref)
            except Exception:
                blob = _prepare_blob(blob, seated_ref)
        print("EVAL", name, row["id"], flush=True)
        if "capture" in blob:
            samples, t0, traced, handoff_t = audit.roll_policy_full(
                env, model, book, blob, device, limit_s,
            )
        else:
            from evaluation.classify_validation_skills import roll_policy_full_ballistic

            samples, t0, traced, handoff_t = roll_policy_full_ballistic(
                env, model, book, blob, device, limit_s,
            )
        skill = classify_mechanism(samples, t0, handoff_t)
        per_ic.append({
            "id": row["id"],
            "source_dataset": row.get("source_dataset"),
            "categories": list(row.get("categories") or []),
            "primary_category": row.get("primary_category"),
            "success": bool(traced.get("success")),
            "reason": traced.get("reason"),
            "mechanism": skill.get("mechanism"),
            "mechanism_confidence": skill.get("confidence"),
            "mechanism_why": "; ".join(skill.get("why") or []),
            "handoff": handoff_t,
            "n_samples": len(samples),
        })
    return model, per_ic


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def deep_trace(name, ckpt, row, book, device, seated_ref, limit_s):
    model, _ = load_trained_k48(ckpt, device)
    model.eval()
    env = make_training_env(device, model=model)
    blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
    if "ref" not in blob:
        try:
            blob = prepare_blob(blob, seated_ref)
        except Exception:
            blob = _prepare_blob(blob, seated_ref)
    if "capture" in blob:
        samples, t0, traced, handoff_t = audit.roll_policy_full(
            env, model, book, blob, device, limit_s,
        )
    else:
        from evaluation.classify_validation_skills import roll_policy_full_ballistic

        samples, t0, traced, handoff_t = roll_policy_full_ballistic(
            env, model, book, blob, device, limit_s,
        )
    skill = classify_mechanism(samples, t0, handoff_t)
    dest = OUT / "traces" / f"{row['id']}__{name}"
    dest.mkdir(parents=True, exist_ok=True)
    audit.write_csv(dest / "trace.csv", samples, t0)
    events = audit.contact_events(samples, t0)
    (dest / "contact_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    video = dest / "rollout.mp4"
    try:
        if "capture" in blob:
            audit.render_video(env, model, book, blob, device, limit_s, video)
        else:
            raise RuntimeError("ballistic IC: skip render_video (capture-only helper)")
    except Exception as exc:  # noqa: BLE001
        video = None
        (dest / "video_error.txt").write_text(str(exc), encoding="utf-8")
    # Token timeline summary
    tokens = []
    for sample in samples:
        if sample.get("token") is None:
            continue
        tokens.append({
            "t_rel": float(sample["t"] - t0),
            "phase": sample.get("phase"),
            "token": int(sample["token"]),
            "nL": sample.get("nL"),
            "nR": sample.get("nR"),
            "both_off": sample.get("both_off"),
            "obj_z": sample.get("obj_z"),
            "aperture": sample.get("aperture"),
        })
    summary = {
        "checkpoint": str(ckpt),
        "id": row["id"],
        "reason": traced.get("reason"),
        "success": bool(traced.get("success")),
        "handoff_t": handoff_t,
        "mechanism": skill,
        "first_tokens": tokens[:40],
        "handoff_token_step": next((i for i, t in enumerate(tokens) if t["token"] == HANDOFF), None),
        "video": None if video is None else str(video.relative_to(OUT)),
        "n_samples": len(samples),
    }
    (dest / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary, samples, t0, handoff_t, tokens


def first_divergence(sft_tokens, grpo_tokens):
    n = min(len(sft_tokens), len(grpo_tokens))
    for i in range(n):
        if int(sft_tokens[i]["token"]) != int(grpo_tokens[i]["token"]):
            return {
                "step": i,
                "t_rel_sft": sft_tokens[i]["t_rel"],
                "t_rel_grpo": grpo_tokens[i]["t_rel"],
                "sft_token": sft_tokens[i]["token"],
                "grpo_token": grpo_tokens[i]["token"],
                "sft_phase": sft_tokens[i]["phase"],
                "grpo_phase": grpo_tokens[i]["phase"],
                "sft_both_off": sft_tokens[i]["both_off"],
                "grpo_both_off": grpo_tokens[i]["both_off"],
            }
    return {"step": None, "note": "token sequences identical through min length", "n": n}


def offline_token_probs(model, book, env, blob, device, limit_s, force_tokens=None):
    """Deterministic rollout collecting per-step full softmax over tokens."""
    from envs.simulation.capture import restore_capture
    from envs.outcomes.horizon_outcome import recovery_failed
    from envs.step_legal import step_legal
    from training.k47_codebook import decode_action

    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    rows = []
    lost_low = 0
    cap = int(round(float(limit_s) / 0.02))
    for step in range(cap):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        probs = torch.softmax(logits[-1].float(), dim=-1).detach().cpu().numpy()
        chosen = int(torch.argmax(logits[-1]).item()) if force_tokens is None else int(force_tokens[step])
        residual = mean[-1].detach().float().cpu().numpy()
        rows.append({
            "step": step,
            "t_rel": float(env.sim.data.time) - t0,
            "chosen": chosen,
            "p_chosen": float(probs[chosen]),
            "p_handoff": float(probs[HANDOFF]),
            "top5": [
                {"token": int(i), "p": float(probs[i])}
                for i in np.argsort(-probs)[:5]
            ],
            "entropy": float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum()),
        })
        if chosen == HANDOFF:
            break
        _res, action = decode_action(book, chosen, residual)
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            break
        if force_tokens is not None and step + 1 >= len(force_tokens):
            break
    return rows


def flatten_grads(model):
    parts = []
    for param in model.parameters():
        if param.grad is None:
            parts.append(torch.zeros(param.numel()))
        else:
            parts.append(param.grad.detach().float().cpu().reshape(-1).clone())
    return torch.cat(parts)


def _group_loss_terms(model, ref_model, kept, advantages, cfg, device):
    clip = float(cfg["clip"])
    trajectory_losses = []
    kl_terms = []
    for member, advantage in zip(kept, advantages):
        if member["truncated"] or abs(float(advantage)) < 1e-8 or not member["records"]:
            continue
        action_losses = []
        for record in member["records"]:
            surrogate, _ratio = _action_surrogate(model, record, advantage, clip, device)
            action_losses.append(surrogate)
            kl_terms.append(categorical_kl(
                model, ref_model,
                torch.tensor(record["obs"], device=device),
                torch.tensor(record["prev"], device=device),
            ))
        trajectory_losses.append(torch.stack(action_losses).mean())
    if not trajectory_losses:
        return None
    surrogate = torch.stack(trajectory_losses).mean()
    kl = torch.stack(kl_terms).mean()
    return surrogate + float(cfg["kl_coef"]) * kl


def gradient_probe(model, ref_model, group, cfg, device):
    """Compute GRPO loss gradient without optimizer.step."""
    kept = loss_members(group)
    advantages, skipped = group_advantage(
        [m["return"] for m in kept],
        [m["truncated"] for m in kept],
    )
    if skipped:
        return {"skipped": True, "grad_norm_pre_clip": 0.0, "grad_norm_post_clip": 0.0,
                "vec_pre": None, "vec_post": None, "advantages": advantages.tolist()}
    model.zero_grad(set_to_none=True)
    loss = _group_loss_terms(model, ref_model, kept, advantages, cfg, device)
    if loss is None:
        return {"skipped": True, "grad_norm_pre_clip": 0.0, "grad_norm_post_clip": 0.0,
                "vec_pre": None, "vec_post": None, "advantages": advantages.tolist()}
    loss.backward()
    pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
    vec_pre = flatten_grads(model)
    # Recompute for post-clip vector under training clip=1.0
    model.zero_grad(set_to_none=True)
    loss2 = _group_loss_terms(model, ref_model, kept, advantages, cfg, device)
    loss2.backward()
    post = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    vec_post = flatten_grads(model)
    return {
        "skipped": False,
        "loss": float(loss2.detach()),
        "grad_norm_pre_clip": pre,
        "grad_norm_post_clip": post,
        "group": len(kept),
        "success": sum(int(m["label"] == "SUCCESS_12S") for m in kept),
        "advantages": [float(a) for a in advantages],
        "vec_pre": vec_pre,
        "vec_post": vec_post,
    }


def cosine(a, b):
    if a is None or b is None:
        return None
    denom = float(torch.norm(a) * torch.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


def analyze_ic_log():
    logs = [json.loads(line) for line in (V2 / "ic_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    reward_rows = []
    for item in logs:
        found = bool(item.get("found_success"))
        bucket = "A_has_success" if found else "B_all_failure"
        g = int(item.get("final_g") or item.get("group_size") or 0)
        rewards = item.get("rewards") or []
        labels = item.get("labels") or []
        adv = (item.get("update_stats") or {}).get("advantages") or []
        rows.append({
            "epoch": item["epoch"],
            "id": item["id"],
            "source_dataset": item.get("source_dataset"),
            "bucket": bucket,
            "final_g": g,
            "stop_bucket": item.get("stop_bucket"),
            "success_count": item.get("success_count"),
            "failure_count": g - int(item.get("success_count") or 0),
            "return_mean": item.get("return_mean"),
            "return_std": item.get("return_std"),
            "return_max": item.get("return_max"),
            "return_min": item.get("return_min"),
            "loss": (item.get("update_stats") or {}).get("loss"),
            "kl": (item.get("update_stats") or {}).get("kl"),
            "grad_norm": (item.get("update_stats") or {}).get("grad_norm"),
            "optimizer_updated": item.get("optimizer_updated"),
            "recovery_physics": item.get("recovery_physics"),
        })
        for idx, (rew, lab) in enumerate(zip(rewards, labels)):
            a = adv[idx] if idx < len(adv) else None
            reward_rows.append({
                "epoch": item["epoch"],
                "id": item["id"],
                "source_dataset": item.get("source_dataset"),
                "bucket": bucket,
                "traj_index": idx,
                "label": lab,
                "is_success": lab == "SUCCESS_12S",
                "reward": rew,
                "advantage": a,
                "final_g": g,
            })
    return logs, rows, reward_rows


def summarize_groups(group_rows):
    out = []
    keyed = defaultdict(list)
    for r in group_rows:
        keyed[(r["epoch"], r["source_dataset"], r["bucket"])].append(r)
    for key, items in sorted(keyed.items()):
        epoch, src, bucket = key
        gs = [i["final_g"] for i in items]
        rets = [i["return_mean"] for i in items if i["return_mean"] is not None]
        stds = [i["return_std"] for i in items if i["return_std"] is not None]
        grads = [i["grad_norm"] for i in items if i["grad_norm"] is not None]
        kls = [i["kl"] for i in items if i["kl"] is not None]
        losses = [i["loss"] for i in items if i["loss"] is not None]
        out.append({
            "epoch": epoch,
            "source_dataset": src,
            "bucket": bucket,
            "n_groups": len(items),
            "mean_g": float(np.mean(gs)),
            "total_rollouts": int(sum(gs)),
            "total_env_steps": int(sum(i.get("recovery_physics") or 0 for i in items)),
            "updated": sum(int(i.get("optimizer_updated")) for i in items),
            "mean_return_mean": None if not rets else float(np.mean(rets)),
            "mean_return_std": None if not stds else float(np.mean(stds)),
            "mean_grad_norm": None if not grads else float(np.mean(grads)),
            "mean_kl": None if not kls else float(np.mean(kls)),
            "mean_loss": None if not losses else float(np.mean(losses)),
        })
    totals = defaultdict(list)
    for r in group_rows:
        totals[r["bucket"]].append(r)
    summary = {}
    total_roll = max(sum(i["final_g"] for i in group_rows), 1)
    total_env = max(sum(i.get("recovery_physics") or 0 for i in group_rows), 1)
    for bucket, items in totals.items():
        summary[bucket] = {
            "n_groups": len(items),
            "fraction_of_updates": len(items) / max(len(group_rows), 1),
            "total_rollouts": int(sum(i["final_g"] for i in items)),
            "rollout_fraction": sum(i["final_g"] for i in items) / total_roll,
            "total_env_steps": int(sum(i.get("recovery_physics") or 0 for i in items)),
            "env_step_fraction": sum(i.get("recovery_physics") or 0 for i in items) / total_env,
        }
    return out, summary


def reward_consistency(reward_rows):
    mixed = [r for r in reward_rows if r["bucket"] == "A_has_success"]
    # per group check failure reward > success max
    by_group = defaultdict(list)
    for r in mixed:
        by_group[(r["epoch"], r["id"])].append(r)
    inversions = []
    for key, items in by_group.items():
        succ = [i for i in items if i["is_success"]]
        fail = [i for i in items if not i["is_success"]]
        if not succ or not fail:
            continue
        max_fail = max(i["reward"] for i in fail if i["reward"] is not None)
        min_succ = min(i["reward"] for i in succ if i["reward"] is not None)
        if max_fail > min_succ + 1e-9:
            inversions.append({
                "epoch": key[0], "id": key[1],
                "max_fail_reward": max_fail,
                "min_success_reward": min_succ,
                "gap": max_fail - min_succ,
            })
    # all-failure: high reward failures
    all_fail = [r for r in reward_rows if r["bucket"] == "B_all_failure"]
    high = sorted(all_fail, key=lambda r: -(r["reward"] if r["reward"] is not None else -1e9))[:30]
    return inversions, high


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(CFG)
    book = load_codebook(CODEBOOK)
    valid_rows = load_merged_rows(VALID)
    train_rows = load_merged_rows(TRAIN)
    from data_generation.oracles.plan_b.release_recatch import _setup
    _sim, _base, seated_ref, _target, _code = _setup()
    limit_s = SELECTED_LIMIT_S

    # Phase 1: re-eval (reuse if already written by a prior partial run)
    if (OUT / "per_ic_comparison.csv").is_file() and (OUT / "phase1_counts.json").is_file():
        print("PHASE1_REUSE", flush=True)
        import csv as _csv
        with (OUT / "per_ic_comparison.csv").open(encoding="utf-8", newline="") as handle:
            comparison = list(_csv.DictReader(handle))
        for row in comparison:
            row["sft_success"] = int(row["sft_success"])
            row["grpo_success"] = int(row["grpo_success"])
        counts = Counter(r["transition"] for r in comparison)
        sft_model, _ = load_trained_k48(SFT, device)
        e5_model, _ = load_trained_k48(E5, device)
        sft_model.eval()
        e5_model.eval()
    else:
        sft_model, sft_ic = eval_checkpoint("sft", SFT, valid_rows, book, device, seated_ref, limit_s)
        e5_model, e5_ic = eval_checkpoint("e5", E5, valid_rows, book, device, seated_ref, limit_s)
        sft_map = {r["id"]: r for r in sft_ic}
        e5_map = {r["id"]: r for r in e5_ic}
        comparison = []
        counts = Counter()
        for row in valid_rows:
            sid = row["id"]
            a, b = sft_map[sid], e5_map[sid]
            if a["success"] and not b["success"]:
                tag = "SFT_ok_GRPO_fail"
            elif (not a["success"]) and b["success"]:
                tag = "SFT_fail_GRPO_ok"
            elif a["success"] and b["success"]:
                tag = "both_ok"
            else:
                tag = "both_fail"
            counts[tag] += 1
            comparison.append({
                "ic_id": sid,
                "source": row.get("source_dataset"),
                "categories": "|".join(row.get("categories") or []),
                "sft_success": int(a["success"]),
                "grpo_success": int(b["success"]),
                "transition": tag,
                "sft_mechanism": a["mechanism"],
                "grpo_mechanism": b["mechanism"],
                "sft_reason": a["reason"],
                "grpo_reason": b["reason"],
                "sft_handoff": a["handoff"],
                "grpo_handoff": b["handoff"],
            })
        write_csv(OUT / "per_ic_comparison.csv", comparison)
        (OUT / "phase1_counts.json").write_text(json.dumps({
            "counts": dict(counts),
            "sft_success": sum(r["sft_success"] for r in comparison),
            "grpo_success": sum(r["grpo_success"] for r in comparison),
            "protocol": {
                "obs": "39D via env.observation",
                "action": "argmax logits (deterministic)",
                "SUCCESS_12S": "roll_policy_full / finish_recovery unchanged",
                "models_eval_mode": True,
                "device": str(device),
            },
        }, indent=2), encoding="utf-8")

    # Phase 3: epoch transitions from saved validations
    epoch_rows = []
    for epoch in range(0, 6):
        if epoch == 0:
            path = V2 / "baseline_sft_ref" / "baseline_sft.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            per = payload["per_ic"]
            label = "SFT"
        else:
            path = V2 / f"validation_epoch_{epoch:04d}.json"
            if not path.is_file():
                continue
            per = json.loads(path.read_text(encoding="utf-8"))["per_ic"]
            label = f"E{epoch}"
        by_id = {r["id"]: r for r in per}
        for row in valid_rows:
            item = by_id[row["id"]]
            epoch_rows.append({
                "epoch_label": label,
                "epoch_index": epoch,
                "ic_id": row["id"],
                "source": row.get("source_dataset"),
                "success": int(bool(item["success"])),
                "reason": item.get("reason"),
                "mechanism": item.get("mechanism"),
                "handoff": item.get("handoff"),
            })
    write_csv(OUT / "epoch_transition.csv", epoch_rows)

    # Phase 2: deep traces for transitions
    transitions = [r for r in comparison if r["transition"] in ("SFT_ok_GRPO_fail", "SFT_fail_GRPO_ok")]
    by_valid = {r["id"]: r for r in valid_rows}
    divergence_report = {}
    for tr in transitions:
        row = by_valid[tr["ic_id"]]
        sft_sum, _sft_s, _sft_t0, _sft_h, sft_tok = deep_trace("sft", SFT, row, book, device, seated_ref, limit_s)
        e5_sum, _e5_s, _e5_t0, _e5_h, e5_tok = deep_trace("e5", E5, row, book, device, seated_ref, limit_s)
        div = first_divergence(sft_tok, e5_tok)
        # Offline probs at SFT-forced prefix then free
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "ref" not in blob:
            try:
                blob = prepare_blob(blob, seated_ref)
            except Exception:
                blob = _prepare_blob(blob, seated_ref)
        env_s = make_training_env(device, model=sft_model)
        env_g = make_training_env(device, model=e5_model)
        # Compare distributions on shared first K states by forcing SFT tokens for both
        force = [t["token"] for t in sft_tok if t["token"] != HANDOFF][:20]
        force = force + [HANDOFF]  # include handoff if present in sft path length
        # Better: step-by-step compare using each model's own obs from its rollout at divergence step
        # Fixed-state KL: use SFT history at divergence step
        kl_audit = {"divergence": div}
        if div.get("step") is not None and div["step"] > 0 and "capture" in blob:
            # Rebuild SFT state at divergence by replaying SFT tokens
            force_prefix = [t["token"] for t in sft_tok[: div["step"]]]
            sft_model.eval()
            e5_model.eval()
            env = make_training_env(device, model=sft_model)
            from envs.simulation.capture import restore_capture
            from envs.step_legal import step_legal
            from training.k47_codebook import decode_action
            from envs.outcomes.horizon_outcome import recovery_failed
            restore_capture(env, blob["capture"])
            lost_low = 0
            for tok in force_prefix:
                obs = np.asarray(env.observation(), np.float32)
                env._obs_seq.append(obs)
                env._prev_seq.append(env._prev.copy())
                with torch.no_grad():
                    logits, mean, _ = sft_model(
                        torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                        torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
                    )
                residual = mean[-1].detach().float().cpu().numpy()
                if tok == HANDOFF:
                    break
                _r, action = decode_action(book, tok, residual)
                executed = step_legal(env.sim, action)
                env._prev = np.asarray(executed, np.float32)
                failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
                if failed:
                    break
            obs = np.asarray(env.observation(), np.float32)
            env._obs_seq.append(obs)
            env._prev_seq.append(env._prev.copy())
            obs_t = torch.tensor(np.stack(env._obs_seq[-160:]), device=device)
            prev_t = torch.tensor(np.stack(env._prev_seq[-160:]), device=device)
            with torch.no_grad():
                logits_s, _, _ = sft_model(obs_t, prev_t)
                logits_g, _, _ = e5_model(obs_t, prev_t)
            ps = torch.softmax(logits_s[-1].float(), dim=-1)
            pg = torch.softmax(logits_g[-1].float(), dim=-1)
            # KL(SFT || GRPO) and KL(GRPO || SFT)
            kl_s_g = float((ps * (torch.log(ps + 1e-12) - torch.log(pg + 1e-12))).sum())
            kl_g_s = float((pg * (torch.log(pg + 1e-12) - torch.log(ps + 1e-12))).sum())
            top_s = torch.topk(ps, 8)
            top_g = torch.topk(pg, 8)
            kl_audit["fixed_state_at_divergence"] = {
                "kl_sft_parallel_grpo": kl_s_g,
                "kl_grpo_parallel_sft": kl_g_s,
                "p_handoff_sft": float(ps[HANDOFF]),
                "p_handoff_grpo": float(pg[HANDOFF]),
                "sft_argmax": int(torch.argmax(ps)),
                "grpo_argmax": int(torch.argmax(pg)),
                "sft_top8": [{"token": int(i), "p": float(p)} for i, p in zip(top_s.indices, top_s.values)],
                "grpo_top8": [{"token": int(i), "p": float(p)} for i, p in zip(top_g.indices, top_g.values)],
                "delta_p_sft_token": float(pg[int(torch.argmax(ps))] - ps[int(torch.argmax(ps))]),
            }
        divergence_report[tr["ic_id"]] = {
            "transition": tr["transition"],
            "sft": {k: sft_sum[k] for k in ("success", "reason", "handoff_t", "mechanism") if k in sft_sum},
            "e5": {k: e5_sum[k] for k in ("success", "reason", "handoff_t", "mechanism") if k in e5_sum},
            "kl_action": kl_audit,
        }
    (OUT / "transition_deep_analysis.json").write_text(
        json.dumps(_jsonable(divergence_report), indent=2), encoding="utf-8",
    )

    # Phase 4–6: ic_log
    logs, group_rows, reward_rows = analyze_ic_log()
    write_csv(OUT / "group_success_statistics.csv", group_rows)
    write_csv(OUT / "reward_advantage_audit.csv", reward_rows)
    group_by_epoch_src, group_totals = summarize_groups(group_rows)
    write_csv(OUT / "group_bucket_summary.csv", group_by_epoch_src)
    inversions, high_fail = reward_consistency(reward_rows)
    (OUT / "reward_inversions.json").write_text(json.dumps({
        "mixed_group_fail_reward_above_success": inversions[:50],
        "n_inversions": len(inversions),
        "all_failure_highest_rewards": high_fail,
        "group_totals": group_totals,
    }, indent=2), encoding="utf-8")

    # Advantage anomalies in B groups
    b_anoms = []
    for item in logs:
        if item.get("found_success"):
            continue
        std = item.get("return_std")
        adv = (item.get("update_stats") or {}).get("advantages") or []
        if std is not None and std < 1e-8:
            b_anoms.append({"epoch": item["epoch"], "id": item["id"], "issue": "near_zero_std", "std": std, "updated": item.get("optimizer_updated")})
        if adv and max(abs(a) for a in adv) > 5:
            b_anoms.append({"epoch": item["epoch"], "id": item["id"], "issue": "large_advantage", "max_abs_adv": max(abs(a) for a in adv)})
    (OUT / "all_failure_anomalies.json").write_text(json.dumps(b_anoms, indent=2), encoding="utf-8")

    # Phase 5: gradient probes at E5 (diagnostic, not historical)
    print("GRADIENT_PROBE", flush=True)
    e5_train, _ = load_trained_k48(E5, device)
    e5_train.train()
    ref_sft, _ = load_trained_k48(SFT, device)
    ref_sft.eval()
    for p in ref_sft.parameters():
        p.requires_grad_(False)
    env = make_training_env(device, model=e5_train)
    # Pick latest-epoch examples
    e5_logs = [r for r in logs if r["epoch"] == 5]
    succ_ex = next(r for r in e5_logs if r.get("found_success") and r.get("source_dataset") == "heuristic")
    fail_ex = next(r for r in e5_logs if (not r.get("found_success")) and r.get("source_dataset") == "hard" and r.get("final_g") == 64)
    train_by_id = {r["id"]: r for r in train_rows}

    def probe_ic(row_meta, seed):
        row = train_by_id[row_meta["id"]]
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "ref" not in blob:
            blob = _prepare_blob(blob, seated_ref)
        log, group = explore_ic_independent_g(
            env, e5_train, book, blob, device, limit_s, "v2", seed, row["id"],
        )
        stats = gradient_probe(e5_train, ref_sft, group, cfg, device)
        return log, stats

    print("PROBE_SUCCESS", succ_ex["id"], flush=True)
    succ_log, succ_grad = probe_ic(succ_ex, 900001)
    print("PROBE_FAIL", fail_ex["id"], flush=True)
    fail_log, fail_grad = probe_ic(fail_ex, 900002)
    # Also a hard success if any
    hard_succ = next((r for r in e5_logs if r.get("found_success") and r.get("source_dataset") == "hard"), None)
    hard_grad = None
    hard_log = None
    if hard_succ:
        print("PROBE_HARD_SUCCESS", hard_succ["id"], flush=True)
        hard_log, hard_grad = probe_ic(hard_succ, 900003)

    grad_audit = {
        "note": "Diagnostic gradients at frozen E5 checkpoint with freshly sampled groups. NOT historical training gradients.",
        "success_example": {
            "id": succ_ex["id"],
            "logged_stop": succ_ex.get("stop_bucket"),
            "probe_stop": succ_log.get("stop_bucket"),
            "probe_final_g": succ_log.get("final_g"),
            "grad_norm_pre": succ_grad.get("grad_norm_pre_clip"),
            "grad_norm_post": succ_grad.get("grad_norm_post_clip"),
            "skipped": succ_grad.get("skipped"),
            "success_count": succ_grad.get("success"),
        },
        "all_failure_example": {
            "id": fail_ex["id"],
            "logged_stop": fail_ex.get("stop_bucket"),
            "probe_stop": fail_log.get("stop_bucket"),
            "probe_final_g": fail_log.get("final_g"),
            "grad_norm_pre": fail_grad.get("grad_norm_pre_clip"),
            "grad_norm_post": fail_grad.get("grad_norm_post_clip"),
            "skipped": fail_grad.get("skipped"),
            "success_count": fail_grad.get("success"),
            "return_std_probe": fail_log.get("return_std"),
            "advantages": fail_grad.get("advantages"),
        },
        "cosine_success_vs_all_failure_preclip": cosine(succ_grad.get("vec_pre"), fail_grad.get("vec_pre")),
        "cosine_success_vs_all_failure_postclip": cosine(succ_grad.get("vec_post"), fail_grad.get("vec_post")),
        "code_loss_reduction": {
            "file": "training/adaptive_grpo.py:apply_group_update",
            "trajectory_mean_then_group_mean": True,
            "g64_not_8x_surrogate": True,
            "one_adam_step_per_ic": True,
            "grad_clip": 1.0,
            "kl_coef": float(cfg["kl_coef"]),
            "kl_is_action_mean_within_group": True,
        },
    }
    if hard_grad is not None:
        grad_audit["hard_success_example"] = {
            "id": hard_succ["id"],
            "grad_norm_pre": hard_grad.get("grad_norm_pre_clip"),
            "grad_norm_post": hard_grad.get("grad_norm_post_clip"),
            "cosine_vs_all_failure_postclip": cosine(hard_grad.get("vec_post"), fail_grad.get("vec_post")),
            "cosine_vs_heuristic_success_postclip": cosine(hard_grad.get("vec_post"), succ_grad.get("vec_post")),
        }
    (OUT / "gradient_audit.md").write_text(
        "# Gradient audit\n\n"
        + "```json\n" + json.dumps(_jsonable({k: v for k, v in grad_audit.items() if "vec" not in str(k)}), indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "gradient_audit.json").write_text(
        json.dumps(_jsonable({k: v for k, v in grad_audit.items() if not str(k).startswith("vec") and "vec_" not in str(k)}), indent=2),
        encoding="utf-8",
    )

    # KL definition from training summaries
    kl_md = [
        "# KL / action-distribution audit",
        "",
        "## Training-logged KL",
        "",
        "- Reference: Balanced SFT init (`init_checkpoint`), frozen.",
        "- Within each IC update: mean categorical KL over **policy actions** in contributing trajectories",
        "  (`categorical_kl` on token distribution; see `apply_group_update`).",
        "- Not trajectory-equal for KL (action-mean). Surrogate is trajectory-equal.",
        "- Epoch summary `kl_mean` = mean of per-IC update KL values that were finite.",
        "- v2 E5 `kl_mean` ≈ 0.085 (from summary_epoch_0005.json).",
        "",
        "## Fixed-state KL on transition ICs",
        "",
        "```json",
        json.dumps(_jsonable(divergence_report), indent=2),
        "```",
        "",
    ]
    (OUT / "kl_action_distribution_audit.md").write_text("\n".join(kl_md), encoding="utf-8")

    meta = {
        "wall_s": time.perf_counter() - started,
        "transition_ics": [r["ic_id"] for r in transitions],
        "counts": dict(counts),
        "group_totals": group_totals,
        "n_reward_inversions": len(inversions),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("DIAG_DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
