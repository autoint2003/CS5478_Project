"""Closed-loop comparison of GRPO v2 checkpoints u174, u261, and u783.

Does not train and does not edit reward, physics, or checkpoints.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.legacy.demo.diagnose_sft_epoch40_closed_loop import (
    LIMIT,
    _apply_motion,
    _brief,
    _sample,
    roll_traced,
)
from training.history_aware_critic_ablation import restore_capture
from envs.horizon_outcome import finish_recovery, recovery_failed, roll_policy
from training.unified_k47 import load_codebook
from training.unified_k48 import HANDOFF, load_trained_k48
from training.autonomous_episode import make_training_env

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/evaluations/raw/grpo_v2_u174_u783"
CKPT = {
    "u174": ROOT / "results/grpo/v2/checkpoint_u0174.pt",
    "u261": ROOT / "results/grpo/v2/checkpoint_u0261.pt",
    "u783": ROOT / "results/grpo/v2/checkpoint_u0783.pt",
}
FOCUS = ("k48r_valid_00004", "k48r_valid_00168")


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _windows(env, device):
    obs = np.asarray(env.observation(), np.float32)
    env._obs_seq.append(obs)
    env._prev_seq.append(env._prev.copy())
    window_obs = torch.tensor(np.stack(env._obs_seq[-160:]), device=device)
    window_prev = torch.tensor(np.stack(env._prev_seq[-160:]), device=device)
    return window_obs, window_prev


def _decide_full(model, window_obs, window_prev):
    with torch.no_grad():
        logits, mean, _act = model(window_obs, window_prev)
    last = logits[-1].float()
    probs = torch.softmax(last, dim=-1)
    token = int(torch.argmax(last).item())
    order = torch.argsort(probs, descending=True)
    rank = int((order == HANDOFF).nonzero()[0].item()) + 1
    return {
        "token": token,
        "probs": probs.detach().cpu().numpy(),
        "p47": float(probs[HANDOFF].item()),
        "rank47": rank,
        "mean": mean[-1].detach().float().cpu().numpy(),
        "logits": last.detach().cpu().numpy(),
    }


def roll_detail(env, model, book, blob, device, handoff_model=None, mask_handoff=False):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    how = None
    steps = 0
    rows = []
    tokens = []
    had_contact = False
    lost_at = None
    rebuilt_at = None
    for _ in range(200):
        if float(env.sim.data.time) - t0 >= LIMIT - 1e-9:
            how = "forced"
            break
        window_obs, window_prev = _windows(env, device)
        motion = _decide_full(model, window_obs, window_prev)
        hand = motion if handoff_model is None else _decide_full(handoff_model, window_obs, window_prev)
        state = _sample(env)
        token = motion["token"]
        source = "policy_argmax"
        if mask_handoff or (handoff_model is not None and hand["token"] != HANDOFF and token == HANDOFF):
            token = int(np.argmax(motion["probs"][:HANDOFF]))
            source = "motion_head_argmax_token47_masked"
        elif handoff_model is not None and hand["token"] == HANDOFF:
            state.update(state_decision(motion, hand, source="u174_argmax_handoff"))
            state["phase"] = "handoff"
            rows.append(state)
            how = "early"
            break
        if token == HANDOFF and not mask_handoff and handoff_model is None:
            state.update(state_decision(motion, hand, source="policy_argmax"))
            state["phase"] = "handoff"
            rows.append(state)
            how = "early"
            break
        _residual, action = __import__("training.unified_k47", fromlist=["decode_action"]).decode_action(book, token, motion["mean"])
        executed = _apply_motion(env, book, token, motion["mean"])
        after = _sample(env)
        state.update(state_decision(motion, hand, source))
        state["action"] = executed
        state["residual"] = np.asarray(_residual, np.float32)
        state["grip"] = float(executed[6])
        state["wrist_cmd"] = (executed[3:6] * 4.0).astype(np.float32)
        state["nL_after"] = int(after["nL"])
        state["nR_after"] = int(after["nR"])
        rows.append(state)
        tokens.append(int(token))
        steps += 1
        contacts = int(after["nL"]) + int(after["nR"])
        if contacts > 0:
            had_contact = True
        elif had_contact and lost_at is None:
            lost_at = float(env.sim.data.time) - t0
        if lost_at is not None and rebuilt_at is None and contacts > 0:
            rebuilt_at = float(env.sim.data.time) - t0
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "reason": "DROP_DURING_RECOVERY",
                "success": False,
                "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "recovery_steps": steps,
                "nominal_elapsed": 0.0,
                "lift_completed": False,
            }, rows, tokens, lost_at, rebuilt_at
    if how is None:
        how = "forced"
    out = finish_recovery(env, blob["ref"], t0, how)
    out["recovery_steps"] = steps
    return out, rows, tokens, lost_at, rebuilt_at


def state_decision(motion, hand, source):
    return {
        "token": int(motion["token"]),
        "executed_source": source,
        "p47": float(motion["p47"]),
        "rank47": int(motion["rank47"]),
        "handoff_p47": float(hand["p47"]),
        "handoff_rank47": int(hand["rank47"]),
        "handoff_token": int(hand["token"]),
    }


def mechanism(rows):
    motion = [row for row in rows if row.get("action") is not None]
    if len(motion) < 2:
        return "inconclusive"
    left = np.array([row["nL_after"] for row in motion])
    right = np.array([row["nR_after"] for row in motion])
    contact = (left + right) > 0
    run = longest = 0
    for touched in contact:
        run = 0 if touched else run + 1
        longest = max(longest, run)
    hand = np.stack([row["p_hand"] for row in motion])
    gravity = np.stack([row["g_h"] for row in motion])
    relative = np.stack([row["p_rel"] for row in motion])
    bilateral = float(np.mean((left > 0) & (right > 0)))
    hand_move = float(np.linalg.norm(hand[-1] - hand[0]))
    dgx = float(gravity[-1, 0] - gravity[0, 0])
    drx = float((relative[-1, 0] - relative[0, 0]) * 1000.0)
    recatch = bool(longest >= 3 and contact[-1] and np.any(~contact) and hand_move > 0.01)
    gravity_slide = bool(bilateral >= 0.5 and longest < 3 and abs(dgx) >= 3.0 and abs(drx) >= 2.0 and drx * gravity[-1, 0] > 0)
    preserving = bool(bilateral >= 0.7 and longest < 3 and abs(drx) >= 3.0 and not gravity_slide)
    labels = [name for name, flag in (("recatch", recatch), ("gravity-assisted sliding", gravity_slide), ("contact-preserving sliding", preserving)) if flag]
    if len(labels) > 1:
        return "mixed"
    return labels[0] if labels else "inconclusive"


def summarize(out, rows, tokens, lost_at, rebuilt_at):
    decisions = [row for row in rows if "p47" in row]
    motion = [row for row in rows if row.get("action") is not None]
    first = decisions[0] if decisions else None
    handoff_row = next((row for row in decisions if row.get("phase") == "handoff"), None)
    grips = [float(row["grip"]) for row in motion]
    residuals = [row["residual"] for row in motion if row.get("residual") is not None]
    record = {
        **_brief(out),
        "p47_first": None if first is None else first["p47"],
        "rank47_first": None if first is None else first["rank47"],
        "token_first": None if first is None else first["token"],
        "p47_at_handoff": None if handoff_row is None else handoff_row["p47"],
        "rank47_at_handoff": None if handoff_row is None else handoff_row["rank47"],
        "p47_max": None if not decisions else float(max(row["p47"] for row in decisions)),
        "token_counts": dict(Counter(tokens)),
        "mean_grip": None if not grips else float(np.mean(grips)),
        "first_grip": None if not grips else grips[0],
        "mean_abs_residual": None if not residuals else float(np.mean(np.abs(np.stack(residuals)))),
        "contact_lost_s": lost_at,
        "contact_rebuilt_s": rebuilt_at,
        "mechanism": mechanism(rows),
    }
    return record


def splice_handoff(motion_model, handoff_model):
    with torch.no_grad():
        motion_model.logits.weight.data[HANDOFF].copy_(handoff_model.logits.weight.data[HANDOFF])
        motion_model.logits.bias.data[HANDOFF].copy_(handoff_model.logits.bias.data[HANDOFF])
    return motion_model


def log_report():
    rows = _rows(ROOT / "results/grpo/v2/training_log.jsonl")
    updates = [int(row["update"]) for row in rows]
    gaps = [updates[i] for i in range(1, len(updates)) if updates[i] != updates[i - 1] + 1]
    by_epoch = {}
    for row in rows:
        epoch = (int(row["update"]) - 1) // 87 + 1
        bucket = by_epoch.setdefault(epoch, {"success": 0, "group": 0, "kl": [], "reward_std": [], "skipped": 0, "loss_none": 0, "totals": Counter()})
        bucket["success"] += int(row["success"])
        bucket["group"] += int(row["group"])
        if row.get("kl") is not None:
            bucket["kl"].append(float(row["kl"]))
        if row.get("reward_std") is not None:
            bucket["reward_std"].append(float(row["reward_std"]))
        bucket["skipped"] += int(bool(row.get("advantage_skipped")))
        bucket["loss_none"] += int(row.get("policy_loss") is None)
        bucket["totals"][int(row["total"])] += 1
    epochs = []
    for epoch, bucket in sorted(by_epoch.items()):
        epochs.append({
            "epoch": epoch,
            "updates": bucket["totals"].total() if hasattr(bucket["totals"], "total") else sum(bucket["totals"].values()),
            "rollout_success": bucket["success"],
            "rollouts": bucket["group"],
            "success_rate": bucket["success"] / max(bucket["group"], 1),
            "kl_mean": float(np.mean(bucket["kl"])) if bucket["kl"] else None,
            "kl_max": float(np.max(bucket["kl"])) if bucket["kl"] else None,
            "reward_std_mean": float(np.mean(bucket["reward_std"])) if bucket["reward_std"] else None,
            "advantage_skipped": bucket["skipped"],
            "updates_without_loss": bucket["loss_none"],
            "logged_total": dict(bucket["totals"]),
        })
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return {
        "rows": len(rows),
        "first_update": updates[0] if updates else None,
        "last_update": updates[-1] if updates else None,
        "gaps": gaps[:10],
        "duplicate_updates": int(len(updates) - len(set(updates))),
        "logged_keys": sorted(keys),
        "missing": ["token 47 probability", "policy entropy", "gradient norm", "parameter update norm"],
        "epochs": epochs,
    }


def checkpoint_report(device):
    book = load_codebook(ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy")
    found = []
    models = {}
    for name, path in CKPT.items():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model, proto = load_trained_k48(path, device)
        found.append({
            "name": name,
            "keys": sorted(saved.keys()),
            "update": int(saved.get("update", -1)),
            "reward_version": saved.get("reward_version"),
            "handoff_field": int(saved.get("handoff", -1)),
            "logits": list(model.logits.weight.shape),
            "handoff_index": int(model.handoff_index),
            "proto_matches_checkpoint": bool(np.array_equal(proto, np.asarray(saved["proto"], np.float32))),
            "proto_matches_config_codebook": bool(np.array_equal(proto, book)),
        })
        models[name] = (model, proto)
    return found, models, book


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    (OUT / "training_log_summary.json").write_text(json.dumps(log_report(), indent=2), encoding="utf-8")
    checks, models, _config_book = checkpoint_report(device)
    (OUT / "checkpoint_load_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    train = _rows(ROOT / "data/grpo_initial_conditions.jsonl")
    valid = _rows(ROOT / "data/valid_recovery.jsonl")
    blobs = {}
    for row in train + valid:
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blobs[row["id"]] = pickle.load(handle)
    env = make_training_env(device, model=models["u174"][0])
    episodes = []
    for split, manifest in (("valid", valid), ("train", train)):
        for row in manifest:
            episodes.append((split, row))
    results = {name: [] for name in CKPT}
    for name, (model, book) in models.items():
        model.eval()
        for split, row in episodes:
            blob = blobs[row["id"]]
            out, trace, tokens, lost_at, rebuilt_at = roll_detail(env, model, book, blob, device)
            if row["id"] in FOCUS:
                official = roll_policy(env, model, book, blob, device, cap_steps=200, limit_s=LIMIT)
                if official.get("reason") != out.get("reason") or int(official.get("recovery_steps") or -1) != int(out.get("recovery_steps") or -2):
                    raise RuntimeError(f"trace diverged from roll_policy on {name} {row['id']}")
            record = {"id": row["id"], "split": split, "checkpoint": name, **summarize(out, trace, tokens, lost_at, rebuilt_at)}
            results[name].append(record)
            print("EVAL", name, split, row["id"], record["reason"], record["handoff"], record["recovery_steps"], flush=True)
    (OUT / "closed_loop.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    by = {name: {row["id"]: row for row in rows} for name, rows in results.items()}
    regress = [row["id"] for row in results["u174"] if row["success"] and not by["u783"][row["id"]]["success"]]
    targets = []
    for row in valid:
        targets.append(row)
    for episode_id in regress:
        if episode_id not in {row["id"] for row in targets}:
            targets.append(next(row for _split, row in episodes if row["id"] == episode_id))
    u174, book = models["u174"]
    u783, book783 = models["u783"]
    if not np.array_equal(book, book783):
        raise RuntimeError("checkpoints do not share the motion codebook")
    hybrid = load_trained_k48(CKPT["u783"], device)[0]
    splice_handoff(hybrid, u174)
    hybrid.eval()
    u783.eval()
    counter = []
    for row in targets:
        blob = blobs[row["id"]]
        for label, runner in (
            ("u783_original", lambda: roll_detail(env, u783, book, blob, device)),
            ("u783_mask_early_handoff", lambda: roll_detail(env, u783, book, blob, device, mask_handoff=True)),
            ("u174_handoff_decides_u783_motion", lambda: roll_detail(env, u783, book, blob, device, handoff_model=u174)),
            ("u783_body_u174_logit_row", lambda: roll_detail(env, hybrid, book, blob, device)),
        ):
            out, trace, tokens, lost_at, rebuilt_at = runner()
            item = {"id": row["id"], "variant": label, **summarize(out, trace, tokens, lost_at, rebuilt_at)}
            counter.append(item)
            print("CF", label, row["id"], item["reason"], item["recovery_steps"], flush=True)
    (OUT / "counterfactuals.json").write_text(json.dumps(counter, indent=2), encoding="utf-8")
    (OUT / "u174_success_u783_fail.json").write_text(json.dumps(regress, indent=2), encoding="utf-8")
    print("REGRESSION_DONE", len(regress), flush=True)


def counterfactuals():
    device = torch.device("cpu")
    results = json.loads((OUT / "closed_loop.json").read_text(encoding="utf-8"))
    by = {name: {row["id"]: row for row in rows} for name, rows in results.items()}
    train = _rows(ROOT / "data/grpo_initial_conditions.jsonl")
    valid = _rows(ROOT / "data/valid_recovery.jsonl")
    regress = [row["id"] for row in results["u174"] if row["success"] and not by["u783"][row["id"]]["success"]]
    (OUT / "u174_success_u783_fail.json").write_text(json.dumps(regress, indent=2), encoding="utf-8")
    wanted = {row["id"] for row in valid} | set(regress)
    rows = [row for row in valid + train if row["id"] in wanted]
    blobs = {}
    for row in rows:
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blobs[row["id"]] = pickle.load(handle)
    u174, book = load_trained_k48(CKPT["u174"], device)
    u783, book783 = load_trained_k48(CKPT["u783"], device)
    if not np.array_equal(book, book783):
        raise RuntimeError("checkpoints do not share the motion codebook")
    hybrid = load_trained_k48(CKPT["u783"], device)[0]
    splice_handoff(hybrid, u174)
    delta = float(torch.max(torch.abs(hybrid.logits.weight[HANDOFF] - u174.logits.weight[HANDOFF])))
    motion_delta = float(torch.max(torch.abs(hybrid.logits.weight[:HANDOFF] - u783.logits.weight[:HANDOFF])))
    (OUT / "hybrid_splice_check.json").write_text(json.dumps({
        "handoff_row_max_abs_diff_vs_u174": delta,
        "motion_rows_max_abs_diff_vs_u783": motion_delta,
    }, indent=2), encoding="utf-8")
    env = make_training_env(device, model=u174)
    counter = []
    for row in rows:
        blob = blobs[row["id"]]
        variants = (
            ("u783_original", u783, None, False),
            ("u783_mask_early_handoff", u783, None, True),
            ("u174_handoff_decides_u783_motion", u783, u174, False),
            ("u783_body_u174_logit_row", hybrid, None, False),
        )
        if row["id"] in FOCUS:
            variants = (("u174_reference", u174, None, False),) + variants
        for label, model, handoff_model, mask in variants:
            out, trace, tokens, lost_at, rebuilt_at = roll_detail(
                env, model, book, blob, device, handoff_model=handoff_model, mask_handoff=mask,
            )
            item = {"id": row["id"], "variant": label, **summarize(out, trace, tokens, lost_at, rebuilt_at)}
            if row["id"] in FOCUS and label in {"u783_original", "u174_reference"}:
                item["p47_trace"] = [float(step["p47"]) for step in trace if "p47" in step]
                item["rank47_trace"] = [int(step["rank47"]) for step in trace if "rank47" in step]
                item["token_trace"] = [int(step["token"]) for step in trace if "token" in step]
            counter.append(item)
            print("CF", label, row["id"], item["reason"], item["recovery_steps"], flush=True)
    (OUT / "counterfactuals.json").write_text(json.dumps(counter, indent=2), encoding="utf-8")
    print("COUNTERFACTUAL_DONE", len(regress), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--counterfactual":
        counterfactuals()
    else:
        main()
