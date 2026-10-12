"""Closed-loop diagnosis of the finished K=47 supervised checkpoint.

Does not train and does not update the codebook, labels, or physics.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.autonomous_episode import make_training_env
from training.history_aware_critic_ablation import restore_capture
from training.nominal_pose_audit import errors
from training.privileged_ball_impact_search import replay, step_legal
from training.recovery_success import PERSIST_STEPS, apply_handoff, basin_ok, label_of, nominal_validate
from training.unified_k47 import CAP, CODEBOOK, episode_arrays, load_codebook, read_jsonl

OUT = ROOT / "results/diagnostics/raw/unified_k47/diagnosis"
CKPT = ROOT / "results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt"
EVAL = ROOT / "results/diagnostics/raw/unified_k47/eval/heldout_train.json"
SUPERVISED = ROOT / "results/diagnostics/raw/legal_recovery_v2/supervised_targets.jsonl"
REPLAY = ROOT / "results/diagnostics/raw/legal_recovery_v2/replay_k47_all.jsonl"
GRPO = ROOT / "results/diagnostics/raw/legal_recovery_v2/grpo_initial_conditions.jsonl"


def load_model(book, device):
    from training.formal_grpo import load_models

    model, _ref = load_models(
        CKPT,
        ROOT / "results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt",
        device,
    )
    model.eval()
    if model.logits.out_features != 47:
        raise RuntimeError("checkpoint is not K=47")
    return model


def teacher_forward(model, obs, prev, device):
    with torch.no_grad():
        logits, residual, act = model(
            torch.tensor(obs, device=device),
            torch.tensor(prev, device=device),
        )
    return logits, residual, act


def offline(model, book, rows, device):
    proto = book
    per = []
    pred_hist = np.zeros(47, np.int64)
    tgt_hist = np.zeros(47, np.int64)
    for row in rows:
        obs, prev, actions, tokens, residuals = episode_arrays(row, book)
        logits, residual, act = teacher_forward(model, obs, prev, device)
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
        resid = residual.detach().cpu().numpy().astype(np.float32)
        executed = np.clip(proto[pred] + resid, -1.0, 1.0).astype(np.float32)
        target_exec = np.clip(proto[tokens] + residuals, -1.0, 1.0).astype(np.float32)
        network_act = act.detach().cpu().numpy().astype(np.float32)
        ce = float(F.cross_entropy(logits, torch.tensor(tokens, device=device)))
        mse = float(((residual - torch.tensor(residuals, device=device)) ** 2).mean())
        correct = pred == tokens
        none = (obs[:, 2] <= 0.5) & (obs[:, 3] <= 0.5)
        transition = np.zeros(len(tokens), dtype=bool)
        if len(tokens) > 1:
            transition[1:] = tokens[1:] != tokens[:-1]
        grip_switch = np.zeros(len(tokens), dtype=bool)
        if len(actions) > 1:
            grip_switch[1:] = np.sign(actions[1:, 6]) != np.sign(actions[:-1, 6])
        pred_hist += np.bincount(pred, minlength=47)
        tgt_hist += np.bincount(tokens, minlength=47)
        per.append({
            "id": row["id"],
            "strategy": row["strategy"],
            "mechanism": row["mechanism"],
            "weight": float(row["sample_weight"]),
            "steps": int(len(tokens)),
            "ce": ce,
            "mse": mse,
            "correct": int(correct.sum()),
            "none_steps": int(none.sum()),
            "none_correct": int(correct[none].sum()) if none.any() else 0,
            "transition_steps": int(transition.sum()),
            "transition_correct": int(correct[transition].sum()) if transition.any() else 0,
            "grip_switch_steps": int(grip_switch.sum()),
            "grip_switch_correct": int(correct[grip_switch].sum()) if grip_switch.any() else 0,
            "resid_l2": float(np.linalg.norm(resid - residuals, axis=1).mean()),
            "action_l2": float(np.linalg.norm(executed - actions, axis=1).mean()),
            "action_linf": float(np.max(np.abs(executed - actions), axis=1).mean()),
            "target_decode_linf": float(np.max(np.abs(target_exec - actions))),
            "network_matches_decode": bool(np.max(np.abs(network_act - executed)) < 1e-5),
            "first_token_error": int(np.argmax(~correct)) if not bool(correct.all()) else None,
            "held_out_id": int(row["id"].rsplit("_", 1)[-1]) % 5 == 0,
        })
        print("OFFLINE", row["id"], row["strategy"], round(ce, 4), round(float(correct.mean()), 3), flush=True)
    return per, pred_hist, tgt_hist


def rates(items, key_correct, key_steps):
    steps = sum(item[key_steps] for item in items)
    correct = sum(item[key_correct] for item in items)
    return None if steps == 0 else correct / steps


def summarize_offline(per):
    def pack(items):
        steps = sum(item["steps"] for item in items)
        return {
            "trajectories": len(items),
            "steps": steps,
            "token_top1": rates(items, "correct", "steps"),
            "mean_ce": float(np.mean([item["ce"] for item in items])),
            "mean_mse": float(np.mean([item["mse"] for item in items])),
            "weighted_loss": float(np.mean([item["weight"] * (item["ce"] + item["mse"]) for item in items])),
            "mean_resid_l2": float(np.mean([item["resid_l2"] for item in items])),
            "mean_action_l2": float(np.mean([item["action_l2"] for item in items])),
            "mean_action_linf": float(np.mean([item["action_linf"] for item in items])),
            "max_target_decode_linf": float(max(item["target_decode_linf"] for item in items)),
            "none_contact_top1": rates(items, "none_correct", "none_steps"),
            "transition_top1": rates(items, "transition_correct", "transition_steps"),
            "grip_switch_top1": rates(items, "grip_switch_correct", "grip_switch_steps"),
        }

    by_mech = {}
    for name in sorted({item["mechanism"] for item in per}):
        by_mech[name] = pack([item for item in per if item["mechanism"] == name])
    return {
        "all": pack(per),
        "id_mod5_eq_0": pack([item for item in per if item["held_out_id"]]),
        "id_mod5_ne_0": pack([item for item in per if not item["held_out_id"]]),
        "by_mechanism": by_mech,
    }


def audit_eval(eval_rows, supervised, replay_rows):
    by_id = defaultdict(list)
    for row in supervised:
        by_id[row["id"]].append(row)
    search = defaultdict(list)
    for row in replay_rows:
        search[row["id"]].append(row)
    out = []
    for row in eval_rows:
        demos = by_id.get(row["id"], [])
        holds = search.get(row["id"], [])
        out.append({
            "id": row["id"],
            "autonomous_label": row["label"],
            "truncated": bool(row["truncated"]),
            "in_supervised_training": bool(demos),
            "verified_k47_demos": len(demos),
            "mechanisms": sorted({item["mechanism"] for item in demos}),
            "strategies": sorted({item["strategy"] for item in demos}),
            "excluded_from_codebook_fit": True,
            "continuous_successes": [
                {"strategy": item["strategy"], "family": item["family"], "k47_replay": item["label"], "status": item["status"]}
                for item in holds
            ],
        })
    return out


def autonomous(env, model, book, blob, device, cap=120):
    restore_capture(env, blob["capture"])
    live = np.asarray(env.observation(), np.float32)
    history = {
        "obs_seq": len(env._obs_seq),
        "prev_seq": len(env._prev_seq),
        "prev_linf": float(np.max(np.abs(env._prev))),
        "live_obs": live,
    }
    steps = []
    basin_run = 0
    fell = False
    hold = None
    for _step in range(cap):
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        residual = mean[-1].detach().float().cpu().numpy().astype(np.float32)
        residual = np.clip(residual, -CAP, CAP)
        action = np.clip(book[token] + residual, -1.0, 1.0).astype(np.float32)
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        measured = errors(env.sim, blob["ref"])
        steps.append({
            "token": token,
            "residual": [round(float(x), 4) for x in residual],
            "action": [round(float(x), 4) for x in executed],
            "ep": round(float(measured["ep"]), 5),
            "eR": round(float(measured["eR"]), 5),
            "vrel": round(float(measured["vrel"]), 5),
            "nL": int(measured["nL"]),
            "nR": int(measured["nR"]),
            "z": round(float(measured["z"]), 4),
            "p_rel": [round(float(x), 4) for x in obs[27:30]],
            "v_rel": [round(float(x), 4) for x in obs[30:33]],
        })
        if measured["z"] < 0.30:
            fell = True
            break
        inside = basin_ok(measured, float(executed[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        if basin_run >= PERSIST_STEPS:
            apply_handoff(env.sim, blob["ref"])
            hold = nominal_validate(env, blob["ref"])
            break
    label = label_of(fell, hold is not None, hold is not None, hold)
    return {
        "label": label,
        "truncated": hold is None and not fell,
        "steps": steps,
        "history": {key: value for key, value in history.items() if key != "live_obs"},
        "live_obs": history["live_obs"],
        "handoff_step": None if hold is None else len(steps),
    }


def compare_demo(auto, demo_row, book):
    obs, prev, actions, tokens, residuals = episode_arrays(demo_row, book)
    live = auto["live_obs"]
    obs0 = float(np.max(np.abs(live - obs[0])))
    steps = auto["steps"]
    n = min(len(steps), len(actions))
    token_mismatch = None
    action_divergence = None
    state_gap = None
    for index in range(n):
        if token_mismatch is None and int(steps[index]["token"]) != int(tokens[index]):
            token_mismatch = index
        if action_divergence is None and float(np.max(np.abs(np.asarray(steps[index]["action"]) - actions[index]))) > 0.05:
            action_divergence = index
        # The saved observation at index 0 is the pre-action state. Later saved
        # observations are the states produced by the teacher action.
        if index + 1 < len(obs) and state_gap is None:
            # Filled by the caller when a teacher trace is available.
            pass
    return {
        "strategy": demo_row["strategy"],
        "mechanism": demo_row["mechanism"],
        "teacher_steps": int(len(actions)),
        "obs0_linf": obs0,
        "prev0_is_zero": bool(np.max(np.abs(prev[0])) == 0.0),
        "first_token_mismatch": token_mismatch,
        "first_action_linf_gt_0.05": action_divergence,
        "matched_prefix": n if token_mismatch is None else token_mismatch,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    model = load_model(book, device)
    supervised = read_jsonl(SUPERVISED)
    eval_rows = json.loads(EVAL.read_text(encoding="utf-8"))
    replay_rows = read_jsonl(REPLAY)
    per, pred_hist, tgt_hist = offline(model, book, supervised, device)
    summary = summarize_offline(per)
    summary["predicted_token_counts"] = pred_hist.tolist()
    summary["target_token_counts"] = tgt_hist.tolist()
    summary["decode_consistent"] = all(item["network_matches_decode"] for item in per)
    ics = audit_eval(eval_rows, supervised, replay_rows)
    by_demo = Counter(row["autonomous_label"] for row in ics if row["in_supervised_training"])
    by_none = Counter(row["autonomous_label"] for row in ics if not row["in_supervised_training"])

    env = make_training_env(device, model=model)
    demos_by_id = defaultdict(list)
    for row in supervised:
        demos_by_id[row["id"]].append(row)
    comparisons = []
    replays = []
    traces = {}
    wanted = set()
    for row in ics:
        if not row["in_supervised_training"]:
            continue
        path = ROOT / next(item["snapshot"] for item in demos_by_id[row["id"]])
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        auto = autonomous(env, model, book, blob, device)
        if auto["label"] != row["autonomous_label"]:
            print("LABEL_DIFF", row["id"], row["autonomous_label"], auto["label"], flush=True)
        row["rerun_label"] = auto["label"]
        row["history_len"] = auto["history"]["obs_seq"]
        row["prev_linf_at_restore"] = auto["history"]["prev_linf"]
        demo_cmp = [compare_demo(auto, demo, book) for demo in demos_by_id[row["id"]]]
        row["demo_comparisons"] = demo_cmp
        comparisons.append({"id": row["id"], "auto": auto["label"], "demos": demo_cmp})
        print("AUTO", row["id"], auto["label"], "obs0", round(min(item["obs0_linf"] for item in demo_cmp), 6), flush=True)
        for demo in demos_by_id[row["id"]]:
            actions = np.load(ROOT / demo["continuous_npz"])["actions"].astype(np.float32)
            played = replay(env, blob, actions)
            replays.append({
                "id": demo["id"],
                "strategy": demo["strategy"],
                "stored": demo["k47_replay_label"],
                "replayed": played["existing_label"],
                "match": played["existing_label"] == demo["k47_replay_label"],
            })
            print("REPLAY", demo["id"], demo["strategy"], played["existing_label"], flush=True)
        traces[row["id"]] = {
            "label": auto["label"],
            "handoff_step": auto["handoff_step"],
            "steps": auto["steps"],
        }
        wanted.add(row["id"])

    # One drop without a demonstration, for the failure-mode trace.
    drop_ids = [row["id"] for row in ics if row["autonomous_label"] == "DROP_DURING_RECOVERY"]
    if drop_ids:
        grpo = {row["id"]: row for row in read_jsonl(GRPO)}
        drop_id = drop_ids[0]
        with (ROOT / grpo[drop_id]["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        auto = autonomous(env, model, book, blob, device)
        traces[drop_id] = {"label": auto["label"], "handoff_step": auto["handoff_step"], "steps": auto["steps"], "history": auto["history"]}
        print("AUTO_DROP", drop_id, auto["label"], len(auto["steps"]), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "offline_summary": summary,
        "offline_rows": per,
        "eval_ics": ics,
        "autonomous_with_demo": dict(by_demo),
        "autonomous_without_demo": dict(by_none),
        "replays": replays,
        "comparisons": comparisons,
    }
    (OUT / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    # Full traces are large; keep the four classes separately.
    chosen = {}
    for label in ("SUCCESS_12S", "BASIN_NOT_REACHED", "DROP_AFTER_HANDOFF", "DROP_DURING_RECOVERY"):
        pick = next((row["id"] for row in ics if row["autonomous_label"] == label and row["id"] in traces), None)
        if pick is None:
            pick = next((key for key, value in traces.items() if value["label"] == label), None)
        if pick is not None:
            chosen[label] = {"id": pick, **traces[pick]}
    (OUT / "representative_traces.json").write_text(json.dumps(chosen), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
