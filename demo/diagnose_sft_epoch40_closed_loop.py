"""Closed-loop diagnosis of the first K48 SFT epoch-40 checkpoint.

Does not train, does not edit checkpoints, physics, the nominal controller,
or the dataset. Writes under results/evaluations/.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from envs.airborne_obs import relative_state
from training.autonomous_episode import make_training_env
from training.history_aware_critic_ablation import restore_capture
from training.horizon_outcome import (
    _contacts,
    finish_recovery,
    recovery_failed,
    replay_commands,
)
from training.privileged_ball_impact_search import step_legal
from training.unified_k47 import decode_action
from training.unified_k48 import HANDOFF, load_trained_k48

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results/sft/checkpoint_epoch_0040.pt"
OUT = ROOT / "results/evaluations/raw/sft_epoch40_diagnosis"
LIMIT = 4.0
CAP = 200
W_SCALE = 4.0


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample(env):
    sim = env.sim
    rel = relative_state(sim.model, sim.data, sim.ids)
    hand = np.array(sim.data.xmat[sim.ids.hand_body], float).reshape(3, 3)
    gravity = hand.T @ np.array([0.0, 0.0, -9.81])
    n_l, n_r, z = _contacts(sim)
    return {
        "t": float(sim.data.time),
        "p_hand": np.asarray(rel["p_hand_w"], np.float32),
        "p_rel": np.asarray(rel["p_rel_h"], np.float32),
        "v_rel": np.asarray(rel["v_rel_h"], np.float32),
        "rot_rel": np.asarray(rel["rot_rel"], np.float32),
        "w_hand": np.asarray(rel["w_hand_h"], np.float32),
        "g_h": np.asarray(gravity, np.float32),
        "nL": int(n_l),
        "nR": int(n_r),
        "z": float(z),
    }


def _decide(env, model, device):
    obs = np.asarray(env.observation(), np.float32)
    env._obs_seq.append(obs)
    env._prev_seq.append(env._prev.copy())
    with torch.no_grad():
        logits, mean, _act = model(
            torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
            torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
        )
    last = logits[-1].float()
    probs = torch.softmax(last, dim=-1)
    token = int(torch.argmax(last).item())
    return token, float(probs[HANDOFF].item()), mean[-1].detach().float().cpu().numpy()


def _apply_motion(env, book, token, mean):
    _residual, action = decode_action(book, token, mean)
    executed = np.asarray(step_legal(env.sim, action), np.float32)
    env._prev = executed
    return executed


def roll_traced(env, model, book, blob, device, mask_handoff=False, stop_steps=None):
    """Same control law as roll_policy. mask_handoff keeps the motion-head argmax."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    how = None
    steps = 0
    rows = []
    actions = []
    for _ in range(CAP):
        if float(env.sim.data.time) - t0 >= LIMIT - 1e-9:
            how = "forced"
            break
        if stop_steps is not None and steps >= int(stop_steps):
            how = "scheduled"
            break
        token, p47, mean = _decide(env, model, device)
        state = _sample(env)
        if mask_handoff and token == HANDOFF:
            logits_token = token
            with torch.no_grad():
                logits, mean_t, _act = model(
                    torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                    torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
                )
            motion = logits[-1, :HANDOFF].float()
            token = int(torch.argmax(motion).item())
            mean = mean_t[-1].detach().float().cpu().numpy()
            source = "motion_head_argmax_token47_masked"
            chosen = logits_token
        else:
            source = "policy_argmax"
            chosen = token
        state.update({"token": int(chosen), "executed_token": int(token), "p47": p47, "source": source, "phase": "decision"})
        if not mask_handoff and token == HANDOFF:
            state["phase"] = "handoff"
            rows.append(state)
            how = "early"
            break
        executed = _apply_motion(env, book, token, mean)
        state["action"] = executed
        state["wrist_cmd"] = (executed[3:6] * W_SCALE).astype(np.float32)
        state["grip"] = float(executed[6])
        rows.append(state)
        actions.append(executed)
        steps += 1
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            out = {
                "reason": "DROP_DURING_RECOVERY",
                "success": False,
                "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "recovery_steps": steps,
                "nominal_elapsed": 0.0,
                "lift_completed": False,
            }
            return out, rows, np.asarray(actions, np.float32)
    if how is None:
        how = "forced"
    out = finish_recovery(env, blob["ref"], t0, how)
    out["recovery_steps"] = steps
    return out, rows, np.asarray(actions, np.float32)


def replay_prefix(env, blob, actions, n_steps):
    """Replay a recorded legal prefix, then the unchanged nominal handoff."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    used = np.asarray(actions, np.float32)[: int(n_steps)]
    rows = []
    for action in used:
        if float(env.sim.data.time) - t0 >= LIMIT - 1e-9:
            break
        state = _sample(env)
        executed = np.asarray(step_legal(env.sim, action), np.float32)
        state["action"] = executed
        state["wrist_cmd"] = (executed[3:6] * W_SCALE).astype(np.float32)
        state["grip"] = float(executed[6])
        state["source"] = "recorded_policy_prefix"
        rows.append(state)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "reason": "DROP_DURING_RECOVERY",
                "success": False,
                "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "recovery_steps": len(rows),
                "nominal_elapsed": 0.0,
                "lift_completed": False,
            }, rows
    how = "forced" if float(env.sim.data.time) - t0 >= LIMIT - 1e-9 else "scheduled"
    out = finish_recovery(env, blob["ref"], t0, how)
    out["recovery_steps"] = len(rows)
    return out, rows


def replay_expert(env, blob, actions):
    out = replay_commands(env, blob, actions, limit_s=LIMIT, early_basin=False)
    return out


def _brief(out):
    return {
        "reason": out.get("reason"),
        "success": bool(out.get("success")),
        "handoff": out.get("handoff"),
        "recovery_elapsed": out.get("recovery_elapsed"),
        "recovery_steps": out.get("recovery_steps"),
        "nominal_elapsed": out.get("nominal_elapsed"),
        "lift_completed": bool(out.get("lift_completed")),
    }


def _save_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(path, n=np.array(0))
        return
    pack = {
        "t": np.array([row["t"] for row in rows], np.float64),
        "p_hand": np.stack([row["p_hand"] for row in rows]),
        "p_rel": np.stack([row["p_rel"] for row in rows]),
        "v_rel": np.stack([row["v_rel"] for row in rows]),
        "rot_rel": np.stack([row["rot_rel"] for row in rows]),
        "w_hand": np.stack([row["w_hand"] for row in rows]),
        "g_h": np.stack([row["g_h"] for row in rows]),
        "nL": np.array([row["nL"] for row in rows], np.int16),
        "nR": np.array([row["nR"] for row in rows], np.int16),
        "z": np.array([row["z"] for row in rows], np.float32),
        "token": np.array([row.get("token", -1) for row in rows], np.int16),
        "p47": np.array([row.get("p47", np.nan) for row in rows], np.float32),
    }
    actions = [row.get("action") for row in rows]
    if any(item is not None for item in actions):
        pack["action"] = np.stack([np.zeros(7, np.float32) if item is None else item for item in actions])
        pack["has_action"] = np.array([item is not None for item in actions])
    np.savez_compressed(path, **pack)


def _runs(mask):
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def mechanism(rows):
    motion = [row for row in rows if row.get("action") is not None]
    if len(motion) < 2:
        return {"label": "inconclusive", "reason": "fewer than two executed steps"}
    p_rel = np.stack([row["p_rel"] for row in motion])
    v_rel = np.stack([row["v_rel"] for row in motion])
    rot = np.stack([row["rot_rel"] for row in motion])
    w_hand = np.stack([row["w_hand"] for row in motion])
    g_h = np.stack([row["g_h"] for row in motion])
    p_hand = np.stack([row["p_hand"] for row in motion])
    n_l = np.array([row["nL"] for row in motion])
    n_r = np.array([row["nR"] for row in motion])
    airborne = (n_l + n_r) == 0
    bilateral = (n_l > 0) & (n_r > 0)
    gap = _runs(airborne)
    dx = float(p_rel[-1, 0] - p_rel[0, 0])
    hand_move = float(np.linalg.norm(p_hand[-1] - p_hand[0]))
    wy = w_hand[:, 1]
    mean_wy = float(np.mean(wy))
    mean_gx = float(np.mean(g_h[:, 0]))
    recatch = bool(gap >= 3 and (n_l[-1] + n_r[-1]) > 0 and hand_move > 0.01)
    sliding = bool(float(np.mean(bilateral)) >= 0.7 and abs(dx) >= 0.003 and gap < 3)
    gravity = bool(
        float(np.mean(bilateral)) >= 0.5
        and abs(mean_wy) >= 0.8
        and abs(dx) >= 0.003
        and dx * mean_gx > 0
    )
    labels = []
    if recatch:
        labels.append("recatch")
    if gravity:
        labels.append("gravity-assisted sliding")
    elif sliding:
        labels.append("controlled sliding")
    if len(labels) > 1:
        label = "mixed"
    elif len(labels) == 1:
        label = labels[0]
    else:
        label = "inconclusive"
    return {
        "label": label,
        "components": labels,
        "steps": int(len(motion)),
        "airborne_run_steps": int(gap),
        "airborne_fraction": float(np.mean(airborne)),
        "bilateral_fraction": float(np.mean(bilateral)),
        "delta_rh_x_m": dx,
        "rh_x_start_m": float(p_rel[0, 0]),
        "rh_x_end_m": float(p_rel[-1, 0]),
        "hand_translation_m": hand_move,
        "mean_wy_rad_s": mean_wy,
        "max_abs_wy_rad_s": float(np.max(np.abs(wy))),
        "mean_g_h_x": mean_gx,
        "mean_vrel_m_s": float(np.mean(np.linalg.norm(v_rel, axis=1))),
        "delta_rot_rel_rad": float(np.linalg.norm(rot[-1] - rot[0])),
        "mean_grip": float(np.mean([row["grip"] for row in motion])),
    }


def _summary_stats(records, key):
    groups = {}
    for row in records:
        groups.setdefault(row[key], []).append(row)
    out = {}
    for name, rows in groups.items():
        reasons = {}
        handoffs = {}
        elapsed = []
        for row in rows:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
            label = "none" if row["handoff"] is None else row["handoff"]
            handoffs[label] = handoffs.get(label, 0) + 1
            if row.get("recovery_elapsed") is not None:
                elapsed.append(float(row["recovery_elapsed"]))
        elapsed = np.asarray(elapsed, np.float64)
        out[name] = {
            "n": len(rows),
            "reasons": reasons,
            "handoff": handoffs,
            "recovery_elapsed_s": None if len(elapsed) == 0 else {
                "min": float(elapsed.min()),
                "median": float(np.median(elapsed)),
                "max": float(elapsed.max()),
            },
        }
    return out


def _handoff_distribution(records):
    buckets = {"success": [], "failure": []}
    forced = early = dropped = 0
    early_under_1s = 0
    never_token = 0
    for row in records:
        if row["handoff"] == "forced":
            forced += 1
        elif row["handoff"] == "early":
            early += 1
            if float(row["recovery_elapsed"]) < 1.0:
                early_under_1s += 1
        else:
            dropped += 1
        if row["reason"] != "SUCCESS_12S" and int(row.get("token47_chosen", 0)) == 0:
            never_token += 1
        target = "success" if row["success"] else "failure"
        if row.get("recovery_elapsed") is not None and row["handoff"] is not None:
            buckets[target].append(float(row["recovery_elapsed"]))
    def pack(values):
        if not values:
            return None
        arr = np.asarray(values, np.float64)
        return {"n": int(len(arr)), "min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}
    return {
        "forced": forced,
        "early": early,
        "no_handoff_drop": dropped,
        "early_under_1s": early_under_1s,
        "failures_without_token47": never_token,
        "success_handoff_s": pack(buckets["success"]),
        "failure_handoff_s": pack(buckets["failure"]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model, book = load_trained_k48(CKPT, device)
    env = make_training_env(device, model=model)
    train = _rows(ROOT / "data/grpo_initial_conditions.jsonl")
    valid = _rows(ROOT / "data/valid_recovery.jsonl")
    demos = _rows(ROOT / "data/valid_demonstrations.jsonl")
    policy_rows = []
    traces = {}
    for split, manifest in (("valid", valid), ("train", train)):
        for row in manifest:
            with (ROOT / row["snapshot"]).open("rb") as handle:
                blob = pickle.load(handle)
            out, trace, actions = roll_traced(env, model, book, blob, device)
            record = {"id": row["id"], "split": split, "snapshot": row["snapshot"], **_brief(out)}
            decisions = [item for item in trace if "p47" in item]
            record["p47_max"] = None if not decisions else float(max(item["p47"] for item in decisions))
            record["p47_at_handoff"] = None
            record["token47_chosen"] = int(any(item.get("token") == HANDOFF for item in trace))
            if out.get("handoff") == "early" and decisions:
                record["p47_at_handoff"] = float(decisions[-1]["p47"])
            policy_rows.append(record)
            traces[row["id"]] = (trace, actions, blob)
            _save_trace(OUT / "policy" / f"{row['id']}.npz", trace)
            print("POLICY", split, row["id"], record["reason"], record["handoff"], record["recovery_steps"], flush=True)
    _dump(OUT / "policy_rollouts.json", policy_rows)

    comparisons = []
    for row in valid:
        trace, _actions, blob = traces[row["id"]]
        experts = [demo for demo in demos if demo["snapshot"] == row["snapshot"]]
        policy_mech = mechanism(trace)
        expert_runs = []
        for demo in experts:
            packed = np.load(ROOT / demo["continuous_npz"])
            expert_out = replay_expert(env, blob, packed["actions"])
            expert_runs.append({
                "strategy": demo.get("strategy"),
                "stored_mechanism_label": demo.get("mechanism"),
                "expert_file_says_success": bool(demo.get("success")),
                **_brief(expert_out),
                "expert_steps_in_file": int(len(packed["actions"])),
            })
            print("EXPERT", row["id"], demo.get("strategy"), expert_out.get("reason"), flush=True)
        policy = next(item for item in policy_rows if item["id"] == row["id"])
        comparisons.append({
            "id": row["id"],
            "snapshot": row["snapshot"],
            "policy": policy,
            "policy_mechanism_from_trace": policy_mech,
            "experts_on_same_snapshot": expert_runs,
        })
    _dump(OUT / "valid_expert_comparison.json", comparisons)

    drop_after = [row["id"] for row in policy_rows if row["split"] == "valid" and row["reason"] == "DROP_AFTER_HANDOFF"]
    branches = []
    for episode_id in drop_after:
        trace, actions, blob = traces[episode_id]
        actual = next(row for row in policy_rows if row["id"] == episode_id)
        n_actual = int(len(actions))
        early_points = sorted(set([step for step in range(5, n_actual + 1, 5)] + [n_actual]))
        for steps in early_points:
            out, _rows_trace = replay_prefix(env, blob, actions, steps)
            branches.append({
                "id": episode_id,
                "kind": "recorded_prefix",
                "steps": steps,
                "action_source": "prefix of the actions the SFT policy actually executed before its own handoff",
                "versus_policy": "earlier" if steps < n_actual else "same_time",
                **_brief(out),
            })
            print("BRANCH", episode_id, "prefix", steps, out.get("reason"), flush=True)
        if actual["handoff"] == "early" and n_actual < CAP:
            later = list(range(n_actual + 5, CAP + 1, 10))
            for steps in later:
                out, later_trace, _later_actions = roll_traced(
                    env, model, book, blob, device, mask_handoff=True, stop_steps=steps,
                )
                branches.append({
                    "id": episode_id,
                    "kind": "delayed_motion_head",
                    "steps": int(out.get("recovery_steps") or 0),
                    "requested_steps": steps,
                    "action_source": "same checkpoint, deterministic argmax over motion tokens 0-46; token 47 is masked and is not executed",
                    "versus_policy": "later",
                    "masked_decisions": int(sum(1 for item in later_trace if item.get("source") == "motion_head_argmax_token47_masked")),
                    **_brief(out),
                })
                print("BRANCH", episode_id, "delay", steps, out.get("reason"), out.get("recovery_steps"), flush=True)
                if out.get("reason") == "DROP_DURING_RECOVERY":
                    break
    _dump(OUT / "handoff_branches.json", branches)

    focus = []
    for episode_id in ("k48r_valid_00004", "inward_extra_250043"):
        trace, _actions, _blob = traces[episode_id]
        focus.append({"id": episode_id, "mechanism": mechanism(trace), "policy": next(row for row in policy_rows if row["id"] == episode_id)})
    _dump(OUT / "success_mechanisms.json", focus)
    _dump(OUT / "split_summary.json", _summary_stats(policy_rows, "split"))
    handoff = {
        split: _handoff_distribution([row for row in policy_rows if row["split"] == split])
        for split in ("train", "valid")
    }
    _dump(OUT / "handoff_stats.json", handoff)
    print("DIAGNOSIS_DONE", len(policy_rows), len(branches), flush=True)


if __name__ == "__main__":
    main()
