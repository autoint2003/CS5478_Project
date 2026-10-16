"""Phase 1-3 evidence for the Horizontal Recatch root-cause audit.

Does not modify physics, controllers, the codebook, reward, or checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from envs.simulation.capture import restore_snapshot, take_snapshot
from envs.outcomes.horizon_outcome import _fail_recovery, recovery_failed
from envs.outcomes.recovery_success import nominal_step
from envs.outcomes.table_drop import export_state
from envs.step_legal import step_legal
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.grpo_math import SIGMA, gaussian_logprob
from training.k47_codebook import CAP, decode_action, load_codebook
from training.k48_config import ROOT
from experiments.recatch.diagnose_horizontal_recatch import (
    CKPT,
    begin,
    load_expert,
    manifest_by_id,
    play,
    policy_distribution,
    prepare_horizontal,
    sample_action,
    snapshot_state,
)

OUT = ROOT / "results/diagnostics/horizontal_recatch_root_cause/raw"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"


def _json(value):
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.round(np.asarray(value, np.float64), 6).tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(_json(payload), indent=2), encoding="utf-8")
    print("WROTE", path.name, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slim(state):
    return {
        "time": state["time"],
        "bottom": state["bottom"],
        "gap_to_table": state["gap_to_table"],
        "obj_z": state["obj_z"],
        "v_obj": state["v_obj"],
        "tilt_deg": state["tilt_deg"],
        "nL": state["nL"],
        "nR": state["nR"],
        "finger_gap": state["finger_gap"],
        "latched": state["latched"],
        "latch_kind": state["latch_kind"],
        "fg_cmd": state["fg_cmd"],
        "phase": state["phase"],
        "p_des": state["p_des"],
        "qpos_head": np.asarray(state["qpos"])[:8],
    }


def one_step(env, blob, action):
    before = snapshot_state(env)
    done = step_legal(env.sim, action)
    after = snapshot_state(env)
    failed, _lost = recovery_failed(env.sim, blob["ref"], 0)
    reason = None
    if failed:
        reason = _fail_recovery(env.sim, blob["ref"], before["time"])["reason"]
    return {
        "before": slim(before),
        "after": slim(after),
        "executed": np.asarray(done, np.float32),
        "failed": bool(failed),
        "reason": reason,
        "latch": export_state(env.sim),
    }


def main():
    book = load_codebook()
    device = torch.device("cpu")
    model, _saved_book = load_trained_k48(CKPT, device)
    model.eval()
    weight = hashlib.sha256()
    for key, value in model.state_dict().items():
        weight.update(key.encode())
        weight.update(value.detach().cpu().numpy().tobytes())
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    adaptive = ROOT / "results/grpo/adaptive_hybrid/checkpoint_adaptive.pt"
    write("baseline_identity.json", {
        "checkpoint": "results/sft/current/checkpoint_epoch_0040.pt",
        "checkpoint_sha256": sha256(CKPT),
        "epoch": int(saved["epoch"]),
        "logits": list(saved["model"]["logits.weight"].shape),
        "parameter_sha256": weight.hexdigest(),
        "codebook": str(CODEBOOK.relative_to(ROOT)).replace("\\", "/"),
        "codebook_sha256": sha256(CODEBOOK),
        "codebook_shape": list(book.shape),
        "adaptive_checkpoint_exists": adaptive.is_file(),
        "model_mode_in_this_audit": "eval",
        "model_mode_in_adaptive_grpo": "train",
        "dropout": 0.0,
        "sigma": SIGMA,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "residual_cap": CAP,
        "policy_dt_s": 0.02,
        "mujoco_dt_s": 0.002,
        "recovery_limit_s": 4.0,
        "nominal_s": 12.0,
        "privileged_step_legal": "calls envs.step_legal.step_legal",
    })

    from data_generation.strategies.recatch_oracle import _setup

    _sim, _base, seated_ref, _target, _code = _setup()
    horizontal = prepare_horizontal(seated_ref)
    tapes = manifest_by_id()
    inventory = []
    for row, blob in horizontal:
        meta = tapes[row["id"]]
        inventory.append({
            "id": row["id"],
            "split": "train",
            "snapshot": row["snapshot"],
            "snapshot_sha256": sha256(ROOT / row["snapshot"]),
            "had_ref_in_pickle": blob["_had_ref"],
            "grasp_scene": meta.get("grasp_scene"),
            "strategy": meta.get("strategy"),
            "formal": meta.get("formal"),
            "task_success": meta.get("task_success"),
            "observed_mechanism": meta.get("observed_mechanism"),
            "continuous_npz": meta.get("continuous_npz"),
            "token_residual_npz": meta.get("token_residual_npz"),
            "speed": meta.get("speed"),
            "pitch": meta.get("pitch"),
            "axial": meta.get("axial"),
            "ecc": meta.get("ecc"),
        })
    valid_rows = []
    for line in (ROOT / "data/valid_skill_quotas.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if "horizontal_recatch" in item.get("categories", []):
            valid_rows.append(item)
    for item in valid_rows:
        meta = tapes[item["id"]]
        inventory.append({
            "id": item["id"],
            "split": "valid",
            "snapshot": item["snapshot"],
            "snapshot_sha256": sha256(ROOT / item["snapshot"]),
            "grasp_scene": meta.get("grasp_scene"),
            "strategy": meta.get("strategy"),
            "formal": meta.get("formal"),
            "task_success": meta.get("task_success"),
            "observed_mechanism": meta.get("observed_mechanism"),
            "continuous_npz": meta.get("continuous_npz"),
            "token_residual_npz": meta.get("token_residual_npz"),
            "speed": meta.get("speed"),
            "pitch": meta.get("pitch"),
            "axial": meta.get("axial"),
            "ecc": meta.get("ecc"),
        })
    snaps = [row["snapshot"] for row in inventory]
    write("inventory.json", {
        "train": sum(row["split"] == "train" for row in inventory),
        "valid": sum(row["split"] == "valid" for row in inventory),
        "unique_snapshots": len(set(snaps)),
        "duplicate_snapshots": len(snaps) - len(set(snaps)),
        "rows": inventory,
    })

    from training.recovery_episode import make_training_env

    env = make_training_env(device, model=model)
    previous = json.loads((ROOT / "results/diagnostics/horizontal_recatch_sampling/expert_replay.json").read_text(encoding="utf-8"))
    write("expert_replay_train.json", previous)

    valid_replay = []
    valid_blobs = []
    for item in valid_rows:
        meta = tapes[item["id"]]
        with (ROOT / item["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if "ref" not in blob:
            blob = dict(blob)
            blob["ref"] = seated_ref
        _actions, tokens, residuals, decoded = load_expert({"id": item["id"]}, book)[1:]
        result = play(env, blob, decoded, observe=True)
        valid_replay.append({
            "id": item["id"],
            "strategy": meta["strategy"],
            "reason": result["reason"],
            "success": result["success"],
            "steps": result["steps"],
            "tape_steps": int(len(decoded)),
            "decoded_vs_stored_linf": float(np.max(np.abs(decoded - _actions))),
            "start_latched": result["rows"][0]["latched"],
            "start_bottom": result["rows"][0]["bottom"],
            "start_gap": result["rows"][0]["gap_to_table"],
            "start_nL": result["rows"][0]["nL"],
            "start_nR": result["rows"][0]["nR"],
            "start_vz": float(result["rows"][0]["v_obj"][2]),
        })
        valid_blobs.append((item, blob, tokens, residuals, decoded))
        print("VALID", item["id"], result["reason"], result["steps"], flush=True)
    write("expert_replay_valid.json", valid_replay)

    # Step alignment of two GRPO-path restores, plus restore-at-k, on three train ICs.
    chosen_ids = ["ref_s6_p10_a0_e0", "s6_p15_a+0.012_e0.008", "s7_p8_a+0.012_e0.008"]
    by_id = {row["id"]: (row, blob) for row, blob in horizontal}
    equivalence = []
    first_steps = []
    for episode_id in chosen_ids:
        row, blob = by_id[episode_id]
        _meta, actions, tokens, residuals, decoded = load_expert(row, book)
        first = play(env, blob, decoded, observe=True)
        second = play(env, blob, decoded, observe=True)
        state_mismatch = []
        for index, (left, right) in enumerate(zip(first["rows"], second["rows"])):
            qpos = float(np.max(np.abs(left["qpos"] - right["qpos"])))
            qvel = float(np.max(np.abs(left["qvel"] - right["qvel"])))
            if qpos > 1e-6 or qvel > 1e-6 or left["latched"] != right["latched"] or left["nL"] != right["nL"]:
                state_mismatch.append({"step": index, "qpos": qpos, "qvel": qvel})
        cuts = [k for k in (1, 2, 3, 5) if k < len(decoded)]
        begin(env, blob)
        snaps = {}
        uninterrupted = [snapshot_state(env)]
        lost_low = 0
        stopped = None
        for index, action in enumerate(decoded):
            if index in cuts:
                snaps[index] = take_snapshot(env)
            step_legal(env.sim, action)
            uninterrupted.append(snapshot_state(env))
            failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
            if failed:
                stopped = index + 1
                break
        branches = []
        for k, snap in snaps.items():
            restore_snapshot(env, snap)
            restored = [snapshot_state(env)]
            lost = 0
            branch_stop = None
            for action in decoded[k:]:
                step_legal(env.sim, action)
                restored.append(snapshot_state(env))
                failed, lost = recovery_failed(env.sim, blob["ref"], lost)
                if failed:
                    branch_stop = k + len(restored) - 1
                    break
            paired = []
            for offset, state in enumerate(restored):
                ref = uninterrupted[k + offset]
                qpos = float(np.max(np.abs(state["qpos"] - ref["qpos"])))
                qvel = float(np.max(np.abs(state["qvel"] - ref["qvel"])))
                obs_gap = None
                if qpos > 1e-5 or qvel > 1e-5 or state["latched"] != ref["latched"]:
                    paired.append({"offset": offset, "qpos": qpos, "qvel": qvel, "latch": state["latched"] != ref["latched"]})
            branches.append({
                "k": k,
                "compared_steps": len(restored),
                "mismatches": paired[:6],
                "mismatch_count": len(paired),
                "branch_stopped": branch_stop,
                "uninterrupted_stopped": stopped,
            })
            print("RESTORE", episode_id, "k", k, "mismatch", len(paired), flush=True)
        equivalence.append({
            "id": episode_id,
            "repeat_reason": [first["reason"], second["reason"]],
            "repeat_state_mismatches": state_mismatch[:6],
            "branches": branches,
        })

        begin(env, blob)
        zero = one_step(env, blob, np.zeros(7, np.float32))
        begin(env, blob)
        t0 = float(env.sim.data.time)
        for _ in range(10):
            nominal_step(env.sim)
        nominal_after = snapshot_state(env)
        nominal_failed, _lost = recovery_failed(env.sim, blob["ref"], 0)
        _residual, expert_action = decode_action(book, int(tokens[0]), residuals[0])
        begin(env, blob)
        expert = one_step(env, blob, expert_action)
        begin(env, blob)
        _obs, probs, mean, entropy = policy_distribution(model, env, device)
        top = int(np.argmax(probs))
        deterministic = {"token": top, "probability": float(probs[top]), "entropy": entropy, "handoff_probability": float(probs[HANDOFF])}
        if top == HANDOFF:
            deterministic["failed"] = False
            deterministic["note"] = "argmax is HANDOFF; no recovery step was taken"
        else:
            begin(env, blob)
            policy_distribution(model, env, device)
            _residual, action = decode_action(book, top, mean)
            deterministic.update(one_step(env, blob, action))
            deterministic["residual_versus_expert_linf"] = float(np.max(np.abs(mean - residuals[0])))
        stochastic = []
        for seed in range(8):
            begin(env, blob)
            token, pre, action, sample_probs, sample_entropy = sample_action(
                model, env, np.random.default_rng(9000 + seed), device, book,
            )
            if token == HANDOFF:
                stochastic.append({"seed": seed, "token": token, "probability": float(sample_probs[token]), "failed": False, "note": "HANDOFF"})
                continue
            # sample_action already consumed the observation append on a fresh begin.
            result = one_step(env, blob, action)
            # one_step does not re-begin; the sim is already after sample_action's observation, before the step.
            # sample_action only reads observation. one_step then steps. Good.
            stochastic.append({
                "seed": seed,
                "token": int(token),
                "probability": float(sample_probs[token]),
                "expert_token": int(tokens[0]),
                "token_matches_expert": int(token) == int(tokens[0]),
                "residual_versus_expert_linf": float(np.max(np.abs(pre - residuals[0]))),
                "failed": result["failed"],
                "reason": result["reason"],
                "after_bottom": result["after"]["bottom"],
                "after_gap": result["after"]["gap_to_table"],
                "latch_kind": result["latch"]["kind"],
                "nL": result["after"]["nL"],
                "nR": result["after"]["nR"],
            })
        order = np.argsort(-probs)
        first_steps.append({
            "id": episode_id,
            "expert_token": int(tokens[0]),
            "expert_probability": float(probs[int(tokens[0])]),
            "expert_rank": int(np.where(order == int(tokens[0]))[0][0]) + 1,
            "zero_action": {"failed": zero["failed"], "reason": zero["reason"], "before": zero["before"], "after": zero["after"], "executed": zero["executed"]},
            "nominal_20ms": {
                "failed": bool(nominal_failed),
                "reason": None if not nominal_failed else _fail_recovery(env.sim, blob["ref"], t0)["reason"],
                "after": slim(nominal_after),
            },
            "expert_first": {"failed": expert["failed"], "reason": expert["reason"], "before": expert["before"], "after": expert["after"], "executed": expert["executed"]},
            "deterministic": deterministic,
            "stochastic": stochastic,
        })
        print("FIRST", episode_id, "expert_fail", expert["failed"], "zero_fail", zero["failed"], "stoch_fail", sum(int(item.get("failed")) for item in stochastic), flush=True)
    write("execution_equivalence.json", equivalence)
    write("first_step.json", first_steps)

    # Same-seed independent token replay, two repeats, three ICs. Stops at recovery_failed.
    from experiments.recatch.diagnose_horizontal_recatch import rollout_policy

    repeats = []
    for episode_id in chosen_ids:
        _row, blob = by_id[episode_id]
        pair = []
        for _repeat in range(2):
            result = rollout_policy(env, blob, model, book, np.random.default_rng(4242), device)
            pair.append(result)
        repeats.append({"id": episode_id, "first": pair[0], "second": pair[1], "same_reason": pair[0]["reason"] == pair[1]["reason"], "same_first_token": pair[0]["first_token"] == pair[1]["first_token"], "same_elapsed": abs(pair[0]["recovery_elapsed"] - pair[1]["recovery_elapsed"]) < 1e-6})
        print("REPEAT", episode_id, pair[0]["reason"], pair[0]["recovery_elapsed"], pair[1]["reason"], flush=True)
    write("independent_repeat.json", repeats)

    probs = json.loads((ROOT / "results/diagnostics/horizontal_recatch_sampling/probabilities.json").read_text(encoding="utf-8"))
    summary = []
    for episode in probs:
        steps = episode["steps"]
        if not steps:
            continue
        early = steps[:10]
        summary.append({
            "id": episode["id"],
            "category": episode["category"],
            "strategy": episode.get("strategy"),
            "n_steps": len(steps),
            "step0_token": steps[0]["expert_token"],
            "step0_probability": steps[0]["probability"],
            "step0_rank": steps[0]["rank"],
            "step0_entropy": steps[0]["entropy"],
            "step0_handoff": steps[0]["handoff_probability"],
            "step0_top_token": steps[0]["top_token"],
            "step0_top_probability": steps[0]["top_probability"],
            "step0_residual_l2": steps[0]["residual_l2"],
            "min_probability": min(step["probability"] for step in steps),
            "median_probability": float(np.median([step["probability"] for step in steps])),
            "mean_log_prob": float(np.mean([step["log_prob"] for step in steps])),
            "sum_log_prob": float(np.sum([step["log_prob"] for step in steps])),
            "early10_min_probability": min(step["probability"] for step in early),
            "early10_mean_rank": float(np.mean([step["rank"] for step in early])),
            "max_handoff": max(step["handoff_probability"] for step in steps),
            "mean_residual_l2": float(np.mean([step["residual_l2"] for step in steps])),
        })
    write("probability_summary.json", summary)

    log_path = ROOT / "results/grpo/adaptive_hybrid/ic_log.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    updates_before = 0
    horizontal_log = []
    for row in rows:
        if row.get("primary_category") == "horizontal_recatch":
            horizontal_log.append({
                "id": row["id"],
                "updates_before": updates_before,
                "stage_a_success": row["stage_a_success"],
                "stage_b_success": row["stage_b_success"],
                "tree_uniform_success": row["tree_uniform_success"],
                "tree_entropy_success": row["tree_entropy_success"],
                "updated": row["updated"],
                "outcomes": row["outcomes"],
                "rewards": row["rewards"],
                "tree_uniform_branch_t": [item.get("branch_t") for item in row.get("tree_uniform") or []],
                "tree_entropy_branch_t": [item.get("branch_t") for item in row.get("tree_entropy") or []],
                "tree_uniform_labels": [item.get("label") for item in row.get("tree_uniform") or []],
                "tree_entropy_labels": [item.get("label") for item in row.get("tree_entropy") or []],
                "tree_uniform_suffix_len": [len(item.get("suffix_tokens") or []) for item in row.get("tree_uniform") or []],
                "tree_entropy_suffix_len": [len(item.get("suffix_tokens") or []) for item in row.get("tree_entropy") or []],
            })
        if row.get("updated"):
            updates_before += 1
    write("adaptive_log_horizontal.json", {"ics_in_log": len(rows), "horizontal": horizontal_log, "updates_so_far": updates_before})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
