"""Migrate recovery labels onto an independent nominal horizon. Does not train.

Historical manifests under legal_recovery_v2 are read and not rewritten.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.autonomous_episode import make_training_env
from training.formal_grpo import load_models
from training.horizon_outcome import (
    LAMBDA_TIME,
    NOMINAL_S,
    T_RECOVERY_MAX,
    physical_reward,
    replay_commands,
)
from training.pre_grpo_sanity import advantages
from training.unified_k47 import CAP, episode_arrays, group_advantage, load_codebook, read_jsonl, supervised_loss
import torch

RAW = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2"
OUT = ROOT / "demo_results/diagnostics/raw/horizon_migration"
OLD_EVAL = ROOT / "demo_results/diagnostics/raw/unified_k47/eval/heldout_train.json"
CKPT = ROOT / "demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt"


def mechanism(row):
    multi = int(row.get("none_intervals") or 0) >= 2
    if row.get("family") == "sliding" and not multi:
        return "contact_preserving_slide"
    if row.get("family") == "sliding":
        return "sliding"
    if multi:
        return "repeated_recatch"
    if row.get("strategy") == "chase_lost":
        return "airborne_chase"
    return "release_regrasp"


def load_done(path):
    done = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["id"], row["strategy"], row["source"])] = row
    return done


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def meta_by_key():
    rows = read_jsonl(RAW / "train_manifest.jsonl")
    found = {}
    for row in rows:
        for traj in row.get("trajectories") or []:
            found[(row["id"], traj["strategy"])] = {
                "snapshot": row["snapshot"],
                "family": traj.get("family"),
                "none_intervals": traj.get("none_intervals"),
                "continuous_npz": traj.get("continuous_npz"),
                "token_residual_npz": traj.get("candidate_k47_tokens"),
            }
    return found


def decoded_actions(token_path, book):
    packed = np.load(token_path)
    tokens = packed["tokens"].astype(np.int64)
    residuals = packed["residuals"].astype(np.float32)
    if int(tokens.min()) < 0 or int(tokens.max()) >= len(book):
        return None, "token outside K=47"
    if float(np.max(np.abs(residuals))) > CAP + 1e-5:
        return None, "residual outside ±0.25"
    residual = np.clip(residuals, -CAP, CAP)
    return np.clip(book[tokens] + residual, -1.0, 1.0).astype(np.float32), None


def migrate(env, book):
    replay_rows = read_jsonl(RAW / "replay_k47_all.jsonl")
    meta = meta_by_key()
    continuous_path = OUT / "continuous.jsonl"
    decoded_path = OUT / "decoded.jsonl"
    done_c = load_done(continuous_path)
    done_d = load_done(decoded_path)
    cache = {}
    for row in replay_rows:
        key = (row["id"], row["strategy"])
        info = meta.get(key, {})
        snapshot = info.get("snapshot") or f"demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train/{row['id']}.pkl"
        path = ROOT / snapshot
        if not path.is_file():
            record = {**row, "source": "continuous", "reason": "INVALID", "error": f"missing snapshot {snapshot}"}
            if (row["id"], row["strategy"], "continuous") not in done_c:
                append(continuous_path, record)
            continue
        if path not in cache:
            with path.open("rb") as handle:
                cache[path] = pickle.load(handle)
        blob = cache[path]
        npz = ROOT / (info.get("continuous_npz") or f"demo_results/diagnostics/raw/legal_recovery_v2/trajectories/{row['id']}__{row['strategy']}.npz")
        if (row["id"], row["strategy"], "continuous") not in done_c:
            if not npz.is_file():
                append(continuous_path, {**row, "source": "continuous", "reason": "INVALID", "error": "missing action file"})
            else:
                actions = np.load(npz)["actions"].astype(np.float32)
                out = replay_commands(env, blob, actions, limit_s=None, early_basin=False)
                record = {
                    "id": row["id"], "strategy": row["strategy"], "family": row.get("family") or info.get("family"),
                    "source": "continuous", "historical_k47_label": row.get("label"), "historical_status": row.get("status"),
                    "none_intervals": info.get("none_intervals"), "steps": int(len(actions)),
                    "reason": out["reason"], "success": bool(out.get("success")), "handoff": out.get("handoff"),
                    "recovery_elapsed": out.get("recovery_elapsed"), "nominal_elapsed": out.get("nominal_elapsed"),
                    "lift_completed": out.get("lift_completed"), "t_handoff": out.get("t_handoff"), "t_end": out.get("t_end"),
                    "reward": out.get("reward"),
                }
                append(continuous_path, record)
                done_c[(row["id"], row["strategy"], "continuous")] = record
                print("CONT", row["id"], row["strategy"], out["reason"], flush=True)
        token_path = ROOT / (info.get("token_residual_npz") or f"demo_results/diagnostics/raw/legal_recovery_v2/trajectories/{row['id']}__{row['strategy']}__candidate_k47.npz")
        if (row["id"], row["strategy"], "decoded") in done_d:
            continue
        if row.get("status") == "NOT_REPRESENTABLE":
            record = {
                "id": row["id"], "strategy": row["strategy"], "family": row.get("family") or info.get("family"),
                "source": "decoded", "reason": "UNRESOLVED", "success": False,
                "error": "nearest K=47 prototype leaves at least one command outside ±0.25; clipped tokens are not the saved legal actions",
                "historical_k47_label": row.get("label"), "historical_status": row.get("status"),
                "none_intervals": info.get("none_intervals"),
            }
            append(decoded_path, record)
            done_d[(row["id"], row["strategy"], "decoded")] = record
            print("DEC", row["id"], row["strategy"], "UNRESOLVED", flush=True)
            continue
        if not token_path.is_file():
            append(decoded_path, {"id": row["id"], "strategy": row["strategy"], "source": "decoded", "reason": "INVALID", "error": "missing token file"})
            continue
        actions, error = decoded_actions(token_path, book)
        if actions is None:
            append(decoded_path, {
                "id": row["id"], "strategy": row["strategy"], "family": row.get("family") or info.get("family"),
                "source": "decoded", "reason": "UNRESOLVED", "error": error,
                "historical_k47_label": row.get("label"), "none_intervals": info.get("none_intervals"),
            })
            print("DEC", row["id"], row["strategy"], error, flush=True)
            continue
        out = replay_commands(env, blob, actions, limit_s=None, early_basin=False)
        append(decoded_path, {
            "id": row["id"], "strategy": row["strategy"], "family": row.get("family") or info.get("family"),
            "source": "decoded", "historical_k47_label": row.get("label"), "historical_status": row.get("status"),
            "none_intervals": info.get("none_intervals"), "steps": int(len(actions)),
            "reason": out["reason"], "success": bool(out.get("success")), "handoff": out.get("handoff"),
            "recovery_elapsed": out.get("recovery_elapsed"), "nominal_elapsed": out.get("nominal_elapsed"),
            "lift_completed": out.get("lift_completed"), "reward": out.get("reward"),
            "snapshot": snapshot,
            "continuous_npz": str(npz.relative_to(ROOT)).replace("\\", "/"),
            "token_residual_npz": str(token_path.relative_to(ROOT)).replace("\\", "/"),
        })
        print("DEC", row["id"], row["strategy"], out["reason"], flush=True)
        done_d[(row["id"], row["strategy"], "decoded")] = out


def write_views():
    decoded = [json.loads(line) for line in (OUT / "decoded.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    search = read_jsonl(RAW / "search.jsonl")
    missing = [row for row in search if not (RAW / "trajectories" / f"{row['id']}__{row['strategy']}.npz").is_file()]
    (OUT / "unresolved_searches.json").write_text(json.dumps({
        "count": len(missing),
        "reason": "search.jsonl stores labels only; the recovery action array was not saved",
        "by_historical_label": dict(Counter(row.get("label") for row in missing)),
    }, indent=2), encoding="utf-8")
    for row in decoded:
        if row.get("historical_status") == "NOT_REPRESENTABLE" and row.get("reason") != "UNRESOLVED":
            row["clipped_replay_reason"] = row.get("reason")
            row["reason"] = "UNRESOLVED"
            row["success"] = False
            row["error"] = "nearest K=47 prototype leaves at least one command outside ±0.25; clipped tokens are not the saved legal actions"
    text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in decoded)
    (OUT / "decoded.jsonl").write_text(text, encoding="utf-8")
    continuous = [json.loads(line) for line in (OUT / "continuous.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    success = [row for row in decoded if row.get("success") and row.get("reason") == "SUCCESS_12S"]
    counts = Counter(mechanism(row) for row in success)
    supervised = []
    for row in success:
        name = mechanism(row)
        supervised.append({
            "id": row["id"],
            "snapshot": row["snapshot"],
            "strategy": row["strategy"],
            "family": row.get("family"),
            "mechanism": name,
            "continuous_npz": row["continuous_npz"],
            "token_residual_npz": row["token_residual_npz"],
            "none_intervals": row.get("none_intervals"),
            "steps_are_one_episode": True,
            "sample_weight": 1.0 / counts[name],
            "historical_k47_label": row.get("historical_k47_label"),
            "horizon_reason": row["reason"],
            "handoff": row.get("handoff"),
            "recovery_elapsed": row.get("recovery_elapsed"),
            "supervisory_target": True,
        })
    with (OUT / "supervised_targets.jsonl").open("w", encoding="utf-8") as handle:
        for row in supervised:
            handle.write(json.dumps(row) + "\n")
    grpo = read_jsonl(RAW / "grpo_initial_conditions.jsonl")
    with (OUT / "grpo_initial_conditions.jsonl").open("w", encoding="utf-8") as handle:
        for row in grpo:
            row = dict(row)
            row["horizon"] = "recovery_limit_then_nominal_12s"
            handle.write(json.dumps(row) + "\n")
    elapsed = [float(row["recovery_elapsed"]) for row in success if row.get("recovery_elapsed") is not None]
    steps = [int(row["steps"]) for row in success if row.get("steps") is not None]
    cross = Counter((row.get("historical_k47_label"), row.get("reason")) for row in decoded)
    summary = {
        "decoded_rows": len(decoded),
        "decoded_success": len(success),
        "decoded_ics": len({row["id"] for row in success}),
        "by_mechanism": dict(counts),
        "mechanism_weight_sums": {name: round(counts[name] * (1.0 / counts[name]), 6) for name in counts},
        "by_reason": dict(Counter(row.get("reason") for row in decoded)),
        "historical_to_new": {f"{old}->{new}": count for (old, new), count in sorted(cross.items(), key=lambda item: str(item[0]))},
        "handoff": dict(Counter(row.get("handoff") for row in success)),
        "multi_stage": sum(1 for row in success if int(row.get("none_intervals") or 0) >= 2),
        "recovery_elapsed_s": {
            "min": min(elapsed) if elapsed else None,
            "median": float(np.median(elapsed)) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
        },
        "recovery_steps": {"min": min(steps) if steps else None, "max": max(steps) if steps else None},
        "continuous_rows": len(continuous),
        "continuous_by_reason": dict(Counter(row.get("reason") for row in continuous)),
        "unresolved_searches": len(missing),
        "grpo_ics": len(grpo),
        "lambda_time": LAMBDA_TIME,
        "nominal_seconds": NOMINAL_S,
        "recovery_limit_s": T_RECOVERY_MAX,
    }
    (OUT / "view_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("VIEWS", json.dumps(summary), flush=True)


def timing(env, book, model):
    book = load_codebook()
    path = ROOT / "demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train/ball_train_v2_0021.pkl"
    with path.open("rb") as handle:
        blob = pickle.load(handle)
    from training.history_aware_critic_ablation import restore_capture
    restore_capture(env, blob["capture"])
    err = float(np.max(np.abs(np.array(env.sim.data.qpos) - blob["capture"]["physics"]["qpos"])))
    actions = np.load(RAW / "trajectories/ball_train_v2_0021__release_long.npz")["actions"]
    if len(actions) != 46:
        raise RuntimeError(f"0021 length {len(actions)}")
    full = replay_commands(env, blob, actions, limit_s=None, early_basin=False)
    # Forced handoff after 10 commands, whatever the simulator clock is.
    from training.privileged_ball_impact_search import step_legal
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    for action in actions[:10]:
        step_legal(env.sim, action)
    from training.horizon_outcome import finish_recovery
    forced = finish_recovery(env, blob["ref"], t0, "forced")
    nominal = float(full["nominal_elapsed"])
    if abs(nominal - (float(full["t_end"]) - float(full["t_handoff"]))) > 1e-6:
        raise RuntimeError("nominal elapsed is not measured from the handoff")
    if full["reason"] == "SUCCESS_12S" and not (NOMINAL_S - 1e-3 <= nominal <= NOMINAL_S + 0.01):
        raise RuntimeError(f"0021 nominal elapsed {nominal}")
    forced_elapsed = float(forced["t_end"]) - float(forced["t_handoff"])
    if abs(forced_elapsed - float(forced["nominal_elapsed"])) > 1e-6:
        raise RuntimeError("forced handoff clock is not handoff-relative")
    quick_fail = physical_reward("DROP_DURING_RECOVERY", 0.2)
    slow_ok = physical_reward("SUCCESS_12S", T_RECOVERY_MAX)
    fast_ok = physical_reward("SUCCESS_12S", 0.4)
    if not (fast_ok > slow_ok > 0.0 > quick_fail):
        raise RuntimeError(f"reward order {fast_ok} {slow_ok} {quick_fail}")
    values = [fast_ok, slow_ok, quick_fail, quick_fail]
    adv, _std, skipped = advantages(values)
    same, _std2, skipped_same = advantages([slow_ok, slow_ok, slow_ok])
    grouped, skipped_trunc = group_advantage([slow_ok, quick_fail, 0.0], [False, False, True])
    if skipped or skipped_trunc or not skipped_same:
        raise RuntimeError("advantage flags")
    if abs(float(grouped[2])) > 0.0:
        raise RuntimeError("truncated return entered the advantage")
    # One supervised loss on the migrated row if it exists, else the historical row, with no optimizer step.
    sup_path = OUT / "supervised_targets.jsonl"
    source = sup_path if sup_path.is_file() else RAW / "supervised_targets.jsonl"
    rows = read_jsonl(source)
    case = next(row for row in rows if row["id"] == "ball_train_v2_0021" and row["strategy"] == "release_long")
    device = torch.device("cpu")
    from training.unified_k47 import collate
    obs, prev, tokens, residuals, mask, weight = collate([case], book, device)
    if int(mask.sum()) != 46:
        raise RuntimeError("0021 was split")
    loss, token_loss, resid_loss = supervised_loss(model, obs, prev, tokens, residuals, weight, mask)
    loss.backward()
    model.zero_grad(set_to_none=True)
    sealed = any("validation" in row.get("snapshot", "") or "final_test" in row.get("snapshot", "") for row in read_jsonl(RAW / "grpo_initial_conditions.jsonl"))
    payload = {
        "qpos_err": err,
        "full_0021": {key: full.get(key) for key in ("reason", "handoff", "recovery_elapsed", "nominal_elapsed", "t_handoff", "t_end", "lift_completed")},
        "forced_10_steps": {key: forced.get(key) for key in ("reason", "handoff", "t_handoff", "t_end", "nominal_elapsed", "lift_completed")},
        "rewards": {"fast_success": fast_ok, "slow_success": slow_ok, "quick_failure": quick_fail, "lambda": LAMBDA_TIME},
        "advantage_skipped_identical": bool(skipped_same),
        "loss_finite": bool(torch.isfinite(loss)),
        "token_loss": float(token_loss),
        "resid_loss": float(resid_loss),
        "sealed_in_grpo": sealed,
        "adv0": float(adv[0]),
    }
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("TIMING", json.dumps(payload), flush=True)


def eval_checkpoint(env, book, model):
    from training.horizon_outcome import roll_policy
    del model
    device = torch.device("cpu")
    loaded, _ref = load_models(CKPT, ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt", device)
    loaded.eval()
    env.model = loaded
    old = {row["id"]: row for row in json.loads(OLD_EVAL.read_text(encoding="utf-8"))}
    rows = []
    for row in read_jsonl(RAW / "grpo_initial_conditions.jsonl"):
        if row.get("source") != "ball_impact_v2":
            continue
        if int(row["id"].rsplit("_", 1)[-1]) % 5 != 0:
            continue
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        out = roll_policy(env, loaded, book, blob, device)
        rows.append({
            "id": row["id"],
            "old_label": old.get(row["id"], {}).get("label"),
            "old_truncated": old.get(row["id"], {}).get("truncated"),
            "reason": out["reason"],
            "success": bool(out.get("success")),
            "handoff": out.get("handoff"),
            "recovery_elapsed": out.get("recovery_elapsed"),
            "nominal_elapsed": out.get("nominal_elapsed"),
            "lift_completed": out.get("lift_completed"),
            "recovery_steps": out.get("recovery_steps"),
            "end": out.get("end"),
        })
        print("POLICY", row["id"], rows[-1]["old_label"], out["reason"], out.get("handoff"), flush=True)
        (OUT / "checkpoint_eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("POLICY_DONE", Counter(row["reason"] for row in rows), flush=True)


def main():
    from training.unified_k47 import build_model
    mode = sys.argv[1] if len(sys.argv) > 1 else "--timing"
    device = torch.device("cpu")
    book = load_codebook()
    model = build_model(book, device)
    env = make_training_env(device, model=model)
    if mode == "--timing":
        timing(env, book, model)
        return
    if mode == "--migrate":
        migrate(env, book)
        return
    if mode == "--views":
        write_views()
        return
    if mode == "--eval-checkpoint":
        eval_checkpoint(env, book, model)
        return
    raise SystemExit("use --timing, --migrate, --views, or --eval-checkpoint")


if __name__ == "__main__":
    main()
