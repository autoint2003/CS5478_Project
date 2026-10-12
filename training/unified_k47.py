"""K=47 supervised initialization, closed-loop evaluation, and GRPO.

`--check` validates data, the codebook, one forward, one supervised loss, and
the GRPO log-probability without an optimizer step. The training subcommands
are not invoked by that check.

The historical K=36 GRPO pilot remains `training/formal_grpo.py`. This file
does not call it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.action_representation_expansion import build_policy
from training.formal_grpo import _reject_head_mismatch, load_models
from training.pre_grpo_sanity import SIGMA, advantages, gaussian_logprob

RAW = ROOT / "results/diagnostics/raw/legal_recovery_v2"
CODEBOOK = RAW / "codebook_selected_k47.npy"
K36_PATH = ROOT / "results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt"
EXTRA_PATH = RAW / "candidate_extra.npy"
SUPERVISED = RAW / "supervised_targets.jsonl"
GRPO_ICS = RAW / "grpo_initial_conditions.jsonl"
FRAGILE = RAW / "fragile_trajectories.jsonl"
CAP = 0.25
REQUIRED_K = 47


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_codebook(path=CODEBOOK):
    book = np.load(path).astype(np.float32)
    if book.shape != (REQUIRED_K, 7):
        raise RuntimeError(f"selected codebook must be ({REQUIRED_K}, 7), got {book.shape}")
    k36 = np.asarray(torch.load(K36_PATH, map_location="cpu", weights_only=False)["proto"], np.float32)
    extra = np.load(EXTRA_PATH).astype(np.float32)
    if not np.array_equal(book[:36], k36):
        raise RuntimeError("tokens 0-35 do not match the original K=36 table")
    if extra.shape != (11, 7) or not np.array_equal(book[36:], extra):
        raise RuntimeError("tokens 36-46 do not match the selected appended centers")
    return book


def build_model(book, device):
    model = build_policy(book, device, CAP)
    if model.logits.out_features != book.shape[0] or model.res.out_features != 7:
        raise RuntimeError("policy head does not match the codebook")
    return model


def decode_action(proto, token, pre):
    """Same map used by the supervised target, the GRPO sample, and evaluation."""
    if int(token) < 0 or int(token) >= len(proto):
        raise RuntimeError(f"token {int(token)} is outside the motion codebook")
    residual = np.clip(np.asarray(pre, np.float32), -CAP, CAP)
    action = np.clip(np.asarray(proto[int(token)], np.float32) + residual, -1.0, 1.0)
    return residual, action


def episode_arrays(row, book):
    blob = np.load(ROOT / row["continuous_npz"])
    actions = blob["actions"].astype(np.float32)
    obs = blob["observations"].astype(np.float32)
    if obs.ndim != 2 or obs.shape[1] != 39 or len(obs) != len(actions):
        raise RuntimeError(f"{row['id']} {row['strategy']} observation alignment")
    token_path = ROOT / row["token_residual_npz"]
    packed = np.load(token_path)
    tokens = packed["tokens"].astype(np.int64)
    residuals = packed["residuals"].astype(np.float32)
    if len(tokens) != len(actions):
        raise RuntimeError(f"{row['id']} {row['strategy']} was split across token files")
    if int(tokens.max()) >= book.shape[0] or int(tokens.min()) < 0:
        raise RuntimeError(f"{row['id']} token outside K={book.shape[0]}")
    if float(np.max(np.abs(residuals))) > CAP + 1e-5:
        raise RuntimeError(f"{row['id']} residual outside ±{CAP}")
    prev = np.zeros_like(actions)
    if len(actions) > 1:
        prev[1:] = actions[:-1]
    return obs, prev, actions, tokens, residuals


def supervised_loss(model, obs, prev, tokens, residuals, weight, mask):
    """Token cross-entropy plus residual MSE. Padded steps are masked out.

    `residuals` are the stored bounded targets. The network residual is
    `0.25 * tanh(head)`, which is the mean used at rollout.
    """
    logits, pred, _act = model(obs, prev)
    if logits.shape[-1] != REQUIRED_K or pred.shape[-1] != 7:
        raise RuntimeError(f"forward shape {tuple(logits.shape)} {tuple(pred.shape)}")
    per_token = []
    per_resid = []
    for index in range(logits.shape[0]):
        active = mask[index]
        if not bool(active.any()):
            raise RuntimeError("a batch row has no unmasked steps")
        per_token.append(F.cross_entropy(logits[index][active], tokens[index][active]))
        per_resid.append(((pred[index][active] - residuals[index][active]) ** 2).mean())
    token_vec = torch.stack(per_token)
    resid_vec = torch.stack(per_resid)
    w = weight if torch.is_tensor(weight) else torch.as_tensor(weight, device=logits.device, dtype=logits.dtype)
    w = w.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if w.shape[0] != token_vec.shape[0]:
        raise RuntimeError("sample weights do not match the batch")
    # Mean of per-trajectory losses times sample_weight. Dividing by the sum of
    # the weights would cancel the mechanism balance.
    loss = (w * (token_vec + resid_vec)).mean()
    return loss, token_vec.mean().detach(), resid_vec.mean().detach()


def collate(rows, book, device):
    packed = [episode_arrays(row, book) for row in rows]
    length = max(item[0].shape[0] for item in packed)
    obs = torch.zeros(len(packed), length, 39, device=device)
    prev = torch.zeros(len(packed), length, 7, device=device)
    tokens = torch.zeros(len(packed), length, dtype=torch.long, device=device)
    residuals = torch.zeros(len(packed), length, 7, device=device)
    mask = torch.zeros(len(packed), length, dtype=torch.bool, device=device)
    weights = []
    for index, (obs_i, prev_i, _act, tok_i, res_i) in enumerate(packed):
        n = obs_i.shape[0]
        obs[index, :n] = torch.tensor(obs_i, device=device)
        prev[index, :n] = torch.tensor(prev_i, device=device)
        tokens[index, :n] = torch.tensor(tok_i, device=device)
        residuals[index, :n] = torch.tensor(res_i, device=device)
        mask[index, :n] = True
        weights.append(float(rows[index]["sample_weight"]))
    return obs, prev, tokens, residuals, mask, torch.tensor(weights, device=device, dtype=torch.float32)


def unified_step_logprob(model, rec):
    """log pi(token, pre-clip residual | history) for the single K-wide head.

    This replaces the K=36 special case that switches to a second residual head
    when the token is 35. The density is the sampled token plus a diagonal
    normal on the residual before the ±0.25 clip. The codebook add and the
    clip are deterministic and have no density.
    """
    device = next(model.parameters()).device
    obs = rec["obs"] if torch.is_tensor(rec["obs"]) else torch.tensor(rec["obs"], device=device)
    prev = rec["prev"] if torch.is_tensor(rec["prev"]) else torch.tensor(rec["prev"], device=device)
    obs = obs.to(device)
    prev = prev.to(device)
    logits, mean, _act = model(obs, prev)
    logit = logits[-1].float()
    token = int(rec["token"])
    if token < 0 or token >= logit.shape[-1]:
        raise RuntimeError(f"sampled token {token} outside logit width {logit.shape[-1]}")
    token_lp = torch.log_softmax(logit, dim=-1)[token]
    resid_lp = gaussian_logprob(torch.tensor(rec["pre"], device=device), mean[-1].float())
    return token_lp + resid_lp


def categorical_kl(model, ref_model, obs, prev):
    """KL(pi || pi_ref) on equal-width logit vectors. No token is padded."""
    logits, _mean, _act = model(obs, prev)
    with torch.no_grad():
        old, _old_mean, _old_act = ref_model(obs, prev)
    if logits.shape[-1] != old.shape[-1]:
        raise RuntimeError(f"KL requires equal token counts, got {logits.shape[-1]} and {old.shape[-1]}")
    log_q = torch.log_softmax(logits[-1].float(), dim=-1)
    log_p = torch.log_softmax(old[-1].float(), dim=-1)
    q = log_q.exp()
    return (q * (log_q - log_p)).sum()


def group_advantage(returns, truncated):
    """Group-relative advantages. Identical returns get advantage 0.

    A truncated rollout is omitted. It is not scored as a drop.
    """
    keep = [value for value, flag in zip(returns, truncated) if not flag]
    if len(keep) < 2:
        return np.zeros(len(returns), np.float64), True
    adv_keep, _std, skipped = advantages(keep)
    out = np.zeros(len(returns), np.float64)
    cursor = 0
    for index, flag in enumerate(truncated):
        if flag:
            continue
        out[index] = adv_keep[cursor]
        cursor += 1
    return out, skipped


def assert_datasets(book):
    supervised = read_jsonl(SUPERVISED)
    grpo = read_jsonl(GRPO_ICS)
    fragile = read_jsonl(FRAGILE)
    if len(supervised) != 288:
        raise RuntimeError(f"supervised count {len(supervised)}")
    ids = {row["id"] for row in supervised}
    if len(ids) != 151:
        raise RuntimeError(f"supervised IC count {len(ids)}")
    if any(not row.get("supervisory_target") for row in supervised):
        raise RuntimeError("supervised file contains a non-target")
    fragile_keys = {(row["id"], row["strategy"]) for row in fragile}
    if fragile_keys & {(row["id"], row["strategy"]) for row in supervised}:
        raise RuntimeError("a fragile trajectory is also a supervisory target")
    if len(grpo) != 312:
        raise RuntimeError(f"GRPO IC count {len(grpo)}")
    banned = ("validation", "final_test", "ball_validation", "ball_final")
    for row in grpo + supervised:
        text = row["id"] + " " + row.get("snapshot", "")
        if any(word in text for word in banned):
            raise RuntimeError(f"sealed split leaked into training data: {row['id']}")
    weights = {}
    for row in supervised:
        weights.setdefault(row["mechanism"], 0.0)
        weights[row["mechanism"]] += float(row["sample_weight"])
    if len(weights) != 5:
        raise RuntimeError(f"mechanism groups {sorted(weights)}")
    for name, total in weights.items():
        if abs(total - 1.0) > 1e-3:
            raise RuntimeError(f"mechanism {name} weight sum {total}")
    case = next(row for row in supervised if row["id"] == "ball_train_v2_0021" and row["strategy"] == "release_long")
    _obs, prev, actions, tokens, residuals = episode_arrays(case, book)
    if len(actions) != 46 or int(case["none_intervals"]) != 3:
        raise RuntimeError("0021 is not the intact 46-step episode")
    if not np.array_equal(prev[1:], actions[:-1]):
        raise RuntimeError("previous-action alignment")
    if not np.array_equal(prev[0], np.zeros(7, np.float32)):
        raise RuntimeError("episode history was reset with a nonzero previous action")
    _ = tokens, residuals
    return supervised, grpo, weights


def assert_horizon_manifest(book):
    """The migrated supervised and GRPO views. Does not replace the frozen 288-row check."""
    from training.horizon_outcome import LAMBDA_TIME, NOMINAL_S, T_RECOVERY_MAX

    folder = ROOT / "results/diagnostics/raw/horizon_migration"
    supervised = read_jsonl(folder / "supervised_targets.jsonl")
    grpo = read_jsonl(folder / "grpo_initial_conditions.jsonl")
    if len(supervised) != 296:
        raise RuntimeError(f"migrated supervised count {len(supervised)}")
    if len({row["id"] for row in supervised}) != 153:
        raise RuntimeError("migrated supervised IC count")
    if len(grpo) != 312 or any(row.get("horizon") != "recovery_limit_then_nominal_12s" for row in grpo):
        raise RuntimeError("migrated GRPO manifest")
    banned = ("validation", "final_test", "ball_validation", "ball_final")
    for row in grpo + supervised:
        text = row["id"] + " " + row.get("snapshot", "")
        if any(word in text for word in banned):
            raise RuntimeError(f"sealed split leaked into the migrated training data: {row['id']}")
    weights = {}
    for row in supervised:
        weights.setdefault(row["mechanism"], 0.0)
        weights[row["mechanism"]] += float(row["sample_weight"])
    for name, total in weights.items():
        if abs(total - 1.0) > 1e-3:
            raise RuntimeError(f"migrated mechanism {name} weight sum {total}")
    case = next(row for row in supervised if row["id"] == "ball_train_v2_0021" and row["strategy"] == "release_long")
    _obs, prev, actions, _tokens, residuals = episode_arrays(case, book)
    if len(actions) != 46 or float(np.max(np.abs(residuals))) > CAP + 1e-5:
        raise RuntimeError("migrated 0021 is not an intact in-bound episode")
    if not np.array_equal(prev[0], np.zeros(7, np.float32)):
        raise RuntimeError("migrated episode history was reset with a nonzero previous action")
    return {
        "supervised": len(supervised),
        "lambda_time": LAMBDA_TIME,
        "nominal_seconds": NOMINAL_S,
        "recovery_limit_s": T_RECOVERY_MAX,
    }


def reject_mismatched_horizon(cfg):
    from training.horizon_outcome import LAMBDA_TIME, NOMINAL_S, T_RECOVERY_MAX

    expected = {
        "lambda_time": LAMBDA_TIME,
        "nominal_seconds": NOMINAL_S,
        "recovery_limit_s": T_RECOVERY_MAX,
    }
    for key, value in expected.items():
        if key in cfg and abs(float(cfg[key]) - float(value)) > 1e-9:
            raise RuntimeError(f"{key} is {cfg[key]} in the config and {value} in horizon_outcome")


def one_mujoco_step(book):
    from training.autonomous_episode import make_training_env
    from training.history_aware_critic_ablation import restore_capture
    from training.privileged_ball_impact_search import step_legal
    from training.recovery_success import label_of

    device = torch.device("cpu")
    model = build_model(book, device)
    env = make_training_env(device, model=model)
    path = ROOT / "results/diagnostics/raw/ball_impact_dataset_v2/accepted/train/ball_train_v2_0021.pkl"
    with path.open("rb") as handle:
        blob = pickle.load(handle)
    restore_capture(env, blob["capture"])
    err = float(np.max(np.abs(np.array(env.sim.data.qpos) - blob["capture"]["physics"]["qpos"])))
    if err > 1e-6:
        raise RuntimeError(f"restore qpos error {err}")
    obs = np.asarray(env.observation(), np.float32)
    env._obs_seq.append(obs)
    env._prev_seq.append(env._prev.copy())
    logits, mean, act = model(
        torch.tensor(np.stack(env._obs_seq), device=device),
        torch.tensor(np.stack(env._prev_seq), device=device),
    )
    token = int(torch.argmax(logits[-1]).item())
    pre = mean[-1].detach().float().cpu().numpy()
    _residual, action = decode_action(model.proto.detach().cpu().numpy(), token, pre)
    executed = step_legal(env.sim, action)
    env._prev = np.asarray(executed, np.float32)
    if len(env._obs_seq) != 1 or env._obs_seq[0].shape != (39,) or env._prev.shape != (7,):
        raise RuntimeError("history was not the observation plus the executed action")
    # A step that has not handed off is not a success label.
    probe = label_of(False, False, False, None)
    if probe == "SUCCESS_12S":
        raise RuntimeError("basin miss was labeled as success")
    return {"token": token, "action": [round(float(x), 4) for x in executed], "qpos_err": err, "label_without_handoff": probe}


def check_loader(book, device):
    model = build_model(book, device)
    logits, residual, action = model(torch.zeros(2, 3, 39, device=device), torch.zeros(2, 3, 7, device=device))
    if tuple(logits.shape) != (2, 3, REQUIRED_K) or tuple(residual.shape) != (2, 3, 7):
        raise RuntimeError("forward shape")
    if float(residual.detach().abs().max()) > CAP + 1e-6:
        raise RuntimeError("residual cap")
    folder = RAW / "_k47_check"
    folder.mkdir(parents=True, exist_ok=True)
    good = folder / "roundtrip.pt"
    torch.save({"proto": book, "res_cap": CAP, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, good)
    loaded, _ref = load_models(good, ROOT / "results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt", device)
    if loaded.logits.out_features != REQUIRED_K:
        raise RuntimeError("reloaded head width")
    bad_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    bad_state["logits.weight"] = torch.zeros(36, 64)
    bad_state["logits.bias"] = torch.zeros(36)
    bad = folder / "bad_head.pt"
    torch.save({"proto": book, "res_cap": CAP, "model": bad_state}, bad)
    rejected = False
    try:
        _reject_head_mismatch(model, bad_state)
    except RuntimeError as exc:
        rejected = "incompatible token head" in str(exc)
    good.unlink(missing_ok=True)
    bad.unlink(missing_ok=True)
    if not any(folder.iterdir()):
        folder.rmdir()
    if not rejected:
        raise RuntimeError("a 36-wide logit matrix was accepted by the K=47 head")
    _ = loaded, action
    return True


def run_check():
    book = load_codebook()
    supervised, _grpo, weights = assert_datasets(book)
    device = torch.device("cpu")
    check_loader(book, device)
    # Historical K=36 checkpoint still loads, and is not remapped.
    legacy, _ref = load_models(
        K36_PATH,
        ROOT / "results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt",
        device,
    )
    if legacy.logits_old.out_features != 35:
        raise RuntimeError("K=36 head was rewritten")
    model = build_model(book, device)
    batch_rows = [
        supervised[0],
        next(row for row in supervised if row["id"] == "ball_train_v2_0021" and row["strategy"] == "release_long"),
    ]
    obs, prev, tokens, residuals, mask, weight = collate(batch_rows, book, device)
    if tuple(weight.shape) != (2,) or abs(float(weight[1]) - float(batch_rows[1]["sample_weight"])) > 1e-6:
        raise RuntimeError("per-trajectory mechanism weight was not kept")
    if int(mask[1].sum()) != 46:
        raise RuntimeError("0021 mask is not the full episode")
    loss, token_loss, resid_loss = supervised_loss(model, obs, prev, tokens, residuals, weight, mask)
    if float(token_loss) <= 0.0 or float(resid_loss) < 0.0:
        raise RuntimeError("supervised terms did not both contribute")
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads:
        raise RuntimeError("supervised loss produced no gradient")
    model.zero_grad(set_to_none=True)
    # Synthetic GRPO group: two successes, one failure, one truncation.
    returns = [1.0, 1.0, 0.0, 0.0]
    truncated = [False, False, False, True]
    adv, skipped = group_advantage(returns, truncated)
    if skipped or abs(float(adv[3])) > 0.0:
        raise RuntimeError("truncation entered the advantage")
    identical, skipped_same = group_advantage([1.0, 1.0, 1.0], [False, False, False])
    if not skipped_same or float(np.max(np.abs(identical))) > 0.0:
        raise RuntimeError("identical returns did not get zero advantage")
    rec = {
        "obs": obs[1, :46].detach(),
        "prev": prev[1, :46].detach(),
        "token": int(tokens[1, 0]),
        "pre": residuals[1, 0].detach().cpu().numpy(),
        "valid_recovery": True,
    }
    step_lp = unified_step_logprob(model, rec)
    if not torch.isfinite(step_lp):
        raise RuntimeError("GRPO log-probability is not finite")
    step_lp.backward()
    model.zero_grad(set_to_none=True)
    physical = one_mujoco_step(book)
    horizon = assert_horizon_manifest(book)
    from envs.config_util import load_yaml
    long_cfg = load_yaml(ROOT / "config/unified_k47_horizon_long.yaml")
    long_ics = read_jsonl(ROOT / long_cfg["grpo_initial_conditions"])
    long_updates = int(long_cfg["grpo_updates"])
    covered = {long_ics[(update - 1) % len(long_ics)]["id"] for update in range(1, long_updates + 1)}
    if len(long_ics) != 312 or len(covered) != 312:
        raise RuntimeError(f"long GRPO covers {len(covered)} of {len(long_ics)}")
    print("CHECK_OK", "supervised", len(supervised), "weights", {key: round(value, 4) for key, value in weights.items()}, "physical", physical, "horizon", horizon, "grpo_cover", len(covered), flush=True)


def train_supervised(cfg):
    book = load_codebook(ROOT / cfg["codebook"])
    rows = read_jsonl(ROOT / cfg["supervised_targets"])
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg["seed"]))
    model = build_model(book, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    out = ROOT / cfg["supervised_output"]
    historical = ROOT / "results/diagnostics/raw/unified_k47/supervised"
    if out.resolve() == historical.resolve():
        raise RuntimeError("refusing to write into the historical epoch-40 checkpoint directory")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["seed"]))
    batch = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(rows))
        total = 0.0
        seen = 0
        for start in range(0, len(rows), batch):
            chosen = [rows[int(index)] for index in order[start:start + batch]]
            obs, prev, tokens, residuals, mask, weight = collate(chosen, book, device)
            opt.zero_grad(set_to_none=True)
            loss, _token, _resid = supervised_loss(model, obs, prev, tokens, residuals, weight, mask)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            seen += 1
        torch.save(
            {"proto": book, "res_cap": CAP, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epoch": epoch},
            out / f"checkpoint_epoch_{epoch:04d}.pt",
        )
        print("SUPERVISED", epoch, round(total / max(seen, 1), 6), flush=True)
    print("WROTE", out / f"checkpoint_epoch_{epochs:04d}.pt", flush=True)


def evaluate(cfg, checkpoint):
    from training.autonomous_episode import make_training_env
    from training.horizon_outcome import T_RECOVERY_MAX, roll_policy

    book = load_codebook(ROOT / cfg["codebook"])
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ref = load_models(Path(checkpoint), ROOT / "results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt", device)
    if model.logits.out_features != REQUIRED_K:
        raise RuntimeError("evaluation checkpoint is not K=47")
    model.eval()
    env = make_training_env(device, model=model)
    held = [
        row for row in read_jsonl(ROOT / cfg["grpo_initial_conditions"])
        if row["source"] == "ball_impact_v2" and int(row["id"].rsplit("_", 1)[-1]) % 5 == 0
    ]
    limit = int(cfg.get("eval_limit") or len(held))
    rows = []
    for row in held[:limit]:
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if blob["spec"].get("split") not in (None, "train"):
            raise RuntimeError(f"evaluation refused non-train snapshot {row['id']}")
        out = roll_policy(env, model, book, blob, device, cap_steps=int(cfg.get("eval_cap_steps") or 120))
        rows.append({
            "id": row["id"],
            "label": out["reason"],
            "success": bool(out.get("success")),
            "handoff": out.get("handoff"),
            "recovery_elapsed": out.get("recovery_elapsed"),
            "nominal_elapsed": out.get("nominal_elapsed"),
            "lift_completed": out.get("lift_completed"),
            "recovery_limit_s": T_RECOVERY_MAX,
            "truncated": out["reason"] == "INVALID",
        })
        print("EVAL", row["id"], out["reason"], out.get("handoff"), flush=True)
    out = ROOT / cfg["eval_output"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout_train.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("EVAL_DONE", len(rows), sum(1 for row in rows if row["success"]), flush=True)
    baseline_path = cfg.get("baseline_eval")
    if baseline_path:
        baseline = json.loads((ROOT / baseline_path).read_text(encoding="utf-8"))
        base_success = sum(1 for row in baseline if row.get("success") or row.get("reason") == "SUCCESS_12S")
        new_success = sum(1 for row in rows if row["success"])
        print("BASELINE", f"{base_success}/{len(baseline)}", "NEW", f"{new_success}/{len(rows)}", baseline_path, flush=True)


def train_grpo(cfg, checkpoint):
    from training.autonomous_episode import make_training_env
    from training.history_aware_critic_ablation import restore_capture
    from training.horizon_outcome import (
        finish_recovery,
        physical_reward,
        recovery_failed,
        recovery_steps,
    )
    from training.nominal_pose_audit import errors
    from training.privileged_ball_impact_search import step_legal
    from training.recovery_success import PERSIST_STEPS, basin_ok

    book = load_codebook(ROOT / cfg["codebook"])
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ref_model = load_models(Path(checkpoint), Path(checkpoint), device)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    env = make_training_env(device, model=model)
    ics = read_jsonl(ROOT / cfg["grpo_initial_conditions"])
    if any("final_test" in row["snapshot"] or "validation" in row["snapshot"] for row in ics):
        raise RuntimeError("GRPO manifest contains a sealed split")
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["grpo_lr"]))
    out = ROOT / cfg["grpo_output"]
    out.mkdir(parents=True, exist_ok=True)
    group = int(cfg["group"])
    updates = int(cfg["grpo_updates"])
    covered = {ics[(update - 1) % len(ics)]["id"] for update in range(1, updates + 1)}
    if cfg.get("cover_all_ics") and len(covered) != len(ics):
        raise RuntimeError(f"GRPO updates {updates} cover {len(covered)} of {len(ics)} initial conditions")
    cap_steps = int(cfg.get("rollout_cap_steps") or recovery_steps())
    for update in range(1, updates + 1):
        row = ics[(update - 1) % len(ics)]
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if str(blob.get("spec", {}).get("split", "train")) != "train":
            raise RuntimeError(f"GRPO refused {row['id']}")
        rng = np.random.default_rng(int(cfg["seed"]) + 19 * update)
        members = []
        for _member in range(group):
            restore_capture(env, blob["capture"])
            records = []
            basin_run = 0
            lost_low = 0
            how = None
            invalid = False
            t0 = float(env.sim.data.time)
            for _step in range(cap_steps):
                obs = np.asarray(env.observation(), np.float32)
                env._obs_seq.append(obs)
                env._prev_seq.append(env._prev.copy())
                window_obs = np.stack(env._obs_seq[-160:]).astype(np.float32)
                window_prev = np.stack(env._prev_seq[-160:]).astype(np.float32)
                with torch.no_grad():
                    logits, mean, _act = model(
                        torch.tensor(window_obs, device=device),
                        torch.tensor(window_prev, device=device),
                    )
                logit = logits[-1].float()
                probs = torch.softmax(logit, dim=-1).cpu().numpy()
                probs = np.clip(probs, 0.0, None)
                probs = probs / probs.sum()
                token = int(rng.choice(len(probs), p=probs))
                noise = rng.normal(0.0, SIGMA, size=7).astype(np.float32)
                pre = mean[-1].detach().float().cpu().numpy() + noise
                _residual, action = decode_action(book, token, pre)
                token_lp = torch.log_softmax(logit, dim=-1)[token]
                resid_lp = gaussian_logprob(torch.tensor(pre, device=device), mean[-1].float())
                records.append({
                    "obs": window_obs,
                    "prev": window_prev,
                    "token": token,
                    "pre": pre.astype(np.float32),
                    "logp": float(token_lp.detach()) + float(resid_lp.detach()),
                    "valid_recovery": True,
                })
                executed = step_legal(env.sim, action)
                env._prev = np.asarray(executed, np.float32)
                failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
                if failed:
                    how = "drop"
                    break
                inside = basin_ok(errors(env.sim, blob["ref"]), float(executed[6]) > 0.5)
                basin_run = basin_run + 1 if inside else 0
                if basin_run >= PERSIST_STEPS:
                    how = "early"
                    break
            if how == "drop":
                elapsed = float(env.sim.data.time) - t0
                outcome = {"reason": "DROP_DURING_RECOVERY", "reward": physical_reward("DROP_DURING_RECOVERY", elapsed)}
            else:
                outcome = finish_recovery(env, blob["ref"], t0, "forced" if how is None else how)
                invalid = outcome["reason"] == "INVALID"
            members.append({
                "records": records,
                "return": None if invalid else float(outcome["reward"]),
                "truncated": invalid,
                "label": outcome["reason"],
            })
        adv, skipped = group_advantage(
            [member["return"] for member in members],
            [member["truncated"] for member in members],
        )
        opt.zero_grad(set_to_none=True)
        if not skipped:
            terms = []
            for member, advantage in zip(members, adv):
                if member["truncated"] or abs(float(advantage)) < 1e-8 or not member["records"]:
                    continue
                logp = torch.stack([unified_step_logprob(model, rec) for rec in member["records"]]).mean()
                behavior = float(np.mean([rec["logp"] for rec in member["records"]]))
                ratio = torch.exp(torch.clamp(logp - behavior, -20.0, 20.0))
                unclipped = ratio * float(advantage)
                clipped = torch.clamp(ratio, 1.0 - float(cfg["clip"]), 1.0 + float(cfg["clip"])) * float(advantage)
                terms.append(-torch.minimum(unclipped, clipped))
            if terms:
                surrogate = torch.stack(terms).mean()
                kl_terms = []
                for member in members:
                    if member["truncated"] or not member["records"]:
                        continue
                    rec = member["records"][0]
                    kl_terms.append(categorical_kl(
                        model, ref_model,
                        torch.tensor(rec["obs"], device=device),
                        torch.tensor(rec["prev"], device=device),
                    ))
                kl = torch.stack(kl_terms).mean() if kl_terms else torch.zeros((), device=device)
                loss = surrogate + float(cfg["kl_coef"]) * kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        if update % int(cfg.get("grpo_save_every") or 5) == 0 or update == updates:
            torch.save(
                {"proto": book, "res_cap": CAP, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "update": update},
                out / f"checkpoint_u{update:04d}.pt",
            )
        print("GRPO", update, [member["label"] for member in members], flush=True)
    print("GRPO_DONE", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description="K=47 unified recovery. --check does not train.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--config", default="config/unified_k47.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    if args.check:
        run_check()
        return
    from envs.config_util import load_yaml
    cfg = load_yaml(args.config)
    reject_mismatched_horizon(cfg)
    if args.supervised:
        train_supervised(cfg)
        return
    if args.eval:
        if not args.checkpoint:
            raise SystemExit("--eval requires --checkpoint")
        evaluate(cfg, args.checkpoint)
        return
    if args.grpo:
        if not args.checkpoint:
            raise SystemExit("--grpo requires --checkpoint")
        train_grpo(cfg, args.checkpoint)
        return
    raise SystemExit("choose --check, --supervised, --eval, or --grpo")


if __name__ == "__main__":
    main()
