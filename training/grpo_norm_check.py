"""Implementation check for the length-normalized GRPO term.

Does not start formal training. Writes a small JSON under
results/diagnostics/raw/grpo_norm_check/ and does not replace policy A, B, or C.
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

from training.action_representation_expansion import build_policy
from training.autonomous_episode import make_training_env
from training.final_rl_readiness import CAP, SIGMA
from training.future_state_ablation import FuturePolicy
from training.pre_grpo_sanity import (
    CAP_STEPS,
    KL_WEAK,
    LR,
    TRAIN_ORDER,
    actor_surrogate,
    gaussian_logprob,
    grad_norm,
    one_update,
    recovery_step_logprob,
    sequence_logprob,
    sha256,
)
from training.supervised_coverage_warmstart import roll_ic

RAW = ROOT / "results" / "diagnostics" / "raw" / "grpo_norm_check"
SANITY = ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity"
A_PATH = SANITY / "policy_A_reference.pt"
B_PATH = SANITY / "policy_B_init.pt"
CHECK_UPDATES = 8
BEHAVIOR_IDS = ("slip_a", "slip_b", "air_a", "air_b", "imp_c")
LENGTHS = (4, 16, 64)


def load_models(device):
    saved_a = torch.load(A_PATH, map_location="cpu", weights_only=False)
    ref_model = build_policy(np.asarray(saved_a["proto"], np.float32), device, float(saved_a["res_cap"]))
    ref_model.load_state_dict(saved_a["model"])
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    saved_b = torch.load(B_PATH, map_location="cpu", weights_only=False)
    proto = np.asarray(saved_b["proto"], np.float32)
    model = FuturePolicy(proto).to(device)
    model.load_state_dict(saved_b["model"])
    for parameter in model.fut.parameters():
        parameter.requires_grad_(False)
    model.train()
    return model, ref_model


def one_record(model, obs, prev, pre):
    rec = {
        "obs": np.asarray(obs, np.float32),
        "prev": np.asarray(prev, np.float32),
        "token": 0,
        "pre": np.asarray(pre, np.float32),
        "logp": 0.0,
        "valid_recovery": True,
    }
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, res_old, res35 = model.heads(model.encode(
            torch.tensor(rec["obs"], device=device),
            torch.tensor(rec["prev"], device=device),
        ))
        token = int(torch.argmax(logits[-1]).item())
    rec["token"] = token
    step, _token_lp, _resid_lp = recovery_step_logprob(model, rec)
    rec["logp"] = float(step.detach())
    model.zero_grad(set_to_none=True)
    return rec


def surrogate_grad(model, records, reduction):
    model.zero_grad(set_to_none=True)
    loss, used, lengths = actor_surrogate(model, [{"records": records}], np.array([1.0]), reduction)
    if not used or not loss.requires_grad:
        raise RuntimeError("synthetic surrogate produced no gradient")
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    value = grad_norm(params)
    model.zero_grad(set_to_none=True)
    return value, float(loss.detach()), int(lengths[0])


def length_check(model):
    rng = np.random.default_rng(5)
    base_obs = rng.normal(size=(8, 39)).astype(np.float32) * 0.05
    base_prev = rng.normal(size=(8, 7)).astype(np.float32) * 0.05
    base_pre = rng.normal(size=7).astype(np.float32) * 0.02
    identical = one_record(model, base_obs, base_prev, base_pre)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    rows = []
    for kind in ("identical", "varied"):
        for n_steps in LENGTHS:
            if kind == "identical":
                records = [identical for _ in range(n_steps)]
            else:
                records = []
                for i in range(n_steps):
                    obs = base_obs + rng.normal(size=base_obs.shape).astype(np.float32) * 0.01
                    prev = base_prev + rng.normal(size=base_prev.shape).astype(np.float32) * 0.01
                    pre = base_pre + rng.normal(size=7).astype(np.float32) * 0.01
                    records.append(one_record(model, obs, prev, pre))
            with torch.no_grad():
                summed = float(sequence_logprob(model, records, reduction="sum"))
                averaged = float(sequence_logprob(model, records, reduction="mean"))
            g_sum, loss_sum, n_sum = surrogate_grad(model, records, "sum")
            g_mean, loss_mean, n_mean = surrogate_grad(model, records, "mean")
            rows.append({
                "kind": kind,
                "valid_steps": n_steps,
                "counted_steps": n_mean,
                "sum_counted": n_sum,
                "sum_logprob": summed,
                "mean_logprob": averaged,
                "sum_over_mean_logprob": summed / averaged if averaged != 0 else None,
                "old_actor_grad": g_sum,
                "new_actor_grad": g_mean,
                "old_over_new_grad": g_sum / g_mean if g_mean else None,
                "old_loss": loss_sum,
                "new_loss": loss_mean,
            })
            print("LEN", kind, n_steps, "old", round(g_sum, 4), "new", round(g_mean, 4), flush=True)
    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    unchanged = all(torch.equal(a, b) for a, b in zip(before, after))
    return rows, bool(unchanged)


def formula_audit(model):
    rng = np.random.default_rng(11)
    obs = rng.normal(size=(6, 39)).astype(np.float32) * 0.05
    prev = rng.normal(size=(6, 7)).astype(np.float32) * 0.05
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, res_old, res35 = model.heads(model.encode(
            torch.tensor(obs, device=device),
            torch.tensor(prev, device=device),
        ))
        token = int(torch.argmax(logits[-1]).item())
        mean = (res35 if token == 35 else res_old)[-1].float()
        pre = (mean + 0.5).detach().cpu().numpy().astype(np.float32)
    rec = {
        "obs": obs,
        "prev": prev,
        "token": token,
        "pre": pre,
        "logp": 0.0,
        "valid_recovery": True,
    }
    step, token_lp, resid_lp = recovery_step_logprob(model, rec)
    pre_t = torch.tensor(pre, device=device)
    with torch.no_grad():
        logits, res_old, res35 = model.heads(model.encode(
            torch.tensor(obs, device=device),
            torch.tensor(prev, device=device),
        ))
        mean = (res35 if token == 35 else res_old)[-1].float()
        indep_token = torch.log_softmax(logits[-1].float(), dim=-1)[token]
        per_dim = torch.distributions.Normal(mean, SIGMA).log_prob(pre_t)
        clipped = gaussian_logprob(torch.tensor(np.clip(pre, -CAP, CAP), device=device), mean)
    model.zero_grad(set_to_none=True)
    return {
        "expression": "log softmax(logits)[token] + sum_d log Normal(pre_d; residual_mean_d(token), 0.05^2)",
        "token_counted_once": bool(token_lp.ndim == 0),
        "residual_dimensions": int(per_dim.numel()),
        "token_matches": bool(torch.allclose(token_lp.detach(), indep_token)),
        "residual_matches_normal": bool(torch.allclose(resid_lp.detach(), per_dim.sum())),
        "step_is_token_plus_residual": bool(torch.allclose(step.detach(), token_lp.detach() + resid_lp.detach())),
        "deterministic_clip_not_in_density": bool(abs(float(resid_lp.detach()) - float(clipped)) > 1e-3),
        "token_logprob": float(token_lp.detach()),
        "residual_logprob": float(resid_lp.detach()),
    }


def behavior_eval(env, model):
    model.eval()
    env.model = model
    rows = []
    for episode_id in BEHAVIOR_IDS:
        out = roll_ic(env, episode_id)
        rows.append({
            "id": episode_id,
            "label": out.get("label"),
            "success": bool(out.get("success")),
            "token_35": out.get("token_35"),
            "tokens": (out.get("tokens") or [])[:8],
        })
    model.train()
    return rows


def previous_grads():
    payload = json.loads((SANITY / "sanity.json").read_text(encoding="utf-8"))
    rows = payload["runs"][0]["updates"][:CHECK_UPDATES]
    return [
        {
            "update": row["update"],
            "id": row["id"],
            "logged_sum_grad": row["grad_grpo"],
            "logged_kl_grad": row["grad_kl"],
        }
        for row in rows
    ]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    hash_before = {"A": sha256(A_PATH), "B": sha256(B_PATH)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ref_model = load_models(device)
    formula = formula_audit(model)
    print("FORMULA", formula["token_counted_once"], formula["residual_dimensions"], formula["residual_matches_normal"], flush=True)
    lengths, weights_unchanged = length_check(model)
    env = make_training_env()
    env.device = device
    before = behavior_eval(env, model)
    print("BEHAVE0", [(row["id"], row["label"]) for row in before], flush=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    updates = []
    for index in range(1, CHECK_UPDATES + 1):
        episode_id = TRAIN_ORDER[(index - 1) % len(TRAIN_ORDER)]
        rng = np.random.default_rng(3000 + 19 * index + sum(ord(char) for char in episode_id))
        row, _audits = one_update(
            env, model, ref_model, opt, episode_id, KL_WEAK, rng, index, also_sum_grad=True,
        )
        if row["successes"] and max(row["valid_steps"]) > CAP_STEPS:
            raise RuntimeError("a successful rollout counted steps beyond the recovery cap")
        updates.append(row)
    after = behavior_eval(env, model)
    print("BEHAVE8", [(row["id"], row["label"]) for row in after], flush=True)
    hash_after = {"A": sha256(A_PATH), "B": sha256(B_PATH)}
    payload = {
        "hashes_before": hash_before,
        "hashes_after": hash_after,
        "hashes_unchanged": hash_before == hash_after,
        "formula": formula,
        "length": lengths,
        "weights_unchanged_by_length_check": weights_unchanged,
        "behavior_before": before,
        "behavior_after": after,
        "updates": updates,
        "previous_logged_sum_grads": previous_grads(),
        "formal_training_started": False,
    }
    path = RAW / "check.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
