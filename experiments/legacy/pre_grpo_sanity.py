"""Critic-free GRPO update used by the pre-training sanity check.

Policy A is the K=35 pre-airborne reference. Policy B is the saved K=36
supervised checkpoint with the best closed-loop union that still holds slip,
natural-loss, and airborne. The completed 40-update run wrote policy C beside
them. This module does not overwrite those files.

The actor term is the clipped group-relative surrogate on the mean
token-plus-residual log-probability over valid recovery steps. Nominal hold
steps are not in that mean. FINAL_TEST is not rolled out.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs.recovery_success as success
from models.k48_policy import build_policy
from training.autonomous_episode import Z_ABORT, make_training_env, seat_base
from training.autonomous_rl_sanity import supports_of
from experiments.legacy.final_rl_readiness import CAP, CATALOG, CKPT, GROUP, load_ic
from experiments.legacy.future_state_ablation import FuturePolicy, counts, evaluate, run34
from training.history_aware_critic_ablation import restore_capture
from training.nominal_pose_audit import errors
from experiments.legacy.supervised_coverage_warmstart import dataset
import training.natural_loss_recatch_transformer_v3 as v3
from training.grpo_math import LOG2PI, SIGMA, advantages, gaussian_logprob

RAW = ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity"
REF_SRC = Path(CKPT)
INIT_SRC = ROOT / "results" / "diagnostics" / "raw" / "future_state_ablation" / "ckpt_l0.1_s2.pt"
UPDATES = 40
EVAL_AT = (0, 10, 20, 40)
KL_WEAK = 0.05
KL_MEDIUM = 0.2
LR = 3e-5
CLIP = 0.2
CAP_STEPS = 100
TRAIN_ORDER = (
    "slip_a", "air_a", "imp_c", "slip_b", "air_b", "slip_d",
    "imp_a", "seat_a", "slip_c", "imp_b", "seat_b", "imp_d",
)
DRIFT_NAMES = ("demo_nl_dz14.6", "demo_impact_zp6", "demo_wrist_reseat", "success_air_a")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_copy(src, dst):
    if not dst.exists():
        shutil.copy2(src, dst)
    return sha256(dst)


def family_of(episode_id):
    for spec in CATALOG:
        if spec["id"] == episode_id:
            return spec["family"]
    raise KeyError(episode_id)


def retain_family(family):
    return family != "airborne"


def grad_norm(params):
    total = 0.0
    for parameter in params:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total ** 0.5




def sample_rollout(env, model, ref, rng):
    records = []
    tokens, grips = [], []
    fell = False
    basin_run = 0
    for _ in range(CAP_STEPS):
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        obs_w = np.stack(env._obs_seq[-160:]).astype(np.float32)
        prev_w = np.stack(env._prev_seq[-160:]).astype(np.float32)
        with torch.no_grad():
            logits, res_old, res35 = model.heads(model.encode(
                torch.tensor(obs_w, device=env.device),
                torch.tensor(prev_w, device=env.device),
            ))
        logit = logits[-1].float()
        probs = torch.softmax(logit, dim=-1).cpu().numpy()
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        token = int(rng.choice(len(probs), p=probs))
        mean = (res35 if token == 35 else res_old)[-1].float()
        noise = rng.normal(0.0, SIGMA, size=7).astype(np.float32)
        pre = mean.detach().cpu().numpy() + noise
        resid = np.clip(pre, -CAP, CAP).astype(np.float32)
        proto = model.proto[token].float().cpu().numpy()
        act = np.clip(proto + resid, -1.0, 1.0).astype(np.float32)
        token_lp = torch.log_softmax(logit, dim=-1)[token]
        resid_lp = gaussian_logprob(torch.tensor(pre, device=mean.device), mean)
        logp = float(token_lp) + float(resid_lp)
        # This list is the valid-step mask. Nominal validation, padding, and
        # steps after the drop are never appended.
        records.append({
            "obs": obs_w,
            "prev": prev_w,
            "token": token,
            "pre": pre.astype(np.float32),
            "logp": logp,
            "valid_recovery": True,
        })
        v3.step_v3(env.sim, act)
        env._prev = act
        tokens.append(token)
        grips.append(float(act[6]))
        row = errors(env.sim, ref)
        if row["z"] < Z_ABORT:
            fell = True
            break
        inside = success.basin_ok(row, float(act[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        if basin_run >= success.PERSIST_STEPS:
            audit = success.apply_handoff(env.sim, ref)
            hold = success.nominal_validate(env, ref)
            label = success.label_of(False, True, True, hold)
            return _pack(label, tokens, grips, records, audit, hold, True)
    label = success.label_of(fell, False, False, None)
    return _pack(label, tokens, grips, records, None, None, False)


def _pack(label, tokens, grips, records, audit, hold, handed):
    ok = bool(success.is_success(label))
    audit_slim = None
    if audit is not None and hold is not None:
        audit_slim = {
            "label": label,
            "success": ok,
            "return_uses_success_flag": ok,
            "held_10": bool(hold["held_10"]),
            "held_12": bool(hold["held_12"]),
            "t_handoff": float(hold["t_handoff"]),
            "recovery_called_during_hold": bool(hold["recovery_called_during_hold"]),
            "physics_edited": bool(audit["physics_edited"]),
            "qpos_max_abs": float(audit["qpos_max_abs"]),
            "v_cmd_norm": float(audit["v_cmd_norm"]),
            "w_cmd_norm": float(audit["w_cmd_norm"]),
            "legacy_retained_to_12": hold.get("legacy_retained_to_12"),
        }
    return {
        "label": label,
        "success": ok,
        "return": 1.0 if ok else 0.0,
        "tokens": tokens,
        "token34_run": run34(tokens),
        "token_35": int(sum(t == 35 for t in tokens)),
        "open_steps": int(sum(g < -0.5 for g in grips)),
        "records": records,
        "audit": audit_slim,
        "handed": handed,
    }


def valid_records(records):
    return [rec for rec in records if rec.get("valid_recovery")]


def recovery_step_logprob(model, rec):
    """Log-density of one stochastic recovery action.

    log pi(a_t | h_t) = log softmax(logits_t)[k_t]
        + sum_{d=1..7} log Normal(pre_t,d; mu_t,d(k_t), sigma^2)

    pre_t is the residual before the +/-0.25 clip. Adding the codebook vector
    and clipping the command are deterministic and are not given density.
    Only the last position is scored. Earlier history is context, not a
    second action, and there is no padded position in this window.
    """
    if not rec.get("valid_recovery"):
        raise RuntimeError("log-prob requested for a step outside recovery control")
    device = next(model.parameters()).device
    obs = torch.tensor(rec["obs"], device=device)
    prev = torch.tensor(rec["prev"], device=device)
    logits, res_old, res35 = model.heads(model.encode(obs, prev))
    logit = logits[-1].float()
    token = int(rec["token"])
    mean = (res35 if token == 35 else res_old)[-1].float()
    pre = torch.tensor(rec["pre"], device=device)
    token_lp = torch.log_softmax(logit, dim=-1)[token]
    resid_lp = gaussian_logprob(pre, mean)
    return token_lp + resid_lp, token_lp, resid_lp


def sequence_logprob(model, records, reduction="mean"):
    """Mean log-probability over valid recovery steps.

    reduction='sum' exists only so a diagnostic can compare the old
    length-scaled surrogate. Training calls this with the mean.
    """
    chosen = valid_records(records)
    device = next(model.parameters()).device
    if not chosen:
        return torch.zeros((), device=device)
    total = None
    for rec in chosen:
        step, _token_lp, _resid_lp = recovery_step_logprob(model, rec)
        total = step if total is None else total + step
    if reduction == "sum":
        return total
    if reduction == "mean":
        return total / len(chosen)
    raise ValueError(reduction)


def behavior_logprob(records, reduction="mean"):
    chosen = valid_records(records)
    if not chosen:
        return 0.0
    values = [float(rec["logp"]) for rec in chosen]
    if reduction == "sum":
        return float(np.sum(values))
    if reduction == "mean":
        return float(np.mean(values))
    raise ValueError(reduction)


def retention_kl(model, ref_model, records):
    pieces = []
    for rec in valid_records(records):
        obs = torch.tensor(rec["obs"], device=next(model.parameters()).device)
        prev = torch.tensor(rec["prev"], device=obs.device)
        cur = model.heads(model.encode(obs, prev))[0][-1].float()
        with torch.no_grad():
            old = ref_model(obs, prev)[0][-1].float()
        pad = old.new_full((1,), -100.0)
        ref = torch.cat([old, pad], dim=0)
        log_q = torch.log_softmax(cur, dim=-1)
        log_p = torch.log_softmax(ref, dim=-1)
        q = log_q.exp()
        pieces.append((q * (log_q - log_p)).sum())
    if not pieces:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(pieces).mean()




def actor_surrogate(model, members, adv, reduction="mean"):
    terms = []
    lengths = []
    for member, advantage in zip(members, adv):
        records = valid_records(member["records"])
        if abs(float(advantage)) < 1e-8 or not records:
            continue
        logp = sequence_logprob(model, records, reduction=reduction)
        behavior = behavior_logprob(records, reduction=reduction)
        ratio = torch.exp(torch.clamp(logp - behavior, -20.0, 20.0))
        unclipped = ratio * float(advantage)
        clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * float(advantage)
        terms.append(-torch.minimum(unclipped, clipped))
        lengths.append(len(records))
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device), 0, lengths
    return torch.stack(terms).mean(), len(terms), lengths


def one_update(env, model, ref_model, opt, episode_id, kl_coef, rng, update_index, also_sum_grad=False, blob=None, family=None):
    if blob is None:
        family = family_of(episode_id)
        train_ids = {spec["id"] for spec in CATALOG if spec["split"] == "train"}
        if episode_id not in train_ids:
            raise RuntimeError(f"{episode_id} is outside the train split")
        blob = load_ic(episode_id)
    else:
        split = str(blob.get("spec", {}).get("split", ""))
        if split == "final_test":
            raise RuntimeError(f"FINAL_TEST is not used: {episode_id}")
        if split != "train":
            raise RuntimeError(f"{episode_id} is outside the train split")
        if family is None:
            family = "impact"
    members = []
    for _g in range(GROUP):
        restore_capture(env, blob["capture"])
        members.append(sample_rollout(env, model, blob["ref"], rng))
    returns = [m["return"] for m in members]
    adv, std, skipped = advantages(returns)
    params = [p for p in model.parameters() if p.requires_grad]
    opt.zero_grad(set_to_none=True)
    g_sum = None
    if also_sum_grad and not skipped:
        summed, used_sum, _sum_lengths = actor_surrogate(model, members, adv, reduction="sum")
        if used_sum and summed.requires_grad:
            summed.backward()
            g_sum = grad_norm(params)
            opt.zero_grad(set_to_none=True)
        del summed
    grpo, used, lengths = (torch.zeros((), device=env.device), 0, [])
    if not skipped:
        grpo, used, lengths = actor_surrogate(model, members, adv, reduction="mean")
    kl = torch.zeros((), device=env.device)
    if retain_family(family):
        records = []
        for member in members:
            records.extend(valid_records(member["records"])[:8])
        if records:
            kl = retention_kl(model, ref_model, records)
    g_grpo = 0.0
    g_kl = 0.0
    if used and grpo.requires_grad:
        grpo.backward(retain_graph=True)
        g_grpo = grad_norm(params)
        opt.zero_grad(set_to_none=True)
    if retain_family(family) and kl.requires_grad:
        (kl_coef * kl).backward(retain_graph=True)
        g_kl = grad_norm(params)
        opt.zero_grad(set_to_none=True)
    loss = grpo + kl_coef * kl
    stepped = False
    pre_clip = 0.0
    if loss.requires_grad and (used or retain_family(family)):
        loss.backward()
        pre_clip = grad_norm(params)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        stepped = True
    opt.zero_grad(set_to_none=True)
    finite = all(np.isfinite(x) for x in (float(grpo.detach()), float(kl.detach()), g_grpo, g_kl, pre_clip))
    row = {
        "update": update_index,
        "id": episode_id,
        "family": family,
        "retention": retain_family(family),
        "kl_coef": kl_coef,
        "returns": returns,
        "advantages": [round(float(x), 4) for x in adv],
        "advantage_std": std,
        "grpo_skipped": bool(skipped or used == 0),
        "labels": [m["label"] for m in members],
        "successes": int(sum(m["success"] for m in members)),
        "token34_runs": [m["token34_run"] for m in members],
        "token35": [m["token_35"] for m in members],
        "tokens": [m["tokens"][:8] for m in members],
        "valid_steps": [int(sum(rec.get("valid_recovery") for rec in m["records"])) for m in members],
        "grpo_loss": float(grpo.detach()),
        "kl": float(kl.detach()),
        "grad_grpo": g_grpo,
        "grad_grpo_sum": None if g_sum is None else g_sum,
        "grad_kl": g_kl,
        "grad_pre_clip": pre_clip,
        "finite": bool(finite),
        "stepped": stepped,
    }
    audits = [m["audit"] for m in members if m["audit"] is not None]
    print(
        "UPD", update_index, episode_id, row["successes"], "/", GROUP,
        "steps", row["valid_steps"],
        "skip", row["grpo_skipped"], "KL", round(row["kl"], 4),
        "g", round(g_grpo, 4), round(g_kl, 4),
        "sum", None if g_sum is None else round(g_sum, 4), flush=True,
    )
    return row, audits


def drift_report(model, ref_model, trajs, device):
    rows = []
    model.eval()
    ref_model.eval()
    with torch.no_grad():
        for tr in trajs:
            if tr["name"] not in DRIFT_NAMES:
                continue
            n = min(8, len(tr["obs"]))
            obs = torch.tensor(tr["obs"][:n], device=device)
            prev = torch.tensor(tr["prev"][:n], device=device)
            cur_logits, cur_res, cur_act = model(obs, prev)
            ref_logits, ref_res, ref_act = ref_model(obs, prev)
            cur_tok = cur_logits.argmax(-1).cpu().numpy()
            ref_tok = ref_logits.argmax(-1).cpu().numpy()
            pad = ref_logits.new_full(ref_logits.shape[:-1] + (1,), -100.0)
            ref_full = torch.cat([ref_logits, pad], dim=-1)
            log_q = torch.log_softmax(cur_logits.float(), dim=-1)
            log_p = torch.log_softmax(ref_full.float(), dim=-1)
            kl = (log_q.exp() * (log_q - log_p)).sum(-1).mean()
            switched = int(np.sum(cur_tok != ref_tok))
            rows.append({
                "name": tr["name"],
                "steps": int(n),
                "ref_tokens": [int(t) for t in ref_tok],
                "cur_tokens": [int(t) for t in cur_tok],
                "token_switches_vs_ref": switched,
                "token_35": int(np.sum(cur_tok == 35)),
                "action_l2": float(torch.norm(cur_act - ref_act).item()),
                "residual_l2": float(torch.norm(cur_res - ref_res).item()),
                "kl": float(kl),
            })
    return rows


def slim_eval(env, table, base_q, q_ref):
    rows = evaluate(env, table, base_q, q_ref)
    summary = counts(rows)
    timing = []
    for row in rows:
        if row["id"] in ("slip_a", "slip_b", "slip_d", "nl_dz14.6", "nl_dvz_p", "air_a", "air_b", "imp_c"):
            timing.append({
                "id": row["id"],
                "label": row["label"],
                "success": row["success"],
                "token34_run": row["token34_run"],
                "tokens": row["tokens"],
                "token_35": row["token_35"],
                "open_step": row["open_step"],
                "chase_steps": row["chase_steps"],
                "close_step": row["close_step"],
            })
    return summary, timing, [{k: v for k, v in row.items() if k != "tokens"} for row in rows]


def collapsed(before, after):
    old_before = before["slip"] + before["natural"] + before["impact"]
    old_after = after["slip"] + after["natural"] + after["impact"]
    return (old_before - old_after) >= 3 or (before["airborne"] - after["airborne"]) >= 4


def run_schedule(env, model, ref_model, table, base_q, q_ref, trajs, kl_coef, tag, payload):
    device = env.device
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    audits = []
    block = {"tag": tag, "kl_coef": kl_coef, "updates": [], "evals": []}
    payload["runs"].append(block)

    def checkpoint(update_index):
        model.eval()
        env.model = model
        summary, timing, closed = slim_eval(env, table, base_q, q_ref)
        drift = drift_report(model, ref_model, trajs, device)
        block["evals"].append({
            "update": update_index,
            "counts": summary,
            "timing": timing,
            "closed": closed,
            "drift": drift,
        })
        path = RAW / "sanity.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "update": update_index, "kl_coef": kl_coef}, RAW / f"policy_C_{tag}_u{update_index}.pt")
        print("EVAL", tag, update_index, summary, flush=True)
        model.train()
        return summary

    init_counts = checkpoint(0)
    for index in range(1, UPDATES + 1):
        episode_id = TRAIN_ORDER[(index - 1) % len(TRAIN_ORDER)]
        rng = np.random.default_rng(3000 + 19 * index + sum(ord(c) for c in episode_id))
        row, found = one_update(env, model, ref_model, opt, episode_id, kl_coef, rng, index)
        block["updates"].append(row)
        for audit in found:
            if len(audits) < 6:
                audits.append({"update": index, "id": episode_id, **audit})
        if index in EVAL_AT:
            summary = checkpoint(index)
            if index == 10 and kl_coef == KL_WEAK and collapsed(init_counts, summary):
                block["collapsed_at_10"] = True
                payload["audits"] = audits
                RAW.joinpath("sanity.json").write_text(json.dumps(payload), encoding="utf-8")
                return init_counts, summary, True
    payload["audits"] = payload.get("audits", []) + audits
    RAW.joinpath("sanity.json").write_text(json.dumps(payload), encoding="utf-8")
    final = block["evals"][-1]["counts"]
    return init_counts, final, False


def main():
    frozen = [RAW / "sanity.json", RAW / "policy_A_reference.pt", RAW / "policy_B_init.pt"]
    frozen.extend(RAW.glob("policy_C_*.pt"))
    present = [path for path in frozen if path.exists()]
    if present:
        raise SystemExit(
            "Refusing to overwrite the completed pre-GRPO sanity artifacts. "
            "Formal training is: python -u training/formal_grpo.py --config config/formal_grpo.yaml"
        )
    RAW.mkdir(parents=True, exist_ok=True)
    copy_a = RAW / "policy_A_reference.pt"
    copy_b = RAW / "policy_B_init.pt"
    hash_a = freeze_copy(REF_SRC, copy_a)
    hash_b = freeze_copy(INIT_SRC, copy_b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_a = torch.load(copy_a, map_location="cpu", weights_only=False)
    ref_model = build_policy(np.asarray(saved_a["proto"], np.float32), device, float(saved_a["res_cap"]))
    ref_model.load_state_dict(saved_a["model"])
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    saved_b = torch.load(copy_b, map_location="cpu", weights_only=False)
    proto = np.asarray(saved_b["proto"], np.float32)
    init_state = {k: v for k, v in saved_b["model"].items()}
    model = FuturePolicy(proto).to(device)
    model.load_state_dict(init_state)
    for parameter in model.fut.parameters():
        parameter.requires_grad_(False)
    env = make_training_env()
    env.device = device
    table = supports_of(env.sim)
    base_q, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    trajs = dataset()
    payload = {
        "reference": {"path": str(copy_a.relative_to(ROOT)).replace("\\", "/"), "sha256": hash_a, "source": str(REF_SRC.relative_to(ROOT)).replace("\\", "/")},
        "initialization": {
            "path": str(copy_b.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hash_b,
            "source": str(INIT_SRC.relative_to(ROOT)).replace("\\", "/"),
            "selection": "saved epoch-240 lambda 0.1 seed 2; union 11 with slip 2, natural-loss 2, airborne 7. Tied with lambda 0 seed 1, which had already lost natural-loss.",
        },
        "objective": {
            "grpo": "clipped group-relative advantage on the mean recovery-step log-probability; return is binary SUCCESS_10S/SUCCESS_12S",
            "kl": "categorical KL from the trainable policy to K=35 reference on non-airborne groups only",
            "bc_on_demonstrations": 0.0,
            "return": "1 if basin, explicit handoff, and nominal-only 10 s hold; else 0. Shaping is not added.",
            "critic": None,
            "lr": LR,
            "clip": CLIP,
            "sigma": SIGMA,
            "group": GROUP,
            "updates": UPDATES,
        },
        "runs": [],
        "audits": [],
    }
    print("FROZEN", hash_a[:12], hash_b[:12], flush=True)
    _init, _final, failed = run_schedule(env, model, ref_model, table, base_q, q_ref, trajs, KL_WEAK, "weak", payload)
    if failed:
        print("RESTART medium", flush=True)
        model.load_state_dict(init_state)
        for parameter in model.fut.parameters():
            parameter.requires_grad_(False)
        run_schedule(env, model, ref_model, table, base_q, q_ref, trajs, KL_MEDIUM, "medium", payload)
    print("WROTE", RAW / "sanity.json", flush=True)


if __name__ == "__main__":
    main()
