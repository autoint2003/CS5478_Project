"""Trace M111 token+residual action semantics and KL coverage (no training)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.policy_sample import (
    categorical_kl_masked,
    sample_step,
    token_logprob,
)
from experiments.precision.p61_reproducible_joint_eval.eval_core import load_actor, make_eval_env
from experiments.precision.p61_reproducible_joint_eval.paths import OUT, P57A_CKPT, P59_PASS1
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p53_independent_grpo64.probabilistic_eval import load_heldout_starts
from models.m111_checkpoint import HANDOFF, decode_m111_action


def run_residual_kl_audit(*, device: str = "cpu") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    starts = load_heldout_starts(per_class=1)
    st = starts["C"][0]
    actor, book, rho, saved = load_actor(P59_PASS1, device_t)
    ref, _, _, _ = load_actor(P57A_CKPT, device_t)  # illustrative frozen-ref style
    env = make_eval_env(book, rho)

    seed = rollout_seed(split="heldout", traj_id=st.traj_id, rollout_i=0)
    seed_all(seed)
    env.reset(st)
    tok, resid, logp_beh, meta = sample_step(
        actor, env.obs_seq, env.prev_seq, device_t, temperature=1.0
    )
    # Recompute token logp
    obs_w = np.stack(env.obs_seq[-160:]).astype(np.float32)
    prev_w = np.stack(env.prev_seq[-160:]).astype(np.float32)
    logp_re = float(token_logprob(actor, obs_w, prev_w, tok, device_t).detach().cpu())
    # Second sample_step without reseeding advances RNG → different token possible
    # Residual: call actor twice with same seed → same resid
    seed_all(seed)
    env.reset(st)
    with torch.no_grad():
        o = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device_t, dtype=torch.float32)
        p = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device_t, dtype=torch.float32)
        _logits, res1, _ = actor(o, p)
        _logits, res2, _ = actor(o, p)
    resid_det = bool(np.allclose(res1[-1].cpu().numpy(), res2[-1].cpu().numpy()))

    action = decode_m111_action(book, tok, resid, rho=rho)
    findings = {
        "token": {
            "HANDOFF_index": int(HANDOFF),
            "sampled_token": int(tok),
            "stochastic": True,
            "logp_behavior": float(logp_beh),
            "logp_recomputed": logp_re,
            "logp_match": abs(float(logp_beh) - logp_re) < 1e-5,
            "note": "Categorical over M111 motion tokens; HANDOFF logit masked (−1e9).",
        },
        "residual": {
            "dim": 7,
            "deterministic_given_obs": resid_det,
            "stochastic_sampling": False,
            "in_grpo_likelihood": False,
            "trainable_head_exists": True,
            "note": (
                "Residual is the network mean (no N(μ,σ²) sample). "
                "GRPO ratio uses token logp only; gaussian_logprob unused in P53–P60."
            ),
            "executed_action": "decode_m111_action(book, token, residual, rho) with residual clipped to budget",
            "sample_resid_l2": float(np.linalg.norm(resid)),
            "action_l2": float(np.linalg.norm(action)),
        },
        "kl": {
            "reference": "Frozen copy of the *training init* checkpoint (not behavior buffer).",
            "estimator": "categorical_kl_masked on a SINGLE first record of the first nonempty member",
            "states_covered": "One (obs,prev) window per optimizer step — not mean over group/IC/class",
            "gradients": "Flow through trainable actor only (ref under no_grad)",
            "c_only_implication": (
                "During P58/P59 C-only GRPO, KL is computed on C continuation states present in "
                "the current group. A/B/D states are not regularly visited, so KL does not "
                "directly regularize A/B/D behavior; drift on A/B/D is possible and was observed "
                "mildly in P59 ABD validation."
            ),
        },
        "ckpt_meta": {
            "eval_actor": str(P59_PASS1),
            "illustrative_ref": str(P57A_CKPT),
            "saved_keys_sample": sorted(list(saved.keys()))[:20],
        },
    }

    # Tiny KL numeric probe
    kl = float(categorical_kl_masked(actor, ref, obs_w, prev_w, device_t).detach().cpu())
    findings["kl"]["example_kl_p59_vs_p57a_one_state"] = kl

    (OUT / "RESIDUAL_KL_AUDIT.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    lines = [
        "# RESIDUAL_KL_AUDIT",
        "",
        "Inspection only — no training changes.",
        "",
        "## Token + residual → action",
        "",
        "1. Policy outputs logits over 112 classes and a 7-D residual mean.",
        "2. HANDOFF logit is masked; sample categorical token (stochastic at T=1).",
        "3. Residual is **deterministic** given observations (no Gaussian noise).",
        "4. `decode_m111_action(book, token, residual, rho)` produces the 7-D executed action.",
        "",
        f"- Sampled token logp match recompute: **{findings['token']['logp_match']}**",
        f"- Residual identical on repeated forward: **{resid_det}**",
        "",
        "## What GRPO optimizes",
        "",
        "- Stored `logp_behavior` = **token** log-probability only.",
        "- Importance ratio uses length-normalized mean token logp.",
        "- Residual head receives **no GRPO likelihood term** (may still move via shared trunk if any).",
        "",
        "## Reference KL",
        "",
        findings["kl"]["reference"],
        "",
        findings["kl"]["estimator"],
        "",
        findings["kl"]["c_only_implication"],
        "",
        f"Example KL(P59_pass1 ∥ P57_A) on one C state: **{kl:.6f}**",
        "",
    ]
    (OUT / "RESIDUAL_KL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return findings


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(json.dumps(run_residual_kl_audit(device=args.device), indent=2))
