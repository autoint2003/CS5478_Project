"""P29 full-history rollout + collate (no fixed-K truncation)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import OBS_DIM_AIR, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402


def collate_full_history(samples: list[dict[str, Any]], device) -> tuple:
    """Pad variable-length full episodes. No WINDOW=160 cap."""
    if not samples:
        raise RuntimeError("empty collate")
    length = max(int(s["observations"].shape[0]) for s in samples)
    b = len(samples)
    obs = torch.zeros(b, length, OBS_DIM_AIR, device=device)
    prev = torch.zeros(b, length, 7, device=device)
    tokens = torch.zeros(b, length, dtype=torch.long, device=device)
    residuals = torch.zeros(b, length, 7, device=device)
    mask = torch.zeros(b, length, dtype=torch.bool, device=device)
    pad_mask = torch.ones(b, length, dtype=torch.bool, device=device)  # True = PAD
    weights = []
    for i, s in enumerate(samples):
        n = int(s["observations"].shape[0])
        obs[i, :n] = torch.as_tensor(s["observations"], device=device)
        prev[i, :n] = torch.as_tensor(s["prev_actions"], device=device)
        tokens[i, :n] = torch.as_tensor(s["tokens"], device=device)
        residuals[i, :n] = torch.as_tensor(s["residuals"], device=device)
        lm = np.asarray(s["loss_mask"], np.bool_)
        mask[i, :n] = torch.as_tensor(lm, device=device)
        pad_mask[i, :n] = False
        weights.append(float(s.get("sample_weight", 1.0)))
    return obs, prev, tokens, residuals, mask, pad_mask, torch.tensor(weights, device=device, dtype=torch.float32)


def policy_step_full(model, book, rho, obs_seq, prev_seq, device):
    """Full-history inference: feed entire available history, take last logit."""
    with torch.no_grad():
        logits, resid, _ = model(
            torch.as_tensor(np.stack(obs_seq), device=device),
            torch.as_tensor(np.stack(prev_seq), device=device),
        )
    tok = int(torch.argmax(logits[-1]).item())
    if tok == HANDOFF:
        return tok, np.zeros(7, np.float32), None, logits[-1].float().cpu().numpy()
    r = resid[-1].float().cpu().numpy().astype(np.float32)
    a = decode_m111_action(book, tok, r, rho=rho)
    return tok, r, a, logits[-1].float().cpu().numpy()


def policy_step_window(model, book, rho, obs_seq, prev_seq, device, K: int):
    """Legacy fixed-window infer (P28 reference)."""
    obs_in = obs_seq[-K:]
    prev_in = prev_seq[-K:]
    with torch.no_grad():
        logits, resid, _ = model(
            torch.as_tensor(np.stack(obs_in), device=device),
            torch.as_tensor(np.stack(prev_in), device=device),
        )
    tok = int(torch.argmax(logits[-1]).item())
    if tok == HANDOFF:
        return tok, np.zeros(7, np.float32), None, logits[-1].float().cpu().numpy()
    r = resid[-1].float().cpu().numpy().astype(np.float32)
    a = decode_m111_action(book, tok, r, rho=rho)
    return tok, r, a, logits[-1].float().cpu().numpy()


def rollout(
    env,
    model,
    book,
    rho,
    ep,
    *,
    device,
    mode: str = "full",
    K: int = 32,
    max_steps: int = 800,
    start_kind: str = "recovery",
) -> dict[str, Any]:
    if start_kind == "d2":
        restore_capture(env, ep["capture"])
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        ref = ep["ref"]
    else:
        restore_start(env, ep)
        ref = ep["ref"]

    hist = env._hist
    obs_seq: list = []
    prev_seq: list = []
    tokens: list[int] = []
    n_motion = 0
    termination_kind = "timeout"
    step_fn = policy_step_full if mode == "full" else lambda *a, **k: policy_step_window(*a, **k, K=K)

    for _ in range(max_steps):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        if mode == "full":
            tok, _r, action, _ = policy_step_full(model, book, rho, obs_seq, prev_seq, device)
        else:
            tok, _r, action, _ = policy_step_window(model, book, rho, obs_seq, prev_seq, device, K)
        tokens.append(tok)
        if action is None:
            termination_kind = "policy_handoff"
            break
        step_legal(env.sim, action)
        env._prev = action
        n_motion += 1
    else:
        termination_kind = "timeout"

    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    rls = []
    if tokens:
        cur, n = tokens[0], 1
        for t in tokens[1:]:
            if t == cur:
                n += 1
            else:
                rls.append(n)
                cur, n = t, 1
        rls.append(n)
    return {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "n_motion": n_motion,
        "termination_kind": termination_kind,
        "explicit_handoff": int(termination_kind == "policy_handoff"),
        "forced_timeout": int(termination_kind == "timeout"),
        "SUCCESS_NEW_policy_handoff": int(bool(hand.get("SUCCESS_NEW")) and termination_kind == "policy_handoff"),
        "immediate_handoff": int(n_motion == 0 and termination_kind == "policy_handoff"),
        "frac_token_3": sum(1 for t in tokens if t == 3) / max(1, len(tokens)),
        "max_token_run": max(rls) if rls else 0,
        "eval_reason": hand.get("std_reason") or hand.get("reason"),
        "history_len_at_end": len(obs_seq),
        "tokens_head": "|".join(str(t) for t in tokens[:12]),
    }


def full_history_parity(model, ep: dict, device, n_check: int = 64) -> list[dict]:
    """Compare (A) one full-episode forward logits[t] vs (B) prefix forward logits[-1].

    Causal attention must make these agree. Also verifies early tokens remain in the graph.
    """
    obs = ep["observations"]
    prev = ep["prev_actions"]
    n = int(obs.shape[0])
    n_use = min(n, max(n_check, 1))
    rows = []
    model.eval()
    with torch.no_grad():
        logits_full, resid_full, _ = model(
            torch.as_tensor(obs[:n_use], device=device),
            torch.as_tensor(prev[:n_use], device=device),
        )
    for t in range(n_use):
        with torch.no_grad():
            logits_p, resid_p, _ = model(
                torch.as_tensor(obs[: t + 1], device=device),
                torch.as_tensor(prev[: t + 1], device=device),
            )
        tok_b = int(torch.argmax(logits_full[t]).item())
        tok_s = int(torch.argmax(logits_p[-1]).item())
        logit_b = logits_full[t].float().cpu().numpy()
        logit_s = logits_p[-1].float().cpu().numpy()
        r_b = resid_full[t].float().cpu().numpy()
        r_s = resid_p[-1].float().cpu().numpy()
        rows.append(
            {
                "t": t,
                "tok_stream": tok_s,
                "tok_batch": tok_b,
                "tok_match": int(tok_s == tok_b),
                "max_logit_abs_diff": float(np.max(np.abs(logit_s - logit_b))),
                "resid_max_abs_diff": float(np.max(np.abs(r_s - r_b))),
            }
        )
        if tok_s != tok_b or float(np.max(np.abs(logit_s - logit_b))) > 1e-4:
            break
    return rows
