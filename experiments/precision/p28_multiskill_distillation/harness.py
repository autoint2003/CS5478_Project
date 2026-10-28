"""Shared corrected rollout harness for P28 (timeout ≠ policy HANDOFF)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402


def policy_step_K(model, book, rho, obs_seq, prev_seq, device, K: int):
    obs_in = obs_seq[-K:]
    prev_in = prev_seq[-K:]
    with torch.no_grad():
        logits, resid, _ = model(
            torch.as_tensor(np.stack(obs_in), device=device),
            torch.as_tensor(np.stack(prev_in), device=device),
        )
    tok = int(torch.argmax(logits[-1]).item())
    if tok == HANDOFF:
        return tok, np.zeros(7, np.float32), None, logits[-1].detach().float().cpu().numpy()
    r = resid[-1].float().cpu().numpy().astype(np.float32)
    a = decode_m111_action(book, tok, r, rho=rho)
    return tok, r, a, logits[-1].detach().float().cpu().numpy()


def rollout_corrected(
    env,
    model,
    book,
    rho,
    ep,
    *,
    device,
    K: int,
    max_steps: int = 800,
    start_kind: str = "recovery",
    record_traj: bool = False,
) -> dict[str, Any]:
    """Closed-loop rollout with explicit termination taxonomy.

    termination_kind ∈ {policy_handoff, timeout, early_break}.
    Physical STRICT is always computed at end, but timeout is never labeled as
    a policy-generated HANDOFF.
    """
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
    residuals: list = []
    actions: list = []
    logit_rows: list = []
    n_motion = 0
    termination_kind = "timeout"

    for _ in range(max_steps):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, r, action, logit = policy_step_K(model, book, rho, obs_seq, prev_seq, device, K)
        tokens.append(tok)
        residuals.append(r)
        logit_rows.append(logit)
        if action is None:
            termination_kind = "policy_handoff"
            break
        step_legal(env.sim, action)
        env._prev = action
        actions.append(action)
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

    out: dict[str, Any] = {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "n_motion": n_motion,
        "termination_kind": termination_kind,
        "explicit_handoff": int(termination_kind == "policy_handoff"),
        "forced_timeout": int(termination_kind == "timeout"),
        # Never treat timeout as policy HANDOFF success credit:
        "SUCCESS_NEW_policy_handoff": int(
            bool(hand.get("SUCCESS_NEW")) and termination_kind == "policy_handoff"
        ),
        "immediate_handoff": int(n_motion == 0 and termination_kind == "policy_handoff"),
        "frac_token_3": sum(1 for t in tokens if t == 3) / max(1, len(tokens)),
        "max_token_run": max(rls) if rls else 0,
        "eval_reason": hand.get("std_reason") or hand.get("reason"),
        "tokens_head": "|".join(str(t) for t in tokens[:12]),
    }
    if record_traj:
        out["traj"] = {
            "observations": np.stack(obs_seq).astype(np.float32),
            "prev_actions": np.stack(prev_seq).astype(np.float32),
            "tokens": np.asarray(tokens, np.int64),
            "residuals": np.stack(residuals).astype(np.float32) if residuals else np.zeros((0, 7), np.float32),
            "actions": np.stack(actions).astype(np.float32) if actions else np.zeros((0, 7), np.float32),
            "logits": np.stack(logit_rows).astype(np.float32) if logit_rows else np.zeros((0, 112), np.float32),
        }
    return out


def offline_vs_stream_parity(
    model,
    book,
    rho,
    ep_npz: dict,
    *,
    device,
    K: int,
    n_check: int = 64,
) -> list[dict]:
    """Compare batched offline forward vs streaming last-K inference on expert tape."""
    obs = ep_npz["observations"]
    prev = ep_npz["prev_actions"]
    n = int(obs.shape[0])
    rows = []
    model.eval()
    for t in range(min(n, n_check)):
        # Streaming: feed history [0..t] truncated to K, take last logit
        o_seq = [obs[i] for i in range(t + 1)]
        p_seq = [prev[i] for i in range(t + 1)]
        tok_s, r_s, a_s, logit_s = policy_step_K(model, book, rho, o_seq, p_seq, device, K)

        # Offline window ending at t (same truncation)
        hs = max(0, t + 1 - K)
        o_win = torch.as_tensor(obs[hs : t + 1], device=device)
        p_win = torch.as_tensor(prev[hs : t + 1], device=device)
        with torch.no_grad():
            logits_b, resid_b, _ = model(o_win, p_win)
        tok_b = int(torch.argmax(logits_b[-1]).item())
        logit_b = logits_b[-1].float().cpu().numpy()
        r_b = resid_b[-1].float().cpu().numpy()
        max_logit_diff = float(np.max(np.abs(logit_s - logit_b)))
        rows.append(
            {
                "t": t,
                "tok_stream": tok_s,
                "tok_offline": tok_b,
                "tok_match": int(tok_s == tok_b),
                "max_logit_abs_diff": max_logit_diff,
                "resid_max_abs_diff": float(np.max(np.abs(r_s - r_b))),
                "expert_token": int(ep_npz["tokens"][t]),
            }
        )
        if tok_s != tok_b or max_logit_diff > 1e-4:
            break
    return rows
