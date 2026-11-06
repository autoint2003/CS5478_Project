"""Drive StabilizeGrpoEnv with Expert continuous 7D actions (mining / feasibility only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p53_independent_grpo64.success_relaxed import evaluate_success_relaxed_12s
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p57_post_recatch_stabilization.reward_stabilize import (
    combined_step_reward,
    terminal_reward,
)
from models.m111_checkpoint import HANDOFF


def step_raw(env: StabilizeGrpoEnv, action: np.ndarray) -> tuple[float, bool, dict]:
    """Mirror StabilizeGrpoEnv.step but apply a continuous 7D Expert action."""
    assert env.start is not None and env.mon is not None and env._prev_m is not None
    m0 = env._prev_m
    a = np.asarray(action, np.float32).reshape(7)
    # Treat continuous expert step as non-token-switch for reward bookkeeping
    switched = False
    env.prev_tok = 0  # synthetic; not used as PPO sample

    step_legal(env.env.sim, a)
    env.env._prev = a.copy()
    env.t += 1

    m1 = state_metrics(env.env.sim)
    tr = env.recatch.update(env.t, m1)
    if tr["recatch"]:
        env.diag["recatch"] = 1
        env.diag["recatch_step"] = tr["recatch_step"]
        env.diag["post_recatch_steps"] = env.recatch.steps_since_recatch
    if tr["stable_hold"]:
        env.diag["stable_hold"] = 1
    if tr.get("lost_after_recatch_latched") or tr.get("lost_after_recatch"):
        env.diag["lost_after_recatch"] = 1

    r_step, bd, bds = combined_step_reward(
        m0,
        m1,
        token_switched=switched,
        post_recatch=bool(tr["recatch"]),
        just_lost_bilateral=bool(tr.get("lost_after_recatch")),
        base_weights=env.pose_weights,
        stab_weights=env.stab_weights,
        use_stabilization=env.use_stabilization,
    )
    for k in ("pose", "contact", "slip", "imbalance", "time", "switch"):
        env.comp_sums[k] += getattr(bd, k)
    for k in ("hold", "calm", "post_slip", "lost_contact", "overgrip"):
        env.stab_sums[k] += getattr(bds, k)
    env.stab_sums["stab_total"] += bds.total

    drop = bool(m1["latched"])
    hand = False
    hand_reason = None
    hand_diag: dict[str, Any] = {}
    st = env.mon.observe(env.env.sim, env.t, tape_ended=False)
    if st["ready"]:
        hand = True
        hand_reason = st["reason"]
        hand_diag = {
            "handoff_time": float(env.env.sim.data.time),
            "handoff_step": int(env.t),
            "e_pos_mm": float(m1["e_pos_mm"]),
            "e_theta_deg": float(m1["theta"]),
            "fn_L": float(m1.get("fn_L") or 0.0),
            "fn_R": float(m1.get("fn_R") or 0.0),
            "min_pad": int(m1.get("min_pad") or 0),
            "bilateral": int(m1.get("bilateral") or 0),
            "dwell_ready": bool(st.get("ready")),
            "r3f_reason": hand_reason,
        }

    truncated = env.t >= env.max_steps and not drop and not hand
    done = drop or hand or truncated

    r_term = 0.0
    s12 = snew = srel = False
    lost_after = False
    timeout_fail = False
    fail_reason = None
    ref = env.start.ref if env.start.kind == "capture" else (env.start.ep or {}).get("ref")
    if done:
        if drop:
            fail_reason = "DROP_BEFORE"
            r_term, bd_t = terminal_reward(
                drop=True,
                success_relaxed=False,
                success_12s=False,
                success_new=False,
                lost_after_hand=False,
                timeout_fail=False,
                weights=env.pose_weights,
            )
        elif truncated and not hand:
            timeout_fail = True
            fail_reason = "TIMEOUT"
            r_term, bd_t = terminal_reward(
                drop=False,
                success_relaxed=False,
                success_12s=False,
                success_new=False,
                lost_after_hand=False,
                timeout_fail=True,
                weights=env.pose_weights,
            )
        else:
            ev = evaluate_handoff_with_axis(env.env.sim, ref, stabilize_n=8)
            rel = evaluate_success_relaxed_12s(ev)
            s12 = bool(rel["SUCCESS_12S"])
            snew = bool(rel["SUCCESS_NEW"])
            srel = bool(rel["SUCCESS_RELAXED_12S"])
            reason = str(ev.get("reason") or "")
            lost_after = (not s12) and (
                "LOST" in reason.upper() or "DROP" in reason.upper() or "GROUND" in reason.upper()
            )
            timeout_fail = (not s12) and not lost_after
            fail_reason = None if srel else (reason or ("POSE_FAIL" if s12 else "NOMINAL_FAIL"))
            r_term, bd_t = terminal_reward(
                drop=False,
                success_relaxed=srel,
                success_12s=s12,
                success_new=snew,
                lost_after_hand=lost_after,
                timeout_fail=timeout_fail,
                weights=env.pose_weights,
            )
            hand_diag["nominal_reason"] = reason
            hand_diag["pre_pos_mm"] = ev.get("pre_pos_mm")
            hand_diag["pre_theta"] = ev.get("pre_theta")
        env.comp_sums["terminal"] += bd_t.terminal
        r_step += r_term

    if not done:
        obs = env._observe()
        env.obs_seq.append(obs)
        env.prev_seq.append(env.env._prev.copy())
    env._prev_m = m1

    info = {
        "skill": env.start.skill,
        "traj_id": env.start.traj_id,
        "metrics": m1,
        "drop": int(drop),
        "handoff": int(hand),
        "handoff_reason": hand_reason
        if hand
        else ("DROP_BEFORE" if drop else ("TIMEOUT" if truncated else None)),
        "handoff_diag": hand_diag,
        "truncated": int(truncated),
        "SUCCESS_12S": int(s12),
        "SUCCESS_NEW": int(snew),
        "SUCCESS_RELAXED_12S": int(srel),
        "fail_reason": fail_reason,
        "c_diag": dict(env.diag),
        "t": int(env.t),
        "expert_raw": 1,
    }
    return float(r_step), bool(done), info


def expert_actions(ep: dict) -> np.ndarray:
    acts = ep.get("actions")
    if acts is None:
        acts = ep.get("prefix_actions")
    return np.asarray(acts, np.float32).reshape(-1, 7)


def ensure_inner_capture_buffers(env: StabilizeGrpoEnv) -> None:
    """take_snapshot/take_capture require _obs_seq/_prev_seq on the inner harness env."""
    inner = env.env
    if not hasattr(inner, "_obs_seq") or inner._obs_seq is None:
        inner._obs_seq = []
    if not hasattr(inner, "_prev_seq") or inner._prev_seq is None:
        inner._prev_seq = []


# silence unused import warning for HANDOFF (kept for parity with env.step)
_ = HANDOFF
