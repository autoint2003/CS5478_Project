"""P110 repaired takeover: full wrapper restore + state-conditioned expert (no absolute tape index).

Reuses P108/P60 restore backbone and P55 Plan-C online search. Does not invent a new oracle.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import torch

from data_generation.oracles.plan_c.expert_search import run_stage1_catch_and_lift, search_catch
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers, step_raw
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p71_contact_safe_alignment.controllers import CtrlState
from experiments.precision.p73_first_contact_bilateral.controller import WrapState, select_token, v2_spec
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor
from models.m111_checkpoint import HANDOFF, decode_m111_action
from training.recovery_execution import make_snap

DT_POLICY = 0.02
PREFIX_DURATIONS_S = (0.2, 0.5, 1.0)


def export_wrap(st: WrapState) -> dict[str, Any]:
    v2 = st.v2
    return {
        "v2": {f.name: getattr(v2, f.name) for f in fields(CtrlState)},
        "post_i": int(st.post_i),
        "post_active": bool(st.post_active),
        "post_done": bool(st.post_done),
        "saw_contact": bool(st.saw_contact),
        "forced": list(st.forced),
        "stop_reason": st.stop_reason,
    }


def import_wrap(blob: dict[str, Any] | None) -> WrapState:
    st = WrapState()
    if not blob:
        return st
    v2b = blob.get("v2") or {}
    st.v2 = CtrlState(**{f.name: v2b.get(f.name, getattr(CtrlState(), f.name)) for f in fields(CtrlState)})
    st.post_i = int(blob.get("post_i") or 0)
    st.post_active = bool(blob.get("post_active"))
    st.post_done = bool(blob.get("post_done"))
    st.saw_contact = bool(blob.get("saw_contact"))
    st.forced = list(blob.get("forced") or [])
    st.stop_reason = blob.get("stop_reason")
    return st


def sync_wrapper_to_inner(env: StabilizeGrpoEnv) -> None:
    """Keep inner harness observation windows aligned with wrapper policy history."""
    ensure_inner_capture_buffers(env)
    env.env._obs_seq = [np.asarray(x, np.float32).copy() for x in env.obs_seq]
    env.env._prev_seq = [np.asarray(x, np.float32).copy() for x in env.prev_seq]
    if env.prev_seq:
        env.env._prev = np.asarray(env.prev_seq[-1], np.float32).copy()


def take_p110_state(env: StabilizeGrpoEnv, wrap: WrapState, *, meta: dict | None = None) -> dict:
    st = take_full_state(env, meta=meta)
    st["wrap"] = export_wrap(wrap)
    st["version"] = 2
    # contact bookkeeping from live metrics
    g = relative_geometry(env.env.sim)
    m = state_metrics(env.env.sim)
    st["contact"] = {
        "bilateral": int(bool(g.get("bilateral"))),
        "e_pos_mm": float(g.get("e_pos_mm") or 0.0),
        "vrel": float(m["vrel"]) if m.get("vrel") is not None else None,
        "fn_L": float(m.get("fn_L") or 0.0),
        "fn_R": float(m.get("fn_R") or 0.0),
    }
    return st


def restore_p110_state(env: StabilizeGrpoEnv, state: dict, start: MixedStart) -> WrapState:
    restore_full_state(env, state, start)
    sync_wrapper_to_inner(env)
    return import_wrap(state.get("wrap"))


def plan_c_expert_actions_from_state(env: StabilizeGrpoEnv, *, seed: int, max_rollouts: int = 12) -> dict:
    """State-conditioned Plan-C expert from current MuJoCo state (P55 pattern). No absolute tape index."""
    start = env.start
    assert start is not None
    ref = start.ref if start.kind == "capture" else (start.ep or {}).get("ref")
    snap = make_snap(env.env.sim)
    rng = np.random.default_rng(seed)
    try:
        search = search_catch(env.env.sim, snap, ref, rng=rng, max_rollouts=max_rollouts)
    except Exception as e:
        return {
            "accepted": False,
            "status": f"CATCH_EXCEPTION:{type(e).__name__}",
            "actions": np.zeros((0, 7), np.float32),
            "error": str(e)[:240],
        }
    best = search.get("best")
    if not best or not best.get("end_bilateral"):
        return {
            "accepted": False,
            "status": "CATCH_NO_BILATERAL",
            "actions": np.zeros((0, 7), np.float32),
            "funnel": search.get("funnel"),
        }
    stage = run_stage1_catch_and_lift(env.env.sim, snap, ref, best)
    acts = [np.asarray(a, np.float32).reshape(7) for a in (stage.get("stage1_actions") or [])]
    ok = bool(stage.get("ok")) and len(acts) > 0
    return {
        "accepted": ok,
        "status": "CATCH_STAGE1_OK" if ok else str(stage.get("reason") or "CATCH_STAGE1_FAIL"),
        "actions": np.stack(acts).astype(np.float32) if acts else np.zeros((0, 7), np.float32),
        "funnel": search.get("funnel"),
        "stage1_ok": bool(stage.get("ok")),
        "note": "state_conditioned_search_catch; not nominal expert_acts[t]",
    }


@torch.no_grad()
def rich_actor_rollout(actor, start: MixedStart, book, rho, device, *, seed: int, max_steps: int | None = None) -> dict:
    """Closed-loop P64/Actor roll with full-state snapshots at candidate intervention points."""
    from experiments.precision.final_independent_success_rate.paths import MAX_EP_STEPS, T_MAX

    seed_all(seed, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book,
        rho,
        max_steps=max_steps or MAX_EP_STEPS,
        t_max=T_MAX,
        weights=PoseRewardWeights(),
        use_stabilization=False,
    )
    try:
        env.reset(start)
    except Exception as e:
        return {"error": f"reset_failed:{type(e).__name__}:{e}"}
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    sync_wrapper_to_inner(env)
    wrap = WrapState()
    done = False
    info: dict = {}
    steps = []
    states_by_t: dict[int, dict] = {}
    bil_hist = []
    while not done:
        t = int(env.t)
        g0 = relative_geometry(env.env.sim)
        m0 = state_metrics(env.env.sim)
        # snapshot BEFORE action (continuation state)
        states_by_t[t] = take_p110_state(
            env,
            wrap,
            meta={"tag": "pre_action", "traj_id": start.traj_id},
        )
        obs_t = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev_t = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        logits, res, _ = actor(obs_t, prev_t)
        tok_a = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
        tok, wrap, dbg = select_token(
            env_t=t, obs=last_obs(env), actor_tok=tok_a, st=wrap, v2=v2_spec(), rule=None
        )
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        action = (
            np.zeros(7, np.float32)
            if int(tok) == HANDOFF
            else np.asarray(decode_m111_action(book, int(tok), resid, rho), np.float32)
        )
        bil_before = int(bool(g0["bilateral"]))
        steps.append(
            {
                "t": t,
                "sim_time": float(env.env.sim.data.time),
                "tok": int(tok),
                "tok_a": tok_a,
                "action": action,
                "obs": env.obs_seq[-1].astype(np.float32).copy(),
                "prev": env.prev_seq[-1].astype(np.float32).copy(),
                "bilateral_before": bil_before,
                "e_pos_mm": float(g0["e_pos_mm"]),
                "vrel": float(m0["vrel"]) if m0.get("vrel") is not None else None,
                "wrap": export_wrap(wrap),
            }
        )
        _r, done, info = env.step(int(tok), resid)
        sync_wrapper_to_inner(env)
        g1 = relative_geometry(env.env.sim)
        steps[-1]["bilateral_after"] = int(bool(g1["bilateral"]))
        steps[-1]["handoff"] = int(info.get("handoff") or 0)
        bil_hist.append(steps[-1]["bilateral_after"])
    term = _r3_terminal(env, info)
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "steps": steps,
        "states_by_t": states_by_t,
        "dt_policy": DT_POLICY,
    }


def select_intervention_times(roll: dict) -> list[tuple[int, str]]:
    """Predefined observable intervention criteria (not P108 early-E taxonomy)."""
    steps = roll.get("steps") or []
    out: list[tuple[int, str]] = []
    if not steps:
        return out
    # 1. Recovery entry
    out.append((0, "recovery_entry"))
    # 2. First substantial action jump while still progressing
    for i in range(1, len(steps)):
        jump = float(np.linalg.norm(steps[i]["action"] - steps[i - 1]["action"]))
        if jump > 0.5:
            out.append((i, "first_substantial_divergence"))
            break
    # 3. Contact-mode transition (0→1 or 1→0 after entry)
    for i in range(1, len(steps)):
        b0 = int(steps[i - 1]["bilateral_after"])
        b1 = int(steps[i]["bilateral_after"])
        if b0 != b1:
            out.append((i, "contact_mode_transition"))
            break
    # 4. Early regrasp instability: after first 0→1, then e_pos rises
    regrasp = None
    for i in range(1, len(steps)):
        if int(steps[i - 1]["bilateral_after"]) == 0 and int(steps[i]["bilateral_after"]) == 1:
            regrasp = i
            break
    if regrasp is not None:
        for i in range(regrasp + 1, min(regrasp + 15, len(steps))):
            if steps[i]["e_pos_mm"] > steps[regrasp]["e_pos_mm"] + 8.0:
                out.append((i, "early_regrasp_instability"))
                break
        # 5. Post-regrasp stabilization failure: lose bilateral after regrasp
        for i in range(regrasp + 1, len(steps)):
            if int(steps[i - 1]["bilateral_after"]) == 1 and int(steps[i]["bilateral_after"]) == 0:
                out.append((i, "post_regrasp_stabilization_failure"))
                break
    # de-dup nearby times (keep first tag per ~5-step cluster)
    kept: list[tuple[int, str]] = []
    for t, tag in out:
        if any(abs(t - kt) < 5 for kt, _ in kept):
            continue
        kept.append((t, tag))
    return kept[:4]


@torch.no_grad()
def continue_from_state_fixed(
    actor_or_none,
    start: MixedStart,
    state: dict,
    book,
    rho,
    device,
    *,
    mode: str,
    seed: int,
    prefix_s: float = 0.0,
    expert_seed: int | None = None,
    max_expert_rollouts: int = 12,
) -> dict:
    """Integrity-fixed CF continuation.

    modes:
      actor — continue Actor with restored WrapState
      expert — state-conditioned Plan-C online expert for remaining horizon
      prefix_expert — online expert for prefix_s seconds, then Actor
    """
    from experiments.precision.final_independent_success_rate.paths import MAX_EP_STEPS, T_MAX

    seed_all(seed, deterministic_torch=True)
    max_steps = int(state.get("max_steps") or MAX_EP_STEPS)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=max_steps, t_max=int(state.get("t_max") or T_MAX),
        weights=PoseRewardWeights(), use_stabilization=False,
    )
    env.reset(start)
    wrap = restore_p110_state(env, state, start)
    fidelity_ok = abs(float(env.env.sim.data.time) - float(state["sim_time"])) < 1e-6
    # multi-field fidelity
    g = relative_geometry(env.env.sim)
    contact0 = state.get("contact") or {}
    bil_err = abs(int(bool(g.get("bilateral"))) - int(contact0.get("bilateral") or 0))
    epos_err = abs(float(g.get("e_pos_mm") or 0.0) - float(contact0.get("e_pos_mm") or 0.0))

    expert_plan = None
    expert_acts = np.zeros((0, 7), np.float32)
    if mode in ("expert", "prefix_expert"):
        expert_plan = plan_c_expert_actions_from_state(
            env, seed=int(expert_seed if expert_seed is not None else seed), max_rollouts=max_expert_rollouts
        )
        # re-restore after search_catch mutated sim during planning
        wrap = restore_p110_state(env, state, start)
        _ea = expert_plan.get("actions")
        expert_acts = np.zeros((0, 7), np.float32) if _ea is None else np.asarray(_ea, np.float32).reshape(-1, 7)

    prefix_k = int(round(float(prefix_s) / DT_POLICY)) if mode == "prefix_expert" else 0
    done = False
    info: dict = {}
    t0 = int(env.t)
    expert_steps_used = 0
    expert_idx = 0
    executed_actions = []
    while not done and env.t < env.max_steps:
        use_expert = False
        if mode == "expert" and expert_idx < len(expert_acts):
            use_expert = True
        elif mode == "prefix_expert" and expert_steps_used < prefix_k and expert_idx < len(expert_acts):
            use_expert = True
        if use_expert:
            a = np.asarray(expert_acts[expert_idx], np.float32).reshape(7)
            _r, done, info = step_raw(env, a)
            sync_wrapper_to_inner(env)
            # keep wrap contact flag consistent for Actor resume
            feat_obs = last_obs(env)
            if feat_obs is not None:
                from experiments.precision.p70_geometry_to_r3f.obs_proxy import obs_features

                if obs_features(feat_obs)["any_contact_obs"]:
                    wrap.saw_contact = True
            expert_steps_used += 1
            expert_idx += 1
            executed_actions.append(a.copy())
        else:
            if actor_or_none is None:
                break
            obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
            prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
            logits, res, _ = actor_or_none(obs, prev)
            tok_a = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
            tok, wrap, _ = select_token(
                env_t=int(env.t), obs=last_obs(env), actor_tok=tok_a, st=wrap, v2=v2_spec(), rule=None
            )
            resid = res[-1].detach().cpu().numpy().astype(np.float32)
            action = (
                np.zeros(7, np.float32)
                if int(tok) == HANDOFF
                else np.asarray(decode_m111_action(book, int(tok), resid, rho), np.float32)
            )
            _r, done, info = env.step(int(tok), resid)
            sync_wrapper_to_inner(env)
            executed_actions.append(action)
        # if expert exhausted mid-episode under expert mode, stop (cannot invent actions)
        if mode == "expert" and expert_idx >= len(expert_acts) and not done:
            break
    return {
        "mode": mode,
        "prefix_s": float(prefix_s),
        "intervention_t": t0,
        "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "snapshot_fidelity_ok": int(fidelity_ok),
        "bilateral_restore_err": int(bil_err),
        "e_pos_restore_err_mm": float(epos_err),
        "expert_steps_used": expert_steps_used,
        "expert_plan_status": None if expert_plan is None else expert_plan.get("status"),
        "expert_plan_accepted": None if expert_plan is None else int(bool(expert_plan.get("accepted"))),
        "expert_offpolicy_caveat": 0,  # state-conditioned; not absolute tape
        "n_expert_actions_planned": int(len(expert_acts)),
        "executed_actions": np.stack(executed_actions).astype(np.float32) if executed_actions else np.zeros((0, 7), np.float32),
        "wrap_restored": 1,
        "note": "P110 fixed: WrapState restored; Plan-C online from restored physics; no expert_acts[t].",
    }


def multi_step_restore_determinism(env_factory, start, state, book, rho, n_steps: int = 5) -> dict:
    """Restore twice and step actor-free with identical zero actions; compare sim time + qpos."""
    diffs = []
    for trial in range(2):
        env = env_factory()
        env.reset(start)
        restore_p110_state(env, state, start)
        q0 = env.env.sim.data.qpos.copy()
        t0 = float(env.env.sim.data.time)
        for _ in range(n_steps):
            step_raw(env, np.zeros(7, np.float32))
        diffs.append((t0, q0, float(env.env.sim.data.time), env.env.sim.data.qpos.copy()))
    t_ok = abs(diffs[0][0] - diffs[1][0]) < 1e-9
    q_ok = np.allclose(diffs[0][1], diffs[1][1], atol=1e-8)
    end_t_ok = abs(diffs[0][2] - diffs[1][2]) < 1e-9
    end_q_ok = np.allclose(diffs[0][3], diffs[1][3], atol=1e-7)
    return {
        "restore_time_match": int(t_ok),
        "restore_qpos_match": int(q_ok),
        "after_n_steps_time_match": int(end_t_ok),
        "after_n_steps_qpos_match": int(end_q_ok),
        "n_steps": n_steps,
        "ok": int(t_ok and q_ok and end_t_ok and end_q_ok),
    }
