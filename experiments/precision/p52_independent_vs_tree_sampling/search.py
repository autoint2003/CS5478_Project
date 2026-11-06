"""M111+R3F independent sampling and Tree80-style branching (frozen P47T)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch

from envs.observations.airborne_obs import observe_airborne
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline
from experiments.precision.p52_independent_vs_tree_sampling.branch_state import (
    restore_branch_state,
    take_branch_state,
)
from experiments.precision.p52_independent_vs_tree_sampling.paths import (
    CAPTURE_STRIDE,
    MAX_CAPTURES_PER_TRUNK,
    N_CANDIDATES,
    N_ENTROPY_PER_TRUNK,
    N_TRUNKS,
    N_UNIFORM_PER_TRUNK,
    T_MAX,
)
from models.m111_checkpoint import HANDOFF, MOTION_K, decode_m111_action
from models.transformer_policy import DT_POLICY
from training.adaptive_hybrid import select_trunk_branches
from training.m111_dataset import WINDOW


def token_entropy(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, np.float64), 0.0, None)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / s
    pos = p[p > 0]
    return float(-(pos * np.log(pos)).sum())


def sample_m111_token(model, obs_seq, prev_seq, device, rng) -> tuple[int, np.ndarray, float]:
    """Stochastic motion-token sample; HANDOFF masked (R3F decides handoff)."""
    obs = torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device, dtype=torch.float32)
    with torch.no_grad():
        logits, resid, _ = model(obs, prev)
        lg = logits[-1].float().clone()
        lg[HANDOFF] = -1e9
        probs = torch.softmax(lg, dim=-1).cpu().numpy()
    probs = np.clip(probs, 0.0, None)
    probs = probs / max(float(probs.sum()), 1e-12)
    tok = int(rng.choice(len(probs), p=probs))
    if tok == HANDOFF or tok < 0 or tok >= MOTION_K:
        tok = int(np.argmax(probs[:MOTION_K]))
    resid_np = resid[-1].float().cpu().numpy().astype(np.float32)
    return tok, resid_np, token_entropy(probs)


def _observe(env) -> np.ndarray:
    return np.asarray(
        observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
        np.float32,
    )


def reset_ic(env, job: dict) -> Any:
    skill = job["skill"]
    if skill in ("A1", "B1", "C"):
        restore_start(env, job["ep"])
        env._obs_seq = []
        env._prev_seq = []
        return job["ep"]["ref"]
    restore_capture(env, job["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return job["ref"]


@dataclass
class CostAcc:
    sim_steps: int = 0
    policy_calls: int = 0
    snapshots: int = 0
    restores: int = 0
    nominal_evals: int = 0
    wall_s: float = 0.0


@dataclass
class CandidateResult:
    method: str
    origin: str
    cand_index: int
    ic_id: str
    skill: str
    search_seed: int
    SUCCESS_12S: int
    SUCCESS_NEW: int
    fail_category: str
    handoff_reason: str
    n_policy_steps: int
    tokens: list[int] = field(default_factory=list)
    parent_id: int | None = None
    branch_step: int | None = None
    branch_strategy: str | None = None
    prefix_len: int = 0
    sim_steps: int = 0
    policy_calls: int = 0
    snapshots: int = 0
    restores: int = 0
    nominal_evals: int = 0
    wall_s: float = 0.0

    def to_row(self) -> dict:
        d = asdict(self)
        d["token_seq"] = " ".join(str(t) for t in self.tokens[:64])
        d["n_tokens"] = len(self.tokens)
        return d


def _finalize(env, ref, *, drop: bool, handoff: bool, hand_reason: str, costs: CostAcc) -> dict:
    if drop:
        return {
            "SUCCESS_12S": 0,
            "SUCCESS_NEW": 0,
            "fail_category": "DROP_BEFORE",
            "handoff_reason": "DROP_BEFORE",
            "handoff": 0,
        }
    costs.nominal_evals += 1
    if not handoff:
        hand_reason = "TIMEOUT_FALLBACK"
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    s12 = int(bool(hand.get("SUCCESS_12S")))
    snew = int(bool(hand.get("SUCCESS_NEW")))
    if snew:
        fail = "SUCCESS"
    elif s12:
        fail = "HOLD_FAIL_12S_OK"
    else:
        fail = f"POST_HANDOFF:{hand.get('reason')}"
    return {
        "SUCCESS_12S": s12,
        "SUCCESS_NEW": snew,
        "fail_category": fail,
        "handoff_reason": hand_reason,
        "handoff": 1,
    }


def rollout_independent(
    env,
    model,
    book,
    rho,
    job: dict,
    device,
    rng,
    *,
    keep_captures: bool = False,
    max_steps: int = T_MAX,
) -> dict:
    t0 = time.perf_counter()
    costs = CostAcc()
    ref = reset_ic(env, job)
    mon = R3FOnline.create(t_max=max_steps)

    tokens: list[int] = []
    captures: dict[int, dict] = {}
    steps_meta: list[dict] = []
    n_policy = 0
    drop = False
    handoff = False
    hand_reason = "NONE"
    step = 0

    while True:
        if latched(env.sim):
            drop = True
            break
        st = mon.observe(env.sim, step, tape_ended=False)
        if st["ready"]:
            handoff = True
            hand_reason = st["reason"]
            break
        if step >= max_steps:
            break

        # Sparse pre-action captures (full MjData); stride avoids OOM at T_max=3100
        do_cap = (
            keep_captures
            and (n_policy % CAPTURE_STRIDE == 0)
            and (len(captures) < MAX_CAPTURES_PER_TRUNK)
        )
        if do_cap:
            costs.snapshots += 1
            pre_bp = take_branch_state(env, mon, step=n_policy, entropy=0.0, token=-1)

        obs = _observe(env)
        env._obs_seq.append(obs)
        env._prev_seq.append(np.asarray(env._prev, np.float32).copy())
        tok, resid, ent = sample_m111_token(model, env._obs_seq, env._prev_seq, device, rng)
        costs.policy_calls += 1

        if do_cap:
            pre_bp["entropy"] = float(ent)
            pre_bp["token"] = int(tok)
            captures[n_policy] = pre_bp

        steps_meta.append(
            {
                "step": n_policy,
                "entropy": float(ent),
                "token": int(tok),
                "lost_low": 0,
                "t": step * float(DT_POLICY),
            }
        )
        tokens.append(tok)
        action = decode_m111_action(book, tok, resid, rho=rho)
        step_legal(env.sim, np.asarray(action, np.float32))
        env._prev = np.asarray(action, np.float32).reshape(7)
        costs.sim_steps += 1
        n_policy += 1
        step += 1

    costs.wall_s = time.perf_counter() - t0
    fin = _finalize(env, ref, drop=drop, handoff=handoff, hand_reason=hand_reason, costs=costs)
    return {
        **fin,
        "tokens": tokens,
        "n_policy_steps": n_policy,
        "captures": captures,
        "steps": steps_meta,
        "costs": costs,
        "ref": ref,
    }


def continue_from_branch(
    env,
    model,
    book,
    rho,
    ref,
    device,
    rng,
    branch_state: dict,
    *,
    max_steps: int = T_MAX,
) -> dict:
    """Restore pre-action branch point and sample a new suffix."""
    t0 = time.perf_counter()
    costs = CostAcc(restores=1)
    mon = restore_branch_state(env, branch_state)
    start_policy_step = int(branch_state["step"])
    step = start_policy_step
    tokens: list[int] = []
    n_suffix = 0
    drop = False
    handoff = False
    hand_reason = "NONE"

    while True:
        if latched(env.sim):
            drop = True
            break
        st = mon.observe(env.sim, step, tape_ended=False)
        if st["ready"]:
            handoff = True
            hand_reason = st["reason"]
            break
        if step >= max_steps:
            break

        obs = _observe(env)
        env._obs_seq.append(obs)
        env._prev_seq.append(np.asarray(env._prev, np.float32).copy())
        tok, resid, _ent = sample_m111_token(model, env._obs_seq, env._prev_seq, device, rng)
        costs.policy_calls += 1
        tokens.append(tok)
        action = decode_m111_action(book, tok, resid, rho=rho)
        step_legal(env.sim, np.asarray(action, np.float32))
        env._prev = np.asarray(action, np.float32).reshape(7)
        costs.sim_steps += 1
        n_suffix += 1
        step += 1

    costs.wall_s = time.perf_counter() - t0
    fin = _finalize(env, ref, drop=drop, handoff=handoff, hand_reason=hand_reason, costs=costs)
    return {
        **fin,
        "tokens": tokens,
        "n_policy_steps": start_policy_step + n_suffix,
        "suffix_steps": n_suffix,
        "prefix_len": start_policy_step,
        "costs": costs,
    }


def allocate_branches_for_trunks(trunks: list[dict], seed: int) -> list[dict]:
    points = []
    rng_u = np.random.default_rng(int(seed) + 300)
    rng_e = np.random.default_rng(int(seed) + 400)
    for ti, trunk in enumerate(trunks):
        cands = []
        for step_row in trunk.get("steps") or []:
            idx = int(step_row["step"])
            if idx not in (trunk.get("captures") or {}):
                continue
            cands.append(
                {
                    "traj": ti,
                    "step": idx,
                    "entropy": float(step_row["entropy"]),
                    "lost_low": 0,
                    "t": float(step_row["t"]),
                    "token": int(step_row["token"]),
                }
            )
        if not cands:
            continue
        for p in select_trunk_branches(cands, rng_u, N_UNIFORM_PER_TRUNK, "uniform"):
            points.append({**p, "strategy": "uniform"})
        for p in select_trunk_branches(cands, rng_e, N_ENTROPY_PER_TRUNK, "entropy"):
            points.append({**p, "strategy": "entropy"})
    return points


def _to_cand(raw: dict, *, method: str, origin: str, index: int, job: dict, search_seed: int, **extra) -> CandidateResult:
    c: CostAcc = raw["costs"]
    return CandidateResult(
        method=method,
        origin=origin,
        cand_index=index,
        ic_id=job["traj_id"],
        skill=job["skill"],
        search_seed=search_seed,
        SUCCESS_12S=int(raw["SUCCESS_12S"]),
        SUCCESS_NEW=int(raw["SUCCESS_NEW"]),
        fail_category=str(raw["fail_category"]),
        handoff_reason=str(raw["handoff_reason"]),
        n_policy_steps=int(raw["n_policy_steps"]),
        tokens=list(raw["tokens"]),
        sim_steps=c.sim_steps,
        policy_calls=c.policy_calls,
        snapshots=c.snapshots,
        restores=c.restores,
        nominal_evals=c.nominal_evals,
        wall_s=c.wall_s,
        **extra,
    )


def search_independent(env, model, book, rho, job, device, *, search_seed: int, n: int = N_CANDIDATES) -> list[CandidateResult]:
    out = []
    for i in range(n):
        rng = np.random.default_rng(int(search_seed) + 17 * (i + 1))
        raw = rollout_independent(env, model, book, rho, job, device, rng, keep_captures=False)
        out.append(_to_cand(raw, method="independent", origin="independent", index=i, job=job, search_seed=search_seed))
    return out


def search_tree(env, model, book, rho, job, device, *, search_seed: int, n: int = N_CANDIDATES) -> list[CandidateResult]:
    """Tree80-adapted: N_TRUNKS trunks + uniform/entropy branches → n terminals."""
    results: list[CandidateResult] = []
    trunks: list[dict] = []
    n_trunks = min(N_TRUNKS, n)
    for i in range(n_trunks):
        rng = np.random.default_rng(int(search_seed) + 17 * (i + 1))
        raw = rollout_independent(env, model, book, rho, job, device, rng, keep_captures=True)
        results.append(
            _to_cand(raw, method="tree", origin="trunk", index=len(results), job=job, search_seed=search_seed)
        )
        trunks.append(raw)

    points = allocate_branches_for_trunks(trunks, search_seed)
    need = n - len(results)
    # Retain only captures required by allocated branches (free MjData ASAP)
    needed: dict[int, set[int]] = {}
    for point in points[:need]:
        needed.setdefault(int(point["traj"]), set()).add(int(point["step"]))
    for ti, trunk in enumerate(trunks):
        keep = needed.get(ti, set())
        trunk["captures"] = {k: v for k, v in trunk["captures"].items() if k in keep}

    for bi, point in enumerate(points[:need]):
        trunk = trunks[int(point["traj"])]
        cap = trunk["captures"].get(int(point["step"]))
        if cap is None:
            continue
        rng = np.random.default_rng(int(search_seed) + 1000 + 31 * (bi + 1))
        raw = continue_from_branch(env, model, book, rho, trunk["ref"], device, rng, cap)
        # Drop used capture to free MjData
        trunk["captures"].pop(int(point["step"]), None)
        prefix_toks = trunk["tokens"][: int(point["step"])]
        raw = dict(raw)
        raw["tokens"] = prefix_toks + list(raw["tokens"])
        results.append(
            _to_cand(
                raw,
                method="tree",
                origin="tree",
                index=len(results),
                job=job,
                search_seed=search_seed,
                parent_id=int(point["traj"]),
                branch_step=int(point["step"]),
                branch_strategy=str(point.get("strategy")),
                prefix_len=int(point["step"]),
            )
        )
        if len(results) >= n:
            break

    # Release remaining trunk captures
    for trunk in trunks:
        trunk["captures"] = {}

    while len(results) < n:
        i = len(results)
        rng = np.random.default_rng(int(search_seed) + 5000 + 17 * (i + 1))
        raw = rollout_independent(env, model, book, rho, job, device, rng, keep_captures=False)
        results.append(
            _to_cand(raw, method="tree", origin="independent_pad", index=i, job=job, search_seed=search_seed)
        )
    return results[:n]
