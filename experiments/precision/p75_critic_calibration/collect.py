"""Phase 1 — freeze Actor+V2; collect on-policy hybrid trajectories."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae, explained_variance
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p71_contact_safe_alignment.controllers import CtrlState, make_specs, select_token
from experiments.precision.p75_critic_calibration.paths import (
    C_FRACTION,
    DATA_DIR,
    GAMMA,
    GAE_LAMBDA,
    MAX_EP_STEPS,
    N_EVAL_FRAC_ICS,
    N_TARGET_EP,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    T_MAX,
)


def actor_param_digest(actor) -> str:
    h = hashlib.sha256()
    for p in actor.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def _stage_at(
    *,
    saw_fc: bool,
    ever_opp: bool,
    ever_frame: bool,
    ever_r3: bool,
    lost_after_fc: bool,
) -> str:
    if not saw_fc:
        return "pre_fc"
    if ever_r3:
        return "post_r3"
    if ever_frame:
        return "frame_to_r3"
    if ever_opp:
        return "opposing_to_frame"
    if lost_after_fc:
        return "fc_to_contact_loss"
    return "fc_to_opposing"


def rollout_episode(env, actor, critic, start, device, *, seed: int) -> dict:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    st = CtrlState()
    v2 = make_specs([])["V2"]
    done = False
    info: dict = {}

    obs_f, prev_f, rews, dones, stages, overrides, toks, vals, logps, logp_ok = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    saw_fc = ever_opp = ever_frame = ever_r3 = lost_after_fc = had_c = False

    while not done:
        stage = _stage_at(
            saw_fc=saw_fc,
            ever_opp=ever_opp,
            ever_frame=ever_frame,
            ever_r3=ever_r3,
            lost_after_fc=lost_after_fc,
        )
        # Snapshot history prefixes used by Critic/Actor at this decision
        obs_f.append(np.asarray(env.obs_seq[-1], np.float32).copy())
        prev_f.append(np.asarray(env.prev_seq[-1], np.float32).copy())

        tok_a, resid, logp, val = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, dbg = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, spec=v2
        )
        executed = int(tok)
        v2_ov = bool(dbg.get("forced") is not None) or (executed != int(tok_a))

        r, done, info = env.step(executed, resid)
        rews.append(float(r))
        dones.append(1.0 if done else 0.0)
        stages.append(stage)
        overrides.append(int(v2_ov))
        toks.append(executed)
        vals.append(float(val))
        logps.append(float(logp) if not v2_ov else 0.0)
        logp_ok.append(0 if v2_ov else 1)

        g = relative_geometry(env.env.sim)
        if g["has_pad_contact"]:
            if not saw_fc:
                saw_fc = True
            had_c = True
        elif saw_fc and had_c:
            lost_after_fc = True
        if g["bilateral"] and g["correct_sidewall"]:
            ever_opp = True
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if preds["frame_ok"]:
            ever_frame = True
        if (preds["dwell_complete"] and preds["frame_ok"]) or int(info.get("handoff") or 0):
            ever_r3 = True

    rewards = np.asarray(rews, np.float32)
    values = np.asarray(list(vals) + [0.0], np.float32)
    done_a = np.asarray(dones, np.float32)
    # λ=1, γ=1 → ret == MC return for true-terminal trajectories
    adv, ret = compute_gae(rewards, values, done_a, GAMMA, 1.0)
    # Also store GAE(λ=0.95) advantages under baseline values for Phase 5 baseline
    adv095, _ = compute_gae(rewards, values, done_a, GAMMA, GAE_LAMBDA)

    return {
        "skill": start.skill,
        "traj_id": start.traj_id,
        "seed": int(seed),
        "obs_frames": np.stack(obs_f).astype(np.float32),
        "prev_frames": np.stack(prev_f).astype(np.float32),
        "rewards": rewards,
        "dones": done_a,
        "stages": stages,
        "v2_override": np.asarray(overrides, np.int8),
        "tokens": np.asarray(toks, np.int64),
        "actor_logp": np.asarray(logps, np.float32),
        "behavior_logp_valid": np.asarray(logp_ok, np.int8),
        "baseline_values": np.asarray(vals, np.float32),
        "mc_returns": ret.astype(np.float32),
        "baseline_adv_gae095": adv095.astype(np.float32),
        "ep_return": float(rewards.sum()),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "r3_ready": int(bool(ever_r3 or int(info.get("handoff") or 0))),
        "reached_opposing": int(ever_opp),
        "lost_contact_after_fc": int(lost_after_fc),
        "timeout": int(
            bool(info.get("truncated") or 0)
            or str(info.get("fail_reason") or "") == "TIMEOUT"
            or str(info.get("handoff_reason") or "") == "TIMEOUT"
        ),
        "n_steps": len(rews),
    }


def _pick_ics(schedule, *, n_ep: int, c_frac: float, seed: int):
    blocked = blocked_eval_ids()
    by = {"A1": [], "B1": [], "C": [], "D": []}
    for s in schedule:
        if s.traj_id in blocked:
            continue
        if s.skill in by:
            by[s.skill].append(s)
    rng = np.random.default_rng(seed)
    n_c = int(round(n_ep * c_frac))
    n_rest = n_ep - n_c
    # Allocate ICs: ~16 rollouts/IC average
    rolls = 16
    n_c_ics = max(8, n_c // rolls)
    n_each_abd = max(4, (n_rest // 3) // rolls)

    def sample(pool, k):
        k = min(k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[int(i)] for i in idx]

    c_ics = sample(by["C"], n_c_ics)
    a_ics = sample(by["A1"], n_each_abd)
    b_ics = sample(by["B1"], n_each_abd)
    d_ics = sample(by["D"], n_each_abd)
    # IC-disjoint eval split
    def split(ics):
        rng.shuffle(ics)
        n_ev = max(1, int(round(len(ics) * N_EVAL_FRAC_ICS))) if len(ics) > 3 else max(1, len(ics) // 4)
        return ics[n_ev:], ics[:n_ev]

    plans = []
    for skill_ics, skill in ((c_ics, "C"), (a_ics, "A1"), (b_ics, "B1"), (d_ics, "D")):
        tr, ev = split(list(skill_ics))
        for st in tr:
            plans.append(("train", st))
        for st in ev:
            plans.append(("eval", st))
    return plans, rolls


def collect_dataset(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()
    digest0 = actor_param_digest(actor)

    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    schedule = build_p54_dataset(seed=SEED)["schedule"]
    plans, rolls = _pick_ics(schedule, n_ep=N_TARGET_EP, c_frac=C_FRACTION, seed=SEED + 75)

    train_eps, eval_eps = [], []
    ep_i = 0
    for split, st in plans:
        for j in range(rolls):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=75000 + ep_i)
            ep = rollout_episode(env, actor, critic, st, device_t, seed=seed)
            ep["split"] = split
            (train_eps if split == "train" else eval_eps).append(ep)
            ep_i += 1
            if ep_i % 64 == 0 or ep_i == 1:
                print(
                    f"[collect] {ep_i} {split} {st.skill} {st.traj_id[-18:]} "
                    f"ret={ep['ep_return']:.2f} opp={ep['reached_opposing']} r3={ep['r3_ready']}",
                    flush=True,
                )

    digest1 = actor_param_digest(actor)
    assert digest0 == digest1, "Actor weights changed during collection"

    meta = {
        "n_train": len(train_eps),
        "n_eval": len(eval_eps),
        "n_total": len(train_eps) + len(eval_eps),
        "rolls_per_ic": rolls,
        "actor_digest": digest0,
        "gamma": GAMMA,
        "reward": "P57-A PoseRewardWeights, use_stabilization=False",
        "policy": "P70 V2 + frozen P64 Actor",
        "train_ics": sorted({e["traj_id"] for e in train_eps}),
        "eval_ics": sorted({e["traj_id"] for e in eval_eps}),
        "ic_overlap": sorted(set(e["traj_id"] for e in train_eps) & set(e["traj_id"] for e in eval_eps)),
    }
    assert not meta["ic_overlap"], "Train/eval IC overlap"

    path_tr = DATA_DIR / "train_episodes.pkl"
    path_ev = DATA_DIR / "eval_episodes.pkl"
    path_tr.write_bytes(pickle.dumps(train_eps, protocol=pickle.HIGHEST_PROTOCOL))
    path_ev.write_bytes(pickle.dumps(eval_eps, protocol=pickle.HIGHEST_PROTOCOL))
    (OUT / "COLLECTION_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


if __name__ == "__main__":
    collect_dataset()
