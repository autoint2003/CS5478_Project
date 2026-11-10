"""Collect bounded baseline-only A/B/C/D episodes with failure instrumentation."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import (
    blocked_eval_ids,
    is_blocked_eval,
    sample_stratified,
)
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p69_skill_sequence.skill_families import family_of_token
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p73_first_contact_bilateral.controller import WrapState, select_token, v2_spec
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal
from experiments.precision.p82_global_failure_budget.paths import (
    CACHE,
    MAX_EP_STEPS,
    N_PER_CLASS,
    N_SEEDS,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    SEED,
    SKILLS,
    T_MAX,
)


def _earliest_decisive(ever: dict, term: dict, timed_out: bool) -> str:
    """One earliest decisive failure label (no double-count)."""
    if term["SUCCESS_RELAXED"]:
        return "success"
    if not ever["fc"]:
        return "no_first_contact"
    if not ever["opp"]:
        return "no_opposing_after_fc"
    if not ever["pad4"]:
        return "no_min_pad4"
    if not ever["e9"]:
        return "no_e_pos_le9"
    if not ever["frame"]:
        return "no_frame_ok"
    if not ever["r3"] and not term["r3_ready"]:
        return "no_r3_ready"
    if term["r3_ready"] or term["handoff"]:
        if not term["SUCCESS_RELAXED"]:
            return "post_handoff_or_terminal_fail"
    if timed_out:
        return "timeout"
    return "other_terminal"


def rollout_episode(env, actor, critic, book, start, device, *, seed: int) -> dict:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info = {}
    ever = {k: False for k in ("fc", "sw", "bil", "opp", "pad4", "e9", "frame", "r3", "loss", "recatch_unstable")}
    t_fc = t_opp = t_loss = None
    had = False
    family_seq = []
    switches = 0
    prev_fam = None
    forced_steps = actor_steps = 0
    post_fc_fams = []
    grip_open_near_contact = 0
    approach_near_contact = 0

    while not done:
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, dbg = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
        )
        fam = family_of_token(book, int(tok))
        family_seq.append({"t": int(env.t), "tok": int(tok), "fam": fam, "forced": dbg.get("forced")})
        if prev_fam is not None and fam != prev_fam:
            switches += 1
        prev_fam = fam
        if dbg.get("forced") is not None:
            forced_steps += 1
        else:
            actor_steps += 1

        g_pre = relative_geometry(env.env.sim)
        if g_pre["has_pad_contact"] or st.saw_contact:
            post_fc_fams.append(fam)
            a = np.asarray(book[int(tok)], float).reshape(7)
            if float(a[6]) < -0.3:
                grip_open_near_contact += 1
            if fam == "APPROACH" and float(np.linalg.norm(a[:3])) > 0.8:
                approach_near_contact += 1

        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        g = relative_geometry(env.env.sim)
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)

        if g["has_pad_contact"]:
            if not ever["fc"]:
                t_fc = int(env.t)
            ever["fc"] = True
            had = True
        elif had and not g["has_pad_contact"]:
            ever["loss"] = True
            if t_loss is None:
                t_loss = int(env.t)
        if g["correct_sidewall"]:
            ever["sw"] = True
        if g["bilateral"]:
            ever["bil"] = True
        if g["bilateral"] and g["correct_sidewall"]:
            if not ever["opp"]:
                t_opp = int(env.t)
            ever["opp"] = True
        if g["min_pad"] >= 4:
            ever["pad4"] = True
        if g["e_pos_mm"] <= 9:
            ever["e9"] = True
        if preds["frame_ok"]:
            ever["frame"] = True
        if (preds["dwell_complete"] and preds["frame_ok"]) or int(info.get("handoff") or 0):
            ever["r3"] = True
        # recatch instability proxy
        if getattr(env, "recatch", None) is not None and getattr(env.recatch, "lost_after_recatch", False):
            ever["recatch_unstable"] = True

    term = _r3_terminal(env, info)
    timed_out = bool(int(env.t) >= int(env.max_steps) - 1) and not term["SUCCESS_RELAXED"]
    decisive = _earliest_decisive(ever, term, timed_out)

    return {
        "skill": start.skill,
        "traj_id": start.traj_id,
        "seed": seed,
        "SUCCESS_RELAXED": int(term["SUCCESS_RELAXED"]),
        "r3_ready": int(term["r3_ready"] or ever["r3"]),
        "handoff": int(term["handoff"]),
        "first_contact": int(ever["fc"]),
        "correct_sidewall_ever": int(ever["sw"]),
        "bilateral_ever": int(ever["bil"]),
        "opposing_ever": int(ever["opp"]),
        "min_pad_ge4": int(ever["pad4"]),
        "e_pos_le9": int(ever["e9"]),
        "frame_ok": int(ever["frame"]),
        "contact_loss_ever": int(ever["loss"]),
        "recatch_unstable": int(ever["recatch_unstable"]),
        "timeout": int(timed_out),
        "earliest_decisive_failure": decisive,
        "t_fc": t_fc,
        "t_opp": t_opp,
        "t_loss": t_loss,
        "n_family_switches": switches,
        "post_fc_family_counts": dict(__import__("collections").Counter(post_fc_fams)),
        "grip_open_near_contact_steps": grip_open_near_contact,
        "strong_approach_near_contact_steps": approach_near_contact,
        "actor_steps": actor_steps,
        "forced_steps": forced_steps,
        "family_seq": family_seq[-40:],  # tail for size
        "ep_len": int(env.t),
    }


def collect_baseline(*, device: str = "cuda", force: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / "baseline_episodes.pkl"
    if cache_path.exists() and not force:
        rows = pickle.loads(cache_path.read_bytes())
        return {"rows": rows, "cached": True}

    # Reuse P81 baseline rows if present (inject=0) as seed, then top up
    p81 = CACHE.parents[0].parent / "p81_static_contact_bias" / "cache" / "full_ep_rows.pkl"
    # path: results/experiments/p82.../cache -> parents[1]=p82, parents[2]=experiments
    p81 = OUT.parent / "p81_static_contact_bias" / "cache" / "full_ep_rows.pkl"

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    critic = load_critic(P64_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    blocked = blocked_eval_ids()
    pool = build_p54_dataset(seed=5478)["schedule"]
    by = defaultdict(list)
    for s in pool:
        if is_blocked_eval(s.traj_id, blocked):
            continue
        by[s.skill].append(s)
    rng = np.random.default_rng(SEED)
    selected = []
    for sk in SKILLS:
        cand = list(by.get(sk, []))
        rng.shuffle(cand)
        selected.extend(cand[:N_PER_CLASS])

    rows = []
    for ii, start in enumerate(selected):
        for k in range(N_SEEDS):
            seed = rollout_seed(split="train", traj_id=start.traj_id, rollout_i=83000 + 20 * ii + k)
            rec = rollout_episode(env, actor, critic, book, start, device_t, seed=seed)
            rows.append(rec)
        print(
            f"[base] {ii+1}/{len(selected)} {start.skill} {start.traj_id[-16:]} "
            f"succ={rows[-1]['SUCCESS_RELAXED']} fail={rows[-1]['earliest_decisive_failure']}",
            flush=True,
        )

    cache_path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    n_forced_ep = sum(1 for r in rows if int(r.get("forced_steps") or 0) > 0)
    mean_forced = float(np.mean([int(r.get("forced_steps") or 0) for r in rows])) if rows else 0.0
    v2_activation_rate = float(n_forced_ep / len(rows)) if rows else 0.0
    meta = {
        "n": len(rows),
        "per_class": {sk: sum(1 for r in rows if r["skill"] == sk) for sk in SKILLS},
        "blocked_eval_excluded": True,
        "blocked_eval_uses_root_id": True,
        "reused_p81_baseline": False,
        "p81_cache_exists": p81.exists(),
        "v2_activation": {
            "episodes_with_forced_gt0": n_forced_ep,
            "n_episodes": len(rows),
            "activation_rate": v2_activation_rate,
            "mean_forced_steps": mean_forced,
            "note": (
                "V2 require_no_contact; contact-at-start ICs yield ~Actor-only. "
                "Do not attribute MSP to V2 closed-loop when activation_rate is near 0."
            ),
        },
    }
    if rows and v2_activation_rate < 0.05:
        print(
            f"[p82] WARNING: V2 activation_rate={v2_activation_rate:.3f} "
            f"({n_forced_ep}/{len(rows)}); panel is ≈ Actor-only.",
            flush=True,
        )
    (CACHE / "collection_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"rows": rows, "cached": False, "meta": meta}


if __name__ == "__main__":
    collect_baseline()
