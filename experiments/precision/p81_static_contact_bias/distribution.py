"""Phase 1 — Actor action distribution during FC recovery window."""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.distributions import Categorical

from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.paths import WINDOW
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p69_skill_sequence.skill_families import family_of_token
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs, obs_features
from experiments.precision.p71_contact_safe_alignment.align_tokens import decode_token
from experiments.precision.p73_first_contact_bilateral.controller import WrapState, select_token, v2_spec
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal
from experiments.precision.p81_static_contact_bias.paths import (
    CACHE,
    MAX_EP_STEPS,
    N_DIST_ICS,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    SEED,
    T_MAX,
    TOKEN_56,
    TOP_K,
)


STAGES = (
    "pre_contact",
    "first_actor_owned_fc",
    "early_post_contact",
    "opposing_formation",
    "before_contact_loss",
    "r3_approach",
)


@torch.no_grad()
def _dist_info(actor, obs_seq, prev_seq, device, book) -> dict:
    obs = torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device, dtype=torch.float32)
    logits, res, _ = actor(obs, prev)
    logit = mask_handoff_logits(logits[-1].float())
    dist = Categorical(logits=logit)
    probs = dist.probs.detach().cpu().numpy()
    ent = float(dist.entropy().item())
    top = np.argsort(-probs)[:TOP_K]
    return {
        "entropy": ent,
        "p_tok56": float(probs[TOKEN_56]),
        "argmax": int(np.argmax(probs)),
        "top_k": [
            {
                "token": int(i),
                "prob": float(probs[i]),
                "family": family_of_token(book, int(i)),
            }
            for i in top
        ],
        "resid": res[-1].detach().cpu().numpy().astype(np.float32),
    }


def _stage_tag(st: WrapState, g: dict, feat: dict, ever_opp: bool, ever_contact: bool, losing: bool) -> str | None:
    if not ever_contact and not g["has_pad_contact"]:
        return "pre_contact"
    if g["has_pad_contact"] and st.saw_contact and not ever_opp:
        # first actor-owned after contact: phase post_v2_or_actor
        return None  # resolved by caller with counters
    return None


def run_distribution_audit(*, device: str = "cuda", force: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / "dist_rows.pkl"
    if cache_path.exists() and not force:
        rows = pickle.loads(cache_path.read_bytes())
        return _summarize_dist(rows)

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    critic = load_critic(P64_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    blocked = blocked_eval_ids()
    starts = [s for s in build_p54_dataset(seed=5478)["schedule"] if s.skill == "C" and s.traj_id not in blocked]
    starts = starts[:N_DIST_ICS]
    v2 = v2_spec()
    rows = []

    for ii, start in enumerate(starts):
        seed = rollout_seed(split="train", traj_id=start.traj_id, rollout_i=81000 + ii)
        seed_all(seed, deterministic_torch=True)
        env.reset(start)
        ensure_inner_capture_buffers(env)
        st = WrapState()
        done = False
        info = {}
        ever_contact = ever_opp = ever_loss = False
        fc_t = None
        captured = set()
        had_contact = False
        stage_events = []

        while not done:
            # Distribution BEFORE action
            dinfo = _dist_info(actor, env.obs_seq, env.prev_seq, device_t, book)
            tok_a, resid, logp, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device_t)
            tok, st, dbg = select_token(
                env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
            )
            g = relative_geometry(env.env.sim)
            feat = obs_features(last_obs(env))
            actor_owns = dbg.get("forced") is None and int(tok) == int(tok_a)

            stage = None
            if not ever_contact and not g["has_pad_contact"] and "pre_contact" not in captured and env.t >= 1:
                stage = "pre_contact"
            if g["has_pad_contact"] and fc_t is None:
                fc_t = int(env.t)
            if (
                g["has_pad_contact"]
                and st.saw_contact
                and actor_owns
                and "first_actor_owned_fc" not in captured
            ):
                stage = "first_actor_owned_fc"
            if (
                fc_t is not None
                and int(env.t) >= fc_t + 2
                and g["has_pad_contact"]
                and not ever_opp
                and "early_post_contact" not in captured
                and actor_owns
            ):
                stage = "early_post_contact"
            if g["bilateral"] and g["correct_sidewall"] and "opposing_formation" not in captured and actor_owns:
                stage = "opposing_formation"
                ever_opp = True
            # before contact loss: had contact, still contact, min_pad<=1, e_pos rising
            if (
                had_contact
                and g["has_pad_contact"]
                and g["min_pad"] <= 1
                and float(g["e_pos_mm"]) > 12
                and "before_contact_loss" not in captured
                and actor_owns
            ):
                stage = "before_contact_loss"
            # r3 approach: e_pos small + bilateral
            if (
                g["bilateral"]
                and float(g["e_pos_mm"]) <= 12
                and "r3_approach" not in captured
                and actor_owns
            ):
                stage = "r3_approach"

            if stage is not None and stage not in captured:
                captured.add(stage)
                rec = {
                    "traj_id": start.traj_id,
                    "seed": seed,
                    "t": int(env.t),
                    "stage": stage,
                    "selected_token": int(tok_a),
                    "executed_token": int(tok),
                    "actor_owns": bool(actor_owns),
                    "forced": dbg.get("forced"),
                    "p_tok56": dinfo["p_tok56"],
                    "entropy": dinfo["entropy"],
                    "argmax": dinfo["argmax"],
                    "top_k": dinfo["top_k"],
                    "selected_family": family_of_token(book, int(tok_a)),
                    "tok56_family": family_of_token(book, TOKEN_56),
                    "selected_decode": decode_token(book, int(tok_a)),
                    "e_pos_mm": float(g["e_pos_mm"]),
                    "contact": int(g["has_pad_contact"]),
                    "nL": int(g["nL"]),
                    "nR": int(g["nR"]),
                    "obs_L": int(bool(feat.get("contact_L"))),
                    "obs_R": int(bool(feat.get("contact_R"))),
                    "bilateral": int(g["bilateral"]),
                    "correct_sidewall": int(g["correct_sidewall"]),
                    "aperture": float(feat.get("aperture") or 0),
                    "ap_dot": float(feat.get("ap_dot") or 0),
                    "underselect_56": bool(dinfo["p_tok56"] < 0.15 and int(tok_a) != TOKEN_56),
                    "selects_56": bool(int(tok_a) == TOKEN_56),
                }
                stage_events.append(rec)
                rows.append(rec)

            _r, done, info = env.step(int(tok), resid)
            ensure_inner_capture_buffers(env)
            g2 = relative_geometry(env.env.sim)
            if g2["has_pad_contact"]:
                had_contact = ever_contact = True
            elif had_contact and not g2["has_pad_contact"]:
                ever_loss = True
            if g2["bilateral"] and g2["correct_sidewall"]:
                ever_opp = True

        term = _r3_terminal(env, info)
        for rec in stage_events:
            rec["ep_r3"] = term["r3_ready"]
            rec["ep_success"] = term["SUCCESS_RELAXED"]
            rec["ep_contact_loss"] = int(ever_loss)
            rec["ep_opposing"] = int(ever_opp)

        print(
            f"[dist] {ii+1}/{len(starts)} {start.traj_id[-20:]} stages={sorted(captured)} "
            f"succ={term['SUCCESS_RELAXED']}",
            flush=True,
        )

    cache_path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    return _summarize_dist(rows)


def _summarize_dist(rows: list[dict]) -> dict:
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)

    stage_sum = {}
    for st, rs in by_stage.items():
        toks = Counter(r["selected_token"] for r in rs)
        fams = Counter(r["selected_family"] for r in rs)
        stage_sum[st] = {
            "n": len(rs),
            "n_ics": len({r["traj_id"] for r in rs}),
            "mean_p_tok56": float(np.mean([r["p_tok56"] for r in rs])),
            "frac_select_56": float(np.mean([r["selects_56"] for r in rs])),
            "frac_underselect_56": float(np.mean([r["underselect_56"] for r in rs])),
            "mean_entropy": float(np.mean([r["entropy"] for r in rs])),
            "top_tokens": toks.most_common(8),
            "families": dict(fams),
            "mean_e_pos": float(np.mean([r["e_pos_mm"] for r in rs])),
            "frac_ep_success_when_observed": float(np.mean([r["ep_success"] for r in rs])),
        }

    # At first_actor_owned_fc: does Actor underselect 56?
    fc = by_stage.get("first_actor_owned_fc") or []
    report = {
        "n_rows": len(rows),
        "n_ics": len({r["traj_id"] for r in rows}),
        "by_stage": stage_sum,
        "fc_decision": {
            "n": len(fc),
            "mean_p_tok56": float(np.mean([r["p_tok56"] for r in fc])) if fc else None,
            "frac_select_56": float(np.mean([r["selects_56"] for r in fc])) if fc else None,
            "frac_argmax_is_56": float(np.mean([r["argmax"] == TOKEN_56 for r in fc])) if fc else None,
            "systematic_underselect": bool(fc and np.mean([r["p_tok56"] for r in fc]) < 0.2 and np.mean([r["selects_56"] for r in fc]) < 0.15),
            "note": (
                "Distinguish sampling: if argmax≈56 but samples elsewhere → sampling noise; "
                "if p(56) low and argmax≠56 → genuine preference against 56."
            ),
        },
    }
    (OUT / "ACTOR_ACTION_DISTRIBUTION.md").write_text(
        "# Actor Action Distribution (P81 Phase 1)\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "ACTOR_ACTION_DISTRIBUTION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["fc_decision"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_distribution_audit()
