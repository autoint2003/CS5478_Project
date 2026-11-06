"""Phase 1 — snapshot restore audit."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p73_first_contact_bilateral.controller import WrapState, select_token, v2_spec
from experiments.precision.p79_matched_cf_coverage.paths import MAX_EP_STEPS, OUT, P64_ACTOR, P64_CRITIC, SEED, T_MAX


def _qpos(env):
    return np.asarray(env.env.sim.data.qpos, float).copy()


def run_restore_audit(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
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
    start = starts[0]
    seed = rollout_seed(split="train", traj_id=start.traj_id, rollout_i=79001)
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()
    done = False
    state = None
    while not done:
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device_t)
        tok, st, dbg = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
        )
        r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        g = relative_geometry(env.env.sim)
        if g["has_pad_contact"] and state is None:
            # Capture at first contact — next decision is post-contact
            state = take_full_state(
                env, meta={"kind": "first_contact", "traj_id": start.traj_id, "seed": seed, "t": int(env.t)}
            )
            break
    if state is None:
        report = {"ok": False, "error": "no_contact_in_audit_episode"}
        (OUT / "SNAPSHOT_RESTORE_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    # Live reference after capture
    q0 = _qpos(env)
    t0 = int(env.t)
    obs0 = np.asarray(env.obs_seq[-1], float).copy()
    n_hist0 = len(env.obs_seq)
    g0 = relative_geometry(env.env.sim)

    # Restore twice and compare
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    q1 = _qpos(env)
    t1 = int(env.t)
    obs1 = np.asarray(env.obs_seq[-1], float).copy()
    n_hist1 = len(env.obs_seq)
    g1 = relative_geometry(env.env.sim)

    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    q2 = _qpos(env)

    # Determinism: same forced token after restore, two seeds
    def step_once(seed_i, force_tok=56):
        seed_all(seed_i, deterministic_torch=True)
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device_t)
        # force single token then read geometry after one step
        r, done, info = env.step(int(force_tok), resid)
        return {
            "qpos": _qpos(env),
            "e_pos": float(relative_geometry(env.env.sim)["e_pos_mm"]),
            "actor_would": int(tok_a),
            "reward": float(r),
        }

    a = step_once(SEED + 1)
    b = step_once(SEED + 1)  # same seed
    c = step_once(SEED + 2)  # different seed

    det_same = bool(np.allclose(a["qpos"], b["qpos"], atol=1e-9) and abs(a["e_pos"] - b["e_pos"]) < 1e-9)
    det_diff_seed = bool(np.allclose(a["qpos"], c["qpos"], atol=1e-9) and abs(a["e_pos"] - c["e_pos"]) < 1e-9)

    # Actor ownership at FC: V2 should not force (dbg at next decision)
    seed_all(SEED + 3, deterministic_torch=True)
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    st2 = WrapState()
    st2.saw_contact = True
    st2.v2.phase = "done"
    st2.v2.activated = True
    tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device_t)
    tok, st2, dbg = select_token(
        env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st2, v2=v2_spec(), rule=None
    )
    actor_owns = bool(dbg.get("forced") is None) and int(tok) == int(tok_a)

    report = {
        "ok": True,
        "checks": {
            "qpos_matches_live": bool(np.allclose(q0, q1, atol=1e-8)),
            "qpos_restore_idempotent": bool(np.allclose(q1, q2, atol=1e-8)),
            "timestep_match": t0 == t1 == int(state["policy_step"]),
            "obs_hist_len_match": n_hist0 == n_hist1,
            "obs_last_match": bool(np.allclose(obs0, obs1, atol=1e-5)),
            "geometry_e_pos_match": abs(float(g0["e_pos_mm"]) - float(g1["e_pos_mm"])) < 1e-5,
            "same_seed_deterministic": det_same,
            "diff_seed_also_identical": det_diff_seed,
            "actor_owns_fc_decision": actor_owns,
            "v2_forced_at_fc": dbg.get("forced"),
            "phase_at_fc": dbg.get("phase"),
        },
        "implications": {
            "if_deterministic": (
                "Do not treat repeated identical seeds as independent evidence. "
                "Use matched perturbations or single-seed comparisons."
            ),
            "obs_before_action": "Features extracted after restore, before CF intervention step.",
            "matched_start": "All candidates restore the same full_state blob.",
        },
        "state_keys": sorted(state.keys()),
        "includes": {
            "mujoco_snap": "snap" in state,
            "r3f": "r3f" in state,
            "obs_seq": "obs_seq" in state or "policy_windows" in state or "obs" in str(state.keys()),
            "recatch": "recatch" in state,
            "policy_step": "policy_step" in state,
        },
    }
    # Inspect nested keys for obs history
    for k in state:
        if "obs" in k.lower() or "prev" in k.lower() or "window" in k.lower():
            report["includes"][k] = True

    (OUT / "SNAPSHOT_RESTORE_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "SNAPSHOT_RESTORE_AUDIT.md").write_text(
        "# Snapshot Restore Audit (P79 Phase 1)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(report["checks"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_restore_audit()
