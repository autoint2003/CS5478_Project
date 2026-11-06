"""Instrumented V2+Actor rollouts for ownership / observability / Critic audits."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p69_skill_sequence.skill_families import family_of_token
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs, obs_features
from experiments.precision.p71_contact_safe_alignment.controllers import CtrlState, select_token, make_specs
from experiments.precision.p74_contact_learning_readiness.sample_ext import act_diagnose, logp_of_token

WINDOWS = (
    "pre_fc",
    "fc_to_opposing",
    "fc_to_contact_loss",
    "opposing_to_frame",
    "frame_to_r3",
)


def _window_at_decision(
    *,
    saw_fc_before: bool,
    ever_opp_before: bool,
    ever_frame_before: bool,
    ever_r3_before: bool,
    lost_after_fc_before: bool,
) -> str | None:
    """Window for the decision about to be taken (pre-step state).

    Note: Spec window \"FC → contact loss\" is a *retrospective path*. Live tags:
    - `fc_to_opposing`: after FC, before opposing, contact not yet lost
    - `fc_to_contact_loss`: after first post-FC contact loss, still before opposing
      (recovery attempts). Ownership of the pre-loss segment is aggregated offline.
    """
    if not saw_fc_before:
        return "pre_fc"
    if ever_r3_before:
        return None
    if ever_frame_before:
        return "frame_to_r3"
    if ever_opp_before:
        return "opposing_to_frame"
    if lost_after_fc_before:
        return "fc_to_contact_loss"
    return "fc_to_opposing"


def rollout_audit(env, actor, critic, book, start, device, *, seed: int) -> dict[str, Any]:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    st = CtrlState()
    v2 = make_specs([])["V2"]
    done = False
    info: dict = {}

    steps: list[dict] = []
    returns_rev: list[float] = []  # collect rewards then reverse-cumsum

    fc_t = None
    ever_opp = ever_frame = ever_r3 = False
    lost_after_fc = False
    had_contact = False
    obj0 = None
    obj_at_fc = None
    saw_fc = False

    while not done:
        g = relative_geometry(env.env.sim)
        if obj0 is None:
            obj0 = np.asarray(g["obj_pos"], float)
        # Pre-step privileged contact for window (decision-time)
        saw_fc_before = saw_fc
        ever_opp_before = ever_opp
        ever_frame_before = ever_frame
        ever_r3_before = ever_r3
        lost_before = lost_after_fc
        wtag = _window_at_decision(
            saw_fc_before=saw_fc_before,
            ever_opp_before=ever_opp_before,
            ever_frame_before=ever_frame_before,
            ever_r3_before=ever_r3_before,
            lost_after_fc_before=lost_before,
        )

        feat = obs_features(last_obs(env))
        obs_vec = np.asarray(last_obs(env), float).reshape(-1)

        diag = act_diagnose(actor, critic, env.obs_seq, env.prev_seq, device)
        actor_tok = int(diag["tok"])
        resid = diag["resid"]
        tok, st, dbg = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=actor_tok, st=st, spec=v2
        )
        executed = int(tok)
        v2_override = bool(dbg.get("forced") is not None) or (executed != actor_tok)
        # Valid behavior logp for executed action:
        # - if Actor-controlled: use sampled logp (same token) or recompute
        # - if V2 override: Actor logp of executed token is NOT the behavior logp
        if v2_override:
            actor_logp_of_exec = logp_of_token(actor, env.obs_seq, env.prev_seq, device, executed)
            behavior_logp_valid = False
            behavior_logp = None
        else:
            actor_logp_of_exec = float(diag["logp"]) if executed == actor_tok else logp_of_token(
                actor, env.obs_seq, env.prev_seq, device, executed
            )
            behavior_logp_valid = True
            behavior_logp = actor_logp_of_exec

        r, done, info = env.step(executed, resid)
        returns_rev.append(float(r))

        g1 = relative_geometry(env.env.sim)
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)

        if not saw_fc and g1["has_pad_contact"]:
            saw_fc = True
            fc_t = int(env.t)
            obj_at_fc = np.asarray(g1["obj_pos"], float)

        in_contact = bool(g1["has_pad_contact"])
        if saw_fc:
            if in_contact:
                had_contact = True
            elif had_contact:
                lost_after_fc = True

        opposing = bool(g1["bilateral"] and g1["correct_sidewall"])
        if opposing:
            ever_opp = True
        if preds["frame_ok"]:
            ever_frame = True
        handoff = int(info.get("handoff") or 0)
        if (preds["dwell_complete"] and preds["frame_ok"]) or handoff:
            ever_r3 = True

        disp_fc = (
            float(np.linalg.norm(np.asarray(g1["obj_pos"]) - obj_at_fc)) if obj_at_fc is not None else None
        )

        steps.append(
            {
                "t": int(env.t),
                "window": wtag,
                "actor_tok": actor_tok,
                "executed": executed,
                "v2_override": int(v2_override),
                "v2_phase": st.phase,
                "v2_forced": None if dbg.get("forced") is None else int(dbg["forced"]),
                "behavior_logp_valid": int(behavior_logp_valid),
                "behavior_logp": behavior_logp,
                "actor_logp_sample": float(diag["logp"]),
                "actor_logp_of_exec": float(actor_logp_of_exec),
                "actor_prob": float(diag["prob"]),
                "entropy": float(diag["entropy"]),
                "value": float(diag["value"]),
                "family_actor": family_of_token(book, actor_tok),
                "family_exec": family_of_token(book, executed),
                "reward": float(r),
                # obs features (deployable)
                "obs_contact_L": int(feat["contact_L"]),
                "obs_contact_R": int(feat["contact_R"]),
                "obs_any_contact": int(feat["any_contact_obs"]),
                "obs_e_hat_x": float(feat["e_hat_x"]),
                "obs_est_valid": int(feat["estimate_valid"]),
                "obs_aperture": float(feat["aperture"]),
                "obs_ap_dot": float(feat["ap_dot"]),
                "obs_hand_speed": float(feat["hand_speed"]),
                "obs_vec8": [
                    float(obs_vec[i]) if i < obs_vec.size else 0.0
                    for i in (2, 3, 0, 1, 12, 13, 18, 19)  # L,R,e_hat,valid,ap,ap_dot,vhx,vhy
                ],
                # privileged diagnostics (labels only)
                "e_pos_mm": float(g1["e_pos_mm"]),
                "theta_deg": float(g1["theta_deg"]),
                "nL": int(g1["nL"]),
                "nR": int(g1["nR"]),
                "min_pad": int(g1["min_pad"]),
                "bilateral": int(g1["bilateral"]),
                "correct_sidewall": int(g1["correct_sidewall"]),
                "opposing": int(opposing),
                "has_pad_contact": int(in_contact),
                "frame_ok": int(preds["frame_ok"]),
                "obj_disp_from_fc": disp_fc,
                "top5_tok": diag["top5_tok"],
                "top5_prob": diag["top5_prob"],
            }
        )

    # Monte Carlo returns from each step
    G = 0.0
    mc = [0.0] * len(returns_rev)
    for i in range(len(returns_rev) - 1, -1, -1):
        G = returns_rev[i] + G  # γ=1 episodic undiscounted for audit
        mc[i] = G
    for i, s in enumerate(steps):
        s["mc_return"] = float(mc[i])
        s["adv_proxy"] = float(mc[i] - s["value"])

    handoff = int(info.get("handoff") or 0)
    succ = int(info.get("SUCCESS_RELAXED_12S") or 0)
    tr = env.mon.tracker
    ok_s = float(tr.ok_s) if tr else 0.0
    preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
    r3 = int(bool((preds["dwell_complete"] and preds["frame_ok"]) or handoff))

    return {
        "traj_id": start.traj_id,
        "seed": seed,
        "r3_ready": r3,
        "SUCCESS_RELAXED": succ,
        "handoff": handoff,
        "reached_opposing": int(ever_opp),
        "reached_frame_ok": int(ever_frame),
        "lost_contact_after_fc": int(lost_after_fc),
        "fc_t": fc_t,
        "n_steps": len(steps),
        "steps": steps,
    }
