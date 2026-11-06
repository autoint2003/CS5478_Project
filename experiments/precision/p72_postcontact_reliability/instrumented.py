"""Instrumented V2 rollouts for post-contact funnel + timelines."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p69_skill_sequence.skill_families import family_of_token
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p72_postcontact_reliability.v2_wrap import (
    PostRefinement,
    WrapState,
    select_token,
    v2_spec,
)

STAGES = (
    "first_contact",
    "opposing_bilateral",  # bilateral + correct sidewall
    "sustained_bilateral",  # bil run >= 3
    "min_pad_ge4",
    "frame_ok",
    "margin_ok",
    "r3_ready",
    "handoff",
    "SUCCESS_RELAXED",
)


def _snap(env, book, tok: int | None = None) -> dict[str, Any]:
    g = relative_geometry(env.env.sim)
    tr = env.mon.tracker
    ok_s = float(tr.ok_s) if tr else 0.0
    preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
    fam = None if tok is None or book is None else family_of_token(book, int(tok))
    residual = None
    if tok is not None and book is not None:
        a = np.asarray(book[int(tok)], float).reshape(7)
        residual = {
            "lin_norm": float(np.linalg.norm(a[:3])),
            "ang_norm": float(np.linalg.norm(a[3:6])),
            "grip": float(a[6]),
        }
    return {
        "t": int(env.t),
        "e_pos_mm": float(g["e_pos_mm"]),
        "theta_deg": float(g["theta_deg"]),
        "nL": int(g["nL"]),
        "nR": int(g["nR"]),
        "min_pad": int(g["min_pad"]),
        "bilateral": int(g["bilateral"]),
        "correct_sidewall": int(g["correct_sidewall"]),
        "has_pad_contact": int(g["has_pad_contact"]),
        "obj_pos": list(map(float, g["obj_pos"])),
        "frame_ok": int(preds["frame_ok"]),
        "margin_ok": int(preds["margin_ok"]),
        "min_pad_ok": int(preds["min_pad_ok"]),
        "r3_pos_ok": int(preds["r3_pos_ok"]),
        "r3_theta_ok": int(preds["r3_theta_ok"]),
        "working_ok": int(preds["working_ok"]),
        "dwell_complete": int(preds["dwell_complete"]),
        "ok_s": ok_s,
        "token": None if tok is None else int(tok),
        "family": fam,
        "residual": residual,
    }


def rollout_instrumented(
    env,
    actor,
    critic,
    book,
    start,
    device,
    *,
    seed: int,
    post: PostRefinement | None = None,
    tag: str = "V2",
    record_trace: bool = True,
) -> dict[str, Any]:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}

    first_pass: dict[str, dict | None] = {k: None for k in STAGES}
    bil_run = 0
    contact_run = 0
    ever = {k: False for k in STAGES}
    lost_after = {k: 0 for k in STAGES}  # regressions after having reached
    had = {k: False for k in ("bilateral", "min_pad_ge4", "frame_ok", "margin_ok", "contact")}
    residence = {k: 0 for k in ("bilateral", "min_pad_ge4", "frame_ok", "margin_ok", "contact")}
    post_tokens = []
    trace = []
    obj0 = None
    e_at_fc = None
    t_to_12 = t_to_9 = None
    min_e_after_fc = None
    max_ok_s = 0.0
    first_violated_near = None
    near_ready_seen = False

    while not done:
        m0 = _snap(env, book, None)
        if obj0 is None:
            obj0 = np.asarray(m0["obj_pos"], float)

        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, dbg = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, post=post
        )
        if st.saw_contact:
            post_tokens.append(int(tok))

        _r, done, info = env.step(int(tok), resid)
        m1 = _snap(env, book, int(tok))
        m1["obj_disp_from_t0"] = float(np.linalg.norm(np.asarray(m1["obj_pos"]) - obj0))
        m1["dbg_phase"] = dbg.get("phase")

        if record_trace and (st.saw_contact or m1["has_pad_contact"]):
            # keep post-contact trace only (compact)
            trace.append({k: m1[k] for k in (
                "t", "e_pos_mm", "theta_deg", "nL", "nR", "min_pad", "bilateral",
                "correct_sidewall", "has_pad_contact", "frame_ok", "margin_ok", "ok_s",
                "token", "family", "obj_disp_from_t0", "dbg_phase",
            )})

        # contact runs
        if m1["has_pad_contact"]:
            contact_run += 1
            had["contact"] = True
            residence["contact"] += 1
        else:
            if had["contact"]:
                lost_after["first_contact"] += 1
            contact_run = 0

        # Opposing = bilateral + correct sidewall (matches stage definition)
        opposing_now = bool(m1["bilateral"] and m1["correct_sidewall"])
        if opposing_now:
            bil_run += 1
            had["bilateral"] = True
            residence["bilateral"] += 1
        else:
            if had["bilateral"]:
                lost_after["opposing_bilateral"] += 1
                if ever["sustained_bilateral"]:
                    lost_after["sustained_bilateral"] += 1
            bil_run = 0

        if m1["min_pad"] >= 4:
            had["min_pad_ge4"] = True
            residence["min_pad_ge4"] += 1
        elif had["min_pad_ge4"]:
            lost_after["min_pad_ge4"] += 1

        if m1["frame_ok"]:
            had["frame_ok"] = True
            residence["frame_ok"] += 1
        elif had["frame_ok"]:
            lost_after["frame_ok"] += 1

        if m1["margin_ok"]:
            had["margin_ok"] = True
            residence["margin_ok"] += 1
        elif had["margin_ok"]:
            lost_after["margin_ok"] += 1

        max_ok_s = max(max_ok_s, float(m1["ok_s"]))

        def mark(name, cond, payload):
            if cond and first_pass[name] is None:
                first_pass[name] = payload
                ever[name] = True

        mark(
            "first_contact",
            m1["has_pad_contact"],
            {"t": m1["t"], "e_pos_mm": m1["e_pos_mm"], "theta_deg": m1["theta_deg"], "min_pad": m1["min_pad"], "sw": m1["correct_sidewall"]},
        )
        if first_pass["first_contact"] is not None:
            if e_at_fc is None:
                e_at_fc = float(m1["e_pos_mm"]) if first_pass["first_contact"]["t"] == m1["t"] else float(first_pass["first_contact"]["e_pos_mm"])
            if min_e_after_fc is None or m1["e_pos_mm"] < min_e_after_fc:
                min_e_after_fc = float(m1["e_pos_mm"])
            if t_to_12 is None and m1["e_pos_mm"] <= 12:
                t_to_12 = int(m1["t"] - first_pass["first_contact"]["t"])
            if t_to_9 is None and m1["e_pos_mm"] <= 9:
                t_to_9 = int(m1["t"] - first_pass["first_contact"]["t"])

        mark(
            "opposing_bilateral",
            bool(m1["bilateral"] and m1["correct_sidewall"]),
            {"t": m1["t"], "e_pos_mm": m1["e_pos_mm"], "min_pad": m1["min_pad"]},
        )
        mark(
            "sustained_bilateral",
            # Require opposing sidewall so funnel stages stay nested (ever rates monotonic).
            opposing_now and bil_run >= 3,
            {"t": m1["t"], "e_pos_mm": m1["e_pos_mm"], "min_pad": m1["min_pad"], "bil_run": bil_run},
        )
        mark("min_pad_ge4", m1["min_pad"] >= 4, {"t": m1["t"], "e_pos_mm": m1["e_pos_mm"], "min_pad": m1["min_pad"]})
        mark("frame_ok", bool(m1["frame_ok"]), {"t": m1["t"], "e_pos_mm": m1["e_pos_mm"], "ok_s": m1["ok_s"]})
        mark("margin_ok", bool(m1["margin_ok"]), {"t": m1["t"], "e_pos_mm": m1["e_pos_mm"]})

        # near-ready: frame_ok started but not dwell
        if m1["frame_ok"] and m1["ok_s"] > 0:
            near_ready_seen = True
        if near_ready_seen and first_violated_near is None and not m1["frame_ok"]:
            viol = []
            if not m1["bilateral"]:
                viol.append("bilateral")
            if m1["min_pad"] < 4:
                viol.append("min_pad")
            if not m1["margin_ok"]:
                viol.append("margin_ok")
            if not m1["working_ok"]:
                viol.append("working_ok")
            if not m1["frame_ok"]:
                viol.append("frame_ok")
            first_violated_near = {"t": m1["t"], "violations": viol or ["frame_ok"], "ok_s": m1["ok_s"]}

    handoff = int(info.get("handoff") or 0)
    succ = int(info.get("SUCCESS_RELAXED_12S") or 0)
    tr = env.mon.tracker
    ok_s = float(tr.ok_s) if tr else 0.0
    preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
    r3 = int(bool((preds["dwell_complete"] and preds["frame_ok"]) or handoff))
    if r3:
        ever["r3_ready"] = True
        if first_pass["r3_ready"] is None:
            first_pass["r3_ready"] = {"t": int(env.t), "e_pos_mm": float(relative_geometry(env.env.sim)["e_pos_mm"])}
    if handoff:
        ever["handoff"] = True
        first_pass["handoff"] = first_pass["handoff"] or {"t": int(env.t)}
    if succ:
        ever["SUCCESS_RELAXED"] = True
        first_pass["SUCCESS_RELAXED"] = first_pass["SUCCESS_RELAXED"] or {"t": int(env.t)}

    # post-contact skill summary
    fams = [family_of_token(book, t) for t in post_tokens]
    n_switch = sum(1 for a, b in zip(fams, fams[1:]) if a != b)

    return {
        "tag": tag,
        "traj_id": start.traj_id,
        "seed": seed,
        "first_pass": first_pass,
        "ever": {k: int(v) for k, v in ever.items()},
        "lost_after_counts": lost_after,
        "residence_steps": residence,
        "e_at_fc": e_at_fc,
        "min_e_after_fc": min_e_after_fc,
        "t_to_12": t_to_12,
        "t_to_9": t_to_9,
        "reached_e9": int(t_to_9 is not None),
        "reached_e12": int(t_to_12 is not None),
        "r3_ready": r3,
        "handoff": handoff,
        "SUCCESS_RELAXED": succ,
        "max_ok_s": max_ok_s,
        "near_ready_seen": int(near_ready_seen),
        "first_violated_near": first_violated_near,
        "post_tokens": post_tokens[:40],
        "post_families": fams[:40],
        "n_family_switches_post": n_switch,
        "n_post_steps": len(post_tokens),
        "trace": trace if record_trace else [],
        "post_refinement": None if post is None else post.name,
    }
