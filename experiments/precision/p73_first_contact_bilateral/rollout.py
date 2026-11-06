"""Instrumented V2 / P73 rollouts for first-contact → bilateral analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import contact_points_summary, relative_geometry
from experiments.precision.p69_skill_sequence.skill_families import family_of_token
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs, obs_features
from experiments.precision.p73_first_contact_bilateral.controller import (
    EventRule,
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.paths import FC_WINDOW


def _geom(env) -> dict[str, Any]:
    g = relative_geometry(env.env.sim)
    cps = contact_points_summary(env.env.sim)
    force = float(sum(float(p.get("Fn") or 0) for p in cps.get("points") or []))
    sides = [p.get("side") for p in (cps.get("points") or []) if p.get("side")]
    side0 = None
    if sides:
        # majority side at this step
        side0 = max(set(sides), key=sides.count)
    return {
        "t": int(env.t),
        "e_pos_mm": float(g["e_pos_mm"]),
        "theta_deg": float(g["theta_deg"]),
        "nL": int(g["nL"]),
        "nR": int(g["nR"]),
        "min_pad": int(g["min_pad"]),
        "bilateral": int(g["bilateral"]),
        "correct_sidewall": int(g["correct_sidewall"]),
        "opposing": int(bool(g["bilateral"] and g["correct_sidewall"])),
        "has_pad_contact": int(g["has_pad_contact"]),
        "obj_pos": list(map(float, g["obj_pos"])),
        "vrel": float(g.get("vrel") or 0),
        "force_sum": force,
        "contact_side": side0,
        "grasp_kind": str(g.get("grasp_kind") or ""),
        "n_contacts": int(g.get("n_contacts") or 0),
    }


def _r3_terminal(env, info: dict) -> dict[str, int]:
    handoff = int(info.get("handoff") or 0)
    succ = int(info.get("SUCCESS_RELAXED_12S") or 0)
    tr = env.mon.tracker
    ok_s = float(tr.ok_s) if tr else 0.0
    preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
    r3 = int(bool((preds["dwell_complete"] and preds["frame_ok"]) or handoff))
    return {
        "r3_ready": r3,
        "handoff": handoff,
        "SUCCESS_RELAXED": succ,
        "frame_ok": int(preds["frame_ok"]),
        "margin_ok": int(preds["margin_ok"]),
        "ok_s": ok_s,
    }


def classify_fc_failure(rec: dict) -> list[str]:
    """Overlapping failure labels A–F from first-contact window record.

    Note: correct_sidewall is often false on the exact first-contact frame even in
    successes; label A uses *ever* correct sidewall in the FC window instead.
    Object displacement for B is measured from the FC pose (not episode t0).
    """
    labels = []
    if not rec.get("reached_opposing"):
        # A: never achieved correct sidewall geometry in the FC window
        if not rec.get("ever_correct_sw_in_window"):
            labels.append("A_incorrect_geometry")
        # B: object pushed away from FC pose + e_pos worsens or stalls high
        disp = float(rec.get("max_obj_disp_from_fc") or 0)
        e0 = float(rec.get("e_at_fc") or 0)
        e1 = float(rec.get("e_at_end_window") if rec.get("e_at_end_window") is not None else e0)
        if disp >= 0.008 and e1 > e0 + 2.0:
            labels.append("B_object_pushed_away")
        elif disp >= 0.015 and (rec.get("min_e_in_window") or 99) > e0 - 1.0:
            labels.append("B_object_pushed_away")
        # C: skill switches before bil (flag only; rates are normalized separately)
        if (rec.get("switches_before_bil_or_end") or 0) >= 2:
            labels.append("C_skill_switch_before_bil")
        # D: contact lost before opposite engages
        if rec.get("lost_contact_before_opposing"):
            labels.append("D_contact_lost_before_opposite")
        # E: bilateral without correct opposing sidewall
        if rec.get("reached_bilateral") and not rec.get("reached_opposing"):
            labels.append("E_bil_but_wrong_sidewall")
        if not labels:
            labels.append("F_other_ambiguous")
    else:
        if not rec.get("reached_e9"):
            labels.append("E_bil_geometry_outside_margin")
        else:
            labels.append("ok_reached_opposing")
    return labels


def rollout(
    env,
    actor,
    critic,
    book,
    start,
    device,
    *,
    seed: int,
    rule: EventRule | None = None,
    tag: str = "V2",
    record_fc_window: bool = True,
) -> dict[str, Any]:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}

    fc_t = None
    e_at_fc = None
    fc_side = None
    fc_sw = None
    fc_nL = fc_nR = None
    obj0 = None
    obj_at_fc = None

    ever_opp = ever_bil = ever_pad4 = ever_e9 = ever_e12 = ever_frame = False
    lost_before_opp = False
    had_contact = False
    t_to_opp = t_to_bil = t_to_e9 = None
    min_e_after_fc = None
    min_e_after_opp = None
    max_obj_disp = 0.0
    max_obj_disp_from_fc = 0.0
    max_force = 0.0
    ever_correct_sw_in_window = False

    post_tokens: list[int] = []
    post_fams: list[str] = []
    fc_window_trace: list[dict] = []
    switches_before_bil = 0
    decisions_before_bil = 0
    window_end_reason = None
    e_at_end_window = None

    while not done:
        g0 = _geom(env)
        if obj0 is None:
            obj0 = np.asarray(g0["obj_pos"], float)

        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, dbg = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=rule
        )

        _r, done, info = env.step(int(tok), resid)
        g1 = _geom(env)
        feat = obs_features(last_obs(env))
        disp = float(np.linalg.norm(np.asarray(g1["obj_pos"]) - obj0))
        max_obj_disp = max(max_obj_disp, disp)
        max_force = max(max_force, float(g1["force_sum"]))

        # First contact (privileged pad for analysis)
        if fc_t is None and g1["has_pad_contact"]:
            fc_t = int(g1["t"])
            e_at_fc = float(g1["e_pos_mm"])
            fc_side = g1["contact_side"]
            fc_sw = int(g1["correct_sidewall"])
            fc_nL, fc_nR = int(g1["nL"]), int(g1["nR"])
            obj_at_fc = np.asarray(g1["obj_pos"], float)
            min_e_after_fc = float(g1["e_pos_mm"])

        if fc_t is not None:
            post_tokens.append(int(tok))
            fam = family_of_token(book, int(tok))
            post_fams.append(fam)
            if obj_at_fc is not None:
                dfc = float(np.linalg.norm(np.asarray(g1["obj_pos"]) - obj_at_fc))
                max_obj_disp_from_fc = max(max_obj_disp_from_fc, dfc)
            if min_e_after_fc is None or g1["e_pos_mm"] < min_e_after_fc:
                min_e_after_fc = float(g1["e_pos_mm"])
            if g1["e_pos_mm"] <= 12:
                ever_e12 = True
            if g1["e_pos_mm"] <= 9:
                ever_e9 = True
                if t_to_e9 is None:
                    t_to_e9 = int(g1["t"] - fc_t)

            if g1["has_pad_contact"]:
                had_contact = True
            elif had_contact and not ever_opp:
                lost_before_opp = True

            if g1["bilateral"] and not ever_bil:
                ever_bil = True
                t_to_bil = int(g1["t"] - fc_t)
            if g1["opposing"] and not ever_opp:
                ever_opp = True
                t_to_opp = int(g1["t"] - fc_t)
                min_e_after_opp = float(g1["e_pos_mm"])
            if ever_opp and (min_e_after_opp is None or g1["e_pos_mm"] < min_e_after_opp):
                min_e_after_opp = float(g1["e_pos_mm"])
            if g1["min_pad"] >= 4:
                ever_pad4 = True

            # switches before bil / window end
            if not ever_bil:
                decisions_before_bil += 1
                if len(post_fams) >= 2 and post_fams[-1] != post_fams[-2]:
                    switches_before_bil += 1

            in_window = int(g1["t"] - fc_t) <= FC_WINDOW
            if in_window and g1["correct_sidewall"]:
                ever_correct_sw_in_window = True
            if record_fc_window and in_window:
                fc_window_trace.append(
                    {
                        "t": g1["t"],
                        "dt": int(g1["t"] - fc_t),
                        "e_pos_mm": g1["e_pos_mm"],
                        "theta_deg": g1["theta_deg"],
                        "nL": g1["nL"],
                        "nR": g1["nR"],
                        "bilateral": g1["bilateral"],
                        "opposing": g1["opposing"],
                        "correct_sidewall": g1["correct_sidewall"],
                        "force_sum": g1["force_sum"],
                        "obj_disp": float(np.linalg.norm(np.asarray(g1["obj_pos"]) - obj_at_fc))
                        if obj_at_fc is not None
                        else 0.0,
                        "token": int(tok),
                        "family": fam,
                        "dbg_phase": dbg.get("phase"),
                        "obs_bilateral": int(feat["contact_L"] and feat["contact_R"]),
                    }
                )
            if in_window:
                e_at_end_window = float(g1["e_pos_mm"])
            elif window_end_reason is None:
                if ever_opp:
                    window_end_reason = "opposing"
                elif lost_before_opp:
                    window_end_reason = "contact_lost"
                else:
                    window_end_reason = "window_elapsed"

        preds = evaluate_predicates(
            env.env.sim, ok_s=float(env.mon.tracker.ok_s) if env.mon.tracker else 0.0
        )
        if preds["frame_ok"]:
            ever_frame = True

    term = _r3_terminal(env, info)
    switches_norm = (
        float(switches_before_bil) / max(decisions_before_bil - 1, 1) if decisions_before_bil else 0.0
    )

    fc_rec = {
        "fc_t": fc_t,
        "e_at_fc": e_at_fc,
        "fc_side": fc_side,
        "fc_correct_sidewall": fc_sw,
        "fc_nL": fc_nL,
        "fc_nR": fc_nR,
        "fc_has_contact": int(fc_t is not None),
        "reached_opposing": int(ever_opp),
        "reached_bilateral": int(ever_bil),
        "reached_pad4": int(ever_pad4),
        "reached_e9": int(ever_e9),
        "reached_e12": int(ever_e12),
        "reached_frame_ok": int(ever_frame),
        "t_to_opp": t_to_opp,
        "t_to_bil": t_to_bil,
        "t_to_e9": t_to_e9,
        "min_e_in_window": min_e_after_fc,
        "min_e_after_opposing": min_e_after_opp,
        "e_at_end_window": e_at_end_window,
        "max_obj_disp": max_obj_disp,
        "max_obj_disp_from_fc": max_obj_disp_from_fc,
        "ever_correct_sw_in_window": int(ever_correct_sw_in_window),
        "max_force": max_force,
        "lost_contact_before_opposing": int(lost_before_opp),
        "switches_before_bil_or_end": switches_before_bil,
        "decisions_before_bil": decisions_before_bil,
        "switch_rate_before_bil": switches_norm,
        "window_end_reason": window_end_reason,
        "post_tokens": post_tokens[:24],
        "post_families": post_fams[:24],
        "stop_reason": st.stop_reason,
        "fc_window_trace": fc_window_trace if record_fc_window else [],
    }
    fc_rec["failure_labels"] = classify_fc_failure(fc_rec)

    return {
        "tag": tag,
        "traj_id": start.traj_id,
        "seed": seed,
        "rule": None if rule is None else rule.name,
        **term,
        **{k: v for k, v in fc_rec.items() if k != "fc_window_trace"},
        "fc_window_trace": fc_rec["fc_window_trace"],
        "ever": {
            "opposing_bilateral": int(ever_opp),
            "bilateral": int(ever_bil),
            "min_pad_ge4": int(ever_pad4),
            "e9": int(ever_e9),
            "frame_ok": int(ever_frame),
        },
        "contact_loss": int(lost_before_opp),
    }
