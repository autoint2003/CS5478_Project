"""Shared rich rollout for P71 (geometry stages + R3 funnel)."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs, obs_features
from experiments.precision.p71_contact_safe_alignment.controllers import CtrlState, Spec, select_token


def _snap(env) -> dict[str, Any]:
    g = relative_geometry(env.env.sim)
    tr = env.mon.tracker
    ok_s = float(tr.ok_s) if tr else 0.0
    preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
    return {
        "t": int(env.t),
        "e_pos_mm": float(g["e_pos_mm"]),
        "theta_deg": float(g["theta_deg"]),
        "correct_sidewall": int(g["correct_sidewall"]),
        "min_pad": int(g["min_pad"]),
        "nL": int(g["nL"]),
        "nR": int(g["nR"]),
        "bilateral": int(g["bilateral"]),
        "has_pad_contact": int(g["has_pad_contact"]),
        "frame_ok": int(preds["frame_ok"]),
        "margin_ok": int(preds["margin_ok"]),
        "ok_s": ok_s,
        "obj_pos": list(g["obj_pos"]),
        "hand_pos": list(g["hand_pos"]),
        "rel_dist_m": float(g["rel_dist_m"]),
        "vrel": float(g.get("vrel") or 0),
    }


def rollout(
    env,
    actor,
    critic,
    start,
    device,
    *,
    seed: int,
    spec: Spec | None,
    tag: str = "",
) -> dict[str, Any]:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    st = CtrlState()
    done = False
    info: dict = {}

    e_before_fc = None
    first_contact = None
    first_bil = None
    min_e_before_r3 = None
    at_r3 = None
    at_handoff = None
    best_near = None
    ever_sw = False
    max_pad = 0
    ever_frame = ever_margin = False
    contact_loss = False
    had_contact = False
    forced_log = []
    rem_at_end = None

    while not done:
        m0 = _snap(env)
        if not m0["has_pad_contact"]:
            e_before_fc = {
                "t": m0["t"],
                "e_pos_mm": m0["e_pos_mm"],
                "theta_deg": m0["theta_deg"],
                "min_pad": m0["min_pad"],
            }

        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        if spec is None:
            tok = int(tok_a)
            dbg = {}
        else:
            tok, st, dbg = select_token(
                env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, spec=spec
            )
            if dbg.get("forced") is not None:
                forced_log.append({"t": int(env.t), "tok": int(tok), "e": m0["e_pos_mm"], "phase": st.phase})

        _r, done, info = env.step(int(tok), resid)
        m1 = _snap(env)

        if m1["has_pad_contact"]:
            had_contact = True
        if had_contact and not m1["has_pad_contact"]:
            contact_loss = True

        if first_contact is None and m1["has_pad_contact"]:
            first_contact = {
                "t": m1["t"],
                "e_pos_mm": m1["e_pos_mm"],
                "theta_deg": m1["theta_deg"],
                "min_pad": m1["min_pad"],
                "correct_sidewall": m1["correct_sidewall"],
                "nL": m1["nL"],
                "nR": m1["nR"],
            }

        if first_bil is None and m1["bilateral"] and m1["min_pad"] >= 1:
            first_bil = {
                "t": m1["t"],
                "e_pos_mm": m1["e_pos_mm"],
                "theta_deg": m1["theta_deg"],
                "min_pad": m1["min_pad"],
                "correct_sidewall": m1["correct_sidewall"],
            }

        ever_sw = ever_sw or bool(m1["correct_sidewall"])
        max_pad = max(max_pad, int(m1["min_pad"]))
        ever_frame = ever_frame or bool(m1["frame_ok"])
        ever_margin = ever_margin or bool(m1["margin_ok"])

        if m1["has_pad_contact"]:
            if best_near is None or m1["e_pos_mm"] < best_near["e_pos_mm"]:
                best_near = {
                    "t": m1["t"],
                    "e_pos_mm": m1["e_pos_mm"],
                    "theta_deg": m1["theta_deg"],
                    "min_pad": m1["min_pad"],
                    "frame_ok": m1["frame_ok"],
                    "margin_ok": m1["margin_ok"],
                }
            # min e before R3: track until dwell
            if m1["ok_s"] < 0.5:
                if min_e_before_r3 is None or m1["e_pos_mm"] < min_e_before_r3["e_pos_mm"]:
                    min_e_before_r3 = {
                        "t": m1["t"],
                        "e_pos_mm": m1["e_pos_mm"],
                        "theta_deg": m1["theta_deg"],
                        "min_pad": m1["min_pad"],
                    }

        if at_r3 is None and m1["frame_ok"] and m1["ok_s"] >= 0.5:
            at_r3 = {
                "t": m1["t"],
                "e_pos_mm": m1["e_pos_mm"],
                "theta_deg": m1["theta_deg"],
                "min_pad": m1["min_pad"],
            }

    rem_at_end = int(env.max_steps - env.t)
    handoff = int(info.get("handoff") or 0)
    if handoff and at_handoff is None:
        # approximate with final snap
        mf = _snap(env)
        at_handoff = {
            "t": mf["t"],
            "e_pos_mm": mf["e_pos_mm"],
            "theta_deg": mf["theta_deg"],
            "min_pad": mf["min_pad"],
        }

    tr = env.mon.tracker
    ok_s = float(tr.ok_s) if tr else 0.0
    preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
    # R3-ready = dwell completed with frame_ok, or HANDOFF.
    # Do NOT use "ever saw ok_s>=0.5" (at_r3) — that over-counts transient frames.
    r3 = bool((preds["dwell_complete"] and preds["frame_ok"]) or handoff)

    return {
        "tag": tag,
        "variant": None if spec is None else spec.name,
        "traj_id": start.traj_id,
        "seed": seed,
        "activated": int(st.activated),
        "n_forced": len(forced_log),
        "forced_log": forced_log,
        "abort_reason": st.abort_reason,
        "switch_reason": st.switch_reason,
        "e_before_first_contact": e_before_fc,
        "first_contact": first_contact,
        "first_bilateral": first_bil,
        "min_e_before_r3": min_e_before_r3,
        "at_r3_ready": at_r3,
        "at_handoff": at_handoff,
        "best_near": best_near,
        "correct_sidewall": int(ever_sw),
        "min_pad_ge4": int(max_pad >= 4),
        "max_min_pad": int(max_pad),
        "frame_ok_ever": int(ever_frame),
        "margin_ok_ever": int(ever_margin),
        "r3_ready": int(r3),
        "handoff": handoff,
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "contact_loss": int(contact_loss),
        "time_remaining": rem_at_end,
        "steps_used": int(env.t),
    }


def agg(rows: list[dict]) -> dict:
    def mean(k):
        xs = [float(r[k]) for r in rows if r.get(k) is not None and not isinstance(r.get(k), dict)]
        return float(np.mean(xs)) if xs else None

    def mean_path(*path):
        xs = []
        for r in rows:
            cur = r
            for p in path:
                cur = (cur or {}).get(p) if isinstance(cur, dict) else None
            if cur is not None:
                xs.append(float(cur))
        return float(np.mean(xs)) if xs else None

    def frac_e(path, thr):
        xs = []
        for r in rows:
            cur = r
            for p in path:
                cur = (cur or {}).get(p) if isinstance(cur, dict) else None
            if cur is not None:
                xs.append(float(cur) <= thr)
        return float(np.mean(xs)) if xs else None

    return {
        "n": len(rows),
        "activation_rate": mean("activated"),
        "e_before_fc": mean_path("e_before_first_contact", "e_pos_mm"),
        "e_first_contact": mean_path("first_contact", "e_pos_mm"),
        "frac_fc_le25": frac_e(["first_contact", "e_pos_mm"], 25),
        "frac_fc_le12": frac_e(["first_contact", "e_pos_mm"], 12),
        "frac_fc_le9": frac_e(["first_contact", "e_pos_mm"], 9),
        "e_first_bil": mean_path("first_bilateral", "e_pos_mm"),
        "min_e_before_r3": mean_path("min_e_before_r3", "e_pos_mm"),
        "e_at_r3": mean_path("at_r3_ready", "e_pos_mm"),
        "correct_sidewall": mean("correct_sidewall"),
        "min_pad_ge4": mean("min_pad_ge4"),
        "frame_ok_ever": mean("frame_ok_ever"),
        "margin_ok_ever": mean("margin_ok_ever"),
        "r3_ready": mean("r3_ready"),
        "handoff": mean("handoff"),
        "SUCCESS_RELAXED": mean("SUCCESS_RELAXED"),
        "contact_loss_rate": mean("contact_loss"),
        "mean_time_remaining": mean("time_remaining"),
        "mean_steps": mean("steps_used"),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    k = int(k)
    n = int(n)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0, centre - half)), float(min(1, centre + half)))


def count_r3(rows: list[dict]) -> tuple[int, int]:
    """Exact (k, n) for Wilson CI — prefer this over round(rate * n)."""
    n = len(rows)
    k = int(sum(int(r.get("r3_ready") or 0) for r in rows))
    return k, n
