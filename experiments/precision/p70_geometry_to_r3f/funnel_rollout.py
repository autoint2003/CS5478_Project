"""Phase 1–2: rich geometry→R3F funnel logging across policies / variants."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.controllers import (
    VARIANTS,
    ControllerState,
    VariantSpec,
    eligibility,
    select_token,
)
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs, obs_features
from experiments.precision.p70_geometry_to_r3f.paths import (
    E_POS_THRESHOLDS_MM,
    MAX_EP_STEPS,
    N_FUNNEL_ICS,
    N_FUNNEL_ROLLOUTS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_A_ACTOR,
    P69_B_ACTOR,
    P69_SPLIT,
    SEED,
    T_MAX,
)


def _c_starts(traj_ids: list[str] | None = None, n: int | None = None) -> list:
    blocked = blocked_eval_ids()
    all_c = [
        s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    ]
    by = {s.traj_id: s for s in all_c}
    if traj_ids is not None:
        return [by[t] for t in traj_ids if t in by]
    return all_c[: (n or len(all_c))]


def _load_heldout() -> list[str]:
    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    return list(split.get("heldout_train_ics") or [])[:N_FUNNEL_ICS]


def snapshot_metrics(env) -> dict[str, Any]:
    g = relative_geometry(env.env.sim)
    tr = env.mon.tracker
    ok_s = float(tr.ok_s) if tr else 0.0
    preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
    feat = obs_features(last_obs(env))
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
        "min_pad_ok": int(preds["min_pad_ok"]),
        "r3_pos_ok": int(preds["r3_pos_ok"]),
        "working_pos_ok": int(preds["working_pos_ok"]),
        "ok_s": ok_s,
        "obs_contact": int(feat["any_contact_obs"]),
        "hand_speed": float(feat["hand_speed"]),
        "aperture": float(feat["aperture"]),
        "obj_pos": g["obj_pos"],
    }


def rollout_funnel(
    env,
    actor,
    critic,
    start,
    device,
    *,
    seed: int,
    spec: VariantSpec,
    tag: str,
) -> dict[str, Any]:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    st = ControllerState()
    done = False
    info: dict = {}

    eligible_any = False
    activated = False
    act_t = None
    e_at_act = None
    e_after_seq = None
    first_contact = None
    sustained_bil = None  # first time bilateral & min_pad>=2 for ≥1 step (instant)
    bil_run = 0
    near_ready = None  # closest approach: min e_pos while frame-ish or bilateral
    best_near = None
    ever_sw = False
    max_pad = 0
    ever_frame = ever_margin = ever_r3 = False
    forced_log = []
    e_trace_pre = []

    while not done:
        m0 = snapshot_metrics(env)
        feat = obs_features(last_obs(env))
        elig = eligibility(int(env.t), feat, spec)
        if elig:
            eligible_any = True

        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, dbg = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, spec=spec
        )
        if st.activated and not activated:
            activated = True
            act_t = int(env.t)
            e_at_act = float(m0["e_pos_mm"])
        if dbg.get("forced") is not None:
            forced_log.append({"t": int(env.t), "tok": int(tok), "e_pos_mm": m0["e_pos_mm"], "phase": st.phase})

        if not m0["has_pad_contact"]:
            e_trace_pre.append(m0["e_pos_mm"])

        _r, done, info = env.step(int(tok), resid)
        m1 = snapshot_metrics(env)

        # Capture e_pos when forced sequence completes (or contact ends intervention)
        if activated and e_after_seq is None and forced_log:
            seq_done = st.phase == "done"
            if spec.kind == "fixed" and len(forced_log) >= len(spec.sequence):
                seq_done = True
            if seq_done or m1["has_pad_contact"]:
                e_after_seq = float(m1["e_pos_mm"])

        if first_contact is None and m1["has_pad_contact"]:
            first_contact = {
                "t": int(env.t),
                "e_pos_mm": m1["e_pos_mm"],
                "theta_deg": m1["theta_deg"],
                "correct_sidewall": m1["correct_sidewall"],
                "min_pad": m1["min_pad"],
            }

        if m1["bilateral"] and m1["min_pad"] >= 2:
            bil_run += 1
        else:
            bil_run = 0
        if sustained_bil is None and bil_run >= 3:
            sustained_bil = {
                "t": int(env.t),
                "e_pos_mm": m1["e_pos_mm"],
                "theta_deg": m1["theta_deg"],
                "min_pad": m1["min_pad"],
                "correct_sidewall": m1["correct_sidewall"],
            }

        ever_sw = ever_sw or bool(m1["correct_sidewall"])
        max_pad = max(max_pad, int(m1["min_pad"]))
        ever_frame = ever_frame or bool(m1["frame_ok"])
        ever_margin = ever_margin or bool(m1["margin_ok"])
        if m1["frame_ok"] and m1["ok_s"] > 0:
            near_ready = True
        # closest approach while in contact
        if m1["has_pad_contact"]:
            if best_near is None or m1["e_pos_mm"] < best_near["e_pos_mm"]:
                best_near = {
                    "t": int(env.t),
                    "e_pos_mm": m1["e_pos_mm"],
                    "theta_deg": m1["theta_deg"],
                    "min_pad": m1["min_pad"],
                    "frame_ok": m1["frame_ok"],
                    "margin_ok": m1["margin_ok"],
                }
        if m1["frame_ok"] and m1["ok_s"] >= 0.5:  # dwell-ish
            ever_r3 = True

    # dwell from env
    r3 = bool(ever_r3 or int(info.get("handoff") or 0))
    # also check tracker dwell via last predicates
    tr = env.mon.tracker
    ok_s = float(tr.ok_s) if tr else 0.0
    preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
    if preds["dwell_complete"] and preds["frame_ok"]:
        r3 = True

    causes = []
    if spec.kind != "none":
        if not eligible_any:
            causes.append("A_rarely_eligible")
        elif not activated:
            causes.append("A_eligible_not_activated")
        if activated and act_t is not None and act_t >= 5:
            causes.append("B_late_activation")
        if activated and e_after_seq is not None and e_after_seq > 25:
            causes.append("C_epos_still_above_diag_25")
        if activated and e_after_seq is not None and e_after_seq > 9:
            causes.append("C_epos_above_r3_margin_9")
        if (
            first_contact
            and e_after_seq is not None
            and first_contact["e_pos_mm"] > e_after_seq + 15
        ):
            causes.append("D_alignment_lost_at_contact")
        if ever_sw and max_pad < 4:
            causes.append("E_sidewall_but_minpad_lt4")
        if (
            best_near
            and best_near["e_pos_mm"] <= 25
            and best_near.get("margin_ok")
            and not r3
            and not int(info.get("handoff") or 0)
        ):
            causes.append("F_geometry_ok_then_lost")
        if activated and e_at_act is not None and e_at_act > 150 and (e_after_seq or 999) > 50:
            causes.append("G_large_error_subset_insufficient")

    return {
        "tag": tag,
        "variant": spec.name,
        "traj_id": start.traj_id,
        "seed": seed,
        "eligible": int(eligible_any),
        "activated": int(activated),
        "act_t": act_t,
        "e_at_activation": e_at_act,
        "e_after_sequence": e_after_seq,
        "first_contact": first_contact,
        "sustained_bilateral": sustained_bil,
        "best_near_r3": best_near,
        "correct_sidewall": int(ever_sw),
        "max_min_pad": int(max_pad),
        "min_pad_ge4": int(max_pad >= 4),
        "frame_ok_ever": int(ever_frame),
        "margin_ok_ever": int(ever_margin),
        "near_ready": int(bool(near_ready)),
        "r3_ready": int(r3),
        "handoff": int(info.get("handoff") or 0),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "n_forced": len(forced_log),
        "forced_log": forced_log,
        "precontact_e_min": float(min(e_trace_pre)) if e_trace_pre else None,
        "precontact_e_last": float(e_trace_pre[-1]) if e_trace_pre else None,
        "causes": causes,
        "abort_reason": st.abort_reason,
    }


def _agg(rows: list[dict]) -> dict:
    def mean(k, pred=None):
        xs = []
        for r in rows:
            if pred and not pred(r):
                continue
            v = r.get(k)
            if v is None:
                continue
            if isinstance(v, dict):
                continue
            xs.append(float(v))
        return float(np.mean(xs)) if xs else None

    def mean_nested(path):
        xs = []
        for r in rows:
            cur = r
            for p in path:
                cur = (cur or {}).get(p) if isinstance(cur, dict) else None
            if cur is not None:
                xs.append(float(cur))
        return float(np.mean(xs)) if xs else None

    return {
        "n": len(rows),
        "eligible_rate": mean("eligible"),
        "activation_rate": mean("activated"),
        "e_at_activation": mean("e_at_activation"),
        "e_after_sequence": mean("e_after_sequence"),
        "e_first_contact": mean_nested(["first_contact", "e_pos_mm"]),
        "e_sustained_bil": mean_nested(["sustained_bilateral", "e_pos_mm"]),
        "e_best_near": mean_nested(["best_near_r3", "e_pos_mm"]),
        "correct_sidewall": mean("correct_sidewall"),
        "min_pad_ge4": mean("min_pad_ge4"),
        "frame_ok_ever": mean("frame_ok_ever"),
        "margin_ok_ever": mean("margin_ok_ever"),
        "r3_ready": mean("r3_ready"),
        "handoff": mean("handoff"),
        "SUCCESS_RELAXED": mean("SUCCESS_RELAXED"),
    }


def absolute_threshold_analysis(rows: list[dict]) -> dict:
    """Phase 2: absolute e_pos fractions at key stages."""
    stages = {
        "after_sequence": lambda r: r.get("e_after_sequence"),
        "first_contact": lambda r: (r.get("first_contact") or {}).get("e_pos_mm"),
        "best_near": lambda r: (r.get("best_near_r3") or {}).get("e_pos_mm"),
        "at_activation": lambda r: r.get("e_at_activation"),
    }
    out = {}
    for stage, fn in stages.items():
        vals = [float(fn(r)) for r in rows if fn(r) is not None]
        if not vals:
            out[stage] = None
            continue
        a = np.asarray(vals, float)
        thr = {f"frac_le_{int(t)}mm": float(np.mean(a <= t)) for t in E_POS_THRESHOLDS_MM}
        thr.update({"n": len(a), "mean": float(a.mean()), "median": float(np.median(a))})
        out[stage] = thr

    # Effectiveness by activation-error band
    bands = {"near_le50": [], "moderate_50_150": [], "large_gt150": []}
    for r in rows:
        e0 = r.get("e_at_activation")
        if e0 is None or not r.get("activated"):
            continue
        e1 = r.get("e_after_sequence")
        key = "near_le50" if e0 <= 50 else ("moderate_50_150" if e0 <= 150 else "large_gt150")
        bands[key].append(
            {
                "e0": e0,
                "e1": e1,
                "delta": None if e1 is None else e1 - e0,
                "r3": r.get("r3_ready"),
                "sw": r.get("correct_sidewall"),
            }
        )
    band_sum = {}
    for k, xs in bands.items():
        if not xs:
            band_sum[k] = {"n": 0}
            continue
        d = [x["delta"] for x in xs if x["delta"] is not None]
        band_sum[k] = {
            "n": len(xs),
            "mean_delta": float(np.mean(d)) if d else None,
            "mean_e1": float(np.mean([x["e1"] for x in xs if x["e1"] is not None])) if xs else None,
            "frac_e1_le25": float(np.mean([x["e1"] is not None and x["e1"] <= 25 for x in xs])),
            "frac_e1_le9": float(np.mean([x["e1"] is not None and x["e1"] <= 9 for x in xs])),
            "r3_rate": float(np.mean([x["r3"] for x in xs])),
            "sw_rate": float(np.mean([x["sw"] for x in xs])),
        }
    out["by_activation_band"] = band_sum
    return out


def cause_histogram(rows: list[dict]) -> dict:
    c = defaultdict(int)
    for r in rows:
        for k in r.get("causes") or []:
            c[k] += 1
        if not r.get("causes"):
            c["none_flagged"] += 1
    return dict(c)


def run_phase12(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    critic_path = P64_STEP4_CRITIC

    policies = [
        ("P64", P64_STEP4_ACTOR, "V0_baseline"),
        ("P64_plus_V1", P64_STEP4_ACTOR, "V1_fixed"),
        ("P69A", P69_A_ACTOR, "V0_baseline"),
        ("P69B", P69_B_ACTOR, "V0_baseline"),
    ]

    starts = _c_starts(_load_heldout())
    all_rows = []
    by_tag = defaultdict(list)

    for tag, actor_path, var_name in policies:
        path = Path(actor_path)
        if not path.is_file():
            print(f"[funnel] SKIP missing {tag}: {path}", flush=True)
            continue
        actor, book, rho, _ = load_actor(path, device_t)
        critic = load_critic(critic_path, actor, device_t)
        env = StabilizeGrpoEnv(
            book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
        )
        spec = VARIANTS[var_name]
        for st in starts:
            for i in range(N_FUNNEL_ROLLOUTS):
                seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=97000 + i)
                row = rollout_funnel(env, actor, critic, st, device_t, seed=seed, spec=spec, tag=tag)
                all_rows.append(row)
                by_tag[tag].append(row)
                print(
                    f"[funnel] {tag} {st.traj_id[-24:]} i={i} act={row['activated']} "
                    f"e0={row['e_at_activation']} e1={row['e_after_sequence']} "
                    f"fc={None if not row['first_contact'] else round(row['first_contact']['e_pos_mm'],1)} "
                    f"r3={row['r3_ready']} sw={row['correct_sidewall']} causes={row['causes']}",
                    flush=True,
                )

    summary = {tag: _agg(rows) for tag, rows in by_tag.items()}
    abs_by = {tag: absolute_threshold_analysis(rows) for tag, rows in by_tag.items()}
    causes_by = {tag: cause_histogram(rows) for tag, rows in by_tag.items()}

    # Per-IC for V1 vs P64
    per_ic = []
    base = by_tag.get("P64") or []
    v1 = by_tag.get("P64_plus_V1") or []
    by_ic_b = defaultdict(list)
    by_ic_v = defaultdict(list)
    for r in base:
        by_ic_b[r["traj_id"]].append(r)
    for r in v1:
        by_ic_v[r["traj_id"]].append(r)
    for tid in sorted(set(by_ic_b) | set(by_ic_v)):
        rb, rv = by_ic_b.get(tid, []), by_ic_v.get(tid, [])
        per_ic.append(
            {
                "traj_id": tid,
                "P64": _agg(rb),
                "P64_plus_V1": _agg(rv),
                "delta_r3": (_agg(rv).get("r3_ready") or 0) - (_agg(rb).get("r3_ready") or 0),
                "delta_e_fc": (
                    None
                    if _agg(rv).get("e_first_contact") is None or _agg(rb).get("e_first_contact") is None
                    else _agg(rv)["e_first_contact"] - _agg(rb)["e_first_contact"]
                ),
                "delta_sw": (_agg(rv).get("correct_sidewall") or 0) - (_agg(rb).get("correct_sidewall") or 0),
            }
        )

    report = {
        "summary": summary,
        "absolute_thresholds": abs_by,
        "causes": causes_by,
        "per_ic_P64_vs_V1": per_ic,
        "n_rows": len(all_rows),
    }
    (OUT / "FUNNEL_RAW.json").write_text(
        json.dumps({"rows": all_rows, **report}, indent=2), encoding="utf-8"
    )
    (OUT / "GEOMETRY_TO_R3F_FUNNEL.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "ABSOLUTE_ALIGNMENT_ANALYSIS.json").write_text(
        json.dumps(abs_by, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    run_phase12()
