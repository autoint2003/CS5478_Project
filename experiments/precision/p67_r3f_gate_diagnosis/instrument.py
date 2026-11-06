"""Instrumented C rollouts: per-step R3 predicates + stage events."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import (
    PREDICATE_NAMES,
    evaluate_predicates,
    first_false_after_near,
    gate_spec,
)
from experiments.precision.p67_r3f_gate_diagnosis.paths import (
    MAX_EP_STEPS,
    N_ICS,
    N_PER_IC,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    T_MAX,
    TRACE_DIR,
)


def _train_c_starts(n: int) -> list:
    blocked = blocked_eval_ids()
    return [
        s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    ][:n]


def rollout_instrumented(env, actor, critic, start, device, *, seed: int) -> dict:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    timeline = []
    first_true: dict[str, int | None] = {k: None for k in PREDICATE_NAMES}
    first_true.update(
        {
            "correct_sidewall": None,
            "sustained_bilateral_5": None,
            "stable_hold": None,
            "near_ready": None,  # frame_ok active with ok_s>0
            "r3_ready": None,
            "handoff": None,
        }
    )
    ever = Counter()
    bil_run = 0
    max_bil_run = 0
    max_ok_s = 0.0
    max_frame_ok_run = 0
    frame_ok_run = 0
    longest_almost = 0.0  # longest continuous frame_ok interval (seconds via ok_s peak)
    near_then_break: list[dict] = []
    prev_near = False
    tokens = []
    done = False
    info: dict[str, Any] = {}

    while not done:
        tok, resid, logp, val = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tokens.append(
            {
                "t": int(env.t),
                "token": int(tok),
                "logp": float(logp),
                "value": float(val),
            }
        )
        r, done, info = env.step(tok, resid)
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr is not None else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s, dwell_s=float(tr.dwell_s) if tr else 0.5)
        m = info.get("metrics") or state_metrics(env.env.sim)
        fn_L = float(m.get("fn_L") or 0)
        fn_R = float(m.get("fn_R") or 0)
        preds["fn_L"] = fn_L
        preds["fn_R"] = fn_R
        preds["fn_asym"] = abs(fn_L - fn_R) / max(fn_L + fn_R, 1e-6)
        preds["vrel"] = float(m.get("vrel") or 0)
        preds["min_pad_metric"] = int(m.get("min_pad") or 0)
        preds["t"] = int(env.t)
        preds["token"] = int(tok)
        preds["value"] = float(val)
        preds["stable_hold"] = int(env.diag.get("stable_hold") or 0)
        preds["recatch"] = int(env.diag.get("recatch") or 0)

        if preds["bilateral"]:
            bil_run += 1
            max_bil_run = max(max_bil_run, bil_run)
        else:
            bil_run = 0
        if preds["frame_ok"]:
            frame_ok_run += 1
            max_frame_ok_run = max(max_frame_ok_run, frame_ok_run)
        else:
            frame_ok_run = 0
        max_ok_s = max(max_ok_s, ok_s)
        longest_almost = max(longest_almost, ok_s)

        for k in PREDICATE_NAMES:
            if preds.get(k) and first_true[k] is None:
                first_true[k] = int(env.t)
            if preds.get(k):
                ever[k] += 1
        if preds["correct_sidewall"]:
            ever["correct_sidewall"] += 1
            if first_true["correct_sidewall"] is None:
                first_true["correct_sidewall"] = int(env.t)
        if max_bil_run >= 5 and first_true["sustained_bilateral_5"] is None:
            first_true["sustained_bilateral_5"] = int(env.t)
            ever["sustained_bilateral_5"] += 1
        if env.diag.get("stable_hold") and first_true["stable_hold"] is None:
            first_true["stable_hold"] = int(env.t)
            ever["stable_hold"] += 1

        near = bool(preds["frame_ok"] and ok_s > 1e-9)
        if near and first_true["near_ready"] is None:
            first_true["near_ready"] = int(env.t)
            ever["near_ready"] += 1
        if preds["dwell_complete"] and preds["frame_ok"] and first_true["r3_ready"] is None:
            first_true["r3_ready"] = int(env.t)
            ever["r3_ready"] += 1

        # Break after near-ready: previous step was near, this step frame_ok false
        if prev_near and not preds["frame_ok"]:
            near_then_break.append(
                {
                    "t": int(env.t),
                    "ok_s_before_break": float(timeline[-1]["ok_s"]) if timeline else 0.0,
                    "false_preds": first_false_after_near(preds),
                    "theta": preds["theta"],
                    "e_pos": preds["e_pos"],
                    "nL": preds["nL"],
                    "nR": preds["nR"],
                    "grasp_kind": preds["grasp_kind"],
                    "fn_asym": preds["fn_asym"],
                    "token": int(tok),
                }
            )
        prev_near = near

        timeline.append(preds)

        if info.get("handoff") and first_true["handoff"] is None:
            first_true["handoff"] = int(env.t)
            ever["handoff"] += 1

    # Episode-level stage flags
    stages = {
        "ever_bilateral": int(any(p["bilateral"] for p in timeline)),
        "ever_correct_sidewall": int(first_true["correct_sidewall"] is not None),
        "sustained_bilateral_5": int(max_bil_run >= 5),
        "sustained_valid_sidewall": int(
            any(
                p["correct_sidewall"] and p["bilateral"] and p["min_pad_ok"]
                for p in timeline
            )
            and max_bil_run >= 5
        ),
        "stable_hold": int(env.diag.get("stable_hold") or 0),
        "near_ready": int(first_true["near_ready"] is not None),
        "r3_ready": int(first_true["r3_ready"] is not None or int(info.get("handoff") or 0)),
        "handoff": int(info.get("handoff") or 0),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "handoff_reason": info.get("handoff_reason"),
    }

    return {
        "traj_id": start.traj_id,
        "seed": seed,
        "stages": stages,
        "first_true": first_true,
        "max_bil_run": max_bil_run,
        "max_frame_ok_run": max_frame_ok_run,
        "max_ok_s": max_ok_s,
        "longest_almost_s": longest_almost,
        "n_near_breaks": len(near_then_break),
        "near_breaks": near_then_break[:20],
        "n_steps": len(timeline),
        "timeline": timeline,
        "tokens": tokens,
        "terminal": {
            "drop": int(info.get("drop") or 0),
            "handoff": int(info.get("handoff") or 0),
            "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
            "fail_reason": info.get("fail_reason"),
            "handoff_reason": info.get("handoff_reason"),
        },
    }


def run_phase1(*, n_ics: int = N_ICS, n_per: int = N_PER_IC, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    starts = _train_c_starts(n_ics)
    episodes = []

    for st in starts:
        for i in range(n_per):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=67000 + i)
            ep = rollout_instrumented(env, actor, critic, st, device_t, seed=seed)
            # drop full timeline from aggregate JSON; keep summary + save sample traces
            slim = {k: v for k, v in ep.items() if k not in ("timeline", "tokens")}
            slim["timeline_len"] = len(ep["timeline"])
            episodes.append(slim)
            if len(episodes) <= 12 or (ep["stages"]["stable_hold"] and not ep["stages"]["r3_ready"]):
                # keep a few full traces for attribution
                tid = f"{st.traj_id}__r{i}".replace("/", "_")[:160]
                (TRACE_DIR / f"{tid}.json").write_text(
                    json.dumps(
                        {
                            "summary": slim,
                            "timeline": ep["timeline"],
                            "tokens": ep["tokens"],
                            "near_breaks": ep["near_breaks"],
                        }
                    ),
                    encoding="utf-8",
                )
            print(
                f"[p67] {st.traj_id[-28:]} i={i} bil={ep['stages']['ever_bilateral']} "
                f"sw={ep['stages']['ever_correct_sidewall']} stab={ep['stages']['stable_hold']} "
                f"near={ep['stages']['near_ready']} r3={ep['stages']['r3_ready']} "
                f"hand={ep['stages']['handoff']} max_ok={ep['max_ok_s']:.3f}",
                flush=True,
            )

    n = max(1, len(episodes))
    stage_keys = [
        "ever_bilateral",
        "ever_correct_sidewall",
        "sustained_bilateral_5",
        "sustained_valid_sidewall",
        "stable_hold",
        "near_ready",
        "r3_ready",
        "handoff",
        "SUCCESS_RELAXED",
    ]
    rates = {k: float(np.mean([e["stages"][k] for e in episodes])) for k in stage_keys}

    def cond(a: str, b: str) -> dict:
        num = sum(1 for e in episodes if e["stages"][a] and e["stages"][b])
        den = sum(1 for e in episodes if e["stages"][a])
        return {"p": (num / den) if den else None, "num": num, "den": den}

    conditionals = {
        f"P({b}|{a})": cond(a, b)
        for a, b in [
            ("ever_bilateral", "ever_correct_sidewall"),
            ("ever_bilateral", "sustained_bilateral_5"),
            ("sustained_bilateral_5", "stable_hold"),
            ("stable_hold", "near_ready"),
            ("stable_hold", "r3_ready"),
            ("sustained_bilateral_5", "r3_ready"),
            ("near_ready", "r3_ready"),
            ("r3_ready", "handoff"),
            ("handoff", "SUCCESS_RELAXED"),
            ("ever_correct_sidewall", "r3_ready"),
            ("sustained_valid_sidewall", "r3_ready"),
        ]
    }

    # Resolve P66 discrepancy narrative
    discrepancy = {
        "p66_reported": {
            "P_r3f_given_sustained_bilateral": 0.207,
            "overall_stable_hold": 0.453,
            "overall_handoff": 0.094,
        },
        "p67_measured": {
            "P_r3_ready_given_sustained_bilateral_5": conditionals["P(r3_ready|sustained_bilateral_5)"],
            "P_r3_ready_given_stable_hold": conditionals["P(r3_ready|stable_hold)"],
            "overall_stable_hold": rates["stable_hold"],
            "overall_r3_ready": rates["r3_ready"],
            "overall_handoff": rates["handoff"],
        },
        "resolution": (
            "Conditional P(R3|sustained)≈0.21 and aggregate (stable≈45%, handoff≈9%) are consistent: "
            "0.45×0.21≈0.095. They are not contradictory. "
            "Also: RecatchTracker stable_hold uses min_pad≥2; R3 frame needs min_pad≥4 — "
            "stable_hold does not imply frame_ok / near_ready."
        ),
    }

    report = {
        "policy": str(P64_STEP4_ACTOR),
        "gate_spec": gate_spec(),
        "n_episodes": n,
        "n_ics": len(starts),
        "rollouts_per_ic": n_per,
        "stage_rates": rates,
        "conditionals": conditionals,
        "discrepancy_resolution": discrepancy,
        "episodes": episodes,
    }
    (OUT / "GATE_PREDICATE_TIMELINES.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    run_phase1()
