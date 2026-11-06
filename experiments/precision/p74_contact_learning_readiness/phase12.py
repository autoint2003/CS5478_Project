"""Phase 1–2: Actor ownership + observation sufficiency."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p74_contact_learning_readiness.instrument import WINDOWS, rollout_audit
from experiments.precision.p74_contact_learning_readiness.paths import (
    MAX_EP_STEPS,
    N_ICS,
    N_ROLLOUTS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_SPLIT,
    SEED,
    T_MAX,
)

OBS_KEYS = (
    "obs_contact_L",
    "obs_contact_R",
    "obs_any_contact",
    "obs_e_hat_x",
    "obs_est_valid",
    "obs_aperture",
    "obs_ap_dot",
    "obs_hand_speed",
)


def _starts(ids):
    blocked = blocked_eval_ids()
    by = {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }
    return [by[t] for t in ids if t in by]


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann–Whitney AUROC; labels 0/1."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    if scores.size < 8 or labels.min() == labels.max():
        return None
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    # P(score_pos > score_neg) + 0.5 P(eq)
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg) + 0.5 * np.sum(p == neg))
    return float(wins / (pos.size * neg.size))


def run_phase12(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    ids = list(split["discovery_ics"])[:N_ICS]
    starts = _starts(ids)

    rows = []
    for st in starts:
        for i in range(N_ROLLOUTS):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=99800 + i)
            r = rollout_audit(env, actor, critic, book, st, device_t, seed=seed)
            rows.append(r)
            n_act = sum(1 for s in r["steps"] if not s["v2_override"])
            print(
                f"[p12] {st.traj_id[-20:]} i={i} r3={r['r3_ready']} opp={r['reached_opposing']} "
                f"actor_frac={n_act/max(len(r['steps']),1):.2f} lost={r['lost_contact_after_fc']}",
                flush=True,
            )

    # --- Ownership ---
    own = {w: {"n": 0, "actor": 0, "v2": 0} for w in WINDOWS}
    own_fail_fc = {"n": 0, "actor": 0}  # fc_to_opposing on episodes that never reach opposing
    own_by_outcome = {
        "succ_opp": {"n": 0, "actor": 0},
        "fail_opp": {"n": 0, "actor": 0},
    }
    for r in rows:
        for s in r["steps"]:
            w = s["window"]
            if w not in own:
                continue
            own[w]["n"] += 1
            if s["v2_override"]:
                own[w]["v2"] += 1
            else:
                own[w]["actor"] += 1
            if w == "fc_to_opposing":
                key = "succ_opp" if r["reached_opposing"] else "fail_opp"
                own_by_outcome[key]["n"] += 1
                if not s["v2_override"]:
                    own_by_outcome[key]["actor"] += 1
                if not r["reached_opposing"]:
                    own_fail_fc["n"] += 1
                    if not s["v2_override"]:
                        own_fail_fc["actor"] += 1

    # Retrospective FC→first-loss segment (spec window 3): on episodes that lose
    # contact after FC, steps tagged fc_to_opposing before the loss event.
    pre_loss = {"n": 0, "actor": 0}
    for r in rows:
        if not r.get("lost_contact_after_fc"):
            continue
        for s in r["steps"]:
            if s["window"] != "fc_to_opposing":
                continue
            pre_loss["n"] += 1
            if not s["v2_override"]:
                pre_loss["actor"] += 1

    ownership = {
        "per_window": {
            w: {
                "n_decisions": own[w]["n"],
                "actor_frac": (own[w]["actor"] / own[w]["n"]) if own[w]["n"] else None,
                "v2_override_frac": (own[w]["v2"] / own[w]["n"]) if own[w]["n"] else None,
            }
            for w in WINDOWS
        },
        "fc_to_opposing_fail_only": {
            "n": own_fail_fc["n"],
            "actor_frac": (own_fail_fc["actor"] / own_fail_fc["n"]) if own_fail_fc["n"] else None,
        },
        "fc_to_first_loss_segment": {
            "n": pre_loss["n"],
            "actor_frac": (pre_loss["actor"] / pre_loss["n"]) if pre_loss["n"] else None,
            "note": "Retrospective path: fc_to_opposing steps on episodes that later lose contact.",
        },
        "fc_to_opposing_by_outcome": {
            k: {
                "n": v["n"],
                "actor_frac": (v["actor"] / v["n"]) if v["n"] else None,
            }
            for k, v in own_by_outcome.items()
        },
        "critical_question": {
            "actor_controls_failed_contact_window": (
                (own_fail_fc["actor"] / own_fail_fc["n"]) if own_fail_fc["n"] else 0.0
            )
            >= 0.5,
            "note": "If actor_frac in failed FC→opposing window ≥0.5, ordinary Actor PPO can in principle change those actions.",
        },
    }

    # --- Observability ---
    # At FC→opposing Actor-controlled steps, compare obs for episodes that eventually succeed vs fail opposing
    # Also: steps immediately before contact loss vs matched in-contact steps on success paths
    def collect(pred_outcome_key: str, window: str, actor_only: bool = True):
        xs = {k: [] for k in OBS_KEYS}
        ys = []
        hist_delta = []  # |obs_t - obs_{t-3}| mean abs on obs_vec8
        for r in rows:
            y = int(r[pred_outcome_key]) if pred_outcome_key in r else int(r.get("reached_opposing") or 0)
            prev8 = None
            for s in r["steps"]:
                if s["window"] != window:
                    continue
                if actor_only and s["v2_override"]:
                    continue
                for k in OBS_KEYS:
                    xs[k].append(float(s[k]))
                ys.append(y)
                v8 = np.asarray(s["obs_vec8"], float)
                if prev8 is not None:
                    hist_delta.append(float(np.mean(np.abs(v8 - prev8))))
                prev8 = v8
        return xs, np.asarray(ys, int), hist_delta

    xs_opp, y_opp, _ = collect("reached_opposing", "fc_to_opposing")
    auroc_opp = {k: _auroc(np.asarray(xs_opp[k]), y_opp) for k in OBS_KEYS}

    # Pre-loss: on fail episodes, last in-contact Actor step before first loss vs success in-contact steps
    pre_loss_scores = {k: [] for k in OBS_KEYS}
    pre_loss_y = []
    for r in rows:
        if not r["lost_contact_after_fc"] and not r["reached_opposing"]:
            continue
        steps = [s for s in r["steps"] if s["window"] in ("fc_to_opposing", "fc_to_contact_loss") and not s["v2_override"]]
        if r["lost_contact_after_fc"]:
            # last contact step before loss
            contact_steps = [s for s in steps if s["has_pad_contact"]]
            if contact_steps:
                s = contact_steps[-1]
                for k in OBS_KEYS:
                    pre_loss_scores[k].append(float(s[k]))
                pre_loss_y.append(0)  # will lose / failed path
        if r["reached_opposing"]:
            contact_steps = [s for s in steps if s["has_pad_contact"] and s["window"] == "fc_to_opposing"]
            if contact_steps:
                s = contact_steps[min(2, len(contact_steps) - 1)]
                for k in OBS_KEYS:
                    pre_loss_scores[k].append(float(s[k]))
                pre_loss_y.append(1)

    auroc_preloss = {
        k: _auroc(np.asarray(pre_loss_scores[k]), np.asarray(pre_loss_y, int)) for k in OBS_KEYS
    }

    # Full-history vs current-frame: compare AUROC using mean of last 3 frames' hand_speed/ap_dot
    def hist_feature(r, window, key, L=3):
        buf = []
        out_s, out_y = [], []
        for s in r["steps"]:
            if s["window"] != window or s["v2_override"]:
                continue
            buf.append(float(s[key]))
            if len(buf) > L:
                buf.pop(0)
            out_s.append(float(np.mean(buf)))
            out_y.append(int(r["reached_opposing"]))
        return out_s, out_y

    hist_auroc = {}
    for key in ("obs_hand_speed", "obs_ap_dot", "obs_e_hat_x", "obs_aperture"):
        sc, lb = [], []
        for r in rows:
            a, b = hist_feature(r, "fc_to_opposing", key)
            sc.extend(a)
            lb.extend(b)
        hist_auroc[key] = _auroc(np.asarray(sc), np.asarray(lb))

    # Means succ vs fail at FC window
    def means(outcome_true: bool):
        pool = defaultdict(list)
        for r in rows:
            if bool(r["reached_opposing"]) != outcome_true:
                continue
            for s in r["steps"]:
                if s["window"] != "fc_to_opposing" or s["v2_override"]:
                    continue
                for k in OBS_KEYS:
                    pool[k].append(float(s[k]))
                pool["e_pos_mm"].append(float(s["e_pos_mm"]))
                pool["correct_sidewall"].append(float(s["correct_sidewall"]))
        return {k: float(np.mean(v)) if v else None for k, v in pool.items()}

    observability = {
        "auroc_obs_vs_eventual_opposing_in_fc_window": auroc_opp,
        "auroc_obs_preloss_vs_success_contact": auroc_preloss,
        "auroc_short_history_mean3_vs_opposing": hist_auroc,
        "mean_obs_fc_window_succ_opp": means(True),
        "mean_obs_fc_window_fail_opp": means(False),
        "best_single_frame_auroc": max(
            ((k, v) for k, v in auroc_opp.items() if v is not None), key=lambda x: abs(x[1] - 0.5), default=(None, None)
        ),
        "limitation": None,
    }
    best = observability["best_single_frame_auroc"]
    sw_succ = (observability["mean_obs_fc_window_succ_opp"] or {}).get("correct_sidewall")
    sw_fail = (observability["mean_obs_fc_window_fail_opp"] or {}).get("correct_sidewall")
    sidewall_hidden = True  # never an Actor obs channel
    weak_auroc = best[1] is None or abs(best[1] - 0.5) < 0.08
    moderate_auroc = best[1] is not None and 0.08 <= abs(best[1] - 0.5) < 0.25
    observability["sidewall_privileged_only"] = True
    observability["mean_correct_sidewall_succ_vs_fail"] = {"succ": sw_succ, "fail": sw_fail}
    if weak_auroc:
        observability["limitation"] = (
            "Deployable obs features show weak discrimination (|AUROC-0.5|<0.08) for eventual "
            "opposing formation; correct sidewall geometry is not in the Actor observation."
        )
    elif moderate_auroc and sidewall_hidden:
        observability["limitation"] = (
            f"Best deployable cue ({best[0]} AUROC={best[1]:.3f}) is only moderate. "
            "Correct sidewall is privileged-only and near-absent on fail paths — essential "
            "contact geometry is not fully observable from Actor inputs alone."
        )
    else:
        observability["limitation"] = None

    # slim rows for later phases (drop huge fields)
    slim = []
    for r in rows:
        slim_steps = []
        for s in r["steps"]:
            slim_steps.append({k: v for k, v in s.items() if k != "probs_snapshot"})
        slim.append({**{k: r[k] for k in r if k != "steps"}, "steps": slim_steps})

    report = {
        "n_episodes": len(rows),
        "r3_rate": float(np.mean([r["r3_ready"] for r in rows])),
        "opp_rate": float(np.mean([r["reached_opposing"] for r in rows])),
        "ownership": ownership,
        "observability": observability,
        "discovery_ics": ids,
    }
    (OUT / "AUDIT_RAW.json").write_text(json.dumps({"rows": slim, **report}, indent=2), encoding="utf-8")
    (OUT / "ACTOR_ACTION_OWNERSHIP.json").write_text(json.dumps(ownership, indent=2), encoding="utf-8")
    (OUT / "ACTOR_ACTION_OWNERSHIP.md").write_text(
        "# Actor Action Ownership (P74 Phase 1)\n\n"
        f"Episodes n={report['n_episodes']}  opp={report['opp_rate']:.3f}  r3={report['r3_rate']:.3f}\n\n"
        "## Actor-controlled fraction by window\n```json\n"
        + json.dumps(ownership, indent=2)
        + "\n```\n\n"
        "**Critical:** If `fc_to_opposing_fail_only.actor_frac` is high, PPO on the Actor can target contact-formation failures.\n",
        encoding="utf-8",
    )
    (OUT / "CONTACT_OBSERVABILITY.json").write_text(json.dumps(observability, indent=2), encoding="utf-8")
    (OUT / "CONTACT_OBSERVABILITY.md").write_text(
        "# Contact Observability (P74 Phase 2)\n\n"
        "Privileged geometry used only as labels. Actor/Critic inputs remain deployable obs.\n\n"
        f"```json\n{json.dumps(observability, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"ownership": ownership["per_window"], "fail_fc_actor": ownership["fc_to_opposing_fail_only"], "auroc": auroc_opp}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_phase12()
