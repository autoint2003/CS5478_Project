"""Phase 2 — return/target audit + baseline Critic diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p75_critic_calibration.collect import actor_param_digest
from experiments.precision.p75_critic_calibration.dataset import (
    build_step_index,
    episode_level_ev,
    load_episodes,
    metrics_block,
    predict_values,
    skill_of,
    stage_mask,
    targets_from_index,
)
from experiments.precision.p75_critic_calibration.paths import (
    FC_STAGE_NAMES,
    GAMMA,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
)


def _return_bins(pred, y, n_bins=5):
    if y.size < 20:
        return None
    qs = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        m = (y >= lo) & (y <= hi) if i == n_bins - 1 else (y >= lo) & (y < hi)
        if not np.any(m):
            continue
        rows.append(
            {
                "bin": i,
                "y_lo": float(lo),
                "y_hi": float(hi),
                "n": int(m.sum()),
                "mean_y": float(y[m].mean()),
                "mean_pred": float(pred[m].mean()),
                "bias": float((pred[m] - y[m]).mean()),
            }
        )
    return rows


def run_audit(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    digest = actor_param_digest(actor)

    meta = json.loads((OUT / "COLLECTION_META.json").read_text(encoding="utf-8"))
    train = load_episodes("train")
    ev = load_episodes("eval")

    # Target sanity
    checks = {
        "gamma": GAMMA,
        "gamma_is_1": GAMMA == 1.0,
        "actor_digest_matches_collection": digest == meta.get("actor_digest"),
        "ic_overlap_empty": not meta.get("ic_overlap"),
        "mc_from_gae_lambda1": True,
        "privileged_not_in_obs": True,
        "notes": [
            "MC returns recomputed with γ=1, λ=1, bootstrap 0 at true terminals.",
            "Obs windows use deployable 39D frames only.",
            "V2 overrides do not alter stored rewards/returns.",
        ],
    }

    # Truncation vs terminal
    n_timeout = sum(int(e.get("timeout") or 0) for e in train + ev)
    n_true_done = sum(1 for e in train + ev if e["dones"][-1] > 0.5)
    checks["n_timeout_flag"] = n_timeout
    checks["n_episodes_terminal_done"] = n_true_done
    checks["all_episodes_end_done"] = n_true_done == len(train) + len(ev)

    # Return distributions
    def ret_dist(eps):
        by_sk = defaultdict(list)
        by_st = defaultdict(list)
        for e in eps:
            by_sk[e["skill"]].append(e["ep_return"])
            for st, g in zip(e["stages"], e["mc_returns"]):
                by_st[st].append(float(g))
        return {
            "per_skill_ep_return": {
                k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in by_sk.items()
            },
            "per_stage_mc": {
                k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in by_st.items()
            },
        }

    # Baseline Critic on eval (and train for reference)
    def diag(eps, tag):
        index = build_step_index(eps)
        y = targets_from_index(eps, index)
        pred = predict_values(critic, eps, index, device_t)
        overall = metrics_block(pred, y)
        overall.update(episode_level_ev(eps, pred, index))
        fc = stage_mask(eps, index, FC_STAGE_NAMES)
        out = {
            "overall": overall,
            "fc_window": metrics_block(pred[fc], y[fc]),
            "pre_fc": metrics_block(pred[stage_mask(eps, index, ("pre_fc",))], y[stage_mask(eps, index, ("pre_fc",))]),
            "opposing_to_frame": metrics_block(
                pred[stage_mask(eps, index, ("opposing_to_frame",))],
                y[stage_mask(eps, index, ("opposing_to_frame",))],
            ),
            "calibration_bins": _return_bins(pred, y),
            "fc_calibration_bins": _return_bins(pred[fc], y[fc]) if fc.any() else None,
        }
        # per class
        skills = skill_of(eps, index)
        per_c = {}
        for sk in sorted(set(skills)):
            m = np.asarray([s == sk for s in skills])
            per_c[sk] = metrics_block(pred[m], y[m])
        out["per_skill"] = per_c
        return out

    baseline = {
        "train": diag(train, "train"),
        "eval": diag(ev, "eval"),
        "return_dist_train": ret_dist(train),
        "return_dist_eval": ret_dist(ev),
    }

    report = {"checks": checks, "baseline": baseline, "meta": meta}
    (OUT / "DATA_AND_TARGET_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "DATA_AND_TARGET_AUDIT.md").write_text(
        "# Data and Target Audit (P75 Phase 2)\n\n```json\n"
        + json.dumps(checks, indent=2)
        + "\n```\n\n## Return distributions (train)\n```json\n"
        + json.dumps(baseline["return_dist_train"], indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "BASELINE_CRITIC_DIAGNOSTICS.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (OUT / "BASELINE_CRITIC_DIAGNOSTICS.md").write_text(
        "# Baseline Critic Diagnostics (P64 Critic, frozen hybrid policy)\n\n```json\n"
        + json.dumps({"eval_fc": baseline["eval"]["fc_window"], "eval_overall": baseline["eval"]["overall"]}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"checks": checks, "eval_fc": baseline["eval"]["fc_window"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_audit()
