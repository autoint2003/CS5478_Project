"""Phase 4–5 — independent Critic eval + advantage quality (no Actor updates)."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae
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
    CKPT_DIR,
    FC_STAGE_NAMES,
    GAE_LAMBDA,
    GAMMA,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
)


def _bootstrap_ci(values: list[float], n_boot: int = 400, seed: int = SEED):
    if len(values) < 5:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, float)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(float(arr[idx].mean()))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _ep_fc_evs(episodes, pred, index):
    """Per-episode FC-window EV for IC-level CI."""
    by_ep = defaultdict(lambda: {"p": [], "y": []})
    fc = stage_mask(episodes, index, FC_STAGE_NAMES)
    for k, (ei, t) in enumerate(index):
        if not fc[k]:
            continue
        by_ep[ei]["p"].append(float(pred[k]))
        by_ep[ei]["y"].append(float(episodes[ei]["mc_returns"][t]))
    from experiments.precision.p63_transformer_critic_ppo.gae import explained_variance

    evs = []
    for v in by_ep.values():
        if len(v["y"]) >= 3:
            evs.append(explained_variance(np.asarray(v["p"]), np.asarray(v["y"])))
    return evs


def _load_named_critic(name, actor, device):
    if name == "baseline":
        return load_critic(P64_STEP4_CRITIC, actor, device)
    path = CKPT_DIR / f"critic_{name}.pt"
    crit = load_critic(P64_STEP4_CRITIC, actor, device)  # init shape
    blob = torch.load(path, map_location=device, weights_only=False)
    crit.load_state_dict(blob["critic"])
    return crit


def _eval_critic(name, critic, episodes, device):
    index = build_step_index(episodes)
    y = targets_from_index(episodes, index)
    pred = predict_values(critic, episodes, index, device)
    fc = stage_mask(episodes, index, FC_STAGE_NAMES)
    skills = skill_of(episodes, index)
    out = {
        "overall": metrics_block(pred, y),
        "fc_window": metrics_block(pred[fc], y[fc]),
        "pre_fc": metrics_block(pred[stage_mask(episodes, index, ("pre_fc",))], y[stage_mask(episodes, index, ("pre_fc",))]),
        "opposing_to_frame": metrics_block(
            pred[stage_mask(episodes, index, ("opposing_to_frame",))],
            y[stage_mask(episodes, index, ("opposing_to_frame",))],
        ),
        "frame_to_r3": metrics_block(
            pred[stage_mask(episodes, index, ("frame_to_r3",))],
            y[stage_mask(episodes, index, ("frame_to_r3",))],
        ),
        "episode_level": episode_level_ev(episodes, pred, index),
    }
    ep_fc = _ep_fc_evs(episodes, pred, index)
    out["fc_ep_ev_mean"] = float(np.mean(ep_fc)) if ep_fc else None
    out["fc_ep_ev_wilson_like_ci"] = _bootstrap_ci(ep_fc)
    per = {}
    for sk in sorted(set(skills)):
        m = np.asarray([s == sk for s in skills])
        per[sk] = metrics_block(pred[m], y[m])
        if sk == "C":
            per[sk]["fc"] = metrics_block(pred[m & fc], y[m & fc])
    out["per_skill"] = per
    out["_pred"] = pred
    out["_index"] = index
    out["_y"] = y
    return out


def _advantage_quality(name, critic, episodes, device, pred_cache=None):
    """Recompute GAE(λ=0.95) with Critic values; analyze FC advantages (no Actor update)."""
    adv_fc_succ, adv_fc_fail, adv_pre_loss, adv_pre_opp = [], [], [], []
    means = []
    for ep in episodes:
        T = ep["n_steps"]
        # values at each step from critic predictions — recompute quickly per episode
        index = [(0, t) for t in range(T)]
        # hack: single-ep list
        pred = predict_values(critic, [ep], index, device)
        values = np.concatenate([pred, [0.0]]).astype(np.float32)
        adv, ret = compute_gae(ep["rewards"], values, ep["dones"], GAMMA, GAE_LAMBDA)
        means.append(float(np.mean(adv)))
        for t in range(T):
            st = ep["stages"][t]
            if st not in FC_STAGE_NAMES:
                continue
            a = float(adv[t])
            if ep["reached_opposing"]:
                adv_fc_succ.append(a)
            else:
                adv_fc_fail.append(a)
            if st == "fc_to_contact_loss" or (ep["lost_contact_after_fc"] and st == "fc_to_opposing"):
                # before/around loss path
                if ep["lost_contact_after_fc"] and not ep["reached_opposing"]:
                    adv_pre_loss.append(a)
            if st == "fc_to_opposing" and not ep["reached_opposing"]:
                adv_pre_opp.append(a)

    def pack(xs):
        if not xs:
            return None
        arr = np.asarray(xs, float)
        return {
            "n": len(xs),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "frac_pos": float(np.mean(arr > 0)),
            "p10": float(np.quantile(arr, 0.1)),
            "p90": float(np.quantile(arr, 0.9)),
        }

    # Matched CF consistency (P73): soft_approach improved R3 from FC snaps —
    # check whether mean advantage on FC steps of eventual-success eps > fail
    gap = None
    if adv_fc_succ and adv_fc_fail:
        gap = float(np.mean(adv_fc_succ) - np.mean(adv_fc_fail))

    return {
        "name": name,
        "adv_mean_all_eps": float(np.mean(means)) if means else None,
        "fc_succ": pack(adv_fc_succ),
        "fc_fail": pack(adv_fc_fail),
        "pre_loss_path": pack(adv_pre_loss),
        "pre_opp_fail": pack(adv_pre_opp),
        "fc_adv_gap_succ_minus_fail": gap,
        "note": (
            "Gap is association with eventual opposing under fixed policy — not causal action ranking. "
            "P73 CF remains external evidence that actions can matter."
        ),
    }


def run_evaluate(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, _, _, _ = load_actor(P64_STEP4_ACTOR, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    eval_eps = load_episodes("eval")

    train_cmp = json.loads((OUT / "CRITIC_TRAINING_COMPARISON.json").read_text(encoding="utf-8"))
    names = ["baseline"] + [k for k in train_cmp if k.startswith(("A_", "B_"))]

    calib = {}
    adv = {}
    for name in names:
        print(f"=== Eval {name} ===", flush=True)
        critic = _load_named_critic(name, actor, device_t)
        e = _eval_critic(name, critic, eval_eps, device_t)
        # strip large arrays before save
        slim = {k: v for k, v in e.items() if not k.startswith("_")}
        calib[name] = slim
        adv[name] = _advantage_quality(name, critic, eval_eps, device_t)

    # Cross-class retention vs baseline
    base = calib["baseline"]
    retention = {}
    for name, blob in calib.items():
        if name == "baseline":
            continue
        retention[name] = {
            sk: {
                "ev_delta": None
                if (base["per_skill"].get(sk) or {}).get("ev") is None or (blob["per_skill"].get(sk) or {}).get("ev") is None
                else blob["per_skill"][sk]["ev"] - base["per_skill"][sk]["ev"],
                "bias_delta": None
                if (base["per_skill"].get(sk) or {}).get("bias") is None or (blob["per_skill"].get(sk) or {}).get("bias") is None
                else blob["per_skill"][sk]["bias"] - base["per_skill"][sk]["bias"],
            }
            for sk in base.get("per_skill", {})
        }

    (OUT / "FC_WINDOW_CALIBRATION.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    (OUT / "FC_WINDOW_CALIBRATION.md").write_text(
        "# FC-Window Calibration (P75 Phase 4)\n\nIndependent eval ICs.\n\n```json\n"
        + json.dumps({k: {"fc": v["fc_window"], "overall": v["overall"], "fc_ep_ci": v.get("fc_ep_ev_wilson_like_ci")} for k, v in calib.items()}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "ADVANTAGE_QUALITY.json").write_text(json.dumps(adv, indent=2), encoding="utf-8")
    (OUT / "ADVANTAGE_QUALITY.md").write_text(
        "# Advantage Quality (P75 Phase 5)\n\nGAE γ=1, λ=0.95; Actor not updated.\n\n```json\n"
        + json.dumps(adv, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "CROSS_CLASS_RETENTION.json").write_text(json.dumps(retention, indent=2), encoding="utf-8")
    (OUT / "CROSS_CLASS_RETENTION.md").write_text(
        "# Cross-Class Retention\n\nΔ EV / bias vs baseline P64 Critic on eval ICs.\n\n```json\n"
        + json.dumps(retention, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v["fc_window"] for k, v in calib.items()}, indent=2), flush=True)
    return {"calib": calib, "adv": adv, "retention": retention}


if __name__ == "__main__":
    run_evaluate()
