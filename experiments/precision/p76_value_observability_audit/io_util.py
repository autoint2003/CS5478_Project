"""Shared I/O: load P75 episodes, critics, cached predictions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.gae import explained_variance
from experiments.precision.p75_critic_calibration.dataset import (
    build_step_index,
    predict_values,
    stage_mask,
    targets_from_index,
)
from experiments.precision.p76_value_observability_audit.paths import (
    CACHE,
    FC_STAGE_NAMES,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P75_CKPT,
    P75_DATA,
    SEED,
)


def load_episodes(split: str) -> list[dict]:
    return pickle.loads((P75_DATA / f"{split}_episodes.pkl").read_bytes())


def load_named_critic(name: str, actor, device):
    if name == "P64":
        return load_critic(P64_STEP4_CRITIC, actor, device)
    path = P75_CKPT / f"critic_{name}.pt"
    crit = load_critic(P64_STEP4_CRITIC, actor, device)
    blob = torch.load(path, map_location=device, weights_only=False)
    crit.load_state_dict(blob["critic"])
    return crit


def ensure_preds(name: str, episodes: list[dict], split: str, device, *, force: bool = False) -> dict:
    """Cache raw Critic predictions for a split."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"preds_{name}_{split}.npz"
    if path.exists() and not force:
        return dict(np.load(path, allow_pickle=False))
    index = build_step_index(episodes)
    actor, _, _, _ = load_actor(P64_STEP4_ACTOR, device)
    for p in actor.parameters():
        p.requires_grad_(False)
    critic = load_named_critic(name, actor, device)
    pred = predict_values(critic, episodes, index, device)
    y = targets_from_index(episodes, index)
    fc = stage_mask(episodes, index, FC_STAGE_NAMES)
    skills = np.asarray([episodes[ei]["skill"] for ei, _ in index])
    stages = np.asarray([episodes[ei]["stages"][t] for ei, t in index])
    ep_idx = np.asarray([ei for ei, _ in index], np.int32)
    t_idx = np.asarray([t for _, t in index], np.int32)
    traj = np.asarray([episodes[ei]["traj_id"] for ei, _ in index])
    out = {
        "pred": pred.astype(np.float32),
        "y": y.astype(np.float32),
        "fc": fc.astype(np.bool_),
        "skills": skills,
        "stages": stages,
        "ep_idx": ep_idx,
        "t_idx": t_idx,
        "traj": traj,
    }
    # np.savez can't store object arrays of str easily with allow_pickle=False for all —
    # save numeric + separate json for traj/skills/stages
    np.savez_compressed(
        path,
        pred=out["pred"],
        y=out["y"],
        fc=out["fc"],
        ep_idx=out["ep_idx"],
        t_idx=out["t_idx"],
    )
    meta = {
        "skills": skills.tolist(),
        "stages": stages.tolist(),
        "traj": traj.tolist(),
        "n": int(len(pred)),
        "name": name,
        "split": split,
    }
    (CACHE / f"preds_{name}_{split}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    out["skills"] = skills
    out["stages"] = stages
    out["traj"] = traj
    return out


def load_preds(name: str, split: str) -> dict:
    path = CACHE / f"preds_{name}_{split}.npz"
    meta = json.loads((CACHE / f"preds_{name}_{split}_meta.json").read_text(encoding="utf-8"))
    z = np.load(path)
    return {
        "pred": z["pred"],
        "y": z["y"],
        "fc": z["fc"],
        "ep_idx": z["ep_idx"],
        "t_idx": z["t_idx"],
        "skills": np.asarray(meta["skills"]),
        "stages": np.asarray(meta["stages"]),
        "traj": np.asarray(meta["traj"]),
    }


def metrics_block(pred: np.ndarray, y: np.ndarray) -> dict:
    if pred.size == 0:
        return {"n": 0}
    err = pred - y
    return {
        "n": int(pred.size),
        "ev": float(explained_variance(pred, y)),
        "corr": float(np.corrcoef(pred, y)[0, 1])
        if pred.size > 2 and np.std(y) > 1e-9 and np.std(pred) > 1e-9
        else None,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "pred_std": float(np.std(pred)),
        "y_std": float(np.std(y)),
        "pred_mean": float(np.mean(pred)),
        "y_mean": float(np.mean(y)),
        "pred_var_over_y_var": float(np.var(pred) / max(np.var(y), 1e-12)),
    }


def ic_level_ev(pred, y, traj_ids) -> dict:
    from collections import defaultdict

    by = defaultdict(lambda: {"p": [], "y": []})
    for i, tid in enumerate(traj_ids):
        by[tid]["p"].append(float(pred[i]))
        by[tid]["y"].append(float(y[i]))
    evs = []
    for v in by.values():
        p = np.asarray(v["p"])
        yy = np.asarray(v["y"])
        if yy.size >= 4 and np.var(yy) > 1e-9:
            evs.append(explained_variance(p, yy))
    if not evs:
        return {"n_ics": len(by), "mean_ic_ev": None, "ci95": None}
    rng = np.random.default_rng(SEED)
    arr = np.asarray(evs, float)
    boots = []
    for _ in range(400):
        idx = rng.integers(0, len(arr), size=len(arr))
        boots.append(float(arr[idx].mean()))
    return {
        "n_ics": len(by),
        "n_ics_with_ev": len(evs),
        "mean_ic_ev": float(arr.mean()),
        "median_ic_ev": float(np.median(arr)),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
    }


def fit_affine(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares V_cal = a * V + b."""
    if pred.size < 2:
        return 1.0, 0.0
    # [V, 1] @ [a, b]^T = y
    A = np.column_stack([pred.astype(np.float64), np.ones(len(pred))])
    coef, _, _, _ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
    return float(coef[0]), float(coef[1])


def fit_bias(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return 1.0, float(np.mean(y) - np.mean(pred))


def apply_affine(pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return (a * pred + b).astype(np.float32)


def split_train_ics(train_eps: list[dict], *, val_frac: float, seed: int):
    """IC-disjoint fit vs calibration-validation within P75 train."""
    rng = np.random.default_rng(seed)
    ics_by: dict[str, set] = {}
    for ep in train_eps:
        ics_by.setdefault(ep["skill"], set()).add(ep["traj_id"])
    fit_ics, val_ics = set(), set()
    for ics in ics_by.values():
        lst = sorted(ics)
        rng.shuffle(lst)
        n_val = max(1, int(round(len(lst) * val_frac))) if len(lst) > 3 else max(1, len(lst) // 4)
        val_ics.update(lst[:n_val])
        fit_ics.update(lst[n_val:])
    return fit_ics, val_ics

