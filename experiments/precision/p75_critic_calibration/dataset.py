"""Build Critic training windows from stored episode frames."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p63_transformer_critic_ppo.gae import explained_variance
from experiments.precision.p75_critic_calibration.paths import DATA_DIR, FC_STAGE_NAMES, WINDOW


def load_episodes(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}_episodes.pkl"
    return pickle.loads(path.read_bytes())


def build_step_index(episodes: list[dict]) -> list[tuple[int, int]]:
    """List of (ep_idx, t) for every decision step."""
    idx = []
    for ei, ep in enumerate(episodes):
        for t in range(int(ep["n_steps"])):
            idx.append((ei, t))
    return idx


def window_at(ep: dict, t: int, window: int = WINDOW):
    o = ep["obs_frames"]
    p = ep["prev_frames"]
    lo = max(0, t + 1 - window)
    return o[lo : t + 1], p[lo : t + 1]


@torch.no_grad()
def predict_values(critic, episodes: list[dict], index: list[tuple[int, int]], device, *, mb: int = 64) -> np.ndarray:
    preds = []
    critic.eval()
    for start in range(0, len(index), mb):
        chunk = index[start : start + mb]
        obs_list, prev_list = [], []
        for ei, t in chunk:
            o, p = window_at(episodes[ei], t)
            obs_list.append(o)
            prev_list.append(p)
        from experiments.precision.p63_transformer_critic_ppo.sampling import pad_batch

        o, p, lengths, pad_mask = pad_batch(obs_list, prev_list, device)
        vals = critic(o, p, pad_mask=pad_mask)
        last = torch.stack([vals[j, lengths[j] - 1] for j in range(len(chunk))])
        preds.append(last.cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros(0, np.float32)


def targets_from_index(episodes: list[dict], index: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([episodes[ei]["mc_returns"][t] for ei, t in index], np.float32)


def stage_mask(episodes: list[dict], index: list[tuple[int, int]], stages: tuple[str, ...]) -> np.ndarray:
    return np.asarray([episodes[ei]["stages"][t] in stages for ei, t in index], dtype=bool)


def skill_of(episodes: list[dict], index: list[tuple[int, int]]) -> list[str]:
    return [episodes[ei]["skill"] for ei, t in index]


def metrics_block(pred: np.ndarray, y: np.ndarray) -> dict:
    if pred.size == 0:
        return {"n": 0}
    err = pred - y
    return {
        "n": int(pred.size),
        "ev": explained_variance(pred, y),
        "corr": float(np.corrcoef(pred, y)[0, 1]) if pred.size > 2 and np.std(y) > 1e-9 and np.std(pred) > 1e-9 else None,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "pred_mean": float(np.mean(pred)),
        "y_mean": float(np.mean(y)),
    }


def episode_level_ev(episodes: list[dict], pred: np.ndarray, index: list[tuple[int, int]]) -> dict:
    """Aggregate per-episode mean |error| / var for IC-aware reporting."""
    from collections import defaultdict

    by_ep = defaultdict(lambda: {"p": [], "y": []})
    for k, (ei, t) in enumerate(index):
        by_ep[ei]["p"].append(float(pred[k]))
        by_ep[ei]["y"].append(float(episodes[ei]["mc_returns"][t]))
    evs = []
    for v in by_ep.values():
        p = np.asarray(v["p"])
        y = np.asarray(v["y"])
        if y.size >= 4:
            evs.append(explained_variance(p, y))
    return {
        "n_episodes": len(by_ep),
        "mean_ep_ev": float(np.mean(evs)) if evs else None,
        "median_ep_ev": float(np.median(evs)) if evs else None,
    }
