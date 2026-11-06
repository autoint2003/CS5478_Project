"""Phase 4 — observation aliasing analysis (non-neural; P74 privileged + P75 histories)."""

from __future__ import annotations

import json

import numpy as np

from experiments.precision.p76_value_observability_audit.io_util import load_episodes
from experiments.precision.p76_value_observability_audit.paths import (
    FC_STAGE_NAMES,
    NN_RADIUS_QUANTILE,
    OUT,
    P74_AUDIT,
    SEED,
    SHORT_HIST,
    WINDOW,
)


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


def _pairwise_stats(X: np.ndarray, labels: np.ndarray, outcomes: np.ndarray, *, q: float, seed: int):
    """Among nearest neighbors in feature space, measure label/outcome disagreement."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n < 20:
        return {"n": n, "insufficient": True}
    # subsample for O(n^2) if large
    if n > 800:
        idx = rng.choice(n, size=800, replace=False)
        X = X[idx]
        if labels is not None:
            labels = labels[idx]
        if outcomes is not None:
            outcomes = outcomes[idx]
        n = len(X)
    # radius from random pairwise sample
    sample_i = rng.integers(0, n, size=min(2000, n * 2))
    sample_j = rng.integers(0, n, size=len(sample_i))
    dsample = np.linalg.norm(X[sample_i] - X[sample_j], axis=1)
    radius = float(np.quantile(dsample[dsample > 0], q)) if np.any(dsample > 0) else 1.0

    # For each point, find neighbors within radius (exclude self)
    # Use chunked distances
    sw_disagree = []
    out_disagree = []
    ret_spread = []
    n_with_nn = 0
    for i in range(n):
        d = np.linalg.norm(X - X[i], axis=1)
        nn = (d > 0) & (d <= radius)
        if not np.any(nn):
            continue
        n_with_nn += 1
        if labels is not None:
            sw_disagree.append(float(np.mean(labels[nn] != labels[i])))
        if outcomes is not None:
            out_disagree.append(float(np.mean(outcomes[nn] != outcomes[i])))
            # if we packed returns in outcomes as continuous, skip — outcomes are binary here
    return {
        "n": n,
        "radius": radius,
        "n_with_nn": n_with_nn,
        "frac_with_nn": float(n_with_nn / max(n, 1)),
        "mean_sidewall_disagreement_in_nn": float(np.mean(sw_disagree)) if sw_disagree else None,
        "mean_outcome_disagreement_in_nn": float(np.mean(out_disagree)) if out_disagree else None,
        "interpretation": (
            "High sidewall disagreement among obs-neighbors → hidden-geometry aliasing. "
            "High outcome disagreement with low sidewall disagreement → stochasticity."
        ),
    }


def _feat_current(obs: np.ndarray) -> np.ndarray:
    return obs.astype(np.float64)


def _feat_short(frames: np.ndarray, t: int, k: int = SHORT_HIST) -> np.ndarray:
    lo = max(0, t + 1 - k)
    chunk = frames[lo : t + 1]
    # pad to k * D by repeating first
    if chunk.shape[0] < k:
        pad = np.repeat(chunk[:1], k - chunk.shape[0], axis=0)
        chunk = np.concatenate([pad, chunk], axis=0)
    return chunk.reshape(-1).astype(np.float64)


def _feat_full(frames: np.ndarray, t: int, window: int = WINDOW) -> np.ndarray:
    lo = max(0, t + 1 - window)
    chunk = frames[lo : t + 1]
    # summary stats to keep dim manageable: mean/std/last
    last = chunk[-1]
    mu = chunk.mean(axis=0)
    sd = chunk.std(axis=0)
    return np.concatenate([last, mu, sd]).astype(np.float64)


def analyze_p75_aliasing(eval_eps: list[dict]) -> dict:
    """Match FC-window states by observable history; compare outcomes / returns."""
    rows = []
    for ei, ep in enumerate(eval_eps):
        if ep["skill"] != "C":
            continue
        for t in range(ep["n_steps"]):
            if ep["stages"][t] not in FC_STAGE_NAMES:
                continue
            rows.append(
                {
                    "ei": ei,
                    "t": t,
                    "obs": ep["obs_frames"][t],
                    "frames": ep["obs_frames"],
                    "outcome_opp": int(ep["reached_opposing"]),
                    "outcome_loss": int(ep["lost_contact_after_fc"]),
                    "G": float(ep["mc_returns"][t]),
                    "tok": int(ep["tokens"][t]),
                    "stage": ep["stages"][t],
                }
            )
    if len(rows) < 30:
        return {"n": len(rows), "insufficient": True}

    feats = {
        "A_current_39d": np.stack([_feat_current(r["obs"]) for r in rows]),
        "B_short_hist": np.stack([_feat_short(r["frames"], r["t"]) for r in rows]),
        "C_full_hist_summary": np.stack([_feat_full(r["frames"], r["t"]) for r in rows]),
    }
    outcomes = np.asarray([r["outcome_opp"] for r in rows], np.int8)
    Gs = np.asarray([r["G"] for r in rows], float)
    toks = np.asarray([r["tok"] for r in rows], np.int64)

    out = {"n_fc_steps_C": len(rows), "representations": {}}
    for name, Xraw in feats.items():
        X = _zscore(Xraw)
        # no privileged sidewall in P75 → labels=None; outcome disagreement
        stats = _pairwise_stats(X, None, outcomes, q=NN_RADIUS_QUANTILE, seed=SEED)
        # return spread among neighbors
        rng = np.random.default_rng(SEED + 1)
        n = len(X)
        idx = np.arange(n) if n <= 800 else rng.choice(n, size=800, replace=False)
        Xs, Gs_s, out_s, tok_s = X[idx], Gs[idx], outcomes[idx], toks[idx]
        dsample = np.linalg.norm(Xs[rng.integers(0, len(Xs), 1500)] - Xs[rng.integers(0, len(Xs), 1500)], axis=1)
        radius = float(np.quantile(dsample[dsample > 0], NN_RADIUS_QUANTILE))
        spreads, act_diff = [], []
        for i in range(len(Xs)):
            d = np.linalg.norm(Xs - Xs[i], axis=1)
            nn = (d > 0) & (d <= radius)
            if nn.sum() < 2:
                continue
            spreads.append(float(np.std(Gs_s[nn])))
            act_diff.append(float(np.mean(tok_s[nn] != tok_s[i])))
        # within-IC vs across-IC outcome disagreement among neighbors
        out[name] = {
            **stats,
            "mean_nn_return_std": float(np.mean(spreads)) if spreads else None,
            "mean_nn_action_disagreement": float(np.mean(act_diff)) if act_diff else None,
            "global_return_std": float(np.std(Gs)),
            "global_opp_rate": float(np.mean(outcomes)),
        }
        out["representations"][name] = out[name]
    # distinguishable success vs fail before loss: AUROC on features
    from experiments.precision.p76_value_observability_audit.bottleneck import auroc_linear

    for name, Xraw in feats.items():
        X = _zscore(Xraw)
        out["representations"][name]["auroc_opp"] = auroc_linear(X, outcomes)
    return out


def analyze_p74_privileged() -> dict:
    """P74 AUDIT_RAW: obs neighbors with privileged correct_sidewall labels."""
    if not P74_AUDIT.exists():
        return {"available": False}
    raw = json.loads(P74_AUDIT.read_text(encoding="utf-8"))
    steps = []
    for ep in raw["rows"]:
        reached = int(ep.get("reached_opposing") or 0)
        for s in ep.get("steps") or []:
            w = str(s.get("window") or "")
            if w not in ("fc_to_opposing", "fc_to_contact_loss"):
                continue
            vec = s.get("obs_vec8")
            if not vec:
                continue
            steps.append(
                {
                    "x": np.asarray(vec, float),
                    "sw": int(s.get("correct_sidewall") or 0),
                    "opp": int(s.get("opposing") or 0),
                    "reached": reached,
                    "G": float(s.get("mc_return") if s.get("mc_return") is not None else 0.0),
                    "exec": int(s.get("executed") or 0),
                }
            )
    X = _zscore(np.stack([s["x"] for s in steps]))
    sw = np.asarray([s["sw"] for s in steps], np.int8)
    reached = np.asarray([s["reached"] for s in steps], np.int8)
    stats = _pairwise_stats(X, sw, reached, q=NN_RADIUS_QUANTILE, seed=SEED + 7)
    # Rare-positive conditional: among sidewall=1 anchors, do obs-neighbors share sidewall?
    rng = np.random.default_rng(SEED + 9)
    pos = np.where(sw == 1)[0]
    cond = None
    if len(pos) >= 5:
        dsample = np.linalg.norm(
            X[rng.integers(0, len(X), 1500)] - X[rng.integers(0, len(X), 1500)], axis=1
        )
        radius = float(np.quantile(dsample[dsample > 0], NN_RADIUS_QUANTILE))
        disag, nn_reach_mix = [], []
        for i in pos:
            d = np.linalg.norm(X - X[i], axis=1)
            nn = (d > 0) & (d <= radius)
            if not np.any(nn):
                continue
            disag.append(float(np.mean(sw[nn] != 1)))
            nn_reach_mix.append(float(np.mean(reached[nn] != reached[i])))
        cond = {
            "n_sidewall_pos": int(len(pos)),
            "n_with_nn": int(len(disag)),
            "radius": radius,
            "mean_frac_neighbors_without_sidewall": float(np.mean(disag)) if disag else None,
            "mean_outcome_disagreement_from_pos": float(np.mean(nn_reach_mix)) if nn_reach_mix else None,
            "note": (
                "Unconditional NN disagreement is depressed by rare sidewall positives. "
                "Conditional metric: fraction of obs-neighbors of sidewall=1 states that lack sidewall."
            ),
        }
    alias_frac = stats.get("mean_sidewall_disagreement_in_nn")
    cond_frac = None if cond is None else cond.get("mean_frac_neighbors_without_sidewall")
    # Strong if unconditional high OR conditional shows pos anchors sit in mostly sidewall=0 neighborhoods
    strong = bool(
        (alias_frac is not None and alias_frac > 0.15)
        or (cond_frac is not None and cond_frac > 0.5 and (cond or {}).get("n_with_nn", 0) >= 5)
    )
    stoch_frac = stats.get("mean_outcome_disagreement_in_nn")
    return {
        "available": True,
        "n_contact_steps": len(steps),
        "sidewall_rate": float(np.mean(sw)),
        "neighbor_stats": stats,
        "sidewall_positive_conditional": cond,
        "aliasing_evidence": {
            "sidewall_disagreement_among_obs_neighbors": alias_frac,
            "sidewall_pos_neighbor_without_sidewall": cond_frac,
            "strong_aliasing": strong,
            "outcome_disagreement_among_obs_neighbors": stoch_frac,
            "note": (
                "If sidewall disagreement is high in deployable-obs neighborhoods, "
                "hidden geometry aliases observations. Rare-positive conditional avoids "
                "base-rate masking."
            ),
        },
    }


def run_aliasing() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    eval_eps = load_episodes("eval")
    p75 = analyze_p75_aliasing(eval_eps)
    p74 = analyze_p74_privileged()
    report = {
        "p75_observable_histories": p75,
        "p74_privileged_sidewall": p74,
        "synthesis": {
            "hidden_sidewall_not_in_obs": True,
            "p74_strong_aliasing": (p74.get("aliasing_evidence") or {}).get("strong_aliasing"),
            "sidewall_pos_neighbor_without_sidewall": (p74.get("aliasing_evidence") or {}).get(
                "sidewall_pos_neighbor_without_sidewall"
            ),
            "p75_outcome_disagreement": {
                k: (v or {}).get("mean_outcome_disagreement_in_nn")
                for k, v in (p75.get("representations") or {}).items()
            },
        },
    }
    (OUT / "OBSERVATION_ALIASING.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "OBSERVATION_ALIASING.md").write_text(
        "# Observation Aliasing (P76 Phase 4)\n\nNon-neural nearest-neighbor diagnostics.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(report["synthesis"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_aliasing()
