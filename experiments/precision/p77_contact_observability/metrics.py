"""Frozen-feature diagnostic metrics (no neural training)."""

from __future__ import annotations

import numpy as np


def auroc(scores: np.ndarray, y: np.ndarray) -> float | None:
    y = np.asarray(y).astype(np.int8)
    scores = np.asarray(scores, float)
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    correct = 0.0
    for p in pos:
        correct += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return float(correct / (len(pos) * len(neg)))


def auprc(scores: np.ndarray, y: np.ndarray) -> float | None:
    y = np.asarray(y).astype(np.int8)
    scores = np.asarray(scores, float)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-scores)
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    recall = tp / max(y.sum(), 1)
    precision = tp / np.maximum(tp + fp, 1)
    # add (0,1) start
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def precision_recall_at_fpr(scores: np.ndarray, y: np.ndarray, fpr_target: float = 0.1) -> dict:
    y = np.asarray(y).astype(np.int8)
    scores = np.asarray(scores, float)
    neg = scores[y == 0]
    if len(neg) == 0 or y.sum() == 0:
        return {"precision": None, "recall": None, "threshold": None, "fpr": None}
    thr = float(np.quantile(neg, 1.0 - fpr_target))
    pred = scores >= thr
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    return {
        "threshold": thr,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "fpr": float(fp / max(fp + tn, 1)),
        "tpr": float(tp / max(tp + fn, 1)),
    }


def calibration_bins(scores: np.ndarray, y: np.ndarray, n_bins: int = 5) -> list[dict]:
    scores = np.asarray(scores, float)
    y = np.asarray(y).astype(np.float64)
    # map scores to [0,1] via rank
    ranks = scores.argsort().argsort().astype(float) / max(len(scores) - 1, 1)
    qs = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (ranks >= qs[i]) & (ranks <= qs[i + 1] if i == n_bins - 1 else ranks < qs[i + 1])
        if not np.any(m):
            continue
        rows.append(
            {
                "bin": i,
                "n": int(m.sum()),
                "mean_score_rank": float(ranks[m].mean()),
                "empirical_rate": float(y[m].mean()),
            }
        )
    return rows


def ridge_scores(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, lam: float = 1e-2):
    """Fit ridge on train; return test scores. y binary or continuous."""
    ytr = y[tr].astype(np.float64)
    if len(np.unique((ytr > 0.5).astype(int) if set(np.unique(ytr)) <= {0, 1, 0.0, 1.0} else ytr)) < 2 and ytr.ndim == 1:
        # still fit
        pass
    A = np.column_stack([X[tr], np.ones(tr.sum())])
    ata = A.T @ A + lam * np.eye(A.shape[1])
    try:
        coef = np.linalg.solve(ata, A.T @ ytr)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(ata, A.T @ ytr, rcond=None)[0]
    return np.column_stack([X[te], np.ones(te.sum())]) @ coef, coef


def eval_binary_probe(X: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, eval_mask: np.ndarray) -> dict:
    y = np.asarray(y).astype(np.int8)
    prev = float(y[eval_mask].mean()) if eval_mask.any() else None
    if fit_mask.sum() < 10 or eval_mask.sum() < 5 or len(np.unique(y[fit_mask])) < 2:
        return {"n_fit": int(fit_mask.sum()), "n_eval": int(eval_mask.sum()), "prevalence": prev, "insufficient": True}
    mu, sd = X[fit_mask].mean(0), X[fit_mask].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xz = (X - mu) / sd
    scores, _ = ridge_scores(Xz, y, fit_mask, eval_mask)
    ye = y[eval_mask]
    return {
        "n_fit": int(fit_mask.sum()),
        "n_eval": int(eval_mask.sum()),
        "prevalence": prev,
        "auroc": auroc(scores, ye),
        "auprc": auprc(scores, ye),
        "at_fpr10": precision_recall_at_fpr(scores, ye, 0.10),
        "calibration": calibration_bins(scores, ye),
        "insufficient": False,
    }


def detection_latency(ep_scores_by_t: list[tuple[list[float], int | None, int | None]], thr: float) -> dict:
    """Per episode: first time score>=thr vs first loss / opposing time. lead = event_t - detect_t."""
    leads_loss, leads_opp, hits, misses = [], [], 0, 0
    for scores, t_loss, t_opp in ep_scores_by_t:
        det = None
        for t, s in enumerate(scores):
            if s >= thr:
                det = t
                break
        if det is None:
            misses += 1
            continue
        hits += 1
        if t_loss is not None:
            leads_loss.append(float(t_loss - det))
        if t_opp is not None:
            leads_opp.append(float(t_opp - det))
    return {
        "n": len(ep_scores_by_t),
        "detect_rate": float(hits / max(len(ep_scores_by_t), 1)),
        "mean_lead_before_loss": float(np.mean(leads_loss)) if leads_loss else None,
        "mean_lead_before_opp": float(np.mean(leads_opp)) if leads_opp else None,
        "frac_detect_before_loss": float(np.mean([l > 0 for l in leads_loss])) if leads_loss else None,
    }
