"""Phase 5 — information bottleneck assessment (statistical, non-neural)."""

from __future__ import annotations

import json

import numpy as np

from experiments.precision.p76_value_observability_audit.io_util import load_episodes, load_preds, metrics_block
from experiments.precision.p76_value_observability_audit.paths import FC_STAGE_NAMES, OUT, SEED, SHORT_HIST, WINDOW


def auroc_linear(X: np.ndarray, y: np.ndarray) -> float | None:
    """AUROC of a ridge-linear score fit with leave-one-IC-out style holdout via random split.

    Not a neural Critic — frozen linear probe for distinguishability only.
    """
    y = np.asarray(y).astype(np.int8)
    if y.size < 20 or len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(SEED)
    n = len(y)
    idx = rng.permutation(n)
    n_te = max(5, n // 5)
    te, tr = idx[:n_te], idx[n_te:]
    if len(np.unique(y[tr])) < 2:
        return None
    # ridge
    A = np.column_stack([X[tr], np.ones(len(tr))])
    lam = 1e-2
    ata = A.T @ A + lam * np.eye(A.shape[1])
    coef = np.linalg.solve(ata, A.T @ y[tr].astype(np.float64))
    scores = np.column_stack([X[te], np.ones(len(te))]) @ coef
    return float(_auroc(scores, y[te]))


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann–Whitney
    correct = 0.0
    for p in pos:
        correct += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return correct / (len(pos) * len(neg))


def _feat_pack(ep, t, kind: str) -> np.ndarray:
    frames = ep["obs_frames"]
    if kind == "current":
        return frames[t].astype(np.float64)
    if kind == "short":
        lo = max(0, t + 1 - SHORT_HIST)
        chunk = frames[lo : t + 1]
        if chunk.shape[0] < SHORT_HIST:
            pad = np.repeat(chunk[:1], SHORT_HIST - chunk.shape[0], axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        return chunk.reshape(-1)
    # full summary
    lo = max(0, t + 1 - WINDOW)
    chunk = frames[lo : t + 1]
    return np.concatenate([chunk[-1], chunk.mean(0), chunk.std(0)])


def run_bottleneck() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    eval_eps = load_episodes("eval")
    # FC steps on C
    Xc = {k: [] for k in ("current", "short", "full")}
    y_opp, y_loss, Gs, ep_ids = [], [], [], []
    for ei, ep in enumerate(eval_eps):
        if ep["skill"] != "C":
            continue
        for t in range(ep["n_steps"]):
            if ep["stages"][t] not in FC_STAGE_NAMES:
                continue
            for k in Xc:
                Xc[k].append(_feat_pack(ep, t, k))
            y_opp.append(int(ep["reached_opposing"]))
            y_loss.append(int(ep["lost_contact_after_fc"]))
            Gs.append(float(ep["mc_returns"][t]))
            ep_ids.append(ei)

    y_opp = np.asarray(y_opp)
    y_loss = np.asarray(y_loss)
    Gs = np.asarray(Gs, float)
    report = {
        "n_fc_C": int(len(y_opp)),
        "opp_rate": float(y_opp.mean()) if len(y_opp) else None,
        "loss_rate": float(y_loss.mean()) if len(y_loss) else None,
        "mc_std": float(Gs.std()) if len(Gs) else None,
        "mc_mean": float(Gs.mean()) if len(Gs) else None,
        "distinguishability": {},
    }

    for k, rows in Xc.items():
        X = np.stack(rows)
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xz = (X - mu) / sd
        # return R^2 of linear probe
        A = np.column_stack([Xz, np.ones(len(Xz))])
        coef, *_ = np.linalg.lstsq(A, Gs, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((Gs - pred) ** 2))
        ss_tot = float(np.sum((Gs - Gs.mean()) ** 2)) + 1e-12
        report["distinguishability"][k] = {
            "auroc_eventual_opposing": auroc_linear(Xz, y_opp),
            "auroc_eventual_contact_loss": auroc_linear(Xz, y_loss),
            "linear_return_r2": float(1.0 - ss_res / ss_tot),
            "residual_std": float(np.sqrt(ss_res / max(len(Gs), 1))),
            "mc_std": float(Gs.std()),
        }

    # Residual unpredictability after best linear probe
    best = max(
        report["distinguishability"].items(),
        key=lambda kv: (kv[1].get("linear_return_r2") or -1),
    )
    # Critic residual comparison (calibration error vs irreducible)
    p64 = load_preds("P64", "eval")
    aw = load_preds("A_w1", "eval")
    # load selected affine for A_w1
    ind = json.loads((OUT / "INDEPENDENT_IC_VALIDATION.json").read_text(encoding="utf-8"))
    a = float(ind["results"]["A_w1"]["a"])
    b = float(ind["results"]["A_w1"]["b"])
    fc = p64["fc"] & (p64["skills"] == "C")
    raw_m = metrics_block(p64["pred"][fc], p64["y"][fc])
    aw_raw = metrics_block(aw["pred"][fc], aw["y"][fc])
    aw_cal = metrics_block((a * aw["pred"] + b)[fc], aw["y"][fc])

    # Episode-level: are success/fail distinguishable before contact loss?
    # Use first FC step feature of each C episode
    first_rows, labels = [], []
    for ep in eval_eps:
        if ep["skill"] != "C":
            continue
        ts = [t for t in range(ep["n_steps"]) if ep["stages"][t] in FC_STAGE_NAMES]
        if not ts:
            continue
        t0 = ts[0]
        first_rows.append(_feat_pack(ep, t0, "full"))
        labels.append(int(ep["reached_opposing"]))
    Xf = np.stack(first_rows) if first_rows else np.zeros((0, 1))
    if len(Xf):
        mu, sd = Xf.mean(0), Xf.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        pre_loss_auroc = auroc_linear((Xf - mu) / sd, np.asarray(labels))
    else:
        pre_loss_auroc = None

    # Bootstrap IC-level AUROC uncertainty on episode-first features
    ci = None
    if len(Xf) >= 20:
        rng = np.random.default_rng(SEED)
        labs = np.asarray(labels)
        Xz = (Xf - mu) / sd
        boots = []
        for _ in range(300):
            idx = rng.integers(0, len(Xz), size=len(Xz))
            if len(np.unique(labs[idx])) < 2:
                continue
            boots.append(auroc_linear(Xz[idx], labs[idx]))
        boots = [b for b in boots if b is not None and b == b]
        if boots:
            ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]

    report["components"] = {
        "1_return_unpredictability": {
            "mc_std_fc_C": report["mc_std"],
            "best_linear_r2": best[1].get("linear_return_r2"),
            "best_repr": best[0],
            "residual_std_after_linear": best[1].get("residual_std"),
            "note": "High residual after linear probe ⇒ stochasticity and/or nonlinear/hidden factors.",
        },
        "2_hidden_state_ambiguity": {
            "ref": "See OBSERVATION_ALIASING.md (P74 sidewall neighbor disagreement).",
            "pre_opposing_auroc_at_first_fc": pre_loss_auroc,
            "auroc_ci95_episode": ci,
        },
        "3_action_dependent_variability": {
            "external_p73": "Matched FC counterfactuals: different M111 sequences change outcomes.",
            "implies": "Some return variance is action-dependent, not pure noise.",
        },
        "4_calibration_errors": {
            "P64_fc_C": raw_m,
            "A_w1_raw_fc_C": aw_raw,
            "A_w1_cal_fc_C": aw_cal,
            "note": "Gap between linear-probe ceiling and Critic EV isolates model/calibration error.",
        },
    }
    # Ceiling comparison
    ceiling_r2 = best[1].get("linear_return_r2") or 0.0
    critic_ev = aw_cal.get("ev") if aw_cal.get("n", 0) else raw_m.get("ev")
    report["verdict_bits"] = {
        "linear_probe_ceiling_r2": ceiling_r2,
        "calibrated_critic_ev": critic_ev,
        "calibration_closes_gap_to_ceiling": bool(
            critic_ev is not None and ceiling_r2 is not None and critic_ev > ceiling_r2 - 0.05 and critic_ev > 0.05
        ),
        "obs_moderately_predictive": bool(pre_loss_auroc is not None and pre_loss_auroc > 0.65),
        "obs_strongly_predictive": bool(pre_loss_auroc is not None and pre_loss_auroc > 0.8),
        "claim_info_theoretic_limit": False,
        "claim_note": (
            "Do not claim a hard information-theoretic limit; report empirical distinguishability "
            "and residual variance only."
        ),
    }

    (OUT / "INFORMATION_BOTTLENECK.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "INFORMATION_BOTTLENECK.md").write_text(
        "# Information Bottleneck (P76 Phase 5)\n\nStatistical probes only — no NN training.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(report["verdict_bits"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_bottleneck()
