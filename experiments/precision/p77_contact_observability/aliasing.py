"""Phase 3 — rare-positive aliasing with enrichment and multi-radius analysis."""

from __future__ import annotations

import json

import numpy as np

from experiments.precision.p77_contact_observability.paths import FC_WINDOWS, OUT, P74_AUDIT, P76_ALIAS, SEED


def _zscore(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


def _neighbor_metrics(X, sw, stage, radii_q, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > 900:
        idx = rng.choice(n, size=900, replace=False)
        X, sw, stage = X[idx], sw[idx], stage[idx]
        n = len(X)
    # pairwise distance sample for radius calibration
    i = rng.integers(0, n, 2500)
    j = rng.integers(0, n, 2500)
    d0 = np.linalg.norm(X[i] - X[j], axis=1)
    d0 = d0[d0 > 0]
    base = float(np.mean(sw))
    out = {"base_rate": base, "n": n, "by_radius": {}}
    for q in radii_q:
        radius = float(np.quantile(d0, q))
        pos = np.where(sw == 1)[0]
        neg = np.where(sw == 0)[0]
        pos_pure, neg_pure, pos_enrich = [], [], []
        matched_pairs = 0
        matched_outcome_divergent = 0  # filled later if outcomes passed
        for i in pos:
            d = np.linalg.norm(X - X[i], axis=1)
            nn = (d > 0) & (d <= radius)
            if not np.any(nn):
                continue
            # positive-neighbor purity = frac of neighbors that are also sidewall=1
            purity = float(np.mean(sw[nn] == 1))
            pos_pure.append(purity)
            pos_enrich.append(purity / max(base, 1e-9))
        for i in neg[: min(len(neg), 400)]:
            d = np.linalg.norm(X - X[i], axis=1)
            nn = (d > 0) & (d <= radius)
            if not np.any(nn):
                continue
            neg_pure.append(float(np.mean(sw[nn] == 0)))
        out["by_radius"][str(q)] = {
            "radius": radius,
            "n_pos_with_nn": len(pos_pure),
            "positive_neighbor_purity": float(np.mean(pos_pure)) if pos_pure else None,
            "negative_neighbor_purity": float(np.mean(neg_pure)) if neg_pure else None,
            "enrichment_over_base": float(np.mean(pos_enrich)) if pos_enrich else None,
            "frac_neighbors_without_sidewall_from_pos": float(1.0 - np.mean(pos_pure)) if pos_pure else None,
        }
    return out


def _matched_pairs(X, sw, reached, tokens, stage, q=0.05, seed=SEED):
    """Obs-similar pairs with different sidewall; compare outcomes/actions."""
    rng = np.random.default_rng(seed)
    n = len(X)
    i = rng.integers(0, n, 2000)
    j = rng.integers(0, n, 2000)
    d0 = np.linalg.norm(X[i] - X[j], axis=1)
    radius = float(np.quantile(d0[d0 > 0], q))
    pairs = []
    pos = np.where(sw == 1)[0]
    for a in pos:
        d = np.linalg.norm(X - X[a], axis=1)
        # nearest neg at same stage
        cand = np.where((sw == 0) & (stage == stage[a]) & (d > 0) & (d <= radius))[0]
        if len(cand) == 0:
            continue
        b = cand[np.argmin(d[cand])]
        pairs.append(
            {
                "dist": float(d[b]),
                "same_stage": True,
                "reached_pos": int(reached[a]),
                "reached_neg": int(reached[b]),
                "tok_pos": int(tokens[a]),
                "tok_neg": int(tokens[b]),
                "action_differs": int(tokens[a] != tokens[b]),
                "outcome_differs": int(reached[a] != reached[b]),
            }
        )
    if not pairs:
        return {"n_pairs": 0, "radius": radius}
    return {
        "n_pairs": len(pairs),
        "radius": radius,
        "frac_outcome_differs": float(np.mean([p["outcome_differs"] for p in pairs])),
        "frac_action_differs": float(np.mean([p["action_differs"] for p in pairs])),
        "mean_reached_when_sw1": float(np.mean([p["reached_pos"] for p in pairs])),
        "mean_reached_when_sw0_matched": float(np.mean([p["reached_neg"] for p in pairs])),
        "interpretation": (
            "Outcome divergence under matched obs+stage with different sidewall supports genuine "
            "geometry aliasing (not only label noise). Action divergence suggests different M111 "
            "choices may be warranted — needs CF evidence for causality."
        ),
    }


def run_aliasing() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(P74_AUDIT.read_text(encoding="utf-8"))
    xs, sw, reached, toks, stages = [], [], [], [], []
    for ep in raw["rows"]:
        for s in ep["steps"]:
            if s.get("window") not in FC_WINDOWS:
                continue
            xs.append(np.asarray(s["obs_vec8"], float))
            sw.append(int(s.get("correct_sidewall") or 0))
            reached.append(int(ep.get("reached_opposing") or 0))
            toks.append(int(s.get("executed") or 0))
            stages.append(str(s.get("window")))
    X = _zscore(np.stack(xs))
    sw = np.asarray(sw, np.int8)
    reached = np.asarray(reached, np.int8)
    toks = np.asarray(toks, np.int64)
    stages = np.asarray(stages)

    multi = _neighbor_metrics(X, sw, stages, radii_q=(0.02, 0.05, 0.10, 0.20))
    # matched stage only subset
    pairs = _matched_pairs(X, sw, reached, toks, stages, q=0.05)

    p76 = json.loads(P76_ALIAS.read_text(encoding="utf-8")) if P76_ALIAS.exists() else {}
    report = {
        "p76_reference": {
            "sidewall_pos_neighbor_without_sidewall": (
                (p76.get("p74_privileged_sidewall") or {}).get("aliasing_evidence") or {}
            ).get("sidewall_pos_neighbor_without_sidewall"),
            "unconditional_disagreement": (
                (p76.get("p74_privileged_sidewall") or {}).get("aliasing_evidence") or {}
            ).get("sidewall_disagreement_among_obs_neighbors"),
        },
        "prevalence": float(sw.mean()),
        "multi_radius": multi,
        "matched_pairs_diff_sidewall": pairs,
        "verdict": {
            "robust_after_base_rate_correction": bool(
                (multi["by_radius"].get("0.05") or {}).get("enrichment_over_base") is not None
                and (multi["by_radius"]["0.05"]["enrichment_over_base"] < 5.0
                     or (multi["by_radius"]["0.05"].get("positive_neighbor_purity") or 0) < 0.4)
            ),
            "note": (
                "Enrichment>>1 with still-low absolute purity ⇒ neighbors are enriched vs base rate "
                "but remain heavily contaminated — aliasing is real but not irreducible without better features."
            ),
        },
    }
    # refine verdict text numbers
    r05 = multi["by_radius"].get("0.05") or {}
    report["verdict"]["at_q05"] = r05
    # Robust aliasing: neighbors of positives are still mostly negative despite enrichment
    report["verdict"]["aliasing_robust"] = bool(
        r05.get("frac_neighbors_without_sidewall_from_pos") is not None
        and r05["frac_neighbors_without_sidewall_from_pos"] > 0.5
        and (r05.get("positive_neighbor_purity") or 1.0) < 0.4
    )

    (OUT / "RARE_POSITIVE_ALIASING.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "RARE_POSITIVE_ALIASING.md").write_text(
        "# Rare-Positive Aliasing (P77 Phase 3)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"prevalence": report["prevalence"], "q05": r05, "pairs": pairs}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_aliasing()
