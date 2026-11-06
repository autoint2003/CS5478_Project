"""Phases 6–9 — ranking benchmark, representation compare, aliasing, validity."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np

from experiments.precision.p77_contact_observability.metrics import auroc, auprc, eval_binary_probe
from experiments.precision.p79_matched_cf_coverage.paths import CACHE, OUT, PRIMARY_CANDIDATES, SEED


def _agg_snap_cand(rows: list[dict]) -> dict:
    snap_cand: dict = defaultdict(dict)
    for r in rows:
        snap_cand[r["snap_id"]].setdefault(r["candidate"], []).append(r)
    out = {}
    for sid, cands in snap_cand.items():
        out[sid] = {}
        for c, rs in cands.items():
            out[sid][c] = {
                "r3": float(np.mean([x["r3_ready"] for x in rs])),
                "succ": float(np.mean([x["SUCCESS_RELAXED"] for x in rs])),
                "opp": float(np.mean([x["opposing_bilateral"] for x in rs])),
                "loss": float(np.mean([x["contact_loss"] for x in rs])),
                "traj_id": rs[0]["traj_id"],
                "split": rs[0].get("split", "fit"),
                "stratum": rs[0].get("stratum"),
                "sw0": rs[0].get("sw0"),
                "rep_A": rs[0]["rep_A"],
                "rep_B": rs[0]["rep_B"],
                "rep_C": rs[0]["rep_C"],
                "rep_D": rs[0]["rep_D"],
                "e0": rs[0].get("e0"),
            }
    return out


def _pref_tuple(s: dict) -> tuple:
    return (s["succ"], s["r3"], s["opp"], -s["loss"])


def _best_cand(cands: dict) -> str | None:
    if not cands:
        return None
    # unresolved if all equal
    vals = [_pref_tuple(v) for v in cands.values()]
    if len(set(vals)) == 1:
        return None  # all tie
    return max(cands.keys(), key=lambda c: _pref_tuple(cands[c]))


def build_ranking_benchmark(rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = _agg_snap_cand(rows)
    # IC split: use snap split field; also ensure IC-disjoint
    fit_ics = sorted({v[PRIMARY_CANDIDATES[0]]["traj_id"] for v in snap.values() if v[PRIMARY_CANDIDATES[0]]["split"] != "eval"})
    eval_ics = sorted({v[PRIMARY_CANDIDATES[0]]["traj_id"] for v in snap.values() if v[PRIMARY_CANDIDATES[0]]["split"] == "eval"})
    # If too few eval, re-split by IC
    all_ics = sorted({v[PRIMARY_CANDIDATES[0]]["traj_id"] for v in snap.values()})
    if len(eval_ics) < max(3, len(all_ics) // 5):
        rng = np.random.default_rng(SEED + 21)
        ics = list(all_ics)
        rng.shuffle(ics)
        n_ev = max(3, int(round(len(ics) * 0.35)))
        eval_ics, fit_ics = ics[:n_ev], ics[n_ev:]

    pairs = []
    for sid, cands in snap.items():
        tid = cands[PRIMARY_CANDIDATES[0]]["traj_id"]
        names = list(cands.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ka, kb = _pref_tuple(cands[a]), _pref_tuple(cands[b])
                if ka == kb:
                    label = 0  # tie
                    pref = "tie"
                elif ka > kb:
                    label = 1
                    pref = a
                else:
                    label = -1
                    pref = b
                pairs.append(
                    {
                        "snap_id": sid,
                        "traj_id": tid,
                        "a": a,
                        "b": b,
                        "pref": pref,
                        "label": label,
                        "split": "eval" if tid in set(eval_ics) else "fit",
                        "stratum": cands[a].get("stratum"),
                        "sw0": cands[a].get("sw0"),
                        "delta_r3": abs(cands[a]["r3"] - cands[b]["r3"]),
                        "delta_succ": abs(cands[a]["succ"] - cands[b]["succ"]),
                    }
                )

    bench = {
        "n_snaps": len(snap),
        "n_ics": len(all_ics),
        "n_fit_ics": len(fit_ics),
        "n_eval_ics": len(eval_ics),
        "fit_ics": fit_ics,
        "eval_ics": eval_ics,
        "n_pairs": len(pairs),
        "n_tie_pairs": sum(1 for p in pairs if p["pref"] == "tie"),
        "n_ranked_pairs": sum(1 for p in pairs if p["pref"] != "tie"),
        "candidates": list(PRIMARY_CANDIDATES),
        "preference_definition": "lexicographic: SUCCESS_RELAXED > r3_ready > opposing > -contact_loss; ties if equal",
        "note": "Uncertain/equal outcomes recorded as ties. Same candidate set for all representation comparisons.",
    }
    (CACHE / "ranking_snap.pkl").write_bytes(pickle.dumps({"snap": snap, "pairs": pairs, "bench": bench}, protocol=pickle.HIGHEST_PROTOCOL))
    (OUT / "ACTION_RANKING_BENCHMARK.md").write_text(
        "# Action Ranking Benchmark (P79 Phase 6)\n\n```json\n" + json.dumps(bench, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return {"snap": snap, "pairs": pairs, "bench": bench}


def _onehot(name: str, names: list[str]) -> np.ndarray:
    v = np.zeros(len(names), float)
    v[names.index(name)] = 1.0
    return v


def _fit_score(X: np.ndarray, y: np.ndarray, lam: float = 1e-2):
    A = np.column_stack([X, np.ones(len(X))])
    coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return coef


def _score(coef, x):
    return float(np.concatenate([x, [1.0]]) @ coef)


def representation_comparison(rows: list[dict], snap: dict, pairs: list[dict], bench: dict) -> dict:
    names = list(PRIMARY_CANDIDATES)
    fit_ics, eval_ics = set(bench["fit_ics"]), set(bench["eval_ics"])

    # Action-outcome probes on rows
    disc = {"tasks": {}}
    for ykey in ("r3_ready", "SUCCESS_RELAXED", "opposing_bilateral", "contact_loss"):
        disc["tasks"][ykey] = {}
        for rep in ("A", "B", "C", "D"):
            X, y, tids = [], [], []
            for r in rows:
                # use seed_k==0 only to avoid duplicating deterministic seeds
                if r.get("seed_k", 0) != 0:
                    continue
                X.append(np.concatenate([np.asarray(r[f"rep_{rep}"], float), _onehot(r["candidate"], names)]))
                y.append(int(r[ykey]))
                tids.append(r["traj_id"])
            if not X:
                continue
            X = np.stack(X)
            y = np.asarray(y, np.int8)
            tids = np.asarray(tids)
            fit = np.asarray([t in fit_ics for t in tids])
            ev = np.asarray([t in eval_ics for t in tids])
            X_act = np.stack([_onehot(r["candidate"], names) for r in rows if r.get("seed_k", 0) == 0])
            X_state = np.stack([np.asarray(r[f"rep_{rep}"], float) for r in rows if r.get("seed_k", 0) == 0])
            disc["tasks"][ykey][rep] = {
                "state_plus_action": eval_binary_probe(X, y, fit, ev),
                "action_only": eval_binary_probe(X_act, y, fit, ev),
                "state_only": eval_binary_probe(X_state, y, fit, ev),
                "n_pos_eval": int(y[ev].sum()) if ev.any() else 0,
                "n_neg_eval": int((1 - y[ev]).sum()) if ev.any() else 0,
            }

    # Pairwise ranking accuracy on held-out ICs (non-tie pairs)
    ranking = {}
    for rep in ("A", "B", "C", "D"):
        # Fit score(state, action) → r3 on fit snaps
        Xs, ys = [], []
        for sid, cands in snap.items():
            tid = cands[names[0]]["traj_id"]
            if tid not in fit_ics:
                continue
            for c, s in cands.items():
                Xs.append(np.concatenate([np.asarray(s[f"rep_{rep}"], float), _onehot(c, names)]))
                ys.append(0.5 * s["succ"] + 0.5 * s["r3"])
        if len(Xs) < 8:
            ranking[rep] = {"insufficient": True, "n_fit": len(Xs)}
            continue
        Xtr = np.stack(Xs)
        ytr = np.asarray(ys, float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        coef = _fit_score((Xtr - mu) / sd, ytr)

        correct = total = 0
        tie_aware_correct = tie_aware_total = 0
        regret_sum = 0.0
        regret_n = 0
        beat_base = base_n = 0
        per_ic = defaultdict(lambda: {"correct": 0, "total": 0})

        for p in pairs:
            if p["traj_id"] not in eval_ics:
                continue
            sid = p["snap_id"]
            cands = snap[sid]
            a, b = p["a"], p["b"]

            def sc(c):
                x = np.concatenate([np.asarray(cands[c][f"rep_{rep}"], float), _onehot(c, names)])
                return _score(coef, (x - mu) / sd)

            sa, sb = sc(a), sc(b)
            if p["pref"] == "tie":
                tie_aware_total += 1
                if abs(sa - sb) <= 0.05 * (abs(sa) + abs(sb) + 1e-6):
                    tie_aware_correct += 1
                continue
            total += 1
            pred = a if sa >= sb else b
            ok = int(pred == p["pref"])
            correct += ok
            per_ic[p["traj_id"]]["correct"] += ok
            per_ic[p["traj_id"]]["total"] += 1
            tie_aware_total += 1
            tie_aware_correct += ok

        # ranking regret + vs baseline on eval snaps
        for sid, cands in snap.items():
            tid = cands[names[0]]["traj_id"]
            if tid not in eval_ics:
                continue
            best = _best_cand(cands)
            if best is None:
                continue

            def sc(c):
                x = np.concatenate([np.asarray(cands[c][f"rep_{rep}"], float), _onehot(c, names)])
                return _score(coef, (x - mu) / sd)

            pred = max(cands.keys(), key=sc)
            regret_sum += float(_pref_tuple(cands[best])[1] - _pref_tuple(cands[pred])[1])  # r3 gap
            regret_n += 1
            if "baseline" in cands:
                base_n += 1
                if _pref_tuple(cands[pred]) > _pref_tuple(cands["baseline"]):
                    beat_base += 1

        # IC bootstrap CI on pairwise acc
        ic_accs = [v["correct"] / v["total"] for v in per_ic.values() if v["total"] > 0]
        boot = None
        if len(ic_accs) >= 3:
            rng = np.random.default_rng(SEED + 33)
            boots = []
            arr = np.asarray(ic_accs, float)
            for _ in range(500):
                samp = rng.choice(arr, size=len(arr), replace=True)
                boots.append(float(samp.mean()))
            boot = {
                "mean": float(np.mean(boots)),
                "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                "n_ics": len(ic_accs),
            }

        ranking[rep] = {
            "pairwise_accuracy": float(correct / total) if total else None,
            "n_pairs": total,
            "tie_aware_accuracy": float(tie_aware_correct / tie_aware_total) if tie_aware_total else None,
            "n_tie_aware": tie_aware_total,
            "mean_r3_regret_vs_best": float(regret_sum / regret_n) if regret_n else None,
            "n_regret_snaps": regret_n,
            "frac_beats_baseline": float(beat_base / base_n) if base_n else None,
            "ic_bootstrap": boot,
            "per_ic": {k: dict(v) for k, v in per_ic.items()},
        }

    # Paired eng vs history: D vs C, B vs A
    def paired_delta(r_hi, r_lo):
        a = ranking.get(r_hi, {}).get("pairwise_accuracy")
        b = ranking.get(r_lo, {}).get("pairwise_accuracy")
        if a is None or b is None:
            return None
        return {
            "point_delta": float(a - b),
            "hi": a,
            "lo": b,
            "ci95_hi": (ranking[r_hi].get("ic_bootstrap") or {}).get("ci95"),
            "ci95_lo": (ranking[r_lo].get("ic_bootstrap") or {}).get("ci95"),
        }

    report = {
        "discrimination": disc,
        "ranking": ranking,
        "paired_B_minus_A": paired_delta("B", "A"),
        "paired_D_minus_C": paired_delta("D", "C"),
        "paired_D_minus_A": paired_delta("D", "A"),
        "summary": {
            "rank_A": ranking.get("A", {}).get("pairwise_accuracy"),
            "rank_B": ranking.get("B", {}).get("pairwise_accuracy"),
            "rank_C": ranking.get("C", {}).get("pairwise_accuracy"),
            "rank_D": ranking.get("D", {}).get("pairwise_accuracy"),
            "n_eval_pairs": ranking.get("A", {}).get("n_pairs") or ranking.get("B", {}).get("n_pairs"),
        },
    }
    (OUT / "REPRESENTATION_COMPARISON.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT / "REPRESENTATION_COMPARISON.md").write_text(
        "# Representation Comparison (P79 Phase 7)\n\n"
        "A=current 39D/27D · B=current+eng · C=history · D=history+eng\n\n```json\n"
        + json.dumps({"summary": report["summary"], "ranking": ranking, "paired": {
            "B-A": report["paired_B_minus_A"],
            "D-C": report["paired_D_minus_C"],
            "D-A": report["paired_D_minus_A"],
        }}, indent=2, default=str)
        + "\n```\n",
        encoding="utf-8",
    )
    return report


def action_relevant_aliasing(snap: dict, pairs: list[dict]) -> dict:
    """Pairs of snaps with similar obs history but different hidden geometry."""
    # Build feature vectors from rep_C (history)
    items = []
    for sid, cands in snap.items():
        s0 = cands[PRIMARY_CANDIDATES[0]]
        items.append(
            {
                "snap_id": sid,
                "traj_id": s0["traj_id"],
                "sw0": int(s0.get("sw0") or 0),
                "feat": np.asarray(s0["rep_C"], float),
                "eng": np.asarray(s0["rep_B"], float),
                "best": _best_cand(cands),
                "cands": cands,
            }
        )
    # Normalize
    if len(items) < 4:
        report = {"insufficient": True, "n": len(items)}
        (OUT / "ACTION_RELEVANT_ALIASING.md").write_text(
            "# Action-Relevant Aliasing (P79 Phase 8)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
            encoding="utf-8",
        )
        return report

    F = np.stack([it["feat"] for it in items])
    mu, sd = F.mean(0), F.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (F - mu) / sd
    # For each item, find nearest neighbor with different sw0
    cats = {"same_pref": 0, "diff_pref": 0, "insufficient_pref": 0}
    pairs_out = []
    eng_sep_diff = []
    hist_sep_diff = []
    eng_sep_same = []
    hist_sep_same = []
    for i, it in enumerate(items):
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[i] = 1e9
        order = np.argsort(d)
        matched = None
        for j in order[:10]:
            if items[j]["sw0"] != it["sw0"] and items[j]["traj_id"] != it["traj_id"]:
                matched = j
                break
        if matched is None:
            continue
        jt = items[matched]
        # similar histories: distance in lowest 25% of cross-IC distances
        if d[matched] > np.quantile(d[d < 1e8], 0.35):
            continue
        bi, bj = it["best"], jt["best"]
        if bi is None or bj is None:
            cat = 3
            cats["insufficient_pref"] += 1
        elif bi == bj:
            cat = 1
            cats["same_pref"] += 1
            eng_sep_same.append(float(np.linalg.norm(it["eng"] - jt["eng"])))
            hist_sep_same.append(float(d[matched]))
        else:
            cat = 2
            cats["diff_pref"] += 1
            eng_sep_diff.append(float(np.linalg.norm(it["eng"] - jt["eng"])))
            hist_sep_diff.append(float(d[matched]))
        pairs_out.append(
            {
                "a": it["snap_id"],
                "b": jt["snap_id"],
                "sw_a": it["sw0"],
                "sw_b": jt["sw0"],
                "best_a": bi,
                "best_b": bj,
                "category": cat,
                "hist_dist": float(d[matched]),
            }
        )

    # Does eng separate cat2 better than hist?
    eng_helps = None
    if eng_sep_diff and eng_sep_same:
        # higher eng distance for diff_pref vs same_pref relative to hist
        eng_helps = bool(np.mean(eng_sep_diff) > np.mean(eng_sep_same) * 1.05)

    report = {
        "n_alias_pairs": len(pairs_out),
        "categories": {
            "1_diff_geom_same_pref": cats["same_pref"],
            "2_diff_geom_diff_pref": cats["diff_pref"],
            "3_insufficient_pref": cats["insufficient_pref"],
        },
        "eng_separates_cat2_better": eng_helps,
        "mean_eng_dist_cat2": float(np.mean(eng_sep_diff)) if eng_sep_diff else None,
        "mean_eng_dist_cat1": float(np.mean(eng_sep_same)) if eng_sep_same else None,
        "mean_hist_dist_cat2": float(np.mean(hist_sep_diff)) if hist_sep_diff else None,
        "action_relevant_aliasing_established": bool(cats["diff_pref"] >= 5 and eng_helps),
        "examples": pairs_out[:15],
        "note": "Privileged sidewall used only for retrospective pairing. Question is action preference, not geometry AUROC.",
    }
    (OUT / "ACTION_RELEVANT_ALIASING.md").write_text(
        "# Action-Relevant Aliasing (P79 Phase 8)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "ACTION_RELEVANT_ALIASING.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def statistical_validity(rows: list[dict], outcome: dict, rank_report: dict, alias: dict) -> dict:
    ics = sorted({r["traj_id"] for r in rows})
    # label counts
    labels = {}
    for ykey in ("r3_ready", "SUCCESS_RELAXED", "opposing_bilateral", "contact_loss"):
        ys = [r[ykey] for r in rows if r.get("seed_k", 0) == 0]
        labels[ykey] = {"n": len(ys), "n_pos": int(sum(ys)), "n_neg": int(len(ys) - sum(ys))}

    # natural vs enriched (disagreement snaps)
    snap_ids_disagree = set()
    if (CACHE / "comparisons.pkl").exists():
        comps = pickle.loads((CACHE / "comparisons.pkl").read_bytes())
        for c in comps:
            if c["pref"] != "tie":
                snap_ids_disagree.add(c["snap_id"])

    sw_correct = [r for r in rows if r.get("sw0") == 1 and r.get("seed_k", 0) == 0]
    sw_wrong = [r for r in rows if r.get("sw0") == 0 and r.get("seed_k", 0) == 0]

    # IC influence: leave-one-out on disagree rate
    by_ic = defaultdict(list)
    for r in rows:
        by_ic[r["traj_id"]].append(r)

    report = {
        "n_independent_ics": len(ics),
        "effective_sample_note": "Independent unit = traj_id (IC). Seeds/snaps within IC are clustered.",
        "label_counts": labels,
        "natural_population": {
            "n_rows": len(rows),
            "n_snaps": len({r["snap_id"] for r in rows}),
            "outcome_summary": {
                k: outcome.get(k)
                for k in (
                    "n_nontrivial_outcome_disagreements",
                    "disagree_rate",
                    "n_ties",
                    "n_valid_action_comparisons",
                )
            },
        },
        "disagreement_enriched": {
            "n_snaps": len(snap_ids_disagree),
            "note": "Diagnostic only — not reported as natural-policy performance.",
        },
        "sidewall_strata": {
            "correct_n_rows": len(sw_correct),
            "incorrect_n_rows": len(sw_wrong),
            "correct_ics": len({r["traj_id"] for r in sw_correct}),
            "incorrect_ics": len({r["traj_id"] for r in sw_wrong}),
        },
        "ranking_eval_pairs": (rank_report.get("summary") or {}).get("n_eval_pairs"),
        "determinism": outcome.get("determinism"),
        "aliasing": {
            "n_pairs": alias.get("n_alias_pairs"),
            "established": alias.get("action_relevant_aliasing_established"),
        },
        "warnings": [],
    }
    for ykey, lab in labels.items():
        if lab["n_pos"] < 5 or lab["n_neg"] < 5:
            report["warnings"].append(f"Insufficient class balance for {ykey}: {lab}")
    if (rank_report.get("summary") or {}).get("n_eval_pairs") is not None and report["ranking_eval_pairs"] is not None and report["ranking_eval_pairs"] < 8:
        report["warnings"].append("Held-out ranked pairs < 8 — do not claim robust ranking improvement.")
    if len(ics) < 30:
        report["warnings"].append(
            f"Independent ICs={len(ics)} below planning target (≥30 eval / ~80–120 total). Report coverage gap."
        )

    (OUT / "STATISTICAL_VALIDITY.md").write_text(
        "# Statistical Validity (P79 Phase 9)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report


def run_analysis(rows: list[dict], outcome: dict) -> dict:
    rb = build_ranking_benchmark(rows)
    rep = representation_comparison(rows, rb["snap"], rb["pairs"], rb["bench"])
    alias = action_relevant_aliasing(rb["snap"], rb["pairs"])
    valid = statistical_validity(rows, outcome, rep, alias)
    return {"ranking_bench": rb["bench"], "rep": rep, "alias": alias, "valid": valid}


if __name__ == "__main__":
    rows = pickle.loads((CACHE / "cf_rows_pilot.pkl").read_bytes())
    from experiments.precision.p79_matched_cf_coverage.cf_run import analyze_outcomes

    outcome = analyze_outcomes(rows)
    run_analysis(rows, outcome)
