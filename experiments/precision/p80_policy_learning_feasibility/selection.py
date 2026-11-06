"""Phases 2–4 — static vs conditional selection, reversals, offline value."""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict

import numpy as np

from experiments.precision.p80_policy_learning_feasibility.paths import (
    CACHE,
    OUT,
    P79_CACHE,
    PRIMARY_CANDIDATES,
    SEED,
)


def _pref_tuple(s: dict) -> tuple:
    return (s["succ"], s["r3"], s["opp"], -s["loss"])


def _agg_snap(rows: list[dict]) -> dict:
    snap_cand: dict = defaultdict(dict)
    for r in rows:
        if r.get("seed_k", 0) != 0:
            continue
        snap_cand[r["snap_id"]][r["candidate"]] = {
            "r3": float(r["r3_ready"]),
            "succ": float(r["SUCCESS_RELAXED"]),
            "opp": float(r["opposing_bilateral"]),
            "loss": float(r["contact_loss"]),
            "contact_preserved": float(r["contact_preserved"]),
            "e9": float(r["e_pos_le9"]),
            "traj_id": r["traj_id"],
            "stratum": r.get("stratum"),
            "sw0": r.get("sw0"),
            "rep_C": np.asarray(r["rep_C"], float),
            "kind": r.get("kind"),
        }
    return snap_cand


def _best(cands: dict) -> str | None:
    vals = [_pref_tuple(v) for v in cands.values()]
    if len(set(vals)) == 1:
        return None
    return max(cands.keys(), key=lambda c: _pref_tuple(cands[c]))


def _metrics(s: dict) -> dict:
    return {
        "r3_ready": s["r3"],
        "SUCCESS_RELAXED": s["succ"],
        "opposing_bilateral": s["opp"],
        "contact_loss": s["loss"],
        "contact_preserved": s.get("contact_preserved"),
        "e_pos_le9": s.get("e9"),
    }


def fit_history_ranker(snap: dict, fit_ics: set[str]):
    """Frozen diagnostic: ridge score(rep_C ⊕ action_onehot) → 0.5*succ+0.5*r3 on fit ICs."""
    names = list(PRIMARY_CANDIDATES)
    Xs, ys = [], []
    for sid, cands in snap.items():
        tid = cands[names[0]]["traj_id"]
        if tid not in fit_ics:
            continue
        for c, s in cands.items():
            oh = np.zeros(len(names))
            oh[names.index(c)] = 1.0
            Xs.append(np.concatenate([s["rep_C"], oh]))
            ys.append(0.5 * s["succ"] + 0.5 * s["r3"])
    X = np.stack(Xs)
    y = np.asarray(ys, float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xz = (X - mu) / sd
    A = np.column_stack([Xz, np.ones(len(Xz))])
    coef = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ y)

    def score(cands: dict, c: str) -> float:
        oh = np.zeros(len(names))
        oh[names.index(c)] = 1.0
        x = np.concatenate([cands[c]["rep_C"], oh])
        xz = (x - mu) / sd
        return float(np.concatenate([xz, [1.0]]) @ coef)

    def choose(cands: dict) -> str:
        return max(cands.keys(), key=lambda c: score(cands, c))

    return {"coef": coef, "mu": mu, "sd": sd, "names": names, "score": score, "choose": choose, "n_fit": len(Xs)}


def best_fixed_on_fit(snap: dict, fit_ics: set[str]) -> str:
    """Select best fixed candidate using training ICs only (mean succ,r3,opp)."""
    totals = {c: [] for c in PRIMARY_CANDIDATES}
    for sid, cands in snap.items():
        tid = cands[PRIMARY_CANDIDATES[0]]["traj_id"]
        if tid not in fit_ics:
            continue
        for c, s in cands.items():
            totals[c].append(_pref_tuple(s))
    # mean of tuples componentwise then lex
    means = {}
    for c, vals in totals.items():
        arr = np.asarray(vals, float)
        means[c] = tuple(arr.mean(0).tolist())
    return max(means.keys(), key=lambda c: means[c]), means


def ic_bootstrap(values_by_ic: dict[str, float], n_boot: int = 1000, seed: int = SEED) -> dict:
    ics = list(values_by_ic.keys())
    if len(ics) < 3:
        return {"n_ics": len(ics), "mean": float(np.mean(list(values_by_ic.values()))) if ics else None, "ci95": None}
    rng = np.random.default_rng(seed)
    arr = np.asarray([values_by_ic[i] for i in ics], float)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(samp.mean()))
    return {
        "n_ics": len(ics),
        "mean": float(arr.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
    }


def run_selection_analyses() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = pickle.loads((P79_CACHE / "cf_rows_expand.pkl").read_bytes())
    pilot = pickle.loads((P79_CACHE / "cf_rows_pilot.pkl").read_bytes())
    rb = pickle.loads((P79_CACHE / "ranking_snap.pkl").read_bytes())
    snap = _agg_snap(rows)
    fit_ics = set(rb["bench"]["fit_ics"])
    eval_ics = set(rb["bench"]["eval_ics"])

    ranker = fit_history_ranker(snap, fit_ics)
    fixed_name, fixed_means = best_fixed_on_fit(snap, fit_ics)

    # ---- Phase 2/4: method comparison on eval ICs ----
    methods = ["A_actor", "B_always_56", "C_best_fixed_fit", "D_history_ranker", "E_oracle"]
    per_snap = []
    by_method_ic = {m: defaultdict(list) for m in methods}

    for sid, cands in snap.items():
        tid = cands[PRIMARY_CANDIDATES[0]]["traj_id"]
        if tid not in eval_ics:
            continue
        choices = {
            "A_actor": "baseline",
            "B_always_56": "tok_56_soft",
            "C_best_fixed_fit": fixed_name,
            "D_history_ranker": ranker["choose"](cands),
            "E_oracle": _best(cands) or "baseline",  # if all tie, any
        }
        oracle = _best(cands)
        row = {"snap_id": sid, "traj_id": tid, "stratum": cands[PRIMARY_CANDIDATES[0]].get("stratum"), "sw0": cands[PRIMARY_CANDIDATES[0]].get("sw0")}
        for m, ch in choices.items():
            met = _metrics(cands[ch])
            row[m] = {"choice": ch, **met}
            by_method_ic[m][tid].append(met["SUCCESS_RELAXED"])
            # also store r3 per ic mean later
        # regrets vs oracle on succ/r3
        if oracle is not None:
            o = cands[oracle]
            for m, ch in choices.items():
                if m == "E_oracle":
                    row[m]["regret_r3"] = 0.0
                    row[m]["regret_succ"] = 0.0
                else:
                    row[m]["regret_r3"] = float(o["r3"] - cands[ch]["r3"])
                    row[m]["regret_succ"] = float(o["succ"] - cands[ch]["succ"])
        else:
            for m in methods:
                row[m]["regret_r3"] = 0.0
                row[m]["regret_succ"] = 0.0
        per_snap.append(row)

    def summarize(method: str) -> dict:
        rows_m = [p[method] for p in per_snap]
        succ_by_ic = {tid: float(np.mean(vs)) for tid, vs in by_method_ic[method].items()}
        r3_by_ic = defaultdict(list)
        for p in per_snap:
            r3_by_ic[p["traj_id"]].append(p[method]["r3_ready"])
        r3_ic = {t: float(np.mean(v)) for t, v in r3_by_ic.items()}
        regret_by_ic = defaultdict(list)
        for p in per_snap:
            regret_by_ic[p["traj_id"]].append(p[method].get("regret_r3", 0.0))
        reg_ic = {t: float(np.mean(v)) for t, v in regret_by_ic.items()}
        return {
            "n_eval_snaps": len(rows_m),
            "n_eval_ics": len(succ_by_ic),
            "SUCCESS_RELAXED": float(np.mean([r["SUCCESS_RELAXED"] for r in rows_m])),
            "r3_ready": float(np.mean([r["r3_ready"] for r in rows_m])),
            "opposing_bilateral": float(np.mean([r["opposing_bilateral"] for r in rows_m])),
            "contact_loss": float(np.mean([r["contact_loss"] for r in rows_m])),
            "contact_preserved": float(np.mean([r["contact_preserved"] for r in rows_m if r.get("contact_preserved") is not None])),
            "e_pos_le9": float(np.mean([r["e_pos_le9"] for r in rows_m if r.get("e_pos_le9") is not None])),
            "mean_regret_r3": float(np.mean([r.get("regret_r3", 0) for r in rows_m])),
            "choice_counts": dict(Counter(r["choice"] for r in rows_m)),
            "ic_bootstrap_succ": ic_bootstrap(succ_by_ic, seed=SEED + 1),
            "ic_bootstrap_r3": ic_bootstrap(r3_ic, seed=SEED + 2),
            "ic_bootstrap_regret_r3": ic_bootstrap(reg_ic, seed=SEED + 3),
        }

    method_summary = {m: summarize(m) for m in methods}

    # Paired deltas: D - C, D - A, C - A on IC means
    def paired_delta(m_hi, m_lo, key="SUCCESS_RELAXED"):
        deltas = {}
        for p in per_snap:
            tid = p["traj_id"]
            deltas.setdefault(tid, []).append(p[m_hi][key] - p[m_lo][key])
        by_ic = {t: float(np.mean(v)) for t, v in deltas.items()}
        return ic_bootstrap(by_ic, seed=SEED + hash(m_hi + m_lo) % 1000)

    paired = {
        "D_minus_C_succ": paired_delta("D_history_ranker", "C_best_fixed_fit", "SUCCESS_RELAXED"),
        "D_minus_C_r3": paired_delta("D_history_ranker", "C_best_fixed_fit", "r3_ready"),
        "D_minus_A_succ": paired_delta("D_history_ranker", "A_actor", "SUCCESS_RELAXED"),
        "D_minus_A_r3": paired_delta("D_history_ranker", "A_actor", "r3_ready"),
        "C_minus_A_succ": paired_delta("C_best_fixed_fit", "A_actor", "SUCCESS_RELAXED"),
        "B_minus_A_succ": paired_delta("B_always_56", "A_actor", "SUCCESS_RELAXED"),
        "D_minus_B_succ": paired_delta("D_history_ranker", "B_always_56", "SUCCESS_RELAXED"),
    }

    static_vs = {
        "best_fixed_selected_on_fit_only": fixed_name,
        "fixed_fit_means": {k: list(v) for k, v in fixed_means.items()},
        "methods": method_summary,
        "paired_ic_deltas": paired,
        "n_eval_snaps": len(per_snap),
        "n_eval_ics": len(eval_ics),
        "note": (
            "Best fixed token chosen on fit ICs only. History ranker fitted on fit ICs only. "
            "Oracle E is retrospective upper bound — not a deployable policy. "
            "Do not claim improvement from ranking accuracy alone."
        ),
    }
    (OUT / "STATIC_VS_CONDITIONAL.md").write_text(
        "# Static vs Conditional Action Selection (P80 Phase 2)\n\n```json\n"
        + json.dumps(static_vs, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )

    # ---- Phase 3: preference reversals ----
    pref_counts = Counter()
    pref_ics = defaultdict(set)
    reliable = []  # snaps where best uniquely beats second by primary axis
    all_best = []
    for sid, cands in snap.items():
        tid = cands[PRIMARY_CANDIDATES[0]]["traj_id"]
        best = _best(cands)
        if best is None:
            pref_counts["all_tie"] += 1
            pref_ics["all_tie"].add(tid)
            all_best.append({"snap_id": sid, "traj_id": tid, "best": None, "reliable": False})
            continue
        pref_counts[best] += 1
        pref_ics[best].add(tid)
        # reliability: best strictly better on succ or r3 than every other
        bs = cands[best]
        margin_ok = True
        for c, s in cands.items():
            if c == best:
                continue
            if not (bs["succ"] > s["succ"] or (bs["succ"] == s["succ"] and bs["r3"] > s["r3"])):
                # may only win on opp/loss
                if bs["succ"] == s["succ"] and bs["r3"] == s["r3"]:
                    margin_ok = False
        reliable.append(margin_ok)
        all_best.append(
            {
                "snap_id": sid,
                "traj_id": tid,
                "best": best,
                "reliable": margin_ok,
                "succ": bs["succ"],
                "r3": bs["r3"],
            }
        )

    # Reversals: states where best != tok_56 AND tok_56 not tied for best
    n_56 = pref_counts.get("tok_56_soft", 0)
    n_other = sum(v for k, v in pref_counts.items() if k not in ("tok_56_soft", "all_tie"))
    n_tie_all = pref_counts.get("all_tie", 0)
    reversals = [x for x in all_best if x["best"] is not None and x["best"] != "tok_56_soft" and x["reliable"]]
    # Effect sizes: delta succ/r3 vs tok_56 on those snaps
    reversal_effects = []
    for x in reversals:
        cands = snap[x["snap_id"]]
        d_succ = cands[x["best"]]["succ"] - cands["tok_56_soft"]["succ"]
        d_r3 = cands[x["best"]]["r3"] - cands["tok_56_soft"]["r3"]
        reversal_effects.append({"snap_id": x["snap_id"], "traj_id": x["traj_id"], "best": x["best"], "d_succ": d_succ, "d_r3": d_r3})

    # Seed sensitivity from pilot (2 seeds): does preferred action flip?
    pilot_snap = defaultdict(lambda: defaultdict(list))
    for r in pilot:
        pilot_snap[r["snap_id"]][r["candidate"]].append(r)
    seed_flips = 0
    seed_compared = 0
    for sid, cands in pilot_snap.items():
        # per seed_k best
        by_k = defaultdict(dict)
        for c, rs in cands.items():
            for r in rs:
                by_k[r["seed_k"]][c] = {
                    "succ": float(r["SUCCESS_RELAXED"]),
                    "r3": float(r["r3_ready"]),
                    "opp": float(r["opposing_bilateral"]),
                    "loss": float(r["contact_loss"]),
                }
        if len(by_k) < 2:
            continue
        bests = []
        for k, cd in by_k.items():
            b = _best(cd)
            bests.append(b)
        seed_compared += 1
        if len(set(bests)) > 1:
            seed_flips += 1

    # Does history predict reversals? On eval snaps where oracle != 56
    hist_rev_correct = hist_rev_total = 0
    for p in per_snap:
        cands = snap[p["snap_id"]]
        oracle = _best(cands)
        if oracle is None or oracle == "tok_56_soft":
            continue
        hist_rev_total += 1
        if p["D_history_ranker"]["choice"] == oracle:
            hist_rev_correct += 1

    frac_56 = n_56 / len(snap) if snap else None
    # Classification
    if frac_56 is not None and frac_56 >= 0.7 and n_other <= 5:
        pref_mode = "global_bias_tok56"
    elif n_other >= 5 and len(reversals) >= 5:
        pref_mode = "state_conditioned_reversals"
    elif n_tie_all / max(len(snap), 1) >= 0.5:
        pref_mode = "mostly_ties_weak_signal"
    else:
        pref_mode = "mixed"

    reversals_report = {
        "n_snaps": len(snap),
        "preference_counts": dict(pref_counts),
        "preference_ic_counts": {k: len(v) for k, v in pref_ics.items()},
        "frac_tok56_preferred": frac_56,
        "frac_other_preferred": n_other / len(snap) if snap else None,
        "frac_all_tie": n_tie_all / len(snap) if snap else None,
        "n_reliable_reversals_vs_56": len(reversals),
        "reversal_ics": sorted({x["traj_id"] for x in reversals}),
        "mean_reversal_d_succ_vs_56": float(np.mean([e["d_succ"] for e in reversal_effects])) if reversal_effects else None,
        "mean_reversal_d_r3_vs_56": float(np.mean([e["d_r3"] for e in reversal_effects])) if reversal_effects else None,
        "reversal_examples": reversal_effects[:20],
        "seed_sensitivity_pilot": {
            "n_snaps_compared": seed_compared,
            "n_preference_flips_across_seeds": seed_flips,
            "flip_rate": float(seed_flips / seed_compared) if seed_compared else None,
        },
        "history_predicts_reversals_eval": {
            "n": hist_rev_total,
            "n_correct": hist_rev_correct,
            "accuracy": float(hist_rev_correct / hist_rev_total) if hist_rev_total else None,
        },
        "preference_mode": pref_mode,
        "interpretation": {
            "1_global_superior": pref_mode == "global_bias_tok56",
            "2_genuine_state_conditioned": pref_mode == "state_conditioned_reversals",
            "3_noise_driven": (seed_flips / seed_compared >= 0.25) if seed_compared else None,
        },
    }
    (OUT / "ACTION_PREFERENCE_REVERSALS.md").write_text(
        "# Action Preference Reversals (P80 Phase 3)\n\n```json\n"
        + json.dumps(reversals_report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )

    # ---- Phase 4 offline selection value (compact) ----
    # Ranking accuracy vs utility already separated in static_vs
    d_sum = method_summary["D_history_ranker"]
    c_sum = method_summary["C_best_fixed_fit"]
    a_sum = method_summary["A_actor"]
    e_sum = method_summary["E_oracle"]
    # When D chooses same as C
    same_as_fixed = sum(1 for p in per_snap if p["D_history_ranker"]["choice"] == p["C_best_fixed_fit"]["choice"])
    offline = {
        "held_out": {
            "history_ranker": d_sum,
            "best_fixed": c_sum,
            "actor_baseline": a_sum,
            "oracle": e_sum,
            "always_56": method_summary["B_always_56"],
        },
        "frac_ranker_matches_best_fixed": float(same_as_fixed / len(per_snap)) if per_snap else None,
        "paired_D_minus_C_succ": paired["D_minus_C_succ"],
        "paired_D_minus_C_r3": paired["D_minus_C_r3"],
        "regret": {
            "D_mean_regret_r3": d_sum["mean_regret_r3"],
            "C_mean_regret_r3": c_sum["mean_regret_r3"],
            "A_mean_regret_r3": a_sum["mean_regret_r3"],
            "oracle_regret": 0.0,
        },
        "ties_note": (
            "When all candidates tie, oracle/regret defined as 0 and choice arbitrary. "
            "Unresolved secondary-only differences excluded from 'reliable reversal' counts."
        ),
        "ranking_accuracy_is_not_utility": (
            "P79 history pairwise ranking accuracy 0.75 is a separate metric from "
            "held-out SUCCESS/R3 rates of the selected action."
        ),
    }
    (OUT / "OFFLINE_SELECTION_VALUE.md").write_text(
        "# Offline Selection Value (P80 Phase 4)\n\n```json\n"
        + json.dumps(offline, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )

    (CACHE / "selection.pkl").write_bytes(
        pickle.dumps(
            {
                "per_snap": per_snap,
                "static_vs": static_vs,
                "reversals": reversals_report,
                "offline": offline,
                "fixed_name": fixed_name,
                "fit_ics": sorted(fit_ics),
                "eval_ics": sorted(eval_ics),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    print(
        json.dumps(
            {
                "fixed": fixed_name,
                "D_succ": d_sum["SUCCESS_RELAXED"],
                "C_succ": c_sum["SUCCESS_RELAXED"],
                "A_succ": a_sum["SUCCESS_RELAXED"],
                "E_succ": e_sum["SUCCESS_RELAXED"],
                "D_minus_C": paired["D_minus_C_succ"],
                "pref_mode": pref_mode,
                "n_rev": len(reversals),
            },
            indent=2,
        ),
        flush=True,
    )
    return {
        "static_vs": static_vs,
        "reversals": reversals_report,
        "offline": offline,
        "paired": paired,
        "fixed_name": fixed_name,
        "per_snap": per_snap,
        "snap": snap,
        "fit_ics": fit_ics,
        "eval_ics": eval_ics,
    }


if __name__ == "__main__":
    run_selection_analyses()
