"""Phase 2 — controlled representation comparison (IC-disjoint, paired Δ)."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np

from experiments.precision.p77_contact_observability.metrics import auroc, eval_binary_probe
from experiments.precision.p78_action_relevant_representation.features import (
    engineered_from_obs27_hist,
    obs27,
    rep_A_current,
    rep_B_current_plus_eng,
    rep_C_history,
    rep_D_history_plus_eng,
    vec8_from_obs27,
)
from experiments.precision.p78_action_relevant_representation.paths import (
    ENG_HIST,
    FC_WINDOWS,
    OUT,
    P74_AUDIT,
    P75_DATA,
    P77_SENSE,
    SEED,
    SHORT_HIST,
    VAL_FRAC,
)


def _ic_split(ids, val_frac, seed):
    rng = np.random.default_rng(seed)
    ids = sorted(set(ids))
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac))) if len(ids) > 3 else max(1, len(ids) // 4)
    return set(ids[n_val:]), set(ids[:n_val])


def _paired_delta_auroc(Xa, Xb, y, fit, ev, n_boot=300, seed=SEED):
    """IC-level paired bootstrap of AUROC(B)-AUROC(A)."""
    if ev.sum() < 5 or len(np.unique(y[fit])) < 2:
        return None
    # Need traj ids — pass via external; here approximate by resampling eval rows clustered
    # Caller should use _paired_by_ic
    return None


def _paired_by_ic(rows, key_a, key_b, ykey, fit_ics, eval_ics, n_boot=400, seed=SEED):
    rng = np.random.default_rng(seed)
    # Fit once on fit ICs
    fit_rows = [r for r in rows if r["traj_id"] in fit_ics]
    eval_rows = [r for r in rows if r["traj_id"] in eval_ics]
    if len(fit_rows) < 10 or len(eval_rows) < 5:
        return {"insufficient": True}
    Xa_f = np.stack([r[key_a] for r in fit_rows])
    Xb_f = np.stack([r[key_b] for r in fit_rows])
    y_f = np.asarray([r[ykey] for r in fit_rows], np.int8)
    if len(np.unique(y_f)) < 2:
        return {"insufficient": True}

    def fit_scores(Xf, X_all_eval):
        mu, sd = Xf.mean(0), Xf.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        A = np.column_stack([(Xf - mu) / sd, np.ones(len(Xf))])
        ata = A.T @ A + 1e-2 * np.eye(A.shape[1])
        coef = np.linalg.solve(ata, A.T @ y_f.astype(float))
        Xe = (X_all_eval - mu) / sd
        return np.column_stack([Xe, np.ones(len(Xe))]) @ coef

    # Group eval by IC
    by = defaultdict(list)
    for r in eval_rows:
        by[r["traj_id"]].append(r)
    ics = sorted(by.keys())
    if len(ics) < 3:
        return {"insufficient": True, "n_eval_ics": len(ics)}

    def ic_aurocs(ic_list):
        er = [r for ic in ic_list for r in by[ic]]
        if not er:
            return None, None
        ya = np.asarray([r[ykey] for r in er], np.int8)
        if len(np.unique(ya)) < 2:
            return None, None
        sa = fit_scores(Xa_f, np.stack([r[key_a] for r in er]))
        sb = fit_scores(Xb_f, np.stack([r[key_b] for r in er]))
        return auroc(sa, ya), auroc(sb, ya)

    a0, b0 = ic_aurocs(ics)
    deltas = []
    for _ in range(n_boot):
        sample = [ics[i] for i in rng.integers(0, len(ics), size=len(ics))]
        aa, bb = ic_aurocs(sample)
        if aa is None or bb is None:
            continue
        deltas.append(bb - aa)
    if not deltas:
        return {"point_delta": None if a0 is None or b0 is None else b0 - a0, "ci95": None, "n_boot": 0}
    return {
        "auroc_A": a0,
        "auroc_B": b0,
        "point_delta": float(b0 - a0) if a0 is not None and b0 is not None else None,
        "mean_boot_delta": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "n_boot": len(deltas),
        "n_eval_ics": len(ics),
    }


def _build_p74_rows():
    """Current sidewall (t) on FC windows from P74 obs_vec8 → lift to fake 27D pad."""
    raw = json.loads(P74_AUDIT.read_text(encoding="utf-8"))
    rows = []
    for ep in raw["rows"]:
        # Reconstruct minimal 27D: place vec8 into indices, zeros elsewhere
        hist27 = []
        for s in ep["steps"]:
            o = np.zeros(27, float)
            v8 = np.asarray(s["obs_vec8"], float)
            for j, idx in enumerate((2, 3, 0, 1, 12, 13, 18, 19)):
                o[idx] = v8[j]
            hist27.append(o)
        for ti, s in enumerate(ep["steps"]):
            if s.get("window") not in FC_WINDOWS:
                continue
            h = hist27[max(0, ti + 1 - max(ENG_HIST, SHORT_HIST * 3)) : ti + 1]
            # ensure eng hist length
            hh = hist27[max(0, ti + 1 - ENG_HIST) : ti + 1]
            rows.append(
                {
                    "traj_id": ep["traj_id"],
                    "A": rep_A_current(hist27[ti]),
                    "B": rep_B_current_plus_eng(hh),
                    "C": rep_C_history(h if len(h) >= 2 else hist27[: ti + 1]),
                    "D": rep_D_history_plus_eng(hist27[max(0, ti + 1 - max(ENG_HIST, 8)) : ti + 1]),
                    "y_sw_t": int(s.get("correct_sidewall") or 0),
                    "y_opp": int(ep.get("reached_opposing") or 0),
                    "y_loss": int(ep.get("lost_contact_after_fc") or 0),
                }
            )
    return rows


def _build_p75_rows():
    train = pickle.loads((P75_DATA / "train_episodes.pkl").read_bytes())
    ev = pickle.loads((P75_DATA / "eval_episodes.pkl").read_bytes())
    rows = []
    for split, eps in (("fit", train), ("eval", ev)):
        for ep in eps:
            if ep["skill"] != "C":
                continue
            frames = ep["obs_frames"]
            for t in range(ep["n_steps"]):
                if ep["stages"][t] not in FC_WINDOWS:
                    continue
                hist = [obs27(frames[i]) for i in range(max(0, t + 1 - 16), t + 1)]
                hh = hist[-ENG_HIST:]
                rows.append(
                    {
                        "traj_id": ep["traj_id"],
                        "split_hint": split,
                        "A": rep_A_current(hist[-1]),
                        "B": rep_B_current_plus_eng(hh),
                        "C": rep_C_history(hist),
                        "D": rep_D_history_plus_eng(hist),
                        "y_opp": int(ep["reached_opposing"]),
                        "y_loss": int(ep["lost_contact_after_fc"]),
                        "y_succ": int(ep.get("SUCCESS_RELAXED") or 0),
                    }
                )
    return rows


def _build_p77_rows():
    if not P77_SENSE.exists():
        return []
    eps = pickle.loads(P77_SENSE.read_bytes())
    rows = []
    for ep in eps:
        hist = []
        for ti, s in enumerate(ep["steps"]):
            o = np.asarray(s["obs27"], float)
            hist.append(o)
            if s["window"] not in FC_WINDOWS:
                continue
            if "correct_sidewall_next" not in s:
                continue
            hh = hist[-ENG_HIST:]
            rows.append(
                {
                    "traj_id": ep["traj_id"],
                    "A": rep_A_current(o),
                    "B": rep_B_current_plus_eng(hh),
                    "C": rep_C_history(hist[-max(8, ENG_HIST) :]),
                    "D": rep_D_history_plus_eng(hist[-max(8, ENG_HIST) :]),
                    "y_sw_next": int(s["correct_sidewall_next"]),
                    "y_sw_t": int(s["correct_sidewall"]),
                    "y_opp": int(ep["reached_opposing"]),
                    "y_loss": int(ep["lost_contact_after_fc"]),
                }
            )
    return rows


def _eval_task(rows, ykey, fit_ics, eval_ics, reps=("A", "B", "C", "D")):
    fit = np.asarray([r["traj_id"] in fit_ics for r in rows])
    ev = np.asarray([r["traj_id"] in eval_ics for r in rows])
    y = np.asarray([r[ykey] for r in rows], np.int8)
    out = {"prevalence_eval": float(y[ev].mean()) if ev.any() else None, "reps": {}}
    for rep in reps:
        X = np.stack([r[rep] for r in rows])
        out["reps"][rep] = eval_binary_probe(X, y, fit, ev)
    # Paired deltas vs A
    out["paired_vs_A"] = {}
    for rep in reps:
        if rep == "A":
            continue
        out["paired_vs_A"][rep] = _paired_by_ic(rows, "A", rep, ykey, fit_ics, eval_ics)
    return out


def run_compare() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"limitations": []}

    # --- P74: current sidewall ---
    p74 = _build_p74_rows()
    fit_ics, eval_ics = _ic_split([r["traj_id"] for r in p74], VAL_FRAC, SEED)
    report["p74_current_sidewall_t"] = {
        "n": len(p74),
        "n_fit_ics": len(fit_ics),
        "n_eval_ics": len(eval_ics),
        **_eval_task(p74, "y_sw_t", fit_ics, eval_ics),
    }
    report["p74_eventual_opposing"] = _eval_task(p74, "y_opp", fit_ics, eval_ics)
    report["p74_eventual_loss"] = _eval_task(p74, "y_loss", fit_ics, eval_ics)

    # --- P75: outcomes with official IC split ---
    p75 = _build_p75_rows()
    fit75 = {r["traj_id"] for r in p75 if r.get("split_hint") == "fit"}
    eval75 = {r["traj_id"] for r in p75 if r.get("split_hint") == "eval"}
    report["p75_eventual_opposing"] = {
        "n": len(p75),
        **_eval_task(p75, "y_opp", fit75, eval75),
    }
    report["p75_eventual_loss"] = _eval_task(p75, "y_loss", fit75, eval75)

    # --- P77 sense: next-sidewall (pre-action features) ---
    p77 = _build_p77_rows()
    if len(p77) < 30:
        report["limitations"].append("P77 sensing diag missing or too small for next-sidewall task.")
        report["p77_sidewall_t_plus_1"] = {"insufficient": True, "n": len(p77)}
    else:
        # stratify by positive next-sw
        by = {}
        for r in p77:
            by.setdefault(r["traj_id"], False)
            by[r["traj_id"]] = by[r["traj_id"]] or bool(r["y_sw_next"])
        pos = [i for i, v in by.items() if v]
        neg = [i for i, v in by.items() if not v]
        rng = np.random.default_rng(SEED + 3)
        fit_ics, eval_ics = set(), set()
        for pool in (pos, neg):
            if not pool:
                continue
            lst = list(pool)
            rng.shuffle(lst)
            n_val = max(1, int(round(len(lst) * VAL_FRAC))) if len(lst) > 3 else max(1, len(lst) // 4)
            eval_ics.update(lst[:n_val])
            fit_ics.update(lst[n_val:])
        report["p77_sidewall_t_plus_1"] = {
            "n": len(p77),
            "n_fit_ics": len(fit_ics),
            "n_eval_ics": len(eval_ics),
            **_eval_task(p77, "y_sw_next", fit_ics, eval_ics),
        }

    # Stability summary
    b_vs_a = report["p74_current_sidewall_t"]["paired_vs_A"].get("B") or {}
    d_vs_a = report["p74_current_sidewall_t"]["paired_vs_A"].get("D") or {}
    report["summary"] = {
        "p74_sw_A": report["p74_current_sidewall_t"]["reps"]["A"].get("auroc"),
        "p74_sw_B": report["p74_current_sidewall_t"]["reps"]["B"].get("auroc"),
        "p74_sw_C": report["p74_current_sidewall_t"]["reps"]["C"].get("auroc"),
        "p74_sw_D": report["p74_current_sidewall_t"]["reps"]["D"].get("auroc"),
        "paired_B_minus_A": b_vs_a,
        "paired_D_minus_A": d_vs_a,
        "eng_beats_history": (
            (report["p74_current_sidewall_t"]["reps"]["B"].get("auroc") or 0)
            > (report["p74_current_sidewall_t"]["reps"]["C"].get("auroc") or 0)
        ),
        "n_p74_ics_note": "P74 has ~100 episodes; IC bootstrap CIs may be wide.",
    }

    (OUT / "REPRESENTATION_COMPARISON.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        "summary": report["summary"],
        "p77_next_sw": {
            k: (report.get("p77_sidewall_t_plus_1", {}).get("reps") or {}).get(k, {}).get("auroc")
            for k in ("A", "B", "C", "D")
        },
        "p75_opp": {
            k: (report["p75_eventual_opposing"].get("reps") or {}).get(k, {}).get("auroc")
            for k in ("A", "B", "C", "D")
        },
        "p75_loss": {
            k: (report["p75_eventual_loss"].get("reps") or {}).get(k, {}).get("auroc")
            for k in ("A", "B", "C", "D")
        },
    }
    (OUT / "REPRESENTATION_COMPARISON.md").write_text(
        "# Representation Comparison (P78 Phase 2)\n\n"
        "A=current 27D · B=A+engineered · C=history summary · D=C+engineered\n\n```json\n"
        + json.dumps(compact, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_compare()
