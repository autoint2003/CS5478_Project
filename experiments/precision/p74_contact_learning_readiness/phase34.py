"""Phase 3–4: learnable decisions + contact-stage Critic audit."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.precision.p74_contact_learning_readiness.paths import N_CRITIC_MAX, OUT, P73_CF


def _corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 8 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ev(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.size < 8:
        return None
    var = float(np.var(y_true))
    if var < 1e-12:
        return None
    return float(1.0 - np.var(y_true - y_pred) / var)


def run_phase34() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads((OUT / "AUDIT_RAW.json").read_text(encoding="utf-8"))
    rows = raw["rows"]

    # --- Learnable decisions ---
    fam_succ = Counter()
    fam_fail = Counter()
    ent_succ, ent_fail = [], []
    # Actor-controlled FC window
    for r in rows:
        bucket = fam_succ if r["reached_opposing"] else fam_fail
        for s in r["steps"]:
            if s["window"] != "fc_to_opposing" or s["v2_override"]:
                continue
            bucket[s["family_actor"]] += 1
            (ent_succ if r["reached_opposing"] else ent_fail).append(float(s["entropy"]))

    def fam_dist(c: Counter):
        n = sum(c.values()) or 1
        return {k: v / n for k, v in c.items()}

    # Matched e_pos bins: within bin, does actor family differ by eventual outcome?
    bins = [(0, 15), (15, 25), (25, 40), (40, 80)]
    bin_analysis = []
    for lo, hi in bins:
        succ_f, fail_f = Counter(), Counter()
        for r in rows:
            for s in r["steps"]:
                if s["window"] != "fc_to_opposing" or s["v2_override"]:
                    continue
                e = float(s["e_pos_mm"])
                if not (lo <= e < hi):
                    continue
                (succ_f if r["reached_opposing"] else fail_f)[s["family_actor"]] += 1
        bin_analysis.append(
            {
                "e_pos_mm": [lo, hi],
                "succ_family": fam_dist(succ_f),
                "fail_family": fam_dist(fail_f),
                "n_succ": sum(succ_f.values()),
                "n_fail": sum(fail_f.values()),
            }
        )

    # Within-episode action variation: unique actor tokens in FC window
    variations = []
    for r in rows:
        toks = [s["actor_tok"] for s in r["steps"] if s["window"] == "fc_to_opposing" and not s["v2_override"]]
        if len(toks) >= 2:
            variations.append(len(set(toks)) / len(toks))

    # P73 CF evidence: different M111 sequences from matched FC states change R3
    cf_evidence = None
    if P73_CF.exists():
        cf = json.loads(P73_CF.read_text(encoding="utf-8"))
        fc = (cf.get("cf_results") or {}).get("first_contact") or {}
        base = fc.get("baseline") or {}
        deltas = {}
        for name, blob in fc.items():
            if name == "baseline" or not blob:
                continue
            deltas[name] = {
                "d_r3": (blob.get("r3_ready") or 0) - (base.get("r3_ready") or 0),
                "d_opp": (blob.get("reached_opposing") or 0) - (base.get("reached_opposing") or 0),
                "d_e9": (blob.get("reached_e9") or 0) - (base.get("reached_e9") or 0),
            }
        cf_evidence = {
            "source": "P73 first_contact matched counterfactuals",
            "baseline_r3": base.get("r3_ready"),
            "deltas": deltas,
            "interpretation": (
                "Different existing M111 sequences from identical FC states change R3 "
                "(and sometimes e9) without always changing opposing rate — evidence that "
                "state-dependent M111 choice can alter outcomes, but not that a fixed rule generalizes."
            ),
        }

    learnable = {
        "family_dist_fc_window": {"succ_opp": fam_dist(fam_succ), "fail_opp": fam_dist(fam_fail)},
        "mean_entropy_fc": {
            "succ": float(np.mean(ent_succ)) if ent_succ else None,
            "fail": float(np.mean(ent_fail)) if ent_fail else None,
        },
        "matched_e_pos_bins": bin_analysis,
        "mean_unique_tok_frac_in_fc_window": float(np.mean(variations)) if variations else None,
        "p73_cf_evidence": cf_evidence,
        "evidence_of_learnable_selection": bool(
            cf_evidence
            and any(abs((d.get("d_r3") or 0)) >= 0.05 or abs((d.get("d_e9") or 0)) >= 0.05 for d in (cf_evidence.get("deltas") or {}).values())
        ),
    }
    (OUT / "LEARNABLE_DECISIONS.json").write_text(json.dumps(learnable, indent=2), encoding="utf-8")
    (OUT / "LEARNABLE_DECISIONS.md").write_text(
        "# Learnable Decisions (P74 Phase 3)\n\n```json\n" + json.dumps(learnable, indent=2) + "\n```\n",
        encoding="utf-8",
    )

    # --- Critic audit (γ=1 matches P63/P64 GAMMA) ---
    from experiments.precision.p74_contact_learning_readiness.paths import SEED

    rng = np.random.default_rng(SEED)

    stage_stats = {}
    for stage in ("pre_fc", "fc_to_opposing", "fc_to_contact_loss", "opposing_to_frame", "frame_to_r3"):
        pool = []
        for r in rows:
            for s in r["steps"]:
                if s["window"] != stage:
                    continue
                pool.append(s)
        if len(pool) > N_CRITIC_MAX:
            idx = rng.choice(len(pool), size=N_CRITIC_MAX, replace=False)
            pool = [pool[i] for i in idx]
        vals = [float(s["value"]) for s in pool]
        mcs = [float(s["mc_return"]) for s in pool]
        advs = [float(s["adv_proxy"]) for s in pool]
        if vals:
            bias = float(np.mean(np.asarray(vals) - np.asarray(mcs)))
            stage_stats[stage] = {
                "n": len(vals),
                "mean_V": float(np.mean(vals)),
                "mean_MC": float(np.mean(mcs)),
                "bias_V_minus_MC": bias,
                "corr_V_MC": _corr(vals, mcs),
                "explained_variance": _ev(mcs, vals),
                "mean_abs_adv_proxy": float(np.mean(np.abs(advs))),
                "subsample": "random_cap" if len(pool) == N_CRITIC_MAX else "all",
            }
        else:
            stage_stats[stage] = {"n": 0}

    # Bias specifically before correct sidewall (FC window, correct_sidewall==0)
    def subset_bias(pred):
        vals, mcs = [], []
        for r in rows:
            for s in r["steps"]:
                if s["window"] != "fc_to_opposing":
                    continue
                if not pred(s, r):
                    continue
                vals.append(float(s["value"]))
                mcs.append(float(s["mc_return"]))
                if len(vals) >= N_CRITIC_MAX:
                    return vals, mcs
        return vals, mcs

    v1, m1 = subset_bias(lambda s, r: s["correct_sidewall"] == 0)
    v2, m2 = subset_bias(lambda s, r: s["correct_sidewall"] == 1)
    v3, m3 = subset_bias(lambda s, r: (not r["reached_opposing"]) and s["has_pad_contact"])
    v4, m4 = subset_bias(lambda s, r: r["reached_opposing"] and s["has_pad_contact"])

    def pack(vals, mcs):
        if not vals:
            return None
        return {
            "n": len(vals),
            "bias_V_minus_MC": float(np.mean(np.asarray(vals) - np.asarray(mcs))),
            "ev": _ev(mcs, vals),
            "corr": _corr(vals, mcs),
            "mean_MC": float(np.mean(mcs)),
            "mean_V": float(np.mean(vals)),
        }

    # Distinguishable return advantage: MC on succ-path FC steps vs fail-path
    mc_succ = [float(s["mc_return"]) for r in rows if r["reached_opposing"] for s in r["steps"] if s["window"] == "fc_to_opposing"]
    mc_fail = [float(s["mc_return"]) for r in rows if not r["reached_opposing"] for s in r["steps"] if s["window"] == "fc_to_opposing"]

    critic = {
        "by_stage": stage_stats,
        "before_correct_sidewall": pack(v1, m1),
        "at_correct_sidewall": pack(v2, m2),
        "on_fail_opp_contact_steps": pack(v3, m3),
        "on_succ_opp_contact_steps": pack(v4, m4),
        "mc_return_gap_succ_minus_fail_fc_window": (
            None if not mc_succ or not mc_fail else float(np.mean(mc_succ) - np.mean(mc_fail))
        ),
        "critic_useful_for_contact_ppo": None,
    }
    # Heuristic: useful if EV in fc_to_opposing > 0.05 and MC gap distinguishable
    fc_ev = (stage_stats.get("fc_to_opposing") or {}).get("explained_variance")
    gap = critic["mc_return_gap_succ_minus_fail_fc_window"]
    critic["critic_useful_for_contact_ppo"] = bool(
        (fc_ev is not None and fc_ev > 0.05) and (gap is not None and abs(gap) > 0.05)
    )
    critic["note"] = (
        "EV/bias computed vs undiscounted episodic MC return under hybrid V2+Actor policy. "
        "Not reused from P67 post-stable diagnosis."
    )

    (OUT / "CONTACT_STAGE_CRITIC.json").write_text(json.dumps(critic, indent=2), encoding="utf-8")
    (OUT / "CONTACT_STAGE_CRITIC.md").write_text(
        "# Contact-Stage Critic (P74 Phase 4)\n\n```json\n" + json.dumps(critic, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"learnable": learnable["evidence_of_learnable_selection"], "critic_fc": stage_stats.get("fc_to_opposing"), "useful": critic["critic_useful_for_contact_ppo"]}, indent=2), flush=True)
    return {"learnable": learnable, "critic": critic}


if __name__ == "__main__":
    run_phase34()
