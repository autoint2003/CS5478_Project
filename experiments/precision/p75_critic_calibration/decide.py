"""Phase 6–7 — diagnose + P76 decision + FINAL_REPORT."""

from __future__ import annotations

import json

import numpy as np

from experiments.precision.p75_critic_calibration.paths import OUT


def _f(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def decide() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((OUT / "BASELINE_CRITIC_DIAGNOSTICS.json").read_text(encoding="utf-8"))
    calib = json.loads((OUT / "FC_WINDOW_CALIBRATION.json").read_text(encoding="utf-8"))
    adv = json.loads((OUT / "ADVANTAGE_QUALITY.json").read_text(encoding="utf-8"))
    retention = json.loads((OUT / "CROSS_CLASS_RETENTION.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "DATA_AND_TARGET_AUDIT.json").read_text(encoding="utf-8"))

    base_fc_ev = _f(calib, "baseline", "fc_window", "ev")
    base_fc_corr = _f(calib, "baseline", "fc_window", "corr")
    base_fc_bias = _f(calib, "baseline", "fc_window", "bias")

    # Best trained variant by eval FC EV
    trained = {k: v for k, v in calib.items() if k != "baseline"}
    best_name, best = None, None
    for name, blob in trained.items():
        ev = _f(blob, "fc_window", "ev")
        if ev is None:
            continue
        if best is None or ev > _f(best, "fc_window", "ev", default=-1e9):
            best_name, best = name, blob

    best_fc_ev = _f(best, "fc_window", "ev") if best else None
    best_fc_corr = _f(best, "fc_window", "corr") if best else None
    best_fc_bias = _f(best, "fc_window", "bias") if best else None

    d_ev = None if best_fc_ev is None or base_fc_ev is None else best_fc_ev - base_fc_ev
    d_corr = None if best_fc_corr is None or base_fc_corr is None else best_fc_corr - base_fc_corr
    # bias: improvement = |bias| decreases
    bias_improved = (
        best_fc_bias is not None
        and base_fc_bias is not None
        and abs(best_fc_bias) < abs(base_fc_bias) - 0.05
    )

    # Advantage quality
    base_gap = _f(adv, "baseline", "fc_adv_gap_succ_minus_fail")
    best_gap = _f(adv, best_name, "fc_adv_gap_succ_minus_fail") if best_name else None
    adv_improved = (
        best_gap is not None
        and base_gap is not None
        and best_gap > base_gap + 0.02
        and best_gap > 0
    )

    # Cross-class harm: any A/B/D EV drop > 0.05
    harm = False
    harm_detail = {}
    if best_name and best_name in retention:
        for sk, d in retention[best_name].items():
            if sk in ("A1", "B1", "D") and d.get("ev_delta") is not None and d["ev_delta"] < -0.05:
                harm = True
                harm_detail[sk] = d["ev_delta"]

    ev_pos = best_fc_ev is not None and best_fc_ev > 0.05
    ev_improved = d_ev is not None and d_ev > 0.05
    corr_improved = d_corr is not None and d_corr > 0.05

    # Ordinary vs weighted
    a_ev = _f(calib, "A_w1", "fc_window", "ev")
    b_best = None
    for k in trained:
        if k.startswith("B_") and (_f(calib, k, "fc_window", "ev") is not None):
            if b_best is None or _f(calib, k, "fc_window", "ev") > _f(calib, b_best, "fc_window", "ev"):
                b_best = k
    b_ev = _f(calib, b_best, "fc_window", "ev") if b_best else None
    weighted_helps = (
        a_ev is not None and b_ev is not None and b_ev > a_ev + 0.02
    )

    # Diagnose failure modes if still poor
    diagnose = []
    if not ev_pos:
        diagnose.append("A_sparse_return_variance_or_hard_targets")
        fc_std = None
        try:
            fc_std = _f(baseline, "return_dist_eval", "per_stage_mc", "fc_to_opposing", "std")
        except Exception:  # noqa: BLE001
            pass
        if fc_std is not None and fc_std > 5:
            diagnose.append("A_high_mc_variance_in_fc_window")
        diagnose.append("B_partial_observability_sidewall_hidden")
        diagnose.append("F_value_aliasing_hidden_geometry")

    if ev_improved and corr_improved and bias_improved and adv_improved and not harm:
        case = "A"
        reason = (
            f"Best Critic `{best_name}` improves FC EV ({base_fc_ev:.3f}→{best_fc_ev:.3f}), "
            f"correlation, bias, and advantage gap without major cross-class harm."
        )
        recommend_ppo = True
    elif ev_improved and corr_improved and not adv_improved and not harm:
        case = "B"
        reason = (
            f"FC EV/corr improve (`{best_name}`) but advantage-quality diagnostics do not — "
            "do not start Actor PPO; investigate credit assignment."
        )
        recommend_ppo = False
    elif harm and (ev_improved or (best_fc_ev is not None and base_fc_ev is not None and best_fc_ev >= base_fc_ev - 0.02)):
        # Calibration competitive/improving on FC but damages A/B/D → Case D
        case = "D"
        reason = (
            f"FC-oriented Critic calibration harms other classes ({harm_detail}). "
            "Reject specialized/retrained Critic; keep P64 Critic + V2."
        )
        recommend_ppo = False
    elif not ev_improved or not ev_pos:
        case = "C"
        extra = f" Also harms A/B/D ({harm_detail})." if harm else ""
        reason = (
            f"Critic remains uninformative in FC window "
            f"(baseline EV={base_fc_ev:.4f}, best `{best_name}` EV={best_fc_ev}). "
            f"Corr may rise while EV falls (scale/bias). Diagnoses: {diagnose}.{extra} "
            "Do not keep spending Critic budget blindly; retain P64 Critic + V2."
        )
        recommend_ppo = False
    else:
        case = "C"
        reason = "Partial improvements insufficient for Case A gates; retain P64 Critic + V2."
        recommend_ppo = False

    decision = {
        "case": case,
        "best_critic": best_name,
        "recommend_matched_actor_ppo_p76": recommend_ppo,
        "auto_start_p76": False,
        "reason": reason,
        "metrics": {
            "baseline_fc_ev": base_fc_ev,
            "baseline_fc_corr": base_fc_corr,
            "baseline_fc_bias": base_fc_bias,
            "best_fc_ev": best_fc_ev,
            "best_fc_corr": best_fc_corr,
            "best_fc_bias": best_fc_bias,
            "delta_fc_ev": d_ev,
            "delta_fc_corr": d_corr,
            "bias_improved": bias_improved,
            "adv_improved": adv_improved,
            "base_adv_gap": base_gap,
            "best_adv_gap": best_gap,
            "ordinary_A_fc_ev": a_ev,
            "best_B": b_best,
            "best_B_fc_ev": b_ev,
            "weighted_helps": weighted_helps,
            "cross_class_harm": harm,
            "harm_detail": harm_detail,
        },
        "diagnose": diagnose,
        "audit_ok": _f(audit, "checks", "actor_digest_matches_collection"),
    }

    (OUT / "P76_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "P76_DECISION.md").write_text(
        f"# P76 Decision\n\n**Case {case}** — recommend Actor PPO: **{recommend_ppo}** "
        f"(auto-start: False)\n\n{reason}\n\n```json\n{json.dumps(decision, indent=2)}\n```\n",
        encoding="utf-8",
    )

    answers = {
        "1_fc_ev_positive": {"best_fc_ev": best_fc_ev, "positive": bool(ev_pos)},
        "2_corr_improved": {"baseline": base_fc_corr, "best": best_fc_corr, "delta": d_corr},
        "3_ordinary_sufficient": {
            "A_fc_ev": a_ev,
            "beats_baseline": a_ev is not None and base_fc_ev is not None and a_ev > base_fc_ev + 0.05,
        },
        "4_weighted_helps": {"weighted_helps": weighted_helps, "best_B": b_best, "b_ev": b_ev, "a_ev": a_ev},
        "5_generalizes_C_ics": {
            "fc_ep_ev_ci": _f(calib, best_name or "baseline", "fc_ep_ev_wilson_like_ci"),
            "eval_ics": _f(audit, "meta", "eval_ics"),
        },
        "6_abd_preserved": {"harm": harm, "retention_best": retention.get(best_name) if best_name else None},
        "7_advantage_quality": {"improved": adv_improved, "base_gap": base_gap, "best_gap": best_gap},
        "8_justify_actor_ppo": {"case": case, "recommend": recommend_ppo},
    }
    (OUT / "FINAL_ANSWERS.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    (OUT / "FINAL_REPORT.md").write_text(
        f"# P75 Final Report — Contact-Stage Critic Calibration\n\n"
        f"**Case {case}** — Critic-only; Actor/V2 unchanged; no aux reward; no overlays.\n\n"
        f"{reason}\n\n## Answers\n"
        + "\n".join(f"### {k}\n```json\n{json.dumps(v, indent=2)}\n```\n" for k, v in answers.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2), flush=True)
    return decision


if __name__ == "__main__":
    decide()
