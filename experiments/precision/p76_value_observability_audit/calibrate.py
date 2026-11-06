"""Phase 1–2 — frozen Critic post-hoc calibration + stage/cross-class validation."""

from __future__ import annotations

import json

import numpy as np
import torch

from experiments.precision.p76_value_observability_audit.io_util import (
    apply_affine,
    ensure_preds,
    fit_affine,
    fit_bias,
    ic_level_ev,
    load_episodes,
    load_preds,
    metrics_block,
    split_train_ics,
)
from experiments.precision.p76_value_observability_audit.paths import (
    CALIB_VAL_FRAC,
    FC_STAGE_NAMES,
    OUT,
    SEED,
)


CRITICS = ("P64", "A_w1", "B_w4")


def _mask_ics(blob, ics: set) -> np.ndarray:
    return np.asarray([t in ics for t in blob["traj"]], dtype=bool)


def _stage_metrics(pred, y, stages, skills) -> dict:
    out = {}
    for st in sorted(set(stages.tolist())):
        m = stages == st
        out[st] = metrics_block(pred[m], y[m])
    out["fc_window"] = metrics_block(
        pred[np.isin(stages, list(FC_STAGE_NAMES))],
        y[np.isin(stages, list(FC_STAGE_NAMES))],
    )
    per_sk = {}
    for sk in sorted(set(skills.tolist())):
        m = skills == sk
        per_sk[sk] = metrics_block(pred[m], y[m])
        if sk == "C":
            fc = m & np.isin(stages, list(FC_STAGE_NAMES))
            per_sk[sk]["fc"] = metrics_block(pred[fc], y[fc])
    out["per_skill"] = per_sk
    return out


def _why_corr_vs_ev(raw_m, note_name: str) -> dict:
    """Decompose: EV ignores additive bias; scale mismatch hurts EV even if corr rises."""
    return {
        "critic": note_name,
        "ev_invariant_to_additive_bias": True,
        "identity": "EV = 1 - Var(y-pred)/Var(y); Corr = Cov(pred,y)/(s_pred s_y)",
        "pred_var_over_y_var": raw_m.get("pred_var_over_y_var"),
        "bias": raw_m.get("bias"),
        "corr": raw_m.get("corr"),
        "ev": raw_m.get("ev"),
        "explanation": (
            "High correlation with wrong prediction scale (Var(pred) ≠ Var(y) or wrong slope) "
            "inflates residual variance → EV falls while Corr rises. Additive bias alone cannot "
            "explain EV change; look at affine slope a and pred_var/y_var."
        ),
    }


def run_calibrate(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    train = load_episodes("train")
    eval_eps = load_episodes("eval")
    fit_ics, val_ics = split_train_ics(train, val_frac=CALIB_VAL_FRAC, seed=SEED)
    eval_ics = sorted({e["traj_id"] for e in eval_eps})
    assert not (fit_ics & val_ics)
    assert not (fit_ics & set(eval_ics))
    assert not (val_ics & set(eval_ics))

    split_meta = {
        "n_train_eps": len(train),
        "n_eval_eps": len(eval_eps),
        "n_fit_ics": len(fit_ics),
        "n_calib_val_ics": len(val_ics),
        "n_eval_ics": len(eval_ics),
        "fit_ics": sorted(fit_ics),
        "calib_val_ics": sorted(val_ics),
        "note": "Eval ICs untouched until procedure frozen; no privileged labels used for selection.",
    }
    (OUT / "CALIB_SPLIT.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    # Ensure predictions cached
    for name in CRITICS:
        print(f"[cache] {name} train", flush=True)
        ensure_preds(name, train, "train", device_t)
        print(f"[cache] {name} eval", flush=True)
        ensure_preds(name, eval_eps, "eval", device_t)

    affine_report = {}
    scale_report = {}
    independent = {}
    stage_report = {}
    selected = {}

    for name in CRITICS:
        tr = load_preds(name, "train")
        ev = load_preds(name, "eval")
        m_fit = _mask_ics(tr, fit_ics)
        m_val = _mask_ics(tr, val_ics)
        # Fit on FC-window of fit ICs (primary target) AND also report overall-fit variant
        fc_fit = m_fit & tr["fc"]
        candidates = {}

        # A raw
        candidates["raw"] = {"a": 1.0, "b": 0.0, "fit_on": "none"}

        # B bias-only on FC fit
        a_b, b_b = fit_bias(tr["pred"][fc_fit], tr["y"][fc_fit])
        candidates["bias_fc"] = {"a": a_b, "b": b_b, "fit_on": "fc_fit_ics"}

        # C affine on FC fit
        a_f, b_f = fit_affine(tr["pred"][fc_fit], tr["y"][fc_fit])
        candidates["affine_fc"] = {"a": a_f, "b": b_f, "fit_on": "fc_fit_ics"}

        # Also overall affine (diagnostic; may be selected if better on calib-val FC)
        a_o, b_o = fit_affine(tr["pred"][m_fit], tr["y"][m_fit])
        candidates["affine_all"] = {"a": a_o, "b": b_o, "fit_on": "all_fit_ics"}

        # Select on calib-val FC EV (no eval, no privileged)
        fc_val = m_val & tr["fc"]
        scored = {}
        for cname, c in candidates.items():
            p = apply_affine(tr["pred"][fc_val], c["a"], c["b"])
            scored[cname] = {
                **c,
                "calib_val_fc": metrics_block(p, tr["y"][fc_val]),
            }
        best = max(scored.keys(), key=lambda k: scored[k]["calib_val_fc"].get("ev", -1e9))
        selected[name] = {"method": best, **scored[best]}

        # Freeze procedure; evaluate once on independent eval
        meth = selected[name]["method"]
        a, b = selected[name]["a"], selected[name]["b"]
        methods_eval = {}
        for cname, c in candidates.items():
            p = apply_affine(ev["pred"], c["a"], c["b"])
            methods_eval[cname] = {
                **c,
                "fc": metrics_block(p[ev["fc"]], ev["y"][ev["fc"]]),
                "overall": metrics_block(p, ev["y"]),
                "ic_fc": ic_level_ev(p[ev["fc"]], ev["y"][ev["fc"]], ev["traj"][ev["fc"]]),
            }
        # Selected method full stage breakdown
        p_sel = apply_affine(ev["pred"], a, b)
        stage_report[name] = {
            "selected": meth,
            "a": a,
            "b": b,
            "raw": _stage_metrics(ev["pred"], ev["y"], ev["stages"], ev["skills"]),
            "calibrated": _stage_metrics(p_sel, ev["y"], ev["stages"], ev["skills"]),
        }
        # Cross-class harm: calibrated vs raw EV deltas
        harm = {}
        for sk, blob in stage_report[name]["calibrated"]["per_skill"].items():
            raw_ev = stage_report[name]["raw"]["per_skill"][sk].get("ev")
            cal_ev = blob.get("ev")
            harm[sk] = {
                "raw_ev": raw_ev,
                "cal_ev": cal_ev,
                "delta": None if raw_ev is None or cal_ev is None else cal_ev - raw_ev,
            }
        stage_report[name]["cross_class_delta_vs_raw"] = harm

        affine_report[name] = {
            "candidates_calib_val": scored,
            "selected": selected[name],
            "independent_eval": methods_eval,
        }
        scale_report[name] = {
            "raw_fc": methods_eval["raw"]["fc"],
            "bias_fc": methods_eval["bias_fc"]["fc"],
            "affine_fc": methods_eval["affine_fc"]["fc"],
            "affine_all": methods_eval["affine_all"]["fc"],
            "selected_method": meth,
            "selected_fc": methods_eval[meth]["fc"],
            "corr_vs_ev_note": _why_corr_vs_ev(methods_eval["raw"]["fc"], name),
        }
        independent[name] = {
            "selected": meth,
            "a": a,
            "b": b,
            "fc": methods_eval[meth]["fc"],
            "overall": methods_eval[meth]["overall"],
            "ic_fc": methods_eval[meth]["ic_fc"],
            "raw_fc": methods_eval["raw"]["fc"],
            "delta_fc_ev_vs_raw": methods_eval[meth]["fc"].get("ev", 0) - methods_eval["raw"]["fc"].get("ev", 0),
        }
        print(
            f"[calib] {name} selected={meth} a={a:.4f} b={b:.4f} "
            f"eval_fc_ev raw={methods_eval['raw']['fc'].get('ev'):.4f} "
            f"sel={methods_eval[meth]['fc'].get('ev'):.4f}",
            flush=True,
        )

    # Compare A_w1 raw vs P64 for the corr/EV paradox
    scale_report["paradox_A_w1_vs_P64"] = {
        "P64_fc": scale_report["P64"]["raw_fc"],
        "A_w1_fc": scale_report["A_w1"]["raw_fc"],
        "note": (
            "A_w1 raises Corr but lowers EV vs P64 when pred_var/y_var and residual scale worsen. "
            "Bias-only cannot fix EV; affine may if slope is the issue."
        ),
    }

    (OUT / "AFFINE_CALIBRATION.json").write_text(json.dumps(affine_report, indent=2), encoding="utf-8")
    (OUT / "AFFINE_CALIBRATION.md").write_text(
        "# Affine Calibration (P76 Phase 1)\n\n"
        "Fit on train-fit ICs only; select on calib-val ICs; evaluate once on independent eval ICs.\n\n"
        f"Selected: `{json.dumps(selected, indent=2)}`\n\n```json\n"
        + json.dumps({k: v["independent_eval"][v["selected"]["method"]] for k, v in affine_report.items()}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "VALUE_SCALE_ANALYSIS.json").write_text(json.dumps(scale_report, indent=2), encoding="utf-8")
    (OUT / "VALUE_SCALE_ANALYSIS.md").write_text(
        "# Value Scale Analysis\n\nEV is invariant to additive bias; scale/slope mismatches drive EV.\n\n```json\n"
        + json.dumps(scale_report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "INDEPENDENT_IC_VALIDATION.json").write_text(
        json.dumps({"split": split_meta, "results": independent, "stage": stage_report}, indent=2),
        encoding="utf-8",
    )
    (OUT / "INDEPENDENT_IC_VALIDATION.md").write_text(
        "# Independent IC Validation + Stage/Cross-Class (P76 Phase 1–2)\n\n```json\n"
        + json.dumps({"results": independent, "cross_class": {k: v["cross_class_delta_vs_raw"] for k, v in stage_report.items()}}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return {"selected": selected, "independent": independent, "stage": stage_report}


if __name__ == "__main__":
    run_calibrate()
