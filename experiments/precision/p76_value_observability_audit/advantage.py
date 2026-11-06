"""Phase 3 — GAE recalculation under raw vs affine-calibrated values (no Actor update)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae
from experiments.precision.p76_value_observability_audit.io_util import apply_affine, load_episodes, load_preds
from experiments.precision.p76_value_observability_audit.paths import (
    FC_STAGE_NAMES,
    GAE_LAMBDA,
    GAMMA,
    OUT,
)


def _pack(xs):
    if not xs:
        return None
    arr = np.asarray(xs, float)
    return {
        "n": len(xs),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "var": float(arr.var()),
        "frac_pos": float(np.mean(arr > 0)),
        "p10": float(np.quantile(arr, 0.1)),
        "p90": float(np.quantile(arr, 0.9)),
    }


def _adv_for_values(episodes, values_by_ep: list[np.ndarray]) -> dict:
    adv_fc_succ, adv_fc_fail, adv_pre_loss, all_adv = [], [], [], []
    sign_flip_rate = []
    for ep, vals in zip(episodes, values_by_ep):
        T = ep["n_steps"]
        assert len(vals) == T
        values = np.concatenate([vals.astype(np.float32), [0.0]])
        adv, _ = compute_gae(ep["rewards"], values, ep["dones"], GAMMA, GAE_LAMBDA)
        all_adv.append(adv)
        for t in range(T):
            st = ep["stages"][t]
            if st not in FC_STAGE_NAMES:
                continue
            a = float(adv[t])
            if ep["reached_opposing"]:
                adv_fc_succ.append(a)
            else:
                adv_fc_fail.append(a)
            if ep["lost_contact_after_fc"] and not ep["reached_opposing"]:
                adv_pre_loss.append(a)
        # sign stability vs baseline stored GAE if present
        if "baseline_adv_gae095" in ep:
            base = ep["baseline_adv_gae095"]
            m = np.abs(base) > 0.5
            if m.any():
                sign_flip_rate.append(float(np.mean(np.sign(adv[m]) != np.sign(base[m]))))

    gap = None
    if adv_fc_succ and adv_fc_fail:
        gap = float(np.mean(adv_fc_succ) - np.mean(adv_fc_fail))
    # Separability: Cohen's d
    d = None
    if adv_fc_succ and adv_fc_fail:
        s1, s0 = np.asarray(adv_fc_succ), np.asarray(adv_fc_fail)
        pooled = np.sqrt(0.5 * (s1.var() + s0.var()) + 1e-12)
        d = float((s1.mean() - s0.mean()) / pooled)

    return {
        "adv_mean": float(np.mean([a.mean() for a in all_adv])) if all_adv else None,
        "adv_var_mean_ep": float(np.mean([a.var() for a in all_adv])) if all_adv else None,
        "fc_succ": _pack(adv_fc_succ),
        "fc_fail": _pack(adv_fc_fail),
        "pre_loss_path": _pack(adv_pre_loss),
        "fc_adv_gap_succ_minus_fail": gap,
        "cohens_d_fc_succ_vs_fail": d,
        "sign_flip_vs_baseline_mean": float(np.mean(sign_flip_rate)) if sign_flip_rate else None,
        "note": (
            "Succ/fail gap is association under fixed policy, not causal action ranking. "
            "V(s) does not rank actions from the same state."
        ),
    }


def _values_from_preds(episodes, blob, *, a=1.0, b=0.0) -> list[np.ndarray]:
    """Rebuild per-episode value arrays from cached flat predictions."""
    cal = apply_affine(blob["pred"], a, b)
    out = []
    for ei, ep in enumerate(episodes):
        T = ep["n_steps"]
        # predictions are in episode order from build_step_index
        idx = np.where(blob["ep_idx"] == ei)[0]
        assert len(idx) == T, (ei, len(idx), T)
        # ensure time order
        order = np.argsort(blob["t_idx"][idx])
        out.append(cal[idx[order]])
    return out


def run_advantage() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    eval_eps = load_episodes("eval")
    ind = json.loads((OUT / "INDEPENDENT_IC_VALIDATION.json").read_text(encoding="utf-8"))
    sel = ind["results"]

    configs = {
        "P64_raw": ("P64", 1.0, 0.0),
        "A_w1_raw": ("A_w1", 1.0, 0.0),
        "A_w1_cal": ("A_w1", float(sel["A_w1"]["a"]), float(sel["A_w1"]["b"])),
        "P64_cal": ("P64", float(sel["P64"]["a"]), float(sel["P64"]["b"])),
    }

    report = {}
    for tag, (name, a, b) in configs.items():
        blob = load_preds(name, "eval")
        vals = _values_from_preds(eval_eps, blob, a=a, b=b)
        report[tag] = {"a": a, "b": b, "critic": name, **_adv_for_values(eval_eps, vals)}
        print(
            f"[adv] {tag} gap={report[tag]['fc_adv_gap_succ_minus_fail']} "
            f"d={report[tag]['cohens_d_fc_succ_vs_fail']}",
            flush=True,
        )

    # P73 CF reference (external): action matters from matched FC states
    p73 = Path(__file__).resolve().parents[3] / "results/experiments/p73_contact_formation"
    cf_note = {
        "external": "P73 matched FC counterfactuals showed M111 sequence changes can alter R3/opposing.",
        "implication": (
            "Good Critic values are necessary but not sufficient for action-level credit; "
            "advantage quality also needs distinguishable Q-like separation along executed transitions."
        ),
    }
    report["p73_cf_reference"] = cf_note
    report["comparison"] = {
        "gap_P64_raw": report["P64_raw"]["fc_adv_gap_succ_minus_fail"],
        "gap_A_w1_raw": report["A_w1_raw"]["fc_adv_gap_succ_minus_fail"],
        "gap_A_w1_cal": report["A_w1_cal"]["fc_adv_gap_succ_minus_fail"],
        "cal_improves_gap_vs_A_raw": (
            report["A_w1_cal"]["fc_adv_gap_succ_minus_fail"] is not None
            and report["A_w1_raw"]["fc_adv_gap_succ_minus_fail"] is not None
            and report["A_w1_cal"]["fc_adv_gap_succ_minus_fail"]
            > report["A_w1_raw"]["fc_adv_gap_succ_minus_fail"] + 0.05
        ),
        "cal_beats_P64_gap": (
            report["A_w1_cal"]["fc_adv_gap_succ_minus_fail"] is not None
            and report["P64_raw"]["fc_adv_gap_succ_minus_fail"] is not None
            and report["A_w1_cal"]["fc_adv_gap_succ_minus_fail"]
            > report["P64_raw"]["fc_adv_gap_succ_minus_fail"] + 0.05
        ),
    }

    (OUT / "ADVANTAGE_DIAGNOSTICS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "ADVANTAGE_DIAGNOSTICS.md").write_text(
        "# Advantage Diagnostics (P76 Phase 3)\n\nGAE γ=1, λ=0.95; Actor frozen.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    run_advantage()
