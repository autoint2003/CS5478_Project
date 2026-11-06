"""Phase 6 — control / action relevance (no policy training)."""

from __future__ import annotations

import json

import numpy as np

from experiments.precision.p77_contact_observability.paths import OUT, P73_CF, SEED


def _p73_action_matters() -> dict:
    if not P73_CF.exists():
        return {"available": False}
    cf = json.loads(P73_CF.read_text(encoding="utf-8"))
    fc = cf.get("cf_results", {}).get("first_contact", {})
    base = fc.get("baseline", {})
    rows = []
    for rule, blob in fc.items():
        if rule == "baseline":
            continue
        rows.append(
            {
                "rule": rule,
                "d_opp": float(blob.get("reached_opposing", 0) - base.get("reached_opposing", 0)),
                "d_r3": float(blob.get("r3_ready", 0) - base.get("r3_ready", 0)),
                "d_success": float(blob.get("SUCCESS_RELAXED", 0) - base.get("SUCCESS_RELAXED", 0)),
                "r3": blob.get("r3_ready"),
                "opp": blob.get("reached_opposing"),
            }
        )
    rows.sort(key=lambda r: -abs(r["d_r3"]))
    return {
        "available": True,
        "baseline": base,
        "top_deltas": rows[:8],
        "soft_approach": next((r for r in rows if r["rule"] == "soft_approach"), None),
        "implication": (
            "Matched FC counterfactuals show M111 sequence choice changes R3/success — "
            "action-relevant discrimination exists at FC. New observations help only if they "
            "identify which matched states need which sequence."
        ),
    }


def _pair_action_relevance() -> dict:
    """From Phase 3 matched pairs + sensing diag if available."""
    alias = json.loads((OUT / "RARE_POSITIVE_ALIASING.json").read_text(encoding="utf-8"))
    pairs = alias.get("matched_pairs_diff_sidewall") or {}
    gain = {}
    gpath = OUT / "INCREMENTAL_INFORMATION_GAIN.json"
    if gpath.exists():
        gain = json.loads(gpath.read_text(encoding="utf-8"))
    sw_base = ((gain.get("tasks") or {}).get("correct_sidewall_now") or {}).get("base") or {}
    sw_ft = ((gain.get("tasks") or {}).get("correct_sidewall_now") or {}).get("plus_ft") or {}
    return {
        "1_better_future_success_prediction": {
            "p75_style": "See EXISTING_OBSERVATION_INFORMATION / info-gain eventual_opposing AUROC.",
            "note": "Classification of eventual success ≠ action ranking.",
        },
        "2_better_current_geometry": {
            "sidewall_auroc_base": sw_base.get("auroc"),
            "sidewall_auroc_plus_ft": sw_ft.get("auroc"),
            "delta": None
            if sw_base.get("auroc") is None or sw_ft.get("auroc") is None
            else sw_ft["auroc"] - sw_base["auroc"],
        },
        "3_action_choice_discrimination": {
            "p73_cf": "Matched states: different M111 overlays → different R3 (soft_approach helps).",
            "matched_obs_diff_sidewall_pairs": pairs,
            "action_differs_frac": pairs.get("frac_action_differs"),
            "outcome_differs_frac": pairs.get("frac_outcome_differs"),
            "sufficient_for_control_claim": False,
            "reason": (
                "Obs-matched pairs with different sidewall show outcome/action divergence associations, "
                "but do not prove that a sensing feature recovers the CF-optimal action. "
                "Need sensing that separates CF-success vs CF-fail state groups before Actor expansion."
            ),
        },
    }


def run_action_relevance() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "p73_counterfactuals": _p73_action_matters(),
        "discrimination": _pair_action_relevance(),
        "conclusion_gate": {
            "improved_classification_alone_insufficient": True,
            "require_action_separation": True,
        },
    }
    # Geometry AUROC lift ≠ proof of action-choice discrimination.
    # Matched-pair action_differs is only associative; P73 shows actions can matter externally.
    # Do NOT set action_relevant=True from classification lift alone.
    d = report["discrimination"]["2_better_current_geometry"].get("delta")
    ad = report["discrimination"]["3_action_choice_discrimination"].get("action_differs_frac")
    report["action_relevant_sensing_signal"] = False
    report["action_relevance_note"] = (
        f"Ft geometry ΔAUROC={d}; matched-pair action_differs={ad}. "
        "Classification improvement alone is insufficient for Actor obs expansion (Case C gate). "
        "P73 CF remains external evidence that M111 choice matters at FC."
    )
    (OUT / "ACTION_RELEVANCE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "ACTION_RELEVANCE.md").write_text(
        "# Action Relevance (P77 Phase 6)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"action_relevant": report["action_relevant_sensing_signal"], "geo_delta": d}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_action_relevance()
