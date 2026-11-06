"""Phases 6–7 — mechanism diagnosis + cross-class risk."""

from __future__ import annotations

import json
import pickle
from collections import Counter

import numpy as np

from experiments.precision.p81_static_contact_bias.paths import CACHE, OUT, TOKEN_56


def write_mechanism(semantics: dict, dist: dict, bias: dict, val: dict, timing: dict, full: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fam = (semantics.get("codebook_decode") or {}).get("family")
    grip = (semantics.get("components") or {}).get("gripper")
    fc = dist.get("fc_decision") or {}
    stage = dist.get("by_stage") or {}

    # What does Actor pick at FC?
    fc_fams = (stage.get("first_actor_owned_fc") or {}).get("families") or {}
    top_toks = (stage.get("first_actor_owned_fc") or {}).get("top_tokens") or []

    explanations = {
        "1_grip_too_early": {
            "plausible": bool(fc_fams.get("GRIP", 0) >= fc_fams.get("APPROACH", 0) and fam == "APPROACH"),
            "evidence": {"fc_families": fc_fams, "tok56_family": fam},
        },
        "2_leaves_approach_too_early": {
            "plausible": bool(fam in ("APPROACH", "ORIENT") and fc.get("frac_select_56", 1) < 0.2),
            "evidence": {"frac_select_56": fc.get("frac_select_56"), "mean_p_tok56": fc.get("mean_p_tok56")},
        },
        "3_excessive_motion_near_contact": {
            "plausible": bool(
                (semantics.get("components") or {}).get("lin_norm", 0) < 0.3
                and any(t[0] != TOKEN_56 for t in top_toks[:3])
            ),
            "evidence": {"tok56_lin_norm": (semantics.get("components") or {}).get("lin_norm"), "top_tokens": top_toks[:5]},
        },
        "4_poor_probability_calibration": {
            "plausible": bool(
                fc.get("mean_p_tok56") is not None
                and fc.get("frac_select_56") is not None
                and abs(fc["mean_p_tok56"] - fc["frac_select_56"]) > 0.1
            ),
            "evidence": {"mean_p": fc.get("mean_p_tok56"), "frac_select": fc.get("frac_select_56"), "frac_argmax_56": fc.get("frac_argmax_is_56")},
        },
        "5_generic_contact_preserving": {
            "plausible": bool((val.get("tok56_success_mean") or 0) > (val.get("actor_success_mean") or 0)),
            "evidence": {
                "timing_broad": (timing.get("narrow_vs_broad") or {}).get("helps_multiple_stages"),
                "d_loss": None,
            },
        },
        "6_candidate_set_or_eval_artifact": {
            "plausible": bool(
                (bias.get("artifact_assessment") or {}).get("many_ties")
                or (bias.get("impact_concentration") or {}).get("dominated_by_few_states")
                or not val.get("ci_excludes_0_for_succ")
            ),
            "evidence": {
                "categories": bias.get("categories"),
                "top3_share": (bias.get("impact_concentration") or {}).get("top3_share_of_positive_delta_succ"),
                "ci": (val.get("paired_delta_succ_ic") or {}).get("ci95"),
                "oracle_note": (bias.get("artifact_assessment") or {}).get("note"),
            },
        },
    }

    # Rank by plausible flags + strength
    supported = [k for k, v in explanations.items() if v["plausible"]]
    primary = "6_candidate_set_or_eval_artifact"
    if "1_grip_too_early" in supported and fc_fams.get("GRIP", 0) > 0:
        primary = "1_grip_too_early"
    elif val.get("ci_excludes_0_for_succ") and "2_leaves_approach_too_early" in supported:
        primary = "2_leaves_approach_too_early"
    elif "6_candidate_set_or_eval_artifact" in supported and not val.get("ci_excludes_0_for_succ"):
        primary = "6_candidate_set_or_eval_artifact"

    report = {
        "tok56_decoded_family": fam,
        "tok56_gripper": grip,
        "explanations": explanations,
        "primary_explanation": primary,
        "static_preference_implies_obs_deficiency": False,
        "correlation_vs_intervention": {
            "distribution_underselect": fc.get("systematic_underselect"),
            "matched_intervention_ci_excludes_0": val.get("ci_excludes_0_for_succ"),
            "full_episode_msp_delta": full.get("msp_delta"),
        },
        "note": "Do not modify reward or observation representation based on this diagnostic.",
    }
    (OUT / "MECHANISM_DIAGNOSIS.md").write_text(
        "# Mechanism Diagnosis (P81 Phase 6)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report


def write_cross_class(full: dict, val: dict, timing: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    per = full.get("per_class") or {}
    abd = full.get("abd_deltas") or {}
    report = {
        "help_c_without_harm_abd": None,
        "risks": {
            "disrupt_already_correct_contact": (timing.get("already_opposing_adverse") or {}).get("mean_d_succ"),
            "abd_deltas_full_episode": abd,
            "handoff": (full.get("secondary") or {}),
            "timeout_proxy_contact_loss": {
                "baseline": (full.get("secondary") or {}).get("contact_loss_baseline"),
                "inject56": (full.get("secondary") or {}).get("contact_loss_inject56"),
            },
        },
        "c_only_insufficient_for_authorization": True,
        "authorize_policy_modification": False,
        "analysis": [],
    }
    if per:
        c_d = (per.get("C") or {}).get("mean_d_succ")
        abd_harm = any(v is not None and v < -0.05 for v in abd.values())
        report["help_c_without_harm_abd"] = bool(c_d is not None and c_d > 0 and not abd_harm)
        report["analysis"].append(
            f"C ΔSUCCESS={c_d}; A/B/D deltas={abd}; MSP Δ={full.get('msp_delta')}"
        )
    if full.get("skipped"):
        report["analysis"].append("Full-episode transfer skipped — cross-class risk inferred as unresolved.")
        report["help_c_without_harm_abd"] = False

    (OUT / "CROSS_CLASS_RETENTION.md").write_text(
        "# Cross-Class Retention (P81 Phase 7)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report
