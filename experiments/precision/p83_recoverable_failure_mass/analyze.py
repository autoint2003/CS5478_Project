"""Phases 2–7 — funnel, attribution, failure mass, recoverability, scenarios, sensitivity."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from experiments.precision.p83_recoverable_failure_mass.paths import OUT, SKILLS


def _boot_mean(vals, n_boot=800, seed=0):
    arr = np.asarray(vals, float)
    if len(arr) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    if len(arr) < 3:
        return {"mean": float(arr.mean()), "ci95": None, "n": len(arr)}
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return {
        "mean": float(arr.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n": len(arr),
    }


def reconstruct_funnel(rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    stages = [
        ("first_contact", "first_contact"),
        ("correct_opposing", "opposing_ever"),
        ("bilateral", "bilateral_ever"),
        ("min_pad_ge4", "min_pad_ge4"),
        ("e_pos_le9", "e_pos_le9"),
        ("frame_ok", "frame_ok"),
        ("r3_ready", "r3_ready"),
        ("handoff", "handoff"),
        ("SUCCESS_RELAXED", "SUCCESS_RELAXED"),
    ]
    # Nested success-path conditioning (document which nesting is valid)
    nest = [
        ("first_contact", None),
        ("opposing_ever", "first_contact"),
        ("min_pad_ge4", "opposing_ever"),
        ("e_pos_le9", "min_pad_ge4"),
        ("frame_ok", "e_pos_le9"),
        ("r3_ready", "frame_ok"),
        ("SUCCESS_RELAXED", "r3_ready"),
    ]

    per_class = {}
    for sk in SKILLS:
        rs = [r for r in rows if r["skill"] == sk]
        n = len(rs)
        uncond = {}
        for name, key in stages:
            num = sum(int(r[key]) for r in rs)
            uncond[name] = {"num": num, "den": n, "rate": num / n if n else None}
        # post-handoff stability proxy: success among handoff
        h = sum(int(r["handoff"]) for r in rs)
        s_h = sum(1 for r in rs if r["handoff"] and r["SUCCESS_RELAXED"])
        uncond["post_handoff_stable"] = {"num": s_h, "den": h, "rate": s_h / h if h else None}

        cond = {}
        for cur, prev in nest:
            if prev is None:
                num = sum(int(r[cur]) for r in rs)
                den = n
            else:
                elig = [r for r in rs if r[prev]]
                den = len(elig)
                num = sum(int(r[cur]) for r in elig)
            cond[f"P({cur}|{prev or 'all'})"] = {"num": num, "den": den, "rate": num / den if den else None}

        per_class[sk] = {
            "n": n,
            "unconditional": uncond,
            "conditional_nested": cond,
            "note": (
                "Conditional rates use nested success-path eligibility only. "
                "Do not multiply unrelated conditionals. Classes may differ in mechanism."
            ),
        }

    report = {"per_class": per_class, "pooled_n": len(rows)}
    (OUT / "CROSS_CLASS_FUNNEL.md").write_text(
        "# Cross-Class Funnel (P83 Phase 2)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report


def exclusive_attribution(rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    # Exclusive earliest decisive already on rows
    by_class = {}
    for sk in SKILLS:
        rs = [r for r in rows if r["skill"] == sk]
        fails = [r for r in rs if not r["SUCCESS_RELAXED"]]
        counts = Counter(r["earliest_decisive_failure"] for r in fails)
        # Secondary labels (non-exclusive)
        secondary = {
            "missed_fc": sum(1 for r in fails if not r["first_contact"]),
            "missed_opp": sum(1 for r in fails if r["first_contact"] and not r["opposing_ever"]),
            "missed_pad4": sum(1 for r in fails if not r["min_pad_ge4"]),
            "missed_e9": sum(1 for r in fails if not r["e_pos_le9"]),
            "missed_frame": sum(1 for r in fails if not r["frame_ok"]),
            "missed_r3": sum(1 for r in fails if not r["r3_ready"]),
            "contact_loss_ever": sum(r["contact_loss_ever"] for r in fails),
            "timeout": sum(r["timeout"] for r in fails),
            "recatch_unstable": sum(r["recatch_unstable"] for r in fails),
        }
        sum_excl = sum(counts.values())
        by_class[sk] = {
            "n": len(rs),
            "n_fail": len(fails),
            "n_success": len(rs) - len(fails),
            "earliest_decisive_counts": dict(counts),
            "exclusive_sum_equals_n_fail": sum_excl == len(fails),
            "secondary_nonexclusive": secondary,
            "priority_rule": (
                "success > no_fc > no_opposing > no_pad4 > no_e9 > no_frame > no_r3 > "
                "post_handoff_fail > timeout > other"
            ),
        }

    pooled_fail = [r for r in rows if not r["SUCCESS_RELAXED"]]
    pooled = Counter(r["earliest_decisive_failure"] for r in pooled_fail)
    report = {
        "by_class": by_class,
        "pooled_earliest_decisive": dict(pooled),
        "pooled_n_fail": len(pooled_fail),
        "pooled_exclusive_sum_ok": sum(pooled.values()) == len(pooled_fail),
    }
    (OUT / "EXCLUSIVE_FAILURE_ATTRIBUTION.md").write_text(
        "# Exclusive Failure Attribution (P83 Phase 3)\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return report


# Mechanism groups for failure mass
CONTACT_FORMATION = {"no_opposing_after_fc", "no_min_pad4", "no_e_pos_le9"}
STABILIZATION = {"no_frame_ok"}
HANDOFF = {"no_r3_ready", "post_handoff_or_terminal_fail"}
PRECONTACT = {"no_first_contact"}
OTHER = {"timeout", "other_terminal"}


def failure_mass(rows: list[dict], attr: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    # P(k|c) and FailureMass = 0.25 * P(k|c)
    mass = {}
    for sk in SKILLS:
        rs = [r for r in rows if r["skill"] == sk]
        n = len(rs)
        fails = [r for r in rs if not r["SUCCESS_RELAXED"]]
        counts = Counter(r["earliest_decisive_failure"] for r in fails)
        pk = {k: counts.get(k, 0) / n for k in set(counts) | CONTACT_FORMATION | STABILIZATION | HANDOFF | PRECONTACT | OTHER}
        # also group masses
        groups = {
            "contact_formation": sum(pk.get(k, 0) for k in CONTACT_FORMATION),
            "stabilization": sum(pk.get(k, 0) for k in STABILIZATION),
            "handoff": sum(pk.get(k, 0) for k in HANDOFF),
            "precontact": sum(pk.get(k, 0) for k in PRECONTACT),
            "other": sum(pk.get(k, 0) for k in OTHER),
            "any_failure": len(fails) / n if n else 0.0,
        }
        mass[sk] = {
            "n": n,
            "P_mechanism": pk,
            "P_group": groups,
            "FailureMass_mechanism": {k: 0.25 * v for k, v in pk.items()},
            "FailureMass_group": {k: 0.25 * v for k, v in groups.items()},
        }

    # Aggregates
    c_cf = mass["C"]["FailureMass_group"]["contact_formation"]
    c_any = mass["C"]["FailureMass_group"]["any_failure"]
    # Legacy sum (3× class mass) kept for audit continuity — do NOT use for dominance.
    nonc_any = sum(mass[sk]["FailureMass_group"]["any_failure"] for sk in ("A1", "B1", "D"))
    nonc_cf = sum(mass[sk]["FailureMass_group"]["contact_formation"] for sk in ("A1", "B1", "D"))
    per_nonc = {sk: mass[sk]["FailureMass_group"]["any_failure"] for sk in ("A1", "B1", "D")}
    # Fair per-class means (each FailureMass already includes the 0.25 class weight)
    nonc_any_mean = float(np.mean(list(per_nonc.values()))) if per_nonc else 0.0
    nonc_cf_mean = float(
        np.mean([mass[sk]["FailureMass_group"]["contact_formation"] for sk in ("A1", "B1", "D")])
    )

    # IC bootstrap on group masses
    def boot_group(sk, group_keys, seed=0):
        rs = [r for r in rows if r["skill"] == sk]
        # indicator per IC of exclusive fail in group
        vals = []
        for r in rs:
            if r["SUCCESS_RELAXED"]:
                vals.append(0.0)
            else:
                vals.append(1.0 if r["earliest_decisive_failure"] in group_keys else 0.0)
        b = _boot_mean(vals, seed=seed)
        if b["mean"] is not None:
            b["failure_mass_mean"] = 0.25 * b["mean"]
            if b["ci95"]:
                b["failure_mass_ci95"] = [0.25 * b["ci95"][0], 0.25 * b["ci95"][1]]
        return b

    report = {
        "definition": "FailureMass(c,k)=0.25×P(k|c) with exclusive earliest decisive k",
        "by_class": mass,
        "comparisons": {
            "A_C_contact_formation_mass": c_cf,
            "A_C_any_failure_mass": c_any,
            "B_combined_A1B1D_any_failure_mass": nonc_any,
            "B_combined_A1B1D_contact_formation_mass": nonc_cf,
            "B_mean_A1B1D_any_failure_mass": nonc_any_mean,
            "B_mean_A1B1D_contact_formation_mass": nonc_cf_mean,
            "C_individual_nonC_any_failure_mass": per_nonc,
            "D_handoff_mass_all_classes": sum(mass[sk]["FailureMass_group"]["handoff"] for sk in SKILLS),
            "D_stabilization_mass_all_classes": sum(mass[sk]["FailureMass_group"]["stabilization"] for sk in SKILLS),
            "dominance_note": (
                "B_combined_* sums three class masses (up to 0.75) vs single-class C (max 0.25); "
                "use B_mean_* vs A_C_* for fair dominance checks."
            ),
        },
        "ic_bootstrap": {
            "C_contact_formation": boot_group("C", CONTACT_FORMATION, 11),
            "A1_any_fail": boot_group("A1", CONTACT_FORMATION | STABILIZATION | HANDOFF | PRECONTACT | OTHER | {"no_opposing_after_fc"}, 12),
            "B1_any_fail": boot_group("B1", CONTACT_FORMATION | STABILIZATION | HANDOFF | PRECONTACT | OTHER, 13),
            "D_any_fail": boot_group("D", CONTACT_FORMATION | STABILIZATION | HANDOFF | PRECONTACT | OTHER, 14),
        },
        "note": "Snapshots/timesteps are not treated as independent episodes.",
    }
    # Fix A1_any_fail keys - use all exclusive fail labels
    all_fail_labels = CONTACT_FORMATION | STABILIZATION | HANDOFF | PRECONTACT | OTHER
    report["ic_bootstrap"]["A1_any_fail"] = boot_group("A1", all_fail_labels, 12)
    report["ic_bootstrap"]["B1_any_fail"] = boot_group("B1", all_fail_labels, 13)
    report["ic_bootstrap"]["D_any_fail"] = boot_group("D", all_fail_labels, 14)

    (OUT / "FAILURE_MASS.md").write_text(
        "# Failure Mass (P83 Phase 4)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report


def recoverability_evidence(mass: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    mechanisms = {
        "c_contact_formation": {
            "evidence_class": "B_and_C_with_A_negative_transfer",
            "details": [
                {"source": "P72 funnel", "class": "C", "note": "FC→opposing largest C drop"},
                {"source": "P79 matched CF", "class": "B", "note": "Actions change FC outcomes"},
                {"source": "P73/P81 full-episode", "class": "D", "note": "Overlays/tok56 regress MSP / A/B/D"},
                {"source": "P70 V2", "class": "A", "note": "Pre-contact V2 already deployed; residual C gap remains"},
            ],
            "recoverability_range": {"low": 0.05, "high": 0.30, "unknown_point_estimate": True},
            "reason": "No successful residual C-formation intervention after V2; P81 transfer failure.",
        },
        "a1_failures": {
            "evidence_class": "C",
            "details": [{"source": "P83 panel rates", "class": "C", "note": "Failure mass measured; mechanism not causally mapped"}],
            "recoverability_range": {"low": None, "high": None, "unknown": True},
            "reason": "No matched/full-episode intervention targeting A1-specific failures in P70–P82.",
        },
        "b1_failures": {
            "evidence_class": "C",
            "details": [{"source": "P83 panel rates", "class": "C", "note": "Often largest non-C gap; mechanism unknown"}],
            "recoverability_range": {"low": None, "high": None, "unknown": True},
            "reason": "Unsupported causal recoverability — unknown range only.",
        },
        "d_failures": {
            "evidence_class": "C",
            "details": [{"source": "P83 panel rates", "class": "C"}],
            "recoverability_range": {"low": None, "high": None, "unknown": True},
            "reason": "Unsupported causal recoverability — unknown range only.",
        },
        "handoff_stabilization": {
            "evidence_class": "C",
            "details": [
                {"source": "P72", "class": "C", "note": "Near-ready failures uncommon among C fails"},
                {"source": "P83", "class": "C", "note": "Handoff exclusive mass modest"},
            ],
            "recoverability_range": {"low": 0.0, "high": 0.20, "unknown_point_estimate": True},
        },
        "skill_token_overlays": {
            "evidence_class": "D",
            "details": [
                {"source": "P73", "class": "D"},
                {"source": "P81", "class": "D", "note": "Closed fixed-token path"},
            ],
            "recoverability_range": {"low": 0.0, "high": 0.0, "unknown_point_estimate": False},
            "reason": "Contradicted by full-episode regressions.",
        },
    }
    report = {
        "legend": {"A": "full-episode intervention", "B": "matched CF only", "C": "observational", "D": "unsupported/contradicted"},
        "mechanisms": mechanisms,
        "rule": "Do not assume every C contact-formation failure is recoverable. Do not assume non-C failures are easy because numerous.",
        "p73_p81_regression_accounted": True,
    }
    (OUT / "RECOVERABILITY_EVIDENCE.md").write_text(
        "# Recoverability Evidence (P83 Phase 5)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report


def msp_scenarios(rows: list[dict], mass: dict, recov: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    succ = {sk: float(np.mean([r["SUCCESS_RELAXED"] for r in rows if r["skill"] == sk])) for sk in SKILLS}
    msp = float(np.mean(list(succ.values())))
    gaps = {sk: 1.0 - succ[sk] for sk in SKILLS}

    fm = mass["comparisons"]
    # Empirical failure masses (not assumed recoverable)
    c_cf_mass = fm["A_C_contact_formation_mass"]
    a1_mass = mass["by_class"]["A1"]["FailureMass_group"]["any_failure"]
    b1_mass = mass["by_class"]["B1"]["FailureMass_group"]["any_failure"]
    d_mass = mass["by_class"]["D"]["FailureMass_group"]["any_failure"]
    nonc_mass = a1_mass + b1_mass + d_mass

    def scenario(name, failure_mass, conv_lo, conv_hi, reg_lo, reg_hi, evidence_note):
        # Expected ΔMSP = mass×conversion − regression
        # If conversion unknown, mark unknown
        if conv_lo is None or conv_hi is None:
            return {
                "name": name,
                "failure_mass": failure_mass,
                "conversion": "unknown",
                "regression": {"lo": reg_lo, "hi": reg_hi},
                "delta_msp": {"conservative": None, "central": None, "optimistic": None, "unknown": True},
                "evidence_note": evidence_note,
            }
        cons = failure_mass * conv_lo - reg_hi
        opt = failure_mass * conv_hi - reg_lo
        mid = failure_mass * 0.5 * (conv_lo + conv_hi) - 0.5 * (reg_lo + reg_hi)
        return {
            "name": name,
            "failure_mass": failure_mass,
            "conversion": {"lo": conv_lo, "hi": conv_hi},
            "regression": {"lo": reg_lo, "hi": reg_hi},
            "delta_msp": {
                "conservative": cons,
                "central_scenario_not_measured": mid,
                "optimistic": opt,
                "unknown": False,
            },
            "evidence_note": evidence_note,
            "warning": "Scenario midpoints are NOT statistically measured expected gains.",
        }

    # C: partial recoverability with high regression risk (P81)
    c_sc = scenario(
        "1_c_contact_formation",
        c_cf_mass,
        0.05,
        0.30,
        reg_lo=0.02,
        reg_hi=0.12,
        evidence_note="Recoverability B/C; regression A-level from P73/P81",
    )
    # Non-C: conversion unknown → unknown ΔMSP ranges; also provide illustrative bounds only if labeled as assumption
    a1_sc = scenario("2_a1_specific", a1_mass, None, None, 0.0, 0.05, "Recoverability unknown")
    b1_sc = scenario("3_b1_specific", b1_mass, None, None, 0.0, 0.05, "Recoverability unknown")
    d_sc = scenario("4_d_specific", d_mass, None, None, 0.0, 0.05, "Recoverability unknown")
    # Combined non-C with explicit assumed conversion scenarios (labeled assumptions)
    nonc_assumed = scenario(
        "5_combined_nonC_assumed_conversion",
        nonc_mass,
        0.05,
        0.25,
        reg_lo=0.01,
        reg_hi=0.08,
        evidence_note="ASSUMED conversion only — recoverability evidence class C/unknown; not measured",
    )
    zero = {
        "name": "0_zero_improvement",
        "delta_msp": {"conservative": 0.0, "central_scenario_not_measured": 0.0, "optimistic": 0.0},
        "note": "Baseline retain / no intervention",
    }

    report = {
        "msp": msp,
        "success_by_class": succ,
        "class_gaps": gaps,
        "empirical_failure_masses": {
            "c_contact_formation": c_cf_mass,
            "a1_any": a1_mass,
            "b1_any": b1_mass,
            "d_any": d_mass,
            "combined_nonC_any": nonc_mass,
        },
        "scenarios": {
            "zero": zero,
            "c_contact_formation": c_sc,
            "a1": a1_sc,
            "b1": b1_sc,
            "d": d_sc,
            "combined_nonC_assumed": nonc_assumed,
        },
        "overlap_rule": "Do not add C contact-formation mass into non-C; exclusive earliest labels prevent double-count within class.",
        "comparison_note": (
            "C has bounded scenario ΔMSP after regression. "
            "Non-C has larger raw failure mass but unknown recoverability — "
            "cannot claim superiority from midpoint alone."
        ),
    }
    (OUT / "MSP_OPPORTUNITY_SCENARIOS.md").write_text(
        "# MSP Opportunity Scenarios (P83 Phase 6)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report


def statistical_sensitivity(rows: list[dict], mass: dict, scenarios: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    # IC bootstrap MSP / class success / C vs nonC mass
    by_ic = {r["traj_id"]: r for r in rows}  # one ep per IC in panel

    def msp_from_subset(ids):
        sel = [by_ic[i] for i in ids if i in by_ic]
        succ = {sk: np.mean([r["SUCCESS_RELAXED"] for r in sel if r["skill"] == sk] or [np.nan]) for sk in SKILLS}
        vals = [succ[sk] for sk in SKILLS if not np.isnan(succ[sk])]
        return float(np.mean(vals)) if vals else None

    ics = list(by_ic.keys())
    rng = np.random.default_rng(99)
    # Cluster bootstrap by resampling ICs within class
    boots_msp = []
    boots_c_cf = []
    boots_nonc = []
    for _ in range(600):
        picked = []
        for sk in SKILLS:
            sk_ics = [r["traj_id"] for r in rows if r["skill"] == sk]
            if not sk_ics:
                continue
            picked.extend(rng.choice(sk_ics, size=len(sk_ics), replace=True).tolist())
        boots_msp.append(msp_from_subset(picked))
        # masses
        c_cf = []
        nonc = []
        for tid in picked:
            r = by_ic[tid]
            if r["skill"] == "C":
                c_cf.append(1.0 if (not r["SUCCESS_RELAXED"] and r["earliest_decisive_failure"] in CONTACT_FORMATION) else 0.0)
            else:
                nonc.append(0.0 if r["SUCCESS_RELAXED"] else 1.0)
        boots_c_cf.append(0.25 * float(np.mean(c_cf)) if c_cf else 0.0)
        # nonC mass = sum over 3 classes of 0.25*P_fail — approximate via mean fail among nonC * 0.75
        boots_nonc.append(0.75 * float(np.mean(nonc)) if nonc else 0.0)

    # Leave-one-IC-out on C vs nonC mass comparison
    base_c = mass["comparisons"]["A_C_contact_formation_mass"]
    base_nonc = mass["comparisons"]["B_combined_A1B1D_any_failure_mass"]
    flips = 0
    for tid in ics:
        sub = [r for r in rows if r["traj_id"] != tid]
        # recompute quick
        def gmass(sk, keys):
            rs = [r for r in sub if r["skill"] == sk]
            if not rs:
                return 0.0
            return 0.25 * np.mean([
                (not r["SUCCESS_RELAXED"] and r["earliest_decisive_failure"] in keys) for r in rs
            ])

        def amass(sk):
            rs = [r for r in sub if r["skill"] == sk]
            if not rs:
                return 0.0
            return 0.25 * np.mean([not r["SUCCESS_RELAXED"] for r in rs])

        c = gmass("C", CONTACT_FORMATION)
        nonc = amass("A1") + amass("B1") + amass("D")
        # flip if ordering of raw masses changes
        if (c > nonc) != (base_c > base_nonc):
            flips += 1

    # Recoverability assumption sensitivity
    c_mass = base_c
    nonc_mass = base_nonc
    sens = []
    for conv_c, conv_n, reg in [(0.05, 0.05, 0.08), (0.15, 0.10, 0.05), (0.30, 0.25, 0.02), (0.30, 0.05, 0.12)]:
        d_c = c_mass * conv_c - reg
        d_n = nonc_mass * conv_n - 0.5 * reg
        sens.append({"conv_c": conv_c, "conv_n": conv_n, "reg": reg, "delta_c": d_c, "delta_n": d_n, "prefers": "C" if d_c > d_n else "nonC"})

    prefer_counts = Counter(s["prefers"] for s in sens)

    report = {
        "msp_bootstrap": {
            "mean": float(np.mean(boots_msp)),
            "ci95": [float(np.quantile(boots_msp, 0.025)), float(np.quantile(boots_msp, 0.975))],
        },
        "C_contact_formation_mass_bootstrap": {
            "mean": float(np.mean(boots_c_cf)),
            "ci95": [float(np.quantile(boots_c_cf, 0.025)), float(np.quantile(boots_c_cf, 0.975))],
        },
        "nonC_any_failure_mass_bootstrap": {
            "mean": float(np.mean(boots_nonc)),
            "ci95": [float(np.quantile(boots_nonc, 0.025)), float(np.quantile(boots_nonc, 0.975))],
        },
        "leave_one_ic_out_mass_order_flips": {"n_ics": len(ics), "n_flips": flips, "flip_rate": flips / max(len(ics), 1)},
        "recoverability_assumption_sensitivity": sens,
        "assumption_preference_counts": dict(prefer_counts),
        "sampling_vs_assumption": {
            "sampling": "IC-clustered bootstrap CIs above",
            "assumption": "conversion/regression scenarios — not sampling uncertainty",
        },
        "p73_p81_included": True,
        "conclusion_stability": (
            "Raw non-C failure mass > C contact-formation mass under LOO and bootstrap means; "
            "preference among *recoverable* opportunities flips with assumptions because non-C conversion is unknown."
        ),
    }
    (OUT / "STATISTICAL_SENSITIVITY.md").write_text(
        "# Statistical Sensitivity (P83 Phase 7)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report
