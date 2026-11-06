"""Phases 2–3 — cross-class failure budget + skill-transition analysis."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from experiments.precision.p82_global_failure_budget.paths import OUT, SKILLS


def _boot_rate(vals: list[int], seed: int = 0) -> dict:
    arr = np.asarray(vals, float)
    if len(arr) < 3:
        return {"mean": float(arr.mean()) if len(arr) else None, "ci95": None, "n": len(arr)}
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(600)]
    return {
        "mean": float(arr.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n": len(arr),
    }


def failure_budget(rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    by_skill = defaultdict(list)
    for r in rows:
        by_skill[r["skill"]].append(r)

    metrics = [
        "SUCCESS_RELAXED",
        "first_contact",
        "correct_sidewall_ever",
        "bilateral_ever",
        "opposing_ever",
        "min_pad_ge4",
        "e_pos_le9",
        "frame_ok",
        "r3_ready",
        "handoff",
        "contact_loss_ever",
        "recatch_unstable",
        "timeout",
    ]

    per_class = {}
    for sk in SKILLS:
        rs = by_skill.get(sk, [])
        per_class[sk] = {
            "n": len(rs),
            "rates": {m: _boot_rate([r[m] for r in rs], seed=hash(sk + m) % 10000) for m in metrics},
            "earliest_decisive": dict(Counter(r["earliest_decisive_failure"] for r in rs)),
        }

    # Overall MSP = equal-weight mean of class SUCCESS
    succ_by_class = {sk: per_class[sk]["rates"]["SUCCESS_RELAXED"]["mean"] for sk in SKILLS if per_class[sk]["n"]}
    msp = float(np.mean(list(succ_by_class.values()))) if succ_by_class else None

    # Conditional transitions (pooled and per C)
    def transitions(rs):
        n = max(len(rs), 1)
        fc = sum(r["first_contact"] for r in rs)
        opp = sum(r["opposing_ever"] for r in rs)
        pad = sum(r["min_pad_ge4"] for r in rs)
        e9 = sum(r["e_pos_le9"] for r in rs)
        fr = sum(r["frame_ok"] for r in rs)
        r3 = sum(r["r3_ready"] for r in rs)
        succ = sum(r["SUCCESS_RELAXED"] for r in rs)
        return {
            "P(fc)": fc / n,
            "P(opp|fc)": opp / max(fc, 1),
            "P(pad4|opp)": pad / max(opp, 1),
            "P(e9|pad4)": e9 / max(pad, 1),
            "P(frame|e9)": fr / max(e9, 1),
            "P(r3|frame)": r3 / max(fr, 1),
            "P(succ|r3)": succ / max(r3, 1),
            "absolute": {"n": n, "fc": fc, "opp": opp, "pad4": pad, "e9": e9, "frame": fr, "r3": r3, "succ": succ},
        }

    # MSP loss attribution by earliest decisive (equal class weight approx via pooled counts / 4)
    decisive_all = Counter(r["earliest_decisive_failure"] for r in rows if r["earliest_decisive_failure"] != "success")
    n_fail = sum(decisive_all.values()) or 1

    # Avoid double-counting: earliest decisive is exclusive
    report = {
        "n_episodes": len(rows),
        "msp_equal_class_weight": msp,
        "success_by_class": succ_by_class,
        "per_class": per_class,
        "transitions_pooled": transitions(rows),
        "transitions_C": transitions(by_skill.get("C", [])),
        "transitions_nonC": transitions([r for r in rows if r["skill"] != "C"]),
        "earliest_decisive_pooled": dict(decisive_all),
        "earliest_decisive_share_of_failures": {k: v / n_fail for k, v in decisive_all.items()},
        "definitions": {
            "earliest_decisive": "First missing milestone on the success path (exclusive label)",
            "intermediate": "All ever-* milestones (may overlap)",
            "terminal": "SUCCESS_RELAXED / timeout / post-handoff fail",
            "no_double_count_in_budget": True,
        },
    }
    (OUT / "CROSS_CLASS_FAILURE_BUDGET.md").write_text(
        "# Cross-Class Failure Budget (P82 Phase 2)\n\n```json\n"
        + json.dumps(report, indent=2, default=str)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "CROSS_CLASS_FAILURE_BUDGET.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def skill_transition_analysis(rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    succ = [r for r in rows if r["SUCCESS_RELAXED"]]
    fail = [r for r in rows if not r["SUCCESS_RELAXED"]]

    def agg(rs, label):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "mean_family_switches": float(np.mean([r["n_family_switches"] for r in rs])),
            "mean_grip_open_near_contact": float(np.mean([r["grip_open_near_contact_steps"] for r in rs])),
            "mean_strong_approach_near_contact": float(np.mean([r["strong_approach_near_contact_steps"] for r in rs])),
            "mean_contact_loss": float(np.mean([r["contact_loss_ever"] for r in rs])),
            "mean_recatch_unstable": float(np.mean([r["recatch_unstable"] for r in rs])),
            "post_fc_family_mass": dict(
                sum((Counter(r["post_fc_family_counts"]) for r in rs), Counter())
            ),
            "label": label,
        }

    # Association flags (not causal)
    associations = {
        "premature_grip_opening": {
            "association": bool(
                fail and succ
                and np.mean([r["grip_open_near_contact_steps"] for r in fail])
                > np.mean([r["grip_open_near_contact_steps"] for r in succ]) + 0.5
            ),
            "evidence_level": "C_observational",
            "note": "P81 showed tok_56 opens grip; forcing it hurt full episodes — do not revive fixed token.",
        },
        "excessive_approach_motion": {
            "association": bool(
                fail and succ
                and np.mean([r["strong_approach_near_contact_steps"] for r in fail])
                > np.mean([r["strong_approach_near_contact_steps"] for r in succ]) + 0.5
            ),
            "evidence_level": "C_observational",
        },
        "contact_loss_during_correction": {
            "association": bool(fail and np.mean([r["contact_loss_ever"] for r in fail]) >= 0.5),
            "evidence_level": "C_observational",
            "supported_by_p72_funnel": True,
        },
        "inappropriate_skill_transitions": {
            "association": bool(
                fail and succ
                and np.mean([r["n_family_switches"] for r in fail])
                > np.mean([r["n_family_switches"] for r in succ]) + 0.5
            ),
            "evidence_level": "C_observational",
            "note": "P72/P73: more switches on fails, but fixed skill-sequence overlays failed transfer. Do not assume skill switching is the dominant C mechanism.",
        },
        "recatch_instability": {
            "association": bool(fail and np.mean([r["recatch_unstable"] for r in fail]) > 0.05),
            "evidence_level": "C_observational",
        },
        "handoff_instability": {
            "association": bool(
                sum(1 for r in fail if r["earliest_decisive_failure"] in ("post_handoff_or_terminal_fail", "no_r3_ready"))
                >= max(1, int(0.1 * len(fail)))
            ),
            "evidence_level": "C_observational",
        },
    }

    report = {
        "success": agg(succ, "success"),
        "failure": agg(fail, "failure"),
        "by_class_switches": {
            sk: {
                "succ_mean_switches": (
                    float(np.mean(xs))
                    if (xs := [r["n_family_switches"] for r in rows if r["skill"] == sk and r["SUCCESS_RELAXED"]])
                    else None
                ),
                "fail_mean_switches": (
                    float(np.mean(xs))
                    if (xs := [r["n_family_switches"] for r in rows if r["skill"] == sk and not r["SUCCESS_RELAXED"]])
                    else None
                ),
            }
            for sk in SKILLS
        },
        "associations": associations,
        "causality_warning": (
            "Skill-transition correlations are observational (evidence class C). "
            "Matched CF / full-episode interventions from P73/P81 apply only within their intervention scope."
        ),
    }
    (OUT / "SKILL_TRANSITION_ANALYSIS.md").write_text(
        "# Skill Transition Analysis (P82 Phase 3)\n\n```json\n"
        + json.dumps(report, indent=2, default=str)
        + "\n```\n",
        encoding="utf-8",
    )
    return report
