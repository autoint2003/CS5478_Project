"""Phases 4–6 — online signature, robustness, cross-class safety (skipped if gate fails)."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from experiments.precision.p84_nonC_mechanism_discovery.paths import OUT


def _skip_reports(reason: str) -> dict:
    skip = {
        "skipped": True,
        "reason": reason,
        "note": "Signature analysis only runs if Phase 3 concentration gate passes.",
    }
    for name, title in [
        ("ONLINE_SIGNATURE_ANALYSIS.md", "Online Signature Analysis (P84 Phase 4)"),
        ("SIGNATURE_ROBUSTNESS.md", "Signature Robustness (P84 Phase 5)"),
        ("CROSS_CLASS_SAFETY.md", "Cross-Class Safety (P84 Phase 6)"),
    ]:
        (OUT / name).write_text(f"# {title}\n\n```json\n" + json.dumps(skip, indent=2) + "\n```\n", encoding="utf-8")
    return skip


def analyze_signatures(rows: list[dict], gate: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if not gate.get("gate_passed"):
        return _skip_reports("Concentration gate failed — no dominant B1 mechanism for signature search.")

    # If gate somehow passed in future runs, build causal proxies from existing fields only
    top = gate["largest_mode"]["mode"]
    b1 = [r for r in rows if r["skill"] == "B1"]
    affected = [r for r in b1 if not r["SUCCESS_RELAXED"] and r["earliest_decisive_failure"] == top]
    succ = [r for r in b1 if r["SUCCESS_RELAXED"]]
    other_fail = [r for r in b1 if not r["SUCCESS_RELAXED"] and r["earliest_decisive_failure"] != top]

    # Causal proxies available mid-episode from logged fields (not future labels)
    # Use early signals: contact_loss before late milestones, family switches, grip_open, approach
    candidates = []

    def eval_rule(name, pred):
        # pred(r) -> bool detected; for fails in affected, need detection with lead time proxy
        cov = sum(1 for r in affected if pred(r))
        fp_succ = sum(1 for r in succ if pred(r))
        fp_other = sum(1 for r in other_fail if pred(r))
        # lead time: if t_loss or high switches early — rough
        leads = []
        for r in affected:
            if not pred(r):
                continue
            # heuristic lead: episode not yet at terminal length
            leads.append(max(0, (r.get("ep_len") or 0) - (r.get("t_loss") or r.get("t_opp") or r.get("t_fc") or 0)))
        return {
            "name": name,
            "coverage_among_affected": cov / max(len(affected), 1),
            "n_affected": len(affected),
            "n_detected_affected": cov,
            "false_alarms_success_b1": fp_succ,
            "n_success_b1": len(succ),
            "false_alarms_other_b1_fail": fp_other,
            "mean_lead_proxy_steps": float(np.mean(leads)) if leads else None,
            "uses_only_causal_fields": True,
            "uses_terminal_label": False,
            "uses_privileged_geometry": False,
        }

    candidates.append(
        eval_rule("contact_loss_ever_flag", lambda r: bool(r.get("contact_loss_ever")))
    )
    candidates.append(
        eval_rule("high_family_switches_ge4", lambda r: (r.get("n_family_switches") or 0) >= 4)
    )
    candidates.append(
        eval_rule("grip_open_near_contact_ge2", lambda r: (r.get("grip_open_near_contact_steps") or 0) >= 2)
    )

    report = {
        "target_mode": top,
        "candidates": candidates,
        "limitation": (
            "Logged fields are episode aggregates; true earliest detection time requires step logs. "
            "P84 does not collect new episodes; treat lead-time as proxy only."
        ),
        "actionable": False,  # filled below
    }
    # Require coverage>=0.5, fp_succ<=2/6, lead>0
    ok = [
        c
        for c in candidates
        if c["coverage_among_affected"] >= 0.5
        and c["false_alarms_success_b1"] <= 2
        and (c["mean_lead_proxy_steps"] or 0) > 0
    ]
    report["actionable"] = bool(ok)
    report["best"] = ok[0] if ok else None

    (OUT / "ONLINE_SIGNATURE_ANALYSIS.md").write_text(
        "# Online Signature Analysis (P84 Phase 4)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )

    # Robustness
    robust = {
        "skipped": False,
        "small_sample_warning": f"Only {len(succ)} successful B1 episodes — FP estimates imprecise",
        "threshold_perturbation": [],
        "oof_note": "No high-capacity classifier fitted; rules are fixed thresholds",
    }
    for thr in (3, 4, 5):
        c = eval_rule(f"switches_ge{thr}", lambda r, t=thr: (r.get("n_family_switches") or 0) >= t)
        robust["threshold_perturbation"].append(c)
    (OUT / "SIGNATURE_ROBUSTNESS.md").write_text(
        "# Signature Robustness (P84 Phase 5)\n\n```json\n" + json.dumps(robust, indent=2) + "\n```\n",
        encoding="utf-8",
    )

    # Cross-class
    cross = {"by_class": {}}
    for sk in ("A1", "C", "D"):
        rs = [r for r in rows if r["skill"] == sk]
        cross["by_class"][sk] = {
            "n": len(rs),
            "trigger_contact_loss": sum(r.get("contact_loss_ever") or 0 for r in rs) / max(len(rs), 1),
            "trigger_switches_ge4": sum(1 for r in rs if (r.get("n_family_switches") or 0) >= 4) / max(len(rs), 1),
            "trigger_on_successes_contact_loss": (
                sum(1 for r in rs if r["SUCCESS_RELAXED"] and r.get("contact_loss_ever"))
                / max(sum(1 for r in rs if r["SUCCESS_RELAXED"]), 1)
            ),
        }
    cross["note"] = "High trigger rates on other classes ⇒ cross-class risk if used as controller signal"
    (OUT / "CROSS_CLASS_SAFETY.md").write_text(
        "# Cross-Class Safety (P84 Phase 6)\n\n```json\n" + json.dumps(cross, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return {"signature": report, "robustness": robust, "cross": cross}
