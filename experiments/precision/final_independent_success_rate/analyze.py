"""Write SR tables, V2 audit, failure summary, final report."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.precision.final_independent_success_rate.paths import OUT, SKILLS, TARGET_PER_CLASS
from experiments.precision.final_independent_success_rate.stats import class_sr, equal_weight_overall


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.2f}%"


def write_reports(rows: list[dict], manifest: dict) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)

    # Drop infra errors from primary SR (log them); user said infra may rerun — we keep them listed
    valid = [r for r in rows if not r.get("infra_error")]
    infra = [r for r in rows if r.get("infra_error")]

    by_class_bits: dict[str, list[int]] = {sk: [] for sk in SKILLS}
    for r in valid:
        by_class_bits[r["skill"]].append(int(r["SUCCESS_RELAXED_12S"]))

    per_class = {sk: class_sr(by_class_bits[sk]) for sk in SKILLS}
    overall = equal_weight_overall(by_class_bits, seed=5478)

    per_path = OUT / "PER_CLASS_SUCCESS_RATE.json"
    per_path.write_text(
        json.dumps(
            {
                "metric": "Episode Success Rate = SUCCESS_RELAXED_12S successes / independent episodes",
                "not_msp64": True,
                "evaluation": "small-scale independent evaluation",
                "target_per_class": TARGET_PER_CLASS,
                "by_class": per_class,
                "shortfalls": manifest.get("shortfalls"),
                "n_infra_errors_excluded_from_sr": len(infra),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    overall_path = OUT / "OVERALL_SUCCESS_RATE.json"
    overall_path.write_text(
        json.dumps(
            {
                "metric": "Equal-weight mean of class Episode Success Rates",
                "not_msp64": True,
                "evaluation": "small-scale independent evaluation",
                **overall,
                "per_class_n": {sk: per_class[sk]["n_independent_episodes"] for sk in SKILLS},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # V2 audit
    n = len(valid)
    n_v2 = sum(int(r.get("v2_activated") or 0) for r in valid)
    mean_forced = float(sum(int(r.get("forced_steps") or 0) for r in valid) / n) if n else 0.0
    actor_dom = sum(int(r.get("predominantly_actor_controlled") or 0) for r in valid)
    v2_md = [
        "# V2 Activation Audit",
        "",
        "Small-scale independent evaluation of frozen P64 + P70 V2.",
        "",
        f"- Valid episodes: **{n}**",
        f"- Episodes with ≥1 V2 forced step: **{n_v2}** ({_pct(n_v2 / n if n else None)})",
        f"- Mean forced steps / episode: **{mean_forced:.3f}**",
        f"- Predominantly Actor-controlled: **{actor_dom}/{n}**",
        "",
        "## Interpretation",
        "",
    ]
    if n and (n_v2 / n) < 0.05:
        v2_md.append(
            "V2 activation is rare. The evaluated system is **mostly Actor-controlled** on this set. "
            "Do **not** attribute success rates to V2 without a paired Actor-only comparison."
        )
    else:
        v2_md.append(
            "V2 activated on a non-negligible fraction of episodes; still report Actor/V2 mix explicitly."
        )
    v2_md.append("")
    by_sk_v2 = {
        sk: {
            "n": sum(1 for r in valid if r["skill"] == sk),
            "v2_on": sum(1 for r in valid if r["skill"] == sk and r.get("v2_activated")),
        }
        for sk in SKILLS
    }
    v2_md.append("```json")
    v2_md.append(json.dumps(by_sk_v2, indent=2))
    v2_md.append("```")
    (OUT / "V2_ACTIVATION_AUDIT.md").write_text("\n".join(v2_md) + "\n", encoding="utf-8")

    # Failure summary
    fail_rows = [r for r in valid if not r["SUCCESS_RELAXED_12S"]]
    fail_by = defaultdict(Counter)
    for r in fail_rows:
        fail_by[r["skill"]][r.get("earliest_decisive_failure") or "unknown"] += 1
    # C funnel
    c_rows = [r for r in valid if r["skill"] == "C"]
    c_n = len(c_rows)
    funnel = {}
    if c_n:
        funnel = {
            "n": c_n,
            "P_first_contact": sum(r["first_contact"] for r in c_rows) / c_n,
            "P_opposing_ever": sum(r["opposing_ever"] for r in c_rows) / c_n,
            "P_min_pad_ge4": sum(r["min_pad_ge4"] for r in c_rows) / c_n,
            "P_e_pos_le9": sum(r["e_pos_le9"] for r in c_rows) / c_n,
            "P_frame_ok": sum(r["frame_ok"] for r in c_rows) / c_n,
            "P_r3_ready": sum(r["r3_ready"] for r in c_rows) / c_n,
            "P_SUCCESS_RELAXED": sum(r["SUCCESS_RELAXED_12S"] for r in c_rows) / c_n,
        }
    ep_lens = [int(r["ep_len"]) for r in valid]
    fail_md = [
        "# Failure Summary (secondary)",
        "",
        "## Exclusive earliest decisive failure by class",
        "",
        "```json",
        json.dumps({sk: dict(fail_by[sk]) for sk in SKILLS}, indent=2),
        "```",
        "",
        "## Secondary rates",
        "",
        f"- SUCCESS_NEW: {sum(r['SUCCESS_NEW'] for r in valid)}/{n} = {_pct(sum(r['SUCCESS_NEW'] for r in valid)/n if n else None)}",
        f"- R3-ready: {sum(r['r3_ready'] for r in valid)}/{n} = {_pct(sum(r['r3_ready'] for r in valid)/n if n else None)}",
        f"- Handoff: {sum(r['handoff'] for r in valid)}/{n}",
        "",
        "## Class C contact-formation funnel (observational)",
        "",
        "```json",
        json.dumps(funnel, indent=2),
        "```",
        "",
        "## Episode duration (steps)",
        "",
        f"- n={len(ep_lens)}; mean={float(sum(ep_lens)/len(ep_lens) if ep_lens else 0):.2f}; "
        f"min={min(ep_lens) if ep_lens else None}; max={max(ep_lens) if ep_lens else None}",
        "",
        f"Infra errors (excluded from SR): {len(infra)}",
        "",
    ]
    (OUT / "FAILURE_SUMMARY.md").write_text("\n".join(fail_md), encoding="utf-8")

    # Final report
    lines = [
        "# Final Evaluation Report — Small-Scale Independent Evaluation",
        "",
        "**Scope:** small-scale independent evaluation (target 10 ICs/class × 1 rollout = 40).  "
        "This is **not** a large-N final-test campaign and **not** MSP-64.",
        "",
        f"**Sample-size limitation:** target {TARGET_PER_CLASS}/class; "
        f"actual n = {json.dumps({sk: per_class[sk]['n_independent_episodes'] for sk in SKILLS})}.  "
        f"Shortfalls: `{json.dumps(manifest.get('shortfalls'))}`.",
        "",
        "## Primary metric: Episode Success Rate (SUCCESS_RELAXED_12S)",
        "",
        "| Class | Independent Episodes | Successes | Success Rate | 95% Wilson CI |",
        "|---|---:|---:|---:|---|",
    ]
    for sk in SKILLS:
        s = per_class[sk]
        ci = s["ci95_wilson"]
        ci_s = (
            f"[{100*ci['low']:.2f}%, {100*ci['high']:.2f}%]"
            if ci.get("low") is not None
            else "n/a"
        )
        lines.append(
            f"| {sk} | {s['n_independent_episodes']} | {s['n_success']} | "
            f"{_pct(s['success_rate'])} | {ci_s} |"
        )
    lines += [
        "",
        f"**Overall (equal-weight mean of class SRs):** {_pct(overall['equal_weight_success_rate'])}  ",
        f"Total successes / episodes: **{overall['n_success_total']} / {overall['n_independent_episodes_total']}**  ",
        f"95% stratified bootstrap CI: "
        + (
            f"[{100*overall['ci95_stratified_bootstrap'][0]:.2f}%, "
            f"{100*overall['ci95_stratified_bootstrap'][1]:.2f}%]"
            if overall.get("ci95_stratified_bootstrap")
            else "n/a"
        ),
        "",
        "## Frozen system",
        "",
        "```json",
        json.dumps(manifest.get("frozen_system"), indent=2),
        "```",
        "",
        "## V2 caveat",
        "",
        f"V2 activated on **{n_v2}/{n}** episodes. "
        + (
            "System is **mostly Actor-controlled** on this set; do not claim V2 gains without Actor-only paired eval."
            if n and n_v2 / n < 0.05
            else "See V2_ACTIVATION_AUDIT.md."
        ),
        "",
        "## Historical non-comparisons",
        "",
        "Do **not** treat as the same estimand as:",
        "",
        "- P64/P65 development MSP-style repeated-rollout scores",
        "- P70 V2 diagnostic panel (different IC distribution; V2 activation ~1.0 there)",
        "- P82/P83 diagnostic panels (train-overlapping; P83 MSP=0.2734375 is **not** independent final-test)",
        "- P79/P80 matched CF deltas",
        "",
        "## Artifacts",
        "",
        "- `EVAL_PROTOCOL.md`",
        "- `FROZEN_TEST_MANIFEST.json`",
        "- `DATA_LEAKAGE_AUDIT.md`",
        "- `EPISODE_RESULTS.csv`",
        "- `PER_CLASS_SUCCESS_RATE.json`",
        "- `OVERALL_SUCCESS_RATE.json`",
        "- `V2_ACTIVATION_AUDIT.md`",
        "- `FAILURE_SUMMARY.md`",
        "",
    ]
    (OUT / "FINAL_EVALUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    return {"per_class": per_class, "overall": overall, "n_valid": n, "n_v2": n_v2}
