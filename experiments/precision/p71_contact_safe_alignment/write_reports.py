"""Assemble P71 deliverables + P72 decision."""

from __future__ import annotations

import json

from experiments.precision.p71_contact_safe_alignment.paths import OUT


def _load(name: str):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_p72(p1, p2, matched, panel) -> dict:
    case = (matched or {}).get("case") or "unknown"
    frozen = (matched or {}).get("frozen") or {}
    best = frozen.get("best_name") or "V2"
    held = (matched or {}).get("held_agg") or {}
    deltas = (matched or {}).get("deltas_vs_V0_held") or {}
    bd = deltas.get(best) or {}
    v2d = deltas.get("V2") or {}

    p61 = (panel or {}).get("P61_dev_C") or {}
    gen_ok = False
    if p61:
        bkey = best if best in p61 else ("V2" if "V2" in p61 else None)
        if bkey and "V0" in p61:
            gen_ok = (p61[bkey]["agg"].get("r3_ready") or 0) > (p61["V0"]["agg"].get("r3_ready") or 0) + 0.02

    # Training gate: only if Case B with R3+SUCCESS on held AND panel generalization, not geometry-only
    r3_up = (bd.get("r3_ready") or 0) > 0.02
    succ_up = (bd.get("SUCCESS_RELAXED") or 0) > 0.02
    geom_only = ((bd.get("frac_fc_le9") or 0) > 0.05 or (bd.get("e_first_contact") or 0) < -5) and not r3_up
    recommend_train = bool(
        case in ("B",) and r3_up and succ_up and gen_ok and not geom_only and best != "V0"
    )
    # Case A / A_with_D: post-contact path works; do not train yet — refine no-train first.
    if str(case).startswith("A"):
        recommend_train = False

    if geom_only:
        reason = "Geometry-only gains — do not train"
    elif str(case).startswith("A"):
        reason = (
            "Case A: V2 succeeds via post-contact correction; first-contact ≤9 mm gate is too strict. "
            "Naive V3–V5 alignment extensions fail (D) — keep V2; refine post-contact reliability before any PPO."
            if "D" in str(case)
            else "Case A: V2 succeeds via post-contact correction; refine reliability of that path before PPO"
        )
    elif recommend_train:
        reason = "Case B with held R3/SUCCESS + panel generalization — small anchored pilot could be considered (not auto-launched)"
    elif case == "C":
        reason = "Case C: geometry without success — do not train"
    elif case == "D":
        reason = "Case D: premature contact / failed extension — reject naive APPROACH extension"
    else:
        reason = "Insufficient generalization or no clear R3/SUCCESS gain over V2 — do not train"

    decision = {
        "case": case,
        "best_variant": best,
        "recommend_train": recommend_train,
        "reason": reason,
        "held_delta_best": bd,
        "held_delta_V2": v2d,
        "panel_generalizes": gen_ok,
        "next": (
            "No training. If Case A: stabilize post-contact correction cues. "
            "If Case B without panel: re-check proxies. If Case C: pad/orientation bottleneck. "
            "If Case D: reorder non-56 alignment tokens."
        ),
    }
    (OUT / "P72_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    md = [
        "# P72 Decision (from P71)\n",
        f"**Case**: `{case}`  **best**: `{best}`  **recommend_train**: `{recommend_train}`\n",
        f"{decision['reason']}\n",
        f"## Next\n{decision['next']}\n",
        "```json\n" + json.dumps(decision, indent=2) + "\n```\n",
        "No training launched.\n",
    ]
    (OUT / "P72_DECISION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return decision


def write_final(p1, p2, matched, panel, decision) -> str:
    held = (matched or {}).get("held_agg") or {}
    answers = {
        "1_r3_with_fc_e_gt9": {
            "frac_succ_fc_gt9": (p1 or {}).get("frac_succ_fc_gt9"),
            "frac_then_correct_le9": (p1 or {}).get("frac_succ_fc_gt9_then_correct_le9"),
            "case_A": (p1 or {}).get("case_A_post_contact_alignment"),
        },
        "2_how_V2_corrects_after_contact": (
            "See V2_SUCCESS_GEOMETRY stages: e_fc → e_bil → min_e_pre_r3 → e_r3 among successes"
        ),
        "3_token56_premature_contact": {
            "frac_first_contact": (p2 or {}).get("frac_first_contact"),
            "dominant": (p2 or {}).get("dominant"),
            "decode_grip": ((p2 or {}).get("decode") or {}).get("grip"),
        },
        "4_safe_align_reduces_fc_error": ((matched or {}).get("deltas_vs_V0_held") or {}),
        "5_translates_to_r3": {k: (held.get(k) or {}).get("r3_ready") for k in held},
        "6_terminal_success_train_C": {k: (held.get(k) or {}).get("SUCCESS_RELAXED") for k in held},
        "7_best_generalizing": {
            "frozen": (matched or {}).get("frozen", {}).get("best_name"),
            "panel": (panel or {}).get("P61_dev_C"),
        },
        "8_training_justified": decision.get("recommend_train"),
        "case": decision.get("case"),
        "decision": decision,
    }
    md = [
        "# P71 Final Report — Contact-Safe Pre-Contact Alignment\n",
        f"**Case {decision.get('case')}** — train=`{decision.get('recommend_train')}`: {decision.get('reason')}\n",
        "## Answers\n",
    ]
    for k, v in answers.items():
        md.append(f"### {k}\n```\n{json.dumps(v, indent=2, default=str)}\n```\n")
    md.append("No PPO/GRPO/BC/DAgger. Final-test untouched.\n")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "FINAL_ANSWERS.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
    return str(decision.get("case"))


def write_all() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = _load("V2_SUCCESS_GEOMETRY.json")
    p2 = _load("TOKEN56_CONTACT_DYNAMICS.json")
    matched = _load("MATCHED_COUNTERFACTUALS.json")
    panel = _load("PANEL_VALIDATION.json")
    decision = write_p72(p1, p2, matched, panel)
    return write_final(p1, p2, matched, panel, decision)


if __name__ == "__main__":
    print(write_all())
