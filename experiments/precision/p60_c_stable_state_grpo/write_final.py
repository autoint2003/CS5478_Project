"""Compose FINAL_REPORT.md from P60 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p60_c_stable_state_grpo.paths import OUT


def write_final() -> Path:
    def load(name: str) -> dict:
        p = OUT / name
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    sel = load("CHECKPOINT_SELECTION.json")
    man = load("STATE_SELECTION_MANIFEST.json")
    rest = load("STATE_RESTORATION_TEST.json")
    probe = load("FEASIBILITY_PROBE.json")
    train = load("P60_TRAIN_SUMMARY.json")
    held = load("HELDOUT_EVAL.json")

    # Prefer fullest validation
    val = {}
    for name in ("VALIDATION_after_pass2.json", "VALIDATION.json", "VALIDATION_after_pass1.json"):
        if (OUT / name).is_file():
            val = load(name)
            break

    rc = val.get("reports_c") or {}
    ra = val.get("reports_abd") or {}
    best = held.get("best_ckpt") or "P60_pass1"
    hr = (held.get("reports") or {}).get(best) or {}
    h57 = (held.get("reports") or {}).get("P57_A") or {}
    h59 = (held.get("reports") or {}).get("P59_pass1") or {}

    def msp_c(name: str):
        return (rc.get(name) or {}).get("MSP_RELAXED")

    def abd(name: str):
        return (rc_abd := (ra.get(name) or {})).get("MSP_RELAXED")

    c60_1, c60_2 = msp_c("P60_pass1"), msp_c("P60_pass2")
    c59_1 = msp_c("P59_pass1")
    improved_c = (
        best in rc
        and c59_1 is not None
        and (rc.get(best) or {}).get("MSP_RELAXED") is not None
        and float((rc[best])["MSP_RELAXED"]) > float(c59_1) + 1e-9
    )
    held_msp = hr.get("MSP_RELAXED_12S")
    held_c = ((hr.get("by_class") or {}).get("C") or {}).get("MSP_RELAXED")
    held59 = h59.get("MSP_RELAXED_12S")
    held_ok = held_msp is not None and held59 is not None and float(held_msp) + 1e-9 >= float(held59) - 0.01

    # Stage: SUCCESS|stable from C_STAGE
    st60 = ((rc.get(best) or {}).get("stage_transitions") or {})
    st59 = ((rc.get("P59_pass1") or {}).get("stage_transitions") or {})

    if not probe.get("viable_for_training", True) and not train:
        verdict = "Not viable — feasibility probe stopped training."
    elif improved_c and held_ok:
        verdict = "Promising — full-IC C validation improved; held-out overall preserved."
    elif improved_c:
        verdict = "Mixed — C validation improved but held-out/ABD caution."
    elif st60.get("SUCCESS_given_stable") and st59.get("SUCCESS_given_stable") and float(
        st60["SUCCESS_given_stable"]
    ) > float(st59["SUCCESS_given_stable"]) + 1e-9:
        verdict = (
            "Continuation conversion improved on val stages, but full-episode C MSP did not clearly rise."
        )
    else:
        verdict = "Not promising for promotion — stable-state continuation did not raise full-episode C success."

    lines = [
        "# FINAL_REPORT — P60 C Stable-State Continuation GRPO-64",
        "",
        f"## Verdict",
        "",
        f"**{verdict}**",
        "",
        f"Selected checkpoint: **`{best}`** (validation C MSP; held-out not used for selection).",
        "",
        "| Model | C val MSP | ABD val | Held-out MSP | Held-out C |",
        "|-------|----------:|--------:|-------------:|-----------:|",
    ]
    for name in ("P59_pass1", "P59_pass2", "P60_pass1", "P60_pass2"):
        if name not in rc and name != best:
            continue
        r = rc.get(name) or {}
        a = ra.get(name) or {}
        if name == best:
            hm = held_msp
            hc = held_c
        elif name == "P59_pass1":
            hm = h59.get("MSP_RELAXED_12S")
            hc = ((h59.get("by_class") or {}).get("C") or {}).get("MSP_RELAXED")
        else:
            hm = hc = None
        lines.append(
            f"| {name} | {r.get('MSP_RELAXED', float('nan')) if r else float('nan'):.4f} | "
            f"{a.get('MSP_RELAXED', float('nan')) if a else float('nan'):.4f} | "
            f"{(hm if hm is not None else float('nan')):.4f} | "
            f"{(hc if hc is not None else float('nan')):.4f} |"
        )
    # P57 row from heldout
    if h57:
        lines.append(
            f"| P57_A (held-out only) | — | — | {h57.get('MSP_RELAXED_12S', float('nan')):.4f} | "
            f"{((h57.get('by_class') or {}).get('C') or {}).get('MSP_RELAXED', float('nan')):.4f} |"
        )

    lines += [
        "",
        "## Setup",
        "",
        f"- Init: `{sel.get('selected')}` — " + "; ".join(sel.get("rationale") or [])[:200],
        f"- States: {man.get('n_states')} stable-but-failed C intermediates "
        f"(patterns={man.get('pattern_counts')})",
        f"- Restoration test pass: **{rest.get('all_ok')}**",
        f"- Feasibility viable: **{probe.get('viable_for_training')}** "
        f"(mean cont. success={probe.get('mean_p_success')})",
        f"- Train: {train.get('n_states')} × 64 × {train.get('n_passes')} = "
        f"**{train.get('n_continuations')}** continuations; wall {float(train.get('wall_s') or 0)/3600:.2f} h",
        "- Unchanged: M111, P57-A reward, R3F HANDOFF, SUCCESS_RELAXED_12S, frozen KL; no expert BC / shaping",
        "",
        "## Stage transitions (C val)",
        "",
        f"- P59_pass1 SUCCESS|stable={st59.get('SUCCESS_given_stable')}, "
        f"p_stable={st59.get('p_stable')}, p_success={st59.get('p_success')}",
        f"- {best} SUCCESS|stable={st60.get('SUCCESS_given_stable')}, "
        f"p_stable={st60.get('p_stable')}, p_success={st60.get('p_success')}",
        "",
        "## Success criteria",
        "",
        f"1. Full-episode C val > P59_pass1: "
        f"**{'YES' if improved_c else 'NO'}**",
        "2. Gains reflect continuation (SUCCESS|stable), not only acquisition: see C_STAGE_TRANSITIONS.md",
        "3. ABD without substantial collapse: see VALIDATION_REPORT.md",
        f"4. Held-out overall preserved/improved vs P59: "
        f"**{'YES' if held_ok else 'NO/UNCERTAIN'}**",
        "",
        "Primary objective is full-episode physical recovery probability, "
        "not intermediate-state continuation success alone.",
        "",
    ]
    path = OUT / "FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_final())
