"""Assemble P72 markdown + P73 decision."""

from __future__ import annotations

import json

from experiments.precision.p72_postcontact_reliability.paths import OUT


def _load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_p73(funnel, mech, val) -> dict:
    case = (mech or {}).get("case") or "E"
    promote = bool((val or {}).get("promote_p72_over_v2"))
    chosen = (mech or {}).get("chosen_intervention")
    held = (val or {}).get("held_agg") or {}
    d = ((val or {}).get("held_deltas") or {}).get("P72_minus_V2") or {}

    if promote:
        decision = "replace_v2_with_p72"
        reason = f"P72 `{chosen}` improves full-episode R3 vs V2 on held-out ICs and passes promotion gates."
    else:
        decision = "retain_v2"
        reason = (
            f"No deployable post-contact refinement beat V2 (case={case}, chosen={chosen}). "
            "Keep P70 V2 unchanged."
        )

    # answer helpers from funnel
    f_all = (funnel or {}).get("funnel_all") or {}
    ever = f_all.get("first_passage_ever") or {}
    cond = f_all.get("conditional") or {}
    g_s = (funnel or {}).get("geom_succ") or {}
    g_f = (funnel or {}).get("geom_fail") or {}

    answers = {
        "1_fail_after_fc": {
            "ever": ever,
            "conditional_dropoffs": cond,
            "note": "Largest ever→next drops indicate primary bottleneck stage",
        },
        "2_why_some_reach_e9": {"succ": g_s, "fail": g_f},
        "3_geometry_vs_pad": {
            "fail_frac_reach_e9": g_f.get("frac_reach_e9"),
            "fail_frac_pad4": g_f.get("frac_pad4"),
            "succ_frac_reach_e9": g_s.get("frac_reach_e9"),
            "succ_frac_pad4": g_s.get("frac_pad4"),
        },
        "4_interrupt_sequences": {
            "switch_gap": None
            if g_s.get("mean_post_switches") is None or g_f.get("mean_post_switches") is None
            else g_f["mean_post_switches"] - g_s["mean_post_switches"],
            "case": case,
        },
        "5_near_ready": {
            "near_fail_n": (funnel or {}).get("near_fail_n"),
            "violations": (funnel or {}).get("near_fail_first_violation"),
        },
        "6_bounded_obs_intervention_helps": {
            "promote": promote,
            "delta_r3_p72_v2": d.get("r3_ready"),
            "delta_success": d.get("SUCCESS_RELAXED"),
        },
        "7_generalizes": {
            "held": held,
            "p61": (val or {}).get("p61_agg"),
            "p65": (val or {}).get("p65_agg"),
        },
        "8_replace_or_retain": decision,
    }

    out = {
        "decision": decision,
        "recommend_train": False,
        "promote_p72": promote,
        "retain_v2": not promote,
        "case": case,
        "chosen": chosen,
        "reason": reason,
        "answers": answers,
    }
    (OUT / "P73_DECISION.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "P73_DECISION.md").write_text(
        f"# P73 Decision\n\n**{decision}** — train=False\n\n{reason}\n\n"
        f"```json\n{json.dumps({k: out[k] for k in out if k != 'answers'}, indent=2)}\n```\n",
        encoding="utf-8",
    )
    return out


def write_final(p73) -> str:
    md = [
        "# P72 Final Report — Post-Contact Correction Reliability\n",
        f"**Decision: {p73.get('decision')}** (no training)\n",
        f"{p73.get('reason')}\n",
        "## Answers\n",
    ]
    for k, v in (p73.get("answers") or {}).items():
        md.append(f"### {k}\n```\n{json.dumps(v, indent=2, default=str)}\n```\n")
    md.append("Pre-contact V2 unchanged. Final-test untouched.\n")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return str(p73.get("decision"))


def write_all() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    funnel = _load("POSTCONTACT_FUNNEL.json")
    mech = _load("DOMINANT_MECHANISM.json")
    val = _load("FULL_EPISODE_VALIDATION.json")
    p73 = write_p73(funnel, mech, val)
    return write_final(p73)


if __name__ == "__main__":
    print(write_all())
