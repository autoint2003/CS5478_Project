"""FINAL_REPORT + answers."""

from __future__ import annotations

import json

from experiments.precision.p74_contact_learning_readiness.paths import OUT


def _load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_final() -> str:
    own = _load("ACTOR_ACTION_OWNERSHIP.json")
    obs = _load("CONTACT_OBSERVABILITY.json")
    learn = _load("LEARNABLE_DECISIONS.json")
    crit = _load("CONTACT_STAGE_CRITIC.json")
    case = _load("P74_CASE.json")
    pilot = _load("P75_MATCHED_PILOT_PLAN.json")

    c = case.get("case") or "?"
    reason = case.get("reason") or ""

    answers = {
        "1_actor_controls_contact_failures": {
            "fc_fail_actor_frac": (own.get("fc_to_opposing_fail_only") or {}).get("actor_frac"),
            "per_window": own.get("per_window"),
            "verdict": (own.get("critical_question") or {}).get("actor_controls_failed_contact_window"),
        },
        "2_obs_distinguish_contact_states": {
            "auroc_fc_vs_opp": obs.get("auroc_obs_vs_eventual_opposing_in_fc_window"),
            "auroc_preloss": obs.get("auroc_obs_preloss_vs_success_contact"),
            "history_auroc": obs.get("auroc_short_history_mean3_vs_opposing"),
            "limitation": obs.get("limitation"),
        },
        "3_different_m111_change_outcomes": {
            "evidence": learn.get("evidence_of_learnable_selection"),
            "p73_cf": learn.get("p73_cf_evidence"),
            "entropy": learn.get("mean_entropy_fc"),
        },
        "4_critic_useful_fc_window": {
            "fc_stage": (crit.get("by_stage") or {}).get("fc_to_opposing"),
            "useful": crit.get("critic_useful_for_contact_ppo"),
            "mc_gap": crit.get("mc_return_gap_succ_minus_fail_fc_window"),
        },
        "5_bounded_contact_objective": {
            "designed": True,
            "file": "REWARD_OBJECTIVE_DESIGN.md",
            "note": "Sparse sustained-opposing bonus + pre-opp contact-loss penalty; terminal unchanged.",
        },
        "6_ppo_likelihoods_under_v2": {
            "file": "PPO_LIKELIHOOD_AUDIT.md",
            "rule": "PPO policy loss only when behavior_logp_valid; V2 overrides excluded from ratios.",
        },
        "7_smallest_justified_experiment": pilot,
        "8_p75_recommendation": {
            "case": c,
            "train": bool(pilot.get("proceed_with_training_in_P75")),
            "critic_first": bool(pilot.get("proceed_critic_first")),
            "stop": bool(pilot.get("stop")),
            "change_mechanism": bool(pilot.get("change_learning_mechanism")),
            "reason": reason,
        },
    }

    (OUT / "FINAL_ANSWERS.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    md = [
        "# P74 Final Report — Contact Learning Readiness\n",
        f"**Case {c}** — no training in P74 — V2 unchanged — no new sequence overlays\n",
        f"{reason}\n",
        "## Answers\n",
    ]
    for k, v in answers.items():
        md.append(f"### {k}\n```json\n{json.dumps(v, indent=2, default=str)}\n```\n")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return c


if __name__ == "__main__":
    print(write_final())
