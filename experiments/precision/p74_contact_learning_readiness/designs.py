"""Phase 5–7: reward objective, PPO likelihood design, matched pilot plan (no training)."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p74_contact_learning_readiness.paths import OUT


def _case(ownership, observability, learnable, critic) -> tuple[str, str]:
    actor_frac = ((ownership.get("fc_to_opposing_fail_only") or {}).get("actor_frac")) or 0.0
    lim = observability.get("limitation")
    best = observability.get("best_single_frame_auroc") or (None, None)
    # Treat moderate-only cues + privileged sidewall as insufficient observability (Case C risk)
    weak_obs = lim is not None or best[1] is None or abs((best[1] or 0.5) - 0.5) < 0.08
    learn = bool(learnable.get("evidence_of_learnable_selection"))
    crit_ok = bool(critic.get("critic_useful_for_contact_ppo"))

    if actor_frac < 0.5:
        return "B", (
            "V2 overrides most decisions in the failed FC→opposing window; "
            "ordinary Actor-only PPO cannot change those actions. Need a controller with valid likelihoods."
        )
    # Critic failure with Actor control + learnability → Case D takes priority over mild C,
    # but strong observability limitation without Critic recovery path stays C.
    if weak_obs and not crit_ok and learn:
        # Both C and D apply: report D primary (Critic blocks PPO) with C caveat in reason
        return "D", (
            "Actor controls the FC window and CF shows M111 choices can change outcomes, but "
            "(1) Critic EV in the FC window is unusable for advantages and "
            "(2) deployable obs only moderately cue contact outcomes while correct sidewall is "
            "privileged-only. Calibrate Critic first; do not train Actor objective blindly "
            "until observability of contact geometry is clearer."
        )
    if weak_obs and not learn:
        return "C", (
            "Deployable observations weakly distinguish contact-formation outcomes and there is little "
            "evidence of learnable state-dependent M111 selection under current labels."
        )
    if weak_obs and learn:
        return "C", (
            "Counterfactuals show M111 choices can change outcomes from matched FC states, but existing "
            "observations weakly discriminate viable vs nonviable contact; do not train the Actor blindly. "
            "Recover or augment contact geometry signals in the history first."
        )
    if learn and not crit_ok:
        return "D", (
            "Contact actions appear learnable and Actor-controlled, but the Critic is weak in the FC window. "
            "Calibrate Critic / advantages before changing the Actor objective."
        )
    if learn and crit_ok and actor_frac >= 0.5 and not weak_obs:
        return "A", (
            "Actor controls the failed contact window, observations carry usable signal, and Critic EV is "
            "adequate — a small matched PPO pilot with a bounded contact-formation objective is justified."
        )
    if not learn:
        return "E", "No clear evidence of learnable contact-formation improvement under V2+Actor; retain V2."
    return "A", (
        "Sufficient Actor control with some learnability evidence — proceed only with a tightly scoped matched pilot."
    )


def write_designs(*, ownership, observability, learnable, critic) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    case, reason = _case(ownership, observability, learnable, critic)

    # --- Reward design (equations) ---
    reward_md = """# Reward Objective Design (P74 Phase 5)

No training executed. Design only.

## Constraints

- Terminal `SUCCESS_RELAXED` / R3F / M111 / physics unchanged.
- Actor & Critic inputs remain deployable observations only.
- Privileged geometry (`correct_sidewall`, `opposing`, `min_pad`) may label **training rewards only**.

## Candidate: Bounded Sustained-Opposing Progress Bonus

Let indicators at step \\(t\\) (privileged, train-time only):

- \\(c_t\\): pad contact present
- \\(o_t\\): opposing bilateral (`bilateral ∧ correct_sidewall`)
- \\(p_t\\): `min_pad ≥ 4`
- \\(f_t\\): `frame_ok` (same R3 frame predicate; not a new success)

Event counters (episode-local):

- \\(E^{\\text{opp}}_t = \\mathbf{1}[o_t ∧ ¬o_{t-1}]\\) first opposing
- \\(S^{\\text{opp}}_t = \\mathbf{1}[o_t ∧ o_{t-1} ∧ o_{t-2}]\\) sustained ≥3 steps (anti-flicker)
- \\(L_t = \\mathbf{1}[¬c_t ∧ c_{t-1} ∧ ¬o_{\\le t}]\\) contact loss before ever opposing

**Auxiliary reward** (added to existing step reward \\(r^{\\text{base}}_t\\)):

\\[
r^{\\text{aux}}_t =
\\alpha\\, E^{\\text{opp}}_t
+ \\beta\\, S^{\\text{opp}}_t
+ \\gamma\\, \\mathbf{1}[p_t ∧ o_t]
- \\delta\\, L_t
\\]

Suggested bounds (small vs terminal):

| Coeff | Value | Cap per episode |
|-------|------:|----------------:|
| \\(\\alpha\\) | +0.15 | once |
| \\(\\beta\\) | +0.02 | ≤ 10 steps → ≤0.20 |
| \\(\\gamma\\) | +0.01 | ≤ 10 → ≤0.10 |
| \\(\\delta\\) | +0.10 | once |

Total aux ≤ ~0.45; terminal success remains ≫ aux (existing terminal scale).

**Not rewarded:** transient bilateral without correct sidewall; indefinite contact avoidance; pad count without opposing; raw e_pos shaping.

## Expected return effect

- Raises Monte Carlo return for trajectories that form **sustained correct** opposing contact.
- Penalizes the dominant P73 failure mode (contact loss before opposite engagement).
- Does not change R3 dwell or SUCCESS definition.

## Why different from earlier C reward shaping

Prior C shaping often densified pose/contact continuously (easy to farm / misalign with R3).
This objective is **sparse, event-gated, sustained-correct**, and capped — aimed only at the FC→opposing bottleneck under frozen V2 pre-contact.
"""
    (OUT / "REWARD_OBJECTIVE_DESIGN.md").write_text(reward_md, encoding="utf-8")

    # --- PPO likelihood ---
    ppo_md = f"""# PPO Likelihood Audit (P74 Phase 6)

## Hybrid policy

Executed token \\(a_t\\) is either:

1. **Actor-controlled:** \\(a_t ∼ \\pi_\\theta(\\cdot|h_t)\\) with stored `behavior_logp` = \\(\\log \\pi_\\theta(a_t|h_t)\\).
2. **V2-controlled:** deterministic override; **do not** use \\(\\log \\pi_\\theta(a_t|h_t)\\) as the behavior probability.

## Required bookkeeping (per step)

| Field | Actor step | V2 override |
|-------|------------|-------------|
| `v2_override` | 0 | 1 |
| `behavior_logp_valid` | 1 | 0 |
| `behavior_logp` | \\(\\log\\pi(a_t\\|h_t)\\) | `null` |
| PPO ratio / policy loss | yes | **exclude** |
| Value / return target | yes | yes (hybrid return) |

## Branch design (matched pilot — not run in P74)

### Branch A — Control
Original PPO reward; same Actor/Critic init; V2 integration unchanged.

### Branch B — Contact-Formation Learning
Identical to A except add the auxiliary contact reward from `REWARD_OBJECTIVE_DESIGN.md`.

Shared: starting Actor/Critic, ICs, sim-step budget, PPO hparams, V2, valid logp masking.

## If Case B (insufficient Actor control)

Do **not** run ordinary Actor PPO expecting FC-window improvement.
Options: learn a post-contact policy with its own likelihoods; or distill V2+post-contact expert with correct action probs — separately designed, not a silent V2 bypass.

## Current audit case: **{case}**

{reason}
"""
    (OUT / "PPO_LIKELIHOOD_AUDIT.md").write_text(ppo_md, encoding="utf-8")

    # --- Pilot plan ---
    proceed = case in ("A", "D")
    pilot = {
        "case": case,
        "proceed_with_training_in_P75": proceed and case == "A",
        "proceed_critic_first": case == "D",
        "stop": case in ("B", "C", "E"),
        "change_learning_mechanism": case == "B",
        "budget": {
            "episodes_per_branch": 2048,
            "branches": ["A_control", "B_contact_aux"],
            "matched_sim_steps": True,
            "IC_pool": "original P64/P69 train C pool",
            "starting_ckpt": "P64_step4 actor+critic",
            "V2": "frozen P70 V2 pre-contact",
        },
        "primary_metrics": [
            "MSP_RELAXED",
            "C SUCCESS_RELAXED",
            "C R3-ready",
            "C opposing-bilateral formation",
        ],
        "secondary_metrics": [
            "correct sidewall",
            "contact loss before opposing",
            "pad4",
            "R3 frame_ok",
            "Actor-controlled action fraction in FC window",
            "KL",
            "Critic EV in fc_to_opposing",
        ],
        "panels": "P61/P65 frozen compare only; final-test untouched",
        "stop_conditions": [
            "If Actor FC-window control falls below 0.5 under the hybrid logger → abort Actor PPO framing",
            "If B regresses A/B/D retention or C R3 vs A beyond noise → do not promote",
        ],
    }
    (OUT / "P75_MATCHED_PILOT_PLAN.json").write_text(json.dumps(pilot, indent=2), encoding="utf-8")
    (OUT / "P75_MATCHED_PILOT_PLAN.md").write_text(
        f"# P75 Matched Pilot Plan (P74 Phase 7)\n\n**Case {case}** — training in P74: **no**\n\n{reason}\n\n"
        f"```json\n{json.dumps(pilot, indent=2)}\n```\n\n"
        + (
            "## Recommended P75 action\nRun Branch A vs B as specified; gate on C opposing + R3 + retention.\n"
            if pilot["proceed_with_training_in_P75"]
            else (
                "## Recommended P75 action\nCritic calibration pilot only (no Actor objective change yet).\n"
                if pilot["proceed_critic_first"]
                else "## Recommended P75 action\n**Do not train** ordinary Actor PPO for contact formation under current evidence.\n"
            )
        ),
        encoding="utf-8",
    )

    decision = {"case": case, "reason": reason, "pilot": pilot}
    (OUT / "P74_CASE.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    own = json.loads((OUT / "ACTOR_ACTION_OWNERSHIP.json").read_text(encoding="utf-8"))
    obs = json.loads((OUT / "CONTACT_OBSERVABILITY.json").read_text(encoding="utf-8"))
    learn = json.loads((OUT / "LEARNABLE_DECISIONS.json").read_text(encoding="utf-8"))
    crit = json.loads((OUT / "CONTACT_STAGE_CRITIC.json").read_text(encoding="utf-8"))
    print(json.dumps(write_designs(ownership=own, observability=obs, learnable=learn, critic=crit), indent=2))
