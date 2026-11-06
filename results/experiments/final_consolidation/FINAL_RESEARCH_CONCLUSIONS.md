# Final Research Conclusions

## Frozen system

The reproducible baseline is:

- **P64 Actor** (`P64_step4_actor.pt`, MD5 `9f4c5041…b6e2`)
- **P64 Critic** (`P64_step4_critic.pt`, MD5 `95de94e5…81cb`)
- **P70 V2** adaptive pre-contact controller (proprioceptive ready_score; approach pool `{56,50,71}`)
- **Original M111** discrete skill vocabulary (book size 111)
- **Original R3F** predicates and **SUCCESS_RELAXED_12S**

Canonical inference: `p82_global_failure_budget.collect.rollout_episode` with V2 and **`rule=None`** (no P73/P81 overlays).

## Why this system was selected

1. **Positive no-train closed-loop evidence for V2** on the P70 held panel: SUCCESS_RELAXED / R3-ready ≈ **0.240** vs Actor-only ≈ **0.083** (Δ ≈ +0.156), without improving mean first-contact `e_pos` — hence PPO was correctly **not** launched from P70’s absolute-alignment gate, but V2 was retained as a deployable controller.
2. **Subsequent Critic, sensing, observation, and action-selection studies (P75–P80)** did not justify Actor/Critic retraining or observation expansion under pre-registered gates.
3. **Closed-loop interventions that looked good in matched CF failed in full episodes** (P73 overlay ΔSUCCESS ≈ −0.094 vs V2; P81 tok_56 MSP Δ = −0.25), so they were rejected rather than presented as optimizations.
4. **P82–P84** showed substantial non-C failure mass, but B1 exclusive modes are **diffuse** (largest 9/26 < 11). Concentration gate failed → **stop** rather than invent a mechanism-specific P85.

Selection criterion: best **documented, hash-frozen, deployable** stack after a closed diagnostic campaign — **not** “globally optimal policy.”

## Canonical reported performance

On the validated **P83** panel (32 independent ICs/class; blocked final-test excluded):

| A1 | B1 | C | D | Equal-weight MSP |
|---:|---:|---:|---:|---:|
| 8/32 | 6/32 | 4/32 | 17/32 | **35/128 = 0.2734375** |

Bootstrap MSP 95% CI ≈ **[0.1953, 0.3516]**.  
**Not** a pristine final-test result. Untouched final-test ICs remain unevaluated.

## What did not work (do not reframe as success)

- Critic recalibration / weighted critics: FC EV uninformative or worse (P75/P76 Case C).
- Obs/sensing expansion for training: deferred / insufficient action-relevance (P77 Case A study-only; P78 Case C).
- History-conditioned selection over fixed tok_56: no held-out gain (P80 Case B).
- Fixed-token and post-contact overlays: matched benefits, full-episode harm (P73/P81).
- Further non-C mechanism targeting: concentration gate fail (P84 Case D).

## Failure structure (high level)

- Class **C** remains hard at contact formation (opposing after first contact is a major exclusive sink).
- **Non-C** aggregate failure mass exceeds C contact-formation mass on P83, but recoverability is unproven and B1 modes are not concentrated enough for a safe next intervention.
- Matched counterfactuals are necessary but **not sufficient** for closed-loop claims.

## Campaign status

| Item | Status |
|---|---|
| Optimization after P84 | **Closed** |
| P85 | **Not launched** |
| Actor/Critic training | **Not run** in this consolidation |
| Final-test evaluation | **Not run** |
| Deliverable intent | Preserve and communicate completed research |

## Bottom line

Adaptive grasp recovery with discrete multi-skill actions and a lightweight pre-contact controller yields a **reproducible but still low-success** system (MSP ≈ 0.27 on the diagnostic panel), with clear negative evidence against several tempting closed-loop “fixes.” The scientific contribution is the frozen stack, the verified evaluation table, and the evidence taxonomy that separates observation, matched CF, and full-episode causality — not a claim that M111 or V2 saturates the problem.
