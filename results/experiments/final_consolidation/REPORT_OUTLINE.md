# Report Outline — Adaptive Grasp Recovery (Frozen System)

Suggested sectioning for the course/technical report. Cite consolidation tables/figures; avoid dumping every experiment.

---

## 1. Problem definition

- Multi-skill adaptive grasp / recovery under sparse contact success.
- Four IC classes A1/B1/C/D; equal-weight MSP as primary aggregate.
- Terminal criteria: original R3F predicates + `SUCCESS_RELAXED_12S`.

## 2. Motivation for discrete multi-skill actions

- M111 skill-aware discrete vocabulary (APPROACH/GRIP/ORIENT/…).
- Enables interpretable interventions and matched action counterfactuals.
- **Do not claim global optimality** of M111 vs continuous control.

Figure: `fig02_m111_action_representation`.

## 3. Methodology

- Actor–Critic with discrete tokens; StabilizeGrpoEnv.
- P70 V2 adaptive pre-contact controller (proprioceptive ready_score).
- Diagnostic ladder: Critic calib → observability → matched CF → policy feasibility → static bias → failure budget → recoverability → concentration gate.
- Evidence hierarchy: observational / matched CF / full-episode intervention / scenario bounds.

Figure: `fig01_final_system_architecture`, `fig03_train_eval_pipeline`.

## 4. Experimental setup

- Checkpoints, hashes, V2 hyperparameters (Manifest).
- IC pool from P54; blocked eval / final-test excluded from diagnostic panels.
- Canonical rollout: `p82.collect.rollout_episode` with `rule=None`.
- Seeds: P83 `SEED=5483`, `N_SEEDS=1`.

## 5. Main results

- **Canonical panel (P83):** A1 8/32, B1 6/32, C 4/32, D 17/32; MSP 35/128 = 0.2734375; bootstrap CI [0.1953, 0.3516].
- Clearly label: **not** pristine final-test; **V2 inactive on 126/128 eps** (MSP ≈ Actor-only).
- P70 V2 closed-loop lift on its held panel (SUCCESS/R3 ≈ 0.240 vs V0 ≈ 0.083) — **separate distribution**; do not claim it applies to P83.

Figures: `fig04_p83_per_class_success`.

## 6. Ablations and diagnostics (negative results as negatives)

| Theme | Result (one line) |
|---|---|
| Critic recalibration (P75/P76) | FC EV remains uninformative / worse; retain P64 Critic |
| Sensing / obs (P77/P78) | Sidewall more predictable with eng features; action-relevance underpowered → no obs change |
| History policy (P79/P80) | Actions matter; eng ≯ history; history ≯ fixed tok_56 |
| Fixed-token / overlay (P73/P81) | Matched gains; **full-episode regressions** |

Figure: `fig07_matched_vs_full_episode`.

## 7. Failure analysis

- C contact funnel; exclusive cross-class budget (retain SUM 0.508 **and** fair mean 0.169); B1 diffuse modes; P84 gate fail; **P83 Case D post-audit revision**.
- Distinguish evidence types (Failure Analysis Summary).

Figures: `fig05_c_contact_formation_funnel`, `fig06_cross_class_failure_attribution`.

## 8. Limitations

- Single seed per IC on P83; mixed reuse + new ICs.
- Panel ≠ final-test.
- Matched CF ≠ closed-loop transfer.
- Scenario MSP opportunities are not achieved gains.
- No claim that V2/M111 are globally optimal.

## 9. Conclusion

- Frozen system: P64 Actor/Critic + P70 V2 + original M111 + original R3F / SUCCESS_RELAXED_12S.
- Selected because it is the best **validated deployable no-train** stack after failed closed-loop interventions and a failed concentration gate for further mechanism-specific optimization.
- Optimization campaign closed at P83 Case D (post-audit) + P84 Case D; no P85; no new causal probe.
