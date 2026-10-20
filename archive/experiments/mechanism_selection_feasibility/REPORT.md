# Mechanism-Selection Feasibility — REPORT

Read-only audit. No training, no reward/physics/checkpoint changes.

See also `OBSERVATION_AUDIT.md`, `MECHANISM_CLASSIFICATION.md`.

## Q1 — Are the seven `[36,36,47]` physically Recatch?

- Substep labels: {'AMBIGUOUS': 7}
- Status: **SUPPORTED_BUT_INCOMPLETE**
- Token 36 has **grip-open** command (a_grip=-1 → TAU_OPEN). Substep logs show both-off during the 40 ms recovery; bilateral recontact **before HANDOFF** was not consistently observed.
- If regrasp occurs only after HANDOFF (nominal), this is **not** recovery-phase Release–Recatch under the project classifier; call it **AMBIGUOUS fast open-then-handoff** unless sustained both-off + pre-HANDOFF recontact is logged.
- Evidence: `grpo_only_substep_classification.csv`, `raw/grpo_only_substep/`.

## Q2–Q3 — Sliding / Recatch favorable ICs

- Competition search tags: {'BOTH_VERIFIED': 14, 'RECATCH_FAVORABLE': 22, 'SLIDING_FAVORABLE': 4}
- Sliding-favorable: **4** → VERIFIED
- Recatch-favorable: **22** → VERIFIED
- BOTH_VERIFIED: **14**
- Oracle budget: `_try_sliding` schedules + `vertical_specs()` only; search failure ≠ physical impossibility (`UNRESOLVED`).
- Manifest: `IC_MANIFEST.csv`.

## Q4 — Are classes separable in 39D history?

- Group-mean L2 between Sliding-favorable vs Recatch-favorable IC obs: **1.1175**
- Top dims: [{'dim': 9, 'name': 'v_R', 'delta_mean': 0.708526611328125, 'std_s': 0.23894065618515015, 'std_r': 0.6289573311805725}, {'dim': 6, 'name': 'u_L', 'delta_mean': -0.5169547200202942, 'std_s': 0.9833376407623291, 'std_r': 0.8660253882408142}, {'dim': 0, 'name': 'e_hat_x', 'delta_mean': -0.4856436848640442, 'std_s': 0.9856827259063721, 'std_r': 0.8660253882408142}, {'dim': 8, 'name': 'u_R', 'delta_mean': 0.42534777522087097, 'std_s': 0.9253891110420227, 'std_r': 0.8660253882408142}, {'dim': 18, 'name': 'v_hx', 'delta_mean': -0.1600770652294159, 'std_s': 0.48628920316696167, 'std_r': 0.35373929142951965}]
- Status: **SUPPORTED_BUT_INCOMPLETE** — differences exist at IC restore, but ICs are not matched twins; does not prove the current Transformer uses them.
- Matched short-action probe (2× token36 proto): `obs_response_probe.csv`.
- Forbidden: friction/mass/labels are not in 39D (see OBSERVATION_AUDIT).

## Q5 — SFT vs GRPO on competition ICs

- Sliding-favorable: {'n': 3, 'sft': 1, 'grpo': 1, 'holdout_n': 2}
- Recatch-favorable: {'n': 17, 'sft': 0, 'grpo': 3, 'holdout_n': 7}
- BOTH_VERIFIED: {'n': 10, 'sft': 5, 'grpo': 8, 'holdout_n': 5}
- On Sliding-favorable: SFT Contact-Preserving successes=1, GRPO Contact-Preserving successes=1 (of 3).
- Full table: `PAIRED_EVALUATION.csv`. Holdout flag in column `holdout`.

## Q6 — Does evidence support “GRPO loses Sliding for fast Recatch”?

- On Sliding-favorable (n=3 eval): SFT 1 vs GRPO 1 — **no Sliding loss in this tiny slice** → not supported here.
- On BOTH_VERIFIED (n=10): SFT 5/10 all `Contact-Preserving Adjustment`; GRPO 8/10 with **4 Contact-Preserving + 4 Other/Unclear**.
- On Recatch-favorable (n=17): SFT 0/17, GRPO 3/17 all `Other/Unclear` (not verified Recatch labels).
- Combined status: **SUPPORTED_BUT_INCOMPLETE** for “GRPO shifts toward fast ambiguous open-handoff on some recoverable ICs”; **UNRESOLVED** for “GRPO systematically destroys verified Sliding skill.”
- Prior 7 GRPO-only hard-recovery gains remain **AMBIGUOUS** open-then-nominal-regrasp, not classifier Recatch.

## Q7 — Next priority

1. **Expand Sliding-favorable coverage** (only 4 found under current oracle budget; need more holdout).
2. Keep recording per-trajectory mechanism labels in future GRPO logs.
3. Defer reward λ changes; if acting on training, prefer sampling / all-success time-advantage monitoring (prior audit) after IC coverage improves.
- Do **not** auto-train; wait for human decision.

## Stop
Audit complete. No SFT/GRPO launched; reward unmodified.
