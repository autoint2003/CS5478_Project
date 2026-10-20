# REPORT — Plan A controlled sliding / pivoting legality audit

Independent audit. No physics/controller/action-map/obs/SUCCESS/K49 changes. No SFT/GRPO/Plan C.

Wall time: 339.8s

## Experts audited

- `hard_910263` root=`hard_910263` source=`k49_slow_return_position_refinement` available=1 tokens=144 claimed=`SLIDING+PIVOTING+RETURN`
- `hrv1_920031` root=`hrv1_920031` source=`k49_slow_return_position_refinement` available=1 tokens=161 claimed=`SLIDING+PIVOTING+RETURN`
- `hard_910101` root=`hard_910101` source=`sliding_position_correction_v1` available=1 tokens=29 claimed=`PIVOTING`
- `hrv1_920031_return_aware` root=`hrv1_920031` source=`return_aware_sliding_k49` available=1 tokens=70 claimed=`SLIDING+RETURN(tok48)`

## Verdicts (separate columns)

| Expert | Legality | Sliding | Pivoting | Contact | Exact replay | SUCCESS_NEW | GROUND_SCENE |
|---|---|---|---|---|---|---|---|
| hard_910263 | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL |
| hrv1_920031 | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| hard_910101 | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL |
| hrv1_920031_return_aware | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |

## Headline findings

1. **Legality PASS** for all four: only `step_legal`, frozen K49 decode, no HANDOFF-as-motion, no detected teleports.
2. **Exact replay + SUCCESS_NEW PASS** on elevated-table for all four (triple identical).
3. **Mechanism ≠ success:** `hard_910263` and `hard_910101` have SUCCESS_NEW but **FAIL** contact-preserving sliding/pivoting (bilateral gap ≈ 40 ms).
4. **Best elevated mechanism evidence:** `hrv1_920031` and `hrv1_920031_return_aware` PASS sliding + pivoting + contact.
5. **GROUND_SCENE:** matched transfer keeps bilateral grasp and can still reach SUCCESS_NEW / SUCCESS_12S, but **claimed sliding/pivoting gates FAIL** on ground for all four — do not assume Plan A mechanisms transfer.

See `FAILURE_ANALYSIS.md` and `VERDICTS.csv`.

## Method notes

- Recovery actions executed only via `envs.step_legal.step_legal` after frozen K49/`K49_tok48_wy0p25` decode.
- HANDOFF via `apply_handoff` (terminal); not a motion token in the audited tapes.
- Sliding: hand-frame relative COM axial displacement during bilateral contact (threshold 3 mm), with co-motion rejection when world hand/object move together without relative axial slip.
- Pivoting: change in cylinder-axis alignment vs hand grasp axis during bilateral contact (threshold 5°).
- Contact preservation fails if bilateral gap ≥ 0.020 s (one policy step).
- Exact replay: three identical elevated-table rollouts; PASS requires identical SUCCESS_NEW and SUCCESS_12S flags.
- GROUND_SCENE: matched qpos/qvel/ctrl/FSM transfer (nq identical); not a blind incompatible snapshot load.

Raw traces: `raw/*__elevated_trace.json`, `raw/*__ground_trace.json`.

Reusable code: `evaluation/plan_a_mechanism_audit.py`, runner `scripts/run_plan_a_sliding_pivoting_legality_audit.py`.
