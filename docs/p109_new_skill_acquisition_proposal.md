# P109 Proposal — Updated after P109-A

## P109-A decision

`P109A_PREDICTIVE_SIGNAL_WEAK`

### Answers

```json
{
  "Q1_predict_from_observable": false,
  "Q2_history_helps": true,
  "Q3_p64_beyond_history": false,
  "Q3_detail": "True at 0.2s (C>B); False at 0.5s (B>C)",
  "Q4_early_warning_before_loss": false,
  "Q4_detail": "8 before_instability events but median lead 0.05s < pre-reg 0.10s",
  "Q5_generalizes_across_roots": true,
  "Q6_gradient_path_exists": true,
  "Q7_justify_p109b": false
}
```

### Key metrics (T2 contact-loss, logreg, new-root test)

| Horizon | A current | B history | C P64 | C−A |
|---|---:|---:|---:|---:|
| 0.2s | 0.427 | 0.527 | 0.626 | +0.199 |
| 0.5s | 0.489 | 0.666 | 0.557 | +0.068 |

Early-warning median lead @ thr=0.5: **0.05s** (pre-registered minimum was 0.10s).

### Gate outcome

Label integrity OK · root-disjoint new development test OK · modest out-of-root AUROC · **early-warning lead failed** · gradient path exists · history competitive at 0.5s.

**P109-B is not recommended** under the pre-registered criteria.

## Conditional P109-B (proposal only; NOT executed; NOT authorized)

If and only if a future audit passes early-warning lead gates and human approves:

| Arm | Description |
|---|---|
| A | Frozen P64 |
| B | SFT + fixed retention; final Transformer block trainable |
| C | Identical to B + future contact auxiliary (train-only head) |

Matched init/data/seeds/budget/retention. One unified Actor. No sealed panel in B.

### Falsification

C ≯ B on new-root C `SUCCESS_RELAXED_12S` under matched budget → reject H2 for this regime.

## Notes

P108 corrected ranking preferred H2 over H1 after removing the biased expert-success bonus. P109-A tested the predictive premises of H2 without Actor updates and found a weak / gate-failing signal. Prefer stopping Actor-side contact auxiliaries until stronger early-warning evidence exists.
