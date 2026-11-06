# Failure Analysis Summary

Evidence types are labeled explicitly. Do not upgrade weaker evidence.

---

## 1. Class C contact-formation funnel

**Type:** Observational nested rates on the P83 panel (n=32 Class C).  
**Source:** `p83_recoverable_failure_mass/CROSS_CLASS_FUNNEL.md`

| Nested step | num/den | rate |
|---|---:|---:|
| P(first_contact \| all) | 29/32 | 0.90625 |
| P(opposing_ever \| first_contact) | 17/29 | 0.5862… |
| P(min_pad_ge4 \| opposing_ever) | 15/17 | 0.8824… |
| P(e_pos_le9 \| min_pad_ge4) | 8/16 | 0.5 |
| P(frame_ok \| e_pos_le9) | 6/8 | 0.75 |
| P(r3_ready \| frame_ok) | 4/6 | 0.666… |
| P(SUCCESS_RELAXED \| r3_ready) | 4/4 | 1.0 |

Unconditional SUCCESS_RELAXED on C: **4/32 = 0.125**.  
Largest exclusive earliest failures on C: `no_opposing_after_fc` **12**, `no_e_pos_le9` **7** (`EXCLUSIVE_FAILURE_ATTRIBUTION.md`).

**Interpretation limit:** Nested conditionals describe the success path; they are not causal intervention effects.

---

## 2. Cross-class exclusive failure budget

**Type:** Observational exclusive attribution (one earliest decisive label per fail).  
**Source:** `EXCLUSIVE_FAILURE_ATTRIBUTION.md`, `FAILURE_MASS.md`

| Class | n_fail | Top exclusive modes (counts) |
|---|---:|---|
| A1 | 24 | no_e_pos_le9 13; no_r3_ready 5; no_min_pad4 4 |
| B1 | 26 | no_e_pos_le9 9; no_min_pad4 5; no_frame_ok 4 |
| C | 28 | no_opposing_after_fc 12; no_e_pos_le9 7; no_first_contact 3 |
| D | 15 | no_opposing_after_fc 6; no_min_pad4 4; no_e_pos_le9 3 |

Equal-weight masses (P83):

- C contact-formation mass = **0.1640625**
- Combined A1/B1/D any-failure mass = **0.5078125**
- No single non-C class ≥ 50% of that combined mass (P83 Case C)

---

## 3. B1 failure taxonomy & P84 concentration gate

**Type:** Observational taxonomy + pre-registered concentration gate.  
**Source:** `p84_nonC_mechanism_discovery/FAILURE_CONCENTRATION.md`, `FINAL_REPORT.md`

Gate: ≥40% of B1 failures **and** ≥11 failed ICs (`ceil(0.4×26)=11`).

| Mode | n_failed_ics | frac of 26 | Pass? |
|---|---:|---:|---|
| no_e_pos_le9 (largest) | 9 | 0.346… | **No** (9 < 11) |
| no_min_pad4 | 5 | 0.192… | No |
| no_frame_ok | 4 | 0.154… | No |

**Result:** Case D — modes are diffuse; stop optimization direction. Max MSP failure-mass upper for largest mode = **0.0703125** (scenario upper bound, not realized gain).

Online signature / recoverability: reported **not** credible for intervention (`answers.8_recoverability_credible: false`).

---

## 4. Matched-state vs full-episode transfer failure

### P73 soft_approach overlay

| Evidence | Result |
|---|---|
| Matched / window CF (selected) | Looked useful in CF selection (`soft_approach` chosen) |
| Full-episode held (n=96) | ΔSUCCESS vs V2 = **−0.09375**; opposing/frame also regress |
| **Type** | Full-episode causal intervention (overlay on) vs V2 |

### P81 tok_56 static bias

| Evidence | Result |
|---|---|
| Matched CF mean ΔSUCCESS | **+0.0875** (CI excludes 0 per report) |
| Full-episode MSP Δ | **−0.25** (0.34375 → 0.09375); A/B/D regress |
| **Type** | Matched counterfactual ≠ full-episode causal transfer |

**Cross-cutting conclusion:** Local matched action advantages repeatedly fail to generalize under closed-loop full episodes (also consistent with P73).

---

## 5. Limitations of action-ranking evidence

| Study | Limitation |
|---|---|
| P78 | Ranking n_pairs=3; Case C — insufficient for obs change |
| P79 | Actions change outcomes (disagree≈0.31), but eng features do not beat history; not a train green-light |
| P80 | History ranker ties best fixed tok_56; mostly ties / weak reversals (2 reliable); dataset insufficient for Actor pilot |
| Ranking SUCCESS on matched snaps | Not proof of full-episode MSP improvement (P81 falsifies transfer) |

---

## 6. Evidence-type legend (use in report)

| Label | Meaning | Examples |
|---|---|---|
| Observational correlation | Rates / funnels / AUROCs / exclusive counts on logged rollouts | P83 funnel; P77 AUROC; failure masses |
| Matched counterfactual | Restore same state; swap action; compare short/terminal outcomes | P79/P81 matched Δ |
| Full-episode causal intervention | Closed-loop policy change for whole episode vs baseline | P73 overlay; P81 inject56; P70 V2 vs V0 |
| Scenario opportunity estimate | Upper-bound MSP if conversion assumptions hold | P83 MSP scenarios; P84 mass upper 0.0703 |

Never present scenario estimates or matched Δ as achieved closed-loop gains.
