# Verified Results Tables

All numbers below are copied from saved artifacts. Missing cells are left blank rather than estimated. **Do not treat rows from different panels as paired.**

---

## Table A — Main checkpoints

| Asset | Path | MD5 | Source |
|---|---|---|---|
| Actor | `.../p64_ppo_multiround/checkpoints/P64_step4_actor.pt` | `9f4c50413535e4c7db3d3a868bc1b6e2` | P82 freeze + live re-hash |
| Critic | `.../p64_ppo_multiround/checkpoints/P64_step4_critic.pt` | `95de94e5a5d31ba8acddbfb49a6981cb` | P82 freeze + live re-hash |
| M111 book size | loaded from Actor | 111 | live `load_actor` |

---

## Table B — P70 closed-loop variants (held full episodes, n=96 each)

Source: `p70_geometry_to_r3f/FULL_EPISODE_VALIDATION.md` / `FINAL_REPORT.md`

| Variant | R3-ready | SUCCESS_RELAXED | sidewall | min_pad≥4 | e first contact (mm) |
|---|---:|---:|---:|---:|---:|
| V0 (Actor only) | 0.083333… | 0.083333… | 0.240 | 0.469 | 19.7 |
| V1 fixed seq | 0.15625 | 0.145833… | 0.469 | 0.510 | 21.2 |
| **V2 adaptive** | **0.239583…** | **0.239583…** | 0.458 | **0.573** | 21.8 |

V2 − V0 ΔSUCCESS_RELAXED / ΔR3-ready = **+0.15625** (`FINAL_ANSWERS.json`).  
Decision: `do_not_train` (absolute first-contact e_pos gate failed).

P70 Phase-1 funnel (separate n=64 panel; **not** the n=96 closed-loop panel) — `GEOMETRY_TO_R3F_FUNNEL.json` summary:

| Tag | n | SUCCESS_RELAXED | r3_ready | correct_sidewall | e_first_contact |
|---|---:|---:|---:|---:|---:|
| P64 | 64 | 0.140625 | 0.140625 | 0.34375 | 21.0819… |
| P64+V1 | 64 | 0.15625 | 0.203125 | 0.515625 | 21.2049… |

---

## Table C — Canonical final evaluation panel (P83)

Source: `p83_recoverable_failure_mass/BASELINE_PANEL_VALIDATION.md`

| Class | Successes / n | Rate |
|---|---:|---:|
| A1 | **8 / 32** | 0.25 |
| B1 | **6 / 32** | 0.1875 |
| C | **4 / 32** | 0.125 |
| D | **17 / 32** | 0.53125 |
| **Equal-weight MSP** | **35 / 128** | **0.2734375** |

Bootstrap MSP 95% CI (IC-clustered): **[0.1953125, 0.3515625]**  
Source: `STATISTICAL_SENSITIVITY.md`.

**Limitations (must report):**

- Panel mixes 64 P82-reused + 64 newly collected ICs (selection not outcome-based).
- `N_SEEDS = 1` per IC.
- Blocked eval / final-test excluded; **this is not a pristine final-test result**.
- Used for diagnostics after freeze; not an untouched holdout claim.
- **V2 activation:** `forced_steps > 0` in only **2/128** episodes (B1 only). P83 rates are ≈ **P64 Actor-only**, not active V2. Source: `panel_32.pkl` + `WORKFLOW_AUDIT.md` ([frozen-path audit](3f39613c-53f3-4b2c-a38f-d31de7709092)).
- Heavy overlap with P64 training IC exposure (~120/128).

---

## Table D — Controlled comparisons & negative closed-loop results

### P73 post-contact overlay vs V2 (held n=96)

Source: `p73_first_contact_bilateral/P74_DECISION.json` → `answers.6_terminal_success`

| System | SUCCESS_RELAXED | r3_ready | reached_opposing |
|---|---:|---:|---:|
| V0 | 0.135416… | 0.135416… | 0.291666… |
| V2 | 0.229166… | 0.25 | 0.53125 |
| P73 soft_approach | 0.135416… | 0.135416… | 0.302083… |
| **Δ (P73 − V2)** | **−0.09375** | **−0.114583…** | **−0.229166…** |

Decision: Case C, `retain_v2`.

> Note: P73 V0/V2 rates differ from P70 Table B — **different evaluation campaigns/panels**.

### P81 matched CF vs full-episode tok_56 bias

Source: `p81_static_contact_bias/FINAL_REPORT.json`, `FULL_EPISODE_TRANSFER.md`

| Evidence type | Metric | Value |
|---|---|---:|
| Matched snapshots | mean ΔSUCCESS (tok_56 − actor) | **+0.0875** |
| Full episode (8 ICs × 4 classes) | MSP baseline | 0.34375 |
| Full episode | MSP inject56 | 0.09375 |
| Full episode | **MSP Δ** | **−0.25** |
| Full episode per-class ΔSUCCESS | A1 / B1 / C / D | −0.25 / −0.375 / 0.0 / −0.375 |

Decision: Case B — matched gain does **not** transfer; stop fixed-token direction.

---

## Table E — Critic diagnostics (P75 / P76)

Source: `p75_critic_calibration/P76_DECISION.json`, `p76_value_observability_audit/FINAL_REPORT.md`

| Metric (FC window) | P64 Critic | Best alt `A_w1` |
|---|---:|---:|
| EV | −0.050575… | −0.650483… |
| Corr | 0.151289… | 0.466981… |
| Bias | 4.891… | 5.174… |
| ΔEV (best − baseline) | | **−0.599908…** |

P76: affine calibration does not restore useful overall FC EV (`A_w1` cal EV ≈ −0.57875). Case C → retain P64 Critic; stop Critic-calibration budget.

---

## Table F — Observation / sensing diagnostics (P77 / P78)

### P77 sidewall discrimination AUROC

Source: `p77_contact_observability/FINAL_REPORT.md`

| Feature set | Sidewall AUROC |
|---|---:|
| Current / engineered current (A) | 0.864635… |
| Short history (B) | 0.853855… |
| Full history (C) | 0.828150… |
| Engineered (reported D in answers) | 0.919361… |

Case A: engineered/history features help discrimination → recommend controlled obs experiment; **no automatic train**.

### P78 action-relevance ranking

Source: `p78_action_relevant_representation/FINAL_REPORT.md`

- Case **C** (insufficient action-relevance evidence).
- Point ranks A≈0.806, B≈0.910, C≈0.916; IC-bootstrap CI for eng lift includes 0; history alone matches eng.
- Matched CF ranking `n_pairs=3` (underpowered).

---

## Table G — Action-selection findings (P79 / P80)

### P79 matched CF coverage

Source: `p79_matched_cf_coverage/FINAL_REPORT.json`

| Item | Value |
|---|---|
| Independent ICs | 66 |
| Disagree rate (nontrivial outcome diffs) | 0.310606… (86/396) |
| Rank A / B / C / D | 0.65 / 0.725 / 0.75 / 0.75 |
| eng_over_current | true |
| eng_over_history | **false** |
| Case | **B** |

### P80 history-conditioned feasibility

Source: `p80_policy_learning_feasibility/FINAL_REPORT.json`

| Method (held SUCCESS) | Value |
|---|---:|
| History ranker D | 0.352941… |
| Best fixed C (`tok_56_soft`) | 0.352941… |
| Always-56 | 0.352941… |
| Actor | 0.294117… |
| D − C paired mean / CI | 0.0 / [0.0, 0.0] |
| Reliable reversals vs 56 | 2 |
| Preference mode | mostly_ties_weak_signal |
| Case | **B** (investigate static bias; do not train Actor) |

---

## Table H — Failure-budget conclusions (P82 / P83 / P84)

### P83 failure masses (equal-weight)

Source: `FAILURE_MASS.md` comparisons

| Quantity | Value |
|---|---:|
| A — C contact-formation mass | 0.1640625 |
| B — combined A1/B1/D any-failure mass (**legacy SUM**) | 0.5078125 |
| B′ — **mean** A1/B1/D any-failure mass (fair) | 0.1692708… |
| B — combined A1/B1/D contact-formation mass | 0.3671875 |
| C — A1 / B1 / D individual any-failure mass | 0.1875 / 0.203125 / 0.1171875 |
| D — handoff mass (all classes) | 0.09375 |
| D — stabilization mass (all classes) | 0.0625 |

**Post-audit decision revision:** Case **D** → `stop_consolidate_baseline` (fair mean 0.169 vs C_cf 0.164).  
Original Case **C** (driven by legacy SUM 0.508) archived in `P83_DECISION_ORIGINAL_CASE_C.json`.  
SUM vs mean vs C_cf are **not** homogeneous recoverability metrics.  
P83 MSP 0.2734375 with V2 activation **2/128** → ≈ Actor-only (not V2 closed-loop evidence).

### P84 B1 concentration gate

Source: `FAILURE_CONCENTRATION.md` / `FINAL_REPORT.json`

| Mode | Count / 26 fails | Pass ≥40% (≥11)? |
|---|---:|---|
| no_e_pos_le9 (largest) | **9 / 26** (0.346…) | **No** |
| no_min_pad4 | 5 / 26 | No |
| no_frame_ok | 4 / 26 | No |

Case **D**: stop; consolidate baseline. **P85 not launched.**

---

## Explicit non-comparisons

Do **not** present as one progression curve:

1. P70 n=96 V0 SUCCESS 0.083 vs P73 n=96 V0 SUCCESS 0.135  
2. P81 n=32 MSP 0.34375 vs P83 n=128 MSP 0.2734375  
3. Matched CF ΔSUCCESS vs full-episode MSP Δ as if the same estimand
