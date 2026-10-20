# Wrist-Tilt-Assisted Gravity Pivoting — PILOT REPORT

**Stop after bounded pilot.** No training / reward / SUCCESS_12S / physics edits.

```text
python -u scripts/pilot_wrist_tilt_gravity_pivoting.py --device cuda --seed 20261018 --max-per-ic 35
```

Artifacts: `PHASE1_PRIOR_PILOT_ANALYSIS.json`, `IC_AUDIT.csv`, `EXPERIMENT_TABLE.csv`, `RAW/`, `videos/`, `summary.json`.

---

## Phase 1 — Failed undirected pilot (VERIFIED)

Source: `results/diagnostics/gravity_assisted_pivoting_feasibility/`

| Question | Finding | Confidence |
|---|---|---|
| 1. Wrist pre-tilt before grip release? | **No.** 588/588 search rows **started with open** tokens (`a6<-0.5`). Close+wrist starts were not systematically evaluated. | VERIFIED |
| 2. ~6 mm lever too small? | Levers **5.5–7.1 mm**; `‖τ_g‖≈0.004–0.014 N·m` ≪ secure grip (18 N·m). Small, but **not proven fatal** once grip is unloaded to mid. | VERIFIED magnitudes; lethality = HYPOTHESIS until this pilot |
| 3. Gravity projection favorable? | Only **1/6** ICs at t0 (`hrv1_920031`). | VERIFIED |
| 4. Distinct K48 grip levels? | **Only 3** values of `a[6]`: `−1` (τ=+2), `≈−0.48` (τ≈−3.2, tok **12**), `+1` (τ=−18). | VERIFIED |
| 5. Release–regrasp lost contact on unload? | Both prior VERIFIED_RELEASE cases began with tok **39** (open τ=+2 + translation); sustained zero-contact ~70 ms. | VERIFIED |

**Hypothesis motivating this pilot:** closed-wrist pre-tilt can improve gravity alignment; mid unload (tok12) may allow relative rotation without sustained bilateral loss.

---

## Phase 2 — Candidates

Reused physics snapshots (no teleport), prioritized high `e_theta` + prior eccentric set:

`hard_910101`, `hrv1_920131`, `hrv1_920099`, `hrv1_920031`, `hard_910172`, `hrv1_920151`.

Eccentricity (hand-frame COM xy): **≈8.9–12.5 mm**. All bilateral at restore.

**Tilt probes (closed tokens 6/4/10 ×12 steps):** favorable gravity after tilt on **hard_910101, hrv1_920031, hrv1_920151** (`IC_AUDIT.csv`). Others remained misaligned.

---

## Phase 3 — Structured legal sequences

Fixed recipe (K48 only, zero residual, `step_legal`):

1. **Tilt** while secure: tokens `{6,4,10}` (`a6=+1`, nonzero wrist).
2. **Unload**: tok **12** (mid) or **36** (open).
3. **Brake**: tok `{0,14}` secure.
4. HANDOFF + full nominal `SUCCESS_12S`.

Budget: ≤35 sequences/IC from a 923-sequence catalog (tilt-first prioritized). Controls: few open-first sequences.

Token semantics recorded in `RAW/structured_sequence_catalog.json`.

---

## Phase 4 — Verification result

### Label counts (`EXPERIMENT_TABLE.csv`, 186 trials)

| Label | N |
|---|---:|
| **VERIFIED_CONTACT_PRESERVING_PIVOT** | **1** |
| VERIFIED_SINGLE_CONTACT_PIVOT | 0 |
| VERIFIED_RELEASE_REGRASP | 3 |
| RELATIVE_ORIENTATION_CHANGE_BUT_UNSTABLE | 17 |
| NO_MEANINGFUL_RELATIVE_ROTATION | 12 |
| STABLE_BUT_CONFIGURATION_INCORRECT | 1 |
| AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE | 152 |

### Verified CPP (EXPERIMENTAL, substep-backed)

**IC:** `hrv1_920031`  
**Sequence:** `tilt6×20 → unload12×15 → brake0×8` then HANDOFF  
**Raw:** `RAW/hrv1_920031_tilt6x20_u12x15_b0x8.json`  
**Video:** `videos/hrv1_920031_best_structured.mp4`

| Phase | Substeps | Bilateral frac | Zero-contact subs | eθ (°) | Notes |
|---|---:|---:|---:|---|---|
| tilt | 200 | **1.00** | 0 | 13.67 → 13.36 | Wrist tilt ~178°→91°; gravity align 0.005→**0.012** (helps↑) |
| unload | 150 | **0.89** | 2 (≤2 ms streaks) | 13.35 → **4.96** | `‖w_rel‖` up to 6.4 rad/s — relative rotation |
| brake | 80 | **1.00** | 0 | ~5.04 | Holds |
| nominal | — | — | — | final **5.02** | **SUCCESS_12S** |

Topology: `frac_bilateral=0.96`, `max_zero_ms≈2`, **not** `sustained_release`.  
Δeθ mid/final ≈ **8.6°**. Correction is **object–gripper** (`e_theta` from `u_C_H`), not world-frame-only wrist carry.

**Caveat:** This is **one** IC under a **bounded** structured set. Do not over-generalize.

### Release–regrasp still appears

Several tilt→**open36**→brake trials get large Δeθ + SUCCESS_12S but `max_zero_ms≈46–48` → labeled **VERIFIED_RELEASE_REGRASP** (e.g. `hrv1_920131` / `hrv1_920099`).

---

## Phase 5 — Limiting factors

| Factor | Evidence |
|---|---|
| Favorable gravity after tilt? | **Sometimes yes** (3/6 ICs in probes; CPP case increased align). |
| Controlled relative rotation? | **Yes** on CPP unload phase (eθ↓ with large `w_rel`). |
| Contact retained? | **Yes** for mid-unload CPP; **often no** for open-unload. |
| Brake + nominal survival? | **Yes** on CPP case. |
| Primary bottleneck when CPP fails | (1) **Gravity still misaligned after tilt** on 3/6 ICs; (2) **open unload ⇒ release**; (3) **only 3 grip levels** — mid (12) is the only intermediate; (4) small lever still limits torque; (5) **bounded coverage** (35/IC). |

**Not claimed impossible** outside this budget. Wrist pre-tilt + mid unload is a **plausible legal mechanism** with at least one verified instance.

---

## Answers (compact)

1. Prior pilot did **not** systematically wrist-tilt-before-unload (VERIFIED).  
2. Structured tilt→mid→brake produced **1× VERIFIED_CONTACT_PRESERVING_PIVOT** on `hrv1_920031` (EXPERIMENTAL).  
3. Open unload still tends to release–regrasp.  
4. K48 grip resolution is coarse (3 levels); tok12 is critical.  
5. Next (separate decision): expand structured schedule around tilt×mid on gravity-help ICs; do not resume open-first undirected search.

## Stop

Pilot complete.
