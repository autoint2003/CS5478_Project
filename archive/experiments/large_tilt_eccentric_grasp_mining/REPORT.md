# Large-Tilt Eccentric Grasp IC Mining — PILOT REPORT

**Stop after bounded pilot.** No teleport, no training, no reward / SUCCESS_12S / physics / K48 / 39D changes.

```text
python -u scripts/pilot_large_tilt_eccentric_grasp_mining.py --device cuda --seed 20261018 --max-candidates 40 --short-budget 28 --active-budget 36 --active-top 8 --gen-extra 10
python -u scripts/_tmp_verify_large_tilt_cpp.py   # substep contact re-check + CPP videos
```

Artifacts: `IC_GENERATION_AUDIT.csv`, `SHORT_HANDOFF_FILTER.csv`, `ACTIVE_CORRECTION_RESULTS.csv`, `CONTACT_MECHANISM_AUDIT.md`, `RAW/`, `snapshots/`, `videos/`, `summary.json`.

---

## Prior evidence (review)

| Source | Relevant fact |
|---|---|
| `wrist_tilt_gravity_pivoting/` | 1× VERIFIED bilateral CPP on `hrv1_920031` (eθ≈13.7°→5.0°); recipe tilt6×20→mid12×15→brake0×8 |
| `pose_correction_counterfactual_v1/` | Same IC: pivot edge vs naïve short `[30]` is clear; vs best short `[34,32]` only ~1.8° window θ (short better on position) |
| Snapshot scan (this pilot) | 117 unique bilateral snaps; bins `<15`:99, `15–25`:11, `25–40`:5, `>40`:2; max eθ **51.3°** (`hrv1_920117`) |

Verified pivot IC origin: **ballistic impact** hard-recovery snapshot `hrv1_920031` (not teleported). Geometry: bilateral contact, modest eccentricity (~10–15 mm class), gravity alignment that becomes favorable after closed wrist tilt.

---

## Pool construction

### Existing physical snapshots

18 bilateral candidates with eθ≥15° selected into the pilot pool (deduped fingerprints), spanning all discovery bins:

| Bin | Existing N | Max eθ example |
|---|---:|---|
| 15–25° | 11 | `hrv1_920178` 23.8° |
| 25–40° | 5 | `hrv1_leftover_hard_910185` 37.0° |
| >40° | 2 | `hrv1_920117` **51.3°**, `hrv1_leftover_hard_910144` 42.7° |

### Legal generation (no teleport)

From bases `hrv1_920031`, `hrv1_920117`, `hrv1_920131`: closed/wrist + mid/open unload recipes under `step_legal`, snapshot when still bilateral and eθ≥15°.

- **6** new snapshots retained (pool total **24**).
- Generated eθ up to **35.0°** (`gen_hrv1_920031_2_*`); **no** legally generated state exceeded 40° in this budget.
- Several generated states have **small eccentricity** (3–4 mm) despite large eθ — tilt without lever.

**Generation bottleneck for ≫40°:** not emptiness of the world — ballistic leftovers already reach 51°. Additional legal mid-unload/wrist from recoverable bases did not push past ~35° while retaining bilateral contact. Stopping expansion as specified.

---

## Short-HANDOFF filter (24 ICs)

Budget: immediate `[]`, known shorts `[30]/[44]/…`, plus bounded generic 1–2 token search (≤28 evals/IC). Full unchanged nominal continuation; window = last 2 s of 12 s.

| Metric | N |
|---|---:|
| Immediate HANDOFF SUCCESS_12S | 2 |
| Any best-short SUCCESS_12S | 15 |
| **stable_but_misaligned** (S12 + final window eθ still large vs init) | **14** |

Notable:

- **`hrv1_920117` (51.3°):** short `[30,30]` → S12=1, window θ **39.4°** — stable, grossly misaligned.
- **`hrv1_leftover_hard_910144` (42.7°):** short `[13,7]` → window θ **12.4°** — short already recovers a lot of orientation without a long pivot recipe.
- **`gen_hrv1_920031_0_*` (33.1°):** immediate HANDOFF already S12; window θ stays **~33°** — textbook “stable wrong config.”
- Failures of short search (9 ICs) are **not** claimed to require pivoting; many simply drop under this budget.

---

## Active correction (top 8)

Structured catalog: closed tilt `{4,6,10}` × mid/open unload `{12,36}` × brake `{0,14}`, plus open-first recatch proposals. **≤36 sequences/IC**. Seed `20261018`.

| IC | eθ0 | Best short θ | Best active | Mech | Δθ vs short |
|---|---:|---:|---|---|---:|
| `hrv1_920117` | 51.3 | 39.4 | — | none S12 in budget | — |
| `hrv1_leftover_hard_910144` | 42.7 | **12.4** | — | — | — |
| `hrv1_leftover_hard_910318` | 35.7 | 22.5 | — | — | — |
| `gen_hrv1_920031_0_*` | 33.1 | 33.1 | 33.4 | no meaningful rot. | −0.3 |
| `hrv1_920178` | 23.8 | 19.2 | — | — | — |
| **`hard_910263`** | **23.6** | **14.4** | **1.43** | **CONTACT_PRESERVING_PIVOT** | **+13.0** |
| `hard_910101` | 19.1 | 19.3 | 19.2 | no meaningful rot. | +0.1 |
| `gen_hrv1_920031_1_*` | 18.8 | 17.5 | 14.4 | RELEASE_RECATCH | +3.0 |

**2 / 8** ICs improved ≥2° vs best short under identical evaluation.

### Verified CPP (substep-backed): `hard_910263`

Sequence: `tilt4×20 → unload12×5 → brake0×3` then HANDOFF.  
Raw: `RAW/hard_910263_cpp_verify.json`, videos `videos/hard_910263_{short,pivot}.mp4`.

| Phase | Substeps | Bilateral frac | Zero-contact | eθ (°) |
|---|---:|---:|---:|---|
| tilt (closed) | 200 | **1.00** | 0 | 23.63 → 14.22 |
| unload (mid) | 50 | **0.90** | 0 | 14.22 → 5.69 |
| brake | 30 | **1.00** | 0 | 4.78 → **1.49** |
| nominal window | — | — | — | **1.43** (S12=1) |

Vs short `[32,32]` window **14.42°** / pos 17.7 mm: pivot window **1.43°** / pos 17.0 mm. Orientation advantage is **large and persistent**; position comparable.

Initial state: gravity_helps=1, ecc_xy≈17.8 mm, bilateral (3L/3R).

### Recatch under same budget

Only `gen_hrv1_920031_1_*` produced a labeled RELEASE_RECATCH with S12 (60 zero-contact substeps). Window θ 14.4° vs short 17.5° — modest; **does not match** the CPP orientation result on `hard_910263`.

---

## Answers (Q1–Q7)

| # | Answer | Confidence |
|---|---|---|
| **Q1** | **Yes.** Legal dynamics already yield recoverable bilateral states with eθ up to **51°** (ballistic leftovers) and legally generated bilateral states up to **~35°**. | VERIFIED |
| **Q2** | **No clear positive correlation.** The clearest CPP advantage was at **23.6°**, not at >40°. Several large-tilt ICs either (a) short-correct to ~12°, (b) stabilize at huge residual θ, or (c) fail active search in budget. | EXPERIMENTAL |
| **Q3** | **Yes.** 14/24 short-filtered ICs are stable-but-misaligned, including both >40° cases and the 33° generated IC with immediate-HANDOFF S12. | VERIFIED |
| **Q4** | **Yes, for at least one new IC (`hard_910263`).** Contact-preserving pivoting corrected 23.6°→1.4° and held through nominal. **Not** demonstrated for the >40° set in this budget. | VERIFIED (one IC) |
| **Q5** | **Not equivalently.** One recatch hit under comparable budget with only ~3° edge vs short; CPP on `hard_910263` was far stronger on orientation. | EXPERIMENTAL |
| **Q6** | Beyond raw tilt: **favorable gravity projection**, **sufficient COM eccentricity / lever (~15–20 mm)**, **bilateral contact retained under mid unload**, and **not already short-corrected**. Extreme tilt with tiny lever or post-impact contact chaos did not yield CPP here. | HYPOTHESIS grounded in contrasts |
| **Q7** | **Not yet sufficient for a full pose-aware training distribution.** Evidence now includes **two** distinct CPP ICs (`hrv1_920031`, `hard_910263`) with a clearer short-vs-pivot gap on the second, but hit rate under bounded search is still low (2/8 improve ≥2°; 1/8 CPP). Need a larger curated set of gravity-help + eccentric bilateral ICs before training. | JUDGEMENT |

---

## Threshold note

Do not pick SUCCESS_NEW solely to favor pivoting. On `hard_910263`, any θ threshold between ~2° and ~12° with pos ≤20 mm separates pivot from best short on orientation; position is not the discriminator. On `hrv1_920031` (prior), position thresholds reverse the ranking vs optimized shorts — see counterfactual benchmark.

---

## Stop

Bounded pilot complete. No reward edits. No SFT/GRPO.
