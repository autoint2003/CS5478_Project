# Gravity-Assisted Contact-Preserving Pivoting — FEASIBILITY PILOT REPORT

**Stop after pilot.** No reward/SUCCESS_12S/training/mining changes.

## Reproducible command

```text
python -u scripts/pilot_gravity_assisted_pivoting.py --device cuda --seed 20261018 --max-traj-per-ic 180
```

Outputs: this directory (`CANDIDATE_IC_AUDIT.csv`, `LEGAL_SEARCH_SUMMARY.csv`, `CONTACT_TOPOLOGY_AUDIT.md`, `COUNTERFACTUAL_HANDOFF.md`, `RAW/`, `videos/`).

## Reused sources (VERIFIED paths)

| Role | Path |
|---|---|
| Config metrics | `evaluation/grasp_configuration.py` |
| Contacts | `envs/contacts/frames.py::extract_contacts` |
| Legal step | `envs/step_legal.py` (+ diagnostic substep replica in script) |
| Action map / τ_grip | `controllers/recovery/action_map.py` (`TAU_OPEN=+2`, `TAU_SECURE=-18`) |
| HANDOFF + nominal | `envs/outcomes/recovery_success.py`, `horizon_outcome.py` |
| Snapshot restore | `envs/simulation/capture.py::restore_capture` |
| Prior IC list / shorts | `nontrivial_recovery_ic_mining/`, `grasp_configuration_migration/` |

## Physical feasibility before search (Q1)

### Geometry

- Cylinder axis = body +Z; COM at body origin; finger pads on `left_finger`/`right_finger`.
- Candidate pivot = mean finger–object contact position.
- Measured lever arm COM→pivot: **≈5–7 mm** on all 6 ICs.
- `|τ_g|` about pivot: **≈0.004–0.014 N·m**.
- Secure grip torque scale: **18 N·m**; open: **2 N·m**.

### Directionality

Only **1/6** ICs (`hrv1_920031`) has gravity torque **aligned** with the hand-frame rotation that reduces `e_theta`. On the other five (including stability-only `hard_910101`), gravity torque **opposes** or is near-orthogonal to error reduction.

**Verdict (VERIFIED measurements + EXPERIMENTAL):** Gravity-assisted rotation *in the desired direction* is **not generally available** on these eccentric grasps. Even when aligned, `|τ_g|` is tiny vs grip; **unload** is required for relative motion.

## Search budget (EXPERIMENTAL)

- 6 ICs × ≤180 evaluated trajectories (plus baseline/prefix re-runs).
- Horizon cap 2 s (≤100 tokens); typical proposals 3–25 tokens.
- Zero residual decode; full 7D codebook rows inspected (no “pure open” assumption).
- Seed `20261018`.
- **0× VERIFIED_CONTACT_PRESERVING_PIVOT**; **0× VERIFIED_SINGLE_CONTACT_PIVOT**; **2× VERIFIED_RELEASE_REGRASP**.

## K48 control (Q2)

K48 **can** command grip unload→reload and wrist/translation mixes (`decode_action` + `map_action`). Soft-open / mid / close families exist (e.g. tok39 open+trans, tok12 mid τ≈−3.2, tok14/0 secure).  

**Limitation (EXPERIMENTAL):** In this budget, sequences that produce meaningful Δ`e_theta` almost always show **sustained zero-contact** in substep logs — i.e. regulation without losing contact was **not** demonstrated.

## Main empirical results

1. Short HANDOFF baselines stabilize (`SUCCESS_12S`) but **do not** correct large tilts (eθ_final ≈ eθ_0 within ~1° on high-error ICs).
2. Longer legal sequences can improve final eθ and beat short baselines on some ICs (`hrv1_920031`, `hrv1_920151`) with `SUCCESS_12S`, but contact topology = **release–regrasp**, not CPP.
3. Large mid-phase Δθ with open tokens often fails after HANDOFF (`RELATIVE_ORIENTATION_CHANGE_BUT_UNSTABLE`).

Videos: `videos/hrv1_920031_best_attempt.mp4`, `videos/hard_910101_short_baseline.mp4`, `videos/hard_910101_immediate_handoff.mp4`.

---

## Decision questions

### Q1. Does contact geometry allow gravity-assisted relative rotation in the desired direction?

**Mostly no on this IC set.** Only 1/6 has helpful torque alignment; all have tiny `|τ_g|` (≪ grip).  
Labels: geometry **VERIFIED**; helpfulness **EXPERIMENTAL**.

### Q2. Does K48 provide enough grip/wrist control to initiate, regulate, and brake that rotation?

**Initiate/brake unload–reload: yes (capability VERIFIED in codebook).**  
**Regulate while preserving contact: not evidenced (UNRESOLVED / failed under budget).**

### Q3. Was any genuine contact-preserving relative orientation correction observed?

**No.** `VERIFIED_CONTACT_PRESERVING_PIVOT = 0`.  
**UNRESOLVED** as global impossibility; **VERIFIED absence** in this bounded pilot.

### Q4. Did any legal contact-preserving trajectory achieve SUCCESS_12S with improved configuration?

**No contact-preserving one.** Release–regrasp trajectories did (`hrv1_920151`, and config-improving `hrv1_920031` with release topology).

### Q5. Do short-prefix HANDOFF baselines already achieve equivalent configuration correction?

**No** on high-tilt ICs (short leaves ~12–21°; best longer sequences reach ~2° on some ICs via release-assisted paths).

### Q6. Is there ≥1 IC where active pivoting improves final config beyond short-HANDOFF?

**Active legal sequences: yes for config (`hrv1_920031`, `hrv1_920151`).**  
**Active contact-preserving pivoting: no.**

### Q7. If pilot “fails” CPP, primary evidenced bottleneck?

1. **Contact geometry / torque scale** (primary): lever ~6 mm, `|τ_g|~0.01 N·m` vs grip 2–18 N·m; gravity often misaligned.  
2. **Grip unload ⇒ separation**: orientation change couples to sustained zero-contact.  
3. Secondary: **search coverage** (bounded; not exhaustive) — do **not** claim physical impossibility.  
Not primary: observability (39D GT relative pose sufficient); action discretization alone (release–regrasp still expressible).

---

## Stop

Pilot complete. Do not train, edit reward/success, or expand mining without a separate decision.
