# Validation recovery skill classification

Checkpoints:

- SFT: `results/sft/checkpoint_epoch_0040.pt`
- GRPO: `results/grpo/tree80/checkpoint_epoch_0003.pt`

Validation set: **17** globally unique ICs from `data/valid_skill_quotas.jsonl` (same list as `scripts/train.py --eval` with the tree80 / current eval manifest).

Action rule: official deterministic `roll_policy` — argmax token, residual mean, legal decode, unchanged nominal handoff, full 12 s nominal continuation.

Contact extraction and restore logic reuse `visualization/audit_checkpoint3_skill.py`. Batch runner: `visualization/classify_validation_skills.py`.

All **34** traces were executed (17 × 2). Outputs live under `results/diagnostics/validation_skill_classification/`.

---

## 1. Classification rules

These labels are derived from contact topology and relative motion, **not** from data tags, tokens, or video appearance.

| Code | Name | Rule |
|---|---|---|
| A | Contact-Preserving Adjustment | No sustained both-finger-off (≥ 0.04 s). Contact preserved on every recovery sample. Measurable in-contact adjustment: bilateral contact-centroid migration ≥ 1 mm **or** object travel in the hand frame ≥ 1 mm. |
| B | Release–Recatch | Sustained both-off ≥ 0.04 s (two policy steps), then bilateral recontact before handoff. |
| C | Hybrid | Satisfies B, **and** post-recontact bilateral centroid migration ≥ 2 mm. |
| D | Other / Unclear | Sub-threshold gaps only, preserved contact without measurable adjustment, incomplete regrasp, or insufficient contact geometry. |

Notes:

- A single-frame both-off is not Recatch.
- Unilateral contact is not full grasp loss.
- Horizontal / Vertical Recatch naming follows the original grasp scene of the IC, not cylinder motion during recovery.
- `SUCCESS_12S` is recorded separately and is never used as a mechanism label.
- Sensitivity alternatives are stored per trajectory: both-off 0.06 s; migration 2 mm with post-gap 3 mm.
- Many episodes are marked **borderline** because the both-off duration equals 0.04 s exactly.

Migration method: maximum L2 displacement of the bilateral fingertip-contact centroid in the hand frame, relative to the first bilateral sample in the analyzed window.

---

## 2. Per-IC comparison

| IC | data labels | SFT mechanism | SFT result | GRPO mechanism | GRPO result |
|---|---|---|---|---|---|
| `inward_extra_250043` | gravity_assisted_in_hand_sliding | Other/Unclear | TABLE_DROP | Release-Recatch | DROP_AFTER_HANDOFF |
| `inward_extra_250051` | gravity_assisted_in_hand_sliding | Hybrid | LOST_GRASP | Hybrid | SUCCESS_12S |
| `inward_extra_250131` | gravity_assisted_in_hand_sliding | Hybrid | DROP_AFTER_HANDOFF | Hybrid | SUCCESS_12S |
| `k48r_valid_00004` | gravity_assisted_in_hand_sliding,vertical_recatch | Hybrid | DROP_AFTER_HANDOFF | Hybrid | SUCCESS_12S |
| `k48r_valid_00009` | gravity_assisted_in_hand_sliding | Hybrid | LOST_GRASP | Hybrid | SUCCESS_12S |
| `k48r_valid_00036` | gravity_assisted_in_hand_sliding,vertical_recatch | Release-Recatch | SUCCESS_12S | Hybrid | SUCCESS_12S |
| `k48r_valid_00168` | gravity_assisted_in_hand_sliding,vertical_recatch | Hybrid | LOST_GRASP | Release-Recatch | SUCCESS_12S |
| `s7_p25_a+0.000_e0.000` | horizontal_recatch | Release-Recatch | SUCCESS_12S | Hybrid | SUCCESS_12S |
| `s7_p25_a+0.000_e0.008` | horizontal_recatch | Release-Recatch | SUCCESS_12S | Release-Recatch | SUCCESS_12S |
| `s7_p25_a+0.012_e0.000` | horizontal_recatch | Hybrid | SUCCESS_12S | Release-Recatch | SUCCESS_12S |
| `s7_p25_a+0.012_e0.008` | horizontal_recatch | Hybrid | SUCCESS_12S | Hybrid | SUCCESS_12S |
| `s7_p25_a-0.012_e0.000` | horizontal_recatch | Hybrid | SUCCESS_12S | Release-Recatch | SUCCESS_12S |
| `s7_p25_a-0.012_e0.008` | horizontal_recatch | Hybrid | SUCCESS_12S | Release-Recatch | SUCCESS_12S |
| `slide_quota_400058` | gravity_assisted_in_hand_sliding | Other/Unclear | TABLE_DROP | Hybrid | SUCCESS_12S |
| `slide_quota_400463` | gravity_assisted_in_hand_sliding,vertical_recatch | Hybrid | DROP_AFTER_HANDOFF | Hybrid | DROP_AFTER_HANDOFF |
| `slide_quota_400496` | gravity_assisted_in_hand_sliding | Release-Recatch | LOST_GRASP | Release-Recatch | SUCCESS_12S |
| `slide_quota_400656` | gravity_assisted_in_hand_sliding,vertical_recatch | Hybrid | SUCCESS_12S | Release-Recatch | SUCCESS_12S |

Machine-readable copy: `per_ic_comparison.csv`.

---

## 3. Observed facts

### Success and mechanism counts

| Checkpoint | SUCCESS_12S | A Contact-Preserving | B Release–Recatch | C Hybrid | D Other/Unclear |
|---|---:|---:|---:|---:|---:|
| SFT | **8 / 17** | 0 | 4 | 11 | 2 |
| GRPO Epoch 3 | **15 / 17** | 0 | 8 | 9 | 0 |

Successes by mechanism:

- SFT: Release–Recatch 3, Hybrid 5
- GRPO: Release–Recatch 7, Hybrid 8

### Mechanism transition (SFT → GRPO)

| SFT | GRPO | count |
|---|---|---:|
| Hybrid | Hybrid | 6 |
| Hybrid | Release–Recatch | 5 |
| Release–Recatch | Hybrid | 2 |
| Release–Recatch | Release–Recatch | 2 |
| Other/Unclear | Hybrid | 1 |
| Other/Unclear | Release–Recatch | 1 |

Source: `mechanism_transition.csv`.

### Result / mechanism changes

Fail → success under GRPO (7 ICs):

- `inward_extra_250051`, `inward_extra_250131`, `k48r_valid_00004`, `k48r_valid_00009`, `k48r_valid_00168`, `slide_quota_400058`, `slide_quota_400496`

Of those 7:

- **5** keep the same mechanism family and only improve the terminal outcome (mostly Hybrid→Hybrid or Release–Recatch→Release–Recatch).
- **2** also change mechanism code: `k48r_valid_00168` Hybrid→Release–Recatch; `slide_quota_400058` Other/Unclear→Hybrid.

Mechanism changed with the same success bit (7 ICs): `inward_extra_250043`, `k48r_valid_00036`, and several successful horizontal / overlapping vertical ICs listed in `mechanism_summary.json`.

No GRPO regression from SFT success to failure on this 17-IC set.

### Sliding-labeled ICs (unique snapshots)

Sliding label appears on overlapping categories; counts below are unique ICs that include `gravity_assisted_in_hand_sliding`.

Under GRPO: Release–Recatch **4**, Hybrid **7**, Contact-Preserving **0**.

### Horizontal Recatch ICs

All 6 horizontal ICs succeed under GRPO. Every one has a sustained both-off interval and is classified Release–Recatch or Hybrid. So their successes are contact-loss then regrasp, not contact-preserving sliding.

### Consistency with the prior single-IC audit

`k48r_valid_00009` + GRPO:

- Expected: Hybrid
- Observed: Hybrid
- Both-off: 0.02 → 0.06 s (duration 0.04 s)
- Post-recontact migration: 2.39 mm
- Result: SUCCESS_12S

Match with `results/diagnostics/checkpoint3_skill_audit/`.

---

## 4. Answers to the study questions

### Q1. Does GRPO Epoch 3 really use only Recatch or Hybrid?

**Fact:** On these 17 validation rollouts, GRPO mechanisms are only Release–Recatch (8) and Hybrid (9). Contact-Preserving Adjustment count is 0.

**Interpretation:** For this validation set, GRPO Epoch 3 is Recatch-family dominated. That matches the visual impression, and the contact timelines support it. It does **not** prove the policy never Contact-Preserves outside this list.

### Q2. Does SFT retain more Contact-Preserving / Sliding behavior?

**Fact:** SFT Contact-Preserving count is also **0** on these 17 ICs. SFT is already Hybrid (11) / Release–Recatch (4) / Other (2).

**Interpretation:** On this validation set, GRPO did not “replace Sliding with Recatch” relative to SFT, because SFT itself did not exhibit Contact-Preserving recoveries here.

### Q3. Do new GRPO successes come from better execution of the same skill, or from mechanism change?

**Fact:** 5 of 7 fail→success ICs keep the same mechanism label and only change the terminal outcome. 2 also change mechanism code.

**Interpretation:** Most new successes look like **better completion of an already Recatch-like / Hybrid pattern**, not discovery of a new Contact-Preserving family on the validation set.

### Q4. Is there an evidenced Sliding-only IC where GRPO fails because it prefers Recatch?

Evidence requires the same IC to have Sliding SUCCESS_12S **and** a recorded Recatch failure — not a Sliding tag alone.

Evidenced valid ICs: `inward_extra_250043`, `inward_extra_250051`, `inward_extra_250131`, `k48r_valid_00009`.

Among those, GRPO Recatch/Hybrid failure:

- `inward_extra_250043`: Sliding expert SUCCESS_12S (`inward_reseat`); vertical Recatch attempt `DROP_AFTER_HANDOFF`; GRPO `Release-Recatch` + `DROP_AFTER_HANDOFF`.

**Interpretation:** There is one evidenced IC where a Sliding expert works, an open-overchase Recatch attempt fails, and GRPO also fails while executing a Recatch-like contact pattern. SFT also failed on that IC (`TABLE_DROP`), so this is **not** a GRPO regression from a working SFT Contact-Preserving policy. It is a remaining hard IC, not proof that Recatch bias caused a loss of a previously successful learned Sliding policy on the validation set.

### Q5. Do the traces support at least two distinct physical recovery mechanisms?

**Fact:** Labels present across SFT+GRPO are Hybrid, Release–Recatch, and Other/Unclear. Contact-Preserving Adjustment never appears.

**Interpretation:**

- In the **narrow** sense, B vs C are two related Recatch-family subtypes (brief open–regrasp without vs with measurable post-recontact sliding).
- In the **strong** sense needed for “Sliding and Recatch as separate skills,” these 17 validation traces do **not** show a successful Contact-Preserving family beside Recatch/Hybrid.

---

## 5. Interpretation

GRPO Epoch 3 raises validation success from 8/17 to 15/17 under the same deterministic rule. The contact evidence says those recoveries are release–regrasp based, often with a short both-off near 0.04 s and sometimes a few millimeters of later contact migration (Hybrid).

Because SFT on the same ICs is already Recatch-family dominated, the main GRPO effect visible here is **more reliable completion of Recatch-like recoveries**, not a measured shift away from Contact-Preserving Sliding on the validation set.

---

## 6. Limitations

1. Both-off threshold 0.04 s equals two policy steps; many classifications are confidence=`borderline`. Under 0.06 s, several B/C labels would move toward D or Contact-Preserving-like if contact otherwise holds — see per-trace `sensitivity` in `mechanism_summary.json`.
2. Failed historical Recatch npz files are often missing; Sliding-success / Recatch-fail evidence uses manifests plus staging attempt logs.
3. Category tags overlap (Sliding ∩ Vertical). All counts use unique snapshot ICs.
4. Migration needs bilateral contacts; sparse or one-sided contact pushes episodes toward D.
5. Representative videos cover one example per GRPO mechanism class plus the consistency IC; they are illustrative, not a substitute for the CSVs.
6. This study does not evaluate training ICs.

---

## 7. Files

| Path | Content |
|---|---|
| `REPORT.md` | This report |
| `per_ic_comparison.csv` | 17-row SFT vs GRPO table |
| `mechanism_summary.json` | Counts, transitions, per-trajectory facts |
| `mechanism_transition.csv` | SFT→GRPO mechanism matrix |
| `traces/` | 34 CSV trajectories |
| `contact_events/` | 34 contact on/off event JSON files |
| `plots/` | 34 recovery contact plots |
| `videos/` | Representative GRPO videos |

---

## 8. Conclusion

**Bounded conclusion:** On the 17 validation ICs, Checkpoint 3 does **not** provide evidence of a second successful Contact-Preserving / Sliding recovery family beside Recatch-like behavior. Every GRPO rollout is Release–Recatch or Hybrid by the contact rules above, and SFT is already in that same family. The honest reading of these traces is that validation success is driven by a **Recatch-leaning / Hybrid contact pattern**, with Hybrid vs pure Release–Recatch as within-family variation, not as proof of multi-skill Sliding+Recatch unified control on this held-out set.
