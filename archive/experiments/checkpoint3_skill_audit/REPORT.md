# Checkpoint 3 skill audit

Primary IC: `k48r_valid_00009`. Checkpoint: `results/grpo/tree80/checkpoint_epoch_0003.pt`.

This report separates **direct physical facts**, **skill classification derived from those facts**, and **speculation that is not proven**.

Diagnostic script: `visualization/audit_checkpoint3_skill.py`. Outputs live under `results/diagnostics/checkpoint3_skill_audit/`.

## 0. Support check before the audit

| Need | Status |
|---|---|
| Exact same snapshot for Sliding success and Recatch attempt | Available: `data/snapshots/valid/k48r_valid_00009.pkl` |
| Sliding SUCCESS_12S after full 12 s nominal | Available in `data/controlled_sliding/manifest.jsonl` and confirmed by expert replay |
| Recatch failure on that IC | Available as attempt log; **failed trajectory npz was not saved** |
| Checkpoint 3 deterministic eval matching `--eval` | Available via `roll_policy` (argmax token, residual mean) |
| Per-step token / residual / legal action / contact CSV | Written by the diagnostic |
| Video | Written by the diagnostic |

No approximate IC was substituted. Physics, the nominal controller, the K48 codebook, the 39D observation, legal actions, `reward_v2`, `SUCCESS_12S`, and the checkpoint were not modified.

## 1. Matched Sliding-success / Recatch-fail IC

### Direct evidence

Sliding success on this exact IC:

- Manifest: `data/controlled_sliding/manifest.jsonl`
- Strategy `controlled_slide`, mechanism `contact_preserving_slide`
- Stored outcome: `reason=SUCCESS_12S`, `success=True`, `verified_success=True`
- Snapshot: `data/snapshots/valid/k48r_valid_00009.pkl`
- Continuous actions: `data/controlled_sliding/valid/k48r_valid_00009__controlled_slide__verified.npz`
- Recovery `2.500` s then nominal `12.000` s, `lift_completed=True`
- Also listed under `data/gravity_assisted_in_hand_sliding/manifest.jsonl` with the same snapshot and SUCCESS_12S

The label `gravity_assisted_in_hand_sliding` alone was **not** used to claim Sliding-only. The claim rests on a verified Sliding tape plus a recorded Recatch failure on the same id.

Recorded vertical Recatch failure on the same episode id:

- `data/staging/recovery_expansion/budget_attempts.jsonl`
- `strategy=vertical_recatch`, `reason=LOST_GRASP`, `success=False`, `t_off=0.02`, `t_pad=0.08`
- Not present in `data/vertical_recatch/manifest.jsonl`

### Gap

- The failed Recatch trajectory npz was not archived. Only the attempt log row exists.
- Therefore the historical failure cannot be re-inspected bit-for-bit. A live `follow_early` oracle replay on the same snapshot is the available substitute and is labeled as such below.

### Other exact matches (not the primary audit IC)

Same Sliding-success + vertical-Recatch-fail pattern on valid ICs in `budget_attempts.jsonl`:

- `inward_extra_250043` → vertical `DROP_AFTER_HANDOFF`
- `inward_extra_250051` → vertical `LOST_GRASP`
- `inward_extra_250131` → vertical `LOST_GRASP`
- `k48r_valid_00009` → vertical `LOST_GRASP` (this audit)

Train-side matches in `followup_attempts.jsonl`: `ball_train_v2_0012`, `0031`, `0036`, `0038`, `0077`.

Note: the older matched search `results/archive/demo_results/diagnostics/SLIDING_VS_REGRASP_MATCHED_SEARCH.md` compared Sliding vs release-regrasp on `ball_train_0001`–`0018` and found the **opposite** family ranking there (regrasp verified more often than strict sliding). That archive set is not this validation IC.

## 2. Checkpoint 3 deterministic evaluation

Action rule matches `envs/outcomes/horizon_outcome.roll_policy` / `scripts/train.py --eval`: argmax token, residual mean, legal decode, then the official 12 s nominal verify.

### Direct observations

| Source | Outcome |
|---|---|
| Official `roll_policy` | `SUCCESS_12S`, early handoff, recovery `0.74` s, nominal `12.00` s, lift completed |
| Traced closed loop in this audit | Same: `SUCCESS_12S`, early handoff |

Recovery tokens (before HANDOFF):

`[39, 39, 0, 23, 5, 23, 0, 33×19, 0×11]` then token `47` HANDOFF.

Codebook meanings used in the recovery stage:

- Token `39`: open-grip motion `[0.242, -0.030, 0.139, 0, 0, 0, -1]`
- Token `33`: downward close `[0, 0, -0.40, 0, 0, 0, +1]`
- Token `0`: closed hold / near-zero motion with secure grip channel

Early contact / grip timeline from `k48r_valid_00009__checkpoint3_trace.csv`:

| t_rel (s) | token | nL | nR | aperture (m) | tau |
|---:|---:|---:|---:|---:|---:|
| 0.00 | (pre) | 2 | 5 | 0.0170 | -18.0 |
| 0.02 | 39 | 0 | 0 | 0.0194 | +1.99 |
| 0.04 | 39 | 0 | 0 | 0.0245 | +2.00 |
| 0.06 | 0 | 10 | 0 | 0.0159 | -18.0 |
| 0.08 | 23 | 10 | 19 | 0.0142 | -18.0 |
| 0.10+ | 5/23/0/33… | >0 | >0 | ~0.017–0.018 | -18.0 |

Contact events (`k48r_valid_00009__checkpoint3_contact_events.json`):

- both-off starts at `t_rel=0.02`
- both-off ends at `t_rel=0.06` (duration `0.04` s)
- bilateral again at `t_rel=0.08`
- no later both-off before handoff

Measured recovery facts:

- recovery samples: 38
- bilateral samples: 35
- both-off intervals: one, duration `0.040` s
- max contact migration: `4.12` mm
- max object travel in hand frame: `13.1` mm
- post-recontact migration: `2.39` mm
- max aperture: `24.5` mm
- min tau after close: `-18.0`

Artifacts:

- `k48r_valid_00009__checkpoint3_trace.csv`
- `k48r_valid_00009__checkpoint3_contact_events.json`
- `k48r_valid_00009__checkpoint3.mp4`
- `summary.json`

## 3. Skill classification

**Classification: `Hybrid`**

### Rules used

- **Recatch**: original grasp is lost with sustained both-finger-off for at least `0.04` s (two policy steps), then bilateral contact is re-established before handoff. A single-frame gap is not enough.
- **Sliding**: contact is preserved on every recovery sample, with contact-point migration or in-hand object travel ≥ 1 mm.
- **Hybrid**: a qualifying Recatch phase is followed by continued bilateral contact with further in-hand adjustment.
- Horizontal / Vertical Recatch naming follows the original grasp scene. This IC is a vertical-cylinder Sliding scene, not a horizontal-recatch impact cell.

### Why this IC is Hybrid, not pure Recatch or pure Sliding

1. The initial bilateral grasp is lost for `0.04` s while token `39` opens the gripper (`tau → +2`).
2. Bilateral contact returns by `0.08` s.
3. After recontact, contact remains bilateral through handoff while the contact centroid still migrates by about `2.4` mm and the object moves about `13` mm in the hand frame under closed grip (`tau=-18`, mostly tokens `33` then `0`).

So Checkpoint 3 does execute a real, brief release-and-regrasp, then a contact-preserving settle. It is **not** pure contact-preserving Sliding, and it is **not** a long airborne Recatch chase.

### Borderline note (fact about the threshold)

The both-off duration equals the `0.04` s threshold exactly. Under a stricter threshold (for example `0.06` s) this episode would not count as Recatch and would fall to Unclear/Sliding-like. The chosen threshold is stated explicitly so the classification can be contested.

### Sliding expert on the same snapshot

- Replay: `SUCCESS_12S`
- Classification: **Sliding**
- Facts: 126/126 bilateral samples, **zero** both-off intervals, migration `3.3` mm
- Trace: `k48r_valid_00009__sliding_expert_trace.csv`

This confirms the stored Sliding demo really is contact-preserving on this IC. Checkpoint 3 does **not** reproduce that Sliding contact pattern.

### Live vertical oracle replay on the same snapshot

Fresh official `follow_early` law (`data_generation/strategies/vertical_recatch_search.py`), not the missing historical npz.

- Reaches mode `handoff` after release/recontact: `t_off=0.02`, `t_re=0.08`, `t_pad=0.08`
- Then fails the nominal verify with **`LOST_GRASP`**
- `success=False` under the same SUCCESS_12S / HELD_EXTENSION acceptance used by the vertical dataset builder

So the old Recatch law **does** lose and regain contact on this IC; the failure is after regrasp / during nominal continuation, not a refusal to open. That points more to **limited oracle coverage / unstable post-regrasp state under that law** than to “Recatch is physically impossible here.”

## 4. Answers to the audit questions

1. **Did Checkpoint 3 really Recatch?**  
   **Partially yes.** It opened, produced a sustained both-off of `0.04` s, then re-established a bilateral grasp. That is a real grasp loss and regrasp, not a video illusion. It is brief.

2. **Was it actually contact-preserving Sliding or a mix?**  
   **Hybrid.** After recontact it keeps bilateral contact and continues in-hand adjustment. It does not match the Sliding expert’s never-break contact pattern.

3. **If it Recatches, why succeed where the old Recatch oracle failed?**  
   **Fact:** live `follow_early` also releases and recontacts, then fails with `LOST_GRASP`; Checkpoint 3 hands off earlier (`0.74` s) and holds 12 s.  
   **Not proven:** that every Recatch law fails, or that Recatch is universally easy on Sliding-labeled ICs. The documented oracle used a small open-overchase grid. Search-coverage limits remain a live explanation.

## 5. Interpretation boundaries

### Fact

- `k48r_valid_00009` has a verified Sliding SUCCESS_12S tape and a logged vertical Recatch failure on the same id and snapshot.
- Checkpoint 3 reaches SUCCESS_12S under the official deterministic eval rule.
- Checkpoint 3 opens (`token 39`, `tau≈+2`), loses both contacts for `0.04` s, then regrasps and holds bilaterally until early handoff.

### Classification

- Label **Hybrid** from the contact rules above.
- Video appearance and gripper motion alone were not used as the decision.

### Speculation / not proven

- That Checkpoint 3 “always Recatches” on all 17 validation ICs.
- That Recatch is generally feasible whenever Sliding works.
- Bit-for-bit identity between the historical failed Recatch attempt and today’s live `follow_early` replay.

## 6. Files

| File | Role |
|---|---|
| `REPORT.md` | This report |
| `summary.json` | Machine-readable summary |
| `k48r_valid_00009__checkpoint3_trace.csv` | Per-step Checkpoint 3 recovery + nominal trace |
| `k48r_valid_00009__checkpoint3_contact_events.json` | Contact on/off events |
| `k48r_valid_00009__sliding_expert_trace.csv` | Sliding expert replay |
| `k48r_valid_00009__vertical_oracle_replay_trace.csv` | Live follow_early contact timeline |
| `k48r_valid_00009__vertical_oracle_contact_events.json` | Live oracle contact events |
| `k48r_valid_00009__checkpoint3.mp4` | Checkpoint 3 video |
| `visualization/audit_checkpoint3_skill.py` | Reproducible diagnostic |
