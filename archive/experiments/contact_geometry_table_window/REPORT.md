# Contact Geometry, Impulse Ejection & Table-Window Diagnostics

**Baseline:** K49 `K49_tok48_wy0p25` checksum `33d4820d1ec09c8b`  
**Interface:** legal AIRBORNE_V3 `a∈[-1,1]^7` / 20 ms. No SFT/GRPO.  
**Diagnostic protocols** (early-onset, table C2) are labeled and **not** original-task experts.

## Budget

| Item | Value |
|------|------:|
| Total rollouts | 1800 |
| Wall (s) | 2400 |
| Used | see `raw/USAGE.json` |

## Phase A summary

| Metric | Value |
|--------|------:|
| Audited (ic×seed) | 60 |
| Dominant early site counts | {"B_outer_finger": 52, "C_fingertip_edge": 8} |
| Fraction outside intended grip (B/C/D dominant) | 1.00 |
| Fraction COM not between fingers (early) | 0.00 |
| Wrong-geom at major impulse | 1.00 |
| Outward ejection events (audit) | 0.73 |
| Onset already contacting | 1.00 |
| Mean time onset→irreversible (s) | 0.228 |

Ejection with wrong-geom: n=44; ejection with high grip_cmd: n=31.

## Matched alignment (Exp A)

| Condition | n | eject rate | mean bil persist | SUCCESS_NEW |
|-----------|--:|----------:|-----------------:|------------:|
| A2_delay_long | 8 | 1.00 | 0.018 | 0 |
| A2_k49_fd | 8 | 1.00 | 0.083 | 1 |
| A2_open_fd | 8 | 1.00 | 0.018 | 0 |
| A2_open_long | 8 | 1.00 | 0.018 | 0 |
| A2_soft_long | 8 | 1.00 | 0.025 | 0 |
| A_k49 | 8 | 1.00 | 0.095 | 0 |
| B_open_align | 8 | 1.00 | 0.018 | 0 |
| C_wrist_align | 8 | 1.00 | 0.018 | 0 |
| D_translate_in | 8 | 1.00 | 0.018 | 0 |
| E_delay_close | 8 | 1.00 | 0.018 | 0 |
| F_soft_then_hard | 8 | 1.00 | 0.025 | 0 |

## Force/timing (Exp B)

n=1562; eject rate=1.00; bil persist=0.018; SUCCESS_NEW=0.

## Early onset (diagnostic)

| Onset | n | eject | bil persist | SUCCESS_NEW (diag flag) |
|-------|--:|------:|------------:|------------------------:|
| original IC | 12 | 1.00 | 0.020 | 0 |
| early diagnostic | 18 | 0.33 | 0.000 | 0 |

## Table counterfactual

| Condition | n | eject | bil persist | SUCCESS_NEW / diag |
|-----------|--:|------:|------------:|-------------------:|
| C1_original | 6 | 1.00 | 0.020 | 0/0 |
| C1_original_open_fd | 6 | 1.00 | 0.020 | 0/0 |
| C2_lower_table | 6 | 1.00 | 0.020 | 0/0 |
| C2_lower_table_open_fd | 6 | 1.00 | 0.020 | 0/0 |
| C2_no_table_contact | 6 | 1.00 | 0.020 | 0/0 |
| C2_no_table_contact_open_fd | 6 | 1.00 | 0.020 | 0/0 |

## Original-task FULL SUCCESS_NEW

Count: **1** — ICs: ['hard_910012']

## Q1–Q12

1. **Yes (with nuance).** Dominant early sites are **100%** outer finger (52/60) or fingertip edge (8/60). COM often still projects between jaws, but pad contact is wrong-side/tip — do not use contact counts alone.
2. **Yes, frequently.** Audit outward-ejection rate **0.73**; major hand impulse often already at Recovery t=0 with positive `J_outward` and `dv_outward` (table impulses separated in CSVs).
3. **Not the sole cause.** Ejections occur with open grip as well (grip_cmd<0 present in eject events). Premature closure can worsen outcomes but **wrong-geometry impact already ejects**.
4. **Not from the current Recovery IC.** Open/delay/soft alignment from the official IC did **not** reduce eject rate (still ~1.0) — the snapshot is already mid-bad-contact.
5. Mean onset→irreversible **~0.23 s**; **100%** of audited seeds already have hand–object contact at Recovery onset.
6. **Yes for HIGH_MOTION set:** Recovery often starts after (or during) the ejecting contact; the useful pre-impact alignment window is largely missed.
7. **Partially (diagnostic).** Early-onset eject rate **0.33** vs original **1.00**, but bil persist stayed ~0 and no diagnostic SUCCESS_NEW — earlier action reduces impact severity, not yet a full catch.
8. **No material unlock.** C2 lower-table / no-table-contact matched C1 eject≈1.0; table shortens the clock but removing it does not by itself enable capture under these actions.
9. Surviving original-env improvement: **`hard_910012` + continuous `fine_down` pose path** (`A2_k49_fd`) — pose correction after capture, **not** an ejection fix for ball-impact ICs.
10. **1** original SUCCESS_NEW (`hard_910012` / `A2_k49_fd`); **0** HIGH_MOTION ejection ICs solved.
11. Dominant: **wrong-geometry contact + outward hand impulse**, then **late Recovery onset**; table clearance secondary; `fine_down` action representation for pose-near only.
12. **Next step:** mine/branch **pre-outer-impact** states (or open→break-contact→re-enter jaw) under unchanged table/physics; keep early-onset as a separate labeled protocol; **do not** start SFT/GRPO.

## Scripts
`scripts/run_contact_geometry_table_window.py`, `scripts/_tmp_continue_contact_geometry.py`

## Hypothesis verdicts (post-continuation)

- Wrong-geometry first contact: **supported** (outer/tip sites; COM-between ≠ valid pad grasp).
- Hand-impulse ejection: **supported** (0.73 audit rate; often at IC onset).
- Premature closure as primary: **not supported alone** (open-grip ejections exist).
- Post-IC alignment prevents ejection: **not supported** on these snapshots.
- Table as primary unlock: **not supported** (C2 ≈ C1).
- Early onset helps impact severity: **partially supported** (eject 1.0→0.33, diagnostic).
- Original SUCCESS_NEW from ejection fix: **no**; only pose `fine_down` on `hard_910012`.

### Most defensible next step

Prioritize **pre-Recovery contact geometry** (avoid outer/tip first impact) over larger chase motions. Keep original table for acceptance. No SFT/GRPO yet.
