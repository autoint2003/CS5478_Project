# K49 Action-Authority Ablation

**Config baseline:** `K49_tok48_wy0p25` checksum `33d4820d1ec09c8b`  
**Physics / controller / SUCCESS_NEW unchanged. No SFT/GRPO.**  
**Condition B:** continuous `a∈[-1,1]^7` via `step_legal` (AIRBORNE_V3) — supported.

## Budget

| Item | Value |
|------|------:|
| Total rollouts | 2800 |
| Wall limit (s) | 3000 |
| Stage split | 40% diagnosis / 60% extension |
| Used | 2701 |

## Q1 — Legal continuous action dimensions and bounds

See `LEGAL_ACTION_INTERFACE_AUDIT.md`. Summary: 7-D normalized action clipped to **[-1,1]**, mapped by AIRBORNE_V3 to hand-frame linear velocity (vx,vy +/-1.0 m/s; vz asymmetric 3.8/1.0), r_des angular +/-4.0 rad/s, grip torque [2.0,-18.0] N*m. Execution period **20 ms**.

## Q2 — Missing / coarse in K49

- Grip: only ~3 levels (-1 / -0.48 / +1); mid-grip modulation missing.
- Fine lateral translation (much smaller than typical token |a_xy|) underrepresented for COM nudges.
- Reverse wy mainly via tok48; soft braking combinations sparse.
- No explicit short “secure hold while micro-translate” prototypes.

## Stage 1 paired results

### `hard_910012` (pose)

| Condition | SUCCESS_NEW | best win pos (mm) | notes |
|-----------|------------:|------------------:|-------|
| K49 | 0 | None | baseline |
| Continuous | 0 | 32.99903401292261 | CEM after capture prefix |

Continuous pose improvement vs K49: **False**

### High-speed capture (paired ICs)

Continuous capture improvement signal: **False**  
(details in `raw/STAGE1_BEST.json`, `CAPTURE_BRAKE_METRICS.csv`)

## Stage 2 — Candidate tokens

Proposed new IDs start at **49** (0–48 preserved, HANDOFF=47).  
Count: **16**. Useful (success or improves flag): **1**.

Smallest useful set: [{"token_id": "63", "name": "fine_down", "intended": "small +vz (hand-frame)"}]

## FULL SUCCESS_NEW counts

| Condition | FULL count |
|-----------|----------:|
| K49 baseline search rows | 0 |
| Continuous CEM rows | 0 |
| Extended-vocab verified | 2 ({'C_fine_trans': 1, 'E_combined': 1}) |

K49 regression (0–48 decode unchanged): prefix+return on hard_910012 reason=TABLE_DROP NEW=0

## Q3–Q12

3. Continuous improves high-speed capture? **False** (SUCCESS_NEW gains: False).
4. Continuous CEM improves contact-preserving pose correction? **False under Stage-1 budget** (best continuous pos ~33 mm). However a **minimal discrete fine-+vz token** later unlocked SUCCESS_NEW (pos ~2.1 mm) — so the gap was partly **missing coarse-to-fine discrete options / search**, not only continuous authority.
5. `hard_910012` SUCCESS_NEW with continuous CEM? **False**. With K49+tok63 (`fine_down`)? **True** (replay-verified).
6. High-speed failures: continuous matched K49 (no FULL) → primarily **physical/controller capture under high v_rel**, not fixed by denser discretization of the same bounds. Pose failure on `hard_910012` was closer to a **discretization/search** gap for gentle COM correction.
7. Candidate tokens with measurable benefit: **['fine_down']**.
8. Smallest useful extension: **1 tokens** (['fine_down']) — empty means do not force a larger vocab.
9. Extended vocab preserves original K49 IDs/semantics? **Yes** (append-only; regression decode 0–48 unchanged).
10. New FULL per condition: K49 rows=0, Continuous=0, Ext=2.
11. Robustness across roots: **1** independent FULL roots in extension verify.
12. Recommendation: **`adopt_minimal_K_extension`**.

## Scripts
`scripts/run_k49_action_authority_ablation.py`


## Verified SUCCESS_NEW (exact continuous-legal replay)

- IC: `hard_910012`
- Conditions: `C_fine_trans`, `E_combined` (identical token sequence)
- New token used: **63 `fine_down`** = `[0, 0, 0.15, 0, 0, 0, 1]` (small +vz, secure grip)
- Sequence: capture prefix (open+t17+hold+brake) + tok63 x10 + tok48 x16
- Replay metrics: theta=0.49801442634043425 deg, pos=2.1264927296041196 mm, hold=2.0200000000006746 s
- Matched K49/continuous Stage1 on same IC: **0 SUCCESS_NEW** (continuous best pos ~33 mm)

### Implication

A **single** fine vertical translation token (ID 63) unlocks SUCCESS_NEW on a previously blocked stable-grasp pose IC under matched prefix. High-speed capture ICs still show **0 FULL** under extension. Smallest useful extension for pose: **{63: fine_down}**; do not inflate vocabulary without more roots.
