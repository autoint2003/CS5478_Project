# Remaining Unsupported / Unverified Primitives

After the flowchart restore (spec + gap audit + ST/DX2/fallback repairs), these items
are **not** claimed fixed. Prefer diagnosis over inventing substitute mechanisms.

| Item | Evidence | Status |
|---|---|---|
| Plan A dtheta settle bilateral loss | Path `01` on `ball_train_v2_0070`: `dtheta:contact_loss` before DOWN restore | Unresolved pivot-settle fragility |
| Cold upright inward dz (no prior dx S1) | Path `04` on Plan B upright: may fail `lateral_ex_too_large` or UP `contact_loss` | Partially mitigated post-dx Case C; cold start unresolved |
| Force-backed `one_sided_mechanical` at pure fingertips-DOWN | `Fn_z≈0` by geometry at DOWN | By design: DX1 designates side; DX2 verifies under gravity |
| DX1 failure on specific ICs (e.g. historical hard_910026) | Prior diagnosis: ST fail then silent verified_lateral | Silent fallback **removed**; ST may still fail closed with explicit reason |
| dx-only then DOWN without dz | Path `02`/`08`: ST works; `SUCCESS_12S=True` but `SUCCESS_NEW=False` | HANDOFF tightness — not ST mechanism failure |
| Plan C postlift gbi_063 | Historically inward contact_loss; path bonus may vary | Do not weaken criteria; mark case-by-case |

## Explicitly repaired (do not regress)

- Silent `correct_dx_verified` after support-transfer failure (default off).
- DX2 bilateral-only abort contradicting S1 (`support_guard`).
- DX2 exit accepting “any bilateral” as S1 (now requires force-backed S1).
- DX1 missing `support_side` designation.
