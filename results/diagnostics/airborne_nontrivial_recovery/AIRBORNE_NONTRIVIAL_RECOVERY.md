# Non-trivial airborne recovery authority

No SAC. No observation/reward change. No ballistic impact. No object teleport. No MP4.
Any success below is a **PRIVILEGED NON-TRIVIAL AUTHORITY CONSTRUCTION**, not an observable or learned policy.

## 1. Why previous recatch was trivial

The earlier clean recapture (`results/diagnostics/airborne_recapture/`) is a **TRIVIAL AIRBORNE RECAPTURE / CONTACT-AUTHORITY PROOF**:

centered airborne grasp → active open → ~38 ms both-off → cylinder falls approximately vertically → **close in place with v_z=0** → bilateral hold.

That proved opening, genuine release, closing, and recontact. It did **not** require spatial correction: the object stayed in the original capture corridor. Those logs are not rewritten.

## 2. Physical construction of the non-trivial state

Nominal APPROACH→DESCEND→CLOSE→LIFT parent. Legal `a3=-1` (`ctrl[7]=+2`) opens until both-off ≥10 ms. Then legal `v_x=+0.08` m/s (hand-frame x) while fingers stay open. Object qpos/qvel are integrated only. Recovery-onset snapshot is taken at the end of that free-flight slew (no extra park). Close uses τ=-18, with v_x=0 (CLOSE_ONLY) or ±legal v_x (ablations) during closure.

Legal bounds: `|v_x|≤0.08` m/s, `|v_z|≤0.08` m/s, `RECOVERY4D_TAU_OPEN=2.0`. RULE yaml `tau_open: -1` unchanged.

- t_mis=0.02: rhx=0.07 mm, Δhand_x=-9.056696743614534e-05, Δobj_x=-7.309185059423129e-05, A=secure_hold, B=secure_hold, C=secure_hold
- t_mis=0.03: rhx=-0.10 mm, Δhand_x=-0.00013564202324678476, Δobj_x=-0.00010963777589134693, A=secure_hold, B=secure_hold, C=secure_hold
- t_mis=0.04: rhx=-0.32 mm, Δhand_x=-0.000180068131949207, Δobj_x=-0.00014618370118846258, A=secure_hold, B=secure_hold, C=secure_hold
- t_mis=0.055: rhx=-0.78 mm, Δhand_x=-0.0002488436674785155, Δobj_x=-0.0002046571816638476, A=table_before_bilateral, B=table_before_bilateral, C=table_before_bilateral
- t_mis=0.07: rhx=-1.30 mm, Δhand_x=-0.0003056770364750494, Δobj_x=-0.0002558214770798095, A=table_before_bilateral, B=table_before_bilateral, C=table_before_bilateral

## 3. Exact state at recovery onset

FAILURE STOP: by the time CLOSE_ONLY fails, legal v_x correction did not restore a stable capture in this coarse set (or CLOSE_ONLY never failed before table).

Last state where CLOSE_ONLY still recaptures (still trivial laterally):

| | |
|---|---|
| t_mis | 0.04 s |
| r_h.xyz mm | [-0.32, -0.26, 111.59] |
| v_rel_h | [-0.021682143531355752, -0.010217382231635356, 0.5167041908742669] |
| aperture mm | 30.6 |
| obj_z / clear m | 0.485 / 0.055 |
| time since both-off | 0.048 s |
| Δhand_x / Δobj_x mm | -0.18 / -0.15 |
| commanded v_x | 0.08 m/s |
| mean world v_hand,x during mismatch | -0.004519729561677654 m/s |

First state where CLOSE_ONLY fails (table, not a lateral miss):

| | |
|---|---|
| t_mis | 0.055 s |
| r_h.xyz mm | [-0.78, -0.44, 121.21] |
| v_rel_h | [-0.0296414320060367, -0.013467279812593503, 0.6665036871930953] |
| obj_z / clear m | 0.476 / 0.046 |
| time since both-off | 0.064 s |
| Δhand_x / Δobj_x mm | -0.25 / -0.20 |
| mean world v_hand,x | -0.004484640701178344 m/s |

## 4. CLOSE_ONLY baseline

For t_mis≤0.04 s, CLOSE_ONLY still produces bilateral capture and motion arrest — the mismatch is only tenths of a millimetre in r_h.x. For t_mis≥0.055 s, CLOSE_ONLY hits the **table before bilateral contact**. Failure mode is vertical reachability, not a missed capture corridor.

## 5. Wrong-sign ablation

Wrong-sign legal v_x does not save the long-mismatch cases (also table_before_bilateral). It is not informative as a causal lateral test because the object never left the original corridor by a mechanically relevant amount.

## 6. Correct-sign spatial recovery

Correct-sign legal v_x + CLOSE **also** tables on every CLOSE_ONLY-failure member of the coarse set. There is no matched-state example in this search where A fails and C succeeds.

## 7. Actual relative hand/object correction

Commanded v_x = +0.08 m/s. Realized world Δhand_x over 20–70 ms is **0.09–0.31 mm**. Δobj_x is almost the same. r_h.x stays well under 2 mm until the table window is already gone. Cartesian tracking at this grasp pose does not deliver the legal v_x in the available ~50 ms.

## 8. Vertical reachability

Both-off ~4.124 s; table historically ~4.232 s (~108 ms). t_mis=0.04 leaves ~48 ms of free flight before CLOSE and still catches (|r_h.x|~0.3 mm). t_mis=0.055 leaves ~64 ms then CLOSE and tables. Legal |v_z|≤0.08 m/s cannot buy a long lateral-slew interval.

## 9. v_z ablation

Not a useful expander here: the missing ingredient is lateral displacement of the capture corridor, not extra downward chase. No v_z search.

## 10. Moving-target escalation

Not performed. Static lateral non-trivial recovery was not established, so a moving-target test would not be interpretable.

## 11. Long-hold result

No non-trivial recovery hold. Trivial CLOSE_ONLY holds at t_mis≤0.04 are the same class as the previous contact-authority recapture.

## 12. Privileged information used

- nL/nR for both-off / recontact labels and close timing after the correction interval
- r_h.x **sign** to choose correct vs wrong v_x
- object pose/twist for logging and onset tables
None of these is claimed as a policy observation.

## 13. Limitations

FAILURE STOP: by the time CLOSE_ONLY fails, legal v_x correction did not restore a stable capture in this coarse set (or CLOSE_ONLY never failed before table).

- Not learned, not observable, not optimized, not general airborne recovery.
- Legal v_x/v_z bounds were not expanded.
- Previous 27D observability audit is unchanged.

## 14. Viewer commands

```text
python training/demo_airborne_recovery.py --mode close_only
python training/demo_airborne_recovery.py --mode lateral_recovery
```

SPACE pause, R restart, `[` `]` speed. Camera initialized once. No MP4.
Headless: `python training/demo_airborne_recovery.py --mode sweep`