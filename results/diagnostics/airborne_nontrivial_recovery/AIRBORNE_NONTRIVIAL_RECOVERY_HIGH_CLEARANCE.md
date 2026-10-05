# High-clearance / table-decoupled airborne recapture

No SAC. No reward/obs/detector change. No teleport. No MP4. No Cartesian retune. No bound expansion.

## OLD RESULT (low-altitude, table-limited, inconclusive)

See `AIRBORNE_NONTRIVIAL_RECOVERY.md`. That FAILURE STOP only showed that the official low-altitude construction reaches the table (~108 ms after both-off) before a meaningful lateral airborne mismatch can develop. It does **not** establish an airborne recapture authority gap.

## NEW RESULT (this file)

AIRBORNE RECOVERY AUTHORITY GAP: CLOSE_ONLY can fail after genuine airborne release (typically object already below the finger volume, not a table hit), but legal |v_x|≤0.08 / |v_z|≤0.08 did not intercept and recapture. Limiter candidates: vertical following (object |v_z| ~ g t exceeds 0.08 m/s within ~8 ms; finger-height window tens of ms) and Cartesian tracking of commanded v_x (realized world speed << 0.08 m/s, so a catchable lateral corridor miss never forms). Closing speed and rotation are secondary. Bounds were not expanded. Controller was not retuned.

## 1. Scientific question

Can legal 4D `(ω_y, v_x, v_z, a_grip)` intercept and recapture a freely moving airborne cylinder after genuine contact loss when CLOSE_ONLY fails because of **airborne relative misalignment**, with the table removed as a terminating confounder?

## 2. High-clearance parent (physical lift, no teleport)

Official `assets/scene.xml` is **not** modified. Diagnostic runtime only.

Sequence: APPROACH → DESCEND → CLOSE → LIFT (nominal FSM to the usual airborne parent), then legal world `v_z=+0.08` m/s with `τ=-18` continues lifting the securely grasped cylinder.

| | low-alt parent (old) | high-clearance parent |
|---|---|---|
| obj_z m | 0.4962 | 0.5583 |
| virtual table clearance m | 0.0658 | 0.1276 |
| nL/nR |  | 7/6 |
| extra lift duration s |  | 1.050 |
| gravity |  | [0.0, 0.0, -9.81] |
| object mass kg |  | 0.2000 |

Achieved virtual object–table clearance: **12.8 cm** (target 10–15 cm).

## 3. Collision-mask audit (object–table only)

After the high parent is established, **only the table geom** mask is zeroed. Object, floor, and finger geoms are unchanged. Finger–object collision, friction, arm dynamics, and actuators are unchanged.

| field | value |
|---|---|
| geom | `table` id=1 |
| table contype/conaffinity before | 1 / 1 |
| table contype/conaffinity after | 0 / 0 |
| object geom (unchanged) | (1, 1) |
| floor geom (unchanged) | (1, 1) |

Full named-geom dump: `high_clearance/collision_mask_audit.json`.

Virtual geometric table clearance is still logged (`clear` = cylinder min-z − TABLE_TOP). Crossing `clear=0` is **not** a physical contact in this diagnostic.

## 4. Release (legal open, no qpos/qvel edits)

`ctrl[7]=+2` (`a3=-1`) until sustained `nL=nR=0` (≥18 ms). Then legal `v_x=+0.08` m/s while open. Object pose evolves only through MuJoCo.

Coarse mismatch durations (s): 0.040, 0.055, 0.080, 0.120, 0.160. First modest CLOSE_ONLY failure is frozen. No optimization.

- t_mis=0.04: rhx=-0.33 mm, rhz=0.117 m, Δhand_x=8.003398211930968e-05, Δhand_y=0.0004071593249951002, Δobj_x=-0.00019462951684645624, dt_off=0.05599999999999383, vclear=0.10900053374958585, A=secure_hold, B=secure_hold, C=secure_hold, C+vz=None
- t_mis=0.055: rhx=-0.79 mm, rhz=0.128 m, Δhand_x=0.00013034626033509022, Δhand_y=0.0007895484012575288, Δobj_x=-0.00027248132358503874, dt_off=0.07199999999999207, vclear=0.09807436793980745, A=miss, B=miss, C=miss, C+vz=miss
- t_mis=0.08: rhx=-1.74 mm, rhz=0.149 m, Δhand_x=0.0002156288038862586, Δhand_y=0.0015726010459333182, Δobj_x=-0.0003892590336929125, dt_off=0.09599999999998943, vclear=0.07697614674251163, A=miss, B=miss, C=miss, C+vz=miss
- t_mis=0.12: rhx=-3.94 mm, rhz=0.198 m, Δhand_x=0.00036915908824824983, Δhand_y=0.0033791348370024003, Δobj_x=-0.0005838885505393687, dt_off=0.13599999999998502, vclear=0.029254933112306925, A=miss, B=miss, C=miss, C+vz=miss
- t_mis=0.16: rhx=-6.82 mm, rhz=0.261 m, Δhand_x=0.0005218872039448041, Δhand_y=0.005727395998903169, Δobj_x=-0.000778518067385825, dt_off=0.17599999999998062, vclear=-0.03416377176354812, A=miss, B=miss, C=miss, C+vz=miss

## 5. Frozen airborne state at recovery onset

No matched A-fail / C-succeed existence. First CLOSE_ONLY failure snapshot is saved under `high_clearance/mismatch_snap.pkl` when available.

| onset (first CLOSE_ONLY fail) | |
|---|---|
| r_h mm | [-0.79, -1.46, 128.02] |
| v_rel_h m/s | [-0.02961017817732722, -0.036064337695121136, 0.7677701384798034] |
| ω_rel_h rad/s | [0.08968436996937142, -0.015977998242765436, 0.028000481925891135] |
| R_rel | [[-0.00021912062357195837, 0.9999931783964564, -0.0036871597064866077], [0.9999910219081523, 0.00023472047696364678, 0.004230958447313916], [0.004231795037277729, -0.0036861995125748327, -0.999984251797953]] |
| aperture mm | 41.4 |
| nL/nR | 0/0 |
| obj_z / virtual clear m | 0.5286 / 0.0981 |
| time since both-off | 0.0720 s |
| Δhand_x / Δhand_y / Δobj_x m | 0.0001 / 0.0007895484012575288 / -0.0003 |
| v_cmd | [-6.620429771815669e-05, 0.07999986697866651, 0.0001300014869638788] |
| v_obj | [-0.004865737921152897, -0.0002285220220387271, -0.7531963519037623] |

AIRBORNE RECOVERY AUTHORITY GAP: CLOSE_ONLY can fail after genuine airborne release (typically object already below the finger volume, not a table hit), but legal |v_x|≤0.08 / |v_z|≤0.08 did not intercept and recapture. Limiter candidates: vertical following (object |v_z| ~ g t exceeds 0.08 m/s within ~8 ms; finger-height window tens of ms) and Cartesian tracking of commanded v_x (realized world speed << 0.08 m/s, so a catchable lateral corridor miss never forms). Closing speed and rotation are secondary. Bounds were not expanded. Controller was not retuned.

### Limiter classification

1. **Vertical following authority:** legal `|v_z|≤0.08` m/s cannot match free-fall `v_z≈-gt` after ~8 ms. Finger-height window is tens of milliseconds. At the first CLOSE_ONLY miss, `r_h.z` has already grown well past the nominal ~0.099 m pinch depth; fingers close on empty space (`CLOSED_EMPTY`). `v_x+v_z+CLOSE` does not restore bilateral capture.
2. **Lateral hand authority / Cartesian tracking:** commanded `v_x=+0.08` maps into world **y** at this grasp (`v_cmd ≈ [0, 0.08, 0]`). Realized `Δhand_y` is millimetres over 40–160 ms, far below `0.08·Δt`. A catchable **beside-the-cylinder** corridor miss never forms while the object is still at finger height.
3. **Closing speed:** secondary; fingers do reach a small aperture (`CLOSED_EMPTY`) with no contacts.
4. **Object rotation:** `ω_rel` at onset is small (~0.1 rad/s); not the limiter.
5. **Not the table:** object–table collision is off. Virtual `clear` is logged. Floor remains enabled and is a later terminator, not the CLOSE_ONLY miss mechanism.

Raw matched logs: `high_clearance/close_only.npz`, `wrong_sign.npz`, `correct_sign.npz`, `correct_vx_vz.npz`.

## 10. Privileged information

- nL/nR for both-off / recontact timing
- sign(r_h.x) to choose correct vs wrong v_x
- object pose/twist for logs
Not a policy observation.

## 11. Limitations

AIRBORNE RECOVERY AUTHORITY GAP: CLOSE_ONLY can fail after genuine airborne release (typically object already below the finger volume, not a table hit), but legal |v_x|≤0.08 / |v_z|≤0.08 did not intercept and recapture. Limiter candidates: vertical following (object |v_z| ~ g t exceeds 0.08 m/s within ~8 ms; finger-height window tens of ms) and Cartesian tracking of commanded v_x (realized world speed << 0.08 m/s, so a catchable lateral corridor miss never forms). Closing speed and rotation are secondary. Bounds were not expanded. Controller was not retuned.

- One existence (or gap) proof. Not optimized. Not learned.
- Floor collision remains enabled; floor hits are reported separately from table.
- `|v_x|,|v_z|≤0.08`. Jacobian gains untouched.

## 12. Viewer

```text
python training/demo_airborne_recovery.py --mode airborne_close_only
python training/demo_airborne_recovery.py --mode airborne_intercept
python training/demo_airborne_recovery.py --mode high_clearance
```

SPACE pause, R restart, `[` `]` speed. No MP4.