# Dynamic throw / recapture authority

No SAC. No observation/reward/detector change. No teleport. No object qvel injection. No MP4.
No action-bound expansion. Cartesian controller not retuned.

## 1. Mechanism hypothesis

Maintain secure contact (`tau=-18`) while commanding legal `omega_y=+/-3` rad/s, transfer momentum through finger-object contact, then actively open (`ctrl[7]=+2`) near a coarse orientation so the cylinder enters a short ballistic arc (ideally `v_obj,z>0`) and later re-enters a reachable capture region.

This is distinct from (1) controlled slip with retained contact and (2) drop-then-chase after unconstrained free fall.

High-clearance parent (physical lift): obj_z=0.5583 m, virtual table clearance **12.8 cm**. Gravity=[0.0, 0.0, -9.81]. Mass=0.2000 kg.
Table geom `table` mask 1/1 -> 0/0. Object and floor unchanged. Official scene.xml unmodified.

## 2. Legal rotation response

Commanded `omega_y` bound = 3.0 rad/s (`RECOVERY4D_W_HY_MAX`). `w_world = r_des @ [0, omega_y, 0]`.

### sign=1

| deg cmd | ang act | nL/nR | |w_hand| | w_hand | v_obj | v_obj,z | carry_err | rho |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.0 | 7/6 | 0.056 | [0.004, 0.056, 0.0] | [-0.007, 0.0, 0.026] | +0.0259 | 0.0017 | 0.22960314440811821 |
| 30 | 29.3 | 8/11 | 2.874 | [2.874, -0.019, 0.015] | [0.008, 0.266, 0.16] | +0.1601 | 0.0892 | 0.2161067450861243 |
| 60 | 59.3 | 6/6 | 2.987 | [2.986, -0.054, 0.054] | [-0.011, 0.09, 0.276] | +0.2765 | 0.0943 | 0.20132177583372968 |
| 90 | 89.5 | 4/4 | 2.992 | [2.989, -0.077, 0.085] | [-0.03, -0.077, 0.293] | +0.2935 | 0.0949 | 0.3968689095006794 |
| 120 | 119.4 | 4/4 | 2.926 | [2.923, -0.093, 0.073] | [-0.037, -0.207, 0.206] | +0.2064 | 0.0918 | 0.7447155650514032 |

### sign=-1

| deg cmd | ang act | nL/nR | |w_hand| | w_hand | v_obj | v_obj,z | carry_err | rho |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.0 | 7/6 | 0.056 | [0.004, 0.056, 0.0] | [-0.007, 0.0, 0.026] | +0.0259 | 0.0017 | 0.22960314440811821 |
| 30 | -29.5 | 9/11 | 2.870 | [-2.87, -0.007, -0.024] | [0.009, -0.263, 0.167] | +0.1670 | 0.0960 | 0.19883525283660383 |
| 60 | -59.4 | 4/6 | 2.982 | [-2.981, -0.044, -0.051] | [-0.004, -0.087, 0.263] | +0.2631 | 0.1013 | 0.378181586001787 |
| 90 | -89.3 | 4/4 | 2.987 | [-2.986, -0.073, -0.078] | [-0.027, 0.072, 0.279] | +0.2787 | 0.1025 | 0.47657097234230356 |
| 120 | -119.4 | 3/4 | 2.920 | [-2.917, -0.094, -0.068] | [-0.038, 0.202, 0.2] | +0.2001 | 0.0996 | 0.8773968179634957 |

## 3. Pre-release velocity audit

At each probe, immediately before opening: object world/hand velocity, hand twist, COM lever arm, Fn/Ft/rho, relative slip.

- target 60.0 deg sign=1 actual=59.3: v_obj=[-0.011153312740348175, 0.08987645339944182, 0.27647835600212617] v_obj,z=0.27647835600212617 v_obj_h=[0.28369622003144634, -0.007055846873389899, -0.06410245208377538] v_hand=[-0.00922465922261099, -0.0013616554667291725, 0.09084778323338884] w_hand=[2.9855622043588244, -0.054162555869420426, 0.05407423326255972] r_world=[0.003175890120289515, 0.08591460611986629, -0.051274832202888754] nL/nR=6/6 Fn=8.991367256104866/9.164176617075814 Ft=0.7751349605175271/1.504310620846011 rho=0.20132177583372968 v_rel=0.2068497564908314 carry_err=0.09430580036140662

## 4. Release-angle probe

Legal open `ctrl[7]=2.0`. Both-off persistence >= 0.018s. No intra-angle timing search.

| target | actual | sign | lost-pre | v_obj,z at FIRST_BOTH_OFF | |w_obj| | upward |
|---|---|---|---|---|---|---|
| 60.0 | 59.3 | 1 | False | 0.2166045819025304 | 3.043 | True |

## 5. Upward-velocity evidence

**YES** at first coarse hit: target 60.0 deg, actual 59.3, sign=1, v_obj,z=0.2166 m/s.

PRIVILEGED construction only. Not optimized.

## 6. Ballistic trajectory

After FIRST_BOTH_OFF, no recatch for 0.8s. z0=0.61705519296276 z_max=0.6171776204541753 up_disp=0.00012242749141533338 t_apex=5.607999999999604.
See `raw/throw_ballistic.npz`.

## 7. Matched no-rotation release

Open at ~0 deg, same grip law. v_obj,z at both-off=-0.04687635190376251. upward=False. ballistic up_disp=0.0.
Causal claim requires rotating release velocity to differ materially from this baseline.

## 8. Privileged recapture construction

**PRIVILEGED DYNAMIC RECAPTURE CONSTRUCTION.** Close timing uses GT (both-off plus a short apex wait), not an observable policy.
After both-off the catcher parks (`omega_y=0`), then `tau=-18`. After brief bilateral arrest, a legal return toward the parent orientation is attempted before the long hold.
captured=False ok_hold=False
Event times: ROTATE_START=5.163999999999653, RELEASE_COMMAND=5.573999999999608, FIRST_BOTH_OFF=5.583999999999607, BALLISTIC_APEX=5.607999999999604, RECATCH_PREP=5.607999999999604, CLOSE_START=5.607999999999604, FIRST_RECONTACT=5.623999999999603, FIRST_BILATERAL=5.625999999999602, MOTION_ARREST=5.677999999999597, RETURN_START=5.677999999999597, RETURN_LOST=6.067999999999554
Recontact and a short bilateral arrest occurred. Sustained secure hold did not. That is recapture/hold failure, not absence of a throw. Likely limiter: residual relative velocity plus gravity along hand-x at ~60 deg. Not optimized further.

## 9. Causal baselines

A: rotate+release+no reclose produces an upward ballistic sample (probe log).
B: rotate+release+reclose: captured=False ok_hold=False.
C: no-rotation + same open: v_obj,z=-0.04687635190376251 (not an upward throw).

## 10. Stable-hold evidence

No sustained airborne recapture hold in this construction.

## 11. CENTER-6 transfer

Not run. Clean throw-recapture mechanism did not succeed, so transfer is not claimed.

## 12. Limitations

THROW MECHANISM YES, STABLE RECAPTURE NOT ESTABLISHED: legal omega_y while secure produced v_obj,z=0.2166 m/s at FIRST_BOTH_OFF (no-rotation baseline v_z negative). Privileged close achieved recontact/brief bilateral arrest but not a sustained airborne hold. Bounds not expanded. CENTER-6 transfer not run.

- Coarse angles and both signs only. No throw optimization.
- Legal bounds unchanged: |omega_y|<=3.0, |v_x|<=0.08, |v_z|<=0.08, tau_open=2.0.
- Table collision disabled after parent; floor remains.

## Viewer

```text
python training/demo_dynamic_recatch.py --mode throw_only
python training/demo_dynamic_recatch.py --mode throw_recatch
python training/demo_dynamic_recatch.py --mode sweep
```

SPACE pause, R restart, `[` `]` speed. Camera initialized once. No MP4.

---

---

# OFFSET-ASSISTED RELEASE-PHASE AUDIT

Wrist keeps rotating during OPEN. Physical release = FIRST_BOTH_OFF, not OPEN_COMMAND. Same `omega_y=+3` and legal open for centered vs offset. No recatch tuning.

## 1. Exact initial offset-state provenance

File: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\ballistic_recovery_transfer\raw\snap_early.pkl` (same pickle as the validated large-angle gravity-recovery EARLY state).
t=3.322 s, r_h mm=[10.87, -0.02, 98.33], documented r_h.x=10.87 mm, match=True.
nL/nR=7/4, aperture=0.0176, phase=lift.
v_rel_h=[0.0006468302137243188, -0.00014019083103779953, -3.850944907106674e-05], w_rel_h=[-0.0029624145694306095, 0.00041599759322282013, -0.04234693598495139].
p_des=[0.4950805525479236, 0.0018408558876627172, 0.6224455596740429], v_cmd=[0.0, 0.0, 0.0], ctrl7=-18.0.
No qpos/qvel edits. Real CENTER-6 r_h.x sign only.

## 2. Centered vs offset constrained swing (tau=-18, no open)

| deg | c v_z | o v_z | c rhx mm | o rhx mm | c lever | o lever | c rho | o rho | c n | o n |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.02586716971369732 | 0.07663877995876277 | 0.2036324206718957 | 10.870572177265952 | 0.09671480447536941 | 0.0989286051495656 | 0.22960314440811821 | 0.24887190269618947 | 7/6 | 7/4 |
| 30 | 0.16371457434187509 | 0.18565974316164274 | 0.10069732979028537 | 10.996433242983118 | 0.09824739019698256 | 0.10042076323212983 | 0.21675065401924307 | 0.1539140748327262 | 8/9 | 4/4 |
| 60 | 0.2780311691181791 | 0.28923874025154267 | -0.2720534639783523 | 10.841594304379138 | 0.10014642103976236 | 0.10223894930716636 | 0.21294143226278386 | 0.16413523581599487 | 6/6 | 4/2 |
| 90 | 0.29306321853524475 | 0.30001649838372935 | -0.8336694088068052 | 10.443872660644523 | 0.10208410894757337 | 0.10412259164184683 | 0.3989224812598221 | 0.20021667694356626 | 4/4 | 2/2 |
| 120 | 0.20346939733560745 | 0.1971897308025923 | -1.3372036827727312 | 10.043943775440056 | 0.10411956590883167 | 0.10607113897430243 | 0.7760453046317446 | 0.17890418667767138 | 4/4 | 4/2 |

Peak bilateral v_obj,z: centered ang=79.19256541988769 vz=0.2998627402239251; offset ang=78.02339875219882 vz=0.309340176185751.

The CENTER-6 offset is **+10.87 mm in r_h.x** (squeeze axis). Wrist rotation is about hand/r_des **y**, so the dominant lever is **r_h.z ~ 99 mm**, not r_h.x. Lever-arm magnitudes stay within ~2 mm of the centered grasp at matched angles. Constrained-swing v_obj,z curves therefore nearly overlay. Offset is not a larger flywheel; it is a slightly asymmetric contact topology (nR often lower: 4 vs 6 at 0 deg; last contact is R in every open trial).

## 3. OPEN_COMMAND -> FIRST_BOTH_OFF latency

| case | theta_cmd | theta_cmd_act | theta_single | theta_BOTH_OFF | latency s |
|---|---|---|---|---|---|
| centered_early | 15.0 | 14.870308534864343 | 16.097755220225693 | 16.407049997534767 | 0.009999999999998899 |
| centered_medium | 45.0 | 44.63678838218563 | 45.65275822406211 | 46.33190440277087 | 0.009999999999998899 |
| centered_late | 90.0 | 89.83445086406877 | 91.55108530079076 | 91.89537361729435 | 0.011999999999998678 |
| offset_early | 15.0 | 14.866531830621573 | 16.091731846897986 | 16.400458302818745 | 0.009999999999998899 |
| offset_medium | 45.0 | 44.87722142081591 | 46.22900911650341 | 46.90718239671002 | 0.011999999999998678 |
| offset_late | 90.0 | 89.70496902133203 | 91.42296766815211 | 91.76751457669407 | 0.011999999999998678 |

## 4-6. FIRST_BOTH_OFF velocity/pose and ballistic

| case | init rhx mm | theta_off | v_obj,z | v_obj | |w_obj| | |w_hand| | rh_off mm | last | dt_apex | rise m | sep_max | corridor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| centered_early | 0.20 | 16.407049997534767 | 0.006168184066937688 | [0.011252332809186898, 0.3311351460969281, 0.006168184066937688] | 2.691 | 2.700 | [0.1, -0.07, 97.81] | R | None | 0.0 | 0.641890151452681 | False |
| centered_medium | 0.20 | 46.33190440277087 | 0.16923183018171548 | [0.004614351163398441, 0.1744222446407959, 0.16923183018171548] | 3.035 | 2.965 | [-0.29, -0.1, 99.55] | R | None | 0.0 | 0.6450734854858086 | False |
| centered_late | 0.20 | 91.89537361729435 | 0.21404261873788089 | [-0.029684271536245612, -0.0806412930887584, 0.21404261873788089] | 3.068 | 3.005 | [-1.35, -0.08, 102.25] | R | 0.0039999999999995595 | 0.00011061047495153353 | 0.6114014644890001 | False |
| offset_early | 10.87 | 16.400458302818745 | 0.04446530118094856 | [0.008590372616031032, 0.3281497486919493, 0.04446530118094856] | 2.670 | 2.696 | [10.9, -0.07, 99.38] | R | None | 0.0 | 0.615652709459728 | False |
| offset_medium | 10.87 | 46.90718239671002 | 0.17157128934415047 | [0.007985559487588072, 0.16456436180655962, 0.17157128934415047] | 3.057 | 2.960 | [10.63, -0.15, 101.25] | R | None | 0.0 | 0.6100523251407157 | False |
| offset_late | 10.87 | 91.76751457669407 | 0.22256136814325508 | [-0.026856018258371212, -0.10827441300593696, 0.22256136814325508] | 3.036 | 3.007 | [9.95, -0.12, 103.76] | R | 0.005999999999999339 | 0.00015816820885961036 | 0.5827482396080689 | False |

## 7. Late-open vs early-open (offset)

EARLY: cmd 15.0 deg -> BOTH_OFF 16.400458302818745 deg, v_z=0.04446530118094856, latency=0.009999999999998899 s.
LATE: cmd 90.0 deg -> BOTH_OFF 91.76751457669407 deg, v_z=0.22256136814325508, latency=0.011999999999998678 s.
Phase effect Delta v_z (early-late) = -0.1781 m/s.

Opening with `ctrl[7]=+2` while the wrist keeps spinning has **~10-12 ms** latency (~2 deg at 3 rad/s). FIRST_BOTH_OFF is therefore almost the command angle, not a large lagged angle. **Release phase still dominates kinematics:** commanding open at 15 deg releases at **16.4 deg** with v_obj,z=+0.044; commanding at 90 deg releases at **91.8 deg** with v_obj,z=+0.223. Stopping the wrist before opening is still the wrong mechanism, but "open at 90" is nearly "release at 92" with this grip range.

## 8. Privileged recatch

Not run.

## 9. Limitations

offset_early vz=0.04446530118094856 BOTH_OFF=16.400458302818745 deg (OPEN cmd 15); offset_late vz=0.22256136814325508 BOTH_OFF=91.76751457669407 deg (OPEN cmd 90); centered_early vz=0.006168184066937688 BOTH_OFF=16.407049997534767 deg. offset_better=False phase_matters=True recatch_ok=None.

Coarse early/medium/late only. Bounds unchanged. No synthetic offset. No MP4.

```text
python training/demo_dynamic_recatch.py --mode centered_throw
python training/demo_dynamic_recatch.py --mode offset_late_open
python training/demo_dynamic_recatch.py --mode offset_early_open
python training/demo_dynamic_recatch.py --mode offset_throw_recatch
```
