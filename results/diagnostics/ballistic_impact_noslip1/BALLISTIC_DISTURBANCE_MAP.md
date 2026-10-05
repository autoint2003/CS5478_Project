# Ballistic disturbance map (noslip=1, m_ball=0.05 kg)

**Diagnostic only. No recovery / recatch / SAC. No MP4.**

## 11. USER VISUAL OBSERVATION: PENDING

Watch the interactive cases in §9 before treating mechanism labels as confirmed.

---

## 1. Corrected loss semantics

Legacy `dropped()` treats a single 2 ms `nL=nR=0` sample as drop. That is **not** used here.

Diagnostic events are kept separate:

- `first_left_off` / `first_right_off` / `first_both_off`
- `both_off_duration` (max consecutive; **includes post-table air time** — prefer the sustained timestamps)
- `bilateral_restored`
- `sustained_both_off_20/50/100ms`
- `object_exits_finger_span` (COM outside finger pads for ≥50 ms)
- `table_contact` (`obj_z < TABLE_DROP`)
- `physical_drop` (table, or span-exit with xy > 0.12 m)
- `end_state`: `held_12s` | `table_contact` | `exited_finger_span` | `truncated`

A 2 ms both-off is logged. It is never labelled DROP / FAILURE / IMMEDIATE LOSS.

---

## 2. Matched impact construction

Nominal FSM: APPROACH → DESCEND → CLOSE → active LIFT.

Release after captured + `obj_z ≥ Z_AIR` + 5 cm hand rise. Weld off. Free flight with **fixed** `T = 0.10 s`. Aim at predicted COM `p_obj + v_obj T` along `+hand-x`. Optional axial offset along the cylinder axis (object z). No teleport. No snapshot replacement at impact.

Cylinder r = 18 mm, half-height 30 mm. Commanded offsets: CENTER 0, UPPER +12 mm, LOWER −12 mm (18 mm remaining to the rim). Ball mass 0.05 kg. `noslip_iterations = 1`.

---

## 3. Central-impact coarse severity

Stopped expanding at 6 m/s after a progressive-instability **candidate**. Did not run 7 m/s.

| v (m/s) | TOF (s) | \|J\| (N·s) | \|r×J\| | Δv (m/s) | Δω (rad/s) | restore | table | end | mechanism |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 3.0 | 0.104 | 0.166 | 5.2e-4 | 0.069 | 0.090 | 2.858 | — | held_12s | IMPACT_DAMPED |
| 4.0 | 0.104 | 0.219 | 6.4e-4 | 0.104 | 0.158 | 2.856 | — | held_12s | IMPACT_REPOSITIONED_STABLE |
| 5.0 | 0.104 | 0.269 | 7.4e-4 | 0.157 | 0.124 | 2.858 | — | held_12s | IMPACT_REPOSITIONED_STABLE |
| 6.0 | 0.104 | 0.317 | 8.3e-4 | 0.235 | 0.406 | 2.860 | 6.132 | table_contact | IMPACT_INITIATED_SLIP |

All four share `t_release = 2.748`, `t_impact = 2.852`. 3–5 m/s: 2–4 ms both-off blips then restore and a full 12 s hold. 6 m/s: see §8.

---

## 4. Off-center impact construction

UPPER/LOWER were **defined** as ±12 mm along the cylinder axis at the predicted COM, to change `r_contact × J` without changing TOF.

**Not executed.** Search stopped when CENTER 6 m/s produced a C candidate.

Actual CENTER contact axial z stayed ≈ −2.1 mm (not an end hit). Radial contact still drifted 15.0 → 12.0 mm from 3 → 6 m/s because launch distance is `v T` with T fixed. More matched than the old `T=dist/v` aimer; not a perfectly identical contact point.

---

## 5. Impulse / angular impulse

`J` is integrated ball–cylinder force on the ball (`mj_contactForce`); object impulse is `−J`. Angular proxy is `r_contact × J_obj` with `r` from pre-impact COM to the first contact point. `Δv`, `Δω` are object twist from the last pre-impact sample to +20 ms.

| v | J_hand_x (N·s) | \|J\| | \|r×J\| | Δv | Δω |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.166 | 0.166 | 5.2e-4 | 0.069 | 0.090 |
| 4 | 0.219 | 0.219 | 6.4e-4 | 0.104 | 0.158 |
| 5 | 0.269 | 0.269 | 7.4e-4 | 0.157 | 0.124 |
| 6 | 0.316 | 0.317 | 8.3e-4 | 0.235 | 0.406 |

The CENTER family is mostly **linear** +hand-x impulse. Angular proxy stays small (~10⁻³). Do not read severity from launch speed alone; here |J| scales roughly with v, as intended.

---

## 6. Post-impact state trajectories

JSON `post_samples` at 0 / 20 / 50 / 100 / 200 / 500 / 1000 ms: `disturbance_map/center_v*.json`. Raw series: `disturbance_map/center_v*.npz`. Diagnostic plot: `disturbance_map/figures/post_impact_rh.png`.

CENTER 6 m/s after the collision transient (not the 2 ms blip):

| t − t_impact | rh.x (mm) | nL/nR | \|v_rel\| | obj z |
|---:|---:|---|---:|---:|
| 0 | 0.26 | 7/6 | 0.0004 | 0.490 |
| 0.20 | 10.62 | 2/2 | 0.0025 | 0.503 |
| 0.50 | 10.89 | 7/4 | 0.0008 | 0.526 |
| 1.00 | 11.28 | 2/4 | 0.0010 | 0.565 |
| 1.40 (lift done) | 11.71 | 2/4 | 0.0012 | 0.595 |
| 2.00 | 12.48 | 2/4 | 0.0021 | 0.602 |
| 2.50 | 13.28 | 2/4 | 0.0018 | 0.603 |
| 3.00 | 16.85 | 2/4 | 0.027 | 0.604 |
| 3.13 | 31.6 | 0/0 | 0.47 | 0.597 |
| 3.28 table | 81.9 | 0/0 | 1.83 | 0.440 |

---

## 7. Mechanism classification

Analysis labels only (not controller logic):

| run | mechanism | meaning here |
|---|---|---|
| 3.0 CENTER | IMPACT_DAMPED | small residual, 12 s hold |
| 4.0 CENTER | IMPACT_REPOSITIONED_STABLE | new offset, damps, 12 s hold |
| 5.0 CENTER | IMPACT_REPOSITIONED_STABLE | larger offset, still holds |
| 6.0 CENTER | IMPACT_INITIATED_SLIP | captured after hit; rh.x creeps for ~3 s; then table |

No DIRECT_EJECTION in this coarse CENTER ladder (search stopped at 6 m/s).

---

## 8. Progressive-instability candidate

**CENTER 6.0 m/s. STOP SEARCHING. Inspect in the viewer before any recovery.**

- mass 0.05 kg, v = 6.0 m/s, CENTER, commanded z-off 0 mm
- first contact cylinder-frame (−0.13, −12.00, −2.10) mm; not an end clip
- |J| = 0.317 N·s (mostly +hand-x); |r×J| = 8.3×10⁻⁴ N·m·s
- t_impact = 2.852 s during active LIFT
- 2 ms both-off at 2.854, bilateral restore at 2.860 — **not** the loss
- lift completes at 4.244 s **while still captured** (n≈2/4)
- deterioration: rh.x 10.6 → 13.3 mm over 0.2–2.5 s after impact with small |v_rel|
- genuine loss: runaway ~5.85 s, sustained both-off 5.984, span-exit 6.056, **table 6.132** (3.28 s after impact)
- files: `disturbance_map/center_v6p0.json`, `center_v6p0.npz`, `progressive_candidate.pkl`

This is a **numerical** C: collision transient, re-grasp, seconds of relative-offset growth under continued lift/hold, then table contact. **USER VISUAL OBSERVATION: PENDING.**

---

## 9. Interactive viewer commands

No MP4. Camera set once; then mouse rotate / pan / zoom. SPACE pause, `[` / `]` speed, R restart case. Overlay: case, t, FSM phase, nL/nR. Does not stop on a contact-count blip.

```text
python training/demo_ballistic_impact.py --case central_stable
python training/demo_ballistic_impact.py --case central_strong
python training/demo_ballistic_impact.py --case progressive

python training/demo_ballistic_impact.py --case center_v3p0
python training/demo_ballistic_impact.py --case center_v4p0
python training/demo_ballistic_impact.py --case center_v5p0
python training/demo_ballistic_impact.py --case center_v6p0
```

Aliases: `central_stable` → 3.0 m/s; `central_strong` → 5.0 m/s (still captured); `progressive` → 6.0 m/s candidate C.

Off-center was not mapped. Unlabelled viewer with the same aimer:

```text
python training/map_ballistic_disturbance.py --site UPPER --v 5
python training/map_ballistic_disturbance.py --site LOWER --v 5
```

---

## 10. Unresolved issues

- Candidate C is not visually confirmed.
- `both_off_duration_max` includes post-table samples; use `t_sustained_both_off_*`.
- Reported `tilt_deg` is near 180° because object z is anti-aligned with hand z. The 6 m/s failure is **rh.x creep**, not a large tilt.
- `J` omits finger forces during the hit.
- Finger-span exit is a COM/pad heuristic.
- CENTER radial contact still moves with speed (12–15 mm).
- Phase B (UPPER/LOWER) is incomplete because search stopped at C.

---

## 11. USER VISUAL OBSERVATION: PENDING
