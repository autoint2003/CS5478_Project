# Gravitational reposition self-arrest (s=2.0, noslip=1)

Privileged diagnostic on the same `S_FAIL` snapshot as the construction (`s=2.0`, mass 0.20 kg, pair μ=1.0, noslip=1). RULE, SAC, reward, observation, noslip, and disturbance family were not changed.

**USER VISUAL OBSERVATION**

- Simple CONTROLLED SLIP: CONFIRMED (prior).
- `--mode self_arrest`: **PENDING** until the user watches it.

---

## 1. Exact NO_BRAKE semantics

Do not interpret the name. Measured `ctrl[7]` matches the commanded τ in `raw/no_brake_ctrl_trace.json`. Source: `run_full(..., do_slip=True, do_brake=False)`.

| Phase | duration | wrist | p_des | r_des | v_cmd | τ / ctrl[7] |
|-------|----------|-------|-------|-------|-------|-------------|
| OBSERVE | 0.10 s | ~0° | freeze-here | freeze-here | 0 | **−18** |
| ROTATE | ~0.44 s | +hand-y, ω=1.2 rad/s | freeze translation | integrate ω | 0 | **−18** |
| freeze() | instant | snap current pose into p_des / r_des | | | 0 | last |
| SLIP | **fixed 1.50 s** (not a saturation detector) | held ~+30° | frozen | frozen | 0 | **−4** (live ctrl[7]=−4) |
| RETURN | ~0.52 s, ω=1.0 rad/s | back to 0° | freeze translation | integrate −ω | 0 | **−4** |
| VERT_HOLD | 2.00 s | ~0° | freeze-here | freeze-here | 0 | **−18** (`TAU_SEC` hardcoded) |
| LIFT + HOLD10 | 2.25 + 10 s | 0° | p_des := p_hand, then +z | frozen ori | (0,0,0.08) then 0 | **−18** |

Measured unique phases at slip-start: wrist 29.72°, τ_cmd=−18, ctrl[7]=−18, v_cmd=w_cmd=0. During SLIP: wrist 29.86°, τ_cmd=−4, **ctrl[7]=−4**, v_cmd=0, w_cmd=0, noslip=1.

**What NO_BRAKE actually skips:** the tilted 0.5 s τ=−18 dwell.

**What it does not skip:** restoring τ=−18 at vertical hold and for the entire nominal continuation.

**Return in NO_BRAKE is a weak-grip return (τ=−4).** Slip is a 1.5 s timeout, not “wait until motion stops.”

---

## 2. Continuous slip-decay trajectory

Wrist frozen at +30°, τ=−4, 3.00 s (`raw/slip_tau4.npz`, `figures/fig_slip_decay.png`). û = (−1, 0, 0) so v_along = −v_x. Useful progress = decrease of r_h.x.

Initial sample after weakening: |v_rel| ≈ 85 mm/s (grip-change transient; finite-Δr_h is much smaller). That spike decays in ~80 ms.

| t from slip (s) | r_h.x mm | r_h.z mm | v_along mm/s | \|v_rel\| mm/s | ρ_max |
|-----------------|----------|----------|--------------|----------------|-------|
| 0.00 | 8.97 | 99.46 | −82.5 | 85.0 | 0.94 |
| 0.25 | 8.66 | 100.00 | +9.3 | 10.2 | 0.52 |
| 0.50 | 8.41 | 100.42 | +4.2 | 4.9 | 0.52 |
| 0.75 | 8.17 | 100.83 | +1.9 | 2.8 | 0.50 |
| 1.00 | 7.94 | 101.24 | +0.99 | 1.94 | 0.50 |
| 1.50 | 7.49 | 102.05 | +0.88 | 1.83 | 0.50 |
| 2.00 | 7.03 | 102.87 | +0.96 | 1.96 | 0.50 |
| 2.75 | 6.35 | 104.08 | +0.91 | 1.85 | 0.51 |

Diagnostic “saturation” as |v_along| < 1 mm/s for 150 ms: **t = 1.00 s**, useful x ≈ **1.03 mm**. Stricter 0.5 mm/s and 0.2 mm/s **never** trigger in 3 s.

After t=1 s the fast transient is over, but motion does **not** stop. Residual ≈ **0.9 mm/s in −x** and ≈ **1.6 mm/s in +z**, ρ stuck near 0.50, bilateral 10/10. Peak useful x in the 3 s window is **2.85 mm**, not 1.2 mm.

The construction’s “saturates near 1.2 mm over 1.2 s” was the **end of the fast transient plus a short observation window**, not a true arrest.

---

## 3. Force / contact / geometry evolution

Wrist-frame gravity is essentially constant (wrist frozen):

g_h(start) ≈ `[−4.87, −0.13, 8.52]`, g_h(sat sample) ≈ `[−4.80, −0.10, 8.55]`, **g_h · û ≈ 4.81–4.87 N/kg** (does not collapse).

| sample | τ | r_h.x mm | nL/nR | Fn_L / Fn_R | Ft_L / Ft_R | ρ_L / ρ_R | centroid_L.x mm | v_rel_h mm/s |
|--------|---|----------|-------|-------------|-------------|-----------|-----------------|--------------|
| 0% (pre-weaken mix) | −18 | 8.97 | 6/6 | 8.95 / 9.01 | 2.81 / 0.98 | 0.37 / 0.13 | 7.40 | [+82, +2, −21] |
| 25% (~0.7 mm) | −4 | 8.27 | 10/10 | **1.99 / 2.01** | 0.99 / 0.98 | **0.50 / 0.50** | 7.65 | [−2.6, +0.8, +2.1] |
| “sat” (~1.0 mm) | −4 | 7.95 | 10/10 | 1.99 / 2.01 | 0.98 / 0.98 | 0.50 / 0.50 | 7.45 | [−1.0, +0.4, +1.6] |
| 50% (~1.4 mm) | −4 | 7.55 | 9/10 | 1.99 / 2.01 | 0.94 / 1.02 | 0.48 / 0.51 | 7.39 | [−0.86, 0, +1.6] |
| 75% (~2.1 mm) | −4 | 6.84 | 10/10 | 1.99 / 2.01 | 0.98 / 0.99 | 0.49 / 0.49 | 6.79 | [−0.90, 0, +1.6] |
| 100% (~2.8 mm) | −4 | 6.12 | 9/10 | 1.99 / 2.01 | 0.95 / 1.02 | 0.48 / 0.51 | 6.27 | ~1 mm/s −x |

Cylinder axis in the hand stays `[−0.147, ~0, 0.989]` (~8.4°) at every sample. Contact centroids travel with r_h.x; pad span in z grows to the pad height (~17 mm). Fn under τ=−4 is ~2 N/finger (vs ~9 N at τ=−18). ρ remains ~0.5 (sliding), not a stick collapse.

Full contact lists: `raw/evolution_samples.json`. Snapshots: `raw/slip_start.*`, `raw/saturation.*` (the 1 s “sat” state, still τ=−4), `raw/pre_return_nobrake.*` (legacy 1.5 s), `raw/post_active_brake.*`.

---

## 4. H1–H4

**H1 gravity alignment — contradicted.** Wrist is frozen. g_h and g_h·û change by ~1%. The driving tangential gravity component does not go away.

**H2 increased frictional support — contradicted for the weak-grip “arrest.”** Under τ=−4, ρ stays ~0.50 and Fn stays ~2 N while residual sliding continues. ρ **does** fall 0.50 → 0.23 when τ is restored to −18 (that is the brake / secure-grip effect, not self-arrest).

**H3 geometric contact lock — contradicted as a hard stop.** Topology stays bilateral. Centroids track the object along −x / +z. Cylinder orientation does not jam. Contacts follow the slip; they do not block it.

**H4 noslip numerical freeze — contradicted.** Residual |v_rel| ≈ 1.8–2.0 mm/s persists for seconds. noslip was not toggled. A NoSlip hard-friction lock would not keep a steady ~1 mm/s along û.

**What actually happens:** τ=−4 at +30° puts both pads near ρ≈0.5 with low Fn. After an 80 ms transient, the object **crawls** along the pads at ~1 mm/s (−x) and ~1.6 mm/s (+z) for as long as the weak grip and tilt remain. The 1.2 mm figure was a short-window observation of that crawl, not an equilibrium.

---

## 5. 5 s tilted weak-grip hold from the 1 s “saturated” snapshot

Wrist held ~+29°, **τ=−4 the whole time**, ctrl[7] end = **−4**, noslip=1, no return.

| | start | +5 s |
|--|-------|------|
| r_h mm | [7.94, 0, 101.24] | [3.33, 0, 109.48] |
| Δr_h mm | — | **[−4.61, 0, +8.24]** |
| \|v_rel\| | 1.94 mm/s | 1.94 mm/s |
| ρ | 0.50 | 0.52 |
| nL/nR | 10/10 | 10/10 |
| drop | no | no |

Mean v_along over the 5 s: **+1.02 mm/s**. This is a **genuine continued crawl**, not a long-lived tilted equilibrium. The 1 s “saturation” was transient.

`raw/hold5.json`, `raw/hold5_tau4.npz`.

---

## 6. Weak vs secure 2 s hold from the identical saturated snapshot

No return.

| | A τ=−4 | B τ=−18 |
|--|--------|---------|
| ctrl[7] end | −4 | −18 |
| Δr_h mm | **[−1.80, 0, +3.26]** | **[−0.16, 0, −0.03]** |
| \|v\| start → end | 1.94 → 1.94 mm/s | 1.94 → **0.076 mm/s** |
| Fn_L/R | 2.0 / 2.0 → 2.0 / 2.0 | 2.0 / 2.0 → **9.0 / 9.0** |
| ρ | 0.50 → 0.51 | 0.50 → **0.23** |
| nL/nR | 10/10 → 9/10 | 10/10 → 10/10 |

**Restoring τ=−18 after the crawl is what actually brakes.** It kills residual velocity, raises Fn, and drops ρ into the stick band. Staying at τ=−4 does not.

---

## 7. WEAK_RETURN vs SECURE_RETURN vs BRAKE_DWELL_RETURN

Same saturation snapshot. After return: **2 s vertical hold, no lift yet.**

| | WEAK_RETURN | SECURE_RETURN | BRAKE_DWELL_RETURN |
|--|-------------|---------------|--------------------|
| return τ | −4 | −18 immediately | −18, 0.5 s tilt dwell, then −18 |
| vertical-hold τ | **−4** | −18 | −18 |
| after 2 s nL/nR | 10/10 | 10/10 | 10/10 |
| e_x mm | 7.62 | 7.97 | 7.95 |
| e_z mm | **+7.54** | +2.79 | +2.77 |
| ρ | **0.50** | 0.23 | 0.23 |
| \|v_rel\| | **1.84 mm/s** | 0.05 mm/s | 0.05 mm/s |
| ctrl[7] | −4 | −18 | −18 |

SECURE_RETURN and BRAKE_DWELL_RETURN are the same mechanical state to measurement precision. WEAK_RETURN is still sliding at vertical (especially +z).

---

## 8. Task continuation

Same nominal lift (0.08 m/s, Δz=0.18 m) + 10 s hold. `continue_nominal` **always** uses τ=−18.

| branch | lift | drop | 10 s hold | end nL/nR | end e_x | end obj_z |
|--------|------|------|-----------|-----------|---------|-----------|
| ZERO | yes | **11.81 s** | no | 0/0 | 162 mm | 0.35 m |
| ROTATE_ONLY | yes | **10.17 s** | no | 0/0 | 24 mm | 0.42 m |
| SLIP_SELF_ARREST | no lift (stay +30°, τ=−4) | no drop in 5 s | — | 10/10 | crawl −4.6 mm x | still in hand |
| WEAK_RETURN | yes | no | **yes** | 6/6 | 8.24 mm | 0.62 m |
| SECURE_RETURN | yes | no | **yes** | 8/9 | 8.54 mm | 0.62 m |
| BRAKE_DWELL_RETURN | yes | no | **yes** | 7/9 | 8.52 mm | 0.62 m |

WEAK_RETURN survives **because lift restores τ=−18**, not because weak vertical hold was stable. That is the same hidden secure restore as legacy NO_BRAKE.

---

## 9. Does brake dwell have causal value?

On this S_FAIL:

- **Tilted 0.5 s dwell vs immediate τ=−18 return:** no task-level difference, and the post-vertical states match.
- **τ=−18 vs staying at τ=−4:** large state-level difference. Only τ=−18 stops the crawl (velocity, ρ, Fn).
- **Legacy FULL vs NO_BRAKE** both survived because both eventually command τ=−18; NO_BRAKE also returned while weak, which this IC tolerated.

**Do not keep BRAKE dwell as a mandatory FSM state from this evidence.**

A better description of the mechanism that actually worked:

1. secure rotate (+30°, τ=−18)
2. controlled weak-grip gravitational crawl (τ=−4) for a useful Δr_h
3. **restore secure grip (τ=−18)** — this is the real brake
4. return vertical at τ=−18
5. nominal continuation at τ=−18

An extra 0.5 s tilted dwell after (3) is optional on this IC.

The 1.2 mm “self-arrest” should **not** be used as a stop condition: at τ=−4 the object keeps crawling. Brake timing should be a **relative-geometry trigger**, then τ→−18, not “wait until v≈0 under weak grip.”

---

## 10. Viewer command

```text
python training/demo_grav_reposition_recovery.py --mode self_arrest
```

Sequence: S_FAIL → rotate +30° at τ=−18 → τ=−4 at +30° for **6 s**. τ does **not** return to −18. One-shot camera, then mouse-owned. Overlay: phase, wrist, τ, progress, v_rel, ρ, nL/nR. Hand-fixed ghost / ruler / trail.

You should see a fast initial relative motion, then a **slow ongoing crawl**, not a perfectly frozen pose. That is the physical result.

---

## 11. USER VISUAL OBSERVATION: pending

---

## Local s perturbation (after the mechanism)

τ=−4 was **not** retuned.

| s | ZERO class | τ=−4 1.5 s |
|---|------------|------------|
| 1.9 | **SURVIVE** (not a delayed-fail IC) | not used as a fail test |
| 2.0 | DELAYED_FAIL, drop 11.81 s | CONTROLLED_SLIP, 1.48 mm, 10/10, not LOSS |
| 2.1 | DELAYED_FAIL, drop 4.78 s | CONTROLLED_SLIP, 1.87 mm, 10/10, not LOSS |

On nearby delayed-fail s=2.1, τ=−4 is still a bounded crawl, not an abrupt LOSS/STICK switch. s=1.9 is not in the failure class, so it is not a recovery benchmark.

---

## Snapshots (not synthesized)

- `raw/slip_start.pkl/.json` — +30°, immediately before τ weakening, live noslip=1
- `raw/saturation.pkl/.json` — 1 s after weaken (fast transient over, crawl remains)
- `raw/pre_return_nobrake.pkl/.json` — 1.5 s slip (legacy NO_BRAKE pre-return)
- `raw/post_active_brake.pkl/.json` — 1 mm + 0.5 s τ=−18
