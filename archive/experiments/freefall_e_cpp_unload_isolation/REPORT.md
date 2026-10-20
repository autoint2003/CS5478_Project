# Matched CPP Unload Failure Isolation

**Script:** `scripts/run_freefall_e_cpp_unload_isolation.py`  
**Out:** `results/diagnostics/freefall_e_cpp_unload_isolation/`  
**Recipe:** `cpp_hrv1_920031` (documented tilt6x20 -> matched pre-unload snapshot -> branches)  
**ICs:** 6 representative recovery failures from `freefall_e_stable_lift_contact_manip`

No SFT/GRPO. Physics / SUCCESS_NEW / ICs unchanged. Privileged diagnostic only.  
Does **not** claim full 6-stage recovery or SUCCESS_NEW.

---

## Setup

1. Replay frozen-E stable-lift prefix (from prior matched snapshots).
2. Run **documented zero-residual** tilt (same decode as verified CPP pilots).
3. Save full MuJoCo + FSM capture at tilt-end (or last bilateral if tilt ends without bil).
4. Fork matched branches A/B/B_prior/C/D/E from that capture (no retarget, no controller reset).

| Branch | What it does |
|--------|----------------|
| A | Hold end-of-tilt wrist/hand, grip=+1 (no unload) |
| B | Documented CPP unload tok12 (a6~-0.48, **zero residual**) + brake |
| B_prior | Residual-carry unload as in `freefall_e_stable_lift_contact_manip` (a6 -> ~-0.73) |
| C | Milder continuous unload a6=-0.15 |
| D | Ramp a6 +1 -> -0.48 over 2x unload steps |
| E | Grip locked +1; leave-tilt wrist then brake |

Per-step logs: bilateral, contact positions, fn/ft, grip actuator force, s_axial, v_rel/w_rel, pose, commands, latch/drop times (`traces/`).

---

## Headline counts (6 ICs)

| Branch | Retained bilateral (all steps) | Lost bil | Table latch |
|--------|-------------------------------:|---------:|------------:|
| A | 0/6 | 5/6 | 4/6 |
| B (documented mid) | **2/6** | 3/6 | 3/6 |
| B_prior (residual-carry) | **0/6** | **6/6** | **6/6** |
| C (mild) | **3/6** | 2/6 | 3/6 |
| D (slow ramp) | **3/6** | 2/6 | 3/6 |
| E (grip held) | **3/6** | 2/6 | 3/6 |

Details: `MATCHED_COMPARISON.csv`.

---

## Failure order and mechanisms

### 1. Prior transfer used a more aggressive unload than verified CPP

Verified pilots decode each token with **zero residual** -> tok12 is exactly a6~-0.48.  
Prior script accumulated residual from tilt (a6=+1, wy=+1) so unload became a6~-0.23 then **-0.71/-0.73** with leftover wrist.

**B_prior loses bilateral on 6/6**, usually within 40-60 ms, then latches. This alone explains the uniform `LATCH_DURING_RECOVERY` in the previous round.

### 2. Documented mid-unload (B) can still kill contact via normal-force collapse

Example `distal_off01_h8` (near-SN distal case):

| t (s) | Branch | bil | fn_tot | a6 | aper | note |
|------:|--------|----:|-------:|---:|-----:|------|
| 0.02 | A | 1 | 17.9 | +1.00 | 0.018 | hold |
| 0.02 | B | 1 | **3.3** | -0.48 | 0.018 | fn collapses immediately |
| 0.12 | B | **0** | 1.5 | -0.48 | 0.018 | first lose bil |
| 0.02 | C | 1 | **7.2** | -0.15 | 0.018 | mild unload keeps force |
| all | C/D/E | 1 | ~6-18 | | | **retain** |

Order: **grip command opens -> normal force collapses -> bilateral lost** (not a target teleport).  
Aperture barely changes at loss (~0.018); this is force/friction loss under mid-open, not a large jaw gape.

On `distal_off01_h4` and `perp_off01_h7_hz`, **B retains** under zero-residual mid-unload. Mid-open is not universally fatal; it is state-dependent.

### 3. Pre-unload states are not quiescent (relative motion)

Pre-unload `v_rel` is **0.23-0.95 m/s** across ICs (see `snapshots/*__pre_unload_meta.json`).  
Hard-recovery CPP successes started from much quieter bilateral holds.  
Branch A (no unload) still eventually loses bil or latches on most ICs -> **tilt-end E states are fragile**; unload is not the only risk.

`perp_off01_h0` / `h2` / `h1`: all branches (including A/E) hit table latch quickly -> **near-table / residual-motion dominated**, not an unload-amplitude question.

### 4. Wrist/hand schedule alone is secondary

Where the grasp is holdable (h8, h4, h7_hz), **E retains** while B may not.  
Grip unload is the primary contact-loss driver on those ICs; leave-tilt wrist with grip locked is not.

### 5. Eccentric lever often missing; CPS-style slide keeps contact but does not create lever

Verified CPP ICs had ~8.9-12.5 mm COM lever. Several freefall-E snaps have `ecc_xy` and/or `|s_axial|` below 8 mm.

Axial slide -> reclamp probe (documented CPS family: mid tok12 x5 + secure translate x10 + reclamp; **no pivoting**):

| IC | snap ecc_xy mm | delta s_axial mm | contact kept |
|----|---------------:|-----------------:|-------------:|
| distal_off01_h8 | 7.7 | +1.9 | yes |
| distal_off01_h4 | 5.8 | -1.1 | yes |
| perp_off01_h7_hz | 3.8 | +0.7 | yes |
| perp_off01_h1 | 3.1 | (lost) | no |

Slide **preserves contact** on 3/4 probed ICs but shifts axial grasp by only ~1-2 mm -- **far from a pivoting-ready eccentric grasp**.  
So: yes, try slide/reclamp before pivoting; **current CPS recipe does not yet solve the lever problem** on these snaps.

---

## Mechanism verdict (Step 5 taxonomy)

| Hypothesis | Verdict on these matched snaps |
|------------|--------------------------------|
| Normal-force insufficient under mid-open | **SUPPORTED** (h8: fn 18->3 then lose; C milder retains) |
| Contact geometry unfit for pivoting | **SUPPORTED** for several ICs (lever <8 mm; slide dS~1-2 mm) |
| Controller switch / residual decode | **SUPPORTED for prior failure** (B_prior 0/6); secondary once decode fixed |
| Unload-only cause of all prior failures | **PARTIAL** -- also high v_rel / near-table latch on A |

---

## Minimal next control changes (not implemented)

1. **Decode:** any CPP attach after E must use **zero-residual** tok decode (match verified pilots). Residual-carry is invalid vs documented CPP.
2. **Do not mid-unload (a6~-0.48) while v_rel is high** -- settle/brake first, or use milder unload (C/D-like) only as a diagnostic lead.
3. **Axial slide -> reclamp** is worth retrying, but needs a recipe that actually moves `|s_axial|` / COM lever by **>=~8 mm** while keeping bilateral -- current CPS mid+secure translate does not.
4. Only after (2)+(3), re-test documented mid-unload -> pivot. Still no obs-only / SFT / GRPO claim.
5. Keep E baseline SUCCESS_NEW path untouched.

---

## Legality

- Privileged diagnostic (MuJoCo forces/contacts, open-loop 7D).
- **Not** 39D observation-only.
- No SUCCESS_NEW recovery claimed this round.
- E baseline from prior experiment preserved.

## Artifacts

| Path | Role |
|------|------|
| `MATCHED_COMPARISON.csv` | per IC x branch event times + mechanism label |
| `AXIAL_SLIDE_PROBE.csv` | slide->reclamp (no pivot) |
| `FAILURE_MECHANISM.md` | taxonomy notes |
| `traces/*__{A,B,B_prior,C,D,E,axial_slide_reclamp}.csv` | per-step raw |
| `snapshots/*__pre_unload.*` | matched tilt-end captures |
| `raw/BUDGETS.json` | decode notes + thresholds |
| `scripts/run_freefall_e_cpp_unload_isolation.py` | runner |
