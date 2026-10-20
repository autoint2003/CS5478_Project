# REPORT — Post-catch end-grasp recovery

Prior dynamic-catch tapes preserved. This run: **243** rollouts (~582 s).

Locked evaluation targets (unchanged):

- `P_TGT ≈ (−0.001, 0, 0.105)` m  
- `U_TGT ≈ (0, 0, −1)`  

Reference operating height taken from each root’s install `cfg.z` (0.55 or 0.60 m).

## Brake-complete geometry

| root | obj_z | clearance | contact s (COM→pad) | dist to nearest end | near-end? |
|---|---|---|---|---|---|
| gbi_002 | 0.047 m | 0.017 m | **−0.028 m** | **1.6 mm** | **yes (extreme)** |
| gbi_010 | 0.033 m | 0.003 m | +0.002 m | 28 mm | no (near mid) |
| gbi_014 | 0.238 m | 0.21 m | +0.015 m | 15 mm | moderate |
| gbi_022 | 0.035 m | 0.005 m | −0.003 m | 27 mm | no (near mid) |

Cylinder half-length = 30 mm. Only **gbi_002** is a true fingertip/end grasp after catch; 010/022 sit near mid-axis but very low altitude.

## Direct answers

1. **How close to the cylinder end?**  
   gbi_002: **1.6 mm** from the nearer end (`s≈−0.028`). Others: 15–28 mm (not end-critical).

2. **Can the end grasp support a slow vertical lift?**  
   **Yes.** With `vz_cmd ∈ {0.08, 0.12}` m/s and ≤12 s horizon, all four roots lift the **cylinder with the hand** to the reference height while keeping bilateral contact (`SLOW_LIFT.csv`). Example gbi_002: z 0.047 → **0.540 m**, continuous bilateral through the lift.

3. **Does wrist rotation create a useful axial gravity component?**  
   At brake-complete, `g_axial = g·u` already favors interior slide for some roots (`gravity_helps_now`). Wrist search changes `g_axial` during the maneuver (`GRAVITY_SLIDING.csv`). Useful when combined with reduced grip — not by itself on every root.

4. **Controlled axial sliding toward interior?**  
   **Yes on gbi_014** (20 genuine trials, |Δs| ≥ 3 mm toward COM, bilateral retained). Best cases move `s` from ≈0.018 → ≈0. Best orders use light grip (0–0.35) + wrist about x/y/z.  
   gbi_002 extreme end-grasp did **not** produce reliable interior migration in this budget (risk of escape).

5. **Does the cylinder actually shift vs finger pads?**  
   **Yes (gbi_014)** — scored from MuJoCo contact axial coordinate `s_mean`, not hand/world motion. Example: `ds ≈ −0.018 m` with bilateral hold.

6. **Safer order?**  
   - **Low clearance (010/022/002):** **B2** (small clearance lift → slide → remaining lift) is required for safe wrist motion; sequences that reach height after B2 dominate `SEQUENCE_COMPARISON.csv` for gbi_002.  
   - **Mean sequence score overall:** **A1** (lift then slide) slightly highest among completed composites that retained grasp.  
   - **Slide-before-lift (B1)** is riskier at <2 cm clearance.

7. **Reach reference operating height?**  
   **Yes** for all four roots via slow lift (`reached_target_z=1`).

8. **Pose correction to locked target?**  
   Orientation often enters band (`e_theta` ≈ 1–4°), but **position stays ~33 mm off** (`e_pos_mm≈33`) because the COM is still axially offset from `P_TGT` after an end/near-end grasp. Pose stage then HANDOFF → **`DROP_AFTER_HANDOFF`** (20 sequence attempts).

9. **SUCCESS_NEW?**  
   **0 / 4**. Capture-tape / lift experts exist; full nominal success after handoff does not.

## Mechanism milestones (kept distinct)

| Milestone | Status |
|---|---|
| Stable post-catch lift to ref height | **PASS** (4/4) |
| Genuine axial regrasp (contact migration) | **PASS on gbi_014**; fail/unproven on extreme end-grasp gbi_002 |
| Pose within locked tolerances | **FAIL** (e_pos ~33 mm) |
| HANDOFF + SUCCESS_NEW | **FAIL** (`DROP_AFTER_HANDOFF`) |

## Limiting constraint

Not lift authority: slow legal upward commands raise the cylinder.  
**Blocker:** after lift, the grasp still leaves COM too far from `P_TGT` (especially gbi_002’s 1.6 mm end grasp). Gravity-assisted sliding can fix this when clearance and grip allow (gbi_014), but has not yet been combined into a lift→slide→pose chain that survives nominal HANDOFF within this budget.

## Video

`videos/gbi_002_slow_lift_example.mp4` — brake-complete end-grasp → slow lift to ~0.54 m with continuous bilateral contact.

## Artifacts

`BRAKE_SNAPSHOT_AUDIT.csv`, `GRASP_GEOMETRY.csv`, `SLOW_LIFT.csv`, `GRAVITY_SLIDING.csv`,  
`SEQUENCE_COMPARISON.csv`, `POSE_RECOVERY.csv`, `HANDOFF_VALIDATION.csv`,  
`EXACT_REPLAY.csv`, `FULL_EXPERTS.csv`, `raw/`, `videos/`.
