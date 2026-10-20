# REPORT — Plan C final SUCCESS_NEW

Prior stages preserved (impact → recatch → brake → slow lift).  
This run: **133/300** rollouts, 170.0s.

Locked tolerances: θ≤10.0°, e_pos≤12.0 mm, hold≥0.5s in final 2.0s.

## Direct answers

1. **Easiest root for position correction?** **gbi_010** (and `gbi_022`): already near mid-axis after lift with e_pos≈5–7 mm and θ≈1–3°. `gbi_014` is harder (θ≈87°, e_pos≈21 mm, lateral-dominant). `gbi_002` tip grasp deferred.

2. **Reduce hand-frame COM error to ≤12 mm?**  
   Elevated state: see `POSITION_ERROR_DECOMPOSITION.csv`. `gbi_010`/`gbi_022` already ≤12 mm without further sliding.

3. **Keep orientation ≤10°?** Yes for `gbi_010`/`gbi_022`. `gbi_014` remains far out of band after lift.

4. **Preserve load-bearing bilateral?** Yes through lift and through successful HANDOFF on 010/022.

5. **Physical cause of DROP_AFTER_HANDOFF?**  
   Pre-handoff pose already in band → immediate/stabilize HANDOFF survives (gbi_010/022). Out-of-band / high theta (gbi_014) → DROP_AFTER_HANDOFF from unfavorable grasp pose and residual relative motion under nominal lift — not a nominal-controller bug.  
   Matched A/B/C: `HANDOFF_FAILURE_AUDIT.csv`.

6. **Legal tail that survives nominal?**  
   For in-band roots: **stabilize (optional) → legal HANDOFF → unchanged nominal** → `SUCCESS_NEW`.  
   No privileged freeze/relatch.

7. **Full SUCCESS_NEW?** **2** root(s) with verified experts. Replay successes: **6** (`EXACT_REPLAY.csv`).

8. **Independent 3× replay from original impact root?** Required for each SUCCESS_NEW expert — see `EXACT_REPLAY.csv` / `SUCCESS_NEW_VALIDATION.csv`.

## Experts

[
  {
    "root_id": "gbi_010",
    "path": "B_immediate_handoff",
    "SUCCESS_NEW": 1,
    "pre_pos_mm": 5.431753506569162,
    "pre_theta": 0.8726520596259588,
    "n_actions": 402,
    "correction": "none_already_in_band"
  },
  {
    "root_id": "gbi_022",
    "path": "B_immediate_handoff",
    "SUCCESS_NEW": 1,
    "pre_pos_mm": 6.821350835655757,
    "pre_theta": 2.521179710629848,
    "n_actions": 448,
    "correction": "none_already_in_band"
  },
  {
    "root_id": "gbi_014",
    "path": "best_failed_tail",
    "SUCCESS_NEW": 0,
    "reason": "DROP_AFTER_HANDOFF",
    "pre_pos_mm": 32.07570223911202,
    "pre_theta": 67.37681033877946,
    "e_pos_after": 23.459322219403894,
    "correction": "slide"
  }
]

## Notes

- Did not regenerate impacts or re-run dynamic catch search.
- Did not modify physics, SUCCESS_NEW, K49, or nominal controller.
- Pose-band entry alone is not claimed as success; only full SUCCESS_NEW counts.

## Artifacts

`POSITION_ERROR_DECOMPOSITION.csv`, `IN_HAND_CORRECTION.csv`, `CONTACT_MIGRATION.csv`,
`HANDOFF_FAILURE_AUDIT.csv`, `FINAL_TAIL_SEARCH.csv`, `SUCCESS_NEW_VALIDATION.csv`,
`EXACT_REPLAY.csv`, `FULL_EXPERTS.csv`, `raw/`, `videos/`.
