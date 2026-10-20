# GROUND_SCENE validation REPORT

Scene: `GROUND_SCENE`  
Ground z: `0.0`  GROUND_DROP_Z: `0.04`  
Budget: `{"max_plan_c_roots": 12, "end_grasp_hold_s": 0.4, "impact_speeds": [1.5, 2.5, 3.5], "impact_offsets_m": [0.0, 0.012, 0.02], "no_sft_grpo": true}`

## Results (honest)
- Geometry table absent: `1`
- GROUND_DROP unit tests pass: `True`
- Rest-on-ground while disarmed OK: `True`
- Armed freefall latches GROUND_DROP: `{'latched': 1, 'z': 0.020272949567362142}`
- End-grasp viable: `3`
- Extra ballistic fall vs table from z=0.50 (approx): `0.16676470035865912` s
- Freefall measure: `{'z0': 0.5, 'z_end': 0.4549808670806839, 't_to_ground_drop': nan, 't_first_detach': 0.020000000000000004, 'ground_drop': 0, 'predicted_ballistic_s': 0.30954901265136553}`
- Plan C bilateral recatch: `0` / `4`
- Plan C SUCCESS_NEW: `0` (none claimed without full pipeline)

Authoritative outcome semantics: `docs/GROUND_SCENE_OUTCOMES.md`.
