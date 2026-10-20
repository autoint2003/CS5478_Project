# REPORT — Impact-aligned early Recovery

Prior experiments preserved under `results/diagnostics/ground_ball_impact_*`.  
This run: **363/400** counterfactual rollouts, 87.9s.

## Direct answers

1. **How much earlier can legal Recovery begin after real impact?**  
   At the first **20 ms policy boundary after actual ball–cylinder contact** (branch **A**).  
   Earliest measured impact onset ≈ `0.006000000000000005` s after launch-hold start (`IMPACT_TO_LOSS_TIMELINE.csv` EVENTS).  
   No Recovery action before impact. Mid-step snaps are not granted illegal mid-bin actions.

2. **When does usable jaw-entry geometry disappear?**  
   On the four prior roots, usable geometry (`ALREADY_INSIDE_JAW`) lasts through branches **A–C**; first infeasible class appears at **D** (sustained 20 ms loss) or **E** (late post-loss / axially past).  
   Detail: `{
  "gbi_002": {
    "first_infeasible_branch": "D",
    "classes": [
      [
        "A",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "B",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "C",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "D",
        "OPENING_TOO_SLOW"
      ],
      [
        "E",
        "AXIALLY_PAST_FINGERS"
      ]
    ]
  },
  "gbi_010": {
    "first_infeasible_branch": "D",
    "classes": [
      [
        "A",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "B",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "C",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "D",
        "OPENING_TOO_SLOW"
      ],
      [
        "E",
        "AXIALLY_PAST_FINGERS"
      ]
    ]
  },
  "gbi_014": {
    "first_infeasible_branch": "D",
    "classes": [
      [
        "A",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "B",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "C",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "D",
        "OPENING_TOO_SLOW"
      ],
      [
        "E",
        "AXIALLY_PAST_FINGERS"
      ]
    ]
  },
  "gbi_022": {
    "first_infeasible_branch": "D",
    "classes": [
      [
        "A",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "B",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "C",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "D",
        "AXIALLY_PAST_FINGERS"
      ],
      [
        "E",
        "AXIALLY_PAST_FINGERS"
      ]
    ]
  },
  "grad_grad_s7.5_lateral_a0.022_t0.0_o0.0": {
    "first_infeasible_branch": null,
    "classes": [
      [
        "A",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "B",
        "ALREADY_INSIDE_JAW"
      ],
      [
        "C",
        "ALREADY_INSIDE_JAW"
      ]
    ]
  }
}`

3. **Can the fingers open sufficiently before cylinder escape?**  
   Opening-to-0.03 m from A/B is physically achievable under legal open (`OPENING_RESPONSE.csv`: **True**).  
   On the four prior roots this does **not** yield capture: escape velocity / axial slip at A already exceeds what open-then-close recovers inside the remaining air time.  
   Closing harder at A on those roots **ejects** the cylinder (squeeze-out); preserve-mode close also fails within ≤1–2 policy steps.

4. **Does earlier intervention improve capture?**  
   On the four prior roots: **no** (≥60 ms continuous bilateral never achieved).  
   On gradual-slip lateral roots generated after that negative result: **True** — early A-branch velocity-matched preserve recovers continuous bilateral.  
   Late-only (D/E) improvement: **False**.

5. **Best intervention type?**  
   Per-root best: `{'gbi_002': 'FAILED_OR_TRANSIENT', 'gbi_010': 'FAILED_OR_TRANSIENT', 'gbi_014': 'FAILED_OR_TRANSIENT', 'gbi_022': 'FAILED_OR_TRANSIENT', 'grad_grad_s7.5_lateral_a0.022_t0.0_o0.0': 'POST_IMPACT_GRASP_PRESERVATION'}`  
   Counts: preserve=1, partial_slip=0, full_detach_recatch=0.  
   No `FULL_DETACH_ACTIVE_RECATCH` was demonstrated (complete separation + later bilateral reacquisition).  
   Best positive outcome is **`POST_IMPACT_GRASP_PRESERVATION`** on gradual-slip laterals (prevents full loss; not a detached recatch).

6. **Continuously bilateral ≥150 ms?**  
   **1** expert root(s). Primary four: 0. Gradual-slip addition: see `FULL_EXPERTS.csv`.

7. **Full SUCCESS_NEW?**  
   **0**. Preserve experts that reach ≥150 ms bilateral still fail handoff/nominal with `DROP_AFTER_HANDOFF` (grasp does not survive lift under unchanged SUCCESS_NEW gate).

8. **Exact replay without hidden intervention?**  
   Capture-tape replay bilateral: **True**. Full SUCCESS_NEW replay: **False** (`EXACT_REPLAY.csv`).  
   Primary four unsuccessful: **True**.

## Mechanism notes

- Snapshots A–E from real hold trajectories; Recovery only post-impact.
- Finger opening via joint **qposadr** (not joint-id indexing).
- `POST_IMPACT_GRASP_PRESERVATION` ≠ `FULL_DETACH_ACTIVE_RECATCH` — reported separately.
- Earliest physically useful legal window: **branch A on gradual-slip impacts** (slip >=60-120 ms impact to sustained loss). Violent top/side impacts (prior gbi_002/010/014/022) leave no preservable window at the first legal step.

## Artifacts

`IMPACT_TO_LOSS_TIMELINE.csv`, `EARLY_BRANCH_SNAPSHOTS.csv`, `JAW_ENTRY_FEASIBILITY.csv`,
`OPENING_RESPONSE.csv`, `EARLY_ACTION_SEARCH.csv`, `CAPTURE_CONTINUITY.csv`,
`RECOVERY_MECHANISM_CLASSIFICATION.csv`, `EXACT_REPLAY.csv`, `FULL_EXPERTS.csv`, `raw/`, `videos/`.
