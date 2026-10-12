# Privileged ball-impact recovery search

Search used privileged contact and pose to choose actions. Every simulated command was a clipped `recovery7d_airborne_v3` 7-vector. The cylinder, the hand, and the ball were not teleported. The ball was not parked. No policy weights were updated.

A candidate is `VERIFIED_SUCCESS` only after replay from the saved TRAIN snapshot reproduces the nominal basin (pinch ≤ 3.03 mm, hand orientation ≤ 0.117 rad, relative speed ≤ 0.020 m/s, both fingers in contact, grip closed, 0.100 s persistence), an explicit handoff, and 10 s of nominal-only control before any drop. The same call continues to 12 s. Every verified trajectory in this run also survived that 12 s diagnostic (`SUCCESS_12S`). A basin entry whose nominal hold drops is `BASIN_REACHED_BUT_HOLD_FAILED`.

## 1. Search implementation and legal-action check

Nine closed-loop laws, plus four noise refinements of the best law when the pinch error fell below 20 mm and no success had been verified. Each law is privileged feedback: hand-frame velocity toward the lift pinch, a feedforward of the object’s relative velocity, a wrist command that returns the hand toward its entry orientation, and a grip that opens while the pinch error is outside 3.5 mm on the release laws and closes once the cylinder is inside that ball. A world-up lift is added only while the fingers are in contact and the cylinder is below the lift band. Recovery is allowed to pass briefly below the 0.44 m height gate during an open chase; it stops on a 12 cm horizontal escape, a fall below 0.30 m, or one second of contact loss below 0.44 m.

```
{
  "version": "recovery7d_airborne_v3",
  "vx_vy_mps": 1.0,
  "vz_down_mps": 3.8,
  "vz_up_mps": 1.0,
  "wrist_rad_s": 4.0,
  "wrist_command_norm": 4.0,
  "grip_open_nm": 2.0,
  "grip_secure_nm": -18.0,
  "ball_parked": false,
  "qpos_written": false
}
```

Wall time for this finished pass: 762.1 s. Attempt rows: 190.

## 2. TRAIN initial conditions

18 accepted TRAIN ball-impact snapshots. FINAL_TEST was not searched. VALIDATION was not used to choose gains.

At entry every snapshot still had contacts on both fingers. The offset that matters for the basin is the pinch position, and it was 7–25 mm, almost entirely along hand +z on the profiled +X case. +X entries were spinning at about 2.5–4.2 rad/s with object-orientation error near 0.15 rad. +Y and oblique entries were slower and more misoriented, up to about 1.0 rad. None was airborne at entry. A one-sample trace does not meet the two-step slip or rolling rules, so those descriptors are not claimed at the single entry instant.

## 3. Budgets and attempt outcomes

```
{
  "attempt_status": {
    "PARTIAL_RECOVERY": 103,
    "BASIN_REACHED_BUT_HOLD_FAILED": 23,
    "VERIFIED_SUCCESS": 49,
    "DROP_OR_ESCAPE": 15
  },
  "verified_rows_by_strategy": {
    "release_short": 10,
    "lift_soft": 11,
    "release_long": 14,
    "chase_lost": 14
  }
}
```

Distinct TRAIN initial conditions with a replay-verified 10 s and 12 s hold: 18 / 18.
Verified strategy executions, counting more than one law on the same initial condition: 49.
One trajectory file is kept per initial condition: the verified law with the smallest pinch error. Firm closed-grip laws left the pinch error at its entry value, because a clamped cylinder translates with the hand. The laws that verify open the fingers, chase the pinch error, close inside 3.5 mm, and lift.

## 4. Per-IC best trajectory

- `ball_train_0001` +X: pinch 0.01475 m, object orientation 0.15183 rad, contacts L/R 2/6, spin 3.4014 rad/s. VERIFIED_SUCCESS via `release_short`, min pinch 0.00271 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.355, max L-inf 0.482, quantized replay 10 s False.
- `ball_train_0002` +X: pinch 0.01564 m, object orientation 0.14612 rad, contacts L/R 2/2, spin 2.8233 rad/s. VERIFIED_SUCCESS via `lift_soft`, min pinch 0.00045 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.296, max L-inf 0.419, quantized replay 10 s False.
- `ball_train_0003` +X: pinch 0.01091 m, object orientation 0.16273 rad, contacts L/R 3/7, spin 2.4904 rad/s. VERIFIED_SUCCESS via `lift_soft`, min pinch 0.00051 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.286, max L-inf 0.441, quantized replay 10 s False.
- `ball_train_0004` +X: pinch 0.01478 m, object orientation 0.1546 rad, contacts L/R 2/6, spin 3.443 rad/s. VERIFIED_SUCCESS via `release_long`, min pinch 0.00083 m, existing label `SUCCESS_12S`.
  Mechanisms: ['controlled_sliding', 'partial_release_regrasp']. ['zeroing lateral channels removed the 10 s hold while slip was measured', 'forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.574, max L-inf 0.52, quantized replay 10 s False.
- `ball_train_0005` +X: pinch 0.01567 m, object orientation 0.14809 rad, contacts L/R 2/3, spin 2.6844 rad/s. VERIFIED_SUCCESS via `release_long`, min pinch 0.00233 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.393, max L-inf 0.467, quantized replay 10 s True.
- `ball_train_0006` +X: pinch 0.01099 m, object orientation 0.16053 rad, contacts L/R 2/7, spin 2.7788 rad/s. VERIFIED_SUCCESS via `lift_soft`, min pinch 0.00178 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.226, max L-inf 0.471, quantized replay 10 s False.
- `ball_train_0007` +X: pinch 0.01498 m, object orientation 0.15202 rad, contacts L/R 1/5, spin 4.1661 rad/s. VERIFIED_SUCCESS via `lift_soft`, min pinch 0.00061 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.419, max L-inf 0.431, quantized replay 10 s True.
- `ball_train_0008` +X: pinch 0.01584 m, object orientation 0.14759 rad, contacts L/R 2/2, spin 3.1245 rad/s. VERIFIED_SUCCESS via `chase_lost`, min pinch 0.00227 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.34, max L-inf 0.561, quantized replay 10 s False.
- `ball_train_0009` +X: pinch 0.0112 m, object orientation 0.16054 rad, contacts L/R 2/8, spin 2.8298 rad/s. VERIFIED_SUCCESS via `release_short`, min pinch 0.00054 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['lateral commands coincided with slip; the no-lateral ablation still held', 'forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.596, max L-inf 0.52, quantized replay 10 s False.
- `ball_train_0010` oblique_+X_+Y: pinch 0.01741 m, object orientation 0.84989 rad, contacts L/R 5/3, spin 0.5724 rad/s. VERIFIED_SUCCESS via `chase_lost`, min pinch 0.00271 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.256, max L-inf 0.636, quantized replay 10 s False.
- `ball_train_0011` +Y: pinch 0.00946 m, object orientation 0.40985 rad, contacts L/R 5/6, spin 0.0361 rad/s. VERIFIED_SUCCESS via `release_long`, min pinch 0.00131 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.344, max L-inf 0.52, quantized replay 10 s False.
- `ball_train_0012` oblique_+X_+Y: pinch 0.02519 m, object orientation 1.0363 rad, contacts L/R 1/2, spin 1.0661 rad/s. VERIFIED_SUCCESS via `lift_soft`, min pinch 0.00123 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.4, max L-inf 0.451, quantized replay 10 s False.
- `ball_train_0013` +Y: pinch 0.01418 m, object orientation 0.57692 rad, contacts L/R 1/1, spin 0.1014 rad/s. VERIFIED_SUCCESS via `lift_soft`, min pinch 0.00177 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.404, max L-inf 0.447, quantized replay 10 s False.
- `ball_train_0014` +Y: pinch 0.00749 m, object orientation 0.49892 rad, contacts L/R 6/4, spin 0.383 rad/s. VERIFIED_SUCCESS via `chase_lost`, min pinch 0.00043 m, existing label `SUCCESS_12S`.
  Mechanisms: ['controlled_sliding', 'partial_release_regrasp']. ['zeroing lateral channels removed the 10 s hold while slip was measured', 'forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.534, max L-inf 0.518, quantized replay 10 s False.
- `ball_train_0015` oblique_+X_+Y: pinch 0.00803 m, object orientation 0.29516 rad, contacts L/R 8/13, spin 0.5324 rad/s. VERIFIED_SUCCESS via `lift_soft`, min pinch 0.00041 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.29, max L-inf 0.433, quantized replay 10 s False.
- `ball_train_0016` +Y: pinch 0.01133 m, object orientation 0.67073 rad, contacts L/R 6/2, spin 0.6564 rad/s. VERIFIED_SUCCESS via `release_long`, min pinch 0.0019 m, existing label `SUCCESS_12S`.
  Mechanisms: ['controlled_sliding', 'partial_release_regrasp']. ['zeroing lateral channels removed the 10 s hold while slip was measured', 'forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.589, max L-inf 0.52, quantized replay 10 s False.
- `ball_train_0017` oblique_+X_+Y: pinch 0.01169 m, object orientation 0.45861 rad, contacts L/R 2/5, spin 1.3697 rad/s. VERIFIED_SUCCESS via `lift_soft`, min pinch 0.00025 m, existing label `SUCCESS_12S`.
  Mechanisms: ['partial_release_regrasp']. ['forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.3, max L-inf 0.443, quantized replay 10 s False.
- `ball_train_0018` +Y: pinch 0.00936 m, object orientation 0.58613 rad, contacts L/R 5/8, spin 0.3398 rad/s. VERIFIED_SUCCESS via `lift_soft`, min pinch 0.00033 m, existing label `SUCCESS_12S`.
  Mechanisms: ['controlled_sliding', 'controlled_rolling', 'partial_release_regrasp']. ['zeroing lateral channels removed the 10 s hold while slip was measured', 'zeroing wrist channels removed the 10 s hold while the rolling signature was present', 'forcing the grip closed removed the 10 s hold']
  K=36 fraction within ±0.25: 0.27, max L-inf 0.682, quantized replay 10 s False.

## 5. Mechanisms

Labels require a matched ablation. Controlled sliding is recorded only when the rollout measured tangential slip and zeroing the lateral channels removed the 10 s hold. Controlled rolling is recorded only when the rolling signature was present and zeroing the wrist channels removed the hold. Opening and then regaining bilateral contact is partial release and regrasp. Forcing the grip to stay closed removed the hold on the verified trajectories that carry that label.

```
{
  "partial_release_regrasp": 18,
  "controlled_sliding": 4,
  "controlled_rolling": 1
}
```

Partial release and regrasp is the mechanism shared by the saved trajectory of every verified initial condition. Controlled sliding met the ablation test on 4 of them. Controlled rolling met it on 1 (`ball_train_0018`). Object spin at entry, including the 2.5–4 rad/s +X cases, is not counted as controlled rolling.

No saved entry was a complete loss or an airborne separation, so this search did not demonstrate an airborne recatch. Asymmetric contact counts were common, and recovery still went through an open chase rather than a contact-preserving unilateral correction. The closed-grip laws are the negative evidence for that contact-preserving route: they kept the fingers shut and did not bring the pinch inside 3.03 mm.

## 6. K=36 representability

For each saved action, `d_t` is the L-infinity distance to the nearest of the 36 prototypes from `policy_B_init.pt`. A step is representable when `d_t ≤ 0.25`. The quantized command is that prototype plus the residual clipped to ±0.25.

Quantized replay preserved the 10 s hold on 2 of 18 saved trajectories. No saved trajectory was inside ±0.25 at every step. The mean absolute gap to the chosen prototype is largest on hand-frame vy and vz. Grip and wrist channels are already close to some prototype. That is an action-coverage gap, not a measurement of whether the Transformer would select the right token.

No token was added and the residual bound was not changed. The recurring missing piece, if a later representation change is justified, is a translational chase during the open phase, not another grip or wrist token.

## 7. Logs

Attempt log: `results/diagnostics/raw/privileged_ball_impact_recovery/search_log.jsonl`.
Summary: `results/diagnostics/raw/privileged_ball_impact_recovery/summary.json`.
Trajectories: `results/diagnostics/raw/privileged_ball_impact_recovery/trajectories/`.

Replay the search, skipping finished TRAIN keys:

```
python -u training/privileged_ball_impact_search.py
```

The sealed 120-update GRPO pilot was not modified.

PRIVILEGED RECOVERY SEARCH COMPLETE

POLICY TRAINING HAS NOT BEEN STARTED.
