# Recovery skill demonstration coverage

No Transformer or GRPO update was run. K=36 and the ±0.25 residual bound were not changed.

## What was collected this pass

| skill | source | level 1 | level 2 (3.03 mm basin) | level 3 (10 s and 12 s nominal hold) |
| --- | --- | --- | --- | --- |
| A, CENTER-6 EARLY, +120° then `τ=-5` | `snap_early.pkl` replay | yes, Δ r_h.x −1.11 mm, bilateral, removed by 0 g | no, residual about 7.6 mm | no under the current basin; object still held at t = 23.2 s |
| B, single-sided support | not recovered | — | — | — |
| C, three saved TRAIN replays | `sliding_vs_regrasp` npz, `handoff_z=None` | yes, open then recontact | yes | yes, 3/3 `SUCCESS_12S` |

The 28 TRAIN captures were not searched with the restored A sequence. The earlier 0/28 A and 0/28 B counts used different controllers and are not skill coverage for these mechanisms.

## Action representation

The CENTER-6 command is legal continuous 7D once it is written in `recovery7d_airborne_v3`: hand-y rate 3 rad/s is wrist action 0.75, secure grip is +1 (`τ=-18`), and `τ=-5` is grip action about −0.3. Translation stays at zero. That sequence was not quantized against the K=36 prototypes in this pass.

Saved C trajectories still leave the open-phase hand-frame `vy` and `vz` outside ±0.25 on a large fraction of steps. That gap is unchanged.

No single-sided B trajectory exists to justify a new prototype.

Physical feasibility, token representability, and closed-loop learning stay separate. Only the first was rechecked, and only on the historical CENTER-6 snapshot plus three C replays.

## Resume

```
python -u training/reproduce_center6_large_angle.py
python -u training/replay_c_subset.py
```

The next search, not run here, is the CENTER-6 sequence transcribed onto TRAIN captures, with level 1 logged separately from the 3.03 mm handoff.

SKILL DEMONSTRATION COVERAGE: PARTIAL

POLICY TRAINING HAS NOT BEEN STARTED
