# Legacy recatch reproduction and local search

This audit reconstructs the stored airborne recatch, replays it through the current `recovery4d` path, and searches only the timing neighborhood of that throw. Nothing was trained. The action space was not changed.

Numbers: `results/diagnostics/raw/legacy_recatch_reproduction_audit/summary.json` and `legacy_tape.npz`. The runner is `training/legacy_recatch_reproduction_audit.py`.

The initial state is the CENTER 6 m/s ballistic branch at `T_EARLY_DRIFT = 3.322 s`. The throw is `recatch_one` at `CLAIMED_OPEN_DEG = 70` and `t_close = 3.722 s`, with the 0.25 s catch-confirm hold used by the comparison log. Continuation to absolute `t = 12` is `continue_long`: the nominal lift controller, with no return-to-upright script inserted.

## 1. Reconstructed legacy trajectory

The fresh run captures and holds. Its event times match `airborne_events.json` within one physics step (2 ms). `ctrl[7]` on the aligned samples matches the stored npz exactly, and object height differs by at most 2.4 mm. Finger contact counts on those samples do not match one-for-one (41 of 72). The reconstruction used below is this fresh `recatch_one` tape.

Commands are constant inside four phases. Hand translation is zero throughout.

| phase | duration | `ω_y` | tendon |
| --- | ---: | ---: | ---: |
| rotate, grip secure | 0.368 s (184 steps) | +4.0 rad/s | −18 |
| keep rotating, grip open | 0.026 s (13 steps) | +4.0 rad/s | +2 |
| stop rotating, grip still open | 0.006 s (3 steps) | 0 | +2 |
| reclose and confirm | 0.318 s (159 steps) | 0 | −18 |

Release and reclose:

| event | time |
| --- | ---: |
| rotate starts | 3.322 s |
| open command | 3.690 s |
| both fingers off | 3.700 s |
| close command | 3.722 s |
| recontact and bilateral | 3.736 s |
| hold confirmed | 4.040 s |

The contact-free interval is 0.036 s (18 physics steps). Minimum hand-origin to object-origin distance during that interval is 0.104 m, which is the pinch offset rather than a long flight: the cylinder stays in front of the palm while both finger contacts are off. At recontact the grip command is already −18, `nL = 11`, `nR = 9`, object height is 0.641 m, and the hand-frame offset is about `[3 mm, 0, 109 mm]`.

During the throw the hand origin reaches 0.226 m/s linear and 3.99 rad/s angular. From open to bilateral the peaks are 0.206 m/s and 3.99 rad/s. Those linear speeds occur with a commanded linear velocity of 0.

Nominal continuation from the end of the confirm hold retains to `t = 12`. Physical label `RETAINED`: both fingers in contact (`10 / 10`), object height 0.699 m, no drop time, no escape time. Planar hand-object offset at the end is 0.110 m, so the older task-box class is `OTHER_FAILURE`. The hand is still in the lift phase because the lift target is 0.713 m. The physical label does not use the 0.08 m box.

## 2. Replay through current recovery4d

Each legacy physics step maps to a legal action and is applied with `map_recovery4d` and `recovery4d_tick` at `n_sub = 1`.

The map reproduces the tape with no command mismatch. On the first rotate step, `v_cmd = 0`, `w_cmd = [4, 0, 0]` rad/s, and `ctrl[7] = −18`. Open steps write `ctrl[7] = +2`. On this snapshot the desired hand y axis is world x, so a body-y rate of 4 rad/s is a world rate of 4 rad/s about x.

Two scene details were then separated by replaying the same commands:

- With `park_ball` after every physics step, which is what `tick_vw` does on this scene, hand and object positions match the legacy tape for all 359 steps. The following nominal continuation matches the legacy `t = 12` result: `RETAINED`, height 0.699 m, planar offset 0.110 m.
- Without that park, the object is already 0.15 mm off on the first step. Contacts on that step still agree (`7 / 4`). The ball is 1.24 m from the cylinder and already below the table, so it is not touching the grasp. Over the whole tape the object drifts by at most 2.1 mm and the hand by at most 0.8 mm. The tape still ends in bilateral contact.

`park_ball` on this scene teleports the ball and calls `mj_forward`. That extra forward is what makes the state tape agree exactly. It is not required for the recatch outcome: the policy-rate replays below do not park the ball, and they still recontact and retain.

The native policy rate holds one action for `n_sub = 10` (0.02 s), with the world twist frozen for those ten steps. Quantizing the tape at that rate gives 19 steps of `[+1, 0, 0, +1]`, 1 step of `[+1, 0, 0, −1]`, and 12 steps of `[0, 0, 0, +1]`. The 6 ms coast is shorter than one policy step, so it is absorbed into the neighboring action. This schedule loses both contacts for 0.018 s, recontacts, and the nominal continuation retains to `t = 12`. A second replay of the same actions retains with the same 0.018 s gap.

## 3. Controller differences that show up in the replay

The commanded `ω_y`, the zero translation, and the tendon values pass through `map_recovery4d` unchanged. `recovery4d_tick` also stores `v_des`, `w_des`, and `fg_cmd`. Those stores do not move the matched tape: with the ball parked, the object and hand agree exactly, so the Cartesian write is the same one `tick_vw` uses.

The state difference that remains is the per-step ball park and its `mj_forward`. Leaving the ball free shifts the cylinder by a fraction of a millimeter immediately and by about 2 mm over the throw. The policy-rate recatch still retains without that park, so the `t = 12` result does not depend on it.

The 0.25 s confirm and the later nominal lift are enough for retention. This replay does not run the legacy return-to-upright. The hand stays at the thrown orientation while the lift continues, and the grasp is still bilateral at `t = 12`.

## 4. Local timing search

Around the quantized schedule, at `a0 = +1`:

- rotate length: 17, 18, 19, 20, 21 policy steps (0.34 s to 0.42 s)
- wrist-open length: 1, 2, 3 policy steps (0.02 s to 0.06 s)
- extra coast-open length: 0, 1, 2 policy steps

That is 45 schedules. Each is followed by the same 0.24 s secure hold and then nominal continuation to `t = 12`.

## 5. Success basin

Fourteen of the 45 schedules lose contact, recontact, and retain on the first run.

| open | extra coast | release lengths that retain | contact-free time | second replay |
| --- | --- | --- | --- | --- |
| 1 step (0.02 s) | 0 | 17 through 21 | 0.018 s | all five retain |
| 1 step | 1 step (0.02 s) | 17 through 21 | 0.046–0.048 s | all five retain; one gap length moves by 2 ms |
| 2 steps (0.04 s) | 0 | 18 through 21 | 0.046 s | 19 and 21 retain; 18 and 20 drop |
| 1 step | 2 steps | none | 0.072–0.074 s | drop |
| 2 steps | 1 or 2 | none | 0.074 s, or no recontact | escape |
| 3 steps (0.06 s) | any | none | 0.076–0.330 s | escape |

Release time can move by ±40 ms and the 0.02 s open still retains. Opening for 0.06 s, or leaving the grip open for a further 0.04 s after a 0.02 s wrist pulse, loses the object. The 0.04 s open is only partly repeatable. On the nominal initial condition the basin is moderately narrow: a band of release times about 80 ms wide, and a reclose that has to arrive within about 40 ms of the open.

Minimum hand-object origin separation on the retaining runs stays near 0.105 m. Peak hand linear speed on the `a0 = +1` runs is 0.228 m/s with commanded linear velocity 0. Peak commanded angular speed is 4 rad/s.

The same timings at `a0 = 0.75` (3.0 rad/s), including the rotate length scaled by `4/3`, also retain on the nominal state. Those runs are the same family at a lower wrist command, so they are not counted as extra seeds.

## 6. Nearby initial conditions and dynamics

The nominal object is 0.20 kg with sliding friction 1.0. The retune around the retaining schedule uses rotate lengths 18, 19, 20 and open lengths 1 and 2, at the same coast of 0.

Branch time shifted by −40, −20, +20, and +40 ms: the 0.02 s open retains at all three rotate lengths on every shift. The 0.04 s open drops on some of those shifts.

One-factor dynamics on the original branch state:

| change | 0.02 s open | 0.04 s open |
| --- | --- | --- |
| mass 0.16 kg | retains at rotate 18, 19, 20 | retains at all three |
| mass 0.24 kg | retains at all three | retains only at rotate 19 |
| friction 0.70 | retains at all three | drops at all three |
| friction 1.15 | retains at all three | drops at all three |

The schedule that matches the stored throw, 19 rotate steps and one open step, retains on every nearby branch time and every dynamics cell above without a change in the action list. The longer open does not.

## 7. Seed set

One independent family survived the repeat check. Its representative is the quantized legacy throw:

- 19 steps `[+1, 0, 0, +1]`
- 1 step `[+1, 0, 0, −1]`
- 12 steps `[0, 0, 0, +1]`

It uses current `recovery4d` actions at `n_sub = 10`, retains on a second replay with the same 0.018 s contact-free interval, and the nominal continuation retains to `t = 12`. The other repeatable members are the same open and a rotate length of 17, 18, 20, or 21, plus the one-step coast variants and the repeatable two-step opens at rotate 19 and 21. They are shifts inside this family, not separate mechanisms.

## 8. What this does and does not show

The stored throw is a 4 rad/s wrist pulse, a grip open of about 30 ms, and an immediate reclose, with no hand-translation command. Current `recovery4d` can issue those commands. Held at the 20 ms policy rate, a 20 ms open still produces a both-fingers-off interval, a recontact, and physical retention to `t = 12` while the nominal lift continues. That same action list still retains for branch times within ±40 ms, mass 0.16 kg and 0.24 kg, and friction 0.70 and 1.15. Opening the grip for 60 ms leaves the local basin.

RECATCh GENERALIZES LOCALLY ACROSS ICS / DYNAMICS
