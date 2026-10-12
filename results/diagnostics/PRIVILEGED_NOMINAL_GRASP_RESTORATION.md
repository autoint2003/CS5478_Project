# Privileged nominal-grasp restoration

The test starts from the K=33 `impact_zp6` reclose at 3.040 s and asks whether a legal v3 action can move the cylinder back into the original pinch. The hand orientation there is already 0.014 rad. The object-in-hand error is 19.6 mm, almost all of it along the hand z axis: `(-2.8, +0.1, +19.4) mm`. Moving the hand and the cylinder together does not reduce that error.

Nothing was trained. The 39-D observation, the K=33 codebook, the ±0.25 residual cap, the v3 action limits, the geometry entry, the nominal basin, and the controller gains were left unchanged. The basin used here is the measured one: hand orientation under 0.117 rad, pinch error under 3.03 mm, relative speed under 0.020 m/s, for 0.100 s, then an explicit nominal handoff.

## What was tried from that same state

A closed grip commanded along the pinch error leaves the 19.6 mm offset unchanged until the pair descends to the abort height. The fingers carry the cylinder. Inward sliding, outward sliding, wrist reseating, and both open-finger schedules do the same: the minimum pinch error stays at 19.6 mm. Compositions of those schedules do not create relative motion either.

A feedback release does. The hand first lifts for 0.20 s while the grip stays closed, so the later drop has room. The grip then opens fully and the hand is driven along +hand z at 2 m/s, the direction of the 19.4 mm error, with a small correction of the 2.8 mm lateral component. When the pinch error falls below 4 mm the grip recloses and a short wrist trim holds the saved nominal orientation. At 3.420 s the pinch error is 2.89 mm, the hand error is 0.0027 rad, both pads are in contact, and the relative speed is 0.013 m/s. The 0.100 s debounce completes at 3.480 s. Nominal control then runs alone.

The minimum pinch error on that trajectory is 2.33 mm. The minimum hand error is 0.0027 rad. Both the 10 s and the 12 s nominal windows hold. Object height at the handoff is 0.498 m. The same law with a 1 m/s hand drop is too slow: the cylinder falls away and the pinch error grows past 40 mm. A 3.5 m/s drop can reach a few millimetres but reaches the abort height before the debounce finishes. The 0.20 s lift and the 2 m/s drop are the pair that both recenters and stays above the table.

The executed sequence, including the policy prefix through the first recatch, is saved in `results/diagnostics/raw/nominal_grasp_restoration/success_trajectory.npz` (39-D observations, previous commands, v3 actions, pinch error, hand error, relative speed, and contacts).

## Local perturbations of the reclose

The same feedback law, not a replay of the recorded numbers, was restarted from the reclose after a hand-frame shift of ±2 mm, a relative-velocity shift of ±0.03 m/s, or an angular-velocity shift of ±0.2 rad/s. Ten of the fourteen perturbed states enter the basin and hold for both 10 s and 12 s. The four misses still reduce the pinch error to 2.6–3.2 mm, then lose the debounce: hand x −2 mm, hand y −2 mm, hand z +2 mm, and hand-x velocity −0.03 m/s. The law is a neighborhood, not a single sample, and it is not uniform across that neighborhood.

## Natural-loss controls

The four recatches that already met the basin were replayed with the same K=33 policy and were not edited. Each again enters the basin and holds for 10 s and 12 s.

| case | handoff pinch error | handoff hand error | 10 s | 12 s |
| --- | ---: | ---: | --- | --- |
| train_dz13.5 | 2.46 mm | 0.057 rad | yes | yes |
| nl_dvz_p | 2.45 mm | 0.058 rad | yes | yes |
| dz13.2 | 2.22 mm | 0.056 rad | yes | yes |
| dvz_m | 2.49 mm | 0.059 rad | yes | yes |

## Main comparison

| case | post-recatch position error | post-recatch orientation error | restoration mechanism | minimum position error | minimum orientation error | nominal basin reached? | handoff occurred? | 10 s nominal stable? | 12 s nominal stable? | mechanism classification |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |
| impact_zp6 closed slide | 19.6 mm | 0.014 rad | closed grip, velocity along the pinch error | 19.6 mm | 0.005 rad | no | no | no | no | closed grasp, no relative motion |
| inward sliding | 19.6 mm | 0.014 rad | existing inward-slide schedule | 19.6 mm | 0.015 rad | no | no | no | no | existing skill, no relative motion |
| wrist reseating | 19.6 mm | 0.014 rad | existing wrist-reseat schedule | 19.6 mm | 0.022 rad | no | no | no | no | existing skill, no relative motion |
| open-finger recatch | 19.6 mm | 0.014 rad | existing open-finger schedule | 19.6 mm | 0.022 rad | no | no | no | no | existing skill, no relative motion |
| impact_zp6 release and regrasp | 19.6 mm | 0.014 rad | lift 0.20 s, full open, hand +z at 2 m/s, reclose | 2.33 mm | 0.0027 rad | yes | yes, 3.480 s | yes | yes | new post-recatch release and regrasp |
| train_dz13.5 | — | — | unmodified learned recatch | 2.46 mm | 0.057 rad | yes | yes | yes | yes | already inside the basin |
| nl_dvz_p | — | — | unmodified learned recatch | 2.45 mm | 0.058 rad | yes | yes | yes | yes | already inside the basin |
| dz13.2 | — | — | unmodified learned recatch | 2.22 mm | 0.056 rad | yes | yes | yes | yes | already inside the basin |
| dvz_m | — | — | unmodified learned recatch | 2.49 mm | 0.059 rad | yes | yes | yes | yes | already inside the basin |

Outward sliding, the longer open-finger schedule, and the three pairwise compositions of the existing skills also leave the pinch error at 19.6 mm or larger. They are the same classification as the existing-skill rows.

LEGAL RECENTERING RESTORES THE NOMINAL GRASP AFTER RECATCh
