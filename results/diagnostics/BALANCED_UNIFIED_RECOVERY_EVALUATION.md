# Balanced unified recovery

The policy interface is unchanged: the 45-D observation, the frozen K=32 tokenizer, the causal Transformer, recovery7d_airborne_v3, and the tracking gains. Family labels are used only to give each verified recovery family an equal share of the training batches. The baseline is the seed-1 120-epoch checkpoint. The balanced model trains for 120 epochs, 9600 optimizer steps, in 152.4 s, from torch seed 1.

A case succeeds only when the recovery policy acts, the nominal controller then continues unchanged, and the object is still in a bilateral grasp above the lift height at absolute t = 12, with no drop and no escape. A 1 s recatch is not the score.

## Dataset

Training trajectories by family: wrist 1, sliding 1, open-finger recatch 1, airborne carry 0, natural-loss 18 windows from the six core offsets. Batch fractions are 0.25 for wrist, sliding, recatch, and natural-loss. Final training loss 0.0547. Mean token accuracy across those four families is 0.970. Held-out natural-loss token accuracy is 0.891 and held-out loss is 0.275.

The six natural-loss offsets are the core cases whose first distal-tip loss the v3 intercept regrasps. They receive the same batch share as wrist reseating, inward sliding, and the open-finger recatch. The seated vertical release and the sideways carry-and-open states did not yield a verified v3 recatch, so they were not added as demonstrations. Held-out natural-loss offsets are not in this training set. The mechanism column is read from the rollout: a large wrist command with contact kept, contact kept without that command, or contact lost and then regained.

## t = 12 by disturbance family

| disturbance family | cases | baseline | balanced policy | dominant successful mechanism | failure mode |
|---|---:|---:|---:|---|---|
| airborne | 4 | 0/4 | 0/4 | none | DROP |
| impact | 4 | 2/4 | 3/4 | wrist reseat | ESCAPE |
| natural | 4 | 0/4 | 0/4 | none | DROP |
| recatch | 4 | 4/4 | 4/4 | contact-preserving | none |
| sliding | 4 | 4/4 | 4/4 | wrist reseat | none |
| wrist | 4 | 3/4 | 4/4 | wrist reseat | none |

Macro-average family success is 0.542 for the baseline and 0.625 for the balanced policy. Case totals are 13/24 and 15/24.

## Cases

| case | family | baseline t=12 | balanced t=12 | balanced mechanism |
|---|---|---|---|---|
| impact_zm6 | impact | RETAINED_TO_12 | RETAINED_TO_12 | wrist reseat |
| impact_zp6 | impact | DROP | ESCAPE | none |
| impact_phi90 | impact | DROP | RETAINED_TO_12 | contact-preserving |
| impact_z3 | impact | RETAINED_TO_12 | RETAINED_TO_12 | wrist reseat |
| slide_suffix_20 | sliding | RETAINED_TO_12 | RETAINED_TO_12 | recontact |
| slide_suffix_36 | sliding | RETAINED_TO_12 | RETAINED_TO_12 | contact-preserving |
| slide_dy1 | sliding | RETAINED_TO_12 | RETAINED_TO_12 | wrist reseat |
| slide_dz1 | sliding | RETAINED_TO_12 | RETAINED_TO_12 | wrist reseat |
| wrist_suffix_3 | wrist | DROP | RETAINED_TO_12 | wrist reseat |
| wrist_suffix_6 | wrist | RETAINED_TO_12 | RETAINED_TO_12 | wrist reseat |
| wrist_suffix_9 | wrist | RETAINED_TO_12 | RETAINED_TO_12 | wrist reseat |
| wrist_suffix_12 | wrist | RETAINED_TO_12 | RETAINED_TO_12 | recontact |
| recatch_suffix_8 | recatch | RETAINED_TO_12 | RETAINED_TO_12 | wrist reseat |
| recatch_suffix_16 | recatch | RETAINED_TO_12 | RETAINED_TO_12 | recontact |
| recatch_suffix_24 | recatch | RETAINED_TO_12 | RETAINED_TO_12 | contact-preserving |
| recatch_suffix_28 | recatch | RETAINED_TO_12 | RETAINED_TO_12 | contact-preserving |
| air_lat_pos | airborne | DROP | DROP | none |
| air_lat_neg | airborne | DROP | DROP | none |
| air_fast_pos | airborne | DROP | DROP | none |
| air_slow_neg | airborne | DROP | DROP | none |
| nl_h_dz14.00 | natural | DROP | DROP | none |
| nl_h_dz14.25 | natural | DROP | DROP | none |
| nl_h_dy | natural | DROP | DROP | none |
| nl_h_dx | natural | DROP | DROP | none |

Wrist, sliding, and open-finger rows are states along those verified demonstrations, including the centered −6 mm impact and the +6 mm and lateral impacts. Airborne rows start after a sideways carry and an opening, with both pads already clear. Natural-loss rows are held-out distal-tip slips. The same snaps are used for both policies.

BALANCED UNIFIED RECOVERY IMPROVES OVERALL PHYSICAL SUCCESS
