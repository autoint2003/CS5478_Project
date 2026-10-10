# Natural-loss recatch data scaling

Adding verified recatch trajectories raises held-out success from 0/24 to 5/24, then stops. At 120 epochs the same Transformer scores **0/24, 5/24, 4/24, and 4/24** with 6, 20, 50, and 100 natural-loss trajectories. The best rate is 5 of 24. Teacher-forced prediction on the held-out slips is already accurate at the larger sizes, and training past 120 epochs does not raise the closed-loop rate.

Every training and held-out trajectory is a clean first loss: slow slip under a closed hold, first bilateral loss that stays lost for 40 ms, distal-tip clearance within +1.0 mm, and a privileged `recovery7d_airborne_v3` intercept that regrasps for 1 s. Center displacement at those losses is 25.4–25.5 mm and the hand-frame separating speed is about +0.05 m/s. Already-separated starts, post-loss snapshots, and the 79 clean losses the intercept cannot regrasp (clearance still about 0, separating speed about +0.34 m/s) were left out. The earlier 36.9 mm state, whose clearance is +5.5 mm, is not in this pool.

The held-out set was frozen before the first training run. On the 0.05 mm vertical-offset grid, every fifth offset is held out, together with −13.4 mm and −14.0 mm, plus three small side perturbations (dy +0.12 mm at −14.10 mm, object vz −0.008 m/s at −13.90 mm, dx +0.10 mm at −14.70 mm). Eight of those candidates passed the oracle test. Each is evaluated from the 0.3 s, 1.0 s, and 2.0 s pre-loss windows, 24 starts in all, for every model. The original six trajectories are the N = 6 subset and sit inside every larger subset.

The model, the K=32 prototypes, the 45-D observation, the v3 map, the optimizer, the batch of 16, the random-suffix training, and the three local-skill trajectories are unchanged. CUDA training used mixed precision on an RTX 3060. Train loss is the weighted random-suffix objective. Validation loss is the unweighted token-plus-residual loss on the held-out 1.0 s suffix, which never enters training. One optimization step is one batch.

## Scaling table

| natural-loss trajectories | epochs | optimization steps | train loss | validation loss | held-out 1 s success | mean \|clearance error\| (mm) | old skills through t=12 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 120 | 1,320 | 0.236 | 0.439 | 0/24 | 0.84 | all three retained |
| 6 | 240 | 2,640 | 0.112 | 0.391 | 2/24 | 0.38 | all three retained |
| 6 | 480 | 5,280 | 0.111 | 0.228 | 3/24 | 1.25 | all three dropped |
| 20 | 120 | 3,840 | 0.217 | 0.216 | 5/24 | 0.30 | all three retained |
| 50 | 120 | 9,240 | 0.114 | 0.182 | 4/24 | 0.25 | all three retained |
| 50 | 240 | 18,480 | 0.078 | 0.113 | 4/24 | 1.11 | all three retained |
| 50 | 480 | 36,960 | 0.059 | 0.137 | 2/24 | 0.62 | all three retained |
| 100 | 120 | 18,240 | 0.085 | 0.090 | 4/24 | 1.33 | all three retained |

Clearance error is the absolute difference between the policy’s distal-tip clearance at its own first persistent loss and the oracle clearance on that same start. The oracle value is about 0.

## What grows with data, and what does not

At a matched 120 epochs, going from 6 to 20 trajectories is the only step that moves closed-loop success (0/24 to 5/24) and cuts the mean clearance error (0.84 mm to 0.30 mm). Fifty trajectories leave success at 4/24 and the clearance error at 0.25 mm. One hundred trajectories leave success at 4/24, and the clearance error rises to 1.33 mm. The pre-loss action error against the hold command does fall, from 0.079 at N = 6 to 0.024 at N = 100. The policy tracks the demonstrated slip more closely, and the cylinder still leaves the demonstrated loss state on most held-out starts.

On the best run (20 trajectories, 120 epochs) the first chase command is the downward cap on every window: normalized `a2` is 0.93–1.0 and the world vertical command is −3.50 to −3.80 m/s. Five windows then hold both pads for 1 s:

| start | clearance at loss (mm) | v_rel,z (m/s) | first command | min distance to seated center (mm) | recontact | reclose |
|---|---:|---:|---|---:|---|---|
| dz −13.40, 2.0 s | +0.02 | −0.31 | −3.77 m/s | 3.8 | yes | yes |
| dz −13.75, 2.0 s | +0.03 | −0.32 | −3.76 m/s | 1.8 | yes | yes |
| dz −14.00, 0.3 s | −0.03 | +0.05 | −3.80 m/s | 8.6 | yes | yes |
| dz −14.25, 2.0 s | +0.04 | −0.32 | −3.77 m/s | 1.7 | yes | yes |
| dz −14.50, 2.0 s | +0.05 | −0.32 | −3.77 m/s | 1.8 | yes | yes |

The other 19 windows also emit that chase command. On the misses the loss clearance is often several tenths of a millimetre to about +1.8 mm, the separating speed is often +0.2 to +0.46 m/s, and the closest the center comes to the seated pose stays 12–33 mm. Recontact happens on 15 of 24 windows and a closing grip on 12; the 1 s hold is the part that stops at 5. The same pattern, with 4 successes, is what N = 50 and N = 100 produce.

## Epochs

On 6 trajectories, validation loss is still moving at 120 epochs (0.44) and is lower at 480 (0.23). Closed-loop success rises from 0/24 to 3/24 over that range, and at 480 epochs wrist reseating, the open-finger recatch, and inward sliding all drop before t = 12.

On 50 trajectories the validation loss is 0.182 at 120 epochs, 0.113 at 240, and 0.137 at 480. Token accuracy is 0.94, 0.96, and 0.97. Residual error stays near 0.011. Closed-loop success is 4/24, 4/24, then 2/24. At 100 trajectories and 120 epochs the held-out token accuracy is 0.975, the action error is 0.013, and the validation loss is 0.090. Further epochs fit the demonstrations more tightly. They do not produce more 1 s regrasps.

The local skills stay retained through t = 12 on every run except the 480-epoch model trained on only 6 trajectories.

The 100 verified trajectories are enough for the supervised chase to be learned. The held-out failures are still the policy’s own slip arriving at a faster or slightly separated loss than the one the intercept was shown. Further copies of the same offline intercept do not close that gap: success is flat from 20 trajectories to 100.

DATA COVERAGE HELPS BUT CLOSED-LOOP SHIFT REMAINS
