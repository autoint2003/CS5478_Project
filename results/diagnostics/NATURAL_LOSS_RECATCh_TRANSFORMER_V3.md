# Natural-loss airborne recatch, transformer v3

The finite −3.8 m/s downward command is enough for a privileged intercept to catch the horizontal cylinder after a natural bilateral slip. A causal Transformer, trained on those catches plus the existing local skills, reproduces the chase when it is shown the demonstration history. In closed loop it completes the full slip-to-regrasp on one nearby state and misses the others. The misses lose contact with the center further below the seated reference than the demonstrations. That center displacement is not a larger air gap: the surface clearance at those losses is still about 0, except when the fingers have crossed.

## Final action mapping

Version `recovery7d_airborne_v3`. Normalized actions from v1 and v2 are not reused under this map. A stored v2 full-down command (`a[2] = +1`, 1 m/s in the hand frame) rewrites to v3 `a[2] = 0.2632`.

On the seated horizontal grasp the hand +z axis is world −z (`R[2,2] = −1`). Positive hand-frame `vz` is therefore downward.

| channel | normalized −1 | normalized +1 |
| --- | --- | --- |
| vx, vy | −1.0 m/s | +1.0 m/s |
| vz | −1.0 m/s hand frame, world +1.0 m/s upward | +3.8 m/s hand frame, world −3.8 m/s downward |
| wx, wy, wz | −4 rad/s | +4 rad/s |
| grip | +2 N·m open | −18 N·m secure |

Zero maps to zero. The upward limit is the verified +1.0 m/s, not a new value. The interface is still a 7-D velocity command. Torque limits are unchanged: ±87 N·m on joints 1–4, ±12 N·m on joints 5–7.

The policy step is 20 ms. Each step rebases the Cartesian position target to the current hand pose and holds the velocity command for that step. Integrating the −3.8 m/s command into the position target runs the setpoint past the cylinder, so that stepper is not used.

## Observation

The input is the original 27-D observation plus the 18-D MuJoCo ground-truth airborne suffix, 45-D total. No RGB-D and no learned perception.

On the successful −13.5 mm demonstration every extra channel stayed finite and matched the clipped formula through contact, natural loss, chase, and recontact (absolute mismatch 0). During the chase, `w_rel_hx` reaches the clip (raw magnitude 2.64 against a scale that saturates at 1). The other 17 extra channels stay inside the scale. That clip is the existing angular-rate scale. It was not changed.

## Natural-loss demonstrations

Each demonstration starts in the seated horizontal grasp, applies a small offset, and holds a closed grip until bilateral contact is lost. The gripper is not opened on purpose. After that loss the same privileged intercept used in the cap study runs under the v3 clip: downward world `vz` at most 3.8 m/s, upward at most 1.0 m/s, `vx` and `vy` at most 1.0 m/s. Reclose is the existing aperture and gap test, then −18 N·m.

A rollout is kept only if it recontacts, recloses, and both pads remain on the cylinder for 1 s (at least 95% of that second, both pads on at the end, excess under 20 mm). Two training windows are cut from each success, starting 1.0 s and 0.3 s before the chase, while both pads are still on. The multi-second slide before that window is not used as a target.

| case | split | loss time (s) | d_center at loss (mm) | d_clearance (mm) | command at loss (m/s) | min d_center (mm) | end pads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dz −13.5 mm | train | 6.566 | 25.47 | 0.0 | −3.8 | −9.0 | 9 / 10 |
| dz −13.3 mm | train | 7.376 | 25.46 | 0.0 | −3.8 | −8.7 | 10 / 10 |
| dz −13.8 mm | train | 5.336 | 25.41 | 0.0 | −3.8 | −9.0 | 10 / 10 |
| dz −14.2 mm | train | 4.474 | 25.45 | 0.0 | −3.8 | −8.4 | 9 / 10 |
| object vz −0.01 m/s on dz −13.5 | train | 6.352 | 25.44 | 0.0 | −3.8 | −9.2 | 10 / 10 |
| dz −13.6 mm | hold | 6.164 | 25.45 | 0.0 | −3.8 | −9.0 | 10 / 10 |
| dz −14.0 mm | hold | 4.662 | 25.49 | 0.0 | −3.8 | −8.4 | 10 / 10 |

That is 5 training conditions and 2 held-out conditions, 14 windows (10 train, 4 hold). Every success loses the cylinder at the same center displacement, about 25.4 mm. The surface clearance there is 0: the distal pad corner is on the 18 mm circle. The first chase command is the cap. The measurement is in `NATURAL_LOSS_RECATCh_COVERAGE_EXPANSION.md`.

Lateral shifts of ±1.0 mm and +1.5 mm, and joint-2 shifts of ±0.002 rad, were already separated at the moment the offset was applied. They were saved as failures and were not given action targets. The earlier ±2 mm and ±0.005 rad states are outside this basin even with an uncapped command, so they were not retried.

## Other skills

Wrist reseating, the 20 ms recatch, and gravity-assisted inward sliding were re-executed under the same v3 map and the same 20 ms stepper, on the trained zm6 impact. All three retain the object through absolute t=12. Their actions are in the supervised set. No skill id is attached.

The opposite-sign zp6 recatch and gravity rollouts are held out. They also retain through t=12 under the demonstration controller.

The existing vertical airborne catch and the two 0.30 m/s lateral catches were recomputed with the v2 oracle, converted into v3, and executed with the same stepper. None of the three recontacted and held. They are not in the training set. Closed-loop airborne evaluation was therefore not run. Those catches remain verified under the stepper they were originally recorded with. They are not verified under the stepper this policy uses.

## Tokenizer

K-means is fit on the physical action, with weights `[25, 25, 2, 2, 2, 2, 3]` on `[vx, vy, vz, wx, wy, wz, grip]`, so small translations dominate large downward commands. The stored prototype is the mean of the true normalized v3 actions in that cluster, and the residual is decoded back to that action.

| | K=16 | K=32 |
| --- | --- | --- |
| empty clusters | 0 | 0 |
| median reconstruction L2 | 0.0086 | 0.0044 |
| local translation MAE | 0.0024 m/s | 0.0 m/s |
| high-vz MAE (hand vz > 1.2 m/s) | 0.118 m/s | 0.054 m/s |
| open / close MAE | 0.030 / 0.002 | 0.0 / 0.0 |
| wrist MAE | 0.019 | 0.0 |
| open and close sharing a token | token 4 | none |
| high-vz sharing a token with a hold | none | none |

K=16 is rejected because open and close share a token. K=32 is the tokenizer used for training. There is no lateral chase in the fitted set, so that reconstruction number is absent. High downward speed is coarser than a local move, by 0.054 m/s, and it does not land in the hold token.

## Training

CUDA was used.

| | |
| --- | --- |
| torch | 2.14.0+cu126 |
| CUDA available | yes |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| CUDA | 12.6 |
| cuDNN | 91002 |
| device | cuda |

The model is the established causal Transformer: 45-D observation plus the previous 7-D action, width 64, 2 layers, 4 heads, feed-forward 128. It outputs one categorical token and a continuous residual bounded by `0.25 tanh`. Training uses random suffixes, the position index restarts on each suffix, length capped at 96, batch 16, Adam at 1e-3, gradient clip 1.0, 120 epochs. Eight random suffixes are drawn from each trajectory each epoch. Steps with a large downward command, an open grip, or a large wrist rate are weighted by 4 in the token loss. Mixed precision stayed on. Batches are pinned and copied with non-blocking transfers. Loss went from 3.42 at epoch 1 to 0.33 at epoch 120. Wall time for the training loop was 13.8 s. Checkpoints are every 10 epochs through `ckpt_120.pt`.

No skill id, loss-transition module, release predictor, SAC, PPO, chunk tokenizer, diffusion, VAE, or Best-of-K.

## Teacher forcing

On every natural-loss window, train and hold-out, the first chase step is predicted as `a[2] = 1.0`, which is the demonstrated full-down command. Action MAE on those windows is 0.011 to 0.075. The trained local skills are 0.07 (gravity), 0.08 (wrist), and 0.17 (recatch). The held-out zp6 recatch and gravity sequences are not fit: MAE 0.80 and 1.15, and the gravity vertical channel alone is off by 0.25.

## Closed loop

Evaluation starts from the saved window, while both pads are still on. The intercept is not called. The Transformer is the only controller.

The row logged at the first both-pads-off sample is the action chosen from the previous observation, which often still has contact. A command near zero on that row does not by itself mean the dive never happened. The dive is the peak command and the hand speed over the following steps.

| case | d_center at both-off (mm) | d_clearance (mm) | peak command (m/s) | hand vz (m/s) | hand az (m/s²) | command at closest center | min d_center (mm) | recontact | reclose | 1 s hold | end pads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dz −13.5, 1.0 s | 27.8 | +0.06 | −3.8 | — | — | — | 17.0 | no | no | no | 0 / 0 |
| dz −13.5, 0.3 s | 27.2 | +0.10 | −3.8 | −1.75 | −45.8 | −0.91 | 12.2 | no | no | no | 0 / 0 |
| dz −13.3, 1.0 s | 36.9 | +5.51 | −3.8 | — | — | — | 17.0 | yes | no | no | 0 / 0 |
| dz −13.8, 1.0 s | 27.9 | +0.20 | −3.8 | — | — | — | 17.0 | no | no | no | 0 / 0 |
| dz −14.2, 1.0 s | 21.5 | +0.38 | −3.8 | −1.82 | −45.7 | −0.90 | 10.7 | no | no | no | 0 / 0 |
| dvz −0.01, 1.0 s | 22.8 | +0.04 | −3.8 | −1.80 | −45.8 | −0.66 | −7.8 | yes | yes | no | 0 / 0 |
| dz −13.6, 1.0 s, hold | 20.1 | +0.39 | −3.8 | −1.83 | −45.6 | −0.91 | 8.6 | no | no | no | 0 / 0 |
| dz −13.6, 0.3 s, hold | 27.8 | +0.01 | −3.8 | −2.06 | −45.6 | −3.45 | 14.2 | no | no | no | 0 / 0 |
| dz −14.0, 1.0 s, hold | 31.2 | +0.37 | −3.8 | — | — | — | 17.0 | yes | no | no | 0 / 0 |
| dz −14.0, 0.3 s, hold | 25.5 | 0.00 | −3.8 | −1.83 | −45.7 | −0.90 | −8.4 | yes | yes | yes | 10 / 10 |

Hand speed and acceleration are omitted when the closest excess is still the initial seat. In those rollouts the object is already clear before the hand has accelerated, so an approach-window peak would describe the slip rather than the dive. Peak command on those rows is still −3.8 m/s, issued late, for a few steps.

The one hold is the held-out −14.0 mm cylinder, started 0.3 s before loss. `d_center` at loss is 25.5 mm and `d_clearance` is 0, matching the demonstrations. Peak command is the cap, peak hand speed is −1.83 m/s, peak acceleration is −45.7 m/s², and joints reach the 87 N·m stop for 2 steps. At the closest point the command has fallen to −0.90 m/s (`a[2] = 0.26`) and the relative vertical speed is +0.04 m/s. The fingers reclose and both pads stay on for 1 s, ending at 10 and 10. That is interception with a backoff, not a command held at the cap.

The same backoff, to about −0.9 m/s, appears on the near misses that do dive (12 mm, 11 mm, and 9 mm short). One held-out short window is still near the cap at its closest point (−3.45 m/s) and remains 14 mm short. The dvz training case does intercept and reclose, then loses both pads before 1 s.

On the long windows both pads come off with `d_center` at 28–37 mm, against 25.4 mm in every demonstration. The surface clearance on those rows is still within 0.4 mm of the distal corner, except the 36.9 mm row, which is 5.5 mm clear. The policy's own actions during the slip move the loss out of the center displacement the intercept was fitted to.

## Old-skill retention

Closed loop on the three trained zm6 skills, scored by the physical t=12 rule: wrist reseat, recatch, and gravity-assisted sliding all finish `RETAINED_TO_12` with both pad counts at 10. The largest normalized translation on those rollouts is 0.05.

The held-out zp6 recatch and gravity rollouts drop the object within 16 steps (0.32 s). On both, the largest normalized vertical command is +1, which is a −3.8 m/s world command. Those two sequences were not in the training set, and teacher forcing already misses them.

Vertical and lateral airborne catches were not scored. The demonstration controller did not produce a successful trajectory under this stepper, so there was no policy target and no closed-loop case.

## Failure attribution

The token reconstruction is not the failure. K=32 separates open from close, separates the high downward command from a hold, and rebuilds local, open, close, and wrist actions exactly. High downward speed is within 0.054 m/s.

The Transformer does initiate the dive when the loss gap is near the demonstrated 25 mm. Teacher forcing emits `a[2] = 1` on the first chase step of every window. In closed loop the successful catch and the near misses reach about −1.8 m/s and −46 m/s², the same actuator-limited spike as the privileged intercept, and the command then falls.

The misses split three ways.

- On the 1 s windows the slip under the policy's own actions loses contact at 28–37 mm. The dive, when it comes, is late. That is closed-loop covariate shift.
- When the dive does happen near the right time, the hand stops 8–12 mm short, with the command already backed off to about −0.9 m/s. That is interception timing inside a basin the dataset only samples at one gap.
- The dvz case reaches the cylinder and recloses, then does not keep both pads for 1 s. That is a reclose-retention failure.

Coverage is the limit on all three. The successful set is one vertical-offset family at one loss gap. Lateral starts, joint-2 starts, and the existing airborne catches are absent, the first two because they were already outside the basin, the last because this stepper did not reproduce them. The held-out zp6 local failures are the same hole: an unseen impact sign is answered with the full downward command.

The architecture was not changed.

NATURAL-LOSS RECATCh IS LEARNABLE BUT NEEDS MORE DATA COVERAGE
