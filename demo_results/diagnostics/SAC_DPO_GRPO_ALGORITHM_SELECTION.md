# SAC, DPO, and GRPO for autonomous recovery

All three methods started from the same immutable `ckpt_39d.pt`. The file timestamp was unchanged at the end. The frozen pipeline was nominal control, geometry entry, the 39-D causal Transformer, stable-grasp exit, nominal continuation, and absolute t = 12. The reward was the existing shaping plus a +1 bonus only for `RETAINED_TO_12`. No method was tuned on the held-out eight.

The shared budgets that finished are 104 and 256 branched simulator rollouts. A 2000-rollout budget was not run. Each branch is a full recovery through absolute t = 12, so 256 branches already cost 494,602 MuJoCo steps. G = 16 was not run.

## What was held fixed

| Item | Setting |
| --- | --- |
| Actor | causal Transformer, K = 32, residual `0.25 * tanh`, clamp `[-1, 1]` |
| Reference | frozen `ckpt_39d.pt` for DPO and for the GRPO KL term |
| Exploration | token drawn from softmax logits; residual `N(0, 0.05²)`; executed action clamped |
| Evaluation | deterministic argmax token plus residual |
| Entry | hand-frame position error > 6 mm for 40 ms |
| Exit | bilateral contact, position error < 14.7 mm, orientation error < 0.35 rad, relative speed < 0.08 m/s, stable 100 ms |
| SAC critic | instantaneous twin MLP, 10-step TD, γ = 0.99 |
| SAC actor | preservation weight 5.0, RL weight 0.05, actor step 3e-5, critic step 1e-3 |
| DPO | β = 0.1, fixed. Pair only if retained return exceeds dropped return by at least 0.5. At most 4 pairs per group |
| GRPO | G = 8, clip 0.2, KL weight 0.01. Advantage is 0 unless the group mixes retention and drop, or a same-outcome return span is at least 0.5 |
| Skill stop | revert the actor if any demonstration RMSE exceeds 0.15 |

DPO uses

```
L = -log sigmoid( β * [ (log π(τ+) - log π_ref(τ+)) - (log π(τ-) - log π_ref(τ-)) ] )
```

with the trajectory log-probability equal to the sum of token log-softmax and Gaussian residual log-probability. The signs in `train_dpo` match that expression.

Checkpoint choice, before any held-out roll, was validation successes, then whether `impact_zm6` stayed retained, then lower demonstration RMSE.

## Pretrained baseline

Validation is 3/5: `impact_z3`, `air_lat_pos`, and `air_fast_pos` retained; `nl_dz14.6` and `nl_dvz_p` drop. `impact_zm6` is retained in 18 geometry-exit steps. Demonstration RMSE is wrist 0.0017, inward sliding 0.0218, open-finger recatch 0.0138. Action saturation on those demonstrations is 0.149.

## Regime 1 — one shared branch pool

Seed 21. Every entered disturbance produced 8 physical branches. SAC transitions, DPO pairs, and GRPO groups were taken from those same branches. No-entry episodes did not occur.

| | Small (13 groups) | Medium (32 groups) |
| --- | ---: | ---: |
| Autonomous rollouts | 104 | 256 |
| MuJoCo steps | 168,786 | 494,602 |
| Policy steps | 6,018 | 12,255 |
| Retained branches | 18 | 62 |
| Dropped branches | 86 | 194 |
| Mixed retained/drop groups | 3 / 13 (23%) | 11 / 32 (34%) |
| All retained | 1 | 2 |
| All dropped | 9 | 19 |
| Same outcome, return span ≥ 0.5 | 0 | 0 |
| Groups with a nonzero GRPO advantage | 3 / 13 | 11 / 32 |

Most groups are all drops. No homogeneous group had a return span of 0.5, so dense shaping was not used as relative supervision.

### SAC on the shared pool

Eighty critic and actor updates at both budgets. Skills stayed under the stop. Wrist RMSE moved from 0.0017 to 0.0040 (small) and 0.0043 (medium). Token choices on the three demonstrations did not change. Action L2 versus the reference was 0.0063 and 0.0069. Saturation stayed 0.149.

Closed-loop outcomes did not change. Validation stayed 3/5. `impact_zm6` stayed retained. Nothing new was solved, and nothing previously solved was lost.

Critic diagnostics after those 80 updates:

| | Small | Medium |
| --- | ---: | ---: |
| Training wall | 40.9 s | 40.8 s |
| Critic loss | 0.0099 | 0.049 |
| Preservation loss | 0.00021 | 0.00021 |
| RL term, mean −Q | −0.043 | −0.062 |
| Mean Q on retained transitions | 0.066 | 0.111 |
| Mean Q on dropped transitions | 0.031 | 0.051 |
| Mean absolute TD error | 0.029 | 0.064 |
| Same-entry action ranking | 3 / 26 | 72 / 110 |

The return gap between retention and drop is about 0.5 or larger. The fitted Q gap is a few hundredths, and both means sit near 0. Ranking the first action of a retained branch above the first action of a dropped branch from the same entry is 3/26 on the small pool and 72/110 (65%) on the medium pool. That is not a reliable action-ranking signal.

### DPO on the shared pool

| | Small | Medium |
| --- | ---: | ---: |
| Retained-versus-drop pairs available | 7 | 84 |
| Pairs used | 4 | 32 |
| Success-versus-success pairs used in the loss | 0 | 0 |
| Optimizer steps | 16 | 128 |
| Training wall | 23.7 s | 26.5 s |
| Final loss | 0.273 | 0.083 |
| Raw sequence-logprob accuracy | 0 / 16 comparisons | 44 / 128 (34%) |
| Mean raw log π(τ+) − log π(τ−) | −35.8 | −146 |
| Mean per-step log π − log π_ref | −0.75 | −1.87 |
| Token disagree | 0 | 0.006 |
| Action L2 | 0.0068 | 0.040 |
| Sliding RMSE | 0.022 | 0.080 |
| Validation | 3/5 | 2/5 |

The raw sequence log-probability of the retained continuation is usually lower because those continuations are not the same length as the drops. The DPO loss, which compares each trajectory with the frozen reference, did decrease from about 0.64 toward 0. The physical result does not follow that decrease.

The small-pool policy leaves every validation case and `impact_zm6` where the pretrained policy left them. The medium-pool policy keeps `impact_zm6` but drops `impact_z3`, which the pretrained policy retained. No validation case is newly solved. Sliding RMSE rises from 0.022 to 0.080, still under the 0.15 revert threshold. Saturation falls from 0.149 to 0.108. The extra pairs change the actor, and the change removes a solved validation impact.

### GRPO on the shared pool

| | Small | Medium |
| --- | ---: | ---: |
| Optimizer steps | 12 | 44 |
| Training wall | 41.9 s | 57.0 s |
| Nonzero-advantage fraction | 0.231 | 0.344 |
| Mean within-group return std | 0.050 | 0.104 |
| Clip fraction on nonzero-advantage samples | 0.56 | 0.63 |
| Token disagree | 0 | 0 |
| Action L2 | 0.0023 | 0.0044 |
| Validation | 3/5 | 3/5 |

About two-thirds of the medium groups contribute advantage 0. On the samples that do update, more than half the probability ratios leave the clip interval `[0.8, 1.2]`. The actor's token choices on the demonstrations still do not change, and no closed-loop outcome changes. `impact_zm6` stays retained. Wrist, sliding, and recatch RMSE stay at 0.002, 0.022, and 0.014.

## Regime 2 — online collection

Each method collects from its own current actor, then updates, under a cap of 13 groups (104 branches). The disturbance stream is one NumPy generator, seed 22, consumed in order SAC, then DPO, then GRPO. The three online runs therefore do not see the same disturbances. Branch success rates in this regime are not a matched comparison. The shared pool above is the matched comparison.

| | SAC | DPO | GRPO |
| --- | ---: | ---: | ---: |
| Groups collected | 8, then stopped | 13 | 13 |
| Rollouts | 64 | 104 | 104 |
| MuJoCo steps | 105,126 | 224,201 | 301,169 |
| Retained / dropped branches | 8 / 56 | 32 / 72 | 52 / 52 |
| Mixed groups | 0 / 8 | 4 / 13 | 6 / 13 |
| Same-outcome span ≥ 0.5 | 0 | 0 | 1 |
| Wall, collection plus updates | 167 s | 172 s | 239 s |
| Validation after the kept weights | 3/5 | 3/5 | 3/5 |
| `impact_zm6` | retained | retained | retained |

Online SAC had no mixed group in the 8 groups it collected, so it also had no retained-versus-drop pair and no same-entry ranking set. On the eighth group the wrist RMSE reached 0.201. Those weights were discarded. The kept actor is the one from before that update: wrist RMSE 0.004, action L2 0.0059, validation still 3/5. Five of the 13 groups were not collected. The critic-driven actor step is the same failure seen in the earlier learning curve: the update can erase the wrist demonstration before it solves a new case.

Online DPO trained one epoch on each new retained-versus-drop batch only. Sixteen pairs were used, from 41 available. Four groups produced a batch. Wrist RMSE rose to 0.071 and token disagreement on the demonstrations rose to 0.019. `impact_zm6` and the three solved validation cases stayed retained. The two validation offsets stayed drops. The last batch still had raw sequence-logprob accuracy 0 and margin −47.

Online GRPO is the only run with a same-outcome group whose return span reached 0.5, and it has the highest mixed-group fraction in the study (6/13, plus that one spread group: 7/13 with a nonzero advantage). The last group had advantage 0, clip fraction 0, and no optimizer step. Earlier groups did take a clipped step. The kept actor is almost the reference: token disagreement 0, action L2 0.0017, demonstration RMSE indistinguishable from the pretrained policy at the reported precision of the physical rollouts. Validation stays 3/5.

## Held-out test

One checkpoint per method was chosen from validation only, then rolled once on the frozen eight.

| Method | Chosen run | Validation | Held-out |
| --- | --- | --- | --- |
| Pretrained | `ckpt_39d.pt` | 3/5 | 3/8 |
| SAC | fixed, 104 rollouts | 3/5 | 3/8 |
| DPO | fixed, 104 rollouts | 3/5 | 3/8 |
| GRPO | online, 104 rollouts | 3/5 | 3/8 |

GRPO's online checkpoint won its tie only because its demonstration RMSE is slightly lower than the fixed GRPO checkpoints. All three of those GRPO policies, and the selected SAC and DPO policies, have the same validation rows as the pretrained policy.

Held-out rows, identical for the pretrained policy and all three selected checkpoints:

| Case | Outcome | Exit |
| --- | --- | --- |
| `impact_zp6` | drop | geometry |
| `impact_phi90` | retained | no geometry entry |
| `air_lat_neg` | retained | geometry |
| `air_slow_neg` | retained | geometry |
| `h_dz14.00` | drop | geometry |
| `h_dz14.25` | drop | geometry |
| `h_dy` | drop | geometry |
| `h_dx` | drop | fall abort |

`impact_phi90` is retained by nominal control because geometry never enters. No selected policy creates a new recovery success. No selected policy loses `air_lat_neg` or `air_slow_neg`.

## Failure modes

On every selected checkpoint the mean action L2 versus `ckpt_39d.pt`, measured on the three demonstrations, is below 0.02, and the token disagreement is 0. The validation drops (`nl_dz14.6`, `nl_dvz_p`) and the held-out drops were already drops under the pretrained policy. The failure mode is that the update did not change the pretrained behavior.

Separate from the selected checkpoints:

- Fixed medium DPO does change the actor. It loses `impact_z3` and moves inward-sliding RMSE from 0.022 to 0.080. That is a previously solved validation case entering a dropping recovery, with a real drift in the sliding demonstration, still short of the 0.15 revert line.
- Online SAC does destroy the wrist demonstration (RMSE 0.201) when the critic is allowed to keep stepping. The revert restores the previous actor, which again matches the pretrained outcomes.

No run shows a wrong-token takeover on the demonstrations. Saturation does not rise on the selected checkpoints.

## Main comparison

Selected checkpoints, plus the pretrained policy. Fixed-data and online numbers above are the full grid; this table is the checkpoint that validation actually selected.

| Method | Simulator rollouts | Simulator steps | Training physical success | Validation success | Old-skill retention | Policy change / KL | Signal quality | Held-out success | Main failure mode |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 0 | 0 | `impact_zm6` retained | 3/5 | wrist 0.002, slide 0.022, recatch 0.014; closed-loop zm6 retained | none | — | 3/8 | two validation offsets and five held-out drops already |
| SAC | 104 | 168,786 | `impact_zm6` retained | 3/5 | wrist 0.004, slide 0.021, recatch 0.014; zm6 retained | token disagree 0; action L2 0.006 | same-entry rank 3/26; Q 0.066 vs 0.031 | 3/8 | failed to change the pretrained behavior |
| DPO | 104 | 168,786 | `impact_zm6` retained | 3/5 | wrist 0.003, slide 0.022, recatch 0.014; zm6 retained | action L2 0.007; per-step log-prob gap −0.75 | 4 retained-vs-drop pairs; raw margin −35.8 | 3/8 | failed to change the pretrained behavior |
| GRPO | 104 | 301,169 | `impact_zm6` retained | 3/5 | wrist 0.002, slide 0.022, recatch 0.014; zm6 retained | token disagree 0; action L2 0.0017 | mixed groups 6/13; nonzero advantage 7/13 | 3/8 | failed to change the pretrained behavior |

Optimizer updates on those selected runs: SAC 80, DPO 16, GRPO a handful of clipped group steps with the last group unused. Wall clock for the whole study was 1,459 s. Fixed-pool training itself was 24–57 s per method. Online runs were 167–239 s because they include the extra simulation.

Across the larger shared budget, SAC at 256 rollouts is still validation 3/5 with rank 72/110, and DPO at 256 rollouts falls to validation 2/5. Those rows are not in the selected table because validation rejected them. They are the evidence that more of the same update does not turn into a new t = 12 success.

## Decision

None of the three raises autonomous t = 12 success over `ckpt_39d.pt` on validation or on the held-out eight. The skills that the selected checkpoints keep are the skills the pretrained policy already had. SAC's 10-step critic still does not rank same-entry actions reliably, and a further online actor step deletes the wrist skill. DPO can obtain retained-versus-drop pairs from the shared branches, but the small set does not move the closed loop, and the larger set moves it by losing `impact_z3`. GRPO's group-relative signal is absent on most groups; where online groups do mix outcomes, the clipped update leaves the deployed tokens and the physical cases unchanged.

NONE OF THE THREE IMPROVES RELIABLY OVER THE PRETRAINED POLICY
