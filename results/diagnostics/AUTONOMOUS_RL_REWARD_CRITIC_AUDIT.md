# Autonomous RL reward and critic audit

The actor, critic, reward, learning rates, and behavior-cloning weight are the ones from the 40-update sanity run. The geometry switch, 39-D observation, `recovery7d_airborne_v3`, gains, and held-out split were not changed.

The original sanity process had not written its weights. This round replayed that same seed and update rule, then continued. At 40 updates the skill errors match the logged sanity result to about 0.00002. One retained training impact took 30 recovery steps in this replay and 31 in the original log. The physical label was the same. `ckpt_39d.pt` was not overwritten. Saved copies are `results/diagnostics/raw/autonomous_rl/ckpt_u0040.pt` and `ckpt_u0100.pt`.

The frozen position table has 160 entries. A recovery longer than that feeds the network only its latest 160 steps. One failed rollout ran for 454 policy steps and used that limit. Shorter episodes, including the 129-step held-out fall, are unchanged.

Training stopped at 100 updates because the wrist-reseating action error rose from 0.003 to 0.197. It did not continue toward 250, 500, 1000, or 2000.

## Learning curve

Validation chooses the checkpoint. The held-out eight were not used for stopping or selection.

| Updates | Fixed training probe | Episodes since previous checkpoint | Validation | Wrist RMSE | Sliding RMSE | Recatch RMSE | `impact_zm6` | Actor L2 from pretraining | Executed clamp fraction |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 40 | 2/6 | 3/6 retained | 3/5 | 0.0030 | 0.0209 | 0.0137 | retained, 18 steps | 0.168 | 0.083 |
| 100 | 2/6 | 3/9 retained | 2/5 | 0.197 | 0.0211 | 0.0147 | retained, 26 steps | 0.439 | 0.095 |

The fixed probe is four training-box impacts, the training offset dz = −13.5 mm, and the training carry at 0.35 m/s, −y. It excludes `impact_zm6`. Both checkpoints retain two of those six.

Validation at 40 updates is the same 3/5 as the pretrained policy: `impact_z3` and both validation carries retain; both validation offsets drop. At 100 updates `air_lat_pos` drops after 6 recovery steps, so validation is 2/5. Sliding and open-finger action error stay at the pretrained level. Wrist reseating does not.

`impact_zm6` still retains at 100 updates, but the recovery grows from 18 steps to 26.

## Validation-selected checkpoint

The 40-update checkpoint has validation 3/5 and the lower skill error. The 100-update checkpoint has validation 2/5. Selection keeps the 40-update actor.

## Reward

Each 20 ms recovery step adds:

- +0.002 if both pads are in contact
- +0.001 times height above 0.44 m, clipped over 0.20 m
- −0.001 times hand-frame position error, clipped over 0.05 m
- −0.0005 times relative speed, clipped over 0.20 m/s

A retained rollout adds +1.0 on its last recovery transition. A drop adds 0. There is no orientation term and no separate stability bonus. The negative position and speed terms are the only penalties.

Gamma is 0.99. A transition is terminal when the physical t = 12 outcome is known. A fall is a drop and is not bootstrapped. A geometry exit is followed by nominal continuation, and only a real `RETAINED_TO_12` receives +1.0. `truncated` stayed false.

### Return separation

Training rollouts, undiscounted return:

| Group | n | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained | 6 | 1.025 | 1.022 | 1.010 | 1.052 |
| Dropped | 9 | 0.127 | 0.096 | 0.002 | 0.481 |

Every retained return is larger than every dropped return. The same is true after discounting: retained discounted returns run from 0.792 to 0.973, dropped ones from 0.002 to 0.190.

Deterministic rerolls of the same seven disturbances under the pretrained actor, the 40-update actor, and the 100-update actor also have pair accuracy 1.0. The closest call is the 100-update actor on one training impact that never recovered: 454 steps, undiscounted return 0.987, against a lowest success of 1.013. Contact shaping on that failure is 0.906. The +1 retention bonus is what keeps it below the successes. A still longer bilateral failure could close that gap. In this sample it did not.

Shaping alone does not separate the groups. Successful shaping is 0.010 to 0.052. Failed shaping reaches 0.987.

### Term-by-term examples

| Rollout | Steps | Outcome | Contact | Height | Position | Speed | Terminal | Total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training impact, retained | 17 | RETAINED_TO_12 | 0.034 | 0.006 | −0.004 | −0.006 | 1.000 | 1.031 |
| `impact_zm6`, 40-update actor | 18 | RETAINED_TO_12 | 0.036 | 0.007 | −0.004 | −0.006 | 1.000 | 1.032 |
| Training air carry | 5 | RETAINED_TO_12 | 0.010 | 0.004 | −0.001 | 0.000 | 1.000 | 1.013 |
| Short training fall | 4 | DROP | 0.004 | 0.001 | −0.001 | −0.002 | 0 | 0.002 |
| Training offset | 5 | DROP | 0.010 | 0.004 | −0.001 | 0.000 | 0 | 0.012 |
| Long training fall | 235 | DROP | 0.468 | 0.083 | −0.054 | −0.016 | 0 | 0.481 |
| 100-update actor, no recovery | 454 | fall below lift | 0.906 | 0.184 | −0.090 | −0.013 | 0 | 0.987 |

On a short retained recovery the terminal bonus is about 30 times the total shaping. On the 454-step failure, contact shaping is almost as large as that bonus. No dense term dominates the short successes. Contact shaping can dominate a long failure.

### Monotonicity

Over 680 collected recovery steps:

| Comparison | Immediate shaping |
| --- | ---: |
| Both pads in contact | 0.00195 |
| Not both pads | −0.00069 |
| Position error at or below the median 11.5 mm | 0.00203 |
| Position error above that median | 0.00176 |
| Relative speed at or below 0.015 m/s | 0.00202 |
| Relative speed above that | 0.00177 |
| Steps from a rollout that retains | 0.00189 |
| Steps from a rollout that drops | 0.00190 |

Contact, position error, and speed point the right way. The position and speed gaps are a few ten-thousandths per step. Six hundred sixty-six of the 680 steps already have both pads in contact, so the contact bonus is usually on. Immediate shaping does not distinguish a step that belongs to an eventual retention from one that belongs to a drop. Orientation is not rewarded. Splitting on the median rotation changes the mean shaping by about 0.0002.

## Critic

The critic is the same twin MLP. Targets use the minimum of the two heads. The actor maximizes the first head only. There is no entropy bonus. Exploration remains Gaussian noise of standard deviation 0.05 on the executed action.

### Scale

|  | 40 updates | Next 60 updates, through 100 |
| --- | ---: | ---: |
| Mean Q1 | −0.131 | −0.126 |
| Mean Q1 standard deviation | 0.056 | 0.048 |
| Mean target | −0.130 | −0.124 |
| Largest absolute target | 1.003 | 1.003 |
| Mean absolute TD error | 0.076 | 0.043 |
| Median absolute TD error | 0.046 | 0.025 |
| 90th percentile absolute TD error | 0.105 | 0.065 |

Targets stay on the scale of the +1 bonus. Predictions stay near −0.13. The critic has not moved its values onto the retention gap.

Q1 on the frozen 40-update transitions:

| Transitions | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| From retained episodes | −0.125 | −0.142 | −0.189 | −0.005 |
| From dropped episodes | −0.130 | −0.139 | −0.201 | 0.025 |

Those two distributions sit on top of each other.

### Action ranking

At each diagnostic state the critic scores the demonstrated action, the current mean action, and an action taken in a failed impact. The failed action is the nearest one in observation space. Median distance to that source state is 1.23 to 1.78, so this is an action swap at the diagnostic state. A ranking counts only when the better action has strictly higher Q1.

At the selected 40-update critic:

| State set | Pairs | Q1(demo) > Q1(fail) | Q1(policy) > Q1(fail) | Mean Q1 demo | Mean Q1 policy | Mean Q1 fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wrist reseating | 18 | 16 | 16 | −0.141 | −0.141 | −0.159 |
| Inward sliding | 52 | 22 | 21 | −0.132 | −0.132 | −0.122 |
| Open-finger recatch | 32 | 24 | 24 | −0.106 | −0.106 | −0.117 |
| Retained impact states, nearest demo action | 47 | 28 | 29 |  |  |  |
| All of the above | 149 | 90 |  |  |  |  |

Pooled accuracy is 90/149. Sliding is ranked backwards: the failed-episode action has the higher value on 30 of 52 states. The minimum of the two heads agrees, 89/149. At 100 updates the pooled accuracy falls to 59/149, and wrist falls to 6/18.

## Actor movement

At 40 updates the mean absolute action change on the three demonstrations is 0.0009 to 0.0017. On the frozen training states it is 0.006. On the validation rollouts it is 0.002. The actor has barely left the pretrained actions.

At 100 updates the wrist demonstration has moved by 0.032 on average, and its grip component by 0.109. Vertical velocity on that demonstration has moved by 0.067. Sliding and recatch mean absolute changes stay near 0.004. Validation actions have moved by 0.018. Clamp fraction of executed training actions stays near 0.09. The policy is not saturating.

This is a material change on the wrist skill, not a frozen actor. Validation got worse while that change happened.

## Preservation loss against the RL loss

There is no entropy term. The actor loss is `5.0 * behavior-cloning MSE + 0.05 * (−Q1)`.

|  | 40 updates | 100 updates |
| --- | ---: | ---: |
| Behavior-cloning objective | 0.0011 | 0.066 |
| RL objective | 0.0064 | 0.0072 |
| Behavior-cloning gradient norm | 0.0050 | 0.190 |
| RL gradient norm | 0.0045 | 0.011 |

At 40 updates the two gradients have the same size. The cloning term is not holding the actor in place. By 100 updates the wrist error has grown, and the cloning gradient is then about 18 times the RL gradient. That growth is the penalty after the drift, not a constraint that prevented it.

## Replay

Fifteen autonomous episodes were collected. Six retained and nine dropped. None failed to enter. The buffer has 680 recovery transitions:

| Family | Episodes | Retained | Transitions |
| --- | ---: | ---: | ---: |
| Impact | 11 | 4 | 660 |
| Grasp offset | 2 | 0 | 10 |
| Airborne release | 2 | 2 | 10 |

Ninety-seven percent of the transitions are impacts. Twelve percent of transitions come from retained episodes, because the drops last longer. The two airborne successes are 10 transitions, not the bulk of the buffer. Demonstration sequences are a cloning target and are not replay rows. Fifteen transitions are terminal, one per episode.

No held-out episode was inserted.

## Held-out result

Scored once, after selection, on the pretrained actor and the 40-update actor.

| Case | Pretrained | 40-update RL | Mode |
| --- | --- | --- | --- |
| `impact_zp6` | DROP, 5 steps | DROP, 5 steps | geometry exit, then drop |
| `impact_phi90` | RETAINED_TO_12, no entry | RETAINED_TO_12, no entry | nominal retains |
| `air_lat_neg` | RETAINED_TO_12, 5 steps | RETAINED_TO_12, 5 steps | retained |
| `air_slow_neg` | RETAINED_TO_12, 5 steps | RETAINED_TO_12, 5 steps | retained |
| `nl_h_dz14.00` | DROP, 5 steps | DROP, 5 steps | geometry exit, then drop |
| `nl_h_dz14.25` | DROP, 5 steps | DROP, 5 steps | geometry exit, then drop |
| `nl_h_dy` | DROP, 5 steps | DROP, 5 steps | geometry exit, then drop |
| `nl_h_dx` | DROP, fall after 129 steps | DROP, fall after 129 steps | fall before a stable exit |

Held-out t = 12 success is 3/8 for both.

The return does separate retention from drops in this sample, and the +1 bonus is what creates that gap. The critic's values for retained and dropped transitions overlap, and it ranks the demonstrated sliding action below a failed-episode action more often than not. Sixty further updates, taken against that critic, moved the wrist grip and cut validation from 3/5 to 2/5. The next change is the critic update, not a longer run of this objective.

CRITIC DOES NOT RANK RECOVERY ACTIONS CORRECTLY
