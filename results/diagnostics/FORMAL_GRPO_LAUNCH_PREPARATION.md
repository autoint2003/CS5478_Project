# Formal GRPO launch preparation

The actor term now averages the log-probability over valid recovery steps. A length check and an 8-update closed-loop check were run. Formal training was not started.

## Normalization

The group-relative advantage is still one number per rollout. The actor surrogate uses the mean log-probability of the recovery steps in that rollout:

```
trajectory_logp_mean = sum_t valid_t * log pi(a_t | h_t) / sum_t valid_t
ratio = exp(trajectory_logp_mean - behavior_mean)
L_GRPO = -min(ratio * A, clip(ratio, 0.8, 1.2) * A)
```

`A` is `(return - group mean) / group std`. The return is 1 only for `SUCCESS_10S` or `SUCCESS_12S`, and 0 otherwise. A group with identical returns gets advantage 0 and no GRPO term. The clip and the advantage formula are unchanged. The behavior term is the mean of the stored per-step log-probabilities, so an on-policy ratio starts at 1 for every length.

Retention is unchanged: categorical KL to the K=35 reference, coefficient 0.05, mean over the first 8 recovery steps, and only on non-airborne groups. There is no behavior-cloning term and no critic.

## Per-step log-probability

For a recovery step with history `h_t`, sampled token `k_t`, and pre-clip residual `pre_t`:

```
log pi(a_t | h_t)
  = log softmax(logits(h_t))[k_t]
    + sum_{d=1..7} log Normal(pre_t,d ; mu_t,d(k_t), 0.05^2)
```

`mu` is the token-conditional residual head: the token-35 head when `k_t` is 35, otherwise the shared head. The token term is one categorical probability. Each of the seven residual dimensions is counted once. `pre_t = mu_t + epsilon` is the Gaussian sample. Clipping that sample to ±0.25, adding the codebook vector, and clipping the command to [-1, 1] are deterministic and are not assigned probability.

A direct comparison on one step matched `torch.distributions.Normal` over all 7 dimensions. The log-probability of the clipped residual differed from the log-probability of `pre_t`, which is the density the update uses.

## Valid steps

The denominator is the number of actions the recovery policy actually sampled. Those are the only records stored. The mean does not divide by the cap of 100.

Excluded from the mean:

- nominal control after the explicit handoff
- the 10 s and 12 s nominal validation
- steps after a terminal drop (the dropping action itself is included, and the loop then stops)
- padded positions (the window is the real history, cropped at 160, and only the current index is scored)
- scripted entry actions (the rollout starts from the stored capture)

On this check every successful member had between 12 and 41 recovery steps. A 10 s hold would be 500 steps, so the hold was not inside the actor term.

## Gradient scale

Identical synthetic rollouts, advantage 1, same policy weights, no optimizer step:

| valid steps | summed log-prob | mean log-prob | old actor grad | new actor grad | old / new |
|---:|---:|---:|---:|---:|---:|
| 4 | 42.154 | 10.539 | 1197.87 | 299.47 | 4.00 |
| 16 | 168.616 | 10.539 | 4791.47 | 299.47 | 16.00 |
| 64 | 674.465 | 10.539 | 19161.19 | 299.47 | 63.99 |

The new gradient stays at 299.47 while the old gradient scales with length. On varied steps the new gradient was 213, 153, and 143 at lengths 4, 16, and 64. It does not grow with length. The length check did not change any trainable weight. Policies A and B were not modified; their hashes matched before and after.

The same 8 train-group setup as the sanity study, weak KL 0.05, recomputed on the samples of this run:

| update | case | returns | advantages | valid steps | mean GRPO grad | sum GRPO grad | KL grad |
|---:|---|---|---|---|---:|---:|---:|
| 2 | air_a | 1, 1, 0, 0 | +1, +1, -1, -1 | 21, 14, 68, 35 | 22.32 | 481.06 | 0 |
| 3 | imp_c | 0, 1, 0, 0 | -0.58, +1.73, -0.58, -0.58 | 100, 41, 19, 100 | 14.84 | 620.83 | 4.57 |
| 5 | air_b | 0, 0, 1, 1 | -1, -1, +1, +1 | 91, 55, 17, 12 | 23.10 | 576.79 | 0 |

Success and failure have opposite signs. All-failure groups (updates 1, 4, 6, 7, 8) produced GRPO gradient 0. Every logged loss and gradient was finite. Airborne groups still have KL gradient 0, because they are outside the reference scope.

On the mixed impact group the mean-GRPO norm is 14.84 and the KL norm is 4.57, a ratio of 3.25. The old summed term on those same samples was 620.83, about 42 times larger. The KL coefficient was not changed. KL is large enough to enter the update, and GRPO is still the larger term. The pre-clip combined norm on that group was 14.78, so the existing norm clip of 1 still sets the step length and now mixes the two directions instead of a length-scaled GRPO direction.

## Eight-update behavior check

Deterministic rollouts, canonical success labels. This was not used to select a checkpoint, and the weights were not saved.

| case | before | after 8 updates | token 35 after | opening tokens after |
|---|---|---|---:|---|
| slip_a | DROP_DURING_RECOVERY | SUCCESS_12S | 0 | 34, 34, 0, 9, 0, 33, 33, 33 |
| slip_b | SUCCESS_12S | SUCCESS_12S | 0 | 34, 34, 0, 9, 0, 33, 33, 33 |
| air_a | SUCCESS_12S | SUCCESS_12S | 1 | 34, 35, 0, 0, 0, 0, 0, 0 |
| air_b | SUCCESS_12S | SUCCESS_12S | 1 | 34, 35, 0, 0, 0, 0, 0, 0 |
| imp_c | DROP_DURING_RECOVERY | SUCCESS_12S | 0 | 32, 32, 0, 0, 14, 0, 0, 0 |

Airborne did not lose token 35. Slip did not leave the two-step token-34 opening. No token id left the K=36 range, and no value was NaN or Inf. One failed stochastic `slip_d` sample used token 35 once; that group was all failures, so its advantage was 0.

## Cleanup

Superseded checkpoints, both archive directories, duplicate videos, and the intermediate reports that are not part of the final narrative were deleted. The launcher still reads policy A and policy B. Details are in `CLEANUP_MANIFEST.md`.

## Frozen policies

| role | path | sha256 |
|---|---|---|
| A. K=35 reference | `results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt` | `f0beb2d857a88c034002f8e71b14458014f50340131ee729b49ae0955a5181c7` |
| B. K=36 initialization | `results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt` | `904d183b4d24527654967b44b6563036901a942c2a3a967c302e57c3f7523bd9` |

The byte-identical sources `learned_nominal_restoration/ckpt.pt` and `future_state_ablation/ckpt_l0.1_s2.pt` were removed in the hard cleanup. These two copies are the files the launcher loads.

Formal training loads the copies and writes only under `results/diagnostics/raw/formal_grpo/`. It does not rewrite A, B, or C.

## Active configs

- `config/formal_grpo.yaml` — launch settings. 120 updates, validation every 12, KL 0.05, learning rate 3e-5, clip 0.2, group 4, residual noise 0.05, gradient clip 1.0. The script refuses to run if those objective values differ from the code.
- `config/recovery_success.yaml` — orientation 0.117 rad, pinch position 3.03 mm, relative speed 0.020 m/s, persistence 0.100 s, primary hold 10 s, diagnostic hold 12 s.
- `config/geometry_switch.yaml` — geometry entry, unchanged.
- `config/nominal.yaml` and `config/sim.yaml` — nominal controller and simulator used by the training environment.

## Train, validation, and final test

Train cycle, in this order: `slip_a`, `air_a`, `imp_c`, `slip_b`, `air_b`, `slip_d`, `imp_a`, `seat_a`, `slip_c`, `imp_b`, `seat_b`, `imp_d`.

Validation rollouts used for checkpoint selection: `imp_e`, `slip_e`, `slip_f`, `seat_c`, `air_c`. `imp_f` has no geometry entry in the frozen readiness set, has no capture, and is not rolled. The selected file is the earliest checkpoint with the highest validation success count, written to `selected_validation.pt`.

Final test is not loaded and is not scored: `imp_g`, `imp_h`, `slip_g`, `slip_h`, `seat_d`, `air_d`. `imp_h` also has no capture. Asking the trainer for any of these ids raises before a rollout.

## Manual launch

From the project root:

```
python -u training/formal_grpo.py --config config/formal_grpo.yaml
```

That command was not executed.
