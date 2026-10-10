# Controlled DAgger for natural-loss recatch

DAgger starts from the 120-epoch supervised model on the same 100 clean natural-loss trajectories. Torch seed 1 is the starting point because that seed keeps wrist reseating, the open-finger recatch, and inward sliding through t=12. The round-0 policy regrasps 3/24 of the frozen held-out windows. The best ordinary supervised rate on this set is 5/24.

The held-out windows are the same 24 clean, oracle-recatchable pre-loss starts used in the data-scaling study. No DAgger state was collected from them. A collected state is kept only when a privileged continuation from that state still reaches a first loss on the distal tip and a 1 s bilateral regrasp. Family names balance the batches and are not given to the Transformer.

## Per-round results

| round | new DAgger states | natural-loss success / 24 | controlled-sliding success | wrist success | existing-recatch success | mean first-loss clearance (mm) | mean separating speed (m/s) | high-vz false positives | training mix |
|---:|---:|---:|---|---|---|---:|---:|---|---|
| 0 | 0 | 3/24 | 5/5 | 2/2 | 2/2 | 0.219 | 0.014 | 0/11 | s0.003 w0.003 r0.003 n0.990 d0.000 |
| 1 | 1 | 3/24 | 5/5 | 2/2 | 2/2 | -0.012 | -0.055 | 0/11 | s0.20 w0.15 r0.15 n0.30 d0.20 |

## What the rounds changed

Closed-loop success stays 3/24. Round 0 holds dz −14.25 at 1.0 s, dz −14.50 at 0.3 s, and the dy held-out case at 1.0 s. Round 1 holds dz −14.00 at 0.3 s and 1.0 s, and dz −14.50 at 1.0 s. The count does not rise, and it stays below the supervised peak of 5/24. Recontact on round 1 is 15/24 and reclose is 14/24. The first chase command is still the −3.8 m/s cap, and the mean time spent at or below −3 m/s is 0.065 s. Mean pre-loss action error against the hold is 0.038. Basin exits go from 12/24 to 14/24.

Round 1 kept one corrective trajectory, a pre-drift state on a training slip whose slide suffix lost contact and whose v3 continuation still regrasped. A second collection then queried 17 more drifted training states and kept none, so aggregation stopped.

## Sampling and optimization

Round 0 samples every trajectory eight times per epoch, which is the supervised recipe. Later rounds keep that same step count and give sliding, wrist, existing recatch, the original natural-loss demonstrations, and the DAgger corrections fixed shares of about 0.20, 0.15, 0.15, 0.30, and 0.20. Growing the correction set does not shrink the sliding share. The original demonstrations stay in every round.

Round 0: 120 epochs, 18240 optimization steps, loaded skill-retaining checkpoint on NVIDIA GeForce RTX 3060 Laptop GPU. Trajectories in the mix: {'sliding': 1, 'wrist': 1, 'recatch': 1, 'natural': 300, 'dagger': 0}. Suffixes per epoch: {'sliding': 8, 'wrist': 8, 'recatch': 8, 'natural': 2400, 'dagger': 0}. Train loss 0.0783.
Round 1: 120 epochs, 18240 optimization steps, 313.6 s on NVIDIA GeForce RTX 3060 Laptop GPU. Trajectories in the mix: {'sliding': 1, 'wrist': 1, 'recatch': 1, 'natural': 300, 'dagger': 1}. Suffixes per epoch: {'sliding': 486, 'wrist': 365, 'recatch': 365, 'natural': 730, 'dagger': 486}. Train loss 0.0585.

Across the rounds the collector considered 53 drifted pre-loss states, of which 1 still had a successful oracle continuation. 5 were near the controlled-sliding cloud. 0 admitted both a contact-preserving slide suffix and a recatch continuation. 0 kept only the slide suffix. 52 were dropped because the continuation already failed.

## Held-out failure location

On round 1, 14 of the 24 windows reach a first loss that the v3 intercept can no longer regrasp. The other 7 misses still have an oracle-recatchable loss and then fail in the chase or the 1 s hold. Every window does lose contact. Mean distal-tip clearance at the policy's first loss is −0.012 mm and the mean separating speed is −0.055 m/s.

## Controlled-sliding protection

The frozen sliding cases are the nominal inward slide, restarts at demonstrated steps 20 and 36, and 1 mm shifts in y and z. All five stay retained through t=12, as do both wrist cases and both open-finger recatch cases. On the nominal slide the grip command changes from one step at −1.0 to 24 steps at −0.48, which is the demonstrated relax level. Excursion on that case falls from 8.2 mm to 6.6 mm, contact loss does not increase, and the world-z command stays off the downward cap. The same pattern appears on the 1 mm shifts. Across the nine skill cases and the two opposite-sign probes, no case emits a world vertical command at or below −1 m/s. Five collected states were near the sliding cloud. None of them still admitted the demonstrated slide suffix, so no recatch label was written onto a state whose contact-preserving continuation succeeded.

DAgger DOES NOT IMPROVE NATURAL-LOSS RECATCh
