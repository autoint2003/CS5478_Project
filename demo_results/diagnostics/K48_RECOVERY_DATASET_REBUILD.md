# K48 recovery dataset rebuild

New files are under `demo_results/diagnostics/raw/k48_recovery_v3/`. Historical snapshots, the original 534-trajectory manifest, checkpoints, and FINAL were not modified.

A state is in the recovery pool only when immediate HANDOFF fails the unchanged 12-second grasp check and a legal K48 action sequence replays to `SUCCESS_12S`. Unsuccessful searches remain in `unknown.jsonl` and are not labeled unrecoverable.

## Counts

Revalidated existing demonstrations: 106/106 `SUCCESS_12S` on the original 35 confirmed-recoverable TRAIN states.
New ballistic draws: 176. Snapshot classes: {None: 55, 'RECOVERY_CANDIDATE': 88, 'NOMINAL_SOLVABLE': 33}.
Search attempts this dataset: 1259. Legal successes: 49. Reasons: {'SUCCESS_12S': 49, 'DROP_AFTER_HANDOFF': 160, 'DROP_DURING_RECOVERY': 340, 'proposal_recovery_failed': 709, 'LOST_GRASP': 1}.
TRAIN recovery ICs: 87. VALID recovery ICs: 5. Control ICs: 7. Unknown: 122 ({'train': 109, 'valid': 13}).
Supervised rows: 172 ({'RECOVERY_CANDIDATE': 165, 'NOMINAL_SOLVABLE': 7}). HANDOFF labels: {'recorded_end': 149, 'early_handoff': 16, 'immediate_handoff': 7}.
Recovery mechanisms: {'airborne_chase': 98, 'release_regrasp': 47, 'repeated_recatch': 9, 'contact_preserving_slide': 11}.
GRPO initial conditions: 94. Scheduled updates (3 epochs): 282.
Targets of 200 TRAIN and 50 VALID recoverable ICs met: False.
Scenario keys shared by TRAIN and VALID recovery manifests: 0.

## Supervised-row reconciliation

The supervised manifest is the TRAIN file. It contains one row per verified TRAIN demonstration plus the immediate-HANDOFF control rows. VALID demonstrations stay in `valid_demonstrations.jsonl` and are not TRAIN examples.

{"supervised_rows": 172, "train_recovery_rows": 165, "train_recovery_ics": 87, "control_rows": 7, "valid_recovery_ics": 5, "train_ids_match": true, "valid_ids_in_supervised": 0, "rows_per_source": {"historical": 58, "pause": 48, "search": 43, "seed_search": 16}}

## Open-loop search

{"chase_lost": {"attempts": 179, "successes": 30, "seconds": 157.5}, "controlled_slide": {"attempts": 214, "successes": 12, "seconds": 177.8}, "controlled_slide_120": {"attempts": 214, "successes": 0, "seconds": 36.8}, "controlled_slide_240": {"attempts": 214, "successes": 0, "seconds": 31.0}, "release_long": {"attempts": 149, "successes": 2, "seconds": 45.4}, "release_short": {"attempts": 147, "successes": 5, "seconds": 69.3}, "slide_50": {"attempts": 142, "successes": 0, "seconds": 43.3}}

`chase_lost` is the only open-loop method with a material success rate. `slide_50` has not produced a verified recovery. Those attempts stay in `search.jsonl`.

## Seeded search

Seeds are legal demonstrations from the same split. A VALID demonstration is never replayed onto a TRAIN initial condition. Every trial restores the target snapshot, decodes through the K48 codebook, and must pass the 12-second check.

{"handoff_time": {"attempts": 79, "successes": 0, "new_ics": 0, "seconds": 2.715}, "perturb": {"attempts": 266, "successes": 9, "new_ics": 9, "seconds": 54.394}, "prefix_handoff": {"attempts": 270, "successes": 0, "new_ics": 0, "seconds": 13.746}, "seed_full": {"attempts": 187, "successes": 7, "new_ics": 7, "seconds": 53.213}}

TRAIN recovery speed bins: {"5-7": 27, "7-8": 29, "8-9": 17, "9-11": 14}.
VALID recovery speed bins: {"9-11": 3, "8-9": 2}.
TRAIN recovery delay bins: {"0-0.1": 23, "0.1-0.2": 24, "0.3-0.5": 23, "0.2-0.3": 17}.

Registered recovery candidates: 214. Wall clock for this process: 340.4 s. Search cost is the sum of per-attempt `seconds` in `search.jsonl`: 561.1 s.

The 200/50 targets are not a reason to relax the 12-second criterion. Resume continues from the logs without repeating completed replays.

## Resume

```
python -u training/rebuild_k48_recovery.py --seed-search --manifest --max-search-trials 800 --seed-trials-per-ic 6
```

This resumes from `seed_search.jsonl`. It does not repeat completed seed keys. Open-loop `slide_50` is not repeated.

## Later training commands

These commands are not part of dataset generation. Supervised training starts from the K=47 checkpoint so the HANDOFF row is expanded again, and it reads only the rebuilt manifest.

```
python -u training/unified_k48.py --supervised --config config/unified_k48_recovery_v3.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_recovery_v3.yaml --checkpoint demo_results/diagnostics/raw/k48_recovery_v3/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_recovery_v3_reward_v2.yaml --checkpoint demo_results/diagnostics/raw/k48_recovery_v3/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --eval --config config/unified_k48_recovery_v3.yaml --checkpoint demo_results/diagnostics/raw/k48_recovery_v3/supervised/checkpoint_epoch_0040.pt
```

The old 26 v1 VALID snapshots stay a separate no-intervention benchmark. They are not in this training pool.
