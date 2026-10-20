# Archive cleanup report

## Rollback

| Item | Value |
|---|---|
| Pre-cleanup HEAD | `d3223833647a8dc7107dc7898bec0b7abef55a47` |
| Branch | `cleanup/ground-scene-prep` |
| Hash ledger | `archive/manifests/ROLLBACK_POINT.json` |
| Manifest | `archive/manifests/ARCHIVE_MANIFEST.csv` |

Verified: branch exists and points at the recorded commit before archival moves.

## Actions taken

| Action | Count | Notes |
|---|---|---|
| Moved (HISTORICAL) | 89 | `_tmp_*` / `_continue_*` scripts + superseded diagnostic dirs |
| Retained ACTIVE_CORE | 11 packages | `controllers/`, `envs/`, … |
| Retained VALIDATED_RESEARCH | 127 rows | pilots, generators, expert dirs, key diagnostics |
| Deleted | **0** | No irreversible deletion |
| REDUNDANT proposed | 50 | Videos/`_frames_*` listed only; not deleted |

## Disk

Sizes are whole-workspace recursive file totals (includes gitignored `results/`, `data/`, videos).

| | Bytes |
|---|---|
| Before | 5,165,714,569 |
| After | 5,165,714,569 |
| Delta | 0 (moves only; no deletes) |

## Archive layout

```text
archive/
  scripts/tmp/                 # scripts/_tmp_* and _continue_*
  experiments/
    sliding/_tmp_*
    recatch/k49/_tmp_*
    ball_impact/_tmp_*
    legacy/organize_k49_new_files.py
  results/diagnostics/         # superseded intermediate diagnostic trees
  manifests/
    ARCHIVE_MANIFEST.csv
    PRESERVED_EVIDENCE.md
    CLEANUP_REPORT.md
    ROLLBACK_POINT.json
    CLEANUP_SUMMARY.json
```

## Safety constraints honored

- No `git clean -fdx`
- No checkpoint rewrite
- No expert success relabeling
- Uncommitted user validated pilots / generators left in active tree
- Historical one-offs moved, not duplicated
- Large videos not copied; regenerable candidates inventoried only
