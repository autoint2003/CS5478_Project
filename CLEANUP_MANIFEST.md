# Cleanup manifest (paper-oriented)

Criterion: keep only files needed for the final paper/presentation, the
reproducible ballistic comparison, or upcoming recovery-policy training.

Git history retains deleted experiments. The working tree is not an archive.

## Deleted (working-tree pass)

| Category | Count (approx.) |
|---|---|
| Obsolete training scripts (STOP/SSR/teleport-offset demos, old impact tests, wrist/inward-slide/recenter audits) | 19 |
| Intermediate diagnostic raw dumps, figures, videos, superseded result trees | ~11,300 |
| Empty/stale `results/buffers`, `checkpoints`, `tb`, `logs`, `figures` | included above |
| **Total files removed** | **11,345** |
| **Disk freed (working tree)** | **~2.95 GB** |

Not deleted: current executable pipeline, `RecoveryEnv` / 4D training entry
points, noslip calibration reports, CENTER-6 ballistic reports, slip and
airborne authority reports, comparison raw logs, `snap_early.pkl`,
`release_map_recatch.json`.

## Git object store

A later pass pruned **unreachable** loose objects that had been hashed by a
mass `git add` and then unstaged (never committed). `.git` went from ~2 GB
to ~2 MB. See `GIT_CLEANUP_REPORT.md`.
