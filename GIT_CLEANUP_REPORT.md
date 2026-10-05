# Git cleanup report

Date: 2026-10-05. **No history rewrite. No push.**

## Before / after

| | Before | After |
|---|---|---|
| `.git` | **2054.1 MB** | **1.8 MB** |
| `git count-objects` loose `size` | **2.00 GiB** (10613 loose objects) | **0 bytes** |
| packed `size-pack` | 1.50 MiB | 1.50 MiB |
| `in-pack` | 265 | 265 |
| `packs` | 1 | 1 |

`HEAD` blobs were never ~2 GB (~2.9 MB of source, markdown, PDFs, small npz).

## Why `.git` was ~2 GB

Loose **unreachable** blobs from an earlier `git add results/diagnostics` of
huge JSON/NPZ/MP4 dumps that were unstaged before commit `696c74c`. They never
entered history. `git gc --prune=now` removed them.

Largest dangling objects were 45–119 MB JSON dumps and diagnostic MP4s.

## History rewrite

**Not performed.** `filter-repo` was unnecessary.

## `.gitignore`

Ignores videos, render frames, `cam_stills/`, checkpoints, buffers, TensorBoard,
and diagnostic raw/npz/pkl/csv, with exceptions for `snap_early.pkl`,
`stages.json`, `release_map_recatch.json`, and `airborne_offset_eval.npz`.

## Remote

`origin` is GitHub. `main` was ahead of `origin/main`. **Do not force-push.**
History SHAs were not rewritten.

Backup bundle (outside repo):
`D:\NUS\CS5478 Intelligent Robots\CS5478_Project_git_backup_696c74c.bundle`
