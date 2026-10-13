# K=48 GRPO physical evaluation

All seven checkpoints were evaluated with argmax decoding on CPU, a 4-second recovery limit, and the unchanged 12-second physical SUCCESS_12S check. The 56 training initial conditions are the fixed v2 cases whose numeric id is divisible by 5.

| Checkpoint | Success | Failures | Early handoff (success) | Forced handoff (success) | Recovery min / median / mean / max (s) |
| --- | --- | --- | --- | --- | --- |
| supervised | 22/56 | 25 drop during, 5 drop after, 4 lost grasp | 20/26 | 2/5 | 0.080 / 1.210 / 1.562 / 4.000 |
| v1 update 312 | 27/56 | 6 drop during, 17 drop after, 6 lost grasp | 27/50 | 0/0 | 0.040 / 0.080 / 0.247 / 2.320 |
| v1 update 624 | 30/56 | 2 drop during, 18 drop after, 6 lost grasp | 30/54 | 0/0 | 0.000 / 0.000 / 0.005 / 0.120 |
| v1 update 936 | 30/56 | 20 drop after, 6 lost grasp | 30/56 | 0/0 | 0.000 / 0.000 / 0.000 / 0.000 |
| v2 update 312 | 30/56 | 5 drop during, 16 drop after, 5 lost grasp | 30/51 | 0/0 | 0.040 / 0.080 / 0.217 / 3.120 |
| v2 update 624 | 30/56 | 2 drop during, 18 drop after, 6 lost grasp | 30/54 | 0/0 | 0.000 / 0.000 / 0.005 / 0.120 |
| v2 update 936 | 30/56 | 20 drop after, 6 lost grasp | 30/56 | 0/0 | 0.000 / 0.000 / 0.000 / 0.000 |

Successful recoveries last 0.140–4.000 s for the supervised policy (median 1.270 s) and 0.040–1.720 s for V2 at update 312 (median 0.080 s). The four later tied checkpoints succeed only by handing off immediately, so their successful recovery duration is 0 s.

Five checkpoints tie at 30/56. Validation does not separate them: each is 26/26, all by early handoff. The supervised policy is 16/26 on the same 26 validation snapshots. No final-split file was opened.

Raw per-IC results are under `demo_results/diagnostics/raw/unified_k48/eval_compare/`. The aggregate is `comparison.json`.
