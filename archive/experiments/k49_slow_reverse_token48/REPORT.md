# K49 Slow Reverse Token 48 — Speed Comparison

**No pulse interleaving.** Speed is encoded only in named discrete tok48 mappings; execution uses normal `step_legal`.  
Historical full-speed K49 (`K49_tok48_wy1p0`) is preserved and not overwritten.

```text
python -u scripts/diagnose_k49_slow_reverse_token48.py --device cuda
```

Module: `training/action_vocab_k49.py` — `build_k49_table(config_id=...)`, `K49_TOK48_SPEED_CONFIGS`, `save_k49_speed_configs()`.

## Configurations (IDs 0–47 + HANDOFF=47 unchanged)

| config_id | wy_scale | tok48 action (a) | historical default |
|---|---:|---|---|
| `K49_tok48_wy1p0` | 1.0 | `[0.048,0,0,0,−0.800,0,1]` | **yes** |
| `K49_tok48_wy0p5` | 0.5 | `[0.048,0,0,0,−0.400,0,1]` | no |
| `K49_tok48_wy0p25` | 0.25 | `[0.048,0,0,0,−0.200,0,1]` | no |

Grip, translation, other angular channels, and token duration unchanged. Only `|a[4]|` (wy) scales.

Tables: `ACTION_MAPPING_CONFIGS/{config_id}_table.npy`.

## Return (identical pre-Return snapshot)

Matched hand target from 100% converge: **5.67°**. All three CONVERGED within the 4 s recovery budget.

| config | n×48 | time (s) | mean \|w_hand\| | Δe_ax (mm) | e_pos end (mm) | remain (s) |
|---|---:|---:|---:|---:|---:|---:|
| wy1p0 | 19 | 0.38 | 3.02 | **−4.84** | 15.10 | 2.48 |
| wy0p5 | 37 | 0.74 | 1.55 | **−2.35** | 13.00 | 2.12 |
| wy0p25 | 74 | 1.48 | 0.79 | **−1.17** | 11.95 | 1.38 |

## Full recovery — `hard_910263`

CPP → Sliding → Secure Return (tok48) → HANDOFF → unchanged nominal. Locked SUCCESS_NEW (10° / 12 mm hold).

| config | S12 | NEW | window θ / pos | handoff pos | hold both | n48 | rec (s) |
|---|---|---|---|---:|---:|---:|---:|
| wy1p0 | 1 | **0** | 6.70° / **15.96 mm** | 15.29 | 0.0 | 19 | 1.58 |
| wy0p5 | 1 | **0** | 6.61° / **13.63 mm** | 13.05 | 0.0 | 37 | 1.94 |
| wy0p25 | 1 | **0** | 6.61° / **12.46 mm** | 11.97 | 0.0 | 74 | 2.68 |

Slower Return monotonically improves axial preservation and final window position, but **no configuration reaches SUCCESS_NEW** (12 mm hold). 25% is closest (12.46 mm mean window; hold=0).

## Selection

**Preferred mapping for further work:** `K49_tok48_wy0p25`  

- 50% is **not** preferred: window 13.63 mm is not comparable to 25% (12.46 mm) relative to the 12 mm gate, despite shorter duration.  
- 25% is necessary to get nearest to the gate and still fits the deadline (~1.4 s remaining after return).  
- **Claim:** lower tok48 speed **reduces** return-induced axial slip but **does not** by itself make `hard_910263` a VERIFIED_FULL_EXPERT.

`SELECTED_CONFIG.json` / `ACTION_MAPPING_CONFIGS/SELECTED_table.npy` point at `K49_tok48_wy0p25`.  
Default `build_k49_table()` without `config_id` remains **historical wy1p0**.

## Stop

No SFT/GRPO. No physics/reward/SUCCESS_NEW changes. No broad expert search.
