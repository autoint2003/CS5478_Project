# Hard IC mining report

## Scope

- SFT checkpoint: `results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt` (equal-weight Epoch 40)
- Physics / nominal / K48 / step_legal / SUCCESS_12S unchanged
- No GRPO training; no SFT modification
- Candidate generation: legal continuous ball-impact via `run_episode` + `take_capture` (not teleports)

## Frozen split rule

See `split_rule.json`. Assigned before outcome-based selection.

## Funnel

| Stage | Count |
|---|---:|
| Generation attempts | 360 |
| Valid impact recovery ICs kept as candidates | 133 |
| Hard by SFT (det fail or stoch < 2/3) | 125 |
| Hard + oracle/stochastic SUCCESS evidence | 32 |
| Hard + recoverability unknown under budget | 93 |
| Selected train / dev / test | 28 / 6 / 3 |

## SFT stochastic rate distribution (candidates)

- mean=0.065, median=0.000, frac_zero=0.880, frac_ge_1=0.008

### SFT det mechanism counts (all candidates)

`{'Other/Unclear': 100, 'Contact-Preserving Adjustment': 24, 'Hybrid': 7, 'Release-Recatch': 2}`

### Oracle success mechanism tags

`{'Release-Recatch': 20, 'Contact-Preserving Adjustment': 19, 'Hybrid': 1}`

## Baseline comparison (SUCCESS rates)

| pool | n | nominal SUCCESS | SFT det SUCCESS | mean stoch rate | CP succ | Recatch succ | Hybrid succ |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_candidates | 133 | 0 | 16 | 0.065 | 13 | 1 | 2 |
| selected_train | 28 | 0 | 9 | 0.202 | 8 | 0 | 1 |
| selected_dev | 6 | 0 | 3 | 0.167 | 2 | 0 | 1 |
| selected_test | 3 | 0 | 1 | 0.000 | 0 | 1 | 0 |

All candidates are nominal-fail (`RECOVERY_CANDIDATE`) by construction.

## Near-duplicate check

- Rejected fingerprints already in `data/snapshots/{train,valid}`: tracked during generation
- Known fingerprints loaded at start: 172
- Historical 17 valid ICs excluded from GRPO mix: ['inward_extra_250043', 'inward_extra_250051', 'inward_extra_250131', 'k48r_valid_00004', 'k48r_valid_00009', 'k48r_valid_00036', 'k48r_valid_00168', 's7_p25_a+0.000_e0.000', 's7_p25_a+0.000_e0.008', 's7_p25_a+0.012_e0.000', 's7_p25_a+0.012_e0.008', 's7_p25_a-0.012_e0.000', 's7_p25_a-0.012_e0.008', 'slide_quota_400058', 'slide_quota_400463', 'slide_quota_400496', 'slide_quota_400656']

## Files

- `candidate_manifest.jsonl`, `screening_results.csv`, `recoverability_search.csv`
- `selected_{train,dev,test}_manifest.jsonl`, `grpo_initial_conditions_hard.jsonl`
- `baseline_comparison.csv`, `grpo_readiness.md`, `snapshots/`, `videos/`

## Phase 0 interface audit (facts)

- Snapshots use `take_capture` / `restore_capture` (`envs/simulation/capture.py`): qpos/qvel/ctrl/time/FSM/switch/supports + policy obs history. Sufficient for exact recovery restart.
- Screening uses `immediate_handoff` (nominal), `roll_policy` (SFT det), `free_rollout` stochastic seeds 11/22/33, mechanism via `classify_mechanism`.
- Oracle: `expand_skill_quotas._try_sliding` + `vertical_recatch_search` (3 specs) + stochastic SFT success as recoverability evidence.
- SUCCESS_12S unchanged: recovery ≤4 s then unmodified nominal 12 s (`horizon_outcome.finish_recovery`).
- No intermediate SFT states used as ICs; only post-impact capture blobs from legal `run_episode`.

## Limitations (facts)

- Round-1 mining is **vertical-grasp ballistic only**. No dedicated Horizontal Recatch hard grid in this batch.
- Oracle budget is narrow; **93/125** hard ICs remain recoverability-unknown (not proven impossible).
- Selected **test set is small (n=3)** — usable as a frozen holdout seed, not a high-powered final benchmark. Expand before claiming GRPO generalization.
- Representative videos under `videos/` show SFT failure modes on hard_verified ICs (expected).

## 中文结论

1. **SFT 尚未稳定解决的 IC：** 125 / 133 候选被标为 hard（det 失败或 3 种子成功率 &lt; 2/3）。
2. **其中具有合法 SUCCESS_12S 证据：** 32 个（sliding / vertical oracle 或随机 SFT 成功）；已选入 train/dev/test 的 hard_verified 共 24+5+3。
3. **Sliding 与 Recatch 困难覆盖：** 有。Oracle 成功标签约 CP 19 / Recatch 20；selected train 类别滑动 12、垂直 Recatch 16。Horizontal 困难网格本轮未做。
4. **是否具备启动 GRPO 条件：** **可做有限 pilot**（train 28，含 24 hard_verified + 4 retention）。不建议在未扩大 test、未接入 Horizontal 前宣称完整统一 GRPO。
5. **仍可能妨碍从强 SFT 继续提升的问题：**
   - 大量 hard_unknown（窄 oracle 预算）若误入训练会导致零成功、无 reward 方差；
   - 等权 SFT 偏 Sliding，GRPO 可能再度压掉 Recatch 或反之；
   - Tree80 默认读 `data/*manifest*`，需单独配置读取 `grpo_initial_conditions_hard.jsonl`；
   - 无 Horizontal 难例 → 统一多技能不完整；
   - test 仅 3 条，checkpoint 选择必须只用 dev。

## Repro

```powershell
python -u visualization/mine_hard_ics.py --device cuda --generate-attempts 160 --seed-offset 910000 --skip-videos
# then expand batch (documented in expand_log.txt) to total 360 attempts
```

