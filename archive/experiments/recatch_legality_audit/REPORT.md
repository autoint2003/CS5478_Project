# Recatch legality audit

Replay uses the stored recovery actions, `legal_verify`, and the unchanged `SUCCESS_12S` check. No dataset, reward, success rule, or simulator file was edited.

## Coverage

Physically replayed: every unique `continuous_npz` referenced by `data/supervised_targets.jsonl` (251), `data/valid_demonstrations.jsonl` (14), `data/recatch/manifest.jsonl` (158), `data/controlled_sliding/manifest.jsonl` (50), `data/inward_sliding/manifest.jsonl` (50), and `data/control/manifest.jsonl` (7). After path dedupe this is 265 trajectories. There is no TEST split in these manifests. `data/grpo_initial_conditions.jsonl` has 87 initial conditions and no action tapes.

Not resimulated: historical `chase_lost` / `release_long` / `release_short` npz files outside those manifests, including `demo_results/diagnostics/raw/k48_recovery_v3/search_attempts` (932), `legal_recovery_v2/trajectories` (472), and `k48_handoff/trajectories` (142). Counts are in `raw/coverage.json`. Those files are not a second copy of the audited verdicts.

A loss of control is both finger geoms out of contact for at least 0.04 s. Table support is object-table or object-floor normal force at or above half the cylinder weight (about 1.96 N) for at least 0.02 s, or an equal impulse. A bounce is an upward speed change of at least 0.10 m/s during that contact together with an impulse of at least 0.005 N·s. A shorter or weaker contact is a brush and is not enough for the illegal class. Three tapes had that table interval and then `SUCCESS_12S`, but the first return to both fingers flickered inside 0.04 s, so the strict regrasp window missed them. They are classed illegal from the table interval plus the official recovered grasp, not from the filename.

## Counts

Deduped trajectories: 265.

| class | count |
| --- | ---: |
| LEGAL_RECATCH | 0 |
| ILLEGAL_TABLE_REGRASP | 158 |
| NOT_RECATCH | 107 |
| UNCERTAIN | 0 |

| split | LEGAL_RECATCH | ILLEGAL_TABLE_REGRASP | NOT_RECATCH | UNCERTAIN |
| --- | ---: | ---: | ---: | ---: |
| train | 0 | 154 | 97 | 0 |
| valid | 0 | 4 | 10 | 0 |

SFT rows with nonzero weight and an illegal replay: 154. Their weight sum is 3.0000 of 5.0000. Validation demonstrations are not in the SFT weight sum. No TEST rows were found.

## Known cases

### ball_train_v2_0051

- `data/recatch/train/ball_train_v2_0051__chase_lost__verified.npz`, snapshot `data/snapshots/train/ball_train_v2_0051.pkl`, split train, strategy chase_lost, SFT True, weight 0.016129032258064516, GRPO initial condition True.
- Replay class `ILLEGAL_TABLE_REGRASP`, official reason `SUCCESS_12S`, SUCCESS_12S True. Consistency qpos 0.0, action error 0.0, contact match True.
- Table interval t=3.290s for 0.492s, supported 0.288s, impulse 0.95377 N·s, upward dv 1.4918 m/s, min z 0.416 m. Peak force 38.002 N on ['object', 'table'] at z=0.4323, nL/nR 0/0, velocity [-0.02029, 0.17235, -1.28692].

### inward_extra_250005

- `data/inward_sliding/train/inward_extra_250005__inward_reseat.npz`, snapshot `data/inward_sliding/snapshots/train/inward_extra_250005.pkl`, split train, strategy inward_reseat, SFT True, weight 0.022727272727272728, GRPO initial condition False.
- Replay class `NOT_RECATCH`, official reason `SUCCESS_12S`, SUCCESS_12S True. Consistency qpos 0.0, action error 0.0, contact match True.
- No object-table or object-floor contact interval was recorded while a finger was off.

### slide_extra_260013

- `data/controlled_sliding/train/slide_extra_260013__controlled_slide__verified.npz`, snapshot `data/snapshots/train/slide_extra_260013.pkl`, split train, strategy controlled_slide, SFT True, weight 0.021739130434782608, GRPO initial condition False.
- Replay class `NOT_RECATCH`, official reason `SUCCESS_12S`, SUCCESS_12S True. Consistency qpos 0.0, action error 0.0, contact match True.
- No object-table or object-floor contact interval was recorded while a finger was off.

## Illegal trajectories

Every row below has a replay-consistent tape, a bilateral finger gap of at least 0.04 s, table support or bounce inside that gap, and `SUCCESS_12S` from the unchanged checker. Full peaks are in `raw/trajectories.jsonl` and `raw/events.jsonl`.

| IC | strategy | split | weight | t0 | supported s | impulse N·s | dv m/s | min z | peak N | nL/nR | SUCCESS_12S |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `ball_train_0017` | chase_lost | train | 0.016129032258064516 | 3.334 | 0.3 | 0.99445 | 1.5953 | 0.4221 | 48.3585 | 0/4 | True |
| `ball_train_0027` | chase_lost | train | 0.016129032258064516 | 3.334 | 0.314 | 0.98687 | 1.5577 | 0.4218 | 46.3059 | 0/4 | True |
| `ball_train_v2_0002` | chase_lost | train | 0.016129032258064516 | 3.030 | 0.296 | 0.90481 | 1.3448 | 0.4235 | 44.8763 | 0/4 | True |
| `ball_train_v2_0012` | chase_lost | train | 0.016129032258064516 | 3.118 | 0.286 | 1.76068 | 1.2725 | 0.4288 | 34.9342 | 0/0 | True |
| `ball_train_v2_0031` | chase_lost | train | 0.016129032258064516 | 3.112 | 0.294 | 0.84716 | 1.4554 | 0.424 | 30.1739 | 0/0 | True |
| `ball_train_v2_0035` | chase_lost | train | 0.1111111111111111 | 3.086 | 0.264 | 0.82237 | 1.425 | 0.4244 | 29.2083 | 0/0 | True |
| `ball_train_v2_0036` | chase_lost | train | 0.016129032258064516 | 3.116 | 0.278 | 0.88612 | 1.4083 | 0.4243 | 41.719 | 5/0 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.272 | 0.87834 | 1.3067 | 0.4245 | 51.471 | 0/8 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.312 | 0.95559 | 1.3067 | 0.4245 | 50.884 | 0/9 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.352 | 1.03289 | 1.3067 | 0.4245 | 50.2964 | 0/9 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.392 | 1.10768 | 1.3067 | 0.4245 | 48.4573 | 0/8 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.472 | 1.70064 | 1.3067 | 0.4245 | 43.6686 | 10/0 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.312 | 0.95729 | 1.3067 | 0.4245 | 51.7362 | 0/8 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.352 | 1.03462 | 1.3067 | 0.4245 | 51.1638 | 0/9 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.392 | 1.10982 | 1.3067 | 0.4245 | 49.5241 | 0/8 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.472 | 1.2674 | 1.3067 | 0.4245 | 49.8369 | 0/10 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.272 | 0.87718 | 1.3067 | 0.4245 | 50.8917 | 0/8 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.272 | 0.87718 | 1.3067 | 0.4245 | 50.8917 | 0/8 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.272 | 0.87718 | 1.3067 | 0.4245 | 50.8917 | 0/8 | True |
| `ball_train_v2_0038` | chase_lost | train | 0.0012406947890818859 | 2.992 | 0.272 | 0.87718 | 1.3067 | 0.4245 | 50.8917 | 0/8 | True |
| `ball_train_v2_0044` | chase_lost | train | 0.016129032258064516 | 3.316 | 0.308 | 2.62922 | 1.5575 | 0.4241 | 75.6719 | 0/0 | True |
| `ball_train_v2_0051` | chase_lost | train | 0.016129032258064516 | 3.290 | 0.288 | 0.95377 | 1.4918 | 0.416 | 38.002 | 0/0 | True |
| `ball_train_v2_0053` | chase_lost | train | 0.016129032258064516 | 2.932 | 0.262 | 0.88079 | 1.3187 | 0.4234 | 52.709 | 0/9 | True |
| `ball_train_v2_0060` | chase_lost | train | 0.016129032258064516 | 3.240 | 0.316 | 0.96612 | 1.4697 | 0.4228 | 43.7353 | 0/4 | True |
| `ball_train_v2_0077` | chase_lost | train | 0.016129032258064516 | 3.278 | 0.25 | 1.44904 | 1.3745 | 0.4265 | 97.6725 | 0/3 | True |
| `ball_train_v2_0082` | chase_lost | train | 0.016129032258064516 | 2.950 | 0.33 | 0.9316 | 1.3568 | 0.4242 | 31.8975 | 0/0 | True |
| `ball_train_v2_0083` | chase_lost | train | 0.016129032258064516 | 2.956 | 0.342 | 1.00335 | 1.161 | 0.4139 | 32.5522 | 0/3 | True |
| `ball_train_v2_0085` | chase_lost | train | 0.1111111111111111 | 2.970 | 0.288 | 0.7841 | 1.1848 | 0.4292 | 32.7399 | 0/0 | True |
| `ball_train_v2_0086` | chase_lost | train | 0.016129032258064516 | 2.934 | 0.352 | 1.00229 | 1.3405 | 0.4247 | 38.0877 | 0/9 | True |
| `ball_train_v2_0087` | chase_lost | train | 0.016129032258064516 | 3.050 | 0.304 | 0.85424 | 1.4011 | 0.4243 | 29.6602 | 0/0 | True |
| `ball_train_v2_0093` | chase_lost | train | 0.016129032258064516 | 3.174 | 0.288 | 0.90727 | 1.4435 | 0.4226 | 45.8478 | 0/4 | True |
| `ball_train_v2_0098` | chase_lost | train | 0.016129032258064516 | 3.214 | 0.302 | 0.95962 | 1.468 | 0.4222 | 52.0583 | 0/12 | True |
| `ball_train_v2_0103` | chase_lost | train | 0.016129032258064516 | 3.192 | 0.288 | 0.95734 | 1.4946 | 0.4233 | 51.4535 | 0/9 | True |
| `ball_train_v2_0113` | chase_lost | train | 0.016129032258064516 | 3.216 | 0.286 | 0.89058 | 1.507 | 0.4231 | 32.6079 | 0/0 | True |
| `ball_train_v2_0116` | chase_lost | train | 0.016129032258064516 | 3.132 | 0.268 | 0.88513 | 1.498 | 0.4228 | 35.8056 | 0/8 | True |
| `ball_train_v2_0135` | chase_lost | train | 0.016129032258064516 | 3.194 | 0.262 | 1.75996 | 1.4867 | 0.4247 | 36.4255 | 0/0 | True |
| `ball_train_v2_0137` | chase_lost | train | 0.016129032258064516 | 2.952 | 0.28 | 0.87197 | 1.2603 | 0.4243 | 49.6595 | 0/7 | True |
| `ball_train_v2_0138` | chase_lost | train | 0.016129032258064516 | 3.094 | 0.294 | 0.9115 | 1.3698 | 0.4234 | 48.6404 | 0/6 | True |
| `ball_train_v2_0149` | chase_lost | train | 0.016129032258064516 | 3.074 | 0.294 | 0.8858 | 1.4487 | 0.423 | 32.7759 | 0/0 | True |
| `ball_train_v2_0155` | chase_lost | train | 0.016129032258064516 | 3.150 | 0.268 | 0.88109 | 1.4352 | 0.4228 | 52.0666 | 0/14 | True |
| `ball_train_v2_0164` | chase_lost | train | 0.016129032258064516 | 3.152 | 0.268 | 0.87497 | 1.4341 | 0.4227 | 49.8976 | 0/8 | True |
| `ball_train_v2_0168` | chase_lost | train | 0.016129032258064516 | 3.232 | 0.264 | 0.90084 | 1.5254 | 0.4219 | 42.6131 | 0/4 | True |
| `ball_train_v2_0171` | chase_lost | train | 0.016129032258064516 | 3.008 | 0.366 | 3.54258 | 1.3223 | 0.4264 | 95.7046 | 0/0 | True |
| `ball_train_v2_0183` | chase_lost | train | 0.016129032258064516 | 3.010 | 0.298 | 0.90575 | 1.3058 | 0.4245 | 46.2538 | 0/5 | True |
| `ball_train_v2_0189` | chase_lost | train | 0.016129032258064516 | 3.176 | 0.272 | 2.69881 | 1.5321 | 0.4235 | 58.5109 | 0/0 | True |
| `ball_train_v2_0190` | chase_lost | train | 0.016129032258064516 | 3.012 | 0.298 | 0.90433 | 1.3024 | 0.4243 | 44.5289 | 0/4 | True |
| `ball_train_v2_0194` | chase_lost | train | 0.016129032258064516 | 3.112 | 0.28 | 0.90471 | 1.4294 | 0.4239 | 44.8059 | 0/4 | True |
| `ball_train_v2_0196` | chase_lost | train | 0.016129032258064516 | 3.170 | 0.426 | 1.8048 | 1.4143 | 0.4266 | 35.6663 | 0/0 | True |
| `ball_train_v2_0200` | chase_lost | train | 0.016129032258064516 | 3.134 | 0.264 | 0.90231 | 1.466 | 0.4221 | 44.5482 | 0/4 | True |
| `ball_train_v2_0208` | chase_lost | train | 0.016129032258064516 | 3.092 | 0.29 | 0.90931 | 1.3783 | 0.4235 | 47.9551 | 0/6 | True |
| `ball_train_v2_0216` | chase_lost | train | 0.016129032258064516 | 3.012 | 0.298 | 0.90295 | 1.3073 | 0.4243 | 44.0032 | 0/4 | True |
| `ball_train_v2_0225` | chase_lost | train | 0.016129032258064516 | 3.130 | 0.262 | 0.88373 | 1.4339 | 0.4228 | 54.2689 | 0/16 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.3 | 0.91674 | 1.451 | 0.424 | 33.2959 | 0/14 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.34 | 1.00292 | 1.451 | 0.424 | 34.0697 | 0/4 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.38 | 1.07088 | 1.451 | 0.424 | 31.9995 | 0/0 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.42 | 1.17275 | 1.451 | 0.424 | 40.164 | 0/4 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.5 | 1.3563 | 1.451 | 0.424 | 52.753 | 0/8 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.34 | 1.00629 | 1.451 | 0.424 | 35.7526 | 0/5 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.38 | 1.07209 | 1.451 | 0.424 | 31.9995 | 0/0 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.42 | 1.17028 | 1.451 | 0.424 | 38.9282 | 0/4 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.5 | 1.33939 | 1.451 | 0.424 | 44.2978 | 0/4 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.3 | 0.91633 | 1.451 | 0.424 | 33.0919 | 0/14 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.3 | 0.91633 | 1.451 | 0.424 | 33.0919 | 0/14 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.3 | 0.91633 | 1.451 | 0.424 | 33.0919 | 0/14 | True |
| `ball_train_v2_0236` | chase_lost | train | 0.0012406947890818859 | 3.152 | 0.3 | 0.91633 | 1.451 | 0.424 | 33.0919 | 0/14 | True |
| `ball_train_v2_0239` | chase_lost | train | 0.016129032258064516 | 3.014 | 0.26 | 0.88231 | 1.3831 | 0.4236 | 53.7163 | 0/8 | True |
| `ball_train_v2_0244` | chase_lost | train | 0.016129032258064516 | 3.100 | 0.306 | 1.09599 | 1.3431 | 0.4287 | 49.5978 | 0/7 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.3 | 1.0016 | 1.5876 | 0.4228 | 53.6787 | 0/14 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.34 | 1.0653 | 1.5876 | 0.4228 | 46.3911 | 0/5 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.38 | 1.14527 | 1.5876 | 0.4228 | 47.1476 | 0/8 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.42 | 1.27507 | 1.5876 | 0.4228 | 46.7583 | 0/8 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.5 | 2.10335 | 1.5876 | 0.4228 | 49.4855 | 0/8 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.34 | 1.06496 | 1.5876 | 0.4228 | 46.2215 | 0/4 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.38 | 1.14334 | 1.5876 | 0.4228 | 46.1822 | 0/4 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.42 | 1.224 | 1.5876 | 0.4228 | 47.2768 | 0/6 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.5 | 1.3811 | 1.5876 | 0.4228 | 47.3446 | 0/8 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.34 | 1.06724 | 1.5876 | 0.4228 | 47.3627 | 0/5 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.38 | 1.14207 | 1.5876 | 0.4228 | 45.5472 | 0/4 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.42 | 1.21745 | 1.5876 | 0.4228 | 43.9977 | 0/4 | True |
| `ball_train_v2_0251` | chase_lost | train | 0.0012406947890818859 | 3.290 | 0.5 | 1.3699 | 1.5876 | 0.4228 | 41.7463 | 0/4 | True |
| `ball_train_v2_0259` | chase_lost | train | 0.016129032258064516 | 3.294 | 0.308 | 0.98918 | 1.5473 | 0.4237 | 47.0843 | 0/10 | True |
| `ball_train_v2_0261` | chase_lost | train | 0.1111111111111111 | 3.190 | 0.346 | 1.02983 | 1.5471 | 0.4233 | 33.0862 | 0/14 | True |
| `ball_train_v2_0268` | chase_lost | train | 0.016129032258064516 | 3.278 | 0.29 | 0.93483 | 1.1339 | 0.4266 | 26.9103 | 0/0 | True |
| `ball_train_v2_0272` | chase_lost | train | 0.016129032258064516 | 2.970 | 0.282 | 0.86291 | 1.2548 | 0.4239 | 44.8504 | 0/4 | True |
| `ball_train_v2_0279` | chase_lost | train | 0.016129032258064516 | 3.094 | 0.336 | 2.75044 | 1.3577 | 0.4274 | 80.8029 | 0/0 | True |
| `ball_train_v2_0280` | chase_lost | train | 0.016129032258064516 | 3.112 | 0.292 | 0.89909 | 1.3896 | 0.4236 | 42.4052 | 0/4 | True |
| `k48r_train_00005` | chase_lost | train | 0.016129032258064516 | 3.092 | 0.306 | 0.92271 | 1.4146 | 0.4245 | 34.0531 | 0/9 | True |
| `k48r_train_00006` | chase_lost | train | 0.016129032258064516 | 3.114 | 0.264 | 0.8904 | 1.5105 | 0.4227 | 41.5537 | 0/8 | True |
| `k48r_train_00007` | chase_lost | train | 0.016129032258064516 | 3.292 | 0.286 | 0.94866 | 1.5557 | 0.4221 | 46.5781 | 0/5 | True |
| `k48r_train_00033` | chase_lost | train | 0.016129032258064516 | 3.334 | 0.3 | 0.99556 | 1.5827 | 0.423 | 50.7473 | 0/7 | True |
| `k48r_train_00081` | chase_lost | train | 0.016129032258064516 | 3.112 | 0.268 | 0.87079 | 1.3713 | 0.424 | 47.2592 | 0/5 | True |
| `k48r_train_00092` | chase_lost | train | 0.016129032258064516 | 3.010 | 0.278 | 0.86365 | 1.3076 | 0.4244 | 43.7158 | 0/4 | True |
| `k48r_train_00094` | chase_lost | train | 0.016129032258064516 | 3.314 | 0.316 | 0.98657 | 1.5341 | 0.4223 | 46.7446 | 0/4 | True |
| `k48r_train_00125` | chase_lost | train | 0.016129032258064516 | 2.970 | 0.27 | 0.87921 | 1.317 | 0.4245 | 51.807 | 0/9 | True |
| `k48r_train_00128` | chase_lost | train | 0.016129032258064516 | 2.974 | 0.258 | 0.87875 | 1.3628 | 0.4231 | 52.0225 | 0/9 | True |
| `k48r_train_00132` | chase_lost | train | 0.016129032258064516 | 3.312 | 0.31 | 0.98218 | 1.5445 | 0.4227 | 43.6207 | 0/4 | True |
| `k48r_train_00148` | chase_lost | train | 0.016129032258064516 | 3.172 | 0.332 | 2.58583 | 1.402 | 0.4276 | 49.2646 | 0/0 | True |
| `k48r_train_00150` | chase_lost | train | 0.016129032258064516 | 3.048 | 0.284 | 0.91217 | 1.3914 | 0.4241 | 48.2062 | 0/5 | True |
| `k48r_train_00152` | chase_lost | train | 0.016129032258064516 | 3.100 | 0.28 | 1.63489 | 1.2948 | 0.4273 | 53.8206 | 18/0 | True |
| `k48r_train_00162` | chase_lost | train | 0.016129032258064516 | 2.994 | 0.318 | 0.92042 | 1.3475 | 0.4249 | 33.7263 | 0/8 | True |
| `k48r_train_00176` | chase_lost | train | 0.016129032258064516 | 3.094 | 0.316 | 0.94743 | 1.4554 | 0.4239 | 31.7335 | 0/0 | True |
| `ball_train_v2_0002` | release_long | train | 0.02857142857142857 | 3.030 | 0.286 | 0.91354 | 1.356 | 0.4234 | 49.2131 | 0/6 | True |
| `ball_train_v2_0031` | release_long | train | 0.1111111111111111 | 3.112 | 0.316 | 0.93994 | 1.4576 | 0.424 | 30.4028 | 0/0 | True |
| `ball_train_v2_0035` | release_long | train | 0.02857142857142857 | 3.086 | 0.28 | 0.92178 | 1.4085 | 0.424 | 51.1939 | 0/8 | True |
| `ball_train_v2_0038` | release_long | train | 0.02857142857142857 | 2.992 | 0.294 | 0.92061 | 1.306 | 0.4245 | 53.0238 | 0/9 | True |
| `ball_train_v2_0041` | release_long | train | 0.02857142857142857 | 3.292 | 0.312 | 0.89413 | 1.5445 | 0.422 | 33.8784 | 0/0 | True |
| `ball_train_v2_0053` | release_long | train | 0.02857142857142857 | 2.932 | 0.262 | 0.88258 | 1.3163 | 0.4234 | 53.6158 | 0/10 | True |
| `ball_train_v2_0059` | release_long | train | 0.02857142857142857 | 3.074 | 0.276 | 2.52071 | 1.4213 | 0.4259 | 51.0905 | 0/0 | True |
| `ball_train_v2_0068` | release_long | train | 0.1111111111111111 | 2.934 | 0.72 | 3.61695 | 1.3704 | 0.4241 | 44.9505 | 0/4 | True |
| `ball_train_v2_0083` | release_long | train | 0.02857142857142857 | 2.956 | 0.36 | 1.02165 | 1.1598 | 0.4136 | 32.307 | 0/0 | True |
| `ball_train_v2_0085` | release_long | train | 0.02857142857142857 | 2.970 | 0.328 | 0.90284 | 1.184 | 0.4293 | 32.706 | 0/0 | True |
| `ball_train_v2_0086` | release_long | train | 0.02857142857142857 | 2.934 | 0.35 | 1.00486 | 1.3438 | 0.4246 | 38.3598 | 0/8 | True |
| `ball_train_v2_0087` | release_long | train | 0.02857142857142857 | 3.050 | 0.324 | 0.97728 | 1.4034 | 0.4242 | 41.8149 | 0/9 | True |
| `ball_train_v2_0093` | release_long | train | 0.02857142857142857 | 3.174 | 0.276 | 0.90966 | 1.4649 | 0.4224 | 46.8602 | 0/5 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.282 | 0.9203 | 1.4679 | 0.4223 | 52.0412 | 0/15 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.322 | 0.99698 | 1.4679 | 0.4223 | 51.1592 | 0/14 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.362 | 1.0789 | 1.4679 | 0.4223 | 52.8803 | 0/18 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.402 | 1.15966 | 1.4679 | 0.4223 | 54.0204 | 0/17 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.482 | 1.31367 | 1.4679 | 0.4223 | 52.5471 | 0/15 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.322 | 0.99922 | 1.4679 | 0.4223 | 52.2789 | 0/10 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.362 | 1.07374 | 1.4679 | 0.4223 | 50.3019 | 0/9 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.402 | 1.15572 | 1.4679 | 0.4223 | 52.0493 | 0/9 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.482 | 1.30824 | 1.4679 | 0.4223 | 49.8306 | 0/8 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.282 | 0.91636 | 1.4679 | 0.4223 | 50.0699 | 0/13 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.282 | 0.91636 | 1.4679 | 0.4223 | 50.0699 | 0/13 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.282 | 0.91636 | 1.4679 | 0.4223 | 50.0699 | 0/13 | True |
| `ball_train_v2_0098` | release_long | train | 0.002197802197802198 | 3.214 | 0.282 | 0.91636 | 1.4679 | 0.4223 | 50.0699 | 0/13 | True |
| `ball_train_v2_0120` | release_long | train | 0.02857142857142857 | 2.932 | 0.264 | 0.86716 | 1.3401 | 0.4232 | 46.1067 | 0/4 | True |
| `ball_train_v2_0127` | release_long | train | 0.02857142857142857 | 3.134 | 0.272 | 0.91646 | 1.4273 | 0.4238 | 50.751 | 0/7 | True |
| `ball_train_v2_0131` | release_long | train | 0.02857142857142857 | 3.006 | 0.266 | 0.85529 | 1.3304 | 0.4234 | 50.0558 | 0/11 | True |
| `ball_train_v2_0155` | release_long | train | 0.02857142857142857 | 3.150 | 0.268 | 0.87812 | 1.433 | 0.4228 | 50.6041 | 0/8 | True |
| `ball_train_v2_0161` | release_long | train | 0.02857142857142857 | 3.272 | 0.3 | 0.98653 | 1.5679 | 0.4227 | 45.8452 | 0/4 | True |
| `ball_train_v2_0164` | release_long | train | 0.02857142857142857 | 3.152 | 0.274 | 0.90506 | 1.4577 | 0.4225 | 45.1551 | 0/4 | True |
| `ball_train_v2_0183` | release_long | train | 0.02857142857142857 | 3.010 | 0.298 | 0.90725 | 1.3112 | 0.4243 | 46.877 | 0/5 | True |
| `ball_train_v2_0186` | release_long | train | 0.1111111111111111 | 3.090 | 0.324 | 0.94395 | 1.4334 | 0.4246 | 31.6081 | 0/0 | True |
| `ball_train_v2_0190` | release_long | train | 0.02857142857142857 | 3.012 | 0.298 | 0.90955 | 1.3118 | 0.4242 | 47.1834 | 0/5 | True |
| `ball_train_v2_0194` | release_long | train | 0.02857142857142857 | 3.112 | 0.278 | 0.91144 | 1.4366 | 0.4237 | 47.7508 | 0/5 | True |
| `ball_train_v2_0200` | release_long | train | 0.02857142857142857 | 3.134 | 0.29 | 0.90789 | 1.3997 | 0.4242 | 47.1494 | 0/5 | True |
| `ball_train_v2_0216` | release_long | train | 0.02857142857142857 | 3.012 | 0.298 | 0.90878 | 1.305 | 0.4243 | 46.9631 | 0/5 | True |
| `ball_train_v2_0225` | release_long | train | 0.02857142857142857 | 3.130 | 0.262 | 0.88074 | 1.4344 | 0.4227 | 52.7447 | 0/10 | True |
| `ball_train_v2_0226` | release_long | train | 0.02857142857142857 | 3.230 | 0.55 | 1.51655 | 1.3169 | 0.4141 | 37.2494 | 0/0 | True |
| `ball_train_v2_0236` | release_long | train | 0.1111111111111111 | 3.152 | 0.3 | 0.90042 | 1.4481 | 0.4241 | 31.8234 | 0/0 | True |
| `ball_train_v2_0251` | release_long | train | 0.02857142857142857 | 3.290 | 0.3 | 1.00166 | 1.5841 | 0.4229 | 53.728 | 0/14 | True |
| `ball_train_v2_0255` | release_long | train | 0.02857142857142857 | 3.030 | 0.28 | 0.92062 | 1.3721 | 0.424 | 52.6591 | 0/15 | True |
| `ball_train_v2_0259` | release_long | train | 0.02857142857142857 | 3.294 | 0.308 | 0.94787 | 1.5495 | 0.4237 | 33.212 | 0/0 | True |
| `ball_train_v2_0272` | release_long | train | 0.1111111111111111 | 2.970 | 0.332 | 0.93329 | 1.3355 | 0.4244 | 29.6388 | 0/0 | True |
| `ball_train_v2_0274` | release_long | train | 0.02857142857142857 | 3.274 | 0.3 | 0.94112 | 1.5062 | 0.4226 | 42.7817 | 0/4 | True |
| `k48r_train_00032` | release_long | train | 0.02857142857142857 | 3.352 | 0.234 | 1.092 | 1.595 | 0.422 | 59.2889 | 0/12 | True |
| `ball_train_v2_0121` | release_short | train | 0.02857142857142857 | 3.088 | 0.468 | 1.42929 | 1.2782 | 0.4129 | 44.7609 | 0/7 | True |
| `ball_train_v2_0199` | release_short | train | 0.02857142857142857 | 2.930 | 0.432 | 1.15302 | 1.2809 | 0.4263 | 30.4863 | 0/0 | True |
| `ball_train_v2_0211` | release_short | train | 0.02857142857142857 | 3.112 | 0.444 | 1.2442 | 1.5136 | 0.423 | 40.9563 | 0/9 | True |
| `k48r_train_00046` | release_short | train | 0.1111111111111111 | 3.074 | 0.694 | 3.48209 | 1.3414 | 0.4288 | 36.5329 | 0/0 | True |
| `k48r_train_00115` | release_short | train | 0.02857142857142857 | 3.200 | 0.532 | 1.38536 | 1.5634 | 0.4249 | 33.2021 | 0/0 | True |
| `k48r_train_00135` | release_short | train | 0.02857142857142857 | 3.332 | 0.386 | 1.1564 | 1.6024 | 0.4137 | 34.2887 | 0/0 | True |
| `k48r_valid_00004` | chase_lost | valid | None | 3.292 | 0.302 | 0.94596 | 1.5026 | 0.4236 | 45.1664 | 0/4 | True |
| `k48r_valid_00009` | chase_lost | valid | None | 3.094 | 0.25 | 0.87933 | 1.4419 | 0.4227 | 51.8232 | 0/9 | True |
| `k48r_valid_00036` | chase_lost | valid | None | 3.290 | 0.3 | 0.94761 | 1.5195 | 0.4225 | 46.1396 | 0/4 | True |
| `k48r_valid_00078` | chase_lost | valid | None | 2.978 | 0.316 | 1.01087 | 1.2847 | 0.4242 | 43.5976 | 13/0 | True |

## Evaluator

`recovery_failed` counts a drop only after 50 control steps (1.0 s) with no finger contact and object height below 0.44 m, or below 0.30 m immediately. `nominal_verify` then accepts `SUCCESS_12S` from bilateral contact, lift, and a 12 s hold. It does not read object-table force. All 158 illegal replays still returned `SUCCESS_12S`, so the current checker accepts a table-supported regrasp when the pickup happens inside that window.

## Contamination

The supervised file contains 154 illegal demonstrations with nonzero weight. They are the `chase_lost`, `release_long`, and `release_short` rows. `controlled_slide` and `inward_reseat` replays in this set did not show a finger-off table interval. GRPO does not train on these action tapes; 83 of the 87 GRPO initial conditions also have an illegal supervised tape. No rows were deleted.

## What was not changed

A minimal later change would be to reject a rollout whose unrecovered interval contains the table-support or bounce test above, and to stop giving those tapes nonzero supervised weight. This audit does not apply that change.

## Evidence levels

- Physics: the class counts, the table force, impulse, height, speed, and finger counts above come from MuJoCo replays whose stored actions and, where present, stored `nL`/`nR`/`obj_z` matched. qpos restore error was 0.
- Video, separate from this replay: the earlier side-by-side videos of `ball_train_v2_0051`, `inward_extra_250005`, and `slide_extra_260013` showed the same outcomes. The classes in this report do not use those videos.
- Not established: legality of historical search npz files that are not referenced by the current manifests. No `LEGAL_RECATCH` tape was found in the replayed set, and none of the replayed tapes were `UNCERTAIN`.
