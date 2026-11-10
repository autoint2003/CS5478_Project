# VIDEO_VALIDATION — P64 presentation clips

- Checkpoint: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\experiments\p64_ppo_multiround\checkpoints\P64_step4_actor.pt` SHA16=`ec08f59816aa2fdd`
- Official criterion: SUCCESS_RELAXED_12S (left) + do-nothing drop (right)
- Continuous dual-sim: seated/lift → ball flight → impact on BOTH panels, then fork
- Eval protocol seeds: split=`heldout`, temperature=1.0 (matches P64_step4)

## Count correction
Provisional 6/16·4/16·3/16·7/16 do NOT match any frozen P64 panel. Use P64_step4 MSP rates or P83 8/32·6/32·4/32·17/32 in slides.

## Per-clip
### A1
- skill: `A1`
- traj_id: `A__slide_quota_400660__ef7cfa9e9ca57a40`
- snapshot_hash: `b730f4b48532b8cd`
- checkpoint: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\experiments\p64_ppo_multiround\checkpoints\P64_step4_actor.pt`
- sha16: `ec08f59816aa2fdd`
- seed: `3705230660`
- seed_requested: `3705230660`
- original_SUCCESS_RELAXED_12S: `1`
- reproduced_SUCCESS_RELAXED_12S: `1`
- right_dropped: `1`
- continuous: `True`
- n_steps: `30`
- recatch: `1`
- behavior_label: `corrective`
- selection_rule: `median_n_steps_among_nontrivial_successes; tie-break full traj_id then seed; behavior_label=corrective`
- video_frames: `989`
- video_duration_s: `41.208333333333336`
- sim_time_s: `13.89400000000131`
- flight: `{"mode": "same_cam_dual_sim_lift+aim", "episode": "slide_quota_400660", "impact_t": 8.133999999999386, "launched": true, "n_shared_frames": 306, "n_total_frames": 989, "flight_aim_s": 0.35, "slow_flight_x": 8, "slow_recovery_x": 3, "spec": {"azimuth_deg": 136.2065796724005, "speed_mps": 6.475037315781252, "z_off_m": 0.002162755481385245, "phi_deg": 27.127407714385473, "delay_s": 0.021351590473114197}, "seed_used": 3705230660, "seed_requested": 3705230660}`
- mp4: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\videos\P64_A1_success.mp4`
- poster: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\videos\P64_A1_poster.png`
- command: `python experiments/precision/p64_presentation_videos/render_p64_family_videos.py`

### B1
- skill: `B1`
- traj_id: `plan_b__slide_extra_260006`
- snapshot_hash: `671abf032db53978`
- checkpoint: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\experiments\p64_ppo_multiround\checkpoints\P64_step4_actor.pt`
- sha16: `ec08f59816aa2fdd`
- seed: `3933932719`
- seed_requested: `3235788579`
- original_SUCCESS_RELAXED_12S: `1`
- reproduced_SUCCESS_RELAXED_12S: `1`
- right_dropped: `1`
- continuous: `True`
- n_steps: `49`
- recatch: `1`
- behavior_label: `corrective`
- selection_rule: `median_n_steps_among_nontrivial_successes; tie-break full traj_id then seed; behavior_label=corrective`
- video_frames: `950`
- video_duration_s: `39.583333333333336`
- sim_time_s: `14.022000000001352`
- flight: `{"mode": "same_cam_dual_sim_lift+aim", "episode": "slide_extra_260006", "impact_t": 7.881999999999354, "launched": true, "n_shared_frames": 210, "n_total_frames": 950, "flight_aim_s": 0.1, "slow_flight_x": 8, "slow_recovery_x": 3, "spec": {"azimuth_deg": 28.904704952105476, "speed_mps": 7.7711606199999546, "z_off_m": 0.003245393794270782, "phi_deg": 17.98147877539035, "delay_s": 0.17471518801814287}, "seed_used": 3933932719, "seed_requested": 3235788579}`
- mp4: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\videos\P64_B1_success.mp4`
- poster: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\videos\P64_B1_poster.png`
- command: `python experiments/precision/p64_presentation_videos/render_p64_family_videos.py`

### C
- skill: `C`
- traj_id: `plan_c__gbi_041_7f94ea38dd`
- snapshot_hash: `62b0ffd0cbfd03e6`
- checkpoint: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\experiments\p64_ppo_multiround\checkpoints\P64_step4_actor.pt`
- sha16: `ec08f59816aa2fdd`
- seed: `1844945303`
- seed_requested: `1844945303`
- original_SUCCESS_RELAXED_12S: `1`
- reproduced_SUCCESS_RELAXED_12S: `1`
- right_dropped: `1`
- continuous: `True`
- n_steps: `29`
- recatch: `1`
- behavior_label: `detach_recatch`
- selection_rule: `median_n_steps_among_nontrivial_successes; tie-break full traj_id then seed; behavior_label=detach_recatch`
- video_frames: `704`
- video_duration_s: `29.333333333333332`
- sim_time_s: `12.920000000000984`
- flight: `{"mode": "same_sim_rebuild+pulled_cfg_launch", "impact_t": 0.019999999999997797, "n_shared_frames": 24, "n_total_frames": 704, "cfg": {"config_id": "gbi_041", "axial_offset": 0.028, "z": 0.65, "tilt_deg": 0.0, "approach": "oblique", "speed": 6.5, "offset": -0.008, "z_off": 0.005212544727511991}, "launch": {"p_launch": [0.6091478218908031, -0.0074823701447181345, 0.5594901029189168], "v0": [-4.2250000000000005, 0.0, -4.875], "speed": 6.4510658034157435, "approach": "oblique", "offset": -0.008, "z_off": 0.005212544727511991, "pullback_m": 0.0, "flight_est_s": 0.0}, "pullback_m": 0.0, "slow_flight_x": 8, "slow_recovery_x": 3, "seed_used": 1844945303, "seed_requested": 1844945303, "pre": {"bilateral": true, "obj_z": 0.5042775581914047, "hand_z": 0.6372905714020711, "topo": "bilateral"}}`
- mp4: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\videos\P64_C_success.mp4`
- poster: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\videos\P64_C_poster.png`
- command: `python experiments/precision/p64_presentation_videos/render_p64_family_videos.py`

### D
- skill: `D`
- traj_id: `p33_ball_train_0001_COMBINED_dx4_dz12`
- snapshot_hash: `e02ffd0651dc97e3`
- checkpoint: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\experiments\p64_ppo_multiround\checkpoints\P64_step4_actor.pt`
- sha16: `ec08f59816aa2fdd`
- seed: `3657265304`
- seed_requested: `2168866308`
- original_SUCCESS_RELAXED_12S: `1`
- reproduced_SUCCESS_RELAXED_12S: `1`
- right_dropped: `1`
- continuous: `True`
- n_steps: `33`
- recatch: `1`
- behavior_label: `corrective`
- selection_rule: `median_n_steps_among_nontrivial_successes; tie-break full traj_id then seed; behavior_label=corrective`
- video_frames: `982`
- video_duration_s: `40.916666666666664`
- sim_time_s: `13.902000000001312`
- flight: `{"mode": "same_cam_dual_sim_lift+aim", "episode": "ball_train_0001", "impact_t": 8.081999999999368, "launched": true, "n_shared_frames": 290, "n_total_frames": 982, "flight_aim_s": 0.3, "slow_flight_x": 8, "slow_recovery_x": 3, "spec": {"azimuth_deg": 0.0, "speed_mps": 6.5, "z_off_m": -0.015, "phi_deg": 0.0, "delay_s": 0.0}, "seed_used": 3657265304, "seed_requested": 2168866308}`
- mp4: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\videos\P64_D_success.mp4`
- poster: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\videos\P64_D_poster.png`
- command: `python experiments/precision/p64_presentation_videos/render_p64_family_videos.py`

## Dependencies
- mujoco Renderer, imageio / imageio-ffmpeg (libx264 yuv420p), numpy, torch, PIL
- No audio. Resolution 1280×720 @ 24fps (two 640×480 panels upscaled).