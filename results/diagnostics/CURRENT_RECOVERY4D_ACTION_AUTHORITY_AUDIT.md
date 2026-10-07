# Current recovery4d action-authority audit

This audit reads the current `RecoveryEnv` / `recovery4d` path and checks it with six 0.10 s pulses from one fixed snapshot. Nothing was trained. No recovery search was run.

Snapshot: `make_one_training_ic` on `config/nominal.yaml`, RNG seed 3. Each pulse resets to that snapshot and then calls `RecoveryEnv.step`. Physics timestep is 0.002 s. The 0.10 s window is 50 steps with `n_sub = 1`, so every physics step still goes through `map_recovery4d` and `recovery4d_tick`. A separate check repeats `a = [0, +1, 0, +1]` at the native policy rate (`n_sub = 10`, five steps, same 0.10 s).

Numbers: `results/diagnostics/raw/current_recovery4d_action_authority_audit/pulses.json`.

## 1. Action definition

`a` is clipped to `[-1, 1]^4` inside `map_recovery4d`.

| channel | meaning |
| --- | --- |
| `a0` | wrist rate about the current `r_des` body y |
| `a1` | linear velocity along the current hand-frame x |
| `a2` | linear velocity along world z |
| `a3` | gripper tendon command, secure at `+1`, open at `-1` |

## 2. Normalized mapping

From `controllers/residual.py`:

```115:139:controllers/residual.py
def map_recovery4d(action: np.ndarray, r_des: np.ndarray, r_hand: np.ndarray) -> dict:
    a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
    ...
    w_hy = float(a[0]) * RECOVERY4D_W_HY_MAX
    v_hx = float(a[1]) * RECOVERY4D_V_HX_MAX
    v_z = float(a[2]) * RECOVERY4D_V_Z_MAX
    tau = RECOVERY4D_TAU_OPEN + 0.5 * (float(a[3]) + 1.0) * (
        RECOVERY4D_TAU_SECURE - RECOVERY4D_TAU_OPEN
    )
    v_world = r_h @ np.array([v_hx, 0.0, 0.0])
    v_world = v_world + np.array([0.0, 0.0, v_z])
    w_world = r_d @ np.array([0.0, w_hy, 0.0])
```

Constants in that file, and the matching keys asserted by `RecoveryEnv._assert_recovery4d_cfg`:

| symbol | value |
| --- | --- |
| `RECOVERY4D_W_HY_MAX` | 4.0 rad/s |
| `RECOVERY4D_V_HX_MAX` | 0.08 m/s |
| `RECOVERY4D_V_Z_MAX` | 0.08 m/s |
| `RECOVERY4D_TAU_SECURE` | −18 |
| `RECOVERY4D_TAU_OPEN` | +2 |

So `a0 = +1` is `ω_y = +4` rad/s, `a1 = +1` is `v_hx = +0.08` m/s, `a2 = +1` is `v_z = +0.08` m/s, and `a3 = +1` is tendon command −18. `a3 = −1` is tendon command +2. `a3 = 0` is −8.

`config/nominal.yaml` `recovery.w_hy_max`, `v_hx_max`, and `v_z_max` are 4.0, 0.08, and 0.08. `recovery.tau_open` in that file is −1.0. `map_recovery4d` does not read it. The value written for a full open is +2.

## 3. Per channel

| | `a0` wrist | `a1` hand x | `a2` world z | `a3` grip |
| --- | --- | --- | --- | --- |
| units | rad/s | m/s | m/s | tendon command, written to `ctrl[7]` |
| scale | 4.0 | 0.08 | 0.08 | linear from +2 to −18 |
| clip | `a` to `[-1, 1]`; no extra clip on `ω_y` | same | same | `tau` clipped to actuator range `[-50, 50]` before the write |
| rate limit | none on this path | none | none | none |
| absolute or residual | absolute `w_world` | absolute `v_world` | absolute, added in world z | absolute tendon command |
| nominal motion added | no | no | no | no |
| value that reaches the controller | `w_cmd = R_des @ [0, ω_y, 0]`, also `w_des` | `v_cmd = R_hand @ [v_hx, 0, 0]`, also `v_des` | same vector, plus `[0, 0, v_z]` | `ctrl[7] = tau`. `fg_cmd` is set to `-tau` and is not the actuator write |

`recovery.dv_max`, `dv_rate`, `dw_max`, and `dfg_rate` belong to `ResidualLimiter`. `RecoveryEnv.step` calls that limiter only when `action_kind` is not `recovery4d`.

Nominal lift is a different function. `GraspFSM.step` sets `v_cmd = [0, 0, lift_speed]` with `lift_speed` 0.08 m/s, and adds a residual only when `in_recovery` is true on that path. `recovery4d_tick` does not call `GraspFSM.step`.

Arm joints are not given the Cartesian velocity directly. `apply_cartesian_ctrl` writes seven joint torques from a Cartesian PD whose reference is `p_des`, `r_des`, `v_des`, and `w_des`, clipped to each actuator range. Those torques are `ctrl[0:7]`.

## 4. Code path

`RecoveryEnv.step` for `action_kind == "recovery4d"`:

```262:265:envs/recovery_env.py
        if self.action_kind == "recovery4d":
            r_h = np.array(self.sim.data.xmat[self.sim.ids.hand_body].reshape(3, 3), float)
            mapped = map_recovery4d(a, self.sim.fsm.r_des, r_h)
            self.sim.recovery4d_tick(mapped["v_world"], mapped["w_world"], mapped["tau"])
```

`BallisticSim.recovery4d_tick` integrates that same world twist for `n_sub` physics steps. It does not add a lift velocity.

```195:224:envs/grasp_sim.py
    def recovery4d_tick(self, v_world, w_world, tau):
        """Absolute Cartesian command. No nominal lift, no clip_fg."""
        ...
        for _ in range(self.n_sub):
            self.fsm.p_des = self.fsm.p_des + v * dt
            self.fsm.r_des = _integrate_rot(self.fsm.r_des, w, dt)
            self.fsm.v_cmd = v.copy()
            self.fsm.w_cmd = w.copy()
            self.fsm.v_des = v.copy()
            self.fsm.w_des = w.copy()
            self.fsm.fg_cmd = float(-tau)
            apply_cartesian_ctrl(..., gripper_tau=tau, v_des=v, w_des=w, **gains)
            mujoco.mj_step(...)
```

`apply_cartesian_ctrl` then sets `data.ctrl[:7]` to the clipped joint torques and `data.ctrl[7]` to the tendon command.

On this snapshot the training IC already meets the online proxy, so some pulse steps report `term = recovered`. The step still applies the command before that flag. Every logged sample has `v_cmd` and `w_cmd` equal to the `map_recovery4d` output from the hand frame at the start of that step.

## 5. Pulse results

Commanded `v_cmd` is at the full channel scale on the first 2 ms sample. It does not ramp. No actuator sat against its range (`sat_ctrl` is false on all six pulses). `ctrl[7]` is −18 for every listed pulse, because each uses `a3 = +1`. `fg_cmd` is +18 on those same samples.

`v_cmd` for `a1 = +1` is `[0, +0.08, 0]` m/s in world. On this snapshot, hand-frame x is world +Y. The speed is still 0.08 m/s.

| pulse | commanded `v_hx`, `v_z`, `ω_y` | `|v_cmd|` | `|w_cmd|` | peak origin linear speed | peak COM linear speed | peak angular speed | displacement in 0.10 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `[+1, 0, 0, +1]` | 0, 0, +4 rad/s | 0 | 4.00 rad/s | 0.130 m/s | 0.225 m/s | 3.16 rad/s | 10.1 mm |
| `[-1, 0, 0, +1]` | 0, 0, −4 rad/s | 0 | 4.00 rad/s | 0.128 m/s | 0.222 m/s | 3.17 rad/s | 9.5 mm |
| `[0, +1, 0, +1]` | +0.08 m/s, 0, 0 | 0.080 m/s | 0 | 0.059 m/s | 0.059 m/s | 0.041 rad/s | 4.35 mm |
| `[0, −1, 0, +1]` | −0.08 m/s, 0, 0 | 0.080 m/s | 0 | 0.059 m/s | 0.059 m/s | 0.044 rad/s | 4.35 mm |
| `[0, 0, +1, +1]` | 0, +0.08 m/s, 0 | 0.080 m/s | 0 | 0.033 m/s | 0.034 m/s | 0.051 rad/s | 1.80 mm |
| `[0, 0, −1, +1]` | 0, −0.08 m/s, 0 | 0.080 m/s | 0 | 0.078 m/s | 0.077 m/s | 0.068 rad/s | 5.71 mm |

Origin linear velocity is `mj_jacBody` at the hand body origin. COM linear velocity is `mj_objectVelocity` at the body COM. They match on the pure translation pulses. They differ on the wrist pulses because the COM is not on the rotation axis.

The native-rate repeat of `[0, +1, 0, +1]` writes the same `|v_cmd| = 0.080` m/s and the same peak origin speed 0.059 m/s.

## 6. Actual maxima

Commanded, from the map and from `v_cmd` / `w_cmd` on these pulses:

- maximum commanded hand-x velocity: **0.08 m/s** in the current hand frame
- maximum commanded world-z velocity: **0.08 m/s**
- maximum commanded wrist `ω_y`: **4.0 rad/s**

Peak actual hand speed over 0.10 s:

- pure hand-x command: **0.059 m/s** at the origin and at the COM
- pure world-z command: **0.034 m/s** upward, **0.078 m/s** downward
- pure wrist command: angular speed **3.17 rad/s**; linear speed **0.130 m/s** at the origin and **0.225 m/s** at the COM

A perfect 0.08 m/s track for 0.10 s would move 8.0 mm. The hand-x pulses move 4.4 mm. The upward pulse moves 1.8 mm. The command is applied immediately; the Cartesian PD does not reach it within 0.10 s, except the downward pulse, which gets close.

## 7. The 0.08 m/s claim

The translation **commands** `a1` and `a2` are scaled to 0.08 m/s, and the pulses write `|v_cmd| = 0.080` with no lift term added.

That number is not a ceiling on hand translation. With `v_cmd = 0`, the wrist pulse moves the hand origin at 0.130 m/s and the COM at 0.225 m/s. The same 0.08 m/s figure is also `fsm.lift_speed` on the nominal lift path, but the recovery4d pulses are not that path: their `v_cmd` matches `map_recovery4d` alone.

FALSE

## 8. Legacy recatch commands

`results/comparison/ballistic_recovery/raw/airborne_recatch.npz` is the stored airborne-recatch log. It contains time, relative position, contact counts, object height, relative speed, aperture, and `ctrl[7]`. It does not contain hand linear or angular velocity, so those speeds are not reconstructed from the file. `ctrl[7]` in the file runs from −18 to +2.

The current generator of that log is `recatch_one` in `training/release_state_map_omega4.py`. Its commands are:

- rotate and open: `ω_y = 4.0` rad/s, `v_x = 0`, `v_z = 0`, tendon −18 while rotating and `TAU_OPEN` while opening
- close and hold: `ω_y = 0`, `v_x = 0`, `v_z = 0`, tendon −18

`TAU_OPEN` there is `RECOVERY4D_TAU_OPEN`, which is +2. `TAU_SEC` is −18. The file’s `ctrl[7]` range matches that pair.

Those commands are produced by `tick_omega_override`, which still clips `v_x` and `v_z` with `RECOVERY4D_V_HX_MAX` and `RECOVERY4D_V_Z_MAX`, then replaces `w_world` with `R_des @ [0, ω_y, 0]` without passing `ω_y` through `a0`. The value it is called with is `OMEGA4 = 4.0`. The current `a0 = +1` command is also 4.0 rad/s, so the clip would not change this trajectory. It integrates with `tick_vw` once per physics step. `recovery4d_tick` holds one world twist for `n_sub` steps and also stores `v_des` / `w_des`. Both pass the same `v_des` and `w_des` into `apply_cartesian_ctrl`. The commanded magnitudes used by this recatch are inside the current `recovery4d` map: `a = [+1, 0, 0, +1]` while secure, and `a = [+1, 0, 0, −1]` while open.

CURRENT RECOVERY4D MATCHES LEGACY RECATCh AUTHORITY
