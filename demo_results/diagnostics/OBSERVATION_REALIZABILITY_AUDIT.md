# Observation realizability audit

The balanced policy reads `observe_airborne` (`envs/airborne_obs.py`, version `obs_gt_airborne_v2`). `training/balanced_unified_recovery.py` calls that function at each policy step. The vector has 45 scalars: `OBS_NAMES` (27) from `envs/observable_obs.py`, then `AIR_NAMES` (18). Every entry is clipped to \([-1, 1]\) after the scale below.

The network input is this 45-vector concatenated with the previous 7-D command (`Policy.forward` in `training/natural_loss_recatch_transformer_v3.py`). That previous command is the controller output from the last step, starting at zero. It is not one of the 45. It is available on a robot as commanded-action state.

No observation entry is object mass, friction, a disturbance parameter, a future state, an oracle label, a success label, or a stored reset pose.

Current simulator reads are noise-free stand-ins for the sensors named below. Object pose and twist are MuJoCo body state used in place of an RGB-D tracker. Finger contact position and normal force are MuJoCo contact state used in place of a pad sensor. Those stand-ins are exact. They are not a sensor-noise model.

## 1. The 45 dimensions

Hand frame \(R_h, p_h\) is `data.xmat` / `data.xpos` of `ids.hand_body`. Object frame \(R_o, p_o\) is the same fields of `ids.object_body`. In the scene XMLs the object inertial origin is `pos="0 0 0"`, so \(p_o\) is the center of mass. `body_twist` calls `mj_objectVelocity` with `flg_local=0` and splits the buffer into angular velocity `vel[0:3]` and linear velocity `vel[3:6]`, both in the world frame.

`read_tactile` builds the tactile dict from `measure_spatial_tactile` and `geometric_from_reading`. Contact points are `data.contact[i].pos`. Normal force is `abs(mj_contactForce[0])`. A contact is kept when one body is `ids.object_body` and the other is a finger body. Pad coordinates \((u, v)\) are the contact point in the finger frame, minus the pad-center offsets in `sensors/spatial_tactile.py`. CoP is the force-weighted average. It is valid only if the summed normal force is at least \(0.01\,\mathrm{N}\). `e_hat_x` is the mean of \(u_L\) and \(-u_R\) over the pads that are valid. `contact.dist` is not read.

| index | name | definition | frame | units before scale | scale, then clip to \([-1,1]\) | simulator field |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `e_hat_x` | mean of valid \(u_L\) and \(-u_R\); 0 if neither pad is valid | finger-pad \(u\), mapped toward hand \(x\) | m | `/ 0.0075` | contact position via `geometric_from_reading` |
| 1 | `estimate_valid` | 1 if at least one pad CoP is valid | — | 0/1 | none | same validity flag |
| 2–3 | `contact_present_L/R` | 1 if that finger has at least one kept contact | — | 0/1 | none | `data.ncon` filtered by finger and object body |
| 4–5 | `valid_L/R` | 1 if that pad's summed normal force is \(\ge 0.01\,\mathrm{N}\) | — | 0/1 | none | `mj_contactForce` |
| 6 | `u_L` | left CoP, finger-local \(x\) minus pad center; 0 if invalid | left finger | m | `/ 0.0075` | contact position |
| 7 | `v_L` | left CoP, finger-local \(z\) minus pad center; 0 if invalid | left finger | m | `/ 0.0085` | contact position |
| 8–9 | `u_R`, `v_R` | same on the right finger | right finger | m | \(u/0.0075\), \(v/0.0085\) | contact position |
| 10–11 | `fn_L/R` | summed absolute normal force on that finger | contact normal | N | `/ 20` | `mj_contactForce` |
| 12 | `aperture` | mean of the two finger joint positions | joint | m | `/ 0.040` | `data.qpos[ids.finger_jnt]` |
| 13 | `ap_dot` | mean of the two finger joint velocities | joint | m/s | `/ 0.080` | `data.qvel[ids.finger_dof]` |
| 14 | `tau_from_secure` | \((\tau - (-18))/17\), \(\tau=\) `data.ctrl[7]` | actuator command | N·m before the shift | the formula itself | `data.ctrl[7]` |
| 15–17 | `g_hx/y/z` | \(R_h^\top g\), \(g=\) `model.opt.gravity` | hand | m/s² | `/ 9.81` | hand `xmat`, `model.opt.gravity` |
| 18–19 | `v_hx`, `v_hy` | hand-frame \(x,y\) of the hand linear velocity | hand | m/s | `/ 0.080` | `mj_objectVelocity` of the hand |
| 20 | `v_z_world` | world-\(z\) component of the hand linear velocity | world | m/s | `/ 0.080` | same hand twist, index 2 |
| 21–23 | `w_hx/y/z` | \(R_h^\top \omega_h\) | hand | rad/s | `/ 5` | hand angular velocity |
| 24 | `hand_z_off_table` | \(p_{h,z} - 0.40\) | world \(z\) | m | `/ 0.25` | hand `xpos[2]`, constant `TABLE_TOP` |
| 25 | `track_z` | \(p_{\mathrm{des},z} - p_{h,z}\) | world \(z\) | m | `/ 0.080` | `fsm.p_des[2]`, hand `xpos[2]` |
| 26 | `e_hat_dot` | \((e_t - e_{t-1})/0.02\) when both samples are valid; otherwise 0 | same as `e_hat_x` | m/s | `/ 0.080` | previous tactile estimate in `ObservableObsState` |
| 27–29 | `p_rel_h*` | \(R_h^\top (p_o - p_h)\) | hand | m | `/ 0.25` | object and hand `xpos`, hand `xmat` |
| 30–32 | `v_rel_h*` | \(R_h^\top (v_o - v_h)\) | hand | m/s | `/ 2` | both bodies' `mj_objectVelocity` linear part |
| 33–35 | `rot_rel_*` | rotation vector of \(R_h^\top R_o\) | axis in the hand frame | rad | `/ \pi` | object and hand `xmat` |
| 36–38 | `w_rel_h*` | \(R_h^\top (\omega_o - \omega_h)\) | hand | rad/s | `/ 8` | both bodies' angular velocity |
| 39–41 | `v_hand_h*` | \(R_h^\top v_h\) | hand | m/s | `/ 2` | hand linear velocity |
| 42–44 | `w_hand_h*` | \(R_h^\top \omega_h\) | hand | rad/s | `/ 8` | hand angular velocity |

The rotation vector uses \(\theta=\arccos(\mathrm{clip}((\mathrm{tr}R-1)/2))\) and the skew-symmetric axis. If \(\theta<10^{-8}\) or \(|2\sin\theta|<10^{-8}\), the code writes zeros. The second case includes a relative rotation of \(\pi\).

Indices 18–19 and 21–23 are the same hand twist as 39–44, with scales \(0.08\,\mathrm{m/s}\) and \(5\,\mathrm{rad/s}\). The airborne copies use \(2\,\mathrm{m/s}\) and \(8\,\mathrm{rad/s}\). A chase faster than \(0.08\,\mathrm{m/s}\) saturates the prefix and remains visible in the airborne copy.

## 2. Sensor class

| index | class | sensor or estimator |
| --- | --- | --- |
| 0, 6–9 | B | pad contact location. The code's source is MuJoCo `contact.pos`, used as a noise-free CoP. |
| 1, 4–5 | B | same pad: CoP is declared valid only above \(0.01\,\mathrm{N}\). |
| 2–3 | B | pad contact detection. |
| 10–11 | B | pad normal force from `mj_contactForce`, as a noise-free force reading. |
| 12–13 | A | finger encoders and their velocities. |
| 14 | A | gripper torque command, `ctrl[7]`. |
| 15–17 | D | hand orientation from encoders and forward kinematics, times a known gravity vector. The code reads `model.opt.gravity`. |
| 18–23 | A | joint encoders and velocities through forward kinematics, or a wrist IMU. The code reads `mj_objectVelocity` of the hand body. |
| 24 | A | hand height from forward kinematics, minus a calibrated table height. The code uses the constant \(0.40\,\mathrm{m}\). |
| 25 | A | controller setpoint `fsm.p_des` minus hand height. |
| 26 | D | backward difference of index 0 over the \(0.02\,\mathrm{s}\) policy step, gated by index 1. |
| 27–38 | B | RGB-D or depth object tracker, expressed in the hand frame from robot kinematics. The code reads object `xpos`, `xmat`, and `mj_objectVelocity` with no noise. |
| 39–44 | D | same hand twist as indices 18–23, divided by the airborne scales. |

The body-id test inside `measure_spatial_tactile` chooses which MuJoCo contacts belong to the object. The 45-vector does not contain those ids. A pad mounted on the finger reports whatever touches that pad, so the deployed sensor does not need the simulator id.

## 3. The 18 airborne channels

Indices 39–44 are hand twist. They do not depend on seeing or touching the object.

Indices 27–38 are object state relative to the current hand. The reference is the live hand pose, not a seated pinch and not a reset snapshot.

**Relative position (27–29).** \(R_h^\top(p_o-p_h)\), metres, scale \(0.25\).

- Between the fingers: a depth camera can track the cylinder while any of the barrel or the rims is visible, using the robot hand pose to form the relative vector. Pad CoP supplies a contact location on each finger. That location is already indices 0 and 6–9. It is not this 3-vector. If the fingers hide the whole cylinder, this channel has no measurement until the tracker sees it again.
- Airborne, no contact: the same RGB-D tracker, if the cylinder stays in view. Tactile sensing has nothing to say once both pads are open.

**Relative orientation (33–35).** Rotation vector of \(R_h^\top R_o\), radians, scale \(\pi\).

- Between the fingers: orientation comes from the visible contour or rims in depth, composed with the measured hand orientation. Contact on the two pads constrains where the surface touches. It does not fix the cylinder's spin about its axis.
- Airborne: the depth tracker continues to supply \(R_o\) while the cylinder is visible. Open pads do not observe orientation.

**Relative linear velocity (30–32).** \(R_h^\top(v_o-v_h)\), m/s, scale \(2\).

- Between the fingers: difference the tracked object position and subtract the hand velocity from proprioception. If the object is hidden, temporal pose tracking stops. Contact-point motion is index 26, which is not this spatial relative velocity.
- Airborne: difference successive RGB-D poses and subtract the measured hand velocity. No tactile term.

**Relative angular velocity (36–38).** \(R_h^\top(\omega_o-\omega_h)\), rad/s, scale \(8\).

- Between the fingers: difference successive orientation estimates from the depth tracker and subtract the measured hand rate. Pad shear or CoP migration can indicate slip. It does not determine the 3-vector \(\omega_o\).
- Airborne: the same differenced orientations. Tactile contact is absent, so it cannot provide \(\omega_o\).

In all four groups the implementation currently copies MuJoCo's object pose and world twist. That copy is a noise-free surrogate for the tracker above. It is not a second, hidden object state.

## 4. Privileged leakage

| quantity | present in the 45? | what the code does |
| --- | --- | --- |
| MuJoCo geom or body ids | no | ids select finger–object contacts inside the tactile emulator; they are not outputs |
| contact penetration `contact.dist` | no | not read |
| object body origin and orientation | yes, indices 27–29 and 33–35 | `data.xpos` / `data.xmat` of the object body |
| object linear and angular velocity | yes, indices 30–32 and 36–38 | `mj_objectVelocity` of the object body |
| object mass | no | not read |
| friction coefficient | no | not read |
| disturbance parameters | no | not read |
| future states | no | the vector uses the current step and, for index 26, the previous tactile sample |
| oracle or success labels | no | not read |
| reset snapshot | no | `ObservableObsState` stores only the last valid `e_hat_x` |

Object pose and twist are currently taken from MuJoCo ground truth. They are the quantities an RGB-D tracker estimates when the cylinder is visible. They are not quantities that exist only inside the contact solver.

Mass, friction, penetration depth, labels, and the reset pose are absent, so they are not a deployment leak in this vector.

## 5. References

| quantity | reference | on a robot |
| --- | --- | --- |
| 27–38 | current hand pose \(R_h, p_h\) | encoders and forward kinematics |
| 15–17 | world gravity, then the hand frame | known gravity and the same hand orientation |
| 24 | table height \(0.40\,\mathrm{m}\) | a calibrated workcell constant, not a per-step object measurement |
| 25 | controller setpoint `fsm.p_des` | internal state of the tracking controller |
| 0, 6–9, 26 | finger-pad center and the previous valid `e_hat_x` | the pad geometry is known; the previous sample is the estimator's own memory |
| 14 | secure torque \(-18\,\mathrm{N\cdot m}\) inside the normalization | a fixed scale, not a measured object property |
| 39–44 | current hand twist | proprioception |

No relative channel uses the seated pinch, a nominal object pose, or the state from reset.

## 6. Conclusion

Every scalar is either a robot or controller quantity, a rescaling of one of those quantities, a tactile CoP or force, or the object pose and twist relative to the hand. The last group is MuJoCo ground truth in this repository. Under the stated stand-in, a depth tracker supplies it when the cylinder is visible, and the hand frame comes from proprioception. Pad sensing supplies the contact channels and does not observe the object once contact is gone.

45-D OBSERVATION IS REALIZABLE GIVEN RGB-D PLUS TACTILE/PROPRIOCEPTION
