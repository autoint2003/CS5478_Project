# SUPERSEDED BY NOSLIP CALIBRATION

The 60° + τ=-18 inward relative motion documented here was **low-ρ regularization
creep** (`noslip_iterations=0`, ρ≈0.10). It is **not** validated gravitational
reposition authority. See `results/diagnostics/_SUPERSEDED_BY_NOSLIP.md`.

The contact-force bookkeeping in this report remains useful as a methods record.

# Vertical slip contact-mechanics audit

Diagnostic only. Same IC / 100 ms delay / +1.2 rad/s / 60 deg / tau=-18 / 1.0 s slide / return 0.8 / 2 s vertical. No strategy change.

## Evidence classes

- **VERIFIED CODE FACT**: MuJoCo API + this extraction path.
- **RAW TRAJECTORY EVIDENCE**: numbers from the replayed representative run.
- **USER VISUAL OBSERVATION**: 60-deg supported inward/down motion; after return, visible distal slide.
- **INTERPRETATION / HYPOTHESIS**: last, and only if tables support it.

## 1. Contact force extraction (VERIFIED CODE FACT)

MuJoCo `mjContact.frame` is stored **transposed** (axes along rows). Contact-frame X is the **normal** (`frame[0:3]`), Y/Z are tangents (`[3:6]`, `[6:8]`). `mj_contactForce` returns 3D force then 3D torque in that frame: `wrench[0]=Fn`, `wrench[1]=Ft1`, `wrench[2]=Ft2` (Newtons / Newton-metres). Torque is zero when `condim` is 1 or 3.

World force on **geom2** (C `geom[1]`): `f = Fn * n + Ft1 * t1 + Ft2 * t2`. Force on cylinder is that vector if the cylinder is geom2, else its negative. Hand-frame vectors use `Rh.T @ v_world` with `Rh = xmat[hand]`.

Live sliding coefficient is `data.contact[i].friction[0]` (not YAML). Pair sliding in this MuJoCo generation follows the **max** of the two geom sliding coeffs (prior audit). `rho_i = |Ft|/(mu Fn)` is logged for A vs B vs C only; it is **not** automatically stick vs slip.

Existing helpers `envs/contact.py:_contact_normal` and `contact_debug` only keep `abs(wrench[0])` as Fn. They do **not** export Ft, live `contact.friction`, or world wrench. This audit uses a separate extractor (`extract_contacts` in `training/vertical_slip_contact_mechanics_audit.py`).

## 2. Live friction

Unique live `contact.friction[0]` at A/B/C: **[1.0]**.

Geom sliding coeffs (`model.geom_friction[0]`) on those contacts:
- object mu0=1.0  vs  geom_71 mu0=1.0
- object mu0=1.0  vs  geom_73 mu0=1.0
- object mu0=1.0  vs  geom_80 mu0=1.0
- object mu0=1.0  vs  geom_82 mu0=1.0

## 3. State times (RAW)

- A (end of 1.0 s slide, before RETURN): **t = 6.1760 s**
- B (vertical distal-slip onset): **t = 8.0060 s**
- C (established creep): **t = 8.7560 s**
- RETURN start: **t = 6.1780 s**
- VERTICAL reached (hold start): **t = 7.5320 s**

Onset B is **not** the first vertical_hold sample (that still has RETURN wrist rate). It is the first persistent `v_rel_h.z > 0.0008 m/s` after 0.30 s of vertical hold.

## State A table (RAW)

t=6.1760 phase=inward_slide wrist=59.44 deg mass=0.2000 kg

- r_h mm = [6.951, -0.0, 104.558]
- v_rel_h = [-0.00121, 0.00053, 0.00065]  (x, y, **z distal**)
- omega_rel_h = [-0.00019, -0.02403, 0.0002]
- g_h = [-8.4475, -0.0891, 4.9868]
- F_g_h = [-1.6895, -0.0178, 0.9974] N
- F_contact_h = [1.6863, 0.0183, -0.9977] N
- F_contact_h + F_g_h = [-0.0032, 0.0005, -0.0003] N
- Tau_contact_h = [0.0, -0.00097, -0.0] N m
- F_L_h = [0.8466, -8.9908, -0.4961]  F_R_h = [0.8397, 9.0092, -0.5015]
- a_COM_h = [0.0001, -0.0, -0.0] m/s^2
- m a_h = [0.0, -0.0, -0.0] N
- residual m*a - (Fc+Fg) hand = [0.0032, -0.0005, 0.0003]  |res|=0.0032 N
- nL/nR = 10/10  aperture=0.01788  tau=-18.0
- hand p = [0.49545, 0.00115, 0.58434]  v = [-0.00456, -0.00182, 0.00129]  w = [-0.00187, -0.00024, -0.0072]
- p_des = [0.49391, 0.00087, 0.5943]  pos_err = [-0.001534, -0.00028, 0.009966]  v_cmd = [0.0, 0.0, 0.0]
- pad L stats {'n': 10, 'x_mm': {'min': 5.793290292278151, 'max': 8.49948155629151, 'mean': 7.1248907883943495}, 'z_mm': {'min': 94.66588883009285, 'max': 111.3999232729169, 'mean': 103.27021624590363}, 'rho_mean': 0.10800508907312253, 'rho_max': 0.12191835170785141, 'Fn_sum': 8.992003100389452, 'pad_x_mm': [-8.500000000000018, 8.500000000000112], 'pad_z_mm': [94.3999999999997, 111.39999999999975]}
- pad R stats {'n': 10, 'x_mm': {'min': 5.799275540918635, 'max': 8.49952557766537, 'mean': 7.140249550032581}, 'z_mm': {'min': 94.70491242751397, 'max': 111.399917354439, 'mean': 103.2719732078884}, 'rho_mean': 0.1075847967330807, 'rho_max': 0.12111429240185555, 'Fn_sum': 9.0101797763386, 'pad_x_mm': [-8.500000000000018, 8.500000000000112], 'pad_z_mm': [94.3999999999997, 111.39999999999975]}

| side | geoms | pos_h mm | Fn | Ft1 | Ft2 | |Ft| | mu | mu Fn | rho | dist | dim |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L | object/geom_71 | 6.65,17.94,106.37 | 0.898 | -0.003 | -0.102 | 0.102 | 1.000 | 0.898 | 0.113 | -0.00012 | 4 |
| L | object/geom_71 | 5.79,17.94,111.40 | 0.899 | -0.010 | -0.109 | 0.109 | 1.000 | 0.899 | 0.122 | -0.00012 | 4 |
| L | object/geom_71 | 6.80,17.94,105.40 | 0.898 | -0.002 | -0.100 | 0.100 | 1.000 | 0.898 | 0.111 | -0.00012 | 4 |
| L | object/geom_71 | 5.84,17.94,111.40 | 0.898 | -0.009 | -0.109 | 0.110 | 1.000 | 0.898 | 0.122 | -0.00012 | 4 |
| L | object/geom_71 | 6.75,17.94,105.40 | 0.899 | -0.002 | -0.100 | 0.100 | 1.000 | 0.899 | 0.111 | -0.00012 | 4 |
| L | object/geom_73 | 7.60,17.94,100.34 | 0.899 | 0.004 | -0.092 | 0.093 | 1.000 | 0.899 | 0.103 | -0.00012 | 4 |
| L | object/geom_73 | 7.39,17.94,101.40 | 0.899 | 0.003 | -0.094 | 0.094 | 1.000 | 0.899 | 0.105 | -0.00012 | 4 |
| L | object/geom_73 | 8.50,17.94,94.93 | 0.901 | 0.011 | -0.084 | 0.085 | 1.000 | 0.901 | 0.094 | -0.00012 | 4 |
| L | object/geom_73 | 7.44,17.94,101.40 | 0.899 | 0.003 | -0.094 | 0.094 | 1.000 | 0.899 | 0.105 | -0.00012 | 4 |
| L | object/geom_73 | 8.50,17.94,94.67 | 0.901 | 0.011 | -0.084 | 0.085 | 1.000 | 0.901 | 0.094 | -0.00012 | 4 |
| R | object/geom_80 | 6.38,-17.94,108.05 | 0.902 | -0.004 | 0.104 | 0.104 | 1.000 | 0.902 | 0.115 | -0.00012 | 4 |
| R | object/geom_80 | 6.79,-17.94,105.40 | 0.902 | -0.001 | 0.100 | 0.100 | 1.000 | 0.902 | 0.111 | -0.00012 | 4 |
| R | object/geom_80 | 5.80,-17.94,111.40 | 0.903 | -0.009 | 0.109 | 0.109 | 1.000 | 0.903 | 0.121 | -0.00012 | 4 |
| R | object/geom_80 | 5.85,-17.94,111.40 | 0.902 | -0.009 | 0.109 | 0.109 | 1.000 | 0.902 | 0.121 | -0.00012 | 4 |
| R | object/geom_80 | 6.74,-17.94,105.40 | 0.903 | -0.001 | 0.100 | 0.100 | 1.000 | 0.903 | 0.111 | -0.00012 | 4 |
| R | object/geom_82 | 7.91,-17.94,98.60 | 0.899 | 0.007 | 0.090 | 0.090 | 1.000 | 0.899 | 0.100 | -0.00012 | 4 |
| R | object/geom_82 | 8.50,-17.94,94.96 | 0.900 | 0.012 | 0.084 | 0.085 | 1.000 | 0.900 | 0.094 | -0.00012 | 4 |
| R | object/geom_82 | 7.44,-17.94,101.40 | 0.899 | 0.004 | 0.094 | 0.094 | 1.000 | 0.899 | 0.104 | -0.00012 | 4 |
| R | object/geom_82 | 7.49,-17.94,101.40 | 0.899 | 0.004 | 0.094 | 0.094 | 1.000 | 0.899 | 0.105 | -0.00012 | 4 |
| R | object/geom_82 | 8.50,-17.94,94.70 | 0.901 | 0.012 | 0.084 | 0.085 | 1.000 | 0.901 | 0.094 | -0.00012 | 4 |

## State B table (RAW)

t=8.0060 phase=vertical_hold wrist=0.49 deg mass=0.2000 kg

- r_h mm = [6.03, -0.0, 107.517]
- v_rel_h = [0.00219, 0.00058, 0.0008]  (x, y, **z distal**)
- omega_rel_h = [-9e-05, 0.01438, 0.00061]
- g_h = [-0.0842, -0.0354, 9.8096]
- F_g_h = [-0.0168, -0.0071, 1.9619] N
- F_contact_h = [0.0176, 0.0042, -1.9755] N
- F_contact_h + F_g_h = [0.0008, -0.0029, -0.0136] N
- Tau_contact_h = [0.0, 0.00056, -0.0] N m
- F_L_h = [0.0186, -8.9979, -0.9907]  F_R_h = [-0.001, 9.0021, -0.9848]
- a_COM_h = [0.0005, 0.0001, 0.0004] m/s^2
- m a_h = [0.0001, 0.0, 0.0001] N
- residual m*a - (Fc+Fg) hand = [-0.0007, 0.0029, 0.0137]  |res|=0.0140 N
- nL/nR = 10/10  aperture=0.01788  tau=-18.0
- hand p = [0.49603, 0.00057, 0.57482]  v = [0.00499, -0.01303, -0.00976]  w = [0.02758, -0.00807, 0.00255]
- p_des = [0.49466, -0.00146, 0.585]  pos_err = [-0.001369, -0.002031, 0.01018]  v_cmd = [0.0, 0.0, 0.0]
- pad L stats {'n': 10, 'x_mm': {'min': 5.30195464287987, 'max': 8.288619977829267, 'mean': 6.727197836437363}, 'z_mm': {'min': 94.40002293409401, 'max': 111.39998252389839, 'mean': 103.18362604335205}, 'rho_mean': 0.11035818113373652, 'rho_max': 0.11219535329223282, 'Fn_sum': 8.997918355558095, 'pad_x_mm': [-8.499999999999956, 8.500000000000004], 'pad_z_mm': [94.39999999999971, 111.39999999999979]}
- pad R stats {'n': 10, 'x_mm': {'min': 5.297783723893574, 'max': 8.313852647275363, 'mean': 6.767053104962262}, 'z_mm': {'min': 94.40001906016407, 'max': 111.3999778990287, 'mean': 103.16757822638593}, 'rho_mean': 0.10966297199871466, 'rho_max': 0.11162226825112466, 'Fn_sum': 9.002052146176672, 'pad_x_mm': [-8.499999999999956, 8.500000000000004], 'pad_z_mm': [94.39999999999971, 111.39999999999979]}

| side | geoms | pos_h mm | Fn | Ft1 | Ft2 | |Ft| | mu | mu Fn | rho | dist | dim |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L | object/geom_71 | 5.80,17.94,108.70 | 0.900 | 0.005 | -0.100 | 0.100 | 1.000 | 0.900 | 0.111 | -0.00012 | 4 |
| L | object/geom_71 | 5.33,17.94,111.40 | 0.900 | 0.008 | -0.101 | 0.101 | 1.000 | 0.900 | 0.112 | -0.00012 | 4 |
| L | object/geom_71 | 6.35,17.94,105.40 | 0.901 | 0.002 | -0.099 | 0.099 | 1.000 | 0.901 | 0.110 | -0.00012 | 4 |
| L | object/geom_71 | 6.43,17.94,105.40 | 0.900 | 0.002 | -0.099 | 0.099 | 1.000 | 0.900 | 0.110 | -0.00012 | 4 |
| L | object/geom_71 | 5.30,17.94,111.40 | 0.900 | 0.008 | -0.101 | 0.101 | 1.000 | 0.900 | 0.112 | -0.00012 | 4 |
| L | object/geom_73 | 7.56,17.94,97.94 | 0.899 | -0.007 | -0.098 | 0.098 | 1.000 | 0.899 | 0.109 | -0.00012 | 4 |
| L | object/geom_73 | 7.03,17.94,101.40 | 0.899 | -0.003 | -0.099 | 0.099 | 1.000 | 0.899 | 0.110 | -0.00012 | 4 |
| L | object/geom_73 | 8.24,17.94,94.40 | 0.900 | -0.011 | -0.097 | 0.098 | 1.000 | 0.900 | 0.109 | -0.00012 | 4 |
| L | object/geom_73 | 8.29,17.94,94.40 | 0.900 | -0.011 | -0.097 | 0.098 | 1.000 | 0.900 | 0.109 | -0.00012 | 4 |
| L | object/geom_73 | 6.94,17.94,101.40 | 0.899 | -0.003 | -0.099 | 0.099 | 1.000 | 0.899 | 0.110 | -0.00012 | 4 |
| R | object/geom_80 | 6.24,-17.94,106.26 | 0.901 | 0.004 | 0.099 | 0.099 | 1.000 | 0.901 | 0.110 | -0.00012 | 4 |
| R | object/geom_80 | 6.34,-17.94,105.40 | 0.902 | 0.003 | 0.099 | 0.099 | 1.000 | 0.902 | 0.110 | -0.00012 | 4 |
| R | object/geom_80 | 5.36,-17.94,111.40 | 0.901 | 0.010 | 0.100 | 0.101 | 1.000 | 0.901 | 0.112 | -0.00012 | 4 |
| R | object/geom_80 | 6.43,-17.94,105.40 | 0.901 | 0.003 | 0.099 | 0.099 | 1.000 | 0.901 | 0.110 | -0.00012 | 4 |
| R | object/geom_80 | 5.30,-17.94,111.40 | 0.902 | 0.010 | 0.100 | 0.101 | 1.000 | 0.902 | 0.112 | -0.00012 | 4 |
| R | object/geom_82 | 7.26,-17.94,100.21 | 0.899 | -0.002 | 0.098 | 0.098 | 1.000 | 0.899 | 0.109 | -0.00012 | 4 |
| R | object/geom_82 | 8.26,-17.94,94.40 | 0.899 | -0.009 | 0.097 | 0.097 | 1.000 | 0.899 | 0.108 | -0.00012 | 4 |
| R | object/geom_82 | 7.14,-17.94,101.40 | 0.899 | -0.001 | 0.098 | 0.098 | 1.000 | 0.899 | 0.109 | -0.00012 | 4 |
| R | object/geom_82 | 8.31,-17.94,94.40 | 0.899 | -0.009 | 0.097 | 0.097 | 1.000 | 0.899 | 0.108 | -0.00012 | 4 |
| R | object/geom_82 | 7.03,-17.94,101.40 | 0.899 | -0.001 | 0.098 | 0.098 | 1.000 | 0.899 | 0.109 | -0.00012 | 4 |

## State C table (RAW)

t=8.7560 phase=vertical_hold wrist=0.80 deg mass=0.2000 kg

- r_h mm = [6.068, -0.0, 108.446]
- v_rel_h = [1e-05, 2e-05, 0.00125]  (x, y, **z distal**)
- omega_rel_h = [-5e-05, 0.01509, 0.00058]
- g_h = [-0.1372, -0.0618, 9.8088]
- F_g_h = [-0.0274, -0.0124, 1.9618] N
- F_contact_h = [0.0263, 0.0133, -1.9604] N
- F_contact_h + F_g_h = [-0.0011, 0.0009, 0.0013] N
- Tau_contact_h = [0.0, 0.00061, 0.0] N m
- F_L_h = [0.0218, -8.9934, -0.9798]  F_R_h = [0.0045, 9.0066, -0.9806]
- a_COM_h = [0.0, 0.0, -0.0] m/s^2
- m a_h = [0.0, 0.0, -0.0] N
- residual m*a - (Fc+Fg) hand = [0.0011, -0.0009, -0.0013]  |res|=0.0020 N
- nL/nR = 10/10  aperture=0.01788  tau=-18.0
- hand p = [0.49584, -0.00157, 0.57512]  v = [-0.00157, 0.00136, 0.0007]  w = [-0.00071, -0.00027, 0.00011]
- p_des = [0.49466, -0.00146, 0.585]  pos_err = [-0.001181, 0.000108, 0.009887]  v_cmd = [0.0, 0.0, 0.0]
- pad L stats {'n': 10, 'x_mm': {'min': 5.5557680253779615, 'max': 8.339883097420433, 'mean': 6.8748308984332915}, 'z_mm': {'min': 94.40002000700832, 'max': 111.39998376088236, 'mean': 103.22070674306967}, 'rho_mean': 0.1092580357199296, 'rho_max': 0.11095791361216167, 'Fn_sum': 8.993366966082116, 'pad_x_mm': [-8.499999999999963, 8.500000000000004], 'pad_z_mm': [94.39999999999974, 111.39999999999978]}
- pad R stats {'n': 10, 'x_mm': {'min': 5.556095769068252, 'max': 8.364074330751624, 'mean': 6.9082947431385415}, 'z_mm': {'min': 94.40001791553598, 'max': 111.39998106757102, 'mean': 103.2285518129894}, 'rho_mean': 0.1091919600332896, 'rho_max': 0.11109010099106005, 'Fn_sum': 9.006603054745337, 'pad_x_mm': [-8.499999999999963, 8.500000000000004], 'pad_z_mm': [94.39999999999974, 111.39999999999978]}

| side | geoms | pos_h mm | Fn | Ft1 | Ft2 | |Ft| | mu | mu Fn | rho | dist | dim |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L | object/geom_71 | 5.99,17.94,108.89 | 0.900 | 0.006 | -0.099 | 0.099 | 1.000 | 0.900 | 0.110 | -0.00012 | 4 |
| L | object/geom_71 | 5.59,17.94,111.40 | 0.900 | 0.009 | -0.099 | 0.100 | 1.000 | 0.900 | 0.111 | -0.00012 | 4 |
| L | object/geom_71 | 6.53,17.94,105.40 | 0.900 | 0.002 | -0.098 | 0.098 | 1.000 | 0.900 | 0.109 | -0.00012 | 4 |
| L | object/geom_71 | 6.58,17.94,105.40 | 0.900 | 0.002 | -0.098 | 0.098 | 1.000 | 0.900 | 0.109 | -0.00012 | 4 |
| L | object/geom_71 | 5.56,17.94,111.40 | 0.900 | 0.009 | -0.099 | 0.100 | 1.000 | 0.900 | 0.111 | -0.00012 | 4 |
| L | object/geom_73 | 7.64,17.94,98.12 | 0.899 | -0.007 | -0.097 | 0.097 | 1.000 | 0.899 | 0.108 | -0.00012 | 4 |
| L | object/geom_73 | 7.15,17.94,101.40 | 0.898 | -0.003 | -0.098 | 0.098 | 1.000 | 0.898 | 0.109 | -0.00012 | 4 |
| L | object/geom_73 | 8.29,17.94,94.40 | 0.899 | -0.012 | -0.096 | 0.097 | 1.000 | 0.899 | 0.108 | -0.00012 | 4 |
| L | object/geom_73 | 8.34,17.94,94.40 | 0.899 | -0.012 | -0.096 | 0.097 | 1.000 | 0.899 | 0.108 | -0.00012 | 4 |
| L | object/geom_73 | 7.08,17.94,101.40 | 0.898 | -0.003 | -0.098 | 0.098 | 1.000 | 0.898 | 0.109 | -0.00012 | 4 |
| R | object/geom_80 | 5.99,-17.94,108.89 | 0.902 | 0.008 | 0.099 | 0.099 | 1.000 | 0.902 | 0.110 | -0.00012 | 4 |
| R | object/geom_80 | 6.54,-17.94,105.40 | 0.902 | 0.004 | 0.098 | 0.099 | 1.000 | 0.902 | 0.109 | -0.00012 | 4 |
| R | object/geom_80 | 5.59,-17.94,111.40 | 0.901 | 0.011 | 0.100 | 0.100 | 1.000 | 0.901 | 0.111 | -0.00012 | 4 |
| R | object/geom_80 | 6.59,-17.94,105.40 | 0.902 | 0.004 | 0.098 | 0.098 | 1.000 | 0.902 | 0.109 | -0.00012 | 4 |
| R | object/geom_80 | 5.56,-17.94,111.40 | 0.902 | 0.011 | 0.100 | 0.100 | 1.000 | 0.902 | 0.111 | -0.00012 | 4 |
| R | object/geom_82 | 7.71,-17.94,98.19 | 0.900 | -0.005 | 0.097 | 0.097 | 1.000 | 0.900 | 0.108 | -0.00012 | 4 |
| R | object/geom_82 | 8.32,-17.94,94.40 | 0.900 | -0.010 | 0.097 | 0.097 | 1.000 | 0.900 | 0.108 | -0.00012 | 4 |
| R | object/geom_82 | 7.25,-17.94,101.40 | 0.899 | -0.001 | 0.098 | 0.098 | 1.000 | 0.899 | 0.109 | -0.00012 | 4 |
| R | object/geom_82 | 8.36,-17.94,94.40 | 0.900 | -0.010 | 0.096 | 0.097 | 1.000 | 0.900 | 0.108 | -0.00012 | 4 |
| R | object/geom_82 | 7.17,-17.94,101.40 | 0.900 | -0.001 | 0.098 | 0.098 | 1.000 | 0.900 | 0.109 | -0.00012 | 4 |

## 8. Force vs acceleration (RAW)

At A/B/C |m a - (F_contact + F_g)| = 0.003 / 0.014 / 0.002 N vs |Fg|~1.96 N. Extractor convention (force on geom2, negated if cylinder is geom1) is consistent enough to interpret wrenches.

## 11. Q1-Q8

**Q1.** g_h A=[-8.447, -0.089, 4.987]  B=[-0.084, -0.035, 9.81]. |g_x| A=8.45 B=0.08; g_z A=4.99 B=9.81. RETURN does rotate gravity in the hand frame toward the finger axis.

**Q2.** L centroid x/z mm A 7.12/103.27 -> B 6.73/103.18 -> C 6.87/103.22; R A 7.14/103.27 B 6.77/103.17.

**Q3.** F_contact_h A=[1.686, 0.018, -0.998] B=[0.018, 0.004, -1.976] C=[0.026, 0.013, -1.96].

**Q4.** rho max L/R A 0.122/0.121 B 0.112/0.112 C 0.111/0.111. rho is a diagnostic, not a stick/slip proof.

**Q5.** rho mean L/R A 0.108/0.108 B 0.110/0.110.

**Q6.** nL/nR A 10/10 B 10/10 C 10/10. z span L A {'min': 94.66588883009285, 'max': 111.3999232729169, 'mean': 103.27021624590363} B {'min': 94.40002293409401, 'max': 111.39998252389839, 'mean': 103.18362604335205}.

**Q7.** v_rel_h.z A=0.00065 (60 deg); B=0.00080; C=0.00125. g_h.z A=4.99 B=9.81 (world gravity dotted into hand-z). Persistent +hand-z relative motion after gravity is mostly +hand-z is the RAW fact to compare with the viewer.

**Q8.** At A/B/C |m a - (F_contact + F_g)| = 0.003 / 0.014 / 0.002 N vs |Fg|~1.96 N. Extractor convention (force on geom2, negated if cylinder is geom1) is consistent enough to interpret wrenches.

## 12. Mechanism class

**E. MIXED**

After RETURN, gravity is almost purely +hand-z (distal). Forces rebalance (Fc ~ -Fg, a~0) so this is quasi-static creep, not a falling acceleration. rho~0.11 so sliding is NOT at the Coulomb limit. condim=4 soft contacts can drift below mu Fn. Contact centroids stay mid-pad while COM e_z walks distal. Hand speed at C is small vs v_rel_z, so it is genuine object-relative distal motion.

This is INTERPRETATION after the tables. It is not a controller change.

## 14. Figures

- timeseries: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\inward_slide_return\figures\timeseries.png`
- pad-plane A/B/C: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\inward_slide_return\figures\pad_plane_ABC.png`

## 15. Viewer correspondence

USER VISUAL: supported 60-deg inward/down; after return, downward along fingers.

RAW correspondence:

- At 60 deg (A), gravity is mostly **-hand-x** (`g_h.x = -8.45`) with a smaller **+hand-z** (`g_h.z = +4.99`). `v_rel_h.z = +0.65 mm/s` (slow distal).
- After vertical (B/C), gravity is almost purely **+hand-z** (`g_h.z = +9.81`). That is world-down expressed in the grasp frame (hand-z is world-down at nominal pinch). Persistent `v_rel_h.z > 0` is distal = down the fingers.
- Do not use the first vertical_hold frame as slip onset: wrist rate is still ~0.76 rad/s there. B is 0.47 s after vertical reached (`t=8.006`).
- Established creep (C): `v_rel_h.z = +1.25 mm/s`, `e_z` climbs ~2.5 mm over the 2 s hold. Hand `|v|` at C is ~2 mm/s but `v_rel` is object-minus-hand, so the 1.25 mm/s is relative.

At A, `v_rel_h.z = 0.00065`; at B = `0.00080`; at C = `0.00125`.

## Q8 addendum (quasi-static)

During C, `m a ~ 0` and `F_contact + F_g ~ 0` (residual 2 mN). The cylinder is **not accelerating under a net force**. It is creeping at ~1.2 mm/s with forces balanced. That is incompatible with a naive "friction too small so it falls (accelerates at g)". It is compatible with regularized `condim=4` contacts (all contacts `dim=4`, live `mu=1.0`, `rho~0.11`).

Sample `solref` is in `contact_mechanics_ABC.json` per contact. MuJoCo 3.12.0, `opt.cone=0` (pyramidal), `impratio=1.0`.

## 16. Machine-readable

`results/diagnostics/inward_slide_return/contact_mechanics_timeseries.npz`
`results/diagnostics/inward_slide_return/contact_mechanics_ABC.json`
