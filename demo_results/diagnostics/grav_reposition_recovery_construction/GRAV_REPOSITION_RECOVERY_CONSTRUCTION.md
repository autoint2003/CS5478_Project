# Gravitational reposition recovery construction (noslip=1)

Privileged/manual construction. RULE was not used. SAC was not trained.

## USER VISUAL OBSERVATION

- Simple CONTROLLED SLIP: **CONFIRMED** (prior primitive viewer).
- S_FAIL recovery FULL: **PENDING** until the user watches `--mode full`.

## 1. noslip=1 S_FAIL revalidation

```json
[
  {
    "s": 2.0,
    "t0_ok": true,
    "why": "ok",
    "class": "DELAYED_FAIL",
    "lift": true,
    "drop": true,
    "t_drop": 11.54600000000213,
    "hold_survived": false,
    "end_nL": 0,
    "end_nR": 0
  }
]
```

S_FAIL s = **2.0**. ZERO class = **DELAYED_FAIL**.

## 2. Exact disturbed initial state

```json
{
  "rh": [
    0.008956006999076217,
    0.0013570630452643254,
    0.09874427344282068
  ],
  "e_x_mm": 8.956006999076216,
  "nL": 10,
  "nR": 0,
  "obj_z": 0.4942330816954122,
  "g_h": [
    0.0005316020907793605,
    -0.21023042049951923,
    9.807747085222765
  ],
  "cyl_h": [
    -0.20600824075729193,
    0.06379651695012288,
    0.9764684373624747
  ],
  "v_rel_h": [
    4.9487539426014316e-05,
    0.00027928961331014833,
    -1.1097685935665986e-05
  ],
  "w_rel_h": [
    0.009510158815047205,
    0.00031706104061703875,
    0.002210832867468722
  ]
}
```

## 3. Wrist-direction derivation

```json
{
  "e_x_mm": 8.999562741113408,
  "rh": [
    0.008999562741113408,
    -0.00018456672436501352,
    0.09891147969533796
  ],
  "g_h": [
    -0.003596063071135587,
    -0.17824838534112666,
    9.808379814294176
  ],
  "cyl_h": [
    -0.20178757053251878,
    0.0258524640947677,
    0.9790880585926938
  ],
  "nL": 7,
  "nR": 4,
  "rho": 0.556222218484266,
  "Fn_L": 8.61545024040547,
  "Fn_R": 9.303714952308399,
  "v_rel_h": [
    0.0005603791786501525,
    -0.0036466920941130147,
    8.387409010380334e-05
  ],
  "w_rel_h": [
    0.009111772408411016,
    -0.0009727302318906578,
    -0.015435926891689449
  ],
  "rule": "primitive +30 deg => g_h.x < 0 => Delta r_h.x < 0. If e_x>0, command +30 to drive object toward r_h.x=0.",
  "theta_cmd_deg": 30.0
}
```

## 4. Local controlled-slip regime

```json
{
  "tau": -3.0,
  "results": [
    {
      "tau": -3.0,
      "kind": "CONTROLLED_SLIP",
      "useful_mm": 0.5326070952785256,
      "rho": 0.6733345995022677,
      "nL": 10,
      "nR": 10
    }
  ]
}
```

## 5. Brake-target comparison

```json
[
  {
    "target_mm": 1.0,
    "hit": true,
    "lost": false,
    "t_hit": 5.541999999999612,
    "prog_at_brake_mm": 1.0020987146631184,
    "rh_at_brake": [
      0.00800689343983391,
      -3.401849802322979e-07,
      0.1015967909580769
    ],
    "end": {
      "t": 6.041999999999557,
      "rh": [
        0.007967494771609894,
        2.10462942302063e-06,
        0.1015902821955925
      ],
      "e_x": 0.007967494771609894,
      "e_z": 0.0025902821955925,
      "v_rel_h": [
        -7.39590108135634e-05,
        0.00039640656261858554,
        -2.4035822283522644e-05
      ],
      "w_rel_h": [
        -0.0014216036893411404,
        1.2500319095005777e-05,
        0.003702419829814966
      ],
      "nL": 10,
      "nR": 10,
      "obj_z": 0.480769997454811,
      "aperture": 0.017881260673530082,
      "tau": -18.0,
      "ctrl": -18.0,
      "g_h": [
        -4.747794079552945,
        -0.03869343497302308,
        8.58446586551847
      ],
      "rho_max": 0.20769284744252828,
      "rho_med": 0.11029174036826184,
      "Fn_L": 8.996460914559062,
      "Fn_R": 9.00381148969191,
      "cyl_h": [
        -0.20329529155411485,
        0.0007386878799073869,
        0.9791171935841697
      ],
      "wrist_deg": null
    },
    "score": 14.854914116224636
  },
  {
    "target_mm": 2.0,
    "hit": false,
    "lost": false,
    "prog": 1.3985679410404581,
    "score": -1000000.0
  },
  {
    "target_mm": 3.0,
    "hit": false,
    "lost": false,
    "prog": 1.3985679410404581,
    "score": -1000000.0
  },
  {
    "target_mm": 4.0,
    "hit": false,
    "lost": false,
    "prog": 1.3985679410404581,
    "score": -1000000.0
  }
]
```

## 6. Selected S_BRAKE

```json
{
  "target_mm": 1.0,
  "score": 14.854914116224636,
  "prog_at_brake_mm": 1.0020987146631184,
  "end_e_x_mm": 7.9674947716098945,
  "end_rh": [
    0.007967494771609894,
    2.10462942302063e-06,
    0.1015902821955925
  ],
  "end_nL": 10,
  "end_nR": 10,
  "end_rho": 0.20769284744252828,
  "end_vrel": 0.00040396264545183933,
  "reason": "Highest score among bilateral 0.5 s secure holds: smaller |e_x|, |e_z|, |v_rel|, |w_rel|, better cyl axis z, without using lift success."
}
```

## 7–9. Return / vertical hold / continuation (FULL)

```json
{
  "broke": null,
  "success": true,
  "cont": {
    "t0": 8.611999999999545,
    "t_lift_done": 10.862000000000297,
    "t_drop": null,
    "t_unilateral": null,
    "t_both_loss": null,
    "lift_completed": true,
    "drop": false,
    "hold_s": 10.0,
    "hold_survived": true,
    "end": {
      "t": 20.863999999999315,
      "rh": [
        0.009140465042116361,
        -1.4141230249459084e-05,
        0.10229377683310116
      ],
      "e_x": 0.009140465042116361,
      "e_z": 0.0032937768331011558,
      "v_rel_h": [
        0.00016135991371472985,
        4.440847715353372e-06,
        3.577806236399374e-05
      ],
      "w_rel_h": [
        0.001977364996526817,
        -2.144337079755992e-05,
        -0.009083366550205635
      ],
      "nL": 6,
      "nR": 5,
      "obj_z": 0.6217514550115449,
      "aperture": 0.01778788640034091,
      "tau": -18.0,
      "ctrl": -18.0,
      "g_h": [
        -0.05304590697512145,
        -0.15476556441725722,
        9.808635672295283
      ],
      "rho_max": 0.21773977898880445,
      "rho_med": 0.17517782872510476,
      "Fn_L": 8.979805638442315,
      "Fn_R": 9.014429447122648,
      "cyl_h": [
        -0.20328594047391305,
        0.0018202788508222426,
        0.9791177217222352
      ],
      "wrist_deg": null
    },
    "p0z": 0.5565220586220518,
    "noslip": {
      "mujoco": "3.12.0",
      "noslip_iterations": 1,
      "noslip_tolerance": 1e-06,
      "timestep": 0.002,
      "gravity": [
        0.0,
        0.0,
        -9.81
      ]
    }
  },
  "vert": {
    "t": 8.611999999999545,
    "rh": [
      0.00806181958841802,
      3.137273872353504e-08,
      0.10213055509554188
    ],
    "e_x": 0.00806181958841802,
    "e_z": 0.003130555095541873,
    "v_rel_h": [
      6.347350353720223e-05,
      2.5799462342438276e-06,
      1.7541143712668563e-06
    ],
    "w_rel_h": [
      0.00019234827453228068,
      -1.0861729825544406e-06,
      -0.0031638552039619855
    ],
    "nL": 10,
    "nR": 10,
    "obj_z": 0.4544534147585775,
    "aperture": 0.017882377792231726,
    "tau": -18.0,
    "ctrl": -18.0,
    "g_h": [
      -0.06462185434996363,
      -0.11143038957107008,
      9.809154259375282
    ],
    "rho_max": 0.21155289752512757,
    "rho_med": 0.10576127146960876,
    "Fn_L": 8.986305326096138,
    "Fn_R": 9.010830401076621,
    "cyl_h": [
      -0.20331970015113024,
      0.0014956897787272202,
      0.9791112615237023
    ],
    "wrist_deg": null
  },
  "wrist_end": 0.4440339083929748
}
```

## 10. ZERO vs ROTATE_ONLY vs NO_BRAKE vs FULL

```json
{
  "ZERO": {
    "mode": "ZERO",
    "class": "DELAYED_FAIL",
    "lift": true,
    "drop": true,
    "t_drop": 11.54600000000213,
    "hold_survived": false,
    "end_nL": 0,
    "end_nR": 0,
    "end_e_x_mm": 80.78751035308258,
    "end_obj_z": 0.43220192380338235
  },
  "ROTATE_ONLY": {
    "mode": "ROTATE_ONLY",
    "broke": null,
    "success": false,
    "lift": true,
    "drop": true,
    "t_drop": 9.828000000001971,
    "hold_survived": false,
    "end_nL": 0,
    "end_nR": 0,
    "end_e_x_mm": 66.36939754717473,
    "end_obj_z": 0.42996665083368224
  },
  "NO_BRAKE": {
    "mode": "NO_BRAKE",
    "broke": null,
    "success": true,
    "lift": true,
    "drop": false,
    "t_drop": null,
    "hold_survived": true,
    "end_nL": 10,
    "end_nR": 9,
    "end_e_x_mm": 7.686591112805681,
    "end_obj_z": 0.6293165527800667
  },
  "FULL": {
    "mode": "FULL",
    "broke": null,
    "success": true,
    "lift": true,
    "drop": false,
    "t_drop": null,
    "hold_survived": true,
    "end_nL": 6,
    "end_nR": 5,
    "end_e_x_mm": 9.140465042116361,
    "end_obj_z": 0.6217514550115449
  }
}
```

## 11. Raw trajectory evidence

See `raw/*.npz` and `figures/`.

## 12. Viewer commands

```text
python training/demo_grav_reposition_recovery.py --mode zero
python training/demo_grav_reposition_recovery.py --mode full
python training/demo_grav_reposition_recovery.py --mode rotate_only
python training/demo_grav_reposition_recovery.py --mode no_brake
```

## 13. USER VISUAL OBSERVATION: pending

## 14. Unresolved

None identified in this construction pass.