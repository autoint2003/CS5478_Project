# Full-3D recatch basin of the rejected policy states

The 52 states are the policy-visited pre-loss states that the previous DAgger round rejected. That label ran a closed hold until the first persistent loss, then the existing vertical intercept. A state here is inside the v3 basin when some command inside recovery7d_airborne_v3 — vx and vy within ±1 m/s, the same asymmetric vertical bounds, wrist within ±4 rad/s, the same grip map, torque limits on — regrasps both pads for 1 s.

## Geometry, separated by component

Lateral position error has median 2.625 mm and 90th percentile 4.039 mm. Vertical position error has median 16.06 mm. Lateral relative speed has median 0.207 m/s. Vertical relative speed has median 0.218 m/s and 90th percentile 0.724 m/s. Cylinder-axis tilt from the seated alignment has median 0.79 deg. Object angular speed has median 5.777 rad/s. Finger opening has median 9.775 mm. Distal-tip clearance has median -5.652 mm. 52 of 52 are still in bilateral contact.

| class | count |
|---|---:|
| CURRENT_INTERCEPT | 0 |
| LATERAL_ONLY | 0 |
| ORIENTATION_ONLY | 0 |
| EITHER_LATERAL_OR_ORIENTATION | 0 |
| COMBINED_ONLY | 0 |
| UNRECOVERABLE | 52 |

Direct use of the current intercept recovers 0. Lateral-first chase, with or without wrist, is involved in 0. Wrist alignment, with or without the lateral-first law, is involved in 0. 52 remain without a successful strategy in this search.

## Policy action on the recoverable states

No state in this set was recovered by the searched v3 strategies, so there is no recovered state on which to score the Transformer's lateral command.

## States

| tag | why rejected before | lat mm | vert mm | v_xy | v_z | tilt deg | finger mm | clear mm | contact | class | policy vx,vy,vz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| dz-13.15_dy-0.2 | no_loss | 2.77 | -9.11 | 0.6763 | -0.0326 | 0.78 | 8.84 | 0.071 | 12/14 | UNRECOVERABLE | 0.02,-0.60,0.31 |
| dz-13.05_dvz-0.012 | no_loss | 2.2 | -5.93 | 0.0632 | -0.0031 | 0.09 | 13.18 | -0.09 | 8/19 | UNRECOVERABLE | 0.12,-0.37,0.28 |
| dvz_m | no_loss | 2.4 | -0.49 | 0.1151 | 0.0128 | 0.21 | 13.7 | -3.956 | 14/23 | UNRECOVERABLE | 0.12,-0.42,0.93 |
| dz-13.15_dvz-0.012 | no_loss | 2.63 | -2.93 | 0.0776 | 0.0146 | 1.28 | 13.0 | -2.653 | 14/21 | UNRECOVERABLE | 0.12,-0.36,0.27 |
| dz-13.50_dy-0.1 | no_loss | 1.95 | -7.37 | 0.6294 | 0.0162 | 0.61 | 8.54 | -1.447 | 10/18 | UNRECOVERABLE | 0.01,-0.54,0.30 |
| dz-13.10 | no_loss | 1.93 | -3.75 | 0.0569 | 0.0204 | 1.21 | 13.53 | -1.452 | 6/12 | UNRECOVERABLE | 0.13,-0.43,0.93 |
| dz-13.45 | no_loss | 5.4 | 2.79 | 0.2104 | 0.032 | 5.5 | 9.12 | -11.622 | 21/23 | UNRECOVERABLE | 0.02,-0.08,0.36 |
| dz-13.45_dvz0.012 | no_loss | 3.77 | 5.74 | 0.2226 | 0.0814 | 1.23 | 9.31 | -11.775 | 21/25 | UNRECOVERABLE | 0.02,-0.08,0.36 |
| dvz_p | no_loss | 2.54 | 7.2 | 0.1181 | 0.0996 | 0.04 | 11.03 | -9.432 | 19/23 | UNRECOVERABLE | 0.03,-0.09,0.37 |
| dz-13.05_dy0.2 | no_loss | 4.44 | 8.73 | 0.2051 | 0.1565 | 1.92 | 10.14 | -11.852 | 18/26 | UNRECOVERABLE | 0.02,-0.09,0.36 |
| dz14.6 | chase_fail | 2.3 | 16.52 | 0.1144 | 0.1729 | 0.8 | 10.86 | -8.755 | 10/15 | UNRECOVERABLE | 0.03,-0.07,0.36 |
| dz-13.50_dy-0.2 | chase_fail | 3.78 | 15.6 | 0.0951 | 0.184 | 2.26 | 10.43 | -10.199 | 11/23 | UNRECOVERABLE | 0.02,-0.10,0.37 |
| dz-14.20_dvz-0.006 | chase_fail | 1.9 | 16.61 | 0.1141 | 0.194 | 1.33 | 9.7 | -9.506 | 14/18 | UNRECOVERABLE | 0.03,-0.09,0.37 |
| dz-14.20_dvz0.006 | chase_fail | 1.91 | 14.78 | 0.1024 | 0.1944 | 0.12 | 9.85 | -9.822 | 18/23 | UNRECOVERABLE | 0.03,-0.08,0.37 |
| dz-13.45_dvz0.006 | chase_fail | 2.62 | 17.65 | 0.4071 | 0.1984 | 1.35 | 10.85 | -8.881 | 7/17 | UNRECOVERABLE | 0.02,-0.07,0.37 |
| dz-13.50_dvz0.006 | chase_fail | 3.11 | 18.37 | 0.1248 | 0.2076 | 2.96 | 10.39 | -8.859 | 9/17 | UNRECOVERABLE | 0.02,-0.10,0.37 |
| dz-13.70_dy0.2 | chase_fail | 3.55 | 17.77 | 0.0589 | 0.2397 | 0.3 | 9.04 | -11.232 | 8/18 | UNRECOVERABLE | 0.03,-0.07,0.37 |
| dz-14.20_dy0.1 | chase_fail | 2.37 | 18.75 | 0.0936 | 0.245 | 0.63 | 9.29 | -9.423 | 9/18 | UNRECOVERABLE | 0.03,-0.08,0.37 |
| dz-14.60_dvz0.012 | chase_fail | 4.03 | 14.65 | 0.1584 | 0.2474 | 1.88 | 9.26 | -12.436 | 14/22 | UNRECOVERABLE | 0.03,-0.07,0.37 |
| dz-14.15_dvz-0.006 | chase_fail | 3.78 | 17.77 | 0.2092 | 0.2653 | 0.03 | 8.83 | -11.515 | 13/22 | UNRECOVERABLE | 0.03,-0.07,0.36 |
| dz-14.05_dvz-0.012 | chase_fail | 1.97 | 19.16 | 0.1195 | 0.2653 | 0.34 | 9.39 | -8.728 | 10/19 | UNRECOVERABLE | 0.02,-0.08,0.37 |
| dz-14.05 | chase_fail | 2.51 | 25.23 | 0.1398 | 0.3367 | 0.99 | 10.93 | -3.815 | 2/8 | UNRECOVERABLE | 0.01,-0.11,0.25 |
| dz-14.05_dvz0.012 | chase_fail | 2.65 | 25.76 | 0.1258 | 0.3657 | 1.47 | 10.85 | -3.542 | 2/4 | UNRECOVERABLE | 0.01,-0.11,0.25 |
| dz-14.20_dy-0.2 | chase_fail | 6.93 | 21.46 | 0.1696 | 0.3671 | 13.67 | 8.99 | -8.86 | 8/11 | UNRECOVERABLE | 0.01,-0.21,0.20 |
| dz-14.35_dy-0.2 | chase_fail | 7.69 | 21.49 | 0.187 | 0.3876 | 14.9 | 8.95 | -9.141 | 5/12 | UNRECOVERABLE | 0.01,-0.21,0.20 |
| dz-14.35 | chase_fail | 3.41 | 22.05 | 0.3055 | 0.4514 | 2.54 | 8.6 | -8.474 | 7/12 | UNRECOVERABLE | 0.03,-0.07,0.37 |
| dz-14.15_dvz-0.012 | chase_fail | 4.56 | 22.61 | 0.3044 | 0.4824 | 2.24 | 8.27 | -8.695 | 7/12 | UNRECOVERABLE | 0.03,-0.07,0.37 |
| dz-14.35_dvz-0.012 | chase_fail | 3.45 | 23.92 | 0.1893 | 0.5478 | 2.05 | 9.13 | -6.698 | 4/11 | UNRECOVERABLE | 0.03,-0.06,0.36 |
| dz-14.55_dvz-0.006 | chase_fail | 2.11 | 24.0 | 0.2245 | 0.5777 | 1.86 | 9.32 | -5.625 | 6/10 | UNRECOVERABLE | 0.02,-0.08,0.37 |
| dz-14.05_dy0.2 | chase_fail | 3.82 | 25.43 | 0.3498 | 0.7256 | 1.4 | 6.77 | -6.204 | 6/10 | UNRECOVERABLE | 0.02,-0.07,0.37 |
| dz-14.45_dy-0.2 | chase_fail | 4.04 | 25.82 | 0.3463 | 0.7511 | 2.81 | 6.71 | -6.028 | 5/9 | UNRECOVERABLE | 0.02,-0.07,0.37 |
| dz-14.35_dy-0.1 | chase_fail | 3.76 | 26.08 | 0.3437 | 0.8006 | 2.01 | 6.71 | -5.626 | 5/9 | UNRECOVERABLE | 0.02,-0.07,0.37 |
| dz-13.10_dy0.1 | chase_fail | 2.37 | 29.34 | 0.8451 | 0.8257 | 2.26 | 6.54 | -2.2 | 3/4 | UNRECOVERABLE | 0.01,-0.53,0.30 |
| dz-13.70_dy-0.2 | chase_fail | 3.77 | 27.77 | 0.2694 | 0.8486 | 1.86 | 8.07 | -3.541 | 1/8 | UNRECOVERABLE | 0.02,-0.08,0.37 |
| dz-14.15_dy0.2 | chase_fail | 2.31 | 29.17 | 0.2539 | 0.9475 | 0.87 | 6.55 | -2.222 | 2/5 | UNRECOVERABLE | 0.03,-0.06,0.36 |
| dz-14.05_dvz-0.006 | no_loss | 2.01 | -9.8 | 0.4145 | -0.115 | 2.19 | 7.96 | 0.557 | 9/10 | UNRECOVERABLE | 0.02,-0.03,0.24 |
| dz-14.45_dy-0.2 | no_loss | 1.87 | -9.32 | 0.3997 | -0.0927 | 0.53 | 7.81 | 0.493 | 11/12 | UNRECOVERABLE | 0.02,-0.03,0.24 |
| dz-14.15 | no_loss | 1.71 | -8.23 | 0.385 | -0.0764 | 0.35 | 9.41 | 0.497 | 13/13 | UNRECOVERABLE | 0.02,-0.03,0.24 |
| dz-13.50_dvz-0.006 | no_loss | 2.47 | -9.03 | 0.3918 | -0.0659 | 0.62 | 8.93 | 0.144 | 12/14 | UNRECOVERABLE | 0.03,-0.61,0.30 |
| dz-14.85_dvz0.006 | no_loss | 2.13 | -5.8 | 0.1435 | -0.0537 | 0.61 | 11.96 | -0.808 | 6/17 | UNRECOVERABLE | 0.06,-0.37,1.00 |
| dz-13.15 | no_loss | 1.4 | 7.04 | 0.0777 | 0.0084 | 0.33 | 17.1 | -1.274 | 9/10 | UNRECOVERABLE | 0.01,-0.64,0.33 |
| dz-13.05_dy0.2 | no_loss | 2.51 | 6.85 | 0.025 | 0.0484 | 0.6 | 15.17 | -5.293 | 19/6 | UNRECOVERABLE | 0.06,-0.37,1.00 |
| dz-13.20_dvz-0.006 | no_loss | 2.99 | 8.34 | 0.0165 | 0.0588 | 0.69 | 15.15 | -5.678 | 21/8 | UNRECOVERABLE | 0.06,-0.38,1.00 |
| dz-13.70_dy0.2 | no_loss | 3.51 | 9.83 | 0.0096 | 0.1151 | 0.54 | 14.06 | -7.231 | 21/10 | UNRECOVERABLE | 0.06,-0.37,1.00 |
| dz-13.20_dvz0.012 | chase_fail | 3.58 | 13.03 | 0.2164 | 0.2279 | 0.12 | 13.92 | -7.536 | 18/9 | UNRECOVERABLE | -0.01,-0.02,1.00 |
| dz-13.45_dvz-0.006 | chase_fail | 2.67 | 18.02 | 0.6085 | 0.4087 | 0.41 | 14.36 | -5.535 | 8/4 | UNRECOVERABLE | -0.00,-0.02,1.00 |
| dz-13.20_dvz0.006 | chase_fail | 2.85 | 13.72 | 0.8709 | 0.4808 | 0.35 | 15.35 | -5.014 | 10/22 | UNRECOVERABLE | 0.02,-0.61,0.32 |
| dz13.2 | chase_fail | 3.05 | 17.7 | 0.5053 | 0.5162 | 0.28 | 14.45 | -5.847 | 10/4 | UNRECOVERABLE | -0.00,-0.03,1.00 |
| dz-13.15_dvz-0.012 | no_loss | 1.53 | 6.09 | 0.4877 | 0.627 | 0.65 | 15.99 | -2.846 | 7/15 | UNRECOVERABLE | 0.02,-0.08,0.37 |
| dz-13.15_dy-0.2 | no_loss | 1.33 | 7.16 | 0.461 | 0.6361 | 0.4 | 16.1 | -2.862 | 7/12 | UNRECOVERABLE | 0.02,-0.08,0.37 |
| dz-13.20_dy-0.2 | chase_fail | 2.62 | 17.85 | 0.6544 | 0.7144 | 0.1 | 14.78 | -4.955 | 10/4 | UNRECOVERABLE | 0.05,-0.38,1.00 |
| dz-13.15_dy0.1 | chase_fail | 0.1 | 19.97 | 0.0003 | 0.0144 | 0.04 | 16.74 | -0.214 | 4/6 | UNRECOVERABLE | -0.00,-0.00,-0.00 |

THE POLICY-VISITED STATES ARE TRULY OUTSIDE THE V3 RECATCh BASIN
