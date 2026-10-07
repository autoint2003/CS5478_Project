# Compound-orientation gravity rolling

The previous tilt used only rotation about hand x. That put gravity across the pads and left the hand-x component near zero. This test adds pitch, so gravity along hand x can be set while a support tilt is held fixed. No policy was trained.

The snapshot is the 6.5 m/s head-on impact with a −6 mm axial offset. At capture the grasp is bilateral, hand-frame position is `[12.3, 0.0, 98.2]` mm, and gravity is `[−0.1, −0.2, 9.81]` m/s². Nominal continuation of this impact drops. The wrist reseat retains it. Inward means hand x decreasing, toward the pre-impact value near 0.3 mm, which requires `g_h.x < 0`.

Raw logs are in `results/diagnostics/raw/compound_orientation_gravity_rolling/`. The runner is `training/compound_orientation_gravity_rolling.py`.

## 1. Matched orientations

Every orientation uses the same support command: 20 steps, 0.40 s, at 1.12 rad/s about `r_des` x. The pitch rate about `r_des` y is the only change. Grip stays secure during this move. All seven pitches in the grid stay bilateral. Pad-height difference stays between 10.4 and 11.2 mm.

| pitch rate | `g_h` (m/s²) | `g_h.x` | hand x after the pitch | pad-height difference |
| --- | --- | ---: | ---: | ---: |
| −0.50 | `[5.99, 3.00, 7.17]` | +5.99 | 13.6 mm | −10.4 mm |
| −0.28 | `[3.54, 3.14, 8.59]` | +3.54 | 13.1 mm | −10.9 mm |
| 0 | `[0.01, 3.20, 9.27]` | +0.01 | 12.6 mm | −11.2 mm |
| +0.28 | `[−3.53, 3.13, 8.60]` | −3.53 | 12.0 mm | −11.0 mm |
| +0.50 | `[−5.98, 2.99, 7.18]` | −5.98 | 11.5 mm | −10.5 mm |

The support tilt is what produces the hand-y gravity of about +3 m/s² and the 11 mm pad step. Pitch sets the hand-x component, from about +6 m/s² (outward) through zero to about −6 m/s² (inward), without giving up bilateral contact.

## 2. How far the grip opens

On the moderate inward pitch (`g_h.x = −3.5`), three relaxations were applied for 0.28 s:

| tendon torque | both pads stay | hand-x change | normal force at the end |
| --- | --- | ---: | --- |
| −5.0 N·m | yes | −0.7 mm | 2.9 N and 2.2 N |
| −3.8 N·m | yes | −1.5 mm | 2.8 N and 1.5 N |
| −2.5 N·m | no, a short one-sided patch | −1.9 mm | 1.7 N and 0.9 N |

The matched comparison uses −3.8 N·m. Aperture moves from about 17.4 mm to 17.9 mm. The −2.5 N·m opening is the first one that leaves a clean bilateral trace.

## 3. Matched causal test

Same snapshot, same support rate, same 0.32 s relaxation. Only the pitch, and for the last row the grip, changes.

| condition | `g_h.x` at the start of the opening | hand x | late hand-x velocity | contacts |
| --- | ---: | --- | ---: | --- |
| neutral pitch, relaxed | +0.01 | 12.6 → 14.2 mm (+1.6) | +14 mm/s | bilateral, then the pad contact runs to z = 111 mm |
| inward pitch, relaxed | −3.7 | 11.9 → 10.3 mm (−1.7) | about −3 mm/s | bilateral; left contact z 94 → 103 mm |
| stronger inward, relaxed | −6.2 | 11.5 → 9.4 mm (−2.2) | about −4 mm/s | bilateral; left contact z 95 → 102 mm |
| outward pitch, relaxed | +3.7 | 13.2 → 66 mm (+53) | +0.31 m/s | bilateral for nine steps, then both pads off |
| inward pitch, secure grip | −3.7 | 12.0 → 11.9 mm (−0.1) | about 0 | bilateral; contact stays at z ≈ 95 mm |

Reversing `g_h.x` reverses the hand-x motion. The inward slopes move the cylinder toward smaller x and shift the contact a few millimetres along the pad, while both fingers stay on. The outward slope of the same size accelerates the cylinder out of the hand. With `g_h.x` near zero the cylinder drifts outward rather than sitting still. Secure grip on the inward slope does not unlock that slide: normal force stays near 9 N and hand x changes by 0.1 mm.

The inward samples keep falling for the whole 0.32 s. They slow down, from a few centimetres per second at the start of the opening to a few millimetres per second at the end. They do not reverse.

## 4. Longer opening and handoff

Holding the stronger inward pitch and the −3.8 N·m grip for 0.80 s, still bilateral, takes hand x from 11.5 mm to 8.2 mm. The trace is still decreasing at the last sample (8.52, 8.45, 8.39, 8.32, 8.26 mm). A slightly looser grip, −3.2 N·m for 0.48 s, also stays bilateral and ends at 8.2 mm. The moderate slope for 0.80 s ends at 9.0 mm.

Retightening the −3.2 N·m run for 0.16 s and handing off to the nominal controller leaves hand x at 8.2 mm, both pads in contact. That continuation is retained at t = 12, object height 0.64 m.

From the capture value of 12.3 mm, the hand-x offset is reduced by about 4 mm. It is not returned to the pre-impact 0.3 mm. The slide is still slow at 8 mm, so friction is limiting the rate, but it has not locked the cylinder at the post-impact offset.

## 5. Wrist reseat on the same snapshot

`[0.8, 0.6, 0, 1]` for 0.24 s changes hand x by −0.1 mm. The contact patch stays at z = 94 mm. Relative rotation is 1.6°. Nominal continuation retains, height 0.63 m.

Both commands survive to t = 12. The wrist command does it without moving hand x or the contact. The compound orientation moves hand x by about 4 mm and shifts the contact along the pad before the handoff. Those are different seating paths.

## 6. Answer

Pitch on top of a fixed support tilt sets `g_h.x` from about −6 to +6 m/s² while both pads stay loaded and the pad step stays near 11 mm. Opening to −3.8 N·m then lets the cylinder move with the sign of that component: inward gravity reduces hand x, outward gravity throws it out of the grasp, and a secure grip on the same inward slope does not move. Extending the inward opening reduces the post-impact offset from 12.3 mm to 8.2 mm and the nominal handoff retains through t = 12.

COMPOUND ORIENTATION PRODUCES INWARD GRAVITY-DRIVEN RESEATING
