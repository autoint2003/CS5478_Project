# Compound-orientation gravity reseating, visual audit

Replay of the established compound-orientation command on the same 6.5 m/s, −6 mm impact snapshot. No new search and no training. Capture state: bilateral, hand-frame position `[12.29, −0.03, 98.2]` mm, `g_h.x ≈ −0.1` m/s².

The five clips share one snapshot, one 2.64 s duration, and one camera rule. Frames are every 0.01 s and played at 20 fps, so the video is five times slower than the simulation (264 frames, 13.2 s). Support roll and pitch are the same simultaneous command used in the earlier audit. They are marked together, not split into a new sequence.

Videos, traces, and stills are in `results/diagnostics/raw/compound_orientation_gravity_rolling_visual_audit/`. The runner is `training/compound_orientation_visual_audit.py`.

## Cameras and overlay

Both cameras use a fixed world azimuth and elevation. The lookat point is the cylinder, so the fingers stay in frame while the hand pitches and lifts. The same offsets are used for every clip.

| view | distance | azimuth | elevation |
| --- | ---: | ---: | ---: |
| main | 0.50 m | 40° | −16° |
| close-up | 0.155 m | 42° | −8° |

The overlay is simulation time, the current phase, hand-frame x, `g_h.x`, the grip command, and left/right contact. The phase label is the event mark:

| clip time | phase | what starts |
| ---: | --- | --- |
| 0.00 s | `support+pitch` | support roll and pitch together, grip +1.00, 0.40 s |
| 0.40 s | `relax` or `secure` | grip −0.48, or grip held at +1.00 on the secure clip, 0.48 s |
| 0.88 s | `retighten` | grip +1.00, 0.16 s |
| 1.04 s | `handoff` | nominal continuation for the remaining 1.60 s |

## The five clips

Commands, all from the capture state:

| clip | command | hand x at 0 / 0.40 / 0.88 / 2.63 s | contacts at 2.63 s | object z at 2.63 s |
| --- | --- | --- | --- | ---: |
| nominal | unchanged nominal continuation | 12.3 / — / — / 85.3 mm | off | 0.404 m |
| wrist reseat | `[0.8, 0.6, 0, 1]` for 0.24 s, hold 0.12 s, then nominal | 12.3 / 12.1 / — / 11.2 mm | on | 0.636 m |
| inward + relaxed grip | `wy = +0.50`, grip −0.48 | 12.3 / 11.5 / 8.2 / 8.0 mm | on | 0.640 m |
| outward + relaxed grip | `wy = −0.50`, grip −0.48 | 12.3 / 13.6 / 111 / 464 mm | off | 0.018 m |
| inward + secure grip | `wy = +0.50`, grip stays +1.00 | 12.3 / 11.5 / 11.4 / 11.2 mm | on | 0.644 m |

Support rate is `wx = 0.28` for 20 steps on the three compound clips. `g_h.x` after that pitch is −6.0 m/s² inward and +6.0 m/s² outward.

The main view shows the gripper upright at the start, pitched down during the slope, the green cylinder between the fingers, and the hand held above the table after the handoff. The close-up shows both black pads on the cylinder. On the inward clip the overlay moves from 12.3 mm to 8.2 mm while both contacts stay on, the grip returns to nominal, and the cylinder is still between the pads at 2.63 s with the table below the hand. On the outward clip the close-up at 0.70 s shows the cylinder lying on the table and both contacts off. On the secure clip the same inward pitch leaves the overlay at 11.4 mm. Nominal is still holding at 2.0 s (x = 14.8 mm, both contacts on) and has lost the cylinder by the end of the same clip.

## Side-by-side

`comparison_inward_outward_secure.mp4` is the three close-ups locked to the same frames: inward slope, outward slope, secure grip. During the relax window the inward frame keeps both contacts and the hand-x readout falls; the outward frame loses the cylinder onto the table; the secure frame keeps the readout near 11.4 mm.

## Is the 4 mm a roll?

From capture to the end of the inward clip, hand x falls from 12.29 mm to 7.98 mm. That 4.3 mm is not one motion.

- During the secure pitch (0.00–0.40 s) hand x falls 0.77 mm while the hand itself moves 24 mm in the world. The hand frame is rotating, so this piece is not relative slip.
- During the relaxed window (0.40–0.88 s) the angular command is zero and hand x falls from 11.52 mm to 8.24 mm, a 3.27 mm change. The hand also translates 33 mm in the world in this window. The secure clip has the same 33 mm of hand motion and a hand-x change of 0.11 mm, so the 3.27 mm is motion of the cylinder inside the hand.
- After retightening, hand x creeps from 8.24 mm to 7.98 mm while the nominal lift holds the grasp.

Inside the relaxed window the relative rotation is 12.2°. In the hand frame that rotation is `[−11.9°, 0.14°, 2.3°]`. A roll that carried the center 3.27 mm along hand x would be a rotation about hand y. The hand-y component is 0.14°, which is an arc of 0.04 mm on the 18 mm radius. The 12° is a tip about hand x, not that roll.

Contact points, same window, millimeters:

| point | at grip release | at retighten |
| --- | --- | --- |
| left pad, hand frame | `[8.48, 17.64, 94.41]` | `[8.16, 18.00, 103.05]` |
| right pad, hand frame | `[8.48, −17.62, 94.41]` | `[8.03, −17.99, 100.38]` |
| left pad, object frame | `[17.64, −1.07, 6.29]` | `[17.98, −0.65, −0.33]` |
| right pad, object frame | `[−17.62, −1.46, 6.35]` | `[−17.99, 0.51, 2.38]` |

On the pads the contact moves about 8.6 mm along hand z and stays within 0.3 mm in hand x. On the cylinder the contact stays on the 18 mm radius. The circumferential coordinate moves less than 0.5 mm. The axial coordinate moves 6.6 mm on the left pad and 4.0 mm on the right. The contact patch migrates along the cylinder axis and along the finger. It does not walk around the circumference.

The secure clip on the same slope moves the pad contact 0.75 mm and rotates the cylinder 0.38° relative to the hand. Friction at grip +1 holds the post-pitch offset. Opening to grip −0.48 is what lets the cylinder move, and the direction follows the sign of `g_h.x`: inward slope reduces hand x, outward slope ejects the cylinder.

This is gravity-assisted reseating by sliding. The traces do not support calling it rolling.

## Traces

`nominal_trace.npz`, `wrist_trace.npz`, `inward_trace.npz`, `outward_trace.npz`, and `secure_trace.npz`. Each row is one rendered frame. Arrays are meters except `g_h` (m/s²), `grip` (command, NaN during nominal), and the contact counts.

- `rh` — object position in the hand frame
- `R_rel` — object orientation relative to the hand, 9 entries
- `cL_h`, `cR_h` — contact points on the two pads, hand frame
- `cL_o`, `cR_o` — the same contacts in the object frame
- `hand_p`, `hand_R`, `obj_p`, `obj_R` — hand and object pose
- `g_h` — gravity in the hand frame
- `grip` — grip command

VISUALLY VERIFIED GRAVITY-ASSISTED SLIDING
