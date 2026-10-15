"""Cartesian gains for the verified catch. The 7D action map is unchanged.

TRACK is the gain set passed to apply_cartesian_ctrl during legal recovery
and the airborne chase. It is not an optimizer schedule.
"""

TRACK = dict(kp_pos=1400.0, kd_pos=80.0, kp_ori=80.0, kd_ori=6.0, kp_null=2.0, kd_null=0.4)
