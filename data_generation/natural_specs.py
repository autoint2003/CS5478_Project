"""Seated natural-loss offsets used by the autonomous training episodes."""

TRAIN_NATURAL = (
    {"tag": "dz13.2", "dz": -13.2},
    {"tag": "dz13.5", "dz": -13.5},
    {"tag": "dz14.2", "dz": -14.2},
    {"tag": "dz14.6", "dz": -14.6},
    {"tag": "dvz_m", "dz": -13.5, "dvz": -0.01},
    {"tag": "dvz_p", "dz": -13.5, "dvz": 0.005},
)
EVAL_NATURAL = (
    {"tag": "h_dz14.00", "dz": -14.00},
    {"tag": "h_dz14.25", "dz": -14.25},
    {"tag": "h_dy", "dz": -14.10, "dy": 0.12},
    {"tag": "h_dx", "dz": -14.70, "dx": 0.10},
)
