import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.precision.run_sign_aware_axial_sliding as M
from data_generation.oracles.precision.sign_aware_axial import desire_direction, waypoint_reorient

env = M.make_env()
blob = M.load_blob(M.B_POST_LAT)
u = np.asarray([0.0, 0.0, 1.0])
d = desire_direction("B_inward", u)
for ang in (70, 85, 100, 115):
    for rate in (0.12, 0.15, 0.20):
        tr = waypoint_reorient(
            env,
            blob,
            d=d,
            axis_name="+wy",
            angle_deg=ang,
            rate_scale=rate,
            fn_target=12.5,
            dx_abort_mm=3.0,
            dtheta_abort_deg=3.0,
        )
        g = np.asarray(tr.get("g_h") or [0, 0, 0], float)
        print(
            f"+wy@{ang} r={rate}: ok={tr.get('ok')} useful={tr.get('useful')} "
            f"dex={tr.get('delta_ex_mm')} dth={tr.get('delta_theta_deg')} "
            f"g={np.round(g, 2)} reason={tr.get('reason')}"
        )
