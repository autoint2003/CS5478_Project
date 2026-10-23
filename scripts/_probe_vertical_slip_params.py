"""Bounded DualLoop slip probe at saved fingertips-up pose."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.finger_supported_vertical import (  # noqa: E402
    U_OPEN_HAND,
    opening_alignment,
)
from data_generation.oracles.precision.force_slip_control import (  # noqa: E402
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import _height_hold_v_hand  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/finger_supported_vertical_axial"
blob = pickle.loads((OUT / "snapshots/best_vertical.pkl").read_bytes())
ref_blob = pickle.loads(
    (ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl").read_bytes()
)
blob["ref"] = ref_blob["ref"]


class E:
    pass


env = E()
env.sim, _ = make_ground_sim()
env.switch = GeometrySwitch()
env._hist = ObservableObsState(DT_POLICY)

configs = [
    dict(name="stiff_prev", FN_MIN=6.5, FN_MAX=14, FN_INIT=9.0, KN=400, V=-0.001, a0=0.55),
    dict(name="sign_aware", FN_MIN=3.8, FN_MAX=11, FN_INIT=5.2, KN=900, V=-0.001, a0=0.55),
    dict(name="soft_default", FN_MIN=3.5, FN_MAX=12, FN_INIT=6.0, KN=800, V=-0.001, a0=0.3),
    dict(name="soft_slow", FN_MIN=3.0, FN_MAX=10, FN_INIT=4.5, KN=600, V=-0.0007, a0=0.2),
]

d = -U_OPEN_HAND
results = []
for cfg in configs:
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    e_x0 = float(s0["e_signed_mm"][0])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    oa = opening_alignment(env.sim)
    ctl = DualLoopController(
        force=ForcePController(kp=0.10, da_max=0.12, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=cfg["V"], fn_min=cfg["FN_MIN"], fn_max=cfg["FN_MAX"], kn=cfg["KN"]),
        use_slip=True,
    )
    ctl.reset(
        a0=cfg["a0"],
        fn0=s0["fn_tot"],
        fn_target=cfg["FN_INIT"],
        slip_axis=tuple((-d).tolist()),
    )
    lost = 0
    reached = False
    abort = None
    series = []
    for _ in range(int(10 / DT)):
        s = sample_state(env.sim)
        _, meta = ctl.step(s)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        disp = float(np.dot(dp, d)) * 1e3
        axis = np.asarray((-d), float)
        v_along = float(np.dot(np.asarray(s2["v_rel_h"], float), axis))
        series.append((disp, v_along, meta["fn_target"], s2["fn_tot"], float(s2["e_signed_mm"][2])))
        if failed or latched(env.sim) or not s2["bilateral"]:
            abort = "contact_loss"
            break
        if abs(float(s2["e_signed_mm"][0]) - e_x0) > 4:
            abort = "lateral"
            break
        if disp >= 2.0:
            reached = True
            break
    row = {
        "name": cfg["name"],
        "reached": reached,
        "disp": series[-1][0] if series else None,
        "abort": abort,
        "max_disp": max(r[0] for r in series) if series else None,
        "min_fn_tgt": min(r[2] for r in series) if series else None,
        "final_ez": series[-1][4] if series else None,
        "mean_v_along": float(np.mean([r[1] for r in series])) if series else None,
        "alpha": oa["alpha_open_up_deg"],
        "gdot": oa["g_dot_d_inward"],
    }
    results.append(row)
    print(row, flush=True)

(OUT / "slip_param_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("wrote", OUT / "slip_param_probe.json", flush=True)
