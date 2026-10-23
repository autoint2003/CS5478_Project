"""Probe Phase-0 tilt from IC toward fingertips-down."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.endpoint_only_axial_roll import (
    DOWN,
    UP,
    alpha_up_deg,
    gates,
    run_tilt_phase,
    tilt_omega_toward,
    u_open_world,
)
from data_generation.oracles.precision.force_slip_control import sample_state
from data_generation.oracles.precision.orientation_planner import joint_limit_margin
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.capture import restore_capture
from envs.simulation.ground_sim import make_ground_sim
from models.transformer_policy import DT_POLICY

B_POST = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl"
blob = pickle.loads(B_POST.read_bytes())


class E:
    pass


env = E()
env.sim, _ = make_ground_sim()
env.switch = GeometrySwitch()
env._hist = ObservableObsState(DT_POLICY)

restore_capture(env, blob["capture"])
print("start α", alpha_up_deg(env.sim), "u", u_open_world(env.sim), "jl", joint_limit_margin(env.sim))
w, ang, d = tilt_omega_toward(env.sim, DOWN, 0.2)
print("cmd ang_err_to_DOWN", ang, "w_body", w, "decomp", d)

for rate, tmax in ((0.20, 15.0), (0.30, 12.0), (0.15, 20.0)):
    restore_capture(env, blob["capture"])
    res = run_tilt_phase(
        env,
        {"capture": blob["capture"], "ref": blob["ref"]},
        target_dir_w=DOWN,
        phase="probe",
        rate_scale=rate,
        max_time_s=tmax,
        ang_tol_deg=4.0,
        contact_aware=False,
        allow_roll=False,
    )
    print(
        f"rate={rate} tmax={tmax} ok={res['ok']} reason={res['reason']} "
        f"α={res['alpha_final']:.1f} n={len(res['actions'])} jl={joint_limit_margin(env.sim):.3f}"
    )
    if res.get("series"):
        print("  last", res["series"][-1])
