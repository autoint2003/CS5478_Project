"""Thin live overlay of OBS channels on frozen constructions. No MP4. No control change."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.observable_reward import read_tactile
from training.ballistic_slip_sufficiency import run_duration
from training.demo_airborne_recapture import advance_to_parent, make_parent_sim, save_parent, try_recapture
from training.demo_ballistic_impact import make_sim
from training.demo_ballistic_recovery import run_zero_full
from training.grav_reposition_v2_viz import apply_camera_preset
from training.map_ballistic_disturbance import CAM, _viewer_overlay


def overlay(sim, vwr, case: str):
    _, tac = read_tactile(sim.model, sim.data, sim.ids)
    _viewer_overlay(
        vwr,
        [
            ("CASE", case),
            ("t", f"{float(sim.data.time):.3f} s"),
            ("OBS e_hat mm", f"{1e3 * tac['e_hat_x']:+.2f}" if tac["estimate_valid"] else "invalid"),
            ("OBS valid L/R", f"{int(tac['valid_L'])}/{int(tac['valid_R'])}"),
            ("OBS cp L/R", f"{int(tac['contact_present_L'])}/{int(tac['contact_present_R'])}"),
            ("OBS u_L mm", "nan" if not tac["valid_L"] else f"{1e3 * tac['u_L']:+.2f}"),
            ("OBS Fn L/R", f"{tac['fn_L']:.2f}/{tac['fn_R']:.2f}"),
            ("keys", "SPACE pause  (mouse look)"),
        ],
    )
    vwr.sync()


def main():
    import mujoco.viewer

    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=("center6_zero", "stable_edge", "recapture"))
    args = p.parse_args()
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)

    if args.case == "recapture":
        sim = make_parent_sim()
        with mujoco.viewer.launch_passive(sim.model, sim.data) as vwr:
            apply_camera_preset(vwr, CAM)
            packed = advance_to_parent(sim, cfg)
            snap = save_parent(sim, packed)
            try_recapture(sim, gains, snap, 2.0, 0.04, 0.0, ctl={"sync": lambda: overlay(sim, vwr, "RECAPTURE")})
            while vwr.is_running():
                overlay(sim, vwr, "RECAPTURE HOLD")
                time.sleep(0.03)
        return

    sim, _ = make_sim()
    with mujoco.viewer.launch_passive(sim.model, sim.data) as vwr:
        apply_camera_preset(vwr, CAM)
        if args.case == "center6_zero":
            run_zero_full(sim, gains, ctl={"sync": lambda: overlay(sim, vwr, "CENTER6_ZERO")})
        else:
            from training.ballistic_large_angle import load_early

            snap = load_early()
            plan = {"theta_deg": 120.0, "tau_slip": -5.0, "slip_s": 0.70, "brake_s": 0.4, "omega_y": 3.0, "stage": "EARLY"}
            run_duration(sim, gains, snap, plan, 0.70)
        while vwr.is_running():
            overlay(sim, vwr, args.case)
            time.sleep(0.03)


if __name__ == "__main__":
    main()
