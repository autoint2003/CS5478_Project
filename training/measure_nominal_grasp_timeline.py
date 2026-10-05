"""Measure current nominal grasp timeline (ball parked). Diagnostic only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.nominal import grasp_orientation  # noqa: E402
from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.physical_recovery import Z_AIR, physical_pack  # noqa: E402
from training.replay_core import MASS, MU, ImpactSim, isolate_ball_object_only  # noqa: E402


def measure(timeout: float = 12.0) -> dict:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = ImpactSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    isolate_ball_object_only(sim)
    sim.set_ball_mass(0.10)
    sim.park_ball()
    dt = float(sim.model.opt.timestep)
    n = int(round(timeout / dt))
    events = {
        "t0": 0.0,
        "t_first_finger_object_contact": None,
        "t_close_start": None,
        "t_close_complete": None,
        "t_lift_start": None,
        "t_first_airborne": None,
        "t_captured": None,
        "t_fsm_hold_success": None,
        "t_z_air_plus_0p25": None,
        "t_recovered_instant": None,
        "definitions": {
            "t_first_finger_object_contact": "first step with nL>0 or nR>0 (finger-object contacts)",
            "t_close_start": "first step with fsm.phase==close",
            "t_close_complete": "first step with fsm.phase==lift (close dwell/timeout done)",
            "t_lift_start": "same as t_close_complete",
            "t_first_airborne": "first obj_z >= Z_AIR (0.48 m) during lift",
            "t_captured": "GraspSim.maybe_capture_reference: lift and not captured",
            "t_fsm_hold_success": "fsm.success: lift z_ok+near for fsm.hold_success=1.0s",
            "t_z_air_plus_0p25": "obj_z>=Z_AIR continuously for 0.25s after first airborne (prior viz hold extra)",
            "t_recovered_instant": "first recovered() True (diagnostic; not used as T_impact)",
            "Z_AIR": Z_AIR,
            "hold_success_s": float(cfg["fsm"]["hold_success"]),
            "lift_clearance": float(cfg["fsm"]["lift_clearance"]),
        },
    }
    air_extra = 0.0
    prev_phase = sim.fsm.phase
    pose_at = {}
    for _ in range(n):
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
        t = float(sim.data.time)
        o = physical_pack(sim)
        ph = str(sim.fsm.phase)
        if events["t_first_finger_object_contact"] is None and (int(o["nL"]) > 0 or int(o["nR"]) > 0):
            events["t_first_finger_object_contact"] = t
        if prev_phase != "close" and ph == "close" and events["t_close_start"] is None:
            events["t_close_start"] = t
        if prev_phase != "lift" and ph == "lift":
            events["t_close_complete"] = t
            events["t_lift_start"] = t
        if events["t_captured"] is None and sim.captured:
            events["t_captured"] = t
        if ph == "lift" and float(o["obj_z"]) >= Z_AIR:
            if events["t_first_airborne"] is None:
                events["t_first_airborne"] = t
                pose_at["first_airborne"] = {
                    "p_obj": np.array(sim.data.xpos[sim.ids.object_body], float).tolist(),
                    "p_hand": np.array(sim.data.xpos[sim.ids.hand_body], float).tolist(),
                    "hand_x": np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3)[:, 0], float).tolist(),
                    "e_x": float(o["e_x"]),
                    "nL": int(o["nL"]),
                    "nR": int(o["nR"]),
                }
            air_extra += dt
            if events["t_z_air_plus_0p25"] is None and air_extra >= 0.25:
                events["t_z_air_plus_0p25"] = t
                pose_at["z_air_plus_0p25"] = {
                    "p_obj": np.array(sim.data.xpos[sim.ids.object_body], float).tolist(),
                    "hand_x": np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3)[:, 0], float).tolist(),
                    "e_x": float(o["e_x"]),
                    "nL": int(o["nL"]),
                    "nR": int(o["nR"]),
                    "v_rel": float(o["v_rel"]),
                }
        else:
            if events["t_first_airborne"] is None:
                air_extra = 0.0
        if events["t_fsm_hold_success"] is None and sim.fsm.success:
            events["t_fsm_hold_success"] = t
            pose_at["fsm_hold_success"] = {
                "p_obj": np.array(sim.data.xpos[sim.ids.object_body], float).tolist(),
                "hand_x": np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3)[:, 0], float).tolist(),
            }
        if events["t_recovered_instant"] is None:
            from envs.physical_recovery import recovered

            if recovered(o):
                events["t_recovered_instant"] = t
        prev_phase = ph
        if events["t_fsm_hold_success"] is not None and events["t_z_air_plus_0p25"] is not None:
            if t > events["t_fsm_hold_success"] + 0.05:
                break
        if sim.dropped():
            events["dropped"] = True
            break
    events["final_t"] = float(sim.data.time)
    events["final_phase"] = sim.fsm.phase
    events["pose_at"] = pose_at
    events["grasp_orientation_hand_x"] = np.array(grasp_orientation(), float).reshape(3, 3)[:, 0].tolist()
    return events


def main() -> int:
    ev = measure()
    out = ROOT / "results" / "logs" / "impact_demo" / "nominal_grasp_timeline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=2), encoding="utf-8")
    print(json.dumps(ev, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
