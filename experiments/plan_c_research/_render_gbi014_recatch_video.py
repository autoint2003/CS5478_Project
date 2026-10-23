"""Render gbi_014 FULL_DETACH_ACTIVE_RECATCH from saved catch expert."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import data_generation.oracles.plan_c.dynamic_catch as dc
import data_generation.oracles.plan_c.interception as er

sys.modules["data_generation.strategies.ground_dynamic_catch"] = dc
sys.modules["data_generation.strategies.ground_early_recovery"] = er

from envs.outcomes.ground_drop import arm, restore_state as restore_drop
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT
from training.recovery_execution import restore_ballistic
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

PKL = ROOT / "results/diagnostics/ground_dynamic_catch_oracle/raw/gbi_014__catch_expert.pkl"
OUT = ROOT / "results/diagnostics/ground_dynamic_catch_oracle/videos/gbi_014__recatch.mp4"


def _hand(sim):
    return np.asarray(sim.data.xpos[sim.ids.hand_body], float)


def _obj(sim):
    return np.asarray(sim.data.xpos[sim.ids.object_body], float)


def main():
    pack = pickle.loads(PKL.read_bytes())
    acts = [np.asarray(a, np.float32) for a in pack["actions"]]
    sim, _ = make_ground_sim()
    restore_ballistic(sim, pack["snap"])
    restore_drop(sim, None)
    arm(sim)

    look = 0.5 * (_obj(sim) + _hand(sim))
    cam = mjv_camera_from_preset(
        {"lookat": look, "distance": 0.55, "azimuth": 135.0, "elevation": -20.0}
    )
    renderer = mujoco.Renderer(sim.model, 480, 640)
    frames = []
    t0 = float(sim.data.time)

    def grab(tag: str):
        cam.lookat[:] = 0.5 * (_obj(sim) + _hand(sim))
        renderer.update_scene(sim.data, cam)
        m = measure(sim, U_TGT, P_TGT)
        img = np.asarray(renderer.render(), np.uint8).copy()
        text = (
            f"gbi_014 RECATCH {tag} t={float(sim.data.time) - t0:.2f}s "
            f"e_pos={m['e_pos_mm']:.1f}mm e_th={m['e_theta_deg']:.1f}deg"
        )
        frames.append(caption(img, text))

    grab("START")
    n_cap = int(len(pack["capture_actions"]))
    for i, a in enumerate(acts):
        step_legal(sim, a)
        if i % 2 == 0 or i == len(acts) - 1 or i == n_cap - 1:
            tag = "CATCH" if i < n_cap else "HOLD"
            grab(tag)

    grab("END")
    renderer.close()
    path = write_frames_mp4(frames, OUT, fps=25)
    print("wrote", path, "frames", len(frames), "actions", len(acts), "capture", n_cap)


if __name__ == "__main__":
    main()
