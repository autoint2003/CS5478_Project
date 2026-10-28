#!/usr/bin/env python
"""Render D0/D1/D2 videos for hard_910305 and hard_910084."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.render_precision_failure_videos import hud, render_pair  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/b2_dz_braking_and_coupled_execution"
EXIT_CACHE = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
VID = OUT / "VIDEOS"
FPS = 20
DT = 0.02
FOCUS = ["hard_910305", "hard_910084"]
CAM = {"lookat": [0, 0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def main() -> int:
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cam = mjv_camera_from_preset(CAM)
    renderer = mujoco.Renderer(env.sim.model, 420, 560)
    focus = json.loads((OUT / "FOCUS_RESULTS.json").read_text(encoding="utf-8"))
    by = {r["hard_id"]: r for r in focus}
    index = []
    for hid in FOCUS:
        blob = pickle.loads((EXIT_CACHE / f"{hid}.pkl").read_bytes())
        info = by.get(hid) or {}
        for var in ("D0", "D1", "D2"):
            act_path = OUT / "raw" / f"{hid}__{var}__actions.npz"
            if not act_path.is_file():
                index.append({"hard_id": hid, "variant": var, "error": "missing actions"})
                continue
            actions = np.load(act_path)["actions"]
            vinfo = info.get(var) or {}
            sn = vinfo.get("SUCCESS_NEW")
            s12 = vinfo.get("SUCCESS_12S")
            restore_capture(env, blob["capture"])
            frames = []
            stride = max(1, len(actions) // 400) if len(actions) > 400 else 1
            for i, a in enumerate(actions):
                step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
                if i % stride != 0 and i != len(actions) - 1:
                    continue
                t = (i + 1) * DT
                lines = hud(env, ic_id=f"{hid}|{var}", phase=var, t=t, fail=None if sn else str(vinfo.get("status") or var))
                lines.append(f"NEW={sn} 12S={s12}")
                frames.append(render_pair(env, cam, renderer, lines))
            for _ in range(FPS):
                err = measure_errors(env.sim)
                lines = hud(env, ic_id=f"{hid}|{var}", phase="SUMMARY", t=0.0, fail=None)
                lines.insert(1, f"NEW={sn} 12S={s12} dz={float(err['dz_mm']):.1f} e_pos={float(err['e_pos_mm']):.1f}")
                frames.append(render_pair(env, cam, renderer, lines))
            dest = VID / f"{hid}__{var}.mp4"
            path = write_frames_mp4(frames, dest, fps=FPS)
            index.append(
                {
                    "hard_id": hid,
                    "variant": var,
                    "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                    "SUCCESS_NEW": sn,
                    "SUCCESS_12S": s12,
                }
            )
            print(f"wrote {dest.name}", flush=True)
    (VID / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
