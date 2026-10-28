#!/usr/bin/env python
"""Videos for A/B/C reconfiguration pilot."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.render_precision_failure_videos import hud, render_pair  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/abc_grasp_reconfiguration"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
VID = OUT / "VIDEOS"
FPS = 20
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
    rows = list(csv.DictReader((OUT / "RECONFIGURATION_RESULTS.csv").open(encoding="utf-8")))
    index = []
    for hid in sorted({r["hard_id"] for r in rows}):
        blob = pickle.loads((EXIT / f"{hid}.pkl").read_bytes())
        for s in ("R0", "R1", "R2", "R3"):
            act_path = OUT / "raw" / f"{hid}__{s}__actions.npz"
            if not act_path.is_file():
                index.append({"hard_id": hid, "strategy": s, "status": "NOT_RUN"})
                continue
            info = next((r for r in rows if r["hard_id"] == hid and r["strategy"] == s), {})
            z = np.load(act_path)
            rec = z["reconfig"] if "reconfig" in z.files else np.zeros((0, 7), np.float32)
            prec = z["precision"] if "precision" in z.files else np.zeros((0, 7), np.float32)
            actions = np.vstack([rec, prec]) if len(rec) or len(prec) else np.zeros((0, 7), np.float32)
            restore_capture(env, blob["capture"])
            frames = []
            n = max(len(actions), 1)
            stride = max(1, n // 400)
            phase = "reconfig"
            for i, a in enumerate(actions):
                if i == len(rec) and len(prec):
                    phase = "precision"
                step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
                if i % stride and i != len(actions) - 1:
                    continue
                skill = s if s == "R0" else f"{s}/{'ABC'[int(s[1])-1]}"
                lines = hud(env, ic_id=f"{hid}|{skill}", phase=phase, t=(i + 1) * 0.02, fail=None)
                lines.append(
                    f"STABLE={info.get('RECONFIG_STABLE')} IMP={info.get('RECONFIG_IMPROVED')} NEW={info.get('FINAL_SUCCESS')}"
                )
                if info.get("released"):
                    lines.append(f"release={info.get('t_release')} reacq={info.get('t_reacquire')}")
                frames.append(render_pair(env, cam, renderer, lines))
            if not frames:
                # hold frame of initial grasp
                lines = hud(env, ic_id=f"{hid}|{s}", phase="init", t=0.0, fail=None)
                frames.append(render_pair(env, cam, renderer, lines))
            dest = VID / f"{hid}__{s}.mp4"
            path = write_frames_mp4(frames, dest, fps=FPS)
            index.append(
                {
                    "hard_id": hid,
                    "strategy": s,
                    "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                    "RECONFIG_STABLE": info.get("RECONFIG_STABLE"),
                    "RECONFIG_IMPROVED": info.get("RECONFIG_IMPROVED"),
                    "FINAL_SUCCESS": info.get("FINAL_SUCCESS"),
                    "released": info.get("released"),
                }
            )
            print(f"wrote {dest.name}", flush=True)
    (VID / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    md = ["# VIDEO_INDEX — A/B/C Grasp Reconfiguration", ""]
    for row in index:
        md.append(
            f"- `{row.get('hard_id')}` {row.get('strategy')}: STABLE={row.get('RECONFIG_STABLE')} "
            f"IMP={row.get('RECONFIG_IMPROVED')} NEW={row.get('FINAL_SUCCESS')} "
            f"released={row.get('released')} → `{row.get('video') or row.get('status')}`"
        )
    (OUT / "VIDEO_INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
