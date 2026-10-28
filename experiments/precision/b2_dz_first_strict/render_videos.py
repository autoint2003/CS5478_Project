#!/usr/bin/env python
"""Videos for strict DZ-first ordering."""

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

OUT = ROOT / "results/diagnostics/b2_dz_first_strict"
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
    rows = list(csv.DictReader((OUT / "STRICT_ORDERING_RESULTS.csv").open(encoding="utf-8")))
    index = []
    for hid in sorted({r["hard_id"] for r in rows}):
        blob = pickle.loads((EXIT / f"{hid}.pkl").read_bytes())
        for s in ("S0", "S1", "S2", "S3"):
            act = OUT / "raw" / f"{hid}__{s}__actions.npz"
            if not act.is_file():
                index.append({"hard_id": hid, "strategy": s, "status": "NOT_RUN"})
                continue
            info = next((r for r in rows if r["hard_id"] == hid and r["strategy"] == s), {})
            actions = np.load(act)["actions"]
            restore_capture(env, blob["capture"])
            frames = []
            stride = max(1, len(actions) // 400) if len(actions) > 400 else 1
            for i, a in enumerate(actions):
                step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
                if i % stride and i != len(actions) - 1:
                    continue
                lines = hud(env, ic_id=f"{hid}|{s}", phase=s, t=(i + 1) * 0.02, fail=None)
                lines.append(f"NEW={info.get('SUCCESS_NEW')} DZ1st={info.get('DZ_FIRST_EXECUTED')}")
                frames.append(render_pair(env, cam, renderer, lines))
            dest = VID / f"{hid}__{s}.mp4"
            path = write_frames_mp4(frames, dest, fps=FPS)
            index.append(
                {
                    "hard_id": hid,
                    "strategy": s,
                    "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                    "SUCCESS_NEW": info.get("SUCCESS_NEW"),
                    "DZ_FIRST_EXECUTED": info.get("DZ_FIRST_EXECUTED"),
                    "DZ_FIRST_BLOCK_REASON": info.get("DZ_FIRST_BLOCK_REASON"),
                }
            )
            print(f"wrote {dest.name}", flush=True)
    (VID / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    md = ["# VIDEO_INDEX — Strict DZ-First", ""]
    for row in index:
        md.append(
            f"- `{row.get('hard_id')}` {row.get('strategy')}: NEW={row.get('SUCCESS_NEW')} "
            f"DZ_FIRST={row.get('DZ_FIRST_EXECUTED')} block={row.get('DZ_FIRST_BLOCK_REASON') or '—'} "
            f"→ `{row.get('video') or row.get('status')}`"
        )
    (OUT / "VIDEO_INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
