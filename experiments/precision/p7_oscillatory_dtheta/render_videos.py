#!/usr/bin/env python
"""P7 videos with θ / DX / DZ / contact / oscillation-phase overlay."""

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
from experiments.precision.p7_oscillatory_dtheta.geometry import signed_theta_to_target  # noqa: E402
from experiments.precision.render_precision_failure_videos import hud, render_pair  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/p7_oscillatory_dtheta"
PRIOR_CAP = ROOT / "results/diagnostics/c_reconfiguration_complete_lift/raw/hard_910277__mid_body_early__capture.pkl"
VID = OUT / "VIDEOS"
FPS = 20
CAM = {"lookat": [0, 0, 0.55], "distance": 0.60, "azimuth": 135, "elevation": -20}


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
    frozen = pickle.loads(PRIOR_CAP.read_bytes())
    cmp = {r["variant"]: r for r in csv.DictReader((OUT / "OSCILLATION_COMPARISON.csv").open(encoding="utf-8"))}
    index = []

    for label in ("O0", "O1", "O2", "O3"):
        act_path = OUT / "raw" / f"{label}__actions.npz"
        if not act_path.is_file():
            continue
        actions = np.load(act_path)["actions"]
        meta_path = OUT / "raw" / f"{label}__meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        series = meta.get("series")  # may be absent for compact meta
        # Load full series from all_packs if needed — overlay uses live measure
        restore_capture(env, frozen)
        frames = []
        n = max(len(actions), 1)
        stride = max(1, n // 400)
        r = cmp.get(label, {})
        for i, a in enumerate(actions):
            step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
            if i % stride and i != len(actions) - 1:
                continue
            t = (i + 1) * 0.02
            th = signed_theta_to_target(env.sim)
            # Phase heuristic from cycle structure / O0
            if label == "O0":
                phase = "correct_dtheta"
            else:
                # approximate: each cycle ~ (0.45*2+0.35)/0.02 ≈ 62 steps
                cyc = i // 62
                within = i % 62
                if within < 22:
                    phase = f"c{cyc}:forward"
                elif within < 44:
                    phase = f"c{cyc}:reverse"
                else:
                    phase = f"c{cyc}:settle"
            lines = hud(env, ic_id=f"910277|{label}", phase=phase, t=t, fail=None)
            lines.append(
                f"θ={th['abs_theta_deg']:.2f} signed={th['signed_theta_deg']:.2f} "
                f"dx={th['dx_mm']:.2f} dz={th['dz_mm']:.2f}"
            )
            lines.append(
                f"NEW={r.get('SUCCESS_NEW')} 12S={r.get('SUCCESS_12S')} "
                f"finalθ={r.get('theta_final')} abort={r.get('abort_reason')}"
            )
            frames.append(render_pair(env, cam, renderer, lines))
        dest = VID / f"910277__{label}.mp4"
        path = write_frames_mp4(frames, dest, fps=FPS)
        index.append(
            {
                "variant": label,
                "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                "theta_final": r.get("theta_final"),
                "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                "SUCCESS_12S": r.get("SUCCESS_12S"),
                "abort_reason": r.get("abort_reason"),
            }
        )
        print(f"wrote {dest.name}", flush=True)

    (VID / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    md = ["# VIDEO_INDEX — P7 Oscillatory dθ", ""]
    for row in index:
        md.append(
            f"- `{row['variant']}`: θ_final={row.get('theta_final')} NEW={row.get('SUCCESS_NEW')} "
            f"12S={row.get('SUCCESS_12S')} abort={row.get('abort_reason')} → `{row.get('video')}`"
        )
    (OUT / "VIDEO_INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
