#!/usr/bin/env python
"""Paired V0/V1/V2 videos with DZ-vs-time strip for seven B2 ICs."""

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
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.render_precision_failure_videos import hud, render_pair  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/dz_zero_target_return_compensation"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
VID = OUT / "VIDEOS"
FPS = 20
DT = 0.02
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


def plot_dz(series, w=560, h=140):
    img = np.full((h, w, 3), 18, np.uint8)
    if len(series) < 2:
        return img
    ts = np.array([float(s.get("t") or 0) for s in series], float)
    dz = np.array([float(s.get("dz_mm") or 0) for s in series], float)
    t0, t1 = float(ts[0]), float(ts[-1] if ts[-1] > ts[0] else ts[0] + 1)
    lo, hi = float(np.min(dz)), float(np.max(dz))
    if abs(hi - lo) < 1e-6:
        hi = lo + 1
    xs = ((ts - t0) / (t1 - t0) * (w - 20) + 10).astype(int)
    ys = (h - 10 - (dz - lo) / (hi - lo) * (h - 20)).astype(int)
    for i in range(1, len(xs)):
        n = max(abs(xs[i] - xs[i - 1]), abs(ys[i] - ys[i - 1]), 1)
        for k in range(n + 1):
            x = int(xs[i - 1] + (xs[i] - xs[i - 1]) * k / n)
            y = int(ys[i - 1] + (ys[i] - ys[i - 1]) * k / n)
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = (80, 255, 120)
    # zero line
    if lo < 0 < hi:
        y0 = int(h - 10 - (0 - lo) / (hi - lo) * (h - 20))
        if 0 <= y0 < h:
            img[y0, :] = (60, 60, 60)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((4, 2), "DZ vs t (green); gray=0", fill=(220, 220, 220))
        return np.asarray(im)
    except Exception:
        return img


def main() -> int:
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cam = mjv_camera_from_preset(CAM)
    renderer = mujoco.Renderer(env.sim.model, 420, 560)
    paired = list(csv.DictReader((OUT / "PAIRED_RESULTS.csv").open(encoding="utf-8")))
    timelines = list(csv.DictReader((OUT / "DZ_TIMELINES.csv").open(encoding="utf-8"))) if (OUT / "DZ_TIMELINES.csv").is_file() else []
    index = []
    hard_ids = sorted({r["hard_id"] for r in paired})
    for hid in hard_ids:
        blob = pickle.loads((EXIT / f"{hid}.pkl").read_bytes())
        for var in ("V0", "V1", "V2"):
            act = OUT / "raw" / f"{hid}__{var}__actions.npz"
            if not act.is_file():
                continue
            actions = np.load(act)["actions"]
            info = next((r for r in paired if r["hard_id"] == hid and r["variant"] == var), {})
            series = [t for t in timelines if t.get("hard_id") == hid and t.get("variant") == var]
            restore_capture(env, blob["capture"])
            frames = []
            stride = max(1, len(actions) // 400) if len(actions) > 400 else 1
            for i, a in enumerate(actions):
                step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
                if i % stride and i != len(actions) - 1:
                    continue
                t = (i + 1) * DT
                lines = hud(env, ic_id=f"{hid}|{var}", phase=var, t=t, fail=None)
                lines.append(
                    f"NEW={info.get('SUCCESS_NEW')} tgt={info.get('pre_return_target_mm') or 0} "
                    f"brake_dz={info.get('dz_at_brake_start')}"
                )
                sim = render_pair(env, cam, renderer, lines)
                sub = series[: max(1, int(len(series) * (i + 1) / max(len(actions), 1)))] if series else [
                    {"t": t, "dz_mm": float(measure_errors(env.sim)["dz_mm"])}
                ]
                plot = plot_dz(sub if len(sub) > 1 else series or sub)
                if plot.shape[1] != sim.shape[1]:
                    p = np.full((plot.shape[0], sim.shape[1], 3), 18, np.uint8)
                    ww = min(plot.shape[1], sim.shape[1])
                    p[:, :ww] = plot[:, :ww]
                    plot = p
                frames.append(np.concatenate([sim, plot], axis=0))
            dest = VID / f"{hid}__{var}.mp4"
            path = write_frames_mp4(frames, dest, fps=FPS)
            index.append(
                {
                    "hard_id": hid,
                    "variant": var,
                    "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                    "SUCCESS_NEW": info.get("SUCCESS_NEW"),
                }
            )
            print(f"wrote {dest.name}", flush=True)
    (VID / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    lines = ["# VIDEO_INDEX — Zero-Target / Return-Compensated DZ", ""]
    for row in index:
        lines.append(
            f"- `{row['hard_id']}` {row['variant']}: NEW={row.get('SUCCESS_NEW')} → `{row.get('video')}`"
        )
    (OUT / "VIDEO_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
