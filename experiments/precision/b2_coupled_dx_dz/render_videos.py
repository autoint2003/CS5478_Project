#!/usr/bin/env python
"""Paired V0/V1/V2 videos + error-vs-time plot overlays for seven B2 ICs."""

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

OUT = ROOT / "results/diagnostics/b2_coupled_dx_dz"
VID = OUT / "VIDEOS"
CACHE = OUT / "stage1_exits"
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


def plot_panel(series: list[dict], w: int = 560, h: int = 160) -> np.ndarray:
    img = np.full((h, w, 3), 18, np.uint8)
    if len(series) < 2:
        return img
    ts = np.array([s["t"] for s in series], float)
    dx = np.array([s["dx_mm"] for s in series], float)
    dz = np.array([s["dz_mm"] for s in series], float)
    th = np.array([s["dtheta_deg"] for s in series], float)
    t0, t1 = float(ts[0]), float(ts[-1]) if ts[-1] > ts[0] else ts[0] + 1.0

    def draw(vals, color):
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if abs(hi - lo) < 1e-6:
            hi = lo + 1.0
        xs = ((ts - t0) / (t1 - t0) * (w - 20) + 10).astype(int)
        ys = (h - 10 - (vals - lo) / (hi - lo) * (h - 20)).astype(int)
        for i in range(1, len(xs)):
            x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[i], ys[i]
            n = max(abs(x1 - x0), abs(y1 - y0), 1)
            for k in range(n + 1):
                x = int(x0 + (x1 - x0) * k / n)
                y = int(y0 + (y1 - y0) * k / n)
                if 0 <= x < w and 0 <= y < h:
                    img[y, x] = color
                    if y + 1 < h:
                        img[y + 1, x] = color

    draw(dx, (80, 180, 255))
    draw(dz, (80, 255, 120))
    draw(th, (255, 160, 80))
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        dr.text((4, 2), "DX(blue) DZ(green) dθ(orange)", fill=(220, 220, 220))
        return np.asarray(im)
    except Exception:
        return img


def stack(sim_img: np.ndarray, plot: np.ndarray) -> np.ndarray:
    w = sim_img.shape[1]
    if plot.shape[1] != w:
        # naive pad/crop
        p = np.full((plot.shape[0], w, 3), 18, np.uint8)
        ww = min(w, plot.shape[1])
        p[:, :ww] = plot[:, :ww]
        plot = p
    return np.concatenate([sim_img, plot], axis=0)


def render_one(env, hid: str, variant: str, cam, renderer) -> dict:
    exit_pkl = CACHE / f"{hid}.pkl"
    act_path = OUT / "raw" / f"{hid}__{variant}__actions.npz"
    meta_path = OUT / "raw" / f"{hid}__{variant}__trace.json"
    case_meta = json.loads((CACHE / f"{hid}__meta.json").read_text(encoding="utf-8"))
    paired = json.loads((OUT / "PAIRED_CASES.json").read_text(encoding="utf-8"))
    row = next(c for c in paired if c["hard_id"] == hid)
    vinfo = (row.get("variants") or {}).get(variant) or {}
    blob = pickle.loads(exit_pkl.read_bytes())
    actions = np.load(act_path)["actions"] if act_path.is_file() else np.zeros((0, 7), np.float32)
    series = []
    if meta_path.is_file():
        series = (json.loads(meta_path.read_text(encoding="utf-8")).get("series") or [])

    restore_capture(env, blob["capture"])
    frames = []
    status = vinfo.get("status")
    sn = vinfo.get("SUCCESS_NEW")
    s12 = vinfo.get("SUCCESS_12S")
    fail = None
    if not sn:
        fails = vinfo.get("failures") or []
        fail = fails[0] if fails else status

    # Hold
    for _ in range(int(0.6 * FPS)):
        lines = hud(env, ic_id=f"{hid}|{variant}", phase="ENTRY", t=0.0, fail=None)
        lines.append(f"expect={case_meta.get('expected_oracle_status')}  NEW={sn} 12S={s12}")
        frames.append(stack(render_pair(env, cam, renderer, lines), plot_panel(series[:1] or [{"t": 0, "dx_mm": 0, "dz_mm": 0, "dtheta_deg": 0}])))

    # Replay with stride for video length
    stride = max(1, int(len(actions) / 400)) if len(actions) > 400 else 1
    trace_i = 0
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
        if i % stride != 0 and i != len(actions) - 1:
            continue
        t = (i + 1) * DT
        while trace_i + 1 < len(series) and series[trace_i + 1]["t"] <= t:
            trace_i += 1
        sub = series[: trace_i + 1] if series else [{"t": t, **{k: float(measure_errors(env.sim).get(k) or 0) for k in ()}}]
        if not series:
            err = measure_errors(env.sim)
            e = err["e_signed_mm"]
            sub = [{"t": t, "dx_mm": e[0], "dz_mm": e[2], "dtheta_deg": err["dtheta_deg"]}]
        phase = f"{variant}|{status}"
        lines = hud(env, ic_id=f"{hid}|{variant}", phase=phase, t=t, fail=str(fail) if fail else None)
        lines.append(f"NEW={sn} 12S={s12}  bil={bool(measure_errors(env.sim).get('bilateral'))}")
        frames.append(stack(render_pair(env, cam, renderer, lines), plot_panel(sub if len(sub) > 1 else series or sub)))

    # Summary hold
    for _ in range(FPS * 2):
        lines = hud(env, ic_id=f"{hid}|{variant}", phase="SUMMARY", t=0.0, fail=str(fail) if fail else None)
        lines.insert(1, f"status={status} NEW={sn} 12S={s12} n_act={len(actions)}")
        frames.append(stack(render_pair(env, cam, renderer, lines), plot_panel(series or [{"t": 0, "dx_mm": 0, "dz_mm": 0, "dtheta_deg": 0}])))

    dest = VID / f"{hid}__{variant}.mp4"
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {
        "hard_id": hid,
        "variant": variant,
        "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
        "n_frames": len(frames),
        "status": status,
        "SUCCESS_NEW": sn,
        "SUCCESS_12S": s12,
    }


def main() -> int:
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cam = mjv_camera_from_preset(CAM)
    renderer = mujoco.Renderer(env.sim.model, 420, 560)
    metas = sorted(CACHE.glob("hard_*.pkl"))
    index = []
    for p in metas:
        hid = p.stem
        for variant in ("V0", "V1", "V2"):
            if not (OUT / "raw" / f"{hid}__{variant}__actions.npz").is_file():
                index.append({"hard_id": hid, "variant": variant, "error": "missing actions"})
                continue
            print(f"render {hid} {variant}", flush=True)
            index.append(render_one(env, hid, variant, cam, renderer))
    (VID / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    lines = ["# VIDEO_INDEX — B2 Coupled DX+DZ", ""]
    for row in index:
        lines.append(
            f"- `{row.get('hard_id')}` **{row.get('variant')}**: "
            f"status={row.get('status')} NEW={row.get('SUCCESS_NEW')} → `{row.get('video') or row.get('error')}`"
        )
    (OUT / "VIDEO_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(index)} videos", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
