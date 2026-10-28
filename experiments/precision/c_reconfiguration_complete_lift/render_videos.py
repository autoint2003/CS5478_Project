#!/usr/bin/env python
"""Videos + height/stability time-series for complete C on primary ICs."""

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

OUT = ROOT / "results/diagnostics/c_reconfiguration_complete_lift"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
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


def write_height_plot(hid: str, variant: str, traces: list) -> Path | None:
    """Simple SVG time-series of obj_z and bilateral (no heavy deps)."""
    if not traces:
        return None
    ts = [float(t.get("t") or 0) for t in traces]
    zs = [float(t.get("obj_z") or 0) for t in traces]
    bil = [1.0 if t.get("bilateral") else 0.0 for t in traces]
    w, h = 640, 240
    pad = 40
    t0, t1 = min(ts), max(ts) if max(ts) > min(ts) else min(ts) + 1e-3
    z0, z1 = min(zs + [0.0]), max(zs + [0.6])
    if z1 <= z0:
        z1 = z0 + 0.1

    def xmap(t):
        return pad + (w - 2 * pad) * (t - t0) / (t1 - t0)

    def ymap(z):
        return h - pad - (h - 2 * pad) * (z - z0) / (z1 - z0)

    poly_z = " ".join(f"{xmap(t):.1f},{ymap(z):.1f}" for t, z in zip(ts, zs))
    poly_b = " ".join(f"{xmap(t):.1f},{ymap(z0 + b * (z1 - z0) * 0.15):.1f}" for t, b in zip(ts, bil))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <rect width="100%" height="100%" fill="#111"/>
  <text x="{pad}" y="20" fill="#ddd" font-size="12">{hid} {variant}: obj_z (cyan) / bilateral (orange band)</text>
  <polyline fill="none" stroke="#3cf" stroke-width="2" points="{poly_z}"/>
  <polyline fill="none" stroke="#fa3" stroke-width="1.5" points="{poly_b}"/>
  <text x="{pad}" y="{h - 8}" fill="#888" font-size="10">t={t0:.2f}→{t1:.2f}s  z={z0:.2f}→{z1:.2f}m</text>
</svg>
"""
    dest = VID / f"{hid}__{variant}__height.svg"
    dest.write_text(svg, encoding="utf-8")
    return dest


def main() -> int:
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cam = mjv_camera_from_preset(CAM)
    renderer = mujoco.Renderer(env.sim.model, 420, 560)
    rows = list(csv.DictReader((OUT / "C_RELEASE_RECATCH_LIFT_RESULTS.csv").open(encoding="utf-8")))
    index = []
    # Prefer primary ICs; one best variant video per IC + all variants listed
    by_ic: dict[str, list] = {}
    for r in rows:
        by_ic.setdefault(r["hard_id"], []).append(r)

    for hid, group in by_ic.items():
        blob = pickle.loads((EXIT / f"{hid}.pkl").read_bytes())
        # pick best: SUCCESS_12S first, else LIFT_COMPLETED, else first
        def score(r):
            return (
                1 if r.get("SUCCESS_12S") in ("True", "true", True) else 0,
                1 if r.get("LIFT_COMPLETED") in ("True", "true", True) else 0,
                1 if r.get("REACQUIRE_EXECUTED") in ("True", "true", True) else 0,
            )

        ordered = sorted(group, key=score, reverse=True)
        for r in ordered:
            variant = r["variant"]
            act = OUT / "raw" / f"{hid}__{variant}__actions.npz"
            tl_path = OUT / "raw" / f"{hid}__{variant}__timeline.json"
            if not act.is_file():
                continue
            actions = np.load(act)["actions"]
            traces = json.loads(tl_path.read_text(encoding="utf-8")) if tl_path.is_file() else []
            plot = write_height_plot(hid, variant, traces)
            restore_capture(env, blob["capture"])
            frames = []
            n = max(len(actions), 1)
            stride = max(1, n // 450)
            # Phase from nearest trace
            for i, a in enumerate(actions):
                step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
                if i % stride and i != len(actions) - 1:
                    continue
                t = (i + 1) * 0.02
                phase = "C"
                for tr in traces:
                    if float(tr.get("t") or 0) <= t:
                        phase = str(tr.get("phase") or "C")
                lines = hud(env, ic_id=f"{hid}|{variant}", phase=phase, t=t, fail=None)
                lines.append(
                    f"12S={r.get('SUCCESS_12S')} LIFT={r.get('LIFT_COMPLETED')} IMP={r.get('RECONFIG_IMPROVED')}"
                )
                lines.append(f"fail={r.get('stage_fail') or '—'}")
                frames.append(render_pair(env, cam, renderer, lines))
            if not frames:
                lines = hud(env, ic_id=hid, phase="init", t=0, fail=None)
                frames.append(render_pair(env, cam, renderer, lines))
            dest = VID / f"{hid}__{variant}.mp4"
            path = write_frames_mp4(frames, dest, fps=FPS)
            index.append(
                {
                    "hard_id": hid,
                    "variant": variant,
                    "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                    "height_plot": str(plot.relative_to(ROOT)).replace("\\", "/") if plot else None,
                    "SUCCESS_12S": r.get("SUCCESS_12S"),
                    "LIFT_COMPLETED": r.get("LIFT_COMPLETED"),
                    "RECONFIG_IMPROVED": r.get("RECONFIG_IMPROVED"),
                    "stage_fail": r.get("stage_fail"),
                }
            )
            print(f"wrote {dest.name}", flush=True)
            # Primary: render all variants; secondary: best only
            if hid not in ("hard_910263", "hard_910277") and r is ordered[0]:
                break

    (VID / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    md = ["# VIDEO_INDEX — Complete C Recatch + Lift", ""]
    for row in index:
        md.append(
            f"- `{row.get('hard_id')}` `{row.get('variant')}`: 12S={row.get('SUCCESS_12S')} "
            f"LIFT={row.get('LIFT_COMPLETED')} IMP={row.get('RECONFIG_IMPROVED')} "
            f"fail={row.get('stage_fail') or '—'} → `{row.get('video')}` "
            f"(height: `{row.get('height_plot')}`)"
        )
    (OUT / "VIDEO_INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
