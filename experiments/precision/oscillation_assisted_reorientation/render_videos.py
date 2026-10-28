#!/usr/bin/env python
"""Render paired OAR videos with HUD overlays."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import gates  # noqa: E402
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.render_precision_failure_videos import hold_frames, overlay, render_pair  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/oscillation_assisted_reorientation"
VID = OUT / "VIDEOS"
DT = 0.02
FPS = 20


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def hud(env, *, ic_id: str, phase: str, t: float, fail: str | None = None, rel: float | None = None) -> list[str]:
    err = measure_errors(env.sim)
    g = gates(env.sim)
    mode = classify_contact_mode(env.sim)
    e = np.asarray(err.get("e_signed_mm") or [0, 0, 0], float)
    return [
        f"IC={ic_id}",
        f"t={t:.2f}s  {phase}" + (f"  FAIL={fail}" if fail else ""),
        f"dθ={float(err.get('dtheta_deg') or 0):.1f}  dx={e[0]:.1f} dy={e[1]:.1f} dz={e[2]:.1f}",
        f"α={float(g.get('alpha_up_deg') or 0):.1f}  bil={bool(err.get('bilateral'))}  "
        f"FnL={float(err.get('fn_L') or 0):.1f} FnR={float(err.get('fn_R') or 0):.1f}"
        + (f"  netRel≈{rel:.2f}°" if rel is not None else ""),
        f"S1={mode.get('support_side')}  FnzL={float(mode.get('Fn_z_L') or 0):.1f} FnzR={float(mode.get('Fn_z_R') or 0):.1f}",
    ]


def replay(env, blob, actions, cam, renderer, ic, tag, fail=None):
    restore_capture(env, blob["capture"])
    frames = []
    t0 = float(env.sim.data.time)
    from evaluation.grasp_geometry import cylinder_axis_hand

    u0 = np.asarray(cylinder_axis_hand(env.sim), float)
    acts = list(actions) if actions is not None and len(actions) else []
    if not acts:
        frames.append(render_pair(env, cam, renderer, hud(env, ic_id=ic, phase=tag + "|EMPTY", t=0.0, fail=fail)))
        return frames
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 2 == 0 or i == len(acts) - 1:
            t = float(env.sim.data.time) - t0
            u1 = np.asarray(cylinder_axis_hand(env.sim), float)
            c = float(np.clip(np.dot(u0 / (np.linalg.norm(u0) + 1e-12), u1 / (np.linalg.norm(u1) + 1e-12)), -1, 1))
            rel = float(np.degrees(np.arccos(c)))
            frames.append(render_pair(env, cam, renderer, hud(env, ic_id=ic, phase=tag, t=t, fail=fail, rel=rel)))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ics", type=int, default=4)
    args = ap.parse_args()
    VID.mkdir(parents=True, exist_ok=True)
    paired = json.loads((OUT / "PAIRED_SINGLE_CYCLE.json").read_text(encoding="utf-8"))
    env = make_env()
    cam = mjv_camera_from_preset({"lookat": [0, 0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20})
    renderer = mujoco.Renderer(env.sim.model, 420, 560)
    index = []

    # Prefer ICs with largest |net| or failures of interest
    scored = []
    for r in paired:
        if r.get("error"):
            continue
        best = 0.0
        for s in (r.get("results") or {}).values():
            best = max(best, abs(float(s.get("net_relative_rotation_deg") or 0)), abs(float(s.get("peak_relative_rotation_deg") or 0)))
        scored.append((-best, r))
    scored.sort(key=lambda t: t[0])
    picks = [t[1] for t in scored[: args.max_ics]]

    for r in picks:
        sid = r["sample_id"]
        blob = pickle.loads((ROOT / r["ic_path"]).read_bytes())
        # Render A/B/C/D for dir+1
        for cond in ("A", "B", "C", "D"):
            key = f"{cond}_dir1"
            act_path = OUT / "traces" / f"{sid}__{key}__actions.npy"
            acts = np.load(act_path) if act_path.exists() else []
            fail = ((r.get("results") or {}).get(key) or {}).get("failure")
            restore_capture(env, blob["capture"])
            frames = hold_frames(env, blob, cam, renderer, sid, hold_s=0.4)
            restore_capture(env, blob["capture"])
            frames += replay(env, blob, acts, cam, renderer, sid, f"OAR_{key}", fail=fail)
            out = VID / f"{sid}__{key}.mp4"
            path = write_frames_mp4(frames, out, fps=FPS)
            index.append(
                {
                    "sample_id": sid,
                    "trial": key,
                    "ok": bool(((r.get("results") or {}).get(key) or {}).get("ok")),
                    "failure": fail,
                    "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                }
            )
            print(f"video {sid} {key}", flush=True)

    (VID / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
