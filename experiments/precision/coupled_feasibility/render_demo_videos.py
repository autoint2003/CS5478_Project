#!/usr/bin/env python
"""Representative paired MP4s for Gravity Pivot / Coupled DXDZ demos."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import correct_dtheta  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.coupled_feasibility.coupled_dx_dz import coupled_dx_dz  # noqa: E402
from experiments.precision.coupled_feasibility.gravity_pivot import gravity_aligned_pivot  # noqa: E402
from experiments.precision.coupled_feasibility.run_paired import make_env  # noqa: E402
from experiments.precision.render_precision_failure_videos import hold_frames, hud, overlay, render_pair  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/precision_coupled_feasibility/VIDEOS"
AUDIT = ROOT / "results/diagnostics/precision_candidate_audit/CLASSIFICATION.jsonl"
FPS = 16


def replay(env, blob, actions, cam, renderer, ic, tag):
    frames = []
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    acts = list(actions or [])
    if not acts:
        frames.append(render_pair(env, cam, renderer, hud(env, ic_id=ic, phase=tag + "|EMPTY", t=0.0)))
        return frames
    stride = max(1, len(acts) // 120)
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % stride == 0 or i == len(acts) - 1:
            frames.append(
                render_pair(
                    env,
                    cam,
                    renderer,
                    hud(env, ic_id=ic, phase=f"{tag}:{i+1}/{len(acts)}", t=float(env.sim.data.time) - t0),
                )
            )
    return frames


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {json.loads(l)["sample_id"]: json.loads(l) for l in AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()}
    # Pick: one orient fail, one known success if present, one ST fail
    picks = [
        "PILOT_A__ball_train_v2_0012__93a263b2b1a49030",
        "PILOT_A__ball_train_v2_0070__962cc37bb1af2e85",
        "B__hard_910220__0412621ee978627d",
        "PILOT_A__ball_train_0017__f423e91b56091488",
    ]
    env = make_env()
    cam = mjv_camera_from_preset({"lookat": [0, 0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20})
    renderer = mujoco.Renderer(env.sim.model, 420, 560)
    pcfg = PrecisionConfig(use_verified_lateral_fallback=False)
    index = []
    for sid in picks:
        row = rows.get(sid)
        if not row:
            continue
        blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
        # Orientation pair
        restore_capture(env, blob["capture"])
        base = correct_dtheta(env, {"capture": blob["capture"], "ref": blob["ref"]}, pcfg, str(row["plan_family"]))
        restore_capture(env, blob["capture"])
        cand = gravity_aligned_pivot(env, blob)
        for label, res in (("baseline_pivot", base), ("gravity_pivot", cand)):
            restore_capture(env, blob["capture"])
            frames = hold_frames(env, blob, cam, renderer, sid, hold_s=0.6)
            frames += replay(env, blob, res.get("actions") or [], cam, renderer, sid, label)
            dest = OUT / f"{sid}__{label}.mp4"
            path = write_frames_mp4(frames, dest, fps=FPS)
            index.append({"sample_id": sid, "kind": label, "ok": bool(res.get("ok")), "video": str(Path(path).relative_to(ROOT)).replace('\\','/') if path else None})
        # Coupled on ST-ish cases
        restore_capture(env, blob["capture"])
        coup = coupled_dx_dz(env, blob, cfg=pcfg)
        restore_capture(env, blob["capture"])
        frames = hold_frames(env, blob, cam, renderer, sid, hold_s=0.6)
        frames += replay(env, blob, coup.get("actions") or [], cam, renderer, sid, "coupled_dx_dz")
        dest = OUT / f"{sid}__coupled_dx_dz.mp4"
        path = write_frames_mp4(frames, dest, fps=FPS)
        index.append({"sample_id": sid, "kind": "coupled_dx_dz", "ok": bool(coup.get("ok")), "reason": coup.get("reason"), "video": str(Path(path).relative_to(ROOT)).replace('\\','/') if path else None})
        print(sid, "done", flush=True)
    renderer.close()
    (OUT / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("wrote", OUT / "INDEX.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
