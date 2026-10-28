#!/usr/bin/env python
"""Render any PRECISION_REQUIRED_BUT_FAILED videos not yet produced by bootstrap."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.precision.render_precision_failure_videos import (  # noqa: E402
    OUT,
    cfg,
    hold_frames,
    hud,
    make_env,
    render_pair,
    render_paired,
    replay_actions_video,
)
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

AUDIT = ROOT / "results/diagnostics/precision_candidate_audit"
FPS = 20


def main() -> int:
    rows = [
        json.loads(l)
        for l in (AUDIT / "CLASSIFICATION.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    failed = [
        r
        for r in rows
        if r.get("class") == "PRECISION_REQUIRED_BUT_FAILED"
        and r.get("subclass") != "success_without_correction_skill"
    ]
    done = {p.stem for p in (OUT / "failures").glob("*.mp4")}
    missing = [r for r in failed if r["sample_id"] not in done]
    print(f"missing failure videos: {len(missing)}", flush=True)

    env = make_env()
    cam = mjv_camera_from_preset({"lookat": [0, 0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20})
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    index = {"failures": [], "paired_extra": []}

    for i, row in enumerate(missing):
        print(f"[{i+1}/{len(missing)}] {row['sample_id']}", flush=True)
        try:
            blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
            restore_capture(env, blob["capture"])
            res = run_precision(env, {"capture": blob["capture"], "ref": blob["ref"]}, str(row["plan_family"]), cfg())
            actions = np.asarray(res.actions or [], np.float32)
            fail = (res.failures or ((row.get("precision") or {}).get("failures")) or [res.status])[0]
            dest = OUT / "failures" / f"{row['sample_id']}.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            restore_capture(env, blob["capture"])
            frames = hold_frames(env, blob, cam, renderer, row["sample_id"])
            if len(actions):
                frames.extend(
                    replay_actions_video(
                        env,
                        blob,
                        actions,
                        cam,
                        renderer,
                        row["sample_id"],
                        phase_prefix="PRECISION_ORACLE",
                        fail_reason=str(fail),
                    )
                )
            else:
                frames.append(
                    render_pair(
                        env,
                        cam,
                        renderer,
                        hud(env, ic_id=row["sample_id"], phase=str(res.status), t=0.0, fail=str(fail)),
                    )
                )
            path = write_frames_mp4(frames, dest, fps=FPS)
            index["failures"].append(
                {
                    "sample_id": row["sample_id"],
                    "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                    "fail": fail,
                    "status": res.status,
                }
            )
        except Exception as exc:
            index["failures"].append({"sample_id": row["sample_id"], "error": str(exc)})
            print(" ERROR", exc, flush=True)

    # Extra paired samples: HARD_IC / C failures not in bootstrap pairs
    extra_paired = [r for r in failed if r["sample_id"] not in done][:6]
    for i, row in enumerate(extra_paired):
        print(f"[pair-extra {i+1}] {row['sample_id']}", flush=True)
        try:
            index["paired_extra"].append(render_paired(env, row, cam, renderer))
        except Exception as exc:
            index["paired_extra"].append({"sample_id": row["sample_id"], "error": str(exc)})

    renderer.close()
    (OUT / "REMAINING_INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    n = len(list((OUT / "failures").glob("*.mp4")))
    print(f"total failure mp4 now: {n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
