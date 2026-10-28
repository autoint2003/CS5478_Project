#!/usr/bin/env python
"""Bootstrap failure videos from experts_v2_official Precision failure JSON.

Used while the full candidate audit is still running. Same hold-clone + Oracle
replay contract as render_precision_failure_videos.py.
"""

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

FAIL_DIR = ROOT / "data/experts_v2_official/diagnostics/failures"
ICS = ROOT / "results/data_generation/stage2_abc_final_v2/ics"
FPS = 20


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fails = []
    for p in sorted(FAIL_DIR.glob("*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row.get("continuation") != "precision":
            continue
        if "PRECISION" not in str(row.get("branch_id")):
            continue
        sid = row["source_ic_id"]
        ic = ICS / f"{sid}.pkl"
        if not ic.is_file():
            hits = sorted(ICS.glob(f"{sid}*.pkl"))
            ic = hits[0] if hits else ic
        if not ic.is_file():
            continue
        plan = "A" if sid.startswith("PILOT_A") or "__A" in sid else (
            "C" if sid.startswith("C__") else ("HARD_IC" if "hard_" in sid or sid.startswith("HARD") or sid.startswith("D") else "B")
        )
        if sid.startswith("B__"):
            plan = "B"
        if sid.startswith("HARD_IC") or "hard_" in sid:
            plan = "B"  # HARD ICs use B-like precision entry in practice
        fails.append(
            {
                "sample_id": sid,
                "plan_family": plan,
                "ic_path": str(ic.relative_to(ROOT)).replace("\\", "/"),
                "precision": {
                    "status": row.get("status"),
                    "failures": row.get("failures") or [],
                    "first_failed_primitive": (row.get("failures") or ["unknown"])[0],
                    "phases": row.get("phases") or [],
                },
                "class": "PRECISION_REQUIRED_BUT_FAILED",
            }
        )

    # Prefer A then B/D nontrivial fails
    fails.sort(key=lambda r: (0 if r["sample_id"].startswith("PILOT_A") else 1, r["sample_id"]))
    print(f"bootstrap failures: {len(fails)}", flush=True)

    env = make_env()
    cam = mjv_camera_from_preset({"lookat": [0, 0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20})
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    index = {"failures": [], "paired": []}

    # All A precision fails + first 12 others
    a_fails = [r for r in fails if r["sample_id"].startswith("PILOT_A") or "A_STABLE" in str(r)]
    other = [r for r in fails if r not in a_fails][:12]
    targets = a_fails + other
    for i, row in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] {row['sample_id']}", flush=True)
        try:
            blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
            restore_capture(env, blob["capture"])
            res = run_precision(env, {"capture": blob["capture"], "ref": blob["ref"]}, row["plan_family"], cfg())
            actions = np.asarray(res.actions or [], np.float32)
            fail = (res.failures or row["precision"].get("failures") or [res.status])[0]
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
                    "status": res.status,
                    "fail": fail,
                    "n_actions": int(len(actions)),
                }
            )
        except Exception as exc:
            index["failures"].append({"sample_id": row["sample_id"], "error": str(exc)})
            print(" ERROR", exc, flush=True)

    for i, row in enumerate(targets[:8]):
        print(f"[pair {i+1}/8] {row['sample_id']}", flush=True)
        try:
            index["paired"].append(render_paired(env, row, cam, renderer))
        except Exception as exc:
            index["paired"].append({"sample_id": row["sample_id"], "error": str(exc)})

    renderer.close()
    (OUT / "BOOTSTRAP_INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("wrote", OUT / "BOOTSTRAP_INDEX.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
