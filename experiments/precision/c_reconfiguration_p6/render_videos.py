#!/usr/bin/env python
"""P6 videos: 910263 baseline vs s_m015; 910277 C→Precision P0/P1."""

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
from experiments.precision.c_reconfiguration_p6.geometry import snapshot_geometry  # noqa: E402
from experiments.precision.render_precision_failure_videos import hud, render_pair  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/c_reconfiguration_p6"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
PRIOR277 = ROOT / "results/diagnostics/c_reconfiguration_complete_lift/raw"
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


def replay(env, capture, actions, cam, renderer, ic_id: str, phase: str, extra: list[str], max_frames: int = 420):
    restore_capture(env, capture)
    frames = []
    n = max(len(actions), 1)
    stride = max(1, n // max_frames)
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
        if i % stride and i != len(actions) - 1:
            continue
        t = (i + 1) * 0.02
        g = snapshot_geometry(env.sim)
        lines = hud(env, ic_id=ic_id, phase=phase, t=t, fail=None)
        lines.append(
            f"s_mean={g.get('contact_s_mean')} kind={g.get('grasp_kind')} "
            f"z={g.get('obj_z'):.3f}" if g.get("obj_z") is not None else f"s_mean={g.get('contact_s_mean')}"
        )
        lines.extend(extra)
        frames.append(render_pair(env, cam, renderer, lines))
    if not frames:
        lines = hud(env, ic_id=ic_id, phase=phase, t=0, fail=None)
        frames.append(render_pair(env, cam, renderer, lines))
    return frames


def main() -> int:
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cam = mjv_camera_from_preset(CAM)
    renderer = mujoco.Renderer(env.sim.model, 420, 560)
    index = []

    # --- 910263: end-grasp baseline vs improved s_m015 ---
    blob263 = pickle.loads((EXIT / "hard_910263.pkl").read_bytes())
    place = {r["label"]: r for r in csv.DictReader((OUT / "910263_RECATCH_PLACEMENT.csv").open(encoding="utf-8"))}
    for label, title in (("baseline_s0", "end-biased mid_body_early"), ("s_m015", "axial −15 mm mid-body")):
        act = np.load(OUT / "raw" / f"910263__{label}__actions.npz")["actions"]
        r = place.get(label, {})
        extra = [
            f"cmd_s={r.get('commanded_s_axial')} real_s={r.get('realized_contact_s_mean')}",
            f"12S={r.get('SUCCESS_12S')} PLACE={r.get('CONTACT_PLACEMENT_VALID')} IMP={r.get('RECONFIG_IMPROVED')}",
            f"dx={r.get('final_dx')} dz={r.get('final_dz')} dθ={r.get('final_dtheta')}",
        ]
        frames = replay(env, blob263["capture"], act, cam, renderer, f"910263|{label}", title, extra)
        dest = VID / f"910263__{label}.mp4"
        path = write_frames_mp4(frames, dest, fps=FPS)
        index.append(
            {
                "hard_id": "hard_910263",
                "variant": label,
                "role": title,
                "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                "SUCCESS_12S": r.get("SUCCESS_12S"),
                "CONTACT_PLACEMENT_VALID": r.get("CONTACT_PLACEMENT_VALID"),
                "RECONFIG_IMPROVED": r.get("RECONFIG_IMPROVED"),
            }
        )
        print(f"wrote {dest.name}", flush=True)

    # --- 910277: C lift then Precision P0 / P1 ---
    # C segment from prior complete-lift mid_body_early; Precision from frozen post-C capture.
    blob277 = pickle.loads((EXIT / "hard_910277.pkl").read_bytes())
    c_act = np.load(PRIOR277 / "hard_910277__mid_body_early__actions.npz")["actions"]
    c_frames = replay(
        env,
        blob277["capture"],
        c_act,
        cam,
        renderer,
        "910277|C",
        "complete_C mid_body_early",
        ["SUCCESS_12S=True (prior)", "freeze → Precision"],
        max_frames=300,
    )
    frozen_cap = pickle.loads(
        (PRIOR277 / "hard_910277__mid_body_early__capture.pkl").read_bytes()
    )
    cmp = {r["variant"]: r for r in csv.DictReader((OUT / "P0_P1_COMPARISON.csv").open(encoding="utf-8"))}
    for label in ("P0", "P1"):
        p_act = np.load(OUT / "raw" / f"910277__{label}__actions.npz")["actions"]
        r = cmp.get(label, {})
        extra = [
            f"NEW={r.get('SUCCESS_NEW')} 12S={r.get('SUCCESS_12S')}",
            f"fail={r.get('PRECISION_FAILURE_REASON')}",
            f"end dx={r.get('final_dx')} dz={r.get('final_dz')} dθ={r.get('final_dtheta')}",
        ]
        p_frames = replay(
            env,
            frozen_cap,
            p_act,
            cam,
            renderer,
            f"910277|{label}",
            f"Precision {label}",
            extra,
            max_frames=200,
        )
        frames = c_frames + p_frames
        dest = VID / f"910277__C_then_{label}.mp4"
        path = write_frames_mp4(frames, dest, fps=FPS)
        index.append(
            {
                "hard_id": "hard_910277",
                "variant": f"C_then_{label}",
                "role": f"C output + Precision {label}",
                "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
                "SUCCESS_12S": r.get("SUCCESS_12S"),
                "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                "PRECISION_FAILURE_REASON": r.get("PRECISION_FAILURE_REASON"),
            }
        )
        print(f"wrote {dest.name}", flush=True)

    (VID / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    md = [
        "# VIDEO_INDEX — P6 C Placement + Precision Continuation",
        "",
        "## 910263 (recatch placement)",
        "",
    ]
    for row in index:
        if row["hard_id"] == "hard_910263":
            md.append(
                f"- `{row['variant']}` ({row['role']}): 12S={row.get('SUCCESS_12S')} "
                f"PLACE={row.get('CONTACT_PLACEMENT_VALID')} IMP={row.get('RECONFIG_IMPROVED')} → `{row.get('video')}`"
            )
    md.extend(["", "## 910277 (C → Precision)", ""])
    for row in index:
        if row["hard_id"] == "hard_910277":
            md.append(
                f"- `{row['variant']}`: NEW={row.get('SUCCESS_NEW')} 12S={row.get('SUCCESS_12S')} "
                f"fail={row.get('PRECISION_FAILURE_REASON')} → `{row.get('video')}`"
            )
    (OUT / "VIDEO_INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
