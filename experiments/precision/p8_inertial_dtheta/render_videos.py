#!/usr/bin/env python
"""P8 videos: ω, α, grip/Fn, θ, relative rotation overlays."""

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
from experiments.precision.p7_oscillatory_dtheta.geometry import (  # noqa: E402
    body_axis_for_correction,
    signed_theta_to_target,
)
from experiments.precision.p8_inertial_dtheta.authority import (  # noqa: E402
    DT,
    hand_omega_along_body,
    rel_omega_along_axis_h,
)
from experiments.precision.render_precision_failure_videos import hud, render_pair  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import sample_state  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/p8_inertial_dtheta"
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
    val = {r["variant"]: r for r in csv.DictReader((OUT / "FINAL_VALIDATION.csv").open(encoding="utf-8"))}
    index = []

    for label in ("I0", "I1", "I2", "I3"):
        act_path = OUT / "raw" / f"{label}__actions.npz"
        if not act_path.is_file():
            continue
        actions = np.load(act_path)["actions"]
        meta = {}
        mp = OUT / "raw" / f"{label}__meta.json"
        if mp.is_file():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        restore_capture(env, frozen)
        body_ax, _ = body_axis_for_correction(env.sim)
        th0 = signed_theta_to_target(env.sim)
        axis_h = np.asarray(th0["axis_h"], float)
        w_prev = hand_omega_along_body(env.sim, body_ax)
        frames = []
        n = max(len(actions), 1)
        stride = max(1, n // 400)
        r = val.get(label, {})
        for i, a in enumerate(actions):
            step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
            w_now = hand_omega_along_body(env.sim, body_ax)
            alpha = (w_now - w_prev) / DT
            w_prev = w_now
            if i % stride and i != len(actions) - 1:
                continue
            t = (i + 1) * DT
            th = signed_theta_to_target(env.sim)
            st = sample_state(env.sim)
            wrel = rel_omega_along_axis_h(env.sim, axis_h)
            # phase from meta series if present
            phase = label
            for s in meta.get("series") or []:
                if abs(float(s.get("t") or 0) - t) < 0.03:
                    phase = f"c{s.get('cycle', '?')}:{s.get('phase', '?')}"
                    break
            lines = hud(env, ic_id=f"910277|{label}", phase=str(phase), t=t, fail=None)
            lines.append(
                f"θ={th['abs_theta_deg']:.2f} signed={th['signed_theta_deg']:.2f} "
                f"w={w_now:.2f} α={alpha:.1f} wrel={wrel:.3f}"
            )
            lines.append(
                f"Fn={float(st.get('fn_tot') or 0):.1f} grip≈a6={float(a[6]):.2f} "
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
    md = ["# VIDEO_INDEX — P8 Inertial Stick-Slip dθ", ""]
    for row in index:
        md.append(
            f"- `{row['variant']}`: θ_final={row.get('theta_final')} NEW={row.get('SUCCESS_NEW')} "
            f"12S={row.get('SUCCESS_12S')} abort={row.get('abort_reason')} → `{row.get('video')}`"
        )
    (OUT / "VIDEO_INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
