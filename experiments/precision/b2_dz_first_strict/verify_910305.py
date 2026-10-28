#!/usr/bin/env python
"""Verify corrected experimental DZ on hard_910305 only (before/after vs prior S1)."""

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
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.b2_dz_first_strict.dz_controller_v2 import run_strict_dz  # noqa: E402
from experiments.precision.render_precision_failure_videos import hud, render_pair  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/dz_wrong_direction_910305"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits/hard_910305.pkl"
PRIOR_ACT = ROOT / "results/diagnostics/b2_dz_first_ordering/raw/hard_910305__S1__actions.npz"
CAM = {"lookat": [0, 0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
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


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def render_actions(env, blob, actions, dest: Path, label: str) -> str | None:
    cam = mjv_camera_from_preset(CAM)
    renderer = mujoco.Renderer(env.sim.model, 420, 560)
    restore_capture(env, blob["capture"])
    frames = []
    stride = max(1, len(actions) // 400) if len(actions) > 400 else 1
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
        if i % stride and i != len(actions) - 1:
            continue
        lines = hud(env, ic_id="hard_910305", phase=label, t=(i + 1) * 0.02, fail=None)
        frames.append(render_pair(env, cam, renderer, lines))
    if not frames:
        return None
    path = write_frames_mp4(frames, dest, fps=FPS)
    return str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    blob = pickle.loads(EXIT.read_bytes())
    blob = {"capture": blob["capture"], "ref": blob["ref"], "sample_id": "hard_910305"}
    pcfg = PrecisionConfig(evaluate_handoff=True, use_verified_lateral_fallback=False)

    restore_capture(env, blob["capture"])
    e0 = measure_errors(env.sim)
    timeline: list = []
    # Bypass |ex| gate for 910305 audit so we can see corrected UP+direction behavior
    result = run_strict_dz(
        env,
        blob,
        pcfg,
        bypass_ex_gate=True,
        do_return_down=True,
        action_budget_left=3000,
        timeline=timeline,
        label="dz_v2_910305",
    )
    acts = list(result.get("actions") or [])
    tl = list(result.get("timeline") or timeline)
    attempts = result.get("attempts")
    np.savez_compressed(OUT / "AFTER_actions.npz", actions=np.asarray(acts, np.float32))

    # Write corrected signed trace
    keys = []
    for row in tl:
        for k in row:
            if k not in keys:
                keys.append(k)
    with (OUT / "SIGNED_MOTION_TRACE_AFTER.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["t"])
        w.writeheader()
        for row in tl:
            w.writerow({k: jsafe(row.get(k)) for k in keys})

    restore_capture(env, result.get("capture") or take_capture(env))
    e1 = measure_errors(env.sim)
    slim = {k: v for k, v in result.items() if k not in ("actions", "timeline", "attempts")}
    verdict = {
        "dz0_mm": float(e0["dz_mm"]),
        "dx0_mm": float(e0["dx_mm"]),
        "dz1_mm": float(result.get("dz1") if result.get("dz1") is not None else e1["dz_mm"]),
        "dx1_mm": float(e1["dx_mm"]),
        "abs_dz0": abs(float(e0["dz_mm"])),
        "abs_dz1": abs(float(result.get("dz1") if result.get("dz1") is not None else e1["dz_mm"])),
        "improved_abs_after_return": bool(result.get("improved_abs_after_return")),
        "bilateral_final": bool(e1.get("bilateral")),
        "stop_reason": result.get("stop_reason"),
        "n_attempts": result.get("n_attempts"),
        "blocked": result.get("blocked"),
        "block_reason": result.get("block_reason"),
        "axial_correction_occurred": result.get("axial_correction_occurred"),
        "ok_controller": result.get("ok"),
        "n_actions": len(acts),
        "success_criterion": (
            "abs_dz_final < abs_dz0 after brake/regrip+return AND bilateral AND not contact-loss"
        ),
        "meets_abs_improvement": bool(result.get("improved_abs_after_return")) and bool(e1.get("bilateral")),
        "result": jsafe(slim),
        "attempts": jsafe(attempts),
    }
    (OUT / "AFTER_VERIFY.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Videos: before = prior S1 if present; after = corrected controller
    after_vid = render_actions(env, blob, acts, OUT / "AFTER_hard_910305_dz_v2.mp4", "AFTER_v2")
    before_vid = None
    if PRIOR_ACT.is_file():
        before_acts = np.load(PRIOR_ACT)["actions"]
        before_vid = render_actions(env, blob, before_acts, OUT / "BEFORE_hard_910305_S1.mp4", "BEFORE_S1")

    (OUT / "VIDEO_INDEX.md").write_text(
        "\n".join(
            [
                "# Videos — hard_910305 DZ direction fix",
                "",
                f"- BEFORE (prior S1): `{before_vid}`",
                f"- AFTER (experimental DZ v2): `{after_vid}`",
                "",
                f"abs DZ {verdict['abs_dz0']:.2f} → {verdict['abs_dz1']:.2f}; "
                f"improved={verdict['meets_abs_improvement']}; stop={verdict['stop_reason']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
